"""Bounded lost-person search policy for the independent minimal runtime.

The policy is intentionally pure: it performs no camera, sensor, clock-thread,
or motor I/O.  The runtime supplies a monotonic timestamp and dispatches the
returned command through the existing LZ30EMA adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .commands import MinimalFollowCommand


@dataclass(frozen=True)
class LostPersonSearchConfig:
    enabled: bool = True
    lost_confirm_frames: int = 1
    turn_memory_sec: float = 1.0
    target_direction_memory_sec: float = 3.0
    target_side_deadband_ratio: float = 0.03
    fallback_direction: str = "left"
    directed_search_sec: float = 2.0
    sweep_half_cycle_sec: float = 3.0
    sweep_cycles_before_spin: int = 2
    timeout_sec: float = 1.5
    turn_percent: int = 8


@dataclass(frozen=True)
class LostPersonSearchStatus:
    state: str
    direction: Optional[str]
    lost_frames: int
    search_elapsed_ms: Optional[float]


class LostPersonSearchPolicy:
    """Search from steering history, target-side history, then a fallback.

    A newly lost target first produces one STOP frame.  This gives the motor a
    bounded zero/STOP transition before one wheel reverses for an in-place
    search.  At the normal 20 Hz loop this costs at most one frame only during
    a loss event; visible-follow latency is unchanged.
    """

    def __init__(self, config: LostPersonSearchConfig) -> None:
        self.config = config
        self._last_direction: Optional[str] = None
        self._last_direction_at: Optional[float] = None
        self._last_target_direction: Optional[str] = None
        self._last_target_direction_at: Optional[float] = None
        self._lost_frames = 0
        self._search_started_at: Optional[float] = None
        self._search_direction: Optional[str] = None
        self._state = "tracking"

    @staticmethod
    def _valid_now(now: float) -> bool:
        return math.isfinite(float(now)) and float(now) >= 0.0

    def record_executed_follow_command(self, command: MinimalFollowCommand, now: float) -> None:
        """Remember only steering which has passed runtime dispatch."""
        if not self._valid_now(now) or not command.moving:
            return
        if command.reason == "steer_left":
            self._last_direction = "left"
            self._last_direction_at = float(now)
        elif command.reason == "steer_right":
            self._last_direction = "right"
            self._last_direction_at = float(now)

    def record_visible_target(self, bbox, frame_width: int, now: float) -> None:
        """Keep a visual direction even when the driving command was straight."""
        if not self._valid_now(now) or frame_width <= 0:
            return
        x1, _, x2, _ = (float(value) for value in bbox)
        offset_ratio = ((x1 + x2) * 0.5 - float(frame_width) * 0.5) / float(frame_width)
        deadband = max(0.0, float(self.config.target_side_deadband_ratio))
        if offset_ratio <= -deadband:
            self._last_target_direction = "left"
            self._last_target_direction_at = float(now)
        elif offset_ratio >= deadband:
            self._last_target_direction = "right"
            self._last_target_direction_at = float(now)

    @property
    def loss_episode_active(self) -> bool:
        """Whether a candidate must be appearance-verified before takeover."""
        return self._state != "tracking"

    def _fresh_direction(self, now: float) -> str:
        if self._last_direction in {"left", "right"} and self._last_direction_at is not None:
            if now - self._last_direction_at <= max(0.0, float(self.config.turn_memory_sec)):
                return self._last_direction
        if self._last_target_direction in {"left", "right"} and self._last_target_direction_at is not None:
            if now - self._last_target_direction_at <= max(0.0, float(self.config.target_direction_memory_sec)):
                return self._last_target_direction
        return "right" if str(self.config.fallback_direction).strip().lower() == "right" else "left"

    def _status(self, now: float, direction: Optional[str]) -> LostPersonSearchStatus:
        elapsed_ms = None
        if self._search_started_at is not None:
            elapsed_ms = max(0.0, now - self._search_started_at) * 1000.0
        return LostPersonSearchStatus(self._state, direction, self._lost_frames, elapsed_ms)

    @staticmethod
    def _opposite(direction: str) -> str:
        return "right" if direction == "left" else "left"

    def _search_direction_for_elapsed(self, now: float, initial_direction: str) -> tuple[str, str]:
        """First look where the target left, then repeatedly scan both sides."""
        if self._search_started_at is None:
            return initial_direction, "directed"
        elapsed = max(0.0, now - self._search_started_at)
        directed = max(0.0, float(self.config.directed_search_sec))
        if elapsed < directed:
            return initial_direction, "directed"
        half_cycle = max(0.1, float(self.config.sweep_half_cycle_sec))
        phase = int((elapsed - directed) / half_cycle)
        # A cycle is one left plus one right scan. Once these bounded sweeps
        # have covered both sides, keep turning in the initial direction to
        # complete a wider in-place scan instead of oscillating forever.
        sweep_half_cycles = max(0, int(self.config.sweep_cycles_before_spin)) * 2
        if phase >= sweep_half_cycles:
            return initial_direction, "continuous_spin"
        direction = self._opposite(initial_direction) if phase % 2 == 0 else initial_direction
        return direction, f"sweep_{phase + 1}"

    def visible(self, now: float) -> LostPersonSearchStatus:
        if not self._valid_now(now):
            raise ValueError("now must be a finite monotonic timestamp")
        # Search uses one reversed wheel. Stop once before normal forward or
        # differential-follow commands can reverse that wheel again.
        was_rotating = self._state.startswith("searching")
        self._lost_frames = 0
        self._search_started_at = None
        self._search_direction = None
        self._state = "search_reacquire_transition_stop" if was_rotating else "tracking"
        return self._status(float(now), self._fresh_direction(float(now)))

    def target_missing(
        self, *, now: float, front_obstacle: bool,
    ) -> tuple[MinimalFollowCommand, LostPersonSearchStatus]:
        if not self._valid_now(now):
            raise ValueError("now must be a finite monotonic timestamp")
        now = float(now)
        initial_direction = self._search_direction or self._fresh_direction(now)
        self._lost_frames += 1

        if front_obstacle:
            self._search_started_at = None
            self._search_direction = None
            self._state = "front_ir"
            return MinimalFollowCommand.stop("front_ir"), self._status(now, initial_direction)
        if not self.config.enabled:
            self._search_started_at = None
            self._search_direction = None
            self._state = "disabled"
            return MinimalFollowCommand.stop("person_missing"), self._status(now, initial_direction)
        if self._lost_frames < max(1, int(self.config.lost_confirm_frames)):
            self._search_started_at = None
            self._search_direction = None
            self._state = "lost_confirming"
            return MinimalFollowCommand.stop("person_missing_confirming"), self._status(now, initial_direction)
        if self._search_started_at is None:
            self._search_started_at = now
            self._search_direction = initial_direction
            self._state = "search_transition_stop"
            return MinimalFollowCommand.stop("search_transition_stop"), self._status(now, initial_direction)
        if now - self._search_started_at + 1e-9 >= max(0.0, float(self.config.timeout_sec)):
            self._state = "search_timeout"
            return MinimalFollowCommand.stop("search_timeout"), self._status(now, self._search_direction)

        direction, phase = self._search_direction_for_elapsed(now, self._search_direction or initial_direction)
        self._state = "searching_" + phase
        if direction == "left":
            command = MinimalFollowCommand.rotate_left(self.config.turn_percent)
        else:
            command = MinimalFollowCommand.rotate_right(self.config.turn_percent)
        return command, self._status(now, direction)
