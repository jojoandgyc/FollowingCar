"""Pure template-update quarantine after a cross-track identity reacquisition.

The owner arms this gate before writing any strong, weak, or partial template.
It must keep that UID's old gallery frozen while held, and pass distance to the
frozen *strong* gallery, not a weighted/partial match or an updated template.
This gate neither assigns UIDs nor restricts control observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Optional


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _identifier(value: Any) -> Optional[int]:
    number = _finite(value)
    return int(number) if number is not None and number.is_integer() and number > 0 else None


@dataclass(frozen=True)
class QuarantineDecision:
    hold: bool
    reason: str
    streak: int
    elapsed_sec: Optional[float] = None


@dataclass
class _HeldIdentity:
    track_id: Optional[int]
    armed_timestamp: Optional[float]
    last_capture_id: Optional[int]
    last_capture_timestamp: Optional[float]
    last_frame_index: Optional[int]
    streak: int = 0
    stable_timestamp: Optional[float] = None
    center: Optional[float] = None
    area: Optional[float] = None

    def clear_streak(self) -> None:
        self.streak = 0
        self.stable_timestamp = None
        self.center = None
        self.area = None


class ReacquireQuarantine:
    """Require elapsed RGB time AND consecutive stable evidence to unfreeze.

    ``reset`` clears proof but deliberately preserves isolation. ``prune`` is
    reserved for explicitly removed identities; there is no time-based expiry.
    The initial arm capture never contributes to the post-reacquisition streak.
    """

    def __init__(
        self,
        *,
        min_duration_sec: float = 1.0,
        required_frames: int = 3,
        max_gap_sec: float = 0.35,
        max_center_jump_ratio: float = 0.20,
        min_area_similarity: float = 0.50,
        max_strong_distance: float = 0.20,
    ) -> None:
        self.min_duration_sec = float(min_duration_sec)
        self.required_frames = max(3, int(required_frames))
        self.max_gap_sec = float(max_gap_sec)
        self.max_center_jump_ratio = float(max_center_jump_ratio)
        self.min_area_similarity = float(min_area_similarity)
        self.max_strong_distance = float(max_strong_distance)
        self._held: Dict[int, _HeldIdentity] = {}

    def arm(
        self,
        uid: Any,
        track_id: Any,
        capture_frame_id: Any = None,
        capture_timestamp: Any = None,
        frame_index: Any = None,
    ) -> QuarantineDecision:
        identity = _identifier(uid)
        if identity is None:
            return QuarantineDecision(True, "invalid_uid", 0)
        track = _identifier(track_id)
        timestamp = _finite(capture_timestamp)
        if timestamp is not None and timestamp <= 0.0:
            timestamp = None
        # A reused raw track must not carry a previous identity's stable proof.
        for other_uid, state in self._held.items():
            if other_uid != identity and track is not None and state.track_id == track:
                state.clear_streak()
        self._held[identity] = _HeldIdentity(
            track, timestamp, _identifier(capture_frame_id), timestamp,
            _identifier(frame_index),
        )
        return QuarantineDecision(True, "armed", 0, 0.0 if timestamp is not None else None)

    def is_held(self, uid: Any) -> bool:
        identity = _identifier(uid)
        return identity is None or identity in self._held

    def reset(self, uid: Any = None) -> None:
        """Discard stable proof, never unfreeze an existing identity."""
        if uid is None:
            for state in self._held.values():
                state.clear_streak()
        else:
            state = self._held.get(_identifier(uid))
            if state is not None:
                state.clear_streak()

    def prune(self, live_uids: Iterable[Any]) -> None:
        """Remove only identities explicitly absent from the owner's gallery."""
        keep = {_identifier(uid) for uid in live_uids}
        self._held = {uid: state for uid, state in self._held.items() if uid in keep}

    def observe(
        self,
        *,
        uid: Any,
        track_id: Any,
        capture_frame_id: Any = None,
        capture_timestamp: Any = None,
        frame_index: Any = None,
        is_fresh: bool = False,
        quality_ok: bool = False,
        quality_tier: Optional[str] = None,
        match_source: Optional[str] = None,
        feature_available: bool = False,
        strong_distance: Any = None,
        center_x_ratio: Any = None,
        area_ratio: Any = None,
    ) -> QuarantineDecision:
        identity, track = _identifier(uid), _identifier(track_id)
        if identity is None:
            # Missing/changed identity on the held track cannot preserve proof.
            for state in self._held.values():
                if track is None or state.track_id == track:
                    state.clear_streak()
            return QuarantineDecision(True, "invalid_uid", 0)
        for other_uid, other_state in self._held.items():
            if other_uid != identity and track is not None and other_state.track_id == track:
                other_state.clear_streak()
        state = self._held.get(identity)
        if state is None:
            return QuarantineDecision(False, "not_armed", 0)

        timestamp = _finite(capture_timestamp)
        capture_id = _identifier(capture_frame_id)
        frame = _identifier(frame_index)
        elapsed = (
            timestamp - state.armed_timestamp
            if timestamp is not None and state.armed_timestamp is not None
            else None
        )

        def reject(reason: str) -> QuarantineDecision:
            state.clear_streak()
            return QuarantineDecision(True, reason, 0, elapsed)

        if track is None:
            return reject("invalid_track")
        if track != state.track_id:
            self.arm(identity, track, capture_id, timestamp, frame)
            return QuarantineDecision(True, "track_changed", 0, 0.0 if timestamp is not None else None)
        if timestamp is None or timestamp <= 0.0 or capture_id is None:
            return reject("missing_capture")
        duplicate = (
            capture_id == state.last_capture_id
            and timestamp == state.last_capture_timestamp
        )
        if not duplicate and (
            state.last_capture_id is not None and capture_id <= state.last_capture_id
            or state.last_capture_timestamp is not None and timestamp <= state.last_capture_timestamp
        ):
            return reject("out_of_order_capture")
        if not duplicate:
            # Invalid-quality samples also consume their RGB sequence number;
            # revisiting that same capture cannot manufacture fresh evidence.
            state.last_capture_id, state.last_capture_timestamp, state.last_frame_index = capture_id, timestamp, frame
        if is_fresh is not True:
            return reject("stale_observation")
        if feature_available is not True:
            return reject("missing_feature")
        if quality_ok is not True or quality_tier != "strong" or match_source != "strong":
            return reject("not_high_quality_strong")
        distance = _finite(strong_distance)
        if distance is None or distance < 0.0 or distance > self.max_strong_distance:
            return reject("frozen_strong_distance")
        center, area = _finite(center_x_ratio), _finite(area_ratio)
        if center is None or not 0.0 <= center <= 1.0 or area is None or not 0.0 < area <= 1.0:
            return reject("invalid_geometry")

        # RGB sequence and timestamp, not control ticks, define a new sample.
        if duplicate:
            return QuarantineDecision(True, "duplicate_capture", state.streak, elapsed)
        if state.armed_timestamp is None:
            # A missing arm timestamp never becomes an implicit elapsed timer.
            state.armed_timestamp = timestamp
            elapsed = 0.0
        if state.stable_timestamp is not None:
            if timestamp - state.stable_timestamp > self.max_gap_sec + 1e-9:
                return reject("capture_gap")
            if (
                abs(center - state.center) > self.max_center_jump_ratio + 1e-9
                or min(area, state.area) / max(area, state.area) < self.min_area_similarity - 1e-9
            ):
                return reject("geometry_discontinuity")
        state.streak += 1
        state.stable_timestamp, state.center, state.area = timestamp, center, area
        if state.streak >= self.required_frames and elapsed >= self.min_duration_sec - 1e-9:
            self._held.pop(identity)
            return QuarantineDecision(False, "released", state.streak, elapsed)
        return QuarantineDecision(
            True, "minimum_duration" if state.streak >= self.required_frames else "confirming",
            state.streak, elapsed,
        )
