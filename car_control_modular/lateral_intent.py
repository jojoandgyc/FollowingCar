from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from typing import Optional
from .outward_trajectory import OutwardTrajectoryLead


@dataclass(frozen=True)
class LateralControlIntent:
    """Immutable vision-to-motion contract for visible-target yaw control."""

    sequence: int
    target_id: int
    frame_index: int
    published_at: float
    valid_until: float
    x_ratio: float
    motion_dx_ratio: float
    target_image_rate_dps: Optional[float]
    mode: str
    base_percent: int
    base_rpm: int
    initial_correction_rpm: int
    correction_limit_rpm: float  # Persistent policy ceiling; PID recomputes transient caps.
    confidence: float
    bbox_quality: str
    reason: str
    capture_frame_id: int = 0
    capture_timestamp: float = 0.0
    # Capture slot at which this intent was computed.  capture_frame_id may
    # intentionally point to an older evidence frame during reacquisition.
    decision_capture_frame_id: int = 0
    # Zero is a command, not the absence of an owner. A new visual sample is
    # required to release a deliberate coast/center hold.
    hold_zero: bool = False
    near_distance_mode: bool = False
    # Predictive/center stop evidence. Runtime qualifies whole-chassis NORMAL
    # from near-mode/no-translation or measured low-speed pivot motion; normal
    # differential forward travel still retains longitudinal authority.
    park_requested: bool = False
    nominal_valid_until: float = 0.0
    image_error_only: bool = False
    # Explicit permission from the visual producer. Tapered/braking/search
    # evidence must never be amplified again by the motor executor.
    response_boost_allowed: bool = False
    visual_error_deg: Optional[float] = None
    countersteer_rpm: int = 0  # certified predictive brake, not a search direction
    outward_lead: Optional[OutwardTrajectoryLead] = None
    braking_image_rate_dps: Optional[float] = None  # capture-bound, taper only
    outward_continuity_rate_dps: Optional[float] = None  # limited yaw only, no projection
    forward_countersteer: bool = False  # cannot become an old pure-pivot pulse

    def continuation_allowed(self, now: float, feedback) -> bool:
        """Short forward-only bridge; feedback cannot extend either deadline."""
        if self.nominal_valid_until <= 0.0 or now <= self.nominal_valid_until:
            return True
        return _forward_continuation_evidence(self, now, feedback)

    def age_sec(self, now: float) -> float:
        return max(0.0, float(now) - float(self.published_at))

    def valid(self, now: float) -> bool:
        return float(now) <= float(self.valid_until)

    def projected_x_ratio(
        self,
        now: float,
        *,
        camera_hfov_deg: float,
        max_projection_sec: float,
        max_projection_ratio: float,
    ) -> tuple[float, float]:
        """Project image position to the control tick using measured image rate."""
        rate_dps = self.target_image_rate_dps
        if rate_dps is None or not math.isfinite(float(rate_dps)):
            return max(0.0, min(1.0, float(self.x_ratio))), 0.0
        horizon = min(
            max(0.0, float(max_projection_sec)),
            self.age_sec(now),
        )
        hfov = max(1.0, float(camera_hfov_deg))
        projected_delta = float(rate_dps) * horizon / hfov
        ratio_limit = max(0.0, float(max_projection_ratio))
        projected_delta = max(-ratio_limit, min(ratio_limit, projected_delta))
        return (
            max(0.0, min(1.0, float(self.x_ratio) + projected_delta)),
            horizon,
        )


class LateralIntentStore:
    """Thread-safe latest-value store; consumers never observe partial fields."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._intent: Optional[LateralControlIntent] = None

    def publish(self, intent: LateralControlIntent) -> LateralControlIntent:
        with self._lock:
            self._sequence += 1
            published = replace(intent, sequence=self._sequence)
            self._intent = published
            return published

    def snapshot(self) -> Optional[LateralControlIntent]:
        with self._lock:
            return self._intent

    def clear(self) -> Optional[LateralControlIntent]:
        with self._lock:
            previous = self._intent
            self._intent = None
            return previous


def _forward_continuation_evidence(intent, now, feedback) -> bool:
    if (intent.mode != "forward" or intent.bbox_quality != "reliable"
            or intent.hold_zero or intent.park_requested or intent.near_distance_mode
            or intent.base_rpm <= 0 or intent.capture_timestamp <= 0.0
            or not 0.0 <= now - intent.capture_timestamp <= 0.35
            or not intent.valid(now) or feedback is None or not feedback.trustworthy):
        return False
    age = now - float(feedback.timestamp)
    if intent.image_error_only:
        # Same deadlines/identity/forward scope, but yaw must not veto the
        # position-only trial. Reserve image-motion uncertainty near center.
        rate = intent.target_image_rate_dps
        if not 0.0 <= age <= .10 or (rate is not None and not math.isfinite(float(rate))):
            return False
        travel = max(15.0, abs(rate or 0.0)) * max(0.0, now - intent.published_at) / 60.0
        return abs(intent.x_ratio - .5) - travel > .10
    yaw = float(feedback.yaw_rate_right_dps)
    raw = getattr(feedback, "raw_yaw_rate_right_dps", None)
    if (not math.isfinite(yaw) or not 0.0 <= age <= 0.10 or abs(yaw) > 20.0
            or (raw is not None and (not math.isfinite(float(raw)) or abs(float(raw) - yaw) > 10.0))):
        return False
    # Stay well outside the center hold band even under a conservative image
    # motion bound. Do not bridge an inward target about to cross the center.
    rate = intent.target_image_rate_dps
    if rate is not None and not math.isfinite(float(rate)):
        return False
    travel = max(abs(yaw), abs(rate or 0.0)) * max(0.0, now - intent.published_at) / 60.0
    return abs(intent.x_ratio - 0.5) - travel > 0.10


def with_forward_continuation(intent: LateralControlIntent, feedback) -> LateralControlIntent:
    """Reserve at most 220ms after publication / 350ms after capture.

    Admission and each fast tick require fresh cached encoder evidence. Neither
    a duplicate image nor encoder refresh is allowed to move these deadlines.
    This is lateral authority only; it cannot renew longitudinal authorization.
    """
    deadline = min(intent.published_at + 0.22, intent.capture_timestamp + 0.35)
    if deadline <= intent.valid_until or not _forward_continuation_evidence(intent, intent.published_at, feedback):
        return intent
    return replace(intent, nominal_valid_until=intent.valid_until, valid_until=deadline)


def slew_signed_rpm(
    previous_rpm: int,
    requested_rpm: int,
    dt_sec: float,
    *,
    rise_rpm_per_sec: float,
    brake_rpm_per_sec: float,
) -> int:
    """Bound yaw output changes while allowing faster braking and reversal."""
    previous = float(previous_rpm)
    requested = float(requested_rpm)
    dt = max(0.0, float(dt_sec))
    braking = bool(
        abs(requested) < abs(previous)
        or (previous * requested < 0.0)
    )
    rate = (
        max(0.0, float(brake_rpm_per_sec))
        if braking
        else max(0.0, float(rise_rpm_per_sec))
    )
    max_delta = rate * dt
    if max_delta <= 0.0:
        return int(round(requested))
    delta = max(-max_delta, min(max_delta, requested - previous))
    output = previous + delta
    if previous * requested < 0.0 and previous * output < 0.0:
        # A reversal must pass through zero; do not jump across it in one tick.
        output = 0.0
    rounded = int(round(output))
    if (
        rounded == 0
        and previous == 0.0
        and requested != 0.0
        and output * requested > 0.0
    ):
        # The motor is confirmed usable at 1 RPM. Preserve the first non-zero
        # command instead of letting float-to-int rounding insert an extra
        # STOP tick before a small correction starts.
        return 1 if requested > 0.0 else -1
    return rounded
