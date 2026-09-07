from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from typing import Optional


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
    correction_limit_rpm: float
    confidence: float
    bbox_quality: str
    reason: str
    capture_frame_id: int = 0
    capture_timestamp: float = 0.0
    # Capture slot at which this intent was computed.  capture_frame_id may
    # intentionally point to an older evidence frame during reacquisition.
    decision_capture_frame_id: int = 0

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
