"""Measured-position yaw taper, independent of longitudinal authorization.

The camera and encoder convention is right-positive chassis yaw. Turning right
moves a stationary subject left in the image. Feedback can only remove some of
the measured visual error; it cannot invent an opposite-side target or increase
the original request. No target velocity is estimated here.
"""
from dataclasses import dataclass
import math
from typing import Optional


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class ShortFollowYawObservation:
    uid: int
    capture_id: int
    capture_timestamp: float
    center_x_ratio: float
    capture_yaw_deg: Optional[float] = None


def tapered_center(observation, *, current_yaw_deg=None, yaw_rate_deg_s=None,
                   camera_hfov_deg=60., damping_sec=.10):
    """Return a same-side center with measured yaw and a small rate taper.

    ``capture_yaw_deg`` must be capture-aligned, and the caller qualifies the
    freshness and reliability of current feedback. Missing feedback is not a
    stop condition. The 100 ms rate horizon is a tuning starting point, not a
    measured braking guarantee; it never grants a new direction or more RPM.
    """
    error = observation.center_x_ratio - .5
    sign = 1. if error >= 0 else -1.
    remaining = abs(error)
    reason = "none"
    if (finite(observation.capture_yaw_deg) and finite(current_yaw_deg)
            and finite(camera_hfov_deg) and camera_hfov_deg > 0):
        # Integrated heading is unwrapped. Never wrap a discontinuity into
        # seemingly useful movement; a >45 degree change within this short
        # observation lifetime is not usable taper evidence.
        change = current_yaw_deg - observation.capture_yaw_deg
        if abs(change) <= 45.:
            completed = max(0., sign * change / camera_hfov_deg)
            if completed > 0:
                remaining = max(0., remaining - completed)
                reason = "measured_yaw_taper"
    if (finite(yaw_rate_deg_s) and abs(yaw_rate_deg_s) <= 180.
            and finite(camera_hfov_deg) and camera_hfov_deg > 0
            and finite(damping_sec) and 0 <= damping_sec <= .20):
        lookahead = max(0., sign * yaw_rate_deg_s) * damping_sec / camera_hfov_deg
        if lookahead > 0:
            remaining = max(0., remaining - lookahead)
            reason = "yaw_rate_taper"
    return .5 + sign * remaining, reason
