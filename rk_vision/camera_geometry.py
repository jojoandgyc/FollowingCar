"""Image displacement for the runtime's right-positive encoder yaw.

``integrated_yaw_deg`` is populated from ``integrated_yaw_right_deg``. A
positive chassis turn therefore moves a stationary person's image left.
The camera convention is fixed; choosing either sign per candidate would
let motion in the wrong direction explain an identity jump.
"""

import math


def yaw_image_shift_ratio(previous_yaw, current_yaw, camera_hfov_deg):
    """Return the predicted image-width shift (right is positive)."""
    previous, current, hfov = map(float, (previous_yaw, current_yaw, camera_hfov_deg))
    if not all(math.isfinite(value) for value in (previous, current, hfov)) or hfov <= 0:
        return math.inf  # Invalid known geometry must never prove continuity.
    return -(current - previous) / max(1.0, hfov)


def horizontal_center_displacement(
    *, current_center, previous_center, current_yaw, previous_yaw, camera_hfov_deg,
):
    """Return raw and physically signed yaw-compensated absolute motion.

    Unknown yaw retains the existing raw-position fallback. With measured
    yaw, the residual is authoritative; a smaller raw displacement cannot
    erase a contradiction in the measured camera motion.
    """
    delta = float(current_center) - float(previous_center)
    raw_jump = abs(delta)
    if current_yaw is None or previous_yaw is None:
        return raw_jump, raw_jump
    shift = yaw_image_shift_ratio(previous_yaw, current_yaw, camera_hfov_deg)
    return raw_jump, abs(delta - shift)
