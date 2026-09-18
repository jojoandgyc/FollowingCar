"""A longer detector observation window is NOT a longer motor lease."""
import math


def roi_age_status(capture_stamp, now, max_age, feedback=None):
    """Keep the original 180ms path; extra time requires fresh low-yaw feedback.

    Identity/geometry/foreground checks remain the caller's responsibility.
    This function neither moves a bbox nor authorizes a motor command.
    """
    values = (capture_stamp, now, max_age)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in values):
        return "invalid"
    age = now - capture_stamp
    if capture_stamp <= 0 or age < 0 or age > min(.25, max_age):
        return "expired"
    if age <= .18:
        return "normal"
    if feedback is None or not getattr(feedback, "trustworthy", False):
        return "feedback_missing"
    stamp = getattr(feedback, "timestamp", None)
    yaw = getattr(feedback, "yaw_rate_right_dps", None)
    raw = getattr(feedback, "raw_yaw_rate_right_dps", None)
    raw = yaw if raw is None else raw
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in (stamp, yaw, raw)):
        return "feedback_invalid"
    if not 0 <= now-stamp <= .15:
        return "feedback_stale"
    if max(abs(yaw), abs(raw)) > 5.0:
        return "turning"
    return "extended"


def roi_age_allowed(capture_stamp, now, max_age, feedback=None):
    return roi_age_status(capture_stamp, now, max_age, feedback) in {"normal", "extended"}
