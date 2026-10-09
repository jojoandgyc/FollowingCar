"""Reduce an existing lease using a stationary-target braking budget.

Never a depth observation, first-start permission, or lease renewal. The original
speed bound reserves travel since capture. Fresh measured speed must independently
fit the remaining budget; a low requested speed cannot hide excessive momentum.
Body speed owns the longitudinal no-acceleration check. The faster wheel still
owns the conservative braking check; differential steering cannot cancel it.
"""
import math


def continuation_feedback_speeds(*, feedback, now, circumference, max_rpm,
                                 original_speed_bound):
    """Validate one cached encoder sample for BOTH continuation paths.

    Return body and outer-wheel speeds in m/s, or an explicit veto. Neither
    averaging nor a fresh timestamp can hide a reversing/overspeeding wheel.
    """
    config = (now, circumference, max_rpm, original_speed_bound)
    if (not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                and math.isfinite(v) for v in config)
            or min(circumference, max_rpm) <= 0 or original_speed_bound < 0):
        return None, None, "invalid_braking_model"
    if feedback is None or not getattr(feedback, "trustworthy", False):
        return None, None, "continuation_feedback_missing"
    values = tuple(getattr(feedback, key, None) for key in (
        "timestamp", "left_forward_rpm", "right_forward_rpm"))
    if (not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                and math.isfinite(v) for v in values)
            or not 0 <= now-values[0] <= .10
            or min(values[1:]) < 0 or max(values[1:]) > max_rpm):
        return None, None, "continuation_feedback_invalid"
    body = .5*(values[1]+values[2])*circumference/60.
    outer = max(values[1:])*circumference/60.
    if body > original_speed_bound + 1e-9:
        return None, None, "continuation_speed_exceeds_bound"
    return body, outer, "continuation_feedback_valid"


def continuation_speed_cap(*, distance, stop_distance, original_speed_bound,
                           sample_age, feedback, now, circumference, max_rpm,
                           deceleration, response_delay):
    vals = (distance, stop_distance, original_speed_bound, sample_age, now,
            circumference, max_rpm, deceleration, response_delay)
    if (not all(isinstance(v, (float, int)) and math.isfinite(v) for v in vals)
            or min(distance, circumference, max_rpm, deceleration) <= 0
            or min(original_speed_bound, sample_age, response_delay) < 0
            or sample_age > .25):
        return 0., "invalid_braking_model"
    _body, measured, reason = continuation_feedback_speeds(
        feedback=feedback, now=now, circumference=circumference, max_rpm=max_rpm,
        original_speed_bound=original_speed_bound)
    if measured is None:
        return 0., reason
    # No target-motion credit: reserve original upper-bound travel since Depth.
    margin = distance-stop_distance-original_speed_bound*sample_age-.02
    # Reserve the entire allowed encoder age, not its fluctuating actual age.
    # A newer encoder sample must NOT increase this old grant's speed cap.
    margin -= original_speed_bound*.10
    if margin <= 0 or measured*response_delay + measured**2/(2*deceleration) > margin:
        return 0., "braking_margin"
    speed = deceleration*(math.sqrt(response_delay**2+2*margin/deceleration)-response_delay)
    return min(speed, original_speed_bound)*60./circumference, "same_grant_reduced_braking_cap"
