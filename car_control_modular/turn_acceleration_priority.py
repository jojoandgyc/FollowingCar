"""Historical offline policy; the production wheel writer no longer calls it.

Limit common acceleration while an authorized wheel differential builds.

This never grants movement, increases either wheel target, adds braking, or
permits reversal. The caller still applies identity, depth and wheel guards.
"""
import math


def limit_common_acceleration(requested, feedback, now):
    requested = tuple(requested)
    if feedback is None or not feedback.trustworthy:
        return requested, "feedback_unavailable"
    measured = (feedback.left_forward_rpm, feedback.right_forward_rpm)
    if (not all(math.isfinite(v) for v in (*requested, *measured, feedback.timestamp, now))
            or not 0 <= now-feedback.timestamp <= .15):
        return requested, "feedback_invalid"
    if min(requested) <= 0 or min(measured) < 0:
        return requested, "not_forward_acceleration"
    diff = requested[0]-requested[1]
    if abs(diff) < 10:
        return requested, "small_correction"
    outer, inner = (0, 1) if diff > 0 else (1, 0)
    if measured[outer]-measured[inner] >= abs(diff)-4:
        return requested, "differential_established"
    # Do not change braking/approach commands. A falling longitudinal request
    # must remain free to slow either wheel immediately.
    if any(requested[i] < measured[i] for i in range(2)):
        return requested, "deceleration_requested"
    inner_cap = math.floor(measured[inner]+2)
    reduction = max(0, requested[inner]-inner_cap)
    candidate = tuple(v-reduction for v in requested)
    if not reduction or any(candidate[i] < measured[i] for i in range(2)):
        return requested, "no_acceleration_to_trim"
    return candidate, "turn_build_common_acceleration_limited"
