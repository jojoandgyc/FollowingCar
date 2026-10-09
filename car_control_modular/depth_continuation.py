"""Reduce an existing lease using a captured braking budget.

Never a depth observation, first-start permission, or lease renewal. The original
speed bound reserves travel since capture. Fresh measured speed must independently
fit the remaining budget; a low requested speed cannot hide excessive momentum.
Body speed owns the longitudinal no-acceleration check. The faster wheel still
owns the conservative braking check; differential steering cannot cancel it.
"""
import math
from dataclasses import dataclass

from .longitudinal_approach import RawDepthMotionEvidence
from .depth_authority_timing import MAX_FORWARD_DEPTH_TTL_SEC


# Match the fresh PI encoder budget only for the relative-motion policy. The
# entire allowance is reserved in its distance budget, independent of the
# fluctuating age of a particular feedback sample.
RELATIVE_CONTINUATION_FEEDBACK_MAX_AGE_SEC = .15
RELATIVE_CONTINUATION_REVERSE_TAIL_RPM = 3.


@dataclass(frozen=True)
class ContinuationMotionEvidence:
    """Frozen same-window motion belonging to exactly one UID/Depth grant.

    This is not a measurement, target-speed feedforward, or renewed lease.
    The publisher must additionally verify identity and the original window's
    use by the fresh controller before copying it here.
    """
    uid: int
    sample_timestamp: float
    target_speed_bound_m_s: float
    range_rate_m_s: float
    span_sec: float
    sample_count: int

    def valid_for(self, uid, stamp):
        return bool(isinstance(uid, int) and not isinstance(uid, bool) and uid > 0
                    and isinstance(self.uid, int) and not isinstance(self.uid, bool)
                    and self.uid == uid
                    and isinstance(stamp, (float, int)) and not isinstance(stamp, bool)
                    and math.isfinite(stamp) and stamp > 0
                    and RawDepthMotionEvidence(
                        self.sample_timestamp, self.range_rate_m_s,
                        self.target_speed_bound_m_s, self.span_sec,
                        self.sample_count).valid_for(stamp, self.range_rate_m_s))


@dataclass(frozen=True)
class RelativeBrakingBudget:
    """One stopping-distance calculation shared by admission and diagnostics.

    ``margin_m`` is distance left after the original grant's travel and the
    fixed feedback-age reserve. ``required_stop_m`` is the currently measured
    outer wheel's relative stopping distance. A zero cap is an immediate
    momentum veto, not an invitation to extend the depth grant.
    """
    margin_m: float
    required_stop_m: float
    target_credit_m_s: float
    max_allowed_rpm: float


def relative_braking_budget(*, distance_m, stop_distance_m, speed_bound_m_s,
                            age_sec, outer_speed_m_s, target_speed_m_s,
                            deceleration_m_s2, response_delay_sec,
                            wheel_circumference_m, max_rpm,
                            same_grant_no_acceleration=True):
    """Compute the existing same-grant relative stop budget, without state.

    The fixed feedback allowance makes this budget nonincreasing as one
    grant ages even when a newer encoder report arrives. The default also
    applies the old grant's no-acceleration limit. Neither option authorizes
    motion: the caller still owns UID, sensor validity and deadline.
    """
    values = (distance_m, stop_distance_m, speed_bound_m_s, age_sec,
              outer_speed_m_s, target_speed_m_s, deceleration_m_s2,
              response_delay_sec, wheel_circumference_m, max_rpm)
    if (type(same_grant_no_acceleration) is not bool
            or not all(isinstance(value, (float, int)) and not isinstance(value, bool)
                and math.isfinite(value) for value in values)
            or min(distance_m, stop_distance_m, deceleration_m_s2,
                   wheel_circumference_m, max_rpm) <= 0
            or min(speed_bound_m_s, age_sec, outer_speed_m_s,
                   response_delay_sec) < 0):
        return None
    target = (max(0., target_speed_m_s-2.*age_sec)
              if target_speed_m_s > 0 else target_speed_m_s)
    margin = (distance_m-stop_distance_m
              - max(0., speed_bound_m_s-target)*age_sec
              - speed_bound_m_s*RELATIVE_CONTINUATION_FEEDBACK_MAX_AGE_SEC-.02)
    closing = max(0., outer_speed_m_s-target)
    try:
        required = closing*response_delay_sec + closing**2/(2.*deceleration_m_s2)
        if margin <= 0 or required > margin:
            cap = 0.
        else:
            # Positive quadratic root: the greatest new relative speed that
            # still leaves room for response delay and modeled deceleration.
            root = deceleration_m_s2*(
                math.sqrt(response_delay_sec**2+2.*margin/deceleration_m_s2)
                - response_delay_sec)
            request_speed = target+root
            if same_grant_no_acceleration:
                request_speed = min(request_speed, speed_bound_m_s)
            cap = min(max_rpm, max(0., request_speed)*60./wheel_circumference_m)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    if not all(math.isfinite(value) for value in (margin, required, target, cap)):
        return None
    return RelativeBrakingBudget(margin, required, target, cap)


def continuation_feedback_speeds(*, feedback, now, circumference, max_rpm,
                                 original_speed_bound, feedback_max_age_sec=.10,
                                 reverse_tail_rpm=0.):
    """Validate one cached encoder sample for BOTH continuation paths.

    Return body and outer-wheel speeds in m/s, or an explicit veto. Legacy
    callers retain their 100ms/no-reverse contract. The relative policy may
    admit a signed low-speed tail only when BOTH wheels are within its small
    bound; absolute outer speed still contributes braking momentum. This is
    not a stationary confirmation or permission to bypass zero-cross guards.
    """
    config = (now, circumference, max_rpm, original_speed_bound,
              feedback_max_age_sec, reverse_tail_rpm)
    if (not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                and math.isfinite(v) for v in config)
            or min(circumference, max_rpm) <= 0 or original_speed_bound < 0
            or not 0 < feedback_max_age_sec <= RELATIVE_CONTINUATION_FEEDBACK_MAX_AGE_SEC
            or not 0 <= reverse_tail_rpm <= RELATIVE_CONTINUATION_REVERSE_TAIL_RPM):
        return None, None, "invalid_braking_model"
    if feedback is None or not getattr(feedback, "trustworthy", False):
        return None, None, "continuation_feedback_missing"
    values = tuple(getattr(feedback, key, None) for key in (
        "timestamp", "left_forward_rpm", "right_forward_rpm"))
    if (not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                and math.isfinite(v) for v in values)
            or not 0 <= now-values[0] <= feedback_max_age_sec
            or max(abs(v) for v in values[1:]) > max_rpm):
        return None, None, "continuation_feedback_invalid"
    if min(values[1:]) < 0 and max(abs(v) for v in values[1:]) > reverse_tail_rpm:
        return None, None, "continuation_feedback_invalid"
    body = max(0., .5*(values[1]+values[2]))*circumference/60.
    outer = max(abs(v) for v in values[1:])*circumference/60.
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
            or sample_age > MAX_FORWARD_DEPTH_TTL_SEC):
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


def relative_continuation_speed_cap(*, distance, stop_distance, original_speed_bound,
                                    sample_age, feedback, now, circumference, max_rpm,
                                    deceleration, response_delay, motion, target_id,
                                    sample_timestamp):
    """Bound an existing grant with age-degrading SAME-window target motion.

    Positive target motion loses 2 m/s² of credit; approaching-target motion
    is never clipped upward to zero. This relative-motion trial accepts some
    approach/overshoot risk, NOT an absolute stationary-target stop guarantee.

    Past travel always spends the ORIGINAL speed bound, not a newly lowered
    command. With fixed grant data the nonzero cap is nonincreasing with age;
    a different encoder sample can only veto it, not increase it. The caller
    must latch any veto so a stopped old grant cannot revive, and must still
    enforce identity, visibility, hazards and the unchanged motor deadline.
    """
    vals = (distance, stop_distance, original_speed_bound, sample_age, now,
            circumference, max_rpm, deceleration, response_delay, sample_timestamp)
    if (not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                and math.isfinite(v) for v in vals)
            or min(distance, stop_distance, circumference, max_rpm, deceleration,
                   sample_timestamp) <= 0
            or min(original_speed_bound, sample_age, response_delay) < 0
            or sample_age > MAX_FORWARD_DEPTH_TTL_SEC
            or not 0 <= now-sample_timestamp <= MAX_FORWARD_DEPTH_TTL_SEC
            or abs(now-sample_timestamp-sample_age) > 1e-9):
        return 0., "invalid_braking_model"
    if (not isinstance(motion, ContinuationMotionEvidence)
            or not motion.valid_for(target_id, sample_timestamp)):
        return 0., "invalid_motion_evidence"
    _body, outer, reason = continuation_feedback_speeds(
        feedback=feedback, now=now, circumference=circumference, max_rpm=max_rpm,
        original_speed_bound=original_speed_bound,
        feedback_max_age_sec=RELATIVE_CONTINUATION_FEEDBACK_MAX_AGE_SEC,
        reverse_tail_rpm=RELATIVE_CONTINUATION_REVERSE_TAIL_RPM)
    if outer is None:
        return 0., reason
    budget = relative_braking_budget(
        distance_m=distance, stop_distance_m=stop_distance,
        speed_bound_m_s=original_speed_bound, age_sec=sample_age,
        outer_speed_m_s=outer, target_speed_m_s=motion.target_speed_bound_m_s,
        deceleration_m_s2=deceleration, response_delay_sec=response_delay,
        wheel_circumference_m=circumference, max_rpm=max_rpm)
    if budget is None:
        return 0., "invalid_braking_model"
    if budget.margin_m <= 0 or budget.required_stop_m > budget.margin_m:
        return 0., "braking_margin"
    return budget.max_allowed_rpm, "same_grant_relative_braking_cap"
