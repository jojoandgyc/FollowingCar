"""Evidence checks for a small braking pulse, never ordinary reverse travel."""
import math


MAX_PULSE_SEC = .080
MAX_CAPTURE_AGE_SEC = .250
MAX_CORRECTION_RPM = 6  # Hard trial ceiling; configuration may only lower it.


def qualified_countersteer(result, x_ratio, configured_limit):
    if result is None:
        return False
    values = (getattr(result, "correction_rpm", 0),
              getattr(result, "visual_error_deg", float("nan")),
              getattr(result, "measured_yaw_rate_dps", float("nan")),
              getattr(result, "target_image_rate_dps", float("nan")),
              getattr(result, "feedback_age_sec", None), x_ratio, configured_limit)
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return False
    correction, error, yaw, rate, age, _, _ = values
    return bool(
        getattr(result, "output_floor_reason", "") == "predictive_countersteer"
        and getattr(result, "predictive_braking", False)
        and getattr(result, "feedback_used", False)
        and getattr(result, "target_rate_valid", False)
        and 0 < abs(correction) <= min(MAX_CORRECTION_RPM, configured_limit)
        and 0 <= age <= .1 and correction * yaw < 0
        and error * yaw > 0 and (x_ratio - .5) * error > 0
        and (rate * error < 0 or (abs(rate) <= 4 and abs(error) <= 6)))


def pulse_feedback_qualified(request, feedback, now):
    """Recheck actual wheel motion immediately before / during brake output."""
    if feedback is None or not feedback.trustworthy:
        return False
    yaw = getattr(feedback, "yaw_rate_right_dps", None)
    raw = getattr(feedback, "raw_yaw_rate_right_dps", None)
    vals = (feedback.timestamp, feedback.left_forward_rpm, feedback.right_forward_rpm,
            yaw, raw, request.capture_timestamp, request.countersteer_rpm,
            request.visual_error_deg, request.image_rate_dps, request.countersteer_until, now)
    if any(v is None or not math.isfinite(float(v)) for v in vals):
        return False
    left, right = feedback.left_forward_rpm, feedback.right_forward_rpm
    correction, error, rate = request.countersteer_rpm, request.visual_error_deg, request.image_rate_dps
    inward = max(0., -rate * (1 if error > 0 else -1))
    return bool(
        0 < abs(correction) <= MAX_CORRECTION_RPM
        and 0 < request.capture_timestamp <= now <= request.countersteer_until
        and now - request.capture_timestamp <= MAX_CAPTURE_AGE_SEC
        and 0 <= now - feedback.timestamp <= .1
        and 4 <= abs(raw) <= 35 and abs(yaw-raw) <= 10 and yaw*raw > 0
        and correction*raw < 0 and error*raw > 0
        # Only a slow pivot: never use this exception for forward travel,
        # asymmetric high-speed feedback or whole-car reversal.
        and left*right <= 0 and abs(left+right) <= 8
        and max(abs(left), abs(right)) <= 15
        and (left-right)*raw > 0 and abs(left-right) >= 3
        # If image motion already puts the target at center, go straight
        # to NORMAL instead of spending another pulse interval here.
        and abs(error)-inward*(now-request.capture_timestamp) > 4
        and (rate*error < 0 or (abs(rate) <= 4 and abs(error) <= 6)))
