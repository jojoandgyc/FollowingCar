"""Stop-only search geometry. This module cannot assign identity or authorize motion."""
import math


def search_brake_reason(*, bbox, width, direction, eligible, now,
                        capture_timestamp, max_age, feedback, policy):
    if not eligible or direction not in ("left", "right") or width <= 0:
        return None
    if (not math.isfinite(capture_timestamp) or capture_timestamp <= 0
            or not 0 <= now-capture_timestamp <= max_age):
        return None
    if (bbox is None or len(bbox) != 4 or not all(math.isfinite(v) for v in bbox)
            or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
        return None
    x = (bbox[0]+bbox[2])/(2*width)
    if not 0 <= x <= 1:
        return None
    left = float(getattr(policy, "center_left_ratio", .45))
    right = float(getattr(policy, "center_right_ratio", .55))
    if left <= x <= right:
        return "candidate_center"
    if feedback is None or not feedback.trustworthy:
        return None
    stamp = feedback.timestamp
    yaw = getattr(feedback, "raw_yaw_rate_right_dps", None)
    if yaw is None:
        yaw = getattr(feedback, "yaw_rate_right_dps", None)
    if (yaw is None or not math.isfinite(yaw) or not math.isfinite(stamp)
            or not 0 <= now-stamp <= policy.visible_steering_pid_feedback_stale_sec):
        return None
    sign = -1 if direction == "left" else 1
    # Only residual search rotation qualifies; normal tracking is untouched.
    if yaw*sign <= 0:
        return None
    hfov = policy.visible_steering_pid_camera_hfov_deg
    error = (x-.5)*hfov
    if error*sign < 0:
        return "candidate_crossed_center"
    decel = policy.visible_steering_pid_predictive_brake_decel_dps2
    if decel <= 0:
        return None
    latency = max(now-capture_timestamp, policy.visible_steering_pid_camera_latency_sec)
    latency += max(0., policy.visible_steering_pid_predictive_brake_response_sec)
    stopping = yaw*yaw/(2*decel)+abs(yaw)*latency
    remaining = max(0., abs(error)-min(.5-left, right-.5)*hfov)
    if stopping+max(0., policy.visible_steering_pid_predictive_brake_margin_deg) >= remaining:
        return "candidate_predictive_stop"
    return None
