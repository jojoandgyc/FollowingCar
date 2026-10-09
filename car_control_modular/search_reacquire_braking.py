"""Search/handoff braking constraints, never identity or motion authorization."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MovingHandoffEvidence:
    uid: int
    raw_track_id: int
    cap: int
    stamp: float
    x: float
    direction: str
    max_age: float


def observe_moving_handoff(owner, *, eligible, uid, raw_track_id, cap,
                           stamp, bbox, width, direction, now, max_age):
    """Publish confirmed geometry only. Replays cannot refresh its lease."""
    if (not eligible or uid is None or raw_track_id is None or cap <= 0
            or direction not in ("left", "right") or width <= 0
            or bbox is None or len(bbox) != 4
            or not all(math.isfinite(v) for v in (*bbox, stamp, now, max_age))
            or not 0 <= bbox[0] < bbox[2] <= width or bbox[3] <= bbox[1]
            or stamp <= 0 or not 0 <= now-stamp <= min(.25, max_age)):
        owner._search_handoff_moving_evidence = None
        return None
    previous = getattr(owner, "_search_handoff_moving_evidence", None)
    if previous is not None and (cap <= previous.cap or stamp <= previous.stamp):
        return previous if (uid, raw_track_id, cap, stamp) == (
            previous.uid, previous.raw_track_id, previous.cap, previous.stamp) else None
    evidence = MovingHandoffEvidence(uid, raw_track_id, cap, stamp,
        (bbox[0]+bbox[2])/(2*width), direction, min(.25, max_age))
    owner._search_handoff_moving_evidence = evidence
    return evidence


def moving_handoff_yaw(owner, *, evidence, uid, base, yaw, feedback, now,
                       policy, execution_delay_sec=.05):
    """Return (yaw constraint, phase), or None when moving proof is unavailable.

    The caller must independently hold a current positive Depth grant and yaw
    authority at the physical write. Counter-differential never increases base
    or reverses a wheel. It is recalculated from real feedback on every tick,
    not a fixed pulse or a command remembered from the visual frame.
    """
    if (evidence is None or evidence is not getattr(owner, "_search_handoff_moving_evidence", None)
            or getattr(owner, "_search_handoff_uid", None) != uid or evidence.uid != uid
            or getattr(owner, "search_state", None) != "none"
            or policy is None or not all(math.isfinite(v) for v in (base, yaw, now))
            or base <= 0 or not 0 <= now-evidence.stamp <= evidence.max_age
            or feedback is None or not feedback.trustworthy):
        return None
    started = getattr(owner, "_search_handoff_started_capture_ts", None)
    if started is None or not math.isfinite(started) or not 0 <= now-started <= 1.25:
        return None
    raw = getattr(feedback, "raw_yaw_rate_right_dps", None)
    filtered = getattr(feedback, "yaw_rate_right_dps", None)
    values = (feedback.timestamp, raw, filtered,
              getattr(feedback, "left_forward_rpm", None), getattr(feedback, "right_forward_rpm", None))
    if (any(v is None or not math.isfinite(v) for v in values)
            or not 0 <= now-feedback.timestamp <= min(.10, policy.visible_steering_pid_feedback_stale_sec)
            or abs(raw-filtered) > 12):
        return None
    decel = policy.visible_steering_pid_predictive_brake_decel_dps2
    gain = policy.visible_steering_pid_predictive_countersteer_gain_rpm_per_dps
    cap = min(6., policy.visible_steering_pid_predictive_countersteer_max_correction_rpm,
              handoff_yaw_limit(owner, uid) or 0., base)
    if not all(math.isfinite(v) and v > 0 for v in (decel, gain, cap)):
        return None
    sign = -1 if evidence.direction == "left" else 1
    error = sign*(evidence.x-.5)*policy.visible_steering_pid_camera_hfov_deg
    band = min(.5-policy.center_left_ratio, policy.center_right_ratio-.5)
    remaining = max(0., error-max(0., band)*policy.visible_steering_pid_camera_hfov_deg
                    -max(0., policy.visible_steering_pid_predictive_brake_margin_deg))
    latency = max(now-evidence.stamp, policy.visible_steering_pid_camera_latency_sec)
    latency += max(0., policy.visible_steering_pid_predictive_brake_response_sec)
    latency += min(.15, max(0., execution_delay_sec))
    safe_rate = decel*(math.sqrt(latency*latency+2*remaining/decel)-latency)
    residual = sign*raw
    bounded = limit_handoff_yaw(owner, uid, yaw)
    if residual > max(2., safe_rate):
        # Even if PID still sees the old side of the image, do not keep
        # accelerating into the stopping envelope. Existing opposite demand
        # is bounded too; neither individual wheel may be commanded reverse.
        correction = min(cap, max(0., residual-safe_rate)*gain)
        return -sign*max(correction, min(cap, max(0., -sign*bounded))), "residual_counter_differential"
    if remaining == 0 and sign*bounded > 0:
        return 0., "center_same_direction_removed"
    return math.copysign(min(abs(bounded), base), bounded), "tracking"


def handoff_straight_allowed(owner, *, moving, evidence, uid, feedback, now):
    """A current intentional yaw zero can retain separately authorized Depth.

    The moving constraint must already allow equal wheels. This supplies no
    yaw authority: a residual requiring counter-differential still needs the
    normal yaw grant. Both writer checks call this with their own current time.
    """
    if moving is None or moving[0] != 0. or feedback is None:
        return False
    if abs(feedback.raw_yaw_rate_right_dps) <= 2.:
        return True
    store = getattr(owner, "_lateral_intent_store", None)
    intent = store.snapshot() if store is not None else None
    return bool(
        moving[1] == "tracking" and evidence is not None and intent is not None
        and intent.target_id == uid == evidence.uid
        and intent.capture_frame_id == evidence.cap
        and intent.capture_timestamp == evidence.stamp
        and intent.bbox_quality == "reliable" and not intent.near_distance_mode
        and math.isfinite(intent.published_at) and intent.published_at <= now
        and intent.valid(now) and 0 <= now-evidence.stamp <= evidence.max_age
        and (intent.nominal_valid_until <= 0 or now <= intent.nominal_valid_until)
        and (intent.hold_zero
             or getattr(owner, "_lateral_intent_zero_sequence", None) == intent.sequence)
    )


class HandoffObservation:
    """Confirmed capture continuity, not identity or motor authorization."""

    def __init__(self):
        self.key = None
        self.samples = []

    def outward(self, *, uid, raw_track_id, cap, stamp, bbox, width, direction):
        key = (uid, raw_track_id, direction)
        if (raw_track_id is None or width <= 0 or len(bbox) != 4
                or not all(math.isfinite(v) for v in (*bbox, stamp))
                or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            self.samples.clear()
            return False
        x = (bbox[0] + bbox[2]) / (2 * width)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if key != self.key:
            self.samples.clear()
        self.key = key
        if self.samples:
            pc, pt, px, pa = self.samples[-1]
            if cap <= pc or stamp <= pt:
                return False
            if not (.025 <= stamp-pt <= .25 and abs(x-px) <= .10
                    and .6 <= area/pa <= 1.67):
                self.samples.clear()
        self.samples.append((cap, stamp, x, area))
        self.samples = self.samples[-3:]
        if len(self.samples) != 3 or direction not in ("left", "right"):
            return False
        sign = -1 if direction == "left" else 1
        span = self.samples[-1][1] - self.samples[0][1]
        slopes = [sign*(b[2]-a[2])/(b[1]-a[1])
                  for a, b in zip(self.samples, self.samples[1:])]
        return (.08 <= span <= .35 and all(sign*(s[2]-.5) > .10 for s in self.samples)
                and all(.03 <= rate <= 1.0 for rate in slopes))


def handoff_retirement_reason(owner, *, eligible, uid, raw_track_id, cap,
                              stamp, bbox, width, direction):
    """Only called for a fresh confirmed UID after search release.

    End the residual-search stop obligation, not an active brake. Normal
    visual braking, wheel reversal, freshness and safety checks still apply.
    A timer alone never emits a wheel command or revives an old grant.
    """
    if not eligible:
        owner._search_handoff_observation = HandoffObservation()
        return None
    if (uid is None or cap <= 0 or width <= 0 or bbox is None or len(bbox) != 4
            or not all(math.isfinite(v) for v in (*bbox, stamp))
            or not 0 <= bbox[0] < bbox[2] <= width or bbox[3] <= bbox[1]):
        return None
    previous = getattr(owner, "_search_handoff_last_capture", None)
    if previous is not None and (cap <= previous[0] or stamp <= previous[1]):
        return None
    owner._search_handoff_last_capture = (cap, stamp)
    started = getattr(owner, "_search_handoff_started_capture_ts", None)
    if started is not None and math.isfinite(started) and stamp-started >= .75:
        return "fresh_confirmed_follow_takeover"
    observation = getattr(owner, "_search_handoff_observation", None)
    if observation is None:
        observation = owner._search_handoff_observation = HandoffObservation()
    if observation.outward(uid=uid, raw_track_id=raw_track_id, cap=cap,
                           stamp=stamp, bbox=bbox, width=width, direction=direction):
        return "confirmed_outward_continuity"
    return None


def handoff_yaw_limit(owner, uid):
    """A search-to-follow ceiling, never an identity or motion permission.

    Lives only during residual search handoff, until stop or fresh takeover.
    No timer/old image can raise it; normal following and other UIDs are intact.
    """
    if (uid is None or getattr(owner, "_search_handoff_uid", None) != uid
            or getattr(owner, "search_state", None) != "none"):
        return None
    value = getattr(owner, "_search_handoff_cap_rpm", None)
    if value is None or not math.isfinite(float(value)) or value < 0:
        return None
    return float(value)


def limit_handoff_yaw(owner, uid, yaw):
    limit = handoff_yaw_limit(owner, uid)
    return yaw if limit is None else math.copysign(min(abs(yaw), limit), yaw)


def search_brake_reason(*, bbox, width, direction, eligible, now,
                        capture_timestamp, max_age, feedback, policy,
                        execution_delay_sec=0.0):
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
    # Time still needed to dispatch STOP, separate from image age/response.
    # Bounded scheduling allowance, not a certified hardware braking model.
    if math.isfinite(execution_delay_sec):
        latency += min(.15, max(0., execution_delay_sec))
    stopping = yaw*yaw/(2*decel)+abs(yaw)*latency
    remaining = max(0., abs(error)-min(.5-left, right-.5)*hfov)
    if stopping+max(0., policy.visible_steering_pid_predictive_brake_margin_deg) >= remaining:
        return "candidate_predictive_stop"
    return None
