"""Search/handoff braking constraints, never identity or motion authorization."""
import math
import time
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


@dataclass(frozen=True)
class HandoffZeroEvidence:
    """Continuous completed speed-zero receipts and independent quiet reads.

    This ends an obsolete residual-search constraint, never grants motion.
    The writer publishes one immutable snapshot; producer reads do no I/O.
    """
    episode: tuple
    receipt: object
    zero_completed_at: float
    quiet_samples: tuple = ()


# This is a low-residual handoff, NOT proof of a stationary chassis. It only
# avoids upgrading an already-applied observation zero into a second parking
# cycle; fresh identity, Depth and wheel-reversal checks remain independent.
OBSERVATION_ZERO_HANDOFF_MAX_SEC = .30
OBSERVATION_ZERO_HANDOFF_FEEDBACK_SEC = .10
OBSERVATION_ZERO_HANDOFF_MAX_WHEEL_RPM = 3.
OBSERVATION_ZERO_HANDOFF_MAX_YAW_DPS = 10.


@dataclass(frozen=True)
class ObservationZeroEvidence:
    episode: tuple
    receipt: object
    stop_generation: int
    completed_at: float


def observation_zero_episode(runtime, previous=None):
    """Only the existing finite search-observation episode can supply zero."""
    owner, backend = runtime.owner, runtime.backend
    controller = getattr(owner, "_follow_controller", None)
    uid = getattr(controller, "active_target_id", None)
    requested = getattr(owner, "_search_retry_zero_requested_at", None)
    epoch = getattr(owner, "_search_epoch", None)
    # Confirmed identity releases the detector's timer before it asks the
    # brake gate about residual motion. That release is not a new motor
    # action: retain the immutable completed-zero origin, never invent one.
    if requested == 0. and previous is not None:
        requested = previous.episode[2]
    if (uid is None or not isinstance(epoch, int) or isinstance(epoch, bool)
            or not _finite_values(requested) or requested <= 0
            or not getattr(owner, "running", False)
            or getattr(owner, "search_state", None) != "searching"
            or getattr(controller, "search_state", None) != "searching"
            or any(getattr(owner, name, False) for name in (
                "_explicit_stop_requested", "_runtime_shutdown_requested",
                "_near_yaw_park_request", "_brake_hold_active",
                "stop_action_execution", "person_detected_flag"))
            or getattr(runtime, "_search_reacquire_brake_request", None) is not None
            or any(getattr(backend, name, False) for name in (
                "motion_write_fault", "parking_release_fault", "normal_zero_hold",
                "parking_current_a", "_parking_current_uncertain"))):
        return None
    return epoch, uid, requested


def note_observation_zero(runtime, *, now, first_ack, previous_receipt):
    """Bind successful observation zeros, never an arbitrary historical zero.

    Caller holds motor_io_lock after a completed speed write. Repeated zero
    receipts may continue the chain, but cannot move its original time limit.
    An intervening nonzero, STOP, in-flight/failed write breaks it permanently
    for this observation; only a genuinely new observation may create one.
    """
    previous = getattr(runtime, "_search_observation_zero_evidence", None)
    runtime._search_observation_zero_evidence = None
    episode = observation_zero_episode(runtime, previous if not first_ack else None)
    backend = runtime.backend
    receipt = getattr(backend, "last_speed_receipt", None)
    submission = getattr(backend, "last_speed_write", None)
    completed = getattr(receipt, "completed_at", None)
    generation = getattr(backend, "stop_write_generation", None)
    if (episode is None or receipt is None or submission is None
            or getattr(submission, "completed_receipt", None) is not receipt
            or (getattr(receipt, "left_rpm", None), getattr(receipt, "right_rpm", None)) != (0, 0)
            or not _finite_values(completed, now, generation)
            or not episode[2] <= completed <= now
            or getattr(submission, "stop_generation", None) != generation):
        return
    if first_ack:
        runtime._search_observation_zero_evidence = ObservationZeroEvidence(
            episode, receipt, generation, completed)
    elif (previous is not None and previous.episode == episode
            and previous.stop_generation == generation
            and previous_receipt is previous.receipt
            and receipt.sequence == previous.receipt.sequence + 1):
        runtime._search_observation_zero_evidence = ObservationZeroEvidence(
            episode, receipt, generation, previous.completed_at)


def observation_zero_handoff(runtime, *, uid, feedback, now):
    """Current low residual after our zero, not stillness or motion permission."""
    evidence = getattr(runtime, "_search_observation_zero_evidence", None)
    backend = runtime.backend
    episode = observation_zero_episode(runtime, evidence)
    if (evidence is None or episode != evidence.episode or uid != episode[1]
            or getattr(backend, "last_speed_receipt", None) is not evidence.receipt
            or getattr(backend, "stop_write_generation", None) != evidence.stop_generation
            or getattr(getattr(backend, "last_speed_write", None), "completed_receipt", None)
                is not evidence.receipt):
        runtime._search_observation_zero_evidence = None
        return None
    if not _finite_values(now) or not 0 <= now-evidence.completed_at <= OBSERVATION_ZERO_HANDOFF_MAX_SEC:
        runtime._search_observation_zero_evidence = None
        return None
    if (feedback is None or not getattr(feedback, "trustworthy", False)
            or not getattr(feedback, "yaw_rate_confirmed", False)
            or getattr(feedback, "left_error", None) != 0
            or getattr(feedback, "right_error", None) != 0):
        return None
    values = tuple(getattr(feedback, name, None) for name in (
        "timestamp", "left_read_started", "left_read_finished", "right_read_started",
        "right_read_finished", "left_forward_rpm", "right_forward_rpm",
        "raw_yaw_rate_right_dps"))
    if not _finite_values(*values):
        return None
    stamp, ls, lf, rs, rf, left, right, yaw = values
    # Use actual post-write encoder reads, not image age, filter lag or the
    # time at which this producer happened to inspect the cached sample.
    if not (evidence.completed_at < ls <= lf <= rs <= rf <= stamp <= now
            and now-stamp <= OBSERVATION_ZERO_HANDOFF_FEEDBACK_SEC
            and max(abs(left), abs(right)) <= OBSERVATION_ZERO_HANDOFF_MAX_WHEEL_RPM
            and abs(yaw) <= OBSERVATION_ZERO_HANDOFF_MAX_YAW_DPS):
        return None
    return evidence


def _finite_values(*values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def paired_search_takeover_available(owner, feedback, now):
    """An acknowledged small search pivot can enter the paired writer directly.

    Called ONLY for a current, independently confirmed candidate. This is not
    motion authorization: the normal identity, range and terminal wheel checks
    still run. It avoids starting a new park/quiet/new-image cycle merely
    because that candidate crossed the old search direction's centre line.
    Unknown motion, a STOP already underway, or abnormal feedback cannot use it.
    All evidence is cached; no serial reads or additional locks are introduced.
    """
    runtime = getattr(owner, "_action_runtime", None)
    backend = getattr(runtime, "backend", None)
    short = getattr(owner, "_short_follow", None)
    config = getattr(short, "config", None)
    ctl = getattr(owner, "_follow_controller", None)
    uid = getattr(ctl, "active_target_id", None)
    if (getattr(owner, "_short_follow_adapter", None) is None
            or not getattr(config, "enabled", False) or type(uid) is not int or uid <= 0
            or not getattr(owner, "running", False)
            or any(getattr(owner, key, False) for key in (
                "_explicit_stop_requested", "_runtime_shutdown_requested",
                "_near_yaw_park_request", "_brake_hold_active", "stop_action_execution"))
            or getattr(runtime, "_search_reacquire_brake_request", None) is not None
            or any(getattr(backend, key, False) for key in (
                "motion_write_fault", "parking_release_fault", "normal_zero_hold",
                "parking_current_a", "_parking_current_uncertain"))):
        return False
    receipt = getattr(backend, "last_speed_receipt", None)
    submission = getattr(backend, "last_speed_write", None)
    completed = getattr(receipt, "completed_at", None)
    if (receipt is None or submission is None
            or getattr(submission, "completed_receipt", None) is not receipt
            or getattr(submission, "stop_generation", None) != getattr(backend, "stop_write_generation", None)
            or not _finite_values(now, completed) or not 0 <= now-completed <= .15):
        return False
    left = receipt.left_rpm * backend.wheel_raw_state_to_target("left", 1, 0x01)
    right = receipt.right_rpm * backend.wheel_raw_state_to_target("right", 1, 0x01)
    if not (_finite_values(left, right) and left == -right and 0 < abs(left) <= 8):
        return False
    if (feedback is None or not getattr(feedback, "trustworthy", False)
            or getattr(feedback, "left_error", None) != 0
            or getattr(feedback, "right_error", None) != 0):
        return False
    stamp = getattr(feedback, "timestamp", None)
    measured = (getattr(feedback, "left_forward_rpm", None),
                getattr(feedback, "right_forward_rpm", None))
    return bool(_finite_values(stamp, *measured) and 0 <= now-stamp <= .10
        and max(map(abs, measured)) <= 10 and max(measured) >= -2
        and all(value >= -2 or (target < 0 and value >= target-2)
                for value, target in zip(measured, (left, right))))


def _zero_handoff_episode(runtime, uid):
    owner = runtime.owner
    backend = runtime.backend
    started = getattr(owner, "_search_handoff_started_capture_ts", None)
    direction = getattr(owner, "_search_handoff_direction", None)
    if (uid is None or getattr(owner, "_search_handoff_uid", None) != uid
            or getattr(getattr(owner, "_follow_controller", None), "active_target_id", None) != uid
            or getattr(owner, "search_state", None) != "none"
            or getattr(getattr(owner, "_follow_controller", None), "search_state", "none") != "none"
            or direction not in ("left", "right") or not _finite_values(started) or started <= 0
            or any(getattr(owner, name, False) for name in (
                "_explicit_stop_requested", "_runtime_shutdown_requested", "_near_yaw_park_request",
                "_brake_hold_active", "stop_action_execution", "person_detected_flag"))
            or any(getattr(runtime, name, False) for name in (
                "_search_reacquire_brake_request", "_near_yaw_park_settling"))
            or any(getattr(backend, name, False) for name in (
                "motion_write_fault", "parking_release_fault", "normal_zero_hold",
                "parking_current_a", "_parking_current_uncertain"))):
        return None
    return uid, started, direction


def _quiet_post_zero(feedback, zero_completed_at, now):
    if (feedback is None or not getattr(feedback, "trustworthy", False)
            or not getattr(feedback, "yaw_rate_confirmed", False)):
        return False
    values = tuple(getattr(feedback, name, None) for name in (
        "timestamp", "left_read_started", "left_read_finished", "right_read_started",
        "right_read_finished", "left_forward_rpm", "right_forward_rpm",
        "raw_yaw_rate_right_dps", "yaw_rate_right_dps"))
    if not _finite_values(*values, zero_completed_at, now):
        return False
    stamp, ls, lf, rs, rf, left, right, raw, filtered = values
    return bool(zero_completed_at < ls <= lf <= rs <= rf <= stamp <= now
        and now-stamp <= .10 and getattr(feedback, "left_error", None) == 0
        and getattr(feedback, "right_error", None) == 0
        and max(abs(left), abs(right)) <= 1. and max(abs(raw), abs(filtered)) <= 2.)


def note_handoff_zero_write(runtime, *, uid, previous_receipt, packet_written, feedback, now):
    """Called only after normal-follow I/O, under the existing motor lock.

    Repeated zero packets continue the episode only if the preceding exact
    receipt is ours. Any intervening speed write, STOP, parking or failed I/O
    breaks the chain. Backends without completed receipts disable this path.
    """
    previous = getattr(runtime, "_search_handoff_zero_evidence", None)
    runtime._search_handoff_zero_evidence = None
    episode = _zero_handoff_episode(runtime, uid)
    receipt = getattr(runtime.backend, "last_speed_receipt", None)
    completed = getattr(receipt, "completed_at", None)
    sequence = getattr(receipt, "sequence", None)
    if (not packet_written or episode is None or receipt is None
            or (getattr(receipt, "left_rpm", None), getattr(receipt, "right_rpm", None)) != (0, 0)
            or not _finite_values(completed, now, sequence) or not 0 < completed <= now
            or receipt is previous_receipt):
        return
    continuous = bool(previous is not None and previous.episode == episode
        and previous.receipt is previous_receipt
        and sequence == getattr(previous_receipt, "sequence", -2)+1
        and previous.zero_completed_at <= completed)
    zero_at = previous.zero_completed_at if continuous else completed
    samples = previous.quiet_samples if continuous else ()
    if not _quiet_post_zero(feedback, zero_at, now):
        samples = ()
    elif not samples:
        samples = (feedback,)
    elif feedback.timestamp < samples[-1].timestamp:
        samples = ()
    elif feedback.timestamp > samples[-1].timestamp:
        if (feedback.left_read_started <= samples[-1].timestamp
                or feedback.timestamp-samples[0].timestamp > .15):
            samples = (feedback,)
        else:
            samples = (samples[0], feedback)
    runtime._search_handoff_zero_evidence = HandoffZeroEvidence(episode, receipt, zero_at, samples)


def _settled_handoff(owner, *, uid, raw_track_id, cap, stamp, direction):
    runtime = getattr(owner, "_action_runtime", None)
    settled = getattr(runtime, "_search_handoff_zero_evidence", None)
    moving = getattr(owner, "_search_handoff_moving_evidence", None)
    if (settled is None or moving is None or len(settled.quiet_samples) != 2
            or (moving.uid, moving.raw_track_id, moving.cap, moving.stamp, moving.direction)
                != (uid, raw_track_id, cap, stamp, direction)
            or _zero_handoff_episode(runtime, uid) != settled.episode
            or getattr(runtime.backend, "last_speed_receipt", None) is not settled.receipt
            or not settled.zero_completed_at < stamp
            or not .04 <= settled.quiet_samples[-1].timestamp-settled.quiet_samples[0].timestamp <= .15):
        return False
    get_feedback = getattr(runtime, "get_steering_feedback", None)
    if not callable(get_feedback):
        return False
    feedback = get_feedback()
    now = time.monotonic()  # cache acquisition cannot renew visual/encoder age
    return bool(0 <= now-stamp <= moving.max_age
        and 0 <= now-settled.quiet_samples[-1].timestamp <= .10
        and _quiet_post_zero(feedback, settled.zero_completed_at, now)
        and feedback.timestamp >= settled.quiet_samples[-1].timestamp
        and _zero_handoff_episode(runtime, uid) == settled.episode
        and getattr(runtime.backend, "last_speed_receipt", None) is settled.receipt)


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
    if _settled_handoff(owner, uid=uid, raw_track_id=raw_track_id, cap=cap,
                        stamp=stamp, direction=direction):
        return "confirmed_post_zero_stillness"
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
