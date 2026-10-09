"""Bounded forward turn establishment from post-write encoder evidence.

All pairs use forward-positive RPM; yaw is HALF the left-right differential.
No identity/Depth authority, serial reads, sleeps or motor writes live here.
"""
from collections import deque
import math


def steering_buildup_allowed(result):
    """Numerical PID eligibility shared by vision and encoder refresh.

    This is only one gate: the immutable intent and writer still check UID,
    capture TTL, center approach, real deceleration, two post-write samples,
    wheel reversal and the fixed 350ms episode. No legacy/unknown evidence.
    """
    if (result is None or getattr(result, "correction_limit_reason", None) != "image_error_only"
            or getattr(result, "predictive_braking", False)
            or getattr(result, "visual_direction_guarded", False)
            or getattr(result, "opposite_yaw_braking", False)
            or getattr(result, "same_direction_overspeed_braking", False)
            or getattr(result, "outward_lead", None) is not None
            or getattr(result, "post_park_recenter", False)
            or getattr(result, "outward_continuity_rate_dps", None) is not None
            or getattr(result, "output_floor_reason", None) != "image_error_only"):
        return False
    try:
        demand = float(result.position_demand_rpm)
        reduction = float(result.brake_reduction_rpm)
        output = float(result.correction_rpm)
        error = float(result.visual_error_deg)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return bool(all(math.isfinite(v) for v in (demand, reduction, output, error))
                and demand > 0 and 0 <= reduction <= 1e-9
                and output*error > 0 and abs(output) in (math.floor(demand), round(demand)))


class TurnBuildup:
    def __init__(self, *, legacy_common_accel_limit=False, response_delay_sec=.5):
        # Pure-policy opt-in for offline A/B only. The runtime constructs the
        # default policy: differential assistance must not revoke an approved
        # longitudinal demand. Brake/authority/wheel limits remain downstream.
        self.legacy_common_accel_limit = legacy_common_accel_limit is True
        if not math.isfinite(response_delay_sec) or not .1 <= response_delay_sec <= 1.:
            raise ValueError('invalid motor response observation delay')
        self.response_delay_sec = response_delay_sec
        self.uid = None
        self.history = deque(maxlen=32)
        self.sign = 0
        self.latest_cap = 0
        self.blocked_cap = 0
        self.feedback_stamp = 0.
        self.lag_count = 0
        self.started = None
        self.closed = False

    def _identity(self, uid):
        if uid != self.uid:
            self.__init__(legacy_common_accel_limit=self.legacy_common_accel_limit,
                          response_delay_sec=self.response_delay_sec)
            self.uid = uid

    def sync_writer(self, previous_sent):
        if self.history and abs(self.history[-1][0]-previous_sent) > 1e-6:
            self.history.clear()
            self.lag_count = 0
            if self.started is not None:
                self.closed = True

    def note_sent(self, uid, pair, stamp, previous_sent):
        self._identity(uid)
        if (self.history and (abs(self.history[-1][0]-previous_sent) > 1e-6
                              or stamp-previous_sent > .20)):
            self.history.clear()
            self.lag_count = 0
            if self.started is not None:
                self.closed = True
        diff = pair[0]-pair[1]
        since = stamp
        if self.history and diff*self.history[-1][1] > 0 and abs(diff-self.history[-1][1]) <= 2:
            since = self.history[-1][2]
        self.history.append((stamp, diff, since))
        if min(pair) < 0 or sum(pair) <= 0 or diff == 0:
            self.sign = 0
            self.blocked_cap = max(self.blocked_cap, self.latest_cap)
            self.started = None
            self.closed = False
            self.lag_count = 0

    def adjust(self, base, yaw, intent, feedback, now, uid, *, limit, permitted=True):
        self._identity(uid)

        def veto(reason):
            self.lag_count = 0
            if self.started is not None:
                self.closed = True
            return base, yaw, reason

        if (intent is not None and intent.target_id == uid and intent.mode == 'forward'
                and intent.bbox_quality == 'reliable' and intent.valid(now)
                and intent.response_boost_allowed and permitted
                and not intent.hold_zero and not intent.park_requested and not intent.near_distance_mode
                and .25 < now-intent.capture_timestamp <= .35):
            # Pause assistance, not ordinary authorized steering. A new
            # qualified frame can resume only within the ORIGINAL deadline.
            self.lag_count = 0
            return base, yaw, 'buildup_wait_fresh_visual'
        if (intent is None or intent.target_id != uid or intent.mode != 'forward'
                or intent.bbox_quality != 'reliable' or not intent.valid(now)
                or not 0 < intent.capture_timestamp <= now
                or now-intent.capture_timestamp > .25 or intent.capture_frame_id <= 0):
            return veto('buildup_visual_unqualified')
        cap = intent.capture_frame_id
        if cap < self.latest_cap:
            return veto('buildup_old_capture')
        self.latest_cap = cap
        error = intent.visual_error_deg
        rate = intent.target_image_rate_dps
        if (error is None or not all(math.isfinite(v) for v in (base, yaw, limit, error))
                or not permitted or not intent.response_boost_allowed
                or intent.hold_zero or intent.park_requested or intent.near_distance_mode
                or base <= abs(yaw) or abs(yaw) < 3 or abs(yaw) > limit
                or abs(error) < 6 or yaw*error <= 0):
            return veto('buildup_braking_or_scope_veto')
        sign = 1 if yaw > 0 else -1
        if rate is not None and (not math.isfinite(rate)
                or abs(error)-max(0., -rate*sign)*(now-intent.capture_timestamp+.18) <= 5):
            return veto('buildup_approaching_center')
        if (feedback is None or not feedback.trustworthy
                or not all(math.isfinite(v) for v in (feedback.timestamp,
                    feedback.left_forward_rpm, feedback.right_forward_rpm))
                or not 0 <= now-feedback.timestamp <= .10
                or min(feedback.left_forward_rpm, feedback.right_forward_rpm) < 0):
            return veto('buildup_feedback_unqualified')
        if self.sign != sign:
            if cap <= self.blocked_cap:
                return veto('buildup_new_capture_required')
            self.sign = sign
            self.started = None
            self.closed = False
            self.lag_count = 0
            self.feedback_stamp = 0.
        if self.closed:
            return base, yaw, 'buildup_episode_complete'
        if self.started is not None and now-self.started >= .35:
            self.closed = True
            return base, yaw, 'buildup_timeout'
        measured = (feedback.left_forward_rpm, feedback.right_forward_rpm)
        measured_diff = (measured[0]-measured[1])*sign
        reference = next((h for h in reversed(self.history) if h[0] < feedback.timestamp), None)
        if (reference is None or reference[1]*sign < 6
                or feedback.timestamp-reference[0] > .20):
            return veto('buildup_no_sent_reference')
        threshold = .6*min(abs(reference[1]), 2*abs(yaw))
        if feedback.timestamp-reference[2] < self.response_delay_sec:
            self.lag_count = 0
            self.feedback_stamp = max(self.feedback_stamp, feedback.timestamp)
            return base, yaw, 'buildup_motor_response_wait'
        if feedback.timestamp > self.feedback_stamp:
            consecutive = 0 < feedback.timestamp-self.feedback_stamp <= .15
            self.feedback_stamp = feedback.timestamp
            lagging = measured_diff < threshold
            self.lag_count = (self.lag_count+1 if consecutive else 1) if lagging else 0
        if measured_diff >= threshold:
            if self.started is not None:
                self.closed = True
            return base, yaw, 'buildup_established'
        if self.started is None:
            if self.lag_count < 2:
                return base, yaw, 'buildup_wait_two_feedback'
            self.started = now
        # Do not turn a falling/braking wheel demand back into acceleration.
        requested = (base+yaw, base-yaw)
        if any(r < m for r, m in zip(requested, measured)):
            return veto('buildup_deceleration_preserved')
        enhanced = min(abs(yaw)*1.4, max(0., limit), 20., base)
        enhanced = max(abs(yaw), math.floor(enhanced))
        if not self.legacy_common_accel_limit:
            # Add yaw around the independently authorized mean. At the yaw
            # ceiling this deliberately leaves the pair unchanged: there is
            # no extra differential available without changing a constraint.
            return base, sign*enhanced, 'buildup_yaw_only'
        inner = measured[1] if sign > 0 else measured[0]
        # Historical comparator, never selected by default runtime wiring.
        new_base = min(base, math.floor(inner+2.)+enhanced)
        return new_base, sign*enhanced, 'buildup_common_accel_limited' if new_base < base else 'buildup_yaw_only'
