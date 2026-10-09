"""One-shot, non-accelerating allocation near the image's soft side margin.

This is not a motor fault detector, an identity bridge or a new motion lease.
It only removes common acceleration from an already authorized forward pair.
The ordinary differential and all later stop/reversal checks remain intact.
"""
from collections import deque
import math


class ViewRetention:
    def __init__(self):
        self.uid = None
        self.captures = deque(maxlen=3)
        self.writes = deque(maxlen=32)
        self.receipt = None
        self.feedback_stamp = 0.
        self.lag_count = 0
        self.deadline = None
        self.spent = False
        self.sign = 0
        self.capture_watermark = 0
        self.capture_time_watermark = 0.

    def _identity(self, uid):
        if uid != self.uid:
            self.__init__()
            self.uid = uid

    def _close(self):
        self.lag_count = 0
        if self.deadline is not None:
            self.spent = True

    def sync_writer(self, stamp, receipt):
        if self.writes and (abs(self.writes[-1][0]-stamp) > 1e-6 or receipt is not self.receipt):
            self.writes.clear()
            self._close()

    def note_sent(self, uid, pair, stamp, previous_sent, receipt=None):
        self._identity(uid)
        if self.writes and (abs(self.writes[-1][0]-previous_sent) > 1e-6
                            or not 0 < stamp-previous_sent <= .20):
            self.writes.clear()
            self._close()
        diff = pair[0]-pair[1]
        since = stamp
        if (self.writes and self.writes[-1][1]*diff > 0
                and abs(diff-self.writes[-1][1]) <= 2):
            since = self.writes[-1][2]
        self.writes.append((stamp, diff, since))
        self.receipt = receipt
        if min(pair) < 0 or sum(pair) <= 0 or diff == 0:
            self._close()

    @staticmethod
    def _finite(*values):
        return all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in values)

    @classmethod
    def evidence_valid(cls, intent, feedback, now, uid, *, permitted, hfov):
        if (intent is None or intent.target_id != uid or intent.mode != 'forward'
                or intent.bbox_quality != 'reliable' or not intent.image_error_only
                or not intent.response_boost_allowed or not permitted
                or intent.hold_zero or intent.park_requested or intent.near_distance_mode
                or intent.countersteer_rpm or intent.forward_countersteer
                or intent.outward_lead is not None
                or intent.braking_image_rate_dps is not None
                or intent.outward_continuity_rate_dps is not None
                or not cls._finite(now, intent.capture_timestamp, intent.visual_error_deg,
                                   intent.target_image_rate_dps, hfov)
                or not intent.valid(now) or intent.capture_frame_id <= 0
                or not 0 < intent.capture_timestamp <= now
                or now-intent.capture_timestamp > .25 or not 40 <= hfov <= 120):
            return False
        error, rate = intent.visual_error_deg, intent.target_image_rate_dps
        if (abs(error) < 16 or not 3 <= abs(rate) <= 60 or error*rate <= 0
                or abs(error) > hfov/2
                or abs(error)+abs(rate)*(now-intent.capture_timestamp+.5) < hfov/2-5):
            return False
        if feedback is None or not feedback.trustworthy:
            return False
        values = (feedback.timestamp, feedback.left_forward_rpm, feedback.right_forward_rpm,
                  feedback.raw_yaw_rate_right_dps, feedback.yaw_rate_right_dps,
                  feedback.left_read_started, feedback.left_read_finished,
                  feedback.right_read_started, feedback.right_read_finished)
        return bool(cls._finite(*values) and 0 <= now-feedback.timestamp <= .10
                    and min(feedback.left_forward_rpm, feedback.right_forward_rpm) >= 0
                    and max(abs(feedback.raw_yaw_rate_right_dps), abs(feedback.yaw_rate_right_dps)) <= 35
                    and abs(feedback.raw_yaw_rate_right_dps-feedback.yaw_rate_right_dps) <= 10
                    and 0 < feedback.left_read_started <= feedback.left_read_finished <= feedback.timestamp
                    and 0 < feedback.right_read_started <= feedback.right_read_finished <= feedback.timestamp)

    @classmethod
    def grant_survives_read(cls, grant, current, timing, *, uid, started, now, ttl, base):
        """Pure final bounds; never re-enter the potentially blocking reader.

        A read begun inside the normal 180ms window did not approve extended
        continuation. Crossing that boundary discards this packet; a later
        tick may independently run the ordinary continuation checks.
        """
        if (not isinstance(grant, tuple) or len(grant) != 4
                or not isinstance(current, tuple) or len(current) != 4
                or not cls._finite(started, now, ttl, base, grant[1], grant[2], grant[3], current[1])
                or grant[0] != 'forward' or grant[2] != uid
                or (current[0], current[2], current[3]) != (grant[0], grant[2], grant[3])
                or not 0 < grant[1] <= current[1] <= 100 or grant[3] <= 0
                or not 0 < ttl <= .25 or not started <= now
                or not 0 <= now-grant[3] <= ttl
                or started-grant[3] <= .18 < now-grant[3]):
            return False
        if timing is not None and getattr(timing, 'snapshot', None) == current:
            deadline = getattr(timing, 'depth_expires_at', None)
            ff_deadline = getattr(timing, 'feedforward_expires_at', None)
            if not cls._finite(deadline) or now > deadline:
                return False
            if ff_deadline is not None:
                if not cls._finite(ff_deadline):
                    return False
                if now >= ff_deadline:
                    percent = getattr(timing, 'distance_only_percent', None)
                    if not cls._finite(percent) or grant[1] > percent:
                        return False
        return True

    def adjust(self, base, yaw, intent, feedback, now, uid, *, permitted, hfov):
        self._identity(uid)

        def unchanged(reason):
            self._close()
            return base, yaw, reason

        new_capture = False
        if (intent is not None and intent.target_id == uid
                and self._finite(intent.capture_timestamp) and intent.capture_frame_id > 0):
            if (intent.capture_frame_id < self.capture_watermark
                    or (intent.capture_frame_id == self.capture_watermark
                        and intent.capture_timestamp != self.capture_time_watermark)
                    or (intent.capture_frame_id > self.capture_watermark
                        and intent.capture_timestamp <= self.capture_time_watermark)):
                return unchanged('view_old_capture')
            new_capture = intent.capture_frame_id > self.capture_watermark
            if new_capture:
                self.capture_watermark = intent.capture_frame_id
                self.capture_time_watermark = intent.capture_timestamp
        # Re-arm only after a NEW reliable central observation; repeated edge
        # frames, a zero write, or an expired grant cannot create another burst.
        if (intent is not None and intent.target_id == uid and intent.bbox_quality == 'reliable'
                and self._finite(intent.visual_error_deg, intent.capture_timestamp, now)
                and 0 <= now-intent.capture_timestamp <= .25 and intent.valid(now)
                and new_capture
                and abs(intent.visual_error_deg) < 12):
            self.captures.clear()
            self.deadline, self.spent, self.sign = None, False, 0
            self.lag_count = 0
        if not self.evidence_valid(intent, feedback, now, uid, permitted=permitted, hfov=hfov):
            # An invalid/unknown motion observation cannot be one of the three
            # samples that establish an outward trajectory.
            if new_capture:
                self.captures.clear()
            return unchanged('view_evidence_unqualified')
        if (not self._finite(base, yaw) or base <= abs(yaw) or not 3 <= abs(yaw) <= 10
                or yaw*intent.visual_error_deg <= 0):
            return unchanged('view_scope_veto')
        sign = 1 if yaw > 0 else -1
        cap = (intent.capture_frame_id, intent.capture_timestamp, abs(intent.visual_error_deg), sign)
        if self.captures and (cap[0] < self.captures[-1][0] or cap[1] < self.captures[-1][1]):
            return unchanged('view_old_capture')
        if self.captures and cap[0] == self.captures[-1][0] and cap[1] != self.captures[-1][1]:
            return unchanged('view_capture_timestamp_changed')
        if not self.captures or cap[0] > self.captures[-1][0]:
            if self.captures and (not 0 < cap[1]-self.captures[-1][1] <= .25
                    or sign != self.captures[-1][3] or cap[2] <= self.captures[-1][2]):
                self.captures.clear()
                self._close()
            self.captures.append(cap)
        if self.sign and self.sign != sign:
            return unchanged('view_direction_changed')
        if self.spent:
            return base, yaw, 'view_episode_complete'
        if self.deadline is not None and now >= self.deadline:
            return unchanged('view_timeout')
        reference = next((h for h in reversed(self.writes)
                          if h[0] < min(feedback.left_read_started, feedback.right_read_started)), None)
        if (reference is None or reference[1]*sign < 6
                or feedback.timestamp-reference[0] > .20
                or feedback.timestamp-reference[2] < .10):
            return unchanged('view_no_causal_write')
        # A newer command issued DURING the non-simultaneous wheel reads
        # invalidates the pair as differential-response evidence.
        if any(reference[0] < h[0] <= feedback.timestamp for h in self.writes):
            return unchanged('view_read_straddled_write')
        threshold = .6*min(abs(reference[1]), abs(2*yaw))
        measured = (feedback.left_forward_rpm, feedback.right_forward_rpm)
        if (measured[0]-measured[1])*sign >= threshold:
            return unchanged('view_turn_established')
        if feedback.timestamp > self.feedback_stamp:
            self.lag_count = self.lag_count+1 if 0 < feedback.timestamp-self.feedback_stamp <= .15 else 1
            self.feedback_stamp = feedback.timestamp
        if len(self.captures) < 3:
            return base, yaw, 'view_wait_three_captures'
        if self.lag_count < 2:
            return base, yaw, 'view_wait_two_feedback'
        requested = (base+yaw, base-yaw)
        if any(r < m for r, m in zip(requested, measured)):
            return unchanged('view_deceleration_preserved')
        inner = measured[1] if sign > 0 else measured[0]
        adjusted = min(base, math.floor(inner+2.)+abs(yaw))
        if adjusted >= base:
            return base, yaw, 'view_no_common_acceleration'
        if self.deadline is None:
            self.deadline, self.sign = now+.35, sign
        return adjusted, yaw, 'view_common_acceleration_held'
