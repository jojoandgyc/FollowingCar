"""One-shot differential buildup assistance; never a motion authority.

All times are monotonic. Motor writer must revalidate the current axes, identity,
braking policy and depth lease after this calculation. No command queue or sleep.
"""
import math


class TurnResponseAssist:
    def __init__(self):
        self.uid = None
        self.sign = 0
        self.start_cap = 0
        self.blocked_cap = 0
        self.started = 0.
        self.closed = False
        self.feedback_stamp = 0.
        self.established_count = 0

    def adjust(self, base, yaw, intent, feedback, now, uid, *, gain=1.4,
               max_yaw=20., duration=.35, boost_permitted=True):
        if uid != self.uid:
            self.__init__()
            self.uid = uid
        valid = bool(intent is not None and intent.target_id == uid
            and intent.bbox_quality == 'reliable' and intent.mode == 'forward'
            and intent.valid(now) and 0 < intent.capture_timestamp <= now
            and now-intent.capture_timestamp <= .25
            and intent.capture_frame_id > 0)
        error = getattr(intent, 'visual_error_deg', None)
        if not valid or error is None or not math.isfinite(error):
            self.closed = True
            return base, yaw, 'visual_unqualified'
        cap = intent.capture_frame_id
        if feedback is None or not feedback.trustworthy or not all(math.isfinite(v) for v in (
                now, feedback.timestamp, feedback.left_forward_rpm, feedback.right_forward_rpm
        )) or not 0 <= now-feedback.timestamp <= .10:
            self.closed = True
            return base, yaw, 'feedback_unqualified'
        left, right = feedback.left_forward_rpm, feedback.right_forward_rpm
        if yaw == 0:
            # Only a NEW frame can start another episode after a zero. A
            # fast-loop coast/restart of the same image cannot rearm a boost.
            self.sign = 0
            self.blocked_cap = max(self.blocked_cap, cap)
            self.closed = True
            # Residual differential is a yaw problem, not a new longitudinal
            # speed limit. Depth braking and explicit parking remain separate.
            return base, yaw, 'zero_yaw'
        sign = 1 if yaw > 0 else -1
        rate = intent.target_image_rate_dps
        approaching = bool(rate is not None and (not math.isfinite(rate)
            or abs(error)-max(0., -rate*sign)*(now-intent.capture_timestamp+.18) <= 5))
        if (base <= 0 or intent.hold_zero or intent.park_requested or intent.near_distance_mode
                or not boost_permitted or not intent.response_boost_allowed or abs(error) < 6 or error*yaw <= 0
                or approaching or intent.correction_limit_rpm < 15):
            self.closed = True
            self.blocked_cap = max(self.blocked_cap, cap)
            return base, yaw, 'braking_or_scope_veto'
        if self.sign != sign:
            if cap <= max(self.blocked_cap, self.start_cap):
                return base, yaw, 'new_frame_required'
            self.sign, self.start_cap, self.started = sign, cap, now
            self.closed = False
            self.feedback_stamp = 0.
            self.established_count = 0
        if self.closed:
            return base, yaw, 'episode_complete'
        if now >= self.started + min(.35, max(0., duration)):
            self.closed = True
            return base, yaw, 'boost_timeout'
        # Opposite/both-reverse motion still goes through the existing guard;
        # never add demand to a reverse-wheel request or make one from forward.
        if min(left, right) < 0 or base <= abs(yaw):
            self.closed = True
            return base, yaw, 'reverse_guard_scope'
        measured_diff = (left-right)*sign
        established = measured_diff >= max(6., min(10., abs(yaw)))
        if feedback.timestamp > self.feedback_stamp:
            continuous = feedback.timestamp-self.feedback_stamp <= .15
            self.established_count = (self.established_count+1 if continuous else 1) if established else 0
            self.feedback_stamp = feedback.timestamp
        if established:
            # Even the first established sample suppresses the extra demand;
            # two distinct samples latch completion so feedback jitter cannot
            # repeatedly restart the same episode.
            if self.established_count >= 2:
                self.closed = True
            return base, yaw, 'turn_established'
        enhanced = min(20., max(0., max_yaw), abs(yaw)*min(1.4, max(1., gain)), base)
        enhanced = max(abs(yaw), math.floor(enhanced))
        return base, sign*enhanced, 'turn_build_boost' if enhanced > abs(yaw) else 'no_boost_headroom'
