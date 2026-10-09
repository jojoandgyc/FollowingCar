"""Executed-wheel history and bounded yaw braking. Never longitudinal authority."""
from collections import deque
import math


def forward_brake_allowed(base, yaw, intent, feedback, now, uid):
    if (intent is None or intent.target_id != uid or intent.mode != "forward"
            or intent.bbox_quality != "reliable" or intent.hold_zero
            or not intent.valid(now) or not 0 < intent.capture_timestamp <= now
            or now-intent.capture_timestamp > .25 or intent.capture_frame_id <= 0
            or not 0 < abs(yaw) <= min(6, base)
            or feedback is None or not feedback.trustworthy):
        return False
    error = intent.x_ratio-.5
    rate = intent.target_image_rate_dps
    raw = getattr(feedback, "raw_yaw_rate_right_dps", None)
    vals = (base, yaw, error, rate, now, feedback.timestamp,
            feedback.left_forward_rpm, feedback.right_forward_rpm,
            feedback.yaw_rate_right_dps, raw)
    if any(v is None or not math.isfinite(v) for v in vals):
        return False
    return bool(0 <= now-feedback.timestamp <= .1
        and min(feedback.left_forward_rpm, feedback.right_forward_rpm) >= 0
        and error*rate < 0 and abs(rate) <= 120
        and error*yaw < 0 and yaw*raw < 0
        and (feedback.left_forward_rpm-feedback.right_forward_rpm)*yaw < 0
        and feedback.yaw_rate_right_dps*raw > 0
        and abs(feedback.yaw_rate_right_dps-raw) <= 10 and 4 <= abs(raw) <= 90
        and (abs(raw) <= 35 or getattr(feedback, "yaw_rate_confirmed", False)))


class TurnResponseTrial:
    def __init__(self):
        self.uid = None
        self.history = deque(maxlen=32)
        self.cap = None
        self.started = None
        self.quiet_stamp = 0.
        self.quiet_count = 0
        self.brake_started = None

    def _identity(self, uid):
        if uid != self.uid:
            self.__init__()
            self.uid = uid

    def record(self, uid, pair, sent_at):
        """Only call after successful hardware write; never record requests."""
        self._identity(uid)
        self.history.append((sent_at, .5*sum(pair), .5*(pair[0]-pair[1])))

    def adjust(self, base, yaw, intent, feedback, now, uid, horizon):
        self._identity(uid)
        horizon = max(0., min(.5, horizon))
        if not horizon:
            self.__init__()
            return base, yaw, "disabled"
        while self.history and now-self.history[0][0] > horizon:
            self.history.popleft()
        # No old forward brake is allowed to become a pivot on depth loss.
        opposite = bool(intent is not None and intent.mode == "forward"
                        and yaw*(intent.x_ratio-.5) < 0)
        braking = opposite or bool(getattr(intent, "forward_countersteer", False))
        if braking and not forward_brake_allowed(base, yaw, intent, feedback, now, uid):
            yaw = 0.
        if braking and yaw:
            if self.brake_started is None:
                self.brake_started = now
            if now-self.brake_started >= .08:
                yaw = 0.  # New captures cannot extend this braking episode.
        if base <= 0:
            return base, yaw, "no_translation"
        # Old trial state can never cap the newly authorized forward axis.
        # Only the short evidence-checked countersteer pulse belongs here.
        self.cap = self.started = None
        self.quiet_stamp, self.quiet_count = 0., 0
        if (not braking and self.brake_started is not None
                and now-self.brake_started >= horizon):
            self.brake_started = None
        if braking and yaw:
            return base, yaw, "forward_brake"
        return base, yaw, "tracking"
