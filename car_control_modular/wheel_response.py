"""Passive, timestamp-aligned differential response diagnostics (no I/O)."""
from collections import deque
import math


class WheelDifferentialResponse:
    def __init__(self):
        self.history = deque(maxlen=32)
        self.uid = None
        self.feedback_ts = 0.0

    def observe_and_note(self, uid, pair, sent_at, previous_sent, feedback,
                         *, packet_written=True):
        # A writer/guard reset or another motion mode invalidates our history.
        if (uid != self.uid or not self.history
                or abs(self.history[-1][0] - previous_sent) > 1e-6
                or sent_at - previous_sent > 0.20):
            self.history.clear()
            self.feedback_ts = 0.0
        self.uid = uid
        reference = None
        lagging = False
        elapsed = None
        measured = None
        fresh = bool(feedback is not None and feedback.trustworthy
                     and math.isfinite(feedback.timestamp)
                     and 0 <= sent_at - feedback.timestamp <= .15
                     and all(math.isfinite(v) for v in
                             (feedback.left_forward_rpm, feedback.right_forward_rpm)))
        new_sample = bool(fresh and feedback.timestamp > self.feedback_ts)
        if new_sample:
            self.feedback_ts = feedback.timestamp
            eligible = [h for h in self.history if h[0] <= feedback.timestamp]
            if eligible:
                reference = eligible[-1]
                measured = feedback.left_forward_rpm - feedback.right_forward_rpm
                elapsed = max(0.0, feedback.timestamp - reference[2])
                # Commissioning observation: motors may need at least 500ms
                # to build speed. Earlier samples are ramp diagnostics only.
                lagging = bool(abs(reference[1]) >= 6 and elapsed >= .50
                               and measured * (1 if reference[1] > 0 else -1) < .5 * abs(reference[1]))
        diff = pair[0] - pair[1]
        since = sent_at
        if self.history:
            last = self.history[-1]
            if diff * last[1] > 0 and abs(diff - last[1]) <= 2:
                since = last[2]
        if packet_written:
            self.history.append((sent_at, diff, since))
        return {
            "reference_diff": None if reference is None else reference[1],
            "reference_sent": None if reference is None else reference[0],
            "measured_diff": measured,
            "sustained_ms": None if elapsed is None else round(elapsed * 1000., 1),
            "new_feedback": new_sample,
            "lagging": lagging,
        }
