"""Decelerate a forward curve before requesting an in-place wheel reversal.

This state owns no motor authority. It can only substitute zero for a yaw-only
request. Fresh forward requests immediately leave it and still pass through
the normal wheel reversal guard. Only actually written forward packets arm it.
"""
import math

from .wheel_zero_cross import bounded_turn_residual


class ForwardLossHandoff:
    def __init__(self):
        self.reset()

    def reset(self):
        self.armed = False
        self.started = None
        self.feedback_stamp = None
        self.quiet_count = 0
        self.zero_sent_at = None
        self.quiet_mode = None
        self.previous_outer = None

    def note_sent(self, pair, now=None):
        if pair == (0, 0) and now is not None and self.zero_sent_at is None:
            self.zero_sent_at = now
        if sum(pair) > 0:
            # Only an ACTUALLY WRITTEN forward command ends this wait. A new
            # forward request can itself be blocked by the wheel guard.
            self.started = None
            self.feedback_stamp = None
            self.quiet_count = 0
            self.zero_sent_at = None
            self.quiet_mode = None
            self.previous_outer = None
        if min(pair) >= 0 and sum(pair) > 0:
            self.armed = True
        elif sum(pair) < 0:
            self.reset()

    def limit(self, pair, feedback, now, *, residual_turn_max_rpm=0.0,
              allow_outer_deceleration=False):
        if sum(pair) != 0:
            if sum(pair) < 0:
                self.reset()
            return pair, None
        if not self.armed:
            return pair, None
        if self.started is None:
            self.started = now
        values = (() if feedback is None else (
            feedback.timestamp, feedback.left_forward_rpm, feedback.right_forward_rpm))
        valid = bool(values and feedback.trustworthy and all(math.isfinite(v) for v in values)
                     and 0 <= now-feedback.timestamp <= .15)
        if not valid:
            self.quiet_count = 0
            return (0, 0), "forward_loss_feedback_wait"
        # CAP347: only the opposing wheel needs a low residual speed. Do not
        # first stop the correctly moving outer wheel and then start another
        # confirmation episode. A fresh pre-handoff sample may DELEGATE, but
        # the wheel guard still requires two new samples AFTER an actual zero
        # write before accepting a bounded opposing residual.
        # No positive translation is manufactured when Depth has expired.
        bound = max(0.0, min(4.0, float(residual_turn_max_rpm)))
        if bounded_turn_residual(pair, (feedback.left_forward_rpm,
                                        feedback.right_forward_rpm), bound):
            self.reset()
            return pair, "forward_loss_residual_turn_guard"
        if feedback.timestamp < self.started:
            self.quiet_count = 0
            return (0, 0), "forward_loss_feedback_wait"
        measured = (feedback.left_forward_rpm, feedback.right_forward_rpm)
        inner = 0 if pair[0] < 0 else 1
        outer = 1-inner
        # CAP523: braking the correctly moving outer wheel to an all-wheel
        # standstill is unnecessary once the reversing wheel is quiet. This
        # still emits ONLY the requested zero-mean pivot, never a rolling arc
        # or positive translation without a Depth grant. Require actual zero
        # I/O followed by two independent, non-accelerating encoder samples.
        decelerating = bool(allow_outer_deceleration and pair[0]*pair[1] < 0
            and max(abs(v) for v in pair) <= 10
            and self.zero_sent_at is not None and feedback.timestamp > self.zero_sent_at
            and abs(measured[inner]) <= 1.0
            and abs(pair[outer]) < measured[outer] <= 200
            and not getattr(feedback, "left_error", 0)
            and not getattr(feedback, "right_error", 0))
        mode = inner if decelerating else "both"
        if mode != self.quiet_mode:
            self.quiet_count = 0
            self.previous_outer = None
            self.quiet_mode = mode
        if self.feedback_stamp is None or feedback.timestamp > self.feedback_stamp:
            continuous = self.feedback_stamp is not None and feedback.timestamp-self.feedback_stamp <= .15
            quiet = max(map(abs, measured)) <= 1.
            if decelerating:
                quiet = self.previous_outer is None or measured[outer] <= self.previous_outer + 1.
                self.previous_outer = measured[outer]
            self.quiet_count = (self.quiet_count+1 if continuous else 1) if quiet else 0
            self.feedback_stamp = feedback.timestamp
        if self.quiet_count < 2:
            return (0, 0), "forward_loss_decelerating"
        self.reset()
        return pair, ("forward_loss_inner_stopped_turn_ready" if decelerating
                      else "forward_loss_stopped_turn_ready")
