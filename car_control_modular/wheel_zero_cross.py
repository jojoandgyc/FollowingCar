"""Nonblocking, feedback-qualified wheel reversal for visible following only."""
import math


class WheelZeroCrossGuard:
    def __init__(self):
        self.last_output = (0, 0)
        self.last_sent = 0.0
        self.pending_signs = None
        self.pending_wheels = ()
        self.pending_full_reverse = False
        self.started = 0.0
        self.quiet_count = 0
        self.feedback_stamp = 0.0
        self.aligned_count = 0
        self.resume_signs = None
        self.resume_at = 0.0

    def reset(self):
        self.__init__()

    def note_sent(self, pair, now):
        self.last_output = pair
        self.last_sent = now

    def limit(self, requested, feedback, now, *, allow_forward_handoff=False):
        signs = tuple(1 if v > 0 else -1 if v < 0 else 0 for v in requested)
        if not any(signs):
            self.reset()
            return (0, 0), "explicit_zero"
        if (feedback is None or not feedback.trustworthy
                or not math.isfinite(feedback.timestamp)
                or not 0 <= now-feedback.timestamp <= .15
                or not all(math.isfinite(v) for v in
                           (feedback.left_forward_rpm, feedback.right_forward_rpm))):
            self.quiet_count = 0
            self.aligned_count = 0
            self.resume_signs = None
            return (0, 0), "feedback_unavailable"
        measured = (feedback.left_forward_rpm, feedback.right_forward_rpm)
        if signs != self.pending_signs:
            self.pending_signs = None
            self.pending_wheels = ()
            self.quiet_count = 0
            self.aligned_count = 0
        if signs != self.resume_signs:
            self.resume_signs = None
        opposing = tuple(i for i in range(2) if signs[i] and (
            measured[i]*signs[i] < -1.0
            or (self.last_output[i]*signs[i] < 0 and feedback.timestamp <= self.last_sent)
        ))
        full_reverse = (all(v < -1.0 for v in measured)
                        or all(v < 0 for v in self.last_output))
        # A fresh, visible forward grant may hand residual one-wheel rotation
        # directly to the motor controller. Do not inject a zero confirmation
        # or a second 5RPM software launch ramp. Search, reverse requests,
        # whole-car reverse transitions and invalid feedback remain guarded.
        forward_pair = all(v >= 0 for v in requested) and any(v > 0 for v in requested)
        if (allow_forward_handoff and forward_pair and not full_reverse
                and not self.pending_full_reverse):
            handed_off = bool(opposing or self.pending_signs or self.resume_signs)
            self.pending_signs = None
            self.pending_wheels = ()
            self.quiet_count = self.aligned_count = 0
            self.resume_signs = None
            return tuple(requested), "forward_controller_handoff" if handed_off else "continuous"
        if (opposing or (self.pending_full_reverse and forward_pair
                         and self.resume_signs is None)) and self.pending_signs is None:
            self.resume_signs = None
            self.pending_signs = signs
            # Preserve whole-car reverse provenance if the requested curve
            # changes (e.g. one requested wheel becomes zero) during a wait.
            self.pending_full_reverse = self.pending_full_reverse or full_reverse
            self.pending_wheels = (0, 1) if self.pending_full_reverse else opposing
            self.started = now
            self.feedback_stamp = feedback.timestamp
        elif opposing and not set(opposing).issubset(self.pending_wheels):
            self.pending_wheels = tuple(sorted(set(self.pending_wheels) | set(opposing)))
            self.quiet_count = 0
            self.aligned_count = 0
        if self.pending_signs is not None and full_reverse:
            self.pending_full_reverse = True
            if self.pending_wheels != (0, 1):
                self.pending_wheels = (0, 1)
                self.quiet_count = self.aligned_count = 0
        if self.pending_signs is not None:
            if feedback.timestamp > self.feedback_stamp:
                feedback_gap = feedback.timestamp - self.feedback_stamp
                self.feedback_stamp = feedback.timestamp
                quiet = all(abs(measured[i]) <= 1.0 for i in self.pending_wheels)
                self.quiet_count = self.quiet_count + 1 if quiet else 0
                # Normal forward motion need not pass through exact zero if
                # TWO distinct new feedback samples prove both wheels have
                # already crossed to the requested side. Reversal/rotation
                # still uses the original quiet-wheel rule.
                aligned = (signs == (1, 1) and not opposing
                           and all(0 <= measured[i] <= requested[i] for i in range(2)))
                self.aligned_count = (self.aligned_count + 1 if feedback_gap <= .15 else 1) if aligned else 0
            aligned_release = self.quiet_count < 2 and self.aligned_count >= 2
            if self.quiet_count < 2 and not aligned_release:
                # Zero both wheels: zeroing only the reversing wheel could
                # introduce unapproved translation or excessive wheel delta.
                return (0, 0), "cross_timeout_zero" if now-self.started >= .25 else "cross_wait_zero"
            self.pending_signs = None
            self.pending_wheels = ()
            self.quiet_count = 0
            self.aligned_count = 0
            if aligned_release:
                self.resume_signs = signs
                self.resume_at = now
                scale = min(1.0, 5.0 / max(abs(v) for v in requested))
                return tuple(int(v * scale) for v in requested), "cross_aligned_resume"
            self.pending_full_reverse = False
        if self.resume_signs is not None:
            # Applied (not merely requested) wheel speeds seed this ramp.
            # Scale both wheels together to preserve the requested curvature.
            step = 80.0 * max(0.0, min(.10, now-self.resume_at))
            scale = min([1.0] + [(abs(self.last_output[i])+step)/abs(v)
                                 for i,v in enumerate(requested) if v])
            self.resume_at = now
            if scale < 1.0:
                return tuple(int(v * scale) for v in requested), "cross_resume_ramp"
            self.resume_signs = None
            self.pending_full_reverse = False
        return tuple(requested), "continuous"
