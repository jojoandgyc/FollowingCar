"""Nonblocking, feedback-qualified wheel reversal for visible following only."""
import math


def wheel_feedback_valid(feedback, now):
    """Same qualification at planning and the last nonzero write boundary."""
    return bool(feedback is not None and feedback.trustworthy
        and not getattr(feedback, "left_error", 0) and not getattr(feedback, "right_error", 0)
        and math.isfinite(feedback.timestamp) and 0 <= now-feedback.timestamp <= .15
        and all(math.isfinite(v) for v in (feedback.left_forward_rpm, feedback.right_forward_rpm)))


def bounded_turn_residual(requested, measured, bound, *, full_reverse=False):
    """Qualified low opposing motion, not an all-wheel standstill requirement.

    Callers validate feedback freshness/trust separately. A correctly turning
    outer wheel may keep moving (or decelerate toward its current target).
    Limit its overspeed as well; this is not permission to ignore fast motion.
    Preserve the existing whole-car reverse rule outside the small residual
    envelope. This predicate alone NEVER authorizes a motor write.
    """
    if requested[0] * requested[1] >= 0 or bound <= 0:
        return False
    bound = min(bound, min(abs(v) for v in requested))
    if max(abs(v) for v in measured) <= bound:
        return True
    if full_reverse or all(v < -1.0 for v in measured):
        return False
    return all(-bound <= v * (1 if target > 0 else -1) <= abs(target) + bound
               for v, target in zip(measured, requested))


class WheelZeroCrossGuard:
    def __init__(self):
        self.last_output = (0, 0)
        self.last_sent = 0.0
        self.zero_since = None
        self.pending_signs = None
        self.pending_wheels = ()
        self.pending_full_reverse = False
        self.started = 0.0
        self.quiet_count = 0
        self.feedback_stamp = 0.0
        self.aligned_count = 0
        self.decelerating_count = 0
        self.resume_signs = None
        self.resume_at = 0.0
        self.commanded_reverse = False
        self.residual_stamp = 0.0
        self.residual_speeds = None
        self.residual_count = 0
        self.residual_forward_until = 0.0
        self.residual_turn_count = 0
        self.residual_turn_signs = None
        self.residual_turn_until = 0.0

    def reset(self):
        self.__init__()

    def note_sent(self, pair, now):
        # Encoder and command streams are asynchronous. A repeated zero must
        # not move the start of the stop episode past every new feedback.
        # Nonzero writes retire this reference; old pre-stop samples still
        # cannot authorize residual-motion handoff.
        if pair == (0, 0):
            if self.zero_since is None:
                self.zero_since = now
        else:
            self.zero_since = None
        self.last_output = pair
        self.last_sent = now
        if all(v < 0 for v in pair):
            self.commanded_reverse = True
        elif all(v > 0 for v in pair):
            self.commanded_reverse = False

    def limit(self, requested, feedback, now, *, allow_forward_handoff=False,
              allow_aligned_turn=False, residual_reverse_max_rpm=0.0,
              residual_turn_max_rpm=0.0, aligned_deceleration_max_rpm=0.0,
              preserve_wait_on_zero=False):
        signs = tuple(1 if v > 0 else -1 if v < 0 else 0 for v in requested)
        if not any(signs):
            if preserve_wait_on_zero:
                # Another execution guard is braking the SAME live turn.
                # Keep its episode/provenance, not a cached motor command.
                # No sample is counted and this call still only emits zero.
                self.quiet_count = self.aligned_count = self.decelerating_count = 0
                self.residual_count = self.residual_turn_count = 0
                self.resume_signs = None
                self.residual_turn_signs = None
                self.residual_turn_until = self.residual_forward_until = 0.
                return (0, 0), "upstream_zero_wait"
            commanded_reverse = self.commanded_reverse
            self.reset()
            self.commanded_reverse = commanded_reverse
            return (0, 0), "explicit_zero"
        if not wheel_feedback_valid(feedback, now):
            self.quiet_count = 0
            self.aligned_count = 0
            self.decelerating_count = 0
            self.resume_signs = None
            self.residual_count = 0
            self.residual_forward_until = 0.0
            self.residual_turn_count = 0
            self.residual_turn_signs = None
            return (0, 0), "feedback_unavailable"
        measured = (feedback.left_forward_rpm, feedback.right_forward_rpm)
        turn_bound = max(0.0, min(4.0, float(residual_turn_max_rpm)))
        if self.residual_turn_signs is not None:
            if (allow_aligned_turn and turn_bound > 0 and signs == self.residual_turn_signs
                    and now <= self.residual_turn_until
                    and all(v * sign >= -turn_bound for v, sign in zip(measured, signs))):
                # Fixed response interval, not a command lease. Caller must
                # revalidate the CURRENT yaw/UID every write. Never replay yaw.
                return tuple(requested), "cross_bounded_residual_turn_continue"
            self.residual_turn_signs = None
        # A bounded residual roll is not a request to keep reversing. After
        # an actual zero write, two DISTINCT, fresh, non-worsening samples may
        # hand a newly approved forward curve to the motor controller. Never
        # use this for commanded reverse, high reverse speed or yaw-only/search.
        bound = max(0.0, min(8.0, float(residual_reverse_max_rpm)))
        if self.residual_forward_until > 0:
            continuing = bool(
                now <= self.residual_forward_until and bound > 0
                and allow_forward_handoff and all(v > 0 for v in requested)
                and not self.commanded_reverse and self.residual_speeds is not None
                and all(v >= -bound and min(v, 0) >= min(old, 0)
                        for v, old in zip(measured, self.residual_speeds))
            )
            if continuing:
                if feedback.timestamp > self.residual_stamp:
                    self.residual_stamp = feedback.timestamp
                    self.residual_speeds = measured
                # Fixed deadline, never renewed by a zero/old/new sample.
                # The caller still validates the CURRENT forward grant.
                return tuple(requested), "cross_bounded_residual_forward_continue"
            self.residual_forward_until = 0.0
        residual_candidate = bool(
            bound > 0 and allow_forward_handoff and all(v > 0 for v in requested)
            and not self.commanded_reverse and self.last_output == (0, 0)
            and self.zero_since is not None and self.zero_since <= feedback.timestamp <= now
            and now - self.last_sent <= .25
            and all(-bound <= v for v in measured)
            and (any(v < -1 for v in measured) or self.pending_full_reverse)
        )
        if not residual_candidate:
            self.residual_count = 0
            self.residual_speeds = None
        elif feedback.timestamp > self.residual_stamp:
            non_worsening = bool(
                self.residual_speeds is not None
                and feedback.timestamp - self.residual_stamp <= .15
                and all(min(v, 0) >= min(old, 0)
                        for v, old in zip(measured, self.residual_speeds))
            )
            self.residual_count = self.residual_count + 1 if non_worsening else 1
            self.residual_stamp = feedback.timestamp
            self.residual_speeds = measured
        if residual_candidate and self.residual_count >= 2:
            self.pending_signs = None
            self.pending_wheels = ()
            self.pending_full_reverse = False
            self.quiet_count = self.aligned_count = 0
            self.resume_signs = None
            self.residual_count = 0
            self.residual_forward_until = now + .15
            return tuple(requested), "cross_bounded_residual_forward_handoff"
        full_reverse = (all(v < -1.0 for v in measured)
                        or all(v < 0 for v in self.last_output))
        # A Depth base may alternate between zero and positive while vision
        # keeps asking for the same turn. Once its inner wheel is braking for
        # reversal, preserve confirmation unless a fresh qualified forward
        # request and new non-opposing feedback supersede it below.
        # Straight/opposite yaw cancels this episode; ordinary forward with
        # no pending turn uses the established controller handoff.
        same_yaw_cross = bool(
            self.pending_signs is not None
            and self.pending_signs[0] * self.pending_signs[1] < 0
            and not self.pending_full_reverse and not full_reverse
            and sum(requested) >= 0
            and (requested[0] - requested[1])
            * (self.pending_signs[0] - self.pending_signs[1]) > 0
        )
        if signs != self.pending_signs and not same_yaw_cross:
            self.pending_signs = None
            self.pending_wheels = ()
            self.quiet_count = 0
            self.aligned_count = 0
            self.decelerating_count = 0
            self.residual_turn_count = 0
        if signs != self.resume_signs:
            self.resume_signs = None
        opposing = tuple(i for i in range(2) if signs[i] and (
            measured[i]*signs[i] < -1.0
            or (self.last_output[i]*signs[i] < 0 and feedback.timestamp <= self.last_sent)
        ))
        # A pending in-place turn is not an instruction to finish a reversal
        # after Depth has replaced it with a forward curve. With fresh forward
        # authority and non-opposing feedback, neither CURRENT wheel target
        # needs to cross zero. Cancel only that obsolete mixed-sign episode.
        # Real reverse feedback, whole-car reverse provenance, search/opt-out
        # and invalid feedback must still take the original guarded path.
        forward_pair = all(v >= 0 for v in requested) and any(v > 0 for v in requested)
        if (same_yaw_cross and allow_forward_handoff and forward_pair
                and not opposing and feedback.timestamp > self.started):
            self.pending_signs = None
            self.pending_wheels = ()
            self.pending_full_reverse = False
            self.quiet_count = self.aligned_count = 0
            self.resume_signs = None
            return tuple(requested), "cross_obsolete_turn_forward_handoff"
        # A fresh, visible forward grant may hand residual one-wheel rotation
        # directly to the motor controller. Do not inject a zero confirmation
        # or a second 5RPM software launch ramp. Search, reverse requests,
        # whole-car reverse transitions and invalid feedback remain guarded.
        if (allow_forward_handoff and forward_pair and not full_reverse
                and not self.pending_full_reverse and not same_yaw_cross):
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
            # CAP1158/1163: already aligned overspeed is not a reversal.
            # Two post-zero samples establish direction; release only when
            # the CURRENT zero-mean command decelerates BOTH wheels. This
            # never licenses ignoring the +12/+36 opposing inner-wheel data.
            decel_limit = min(200., max(0., float(aligned_deceleration_max_rpm)))
            decel_sample = bool(
                allow_aligned_turn and decel_limit > 0
                and signs == self.pending_signs and signs[0]*signs[1] < 0
                and sum(requested) == 0 and max(map(abs, requested)) <= 10
                and not self.pending_full_reverse and not self.commanded_reverse
                and self.last_output == (0, 0) and self.zero_since is not None
                and feedback.timestamp > max(self.zero_since, self.started)
                and not getattr(feedback, "left_error", 0)
                and not getattr(feedback, "right_error", 0)
                and all(0 < v*sign <= decel_limit for v, sign in zip(measured, signs)))
            if feedback.timestamp > self.feedback_stamp:
                feedback_gap = feedback.timestamp - self.feedback_stamp
                self.feedback_stamp = feedback.timestamp
                quiet = all(abs(measured[i]) <= 1.0 for i in self.pending_wheels)
                self.quiet_count = (
                    (self.quiet_count + 1 if feedback_gap <= .15 else 1)
                    if quiet else 0
                )
                # Normal forward motion need not pass through exact zero if
                # TWO distinct new feedback samples prove both wheels have
                # already crossed to the requested side. Rotation keeps the
                # quiet-wheel rule unless explicitly enabled below.
                forward_aligned = (self.pending_signs == signs == (1, 1) and not opposing
                                   and all(0 <= measured[i] <= requested[i] for i in range(2)))
                # A wheel can cross zero BETWEEN feedback samples. Requiring
                # two samples exactly at zero would then wait indefinitely.
                # Only the CURRENT mixed-sign request qualifies, on TWO new
                # samples with both wheels aligned and no overspeed. Search
                # and opt-out retain the old quiet-wheel rule.
                turn_aligned = bool(
                    allow_aligned_turn and self.pending_signs == signs
                    and signs[0] * signs[1] < 0 and not opposing
                    and all(0 <= measured[i] * signs[i] <= abs(requested[i])
                            for i in range(2))
                )
                aligned = forward_aligned or turn_aligned
                self.aligned_count = (self.aligned_count + 1 if feedback_gap <= .15 else 1) if aligned else 0
                self.decelerating_count = (
                    (self.decelerating_count + 1 if feedback_gap <= .15 else 1)
                    if decel_sample else 0)
                residual_bound = min(4.0, max(0.0, float(residual_turn_max_rpm)),
                                     min(abs(v) for v in requested))
                low_turn = bool(allow_aligned_turn and signs == self.pending_signs
                    and signs[0] * signs[1] < 0 and residual_bound > 0
                    and self.last_output == (0, 0) and self.zero_since is not None
                    and feedback.timestamp >= self.zero_since
                    and bounded_turn_residual(requested, measured, residual_bound,
                                              full_reverse=self.pending_full_reverse))
                self.residual_turn_count = (
                    (self.residual_turn_count + 1 if feedback_gap <= .15 else 1)
                    if low_turn else 0)
            elif feedback.timestamp < self.feedback_stamp or not decel_sample:
                self.decelerating_count = 0
            if (self.decelerating_count >= 2 and decel_sample
                    and all(abs(target) <= abs(actual)
                            for target, actual in zip(requested, measured))):
                self.pending_signs = None
                self.pending_wheels = ()
                self.pending_full_reverse = False
                self.quiet_count = self.aligned_count = self.decelerating_count = 0
                self.residual_turn_count = 0
                self.resume_signs = None
                return tuple(requested), "cross_aligned_decelerating_turn_handoff"
            if self.residual_turn_count >= 2:
                self.pending_signs = None
                self.pending_wheels = ()
                self.pending_full_reverse = False
                self.quiet_count = self.aligned_count = self.residual_turn_count = 0
                self.resume_signs = None
                self.residual_turn_signs = signs
                self.residual_turn_until = now + .5
                return tuple(requested), "cross_bounded_residual_turn_handoff"
            aligned_release = self.quiet_count < 2 and self.aligned_count >= 2
            if self.quiet_count < 2 and not aligned_release:
                # Zero both wheels: zeroing only the reversing wheel could
                # introduce unapproved translation or excessive wheel delta.
                return (0, 0), "cross_timeout_zero" if now-self.started >= .25 else "cross_wait_zero"
            self.pending_signs = None
            self.pending_wheels = ()
            self.quiet_count = 0
            self.aligned_count = 0
            if aligned_release and allow_aligned_turn and signs[0] * signs[1] < 0:
                self.pending_full_reverse = False
                self.resume_signs = None
                return tuple(requested), "cross_confirmed_turn_handoff"
            if aligned_release and allow_forward_handoff and forward_pair:
                # TWO fresh samples have already proved that both wheels are
                # on the requested side. The caller has revalidated Depth and
                # the braking cap. Do not add a second 5RPM/80RPM/s launch to
                # the controller's approved forward request. Real reverse,
                # mixed-sign rotation and the opt-out path stay guarded.
                self.pending_full_reverse = False
                self.resume_signs = None
                return tuple(requested), "cross_confirmed_forward_handoff"
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
