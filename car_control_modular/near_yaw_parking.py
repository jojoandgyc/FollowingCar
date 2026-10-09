"""Explicit normal parking intent, separate from zero longitudinal speed.

No motor I/O or independent timeout extends movement authority here.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NearYawParkRequest:
    uid: int
    capture_frame_id: int
    capture_timestamp: float
    requested_at: float
    reason: str
    countersteer_rpm: int = 0
    visual_error_deg: float = 0.0
    image_rate_dps: float = 0.0
    countersteer_until: float = 0.0


class ParkSettlingEvidence:
    """Post-write encoder evidence, never a timer-based motion permission."""

    MIN_HOLD_SEC = .500

    def __init__(self, request, sent_at, *, require_current_release=False,
                 allow_early_quiet_release=False):
        self.request = request
        self.sent_at = float(sent_at)
        self.last_sample = 0.0
        self.quiet_count = 0
        self.quiet_since = None
        self.ready_at = None
        self.ready_yaw_right_deg = None
        self.reason = "await_post_stop_feedback"
        self.require_current_release = require_current_release
        self.current_released_at = None
        self.fault = None
        self.forward_resume_until = 0.0
        # Opt-in only for a normal search stop preceded by fresh low-speed
        # feedback. This changes current-hold dwell, never motion permission.
        self.allow_early_quiet_release = bool(allow_early_quiet_release)
        self.early_quiet_release_completed = False
        self.pre_release_quiet_at = None

    def request_forward_resume(self, capture_timestamp, sample_timestamp, now):
        """Qualified producer hint to release current, never motor authority."""
        if (self.fault or not all(isinstance(v, (int, float)) and math.isfinite(v)
                for v in (capture_timestamp, sample_timestamp, now))
                or not self.sent_at < capture_timestamp <= now
                or not self.sent_at < sample_timestamp <= now
                or now-capture_timestamp > .19 or now-sample_timestamp > .18):
            return False
        # No processing tick or replay extends either physical deadline.
        self.forward_resume_until = min(capture_timestamp+.19, sample_timestamp+.18)
        return True

    def forward_resume_live(self, now):
        return not self.fault and self.sent_at < now < self.forward_resume_until

    def quiet_current_release_ready(self, feedback, now):
        """Two distinct post-STOP samples may finish the opted-in dwell.

        A cached quiet result alone is insufficient: revalidate it on the
        current motor tick. Serial readback/FREE completion and new evidence
        after that completion are still required by the caller.
        """
        if (not self.allow_early_quiet_release or self.fault
                or self.current_released_at is not None):
            return False
        return self.observe(feedback, now)

    def mark_current_released(self, now, *, early_quiet=False):
        """Called after dual 0A readback and FREE STOP completion, never 0RPM."""
        self.early_quiet_release_completed = bool(
            early_quiet and self.allow_early_quiet_release and not self.fault
            and self.ready_at is not None)
        self.pre_release_quiet_at = self.ready_at
        self.current_released_at = float(now)
        self.last_sample = 0.0
        self.quiet_count = 0
        self.quiet_since = self.ready_at = self.ready_yaw_right_deg = None
        self.reason = "await_post_current_release_feedback"

    @property
    def feedback_boundary(self):
        return self.sent_at if self.current_released_at is None else self.current_released_at

    def recovery_gate(self, now):
        if self.fault:
            self.reason = self.fault
            return False
        if not self.minimum_hold_complete(now):
            return False
        if self.require_current_release and self.current_released_at is None:
            self.reason = "await_current_release"
            return False
        return True

    def minimum_hold_complete(self, now):
        # Count from completed stop I/O, not request/camera time. Do not
        # sleep in the executor: emergency handling must remain immediate.
        if (math.isfinite(now) and self.early_quiet_release_completed
                and self.current_released_at is not None
                and self.current_released_at <= now):
            return True
        if not math.isfinite(now) or now < self.sent_at + self.MIN_HOLD_SEC:
            self.reason = f"minimum_normal_hold_{self.MIN_HOLD_SEC * 1000:.0f}ms"
            return False
        return True

    def observe(self, feedback, now):
        if self.fault:
            self.reason = self.fault
            return False
        values = None if feedback is None else (
            feedback.timestamp, feedback.left_forward_rpm, feedback.right_forward_rpm)
        valid = bool(values is not None and feedback.trustworthy
                     and not getattr(feedback, "left_error", 0)
                     and not getattr(feedback, "right_error", 0)
                     and all(math.isfinite(float(v)) for v in values)
                     and self.feedback_boundary < feedback.timestamp <= now
                     and now - feedback.timestamp <= .15)
        if not valid:
            self.quiet_count = 0
            self.quiet_since = self.ready_at = self.ready_yaw_right_deg = None
            self.reason = "feedback_not_fresh_post_stop"
            return False
        if feedback.timestamp < self.last_sample:
            self.quiet_count = 0
            self.quiet_since = self.ready_at = self.ready_yaw_right_deg = None
            self.reason = "feedback_out_of_order"
            return False
        if max(abs(feedback.left_forward_rpm), abs(feedback.right_forward_rpm)) > 1.0:
            self.quiet_count = 0
            self.quiet_since = self.ready_at = self.ready_yaw_right_deg = None
            self.last_sample = feedback.timestamp
            self.reason = "wheels_still_moving"
            return False
        if feedback.timestamp > self.last_sample:
            if feedback.timestamp - self.last_sample > .15:
                self.quiet_count = 0
                self.quiet_since = self.ready_at = self.ready_yaw_right_deg = None
            if self.quiet_since is None:
                self.quiet_since = feedback.timestamp
            self.quiet_count += 1
            self.last_sample = feedback.timestamp
            if self.quiet_count >= 2 and feedback.timestamp - self.quiet_since >= .04:
                if self.ready_at is None:
                    self.ready_at = feedback.timestamp
                    yaw = getattr(feedback, "integrated_yaw_right_deg", None)
                    try:
                        yaw = float(yaw)
                    except (TypeError, ValueError):
                        yaw = float("nan")
                    self.ready_yaw_right_deg = yaw if math.isfinite(yaw) else None
        self.reason = "quiet_confirmed" if self.ready_at is not None else "await_second_quiet_sample"
        return self.ready_at is not None

    def motion_handoff_ready(self, capture_timestamp, feedback, now):
        """Qualified SAME-UID motion goes to the wheel guard, not a quiet gate.

        Parking oscillation is not an identity veto. This only releases the
        ordinary hold; it neither grants depth authority nor bypasses reversal
        protection in the wheel executor. Search handoff uses release_ready.
        """
        if self.forward_resume_live(now):
            if self.require_current_release and self.current_released_at is None:
                self.reason = "await_current_release"
                return False
        elif not self.recovery_gate(now):
            return False
        # Translation needs post-STOP evidence, not another image after the
        # later 0A/FREE transaction. Settled search/recentering stays stricter.
        if not math.isfinite(capture_timestamp) or not self.sent_at < capture_timestamp <= now:
            self.reason = "image_before_stop_write"
            return False
        values = None if feedback is None else (
            feedback.timestamp, feedback.left_forward_rpm, feedback.right_forward_rpm)
        if not (values is not None and feedback.trustworthy
                and all(math.isfinite(float(v)) for v in values)
                and self.sent_at < feedback.timestamp <= now
                and feedback.timestamp >= self.last_sample
                and now - feedback.timestamp <= .15):
            self.reason = "feedback_not_fresh_post_stop"
            return False
        self.reason = "qualified_motion_to_wheel_guard"
        return True

    def release_ready(self, capture_timestamp, feedback, now):
        quiet = self.observe(feedback, now)
        if not self.recovery_gate(now):
            return False
        if not quiet:
            return False
        if not math.isfinite(capture_timestamp) or capture_timestamp <= self.sent_at:
            self.reason = "image_before_stop_write"
            return False
        if capture_timestamp < self.ready_at:
            self.reason = "image_before_quiet_confirmation"
            return False
        self.reason = "post_stop_image_and_quiet"
        return True


def newer_visual_evidence(capture_id, stamp, reference, now, max_age):
    """Both physical provenance fields must advance; publication is not evidence."""
    return bool(
        isinstance(capture_id, int) and not isinstance(capture_id, bool)
        and isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
        and math.isfinite(stamp) and stamp > 0
        and capture_id > reference[0] and stamp > reference[1]
        and 0 <= now - stamp <= max_age
    )


def predictive_or_center_stop(result):
    """A transient PID zero or direction guard is not a parking request."""
    return bool(
        result is not None
        and int(getattr(result, "correction_rpm", 1)) == 0
        and (getattr(result, "predictive_braking", False)
             or getattr(result, "output_floor_reason", "") in {
                 "predictive_brake_coast", "center_hold",
             })
    )


def low_speed_turn_stop_reason(intent, feedback, *, now, active_uid,
                               linear_rpm, max_image_age):
    """Qualify a whole-chassis stop from measured motion, not a mode label.

    Only a pivot-dominated, slow chassis may override a small forward grant.
    Ordinary two-wheel forward travel keeps its independent longitudinal axis.
    This function neither reads hardware nor grants movement authority.
    """
    if (intent.target_id != active_uid or intent.bbox_quality != "reliable"
            or not intent.valid(now) or intent.capture_frame_id <= 0
            or not 0 < intent.capture_timestamp <= now
            or now - intent.capture_timestamp > max_image_age):
        return "unqualified_visual"
    if feedback is None or not feedback.trustworthy:
        return "feedback_unavailable"
    values = (feedback.timestamp, feedback.left_forward_rpm,
              feedback.right_forward_rpm, linear_rpm)
    if not all(math.isfinite(float(v)) for v in values):
        return "feedback_invalid"
    if not 0 <= now - feedback.timestamp <= .15:
        return "feedback_stale"
    if linear_rpm < 0 or linear_rpm > 10:
        return "translation_requested"
    left, right = feedback.left_forward_rpm, feedback.right_forward_rpm
    base = abs((left + right) / 2.)
    yaw = abs((left - right) / 2.)
    if base > 10 or max(abs(left), abs(right)) > 30 or base > yaw:
        return "translation_dominant"
    if yaw < 2:
        return "no_residual_turn"
    return "low_speed_turn_stop"
