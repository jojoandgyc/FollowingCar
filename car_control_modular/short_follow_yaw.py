"""Measured-position yaw taper, independent of longitudinal authorization.

The camera and encoder convention is right-positive chassis yaw. Turning right
moves a stationary subject left in the image. Feedback can only remove some of
the measured visual error; it cannot invent an opposite-side target or increase
the original request. No target velocity is estimated here.
"""
from dataclasses import dataclass, replace
import math
from typing import Optional


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class ShortFollowYawObservation:
    uid: int
    capture_id: int
    capture_timestamp: float
    center_x_ratio: float
    capture_yaw_deg: Optional[float] = None


def tapered_center(observation, *, current_yaw_deg=None, yaw_rate_deg_s=None,
                   camera_hfov_deg=60., damping_sec=.10):
    """Return a same-side center with measured yaw and a small rate taper.

    ``capture_yaw_deg`` must be capture-aligned, and the caller qualifies the
    freshness and reliability of current feedback. Missing feedback is not a
    stop condition. The 100 ms rate horizon is a tuning starting point, not a
    measured braking guarantee; it never grants a new direction or more RPM.
    """
    error = observation.center_x_ratio - .5
    sign = 1. if error >= 0 else -1.
    remaining = abs(error)
    reason = "none"
    if (finite(observation.capture_yaw_deg) and finite(current_yaw_deg)
            and finite(camera_hfov_deg) and camera_hfov_deg > 0):
        # Integrated heading is unwrapped. Never wrap a discontinuity into
        # seemingly useful movement; a >45 degree change within this short
        # observation lifetime is not usable taper evidence.
        change = current_yaw_deg - observation.capture_yaw_deg
        if abs(change) <= 45.:
            completed = max(0., sign * change / camera_hfov_deg)
            if completed > 0:
                remaining = max(0., remaining - completed)
                reason = "measured_yaw_taper"
    if (finite(yaw_rate_deg_s) and abs(yaw_rate_deg_s) <= 180.
            and finite(camera_hfov_deg) and camera_hfov_deg > 0
            and finite(damping_sec) and 0 <= damping_sec <= .20):
        lookahead = max(0., sign * yaw_rate_deg_s) * damping_sec / camera_hfov_deg
        if lookahead > 0:
            remaining = max(0., remaining - lookahead)
            reason = "yaw_rate_taper"
    return .5 + sign * remaining, reason


class ShortFollowYawResponse:
    """Finite executed-response evidence; never a plan or motion permission.

    Persistent understeer can reduce BOTH forward wheels equally, preserving
    the already legal differential. This trades translation for curvature;
    it cannot request more yaw, reverse an inner wheel, or create a STOP.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._receipt = None
        self._binding = None
        self._command_key = None
        self._command_since = None
        self._command_delta = 0
        self._last_capture = None
        self._last_feedback = None
        self._first_low_response = None
        self._count = 0
        self._reduction = 0
        self._last_adjusted = -float("inf")

    def _clear_confirmation(self):
        self._first_low_response = None
        self._count = 0

    @staticmethod
    def _direction(delta):
        return 1 if delta > 0 else -1 if delta < 0 else 0

    def acknowledge(self, plan, receipt, left_rpm, right_rpm):
        """Only the executor's completed dual-wheel write advances lineage."""
        if receipt is None:
            self.reset()
            return
        binding = plan.uid, plan.epoch
        if binding != self._binding:
            self.reset()
            self._binding = binding
        delta = left_rpm - right_rpm
        key = (*binding, self._direction(delta)) if min(left_rpm, right_rpm) > 0 and abs(delta) >= 6 else None
        if key != self._command_key:
            self._clear_confirmation()
            self._command_since = receipt.completed_at if key is not None else None
        self._command_key, self._command_delta = key, abs(delta)
        self._receipt = receipt

    def adjust(self, plan, feedback, receipt, now, config, *, wheel_limit=None):
        """Use only live plans and sustained independent encoder samples.

        Three low-response samples must span 160 ms after 100 ms of actual
        command exposure. Repeated reads do not confirm or deepen reduction.
        Recovery removes at most 2 RPM of reduction per 50 ms, including when
        feedback goes stale. No source/identity deadline is ever modified.
        """
        if (config.yaw_understeer_reduction_rpm <= 0 or not plan.valid(now)
                or (wheel_limit is not None and (not finite(wheel_limit) or wheel_limit < 1))
                or min(plan.left_rpm, plan.right_rpm) <= 0
                or receipt is None or receipt is not self._receipt
                or (plan.uid, plan.epoch) != self._binding):
            self.reset()
            return plan
        delta = plan.left_rpm - plan.right_rpm
        direction = self._direction(delta)
        capture = plan.yaw_capture_id, plan.yaw_capture_timestamp
        ordered = (self._last_capture is None or capture == self._last_capture
                   or (capture[0] > self._last_capture[0] and capture[1] > self._last_capture[1]))
        if ordered:
            self._last_capture = capture
        # Do not interpret near-centre quantization, a new direction, or a
        # retired/tapered request as proof of poor wheel response.
        eligible = (ordered and abs(delta) >= 6
            and abs(plan.yaw_control_center_x_ratio - .5) >= max(.16, config.center_deadband_ratio + .06)
            and self._command_key == (*self._binding, direction)
            and self._command_since is not None
            and 0 <= now - receipt.completed_at <= .15)
        stamp = getattr(feedback, "timestamp", None)
        left = getattr(feedback, "left_forward_rpm", None)
        right = getattr(feedback, "right_forward_rpm", None)
        fresh = (getattr(feedback, "trustworthy", False)
            and not getattr(feedback, "left_error", 0) and not getattr(feedback, "right_error", 0)
            and all(finite(value) for value in (stamp, left, right))
            and 0 <= now - stamp <= .15 and min(left, right) >= 0)
        new_sample = fresh and (self._last_feedback is None or stamp > self._last_feedback)
        measured = direction * (left - right) if fresh else None
        opposite_response = bool(fresh and measured < -2)
        reason = "opposite_response_observing" if opposite_response else "observing"
        if not eligible or not fresh:
            self._clear_confirmation()
            reason = "response_unavailable"
        elif new_sample:
            if self._last_feedback is not None and stamp - self._last_feedback > .15:
                self._clear_confirmation()
            if stamp - self._command_since < .10 - 1e-9:
                # A fresh encoder sample can still predate enough exposure to
                # the executed command. That is not evidence of recovery.
                self._clear_confirmation()
                reason = "command_exposure_pending"
            elif measured < .4 * min(abs(delta), self._command_delta):
                # Both wheels are already qualified as forward/nonnegative.
                # An opposite DIFFERENTIAL is therefore a poor yaw response,
                # not the reverse-wheel safety condition owned by the writer.
                # Count it with weak same-side response only after the same
                # finite command exposure and independent-sample checks.
                if self._first_low_response is None:
                    self._first_low_response = stamp
                self._count += 1
            else:
                self._clear_confirmation()
                reason = "response_recovered"
        if new_sample:
            self._last_feedback = stamp
        confirmed = (eligible and fresh and self._count >= 3
            and self._first_low_response is not None
            and stamp - self._first_low_response >= .16 - 1e-9)
        if confirmed:
            reason = "opposite_response_confirmed" if opposite_response else "understeer_confirmed"
        maximum = min(int(config.yaw_understeer_reduction_rpm), min(plan.left_rpm, plan.right_rpm) - 1)
        if wheel_limit is not None:
            # The writer still applies its original hardware/feedback scale
            # and integer rounding. Do not introduce an inner-wheel zero at
            # THAT boundary. Zero correction always preserves the old pair,
            # including any zero already required by a very low speed limit.
            inner, outer = min(plan.left_rpm, plan.right_rpm), max(plan.left_rpm, plan.right_rpm)
            while maximum > 0:
                scale = min(1., wheel_limit / (outer - maximum))
                if round((inner - maximum) * scale) >= 1:
                    break
                maximum -= 1
        self._reduction = min(self._reduction, maximum)
        if now - self._last_adjusted >= .05 - 1e-9:
            before = self._reduction
            if confirmed and new_sample:
                self._reduction = min(maximum, before + 2)
            elif not confirmed:
                self._reduction = max(0, before - 2)
            if self._reduction != before:
                self._last_adjusted = now
        reduction = self._reduction
        if reduction and not confirmed:
            reason = "response_recovery"
        return replace(plan, left_rpm=plan.left_rpm-reduction, right_rpm=plan.right_rpm-reduction,
            base_rpm=plan.base_rpm-reduction, yaw_common_reduction_rpm=reduction,
            yaw_response_reason=reason, yaw_response_sample_count=self._count)
