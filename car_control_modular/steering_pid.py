from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from .control_types import SteeringFeedback


@dataclass(frozen=True)
class DistancePidConfig:
    """Outer-loop distance controller parameters.

    The output is a forward wheel-speed target in RPM.  The motor driver
    remains responsible for its inner encoder speed loop.
    """

    kp_rpm_per_m: float = 22.0
    ki_rpm_per_m_s: float = 1.5
    kd_rpm_s_per_m: float = 6.0
    integral_limit_m_s: float = 1.5
    deadband_m: float = 0.005
    min_forward_output_rpm: float = 20.0
    max_forward_output_rpm: float = 100.0
    min_reverse_output_rpm: float = 20.0
    max_reverse_output_rpm: float = 100.0
    derivative_filter_alpha: float = 0.25
    # Runtime depth can occasionally jump by several metres between two
    # samples. A zero value keeps the guard disabled for standalone callers.
    max_measurement_jump_m: float = 0.0
    # Limit command slew after the PID calculation. These are disabled by
    # default so existing unit-test/config callers retain the original loop.
    output_rise_rpm_per_sec: float = 0.0
    output_fall_rpm_per_sec: float = 0.0


@dataclass(frozen=True)
class DistancePidResult:
    raw_actual_distance_m: float
    actual_distance_m: float
    target_distance_m: float
    error_m: float
    error_rate_m_s: float
    integral_m_s: float
    p_rpm: float
    i_rpm: float
    d_rpm: float
    output_rpm: int
    unslewed_output_rpm: float
    measurement_jump_clamped: bool
    output_slew_limited: bool


class LongitudinalDistancePid:
    """Distance outer loop producing a bounded forward RPM target.

    Positive error means the person is farther than the target.  The caller
    owns the safety policy and decides when the output must be forced to zero;
    this class only regulates a valid positive-distance forward request.
    """

    def __init__(self, config: DistancePidConfig) -> None:
        self.config = config
        self._last_ts: Optional[float] = None
        self._last_error_m: Optional[float] = None
        self._integral_m_s = 0.0
        self._filtered_error_rate_m_s = 0.0
        self._last_nonzero_sign = 0
        self._last_output_rpm: Optional[float] = None
        self.last_result: Optional[DistancePidResult] = None

    def reset(self) -> None:
        self._last_ts = None
        self._last_error_m = None
        self._integral_m_s = 0.0
        self._filtered_error_rate_m_s = 0.0
        self._last_nonzero_sign = 0
        self._last_output_rpm = None
        self.last_result = None

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        bound = max(0.0, float(limit))
        return max(-bound, min(bound, float(value)))

    def update(
        self,
        actual_distance_m: float,
        target_distance_m: float,
        *,
        now: Optional[float] = None,
    ) -> DistancePidResult:
        c = self.config
        now = time.monotonic() if now is None else float(now)
        raw_actual = float(actual_distance_m)
        actual = raw_actual
        target = float(target_distance_m)
        if not math.isfinite(actual) or not math.isfinite(target):
            raise ValueError("distance PID requires finite distances")
        if (
            self.last_result is not None
            and self._last_ts is not None
            and now <= self._last_ts + 1e-9
            and abs(actual - self.last_result.actual_distance_m) <= 1e-9
            and abs(target - self.last_result.target_distance_m) <= 1e-9
        ):
            return self.last_result

        measurement_jump_clamped = False
        max_jump = max(0.0, float(c.max_measurement_jump_m))
        if (
            max_jump > 0.0
            and self.last_result is not None
            and self._last_ts is not None
        ):
            previous_actual = float(self.last_result.actual_distance_m)
            delta = actual - previous_actual
            # The vision loop may have a long processing gap; that must not
            # make a physically impossible depth jump authoritative.
            if abs(delta) > max_jump:
                actual = previous_actual + math.copysign(max_jump, delta)
                measurement_jump_clamped = True

        error = actual - target
        if self._last_ts is None or self._last_error_m is None:
            dt = 0.10
            raw_rate = 0.0
        else:
            dt = max(0.03, min(0.50, now - self._last_ts))
            raw_rate = (error - self._last_error_m) / dt
        self._last_ts = now
        self._last_error_m = error

        rate_alpha = max(0.0, min(1.0, float(c.derivative_filter_alpha)))
        self._filtered_error_rate_m_s = (
            rate_alpha * raw_rate
            + (1.0 - rate_alpha) * self._filtered_error_rate_m_s
        )

        deadband = max(0.0, float(c.deadband_m))
        if abs(error) <= deadband:
            self._integral_m_s = 0.0
            self._last_nonzero_sign = 0
            output_rpm = 0
            p_rpm = i_rpm = d_rpm = 0.0
        else:
            direction = 1 if error > 0.0 else -1
            if self._last_nonzero_sign not in (0, direction):
                self._integral_m_s = 0.0
                self._filtered_error_rate_m_s = 0.0
            self._last_nonzero_sign = direction
            self._integral_m_s = self._clip(
                self._integral_m_s + error * dt,
                float(c.integral_limit_m_s),
            )
            p_rpm = float(c.kp_rpm_per_m) * error
            i_rpm = float(c.ki_rpm_per_m_s) * self._integral_m_s
            d_rpm = float(c.kd_rpm_s_per_m) * self._filtered_error_rate_m_s
            if direction > 0:
                minimum = max(0.0, float(c.min_forward_output_rpm))
                maximum = max(minimum, float(c.max_forward_output_rpm))
                requested = minimum + p_rpm + i_rpm + d_rpm
                output_rpm = max(
                    int(round(minimum)),
                    min(int(round(maximum)), int(round(requested))),
                )
            else:
                minimum = max(0.0, float(c.min_reverse_output_rpm))
                maximum = max(minimum, float(c.max_reverse_output_rpm))
                requested = -minimum + p_rpm + i_rpm + d_rpm
                output_rpm = min(
                    -int(round(minimum)),
                    max(-int(round(maximum)), int(round(requested))),
                )

        unslewed_output_rpm = float(output_rpm)
        output_slew_limited = False
        if self._last_output_rpm is not None and output_rpm != 0:
            previous_output = float(self._last_output_rpm)
            limit = 0.0
            # Never slew through zero into the wrong travel direction. A
            # sign change is a control transition and is applied immediately;
            # the caller's safety/hysteresis logic owns any braking policy.
            if previous_output * float(output_rpm) < 0.0:
                pass
            elif output_rpm > previous_output:
                limit = max(0.0, float(c.output_rise_rpm_per_sec)) * dt
            else:
                limit = max(0.0, float(c.output_fall_rpm_per_sec)) * dt
            if (
                previous_output * float(output_rpm) >= 0.0
                and limit > 0.0
                and abs(float(output_rpm) - previous_output) > limit
            ):
                output_rpm = int(round(previous_output + math.copysign(limit, float(output_rpm) - previous_output)))
                output_slew_limited = True
        # A true deadband request is an explicit stop and must not be delayed
        # by the comfort slew limiter; safety and distance hysteresis own the
        # decision to stop.
        self._last_output_rpm = float(output_rpm)

        result = DistancePidResult(
            raw_actual_distance_m=raw_actual,
            actual_distance_m=actual,
            target_distance_m=target,
            error_m=error,
            error_rate_m_s=self._filtered_error_rate_m_s,
            integral_m_s=self._integral_m_s,
            p_rpm=p_rpm,
            i_rpm=i_rpm,
            d_rpm=d_rpm,
            output_rpm=output_rpm,
            unslewed_output_rpm=unslewed_output_rpm,
            measurement_jump_clamped=measurement_jump_clamped,
            output_slew_limited=output_slew_limited,
        )
        self.last_result = result
        return result


def encoder_yaw_rate_right_dps(
    left_speed_rpm: int,
    right_speed_rpm: int,
    left_forward_sign: int,
    right_forward_sign: int,
    left_body_deg_per_encoder_deg: float,
    right_body_deg_per_encoder_deg: float,
) -> tuple[float, float, float]:
    """Return normalized wheel RPM and right-positive body yaw rate."""
    left_forward_rpm = float(left_speed_rpm) * (1.0 if int(left_forward_sign) >= 0 else -1.0)
    right_forward_rpm = float(right_speed_rpm) * (1.0 if int(right_forward_sign) >= 0 else -1.0)
    wheel_diff_rpm = left_forward_rpm - right_forward_rpm
    body_scale = (
        float(right_body_deg_per_encoder_deg)
        if wheel_diff_rpm >= 0.0
        else float(left_body_deg_per_encoder_deg)
    )
    yaw_rate = 0.5 * wheel_diff_rpm * 6.0 * body_scale
    return left_forward_rpm, right_forward_rpm, yaw_rate


@dataclass(frozen=True)
class VisualSteeringPidConfig:
    enabled: bool = False
    camera_hfov_deg: float = 90.0
    camera_latency_sec: float = 0.13
    deadband_deg: float = 1.5
    outer_kp_per_sec: float = 1.85
    outer_kd_sec: float = 0.08
    target_rate_feedforward_gain: float = 0.0
    target_rate_feedforward_max_dps: float = 0.0
    # When consecutive image samples provide a target-bearing rate, allow the
    # chassis to exceed that rate only by this much. This keeps position error
    # from continuously accelerating through a moving target. Zero disables
    # the limiter for standalone callers and legacy configurations.
    target_speed_match_max_closing_dps: float = 0.0
    max_yaw_rate_dps: float = 46.0
    rate_kp_rpm_per_dps: float = 0.16
    rate_ki_rpm_per_deg: float = 0.01
    rate_integral_limit_deg: float = 25.0
    max_correction_rpm: float = 16.0
    dynamic_small_error_deg: float = 3.5
    dynamic_large_error_deg: float = 14.0
    dynamic_small_max_yaw_rate_dps: float = 22.0
    dynamic_small_max_correction_rpm: float = 6.0
    dynamic_large_error_base_cap_rpm: float = 28.0
    opposite_yaw_brake_threshold_dps: float = 6.0
    opposite_yaw_brake_boost_rpm: float = 6.0
    # Countersteer is a braking operation while residual chassis yaw is still
    # large. Keep it below the normal tracking limit so a sign change cannot
    # immediately become a full-speed reversal.
    braking_max_correction_rpm: float = 12.0
    # Strong opposite yaw gets a bounded proportional countersteer burst.
    # Disabled for direct library/test callers unless explicitly enabled by
    # the board runtime configuration. This preserves the historical hard
    # braking cap while allowing the runtime to opt into proportional
    # countersteer during fast target crossings.
    fast_countersteer_max_correction_rpm: float = 0.0
    fast_countersteer_gain_rpm_per_dps: float = 0.0
    same_direction_overspeed_threshold_dps: float = 8.0
    same_direction_overspeed_brake_gain_rpm_per_dps: float = 0.25
    visual_direction_guard_enabled: bool = False
    predictive_brake_decel_dps2: float = 0.0
    predictive_brake_margin_deg: float = 0.0
    predictive_brake_response_sec: float = 0.0
    min_effective_error_deg: float = 0.0
    min_effective_correction_rpm: float = 0.0
    mechanical_tier2_error_deg: float = 0.0
    mechanical_tier2_correction_rpm: float = 0.0
    mechanical_tier3_error_deg: float = 0.0
    mechanical_tier3_correction_rpm: float = 0.0
    mechanical_floor_release_ratio: float = 0.0
    startup_kick_error_deg: float = 0.0
    startup_kick_rpm: float = 0.0
    startup_kick_max_sec: float = 0.0
    startup_kick_release_yaw_rate_dps: float = 0.0
    active_brake_yaw_threshold_dps: float = 0.0
    active_brake_min_correction_rpm: float = 0.0
    edge_boost_start_error_deg: float = 18.0
    aggressive_inner_wheel_margin_rpm: float = 0.0
    left_body_deg_per_encoder_deg: float = 0.5225
    right_body_deg_per_encoder_deg: float = 0.5424
    feedback_stale_sec: float = 0.30
    error_filter_alpha: float = 0.55
    derivative_filter_alpha: float = 0.35


@dataclass(frozen=True)
class VisualSteeringPidResult:
    correction_rpm: int
    requested_base_rpm: int
    base_rpm: int
    yaw_rate_limit_dps: float
    correction_limit_rpm: float
    correction_limit_reason: str
    visual_error_deg: float
    compensated_error_deg: float
    filtered_error_deg: float
    error_rate_dps: float
    target_rate_valid: bool
    target_image_rate_dps: float
    target_bearing_rate_dps: float
    target_rate_feedforward_dps: float
    target_speed_match_limited: bool
    target_speed_match_limit_dps: float
    desired_yaw_rate_dps: float
    measured_yaw_rate_dps: float
    feedback_age_sec: Optional[float]
    feedback_used: bool
    feedforward_rpm: float
    rate_p_rpm: float
    rate_i_rpm: float
    unsaturated_rpm: float
    opposite_yaw_braking: bool
    same_direction_overspeed_braking: bool
    visual_direction_guarded: bool
    predictive_braking: bool
    remaining_error_deg: float
    stopping_distance_deg: float
    prediction_latency_sec: float
    yaw_rate_overshoot_dps: float
    overspeed_brake_rpm: float
    edge_boost_active: bool
    output_floor_rpm: float
    output_floor_reason: str
    startup_kick_active: bool
    startup_kick_elapsed_sec: float
    startup_kick_release_reason: str


class VisualSteeringPid:
    """Camera angle outer loop with encoder yaw-rate feedback.

    Positive values consistently mean a right turn. The camera supplies the
    target angle; ABZ-derived wheel RPM supplies the fast inner-loop feedback.
    """

    def __init__(self, config: VisualSteeringPidConfig) -> None:
        self.config = config
        self._last_ts: Optional[float] = None
        self._filtered_error_deg = 0.0
        self._filtered_error_rate_dps = 0.0
        self._rate_integral_deg = 0.0
        self._initialized = False
        self._startup_kick_direction = 0
        self._startup_kick_started_at: Optional[float] = None
        self._startup_kick_armed = True
        self.last_result: Optional[VisualSteeringPidResult] = None

    def reset(self) -> None:
        self._last_ts = None
        self._filtered_error_deg = 0.0
        self._filtered_error_rate_dps = 0.0
        self._rate_integral_deg = 0.0
        self._initialized = False
        self._startup_kick_direction = 0
        self._startup_kick_started_at = None
        self._startup_kick_armed = True
        self.last_result = None

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        limit = max(0.0, float(limit))
        return max(-limit, min(limit, float(value)))

    def update(
        self,
        x_ratio: float,
        base_rpm: int,
        feedback: Optional[SteeringFeedback],
        *,
        now: Optional[float] = None,
        target_image_rate_dps: Optional[float] = None,
        max_correction_override_rpm: Optional[float] = None,
        visual_age_sec: Optional[float] = None,
    ) -> VisualSteeringPidResult:
        c = self.config
        now = time.monotonic() if now is None else float(now)
        visual_error_deg = (max(0.0, min(1.0, float(x_ratio))) - 0.5) * max(
            1.0, float(c.camera_hfov_deg)
        )

        feedback_age_sec: Optional[float] = None
        feedback_used = False
        measured_rate = 0.0
        startup_response_rate = 0.0
        if feedback is not None:
            feedback_age_sec = max(0.0, now - float(feedback.timestamp))
            feedback_used = bool(
                feedback.trustworthy
                and feedback_age_sec <= max(0.05, float(c.feedback_stale_sec))
                and math.isfinite(float(feedback.yaw_rate_right_dps))
            )
            if feedback_used:
                measured_rate = float(feedback.yaw_rate_right_dps)
                # A single encoder interval can report an implausible yaw
                # spike while the chassis is being stopped.  Letting that
                # value enter camera-latency compensation turns a temporary
                # feedback glitch into a full opposite-direction command.
                # The configured yaw envelope is also the maximum rate the
                # outer loop can request, so use it as a conservative bound.
                measured_rate = self._clip(
                    measured_rate,
                    max(1.0, float(c.max_yaw_rate_dps)),
                )
                raw_rate = getattr(feedback, "raw_yaw_rate_right_dps", None)
                startup_response_rate = measured_rate
                if raw_rate is not None and math.isfinite(float(raw_rate)):
                    startup_response_rate = float(raw_rate)

        # 摄像头画面落后于车身运动。用编码器角速度把视觉误差外推到当前时刻，
        # 目标已接近中心但车仍在转时会提前减小轮差，而不是越过中心后再反打。
        prediction_latency_sec = max(0.0, float(c.camera_latency_sec))
        if visual_age_sec is not None and math.isfinite(float(visual_age_sec)):
            prediction_latency_sec = max(
                prediction_latency_sec,
                max(0.0, float(visual_age_sec)),
            )
        prediction_latency_sec += max(
            0.0,
            float(c.predictive_brake_response_sec),
        )
        # Stale frames are rejected before reaching this loop. The cap protects
        # standalone callers from a bad capture timestamp freezing yaw forever.
        prediction_latency_sec = min(
            prediction_latency_sec,
            max(0.25, float(c.feedback_stale_sec)),
        )

        compensated_error = visual_error_deg
        if feedback_used:
            compensated_error -= measured_rate * prediction_latency_sec

        if self._last_ts is None:
            dt = 0.12
        else:
            dt = max(0.03, min(0.40, now - self._last_ts))
        self._last_ts = now

        error_alpha = max(0.0, min(1.0, float(c.error_filter_alpha)))
        derivative_alpha = max(0.0, min(1.0, float(c.derivative_filter_alpha)))
        if not self._initialized:
            self._filtered_error_deg = compensated_error
            raw_error_rate = 0.0
            self._initialized = True
        else:
            previous_error = self._filtered_error_deg
            self._filtered_error_deg = (
                error_alpha * compensated_error + (1.0 - error_alpha) * previous_error
            )
            raw_error_rate = (self._filtered_error_deg - previous_error) / dt
        raw_error_rate = self._clip(raw_error_rate, 120.0)
        self._filtered_error_rate_dps = (
            derivative_alpha * raw_error_rate
            + (1.0 - derivative_alpha) * self._filtered_error_rate_dps
        )

        deadband = max(0.0, float(c.deadband_deg))
        if abs(self._filtered_error_deg) <= deadband:
            effective_error = 0.0
        else:
            effective_error = math.copysign(abs(self._filtered_error_deg) - deadband, self._filtered_error_deg)

        # 小偏差时限制转向力度，避免目标在中心附近左右摆动；偏差越大，
        # 角速度和轮差上限越高，同时逐步降低前进基准，给车身留出转向能力。
        small_error = max(0.0, float(c.dynamic_small_error_deg))
        large_error = max(small_error + 0.1, float(c.dynamic_large_error_deg))
        dynamic_blend = max(
            0.0,
            min(1.0, (abs(self._filtered_error_deg) - small_error) / (large_error - small_error)),
        )
        small_yaw_limit = max(0.0, min(float(c.max_yaw_rate_dps), float(c.dynamic_small_max_yaw_rate_dps)))
        yaw_rate_limit = small_yaw_limit + (
            max(0.0, float(c.max_yaw_rate_dps)) - small_yaw_limit
        ) * dynamic_blend
        small_correction_limit = max(
            0.0,
            min(float(c.max_correction_rpm), float(c.dynamic_small_max_correction_rpm)),
        )
        dynamic_correction_limit = small_correction_limit + (
            max(0.0, float(c.max_correction_rpm)) - small_correction_limit
        ) * dynamic_blend
        requested_base_rpm = max(0, int(base_rpm))
        large_error_base_cap = max(0.0, float(c.dynamic_large_error_base_cap_rpm))
        reduced_base_rpm = float(requested_base_rpm) - (
            max(0.0, float(requested_base_rpm) - large_error_base_cap) * dynamic_blend
        )
        limited_base_rpm = max(0, int(round(reduced_base_rpm)))

        # The bbox-center rate is relative to the camera. Keep the compensated
        # world-bearing rate for diagnostics, but use the direct image rate for
        # containment: outward motion may add same-side yaw and inward motion
        # may only reduce it. It must never power a turn across the target's
        # current side of the image.
        target_rate_valid = bool(
            target_image_rate_dps is not None
            and math.isfinite(float(target_image_rate_dps))
        )
        image_rate = float(target_image_rate_dps) if target_rate_valid else 0.0
        requested_direction = 0
        if abs(visual_error_deg) > deadband:
            requested_direction = 1 if visual_error_deg > 0.0 else -1
        elif (
            target_rate_valid
            and abs(visual_error_deg) > 1e-6
            and visual_error_deg * image_rate > 0.0
            and abs(image_rate) >= 2.0
        ):
            # The target is still inside the deadband but is already moving
            # outward. Start following before it reaches the outer boundary.
            requested_direction = 1 if visual_error_deg > 0.0 else -1
        if target_rate_valid:
            target_bearing_rate = image_rate + measured_rate if feedback_used else image_rate
            target_rate_feedforward = (
                self._clip(
                    max(0.0, float(c.target_rate_feedforward_gain)) * image_rate,
                    max(0.0, float(c.target_rate_feedforward_max_dps)),
                )
                if requested_direction != 0
                else 0.0
            )
        else:
            # A missing motion sample is not a stationary image measurement.
            # In particular, adding encoder yaw to a fabricated 0 dps sample
            # would interpret chassis inertia as target motion and sustain the
            # very rotation that the position loop is trying to stop.
            target_bearing_rate = 0.0
            target_rate_feedforward = 0.0
        desired_rate = (
            float(c.outer_kp_per_sec) * effective_error
            + float(c.outer_kd_sec) * self._filtered_error_rate_dps
            + target_rate_feedforward
        )
        target_return_coast = False
        desired_direction_clamped = False
        if requested_direction != 0:
            desired_direction_clamped = bool(
                desired_rate * requested_direction < 0.0
            )
            inward_rate = max(0.0, -image_rate * requested_direction)
            remaining_to_deadband = max(0.0, abs(visual_error_deg) - deadband)
            target_reaches_deadband_during_latency = bool(
                target_rate_valid
                and inward_rate > 0.0
                and inward_rate * prediction_latency_sec >= remaining_to_deadband
            )
            target_return_coast = bool(
                target_rate_valid
                and image_rate * requested_direction < 0.0
                and (
                    desired_rate * requested_direction <= 0.0
                    or target_reaches_deadband_during_latency
                )
            )
            desired_rate = requested_direction * max(
                0.0,
                requested_direction * desired_rate,
            )
        else:
            desired_rate = 0.0
        desired_rate = self._clip(desired_rate, yaw_rate_limit)
        target_speed_match_limited = False
        target_speed_match_limit = yaw_rate_limit
        max_closing_rate = max(0.0, float(c.target_speed_match_max_closing_dps))
        if (
            max_closing_rate > 0.0
            and target_rate_valid
            and requested_direction != 0
        ):
            # target_bearing_rate is the target's estimated world rate:
            # camera-relative bbox rate + measured chassis yaw. Permit a
            # bounded same-side excess so the aim line can converge without
            # the chassis accelerating far beyond the target and sweeping it
            # across the complete image between two vision updates.
            same_side_target_rate = max(
                0.0,
                requested_direction * target_bearing_rate,
            )
            target_speed_match_limit = min(
                yaw_rate_limit,
                same_side_target_rate + max_closing_rate,
            )
            if requested_direction * desired_rate > target_speed_match_limit:
                desired_rate = requested_direction * target_speed_match_limit
                target_speed_match_limited = True
        # Startup state follows the current camera side, not the filtered/D
        # output. A raw center crossing must rearm the next kick even while the
        # controller is still braking residual encoder yaw.
        if requested_direction == 0:
            # Returning to the visual deadband rearms one future startup kick.
            self._startup_kick_direction = 0
            self._startup_kick_started_at = None
            self._startup_kick_armed = True
        elif requested_direction != self._startup_kick_direction:
            # A real direction change also needs one new static-friction kick.
            self._startup_kick_direction = requested_direction
            self._startup_kick_started_at = None
            self._startup_kick_armed = True

        # Fresh target position is the final yaw-direction authority. Delayed
        # image derivatives and encoder feedback may ask us to slow down, but
        # they may not power a turn away from the side where the target is now.
        visual_direction_guarded = desired_direction_clamped
        if (
            bool(c.visual_direction_guard_enabled)
            and requested_direction != 0
            and desired_rate * requested_direction < 0.0
        ):
            desired_rate = 0.0
            visual_direction_guarded = True
        rate_error = desired_rate - measured_rate

        # 目标要求的转向与车身实际角速度相反时，说明旧方向的惯性尚未消失。
        # 清掉旧积分并临时放宽轮差，直接用反向差速消除角速度，不插入 STOP。
        opposite_yaw_braking = bool(
            feedback_used
            and abs(desired_rate) > 1e-6
            and desired_rate * measured_rate < 0.0
            and abs(measured_rate) >= max(0.0, float(c.opposite_yaw_brake_threshold_dps))
        )
        countersteer_limit = max(0.0, float(c.braking_max_correction_rpm))
        if opposite_yaw_braking:
            self._rate_integral_deg = 0.0
            dynamic_correction_limit = min(
                max(0.0, float(c.max_correction_rpm)),
                dynamic_correction_limit + max(0.0, float(c.opposite_yaw_brake_boost_rpm)),
            )
            fast_countersteer_cap = max(
                countersteer_limit,
                min(
                    max(0.0, float(c.fast_countersteer_max_correction_rpm)),
                    countersteer_limit
                    + max(0.0, float(c.fast_countersteer_gain_rpm_per_dps))
                    * abs(measured_rate),
                ),
            )
            countersteer_limit = min(
                max(0.0, float(c.max_correction_rpm)),
                fast_countersteer_cap,
            )

        # 期望和实际转向同向不代表还应继续驱动。车身角速度已经明显超过
        # 目标时先撤掉轮差，让电机的零速闭环减速。目标尚未越过中心前直接
        # 反向驱动会形成 minimum-RPM 极限环：同侧反打、反馈下降、再转回原侧。
        yaw_rate_overshoot_dps = 0.0
        same_direction_overspeed_braking = False
        visual_error_growing = bool(
            effective_error * self._filtered_error_rate_dps > 0.0
            and abs(self._filtered_error_rate_dps) >= 2.0
        )
        image_error_growing = bool(
            target_rate_valid
            and visual_error_deg * image_rate > 0.0
            and abs(image_rate) >= 2.0
        )
        remaining_error_deg = max(0.0, abs(visual_error_deg) - deadband)
        stopping_distance_deg = 0.0
        predictive_braking = False
        predictive_decel = max(0.0, float(c.predictive_brake_decel_dps2))
        if feedback_used and predictive_decel > 0.0:
            # During image processing and motor-command response the chassis
            # keeps rotating. Include that travel as well as the mechanical
            # coast distance; omitting it makes braking consistently late.
            stopping_distance_deg = (
                (abs(measured_rate) ** 2) / (2.0 * predictive_decel)
                + abs(measured_rate) * prediction_latency_sec
            )
            predictive_braking = bool(
                requested_direction != 0
                and measured_rate * requested_direction > 0.0
                and not image_error_growing
                and stopping_distance_deg
                + max(0.0, float(c.predictive_brake_margin_deg))
                >= remaining_error_deg
            )
        if predictive_braking:
            self._rate_integral_deg = 0.0
        if (
            feedback_used
            and desired_rate * measured_rate > 0.0
            and not visual_error_growing
        ):
            yaw_rate_overshoot_dps = max(
                0.0,
                abs(measured_rate) - abs(desired_rate),
            )
            same_direction_overspeed_braking = bool(
                yaw_rate_overshoot_dps
                >= max(0.0, float(c.same_direction_overspeed_threshold_dps))
            )
        if same_direction_overspeed_braking:
            self._rate_integral_deg = 0.0

        active_brake_min = max(0.0, float(c.active_brake_min_correction_rpm))
        active_brake_yaw_threshold = max(0.0, float(c.active_brake_yaw_threshold_dps))
        active_yaw_damping = bool(
            feedback_used
            and active_brake_min > 0.0
            and abs(measured_rate) >= active_brake_yaw_threshold
            and abs(desired_rate) < max(2.0, 0.35 * abs(measured_rate))
        )
        if same_direction_overspeed_braking or opposite_yaw_braking or active_yaw_damping:
            dynamic_correction_limit = min(
                max(0.0, float(c.max_correction_rpm)),
                max(dynamic_correction_limit, active_brake_min),
            )
            braking_cap = countersteer_limit
            if braking_cap > 0.0:
                dynamic_correction_limit = min(dynamic_correction_limit, braking_cap)

        # 靠近画面边缘时优先保住目标。正常微调仍给内轮保留 3 RPM；
        # 边缘或反向制动期间允许内轮降到配置下限，从而获得更大的瞬时轮差。
        edge_boost_active = bool(
            abs(visual_error_deg) >= max(0.0, float(c.edge_boost_start_error_deg))
        )
        # Edge priority may increase authority only during normal tracking.
        # Once the encoder says the chassis is still moving in the wrong
        # direction (or already overspeeding), restoring the edge limit would
        # turn a braking command back into the large reversal that caused the
        # overshoot in the first place.
        if edge_boost_active and not (
            same_direction_overspeed_braking
            or opposite_yaw_braking
            or active_yaw_damping
        ):
            dynamic_correction_limit = max(
                dynamic_correction_limit,
                max(0.0, float(c.max_correction_rpm)),
            )
        if max_correction_override_rpm is not None:
            dynamic_correction_limit = min(
                dynamic_correction_limit,
                max(0.0, float(max_correction_override_rpm)),
            )

        integral_limit = max(0.0, float(c.rate_integral_limit_deg))
        if feedback_used:
            self._rate_integral_deg = self._clip(
                self._rate_integral_deg + rate_error * dt,
                integral_limit,
            )
        else:
            self._rate_integral_deg *= 0.85

        body_scale = (
            float(c.right_body_deg_per_encoder_deg)
            if desired_rate >= 0.0
            else float(c.left_body_deg_per_encoder_deg)
        )
        yaw_dps_per_diff_rpm = max(0.1, 6.0 * max(0.01, body_scale))
        feedforward_rpm = desired_rate / yaw_dps_per_diff_rpm
        rate_p_rpm = float(c.rate_kp_rpm_per_dps) * rate_error if feedback_used else 0.0
        rate_i_rpm = float(c.rate_ki_rpm_per_deg) * self._rate_integral_deg if feedback_used else 0.0
        overspeed_brake_rpm = 0.0
        if same_direction_overspeed_braking:
            overspeed_brake_rpm = math.copysign(
                max(0.0, float(c.same_direction_overspeed_brake_gain_rpm_per_dps))
                * yaw_rate_overshoot_dps,
                measured_rate,
            )
        unsaturated = feedforward_rpm + rate_p_rpm + rate_i_rpm - overspeed_brake_rpm

        # Yaw authority is independent from longitudinal speed. The wheel
        # mixer clips the two final targets as a pair, so a 0-1 RPM base may
        # still drive one outer wheel slowly instead of suppressing yaw to zero.
        correction_limit = max(0.0, float(dynamic_correction_limit))
        if max_correction_override_rpm is not None:
            correction_limit_reason = "explicit_override"
        elif edge_boost_active:
            correction_limit_reason = "edge_boost"
        elif dynamic_blend <= 0.0:
            correction_limit_reason = "small_error_dynamic"
        else:
            correction_limit_reason = "error_dynamic"
        output = self._clip(unsaturated, correction_limit)
        output_floor_rpm = 0.0
        output_floor_reason = "none"
        if target_return_coast:
            output = 0.0
            output_floor_reason = "target_return_coast"
            self._rate_integral_deg = 0.0
        elif same_direction_overspeed_braking:
            # Zero target is the braking request here. A reverse command is
            # permitted only after fresh vision actually requests the other
            # direction (``opposite_yaw_braking`` below).
            output = 0.0
            output_floor_reason = "same_direction_overspeed_coast"
        elif predictive_braking:
            output = 0.0
            output_floor_reason = "predictive_brake_coast"
        elif active_brake_min > 0.0 and feedback_used:
            brake_magnitude = min(correction_limit, active_brake_min)
            if opposite_yaw_braking and brake_magnitude > 0.0:
                output = math.copysign(
                    max(brake_magnitude, abs(output)),
                    desired_rate,
                )
                output_floor_rpm = brake_magnitude
                output_floor_reason = "opposite_yaw"
            elif active_yaw_damping:
                # At the visual center there is no directional evidence for a
                # powered reversal. Hold zero yaw and let fresh position data
                # decide whether a true direction change is required.
                output = 0.0
                output_floor_rpm = 0.0
                output_floor_reason = "yaw_damping_coast"

        minimum_error = max(0.0, float(c.min_effective_error_deg))
        minimum_correction = max(0.0, float(c.min_effective_correction_rpm))
        absolute_error = abs(self._filtered_error_deg)
        if (
            float(c.mechanical_tier2_error_deg) > 0.0
            and absolute_error >= float(c.mechanical_tier2_error_deg)
        ):
            minimum_correction = max(
                minimum_correction,
                max(0.0, float(c.mechanical_tier2_correction_rpm)),
            )
        if (
            float(c.mechanical_tier3_error_deg) > 0.0
            and absolute_error >= float(c.mechanical_tier3_error_deg)
        ):
            minimum_correction = max(
                minimum_correction,
                max(0.0, float(c.mechanical_tier3_correction_rpm)),
            )
        release_ratio = max(0.0, min(1.0, float(c.mechanical_floor_release_ratio)))
        mechanical_response_established = bool(
            release_ratio > 0.0
            and feedback_used
            and desired_rate * measured_rate > 0.0
            and abs(measured_rate) >= max(2.0, release_ratio * abs(desired_rate))
        )
        if (
            output_floor_reason == "none"
            and minimum_correction > 0.0
            and absolute_error >= minimum_error
            and abs(output) < minimum_correction
            and not mechanical_response_established
        ):
            steering_evidence = output
            if abs(steering_evidence) < 1e-6:
                steering_evidence = desired_rate
            if abs(steering_evidence) < 1e-6:
                steering_evidence = effective_error
            if abs(steering_evidence) > 1e-6:
                floor_magnitude = min(correction_limit, minimum_correction)
                output = math.copysign(floor_magnitude, steering_evidence)
                output_floor_rpm = floor_magnitude
                output_floor_reason = "tracking"

        startup_kick_active = False
        startup_kick_elapsed_sec = 0.0
        startup_kick_release_reason = "disabled"
        startup_kick_error = max(0.0, float(c.startup_kick_error_deg))
        startup_kick_rpm = max(0.0, float(c.startup_kick_rpm))
        startup_kick_max_sec = max(0.0, float(c.startup_kick_max_sec))
        startup_kick_release_yaw = max(
            0.0,
            float(c.startup_kick_release_yaw_rate_dps),
        )
        startup_kick_enabled = bool(
            startup_kick_rpm > 0.0
            and startup_kick_max_sec > 0.0
            and requested_direction != 0
            and abs(visual_error_deg) >= startup_kick_error
        )
        braking_active = bool(
            same_direction_overspeed_braking
            or opposite_yaw_braking
            or active_yaw_damping
            or predictive_braking
        )
        startup_response_established = bool(
            feedback_used
            and requested_direction * startup_response_rate > 0.0
            and abs(startup_response_rate) >= startup_kick_release_yaw
        )
        if startup_kick_enabled:
            startup_kick_release_reason = "not_armed"
            if self._startup_kick_armed and startup_response_established:
                # The chassis is already moving in the requested direction;
                # applying a startup floor now would only add overshoot.
                self._startup_kick_armed = False
                self._startup_kick_started_at = None
                startup_kick_release_reason = "encoder_response"
            elif self._startup_kick_armed and not braking_active:
                if self._startup_kick_started_at is None:
                    self._startup_kick_started_at = now
                    self._rate_integral_deg = 0.0
                startup_kick_elapsed_sec = max(
                    0.0,
                    now - float(self._startup_kick_started_at),
                )
                if startup_kick_elapsed_sec >= startup_kick_max_sec:
                    self._startup_kick_armed = False
                    self._startup_kick_started_at = None
                    startup_kick_release_reason = "timeout"
                else:
                    kick_limit = min(
                        max(0.0, float(c.max_correction_rpm)),
                        startup_kick_rpm,
                    )
                    if max_correction_override_rpm is not None:
                        kick_limit = min(
                            kick_limit,
                            max(0.0, float(max_correction_override_rpm)),
                        )
                    correction_limit = max(correction_limit, kick_limit)
                    kick_magnitude = min(correction_limit, startup_kick_rpm)
                    if kick_magnitude > 0.0:
                        output = math.copysign(kick_magnitude, requested_direction)
                        output_floor_rpm = kick_magnitude
                        output_floor_reason = "startup_kick"
                        correction_limit_reason = "startup_kick"
                        startup_kick_active = True
                        startup_kick_release_reason = "active"
            elif self._startup_kick_armed:
                startup_kick_release_reason = "waiting_for_brake"
        elif requested_direction == 0:
            startup_kick_release_reason = "deadband"
        elif abs(visual_error_deg) < startup_kick_error:
            startup_kick_release_reason = "below_error_threshold"

        # This sign invariant is stronger than the optional center-hold guard:
        # encoder braking may reduce a command to zero, but it cannot power a
        # turn away from the side shown by the latest camera frame.
        if requested_direction != 0 and output * requested_direction < 0.0:
            output = 0.0
            output_floor_rpm = 0.0
            output_floor_reason = "visual_direction_guard"
            visual_direction_guarded = True
            self._rate_integral_deg = 0.0
        elif bool(c.visual_direction_guard_enabled):
            if requested_direction == 0:
                # Inside the visual deadband, zero yaw is an intentional coast;
                # do not let filtered history or encoder P create a reversal.
                output = 0.0
                output_floor_rpm = 0.0
                output_floor_reason = "center_hold"
                self._rate_integral_deg *= 0.5
            elif visual_direction_guarded:
                output = 0.0
                output_floor_rpm = 0.0
                output_floor_reason = "visual_direction_guard"
                self._rate_integral_deg = 0.0
            elif predictive_braking:
                output = 0.0
                output_floor_rpm = 0.0
                output_floor_reason = "predictive_brake_coast"

        if (
            abs(effective_error) <= 1e-6
            and abs(self._filtered_error_rate_dps) < 1.0
            and abs(target_rate_feedforward) < 1.0
            and (not feedback_used or abs(measured_rate) < 2.0)
        ):
            output = 0.0
            self._rate_integral_deg *= 0.5
        correction_rpm = int(round(output))

        result = VisualSteeringPidResult(
            correction_rpm=correction_rpm,
            requested_base_rpm=requested_base_rpm,
            base_rpm=limited_base_rpm,
            yaw_rate_limit_dps=yaw_rate_limit,
            correction_limit_rpm=correction_limit,
            correction_limit_reason=correction_limit_reason,
            visual_error_deg=visual_error_deg,
            compensated_error_deg=compensated_error,
            filtered_error_deg=self._filtered_error_deg,
            error_rate_dps=self._filtered_error_rate_dps,
            target_rate_valid=target_rate_valid,
            target_image_rate_dps=image_rate,
            target_bearing_rate_dps=target_bearing_rate,
            target_rate_feedforward_dps=target_rate_feedforward,
            target_speed_match_limited=target_speed_match_limited,
            target_speed_match_limit_dps=target_speed_match_limit,
            desired_yaw_rate_dps=desired_rate,
            measured_yaw_rate_dps=measured_rate,
            feedback_age_sec=feedback_age_sec,
            feedback_used=feedback_used,
            feedforward_rpm=feedforward_rpm,
            rate_p_rpm=rate_p_rpm,
            rate_i_rpm=rate_i_rpm,
            unsaturated_rpm=unsaturated,
            opposite_yaw_braking=opposite_yaw_braking,
            same_direction_overspeed_braking=same_direction_overspeed_braking,
            visual_direction_guarded=visual_direction_guarded,
            predictive_braking=predictive_braking,
            remaining_error_deg=remaining_error_deg,
            stopping_distance_deg=stopping_distance_deg,
            prediction_latency_sec=prediction_latency_sec,
            yaw_rate_overshoot_dps=yaw_rate_overshoot_dps,
            overspeed_brake_rpm=overspeed_brake_rpm,
            edge_boost_active=edge_boost_active,
            output_floor_rpm=output_floor_rpm,
            output_floor_reason=output_floor_reason,
            startup_kick_active=startup_kick_active,
            startup_kick_elapsed_sec=startup_kick_elapsed_sec,
            startup_kick_release_reason=startup_kick_release_reason,
        )
        self.last_result = result
        return result
