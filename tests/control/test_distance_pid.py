#!/usr/bin/env python3
from __future__ import annotations

import math

from car_control_modular.steering_pid import DistancePidConfig, LongitudinalDistancePid
from car_control_modular.control_types import PersonTarget, SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


def main() -> int:
    pid = LongitudinalDistancePid(
        DistancePidConfig(
            kp_rpm_per_m=9.0,
            ki_rpm_per_m_s=1.0,
            kd_rpm_s_per_m=1.5,
            integral_limit_m_s=1.5,
            deadband_m=0.005,
            min_forward_output_rpm=20.0,
            max_forward_output_rpm=50.0,
            min_reverse_output_rpm=8.0,
            max_reverse_output_rpm=15.0,
        )
    )

    at_target = pid.update(1.500, 1.500, now=0.0)
    near = pid.update(1.510, 1.500, now=0.1)
    far = pid.update(4.000, 1.500, now=0.2)
    reverse = pid.update(1.400, 1.500, now=0.3)
    reverse_fast = pid.update(0.500, 1.500, now=0.4)

    if at_target.output_rpm != 0:
        raise AssertionError(f"target distance must stop: {at_target}")
    if not 20 <= near.output_rpm <= 50:
        raise AssertionError(f"positive distance error must command forward RPM: {near}")
    if far.output_rpm <= near.output_rpm or far.output_rpm != 50:
        raise AssertionError(f"far distance must reach the configured cap: near={near} far={far}")
    if not -15 <= reverse.output_rpm <= -8:
        raise AssertionError(f"negative distance error must command protected reverse RPM: {reverse}")
    if reverse_fast.output_rpm != -15:
        raise AssertionError(f"reverse output must be capped: {reverse_fast}")
    if not all(math.isfinite(value) for value in (far.error_rate_m_s, far.p_rpm, far.i_rpm, far.d_rpm)):
        raise AssertionError(f"PID terms must remain finite: {far}")

    guarded = LongitudinalDistancePid(
        DistancePidConfig(
            kp_rpm_per_m=24.0,
            ki_rpm_per_m_s=0.8,
            kd_rpm_s_per_m=2.0,
            deadband_m=0.15,
            min_forward_output_rpm=20.0,
            max_forward_output_rpm=100.0,
            max_measurement_jump_m=0.80,
            output_rise_rpm_per_sec=180.0,
            output_fall_rpm_per_sec=300.0,
            derivative_filter_alpha=0.12,
        )
    )
    guarded.update(2.46, 1.5, now=0.0)
    jump = guarded.update(5.83, 1.5, now=0.1)
    if not jump.measurement_jump_clamped or jump.actual_distance_m > 3.27:
        raise AssertionError(f"depth jump must be clamped before PID: {jump}")
    if not jump.output_slew_limited or jump.output_rpm >= jump.unslewed_output_rpm:
        raise AssertionError(f"distance output must slew-limit a jump: {jump}")
    transition_pid = LongitudinalDistancePid(
        DistancePidConfig(
            kp_rpm_per_m=24.0,
            min_forward_output_rpm=20.0,
            max_forward_output_rpm=100.0,
            min_reverse_output_rpm=20.0,
            max_reverse_output_rpm=100.0,
            output_rise_rpm_per_sec=180.0,
            output_fall_rpm_per_sec=300.0,
        )
    )
    transition_pid.update(4.0, 1.5, now=0.0)
    reverse_transition = transition_pid.update(0.5, 1.5, now=0.1)
    if reverse_transition.output_rpm >= 0:
        raise AssertionError(f"distance sign reversal must not slew through forward RPM: {reverse_transition}")

    controller = FollowSafetyController(
        FollowPolicyConfig(
            distance_pid_enable=True,
            target_distance_m=1.5,
            brake_distance_m=0.5,
            initial_target_confirm_frames=1,
        )
    )
    person = PersonTarget((250, 100, 390, 420), 1, 0.9, 44800)
    near_follow = controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[person], distance_m=1.51),
    )
    far_follow = controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[person], distance_m=4.0),
    )
    if not near_follow.actions or near_follow.actions[0].kind != "forward":
        raise AssertionError(f"controller must pass positive PID output to DRIVE: {near_follow}")
    if far_follow.actions[0].speed_percent <= near_follow.actions[0].speed_percent:
        raise AssertionError(f"controller RPM target must rise with distance: {near_follow} {far_follow}")

    print("distance_pid: signed outer-loop PID, controller integration and bounds passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
