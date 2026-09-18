#!/usr/bin/env python3
from __future__ import annotations

import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.control_types import SteeringFeedback
from car_control_modular.steering_pid import (
    DistancePidConfig,
    LongitudinalDistancePid,
    VisualSteeringPid,
    VisualSteeringPidConfig,
    encoder_yaw_rate_right_dps,
)


def _feedback(
    rate: float,
    *,
    age_sec: float = 0.0,
    raw_rate: float | None = None,
) -> SteeringFeedback:
    return SteeringFeedback(
        timestamp=time.monotonic() - age_sec,
        yaw_rate_right_dps=rate,
        raw_yaw_rate_right_dps=raw_rate,
        trustworthy=True,
    )


def main() -> int:
    distance_pid = LongitudinalDistancePid(
        DistancePidConfig(
            kp_rpm_per_m=22.0,
            ki_rpm_per_m_s=1.5,
            kd_rpm_s_per_m=6.0,
            min_forward_output_rpm=20.0,
            max_forward_output_rpm=100.0,
            min_reverse_output_rpm=20.0,
            max_reverse_output_rpm=100.0,
        )
    )
    far = distance_pid.update(5.0, 1.5, now=1.0)
    if not 95 <= far.output_rpm <= 100:
        raise AssertionError(f"5m target must use the high-speed forward range: {far}")
    distance_pid.reset()
    approaching = distance_pid.update(1.20, 1.5, now=2.0)
    rushing = distance_pid.update(0.90, 1.5, now=2.1)
    if approaching.output_rpm > -25 or rushing.output_rpm > -35:
        raise AssertionError(
            f"close approaching target must produce prompt reverse: {approaching}, {rushing}"
        )

    left, right, yaw = encoder_yaw_rate_right_dps(
        40,
        -26,
        1,
        -1,
        0.5225,
        0.5424,
    )
    if (left, right) != (40.0, 26.0) or not 22.0 < yaw < 23.5:
        raise AssertionError(f"right-turn encoder normalization failed: {left}, {right}, {yaw}")

    left, right, yaw = encoder_yaw_rate_right_dps(
        26,
        -40,
        1,
        -1,
        0.5225,
        0.5424,
    )
    if (left, right) != (26.0, 40.0) or not -22.5 < yaw < -21.0:
        raise AssertionError(f"left-turn encoder normalization failed: {left}, {right}, {yaw}")

    cfg = VisualSteeringPidConfig(enabled=True)
    controller = VisualSteeringPid(cfg)
    centered = controller.update(0.5, 30, None, now=1.0)
    if centered.correction_rpm != 0:
        raise AssertionError(f"centered target should not turn: {centered}")

    controller.reset()
    right_error = controller.update(0.70, 30, None, now=2.0)
    if not 1 <= right_error.correction_rpm <= 12:
        raise AssertionError(f"right target must produce bounded right correction: {right_error}")

    controller.reset()
    left_error = controller.update(0.30, 30, None, now=3.0)
    if not -12 <= left_error.correction_rpm <= -1:
        raise AssertionError(f"left target must produce bounded left correction: {left_error}")

    target_rate_cfg = VisualSteeringPidConfig(
        enabled=True,
        camera_hfov_deg=60.0,
        camera_latency_sec=0.0,
        deadband_deg=0.0,
        outer_kp_per_sec=0.0,
        outer_kd_sec=0.0,
        target_rate_feedforward_gain=1.0,
        target_rate_feedforward_max_dps=30.0,
        max_correction_rpm=16.0,
        dynamic_small_max_correction_rpm=16.0,
    )
    target_rate_pid = VisualSteeringPid(target_rate_cfg)
    moving_target = target_rate_pid.update(
        0.55,
        20,
        _feedback(0.0),
        now=time.monotonic(),
        target_image_rate_dps=20.0,
    )
    if (
        abs(moving_target.target_bearing_rate_dps - 20.0) > 0.01
        or abs(moving_target.target_rate_feedforward_dps - 20.0) > 0.01
        or moving_target.desired_yaw_rate_dps < 19.9
        or moving_target.correction_rpm <= 0
    ):
        raise AssertionError(
            f"a right-side target moving outward needs right-yaw feedforward: {moving_target}"
        )

    speed_match_pid = VisualSteeringPid(
        VisualSteeringPidConfig(
            enabled=True,
            camera_hfov_deg=60.0,
            camera_latency_sec=0.0,
            deadband_deg=0.0,
            outer_kp_per_sec=2.0,
            outer_kd_sec=0.0,
            target_speed_match_max_closing_dps=8.0,
            max_yaw_rate_dps=45.0,
            max_correction_rpm=16.0,
            dynamic_small_max_correction_rpm=16.0,
        )
    )
    speed_matched = speed_match_pid.update(
        0.90,
        20,
        _feedback(18.0),
        now=time.monotonic(),
        target_image_rate_dps=-2.0,
    )
    if (
        not speed_matched.target_speed_match_limited
        or abs(speed_matched.target_speed_match_limit_dps - 28.0) > 0.01
        or abs(speed_matched.desired_yaw_rate_dps - 28.0) > 0.01
    ):
        raise AssertionError(
            "visible tracking must cap chassis closing speed relative to the target: "
            f"{speed_matched}"
        )

    target_rate_pid.reset()
    stationary_target = target_rate_pid.update(
        0.55,
        20,
        _feedback(15.0),
        now=time.monotonic(),
        target_image_rate_dps=-15.0,
    )
    if (
        abs(stationary_target.target_bearing_rate_dps) > 0.01
        or stationary_target.target_rate_feedforward_dps >= -14.9
        or stationary_target.desired_yaw_rate_dps != 0.0
        or stationary_target.correction_rpm != 0
    ):
        raise AssertionError(
            "inward image motion may brake but must not reverse away from the current side: "
            f"{stationary_target}"
        )

    target_rate_pid.reset()
    inward_with_position = VisualSteeringPid(
        VisualSteeringPidConfig(
            enabled=True,
            camera_hfov_deg=60.0,
            camera_latency_sec=0.0,
            deadband_deg=0.0,
            outer_kp_per_sec=2.0,
            outer_kd_sec=0.0,
            target_rate_feedforward_gain=1.0,
            target_rate_feedforward_max_dps=30.0,
            max_correction_rpm=16.0,
            dynamic_small_max_correction_rpm=16.0,
            visual_direction_guard_enabled=True,
        )
    ).update(
        0.60,
        20,
        _feedback(0.0),
        now=time.monotonic(),
        target_image_rate_dps=-30.0,
    )
    if inward_with_position.desired_yaw_rate_dps != 0.0 or inward_with_position.correction_rpm != 0:
        raise AssertionError(
            "an inward velocity assist must reduce same-side yaw to zero, never reverse it: "
            f"{inward_with_position}"
        )

    controller.reset()
    small_error = controller.update(0.55, 50, _feedback(0.0), now=time.monotonic())
    if abs(small_error.correction_rpm) > 7 or not 47 <= small_error.base_rpm <= 50:
        raise AssertionError(f"small error must retain most base speed and <=7rpm correction: {small_error}")
    if small_error.yaw_rate_limit_dps > 25.0 or small_error.correction_limit_rpm > 7.01:
        raise AssertionError(f"small-error dynamic limits are wrong: {small_error}")

    controller.reset()
    large_error = controller.update(0.90, 50, _feedback(0.0), now=time.monotonic())
    if not 14 <= large_error.correction_rpm <= 16:
        raise AssertionError(f"large error must allow 14-16rpm correction: {large_error}")
    if large_error.base_rpm != 28 or large_error.yaw_rate_limit_dps < 45.9:
        raise AssertionError(f"large error must cap forward base at 28rpm and allow 46dps: {large_error}")

    controller.reset()
    range_dropout = controller.update(
        0.90,
        15,
        _feedback(0.0),
        now=time.monotonic(),
        max_correction_override_rpm=16.0,
    )
    if (
        not 14 <= abs(range_dropout.correction_rpm) <= 16
        or range_dropout.base_rpm != 15
        or not range_dropout.edge_boost_active
    ):
        raise AssertionError(f"range-dropout steering must keep 15rpm base and strong correction: {range_dropout}")

    controller.reset()
    near_zero_base = controller.update(
        0.145,
        1,
        _feedback(0.0),
        now=time.monotonic(),
        max_correction_override_rpm=10.0,
    )
    if abs(near_zero_base.correction_rpm) < 8 or near_zero_base.correction_limit_rpm != 10.0:
        raise AssertionError(
            "large horizontal error must retain yaw authority at a 1rpm longitudinal base: "
            f"{near_zero_base}"
        )
    if near_zero_base.correction_limit_reason != "explicit_override":
        raise AssertionError(f"correction limit reason missing: {near_zero_base}")

    controller.reset()
    normal_reversal = controller.update(0.55, 15, _feedback(0.0), now=time.monotonic())
    controller.reset()
    opposite_reversal = controller.update(0.55, 15, _feedback(-20.0), now=time.monotonic())
    if (
        not opposite_reversal.opposite_yaw_braking
        or opposite_reversal.correction_rpm <= normal_reversal.correction_rpm
        or opposite_reversal.correction_limit_rpm < 11.5
    ):
        raise AssertionError(
            f"opposite yaw must receive stronger countersteer: normal={normal_reversal} opposite={opposite_reversal}"
        )

    bounded_brake_cfg = VisualSteeringPidConfig(
        enabled=True,
        braking_max_correction_rpm=8.0,
        active_brake_min_correction_rpm=6.0,
    )
    bounded_brake = VisualSteeringPid(bounded_brake_cfg).update(
        0.70,
        15,
        _feedback(-20.0),
        now=time.monotonic(),
    )
    if (
        not bounded_brake.opposite_yaw_braking
        or bounded_brake.correction_limit_rpm > 8.01
        or abs(bounded_brake.correction_rpm) > 8
    ):
        raise AssertionError(
            "countersteer must stay below the dedicated braking cap: "
            f"{bounded_brake}"
        )

    fast_countersteer_cfg = VisualSteeringPidConfig(
        enabled=True,
        braking_max_correction_rpm=6.0,
        fast_countersteer_max_correction_rpm=10.0,
        fast_countersteer_gain_rpm_per_dps=0.08,
    )
    fast_countersteer = VisualSteeringPid(fast_countersteer_cfg).update(
        0.70,
        15,
        _feedback(-40.0),
        now=time.monotonic(),
    )
    if (
        not fast_countersteer.opposite_yaw_braking
        or not 9.1 <= fast_countersteer.correction_limit_rpm <= 9.3
        or not 8 <= fast_countersteer.correction_rpm <= 10
    ):
        raise AssertionError(
            "explicit fast countersteer must scale with residual yaw but remain bounded: "
            f"{fast_countersteer}"
        )

    edge_bounded_brake_cfg = VisualSteeringPidConfig(
        enabled=True,
        max_correction_rpm=18.0,
        braking_max_correction_rpm=8.0,
        active_brake_min_correction_rpm=6.0,
        edge_boost_start_error_deg=8.0,
    )
    edge_bounded_brake = VisualSteeringPid(edge_bounded_brake_cfg).update(
        0.10,
        20,
        _feedback(20.0),
        now=time.monotonic(),
    )
    if (
        not edge_bounded_brake.opposite_yaw_braking
        or edge_bounded_brake.correction_limit_rpm > 8.01
        or abs(edge_bounded_brake.correction_rpm) > 8
    ):
        raise AssertionError(
            "edge boost must not override the countersteer braking cap: "
            f"{edge_bounded_brake}"
        )

    controller.reset()
    same_direction_overspeed = controller.update(
        0.66,
        15,
        _feedback(47.0),
        now=time.monotonic(),
    )
    if (
        not same_direction_overspeed.same_direction_overspeed_braking
        or same_direction_overspeed.yaw_rate_overshoot_dps < 8.0
        or same_direction_overspeed.overspeed_brake_rpm <= 0.0
        or same_direction_overspeed.correction_rpm != 0
        or same_direction_overspeed.output_floor_reason != "same_direction_overspeed_coast"
    ):
        raise AssertionError(
            "same-direction yaw overspeed must coast without powered reversal: "
            f"{same_direction_overspeed}"
        )

    unavailable_rate_cfg = VisualSteeringPidConfig(
        enabled=True,
        camera_hfov_deg=60.0,
        deadband_deg=3.0,
        outer_kp_per_sec=1.65,
        target_rate_feedforward_gain=0.30,
        target_rate_feedforward_max_dps=10.0,
        max_yaw_rate_dps=36.0,
    )
    unavailable_rate = VisualSteeringPid(unavailable_rate_cfg).update(
        0.75,
        0,
        _feedback(40.0),
        now=time.monotonic(),
        target_image_rate_dps=None,
    )
    if (
        unavailable_rate.target_rate_valid
        or unavailable_rate.target_bearing_rate_dps != 0.0
        or unavailable_rate.target_rate_feedforward_dps != 0.0
    ):
        raise AssertionError(
            "an unavailable image-rate sample must not turn chassis yaw into target feedforward: "
            f"{unavailable_rate}"
        )

    # Regression from the 2026-08-25 run: with the old 130ms compensation and
    # 6-degree deadband, x=0.705/0.638 produced a left brake while the person
    # was still visibly right of center. The tuned runtime loop must retain a
    # rightward command until the target actually reaches the center band.
    center_lock_cfg = VisualSteeringPidConfig(
        enabled=True,
        camera_hfov_deg=60.0,
        camera_latency_sec=0.08,
        deadband_deg=3.0,
        outer_kp_per_sec=2.40,
        outer_kd_sec=0.10,
        max_yaw_rate_dps=48.0,
        rate_kp_rpm_per_dps=0.18,
        rate_ki_rpm_per_deg=0.015,
        max_correction_rpm=18.0,
        dynamic_small_error_deg=3.0,
        dynamic_large_error_deg=12.0,
        dynamic_small_max_yaw_rate_dps=24.0,
        dynamic_small_max_correction_rpm=7.0,
        same_direction_overspeed_threshold_dps=10.0,
        same_direction_overspeed_brake_gain_rpm_per_dps=0.20,
        error_filter_alpha=0.70,
        derivative_filter_alpha=0.30,
    )
    center_lock = VisualSteeringPid(center_lock_cfg)
    frame_682 = center_lock.update(
        0.705,
        13,
        _feedback(19.53),
        now=time.monotonic(),
        max_correction_override_rpm=12.0,
    )
    if frame_682.correction_rpm <= 0 or frame_682.same_direction_overspeed_braking:
        raise AssertionError(
            "a right-side target must not receive a premature left brake: "
            f"{frame_682}"
        )

    center_lock.reset()
    frame_683 = center_lock.update(
        0.638,
        13,
        _feedback(17.90),
        now=time.monotonic(),
        max_correction_override_rpm=12.0,
    )
    if frame_683.correction_rpm < 0 or frame_683.same_direction_overspeed_braking:
        raise AssertionError(
            "x=0.638 must keep centering right or coast, not reverse left: "
            f"{frame_683}"
        )

    growing_error_cfg = VisualSteeringPidConfig(
        enabled=True,
        camera_hfov_deg=60.0,
        camera_latency_sec=0.0,
        deadband_deg=6.0,
        outer_kp_per_sec=1.0,
        outer_kd_sec=0.0,
        max_yaw_rate_dps=46.0,
        max_correction_rpm=40.0,
        dynamic_small_error_deg=6.0,
        dynamic_large_error_deg=20.0,
        dynamic_small_max_correction_rpm=20.0,
        same_direction_overspeed_threshold_dps=5.0,
    )
    growing_error = VisualSteeringPid(growing_error_cfg)
    growing_now = time.monotonic()
    growing_error.update(0.62, 0, _feedback(20.0), now=growing_now)
    outward = growing_error.update(0.75, 0, _feedback(20.0), now=growing_now + 0.10)
    if outward.same_direction_overspeed_braking or outward.overspeed_brake_rpm != 0.0:
        raise AssertionError(
            "same-direction overspeed braking must not reverse yaw while the visual "
            f"error is still growing toward the edge: {outward}"
        )

    mechanical_cfg = VisualSteeringPidConfig(
        enabled=True,
        camera_hfov_deg=60.0,
        deadband_deg=6.0,
        max_correction_rpm=16.0,
        dynamic_small_error_deg=3.5,
        dynamic_large_error_deg=14.0,
        dynamic_small_max_correction_rpm=6.0,
        same_direction_overspeed_threshold_dps=5.0,
        min_effective_error_deg=6.5,
        min_effective_correction_rpm=8.0,
        active_brake_yaw_threshold_dps=3.0,
        active_brake_min_correction_rpm=10.0,
    )
    controller = VisualSteeringPid(mechanical_cfg)
    effective_floor = controller.update(
        0.62,
        20,
        _feedback(0.0),
        now=time.monotonic(),
        max_correction_override_rpm=16.0,
    )
    if effective_floor.correction_rpm < 8 or effective_floor.output_floor_reason != "tracking":
        raise AssertionError(
            "a target moving outside the center band must clear the mechanical dead zone: "
            f"{effective_floor}"
        )

    tiered_cfg = VisualSteeringPidConfig(
        enabled=True,
        camera_hfov_deg=60.0,
        deadband_deg=6.0,
        max_correction_rpm=40.0,
        dynamic_small_error_deg=6.0,
        dynamic_large_error_deg=20.0,
        dynamic_small_max_correction_rpm=20.0,
        min_effective_error_deg=6.0,
        min_effective_correction_rpm=20.0,
        mechanical_tier2_error_deg=13.0,
        mechanical_tier2_correction_rpm=30.0,
        mechanical_tier3_error_deg=20.0,
        mechanical_tier3_correction_rpm=40.0,
        mechanical_floor_release_ratio=0.70,
        active_brake_yaw_threshold_dps=3.0,
        active_brake_min_correction_rpm=20.0,
    )
    tiered = VisualSteeringPid(tiered_cfg)
    tier1 = tiered.update(0.62, 40, _feedback(0.0), now=time.monotonic())
    tiered.reset()
    tier2 = tiered.update(0.75, 40, _feedback(0.0), now=time.monotonic())
    tiered.reset()
    tier3 = tiered.update(0.90, 40, _feedback(0.0), now=time.monotonic())
    if (tier1.correction_rpm, tier2.correction_rpm, tier3.correction_rpm) != (20, 30, 40):
        raise AssertionError(
            f"mechanical tracking tiers must be 20/30/40rpm: {tier1}, {tier2}, {tier3}"
        )

    tiered.reset()
    response_established = tiered.update(
        0.75,
        40,
        _feedback(12.0),
        now=time.monotonic(),
    )
    if (
        abs(response_established.correction_rpm) >= 20
        or response_established.output_floor_reason != "none"
    ):
        raise AssertionError(
            "mechanical tier must release after encoder yaw response is established: "
            f"{response_established}"
        )

    startup_cfg = VisualSteeringPidConfig(
        enabled=True,
        camera_hfov_deg=60.0,
        deadband_deg=6.0,
        max_correction_rpm=40.0,
        dynamic_small_error_deg=6.0,
        dynamic_large_error_deg=20.0,
        dynamic_small_max_correction_rpm=20.0,
        startup_kick_error_deg=6.0,
        startup_kick_rpm=30.0,
        startup_kick_max_sec=0.60,
        startup_kick_release_yaw_rate_dps=4.0,
    )
    startup = VisualSteeringPid(startup_cfg)
    kick_now = time.monotonic()
    kick_started = startup.update(0.75, 20, _feedback(0.0), now=kick_now)
    if (
        kick_started.correction_rpm != 30
        or not kick_started.startup_kick_active
        or kick_started.output_floor_reason != "startup_kick"
    ):
        raise AssertionError(f"initial steering error must get one startup kick: {kick_started}")

    kick_released = startup.update(0.75, 20, _feedback(5.0), now=kick_now + 0.10)
    if (
        kick_released.startup_kick_active
        or kick_released.startup_kick_release_reason != "encoder_response"
        or abs(kick_released.correction_rpm) >= 30
    ):
        raise AssertionError(
            f"encoder yaw response must immediately return control to continuous PID: {kick_released}"
        )

    startup.reset()
    raw_kick_now = time.monotonic()
    startup.update(0.75, 20, _feedback(0.0, raw_rate=0.0), now=raw_kick_now)
    raw_kick_released = startup.update(
        0.75,
        20,
        _feedback(0.0, raw_rate=5.0),
        now=raw_kick_now + 0.10,
    )
    if (
        raw_kick_released.startup_kick_active
        or raw_kick_released.startup_kick_release_reason != "encoder_response"
    ):
        raise AssertionError(
            "the first raw encoder response must release startup kick before the median: "
            f"{raw_kick_released}"
        )

    no_retrigger = startup.update(0.75, 20, _feedback(0.0), now=kick_now + 0.20)
    if no_retrigger.startup_kick_active or abs(no_retrigger.correction_rpm) >= 30:
        raise AssertionError(
            f"same-side visual error must not retrigger a consumed startup kick: {no_retrigger}"
        )

    startup.update(0.50, 20, _feedback(0.0), now=kick_now + 0.30)
    rearmed = startup.update(0.25, 20, _feedback(0.0), now=kick_now + 0.40)
    if not rearmed.startup_kick_active or rearmed.correction_rpm != -30:
        raise AssertionError(
            f"returning to center must rearm one kick for the next direction: {rearmed}"
        )

    startup.reset()
    timeout_started = startup.update(0.75, 20, _feedback(0.0), now=kick_now + 1.0)
    timeout_released = startup.update(0.75, 20, _feedback(0.0), now=kick_now + 1.61)
    if (
        not timeout_started.startup_kick_active
        or timeout_released.startup_kick_active
        or timeout_released.startup_kick_release_reason != "timeout"
    ):
        raise AssertionError(
            f"startup kick must have a hard time limit even without wheel response: {timeout_released}"
        )

    controller.reset()
    active_overspeed_brake = controller.update(
        0.82,
        20,
        _feedback(35.0),
        now=time.monotonic(),
        max_correction_override_rpm=16.0,
    )
    if (
        active_overspeed_brake.correction_rpm != 0
        or active_overspeed_brake.output_floor_reason != "same_direction_overspeed_coast"
    ):
        raise AssertionError(
            "same-direction overspeed must publish zero yaw without reversing: "
            f"{active_overspeed_brake}"
        )

    controller.reset()
    active_opposite_brake = controller.update(
        0.75,
        20,
        _feedback(-15.0),
        now=time.monotonic(),
        max_correction_override_rpm=16.0,
    )
    if (
        active_opposite_brake.correction_rpm < 10
        or active_opposite_brake.output_floor_reason != "opposite_yaw"
    ):
        raise AssertionError(
            "old opposite yaw must be cancelled with an effective counter-command: "
            f"{active_opposite_brake}"
        )

    controller.reset()
    center_active_brake = controller.update(
        0.50,
        20,
        _feedback(25.0),
        now=time.monotonic(),
        max_correction_override_rpm=16.0,
    )
    if (
        center_active_brake.correction_rpm != 0
        or center_active_brake.output_floor_reason != "yaw_damping_coast"
    ):
        raise AssertionError(
            "residual yaw at the visual center must coast without blind reversal: "
            f"{center_active_brake}"
        )

    controller.reset()
    now = time.monotonic()
    damping = controller.update(0.5, 30, _feedback(25.0), now=now)
    if damping.correction_rpm != 0 or not damping.feedback_used:
        raise AssertionError(f"right-turn inertia at center must request zero-yaw damping: {damping}")

    # Regression from the 2026-09-02 stationary-target run. While the target
    # remained left of center, encoder overspeed used to emit +6 RPM and then
    # return to -5 RPM about 100 ms later, continuously increasing the swing.
    stationary_cfg = VisualSteeringPidConfig(
        enabled=True,
        camera_hfov_deg=60.0,
        camera_latency_sec=0.10,
        deadband_deg=3.0,
        outer_kp_per_sec=2.10,
        outer_kd_sec=0.07,
        max_yaw_rate_dps=45.0,
        rate_kp_rpm_per_dps=0.32,
        rate_ki_rpm_per_deg=0.006,
        max_correction_rpm=12.0,
        dynamic_small_error_deg=3.0,
        dynamic_large_error_deg=10.0,
        dynamic_small_max_yaw_rate_dps=15.0,
        dynamic_small_max_correction_rpm=5.0,
        same_direction_overspeed_threshold_dps=4.0,
        same_direction_overspeed_brake_gain_rpm_per_dps=0.35,
        active_brake_yaw_threshold_dps=7.0,
        active_brake_min_correction_rpm=6.0,
    )
    stationary = VisualSteeringPid(stationary_cfg)
    replay_now = time.monotonic()
    stationary.update(0.375, 17, _feedback(-7.84), now=replay_now)
    left_overspeed = stationary.update(
        0.392,
        17,
        _feedback(-45.0),
        now=replay_now + 0.09,
        target_image_rate_dps=45.0,
        max_correction_override_rpm=6.0,
    )
    if (
        not left_overspeed.same_direction_overspeed_braking
        or left_overspeed.correction_rpm != 0
        or left_overspeed.target_bearing_rate_dps != 0.0
    ):
        raise AssertionError(
            "same-side stationary-target overspeed must cancel feedforward and coast: "
            f"{left_overspeed}"
        )

    crossed = stationary.update(
        0.58,
        17,
        _feedback(-20.0),
        now=replay_now + 0.18,
        target_image_rate_dps=20.0,
        max_correction_override_rpm=6.0,
    )
    if crossed.correction_rpm <= 0 or not crossed.opposite_yaw_braking:
        raise AssertionError(
            "fresh vision crossing to the right must still permit right countersteer: "
            f"{crossed}"
        )

    # Runtime aiming mode: fresh visual position owns direction, braking uses
    # zero yaw, and a genuine small correction may run continuously at 1 RPM.
    aim_cfg = VisualSteeringPidConfig(
        enabled=True,
        camera_hfov_deg=60.0,
        camera_latency_sec=0.0,
        deadband_deg=3.0,
        outer_kp_per_sec=2.10,
        outer_kd_sec=1.0,
        target_rate_feedforward_gain=0.0,
        target_rate_feedforward_max_dps=0.0,
        max_correction_rpm=12.0,
        dynamic_small_max_correction_rpm=5.0,
        visual_direction_guard_enabled=True,
        predictive_brake_decel_dps2=60.0,
        predictive_brake_margin_deg=1.25,
        predictive_brake_response_sec=0.05,
        min_effective_error_deg=3.0,
        min_effective_correction_rpm=1.0,
    )
    aim = VisualSteeringPid(aim_cfg)
    aim.update(0.80, 0, _feedback(0.0), now=10.0)
    guarded = aim.update(0.60, 0, _feedback(0.0), now=10.1)
    if guarded.correction_rpm != 0 or not guarded.visual_direction_guarded:
        raise AssertionError(
            "a right-side target must turn a stale left derivative into zero yaw: "
            f"{guarded}"
        )

    aim.reset()
    one_rpm = aim.update(0.552, 0, None, now=11.0)
    if one_rpm.correction_rpm != 1 or one_rpm.output_floor_reason != "tracking":
        raise AssertionError(
            f"a small right-side aiming error must use continuous 1 RPM: {one_rpm}"
        )

    aim.reset()
    predictive = aim.update(
        0.72,
        0,
        _feedback(30.0),
        now=time.monotonic(),
        visual_age_sec=0.10,
    )
    if (
        predictive.correction_rpm != 0
        or not predictive.predictive_braking
        or predictive.output_floor_reason != "predictive_brake_coast"
        or predictive.prediction_latency_sec < 0.149
        or predictive.stopping_distance_deg < 11.9
    ):
        raise AssertionError(
            "visual and motor latency travel must be included in predictive braking: "
            f"{predictive}"
        )

    # Filtered derivative can still say the error is growing for one update
    # after the raw target has started returning to center. Stopping distance
    # must remain authoritative or braking is delayed by a full vision frame.
    lagged = VisualSteeringPid(aim_cfg)
    lagged.update(0.60, 0, _feedback(0.0), now=12.00)
    lagged.update(0.78, 0, _feedback(0.0), now=12.10)
    lagged_brake = lagged.update(
        0.64,
        0,
        _feedback(18.0),
        now=12.20,
        visual_age_sec=0.10,
    )
    if lagged_brake.correction_rpm != 0 or not lagged_brake.predictive_braking:
        raise AssertionError(
            "stopping distance must brake even while filtered derivative lags: "
            f"{lagged_brake}"
        )

    outward = VisualSteeringPid(aim_cfg).update(
        0.72,
        0,
        _feedback(18.0),
        now=13.00,
        target_image_rate_dps=20.0,
        visual_age_sec=0.10,
    )
    if outward.predictive_braking or outward.correction_rpm <= 0:
        raise AssertionError(
            "a target still moving outward in the latest image must retain yaw authority: "
            f"{outward}"
        )

    # Frame 69 regression: the target is still just right of the aim line,
    # but its measured inward motion will carry it into the deadband before
    # the current image and motor command can take effect. Continuing to power
    # the old right turn here adds avoidable latency before the left response.
    returning = VisualSteeringPid(aim_cfg).update(
        0.570,
        0,
        _feedback(0.0),
        now=13.10,
        target_image_rate_dps=-7.87,
        visual_age_sec=0.106,
    )
    if returning.correction_rpm != 0 or returning.output_floor_reason != "target_return_coast":
        raise AssertionError(
            "an inward-moving target predicted to reach the deadband must cancel "
            f"the old-direction drive: {returning}"
        )

    bounded_reversal_cfg = VisualSteeringPidConfig(
        enabled=True,
        camera_hfov_deg=60.0,
        camera_latency_sec=0.10,
        deadband_deg=3.0,
        outer_kp_per_sec=2.10,
        outer_kd_sec=0.0,
        max_correction_rpm=12.0,
        dynamic_small_max_correction_rpm=5.0,
        opposite_yaw_brake_threshold_dps=5.0,
        braking_max_correction_rpm=3.0,
        fast_countersteer_max_correction_rpm=3.0,
        fast_countersteer_gain_rpm_per_dps=0.0,
        visual_direction_guard_enabled=True,
        predictive_brake_decel_dps2=60.0,
        predictive_brake_margin_deg=1.25,
        predictive_brake_response_sec=0.05,
        min_effective_error_deg=3.0,
        min_effective_correction_rpm=1.0,
    )
    bounded_reversal = VisualSteeringPid(bounded_reversal_cfg).update(
        0.40,
        0,
        _feedback(30.0),
        now=time.monotonic(),
        visual_age_sec=0.10,
    )
    if (
        not bounded_reversal.opposite_yaw_braking
        or bounded_reversal.correction_rpm >= 0
        or abs(bounded_reversal.correction_rpm) > 3
        or bounded_reversal.correction_limit_rpm > 3.01
    ):
        raise AssertionError(
            "a fresh center crossing must use at most 3 RPM until residual yaw "
            f"falls below 5 dps: {bounded_reversal}"
        )

    # A target crossing back to the already-moving side must not immediately
    # re-arm that same rotation. The controller should coast until the
    # measured yaw is quiet for two feedback samples.
    settle_cfg = VisualSteeringPidConfig(
        enabled=True,
        camera_hfov_deg=60.0,
        camera_latency_sec=0.0,
        deadband_deg=3.0,
        outer_kp_per_sec=2.0,
        max_correction_rpm=10.0,
        dynamic_small_max_correction_rpm=5.0,
        opposite_yaw_brake_threshold_dps=3.0,
        braking_max_correction_rpm=5.0,
        visual_direction_guard_enabled=True,
    )
    settle = VisualSteeringPid(settle_cfg)
    settle.update(0.70, 0, _feedback(20.0), now=20.0)
    settle.update(0.40, 0, _feedback(20.0), now=20.1)
    held = settle.update(0.70, 0, _feedback(20.0), now=20.2)
    if held.correction_rpm != 0 or held.output_floor_reason != "reversal_settle":
        raise AssertionError(f"same-side reversal must wait for yaw to settle: {held}")
    settle.update(0.70, 0, _feedback(1.0), now=20.3)
    released = settle.update(0.70, 0, _feedback(1.0), now=20.4)
    if released.correction_rpm <= 0 or released.output_floor_reason == "reversal_settle":
        raise AssertionError(f"reversal gate should release after two quiet samples: {released}")

    controller.reset()
    stale = controller.update(0.5, 30, _feedback(25.0, age_sec=1.0))
    if stale.correction_rpm != 0 or stale.feedback_used:
        raise AssertionError(f"stale feedback must fall back to camera-only control: {stale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
