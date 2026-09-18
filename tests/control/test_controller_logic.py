#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.control_types import (
    ControlAction,
    DistanceState,
    HazardState,
    ObstacleState,
    PersonTarget,
    SensorFrame,
    SteeringFeedback,
)
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


def check_lateral_refresh_policy() -> None:
    cfg = FollowPolicyConfig(
        visible_steering_pid_enable=True,
        visible_steering_pid_camera_hfov_deg=60.0,
        visible_steering_pid_camera_latency_sec=0.0,
        visible_steering_pid_deadband_deg=3.0,
        visible_steering_pid_outer_kp_per_sec=1.50,
        visible_steering_pid_outer_kd_sec=0.0,
        visible_steering_pid_rate_kp_rpm_per_dps=0.32,
        visible_steering_pid_rate_ki_rpm_per_deg=0.0,
        visible_steering_pid_error_filter_alpha=1.0,
        visible_steering_pid_target_rate_feedforward_gain=0.30,
        visible_steering_pid_target_rate_feedforward_max_dps=8.0,
        visible_steering_pid_target_speed_match_max_closing_dps=8.0,
        visible_steering_pid_startup_kick_error_deg=0.0,
        visible_steering_pid_startup_kick_rpm=3.0,
        visible_steering_pid_startup_kick_max_sec=0.10,
        visible_steering_pid_startup_kick_release_yaw_rate_dps=2.0,
        center_left_ratio=0.45,
        center_right_ratio=0.55,
        steer_release_left_ratio=0.47,
        steer_release_right_ratio=0.53,
        parked_recenter_min_rpm=5,
        parked_recenter_max_rpm=14,
        near_distance_rotation_only_max_rpm=10,
        near_distance_disable_rate_feedforward=True,
    )

    def parked_sample(controller, x_ratio, now, *, yaw=0.0, rate=None, limit=10.0):
        cx = x_ratio * 640.0
        target = PersonTarget((cx - 40, 80, cx + 40, 430), 501, 0.9, 28000)
        action = controller._pid_action_for_parked_target(
            target,
            SensorFrame(
                width=640,
                height=480,
                persons=[target],
                distance_m=1.50,
                steering_feedback=SteeringFeedback(
                    timestamp=now, yaw_rate_right_dps=yaw, trustworthy=True,
                ),
            ),
            now,
            "right" if x_ratio > 0.5 else "left",
            target_image_rate_dps=rate,
            max_correction_rpm=limit,
            near_distance_mode=True,
        )
        return action, controller.last_steering_pid_result

    # A near-mode decision and later encoder ticks must keep the same
    # feedforward and closing-rate policy. Exercise each side and an image
    # rate that would otherwise activate each global constraint.
    for side in (-1, 1):
        for image_rate in (0.0, 10.0):
            controller = FollowSafetyController(cfg)
            x_ratio = 0.5 + side * 0.25
            _action, initial = parked_sample(
                controller, x_ratio, 10.0, rate=side * image_rate,
            )
            assert initial is not None
            for tick in (10.04, 10.08, 10.12):
                result = controller.refresh_parked_lateral_pid(
                    x_ratio=x_ratio,
                    base_rpm=initial.base_rpm,
                    feedback=SteeringFeedback(
                        timestamp=tick, yaw_rate_right_dps=0.0, trustworthy=True,
                    ),
                    now=tick,
                    target_image_rate_dps=side * image_rate,
                    max_correction_rpm=initial.correction_limit_rpm,
                    near_distance_mode=True,
                )
                assert result.target_rate_feedforward_dps == 0.0, result
                assert not result.target_speed_match_limited, result
                assert result.desired_yaw_rate_dps == initial.desired_yaw_rate_dps, result
            # Explicit hold remains zero for this same bbox, even after the
            # kick timeout. A fresh nonzero intent can subsequently start.
            for tick in (10.16, 10.30):
                held = controller.refresh_parked_lateral_pid(
                    x_ratio=x_ratio, base_rpm=17, feedback=None, now=tick,
                    target_image_rate_dps=side * 10.0,
                    near_distance_mode=True, hold_zero=True,
                )
                assert held.correction_rpm == 0 and not held.startup_kick_active, held
            resumed = controller.refresh_parked_lateral_pid(
                x_ratio=x_ratio, base_rpm=17, feedback=None, now=10.34,
                near_distance_mode=True,
            )
            assert resumed.correction_rpm * side > 0, resumed

    # Outward image motion can select a side while the near-mode position
    # loop requests zero. That is not permission for a startup kick.
    controller = FollowSafetyController(cfg)
    action, centered = parked_sample(controller, 0.54, 20.0, rate=8.0)
    assert action is None and centered is not None, (action, centered)
    assert centered.desired_yaw_rate_dps == 0.0 and centered.correction_rpm == 0, centered
    assert not centered.startup_kick_active, centered
    centered_refresh = controller.refresh_parked_lateral_pid(
        x_ratio=0.54, base_rpm=17, feedback=None, now=20.04,
        target_image_rate_dps=8.0, near_distance_mode=True,
    )
    assert centered_refresh.correction_rpm == 0 and not centered_refresh.startup_kick_active
    ordinary_refresh = FollowSafetyController(cfg).refresh_parked_lateral_pid(
        x_ratio=0.75, base_rpm=17, feedback=None, now=21.0,
        target_image_rate_dps=10.0,
    )
    assert ordinary_refresh.target_rate_feedforward_dps == 3.0, ordinary_refresh
    # At 15 degrees the approach phase allows 12 dps relative closing speed;
    # target rate 10 + closing 12 = 22, so the 21 dps PID need not be clipped.
    assert abs(ordinary_refresh.target_speed_match_limit_dps - 22.0) < 1e-6, ordinary_refresh
    assert abs(ordinary_refresh.desired_yaw_rate_dps) <= 22.0, ordinary_refresh

    # The 5 RPM launch floor ends as soon as the encoder confirms motion;
    # fine tracking retains 1 RPM and even launch must respect a 2 RPM cap.
    for yaw, limit, expected in ((0.0, 10.0, 5), (2.5, 10.0, 1), (0.0, 2.0, 2)):
        controller = FollowSafetyController(cfg)
        action, initial = parked_sample(controller, 0.58, 30.0, yaw=yaw, limit=limit)
        assert action is not None and initial.correction_rpm == expected, (action, initial)
        refreshed = controller.refresh_parked_lateral_pid(
            x_ratio=0.58, base_rpm=initial.base_rpm,
            feedback=SteeringFeedback(timestamp=30.04, yaw_rate_right_dps=yaw, trustworthy=True),
            now=30.04, max_correction_rpm=initial.correction_limit_rpm,
            near_distance_mode=True,
        )
        assert refreshed.correction_rpm == expected, refreshed


def main() -> int:
    check_lateral_refresh_policy()
    cfg = FollowPolicyConfig(
        max_forward_percent=20,
        forward_speed_le_1_3_percent=12,
        forward_speed_le_1_7_percent=16,
        forward_speed_le_2_1_percent=20,
        forward_speed_le_2_6_percent=20,
        forward_speed_le_3_2_percent=20,
        forward_speed_le_3_8_percent=20,
        forward_speed_le_4_5_percent=20,
        forward_speed_far_percent=20,
    )
    controller = FollowSafetyController(cfg)
    person = PersonTarget((250, 120, 390, 430), track_id=1, confidence=0.9, area=43400)

    normal = SensorFrame(width=640, height=480, persons=[person], distance_m=2.0)
    d1 = controller.decide(1, normal)
    print("normal:", [(a.kind, a.speed_percent, a.reason) for a in d1.actions], d1.explicit_stop_requested, d1.reason)
    if not d1.actions:
        raise AssertionError("normal target should produce a control action")
    if max(a.speed_percent for a in d1.actions) > 20:
        raise AssertionError(f"speed cap exceeded: {d1.actions}")

    # A detector fragment without usable geometry publishes a soft zero-yaw
    # hold without advancing the lost-confirm counter or brake-hold state.
    low_quality = controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[], distance_m=2.46),
        low_quality_visible=True,
    )
    if (
        low_quality.reason != "target_visible_low_quality_hold"
        or low_quality.explicit_stop_requested
        or not low_quality.soft_stop_requested
        or len(low_quality.actions) != 1
        or low_quality.actions[0].kind != "stop"
        or low_quality.actions[0].brake_hold
        or controller.lost_confirm_frames != 0
        or controller.active_target_id != 1
    ):
        raise AssertionError(f"low-quality visible hold failed: {low_quality}")

    search_low_quality_controller = FollowSafetyController(
        FollowPolicyConfig(lost_confirm_frames=2)
    )
    search_low_quality_controller.active_target_id = 1
    search_low_quality_controller._has_seen_person = True
    search_low_quality_controller.search_state = "searching"
    search_low_quality_controller.search_direction = "right"
    search_low_quality_controller.lost_confirm_frames = 3
    search_low_quality_controller._lost_started_at = time.monotonic() - 0.5
    search_low_quality_controller._search_rotation_started_at = time.monotonic() - 0.2
    search_low_quality_controller._search_rotation_accumulated_deg = 135.0
    search_hold = search_low_quality_controller.decide(
        3,
        SensorFrame(width=640, height=480, persons=[], distance_m=2.46),
        low_quality_visible=True,
    )
    if (
        search_hold.reason != "target_visible_low_quality_search_hold"
        or not search_hold.actions
        or search_hold.actions[0].kind != "stop"
        or search_hold.actions[0].brake_hold
        or search_hold.explicit_stop_requested
        or search_low_quality_controller.search_state != "searching"
        or search_low_quality_controller.search_direction != "right"
        or search_low_quality_controller.lost_confirm_frames != 3
        or search_low_quality_controller._search_rotation_accumulated_deg != 135.0
    ):
        raise AssertionError(f"low-quality search must pause blind rotation without hard stop: {search_hold}")

    # Board policy: within 1.80m lateral correction is in-place yaw only;
    # forward follow resumes only after the upper hysteresis boundary.
    near_cfg = FollowPolicyConfig(
        reverse_enable=True,
        near_distance_rotate_only_enable=True,
        near_distance_rotate_only_distance_m=1.80,
        distance_parking_enable=False,
        center_left_ratio=0.40,
        center_right_ratio=0.60,
        visible_steering_pid_enable=False,
    )
    near_controller = FollowSafetyController(near_cfg)
    near_person = PersonTarget((480, 100, 600, 440), track_id=41, confidence=0.9, area=40800)

    def near_frame(distance_m):
        state = DistanceState(
            source="vision_depth",
            raw_distance_m=distance_m,
            filtered_distance_m=distance_m,
            used_distance_m=distance_m,
            source_detail="depth_matched",
            sample_age_sec=0.01,
        )
        return SensorFrame(
            width=640,
            height=480,
            persons=[near_person],
            distance_m=distance_m,
            distance_state=state,
        )

    near_controller.decide(1, near_frame(2.0))
    near_rotation = near_controller.decide(2, near_frame(1.79))
    if (
        len(near_rotation.actions) != 1
        or near_rotation.actions[0].kind not in ("rotate_left", "rotate_right")
        or near_rotation.actions[0].kind == "backward"
    ):
        raise AssertionError(f"1.79m must use in-place rotation only: {near_rotation}")
    missing_depth_rotation = near_controller.decide(
        3,
        SensorFrame(width=640, height=480, persons=[near_person], distance_m=None),
    )
    if (
        not missing_depth_rotation.actions
        or missing_depth_rotation.actions[0].kind not in ("rotate_left", "rotate_right")
        or missing_depth_rotation.actions[0].kind == "backward"
    ):
        raise AssertionError(
            f"missing near-range Depth must retain in-place rotation: {missing_depth_rotation}"
        )

    # A visual near-guard with no first range sample must enter the same
    # in-place mode immediately, rather than starting reverse and switching
    # modes when Depth arrives on the next frame.
    visual_latch_controller = FollowSafetyController(
        FollowPolicyConfig(
            reverse_enable=True,
            near_distance_rotate_only_enable=True,
            near_distance_rotate_only_distance_m=1.80,
            initial_target_confirm_frames=1,
            visible_steering_pid_enable=False,
            reverse_visual_guard_area_ratio=0.45,
            reverse_visual_guard_height_ratio=0.92,
            reverse_confirm_frames=1,
        )
    )
    visual_latch_person = PersonTarget(
        (300, 0, 640, 479), track_id=55, confidence=0.9, area=340 * 479
    )
    visual_latch_controller.decide(
        0,
        SensorFrame(width=640, height=480, persons=[visual_latch_person], distance_m=None),
    )
    visual_latch = visual_latch_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[visual_latch_person], distance_m=None),
    )
    if (
        visual_latch.reason != "near_distance_rotation_only"
        or not visual_latch.actions
        or visual_latch.actions[0].kind == "backward"
        or not visual_latch_controller._near_distance_rotation_only_active
    ):
        raise AssertionError(
            "visual near guard must latch the in-place mode before Depth recovers: "
            f"{visual_latch}"
        )
    resumed_follow = near_controller.decide(4, near_frame(1.81))
    if (
        not resumed_follow.actions
        or resumed_follow.actions[0].kind == "backward"
        or resumed_follow.actions[0].kind not in ("steer_right", "forward")
    ):
        raise AssertionError(f"1.81m must resume forward follow: {resumed_follow}")

    # Rotation-only testing must not pass through close-range reverse or
    # parked-distance steering limits. It uses the full camera/encoder yaw PID
    # and briefly preserves that yaw across at most two dropped visual frames.
    rotation_only_cfg = FollowPolicyConfig(
        lost_confirm_frames=5,
        steer_min_hold_sec=0.25,
        steer_lost_hold_frames=4,
        steer_lost_hold_max_sec=1.0,
        target_distance_m=1.5,
        distance_parking_enable=True,
        reverse_enable=True,
        reverse_start_distance_m=1.5,
        reverse_immediate_distance_m=1.35,
        initial_target_confirm_frames=1,
        center_left_ratio=0.40,
        center_right_ratio=0.60,
        visible_steering_pid_enable=True,
        visible_steering_pid_camera_hfov_deg=60.0,
        visible_steering_pid_deadband_deg=6.0,
        visible_steering_pid_max_correction_rpm=40.0,
        visible_steering_pid_dynamic_small_error_deg=6.0,
        visible_steering_pid_dynamic_large_error_deg=20.0,
        visible_steering_pid_dynamic_small_max_correction_rpm=20.0,
        visible_steering_pid_fallback_max_correction_rpm=40.0,
        visible_steering_pid_lost_hold_max_correction_rpm=20,
        visible_steering_pid_startup_kick_error_deg=6.0,
        visible_steering_pid_startup_kick_rpm=30.0,
        visible_steering_pid_startup_kick_max_sec=0.60,
        visible_steering_pid_startup_kick_release_yaw_rate_dps=4.0,
    )
    rotation_only_controller = FollowSafetyController(rotation_only_cfg)
    rotation_target = PersonTarget(
        (480, 100, 600, 440),
        track_id=31,
        confidence=0.9,
        area=40800,
    )
    zero_yaw_feedback = SteeringFeedback(
        timestamp=time.monotonic(),
        yaw_rate_right_dps=0.0,
        trustworthy=True,
    )
    rotation_only_visible = rotation_only_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[rotation_target],
            distance_m=1.0,
            steering_feedback=zero_yaw_feedback,
        ),
        rotation_only=True,
    )
    if (
        len(rotation_only_visible.actions) != 1
        or rotation_only_visible.actions[0].kind != "steer_right"
        or rotation_only_visible.actions[0].steer_correction_rpm != 30
    ):
        raise AssertionError(
            "rotation-only mode must bypass close-range reverse and emit the "
            f"30rpm startup yaw command: {rotation_only_visible}"
        )
    rotation_only_controller.set_last_dispatched("steer_right")
    for frame_index in (2, 3):
        dropout = rotation_only_controller.decide(
            frame_index,
            SensorFrame(width=640, height=480, persons=[], distance_m=1.0),
            rotation_only=True,
        )
        if (
            len(dropout.actions) != 1
            or dropout.actions[0].kind != "rotate_right"
            or dropout.explicit_stop_requested
        ):
            raise AssertionError(
                f"rotation-only frame {frame_index} must retain bounded yaw: {dropout}"
            )
        rotation_only_controller.set_last_dispatched("steer_right")
    third_dropout = rotation_only_controller.decide(
        4,
        SensorFrame(width=640, height=480, persons=[], distance_m=1.0),
        rotation_only=True,
    )
    if (
        not third_dropout.actions
        or third_dropout.actions[0].kind != "rotate_right"
        or third_dropout.explicit_stop_requested
    ):
        raise AssertionError(
            "rotation-only yaw hold must enter same-direction search without a "
            f"redundant stop after confirmation: {third_dropout}"
        )

    # An oversized but geometrically continuous target remains visible while
    # its center is unsafe for lateral steering. Fresh close Depth may command
    # straight reverse; a full-frame target with missing Depth uses the visual
    # near guard to reverse instead of waiting until recognition is lost.
    unsteerable_cfg = FollowPolicyConfig(
        target_distance_m=1.5,
        brake_distance_m=0.5,
        distance_parking_enable=False,
        reverse_enable=True,
        reverse_start_distance_m=1.5,
        reverse_immediate_distance_m=1.35,
        initial_target_confirm_frames=1,
        center_left_ratio=0.40,
        center_right_ratio=0.60,
        visible_steering_pid_enable=True,
    )
    oversized_person = PersonTarget(
        (0, 0, 640, 470),
        track_id=9,
        confidence=0.9,
        area=640 * 470,
    )
    unsteerable_reverse_controller = FollowSafetyController(unsteerable_cfg)
    unsteerable_reverse_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[oversized_person], distance_m=2.0),
    )
    close_depth = DistanceState(
        source="vision_depth",
        raw_distance_m=1.20,
        filtered_distance_m=1.20,
        used_distance_m=1.20,
        source_detail="depth_matched",
        sample_age_sec=0.01,
    )
    unsteerable_reverse = unsteerable_reverse_controller.decide(
        2,
        SensorFrame(
            width=640,
            height=480,
            persons=[oversized_person],
            distance_m=1.20,
            distance_state=close_depth,
        ),
        target_steerable=False,
    )
    if (
        len(unsteerable_reverse.actions) != 1
        or unsteerable_reverse.actions[0].kind != "backward"
        or unsteerable_reverse.reason != "target_approaching_reverse"
    ):
        raise AssertionError(
            f"fresh close Depth must produce only straight reverse: {unsteerable_reverse}"
        )

    oversized_right_person = PersonTarget(
        (190, 0, 640, 479),
        track_id=9,
        confidence=0.9,
        area=450 * 479,
    )
    limited_reverse = unsteerable_reverse_controller.decide(
        3,
        SensorFrame(
            width=640,
            height=480,
            persons=[oversized_right_person],
            distance_m=1.20,
            distance_state=close_depth,
        ),
        target_steerable=False,
        target_steering_limit_rpm=4.0,
        record_target_motion=False,
    )
    if (
        len(limited_reverse.actions) != 1
        or limited_reverse.actions[0].kind != "backward"
        or not 1 <= abs(limited_reverse.actions[0].steer_correction_rpm) <= 4
    ):
        raise AssertionError(
            "continuous oversized target must retain capped reverse yaw without "
            f"entering full steering: {limited_reverse}"
        )

    unsteerable_hold_controller = FollowSafetyController(unsteerable_cfg)
    unsteerable_hold_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[oversized_person], distance_m=2.0),
    )
    unsteerable_hold_controller.search_state = "searching"
    unsteerable_hold_controller.search_direction = "left"
    unsteerable_hold_controller._lost_started_at = time.monotonic() - 4.9
    unsteerable_hold_controller._search_rotation_started_at = time.monotonic() - 4.9
    unsteerable_hold = unsteerable_hold_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[oversized_person], distance_m=None),
        target_steerable=False,
    )
    if (
        len(unsteerable_hold.actions) != 1
        or unsteerable_hold.actions[0].kind != "backward"
        or unsteerable_hold.explicit_stop_requested
        or unsteerable_hold.reason != "visual_near_guard_reverse"
        or unsteerable_hold_controller.search_state != "none"
        or unsteerable_hold_controller._lost_started_at is not None
    ):
        raise AssertionError(
            f"visible full-frame target must pause search and reverse: {unsteerable_hold}"
        )

    curve_cfg = FollowPolicyConfig(
        target_distance_m=1.5,
        brake_distance_m=0.5,
        min_forward_percent=40,
        max_forward_percent=100,
        forward_min_rpm=20,
        forward_max_rpm=50,
        forward_curve_max_distance_m=5.0,
        forward_curve_exponent=0.75,
        forward_start_distance_m=1.51,
        forward_stop_distance_m=1.50,
        distance_pid_enable=False,
        initial_target_confirm_frames=1,
    )
    curve_controller = FollowSafetyController(curve_cfg)
    at_target = curve_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[person], distance_m=1.5),
    )
    if not at_target.explicit_stop_requested or at_target.actions or at_target.reason != "target_distance_reached":
        raise AssertionError(f"1.50m centered target must stop: {at_target}")

    # IR-only parking keeps a wide Schmitt band around the 1.50m target. The
    # hold state is a zero-speed motor target, not an explicit safety STOP.
    hysteresis_controller = FollowSafetyController(
        FollowPolicyConfig(
            target_distance_m=1.50,
            brake_distance_m=0.50,
            distance_parking_enable=False,
            reverse_enable=True,
            reverse_start_distance_m=1.30,
            reverse_immediate_distance_m=1.30,
            reverse_stop_distance_m=1.45,
            reverse_confirm_frames=1,
            forward_start_distance_m=1.80,
            forward_stop_distance_m=1.65,
            distance_pid_enable=True,
            distance_pid_deadband_m=0.15,
            initial_target_confirm_frames=1,
        )
    )

    def depth_hysteresis_frame(distance_m: float) -> SensorFrame:
        return SensorFrame(
            width=640,
            height=480,
            persons=[person],
            distance_m=distance_m,
            distance_state=DistanceState(
                source="vision_depth",
                raw_distance_m=distance_m,
                filtered_distance_m=distance_m,
                used_distance_m=distance_m,
                source_detail="depth_matched",
                sample_age_sec=0.01,
            ),
        )

    def assert_zero_hold(decision, label: str) -> None:
        if (
            decision.explicit_stop_requested
            or len(decision.actions) != 1
            or decision.actions[0].kind != "forward"
            or decision.actions[0].speed_percent != 0
        ):
            raise AssertionError(f"{label} must replace stale motion with zero DRIVE: {decision}")

    h_inactive = hysteresis_controller.decide(1, depth_hysteresis_frame(1.79))
    assert_zero_hold(h_inactive, "1.79m before forward start")
    h_started = hysteresis_controller.decide(2, depth_hysteresis_frame(1.81))
    if not h_started.actions or h_started.actions[0].kind != "forward" or h_started.actions[0].speed_percent <= 0:
        raise AssertionError(f"1.81m must start forward motion: {h_started}")
    h_continue = hysteresis_controller.decide(3, depth_hysteresis_frame(1.70))
    if not h_continue.actions or h_continue.actions[0].kind != "forward" or h_continue.actions[0].speed_percent <= 0:
        raise AssertionError(f"active forward motion must continue at 1.70m: {h_continue}")
    h_stopped = hysteresis_controller.decide(4, depth_hysteresis_frame(1.64))
    assert_zero_hold(h_stopped, "1.64m forward stop")
    h_wait_restart = hysteresis_controller.decide(5, depth_hysteresis_frame(1.70))
    assert_zero_hold(h_wait_restart, "1.70m while forward latch is inactive")
    h_restarted = hysteresis_controller.decide(6, depth_hysteresis_frame(1.81))
    if not h_restarted.actions or h_restarted.actions[0].kind != "forward" or h_restarted.actions[0].speed_percent <= 0:
        raise AssertionError(f"1.81m must restart forward motion: {h_restarted}")
    h_reverse_started = hysteresis_controller.decide(7, depth_hysteresis_frame(1.29))
    if not h_reverse_started.actions or h_reverse_started.actions[0].kind != "backward":
        raise AssertionError(f"1.29m must start immediate reverse: {h_reverse_started}")
    h_reverse_continue = hysteresis_controller.decide(8, depth_hysteresis_frame(1.44))
    if not h_reverse_continue.actions or h_reverse_continue.actions[0].kind != "backward":
        raise AssertionError(f"active reverse state must continue through 1.44m: {h_reverse_continue}")
    h_reverse_stopped = hysteresis_controller.decide(9, depth_hysteresis_frame(1.46))
    assert_zero_hold(h_reverse_stopped, "1.46m reverse stop")

    off_center_person = PersonTarget((470, 120, 590, 430), track_id=1, confidence=0.9, area=37200)
    off_center_at_target = curve_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[off_center_person], distance_m=1.5),
    )
    if (
        off_center_at_target.explicit_stop_requested
        or not off_center_at_target.actions
        or off_center_at_target.actions[0].kind != "rotate_right"
        or off_center_at_target.actions[0].speed_percent != 0
        or off_center_at_target.reason != "person_parked_recenter_right"
    ):
        raise AssertionError(
            f"1.50m off-center target must rotate in place without forward motion: {off_center_at_target}"
        )

    parked_pid_controller = FollowSafetyController(
        FollowPolicyConfig(
            target_distance_m=1.20,
            brake_distance_m=0.80,
            initial_target_confirm_frames=1,
            visible_steering_pid_enable=True,
            center_deadzone_ratio=0.05,
            parked_recenter_min_rpm=2,
            parked_recenter_max_rpm=5,
        )
    )
    parked_pid = parked_pid_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[off_center_person],
            distance_m=1.20,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=0.0,
                trustworthy=True,
            ),
        ),
    )
    parked_result = parked_pid_controller.last_steering_pid_result
    if (
        not parked_pid.actions
        or parked_pid.actions[0].kind != "rotate_right"
        or parked_pid.reason != "person_parked_recenter_right"
        or parked_result is None
        or not parked_result.feedback_used
        or not 1 <= abs(int(parked_result.correction_rpm)) <= 5
    ):
        raise AssertionError(f"parked recenter must use capped camera/encoder PID: {parked_pid}, {parked_result}")

    near_rotation_controller = FollowSafetyController(
        FollowPolicyConfig(
            target_distance_m=1.50,
            brake_distance_m=0.80,
            near_distance_rotate_only_enable=True,
            near_distance_rotate_only_distance_m=1.80,
            initial_target_confirm_frames=1,
            visible_steering_pid_enable=True,
            center_deadzone_ratio=0.10,
            parked_recenter_min_rpm=2,
            parked_recenter_max_rpm=10,
            near_distance_rotation_only_max_rpm=8,
        )
    )
    near_rotation_frame = SensorFrame(
        width=640,
        height=480,
        persons=[off_center_person],
        distance_m=1.75,
        steering_feedback=SteeringFeedback(
            timestamp=time.monotonic(),
            yaw_rate_right_dps=0.0,
            trustworthy=True,
        ),
    )
    near_rotation_controller.decide(1, near_rotation_frame)
    near_rotation_decision = near_rotation_controller.decide(2, near_rotation_frame)
    near_rotation_result = near_rotation_controller.last_steering_pid_result
    if (
        near_rotation_decision.reason != "near_distance_rotation_only"
        or not near_rotation_decision.actions
        or near_rotation_decision.actions[0].kind != "rotate_right"
        or near_rotation_result is None
        or not near_rotation_result.feedback_used
        or not 2 <= abs(int(near_rotation_result.correction_rpm)) <= 8
        or float(near_rotation_result.correction_limit_rpm) > 8.01
    ):
        raise AssertionError(
            "near-distance in-place rotation must preserve camera/encoder PID output: "
            f"{near_rotation_decision}, {near_rotation_result}"
        )

    # Zero from the camera/encoder loop means the current chassis yaw is
    # already sufficient. It must remain a stop request instead of being
    # replaced by the legacy minimum-RPM edge fallback.
    parked_update = near_rotation_controller._parked_recenter_pid.update
    near_rotation_controller._parked_recenter_pid.update = lambda *args, **kwargs: replace(
        near_rotation_result,
        correction_rpm=0,
    )
    try:
        near_rotation_zero = near_rotation_controller.decide(3, near_rotation_frame)
    finally:
        near_rotation_controller._parked_recenter_pid.update = parked_update
    if (
        near_rotation_zero.reason != "near_distance_rotation_only"
        or not near_rotation_zero.actions
        or near_rotation_zero.actions[0].kind != "stop"
        or not near_rotation_zero.soft_stop_requested
        or near_rotation_zero.actions[0].brake_hold
    ):
        raise AssertionError(
            "an intentional zero-RPM parked PID result must publish a soft zero-yaw hold: "
            f"{near_rotation_zero}"
        )

    # Even inside a wide coarse center zone, a target moving from x=0.55 to
    # x=0.58 is leaving the aim line and must start a bounded right correction.
    projected_recenter_controller = FollowSafetyController(
        FollowPolicyConfig(
            target_distance_m=1.50,
            brake_distance_m=0.80,
            near_distance_rotate_only_enable=True,
            near_distance_rotate_only_distance_m=1.80,
            initial_target_confirm_frames=1,
            visible_steering_pid_enable=True,
            visible_steering_pid_camera_hfov_deg=60.0,
            visible_steering_pid_target_rate_feedforward_gain=0.65,
            visible_steering_pid_target_rate_feedforward_max_dps=24.0,
            visible_motion_history_frames=4,
            visible_motion_lookback_sec=0.10,
            visible_motion_rate_filter_alpha=0.55,
            visible_motion_projection_gain=0.75,
            center_deadzone_ratio=0.10,
            parked_recenter_min_rpm=4,
            parked_recenter_max_rpm=16,
        )
    )
    projected_recenter_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[PersonTarget((312, 120, 392, 430), track_id=19, confidence=0.9, area=24800)],
            distance_m=1.75,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=0.0,
                trustworthy=True,
            ),
        ),
    )
    # decide() uses monotonic time. Make the synthetic first detection exactly
    # 100ms old so this test exercises velocity feedforward instead of relying
    # on how quickly the test process reaches the second call.
    first_motion_sample = projected_recenter_controller._visible_motion_samples[-1]
    projected_recenter_controller._visible_motion_samples[-1] = (
        first_motion_sample[0],
        first_motion_sample[1] - 0.10,
        first_motion_sample[2],
    )
    projected_recenter = projected_recenter_controller.decide(
        2,
        SensorFrame(
            width=640,
            height=480,
            persons=[PersonTarget((331, 120, 411, 430), track_id=19, confidence=0.9, area=24800)],
            distance_m=1.75,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=0.0,
                trustworthy=True,
            ),
        ),
    )
    if (
        projected_recenter.reason != "near_distance_rotation_only"
        or not projected_recenter.actions
        or projected_recenter.actions[0].kind != "rotate_right"
        or projected_recenter_controller.last_steering_pid_result is None
        or projected_recenter_controller.last_steering_pid_result.correction_rpm <= 0
    ):
        raise AssertionError(
            "near-distance PID must follow outward motion before it leaves the coarse center zone: "
            f"{projected_recenter}"
        )

    # Once the bbox has crossed the aim line and continues moving outward, the
    # controller must start the 1 RPM correction immediately even though it is
    # still inside the coarse 0.40-0.60 center band. This reproduces the frame
    # 113 delay where x=0.498 and image rate was already -6.96 dps.
    early_motion_controller = FollowSafetyController(
        FollowPolicyConfig(
            visible_steering_pid_enable=True,
            visible_steering_pid_camera_hfov_deg=60.0,
            visible_steering_pid_camera_latency_sec=0.0,
            visible_steering_pid_deadband_deg=3.0,
            visible_steering_pid_target_rate_feedforward_gain=0.30,
            visible_steering_pid_target_rate_feedforward_max_dps=8.0,
            center_left_ratio=0.40,
            center_right_ratio=0.60,
            center_deadzone_ratio=0.10,
            parked_recenter_min_rpm=1,
            parked_recenter_max_rpm=5,
        )
    )
    early_left_target = PersonTarget(
        (278.72, 80.0, 358.72, 430.0),
        track_id=20,
        confidence=0.9,
        area=28000,
    )
    early_left = early_motion_controller._pid_action_for_parked_target(
        early_left_target,
        SensorFrame(
            width=640,
            height=480,
            persons=[early_left_target],
            distance_m=1.50,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=0.0,
                trustworthy=True,
            ),
        ),
        time.monotonic(),
        "center",
        target_image_rate_dps=-6.96,
    )
    if (
        early_left is None
        or early_left.kind != "rotate_left"
        or early_motion_controller.last_steering_pid_result is None
        or early_motion_controller.last_steering_pid_result.correction_rpm != -1
    ):
        raise AssertionError(
            "outward bbox velocity must start a 1 RPM turn before leaving the coarse center band: "
            f"{early_left}, {early_motion_controller.last_steering_pid_result}"
        )

    timestamped_motion_controller = FollowSafetyController(
        FollowPolicyConfig(
            visible_motion_history_frames=4,
            visible_motion_lookback_sec=0.10,
            visible_motion_rate_filter_alpha=1.0,
            visible_motion_min_ratio=0.0,
            visible_steering_pid_camera_hfov_deg=60.0,
            visible_steering_pid_target_rate_feedforward_max_dps=60.0,
        )
    )
    motion_start = PersonTarget(
        (280, 100, 360, 430), track_id=31, confidence=0.9, area=26400
    )
    motion_end = PersonTarget(
        (312, 100, 392, 430), track_id=31, confidence=0.9, area=26400
    )
    timestamped_motion_controller._record_visible_motion(1, motion_start, 640, 10.0)
    _dx_fast, _projected_fast, fast_rate_dps, fast_dt = (
        timestamped_motion_controller._record_visible_motion(2, motion_end, 640, 10.1)
    )
    timestamped_motion_controller._reset_visible_motion()
    timestamped_motion_controller._record_visible_motion(1, motion_start, 640, 20.0)
    _dx_slow, _projected_slow, slow_rate_dps, slow_dt = (
        timestamped_motion_controller._record_visible_motion(2, motion_end, 640, 20.3)
    )
    if (
        abs(fast_dt - 0.10) > 0.001
        or abs(slow_dt - 0.30) > 0.001
        or not fast_rate_dps > slow_rate_dps * 2.9
    ):
        raise AssertionError(
            "equal image displacement must produce a rate based on real elapsed time: "
            f"fast={fast_rate_dps:.2f}dps/{fast_dt:.3f}s "
            f"slow={slow_rate_dps:.2f}dps/{slow_dt:.3f}s"
        )

    runtime_rate_controller = FollowSafetyController(
        FollowPolicyConfig(
            visible_motion_history_frames=4,
            visible_motion_lookback_sec=0.10,
            visible_motion_rate_filter_alpha=1.0,
            visible_motion_min_ratio=0.0,
            visible_steering_pid_camera_hfov_deg=60.0,
            visible_steering_pid_max_yaw_rate_dps=45.0,
            visible_steering_pid_target_rate_feedforward_max_dps=12.0,
        )
    )
    rate_start = PersonTarget(
        (280, 100, 360, 430), track_id=32, confidence=0.9, area=26400
    )
    rate_end = PersonTarget(
        (328, 100, 408, 430), track_id=32, confidence=0.9, area=26400
    )
    runtime_rate_controller._record_visible_motion(1, rate_start, 640, 30.0)
    _dx, _projected, runtime_rate_dps, _dt = (
        runtime_rate_controller._record_visible_motion(2, rate_end, 640, 30.1)
    )
    if not 44.9 <= runtime_rate_dps <= 45.1:
        raise AssertionError(
            "image-rate measurement must retain chassis-scale motion instead of "
            f"using the 12dps feedforward output cap: {runtime_rate_dps:.2f}dps"
        )

    direction_guard_controller = FollowSafetyController(
        FollowPolicyConfig(
            visible_steering_pid_enable=True,
            visible_steering_pid_camera_hfov_deg=60.0,
            visible_steering_pid_camera_latency_sec=0.08,
            visible_steering_pid_deadband_deg=3.0,
            visible_steering_pid_outer_kp_per_sec=2.40,
            visible_steering_pid_outer_kd_sec=0.10,
            visible_steering_pid_rate_kp_rpm_per_dps=0.18,
            visible_steering_pid_same_direction_overspeed_threshold_dps=10.0,
            visible_steering_pid_same_direction_overspeed_brake_gain_rpm_per_dps=0.20,
            visible_motion_min_ratio=0.015,
            visible_motion_projection_gain=0.75,
            center_left_ratio=0.40,
            center_right_ratio=0.60,
            parked_recenter_min_rpm=4,
            parked_recenter_max_rpm=16,
        )
    )
    right_side_target = PersonTarget(
        (320.0, 0.0, 463.4, 479.0),
        track_id=23,
        confidence=0.9,
        area=68689,
    )
    right_side_frame = SensorFrame(
        width=640,
        height=480,
        persons=[right_side_target],
        distance_m=1.21,
        steering_feedback=SteeringFeedback(
            timestamp=time.monotonic(),
            yaw_rate_right_dps=35.8,
            trustworthy=True,
        ),
    )
    guarded_brake = direction_guard_controller._pid_action_for_parked_target(
        right_side_target,
        right_side_frame,
        time.monotonic(),
        "right",
        motion_dx_ratio=0.0,
        max_correction_rpm=16.0,
    )
    guarded_result = direction_guard_controller.last_steering_pid_result
    if (
        guarded_brake is not None
        or guarded_result is None
        or guarded_result.correction_rpm != 0
        or not guarded_result.same_direction_overspeed_braking
    ):
        raise AssertionError(
            "a target still outside the right center boundary must coast during "
            "same-direction encoder overspeed: "
            f"{guarded_brake}, {guarded_result}"
        )

    direction_guard_controller._visual_steering_pid.reset()
    guarded_driving_brake = direction_guard_controller._pid_action_for_visible_target(
        right_side_target,
        right_side_frame,
        time.monotonic(),
        motion_dx_ratio=0.0,
        max_correction_rpm=16.0,
    )
    if (
        guarded_driving_brake is not None
        or direction_guard_controller.last_steering_pid_result is None
        or direction_guard_controller.last_steering_pid_result.correction_rpm != 0
    ):
        raise AssertionError(
            "the moving follow PID must coast instead of braking left while the target "
            f"is still right: {guarded_driving_brake}"
        )

    left_side_target = PersonTarget(
        (176.6, 0.0, 320.0, 479.0),
        track_id=23,
        confidence=0.9,
        area=68689,
    )
    direction_guard_controller._parked_recenter_pid.reset()
    guarded_left_brake = direction_guard_controller._pid_action_for_parked_target(
        left_side_target,
        SensorFrame(
            width=640,
            height=480,
            persons=[left_side_target],
            distance_m=1.21,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=-35.8,
                trustworthy=True,
            ),
        ),
        time.monotonic(),
        "left",
        motion_dx_ratio=0.0,
        max_correction_rpm=16.0,
    )
    guarded_left_result = direction_guard_controller.last_steering_pid_result
    if (
        guarded_left_brake is not None
        or guarded_left_result is None
        or guarded_left_result.correction_rpm != 0
        or not guarded_left_result.same_direction_overspeed_braking
    ):
        raise AssertionError(
            "the direction guard must symmetrically coast during overspeed "
            "while a target is still on the left: "
            f"{guarded_left_brake}"
        )

    if (
        direction_guard_controller._pid_direction_guard_reason(0.58, 0.0, -2)
        != "target_still_outside_center"
        or direction_guard_controller._pid_direction_guard_reason(0.58, -0.03, -2)
        != "target_still_outside_center"
    ):
        raise AssertionError(
            "fresh target position must remain the final direction authority "
            "regardless of delayed image-motion history"
        )

    direction_guard_controller._parked_recenter_pid.reset()
    small_right_target = PersonTarget(
        (331.2, 80.0, 411.2, 430.0),
        track_id=23,
        confidence=0.9,
        area=28000,
    )
    tracking_floor = direction_guard_controller._pid_action_for_parked_target(
        small_right_target,
        SensorFrame(width=640, height=480, persons=[small_right_target], distance_m=1.21),
        time.monotonic(),
        "right",
        motion_dx_ratio=0.0,
        max_correction_rpm=16.0,
    )
    tracking_floor_result = direction_guard_controller.last_steering_pid_result
    if (
        tracking_floor is None
        or tracking_floor.kind != "rotate_right"
        or tracking_floor_result is None
        or tracking_floor_result.correction_rpm != 1
    ):
        raise AssertionError(
            "ordinary tracking must retain the PID's 1 RPM output without a launch floor: "
            f"{tracking_floor}, {tracking_floor_result}"
        )

    fast_loss_controller = FollowSafetyController(
        FollowPolicyConfig(
            initial_target_confirm_frames=1,
            lost_confirm_sec=0.0,
            lost_confirm_frames=3,
            release_target_on_lost=False,
        )
    )
    fast_loss_target = PersonTarget(
        (260, 100, 380, 430),
        track_id=29,
        confidence=0.9,
        area=39600,
    )
    fast_loss_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[fast_loss_target], distance_m=2.0),
    )
    fast_loss_first_gap = fast_loss_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[], distance_m=2.0),
    )
    fast_loss_second_gap = fast_loss_controller.decide(
        3,
        SensorFrame(width=640, height=480, persons=[], distance_m=2.0),
    )
    fast_loss_confirmed = fast_loss_controller.decide(
        4,
        SensorFrame(width=640, height=480, persons=[], distance_m=2.0),
    )
    if (
        not fast_loss_first_gap.reason.startswith("lost_wait_yaw_")
        or not fast_loss_second_gap.reason.startswith("lost_wait_yaw_")
        or fast_loss_first_gap.shutdown_requested
        or not fast_loss_confirmed.actions
        or fast_loss_confirmed.actions[0].kind not in ("rotate_left", "rotate_right")
        or fast_loss_confirmed.explicit_stop_requested
    ):
        raise AssertionError(
            "three-frame loss confirmation must tolerate two misses and enter search on the third: "
            f"first={fast_loss_first_gap}, second={fast_loss_second_gap}, "
            f"confirmed={fast_loss_confirmed}"
        )

    # A mapped close-up edge crop remains excluded from ReID/Depth and
    # longitudinal motion, but its bounded center may drive yaw-only tracking.
    close_cropped_controller = FollowSafetyController(
        FollowPolicyConfig(
            target_distance_m=1.50,
            brake_distance_m=0.80,
            near_distance_rotate_only_enable=True,
            near_distance_rotate_only_distance_m=1.80,
            initial_target_confirm_frames=1,
            visible_steering_pid_enable=True,
            center_deadzone_ratio=0.10,
            parked_recenter_min_rpm=2,
            parked_recenter_max_rpm=10,
        )
    )
    close_cropped_controller.active_target_id = 41
    close_cropped_controller._has_seen_person = True
    close_cropped_person = PersonTarget(
        (0, 0, 402, 479),
        track_id=41,
        confidence=0.93,
        area=192558,
    )
    close_cropped = close_cropped_controller.decide(
        10,
        SensorFrame(
            width=640,
            height=480,
            persons=[close_cropped_person],
            distance_m=0.76,
            distance_state=DistanceState(
                source="vision_depth",
                used_distance_m=0.76,
                source_detail="depth_multiregion_reused_hold",
            ),
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=0.0,
                trustworthy=True,
            ),
        ),
        target_steerable=False,
        target_steering_limit_rpm=8.5,
        record_target_motion=False,
        low_quality_visible=True,
    )
    if (
        close_cropped.reason != "target_visible_low_quality_yaw"
        or close_cropped.explicit_stop_requested
        or not close_cropped.actions
        or close_cropped.actions[0].kind != "rotate_left"
        or close_cropped_controller.last_steering_pid_result is None
        or abs(close_cropped_controller.last_steering_pid_result.correction_rpm) > 8.5
    ):
        raise AssertionError(
            "mapped close crop must keep bounded yaw without longitudinal control: "
            f"{close_cropped}, {close_cropped_controller.last_steering_pid_result}"
        )

    # A one-frame detector dropout while a near target is being recentered
    # must keep the last yaw direction during lost confirmation, rather than
    # issuing an immediate STOP and entering search.
    near_rotation_controller.set_last_dispatched("rotate_right")
    near_lost_hold = near_rotation_controller.decide(
        4,
        SensorFrame(
            width=640,
            height=480,
            persons=[],
            distance_m=None,
            distance_state=DistanceState(
                source="vision_depth",
                used_distance_m=1.75,
                source_detail="depth_expired",
            ),
        ),
    )
    if (
        not near_lost_hold.actions
        or near_lost_hold.actions[0].kind != "rotate_right"
        or near_lost_hold.explicit_stop_requested
        or near_lost_hold.reason != "lost_wait_yaw_right"
    ):
        raise AssertionError(
            "near-distance detector dropout must hold the previous yaw command: "
            f"{near_lost_hold}"
        )

    exit_on_loss_controller = FollowSafetyController(
        FollowPolicyConfig(
            initial_target_confirm_frames=1,
            lost_confirm_frames=2,
            exit_on_target_loss=True,
            search_before_first_seen=False,
        )
    )
    exit_target = PersonTarget((280, 100, 400, 430), track_id=51, confidence=0.9, area=39600)
    exit_on_loss_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[exit_target], distance_m=2.0),
    )
    exit_on_loss_controller.set_last_dispatched("forward")
    exit_wait = exit_on_loss_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[], distance_m=None),
    )
    if (
        exit_wait.shutdown_requested
        or exit_wait.reason != "lost_wait_yaw_right"
        or not exit_wait.actions
        or exit_wait.actions[0].kind != "rotate_right"
        or exit_wait.actions[0].speed_percent != 0
    ):
        raise AssertionError(f"loss confirmation must precede immediate exit: {exit_wait}")
    exit_decision = exit_on_loss_controller.decide(
        3,
        SensorFrame(width=640, height=480, persons=[], distance_m=None),
    )
    if (
        not exit_decision.shutdown_requested
        or not exit_decision.explicit_stop_requested
        or not exit_decision.clear_action_queue
        or exit_decision.reason != "target_lost_exit"
        or exit_on_loss_controller.search_state != "timed_out"
    ):
        raise AssertionError(f"confirmed target loss must exit without search rotation: {exit_decision}")

    # The independent Depth30 loop must not recalculate the parked yaw PID.
    # It only supervises longitudinal speed and should preserve the latest
    # visual rotation command while the target is inside the near-distance band.
    depth_near_hold = near_rotation_controller.decide(
        3,
        near_rotation_frame,
        longitudinal_only=True,
    )
    if depth_near_hold.actions or depth_near_hold.reason != "longitudinal_near_rotation_hold":
        raise AssertionError(
            "Depth30 near-distance update must preserve the visual yaw command: "
            f"{depth_near_hold}"
        )

    near_center_person = PersonTarget((325, 120, 405, 430), track_id=1, confidence=0.9, area=24800)
    near_center_controller = FollowSafetyController(
        FollowPolicyConfig(
            target_distance_m=1.20,
            brake_distance_m=0.80,
            initial_target_confirm_frames=1,
            visible_steering_pid_enable=True,
            center_deadzone_ratio=0.05,
            parked_recenter_min_rpm=2,
            parked_recenter_max_rpm=5,
        )
    )
    near_center_decision = near_center_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[near_center_person],
            distance_m=1.20,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=0.0,
                trustworthy=True,
            ),
        ),
    )
    near_center_result = near_center_controller.last_steering_pid_result
    if (
        not near_center_decision.actions
        or near_center_result is None
        or abs(int(near_center_result.correction_rpm)) > 3
        or float(near_center_result.correction_limit_rpm) > 3.01
    ):
        raise AssertionError(
            f"parked recenter just outside center must taper to <=3rpm: "
            f"{near_center_decision}, {near_center_result}"
        )
    if (
        parked_pid_controller._parked_recenter_pid.last_result is not parked_result
        or parked_pid_controller._visual_steering_pid.last_result is not None
    ):
        raise AssertionError("parked recenter must not reuse the moving steering PID state")

    parked_missing_distance = parked_pid_controller.decide(
        2,
        SensorFrame(
            width=640,
            height=480,
            persons=[off_center_person],
            distance_m=None,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=0.0,
                trustworthy=True,
            ),
        ),
    )
    if (
        parked_missing_distance.explicit_stop_requested
        or not parked_missing_distance.actions
        or parked_missing_distance.actions[0].kind != "rotate_right"
        or not parked_missing_distance.reason.startswith("person_parked_recenter_")
    ):
        raise AssertionError(
            "mmwave dropout must not interrupt camera recenter while forward remains latched: "
            f"{parked_missing_distance}"
        )

    # 速度曲线测试使用新控制器，不能继承上面的目标距离停车锁存。
    curve_controller = FollowSafetyController(curve_cfg)
    curve_speeds = []
    for frame_index, distance_m in enumerate((1.51, 2.0, 3.0, 4.0, 5.0, 6.0), start=3):
        curve_decision = curve_controller.decide(
            frame_index,
            SensorFrame(width=640, height=480, persons=[person], distance_m=distance_m),
        )
        if not curve_decision.actions or curve_decision.actions[0].kind != "forward":
            raise AssertionError(f"distance curve should drive at {distance_m}m: {curve_decision}")
        curve_speeds.append(curve_decision.actions[0].speed_percent)
    print("forward_distance_curve:", curve_speeds)
    if curve_speeds != sorted(curve_speeds) or curve_speeds[0] < 40 or curve_speeds[-2] != 100 or curve_speeds[-1] != 100:
        raise AssertionError(f"distance curve must rise from 20rpm to a 50rpm cap: {curve_speeds}")

    off_center_far = curve_controller.decide(
        9,
        SensorFrame(width=640, height=480, persons=[off_center_person], distance_m=4.0),
    )
    if not off_center_far.actions or off_center_far.actions[0].kind != "steer_right":
        raise AssertionError(f"far off-center target should use differential forward steer: {off_center_far}")
    if off_center_far.actions[0].speed_percent != curve_speeds[3]:
        raise AssertionError(
            f"steer must retain the 4m forward curve base: {off_center_far.actions[0].speed_percent} != {curve_speeds[3]}"
        )

    pid_cfg = FollowPolicyConfig(
        visible_steering_pid_enable=True,
        visible_steering_pid_deadband_deg=1.2,
        visible_steering_pid_outer_kp_per_sec=1.85,
        visible_steering_pid_outer_kd_sec=0.08,
        visible_steering_pid_max_yaw_rate_dps=46.0,
        visible_steering_pid_rate_kp_rpm_per_dps=0.16,
        visible_steering_pid_max_correction_rpm=16.0,
        visible_steering_pid_dynamic_small_error_deg=3.5,
        visible_steering_pid_dynamic_large_error_deg=14.0,
        visible_steering_pid_dynamic_small_max_yaw_rate_dps=22.0,
        visible_steering_pid_dynamic_small_max_correction_rpm=6.0,
        visible_steering_pid_dynamic_large_error_base_cap_rpm=28.0,
        initial_target_confirm_frames=1,
        target_distance_m=1.2,
        forward_min_rpm=20,
        forward_max_rpm=50,
    )
    pid_right_controller = FollowSafetyController(pid_cfg)
    pid_right = pid_right_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[off_center_person], distance_m=3.0),
    )
    if (
        not pid_right.actions
        or pid_right.actions[0].kind != "steer_right"
        or not 1 <= pid_right.actions[0].steer_correction_rpm <= 16
    ):
        raise AssertionError(f"right-side target must request bounded PID wheel differential: {pid_right}")
    if (
        pid_right_controller._visual_steering_pid.last_result
        is not pid_right_controller.last_steering_pid_result
        or pid_right_controller._parked_recenter_pid.last_result is not None
    ):
        raise AssertionError("moving steering must not reuse the parked recenter PID state")

    # 5m 且人物明显偏右时，距离曲线原本会给 50 RPM。动态 PID 应把
    # 基础速度压到 28 RPM（56%），同时允许 14-16 RPM 轮差快速追上。
    pid_far_controller = FollowSafetyController(pid_cfg)
    pid_far = pid_far_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[off_center_person],
            distance_m=5.0,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=0.0,
                trustworthy=True,
            ),
        ),
    )
    if (
        not pid_far.actions
        or pid_far.actions[0].kind != "steer_right"
        or pid_far.actions[0].speed_percent != 56
        or not 14 <= pid_far.actions[0].steer_correction_rpm <= 16
    ):
        raise AssertionError(f"large error at 5m must use 28rpm base and 14-16rpm correction: {pid_far}")

    pid_small_person = PersonTarget((312, 120, 392, 430), track_id=1, confidence=0.9, area=24800)
    pid_small_controller = FollowSafetyController(pid_cfg)
    pid_small = pid_small_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[pid_small_person],
            distance_m=5.0,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=0.0,
                trustworthy=True,
            ),
        ),
    )
    if (
        not pid_small.actions
        or pid_small.actions[0].kind != "steer_right"
        or not 94 <= pid_small.actions[0].speed_percent <= 100
        or pid_small.actions[0].steer_correction_rpm > 7
    ):
        raise AssertionError(f"small error must retain most base speed and stay within 7rpm: {pid_small}")

    # 视觉已经回到中心、但编码器仍测到车身向右转时，内环必须反向阻尼，
    # 这样不会等到人物越过画面中心后才大幅反打。
    pid_damping_controller = FollowSafetyController(pid_cfg)
    pid_damping = pid_damping_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[person],
            distance_m=3.0,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=25.0,
                trustworthy=True,
            ),
        ),
    )
    if (
        not pid_damping.actions
        or pid_damping.actions[0].kind != "steer_left"
        or pid_damping.actions[0].steer_correction_rpm <= 0
    ):
        raise AssertionError(f"right yaw at image center must request left damping: {pid_damping}")

    # 距离缺失时摄像头仍主导横向闭环：偏离中心继续低速前进；人物接近
    # 画面边缘时允许最高 15 RPM 轮差追住人物，回到中心则维持 15 RPM 直行。
    pid_missing_controller = FollowSafetyController(pid_cfg)
    pid_missing_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[person], distance_m=3.0),
    )
    pid_missing_side = pid_missing_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[off_center_person], distance_m=None),
    )
    if (
        not pid_missing_side.actions
        or pid_missing_side.actions[0].kind != "steer_right"
        or pid_missing_side.actions[0].speed_percent != 30
        or not 4 <= pid_missing_side.actions[0].steer_correction_rpm <= 15
        or not pid_missing_side.reason.endswith("_distance_missing")
    ):
        raise AssertionError(f"distance-missing target must keep 15rpm visual PID: {pid_missing_side}")

    pid_missing_center_controller = FollowSafetyController(pid_cfg)
    pid_missing_center_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[person], distance_m=3.0),
    )
    pid_missing_center = pid_missing_center_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[person], distance_m=None),
    )
    if (
        not pid_missing_center.actions
        or pid_missing_center.actions[0].kind != "forward"
        or pid_missing_center.actions[0].speed_percent != 30
        or pid_missing_center.reason != "visual_pid_center_camera_distance_missing"
    ):
        raise AssertionError(f"centered target must keep 15rpm through range dropout: {pid_missing_center}")

    # 短时毫米波 hold 已携带上次可信距离，不等同于完全缺距。目标仍被
    # 摄像头锁定时应继续走距离曲线，并保留正常 PID 轮差。
    pid_hold_cfg = FollowPolicyConfig(
        visible_steering_pid_enable=True,
        visible_steering_pid_deadband_deg=1.2,
        visible_steering_pid_max_yaw_rate_dps=34.0,
        visible_steering_pid_max_correction_rpm=12.0,
        initial_target_confirm_frames=1,
        target_distance_m=1.5,
        target_distance_release_m=1.7,
        min_forward_percent=40,
        max_forward_percent=100,
        forward_min_rpm=20,
        forward_max_rpm=50,
        forward_curve_max_distance_m=5.0,
        mmwave_hold_forward_percent=80,
        mmwave_hold_decel_step_percent=4,
    )
    pid_hold_state_3m = DistanceState(
        source="vision_mmwave",
        raw_distance_m=None,
        filtered_distance_m=3.0,
        used_distance_m=3.0,
        source_detail="unmatched_hold",
        sample_count=0,
    )
    pid_hold_center_controller = FollowSafetyController(pid_hold_cfg)
    pid_hold_center = pid_hold_center_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[person],
            distance_m=3.0,
            distance_state=pid_hold_state_3m,
        ),
    )
    if (
        not pid_hold_center.actions
        or pid_hold_center.actions[0].kind != "forward"
        or pid_hold_center.actions[0].speed_percent != 72
        or pid_hold_center.reason != "follow_distance_mmwave_hold"
    ):
        raise AssertionError(f"3m mmwave hold must retain the distance curve: {pid_hold_center}")

    pid_hold_state_5m = DistanceState(
        source="vision_mmwave",
        raw_distance_m=None,
        filtered_distance_m=5.0,
        used_distance_m=5.0,
        source_detail="unmatched_hold",
        sample_count=0,
    )
    pid_hold_side_controller = FollowSafetyController(pid_hold_cfg)
    pid_hold_side = pid_hold_side_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[off_center_person],
            distance_m=5.0,
            distance_state=pid_hold_state_5m,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=0.0,
                trustworthy=True,
            ),
        ),
    )
    if (
        not pid_hold_side.actions
        or pid_hold_side.actions[0].kind != "steer_right"
        or pid_hold_side.actions[0].speed_percent != 56
        or pid_hold_side.actions[0].steer_correction_rpm <= 3
        or not pid_hold_side.reason.endswith("_mmwave_hold")
    ):
        raise AssertionError(f"mmwave hold must retain camera PID authority: {pid_hold_side}")

    release_controller = FollowSafetyController(
        FollowPolicyConfig(
            target_distance_m=1.20,
            target_distance_release_m=1.40,
            target_distance_release_hold_sec=0.50,
            target_distance_release_confirm_frames=3,
            initial_target_confirm_frames=1,
        )
    )
    d_release_stop = release_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[person], distance_m=1.20),
    )
    if not d_release_stop.explicit_stop_requested or d_release_stop.reason != "target_distance_reached":
        raise AssertionError(f"1.20m must latch target-distance stop: {d_release_stop}")
    d_release_low = release_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[person], distance_m=1.39),
    )
    if not d_release_low.explicit_stop_requested or d_release_low.reason != "target_distance_hold":
        raise AssertionError(f"1.39m must remain below the 1.40m release threshold: {d_release_low}")
    d_release_wait_1 = release_controller.decide(
        3,
        SensorFrame(width=640, height=480, persons=[person], distance_m=1.40),
    )
    release_controller._target_release_started_at = time.monotonic() - 0.51
    d_release_wait_2 = release_controller.decide(
        4,
        SensorFrame(width=640, height=480, persons=[person], distance_m=1.40),
    )
    d_release_ready = release_controller.decide(
        5,
        SensorFrame(width=640, height=480, persons=[person], distance_m=1.40),
    )
    if d_release_wait_1.reason != "target_distance_release_wait" or d_release_wait_2.reason != "target_distance_release_wait":
        raise AssertionError(f"release must wait for three confirmed frames: {d_release_wait_1}, {d_release_wait_2}")
    if not d_release_ready.actions or d_release_ready.explicit_stop_requested:
        raise AssertionError(f"stable 1.40m must release target-distance stop: {d_release_ready}")

    reverse_cfg = FollowPolicyConfig(
        target_distance_m=1.50,
        target_distance_release_m=1.60,
        target_distance_release_hold_sec=0.50,
        target_distance_release_confirm_frames=3,
        brake_distance_m=0.80,
        initial_target_confirm_frames=1,
        reverse_enable=True,
        reverse_start_distance_m=1.45,
        reverse_stop_distance_m=1.50,
        reverse_full_speed_distance_m=1.00,
        reverse_min_rpm=8,
        reverse_max_rpm=15,
        reverse_min_approach_delta_m=0.03,
        reverse_confirm_frames=2,
        reverse_radar_max_age_sec=0.25,
        forward_max_rpm=50,
    )

    def radar_frame(
        distance_m: float,
        *,
        persons=None,
        fusion_mode: str = "radar",
        source_detail: str = "matched",
        sample_age_sec: float = 0.05,
        obstacles: ObstacleState = ObstacleState(),
    ) -> SensorFrame:
        current_persons = [person] if persons is None else persons
        return SensorFrame(
            width=640,
            height=480,
            persons=current_persons,
            obstacles=obstacles,
            distance_m=distance_m,
            distance_state=DistanceState(
                source="vision_mmwave",
                raw_distance_m=distance_m,
                filtered_distance_m=distance_m,
                used_distance_m=distance_m,
                source_detail=source_detail,
                sample_age_sec=sample_age_sec,
                fusion_mode=fusion_mode,
                fusion_confidence=1.0,
                fusion_radar_distance_m=distance_m,
            ),
        )

    def activated_reverse_controller():
        active = FollowSafetyController(reverse_cfg)
        active.decide(1, radar_frame(1.80))
        first_stop = active.decide(2, radar_frame(1.48))
        first_approach = active.decide(3, radar_frame(1.42))
        active._reverse_last_approach_at = time.monotonic() - 0.03
        reverse = active.decide(4, radar_frame(1.36))
        if not first_stop.explicit_stop_requested or first_stop.reason != "target_distance_reached":
            raise AssertionError(f"1.48m must latch parking before reverse confirmation: {first_stop}")
        if first_approach.actions and first_approach.actions[0].kind == "backward":
            raise AssertionError(f"one approach confirmation must not reverse: {first_approach}")
        if (
            not reverse.actions
            or reverse.actions[0].kind != "backward"
            or reverse.reason != "target_approaching_reverse"
            or reverse.actions[0].speed_percent <= 0
        ):
            raise AssertionError(f"two stable approach confirmations must reverse: {reverse}")
        return active

    single_near_controller = FollowSafetyController(reverse_cfg)
    single_near_controller.decide(1, radar_frame(1.80))
    single_near = single_near_controller.decide(2, radar_frame(1.40))
    if single_near.actions and single_near.actions[0].kind == "backward":
        raise AssertionError(f"a single near radar frame must not reverse: {single_near}")

    immediate_depth_controller = FollowSafetyController(
        replace(
            reverse_cfg,
            distance_parking_enable=False,
            brake_distance_m=0.50,
            reverse_immediate_distance_m=1.35,
        )
    )
    immediate_depth_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[person],
            distance_m=None,
            distance_state=DistanceState(
                source="vision_depth",
                source_detail="center_patch_insufficient",
            ),
        ),
    )
    immediate_frame = SensorFrame(
        width=640,
        height=480,
        persons=[person],
        distance_m=1.30,
        distance_state=DistanceState(
            source="vision_depth",
            raw_distance_m=1.30,
            filtered_distance_m=1.30,
            used_distance_m=1.30,
            source_detail="depth_matched",
            sample_age_sec=0.03,
        ),
    )
    immediate_wait = immediate_depth_controller.decide(2, immediate_frame)
    if immediate_wait.actions and immediate_wait.actions[0].kind == "backward":
        raise AssertionError(f"one close Depth sample must not reverse: {immediate_wait}")
    immediate_depth_controller._reverse_last_approach_at = time.monotonic() - 0.03
    immediate_depth = immediate_depth_controller.decide(
        3,
        immediate_frame,
    )
    if (
        not immediate_depth.actions
        or immediate_depth.actions[0].kind != "backward"
        or immediate_depth.actions[0].speed_percent <= 0
    ):
        raise AssertionError(
            f"two fresh Depth samples below 1.35m must reverse: {immediate_depth}"
        )

    reverse_steer_controller = FollowSafetyController(
        replace(
            reverse_cfg,
            distance_parking_enable=False,
            brake_distance_m=0.50,
            reverse_immediate_distance_m=1.35,
            visible_steering_pid_enable=True,
        )
    )
    right_target = PersonTarget(
        (430, 80, 610, 450),
        track_id=31,
        confidence=0.9,
        area=66600,
    )
    reverse_steer_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[right_target], distance_m=None),
    )
    reverse_steer_frame = SensorFrame(
        width=640,
        height=480,
        persons=[right_target],
        distance_m=1.20,
        distance_state=DistanceState(
            source="vision_depth",
            raw_distance_m=1.20,
            filtered_distance_m=1.20,
            used_distance_m=1.20,
            source_detail="depth_matched",
            sample_age_sec=0.03,
        ),
    )
    reverse_steer_wait = reverse_steer_controller.decide(2, reverse_steer_frame)
    if reverse_steer_wait.actions and reverse_steer_wait.actions[0].kind == "backward":
        raise AssertionError(f"one close Depth sample must not start steered reverse: {reverse_steer_wait}")
    reverse_steer_controller._reverse_last_approach_at = time.monotonic() - 0.03
    reverse_steer = reverse_steer_controller.decide(
        3,
        reverse_steer_frame,
    )
    if (
        not reverse_steer.actions
        or reverse_steer.actions[0].kind != "backward"
        or reverse_steer.actions[0].steer_correction_rpm <= 6
        or reverse_steer.actions[0].steer_correction_rpm > 10
    ):
        raise AssertionError(
            "right-edge target must receive stronger yaw authority while reversing: "
            f"{reverse_steer}"
        )
    centered_reverse_target = PersonTarget(
        (230, 80, 410, 450),
        track_id=31,
        confidence=0.9,
        area=66600,
    )
    centered_reverse = reverse_steer_controller.decide(
        4,
        replace(reverse_steer_frame, persons=[centered_reverse_target]),
    )
    if (
        not centered_reverse.actions
        or centered_reverse.actions[0].kind != "backward"
        or centered_reverse.actions[0].steer_correction_rpm != 0
    ):
        raise AssertionError(
            f"reverse target inside x=0.40..0.60 must use zero wheel difference: {centered_reverse}"
        )

    hold_controller = FollowSafetyController(reverse_cfg)
    hold_controller.decide(1, radar_frame(1.80))
    hold_decision = hold_controller.decide(
        2,
        radar_frame(1.35, fusion_mode="radar_hold", source_detail="unmatched_hold"),
    )
    if hold_decision.actions and hold_decision.actions[0].kind == "backward":
        raise AssertionError(f"radar hold/visual fusion must never trigger reverse: {hold_decision}")

    unstable_controller = activated_reverse_controller()
    unstable_hold = unstable_controller.decide(
        5,
        radar_frame(1.30, fusion_mode="visual_encoder", source_detail="unmatched_hold"),
    )
    if (
        not unstable_hold.actions
        or unstable_hold.actions[0].kind != "backward"
        or unstable_hold.actions[0].speed_percent <= 0
        or unstable_hold.reason != "reverse_distance_missing_hold"
    ):
        raise AssertionError(f"active reverse must survive a short range dropout: {unstable_hold}")

    # 生产配置只允许红外硬停。已由新鲜 Depth 确认进入倒车后，单帧近距离
    # hold 应继续既有倒车，不能退出后被横向分支覆盖成向前差速。
    depth_reverse_cfg = replace(
        reverse_cfg,
        distance_parking_enable=False,
        brake_distance_m=0.50,
        reverse_start_distance_m=1.50,
        reverse_stop_distance_m=1.55,
        reverse_confirm_frames=1,
        reverse_max_rpm=100,
        forward_max_rpm=100,
        distance_pid_enable=True,
    )

    def depth_frame(distance_m: float, detail: str, age_sec: float, raw_distance_m=None) -> SensorFrame:
        return SensorFrame(
            width=640,
            height=480,
            persons=[person],
            distance_m=distance_m,
            distance_state=DistanceState(
                source="vision_depth",
                raw_distance_m=raw_distance_m,
                filtered_distance_m=distance_m,
                used_distance_m=distance_m,
                source_detail=detail,
                sample_age_sec=age_sec,
            ),
        )

    depth_reverse = FollowSafetyController(depth_reverse_cfg)
    depth_reverse.decide(1, depth_frame(1.80, "depth_matched", 0.03, 1.80))
    depth_reverse.decide(2, depth_frame(1.80, "depth_matched", 0.03, 1.80))
    reverse_started = depth_reverse.decide(3, depth_frame(1.35, "depth_matched", 0.03, 1.35))
    reverse_held = depth_reverse.decide(
        4,
        depth_frame(1.35, "far_background_guard_large_bbox_hold", 0.15),
    )
    if not reverse_started.actions or reverse_started.actions[0].kind != "backward":
        raise AssertionError(f"fresh approaching Depth must start reverse: {reverse_started}")
    if not 30 <= reverse_started.actions[0].speed_percent <= 60:
        raise AssertionError(f"reverse feedforward must start within the 30-60 RPM test band: {reverse_started}")
    if depth_reverse._distance_pid_last_update_at is None:
        raise AssertionError("reverse decision must preserve longitudinal PID state between Depth samples")
    if not reverse_held.actions or reverse_held.actions[0].kind != "backward":
        raise AssertionError(f"one fresh Depth hold frame must preserve active reverse: {reverse_held}")

    lost_reverse_controller = activated_reverse_controller()
    lost_reverse = lost_reverse_controller.decide(5, radar_frame(1.30, persons=[]))
    if not lost_reverse.explicit_stop_requested or lost_reverse.reason != "reverse_visual_lost_stop":
        raise AssertionError(f"reverse must stop as soon as the locked visual target disappears: {lost_reverse}")

    ir_reverse_controller = activated_reverse_controller()
    ir_reverse = ir_reverse_controller.decide(
        5,
        radar_frame(1.30, obstacles=ObstacleState(front=True)),
    )
    if not ir_reverse.explicit_stop_requested or ir_reverse.reason != "front_ir":
        raise AssertionError(f"IR safety must immediately stop reverse: {ir_reverse}")

    restored_controller = activated_reverse_controller()
    restored_wait = restored_controller.decide(5, radar_frame(1.50))
    if restored_wait.explicit_stop_requested or not restored_wait.actions:
        raise AssertionError(f"first restored sample must wait without releasing reverse: {restored_wait}")
    restored_controller._reverse_release_last_confirm_at = time.monotonic() - 0.03
    restored_stop = restored_controller.decide(6, radar_frame(1.50))
    if not restored_stop.explicit_stop_requested or restored_stop.reason != "reverse_target_distance_restored":
        raise AssertionError(f"reverse must stop after restoring 1.50m: {restored_stop}")
    release_1 = restored_controller.decide(7, radar_frame(1.60))
    restored_controller._target_release_started_at = time.monotonic() - 0.51
    release_2 = restored_controller.decide(8, radar_frame(1.60))
    release_3 = restored_controller.decide(9, radar_frame(1.60))
    if release_1.reason != "target_distance_release_wait" or release_2.reason != "target_distance_release_wait":
        raise AssertionError(f"1.60m restart must still pass time/frame confirmation: {release_1}, {release_2}")
    if release_3.explicit_stop_requested or not release_3.actions or release_3.actions[0].kind != "forward":
        raise AssertionError(f"stable 1.60m must release parking and resume following: {release_3}")
    print("stable_radar_reverse_distance_control: PASS")

    # 已在 1.2m 停车后，近距离丢失也必须走统一的连续漏检确认。确认
    # 窗口只发布零偏航；达到阈值后才允许按离开方向原地搜索。
    near_lost_controller = FollowSafetyController(
        FollowPolicyConfig(
            target_distance_m=1.20,
            target_distance_release_m=1.40,
            brake_distance_m=0.80,
            lost_confirm_frames=5,
            initial_target_confirm_frames=1,
        )
    )
    near_lost_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[person], distance_m=2.0),
    )
    near_stop = near_lost_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[person], distance_m=1.20),
    )
    if not near_stop.explicit_stop_requested or near_stop.reason != "target_distance_reached":
        raise AssertionError(f"centered 1.2m target must first park: {near_stop}")
    near_missing_frame = SensorFrame(
        width=640,
        height=480,
        persons=[],
        distance_m=1.20,
        distance_state=DistanceState(used_distance_m=1.20, target_latched=True),
    )
    for lost_frame_index in range(3, 7):
        near_wait = near_lost_controller.decide(lost_frame_index, near_missing_frame)
        if (
            near_wait.explicit_stop_requested
            or not near_wait.actions
            or near_wait.actions[0].kind != "stop"
            or near_wait.reason != "near_target_lost_direction_confirm"
        ):
            raise AssertionError(
                f"parked loss confirmation must publish zero yaw: {near_wait}"
            )
    near_lost = near_lost_controller.decide(7, near_missing_frame)
    if (
        near_lost.explicit_stop_requested
        or not near_lost.actions
        or near_lost.actions[0].kind != "rotate_right"
        or near_lost.actions[0].speed_percent != 0
        or near_lost.reason != "lost_wait_near_target_right"
    ):
        raise AssertionError(f"confirmed parked target loss must rotate in place: {near_lost}")

    # 极近距离硬刹车仍高于居中需求，0.8m 内绝不能原地旋转。
    hard_close_controller = FollowSafetyController(
        FollowPolicyConfig(
            target_distance_m=1.20,
            target_distance_release_m=1.40,
            brake_distance_m=0.80,
            lost_confirm_frames=5,
            initial_target_confirm_frames=1,
        )
    )
    hard_close_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[person], distance_m=2.0),
    )
    hard_close_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[person], distance_m=0.70),
    )
    hard_close_lost = hard_close_controller.decide(
        3,
        SensorFrame(
            width=640,
            height=480,
            persons=[],
            distance_m=0.70,
            distance_state=DistanceState(
                used_distance_m=0.70,
                target_latched=True,
                brake_latched=True,
            ),
        ),
    )
    if (
        not hard_close_lost.explicit_stop_requested
        or hard_close_lost.actions
        or hard_close_lost.reason != "distance_too_close"
    ):
        raise AssertionError(f"hard-close target loss must remain stopped: {hard_close_lost}")

    startup_cfg = FollowPolicyConfig(
        search_before_first_seen=False,
        initial_target_confirm_frames=2,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    startup_controller = FollowSafetyController(startup_cfg)
    d_start_empty = startup_controller.decide(1, SensorFrame(width=640, height=480, persons=[]))
    print("startup_wait:", d_start_empty.actions, d_start_empty.reason)
    if d_start_empty.actions or d_start_empty.explicit_stop_requested or d_start_empty.reason != "wait_first_person":
        raise AssertionError(f"startup with no person should idle in place, got {d_start_empty}")
    d_start_confirm_1 = startup_controller.decide(2, normal)
    print("startup_confirm_1:", d_start_confirm_1.actions, d_start_confirm_1.reason, startup_controller.active_target_id)
    if (
        len(d_start_confirm_1.actions) != 1
        or d_start_confirm_1.actions[0].kind != "stop"
        or d_start_confirm_1.actions[0].brake_hold
        or d_start_confirm_1.explicit_stop_requested
        or not d_start_confirm_1.soft_stop_requested
        or startup_controller.active_target_id is not None
    ):
        raise AssertionError(f"first visible uid should only enroll, got {d_start_confirm_1}")
    if d_start_confirm_1.reason != "initial_candidate_confirmation_hold":
        raise AssertionError(
            f"first visible uid should remain stopped while identity-pending, got {d_start_confirm_1.reason}"
        )
    d_start_confirm_2 = startup_controller.decide(3, normal)
    print(
        "startup_confirm_2:",
        [(a.kind, a.speed_percent, a.reason) for a in d_start_confirm_2.actions],
        d_start_confirm_2.reason,
        startup_controller.active_target_id,
    )
    if startup_controller.active_target_id != 1 or not d_start_confirm_2.actions:
        raise AssertionError(f"stable uid should become active target, got {d_start_confirm_2}")
    startup_controller.clear_active_target("test_button")
    startup_other_person = PersonTarget((250, 120, 390, 430), track_id=2, confidence=0.9, area=43400)
    d_clear_confirm_1 = startup_controller.decide(
        4,
        SensorFrame(width=640, height=480, persons=[startup_other_person], distance_m=2.0),
    )
    if (
        len(d_clear_confirm_1.actions) != 1
        or d_clear_confirm_1.actions[0].kind != "stop"
        or d_clear_confirm_1.actions[0].brake_hold
        or startup_controller.active_target_id is not None
        or d_clear_confirm_1.explicit_stop_requested
        or not d_clear_confirm_1.soft_stop_requested
        or d_clear_confirm_1.reason != "initial_candidate_confirmation_hold"
    ):
        raise AssertionError(f"manual clear should return to enrollment wait, got {d_clear_confirm_1}")
    d_clear_confirm_2 = startup_controller.decide(
        5,
        SensorFrame(width=640, height=480, persons=[startup_other_person], distance_m=2.0),
    )
    if startup_controller.active_target_id != 2 or not d_clear_confirm_2.actions:
        raise AssertionError(f"manual clear should allow a newly stable uid, got {d_clear_confirm_2}")

    # 启动阶段的单人几何兜底（-2）不是正式 ReID 锁定：漏一帧只能停车等待，
    # 不能因为 lost_confirm_frames=2 就直接进入原地搜索旋转。
    geometry_startup_controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=2,
            initial_target_confirm_frames=1,
        )
    )
    geometry_person = PersonTarget(
        (120, 100, 300, 430), track_id=-2, confidence=0.9, area=59400
    )
    d_geometry_visible = geometry_startup_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[geometry_person], distance_m=2.0),
    )
    if geometry_startup_controller.active_target_id != -2:
        raise AssertionError(
            "geometry fallback should be represented as the unconfirmed -2 target"
        )
    d_geometry_lost = geometry_startup_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[], distance_m=2.0),
    )
    if (
        not d_geometry_lost.explicit_stop_requested
        or not d_geometry_lost.clear_action_queue
        or not d_geometry_lost.stop_action_execution
        or d_geometry_lost.actions
        or d_geometry_lost.reason != "unconfirmed_target_wait"
        or geometry_startup_controller.search_state != "none"
    ):
        raise AssertionError(
            "unconfirmed geometry loss must stop without entering search: "
            f"{d_geometry_lost} state={geometry_startup_controller.search_state}"
        )
    d_geometry_lost_again = geometry_startup_controller.decide(
        3,
        SensorFrame(width=640, height=480, persons=[], distance_m=2.0),
    )
    if d_geometry_lost_again.reason != "unconfirmed_target_wait" or d_geometry_lost_again.actions:
        raise AssertionError(
            "repeated geometry-only loss must remain stopped, "
            f"got {d_geometry_lost_again}"
        )
    formal_person = PersonTarget(
        (120, 100, 300, 430), track_id=1, confidence=0.9, area=59400
    )
    d_formal_reid = geometry_startup_controller.decide(
        4,
        SensorFrame(width=640, height=480, persons=[formal_person], distance_m=2.0),
    )
    if geometry_startup_controller.active_target_id != 1:
        raise AssertionError(
            "a formal ReID target should be allowed to replace startup geometry fallback"
        )

    edge_controller = FollowSafetyController(cfg)
    left_edge_person = PersonTarget((10, 120, 80, 430), track_id=1, confidence=0.9, area=24850)
    left_edge = SensorFrame(width=640, height=480, persons=[left_edge_person], distance_m=2.0)
    d_edge = edge_controller.decide(1, left_edge)
    print("left_edge:", [(a.kind, a.speed_percent, a.reason) for a in d_edge.actions], d_edge.reason)
    if not d_edge.actions or any(a.kind != "steer_left" for a in d_edge.actions):
        raise AssertionError(f"off-center visible target should use wheel-differential steer, got {d_edge.actions}")
    if d_edge.actions[0].steer_outer_ratio_percent <= d_edge.actions[0].steer_inner_ratio_percent:
        raise AssertionError(f"steer action should make the outer wheel faster, got {d_edge.actions[0]}")
    edge_controller.set_last_dispatched("rotate_left")
    d_edge_lost = edge_controller.decide(2, SensorFrame(width=640, height=480, persons=[], distance_m=2.0))
    print("edge_lost:", [(a.kind, a.reason) for a in d_edge_lost.actions], d_edge_lost.reason)
    if not d_edge_lost.waiting_lost_confirm:
        raise AssertionError(f"missing target after rotate should enter lost confirm, got {d_edge_lost.reason}")
    if (
        not d_edge_lost.actions
        or d_edge_lost.actions[0].kind != "rotate_left"
        or d_edge_lost.explicit_stop_requested
    ):
        raise AssertionError(
            "a single missing frame must remove longitudinal speed but preserve "
            f"bounded same-direction yaw, got {d_edge_lost}"
        )
    if d_edge_lost.reason != "lost_wait_yaw_left":
        raise AssertionError(f"lost confirmation should expose its yaw hint, got {d_edge_lost.reason}")
    if edge_controller._search_rotation_started_at is not None:
        raise AssertionError("pre-search must not consume the confirmed-search timeout")
    d_edge_recovered = edge_controller.decide(3, left_edge)
    if not d_edge_recovered.actions:
        raise AssertionError(f"same target should resume visual correction after a one-frame dropout, got {d_edge_recovered}")
    if not d_edge_recovered.clear_action_queue or not d_edge_recovered.stop_action_execution:
        raise AssertionError(f"visual recovery must cancel a stale rotate pulse before resuming, got {d_edge_recovered}")

    dropout_controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=6,
            steer_min_hold_sec=0.50,
            steer_lost_hold_frames=3,
            steer_lost_hold_max_sec=0.80,
            max_forward_percent=20,
            forward_speed_far_percent=20,
        )
    )
    right_steer_person = PersonTarget((430, 120, 510, 430), track_id=7, confidence=0.9, area=24800)
    d_steer_start = dropout_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[right_steer_person], distance_m=2.0),
    )
    if not d_steer_start.actions or d_steer_start.actions[0].kind != "steer_right":
        raise AssertionError(f"right-side target should start steer_right, got {d_steer_start}")
    dropout_controller.set_last_dispatched("steer_right")
    for frame_index in range(2, 5):
        d_dropout = dropout_controller.decide(
            frame_index,
            SensorFrame(width=640, height=480, persons=[], distance_m=2.0),
        )
        print("steer_dropout_hold:", frame_index, [(a.kind, a.reason) for a in d_dropout.actions])
        if not d_dropout.actions or d_dropout.actions[0].kind != "rotate_right":
            raise AssertionError(f"short dropout must retain only in-place right yaw, got {d_dropout}")
        if d_dropout.reason != "lost_wait_yaw_right":
            raise AssertionError(f"yaw hold should have a distinct reason, got {d_dropout.reason}")
        dropout_controller.set_last_dispatched("steer_right")

    # Even after the frame allowance, the 0.50s minimum prevents an immediate
    # replacement. Once the 0.80s forward-steer hold expires, remove its
    # longitudinal component and continue bounded in-place yaw.
    d_min_hold = dropout_controller.decide(
        5,
        SensorFrame(width=640, height=480, persons=[], distance_m=2.0),
    )
    if not d_min_hold.actions or d_min_hold.actions[0].kind != "rotate_right":
        raise AssertionError(f"fourth dropout must still retain only in-place yaw, got {d_min_hold}")
    dropout_controller.set_last_dispatched("steer_right")
    dropout_controller._last_visible_steer_started_at = time.monotonic() - 1.0
    dropout_controller._lost_started_at = time.monotonic() - 0.81
    d_hold_expired = dropout_controller.decide(
        6,
        SensorFrame(width=640, height=480, persons=[], distance_m=2.0),
    )
    if (
        not d_hold_expired.actions
        or d_hold_expired.actions[0].kind != "rotate_right"
        or d_hold_expired.explicit_stop_requested
    ):
        raise AssertionError(f"expired steer hold must retain only bounded yaw, got {d_hold_expired}")
    if d_hold_expired.reason != "lost_wait_yaw_right":
        raise AssertionError(f"expired hold should report bounded yaw, got {d_hold_expired.reason}")

    close_dropout_controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=5,
            steer_min_hold_sec=0.50,
            steer_lost_hold_frames=3,
            steer_lost_hold_max_sec=0.80,
            target_distance_m=1.5,
            brake_distance_m=0.8,
            max_forward_percent=20,
            forward_speed_far_percent=20,
        )
    )
    close_dropout_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[right_steer_person], distance_m=2.0),
    )
    close_dropout_controller.set_last_dispatched("steer_right")
    d_close_dropout = close_dropout_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[], distance_m=1.5),
    )
    if (
        not d_close_dropout.actions
        or d_close_dropout.actions[0].kind != "rotate_right"
        or d_close_dropout.explicit_stop_requested
    ):
        raise AssertionError(
            "1.50m target distance must remove longitudinal dropout motion while "
            f"preserving bounded yaw, got {d_close_dropout}"
        )

    forward_dropout_controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=6,
            lost_forward_hold_rpm=20,
            lost_forward_hold_max_sec=0.80,
            lost_forward_hold_min_distance_m=1.50,
            forward_min_rpm=20,
            forward_max_rpm=50,
            min_forward_percent=40,
            max_forward_percent=100,
        )
    )
    center_person = PersonTarget((270, 120, 370, 430), track_id=11, confidence=0.9, area=31000)
    d_forward_start = forward_dropout_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[center_person], distance_m=3.0),
    )
    if not d_forward_start.actions or d_forward_start.actions[0].kind != "forward":
        raise AssertionError(f"centered distant target should start forward, got {d_forward_start}")
    forward_dropout_controller.set_last_dispatched("forward")
    for frame_index in range(2, 7):
        d_forward_dropout = forward_dropout_controller.decide(
            frame_index,
            SensorFrame(width=640, height=480, persons=[], distance_m=None),
        )
        if (
            not d_forward_dropout.actions
            or d_forward_dropout.actions[0].kind != "rotate_right"
            or d_forward_dropout.actions[0].speed_percent != 0
            or d_forward_dropout.explicit_stop_requested
        ):
            raise AssertionError(
                f"dropout frame {frame_index} must replace blind forward with in-place yaw, got {d_forward_dropout}"
            )
        if frame_index == 2:
            forward_dropout_controller.set_last_dispatched("rotate_right")

    d_forward_lost_confirmed = forward_dropout_controller.decide(
        7,
        SensorFrame(width=640, height=480, persons=[], distance_m=None),
    )
    if d_forward_lost_confirmed.actions and d_forward_lost_confirmed.actions[0].kind == "forward":
        raise AssertionError(f"six missing frames must end blind forward hold, got {d_forward_lost_confirmed}")

    camera_range_hold_controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_forward_hold_rpm=20,
            lost_forward_hold_max_sec=0.80,
            lost_forward_hold_min_distance_m=1.50,
            forward_min_rpm=20,
            forward_max_rpm=50,
            min_forward_percent=40,
            max_forward_percent=100,
        )
    )
    camera_range_hold_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[center_person], distance_m=3.0),
    )
    camera_range_hold_controller.set_last_dispatched("forward")
    d_camera_range_hold = camera_range_hold_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[center_person], distance_m=None),
    )
    if not d_camera_range_hold.actions or d_camera_range_hold.reason != "distance_missing_camera_hold":
        raise AssertionError(f"visible target should survive a brief range dropout, got {d_camera_range_hold}")
    if d_camera_range_hold.actions[0].speed_percent != 40 or d_camera_range_hold.explicit_stop_requested:
        raise AssertionError(f"range dropout should use 20 rpm without STOP, got {d_camera_range_hold}")

    target_range_hold_controller = FollowSafetyController(
        FollowPolicyConfig(
            target_distance_m=1.5,
            lost_forward_hold_min_distance_m=1.50,
            initial_target_confirm_frames=1,
        )
    )
    target_range_hold_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[center_person], distance_m=1.5),
    )
    target_range_hold_controller.set_last_dispatched("forward")
    d_target_range_missing = target_range_hold_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[center_person], distance_m=None),
    )
    if d_target_range_missing.actions or not d_target_range_missing.explicit_stop_requested:
        raise AssertionError(
            f"a missing range after 1.50m must not resume forward hold: {d_target_range_missing}"
        )

    hysteresis_cfg = FollowPolicyConfig(
        center_left_ratio=0.30,
        center_right_ratio=0.70,
        steer_enter_left_ratio=0.28,
        steer_enter_right_ratio=0.72,
        steer_release_left_ratio=0.35,
        steer_release_right_ratio=0.65,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    hysteresis_controller = FollowSafetyController(hysteresis_cfg)
    boundary_person = PersonTarget((680, 120, 740, 430), track_id=1, confidence=0.9, area=18600)
    boundary_frame = SensorFrame(width=1000, height=480, persons=[boundary_person], distance_m=2.0)
    d_enter = hysteresis_controller.decide(1, boundary_frame)
    print("steer_hysteresis_enter:", [(a.kind, a.reason) for a in d_enter.actions], d_enter.reason)
    if not d_enter.actions or any(a.kind != "forward" for a in d_enter.actions):
        raise AssertionError(f"target inside enter band should keep forward, got {d_enter.actions}")
    hysteresis_controller.set_last_dispatched("steer_right")
    d_release = hysteresis_controller.decide(2, boundary_frame)
    print("steer_hysteresis_release:", [(a.kind, a.reason) for a in d_release.actions], d_release.reason)
    if not d_release.actions or any(a.kind != "steer_right" for a in d_release.actions):
        raise AssertionError(f"active steer should continue until release band, got {d_release.actions}")
    hysteresis_controller.set_last_dispatched("rotate_right")
    d_rotate_release = hysteresis_controller.decide(3, boundary_frame)
    if (
        not d_rotate_release.actions
        or d_rotate_release.actions[0].kind != "steer_right"
    ):
        raise AssertionError(
            "active in-place yaw must use the same release hysteresis as steer: "
            f"{d_rotate_release.actions}"
        )

    # 运行配置同时提供 deadzone 和 enter/release。显式回差必须优先，
    # 否则 0.15 deadzone 会把 0.42/0.58 的快速居中阈值遮蔽成 0.35/0.65。
    runtime_band_controller = FollowSafetyController(
        FollowPolicyConfig(
            center_deadzone_ratio=0.15,
            steer_enter_left_ratio=0.42,
            steer_enter_right_ratio=0.58,
            steer_release_left_ratio=0.46,
            steer_release_right_ratio=0.54,
            max_forward_percent=20,
            forward_speed_far_percent=20,
        )
    )
    runtime_right_person = PersonTarget((570, 120, 630, 430), track_id=1, confidence=0.9, area=18600)
    d_runtime_band = runtime_band_controller.decide(
        1,
        SensorFrame(width=1000, height=480, persons=[runtime_right_person], distance_m=2.0),
    )
    if not d_runtime_band.actions or d_runtime_band.actions[0].kind != "steer_right":
        raise AssertionError(f"explicit enter band should override deadzone, got {d_runtime_band}")

    motion_controller = FollowSafetyController(
        FollowPolicyConfig(
            center_deadzone_ratio=0.08,
            steer_enter_left_ratio=0.42,
            steer_enter_right_ratio=0.58,
            steer_release_left_ratio=0.46,
            steer_release_right_ratio=0.54,
            visible_motion_history_frames=4,
            visible_motion_min_ratio=0.015,
            visible_motion_projection_gain=0.80,
            visible_motion_strong_ratio=0.05,
            max_forward_percent=20,
            forward_speed_far_percent=20,
        )
    )
    for frame_index, center_ratio, expected_kind in (
        (1, 0.50, "forward"),
        (2, 0.52, "forward"),
        (3, 0.57, "steer_right"),
    ):
        center_x = center_ratio * 1000.0
        moving_person = PersonTarget(
            (center_x - 30.0, 120, center_x + 30.0, 430),
            track_id=8,
            confidence=0.9,
            area=18600,
        )
        d_motion = motion_controller.decide(
            frame_index,
            SensorFrame(width=1000, height=480, persons=[moving_person], distance_m=2.0),
        )
        actual_kind = d_motion.actions[0].kind if d_motion.actions else "none"
        if actual_kind != expected_kind:
            raise AssertionError(
                f"visible motion lead frame={frame_index} expected {expected_kind}, got {d_motion}"
            )
        if frame_index == 3 and d_motion.reason != "person_right_strong_motion":
            raise AssertionError(f"outward trajectory should use strong motion correction, got {d_motion.reason}")
        if motion_controller._visible_motion_samples:
            latest_motion_sample = motion_controller._visible_motion_samples[-1]
            motion_controller._visible_motion_samples[-1] = (
                latest_motion_sample[0],
                latest_motion_sample[1] - 0.10,
                latest_motion_sample[2],
            )
        motion_controller.set_last_dispatched(actual_kind)

    # 微调轮差必须随横向误差连续增大，而不是只在普通/强两组固定参数之间跳变。
    profile_cfg = FollowPolicyConfig(
        steer_enter_left_ratio=0.43,
        steer_enter_right_ratio=0.57,
        steer_release_left_ratio=0.46,
        steer_release_right_ratio=0.54,
        visible_steer_fine_inner_ratio_percent=90,
        visible_steer_fine_outer_ratio_percent=100,
        visible_steer_inner_ratio_percent=80,
        visible_steer_outer_ratio_percent=105,
        visible_steer_strong_inner_ratio_percent=70,
        visible_steer_strong_outer_ratio_percent=110,
        visible_steer_strong_margin_ratio=0.20,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    side_profiles = {}
    for side, center_ratios, expected_kind in (
        ("left", (0.42, 0.32, 0.21), "steer_left"),
        ("right", (0.58, 0.68, 0.79), "steer_right"),
    ):
        profiles = []
        for center_ratio in center_ratios:
            profile_controller = FollowSafetyController(profile_cfg)
            center_x = center_ratio * 1000.0
            profile_person = PersonTarget(
                (center_x - 30.0, 120, center_x + 30.0, 430),
                track_id=30,
                confidence=0.9,
                area=18600,
            )
            profile_decision = profile_controller.decide(
                1,
                SensorFrame(width=1000, height=480, persons=[profile_person], distance_m=2.0),
            )
            if not profile_decision.actions or profile_decision.actions[0].kind != expected_kind:
                raise AssertionError(f"{side} proportional steer missing at x={center_ratio}: {profile_decision}")
            profile_action = profile_decision.actions[0]
            profiles.append(
                (profile_action.steer_inner_ratio_percent, profile_action.steer_outer_ratio_percent)
            )
        side_profiles[side] = profiles
        if not (profiles[0][0] > profiles[1][0] > profiles[2][0]):
            raise AssertionError(f"{side} inner wheel must slow progressively: {profiles}")
        if not (profiles[0][1] < profiles[1][1] < profiles[2][1]):
            raise AssertionError(f"{side} outer wheel must accelerate progressively: {profiles}")
    print("proportional_visible_steer:", side_profiles)
    if side_profiles["left"] != side_profiles["right"]:
        raise AssertionError(f"left/right proportional steer must stay symmetric: {side_profiles}")

    # 车身刚从左侧修回中心时，框中心会快速向右移动。即使外推越过右阈值，
    # 当前中心仍离右边界较远时也不能立即反向强修，否则会形成左右摆动。
    reverse_guard_controller = FollowSafetyController(
        FollowPolicyConfig(
            steer_enter_left_ratio=0.40,
            steer_enter_right_ratio=0.60,
            steer_release_left_ratio=0.44,
            steer_release_right_ratio=0.56,
            visible_motion_history_frames=4,
            visible_motion_min_ratio=0.03,
            visible_motion_projection_gain=0.80,
            visible_motion_strong_ratio=0.05,
            max_forward_percent=20,
            forward_speed_far_percent=20,
        )
    )
    for frame_index, center_ratio, expected_kind in (
        (1, 0.33, "steer_left"),
        (2, 0.45, "forward"),
        (3, 0.524, "forward"),
    ):
        center_x = center_ratio * 1000.0
        moving_person = PersonTarget(
            (center_x - 30.0, 120, center_x + 30.0, 430),
            track_id=18,
            confidence=0.9,
            area=18600,
        )
        decision = reverse_guard_controller.decide(
            frame_index,
            SensorFrame(width=1000, height=480, persons=[moving_person], distance_m=2.0),
        )
        actual_kind = decision.actions[0].kind if decision.actions else "none"
        if actual_kind != expected_kind:
            raise AssertionError(
                f"center-crossing motion must not reverse steer frame={frame_index}: {decision}"
            )
        reverse_guard_controller.set_last_dispatched(actual_kind)

    pulse_controller = FollowSafetyController(
        FollowPolicyConfig(
            action_cooldown=2,
            steer_enter_left_ratio=0.40,
            steer_enter_right_ratio=0.60,
            max_forward_percent=20,
            forward_speed_far_percent=20,
        )
    )
    pulse_person = PersonTarget((250, 120, 350, 430), track_id=19, confidence=0.9, area=31000)
    pulse_frame = SensorFrame(width=1000, height=480, persons=[pulse_person], distance_m=2.0)
    first_pulse = pulse_controller.decide(1, pulse_frame)
    pause_frame = pulse_controller.decide(2, pulse_frame)
    second_pulse = pulse_controller.decide(3, pulse_frame)
    if not first_pulse.actions or first_pulse.actions[0].kind != "steer_left":
        raise AssertionError(f"first visible steer pulse missing: {first_pulse}")
    if pause_frame.explicit_stop_requested or not pause_frame.actions:
        raise AssertionError(f"visible steer hold must not request STOP: {pause_frame}")
    if pause_frame.actions[0].kind != "steer_left" or pause_frame.reason != "visible_turn_hold":
        raise AssertionError(f"visible steer cooldown must retain the prior direction: {pause_frame}")
    if not second_pulse.actions or second_pulse.actions[0].kind != "steer_left":
        raise AssertionError(f"visible steer must resume only after observation frame: {second_pulse}")

    visible_rotate_cfg = FollowPolicyConfig(
        center_left_ratio=0.30,
        center_right_ratio=0.70,
        steer_enter_left_ratio=0.28,
        steer_enter_right_ratio=0.72,
        visible_rotate_left_ratio=0.25,
        visible_rotate_right_ratio=0.75,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    visible_rotate_controller = FollowSafetyController(visible_rotate_cfg)
    steer_zone_person = PersonTarget((710, 120, 750, 430), track_id=1, confidence=0.9, area=12400)
    d_steer_zone = visible_rotate_controller.decide(
        1,
        SensorFrame(width=1000, height=480, persons=[steer_zone_person], distance_m=2.0),
    )
    print("visible_steer_zone:", [(a.kind, a.reason) for a in d_steer_zone.actions], d_steer_zone.reason)
    if not d_steer_zone.actions or d_steer_zone.actions[0].kind != "steer_right":
        raise AssertionError(f"target between steer and rotate boundary should steer, got {d_steer_zone.actions}")
    visible_rotate_controller.set_last_dispatched("forward")
    rotate_zone_person = PersonTarget((750, 120, 790, 430), track_id=1, confidence=0.9, area=12400)
    d_rotate_zone = visible_rotate_controller.decide(
        2,
        SensorFrame(width=1000, height=480, persons=[rotate_zone_person], distance_m=2.0),
    )
    print("visible_rotate_zone:", [(a.kind, a.reason) for a in d_rotate_zone.actions], d_rotate_zone.reason)
    if not d_rotate_zone.actions or d_rotate_zone.actions[0].kind != "steer_right":
        raise AssertionError(f"visible target must keep differential steering at the edge, got {d_rotate_zone}")
    if d_rotate_zone.reason == "person_right_rotate":
        raise AssertionError("visible target must never enter the lost-target rotate action")

    centered_cfg = FollowPolicyConfig(
        center_deadzone_ratio=0.08,
        visible_rotate_left_ratio=0.08,
        visible_rotate_right_ratio=0.92,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    centered_controller = FollowSafetyController(centered_cfg)
    centered_frames = (
        (1, PersonTarget((210, 120, 290, 430), track_id=9, confidence=0.9, area=24800), "steer_left"),
        (2, PersonTarget((280, 120, 360, 430), track_id=9, confidence=0.9, area=24800), "forward"),
        (3, PersonTarget((350, 120, 430, 430), track_id=9, confidence=0.9, area=24800), "steer_right"),
    )
    for frame_index, centered_person, expected_kind in centered_frames:
        decision = centered_controller.decide(
            frame_index,
            SensorFrame(width=640, height=480, persons=[centered_person], distance_m=2.0),
        )
        actual_kinds = [action.kind for action in decision.actions]
        print("symmetric_center_deadzone:", frame_index, actual_kinds, decision.reason)
        if actual_kinds != [expected_kind]:
            raise AssertionError(
                f"bbox center feedback should produce {expected_kind}, got {actual_kinds} ({decision.reason})"
            )

    widened_center_controller = FollowSafetyController(
        FollowPolicyConfig(
            center_left_ratio=0.40,
            center_right_ratio=0.60,
            center_deadzone_ratio=0.10,
            steer_enter_left_ratio=0.40,
            steer_enter_right_ratio=0.60,
            steer_release_left_ratio=0.42,
            steer_release_right_ratio=0.58,
            visible_steering_pid_enable=False,
            visible_motion_min_ratio=1.0,
            visible_motion_strong_ratio=1.0,
            max_forward_percent=20,
            forward_speed_far_percent=20,
        )
    )
    for frame_index, center_ratio, expected_kind in (
        (1, 0.40, "forward"),
        (2, 0.59, "forward"),
        (3, 0.39, "steer_left"),
        (4, 0.41, "steer_left"),
        (5, 0.42, "forward"),
        (6, 0.61, "steer_right"),
        (7, 0.59, "steer_right"),
        (8, 0.57, "forward"),
    ):
        center_x = center_ratio * 1000.0
        widened_person = PersonTarget(
            (center_x - 30.0, 120, center_x + 30.0, 430),
            track_id=10,
            confidence=0.9,
            area=18600,
        )
        decision = widened_center_controller.decide(
            frame_index,
            SensorFrame(width=1000, height=480, persons=[widened_person], distance_m=2.0),
        )
        actual_kind = decision.actions[0].kind if decision.actions else "none"
        if actual_kind != expected_kind:
            raise AssertionError(
                f"0.40-0.60 center band/hysteresis mismatch at x={center_ratio}: {decision}"
            )
        widened_center_controller.set_last_dispatched(actual_kind)

    blocked_search_cfg = FollowPolicyConfig(
        brake_distance_m=0.8,
        search_before_first_seen=True,
        side_ir_blocks_rotation=False,
        search_rotate_front_block_enable=False,
        search_rotate_distance_block_enable=True,
    )
    blocked_search_controller = FollowSafetyController(blocked_search_cfg)
    near_empty = SensorFrame(
        width=640,
        height=480,
        persons=[],
        distance_m=0.72,
        distance_state=DistanceState(used_distance_m=0.72, brake_latched=True),
    )
    d_blocked = blocked_search_controller.decide(1, near_empty)
    print("search_rotate_blocked:", d_blocked.explicit_stop_requested, d_blocked.reason)
    if not d_blocked.explicit_stop_requested or d_blocked.actions:
        raise AssertionError(f"near obstacle should block free-search rotate, got {d_blocked}")
    if d_blocked.reason != "search_both_sides_blocked":
        raise AssertionError(f"blocked free-search should explain both sides blocked, got {d_blocked.reason}")

    front_empty = SensorFrame(width=640, height=480, persons=[], obstacles=ObstacleState(front=True))
    d_front_search = blocked_search_controller.decide(2, front_empty)
    print("front_ir_stops_search:", d_front_search.explicit_stop_requested, d_front_search.reason)
    if not d_front_search.explicit_stop_requested or d_front_search.actions or d_front_search.reason != "front_ir":
        raise AssertionError(f"front IR should stop free-search motion, got {d_front_search}")
    if not d_front_search.clear_action_queue or not d_front_search.stop_action_execution:
        raise AssertionError(f"IR stop should clear queued and current actions, got {d_front_search}")

    side_blocked_search_cfg = FollowPolicyConfig(
        search_before_first_seen=True,
        side_ir_blocks_rotation=True,
        search_rotate_front_block_enable=False,
        search_rotate_distance_block_enable=False,
    )
    right_blocked_search_controller = FollowSafetyController(side_blocked_search_cfg)
    d_right_blocked_search = right_blocked_search_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[], obstacles=ObstacleState(right=True)),
    )
    print("right_ir_blocks_search_right:", d_right_blocked_search.explicit_stop_requested, d_right_blocked_search.reason)
    if (
        not d_right_blocked_search.explicit_stop_requested
        or d_right_blocked_search.actions
        or d_right_blocked_search.reason != "right_ir"
    ):
        raise AssertionError(f"right IR should unconditionally stop search motion, got {d_right_blocked_search}")

    left_blocked_search_controller = FollowSafetyController(side_blocked_search_cfg)
    left_blocked_search_controller.search_state = "searching"
    left_blocked_search_controller.search_direction = "left"
    d_left_blocked_search = left_blocked_search_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[], obstacles=ObstacleState(left=True)),
    )
    print("left_ir_blocks_search_left:", d_left_blocked_search.explicit_stop_requested, d_left_blocked_search.reason)
    if (
        not d_left_blocked_search.explicit_stop_requested
        or d_left_blocked_search.actions
        or d_left_blocked_search.reason != "left_ir"
    ):
        raise AssertionError(f"left IR should unconditionally stop search motion, got {d_left_blocked_search}")

    center_person = PersonTarget((430, 120, 570, 430), track_id=1, confidence=0.9, area=43400)
    blocked_clear_cfg = FollowPolicyConfig(
        lost_confirm_frames=1,
        lost_confirm_sec=0.0,
        search_before_first_seen=False,
        side_ir_blocks_rotation=True,
        search_rotate_front_block_enable=False,
        search_rotate_distance_block_enable=False,
        distance_missing_forward_percent=35,
        min_forward_percent=10,
        max_forward_percent=60,
        forward_speed_far_percent=20,
    )
    blocked_clear_controller = FollowSafetyController(blocked_clear_cfg)
    blocked_clear_controller.decide(1, SensorFrame(width=1000, height=480, persons=[center_person], distance_m=2.0))
    d_clear_step_1 = blocked_clear_controller.decide(
        2,
        SensorFrame(width=1000, height=480, persons=[], obstacles=ObstacleState(right=True)),
    )
    print("right_ir_disables_clear_forward:", d_clear_step_1.explicit_stop_requested, d_clear_step_1.reason)
    if not d_clear_step_1.explicit_stop_requested or d_clear_step_1.actions or d_clear_step_1.reason != "right_ir":
        raise AssertionError(f"right IR must not generate a forward clear step, got {d_clear_step_1}")

    ir_policy_cfg = FollowPolicyConfig(
        center_left_ratio=0.30,
        center_right_ratio=0.70,
        visible_rotate_left_ratio=0.25,
        visible_rotate_right_ratio=0.75,
        distance_missing_forward_percent=35,
        min_forward_percent=10,
        max_forward_percent=60,
        forward_speed_le_2_1_percent=55,
        search_rotate_front_block_enable=False,
    )
    ir_center_controller = FollowSafetyController(ir_policy_cfg)
    d_front_center = ir_center_controller.decide(
        1,
        SensorFrame(width=1000, height=480, persons=[center_person], distance_m=2.0, obstacles=ObstacleState(front=True)),
    )
    print("front_ir_center:", d_front_center.explicit_stop_requested, d_front_center.reason)
    if not d_front_center.explicit_stop_requested or d_front_center.reason != "front_ir":
        raise AssertionError(f"front IR should block centered forward, got {d_front_center}")

    front_steer_person = PersonTarget((710, 120, 750, 430), track_id=1, confidence=0.9, area=12400)
    d_front_steer = ir_center_controller.decide(
        2,
        SensorFrame(width=1000, height=480, persons=[front_steer_person], distance_m=2.0, obstacles=ObstacleState(front=True)),
    )
    print("front_ir_steer_zone:", [(a.kind, a.reason) for a in d_front_steer.actions], d_front_steer.explicit_stop_requested, d_front_steer.reason)
    if not d_front_steer.explicit_stop_requested or d_front_steer.reason != "front_ir" or d_front_steer.actions:
        raise AssertionError(f"front IR should block steer without creating rotate, got {d_front_steer}")

    front_rotate_person = PersonTarget((910, 120, 960, 430), track_id=1, confidence=0.9, area=15500)
    d_front_rotate = ir_center_controller.decide(
        3,
        SensorFrame(width=1000, height=480, persons=[front_rotate_person], distance_m=2.0, obstacles=ObstacleState(front=True)),
    )
    print("front_ir_stops_visible_rotate:", d_front_rotate.explicit_stop_requested, d_front_rotate.reason)
    if not d_front_rotate.explicit_stop_requested or d_front_rotate.actions or d_front_rotate.reason != "front_ir":
        raise AssertionError(f"front IR should stop visible rotate_right, got {d_front_rotate}")

    right_steer_person = PersonTarget((710, 120, 750, 430), track_id=1, confidence=0.9, area=12400)
    ir_does_not_rewrite_controller = FollowSafetyController(ir_policy_cfg)
    d_side_ir_keeps_steer = ir_does_not_rewrite_controller.decide(
        1,
        SensorFrame(width=1000, height=480, persons=[right_steer_person], distance_m=2.0, obstacles=ObstacleState(left=True)),
    )
    print("left_ir_stops_opposite_steer:", d_side_ir_keeps_steer.explicit_stop_requested, d_side_ir_keeps_steer.reason)
    if not d_side_ir_keeps_steer.explicit_stop_requested or d_side_ir_keeps_steer.actions or d_side_ir_keeps_steer.reason != "left_ir":
        raise AssertionError(f"left IR should stop even an opposite-direction steer, got {d_side_ir_keeps_steer}")

    left_person = PersonTarget((40, 120, 90, 430), track_id=1, confidence=0.9, area=15500)
    right_person = PersonTarget((910, 120, 960, 430), track_id=1, confidence=0.9, area=15500)
    ir_left_block_controller = FollowSafetyController(ir_policy_cfg)
    d_left_block = ir_left_block_controller.decide(
        1,
        SensorFrame(width=1000, height=480, persons=[left_person], distance_m=2.0, obstacles=ObstacleState(left=True)),
    )
    print("left_ir_blocks_left_rotate:", d_left_block.explicit_stop_requested, d_left_block.reason)
    if not d_left_block.explicit_stop_requested or d_left_block.reason != "left_ir":
        raise AssertionError(f"left IR should block rotate_left toward the left side, got {d_left_block}")
    if not ir_left_block_controller._is_action_blocked(  # noqa: SLF001 - direct policy gate regression
        ControlAction.steer_left(35, 100, 130, "test_left_steer"),
        SensorFrame(width=1000, height=480, obstacles=ObstacleState(left=True)),
    ):
        raise AssertionError("left IR should block steer_left at the action gate")
    if not ir_left_block_controller._is_action_blocked(  # noqa: SLF001 - direct policy gate regression
        ControlAction.steer_right(35, 100, 130, "test_right_steer"),
        SensorFrame(width=1000, height=480, obstacles=ObstacleState(right=True)),
    ):
        raise AssertionError("right IR should block steer_right at the action gate")

    ir_opposite_controller = FollowSafetyController(ir_policy_cfg)
    d_left_allows_right = ir_opposite_controller.decide(
        1,
        SensorFrame(width=1000, height=480, persons=[right_person], distance_m=2.0, obstacles=ObstacleState(left=True)),
    )
    print("left_ir_stops_right_rotate:", d_left_allows_right.explicit_stop_requested, d_left_allows_right.reason)
    if not d_left_allows_right.explicit_stop_requested or d_left_allows_right.actions or d_left_allows_right.reason != "left_ir":
        raise AssertionError(f"left IR should stop even an opposite-direction rotate, got {d_left_allows_right}")

    ir_escape_controller = FollowSafetyController(ir_policy_cfg)
    d_side_escape = ir_escape_controller.decide(
        1,
        SensorFrame(
            width=1000,
            height=480,
            persons=[center_person],
            distance_m=2.0,
            obstacles=ObstacleState(left=True, right=True),
        ),
    )
    print("side_ir_center_stop:", d_side_escape.explicit_stop_requested, d_side_escape.reason)
    if not d_side_escape.explicit_stop_requested or d_side_escape.actions or d_side_escape.reason != "left_ir":
        raise AssertionError(f"centered target with side IR should stop, got {d_side_escape}")

    other_person = PersonTarget((250, 120, 390, 430), track_id=2, confidence=0.9, area=43400)
    switched = SensorFrame(width=640, height=480, persons=[other_person], distance_m=2.0)
    d_switch = controller.decide(2, switched)
    print(
        "switch_guard:",
        [(a.kind, a.speed_percent, a.reason) for a in d_switch.actions],
        d_switch.explicit_stop_requested,
        d_switch.reason,
    )
    if not d_switch.actions or d_switch.actions[0].kind != "rotate_right":
        raise AssertionError("controller must search from the last reliable center, not follow the other track")
    if not d_switch.waiting_lost_confirm:
        raise AssertionError(f"different single track should be treated as lost target, got {d_switch.reason}")
    if d_switch.explicit_stop_requested or d_switch.actions[0].speed_percent != 0:
        raise AssertionError("first lost-confirm frame should use zero-longitudinal yaw")
    if d_switch.reason != "lost_wait_yaw_right":
        raise AssertionError(f"first lost-confirm frame should explain fixed yaw, got {d_switch.reason}")

    loss_direction_cfg = FollowPolicyConfig(
        lost_confirm_sec=3.0,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    loss_direction_controller = FollowSafetyController(loss_direction_cfg)
    left_visible = SensorFrame(
        width=640,
        height=480,
        persons=[PersonTarget((60, 100, 200, 430), track_id=1, confidence=0.9, area=46200)],
        distance_m=2.0,
    )
    loss_direction_controller.decide(1, left_visible)
    d_predict_left = loss_direction_controller.decide(
        2,
        SensorFrame(
            width=640,
            height=480,
            persons=[],
            distance_m=2.0,
        ),
    )
    print("lost_wait_last_center:", [(a.kind, a.reason) for a in d_predict_left.actions], d_predict_left.reason)
    if not d_predict_left.waiting_lost_confirm:
        raise AssertionError(f"lost wait should still be in lost-confirm window, got {d_predict_left.reason}")
    if (
        not d_predict_left.actions
        or d_predict_left.actions[0].kind != "rotate_left"
        or d_predict_left.explicit_stop_requested
    ):
        raise AssertionError("last reliable center must retain bounded in-place yaw during confirmation")
    if loss_direction_controller.active_target_id != 1:
        raise AssertionError(f"lost wait should keep the locked target, got {loss_direction_controller.active_target_id}")
    if d_predict_left.reason != "lost_wait_yaw_left":
        raise AssertionError(f"lost wait should explain bounded yaw, got {d_predict_left.reason}")

    for label, bbox, expected_kind, expected_direction in (
        ("last_center_left", (60, 100, 200, 430), "rotate_left", "left"),
        ("last_center_right", (440, 100, 580, 430), "rotate_right", "right"),
    ):
        far_direction_controller = FollowSafetyController(loss_direction_cfg)
        far_direction_controller.decide(
            1,
            SensorFrame(
                width=640,
                height=480,
                persons=[PersonTarget(bbox, track_id=1, confidence=0.9, area=46200)],
                distance_m=2.0,
            ),
        )
        d_far_direction = far_direction_controller.decide(
            2,
            SensorFrame(
                width=640,
                height=480,
                persons=[],
                distance_m=None,
            ),
        )
        print(
            label,
            [(a.kind, a.reason) for a in d_far_direction.actions],
            far_direction_controller._lost_exit_direction,
        )
        if (
            not d_far_direction.actions
            or d_far_direction.actions[0].kind != expected_kind
            or d_far_direction.explicit_stop_requested
        ):
            raise AssertionError(f"{label} should preserve bounded yaw during confirmation, got {d_far_direction}")
        if far_direction_controller._lost_exit_direction != expected_direction:
            raise AssertionError(
                f"{label} should capture {expected_direction} search direction, "
                f"got {far_direction_controller._lost_exit_direction}"
            )

    reacquire_cfg = FollowPolicyConfig(
        lost_confirm_frames=2,
        release_target_on_lost=True,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    reacquire_controller = FollowSafetyController(reacquire_cfg)
    reacquire_controller.decide(1, normal)
    d_wait = reacquire_controller.decide(2, switched)
    if not d_wait.waiting_lost_confirm:
        raise AssertionError(f"first missing locked target frame should wait, got {d_wait.reason}")
    if (
        d_wait.explicit_stop_requested
        or not d_wait.actions
        or d_wait.actions[0].kind != "rotate_right"
        or d_wait.actions[0].speed_percent != 0
    ):
        raise AssertionError("first missing locked target frame should use fixed in-place yaw")
    d_free = reacquire_controller.decide(3, switched)
    print("free_search:", [(a.kind, a.speed_percent, a.reason) for a in d_free.actions], d_free.reason)
    if reacquire_controller.active_target_id != 2:
        raise AssertionError(f"free search should lock the new visible target, got {reacquire_controller.active_target_id}")
    if not d_free.actions:
        raise AssertionError("free search should follow the new visible target in the confirmed-loss decision")

    locked_search_cfg = FollowPolicyConfig(
        lost_confirm_frames=2,
        release_target_on_lost=False,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    locked_search_controller = FollowSafetyController(locked_search_cfg)
    locked_search_controller.decide(1, normal)
    d_locked_wait = locked_search_controller.decide(2, switched)
    if not d_locked_wait.waiting_lost_confirm:
        raise AssertionError(f"locked search should wait before confirmed lost, got {d_locked_wait.reason}")
    d_locked_search = locked_search_controller.decide(3, switched)
    print(
        "locked_search_keeps_target:",
        [(a.kind, a.speed_percent, a.reason) for a in d_locked_search.actions],
        d_locked_search.reason,
        locked_search_controller.active_target_id,
    )
    if locked_search_controller.active_target_id != 1:
        raise AssertionError(
            f"locked search should keep looking for the original target, got {locked_search_controller.active_target_id}"
        )
    if d_locked_search.actions and any(a.kind == "forward" for a in d_locked_search.actions):
        raise AssertionError(f"locked search should not follow the wrong visible person, got {d_locked_search.actions}")
    locked_search_controller.clear_active_target("test_button")
    d_after_clear = locked_search_controller.decide(5, switched)
    if locked_search_controller.active_target_id != 2:
        raise AssertionError(f"manual clear should allow the visible person to become target, got {locked_search_controller.active_target_id}")
    if not d_after_clear.actions:
        raise AssertionError("manual clear should allow normal follow decisions again")

    timed_cfg = FollowPolicyConfig(
        lost_confirm_sec=3.0,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    timed_controller = FollowSafetyController(timed_cfg)
    timed_controller.decide(1, normal)
    d_timed_wait = timed_controller.decide(2, switched)
    if not d_timed_wait.waiting_lost_confirm:
        raise AssertionError(f"time-based lost-confirm should wait, got {d_timed_wait.reason}")
    if (
        d_timed_wait.explicit_stop_requested
        or not d_timed_wait.actions
        or d_timed_wait.actions[0].kind != "rotate_right"
        or d_timed_wait.actions[0].speed_percent != 0
    ):
        raise AssertionError("time-based lost-confirm should immediately use fixed in-place yaw")

    large_unassigned = PersonTarget((0, 0, 260, 430), track_id=-2, confidence=0.9, area=111800)
    mixed = SensorFrame(width=640, height=480, persons=[large_unassigned, person], distance_m=2.0)
    selected = controller.select_target_for_current_state(mixed.persons)
    if selected is None or selected.track_id != 1:
        raise AssertionError(f"locked controller should ignore larger fallback target, got {selected}")
    d_locked = controller.decide(3, mixed)
    print("locked_target:", [(a.kind, a.speed_percent, a.reason) for a in d_locked.actions], d_locked.reason)
    if controller.last_selected_target is None or controller.last_selected_target.track_id != 1:
        raise AssertionError(f"decision should keep the locked target, got {controller.last_selected_target}")
    if not d_locked.actions or any(a.kind != "forward" for a in d_locked.actions):
        raise AssertionError(f"locked target should drive from the matched person, got {d_locked.actions}")

    missing_distance_controller = FollowSafetyController(
        FollowPolicyConfig(distance_missing_forward_percent=35, initial_target_confirm_frames=1)
    )
    missing_distance = SensorFrame(width=640, height=480, persons=[person], distance_m=None)
    d_missing = missing_distance_controller.decide(1, missing_distance)
    if not d_missing.explicit_stop_requested or d_missing.actions or d_missing.reason != "distance_missing_stop":
        raise AssertionError(f"missing distance must stop even with a nonzero legacy fallback, got {d_missing}")

    hold_cfg = FollowPolicyConfig(
        initial_target_confirm_frames=1,
        min_forward_percent=50,
        max_forward_percent=72,
        forward_speed_le_3_2_percent=67,
        mmwave_hold_forward_percent=50,
    )
    hold_controller = FollowSafetyController(hold_cfg)
    hold_distance_state = DistanceState(
        source="vision_mmwave",
        raw_distance_m=None,
        filtered_distance_m=3.0,
        used_distance_m=3.0,
        source_detail="unmatched_hold",
        sample_count=0,
    )
    d_hold_forward = hold_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[person],
            distance_m=3.0,
            distance_state=hold_distance_state,
        ),
    )
    if not d_hold_forward.actions or d_hold_forward.actions[0].speed_percent != 50:
        raise AssertionError(f"mmwave hold forward must be capped to 50 percent: {d_hold_forward}")
    if d_hold_forward.reason != "follow_distance_mmwave_hold":
        raise AssertionError(f"mmwave hold forward must retain a diagnostic reason: {d_hold_forward}")

    hold_right_person = PersonTarget((470, 120, 590, 430), track_id=1, confidence=0.9, area=37200)
    d_hold_steer = hold_controller.decide(
        2,
        SensorFrame(
            width=640,
            height=480,
            persons=[hold_right_person],
            distance_m=3.0,
            distance_state=hold_distance_state,
        ),
    )
    if not d_hold_steer.actions or d_hold_steer.actions[0].kind != "steer_right":
        raise AssertionError(f"off-center target should continue steering during mmwave hold: {d_hold_steer}")
    if d_hold_steer.actions[0].speed_percent != 50 or "mmwave_hold" not in d_hold_steer.reason:
        raise AssertionError(f"mmwave hold steer must be capped and tagged: {d_hold_steer}")

    smooth_hold_cfg = FollowPolicyConfig(
        initial_target_confirm_frames=1,
        min_forward_percent=50,
        max_forward_percent=72,
        forward_speed_le_3_8_percent=72,
        mmwave_hold_forward_percent=50,
        mmwave_hold_decel_step_percent=8,
        mmwave_hold_recover_step_percent=6,
    )
    smooth_hold_controller = FollowSafetyController(smooth_hold_cfg)
    matched_distance_state = DistanceState(
        source="vision_mmwave",
        raw_distance_m=3.6,
        filtered_distance_m=3.6,
        used_distance_m=3.6,
        source_detail="matched",
        sample_count=1,
    )
    normal_fast = smooth_hold_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[person],
            distance_m=3.6,
            distance_state=matched_distance_state,
        ),
    )
    hold_speeds = []
    for frame_index in range(2, 5):
        decision = smooth_hold_controller.decide(
            frame_index,
            SensorFrame(
                width=640,
                height=480,
                persons=[person],
                distance_m=3.6,
                distance_state=hold_distance_state,
            ),
        )
        hold_speeds.append(decision.actions[0].speed_percent)
    recover_speeds = []
    for frame_index in range(5, 8):
        decision = smooth_hold_controller.decide(
            frame_index,
            SensorFrame(
                width=640,
                height=480,
                persons=[person],
                distance_m=3.6,
                distance_state=matched_distance_state,
            ),
        )
        recover_speeds.append(decision.actions[0].speed_percent)
    if normal_fast.actions[0].speed_percent != 72 or hold_speeds != [64, 56, 50]:
        raise AssertionError(
            f"mmwave hold must decelerate gradually: normal={normal_fast} hold={hold_speeds}"
        )
    if recover_speeds != [56, 62, 68]:
        raise AssertionError(f"mmwave recovery must accelerate gradually: {recover_speeds}")

    missing_distance_side_controller = FollowSafetyController(
        FollowPolicyConfig(
            center_left_ratio=0.42,
            center_right_ratio=0.58,
            visible_rotate_left_ratio=0.15,
            visible_rotate_right_ratio=0.85,
            distance_missing_forward_percent=0,
            initial_target_confirm_frames=1,
        )
    )
    missing_distance_side_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[person], distance_m=2.0),
    )
    right_side_person = PersonTarget((470, 120, 590, 430), track_id=1, confidence=0.9, area=37200)
    d_missing_side = missing_distance_side_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[right_side_person], distance_m=None),
    )
    print("missing_distance_side:", [(a.kind, a.reason) for a in d_missing_side.actions], d_missing_side.reason)
    if d_missing_side.explicit_stop_requested or not d_missing_side.actions:
        raise AssertionError(f"visible off-center target should retain low-speed correction, got {d_missing_side}")
    if d_missing_side.actions[0].kind != "steer_right" or not d_missing_side.reason.endswith("_distance_missing"):
        raise AssertionError(f"missing distance inside the 15% edge must use low-speed steer, got {d_missing_side}")

    # 回放实车日志中的 0.808 -> 0.615 -> center -> 0.403 轨迹。毫米波暂时
    # 失配时可以低速 steer 或停车，但人物仍在 15% 边缘内时绝不能原地旋转过中心。
    replay_kinds = []
    for replay_frame, x_ratio in enumerate((0.808, 0.743, 0.615, 0.546, 0.458, 0.403), start=3):
        center_x = int(round(640 * x_ratio))
        replay_person = PersonTarget(
            (center_x - 40, 120, center_x + 40, 430),
            track_id=1,
            confidence=0.9,
            area=24800,
        )
        replay_decision = missing_distance_side_controller.decide(
            replay_frame,
            SensorFrame(width=640, height=480, persons=[replay_person], distance_m=None),
        )
        replay_kinds.extend(action.kind for action in replay_decision.actions)
    print("missing_distance_cross_center_replay:", replay_kinds)
    if any(kind in ("rotate_left", "rotate_right") for kind in replay_kinds):
        raise AssertionError(f"visible target crossing center must not trigger in-place oscillation: {replay_kinds}")
    if "steer_right" not in replay_kinds or "steer_left" not in replay_kinds:
        raise AssertionError(f"cross-center replay should retain low-speed correction on both sides: {replay_kinds}")

    # A target that has just entered the center band must not trigger a
    # counter-steer solely because the encoder still reports residual yaw.
    # The latest camera position wins; lateral output resumes only after the
    # target leaves the band.
    center_hold_controller = FollowSafetyController(
        FollowPolicyConfig(
            visible_steering_pid_enable=True,
            visible_steering_pid_deadband_deg=0.0,
            visible_steering_pid_visual_direction_guard_enabled=False,
            visible_steering_pid_startup_kick_error_deg=0.0,
            visible_steering_pid_startup_kick_rpm=3.0,
            visible_steering_pid_startup_kick_max_sec=0.10,
            initial_target_confirm_frames=1,
            distance_parking_enable=False,
            center_left_ratio=0.45,
            center_right_ratio=0.55,
            steer_release_left_ratio=0.45,
            steer_release_right_ratio=0.55,
        )
    )
    center_hold_target = PersonTarget(
        (292, 120, 392, 430), track_id=23, confidence=0.9, area=31000
    )
    center_hold_action = center_hold_controller._pid_action_for_visible_target(
        center_hold_target,
        SensorFrame(
            width=640,
            height=480,
            persons=[center_hold_target],
            distance_m=2.0,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                yaw_rate_right_dps=-12.0,
                trustworthy=True,
            ),
        ),
        time.monotonic(),
    )
    if (
        center_hold_action is None
        or center_hold_action.kind != "forward"
        or center_hold_action.steer_correction_rpm != 0
        or center_hold_action.reason != "visual_pid_center_hold"
    ):
        raise AssertionError(
            "center-band target must suppress residual-yaw counter-steer: "
            f"{center_hold_action}, result={center_hold_controller.last_steering_pid_result}"
        )
    if center_hold_controller.last_steering_pid_result.startup_kick_active:
        raise AssertionError("initial center hold must not start a kick")
    for tick in (40.0, 40.04, 40.16):
        held = center_hold_controller.refresh_visible_lateral_pid(
            x_ratio=342.0 / 640.0,
            base_rpm=20,
            feedback=SteeringFeedback(timestamp=tick, yaw_rate_right_dps=-12.0, trustworthy=True),
            now=tick,
            target_image_rate_dps=8.0,
        )
        if held.correction_rpm != 0 or held.startup_kick_active:
            raise AssertionError(f"center refresh must retain zero despite startup/image motion: {held}")
    held_outside = center_hold_controller.refresh_visible_lateral_pid(
        x_ratio=0.70, base_rpm=20, feedback=None, now=40.20, hold_zero=True,
    )
    if held_outside.correction_rpm != 0 or held_outside.startup_kick_active:
        raise AssertionError(f"explicit visible zero intent must survive refresh: {held_outside}")

    no_distance_history_controller = FollowSafetyController(
        FollowPolicyConfig(
            center_left_ratio=0.42,
            center_right_ratio=0.58,
            visible_rotate_left_ratio=0.15,
            visible_rotate_right_ratio=0.85,
            distance_missing_forward_percent=0,
        )
    )
    d_no_distance_history = no_distance_history_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[right_side_person], distance_m=None),
    )
    if d_no_distance_history.actions or not d_no_distance_history.explicit_stop_requested:
        raise AssertionError(f"steer without any trusted distance history must remain stopped, got {d_no_distance_history}")

    # 目标贴边时仍使用差速 steer；目标真正丢失后先停车确认，再按离开方向搜索。
    for side, bbox, visible_kind in (
        ("left", (5, 120, 45, 430), "steer_left"),
        ("right", (955, 120, 995, 430), "steer_right"),
    ):
        direction_controller = FollowSafetyController(
            FollowPolicyConfig(
                center_left_ratio=0.30,
                center_right_ratio=0.70,
                visible_rotate_left_ratio=0.25,
                visible_rotate_right_ratio=0.75,
                initial_target_confirm_frames=1,
                lost_confirm_sec=3.0,
            )
        )
        d_visible_edge = direction_controller.decide(
            1,
            SensorFrame(
                width=1000,
                height=480,
                persons=[PersonTarget(bbox, track_id=1, confidence=0.9, area=12400)],
                distance_m=2.0,
            ),
        )
        d_just_lost = direction_controller.decide(
            2,
            SensorFrame(
                width=1000,
                height=480,
                persons=[],
                distance_m=2.0,
            ),
        )
        visible_actions = [a.kind for a in d_visible_edge.actions]
        lost_actions = [a.kind for a in d_just_lost.actions]
        print("edge_to_lost_direction:", side, visible_actions, lost_actions)
        if (
            visible_actions != [visible_kind]
            or lost_actions != ["rotate_" + side]
            or d_just_lost.explicit_stop_requested
        ):
            raise AssertionError(
                f"{side} edge loss must retain bounded same-direction yaw: "
                f"visible={d_visible_edge}, lost={d_just_lost}"
            )

    five_frame_controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=5,
            lost_confirm_sec=0.0,
            steer_min_hold_sec=0.0,
            steer_lost_hold_frames=4,
            steer_lost_hold_max_sec=1.0,
            release_target_on_lost=False,
        )
    )
    d_five_visible = five_frame_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[right_side_person], distance_m=2.0),
    )
    if not d_five_visible.actions or d_five_visible.actions[0].kind != "steer_right":
        raise AssertionError(f"five-frame test must start from steer_right, got {d_five_visible}")
    five_frame_controller.set_last_dispatched("steer_right")
    for frame_index in range(2, 6):
        held = five_frame_controller.decide(
            frame_index,
            SensorFrame(width=640, height=480, persons=[], distance_m=2.0),
        )
        if not held.actions or held.actions[0].kind != "rotate_right" or held.explicit_stop_requested:
            raise AssertionError(f"missing frame {frame_index - 1}/4 must retain in-place right yaw, got {held}")
        five_frame_controller.set_last_dispatched("rotate_right")
    d_fifth_missing = five_frame_controller.decide(
        6,
        SensorFrame(width=640, height=480, persons=[], distance_m=2.0),
    )
    if (
        not d_fifth_missing.actions
        or d_fifth_missing.actions[0].kind != "rotate_right"
        or d_fifth_missing.explicit_stop_requested
    ):
        raise AssertionError(f"fifth missing frame must enter search without redundant stop, got {d_fifth_missing}")

    search_flow_controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=2,
            lost_confirm_sec=0.0,
            search_timeout_sec=4.0,
            release_target_on_lost=False,
        )
    )
    search_flow_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[right_side_person], distance_m=2.0),
    )
    d_presearch = search_flow_controller.decide(2, SensorFrame(width=640, height=480, persons=[]))
    if not d_presearch.actions or d_presearch.actions[0].kind != "rotate_right" or d_presearch.explicit_stop_requested:
        raise AssertionError(f"right-side loss should retain bounded yaw during confirmation, got {d_presearch}")
    if search_flow_controller._search_rotation_started_at is not None:
        raise AssertionError("search timeout must remain stopped during lost confirmation")
    d_confirmed_search = search_flow_controller.decide(3, SensorFrame(width=640, height=480, persons=[]))
    print(
        "confirmed_search:",
        [(a.kind, a.reason) for a in d_confirmed_search.actions],
        search_flow_controller.search_state,
        search_flow_controller.search_direction,
    )
    if not d_confirmed_search.actions or d_confirmed_search.actions[0].kind != "rotate_right":
        raise AssertionError(f"confirmed right search should use physical rotate_right, got {d_confirmed_search}")
    if search_flow_controller._search_rotation_started_at is None:
        raise AssertionError("confirmed search must start its independent timeout on the first search rotation")
    d_search_recovered = search_flow_controller.decide(
        5,
        SensorFrame(width=640, height=480, persons=[right_side_person], distance_m=2.0),
    )
    if search_flow_controller.search_state != "none" or search_flow_controller._search_rotation_started_at is not None:
        raise AssertionError("same ReID recovery must reset search state and timeout")
    if not d_search_recovered.clear_action_queue or not d_search_recovered.stop_action_execution:
        raise AssertionError(f"same ReID recovery must interrupt queued search rotation, got {d_search_recovered}")

    # A fast edge exit must search instead of terminating immediately. With
    # encoder yaw available, termination is tied to one complete revolution.
    revolution_controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=1,
            search_timeout_sec=0.05,
            search_timeout_exit_program=True,
            search_revolution_deg=360.0,
            search_revolution_feedback_stale_sec=0.80,
            release_target_on_lost=False,
        )
    )
    revolution_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[right_side_person],
            distance_m=2.0,
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                integrated_yaw_right_deg=0.0,
                trustworthy=True,
            ),
        ),
    )
    revolution_controller.decide(
        2,
        SensorFrame(
            width=640,
            height=480,
            persons=[],
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                integrated_yaw_right_deg=0.0,
                trustworthy=True,
            ),
        ),
    )
    revolution_controller.decide(
        3,
        SensorFrame(
            width=640,
            height=480,
            persons=[],
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                integrated_yaw_right_deg=0.0,
                trustworthy=True,
            ),
        ),
    )
    for frame_index, yaw_deg in enumerate((90.0, 180.0, 270.0), start=4):
        d_scan = revolution_controller.decide(
            frame_index,
            SensorFrame(
                width=640,
                height=480,
                persons=[],
                steering_feedback=SteeringFeedback(
                    timestamp=time.monotonic(),
                    integrated_yaw_right_deg=yaw_deg,
                    trustworthy=True,
                ),
            ),
        )
        if d_scan.shutdown_requested:
            raise AssertionError(f"search must not exit before 360 degrees: {d_scan}")
    d_scan_complete = revolution_controller.decide(
        7,
        SensorFrame(
            width=640,
            height=480,
            persons=[],
            steering_feedback=SteeringFeedback(
                timestamp=time.monotonic(),
                integrated_yaw_right_deg=360.0,
                trustworthy=True,
            ),
        ),
    )
    if not d_scan_complete.shutdown_requested or d_scan_complete.reason != "search_revolution_complete":
        raise AssertionError(f"search must exit only after one encoder revolution: {d_scan_complete}")

    # Lost-target search is always in-place. Distance cannot reintroduce
    # longitudinal motion after the visual target is gone.
    distance_search_cfg = FollowPolicyConfig(
        lost_confirm_frames=2,
        release_target_on_lost=False,
        lost_forward_hold_max_sec=0.0,
        max_forward_percent=20,
        forward_speed_far_percent=20,
        visible_steering_pid_enable=False,
    )
    far_search_controller = FollowSafetyController(distance_search_cfg)
    far_search_controller.decide(1, SensorFrame(width=640, height=480, persons=[right_side_person], distance_m=5.0))
    far_search_controller.set_last_dispatched("steer_right")
    far_search_controller.decide(2, SensorFrame(width=640, height=480, persons=[], distance_m=5.0))
    far_search_controller.decide(3, SensorFrame(width=640, height=480, persons=[], distance_m=5.0))
    far_search = far_search_controller.decide(4, SensorFrame(width=640, height=480, persons=[], distance_m=5.0))
    if (
        not far_search.actions
        or far_search.actions[0].kind != "rotate_right"
        or far_search.actions[0].speed_percent != 0
    ):
        raise AssertionError(f"far target loss must use in-place search: {far_search}")

    near_search_cfg = replace(
        distance_search_cfg,
        near_distance_rotate_only_enable=True,
        distance_parking_enable=False,
    )
    near_search_controller = FollowSafetyController(near_search_cfg)
    near_search_controller.decide(1, SensorFrame(width=640, height=480, persons=[right_side_person], distance_m=3.0))
    near_search_controller.set_last_dispatched("rotate_right")
    near_search_controller.decide(2, SensorFrame(width=640, height=480, persons=[], distance_m=3.0))
    near_search_controller.decide(3, SensorFrame(width=640, height=480, persons=[], distance_m=3.0))
    near_search = near_search_controller.decide(4, SensorFrame(width=640, height=480, persons=[], distance_m=3.0))
    if (
        not near_search.actions
        or near_search.actions[0].kind != "rotate_right"
        or near_search.actions[0].speed_percent != 0
    ):
        raise AssertionError(f"near target loss must use in-place search: {near_search}")

    search_flow_controller.search_state = "timed_out"
    search_flow_controller.search_direction = None
    search_flow_controller._search_rotation_started_at = time.monotonic() - 5.0
    d_timeout_reid = search_flow_controller.decide(
        6,
        SensorFrame(width=640, height=480, persons=[right_side_person], distance_m=2.0),
    )
    if search_flow_controller.search_state != "none" or not d_timeout_reid.actions:
        raise AssertionError(f"confirmed original ReID must recover even after search timeout, got {d_timeout_reid}")
    if not d_timeout_reid.clear_action_queue or not d_timeout_reid.stop_action_execution:
        raise AssertionError(f"timed-out ReID recovery must clear the stale stop/search state, got {d_timeout_reid}")

    timeout_controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=1,
            search_timeout_sec=4.0,
            release_target_on_lost=False,
        )
    )
    timeout_controller.decide(1, normal)
    timeout_controller.search_state = "searching"
    timeout_controller.lost_confirm_frames = 1
    timeout_controller._lost_started_at = time.monotonic() - 4.1
    timeout_controller._search_rotation_started_at = time.monotonic() - 4.1
    d_timeout = timeout_controller.decide(2, SensorFrame(width=640, height=480, persons=[]))
    if not d_timeout.explicit_stop_requested or d_timeout.reason != "search_timeout_stop":
        raise AssertionError(f"search timeout must stop and clear rotation, got {d_timeout}")
    if not d_timeout.clear_action_queue or not d_timeout.stop_action_execution:
        raise AssertionError(f"first search timeout must clear queued/current motion, got {d_timeout}")

    deferred_timeout_controller = FollowSafetyController(
        FollowPolicyConfig(search_timeout_sec=5.0, search_timeout_exit_program=True)
    )
    timeout_now = time.monotonic()
    deferred_timeout_controller.search_state = "searching"
    deferred_timeout_controller._lost_started_at = timeout_now - 5.05
    deferred_timeout_controller._search_rotation_started_at = timeout_now - 4.0
    deferred_timeout_controller.defer_search_timeout(0.20)
    if deferred_timeout_controller._search_timeout_decision(timeout_now) is not None:
        raise AssertionError("two-frame ReID observation must pause the five-second search timeout")
    deferred_exit = deferred_timeout_controller._search_timeout_decision(timeout_now + 0.20)
    if deferred_exit is None or deferred_exit.reason != "search_timeout_exit":
        raise AssertionError(f"search timeout must resume after observation grace: {deferred_exit}")

    exit_timeout_controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=1,
            search_timeout_sec=5.0,
            search_timeout_exit_program=True,
            release_target_on_lost=False,
        )
    )
    exit_timeout_controller.decide(1, normal)
    exit_timeout_controller.decide(2, SensorFrame(width=640, height=480, persons=[]))
    # 旋转刚开始，但目标从画面消失已经超过 5 秒：应按总丢失时间退出，
    # 不能再从第一条搜索旋转命令重新计算 5 秒。
    exit_timeout_controller.search_state = "searching"
    exit_timeout_controller.lost_confirm_frames = 2
    exit_timeout_controller._lost_started_at = time.monotonic() - 5.1
    exit_timeout_controller._search_rotation_started_at = time.monotonic()
    d_timeout_exit = exit_timeout_controller.decide(
        3,
        SensorFrame(width=640, height=480, persons=[]),
    )
    if (
        not d_timeout_exit.explicit_stop_requested
        or not d_timeout_exit.shutdown_requested
        or d_timeout_exit.reason != "search_timeout_exit"
    ):
        raise AssertionError(f"five-second total target loss must request safe program exit, got {d_timeout_exit}")
    if not d_timeout_exit.clear_action_queue or not d_timeout_exit.stop_action_execution:
        raise AssertionError(f"program exit must cancel queued/current search motion, got {d_timeout_exit}")

    # Depth parking is disabled on the board: a one-frame detector gap keeps
    # the last yaw side but immediately removes the forward component.
    short_gap_controller = FollowSafetyController(
        FollowPolicyConfig(
            visible_steering_pid_enable=True,
            distance_parking_enable=False,
            steer_min_hold_sec=0.0,
            steer_lost_hold_frames=2,
            steer_lost_hold_max_sec=1.0,
            visible_steering_pid_fallback_base_rpm=15,
            forward_max_rpm=50,
        )
    )
    close_right = PersonTarget((400, 100, 520, 430), track_id=21, confidence=0.9, area=39600)
    d_close_right = short_gap_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[close_right], distance_m=2.0),
    )
    if not d_close_right.actions or d_close_right.actions[0].kind != "steer_right":
        raise AssertionError(f"test setup must start steer_right: {d_close_right}")
    short_gap_controller.set_last_dispatched("steer_right")
    d_short_gap = short_gap_controller.decide(
        2,
        SensorFrame(width=640, height=480, persons=[], distance_m=2.0),
    )
    if (
        not d_short_gap.actions
        or d_short_gap.actions[0].kind != "rotate_right"
        or d_short_gap.explicit_stop_requested
    ):
        raise AssertionError(f"one-frame gap must keep only in-place yaw without STOP: {d_short_gap}")

    # Delayed visual trend may request left while the fresh target is still on
    # the right. The controller must coast instead of powering away from it.
    trend_controller = FollowSafetyController(
        FollowPolicyConfig(
            visible_steering_pid_enable=True,
            visible_steering_pid_target_rate_feedforward_gain=1.0,
            visible_steering_pid_target_rate_feedforward_max_dps=30.0,
            distance_parking_enable=False,
            forward_max_rpm=50,
        )
    )
    trend_target = PersonTarget((272.8, 100, 392.8, 430), track_id=22, confidence=0.9, area=39600)
    trend_frame = SensorFrame(
        width=640,
        height=480,
        persons=[trend_target],
        distance_m=3.0,
    )
    trend_action = trend_controller._pid_action_for_visible_target(
        trend_target,
        trend_frame,
        time.monotonic(),
        motion_dx_ratio=-0.08,
        target_image_rate_dps=-30.0,
    )
    trend_result = trend_controller.last_steering_pid_result
    if (
        (trend_action is not None and trend_action.kind != "forward")
        or trend_result is None
        or trend_result.correction_rpm != 0
    ):
        raise AssertionError(
            "fresh right-side position must suppress stale left trend without "
            f"powering a reversal: {trend_action}, {trend_result}"
        )

    # The independent 30Hz Depth path only changes longitudinal speed. Once
    # reverse starts, missing Depth may hold/zero it but must never unlock it
    # into forward motion. A nearly full-frame person can start guarded reverse
    # even while the center Depth sample is temporarily unavailable.
    depth_only_cfg = FollowPolicyConfig(
        initial_target_confirm_frames=1,
        reverse_enable=True,
        distance_parking_enable=False,
        reverse_start_distance_m=1.30,
        reverse_immediate_distance_m=1.30,
        reverse_stop_distance_m=1.45,
        reverse_distance_missing_hold_sec=0.35,
        reverse_visual_guard_area_ratio=0.45,
        reverse_visual_guard_height_ratio=0.92,
        reverse_visual_guard_rpm=30,
        distance_pid_enable=False,
        forward_max_rpm=100,
    )
    depth_only = FollowSafetyController(depth_only_cfg)
    depth_person = PersonTarget((240, 80, 400, 460), track_id=31, confidence=0.9, area=60800)
    depth_only.active_target_id = 31
    depth_only._has_seen_person = True

    def depth_only_frame(distance, detail="depth_matched", *, person=depth_person, front=False):
        return SensorFrame(
            width=640,
            height=480,
            persons=[person],
            obstacles=ObstacleState(front=front),
            distance_m=distance,
            distance_state=DistanceState(
                source="vision_depth",
                raw_distance_m=distance,
                used_distance_m=distance,
                source_detail=detail,
                sample_age_sec=0.01,
            ),
        )

    forward_depth_only = FollowSafetyController(depth_only_cfg)
    forward_depth_only.active_target_id = 31
    forward_depth_only._has_seen_person = True
    forward_depth_only.decide(90, depth_only_frame(2.20), longitudinal_only=True)
    missing_depth_noop = forward_depth_only.decide(
        91,
        depth_only_frame(None, "insufficient_depth_pixels"),
        longitudinal_only=True,
    )
    if missing_depth_noop.actions or missing_depth_noop.reason != "longitudinal_distance_missing_keep":
        raise AssertionError(
            "short missing Depth must keep the current visual action: "
            f"{missing_depth_noop}"
        )

    # A fused hold with an encoder-predicted distance may continue briefly at
    # the conservative depth hold speed, without resetting the PID immediately.
    hold_controller = FollowSafetyController(depth_only_cfg)
    hold_controller.active_target_id = 31
    hold_controller._has_seen_person = True
    hold_controller.decide(90, depth_only_frame(2.20), longitudinal_only=True)
    held_frame = SensorFrame(
        width=640,
        height=480,
        persons=[depth_person],
        obstacles=ObstacleState(),
        distance_m=2.18,
        distance_state=DistanceState(
            source="vision_depth",
            used_distance_m=2.18,
            source_detail="insufficient_depth_pixels_fused_visual_encoder_hold",
            sample_age_sec=0.05,
            fusion_confidence=0.46,
        ),
    )
    short_hold = hold_controller.decide(91, held_frame, longitudinal_only=True)
    if (
        not short_hold.actions
        or short_hold.actions[0].kind != "forward"
        or short_hold.actions[0].speed_percent > 20
        or short_hold.reason != "longitudinal_distance_short_hold"
    ):
        raise AssertionError(f"short fused Depth hold should keep a capped forward action: {short_hold}")

    # A rate-guarded jump must not feed its fused distance into the normal PID,
    # but it may use the last trusted range for the same short capped hold.
    jump_hold_frame = replace(
        held_frame,
        distance_m=3.02,
        distance_state=replace(
            held_frame.distance_state,
            used_distance_m=3.02,
            source_detail="distance_jump_rate_guard_hold_fused_visual_encoder_hold",
            fusion_confidence=0.53,
        ),
    )
    jump_hold = hold_controller.decide(92, jump_hold_frame, longitudinal_only=True)
    if (
        not jump_hold.actions
        or jump_hold.actions[0].kind != "forward"
        or jump_hold.actions[0].speed_percent > 20
        or jump_hold.reason != "longitudinal_distance_short_hold"
    ):
        raise AssertionError(
            "rate-guarded fused distance must use only capped last-range hold: "
            f"{jump_hold}"
        )
    if abs(float(hold_controller._last_target_distance_m) - 2.20) > 1e-6:
        raise AssertionError(
            "held/fused depth must not replace the last trusted distance anchor: "
            f"{hold_controller._last_target_distance_m}"
        )

    forward_depth_only._last_target_distance_at = time.monotonic() - 1.0
    missing_depth_stop = forward_depth_only.decide(
        92,
        depth_only_frame(None, "insufficient_depth_pixels"),
        longitudinal_only=True,
    )
    if (
        not missing_depth_stop.actions
        or missing_depth_stop.actions[0].kind != "forward"
        or missing_depth_stop.actions[0].speed_percent != 0
        or missing_depth_stop.reason != "longitudinal_distance_low_confidence_stop"
    ):
        raise AssertionError(
            "Depth missing beyond 600ms must stop longitudinal motion: "
            f"{missing_depth_stop}"
        )
    recovery_stage1 = forward_depth_only.decide(
        93,
        depth_only_frame(3.0, "depth_multiregion"),
        longitudinal_only=True,
    )
    if not recovery_stage1.actions or recovery_stage1.actions[0].speed_percent > 25:
        raise AssertionError(f"first recovered Depth command must be capped at 25 RPM: {recovery_stage1}")
    forward_depth_only._depth_recovery_started_at = time.monotonic() - 0.25
    recovery_stage2 = forward_depth_only.decide(
        94,
        depth_only_frame(3.0, "depth_multiregion"),
        longitudinal_only=True,
    )
    if not recovery_stage2.actions or recovery_stage2.actions[0].speed_percent > 45:
        raise AssertionError(f"second recovery stage must be capped at 45 RPM: {recovery_stage2}")

    # A post-timeout Depth re-anchor can lock onto the far background. It must
    # never turn a near target's last good distance into a full-speed forward
    # command while the camera loop is still deciding what happened.
    reanchor = forward_depth_only.decide(
        95,
        depth_only_frame(7.82, "depth_reanchored_after_timeout"),
        longitudinal_only=True,
    )
    if (
        reanchor.reason != "longitudinal_distance_untrusted_hold"
        or not reanchor.actions
        or reanchor.actions[0].kind != "forward"
        or reanchor.actions[0].speed_percent != 0
    ):
        raise AssertionError(
            f"untrusted Depth re-anchor must zero longitudinal motion only: {reanchor}"
        )

    reanchor_followup = forward_depth_only.decide(
        95,
        depth_only_frame(7.82, "depth_multiregion"),
        longitudinal_only=True,
    )
    if (
        reanchor_followup.reason != "longitudinal_distance_untrusted_hold"
        or reanchor_followup.actions[0].kind != "forward"
        or reanchor_followup.actions[0].speed_percent != 0
    ):
        raise AssertionError(f"far value after re-anchor must remain blocked: {reanchor_followup}")

    jump_pending = forward_depth_only.decide(
        96,
        depth_only_frame(7.70, "depth_multiregion_distance_jump_pending_1_of_3"),
        longitudinal_only=True,
    )
    if (
        jump_pending.reason != "longitudinal_distance_untrusted_hold"
        or jump_pending.actions[0].kind != "forward"
        or jump_pending.actions[0].speed_percent != 0
    ):
        raise AssertionError(f"jump-pending Depth must not start forward PID: {jump_pending}")

    reverse_started = depth_only.decide(
        100,
        depth_only_frame(1.20),
        longitudinal_only=True,
    )
    if not reverse_started.actions or reverse_started.actions[0].kind != "backward":
        raise AssertionError(f"fresh near Depth must start longitudinal reverse: {reverse_started}")
    depth_only.last_action_frame = 77
    reverse_missing_hold = depth_only.decide(
        101,
        depth_only_frame(None, "no_depth_frame"),
        longitudinal_only=True,
    )
    if (
        not reverse_missing_hold.actions
        or reverse_missing_hold.actions[0].kind != "backward"
        or reverse_missing_hold.actions[0].speed_percent <= 0
        or depth_only.last_action_frame != 77
    ):
        raise AssertionError(f"short Depth gap must retain reverse without visual state changes: {reverse_missing_hold}")
    depth_only._reverse_missing_started_at = time.monotonic() - 1.0
    reverse_missing_zero = depth_only.decide(
        102,
        depth_only_frame(None, "no_depth_frame"),
        longitudinal_only=True,
    )
    if (
        not reverse_missing_zero.actions
        or reverse_missing_zero.actions[0].kind != "backward"
        or reverse_missing_zero.actions[0].speed_percent != 0
        or not depth_only._reverse_active
    ):
        raise AssertionError(f"long Depth gap must zero but preserve reverse latch: {reverse_missing_zero}")

    far_guard = FollowSafetyController(replace(depth_only_cfg, reverse_confirm_frames=2))
    far_guard.active_target_id = 33
    far_guard._has_seen_person = True
    far_large_person = PersonTarget((80, 0, 560, 478), track_id=33, confidence=0.9, area=229440)
    far_guard.decide(
        108,
        depth_only_frame(2.79, person=far_large_person),
        longitudinal_only=True,
    )
    for frame_index in (109, 110):
        far_hold = far_guard.decide(
            frame_index,
            depth_only_frame(
                None,
                "depth_matched_reused_hold",
                person=far_large_person,
            ),
            longitudinal_only=True,
        )
        if far_hold.actions and far_hold.actions[0].kind == "backward":
            raise AssertionError(f"recent 2.79m Depth must veto visual reverse: {far_hold}")

    target_band_guard = FollowSafetyController(replace(depth_only_cfg, reverse_confirm_frames=2))
    target_band_guard.active_target_id = 36
    target_band_guard._has_seen_person = True
    target_band_person = PersonTarget((80, 0, 560, 478), track_id=36, confidence=0.9, area=229440)
    target_band_guard.decide(
        114,
        depth_only_frame(1.54, person=target_band_person),
        longitudinal_only=True,
    )
    for frame_index in (115, 116):
        target_band_hold = target_band_guard.decide(
            frame_index,
            depth_only_frame(
                None,
                "depth_multiregion_reused_hold",
                person=target_band_person,
            ),
            longitudinal_only=True,
        )
        if target_band_hold.actions and target_band_hold.actions[0].kind == "backward":
            raise AssertionError(
                "recent 1.54m target-band Depth must veto visual reverse: "
                f"{target_band_hold}"
            )

    same_frame_guard = FollowSafetyController(replace(depth_only_cfg, reverse_confirm_frames=2))
    same_frame_person = PersonTarget((80, 0, 560, 478), track_id=35, confidence=0.9, area=229440)
    same_frame_state = depth_only_frame(None, "no_depth_frame", person=same_frame_person)
    first_guard, _ = same_frame_guard._visual_reverse_guard(
        120,
        same_frame_state,
        same_frame_person,
        time.monotonic(),
    )
    cached_guard, _ = same_frame_guard._visual_reverse_guard(
        121,
        same_frame_state,
        same_frame_person,
        time.monotonic(),
    )
    fresh_far_guard, _ = same_frame_guard._visual_reverse_guard(
        121,
        same_frame_state,
        same_frame_person,
        time.monotonic(),
        fresh_distance_m=2.06,
    )
    held_after_fresh_guard, _ = same_frame_guard._visual_reverse_guard(
        121,
        same_frame_state,
        same_frame_person,
        time.monotonic(),
    )
    if first_guard or not cached_guard or fresh_far_guard or held_after_fresh_guard:
        raise AssertionError(
            "fresh 2.06m Depth must invalidate a same-frame visual reverse cache: "
            f"{first_guard=}, {cached_guard=}, {fresh_far_guard=}, {held_after_fresh_guard=}"
        )

    tall_guard = FollowSafetyController(replace(depth_only_cfg, reverse_confirm_frames=2))
    tall_guard.active_target_id = 34
    tall_guard._has_seen_person = True
    tall_person = PersonTarget((173, 18, 349, 479), track_id=34, confidence=0.9, area=81136)
    for frame_index in (111, 112, 113):
        tall_hold = tall_guard.decide(
            frame_index,
            depth_only_frame(None, "no_depth_frame", person=tall_person),
            longitudinal_only=True,
        )
        if tall_hold.actions and tall_hold.actions[0].kind == "backward":
            raise AssertionError(f"full height without sufficient area must not reverse: {tall_hold}")

    visual_guard = FollowSafetyController(replace(depth_only_cfg, reverse_confirm_frames=2))
    visual_guard.active_target_id = 32
    visual_guard._has_seen_person = True
    huge_person = PersonTarget((60, 0, 580, 478), track_id=32, confidence=0.9, area=248560)
    visual_wait = visual_guard.decide(
        110,
        depth_only_frame(None, "center_patch_and_foreground_insufficient", person=huge_person),
        target_steerable=False,
        longitudinal_only=True,
    )
    if visual_wait.actions and visual_wait.actions[0].kind == "backward":
        raise AssertionError(f"one visual near sample must not reverse: {visual_wait}")
    visual_reverse = visual_guard.decide(
        111,
        depth_only_frame(None, "center_patch_and_foreground_insufficient", person=huge_person),
        target_steerable=False,
        longitudinal_only=True,
    )
    if (
        not visual_reverse.actions
        or visual_reverse.actions[0].kind != "backward"
        or visual_reverse.actions[0].speed_percent < 30
        or visual_reverse.reason != "visual_near_guard_reverse"
    ):
        raise AssertionError(f"full-frame target must trigger visual guarded reverse: {visual_reverse}")
    ir_stop = visual_guard.decide(
        112,
        depth_only_frame(None, "no_depth_frame", person=huge_person, front=True),
        longitudinal_only=True,
    )
    if not ir_stop.explicit_stop_requested or ir_stop.reason != "front_ir":
        raise AssertionError(f"IR must remain above visual/Depth reverse: {ir_stop}")

    candidate_controller = FollowSafetyController(
        FollowPolicyConfig(
            initial_target_confirm_frames=2,
            visible_steering_pid_enable=True,
            visible_steering_pid_camera_hfov_deg=60.0,
            visible_steering_pid_deadband_deg=6.0,
            visible_steering_pid_max_correction_rpm=40.0,
            visible_steering_pid_dynamic_small_max_correction_rpm=20.0,
            visible_steering_pid_startup_kick_error_deg=6.0,
            visible_steering_pid_startup_kick_rpm=28.0,
            visible_steering_pid_startup_kick_max_sec=0.35,
            visible_steering_pid_startup_kick_release_yaw_rate_dps=4.0,
            visible_steering_pid_fallback_max_correction_rpm=40.0,
            forward_max_rpm=40,
        )
    )
    candidate = PersonTarget(
        (500, 40, 630, 450),
        track_id=999,
        confidence=0.9,
        area=53300,
    )
    candidate_frame = SensorFrame(
        width=640,
        height=480,
        persons=[candidate],
        steering_feedback=SteeringFeedback(
            timestamp=time.monotonic(),
            yaw_rate_right_dps=0.0,
            trustworthy=True,
        ),
    )
    candidate_decision = candidate_controller.decide(200, candidate_frame)
    if (
        not candidate_decision.actions
        or candidate_decision.actions[0].kind != "rotate_right"
        or candidate_decision.explicit_stop_requested
        or candidate_decision.is_forwarding
        or candidate_decision.reason != "initial_candidate_centering_right"
    ):
        raise AssertionError(
            f"unconfirmed off-center candidate must yaw-center without forward motion: {candidate_decision}"
        )
    if not candidate_decision.clear_action_queue:
        raise AssertionError("the first candidate frame must clear stale queued motion")
    candidate_continuation = candidate_controller.decide(201, candidate_frame)
    if (
        not candidate_continuation.actions
        or candidate_continuation.actions[0].kind != "steer_right"
        or candidate_continuation.explicit_stop_requested
    ):
        raise AssertionError(
            "a candidate may publish steering only after initial confirmation: "
            f"{candidate_continuation}"
        )
    if (
        candidate_controller.active_target_id != 999
        or candidate_controller.last_selected_target is None
    ):
        raise AssertionError(
            "confirmed candidate must become the active published target"
        )

    hazard = SensorFrame(
        width=640,
        height=480,
        persons=[person],
        distance_m=2.0,
        hazard=HazardState(active=True, reason="bunker"),
    )
    d2 = controller.decide(4, hazard)
    print("hazard:", [(a.kind, a.speed_percent, a.reason) for a in d2.actions], d2.explicit_stop_requested, d2.reason)
    if not d2.explicit_stop_requested:
        raise AssertionError("hazard should request an explicit stop")
    if any(a.kind != "stop" for a in d2.actions):
        raise AssertionError(f"hazard should only emit stop actions: {d2.actions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
