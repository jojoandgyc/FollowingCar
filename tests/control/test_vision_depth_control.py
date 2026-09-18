#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.control_types import DistanceState, PersonTarget, SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.distance_runtime import DistanceRuntime, DistanceRuntimeConfig


class FakeAstraSensors:
    def __init__(self):
        self.calls = 0
        self.reference_timestamp = None

    def get_astra_target_distance(
        self,
        bbox,
        width,
        height,
        *,
        target_id=None,
        reference_timestamp=None,
    ):
        self.calls += 1
        self.reference_timestamp = reference_timestamp
        assert bbox == (100.0, 40.0, 300.0, 440.0)
        assert (width, height, target_id) == (640, 480, 9)
        return SimpleNamespace(
            distance_m=1.72,
            raw_distance_m=1.70,
            sample_age_sec=0.03,
            valid_pixels=420,
            detail="depth_matched",
        )


def main() -> int:
    config = DistanceRuntimeConfig(
        distance_source="vision_depth",
        vision_mmwave_source_aliases=frozenset({"vision_mmwave"}),
        module_mmwave_enable=False,
        module_ultrasonic_enable=False,
        vision_hfov_deg=60.0,
        vision_mmwave_angle_margin_deg=8.0,
        vision_mmwave_angle_offset_deg=0.0,
        vision_mmwave_angle_sign=1.0,
        vision_mmwave_match_mode="angle",
        vision_mmwave_distance_bias_m=0.0,
        vision_mmwave_min_distance_m=0.5,
        vision_mmwave_max_distance_m=10.0,
        vision_mmwave_min_output_distance_m=0.03,
        vision_mmwave_hard_stop_ttl_sec=0.3,
        vision_mmwave_log_every_frames=0,
        module_astra_depth_enable=True,
        vision_depth_source_aliases=frozenset({"vision_depth"}),
    )
    sensors = FakeAstraSensors()
    runtime = DistanceRuntime(
        owner=object(),
        config=config,
        sensor_runtime=sensors,
    )
    target = PersonTarget((100.0, 40.0, 300.0, 440.0), 9, 0.9, 80000.0)
    state = runtime.get_frame_distance_state(
        640,
        target,
        frame_height=480,
        target_distance_m=1.5,
        brake_distance_m=0.8,
        capture_timestamp=123.456,
    )
    if state.source != "vision_depth" or state.source_detail != "depth_matched":
        raise AssertionError(f"Depth source was not selected: {state}")
    if state.used_distance_m != 1.72 or state.raw_distance_m != 1.70:
        raise AssertionError(f"Depth distance fields were not propagated: {state}")
    if sensors.reference_timestamp != 123.456:
        raise AssertionError(
            "visual RGB capture timestamp must reach the Astra measurement: "
            f"{sensors.reference_timestamp}"
        )
    recent = runtime.get_recent_vision_depth_state(
        target_distance_m=1.5,
        brake_distance_m=0.8,
    )
    if recent != state or sensors.calls != 1:
        raise AssertionError(
            "action-thread safety reads must reuse the aligned visual-frame Depth cache "
            f"without a stale-bbox sensor read: state={recent} calls={sensors.calls}"
        )

    # Losing Depth must reduce forward speed without disabling camera steering.
    # A near person at the image edge needs enough wheel difference to stay in view.
    fallback_controller = FollowSafetyController(
        FollowPolicyConfig(
            initial_target_confirm_frames=1,
            center_left_ratio=0.45,
            center_right_ratio=0.55,
            visible_steering_pid_enable=True,
            visible_steering_pid_camera_hfov_deg=60.0,
            visible_steering_pid_fallback_base_rpm=15,
            visible_steering_pid_fallback_max_correction_rpm=10.0,
        )
    )
    edge_person = PersonTarget((540.0, 20.0, 612.0, 470.0), 11, 0.9, 32400.0)
    fallback_turn = fallback_controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[edge_person],
            distance_m=None,
            distance_state=DistanceState(
                source="vision_depth",
                source_detail="far_background_guard_large_bbox",
            ),
        ),
    )
    if not fallback_turn.actions or fallback_turn.actions[0].kind != "steer_right":
        raise AssertionError(f"Depth loss must retain visible right steering: {fallback_turn}")
    if fallback_turn.actions[0].steer_correction_rpm < 8:
        raise AssertionError(f"large visual error needs strong fallback wheel difference: {fallback_turn}")
    if fallback_turn.actions[0].speed_percent != 0:
        raise AssertionError(
            "Depth with no fresh anchor must stop longitudinal motion while preserving yaw: "
            f"{fallback_turn}"
        )

    controller = FollowSafetyController(
        FollowPolicyConfig(reverse_enable=True, reverse_radar_max_age_sec=0.25)
    )
    frame = SensorFrame(distance_m=1.20, distance_state=DistanceState(
        source="vision_depth",
        raw_distance_m=1.20,
        filtered_distance_m=1.20,
        used_distance_m=1.20,
        source_detail="depth_matched",
        sample_age_sec=0.04,
    ))
    if controller._stable_reverse_radar_distance(frame) != 1.20:
        raise AssertionError("Fresh target-associated Depth must be allowed to drive reverse control")
    held_frame = SensorFrame(distance_m=1.20, distance_state=DistanceState(
        source="vision_depth",
        raw_distance_m=None,
        used_distance_m=1.20,
        source_detail="insufficient_depth_pixels_hold",
        sample_age_sec=0.04,
    ))
    if controller._stable_reverse_radar_distance(held_frame) is not None:
        raise AssertionError("Held Depth must never initiate reverse")

    # A stable far wall must not release a close-distance stop unless the same
    # visual target is clearly smaller and no longer clipped by a side edge.
    release_controller = FollowSafetyController(
        FollowPolicyConfig(
            target_distance_m=1.50,
            target_distance_release_m=1.60,
            target_distance_release_hold_sec=0.0,
            target_distance_release_confirm_frames=2,
            target_distance_release_visual_shrink_ratio=0.90,
            target_distance_release_visual_edge_margin_ratio=0.02,
            initial_target_confirm_frames=1,
        )
    )

    def depth_frame(person: PersonTarget, distance_m: float) -> SensorFrame:
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
                sample_age_sec=0.02,
            ),
        )

    close_person = PersonTarget((80.0, 0.0, 560.0, 479.0), 9, 0.9, 229920.0)
    close_stop = release_controller.decide(1, depth_frame(close_person, 0.72))
    if not close_stop.explicit_stop_requested:
        raise AssertionError(f"close depth must latch stop: {close_stop}")

    false_far = release_controller.decide(2, depth_frame(close_person, 4.40))
    if not false_far.reason.startswith("target_distance_release_visual_wait_bbox_not_smaller"):
        raise AssertionError(f"unchanged large box must reject far wall: {false_far}")

    clipped_person = PersonTarget((0.0, 0.0, 300.0, 479.0), 9, 0.9, 143700.0)
    clipped_far = release_controller.decide(3, depth_frame(clipped_person, 4.40))
    if not clipped_far.reason.startswith("target_distance_release_visual_wait_bbox_edge_clipped"):
        raise AssertionError(f"edge-clipped box must reject far wall: {clipped_far}")

    smaller_person = PersonTarget((200.0, 40.0, 440.0, 440.0), 9, 0.9, 96000.0)
    visual_ready_1 = release_controller.decide(4, depth_frame(smaller_person, 1.70))
    visual_ready_2 = release_controller.decide(5, depth_frame(smaller_person, 1.70))
    if visual_ready_1.reason != "target_distance_release_wait":
        raise AssertionError(f"first valid visual release frame must still wait: {visual_ready_1}")
    if visual_ready_2.explicit_stop_requested or not visual_ready_2.actions:
        raise AssertionError(f"two stable visually consistent frames should release: {visual_ready_2}")

    print("vision_depth_control: distance selection, reverse and visual release safety passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
