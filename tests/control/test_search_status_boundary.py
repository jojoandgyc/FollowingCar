#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.control_types import SensorFrame, SteeringFeedback


def main() -> int:
    controller = FollowSafetyController(FollowPolicyConfig(search_revolution_deg=360.0))
    controller.search_state = "searching"
    controller.search_direction = "left"
    controller.active_target_id = 7
    controller._search_rotation_started_at = 10.0
    controller._search_rotation_accumulated_deg = 123.5

    status = controller.search_status(now=12.5)
    actual = (
        status.state,
        status.direction,
        status.active_target_id,
        status.progress_deg,
        status.target_deg,
        status.elapsed_sec,
    )
    expected = ("searching", "left", 7, 123.5, 360.0, 2.5)
    if actual != expected:
        raise AssertionError(f"unexpected public search status: {actual}")
    try:
        status.progress_deg = 0.0
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("search status must be immutable")
    if controller._search_rotation_accumulated_deg != 123.5:
        raise AssertionError("status consumer modified controller state")

    hold_controller = FollowSafetyController(
        FollowPolicyConfig(
            search_revolution_deg=360.0,
            search_timeout_sec=60.0,
            search_timeout_exit_program=True,
            lost_confirm_frames=1,
        )
    )
    hold_controller.search_state = "searching"
    hold_controller.search_direction = "right"
    hold_controller.active_target_id = 7
    hold_controller._has_seen_person = True
    hold_controller.lost_confirm_frames = 1
    hold_controller._lost_started_at = time.monotonic()
    hold_controller._search_rotation_started_at = time.monotonic()
    hold_controller._search_rotation_feedback_seen = True
    hold_controller._search_rotation_feedback_last_ts = time.monotonic()
    hold_controller._search_rotation_last_integrated_yaw_deg = 359.0
    hold_controller._search_rotation_origin_integrated_yaw_deg = 0.0
    hold_controller._search_heading_from_loss_deg = 359.0
    hold_controller._search_heading_min_deg = 0.0
    hold_controller._search_heading_max_deg = 359.0
    hold_controller._search_rotation_accumulated_deg = 359.0
    frame = SensorFrame(
        width=640,
        height=480,
        persons=[],
        steering_feedback=SteeringFeedback(
            timestamp=time.monotonic(),
            integrated_yaw_right_deg=361.0,
            trustworthy=True,
        ),
    )
    hold_controller.set_search_observation_hold(True)
    held = hold_controller.decide(1, frame)
    if (
        held.reason != "search_candidate_evidence_observe"
        or held.shutdown_requested
        or hold_controller.search_state != "searching"
        or hold_controller._search_rotation_accumulated_deg != 361.0
    ):
        raise AssertionError(f"candidate observation did not defer 360 exit: {held}")
    hold_controller.set_search_observation_hold(False)
    completed = hold_controller.decide(2, frame)
    if not completed.shutdown_requested or completed.reason != "search_revolution_complete":
        raise AssertionError(f"360 exit did not resume after observation: {completed}")
    print("search_status_boundary_ok", actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
