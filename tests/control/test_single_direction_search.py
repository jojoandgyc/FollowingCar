#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.control_types import PersonTarget, SensorFrame, SteeringFeedback
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


def feedback(yaw_deg: float) -> SteeringFeedback:
    return SteeringFeedback(
        timestamp=time.monotonic(),
        integrated_yaw_right_deg=float(yaw_deg),
        trustworthy=True,
    )


def missing_frame(yaw_deg: float) -> SensorFrame:
    return SensorFrame(
        width=640,
        height=480,
        persons=[],
        distance_m=2.0,
        steering_feedback=feedback(yaw_deg),
    )


def test_search_direction_stays_locked_until_completion() -> None:
    controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=1,
            lost_confirm_sec=0.0,
            search_timeout_sec=60.0,
            search_revolution_deg=360.0,
            release_target_on_lost=False,
        )
    )
    target = PersonTarget((60, 80, 220, 460), track_id=1, confidence=0.9, area=60800)
    controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[target],
            distance_m=2.0,
            steering_feedback=feedback(0.0),
        ),
    )

    first = controller.decide(2, missing_frame(0.0))
    assert first.actions and first.actions[0].kind == "rotate_left"
    assert controller.search_status().stage == "single_direction"

    # Later evidence and headings must not reverse the direction selected at loss.
    for frame_index, yaw_deg in enumerate(
        (-40.0, -100.0, -160.0, -220.0, -280.0, -359.0),
        start=3,
    ):
        decision = controller.decide(
            frame_index,
            missing_frame(yaw_deg),
        )
        assert not decision.shutdown_requested
        assert decision.actions and decision.actions[0].kind == "rotate_left"
        assert controller.search_status().direction == "left"

    completed = controller.decide(9, missing_frame(-360.0))
    assert completed.shutdown_requested
    assert completed.reason == "search_revolution_complete"
    assert controller.search_status().coverage_deg >= 360.0


def test_confirmation_window_uses_first_missing_direction() -> None:
    controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=3,
            lost_confirm_sec=0.0,
            search_timeout_sec=60.0,
            release_target_on_lost=False,
        )
    )
    target = PersonTarget((60, 80, 220, 460), track_id=1, confidence=0.9, area=60800)
    controller.decide(
        1,
        SensorFrame(
            width=640,
            height=480,
            persons=[target],
            distance_m=2.0,
            steering_feedback=feedback(0.0),
        ),
    )

    first = controller.decide(2, missing_frame(0.0))
    second = controller.decide(3, missing_frame(-2.0))
    search = controller.decide(4, missing_frame(-4.0))

    assert first.actions and first.actions[0].kind == "rotate_left"
    assert second.actions and second.actions[0].kind == "rotate_left"
    assert search.actions and search.actions[0].kind == "rotate_left"
    assert controller.search_status().direction == "left"


def main() -> int:
    test_search_direction_stays_locked_until_completion()
    test_confirmation_window_uses_first_missing_direction()
    print("single_direction_search_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
