#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.control_types import PersonTarget, SensorFrame, SteeringFeedback
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


def feedback(yaw_deg: float, *, trustworthy: bool = True) -> SteeringFeedback:
    return SteeringFeedback(
        timestamp=time.monotonic(),
        integrated_yaw_right_deg=float(yaw_deg),
        yaw_rate_right_dps=0.0,
        trustworthy=trustworthy,
    )


def frame(*, yaw_deg: float, person: PersonTarget | None = None) -> SensorFrame:
    return SensorFrame(
        width=640,
        height=480,
        persons=[] if person is None else [person],
        distance_m=2.0,
        steering_feedback=feedback(yaw_deg),
    )


def controller() -> FollowSafetyController:
    return FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=1,
            lost_confirm_sec=0.0,
            search_timeout_sec=60.0,
            search_revolution_deg=360.0,
            release_target_on_lost=False,
            stale_direction_recovery_enable=True,
            stale_direction_observe_frames=2,
            stale_direction_observe_max_sec=0.20,
            stale_direction_probe_enable=True,
            stale_direction_probe_angle_deg=12.0,
            stale_direction_probe_observe_frames=2,
            stale_direction_probe_return_tolerance_deg=2.0,
            center_left_ratio=0.40,
            center_right_ratio=0.60,
        )
    )


RIGHT_TARGET = PersonTarget(
    (410, 80, 590, 460),
    track_id=1,
    confidence=0.95,
    area=68400,
)


def establish_target(item: FollowSafetyController) -> None:
    item.decide(1, frame(yaw_deg=0.0, person=RIGHT_TARGET))
    assert item.active_target_id == 1


def assert_zero(decision) -> None:
    assert decision.actions
    assert decision.actions[0].kind == "stop"
    assert decision.actions[0].speed_percent == 0
    assert not decision.explicit_stop_requested


def test_stale_gap_does_not_promote_old_right_side_to_full_search() -> None:
    item = controller()
    establish_target(item)

    assert item.note_stale_visual_result(frame_width=640)
    status = item.search_status()
    assert status.state == "none"
    assert status.direction is None
    assert status.hint_source == "stale_result_gap"
    assert_zero(item.decide(2, frame(yaw_deg=0.0)))
    assert item.search_status().state == "direction_unresolved"


def test_fresh_target_reacquisition_cancels_direction_probe() -> None:
    item = controller()
    establish_target(item)
    item.note_stale_visual_result(frame_width=640)

    decision = item.decide(2, frame(yaw_deg=0.0, person=RIGHT_TARGET))

    assert decision.actions
    assert not decision.clear_action_queue
    assert not decision.stop_action_execution
    assert not decision.person_detected_flag
    assert not item.stale_direction_recovery_active
    assert item.search_status().state == "none"
    assert item.active_target_id == 1


def test_probe_holds_when_encoder_feedback_is_unavailable() -> None:
    item = controller()
    establish_target(item)
    item.note_stale_visual_result(frame_width=640)
    missing_feedback_frame = SensorFrame(
        width=640,
        height=480,
        persons=[],
        distance_m=2.0,
        steering_feedback=feedback(0.0, trustworthy=False),
    )

    assert_zero(item.decide(2, missing_feedback_frame))
    assert_zero(item.decide(3, missing_feedback_frame))
    decision = item.decide(4, missing_feedback_frame)

    assert_zero(decision)
    assert item.search_status().state == "direction_unresolved"
    assert item.search_status().hint_source == "stale_loss_no_reliable_side"


def test_probe_timeout_stops_when_fresh_encoder_does_not_advance() -> None:
    item = controller()
    establish_target(item)
    item.note_stale_visual_result(frame_width=640)
    assert_zero(item.decide(2, frame(yaw_deg=0.0)))
    assert_zero(item.decide(3, frame(yaw_deg=0.0)))
    assert item.decide(4, frame(yaw_deg=0.0)).actions[0].kind == "stop"

    item._stale_direction_recovery_started_at = time.monotonic() - 61.0
    timed_out = item.decide(5, frame(yaw_deg=0.0))

    assert_zero(timed_out)
    assert timed_out.reason == "stale_loss_direction_unresolved"
    assert item.search_status().state == "direction_unresolved"
    assert item.search_status().hint_source == "stale_loss_no_reliable_side"


def test_confirmed_candidate_keeps_directional_search_state() -> None:
    item = controller()
    establish_target(item)
    item.note_stale_visual_result(frame_width=640)

    assert not item.note_stale_candidate_evidence(
        (500.0, 40.0, 620.0, 460.0),
        frame_width=640,
        confirmed=True,
        source="formal",
    )
    assert item.search_status().state == "searching"
    assert item.search_status().direction == "right"

    resumed = item.decide(2, frame(yaw_deg=0.0))
    assert resumed.actions[0].kind == "rotate_right"
    assert resumed.reason == "search_right"


def test_confirmed_candidate_interrupts_locked_search_and_switches_on_opposite_side() -> None:
    item = controller()
    item.cfg = replace(item.cfg, direction_history_enable=True)
    establish_target(item)
    item._target_direction_history.record_visible(
        100,
        1.00,
        target_id=1,
        bbox=(360.0, 40.0, 570.0, 470.0),
        frame_width=640,
        confidence=0.95,
    )
    item._target_direction_history.record_visible(
        101,
        1.07,
        target_id=1,
        bbox=(420.0, 40.0, 639.0, 470.0),
        frame_width=640,
        confidence=0.95,
    )
    item._target_direction_history.record_missing(102, 1.14)
    item.search_state = "searching"
    item.search_direction = "right"
    item._search_rotation_started_at = time.monotonic()
    item._search_rotation_origin_integrated_yaw_deg = 0.0
    item._search_rotation_last_integrated_yaw_deg = 0.0

    assert not item.note_search_candidate_evidence(
        (390.0, 40.0, 620.0, 460.0),
        frame_width=640,
        confirmed=True,
        source="formal",
    )
    assert item.search_status().state == "searching"
    assert item.search_status().direction == "right"

    corrected = item.decide(3, frame(yaw_deg=20.0))
    assert corrected.actions[0].kind == "rotate_right"
    assert corrected.reason == "search_right"


def test_missing_centering_candidate_keeps_search_side() -> None:
    item = controller()
    item.cfg = replace(item.cfg, lost_confirm_frames=3)
    establish_target(item)
    item.search_state = "searching"
    item.search_direction = "right"

    assert not item.note_search_candidate_evidence(
        (200.0, 40.0, 420.0, 460.0),
        frame_width=640,
        confirmed=True,
        source="formal",
    )
    assert not item.note_search_candidate_missing()
    resumed = item.decide(4, frame(yaw_deg=0.0))
    assert resumed.actions[0].kind == "rotate_right"
    assert resumed.reason in ("search_right", "lost_wait_yaw_right")


def test_blocked_opposite_candidate_flips_search_side_after_candidate_gate() -> None:
    item = controller()
    item.cfg = replace(item.cfg, lost_confirm_frames=3)
    establish_target(item)
    item.search_state = "searching"
    item.search_direction = "left"
    item._lost_exit_direction = "left"

    # The first candidate is inside the center corridor, so the active search
    # side is the only available direction evidence.
    assert not item.note_search_candidate_evidence(
        (266.0, 40.0, 313.0, 460.0),
        frame_width=640,
        confirmed=True,
        source="formal",
    )
    # CandidateGate has already selected this as the highest valid detector
    # candidate for the capture frame. The search controller therefore accepts
    # one blocked/low-confidence side observation without waiting another frame.
    assert not item.note_search_candidate_evidence(
        (400.0, 40.0, 540.0, 460.0),
        frame_width=640,
        confirmed=False,
        source="blocked",
    )
    assert item.search_direction == "right"


def test_opposite_candidate_does_not_pause_active_search() -> None:
    item = controller()
    establish_target(item)
    item.search_state = "searching"
    item.search_direction = "left"

    # A detector-only right-side box during a left search immediately flips
    # the active scan direction.
    assert not item.note_search_candidate_evidence(
        (410.0, 40.0, 560.0, 460.0),
        frame_width=640,
        confirmed=True,
        source="formal",
    )
    assert item.search_state == "searching"
    assert item.search_direction == "right"


def main() -> int:
    test_stale_gap_does_not_promote_old_right_side_to_full_search()
    test_fresh_target_reacquisition_cancels_direction_probe()
    test_probe_holds_when_encoder_feedback_is_unavailable()
    test_probe_timeout_stops_when_fresh_encoder_does_not_advance()
    test_confirmed_candidate_keeps_directional_search_state()
    test_confirmed_candidate_interrupts_locked_search_and_switches_on_opposite_side()
    test_missing_centering_candidate_keeps_search_side()
    test_blocked_opposite_candidate_flips_search_side_after_candidate_gate()
    test_opposite_candidate_does_not_pause_active_search()
    print("stale_direction_recovery_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
