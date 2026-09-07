import time

from car_control_modular.control_types import (
    LateralCandidateEvidence,
    PersonTarget,
    SensorFrame,
)
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.search_candidate_gate import (
    CandidateObservation,
    SearchCandidateGate,
    SearchCandidateGateConfig,
)


def _searching_controller(direction="left", **config_overrides):
    controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=3,
            center_left_ratio=0.30,
            center_right_ratio=0.70,
            **config_overrides,
        )
    )
    controller.active_target_id = 1
    controller._has_seen_person = True
    controller.search_state = "searching"
    controller.search_direction = direction
    controller._lost_exit_direction = direction
    controller.lost_confirm_frames = 3
    controller._lost_started_at = time.monotonic()
    return controller


def _candidate_frame(
    capture_frame_id,
    bbox,
    score,
    *,
    evidence_capture_frame_id=None,
    active_target_match=False,
):
    return SensorFrame(
        width=640,
        height=480,
        persons=[],
        capture_frame_id=capture_frame_id,
        lateral_candidate=LateralCandidateEvidence(
            capture_frame_id=(
                capture_frame_id
                if evidence_capture_frame_id is None
                else evidence_capture_frame_id
            ),
            bbox=bbox,
            score=score,
            source="formal" if score >= 0.25 else "probe",
            active_target_match=active_target_match,
        ),
    )


def test_current_left_candidate_overrides_right_history_on_search_entry():
    controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=3,
            direction_history_enable=True,
            search_cooldown=0,
            search_timeout_sec=30.0,
            search_candidate_untracked_min_score=0.10,
        )
    )
    controller.active_target_id = 1
    controller._has_seen_person = True
    controller.lost_confirm_frames = 2
    controller.search_direction = "right"
    controller._lost_exit_direction = "right"

    decision = controller.decide(
        100,
        _candidate_frame(
            745,
            (20.0, 7.0, 240.0, 463.0),
            0.802,
            active_target_match=True,
        ),
    )

    assert controller.search_state == "searching"
    assert controller.search_direction == "left"
    assert decision.actions and decision.actions[0].kind == "rotate_left"
    assert decision.reason == "search_current_candidate_left"
    assert decision.current_forward_percent == 0
    assert decision.evidence_capture_frame_id == 745


def test_wide_candidate_uses_bbox_center_instead_of_aimline_intersection():
    controller = _searching_controller(
        "right",
        search_cooldown=0,
        search_candidate_untracked_min_score=0.10,
    )

    decision = controller.decide(
        100,
        _candidate_frame(743, (265.9, 0.0, 533.1, 479.0), 0.818),
    )

    assert controller.search_direction == "right"
    assert decision.actions == []
    assert decision.explicit_stop_requested is True
    assert decision.reason == "search_candidate_aimline_brake"
    assert decision.evidence_capture_frame_id == 743


def test_edge_target_crossing_aimline_still_turns_toward_bbox_center():
    controller = _searching_controller(
        "right",
        search_cooldown=0,
        search_candidate_untracked_min_score=0.10,
    )

    decision = controller.decide(
        100,
        _candidate_frame(
            85,
            (0.0, 0.0, 325.0, 479.0),
            0.554,
            active_target_match=True,
        ),
    )

    assert controller.search_direction == "left"
    assert decision.actions == []
    assert decision.explicit_stop_requested is True
    assert decision.reason == "search_candidate_aimline_brake"
    assert decision.evidence_capture_frame_id == 85


def test_candidate_approaching_aimline_uses_slow_search_reason():
    controller = _searching_controller(
        "right",
        search_cooldown=0,
        search_candidate_untracked_min_score=0.10,
        search_candidate_approach_margin_ratio=0.08,
    )

    decision = controller.decide(
        100,
        _candidate_frame(741, (337.0, 0.0, 600.1, 479.0), 0.810),
    )

    assert controller.search_direction == "right"
    assert decision.actions and decision.actions[0].kind == "rotate_right"
    assert decision.reason == "search_candidate_approach_right"
    assert decision.evidence_capture_frame_id == 741


def test_aimline_brake_then_latest_candidate_side_remains_braked_until_identity_confirmation():
    controller = _searching_controller(
        "right",
        search_cooldown=0,
        search_candidate_untracked_min_score=0.10,
    )

    brake = controller.decide(
        100,
        _candidate_frame(743, (200.0, 0.0, 440.0, 479.0), 0.818),
    )
    correction = controller.decide(
        101,
        _candidate_frame(745, (100.0, 20.0, 300.0, 470.0), 0.802),
    )

    assert brake.reason == "search_candidate_aimline_brake"
    assert brake.explicit_stop_requested is True
    assert controller.search_direction == "right"
    assert correction.actions == []
    assert correction.explicit_stop_requested is True
    assert correction.reason == "search_candidate_aimline_brake"
    assert correction.evidence_capture_frame_id == 745


def test_current_untracked_probe_cannot_reverse_active_search_without_continuity():
    controller = _searching_controller(
        "right",
        search_cooldown=0,
        search_candidate_untracked_min_score=0.10,
    )

    decision = controller.decide(
        101,
        _candidate_frame(746, (0.0, 40.0, 180.0, 460.0), 0.20),
    )

    assert controller.search_direction == "right"
    assert decision.actions and decision.actions[0].kind == "rotate_right"
    assert decision.reason == "search_right"


def test_candidate_below_untracked_threshold_cannot_switch_search():
    controller = _searching_controller(
        "right",
        search_cooldown=0,
        search_candidate_untracked_min_score=0.10,
    )

    decision = controller.decide(
        102,
        _candidate_frame(747, (0.0, 40.0, 180.0, 460.0), 0.09),
    )

    assert controller.search_direction == "right"
    assert decision.actions and decision.actions[0].kind == "rotate_right"


def test_candidate_from_another_capture_slot_cannot_switch_search():
    controller = _searching_controller(
        "right",
        search_cooldown=0,
        search_candidate_untracked_min_score=0.10,
    )

    decision = controller.decide(
        103,
        _candidate_frame(
            748,
            (0.0, 40.0, 180.0, 460.0),
            0.90,
            evidence_capture_frame_id=747,
        ),
    )

    assert controller.search_direction == "right"
    assert decision.actions and decision.actions[0].kind == "rotate_right"


def test_current_candidate_bypasses_near_target_missing_wait_for_lateral_only():
    controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=3,
            direction_history_enable=True,
            search_cooldown=0,
            search_candidate_untracked_min_score=0.10,
        )
    )
    controller.active_target_id = 1
    controller._has_seen_person = True
    controller._target_stop_latched = True
    controller.search_direction = "right"
    controller._lost_exit_direction = "right"

    decision = controller.decide(
        104,
        _candidate_frame(
            749,
            (0.0, 40.0, 180.0, 460.0),
            0.20,
            active_target_match=True,
        ),
    )

    assert controller.search_direction == "left"
    assert decision.actions and decision.actions[0].kind == "rotate_left"
    assert decision.current_forward_percent == 0
    assert decision.evidence_capture_frame_id == 749


def test_one_opposite_candidate_switches_search_direction():
    controller = _searching_controller("left")

    # A single fresh box on the right supersedes the stale left sweep.
    controller.note_search_candidate_evidence(
        (500.0, 80.0, 620.0, 420.0),
        frame_width=640,
        confirmed=True,
        source="formal",
    )

    assert controller.search_state == "searching"
    assert controller.search_direction == "right"
    assert controller._lost_exit_direction == "right"


def test_untracked_candidate_above_direction_threshold_cannot_override_history_direction():
    controller = _searching_controller("right")

    controller.note_search_candidate_evidence(
        (0.0, 40.0, 180.0, 460.0),
        frame_width=640,
        confirmed=True,
        source="formal",
        candidate_score=0.28,
        candidate_tracked=False,
    )

    assert controller.search_state == "searching"
    assert controller.search_direction == "right"
    assert controller._lost_exit_direction == "right"


def test_blocked_high_score_untracked_candidate_cannot_reverse_search():
    controller = _searching_controller("left")

    # CandidateGate's bounded hold does not prove that this person is the
    # target, even when detector confidence is high.
    controller.note_search_candidate_evidence(
        (500.0, 40.0, 625.0, 430.0),
        frame_width=640,
        confirmed=False,
        source="blocked",
        candidate_score=0.84,
        candidate_tracked=False,
    )

    assert controller.search_state == "searching"
    assert controller.search_direction == "left"
    assert controller._lost_exit_direction == "left"


def test_discontinuous_opposite_person_keeps_last_target_exit_direction():
    controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=3,
            direction_history_enable=True,
            search_cooldown=0,
            search_timeout_sec=30.0,
            search_candidate_untracked_min_score=0.10,
        )
    )
    controller.active_target_id = 1
    controller._has_seen_person = True
    controller.last_person_center_x = 96.6
    controller._target_direction_history.record_visible(
        575,
        1.0,
        target_id=1,
        bbox=(0.0, 0.0, 193.2, 479.0),
        frame_width=640,
        confidence=0.787,
    )

    decision = controller.decide(
        302,
        _candidate_frame(577, (463.0, 40.0, 598.0, 430.0), 0.445),
    )

    assert decision.actions and decision.actions[0].kind == "rotate_left"
    assert decision.reason == "lost_history_hold_left"
    assert decision.evidence_capture_frame_id == 575
    assert controller._lost_exit_direction != "right"


def test_opposite_candidate_after_center_intersection_remains_braked_until_identity_confirmation():
    controller = _searching_controller(
        "right",
        search_cooldown=0,
        search_candidate_untracked_min_score=0.10,
    )

    brake = controller.decide(
        100,
        _candidate_frame(743, (200.0, 0.0, 440.0, 479.0), 0.80),
    )
    correction = controller.decide(
        101,
        _candidate_frame(745, (100.0, 10.0, 300.0, 470.0), 0.79),
    )

    assert brake.reason == "search_candidate_aimline_brake"
    assert correction.actions == []
    assert correction.explicit_stop_requested is True
    assert correction.reason == "search_candidate_aimline_brake"
    assert controller.search_direction == "right"


def test_low_quality_opposite_candidate_can_override_history_direction():
    controller = _searching_controller("right")

    controller.note_search_candidate_evidence(
        (0.0, 0.0, 140.0, 479.0),
        frame_width=640,
        confirmed=True,
        source="low_quality_search",
        candidate_score=0.90,
        candidate_tracked=True,
    )

    assert controller.search_state == "searching"
    assert controller.search_direction == "left"


def test_search_candidate_gate_selects_highest_probe_score():
    gate = SearchCandidateGate(SearchCandidateGateConfig(hold_frames=1))
    low = CandidateObservation((10.0, 80.0, 130.0, 400.0), 0.12)
    high = CandidateObservation((500.0, 80.0, 620.0, 400.0), 0.18)

    decision = gate.update(
        timestamp=1.0,
        search_active=True,
        width=640,
        height=480,
        probe_candidates=(low, high),
    )

    assert decision.source == "probe"
    assert decision.completed
    assert decision.score == high.score
    assert decision.bbox == high.bbox


def test_low_quality_visible_search_keeps_lateral_motion_without_forward():
    controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=3,
            visible_steering_pid_enable=True,
            parked_recenter_min_rpm=1,
            parked_recenter_max_rpm=14,
        )
    )
    controller.active_target_id = 1
    controller._has_seen_person = True
    controller.search_state = "searching"
    controller.search_direction = "right"
    controller._search_rotation_accumulated_deg = 42.0
    person = PersonTarget(
        (500.0, 80.0, 639.0, 479.0),
        track_id=1,
        confidence=0.8,
        area=50000.0,
    )

    decision = controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[person]),
        target_steerable=False,
        target_steering_limit_rpm=6.0,
        low_quality_visible=True,
    )

    assert decision.reason == "target_visible_low_quality_yaw"
    assert decision.actions
    assert decision.actions[0].kind == "rotate_right"
    assert controller.last_steering_pid_result is not None
    assert 1 <= controller.last_steering_pid_result.correction_rpm <= 6
    assert decision.current_forward_percent == 0
    assert not decision.explicit_stop_requested
    assert controller.search_state == "searching"
    assert controller.search_direction == "right"
    assert controller._search_rotation_accumulated_deg == 42.0
