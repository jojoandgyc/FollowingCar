#!/usr/bin/env python3
from __future__ import annotations

import time

from car_control_modular.control_types import PersonTarget, SensorFrame, SteeringFeedback
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.target_direction_history import TargetDirectionHistory


def _history() -> TargetDirectionHistory:
    return TargetDirectionHistory(
        max_frames=60,
        lookback_samples=6,
        max_capture_gap=12,
        edge_margin_ratio=0.04,
        outer_center_ratio=0.60,
        min_motion_ratio=0.015,
    )


def test_capture_timeline_resolves_outward_right_edge_motion() -> None:
    history = _history()
    history.record_visible(900, 1.00, target_id=1, bbox=(360, 40, 570, 470), frame_width=640, confidence=0.95)
    history.record_visible(901, 1.07, target_id=1, bbox=(420, 40, 639, 470), frame_width=640, confidence=0.95)
    history.record_stale(902, 1.14)
    history.record_missing(903, 1.21)
    history.record_missing(904, 1.28)
    history.record_missing(905, 1.35)

    decision = history.resolve(required_missing_frames=3)

    assert decision.direction == "right"
    assert decision.reason == "capture_timeline_exit_side"
    assert decision.last_visible_capture_frame_id == 901
    assert decision.missing_frames == 3
    last_visible = history.entries[1]
    assert last_visible.edge_side == "right"
    assert last_visible.motion_direction == "right"
    assert last_visible.velocity_ratio_s is not None
    assert last_visible.velocity_ratio_s > 0.0


def test_unknown_and_stale_slots_never_count_as_fresh_missing() -> None:
    history = _history()
    history.record_visible(20, 1.00, target_id=1, bbox=(0, 40, 180, 470), frame_width=640, confidence=0.95)
    history.record_unknown(21, 1.03, "camera_drain_skipped")
    history.record_stale(22, 1.06)
    history.record_missing(23, 1.09)
    history.record_missing(24, 1.12)

    decision = history.resolve(required_missing_frames=3)

    assert decision.direction is None
    assert decision.reason == "missing_confirmation_pending"
    assert decision.missing_frames == 2


def test_latest_reliable_side_is_available_after_async_gap() -> None:
    history = _history()
    history.record_visible(
        30,
        1.00,
        target_id=1,
        bbox=(430, 40, 639, 470),
        frame_width=640,
        confidence=0.90,
    )
    history.record_unknown(31, 1.03, "worker_backlog")
    history.record_stale(32, 1.06)

    decision = history.latest_reliable_side()

    assert decision.direction == "right"
    assert decision.reason == "latest_reliable_capture_side"
    assert decision.last_visible_capture_frame_id == 30


def test_single_edge_sample_always_selects_search_side() -> None:
    history = _history()
    history.record_visible(10, 1.00, target_id=1, bbox=(420, 40, 639, 470), frame_width=640, confidence=0.95)
    history.record_missing(11, 1.07)
    history.record_missing(12, 1.14)
    history.record_missing(13, 1.21)

    decision = history.resolve(required_missing_frames=3)

    assert decision.direction == "right"
    assert decision.reason == "capture_timeline_exit_side"


def test_latest_motion_reversal_does_not_veto_search_side() -> None:
    history = _history()
    history.record_visible(30, 1.00, target_id=1, bbox=(430, 40, 639, 470), frame_width=640, confidence=0.95)
    history.record_visible(31, 1.07, target_id=1, bbox=(350, 40, 590, 470), frame_width=640, confidence=0.95)
    history.record_missing(32, 1.14)
    history.record_missing(33, 1.21)
    history.record_missing(34, 1.28)

    decision = history.resolve(required_missing_frames=3)

    assert decision.direction == "right"
    assert decision.reason == "capture_timeline_exit_side"


def test_stable_side_trace_keeps_search_side_after_inward_motion() -> None:
    history = _history()
    history.record_visible(40, 1.00, target_id=1, bbox=(0, 40, 150, 470), frame_width=640, confidence=0.60)
    history.record_visible(41, 1.07, target_id=1, bbox=(20, 40, 170, 470), frame_width=640, confidence=0.60)
    history.record_visible(42, 1.14, target_id=1, bbox=(40, 40, 190, 470), frame_width=640, confidence=0.60)
    history.record_missing(43, 1.21)
    history.record_missing(44, 1.28)
    history.record_missing(45, 1.35)

    decision = history.resolve(required_missing_frames=3)

    assert decision.direction == "left"
    assert decision.reason == "capture_timeline_exit_side"
    assert decision.confidence > 0.0


def test_visible_exit_trace_survives_delayed_search_service() -> None:
    history = _history()
    history.record_visible(1, 1.00, target_id=1, bbox=(280, 40, 500, 470), frame_width=640, confidence=0.70)
    history.record_visible(2, 1.07, target_id=1, bbox=(0, 40, 220, 470), frame_width=640, confidence=0.40)
    for capture_id in range(3, 90):
        history.record_missing(capture_id, float(capture_id) / 15.0)

    decision = history.resolve(required_missing_frames=3)

    assert decision.direction == "left"
    assert decision.last_visible_capture_frame_id == 2
    assert decision.missing_frames == 60


def test_low_quality_geometry_still_establishes_exit_direction() -> None:
    controller = FollowSafetyController(
        FollowPolicyConfig(
            direction_history_enable=True,
            lost_confirm_frames=3,
            lost_confirm_sec=0.0,
            initial_target_confirm_frames=1,
            search_timeout_sec=60.0,
        )
    )
    first = PersonTarget((280, 50, 500, 470), track_id=1, confidence=0.95, area=92400)
    cropped = PersonTarget((0, 50, 220, 470), track_id=1, confidence=0.40, area=92400)
    controller.decide(1, _frame(1, first))
    controller.decide(2, _frame(2, cropped), low_quality_visible=True)
    controller.decide(3, _frame(3, None))
    controller.decide(4, _frame(4, None))
    decision = controller.decide(5, _frame(5, None))

    assert decision.actions[0].kind == "rotate_left"
    assert controller.search_direction == "left"


def _feedback() -> SteeringFeedback:
    return SteeringFeedback(
        timestamp=time.monotonic(),
        integrated_yaw_right_deg=0.0,
        yaw_rate_right_dps=0.0,
        trustworthy=True,
    )


def _frame(capture_id: int, person: PersonTarget | None) -> SensorFrame:
    return SensorFrame(
        width=640,
        height=480,
        persons=[] if person is None else [person],
        distance_m=2.0,
        steering_feedback=_feedback(),
        capture_frame_id=capture_id,
        capture_timestamp=float(capture_id) / 15.0,
    )


def test_controller_freezes_search_direction_from_capture_timeline() -> None:
    controller = FollowSafetyController(
        FollowPolicyConfig(
            direction_history_enable=True,
            lost_confirm_frames=3,
            lost_confirm_sec=0.0,
            initial_target_confirm_frames=1,
            search_timeout_sec=60.0,
        )
    )
    first = PersonTarget((350, 50, 570, 470), track_id=1, confidence=0.95, area=92400)
    edge = PersonTarget((420, 50, 639, 470), track_id=1, confidence=0.95, area=91980)
    controller.decide(1, _frame(100, first))
    controller.decide(2, _frame(101, edge))

    wait_one = controller.decide(3, _frame(102, None))
    assert controller.search_state == "none"
    wait_two = controller.decide(4, _frame(103, None))
    search = controller.decide(5, _frame(104, None))

    assert wait_one.actions[0].kind == "rotate_right"
    assert wait_one.reason == "lost_history_hold_right"
    assert wait_one.evidence_capture_frame_id == 101
    assert wait_two.actions[0].kind == "rotate_right"
    assert search.actions[0].kind == "rotate_right"
    assert controller.search_direction == "right"
    assert controller.search_status().hint_source == "capture_timeline_exit_side"


def test_unverified_direction_classifier_cannot_replace_last_target_side() -> None:
    controller = FollowSafetyController(
        FollowPolicyConfig(
            direction_history_enable=True,
            lost_confirm_frames=3,
            lost_confirm_sec=0.0,
            initial_target_confirm_frames=1,
            search_timeout_sec=60.0,
        )
    )
    left_target = PersonTarget((0, 50, 120, 470), track_id=1, confidence=0.95, area=50400)
    controller.decide(1, _frame(100, left_target))

    # The asynchronous direction worker sees a right-side person but has no
    # ReID/DeepSORT identity. It must not overwrite the trusted left exit slot.
    controller.note_direction_classifier_evidence(
        101,
        101.0 / 15.0,
        state="visible",
        bbox=(520, 50, 639, 470),
        frame_width=640,
        confidence=0.90,
        reason="detector_formal_person_side",
    )
    wait_one = controller.decide(2, _frame(101, None))
    wait_two = controller.decide(3, _frame(102, None))
    search = controller.decide(4, _frame(103, None))

    assert wait_one.actions[0].kind == "rotate_left"
    assert wait_two.actions[0].kind == "rotate_left"
    assert search.actions[0].kind == "rotate_left"
    assert controller.search_direction == "left"


def test_controller_searches_side_after_reversed_edge_history() -> None:
    controller = FollowSafetyController(
        FollowPolicyConfig(
            direction_history_enable=True,
            lost_confirm_frames=3,
            lost_confirm_sec=0.0,
            initial_target_confirm_frames=1,
            search_timeout_sec=60.0,
        )
    )
    edge = PersonTarget((430, 50, 639, 470), track_id=1, confidence=0.95, area=87780)
    returning = PersonTarget((350, 50, 590, 470), track_id=1, confidence=0.95, area=100800)
    controller.decide(1, _frame(200, edge))
    controller.decide(2, _frame(201, returning))
    controller.decide(3, _frame(202, None))
    controller.decide(4, _frame(203, None))
    decision = controller.decide(5, _frame(204, None))

    assert decision.actions[0].kind == "rotate_right"
    assert controller.search_state == "searching"
    assert controller.search_direction == "right"
    assert controller.search_status().hint_source == "capture_timeline_exit_side"


def test_stale_gap_promotes_strong_capture_history_after_fresh_missing_confirm() -> None:
    controller = FollowSafetyController(
        FollowPolicyConfig(
            direction_history_enable=True,
            stale_direction_recovery_enable=True,
            lost_confirm_frames=3,
            lost_confirm_sec=0.0,
            initial_target_confirm_frames=1,
            search_timeout_sec=60.0,
        )
    )
    first = PersonTarget((350, 50, 570, 470), track_id=1, confidence=0.95, area=92400)
    edge = PersonTarget((420, 50, 639, 470), track_id=1, confidence=0.95, area=91980)
    controller.decide(1, _frame(300, first))
    controller.decide(2, _frame(301, edge))
    assert controller.note_stale_visual_result(
        frame_width=640,
        capture_frame_id=302,
        capture_timestamp=20.13,
    )

    wait_one = controller.decide(3, _frame(303, None))
    wait_two = controller.decide(4, _frame(304, None))
    search = controller.decide(5, _frame(305, None))

    assert wait_one.actions[0].kind == "stop"
    assert wait_two.actions[0].kind == "stop"
    assert search.actions[0].kind == "rotate_right"
    assert search.reason == "search_history_right"
    assert controller.search_direction == "right"
    assert not controller.stale_direction_recovery_active
