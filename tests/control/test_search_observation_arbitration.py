"""Search observation is provisional until the frame's control owner is known.

Only the runtime constructor and hardware/dispatch endpoints are replaced.
Gate preparation, track filtering and final observation ownership are real.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import request_0513_modular as runtime
from car_control_modular.lateral_intent import LateralControlIntent, LateralIntentStore
from car_control_modular.search_candidate_gate import (
    CandidateObservation, SearchCandidateGate, SearchCandidateGateConfig,
    SearchCandidateGateDecision,
)
from rk_vision.tracker import TrackRecord


NOW = 200.0


def _record(track=3, uid=7, bbox=(20.0, 20.0, 200.0, 460.0), *, score=0.9):
    x1, y1, x2, y2 = bbox
    return TrackRecord(
        track_id=track, reid_uid=uid, x1=x1, y1=y1, x2=x2, y2=y2,
        class_id=0, score=score, cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
        area=(x2 - x1) * (y2 - y1), angle_deg=0.0,
        tracker_state=2, time_since_update=0,
    )


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW)
    monkeypatch.setattr(runtime, "VISION_REID_ENABLE", True)
    monkeypatch.setattr(runtime, "VISION_TRACK_LOG_ENABLE", False)
    monkeypatch.setattr(runtime, "VISION_CONTROL_USE_PREDICTED_TRACKS", False)
    monkeypatch.setattr(runtime, "SINGLE_PERSON_GEOMETRY_FALLBACK_ENABLE", False)
    tracker = object.__new__(runtime.PersonTracker)
    tracker.frame_index = 100
    tracker._active_capture_frame_id = 231
    tracker._active_capture_timestamp = NOW - 0.08
    tracker._control_update_lock = threading.RLock()
    tracker._lateral_intent_store = LateralIntentStore()
    tracker.search_state = "searching"
    tracker.search_direction = "left"
    tracker._vision_control_state = "target_lost"
    tracker._search_evidence_observation_active = False
    tracker._search_evidence_observation_source = "none"
    tracker._search_evidence_observation_deadline = 0.0
    tracker._search_evidence_pause_current_frame = False
    tracker._explicit_stop_requested = False
    tracker._runtime_shutdown_requested = False
    tracker.running = True
    tracker._events = []
    tracker._context_events = []
    tracker._assignments = {}
    tracker._deferred_timeout = []
    tracker._lateral_yaw_revision = 0
    tracker._lateral_direction_intent = "left"
    tracker._follow_controller = SimpleNamespace(
        active_target_id=7, search_state="searching", search_direction="left",
        _search_observation_hold=False,
        defer_search_timeout=lambda value: tracker._deferred_timeout.append(value),
    )
    tracker._follow_controller.set_search_observation_hold = lambda value: setattr(
        tracker._follow_controller, "_search_observation_hold", bool(value)
    )
    tracker._bunker_runtime = SimpleNamespace(check_merged_dets=lambda *_args: None)
    tracker._handle_hazard_safety_state = lambda state: False
    tracker._identity_assignment_debug_for_track = lambda track: dict(tracker._assignments.get(track, {}))
    tracker._single_person_geometry_fallback_id = lambda *args: None
    tracker._search_geometry_reacquire_id = lambda *args, **kwargs: None
    tracker._visual_reacquire_hold_match = lambda *args, **kwargs: None
    tracker._visual_reacquire_hold_uid = None
    tracker._visible_unsteerable_uid = None
    tracker._hold_for_confirmed_search_reacquire = lambda *args, **kwargs: False
    tracker._clear_longitudinal_context = lambda **kwargs: tracker._context_events.append("clear")
    tracker._publish_longitudinal_context = lambda *args, **kwargs: tracker._context_events.append("publish")
    tracker._queue_actions_for_persons = lambda w, h, persons, **kwargs: tracker._events.append(
        ("normal", list(persons), kwargs)
    )
    tracker._hold_for_visible_unsteerable_target = lambda **kwargs: (
        tracker._events.append(("limited_yaw", kwargs)) or True
    )
    tracker._last_control_decision_reason = "search_left"
    tracker._action_runtime = SimpleNamespace(cancel_active_rotate_for_observation=lambda reason: None)
    tracker._replace_action_queue = lambda actions, reason: tracker._events.append(
        ("queue", list(actions), reason)
    )
    return tracker


def _observe_decision(**changes):
    args = dict(
        pause_rotation=True, entered=True, completed=False, source="formal",
        reason="formal_candidate_observe_start", bbox=(20.0, 20.0, 200.0, 460.0),
        score=0.9, hold_frame=1, hold_frames=2,
    )
    args.update(changes)
    return SearchCandidateGateDecision(**args)


def _prepare_observation(owner):
    owner._apply_search_candidate_gate_decision(_observe_decision(), prepare_only=True)
    owner._search_evidence_pause_current_frame = True
    owner._follow_controller.set_search_observation_hold(True)


@pytest.mark.parametrize("case", ["same_uid", "missing", "other_uid", "weak", "duplicate_uid"])
def test_real_visual_classification_preserves_only_unique_trusted_depth_owner(owner, case):
    # Use the real context method, not the search fixture's context spy.
    del owner._clear_longitudinal_context
    owner._longitudinal_context_lock = threading.Lock()
    owner._longitudinal_context = {"old_roi": True}
    before = ("forward", 40, 7, NOW-.04)
    owner._depth30_linear_snapshot = before
    owner._current_forward_allow_below_min = True
    owner.search_state = owner._follow_controller.search_state = "none"
    owner._vision_control_state = "target_visible_depth_valid"
    seen_at_queue = []
    owner._queue_actions_for_persons = lambda *a, **kw: seen_at_queue.append(owner._depth30_linear_snapshot)
    records = [_record()]
    if case == "missing":
        records = []
    elif case == "other_uid":
        records = [_record(uid=8)]
    elif case == "duplicate_uid":
        records.append(_record(track=4, bbox=(400., 20., 600., 460.)))
    elif case == "weak":
        owner._assignments[3] = {"mapped_uid": 7, "bbox_quality_ok": False,
                                 "bbox_quality_reason": "edge_touch>2", "reason": "mapped_weak_observed"}
    # Run lateral composition during classification in the dedicated authority
    # tests; here verify the real consume entry and every rejection exit.
    owner._consume_track_records(records, 640, 480, "test")
    if case == "same_uid":
        assert seen_at_queue == [before]
        assert owner._depth30_linear_snapshot == before
        assert owner._current_forward_allow_below_min
    else:
        assert owner._depth30_linear_snapshot is None
        assert all(item is None for item in seen_at_queue)
        assert not owner._current_forward_allow_below_min


def test_gate_preparation_has_no_motor_or_queue_side_effect(owner):
    _prepare_observation(owner)
    assert owner._search_evidence_observation_active
    assert owner._search_evidence_observation_deadline > NOW
    assert owner._events == []


def test_legacy_apply_still_commits_soft_stop_without_hardware(owner):
    owner._apply_search_candidate_gate_decision(_observe_decision())
    assert owner._events == [
        ("queue", [runtime.ACTION_STOP], "search_candidate_evidence_observe")
    ]
    assert owner._use_soft_stop_next


@pytest.mark.parametrize("hold_frames", [1, 2, 3])
def test_completed_observation_releases_window_instead_of_waiting_timeout(owner, hold_frames):
    _prepare_observation(owner)
    owner._apply_search_candidate_gate_decision(
        _observe_decision(entered=False, completed=True, hold_frame=hold_frames, hold_frames=hold_frames),
        prepare_only=True,
    )
    assert not owner._search_evidence_observation_active
    assert owner._search_evidence_observation_deadline == 0.0
    assert not owner._search_evidence_pause_current_frame
    assert not owner._follow_controller._search_observation_hold
    assert owner._events == []


def test_two_frame_gate_completion_does_not_add_300ms_runtime_hold(owner):
    gate = SearchCandidateGate(SearchCandidateGateConfig(hold_frames=2, max_hold_sec=0.30))
    candidate = CandidateObservation((20.0, 20.0, 200.0, 460.0), 0.9)
    first = gate.update(timestamp=NOW, search_active=True, width=640, height=480, formal_candidates=(candidate,))
    owner._apply_search_candidate_gate_decision(first, prepare_only=True)
    second = gate.update(timestamp=NOW + 0.10, search_active=True, width=640, height=480, formal_candidates=(candidate,))
    assert second.completed and second.pause_rotation
    owner._apply_search_candidate_gate_decision(second, prepare_only=True)
    assert not owner._search_evidence_observation_active
    assert owner._events == []


def test_active_uid_normal_control_wins_without_intermediate_observation_stop(owner):
    _prepare_observation(owner)
    owner._consume_track_records([_record()], 640, 480, "test")
    assert [event[0] for event in owner._events] == ["normal"]
    assert owner._events[0][1][0][1] == 7
    assert not owner._search_evidence_pause_current_frame
    assert not owner._follow_controller._search_observation_hold
    assert not owner._search_evidence_observation_active


def test_mapped_partial_person_uses_only_limited_yaw_not_observation_stop(owner):
    _prepare_observation(owner)
    owner._assignments[3] = {
        "mapped_uid": 7, "bbox_quality_ok": False,
        "bbox_quality_reason": "edge_touch>2", "reason": "mapped_weak_observed",
        "match_source": "strong", "distance": 0.03, "strong_distance": 0.03,
    }
    owner._consume_track_records([_record(uid=0, bbox=(0.0, 0.0, 430.0, 479.0))], 640, 480, "test")
    assert [event[0] for event in owner._events] == ["limited_yaw"]
    assert owner._events[0][1]["reid_uid"] == 7
    assert "publish" not in owner._context_events
    assert not owner._search_evidence_pause_current_frame


def _multi_person_capture(owner, *, capture_id, timestamp, competitor_distance):
    owner._active_capture_frame_id = capture_id
    owner._active_capture_timestamp = timestamp
    selected = {
        "uid": 0, "mapped_uid": 7, "bbox_quality_ok": False,
        "bbox_quality_tier": "weak", "bbox_quality_reason": "edge_touch>2",
        "reason": "mapped_weak_observed", "match_source": "strong",
        "distance": 0.08, "strong_distance": 0.08,
    }
    other = {
        "uid": 0, "best_uid": 7, "bbox_quality_ok": True,
        "match_source": "strong", "distance": competitor_distance,
    }
    owner._assignments = {3: selected, 4: other}
    observations = []
    for source_index, track, assignment, bbox in (
        (0, 3, selected, (1.0, 2.0, 246.0, 475.0)),
        (1, 4, other, (400.0, 60.0, 580.0, 460.0)),
    ):
        observations.append({
            "raw_track_id": track, "uid": 0, "detector_bbox": bbox,
            "sample_metadata": {
                "is_fresh": True, "capture_frame_id": capture_id,
                "capture_timestamp": timestamp, "candidate_count": 2,
                "source_detection_index": source_index,
            },
            "assignment": assignment,
        })
    owner._rknn_pipeline = SimpleNamespace(
        tracker=SimpleNamespace(last_identity_observations=observations)
    )
    return [
        _record(track=3, uid=0, bbox=observations[0]["detector_bbox"]),
        _record(track=4, uid=0, bbox=observations[1]["detector_bbox"]),
    ]


def test_multi_person_strong_mapped_crop_gets_only_yaw_on_second_new_capture(owner, monkeypatch):
    _prepare_observation(owner)
    first_records = _multi_person_capture(
        owner, capture_id=231, timestamp=NOW - 0.08, competitor_distance=0.37
    )
    owner._consume_track_records(first_records, 640, 480, "test")
    assert owner._events == [
        ("queue", [runtime.ACTION_STOP], "search_candidate_evidence_observe")
    ]
    owner._events.clear()
    owner.frame_index += 1
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW + 0.10)
    second_records = _multi_person_capture(
        owner, capture_id=235, timestamp=NOW + 0.02, competitor_distance=0.37
    )
    owner._consume_track_records(second_records, 640, 480, "test")
    assert [event[0] for event in owner._events] == ["limited_yaw"]
    assert owner._events[0][1]["track_id"] == 3
    assert owner._events[0][1]["reid_uid"] == 7
    assert "publish" not in owner._context_events
    assert not owner._search_evidence_pause_current_frame
    assert owner._follow_controller.active_target_id == 7


@pytest.mark.parametrize("competitor_distance", [0.10, None])
def test_multi_person_ambiguous_or_unknown_competitor_never_gets_limited_yaw(owner, monkeypatch, competitor_distance):
    _prepare_observation(owner)
    for offset, capture_id in ((0.0, 231), (0.10, 235)):
        monkeypatch.setattr(runtime.time, "monotonic", lambda offset=offset: NOW + offset)
        records = _multi_person_capture(
            owner, capture_id=capture_id, timestamp=NOW - 0.08 + offset,
            competitor_distance=competitor_distance,
        )
        owner._consume_track_records(records, 640, 480, "test")
        owner.frame_index += 1
    assert owner._events == [
        ("queue", [runtime.ACTION_STOP], "search_candidate_evidence_observe"),
        ("queue", [runtime.ACTION_STOP], "search_candidate_evidence_observe"),
    ]
    assert "publish" not in owner._context_events


@pytest.mark.parametrize("case", ["unassigned", "other_uid", "fragment", "no_person"])
def test_unconfirmed_frame_commits_exactly_one_observation_stop(owner, case):
    _prepare_observation(owner)
    records = [_record(uid=0)]
    if case == "other_uid":
        records = [_record(uid=8)]
    elif case == "no_person":
        records = []
    elif case == "fragment":
        records = [_record(uid=0, bbox=(610.0, 30.0, 630.0, 52.0))]
        owner._assignments[3] = {
            "mapped_uid": 7, "best_uid": 7, "bbox_quality_ok": False,
            "bbox_quality_reason": "area<900,area_shrink<0.30", "distance": 0.01,
        }
    else:
        owner._assignments[3] = {"best_uid": 7, "distance": 0.01, "match_source": "strong"}
    owner._consume_track_records(records, 640, 480, "test")
    assert owner._events == [
        ("queue", [runtime.ACTION_STOP], "search_candidate_evidence_observe")
    ]
    assert "publish" not in owner._context_events


def test_hazard_wins_before_observation_or_active_target_control(owner):
    _prepare_observation(owner)

    def hazard(_state):
        owner._events.append(("hazard_stop",))
        return True

    owner._handle_hazard_safety_state = hazard
    owner._consume_track_records([_record()], 640, 480, "test")
    assert owner._events == [("hazard_stop",)]
    assert "publish" not in owner._context_events


def test_confirmed_reacquire_hold_is_not_preceded_by_generic_observation_stop(owner):
    _prepare_observation(owner)

    def reacquire_hold(*args, **kwargs):
        owner._events.append(("identity_confirmation_stop",))
        return True

    owner._hold_for_confirmed_search_reacquire = reacquire_hold
    owner._consume_track_records([_record()], 640, 480, "test")
    assert owner._events == [("identity_confirmation_stop",)]


def test_observation_zero_clears_visible_intent_without_recursive_zero_publish(owner):
    owner.search_state = "none"
    owner._vision_control_state = "target_visible_depth_valid"
    owner._lateral_intent_store.publish(LateralControlIntent(
        sequence=0, target_id=7, frame_index=100, published_at=NOW,
        valid_until=NOW + 0.15, x_ratio=0.7, motion_dx_ratio=0.01,
        target_image_rate_dps=8.0, mode="yaw_only", base_percent=0, base_rpm=0,
        initial_correction_rpm=5, correction_limit_rpm=5.0, confidence=0.9,
        bbox_quality="reliable", reason="visible",
    ))

    def forbidden_clear(reason):
        raise AssertionError("observation cannot recursively publish lateral_zero")

    owner._clear_lateral_intent = forbidden_clear
    owner._publish_observation_soft_zero("search_candidate_evidence_observe", use_stop_action=True)
    assert owner._lateral_intent_store.snapshot() is None
    assert owner._current_steer_correction_rpm == 0
    assert owner._events == [
        ("queue", [runtime.ACTION_STOP], "search_candidate_evidence_observe")
    ]


def test_release_observation_changes_no_target_uid_or_search_direction(owner):
    _prepare_observation(owner)
    owner._release_search_observation_for_control("active_uid_lateral")
    assert owner._follow_controller.active_target_id == 7
    assert owner.search_direction == "left"
    assert owner.search_state == "searching"
    assert not owner._search_evidence_observation_active
    assert not owner._search_evidence_pause_current_frame
    assert owner._events == []
