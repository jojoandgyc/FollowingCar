"""A co-visible different person is excluded locally, not the whole search.

Reuse the hardware-free owner fixture; the production evidence joins, gate,
track consumption and observation withdrawal are exercised without motor I/O.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_search_observation_arbitration import _record, owner
import request_0513_modular as runtime
from car_control_modular.controllers import FollowSafetyController, LateralCandidateEvidence
from car_control_modular.search_candidate_gate import (
    CandidateObservation, SearchCandidateGate, SearchCandidateGateConfig,
)


LEFT = (20.0, 20.0, 200.0, 460.0)
RIGHT = (400.0, 30.0, 600.0, 470.0)


@pytest.fixture
def scene(owner):
    exclusions = {}
    calls = []

    def query(raw_track_id, uid, *, frame_index, capture_timestamp=None):
        calls.append((raw_track_id, uid, frame_index, capture_timestamp))
        return exclusions.get((raw_track_id, uid))

    bank = SimpleNamespace(last_assignments={}, search_exclusion_for=query)
    owner._rknn_pipeline = SimpleNamespace(tracker=SimpleNamespace(
        identity_bank=bank, _frame_index=owner.frame_index,
        last_identity_observations=[],
    ))
    owner._search_candidate_gate = SearchCandidateGate(SearchCandidateGateConfig(hold_frames=2))
    owner._exclusions = exclusions
    owner._query_calls = calls
    return owner


def observe(scene, raw=3, bbox=LEFT, *, uid=0, **metadata):
    sample = dict(is_fresh=True, capture_frame_id=scene._active_capture_frame_id,
                  capture_timestamp=scene._active_capture_timestamp)
    sample.update(metadata)
    observation = dict(raw_track_id=raw, detector_bbox=bbox, uid=uid, sample_metadata=sample)
    scene._rknn_pipeline.tracker.last_identity_observations.append(observation)
    return observation


def exclude(scene, raw=3, uid=7):
    scene._exclusions[raw, uid] = dict(
        reason="co_visible_distinct_person", source_capture_frame_id=199,
        reference_track_id=2, candidate_track_id=raw,
    )


def filtered(scene, candidates, source="formal"):
    return scene._filter_search_excluded_evidence(
        tuple(candidates), scene._search_exclusion_bindings(), source=source,
    )


@pytest.mark.parametrize("source", ["formal", "probe"])
def test_exact_candidate_filtered_other_unknown_and_raw_inputs_retained(scene, source, caplog):
    caplog.set_level("INFO")
    observe(scene)
    observe(scene, 4, RIGHT)
    exclude(scene)
    candidates = [CandidateObservation(LEFT, 0.9), CandidateObservation(RIGHT, 0.8)]
    assert filtered(scene, candidates, source) == (candidates[1],)
    assert len(candidates) == 2
    assert len(scene._rknn_pipeline.tracker.last_identity_observations) == 2
    assert "search_candidate_excluded" in caplog.text
    assert "reference_track_id=2" in caplog.text
    assert scene.search_direction == "left"
    assert scene._events == []


def test_exclusion_is_uid_specific_and_query_uses_capture_clock(scene):
    observe(scene)
    exclude(scene, uid=8)
    candidate = CandidateObservation(LEFT, 0.9)
    assert filtered(scene, [candidate]) == (candidate,)
    assert scene._query_calls[-1] == (3, 7, scene.frame_index, scene._active_capture_timestamp)


def test_expired_query_does_not_revive_last_assignment_exclusion(scene):
    observe(scene)
    scene._rknn_pipeline.tracker.identity_bank.last_assignments[3] = dict(
        search_excluded=True, excluded_uid=7,
        search_exclusion={"reason": "co_visible_distinct_person"},
    )
    candidate = CandidateObservation(LEFT, 0.9)
    assert filtered(scene, [candidate]) == (candidate,)


@pytest.mark.parametrize("metadata", [
    {"is_fresh": False}, {"capture_frame_id": 230}, {"capture_timestamp": 190.0},
])
def test_stale_observation_cannot_attach_exclusion_to_current_box(scene, metadata):
    observe(scene, **metadata)
    exclude(scene)
    candidate = CandidateObservation(LEFT, 0.9)
    assert filtered(scene, [candidate]) == (candidate,)
    assert scene._search_excluded_tracks() == {}


def test_old_track_exclusion_does_not_apply_to_new_track_in_same_box(scene):
    observe(scene, raw=4)
    exclude(scene, raw=3)
    candidate = CandidateObservation(LEFT, 0.9)
    assert filtered(scene, [candidate]) == (candidate,)


def test_unassociated_or_ambiguous_probe_is_not_removed(scene):
    observe(scene)
    exclude(scene)
    probe = CandidateObservation((25.0, 200.0, 90.0, 450.0), 0.14)
    assert filtered(scene, [probe], "probe") == (probe,)
    observe(scene, 4, LEFT)
    full = CandidateObservation(LEFT, 0.9)
    assert filtered(scene, [full]) == (full,)
    assert scene._search_excluded_tracks() == {}


def test_duplicate_raw_track_for_two_detections_is_not_a_reliable_binding(scene):
    observe(scene)
    observe(scene, bbox=RIGHT)
    exclude(scene)
    candidates = [CandidateObservation(LEFT, .9), CandidateObservation(RIGHT, .8)]
    assert filtered(scene, candidates) == tuple(candidates)
    assert scene._search_excluded_tracks() == {}


def begin_observation(scene, bbox=LEFT):
    decision = scene._search_candidate_gate.update(
        timestamp=scene._active_capture_timestamp, search_active=True, width=640, height=480,
        formal_candidates=(CandidateObservation(bbox, 0.9),),
    )
    scene._apply_search_candidate_gate_decision(decision, prepare_only=True)
    scene._search_evidence_pause_current_frame = True
    scene._follow_controller.set_search_observation_hold(True)


def test_excluded_observation_released_other_candidate_keeps_its_own_budget(scene):
    observe(scene)
    observe(scene, 4, RIGHT)
    begin_observation(scene)
    assert scene._search_evidence_observation_track_id == 3
    scene._search_candidate_gate._probe_bbox = RIGHT
    scene._search_candidate_gate._probe_streak = 1
    exclude(scene)
    assert scene._cancel_excluded_search_observation(scene._search_exclusion_bindings())
    assert not scene._search_evidence_observation_active
    assert not scene._search_evidence_pause_current_frame
    assert not scene._follow_controller._search_observation_hold
    assert scene._search_candidate_gate._hold_remaining == 0
    assert scene._search_candidate_gate._probe_bbox == RIGHT
    assert scene._search_candidate_gate._probe_streak == 1
    assert scene.search_direction == scene._follow_controller.search_direction == "left"
    assert scene._events == []
    next_decision = scene._search_candidate_gate.update(
        timestamp=scene._active_capture_timestamp + .07, search_active=True, width=640, height=480,
        formal_candidates=filtered(scene, [CandidateObservation(LEFT, .9), CandidateObservation(RIGHT, .8)]),
    )
    assert next_decision.entered and next_decision.hold_frame == 1
    assert next_decision.bbox == RIGHT


def test_other_track_observation_not_cancelled(scene):
    observe(scene)
    observe(scene, 4, RIGHT)
    begin_observation(scene, RIGHT)
    exclude(scene)
    assert not scene._cancel_excluded_search_observation(scene._search_exclusion_bindings())
    assert scene._search_evidence_observation_active
    assert scene._search_candidate_gate._hold_remaining == 1
    assert scene._search_evidence_observation_track_id == 4


def test_excluded_last_noted_candidate_cannot_retain_missing_centering_hold(scene):
    observe(scene)
    exclude(scene)
    controller = scene._follow_controller
    controller._stale_direction_recovery_active = False
    controller._stale_direction_recovery_stage = "candidate_centering"
    controller._lost_exit_direction = "left"
    controller._reset_stale_direction_recovery = lambda reason: (
        FollowSafetyController._reset_stale_direction_recovery(controller, reason)
    )
    scene._search_last_noted_candidate_track_id = 3
    scene._cancel_excluded_search_observation(scene._search_exclusion_bindings())
    assert controller._stale_direction_recovery_stage == "none"
    assert scene._search_last_noted_candidate_track_id is None
    assert controller.search_direction == controller._lost_exit_direction == "left"
    assert scene._events == []


def test_even_mapped_excluded_person_cannot_be_preferred_active_bbox(scene):
    observe(scene, uid=7)
    exclude(scene)
    scene._assignments[3] = {"mapped_uid": 7, "bbox_quality_ok": True}
    assert scene._active_target_track_record([_record()], 7) == (None, {})


@pytest.mark.parametrize("low_quality", [False, True])
def test_consume_excludes_target_and_lateral_hint_but_retains_hazard_detection(scene, low_quality):
    observe(scene, uid=7)
    exclude(scene)
    scene._assignments[3] = {
        "mapped_uid": 7, "bbox_quality_ok": not low_quality,
        "bbox_quality_reason": "edge_touch>2" if low_quality else "ok",
        "bbox_quality_tier": "weak" if low_quality else "strong",
    }
    hazard_detections = []
    scene._bunker_runtime.check_merged_dets = lambda detections, *_args: hazard_detections.extend(detections)
    rec = _record()
    hint = LateralCandidateEvidence(
        capture_frame_id=scene._active_capture_frame_id, bbox=LEFT, score=.9,
        source="formal", active_target_match=True,
    )
    scene._consume_track_records([rec], 640, 480, "test", lateral_candidate=hint)
    assert hazard_detections == [{"class_id": 0, "score": rec.score, "bbox": LEFT}]
    assert all(event[0] != "limited_yaw" for event in scene._events)
    normal = [event for event in scene._events if event[0] == "normal"]
    assert len(normal) == 1 and normal[0][1] == []
    assert normal[0][2]["lateral_candidate"] is None
    assert "publish" not in scene._context_events


def test_distinct_other_person_does_not_block_correct_active_person(scene):
    observe(scene, uid=0)
    observe(scene, 4, RIGHT, uid=7)
    exclude(scene)
    scene._assignments[3] = {"bbox_quality_ok": True}
    scene._assignments[4] = {"bbox_quality_ok": True, "mapped_uid": 7}
    scene._consume_track_records([_record(uid=0), _record(track=4, uid=7, bbox=RIGHT)], 640, 480, "test")
    normal = [event for event in scene._events if event[0] == "normal"]
    assert len(normal) == 1 and len(normal[0][1]) == 1
    assert normal[0][1][0][0] == RIGHT
