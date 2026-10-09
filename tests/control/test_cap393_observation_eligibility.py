"""CAP393 side crop: no pointless STOP; genuine settled-look rules unchanged."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.search_candidate_gate import (
    CandidateObservation, SearchCandidateGate, SearchCandidateGateConfig,
    SearchCandidateGateDecision,
)
from car_control_modular.search_observation_retry import (
    DEFERRED_EDGE_COVERAGE, DetectorObservationSettlement,
)
from test_search_observation_arbitration import owner


EDGE = (0., 123.48301696777344, 55.38147735595703, 455.72705078125)
CENTER = (220., 30., 410., 460.)


def evidence(owner, cap=393, stamp=10., bbox=EDGE):
    metadata = dict(is_fresh=True, capture_frame_id=cap, capture_timestamp=stamp,
                    control_frame_id=cap, source_detection_index=0,
                    quality_bbox_ok=False, bbox_quality_tier="weak")
    assignment = dict(mapped_uid=1, reason="secondary_evidence_unavailable",
        reacquire_partial_comparable=False, reacquire_partial_state="unknown",
        identity_control_rejected=True,
        identity_competition=dict(uid=1, frame_index=cap, source_detection_index=0,
                                 passed=True, candidate_count=1),
        template_recent_evidence=dict(count=8, distance=.313677, comparable_count=0,
                                      query_coverage="top0_bottom0_side1"),
        template_recent_partial_evidence=dict(count=8, distance=.373281, comparable_count=0,
                                              query_coverage="top0_bottom0_side1"))
    observation = dict(raw_track_id=1, uid=0, detector_bbox=bbox, sample_metadata=metadata)
    owner._assignments = {1: assignment}
    owner._rknn_pipeline = SimpleNamespace(tracker=SimpleNamespace(last_identity_observations=[observation]))
    return assignment, metadata, observation


@pytest.fixture
def configured(owner, monkeypatch):
    owner._follow_controller.active_target_id = 1
    owner._search_candidate_gate = SearchCandidateGate(SearchCandidateGateConfig(hold_frames=2))
    monkeypatch.setattr(runtime, "SEARCH_EVIDENCE_GATE_ENABLE", True)
    monkeypatch.setattr(runtime, "SEARCH_EVIDENCE_RETRY_ENABLE", True)
    return owner


def deferred(owner, cap, stamp, bbox=EDGE, candidates=None):
    return owner._deferred_search_observation_bboxes(
        search_active=True, width=640, height=480,
        formal_candidates=candidates or (CandidateObservation(bbox, .738),),
        capture_id=cap, capture_timestamp=stamp)


def gate_tick(owner, cap, stamp, bbox=EDGE):
    formal = (CandidateObservation(bbox, .738),)
    decision = owner._search_candidate_gate.update(
        timestamp=stamp, search_active=True, width=640, height=480, formal_candidates=formal,
        deferred_observation_bboxes=deferred(owner, cap, stamp, bbox))
    return owner._retry_search_candidate_observation(
        decision, search_active=True, width=640, height=480, formal_candidates=formal,
        capture_id=cap, capture_timestamp=stamp)


@pytest.mark.parametrize("mirror", [False, True])
def test_cap393_exact_crop_defers_only_observation_stop(configured, monkeypatch, mirror):
    o = configured
    bbox = (640-EDGE[2], EDGE[1], 640-EDGE[0], EDGE[3]) if mirror else EDGE
    evidence(o, stamp=29820.345385640, bbox=bbox)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 29820.423806)
    decision = gate_tick(o, 393, 29820.345385640, bbox)
    assert decision.reason == DEFERRED_EDGE_COVERAGE
    assert not decision.entered and not decision.pause_rotation and not decision.completed
    assert not decision.preferred_target_match
    assert not o._search_observation_retry.spent
    assert not o._search_detector_settlement.active
    assert getattr(o, "_search_retry_zero_requested_at", 0) == 0
    assert o._events == [] and o.search_direction == "left"
    assert o._follow_controller.active_target_id == 1
    assert o._assignments[1]["identity_control_rejected"] is True


@pytest.mark.parametrize("case", [
    "old_capture", "old_timestamp", "stale", "duplicate_observations", "other_bbox",
    "missing_coverage", "missing_count", "comparable", "partial_comparable",
    "unknown_partial_state", "confirmed_uid", "wrong_uid", "wrong_source",
    "old_competition", "multi_competition", "failed_competition", "other_candidate",
    "different_reason", "central", "not_touching_edge", "nan",
])
def test_only_explicit_current_unique_incomparable_edge_is_deferred(configured, monkeypatch, case):
    o = configured
    a, m, obs = evidence(o)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 10.08)
    candidates = (CandidateObservation(EDGE, .738),)
    bbox = EDGE
    if case == "old_capture": m["capture_frame_id"] = 392
    elif case == "old_timestamp": m["capture_timestamp"] = 9.99
    elif case == "stale": m["is_fresh"] = False
    elif case == "duplicate_observations": o._rknn_pipeline.tracker.last_identity_observations *= 2
    elif case == "other_bbox": obs["detector_bbox"] = CENTER
    elif case == "missing_coverage": del a["template_recent_partial_evidence"]["query_coverage"]
    elif case == "missing_count": del a["template_recent_evidence"]["comparable_count"]
    elif case == "comparable": a["template_recent_evidence"]["comparable_count"] = 1
    elif case == "partial_comparable": a["reacquire_partial_comparable"] = True
    elif case == "unknown_partial_state": del a["reacquire_partial_state"]
    elif case == "confirmed_uid": obs["uid"] = 1
    elif case == "wrong_uid": a["identity_competition"]["uid"] = 2
    elif case == "wrong_source": a["identity_competition"]["source_detection_index"] = 1
    elif case == "old_competition": a["identity_competition"]["frame_index"] = 392
    elif case == "multi_competition": a["identity_competition"]["candidate_count"] = 2
    elif case == "failed_competition": a["identity_competition"]["passed"] = False
    elif case == "other_candidate": candidates += (CandidateObservation(CENTER, .9),)
    elif case == "different_reason": a["reason"] = "recent_partial_conflict"
    elif case in ("central", "not_touching_edge"):
        bbox = CENTER if case == "central" else (10., 123., 65., 455.)
        obs["detector_bbox"] = bbox
        candidates = (CandidateObservation(bbox, .738),)
    elif case == "nan": obs["detector_bbox"] = (float("nan"), 10., 60., 400.)
    assert deferred(o, 393, 10., bbox, candidates) == ()


def test_coverage_improves_after_defer_can_still_get_one_settled_look(configured, monkeypatch):
    o = configured
    for cap, stamp, width in [(393, 10., 55.381477), (394, 10.05, 59.699886),
                              (396, 10.10, 79.238152), (397, 10.15, 83.730392),
                              (398, 10.20, 83.569046), (400, 10.25, 84.916618)]:
        bbox = (0., EDGE[1], width, EDGE[3])
        evidence(o, cap, stamp, bbox)
        monkeypatch.setattr(runtime.time, "monotonic", lambda stamp=stamp: stamp+.08)
        assert gate_tick(o, cap, stamp, bbox).reason == DEFERRED_EDGE_COVERAGE
        assert not o._search_observation_retry.spent
    evidence(o, 404, 10.4, CENTER)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 10.48)
    d = gate_tick(o, 404, 10.4, CENTER)
    assert d.entered and d.pause_rotation and not d.completed
    assert o._search_observation_retry.spent
    deadline = o._search_detector_settlement.deadline
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 10.50)
    d = gate_tick(o, 404, 10.4, CENTER)
    assert d.pause_rotation and not d.entered and not d.completed
    assert o._search_detector_settlement.deadline == deadline
    assert o._search_candidate_gate._hold_remaining == 1
    o._search_retry_zero_sent_at = 10.49
    fb = SimpleNamespace(timestamp=10.52, trustworthy=True, left_forward_rpm=0., right_forward_rpm=0.)
    o._action_runtime = SimpleNamespace(get_steering_feedback=lambda: fb)
    for cap, stamp, now, feedback_stamp in [(405, 10.54, 10.59, 10.52),
                                           (406, 10.58, 10.64, 10.60),
                                           (407, 10.65, 10.70, 10.60)]:
        evidence(o, cap, stamp, CENTER)
        fb.timestamp = feedback_stamp
        monkeypatch.setattr(runtime.time, "monotonic", lambda now=now: now)
        d = gate_tick(o, cap, stamp, CENTER)
        assert d.completed is (cap == 407)
    assert d.reason == "observation_settled_capture" and not d.preferred_target_match
    evidence(o, 408, 10.71, CENTER)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 10.76)
    d = gate_tick(o, 408, 10.71, CENTER)
    assert not d.entered and not d.pause_rotation
    assert o._search_observation_retry.spent


def test_central_unknown_still_gets_original_bounded_stop(configured, monkeypatch):
    evidence(configured, bbox=CENTER)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 10.08)
    d = gate_tick(configured, 393, 10., CENTER)
    assert d.entered and d.pause_rotation and not d.completed


def test_defer_active_detector_look_is_not_settled_or_identity_confirmation():
    s = DetectorObservationSettlement()
    seed = SearchCandidateGateDecision(entered=True, pause_rotation=True, bbox=EDGE)
    args = dict(search_active=True, zero_sent_at=None, feedback=None, max_hold_sec=.3)
    s.update(seed, now=10., capture_timestamp=9.98, capture_id=393, **args)
    d = s.update(replace(seed, entered=False, reason=DEFERRED_EDGE_COVERAGE),
                 now=10.05, capture_timestamp=10.03, capture_id=394, **args)
    assert d.completed and not d.pause_rotation and not d.preferred_target_match
    assert d.reason == DEFERRED_EDGE_COVERAGE and not s.active


def test_duplicate_and_old_capture_never_consume_or_rearm_gate():
    g = SearchCandidateGate(SearchCandidateGateConfig(hold_frames=2, blocked_reset_missing_frames=2))
    args = dict(search_active=True, width=640, height=480)
    formal = (CandidateObservation(CENTER, .9),)
    assert g.update(timestamp=10., formal_candidates=formal, **args).entered
    for t in (10., 9.99, float("nan")):
        d = g.update(timestamp=t, formal_candidates=formal, **args)
        assert not d.entered and not d.completed and g._hold_remaining == 1
    assert g.update(timestamp=10.05, formal_candidates=formal, **args).completed
    for _ in range(8):
        assert not g.update(timestamp=10.05, **args).entered
        assert g._blocked_missing_frames == 0
    assert not g.update(timestamp=10.1, formal_candidates=formal, **args).entered


def test_preferred_current_identity_support_is_not_filtered():
    g = SearchCandidateGate(SearchCandidateGateConfig())
    d = g.update(timestamp=10., search_active=True, width=640, height=480,
                 formal_candidates=(CandidateObservation(EDGE, .738),), preferred_bbox=EDGE,
                 deferred_observation_bboxes=(EDGE,))
    assert d.entered and d.pause_rotation
