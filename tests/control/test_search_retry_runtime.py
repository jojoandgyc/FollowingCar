"""Retry integration uses current detector/ReID provenance, never real motors."""
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.search_candidate_gate import CandidateObservation, SearchCandidateGate, SearchCandidateGateConfig, SearchCandidateGateDecision
from test_search_observation_arbitration import owner, _record, NOW


BOX = (1., 2., 365., 471.)


@pytest.fixture(autouse=True)
def enable_retry(monkeypatch):
    monkeypatch.setattr(runtime, "SEARCH_EVIDENCE_GATE_ENABLE", True)
    monkeypatch.setattr(runtime, "SEARCH_EVIDENCE_RETRY_ENABLE", True)


def evidence(owner, cap, stamp, *, distance=.21, source="strong", uid=7, bbox=BOX):
    owner._active_capture_frame_id = cap
    owner._active_capture_timestamp = stamp
    owner._assignments[3] = dict(
        best_uid=uid, distance=distance, match_source=source,
        bbox_quality_ok=False, bbox_quality_reason="edge_touch>2",
    )
    owner._rknn_pipeline = SimpleNamespace(tracker=SimpleNamespace(last_identity_observations=[dict(
        raw_track_id=3, detector_bbox=bbox,
        sample_metadata=dict(is_fresh=True, capture_frame_id=cap, capture_timestamp=stamp),
    )]))
    owner._search_candidate_gate = SearchCandidateGate(SearchCandidateGateConfig())


def tick(owner, cap=643, stamp=NOW-.09, **kwargs):
    return owner._retry_search_candidate_observation(
        SearchCandidateGateDecision(source="blocked", reason="candidate_already_observed", bbox=BOX, score=.84),
        search_active=True, width=640, height=480,
        formal_candidates=kwargs.pop("candidates", (CandidateObservation(BOX, .84),)),
        capture_id=cap, capture_timestamp=stamp, **kwargs,
    )


def start(owner, monkeypatch):
    evidence(owner, 643, NOW-.09)
    assert not tick(owner).pause_rotation
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.1)
    evidence(owner, 646, NOW+.01)
    result = tick(owner, 646, NOW+.01)
    assert result.entered and not result.completed
    return result


def test_same_credible_cropped_target_retries_without_claiming_uid_or_direction(owner, monkeypatch):
    result = start(owner, monkeypatch)
    assert result.source == "credible_retry" and not result.preferred_target_match
    assert owner._search_retry_zero_requested_at == NOW+.1
    assert owner._search_retry_zero_sent_at is None
    assert owner._follow_controller.active_target_id == 7
    assert owner.search_direction == "left"
    assert owner._events == []  # provisional gate has no motor side effects


@pytest.mark.parametrize("case", ["partial", "wrong_uid", "high_distance", "unknown", "stale_metadata", "excluded", "two_people", "fragment"])
def test_retry_does_not_weaken_identity_or_geometry_requirements(owner, monkeypatch, case):
    for offset, cap in [(0., 643), (.1, 646)]:
        monkeypatch.setattr(runtime.time, "monotonic", lambda offset=offset: NOW+offset)
        stamp = NOW+offset-.09
        evidence(owner, cap, stamp)
        candidates = (CandidateObservation(BOX, .84),)
        if case == "partial": owner._assignments[3]["match_source"] = "partial"
        if case == "wrong_uid": owner._assignments[3]["best_uid"] = 9
        if case == "high_distance": owner._assignments[3]["distance"] = .31
        if case == "unknown": owner._assignments.clear()
        if case == "excluded": owner._assignments[3]["search_excluded"] = True
        if case == "stale_metadata": owner._rknn_pipeline.tracker.last_identity_observations[0]["sample_metadata"]["capture_frame_id"] = 1
        if case == "two_people": candidates += (CandidateObservation((430., 10., 639., 470.), .9),)
        if case == "fragment": candidates = (CandidateObservation((1., 2., 12., 16.), .84),)
        result = tick(owner, cap, stamp, candidates=candidates)
        assert not result.pause_rotation
    assert not owner._search_observation_retry.spent


def test_retry_owns_one_zero_before_mapped_low_quality_yaw(owner, monkeypatch):
    result = start(owner, monkeypatch)
    owner._apply_search_candidate_gate_decision(result, prepare_only=True)
    owner._search_evidence_pause_current_frame = True
    owner._assignments[3].update(mapped_uid=7, reason="mapped_weak_observed")
    owner._consume_track_records([_record(uid=0, bbox=BOX)], 640, 480, "test")
    assert owner._events == [("queue", [runtime.ACTION_STOP], "search_candidate_retry_observe")]
    assert owner._current_forward_percent == owner._current_rotate_raw_target == 0
    assert owner._use_soft_stop_next
    assert owner._search_observation_retry.active


def test_hazard_still_preempts_retry(owner, monkeypatch):
    result = start(owner, monkeypatch)
    owner._apply_search_candidate_gate_decision(result, prepare_only=True)
    owner._search_evidence_pause_current_frame = True
    owner._handle_hazard_safety_state = lambda _: owner._events.append(("hazard",)) or True
    owner._consume_track_records([_record(uid=0, bbox=BOX)], 640, 480, "test")
    assert owner._events == [("hazard",)]


def test_post_zero_frame_releases_retry_without_new_identity_permission(owner, monkeypatch):
    result = start(owner, monkeypatch)
    owner._apply_search_candidate_gate_decision(result, prepare_only=True)
    owner._search_retry_zero_sent_at = NOW+.11
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.3)
    evidence(owner, 648, NOW+.21)
    result = tick(owner, 648, NOW+.21)
    assert result.completed and not result.pause_rotation
    owner._apply_search_candidate_gate_decision(result, prepare_only=True)
    assert not owner._search_evidence_observation_active
    assert not owner._search_observation_retry.active
    assert owner._search_observation_retry.spent
    assert owner._events == []
    assert owner.search_direction == "left" and owner._follow_controller.active_target_id == 7


def test_disabled_retry_keeps_original_gate(owner, monkeypatch):
    monkeypatch.setattr(runtime, "SEARCH_EVIDENCE_RETRY_ENABLE", False)
    evidence(owner, 643, NOW-.09)
    assert tick(owner).reason == "candidate_already_observed"
    assert not owner._search_observation_retry.spent


def test_logged_641_to_643_box_growth_is_continuous_not_a_new_identity(owner, monkeypatch):
    first = (0.9679, 4.0148, 225.1993, 471.6675)
    evidence(owner, 641, NOW-.09, distance=.244, bbox=first)
    assert not tick(owner, 641, NOW-.09, candidates=(CandidateObservation(first, .332),)).pause_rotation
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.10)
    evidence(owner, 643, NOW+.009403, distance=.2076)
    result = tick(owner, 643, NOW+.009403)
    assert result.entered and result.source == "credible_retry"
    assert owner._follow_controller.active_target_id == 7
