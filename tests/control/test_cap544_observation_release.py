"""CAP533/542 shared pause and CAP544 bank-confirmed early return to control."""
from types import SimpleNamespace

import pytest
import request_0513_modular as runtime
from car_control_modular.search_candidate_gate import (
    SearchCandidateGateDecision, CandidateObservation,
)
from car_control_modular.search_observation_retry import confirmed_observation_for_release
from test_search_observation_arbitration import owner, NOW
from test_search_retry_runtime import evidence, tick, start, BOX, enable_retry


def confirmed(owner, cap=544, stamp=NOW+.11):
    # Relevant decision fields from the CAP544 event. UID translated to this
    # fixture's active UID7; no appearance threshold or detector confidence
    # itself is treated as proof of identity.
    evidence(owner, cap, stamp, source="partial")
    a = owner._assignments[3]
    a.update(uid=7, bbox_quality_ok=True, reason="preferred_search_late_reacquire",
        reacquire_geometry_ok=True,
        reacquire_geometry=dict(ok=True, reason="late_candidate_local_continuity"),
        reacquire_partial_state="match",
        reacquire_partial_confirm_limit=.4,
        reacquire_recent_partial_evidence=dict(count=1, distance=.23129327595233917),
        identity_competition=dict(uid=7, frame_index=252, source_detection_index=0,
                                  candidate_count=1, passed=True, distance=.21359187364578247))
    observation = owner._rknn_pipeline.tracker.last_identity_observations[0]
    observation["uid"] = 7
    m = observation["sample_metadata"]
    m.update(quality_bbox_ok=True, bbox_quality_tier="strong", control_frame_id=252,
             source_detection_index=0, partial_feature_source="osnet_torso")
    return a, m, observation


def ordinary(owner, cap=533, stamp=NOW-.09):
    evidence(owner, cap, stamp)
    return owner._retry_search_candidate_observation(
        SearchCandidateGateDecision(source="formal", entered=True, pause_rotation=True,
            reason="formal_candidate_observe_start", bbox=BOX, score=.949),
        search_active=True, width=640, height=480,
        formal_candidates=(CandidateObservation(BOX, .949),),
        capture_id=cap, capture_timestamp=stamp)


def test_ordinary_pause_and_credible_retry_share_budget(owner, monkeypatch):
    decision = ordinary(owner)
    assert decision.pause_rotation
    assert owner._search_observation_retry.spent
    assert not owner._search_observation_retry.active
    # The original observation still waits for stopped-image proof or timeout.
    for delta, cap in ((.1, 535), (.2, 537), (.31, 539), (.36, 540), (.40, 542)):
        monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+delta)
        evidence(owner, cap, NOW+delta-.09)
        decision = tick(owner, cap, NOW+delta-.09)
        assert not decision.entered
        assert decision.pause_rotation == (delta < .3)
        assert not owner._search_observation_retry.active
    assert owner._search_observation_retry.reason == "budget_spent"
    assert owner._events == []
    # A new search epoch may observe anew; a changed raw tracker ID may not.
    owner._search_epoch = 1
    for delta, cap in ((.5, 544), (.6, 546)):
        monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+delta)
        evidence(owner, cap, NOW+delta-.09)
        decision = tick(owner, cap, NOW+delta-.09)
    assert decision.entered and decision.pause_rotation


@pytest.mark.parametrize("mode", ["ordinary", "retry"])
def test_confirmed_identity_releases_either_pause_before_settled_capture(owner, monkeypatch, mode):
    if mode == "ordinary":
        decision = ordinary(owner)
        stamp, cap = NOW+.01, 544
    else:
        decision = start(owner, monkeypatch)
        stamp, cap = NOW+.11, 648
    owner._apply_search_candidate_gate_decision(decision, prepare_only=True)
    owner._search_evidence_pause_current_frame = True
    owner._follow_controller.set_search_observation_hold(True)
    confirmed(owner, cap, stamp)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: stamp+.09)
    # No all-wheel standstill and capture predates zero; bank confirmation
    # ends observation, NOT the writer's wheel reversal or safety guards.
    owner._search_retry_zero_sent_at = stamp+.01
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: SimpleNamespace(
        trustworthy=True, timestamp=stamp+.05, left_forward_rpm=8, right_forward_rpm=-6))
    result = tick(owner, cap, stamp)
    assert result.completed and not result.pause_rotation
    assert result.reason == "search_retry_identity_confirmed"
    assert not result.preferred_target_match
    owner._apply_search_candidate_gate_decision(result, prepare_only=True)
    assert not owner._search_evidence_observation_active
    assert not owner._search_detector_settlement.active
    assert not owner._search_observation_retry.active
    assert not owner._follow_controller._search_observation_hold
    assert owner._search_observation_retry.spent
    assert owner._search_observation_retry.last_capture == (cap, stamp)
    assert owner._events == []  # no queue, motors, UID or template mutation
    assert owner._follow_controller.active_target_id == 7 and owner.search_direction == "left"


@pytest.mark.parametrize("case", ["uid0", "wrong_uid", "assignment_uid", "stale",
    "weak", "competition", "proof_frame", "proof_detection", "geometry", "partial_conflict",
    "excluded", "identity_rejected", "retained_conflict", "duplicate", "old_capture",
    "wrong_capture", "other_person"])
def test_confirmed_release_cannot_be_inferred_from_low_distance(owner, monkeypatch, case):
    decision = ordinary(owner)
    owner._apply_search_candidate_gate_decision(decision, prepare_only=True)
    stamp, cap = NOW+.01, 544
    a, m, obs = confirmed(owner, cap, stamp)
    if case == "uid0": obs["uid"] = 0
    if case == "wrong_uid": obs["uid"] = 9
    if case == "assignment_uid": a["uid"] = 0
    if case == "stale": m["is_fresh"] = False
    if case == "weak": m["quality_bbox_ok"] = False
    if case == "competition": a["identity_competition"]["passed"] = False
    if case == "proof_frame": a["identity_competition"]["frame_index"] = 251
    if case == "proof_detection": a["identity_competition"]["source_detection_index"] = 1
    if case == "geometry": a["reacquire_geometry_ok"] = False
    if case == "partial_conflict": a["reason"] = "recent_partial_conflict"
    if case == "excluded": a["search_excluded"] = True
    if case == "identity_rejected": a["identity_control_rejected"] = True
    if case == "retained_conflict": a["reacquire_geometry"]["search_contradiction_retained"] = True
    if case == "duplicate": owner._search_observation_retry.last_capture = (cap, stamp)
    if case == "old_capture": owner._search_observation_retry.last_capture = (cap+1, stamp+.01)
    if case == "wrong_capture": m["capture_frame_id"] = cap-1
    candidates = (CandidateObservation(BOX, .84),)
    if case == "other_person": candidates += (CandidateObservation((430., 10., 639., 470.), .9),)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.1)
    result = tick(owner, cap, stamp, candidates=candidates)
    assert result.pause_rotation and not result.completed
    assert owner._search_detector_settlement.active
    assert owner._events == []


def test_confirmed_release_rejects_aged_capture_even_if_metadata_says_fresh(owner, monkeypatch):
    ordinary(owner)
    confirmed(owner, 544, NOW-.11)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.1)
    assert tick(owner, 544, NOW-.11).pause_rotation


def test_first_frame_confirmation_preserves_capture_watermark(owner):
    confirmed(owner, 544, NOW-.09)
    decision = owner._retry_search_candidate_observation(
        SearchCandidateGateDecision(source="formal", entered=True, pause_rotation=True,
                                    bbox=BOX, score=.9),
        search_active=True, width=640, height=480,
        formal_candidates=(CandidateObservation(BOX, .9),), capture_id=544,
        capture_timestamp=NOW-.09)
    assert decision.completed
    assert owner._search_observation_retry.last_capture == (544, NOW-.09)


def test_raw_distance_never_substitutes_for_actual_uid(owner):
    a, m, obs = confirmed(owner)
    a["distance"] = .01
    assert confirmed_observation_for_release(a, m, 7, 7)
    assert not confirmed_observation_for_release(a, m, 0, 7)
