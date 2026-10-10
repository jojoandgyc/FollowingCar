"""Real detector/pipeline/tracker/bank; synthetic embeddings, no hardware.

CAP1490..1522 used mapped_similar_follow on every frame but never qualified
for detector continuation. These tests separate following from learning and
exercise actual extractor call counts at the recorded ~200 ms cadence.
"""
from dataclasses import replace
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from car_control_modular.detector_identity_lease import from_assignment, read_visual_identity_evidence
from car_control_modular.depth_target_geometry import resolve_depth_target_observation
from rk_vision.detector_continuation import FULL_PROOF_TTL_SEC
from rk_vision.pipeline import _detector_color_features
from rk_vision.yolo11 import Detection
from test_detector_continuation_pipeline import fast_pipeline
from test_provisional_association import activated, BOX
from test_reacquire_crosscheck import feature


@pytest.fixture
def similar_pipeline(fast_pipeline):
    return make_similar_pipeline(fast_pipeline)


def make_similar_pipeline(case, *, seed_box=BOX, current_box=BOX):
    p = case.pipeline
    p.config = replace(p.config, identity_similar_follow_enable=True)
    p.tracker, _ = activated(box=seed_box)
    p.tracker.set_search_reacquire_context(active_uid=1, searching=False, direction=None)
    p.set_detector_continuation_context(active_uid=1, allowed=True)
    p.detector.detections = [Detection(current_box, .95, 0)]
    case.distance = .20
    def extract(packet, persons, fmt):
        p.reid.calls += 1
        p.reid.last_partial_features = [None for _ in persons]
        p.reid.last_partial_feature_sources = [None for _ in persons]
        p.reid.last_color_features = _detector_color_features(packet, persons, fmt)
        p.reid.last_timing_ms = {"total": 35., "inference": 15., "features": len(persons)}
        return [feature(case.distance) for _ in persons]
    p.reid.extract = extract
    records = case.step(22, timestamp=3.2)
    assert [(r.track_id, r.reid_uid) for r in records] == [(3, 1)]
    assignment = p.tracker.identity_bank.last_assignments[3]
    assert assignment["reason"] == "mapped_similar_follow"
    assert assignment["template_update_quarantined"]
    assert p.tracker._detector_proof.permission == "similar_follow"
    assert p.tracker._detector_proof.full_count == 2
    return case


@pytest.mark.parametrize("gap", [.05, .15, .199, .21, .249])
def test_similar_follow_skips_reid_without_advancing_bank_or_deadline(similar_pipeline, gap):
    s = similar_pipeline; p = s.pipeline; bank = p.tracker.identity_bank
    before = pickle.dumps(bank)
    metric = pickle.dumps(p.tracker.deepsort.tracker.metric)
    proof = p.tracker._detector_proof
    records = s.step(23, timestamp=3.2+gap)
    assert p.last_identity_processing["mode"] == "detector_continuation"
    assert p.last_identity_processing["permission"] == "similar_follow"
    assert p.detector.calls == 2 and p.reid.calls == 1
    assert p.last_timing_ms["reid_total"] == 0.
    assert [(r.track_id, r.reid_uid) for r in records] == [(3, 1)]
    assignment = p.tracker.control_assignment_for_track(3)
    assert assignment["identity_permission"] == "similar_follow"
    assert assignment["template_update_quarantined"] and not assignment["learning_allowed"]
    assert not assignment["bank_updated"] and not assignment["initial_identity_confirmed"]
    assert assignment["identity_valid_until"] == proof.deadline
    assert assignment["identity_verified_capture"] == 22
    assert pickle.dumps(bank) == before
    assert pickle.dumps(p.tracker.deepsort.tracker.metric) == metric
    assert p.tracker._detector_proof.verified is proof.verified


def test_periodic_full_after_fast_keeps_real_similar_state_and_frozen_gallery(similar_pipeline):
    s = similar_pipeline; p = s.pipeline; bank = p.tracker.identity_bank
    for cap, stamp, mode, calls in ((23, 3.399, "detector_continuation", 1),
                                   (24, 3.59, "full", 2),
                                   (25, 3.789, "detector_continuation", 2),
                                   (26, 3.98, "full", 3)):
        records = s.step(cap, timestamp=stamp, result_age=.10)
        assert [(r.track_id, r.reid_uid) for r in records] == [(3, 1)]
        assert p.last_identity_processing["mode"] == mode
        assert p.reid.calls == calls
        assert bank._reacquire_quarantine.is_held(1)
        assert bank._similar_follow_states[(1, 3)]["last_cap"] == (cap if mode == "full" else cap-1)
        assert not bank.last_assignments[3]["bank_updated"]
    assert p.detector.calls == 5 and p.tracker._frame_index == 26


@pytest.mark.parametrize("fast_stamp, full_stamp, result_age", [
    (3.449, 3.748, .015), (3.399, 3.749, .10), (3.399, 3.799, .10)])
def test_variable_cadence_full_uses_measured_fast_position_not_retimed_embedding(
        similar_pipeline, fast_stamp, full_stamp, result_age):
    from tools.replay_cap334_recovery import gallery_snapshot
    s = similar_pipeline; p = s.pipeline; t = p.tracker
    bank = t.identity_bank
    gallery = gallery_snapshot(bank)
    samples = list(t._provisional_association.entries[3].samples)
    s.step(23, timestamp=fast_stamp, result_age=.10)
    assert p.last_identity_processing["mode"] == "detector_continuation"
    assert bank._similar_follow_states[(1, 3)]["last_timestamp"] == 3.2
    assert [(r[0], r[1]) for r in t._provisional_association.entries[3].samples] == [
        (r[0], r[1]) for r in samples]
    records = s.step(24, timestamp=full_stamp, result_age=result_age)
    assert p.last_identity_processing["mode"] == "full"
    assert p.reid.calls == 2 and p.detector.calls == 3
    assert [(r.track_id, r.reid_uid, r.time_since_update) for r in records] == [(3, 1, 0)]
    assert bank._similar_follow_states[(1, 3)]["last_cap"] == 24
    assert bank._similar_follow_states[(1, 3)]["last_timestamp"] == full_stamp
    assert t._provisional_association.entries[3].samples[-1][:2] == (24, full_stamp)
    assert not t.deepsort.tracker.metric.samples
    assert gallery_snapshot(bank) == gallery
    assert t._detector_full_bridge is None


@pytest.mark.parametrize("fault", ["expired", "new_person", "color", "competition", "geometry", "conflict"])
def test_full_position_bridge_does_not_bypass_current_full_evidence(similar_pipeline, fault):
    s = similar_pipeline; p = s.pipeline; t = p.tracker
    s.step(23, timestamp=3.399)
    options = dict(timestamp=3.749, result_age=.10)
    if fault == "expired": options["timestamp"] = 3.801
    elif fault == "new_person": s.distance = .80
    elif fault == "color": options["image"] = np.full_like(s.frame, (160, 200, 20))
    elif fault == "competition": p.detector.detections.append(Detection((10., 80., 150., 450.), .95, 0))
    elif fault == "geometry": p.detector.detections = [Detection((430., 40., 630., 450.), .95, 0)]
    elif fault == "conflict": t.identity_bank._mapped_geometry_conflicts[3] = dict(uid=1)
    s.step(24, **options)
    assert p.last_identity_processing["mode"] == "full"
    assert p.reid.calls == 2
    assert t._current_detector_position_bridge is None


def test_fast_budget_and_full_capture_lease_cannot_slide(similar_pipeline):
    s = similar_pipeline; p = s.pipeline
    deadline = p.tracker._detector_proof.deadline
    for cap, stamp in ((23, 3.25), (24, 3.30)):
        s.step(cap, timestamp=stamp)
        assert p.last_identity_processing["mode"] == "detector_continuation"
        assert p.tracker._detector_proof.deadline == deadline
    s.step(25, timestamp=3.35)
    assert p.last_identity_processing["mode"] == "full"
    assert p.last_identity_processing["reason"] == "fast_budget_exhausted"
    assert p.reid.calls == 2


@pytest.mark.parametrize("gap", [.25, .299])
def test_slow_cadence_runs_full_before_skipping_would_expire_bank_state(similar_pipeline, gap):
    s = similar_pipeline; p = s.pipeline
    for cap in (23, 24, 25):
        records = s.step(cap, timestamp=3.2+(cap-22)*gap)
        assert [(r.track_id, r.reid_uid) for r in records] == [(3, 1)]
        assert p.last_identity_processing["mode"] == "full"
        assert p.tracker.identity_bank._similar_follow_states[(1, 3)]["last_cap"] == cap
    assert p.detector.calls == p.reid.calls == 4


@pytest.mark.parametrize("gap, mode", [(.10, "detector_continuation"), (.20, "full")])
def test_accepted_crop_consumes_only_existing_permission_with_shorter_full_budget(fast_pipeline, gap, mode):
    s = make_similar_pipeline(fast_pipeline, seed_box=(450., 1., 620., 478.),
                             current_box=(470., 0., 639., 479.))
    p = s.pipeline; t = p.tracker
    assert t.identity_bank.last_assignments[3]["similar_follow"]["crop_continuation"]
    assert t._detector_proof.permission_deadline == pytest.approx(3.1+.75)
    before = pickle.dumps(t.identity_bank)
    records = s.step(23, timestamp=3.2+gap)
    assert [(r.track_id, r.reid_uid) for r in records] == [(3, 1)]
    assert p.last_identity_processing["mode"] == mode
    if mode == "detector_continuation":
        assert pickle.dumps(t.identity_bank) == before
        assert p.reid.calls == 1 and p.detector.calls == 2
    else:
        assert p.reid.calls == p.detector.calls == 2


def test_strict_lane_cannot_ignore_quarantine(fast_pipeline):
    from test_detector_continuation_pipeline import seed
    s = fast_pipeline; p = s.pipeline
    seed(s)
    assert p.tracker._detector_proof.permission == "strong"
    p.tracker.identity_bank.last_assignments[1]["template_update_quarantined"] = True
    s.step(5)
    assert p.last_identity_processing["mode"] == "full"
    assert p.reid.calls == p.detector.calls == 5


def test_initial_reacquire_is_not_enough_to_seed_similar_fast(fast_pipeline):
    from rk_vision.detector_continuation import capture_observation
    tracker, _ = activated()
    assert tracker.identity_bank.last_assignments[3]["reason"] == "similar_follow_reacquire"
    obs = capture_observation(Detection(BOX, .95, 0),
        dict(capture_frame_id=21, capture_timestamp=3.1, integrated_yaw_deg=0.), 640, 480)
    assert not tracker._similar_detector_qualified(1, 3, obs)


@pytest.mark.parametrize("fault", ["gallery", "self_reference", "geometry", "competing",
    "state_missing", "state_position_only", "state_changed", "mapping", "conflict",
    "revoked", "suspect", "recheck", "quarantine_control_reject", "search", "new_track",
    "color", "missing", "new_edge", "capture_gap", "expired", "stale"])
def test_negative_events_force_same_frame_full_without_second_feature_pass(similar_pipeline, fault):
    s = similar_pipeline; p = s.pipeline; b = p.tracker.identity_bank
    a = b.last_assignments[3]; opts = dict(timestamp=3.399)
    if fault == "gallery": a["similar_follow"]["gallery_distance"] = .301
    elif fault == "self_reference":
        a["similar_follow"].update(full_distance=.10, gallery_distance=.60, reference_cap=99)
    elif fault == "geometry": p.detector.detections = [Detection((430., 40., 630., 450.), .95, 0)]
    elif fault == "competing": p.detector.detections.append(Detection((10., 80., 150., 450.), .95, 0))
    elif fault == "state_missing": b._similar_follow_states.clear()
    elif fault == "state_position_only": b._similar_follow_states[(1, 3)]["position_only"] = True
    elif fault == "state_changed": b._similar_follow_states[(1, 3)]["last_cap"] = 999
    elif fault == "mapping": b.track_to_uid[3] = 2
    elif fault == "conflict": b._mapped_geometry_conflicts[3] = dict(uid=1)
    elif fault == "revoked": b._geometry_revoked_uids[1] = True
    elif fault == "suspect": b._reacquire_control_suspects[1] = dict(track_id=3)
    elif fault == "recheck": a["identity_recheck_pending"] = True
    elif fault == "quarantine_control_reject": a["identity_control_rejected"] = True
    elif fault == "search": p.set_detector_continuation_context(active_uid=1, allowed=False)
    elif fault == "new_track": p.tracker.deepsort.tracker.tracks[0].track_id = 4
    elif fault == "color": opts["image"] = np.full_like(s.frame, (160, 200, 20))
    elif fault == "missing": p.detector.detections = []
    elif fault == "new_edge": p.detector.detections = [Detection((2., 40., 202., 450.), .95, 0)]
    elif fault == "capture_gap": opts["timestamp"] = 3.501
    elif fault == "expired": opts["result_age"] = .65
    elif fault == "stale": opts["result_age"] = .181
    s.step(23, **opts)
    assert p.last_identity_processing["mode"] == "full"
    assert p.reid.calls == p.detector.calls == 2


def test_fast_plan_is_rechecked_at_commit_after_identity_change(similar_pipeline):
    s = similar_pipeline; p = s.pipeline; t = p.tracker
    plan = t.plan_detected_continuation(p.detector.detections, image_width=640, image_height=480,
        frame_context=dict(capture_frame_id=23, capture_timestamp=3.399, integrated_yaw_deg=0.),
        active_uid=1, now=3.41, color_features=p.reid.last_color_features, raw_candidate_count=1)
    assert plan is not None
    t.identity_bank.last_assignments[3]["identity_competition"]["passed"] = False
    assert t.commit_detected_continuation(plan, now=3.42) is None
    assert t._frame_index == 22 and t.control_assignment_for_track(3) is None


def test_real_main_lease_and_depth_consumer_accept_follow_only_fast_without_stop(similar_pipeline):
    import request_0513_modular as runtime
    s = similar_pipeline; p = s.pipeline
    records = s.step(23, timestamp=3.399)
    owner = object.__new__(runtime.PersonTracker)
    owner._follow_controller = SimpleNamespace(active_target_id=1)
    owner._rknn_pipeline = p
    owner.running = True
    calls = []
    owner._replace_action_queue = lambda *args, **kw: calls.append(args)
    assert owner._update_detector_identity_lease(records, 23, 3.399, now=s.clock.now, stale=False)
    evidence, now = read_visual_identity_evidence(owner)
    assert evidence.live(1, now) and evidence.permits_depth(1, 3.405, now)
    assert owner._detector_identity_lease.expires_at == pytest.approx(3.2+FULL_PROOF_TTL_SEC)
    assert not calls
    r = records[0]
    depth = resolve_depth_target_observation(target_id=1, display_bbox=(r.x1, r.y1, r.x2, r.y2),
        capture_frame_id=23, capture_timestamp=3.399, observations=p.tracker.last_identity_observations,
        width=640, height=480, expected_raw_track_id=3)
    assert depth is not None and depth.source == "yolo_detector"
    assert not evidence.motion_identity_live(1, owner._detector_identity_lease.expires_at)


def test_fast_assignment_lease_replays_do_not_extend_original_full_proof(similar_pipeline):
    s = similar_pipeline; p = s.pipeline
    s.step(23, timestamp=3.399)
    a = p.tracker.control_assignment_for_track(3)
    assert from_assignment(a, uid=1, track_id=3, capture=23, timestamp=3.399, now=3.42)
    assert from_assignment(a, uid=1, track_id=3, capture=23, timestamp=3.399,
                           now=a["identity_valid_until"]) is None
