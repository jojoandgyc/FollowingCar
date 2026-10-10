"""CAP1542: bank-completed identity exits search once, with range-owned yaw.

Use the real reacquisition gate, depth filters/transfer, adapter, paired planner
and writer. Only acquisition/serial are in memory. Log geometry is reproduced;
the counterfactual right-side case proves yaw is not frozen to old search.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.detector_identity_lease import publish_visual_identity_evidence
from test_cap609_reacquire_depth_transfer import handoff, capture, queue_capture
from test_short_follow_adapter import paired, owner
from test_search_observation_arbitration import _record


def completed_candidate(h, *, bbox=(40.17799, 2.29, 212.926, 478.007)):
    h.a.target = replace(h.a.target, bbox=bbox)
    h.obj.search_state = h.obj._follow_controller.search_state = "searching"
    h.obj.search_direction = h.obj._follow_controller.search_direction = "left"
    # Launcher policy recorded at log line 26, unlike library's legacy
    # 1/6..4/6 defaults. Do not stub the actual reacquisition brake function.
    h.obj._follow_controller.cfg = replace(h.obj._follow_controller.cfg,
        center_left_ratio=.45, center_right_ratio=.55)
    h.obj._current_rotate_raw_target = 7
    h.camera._latest_depth[:] = 1527
    candidate = capture(h, 1542)
    proof = h.obj._validated_visual_observation
    publish_visual_identity_evidence(h.obj, observation=proof, lease=None)
    candidate["debug"]["assignment"] = dict(
        uid=1, mapped_uid=1, reason="similar_follow_reacquire", distance=.2650365,
        identity_permission="similar_follow", bbox_quality_ok=True,
        identity_control_rejected=False, reacquire_geometry_ok=True,
        similar_follow=dict(status="follow", count=2, capture_frame_id=1542,
            capture_timestamp=proof.timestamp, raw_track_id=3,
            completed_confirmation=True, learning_allowed=False),
        reacquire_geometry=dict(ok=True, reason="continuous", current=dict(
            capture_frame_id=1542, capture_timestamp=proof.timestamp, track_id=3,
            frame_index=h.obj.frame_index, bbox=bbox)),
        identity_competition=dict(uid=1, frame_index=h.obj.frame_index, passed=True),
        bank_updated=False, template_update_quarantined=True,
    )
    return candidate


@pytest.mark.parametrize("bbox, sign", [
    ((40.17799, 2.29, 212.926, 478.007), -1),  # actual CAP1542 remains left
    ((470., 3., 630., 478.), 1),  # counterfactual current right, old search left
])
def test_bank_completed_confirmation_reaches_writer_on_first_runtime_frame(
        handoff, monkeypatch, caplog, bbox, sign):
    h = handoff
    caplog.set_level("INFO", logger=runtime.logger.name)
    candidate = completed_candidate(h, bbox=bbox)
    h.obj._publish_search_reacquire_direction_hold = lambda *_: pytest.fail("old search yaw")
    assert not h.obj._hold_for_confirmed_search_reacquire([candidate], width=640, height=480)
    assert h.obj.search_state == h.obj._follow_controller.search_state == "none"
    assert h.obj._reacquire_depth_pending
    transfer = h.obj._search_reacquire_depth_transfer
    assert transfer is not None  # one real accepted range, not a synthetic default
    monkeypatch.setattr(h.distance, "get_frame_distance_state",
                        lambda *_a, **_kw: pytest.fail("duplicate physical scan"))
    assert queue_capture(h)
    plan = h.obj._short_follow.snapshot().plan
    assert plan is not None and plan.moving and not plan.forwarding
    assert plan.base_rpm == 0 and plan.longitudinal_reason == "reacquire_depth_pending"
    assert plan.left_rpm * sign > 0 > plan.right_rpm * sign
    assert plan.depth_timestamp == transfer.distance_state.sample_timestamp
    assert plan.expires_at == pytest.approx(min(plan.depth_timestamp+.3, plan.capture_timestamp+.5))
    assert plan.integral_dt_sec == 0 and plan.i_rpm == 0
    h.motor._service_short_follow()
    assert h.driver.pairs == [(plan.left_rpm, -plan.right_rpm)]
    assert not h.driver.stops and not h.obj._queued_calls
    assert "bank_confirmation_consumed=True" in caplog.text
    assert "result=hold_search" not in caplog.text


def test_explicit_permission_not_reason_whitelist(handoff):
    h = handoff
    candidate = completed_candidate(h)
    candidate["debug"]["assignment"]["reason"] = "another_diagnostic_label"
    assert not h.obj._hold_for_confirmed_search_reacquire([candidate], width=640, height=480)
    assert h.obj._reacquire_depth_pending


def test_first_range_yaw_keeps_original_deadline_and_duplicate_does_not_renew(handoff):
    h = handoff
    candidate = completed_candidate(h)
    assert not h.obj._hold_for_confirmed_search_reacquire([candidate], width=640, height=480)
    assert queue_capture(h)
    original = h.obj._short_follow.snapshot().plan
    h.motor._service_short_follow()
    assert h.driver.pairs and not h.driver.stops
    h.clock[0] += .05
    assert queue_capture(h)  # camera still contains the very same Depth exposure
    assert h.obj._last_frame_distance_state.temporal_status == "duplicate"
    assert h.obj._short_follow.snapshot().plan is original
    # CAP1546's observed full processing latency exceeded the prior range's
    # 300 ms lifetime. This fix does not turn position continuity into range.
    h.clock[0] = original.depth_timestamp + .351
    assert h.obj._validated_visual_observation.live(1, h.clock[0])
    h.motor._service_short_follow()
    assert h.driver.stops
    assert h.obj._short_follow.snapshot().plan.expires_at == original.expires_at


def test_new_confirmed_position_changes_yaw_without_renewing_pending_range(handoff):
    h = handoff
    candidate = completed_candidate(h)
    assert not h.obj._hold_for_confirmed_search_reacquire([candidate], width=640, height=480)
    assert queue_capture(h)
    original = h.obj._short_follow.snapshot().plan
    h.clock[0] += .10
    h.a.target = replace(h.a.target, bbox=(470., 3., 630., 478.))
    capture(h, 1545)
    publish_visual_identity_evidence(h.obj, observation=h.obj._validated_visual_observation, lease=None)
    plan = h.obj._short_follow_adapter.publish_visual_lateral(h.a.target, 640, 480,
        h.obj._active_capture_frame_id, h.obj._active_capture_timestamp, now=h.clock[0])
    assert plan is not None and plan.left_rpm > 0 > plan.right_rpm
    assert not plan.forwarding and plan.depth_timestamp == original.depth_timestamp
    assert plan.expires_at == original.expires_at and plan.base_rpm == 0
    h.motor._service_short_follow()
    assert h.driver.pairs == [(plan.left_rpm, -plan.right_rpm)]
    assert not h.driver.stops


@pytest.mark.parametrize("change", [
    "no_completed_permission", "observe", "one_frame", "wrong_uid", "wrong_raw",
    "old_capture", "old_timestamp", "old_geometry", "geometry_conflict",
    "competition_failed", "old_competition", "recheck", "rejected", "excluded",
    "bad_quality", "no_proof", "old_proof", "expired_proof", "continuation",
])
def test_current_completed_permission_cannot_be_inferred_from_low_distance_or_reason(handoff, change):
    h = handoff
    candidate = completed_candidate(h)
    assignment = candidate["debug"]["assignment"]
    permission = assignment["similar_follow"]
    if change == "no_completed_permission": permission.pop("completed_confirmation")
    elif change == "observe": permission["status"] = "observe"
    elif change == "one_frame": permission["count"] = 1
    elif change == "wrong_uid": assignment["uid"] = 2
    elif change == "wrong_raw": permission["raw_track_id"] = 4
    elif change == "old_capture": permission["capture_frame_id"] -= 1
    elif change == "old_timestamp": permission["capture_timestamp"] -= .01
    elif change == "old_geometry": assignment["reacquire_geometry"]["current"]["capture_frame_id"] -= 1
    elif change == "geometry_conflict": assignment["reacquire_geometry"]["ok"] = False
    elif change == "competition_failed": assignment["identity_competition"]["passed"] = False
    elif change == "old_competition": assignment["identity_competition"]["frame_index"] -= 1
    elif change == "recheck": assignment["identity_recheck_pending"] = True
    elif change == "rejected": assignment["identity_control_rejected"] = True
    elif change == "excluded": assignment["search_excluded"] = True
    elif change == "bad_quality": assignment["bbox_quality_ok"] = False
    else:
        proof = h.obj._validated_visual_observation
        if change == "no_proof": proof = False
        elif change == "old_proof": proof = replace(proof, capture=1540)
        elif change == "expired_proof": proof = replace(proof, expires_at=h.clock[0])
        else: proof = replace(proof, continuation_sample_timestamp=proof.timestamp)
        publish_visual_identity_evidence(h.obj, observation=proof, lease=None)
    assert h.obj._completed_similar_follow_confirmation(assignment, uid=1, track_id=3) is None
    assert not h.obj._short_follow.snapshot().active
    assert not h.driver.pairs


@pytest.mark.parametrize("change", ["rejection", "republication", "uid", "capture", "expiry"])
def test_lateral_only_range_transfer_requires_same_live_identity_publication(handoff, change):
    h = handoff
    candidate = completed_candidate(h)
    assert not h.obj._hold_for_confirmed_search_reacquire([candidate], width=640, height=480)
    proof = h.obj._validated_visual_observation
    if change == "rejection": publish_visual_identity_evidence(h.obj, observation=False, lease=False)
    elif change == "republication": publish_visual_identity_evidence(h.obj, observation=replace(proof), lease=None)
    elif change == "uid": h.obj._follow_controller.active_target_id = 2
    elif change == "capture": h.obj._active_capture_frame_id += 1
    else: h.clock[0] = proof.expires_at
    assert h.obj._take_search_reacquire_depth(640, 480, [h.a.target],
        h.obj._active_capture_frame_id, h.obj._active_capture_timestamp) is None
    assert not h.driver.pairs


def prepare_consumer(h, assignment, monkeypatch):
    obj = h.obj
    obj._identity_assignment_debug_for_track = lambda raw: assignment if raw in (3, -1) else {}
    obj._bunker_runtime = SimpleNamespace(check_merged_dets=lambda *_: None)
    obj._handle_hazard_safety_state = lambda _: False
    obj._single_person_geometry_fallback_id = lambda *_a: None
    obj._search_geometry_reacquire_id = lambda *_a, **_kw: None
    obj._visual_reacquire_hold_match = lambda *_a, **_kw: None
    obj._visual_reacquire_hold_uid = None
    obj._visible_unsteerable_uid = None
    obj._last_control_decision_reason = "search_left"
    monkeypatch.setattr(runtime, "VISION_REID_ENABLE", True)
    monkeypatch.setattr(runtime, "VISION_TRACK_LOG_ENABLE", False)
    monkeypatch.setattr(runtime, "VISION_CONTROL_USE_PREDICTED_TRACKS", False)
    obj._publish_search_reacquire_direction_hold = lambda *_: pytest.fail("old frozen yaw")
    # The imported ranging fixture normally supplies its one fixed target.
    # Keep empty formal persons genuinely empty for probe/missing-frame tests.
    obj._persons_to_targets = lambda persons, **_kw: [h.a.target] if persons else []


def test_real_consume_does_not_early_return_before_current_target_control(handoff, monkeypatch):
    h = handoff
    candidate = completed_candidate(h)
    obj = h.obj
    prepare_consumer(h, deepcopy(candidate["debug"]["assignment"]), monkeypatch)
    rec = _record(track=3, uid=1, bbox=candidate["bbox"], score=.8552175)
    obj._consume_track_records([rec], 640, 480, "test")
    plan = obj._short_follow.snapshot().plan
    assert plan is not None and plan.moving and not plan.forwarding
    h.motor._service_short_follow()
    assert h.driver.pairs == [(plan.left_rpm, -plan.right_rpm)]
    assert not h.driver.stops and not obj._queued_calls


def current_probe(h):
    proof = h.obj._validated_visual_observation
    stamp = proof.timestamp + .1611
    h.clock[0] = stamp + .02
    h.obj._active_capture_frame_id = 1545
    h.obj._active_capture_timestamp = stamp
    h.obj.frame_index += 1
    bbox = (130., 3., 286., 478.)
    permission = dict(source="follow_only_detector_probe", uid=1, track_id=-1,
        reference_track_id=3, capture_frame_id=1545, capture_timestamp=stamp, bbox=bbox,
        reference_capture_frame_id=1542, reference_capture_timestamp=proof.timestamp,
        expires_at=proof.expires_at, gallery_distance=.277,
        center_jump_ratio=.127, area_similarity=.90,
        identity_authorized=False, learning_allowed=False)
    row = dict(raw_track_id=-1, uid=0, detector_bbox=bbox,
        assignment=dict(uid=0, best_uid=1, reason="secondary_evidence_unavailable",
            identity_control_rejected=True, follow_only_position_evidence=permission),
        sample_metadata=dict(is_fresh=True, capture_frame_id=1545, capture_timestamp=stamp,
            detector_confidence=.791, quality_bbox_ok=True, source_detection_index=0))
    h.obj._rknn_pipeline = SimpleNamespace(last_frame_width=640, last_frame_height=480,
        last_identity_processing=dict(mode="full", full_features_current=True,
            capture_frame_id=1545, capture_timestamp=stamp),
        tracker=SimpleNamespace(last_identity_observations=[row],
            associated_position_contradiction=lambda *_: None))
    return row, _record(track=-1, uid=0, bbox=bbox, score=.791)


def test_real_probe_record_through_identity_and_consume_preserves_bounded_pair(handoff, monkeypatch):
    h = handoff
    candidate = completed_candidate(h)
    assert not h.obj._hold_for_confirmed_search_reacquire([candidate], width=640, height=480)
    assert queue_capture(h)
    original = h.obj._short_follow.snapshot().plan
    publication = h.obj._visual_identity_evidence
    row, rec = current_probe(h)
    prepare_consumer(h, row["assignment"], monkeypatch)
    assert h.obj._update_detector_identity_lease([rec], 1545, h.obj._active_capture_timestamp,
        now=h.clock[0], stale=False)
    moved = h.obj._short_follow.snapshot().plan
    assert moved.yaw_capture_id == 1545 and moved is not original
    assert moved.depth_timestamp == original.depth_timestamp
    assert moved.expires_at == original.expires_at
    h.obj._consume_track_records([rec], 640, 480, "test")
    assert h.obj._visual_identity_evidence is publication
    assert h.obj._short_follow.snapshot().plan is moved
    assert rec.reid_uid == 0 and row["assignment"]["uid"] == 0
    h.motor._service_short_follow()
    assert h.driver.pairs == [(moved.left_rpm, -moved.right_rpm)]
    assert not h.driver.stops and not h.obj._queued_calls


@pytest.mark.parametrize("change", ["ordinary_uid0", "old_metadata", "wrong_bbox", "positive_raw",
    "another_uid", "two_records", "wrong_owner", "conflict", "hazard", "expired", "stop"])
def test_probe_exception_does_not_preserve_arbitrary_rejected_records(handoff, change):
    h = handoff
    candidate = completed_candidate(h)
    assert not h.obj._hold_for_confirmed_search_reacquire([candidate], width=640, height=480)
    assert queue_capture(h)
    row, rec = current_probe(h)
    h.obj._identity_assignment_debug_for_track = lambda raw: row["assignment"]
    h.obj._bunker_runtime = SimpleNamespace(config=SimpleNamespace(enabled=False))
    records = [rec]
    if change == "ordinary_uid0": row["assignment"].pop("follow_only_position_evidence")
    elif change == "old_metadata": row["sample_metadata"]["capture_frame_id"] -= 1
    elif change == "wrong_bbox": records = [replace(rec, x1=rec.x1+1)]
    elif change == "positive_raw": records = [replace(rec, track_id=3)]
    elif change == "another_uid": records = [replace(rec, reid_uid=2)]
    elif change == "two_records": records.append(replace(rec, track_id=-2))
    elif change == "wrong_owner": row["assignment"]["follow_only_position_evidence"]["reference_track_id"] = 4
    elif change == "conflict": h.obj._rknn_pipeline.tracker.associated_position_contradiction = lambda *_: "geometry_conflict"
    elif change == "hazard":
        h.obj._bunker_runtime.config = SimpleNamespace(enabled=True, mode="merged", class_ids={9})
        records[0] = replace(rec, class_id=9)
    elif change == "expired": h.clock[0] = h.obj._validated_visual_observation.expires_at
    elif change == "stop": h.obj._explicit_stop_requested = True
    h.obj._update_detector_identity_lease(records, 1545, h.obj._active_capture_timestamp,
        now=h.clock[0], stale=False)
    assert h.obj._validated_visual_observation is False
