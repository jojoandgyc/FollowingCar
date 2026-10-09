"""Source-bound vision publication and independent yaw/depth generations.

Use the real full-result adapter, depth admission/reader and lateral publisher;
only camera/model/serial execution is absent. Events reproduce the full-result
-> depth grant -> old-yaw-expiry interleaving before visual control completes.
"""
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.detector_identity_lease import (
    ValidatedVisualObservation, validated_visual_observation,
)
from test_detector_identity_lease import assignment, fixture_pipeline
from test_distance_only_runtime_authority import distance_only, publish
from test_distance_pi_runtime import pi_owner
from test_lateral_zero_runtime import owner, NOW, _intent


def full_result(owner, *, cap=256, stamp=NOW-.16, now=NOW, uid=1, rejected=False):
    data, records = fixture_pipeline(owner)
    data.clear()
    data.update(uid=uid, bbox_quality_ok=True, identity_control_rejected=rejected)
    records[0].reid_uid = uid
    owner._rknn_pipeline.last_identity_processing = dict(
        mode="full", full_features_current=True, capture_frame_id=cap,
        capture_timestamp=stamp)
    assert owner._update_detector_identity_lease(records, cap, stamp, now=now, stale=False)
    return data, records


@pytest.mark.parametrize("age,remaining", [(.01, .25), (.16, .19), (.34, .01)])
def test_full_proof_deadline_uses_capture_and_original_visibility_window(age, remaining):
    observation = validated_visual_observation(
        uid=1, track_id=1, capture=256, timestamp=NOW-age, now=NOW,
        visibility_window=.25)
    assert observation.expires_at == pytest.approx(NOW+remaining)
    assert observation.live(1, NOW)
    assert not observation.live(2, NOW)
    assert not observation.live(1, observation.expires_at)


def test_fast_visual_publication_cannot_extend_original_identity_proof(owner):
    data, records = fixture_pipeline(owner)
    assert owner._update_detector_identity_lease(records, 4, 99.9, now=100., stale=False)
    observation = owner._validated_visual_observation
    assert observation.kind == "detector_continuation"
    assert observation.expires_at == pytest.approx(100.05)
    data.update(assignment(5, 100.01))
    assert owner._update_detector_identity_lease(records, 5, 100.01, now=100.02, stale=False)
    assert owner._validated_visual_observation.expires_at == observation.expires_at


@pytest.mark.parametrize("fault", ["duplicate", "old_capture", "old_time"])
def test_replayed_capture_cannot_publish_or_renew_visual_evidence(owner, fault):
    data, records = full_result(owner)
    previous = owner._validated_visual_observation
    cap = 255 if fault == "old_capture" else 256 if fault == "duplicate" else 257
    stamp = previous.timestamp+.01 if fault == "old_capture" else previous.timestamp
    owner._rknn_pipeline.last_identity_processing.update(
        capture_frame_id=cap, capture_timestamp=stamp)
    assert not owner._update_detector_identity_lease(records, cap, stamp, now=NOW+.02, stale=False)
    assert owner._validated_visual_observation is previous


@pytest.mark.parametrize("fault", ["late", "uid", "pending", "rejected", "features",
                                   "source_capture", "source_time", "ambiguous", "predicted"])
def test_invalid_full_result_cannot_replace_visibility_with_completion_time(distance_only, fault):
    a = distance_only
    data, records = full_result(a.owner)
    assert publish(a)[1]
    cap, stamp, now = 257, NOW-.14, NOW+.01
    a.owner._rknn_pipeline.last_identity_processing.update(
        capture_frame_id=cap, capture_timestamp=stamp)
    if fault == "late": now = stamp+.351
    if fault == "uid": records[0].reid_uid = 0
    if fault == "pending": data["identity_recheck_pending"] = True
    if fault == "rejected": data["identity_control_rejected"] = True
    if fault == "features": a.owner._rknn_pipeline.last_identity_processing["full_features_current"] = False
    if fault == "source_capture": a.owner._rknn_pipeline.last_identity_processing["capture_frame_id"] = cap-1
    if fault == "source_time": a.owner._rknn_pipeline.last_identity_processing["capture_timestamp"] = stamp-.01
    if fault == "ambiguous": records.append(SimpleNamespace(reid_uid=1, track_id=2, time_since_update=0))
    if fault == "predicted": records[0].time_since_update = 1
    a.owner._update_detector_identity_lease(records, cap, stamp, now=now, stale=False)
    assert a.owner._validated_visual_observation is False
    # Even a later visual-control completion cannot turn rejection into proof.
    a.owner._last_vision_control_ts = a.clock.now
    assert a.owner._fresh_depth_linear_snapshot(1, quiet=True) is None


@pytest.mark.parametrize("old_quality,old_park", [("reliable", False), ("weak", False),
                                                  ("reliable", True)])
@pytest.mark.parametrize("depth_from_rgb", [-.01, -.001, 0., .001, .14])
def test_new_full_identity_and_depth_survive_old_yaw_expiry_before_visual_completion(
        distance_only, old_quality, old_park, depth_from_rgb):
    a, o = distance_only, distance_only.owner
    a.stamp = NOW-.16+depth_from_rgb
    a.feedback[0] = replace(a.feedback[0], timestamp=NOW-.03)
    a.shared = replace(a.shared, sample_timestamp=a.stamp, feedback_timestamp=NOW-.03)
    a.frame = replace(a.frame, steering_feedback=a.feedback[0], distance_state=replace(
        a.frame.distance_state, sample_timestamp=a.stamp, observation_timestamp=a.stamp))
    o._last_vision_control_ts = NOW-.30  # CAP253 control is obsolete.
    intent = _intent(o, capture_frame_id=253, capture_timestamp=NOW-.38,
                     valid_until=NOW-.01, bbox_quality=old_quality, park_requested=old_park)
    ready, finish = Event(), Event()
    errors = []

    def visual_depth_worker():
        try:
            full_result(o)  # actual identity adapter, before ranging/control
            with o._control_update_lock:
                assert publish(a, percent=36)[1]  # real new depth -> 72 RPM grant
            ready.set()
            assert finish.wait(2.)  # visual/lateral work has not completed yet
        except BaseException as exc:
            errors.append(exc)
            ready.set()

    thread = Thread(target=visual_depth_worker)
    thread.start()
    try:
        assert ready.wait(2.)
        assert not errors
        linear = o._depth30_linear_snapshot
        timing = o._depth30_linear_timing
        assert isinstance(o._validated_visual_observation, ValidatedVisualObservation)
        assert o._last_vision_control_ts == NOW-.30
        with o._control_update_lock:
            assert o._publish_lateral_zero(intent, "revoke:expired")
        assert o._depth30_linear_snapshot is linear
        assert o._depth30_linear_timing is timing
        assert o._current_forward_percent == 36
        assert o._current_steer_correction_rpm == 0
        assert o._queued_calls[-1][0] == (runtime.ACTION_STEER_RIGHT,)
        assert not o.suspensions
        assert getattr(o, "_near_yaw_park_request", None) is None
        assert o._fresh_depth_linear_snapshot(1, quiet=True) == linear
    finally:
        finish.set()
        thread.join(2.)
    assert not thread.is_alive() and not errors


@pytest.mark.parametrize("fault", ["other_uid", "same_capture", "older_timestamp",
                                   "older_depth", "grey_other_sample", "visual_expired",
                                   "depth_expired"])
def test_old_weak_yaw_cannot_borrow_unrelated_or_expired_visual_proof(distance_only, fault):
    a, o = distance_only, distance_only.owner
    full_result(o)
    # Depth and RGB are new relative to the old limited yaw, even when Depth
    # was captured 1 ms before RGB. Admission itself stays unchanged.
    a.stamp = NOW-.161
    a.feedback[0] = replace(a.feedback[0], timestamp=NOW-.03)
    a.shared = replace(a.shared, sample_timestamp=a.stamp, feedback_timestamp=NOW-.03)
    a.frame = replace(a.frame, steering_feedback=a.feedback[0], distance_state=replace(
        a.frame.distance_state, sample_timestamp=a.stamp, observation_timestamp=a.stamp))
    assert publish(a)[1]
    intent = _intent(o, capture_frame_id=253, capture_timestamp=NOW-.20,
                     valid_until=NOW-.01, bbox_quality="limited")
    proof = o._validated_visual_observation
    if fault == "other_uid": proof = replace(proof, uid=2)
    elif fault == "same_capture": proof = replace(proof, capture=253)
    elif fault == "older_timestamp": proof = replace(proof, timestamp=NOW-.21)
    elif fault == "older_depth": intent = replace(intent, capture_timestamp=a.stamp)
    elif fault == "grey_other_sample": proof = replace(proof, continuation_sample_timestamp=a.stamp-.001)
    elif fault == "visual_expired": a.clock.now = proof.expires_at
    else: a.clock.now = a.stamp+.251
    o._validated_visual_observation = proof
    a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now)
    o._publish_lateral_zero(intent, "old_limited_expiry")
    assert o._depth30_linear_snapshot is None
    assert o._queued_calls[-1][0] == (runtime.ACTION_STOP,)


def test_old_weak_yaw_grey_bridge_keeps_only_its_original_depth_sample(distance_only):
    a, o = distance_only, distance_only.owner
    recheck_result(a)
    original = o._depth30_linear_snapshot
    timing = o._depth30_linear_timing
    proof = o._validated_visual_observation
    intent = _intent(o, capture_frame_id=253, capture_timestamp=NOW-.38,
                     valid_until=NOW-.01, bbox_quality="limited")
    assert proof.continuation_sample_timestamp == original[3]
    o._publish_lateral_zero(intent, "old_limited_expiry_during_recheck")
    assert o._depth30_linear_snapshot is original
    assert o._depth30_linear_timing is timing
    assert o._validated_visual_observation is proof
    assert o._queued_calls[-1][0] == (runtime.ACTION_STEER_RIGHT,)
    assert not proof.permits_depth(1, original[3]+.001, NOW)


@pytest.mark.parametrize("fault", ["identity", "visual_expired", "depth_expired", "stop", "brake"])
def test_old_yaw_expiry_never_rescues_invalid_new_forward(distance_only, fault):
    a, o = distance_only, distance_only.owner
    full_result(o)
    assert publish(a)[1]
    intent = _intent(o, capture_frame_id=253, capture_timestamp=NOW-.38, valid_until=NOW-.01)
    if fault == "identity": full_result(o, cap=257, stamp=NOW-.14, uid=0)
    if fault == "visual_expired":
        # Expire only visibility; the configured capture lease is independent
        # of the physical Depth lease and may be longer than this fixture's.
        o._validated_visual_observation = replace(
            o._validated_visual_observation, expires_at=a.clock.now)
    if fault == "depth_expired": a.clock.now = NOW+.231
    if fault == "stop": o._explicit_stop_requested = True
    if fault == "brake": o._brake_hold_active = True
    a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now)
    with o._control_update_lock:
        o._publish_lateral_zero(intent, "revoke:expired")
    assert o._fresh_depth_linear_snapshot(1, quiet=True) is None
    assert not o._queued_calls or o._queued_calls[-1][0] == (runtime.ACTION_STOP,)


@pytest.mark.parametrize("superseded_by", ["new_yaw", "new_uid"])
def test_delayed_old_yaw_cancel_cannot_modify_new_generation(distance_only, superseded_by):
    a, o = distance_only, distance_only.owner
    full_result(o)
    publish(a)
    old = _intent(o, capture_frame_id=253, capture_timestamp=NOW-.38)
    if superseded_by == "new_yaw": _intent(o, capture_frame_id=256, capture_timestamp=NOW-.16)
    else: o._follow_controller.active_target_id = 2
    before = (o._depth30_linear_snapshot, o._current_forward_percent,
              o._current_steer_correction_rpm, o._last_vision_correction_at)
    with o._control_update_lock:
        assert not o._publish_lateral_zero(old, "late_cancel")
    after = (o._depth30_linear_snapshot, o._current_forward_percent,
             o._current_steer_correction_rpm, o._last_vision_correction_at)
    assert after == before
    assert not o._queued_calls
    if superseded_by == "new_uid":
        assert o._fresh_depth_linear_snapshot(1, quiet=True) is None


def test_startup_before_uid_lock_and_legacy_adapter_are_not_implicit_proof(owner):
    owner._follow_controller.active_target_id = None
    full_result(owner)
    assert owner._validated_visual_observation is None
    owner._follow_controller.active_target_id = 1
    full_result(owner, cap=257, stamp=NOW-.14)
    assert isinstance(owner._validated_visual_observation, ValidatedVisualObservation)


def recheck_result(a, *, conflict=False):
    o = a.owner
    data, records = full_result(o)
    assert publish(a)[1]
    o._confirmed_identity_execution_anchor = dict(
        uid=1, track_id=1, capture_frame_id=256, capture_timestamp=NOW-.16,
        bbox=(20., 20., 200., 460.), image_width=640)
    o._active_capture_frame_id, o._active_capture_timestamp = 257, NOW-.01
    o._rknn_pipeline.last_frame_width = 640
    o._rknn_pipeline.last_identity_processing.update(capture_frame_id=257,
        capture_timestamp=NOW-.01)
    o._rknn_pipeline.tracker.identity_bank = SimpleNamespace(
        track_to_uid={1: 1}, _mapped_geometry_conflicts={},
        _geometry_revoked_uids={}, _reacquire_control_suspects={})
    data.clear()
    data.update(uid=0, mapped_uid=1, reason="verified_continuation_recheck",
        identity_recheck_pending=True, identity_control_rejected=True,
        identity_recheck_capture=257, identity_recheck_deadline=NOW+.20,
        bbox_quality_ok=False, bbox_quality_tier="reject", bank_updated=False,
        identity_competition=dict(uid=1, frame_index=o.frame_index,
            candidate_count=1, source_detection_index=0, passed=True),
        identity_continuation=dict(status="hold", source="partial"))
    if conflict: data["candidate_geometry_conflict"] = True
    records[0] = SimpleNamespace(reid_uid=0, track_id=1, time_since_update=0,
                                 x1=20., y1=20., x2=200., y2=460.)
    original = o._validated_visual_observation
    o._update_detector_identity_lease(records, 257, NOW-.01, now=NOW, stale=False)
    return original


def test_existing_narrow_grey_bridge_keeps_only_original_physical_grant(distance_only):
    a, o = distance_only, distance_only.owner
    original = recheck_result(a)
    retained = o._validated_visual_observation
    linear, timing = o._depth30_linear_snapshot, o._depth30_linear_timing
    assert retained.capture == original.capture == 256
    assert retained.timestamp == original.timestamp
    assert retained.validated_at == original.validated_at
    assert retained.expires_at == min(original.expires_at, NOW+.20)
    assert retained.continuation_sample_timestamp == a.stamp
    assert o._fresh_depth_linear_snapshot(1, quiet=True) == linear
    # UID0 cannot mint a new physical authorization, nor erase the old one by
    # translating an ineligible positive calculation into a zero decision.
    a.stamp += .01
    a.frame = replace(a.frame, distance_state=replace(a.frame.distance_state,
        sample_timestamp=a.stamp, observation_timestamp=a.stamp))
    a.shared = replace(a.shared, sample_timestamp=a.stamp)
    assert publish(a) == ([], False)
    assert o._depth30_linear_snapshot is linear and o._depth30_linear_timing is timing
    assert not retained.permits_depth(1, a.stamp, NOW)
    assert o._fresh_depth_linear_snapshot(1, quiet=True) == linear


def test_grey_recheck_with_real_conflict_revokes_visual_forward_qualification(distance_only):
    a = distance_only
    recheck_result(a, conflict=True)
    assert a.owner._validated_visual_observation is False
    assert a.owner._fresh_depth_linear_snapshot(1, quiet=True) is None


def test_legitimate_crop_continuation_from_real_tracker_publishes_full_proof(owner):
    import numpy as np
    from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
    from rk_vision.yolo11 import Detection
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1,
        identity_template_memory_enable=True, identity_template_crosscheck_enable=True,
        identity_appearance_region_safety_enable=True))
    normal, crop = (350., 10., 620., 475.), (361.8, 4.3, 631.1, 476.)
    vector = np.array([1., 0., 0.])
    for i, box in enumerate([normal, normal, normal, normal, crop]):
        records = tracker.update([Detection(box, .94, 0)], [vector],
            partial_features=[vector], partial_feature_sources=["osnet_torso"],
            image_width=640, image_height=480, frame_context=dict(
                control_frame_id=1+i, capture_frame_id=1198+i,
                capture_timestamp=NOW-.20+i*.05))
    assert records[0].reid_uid == 1
    assignment = tracker.identity_bank.last_assignments[1]
    assert assignment["reason"] == "mapped_crop_continuation"
    assert assignment["bank_updated"] is False
    owner._rknn_pipeline = SimpleNamespace(tracker=tracker,
        last_identity_processing=dict(mode="full", full_features_current=True,
                                      capture_frame_id=1202, capture_timestamp=NOW))
    assert owner._update_detector_identity_lease(records, 1202, NOW, now=NOW+.01, stale=False)
    assert isinstance(owner._validated_visual_observation, ValidatedVisualObservation)


def test_gallery_quarantine_and_legacy_missing_quality_do_not_reject_full_control(owner):
    data, records = full_result(owner)
    data.pop("bbox_quality_ok")
    data.update(template_update_quarantined=True, bank_updated=False)
    owner._rknn_pipeline.last_identity_processing.update(capture_frame_id=257,
        capture_timestamp=NOW-.14)
    assert owner._update_detector_identity_lease(records, 257, NOW-.14, now=NOW, stale=False)
    assert isinstance(owner._validated_visual_observation, ValidatedVisualObservation)
