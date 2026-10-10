"""CAP563: a completed empty detector frame is not a negative ReID result.

Real identity publication and paired adapter, no camera/motor construction.
The new detector contract deliberately keeps full_features_current=False.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import HazardState, SensorFrame
from car_control_modular.detector_identity_lease import (
    DetectorIdentityLease, ValidatedVisualObservation, publish_visual_identity_evidence,
)
from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowObservation,
)
from test_lateral_zero_runtime import NOW, owner
from test_short_follow_adapter import paired, process


@pytest.fixture
def missing(paired, monkeypatch):
    a = paired
    process(a)
    a.clock = [NOW]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: a.clock[0])
    o = a.owner
    # Real CAP558 -> CAP563 clocks translated to NOW=100; the original pair
    # still had 141 ms at the physical ownership-exit STOP in the recorded run.
    o._short_follow = ShortFollowController(ShortFollowConfig(enabled=True, depth_ttl_sec=.35))
    o._short_follow.activate(1, 99.6)
    o._short_follow.update(ShortFollowObservation(1, 558, 99.667819393,
        99.815140011, 1.757, .9262201786), 99.97)
    o._short_follow_adapter.controller = o._short_follow
    original = o._short_follow.snapshot()
    a.plan = replace(original.plan, left_rpm=53, right_rpm=37)
    o._short_follow._state = replace(original, plan=a.plan)
    a.proof = ValidatedVisualObservation(1, 4, 558, 99.667819393,
        99.8, 100.167819393, "full")
    a.publication = publish_visual_identity_evidence(o, observation=a.proof, lease=None)
    a.epoch = [42]
    o._depth_async_scheduler = SimpleNamespace(publication_snapshot=lambda: (a.epoch[0], None))
    o._identity_processing_watermark = (558, a.proof.timestamp)
    o._rknn_pipeline = SimpleNamespace(
        last_identity_processing={}, tracker=SimpleNamespace(
            associated_position_contradiction=lambda *_: None))
    o._identity_assignment_debug_for_track = lambda _: dict(uid=1, bbox_quality_ok=True, reason="mapped")
    set_capture(a, 563, NOW-.05)
    return a


def set_capture(a, cap, stamp):
    a.cap, a.stamp = cap, stamp
    a.owner._active_capture_frame_id = cap
    a.owner._active_capture_timestamp = stamp
    a.owner._rknn_pipeline.last_identity_processing = dict(
        mode="full", full_features_current=False, capture_frame_id=cap,
        capture_timestamp=stamp, detector_result_complete=True,
        detector_person_count=0, detector_result_capture_frame_id=cap,
        detector_result_capture_timestamp=stamp)


def deliver(a, records=(), *, stale=False):
    return a.owner._update_detector_identity_lease(list(records), a.cap, a.stamp,
        now=a.clock[0], stale=stale, expected_epoch=42)


def handle_missing(a):
    return a.owner._short_follow_adapter.handle(
        SensorFrame(width=640, height=480, persons=[], capture_frame_id=a.cap,
                    capture_timestamp=a.stamp), None,
        is_fresh_depth=False, control_source="vision", target_steerable=True,
        low_quality_visible=False, now=a.clock[0])


def test_cap563_real_empty_contract_keeps_original_arc_without_full_feature_claim(missing):
    a = missing
    before = a.owner._short_follow.snapshot()
    assert a.owner._rknn_pipeline.last_identity_processing["full_features_current"] is False
    assert deliver(a)
    proof = a.owner._validated_visual_observation
    assert proof == replace(a.proof, continuation_sample_timestamp=a.plan.depth_timestamp,
                           expires_at=a.plan.expires_at)
    assert proof.capture == 558 and proof.validated_at == a.proof.validated_at
    assert a.owner._detector_identity_lease is None
    assert handle_missing(a)
    assert a.owner._short_follow.snapshot() is before
    assert a.owner._short_follow.snapshot().plan is a.plan
    assert (a.plan.left_rpm, a.plan.right_rpm) == (53, 37)
    assert not a.owner._queued_calls
    assert not a.owner._late_visual_current_follow
    assert not a.owner._late_visual_identity_only


def test_repeated_empty_frames_never_roll_plan_proof_or_sample_deadline(missing, caplog):
    a = missing
    assert deliver(a)
    publication = a.owner._visual_identity_evidence
    for cap, offset in ((564, .03), (565, .09), (566, .15)):
        a.clock[0] = NOW+offset
        set_capture(a, cap, a.clock[0]-.01)
        assert deliver(a)
        assert handle_missing(a)
        assert a.owner._short_follow.snapshot().plan is a.plan
        assert a.owner._visual_identity_evidence is publication
    a.clock[0] = a.plan.expires_at
    set_capture(a, 567, a.clock[0]-.01)
    assert deliver(a)
    assert a.owner._validated_visual_observation is False
    assert not handle_missing(a)
    assert not a.owner._short_follow.snapshot().active
    records = [r.message for r in caplog.records if r.message.startswith("short_follow_empty_detector_bridge ")]
    assert len(records) == 4
    assert all("identity_renewed=False depth_renewed=False plan_renewed=False" in r for r in records)


def test_bridge_trace_is_written_outside_control_and_plan_locks(missing, monkeypatch):
    a = missing
    messages = []
    def info(message, *args):
        if message.startswith("short_follow_empty_detector_bridge "):
            assert not a.owner._control_update_lock._is_owned()
            assert not a.owner._short_follow._lock._is_owned()
            messages.append(message % args)
    monkeypatch.setattr(runtime.logger, "info", info)
    assert deliver(a)
    assert len(messages) == 1
    assert "capture_frame_id=563 original_cap=558 uid=1 pair=(53, 37)" in messages[0]
    assert "depth_age_ms=184.9 remaining_ms=165.1" in messages[0]


def test_shorter_existing_detector_lease_is_not_removed_or_extended(missing):
    a = missing
    lease = DetectorIdentityLease(1, 4, 558, 99.81, 559, 99.9, NOW+.11)
    proof = replace(a.proof, capture=559, timestamp=99.9, validated_at=99.92,
                    expires_at=lease.expires_at, kind="detector_continuation")
    publish_visual_identity_evidence(a.owner, observation=proof, lease=lease)
    assert deliver(a)
    assert handle_missing(a)
    assert a.owner._validated_visual_observation.expires_at == lease.expires_at
    assert a.owner._detector_identity_lease is lease
    assert a.owner._short_follow.snapshot().plan is a.plan
    a.clock[0] = lease.expires_at
    assert a.plan.valid(a.clock[0])
    assert not handle_missing(a)


@pytest.mark.parametrize("case", ["duplicate", "older_cap", "older_stamp"])
def test_duplicate_or_reordered_empty_results_do_not_publish_again(missing, case):
    a = missing
    assert deliver(a)
    publication = a.owner._visual_identity_evidence
    cap, stamp = a.cap, a.stamp
    if case == "older_cap": cap -= 1
    if case == "older_stamp": stamp -= .01
    set_capture(a, cap, stamp)
    assert not deliver(a)
    assert a.owner._visual_identity_evidence is publication
    assert a.owner._short_follow.snapshot().plan is a.plan


@pytest.mark.parametrize("field,value", [
    ("detector_result_complete", False), ("detector_result_complete", None),
    ("detector_result_complete", 1), ("detector_person_count", 1),
    ("detector_person_count", False), ("detector_person_count", None),
    ("detector_result_capture_frame_id", 562), ("detector_result_capture_frame_id", 563.),
    ("detector_result_capture_timestamp", 99.94),
    ("capture_frame_id", 562), ("capture_timestamp", 99.94),
    ("mode", "probe"),
])
def test_incomplete_nonempty_or_unbound_detector_result_cannot_preserve_arc(missing, field, value):
    a = missing
    a.owner._rknn_pipeline.last_identity_processing[field] = value
    deliver(a)
    assert a.owner._validated_visual_observation is False
    assert not handle_missing(a)


@pytest.mark.parametrize("case", ["future", "stale_processing", "expired_capture"])
def test_invalid_empty_capture_clock_does_not_borrow_old_live_plan(missing, case):
    a = missing
    if case == "future": set_capture(a, 563, NOW+.01)
    elif case == "expired_capture":
        a.clock[0] = NOW+.05
        set_capture(a, 563, a.clock[0]-.351)
    deliver(a, stale=case == "stale_processing")
    assert a.owner._validated_visual_observation is False
    assert not handle_missing(a)


@pytest.mark.parametrize("case", [
    "no_plan", "expired_plan", "inactive", "epoch_mismatch", "wrong_plan_uid",
    "wrong_proof_uid", "rejected_proof", "expired_proof", "wrong_sample",
    "future_depth", "expired_lease", "rejected_lease",
    "explicit_stop", "shutdown", "not_running", "brake", "park", "stop_action",
    "soft_stop", "search", "owner_search", "cached_hazard", "hazard",
    "publication_epoch", "missing_contradiction_reader", "geometry", "competition",
])
def test_existing_ownership_identity_safety_and_contradictions_remain_required(missing, case):
    a = missing
    o = a.owner
    state = o._short_follow.snapshot()
    if case == "no_plan": o._short_follow._state = replace(state, plan=None)
    elif case == "expired_plan": o._short_follow._state = replace(state, plan=replace(a.plan, expires_at=NOW))
    elif case == "inactive": o._short_follow.deactivate("prior_exit", NOW)
    elif case == "epoch_mismatch": o._short_follow._state = replace(state, epoch=state.epoch+1)
    elif case == "wrong_plan_uid": o._short_follow._state = replace(state, plan=replace(a.plan, uid=2))
    elif case == "future_depth": o._short_follow._state = replace(state, plan=replace(a.plan, depth_timestamp=NOW+.01))
    elif case in {"expired_lease", "rejected_lease"}:
        lease = False if case == "rejected_lease" else DetectorIdentityLease(1, 4, 558, 99.7, 559, 99.8, NOW)
        publish_visual_identity_evidence(o, observation=a.proof, lease=lease)
    elif case in {"wrong_proof_uid", "rejected_proof", "expired_proof", "wrong_sample"}:
        proof = {"wrong_proof_uid": replace(a.proof, uid=2), "rejected_proof": False,
                 "expired_proof": replace(a.proof, expires_at=NOW),
                 "wrong_sample": replace(a.proof, continuation_sample_timestamp=NOW-.01)}[case]
        publish_visual_identity_evidence(o, observation=proof, lease=None)
    elif case == "explicit_stop": o._explicit_stop_requested = True
    elif case == "shutdown": o._runtime_shutdown_requested = True
    elif case == "not_running": o.running = False
    elif case == "brake": o._brake_hold_active = True
    elif case == "park": o._near_yaw_park_request = object()
    elif case == "stop_action": o.stop_action_execution = True
    elif case == "soft_stop": o._use_soft_stop_next = True
    elif case == "search": o._follow_controller.search_state = "searching"
    elif case == "owner_search": o.search_state = "searching"
    elif case == "cached_hazard": o._depth_roi_safety_clear = False
    elif case == "hazard": o._current_hazard_state_for_controller = lambda: HazardState(active=True)
    elif case == "publication_epoch": a.epoch[0] += 1
    elif case == "missing_contradiction_reader": o._rknn_pipeline.tracker.associated_position_contradiction = None
    else: o._rknn_pipeline.tracker.associated_position_contradiction = lambda *_: case
    deliver(a)
    assert o._validated_visual_observation is False


def test_identity_publication_does_not_poll_ir_or_advance_debounce(missing):
    a = missing
    a.owner._get_obstacle_status = lambda: pytest.fail("IR poll in identity publication")
    assert deliver(a)
    assert handle_missing(a)


@pytest.mark.parametrize("uid,features", [(0, True), (0, False), (1, False)])
def test_uid0_rejection_and_detected_feature_failure_are_not_empty_frames(missing, uid, features):
    a = missing
    meta = a.owner._rknn_pipeline.last_identity_processing
    meta.update(full_features_current=features, detector_person_count=1)
    a.owner._identity_assignment_debug_for_track = lambda _: dict(
        uid=uid, mapped_uid=1, identity_control_rejected=True,
        bbox_quality_ok=False, reason="secondary_evidence_unavailable")
    deliver(a, [SimpleNamespace(reid_uid=uid, track_id=4, time_since_update=0, class_id=0)])
    assert a.owner._validated_visual_observation is False


def test_new_depth_cannot_use_narrowed_missing_proof_but_new_full_can(missing):
    a = missing
    assert deliver(a)
    a.clock[0] = NOW+.02
    frame = SensorFrame(width=640, height=480, persons=[a.target], distance_m=2.3,
        distance_state=replace(a.owner._distance_runtime.get_frame_distance_state(),
                               sample_timestamp=NOW+.01),
        capture_frame_id=563, capture_timestamp=NOW-.05)
    assert a.owner._short_follow_adapter.handle(frame, a.target,
        is_fresh_depth=True, control_source="depth30", target_steerable=True,
        low_quality_visible=False, now=a.clock[0])
    assert a.owner._short_follow.snapshot().plan is a.plan
    assert not a.owner._validated_visual_observation.permits_depth(1, NOW+.01, a.clock[0])
    set_capture(a, 565, NOW+.01)
    a.owner._rknn_pipeline.last_identity_processing.update(
        full_features_current=True, detector_person_count=1)
    assert deliver(a, [SimpleNamespace(reid_uid=1, track_id=4, time_since_update=0, class_id=0)])
    proof = a.owner._validated_visual_observation
    assert proof.capture == 565 and proof.continuation_sample_timestamp is None
    frame = replace(frame, capture_frame_id=565, capture_timestamp=NOW+.01,
                    distance_state=replace(frame.distance_state, sample_timestamp=NOW+.015))
    assert a.owner._short_follow_adapter.handle(frame, a.target,
        is_fresh_depth=True, control_source="depth30", target_steerable=True,
        low_quality_visible=False, now=a.clock[0])
    replacement = a.owner._short_follow.snapshot().plan
    assert replacement is not a.plan and replacement.depth_timestamp == NOW+.015


def test_normal_depth_commit_during_empty_review_binds_latest_same_epoch_plan(missing):
    a = missing
    committed = []
    def review(*_):
        plan = a.owner._short_follow.update(ShortFollowObservation(
            1, 558, a.plan.capture_timestamp, NOW-.01, 1.8, .9), NOW)
        assert plan is not None and plan is not a.plan
        committed.append(plan)
        return None
    a.owner._rknn_pipeline.tracker.associated_position_contradiction = review
    assert deliver(a)
    assert handle_missing(a)
    latest = committed[0]
    assert a.owner._short_follow.snapshot().plan is latest
    assert latest.epoch == a.plan.epoch
    proof = a.owner._validated_visual_observation
    assert proof.continuation_sample_timestamp == latest.depth_timestamp
    assert proof.expires_at == min(a.proof.expires_at, latest.expires_at)
    assert proof.capture == a.proof.capture and proof.validated_at == a.proof.validated_at
    assert not a.owner._queued_calls


@pytest.mark.parametrize("new_plan", [False, True])
@pytest.mark.parametrize("phase", ["before_read", "during_review"])
def test_later_full_publication_supersedes_empty_without_rejection_or_narrowing(
        missing, monkeypatch, caplog, new_plan, phase):
    a = missing
    published = []
    def complete_full():
        a.owner._rknn_pipeline.last_identity_processing = dict(
            mode="full", full_features_current=True,
            capture_frame_id=565, capture_timestamp=NOW-.02)
        assert a.owner._update_detector_identity_lease(
            [SimpleNamespace(reid_uid=1, track_id=4, time_since_update=0, class_id=0)],
            565, NOW-.02, now=NOW, stale=False, expected_epoch=42)
        if new_plan:
            assert a.owner._short_follow.update(ShortFollowObservation(
                1, 565, NOW-.02, NOW-.01, 2.1, .8), NOW) is not None
        published.append((a.owner._visual_identity_evidence, a.owner._short_follow.snapshot().plan))
    if phase == "during_review":
        a.owner._rknn_pipeline.tracker.associated_position_contradiction = lambda *_: complete_full()
    else:
        real_read = runtime.read_visual_identity_evidence
        def read(o):
            if not published:
                complete_full()
            return real_read(o)
        monkeypatch.setattr(runtime, "read_visual_identity_evidence", read)
    assert deliver(a)  # Never send the superseded result through stale handling.
    publication, plan = published[0]
    assert a.owner._visual_identity_evidence is publication
    assert publication.observation.capture == 565
    assert publication.observation.continuation_sample_timestamp is None
    assert publication.observation.expires_at == pytest.approx(NOW+.48)
    assert publication.lease is None
    assert a.owner._short_follow.snapshot().plan is plan
    assert handle_missing(a)
    assert a.owner._short_follow.snapshot().plan is plan
    assert not a.owner._queued_calls
    assert "short_follow_empty_detector_superseded capture_frame_id=563 uid=1 newer_full_cap=565" in caplog.text
    assert not any(r.message.startswith("short_follow_empty_detector_bridge ") for r in caplog.records)


@pytest.mark.parametrize("fault", ["same_capture", "same_timestamp", "no_watermark", "not_full", "future", "rejected"])
def test_unproven_identity_publication_change_is_not_later_full_permission(missing, fault):
    a = missing
    def review(*_):
        proof = ValidatedVisualObservation(1, 4, 565, NOW-.02, NOW, NOW+.48, "full")
        if fault == "same_capture": proof = replace(proof, capture=a.cap)
        elif fault == "same_timestamp": proof = replace(proof, timestamp=a.stamp)
        elif fault == "not_full": proof = replace(proof, kind="detector_continuation")
        elif fault == "future": proof = replace(proof, timestamp=NOW+.01)
        elif fault == "rejected": proof = False
        if fault != "no_watermark": a.owner._identity_processing_watermark = (565, NOW-.02)
        publish_visual_identity_evidence(a.owner, observation=proof, lease=None)
        return None
    a.owner._rknn_pipeline.tracker.associated_position_contradiction = review
    deliver(a)
    assert a.owner._validated_visual_observation is False


@pytest.mark.parametrize("change", ["epoch", "expired", "explicit_stop", "search", "uid", "newer_capture", "narrow_sample"])
def test_no_narrowed_proof_commit_after_concurrent_scope_change(missing, change):
    a = missing
    if change == "narrow_sample":
        publish_visual_identity_evidence(a.owner, observation=replace(
            a.proof, continuation_sample_timestamp=a.plan.depth_timestamp), lease=None)
    def review(*_):
        if change == "epoch": a.owner._short_follow.revoke("concurrent_stop", NOW)
        elif change == "expired": a.clock[0] = a.plan.expires_at
        elif change == "explicit_stop": a.owner._explicit_stop_requested = True
        elif change == "search": a.owner._follow_controller.search_state = "searching"
        elif change == "uid": a.owner._short_follow.activate(2, NOW)
        else:
            state = a.owner._short_follow.snapshot()
            plan = (replace(a.plan, capture_id=a.cap, capture_timestamp=a.stamp)
                    if change == "newer_capture" else replace(a.plan, depth_timestamp=NOW-.01))
            a.owner._short_follow._state = replace(state, plan=plan)
        return None
    a.owner._rknn_pipeline.tracker.associated_position_contradiction = review
    deliver(a)
    assert a.owner._validated_visual_observation is False
