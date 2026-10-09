"""Late processing may refresh identity, never the old yaw/depth deadlines."""
import ast
from dataclasses import replace
import inspect
import textwrap
import threading
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import DepthTargetObservation, PersonTarget, HazardState
from car_control_modular.depth_async_scheduler import DepthAsyncScheduler
from car_control_modular.detector_identity_lease import DetectorIdentityLease
from test_late_visual_independent_depth import late, authority300, setup, owner
from test_depth_authority_250 import advance, decide_commit
from test_lateral_zero_runtime import _intent


@pytest.fixture
def current_full(late, monkeypatch):
    a, o = late, late.owner
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", .5)
    a.cap, a.capture = 216, a.stamp - .010
    record = a.records[0]
    bbox = (240., 40., 400., 440.)
    record.class_id, record.score, record.area = runtime.PERSON_CLASS_ID, .95, 64000.
    record.x1, record.y1, record.x2, record.y2 = bbox
    a.data.update(mapped_uid=1, reason="mapped", bbox_quality_tier="normal")
    a.observation = dict(raw_track_id=record.track_id, uid=1, display_bbox=bbox,
        detector_bbox=bbox, assignment=a.data,
        sample_metadata=dict(capture_frame_id=a.cap, capture_timestamp=a.capture,
                             source_detection_index=0, is_fresh=True))
    o._rknn_pipeline.tracker.last_identity_observations = [a.observation]
    o._rknn_pipeline.last_identity_processing.update(
        capture_frame_id=a.cap, capture_timestamp=a.capture)
    o._active_capture_frame_id, o._active_capture_timestamp = a.cap, a.capture
    # Original fast proof was valid at exposure but expired during full ReID.
    expired_at = a.clock.now - .016
    a.previous = replace(a.visual, kind="detector_continuation", expires_at=expired_at,
                         validated_at=a.capture-.010)
    o._validated_visual_observation = a.previous
    o._detector_identity_lease = DetectorIdentityLease(
        uid=1, track_id=record.track_id, verified_capture=210,
        verified_timestamp=a.visual.timestamp-.010,
        observation_capture=a.visual.capture, observation_timestamp=a.visual.timestamp,
        expires_at=expired_at)
    old_target = PersonTarget(bbox, 1, .95, 64000., depth_observation=DepthTargetObservation(
        bbox=bbox, target_id=1, raw_track_id=record.track_id,
        capture_frame_id=a.visual.capture, capture_timestamp=a.visual.timestamp))
    o._depth_async_scheduler = DepthAsyncScheduler()
    o._longitudinal_context = o._depth_async_scheduler.submit(dict(
        published_ts=a.stamp, frame_index=213, width=640, height=480,
        person_targets=(old_target,), capture_frame_id=a.visual.capture,
        capture_timestamp=a.visual.timestamp, target_id=1, target_steerable=True),
        now=a.clock.now).as_dict()
    o._depth_async_scheduler.worker_tick(now=a.clock.now)
    o._longitudinal_thread = SimpleNamespace(is_alive=lambda: True)
    o._longitudinal_wake_event = threading.Event()
    o._depth_roi_safety_clear = True
    o._action_runtime_started = True
    o._follow_controller.last_selected_target = old_target
    a.epoch = o._depth_async_scheduler.publication_snapshot()[0]
    return a


def main_identity_publication(a, *, processing=.238):
    """Execute the REAL main callsite block, not a copied adapter policy."""
    method = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.process_external_frame)))
    statements = method.body[0].body
    start = next(i for i, node in enumerate(statements) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "stale_result_discarded" for t in node.targets))
    end = next(i for i in range(start+1, len(statements)) if isinstance(statements[i], ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "evidence_getter" for t in statements[i].targets))
    block = ast.fix_missing_locations(ast.Module(body=statements[start:end], type_ignores=[]))
    local = dict(self=a.owner, records=a.records, current_capture_id=a.cap,
        frame_received_ts=a.capture, vision_finished_mono=a.clock.now,
        vision_result_age_sec=processing, depth_publication_epoch=a.epoch, width=640, height=480)
    exec(compile(block, inspect.getsourcefile(runtime.PersonTracker), "exec"), vars(runtime), local)
    return local


def handle(a):
    return a.owner._handle_stale_vision_result(width=640, result_age_sec=.238)


def test_real_main_publishes_late_identity_and_roi_but_keeps_old_motion_deadlines(current_full):
    a, o = current_full, current_full.owner
    old_depth = o._depth30_linear_snapshot
    old_yaw = o._lateral_intent_store.snapshot()
    old_clock = o._last_vision_control_ts
    assert not a.previous.live(1, a.clock.now)
    assert a.previous.timestamp < a.capture < a.previous.expires_at
    local = main_identity_publication(a)
    assert local["stale_result_discarded"] is True
    proof = o._validated_visual_observation
    assert proof.capture == a.cap and proof.timestamp == a.capture and proof.kind == "full"
    assert proof.expires_at == pytest.approx(a.capture + .5)
    assert o._detector_identity_lease is None
    assert o._late_visual_identity_roi_published
    _, roi = o._depth_async_scheduler.publication_snapshot()
    assert (roi.capture_frame_id, roi.capture_timestamp) == (a.cap, a.capture)
    assert o._longitudinal_wake_event.is_set()
    assert o._depth30_linear_snapshot is old_depth
    assert o._depth30_linear_timing.depth_expires_at == a.deadline
    assert o._lateral_intent_store.snapshot() is old_yaw
    assert handle(a)
    assert o._validated_visual_observation is proof
    assert o.search_state == "none" and not o._queued_calls
    assert o._last_vision_control_ts == old_clock
    assert o._lateral_intent_store.snapshot() is old_yaw
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][0] > 0 and a.backend.pairs[-1][1] < 0
    advance(a, a.stamp + .301)
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][:2] == (0, 0)
    assert o._depth30_linear_timing.depth_expires_at == a.deadline


@pytest.mark.parametrize("fault", ["uid0", "rejected", "pending", "quality", "track",
    "ambiguous", "predicted", "features", "fast", "source_capture", "source_stamp",
    "duplicate", "old_stamp", "too_old", "future", "expired_at_exposure", "grey_proof",
    "no_previous", "previous_unvalidated_at_exposure", "stop", "shutdown", "search",
    "controller_search", "unsafe", "brake", "reacquire"])
def test_bad_identity_or_scope_cannot_use_processing_time_exception(current_full, fault):
    a, o = current_full, current_full.owner
    if fault == "uid0": a.records[0].reid_uid = 0
    elif fault == "rejected": a.data["identity_control_rejected"] = True
    elif fault == "pending": a.data["identity_recheck_pending"] = True
    elif fault == "quality": a.data["bbox_quality_ok"] = False
    elif fault == "track": a.records[0].track_id = 2
    elif fault == "ambiguous": a.records.append(a.records[0])
    elif fault == "predicted": a.records[0].time_since_update = 1
    elif fault == "features": o._rknn_pipeline.last_identity_processing["full_features_current"] = False
    elif fault == "fast": o._rknn_pipeline.last_identity_processing["mode"] = "detector_continuation"
    elif fault == "source_capture": o._rknn_pipeline.last_identity_processing["capture_frame_id"] -= 1
    elif fault == "source_stamp": o._rknn_pipeline.last_identity_processing["capture_timestamp"] -= .001
    elif fault == "duplicate": o._identity_processing_watermark = (a.cap, a.capture)
    elif fault == "old_stamp": o._identity_processing_watermark = (a.cap-1, a.capture+.001)
    elif fault == "too_old": advance(a, a.capture+.350)
    elif fault == "future": advance(a, a.capture-.001)
    elif fault == "expired_at_exposure": o._validated_visual_observation = replace(a.previous, expires_at=a.capture)
    elif fault == "grey_proof": o._validated_visual_observation = replace(a.previous, continuation_sample_timestamp=a.stamp)
    elif fault == "no_previous": o._validated_visual_observation = False
    elif fault == "previous_unvalidated_at_exposure":
        o._validated_visual_observation = replace(a.previous, validated_at=a.capture+.001)
    elif fault == "stop": o._explicit_stop_requested = True
    elif fault == "shutdown": o._runtime_shutdown_requested = True
    elif fault == "search": o.search_state = "searching"
    elif fault == "controller_search": o._follow_controller.search_state = "searching"
    elif fault == "unsafe": o._depth_roi_safety_clear = False
    elif fault == "brake": o._brake_hold_active = True
    elif fault == "reacquire": o._reacquire_depth_pending = True
    previous_roi = o._depth_async_scheduler.publication_snapshot()[1]
    local = main_identity_publication(a)
    assert local["stale_result_discarded"]
    assert not o._late_visual_identity_only
    assert not o._late_visual_identity_roi_published
    assert o._depth_async_scheduler.publication_snapshot()[1] is previous_roi


def test_current_merged_danger_cannot_be_hidden_by_prior_safe_snapshot(current_full):
    a, o = current_full, current_full.owner
    o._bunker_runtime = SimpleNamespace(config=SimpleNamespace(enabled=True, mode="merged", class_ids=(4,)))
    a.records.append(SimpleNamespace(class_id=4, reid_uid=0, track_id=99, time_since_update=0))
    assert o._depth_roi_safety_clear
    main_identity_publication(a)
    assert not o._late_visual_identity_only and o._validated_visual_observation is False


@pytest.mark.parametrize("failure", ["no_geometry", "weak_tier", "worker_dead", "wrong_raw", "stop_at_submit"])
def test_failed_actual_roi_publication_removes_identity_only_exception(current_full, monkeypatch, failure):
    a, o = current_full, current_full.owner
    if failure == "no_geometry": o._rknn_pipeline.tracker.last_identity_observations = []
    elif failure == "weak_tier": a.data["bbox_quality_tier"] = "weak"
    elif failure == "worker_dead": o._longitudinal_thread = SimpleNamespace(is_alive=lambda: False)
    elif failure == "wrong_raw": a.observation["raw_track_id"] += 1
    else:
        original = o._depth_async_scheduler.submit
        def stopped(*args, **kwargs):
            o._explicit_stop_requested = True
            return original(*args, **kwargs)
        monkeypatch.setattr(o._depth_async_scheduler, "submit", stopped)
    main_identity_publication(a)
    assert o._validated_visual_observation is False
    assert o._detector_identity_lease is False
    assert not o._late_visual_identity_only
    assert not o._late_visual_identity_roi_published
    handle(a)
    assert o._depth30_linear_snapshot is None


@pytest.mark.parametrize("when", ["before_update", "during_geometry", "before_handler"])
def test_stop_epoch_cannot_be_laundered_by_same_uid_return(current_full, monkeypatch, when):
    a, o = current_full, current_full.owner
    scheduler = o._depth_async_scheduler
    old_context = o._longitudinal_context
    def revoke_and_return():
        scheduler.revoke("explicit_stop")
        scheduler.submit(old_context, now=a.clock.now)
    if when == "before_update": revoke_and_return()
    elif when == "during_geometry":
        original = runtime.resolve_depth_target_observation
        def resolve(**kwargs):
            result = original(**kwargs)
            revoke_and_return()
            return result
        monkeypatch.setattr(runtime, "resolve_depth_target_observation", resolve)
    main_identity_publication(a)
    if when == "before_handler": revoke_and_return()
    else: assert not o._late_visual_identity_only
    handle(a)
    assert o._depth30_linear_snapshot is None
    assert not o._late_visual_identity_only


def test_helper_alone_cannot_claim_successful_readonly_roi_publication(current_full):
    a, o = current_full, current_full.owner
    assert not o._update_detector_identity_lease(a.records, a.cap, a.capture,
        now=a.clock.now, stale=True, expected_epoch=a.epoch)
    assert o._late_visual_identity_only and not o._late_visual_identity_roi_published
    handle(a)
    assert o._depth30_linear_snapshot is None


def test_expired_depth_is_not_resurrected_but_next_new_sample_can_be_admitted(current_full):
    a, o = current_full, current_full.owner
    advance(a, a.stamp + .301)
    assert o._fresh_depth_linear_snapshot(1) is None
    main_identity_publication(a)
    assert o._late_visual_identity_only
    handle(a)
    assert o.search_state == "none"
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][:2] == (0, 0)
    original_deadline = o._depth30_linear_timing.depth_expires_at
    decision, actions, admitted = decide_commit(a, a.frame(3., rpm=40., stamp=a.clock.now-.01))
    assert admitted
    assert o._depth30_linear_timing.depth_expires_at > original_deadline
    assert o._depth30_linear_snapshot[3] == pytest.approx(a.clock.now-.01)


def test_new_identity_does_not_replay_expired_old_yaw_in_real_writer(current_full):
    a, o = current_full, current_full.owner
    old_yaw = _intent(o, published_at=a.stamp+.190, valid_until=a.stamp+.240,
        capture_frame_id=a.previous.capture, capture_timestamp=a.previous.timestamp,
        decision_capture_frame_id=a.previous.capture, mode="forward",
        near_distance_mode=False, base_percent=a.original[1],
        base_rpm=a.original[1]*2, initial_correction_rpm=5)
    advance(a, a.stamp+.221)
    assert o._has_fresh_lateral_yaw(1)
    assert o._follow_wheel_axes(a.clock.now)[3] == 5
    a.action._service_follow_wheels()
    before = a.backend.pairs[-1][:2]
    assert before[0] > 0 and before[1] < 0 and before[0] != -before[1]
    advance(a, a.stamp+.281)
    assert not old_yaw.valid(a.clock.now)
    main_identity_publication(a)
    assert o._validated_visual_observation.live(1, a.clock.now)
    handle(a)
    assert o._lateral_intent_store.snapshot() is old_yaw
    assert not o._has_fresh_lateral_yaw(1)
    uid, revision, base, yaw = o._follow_wheel_axes(a.clock.now)
    assert base > 0 and yaw == 0
    a.action._service_follow_wheels()
    after = a.backend.pairs[-1][:2]
    assert after[0] > 0 and after[1] < 0 and after[0] == -after[1]
    assert o._depth30_linear_timing.depth_expires_at == a.deadline


def test_next_normal_full_result_and_new_depth_continue_after_identity_only_handler(current_full):
    a, o = current_full, current_full.owner
    main_identity_publication(a)
    handle(a)
    first = o._validated_visual_observation
    advance(a, a.clock.now+.04)
    a.cap += 1
    a.capture = a.clock.now-.02
    o._active_capture_frame_id, o._active_capture_timestamp = a.cap, a.capture
    o._rknn_pipeline.last_identity_processing.update(
        capture_frame_id=a.cap, capture_timestamp=a.capture)
    a.observation["sample_metadata"].update(capture_frame_id=a.cap, capture_timestamp=a.capture)
    local = main_identity_publication(a, processing=.06)
    assert not local["stale_result_discarded"]
    assert not o._late_visual_identity_only and not o._late_visual_identity_roi_published
    assert o._validated_visual_observation.capture == a.cap
    assert o._validated_visual_observation.timestamp > first.timestamp
    assert o._depth_async_scheduler.publication_snapshot()[1].capture_frame_id == a.cap
    _, actions, admitted = decide_commit(a, a.frame(3., rpm=40., stamp=a.clock.now-.01))
    assert admitted and any(action.kind == "forward" for action in actions)
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][0] > 0 and a.backend.pairs[-1][1] < 0


def test_merged_danger_also_vetoes_legacy_preservation_of_still_live_proof(current_full):
    a, o = current_full, current_full.owner
    o._validated_visual_observation = replace(a.previous, kind="full", expires_at=a.clock.now+.1)
    o._detector_identity_lease = None
    o._bunker_runtime = SimpleNamespace(config=SimpleNamespace(enabled=True, mode="merged", class_ids=(4,)))
    a.records.append(SimpleNamespace(class_id=4, reid_uid=0, track_id=99, time_since_update=0))
    assert o._depth_roi_safety_clear
    assert o._fresh_depth_linear_snapshot(1) is not None
    main_identity_publication(a)
    assert o._late_visual_preserved_evidence is None
    handle(a)
    assert o._depth30_linear_snapshot is None


@pytest.mark.parametrize("failure", ["stop", "hazard", "obstacle", "reacquire", "proof_expired"])
def test_live_safety_is_rechecked_after_actual_identity_and_roi_publication(current_full, failure):
    a, o = current_full, current_full.owner
    main_identity_publication(a)
    assert o._late_visual_identity_roi_published
    if failure == "stop": o._explicit_stop_requested = True
    elif failure == "hazard": o._current_hazard_state_for_controller = lambda: HazardState(active=True)
    elif failure == "obstacle": o._get_obstacle_status = lambda: dict(front=True)
    elif failure == "reacquire": o._reacquire_depth_pending = True
    else: advance(a, o._validated_visual_observation.expires_at+.001)
    handle(a)
    assert o._depth30_linear_snapshot is None
    assert not o._late_visual_identity_only and not o._late_visual_identity_roi_published
    if failure == "stop": assert o._explicit_stop_requested
