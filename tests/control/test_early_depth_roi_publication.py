"""CAP223: full identity -> ROI mailbox must not queue behind motor/control.

Real identity adapter, immutable geometry resolver and Depth transaction loop;
only inference, sensors and wheel IO are replaced. No hardware is opened.
"""
import ast
from dataclasses import replace
import inspect
import textwrap
import threading
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.astra_depth import AstraDepthRuntime
from car_control_modular.control_types import HazardState
from test_depth_async_runtime import async_scene, start_loop, join_loop
from test_depth_optimistic_transaction import scene
from test_turn_depth_scheduling import owner, NOW
from test_visual_depth_optimistic_transaction import visual


@pytest.fixture
def early(async_scene):
    o, camera, distance, clock = async_scene
    old = o._longitudinal_context
    cap, stamp = old["capture_frame_id"] + 1, old["capture_timestamp"] + .02
    target = old["person_targets"][0]
    raw = target.depth_observation.raw_track_id
    data = dict(uid=1, mapped_uid=1, reason="mapped", bbox_quality_ok=True,
                bbox_quality_tier="normal")
    observation = dict(raw_track_id=raw, uid=1, display_bbox=target.bbox,
        detector_bbox=target.depth_observation.bbox, assignment=data,
        sample_metadata=dict(capture_frame_id=cap, capture_timestamp=stamp,
                             source_detection_index=0, is_fresh=True))
    record = SimpleNamespace(reid_uid=1, track_id=raw, time_since_update=0,
        class_id=runtime.PERSON_CLASS_ID, score=.95, area=target.area,
        x1=target.bbox[0], y1=target.bbox[1], x2=target.bbox[2], y2=target.bbox[3])
    o._rknn_pipeline = SimpleNamespace(
        tracker=SimpleNamespace(last_identity_observations=[observation]),
        last_identity_processing=dict(mode="full", full_features_current=True,
                                      capture_frame_id=cap, capture_timestamp=stamp))
    o._identity_assignment_debug_for_track = lambda track: data
    o._active_capture_frame_id, o._active_capture_timestamp = cap, stamp
    o.frame_index += 1
    o._longitudinal_wake_event = threading.Event()
    o._depth_roi_safety_clear = True
    epoch = o._depth_async_scheduler.publication_snapshot()[0]
    return SimpleNamespace(o=o, camera=camera, distance=distance, clock=clock,
        cap=cap, stamp=stamp, data=data, observation=observation, records=[record], epoch=epoch)


def validate(a):
    return a.o._update_detector_identity_lease(a.records, a.cap, a.stamp,
                                             now=a.clock[0], stale=False)


def publish(a):
    return a.o._publish_validated_depth_observation(
        a.records, 640, 480, a.cap, a.stamp, expected_epoch=a.epoch)


def run_one_depth(a):
    o = a.o
    o._longitudinal_wake_event = SimpleNamespace(clear=lambda: None,
        set=lambda: None, wait=lambda duration: o._longitudinal_stop_event.set())
    thread, failures = start_loop(o)
    thread.join(3.)
    join_loop(o, thread, failures)


def test_identity_and_roi_publish_finish_while_another_thread_owns_control_lock(early):
    a, o = early, early.o
    old_context = o._longitudinal_context
    old_grant, old_timing = o._depth30_linear_snapshot, o._depth30_linear_timing
    done, failures = threading.Event(), []

    def inference_complete():
        try:
            assert validate(a)
            assert publish(a)
        except BaseException as exc:
            failures.append(exc)
        finally:
            done.set()

    with o._control_update_lock:
        worker = threading.Thread(target=inference_complete)
        worker.start()
        completed = done.wait(2.)
        if completed:
            assert o._longitudinal_wake_event.is_set()
            _, latest = o._depth_async_scheduler.publication_snapshot()
            assert latest.capture_frame_id == a.cap
            assert latest.capture_timestamp == a.stamp
            assert latest.person_targets[0].depth_observation.raw_track_id == a.records[0].track_id
            assert not o.calls and a.camera._last_accepted_ts == 0
            assert o._longitudinal_context is old_context
            assert o._depth30_linear_snapshot is old_grant
            assert o._depth30_linear_timing is old_timing
    worker.join(2.)
    assert completed, "ROI publication waited behind the controller lock"
    assert not worker.is_alive() and not failures


def test_worker_uses_early_mailbox_even_when_compatibility_roi_has_expired(early):
    a, o = early, early.o
    # Legacy visual/lateral work has not yet republished its compatibility ROI.
    o._longitudinal_context = dict(o._longitudinal_context, capture_timestamp=NOW-.251)
    assert validate(a) and publish(a)
    run_one_depth(a)
    assert len(o.calls) == 1
    call = o.calls[0]
    assert call["control_source"] == "depth30"
    assert call["evidence_capture_frame_id"] == a.cap
    assert call["evidence_capture_timestamp"] == a.stamp
    assert call["prepared_depth"].measurement.observation_sample_timestamp == NOW-.02
    assert a.camera._last_accepted_ts == NOW-.02


@pytest.mark.parametrize("fault", ["stop", "shutdown", "stopped", "brake", "uid", "search",
    "controller_search", "reacquire", "unknown_state", "park", "not_started", "no_previous",
    "worker_dead", "hazard", "obstacle", "capture_changed", "capture_time_changed"])
def test_safety_and_stable_identity_prerequisites_do_not_publish(early, fault):
    a, o = early, early.o
    assert validate(a)
    old = o._depth_async_scheduler.publication_snapshot()
    if fault == "stop": o._explicit_stop_requested = True
    elif fault == "shutdown": o._runtime_shutdown_requested = True
    elif fault == "stopped": o.running = False
    elif fault == "brake": o._brake_hold_active = True
    elif fault == "uid": o._follow_controller.active_target_id = 2
    elif fault == "search": o.search_state = "searching"
    elif fault == "controller_search": o._follow_controller.search_state = "searching"
    elif fault == "reacquire": o._reacquire_depth_pending = True
    elif fault == "unknown_state": o._vision_control_state = "lost_confirming"
    elif fault == "park": o._near_yaw_park_request = object()
    elif fault == "not_started": o._action_runtime_started = False
    elif fault == "no_previous": o._follow_controller.last_selected_target = None
    elif fault == "worker_dead": o._longitudinal_thread = SimpleNamespace(is_alive=lambda: False)
    elif fault in {"hazard", "obstacle"}: o._depth_roi_safety_clear = False
    elif fault == "capture_changed": o._active_capture_frame_id += 1
    elif fault == "capture_time_changed": o._active_capture_timestamp += .001
    assert not publish(a)
    assert o._depth_async_scheduler.publication_snapshot() == old
    assert not o.calls and a.camera._last_accepted_ts == 0


@pytest.mark.parametrize("fault", ["uid0", "rejected", "pending", "quality", "excluded",
    "ambiguous", "predicted", "wrong_class", "low_score", "no_provenance",
    "wrong_raw", "wrong_detector_stamp", "wrong_detector_capture", "weak_region"])
def test_identity_and_detector_provenance_must_be_formally_accepted(early, fault):
    a, o = early, early.o
    record = a.records[0]
    if fault == "uid0": record.reid_uid = 0
    elif fault == "rejected": a.data["identity_control_rejected"] = True
    elif fault == "pending": a.data["identity_recheck_pending"] = True
    elif fault == "quality": a.data["bbox_quality_ok"] = False
    elif fault == "excluded": a.data["search_excluded"] = True
    elif fault == "ambiguous": a.records.append(record)
    elif fault == "predicted": record.time_since_update = 1
    elif fault == "wrong_class": record.class_id = runtime.PERSON_CLASS_ID+1
    elif fault == "low_score": record.score = .01
    elif fault == "no_provenance": o._rknn_pipeline.tracker.last_identity_observations = []
    elif fault == "wrong_raw": a.observation["raw_track_id"] += 1
    elif fault == "wrong_detector_stamp": a.observation["sample_metadata"]["capture_timestamp"] -= .01
    elif fault == "wrong_detector_capture": a.observation["sample_metadata"]["capture_frame_id"] -= 1
    elif fault == "weak_region": a.data["bbox_quality_tier"] = "weak"
    old = o._depth_async_scheduler.publication_snapshot()
    validate(a)
    assert not publish(a)
    assert o._depth_async_scheduler.publication_snapshot() == old


@pytest.mark.parametrize("change", ["revoke", "same_uid_reacquire", "fallback", "stop_at_submit",
                                    "uid_at_submit", "identity_at_submit"])
def test_revocation_between_identity_and_mailbox_cannot_resurrect_permission(early, change, monkeypatch):
    a, o = early, early.o
    assert validate(a)
    scheduler = o._depth_async_scheduler
    if change == "revoke": scheduler.revoke("identity_rejected")
    elif change == "same_uid_reacquire":
        ctx = o._longitudinal_context
        scheduler.revoke("identity_rejected")
        scheduler.submit(ctx, now=a.clock[0])
    elif change == "fallback":
        ticket = scheduler.begin_fallback(now=a.clock[0])
        scheduler.finish(ticket, now=a.clock[0])
    else:
        original = scheduler.submit
        def racing_submit(*args, **kwargs):
            if change == "stop_at_submit": o._explicit_stop_requested = True
            elif change == "uid_at_submit": o._follow_controller.active_target_id = 2
            else: o._validated_visual_observation = False
            return original(*args, **kwargs)
        monkeypatch.setattr(scheduler, "submit", racing_submit)
    old = scheduler.publication_snapshot()
    assert not publish(a)
    assert scheduler.publication_snapshot() == old


def test_epoch_change_during_geometry_resolution_rejects_late_publication(early, monkeypatch):
    a, o = early, early.o
    assert validate(a)
    original = runtime.resolve_depth_target_observation
    def resolve(**kwargs):
        result = original(**kwargs)
        o._depth_async_scheduler.revoke("stop_during_resolution")
        return result
    monkeypatch.setattr(runtime, "resolve_depth_target_observation", resolve)
    assert not publish(a)
    assert o._depth_async_scheduler.publication_snapshot()[1] is None


@pytest.mark.parametrize("age", [.251, .501])
def test_late_frame_does_not_relabel_its_original_capture(early, age):
    a, o = early, early.o
    assert validate(a)
    a.clock[0] = a.stamp+age
    assert not publish(a)


def test_duplicate_and_old_frames_do_not_refresh_mailbox_or_motor_deadlines(early):
    a, o = early, early.o
    assert validate(a) and publish(a)
    original = o._depth_async_scheduler.publication_snapshot()[1]
    timing, grant = o._depth30_linear_timing, o._depth30_linear_snapshot
    a.clock[0] += .01
    assert not validate(a)
    # Even an erroneous repeated callback cannot replace the immutable stamp.
    publish(a)
    assert o._depth_async_scheduler.publication_snapshot()[1] is original
    assert o._depth30_linear_timing is timing and o._depth30_linear_snapshot is grant
    a.cap -= 1
    assert not publish(a)


def test_narrow_identity_recheck_bridge_cannot_feed_a_new_depth_sample(early):
    a, o = early, early.o
    assert validate(a)
    o._validated_visual_observation = replace(o._validated_visual_observation,
                                              continuation_sample_timestamp=NOW-.02)
    assert not publish(a)


def test_observation_publication_does_not_read_hardware_or_advance_safety_counters(early):
    a, o = early, early.o
    o._get_obstacle_status = lambda: pytest.fail("early ROI read/debounced IR")
    o._current_hazard_state_for_controller = lambda: pytest.fail("early ROI advanced hazard confirmation")
    assert validate(a) and publish(a)


def test_unevaluated_merged_danger_record_stays_on_normal_safety_path(early):
    a, o = early, early.o
    o._bunker_runtime = SimpleNamespace(config=SimpleNamespace(enabled=True, mode="merged", class_ids=(4,)))
    a.records.append(SimpleNamespace(class_id=4, reid_uid=0, time_since_update=0))
    assert validate(a)
    assert not publish(a)


def test_early_new_roi_does_not_invalidate_running_original_physical_sample(early, monkeypatch):
    a, o = early, early.o
    entered, resume = threading.Event(), threading.Event()
    old_context = o._longitudinal_context
    original = AstraDepthRuntime._select_multiregion_distance
    def scan(private, *args, **kwargs):
        entered.set()
        assert resume.wait(2.)
        return original(private, *args, **kwargs)
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    o._longitudinal_wake_event = SimpleNamespace(clear=lambda: None, set=lambda: None,
        wait=lambda duration: o._longitudinal_stop_event.set())
    thread, failures = start_loop(o)
    try:
        assert entered.wait(2.)
        assert validate(a) and publish(a)
        resume.set()
        thread.join(3.)
        assert len(o.calls) == 1
        assert o.calls[0]["evidence_capture_frame_id"] == old_context["capture_frame_id"]
        assert o.calls[0]["prepared_depth"].measurement.observation_sample_timestamp == NOW-.02
        assert o._depth_async_scheduler.publication_snapshot()[1].capture_frame_id == a.cap
    finally:
        resume.set()
        join_loop(o, thread, failures)


@pytest.mark.parametrize("change", ["stop", "uid", "revoked"])
def test_early_publication_does_not_bypass_depth_prepare_safety(early, change):
    a, o = early, early.o
    assert validate(a) and publish(a)
    if change == "stop": o._explicit_stop_requested = True
    elif change == "uid": o._follow_controller.active_target_id = 2
    else: o._depth_async_scheduler.revoke("identity_rejected")
    run_one_depth(a)
    assert not o.calls and a.camera._last_accepted_ts == 0


def test_real_process_callsite_publishes_after_identity_before_control_consumption():
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.process_external_frame)))
    calls = [(node.lineno, node.func.attr) for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    first = lambda name: min(line for line, method in calls if method == name)
    assert first("publication_snapshot") < first("_update_detector_identity_lease")
    assert first("_update_detector_identity_lease") < first("_publish_validated_depth_observation")
    assert first("_publish_validated_depth_observation") < first("_consume_track_records")
