"""Depth30 pixel work is off-lock; stale work never publishes live filters."""
import ast
from copy import deepcopy
from dataclasses import replace
import inspect
import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import request_0513_modular as runtime_module
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
from car_control_modular.depth_measurement_transaction import RANGING_STATE_FIELDS
from car_control_modular.sensor_modules import SensorRuntime
from car_control_modular.control_types import SteeringFeedback
from test_depth_raw_geometry_runtime import make_runtime
from test_turn_depth_scheduling import owner, context, attempt, NOW


@pytest.fixture
def scene(owner, monkeypatch):
    clock = [NOW]
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: clock[0])
    owner._control_update_lock = threading.RLock()
    owner._longitudinal_context = context(4910, NOW - .06)
    distance, _ = make_runtime(vision_depth_detector_bbox_max_age_sec=.25)
    distance.owner = owner
    owner._follow_controller.select_target_for_current_state = lambda targets: targets[0]
    camera = AstraDepthRuntime(AstraDepthConfig())
    camera._np = np
    camera._latest_depth = np.full((480, 640), 1800, dtype=np.uint16)
    camera._latest_depth_ts = NOW - .02
    sensors = object.__new__(SensorRuntime)
    sensors.config = SimpleNamespace(astra_depth_enable=True)
    sensors.astra_depth = camera
    distance.sensor_runtime = sensors
    owner._distance_runtime = distance
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    return owner, camera, distance, clock


def prepare(scene):
    owner, camera, distance, clock = scene
    target = owner._longitudinal_context["person_targets"][0]
    item = distance.prepare_depth_measurement(640, 480, target)
    assert item is not None
    return item, target


def test_real_prepare_compute_commit_fusion_without_live_scan(scene, monkeypatch):
    owner, camera, distance, clock = scene
    item, target = prepare(scene)
    original = deepcopy(camera._distance_history)
    result = item.transaction.run()
    assert result.raw_distance_m == pytest.approx(1.8)
    assert camera._last_accepted_ts == 0
    assert camera._distance_history == original
    assert distance._vision_depth_fusion._last_accepted_depth_sample_ts is None
    assert distance.commit_prepared_depth(item, target=target)
    assert camera._last_accepted_ts == NOW - .02
    monkeypatch.setattr(camera, "measure_target", lambda *a, **kw: pytest.fail("live scan under control lock"))
    state = distance.get_frame_distance_state(
        640, target, frame_height=480, depth_use_latest=True, prepared_depth=item)
    assert state.raw_distance_m == pytest.approx(1.8)
    assert state.sample_timestamp == NOW - .02
    assert distance._vision_depth_fusion._last_accepted_depth_sample_ts == NOW - .02


def test_heavy_work_and_diagnostic_flush_are_outside_control_and_live_sensor_locks(scene, monkeypatch):
    owner, camera, distance, clock = scene
    calls = []
    original = AstraDepthRuntime._select_multiregion_distance

    def scan(private, *args, **kwargs):
        assert private is not camera
        assert not owner._control_update_lock._is_owned()
        assert not camera._measurement_lock._is_owned()
        calls.append("scan")
        return original(private, *args, **kwargs)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    camera.diagnostics = SimpleNamespace(observe=lambda *a, **kw: calls.append(
        ("diagnostic", owner._control_update_lock._is_owned())))
    assert attempt(owner)
    assert calls == ["scan", ("diagnostic", False)]
    assert len(owner.calls) == 1
    assert owner.calls[0]["prepared_depth"].measurement.raw_distance_m == pytest.approx(1.8)


@pytest.mark.parametrize("change", ["replace_context", "revoke_context", "uid", "search",
    "controller_search", "stop", "brake", "shutdown", "running", "roi_expiry", "sample_expiry",
    "visual_measurement", "camera_close"])
def test_changes_during_pixel_work_reject_without_filter_fusion_or_controller_mutation(scene, monkeypatch, change):
    owner, camera, distance, clock = scene
    original = AstraDepthRuntime._select_multiregion_distance

    def scan(private, *args, **kwargs):
        result = original(private, *args, **kwargs)
        if change == "replace_context": owner._longitudinal_context = dict(owner._longitudinal_context)
        elif change == "revoke_context": owner._longitudinal_context = None
        elif change == "uid": owner._follow_controller.active_target_id = 2
        elif change == "search": owner.search_state = "searching"
        elif change == "controller_search": owner._follow_controller.search_state = "searching"
        elif change == "stop": owner._explicit_stop_requested = True
        elif change == "brake": owner._brake_hold_active = True
        elif change == "shutdown": owner._runtime_shutdown_requested = True
        elif change == "running": owner.running = False
        elif change == "roi_expiry": clock[0] += .24
        elif change == "sample_expiry": clock[0] += .17
        elif change == "visual_measurement": camera._measurement_revision += 1
        elif change == "camera_close": camera._stop_event.set()
        return result

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    assert attempt(owner)
    assert not owner.calls
    assert camera._last_accepted_ts == 0
    assert not camera._distance_history
    assert not camera._attempted_depth_samples
    assert distance._vision_depth_fusion._last_accepted_depth_sample_ts is None
    assert owner._depth30_linear_snapshot == ("forward", 32, 1, NOW - .15)


@pytest.mark.parametrize("lock_name", ["_measurement_lock", "_depth_lock"])
def test_prepare_never_waits_on_live_camera_locks(scene, lock_name):
    owner, camera, distance, clock = scene
    # Use a plain Lock so the same-thread fixture detects a blocking regression.
    camera_lock = threading.Lock()
    setattr(camera, lock_name, camera_lock)
    camera_lock.acquire()
    try:
        assert attempt(owner) is False
        assert not owner.calls
        assert not camera._attempted_depth_samples
    finally:
        camera_lock.release()


def test_busy_commit_control_lock_discards_after_bounded_wait(scene, monkeypatch):
    owner, camera, distance, clock = scene
    owner._control_update_lock = threading.Lock()
    original = AstraDepthRuntime._select_multiregion_distance
    def scan(private, *args, **kwargs):
        result = original(private, *args, **kwargs)
        owner._control_update_lock.acquire()
        return result
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    try:
        assert attempt(owner) is False
        assert not owner.calls
        assert camera._last_accepted_ts == 0
    finally:
        owner._control_update_lock.release()


@pytest.mark.parametrize("change", [None, "context", "uid", "search", "controller_search",
    "stop", "brake", "shutdown", "running", "sample_expiry", "revision", "camera_close"])
def test_commit_short_contention_reuses_scan_but_rechecks_every_proof(scene, monkeypatch, change):
    owner, camera, distance, clock = scene
    original = AstraDepthRuntime._select_multiregion_distance
    scans, waits = [], []

    class BriefContentionLock:
        """Deterministically finish another publisher during the bounded wait."""
        def __init__(self):
            self.locked = False
            self.busy = False

        def acquire(self, blocking=True, timeout=-1):
            if self.busy and not blocking:
                return False
            if self.busy:
                assert timeout == pytest.approx(.020)
                waits.append(timeout)
                clock[0] += .012
                self.busy = False
                if change == "context": owner._longitudinal_context = dict(owner._longitudinal_context)
                elif change == "uid": owner._follow_controller.active_target_id = 2
                elif change == "search": owner.search_state = "searching"
                elif change == "controller_search": owner._follow_controller.search_state = "searching"
                elif change == "stop": owner._explicit_stop_requested = True
                elif change == "brake": owner._brake_hold_active = True
                elif change == "shutdown": owner._runtime_shutdown_requested = True
                elif change == "running": owner.running = False
                elif change == "sample_expiry": clock[0] += .17
                elif change == "revision": camera._measurement_revision += 1
                elif change == "camera_close": camera._stop_event.set()
            assert not self.locked
            self.locked = True
            return True

        def release(self):
            assert self.locked
            self.locked = False

    lock = owner._control_update_lock = BriefContentionLock()

    def scan(private, *args, **kwargs):
        assert not lock.locked
        scans.append(private)
        result = original(private, *args, **kwargs)
        lock.busy = True
        return result

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    assert attempt(owner)
    assert len(scans) == 1 and scans[0] is not camera
    assert waits == [.020]
    assert not lock.locked
    # Neither a successful measurement commit nor a failed wait creates a
    # motor grant. The normal decision/admission path owns that next step.
    assert owner._depth30_linear_snapshot == ("forward", 32, 1, NOW - .15)
    if change is None:
        assert len(owner.calls) == 1
        prepared = owner.calls[0]["prepared_depth"]
        assert prepared.measurement.observation_sample_timestamp == NOW - .02
        assert camera._last_accepted_ts == NOW - .02
    else:
        assert not owner.calls
        assert camera._last_accepted_ts == 0
        assert distance._vision_depth_fusion._last_accepted_depth_sample_ts is None


def test_visual_measurement_can_complete_while_depth30_worker_is_paused(scene, monkeypatch):
    owner, camera, distance, clock = scene
    entered, resume = threading.Event(), threading.Event()
    original = AstraDepthRuntime._select_multiregion_distance
    def scan(private, *args, **kwargs):
        if private is not camera:
            entered.set()
            assert resume.wait(1)
        return original(private, *args, **kwargs)
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    errors = []
    def worker():
        try: attempt(owner)
        except Exception as exc: errors.append(exc)
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(1)
        assert owner._control_update_lock.acquire(timeout=.2)
        try:
            target = owner._longitudinal_context["person_targets"][0]
            state = distance.get_frame_distance_state(640, target, frame_height=480, depth_use_latest=True)
            assert state.raw_distance_m == pytest.approx(1.8)
        finally:
            owner._control_update_lock.release()
    finally:
        resume.set()
        thread.join(1)
    assert not thread.is_alive() and not errors and not owner.calls
    assert camera._last_accepted_ts == NOW - .02
    assert len(camera._distance_history) == 1


def test_duplicate_physical_sample_never_renews_anchor_or_fusion_timestamp(scene):
    owner, camera, distance, clock = scene
    item, target = prepare(scene)
    item.transaction.run()
    assert distance.commit_prepared_depth(item, target=target)
    assert distance.get_frame_distance_state(
        640, target, frame_height=480, depth_use_latest=True, prepared_depth=item).raw_distance_m == 1.8
    clock[0] += .03
    second, target = prepare(scene)
    second.transaction.run()
    assert distance.commit_prepared_depth(second, target=target)
    state = distance.get_frame_distance_state(
        640, target, frame_height=480, depth_use_latest=True, prepared_depth=second)
    assert state.raw_distance_m is None
    assert camera._last_accepted_ts == NOW - .02
    assert distance._vision_depth_fusion._last_accepted_depth_sample_ts == NOW - .02
    assert len(camera._distance_history) == 1


@pytest.mark.parametrize("phase", ["before_run", "after_run"])
def test_uncommitted_diagnostics_never_reach_live_writer(scene, phase):
    owner, camera, distance, clock = scene
    seen = []
    camera.diagnostics = SimpleNamespace(observe=lambda *a, **kw: seen.append(kw))
    item, target = prepare(scene)
    if phase == "after_run": item.transaction.run()
    item.transaction.flush()
    assert seen == []


def test_sensor_revision_prevents_same_snapshot_double_commit(scene):
    first, target = prepare(scene)
    second, _ = prepare(scene)
    first.transaction.run()
    second.transaction.run()
    assert scene[2].commit_prepared_depth(first, target=target)
    assert not scene[2].commit_prepared_depth(second, target=target)
    assert not scene[2].commit_prepared_depth(first, target=target)
    with pytest.raises(RuntimeError): first.transaction.run()


def test_replaced_camera_frames_do_not_mutate_private_frame_snapshot(scene):
    owner, camera, distance, clock = scene
    item, target = prepare(scene)
    camera._latest_depth = np.full((480, 640), 3000, dtype=np.uint16)
    camera._latest_depth_ts += .01
    assert item.transaction.run().raw_distance_m == 1.8


def test_all_mutable_ranging_assignments_are_transactional():
    # Future filter/diagnostic state additions must be copied, not silently
    # shared through the shallow hardware/runtime shell.
    tree = ast.parse(inspect.getsource(AstraDepthRuntime))
    exempt = {"__init__", "start", "_depth_loop", "release", "close"}
    fields = set()
    for method in tree.body[0].body:
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name not in exempt:
            fields.update(node.attr for node in ast.walk(method)
                          if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                          and isinstance(node.value, ast.Name) and node.value.id == "self")
    assert fields - set(RANGING_STATE_FIELDS) - {"_measurement_revision"} == set()


def test_deferred_wait_for_old_cap_not_reported_as_new_caps_latency(owner, caplog):
    owner._depth30_deferred_schedule = (100, NOW - 10, 20)
    with caplog.at_level(logging.INFO): assert attempt(owner)
    line = next(record.message for record in caplog.records if record.message.startswith("depth30_schedule "))
    assert "deferred_count=0 deferred_elapsed_ms=0.0" in line
    assert "superseded_capture_frame_id=100 superseded_count=20" in line


@pytest.mark.parametrize("roi_age,skew,delay", [
    (.1891, .1595, .1008), (.2249, .1775, .0525), (.2379, .172, .0535),
    (.2166, .1642, .0721), (.2, .175, .118),
])
def test_real_main_prepared_path_preserves_bounded_selection_completion(scene, monkeypatch, roi_age, skew, delay):
    owner, camera, distance, clock = scene
    capture = NOW - roi_age
    stamp = capture + skew
    owner._longitudinal_context = context(4910, capture)
    camera._depth_history.append((stamp, camera._latest_depth))
    camera._latest_depth_ts = NOW
    camera._latest_depth = np.full((480, 640), 4500, dtype=np.uint16)
    original = AstraDepthRuntime._select_multiregion_distance
    def delayed(private, *args, **kwargs):
        result = original(private, *args, **kwargs)
        clock[0] = NOW + delay
        return result
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", delayed)
    states = []
    def consume(*args, **kwargs):
        states.append(distance.get_frame_distance_state(
            640, owner._longitudinal_context["person_targets"][0], frame_height=480,
            depth_use_latest=True, prepared_depth=kwargs["prepared_depth"]))
    owner._queue_actions_for_persons_locked = consume
    assert attempt(owner)
    assert clock[0] - capture > .25
    assert clock[0] - stamp < .18
    assert len(states) == 1
    assert states[0].raw_distance_m == pytest.approx(1.8)
    assert states[0].sample_timestamp == pytest.approx(stamp)
    assert camera._last_accepted_ts == pytest.approx(stamp)
    assert camera._latest_depth_ts == NOW  # Hardware/read source never restored from clone.
    assert attempt(owner)  # The expired ROI cannot start another measurement.
    assert len(states) == 1


@pytest.mark.parametrize("delay", [.04, .10])
def test_real_main_normal_latest_keeps_already_selected_sample_when_roi_crosses_age_gate(scene, monkeypatch, delay):
    owner, camera, distance, clock = scene
    owner._longitudinal_context = context(4910, NOW - .17)
    original = AstraDepthRuntime._select_multiregion_distance
    def delayed(private, *args, **kwargs):
        result = original(private, *args, **kwargs)
        clock[0] += delay
        # New yaw would disallow starting another latest-frame ROI after180ms;
        # it does not invalidate the actual previously selected physical frame.
        owner._action_runtime.get_steering_feedback = lambda: SteeringFeedback(
            timestamp=clock[0], trustworthy=True, yaw_rate_right_dps=20.)
        return result
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", delayed)
    assert attempt(owner)
    assert len(owner.calls) == 1
    prepared = owner.calls[0]["prepared_depth"]
    assert prepared.transaction.selected_at == NOW
    state = distance.get_frame_distance_state(
        640, owner._longitudinal_context["person_targets"][0], frame_height=480,
        depth_use_latest=True, prepared_depth=prepared)
    assert state.raw_distance_m == pytest.approx(1.8)
    assert state.sample_timestamp == NOW - .02


def test_natural_old_lease_expiry_does_not_cancel_independent_new_measurement(scene, monkeypatch):
    owner, camera, distance, clock = scene
    original = AstraDepthRuntime._select_multiregion_distance
    def expire_lease(private, *args, **kwargs):
        result = original(private, *args, **kwargs)
        owner._depth30_linear_snapshot = None
        owner._depth30_linear_timing = None
        return result
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", expire_lease)
    assert attempt(owner)
    assert len(owner.calls) == 1
    assert camera._last_accepted_ts == NOW - .02
    # Measurement commit itself never restores a motor grant.
    assert owner._depth30_linear_snapshot is None


def test_short_sensor_ttl_is_rechecked_at_actual_commit(scene):
    owner, camera, distance, clock = scene
    camera.config = replace(camera.config, max_frame_age_sec=.10)
    item, target = prepare(scene)
    item.transaction.run()
    assert item.transaction.result.raw_distance_m == 1.8
    clock[0] += .09  # 110ms physical age: under180 but beyond configured100.
    assert not distance.commit_prepared_depth(item, target=target)
    assert camera._last_accepted_ts == 0


def test_commit_uses_post_validation_clock_not_old_callers_now(scene, monkeypatch):
    owner, camera, distance, clock = scene
    item, target = prepare(scene)
    item.transaction.run()
    geometry = distance._prepared_depth_geometry
    def delayed(*args, **kwargs):
        result = geometry(*args, **kwargs)
        clock[0] += .17
        return result
    monkeypatch.setattr(distance, "_prepared_depth_geometry", delayed)
    assert not distance.commit_prepared_depth(item, target=target)
    assert camera._last_accepted_ts == 0


def test_real_depth_runtime_missing_prepare_does_not_fall_back_to_locked_scan(scene, monkeypatch):
    owner, camera, distance, clock = scene
    monkeypatch.setattr(distance, "prepare_depth_measurement", None)
    monkeypatch.setattr(camera, "measure_target", lambda *a, **kw: pytest.fail("locked fallback"))
    assert attempt(owner)
    assert owner.calls == []
