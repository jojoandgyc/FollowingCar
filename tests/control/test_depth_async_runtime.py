"""Real private Depth scans plus concurrent vision, without hardware."""
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.astra_depth import AstraDepthRuntime
from car_control_modular.control_types import HazardState
from car_control_modular.depth_async_scheduler import DepthAsyncScheduler
from test_depth_optimistic_transaction import scene
from test_turn_depth_scheduling import owner, NOW
from test_visual_depth_optimistic_transaction import visual


@pytest.fixture
def async_scene(visual, monkeypatch):
    obj, camera, distance, clock = visual
    monkeypatch.setattr(runtime, "LATERAL_INTENT_CONTROL_ENABLE", True)
    obj._depth_async_scheduler = DepthAsyncScheduler()
    obj._longitudinal_context = obj._depth_async_scheduler.submit(
        obj._longitudinal_context, now=clock[0]).as_dict()
    obj._depth_async_scheduler.worker_tick(now=clock[0])
    obj._longitudinal_thread = SimpleNamespace(is_alive=lambda: True)
    obj._follow_controller.can_decide_lateral_only = lambda *a, **kw: True
    obj._get_obstacle_status = lambda: {}
    obj._current_hazard_state_for_controller = lambda: HazardState()
    obj._longitudinal_wake_event = SimpleNamespace(
        clear=lambda: None, set=lambda: None,
        wait=lambda duration: obj._longitudinal_stop_event.set())
    return visual


def vision(obj, ctx=None):
    ctx = obj._longitudinal_context if ctx is None else ctx
    return obj._queue_actions_for_persons(640, 480, list(ctx["persons"]),
                                         control_source="vision")


def new_visual(obj, clock, offset=1, advance=.01):
    """Make a genuinely new detector proof, not a relabelled old distance."""
    old = obj._longitudinal_context
    stamp = old["capture_timestamp"] + advance
    target = old["person_targets"][0]
    observation = replace(target.depth_observation,
                          capture_frame_id=old["capture_frame_id"] + offset,
                          capture_timestamp=stamp)
    target = replace(target, depth_observation=observation)
    obj.frame_index += 1
    obj._active_capture_frame_id = observation.capture_frame_id
    obj._active_capture_timestamp = stamp
    obj._persons_to_targets = lambda *a, **kw: [target]
    clock[0] += advance
    return obj._queue_actions_for_persons(640, 480,
        [(target.bbox, target.track_id, target.confidence, target.area)],
        control_source="vision")


def start_loop(obj):
    failures = []
    obj._longitudinal_stop_event.clear()
    def run():
        try:
            obj._longitudinal_control_loop()
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=run, name="test-exclusive-depth30")
    obj._longitudinal_thread = thread
    thread.start()
    return thread, failures


def join_loop(obj, thread, failures):
    obj._longitudinal_stop_event.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert not failures


def test_stable_vision_publishes_roi_and_lateral_without_a_depth_scan(async_scene, monkeypatch):
    obj, camera, distance, clock = async_scene
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance",
                        lambda *a, **kw: pytest.fail("stable vision scanned pixels"))
    old_grant = obj._depth30_linear_snapshot
    assert new_visual(obj, clock)
    assert len(obj.calls) == 1
    assert obj.calls[0]["visual_lateral_only"]
    assert obj.calls[0]["range_deferred"]
    assert obj.calls[0]["control_source"] == "vision"
    assert obj._longitudinal_context["capture_frame_id"] == 4911
    assert obj._depth30_linear_snapshot is old_grant
    assert camera._last_accepted_ts == 0
    assert distance._vision_depth_fusion._last_accepted_depth_sample_ts is None


def test_clear_then_same_uid_republish_cannot_launder_old_scan_epoch(async_scene, monkeypatch):
    obj, camera, distance, clock = async_scene
    original = AstraDepthRuntime._select_multiregion_distance

    def scan(private, *a, **kw):
        old = obj._longitudinal_context
        obj._clear_longitudinal_context(revoke_translation=False, reason="identity_rejected")
        obj._publish_longitudinal_context(640, 480, list(old["persons"]),
                                          person_targets=old["person_targets"])
        assert obj._longitudinal_context["target_id"] == old["target_id"]
        return original(private, *a, **kw)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    thread, failures = start_loop(obj)
    thread.join(timeout=3)
    join_loop(obj, thread, failures)
    assert not obj.calls
    assert camera._last_accepted_ts == 0
    assert not obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).busy
    # Reacquired fresh work is independent and can commit in a later iteration.
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", original)
    thread, failures = start_loop(obj)
    thread.join(timeout=3)
    join_loop(obj, thread, failures)
    assert len(obj.calls) == 1
    assert camera._last_accepted_ts == NOW - .02


@pytest.mark.parametrize("startup", ["no_uid", "no_previous_target", "reacquire"])
def test_bootstrap_or_reacquisition_keeps_sync_path_with_exclusive_scan_slot(async_scene, monkeypatch, startup):
    obj, camera, distance, clock = async_scene
    target = obj._longitudinal_context["person_targets"][0]
    if startup == "no_uid":
        obj._follow_controller.active_target_id = None
    elif startup == "no_previous_target":
        obj._follow_controller.last_selected_target = None
    else:
        obj._reacquire_depth_pending = True
    obj._depth30_linear_snapshot = None
    scans, states = [], []
    original = AstraDepthRuntime._select_multiregion_distance

    def scan(live, *a, **kw):
        # This is the legacy synchronous ranging path, not private speculation.
        assert live is camera
        assert obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).busy
        assert obj._depth_async_scheduler.begin(now=clock[0], max_capture_age_sec=.25) is None
        scans.append(live)
        return original(live, *a, **kw)

    def consume(*a, **kw):
        obj.calls.append(kw)
        assert not kw.get("range_deferred", False)
        assert "prepared_depth" not in kw
        states.append(distance.get_frame_distance_state(640, target, frame_height=480,
                                                         depth_use_latest=True))

    obj._queue_actions_for_persons_locked = consume
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    assert vision(obj)
    assert len(scans) == len(states) == 1
    assert states[0].raw_distance_m == pytest.approx(1.8)
    assert obj._depth30_linear_snapshot is None
    assert not obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).busy


def test_sync_fallback_early_uid_validation_return_releases_scan_slot(async_scene):
    obj, camera, distance, clock = async_scene
    obj._longitudinal_thread = SimpleNamespace(is_alive=lambda: False)
    assert obj._queue_actions_for_persons(640, 480, obj._longitudinal_context["persons"],
                                         control_source="vision", expected_target_id=2) is False
    assert not obj.calls
    assert camera._last_accepted_ts == 0
    assert not obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).busy
    assert distance._vision_depth_fusion._last_accepted_depth_sample_ts is None


def test_blocked_real_scan_survives_new_frames_and_commits_original_physical_sample(async_scene, monkeypatch):
    obj, camera, distance, clock = async_scene
    entered, resume = threading.Event(), threading.Event()
    scans = []
    original = AstraDepthRuntime._select_multiregion_distance

    def scan(private, *a, **kw):
        assert private is not camera
        assert not obj._control_update_lock._is_owned()
        scans.append(private)
        entered.set()
        assert resume.wait(timeout=3)
        return original(private, *a, **kw)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    original_context = obj._longitudinal_context
    old_grant = obj._depth30_linear_snapshot
    thread, failures = start_loop(obj)
    try:
        assert entered.wait(timeout=2)
        for _ in range(5):
            assert new_visual(obj, clock)
        assert len(scans) == 1
        assert len(obj.calls) == 5
        assert all(call["visual_lateral_only"] for call in obj.calls)
        assert obj._longitudinal_context["capture_frame_id"] == 4915
        assert camera._last_accepted_ts == 0
        resume.set()
        thread.join(timeout=3)
        depth_calls = [call for call in obj.calls if call["control_source"] == "depth30"]
        assert len(depth_calls) == 1
        call = depth_calls[0]
        assert call["evidence_capture_frame_id"] == original_context["capture_frame_id"]
        assert call["evidence_capture_timestamp"] == original_context["capture_timestamp"]
        assert call["prepared_depth"].measurement.observation_sample_timestamp == NOW - .02
        assert camera._last_accepted_ts == NOW - .02
        assert len(camera._distance_history) == 1
        # Queue spy has no authority-writing side effects.
        assert obj._depth30_linear_snapshot is old_grant
        assert not obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).busy
    finally:
        resume.set()
        join_loop(obj, thread, failures)


def test_visual_frame_begin_preserves_active_scan_without_renewing_clocks(async_scene, monkeypatch):
    obj, camera, distance, clock = async_scene
    old_context = obj._longitudinal_context
    old_grant = obj._depth30_linear_snapshot
    original = AstraDepthRuntime._select_multiregion_distance

    def scan(private, *a, **kw):
        obj._clear_longitudinal_context(revoke_translation=False, reason="visual_frame_begin")
        assert obj._longitudinal_context is old_context
        assert obj._depth30_linear_snapshot is old_grant
        return original(private, *a, **kw)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    thread, failures = start_loop(obj)
    thread.join(timeout=3)
    join_loop(obj, thread, failures)
    assert len(obj.calls) == 1
    assert camera._last_accepted_ts == NOW - .02
    assert obj._longitudinal_context["capture_timestamp"] == NOW - .06


@pytest.mark.parametrize("change", [
    "uid", "context_clear", "stop", "search", "controller_search",
    "shutdown", "running", "sample_expiry", "revision", "camera_close",
    "scheduler_revoke",
])
def test_actual_scan_rechecks_revocation_and_safety_before_commit(async_scene, monkeypatch, change):
    obj, camera, distance, clock = async_scene
    entered, resume = threading.Event(), threading.Event()
    original = AstraDepthRuntime._select_multiregion_distance
    old_grant = obj._depth30_linear_snapshot

    def scan(private, *a, **kw):
        entered.set()
        assert resume.wait(timeout=3)
        return original(private, *a, **kw)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    thread, failures = start_loop(obj)
    try:
        assert entered.wait(timeout=2)
        if change == "uid":
            obj._follow_controller.active_target_id = 2
        elif change == "context_clear":
            obj._clear_longitudinal_context(revoke_translation=False, reason="identity_rejected")
        elif change == "stop":
            obj._explicit_stop_requested = True
        elif change == "search":
            obj.search_state = "searching"
        elif change == "controller_search":
            obj._follow_controller.search_state = "searching"
        elif change == "shutdown":
            obj._runtime_shutdown_requested = True
        elif change == "running":
            obj.running = False
        elif change == "sample_expiry":
            clock[0] += .3
        elif change == "revision":
            camera._measurement_revision += 1
        elif change == "camera_close":
            camera._stop_event.set()
        else:
            obj._depth_async_scheduler.revoke("hazard")
        resume.set()
        thread.join(timeout=3)
        assert not obj.calls
        assert camera._last_accepted_ts == 0
        assert not camera._distance_history
        assert distance._vision_depth_fusion._last_accepted_depth_sample_ts is None
        assert obj._depth30_linear_snapshot is old_grant
        assert not obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).busy
    finally:
        resume.set()
        join_loop(obj, thread, failures)


def test_worker_timeout_does_not_allow_parallel_fallback_or_revoked_commit(async_scene, monkeypatch):
    obj, camera, distance, clock = async_scene
    entered, resume = threading.Event(), threading.Event()
    scans = []
    original = AstraDepthRuntime._select_multiregion_distance

    def scan(private, *a, **kw):
        scans.append(private)
        entered.set()
        assert resume.wait(timeout=3)
        return original(private, *a, **kw)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    thread, failures = start_loop(obj)
    try:
        assert entered.wait(timeout=2)
        clock[0] += .251
        # Supply a fresh visual proof while the original scan is still stuck.
        old = obj._longitudinal_context
        target = old["person_targets"][0]
        target = replace(target, depth_observation=replace(target.depth_observation,
            capture_frame_id=4911, capture_timestamp=clock[0] - .03))
        obj._persons_to_targets = lambda *a, **kw: [target]
        obj._active_capture_frame_id = 4911
        obj._active_capture_timestamp = target.depth_observation.capture_timestamp
        assert vision(obj)
        assert len(scans) == 1
        assert obj.calls[-1]["range_deferred"]
        assert obj.calls[-1]["visual_lateral_only"]
        assert obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).reason == "revoked_busy"
        resume.set()
        thread.join(timeout=3)
        assert not any(c["control_source"] == "depth30" for c in obj.calls)
        assert camera._last_accepted_ts == 0
    finally:
        resume.set()
        join_loop(obj, thread, failures)
    # A later independent fallback can now measure, not while old work runs.
    obj._longitudinal_thread = SimpleNamespace(is_alive=lambda: False)
    camera._latest_depth_ts = clock[0] - .02
    assert vision(obj)
    assert len(scans) == 2
    assert camera._last_accepted_ts == clock[0] - .02


@pytest.mark.parametrize("unhealthy", ["not_started", "dead", "errors"])
def test_unhealthy_worker_runs_exclusive_sync_fallback_and_releases_slot(async_scene, monkeypatch, unhealthy):
    obj, camera, distance, clock = async_scene
    scheduler = obj._depth_async_scheduler
    if unhealthy == "not_started":
        scheduler = obj._depth_async_scheduler = DepthAsyncScheduler()
    elif unhealthy == "dead":
        obj._longitudinal_thread = SimpleNamespace(is_alive=lambda: False)
    else:
        for _ in range(2):
            ticket = scheduler.begin(now=clock[0], max_capture_age_sec=.25)
            scheduler.finish(ticket, now=clock[0], error=True)
    scans = []
    original = AstraDepthRuntime._select_multiregion_distance

    def scan(private, *a, **kw):
        scans.append(private)
        assert scheduler.health(now=clock[0], worker_alive=True).busy
        assert scheduler.begin(now=clock[0], max_capture_age_sec=.25) is None
        return original(private, *a, **kw)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    assert vision(obj)
    assert len(scans) == 1
    assert len(obj.calls) == 1
    assert "visual_lateral_only" not in obj.calls[0]
    assert obj.calls[0]["prepared_depth"].measurement.raw_distance_m == pytest.approx(1.8)
    assert not scheduler.health(now=clock[0], worker_alive=True).busy
    assert camera._last_accepted_ts == NOW - .02


def test_real_sync_fallback_blocks_depth_loop_from_starting_duplicate_scan(async_scene, monkeypatch):
    obj, camera, distance, clock = async_scene
    obj._longitudinal_thread = SimpleNamespace(is_alive=lambda: False)
    entered, resume = threading.Event(), threading.Event()
    original = AstraDepthRuntime._select_multiregion_distance
    scans, errors, results = [], [], []

    def scan(private, *a, **kw):
        scans.append(threading.current_thread().name)
        entered.set()
        assert resume.wait(timeout=3)
        return original(private, *a, **kw)

    def fallback():
        try:
            results.append(vision(obj))
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    vision_thread = threading.Thread(target=fallback, name="test-exclusive-vision")
    vision_thread.start()
    thread = None
    failures = []
    try:
        assert entered.wait(timeout=2)
        thread, failures = start_loop(obj)
        assert obj._depth_async_scheduler.begin(now=clock[0], max_capture_age_sec=.25) is None
        assert scans == ["test-exclusive-vision"]
        obj._longitudinal_stop_event.set()
        thread.join(timeout=3)
        resume.set()
        vision_thread.join(timeout=3)
        assert results == [True] and not errors
        assert scans == ["test-exclusive-vision"]
        assert camera._last_accepted_ts == NOW - .02
        assert not obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).busy
    finally:
        resume.set()
        obj._longitudinal_stop_event.set()
        vision_thread.join(timeout=3)
        if thread is not None:
            join_loop(obj, thread, failures)
    assert not vision_thread.is_alive()


def test_exception_in_real_worker_releases_slot_records_error_and_allows_fallback(async_scene, monkeypatch):
    obj, camera, distance, clock = async_scene
    original = AstraDepthRuntime._select_multiregion_distance

    def explode(private, *a, **kw):
        raise RuntimeError("injected scan failure")

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", explode)
    thread, failures = start_loop(obj)
    thread.join(timeout=3)
    join_loop(obj, thread, failures)
    health = obj._depth_async_scheduler.health(now=clock[0], worker_alive=True)
    assert health.consecutive_errors == 1 and not health.busy
    assert not obj.calls and camera._last_accepted_ts == 0
    obj._longitudinal_thread = SimpleNamespace(is_alive=lambda: False)
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", original)
    assert vision(obj)
    assert camera._last_accepted_ts == NOW - .02
    assert len(obj.calls) == 1


@pytest.mark.parametrize("failure", ["scan", "prepare"])
def test_sync_fallback_exception_or_early_return_never_leaks_scan_slot(async_scene, monkeypatch, failure):
    obj, camera, distance, clock = async_scene
    obj._longitudinal_thread = SimpleNamespace(is_alive=lambda: False)
    if failure == "scan":
        def explode(*a, **kw):
            raise RuntimeError("injected fallback failure")
        monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", explode)
        with pytest.raises(RuntimeError, match="injected fallback"):
            vision(obj)
    else:
        monkeypatch.setattr(distance, "prepare_depth_measurement", lambda *a, **kw: None)
        assert vision(obj) is False
    assert not obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).busy
    assert not obj.calls
    assert camera._last_accepted_ts == 0
