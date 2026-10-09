"""Visible-target pixel work yields to safety/Depth30 without renewing replay."""
from dataclasses import replace
import threading

import pytest

import request_0513_modular as runtime
from car_control_modular.astra_depth import AstraDepthRuntime
from car_control_modular.control_types import HazardState
from test_depth_optimistic_transaction import scene
from test_search_observation_arbitration import owner as track_owner, _record
from test_turn_depth_scheduling import owner, attempt, NOW


@pytest.fixture
def visual(scene):
    obj, camera, distance, clock = scene
    ctx = obj._longitudinal_context
    targets = ctx["person_targets"]
    obj.frame_index = ctx["frame_index"]
    obj._active_capture_frame_id = ctx["capture_frame_id"]
    obj._active_capture_timestamp = ctx["capture_timestamp"]
    obj._vision_control_state = "target_visible_depth_valid"
    obj._reacquire_depth_pending = False
    obj._depth_longitudinal_authority_enabled = lambda: True
    obj._follow_controller.last_selected_target = targets[0]
    obj._persons_to_targets = lambda *a, **kw: list(targets)
    obj._longitudinal_wake_event = threading.Event()
    return scene


def vision_attempt(obj):
    ctx = obj._longitudinal_context
    return obj._queue_actions_for_persons(640, 480, ctx["persons"], control_source="vision")


def test_visual_pixels_and_diagnostics_run_without_control_or_live_sensor_lock(visual, monkeypatch):
    obj, camera, distance, clock = visual
    scans = []
    original = AstraDepthRuntime._select_multiregion_distance

    def scan(private, *args, **kwargs):
        assert private is not camera
        assert not obj._control_update_lock._is_owned()
        assert not camera._measurement_lock._is_owned()
        assert obj._longitudinal_wake_event.is_set()
        scans.append(private)
        return original(private, *args, **kwargs)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    assert vision_attempt(obj)
    assert len(scans) == len(obj.calls) == 1
    call = obj.calls[0]
    assert call["control_source"] == "vision"
    assert call["depth_use_latest"]
    assert call["prepared_depth"].measurement.sample_timestamp == NOW - .02
    assert call["depth_target_snapshot"][0] == obj._longitudinal_context["person_targets"][0]
    assert obj._depth30_linear_snapshot == ("forward", 32, 1, NOW - .15)


@pytest.mark.parametrize("change", ["context_replace", "context_clear", "geometry", "uid",
    "search", "controller_search", "stop", "brake", "shutdown", "running", "sample_expiry",
    "revision", "camera_close", "capture", "capture_time", "reacquire"])
def test_stale_visual_computation_cannot_mutate_history_decision_or_authority(visual, monkeypatch, change):
    obj, camera, distance, clock = visual
    original = AstraDepthRuntime._select_multiregion_distance

    def scan(private, *args, **kwargs):
        result = original(private, *args, **kwargs)
        if change == "context_replace": obj._longitudinal_context = dict(obj._longitudinal_context)
        elif change == "context_clear": obj._longitudinal_context = None
        elif change == "geometry":
            obj._longitudinal_context["person_targets"] = (replace(
                obj._longitudinal_context["person_targets"][0], bbox=(0., 0., 30., 30.)),)
        elif change == "uid": obj._follow_controller.active_target_id = 2
        elif change == "search": obj.search_state = "searching"
        elif change == "controller_search": obj._follow_controller.search_state = "searching"
        elif change == "stop": obj._explicit_stop_requested = True
        elif change == "brake": obj._brake_hold_active = True
        elif change == "shutdown": obj._runtime_shutdown_requested = True
        elif change == "running": obj.running = False
        elif change == "sample_expiry": clock[0] += .17
        elif change == "revision": camera._measurement_revision += 1
        elif change == "camera_close": camera._stop_event.set()
        elif change == "capture": obj._active_capture_frame_id += 1
        elif change == "capture_time": obj._active_capture_timestamp += .01
        elif change == "reacquire": obj._reacquire_depth_pending = True
        return result

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    assert vision_attempt(obj) is False
    assert not obj.calls
    assert camera._last_accepted_ts == 0
    assert not camera._distance_history and not camera._attempted_depth_samples
    assert distance._vision_depth_fusion._last_accepted_depth_sample_ts is None
    assert obj._depth30_linear_snapshot == ("forward", 32, 1, NOW - .15)


@pytest.mark.parametrize("lock_name", ["_measurement_lock", "_depth_lock"])
def test_visual_busy_sensor_does_not_fall_back_to_locked_scan(visual, monkeypatch, lock_name):
    obj, camera, distance, clock = visual
    monkeypatch.setattr(camera, "measure_target", lambda *a, **kw: pytest.fail("locked rescan"))
    lock = threading.Lock()
    setattr(camera, lock_name, lock)
    with lock:
        assert vision_attempt(obj) is False
    assert not obj.calls and camera._last_accepted_ts == 0


def test_visual_missing_prepare_fails_closed_without_locked_scan(visual, monkeypatch):
    obj, camera, distance, clock = visual
    monkeypatch.setattr(distance, "prepare_depth_measurement", None)
    monkeypatch.setattr(camera, "measure_target", lambda *a, **kw: pytest.fail("locked rescan"))
    assert vision_attempt(obj) is False
    assert not obj.calls and camera._last_accepted_ts == 0


@pytest.mark.parametrize("reason", ["stop", "identity_reset", "new_capture"])
def test_discarded_visual_tail_never_republishes_old_context(track_owner, reason):
    obj = track_owner
    obj.search_state = obj._follow_controller.search_state = "none"
    obj._vision_control_state = "target_visible_depth_valid"
    obj._assignments[3] = {"bbox_quality_ok": True}
    obj._longitudinal_context = None
    obj._longitudinal_context_lock = threading.Lock()

    def discard(*args, **kwargs):
        if reason == "stop": obj._explicit_stop_requested = True
        elif reason == "identity_reset": obj._follow_controller.active_target_id = None
        elif reason == "new_capture": obj._active_capture_frame_id += 1
        return False

    obj._queue_actions_for_persons = discard
    obj._consume_track_records([_record()], 640, 480, "test")
    assert "publish" not in obj._context_events
    assert obj._longitudinal_context is None


@pytest.mark.parametrize("reason", ["hazard", "identity_control_rejected"])
def test_current_hazard_and_identity_rejection_run_before_visual_roi_publication(track_owner, reason):
    obj = track_owner
    obj.search_state = obj._follow_controller.search_state = "none"
    obj._vision_control_state = "target_visible_depth_valid"
    obj._assignments[3] = {"bbox_quality_ok": True}
    if reason == "hazard":
        obj._handle_hazard_safety_state = lambda state: True
        obj._queue_actions_for_persons = lambda *a, **kw: pytest.fail("hazard reached ROI publication")
    else:
        obj._assignments[3]["identity_control_rejected"] = True
        def rejected(w, h, persons, **kw):
            assert not persons  # No target exists to enter optimistic ranging.
            return False
        obj._queue_actions_for_persons = rejected
    obj._consume_track_records([_record()], 640, 480, "test")
    assert "publish" not in obj._context_events


@pytest.mark.parametrize("reason", ["uid", "stop", "search", "shutdown", "expired"])
def test_visual_explicit_expected_target_rejection_reports_discard(visual, reason):
    obj, camera, distance, clock = visual
    ctx = obj._longitudinal_context
    if reason == "uid": obj._follow_controller.active_target_id = 2
    elif reason == "stop": obj._explicit_stop_requested = True
    elif reason == "search": obj.search_state = "searching"
    elif reason == "shutdown": obj._runtime_shutdown_requested = True
    elif reason == "expired": clock[0] += .3
    assert obj._queue_actions_for_persons(
        640, 480, ctx["persons"], expected_target_id=1,
        evidence_capture_timestamp=ctx["capture_timestamp"], control_source="vision",
    ) is False
    assert not obj.calls


@pytest.mark.parametrize("duplicate_after_winner", [False, True])
@pytest.mark.parametrize("newer_winner", [False, True])
def test_depth30_completes_while_visual_scan_paused_and_visual_only_replays_winner(
        visual, monkeypatch, duplicate_after_winner, newer_winner):
    obj, camera, distance, clock = visual
    entered, resume = threading.Event(), threading.Event()
    original = AstraDepthRuntime._select_multiregion_distance
    outcomes, errors, states = [], [], []

    def scan(private, *args, **kwargs):
        assert private is not camera
        assert not obj._control_update_lock._is_owned()
        if threading.current_thread().name == "test-visual-depth":
            entered.set()
            assert resume.wait(2)
        return original(private, *args, **kwargs)

    def consume(*args, **kw):
        assert obj._control_update_lock._is_owned()
        obj.calls.append(kw)
        if kw["control_source"] == "depth30":
            states.append(distance.get_frame_distance_state(
                640, kw["depth_target_snapshot"][0], frame_height=480,
                depth_use_latest=True, prepared_depth=kw["prepared_depth"]))

    def worker():
        try: outcomes.append(vision_attempt(obj))
        except BaseException as exc: errors.append(exc)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    obj._queue_actions_for_persons_locked = consume
    winner_stamp = NOW - (.01 if newer_winner else .02)
    thread = threading.Thread(target=worker, name="test-visual-depth")
    thread.start()
    try:
        assert entered.wait(1)
        assert obj._control_update_lock.acquire(timeout=.2)
        obj._control_update_lock.release()
        if newer_winner:
            camera._latest_depth = camera._latest_depth.copy()
            camera._latest_depth[:] = 1950
            camera._latest_depth_ts = winner_stamp
        assert attempt(obj)  # Real independent transaction + fusion commit.
        assert states[0].raw_distance_m == pytest.approx(1.95 if newer_winner else 1.8)
        if duplicate_after_winner:
            assert attempt(obj)
            assert states[-1].temporal_status == "duplicate"
        assert not resume.is_set()
        winning_state = distance.last_distance_state
    finally:
        resume.set()
        thread.join(2)
    assert not thread.is_alive() and not errors and outcomes == [True]
    replay = obj.calls[-1]["depth_state_replay"]
    assert replay.raw_distance_m is None and replay.sample_timestamp is None
    assert replay.temporal_status == "duplicate"
    assert replay.observation_timestamp == winner_stamp
    assert replay.is_replay_of(winner_stamp)
    assert replay.source_detail.endswith("_hold")
    assert distance.last_distance_state is winning_state
    assert camera._last_accepted_ts == winner_stamp
    assert len(camera._distance_history) == 1
    assert distance._vision_depth_fusion._last_accepted_depth_sample_ts == winner_stamp
    assert obj._depth30_linear_snapshot == ("forward", 32, 1, NOW - .15)


@pytest.mark.parametrize("use_replay", [False, True])
def test_real_visual_process_consumes_prepared_or_replay_without_republish_or_scan(visual, monkeypatch, use_replay):
    obj, camera, distance, clock = visual
    assert vision_attempt(obj)
    call = obj.calls[-1]
    target = call["depth_target_snapshot"][0]
    if use_replay:
        distance.get_frame_distance_state(
            640, target, frame_height=480, depth_use_latest=True, prepared_depth=call["prepared_depth"])
        call["depth_state_replay"] = obj._committed_visual_depth_replay(640, 480, target)
    obj._last_dispatched_action = runtime.ACTION_STOP
    obj._follow_controller.set_last_dispatched = lambda _: None
    obj._get_obstacle_status = lambda: {}
    obj._current_hazard_state_for_controller = lambda: HazardState()
    context = obj._longitudinal_context
    frames = []

    class DecisionReached(Exception): pass

    def decide(_, frame, **kwargs):
        frames.append(frame)
        raise DecisionReached()

    monkeypatch.setattr(camera, "measure_target", lambda *a, **kw: pytest.fail("locked rescan"))
    monkeypatch.setattr(obj, "_publish_longitudinal_context", lambda *a, **kw: pytest.fail("context replaced"))
    obj._follow_controller.decide = decide
    with pytest.raises(DecisionReached):
        obj._process_detections_modular(640, 480, context["persons"], **call)
    assert obj._longitudinal_context is context
    assert frames[0].capture_frame_id == context["capture_frame_id"]
    assert frames[0].capture_timestamp == context["capture_timestamp"]
    assert frames[0].distance_state.sample_timestamp == (None if use_replay else NOW - .02)


@pytest.mark.parametrize("mismatch", ["uid", "geometry", "size", "age", "sensor_ttl", "safety"])
def test_concurrent_replay_requires_same_geometry_age_and_preserves_safety(visual, mismatch):
    obj, camera, distance, clock = visual
    assert vision_attempt(obj)
    call = obj.calls[-1]
    target = call["depth_target_snapshot"][0]
    state = distance.get_frame_distance_state(
        640, target, frame_height=480, depth_use_latest=True, prepared_depth=call["prepared_depth"])
    max_age = .18
    if mismatch == "uid": target = replace(target, track_id=2)
    elif mismatch == "geometry": target = replace(target, bbox=(0., 0., 30., 30.))
    elif mismatch == "size": distance._last_vision_depth_frame_size = (320, 240)
    elif mismatch == "age": clock[0] += .17
    elif mismatch == "sensor_ttl": max_age = .01
    elif mismatch == "safety":
        distance.last_distance_state = replace(state, brake_latched=True, safety_distance_m=.5)
    result = obj._committed_visual_depth_replay(640, 480, target, sample_max_age_sec=max_age)
    if mismatch == "safety":
        assert result.brake_latched and result.safety_distance_m == .5
        assert not result.is_replay_of(NOW - .02)
    else:
        assert result is None
