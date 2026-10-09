"""No-sample preflight is a completed tick, not 5 ms lock contention."""
import threading
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.astra_depth import AstraDepthRuntime
from car_control_modular.depth_async_scheduler import DepthAsyncScheduler
from test_depth_optimistic_transaction import scene
from test_history_depth_compute_budget import history
from test_turn_depth_scheduling import owner, context, attempt, NOW
from test_visual_depth_optimistic_transaction import visual, vision_attempt


def enable_scheduler(scene, monkeypatch):
    obj, camera, distance, clock = scene
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", .5)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_CONTROL_HZ", 30.)
    obj._depth_async_scheduler = DepthAsyncScheduler()
    obj._longitudinal_context = obj._depth_async_scheduler.submit(
        obj._longitudinal_context, now=clock[0]).as_dict()
    return obj._depth_async_scheduler


def unmeasurable_history(scene):
    # Both known frames are too far after the old ROI's capture. The ROI's
    # 500 ms lookup window cannot manufacture the 180 ms association proof.
    history(scene, roi_age=.350, sample_age=.010)


@pytest.mark.parametrize("with_scheduler", [False, True])
def test_unchanged_empty_history_uses_normal_period_without_pixel_work(
        scene, monkeypatch, caplog, with_scheduler):
    obj, camera, distance, clock = scene
    unmeasurable_history(scene)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", .5)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_CONTROL_HZ", 30.)
    scheduler = enable_scheduler(scene, monkeypatch) if with_scheduler else None
    camera._pending_jump_count = 2
    camera._pending_jump_distance_m = 2.6
    camera._pending_jump_timestamp = NOW - .2
    authority, timing = obj._depth30_linear_snapshot, obj._depth30_linear_timing
    state = distance.last_distance_state
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance",
                        lambda *a, **kw: pytest.fail("empty history scanned pixels"))
    waits = []

    def wait(duration):
        waits.append(duration)
        clock[0] += duration
        if len(waits) == 3:
            obj._longitudinal_stop_event.set()

    obj._longitudinal_wake_event = SimpleNamespace(clear=lambda: None, wait=wait)
    with caplog.at_level("INFO"):
        obj._longitudinal_control_loop()
    assert waits == pytest.approx([1 / 30.] * 3)
    assert caplog.text.count("depth30_prepared_skip") == 3
    assert caplog.text.count("reason=history_no_eligible_sample") == 3
    assert obj.calls == []
    assert obj._depth30_linear_snapshot is authority
    assert obj._depth30_linear_timing is timing
    assert distance.last_distance_state is state
    assert camera._measurement_revision == 0
    assert camera._last_accepted_ts == 0
    assert (camera._pending_jump_count, camera._pending_jump_distance_m,
            camera._pending_jump_timestamp) == (2, 2.6, NOW - .2)
    if scheduler is not None:
        health = scheduler.health(now=clock[0], worker_alive=True)
        assert health.healthy and not health.busy and health.consecutive_errors == 0
        assert scheduler.publication_snapshot()[0] == 0


def test_sample_expiring_during_preflight_is_completed_not_immediate_retry(
        scene, monkeypatch, caplog):
    obj, camera, distance, clock = scene
    history(scene, roi_age=.310, sample_age=.140)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", .5)
    original = AstraDepthRuntime._aligned_depth_locked

    def selection_then_delay(private, *args, **kwargs):
        selected = original(private, *args, **kwargs)
        clock[0] += .050
        return selected

    monkeypatch.setattr(AstraDepthRuntime, "_aligned_depth_locked", selection_then_delay)
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance",
                        lambda *a, **kw: pytest.fail("expired preflight scanned pixels"))
    authority = obj._depth30_linear_snapshot
    with caplog.at_level("INFO"):
        assert attempt(obj) is True
    assert "reason=history_sample_expired" in caplog.text
    assert not obj.calls and obj._depth30_linear_snapshot is authority


def test_new_roi_wakes_waiter_then_new_depth_on_same_roi_is_not_suppressed(
        scene, monkeypatch):
    obj, camera, distance, clock = scene
    unmeasurable_history(scene)
    scheduler = enable_scheduler(scene, monkeypatch)
    waits, scans = [], []
    wake = threading.Event()
    original = AstraDepthRuntime._select_multiregion_distance

    def scan(private, *args, **kwargs):
        scans.append(clock[0])
        return original(private, *args, **kwargs)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)

    def wait(duration):
        waits.append(duration)
        if len(waits) == 1:
            assert obj.calls == [] and scans == []
            assert not scheduler.health(now=clock[0], worker_alive=True).busy
            # An actual mailbox publication can interrupt the nominal 30 Hz
            # wait; no fixed sleep or per-CAP no-data blacklist was introduced.
            clock[0] += .001
            camera._latest_depth_ts = clock[0] - .01
            next_context = context(4911, clock[0] - .04)
            obj._longitudinal_context = scheduler.submit(
                next_context, now=clock[0]).as_dict()
            wake.set()
            assert wake.wait(timeout=0)
        elif len(waits) == 2:
            assert len(obj.calls) == len(scans) == 1
            clock[0] += duration
            # One detector ROI can carry several real physical Depth frames.
            camera._latest_depth_ts = clock[0] - .005
        else:
            obj._longitudinal_stop_event.set()

    obj._longitudinal_wake_event = SimpleNamespace(clear=wake.clear, wait=wait)
    obj._longitudinal_control_loop()
    assert waits == pytest.approx([1 / 30.] * 3)
    assert scans[0] == pytest.approx(NOW + .001)
    assert len(scans) == len(obj.calls) == 2
    assert [call["evidence_capture_frame_id"] for call in obj.calls] == [4911, 4911]
    assert camera._last_accepted_ts == pytest.approx(clock[0] - .005)
    assert camera._measurement_revision == 2


@pytest.mark.parametrize("lock_name", ["control", "measurement", "depth"])
def test_actual_lock_contention_keeps_bounded_fast_retry(scene, monkeypatch, lock_name):
    obj, camera, distance, clock = scene
    scheduler = enable_scheduler(scene, monkeypatch)
    lock = threading.Lock()
    if lock_name == "control":
        obj._control_update_lock = lock
    else:
        setattr(camera, "_measurement_lock" if lock_name == "measurement" else "_depth_lock", lock)
    lock.acquire()
    waits = []

    def wait(duration):
        waits.append(duration)
        clock[0] += duration
        if len(waits) == 1:
            assert not obj.calls
            assert not scheduler.health(now=clock[0], worker_alive=True).busy
            lock.release()
        else:
            obj._longitudinal_stop_event.set()

    obj._longitudinal_wake_event = SimpleNamespace(clear=lambda: None, wait=wait)
    try:
        obj._longitudinal_control_loop()
    finally:
        if lock.locked():
            lock.release()
    assert waits == pytest.approx([.005, 1 / 30.])
    assert len(obj.calls) == 1
    assert camera._last_accepted_ts == pytest.approx(NOW - .02)


def test_visual_no_data_keeps_discard_contract_and_releases_fallback_slot(
        visual, monkeypatch):
    obj, camera, distance, clock = visual
    unmeasurable_history(visual)
    ctx = obj._longitudinal_context
    obj._active_capture_frame_id = ctx["capture_frame_id"]
    obj._active_capture_timestamp = ctx["capture_timestamp"]
    obj._persons_to_targets = lambda *a, **kw: list(ctx["person_targets"])
    obj._follow_controller.last_selected_target = ctx["person_targets"][0]
    # Force synchronous ownership (startup/worker unavailable), not a change
    # to detector/identity eligibility or the physical sample proof.
    enable_scheduler(visual, monkeypatch)
    obj._async_visual_lateral_eligible = lambda *a, **kw: False
    obj._visible_latest_depth_eligible = lambda *a, **kw: True
    authority = obj._depth30_linear_snapshot
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance",
                        lambda *a, **kw: pytest.fail("fallback empty history scanned pixels"))
    assert vision_attempt(obj) is False
    assert not obj.calls and obj._depth30_linear_snapshot is authority
    assert not obj._depth_async_scheduler.health(now=clock[0], worker_alive=False).busy
