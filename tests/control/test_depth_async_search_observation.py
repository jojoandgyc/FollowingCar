"""Search candidate depth is an exclusive fallback, never a parallel scan."""
import threading

import pytest

import request_0513_modular as runtime
from car_control_modular.astra_depth import AstraDepthRuntime
from car_control_modular.control_types import DistanceState
from test_depth_async_runtime import async_scene, visual, scene, owner, NOW, start_loop, join_loop


@pytest.fixture
def search_scene(async_scene, monkeypatch):
    obj, camera, distance, clock = async_scene
    observation = obj._longitudinal_context["person_targets"][0].depth_observation
    monkeypatch.setattr(runtime, "resolve_depth_target_observation", lambda **kw: observation)
    obj._confirmed_search_reacquire_uid = 1
    obj._confirmed_search_reacquire_depth_streak = 2
    obj._confirmed_search_reacquire_depth_last_frame = obj.frame_index - 1
    obj._last_frame_distance_state = DistanceState(
        source="vision_depth", used_distance_m=1.9, sample_timestamp=NOW - .08)
    camera._depth_history.append((NOW - .06, camera._latest_depth))
    return async_scene


def observe(obj):
    return obj._observe_search_reacquire_depth(
        bbox=(240., 40., 400., 440.), track_id=3, uid=1, score=.95, area=64000,
        width=640, height=480)


def test_search_observation_busy_preserves_cache_confirmation_and_motor_grant(search_scene, monkeypatch):
    obj, camera, distance, clock = search_scene
    scheduler = obj._depth_async_scheduler
    flight = scheduler.begin(now=clock[0], max_capture_age_sec=.25)
    before = (obj._last_frame_distance_state, obj._depth30_linear_snapshot,
              obj._confirmed_search_reacquire_depth_streak,
              obj._confirmed_search_reacquire_depth_last_frame)
    monkeypatch.setattr(distance, "get_frame_distance_state",
                        lambda *a, **kw: pytest.fail("search bypassed busy scan slot"))
    valid, state = observe(obj)
    assert not valid and state.source_detail == "depth_async_search_pending"
    assert state.raw_distance_m is None and state.sample_timestamp is None
    assert (obj._last_frame_distance_state, obj._depth30_linear_snapshot,
            obj._confirmed_search_reacquire_depth_streak,
            obj._confirmed_search_reacquire_depth_last_frame) == before
    assert not scheduler.valid(flight)
    assert scheduler.health(now=clock[0], worker_alive=True).busy
    assert scheduler.finish(flight, now=clock[0])


def test_search_observation_runs_real_rgb_aligned_scan_in_reserved_slot(search_scene, monkeypatch):
    obj, camera, distance, clock = search_scene
    scans = []
    original = AstraDepthRuntime._select_multiregion_distance
    grant = obj._depth30_linear_snapshot

    def scan(live, *a, **kw):
        assert live is camera
        scheduler = obj._depth_async_scheduler
        assert scheduler.health(now=clock[0], worker_alive=True).busy
        assert scheduler.begin(now=clock[0], max_capture_age_sec=.25) is None
        scans.append(live)
        return original(live, *a, **kw)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    valid, state = observe(obj)
    assert valid and state.raw_distance_m == pytest.approx(1.8)
    assert state.sample_timestamp == NOW - .06  # Original RGB-aligned history.
    assert obj._confirmed_search_reacquire_depth_streak == 3
    assert obj._last_frame_distance_state is state
    assert obj._depth30_linear_snapshot is grant
    assert len(scans) == 1
    assert not obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).busy


def test_search_sync_exception_always_releases_slot(search_scene, monkeypatch):
    obj, camera, distance, clock = search_scene
    previous = obj._last_frame_distance_state

    def explode(*a, **kw):
        raise RuntimeError("search scan failed")

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", explode)
    with pytest.raises(RuntimeError, match="search scan failed"):
        observe(obj)
    assert not obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).busy
    assert obj._last_frame_distance_state is previous
    assert obj._confirmed_search_reacquire_depth_streak == 2


@pytest.mark.parametrize("reason", ["identity_rejected", "stop", "hazard"])
def test_revoked_search_observation_cannot_advance_confirmation_or_display(search_scene, monkeypatch, reason):
    obj, camera, distance, clock = search_scene
    previous, grant = obj._last_frame_distance_state, obj._depth30_linear_snapshot

    def measure(*a, **kw):
        obj._depth_async_scheduler.revoke(reason)
        return DistanceState(source="vision_depth", source_detail="depth_multiregion",
            raw_distance_m=1.8, used_distance_m=1.8, sample_count=1,
            sample_age_sec=.06, sample_timestamp=NOW - .06)

    monkeypatch.setattr(distance, "get_frame_distance_state", measure)
    valid, state = observe(obj)
    assert not valid and state.sample_timestamp is None
    assert obj._last_frame_distance_state is previous
    assert obj._confirmed_search_reacquire_depth_streak == 2
    assert obj._depth30_linear_snapshot is grant
    assert not obj._depth_async_scheduler.health(now=clock[0], worker_alive=True).busy


def test_real_depth30_busy_search_attempt_never_starts_second_scan(search_scene, monkeypatch):
    obj, camera, distance, clock = search_scene
    entered, resume = threading.Event(), threading.Event()
    scans = []
    original = AstraDepthRuntime._select_multiregion_distance
    previous = obj._last_frame_distance_state

    def scan(private, *a, **kw):
        scans.append(private)
        entered.set()
        assert resume.wait(timeout=3)
        return original(private, *a, **kw)

    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", scan)
    thread, failures = start_loop(obj)
    try:
        assert entered.wait(timeout=2)
        valid, state = observe(obj)
        assert not valid and len(scans) == 1
        assert obj._last_frame_distance_state is previous
        assert obj._confirmed_search_reacquire_depth_streak == 2
        resume.set()
        thread.join(timeout=3)
        assert not obj.calls
        assert camera._last_accepted_ts == 0
        # Once the revoked worker has exited, search can use its own aligned scan.
        valid, state = observe(obj)
        assert valid and len(scans) == 2
        assert state.sample_timestamp == NOW - .06
    finally:
        resume.set()
        join_loop(obj, thread, failures)

