"""Optional capture sidecars cannot remove control RGB or hide reader death."""
from collections import deque
from types import SimpleNamespace
import queue
import threading

import pytest

import request_0513_modular as runtime


@pytest.fixture
def owner(monkeypatch):
    obj = object.__new__(runtime.PersonTracker)
    obj.running = True
    obj._runtime_shutdown_requested = False
    obj._runtime_shutdown_reason = None
    obj._capture_stop_event = threading.Event()
    obj._capture_state_lock = threading.Lock()
    obj._capture_frame_id = 0
    obj._capture_metadata_ring = deque(maxlen=30)
    obj._capture_queue = queue.Queue(2)
    obj._capture_failure_reason = None
    obj._capture_started_ts = 99.5
    obj._capture_last_frame_ts = 100.0
    obj._capture_thread = SimpleNamespace(is_alive=lambda: True)
    obj._camera_video_recorder = None
    obj._direction_pool = SimpleNamespace(submit=lambda *a: None)
    obj._action_runtime = SimpleNamespace(get_recording_feedback=lambda: None)
    obj._record_depth_diagnostic_rgb = lambda *a: None
    obj.stops = []
    def stop(reason):
        obj._latch_runtime_shutdown(reason)
        obj.stops.append(reason)
    obj._send_runtime_shutdown_stop = stop
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(runtime, "recording_authority", lambda _: None)
    return obj


def camera_frames(owner, count=2):
    calls = []
    def read():
        calls.append(None)
        if len(calls) > count:
            owner._capture_stop_event.set()
            return False, None
        return True, SimpleNamespace(shape=(480, 640, 3), number=len(calls))
    owner._rknn_camera = SimpleNamespace(read=read)


@pytest.mark.parametrize("sidecar", ["recorder", "direction", "feedback"])
def test_optional_failure_occurs_after_control_publish_and_does_not_kill_capture(owner, sidecar):
    calls = []
    def fail(*a, **k):
        assert not owner._capture_queue.empty(), "sidecar ran before publishing the good control RGB"
        calls.append("failure")
        raise RuntimeError("optional consumer broke")
    if sidecar == "recorder":
        owner._camera_video_recorder = SimpleNamespace(submit=fail)
    elif sidecar == "direction":
        owner._direction_pool.submit = fail
    else:
        owner._camera_video_recorder = SimpleNamespace(submit=lambda *a, **k: None)
        owner._action_runtime.get_recording_feedback = fail
    camera_frames(owner)
    pool = owner._direction_pool
    owner._capture_loop()
    assert calls == ["failure"], "bad sidecar should be disabled after its first failure"
    assert [owner._capture_queue.get_nowait()[0] for _ in range(2)] == [1, 2]
    assert owner._capture_failure_reason is None and owner.stops == [] and owner.running
    assert owner._direction_pool is pool, "changing this ref would select a second camera reader"


def test_recorder_false_result_is_overload_not_failure(owner):
    calls = []
    owner._camera_video_recorder = SimpleNamespace(submit=lambda *a, **k: calls.append(k) or False)
    camera_frames(owner)
    owner._capture_loop()
    assert len(calls) == 2 and not getattr(owner, "_capture_recorder_disabled", False)


def test_both_optional_consumers_fail_independently(owner):
    calls = []
    def fail(name):
        def callback(*a, **k):
            calls.append(name)
            raise OSError(name)
        return callback
    owner._camera_video_recorder = SimpleNamespace(submit=fail("record"))
    owner._direction_pool.submit = fail("direction")
    camera_frames(owner)
    owner._capture_loop()
    assert calls == ["record", "direction"]
    assert owner._capture_queue.qsize() == 2 and owner.stops == []


def test_full_main_queue_retains_latest_frame_before_sidecars(owner):
    camera_frames(owner, count=3)
    owner._capture_loop()
    assert [owner._capture_queue.get_nowait()[0] for _ in range(2)] == [2, 3]


def test_unexpected_reader_exception_is_explicit_fault_and_stops(owner):
    owner._capture_frames = lambda: (_ for _ in ()).throw(ValueError("invalid frame"))
    owner._capture_loop()
    assert owner._capture_failure_reason == "ValueError:invalid frame"
    assert owner.stops == ["runtime_fault_capture"]
    assert not owner.running and owner._runtime_shutdown_requested


@pytest.mark.parametrize("case", ["dead", "stalled", "startup_stalled"])
def test_dead_or_stalled_capture_is_not_silently_restarted(owner, case):
    if case == "dead":
        owner._capture_thread.is_alive = lambda: False
    elif case == "stalled":
        owner._capture_last_frame_ts = 98.0
    else:
        owner._capture_last_frame_ts = 0.0
        owner._capture_started_ts = 98.0
    assert not owner._capture_health_ok()
    assert len(owner.stops) == 1 and owner._capture_failure_reason
    assert not owner._capture_health_ok()
    assert len(owner.stops) == 1


def test_single_delayed_frame_does_not_extend_or_shortcut_motion_ttl(owner):
    owner._capture_last_frame_ts = 99.1
    grant = owner._depth30_linear_snapshot = ("forward", 30, 1, 99.0)
    assert owner._capture_health_ok()
    assert owner._depth30_linear_snapshot is grant and owner.stops == []


def test_sync_mode_has_no_capture_thread_to_restart(owner):
    owner._capture_thread = None
    assert owner._capture_health_ok()


def test_dead_existing_thread_start_request_is_fault_not_reopen(owner, monkeypatch):
    owner._capture_thread.is_alive = lambda: False
    monkeypatch.setattr(runtime.threading, "Thread", lambda **kw: pytest.fail("unexpected reader restart"))
    owner._start_capture_thread()
    assert owner.stops == ["runtime_fault_capture_thread_dead"]


def test_thread_start_failure_latches_shutdown(owner, monkeypatch):
    owner._capture_thread = None
    def start():
        raise RuntimeError("can't start new thread")
    thread = SimpleNamespace(start=start, is_alive=lambda: False)
    monkeypatch.setattr(runtime.threading, "Thread", lambda **kw: thread)
    owner._start_capture_thread()
    assert owner.stops == ["runtime_fault_capture"]
    assert owner._capture_thread is thread
    assert "thread_start:RuntimeError" in owner._capture_failure_reason


def test_empty_queue_main_path_checks_health_again(owner, monkeypatch):
    monkeypatch.setattr(runtime, "RKNN_CAMERA_ENABLE", True)
    owner._rknn_camera = object()
    owner._ensure_rknn_camera_started = lambda: None
    owner._drain_direction_results = lambda: None
    health = []
    owner._capture_health_ok = lambda: health.append(None) or True
    owner._capture_queue = SimpleNamespace(get=lambda **kw: (_ for _ in ()).throw(queue.Empty()))
    owner._process_frame_rknn_camera()
    assert len(health) == 2


def test_live_native_reader_retains_owner_and_defers_release(owner):
    joins = []
    thread = owner._capture_thread = SimpleNamespace(
        is_alive=lambda: True, join=lambda **kw: joins.append(kw))
    camera = owner._rknn_camera = SimpleNamespace(release=lambda: pytest.fail("release raced native read"))
    assert owner._release_rknn_camera() is False
    assert owner._capture_thread is thread and owner._rknn_camera is camera
    assert joins == [{"timeout": runtime.SHUTDOWN_STEP_TIMEOUT_SEC}]
    assert owner._capture_stop_event.is_set()


def test_dead_or_never_started_reader_can_release_without_join(owner):
    owner._capture_thread = SimpleNamespace(is_alive=lambda: False,
        join=lambda **kw: pytest.fail("joining a never-started reader"))
    released = []
    owner._rknn_camera = SimpleNamespace(release=lambda: released.append(True))
    assert owner._release_rknn_camera() is True
    assert released == [True]
    assert owner._capture_thread is None and owner._rknn_camera is None


def test_reader_exits_during_join_before_camera_release(owner):
    events = []
    alive = [True]
    def join(**kw):
        events.append("joined")
        alive[0] = False
    owner._capture_thread = SimpleNamespace(is_alive=lambda: alive[0], join=join)
    owner._rknn_camera = SimpleNamespace(release=lambda: events.append("released"))
    assert owner._release_rknn_camera() is True
    assert events == ["joined", "released"]
