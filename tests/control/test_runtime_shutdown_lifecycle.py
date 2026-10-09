"""Shutdown reaches STOP before control-worker cleanup; no devices are opened."""
from types import SimpleNamespace
import threading

import pytest

import request_0513_modular as runtime


class BusyLock:
    def __init__(self):
        self.attempts = []

    def acquire(self, blocking=True):
        self.attempts.append(blocking)
        assert blocking is False, "shutdown must not block on a control-state mutex"
        return False


class Worker:
    def __init__(self, alive):
        self.alive = alive
        self.joins = []

    def join(self, timeout):
        self.joins.append(timeout)

    def is_alive(self):
        return self.alive


@pytest.fixture
def owner():
    obj = object.__new__(runtime.PersonTracker)
    obj.running = True
    obj._runtime_shutdown_requested = False
    obj._runtime_shutdown_reason = None
    obj._explicit_stop_requested = False
    obj.action_stop_event = threading.Event()
    obj._control_update_lock = BusyLock()
    obj._longitudinal_context_lock = threading.Lock()
    obj._longitudinal_stop_event = threading.Event()
    obj._longitudinal_wake_event = threading.Event()
    obj._longitudinal_context = {"old": "roi"}
    obj._depth30_linear_snapshot = ("forward", 50, 1, 100.0)
    obj._depth30_linear_timing = object()
    obj._motor_backend = SimpleNamespace(driver=object())
    obj.stops = []

    def stop(reason):
        assert obj._runtime_shutdown_requested and obj._explicit_stop_requested
        assert not obj.running
        assert not obj.action_stop_event.is_set(), "executor must survive until first STOP attempt"
        obj.stops.append(reason)

    obj._action_runtime = SimpleNamespace(send_stop_with_brake_hold=stop)
    return obj


def test_signal_latch_has_no_lock_io_logging_or_executor_shutdown(owner, monkeypatch):
    monkeypatch.setattr(runtime.logger, "info", lambda *a, **k: pytest.fail("signal callback logged"))
    owner._latch_runtime_shutdown("runtime_shutdown_signal")
    assert owner._control_update_lock.attempts == []
    assert owner.stops == [] and not owner.action_stop_event.is_set()
    assert not owner.running and owner._runtime_shutdown_requested
    assert owner.stop_action_execution and not owner._use_soft_stop_next


def test_direct_stop_ignores_busy_control_lock_and_preserves_first_reason(owner):
    owner._latch_runtime_shutdown("runtime_fault_capture")
    assert owner._send_runtime_shutdown_stop("runtime_shutdown")
    assert owner.stops == ["runtime_fault_capture"]
    assert owner._control_update_lock.attempts == []


def test_failed_initialization_does_not_open_motor_to_stop(owner):
    owner._motor_backend.driver = None
    owner._motor_backend.ensure_driver = lambda: pytest.fail("implicit driver initialization")
    assert not owner._send_runtime_shutdown_stop()
    assert owner.stops == [] and owner._runtime_shutdown_requested


def test_failed_stop_keeps_shutdown_latched_and_executor_available(owner):
    owner._action_runtime.send_stop_with_brake_hold = lambda _: (_ for _ in ()).throw(OSError("serial fault"))
    assert not owner._send_runtime_shutdown_stop()
    assert owner._runtime_shutdown_requested and not owner.running
    assert not owner.action_stop_event.is_set()


def test_live_depth_worker_reference_is_retained_after_bounded_join(owner):
    worker = owner._longitudinal_thread = Worker(True)
    owner._stop_longitudinal_thread()
    assert worker.joins == [.5]
    assert owner._longitudinal_thread is worker
    assert owner._longitudinal_stop_event.is_set() and owner._longitudinal_wake_event.is_set()
    assert owner._control_update_lock.attempts == []


def test_stopped_worker_cleanup_never_waits_for_control_lock(owner):
    owner._longitudinal_thread = Worker(False)
    owner._stop_longitudinal_thread()
    assert owner._longitudinal_thread is None
    assert owner._control_update_lock.attempts == [False]
    assert owner._longitudinal_context is not None


def test_stopped_worker_cleanup_never_waits_for_context_lock(owner):
    owner._longitudinal_thread = Worker(False)
    owner._control_update_lock = threading.Lock()
    owner._longitudinal_context_lock = BusyLock()
    owner._stop_longitudinal_thread()
    assert owner._longitudinal_context_lock.attempts == [False]
    assert not owner._control_update_lock.locked()


def test_quiet_worker_state_is_cleaned_without_controller_callbacks(owner):
    owner._longitudinal_thread = Worker(False)
    owner._control_update_lock = threading.Lock()
    owner._revoke_depth_linear_authority = lambda _: pytest.fail("unneeded cleanup callback")
    owner._stop_longitudinal_thread()
    assert owner._longitudinal_context is None
    assert owner._depth30_linear_snapshot is None and owner._depth30_linear_timing is None
    assert not owner._control_update_lock.locked()


def test_live_lateral_worker_is_not_forgotten_or_cleaned_concurrently(owner):
    owner._lateral_intent_stop_event = threading.Event()
    worker = owner._lateral_intent_thread = Worker(True)
    owner._lateral_intent_store = object()
    owner._clear_lateral_intent = lambda _: pytest.fail("concurrent lateral cleanup")
    owner._stop_lateral_intent_thread()
    assert worker.joins == [1.0] and owner._lateral_intent_thread is worker


def test_real_run_finally_stops_before_diagnostics_or_worker_joins(owner):
    events = []
    owner._vision_engine = "fake"
    owner.motor_io_lock = threading.Lock()
    owner._motor_backend.ensure_driver = lambda: None
    owner._motor_backend.close = lambda: events.append("motor_close")
    owner._start_action_thread = owner._start_lateral_intent_thread = owner._start_longitudinal_thread = lambda: None
    owner.process_frame = lambda: (_ for _ in ()).throw(RuntimeError("main failure"))
    original_stop = owner._action_runtime.send_stop_with_brake_hold
    owner._action_runtime.send_stop_with_brake_hold = lambda reason: (original_stop(reason), events.append("stop"))
    owner._action_runtime.send_robot_command = lambda _: events.append("second_stop")
    owner._action_runtime.join_feedback = lambda **kw: events.append("join_feedback")
    owner._follow_controller = SimpleNamespace(search_status=lambda: SimpleNamespace(progress_deg=0.0))
    owner._search_diagnostics = SimpleNamespace(active=True, finish=lambda *a, **k: events.append("diagnostics"))
    owner._stop_lateral_intent_thread = lambda: events.append("join_lateral")
    owner._stop_longitudinal_thread = lambda: events.append("join_depth")
    owner.action_thread = SimpleNamespace(join=lambda **kw: events.append("join_action"))
    owner._run_shutdown_step = lambda _, callback: callback()
    owner._close_camera_video_recorder = owner._release_rknn_camera = owner._close_rknn_pipeline = lambda: None
    owner._bunker_runtime = owner._sensor_runtime = SimpleNamespace(close=lambda: None)
    with pytest.raises(RuntimeError, match="main failure"):
        owner.run()
    assert events[:4] == ["stop", "diagnostics", "join_lateral", "join_depth"]
    assert owner.action_stop_event.is_set()


def test_main_installs_latch_only_signal_callback(owner, monkeypatch):
    handlers = {}
    monkeypatch.setattr(runtime, "PersonTracker", lambda **kw: owner)
    monkeypatch.setattr(runtime.signal, "signal", lambda signum, callback: handlers.setdefault(signum, callback))
    owner.run = lambda: handlers[runtime.signal.SIGTERM](runtime.signal.SIGTERM, None)
    runtime.main()
    assert owner._runtime_shutdown_reason == "runtime_shutdown_signal"
    assert not owner.running and owner.stops == []
    assert not owner.action_stop_event.is_set()


def test_deferred_shutdown_callback_is_not_reported_as_completed(owner, caplog):
    assert not owner._run_shutdown_step("fake blocked reader", lambda: False)
    assert "cleanup_deferred=True" in caplog.text
    assert "退出清理步骤完成" not in caplog.text
