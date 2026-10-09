"""Timing output runs after unlock and distinguishes waiting from work."""
from types import SimpleNamespace
import threading

import pytest

import request_0513_modular as runtime


def test_control_stage_output_is_after_unlock_with_separate_wait_and_cpu(monkeypatch):
    obj = object.__new__(runtime.PersonTracker)
    obj.running = True
    obj._runtime_shutdown_requested = False
    obj._control_update_lock = threading.Lock()
    ticks = iter([1.0, 1.020, 1.090])
    cpu = iter([.10, .11])
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(runtime.time, "thread_time", lambda: next(cpu))
    def process(*a, **kw):
        assert obj._control_update_lock.locked()
        obj._control_lock_stage_snapshot = (477, "vision", 65., 40., 30., 12., 25.)
    obj._queue_actions_for_persons_locked = process
    records = []
    def log(fmt, *args):
        assert not obj._control_update_lock.locked()
        records.append(fmt % args)
    monkeypatch.setattr(runtime.logger, "info", log)
    assert obj._queue_actions_for_persons(640, 480, [])
    assert "lock_wait_ms=20.00 lock_hold_ms=70.00 lock_hold_cpu_ms=10.00" in records[0]
    assert "ranging_ms=30.0 ranging_cpu_ms=12.0" in records[0]
    assert "log_after_unlock=True" in records[0]


def test_exception_releases_lock_and_records_wait_without_stale_stage(monkeypatch):
    obj = object.__new__(runtime.PersonTracker)
    obj.running = True
    obj._runtime_shutdown_requested = False
    obj._control_update_lock = threading.Lock()
    obj._control_lock_stage_snapshot = (1, "old", 99., 90., 80., 40., 9.)
    ticks = iter([1., 1.020, 1.090])
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: next(ticks))
    obj._queue_actions_for_persons_locked = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("decision fault"))
    records = []
    monkeypatch.setattr(runtime.logger, "info", lambda fmt, *args: records.append(fmt % args))
    with pytest.raises(RuntimeError, match="decision fault"):
        obj._queue_actions_for_persons(640, 480, [], evidence_capture_frame_id=499)
    assert not obj._control_update_lock.locked()
    assert "capture_frame_id=499 source=vision" in records[0]
    assert "process_ms=None" in records[0]


def test_shutdown_rejects_control_before_acquiring_any_lock():
    obj = object.__new__(runtime.PersonTracker)
    obj.running = False
    obj._runtime_shutdown_requested = True
    obj._control_update_lock = SimpleNamespace(acquire=lambda **kw: pytest.fail("shutdown waited on control"))
    assert obj._queue_actions_for_persons(640, 480, [])


def test_shutdown_arriving_during_lock_wait_never_starts_measurement():
    obj = object.__new__(runtime.PersonTracker)
    obj.running = True
    obj._runtime_shutdown_requested = False
    released = []
    def acquire(**kw):
        obj._runtime_shutdown_requested = True
        return True
    obj._control_update_lock = SimpleNamespace(acquire=acquire, release=lambda: released.append(True))
    obj._queue_actions_for_persons_locked = lambda *a, **kw: pytest.fail("measurement after shutdown")
    assert obj._queue_actions_for_persons(640, 480, [])
    assert released == [True]
