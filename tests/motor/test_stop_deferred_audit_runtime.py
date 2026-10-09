"""Non-FOLLOW STOP diagnostics must leave the serial exclusion first."""

import json
import logging

import pytest

from test_follow_wheel_periodic import setup_periodic
from test_motion_write_fault import FaultDriver
from test_near_yaw_park_execution import request_park
from test_search_handoff_execution import arm


def observe_loggers(runtime, owner, *, separate):
    control = logging.Logger("deferred-stop-control", level=logging.INFO)
    backend = logging.Logger("deferred-stop-backend", level=logging.INFO) if separate else control
    runtime.logger, runtime.backend.logger = control, backend
    received = []

    class CheckedHandler(logging.Handler):
        def emit(self, record):
            # Record before asserting: deferred diagnostics deliberately
            # suppress handler failures, so an assertion alone could hide a
            # regression by being swallowed at the output boundary.
            locked = owner.motor_io_lock.locked()
            received.append((locked, record.msg, record.getMessage()))
            assert not locked

    control.addHandler(CheckedHandler())
    if separate:
        backend.addHandler(CheckedHandler())
    return received


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("entry", [
    "normal", "emergency", "fault", "shutdown", "search", "near",
])
def test_stop_entry_emits_audit_and_control_logs_after_serial_unlock(monkeypatch, separate, entry):
    runtime, owner, driver, _symbols, clock, _state = setup_periodic(monkeypatch)
    if entry == "search":
        arm(runtime, owner, clock)
    elif entry == "near":
        request_park(owner, clock)
    elif entry == "fault":
        runtime.backend.motion_write_fault = "prior_partial_write"
    elif entry == "shutdown":
        owner._runtime_shutdown_requested = True
    received = observe_loggers(runtime, owner, separate=separate)

    if entry in {"normal", "emergency"}:
        assert runtime.send_percent_brake(entry, "direct_stop")
    elif entry == "fault":
        assert runtime._service_motion_write_fault()
    elif entry == "shutdown":
        assert runtime._service_runtime_shutdown()
    elif entry == "search":
        assert runtime._service_search_reacquire_brake()
    else:
        assert runtime._service_near_yaw_park()

    audits = [json.loads(text.split("motor_zero_audit ", 1)[1])
              for _, message, text in received if message == "motor_zero_audit %s"]
    assert received and audits
    assert not any(locked for locked, _, _ in received)
    assert all(audit["writes_complete"] for audit in audits)
    expected_stops = [1] if entry in {"emergency", "fault", "shutdown"} else [1, 0]
    assert driver.stops == expected_stops
    assert driver.pairs == ([] if expected_stops == [1] else [(0, 0)])


@pytest.mark.parametrize("entry", ["search", "near"])
def test_repeated_parking_tick_stays_deferred_without_new_stop(monkeypatch, entry):
    runtime, owner, driver, _symbols, clock, _state = setup_periodic(monkeypatch)
    if entry == "search":
        arm(runtime, owner, clock)
        service = runtime._service_search_reacquire_brake
    else:
        request_park(owner, clock)
        service = runtime._service_near_yaw_park
    received = observe_loggers(runtime, owner, separate=True)
    assert service()
    first_stops = tuple(driver.stops)
    first_pairs = tuple(driver.pairs)
    clock[0] += .01
    assert service()
    assert tuple(driver.stops) == first_stops
    assert tuple(driver.pairs) == first_pairs
    assert not any(locked for locked, _, _ in received)


@pytest.mark.parametrize("entry", ["direct", "fault", "shutdown"])
def test_failed_stop_ack_flushes_outside_lock_and_preserves_safety(monkeypatch, entry):
    runtime, owner, _driver, _symbols, _clock, _state = setup_periodic(monkeypatch)
    driver = runtime.backend.driver = FaultDriver(fail_write=999, stop_fail={"right"})
    received = observe_loggers(runtime, owner, separate=True)
    if entry == "direct":
        with pytest.raises(OSError, match="STOP ACK"):
            runtime.send_percent_brake("emergency", "safety_test")
    elif entry == "fault":
        runtime.backend.motion_write_fault = "prior_fault"
        assert runtime._service_motion_write_fault()
    else:
        owner._runtime_shutdown_requested = True
        assert runtime._service_runtime_shutdown()
    assert driver.events == [("stop", "right", 1), ("stop", "left", 1)]
    assert runtime.backend.motion_write_fault
    assert runtime.backend.last_speed_receipt is None
    assert received and not any(locked for locked, _, _ in received)
    audit, = [json.loads(text.split("motor_zero_audit ", 1)[1])
              for _, message, text in received if message == "motor_zero_audit %s"]
    assert not audit["writes_complete"] and audit["failed_sides"] == ["right"]
    assert audit["acknowledged_sides"] == ["left"]


def test_broken_deferred_stop_handler_does_not_change_successful_stop(monkeypatch):
    runtime, owner, driver, _symbols, _clock, _state = setup_periodic(monkeypatch)
    observe_loggers(runtime, owner, separate=True)

    class BrokenHandler(logging.Handler):
        def emit(self, record):
            assert not owner.motor_io_lock.locked()
            raise OSError("sink unavailable")

    runtime.logger.handlers[:] = [BrokenHandler()]
    runtime.backend.logger.handlers[:] = [BrokenHandler()]
    assert runtime.send_percent_brake("emergency", "safety_test")
    assert driver.stops == [1] and not driver.pairs
    assert runtime.backend.motion_write_fault is None
    assert runtime.backend.last_speed_receipt is None
