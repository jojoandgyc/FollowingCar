"""CAP1100: fault STOP must never be followed/preceded by speed-mode writes.

All tests use fake drivers; packet order is not evidence of physical braking.
"""
from dataclasses import replace

import pytest

from test_depth_drive_rpm import make_runtime
from test_cap973_bounded_parking import park


def trace_driver(monkeypatch, driver):
    calls = []
    for name in ("set_right_speed", "set_left_speed", "stop_all"):
        original = getattr(driver, name)
        def record(value, name=name, original=original):
            calls.append((name, int(value)))
            return original(value)
        monkeypatch.setattr(driver, name, record)
    return calls


@pytest.mark.parametrize("mode,value", [("emergency", 1), ("free", 2)])
def test_explicit_stop_and_refresh_only_write_stop(monkeypatch, mode, value):
    rt, _, driver, _ = make_runtime()
    calls = trace_driver(monkeypatch, driver)
    rt.backend.config = replace(rt.backend.config, stop_zero_delay_sec=1.)
    monkeypatch.setattr("car_control_modular.mssd_motor.time.sleep",
                        lambda _: pytest.fail("explicit STOP must not wait for zero-speed delay"))
    for _ in range(3):
        rt.backend.motion_armed = True
        rt.backend.send_stop("hold", mode=mode)
        assert not rt.backend.motion_armed
    assert calls == [("stop_all", value)] * 3
    assert not driver.register_writes


def test_fault_zero_keepalive_and_old_normal_cannot_overwrite_emergency(monkeypatch):
    rt, _, driver, _ = make_runtime()
    calls = trace_driver(monkeypatch, driver)
    rt.backend.parking_release_fault = "free_stop_failed"
    rt.backend.send_stop("fault", "emergency")
    rt.backend.send_targets(0, 0, "old_periodic_zero")
    rt.backend.send_stop("old_normal", "normal")
    rt.backend.refresh_normal_stop("old_normal_refresh")
    with pytest.raises(RuntimeError, match="blocked"):
        rt.backend.send_targets(7, -7, "old_turn")
    assert calls and all(call == ("stop_all", 1) for call in calls)
    assert not driver.register_writes
    assert rt.backend.parking_current_a == 0
    assert not rt.backend.motion_armed


@pytest.mark.parametrize("search", [False, True])
def test_bounded_park_to_fault_and_runtime_refresh_no_zero_reentry(monkeypatch, search):
    rt, owner, driver, _, clock, _, evidence, _ = park(monkeypatch, search)
    calls = trace_driver(monkeypatch, driver)
    original = driver.stop_all
    def failed_free(mode):
        original(mode)
        if mode == 2:
            raise OSError("FREE acknowledgement missing")
    monkeypatch.setattr(driver, "stop_all", failed_free)
    clock[0] = 10.50
    rt._service_follow_wheels()
    assert evidence.current_released_at is None
    assert evidence.fault == "free_stop_failed"
    assert calls == [("stop_all", 2), ("stop_all", 1)]  # failed FREE then EMERGENCY
    calls.clear()
    for now in [14., 15.2, 16.4]:
        clock[0] = now
        rt.send_percent_brake(owner._brake_hold_stop_mode, owner._brake_hold_label)
        rt.backend.send_targets(0, 0, "stale_zero")
    assert all(call == ("stop_all", 1) for call in calls)
    assert rt.backend.parking_current_a == 0
    assert owner._brake_hold_active


def test_failed_stop_does_not_claim_success_or_fall_back_to_speed(monkeypatch, caplog):
    rt, _, driver, _ = make_runtime()
    calls = trace_driver(monkeypatch, driver)
    stop_attempts = []
    def fail(_):
        stop_attempts.append(1)
        raise OSError("left stop acknowledgement missing")
    monkeypatch.setattr(driver, "stop_all", fail)
    rt.backend.motion_armed = True
    rt.backend.parking_release_fault = "free_stop_failed"
    with caplog.at_level("INFO"):
        for _ in range(2):
            with pytest.raises(OSError):
                rt.backend.send_stop("fault", "emergency")
    assert len(stop_attempts) == 2
    assert not calls
    assert not rt.backend.motion_armed
    assert "stop_registers_sent" not in caplog.text
    with pytest.raises(RuntimeError, match="blocked"):
        rt.backend.send_targets(1, 1, "retry_motion")


def test_normal_lifecycle_and_ordinary_speed_zero_stay_distinct(monkeypatch):
    rt, _, driver, _ = make_runtime()
    calls = trace_driver(monkeypatch, driver)
    rt.backend.send_stop("ordinary_park", "normal")
    assert calls == [("set_right_speed", 0), ("set_left_speed", 0), ("stop_all", 1), ("stop_all", 0)]
    assert rt.backend.parking_current_a == 5
    calls.clear()
    rt.backend.send_targets(0, 0, "authorized_current_phase_exit")
    assert rt.backend.parking_current_a == 0
    assert calls == [("set_right_speed", 0), ("set_left_speed", 0)]


def test_fault_shutdown_never_returns_to_speed_mode(monkeypatch):
    rt, _, driver, _ = make_runtime()
    calls = trace_driver(monkeypatch, driver)
    rt.backend.parking_release_fault = "free_stop_failed"
    rt.backend.close()
    assert calls == [("stop_all", 1)]
    assert all(value == 0 for value in driver.registers.values())


def test_stop_log_describes_writes_not_physical_stillness(monkeypatch, caplog):
    rt, _, driver, _ = make_runtime()
    with caplog.at_level("INFO"):
        rt.backend.send_stop("fault", "emergency")
    assert "final_motor_command=stop" in caplog.text
    assert "pre_zero=False post_zero=False" in caplog.text
    assert "physical_stillness=unverified" in caplog.text
