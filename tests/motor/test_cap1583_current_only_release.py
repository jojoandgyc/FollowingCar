"""After 500ms, dual 0A readback then one FREE STOP, never a speed write.

Real backend/executor with fake registers and clocks. No physical braking claim.
"""
import pytest

from test_cap973_bounded_parking import park
from test_cap1100_stop_command_order import trace_driver
from test_depth_drive_rpm import make_runtime
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("search", [False, True])
def test_expiry_reads_back_current_then_free_stop_without_speed(monkeypatch, search, caplog):
    rt, owner, driver, symbols, clock, req, evidence, label = park(monkeypatch, search)
    calls = trace_driver(monkeypatch, driver)
    for name in ("write_register", "read_register"):
        original = getattr(driver, name)
        def record(*args, name=name, original=original, **kw):
            calls.append((name, args, kw))
            return original(*args, **kw)
        monkeypatch.setattr(driver, name, record)
    with caplog.at_level("INFO"):
        clock[0] = 10.5
        rt._service_follow_wheels()
    assert calls == [
        ("write_register", ("right_parking_current", 0.), {"persist": False}),
        ("write_register", ("left_parking_current", 0.), {"persist": False}),
        ("read_register", ("right_parking_current",), {}),
        ("read_register", ("left_parking_current",), {}),
        ("stop_all", 2),
    ]
    assert evidence.current_released_at == 10.5
    assert rt.backend.normal_zero_hold and not rt.backend.motion_armed
    assert "zero_rpm=False speed_write=False motion_authorized=False" in caplog.text
    assert "PARK_RELEASE_ZERO" not in caplog.text
    assert "exit_stop_mode=free" in caplog.text
    calls.clear()
    for now in [10.55, 10.7, 11., 11.5]:
        clock[0] = now
        rt._service_follow_wheels()
        rt.send_percent_brake("normal", label)
        rt.send_percent_drive(0)
        rt.send_robot_command(symbols.rotate_right)
        # Even a legacy caller preparing speed I/O must not discard the hold
        # when current is already 0A. No zero write is required for recovery.
        rt.backend.prepare_speed_mode()
        rt.backend.send_targets(0, 0, "old_zero_keepalive")
    assert calls == []
    assert owner._brake_hold_active and not rt.backend.motion_armed


@pytest.mark.parametrize("search", [False, True])
def test_quiet_and_new_image_can_recover_without_a_zero_packet(monkeypatch, search):
    rt, owner, driver, symbols, clock, req, evidence, label = park(monkeypatch, search)
    calls = trace_driver(monkeypatch, driver)
    clock[0] = 10.5
    rt._service_follow_wheels()
    rt.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
    for now in [10.51, 10.56]:
        clock[0] = now
        rt._service_follow_wheels()
    assert not evidence.release_ready(10.49, feedback(10.56), 10.57)
    assert not evidence.release_ready(10.53, feedback(10.56), 10.57)
    assert evidence.release_ready(10.57, feedback(10.58), 10.58)
    assert calls == [("stop_all", 2)]  # Checking eligibility never sends speed or repeats FREE.


@pytest.mark.parametrize("pair", [(4, 4), (-4, -4), (24, -32)])
def test_backend_next_nonzero_command_and_next_park_still_work(monkeypatch, pair):
    rt, owner, driver, symbols = make_runtime()
    backend = rt.backend
    backend.send_stop("park", "normal")
    calls = trace_driver(monkeypatch, driver)
    backend.release_parking_current_only()
    backend.send_stop("ordinary_park_release_free", "free", preserve_zero=True)
    backend.send_targets(0, 0, "stale_zero")
    assert calls == [("stop_all", 2)] and backend.parking_current_a == 0
    calls.clear()
    # Caller has separately admitted a new motion command. Backend must not
    # insert a zero/STOP between current readback and this nonzero pair.
    backend.send_targets(*pair, "new_authorized_motion")
    assert calls == [("set_right_speed", pair[1]), ("set_left_speed", pair[0])]
    assert backend.motion_armed and not backend.normal_zero_hold
    calls.clear()
    backend.send_stop("new_park", "normal")
    assert backend.parking_current_a == 5
    assert calls[-1] == ("stop_all", 0)


@pytest.mark.parametrize("search", [False, True])
@pytest.mark.parametrize("event", ["write_failure", "hazard", "explicit_stop"])
def test_free_stop_failure_or_safety_change_never_releases_motion(monkeypatch, search, event):
    rt, owner, driver, symbols, clock, req, evidence, label = park(monkeypatch, search)
    calls = trace_driver(monkeypatch, driver)
    original = driver.stop_all
    def stop(mode):
        if int(mode) == 2:
            if event == "write_failure":
                raise OSError("one FREE STOP acknowledgement missing")
            if event == "hazard":
                rt.hard_stop_check = lambda _: True
            if event == "explicit_stop":
                owner._explicit_stop_requested = True
        return original(mode)
    monkeypatch.setattr(driver, "stop_all", stop)
    clock[0] = 10.5
    rt._service_follow_wheels()
    assert evidence.current_released_at is None
    assert evidence.fault == ("free_stop_failed" if event == "write_failure"
                              else "authority_changed_during_free_stop")
    assert rt.backend.parking_release_fault and not rt.backend.motion_armed
    assert owner._brake_hold_active
    assert calls[-1] == ("stop_all", 1)
    assert all(name == "stop_all" for name, value in calls)
    with pytest.raises(RuntimeError, match="blocked"):
        rt.backend.send_targets(4, 4, "old_motion")
