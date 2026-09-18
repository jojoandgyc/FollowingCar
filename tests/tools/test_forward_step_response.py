"""All hardware is fake. Never import or open the real serial driver here."""
import json
import math
from types import SimpleNamespace

import pytest

from tools import test_forward_step_response as step


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class FakeSession:
    def __init__(self, clock, fail=False):
        self.clock = clock
        self.forward_signs = {"left": 1, "right": -1}
        self.bus_status = {"runtime_system_mode": 1}
        self.target = 0
        self.speed = {"left": 0.0, "right": 0.0}
        self.position = {"left": 0.0, "right": 0.0}
        self.last = {"left": clock(), "right": clock()}
        self.closed = self.opened = False
        self.fail = fail
        self.sent = []

    def open(self):
        self.opened = True

    def close(self, ensure_stop):
        assert ensure_stop
        self.target = 0
        self.closed = True

    def command(self, rpm, trial, phase):
        self.sent.append(rpm)
        self.target = rpm
        for side in ("right", "left"):
            started = self.clock()
            self.clock.sleep(.002)
            trial["commands"].append(dict(side=side, phase=phase, forward_rpm=rpm,
                                           raw_rpm=rpm * self.forward_signs[side],
                                           started=started, completed=self.clock(), acknowledged=True))

    def read_wheel(self, side):
        if self.target and self.fail:
            raise RuntimeError("fake serial failure")
        started = self.clock()
        self.clock.sleep(.002)
        dt = self.clock() - self.last[side]
        previous = self.speed[side]
        change = max(-400 * dt, min(300 * dt, self.target - previous))
        self.speed[side] += change
        self.position[side] += (previous + self.speed[side]) / 2 * 6 * dt
        self.last[side] = self.clock()
        sign = self.forward_signs[side]
        return dict(side=side, read_started=started, timestamp=self.clock(),
                    raw_rpm=round(self.speed[side] * sign), forward_rpm=round(self.speed[side]),
                    position_deg=round(self.position[side] * sign), error_code=0)


def args_for(*extra):
    return step.build_parser().parse_args(list(extra))


def sample(timestamp, rpm=0, position=0, side="left", **extra):
    row = dict(side=side, read_started=timestamp - .002, timestamp=timestamp,
               raw_rpm=rpm, forward_rpm=rpm, position_deg=position, error_code=0)
    row.update(extra)
    return row


@pytest.mark.parametrize("argv", [
    ["--rpms", "200"], ["--rpms", "-60"], ["--rpms", "60,60"], ["--rpms", ""],
    ["--duration", "nan"], ["--duration", "1.6"], ["--duration", "0"],
    ["--max-travel", "inf"], ["--max-travel", "3.1"], ["--max-total-travel", "7"],
    ["--max-total-travel", ".3"], ["--wheel-diameter", "nan"], ["--wheel-diameter", "0"],
    ["--settle-timeout", "5"], ["--repeats", "0"], ["--repeats", "4"],
])
def test_invalid_safety_arguments(argv):
    with pytest.raises(ValueError):
        step.validate_args(args_for(*argv))


def test_dry_run_never_constructs_hardware_or_checks_processes(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run touched execution path")
    monkeypatch.setattr(step, "ForwardSession", forbidden)
    monkeypatch.setattr(step, "_ensure_follow_runtime_stopped", forbidden)
    monkeypatch.setattr(step, "acquire_test_lock", forbidden)
    assert step.main(["--output-dir", str(tmp_path / "unused")]) == 0
    assert not (tmp_path / "unused").exists()


def test_run_is_direct_forward_steps_and_zeroes_with_observed_metrics():
    clock = Clock()
    session = FakeSession(clock)
    args = args_for()
    payload = {"trials": []}
    step.run_experiment(session, args, [60, 100], payload, clock, clock.sleep)
    assert session.sent == [60, 0, 100, 0]
    assert len(payload["trials"]) == 2
    for trial in payload["trials"]:
        assert trial["status"] == "complete"
        analysis = step.analyze_trial(trial, .26)
        for wheel in analysis.values():
            assert 0 < wheel["onset_sec"] < wheel["t50_sec"] < wheel["t90_sec"]
            assert wheel["average_acceleration_t90_m_s2"] > 0
            assert wheel["stop_latency_sec"] > 0
            assert wheel["stop_encoder_travel_m"] > 0
            assert wheel["max_sample_gap_sec"] <= step.FEEDBACK_MAX_AGE
            assert wheel["step_ack_duration_sec"] == pytest.approx(.002)


def test_no_t90_crossing_stays_unknown_and_not_a_command_based_estimate():
    trial = {"target_rpm": 100, "commands": [
        dict(side="left", phase="step", started=100, completed=100.01, acknowledged=True),
        dict(side="left", phase="zero", started=100.8, completed=100.81, acknowledged=True)],
        "samples": [dict(sample(100.1 + i * .1, speed), trusted=True, trial_path_m=0)
                    for i, speed in enumerate([0, 5, 20, 50, 60, 65, 65])]}
    wheel = step.analyze_trial(trial, .26)["left"]
    assert wheel["t50_sec"] == pytest.approx(.4)
    assert wheel["t90_sec"] is None
    assert wheel["average_acceleration_t90_m_s2"] is None
    assert wheel["peak_forward_rpm"] == 65
    assert wheel["stop_latency_sec"] is None
    assert any("t90_sec" in reason for reason in wheel["unknown_reasons"])


@pytest.mark.parametrize("changes,match", [
    ({"timestamp": 99.7, "read_started": 99.698}, "过期"),
    ({"timestamp": 100.1, "read_started": 100.098}, "过期"),
    ({"read_started": 99.7}, "耗时"), ({"error_code": 4}, "fault"),
    ({"forward_rpm": -3, "raw_rpm": -3}, "反转"),
    ({"forward_rpm": 111, "raw_rpm": 111}, "110"),
    ({"raw_rpm": float("nan")}, "非有限"),
])
def test_watchdog_rejects_untrustworthy_feedback(changes, match):
    watchdog = step.Watchdog(args_for(), {"left": 1, "right": -1}, Clock())
    with pytest.raises(RuntimeError, match=match):
        watchdog.check(sample(100, **changes) if "timestamp" not in changes else sample(**changes))


def test_watchdog_rejects_repeated_sample_and_encoder_jump():
    clock = Clock()
    watchdog = step.Watchdog(args_for(), {"left": 1, "right": -1}, clock)
    watchdog.check(sample(100))
    with pytest.raises(RuntimeError, match="不递增"):
        watchdog.check(sample(100))
    clock.sleep(.02)
    with pytest.raises(RuntimeError, match="突跳"):
        watchdog.check(sample(clock(), position=100))


def test_encoder_travel_limit_and_wrap():
    clock = Clock()
    args = args_for("--max-travel", ".2")
    watchdog = step.Watchdog(args, {"left": 1, "right": -1}, clock)
    watchdog.check(sample(clock(), position=(1 << 31) - 2))
    clock.sleep(.02)
    watchdog.check(sample(clock(), position=-(1 << 31) + 2))
    assert watchdog.total["left"] == pytest.approx(4 / 360 * math.pi * .26)
    watchdog.total["left"] = .199
    clock.sleep(.02)
    with pytest.raises(RuntimeError, match="单次"):
        watchdog.check(sample(clock(), position=-(1 << 31) + 4))


def test_same_port_test_lock_refuses_concurrent_run(tmp_path):
    first = step.acquire_test_lock(str(tmp_path / "fake_port"))
    try:
        with pytest.raises(RuntimeError, match="测试锁"):
            step.acquire_test_lock(str(tmp_path / "fake_port"))
    finally:
        first.close()
    with step.acquire_test_lock(str(tmp_path / "fake_port")):
        pass


def test_real_session_command_adapter_has_no_ramp_or_above_ceiling_escape():
    writes = []
    session = step.ForwardSession()
    session.forward_signs = {"left": 1, "right": -1}
    session.backend = SimpleNamespace(driver=SimpleNamespace(
        set_right_speed=lambda rpm: writes.append(("right", rpm)),
        set_left_speed=lambda rpm: writes.append(("left", rpm))))
    trial = {"commands": []}
    session.command(100, trial, "step")
    session.command(0, trial, "zero")
    assert writes == [("right", -100), ("left", 100), ("right", 0), ("left", 0)]
    with pytest.raises(ValueError):
        session.command(200, trial, "step")
    with pytest.raises(ValueError):
        session.command(-60, trial, "step")
    assert len(writes) == 4
    assert all(event["acknowledged"] for event in trial["commands"])


def test_final_cleanup_stops_both_sides_despite_write_failure_and_never_restores_current(monkeypatch):
    operations = []
    class Driver:
        def set_right_speed(self, value):
            operations.append(("right_speed", value))
            raise OSError("right zero write failed")
        def set_left_speed(self, value):
            operations.append(("left_speed", value))
        def stop(self, side, mode):
            operations.append((side, mode))
        def write_register(self, name, value, persist):
            operations.append((name, value, persist))
        def read_register(self, name):
            return 0
        def close(self):
            operations.append(("closed",))
    session = step.ForwardSession()
    driver = Driver()
    session.backend = SimpleNamespace(driver=driver, ensure_driver=lambda: driver)
    session.previous_parking_current = (5, 5)
    with pytest.raises(RuntimeError, match="right zero write failed"):
        session.close()
    assert ("left_speed", 0) in operations
    assert ("right", 1) in operations and ("left", 1) in operations
    assert ("right_parking_current", 0.0, False) in operations
    assert ("left_parking_current", 0.0, False) in operations
    assert operations[-1] == ("closed",)
    assert session.backend.driver is None


def test_rejected_nonfinite_sample_does_not_prevent_partial_results(tmp_path):
    payload = {"parameters": {"wheel_diameter": .26}, "status": "aborted", "trials": [
        {"target_rpm": 60, "commands": [], "samples": [dict(sample(100, float("nan")), trusted=False)]}]}
    json_path, csv_path = step.save_results(tmp_path, payload)
    loaded = json.loads(json_path.read_text())
    assert loaded["trials"][0]["samples"][0]["raw_rpm"] == {"invalid_nonfinite": "nan"}
    assert "nan" in csv_path.read_text()


def prepare_main(monkeypatch, tmp_path, confirmation="STEP"):
    import car_control_modular.config_loader as config
    monkeypatch.setattr(config, "load_config_to_env", lambda path: None)
    monkeypatch.setattr(step, "_ensure_follow_runtime_stopped", lambda: None)
    monkeypatch.setattr(step.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: confirmation)
    monkeypatch.setenv("MOTOR_RS485_PORT", str(tmp_path / "fake_serial"))
    return ["--execute", "--rpms", "60", "--output-dir", str(tmp_path)]


def test_execute_still_requires_typed_confirmation(monkeypatch, tmp_path):
    argv = prepare_main(monkeypatch, tmp_path, "no")
    monkeypatch.setattr(step, "ForwardSession", lambda: pytest.fail("opened without confirmation"))
    assert step.main(argv) == 2


def test_execute_refuses_active_follow_process(monkeypatch, tmp_path):
    argv = prepare_main(monkeypatch, tmp_path)
    def active():
        raise RuntimeError("主跟随程序仍在运行")
    monkeypatch.setattr(step, "_ensure_follow_runtime_stopped", active)
    monkeypatch.setattr(step, "ForwardSession", lambda: pytest.fail("opened with active follow"))
    assert step.main(argv) == 2


def test_hardware_failure_still_closes_and_saves_partial_data(monkeypatch, tmp_path):
    argv = prepare_main(monkeypatch, tmp_path)
    clock = Clock()
    session = FakeSession(clock, fail=True)
    monkeypatch.setattr(step, "ForwardSession", lambda: session)
    run = step.run_experiment
    monkeypatch.setattr(step, "run_experiment", lambda s, a, r, p: run(s, a, r, p, clock, clock.sleep))
    assert step.main(argv) == 1
    assert session.opened and session.closed
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert payload["status"] == "aborted"
    assert payload["trials"][0]["status"] == "aborted"
    assert payload["trials"][0]["samples"]
    assert payload["trials"][0]["analysis"]["left"]["t90_sec"] is None
    assert next(tmp_path.glob("*.csv")).is_file()


@pytest.mark.parametrize("runtime,configured,control,expected", [
    (0, 0, 0x38, "compat_mode_0_control_0x38"),
    (1, 1, 0x08, "independent_closed_loop_1"),
    (1, 1, 0x18, "independent_closed_loop_1"),
    (1, 1, 0x28, "independent_closed_loop_1"),
    (1, 1, 0x38, "independent_closed_loop_1"),
])
def test_supported_mode_combinations(runtime, configured, control, expected):
    assert step.validate_controller_mode(
        {"runtime_system_mode": runtime, "control_mode": control}, configured) == expected


@pytest.mark.parametrize("runtime,configured,control", [
    (0, 0, 0), (0, 0, 0x30), (0, 0, 0x18),  # No generic mode-0 bypass.
    (0, 0, 0x39), (1, 1, 0x39), (1, 1, 0x3A), (1, 1, 0x3F),
    (2, 2, 0x38), (3, 3, 0x38), (4, 4, 0x38), (9, 9, 0x38),
    (0, 1, 0x38), (1, 0, 0x38), (None, 0, 0x38), (0, None, 0x38),
    (0, 0, None), (True, 1, 0x38), (1, 1, 0x78),
])
def test_unsafe_unknown_or_inconsistent_mode_still_refused(runtime, configured, control):
    with pytest.raises(RuntimeError, match="未改变模式|拒绝自动改变"):
        step.validate_controller_mode(
            {"runtime_system_mode": runtime, "control_mode": control}, configured)


def fake_open_backend(monkeypatch, runtime_mode=0, configured_mode=0):
    """Exercise the real open/cleanup methods, not a replacement session.open."""
    import car_control_modular.mssd_motor as motor
    operations = []
    registers = {"system_mode": configured_mode, "foc_loop_mode": 1,
                 "closed_loop_acceleration": 350, "closed_loop_deceleration": 450,
                 "left_parking_current": 0, "right_parking_current": 0}
    def read(name):
        operations.append(("read", name))
        return registers[name]
    def write(name, value, persist=False):
        assert name in ("left_parking_current", "right_parking_current")
        assert value == 0 and persist is False
        operations.append(("write", name, value))
    driver = SimpleNamespace(
        set_right_speed=lambda rpm: operations.append(("right_speed", rpm)),
        set_left_speed=lambda rpm: operations.append(("left_speed", rpm)),
        stop=lambda side, mode: operations.append(("stop", side, mode)),
        read_bus_status=lambda: {"runtime_system_mode": runtime_mode, "control_mode": 0x38},
        read_register=read, write_register=write,
        close=lambda: operations.append(("close",)),
    )
    backend = SimpleNamespace(driver=driver, ensure_driver=lambda: driver,
        set_parking_current=lambda value, persist: operations.append(("parking", value, persist)))
    monkeypatch.setattr(motor, "MssdMotorBackend", lambda config: backend)
    return operations


@pytest.mark.parametrize("mode", [0, 1])
def test_actual_open_compatible_mode_does_not_reconfigure_or_send_motion(monkeypatch, mode):
    operations = fake_open_backend(monkeypatch, mode, mode)
    session = step.ForwardSession()
    session.open()
    assert session.controller_diagnostics["configured_system_mode"] == mode
    assert session.controller_diagnostics["closed_loop_acceleration"] == 350
    assert session.controller_diagnostics["closed_loop_deceleration"] == 450
    assert all(entry[1] == 0 for entry in operations if entry[0].endswith("_speed"))
    assert not any(entry[0] == "write" for entry in operations)
    session.close()
    assert operations[-1] == ("close",)


def test_rejected_mode_is_saved_even_when_open_fails_before_first_trial(monkeypatch, tmp_path):
    argv = prepare_main(monkeypatch, tmp_path)
    operations = fake_open_backend(monkeypatch, 4, 4)
    assert step.main(argv) == 1
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert payload["status"] == "aborted" and payload["trials"] == []
    assert payload["bus_status"]["runtime_system_mode"] == 4
    assert payload["controller_diagnostics"]["configured_system_mode"] == 4
    assert "runtime=4 configured=4 control=56" in payload["error"]
    assert all(entry[1] == 0 for entry in operations if entry[0].endswith("_speed"))
    assert ("stop", "left", 1) in operations and ("stop", "right", 1) in operations
    assert operations[-1] == ("close",)
