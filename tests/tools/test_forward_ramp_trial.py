"""Optional forward ramp trials use fake I/O only; never open a serial port."""
import json
from types import SimpleNamespace

import pytest

from car_control_modular.motor_ramp import RampConfigurationError
from tools import test_forward_step_response as step


def args_for(*argv):
    return step.build_parser().parse_args(list(argv))


def fake_backend(monkeypatch, *, mode=0, failure=None):
    import car_control_modular.mssd_motor as motor

    operations, faults, configs = [], [], []
    registers = {
        "system_mode": mode, "foc_loop_mode": 1,
        "closed_loop_acceleration": 350, "closed_loop_deceleration": 450,
        "left_parking_current": 0, "right_parking_current": 0,
    }
    ramp_written = False

    def read(name):
        operations.append(("read", name))
        return registers[name]

    def read_pair(address, count):
        operations.append(("pair", address, count))
        assert (address, count) == (0x005F, 2)
        if failure == "initial_read" and not ramp_written:
            raise OSError("initial ramp read failed")
        if failure == "readback" and ramp_written:
            raise OSError("ramp readback failed")
        return [registers["closed_loop_acceleration"], registers["closed_loop_deceleration"]]

    def write(name, value, persist=False):
        nonlocal ramp_written
        operations.append(("write", name, value, persist))
        assert persist is False
        if name == "closed_loop_acceleration":
            assert not ramp_written, "must never retry or restore the ramp"
            ramp_written = True
            if failure == "write":
                raise OSError("ambiguous ramp write")
            if failure != "acceleration_mismatch":
                registers[name] = value
            if failure == "deceleration_mismatch":
                registers["closed_loop_deceleration"] = 451
        else:
            assert name in ("left_parking_current", "right_parking_current")
            assert value == 0
            registers[name] = value

    def bus():
        operations.append(("bus",))
        return {"runtime_system_mode": mode, "control_mode": 0x38}

    driver = SimpleNamespace(
        set_right_speed=lambda rpm: operations.append(("right_speed", rpm)),
        set_left_speed=lambda rpm: operations.append(("left_speed", rpm)),
        stop=lambda side, mode: operations.append(("stop", side, mode)),
        read_bus_status=bus, read_register=read, write_register=write,
        read_holding_registers=read_pair,
        close=lambda: operations.append(("close",)),
    )
    backend = SimpleNamespace(
        driver=driver, ensure_driver=lambda: driver,
        set_parking_current=lambda value, persist: operations.append(("parking", value, persist)),
        _record_motion_write_fault=lambda label, exc: faults.append((label, exc)),
    )

    def factory(config):
        configs.append(config)
        return backend

    monkeypatch.setattr(motor, "MssdMotorBackend", factory)
    return SimpleNamespace(backend=backend, operations=operations, faults=faults,
                           configs=configs, registers=registers)


def prepare_main(monkeypatch, tmp_path, confirmation="STEP"):
    import car_control_modular.config_loader as config

    monkeypatch.setattr(config, "load_config_to_env", lambda path: None)
    monkeypatch.setattr(step, "_ensure_follow_runtime_stopped", lambda: None)
    monkeypatch.setattr(step, "acquire_test_lock", lambda port: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(step.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: confirmation)
    monkeypatch.setenv("MOTOR_RS485_PORT", str(tmp_path / "fake_serial"))
    return ["--execute", "--rpms", "60", "--output-dir", str(tmp_path)]


@pytest.mark.parametrize("value", [10, 350, 600, 1000])
def test_explicit_trial_values_preserve_motion_limits(value):
    args = args_for("--acceleration-rpm-s", str(value))
    assert step.validate_args(args) == [60, 100]
    assert args.acceleration_rpm_s == value
    assert args.duration == .8 and args.max_travel == 2 and args.max_total_travel == 4
    assert step.MAX_RPM == 100


@pytest.mark.parametrize("value", [0, 9, 1001, 65535])
def test_trial_cli_is_more_bounded_than_controller_register(value):
    with pytest.raises(ValueError, match="10..1000"):
        step.validate_args(args_for("--acceleration-rpm-s", str(value)))


@pytest.mark.parametrize("value", ["350.0", "nan", "600rpm"])
def test_trial_cli_requires_integer(value):
    with pytest.raises(SystemExit):
        args_for("--acceleration-rpm-s", value)


@pytest.mark.parametrize("argv", [[], ["--acceleration-rpm-s", "600"]])
def test_dryrun_does_not_load_config_lock_or_construct_hardware(monkeypatch, tmp_path, capsys, argv):
    import car_control_modular.config_loader as config

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run entered the execution path")

    monkeypatch.setattr(config, "load_config_to_env", forbidden)
    monkeypatch.setattr(step, "ForwardSession", forbidden)
    monkeypatch.setattr(step, "_ensure_follow_runtime_stopped", forbidden)
    monkeypatch.setattr(step, "acquire_test_lock", forbidden)
    destination = tmp_path / "unused"
    assert step.main([*argv, "--output-dir", str(destination)]) == 0
    output = capsys.readouterr().out
    assert "不打开硬件" in output and not destination.exists()
    if argv:
        for text in ("0x005F=600", "0x06", "0x0060", "不持久化", "双轮共享", "影响转向",
                     "退出不恢复", "本次通电会保留", "断电后恢复"):
            assert text in output
    else:
        assert "斜坡参数默认只读" in output


def test_trial_warning_precedes_typed_confirmation(monkeypatch, tmp_path, capsys):
    argv = prepare_main(monkeypatch, tmp_path)
    monkeypatch.setattr(step, "ForwardSession", lambda: pytest.fail("unconfirmed hardware open"))

    def refuse(prompt):
        output = capsys.readouterr().out
        assert "STEP" in prompt
        assert "0x005F=600" in output and "影响转向" in output
        assert "退出不恢复" in output and "断电后恢复" in output
        return "no"

    monkeypatch.setattr("builtins.input", refuse)
    assert step.main([*argv, "--acceleration-rpm-s", "600"]) == 2


@pytest.mark.parametrize("explicit_none", [False, True])
def test_default_open_never_uses_helper_or_inherits_environment(monkeypatch, explicit_none):
    import car_control_modular.motor_ramp as ramp

    monkeypatch.setenv("MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S", "600")
    monkeypatch.setenv("MOTOR_RAMP_DIAGNOSTICS_ENABLE", "1")
    monkeypatch.setattr(ramp, "configure_closed_loop_ramp",
                        lambda *a, **k: pytest.fail("default tool must retain original SDK reads"))
    fake = fake_backend(monkeypatch)
    assert args_for().acceleration_rpm_s is None
    session = step.ForwardSession()
    if explicit_none:
        session.trial_acceleration_rpm_s = None
    session.open()
    assert fake.configs[0].closed_loop_acceleration_rpm_s is None
    assert fake.configs[0].ramp_diagnostics_enable is False
    assert session.controller_diagnostics["closed_loop_acceleration"] == 350
    assert "ramp_trial" not in session.controller_diagnostics
    assert not any(op[0] in ("write", "pair") for op in fake.operations)
    session.close()


@pytest.mark.parametrize("value", [350, 600])
def test_explicit_trial_is_after_safe_startup_and_never_restored(monkeypatch, value):
    fake = fake_backend(monkeypatch)
    session = step.ForwardSession()
    session.trial_acceleration_rpm_s = value
    session.open()
    diagnostic = session.controller_diagnostics["ramp_trial"]
    assert diagnostic == {
        "before_acceleration_rpm_s": 350, "acceleration_rpm_s": value,
        "deceleration_rpm_s": 450, "requested_acceleration_rpm_s": value,
        "changed": value != 350, "persist": False, "mode_policy": "compat_mode_0_control_0x38",
    }
    assert session.controller_diagnostics["closed_loop_acceleration"] == value
    assert session.controller_diagnostics["closed_loop_deceleration"] == 450
    first_pair = fake.operations.index(("pair", 0x005F, 2))
    for op in (("right_speed", 0), ("left_speed", 0), ("stop", "right", 1),
               ("stop", "left", 1), ("parking", 0, False), ("read", "system_mode"), ("bus",)):
        assert fake.operations.index(op) < first_pair
    session.close()
    ramp_writes = [op for op in fake.operations if op[:2] == ("write", "closed_loop_acceleration")]
    assert ramp_writes == ([] if value == 350 else [("write", "closed_loop_acceleration", value, False)])
    assert not any(op[:2] == ("write", "closed_loop_deceleration") for op in fake.operations)
    assert fake.registers["closed_loop_acceleration"] == value
    assert fake.registers["closed_loop_deceleration"] == 450
    assert all(op[1] == 0 for op in fake.operations if op[0].endswith("_speed"))
    assert not fake.faults and fake.operations[-1] == ("close",)


@pytest.mark.parametrize("failure", ["write", "readback", "acceleration_mismatch", "deceleration_mismatch"])
def test_failed_ramp_attempt_latches_fault_aborts_and_does_not_retry_or_restore(monkeypatch, failure):
    fake = fake_backend(monkeypatch, failure=failure)
    session = step.ForwardSession()
    session.trial_acceleration_rpm_s = 600
    with pytest.raises(RampConfigurationError) as caught:
        session.open()
    assert fake.faults == [("step_ramp_trial", caught.value)]
    diagnostic = session.controller_diagnostics["ramp_trial"]
    assert diagnostic["write_attempted"] and diagnostic["verified"] is False
    assert diagnostic["requested_acceleration_rpm_s"] == 600
    session.close()
    assert [op for op in fake.operations if op[:2] == ("write", "closed_loop_acceleration")] == [
        ("write", "closed_loop_acceleration", 600, False)]
    assert not any(op[:2] == ("write", "closed_loop_deceleration") for op in fake.operations)
    assert all(op[1] == 0 for op in fake.operations if op[0].endswith("_speed"))
    assert fake.operations[-1] == ("close",)


@pytest.mark.parametrize("kwargs", [{"mode": 4}, {"failure": "initial_read"}])
def test_prewrite_failure_never_modifies_ramp_or_latches_write_fault(monkeypatch, kwargs):
    fake = fake_backend(monkeypatch, **kwargs)
    session = step.ForwardSession()
    session.trial_acceleration_rpm_s = 600
    with pytest.raises((RuntimeError, OSError)):
        session.open()
    assert not fake.faults
    assert not any(op[0] == "write" for op in fake.operations)
    session.close()


@pytest.mark.parametrize("failure", [None, "write"])
def test_main_saves_explicit_trial_and_aborts_before_experiment_on_ramp_fault(monkeypatch, tmp_path, failure):
    argv = prepare_main(monkeypatch, tmp_path)
    fake = fake_backend(monkeypatch, failure=failure)
    runs = []
    monkeypatch.setattr(step, "run_experiment", lambda session, args, rpms, payload: runs.append(args))
    assert step.main([*argv, "--acceleration-rpm-s", "600"]) == (1 if failure else 0)
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert payload["parameters"]["acceleration_rpm_s"] == 600
    assert payload["controller_diagnostics"]["ramp_trial"]["requested_acceleration_rpm_s"] == 600
    if not failure:
        assert payload["controller_diagnostics"]["closed_loop_acceleration"] == 600
    assert payload["status"] == ("aborted" if failure else "complete")
    assert len(runs) == (0 if failure else 1)
    assert fake.operations[-1] == ("close",)
