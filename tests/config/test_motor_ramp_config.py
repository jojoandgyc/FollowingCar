"""900 RPM/s trial profile, explicit overrides, real runtime binding."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from car_control_modular.config_loader import load_config_to_env

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "car_control_modular/config/reid_runtime.ini"


def test_profile_selects_900_acceleration_without_changing_deceleration_or_limits(monkeypatch):
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert env["MOTOR_RAMP_DIAGNOSTICS_ENABLE"] == "1"
    assert env["MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S"] == "900"
    assert "MOTOR_CLOSED_LOOP_DECELERATION_RPM_S" not in env
    assert env["MOTOR_RS485_MAX_TARGET"] == "200"
    assert env["ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC"] == "0.30"


@pytest.mark.parametrize("value", ["350", "600", "900"])
def test_explicit_acceleration_selection_survives_ini_default(monkeypatch, value):
    env = {"MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S": value}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert env["MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S"] == value


def test_empty_environment_value_inherits_profile(monkeypatch):
    env = {"MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S": ""}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert env["MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S"] == "900"


def test_real_entrypoint_uses_profile_900_without_an_environment_override():
    env = os.environ.copy()
    env.pop("MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S", None)
    code = """
import request_0513_modular as r
assert r.MSSD_MOTOR_CONFIG.closed_loop_acceleration_rpm_s == 900
assert r.MSSD_MOTOR_CONFIG.startup_parking_enabled
"""
    result = subprocess.run([sys.executable, "-c", code, "--config", str(CONFIG)],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", ["", "350", "600", "900", "9", "0", "65536", "350.5", "nan", "true"])
def test_real_entrypoint_validates_and_binds_without_hardware(value):
    env = os.environ.copy()
    env["MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S"] = value
    # Import evaluates config, but must never create a serial connection.
    code = """
import request_0513_modular as r
assert r.MSSD_MOTOR_CONFIG.ramp_diagnostics_enable
print('RAMP=', r.MSSD_MOTOR_CONFIG.closed_loop_acceleration_rpm_s)
"""
    result = subprocess.run([sys.executable, "-c", code, "--config", str(CONFIG)],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
    if value in ("", "350", "600", "900"):
        assert result.returncode == 0, result.stderr
        assert "RAMP= " + (value or "900") in result.stdout
    else:
        assert result.returncode != 0 and "ValueError" in result.stderr
