"""Configured startup acceleration, explicit override, real runtime binding."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from car_control_modular.config_loader import load_config_to_env

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "car_control_modular/config/reid_runtime.ini"


def test_profile_selects_600_acceleration_without_overriding_deceleration(monkeypatch):
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert env["MOTOR_RAMP_DIAGNOSTICS_ENABLE"] == "1"
    assert env["MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S"] == "600"
    assert "MOTOR_CLOSED_LOOP_DECELERATION_RPM_S" not in env
    assert env["MOTOR_RS485_MAX_TARGET"] == "200"
    assert env["ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC"] == "0.25"


def test_explicit_acceleration_selection_survives_ini_default(monkeypatch):
    env = {"MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S": "350"}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert env["MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S"] == "350"


@pytest.mark.parametrize("value", ["", "350", "600", "9", "0", "65536", "350.5", "nan", "true"])
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
    if value in ("", "350", "600"):
        assert result.returncode == 0, result.stderr
        assert "RAMP= " + (value or "600") in result.stdout
    else:
        assert result.returncode != 0 and "ValueError" in result.stderr
