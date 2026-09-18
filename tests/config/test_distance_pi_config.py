"""Explicit distance PI selection and reversible, hardware-free configuration."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from car_control_modular.config_loader import load_config_to_env


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "car_control_modular/config/reid_runtime.ini"


def test_runtime_profile_explicitly_selects_pi_and_retains_rollback(monkeypatch):
    env = {}
    monkeypatch.setattr(os, "environ", env)
    loaded = load_config_to_env(str(CONFIG))
    assert env["DISTANCE_CONTROL_MODE"] == "distance_pi"
    assert env["DISTANCE_APPROACH_ENABLE"] == "1"
    assert env["DISTANCE_PI_KP_PER_SEC"] == "3.0"
    assert env["DISTANCE_PI_KI_PER_SEC2"] == "0.4"
    assert env["DISTANCE_PI_INTEGRAL_MAX_M_S"] == "0.8"
    assert env["DISTANCE_PI_MEMORY_SEC"] == "0.35"
    assert env["DISTANCE_PI_MOTION_MEMORY_SEC"] == "0.35"
    assert env["DISTANCE_PI_LAUNCH_REQUEST_RPM"] == "180"
    assert env["ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC"] == "0.25"
    assert loaded.values["distance_pid"]["control_mode"] == "distance_pi"


@pytest.mark.parametrize("mode", ["distance_pi", "approach", "legacy"])
def test_explicit_process_mode_overrides_ini(monkeypatch, mode):
    env = {"FOLLOW_DISTANCE_CONTROL_MODE": mode}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert env["DISTANCE_CONTROL_MODE"] == mode
    assert env["DISTANCE_PI_KI_PER_SEC2"] == "0.4"


@pytest.mark.parametrize("option,expected", [
    ("", "legacy"), ("approach_enable=false", "legacy"),
    ("approach_enable=true", "approach"),
    ("approach_enable=true\ncontrol_mode=legacy", "legacy"),
    ("approach_enable=false\ncontrol_mode=distance_pi", "distance_pi"),
])
def test_old_config_falls_back_without_requiring_a_new_option(tmp_path, monkeypatch, option, expected):
    config = tmp_path / "follow.ini"
    config.write_text("[distance_pid]\nenable=true\n" + option, encoding="utf-8")
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(config))
    assert env["DISTANCE_CONTROL_MODE"] == expected
    assert "DISTANCE_PI_LAUNCH_REQUEST_RPM" not in env


@pytest.mark.parametrize("mode", ["pi", "DISTANCE_PI", "typo"])
def test_invalid_mode_fails_before_environment_mutation(monkeypatch, mode):
    env = {"FOLLOW_DISTANCE_CONTROL_MODE": mode}
    monkeypatch.setattr(os, "environ", env)
    with pytest.raises(ValueError, match="FOLLOW_DISTANCE_CONTROL_MODE"):
        load_config_to_env(str(CONFIG))
    assert env == {"FOLLOW_DISTANCE_CONTROL_MODE": mode}


def test_pi_p_trial_warns_that_it_does_not_tune_forward_pi(monkeypatch):
    env = {"FOLLOW_DISTANCE_P_TRIAL": "36"}
    monkeypatch.setattr(os, "environ", env)
    with pytest.warns(RuntimeWarning, match="does not tune distance_pi"):
        load_config_to_env(str(CONFIG))
    assert env["DISTANCE_CONTROL_MODE"] == "distance_pi"
    assert env["DISTANCE_PID_KP_RPM_PER_M"] == "36"
    assert env["DISTANCE_PI_KP_PER_SEC"] == "3.0"


@pytest.mark.parametrize("mode,approach", [("distance_pi", False), ("approach", True), ("legacy", False)])
def test_entrypoint_mode_override_is_effective_without_hardware(mode, approach):
    env = dict(os.environ, FOLLOW_DISTANCE_CONTROL_MODE=mode)
    env.pop("FOLLOW_DISTANCE_P_TRIAL", None)
    result = subprocess.run(
        [sys.executable, "-c", "import request_0513_modular as r; "
         f"assert r.DISTANCE_CONTROL_MODE == {mode!r}; "
         f"assert r.DISTANCE_APPROACH_ENABLE is {approach!r}", "--config", str(CONFIG)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_entrypoint_rejects_invalid_pi_gain_without_hardware(value):
    env = dict(os.environ, FOLLOW_DISTANCE_CONTROL_MODE="distance_pi", DISTANCE_PI_KI_PER_SEC2=value)
    env.pop("FOLLOW_DISTANCE_P_TRIAL", None)
    result = subprocess.run(
        [sys.executable, "-c", "import request_0513_modular"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "DISTANCE_PI_KI_PER_SEC2 must be finite and nonnegative" in result.stderr


@pytest.mark.parametrize("requested,expected", [("0.18", .18), ("0.25", .25), ("0.35", .25)])
def test_forward_grant_ttl_has_explicit_rollback_and_250ms_ceiling(requested, expected):
    env = dict(os.environ, ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC=requested,
               ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC="0.18")
    env.pop("FOLLOW_DISTANCE_P_TRIAL", None)
    result = subprocess.run(
        [sys.executable, "-c", "import request_0513_modular as r; "
         f"assert r.ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC == {expected!r}; "
         "assert r.ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC == .18; "
         "assert r.ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC == .18"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("value", ["-1", "201", "nan", "inf"])
def test_entrypoint_rejects_invalid_launch_before_hardware(value):
    env = dict(os.environ, DISTANCE_PI_LAUNCH_REQUEST_RPM=value)
    env.pop("FOLLOW_DISTANCE_P_TRIAL", None)
    result = subprocess.run([sys.executable, "-c", "import request_0513_modular"],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "DISTANCE_PI_LAUNCH_REQUEST_RPM must be finite and within 0..200" in result.stderr


def test_launch_main_profile_is_loaded_by_real_entrypoint(tmp_path):
    env = dict(os.environ)
    env.pop("FOLLOW_DISTANCE_P_TRIAL", None)
    result = subprocess.run(
        [sys.executable, "-c", "import request_0513_modular as r; "
         "assert r.DISTANCE_PI_LAUNCH_REQUEST_RPM == 180; "
         "assert r.DISTANCE_PI_KP_PER_SEC == 3; "
         "assert r.DISTANCE_PI_MOTION_MEMORY_SEC == .35; "
         "assert r.ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC == .25", "--config", str(CONFIG)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rollback = tmp_path / "rollback.ini"
    rollback.write_text(CONFIG.read_text(encoding="utf-8").replace(
        "pi_launch_request_rpm = 180", "pi_launch_request_rpm = 0"), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", "import request_0513_modular as r; "
         "assert r.DISTANCE_PI_LAUNCH_REQUEST_RPM == 0; "
         "assert r.DISTANCE_PI_KP_PER_SEC == 3", "--config", str(rollback)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
