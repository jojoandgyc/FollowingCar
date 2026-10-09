"""Real launcher configuration removes target-motion control, without hardware."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from car_control_modular.config_loader import load_config_to_env

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "car_control_modular/config/reid_runtime.ini"


@pytest.mark.parametrize("override, expected", [(None, "0"), ("1", "1"), ("false", "0")])
def test_profile_default_and_explicit_legacy_comparison(monkeypatch, override, expected):
    env = {} if override is None else {"DISTANCE_TARGET_MOTION_CONTROL_ENABLE": override}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert env["DISTANCE_TARGET_MOTION_CONTROL_ENABLE"] == expected
    assert env["DISTANCE_CONTROL_MODE"] == "distance_pi"
    assert env["ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC"] == "0.30"
    assert env["DISTANCE_PI_KP_PER_SEC"] == "3.0"


def test_invalid_switch_rejected_before_partial_environment_changes(monkeypatch):
    env = {"DISTANCE_TARGET_MOTION_CONTROL_ENABLE": "unknown"}
    monkeypatch.setattr(os, "environ", env)
    with pytest.raises(ValueError, match="target_motion_control_enable must be a boolean"):
        load_config_to_env(str(CONFIG))
    assert env == {"DISTANCE_TARGET_MOTION_CONTROL_ENABLE": "unknown"}


@pytest.mark.parametrize("mode", ["approach", "legacy"])
@pytest.mark.parametrize("switch", [None, "0", "false", "no", "off"])
def test_distance_only_rejects_legacy_mode_before_environment_changes(monkeypatch, mode, switch):
    env = {"FOLLOW_DISTANCE_CONTROL_MODE": mode}
    if switch is not None:
        env["DISTANCE_TARGET_MOTION_CONTROL_ENABLE"] = switch
    initial = dict(env)
    monkeypatch.setattr(os, "environ", env)
    with pytest.raises(ValueError, match="false requires distance_pi"):
        load_config_to_env(str(CONFIG))
    assert env == initial


@pytest.mark.parametrize("mode", ["approach", "legacy"])
def test_legacy_rollback_requires_explicit_motion_enable(monkeypatch, mode):
    env = {"FOLLOW_DISTANCE_CONTROL_MODE": mode, "DISTANCE_TARGET_MOTION_CONTROL_ENABLE": "1"}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert env["DISTANCE_CONTROL_MODE"] == mode
    assert env["DISTANCE_TARGET_MOTION_CONTROL_ENABLE"] == "1"


@pytest.mark.parametrize("mode", ["approach", "legacy"])
def test_invalid_ini_combination_rejected_before_environment_changes(tmp_path, monkeypatch, mode):
    config = tmp_path / "invalid.ini"
    config.write_text(
        f"[distance_pid]\ncontrol_mode={mode}\ntarget_motion_control_enable=false\n",
        encoding="utf-8")
    env = {}
    monkeypatch.setattr(os, "environ", env)
    with pytest.raises(ValueError, match="false requires distance_pi"):
        load_config_to_env(str(config))
    assert env == {}


def test_real_entrypoint_loads_disabled_control_and_binds_policy():
    env = dict(os.environ)
    for name in ("DISTANCE_TARGET_MOTION_CONTROL_ENABLE", "FOLLOW_DISTANCE_CONTROL_MODE"):
        env.pop(name, None)
    result = subprocess.run([sys.executable, "-c",
        "import request_0513_modular as r; "
        "assert r.DISTANCE_CONTROL_MODE == 'distance_pi'; "
        "assert r.DISTANCE_TARGET_MOTION_CONTROL_ENABLE is False",
        "--config", str(CONFIG)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    tree = ast.parse((ROOT / "request_0513_modular.py").read_text())
    bindings = [kw.value.id for node in ast.walk(tree) if isinstance(node, ast.Call)
        for kw in node.keywords if kw.arg == "distance_target_motion_control_enable"
        and isinstance(kw.value, ast.Name)]
    assert bindings == ["DISTANCE_TARGET_MOTION_CONTROL_ENABLE"]


def test_direct_entrypoint_rejects_invalid_switch():
    env = dict(os.environ, DISTANCE_TARGET_MOTION_CONTROL_ENABLE="unknown")
    result = subprocess.run([sys.executable, "-c", "import request_0513_modular"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "DISTANCE_TARGET_MOTION_CONTROL_ENABLE must be a boolean" in result.stderr


@pytest.mark.parametrize("mode", ["approach", "legacy"])
def test_direct_entrypoint_rejects_unsupported_distance_only_mode(mode):
    env = dict(os.environ, FOLLOW_DISTANCE_CONTROL_MODE=mode,
               DISTANCE_TARGET_MOTION_CONTROL_ENABLE="0")
    result = subprocess.run([sys.executable, "-c", "import request_0513_modular"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "false requires distance_pi" in result.stderr
