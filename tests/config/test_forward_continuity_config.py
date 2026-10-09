"""Runtime profile and rollback wiring; importing the module starts no hardware."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from car_control_modular.config_loader import load_config_to_env


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "car_control_modular/config/reid_runtime.ini"


def test_continuity_profile_is_explicit_and_keeps_physical_deadlines(monkeypatch):
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert float(env["VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC"]) == .5
    assert env["DISTANCE_PI_OBSERVED_FEEDBACK_RESERVE"].lower() == "true"
    assert float(env["ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC"]) == .3
    assert float(env["DISTANCE_PI_BRAKING_STOP_DISTANCE_M"]) == 1.1
    assert float(env["DISTANCE_APPROACH_DECELERATION_M_S2"]) == 1.
    assert float(env["DISTANCE_APPROACH_RESPONSE_DELAY_SEC"]) == .15


@pytest.mark.parametrize("visibility,reserve", [(.5, "true"), (.25, "false")])
def test_profile_and_environment_rollback_reach_runtime(visibility, reserve):
    env = dict(os.environ, VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC=str(visibility),
               DISTANCE_PI_OBSERVED_FEEDBACK_RESERVE=reserve)
    result = subprocess.run(
        [sys.executable, "-c", "import request_0513_modular as r; "
         f"assert r.VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC == {visibility}; "
         f"assert r.DISTANCE_PI_OBSERVED_FEEDBACK_RESERVE is {reserve == 'true'}",
         "--config", str(CONFIG)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("key,value,message", [
    ("VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC", ".501", "0.05..0.50"),
    ("VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC", "nan", "0.05..0.50"),
    ("DISTANCE_PI_OBSERVED_FEEDBACK_RESERVE", "maybe", "must be a boolean"),
])
def test_invalid_settings_rejected_before_hardware(key, value, message):
    env = dict(os.environ)
    env[key] = value
    result = subprocess.run(
        [sys.executable, "-c", "import request_0513_modular",
         "--config", str(CONFIG)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert message in result.stderr


def test_runtime_binds_shared_reserve_policy_once():
    tree = ast.parse((ROOT / "request_0513_modular.py").read_text())
    bindings = [kw.value.id for node in ast.walk(tree) if isinstance(node, ast.Call)
                for kw in node.keywords if kw.arg == "distance_pi_observed_feedback_reserve"
                and isinstance(kw.value, ast.Name)]
    assert bindings == ["DISTANCE_PI_OBSERVED_FEEDBACK_RESERVE"]
