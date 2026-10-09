"""INI/env/runtime wiring for ROI scheduling, no hardware initialization."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from car_control_modular.config_loader import load_config_to_env


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "car_control_modular/config/reid_runtime.ini"
KEY = "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC"


def test_profile_sets_500_without_touching_depth_and_visual_clocks(monkeypatch):
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert float(env[KEY]) == .5
    assert float(env["ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC"]) == .3
    assert float(env["VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC"]) == .5


@pytest.mark.parametrize("limit", [.18, .25, .319, .5])
def test_explicit_shorter_roi_window_remains_a_rollback(monkeypatch, limit):
    env = {KEY: str(limit)}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert float(env[KEY]) == limit


@pytest.mark.parametrize("limit", [None, .18, .25, .319, .5])
def test_runtime_has_no_hidden_250_or_300_cap(limit):
    env = dict(os.environ)
    env.pop(KEY, None)
    if limit is not None:
        env[KEY] = str(limit)
    expected = .5 if limit is None else limit
    code = ("import request_0513_modular as r; "
            f"assert r.ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC == {expected}; "
            "assert r.ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC == .18; "
            "assert r.ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC == .3")
    result = subprocess.run([sys.executable, "-c", code, "--config", str(CONFIG)],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
