"""Verify the real launcher binding, not just a hand-written test config."""
import os
from pathlib import Path
import subprocess
import sys

from car_control_modular.config_loader import load_config_to_env

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "car_control_modular/config/reid_runtime.ini"


def test_actual_entrypoint_enables_scoped_forward_handoff():
    env = dict(os.environ)
    env.pop("FOLLOW_FORWARD_HANDOFF_ENABLE", None)
    result = subprocess.run([
        sys.executable, "-c", "import request_0513_modular as r; "
        "assert r.ACTION_RUNTIME_CONFIG.follow_forward_handoff_enable is True; "
        "assert r.ACTION_RUNTIME_CONFIG.follow_wheel_period_sec == .05",
        "--config", str(CONFIG),
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ini_switch_can_restore_guard_without_other_config_changes(tmp_path, monkeypatch):
    config = tmp_path / "off.ini"
    config.write_text("[lateral_intent]\nfollow_forward_handoff_enable=false\n", encoding="utf8")
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(config))
    assert env["FOLLOW_FORWARD_HANDOFF_ENABLE"] == "0"


def test_old_ini_does_not_opt_into_forward_handoff(tmp_path, monkeypatch):
    config = tmp_path / "old.ini"
    config.write_text("[lateral_intent]\nfollow_wheel_period_sec=.05\n", encoding="utf8")
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(config))
    assert "FOLLOW_FORWARD_HANDOFF_ENABLE" not in env
