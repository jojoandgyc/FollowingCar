"""Opt-in board trial; old/rotation-only configurations retain old policy."""
import os
from pathlib import Path

from car_control_modular.config_loader import load_config_to_env


ROOT = Path(__file__).resolve().parents[2]


def test_board_profile_enables_relative_trial_without_changing_physical_ttl(monkeypatch):
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(ROOT / "car_control_modular/config/reid_runtime.ini"))
    assert env["ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE"] == "1"
    assert env["ASTRA_DEPTH_CONTINUATION_OVERSHOOT_M"] == "0.20"
    assert env["ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC"] == "0.30"
    assert env["TARGET_DISTANCE"] == "1.4"
    assert env["DISTANCE_PI_KP_PER_SEC"] == "3.0"
    assert env["DISTANCE_APPROACH_DECELERATION_M_S2"] == "1.00"
    assert env["DISTANCE_APPROACH_RESPONSE_DELAY_SEC"] == "0.15"


def test_rollback_profile_does_not_require_new_motion_evidence(tmp_path, monkeypatch):
    config = tmp_path / "rollback.ini"
    config.write_text("[astra_depth]\nrelative_continuation_enable=false\n"
                      "continuation_overshoot_m=0.0\n", encoding="utf-8")
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(config))
    assert env["ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE"] == "0"
    assert env["ASTRA_DEPTH_CONTINUATION_OVERSHOOT_M"] == "0.0"


def test_legacy_profile_does_not_silently_enable_relative_trial(tmp_path, monkeypatch):
    config = tmp_path / "legacy.ini"
    config.write_text("[astra_depth]\ncontinuation_speed_cap_enable=true\n", encoding="utf-8")
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(config))
    assert "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE" not in env
    assert "ASTRA_DEPTH_CONTINUATION_OVERSHOOT_M" not in env
