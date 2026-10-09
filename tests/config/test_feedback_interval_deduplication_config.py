"""Opt-in physical-interval coverage wiring; never instantiate motor hardware."""
import ast
import configparser
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from car_control_modular.config_loader import load_config_to_env
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.distance_pi import DistancePiConfig


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "car_control_modular/config/reid_runtime.ini"
KEY = "DISTANCE_PI_FEEDBACK_INTERVAL_DEDUPLICATION"


def test_profile_enables_coverage_deduplication_without_relaxing_physical_limits(monkeypatch):
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert env[KEY].lower() == "true"
    assert env["DISTANCE_PI_OBSERVED_FEEDBACK_RESERVE"].lower() == "true"
    assert env["DISTANCE_TARGET_MOTION_CONTROL_ENABLE"] == "0"
    assert float(env["DISTANCE_PI_BRAKING_STOP_DISTANCE_M"]) == 1.1
    assert float(env["DISTANCE_APPROACH_DECELERATION_M_S2"]) == 1.
    assert float(env["DISTANCE_APPROACH_RESPONSE_DELAY_SEC"]) == .15
    assert float(env["ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC"]) == .3


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "1", "true", "on", "yes"])
def test_environment_selection_survives_ini(monkeypatch, value):
    env = {KEY: value}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert env[KEY] == value


def runtime_import(config, value=None):
    env = dict(os.environ)
    env.pop(KEY, None)
    if value is not None:
        env[KEY] = value
    return subprocess.run(
        [sys.executable, "-c",
         "import request_0513_modular as r; "
         f"print('DEDUP_SETTING=' + str(r.{KEY}))",
         "--config", str(config)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("value,expected", [(None, True), ("", True), ("false", False), (" ON ", True)])
def test_runtime_profile_and_environment_override_are_strict_booleans(value, expected):
    # The shared loader treats an empty environment value as unspecified.
    result = runtime_import(CONFIG, value)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"DEDUP_SETTING={expected}" in result.stdout


def test_legacy_config_without_new_key_defaults_to_disabled(tmp_path, monkeypatch):
    parser = configparser.ConfigParser()
    parser.read(CONFIG)
    parser.remove_option("distance_pid", "pi_feedback_interval_deduplication")
    legacy = tmp_path / "legacy.ini"
    with legacy.open("w") as output:
        parser.write(output)
    env = {}
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", env)
        load_config_to_env(str(legacy))
    assert KEY not in env
    result = runtime_import(legacy)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DEDUP_SETTING=False" in result.stdout


@pytest.mark.parametrize("value", ["maybe", "2", "nan"])
def test_invalid_boolean_rejected_on_import_before_hardware(value):
    result = runtime_import(CONFIG, value)
    assert result.returncode != 0
    assert f"{KEY} must be a boolean" in result.stderr


def test_runtime_binds_config_field_once():
    tree = ast.parse((ROOT / "request_0513_modular.py").read_text())
    bindings = [kw.value.id for node in ast.walk(tree) if isinstance(node, ast.Call)
                for kw in node.keywords if kw.arg == "distance_pi_feedback_interval_deduplication"
                and isinstance(kw.value, ast.Name)]
    assert bindings == [KEY]


@pytest.mark.parametrize("enabled", [False, True])
def test_controller_preserves_explicit_policy_without_changing_braking_parameters(enabled):
    assert FollowPolicyConfig().distance_pi_feedback_interval_deduplication is False
    assert DistancePiConfig().feedback_interval_deduplication is False
    controller = FollowSafetyController(FollowPolicyConfig(
        distance_pid_enable=True, distance_control_mode="distance_pi",
        distance_target_motion_control_enable=False, target_distance_m=1.4,
        distance_pi_observed_feedback_reserve=True,
        distance_pi_feedback_interval_deduplication=enabled,
        distance_pi_braking_stop_distance_m=1.1,
        distance_approach_deceleration_m_s2=1., distance_approach_response_delay_sec=.15))
    profile = controller._distance_pid._distance_pi.config
    assert profile.feedback_interval_deduplication is enabled
    assert profile.observed_feedback_reserve is True
    assert profile.stationary_stop_preview_distance_m == 1.1
    assert profile.deceleration_m_s2 == 1.
    assert profile.response_delay_sec == .15
    assert controller._braking_interval_speed_bound_reader is None


@pytest.mark.parametrize("target_motion", [False, True])
def test_real_runtime_callback_forwards_physical_interval_only_in_distance_mode(target_motion):
    tree = ast.parse((ROOT / "request_0513_modular.py").read_text())
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Attribute)
                           and t.attr == "_braking_interval_speed_bound_reader" for t in node.targets)]
    assert len(assignments) == 1
    calls = []
    expected = object()

    def reader(uid, sample_timestamp, now):
        calls.append((uid, sample_timestamp, now))
        return expected

    owner = SimpleNamespace(
        _follow_controller=SimpleNamespace(),
        _action_runtime=SimpleNamespace(braking_interval_speed_bound_rpm=reader))
    exec(compile(ast.Module(body=assignments, type_ignores=[]), __file__, "exec"),
         {"self": owner, "DISTANCE_TARGET_MOTION_CONTROL_ENABLE": target_motion})
    callback = owner._follow_controller._braking_interval_speed_bound_reader
    if target_motion:
        assert callback is None and not calls
    else:
        assert callback(uid=1, sample_timestamp=100., now=100.12) is expected
        assert calls == [(1, 100., 100.12)]
