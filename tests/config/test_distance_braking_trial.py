"""Explicit aggressive trial, with one shared physical budget and no hardware."""
import ast
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from car_control_modular.config_loader import load_config_to_env
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.sample_braking import SampleBrakingAssessment

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "car_control_modular/config/reid_runtime.ini"


def test_trial_profile_reaches_controller_and_keeps_other_protections(monkeypatch):
    env = {}
    monkeypatch.setattr(os, "environ", env)
    load_config_to_env(str(CONFIG))
    assert env["DISTANCE_PI_BRAKING_STOP_DISTANCE_M"] == "1.10"
    assert env["DISTANCE_APPROACH_DECELERATION_M_S2"] == "1.00"
    assert env["DISTANCE_APPROACH_RESPONSE_DELAY_SEC"] == "0.15"
    assert env["DISTANCE_TARGET_MOTION_CONTROL_ENABLE"] == "0"
    assert env["ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC"] == "0.30"
    cfg = FollowPolicyConfig(
        distance_pid_enable=True, distance_control_mode="distance_pi",
        distance_target_motion_control_enable=False, target_distance_m=1.4,
        reverse_start_distance_m=1.2, brake_distance_m=.5,
        distance_pi_braking_stop_distance_m=float(env["DISTANCE_PI_BRAKING_STOP_DISTANCE_M"]),
        distance_approach_deceleration_m_s2=float(env["DISTANCE_APPROACH_DECELERATION_M_S2"]),
        distance_approach_response_delay_sec=float(env["DISTANCE_APPROACH_RESPONSE_DELAY_SEC"]))
    controller = FollowSafetyController(cfg)
    profile = controller._distance_pid._distance_pi.config
    assert profile.stationary_stop_preview_distance_m == 1.1
    assert profile.deceleration_m_s2 == 1.
    assert profile.response_delay_sec == .15
    assert controller.cfg.target_distance_m == 1.4
    assert controller.cfg.reverse_start_distance_m == 1.2


def test_runtime_passes_the_explicit_boundary_without_hardware():
    result = subprocess.run(
        [sys.executable, "-c", "import request_0513_modular as r; "
         "assert r.DISTANCE_PI_BRAKING_STOP_DISTANCE_M == 1.1; "
         "assert r.DISTANCE_APPROACH_DECELERATION_M_S2 == 1.; "
         "assert r.DISTANCE_APPROACH_RESPONSE_DELAY_SEC == .15; "
         "assert r.TARGET_DISTANCE == 1.4", "--config", str(CONFIG)],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    tree = ast.parse((ROOT / "request_0513_modular.py").read_text())
    bindings = [kw.value.id for node in ast.walk(tree) if isinstance(node, ast.Call)
                for kw in node.keywords if kw.arg == "distance_pi_braking_stop_distance_m"
                and isinstance(kw.value, ast.Name)]
    assert bindings == ["DISTANCE_PI_BRAKING_STOP_DISTANCE_M"]


@pytest.mark.parametrize("value", [0., .54, 1.41, True, "1.1", float("nan"), float("inf")])
def test_malformed_or_outside_stop_boundary_is_rejected(value):
    with pytest.raises(ValueError, match="braking stop distance"):
        FollowSafetyController(FollowPolicyConfig(
            target_distance_m=1.4, brake_distance_m=.5,
            distance_pi_braking_stop_distance_m=value))


def evidence(distance, *, trial):
    return SampleBrakingAssessment(
        1, 100., 100.04, distance, 60., 60., 100.04, 0.,
        1.1 if trial else 1.2, .816814, 1. if trial else .7,
        .15 if trial else .2, 200., outer_allowance_rpm=10.)


@pytest.mark.parametrize("distance,new_low,new_high", [(1.8, 54., 55.), (2., 66., 67.)])
def test_same_60rpm_input_no_longer_prematurely_zeroes_at_far_distance(distance, new_low, new_high):
    old = evidence(distance, trial=False).budget(100.04, 60.)
    new = evidence(distance, trial=True).budget(100.04, 60.)
    assert old.cap_rpm == 0.
    assert new_low < new.cap_rpm < new_high
    assert new.margin_m >= new.required_stop_m


def test_sixty_rpm_maintenance_boundary_moves_inward_without_changing_target():
    assert evidence(2.234, trial=False).budget(100.04, 60.).cap_rpm < 60.
    assert evidence(2.235, trial=False).budget(100.04, 60.).cap_rpm >= 60.
    assert evidence(1.892, trial=True).budget(100.04, 60.).cap_rpm < 60.
    assert evidence(1.893, trial=True).budget(100.04, 60.).cap_rpm >= 60.


@pytest.mark.parametrize("distance", [1.1, 1.4, 1.6])
def test_real_close_momentum_still_stops(distance):
    budget = evidence(distance, trial=True).budget(100.04, 60.)
    assert budget.cap_rpm == 0.
    assert budget.reason == "shared_braking_momentum"


def test_trial_does_not_extend_depth_or_encoder_deadlines():
    sample = evidence(3.5, trial=True)
    assert sample.budget(100.30001, 60., authorized_rpm=60.).cap_rpm == 0.
    with pytest.raises(ValueError):
        replace(sample, feedback_timestamp=sample.checked_at-.151)


def test_trial_margin_not_refunded_by_normal_lower_command():
    sample = evidence(3.5, trial=True)
    high = sample.budget(100.1, 60., authorized_rpm=80., execution_bound_rpm=90., completed_rpm=80.)
    low = sample.budget(100.1, 60., authorized_rpm=50., execution_bound_rpm=90., completed_rpm=80.)
    assert low.cap_rpm == 50.
    assert high.margin_m == low.margin_m
