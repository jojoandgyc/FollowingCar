"""Separate inward braking memory from permission to restart/accelerate."""
from dataclasses import replace

import pytest

from car_control_modular.visual_steering_evidence import CaptureSteeringEvidence
from car_control_modular.outward_trajectory import make_outward_lead
from test_cap443_capture_braking import sample
from test_cap82_countersteer_provenance import controller
from car_control_modular.predictive_turn_brake import qualified_countersteer


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("weak_first", [True, False])
def test_inward_history_survives_quality_change_but_never_grants_lead(sign, weak_first):
    evidence = CaptureSteeringEvidence()
    for i, offset in enumerate([.28, .25, .21]):
        t, f = sample(210+i, 10+i*.1, .5+sign*offset)
        if (i < 2) == weak_first:
            t = replace(t, depth_observation=None,
                        braking_observation=replace(t.depth_observation, source="yolo_braking_only"))
        obs = evidence.observe(t, f, f.capture_timestamp+.09, 66)
    assert obs.rate_dps == pytest.approx(-sign*26.4)
    assert "braking_continuity" in obs.reason
    assert obs.control_x == obs.tracker_x
    assert not obs.outward_consistent
    assert make_outward_lead(obs, 10.29, hfov=66, deadband=3, release_margin=1.8, enabled=True) is None
    assert evidence.observe(t, f, 10.30, 66) is obs
    assert len(evidence.brake_samples) == 3


@pytest.mark.parametrize("change", ["uid", "raw", "gap", "bad", "outward"])
def test_braking_bridge_cannot_cross_invalid_or_outward_evidence(change):
    evidence = CaptureSteeringEvidence()
    for i, x in enumerate([.78, .75, .71]):
        t, f = sample(i+1, 10+i*.1, x)
        if i < 2:
            t = replace(t, depth_observation=None,
                        braking_observation=replace(t.depth_observation, source="yolo_braking_only"))
        else:
            if change == "uid": t, f = sample(3, 10.2, x, uid=2)
            if change == "raw": t, f = sample(3, 10.2, x, raw=99)
            if change == "gap": t, f = sample(3, 10.6, x)
            if change == "bad": t = replace(t, confidence=.2)
            if change == "outward": t, f = sample(3, 10.2, .79)
        obs = evidence.observe(t, f, f.capture_timestamp+.09, 66)
    assert obs.rate_dps is None


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("yaw,expected", [(8, 3), (17.9, 4), (28, 6)])
def test_countersteer_trial_3_to_6_rpm_reaches_direction_guard(sign, yaw, expected):
    c, f = controller(sign)
    c.cfg = replace(c.cfg, visible_steering_pid_predictive_countersteer_max_correction_rpm=6)
    c._parked_recenter_pid.config = replace(c._parked_recenter_pid.config,
        predictive_countersteer_max_correction_rpm=6,
        predictive_countersteer_min_correction_rpm=3,
        predictive_countersteer_gain_rpm_per_dps=.25)
    f = replace(f, yaw_rate_right_dps=sign*yaw, raw_yaw_rate_right_dps=sign*yaw)
    result = c.refresh_parked_lateral_pid(x_ratio=.5+sign*.2071, base_rpm=0,
        feedback=f, now=10, target_image_rate_dps=-sign*28.01, visual_age_sec=.1125,
        near_distance_mode=True, max_correction_rpm=7)
    assert result.correction_rpm == -sign*expected
    assert qualified_countersteer(result, .5+sign*.2071, 6)


@pytest.mark.parametrize("cap,policy,expected", [(0, 7, 0), (6, 2, 2), (100, 7, 6)])
def test_countersteer_disabled_lower_policy_and_hard_cap(cap, policy, expected):
    c, f = controller()
    c._parked_recenter_pid.config = replace(c._parked_recenter_pid.config,
        predictive_countersteer_max_correction_rpm=cap,
        predictive_countersteer_min_correction_rpm=100,
        predictive_countersteer_gain_rpm_per_dps=100)
    result = c._parked_recenter_pid.update(.7071, 0, f, now=10,
        target_image_rate_dps=-28.01, visual_age_sec=.1125,
        max_correction_override_rpm=policy)
    assert abs(result.correction_rpm) == expected


def test_countersteer_trial_config_reaches_both_controller_profiles(monkeypatch):
    import ast
    import os
    from pathlib import Path
    from car_control_modular.config_loader import load_config_to_env
    from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(os, "environ", {})
    load_config_to_env(str(root / "car_control_modular/config/reid_runtime.ini"))
    tree = ast.parse((root / "request_0513_modular.py").read_text())
    fields = {"max_correction_rpm": 6, "min_correction_rpm": 3, "gain_rpm_per_dps": .25}
    cfg = {}
    for field, expected in fields.items():
        name = "VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_" + field.upper()
        assert float(os.environ[name]) == expected
        assign = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        value = eval(compile(ast.Expression(assign.value), "config", "eval"), {"os": os})
        key = "visible_steering_pid_predictive_countersteer_" + field
        expr = next(k.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                    for k in n.keywords if k.arg == key)
        cfg[key] = eval(compile(ast.Expression(expr), "binding", "eval"), {name: value})
    controller = FollowSafetyController(FollowPolicyConfig(**cfg))
    for profile in [controller._parked_recenter_pid, controller._visual_steering_pid]:
        for field, expected in fields.items():
            assert getattr(profile.config, "predictive_countersteer_"+field) == expected


def test_cap214_recovery_uses_braking_history_in_main_and_fast_pid():
    c, f = controller(-1)
    c.cfg = replace(c.cfg, visible_steering_pid_image_brake_assist=True,
        visible_steering_pid_image_capture_motion=True, visible_steering_pid_camera_hfov_deg=66)
    for cap, stamp, x, tracker in [(209, 10., .1844, .2035),
                                  (210, 10.064, .2024, .212),
                                  (212, 10.164, .2358, .2274),
                                  (214, 10.264, .2892, .2637)]:
        t, frame = sample(cap, stamp, x, tracker_x=tracker)
        if cap != 214:
            t = replace(t, depth_observation=None,
                        braking_observation=replace(t.depth_observation, source="yolo_braking_only"))
        obs = c._capture_steering_observation(t, frame, stamp+.1)
    f = replace(f, timestamp=10.334, yaw_rate_right_dps=-14.11, raw_yaw_rate_right_dps=-14.11)
    frame = replace(frame, steering_feedback=f)
    c._pid_action_for_parked_target(t, frame, 10.364, "left", near_distance_mode=True)
    main = c.last_steering_pid_result
    assert main.target_image_rate_dps == pytest.approx(35.244)
    assert main.predictive_braking or abs(main.correction_rpm) < 7
    fast = c.refresh_parked_lateral_pid(x_ratio=obs.control_x, base_rpm=0, feedback=f,
        now=10.364, target_image_rate_dps=obs.rate_dps, visual_age_sec=.1,
        near_distance_mode=True, max_correction_rpm=7)
    assert fast.correction_rpm == main.correction_rpm
