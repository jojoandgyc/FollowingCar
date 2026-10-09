"""CAP660 slow yaw taper, HFOV binding, and center/fast-crossing vetoes."""
import ast
import os
from dataclasses import replace
from pathlib import Path

import pytest

from car_control_modular.config_loader import load_config_to_env
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig


def config():
    return VisualSteeringPidConfig(enabled=True, image_error_only=True,
        image_brake_assist=True, image_capture_motion=True,
        image_slow_brake_continuity_sec=.5, camera_hfov_deg=66,
        deadband_deg=3, dynamic_large_error_deg=10, max_correction_rpm=15,
        predictive_brake_decel_dps2=60, predictive_brake_margin_deg=1.25,
        predictive_brake_response_sec=.05)


def feedback(yaw=-7.84):
    return SteeringFeedback(timestamp=10, trustworthy=True,
        yaw_rate_right_dps=yaw, raw_yaw_rate_right_dps=yaw)


@pytest.mark.parametrize('sign', [-1, 1])
def test_slow_yaw_does_not_clear_correction_before_visual_center(sign):
    x = .5 + sign * .0853
    old = VisualSteeringPid(replace(config(), image_slow_brake_continuity_sec=0))
    new = VisualSteeringPid(config())
    args = dict(now=10, visual_age_sec=.159, target_image_rate_dps=-sign*2.14)
    assert old.update(x, 20, feedback(sign*7.84), **args).correction_rpm == 0
    result = new.update(x, 20, feedback(sign*7.84), **args)
    assert result.correction_rpm == sign*2
    assert not result.predictive_braking
    assert result.forward_phase == 'image_brake_assist:slow_visual_continuity'
    assert result.base_rpm == 20


@pytest.mark.parametrize('x,rate,yaw', [(.5, 0, -7.84), (.4317, 5.74, -4.7),
                                     (.415, 30, -7.84), (.415, 2, -30)])
def test_center_or_fast_approach_still_stops(x, rate, yaw):
    result = VisualSteeringPid(config()).update(x, 20, feedback(yaw), now=10,
        visual_age_sec=.159, target_image_rate_dps=rate)
    assert result.correction_rpm == 0


@pytest.mark.parametrize('age,rate', [(.3, 2.14), (.159, None), (.159, float('nan'))])
def test_old_or_invalid_visual_rate_cannot_suppress_brake(age, rate):
    result = VisualSteeringPid(config()).update(.4147, 20, feedback(), now=10,
        visual_age_sec=age, target_image_rate_dps=rate)
    assert result.correction_rpm == 0


def test_no_minimum_on_time_and_override_still_zeroes():
    p = VisualSteeringPid(config())
    assert p.update(.4147, 20, feedback(), now=10, visual_age_sec=.159,
                    target_image_rate_dps=2.14).correction_rpm == -2
    assert p.update(.5, 20, feedback(), now=10.01, visual_age_sec=.01,
                    target_image_rate_dps=20).correction_rpm == 0
    assert p.update(.2, 20, None, now=10.02, max_correction_override_rpm=0).correction_rpm == 0


def test_runtime_config_binding_without_importing_hardware(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(os, 'environ', {})
    load_config_to_env(str(root/'car_control_modular/config/reid_runtime.ini'))
    tree = ast.parse((root/'request_0513_modular.py').read_text())
    names = {'VISIBLE_STEERING_PID_CAMERA_HFOV_DEG',
             'VISIBLE_STEERING_PID_IMAGE_SLOW_BRAKE_CONTINUITY_SEC'}
    env = {'os': os}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'config_binding', 'exec'), env)
    args = {'visible_steering_pid_camera_hfov_deg',
            'visible_steering_pid_image_slow_brake_continuity_sec', 'follow_turn_residual_max_rpm'}
    bindings = {kw.arg: eval(compile(ast.Expression(kw.value), 'binding', 'eval'), env)
                for node in ast.walk(tree) if isinstance(node, ast.Call)
                for kw in node.keywords if kw.arg in args}
    assert bindings.pop('follow_turn_residual_max_rpm') == 4
    assert bindings['visible_steering_pid_camera_hfov_deg'] == 66
    assert bindings['visible_steering_pid_image_slow_brake_continuity_sec'] == .5
    controller = FollowSafetyController(FollowPolicyConfig(**bindings,
        visible_steering_pid_enable=True, visible_steering_pid_image_error_only=True))
    assert controller._visual_steering_pid.config.image_slow_brake_continuity_sec == .5
    for x, expected in [(0, -33), (1, 33)]:
        r = controller._visual_steering_pid.update(x, 0, None, now=10)
        assert r.visual_error_deg == expected
