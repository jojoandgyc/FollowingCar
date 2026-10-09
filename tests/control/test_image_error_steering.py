from dataclasses import replace
import pytest

from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


def config():
    return VisualSteeringPidConfig(enabled=True, image_error_only=True,
        camera_hfov_deg=60, deadband_deg=3, dynamic_small_error_deg=6,
        dynamic_large_error_deg=18, dynamic_small_max_correction_rpm=5,
        max_correction_rpm=10)


@pytest.mark.parametrize("x,expected", [(.5, 0), (.46, 0), (.54, 0),
    (.4, -5), (.6, 5), (.3, -7), (.7, 7), (.1, -10), (.9, 10)])
@pytest.mark.parametrize("yaw", [-100, -17.24, 0, 35, 100, None])
def test_wheel_difference_depends_only_on_image_offset(x, expected, yaw):
    fb = None if yaw is None else SteeringFeedback(timestamp=10, trustworthy=True,
        yaw_rate_right_dps=yaw, raw_yaw_rate_right_dps=4.7)
    pid = VisualSteeringPid(config())
    for t in [10, 10.05, 10.10]:
        r = pid.update(x, 82, fb, now=t, target_image_rate_dps=-90,
                       forward_tracking=True, visual_age_sec=.15)
        assert r.correction_rpm == expected
        assert r.base_rpm == 82 and not r.feedback_used
        assert not r.predictive_braking and not r.same_direction_overspeed_braking
        assert r.correction_limit_reason == "image_error_only"


@pytest.mark.parametrize("limit", [0, 3, 5, 10])
def test_explicit_policy_limit_and_zero_still_apply(limit):
    r = VisualSteeringPid(config()).update(.1, 0, None, now=10,
                                          max_correction_override_rpm=limit)
    assert r.correction_rpm == -limit


def test_controller_main_and_fast_paths_share_image_mode():
    cfg = FollowPolicyConfig(visible_steering_pid_enable=True,
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_camera_hfov_deg=60,
        visible_steering_pid_max_correction_rpm=10)
    c = FollowSafetyController(cfg)
    fb = SteeringFeedback(timestamp=10, trustworthy=True, yaw_rate_right_dps=-35)
    result = c._update_lateral_pid(c._visual_steering_pid, x_ratio=.25, base_rpm=58,
                                  feedback=fb, now=10)
    fast = c.refresh_visible_lateral_pid(x_ratio=.25, base_rpm=58, feedback=fb, now=10.05)
    assert result.correction_rpm == fast.correction_rpm < 0
    assert c._parked_recenter_pid.config.image_error_only


def test_image_zero_not_rebuilt_by_outer_zero_guard():
    r = VisualSteeringPid(config()).update(.54, 0, None, now=10)
    assert FollowSafetyController._pid_zero_guard_correction(r, current_x_ratio=.54,
        center_left_ratio=.48, center_right_ratio=.52, min_correction_rpm=5) == 0


def test_ini_and_real_constructor_bindings(monkeypatch):
    import ast
    import os
    from pathlib import Path
    from car_control_modular.config_loader import load_config_to_env
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(os, "environ", {})
    load_config_to_env(str(root / "car_control_modular/config/reid_runtime.ini"))
    assert os.environ["VISIBLE_STEERING_PID_IMAGE_ERROR_ONLY"] == "1"
    assert float(os.environ["FOLLOW_RESIDUAL_REVERSE_MAX_RPM"]) == 8
    tree = ast.parse((root / "request_0513_modular.py").read_text())
    bindings = {k.arg: k.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                for k in n.keywords if k.arg in {
                    "follow_residual_reverse_max_rpm", "visible_steering_pid_image_error_only"}}
    values = {name: eval(compile(ast.Expression(expr), "binding", "eval"),
              {"os": os, "VISIBLE_STEERING_PID_IMAGE_ERROR_ONLY": True})
              for name, expr in bindings.items()}
    assert values == {"follow_residual_reverse_max_rpm": 8,
                      "visible_steering_pid_image_error_only": True}


@pytest.mark.parametrize("yaw", [-35, 35, float("nan")])
def test_position_mode_continuation_does_not_use_yaw(yaw):
    from test_cap508_forward_steering import intent, fb
    from car_control_modular.lateral_intent import with_forward_continuation
    original = intent(image_error_only=True)
    extended = with_forward_continuation(original, fb(100, yaw))
    assert extended.valid_until == pytest.approx(100.22)
    assert extended.continuation_allowed(100.18, fb(100.18, yaw))
    assert not extended.valid(100.221)
    assert not extended.continuation_allowed(100.18, fb(99.8, yaw))
    near = intent(image_error_only=True, x_ratio=.54)
    assert with_forward_continuation(near, fb(100, yaw)) == near
