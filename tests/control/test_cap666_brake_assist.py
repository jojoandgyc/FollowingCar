"""Position-led steering and actual-post-stop evidence; no hardware access."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import SteeringFeedback, SensorFrame, PersonTarget
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig
from car_control_modular.near_yaw_parking import NearYawParkRequest, ParkSettlingEvidence
from car_control_modular.controllers import FollowSafetyController, FollowPolicyConfig


def config():
    return VisualSteeringPidConfig(enabled=True, image_error_only=True, image_brake_assist=True,
        camera_hfov_deg=60, deadband_deg=3, dynamic_small_error_deg=3,
        dynamic_large_error_deg=10, dynamic_small_max_correction_rpm=5,
        max_correction_rpm=10, predictive_brake_decel_dps2=60,
        predictive_brake_margin_deg=1.25, predictive_brake_response_sec=.05)


def fb(yaw=0, stamp=10., **kw):
    return replace(SteeringFeedback(timestamp=stamp, trustworthy=True,
        yaw_rate_right_dps=yaw, raw_yaw_rate_right_dps=yaw), **kw)


def output(x, feedback=None):
    return VisualSteeringPid(config()).update(x, 58, feedback, now=10., visual_age_sec=.139)


def test_cap1154_approaching_center_brakes_but_wrong_way_motion_does_not():
    assert output(.658, fb(31)).correction_rpm == 0
    assert output(.658, fb(31)).predictive_braking
    assert output(.658, fb(0)).correction_rpm == 9
    assert output(.658, fb(-35)).correction_rpm == 9


@pytest.mark.parametrize("x", [.08, .2, .34, .4, .44, .5, .56, .6, .66, .8, .92])
@pytest.mark.parametrize("yaw", [-40, -15, 0, 15, 40])
def test_yaw_can_only_reduce_same_side_demand_and_never_changes_base(x, yaw):
    baseline, result = output(x), output(x, fb(yaw))
    assert abs(result.correction_rpm) <= abs(baseline.correction_rpm)
    assert result.correction_rpm * (x-.5) >= 0
    assert result.base_rpm == result.requested_base_rpm == 58
    if yaw*(x-.5) <= 0:
        assert result.correction_rpm == baseline.correction_rpm
        assert not result.feedback_used


@pytest.mark.parametrize("feedback", [None, fb(35, stamp=9.8), fb(35, stamp=10.01),
    fb(35, trustworthy=False), fb(35, raw_yaw_rate_right_dps=None),
    fb(35, raw_yaw_rate_right_dps=4.7), fb(35, raw_yaw_rate_right_dps=-35),
    fb(35, left_forward_rpm=float("nan")), fb(35, raw_yaw_rate_right_dps=float("nan"))])
def test_bad_or_time_mismatched_yaw_cannot_cancel_correction(feedback):
    result = output(.658, feedback)
    assert result.correction_rpm == 9
    assert not result.feedback_used


def test_cap332_lagging_filtered_yaw_does_not_erase_left_correction():
    result = output(.253, fb(-17.24, raw_yaw_rate_right_dps=-4.7))
    assert result.correction_rpm == -10
    assert not result.feedback_used


@pytest.mark.parametrize("sign", [-1, 1])
def test_small_error_taper_and_center_release_hysteresis(sign):
    pid = VisualSteeringPid(config())
    near = pid.update(.5 + sign*.06, 0, None, now=10)
    assert 0 < abs(near.correction_rpm) < 5
    assert near.output_floor_rpm == 0
    assert pid.update(.5, 0, None, now=10.05).correction_rpm == 0
    held = pid.update(.5 + sign*.06, 0, None, now=10.1)
    assert held.correction_rpm == 0 and held.output_floor_reason == "center_hold"
    assert pid.update(.5 + sign*.1, 0, None, now=10.15).correction_rpm * sign > 0


def test_main_and_fast_controller_paths_share_taper():
    c = FollowSafetyController(FollowPolicyConfig(visible_steering_pid_enable=True,
        visible_steering_pid_image_error_only=True, visible_steering_pid_image_brake_assist=True,
        visible_steering_pid_camera_hfov_deg=60, visible_steering_pid_deadband_deg=3,
        visible_steering_pid_predictive_brake_decel_dps2=60,
        visible_steering_pid_predictive_brake_margin_deg=1.25,
        visible_steering_pid_predictive_brake_response_sec=.05,
        visible_steering_pid_dynamic_small_error_deg=3,
        visible_steering_pid_dynamic_large_error_deg=10,
        visible_steering_pid_max_correction_rpm=10))
    main = c._update_lateral_pid(c._visual_steering_pid, x_ratio=.658, base_rpm=58,
        feedback=fb(31), now=10, visual_age_sec=.139)
    fast = c.refresh_visible_lateral_pid(x_ratio=.658, base_rpm=58,
        feedback=fb(31), now=10.01, visual_age_sec=.149)
    assert main.correction_rpm == fast.correction_rpm == 0
    assert main.base_rpm == fast.base_rpm == 58
    assert c._parked_recenter_pid.config.image_brake_assist


def test_intentional_zero_does_not_fall_through_to_legacy_turn(monkeypatch):
    c = FollowSafetyController(FollowPolicyConfig(visible_steering_pid_enable=True, forward_max_rpm=200,
        center_left_ratio=.45, center_right_ratio=.55,
        visible_steering_pid_image_error_only=True, visible_steering_pid_image_brake_assist=True))
    monkeypatch.setattr(c, "_visible_base_forward_percent", lambda *a, **kw: 29)
    monkeypatch.setattr(c, "_update_lateral_pid", lambda *a, **kw: output(.658, fb(31)))
    target = PersonTarget(track_id=1, bbox=(370, 10, 470, 450), confidence=.9, area=44000)
    action = c._pid_action_for_visible_target(target, SensorFrame(width=640, height=480,
        distance_m=2.0, capture_timestamp=9.9, steering_feedback=fb(31)), 10.)
    assert action.kind == "forward" and action.speed_percent == 29
    assert action.steer_correction_rpm == 0
    assert action.reason == "visual_pid_image_yaw_zero"


@pytest.mark.parametrize("yaw,expected_park", [(31, True), (-31, False), (0, False)])
def test_near_target_actual_controller_uses_prediction_not_opposite_kick(monkeypatch, yaw, expected_park):
    monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: 10.)
    c = FollowSafetyController(FollowPolicyConfig(visible_steering_pid_enable=True,
        visible_steering_pid_image_error_only=True, visible_steering_pid_image_brake_assist=True,
        center_left_ratio=.45, center_right_ratio=.55,
        visible_steering_pid_camera_hfov_deg=60, visible_steering_pid_deadband_deg=3,
        visible_steering_pid_predictive_brake_decel_dps2=60,
        visible_steering_pid_predictive_brake_margin_deg=1.25,
        visible_steering_pid_predictive_brake_response_sec=.05,
        visible_steering_pid_dynamic_large_error_deg=10))
    monkeypatch.setattr(c, "_near_settle_hold_active", lambda *a, **kw: False)
    target = PersonTarget((381.12, 60, 461.12, 440), 1, .95, 30400)
    frame = SensorFrame(width=640, height=480, persons=[target], distance_m=1.28,
        capture_frame_id=1154, capture_timestamp=9.861, steering_feedback=fb(yaw))
    result = c._near_distance_rotation_only_decision(1154, frame, target, 1.28, 1.53,
        target_steerable=True)
    assert result.near_yaw_park_requested is expected_park
    assert result.actions[0].kind == ("stop" if expected_park else "rotate_right")
    assert result.current_forward_percent == 0


def quiet(t, rpm=0):
    return SimpleNamespace(timestamp=t, trustworthy=True,
        left_forward_rpm=rpm, right_forward_rpm=rpm)


def evidence():
    return ParkSettlingEvidence(NearYawParkRequest(1, 1128, 9.9, 9.98, "center_hold"), 10.)


def test_pre_stop_image_cannot_unlock_even_after_quiet():
    e = evidence()
    assert not e.observe(quiet(10.42), 10.42)
    assert e.observe(quiet(10.48), 10.48)
    assert not e.release_ready(9.97, quiet(10.48), 10.50)
    assert e.reason == "image_before_stop_write"
    assert not e.release_ready(10.43, quiet(10.48), 10.50)
    assert e.reason == "image_before_quiet_confirmation"
    assert e.release_ready(10.49, quiet(10.48), 10.50)


def test_repeated_feedback_is_not_two_samples_and_close_samples_need_quiet_span():
    e = evidence()
    for now in [10.02, 10.04, 10.06]:
        assert not e.observe(quiet(10.02), now)
    assert e.quiet_count == 1
    assert not e.observe(quiet(10.04), 10.06)
    assert e.observe(quiet(10.08), 10.09)


@pytest.mark.parametrize("feedback,now", [(None, 10.1), (quiet(9.99), 10.1),
    (quiet(10.01), 10.3), (quiet(10.2), 10.1), (quiet(float("nan")), 10.1),
    (quiet(10.09, 2), 10.1), (quiet(10.09, -2), 10.1), (quiet(10.09, float("nan")), 10.1)])
def test_invalid_or_moving_feedback_revokes_readiness(feedback, now):
    e = evidence()
    e.observe(quiet(10.02), 10.02)
    assert e.observe(quiet(10.08), 10.08)
    assert not e.observe(feedback, now)
    assert e.ready_at is None


def test_feedback_gap_restarts_quiet_evidence_not_motion_authority():
    e = evidence()
    e.observe(quiet(10.02), 10.02)
    assert not e.observe(quiet(10.3), 10.3)
    assert e.observe(quiet(10.36), 10.36)
    assert e.sent_at == 10.


def test_running_configuration_binds_both_new_settings(monkeypatch):
    import ast
    import os
    from pathlib import Path
    from car_control_modular.config_loader import load_config_to_env
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(os, "environ", {})
    load_config_to_env(str(root / "car_control_modular/config/reid_runtime.ini"))
    assert os.environ["VISIBLE_STEERING_PID_IMAGE_BRAKE_ASSIST"] == "1"
    assert float(os.environ["VISIBLE_STEERING_PID_IMAGE_CENTER_RELEASE_MARGIN_DEG"]) == 1.8
    tree = ast.parse((root / "request_0513_modular.py").read_text())
    bindings = {k.arg: k.value for n in ast.walk(tree) if isinstance(n, ast.Call)
        for k in n.keywords if k.arg in {"visible_steering_pid_image_brake_assist",
                                       "visible_steering_pid_image_center_release_margin_deg"}}
    values = {name: eval(compile(ast.Expression(expr), "binding", "eval"), {
        "VISIBLE_STEERING_PID_IMAGE_BRAKE_ASSIST": True,
        "VISIBLE_STEERING_PID_IMAGE_CENTER_RELEASE_MARGIN_DEG": 1.8})
        for name, expr in bindings.items()}
    assert values == {"visible_steering_pid_image_brake_assist": True,
                      "visible_steering_pid_image_center_release_margin_deg": 1.8}


def test_encoder_sample_clock_is_inside_read_write_exclusion():
    import ast
    import inspect
    import textwrap
    from car_control_modular.action_runtime import MotionActionRuntime
    tree = ast.parse(textwrap.dedent(inspect.getsource(MotionActionRuntime._steering_feedback_loop)))
    reads = [node for node in ast.walk(tree) if isinstance(node, ast.With)
             and "owner.motor_io_lock" in ast.unparse(node.items[0].context_expr)]
    assert len(reads) == 1
    statements = [ast.unparse(node) for node in reads[0].body]
    assert statements[-1] == "sample_ts = time.monotonic()"
    assert statements.index("left = driver.read_motor_status('left')") < statements.index(
        "right = driver.read_motor_status('right')")
    assert statements[0] == "left_read_started = time.monotonic()"
    assert "left_read_finished = time.monotonic()" in statements
    assert "right_read_started = time.monotonic()" in statements
