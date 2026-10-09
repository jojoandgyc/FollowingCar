"""CAP82: main parked PID and fast refresh must retain certified braking."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import SteeringFeedback, ControlAction, PersonTarget
from car_control_modular.controllers import FollowSafetyController, FollowPolicyConfig
from car_control_modular.steering_pid import VisualSteeringPid
from car_control_modular.predictive_turn_brake import qualified_countersteer
from test_cap443_capture_braking import pid_config, sample
from test_lateral_zero_runtime import owner, NOW, _intent


def controller(sign=1, now=10):
    cfg = FollowPolicyConfig(visible_steering_pid_enable=True,
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_predictive_countersteer_max_correction_rpm=2.5,
        visible_steering_pid_max_correction_rpm=10, parked_recenter_max_rpm=7,
        near_distance_rotation_only_max_rpm=7)
    c = FollowSafetyController(cfg)
    c._parked_recenter_pid = VisualSteeringPid(replace(pid_config(),
        camera_hfov_deg=66, image_motion_response_sec=.25,
        predictive_brake_decel_dps2=60, predictive_brake_response_sec=.1,
        predictive_countersteer_max_correction_rpm=2.5,
        predictive_countersteer_min_yaw_rate_dps=4))
    f = SteeringFeedback(timestamp=now-.03, trustworthy=True,
        left_forward_rpm=sign*8, right_forward_rpm=-sign*5,
        yaw_rate_right_dps=sign*17.9, raw_yaw_rate_right_dps=sign*21.15)
    return c, f


@pytest.mark.parametrize("sign", [-1, 1])
def test_real_main_and_fast_parked_paths_preserve_countersteer(sign):
    c, f = controller(sign)
    x = .5 + sign*.2071
    t, frame = sample(82, 9.8875, x)
    frame = replace(frame, steering_feedback=f)
    action = c._pid_action_for_parked_target(t, frame, 10,
        "right" if sign > 0 else "left", target_image_rate_dps=-sign*28.01,
        near_distance_mode=True)
    assert action.kind == ("rotate_left" if sign > 0 else "rotate_right")
    assert c.last_steering_pid_result.correction_rpm == -2*sign
    result = c.refresh_parked_lateral_pid(x_ratio=x, base_rpm=0, feedback=f,
        now=10, target_image_rate_dps=-sign*28.01, visual_age_sec=.1125,
        near_distance_mode=True, max_correction_rpm=7)
    assert result.correction_rpm == -2*sign
    assert qualified_countersteer(result, x, 2.5)
    # A plain opposite correction STILL fails, even with the same numbers.
    assert c._pid_direction_guard_reason(x, 0, -2*sign) == "target_still_outside_center"
    for altered in [replace(result, predictive_braking=False),
                    replace(result, feedback_used=False),
                    replace(result, target_rate_valid=False),
                    replace(result, output_floor_reason="image_error_only"),
                    replace(result, correction_rpm=-7*sign),
                    replace(result, feedback_age_sec=.2)]:
        assert c._pid_direction_guard_reason(x, 0, altered.correction_rpm,
                    braking_result=altered) == "target_still_outside_center"


def braking_result():
    c, f = controller(now=NOW)
    c.active_target_id = 1
    result = c.refresh_parked_lateral_pid(x_ratio=.7071, base_rpm=0, feedback=f,
        now=NOW, target_image_rate_dps=-28.01, visual_age_sec=.1125,
        near_distance_mode=True, max_correction_rpm=7)
    return c, f, result


def test_visual_publisher_latches_brake_not_ordinary_opposite_turn(owner):
    c, f, result = braking_result()
    owner._follow_controller = c
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: f)
    x = .7071*640
    assert owner._publish_lateral_intent_from_decision(width=640,
        target=PersonTarget((x-60, 20, x+60, 460), 1, .95, 52800),
        runtime_actions=[ControlAction.rotate_left("predictive")],
        control_source="vision", target_steerable=True, low_quality_visible=False)
    req = owner._near_yaw_park_request
    assert req.countersteer_rpm == -2
    assert req.capture_frame_id == 576
    assert req.capture_timestamp == NOW-.09
    assert req.countersteer_until <= req.capture_timestamp+.25
    assert not owner._queued_calls  # no ordinary reverse command escape


def test_fast_loop_new_brake_enters_same_parking_owner(owner):
    c, f, result = braking_result()
    owner._follow_controller = c
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: f)
    i = _intent(owner, x_ratio=.7071, target_image_rate_dps=-28.01,
                initial_correction_rpm=7, correction_limit_rpm=7)
    owner._lateral_intent_last_sequence = i.sequence
    owner._service_lateral_intent(NOW)
    assert owner._near_yaw_park_request.countersteer_rpm == -2
    assert not owner._queued_calls


@pytest.mark.parametrize("change", ["forward", "weak", "expired", "disabled"])
def test_brake_publisher_does_not_expand_motion_scope(owner, change):
    c, f, result = braking_result()
    owner._follow_controller = c
    i = _intent(owner, x_ratio=.7071, target_image_rate_dps=-28.01)
    if change == "forward": owner._depth30_linear_snapshot = ("forward", 20, 1, NOW-.01)
    if change == "weak": i = replace(i, bbox_quality="limited")
    if change == "expired": i = replace(i, valid_until=NOW-.01)
    if change == "disabled": c.cfg = replace(c.cfg, visible_steering_pid_predictive_countersteer_max_correction_rpm=0)
    assert not owner._request_predictive_turn_brake(i, result)
    assert getattr(owner, "_near_yaw_park_request", None) is None


def test_pulse_duration_is_budgeted_before_braking_only_for_pivots():
    cfg = FollowPolicyConfig(visible_steering_pid_image_brake_assist=True,
        visible_steering_pid_predictive_countersteer_max_correction_rpm=2.5)
    c = FollowSafetyController(cfg)
    assert c._parked_recenter_pid.config.predictive_countersteer_response_sec == .08
    assert c._visual_steering_pid.config.predictive_countersteer_response_sec == 0
    off = FollowSafetyController(replace(cfg, visible_steering_pid_predictive_countersteer_max_correction_rpm=0))
    assert off._parked_recenter_pid.config.predictive_countersteer_response_sec == 0


def test_countersteer_evidence_cannot_release_existing_parking(owner):
    c, f, result = braking_result()
    owner._follow_controller = c
    owner._near_yaw_park_request = object()
    owner._release_near_yaw_park = lambda **kw: pytest.fail("braking released parking")
    x = .7071*640
    assert not owner._publish_lateral_intent_from_decision(width=640,
        target=PersonTarget((x-60, 20, x+60, 460), 1, .95, 52800),
        runtime_actions=[ControlAction.rotate_left("predictive")],
        control_source="vision", target_steerable=True, low_quality_visible=False)
