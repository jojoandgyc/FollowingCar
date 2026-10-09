"""CAP508-624 policy replay, no camera, serial port or motor threads."""
from dataclasses import replace
from types import SimpleNamespace
import os
from pathlib import Path

import pytest

from car_control_modular.control_types import SteeringFeedback, ControlAction, PersonTarget
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig
from car_control_modular.controllers import FollowSafetyController, FollowPolicyConfig
from car_control_modular.lateral_intent import LateralControlIntent, with_forward_continuation
from test_lateral_zero_runtime import owner, runtime, NOW
from car_control_modular.config_loader import load_config_to_env


def config():
    return VisualSteeringPidConfig(
        enabled=True, camera_hfov_deg=60, camera_latency_sec=.13, deadband_deg=3,
        outer_kp_per_sec=1.5, outer_kd_sec=0, rate_kp_rpm_per_dps=.32,
        max_yaw_rate_dps=35, max_correction_rpm=10,
        dynamic_small_error_deg=3, dynamic_large_error_deg=10,
        dynamic_small_max_correction_rpm=5, target_speed_match_max_closing_dps=8,
        opposite_yaw_brake_threshold_dps=3, braking_max_correction_rpm=5,
        fast_countersteer_max_correction_rpm=5,
        predictive_brake_decel_dps2=45, predictive_brake_margin_deg=1,
        same_direction_overspeed_threshold_dps=4, visual_direction_guard_enabled=True,
    )


def fb(t, yaw=0, **kw):
    return replace(SteeringFeedback(timestamp=t, trustworthy=True,
                   yaw_rate_right_dps=yaw, raw_yaw_rate_right_dps=yaw), **kw)


def run(x, yaw, enabled=True, **kwargs):
    pid = VisualSteeringPid(config())
    for t in (10., 10.1, 10.21):
        result = pid.update(x, 90, fb(t, yaw), now=t, target_image_rate_dps=0,
                            forward_tracking=enabled, **kwargs)
    return result


@pytest.mark.parametrize("side", [-1, 1])
def test_small_error_has_six_rpm_pair_difference(side):
    x = .5 + side * .063
    assert abs(run(x, 0, False).correction_rpm) == 1
    result = run(x, 0)
    assert result.correction_rpm == side * 3
    assert result.output_floor_reason == 'forward_tracking_floor'


@pytest.mark.parametrize("side", [-1, 1])
def test_far_error_increases_closing_not_max_rpm(side):
    result = run(.5 + side * .273, 0)
    assert result.target_speed_match_limit_dps == 20
    assert result.correction_rpm == side * 10
    assert abs(run(.5 + side * .273, 0, False).desired_yaw_rate_dps) == 12


@pytest.mark.parametrize("side", [-1, 1])
def test_same_side_countersteer_after_stable_direction(side):
    x = .5 + side * .273
    assert abs(run(x, -side * 9.4, False).correction_rpm) == 5
    result = run(x, -side * 9.4)
    assert result.correction_rpm == side * 10
    assert result.forward_phase == 'same_side_countersteer'


@pytest.mark.parametrize("side", [-1, 1])
def test_new_side_crossing_keeps_brake_cap(side):
    pid = VisualSteeringPid(config())
    pid.update(.5 - side * .273, 90, fb(10, -side * 9.4), now=10, forward_tracking=True)
    result = pid.update(.5 + side * .273, 90, fb(10.05, -side * 9.4), now=10.05, forward_tracking=True)
    assert result.correction_limit_rpm <= 5
    assert result.forward_phase != 'same_side_countersteer'


@pytest.mark.parametrize("side", [-1, 1])
def test_far_overspeed_tapers_near_center_still_stops(side):
    x = .5 + side * .295
    assert run(x, side * 22.78, False).correction_rpm == 0
    result = run(x, side * 22.78)
    assert 0 < side * result.correction_rpm <= 5
    assert result.output_floor_reason == 'forward_overspeed_taper'
    assert run(.5 + side * .05, side * 20).correction_rpm == 0


@pytest.mark.parametrize("cap", [0, 2, 5])
def test_explicit_override_always_wins(cap):
    assert abs(run(.773, -9.4, max_correction_override_rpm=cap).correction_rpm) <= cap


@pytest.mark.parametrize("changes", [
    {'timestamp': 9.8}, {'timestamp': 10.1}, {'trustworthy': False},
    {'raw_yaw_rate_right_dps': -30.}, {'left_forward_rpm': -10.},
])
def test_bad_feedback_does_not_enable_new_policy(changes):
    result = VisualSteeringPid(config()).update(.773, 90, fb(10, **changes), now=10,
                                               forward_tracking=True)
    assert not result.forward_tracking_active


def intent(**changes):
    return replace(LateralControlIntent(
        sequence=1, target_id=1, frame_index=1, published_at=100,
        valid_until=100.15, x_ratio=.77, motion_dx_ratio=0,
        target_image_rate_dps=0, mode='forward', base_percent=30,
        base_rpm=60, initial_correction_rpm=8, correction_limit_rpm=10,
        confidence=.95, bbox_quality='reliable', reason='visible',
        capture_frame_id=10, capture_timestamp=99.87), **changes)


def test_bridge_is_capture_bounded_and_never_renewed():
    original = intent()
    extended = with_forward_continuation(original, fb(100))
    assert extended.valid_until == pytest.approx(100.22)
    assert extended.nominal_valid_until == original.valid_until
    assert extended.continuation_allowed(100.18, fb(100.18))
    assert not extended.valid(100.221)
    assert with_forward_continuation(extended, fb(100)).valid_until == extended.valid_until
    older = with_forward_continuation(intent(capture_timestamp=99.81), fb(100))
    assert older.valid_until == pytest.approx(100.16)


@pytest.mark.parametrize("changes", [
    {'mode': 'yaw_only'}, {'mode': 'reverse'}, {'bbox_quality': 'limited'},
    {'hold_zero': True}, {'park_requested': True}, {'near_distance_mode': True},
    {'capture_timestamp': 0}, {'capture_timestamp': 99.70}, {'x_ratio': .55},
    {'target_image_rate_dps': float('nan')}, {'base_rpm': 0},
])
def test_bridge_excludes_other_modes_and_center(changes):
    original = intent(**changes)
    assert with_forward_continuation(original, fb(100)) == original


@pytest.mark.parametrize("feedback", [None, fb(100.0), fb(100.18, 25),
                                        fb(100.18, 1, raw_yaw_rate_right_dps=25)])
def test_bridge_revalidates_feedback_each_tick(feedback):
    extended = with_forward_continuation(intent(), fb(100))
    assert not extended.continuation_allowed(100.18, feedback)


def test_real_controller_camera_and_fast_tick_share_policy():
    controller = FollowSafetyController(FollowPolicyConfig(visible_steering_pid_forward_tracking_enable=True))
    controller._visual_steering_pid = VisualSteeringPid(config())
    controller._parked_recenter_pid = VisualSteeringPid(config())
    first = controller._update_lateral_pid(controller._visual_steering_pid, x_ratio=.773,
                base_rpm=90, feedback=fb(10), now=10, target_image_rate_dps=0)
    second = controller.refresh_visible_lateral_pid(x_ratio=.773, base_rpm=90,
                feedback=fb(10.05), now=10.05, target_image_rate_dps=0)
    parked = controller.refresh_parked_lateral_pid(x_ratio=.773, base_rpm=90,
                feedback=fb(10.1), now=10.1, target_image_rate_dps=0)
    reverse = controller.refresh_visible_lateral_pid(x_ratio=.773, base_rpm=90,
                feedback=fb(10.15), now=10.15, target_image_rate_dps=0, allow_forward_tracking=False)
    assert first.forward_tracking_active and second.forward_tracking_active
    assert not parked.forward_tracking_active and not reverse.forward_tracking_active


def test_real_publisher_does_not_extend_duplicate_frame(owner, monkeypatch):
    monkeypatch.setattr(runtime, 'VISIBLE_STEERING_PID_FORWARD_TRACKING_ENABLE', True)
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: fb(NOW))
    owner._last_control_decision_reason = 'visual_pid_right_encoder'
    owner._follow_controller.last_steering_pid_result.base_rpm = 60
    args = dict(width=640, target=PersonTarget((410, 0, 550, 450), 1, .95, 63000),
        runtime_actions=[ControlAction.steer_right(30, 100, 100, 'visible', correction_rpm=8)],
        control_source='vision', target_steerable=True, low_quality_visible=False)
    assert owner._publish_lateral_intent_from_decision(**args)
    first = owner._lateral_intent_store.snapshot()
    assert first.valid_until == pytest.approx(NOW + .22)
    assert not owner._publish_lateral_intent_from_decision(**args)
    assert owner._lateral_intent_store.snapshot() is first
    owner._lateral_intent_store.clear()
    assert not owner._publish_lateral_intent_from_decision(**args)
    owner._last_command_capture_frame += 1
    assert owner._publish_lateral_intent_from_decision(**args)


def test_fast_loop_rejects_invalid_bridge_without_refresh(owner):
    owner._lateral_intent_store.publish(with_forward_continuation(intent(), fb(100)))
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: fb(100))
    owner._service_lateral_intent(100.18)
    assert owner._lateral_intent_store.snapshot() is None


def test_motor_gate_checks_bridge_even_when_control_loop_has_not_run(owner, monkeypatch):
    owner._lateral_intent_store.publish(with_forward_continuation(intent(), fb(100)))
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: 100.18)
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: fb(100))
    assert not owner._has_fresh_lateral_yaw(1)
    owner._action_runtime.get_steering_feedback = lambda: fb(100.18)
    assert owner._has_fresh_lateral_yaw(1)


def test_runtime_config_opts_in_without_changing_other_limits(monkeypatch):
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    path = Path(__file__).resolve().parents[2] / 'car_control_modular/config/reid_runtime.ini'
    load_config_to_env(str(path))
    assert os.environ['VISIBLE_STEERING_PID_FORWARD_TRACKING_ENABLE'] == '1'
    # Current user-approved trial: 10 per wheel = 20 RPM total differential.
    assert float(os.environ['VISIBLE_STEERING_PID_MAX_CORRECTION_RPM']) == 10
    assert float(os.environ['LATERAL_INTENT_TTL_SEC']) == .15
    assert float(os.environ['ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC']) == .30


@pytest.mark.parametrize('mode', ['center', 'near', 'explicit_hold', 'stale_image'])
def test_real_controller_new_policy_preserves_stop_boundaries(mode):
    controller = FollowSafetyController(FollowPolicyConfig(visible_steering_pid_forward_tracking_enable=True))
    controller._visual_steering_pid = VisualSteeringPid(config())
    result = controller._update_lateral_pid(controller._visual_steering_pid,
        x_ratio=.5 if mode == 'center' else .773, base_rpm=90, feedback=fb(10), now=10,
        near_distance_mode=mode == 'near', hold_zero=mode == 'explicit_hold',
        visual_age_sec=.4 if mode == 'stale_image' else .1)
    if mode in ('center', 'explicit_hold'):
        assert result.correction_rpm == 0
    if mode != 'center':
        assert not result.forward_tracking_active
