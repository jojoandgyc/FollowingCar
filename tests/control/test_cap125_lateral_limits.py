"""Separate transient brake authority from the persistent fast-loop policy."""
from types import SimpleNamespace
from dataclasses import replace

import pytest

from car_control_modular.control_types import ControlAction, PersonTarget, SteeringFeedback
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig
from test_lateral_zero_runtime import owner, runtime, NOW


def make_pid():
    return VisualSteeringPid(VisualSteeringPidConfig(
        enabled=True, camera_hfov_deg=60, camera_latency_sec=0,
        max_correction_rpm=10, braking_max_correction_rpm=5,
        fast_countersteer_max_correction_rpm=5,
        opposite_yaw_brake_threshold_dps=2,
        target_speed_match_max_closing_dps=8, outer_kp_per_sec=1.5,
        edge_boost_start_error_deg=16,
    ))


def update(pid, t, yaw, x=.16, override=None):
    return pid.update(x, 60, SteeringFeedback(timestamp=t, trustworthy=True,
                      yaw_rate_right_dps=yaw), now=t, target_image_rate_dps=0,
                      max_correction_override_rpm=override)


@pytest.mark.parametrize("x,yaw", [(.16, 10), (.84, -10)])
def test_temporary_braking_does_not_pin_next_refresh_to_five(x, yaw):
    pid = make_pid()
    first = update(pid, 10., yaw, x)
    assert first.correction_limit_rpm == 5
    assert first.correction_policy_limit_rpm == 10
    next_result = update(pid, 10.05, 0, x, first.correction_policy_limit_rpm)
    assert next_result.correction_limit_rpm == 10
    assert 5 < abs(next_result.correction_rpm) <= 10
    # A REAL continuing wrong-way motion must retain the brake limit.
    again = update(pid, 10.1, yaw, x, first.correction_policy_limit_rpm)
    assert again.correction_limit_rpm == 5


@pytest.mark.parametrize("ceiling", [0, 3, 6])
def test_explicit_caller_policy_survives_refresh(ceiling):
    pid = make_pid()
    first = update(pid, 10., 10, override=ceiling)
    assert first.correction_policy_limit_rpm == ceiling
    second = update(pid, 10.05, 0, override=first.correction_policy_limit_rpm)
    assert abs(second.correction_rpm) <= ceiling
    assert second.correction_limit_rpm <= ceiling


def test_near_center_and_predictive_braking_remain_active():
    pid = make_pid()
    assert update(pid, 10., 0, .5).correction_rpm == 0
    near = update(pid, 10.05, 0, .46)
    assert near.correction_limit_rpm < 10
    pid = VisualSteeringPid(replace(pid.config, predictive_brake_decel_dps2=45))
    braking = update(pid, 10.10, -20, .44)
    assert braking.predictive_braking
    assert braking.correction_rpm == 0


def publish(owner, result, action=None, low=False):
    owner._follow_controller.last_steering_pid_result = result
    owner._last_control_decision_reason = "visual_pid_left_encoder"
    return owner._publish_lateral_intent_from_decision(
        width=640, target=PersonTarget((20., 20., 185., 460.), 1, .95, 72600.),
        runtime_actions=[action or ControlAction.steer_left(30, 100, 100, "test", correction_rpm=5)],
        control_source="vision", target_steerable=not low, low_quality_visible=low,
    )


def test_real_publisher_and_fast_tick_use_policy_not_transient_cap(owner, monkeypatch):
    pid = make_pid()
    first = update(pid, NOW-.01, 10)
    assert publish(owner, first)
    intent = owner._lateral_intent_store.snapshot()
    assert intent.correction_limit_rpm == 10 and intent.initial_correction_rpm == -5
    owner._lateral_intent_last_sequence = intent.sequence
    owner._lateral_intent_last_correction_rpm = -5
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: SteeringFeedback(
        timestamp=NOW, trustworthy=True))
    calls = []
    def refresh(**kw):
        calls.append(kw)
        return update(pid, NOW, 0, override=kw['max_correction_rpm'])
    owner._follow_controller.refresh_visible_lateral_pid = refresh
    # This fixture tests yaw only, not independent depth-authority policy.
    monkeypatch.setattr(owner, '_depth_longitudinal_authority_enabled', lambda: False)
    owner._service_lateral_intent(NOW)
    assert calls[0]['max_correction_rpm'] == 10
    assert abs(owner._lateral_intent_last_correction_rpm) > 5
    # Refresh never extends the original intent deadline.
    assert owner._lateral_intent_store.snapshot().valid_until == intent.valid_until


@pytest.mark.parametrize("kind", ["yaw_only", "reverse", "limited"])
def test_non_forward_paths_keep_their_original_ceiling(owner, kind):
    first = update(make_pid(), NOW-.01, 10)
    action = (ControlAction.backward(20, 'reverse') if kind == 'reverse'
              else ControlAction.rotate_left('test'))
    assert publish(owner, first, action, low=kind == 'limited')
    assert owner._lateral_intent_store.snapshot().correction_limit_rpm == 5
