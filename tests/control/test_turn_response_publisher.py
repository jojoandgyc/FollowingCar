"""Real visual producer: do not authorize boost from a tapered PID result."""
import pytest
from test_lateral_zero_runtime import owner, _target
from car_control_modular.control_types import ControlAction
from types import SimpleNamespace
from test_lateral_zero_runtime import _intent, NOW, runtime


@pytest.mark.parametrize('phase,predictive,allowed', [
    ('image_error_only', False, True),
    ('image_brake_assist:no_qualified_yaw', False, True),
    ('image_visual_brake:taper', False, False),
    ('image_brake_assist:taper', False, False),
    ('image_error_only', True, False),
    ('legacy', False, False),
])
def test_actual_producer_preserves_error_and_explicit_permission(owner, phase, predictive, allowed):
    owner._last_control_decision_reason = 'visible_follow'
    result = owner._follow_controller.last_steering_pid_result
    result.visual_error_deg = 15.5
    result.forward_phase = phase
    result.predictive_braking = predictive
    result.correction_policy_limit_rpm = 15.
    result.correction_limit_reason = 'image_error_only' if phase != 'legacy' else 'legacy'
    result.output_floor_reason = 'image_error_only'
    result.position_demand_rpm = 5.
    result.brake_reduction_rpm = 1. if 'taper' in phase else 0.
    assert owner._publish_lateral_intent_from_decision(
        width=640, target=_target(), runtime_actions=[ControlAction.forward(21, 'follow')],
        control_source='vision', target_steerable=True, low_quality_visible=False)
    intent = owner._lateral_intent_store.snapshot()
    assert intent.visual_error_deg == 15.5
    assert intent.response_boost_allowed is allowed
    assert intent.mode == 'forward'


def test_real_fast_loop_replaces_build_permission_with_braking_veto(owner, monkeypatch):
    monkeypatch.setattr(runtime, 'MODULE_ASTRA_DEPTH_ENABLE', False)
    current = _intent(owner, mode='forward', base_rpm=42, base_percent=21,
                      near_distance_mode=False, response_boost_allowed=True)
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    result = owner._follow_controller.last_steering_pid_result
    result.forward_phase = 'image_error_only'
    result.predictive_braking = False
    result.correction_limit_reason = 'image_error_only'
    result.output_floor_reason = 'image_error_only'
    result.visual_error_deg = 15.
    result.position_demand_rpm = 5.
    result.brake_reduction_rpm = 0.
    owner._service_lateral_intent(NOW)
    assert owner._lateral_turn_response_policy == (current.sequence, True)
    result.forward_phase = 'image_visual_brake:taper'
    result.brake_reduction_rpm = 1.
    owner._follow_controller.refresh_visible_lateral_pid = lambda **kwargs: result
    owner._service_lateral_intent(NOW+.04)
    assert owner._lateral_turn_response_policy == (current.sequence, False)
