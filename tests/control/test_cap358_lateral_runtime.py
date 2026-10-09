"""Production lateral fast-loop must honor bounded deceleration after slew.

Uses a constructor-free main owner and real PID results; all I/O is a queue
spy or cached fake feedback. No device, motor thread or launcher is opened.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.steering_pid import VisualSteeringPid
from test_cap358_predictive_continuity import config, call, feedback
from test_lateral_zero_runtime import NOW, _intent, owner, runtime


def prepare(owner, monkeypatch, sign=-1, reason='fresh_position_deceleration', *, refresh=False):
    pid = VisualSteeringPid(config())
    if reason == 'recent_outward_deceleration':
        call(pid, now=NOW-.133, age=.15, error=sign*7.7022, rate=sign*32.54,
             fb=feedback(NOW-.133, 24, 23, -sign*1.6272, -sign*3.2544))
        f = feedback(NOW, 37+sign*4, 37-sign*4, sign*12.54, sign*12.54)
        result = call(pid, now=NOW, age=.283, error=sign*7.7022,
                      rate=sign*32.54, fb=f)
    else:
        f = feedback(NOW, 31+sign*11, 31-sign*11, sign*34.485, sign*28.215)
        result = call(pid, now=NOW, error=sign*21.9045, fb=f)
    assert result.brake_continuity_reason == reason
    assert 0 < abs(result.correction_rpm) <= 2
    owner._follow_controller.last_steering_pid_result = result
    owner._follow_controller.cfg = SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_max_correction_rpm=10.,
        near_distance_rotation_only_max_rpm=7.,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_predictive_countersteer_max_correction_rpm=0.)
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: f)
    current = _intent(owner, initial_correction_rpm=result.correction_rpm,
        x_ratio=.5+sign*.33, near_distance_mode=False, target_image_rate_dps=None)
    owner._lateral_intent_last_correction_rpm = sign*10
    owner._lateral_intent_last_tick_ts = NOW-.001
    owner._lateral_intent_last_log_direction = 'none'
    owner._lateral_intent_last_sequence = current.sequence if refresh else -1
    owner._follow_controller.refresh_parked_lateral_pid = lambda **kwargs: result
    monkeypatch.setattr(runtime, 'LATERAL_INTENT_BRAKE_RPM_PER_SEC', 90.)
    return current, result


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('reason', ['fresh_position_deceleration', 'recent_outward_deceleration'])
@pytest.mark.parametrize('refresh', [False, True])
def test_first_and_reused_fast_tick_cannot_slew_above_deceleration_cap(
        owner, monkeypatch, caplog, sign, reason, refresh):
    _, result = prepare(owner, monkeypatch, sign, reason, refresh=refresh)
    with caplog.at_level('INFO'):
        owner._service_lateral_intent(NOW)
    assert owner._lateral_intent_last_correction_rpm == result.correction_rpm
    assert owner._current_rotate_raw_target == abs(result.correction_rpm)
    assert owner._current_forward_percent == 0
    expected = runtime.ACTION_ROTATE_RIGHT if sign > 0 else runtime.ACTION_ROTATE_LEFT
    assert owner._queued_calls[-1][0] == (expected,)
    assert 'brake_continuity_reason='+reason in caplog.text


@pytest.mark.parametrize('sign', [-1, 1])
def test_continuity_cannot_carry_opposite_residual_through_short_slew(owner, monkeypatch, caplog, sign):
    current, _ = prepare(owner, monkeypatch, sign)
    owner._lateral_intent_last_correction_rpm = -sign*10
    with caplog.at_level('INFO'):
        owner._service_lateral_intent(NOW)
    assert owner._lateral_intent_last_correction_rpm == 0
    assert owner._lateral_intent_zero_sequence == current.sequence
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)
    assert 'brake_continuity_reason=fresh_position_deceleration' in caplog.text


@pytest.mark.parametrize('stop', ['expired', 'explicit', 'shutdown'])
def test_continuity_never_overrides_expiry_or_safety_stop(owner, monkeypatch, stop):
    current, _ = prepare(owner, monkeypatch)
    if stop == 'expired':
        owner._lateral_intent_store.publish(replace(current, valid_until=NOW-.001))
    elif stop == 'explicit':
        owner._explicit_stop_requested = True
    else:
        owner._runtime_shutdown_requested = True
    owner._follow_controller.refresh_parked_lateral_pid = lambda **kwargs: pytest.fail('unsafe refresh')
    owner._service_lateral_intent(NOW)
    assert all(actions == (runtime.ACTION_STOP,) for actions, _ in owner._queued_calls)
    assert not owner._has_fresh_lateral_yaw(1)


@pytest.mark.parametrize('reason,remaining', [('none', 0.), ('fresh_position_deceleration', 0.)])
def test_ordinary_slew_is_not_replaced_by_continuity_rule(owner, monkeypatch, reason, remaining):
    _, result = prepare(owner, monkeypatch)
    owner._follow_controller.last_steering_pid_result = replace(result,
        brake_continuity_reason=reason, brake_continuity_remaining_sec=remaining)
    owner._service_lateral_intent(NOW)
    # Existing 1ms comfort slew retains -10, then ordinary yaw limit clips -7.
    assert owner._lateral_intent_last_correction_rpm == -7
    assert owner._current_rotate_raw_target == 7


def test_zero_publisher_keeps_old_signature_and_accepts_optional_pid_result(owner, monkeypatch, caplog):
    current, result = prepare(owner, monkeypatch)
    with caplog.at_level('INFO'):
        assert owner._publish_lateral_zero(current, 'legacy_caller')
    assert 'brake_continuity_reason=not_evaluated' in caplog.text
    current = _intent(owner, capture_frame_id=578)
    with caplog.at_level('INFO'):
        assert owner._publish_lateral_zero(current, 'continuity_caller', pid_result=result)
    assert 'brake_continuity_reason=fresh_position_deceleration' in caplog.text
