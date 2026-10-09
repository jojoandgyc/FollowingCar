"""Actual PID -> publisher -> fast-loop numerical brake eligibility."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import ControlAction, SteeringFeedback
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig
from car_control_modular.turn_buildup import steering_buildup_allowed
from test_lateral_zero_runtime import owner, _target, NOW, runtime


def result(yaw=6.27, sign=1, **changes):
    cfg=VisualSteeringPidConfig(enabled=True,image_error_only=True,image_brake_assist=True,
        image_capture_motion=True,camera_hfov_deg=66,deadband_deg=3,
        dynamic_large_error_deg=10,max_correction_rpm=10,
        predictive_brake_decel_dps2=60,predictive_brake_margin_deg=1.25,
        predictive_brake_response_sec=.1,execution_response_trial_sec=.35)
    f=SteeringFeedback(timestamp=NOW,trustworthy=True,
        yaw_rate_right_dps=sign*yaw,raw_yaw_rate_right_dps=sign*yaw)
    args=dict(now=NOW,visual_age_sec=.2153,target_image_rate_dps=sign*21.7)
    args.update(changes)
    return VisualSteeringPid(cfg).update(.5+sign*.2889,68,f,**args)


@pytest.mark.parametrize('sign',[-1,1])
def test_cap692_taper_label_without_reduction_is_eligible(sign):
    r=result(sign=sign)
    assert r.forward_phase == 'image_brake_assist:taper'
    assert r.correction_rpm == sign*10 and not r.predictive_braking
    assert r.position_demand_rpm == 10 and r.brake_reduction_rpm == 0
    assert steering_buildup_allowed(r)
    assert steering_buildup_allowed(replace(r,forward_phase='diagnostic_label_changed'))


def test_sub_rpm_braking_cannot_hide_behind_rounding_to_same_command():
    r=result(yaw=14.9)
    assert r.correction_rpm == 10
    assert 0 < r.brake_reduction_rpm < .5
    assert not steering_buildup_allowed(r)


@pytest.mark.parametrize('kind',['taper','stop','inward','zero_limit','relief'])
def test_actual_braking_and_bounded_relief_are_not_amplified(kind):
    args={'taper':dict(yaw=20), 'stop':dict(yaw=60),
          'inward':dict(target_image_rate_dps=-60),
          'zero_limit':dict(max_correction_override_rpm=0),
          'relief':dict(yaw=30)}[kind]
    r=result(**args)
    assert not steering_buildup_allowed(r)
    if kind == 'relief':
        assert r.forward_phase == 'image_outward_continuity' and r.correction_rpm == 4
        assert r.brake_reduction_rpm > 0


@pytest.mark.parametrize('change',[
    dict(position_demand_rpm=None), dict(brake_reduction_rpm=None),
    dict(position_demand_rpm=float('nan')), dict(brake_reduction_rpm=float('nan')),
    dict(brake_reduction_rpm=-1), dict(correction_rpm=0),dict(correction_rpm=-10),
    dict(correction_rpm=8),dict(outward_lead=object()),dict(post_park_recenter=True),
    dict(outward_continuity_rate_dps=3),dict(visual_direction_guarded=True),
    dict(same_direction_overspeed_braking=True),dict(opposite_yaw_braking=True),
    dict(output_floor_reason='center_hold'),dict(correction_limit_reason='legacy'),
])
def test_missing_or_contradictory_evidence_fails_closed(change):
    assert not steering_buildup_allowed(replace(result(),**change))


def test_real_publisher_accepts_unreduced_taper_and_fast_loop_revokes_actual_taper(owner,monkeypatch):
    monkeypatch.setattr(runtime,'MODULE_ASTRA_DEPTH_ENABLE',False)
    owner._last_control_decision_reason='visible_follow'
    c=owner._follow_controller
    c.last_steering_pid_result=result()
    assert owner._publish_lateral_intent_from_decision(width=640,target=_target(),
        runtime_actions=[ControlAction.forward(34,'follow')],control_source='vision',
        target_steerable=True,low_quality_visible=False)
    intent=owner._lateral_intent_store.snapshot()
    assert intent.response_boost_allowed
    owner._action_runtime=SimpleNamespace(get_steering_feedback=lambda:None)
    owner._service_lateral_intent(NOW)
    assert owner._lateral_turn_response_policy == (intent.sequence,True)
    c.refresh_visible_lateral_pid=lambda **kw:result(yaw=17.5)
    owner._service_lateral_intent(NOW+.04)
    assert owner._lateral_turn_response_policy == (intent.sequence,False)
    assert owner._depth30_linear_snapshot is None


def test_missing_numeric_fields_cannot_gain_permission_by_phase_label():
    assert not steering_buildup_allowed(SimpleNamespace(forward_phase='image_error_only'))
    assert not steering_buildup_allowed(None)
