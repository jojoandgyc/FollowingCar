"""Distance qualification must not depend on optional human-speed matching."""
from dataclasses import replace
from pathlib import Path
import os

import pytest

from car_control_modular.controllers import FollowSafetyController
from car_control_modular.control_types import DepthTargetObservation
from car_control_modular.longitudinal_approach import closure_rotation_bound, bounded_encoder_fallback
from car_control_modular.config_loader import load_config_to_env
from test_distance_tracking_response import setup, decide


def controller(setup, matching=True, feedforward=True):
    clock, old, frame = setup
    c = FollowSafetyController(replace(old.cfg,distance_approach_enable=True,
        distance_approach_matching_enable=matching,distance_approach_no_matching_max_rpm=60,
        distance_feedforward_enable=feedforward,distance_turn_compensation_enable=True,
        distance_feedforward_wheel_circumference_m=.816814,forward_max_rpm=200))
    c.active_target_id=1;c._has_seen_person=True
    return clock,c,frame


def off_axis(frame, clock, *, distance=3., yaw=0., age=.05, rpm=40.):
    f=frame(distance,rpm=rpm,yaw=yaw)
    p=replace(f.persons[0],bbox=(400.,50.,560.,460.))
    obs=DepthTargetObservation(p.bbox,1,1,100,clock.now-age)
    return replace(f,persons=[replace(p,depth_observation=obs)])


@pytest.mark.parametrize('yaw', [0., 4., -4.])
def test_off_axis_human_rejection_still_has_distance_budget(setup,yaw):
    clock,c,frame=controller(setup)
    for _ in range(8):
        decide(c,off_axis(frame,clock,yaw=yaw));clock.now+=.033
    assert c._tracking_base_rpm(3.,clock.now) is None
    assert c._braking_rate_source=='raw_depth_window'
    assert len(c._raw_closing_window.samples)>=3
    assert c.last_distance_pid_result.output_rpm==60
    assert c.last_distance_pid_result.tracking_base_rpm==0


def test_ff_reset_alone_cannot_erase_closure(setup):
    clock,c,frame=controller(setup)
    for _ in range(3):
        decide(c,frame(2.4,rpm=30));clock.now+=.05
    before=list(c._raw_closing_window.samples)
    c._clear_longitudinal_velocity_evidence(frame(2.4),'bearing_limit')
    assert c._raw_closing_window.samples==before
    assert c._matching_motion_window.samples==[]


def test_rejected_raw_jump_cannot_be_reseeded_by_same_frame(setup):
    clock,c,frame=controller(setup)
    for _ in range(3):
        f=frame(2.4,rpm=30)
        c._observe_longitudinal_motion(f,f.persons[0]);clock.now+=.04
    f=frame(1.8,rpm=30)
    for _ in range(3):
        c._observe_longitudinal_motion(f,f.persons[0])
        assert c._braking_rate_source=='raw_jump_protection'
        assert c._raw_closing_window.samples==[]


@pytest.mark.parametrize('feedforward',[True,False])
def test_distance_only_mode_does_not_require_estimator_or_bridge(setup,feedforward):
    clock,c,frame=controller(setup,matching=False,feedforward=feedforward)
    for _ in range(8):
        decide(c,frame(3.,rpm=40));clock.now+=.033
    assert c.last_distance_pid_result.tracking_base_rpm==0
    assert c.last_distance_pid_result.output_rpm==60
    assert c._braking_rate_source=='raw_depth_window'
    assert c._bridge_longitudinal_motion(frame(3.),1,clock.now,clock.now,0,0,'warming_up') is False
    if feedforward:
        assert c._longitudinal_motion_evidence.eligible # diagnostics still run


@pytest.mark.parametrize('change', ['yaw','geometry_age','depth_age','hazard','feedback_age'])
def test_independent_closure_keeps_safety_gates(setup,change):
    clock,c,frame=controller(setup)
    for _ in range(3):
        decide(c,off_axis(frame,clock));clock.now+=.05
    f=off_axis(frame,clock)
    if change=='yaw':f=replace(f,steering_feedback=replace(f.steering_feedback,yaw_rate_right_dps=20.))
    if change=='geometry_age':f=off_axis(frame,clock,age=.3)
    if change=='depth_age':f=frame(3.,stamp=clock.now-.181)
    if change=='hazard':f=replace(f,hazard=replace(f.hazard,active=True))
    if change=='feedback_age':f=replace(f,steering_feedback=replace(f.steering_feedback,timestamp=clock.now-.31))
    decide(c,f)
    assert not c._raw_closing_window.samples


@pytest.mark.parametrize('matching',[True,False])
def test_setpoint_static_target_cannot_start_sixty(setup,matching):
    clock,c,frame=controller(setup,matching)
    for _ in range(6):
        d=decide(c,frame(1.5,rpm=0));clock.now+=.05
        assert all(a.speed_percent==0 for a in d.actions)


def test_cap139_stale_feedback_bound_not_max_configuration(setup):
    clock,c,frame=controller(setup)
    decide(c,frame(2.44,rpm=39))
    c.last_distance_pid_result=replace(c.last_distance_pid_result,output_rpm=107)
    clock.now+=.05
    f=frame(2.44,rpm=39)
    f=replace(f,steering_feedback=replace(f.steering_feedback,timestamp=clock.now-.178))
    c._observe_longitudinal_motion(f,f.persons[0])
    assert c._braking_rate_source=='encoder_age_bound'
    assert c._braking_range_rate==pytest.approx(-107*.816814/60)
    assert c._raw_closing_window.samples==[] # stale feedback isn't a fresh closure observation


@pytest.mark.parametrize('age,prior_request,expected_source', [(.1,107,'encoder_fallback'),
    (.178,107,'encoder_age_bound'),(.178,0,'encoder_age_bound'),(.301,20,'encoder_unavailable_max_bound')])
def test_fallback_age_and_request_bounds(age,prior_request,expected_source):
    rpm,source=bounded_encoder_fallback(now=100,stamp=100-age,left=40,right=38,
        trustworthy=True,max_rpm=200,feedback_limit=130,last_request=prior_request,rise_rpm_s=240)
    assert source==expected_source
    if source=='encoder_age_bound':
        assert prior_request<=rpm<200
        assert rpm>=39+240*age-1e-8


@pytest.mark.parametrize('yaw',[0.,5.,-5.,15.,-15.])
def test_rotation_bound_is_sign_independent_and_never_fake_forward(yaw):
    bound=closure_rotation_bound(depth=2.,bearing_deg=15.,yaw_dps=yaw,geometry_age=.1)
    assert bound is not None and bound>=0
    assert bound==closure_rotation_bound(depth=2.,bearing_deg=-15.,yaw_dps=-yaw,geometry_age=.1)


@pytest.mark.parametrize('kwargs',[{'yaw_dps':16.},{'geometry_age':.3},{'bearing_deg':46.},
                                   {'depth':float('nan')}])
def test_invalid_rotation_bound_never_opens_budget(kwargs):
    args=dict(depth=2.,bearing_deg=15.,yaw_dps=5.,geometry_age=.1)
    assert closure_rotation_bound(**dict(args,**kwargs)) is None


@pytest.mark.parametrize('mode,expected',[('optional','1'),('distance_only','0')])
def test_explicit_trial_overrides_ini_without_disabling_depth(monkeypatch,mode,expected):
    monkeypatch.setattr(os,'environ',dict(os.environ))
    monkeypatch.setenv('FOLLOW_MATCHING_MODE',mode)
    load_config_to_env(str(Path(__file__).resolve().parents[2]/'car_control_modular/config/reid_runtime.ini'))
    assert os.environ['DISTANCE_APPROACH_MATCHING_ENABLE']==expected
    assert os.environ['DISTANCE_FEEDFORWARD_ENABLE']=='1'
    # Matching mode does not override the board's bounded 250ms grant TTL.
    assert os.environ['ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC']=='0.25'


def test_invalid_trial_fails_before_hardware(monkeypatch):
    monkeypatch.setenv('FOLLOW_MATCHING_MODE','typo')
    with pytest.raises(ValueError,match='FOLLOW_MATCHING_MODE'):
        load_config_to_env(str(Path(__file__).resolve().parents[2]/'car_control_modular/config/reid_runtime.ini'))
