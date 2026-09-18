"""CAP193 yaw sign disagreement and CAP253 102/100RPM: no hardware."""
from dataclasses import replace
import pytest

from car_control_modular.controllers import FollowSafetyController
from car_control_modular.control_types import DepthTargetObservation
from car_control_modular.longitudinal_feedforward import (
    bounded_disagreeing_yaw_rotation, depth_rotation_rate,
)
from test_distance_tracking_response import setup, decide


def geometry(**changes):
    return dict(depth=1.96, bbox=(175.08,30.33,380.95,476.28), width=640,
                hfov_deg=60, capture_stamp=100., depth_stamp=100.138,
                low_yaw_max_age_sec=.25, **changes)


def test_cap193_envelope_is_small_and_conservative_in_both_directions():
    args=geometry()
    result=bounded_disagreeing_yaw_rotation(raw_yaw=-9.405, filtered_yaw=3.2544, **args)
    assert result is not None and 0 < result[0] <= .04
    for i in range(-100,101):
        actual=depth_rotation_rate(**args,yaw=9.405*i/100)
        # range_rate - bound cannot overestimate speed from a yaw inside the envelope.
        assert actual is not None and actual[0] <= result[0]+1e-12


@pytest.mark.parametrize('changes', [dict(raw_yaw=16), dict(raw_yaw=float('nan')),
    dict(filtered_yaw=float('nan')), dict(raw_yaw=9), dict(depth=4),
    dict(depth_stamp=100.181), dict(depth_stamp=99.97),
    dict(bbox=(0,0,60,400)), dict(bbox=(400,10,100,400))])
def test_envelope_rejects_large_unknown_or_misaligned_rotation(changes):
    args=dict(raw_yaw=-9.405,filtered_yaw=3.2544,**geometry())
    args.update(changes)
    assert bounded_disagreeing_yaw_rotation(**args) is None


def current(frame,clock,*,distance=2.,raw=3.2544,filtered=3.2544,rpm=62):
    f=frame(distance,rpm=rpm,yaw=filtered)
    obs=DepthTargetObservation((175.08,30.33,380.95,476.28),1,1,193,clock.now-.138)
    return replace(f,persons=[replace(f.persons[0],bbox=obs.bbox,depth_observation=obs)],
                   steering_feedback=replace(f.steering_feedback,raw_yaw_rate_right_dps=raw))


def seeded(setup):
    clock,old,frame=setup
    c=FollowSafetyController(replace(old.cfg,forward_max_rpm=200,
        distance_turn_compensation_enable=True,distance_matching_base_max_rpm=80))
    c.active_target_id=1
    c._has_seen_person=True
    for _ in range(4):
        decide(c,current(frame,clock))
        clock.now+=.05
    assert c._longitudinal_motion_evidence.eligible
    return clock,c,frame


def test_cap193_sign_disagreement_no_longer_erases_matching(setup,caplog):
    caplog.set_level('INFO',logger='PersonTracker')
    clock,c,frame=seeded(setup)
    previous=c._longitudinal_motion_evidence
    clock.now+=.11  # 160ms since previous physical Depth
    f=current(frame,clock,distance=1.956,raw=-9.405)
    decide(c,f)
    assert c._longitudinal_motion_evidence.eligible
    assert c._longitudinal_motion_evidence.sample_count==previous.sample_count+1
    assert c._longitudinal_motion_evidence.chain_reset_reason is None
    assert c.last_distance_pid_result.tracking_base_rpm>0
    assert 'compensation=yaw_disagreement_bounded' in caplog.text
    # A repeat cannot move either timestamp or recompute PID.
    pid=c.last_distance_pid_result
    clock.now+=.02
    decide(c,f)
    assert c.last_distance_pid_result is pid
    assert c._longitudinal_motion_stamp==f.distance_state.sample_timestamp


@pytest.mark.parametrize('condition',['near','near_raw','depth_stale','depth_jump','uid',
    'hazard','search','untrusted','feedback_stale','feedback_future','feedback_skew',
    'yaw','geometry','wide_envelope','no_raw_box'])
def test_disagreement_does_not_bypass_existing_gates(setup,condition,caplog):
    caplog.set_level('INFO',logger='PersonTracker')
    clock,c,frame=seeded(setup)
    clock.now+=.2  # prior bridge is expired; no alternate evidence to fall back on
    f=current(frame,clock,raw=-9.405)
    if condition=='near': f=replace(f,distance_m=1.6)
    if condition=='near_raw':f=replace(f,distance_state=replace(f.distance_state,raw_distance_m=1.6))
    if condition=='depth_stale':f=replace(f,distance_state=replace(f.distance_state,sample_timestamp=clock.now-.181))
    if condition=='depth_jump':f=replace(f,distance_state=replace(f.distance_state,source_detail='distance_jump_pending'))
    if condition=='uid':c.active_target_id=2
    if condition=='hazard':f=replace(f,hazard=replace(f.hazard,active=True))
    if condition=='search':c.search_state='searching'
    if condition=='untrusted':f=replace(f,steering_feedback=replace(f.steering_feedback,trustworthy=False))
    if condition=='feedback_stale':f=replace(f,steering_feedback=replace(f.steering_feedback,timestamp=clock.now-.16))
    if condition=='feedback_future':f=replace(f,steering_feedback=replace(f.steering_feedback,timestamp=clock.now+.01))
    if condition=='feedback_skew':f=replace(f,steering_feedback=replace(f.steering_feedback,timestamp=clock.now-.06))
    if condition=='yaw':f=replace(f,steering_feedback=replace(f.steering_feedback,raw_yaw_rate_right_dps=-16))
    if condition=='geometry':f=replace(f,persons=[replace(f.persons[0],depth_observation=replace(f.persons[0].depth_observation,capture_timestamp=clock.now-.181))])
    if condition=='wide_envelope':f=replace(f,distance_m=4,distance_state=replace(f.distance_state,raw_distance_m=4,used_distance_m=4))
    if condition=='no_raw_box':f=replace(f,persons=[replace(f.persons[0],depth_observation=None)])
    caplog.clear()
    c._observe_longitudinal_motion(f,f.persons[0])
    assert c._tracking_base_rpm(f.distance_m,clock.now) is None
    assert 'compensation=yaw_disagreement_bounded' not in caplog.text


def test_102_100_wheel_feedback_keeps_physical_samples(setup):
    clock,c,frame=seeded(setup)
    f=current(frame,clock,rpm=101)
    f=replace(f,steering_feedback=replace(f.steering_feedback,left_forward_rpm=102,right_forward_rpm=100))
    count=c._longitudinal_motion_evidence.sample_count
    decide(c,f)
    assert c._longitudinal_motion_evidence.sample_count==count+1
    assert c._longitudinal_motion_evidence.eligible
    assert c._longitudinal_feedforward._previous.ego_forward_m_s==pytest.approx(1.01)
    assert c._longitudinal_feedforward.config.max_abs_ego_rpm==105
    assert c.cfg.forward_max_rpm==200 and c.cfg.distance_matching_base_max_rpm==80


@pytest.mark.parametrize('rpm',[105.1,150,200,float('inf'),float('nan')])
def test_200_motor_ceiling_does_not_allow_unverified_200_feedback(setup,rpm):
    clock,c,frame=seeded(setup)
    decide(c,current(frame,clock,rpm=rpm))
    assert c._tracking_base_rpm(2.,clock.now) is None


def test_standard_100rpm_configuration_keeps_original_feedback_limit(setup):
    assert setup[1]._longitudinal_feedforward.config.max_abs_ego_rpm==100


def test_far_stationary_target_cannot_gain_positive_speed_from_sign_ambiguity(setup):
    clock,old,frame=setup
    c=FollowSafetyController(replace(old.cfg,distance_turn_compensation_enable=True))
    c.active_target_id=1
    c._has_seen_person=True
    for _ in range(5):
        f=current(frame,clock,raw=-9.405,rpm=0)
        c._observe_longitudinal_motion(f,f.persons[0])
        assert not c._longitudinal_motion_evidence.eligible
        assert c._longitudinal_motion_evidence.target_rpm==0
        clock.now+=.04


def test_measurement_margin_is_not_a_new_matching_or_actuation_budget(setup):
    clock,c,frame=seeded(setup)
    for _ in range(10):
        decide(c,current(frame,clock,rpm=104))
        clock.now+=.04
    assert c._longitudinal_motion_evidence.eligible
    assert c._longitudinal_motion_evidence.target_rpm<=80
    clock.now+=.181
    f=current(frame,clock,rpm=104)
    f=replace(f,distance_state=replace(f.distance_state,sample_timestamp=clock.now-.181))
    decide(c,f)
    assert c._tracking_base_rpm(2.,clock.now) is None
