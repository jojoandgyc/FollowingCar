from dataclasses import replace
import pytest
from car_control_modular.longitudinal_feedforward import LongitudinalFeedforwardConfig, LongitudinalFeedforwardEstimator
from car_control_modular.controllers import FollowSafetyController
from test_longitudinal_feedforward import update
from test_distance_tracking_response import setup, decide


def warmed(distance=2.05):
    e=LongitudinalFeedforwardEstimator(LongitudinalFeedforwardConfig(max_tracking_base_rpm=80))
    for i in range(5):
        update(e,now=100+i*.04,distance=distance,ego=20)
    return e


def suspect(e,now=100.20,distance=2.0412,**changes):
    return update(e,now=now,distance=distance,ego=20,**changes)


def test_cap345_single_small_negative_does_not_erase_positive_window():
    e=warmed()
    old=e.last_result.target_rpm
    r=suspect(e)
    assert r.instantaneous_target_speed_m_s==pytest.approx(-.02)
    assert r.window_target_speed_m_s>.10
    assert r.decline_policy=='stop_suspect_bounded'
    assert 0<r.target_rpm<old
    assert r.target_rpm==pytest.approx(old-80*.04)
    assert r.sample_count==6
    assert len(e._speed_samples)>=4
    again=suspect(e)  # same physical Depth cannot consume the next confirmation
    assert again.status=='duplicate_depth' and not again.eligible
    assert e._stop_suspect_pending
    assert suspect(e,now=100.24,distance=2.0324).status=='no_forward_motion'


def test_one_positive_between_weak_samples_cannot_repeat_grace_forever():
    e=warmed(); suspect(e)
    update(e,now=100.24,distance=2.0412,ego=20)
    assert suspect(e,now=100.28,distance=2.0324).status=='no_forward_motion'


def test_two_positive_samples_rearm_a_later_independent_suspect():
    e=warmed(); suspect(e)
    for now in (100.24,100.28): update(e,now=now,distance=2.0412,ego=20)
    assert not e._stop_suspect_pending


@pytest.mark.parametrize('condition',['strong_negative','near','ttc','no_window','old_pair',
    'uid','untrusted','depth_stale','feedback_stale','jump','yaw'])
def test_suspect_does_not_soften_real_safety_rejections(condition):
    e=warmed(1.79 if condition=='near' else 2.05)
    args=dict(now=100.20,distance=1.7812 if condition=='near' else 2.0412)
    if condition=='strong_negative':args['distance']=2.034  # target -0.2 m/s
    if condition=='ttc':
        e=warmed(1.81); args['distance']=1.78
    if condition=='no_window':e._speed_samples=e._speed_samples[-1:]
    if condition=='old_pair':args.update(now=100.30,distance=2.0192)
    if condition=='uid':args['target_id']=2
    if condition=='untrusted':args['trusted']=False
    if condition=='depth_stale':args['sample_timestamp']=99.8
    if condition=='feedback_stale':args['feedback_timestamp']=99.8
    if condition=='jump':args['distance']=3.
    if condition=='yaw':args['yaw_rate_dps']=20
    r=suspect(e,**args)
    assert r.decline_policy!='stop_suspect_bounded'
    assert not r.eligible


def test_bad_frame_clears_suspect_history_instead_of_resuming_old_walk():
    e=warmed();suspect(e)
    update(e,now=100.24,distance=2.0412,ego=20,trusted=False)
    assert not e._stop_suspect_pending
    assert update(e,now=100.28,distance=2.0412,ego=20).status=='warming_up'


def controller_with_walk(setup,bias=5):
    clock,old,frame=setup
    c=FollowSafetyController(replace(old.cfg,distance_matching_base_max_rpm=80,
        distance_matching_test_bias_rpm=bias,forward_max_rpm=200))
    c.active_target_id=1;c._has_seen_person=True
    for i in range(5):
        if i:clock.now+=.04
        decide(c,frame(2.1,rpm=30))
    return clock,c,frame


def test_bias_changes_pid_request_not_estimator_or_sample_time(setup):
    clock,c,frame=controller_with_walk(setup)
    e=c._longitudinal_motion_evidence
    assert e.target_rpm==pytest.approx(30)
    assert c.last_distance_pid_result.tracking_base_rpm==pytest.approx(35)
    for _ in range(10):
        assert c._tracking_base_rpm(2.1,clock.now)==pytest.approx(35)
        assert c._longitudinal_motion_evidence is e
    assert c._longitudinal_motion_stamp==e.sample_timestamp
    assert c._longitudinal_bridge.origin.target_rpm==pytest.approx(30)
    assert c._tracking_base_rpm(2.1,clock.now+.181) is None


@pytest.mark.parametrize('distance,bonus',[(1.7,0),(1.85,2.5),(2.,5),(3.,5)])
def test_bias_tapers_near_setpoint(setup,distance,bonus):
    clock,c,frame=controller_with_walk(setup)
    assert c._tracking_base_rpm(distance,clock.now)==pytest.approx(30+bonus)


@pytest.mark.parametrize('change',['bridge','suspect','stop','small_instant','small_window','warming',
    'replay','uid','capped'])
def test_bias_cannot_add_to_unqualified_or_already_capped_estimate(setup,change):
    clock,c,frame=controller_with_walk(setup)
    e=c._longitudinal_motion_evidence
    if change=='bridge':e=replace(e,status='transient_bridge')
    if change=='suspect':e=replace(e,decline_policy='stop_suspect_bounded')
    if change=='stop':e=replace(e,decline_policy='stop_evidence',eligible=False)
    if change=='small_instant':e=replace(e,instantaneous_target_speed_m_s=.01)
    if change=='small_window':e=replace(e,window_target_speed_m_s=.05)
    if change=='warming':e=replace(e,status='warming_up',eligible=False)
    if change=='replay':c._distance_pid_sample_timestamp=None
    if change=='uid':c.active_target_id=2
    if change=='capped':e=replace(e,target_rpm=80)
    c._longitudinal_motion_evidence=e
    result=c._tracking_base_rpm(2.1,clock.now)
    assert result is None or result==pytest.approx(e.target_rpm)


@pytest.mark.parametrize('condition',['hazard','stale','lost','bad_depth'])
def test_full_decision_still_revokes_bias_under_safety_conditions(setup,condition):
    clock,c,frame=controller_with_walk(setup)
    clock.now+=.04;f=frame(2.1,rpm=30)
    if condition=='hazard':f=replace(f,hazard=replace(f.hazard,active=True))
    if condition=='stale':f=replace(f,distance_state=replace(f.distance_state,sample_timestamp=clock.now-.181))
    if condition=='lost':f=replace(f,persons=[])
    if condition=='bad_depth':f=replace(f,distance_state=replace(f.distance_state,source_detail='distance_jump_pending'))
    decide(c,f)
    assert c._tracking_base_rpm(2.1,clock.now) is None
