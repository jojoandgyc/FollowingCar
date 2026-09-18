"""CAP253->255: independent depth clocks, bounded priors, no hardware."""
from dataclasses import replace
import pytest
from car_control_modular.control_types import DepthTargetObservation
from car_control_modular.longitudinal_feedforward import (
    LongitudinalFeedforwardEstimator, LongitudinalFeedforwardConfig,
    LongitudinalFeedforwardBridge, LongitudinalFeedforwardEvidence, depth_rotation_rate,
)
from test_distance_tracking_response import setup, decide
from test_scheduling_gap_evidence import missing
from test_longitudinal_feedforward import update


@pytest.mark.parametrize('age,yaw,allowed',[(.18,15,True),(.228,1,True),(.25,5,True),
    (.251,0,False),(.181,5.01,False),(.228,10,False),(-.03,0,False)])
def test_extended_alignment_only_low_yaw_not_longer_depth_lease(age,yaw,allowed):
    result=depth_rotation_rate(depth=2,bbox=(250,50,390,460),width=640,hfov_deg=60,
        capture_stamp=100,depth_stamp=100+age,yaw=yaw,low_yaw_max_age_sec=.25)
    assert (result is not None)==allowed


def observation(frame, clock, *, yaw=1, age=.1, distance=2.834, stamp=None):
    f=frame(distance,rpm=50,yaw=yaw,stamp=stamp)
    p=f.persons[0]
    raw=DepthTargetObservation(p.bbox,1,1,253,clock.now-age)
    return replace(f,persons=[replace(p,depth_observation=raw)])


def test_cap255_new_depth_remains_matching_across_180ms_rgb_boundary(setup):
    clock,c,frame=setup
    c.cfg=replace(c.cfg,distance_turn_compensation_enable=True)
    for age in (.10,.12,.228,.24):
        decide(c,observation(frame,clock,age=age))
        clock.now+=.05
    assert c._longitudinal_motion_evidence.sample_count==4
    assert c._longitudinal_motion_evidence.eligible
    assert c._longitudinal_motion_evidence.chain_reset_reason is None


def test_extended_rgb_cannot_make_old_physical_depth_fresh(setup):
    clock,c,frame=setup
    c.cfg=replace(c.cfg,distance_turn_compensation_enable=True)
    decide(c,observation(frame,clock))
    clock.now+=.1
    decide(c,observation(frame,clock,age=.24,stamp=clock.now-.181))
    assert c._tracking_base_rpm(2.834,clock.now) is None


def test_stable_compensated_no_observation_preserves_original_baseline_only(setup,caplog):
    clock,c,frame=setup
    c.cfg=replace(c.cfg,distance_turn_compensation_enable=True)
    for _ in range(3):
        decide(c,observation(frame,clock,yaw=10)); clock.now+=.03
    prior=c._longitudinal_feedforward._previous
    count=c._longitudinal_feedforward._sample_count
    for _ in range(3):
        f=missing(frame,yaw=10)
        c._observe_longitudinal_motion(f,f.persons[0]); clock.now+=.02
        assert c._longitudinal_feedforward._previous is prior
        assert c._longitudinal_feedforward._sample_count==count
        assert c._tracking_base_rpm(2.834,clock.now) is None
    decide(c,observation(frame,clock,yaw=10))
    assert c._longitudinal_motion_evidence.sample_count==count+1
    assert 'compensated_gap_preserved=True' in caplog.text


@pytest.mark.parametrize('change',['yaw','gap','mode','untrusted','uid'])
def test_next_endpoint_rechecks_preserved_gap(change):
    e=LongitudinalFeedforwardEstimator()
    update(e,now=100,ego=30,yaw_rate_dps=10,rotation_rate_m_s=.01)
    update(e,now=100.05,ego=30,yaw_rate_dps=10,rotation_rate_m_s=.01)
    assert e.preserve_compensated_gap(now=100.07,yaw=10)
    kwargs=dict(now=100.10,ego=30,yaw_rate_dps=10,rotation_rate_m_s=.01)
    if change=='yaw':kwargs['yaw_rate_dps']=14
    if change=='gap':kwargs['now']=100.24
    if change=='mode':kwargs.update(yaw_rate_dps=1,rotation_rate_m_s=None)
    if change=='untrusted':kwargs['trusted']=False
    if change=='uid':kwargs['target_id']=2
    r=update(e,**kwargs)
    assert not r.eligible
    assert r.sample_count<=1


@pytest.mark.parametrize('now,yaw',[(100.24,10),(100.07,14),(100.07,16),(100.07,float('nan'))])
def test_gap_preservation_fails_on_expiry_or_turn_change(now,yaw):
    e=LongitudinalFeedforwardEstimator()
    update(e,now=100.05,ego=30,yaw_rate_dps=10,rotation_rate_m_s=.01)
    assert not e.preserve_compensated_gap(now=now,yaw=yaw)


def bridge():
    b=LongitudinalFeedforwardBridge()
    b.remember(LongitudinalFeedforwardEvidence('ready',eligible=True,target_id=1,
        sample_timestamp=100,target_rpm=49.13),2.834)
    return b


def evaluate(b,**changes):
    args=dict(now=100.176,stamp=100.153,uid=1,distance=2.834,yaw=1,bearing=0,
              previous_output=82,baseline=20,fall_rate_rpm_per_sec=80,near_distance=1.7)
    args.update(changes)
    return b.evaluate(**args)


def test_cap255_bridge_drops_14rpm_not_29rpm_and_never_renews():
    b=bridge()
    r=evaluate(b)
    assert r.target_rpm==pytest.approx(35.05)
    assert b.origin.sample_timestamp==100
    assert evaluate(b,now=100.181,stamp=100.17) is None


@pytest.mark.parametrize('changes',[{'distance':2.82},{'distance':1.65},{'distance':3.2},
    {'uid':2},{'yaw':16},{'previous_output':0},{'stamp':100},
    {'fall_rate_rpm_per_sec':float('nan')},{'near_distance':None}])
def test_new_bridge_rejects_closing_near_jump_expired_or_invalid(changes):
    assert evaluate(bridge(),**changes) is None


def test_bridge_cannot_increase_an_already_reduced_output():
    assert evaluate(bridge(),previous_output=24).target_rpm==24


def test_controller_uses_new_bridge_policy_without_moving_its_deadline(setup,caplog):
    clock,c,frame=setup
    c.cfg=replace(c.cfg,distance_matching_base_max_rpm=80)
    c._longitudinal_bridge=bridge()
    c._distance_pid._last_output_rpm=82
    clock.now=100.176
    f=frame(2.834,rpm=50,stamp=100.153)
    assert c._bridge_longitudinal_motion(f,1,clock.now,100.153,1,0,'warming_up')
    assert c._longitudinal_motion_evidence.target_rpm==pytest.approx(35.05)
    assert c._longitudinal_bridge.origin.sample_timestamp==100
    assert 'decay_policy=bounded_80rpm_s' in caplog.text
    # Prior memory is separate from the fresh physical Depth deadline.
    c._distance_pid_sample_timestamp = 100.153
    assert c._tracking_base_rpm(2.834,100.181) is not None
    assert c._tracking_base_rpm(2.834,100.351) is None


def test_mode_change_remains_rejected_without_new_alignment_proof():
    e=LongitudinalFeedforwardEstimator()
    update(e,rotation_rate_m_s=.01)
    r=update(e,now=100.05,rotation_rate_m_s=None)
    assert r.chain_reset_reason=='compensation_mode_change'
    assert not r.eligible


def test_sudden_turn_stop_during_gap_also_rebuilds_derivative(setup):
    clock,c,frame=setup
    c.cfg=replace(c.cfg,distance_turn_compensation_enable=True)
    decide(c,observation(frame,clock,yaw=10))
    clock.now+=.03
    f=missing(frame,yaw=0)
    c._observe_longitudinal_motion(f,f.persons[0])
    assert c._longitudinal_feedforward._previous is None


def test_stationary_target_does_not_gain_speed_from_preserved_turn_gap():
    import math
    e=LongitudinalFeedforwardEstimator()
    # Fixed world point: camera rotating at 10deg/s; exact Z(t).
    omega=math.radians(10)
    for index,t in enumerate((0,.05,.15)):
        z=2*math.cos(omega*t)+.1*math.sin(omega*t)
        x=.1*math.cos(omega*t)-2*math.sin(omega*t)
        r=update(e,now=100+t,distance=z,ego=0,yaw_rate_dps=10,
                 rotation_rate_m_s=omega*x)
        if index:
            assert not r.eligible
            assert abs(r.target_speed_m_s)<.001
        if index==1:
            assert e.preserve_compensated_gap(now=100.08,yaw=10)
