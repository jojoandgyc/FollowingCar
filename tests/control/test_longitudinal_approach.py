"""Hardware-free speed reference and real controller/runtime boundary tests."""
from dataclasses import replace
from collections import deque
import os
from pathlib import Path

import pytest

from car_control_modular.longitudinal_approach import ApproachConfig, approach_reference
from car_control_modular.steering_pid import DistancePidConfig, LongitudinalDistancePid
from car_control_modular.controllers import FollowSafetyController
from car_control_modular.config_loader import load_config_to_env
from car_control_modular.control_types import ControlAction
import request_0513_modular as runtime
from test_distance_tracking_response import setup, decide


def ref(error, base=40., rate=0., config=None, target=1.5):
    return approach_reference(config or ApproachConfig(), error_m=(target+error)-target,
                              deadband_m=.03, tracking_base_rpm=base,
                              range_rate_m_s=rate, max_output_rpm=200.)


@pytest.mark.parametrize('error', [.10, .20, .50, 1., 2.])
def test_bounded_positive_correction_and_correct_units(error):
    r=ref(error)
    assert 0 < r.correction_rpm <= .60*60/.816814
    assert r.output_rpm == pytest.approx(40+r.correction_rpm)
    assert r.output_rpm <= r.cap_rpm <= 200
    assert r.correction_rpm*.816814/60 <= error-.03+1e-9


def test_half_metre_error_has_meaningful_catchup_not_old_twelve_rpm():
    r=ref(.5)
    assert r.correction_rpm == pytest.approx(.47*60/.816814)
    assert 34 < r.correction_rpm < 35 # old P24 added12; not 200RPM full throttle.


@pytest.mark.parametrize('base', [0., 20., 40., 80.])
def test_at_setpoint_matches_motion_but_static_stops(base):
    r=ref(0.,base)
    assert r.output_rpm == base
    assert r.correction_rpm == 0
    assert r.mode == ('matching' if base else 'stop')


def test_unknown_speed_does_not_invent_launch_bias_or_stopped_identity():
    assert ref(0,None).output_rpm == 0
    r=ref(.09,None)
    assert 0 < r.output_rpm < 5
    assert r.mode == 'distance_only'


def test_earlier_deceleration_for_faster_actual_closure():
    slow=ref(.2,0,rate=-.10)
    fast=ref(.2,0,rate=-.80)
    assert fast.output_rpm < slow.output_rpm
    assert fast.mode == 'decelerating'
    assert fast.braking_distance_m == pytest.approx(.8*.2+.8**2/(2*.4))
    assert ref(.1,0,rate=-.8).output_rpm == 0
    assert ref(.2,40,rate=-1).output_rpm == 40 # remove chase, preserve matching.


def test_delay_and_weaker_braking_can_only_reduce_envelope():
    normal=ref(1)
    assert ref(1,config=replace(ApproachConfig(),response_delay_sec=.5)).cap_rpm < normal.cap_rpm
    assert ref(1,config=replace(ApproachConfig(),deceleration_m_s2=.2)).cap_rpm < normal.cap_rpm


def test_measured_sample_age_adds_to_delay_not_to_authority_lifetime():
    c=pid()
    r=c.update(2.,1.5,now=100,tracking_base_rpm=0,measurement_age_sec=.15)
    young=pid().update(2.,1.5,now=100,tracking_base_rpm=0)
    assert r.approach_delay_sec==pytest.approx(.35)
    assert r.approach_cap_rpm<young.approach_cap_rpm


@pytest.mark.parametrize('initial_speed',[0.,.4,.8])
def test_stationary_target_with_assumed_braking_and_150ms_actuator_delay(initial_speed):
    # Ideal range and known stationary target; NOT an end-to-end hardware claim.
    c=pid(output_rise_rpm_per_sec=240,output_fall_rpm_per_sec=300)
    queue=deque([initial_speed]*3)
    distance,speed=2.9,initial_speed
    distances=[]
    for i in range(240):
        request=c.update(distance,1.5,now=100+i*.05,tracking_base_rpm=0).output_rpm
        queue.append(request*.816814/60)
        command=queue.popleft()
        speed+=max(-.4*.05,min(.8*.05,command-speed))
        distance-=speed*.05
        distances.append(distance)
    assert min(distances)>=1.40
    assert distances[-1]<=1.65


def test_constant_walking_target_converges_under_ideal_estimate_not_stop_start():
    c=pid(output_rise_rpm_per_sec=240,output_fall_rpm_per_sec=300)
    distance,speed=2.,0.
    queue=deque([0.]*3)
    for i in range(240):
        request=c.update(distance,1.5,now=100+i*.05,tracking_base_rpm=.6*60/.816814).output_rpm
        queue.append(request*.816814/60)
        command=queue.popleft()
        speed+=max(-.4*.05,min(.8*.05,command-speed))
        distance+=(.6-speed)*.05
    assert 1.4<=distance<=1.65
    assert speed==pytest.approx(.6,abs=.02)


def test_instant_target_stop_has_nonzero_unavoidable_physical_travel():
    # Even an immediate zero command cannot cancel inertia. Keep this limitation
    # explicit: matching0.6m/s at1.5m cannot guarantee >=1.4m if braking is0.4m/s².
    speed=.6
    travel=speed*.15+speed*speed/(2*.4)
    assert travel==pytest.approx(.54)
    assert 1.5-travel<1.4


def test_target_setting_translation_and_too_close():
    assert ref(.5,target=1.6) == ref(.5,target=1.5)
    assert ref(-.1,80).output_rpm == 0


@pytest.mark.parametrize('field', ['gain_per_sec','max_catchup_m_s','deceleration_m_s2',
                                  'response_delay_sec','wheel_circumference_m'])
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf'), 100.])
def test_invalid_model_config_rejected(field,value):
    with pytest.raises(ValueError):
        replace(ApproachConfig(),**{field:value})


@pytest.mark.parametrize('error,base,rate', [(float('nan'),0,0),(1,float('inf'),0),
                                          (1,-1,0),(1,0,float('nan'))])
def test_invalid_inputs_cannot_produce_motion(error,base,rate):
    with pytest.raises(ValueError): ref(error,base,rate)


def pid(**kw):
    return LongitudinalDistancePid(DistancePidConfig(
        approach_profile=ApproachConfig(), deadband_m=.03,
        max_forward_output_rpm=200, **kw))


def test_forward_replaces_not_stacks_pid_and_zeroes_integral():
    c=pid(kp_rpm_per_m=100,ki_rpm_per_m_s=100,kd_rpm_s_per_m=100)
    for i in range(10):
        r=c.update(2,1.5,now=100+i*.1,tracking_base_rpm=40)
        assert r.output_rpm == 74
        assert r.i_rpm == r.d_rpm == r.integral_m_s == 0
        assert r.distance_only_rpm == pytest.approx(ref(.5,None).output_rpm)


def test_initial_acceleration_bounded_and_braking_bypasses_fall_slew():
    c=pid(output_rise_rpm_per_sec=240,output_fall_rpm_per_sec=1)
    assert c.update(2.5,1.5,now=100,tracking_base_rpm=40).output_rpm <=24
    c.update(2.5,1.5,now=100.1,tracking_base_rpm=40)
    r=c.update(1.6,1.5,now=100.2,tracking_base_rpm=0)
    assert r.output_rpm == 0
    assert r.approach_mode == 'decelerating'


def test_duplicate_sample_does_not_accelerate_and_limit_feedback_is_used():
    c=pid(output_rise_rpm_per_sec=100)
    first=c.update(2.5,1.5,now=100,tracking_base_rpm=40)
    assert c.update(2.5,1.5,now=100,tracking_base_rpm=40) is first
    assert c.accept_output_limit(100,4)
    assert c.update(2.5,1.5,now=100.1,tracking_base_rpm=40).output_rpm <=14


def test_reverse_legacy_branch_unchanged():
    c=pid()
    legacy=LongitudinalDistancePid(replace(c.config,approach_profile=None))
    a,b=c.update(1.,1.5,now=100),legacy.update(1.,1.5,now=100)
    assert a == b and a.output_rpm < 0


def enabled_controller(setup):
    clock,old,frame=setup
    c=FollowSafetyController(replace(old.cfg,distance_approach_enable=True,
        distance_feedforward_wheel_circumference_m=.816814,
        distance_matching_base_max_rpm=80,forward_max_rpm=200))
    c.active_target_id=1;c._has_seen_person=True
    return clock,c,frame


def test_real_controller_matches_at_setpoint_and_stops_static(setup):
    clock,c,frame=enabled_controller(setup)
    decide(c,frame(1.5,rpm=30));clock.now+=.1
    out=decide(c,frame(1.5,rpm=30))
    assert out.current_forward_percent==15
    assert c.last_distance_pid_result.approach_mode=='matching'
    clock.now+=.1
    # Existing estimator integrates endpoint wheel speeds: 30 -> 0 still
    # means mean15 during this interval, not an instantaneous stopped person.
    assert decide(c,frame(1.5,rpm=0)).current_forward_percent<out.current_forward_percent
    clock.now+=.1
    assert decide(c,frame(1.5,rpm=0)).current_forward_percent==0


def test_near_profile_no_twenty_rpm_reinjection_or_158_premature_hold(setup):
    clock,c,frame=enabled_controller(setup)
    decide(c,frame(1.65,yaw=20));clock.now+=.1
    decide(c,frame(1.56,yaw=20))
    assert c.last_distance_pid_result is not None # no legacy reset at1.58.
    assert c.last_distance_pid_result.output_rpm < 20
    clock.now+=.1
    assert decide(c,frame(1.53,yaw=20)).current_forward_percent==0


@pytest.mark.parametrize('bad',['expired','hazard','obstacle','lost','jump','held'])
def test_profile_never_bypasses_existing_authority_guards(setup,bad):
    clock,c,frame=enabled_controller(setup)
    decide(c,frame(2.,rpm=40));clock.now+=.1
    f=frame(2.,rpm=40)
    if bad=='expired': f=frame(2.,stamp=clock.now-.3)
    if bad=='hazard': f=replace(f,hazard=replace(f.hazard,active=True))
    if bad=='obstacle': f=replace(f,obstacles=replace(f.obstacles,front=True))
    if bad=='lost': f=replace(f,persons=[])
    if bad=='jump': f=frame(2.,detail='distance_jump_pending')
    if bad=='held': f=replace(f,distance_state=replace(f.distance_state,
        raw_distance_m=None,source_detail='depth_multiregion_reused_hold'))
    out=decide(c,f)
    assert not any(a.kind=='forward' and a.speed_percent>0 for a in out.actions)


def test_replay_ff_revocation_uses_cached_distance_only_without_launch_twenty(setup):
    clock,c,frame=enabled_controller(setup)
    decide(c,frame(2.,rpm=40));clock.now+=.1
    f=frame(2.,rpm=40);decide(c,f)
    r=c.last_distance_pid_result
    assert r.tracking_base_rpm>0
    c._longitudinal_motion_evidence=None
    r2=c._update_distance_pid(2.,now=clock.now)
    assert r2.output_rpm<=r.distance_only_rpm
    assert r2.tracking_base_rpm==0
    assert c._distance_pid._last_ts==f.distance_state.sample_timestamp
    assert c.distance_only_forward_percent(f,f.distance_state.sample_timestamp)*2<=r.distance_only_rpm+1


def test_profile_disables_experimental_matching_bias(setup):
    clock,c,frame=enabled_controller(setup)
    c.cfg=replace(c.cfg,distance_matching_test_bias_rpm=10)
    for _ in range(5):
        decide(c,frame(2.5,rpm=40));clock.now+=.03
    evidence=c._longitudinal_motion_evidence
    assert c._tracking_base_rpm(2.5,clock.now)==evidence.target_rpm


def test_profile_feedback_budget_does_not_reject_its_own_command(setup):
    clock,c,frame=enabled_controller(setup)
    limit=c._longitudinal_feedforward.config.max_abs_ego_rpm
    assert limit==pytest.approx(80+.6*60/.816814+5)
    decide(c,frame(2.8,rpm=120));clock.now+=.03
    # Car1.63m/s closing0.55m/s -> walking person1.08m/s, not a1.63m/s person.
    decide(c,frame(2.8-.55*.03,rpm=120))
    assert c._longitudinal_motion_evidence.status!='ego_speed_out_of_bounds'
    assert c._longitudinal_motion_evidence.eligible
    clock.now+=.03
    decide(c,frame(2.8,rpm=150))
    assert c._longitudinal_motion_evidence.status=='ego_speed_out_of_bounds'


def test_runtime_allows_requested_profile_correction_not_legacy_twenty(monkeypatch):
    for name,val in dict(DISTANCE_APPROACH_ENABLE=True,FOLLOW_ROTATION_ONLY=False,
        DISTANCE_APPROACH_MAX_CATCHUP_M_S=.6,FORWARD_MAX_RPM=200,DISTANCE_MATCHING_BASE_MAX_RPM=80,
        VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M=.816814,
        ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT=20,
        ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT=100,
        ASTRA_DEPTH_LONGITUDINAL_FAR_DISTANCE_M=2.2,ASTRA_DEPTH_NEAR_GUARD_DISTANCE_M=1.8,
        TARGET_DISTANCE=1.5,DISTANCE_PID_DEADBAND_M=.03).items():
        monkeypatch.setattr(runtime,name,val)
    def cap(percent,dist=1.9):
        return runtime.PersonTracker._cap_depth_longitudinal_actions(
            [ControlAction.forward(percent,'test')],dist,60)[0].speed_percent
    assert cap(45)==45 # 90RPM =60+30, formerly capped80.
    assert cap(0)==0
    assert cap(80)<=52 # no unbounded boost to160RPM.
    monkeypatch.setattr(runtime,'DISTANCE_APPROACH_ENABLE',False)
    assert cap(45)==40


def test_config_wires_profile_with_shared_grant_ttl_and_unchanged_target(monkeypatch):
    env={};monkeypatch.setattr(os,'environ',env)
    config=Path(__file__).resolve().parents[2]/'car_control_modular/config/reid_runtime.ini'
    load_config_to_env(str(config))
    assert env['DISTANCE_APPROACH_ENABLE']=='1'
    assert float(env['DISTANCE_APPROACH_GAIN_PER_SEC'])==1.
    assert float(env['DISTANCE_APPROACH_DECELERATION_M_S2'])==.4
    assert float(env['DISTANCE_APPROACH_MAX_CATCHUP_M_S'])==.6
    assert float(env['DISTANCE_APPROACH_RESPONSE_DELAY_SEC'])==.2
    assert float(env['TARGET_DISTANCE'])==1.5
    assert float(env['ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC'])==.25
