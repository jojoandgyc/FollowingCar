"""CAP730 and CAP833: cached PID fallback, no I/O, no renewed old permission."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision
from test_distance_tracking_response import setup, decide
from test_lateral_zero_runtime import owner, NOW


def prepare(setup, owner, monkeypatch):
    clock,c,frame=setup
    c.cfg=replace(c.cfg,forward_max_rpm=200,distance_matching_base_max_rpm=80)
    monkeypatch.setattr(runtime,'FORWARD_MAX_RPM',200)
    monkeypatch.setattr(runtime,'ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT',100)
    f=frame(2.377,rpm=38,stamp=NOW-.042)
    c._distance_pid_sample_timestamp=f.distance_state.sample_timestamp
    r=c._update_distance_pid(f.distance_m,now=clock.now)
    # Same terms as the existing PID, but with a walking prior in its output.
    c.last_distance_pid_result=replace(r,output_rpm=66,tracking_base_rpm=42.)
    c._distance_pid.last_result=c.last_distance_pid_result
    c._distance_pid._last_output_rpm=66.
    c._longitudinal_motion_evidence=SimpleNamespace(status='transient_bridge',eligible=False)
    c._longitudinal_bridge.origin=SimpleNamespace(sample_timestamp=NOW-.201)
    owner._follow_controller=c
    owner._depth30_linear_snapshot=('forward',37,1,NOW-.201)
    return clock,c,frame,f


def commit(o,f,percent=33):
    return o._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(percent,'longitudinal_distance_pid')],
                        reason='longitudinal_distance_pid'), f,1,is_fresh_depth=True)


def test_cap730_new_range_authorizes_distance_only_not_zero_or_old_66(setup,owner,monkeypatch,caplog):
    clock,c,frame,f=prepare(setup,owner,monkeypatch)
    before=c._distance_pid._last_ts,c._distance_pid._integral_m_s
    actions,accepted=commit(owner,f)
    assert accepted and 0<actions[0].speed_percent<=20 # <=40RPM, no old66RPM revival.
    timing=owner._depth30_linear_timing
    assert timing.depth_expires_at==pytest.approx(f.distance_state.sample_timestamp+.18)
    assert timing.feedforward_timestamp is None
    assert c._distance_pid._last_ts==before[0]
    assert c._distance_pid._integral_m_s<=before[1] # only anti-windup allowed.
    assert 'old_grant_reused=False new_depth_only=True' in caplog.text
    assert 'pid_recomputed=False' in caplog.text
    monkeypatch.setattr(runtime.time,'monotonic',lambda: NOW+.139)
    assert owner._fresh_depth_linear_snapshot(1) is None


@pytest.mark.parametrize('bad',['hazard','obstacle','near','raw_near','brake','safety',
    'depth_jump','depth_hold','stale','uid','search','feedback','feedback_stale','feedback_future',
    'yaw','raw_yaw','reverse','too_fast','nan','pid_mismatch'])
def test_new_range_recovery_does_not_bypass_safety(setup,owner,monkeypatch,bad):
    clock,c,frame,f=prepare(setup,owner,monkeypatch)
    if bad=='hazard':f=replace(f,hazard=replace(f.hazard,active=True))
    if bad=='obstacle':f=replace(f,obstacles=replace(f.obstacles,front=True))
    if bad=='near':f=replace(f,distance_m=1.7)
    if bad=='raw_near':f=replace(f,distance_state=replace(f.distance_state,raw_distance_m=1.7))
    if bad=='brake':f=replace(f,distance_state=replace(f.distance_state,brake_latched=True))
    if bad=='safety':f=replace(f,distance_state=replace(f.distance_state,safety_distance_m=.4))
    if bad=='depth_jump':f=replace(f,distance_state=replace(f.distance_state,source_detail='distance_jump_pending'))
    if bad=='depth_hold':f=replace(f,distance_state=replace(f.distance_state,source_detail='depth_multiregion_reused_hold'))
    if bad=='stale':f=replace(f,distance_state=replace(f.distance_state,sample_timestamp=NOW-.181))
    if bad=='uid':c.active_target_id=2
    if bad=='search':c.search_state='searching'
    if bad=='feedback':f=replace(f,steering_feedback=replace(f.steering_feedback,trustworthy=False))
    if bad=='feedback_stale':f=replace(f,steering_feedback=replace(f.steering_feedback,timestamp=NOW-.151))
    if bad=='feedback_future':f=replace(f,steering_feedback=replace(f.steering_feedback,timestamp=NOW+.01))
    if bad=='yaw':f=replace(f,steering_feedback=replace(f.steering_feedback,yaw_rate_right_dps=16))
    if bad=='raw_yaw':f=replace(f,steering_feedback=replace(f.steering_feedback,raw_yaw_rate_right_dps=-66))
    if bad=='reverse':f=replace(f,steering_feedback=replace(f.steering_feedback,left_forward_rpm=-1))
    if bad=='too_fast':f=replace(f,steering_feedback=replace(f.steering_feedback,left_forward_rpm=106))
    if bad=='nan':f=replace(f,steering_feedback=replace(f.steering_feedback,left_forward_rpm=float('nan')))
    if bad=='pid_mismatch':c._distance_pid_last_sample_timestamp=NOW-.06
    assert c.fresh_distance_recovery_percent(f,f.distance_state.sample_timestamp,NOW)==0


@pytest.mark.parametrize('bad',['stop','shutdown','not_running','parking','search','low_quality','backward'])
def test_runtime_blocks_recovery_even_if_range_only_getter_is_positive(setup,owner,monkeypatch,bad):
    _,c,_,f=prepare(setup,owner,monkeypatch)
    c.fresh_distance_recovery_percent=lambda *args:20
    if bad=='stop':owner._explicit_stop_requested=True
    if bad=='shutdown':owner._runtime_shutdown_requested=True
    if bad=='not_running':owner.running=False
    if bad=='parking':owner._brake_hold_active=True
    if bad=='search':owner.search_state='searching'
    if bad=='low_quality':owner._vision_control_state='target_visible_low_quality'
    if bad=='backward':owner._depth30_linear_snapshot=('backward',20,1,NOW-.2)
    actions,_=commit(owner,f)
    assert not any(a.kind=='forward' and a.speed_percent>0 for a in actions)


def test_zero_wheels_resume_only_one_bounded_step(setup,owner,monkeypatch):
    _,c,_,f=prepare(setup,owner,monkeypatch)
    f=replace(f,steering_feedback=replace(f.steering_feedback,left_forward_rpm=0,right_forward_rpm=0))
    assert 0<c.fresh_distance_recovery_percent(f,f.distance_state.sample_timestamp,NOW)<=6 # 12RPM


def test_replay_and_old_sample_cannot_reissue_fallback(setup,owner,monkeypatch):
    _,_,_,f=prepare(setup,owner,monkeypatch)
    commit(owner,f)
    timing=owner._depth30_linear_timing
    assert commit(owner,f)==([],False)
    assert owner._depth30_linear_timing is timing
    old=replace(f,distance_state=replace(f.distance_state,sample_timestamp=NOW-.06))
    assert commit(owner,old)==([],False)


@pytest.mark.parametrize('distance',[1.54,1.55,1.563,1.579])
def test_near_warming_cannot_reinject_launch_twenty(setup,distance,caplog):
    clock,c,frame=setup
    decide(c,frame(1.65,rpm=10,yaw=20))
    assert c._forward_active
    clock.now+=.1
    # Fresh range, estimator rebuilding after unsupported yaw.
    result=decide(c,frame(distance,rpm=8))
    assert result.current_forward_percent==0
    assert not c._forward_active
    assert 'near_no_matching_stop' in caplog.text
    clock.now+=.01
    assert decide(c,frame(distance,rpm=8,yaw=20)).current_forward_percent==0


def test_new_distance_outside_start_band_can_restart(setup):
    clock,c,frame=setup
    decide(c,frame(1.65,yaw=20));clock.now+=.1
    assert decide(c,frame(1.56,yaw=20)).current_forward_percent==0
    clock.now+=.1
    assert decide(c,frame(1.59,yaw=20)).current_forward_percent>0


def test_valid_person_motion_still_matches_inside_start_band(setup):
    clock,c,frame=setup
    decide(c,frame(1.55,rpm=25));clock.now+=.1
    r=decide(c,frame(1.55,rpm=25))
    assert r.current_forward_percent>0
    assert c.last_distance_pid_result.tracking_base_rpm>0


def test_disable_feedforward_preserves_legacy_hysteresis(setup):
    clock,c,frame=setup
    c.cfg=replace(c.cfg,distance_feedforward_enable=False)
    decide(c,frame(1.65));clock.now+=.1
    assert decide(c,frame(1.56)).current_forward_percent>0
