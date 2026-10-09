"""CAP74/96–108 wheel requests with actual writer, fake feedback/motor only."""
from types import SimpleNamespace
from dataclasses import replace
import pytest

from car_control_modular.turn_response_trial import TurnResponseTrial
from test_cap1663_turn_response import prepare, intent
from test_visible_wheel_continuity import feedback


def setup(monkeypatch):
    r,o,d,s,clock=prepare(monkeypatch)
    r.backend.config=replace(r.backend.config,max_target=200)
    # Even a legacy config must not silently restore longitudinal trimming.
    r.config.follow_turn_acceleration_priority_enable=True
    o._follow_controller.cfg=SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_max_correction_rpm=10,
        near_distance_rotation_only_max_rpm=7)
    return r,o,d,s,clock


@pytest.mark.parametrize('mirror',[False,True])
@pytest.mark.parametrize('cap,base,yaw,measured',[
    (74,110,10,(32,29)), (96,106,8,(21,19)),
    (96,114,8,(20,18)), (96,116,0,(29,19)),
    (102,104,2,(27,23)), (107,122,0,(25,22)),
    (108,130,0,(24,22)),
])
def test_logged_forward_requests_are_preserved_through_real_writer(
        monkeypatch,cap,base,yaw,measured,mirror,caplog):
    r,o,d,s,clock=setup(monkeypatch)
    if mirror:yaw=-yaw;measured=measured[::-1]
    o._fresh_depth_linear_snapshot=lambda uid,now=None:('forward',base/2)
    o._lateral_intent_store.publish(intent(cap=cap,sign=-1 if mirror else 1,
        correction_limit_rpm=10,response_boost_allowed=False))
    r.get_steering_feedback=lambda:feedback(clock[0],*measured)
    r._turn_response_trial=TurnResponseTrial()
    r._turn_response_trial.record(1,(36,20) if not mirror else (20,36),9.95)
    with caplog.at_level('INFO'):
        r._send_follow_wheel_targets(base+yaw,-(base-yaw),'FOLLOW20')
    assert d.pairs[-1]==(base+yaw,-(base-yaw))
    assert (d.pairs[-1][0]-d.pairs[-1][1])/2==base
    assert 'turn_finish_common_cap' not in caplog.text
    assert 'turn_build_common_acceleration_limited' not in caplog.text
    assert 'execution_base_loss_rpm=0.0' in caplog.text
    assert not d.stops


def test_many_new_frames_and_stale_low_history_never_lock_24rpm(monkeypatch):
    r,o,d,s,clock=setup(monkeypatch)
    r._turn_response_trial=TurnResponseTrial()
    r._turn_response_trial.record(1,(36,20),9.95)
    r._turn_response_trial.cap=24  # Previously held state cannot survive.
    for n,base in enumerate([116,104,106,122,130,110,20,0]):
        clock[0]=10+n*.1
        o._fresh_depth_linear_snapshot=lambda uid,now=None:('forward',base/2) if base else None
        o._lateral_intent_store.publish(intent(clock[0],96+n,sign=1,hold_zero=True,
                                             visual_error_deg=0.,correction_limit_rpm=10))
        r.get_steering_feedback=lambda:feedback(clock[0],29,19)
        r._send_follow_wheel_targets(base,-base,'FOLLOW20')
        assert d.pairs[-1]==(base,-base)
    assert not d.stops


@pytest.mark.parametrize('base,cap,expected_base,expected_yaw',[
    (190,200,190,10),(195,200,195,5),(200,200,200,0),
    (195,180,180,0),(170,180,170,10),
])
@pytest.mark.parametrize('sign',[-1,1])
def test_outer_wheel_ceiling_shrinks_yaw_without_trimming_valid_base(
        monkeypatch,base,cap,expected_base,expected_yaw,sign):
    r,o,d,s,clock=setup(monkeypatch)
    o._fresh_depth_linear_snapshot=lambda uid,now=None:('forward',100)
    o._lateral_intent_store.publish(intent(sign=sign,correction_limit_rpm=10))
    r.get_steering_feedback=lambda:feedback(clock[0],20,20)
    r._send_follow_wheel_targets(base+sign*10,-(base-sign*10),'FOLLOW20',max_target_override=cap)
    left,right=d.pairs[-1][0],-d.pairs[-1][1]
    assert (left+right)/2==expected_base
    assert (left-right)/2==sign*expected_yaw
    assert max(left,right)<=cap


@pytest.mark.parametrize('change',['depth_loss','lower_depth','yaw_loss','identity_loss'])
def test_axis_authority_still_rechecked_before_write(monkeypatch,change):
    r,o,d,s,clock=setup(monkeypatch)
    o._fresh_depth_linear_snapshot=lambda uid,now=None:('forward',55)
    o._lateral_intent_store.publish(intent(sign=1,correction_limit_rpm=10,response_boost_allowed=False))
    r.get_steering_feedback=lambda:feedback(clock[0],32,29)
    r._send_follow_wheel_targets(120,-100,'FOLLOW20')
    assert d.pairs[-1]==(120,-100)
    if change=='depth_loss':
        o._fresh_depth_linear_snapshot=lambda uid,now=None:None
    elif change=='lower_depth':
        o._fresh_depth_linear_snapshot=lambda uid,now=None:('forward',10)
    elif change=='yaw_loss':o._has_fresh_lateral_yaw=lambda uid:False
    else:o._vision_control_state='target_visible_low_quality'
    r._send_follow_wheel_targets(120,-100,'FOLLOW20')
    if change in ('depth_loss','identity_loss'):assert d.pairs[-1]==(0,0)
    elif change=='lower_depth':assert d.pairs[-1]==(30,-10)
    else:assert d.pairs[-1]==(110,-110)
