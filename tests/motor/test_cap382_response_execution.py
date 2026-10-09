"""Actual writer + response model: no serial hardware, no sleeps or queues."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from car_control_modular.turn_response_trial import TurnResponseTrial, forward_brake_allowed
from test_cap1663_turn_response import prepare, intent
from test_visible_wheel_continuity import feedback


def brake_intent(now=10., cap=382, **kw):
    return intent(now,cap,x_ratio=.35,target_image_rate_dps=20.,
        correction_limit_rpm=10,response_boost_allowed=False,forward_countersteer=True,**kw)


def fb(now=10.,left=23,right=42):
    f=feedback(now,left,right)
    f.yaw_rate_right_dps=-25.
    f.raw_yaw_rate_right_dps=-28.
    return f


def test_taper_and_zero_preserve_current_longitudinal_authority():
    t=TurnResponseTrial();t.record(1,(19,39),9.95)
    b,y,phase=t.adjust(70,-7,intent(),fb(),10.,1,.35)
    assert (b,y,phase)==(70.,-7.,"tracking")
    t.record(1,(b+y,b-y),10.)
    for now in [10.05,10.10,10.20,10.34]:
        b,y,phase=t.adjust(84,0,intent(now),fb(now,34,56),now,1,.35)
        assert b==84 and y==0 and t.cap is None
        t.record(1,(b,b),now)
    # Quiet feedback is no longer a prerequisite for longitudinal recovery.
    f=fb(10.36,30,30)
    assert t.adjust(84,0,intent(10.36),f,10.36,1,.35)[0]==84
    assert t.adjust(84,0,intent(10.37),f,10.37,1,.35)[0]==84
    assert t.adjust(84,0,intent(10.42),fb(10.42,30,30),10.42,1,.35)[0]==84


def test_history_never_grants_motion_or_raises_braking_and_uid_resets():
    t=TurnResponseTrial();t.record(1,(19,39),9.95)
    assert t.adjust(70,-7,intent(),fb(),10,1,.35)[0]==70
    assert t.adjust(8,0,intent(),fb(),10.02,1,.35)[:2]==(8,0)
    assert t.adjust(0,0,None,None,10.03,1,.35)[:2]==(0,0)
    assert t.adjust(70,-7,intent(),fb(),10.04,2,.35)[:2]==(70,-7)
    assert t.adjust(70,-7,intent(),fb(),10.05,2,0)[:2]==(70,-7)


@pytest.mark.parametrize("bad", ["stale_image","expired","uid","limited","no_rate",
    "outward","stale_feedback","reverse_feedback","disagree","wrong_yaw","no_depth","weak","oversize"])
def test_forward_brake_final_evidence_veto(bad):
    i=brake_intent();f=fb();base,yaw=40,6
    assert forward_brake_allowed(base,yaw,i,f,10,1)
    if bad=="stale_image":i=replace(i,capture_timestamp=9.7)
    if bad=="expired":i=replace(i,valid_until=9.9)
    if bad=="uid":i=replace(i,target_id=2)
    if bad=="limited":i=replace(i,bbox_quality="limited")
    if bad=="no_rate":i=replace(i,target_image_rate_dps=None)
    if bad=="outward":i=replace(i,target_image_rate_dps=-20)
    if bad=="stale_feedback":f.timestamp=9.8
    if bad=="reverse_feedback":f.left_forward_rpm=-1
    if bad=="disagree":f.raw_yaw_rate_right_dps=25
    if bad=="wrong_yaw":f.raw_yaw_rate_right_dps=f.yaw_rate_right_dps=25
    if bad=="no_depth":base=0
    if bad=="weak":f.raw_yaw_rate_right_dps=f.yaw_rate_right_dps=-2
    if bad=="oversize":yaw=7
    assert not forward_brake_allowed(base,yaw,i,f,10,1)
    t=TurnResponseTrial()
    assert t.adjust(base,yaw,i,f,10,1,.35)[1]==0


def test_forward_brake_pulse_bounded_across_new_frames_and_no_depth_loss_pivot():
    t=TurnResponseTrial(); t.record(1,(19,39),9.95)
    for j,now in enumerate([10.,10.05,10.081,10.15,10.2]):
        b,y,p=t.adjust(40,6,brake_intent(now,382+j),fb(now),now,1,.35)
        assert y==(6 if j<2 else 0)
        assert b-y>=0 and b+y>=0
        t.record(1,(b+y,b-y),now)
    assert t.adjust(0,6,brake_intent(10.21),fb(10.21),10.21,1,.35)[1]==0


def setup_writer(monkeypatch):
    r,o,d,s,clock=prepare(monkeypatch)
    r.config.follow_turn_response_assist_enable=False
    o._follow_controller.cfg=SimpleNamespace(visible_steering_pid_image_error_only=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_max_correction_rpm=10,near_distance_rotation_only_max_rpm=7)
    o._fresh_depth_linear_snapshot=lambda uid,now=None:("forward",20)
    o._lateral_intent_store.publish(brake_intent())
    r.get_steering_feedback=lambda:fb(clock[0])
    return r,o,d,s,clock


def test_real_writer_forward_brake_not_pivot_then_zero_diff_on_deadline(monkeypatch):
    r,o,d,s,clock=setup_writer(monkeypatch)
    r._send_follow_wheel_targets(46,-34,"FOLLOW20")
    assert d.pairs[-1][0]+d.pairs[-1][1]==12
    assert d.pairs[-1][0]>=0 and d.pairs[-1][1]<=0 and not d.stops
    assert len(r._turn_response_trial.history)==1
    clock[0]=10.09
    r._send_follow_wheel_targets(46,-34,"FOLLOW20")
    assert d.pairs[-1][0]==-d.pairs[-1][1]
    o._fresh_depth_linear_snapshot=lambda uid,now=None:None
    r._send_follow_wheel_targets(6,6,"FOLLOW20")
    assert d.pairs[-1]==(0,0)


@pytest.mark.parametrize("change",["time","intent","feedback","depth_after_feedback"])
def test_revalidate_after_guard_before_actual_brake_write(monkeypatch,change):
    r,o,d,s,clock=setup_writer(monkeypatch)
    original=r._visible_wheel_guard.limit
    def replace_evidence(*args,**kw):
        result=original(*args,**kw)
        if change=="time":clock[0]+=.12
        if change=="intent":o._lateral_intent_store.publish(brake_intent(cap=383))
        if change=="feedback":r.get_steering_feedback=lambda:fb(clock[0],30,30)
        if change=="depth_after_feedback":
            def reading():
                o._fresh_depth_linear_snapshot=lambda uid,now=None:None
                return fb(clock[0])
            r.get_steering_feedback=reading
        return result
    r._visible_wheel_guard.limit=replace_evidence
    r._send_follow_wheel_targets(46,-34,"FOLLOW20")
    assert d.pairs[-1]==(0,0)
    assert not r._turn_response_trial.history
