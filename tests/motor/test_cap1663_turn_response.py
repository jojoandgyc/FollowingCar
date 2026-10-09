"""Bounded turn buildup and center residual handoff; fake motor and clock."""
from dataclasses import replace
import ast
from pathlib import Path

import pytest

from car_control_modular.lateral_intent import LateralControlIntent, LateralIntentStore
from car_control_modular.turn_response_assist import TurnResponseAssist
from test_cap676_turn_continuity import runtime
from test_visible_wheel_continuity import feedback


def intent(now=10., cap=1663, sign=-1, **kwargs):
    value = LateralControlIntent(1,1,10,now,now+.2,.5+sign*.15,0.,sign*40.,
        'forward',21,42,sign*10,15.,.9,'reliable','visual',
        capture_frame_id=cap,capture_timestamp=now-.13,
        response_boost_allowed=True,visual_error_deg=sign*9.9)
    return replace(value, **kwargs)


@pytest.mark.parametrize('sign', [-1,1])
def test_cap1663_opposite_residual_gets_bounded_boost(sign):
    a = TurnResponseAssist()
    measured = (69,53) if sign < 0 else (53,69)
    b,y,phase = a.adjust(42,sign*10,intent(sign=sign),feedback(10,*measured),10,1)
    assert (b,y,phase) == (42,sign*14,'turn_build_boost')
    assert a.adjust(34,sign*15,intent(10.1,1666,sign),feedback(10.1,*measured),10.1,1)[1] == sign*20


def test_feedback_establishment_removes_extra_not_normal_yaw():
    a = TurnResponseAssist()
    a.adjust(42,-10,intent(),feedback(10,69,53),10,1)
    for t,cap in [(10.05,1664),(10.10,1665)]:
        assert a.adjust(42,-10,intent(t,cap),feedback(t,40,51),t,1)[1:] == (-10,'turn_established')
    # A jittering later sample/new frame cannot retrigger the same episode.
    assert a.adjust(42,-10,intent(10.15,1666),feedback(10.15,51,50),10.15,1)[1:] == (-10,'episode_complete')


def test_duplicate_feedback_cannot_confirm_twice():
    a = TurnResponseAssist()
    a.adjust(42,-10,intent(),feedback(10,69,53),10,1)
    f = feedback(10.05,40,51)
    for t in [10.05,10.06,10.07]:
        assert a.adjust(42,-10,intent(t,1664),f,t,1)[1] == -10
    assert a.established_count == 1


def test_new_frames_cannot_extend_fixed_boost_deadline():
    a = TurnResponseAssist()
    for i in range(8):
        now=10+i*.05
        result=a.adjust(42,-15,intent(now,1663+i),feedback(now,69,53),now,1)
        assert result[1] == (-20 if i<7 else -15)
        assert a.started == 10
    assert a.adjust(42,-15,intent(10.5,1675),feedback(10.5,69,53),10.5,1)[1] == -15


@pytest.mark.parametrize('change', ['approach','park','hold','near','limited','search','reverse',
    'denied','stale_image','expired','stale_feedback','missing_feedback','reverse_feedback','small_error','policy_cap'])
def test_vetoes_do_not_amplify(change):
    i=intent();f=feedback(10,69,53)
    if change=='approach':i=replace(i,target_image_rate_dps=40.)
    if change=='park':i=replace(i,park_requested=True)
    if change=='hold':i=replace(i,hold_zero=True)
    if change=='near':i=replace(i,near_distance_mode=True)
    if change=='limited':i=replace(i,bbox_quality='limited')
    if change=='search':i=replace(i,mode='yaw_only')
    if change=='reverse':i=replace(i,mode='reverse')
    if change=='denied':i=replace(i,response_boost_allowed=False)
    if change=='stale_image':i=replace(i,capture_timestamp=9.7)
    if change=='expired':i=replace(i,valid_until=9.9)
    if change=='small_error':i=replace(i,visual_error_deg=-5.)
    if change=='policy_cap':i=replace(i,correction_limit_rpm=10.)
    if change=='stale_feedback':f.timestamp=9.8
    if change=='missing_feedback':f=None
    if change=='reverse_feedback':f.left_forward_rpm=-4
    assert TurnResponseAssist().adjust(42,-10,i,f,10,1)[:2]==(42,-10)


def test_zero_yaw_same_image_cannot_rearm():
    a=TurnResponseAssist();i=intent()
    a.adjust(42,-10,i,feedback(10,69,53),10,1)
    a.adjust(42,0,i,feedback(10.01,69,53),10.01,1)
    assert a.adjust(42,-10,i,feedback(10.02,69,53),10.02,1)[1]==-10
    assert a.adjust(42,-10,intent(10.03,1664),feedback(10.03,69,53),10.03,1)[1]==-14


def test_boost_never_creates_reversal_or_raises_translation():
    for base in [0,8,10,12,15,20,100]:
        b,y,_=TurnResponseAssist().adjust(base,-15,intent(),feedback(10,69,53),10,1)
        assert b==base
        if base>15: assert b+y>=0 and b-y>=0
        else: assert y==-15


def test_center_residual_does_not_cap_authorized_translation():
    a=TurnResponseAssist()
    i=intent(visual_error_deg=0.,hold_zero=True,park_requested=True,response_boost_allowed=False)
    assert a.adjust(86,0,i,feedback(10,47,35),10,1)==(86,0,'zero_yaw')
    assert a.adjust(20,0,i,feedback(10,47,35),10,1)[0]==20  # no raising braking request
    assert a.adjust(86,0,i,feedback(10,47,45),10,1)[0]==86
    assert a.adjust(0,0,i,feedback(10,47,35),10,1)[0]==0


def prepare(monkeypatch):
    r,o,d,s,clock=runtime(monkeypatch)
    r.config.follow_turn_response_assist_enable=True
    o._lateral_intent_store=LateralIntentStore()
    o._lateral_intent_store.publish(intent())
    o._fresh_depth_linear_snapshot=lambda uid,now=None: ('forward',21)
    r.get_steering_feedback=lambda:feedback(clock[0],69,53)
    return r,o,d,s,clock


def test_real_writer_normal_boost_established_and_expiry(monkeypatch,caplog):
    r,o,d,_,clock=prepare(monkeypatch)
    with caplog.at_level('INFO'):r._send_follow_wheel_targets(32,-52,'FOLLOW20')
    assert d.pairs[-1]==(28,-56)
    assert 'response_phase=turn_build_boost' in caplog.text
    clock[0]=10.05
    r.get_steering_feedback=lambda:feedback(clock[0],40,51)
    r._send_follow_wheel_targets(32,-52,'FOLLOW20')
    assert d.pairs[-1]==(32,-52)
    o._fresh_depth_linear_snapshot=lambda uid,now=None:None
    o._has_fresh_lateral_yaw=lambda uid:False
    r._send_follow_wheel_targets(32,-52,'FOLLOW20')
    assert d.pairs[-1]==(0,0)


def test_real_writer_center_residual_and_no_normal_inserted(monkeypatch):
    r,o,d,_,clock=prepare(monkeypatch)
    o._fresh_depth_linear_snapshot=lambda uid,now=None:('forward',43)
    o._lateral_intent_store.publish(intent(visual_error_deg=0.,hold_zero=True))
    o._has_fresh_lateral_yaw=lambda uid:False
    r.get_steering_feedback=lambda:feedback(clock[0],47,35)
    r._send_follow_wheel_targets(86,-86,'FOLLOW20')
    assert d.pairs[-1]==(86,-86) and not d.stops


def test_boost_still_passes_guard_and_intent_replacement_veto(monkeypatch):
    r,o,d,_,clock=prepare(monkeypatch)
    original=r._visible_wheel_guard.limit
    def replacement(*args,**kwargs):
        o._lateral_intent_store.publish(intent(10,1664))
        return original(*args,**kwargs)
    r._visible_wheel_guard.limit=replacement
    assert r._send_follow_wheel_targets(32,-52,'FOLLOW20') is False
    assert not d.pairs


def test_real_config_binding(monkeypatch):
    from car_control_modular.config_loader import load_config_to_env
    import os
    root=Path(__file__).resolve().parents[2]
    monkeypatch.setattr(os,'environ',{})
    load_config_to_env(str(root/'car_control_modular/config/reid_runtime.ini'))
    assert os.environ['FOLLOW_TURN_RESPONSE_ASSIST_ENABLE']=='1'
    tree=ast.parse((root/'request_0513_modular.py').read_text())
    values=[k.value for n in ast.walk(tree) if isinstance(n,ast.Call)
        for k in n.keywords if k.arg=='follow_turn_response_assist_enable']
    assert len(values)==1
    assert eval(compile(ast.Expression(values[0]),'binding','eval'),{'os':os}) is True


def test_fast_loop_taper_overrides_original_visual_boost_permission(monkeypatch):
    r,o,d,_,clock=prepare(monkeypatch)
    i=o._lateral_intent_store.snapshot()
    o._lateral_turn_response_policy=(i.sequence, False)
    r._send_follow_wheel_targets(32,-52,'FOLLOW20')
    assert d.pairs[-1]==(32,-52)


def test_braking_policy_change_during_guard_vetoes_boost(monkeypatch):
    r,o,d,_,clock=prepare(monkeypatch)
    original=r._visible_wheel_guard.limit
    def brake(*args,**kwargs):
        o._lateral_turn_response_policy=(o._lateral_intent_store.snapshot().sequence,False)
        return original(*args,**kwargs)
    r._visible_wheel_guard.limit=brake
    assert r._send_follow_wheel_targets(32,-52,'FOLLOW20') is False
    assert not d.pairs


@pytest.mark.parametrize('exit_kind', ['expired','danger','park'])
def test_periodic_boost_cannot_delay_safety_or_park(monkeypatch, exit_kind):
    from test_follow_wheel_periodic import setup_periodic
    from test_near_yaw_park_execution import request_park
    r,o,d,_,clock,state=setup_periodic(monkeypatch)
    r.config.follow_turn_response_assist_enable=True
    o._lateral_intent_store=LateralIntentStore()
    o._lateral_intent_store.publish(intent())
    state[:]=[42.,-10.,10.20,10.20]
    r.get_steering_feedback=lambda:feedback(clock[0],69,53)
    r._service_follow_wheels()
    assert d.pairs[-1]==(28,-56)
    before=list(d.pairs)
    clock[0]=10.005
    if exit_kind=='expired':
        state[2:]=[10.004,10.004]
        r._service_follow_wheels()
        assert d.pairs[-1]==(0,0)
    elif exit_kind=='danger':
        calls=[]
        r.hard_stop_check=lambda action:True
        r.send_stop_with_brake_hold=calls.append
        r._service_follow_wheels()
        assert calls==['follow20_hard_stop'] and d.pairs==before
    else:
        request_park(o,clock)
        r._service_follow_wheels()
        assert d.stops
        held=list(d.pairs)
        clock[0]=10.09
        r._service_follow_wheels()
        assert d.pairs==held


def test_boost_respects_outer_wheel_cap(monkeypatch):
    r,o,d,_,_=prepare(monkeypatch)
    o._fresh_depth_linear_snapshot=lambda uid,now=None:('forward',100)
    r._send_follow_wheel_targets(170,-200,'FOLLOW20',max_target_override=200)
    assert d.pairs[-1]==(170,-200)  # Preserve base 185; shrink enhanced yaw to 15.


def test_uid_change_starts_independent_episode_and_wrong_uid_cannot_boost():
    a=TurnResponseAssist()
    a.adjust(42,-10,intent(),feedback(10,69,53),10,1)
    assert a.adjust(42,-10,intent(10.1,1664),feedback(10.1,69,53),10.1,2)[1]==-10
    assert a.adjust(42,-10,intent(10.2,1665,target_id=2),feedback(10.2,69,53),10.2,2)[1]==-14
