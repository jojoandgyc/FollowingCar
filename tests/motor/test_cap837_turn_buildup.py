"""CAP722/837 magnitudes; pure policy and real writer with fake serial only."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from functools import partial
from car_control_modular.turn_buildup import TurnBuildup as ProductionTurnBuildup

# Historical 100ms response fixture; production 500ms timing is separately
# covered by test_cap1252_motor_latency with the real writer default.
TurnBuildup = partial(ProductionTurnBuildup, response_delay_sec=.1)
from test_cap1663_turn_response import intent, prepare
from test_visible_wheel_continuity import feedback


def image_intent(t=10., cap=832, sign=1, **changes):
    return intent(t, cap, sign, correction_limit_rpm=10., visual_error_deg=sign*25.,
                  target_image_rate_dps=None, **changes)


def step(a, t, cap, pair=(19,19), base=82, yaw=10, i=None):
    return a.adjust(base, yaw, i or image_intent(t,cap,1 if yaw>0 else -1),
                    feedback(t,*pair), t, 1, limit=10.)


def armed():
    a=TurnBuildup()
    a.note_sent(1,(92,72),9.86,0.)
    assert step(a,10.,832)==(82,10,'buildup_wait_two_feedback')
    return a


@pytest.mark.parametrize('sign',[-1,1])
def test_two_new_feedback_build_turn_inside_existing_twenty_rpm_limit(sign):
    a=TurnBuildup();a.note_sent(1,(82+sign*10,82-sign*10),9.86,0.)
    assert step(a,10.,832,yaw=sign*10)[0]==82
    b,y,p=step(a,10.05,834,yaw=sign*10)
    assert (b,y,p)==(82,sign*10,'buildup_yaw_only')
    assert min(b-y,b+y)>=0 and abs(2*y)<=20


def test_duplicate_or_out_of_order_feedback_cannot_start():
    a=armed()
    for t in [10.,10.,9.99]:
        assert a.adjust(82,10,image_intent(10.,834),feedback(t,19,19),10.01,1,limit=10.)[0]==82
    assert a.started is None


def test_feedback_before_actual_command_and_missing_writes_not_evidence():
    for sent in [None,10.02]:
        a=TurnBuildup()
        if sent is not None:a.note_sent(1,(92,72),sent,0.)
        for t in [10.,10.01]:assert step(a,t,832)[0]==82
        assert a.started is None


def test_fixed_deadline_not_extended_by_new_frames():
    a=armed();b,y,_=step(a,10.05,834)
    start=a.started
    a.note_sent(1,(b+y,b-y),10.05,9.86)
    for j in range(1,8):
        t=10.05+j*.05
        b,y,p=step(a,t,834+j)
        assert b==82
        if t-start < .35:assert p=='buildup_yaw_only'
        else:assert p=='buildup_timeout'
        a.note_sent(1,(b+y,b-y),t,t-.05)
        assert a.started==start
    assert step(a,10.5,850)[0]==82


def test_turn_established_releases_and_jitter_does_not_rearm():
    a=armed();assert step(a,10.05,834)[2]=='buildup_yaw_only'
    a.note_sent(1,(92,72),10.05,9.86)
    assert step(a,10.1,835,pair=(35,20))[2]=='buildup_established'
    assert step(a,10.15,837)[0]==82


@pytest.mark.parametrize('bad',['near','park','hold','search','weak','denied','expired',
    'stale','approaching','oldcap','smallerror','wronguid'])
def test_visual_and_brake_vetoes(bad):
    a=armed();i=image_intent(10.05,834)
    changes={'near':dict(near_distance_mode=True),'park':dict(park_requested=True),
        'hold':dict(hold_zero=True),'search':dict(mode='yaw_only'),'weak':dict(bbox_quality='limited'),
        'denied':dict(response_boost_allowed=False),'expired':dict(valid_until=10.),
        'stale':dict(capture_timestamp=9.7),'approaching':dict(target_image_rate_dps=-100.),
        'oldcap':dict(capture_frame_id=800),'smallerror':dict(visual_error_deg=4.),
        'wronguid':dict(target_id=2)}
    assert step(a,10.05,834,i=replace(i,**changes[bad]))[:2]==(82,10)


@pytest.mark.parametrize('pair',[(90,80),(-1,20),(20,-1)])
def test_braking_or_reversing_feedback_cannot_raise_demand(pair):
    a=armed();assert step(a,10.05,834,pair=pair)[:2]==(82,10)


@pytest.mark.parametrize('kind',['missing','stale','nan','untrusted'])
def test_bad_encoder_evidence(kind):
    a=armed();f=feedback(10.05,19,19)
    if kind=='missing':f=None
    if kind=='stale':f.timestamp=9.8
    if kind=='nan':f.right_forward_rpm=float('nan')
    if kind=='untrusted':f.trustworthy=False
    assert a.adjust(82,10,image_intent(10.05,834),f,10.05,1,limit=10.)[:2]==(82,10)


def test_zero_requires_new_image_and_new_sent_feedback_proof():
    a=armed();assert step(a,10.05,834)[2]=='buildup_yaw_only'
    a.note_sent(1,(0,0),10.06,9.86)
    assert step(a,10.10,834)[0]==82
    assert step(a,10.15,835)[0]==82
    a.note_sent(1,(92,72),10.15,10.06)
    assert step(a,10.26,836)[0]==82
    assert step(a,10.31,837)[2]=='buildup_yaw_only'


def setup(monkeypatch):
    r,o,d,s,clock=prepare(monkeypatch)
    r._turn_buildup = TurnBuildup()
    o._follow_controller.cfg=SimpleNamespace(visible_steering_pid_image_error_only=True,
        visible_steering_pid_max_correction_rpm=10.,near_distance_rotation_only_max_rpm=7.)
    o._fresh_depth_linear_snapshot=lambda uid,now=None:('forward',41)
    o._lateral_intent_store.publish(image_intent())
    r.get_steering_feedback=lambda:feedback(clock[0],19,19)
    return r,o,d,s,clock


def run_until_buildup(r,o,clock):
    for t in [10.,10.11,10.16]:
        clock[0]=t
        o._lateral_intent_store.publish(image_intent(t,int(832+(t-10)*20)))
        r._send_follow_wheel_targets(92,-72,'FOLLOW20')


def test_real_writer_preserves_base_and_logs_zero_base_loss(monkeypatch,caplog):
    r,o,d,_,clock=setup(monkeypatch)
    with caplog.at_level('INFO'):run_until_buildup(r,o,clock)
    assert d.pairs==[(92,-72)]*3
    assert not d.stops
    assert 'response_phase=buildup_yaw_only' in caplog.text
    assert 'common_accel_removed_rpm=0.0' in caplog.text
    assert 'execution_base_loss_rpm=0.0' in caplog.text


@pytest.mark.parametrize('kind',['identity','depth','yaw','brake_policy'])
def test_guard_time_revocations_still_veto_new_adjusted_packet(monkeypatch,kind):
    r,o,d,_,clock=setup(monkeypatch)
    # Leave yaw headroom so the additive policy actually changes the packet;
    # at yaw=10 the new policy correctly leaves the original pair unchanged.
    for t in [10.,10.11]:
        clock[0]=t;o._lateral_intent_store.publish(image_intent(t,832))
        r._send_follow_wheel_targets(88,-76,'FOLLOW20')
    previous=r._visible_wheel_guard.limit
    def revoke(*args,**kwargs):
        result=previous(*args,**kwargs)
        if kind=='identity':o._follow_controller.active_target_id=2
        if kind=='depth':o._fresh_depth_linear_snapshot=lambda uid,now=None:None
        if kind=='yaw':o._has_fresh_lateral_yaw=lambda uid:False
        if kind=='brake_policy':o._lateral_turn_response_policy=(o._lateral_intent_store.snapshot().sequence,False)
        return result
    r._visible_wheel_guard.limit=revoke
    clock[0]=10.16;o._lateral_intent_store.publish(image_intent(10.16,834))
    n=len(d.pairs);r._send_follow_wheel_targets(88,-76,'FOLLOW20')
    assert all(p==(0,0) for p in d.pairs[n:])


def test_unconfirmed_identity_does_not_inherit_visible_turn_path(monkeypatch):
    r,o,d,s,clock=setup(monkeypatch);run_until_buildup(r,o,clock)
    o._vision_control_state='lost_confirming'
    o._fresh_depth_linear_snapshot=lambda uid,now=None:None
    assert not r._visible_wheel_control_active()
    assert r.needs_transition_stop(s.steer_right,s.rotate_right)
    n=len(d.pairs)
    r._send_follow_wheel_targets(92,-72,'FOLLOW20',visible_required=True)
    assert len(d.pairs)==n


@pytest.mark.parametrize('change',['time','danger'])
def test_last_moment_expiry_and_danger_preempt_adjustment(monkeypatch,change):
    r,o,d,_,clock=setup(monkeypatch)
    for t in [10.,10.11]:
        clock[0]=t;o._lateral_intent_store.publish(image_intent(t,832))
        r._send_follow_wheel_targets(88,-76,'FOLLOW20')
    original=r._visible_wheel_guard.limit
    def wait(*args,**kwargs):
        result=original(*args,**kwargs)
        if change=='time':clock[0]+=.12
        else:r.hard_stop_check=lambda action:True
        return result
    r._visible_wheel_guard.limit=wait
    clock[0]=10.16;o._lateral_intent_store.publish(image_intent(10.16,834))
    n=len(d.pairs)
    assert r._send_follow_wheel_targets(88,-76,'FOLLOW20') is False
    assert len(d.pairs)==n
    if change=='danger':assert d.stops


def test_uid_or_writer_reset_discards_pre_reset_proof():
    a=armed();a.sync_writer(0.)
    assert step(a,10.05,834)[0]==82
    a=armed()
    i=replace(image_intent(10.05,834),target_id=2)
    assert a.adjust(82,10,i,feedback(10.05,19,19),10.05,2,limit=10.)[0]==82


def test_main_program_log_arguments_and_units():
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).resolve().parents[2]/'request_0513_modular.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and n.args
           and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)
           and n.args[0].value.startswith('视觉转向PID:')]
    assert len(calls)==1
    call=calls[0]
    rendered=call.args[0].value % tuple(0 for _ in call.args[1:])
    assert '单轮修正' in rendered and '请求双轮差' in rendered
    assert isinstance(call.args[-1],ast.BinOp) and call.args[-1].left.value==2


def test_search_writer_invalidates_visible_command_evidence(monkeypatch):
    r,o,d,s,clock=setup(monkeypatch);run_until_buildup(r,o,clock)
    assert r._turn_buildup.history
    o._vision_control_state='lost_confirming'
    o.current_command=s.stop
    r.send_robot_command(s.stop)
    assert not r._turn_buildup.history


@pytest.mark.parametrize('yaw,limit',[(3,5),(5,5),(5,7),(7,7),(7,10),(10,10)])
def test_lower_policy_caps_and_headroom(yaw,limit):
    a=TurnBuildup();a.note_sent(1,(40+yaw,40-yaw),9.86,0.)
    for t in (10.,10.05):
        b,y,phase=a.adjust(40,yaw,image_intent(t,834),feedback(t,10,10),t,1,limit=limit)
    assert b==40 and 0<y<=limit and b>=abs(y)
    assert phase=='buildup_yaw_only'
