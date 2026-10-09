"""Soft image-margin allocation, not a general forward speed limiter."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.view_retention import ViewRetention
from test_cap837_turn_buildup import image_intent, setup as base_setup


def setup(monkeypatch):
    r,o,d,s,clock=base_setup(monkeypatch)
    o._fresh_depth_linear_snapshot=lambda uid,now=None:('forward',41,uid,clock[0]-.02)
    o._follow_controller.cfg.depth_longitudinal_sample_max_age_sec=.25
    return r,o,d,s,clock


def intent(t, cap, error=28., sign=1, **kw):
    return replace(image_intent(t, cap, sign), image_error_only=True,
        **dict(dict(visual_error_deg=error*sign, target_image_rate_dps=4.*sign), **kw))


def feedback(t, l=20., r=20., **kw):
    return SimpleNamespace(**dict(dict(timestamp=t, left_forward_rpm=l, right_forward_rpm=r,
        raw_yaw_rate_right_dps=(l-r)*1.57, yaw_rate_right_dps=(l-r)*1.57, trustworthy=True,
        left_read_started=t-.02, left_read_finished=t-.01,
        right_read_started=t-.01, right_read_finished=t), **kw))


def adjust(p, t, cap, error=28., base=86., yaw=10., i=None, f=None, permitted=True):
    return p.adjust(base, yaw, i or intent(t,cap,error,1 if yaw>0 else -1),
        f or feedback(t-.002), t, 1, permitted=permitted, hfov=66.)


def armed(sign=1):
    p=ViewRetention()
    p.note_sent(1,(86+10*sign,86-10*sign),9.85,0)
    for j,t in enumerate([10.,10.05]):
        assert adjust(p,t,297+j,error=28+j*.3,yaw=10*sign)[0]==86
        p.note_sent(1,(86+10*sign,86-10*sign),t,9.85 if j==0 else 10.)
    return p


@pytest.mark.parametrize('sign',[-1,1])
def test_three_captures_two_postwrite_feedback_and_twenty_diff(sign):
    p=armed(sign)
    b,y,reason=adjust(p,10.10,302,error=28.8,yaw=10*sign)
    assert (b,y,reason)==(32.,10*sign,'view_common_acceleration_held')
    assert min(b+y,b-y)>=0 and abs(2*y)==20
    assert b+y<=86+10*sign and b-y<=86-10*sign
    assert p.deadline==pytest.approx(10.45)


@pytest.mark.parametrize('kind',['near','park','hold','weak','yawonly','expired','stale',
    'unknown_rate','inward','small_error','not_margin','no_permission','countersteer',
    'lead','crop_rate','image_off'])
def test_nonrisk_or_braking_evidence_does_not_reduce_forward(kind):
    p=armed(); i=intent(10.1,302,28.8)
    changes={'near':{'near_distance_mode':True},'park':{'park_requested':True},
        'hold':{'hold_zero':True},'weak':{'bbox_quality':'limited'},'yawonly':{'mode':'yaw_only'},
        'expired':{'valid_until':10.},'stale':{'capture_timestamp':9.7},
        'unknown_rate':{'target_image_rate_dps':None},'inward':{'target_image_rate_dps':-4},
        'small_error':{'visual_error_deg':10},'not_margin':{'visual_error_deg':17},
        'no_permission':{'response_boost_allowed':False},'countersteer':{'countersteer_rpm':-3},
        'lead':{'outward_lead':object()},'crop_rate':{'outward_continuity_rate_dps':4},
        'image_off':{'image_error_only':False}}
    assert adjust(p,10.1,302,i=replace(i,**changes[kind]))[:2]==(86.,10.)


@pytest.mark.parametrize('changes',[{'trustworthy':False},{'timestamp':9.8},
    {'left_read_started':None},{'right_read_started':9.84},{'raw_yaw_rate_right_dps':float('nan')},
    {'raw_yaw_rate_right_dps':40.,'yaw_rate_right_dps':40.},
    {'left_forward_rpm':-1.},{'right_forward_rpm':100.}])
def test_feedback_safety_and_read_causality(changes):
    p=armed()
    assert adjust(p,10.1,302,error=28.8,f=feedback(10.098,**changes))[:2]==(86.,10.)


def test_feedback_pair_straddling_new_write_is_not_causal():
    p=armed();p.note_sent(1,(96,76),10.09,10.05)
    assert adjust(p,10.1,302,error=28.8)[2]=='view_read_straddled_write'


def test_duplicate_capture_or_feedback_cannot_supply_missing_samples():
    p=ViewRetention();p.note_sent(1,(96,76),9.85,0.)
    for t in [10.,10.04,10.08]:
        assert adjust(p,t,297,i=intent(10.,297))[0]==86
    assert len(p.captures)==1 and p.deadline is None
    p=armed();p.feedback_stamp=10.098;p.lag_count=1
    for j in range(3):assert adjust(p,10.1+j*.01,302,error=28.8,f=feedback(10.098))[0]==86
    assert p.deadline is None


def test_fixed_episode_is_not_renewed_by_new_capture_or_zero():
    p=armed();assert adjust(p,10.1,302,error=28.8)[0]==32
    deadline=p.deadline
    p.note_sent(1,(42,22),10.1,10.05)
    for j in range(1,8):
        t=10.1+j*.05
        b,y,_=adjust(p,t,302+j,error=28.8+j*.1)
        p.note_sent(1,(b+y,b-y),t,t-.05)
        assert p.deadline==deadline
    assert adjust(p,10.5,320,error=30)[0]==86
    p.note_sent(1,(0,0),10.5,10.45)
    assert adjust(p,10.55,321,error=30.1)[0]==86


def test_no_rearm_until_new_center_observation():
    p=armed();adjust(p,10.1,302,error=28.8)
    adjust(p,10.46,303,error=29.)
    assert p.spent
    adjust(p,10.5,304,error=10.)
    assert not p.spent and p.deadline is None


def test_old_center_frame_cannot_rearm_after_candidate_history_was_cleared():
    p=armed();adjust(p,10.1,302,error=28.8)
    adjust(p,10.15,303,i=intent(10.15,303,target_image_rate_dps=None))
    assert not p.captures and p.spent
    assert adjust(p,10.2,302,i=intent(10.1,302,10.))[2]=='view_old_capture'
    assert p.spent and p.deadline==pytest.approx(10.45)


@pytest.mark.parametrize('kind',['writer','receipt','uid'])
def test_external_writer_and_uid_invalidate_evidence(kind):
    p=armed();adjust(p,10.1,302,error=28.8)
    if kind=='writer':p.sync_writer(0.,None)
    if kind=='receipt':p.sync_writer(10.05,object())
    if kind=='uid':p.note_sent(2,(96,76),10.12,0.)
    assert adjust(p,10.15,303,error=29.)[0]==86


def test_real_logged_cap297_300_302_inputs_establish_edge_episode():
    p=ViewRetention()
    # run_20260929_234529: actual dispatch rows 5215,5229,5278,5335.
    # No intermediate writes or feedback are synthesized. "now" is recovered
    # from the recorded feedback timestamp + pre-write feedback age.
    rows=[(297,21729.155904129,28.34,3.29,(6,14),44,15.3,21729.387991,
           (21729.336704298,21729.342206451,21729.342207034,21729.36137984)),
          (297,21729.155904129,28.34,3.29,(10,13),44,15.8,21729.439740,
           (21729.388537694,21729.400324228,21729.400325103,21729.411038332)),
          (300,21729.317769195,29.02,4.22,(21,25),48,43.9,21729.580064,
           (21729.489195727,21729.503398659,21729.503399242,21729.51467654)),
          (302,21729.414874,29.36,3.52,(36,38),86,31.5,21729.668505,
           (21729.601928163,21729.608704866,21729.608705449,21729.614933833))]
    previous=21729.333991
    p.note_sent(1,(44,44),previous,0.)
    outputs=[]
    for cap,capture,error,rate,pair,base,age,sent,reads in rows:
        now=reads[-1]+age/1000.
        i=replace(intent(now,cap,error,target_image_rate_dps=rate),capture_timestamp=capture)
        result=adjust(p,now,cap,base=base,i=i,f=feedback(reads[-1],*pair,
            left_read_started=reads[0],left_read_finished=reads[1],
            right_read_started=reads[2],right_read_finished=reads[3]))
        outputs.append(result)
        p.note_sent(1,(base+10,base-10),sent,previous);previous=sent
    assert outputs[-1]==(50.,10.,'view_common_acceleration_held')
    assert outputs[0][0]==44 and outputs[2][0]==48


def test_writer_applies_reduction_and_preserves_direction(monkeypatch,caplog):
    r,o,d,_,clock=setup(monkeypatch)
    o._follow_controller.cfg.visible_steering_pid_camera_hfov_deg=66.
    cache=[None]
    r.get_steering_feedback=lambda:cache[0]
    for j,t in enumerate([10.,10.1,10.2,10.25]):
        clock[0]=t
        published=o._lateral_intent_store.publish(intent(t,297+j,28+j*.2))
        o._lateral_turn_response_policy=(published.sequence,True)
        cache[0]=feedback(t-.002)
        with caplog.at_level('INFO'):r._send_follow_wheel_targets(92,-72,'FOLLOW20')
    assert d.pairs[-1]==(42,-22)
    assert 'view_common_acceleration_held' in caplog.text
    assert all(0<=p[0]<=92 and 0<=-p[1]<=72 and p[0]+p[1]<=20 for p in d.pairs)


@pytest.mark.parametrize('kind',['deadline','feedback','depth','identity','yaw','policy','intent','danger'])
def test_writer_rechecks_after_blocking_guards(monkeypatch,kind):
    r,o,d,_,clock=setup(monkeypatch)
    o._follow_controller.cfg.visible_steering_pid_camera_hfov_deg=66.
    cache=[None];r.get_steering_feedback=lambda:cache[0]
    def publish(t,cap):
        clock[0]=t
        published=o._lateral_intent_store.publish(intent(t,cap,28+(cap-297)*.2))
        o._lateral_turn_response_policy=(published.sequence,True)
        cache[0]=feedback(t-.002)
    for j,t in enumerate([10.,10.1,10.2]):
        publish(t,297+j);r._send_follow_wheel_targets(92,-72,'FOLLOW20')
    original=r._visible_wheel_guard.limit
    def changed(*args,**kw):
        result=original(*args,**kw)
        if kind=='deadline':clock[0]+=.36
        if kind=='feedback':cache[0]=feedback(clock[0],30,30)
        if kind=='depth':o._fresh_depth_linear_snapshot=lambda uid,now=None:None
        if kind=='identity':o._follow_controller.active_target_id=2
        if kind=='yaw':o._has_fresh_lateral_yaw=lambda uid:False
        if kind=='policy':o._lateral_turn_response_policy=(o._lateral_intent_store.snapshot().sequence,False)
        if kind=='intent':o._lateral_intent_store.publish(intent(clock[0],300,28.6))
        if kind=='danger':r.hard_stop_check=lambda action:True
        return result
    r._visible_wheel_guard.limit=changed
    publish(10.25,300);n=len(d.pairs)
    r._send_follow_wheel_targets(92,-72,'FOLLOW20')
    assert all(pair==(0,0) for pair in d.pairs[n:])
    if kind=='danger':assert d.stops


@pytest.mark.parametrize('kind',['feedback','visual_ttl','depth_ttl','continuation_boundary','receipt'])
def test_last_depth_reader_cannot_hide_new_feedback_or_elapsed_time(monkeypatch,kind):
    r,o,d,_,clock=setup(monkeypatch)
    o._follow_controller.cfg.visible_steering_pid_camera_hfov_deg=66.
    cache=[None];r.get_steering_feedback=lambda:cache[0]
    for j,t in enumerate([10.,10.1,10.2]):
        clock[0]=t
        published=o._lateral_intent_store.publish(intent(t,297+j,28+j*.2))
        o._lateral_turn_response_policy=(published.sequence,True)
        cache[0]=feedback(t-.002)
        r._send_follow_wheel_targets(92,-72,'FOLLOW20')
    clock[0]=10.25
    published=o._lateral_intent_store.publish(intent(10.25,300,28.6))
    o._lateral_turn_response_policy=(published.sequence,True)
    cache[0]=feedback(10.248)
    original=o._fresh_depth_linear_snapshot
    calls=[]
    def blocked_reader(uid,now=None):
        value=original(uid,now=now)
        calls.append(now)
        if len(calls)==4:
            if kind=='feedback':cache[0]=feedback(clock[0],30,30)
            if kind=='receipt':r.backend.last_speed_receipt=object()
            if kind=='visual_ttl':clock[0]+=.26
            if kind=='depth_ttl':
                value=('forward',41,uid,clock[0]-.249)
                clock[0]+=.002
            if kind=='continuation_boundary':
                value=('forward',41,uid,clock[0]-.179)
                clock[0]+=.002
        return value
    o._fresh_depth_linear_snapshot=blocked_reader
    n=len(d.pairs)
    assert r._send_follow_wheel_targets(92,-72,'FOLLOW20') is False
    assert len(calls)==4  # no unbounded validation/read loop
    assert not d.pairs[n:]


@pytest.mark.parametrize('error,rate',[(7.25,6.50),(8.8,5.75),(9.64,4.13),
    (9.68,.46),(17.8,-5.78),(17.2,-3.50),(17.05,-2.57),(15.35,-8.54)])
def test_cap353_362_and_cap391_399_never_enter_edge_path(error,rate):
    p=armed()
    assert adjust(p,10.1,302,i=intent(10.1,302,error,target_image_rate_dps=rate))[0]==86
