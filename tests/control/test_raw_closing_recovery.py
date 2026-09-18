"""Raw depth, missing matching speed and live-grant recovery; no hardware."""
from dataclasses import replace
import pytest

from car_control_modular.longitudinal_approach import ApproachConfig, RawDepthClosingWindow, approach_reference
from car_control_modular.controllers import FollowSafetyController
from test_distance_tracking_response import setup, decide
from test_longitudinal_approach import enabled_controller, pid
from test_scheduling_gap_evidence import missing


@pytest.mark.parametrize('old_raw,new_raw,dt,old_used,new_used,expected', [
    (1.779248407643312,1.7627692307692306,.032445041,1.8276512896094326,1.779248407643312,-.50791),
    (2.057,2.052,.032049552,2.1258,2.057,-.1560),
    (2.658022127659575,2.6333031847133754,.037722907,2.89230264132313,2.658022127659575,-.6553),
])
def test_real_log_median_steps_do_not_become_braking_velocity(old_raw,new_raw,dt,old_used,new_used,expected):
    w=RawDepthClosingWindow()
    assert w.update(uid=1,stamp=100,raw=old_raw) is None
    rate=w.update(uid=1,stamp=100+dt,raw=new_raw)
    assert rate==pytest.approx(expected,abs=.0001)
    c=pid()
    c.update(old_used,1.5,now=100,braking_range_rate_m_s=0)
    r=c.update(new_used,1.5,now=100+dt,braking_range_rate_m_s=rate,raw_closure_valid=True)
    assert r.approach_closing_m_s==pytest.approx(-expected,abs=.0001)
    assert r.output_rpm>0
    assert r.approach_braking_distance_m<1.


def test_regression_uses_depth_time_and_integrated_rotation():
    w=RawDepthClosingWindow()
    for i in range(5):
        # .2m/s camera-rotation contribution + -.5m/s true closure.
        rate=w.update(uid=1,stamp=100+i*.033,raw=2.-.3*i*.033,rotation=.2)
    assert rate==pytest.approx(-.5)
    assert len(w.samples)==5
    assert w.span==pytest.approx(.132)


@pytest.mark.parametrize('case',['uid','expired','rotation_mode','nan','jump'])
def test_invalid_or_discontinuous_samples_do_not_get_far_budget(case):
    w=RawDepthClosingWindow()
    w.update(uid=1,stamp=100,raw=2.)
    w.update(uid=1,stamp=100.05,raw=2.01)
    args=dict(uid=1,stamp=100.1,raw=2.02)
    if case=='uid': args['uid']=2
    if case=='expired': args['stamp']=100.3
    if case=='rotation_mode': args['rotation']=.1
    if case=='nan': args['raw']=float('nan')
    if case=='jump': args['raw']=4.
    assert w.update(**args) is None


def test_duplicate_and_historical_samples_never_retime_or_accumulate():
    w=RawDepthClosingWindow()
    w.update(uid=1,stamp=100,raw=2.)
    r=w.update(uid=1,stamp=100.05,raw=1.99)
    before=list(w.samples)
    for ts in (100,100.05,99):
        assert w.update(uid=1,stamp=ts,raw=2.)==r
        assert w.samples==before


def reference(error,rate=0.,base=None,valid=True):
    return approach_reference(ApproachConfig(no_matching_max_rpm=60),error_m=error,
        deadband_m=.03,tracking_base_rpm=base,range_rate_m_s=rate,
        raw_closure_valid=valid,max_output_rpm=200,measurement_age_sec=.05)


def test_far_no_matching_sixty_is_not_a_near_floor_or_a_human_speed():
    assert reference(1.5).output_rpm==pytest.approx(60)
    assert reference(1.5,valid=False).output_rpm<45
    assert reference(0).output_rpm==0
    assert reference(-.1).output_rpm==0
    assert reference(.06).output_rpm<3
    assert reference(.2,rate=-1.).output_rpm==0
    assert reference(1.5,base=40).output_rpm==pytest.approx(40+.6*60/.816814)


def test_config_rejects_unsafe_nonfinite_budget():
    for value in (-1,61,float('nan'),float('inf')):
        with pytest.raises(ValueError): ApproachConfig(no_matching_max_rpm=value)


def test_real_controller_braking_uses_raw_not_fused_distance(setup):
    clock,c,frame=enabled_controller(setup)
    # Both consumers receive the SAME accepted raw sample, despite fused lag.
    first = frame(2.1258,rpm=20)
    first = replace(first,distance_state=replace(first.distance_state,raw_distance_m=2.057))
    decide(c,first)
    clock.now+=.032049552
    f=frame(2.057,rpm=20)
    f=replace(f,distance_state=replace(f.distance_state,raw_distance_m=2.052))
    decide(c,f)
    assert c._braking_rate_source=='raw_depth_window'
    assert c.last_distance_pid_result.approach_closing_m_s==pytest.approx(.1560,abs=.0001)
    assert c.last_distance_pid_result.output_rpm>10


def test_real_no_matching_far_request_reaches_sixty_without_fake_ff(setup):
    clock,old,frame=setup
    c=FollowSafetyController(replace(old.cfg,distance_approach_enable=True,
        distance_approach_no_matching_max_rpm=60,forward_max_rpm=200,
        distance_feedforward_wheel_circumference_m=.816814,
        distance_pid_output_rise_rpm_per_sec=240))
    c.active_target_id=1;c._has_seen_person=True
    # Static range + stationary wheels -> no_forward_motion, NOT unavailable depth.
    for _ in range(8):
        decide(c,frame(3.,rpm=0));clock.now+=.033
    r=c.last_distance_pid_result
    assert r.tracking_base_rpm==0
    assert r.output_rpm==60


def make_gap(setup):
    clock,old,frame=setup
    c=FollowSafetyController(replace(old.cfg,distance_approach_enable=True,
        depth_measured_recovery_enable=True,distance_pid_output_rise_rpm_per_sec=240))
    c.active_target_id=1;c._has_seen_person=True
    c._depth_recovery_anchor=(1,100.,2.2)
    c._depth_last_approved_forward_rpm=76.
    c._depth_quality_degraded=False
    clock.now=100.02
    c._note_depth_quality_failure(missing(frame),clock.now)
    clock.now=100.05
    return clock,c,frame


def test_live_authorization_normal_acceleration_does_not_restart_recovery(setup,caplog):
    clock,c,frame=make_gap(setup)
    c._live_longitudinal_authority_reader=lambda uid: ('forward',76,1,100.)
    # Real CAP144 failure: wheels23.5, previous command76, new request77.
    output=c._limit_depth_quality_forward_percent(frame(2.21,rpm=23.5),77,clock.now)
    assert output==77
    assert c._depth_schedule_recovery is None
    assert c._depth_recovery_started_at is None
    assert 'normal_live_continuation=True' in caplog.text


@pytest.mark.parametrize('case',['revoked','expired','uid','reverse','zero','stopped_wheels'])
def test_true_stop_or_expiry_cannot_use_normal_continuation(setup,case):
    clock,c,frame=make_gap(setup)
    grant=('forward',76,1,100.)
    if case=='revoked': grant=None
    if case=='expired': grant=('forward',76,1,99.8)
    if case=='uid': grant=('forward',76,2,100.)
    if case=='reverse': grant=('backward',76,1,100.)
    if case=='zero': grant=('forward',0,1,100.)
    c._live_longitudinal_authority_reader=lambda uid: grant
    output=c._limit_depth_quality_forward_percent(frame(2.21,rpm=0 if case=='stopped_wheels' else 23.5),77,clock.now)
    assert output<60
    assert c._depth_schedule_recovery is not None


def test_repeated_attempt_failures_with_live_grants_use_one_normal_slew(setup):
    clock,c,frame=make_gap(setup)
    grant=('forward',76,1,100.)
    c._live_longitudinal_authority_reader=lambda uid: grant
    for _ in range(4):
        output=c._limit_depth_quality_forward_percent(frame(2.21,rpm=30),100,clock.now)
        assert grant[1] <= output <= grant[1]+13
        grant=('forward',output,1,clock.now)
        clock.now+=.02
        c._note_depth_quality_failure(missing(frame),clock.now)
        clock.now+=.03
        assert c._depth_schedule_recovery is None


def test_unverified_turn_resets_raw_window_and_blocks_sixty_branch(setup):
    clock,c,frame=enabled_controller(setup)
    decide(c,frame(3.,rpm=30));clock.now+=.05
    decide(c,frame(3.,rpm=30))
    assert c._braking_rate_source=='raw_depth_window'
    clock.now+=.05
    decide(c,frame(3.,rpm=30,yaw=20))
    assert c._braking_rate_source=='encoder_fallback'
    assert c._raw_closing_window.samples==[]
    clock.now+=.05
    decide(c,frame(3.,rpm=30))
    assert c._braking_rate_source=='encoder_fallback' # no difference across the turn


def test_runtime_and_ini_budget_do_not_make_sixty_a_floor(monkeypatch):
    import os
    from pathlib import Path
    import request_0513_modular as rt
    from car_control_modular.config_loader import load_config_to_env
    from car_control_modular.control_types import ControlAction
    env={};monkeypatch.setattr(os,'environ',env)
    load_config_to_env(str(Path(__file__).resolve().parents[2]/'car_control_modular/config/reid_runtime.ini'))
    assert float(env['DISTANCE_APPROACH_NO_MATCHING_MAX_RPM'])==60
    assert float(env['ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC'])==.25
    assert rt.ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC==.18
    for name,value in dict(DISTANCE_APPROACH_ENABLE=True,DISTANCE_APPROACH_NO_MATCHING_MAX_RPM=60,
            FORWARD_MAX_RPM=200,FOLLOW_ROTATION_ONLY=False,ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT=100).items():
        monkeypatch.setattr(rt,name,value)
    for requested in (0,5,30):
        a=rt.PersonTracker._cap_depth_longitudinal_actions(
            [ControlAction.forward(requested,'test')],distance_m=3.,distance_control_percent=requested)[0]
        assert a.speed_percent==requested
