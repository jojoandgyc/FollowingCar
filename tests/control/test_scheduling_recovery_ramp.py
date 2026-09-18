"""No hardware: a scheduling recovery cap is not a new depth authorization."""
from dataclasses import replace
import pytest

from test_distance_tracking_response import setup, decide
from test_scheduling_gap_evidence import recovery, missing


def begin(setup, rpm=40, gap=.30):
    return recovery(setup, physical_gap=gap, processing_gap=gap+.02, rpm=rpm)


def test_slowed_wheels_resume_with_acceleration_not_fixed_25(setup, caplog):
    clock, c, frame, current = begin(setup)
    assert c._limit_depth_quality_forward_percent(current, 70, clock.now) == 52
    assert c._depth_recovery_started_at is None
    assert c._depth_schedule_recovery is not None
    assert 'depth_scheduling_recovery' in caplog.text
    assert 'lost_rpm=18.00 recovery_policy=scheduling' in caplog.text
    clock.now += .05
    assert c._limit_depth_quality_forward_percent(frame(2.03, rpm=45), 70, clock.now) == 68
    assert c._depth_schedule_recovery[3] == 55  # Initial approval, not a ceiling.


def test_stopped_wheels_do_not_jump_to_old_command(setup):
    clock, c, frame, current = begin(setup, rpm=0)
    assert c._limit_depth_quality_forward_percent(current, 70, clock.now) == 12
    clock.now += .05
    assert 23 <= c._limit_depth_quality_forward_percent(frame(2.03, rpm=10, stamp=clock.now-.02), 70, clock.now) <= 24
    clock.now += .05
    assert 35 <= c._limit_depth_quality_forward_percent(frame(2.04, rpm=20, stamp=clock.now-.02), 70, clock.now) <= 36


def test_duplicate_sample_cannot_ramp_or_renew_deadline(setup):
    clock, c, frame, current = begin(setup)
    c._limit_depth_quality_forward_percent(current, 70, clock.now)
    original = c._depth_schedule_recovery
    for dt in (.01, .03, .07):
        assert c._limit_depth_quality_forward_percent(current, 80, clock.now+dt) == 52
        assert c._depth_schedule_recovery == original
    assert current.distance_state.sample_timestamp == 100.30


@pytest.mark.parametrize('case', ['near','closing','jump','hazard','obstacle','safety','brake',
    'latched','uid','persons','search','old_depth','future_depth','old_feedback','untrusted',
    'reverse','fast_wheel','yaw','nan','background','zero','long_gap'])
def test_ramp_rejection_keeps_strict_path(setup, case):
    clock, c, frame, f = begin(setup, gap=.51 if case=='long_gap' else .30)
    hint = c._depth_gap_resume_hint
    request = 70
    if case in ('near','closing','jump'):
        d={'near':1.75,'closing':1.94,'jump':2.5}[case]
        f=replace(f,distance_m=d,distance_state=replace(f.distance_state,raw_distance_m=d))
    elif case=='hazard': f=replace(f,hazard=replace(f.hazard,active=True))
    elif case=='obstacle': f=replace(f,obstacles=replace(f.obstacles,front=True))
    elif case=='safety': f=replace(f,distance_state=replace(f.distance_state,safety_distance_m=.4))
    elif case=='brake': f=replace(f,distance_state=replace(f.distance_state,brake_latched=True))
    elif case=='latched': c._target_stop_latched=True
    elif case=='uid': c.active_target_id=2
    elif case=='persons': f=replace(f,persons=[])
    elif case=='search': c.search_state='searching'
    elif case=='old_depth': f=replace(f,distance_state=replace(f.distance_state,sample_timestamp=clock.now-.181))
    elif case=='future_depth': f=replace(f,distance_state=replace(f.distance_state,sample_timestamp=clock.now+.01))
    elif case=='old_feedback': f=replace(f,steering_feedback=replace(f.steering_feedback,timestamp=clock.now-.151))
    elif case=='untrusted': f=replace(f,steering_feedback=replace(f.steering_feedback,trustworthy=False))
    elif case in ('reverse','fast_wheel','nan'):
        f=replace(f,steering_feedback=replace(f.steering_feedback,left_forward_rpm={'reverse':-1,'fast_wheel':101,'nan':float('nan')}[case]))
    elif case=='yaw': f=replace(f,steering_feedback=replace(f.steering_feedback,yaw_rate_right_dps=16))
    elif case=='background': f=replace(f,distance_state=replace(f.distance_state,source_detail='depth_far_background_guard'))
    elif case=='zero': request=0
    assert c._scheduling_recovery_cap(f, request, clock.now, hint) is None
    assert c._depth_schedule_recovery is None


def test_gap_reference_retained_but_expired_depth_cannot_drive(setup):
    clock, c, frame, f = begin(setup)
    c._note_depth_quality_failure(missing(frame), clock.now)
    assert c._depth_gap_resume_hint is not None  # Reference only, not a lease.
    result = decide(c, missing(frame))
    assert not any(a.kind in ('forward','steer_left','steer_right') and a.speed_percent > 0
                   for a in result.actions)


def test_non_scheduling_failure_discards_reference(setup):
    clock, c, frame, f = begin(setup)
    m=missing(frame)
    c._note_depth_quality_failure(replace(m,distance_state=replace(
        m.distance_state,source_detail='depth_invalid_pixels')),clock.now)
    assert c._depth_gap_resume_hint is None
    assert c._limit_depth_quality_forward_percent(f,70,clock.now)==25


def test_actual_approved_not_unlimited_pid_becomes_reference(setup):
    clock,c,frame,f=begin(setup)
    c._limit_depth_quality_forward_percent(f,40,clock.now)
    c._distance_pid._last_output_rpm=150
    clock.now+=.03
    c._note_depth_quality_failure(missing(frame),clock.now)
    assert c._depth_gap_resume_hint[3]==40


def test_200rpm_scale_is_not_percent(setup):
    clock,c,frame,f=begin(setup)
    c.cfg=replace(c.cfg,forward_max_rpm=200)
    assert c._limit_depth_quality_forward_percent(f,35,clock.now)==26
    assert c._depth_last_approved_forward_rpm==52


def test_runtime_reset_discards_ramp(setup):
    clock,c,frame,f=begin(setup)
    c._limit_depth_quality_forward_percent(f,70,clock.now)
    c._reset_longitudinal_motion()
    assert c._depth_schedule_recovery is None


def test_near_sample_immediately_exits_active_ramp(setup):
    clock,c,frame,f=begin(setup)
    c._limit_depth_quality_forward_percent(f,70,clock.now)
    clock.now+=.04
    c._limit_depth_quality_forward_percent(frame(1.79,rpm=40),0,clock.now)
    assert c._depth_schedule_recovery is None


def test_stale_fresh_label_cannot_extend_active_ramp(setup):
    clock,c,frame,f=begin(setup)
    c._limit_depth_quality_forward_percent(f,70,clock.now)
    assert c._scheduling_recovery_cap(f,70,100.481,None) is None
    assert c._depth_schedule_recovery is None


def test_rate_respects_lower_configured_rise(setup):
    clock,c,frame,f=begin(setup)
    c.cfg=replace(c.cfg,distance_pid_output_rise_rpm_per_sec=100)
    assert c._limit_depth_quality_forward_percent(f,70,clock.now)==45


def test_gap_history_does_not_extend_on_repeated_attempts(setup):
    clock,c,frame,f=begin(setup)
    hint=c._depth_gap_resume_hint
    for elapsed in (.38,.45,.49):
        clock.now=100+elapsed
        c._note_depth_quality_failure(missing(frame),clock.now)
        assert c._depth_gap_resume_hint is hint
    clock.now=100.501
    c._note_depth_quality_failure(missing(frame),clock.now)
    assert c._depth_gap_resume_hint is None


def test_ramp_does_not_increase_above_measured_envelope(setup):
    clock,c,frame,f=begin(setup,rpm=0)
    c._limit_depth_quality_forward_percent(f,70,clock.now)
    for _ in range(5):
        clock.now+=.05
        result=c._limit_depth_quality_forward_percent(frame(2.02,rpm=0),70,clock.now)
        assert result<=36


def test_unapproved_pid_request_cannot_seed_recovery(setup):
    clock,c,frame,f=begin(setup)
    c._depth_gap_resume_hint=None
    c._depth_last_approved_forward_rpm=None
    c._distance_pid._last_output_rpm=150
    c._note_depth_quality_failure(missing(frame),clock.now)
    assert c._depth_gap_resume_hint[3]==0
    assert c._scheduling_recovery_cap(f,70,clock.now,c._depth_gap_resume_hint) is None
