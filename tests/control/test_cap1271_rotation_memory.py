"""Marginal rotation uncertainty must not insert corrupted derivative samples."""
from dataclasses import replace
import pytest

from test_distance_tracking_response import setup
from test_distance_pi_controller import configured, step
from test_cap437_braking_geometry import observation


def prime(setup):
    clock,c,frame=configured(setup,target_distance_m=1.4,
        distance_pi_launch_request_rpm=180.,distance_pi_kp_per_sec=3.,
        distance_pi_motion_memory_sec=.35,depth_longitudinal_sample_max_age_sec=.25)
    for d in (1.90,1.91,1.92):
        step(c,observation(frame,clock,d,rpm=70.,yaw=5.,raw_yaw=5.,x=.92,age=.138))
        clock.now+=.05
    return clock,c,frame


def marginal(frame,clock,d=1.93,yaw=15.):
    return observation(frame,clock,d,rpm=75.,yaw=yaw,raw_yaw=yaw,x=.92,age=.138)


def test_marginal_rotation_breaks_raw_window_but_keeps_bounded_memory(setup,caplog):
    clock,c,frame=prime(setup)
    origin=c._raw_closing_window.samples[-1][0]
    before=c.last_distance_pid_result.output_rpm
    with caplog.at_level('INFO'): step(c,marginal(frame,clock))
    r=c.last_distance_pid_result
    assert not c._raw_closing_window.samples
    assert c._closure_rotation_gap==(1,origin)
    assert r.pi_brake_source=='relative_motion_memory'
    assert 0 <= r.output_rpm <= before
    assert 'reason=rotation_uncertainty_gap' in caplog.text
    clock.now+=.05
    step(c,observation(frame,clock,1.94,rpm=75.,yaw=5.,raw_yaw=5.,x=.92,age=.138))
    assert len(c._raw_closing_window.samples)==1  # No differencing across rejected turn.
    clock.now+=.05
    step(c,observation(frame,clock,1.95,rpm=75.,yaw=5.,raw_yaw=5.,x=.92,age=.138))
    assert c.last_distance_pid_result.pi_brake_source=='raw_relative_motion'


def test_repeated_marginal_frames_cannot_extend_memory_origin(setup):
    clock,c,frame=prime(setup)
    origin=c._raw_closing_window.samples[-1][0]
    for _ in range(5):
        step(c,marginal(frame,clock))
        assert c._closure_rotation_gap in (None,(1,origin))
        clock.now+=.05
    assert c.last_distance_pid_result.pi_brake_source!='relative_motion_memory'


@pytest.mark.parametrize('case',['high_yaw','uncertainty','feedback','near','hazard','identity'])
def test_memory_never_overrides_real_safety_rejection(setup,case):
    clock,c,frame=prime(setup)
    f=marginal(frame,clock)
    if case=='high_yaw': f=marginal(frame,clock,yaw=36.)
    if case=='uncertainty': f=marginal(frame,clock,yaw=30.)
    if case=='near': f=marginal(frame,clock,d=1.39)
    if case=='feedback': f=replace(f,steering_feedback=replace(f.steering_feedback,trustworthy=False))
    if case=='hazard': f=replace(f,hazard=replace(f.hazard,active=True))
    if case=='identity': c.active_target_id=2
    step(c,f)
    assert c.last_distance_pid_result is None or c.last_distance_pid_result.pi_brake_source!='relative_motion_memory'


def test_actual_shrinking_distance_can_still_command_zero(setup):
    clock,c,frame=prime(setup)
    step(c,marginal(frame,clock,d=1.6,yaw=18.))
    assert c.last_distance_pid_result.output_rpm==0


def test_marginal_gap_avoids_old_stationary_person_fallback_when_range_recedes(setup):
    clock,c,frame=prime(setup)
    previous=c.last_distance_pid_result.output_rpm
    step(c,marginal(frame,clock))
    retained=c.last_distance_pid_result.output_rpm
    clock,old,frame=prime(setup)
    # Numerical historical comparator: the previous code erased this memory
    # whenever rotation uncertainty crossed .25, despite no closing evidence.
    old._distance_pid.invalidate_motion_memory()
    step(old,marginal(frame,clock))
    assert old.last_distance_pid_result.pi_brake_source=='motion_unknown_bound'
    assert old.last_distance_pid_result.output_rpm < retained <= previous
