"""CAP386 marginal ROI expiry: real controller, no hardware or renewed lease."""
from dataclasses import replace
import pytest

from test_distance_tracking_response import setup
from test_distance_pi_controller import configured, step
from test_cap437_braking_geometry import observation


def primed(setup):
    clock, c, frame = configured(setup, target_distance_m=1.4,
        distance_pi_launch_request_rpm=180., distance_pi_kp_per_sec=3.,
        distance_pi_motion_memory_sec=.35, depth_longitudinal_sample_max_age_sec=.25)
    for d in (1.97, 1.98, 1.99):
        step(c, observation(frame, clock, d, rpm=62., yaw=-3.135,
                           raw_yaw=-3.135, x=.337, age=.1676))
        clock.now += .05
    return clock, c, frame


def current(frame, clock, **kw):
    args = dict(distance=1.987, rpm=90., yaw=-3.135, raw_yaw=-3.135,
                x=.337, age=.254146)
    args.update(kw)
    return observation(frame, clock, **args)


def test_marginal_expiry_retains_old_window_without_authorizing_new_closure(setup):
    clock, c, frame = primed(setup)
    before = list(c._raw_closing_window.samples)
    prior = c.last_distance_pid_result.output_rpm
    step(c, current(frame, clock))
    r = c.last_distance_pid_result
    assert c._raw_closing_window.samples == before
    assert c._braking_rate_source != 'raw_depth_window'
    assert r.pi_brake_source == 'relative_motion_memory'
    assert 30 < r.output_rpm <= prior
    # A fresh aligned observation resumes the old physical timeline.
    clock.now += .05
    step(c, current(frame, clock, age=.15, distance=1.98))
    assert c.last_distance_pid_result.pi_brake_source == 'raw_relative_motion'


@pytest.mark.parametrize('change', ['old_roi', 'yaw', 'near', 'feedback', 'identity', 'hazard'])
def test_soft_gap_does_not_hide_real_safety_rejection(setup, change):
    clock, c, frame = primed(setup)
    f = current(frame, clock)
    if change == 'old_roi': f = current(frame, clock, age=.31)
    if change == 'yaw': f = current(frame, clock, yaw=36., raw_yaw=36.)
    if change == 'near': f = current(frame, clock, distance=1.39)
    if change == 'feedback': f = replace(f, steering_feedback=replace(f.steering_feedback, trustworthy=False))
    if change == 'identity': c.active_target_id = 2
    if change == 'hazard': f = replace(f, hazard=replace(f.hazard, active=True))
    step(c, f)
    assert c.last_distance_pid_result is None or c.last_distance_pid_result.pi_brake_source != 'relative_motion_memory'


def test_repeated_gap_never_renews_retained_window(setup):
    clock, c, frame = primed(setup)
    origin = c._raw_closing_window.samples[-1][0]
    for _ in range(5):
        step(c, current(frame, clock))
        assert not c._raw_closing_window.samples or c._raw_closing_window.samples[-1][0] == origin
        clock.now += .05
    assert not c._raw_closing_window.samples
    assert c.last_distance_pid_result.pi_brake_source != 'relative_motion_memory'


def test_new_shrinking_range_still_brakes_during_soft_gap(setup):
    clock, c, frame = primed(setup)
    step(c, current(frame, clock, distance=1.5))
    assert c.last_distance_pid_result.output_rpm == 0
