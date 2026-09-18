"""No moving/stopped classification required for a fresh-distance PI update."""
from dataclasses import replace
import math

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_approach import RawDepthClosingWindow
from test_distance_tracking_response import setup
from test_distance_pi_controller import configured, step, missing


def pi(memory=.35):
    return DistancePiController(DistancePiConfig(kp_per_sec=3., launch_request_rpm=180.,
        physical_ttl_sec=.25, motion_memory_sec=memory))


def sample(c, ts=100., *, now=None, distance=2.2, ego=80., rate=.2, valid=True, allow=True, jump=False):
    return c.update(distance, 1.5, sample_timestamp=ts, execution_now=ts if now is None else now,
        deadband_m=.03, max_output_rpm=200., ego_forward_rpm=ego, range_rate_m_s=rate,
        raw_closure_valid=valid, allow_motion_memory=allow, measurement_jump_clamped=jump,
        raw_distance_m=distance)


@pytest.mark.parametrize('gap', [.181, .22, .255, .29])
def test_extended_window_uses_real_gap_endpoints_not_warmup(gap):
    w = RawDepthClosingWindow(max_gap_sec=.30)
    assert w.update(uid=1, stamp=100., raw=2., rotation=0.) is None
    assert w.update(uid=1, stamp=100.+gap, raw=2.+gap*.4, rotation=0.) == pytest.approx(.4)
    assert len(w.samples) == 2 and w.samples[0][0] == 100.
    assert w.span == pytest.approx(gap)
    before = list(w.samples)
    assert w.update(uid=1, stamp=100.+gap, raw=10., rotation=0.) == pytest.approx(.4)
    assert w.samples == before


def test_normal_fit_window_stays_short_after_gap():
    w = RawDepthClosingWindow(max_gap_sec=.30)
    for ts in (100., 100.22, 100.27):
        w.update(uid=1, stamp=ts, raw=2.+(ts-100)*.4, rotation=0.)
    assert [x[0] for x in w.samples] == [100.22, 100.27]
    assert w.span == pytest.approx(.05)


@pytest.mark.parametrize('event', ['long_gap', 'uid', 'rotation_mode', 'outlier'])
def test_history_bridge_does_not_cross_invalid_evidence(event):
    w = RawDepthClosingWindow(max_gap_sec=.30)
    w.update(uid=1, stamp=100., raw=2., rotation=0.)
    args = dict(uid=1, stamp=100.22, raw=2.1, rotation=0.)
    if event == 'long_gap': args['stamp'] = 100.31
    if event == 'uid': args['uid'] = 2
    if event == 'rotation_mode': args['rotation'] = None
    if event == 'outlier': args['raw'] = 4.
    assert w.update(**args) is None


def test_legacy_window_retains_180ms_reset():
    w = RawDepthClosingWindow()
    w.update(uid=1, stamp=100., raw=2., rotation=0.)
    assert w.update(uid=1, stamp=100.22, raw=2.1, rotation=0.) is None


def test_unknown_motion_is_bounded_memory_not_stationary_assumption():
    c = pi()
    first = sample(c)
    c.accept_output_limit(100., first.output_rpm-2.)
    limited = c.last_result.output_rpm
    b = sample(c, 100.05, valid=False, rate=-80*.816814/60.)
    assert b.brake_source == 'relative_motion_memory'
    assert b.motion_origin_ts == 100.
    assert b.motion_uncertainty_m_s == pytest.approx(.1)
    assert 80 < b.output_rpm <= limited


def test_repeated_unknown_samples_cannot_refresh_memory_or_accelerate():
    c = pi()
    previous = sample(c).output_rpm
    for ts in (100.05, 100.1, 100.2, 100.3):
        r = sample(c, ts, valid=False, rate=-1.)
        assert r.brake_source == 'relative_motion_memory'
        assert r.motion_origin_ts == 100.
        assert r.output_rpm <= previous
        previous = r.output_rpm
    r = sample(c, 100.36, valid=False, rate=-1.)
    assert r.brake_source == 'motion_unknown_bound'
    assert c._motion_memory is None


@pytest.mark.parametrize('event', ['permission', 'feedback', 'near', 'jump', 'safety', 'target', 'raw_fault'])
def test_memory_cannot_hide_safety_or_identity_event(event):
    c = pi()
    sample(c)
    args = dict(valid=False, rate=-1.)
    if event == 'permission': args['allow'] = False
    if event == 'feedback': args['ego'] = None
    if event == 'near': args['distance'] = 1.7
    if event == 'jump': args['jump'] = True
    if event == 'safety': c.suspend(100.02, 'hazard', retain=False)
    if event == 'target': c.reset()
    if event == 'raw_fault': c.invalidate_motion_memory()
    r = sample(c, 100.05, **args)
    assert r.brake_source != 'relative_motion_memory'
    assert c._motion_memory is None


def test_actual_fast_closing_overrides_memory_immediately():
    c = pi()
    sample(c)
    r = sample(c, 100.05, distance=1.6, rate=-2., ego=120.)
    assert r.brake_source == 'raw_relative_motion'
    assert r.output_rpm == 0


def test_new_raw_approach_vetoes_old_memory_even_without_window_rate():
    c = pi()
    sample(c, distance=2.5)
    r = sample(c, 100.2, distance=2.05, valid=False)
    assert r.brake_source == 'relative_motion_memory'
    assert r.closing_m_s >= 2.25
    assert r.output_rpm == 0


def test_unapproved_zero_cannot_relaunch_from_memory():
    c = pi()
    sample(c)
    c.accept_output_limit(100., 0.)
    assert sample(c, 100.05, valid=False).output_rpm == 0


@pytest.mark.parametrize('event', ['expired', 'revoked'])
def test_fresh_recovery_memory_cannot_restore_old_high_command(event):
    c = pi()
    sample(c)
    if event == 'revoked': c.suspend(100.02, 'expired', reset_execution=True)
    r = sample(c, 100.26 if event == 'expired' else 100.05, ego=20., valid=False)
    assert r.output_rpm <= 20
    assert r.sample_dt_sec == 0.


def test_stale_duplicate_and_reordered_depth_do_not_refresh_origin():
    c = pi()
    sample(c)
    origin = c._motion_memory
    assert sample(c, 100., now=100.05).status == 'duplicate'
    assert sample(c, 99.99).status == 'out_of_order'
    assert sample(c, 100.1, now=100.4).status == 'stale_sample'
    assert c._motion_memory == origin


@pytest.mark.parametrize('gap', [.20, .22, .26, .29])
def test_actual_controller_keeps_closure_after_short_gap(setup, gap):
    clock, c, frame = configured(setup, distance_pi_launch_request_rpm=180.,
        distance_pi_motion_memory_sec=.35, depth_longitudinal_sample_max_age_sec=.25)
    step(c, frame(1.9, rpm=70.))
    clock.now += .05
    step(c, frame(1.92, rpm=70.))
    before = c.last_distance_pid_result
    clock.now += gap
    step(c, frame(1.92+gap*.4, rpm=70.))
    r = c.last_distance_pid_result
    assert r.pi_brake_source == 'raw_relative_motion'
    assert r.output_rpm >= before.output_rpm > 90
    assert r.pi_sample_dt_sec == 0.  # no integration of gap
    assert c._raw_closing_window.span == pytest.approx(gap)


def test_no_measurement_attempt_does_not_clear_short_history(setup):
    clock, c, frame = configured(setup, distance_pi_motion_memory_sec=.35,
        depth_longitudinal_sample_max_age_sec=.25, distance_pi_launch_request_rpm=180.)
    step(c, frame(2.2, rpm=80.))
    clock.now += .05
    step(c, frame(2.22, rpm=80.))
    ts = c._raw_closing_window.samples[-1][0]
    clock.now += .20
    step(c, missing(frame))
    assert c._raw_closing_window.samples[-1][0] == ts
    clock.now += .02
    step(c, frame(2.308, rpm=80.))
    assert c.last_distance_pid_result.pi_brake_source == 'raw_relative_motion'


@pytest.mark.parametrize('value', [-.1, .36, math.inf, math.nan, True])
def test_invalid_motion_memory_config_rejected(value):
    with pytest.raises(ValueError):
        pi(value)


def test_zero_feature_switch_preserves_legacy_fallback():
    c = pi(0.)
    sample(c)
    assert sample(c, 100.05, valid=False).brake_source == 'stationary_fallback'


@pytest.mark.parametrize('rate', [math.nan, math.inf, 3.01, -3.01])
def test_bad_claimed_raw_rate_must_not_be_hidden_by_memory(rate):
    c = pi()
    sample(c)
    r = sample(c, 100.05, rate=rate)
    assert r.brake_source != 'relative_motion_memory'
    assert c._motion_memory is None


def test_actual_warmup_recovery_uses_memory_without_restarting_at_fixed_low_speed(setup):
    clock, c, frame = configured(setup, distance_pi_motion_memory_sec=.35,
        depth_longitudinal_sample_max_age_sec=.25, distance_pi_launch_request_rpm=180.)
    step(c, frame(2.3, rpm=80.))
    clock.now += .05
    step(c, frame(2.32, rpm=80.))
    origin = c._distance_pid_last_sample_timestamp
    clock.now += .31
    decision = step(c, frame(2.35, rpm=80.))
    r = c.last_distance_pid_result
    assert r.pi_brake_source == 'relative_motion_memory'
    assert r.pi_motion_origin_ts == origin
    assert 50 < r.output_rpm <= 80
    assert r.pi_sample_dt_sec == 0
    assert not any(a.kind == 'stop' for a in decision.actions)


@pytest.mark.parametrize('event', ['hazard', 'identity', 'yaw', 'untrusted'])
def test_real_controller_rejection_clears_motion_memory(setup, event):
    clock, c, frame = configured(setup, distance_pi_motion_memory_sec=.35,
        depth_longitudinal_sample_max_age_sec=.25, distance_pi_launch_request_rpm=180.)
    step(c, frame(2.3, rpm=80.))
    clock.now += .05
    step(c, frame(2.32, rpm=80.))
    assert c._distance_pid._distance_pi._motion_memory is not None
    clock.now += .05
    f = frame(2.34, rpm=80.)
    if event == 'hazard': f = replace(f, hazard=replace(f.hazard, active=True))
    if event == 'identity': c.active_target_id = 2
    if event == 'yaw': f = replace(f, steering_feedback=replace(f.steering_feedback, yaw_rate_right_dps=30.))
    if event == 'untrusted': f = replace(f, steering_feedback=replace(f.steering_feedback, trustworthy=False))
    step(c, f)
    assert c._distance_pid._distance_pi._motion_memory is None
