"""Far catch-up is not a stop; near, invalid and expired evidence still wins."""
from dataclasses import replace

import pytest

from car_control_modular.longitudinal_feedforward import (
    LongitudinalFeedforwardConfig, LongitudinalFeedforwardEstimator,
)
from test_longitudinal_feedforward import update
from test_matching_speed_budget import warmed
from test_distance_tracking_response import setup
from test_scheduling_gap_evidence import recovery


def estimator():
    return LongitudinalFeedforwardEstimator(LongitudinalFeedforwardConfig(
        max_tracking_base_rpm=80, target_speed_filter_alpha=1))


def test_far_closing_still_positive_uses_bounded_decline():
    e = warmed()
    r = update(e, now=100.9, distance=1.97, ego=10)
    assert r.instantaneous_target_speed_m_s == pytest.approx(.15)
    assert r.target_rpm == pytest.approx(72)
    assert r.matching_rate_limited
    assert r.decline_policy == 'bounded_far_positive'


def test_noisy_depth_uses_window_not_last_pair_only():
    e = estimator()
    # True target .5m/s, car .7m/s; +/-2mm depth noise changes pair rates.
    for i, noise in enumerate((0, .002, -.002, .002, -.002)):
        r = update(e, now=100+i*.04, distance=2.7-.2*i*.04+noise, ego=70)
    assert r.speed_window_samples == 5
    assert r.speed_window_sec == pytest.approx(.16)
    assert abs(r.window_target_speed_m_s-.5) < abs(r.instantaneous_target_speed_m_s-.5)
    assert r.window_target_speed_m_s == pytest.approx(.49, abs=.01)


def test_integrated_ego_acceleration_and_rotation_share_window_interval():
    e = estimator()
    z, previous_ego, previous_rotation = 2.7, .3, .01
    for i, (ego, rotation) in enumerate(((.3,.01),(.4,.02),(.6,.03),(.7,.04))):
        if i:
            z += (.5 - .5*(previous_ego+ego) + .5*(previous_rotation+rotation))*.04
        r = update(e, now=100+i*.04, distance=z, ego=ego*100,
                   rotation_rate_m_s=rotation, yaw_rate_dps=10)
        previous_ego, previous_rotation = ego, rotation
    assert r.window_target_speed_m_s == pytest.approx(.5)
    assert r.speed_window_samples == 4


def test_window_has_bounded_age_and_duplicates_do_not_add_points():
    e = estimator()
    for i in range(20):
        r = update(e, now=100+i*.03, distance=2.7, ego=60)
        assert r.speed_window_sec <= .18000001
        count = len(e._speed_samples)
        assert not update(e, now=100+i*.03, distance=2.7, ego=60).eligible
        assert len(e._speed_samples) == count
    assert r.speed_window_samples <= 7
    assert r.sample_timestamp == pytest.approx(100+.57-.02)


@pytest.mark.parametrize('change', ['uid','gap','untrusted','jump','mode','yaw','stale','feedback'])
def test_invalid_or_new_chain_cannot_inherit_window(change):
    e = estimator()
    for i in range(4): update(e, now=100+i*.04, distance=2.7, ego=60)
    kw = dict(now=100.16, distance=2.7, ego=60)
    if change == 'uid': kw['target_id'] = 2
    if change == 'gap': kw['now'] = 100.5
    if change == 'untrusted': kw['trusted'] = False
    if change == 'jump': kw['distance'] = 3.5
    if change == 'mode': kw['rotation_rate_m_s'] = .01
    if change == 'yaw': kw['yaw_rate_dps'] = 20
    if change == 'stale': kw.update(now=100.5, sample_timestamp=100.2)
    if change == 'feedback': kw['feedback_timestamp'] = 99
    r = update(e, **kw)
    assert not r.eligible
    assert len(e._speed_samples) <= 1


def test_repeated_target_stop_clears_history_after_one_bounded_sample():
    e = estimator()
    for i in range(6): update(e, now=100+i*.04, distance=2.7, ego=60)
    r = update(e, now=100.24, distance=2.676, ego=60)
    assert r.decline_policy == 'stop_suspect_bounded'
    assert r.target_rpm < 60
    r = update(e, now=100.28, distance=2.652, ego=60)
    assert r.status == 'no_forward_motion'
    assert not r.eligible and r.target_rpm == 0
    assert r.decline_policy == 'stop_evidence'
    assert len(e._speed_samples) == 1


def test_imminent_near_band_does_not_rate_limit_decline():
    e = estimator()
    for i in range(9): update(e, now=100+i*.04, distance=1.9, ego=100)
    r = update(e, now=100.36, distance=1.876, ego=100)
    assert r.decline_policy == 'immediate_near_or_ttc'
    assert r.target_rpm == pytest.approx(40)
    assert not r.matching_rate_limited


def far_recovery(setup, **kwargs):
    clock, c, frame, current = recovery(setup, **kwargs)
    h = c._depth_gap_resume_hint
    c._depth_gap_resume_hint = (*h[:2], 2.8, *h[3:])
    c._depth_recovery_anchor = (1, 100., 2.8)
    return clock, c, frame, current


def closing_frame(current, distance=2.74):
    return replace(current, distance_m=distance, distance_state=replace(current.distance_state,
        raw_distance_m=distance, used_distance_m=distance, filtered_distance_m=distance))


def test_six_cm_far_closure_keeps_measured_recovery_not_25(setup, caplog):
    clock, c, _, current = far_recovery(setup)
    current = closing_frame(current)
    assert c._far_closing_recovery_continuous(current, c._depth_gap_resume_hint, clock.now)
    assert c._limit_depth_quality_forward_percent(current, 55, clock.now) == 55
    assert 'distance_policy=far_closing_measured' in caplog.text
    assert 'continuity_restored=True' in caplog.text
    assert 'depth_speed_recovery_started' not in caplog.text
    assert c._depth_gap_resume_hint is None


@pytest.mark.parametrize('change', ['near','fast_closure','large_closure','yaw','reverse',
    'stale_encoder','untrusted_encoder','old_depth','long_gap','hazard','uid','rejected'])
def test_far_closing_recovery_cannot_bypass_safety(setup, caplog, change):
    clock, c, _, current = far_recovery(setup, physical_gap=.22 if change=='long_gap' else .17)
    current = closing_frame(current)
    if change == 'near': current = closing_frame(current, 1.79)
    if change == 'large_closure': current = closing_frame(current, 2.65)
    if change == 'fast_closure':
        current = replace(current, distance_state=replace(current.distance_state, sample_timestamp=100.04))
    if change == 'yaw': current = replace(current, steering_feedback=replace(current.steering_feedback, yaw_rate_right_dps=16))
    if change == 'reverse': current = replace(current, steering_feedback=replace(current.steering_feedback, left_forward_rpm=-1))
    if change == 'stale_encoder': current = replace(current, steering_feedback=replace(current.steering_feedback, timestamp=99))
    if change == 'untrusted_encoder': current = replace(current, steering_feedback=replace(current.steering_feedback, trustworthy=False))
    if change == 'old_depth': current = replace(current, distance_state=replace(current.distance_state, sample_timestamp=100.01))
    if change == 'hazard': current = replace(current, hazard=replace(current.hazard, active=True))
    if change == 'uid': c.active_target_id = 2
    if change == 'rejected': current = replace(current, distance_state=replace(current.distance_state, source_detail='depth_far_background_guard'))
    c._limit_depth_quality_forward_percent(current, 55, clock.now)
    assert 'distance_policy=far_closing_measured' not in caplog.text


def test_recovery_does_not_raise_zero_request(setup):
    clock, c, _, current = far_recovery(setup)
    assert c._limit_depth_quality_forward_percent(closing_frame(current), 0, clock.now) == 0


def test_recovery_near_band_ttc_still_requires_strict_restart(setup, caplog):
    clock, c, _, current = recovery(setup)
    # 1.94m is above the 1.8m recovery band, but only .397s from it at this rate.
    current = closing_frame(current, 1.94)
    assert not c._far_closing_recovery_continuous(current, c._depth_gap_resume_hint, clock.now)
    assert c._limit_depth_quality_forward_percent(current, 55, clock.now) == 25
    assert 'distance_policy=far_closing_measured' not in caplog.text


def test_legacy_estimator_retains_pairwise_behavior():
    e = LongitudinalFeedforwardEstimator()
    for i in range(4): r = update(e, now=100+i*.04, distance=2.7, ego=30)
    assert r.window_target_speed_m_s is None
    assert r.target_rpm == pytest.approx(30)
