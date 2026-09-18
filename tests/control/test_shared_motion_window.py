"""No hardware: one physical timeline for closure and target velocity."""
from dataclasses import replace

import pytest

from car_control_modular.longitudinal_approach import RawDepthClosingWindow
from car_control_modular.longitudinal_feedforward import (
    LongitudinalFeedforwardConfig, LongitudinalFeedforwardEstimator,
)
from test_longitudinal_feedforward import update
from test_distance_tracking_response import setup, decide
from test_longitudinal_approach import enabled_controller


def estimator():
    window = RawDepthClosingWindow()
    config = LongitudinalFeedforwardConfig(max_tracking_base_rpm=80,
        max_sample_age_sec=.18, max_sample_gap_sec=.18)
    return LongitudinalFeedforwardEstimator(config, shared_window=window), window


@pytest.mark.parametrize('human_speed', [0., .2, .6, 1.])
def test_same_window_integrates_accelerating_ego_and_rotation(human_speed):
    e, w = estimator()
    z, old_rpm, old_rotation = 3., 20., .01
    update(e, distance=z, ego=old_rpm, rotation_rate_m_s=old_rotation)
    for i, rpm in enumerate((30., 50., 70.), 1):
        rotation = .01+i*.01
        z += (human_speed-.5*(rpm+old_rpm)*.01+.5*(rotation+old_rotation))*.05
        result = update(e, now=100+i*.05, distance=z, ego=rpm, rotation_rate_m_s=rotation)
        assert result.range_rate_m_s == w.rate
        assert result.window_target_speed_m_s == pytest.approx(human_speed, abs=1e-10)
        assert result.speed_window_sec == w.span
        assert result.speed_window_samples == len(w.samples)
        old_rpm, old_rotation = rpm, rotation
    if human_speed == 0:
        assert not result.eligible
        assert result.status == 'no_forward_motion'
        assert w.rate is not None  # valid closure is not lost with zero matching


def test_failed_attempt_does_not_move_timestamps_or_restart_small_turn():
    e, w = estimator()
    update(e, ego=30., rotation_rate_m_s=0.)
    update(e, now=100.05, ego=30., rotation_rate_m_s=0.)
    before = list(w.samples)
    assert e.preserve_compensated_gap(now=100.07, yaw=4.)
    assert w.samples == before
    assert e.last_result.sample_timestamp == pytest.approx(100.03)
    result = update(e, now=100.10, ego=30., yaw_rate_dps=4.8, rotation_rate_m_s=0.)
    assert result.eligible and result.sample_count == 3
    assert result.chain_reset_reason is None
    assert w.span == pytest.approx(.10)


@pytest.mark.parametrize('rotation', [None, 0.])
def test_low_uncertainty_missing_rotation_is_not_a_mode_reset(rotation):
    e, w = estimator()
    update(e, ego=30., rotation_rate_m_s=rotation)
    result = update(e, now=100.05, ego=30., rotation_rate_m_s=0. if rotation is None else None)
    assert result.eligible and len(w.samples) == 2


def test_unverified_rotation_still_rebuilds():
    e, w = estimator()
    update(e, ego=30., rotation_rate_m_s=0.)
    assert e.preserve_compensated_gap(now=100.01, yaw=-12.)
    result = update(e, now=100.05, ego=30., yaw_rate_dps=12., rotation_rate_m_s=0.)
    assert result.status == 'warming_up'
    assert result.chain_reset_reason == 'shared_gap_rotation_uncertainty'
    assert len(w.samples) == 1


def test_real_physical_gap_rebuilds_without_using_old_depth():
    e, w = estimator()
    update(e, ego=30.)
    result = update(e, now=100.20, ego=30.)
    assert result.status == 'warming_up'
    assert result.chain_reset_reason == 'shared_physical_gap'
    assert len(w.samples) == 1
    assert not e.preserve_compensated_gap(now=100.5, yaw=0.)


@pytest.mark.parametrize('change,status', [
    ({'yaw_rate_dps':16., 'rotation_rate_m_s':0.}, 'turning'),
    ({'yaw_rate_dps':6.}, 'turning'),
    ({'target_bearing_deg':11.}, 'off_axis'),
    ({'trusted':False}, 'untrusted_depth'),
    ({'distance':1.4}, 'too_close'),
    ({'distance':3.}, 'range_rate_out_of_bounds'),
    ({'ego':150.}, 'ego_speed_out_of_bounds'),
    ({'feedback_timestamp':99.8}, 'stale_feedback'),
    ({'sample_timestamp':99.99, 'now':100.20}, 'stale_depth'),
    ({'yaw_rate_dps':5., 'target_bearing_deg':10.}, 'rotation_uncertainty'),
])
def test_current_safety_rejections_clear_shared_history(change, status):
    e, w = estimator()
    update(e, ego=30.)
    result = update(e, **dict({'now':100.05,'ego':30.}, **change))
    assert result.status == status
    assert not result.eligible and not w.samples


@pytest.mark.parametrize('stamp,status', [(99.98,'duplicate_depth'),(99.97,'out_of_order_depth'),
                                          (99.99,'interval_too_short')])
def test_repeated_or_sub_resolution_sample_does_not_advance_window(stamp,status):
    e, w = estimator()
    update(e, ego=30.)
    before = list(w.samples)
    result = update(e, now=100.03, sample_timestamp=stamp, ego=30.)
    assert result.status == status
    assert not result.eligible and w.samples == before


def test_uid_change_cannot_reuse_old_motion():
    e, w = estimator()
    update(e, ego=30.)
    update(e, now=100.05, ego=30.)
    result = update(e, now=100.10, ego=30., target_id=2)
    assert result.status == 'warming_up'
    assert w.uid == 2 and len(w.samples) == 1


def test_near_stop_removes_matching_without_launch_baseline():
    e, w = estimator()
    update(e, distance=1.6, ego=30.)
    assert update(e, now=100.05, distance=1.6, ego=30.).eligible
    result = update(e, now=100.10, distance=1.585, ego=30.)
    assert result.status == 'no_forward_motion'
    assert result.target_rpm == 0 and w.rate < 0


def test_repeated_controller_read_keeps_closure_without_advancing_evidence(setup):
    clock, c, frame = enabled_controller(setup)
    decide(c, frame(2.4, rpm=30.))
    clock.now += .05
    f = frame(2.4, rpm=30.)
    decide(c, f)
    before = list(c._raw_closing_window.samples)
    stamp = c._longitudinal_motion_stamp
    decide(c, f)
    assert c._braking_rate_source == 'raw_depth_window'
    assert c._braking_range_rate == 0.
    assert c._raw_closing_window.samples == before
    assert c._longitudinal_motion_stamp == stamp


def test_current_repeated_frame_danger_cannot_reuse_closure(setup):
    clock, c, frame = enabled_controller(setup)
    decide(c, frame(2.4, rpm=30.))
    clock.now += .05
    f = frame(2.4, rpm=30.)
    decide(c, f)
    danger = replace(f, hazard=replace(f.hazard, active=True))
    decide(c, danger)
    assert not c._raw_closing_window.samples
