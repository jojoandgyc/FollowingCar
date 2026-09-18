"""Hardware-free target-speed estimation and fail-closed evidence tests."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.longitudinal_feedforward import (
    LongitudinalFeedforwardConfig, LongitudinalFeedforwardEstimator,
)


def update(estimator, *, now=100.0, distance=2.0, ego=0.0, **changes):
    inputs = dict(
        now=now, sample_timestamp=now - 0.02, distance_m=distance, target_id=1,
        feedback_timestamp=now - 0.02, ego_forward_rpm=ego, yaw_rate_dps=0.0,
        trusted=True, target_bearing_deg=0.0,
    )
    inputs.update(changes)
    return estimator.update(**inputs)


def moving_pair(estimator=None):
    estimator = estimator or LongitudinalFeedforwardEstimator()
    first = update(estimator)
    second = update(estimator, now=100.10, distance=2.03)
    return estimator, first, second


def assert_zero(result):
    assert not result.eligible
    assert result.target_rpm == 0.0 and result.feedforward_rpm == 0.0


def test_needs_two_distinct_depth_samples_and_estimates_walking_target():
    _estimator, first, second = moving_pair()
    assert_zero(first)
    assert first.status == "warming_up" and first.sample_count == 1
    assert second.status == "ready" and second.eligible and second.sample_count == 2
    assert second.range_rate_m_s == pytest.approx(0.3)
    assert second.target_speed_m_s == pytest.approx(0.3)
    assert second.target_rpm == pytest.approx(30.0)
    assert second.feedforward_rpm == pytest.approx(10.0)
    assert second.target_id == 1 and second.sample_timestamp == pytest.approx(100.08)
    with pytest.raises(FrozenInstanceError):
        second.target_rpm = 100.0


def test_person_and_car_at_point_three_m_s_match_thirty_rpm_at_one_point_five_m():
    estimator = LongitudinalFeedforwardEstimator()
    update(estimator, distance=1.5, ego=30.0)
    result = update(estimator, now=100.1, distance=1.5, ego=30.0)
    assert result.eligible
    assert result.range_rate_m_s == 0.0
    assert result.target_speed_m_s == pytest.approx(0.3)
    assert result.target_rpm == pytest.approx(30.0)  # Not capped at 20 or biased to 50.
    assert result.feedforward_rpm == pytest.approx(10.0)


@pytest.mark.parametrize("ego_rpm", [0.0, 10.0, 30.0, 60.0, 100.0, -30.0])
def test_stationary_target_cancels_ego_motion_and_cannot_self_accelerate(ego_rpm):
    estimator = LongitudinalFeedforwardEstimator()
    update(estimator, distance=2.0, ego=ego_rpm)
    result = update(estimator, now=100.1, distance=2.0 - ego_rpm * 0.6 / 60.0 * 0.1,
                    ego=ego_rpm)
    assert result.range_rate_m_s == pytest.approx(-ego_rpm * 0.6 / 60.0)
    assert result.target_speed_m_s == pytest.approx(0.0, abs=1e-10)
    assert result.status == "no_forward_motion"
    assert_zero(result)


@pytest.mark.parametrize("previous_rpm,current_rpm", [(0.0, 60.0), (60.0, 0.0), (10.0, 40.0)])
def test_ego_acceleration_uses_interval_average_not_final_speed(previous_rpm, current_rpm):
    estimator = LongitudinalFeedforwardEstimator()
    update(estimator, ego=previous_rpm)
    interval_speed = (previous_rpm + current_rpm) * 0.5 * 0.6 / 60.0
    result = update(estimator, now=100.1, distance=2.0 - interval_speed * 0.1, ego=current_rpm)
    assert result.target_speed_m_s == pytest.approx(0.0, abs=1e-10)
    assert_zero(result)


def test_static_target_cancellation_stays_zero_across_successive_ego_speeds():
    estimator = LongitudinalFeedforwardEstimator()
    distance, previous_rpm = 2.0, 0.0
    update(estimator, distance=distance, ego=previous_rpm)
    for index, rpm in enumerate((10, 20, 30, 40, 30, 20, 10, 0), start=1):
        distance -= 0.5 * (previous_rpm + rpm) * 0.6 / 60.0 * 0.1
        result = update(estimator, now=100.0 + 0.1 * index, distance=distance, ego=rpm)
        assert result.target_speed_m_s == pytest.approx(0.0, abs=1e-10)
        assert_zero(result)
        previous_rpm = rpm


def test_low_matching_speed_has_no_startup_floor():
    estimator = LongitudinalFeedforwardEstimator()
    update(estimator, distance=1.5, ego=10)
    result = update(estimator, now=100.1, distance=1.5, ego=10)
    assert result.eligible and result.target_rpm == pytest.approx(10.0)
    assert result.feedforward_rpm == 0.0


def test_speed_above_match_cap_preserves_estimate_but_caps_additional_rpm():
    estimator = LongitudinalFeedforwardEstimator()
    update(estimator)
    result = update(estimator, now=100.1, distance=2.10)
    assert result.status == "ready_capped" and result.eligible
    assert result.target_speed_m_s == pytest.approx(1.0)
    assert result.target_rpm == 40.0 and result.feedforward_rpm == 20.0


def test_configured_baseline_is_replaced_not_added_twice_and_extra_cap_cannot_exceed_twenty():
    estimator = LongitudinalFeedforwardEstimator(LongitudinalFeedforwardConfig(
        baseline_rpm=10.0, max_feedforward_rpm=100.0,
    ))
    update(estimator)
    result = update(estimator, now=100.1, distance=2.10)
    assert result.target_rpm == 30.0 and result.feedforward_rpm == 20.0
    estimator = LongitudinalFeedforwardEstimator(LongitudinalFeedforwardConfig(
        max_feedforward_rpm=5.0,
    ))
    assert moving_pair(estimator)[2].target_rpm == 25.0


def test_static_or_approaching_target_clears_prior_positive_smoothing_immediately():
    estimator, _first, positive = moving_pair()
    assert positive.eligible
    static = update(estimator, now=100.2, distance=2.03)
    assert_zero(static)
    approaching = update(estimator, now=100.3, distance=2.02)
    assert approaching.target_speed_m_s == pytest.approx(-0.1)
    assert_zero(approaching)


@pytest.mark.parametrize("speed", [0.0, 0.01, 0.029])
def test_small_forward_noise_does_not_authorize_motion(speed):
    estimator = LongitudinalFeedforwardEstimator()
    update(estimator)
    assert_zero(update(estimator, now=100.1, distance=2.0 + speed * 0.1))


def test_positive_target_speed_is_filtered_after_first_estimate():
    estimator, _first, _second = moving_pair()
    result = update(estimator, now=100.2, distance=2.05)
    assert result.target_speed_m_s == pytest.approx(0.3 + 0.35 * (0.2 - 0.3))
    assert result.target_rpm == pytest.approx(26.5)


@pytest.mark.parametrize("sample_stamp,status", [(100.08, "duplicate_depth"), (99.99, "out_of_order_depth")])
def test_duplicate_and_old_cross_source_samples_do_not_count_or_change_chain(sample_stamp, status):
    estimator, _first, _second = moving_pair()
    previous = estimator._previous
    filtered = estimator._filtered_target_speed
    rejected = update(estimator, now=100.15, sample_timestamp=sample_stamp, distance=4.0, trusted=False)
    assert_zero(rejected)
    assert rejected.status == status and rejected.sample_count == 2
    assert estimator._previous is previous and estimator._filtered_target_speed == filtered
    next_valid = update(estimator, now=100.2, distance=2.06)
    assert next_valid.sample_count == 3 and next_valid.target_rpm == pytest.approx(30.0)


def test_short_intervals_do_not_inflate_count_or_make_noisy_derivative():
    estimator = LongitudinalFeedforwardEstimator()
    update(estimator)
    skipped = update(estimator, now=100.01, distance=2.003)
    assert skipped.status == "interval_too_short" and skipped.sample_count == 1
    assert_zero(skipped)
    result = update(estimator, now=100.033, distance=2.0099)
    assert result.sample_count == 2 and result.target_rpm == pytest.approx(30.0)


def test_long_gap_restarts_with_one_current_fresh_sample():
    estimator, _first, _second = moving_pair()
    gap = update(estimator, now=100.4, distance=2.12)
    assert gap.status == "gap_reset" and gap.sample_count == 1
    assert_zero(gap)
    assert update(estimator, now=100.5, distance=2.15).eligible


@pytest.mark.parametrize("changes,status", [
    ({"trusted": False}, "untrusted_depth"),
    ({"trusted": "true"}, "untrusted_depth"),
    ({"distance_m": None}, "invalid_distance"),
    ({"distance_m": float("nan")}, "invalid_distance"),
    ({"distance_m": True}, "invalid_distance"),
    ({"distance_m": 0.0}, "invalid_distance"),
    ({"distance_m": 1.46}, "too_close"),
    ({"distance_m": 0.4}, "too_close"),
    ({"feedback_timestamp": None}, "invalid_feedback"),
    ({"feedback_timestamp": 100.3}, "future_feedback"),
    ({"feedback_timestamp": 100.0}, "stale_feedback"),
    ({"feedback_timestamp": 100.04}, "stale_feedback"),
    ({"ego_forward_rpm": 100.1}, "ego_speed_out_of_bounds"),
    ({"ego_forward_rpm": -100.1}, "ego_speed_out_of_bounds"),
    ({"ego_forward_rpm": float("inf")}, "invalid_feedback"),
    ({"yaw_rate_dps": 5.01}, "turning"),
    ({"yaw_rate_dps": -5.01}, "turning"),
    ({"yaw_rate_dps": None}, "invalid_feedback"),
    ({"target_bearing_deg": 10.01}, "off_axis"),
    ({"target_bearing_deg": -10.01}, "off_axis"),
    ({"target_bearing_deg": float("nan")}, "invalid_feedback"),
    ({"distance_m": 2.34}, "depth_jump"),
    ({"distance_m": 2.23}, "range_rate_out_of_bounds"),
    ({"distance_m": 2.13, "ego_forward_rpm": 100.0}, "target_speed_out_of_bounds"),
])
def test_new_invalid_evidence_clears_ff_and_requires_two_good_samples_again(changes, status):
    estimator, _first, _second = moving_pair()
    # For the target-speed cap case both ego endpoints must carry 100 RPM.
    if status == "target_speed_out_of_bounds":
        estimator._previous = replace(estimator._previous, ego_forward_m_s=1.0)
    result = update(estimator, now=100.2, distance=2.06, **changes)
    assert_zero(result)
    assert result.status == status and result.sample_count == 0
    retry = update(estimator, now=100.21, sample_timestamp=100.18, distance=2.06)
    assert retry.status == "duplicate_depth" and retry.sample_count == 0
    assert update(estimator, now=100.3, distance=2.09).status == "warming_up"
    assert update(estimator, now=100.4, distance=2.12).eligible


def test_feedback_must_be_aligned_with_depth_not_just_recent_at_processing():
    estimator = LongitudinalFeedforwardEstimator()
    result = update(estimator, sample_timestamp=99.78, feedback_timestamp=99.98)
    assert_zero(result)
    assert result.status == "feedback_depth_misaligned"


def test_depth_and_feedback_stale_after_processing_never_refresh_evidence():
    estimator, _first, _second = moving_pair()
    stale = update(estimator, now=100.5, sample_timestamp=100.10)
    assert stale.status == "stale_depth" and stale.sample_count == 0
    assert_zero(stale)
    assert update(estimator, now=100.6, distance=2.15).status == "warming_up"


def test_future_depth_does_not_poison_watermark_but_requires_new_confirmation():
    estimator, _first, _second = moving_pair()
    future = update(estimator, now=100.2, sample_timestamp=1000.0)
    assert future.status == "future_depth" and future.sample_count == 0
    assert_zero(future)
    assert update(estimator, now=100.3, distance=2.09).status == "warming_up"
    assert update(estimator, now=100.4, distance=2.12).eligible


@pytest.mark.parametrize("target", [None, 0, -1, 1.5, True, float("nan"), float("inf")])
def test_invalid_or_unbound_uid_cannot_produce_feedforward(target):
    estimator, _first, _second = moving_pair()
    result = update(estimator, now=100.2, target_id=target)
    assert result.status == "invalid_target" and result.sample_count == 0
    assert_zero(result)


def test_target_switch_resets_all_velocity_and_filter_history():
    estimator, _first, _second = moving_pair()
    switched = update(estimator, now=100.2, target_id=2, distance=1.5, ego=10)
    assert switched.status == "warming_up" and switched.target_id == 2
    result = update(estimator, now=100.3, target_id=2, distance=1.5, ego=10)
    assert result.target_rpm == pytest.approx(10.0)
    estimator.reset()
    assert estimator.last_result.status == "reset"
    assert update(estimator, distance=1.5, ego=10).status == "warming_up"


def test_safety_distance_boundary_and_configurable_margin():
    estimator = LongitudinalFeedforwardEstimator()
    update(estimator, distance=1.47, ego=30)
    assert update(estimator, now=100.1, distance=1.47, ego=30).eligible
    tighter = LongitudinalFeedforwardEstimator(LongitudinalFeedforwardConfig(min_distance_m=1.5))
    assert update(tighter, distance=1.49, ego=30).status == "too_close"


def test_reused_feedback_is_allowed_but_older_feedback_is_not():
    estimator = LongitudinalFeedforwardEstimator()
    update(estimator, feedback_timestamp=99.99)
    valid = update(estimator, now=100.033, distance=2.0099, feedback_timestamp=99.99)
    assert valid.eligible
    invalid = update(estimator, now=100.05, distance=2.015, feedback_timestamp=99.98)
    assert invalid.status == "out_of_order_feedback"
    assert_zero(invalid)


@pytest.mark.parametrize("change", [
    {"wheel_circumference_m": 0.0}, {"wheel_circumference_m": None},
    {"wheel_circumference_m": "0.6"}, {"max_sample_gap_sec": "0.25"},
    {"baseline_rpm": -1}, {"max_feedforward_rpm": -1},
    {"min_distance_m": 0.5}, {"max_sample_age_sec": float("nan")},
    {"max_feedback_age_sec": 0}, {"max_sample_gap_sec": 0.01},
    {"max_abs_yaw_rate_dps": -1}, {"target_speed_filter_alpha": 1.1},
    {"target_speed_filter_alpha": 0}, {"target_speed_deadband_m_s": 2.0},
])
def test_invalid_configuration_fails_closed(change):
    estimator = LongitudinalFeedforwardEstimator(replace(LongitudinalFeedforwardConfig(), **change))
    result = update(estimator)
    assert result.status == "invalid_config"
    assert_zero(result)


@pytest.mark.parametrize("changes", [
    {"now": float("nan")}, {"now": 0}, {"sample_timestamp": None},
    {"sample_timestamp": float("inf")}, {"sample_timestamp": -1},
])
def test_invalid_timestamp_clears_evidence(changes):
    estimator, _first, _second = moving_pair()
    result = update(estimator, **changes)
    assert result.status == "invalid_timestamp" and result.sample_count == 0
    assert_zero(result)
