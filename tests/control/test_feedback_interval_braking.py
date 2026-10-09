"""Bounded feedback-age overlap, not relaxed emergency deceleration.

Pure model and real controller checks. No sensor/motor hardware is opened.
"""
from dataclasses import replace
import math

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.sample_braking import SampleBrakingAssessment
from test_distance_pi_controller import configured, step
from test_distance_tracking_response import setup


def cap135(covered=True):
    return SampleBrakingAssessment(
        1, 8817.457824757, 8817.579895293, 2.225378219278882,
        76., 76., 8817.457083359, 0., 1.1, .816814, 1., .15, 200.,
        outer_allowance_rpm=10., observed_feedback_reserve=True,
        feedback_interval_covered=covered)


def test_cap135_reduces_to_69rpm_instead_of_zero_without_lowering_stop_boundary():
    old, new = cap135(False), cap135()
    now = 8817.733297922
    args = dict(authorized_rpm=74., execution_bound_rpm=84.)
    before, after = old.budget(now, 75., **args), new.budget(now, 75., **args)
    assert before.reason == "shared_braking_momentum" and before.cap_rpm == 0.
    assert before.required_stop_m-before.margin_m == pytest.approx(.0111730332)
    assert after.reason == "shared_braking_cap" and 69. < after.cap_rpm < 70.
    assert after.required_stop_m == before.required_stop_m
    assert after.required_stop_m < after.margin_m
    assert new.effective_feedback_reserve_sec == .05
    assert new.stop_distance_m == old.stop_distance_m == 1.1
    assert new.deceleration_m_s2 == old.deceleration_m_s2 == 1.
    assert new.response_delay_sec == old.response_delay_sec == .15
    # Neither a larger request nor a later control tick renews this sample.
    assert new.budget(new.sample_timestamp+.300001, 75., **args).cap_rpm == 0.


@pytest.mark.parametrize("checked_age", [.05, .08, .12, .149])
def test_constant_motion_does_not_pay_processing_delay_twice(checked_age):
    model = SampleBrakingAssessment(
        1, 100., 100.+checked_age, 3., 84., 84., 100., 0.,
        1.1, .816814, 1., .15, 200., outer_allowance_rpm=10.,
        observed_feedback_reserve=True, feedback_interval_covered=True)
    result = model.budget(100.25, 84., authorized_rpm=74., execution_bound_rpm=84.)
    speed = 84.*.816814/60.
    assert result.margin_m == pytest.approx(3.-1.1-speed*(.25+.05)-.02)
    assert result.required_stop_m == pytest.approx(speed*.15+speed**2/2.)
    assert result.cap_rpm == 74.


@pytest.mark.parametrize("feedback_offset,reserve", [
    (-.12, .12), (-.06, .06), (-.03, .05), (0., .05), (.04, .05),
])
def test_pre_depth_lag_and_execution_period_floor_are_retained(feedback_offset, reserve):
    model = SampleBrakingAssessment(
        1, 100., 100.02 if feedback_offset < 0 else 100.05, 3., 40., 40.,
        100.+feedback_offset, 0., 1.1, .816814, 1., .15, 200.,
        observed_feedback_reserve=True, feedback_interval_covered=True)
    assert model.effective_feedback_reserve_sec == pytest.approx(reserve)


@pytest.mark.parametrize("distance", [1.1, 1.4, 1.8, 2.2, 2.6, 3.5])
@pytest.mark.parametrize("rpm", [0., 20., 60., 80., 120., 180.])
def test_old_grant_cannot_accelerate_or_refund_travel(distance, rpm):
    model = SampleBrakingAssessment(
        1, 100., 100.12, distance, rpm, rpm, 100.01, 0.,
        1.1, .816814, 1., .15, 200., observed_feedback_reserve=True,
        feedback_interval_covered=True)
    approved = math.floor(model.budget(model.checked_at, rpm).cap_rpm)
    budgets = [model.budget(now, max(0., rpm-i*5), authorized_rpm=approved,
                           completed_rpm=rpm)
               for i, now in enumerate([100.12, 100.18, 100.25, 100.299])]
    assert all(0 <= b.cap_rpm <= approved for b in budgets)
    assert all(a.cap_rpm >= b.cap_rpm and a.margin_m >= b.margin_m
               for a, b in zip(budgets, budgets[1:]))
    assert model.effective_feedback_reserve_sec == .05


@pytest.mark.parametrize("fault", ["near", "fast", "expired", "higher_write", "negative_target"])
def test_overlap_correction_does_not_relax_real_stop_or_history_guards(fault):
    model = cap135()
    now, rpm = model.checked_at, 76.
    kwargs = dict(authorized_rpm=74., execution_bound_rpm=84.)
    if fault == "near": model = replace(model, distance_m=1.1)
    elif fault == "fast": rpm = 150.
    elif fault == "expired": now = model.sample_timestamp+.300001
    elif fault == "higher_write": kwargs["completed_rpm"] = 84.01
    else:
        with pytest.raises(ValueError):
            replace(model, target_speed_m_s=-.2)
        return
    assert model.budget(now, rpm, **kwargs).cap_rpm == 0.


def test_cap215_high_speed_stop_is_not_relabelled_as_a_minor_margin_error():
    model = SampleBrakingAssessment(
        1, 100., 100.083904, 2.801192, 121., 121., 100.0717, 0.,
        1.1, .816814, 1., .15, 200., outer_allowance_rpm=10.,
        observed_feedback_reserve=True, feedback_interval_covered=True)
    result = model.budget(model.checked_at, 121.)
    assert result.reason == "shared_braking_momentum" and result.cap_rpm == 0.
    assert result.required_stop_m-result.margin_m > .14


@pytest.mark.parametrize("history", [None, -1., True, float("nan"), float("inf"), 206.])
def test_controller_without_valid_history_keeps_the_original_reserve(setup, history):
    clock, controller, frame = configured(setup,
        distance_target_motion_control_enable=False,
        distance_pi_observed_feedback_reserve=True,
        distance_pi_feedback_interval_deduplication=True,
        distance_pi_braking_stop_distance_m=1.1,
        distance_approach_deceleration_m_s2=1., distance_approach_response_delay_sec=.15)
    controller._braking_execution_bound_reader = lambda uid, now: 76.
    controller._braking_interval_speed_bound_reader = lambda uid, stamp, now: history
    current = frame(2.22538, rpm=76., stamp=clock.now-.12)
    current = replace(current, steering_feedback=replace(
        current.steering_feedback, timestamp=clock.now-.12))
    step(controller, current)
    model = controller.last_distance_pid_result.pi_braking_assessment
    assert not model.feedback_interval_covered
    assert model.effective_feedback_reserve_sec == pytest.approx(.12)


@pytest.mark.parametrize("enabled,observed,history", [
    (True, True, 76.), (True, True, 110.), (False, True, 76.), (True, False, 76.),
])
def test_controller_freezes_history_bound_and_policy_with_original_sample(setup, enabled, observed, history):
    clock, controller, frame = configured(setup,
        distance_target_motion_control_enable=False,
        distance_pi_observed_feedback_reserve=observed,
        distance_pi_feedback_interval_deduplication=enabled,
        distance_pi_braking_stop_distance_m=1.1,
        distance_approach_deceleration_m_s2=1., distance_approach_response_delay_sec=.15)
    calls = []
    controller._braking_execution_bound_reader = lambda uid, now: 76.
    controller._braking_interval_speed_bound_reader = lambda uid, stamp, now: calls.append(
        (uid, stamp, now)) or history
    current = frame(2.22538, rpm=76., stamp=clock.now-.12)
    current = replace(current, steering_feedback=replace(
        current.steering_feedback, timestamp=clock.now-.12))
    step(controller, current)
    model = controller.last_distance_pid_result.pi_braking_assessment
    assert model.feedback_interval_covered is (enabled and observed)
    assert calls == ([(1, current.distance_state.sample_timestamp, clock.now)] if enabled and observed else [])
    assert model.travel_bound_rpm == max(76., history if enabled and observed else 0.)
    assert model.effective_feedback_reserve_sec == pytest.approx(.05 if enabled and observed
                                                               else .12 if observed else .15)


def test_standalone_pi_never_invents_interval_coverage_from_an_enabled_switch():
    pi = DistancePiController(DistancePiConfig(
        use_target_motion=False, observed_feedback_reserve=True,
        feedback_interval_deduplication=True, deceleration_m_s2=1.,
        response_delay_sec=.15, stationary_stop_preview_distance_m=1.1))
    result = pi.update(2.5, 1.4, sample_timestamp=100., execution_now=100.12,
        deadband_m=.03, max_output_rpm=200.,
        ego_forward_rpm=60., preview_outer_forward_rpm=60.,
        preview_feedback_timestamp=100., raw_distance_m=2.5)
    assert result.braking_assessment is None  # No publishable shared proof.
    legacy = SampleBrakingAssessment(1, 100., 100.12, 2.5, 60., 60., 100., 0.,
        1.1, .816814, 1., .15, 200., observed_feedback_reserve=True)
    assert result.stationary_preview_margin_m == pytest.approx(
        legacy.budget(100.12, 60.).margin_m)
