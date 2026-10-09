"""Command reductions do not rewrite a physical sample's admitted history."""
from dataclasses import replace
import math

import pytest

from car_control_modular.sample_braking import SampleBrakingAssessment


@pytest.fixture
def assessment():
    return SampleBrakingAssessment(
        uid=1, sample_timestamp=100., checked_at=100.02,
        distance_m=3.5, travel_bound_rpm=20., outer_rpm=20.,
        feedback_timestamp=100.02, target_speed_m_s=0.,
        stop_distance_m=1.2, circumference_m=math.pi*.26,
        deceleration_m_s2=.7, response_delay_sec=.2, max_rpm=200.,
        outer_allowance_rpm=10.)


def test_lower_command_preserves_legal_completed_write_and_travel(assessment):
    original = assessment.budget(100.07, 80., authorized_rpm=80.,
                                 execution_bound_rpm=90., completed_rpm=80.)
    reduced = assessment.budget(100.07, 80., authorized_rpm=50.,
                                execution_bound_rpm=90., completed_rpm=80.)
    assert original.cap_rpm == 80.
    assert reduced.cap_rpm == 50.
    assert reduced.reason == "shared_braking_cap"
    assert reduced.margin_m == original.margin_m
    assert reduced.required_stop_m == original.required_stop_m


def test_completed_write_above_original_outer_budget_still_rejected(assessment):
    result = assessment.budget(100.07, 80., authorized_rpm=50.,
                               execution_bound_rpm=90., completed_rpm=91.)
    assert result.cap_rpm == 0.
    assert result.reason == "shared_braking_new_higher_write"


def test_missing_original_budget_keeps_legacy_higher_write_check(assessment):
    result = assessment.budget(100.07, 80., authorized_rpm=50., completed_rpm=80.)
    assert result.cap_rpm == 0.
    assert result.reason == "shared_braking_new_higher_write"


@pytest.mark.parametrize("bound", [-1., float("nan"), float("inf"), True, "90", 59., 200.])
def test_execution_budget_cannot_be_malformed_smaller_or_invented(assessment, bound):
    result = assessment.budget(100.07, 20., authorized_rpm=50.,
                               execution_bound_rpm=bound, completed_rpm=20.)
    assert result.cap_rpm == 0.
    assert result.reason in {"invalid_shared_braking_evidence",
                             "invalid_shared_braking_execution_bound"}


def test_execution_budget_cannot_shrink_capture_travel_bound(assessment):
    assessment = replace(assessment, travel_bound_rpm=90., outer_rpm=90.)
    result = assessment.budget(100.07, 20., authorized_rpm=20.,
                               execution_bound_rpm=30., completed_rpm=20.)
    assert result.cap_rpm == 0.
    assert result.reason == "invalid_shared_braking_execution_bound"


def test_new_execution_budget_requires_a_grant(assessment):
    result = assessment.budget(100.02, 20., execution_bound_rpm=90.)
    assert result.cap_rpm == 0.
    assert result.reason == "invalid_shared_braking_evidence"


@pytest.mark.parametrize("distance", [-1., 0., float("nan"), float("inf"), True, "3", 3.501])
def test_late_distance_cannot_extend_or_invalidate_numeric_budget(assessment, distance):
    result = assessment.budget(100.07, 20., authorized_rpm=50.,
                               execution_bound_rpm=90., tightened_distance_m=distance)
    assert result.cap_rpm == 0.
    assert result.reason == "invalid_shared_braking_evidence"


def test_one_millimeter_closer_uses_same_feedback_qualification(assessment):
    assessment = replace(assessment, feedback_timestamp=99.90)
    original = assessment.budget(100.07, 80., authorized_rpm=50.,
                                 execution_bound_rpm=90., completed_rpm=80.)
    closer = assessment.budget(100.07, 80., authorized_rpm=50.,
                               execution_bound_rpm=90., completed_rpm=80.,
                               tightened_distance_m=3.499)
    assert original.cap_rpm == closer.cap_rpm == 50.
    assert closer.margin_m == pytest.approx(original.margin_m-.001)
    assert assessment.feedback_timestamp == 99.90
    assert assessment.distance_m == 3.5
    assert assessment.valid_for(1, 100.)


def test_closer_distance_reduces_cap_not_original_admission(assessment):
    result = assessment.budget(100.07, 20., authorized_rpm=80.,
                               execution_bound_rpm=90., completed_rpm=80.,
                               tightened_distance_m=2.)
    assert 0. < result.cap_rpm < 80.
    assert result.reason == "shared_braking_cap"
    # A new assessment at 2m would never admit the old 80RPM. Tightening must
    # therefore not replace the original assessment before validating history.
    replaced = replace(assessment, distance_m=2.).budget(
        100.07, 20., authorized_rpm=80., completed_rpm=80.)
    assert replaced.reason == "shared_braking_request_exceeds_plan"


@pytest.mark.parametrize("now", [100.02, 100.07, 100.15, 100.249])
@pytest.mark.parametrize("outer", [0., 20., 50., 80.])
def test_closer_never_increases_cap_or_refunds_travel(assessment, now, outer):
    budgets = [assessment.budget(now, outer, authorized_rpm=50.,
                                execution_bound_rpm=90., completed_rpm=80.,
                                tightened_distance_m=distance)
               for distance in [3.5, 3.499, 3., 2.5, 2., 1.5, 1.1]]
    assert all(a.cap_rpm >= b.cap_rpm for a, b in zip(budgets, budgets[1:]))
    assert all(a.margin_m >= b.margin_m for a, b in zip(budgets, budgets[1:]))


def test_newer_feedback_or_reduced_command_cannot_refund_history(assessment):
    budgets = [assessment.budget(100.2, 0., authorized_rpm=command,
                                execution_bound_rpm=90., completed_rpm=0.)
               for command in [80., 50., 20., 0.]]
    assert len({budget.margin_m for budget in budgets}) == 1
    assert all(a.cap_rpm >= b.cap_rpm for a, b in zip(budgets, budgets[1:]))


def test_original_deadline_still_applies_after_distance_or_command_tightening(assessment):
    result = assessment.budget(100.300001, 0., authorized_rpm=20.,
                               execution_bound_rpm=90., completed_rpm=0.,
                               tightened_distance_m=3.499)
    assert result.cap_rpm == 0.
    assert result.reason == "invalid_shared_braking_evidence"


def test_original_admission_still_applies_with_execution_budget(assessment):
    result = assessment.budget(100.07, 20., authorized_rpm=150.,
                               execution_bound_rpm=160., completed_rpm=20.)
    assert result.cap_rpm == 0.
    assert result.reason == "shared_braking_request_exceeds_plan"


def test_fresh_distance_tightening_equivalent_to_closer_fresh_sample(assessment):
    for distance in [3.5, 3., 2., 1.3]:
        tightened = assessment.budget(100.02, 20., tightened_distance_m=distance)
        fresh = replace(assessment, distance_m=distance).budget(100.02, 20.)
        assert tightened.cap_rpm == pytest.approx(fresh.cap_rpm)
        assert tightened.margin_m == pytest.approx(fresh.margin_m)
        assert tightened.required_stop_m == fresh.required_stop_m
        assert tightened.reason == fresh.reason
