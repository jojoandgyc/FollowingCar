"""Independent range checks for the shared sample budget; no motor access."""
from dataclasses import replace
import math

import pytest

from car_control_modular.sample_braking import SampleBrakingAssessment


def sample(distance, bound, target=0.0):
    return SampleBrakingAssessment(
        uid=1, sample_timestamp=100., checked_at=100.06,
        distance_m=distance, travel_bound_rpm=bound, outer_rpm=bound,
        feedback_timestamp=100.04, target_speed_m_s=target,
        stop_distance_m=1.2, circumference_m=math.pi*.26,
        deceleration_m_s2=.7, response_delay_sec=.2, max_rpm=200.)


@pytest.mark.parametrize("distance", [1.3, 1.5, 1.8, 2.2, 2.6, 4.0])
@pytest.mark.parametrize("bound", [0., 10., 30., 60., 100., 180.])
@pytest.mark.parametrize("target", [0., -.2, -.5])
def test_age_cannot_increase_fixed_grant_ceiling_or_margin(distance, bound, target):
    evidence = sample(distance, bound, target)
    fresh = evidence.budget(evidence.checked_at, bound)
    approved = math.floor(fresh.cap_rpm)
    budgets = [evidence.budget(now, bound, authorized_rpm=approved,
                               completed_rpm=bound)
               for now in [100.06, 100.08, 100.12, 100.18, 100.22, 100.249,
                           100.275, 100.299]]
    assert all(math.isfinite(b.cap_rpm) and 0 <= b.cap_rpm <= approved for b in budgets)
    assert all(a.cap_rpm >= b.cap_rpm for a, b in zip(budgets, budgets[1:]))
    assert all(a.margin_m >= b.margin_m for a, b in zip(budgets, budgets[1:]))
    assert evidence.budget(100.300001, 0., authorized_rpm=approved).cap_rpm == 0


@pytest.mark.parametrize("distance", [1.4, 1.8, 2.4, 3.5])
@pytest.mark.parametrize("bound", [0., 20., 50., 90.])
def test_approaching_target_cannot_get_more_speed_than_stationary_target(distance, bound):
    evidence = sample(distance, bound)
    caps = [replace(evidence, target_speed_m_s=target).budget(100.06, bound).cap_rpm
            for target in [0., -.1, -.3, -.6]]
    assert all(a >= b for a, b in zip(caps, caps[1:]))


@pytest.mark.parametrize("distance", [1.5, 1.8, 2.5, 4.])
def test_unexecuted_launch_has_no_retroactive_travel_charge(distance):
    evidence = sample(distance, 0.)
    budgets = []
    for age in [.02, .06, .10, .14]:
        changed = replace(evidence, checked_at=100.+age,
                          feedback_timestamp=100.+age-.01)
        budgets.append(changed.budget(changed.checked_at, 0.))
    assert all(b.cap_rpm == pytest.approx(budgets[0].cap_rpm) for b in budgets)
    assert all(b.margin_m == pytest.approx(budgets[0].margin_m) for b in budgets)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., True, None])
def test_malformed_commit_inputs_fail_closed(value):
    evidence = sample(2.5, 30.)
    assert evidence.budget(value, 20., authorized_rpm=30.).cap_rpm == 0
    assert evidence.budget(100.1, value, authorized_rpm=30.).cap_rpm == 0
    if value is not None:  # None explicitly means initial assessment, not a grant.
        assert evidence.budget(100.1, 20., authorized_rpm=value).cap_rpm == 0
        assert evidence.budget(100.1, 20., authorized_rpm=30., completed_rpm=value).cap_rpm == 0


def test_fresh_slower_feedback_cannot_shrink_accumulated_travel_charge():
    evidence = sample(4., 60.)
    authorized = 40.
    same_time = [evidence.budget(100.2, outer, authorized_rpm=authorized,
                               completed_rpm=outer) for outer in [60., 30., 0.]]
    assert len({b.margin_m for b in same_time}) == 1
    assert len({b.cap_rpm for b in same_time}) == 1
    assert evidence.travel_bound_rpm == 60.


def test_same_numeric_speed_does_not_license_a_higher_intervening_write():
    evidence = sample(4., 30.)
    allowed = evidence.budget(100.1, 20., authorized_rpm=40., completed_rpm=40.)
    revoked = evidence.budget(100.1, 20., authorized_rpm=40., completed_rpm=41.)
    assert allowed.cap_rpm > 0
    assert revoked.cap_rpm == 0
    assert revoked.reason == "shared_braking_new_higher_write"
