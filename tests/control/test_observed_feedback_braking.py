"""Frozen measured-age reserve: CAP169 replay and physical-budget invariants.

Offline only. Passing these checks is not a measured stopping-distance claim.
"""
from dataclasses import replace
import math

import pytest

from car_control_modular.sample_braking import SampleBrakingAssessment
from test_distance_pi_controller import configured, step
from test_distance_tracking_response import setup


def cap169(observed=True):
    return SampleBrakingAssessment(
        uid=1, sample_timestamp=6290.482592609, checked_at=6290.624331163,
        distance_m=2.348, travel_bound_rpm=88., outer_rpm=79.,
        feedback_timestamp=6290.572344931, target_speed_m_s=0.,
        stop_distance_m=1.1, circumference_m=.816814, deceleration_m_s2=1.,
        response_delay_sec=.15, max_rpm=200., outer_allowance_rpm=10.,
        observed_feedback_reserve=observed)


def test_cap169_measured_reserve_keeps_legal_forward_instead_of_momentum_stop():
    old, new = cap169(False), cap169(True)
    now = old.sample_timestamp+.203608537
    params = dict(authorized_rpm=72., execution_bound_rpm=88.)
    before, after = old.budget(now, 83., **params), new.budget(now, 83., **params)
    assert before.reason == "shared_braking_momentum" and before.cap_rpm == 0
    assert before.margin_m == pytest.approx(.804379141473)
    assert after.reason == "shared_braking_cap" and after.cap_rpm == 72
    assert after.required_stop_m == before.required_stop_m
    assert after.required_stop_m < after.margin_m
    assert new.effective_feedback_reserve_sec == pytest.approx(.051986232)
    assert new.stop_distance_m == 1.1 and new.deceleration_m_s2 == 1.


@pytest.mark.parametrize("age,expected", [(0., .05), (.01, .05), (.05, .05), (.09, .09), (.149, .149)])
def test_reserve_is_frozen_at_admission_and_never_below_one_execution_period(age, expected):
    model = replace(cap169(), feedback_timestamp=cap169().checked_at-age)
    assert model.effective_feedback_reserve_sec == pytest.approx(expected)
    before = model.budget(model.checked_at, 79.)
    for dt in (.0, .025, .07, .12):
        now = model.checked_at+dt
        late = model.budget(now, 70., authorized_rpm=min(60., before.cap_rpm),
                            execution_bound_rpm=88.)
        assert model.effective_feedback_reserve_sec == pytest.approx(expected)
        assert late.cap_rpm <= 60.


@pytest.mark.parametrize("distance", [1.15, 1.5, 1.8, 2.2, 2.6, 4.])
@pytest.mark.parametrize("rpm", [0., 10., 30., 60., 100., 180.])
def test_old_grant_never_accelerates_or_recovers_spent_space(distance, rpm):
    model = SampleBrakingAssessment(
        1, 100., 100.06, distance, rpm, rpm, 100.04, 0., 1.1,
        .816814, 1., .15, 200., observed_feedback_reserve=True)
    fresh = model.budget(model.checked_at, rpm)
    approved = math.floor(fresh.cap_rpm)
    rows = [model.budget(now, max(0., rpm-i*2), authorized_rpm=approved,
                         completed_rpm=rpm)
            for i, now in enumerate((100.06, 100.12, 100.18, 100.25, 100.299))]
    assert all(0 <= x.cap_rpm <= approved for x in rows)
    assert all(a.cap_rpm >= b.cap_rpm and a.margin_m >= b.margin_m
               for a, b in zip(rows, rows[1:]))
    assert model.budget(100.301, 0., authorized_rpm=approved).cap_rpm == 0


@pytest.mark.parametrize("bad", ["too_close", "too_fast", "higher_write", "late_feedback", "future_feedback", "bad_switch"])
def test_more_responsive_policy_does_not_remove_physical_protections(bad):
    model = cap169()
    if bad in {"late_feedback", "future_feedback", "bad_switch"}:
        patch = ({"feedback_timestamp": model.checked_at-.151} if bad == "late_feedback"
                 else {"feedback_timestamp": model.checked_at+.001} if bad == "future_feedback"
                 else {"observed_feedback_reserve": 1})
        with pytest.raises(ValueError):
            replace(model, **patch)
        return
    if bad == "too_close":
        model = replace(model, distance_m=1.1)
    result = model.budget(model.checked_at, 160. if bad == "too_fast" else 79.,
                           authorized_rpm=72., execution_bound_rpm=88.,
                           completed_rpm=100. if bad == "higher_write" else 72.)
    assert result.cap_rpm == 0


@pytest.mark.parametrize("observed", [False, True])
def test_real_pi_controller_uses_the_selected_shared_policy(setup, observed):
    clock, controller, frame = configured(
        setup, distance_target_motion_control_enable=False,
        distance_pi_observed_feedback_reserve=observed,
        distance_pi_braking_stop_distance_m=1.1,
        distance_approach_deceleration_m_s2=1., distance_approach_response_delay_sec=.15)
    controller._braking_execution_bound_reader = lambda uid, now: 80.
    current = frame(2.348, rpm=70., stamp=clock.now-.10)
    current = replace(current, steering_feedback=replace(current.steering_feedback,
                                                         timestamp=clock.now-.02))
    step(controller, current)
    model = controller.last_distance_pid_result.pi_braking_assessment
    assert model.observed_feedback_reserve is observed
    assert model.effective_feedback_reserve_sec == pytest.approx(.05 if observed else .15)
