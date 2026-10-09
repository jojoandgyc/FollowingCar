"""New-depth stationary-stop preview is reduction-only and independently reversible."""

from dataclasses import replace

import pytest

from car_control_modular.depth_continuation import relative_braking_budget
from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence


CIRCUMFERENCE = .816814


def controller(*, preview, kp=3., launch=180., prior_rpm=120., suspended=False):
    config = DistancePiConfig(
        kp_per_sec=kp, ki_per_sec2=.4, wheel_circumference_m=CIRCUMFERENCE,
        deceleration_m_s2=.7, response_delay_sec=.2, physical_ttl_sec=.25,
        launch_request_rpm=launch, launch_full_error_m=.5,
        stationary_stop_preview_enabled=preview,
        stationary_stop_preview_distance_m=1.2)
    pi = DistancePiController(config)
    pi._last_sample_ts = 99.9
    pi._last_execution_ts = 99.98
    pi._last_target = 1.4
    pi._approved_rpm = pi._last_output_rpm = prior_rpm
    pi._execution_suspended = suspended
    return pi


def update(pi, *, stamp=100., distance=2.752, age=.0845, wheel=49.,
           target_speed=.685, completed=None, raw=True, feedback_age=.02,
           integral=None, qualified=True):
    if integral is not None:
        pi.integral_m_s = integral
    ego = wheel*CIRCUMFERENCE/60.
    rate = target_speed-ego
    return pi.update(
        distance, 1.4, sample_timestamp=stamp, execution_now=stamp+age,
        deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
        ego_forward_rpm=wheel,
        preview_outer_forward_rpm=wheel,
        preview_feedback_timestamp=stamp+age-feedback_age,
        preview_completed_rpm=completed,
        raw_distance_m=distance if raw else None,
        range_rate_m_s=rate, raw_closure_valid=qualified,
        raw_motion_evidence=(RawDepthMotionEvidence(
            stamp, rate, target_speed, .12, 3) if qualified else None))


@pytest.mark.parametrize('cap,distance,age,wheel,target,completed', [
    (139, 2.752, .0845, 49., .685, 120.),
    (141, 2.770, .0652, 44., .845, 120.),
])
def test_cap139_141_reduces_high_request_before_cap165(
        cap, distance, age, wheel, target, completed):
    old = update(controller(preview=False), distance=distance, age=age,
                 wheel=wheel, target_speed=target, completed=completed)
    trial = update(controller(preview=True), distance=distance, age=age,
                   wheel=wheel, target_speed=target, completed=completed)
    assert old.output_rpm >= 130
    budget = relative_braking_budget(
        distance_m=distance, stop_distance_m=1.2,
        speed_bound_m_s=max(wheel, completed)*CIRCUMFERENCE/60., age_sec=age,
        outer_speed_m_s=wheel*CIRCUMFERENCE/60., target_speed_m_s=0.,
        deceleration_m_s2=.7, response_delay_sec=.2,
        wheel_circumference_m=CIRCUMFERENCE, max_rpm=200.,
        same_grant_no_acceleration=False)
    # Keep the physical stopping budget, without the former extra 15RPM cut.
    assert trial.stationary_preview_cap_rpm == pytest.approx(budget.max_allowed_rpm)
    assert trial.output_rpm == int(budget.max_allowed_rpm)
    assert trial.output_rpm < old.output_rpm
    assert trial.stationary_preview_loss_rpm > 40.
    assert trial.stationary_preview_status == 'bounded'
    assert trial.stationary_preview_required_stop_m < trial.stationary_preview_margin_m


def test_unissued_p_or_launch_does_not_spend_past_distance():
    # Same physical measurement and completed command: changing an as-yet
    # unissued PI request must not lower the PREVIEW stopping-speed envelope.
    slow = update(controller(preview=True, kp=1., launch=0., prior_rpm=0.),
                  completed=None)
    fast = update(controller(preview=True, kp=3., launch=180., prior_rpm=0.),
                  completed=None)
    assert fast.pi_demand_rpm > slow.pi_demand_rpm
    assert fast.stationary_preview_cap_rpm == pytest.approx(
        slow.stationary_preview_cap_rpm)
    assert fast.stationary_preview_margin_m == pytest.approx(
        slow.stationary_preview_margin_m)


def test_expired_120_rpm_history_does_not_reenter_new_grant_budget():
    stopped_history = update(controller(preview=True, prior_rpm=120., suspended=True),
                             distance=3.2, wheel=0., target_speed=0., completed=None)
    clean_history = update(controller(preview=True, prior_rpm=0., suspended=True),
                           distance=3.2, wheel=0., target_speed=0., completed=None)
    assert stopped_history.stationary_preview_cap_rpm == pytest.approx(
        clean_history.stationary_preview_cap_rpm)
    assert stopped_history.stationary_preview_margin_m == pytest.approx(
        clean_history.stationary_preview_margin_m)


@pytest.mark.parametrize('change', [
    {'raw': False}, {'feedback_age': .151}, {'feedback_age': -.001},
])
def test_unavailable_new_evidence_does_not_change_original_policy(change):
    old = update(controller(preview=False), **change)
    trial = update(controller(preview=True), **change)
    assert trial.stationary_preview_status == 'unavailable'
    assert trial.stationary_preview_cap_rpm is None
    assert trial.stationary_preview_loss_rpm == 0.
    assert trial.output_rpm == old.output_rpm
    assert trial.cap_rpm == pytest.approx(old.cap_rpm)


def test_missing_outer_feedback_does_not_invent_preview_permission():
    old = update(controller(preview=False))
    pi = controller(preview=True)
    original = pi.update

    def no_outer(*args, **kwargs):
        kwargs['preview_outer_forward_rpm'] = None
        return original(*args, **kwargs)

    pi.update = no_outer
    trial = update(pi)
    assert trial.output_rpm == old.output_rpm
    assert trial.stationary_preview_status == 'unavailable'


def test_same_stop_model_still_vetoes_fast_close_and_near_distance():
    fast = update(controller(preview=True), distance=2.35472, age=.1685,
                  wheel=85., target_speed=.378, completed=None)
    assert fast.stationary_preview_status == 'momentum_brake'
    assert fast.output_rpm == 0
    near = update(controller(preview=True), distance=1.19, age=.03,
                  wheel=20., target_speed=0., completed=None)
    assert near.output_rpm == 0
    assert near.stationary_preview_cap_rpm == 0.


def test_farther_new_depth_can_accelerate_after_earlier_preview_brake():
    pi = controller(preview=True, prior_rpm=95.)
    first = update(pi, distance=2.55, age=.05, wheel=80.,
                   target_speed=.4, completed=95.)
    second = update(pi, stamp=100.1, distance=3.2, age=.05, wheel=72.,
                    target_speed=1.2, completed=80.)
    assert 55 <= first.output_rpm <= 80
    assert second.output_rpm > first.output_rpm
    assert second.stationary_preview_status == 'bounded'
    assert not second.brake_settling_limited


@pytest.mark.parametrize('cap,distance,age,wheel,target,prior', [
    (519, 1.8322, .0366, 56., .3999, 53.),
    (567, 2.3672, .0977, 78., .5335, 76.),
    (618, 1.8866, .0296, 57., .2861, 55.),
])
def test_cap494_695_momentum_veto_survives_even_when_person_was_walking(
        cap, distance, age, wheel, target, prior):
    trial = update(controller(preview=True, prior_rpm=prior), distance=distance,
                   age=age, wheel=wheel, target_speed=target,
                   completed=prior, feedback_age=.02)
    assert trial.motion_window_used
    assert trial.stationary_preview_status == 'momentum_brake'
    assert trial.stationary_preview_required_stop_m > trial.stationary_preview_margin_m
    assert trial.output_rpm == 0


@pytest.mark.parametrize('distance,age,wheel,prior', [
    (1.8322, .0366, 56., 53.),
    (2.3672, .0977, 78., 76.),
    (1.8866, .0296, 57., 55.),
])
def test_walking_evidence_cannot_buy_future_stopping_space(
        distance, age, wheel, prior):
    stopped = update(controller(preview=True, prior_rpm=prior), distance=distance,
                     age=age, wheel=wheel, target_speed=0., completed=prior)
    walking = update(controller(preview=True, prior_rpm=prior), distance=distance,
                     age=age, wheel=wheel, target_speed=.8, completed=prior)
    assert stopped.stationary_preview_cap_rpm == walking.stationary_preview_cap_rpm == 0.
    assert stopped.stationary_preview_margin_m == pytest.approx(
        walking.stationary_preview_margin_m)
    assert stopped.stationary_preview_required_stop_m == pytest.approx(
        walking.stationary_preview_required_stop_m)
    assert stopped.output_rpm == walking.output_rpm == 0


def test_cap539_unqualified_motion_remains_stationary_hard_brake():
    trial = update(controller(preview=True, prior_rpm=62.), distance=2.0304,
                   age=.0115, wheel=75., target_speed=0., completed=62.,
                   feedback_age=.02, qualified=False)
    assert not trial.motion_window_used
    assert trial.stationary_preview_status == 'momentum_brake'
    assert trial.output_rpm == 0
    assert trial.stationary_preview_required_stop_m > trial.stationary_preview_margin_m


def test_cap596_old_motion_window_cannot_keep_spending_target_speed():
    trial = update(controller(preview=True, prior_rpm=73.), distance=2.3416,
                   age=.1523, wheel=77., target_speed=.448, completed=73.,
                   feedback_age=.02)
    assert trial.stationary_preview_status == 'momentum_brake'
    assert trial.output_rpm == 0


def test_new_safe_depth_window_releases_only_preview_zero_settling():
    pi = controller(preview=True, prior_rpm=62.)
    stopped = update(pi, distance=2.0304, age=.0115, wheel=75.,
                     target_speed=0., completed=62., qualified=False)
    resumed = update(pi, stamp=100.1, distance=2.02, age=.04,
                     wheel=40., target_speed=.5)
    assert stopped.output_rpm == 0
    assert resumed.brake_settling_preview_released
    assert not resumed.brake_settling_limited
    assert resumed.output_rpm > 0

    pi = controller(preview=True, prior_rpm=62.)
    update(pi, distance=2.0304, age=.0115, wheel=75., target_speed=0.,
           completed=62., qualified=False)
    unverified = update(pi, stamp=100.1, distance=2.02, age=.04,
                        wheel=40., target_speed=.5, qualified=False)
    assert not unverified.brake_settling_preview_released
    assert unverified.output_rpm == 0


def test_new_walking_evidence_does_not_release_brake_while_wheels_still_fast():
    pi = controller(preview=True, prior_rpm=62.)
    first = update(pi, distance=2.0304, age=.0115, wheel=75.,
                   target_speed=0., completed=62., qualified=False)
    still_fast = update(pi, stamp=100.1, distance=2.02, age=.04,
                        wheel=75., target_speed=.8, completed=62.)
    assert first.output_rpm == 0
    assert still_fast.stationary_preview_status == 'momentum_brake'
    assert still_fast.output_rpm == 0
    assert not still_fast.brake_settling_preview_released


def test_following_person_can_stop_between_samples_without_borrowed_motion_credit():
    pi = controller(preview=True, prior_rpm=40.)
    walking = update(pi, distance=1.89, age=.03, wheel=45.,
                     target_speed=.5, completed=40.)
    stopped = update(pi, stamp=100.1, distance=1.8322, age=.0366,
                     wheel=56., target_speed=0., completed=44.)
    assert walking.output_rpm > 0
    assert stopped.stationary_preview_status == 'momentum_brake'
    assert stopped.output_rpm == 0
    assert not stopped.brake_settling_preview_released


def test_expired_encoder_feedback_cannot_release_preview_brake_wait():
    pi = controller(preview=True, prior_rpm=62.)
    stopped = update(pi, distance=2.0304, age=.0115, wheel=75.,
                     target_speed=0., completed=62., qualified=False)
    stale = update(pi, stamp=100.1, distance=2.02, age=.04,
                   wheel=40., target_speed=.5, feedback_age=.151)
    assert stopped.output_rpm == 0
    assert stale.stationary_preview_status == 'unavailable'
    assert stale.output_rpm == 0
    assert not stale.brake_settling_preview_released


def test_at_target_zero_does_not_create_preview_release_anchor():
    pi = controller(preview=True, prior_rpm=23.)
    stopped = update(pi, distance=1.4, age=.02, wheel=34.,
                     target_speed=0., completed=23.)
    assert stopped.output_rpm == 0
    assert stopped.stationary_preview_loss_rpm == 0
    assert not pi._preview_zero_brake_anchor


def test_preview_brake_freezes_positive_pi_windup():
    pi = controller(preview=True, prior_rpm=120.)
    before = .2
    result = update(pi, distance=2.752, wheel=49., completed=120., integral=before)
    assert result.stationary_preview_loss_rpm > 0.
    assert result.integral_frozen
    assert pi.integral_m_s <= before


def test_opt_out_is_numerically_equivalent_even_with_preview_arguments():
    baseline = update(controller(preview=False))
    alternate = update(controller(preview=False), completed=120.)
    assert replace(baseline, stationary_preview_status='disabled') == alternate


def test_new_sample_can_accelerate_beyond_past_speed_when_stop_budget_allows():
    inputs = dict(distance_m=3.2, stop_distance_m=1.2, speed_bound_m_s=80.*CIRCUMFERENCE/60.,
                  age_sec=.05, outer_speed_m_s=72.*CIRCUMFERENCE/60., target_speed_m_s=0.,
                  deceleration_m_s2=.7, response_delay_sec=.2,
                  wheel_circumference_m=CIRCUMFERENCE, max_rpm=200.)
    old_grant = relative_braking_budget(**inputs)
    new_sample = relative_braking_budget(**inputs, same_grant_no_acceleration=False)
    assert old_grant.max_allowed_rpm == pytest.approx(80.)
    assert new_sample.max_allowed_rpm > old_grant.max_allowed_rpm
    assert new_sample.margin_m == pytest.approx(old_grant.margin_m)


def test_preview_uses_fixed_old_grant_feedback_reserve():
    inputs = dict(distance_m=1.8322, stop_distance_m=1.2,
                  speed_bound_m_s=56*CIRCUMFERENCE/60., age_sec=.0366,
                  outer_speed_m_s=56*CIRCUMFERENCE/60., target_speed_m_s=0.,
                  deceleration_m_s2=.7, response_delay_sec=.2,
                  wheel_circumference_m=CIRCUMFERENCE, max_rpm=200.)
    budget = relative_braking_budget(**inputs, same_grant_no_acceleration=False)
    expected = (inputs['distance_m']-inputs['stop_distance_m']
                -inputs['speed_bound_m_s']*(inputs['age_sec']+.15)-.02)
    assert budget.margin_m == pytest.approx(expected)
    assert budget.max_allowed_rpm == 0.
