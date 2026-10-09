"""Current own-wheel evidence is not a target-velocity endpoint pair."""
from dataclasses import replace

import pytest

from car_control_modular.sample_braking import SampleBrakingAssessment, sample_feedback_time_valid
from test_depth_authority_250 import authority, decide_commit
from test_distance_pi_controller import configured, step
from test_distance_tracking_response import setup
from test_fresh_distance_restart_progress import prepare
from test_lateral_zero_runtime import owner


@pytest.mark.parametrize("age", [.149, .151, .1692, .179])
@pytest.mark.parametrize("feedback_age", [.001, .020, .149])
def test_real_controller_admits_independently_fresh_feedback(
        authority, setup, monkeypatch, age, feedback_age):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, distance=1.958, pair=(1., 2.),
        age=age, feedback_age=feedback_age)
    _, actions, accepted = decide_commit(a, frame)
    result = a.controller.last_distance_pid_result
    model = result.pi_braking_assessment
    assert model.current_feedback_independent
    assert model.feedback_timestamp == frame.steering_feedback.timestamp
    assert result.pi_stationary_preview_status == "bounded"
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(model.sample_timestamp+.30)
    assert a.owner._depth30_linear_timing.continuation_distance_m == pytest.approx(1.958)


@pytest.mark.parametrize("pure,valid", [(False, False), (True, True)])
def test_legacy_target_velocity_endpoint_still_requires150ms_pairing(setup, pure, valid):
    clock, c, frame = configured(setup, distance_target_motion_control_enable=not pure)
    current = frame(2.5, rpm=10., stamp=clock.now-.1692)
    current = replace(current, steering_feedback=replace(current.steering_feedback, timestamp=clock.now-.001))
    step(c, current)
    assert (c._distance_pi_ego_forward_rpm is not None) is valid


@pytest.mark.parametrize("sample_age,feedback_age", [(.181, .001), (.1, .151), (.1, -.001), (-.001, .01)])
def test_separate_clocks_do_not_extend_either_physical_lifetime(sample_age, feedback_age):
    assert not sample_feedback_time_valid(100.-sample_age, 100.-feedback_age, 100.,
                                          current_feedback_independent=True)


def assessment(**changes):
    args = dict(uid=1, sample_timestamp=100., checked_at=100.1692, distance_m=2.4,
        travel_bound_rpm=100., outer_rpm=2., feedback_timestamp=100.1682,
        target_speed_m_s=0., stop_distance_m=1.1, circumference_m=.816814,
        deceleration_m_s2=1., response_delay_sec=.15, max_rpm=200.,
        observed_feedback_reserve=True, feedback_interval_covered=True,
        current_feedback_independent=True)
    args.update(changes)
    return SampleBrakingAssessment(**args)


def test_fresher_low_feedback_does_not_erase_high_interval_travel():
    model = assessment()
    old_pair = replace(model, feedback_timestamp=100.14, current_feedback_independent=False)
    current = model.budget(model.checked_at, 2.)
    old = old_pair.budget(old_pair.checked_at, 2.)
    assert current.cap_rpm == old.cap_rpm
    assert current.margin_m == old.margin_m
    assert model.travel_bound_rpm == 100.
    assert model.effective_feedback_reserve_sec == .05
    with pytest.raises(ValueError):
        replace(model, current_feedback_independent=False)
    with pytest.raises(ValueError):
        replace(model, target_speed_m_s=-.1)
    assert model.budget(100.300001, 2., authorized_rpm=12.).cap_rpm == 0.


def test_controller_keeps_complete_execution_history_with_current_feedback(authority, setup, monkeypatch):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, distance=2.4, pair=(1., 2.), age=.1692, feedback_age=.001)
    profile = a.controller._distance_pid._distance_pi
    monkeypatch.setattr(profile, "config", replace(profile.config, feedback_interval_deduplication=True))
    a.controller._braking_execution_bound_reader = lambda *_: 80.
    a.controller._braking_interval_speed_bound_reader = lambda *_: 100.
    a.feedback = frame.steering_feedback
    a.controller.decide(10, frame, longitudinal_only=True)
    model = a.controller.last_distance_pid_result.pi_braking_assessment
    assert model.current_feedback_independent and model.feedback_interval_covered
    assert model.outer_rpm == 2. and model.travel_bound_rpm == 100.
    assert model.feedback_timestamp-frame.distance_state.sample_timestamp > .15


@pytest.mark.parametrize("flag", [None, 1, "yes"])
def test_clock_mode_flag_requires_boolean(flag):
    with pytest.raises(ValueError):
        assessment(current_feedback_independent=flag)
