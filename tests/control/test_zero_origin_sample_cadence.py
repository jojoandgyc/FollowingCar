"""Fresh range cadence cannot repeatedly expire a nonexistent positive grant.

CAP209 -> CAP213 had 199.8 ms between physical samples plus 66 ms processing
age. Both samples were fresh, the previous request was zero and current wheel
motion was zero, yet the old positive-lease test restarted the ramp at zero.
These checks exercise only pure control and fake-clock range admission.
"""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.sample_braking import SampleBrakingAssessment
from test_depth_authority_250 import authority, advance, decide_commit, writer
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def controller():
    return DistancePiController(DistancePiConfig(
        kp_per_sec=3., physical_ttl_sec=.25, fresh_update_max_age_sec=.18,
        retain_integral_sec=.35, use_target_motion=False,
        stationary_stop_preview_distance_m=1.1, launch_request_rpm=180.,
        launch_full_error_m=.5, deceleration_m_s2=.7))


def step(pi, stamp, *, age=.07, distance=1.8, wheel=0., **changes):
    now = stamp+age
    assessment = (SampleBrakingAssessment(
        1, stamp, now, distance, abs(wheel), abs(wheel), now,
        0., 1.1, .816814, .7, .2, 200., outer_allowance_rpm=10.)
        if 0 <= age <= .18 else None)
    args = dict(sample_timestamp=stamp, execution_now=now,
        deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
        ego_forward_rpm=wheel, raw_distance_m=distance,
        braking_assessment=assessment, preview_outer_forward_rpm=abs(wheel),
        preview_feedback_timestamp=now)
    args.update(changes)
    return pi.update(distance, 1.4, **args)


@pytest.mark.parametrize("gap,age", [(.20, .051), (.20, .07), (.20, .11), (.24, .07)])
def test_new_fresh_sample_can_leave_zero_when_old_zero_sample_age_exceeds_lease(gap, age):
    pi = controller()
    first = step(pi, 100., age=age)
    assert first.output_rpm == 0 and first.cap_rpm > 20
    result = step(pi, 100.+gap, age=age)
    assert result.cap_rpm > 20
    assert 0 < result.output_rpm <= 12
    assert result.output_rpm <= result.cap_rpm
    assert result.sample_dt_sec == 0  # No integration over the old lease gap.
    assert pi._last_sample_ts == 100.+gap


def test_live_zero_origin_keeps_existing_normal_ramp():
    pi = controller()
    assert step(pi, 100., age=.04).output_rpm == 0
    result = step(pi, 100.2, age=.04)
    assert result.output_rpm > 0
    assert result.output_rpm <= result.cap_rpm


def test_downstream_zero_limit_cannot_relabel_a_positive_request_as_zero_origin():
    pi = controller()
    first = step(pi, 100., wheel=8.)
    assert first.output_rpm > 0
    assert pi.accept_output_limit(100., 0.)
    assert pi._sample_requested_rpm > 0
    result = step(pi, 100.2)
    assert result.output_rpm == 0


@pytest.mark.parametrize("reason", ["emergency_stop", "identity_lost", "unknown"])
def test_actual_execution_suspension_is_not_relabelled_as_zero_origin(reason):
    pi = controller()
    step(pi, 100.)
    pi.suspend(100.15, reason, reset_execution=True)
    result = step(pi, 100.2)
    assert result.output_rpm == 0


@pytest.mark.parametrize("fault", ["park", "rejected", "jump", "near", "momentum",
    "reverse", "feedback_missing", "feedback_stale", "invalid_assessment", "stale_sample",
    "duplicate", "older"])
def test_zero_origin_cannot_override_new_evidence_or_stop_constraints(fault):
    pi = controller()
    step(pi, 100.)
    kwargs = {}
    stamp = 100.2
    if fault == "park":
        pi.set_normal_parking(True)
    elif fault == "rejected":
        pi.reject_output(100.)
    elif fault == "jump":
        kwargs["measurement_jump_clamped"] = True
    elif fault == "near":
        kwargs["distance"] = 1.41
    elif fault == "momentum":
        kwargs["wheel"] = 80.
    elif fault == "reverse":
        kwargs["wheel"] = -6.
    elif fault == "feedback_missing":
        kwargs["preview_outer_forward_rpm"] = None
    elif fault == "feedback_stale":
        kwargs["preview_feedback_timestamp"] = 100.
    elif fault == "invalid_assessment":
        kwargs["braking_assessment"] = object()
    elif fault == "stale_sample":
        kwargs["age"] = .181
    elif fault == "duplicate":
        stamp = 100.
        kwargs["age"] = .27
    elif fault == "older":
        stamp = 99.99
        kwargs["age"] = .28
    result = step(pi, stamp, **kwargs)
    assert result.output_rpm == 0
    if fault in {"stale_sample", "duplicate", "older"}:
        assert pi._last_sample_ts == 100.


def configure_authority(a, setup, *, rise=240.):
    _, a.controller, _ = configured(setup, target_distance_m=1.4,
        distance_pi_kp_per_sec=3., distance_pi_launch_request_rpm=180.,
        distance_pi_launch_full_error_m=.5, distance_target_motion_control_enable=False,
        depth_longitudinal_sample_max_age_sec=.25,
        distance_pi_braking_stop_distance_m=1.1,
        distance_pid_output_rise_rpm_per_sec=rise,
        distance_approach_deceleration_m_s2=.7)
    a.owner._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    a.controller._braking_execution_bound_reader = lambda uid, now: 0.


@pytest.mark.parametrize("fault", [None, "uid", "hazard", "obstacle"])
def test_current_identity_and_safety_still_govern_real_range_admission(authority, setup, fault):
    a = authority
    configure_authority(a, setup)
    advance(a, 100.07)
    first = a.frame(1.8, stamp=100., rpm=0.)
    _, actions, _ = decide_commit(a, first)
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    advance(a, 100.27)
    current = a.frame(1.8, stamp=100.2, rpm=0.)
    if fault == "uid":
        current = replace(current, persons=[replace(current.persons[0], track_id=2)])
    elif fault == "hazard":
        current = replace(current, hazard=replace(current.hazard, active=True))
    elif fault == "obstacle":
        current = replace(current, obstacles=replace(current.obstacles, front=True))
    _, actions, accepted = decide_commit(a, current)
    positive = any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    if fault is None:
        assert accepted and positive
        assert a.owner._depth30_linear_snapshot[3] == 100.2
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(100.45)
    else:
        assert not positive


def test_eight_fresh_samples_do_not_alternate_zero_but_expired_grants_still_stop(authority, setup):
    a = authority
    configure_authority(a, setup)
    action, backend = writer(a)
    action.get_steering_feedback = lambda: a.feedback
    outputs = []
    old_deadline = None
    for i in range(8):
        stamp, now = 100.+i*.2, 100.+i*.2+.07
        if old_deadline is not None:
            # There really is an approximately 20 ms gap before processing
            # the next fresh frame. It must not acquire blind continuation.
            advance(a, old_deadline+.001)
            assert a.owner._fresh_depth_linear_snapshot(1) is None
            action._service_follow_wheels()
            assert backend.pairs[-1][:2] == (0, 0)
        advance(a, now)
        current = a.frame(1.8, stamp=stamp, rpm=0.)
        _, actions, accepted = decide_commit(a, current)
        output = a.controller.last_distance_pid_result.output_rpm
        outputs.append(output)
        positive = any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
        if i == 0:
            assert output == 0 and not positive
            continue
        assert accepted and positive, outputs
        assert 0 < output <= 12., outputs
        assert a.owner._depth30_linear_snapshot[3] == stamp
        old_deadline = a.owner._depth30_linear_timing.depth_expires_at
        assert old_deadline == pytest.approx(stamp+.25)
        assert old_deadline > now
        action._service_follow_wheels()
        assert backend.pairs[-1][0] > 0 and backend.pairs[-1][1] < 0
    assert len(outputs) == 8 and all(value > 0 for value in outputs[1:])


@pytest.mark.parametrize("fault", ["stop", "identity_withdrawal", "unknown_withdrawal",
    "admission_rejected", "park", "near_filtered", "near_raw", "exact_start_distance",
    "negative_wheel", "small_pivot", "pivot", "stale_feedback"])
def test_positive_grant_expiry_restart_keeps_stop_and_current_sample_guards(authority, setup, fault):
    a = authority
    configure_authority(a, setup)
    # Establish a genuinely admitted positive grant without a fake historical
    # motor packet. This is distance-qualified restart, not receipt recovery.
    for i in range(2):
        advance(a, 100.+i*.2+.07)
        decide_commit(a, a.frame(1.8, stamp=100.+i*.2, rpm=0.))
    old = a.owner._depth30_linear_snapshot
    assert old is not None and old[1] > 0
    if fault in {"stop", "identity_withdrawal", "unknown_withdrawal"}:
        a.owner._revoke_depth_linear_authority(
            "emergency_stop" if fault == "stop" else
            "identity_lost" if fault == "identity_withdrawal" else "unknown")
    elif fault == "admission_rejected":
        a.controller.reject_longitudinal_sample(old[3], "admission_rejected")
    elif fault == "park":
        a.controller.set_normal_parking(True, 1)
    advance(a, 100.47)
    current = a.frame(1.8, stamp=100.4, rpm=0.)
    if fault == "near_filtered":
        current = replace(current, distance_m=1.45)
    elif fault == "near_raw":
        current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=1.45))
    elif fault == "exact_start_distance":
        current = a.frame(a.controller.cfg.forward_start_distance_m, stamp=100.4, rpm=0.)
    elif fault == "negative_wheel":
        current = replace(current, steering_feedback=replace(current.steering_feedback,
            left_forward_rpm=-4., right_forward_rpm=0.))
    elif fault == "small_pivot":
        current = replace(current, steering_feedback=replace(current.steering_feedback,
            left_forward_rpm=5., right_forward_rpm=-5.))
    elif fault == "pivot":
        current = replace(current, steering_feedback=replace(current.steering_feedback,
            left_forward_rpm=80., right_forward_rpm=-80.))
    elif fault == "stale_feedback":
        current = replace(current, steering_feedback=replace(current.steering_feedback,
            timestamp=100.319))
    a.feedback = current.steering_feedback
    a.controller.decide(10, current, longitudinal_only=True)
    result = a.controller.last_distance_pid_result
    assert result is None or not result.pi_fresh_grant_recovery_used


@pytest.mark.parametrize("feedback_age", [.099, .12, .149])
def test_stationary_restart_uses_same_feedback_age_contract_as_shared_braking(authority, setup, feedback_age):
    a = authority
    configure_authority(a, setup)
    for i in range(2):
        advance(a, 100.+i*.2+.07)
        decide_commit(a, a.frame(1.8, stamp=100.+i*.2, rpm=0.))
    advance(a, 100.47)
    current = a.frame(1.8, stamp=100.4, rpm=0.)
    current = replace(current, steering_feedback=replace(current.steering_feedback,
        timestamp=a.clock.now-feedback_age))
    _, actions, accepted = decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert result.pi_fresh_grant_recovery_used
    assert not result.pi_depth_expiry_recovery_used
    assert 0 < result.output_rpm <= 12.
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)


@pytest.mark.parametrize("rise", [60., 240., 1000., 0.])
def test_repeated_real_new_grants_obey_stationary_ramp_credit_cap(authority, setup, rise):
    a = authority
    configure_authority(a, setup, rise=rise)
    outputs = []
    for i in range(8):
        stamp = 100.+i*.2
        advance(a, stamp+.07)
        _, actions, accepted = decide_commit(a, a.frame(1.8, stamp=stamp, rpm=0.))
        result = a.controller.last_distance_pid_result
        outputs.append(result.output_rpm)
        if rise > 0 and i == 0:
            assert result.output_rpm == 0
            continue
        assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions), outputs
        assert 0 < result.output_rpm <= result.approach_cap_rpm
        if rise > 0:
            assert result.output_rpm <= min(12., rise*.05), outputs
            if i > 1:
                assert result.pi_fresh_grant_recovery_used
                assert result.pi_fresh_grant_recovery_step_sec == pytest.approx(.05)
                assert not result.pi_depth_expiry_recovery_used
        else:
            # Rise=0 still means the global software ramp is disabled. It
            # must not be silently changed to a 12RPM ramp or a forced stop.
            assert not result.pi_depth_expiry_recovery_used
            assert result.pi_depth_expiry_recovery_step_sec == 0.
            assert not result.pi_fresh_grant_recovery_used
            assert result.output_rpm > 12.
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp+.25)


@pytest.mark.parametrize("rise", [60., 240., 1000.])
def test_pure_zero_origin_uses_same_hard_increment_bound(rise):
    pi = controller()
    assert step(pi, 100., rise_rpm_per_sec=rise).output_rpm == 0
    result = step(pi, 100.2, rise_rpm_per_sec=rise)
    assert 0 < result.output_rpm <= min(12., rise*.05)


@pytest.mark.parametrize("invalid_rise", [0., -1., float("nan"), float("inf"), True])
def test_stationary_expiry_helper_does_not_invent_credit_for_invalid_rise(authority, setup, invalid_rise):
    a = authority
    configure_authority(a, setup)
    for i in range(2):
        advance(a, 100.+i*.2+.07)
        decide_commit(a, a.frame(1.8, stamp=100.+i*.2, rpm=0.))
    advance(a, 100.47)
    a.controller.cfg = replace(a.controller.cfg, distance_pid_output_rise_rpm_per_sec=invalid_rise)
    current = a.frame(1.8, stamp=100.4, rpm=0.)
    assert a.controller._depth_expiry_recovery_step(current, a.clock.now, 100.4) == 0.
