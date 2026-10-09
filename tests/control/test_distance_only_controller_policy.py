"""Controller integration: diagnostic human motion cannot command speed.

Fake physical clocks/depth/encoders only. Identity, current distance and actual
wheel feedback remain mandatory; no motor, camera or serial connection.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.controllers import FollowSafetyController
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence
from car_control_modular.longitudinal_execution import ForwardExecutionAnchor
from car_control_modular.mssd_motor import MotorSpeedReceipt
from car_control_modular.sample_braking import SampleBrakingAssessment
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup


def distance_only(setup, **changes):
    return configured(setup, distance_target_motion_control_enable=False,
                      target_distance_m=1.4, **changes)


def observe(controller, frame):
    controller._observe_longitudinal_motion(frame, frame.persons[0])


def test_disabled_policy_skips_estimators_but_keeps_physical_encoder_inputs(setup, monkeypatch):
    clock, controller, frame = distance_only(setup)
    def forbidden(*args, **kwargs):
        pytest.fail("distance-only mode must not evaluate human velocity or closure")
    monkeypatch.setattr(controller, "_observe_distance_closure", forbidden)
    monkeypatch.setattr(controller._longitudinal_feedforward, "update", forbidden)
    current = frame(2.2, rpm=24.)
    current = replace(current, steering_feedback=replace(current.steering_feedback,
                      right_forward_rpm=28.))
    observe(controller, current)
    assert controller._distance_approach_sample_trusted
    assert controller._distance_pi_raw_distance_m == 2.2
    assert controller._distance_pid_sample_timestamp == clock.now
    assert controller._distance_pi_ego_forward_rpm == 26.
    assert controller._distance_pi_outer_forward_rpm == 28.
    assert controller._distance_pi_feedback_timestamp == clock.now
    assert controller._braking_motion_evidence is None
    assert controller._longitudinal_motion_evidence is None
    assert not controller._distance_pi_motion_memory_allowed


@pytest.mark.parametrize("fault", ["old_feedback", "untrustworthy", "future_feedback", "old_depth", "uid", "hazard"])
def test_disabled_policy_does_not_manufacture_trusted_encoder_or_depth(setup, fault):
    clock, controller, frame = distance_only(setup)
    current = frame(2.2, rpm=20.)
    if fault == "old_feedback":
        current = replace(current, steering_feedback=replace(current.steering_feedback,
                          timestamp=clock.now-.151))
    elif fault == "untrustworthy":
        current = replace(current, steering_feedback=replace(current.steering_feedback,
                          trustworthy=False))
    elif fault == "future_feedback":
        current = replace(current, steering_feedback=replace(current.steering_feedback,
                          timestamp=clock.now+.001))
    elif fault == "old_depth":
        current = frame(2.2, rpm=20., stamp=clock.now-.181)
    elif fault == "uid":
        current = frame(2.2, rpm=20., uid=2)
    else:
        current = replace(current, hazard=replace(current.hazard, active=True))
    observe(controller, current)
    assert controller._distance_pi_ego_forward_rpm is None
    assert controller._distance_pi_outer_forward_rpm is None
    assert controller._distance_pi_feedback_timestamp is None
    if fault in {"old_depth", "uid", "hazard"}:
        assert not controller._distance_approach_sample_trusted


@pytest.mark.parametrize("diagnostics", ["absent", "approaching", "receding"])
def test_sample_braking_uses_ego_and_distance_without_human_motion_window(setup, diagnostics):
    clock, controller, frame = distance_only(
        setup, distance_pi_stationary_stop_preview_enabled=True)
    current = frame(2.2, rpm=20.)
    observe(controller, current)
    controller._braking_execution_bound_reader = lambda uid, now: 35.
    if diagnostics != "absent":
        rate = -2. if diagnostics == "approaching" else 2.
        controller._braking_range_rate = rate
        controller._braking_rate_source = "raw_depth_window"
        controller._braking_motion_evidence = RawDepthMotionEvidence(
            clock.now, rate, rate, .1, 3)
    assessment = controller._fresh_braking_assessment(current, clock.now, clock.now)
    assert isinstance(assessment, SampleBrakingAssessment)
    assert assessment.target_speed_m_s == 0.
    assert assessment.travel_bound_rpm == 35.
    assert assessment.outer_rpm == 20.
    assert assessment.distance_m == 2.2
    assert assessment.sample_timestamp == clock.now
    assert assessment.feedback_timestamp == clock.now


@pytest.mark.parametrize("fault", ["old_depth", "old_feedback", "untrustworthy", "uid", "hazard"])
def test_distance_only_braking_assessment_still_requires_current_physical_evidence(setup, fault):
    clock, controller, frame = distance_only(
        setup, distance_pi_stationary_stop_preview_enabled=True)
    current = frame(2.2, rpm=20.)
    if fault == "old_depth":
        current = frame(2.2, rpm=20., stamp=clock.now-.181)
    elif fault == "old_feedback":
        current = replace(current, steering_feedback=replace(current.steering_feedback,
                          timestamp=clock.now-.151))
    elif fault == "untrustworthy":
        current = replace(current, steering_feedback=replace(current.steering_feedback,
                          trustworthy=False))
    elif fault == "uid":
        current = frame(2.2, rpm=20., uid=2)
    else:
        current = replace(current, hazard=replace(current.hazard, active=True))
    observe(controller, current)
    controller._braking_execution_bound_reader = lambda uid, now: 35.
    assert controller._fresh_braking_assessment(
        current, clock.now, current.distance_state.sample_timestamp) is None


def recovery(setup):
    clock, controller, frame = distance_only(setup)
    current = frame(2.2, rpm=1.)
    observe(controller, current)
    old = clock.now-.1
    controller._distance_pid_last_sample_timestamp = old
    controller._distance_pid_last_forward_control = True
    controller._distance_pi_admitted_grant = (1, old)
    controller._distance_pi_grant_withdrawal = (1, old, "no_live_grant_before_pi")
    controller._distance_pid._distance_pi._last_execution_ts = old
    controller._distance_pid._distance_pi._last_raw_distance_m = 2.3
    # Deliberately inject an old diagnostic estimate. The accepted CURRENT
    # distance still has safe margin; the diagnostic may not veto its ramp.
    controller._braking_range_rate = -1.
    controller._braking_motion_evidence = RawDepthMotionEvidence(clock.now, -1., -.8, .1, 3)
    return clock, controller, current


def test_new_grant_recovery_ignores_negative_velocity_and_decreasing_raw_endpoint(setup):
    clock, controller, current = recovery(setup)
    assert current.distance_state.raw_distance_m < controller._distance_pid._distance_pi._last_raw_distance_m
    assert controller._fresh_grant_recovery_step(current, clock.now, clock.now) == .05


@pytest.mark.parametrize("distance,completed,expected", [
    # New stationary wheels and independently safe distance need no old
    # positive receipt merely to take one bounded NEW-sample starting step.
    (2.2, False, .05), (1.7, True, .05), (1.7, False, .05),
    (1.4, True, 0.), (1.4, False, 0.),
])
def test_expiry_restart_uses_current_distance_and_packet_provenance_not_receding_estimate(
        setup, distance, completed, expected):
    clock, controller, frame = distance_only(setup, depth_longitudinal_sample_max_age_sec=.25)
    current = frame(distance, rpm=0.)
    observe(controller, current)
    old = clock.now-.28
    controller._distance_pid_last_sample_timestamp = old
    controller._distance_pid_last_forward_control = True
    controller._distance_pi_admitted_grant = (1, old)
    controller._distance_pi_grant_withdrawal = (1, old, "physical_depth_expired")
    controller._distance_pid._distance_pi._last_execution_ts = old
    controller.last_distance_pid_result = SimpleNamespace(
        approach_mode="distance_pi", output_rpm=40., pi_braking_distance_input_m=distance+.1)
    controller._braking_range_rate = -1.
    controller._braking_motion_evidence = RawDepthMotionEvidence(clock.now, -1., -.8, .1, 3)
    stamp = clock.now-.05
    anchor = (ForwardExecutionAnchor(1, old, 40., stamp,
                                    MotorSpeedReceipt(1, 40, -40, stamp))
              if completed else None)
    step = controller._depth_expiry_recovery_step(
        current, clock.now, clock.now, recent_execution_anchor=anchor)
    assert step == 0.  # Pure distance no longer enters the old expiry-age policy.
    assert controller._fresh_grant_recovery_step(current, clock.now, clock.now) == expected
    assert controller._distance_pi_admitted_grant == (1, old)  # Not authorization.


@pytest.mark.parametrize("fault", ["near", "old_depth", "old_feedback", "untrustworthy", "large_reverse",
                                   "uid", "hazard", "parking", "active_reverse", "safety_withdrawal"])
def test_new_distance_recovery_does_not_bypass_real_stop_or_evidence_requirements(setup, fault):
    clock, controller, current = recovery(setup)
    if fault == "near":
        current = replace(current, distance_m=1.4, distance_state=replace(
            current.distance_state, raw_distance_m=1.4))
    elif fault == "old_depth":
        current = replace(current, distance_state=replace(current.distance_state,
                          sample_timestamp=clock.now-.181))
    elif fault in {"old_feedback", "untrustworthy", "large_reverse"}:
        changes = {"old_feedback": dict(timestamp=clock.now-.101),
                   "untrustworthy": dict(trustworthy=False),
                   "large_reverse": dict(left_forward_rpm=-6.)}[fault]
        current = replace(current, steering_feedback=replace(current.steering_feedback, **changes))
    elif fault == "uid":
        controller.active_target_id = 2
    elif fault == "hazard":
        current = replace(current, hazard=replace(current.hazard, active=True))
    elif fault == "parking":
        controller._normal_parking_uid = 1
    elif fault == "active_reverse":
        controller._reverse_active = True
    else:
        controller._distance_pi_grant_withdrawal = (1, clock.now-.1, "emergency_stop")
    assert controller._fresh_grant_recovery_step(
        current, clock.now, current.distance_state.sample_timestamp) == 0.


@pytest.mark.parametrize("approach", [0., .3, 3.])
def test_reverse_request_has_no_human_approach_feedforward_or_floor(setup, monkeypatch, approach):
    clock, controller, frame = distance_only(setup, reverse_enable=True)
    monkeypatch.setattr(controller, "_update_distance_pid",
                        lambda *args, **kwargs: SimpleNamespace(output_rpm=-12))
    percent = controller._reverse_percent_for_distance(
        1.1, approach_speed_m_s=approach, now=clock.now, frame=frame(1.1))
    assert percent == 6
    assert controller._reverse_last_base_rpm == 12
    assert controller._reverse_last_feedforward_rpm == 0
    assert controller._reverse_last_output_rpm == 12


def test_real_reverse_pid_has_no_range_derivative(setup):
    clock, controller, frame = distance_only(
        setup, reverse_enable=True, distance_pid_kd_rpm_s_per_m=40.)
    assert controller._distance_pid.config.kd_rpm_s_per_m == 0.
    for distance in (1.2, 1.1, 1.0):
        current = frame(distance)
        observe(controller, current)
        controller._reverse_percent_for_distance(
            distance, approach_speed_m_s=3., now=clock.now, frame=current)
        assert controller.last_distance_pid_result.d_rpm == 0.
        clock.now += .05


@pytest.mark.parametrize("mode", ["distance_pi", "legacy"])
def test_master_switch_blocks_tracking_base_even_if_legacy_feedforward_enabled(setup, mode):
    clock, controller, frame = distance_only(setup)
    controller = FollowSafetyController(replace(controller.cfg,
        distance_control_mode=mode, distance_approach_enable=False,
        distance_feedforward_enable=True, distance_matching_test_bias_rpm=10.))
    controller.active_target_id = 1
    controller._has_seen_person = True
    controller._distance_pid_sample_timestamp = clock.now
    controller._longitudinal_motion_evidence = SimpleNamespace(
        status="ready", eligible=True, target_id=1, sample_timestamp=clock.now,
        target_rpm=80., decline_policy="bounded_far_positive", speed_window_samples=3,
        speed_window_sec=.1, instantaneous_target_speed_m_s=1., window_target_speed_m_s=1.)
    assert controller._tracking_base_rpm(2.2, clock.now) is None
