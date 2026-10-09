"""Fresh-depth recovery at a barely expired lease, using fake sensors only."""
from dataclasses import replace

import pytest

from car_control_modular.control_types import SteeringFeedback
from car_control_modular.depth_continuation import (
    ContinuationMotionEvidence, relative_continuation_speed_cap,
)
from test_depth_authority_250 import authority, advance, decide_commit
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def prepare(a, setup, *, old_used, old_raw, new_used, new_raw,
            old_rpm, new_rpm, sample_gap, process_gap):
    _, a.controller, _ = configured(
        setup, target_distance_m=1.4, distance_pi_kp_per_sec=3.,
        distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
        distance_pi_motion_memory_sec=.35,
        distance_approach_deceleration_m_s2=.7,
        depth_longitudinal_sample_max_age_sec=.25)
    a.owner._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    first = a.frame(old_used, rpm=old_rpm)
    first = replace(first, distance_state=replace(first.distance_state,
                                                  raw_distance_m=old_raw))
    _, actions, accepted = decide_commit(a, first)
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    old_stamp = first.distance_state.sample_timestamp
    advance(a, old_stamp + process_gap)
    fresh = a.frame(new_used, rpm=new_rpm, stamp=old_stamp + sample_gap)
    fresh = replace(fresh, distance_state=replace(fresh.distance_state,
                                                  raw_distance_m=new_raw))
    return fresh, old_stamp


@pytest.mark.parametrize("case", [
    # CAP103: stationary wheels, but accepted raw Depth shows a receding person.
    (1.6377, 1.7553, 1.7553, 1.9382, 20., 0., .2328, .2651, 36),
    # CAP124: far person and moving wheels. The old +12 RPM step was 56 RPM.
    (2.48, 2.526, 2.517, 2.554, 44.5, 44.5, .1690, .2537, 80),
])
def test_new_receding_depth_short_expiry_gets_bounded_nonzero_ramp(
        authority, setup, case):
    a = authority
    old_used, old_raw, new_used, new_raw, old_rpm, new_rpm, gap, elapsed, minimum = case
    fresh, old_stamp = prepare(
        a, setup, old_used=old_used, old_raw=old_raw,
        new_used=new_used, new_raw=new_raw, old_rpm=old_rpm,
        new_rpm=new_rpm, sample_gap=gap, process_gap=elapsed)
    old_deadline = a.owner._depth30_linear_timing.depth_expires_at
    a.feedback = fresh.steering_feedback
    decision = a.controller.decide(10, fresh, longitudinal_only=True)
    assert a.owner._depth30_linear_timing.depth_expires_at == old_deadline
    actions, accepted = a.owner._commit_depth_linear_decision(
        decision, fresh, 1, is_fresh_depth=True)
    result = a.controller.last_distance_pid_result
    assert accepted and result.pi_status == "recovering"
    assert result.pi_depth_expiry_recovery_step_sec == pytest.approx(.15)
    assert result.output_rpm == minimum
    assert result.output_rpm <= min(result.approach_cap_rpm, new_rpm + 36.)
    assert any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot[3] == old_stamp + gap
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(old_stamp+gap+.25)


def test_cap165_independent_relative_brake_still_vetoes_forward():
    stamp = 10030.841595598
    age = .1685
    cap, reason = relative_continuation_speed_cap(
        distance=2.35472, stop_distance=1.4,
        original_speed_bound=1.225221, sample_age=age,
        feedback=SteeringFeedback(timestamp=10030.970118197,
                                  trustworthy=True,
                                  left_forward_rpm=85., right_forward_rpm=84.),
        now=stamp+age, circumference=.816814, max_rpm=200.,
        deceleration=.7, response_delay=.2,
        motion=ContinuationMotionEvidence(1, stamp, .3781920536,
                                          -.8222640031, .1664, 3),
        target_id=1, sample_timestamp=stamp)
    assert (cap, reason) == (0., "braking_margin")


@pytest.mark.parametrize("replay", ["duplicate", "older", "expired_duplicate"])
def test_replayed_depth_cannot_repeat_fast_step_or_refresh_deadline(
        authority, setup, replay):
    a = authority
    fresh, old_stamp = prepare(
        a, setup, old_used=2.48, old_raw=2.526,
        new_used=2.517, new_raw=2.554, old_rpm=44.5,
        new_rpm=44.5, sample_gap=.169, process_gap=.2537)
    _, actions, accepted = decide_commit(a, fresh)
    result = a.controller.last_distance_pid_result
    assert accepted and result.pi_depth_expiry_recovery_step_sec == pytest.approx(.15)
    assert result.pi_sample_dt_sec == 0.  # Never integrate the blind interval.
    stamp = fresh.distance_state.sample_timestamp
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    integral = a.controller._distance_pid._distance_pi.integral_m_s
    prior_request = result.output_rpm

    advance(a, stamp + (.251 if replay == "expired_duplicate" else .05))
    repeat_stamp = old_stamp if replay == "older" else stamp
    repeated = replace(
        fresh, distance_state=replace(fresh.distance_state,
                                      sample_timestamp=repeat_stamp),
        steering_feedback=replace(fresh.steering_feedback, timestamp=a.clock.now))
    assert a.controller._depth_expiry_recovery_step(
        repeated, a.clock.now, repeat_stamp) == 0.
    _, replay_actions, replay_accepted = decide_commit(a, repeated)
    assert not replay_accepted
    assert not any(x.kind == "forward" and x.speed_percent > 0
                   for x in replay_actions)
    assert a.controller._distance_pid._distance_pi._last_sample_ts == stamp
    assert a.controller._distance_pid._distance_pi.integral_m_s == integral
    assert a.controller.last_distance_pid_result.output_rpm <= prior_request
    assert a.owner._depth30_linear_timing.depth_expires_at == deadline
    if replay == "expired_duplicate":
        assert a.owner._fresh_depth_linear_snapshot(1) is None


@pytest.mark.parametrize("fault", ["no_recession", "near", "long_expiry",
                                   "reverse", "stale_feedback", "identity_stop",
                                   "emergency_then_expiry", "hazard", "obstacle",
                                   "parking", "stale_depth", "duplicate"])
def test_short_expiry_fast_ramp_needs_all_new_measurement_evidence(
        authority, setup, fault):
    a = authority
    fresh, old_stamp = prepare(
        a, setup, old_used=2.48, old_raw=2.526,
        new_used=2.517, new_raw=2.554, old_rpm=44.5,
        new_rpm=44.5, sample_gap=.169, process_gap=.2537)
    if fault == "no_recession":
        fresh = replace(fresh, distance_state=replace(
            fresh.distance_state, raw_distance_m=2.526))
    elif fault == "near":
        fresh = replace(fresh, distance_m=1.64, distance_state=replace(
            fresh.distance_state, raw_distance_m=1.64))
    elif fault == "long_expiry":
        advance(a, old_stamp+.291)
        fresh = a.frame(2.517, rpm=44.5, stamp=old_stamp+.169)
        fresh = replace(fresh, distance_state=replace(
            fresh.distance_state, raw_distance_m=2.554))
    elif fault == "reverse":
        fresh = replace(fresh, steering_feedback=replace(
            fresh.steering_feedback, left_forward_rpm=-1.))
    elif fault == "stale_feedback":
        fresh = replace(fresh, steering_feedback=replace(
            fresh.steering_feedback, timestamp=a.clock.now-.11))
    elif fault == "identity_stop":
        a.controller.suspend_longitudinal_authority(a.clock.now, "identity_lost")
    elif fault == "emergency_then_expiry":
        a.controller.suspend_longitudinal_authority(old_stamp+.20, "emergency_stop")
        a.controller.suspend_longitudinal_authority(old_stamp+.251, "physical_depth_expired")
    elif fault == "hazard":
        fresh = replace(fresh, hazard=replace(fresh.hazard, active=True))
    elif fault == "obstacle":
        fresh = replace(fresh, obstacles=replace(fresh.obstacles, front=True))
    elif fault == "parking":
        a.controller._normal_parking_uid = 1
    elif fault == "stale_depth":
        fresh = replace(fresh, distance_state=replace(
            fresh.distance_state, sample_timestamp=old_stamp+.060))
    elif fault == "duplicate":
        fresh = replace(fresh, distance_state=replace(
            fresh.distance_state, sample_timestamp=old_stamp))
    decide_commit(a, fresh)
    result = a.controller.last_distance_pid_result
    assert result is None or result.pi_depth_expiry_recovery_step_sec != pytest.approx(.15)
