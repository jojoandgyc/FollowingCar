"""A fresh far sample may restart a stopped command from tiny wheel motion.

The old physical lease stays expired. Real controller/runtime with fake clocks;
there is no motor, camera or serial I/O.
"""
from dataclasses import replace

import pytest

from car_control_modular.longitudinal_approach import RawDepthMotionEvidence
from test_depth_authority_250 import authority, advance, decide_commit, seed
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def prepare(a, setup, *, pair=(1., 2.), distance=2.1000, raw=2.1073961,
            elapsed=.3006, offset=.2702454):
    _, a.controller, _ = configured(
        setup, target_distance_m=1.4, distance_pi_kp_per_sec=3.,
        distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
        distance_pi_motion_memory_sec=.30,
        distance_approach_deceleration_m_s2=.7,
        depth_longitudinal_sample_max_age_sec=.25)
    a.owner._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    # No completed positive write survives: reproduce the old lease's actual
    # expiry/zero rather than supplying a fake successful speed receipt.
    a.controller._recent_longitudinal_execution_reader = lambda *_: None
    stamp, old = seed(a, distance=2.0999861, rpm=7.)
    advance(a, stamp+.251)
    a.owner._revoke_depth_linear_authority("physical_depth_expired")
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    advance(a, stamp+elapsed)
    current = a.frame(distance, stamp=stamp+offset, rpm=.5*sum(pair))
    current = replace(current, distance_state=replace(
        current.distance_state, raw_distance_m=raw), steering_feedback=replace(
            current.steering_feedback, left_forward_rpm=pair[0], right_forward_rpm=pair[1],
            timestamp=stamp+offset+.0092, yaw_rate_right_dps=1.6272,
            raw_yaw_rate_right_dps=-1.5675))
    return current, old


@pytest.mark.parametrize("pair", [(1., 2.), (0., 0.), (0., 1.), (2., 0.), (2., 5.)])
def test_cap532_new_far_nonclosing_sample_gets_bounded_step(authority, setup, pair):
    a = authority
    current, old = prepare(a, setup, pair=pair)
    _, actions, accepted = decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert accepted
    assert result.pi_depth_expiry_recovery_step_sec == pytest.approx(.05)
    assert not result.pi_depth_expiry_completed_anchor_used
    assert 12 <= result.output_rpm <= min(result.approach_cap_rpm, .5*sum(pair)+12.)
    assert any(x.kind == "forward" and x.speed_percent >= 6 for x in actions)
    assert a.owner._depth30_linear_snapshot[3] == current.distance_state.sample_timestamp
    assert a.owner._depth30_linear_snapshot is not old
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(
        current.distance_state.sample_timestamp+.25)


@pytest.mark.parametrize("fault", [
    "approaching", "near", "reverse", "untrustworthy", "asymmetric", "yaw",
    "old_feedback", "unaligned_feedback", "long_gap", "identity", "hazard",
    "obstacle", "no_window", "wrong_window", "negative_target", "parking",
])
def test_low_speed_restart_requires_all_new_evidence(authority, setup, monkeypatch, fault):
    a = authority
    current, _ = prepare(a, setup, elapsed=.41 if fault == "long_gap" else .3006)
    if fault == "approaching":
        current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=1.98))
    elif fault == "near":
        current = replace(current, distance_m=1.89, distance_state=replace(
            current.distance_state, raw_distance_m=1.89))
    elif fault in {"reverse", "untrustworthy", "asymmetric", "yaw", "old_feedback", "unaligned_feedback"}:
        changes = {
            "reverse": dict(left_forward_rpm=-1.),
            "untrustworthy": dict(trustworthy=False),
            "asymmetric": dict(left_forward_rpm=0., right_forward_rpm=6.),
            "yaw": dict(yaw_rate_right_dps=16.),
            "old_feedback": dict(timestamp=a.clock.now-.11),
            "unaligned_feedback": dict(timestamp=current.distance_state.sample_timestamp-.101),
        }[fault]
        current = replace(current, steering_feedback=replace(current.steering_feedback, **changes))
    elif fault == "identity":
        a.controller.suspend_longitudinal_authority(a.clock.now, "identity_lost")
    elif fault == "hazard":
        current = replace(current, hazard=replace(current.hazard, active=True))
    elif fault == "obstacle":
        current = replace(current, obstacles=replace(current.obstacles, front=True))
    elif fault == "parking":
        a.controller._normal_parking_uid = 1
    elif fault in {"no_window", "wrong_window", "negative_target"}:
        step = a.controller._depth_expiry_recovery_step

        def corrupt_window(*args, **kwargs):
            evidence = a.controller._braking_motion_evidence
            assert isinstance(evidence, RawDepthMotionEvidence)
            a.controller._braking_motion_evidence = (
                None if fault == "no_window" else replace(evidence,
                    **({"sample_timestamp": evidence.sample_timestamp-.01}
                       if fault == "wrong_window" else {"target_speed_bound_m_s": -.01})))
            return step(*args, **kwargs)

        monkeypatch.setattr(a.controller, "_depth_expiry_recovery_step", corrupt_window)
    decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert result is None or result.pi_depth_expiry_recovery_step_sec == 0.


def test_repeated_sample_cannot_add_step_or_renew_new_lease(authority, setup):
    a = authority
    current, _ = prepare(a, setup)
    _, _, accepted = decide_commit(a, current)
    assert accepted
    result = a.controller.last_distance_pid_result
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    for delta in (.01, .04, .09):
        advance(a, current.distance_state.sample_timestamp+.04+delta)
        _, _, accepted = decide_commit(a, current)
        assert not accepted
        assert a.controller.last_distance_pid_result.output_rpm <= result.output_rpm
        assert a.owner._depth30_linear_timing.depth_expires_at == deadline
    advance(a, deadline+.001)
    assert a.owner._fresh_depth_linear_snapshot(1) is None


def test_calculation_cannot_revive_old_lease_and_late_admission_still_fails(authority, setup):
    a = authority
    current, _ = prepare(a, setup)
    a.feedback = current.steering_feedback
    decision = a.controller.decide(10, current, longitudinal_only=True)
    assert a.controller.last_distance_pid_result.output_rpm == 13
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    advance(a, current.distance_state.sample_timestamp+.181)
    actions, _ = a.owner._commit_depth_linear_decision(
        decision, current, 1, is_fresh_depth=True)
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot is None


def test_fresh_recovery_step_remains_below_new_braking_envelope(authority, setup, monkeypatch):
    a = authority
    current, _ = prepare(a, setup)
    # Exercise the same real raw-window gate while lowering the physical
    # stopping model, not raising/fabricating encoder speed for this sample.
    pi = a.controller._distance_pid._distance_pi
    monkeypatch.setattr(pi, "config", replace(pi.config, deceleration_m_s2=.005))
    _, actions, _ = decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert result.pi_depth_expiry_recovery_step_sec == pytest.approx(.05)
    assert result.output_rpm <= result.approach_cap_rpm < 13
    assert all(x.speed_percent*2 <= result.approach_cap_rpm
               for x in actions if x.kind == "forward")


def test_stationary_near_target_does_not_inherit_far_restart_credit(authority, setup):
    a = authority
    current, _ = prepare(a, setup, pair=(0., 0.), distance=1.89, raw=1.89)
    _, actions, _ = decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert result.pi_depth_expiry_recovery_step_sec == 0
    assert result.output_rpm == 0
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
