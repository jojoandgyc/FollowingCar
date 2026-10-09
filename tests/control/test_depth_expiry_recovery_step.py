"""A new far Depth sample can soften only a short physical-expiry restart.

Controller and grant admission use fake time; no camera or motor I/O.
"""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from test_depth_authority_250 import authority, advance, decide_commit, seed
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


@pytest.mark.parametrize("explicit_expiry", [False, True])
def test_short_real_depth_expiry_uses_one_bounded_fresh_step(authority, explicit_expiry, caplog):
    a = authority
    old_stamp, old_grant = seed(a, distance=2.5, rpm=40.)
    assert a.controller._distance_pi_admitted_grant == (1, old_stamp)
    old_deadline = a.owner._depth30_linear_timing.depth_expires_at
    if explicit_expiry:
        advance(a, old_stamp + .251)
        a.owner._revoke_depth_linear_authority("physical_depth_expired")
    advance(a, old_stamp + .258)
    fresh = a.frame(2.49, rpm=26., stamp=a.clock.now-.017)
    with caplog.at_level("INFO"):
        _, actions, accepted = decide_commit(a, fresh)
    result = a.controller.last_distance_pid_result
    assert accepted and result.pi_status == "recovering"
    assert result.pi_depth_expiry_recovery_used
    assert result.pi_depth_expiry_recovery_step_sec == pytest.approx(.05)
    assert 26 < result.output_rpm <= min(result.approach_cap_rpm, 26+240*.05)
    assert any(x.kind == "forward" and x.speed_percent > 13 for x in actions)
    assert a.owner._depth30_linear_snapshot[3] == fresh.distance_state.sample_timestamp
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(
        fresh.distance_state.sample_timestamp + .25)
    assert old_deadline == pytest.approx(old_stamp+.25)
    assert old_grant[3] == old_stamp
    assert "distance_pi_depth_expiry_recovery" in caplog.text
    assert "prior_lease_reused=False new_grant_required=True" in caplog.text


@pytest.mark.parametrize("withdrawal", ["identity_lost", "emergency_stop", "lateral_depth_ineligible"])
def test_nonphysical_withdrawal_never_gets_depth_expiry_step(authority, withdrawal):
    a = authority
    old_stamp, _ = seed(a, distance=2.5, rpm=40.)
    advance(a, old_stamp+.20)
    a.owner._revoke_depth_linear_authority(withdrawal)
    advance(a, old_stamp+.258)
    fresh = a.frame(2.49, rpm=26., stamp=a.clock.now-.017)
    decide_commit(a, fresh)
    result = a.controller.last_distance_pid_result
    assert result.pi_status == "recovering"
    assert not result.pi_depth_expiry_recovery_used
    assert result.output_rpm <= 26


@pytest.mark.parametrize("case", ["long_gap", "near", "reverse_tail", "stale_feedback", "yaw"])
def test_fresh_step_requires_short_gap_far_depth_and_two_forward_wheels(authority, case):
    a = authority
    old_stamp, _ = seed(a, distance=2.5, rpm=40.)
    advance(a, old_stamp+(.46 if case == "long_gap" else .258))
    fresh = a.frame(1.85 if case == "near" else 2.49, rpm=26., stamp=a.clock.now-.017)
    fb = fresh.steering_feedback
    if case == "reverse_tail":
        fb = replace(fb, left_forward_rpm=53., right_forward_rpm=-1.)
    elif case == "stale_feedback":
        fb = replace(fb, timestamp=a.clock.now-.11)
    elif case == "yaw":
        fb = replace(fb, yaw_rate_right_dps=16.)
    fresh = replace(fresh, steering_feedback=fb)
    decide_commit(a, fresh)
    result = a.controller.last_distance_pid_result
    assert result is not None
    assert not result.pi_depth_expiry_recovery_used


def test_first_far_observation_has_no_prior_grant_to_recover(authority):
    a = authority
    _, actions, _ = decide_commit(a, a.frame(2.49, rpm=26.))
    result = a.controller.last_distance_pid_result
    assert not result.pi_depth_expiry_recovery_used
    assert result.output_rpm <= 26


@pytest.mark.parametrize("withdrawal", ["identity_then_expiry", "zero_approved"])
def test_prior_unsafe_or_zero_withdrawal_cannot_be_relabelled_as_expiry(authority, withdrawal):
    a = authority
    old_stamp, _ = seed(a, distance=2.5, rpm=40.)
    if withdrawal == "identity_then_expiry":
        a.controller.suspend_longitudinal_authority(old_stamp+.10, "identity_lost")
        a.controller.suspend_longitudinal_authority(old_stamp+.251, "physical_depth_expired")
    else:
        a.controller.accept_longitudinal_limit(old_stamp, 0.)
    advance(a, old_stamp+.258)
    decide_commit(a, a.frame(2.49, rpm=26., stamp=a.clock.now-.017))
    result = a.controller.last_distance_pid_result
    assert not result.pi_depth_expiry_recovery_used
    assert result.output_rpm <= 26


@pytest.mark.parametrize("withdrawal", ["identity_lost", "parked"])
def test_pi_cannot_use_caller_step_after_stop_or_without_expired_sample(withdrawal):
    pi = DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    kwargs = dict(target_distance_m=1.4, deadband_m=.03, max_output_rpm=200.,
                  rise_rpm_per_sec=240., ego_forward_rpm=26.,
                  depth_expiry_recovery_step_sec=.05)
    first = pi.update(2.5, sample_timestamp=100., execution_now=100., **kwargs)
    assert not first.depth_expiry_recovery_used
    if withdrawal == "parked":
        pi.set_normal_parking(True)
    else:
        pi.suspend(100.20, "identity_lost", reset_execution=True)
    stopped = pi.update(2.5, sample_timestamp=100.26, execution_now=100.26, **kwargs)
    assert stopped.output_rpm <= 26 and not stopped.depth_expiry_recovery_used
    with pytest.raises(ValueError):
        pi.update(2.5, sample_timestamp=100.4, execution_now=100.4,
                  **dict(kwargs, depth_expiry_recovery_step_sec=.151))
