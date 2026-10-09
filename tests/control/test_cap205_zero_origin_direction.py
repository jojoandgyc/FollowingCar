"""Synthetic direction boundaries for zero-origin recovery, not a vehicle replay.

Small signed feedback may produce a bounded PI request; the real wheel writer
still owns reversal/STOP protection. Future samples below are explicit test
inputs, not predictions of how the recorded CAP205 vehicle would have moved.
"""
from dataclasses import replace

import pytest

from car_control_modular.sample_braking import SampleBrakingAssessment
from test_depth_authority_250 import authority, advance, decide_commit, writer
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner
from test_zero_origin_sample_cadence import configure_authority, controller, step


def pure_pair(pi, pair, stamp=100.2):
    now = stamp + .07
    outer = max(map(abs, pair))
    assessment = SampleBrakingAssessment(
        1, stamp, now, 1.8, outer, outer, now,
        0., 1.1, .816814, .7, .2, 200., outer_allowance_rpm=10.)
    return step(pi, stamp, wheel=.5*sum(pair), braking_assessment=assessment,
                preview_outer_forward_rpm=outer)


@pytest.mark.parametrize("pair", [(-5., -5.), (-4., -4.), (-1., -1.),
    (-.5, -.5), (-1., 0.), (-5., 0.), (-5., 5.)])
def test_pure_pi_signed_small_tail_only_proposes_bounded_new_request(pair):
    pi = controller()
    assert step(pi, 100.).output_rpm == 0
    result = pure_pair(pi, pair)
    assert result.final_limit_reason == "zero_origin_restart"
    assert 0 < result.output_rpm <= 12
    assert result.output_rpm <= result.cap_rpm
    assert result.sample_timestamp == 100.2
    assert result.sample_dt_sec == 0


@pytest.mark.parametrize("pair", [(-5.01, 5.01), (-20., -20.), (-80., 80.),
    (-80., 0.), (80., -80.)])
def test_reverse_or_pivot_cannot_cancel_in_the_mean_to_gain_zero_origin_credit(pair):
    pi = controller()
    step(pi, 100.)
    result = pure_pair(pi, pair)
    assert result.output_rpm == 0
    assert result.final_limit_reason != "zero_origin_restart"


def prepare_real_request(a, setup, pair):
    configure_authority(a, setup)
    advance(a, 100.07)
    decide_commit(a, a.frame(1.8, stamp=100., rpm=0.))
    assert a.controller.last_distance_pid_result.output_rpm == 0
    action, backend = writer(a)
    # Match the logged production handoff policy, including its independent
    # commanded-reverse gate. The fake backend never touches real motors.
    action.config.follow_forward_handoff_enable = True
    action.config.follow_residual_reverse_max_rpm = 8.
    action.get_steering_feedback = lambda: a.feedback
    advance(a, 100.27)
    current = a.frame(1.8, stamp=100.2, rpm=.5*sum(pair))
    current = replace(current, steering_feedback=replace(current.steering_feedback,
        left_forward_rpm=pair[0], right_forward_rpm=pair[1]))
    _, actions, accepted = decide_commit(a, current)
    return action, backend, actions, accepted


@pytest.mark.parametrize("pair,may_write_forward", [
    ((-1., 0.), True), ((-.5, -.5), True),
    ((-3., 0.), True), ((-3., 3.), True),
    ((-3., -3.), False), ((-1.01, -1.01), False),
])
def test_real_writer_preserves_existing_noise_and_one_wheel_handoff_policy(
        authority, setup, pair, may_write_forward):
    a = authority
    action, backend, actions, accepted = prepare_real_request(a, setup, pair)
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    result = a.controller.last_distance_pid_result
    assert result.pi_final_limit_reason == "zero_origin_restart"
    assert 0 < result.output_rpm <= 12
    assert a.owner._depth30_linear_snapshot[3] == 100.2
    action._service_follow_wheels()
    pair_written = backend.pairs[-1][:2]
    if may_write_forward:
        assert pair_written[0] > 0 and pair_written[1] < 0
    else:
        assert pair_written == (0, 0)
        assert action._visible_wheel_guard.pending_full_reverse


@pytest.mark.parametrize("pair", [(-5., 0.), (-5., 5.), (-5., -5.)])
def test_pi_five_rpm_scope_does_not_override_runtime_three_rpm_tail_admission(
        authority, setup, pair):
    a = authority
    action, backend, actions, _ = prepare_real_request(a, setup, pair)
    # PI requests 12 RPM, but existing admission admits negative tails only
    # within 3 RPM. Rejection also poisons this unexecuted PI request.
    pi = a.controller._distance_pid._distance_pi
    assert pi.last_result.final_limit_reason == "zero_origin_restart"
    assert pi._sample_rejected
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot is None
    action._service_follow_wheels()
    assert not any(left > 0 and right < 0 for left, right, *_ in backend.pairs)


@pytest.mark.parametrize("pair", [(-20., -20.), (-80., 80.), (80., -80.)])
def test_real_controller_big_reverse_or_pivot_never_gets_zero_origin_forward_grant(
        authority, setup, pair):
    a = authority
    action, backend, actions, accepted = prepare_real_request(a, setup, pair)
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot is None
    action._service_follow_wheels()
    assert not any(left > 0 and right < 0 for left, right, *_ in backend.pairs)


def test_commanded_reverse_and_actual_opposing_motion_still_wait_for_zero(authority, setup):
    a = authority
    action, backend, _, accepted = prepare_real_request(a, setup, (-2., -2.))
    assert accepted
    guard = action._visible_wheel_guard
    action._visible_wheel_uid = 1  # The prior reverse write belonged to this UID.
    guard.note_sent((-8, -8), a.clock.now-.01)
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (0, 0)
    assert guard.commanded_reverse and guard.pending_full_reverse
    # Re-reading the same encoder sample is not another stop confirmation.
    quiet = guard.quiet_count
    advance(a, a.clock.now+.051)
    a.feedback = replace(a.feedback, timestamp=100.27)
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (0, 0)
    assert guard.quiet_count == quiet


def test_post_reverse_sub_rpm_feedback_requires_two_new_quiet_samples(
        authority, setup):
    a = authority
    action, backend, _, accepted = prepare_real_request(a, setup, (-.5, -.5))
    assert accepted
    action._visible_wheel_uid = 1
    guard = action._visible_wheel_guard
    guard.note_sent((-8, -8), a.clock.now-.01)
    # A real reverse write cannot become ordinary forward merely because one
    # encoder sample enters the noise band. Preserve the reverse provenance
    # through zero, then use the existing two distinct quiet-sample proof.
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (0, 0)
    assert guard.commanded_reverse and guard.pending_full_reverse
    for index in range(2):
        advance(a, a.clock.now+.051)
        action._service_follow_wheels()
        if index == 0:
            assert backend.pairs[-1][:2] == (0, 0)
            assert guard.commanded_reverse
    assert backend.pairs[-1][0] > 0 and backend.pairs[-1][1] < 0
    assert not guard.commanded_reverse


@pytest.mark.parametrize("withdrawal", ["emergency_stop", "identity_lost", "unknown"])
def test_new_request_cannot_bypass_a_later_real_withdrawal(authority, setup, withdrawal):
    a = authority
    action, backend, _, accepted = prepare_real_request(a, setup, (-1., 0.))
    assert accepted
    a.owner._revoke_depth_linear_authority(withdrawal)
    action._service_follow_wheels()
    assert a.owner._depth30_linear_snapshot is None
    assert not any(left > 0 and right < 0 for left, right, *_ in backend.pairs)
