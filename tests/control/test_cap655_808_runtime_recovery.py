"""A lateral intent may withdraw a physically expired range grant."""

import threading

import pytest

from car_control_modular.longitudinal_execution import ForwardExecutionAnchor
from test_depth_authority_250 import authority, advance, seed, decide_commit
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


@pytest.mark.parametrize("elapsed,physical", [(.24, False), (.254, True)])
def test_lateral_expiry_is_physical_only_after_the_depth_deadline(authority, elapsed, physical):
    a = authority
    stamp, _grant = seed(a, distance=2.23, rpm=30.)
    advance(a, stamp+elapsed)
    a.controller.suspend_longitudinal_authority(
        a.clock.now, "lateral_zero_no_qualified_depth:revoke:expired")
    expected = ("physical_depth_expired" if physical else
                "lateral_zero_no_qualified_depth:revoke:expired")
    assert a.controller._distance_pi_grant_withdrawal == (1, stamp, expected)
    assert a.controller._distance_pid._distance_pi._expiry_recovery_forbidden is not physical


def test_far_new_depth_gets_one_bounded_step_after_lateral_expiry(authority):
    a = authority
    stamp, _grant = seed(a, distance=2.23, rpm=30.)
    advance(a, stamp+.254)
    a.owner._revoke_depth_linear_authority("lateral_zero_no_qualified_depth:revoke:expired")
    assert a.owner._depth30_linear_snapshot is None
    current = a.frame(2.23, rpm=30., stamp=stamp+.242)
    _decision, actions, accepted = decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert result.pi_status == "recovering"
    assert result.pi_depth_expiry_recovery_step_sec == pytest.approx(.05)
    assert 30 < result.output_rpm <= 30+240*.05
    assert a.owner._depth30_linear_snapshot[3] == current.distance_state.sample_timestamp


@pytest.mark.parametrize("completed_packet", [False, True])
def test_recent_completed_packet_bridges_new_depth_only_at_walking_distance(
        authority, setup, completed_packet):
    a = authority
    _clock, a.controller, _frame = configured(
        setup, target_distance_m=1.4, distance_pi_kp_per_sec=3.,
        distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
        depth_longitudinal_sample_max_age_sec=.25)
    a.owner._follow_controller = a.controller
    a.owner.motor_io_lock = threading.RLock()
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    stamp, _grant = seed(a, distance=1.82, rpm=34.)
    advance(a, stamp+.264)
    proof = ForwardExecutionAnchor(1, stamp, 34., stamp+.237, object())
    a.controller._recent_longitudinal_execution_reader = (
        lambda uid, _now: proof if completed_packet and uid == 1 else None)
    current = a.frame(1.82, rpm=17.5, stamp=stamp+.232)
    _decision, actions, accepted = decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert result.pi_status == "recovering"
    assert result.pi_depth_expiry_completed_anchor_used is completed_packet
    assert result.output_rpm <= result.approach_cap_rpm
    assert a.owner._depth30_linear_snapshot[3] == current.distance_state.sample_timestamp
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    if completed_packet:
        assert result.output_rpm > 29
    else:
        assert result.output_rpm <= 17


def test_intervening_zero_invalidates_expiry_execution_proof(authority, setup, monkeypatch):
    a = authority
    _clock, a.controller, _frame = configured(
        setup, target_distance_m=1.4, distance_pi_kp_per_sec=3.,
        distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
        depth_longitudinal_sample_max_age_sec=.25)
    a.owner._follow_controller = a.controller
    a.owner.motor_io_lock = threading.RLock()
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    stamp, _grant = seed(a, distance=1.82, rpm=34.)
    advance(a, stamp+.264)
    completed = [ForwardExecutionAnchor(1, stamp, 34., stamp+.237, object())]
    a.controller._recent_longitudinal_execution_reader = lambda _uid, _now: completed[0]
    original = a.controller._distance_pid.update

    def update_then_zero(*args, **kwargs):
        result = original(*args, **kwargs)
        completed[0] = None  # A completed zero/STOP replaces the receipt.
        return result

    monkeypatch.setattr(a.controller._distance_pid, "update", update_then_zero)
    current = a.frame(1.82, rpm=17.5, stamp=stamp+.232)
    _decision, actions, _accepted = decide_commit(a, current)
    assert a.controller.last_distance_pid_result.output_rpm == 0
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot is None


@pytest.mark.parametrize("new_depth_age,admitted", [(.177, False), (.120, True)])
def test_new_forward_grant_requires_actual_dispatch_time_without_braking_evidence(
        authority, new_depth_age, admitted):
    a = authority
    stamp, _grant = seed(a, distance=2.02, rpm=30.)
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    advance(a, stamp+.208)
    current = a.frame(2.02, rpm=30., stamp=a.clock.now-new_depth_age)
    a.owner._depth_continuation_evidence = lambda *_args: (None, None)
    _decision, actions, accepted = decide_commit(a, current)
    if admitted:
        assert accepted and any(action.kind == "forward" and action.speed_percent > 0
                                for action in actions)
        assert a.owner._depth30_linear_snapshot[3] == current.distance_state.sample_timestamp
    else:
        # The still-live prior grant may continue, but the unsuitable newer
        # sample cannot get its own deadline or acceleration budget. A failed
        # unchanged update no longer republishes the existing motor target.
        assert not accepted and actions == []
        assert a.owner._fresh_depth_linear_snapshot(1)[1] > 0
        assert a.owner._depth30_linear_snapshot[3] == stamp
        assert a.owner._depth30_linear_timing.depth_expires_at == deadline
