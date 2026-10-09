"""CAP222/233: new evidence may restart, stale identity may not authorize.

Uses the real PI, controller and depth-admission path with fake clocks only.
"""
from dataclasses import replace

import pytest

from test_depth_authority_250 import authority, advance, decide_commit
from test_distance_tracking_response import setup
from test_fresh_distance_restart_progress import prepare
from test_lateral_zero_runtime import owner


REASONS = ["stale_vision_result", "lateral_depth:continuation_feedback_invalid"]


def pending(a, setup, monkeypatch, reason, *, distance=1.9207109, pair=(0., 0.)):
    frame, old = prepare(a, setup, monkeypatch, reason=reason, distance=distance,
                         pair=pair, gap=.319, age=.119, feedback_age=.033)
    calls = []

    def identity(uid, stamp, now):
        calls.append((uid, stamp, now))
        return True

    a.controller._longitudinal_restart_identity_reader = identity
    return frame, old, calls


@pytest.mark.parametrize("reason", REASONS)
def test_cap222_to233_new_distance_progresses_instead_of_restarting_at_zero(
        authority, setup, monkeypatch, reason):
    a = authority
    frame, old, calls = pending(a, setup, monkeypatch, reason)
    outputs = []
    for distance in (1.9207109, 2.1050495, 2.3316):
        _, actions, accepted = decide_commit(a, frame)
        result = a.controller.last_distance_pid_result
        outputs.append(result.output_rpm)
        assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
        assert result.output_rpm <= result.approach_cap_rpm
        assert a.owner._depth30_linear_snapshot[3] == frame.distance_state.sample_timestamp
        assert a.owner._depth30_linear_snapshot[3] != old[3]
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(
            frame.distance_state.sample_timestamp + .30)
        advance(a, a.clock.now+.13)
        frame = a.frame(distance, rpm=0., stamp=a.clock.now-.06)
    assert outputs[0] == 12
    assert outputs[0] < outputs[1] < outputs[2]
    assert calls and calls[0][0] == 1 and calls[0][1] > old[3]


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("proof", [None, False, "raises", 1])
def test_pending_identity_preview_zero_does_not_poison_later_new_confirmation(
        authority, setup, monkeypatch, reason, proof):
    a = authority
    frame, old, _ = pending(a, setup, monkeypatch, reason)

    def reader(*args):
        if proof == "raises":
            raise RuntimeError("identity publication unavailable")
        return proof

    a.controller._longitudinal_restart_identity_reader = None if proof is None else reader
    _, actions, _ = decide_commit(a, frame)
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.controller._distance_pi_grant_withdrawal == (1, old[3], reason)
    assert not a.controller.last_distance_pid_result.pi_fresh_grant_recovery_used
    # Losing availability again must not erase the stronger need for a NEW
    # identity confirmation, nor turn this into old-receipt continuity.
    a.controller.suspend_longitudinal_authority(a.clock.now, "no_live_grant_before_pi")
    a.controller.suspend_longitudinal_authority(a.clock.now, "physical_depth_expired")
    assert a.controller._distance_pi_grant_withdrawal == (1, old[3], reason)
    a.controller._longitudinal_restart_identity_reader = lambda *_: True
    advance(a, a.clock.now+.13)
    frame = a.frame(2.105, rpm=0., stamp=a.clock.now-.05)
    _, actions, accepted = decide_commit(a, frame)
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.controller.last_distance_pid_result.pi_fresh_grant_recovery_used
    assert not a.controller.last_distance_pid_result.pi_execution_recovery_anchor_used
    assert a.controller.last_distance_pid_result.output_rpm == 12


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("hard_reason", ["identity_conflict", "emergency_stop", "momentum_stop",
    "zero_approved", "lateral_depth:shared_braking_interval_lost",
    "lateral_depth:shared_braking_new_higher_write", "reverse_transition"])
def test_identity_restart_reason_cannot_hide_established_safety_withdrawal(
        authority, setup, monkeypatch, reason, hard_reason):
    a = authority
    frame, old, _ = pending(a, setup, monkeypatch, hard_reason)
    a.controller.suspend_longitudinal_authority(a.clock.now, reason)
    a.controller.suspend_longitudinal_authority(a.clock.now, "no_live_grant_before_pi")
    assert a.controller._distance_pi_grant_withdrawal == (1, old[3], hard_reason)
    decide_commit(a, frame)
    assert not a.controller.last_distance_pid_result.pi_fresh_grant_recovery_used
    assert a.controller.last_distance_pid_result.output_rpm == 0


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("fault", ["reverse_left", "reverse_right", "yaw", "feedback_stale",
    "feedback_skew", "feedback_untrusted", "near", "raw_near", "old_depth", "duplicate",
    "active_reverse", "parking", "identity_missing", "hazard", "obstacle"])
def test_new_identity_does_not_replace_current_physical_or_control_safety(
        authority, setup, monkeypatch, reason, fault):
    a = authority
    frame, old, _ = pending(a, setup, monkeypatch, reason)
    fb_changes = {
        "reverse_left": {"left_forward_rpm": -4.},
        "reverse_right": {"right_forward_rpm": -4.},
        "yaw": {"raw_yaw_rate_right_dps": 16.},
        "feedback_stale": {"timestamp": a.clock.now-.151},
        "feedback_skew": {"timestamp": frame.distance_state.sample_timestamp+.151},
        "feedback_untrusted": {"trustworthy": False},
    }
    if fault in fb_changes:
        frame = replace(frame, steering_feedback=replace(frame.steering_feedback, **fb_changes[fault]))
    elif fault in {"near", "raw_near"}:
        # Settled feedback-invalid recovery now uses the normal start band;
        # stale identity still requires the original farther-range proof.
        near = (a.controller.cfg.forward_start_distance_m
                if reason == "lateral_depth:continuation_feedback_invalid" else 1.89)
        frame = replace(frame, distance_m=near if fault == "near" else 2.5,
                        distance_state=replace(frame.distance_state, raw_distance_m=near))
    elif fault in {"old_depth", "duplicate"}:
        frame = replace(frame, distance_state=replace(frame.distance_state,
            sample_timestamp=old[3] if fault == "duplicate" else a.clock.now-.181))
    elif fault == "identity_missing":
        frame = replace(frame, persons=[])
    elif fault == "hazard":
        frame = replace(frame, hazard=replace(frame.hazard, active=True))
    elif fault == "obstacle":
        frame = replace(frame, obstacles=replace(frame.obstacles, front=True))
    elif fault == "parking":
        a.controller.set_normal_parking(True, 1)
    elif fault == "active_reverse":
        qualify = a.controller._fresh_grant_recovery_step

        def during_reverse(*args):
            a.controller._reverse_active = True
            return qualify(*args)

        monkeypatch.setattr(a.controller, "_fresh_grant_recovery_step", during_reverse)
    decide_commit(a, frame)
    r = a.controller.last_distance_pid_result
    assert r is None or not r.pi_fresh_grant_recovery_used


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("pair", [(0., 0.), (-1., 1.), (-2., -1.), (-3., 3.)])
def test_quiet_tail_restart_remains_bounded_and_duplicate_cannot_add_credit(
        authority, setup, monkeypatch, reason, pair):
    a = authority
    frame, _, _ = pending(a, setup, monkeypatch, reason, pair=pair)
    _, actions, accepted = decide_commit(a, frame)
    r = a.controller.last_distance_pid_result
    assert accepted and r.pi_fresh_grant_recovery_used
    assert 0 < r.output_rpm <= max(0., sum(pair)/2.)+12.
    assert any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    before = r.output_rpm
    advance(a, a.clock.now+.02)
    decide_commit(a, frame)
    assert a.controller.last_distance_pid_result.output_rpm == before
    assert a.owner._depth30_linear_timing.depth_expires_at == deadline


@pytest.mark.parametrize("reason", REASONS)
def test_actual_existing_grant_zero_is_still_not_an_observation_only_gap(
        authority, setup, monkeypatch, reason):
    a = authority
    frame, _, _ = pending(a, setup, monkeypatch, reason)
    decide_commit(a, frame)
    stamp = frame.distance_state.sample_timestamp
    a.controller.accept_longitudinal_limit(stamp, 0.)
    assert a.controller._distance_pi_grant_withdrawal == (1, stamp, "zero_approved")
    a.controller.suspend_longitudinal_authority(a.clock.now, reason)
    advance(a, a.clock.now+.13)
    decide_commit(a, a.frame(2.2, rpm=0., stamp=a.clock.now-.05))
    assert not a.controller.last_distance_pid_result.pi_fresh_grant_recovery_used
