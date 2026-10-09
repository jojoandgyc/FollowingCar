"""Fresh distance recovery after settled wheels or completed range confirmation.

Real PI, controller and grant admission with in-memory clocks/feedback only.
"""
from dataclasses import replace

import pytest

from test_depth_authority_250 import authority, advance, decide_commit
from test_distance_tracking_response import setup
from test_fresh_distance_restart_progress import prepare
from test_lateral_zero_runtime import owner


INVALID_FEEDBACK = "lateral_depth:continuation_feedback_invalid"


def current_identity(a, answer=True):
    a.controller._longitudinal_restart_identity_reader = lambda *_: answer
    a.controller.cfg = replace(a.controller.cfg, forward_start_distance_m=1.48,
                               forward_stop_distance_m=1.43)


def pending_confirmation(a, setup, monkeypatch, *, reason="physical_depth_expired"):
    _, old = prepare(a, setup, monkeypatch, gap=.1, age=.12, reason=reason)
    current_identity(a)
    for offset, count in ((.34, 1), (.48, 2)):
        advance(a, old[3]+offset)
        frame = a.frame(None, rpm=0.)
        frame = replace(frame, distance_state=replace(frame.distance_state,
            sample_timestamp=None, observation_timestamp=a.clock.now-.04,
            temporal_status="new_sample",
            source_detail=f"distance_jump_pending_{count}_of_3"))
        _, actions, _ = decide_commit(a, frame, fresh=False)
        assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
        assert a.controller._distance_pid_last_sample_timestamp is None
        assert a.controller._distance_pi_admitted_grant is None
        assert a.controller._distance_pid._distance_pi.integral_m_s == 0.
    advance(a, old[3]+.65)
    frame = a.frame(2.6148, rpm=.5, stamp=old[3]+.58)
    frame = replace(frame, distance_state=replace(frame.distance_state,
        source_detail="depth_confirmed_jump", fusion_mode="depth_radar_depth_confirmed"),
        steering_feedback=replace(frame.steering_feedback, left_forward_rpm=1.,
                                  right_forward_rpm=0., timestamp=a.clock.now-.04))
    return frame, old


@pytest.mark.parametrize("distance", [1.49, 1.5744871734, 1.89, 1.901])
def test_cap171_new_identity_and_settled_forward_wheels_remove_old_distance_cliff(
        authority, setup, monkeypatch, distance):
    a = authority
    frame, old = prepare(a, setup, monkeypatch, distance=distance, pair=(1., 0.),
                         reason=INVALID_FEEDBACK, gap=.34, age=.13, feedback_age=.0394)
    current_identity(a)
    _, actions, accepted = decide_commit(a, frame)
    r = a.controller.last_distance_pid_result
    assert accepted and r.pi_fresh_grant_recovery_used
    assert 0 < r.output_rpm <= min(r.approach_cap_rpm, 12.5)
    assert any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot[3] == frame.distance_state.sample_timestamp != old[3]
    for _ in range(3):
        previous = r.output_rpm
        advance(a, a.clock.now+.10)
        frame = a.frame(distance, rpm=previous, stamp=a.clock.now-.04)
        _, _, accepted = decide_commit(a, frame)
        r = a.controller.last_distance_pid_result
        assert accepted and r.output_rpm > 0
        assert r.output_rpm <= r.approach_cap_rpm


@pytest.mark.parametrize("reason,pair,identity", [
    ("stale_vision_result", (0., 0.), True),
    (INVALID_FEEDBACK, (1., 0.), False),
    (INVALID_FEEDBACK, (-5., 3.), True),
    (INVALID_FEEDBACK, (-2., 1.), True),
    (INVALID_FEEDBACK, (4., 4.), True),
    ("emergency_stop", (0., 0.), True),
    ("identity_conflict", (0., 0.), True),
    ("reverse_transition", (0., 0.), True),
    ("lateral_depth:shared_braking_interval_lost", (0., 0.), True),
])
def test_near_restart_keeps_identity_reverse_and_real_stop_constraints(
        authority, setup, monkeypatch, reason, pair, identity):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, distance=1.5745, pair=pair, reason=reason)
    current_identity(a, identity)
    # A later data-gap notice cannot erase an earlier stronger reason.
    a.controller.suspend_longitudinal_authority(a.clock.now, "no_live_grant_before_pi")
    _, _, _ = decide_commit(a, frame)
    assert not a.controller.last_distance_pid_result.pi_fresh_grant_recovery_used


def test_cap220_first_confirmed_sample_progresses_without_resurrecting_old_state(
        authority, setup, monkeypatch):
    a = authority
    frame, old = pending_confirmation(a, setup, monkeypatch)
    assert a.controller._distance_pi_confirmation_restart is not None
    _, actions, accepted = decide_commit(a, frame)
    r = a.controller.last_distance_pid_result
    assert accepted and r.pi_fresh_grant_recovery_used and r.output_rpm == 12
    assert r.pi_sample_dt_sec == 0. and r.pi_integral_frozen
    assert a.controller._distance_pi_confirmation_restart is None
    assert a.owner._depth30_linear_snapshot[3] == frame.distance_state.sample_timestamp != old[3]
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    advance(a, a.clock.now+.01)
    decide_commit(a, frame)
    assert a.controller.last_distance_pid_result.output_rpm == 12
    assert a.owner._depth30_linear_timing.depth_expires_at == deadline
    for _ in range(4):
        stamp = frame.distance_state.sample_timestamp
        advance(a, stamp+.301)
        a.owner._revoke_depth_linear_authority("physical_depth_expired")
        assert a.owner._fresh_depth_linear_snapshot(1) is None
        advance(a, stamp+.45)
        frame = a.frame(2.62, rpm=r.output_rpm, stamp=stamp+.38)
        _, actions, accepted = decide_commit(a, frame)
        previous = r.output_rpm
        r = a.controller.last_distance_pid_result
        assert accepted and previous < r.output_rpm <= previous+12
        assert r.pi_sample_dt_sec == 0.


@pytest.mark.parametrize("fault", ["identity", "reverse", "moving", "feedback_old",
    "depth_old", "not_confirmed", "mismatched_confirmation", "old_sample",
    "uid", "near", "hazard", "obstacle", "parking", "active_reverse", "safety_stop"])
def test_confirmation_restart_requires_all_new_physical_and_identity_evidence(
        authority, setup, monkeypatch, fault):
    a = authority
    frame, old = pending_confirmation(a, setup, monkeypatch)
    if fault == "identity": current_identity(a, False)
    elif fault in {"reverse", "moving", "feedback_old"}:
        changes = {"reverse": dict(left_forward_rpm=-5., right_forward_rpm=3.),
                   "moving": dict(left_forward_rpm=4., right_forward_rpm=4.),
                   "feedback_old": dict(timestamp=a.clock.now-.151)}[fault]
        frame = replace(frame, steering_feedback=replace(frame.steering_feedback, **changes))
    elif fault in {"depth_old", "old_sample", "not_confirmed", "mismatched_confirmation"}:
        changes = {"depth_old": dict(sample_timestamp=a.clock.now-.181),
                   "old_sample": dict(sample_timestamp=old[3]),
                   "not_confirmed": dict(source_detail="depth_multiregion"),
                   "mismatched_confirmation": dict(fusion_mode="depth_radar")}[fault]
        frame = replace(frame, distance_state=replace(frame.distance_state, **changes))
    elif fault == "uid": frame = replace(frame, persons=[replace(frame.persons[0], track_id=2)])
    elif fault == "near":
        frame = replace(frame, distance_m=1.47,
                        distance_state=replace(frame.distance_state, raw_distance_m=1.47))
    elif fault == "hazard": frame = replace(frame, hazard=replace(frame.hazard, active=True))
    elif fault == "obstacle": frame = replace(frame, obstacles=replace(frame.obstacles, front=True))
    elif fault == "parking": a.controller.set_normal_parking(True, 1)
    elif fault == "active_reverse":
        qualify = a.controller._fresh_grant_recovery_step
        def while_reversing(*args):
            a.controller._reverse_active = True
            return qualify(*args)
        monkeypatch.setattr(a.controller, "_fresh_grant_recovery_step", while_reversing)
    elif fault == "safety_stop": a.controller.suspend_longitudinal_authority(a.clock.now, "emergency_stop")
    _, _, _ = decide_commit(a, frame)
    r = a.controller.last_distance_pid_result
    assert r is None or not r.pi_fresh_grant_recovery_used


@pytest.mark.parametrize("reason", ["identity_conflict", "emergency_stop", "reverse_transition",
    "zero_approved", "lateral_depth:shared_braking_momentum", "lateral_depth:shared_braking_interval_lost"])
def test_pending_confirmation_cannot_launder_prior_safety_withdrawal(
        authority, setup, monkeypatch, reason):
    a = authority
    frame, _ = pending_confirmation(a, setup, monkeypatch, reason=reason)
    assert a.controller._distance_pi_confirmation_restart is None
    decide_commit(a, frame)
    assert not a.controller.last_distance_pid_result.pi_fresh_grant_recovery_used


@pytest.mark.parametrize("fault", ["uid", "hazard", "ordinary_reset", "unavailable", "stale"])
def test_pending_marker_is_discarded_when_its_confirmation_lifecycle_ends(
        authority, setup, monkeypatch, fault):
    a = authority
    frame, _ = pending_confirmation(a, setup, monkeypatch)
    assert a.controller._distance_pi_confirmation_restart is not None
    if fault == "ordinary_reset":
        a.controller._reset_distance_pid()
    else:
        interrupted = frame
        if fault == "uid":
            interrupted = replace(frame, persons=[replace(frame.persons[0], track_id=2)])
        elif fault == "hazard":
            interrupted = replace(frame, hazard=replace(frame.hazard, active=True))
        elif fault == "unavailable":
            interrupted = replace(frame, distance_m=None, distance_state=replace(frame.distance_state,
                source_detail="depth_unavailable", sample_timestamp=None, raw_distance_m=None))
        elif fault == "stale":
            interrupted = replace(frame, distance_state=replace(frame.distance_state,
                sample_timestamp=a.clock.now-.301))
        decide_commit(a, interrupted, fresh=False)
    assert a.controller._distance_pi_confirmation_restart is None
    advance(a, a.clock.now+.1)
    frame = replace(frame, distance_state=replace(frame.distance_state, sample_timestamp=a.clock.now-.04),
                    steering_feedback=replace(frame.steering_feedback, timestamp=a.clock.now-.02))
    decide_commit(a, frame)
    assert not a.controller.last_distance_pid_result.pi_fresh_grant_recovery_used


@pytest.mark.parametrize("step,expected", [(0., 0), (.025, 6), (.05, 12)])
def test_pure_first_sample_requires_explicit_bounded_current_assessment_step(step, expected):
    from test_pure_distance_pi import controller, sample, update
    pi = controller()
    evidence = replace(sample(), travel_bound_rpm=1., outer_rpm=1.)
    result = update(pi, wheel=.5, preview_outer_forward_rpm=1., braking_assessment=evidence,
                    depth_expiry_expected_uid=1, fresh_grant_recovery_step_sec=step)
    assert result.output_rpm == expected
    assert result.sample_dt_sec == 0.
    assert result.fresh_grant_recovery_used is (step > 0)
    duplicate = update(pi, wheel=.5, preview_outer_forward_rpm=1., braking_assessment=evidence,
                       depth_expiry_expected_uid=1, fresh_grant_recovery_step_sec=step)
    assert duplicate.output_rpm == expected and duplicate.status == "duplicate"


@pytest.mark.parametrize("fault", ["missing_shared", "wrong_uid", "reverse", "moving_outer",
                                   "parking", "emergency"])
def test_pure_first_sample_cannot_invent_authority_from_a_requested_tick(fault):
    from test_pure_distance_pi import controller, sample, update
    pi = controller()
    evidence = replace(sample(), travel_bound_rpm=5., outer_rpm=1.)
    args = dict(wheel=.5, preview_outer_forward_rpm=1., braking_assessment=evidence,
                depth_expiry_expected_uid=1, fresh_grant_recovery_step_sec=.05)
    if fault == "missing_shared": args["braking_assessment"] = None
    elif fault == "wrong_uid": args["depth_expiry_expected_uid"] = 2
    elif fault == "reverse": args["wheel"] = -1.
    elif fault == "moving_outer":
        args.update(preview_outer_forward_rpm=5., braking_assessment=replace(evidence, outer_rpm=5.))
    elif fault == "parking": pi.set_normal_parking(True)
    elif fault == "emergency": pi.suspend(100., "emergency_stop", reset_execution=True)
    result = update(pi, **args)
    assert not result.fresh_grant_recovery_used and result.output_rpm == 0
