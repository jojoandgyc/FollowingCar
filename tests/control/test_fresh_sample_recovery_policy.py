"""One new physical sample can recover ordinary gaps, independent of old TTL.

Real distance PI, controller and admission with in-memory fixtures only.
"""
from dataclasses import replace

import pytest

from test_depth_authority_250 import authority, advance, decide_commit
from test_distance_tracking_response import setup
from test_fresh_distance_restart_progress import prepare
from test_lateral_zero_runtime import owner


@pytest.mark.parametrize("distance,pair,gap,age", [
    (1.7318, (13., 14.), .30, .12),  # CAP73: envelope48, old request13.
    (1.6974, (8., 9.), .204, .138), # CAP77: old request8.
    (1.6824, (8., 8.), .2344, .12), # CAP82: old request8.
    (1.7660, (7., 7.), .2019, .13), # CAP93: old request7.
    (1.8480, (3., 0.), .461835545, .1199),  # CAP103: request1 -> quantized0.
])
def test_recorded_midrange_recoveries_get_new_bounded_progress(
        authority, setup, monkeypatch, distance, pair, gap, age):
    a = authority
    frame, old = prepare(a, setup, monkeypatch, distance=distance,
                         pair=pair, gap=gap, age=age, feedback_age=.0474)
    _, actions, accepted = decide_commit(a, frame)
    result = a.controller.last_distance_pid_result
    assert accepted and result.pi_fresh_grant_recovery_used
    assert not result.pi_depth_expiry_recovery_used
    assert max(0., sum(pair)/2) < result.output_rpm <= max(0., sum(pair)/2)+12
    assert result.output_rpm <= result.approach_cap_rpm
    assert any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot[3] == frame.distance_state.sample_timestamp != old[3]
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(
        frame.distance_state.sample_timestamp+.30)


@pytest.mark.parametrize("distance", [1.59, 1.848, 1.899, 1.901, 2.4])
@pytest.mark.parametrize("gap", [.025, .149, .151, .299, .451, 1.2])
def test_new_sample_has_no_150ms_expiry_or_190cm_distance_cliff(
        authority, setup, monkeypatch, distance, gap):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, distance=distance, pair=(0., 0.),
        gap=gap, age=.08, feedback_age=.02, reason="no_live_grant_before_pi")
    _, actions, accepted = decide_commit(a, frame)
    r = a.controller.last_distance_pid_result
    assert accepted and r.pi_fresh_grant_recovery_used
    assert r.pi_fresh_grant_recovery_step_sec == pytest.approx(min(.05, gap))
    assert 2 <= r.output_rpm <= min(r.approach_cap_rpm, 240.*min(.05, gap))
    assert any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert r.pi_sample_dt_sec == 0.  # Never integrate the blind interval.


@pytest.mark.parametrize("pair", [(-1., -2.), (-3., 2.), (0., 0.), (1., 2.), (8., 9.), (30., 32.)])
def test_current_wheel_anchor_not_five_rpm_threshold_controls_new_request(
        authority, setup, monkeypatch, pair):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, distance=2.4, pair=pair)
    _, _, accepted = decide_commit(a, frame)
    r = a.controller.last_distance_pid_result
    assert accepted and r.pi_fresh_grant_recovery_used
    assert max(0., sum(pair)/2) < r.output_rpm <= max(0., sum(pair)/2)+12.


def test_repeated_gaps_make_progress_as_measured_wheels_accelerate(authority, setup, monkeypatch):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, distance=2.4, pair=(1., 2.))
    speeds = []
    for _ in range(4):
        _, _, accepted = decide_commit(a, frame)
        assert accepted
        r = a.controller.last_distance_pid_result
        speeds.append(r.output_rpm)
        stamp = frame.distance_state.sample_timestamp
        advance(a, stamp+.301)
        a.owner._revoke_depth_linear_authority("physical_depth_expired")
        assert a.owner._fresh_depth_linear_snapshot(1) is None
        advance(a, stamp+.52)
        frame = a.frame(2.4, rpm=r.output_rpm, stamp=stamp+.4)
    assert speeds == [13, 25, 37, 49]  # Only real wheel response supplies the next anchor.


@pytest.mark.parametrize("gap", [.005, .008])
def test_subquantum_recovery_preview_does_not_invent_a_safety_withdrawal(
        authority, setup, monkeypatch, gap):
    a = authority
    frame, old = prepare(a, setup, monkeypatch, distance=1.848, pair=(0., 0.),
        gap=gap, age=.04, feedback_age=.01, reason="no_live_grant_before_pi")
    _, _, _ = decide_commit(a, frame)
    assert a.owner._depth30_linear_snapshot is None
    assert a.controller._distance_pi_grant_withdrawal == (1, old[3], "no_live_grant_before_pi")
    advance(a, a.clock.now+.05)
    current = a.frame(1.86, rpm=0., stamp=frame.distance_state.sample_timestamp+.05)
    _, actions, accepted = decide_commit(a, current)
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)


@pytest.mark.parametrize("reason", ["identity_conflict", "emergency_stop", "reverse_transition",
    "zero_approved", "lateral_depth:shared_braking_momentum",
    "lateral_depth:shared_braking_interval_lost", "lateral_depth:shared_braking_new_higher_write"])
def test_ordinary_gap_cannot_launder_a_real_stop(authority, setup, monkeypatch, reason):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, distance=1.848, pair=(0., 0.), reason=reason)
    a.controller.suspend_longitudinal_authority(a.clock.now, "no_live_grant_before_pi")
    _, actions, _ = decide_commit(a, frame)
    assert not a.controller.last_distance_pid_result.pi_fresh_grant_recovery_used
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)


@pytest.mark.parametrize("fault", ["reverse", "opposite_wheel", "overspeed", "near", "raw_near",
    "feedback_old", "feedback_future", "depth_old", "depth_future", "hazard", "obstacle",
    "uid", "parking", "untrusted", "brake_latched", "momentum"])
def test_new_recovery_still_requires_complete_current_safety_evidence(
        authority, setup, monkeypatch, fault):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, distance=1.848, pair=(0., 0.))
    changes = {"reverse": dict(left_forward_rpm=-6., right_forward_rpm=-6.),
        "opposite_wheel": dict(left_forward_rpm=-3., right_forward_rpm=7.),
        "overspeed": dict(left_forward_rpm=206.),
        "feedback_old": dict(timestamp=a.clock.now-.151),
        "feedback_future": dict(timestamp=a.clock.now+.001),
        "untrusted": dict(trustworthy=False),
        "momentum": dict(left_forward_rpm=100., right_forward_rpm=100.)}
    if fault in changes:
        frame = replace(frame, steering_feedback=replace(frame.steering_feedback, **changes[fault]))
    elif fault in {"near", "raw_near"}:
        frame = replace(frame, distance_m=1.42 if fault == "near" else frame.distance_m,
            distance_state=replace(frame.distance_state, raw_distance_m=1.42))
    elif fault in {"depth_old", "depth_future"}:
        frame = replace(frame, distance_state=replace(frame.distance_state,
            sample_timestamp=a.clock.now+(.001 if fault == "depth_future" else -.181)))
    elif fault == "hazard": frame = replace(frame, hazard=replace(frame.hazard, active=True))
    elif fault == "obstacle": frame = replace(frame, obstacles=replace(frame.obstacles, front=True))
    elif fault == "uid": frame = replace(frame, persons=[replace(frame.persons[0], track_id=2)])
    elif fault == "parking": a.controller.set_normal_parking(True, 1)
    elif fault == "brake_latched": frame = replace(frame, distance_state=replace(frame.distance_state, brake_latched=True))
    a.feedback = frame.steering_feedback
    a.controller.decide(10, frame, longitudinal_only=True)
    r = a.controller.last_distance_pid_result
    assert r is None or not r.pi_fresh_grant_recovery_used


@pytest.mark.parametrize("delta", [0., -.01])
def test_duplicate_or_older_depth_has_no_new_ramp_budget(authority, setup, monkeypatch, delta):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, distance=1.848, pair=(0., 0.))
    decide_commit(a, frame)
    result = a.controller.last_distance_pid_result
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    advance(a, a.clock.now+.01)
    frame = replace(frame, distance_state=replace(frame.distance_state,
        sample_timestamp=frame.distance_state.sample_timestamp+delta))
    decide_commit(a, frame)
    assert a.controller.last_distance_pid_result.output_rpm == result.output_rpm
    assert a.owner._depth30_linear_timing.depth_expires_at == deadline
