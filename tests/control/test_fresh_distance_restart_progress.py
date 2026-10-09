"""New physical range can restart after an ordinary gap, not renew its lease.

Real controller and depth admission; fake clock/feedback only, no hardware.
"""
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from test_depth_authority_250 import authority, advance, decide_commit
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def prepare(a, setup, monkeypatch, *, distance=2.6771, pair=(1., 1.),
            gap=.534, age=.164, reason="physical_depth_expired", feedback_age=.034):
    _, a.controller, _ = configured(setup, target_distance_m=1.4,
        distance_pi_kp_per_sec=3., distance_target_motion_control_enable=False,
        distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
        depth_longitudinal_sample_max_age_sec=.30,
        distance_pi_braking_stop_distance_m=1.1,
        distance_approach_deceleration_m_s2=1.,
        distance_approach_response_delay_sec=.15,
        distance_pi_observed_feedback_reserve=True)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC", .30)
    a.owner._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    # This controller/admission test issues no motor writes. There is no
    # completed-command history to add to the independently measured wheels.
    a.controller._braking_execution_bound_reader = lambda uid, now: None
    start = a.clock.now
    _, actions, accepted = decide_commit(a, a.frame(2.5, rpm=40., stamp=start))
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    old_grant = a.owner._depth30_linear_snapshot
    advance(a, start+min(.20, gap+age-.001))
    a.owner._revoke_depth_linear_authority(reason)
    advance(a, start+gap+age)
    frame = a.frame(distance, rpm=.5*sum(pair), stamp=start+gap)
    frame = replace(frame, steering_feedback=replace(frame.steering_feedback,
        left_forward_rpm=pair[0], right_forward_rpm=pair[1],
        timestamp=a.clock.now-feedback_age, yaw_rate_right_dps=0.,
        raw_yaw_rate_right_dps=0.))
    return frame, old_grant


@pytest.mark.parametrize("distance,pair,gap,age,feedback_age", [
    # CAP127: 2.677m, near-still wheels and >100ms Depth/encoder skew.
    (2.6771, (1., 1.), .534, .164, .034),
    # CAP152-shaped independent recovery: 2.711m and zero body speed with a
    # small signed wheel tail. Unknown interval-loss reasons are NOT allowed.
    (2.7112, (-1., 1.), .2446, .098, .02),
    (2.5, (0., 0.), .4, .10, .12),
])
def test_new_far_distance_gets_bounded_nonzero_request_after_ordinary_stop(
        authority, setup, monkeypatch, distance, pair, gap, age, feedback_age):
    a = authority
    frame, old = prepare(a, setup, monkeypatch, distance=distance, pair=pair,
        gap=gap, age=age, feedback_age=feedback_age)
    _, actions, accepted = decide_commit(a, frame)
    r = a.controller.last_distance_pid_result
    assert accepted and r.pi_fresh_grant_recovery_used
    assert r.pi_final_limit_reason == "fresh_grant_recovery_step"
    assert 2 <= r.output_rpm <= min(r.approach_cap_rpm, max(0., .5*sum(pair))+12.)
    assert any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot[3] == frame.distance_state.sample_timestamp != old[3]
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(
        frame.distance_state.sample_timestamp+.30)


@pytest.mark.parametrize("reason", ["physical_depth_expired", "no_live_grant_before_pi",
    "lateral_depth:continuation_feedback_stale", "lateral_depth:continuation_feedback_unavailable"])
def test_repeated_short_ordinary_gaps_do_not_repeat_zero_or_one_rpm(
        authority, setup, monkeypatch, reason):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, reason=reason)
    for _ in range(4):
        _, actions, accepted = decide_commit(a, frame)
        assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
        r = a.controller.last_distance_pid_result
        assert 2 <= r.output_rpm <= 13.
        deadline = a.owner._depth30_linear_timing.depth_expires_at
        stamp = frame.distance_state.sample_timestamp
        advance(a, a.clock.now+.01)
        decide_commit(a, frame)  # Same observation supplies no additional ramp.
        assert a.controller.last_distance_pid_result.output_rpm == r.output_rpm
        assert a.owner._depth30_linear_timing.depth_expires_at == deadline
        a.owner._revoke_depth_linear_authority(reason)
        advance(a, stamp+.45)
        frame = a.frame(2.65, rpm=1., stamp=stamp+.32)


@pytest.mark.parametrize("fault", ["identity", "danger", "unknown_interval", "higher_write",
    "danger_then_gap", "reverse_left", "reverse_right", "yaw", "feedback_stale",
    "feedback_future", "untrustworthy", "near", "close_raw", "parking", "active_reverse",
    "duplicate", "old_depth", "rejected"])
def test_new_restart_cannot_bypass_real_stop_or_invalid_physical_evidence(
        authority, setup, monkeypatch, fault):
    a = authority
    reason = {"identity": "identity_lost", "danger": "emergency_stop",
        "unknown_interval": "lateral_depth:shared_braking_interval_lost",
        "higher_write": "lateral_depth:shared_braking_new_higher_write",
        "danger_then_gap": "emergency_stop"}.get(fault, "physical_depth_expired")
    frame, old = prepare(a, setup, monkeypatch, reason=reason)
    if fault == "danger_then_gap":
        a.controller.suspend_longitudinal_authority(a.clock.now, "physical_depth_expired")
    elif fault in {"reverse_left", "reverse_right", "yaw", "feedback_stale",
                   "feedback_future", "untrustworthy"}:
        changes = {"reverse_left": dict(left_forward_rpm=-6.),
            "reverse_right": dict(right_forward_rpm=-6.),
            "yaw": dict(raw_yaw_rate_right_dps=16.),
            "feedback_stale": dict(timestamp=a.clock.now-.151),
            "feedback_future": dict(timestamp=a.clock.now+.001),
            "untrustworthy": dict(trustworthy=False)}[fault]
        frame = replace(frame, steering_feedback=replace(frame.steering_feedback, **changes))
    elif fault in {"near", "close_raw"}:
        raw = 1.1 if fault == "close_raw" else 1.42
        frame = replace(frame, distance_state=replace(frame.distance_state, raw_distance_m=raw),
                        distance_m=frame.distance_m if fault == "close_raw" else raw)
    elif fault == "parking":
        a.controller.set_normal_parking(True, 1)
    elif fault == "active_reverse":
        qualify = a.controller._fresh_grant_recovery_step
        def while_reversing(*args):
            a.controller._reverse_active = True
            return qualify(*args)
        monkeypatch.setattr(a.controller, "_fresh_grant_recovery_step", while_reversing)
    elif fault in {"duplicate", "old_depth"}:
        stamp = old[3] if fault == "duplicate" else a.clock.now-.181
        frame = replace(frame, distance_state=replace(frame.distance_state, sample_timestamp=stamp))
    elif fault == "rejected":
        a.controller._distance_pid.reject_output(old[3])
    a.feedback = frame.steering_feedback
    a.controller.decide(10, frame, longitudinal_only=True)
    r = a.controller.last_distance_pid_result
    assert r is None or not r.pi_fresh_grant_recovery_used


@pytest.mark.parametrize("interval", [.005, .01, .025, .05])
def test_recovery_tick_cannot_spend_more_than_new_sample_or_decision_progress(
        authority, setup, monkeypatch, interval):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, gap=interval, age=.04,
        reason="no_live_grant_before_pi", pair=(0., 0.), feedback_age=.01)
    a.feedback = frame.steering_feedback
    a.controller.decide(10, frame, longitudinal_only=True)
    r = a.controller.last_distance_pid_result
    assert r.pi_fresh_grant_recovery_step_sec == pytest.approx(interval)
    assert r.output_rpm <= 240.*interval+1e-8
