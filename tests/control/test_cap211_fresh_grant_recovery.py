"""New far Depth after a withdrawn grant: bounded request, no old lease reuse."""
from dataclasses import replace

import pytest
import request_0513_modular as runtime

from test_depth_authority_250 import authority, advance, decide_commit, seed
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner
from car_control_modular.distance_pi import DistancePiConfig, DistancePiController


CASES = {
    211: (1.8785983776, 1.9078, 1.947, (0., -4.), .177716812, .0375,
          "lateral_zero_no_qualified_depth:pid_zero:visual_pid_right_encoder", 6.5088, 6.5088),
    257: (2.438, 2.5581238588, 2.5581238588, (-1., -1.), .423853811, .084,
          "lateral_zero_no_qualified_depth:revoke:expired_waiting_control_lock", 0., 0.),
    299: (2.3162, 2.3192129817, 2.3320039483, (2., -1.), .176828993, .0698,
          "late_depth_continuation_feedback_invalid", -28.215, 4.8816),
}


def prepare(a, setup, cap=257):
    _, a.controller, _ = configured(
        setup, target_distance_m=1.4, distance_pi_kp_per_sec=3.,
        distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
        distance_pi_motion_memory_sec=.35, distance_approach_deceleration_m_s2=.7,
        depth_longitudinal_sample_max_age_sec=.25)
    a.owner._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    old_distance, used, raw, pair, offset, age, reason, yaw, raw_yaw = CASES[cap]
    stamp, old = seed(a, distance=old_distance, rpm=8.)
    advance(a, stamp+.15)
    a.owner._revoke_depth_linear_authority(reason)
    advance(a, stamp+offset+age)
    current = a.frame(used, stamp=stamp+offset, rpm=.5*sum(pair), capture_frame_id=cap)
    current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=raw),
                      steering_feedback=replace(current.steering_feedback,
                          left_forward_rpm=pair[0], right_forward_rpm=pair[1],
                          yaw_rate_right_dps=yaw, raw_yaw_rate_right_dps=raw_yaw))
    return current, old


@pytest.mark.parametrize("cap", [211, 257, 299])
def test_recorded_signed_tails_get_only_new_bounded_pi_request(authority, setup, cap):
    a = authority
    current, old = prepare(a, setup, cap)
    a.feedback = current.steering_feedback
    a.controller.decide(10, current, longitudinal_only=True)
    result = a.controller.last_distance_pid_result
    assert result.pi_fresh_grant_recovery_used
    assert result.pi_fresh_grant_recovery_step_sec == .05
    assert result.pi_final_limit_reason == "fresh_grant_recovery_step"
    assert 0 < result.output_rpm <= min(result.approach_cap_rpm,
        max(0., .5*sum(CASES[cap][3]))+12.)
    assert a.owner._depth30_linear_snapshot is None  # calculation is not admission
    assert old[3] < a.controller._distance_pid_last_sample_timestamp == current.distance_state.sample_timestamp


@pytest.mark.parametrize("fault", ["near", "closing", "reverse", "yaw", "stale_feedback",
    "untrustworthy", "hazard", "obstacle", "parking", "active_reverse", "identity",
    "emergency_then_continuity", "old_sample", "old_depth", "wrong_uid"])
def test_fresh_recovery_cannot_relabel_an_unsafe_or_unknown_sample(authority, setup, monkeypatch, fault):
    a = authority
    current, _ = prepare(a, setup)
    if fault in {"near", "closing"}:
        distance = 1.89 if fault == "near" else 2.40
        current = replace(current, distance_m=distance,
            distance_state=replace(current.distance_state, raw_distance_m=distance))
    elif fault in {"reverse", "yaw", "stale_feedback", "untrustworthy"}:
        changes = {
            "reverse": dict(left_forward_rpm=-6.),
            "yaw": dict(raw_yaw_rate_right_dps=16.),
            "stale_feedback": dict(timestamp=a.clock.now-.101),
            "untrustworthy": dict(trustworthy=False),
        }[fault]
        current = replace(current, steering_feedback=replace(current.steering_feedback, **changes))
    elif fault == "hazard": current = replace(current, hazard=replace(current.hazard, active=True))
    elif fault == "obstacle": current = replace(current, obstacles=replace(current.obstacles, front=True))
    elif fault == "parking": a.controller._normal_parking_uid = 1
    elif fault == "active_reverse":
        qualify = a.controller._fresh_grant_recovery_step

        def during_reverse(*args):
            a.controller._reverse_active = True
            return qualify(*args)

        monkeypatch.setattr(a.controller, "_fresh_grant_recovery_step", during_reverse)
    elif fault in {"identity", "emergency_then_continuity"}:
        a.controller.suspend_longitudinal_authority(a.clock.now, "identity_lost" if fault == "identity" else "emergency_stop")
        if fault == "emergency_then_continuity":
            a.controller.suspend_longitudinal_authority(a.clock.now, "late_depth_continuation_feedback_invalid")
    elif fault == "wrong_uid": current = replace(current, persons=[replace(current.persons[0], track_id=2)])
    elif fault in {"old_sample", "old_depth"}:
        stamp = a.controller._distance_pid_last_sample_timestamp if fault == "old_sample" else a.clock.now-.181
        current = replace(current, distance_state=replace(current.distance_state, sample_timestamp=stamp))
    a.feedback = current.steering_feedback
    a.controller.decide(10, current, longitudinal_only=True)
    result = a.controller.last_distance_pid_result
    assert result is None or not result.pi_fresh_grant_recovery_used


def test_new_grant_recovery_does_not_extend_old_or_duplicate_depth(authority, setup):
    a = authority
    current, old = prepare(a, setup)
    _, actions, accepted = decide_commit(a, current)
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    stamp = current.distance_state.sample_timestamp
    assert a.owner._depth30_linear_snapshot[3] == stamp != old[3]
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    assert deadline == pytest.approx(stamp+.25)
    prior = a.controller.last_distance_pid_result.output_rpm
    advance(a, a.clock.now+.01)
    decide_commit(a, current)
    assert a.controller.last_distance_pid_result.output_rpm <= prior
    assert a.owner._depth30_linear_timing.depth_expires_at == deadline
    advance(a, deadline+.001)
    assert a.owner._fresh_depth_linear_snapshot(1) is None


def test_cap211_signed_tail_cannot_acquire_relative_continuation(authority, setup, monkeypatch):
    a = authority
    current, _ = prepare(a, setup, 211)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE", True)
    _, actions, accepted = decide_commit(a, current)
    # A fresh request can be admitted, but -4 RPM remains beyond the relative
    # continuation policy's existing 3 RPM tail. It gets no longer lease.
    assert a.controller.last_distance_pid_result.pi_fresh_grant_recovery_used
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_timing.continuation_distance_m is None
    assert a.owner._depth30_linear_timing.continuation_motion is None
    advance(a, current.distance_state.sample_timestamp+.181)
    assert a.owner._fresh_depth_linear_snapshot(1) is None


@pytest.mark.parametrize("rejection_reason", ["depth_dispatch_budget", "physical_depth_expired"])
def test_rejected_fresh_sample_cannot_erase_prior_safety_withdrawal(authority, setup, rejection_reason):
    a = authority
    current, _ = prepare(a, setup)
    a.controller.suspend_longitudinal_authority(a.clock.now, "emergency_stop")
    current = replace(current, steering_feedback=replace(current.steering_feedback,
        left_forward_rpm=4., right_forward_rpm=4.))
    a.feedback = current.steering_feedback
    a.controller.decide(10, current, longitudinal_only=True)
    first = a.controller.last_distance_pid_result
    assert not first.pi_fresh_grant_recovery_used and first.output_rpm == 4
    # No positive admission intervenes: a dispatch-budget refusal must not
    # convert the original safety withdrawal into continuity startup credit.
    a.controller.reject_longitudinal_sample(current.distance_state.sample_timestamp,
                                           rejection_reason)
    assert a.controller._distance_pi_grant_withdrawal[2] == "emergency_stop"
    advance(a, a.clock.now+.03)
    next_frame = a.frame(current.distance_m+.01, stamp=a.clock.now, rpm=4., capture_frame_id=258)
    a.feedback = next_frame.steering_feedback
    a.controller.decide(10, next_frame, longitudinal_only=True)
    result = a.controller.last_distance_pid_result
    assert not result.pi_fresh_grant_recovery_used
    assert result.output_rpm <= 4


@pytest.mark.parametrize("case", ["normal", "large_rise", "near", "reverse", "identity", "parking", "unknown_raw"])
def test_pure_new_grant_step_is_bounded_and_cannot_bypass_stop(case):
    c = DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    args = dict(deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
                ego_forward_rpm=8., raw_distance_m=2.5)
    c.update(2.5, 1.4, sample_timestamp=100., execution_now=100., **args)
    c.suspend(100.1, "identity_lost" if case == "identity" else
              "late_depth_continuation_feedback_invalid", reset_execution=True)
    if case == "parking": c.set_normal_parking(True)
    args.update(ego_forward_rpm=-6. if case == "reverse" else 0.,
                raw_distance_m=None if case == "unknown_raw" else 1.8 if case == "near" else 2.51,
                rise_rpm_per_sec=1000. if case == "large_rise" else 240.)
    r = c.update(1.8 if case == "near" else 2.51, 1.4,
                 sample_timestamp=100.15, execution_now=100.16,
                 fresh_grant_recovery_step_sec=.05, **args)
    if case in {"normal", "large_rise"}:
        assert r.fresh_grant_recovery_used and 0 < r.output_rpm <= 12
    else:
        assert not r.fresh_grant_recovery_used
        assert r.output_rpm == 0
