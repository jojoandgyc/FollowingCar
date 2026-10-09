"""Actual range reader -> lateral withdrawal -> new PI grant -> fake wheel I/O."""
from dataclasses import replace

import pytest
import request_0513_modular as runtime
from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from car_control_modular.distance_pi import fresh_grant_recovery_reason_allowed
from test_depth_authority_250 import authority, advance, decide_commit, seed, writer
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner, _intent, _target
from car_control_modular.control_types import ControlAction


def prepare(a, setup):
    _, a.controller, _ = configured(setup, target_distance_m=1.4,
        distance_pi_kp_per_sec=3., distance_pi_launch_request_rpm=180.,
        distance_pi_launch_full_error_m=.5, distance_approach_deceleration_m_s2=.7,
        depth_longitudinal_sample_max_age_sec=.25,
        distance_target_motion_control_enable=False)
    a.owner._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    a.controller._braking_execution_bound_reader = lambda uid, now: 8.
    stamp, old = seed(a, distance=3.0, rpm=8.)
    advance(a, stamp+.15)
    return stamp, old


@pytest.mark.parametrize("cause", ["visibility_expired", "continuation_feedback_stale",
                                    "continuation_feedback_unavailable"])
@pytest.mark.parametrize("quiet", [False, True])
def test_continuity_reason_survives_to_fresh_bounded_wheel_request(authority, setup, cause, quiet):
    a = authority
    stamp, old = prepare(a, setup)
    if cause == "visibility_expired":
        a.owner._validated_visual_observation = ValidatedVisualObservation(
            1, 1, 246, stamp-.1, stamp, stamp+.12, "full")
    elif cause == "continuation_feedback_stale":
        a.feedback = replace(a.feedback, timestamp=a.clock.now-.151)
    else:
        a.feedback = None
    assert a.owner._fresh_depth_linear_snapshot(1, quiet=quiet) is None
    assert a.owner._depth30_read_veto == (1, stamp, cause)
    intent = _intent(a.owner, mode="forward", near_distance_mode=False)
    a.owner._publish_lateral_zero(intent, "pid_zero:visual_pid_left_camera")
    assert a.controller._distance_pi_grant_withdrawal == (1, stamp, "lateral_depth:"+cause)
    assert a.owner._depth30_linear_snapshot is None
    # Restored encoders/visual alone cannot revive the stopped sample.
    advance(a, stamp+.231)
    a.owner._validated_visual_observation = ValidatedVisualObservation(
        1, 1, 250, stamp+.199, a.clock.now, stamp+.4, "full")
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    frame = a.frame(3.018, stamp=stamp+.199, rpm=0., capture_frame_id=250)
    _, actions, accepted = decide_commit(a, frame)
    result = a.controller.last_distance_pid_result
    assert result.pi_fresh_grant_recovery_used
    assert 0 < result.output_rpm <= 12.
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot[3] != old[3]
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp+.449)
    action, backend = writer(a)
    action.get_steering_feedback = lambda: a.feedback
    action._service_follow_wheels()
    assert backend.pairs[-1][0] > 0 and backend.pairs[-1][1] < 0
    # An actual new grant still has its original physical deadline.
    advance(a, stamp+.450)
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (0, 0)


@pytest.mark.parametrize("cause", ["shared_braking_momentum", "scope_invalid",
    "visibility_rejected", "visibility_identity_mismatch", "visibility_sample_mismatch",
    "continuation_feedback_invalid", "continuation_speed_exceeds_bound", "unknown"])
def test_safety_or_unknown_cause_cannot_use_continuity_restart(authority, setup, monkeypatch, cause):
    a = authority
    stamp, _ = prepare(a, setup)
    monkeypatch.setattr(runtime.PersonTracker, "_depth_forward_continuation_limit",
                        lambda *args, **kwargs: (0, cause))
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    a.owner._publish_lateral_zero(_intent(a.owner, mode="forward", near_distance_mode=False), "pid_zero:test")
    assert a.controller._distance_pi_grant_withdrawal[2] == "lateral_depth:"+cause
    assert not fresh_grant_recovery_reason_allowed("lateral_depth:"+cause)
    advance(a, stamp+.23)
    frame = a.frame(3.018, rpm=0., capture_frame_id=250)
    a.feedback = frame.steering_feedback
    a.controller.decide(10, frame, longitudinal_only=True)
    assert not a.controller.last_distance_pid_result.pi_fresh_grant_recovery_used
    assert a.controller.last_distance_pid_result.output_rpm == 0


@pytest.mark.parametrize("fault", ["stop", "identity", "hazard", "duplicate", "expired"])
def test_continuity_reason_does_not_override_new_frame_protection(authority, setup, fault):
    a = authority
    stamp, _ = prepare(a, setup)
    a.feedback = None
    a.owner._fresh_depth_linear_snapshot(1)
    a.owner._publish_lateral_zero(_intent(a.owner, mode="forward", near_distance_mode=False), "pid_zero:test")
    advance(a, stamp+.23)
    frame = a.frame(3.018, rpm=0., capture_frame_id=250)
    if fault == "stop":
        a.owner._explicit_stop_requested = True
        a.controller.suspend_longitudinal_authority(a.clock.now, "emergency_stop")
    elif fault == "identity":
        a.controller.active_target_id = 2
    elif fault == "hazard":
        frame = replace(frame, hazard=replace(frame.hazard, active=True))
    elif fault == "duplicate":
        frame = replace(frame, distance_state=replace(frame.distance_state, sample_timestamp=stamp))
    else:
        frame = replace(frame, distance_state=replace(frame.distance_state, sample_timestamp=a.clock.now-.181))
    _, actions, _ = decide_commit(a, frame)
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)


def test_veto_provenance_is_sample_scoped_and_does_not_age_brake_into_timeout(owner):
    old = ("forward", 10, 1, 99.9)
    owner._depth30_read_veto = (1, 99.9, "shared_braking_momentum")
    assert owner._depth_linear_withdrawal_reason(old, 100.3, "unknown") == "lateral_depth:shared_braking_momentum"
    assert owner._depth_linear_withdrawal_reason(("forward", 10, 1, 100.2), 100.3, "unknown") == "unknown"


def test_reader_cannot_replace_established_brake_with_later_expiry(authority, setup, monkeypatch):
    a = authority
    stamp, old = prepare(a, setup)
    monkeypatch.setattr(runtime.PersonTracker, "_depth_forward_continuation_limit",
                        lambda *args, **kwargs: (0, "shared_braking_momentum"))
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    advance(a, stamp+.251)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.owner._depth_linear_withdrawal_reason(old, a.clock.now, "unknown") == "lateral_depth:shared_braking_momentum"


def test_proven_physical_expiry_is_not_generic_identity_failure(authority, setup):
    a = authority
    stamp, old = prepare(a, setup)
    advance(a, stamp+.251)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.owner._depth30_read_veto == (1, stamp, "physical_depth_expired")
    assert a.owner._depth_linear_withdrawal_reason(old, a.clock.now, "unknown") == "physical_depth_expired"


def test_overtaken_motor_read_cannot_erase_newer_samples_brake_reason(authority, setup):
    a = authority
    stamp, old = prepare(a, setup)
    reason = (1, stamp+.10, "shared_braking_momentum")

    def publish_during_feedback():
        a.owner._depth30_linear_snapshot = (*old[:3], stamp+.10)
        a.owner._depth30_read_veto = reason
        return a.feedback

    a.owner._action_runtime.get_steering_feedback = publish_during_feedback
    assert a.owner._fresh_depth_linear_snapshot(1, quiet=True) is None
    assert a.owner._depth30_read_veto == reason


@pytest.mark.parametrize("fault,reason", [
    ("reject", "visibility_rejected"), ("uid", "visibility_identity_mismatch"),
    ("sample", "visibility_sample_mismatch"), ("future", "visibility_invalid_clock")])
def test_rejected_or_mismatched_visual_proof_is_not_timeout(authority, setup, fault, reason):
    a = authority
    stamp, _ = prepare(a, setup)
    proof = ValidatedVisualObservation(1, 1, 246, stamp, stamp, stamp+.3, "full")
    a.owner._validated_visual_observation = (
        False if fault == "reject" else replace(proof, uid=2) if fault == "uid" else
        replace(proof, continuation_sample_timestamp=stamp+.01) if fault == "sample" else
        replace(proof, validated_at=a.clock.now+.01))
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.owner._depth30_read_veto == (1, stamp, reason)
    assert not fresh_grant_recovery_reason_allowed("lateral_depth:"+reason)


def test_policy_generation_is_published_then_cancelled_with_its_intent(owner):
    owner._lateral_turn_response_policy = (987, True)
    owner._last_control_decision_reason = "visible_follow"
    assert owner._publish_lateral_intent_from_decision(
        width=640, target=_target(), runtime_actions=[ControlAction.forward(20, "follow")],
        control_source="vision", target_steerable=True, low_quality_visible=False)
    intent = owner._lateral_intent_store.snapshot()
    assert owner._lateral_turn_response_policy == (intent.sequence, intent.response_boost_allowed)
    owner._publish_lateral_zero(intent, "center")
    assert owner._lateral_turn_response_policy == (intent.sequence, False)
    owner._clear_lateral_intent("expired")
    assert owner._lateral_turn_response_policy is None


def test_old_zero_cannot_erase_new_policy_generation(owner):
    old = _intent(owner)
    new = _intent(owner)
    owner._lateral_turn_response_policy = (new.sequence, True)
    assert not owner._publish_lateral_zero(old, "expired")
    assert owner._lateral_turn_response_policy == (new.sequence, True)
