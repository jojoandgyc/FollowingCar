"""Fresh evidence may continue a real command, never resurrect a stopped one."""
from dataclasses import replace

import pytest

from test_short_expiry_recovery_runtime import pending, decide, commit
from test_depth_authority_250 import authority, advance
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


@pytest.mark.parametrize("reason", [
    "lateral_depth:continuation_feedback_stale",
    "lateral_depth:continuation_feedback_unavailable",
])
def test_new_depth_after_feedback_gap_uses_current_packet_not_fixed_restart(pending, reason):
    p = pending
    controller = p.a.controller
    controller.suspend_longitudinal_authority(p.a.clock.now, reason)
    # Old motion authorization remains invalid. Only the completed packet is
    # remembered, not an intention that never reached the motor.
    assert p.a.owner._fresh_depth_linear_snapshot(1) is None
    decision = decide(p)
    result = controller.last_distance_pid_result
    assert result.pi_execution_recovery_anchor_used
    assert result.pi_status == "execution_continuity"
    assert 90 <= result.output_rpm <= result.approach_cap_rpm < 93
    actions, accepted = commit(p, decision)
    assert accepted and any(a.kind == "forward" and a.speed_percent >= 45 for a in actions)
    p.action._service_follow_wheels()
    assert (0, 0) not in p.backend.driver.pairs


@pytest.mark.parametrize("stopped", [False, True])
def test_real_revocation_clears_old_grant_but_new_depth_can_use_uninterrupted_receipt(pending, stopped):
    p = pending
    _, accepted = commit(p, decide(p))
    assert accepted
    p.action._service_follow_wheels()
    source = p.a.owner._depth30_linear_snapshot
    receipt = p.backend.last_speed_receipt
    advance(p.a, p.a.clock.now+.02)
    p.a.feedback = replace(p.a.feedback, timestamp=p.a.clock.now-.151)
    assert p.a.owner._fresh_depth_linear_snapshot(1) is None
    p.a.owner._revoke_depth_linear_authority("lateral_depth:continuation_feedback_stale")
    assert p.a.owner._depth30_linear_snapshot is None
    assert p.a.controller._distance_pi_grant_withdrawal == (
        source[2], source[3], "lateral_depth:continuation_feedback_stale")
    if stopped:
        p.backend.send_targets(0, 0, "FEEDBACK_GAP_STOP")
    else:
        assert p.backend.last_speed_receipt is receipt
    p.frame = p.a.frame(2.6, rpm=40., stamp=p.a.clock.now-.01)
    p.a.feedback = p.frame.steering_feedback
    decision = decide(p)
    result = p.a.controller.last_distance_pid_result
    assert result.pi_execution_recovery_anchor_used is not stopped
    if stopped:
        assert result.output_rpm <= 52
    else:
        assert result.output_rpm > 75
    actions, accepted = commit(p, decision)
    assert accepted and any(a.kind == "forward" for a in actions)
    assert p.a.owner._depth30_linear_snapshot[3] == p.frame.distance_state.sample_timestamp


@pytest.mark.parametrize("reason", [
    "lateral_depth:visibility_expired", "lateral_depth:continuation_feedback_invalid",
    "lateral_depth:continuation_speed_exceeds_bound", "identity_conflict",
    "momentum_stop", "zero_approved", "admission_rejected",
])
def test_safety_withdrawal_cannot_be_relabelled_as_temporary_feedback_gap(pending, reason):
    p = pending
    p.a.controller.suspend_longitudinal_authority(p.a.clock.now, reason)
    p.a.controller.suspend_longitudinal_authority(
        p.a.clock.now, "lateral_depth:continuation_feedback_stale")
    decision = decide(p)
    result = p.a.controller.last_distance_pid_result
    assert result is None or not result.pi_execution_recovery_anchor_used
    actions, _ = commit(p, decision)
    assert not any(a.kind == "forward" and a.speed_percent > 12 for a in actions)


@pytest.mark.parametrize("event", ["zero", "stop", "other_writer", "feedback_stale", "old_depth", "long_gap"])
def test_feedback_gap_does_not_bypass_current_evidence_or_motor_receipt(pending, event):
    p = pending
    p.a.controller.suspend_longitudinal_authority(
        p.a.clock.now, "lateral_depth:continuation_feedback_stale")
    if event == "zero":
        p.backend.send_targets(0, 0, "TEST_ZERO")
    elif event == "stop":
        p.backend.send_stop("TEST_STOP", mode="emergency")
    elif event == "other_writer":
        p.backend.send_targets(84, -84, "OTHER_WRITER")
    elif event == "feedback_stale":
        p.frame = replace(p.frame, steering_feedback=replace(
            p.frame.steering_feedback, timestamp=p.a.clock.now-.151))
        p.a.feedback = p.frame.steering_feedback
    elif event == "old_depth":
        p.frame = replace(p.frame, distance_state=replace(
            p.frame.distance_state, sample_timestamp=p.old_stamp))
    else:
        advance(p.a, p.old_stamp+.36)
        p.frame = p.a.frame(2.6, rpm=0., stamp=p.a.clock.now-.01)
        p.a.feedback = p.frame.steering_feedback
    actions, _ = commit(p, decide(p))
    result = p.a.controller.last_distance_pid_result
    assert result is None or not result.pi_execution_recovery_anchor_used
    assert not any(a.kind == "forward" and a.speed_percent > 12 for a in actions)


@pytest.mark.parametrize("event", ["zero", "stop", "other_writer"])
def test_recovery_receipt_changed_inside_admission_is_rejected(pending, monkeypatch, event):
    p = pending
    p.a.controller.suspend_longitudinal_authority(
        p.a.clock.now, "lateral_depth:continuation_feedback_stale")
    decision = decide(p)
    assert p.a.controller._distance_pi_recovery_execution_proof is not None
    original = p.a.controller.accept_longitudinal_limit

    def approve_then_replace(sample, rpm):
        original(sample, rpm)
        if event == "stop":
            p.backend.send_stop("STOP_DURING_ADMISSION", mode="emergency")
        else:
            pair = (0, 0) if event == "zero" else (84, -84)
            p.backend.send_targets(*pair, "WRITER_DURING_ADMISSION")

    monkeypatch.setattr(p.a.controller, "accept_longitudinal_limit", approve_then_replace)
    actions, _ = commit(p, decision)
    assert not any(a.kind == "forward" and a.speed_percent > 0 for a in actions)
    assert p.a.owner._depth30_linear_snapshot is None


def test_recovery_receipt_final_check_and_publication_are_atomic(pending, monkeypatch):
    p = pending
    p.a.controller.suspend_longitudinal_authority(
        p.a.clock.now, "lateral_depth:continuation_feedback_stale")
    decision = decide(p)
    original_check = p.a.controller.longitudinal_execution_proof_valid
    original_publish = p.a.owner._publish_depth_linear_pair
    checks, publishes = [], []

    def checked(*args):
        checks.append(p.backend.io_lock.locked())
        return original_check(*args)

    def published(*args):
        publishes.append(p.backend.io_lock.locked())
        assert p.backend.io_lock.locked()
        return original_publish(*args)

    monkeypatch.setattr(p.a.controller, "longitudinal_execution_proof_valid", checked)
    monkeypatch.setattr(p.a.owner, "_publish_depth_linear_pair", published)
    with p.a.owner._control_update_lock:
        actions, accepted = commit(p, decision)
    assert accepted and any(a.kind == "forward" and a.speed_percent > 0 for a in actions)
    assert checks == [False, True]
    assert publishes == [True]
    assert not p.backend.io_lock.locked()


@pytest.mark.parametrize("reason", [
    "late_depth_safety", "decision_safety_or_target:hazard", "active_target_cleared",
    "late_depth_near_or_untrusted", "depth_safety:stop", "visual_target_missing_or_ambiguous",
])
def test_safety_revoke_after_snapshot_removed_still_invalidates_receipt_memory(pending, reason):
    p = pending
    p.a.owner._revoke_depth_linear_authority("lateral_depth:continuation_feedback_stale")
    assert p.a.owner._depth30_linear_snapshot is None
    p.a.owner._revoke_depth_linear_authority(reason)
    assert p.a.controller._distance_pi_grant_withdrawal[2] == reason
    # A subsequent benign timeout cannot erase the known safety event.
    p.a.owner._revoke_depth_linear_authority("lateral_depth:continuation_feedback_stale")
    decision = decide(p)
    result = p.a.controller.last_distance_pid_result
    assert not result.pi_execution_recovery_anchor_used
    actions, _ = commit(p, decision)
    assert not any(a.kind == "forward" and a.speed_percent > 12 for a in actions)
