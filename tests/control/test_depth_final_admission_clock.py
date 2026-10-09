"""Fresh grants use publication time; existing grants retain their fixed TTL.

Only existing offline fixtures and fake motor I/O are used. No tracker
constructor, device, or runtime thread is started.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence
from test_depth_authority_250 import advance, authority, decide_commit, seed, writer
from test_distance_pi_runtime import pi_owner
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import NOW, owner
from test_relative_depth_continuation_runtime import at_age, publish, relative


def fresh_request(state, stamp, *, kind="forward", with_motion=True):
    controller = state.owner._follow_controller
    controller._distance_pid_last_sample_timestamp = stamp
    controller._braking_rate_source = "raw_depth_window"
    controller._braking_range_rate = .546
    controller._braking_motion_evidence = RawDepthMotionEvidence(stamp, .546, .764, .16, 3)
    controller.last_distance_pid_result = SimpleNamespace(
        approach_mode="distance_pi", output_rpm=116,
        pi_motion_window_used=with_motion, pi_brake_source="raw_relative_motion",
    )
    frame = state.frame(2.5648 if kind == "forward" else .8, stamp=stamp)
    # At175ms, a current encoder report would be >150ms from capture. Use
    # an ordinary recent report that fits both independent age checks.
    frame = replace(frame, steering_feedback=replace(
        frame.steering_feedback, timestamp=state.clock.now-.030))
    state.feedback = frame.steering_feedback
    decision = ControlDecision(
        actions=[getattr(ControlAction, kind)(58, "distance_pi")], reason="distance_pi")
    return frame, decision


def commit(state, frame, decision):
    return state.owner._commit_depth_linear_decision(decision, frame, 1, is_fresh_depth=True)


def delay_evidence(monkeypatch, state, stamp, final_age):
    original = state.owner._depth_continuation_evidence

    def delayed(*args):
        evidence = original(*args)
        state.clock.now = stamp+final_age
        return evidence

    monkeypatch.setattr(state.owner, "_depth_continuation_evidence", delayed)


@pytest.mark.parametrize("with_motion", [True, False])
def test_175_to_181ms_work_cannot_create_first_motor_grant(relative, monkeypatch, with_motion):
    state = relative
    stamp = NOW-.175
    frame, decision = fresh_request(state, stamp, with_motion=with_motion)
    assert state.owner._depth30_linear_snapshot is None
    assert getattr(state.owner._follow_controller, "_distance_pi_execution_anchor_proof", None) is None
    delay_evidence(monkeypatch, state, stamp, .181)
    publications = []
    original_publish = state.owner._publish_depth_linear_pair

    def observed_publication(*args):
        publications.append(args)
        return original_publish(*args)

    monkeypatch.setattr(state.owner, "_publish_depth_linear_pair", observed_publication)
    actions, accepted = commit(state, frame, decision)
    assert accepted and actions[0].speed_percent == 0
    assert actions[0].reason == "depth_admission_expired"
    assert not publications
    assert state.owner._depth30_linear_snapshot is None
    assert state.owner._depth30_linear_timing is None
    assert state.owner.rejections == [(stamp, "depth_admission_expired")]
    assert state.owner._depth30_linear_sample_watermark == (1, stamp)
    action, backend = writer(state)
    action._service_follow_wheels()
    assert not backend.pairs or all(pair[:2] == (0, 0) for pair in backend.pairs)
    # Replaying the rejected physical sample cannot recover its permission.
    assert commit(state, frame, decision) == ([], False)
    assert state.owner._fresh_depth_linear_snapshot(1) is None


@pytest.mark.parametrize("final_age,allowed", [(.175, True), (.179999, True), (.180001, False)])
def test_publication_rechecks_both_sides_of_fresh_window(relative, monkeypatch, final_age, allowed):
    state = relative
    stamp = NOW-.175
    frame, decision = fresh_request(state, stamp)
    delay_evidence(monkeypatch, state, stamp, final_age)
    controller = state.owner._follow_controller
    controller.longitudinal_execution_proof_valid = lambda *_: pytest.fail("No receipt proof was requested")
    actions, accepted = commit(state, frame, decision)
    assert accepted
    assert (actions[0].speed_percent > 0) is allowed
    snapshot = state.owner._fresh_depth_linear_snapshot(1)
    assert (snapshot is not None) is allowed
    if allowed:
        assert snapshot[3] == stamp
        assert state.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp+.25)
        assert not state.owner.rejections


@pytest.mark.parametrize("final_age,allowed", [(.149999, True), (.150001, False)])
def test_dispatch_budget_is_rechecked_without_receipt_proof(relative, monkeypatch, final_age, allowed):
    state = relative
    stamp = NOW-.14
    monkeypatch.setattr(runtime, "MOTOR_RS485_TARGET_MIN_INTERVAL_SEC", .10)
    frame, decision = fresh_request(state, stamp)
    delay_evidence(monkeypatch, state, stamp, final_age)
    actions, accepted = commit(state, frame, decision)
    assert accepted and (actions[0].speed_percent > 0) is allowed
    if not allowed:
        assert actions[0].reason == "depth_dispatch_budget"
        assert state.owner.rejections == [(stamp, "depth_dispatch_budget")]
        assert state.owner._depth30_linear_snapshot is None


def test_reverse_publication_rechecks_its_shorter_dispatch_budget(relative, monkeypatch):
    state = relative
    stamp = NOW-.129
    frame, decision = fresh_request(state, stamp, kind="backward")
    original_info = runtime.logger.info

    def delayed_log(message, *args, **kwargs):
        if message.startswith("depth_linear_limit "):
            state.clock.now = stamp+.131
        original_info(message, *args, **kwargs)

    monkeypatch.setattr(runtime.logger, "info", delayed_log)
    actions, accepted = commit(state, frame, decision)
    assert accepted and actions[0].speed_percent == 0
    assert actions[0].reason == "depth_dispatch_budget"
    assert state.owner._depth30_linear_snapshot is None


def test_late_observation_only_continues_existing_grant_with_original_deadline(relative):
    state = relative
    stamp, _ = publish(state)
    before = state.owner._depth30_linear_timing
    watermark = state.owner._depth30_linear_sample_watermark
    at_age(state, stamp, .210)
    frame = state.frame(10., stamp=stamp+.001)
    decision = ControlDecision(actions=[ControlAction.forward(100, "late")])
    actions, accepted = commit(state, frame, decision)
    current = state.owner._depth30_linear_timing
    assert accepted and actions[0].speed_percent > 0
    assert current.snapshot[3] == stamp
    assert current.depth_expires_at == before.depth_expires_at
    assert current.continuation_motion is before.continuation_motion
    assert state.owner._depth30_linear_sample_watermark == watermark
    assert at_age(state, stamp, .249) is not None
    assert at_age(state, stamp, .251) is None


@pytest.mark.parametrize("veto", ["fresh_window", "dispatch_budget"])
def test_rejected_new_admission_only_tightens_independently_live_previous_grant(
        relative, monkeypatch, veto):
    state = relative
    original_stamp, _ = publish(state)
    old_age = .180 if veto == "fresh_window" else .145
    before = at_age(state, original_stamp, old_age)
    timing = state.owner._depth30_linear_timing
    stamp = state.clock.now-(.175 if veto == "fresh_window" else .14)
    frame, decision = fresh_request(state, stamp)
    if veto == "dispatch_budget":
        monkeypatch.setattr(runtime, "MOTOR_RS485_TARGET_MIN_INTERVAL_SEC", .10)
    final_age = .181 if veto == "fresh_window" else .151
    delay_evidence(monkeypatch, state, stamp, final_age)
    actions, accepted = commit(state, frame, decision)
    current = state.owner._fresh_depth_linear_snapshot(1)
    assert accepted and actions[0].speed_percent > 0
    assert current[3] == original_stamp
    assert 0 < current[1] <= before[1]
    assert state.owner._depth30_linear_timing.depth_expires_at == timing.depth_expires_at
    assert state.owner._depth30_linear_timing.continuation_motion is timing.continuation_motion
    assert state.owner._depth30_linear_timing.accepted_depth_timestamp == timing.accepted_depth_timestamp
    expected_reason = "depth_admission_expired" if veto == "fresh_window" else "depth_dispatch_budget"
    assert state.owner.rejections == [(stamp, expected_reason)]
    # The rejected fresh sample remains watermarked; continuing the old one
    # must not turn either physical sample into a newly approved PI update.
    assert state.owner._depth30_linear_sample_watermark == (1, stamp)
    approvals = list(state.owner.approvals)
    later = at_age(state, original_stamp, .210)
    assert later[3] == original_stamp and 0 < later[1] <= current[1]
    assert state.owner.approvals == approvals
    assert state.owner._depth30_linear_sample_watermark == (1, stamp)


@pytest.mark.parametrize("fault", ["near", "hazard", "uid", "feedback"])
def test_expired_new_admission_cannot_preserve_old_grant_after_safety_veto(
        relative, monkeypatch, fault):
    state = relative
    original_stamp, _ = publish(state)
    at_age(state, original_stamp, .180)
    stamp = state.clock.now-.175
    frame, decision = fresh_request(state, stamp)
    if fault == "near":
        frame = replace(frame, distance_m=1.4, distance_state=replace(
            frame.distance_state, raw_distance_m=1.4))
    elif fault == "hazard":
        frame = replace(frame, hazard=replace(frame.hazard, active=True))
    original_evidence = state.owner._depth_continuation_evidence

    def delayed(*args):
        evidence = original_evidence(*args)
        state.clock.now = stamp+.181
        if fault == "uid":
            state.owner._follow_controller.active_target_id = 2
        elif fault == "feedback":
            state.feedback = None
        return evidence

    monkeypatch.setattr(state.owner, "_depth_continuation_evidence", delayed)
    actions, accepted = commit(state, frame, decision)
    assert accepted and not any(action.speed_percent > 0 for action in actions)
    assert state.owner._depth30_linear_snapshot is None
    assert state.owner._fresh_depth_linear_snapshot(1) is None


def test_real_pi_rejects_new_sample_while_old_grant_continues_and_next_sample_recovers(
        authority, monkeypatch):
    state = authority
    for _ in range(3):
        original_stamp, _ = seed(state)
        advance(state, state.clock.now+.05)
    advance(state, original_stamp+.180)
    before = state.owner._fresh_depth_linear_snapshot(1)
    timing = state.owner._depth30_linear_timing
    stamp = state.clock.now-.175
    frame = state.frame(2.5, rpm=40., stamp=stamp)
    frame = replace(frame, steering_feedback=replace(
        frame.steering_feedback, timestamp=state.clock.now-.030))
    pi = state.controller._distance_pid._distance_pi
    integral = pi.integral_m_s
    with monkeypatch.context() as delayed:
        delay_evidence(delayed, state, stamp, .181)
        _decision, actions, accepted = decide_commit(state, frame)
    current = state.owner._fresh_depth_linear_snapshot(1)
    assert accepted and actions[0].speed_percent > 0
    assert current[3] == original_stamp and 0 < current[1] <= before[1]
    assert state.owner._depth30_linear_timing.depth_expires_at == timing.depth_expires_at
    assert pi.integral_m_s == pytest.approx(integral)
    # Rejection suspends the unexecuted NEW ramp; the original motor lease
    # stays independent. An ordinary fresh sample can recover from feedback.
    assert pi._execution_suspended
    advance(state, state.clock.now+.03)
    current = state.frame(2.5, rpm=40.)
    _decision, actions, accepted = decide_commit(state, current)
    assert accepted and actions[0].speed_percent > 0
    assert state.owner._fresh_depth_linear_snapshot(1)[3] == current.distance_state.sample_timestamp
    assert not pi._execution_suspended
