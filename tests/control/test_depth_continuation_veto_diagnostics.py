"""Report the actual lost evidence, without relaxing admission checks."""
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from test_depth_authority_250 import authority, decide_commit
from test_distance_tracking_response import setup
from test_fresh_distance_restart_progress import prepare
from test_lateral_zero_runtime import owner


def measured(a, setup, monkeypatch):
    frame, _ = prepare(a, setup, monkeypatch, distance=2.6, pair=(5., 3.),
                       age=.085, feedback_age=.01)
    decide_commit(a, frame)
    assert a.controller.last_distance_pid_result.pi_braking_assessment.outer_rpm == 5.
    return frame


def test_cap223_reverse_pair_is_not_reported_as_no_model(authority, setup, monkeypatch):
    a = authority
    frame = measured(a, setup, monkeypatch)
    reverse = replace(frame, steering_feedback=replace(frame.steering_feedback, left_forward_rpm=-5.))
    assessment = a.controller.last_distance_pid_result.pi_braking_assessment
    assert assessment.valid_for(1, frame.distance_state.sample_timestamp)
    assert a.owner._depth_continuation_evidence(reverse, 1, 10, a.clock.now) == (None, None)
    assert a.owner._depth_continuation_evidence_veto == "feedback_reverse_exceeds_tail"
    # A later valid evaluation must not inherit this observation's reason.
    assert a.owner._depth_continuation_evidence(frame, 1, 10, a.clock.now)[0] == pytest.approx(2.6)
    assert a.owner._depth_continuation_evidence_veto is None


@pytest.mark.parametrize("fault,reason", [
    ("feedback_stale", "feedback_expired_or_future"),
    ("feedback_future", "feedback_expired_or_future"),
    ("missing", "distance_or_feedback_unavailable"),
    ("untrustworthy", "feedback_untrustworthy"),
    ("hazard", "hazard_or_obstacle"),
    ("brake", "depth_safety_or_brake_latched"),
    ("uid", "target_not_present"),
])
def test_veto_is_specific_without_changing_rejection(authority, setup, monkeypatch, fault, reason):
    a = authority
    frame = measured(a, setup, monkeypatch)
    if fault == "missing": frame = replace(frame, steering_feedback=None)
    elif fault == "untrustworthy":
        frame = replace(frame, steering_feedback=replace(frame.steering_feedback, trustworthy=False))
    elif fault.startswith("feedback_"):
        stamp = a.clock.now+(.001 if fault == "feedback_future" else -.151)
        frame = replace(frame, steering_feedback=replace(frame.steering_feedback, timestamp=stamp))
    elif fault == "hazard": frame = replace(frame, hazard=replace(frame.hazard, active=True))
    elif fault == "brake": frame = replace(frame, distance_state=replace(frame.distance_state, brake_latched=True))
    elif fault == "uid": frame = replace(frame, persons=[])
    assert a.owner._depth_continuation_evidence(frame, 1, 10, a.clock.now) == (None, None)
    assert a.owner._depth_continuation_evidence_veto == reason


def test_admission_log_preserves_actual_rejection_details(authority, setup, monkeypatch):
    a = authority
    frame = measured(a, setup, monkeypatch)
    records = []
    original = a.owner._depth_continuation_evidence
    # Encoder changes before admission: keep its real validator and matching
    # model, only supply the independently arrived wheel report.
    def evidence(current, uid, percent, now):
        current = replace(current, steering_feedback=replace(current.steering_feedback, left_forward_rpm=-5.))
        return original(current, uid, percent, now)
    a.owner._depth_continuation_evidence = evidence
    monkeypatch.setattr(runtime.logger, "info", lambda message, *args, **kwargs: records.append(message % args))
    # The first model was already admitted; use a genuinely newer measurement.
    a.clock.now += .05
    fresh = replace(frame, distance_state=replace(frame.distance_state,
        sample_timestamp=frame.distance_state.sample_timestamp+.05),
        steering_feedback=replace(frame.steering_feedback, timestamp=frame.steering_feedback.timestamp+.05))
    decide_commit(a, fresh)
    assert any("admission_veto" in line and "evidence_veto=feedback_reverse_exceeds_tail" in line for line in records)
