"""New range -> distance PI -> admission -> wheel writer, with no hardware."""
from dataclasses import replace

import pytest

from test_depth_authority_250 import authority, advance, decide_commit, writer
from test_distance_tracking_response import setup
from test_fresh_distance_restart_progress import prepare
from test_lateral_zero_runtime import owner


@pytest.mark.parametrize("depth_age,feedback_age", [(.1692, .001), (.175, .01), (.10, .12)])
def test_current_feedback_contract_survives_admission_and_motor_write(
        authority, setup, monkeypatch, depth_age, feedback_age):
    a = authority
    frame, old = prepare(a, setup, monkeypatch, distance=1.848,
        pair=(1.5, 1.5), gap=.534, age=depth_age, feedback_age=feedback_age)
    _, actions, accepted = decide_commit(a, frame)
    assessment = a.controller.last_distance_pid_result.pi_braking_assessment
    assert assessment.current_feedback_independent
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot[3] != old[3]
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    assert deadline == pytest.approx(frame.distance_state.sample_timestamp+.30)
    assert a.owner._fresh_depth_linear_snapshot(1) is not None
    action, backend = writer(a)
    action.get_steering_feedback = lambda: a.feedback
    action._service_follow_wheels()
    assert backend.pairs[-1][0] > 0 > backend.pairs[-1][1]
    # A duplicate cannot grow PI or renew the actual capture deadline.
    rpm = a.controller.last_distance_pid_result.output_rpm
    advance(a, a.clock.now+.01)
    decide_commit(a, replace(frame, steering_feedback=a.feedback))
    assert a.controller.last_distance_pid_result.output_rpm == rpm
    assert a.owner._depth30_linear_timing.depth_expires_at == deadline
    # Actual expiry still ends physical motion authority.
    advance(a, deadline+.001)
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (0, 0)


@pytest.mark.parametrize("fault", ["stale_feedback", "future_feedback", "stale_depth",
                                 "changed_uid", "hard_stop", "reverse"])
def test_independent_feedback_is_not_permission_to_bypass_stop(
        authority, setup, monkeypatch, fault):
    a = authority
    frame, _ = prepare(a, setup, monkeypatch, distance=1.848,
        pair=(1.5, 1.5), age=.1692, feedback_age=.001)
    if fault in {"stale_feedback", "future_feedback", "reverse"}:
        changes = {"stale_feedback": dict(timestamp=a.clock.now-.151),
                   "future_feedback": dict(timestamp=a.clock.now+.001),
                   "reverse": dict(left_forward_rpm=-10., right_forward_rpm=-10.)}[fault]
        frame = replace(frame, steering_feedback=replace(frame.steering_feedback, **changes))
    elif fault == "stale_depth":
        frame = replace(frame, distance_state=replace(frame.distance_state,
                         sample_timestamp=a.clock.now-.181))
    _, _, _ = decide_commit(a, frame)
    if fault == "changed_uid":
        a.controller.active_target_id = 2
    elif fault == "hard_stop":
        a.owner._explicit_stop_requested = True
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    action, backend = writer(a)
    action.get_steering_feedback = lambda: a.feedback
    action._service_follow_wheels()
    assert not backend.pairs or all(pair[:2] == (0, 0) for pair in backend.pairs)
