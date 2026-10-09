"""Small-error chase must survive a short expiry, not a stop or stale depth."""
from dataclasses import replace
import threading

import pytest

from car_control_modular.longitudinal_execution import ForwardExecutionAnchor
from test_depth_authority_250 import authority, advance, decide_commit
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


CASES = [
    (1.4868, 1.4943877, 1.4943877, 1.5485901, 12., 5.5, .1971, .2574),
    (1.4943877, 1.5485901, 1.5485901, 1.6438, 4., 2., .2137, .2530),
]


def prepare(a, setup, case, *, proof=True):
    _, a.controller, _ = configured(
        setup, target_distance_m=1.4, distance_pi_kp_per_sec=3.,
        distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
        distance_pi_motion_memory_sec=.30,
        depth_longitudinal_sample_max_age_sec=.25)
    a.owner._follow_controller = a.controller
    a.owner.motor_io_lock = threading.RLock()
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    old_distance, old_raw, distance, raw, sent_rpm, ego, offset, elapsed = case
    first = a.frame(old_distance, rpm=sent_rpm)
    first = replace(first, distance_state=replace(first.distance_state, raw_distance_m=old_raw))
    decide_commit(a, first)
    stamp = first.distance_state.sample_timestamp
    assert a.owner._depth30_linear_snapshot is not None
    completed = [ForwardExecutionAnchor(1, stamp, sent_rpm, stamp+.24, object()) if proof else None]
    a.controller._recent_longitudinal_execution_reader = lambda _uid, _now: completed[0]
    advance(a, stamp+elapsed)
    current = a.frame(distance, rpm=ego, stamp=stamp+offset)
    current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=raw))
    return current, completed


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("proof", [False, True])
def test_new_receding_depth_gets_one_step_without_waiting_for_large_error(authority, setup, case, proof):
    a = authority
    current, _ = prepare(a, setup, case, proof=proof)
    _, actions, accepted = decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert accepted and result.pi_status == "recovering"
    assert result.output_rpm <= min(result.approach_cap_rpm, case[5]+240*.05)
    assert not result.pi_depth_expiry_completed_anchor_used  # No old high RPM near the person.
    if proof:
        assert result.pi_depth_expiry_recovery_step_sec == pytest.approx(.05)
        assert result.output_rpm >= 14
        assert any(x.kind == "forward" and x.speed_percent >= 7 for x in actions)
    else:
        assert result.pi_depth_expiry_recovery_step_sec == 0
        assert result.output_rpm <= case[5]
    assert a.owner._depth30_linear_snapshot[3] == current.distance_state.sample_timestamp


@pytest.mark.parametrize("fault", ["approaching", "flat", "reverse", "yaw", "identity_stop", "packet_changed"])
def test_near_recovery_cannot_borrow_a_packet_without_new_receding_evidence(
        authority, setup, monkeypatch, fault):
    a = authority
    current, completed = prepare(a, setup, CASES[1])
    if fault in {"approaching", "flat"}:
        old_raw = CASES[1][1]
        raw = old_raw-.03 if fault == "approaching" else old_raw
        current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=raw))
    elif fault == "reverse":
        current = replace(current, steering_feedback=replace(
            current.steering_feedback, left_forward_rpm=-1., right_forward_rpm=3.))
    elif fault == "yaw":
        current = replace(current, steering_feedback=replace(
            current.steering_feedback, yaw_rate_right_dps=16.))
    elif fault == "identity_stop":
        a.controller.suspend_longitudinal_authority(a.clock.now, "identity_lost")
    else:
        update = a.controller._distance_pid.update

        def update_then_stop(*args, **kwargs):
            result = update(*args, **kwargs)
            completed[0] = None
            return result

        monkeypatch.setattr(a.controller._distance_pid, "update", update_then_stop)
    _, actions, _ = decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    if fault == "packet_changed":
        assert result.output_rpm == 0
        assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
        assert a.owner._depth30_linear_snapshot is None
    else:
        assert result.pi_depth_expiry_recovery_step_sec == 0


@pytest.mark.parametrize("during_approval", [False, True])
def test_near_recovery_rechecks_completed_packet_at_admission(
        authority, setup, monkeypatch, during_approval):
    a = authority
    current, completed = prepare(a, setup, CASES[1])
    a.feedback = current.steering_feedback
    decision = a.controller.decide(10, current, longitudinal_only=True)
    assert a.controller.last_distance_pid_result.output_rpm == 14
    assert a.controller._distance_pi_expiry_execution_proof is not None
    if during_approval:
        accept = a.controller.accept_longitudinal_limit

        def accept_then_zero(*args, **kwargs):
            accept(*args, **kwargs)
            completed[0] = None

        monkeypatch.setattr(a.controller, "accept_longitudinal_limit", accept_then_zero)
    else:
        completed[0] = None
    actions, _ = a.owner._commit_depth_linear_decision(
        decision, current, 1, is_fresh_depth=True)
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot is None


def test_repeating_recovered_depth_cannot_add_another_step_or_extend_lease(authority, setup):
    a = authority
    current, completed = prepare(a, setup, CASES[1])
    _, _, accepted = decide_commit(a, current)
    assert accepted
    result = a.controller.last_distance_pid_result
    stamp = current.distance_state.sample_timestamp
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    assert a.controller._distance_pi_expiry_execution_proof is None
    completed[0] = None  # New admitted grant no longer depends on the OLD receipt.
    advance(a, a.clock.now + .01)
    _, _, accepted = decide_commit(a, current)
    assert not accepted
    assert a.controller.last_distance_pid_result.output_rpm <= result.output_rpm
    assert a.controller._distance_pid_last_sample_timestamp == stamp
    assert a.owner._depth30_linear_timing.depth_expires_at == deadline


@pytest.mark.parametrize("field,value", [
    ("uid", True), ("uid", 1.), ("uid", 2),
    ("rpm", True), ("rpm", None), ("rpm", float("nan")),
    ("sent_at", None), ("sent_at", float("nan")),
])
def test_completed_packet_requires_valid_identity_and_numeric_proof(authority, setup, field, value):
    a = authority
    current, completed = prepare(a, setup, CASES[1])
    completed[0] = replace(completed[0], **{field: value})
    decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert result.pi_depth_expiry_recovery_step_sec == 0
    assert result.output_rpm <= CASES[1][5]
