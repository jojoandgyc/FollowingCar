"""Production bindings -> PI -> depth admission -> writer; fake serial only."""
import ast
import inspect
import textwrap
from dataclasses import replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.controllers import FollowSafetyController
from car_control_modular.longitudinal_execution import ForwardExecutionAnchor
from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig
from test_depth_authority_250 import authority, advance, seed, writer
from test_distance_tracking_response import setup
from test_execution_anchor_admission import AdmissionDriver
from test_lateral_zero_runtime import owner


@pytest.fixture
def pending(authority, monkeypatch):
    a = authority
    a.controller = FollowSafetyController(replace(a.controller.cfg,
        target_distance_m=1.4, distance_pi_kp_per_sec=3.,
        distance_target_motion_control_enable=False,
        distance_pi_braking_stop_distance_m=1.1,
        distance_approach_deceleration_m_s2=1., distance_approach_response_delay_sec=.15,
        distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
        visible_steering_pid_max_correction_rpm=10.))
    a.controller.active_target_id = 1
    a.controller._has_seen_person = True
    a.owner._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    a.controller._braking_execution_bound_reader = lambda uid, now: 92.
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE", False)
    stamp, source = seed(a, distance=2.65, rpm=92.)
    old_timing = a.owner._depth30_linear_timing
    backend = MssdMotorBackend(MssdMotorConfig(
        port="unused-offline", slave_id=1, baudrate=115200, timeout=.1,
        lib_dir="unused-offline", max_target=200, percent_limit=100,
        left_sign=-1, right_sign=1, forward_target_sign=-1,
        m1_is_left_wheel=True, exit_parking_mode_on_arm=False,
        stop_mode="emergency", stop_zero_delay_sec=0., startup_parking_enabled=False))
    backend.driver = AdmissionDriver()
    action, _unused = writer(a)
    action.backend = backend
    action.get_steering_feedback = lambda: a.feedback
    action.continuation_executed_speed_bound_rpm = lambda uid, now: 92.
    a.owner.motor_io_lock = backend.io_lock
    advance(a, stamp+.2523-.062064143)
    # Actual completed backend writes, with protocol wheel signs. The proof
    # is associated with this valid old grant, never an unexecuted request.
    assert a.owner._fresh_depth_linear_snapshot(1)[1]*2 >= 84
    backend.send_targets(84, -84, "FOLLOW20")
    receipt = backend.last_speed_receipt
    action._forward_execution_anchor = ForwardExecutionAnchor(
        1, stamp, 84., receipt.completed_at, receipt)
    names = {"_longitudinal_execution_reader", "_recent_longitudinal_execution_reader",
             "_longitudinal_recovery_reader", "_longitudinal_recovery_validator"}
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.__init__)))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr in names for t in node.targets)]
    assert len(assignments) == 4
    exec(compile(ast.Module(body=assignments, type_ignores=[]), __file__, "exec"),
         {"self": a.owner, "PersonTracker": runtime.PersonTracker})
    advance(a, stamp+.2523)
    frame = a.frame(2.595857142857143, rpm=0., stamp=stamp+.193882308)
    frame = replace(frame, distance_state=replace(frame.distance_state,
        raw_distance_m=2.5721217400714202))
    a.feedback = frame.steering_feedback
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.owner._depth30_linear_snapshot is source
    assert a.controller._longitudinal_recovery_reader(1, a.clock.now) is not None
    return SimpleNamespace(a=a, action=action, backend=backend, frame=frame,
                           old_stamp=stamp, source=source, old_timing=old_timing)


def decide(p):
    return p.a.controller.decide(10, p.frame, longitudinal_only=True)


def commit(p, decision):
    return p.a.owner._commit_depth_linear_decision(decision, p.frame, 1, is_fresh_depth=True)


def test_current_84rpm_packet_allows_fresh_92rpm_budget_without_24rpm_restart(pending):
    p = pending
    decision = decide(p)
    result = p.a.controller.last_distance_pid_result
    assert result.pi_execution_recovery_anchor_used
    assert result.pi_status == "execution_continuity"
    assert 90 <= result.output_rpm <= result.approach_cap_rpm < 93
    assert p.a.owner._depth30_linear_snapshot is p.source
    assert p.a.owner._depth30_linear_timing is p.old_timing
    assert p.old_timing.depth_expires_at == pytest.approx(p.old_stamp+.25)
    assert p.backend.driver.pairs == [(84, -84)]
    actions, accepted = commit(p, decision)
    assert accepted and any(a.kind == "forward" and a.speed_percent >= 45 for a in actions)
    new_stamp = p.frame.distance_state.sample_timestamp
    assert p.a.owner._depth30_linear_snapshot[3] == new_stamp
    assert p.a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(new_stamp+.25)
    assert p.a.controller._distance_pi_recovery_execution_proof is None
    # Publication consumed the old proof. It cannot be re-associated with the
    # next source sample unless the new command is truly written afterwards.
    assert p.a.controller._longitudinal_recovery_reader(1, p.a.clock.now) is None
    p.action._service_follow_wheels()
    assert p.backend.driver.pairs[-1][0] >= 88
    assert p.backend.driver.pairs[-1][1] <= -88
    assert (0, 0) not in p.backend.driver.pairs


def test_recovery_does_not_also_require_old_100ms_receipt_window(pending, monkeypatch):
    p = pending
    original = p.a.controller._distance_pid.update

    def slower(*args, **kwargs):
        result = original(*args, **kwargs)
        advance(p.a, p.a.clock.now+.041)
        return result

    monkeypatch.setattr(p.a.controller._distance_pid, "update", slower)
    decision = decide(p)
    assert p.a.controller.last_distance_pid_result.pi_execution_recovery_anchor_used
    assert p.a.controller._recent_longitudinal_execution_reader(1, p.a.clock.now) is None
    assert p.a.controller._distance_pi_expiry_execution_proof is None
    actions, accepted = commit(p, decision)
    assert accepted and any(a.kind == "forward" and a.speed_percent > 12 for a in actions)


@pytest.mark.parametrize("phase", ["before_pi", "during_pi", "admission"])
@pytest.mark.parametrize("event", ["zero", "stop", "reverse", "same_speed_new_receipt", "uid", "park"])
def test_zero_stop_and_identity_change_invalidate_execution_memory(pending, monkeypatch, phase, event):
    p = pending

    def change():
        if event == "zero": p.backend.send_targets(0, 0, "OTHER_ZERO")
        elif event == "stop": p.backend.send_stop("OTHER_STOP", mode="emergency")
        elif event == "reverse": p.backend.send_targets(-30, 30, "OTHER_REVERSE")
        elif event == "same_speed_new_receipt": p.backend.send_targets(84, -84, "OTHER_WRITER")
        elif event == "uid": p.a.controller.active_target_id = 2
        else: p.a.owner._brake_hold_active = True

    if phase == "before_pi": change()
    elif phase == "during_pi":
        original = p.a.controller._distance_pid.update

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            change()
            return result

        monkeypatch.setattr(p.a.controller._distance_pid, "update", changed)
    decision = decide(p)
    if phase == "admission": change()
    actions, _ = commit(p, decision)
    if phase == "before_pi":
        result = p.a.controller.last_distance_pid_result
        assert result is None or not result.pi_execution_recovery_anchor_used
        # Independently qualified new evidence may still use normal measured
        # recovery; it must not inherit the obsolete high command.
        assert not any(a.kind == "forward" and a.speed_percent > 12 for a in actions)
    else:
        assert not any(a.kind == "forward" and a.speed_percent > 0 for a in actions)


@pytest.mark.parametrize("fault", ["hazard", "obstacle", "feedback_stale", "reverse_feedback", "replay", "old_depth", "late_depth"])
def test_new_evidence_guards_remain_independent_of_positive_packet(pending, fault):
    p = pending
    if fault == "hazard": p.frame = replace(p.frame, hazard=replace(p.frame.hazard, active=True))
    elif fault == "obstacle": p.frame = replace(p.frame, obstacles=replace(p.frame.obstacles, front=True))
    elif fault in {"feedback_stale", "reverse_feedback"}:
        p.frame = replace(p.frame, steering_feedback=replace(p.frame.steering_feedback,
            timestamp=p.a.clock.now-.151 if fault == "feedback_stale" else p.a.clock.now,
            left_forward_rpm=-1. if fault == "reverse_feedback" else 0.))
        p.a.feedback = p.frame.steering_feedback
    else:
        stamp = p.old_stamp if fault == "replay" else p.old_stamp-.001 if fault == "old_depth" else p.a.clock.now-.181
        p.frame = replace(p.frame, distance_state=replace(p.frame.distance_state,
            sample_timestamp=stamp, sample_age_sec=p.a.clock.now-stamp))
    actions, _ = commit(p, decide(p))
    result = p.a.controller.last_distance_pid_result
    assert result is None or not result.pi_execution_recovery_anchor_used
    assert not any(a.kind == "forward" and a.speed_percent > 12 for a in actions)
