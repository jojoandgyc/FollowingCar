"""CAP226: an old depth deadline is not a new motor-write transaction.

Real controller, admission, runtime receipt readers and backend bookkeeping.
Only the clock/serial driver and active-mode predicate are replaced; no I/O.
"""
import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.action_runtime import MotionActionRuntime
from car_control_modular.longitudinal_execution import ForwardExecutionAnchor
from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig
from test_depth_authority_250 import authority, advance, seed
from test_distance_tracking_response import setup
from test_execution_anchor_admission import AdmissionDriver
from test_lateral_zero_runtime import owner


@pytest.fixture
def boundary(authority):
    a = authority
    stamp, source = seed(a, distance=2.5, rpm=40.)
    timing = a.owner._depth30_linear_timing
    backend = MssdMotorBackend(MssdMotorConfig(
        port="unused-offline", slave_id=1, baudrate=115200, timeout=.1,
        lib_dir="unused-offline", max_target=200, percent_limit=100,
        left_sign=-1, right_sign=1, forward_target_sign=-1,
        m1_is_left_wheel=True, exit_parking_mode_on_arm=False,
        stop_mode="emergency", stop_zero_delay_sec=0., startup_parking_enabled=False,
    ))
    backend.driver = AdmissionDriver()
    a.owner.motor_io_lock = backend.io_lock
    advance(a, stamp + .1932)
    backend.send_targets(40, -40, "FOLLOW20")
    receipt = backend.last_speed_receipt
    anchor = ForwardExecutionAnchor(1, stamp, 40., receipt.completed_at, receipt)
    action = object.__new__(MotionActionRuntime)
    action.owner, action.backend = a.owner, backend
    action._forward_execution_anchor = anchor
    action._visible_wheel_control_active = lambda: True
    action.get_steering_feedback = lambda: a.feedback
    a.owner._action_runtime = action

    # Compile exactly the production bindings, not an ad-hoc reader signature.
    names = {"_longitudinal_execution_reader", "_recent_longitudinal_execution_reader"}
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.__init__)))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Attribute) and target.attr in names
                           for target in node.targets)]
    assert len(assignments) == 2
    exec(compile(ast.Module(body=assignments, type_ignores=[]), __file__, "exec"),
         {"self": a.owner, "PersonTracker": runtime.PersonTracker})
    advance(a, stamp + .249881)
    a.owner._depth30_continuation_veto = (1, stamp)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert action.forward_execution_anchor(1, stamp, a.clock.now) is anchor
    assert action.recent_forward_execution_anchor(1, a.clock.now) is anchor
    frame = a.frame(2.5, rpm=0., stamp=stamp + .157)
    a.feedback = frame.steering_feedback
    return SimpleNamespace(a=a, backend=backend, action=action, anchor=anchor,
                           stamp=stamp, source=source, timing=timing, frame=frame)


def decide(p):
    return p.a.controller.decide(10, p.frame, longitudinal_only=True)


def commit(p, decision):
    return p.a.owner._commit_depth_linear_decision(
        decision, p.frame, 1, is_fresh_depth=True)


@pytest.mark.parametrize("phase", ["pi", "admission", "locked_publication"])
def test_old_depth_expiring_does_not_invalidate_unchanged_completed_packet(
    boundary, monkeypatch, phase,
):
    p = boundary
    controller = p.a.controller
    if phase == "pi":
        original = controller._distance_pid.update

        def update(*args, **kwargs):
            result = original(*args, **kwargs)
            advance(p.a, p.stamp + .2502)
            return result

        monkeypatch.setattr(controller._distance_pid, "update", update)
    decision = decide(p)
    assert controller.last_distance_pid_result.output_rpm > 0
    assert controller._distance_pi_execution_anchor_proof == (
        p.frame.distance_state.sample_timestamp, p.anchor)
    # No state publication, deadline renewal or motor write during calculation.
    assert p.a.owner._depth30_linear_snapshot is p.source
    assert p.a.owner._depth30_linear_timing is p.timing
    assert p.timing.depth_expires_at == pytest.approx(p.stamp + .25)
    if phase == "admission":
        advance(p.a, p.stamp + .2502)
    if phase == "locked_publication":
        original = controller.accept_longitudinal_limit

        def accept(*args):
            original(*args)
            advance(p.a, p.stamp + .2502)

        monkeypatch.setattr(controller, "accept_longitudinal_limit", accept)
    else:
        assert p.action.forward_execution_anchor(1, p.stamp, p.a.clock.now) is None
        assert p.action.recent_forward_execution_anchor(1, p.a.clock.now) is p.anchor
    actions, accepted = commit(p, decision)
    assert accepted and any(action.kind == "forward" and action.speed_percent > 0
                            for action in actions)
    new_stamp = p.frame.distance_state.sample_timestamp
    assert p.a.owner._depth30_linear_snapshot[3] == new_stamp
    assert p.a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(new_stamp + .25)
    assert controller._distance_pi_execution_anchor_proof is None
    assert p.backend.driver.pairs == [(40, -40)]


@pytest.mark.parametrize("phase", ["pi", "admission"])
@pytest.mark.parametrize("event", [
    "zero", "stop", "same_speed_new_receipt", "source_revoked", "receipt_expired",
    "soft_stop", "brake_hold", "wrong_uid", "new_depth_expired",
])
def test_boundary_relaxation_does_not_override_stop_receipt_or_new_sample_guards(
    boundary, monkeypatch, phase, event,
):
    p = boundary

    def change():
        advance(p.a, p.stamp + .2502)
        if event == "zero":
            p.backend.send_targets(0, 0, "NEW_ZERO")
        elif event == "stop":
            p.backend.send_stop("NEW_STOP", mode="emergency")
        elif event == "same_speed_new_receipt":
            p.backend.send_targets(40, -40, "OTHER_WRITER")
        elif event == "source_revoked":
            p.a.owner._depth30_linear_snapshot = None
        elif event == "receipt_expired":
            advance(p.a, p.anchor.sent_at + .100001)
        elif event == "soft_stop":
            p.a.owner._soft_stop_active = True
        elif event == "brake_hold":
            p.a.owner._brake_hold_active = True
        elif event == "wrong_uid":
            p.a.controller.active_target_id = 2
        else:
            advance(p.a, p.frame.distance_state.sample_timestamp + .180001)

    if phase == "pi":
        original = p.a.controller._distance_pid.update

        def update(*args, **kwargs):
            result = original(*args, **kwargs)
            change()
            return result

        monkeypatch.setattr(p.a.controller._distance_pid, "update", update)
    decision = decide(p)
    if phase == "admission":
        change()
    actions, _ = commit(p, decision)
    assert not any(action.kind == "forward" and action.speed_percent > 0 for action in actions)
    assert p.a.owner._depth30_linear_snapshot is None


def test_new_near_measurement_still_brakes_after_old_depth_boundary(boundary, monkeypatch):
    p = boundary
    p.frame = p.a.frame(1.40, rpm=0., stamp=p.stamp + .157)
    p.a.feedback = p.frame.steering_feedback
    original = p.a.controller._distance_pid.update

    def update(*args, **kwargs):
        result = original(*args, **kwargs)
        advance(p.a, p.stamp + .2502)
        return result

    monkeypatch.setattr(p.a.controller._distance_pid, "update", update)
    actions, _ = commit(p, decide(p))
    assert not any(action.kind == "forward" and action.speed_percent > 0 for action in actions)
    assert p.a.owner._depth30_linear_snapshot is None


@pytest.mark.parametrize("offset", [0., -.001])
def test_duplicate_and_older_depth_cannot_use_receipt_as_new_authority(boundary, offset):
    p = boundary
    p.frame = p.a.frame(2.5, rpm=0., stamp=p.stamp + offset)
    advance(p.a, p.stamp + .2502)
    watermark = p.a.owner._depth30_linear_sample_watermark
    actions, _ = commit(p, decide(p))
    assert not any(action.kind == "forward" and action.speed_percent > 0 for action in actions)
    assert p.a.owner._depth30_linear_sample_watermark == watermark
    assert p.a.owner._fresh_depth_linear_snapshot(1) is None
