"""Production motor readers omit audit work without relaxing motion evidence.

Use real PersonTracker methods and only the constructor's binding expression;
no camera, motor initialization, serial connection or worker is started.
"""
import ast
from dataclasses import replace
import inspect
import textwrap
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import SteeringFeedback
from test_distance_pi_runtime import pi_owner
from test_lateral_zero_runtime import owner, NOW
from test_relative_depth_continuation_runtime import relative, publish


AUDIT_FIELDS = (
    "_last_depth_ff_fallback_timing", "_last_depth_continuation_audit",
    "_last_depth_continuation_feedback_veto", "_last_depth_relative_cap_audit",
    "_last_depth_continuation_cap_audit",
)


def prepare(relative, age=.19):
    stamp, linear = publish(relative)
    relative.clock.now = stamp+age
    relative.owner._last_vision_control_ts = relative.clock.now-.01
    relative.feedback = SteeringFeedback(
        timestamp=relative.clock.now, trustworthy=True,
        left_forward_rpm=16., right_forward_rpm=16.)
    relative.owner._lateral_yaw_revision = 1
    return stamp, linear


def no_audit(owner, monkeypatch):
    sentinels = {name: object() for name in AUDIT_FIELDS}
    for name, value in sentinels.items():
        setattr(owner, name, value)
    monkeypatch.setattr(runtime.logger, "info", lambda *a, **kw: pytest.fail("quiet motor read logged"))
    return sentinels


@pytest.mark.parametrize("age", [.02, .179, .181, .249])
@pytest.mark.parametrize("axes", [False, True])
def test_quiet_read_matches_live_caps_without_touching_audit(relative, monkeypatch, age, axes):
    _, linear = prepare(relative, age)
    o = relative.owner
    timing = o._depth30_linear_timing
    expected = o._follow_wheel_axes(relative.clock.now) if axes else o._fresh_depth_linear_snapshot(1)
    audit = no_audit(o, monkeypatch)
    reads = []
    def feedback():
        reads.append(relative.clock.now)
        return relative.feedback
    o._action_runtime.get_steering_feedback = feedback
    actual = (o._follow_wheel_axes(relative.clock.now, quiet=True) if axes
              else o._fresh_depth_linear_snapshot(1, quiet=True))
    assert actual == expected
    assert len(reads) == 1
    assert o._depth30_linear_snapshot is linear
    assert o._depth30_linear_timing is timing
    assert all(getattr(o, name) is value for name, value in audit.items())


def test_quiet_feedforward_expiry_still_reduces_without_audit(relative, monkeypatch):
    stamp, _ = prepare(relative, .08)
    o = relative.owner
    o._depth30_linear_timing = replace(o._depth30_linear_timing,
        feedforward_timestamp=stamp-.1, feedforward_expires_at=stamp+.04,
        distance_only_percent=10)
    timing = o._depth30_linear_timing
    audit = no_audit(o, monkeypatch)
    actual = o._fresh_depth_linear_snapshot(1, quiet=True)
    assert actual is not None and actual[1] == 10
    assert actual[3] == stamp and o._depth30_linear_timing is timing
    assert all(getattr(o, name) is value for name, value in audit.items())


def test_quiet_cache_read_rechecks_completed_clock_not_prelock_time(relative, monkeypatch):
    stamp, _ = prepare(relative, .15)
    o, reads = relative.owner, []
    before = relative.clock.now
    def read():
        reads.append(before)
        relative.clock.now += .04
        relative.feedback = replace(relative.feedback, timestamp=relative.clock.now)
        return relative.feedback
    o._action_runtime.get_steering_feedback = read
    no_audit(o, monkeypatch)
    current = o._fresh_depth_linear_snapshot(1, now=before, quiet=True)
    assert current is not None and current[3] == stamp
    assert len(reads) == 1
    assert not hasattr(o, "_depth30_continuation_veto")


def test_quiet_feedback_wait_cannot_preserve_expired_physical_authority(relative, monkeypatch):
    stamp, linear = prepare(relative, .23)
    o, reads = relative.owner, []
    timing = o._depth30_linear_timing
    def read():
        reads.append(relative.clock.now)
        relative.clock.now = stamp+.251
        return replace(relative.feedback, timestamp=relative.clock.now)
    o._action_runtime.get_steering_feedback = read
    no_audit(o, monkeypatch)
    assert o._fresh_depth_linear_snapshot(1, now=stamp+.23, quiet=True) is None
    assert len(reads) == 1
    assert o._depth30_linear_snapshot is linear and o._depth30_linear_timing is timing
    assert timing.depth_expires_at == pytest.approx(stamp+.25)


@pytest.mark.parametrize("change", ["uid", "stop", "shutdown", "not_running", "search",
                                     "quality", "grant", "metadata", "brake"])
def test_quiet_cache_wait_cannot_bypass_new_owner_or_veto(relative, monkeypatch, change):
    stamp, linear = prepare(relative)
    o = relative.owner
    def read():
        if change == "uid": o._follow_controller.active_target_id = 2
        elif change == "stop": o._explicit_stop_requested = True
        elif change == "shutdown": o._runtime_shutdown_requested = True
        elif change == "not_running": o.running = False
        elif change == "search": o.search_state = "searching"
        elif change == "quality": o._vision_control_state = "target_visible_low_quality"
        elif change == "brake": o._brake_hold_active = True
        elif change == "metadata":
            o._depth30_linear_timing = replace(o._depth30_linear_timing, distance_only_percent=1)
        else: o._depth30_linear_snapshot = ("forward", linear[1], 1, stamp+.01)
        return relative.feedback
    o._action_runtime.get_steering_feedback = read
    no_audit(o, monkeypatch)
    assert o._fresh_depth_linear_snapshot(1, quiet=True) is None
    if change not in {"grant", "metadata"}:
        assert o._depth30_continuation_veto == (1, stamp)


def test_quiet_bad_feedback_keeps_one_way_safety_veto(relative, monkeypatch):
    stamp, _ = prepare(relative)
    o = relative.owner
    relative.feedback = replace(relative.feedback, timestamp=relative.clock.now-.151)
    audit = no_audit(o, monkeypatch)
    assert o._fresh_depth_linear_snapshot(1, quiet=True) is None
    assert o._depth30_continuation_veto == (1, stamp)
    relative.feedback = replace(relative.feedback, timestamp=relative.clock.now)
    assert o._fresh_depth_linear_snapshot(1, quiet=True) is None
    assert all(getattr(o, name) is value for name, value in audit.items())


def test_nonquiet_axes_keeps_legacy_reader_signature(owner):
    owner._lateral_yaw_revision = 1
    calls = []
    owner._fresh_depth_linear_snapshot = lambda uid, *, now=None: calls.append((uid, now)) or ("forward", 20, 1, NOW-.01)
    assert owner._follow_wheel_axes(NOW)[:2] == (1, 1)
    assert calls == [(1, NOW)]


def test_quiet_axes_does_not_pair_old_uid_with_new_lateral_owner(owner):
    owner._lateral_yaw_revision = 1
    def changed(uid):
        owner._follow_controller.active_target_id = 2
        return False
    owner._has_fresh_lateral_yaw = changed
    assert owner._follow_wheel_axes(NOW, quiet=True) is None


def production_motor_arguments(tracker):
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.__init__)))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Attribute) and target.attr == "_action_runtime"
                           for target in node.targets)]
    assert len(assignments) == 1
    result = {}
    def capture(*args, **kwargs):
        result.update(kwargs)
        return SimpleNamespace(get_steering_feedback=lambda: tracker._test_feedback)
    scope = dict(runtime.__dict__, self=tracker, MotionActionRuntime=capture)
    exec(compile(ast.Module(body=assignments, type_ignores=[]), inspect.getfile(runtime.PersonTracker), "exec"), scope)
    return result


def test_actual_constructor_binds_both_quiet_motor_readers(relative, monkeypatch):
    _, linear = prepare(relative)
    o = relative.owner
    o._test_feedback = relative.feedback
    o._motor_backend = object()  # Constructor call is captured, not executed.
    expected = o._fresh_depth_linear_snapshot(1)
    expected_axes = o._follow_wheel_axes(relative.clock.now)
    args = production_motor_arguments(o)
    # Also validate these ACTUAL constructor keywords against the runtime's
    # real callable signature, without initializing a motor backend.
    inspect.signature(runtime.MotionActionRuntime).bind(
        o, o._motor_backend, runtime.ACTION_RUNTIME_CONFIG, runtime.ACTION_RUNTIME_SYMBOLS, **args)
    audit = no_audit(o, monkeypatch)
    assert args["depth_linear_reader"](1, now=relative.clock.now) == expected
    assert args["follow_axes_reader"](relative.clock.now) == expected_axes
    assert o._depth30_linear_snapshot is linear
    assert all(getattr(o, name) is value for name, value in audit.items())
