"""A later encoder can repair a direction veto, but cannot renew authority."""
from dataclasses import replace

import pytest

from test_unified_forward_snapshot import feedback, writer
from test_visible_wheel_continuity import visible_runtime
from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard


@pytest.mark.parametrize("fault", [None, "still_spinning", "old_feedback", "old_depth",
    "reverse_owner", "pending_cross", "identity", "stop", "duplicate_clock"])
def test_direction_rejected_zero_gets_one_new_quiet_tail_readmission(monkeypatch, fault):
    rt, owner, driver, clock, _, _ = writer(monkeypatch, base=60., yaw=4.)
    owner._follow_controller.cfg.distance_target_motion_control_enable = False
    source, timing = owner._depth30_linear_snapshot, owner._depth30_linear_timing
    # The older fixture assumed positive feedback when selecting outer speed.
    # Preserve the production shared model's absolute outer-wheel bound for
    # signed tails; this remains a real budget, not a successful limit stub.
    def signed_budget(raw, model_timing, now, *, feedback, quiet):
        assert quiet
        result = model_timing.braking_assessment.budget(now,
            max(abs(feedback.left_forward_rpm), abs(feedback.right_forward_rpm)),
            authorized_rpm=raw[1], execution_bound_rpm=raw[1]+10.)
        return result.cap_rpm, result.reason

    owner._depth_forward_continuation_limit = signed_budget
    get_feedback = rt.get_steering_feedback
    stop = rt._ordinary_snapshot_stop
    changed, reasons = [], []

    def spinning_sample():
        if rt._follow_snapshot_planning and not changed:
            changed.append(True)
            rt._steering_feedback = feedback(clock[0], -4., 4.)
        return get_feedback()

    def after_zero_decision(uid, reason, **kwargs):
        reasons.append(reason)
        if len(reasons) == 1:
            if fault != "duplicate_clock":
                clock[0] += .001
            rt._steering_feedback = feedback(clock[0], -1., -2.)
            if fault == "still_spinning":
                rt._steering_feedback = feedback(clock[0], -3., 7.)
            elif fault == "old_feedback":
                rt._steering_feedback = feedback(clock[0]-.151, -1., -2.)
            elif fault == "old_depth":
                clock[0] = timing.depth_expires_at+.001
                rt._steering_feedback = feedback(clock[0], -1., -2.)
            elif fault == "reverse_owner":
                rt._visible_wheel_guard.commanded_reverse = True
            elif fault == "pending_cross":
                rt._visible_wheel_guard.pending_signs = (-1, 1)
                rt._visible_wheel_guard.pending_wheels = (0,)
            elif fault == "identity":
                owner._vision_control_state = "target_lost"
            elif fault == "stop":
                owner._explicit_stop_requested = True
                rt.backend.send_stop("new_stop", mode="emergency", preserve_zero=True)
        return stop(uid, reason, **kwargs)

    monkeypatch.setattr(rt, "get_steering_feedback", spinning_sample)
    monkeypatch.setattr(rt, "_ordinary_snapshot_stop", after_zero_decision)
    rt._service_follow_wheels()
    assert changed and reasons[0] == "nonordinary_or_feedback_changed"
    if fault is None:
        assert len(reasons) == 1
        assert len(driver.pairs) == 2
        assert all(left > 0 > right for left, right in driver.pairs)
        assert driver.pairs[-1] in {(64, -56), (60, -60)}
        assert not driver.stops
    else:
        assert not any(left > 0 > right for left, right in driver.pairs[1:])
        if fault == "stop":
            assert len(driver.pairs) == 1 and driver.stops
    assert owner._depth30_linear_snapshot == source
    assert owner._depth30_linear_timing is timing
    assert timing.depth_expires_at == pytest.approx(10.25)


def test_same_qualified_feedback_object_refresh_is_not_new_admission_evidence(monkeypatch):
    rt, owner, driver, clock, _, _ = writer(monkeypatch, base=60., yaw=4.)
    owner._follow_controller.cfg.distance_target_motion_control_enable = False
    # Only isolate the ownership prerequisite; all grant/feedback predicates
    # and the supersession decision remain production code.
    monkeypatch.setattr(rt, "_follow_plan_owns_motor", lambda: True)
    rt._follow_snapshot_zero_state = rt._snapshot_zero_evidence()
    rt._steering_feedback = replace(rt._steering_feedback, timestamp=clock[0])
    assert not rt._snapshot_zero_superseded(1)
    assert driver.pairs == [(64, -56)]


@pytest.mark.parametrize("first", [(-1., -2.), (-.5, -.5), (0., 0.), (2., 3.)])
def test_actual_reverse_command_retains_two_sample_confirmation_after_zero(first):
    guard = WheelZeroCrossGuard()
    guard.note_sent((-10, -10), 10.)
    guard.note_sent((0, 0), 10.01)
    assert guard.commanded_reverse
    kwargs = dict(allow_forward_handoff=True, allow_quiet_forward_tail=True,
                  residual_reverse_max_rpm=8.)
    assert guard.limit((46, 46), feedback(10.02, *first), 10.02, **kwargs)[0] == (0, 0)
    assert guard.pending_full_reverse and guard.pending_wheels == (0, 1)
    guard.note_sent((0, 0), 10.02)
    assert guard.limit((46, 46), feedback(10.07), 10.07, **kwargs)[0] == (0, 0)
    # Re-reading the same physical feedback cannot complete confirmation.
    assert guard.limit((46, 46), feedback(10.07), 10.08, **kwargs)[0] == (0, 0)
    pair, _ = guard.limit((46, 46), feedback(10.12), 10.12, **kwargs)
    assert pair == (46, 46)
    guard.note_sent(pair, 10.12)
    assert not guard.commanded_reverse and not guard.pending_full_reverse
    assert guard.limit((46, 46), feedback(10.17, 2., 3.), 10.17, **kwargs)[0] == pair


def test_real_writer_commanded_reverse_cannot_use_residual_forward_shortcut(monkeypatch):
    rt, owner, driver, _, clock = visible_runtime(monkeypatch)
    rt.config.follow_forward_handoff_enable = True
    rt.config.follow_residual_reverse_max_rpm = 8.
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("backward", 10)
    rt.get_steering_feedback = lambda: feedback(clock[0], -10., -10.)
    with owner.motor_io_lock:
        rt._send_follow_wheel_targets(-10, 10, "QUALIFIED_REVERSE")
    assert driver.pairs[-1] == (-10, 10)
    assert rt._visible_wheel_guard.commanded_reverse
    clock[0] += .01
    # A real zero ACK does not itself establish physical stillness.
    rt.backend.send_targets(0, 0, "END_REVERSE")
    rt._visible_wheel_guard.note_sent((0, 0), clock[0])
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 24)
    clock[0] += .01
    rt.get_steering_feedback = lambda: feedback(clock[0], -1., -2.)
    with owner.motor_io_lock:
        rt._send_follow_wheel_targets(24, -24, "NEW_FORWARD")
    assert driver.pairs[-1] == (0, 0)
    assert rt._visible_wheel_guard.pending_full_reverse
    for index in range(2):
        clock[0] += .05
        rt.get_steering_feedback = lambda: feedback(clock[0])
        with owner.motor_io_lock:
            rt._send_follow_wheel_targets(24, -24, "NEW_FORWARD")
        assert driver.pairs[-1] == ((0, 0) if index == 0 else (24, -24))
    assert not rt._visible_wheel_guard.commanded_reverse


def test_normal_pending_single_wheel_handoff_remains_allowed(monkeypatch):
    rt, owner, driver, _, clock = visible_runtime(monkeypatch)
    rt.get_steering_feedback = lambda: feedback(clock[0], 0., -3.)
    with owner.motor_io_lock:
        rt._send_follow_wheel_targets(24, -24, "INITIAL_GUARDED_FORWARD")
    assert driver.pairs[-1] == (0, 0)
    assert rt._visible_wheel_guard.pending_signs
    rt.config.follow_forward_handoff_enable = True
    clock[0] += .05
    rt.get_steering_feedback = lambda: feedback(clock[0], 0., -2.)
    with owner.motor_io_lock:
        rt._send_follow_wheel_targets(24, -24, "QUALIFIED_HANDOFF")
    assert driver.pairs[-1] == (24, -24)
    assert not rt._visible_wheel_guard.pending_signs
