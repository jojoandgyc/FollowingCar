"""CAP211: one shared feedback contract through real admission and serial ACKs.

All drivers and clocks are fake. Signed feedback stays signed: accepting a
bounded tail is not a claim that the wheels are stationary.
"""
from dataclasses import replace

import pytest

from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner
from test_depth_authority_250 import authority, advance, decide_commit
from test_feedback_interval_execution import execution_case


def forward_case(execution_case, pair=(-1., -2.), *, prime=True):
    a = execution_case()
    # Real new-sample PI and admission, not a manually injected snapshot or
    # terminal cap. Encoder publication then advances independently of Depth.
    _, actions, accepted = decide_commit(a, a.frame(1.947, rpm=46., stamp=a.stamp))
    assert accepted and any(x.kind == "forward" for x in actions)
    assert a.owner._fresh_depth_linear_snapshot(1) == ("forward", 23, 1, a.stamp)
    a.timing = a.owner._depth30_linear_timing
    a.model = a.timing.braking_assessment
    if prime:
        a.action._service_follow_wheels()
        advance(a, a.clock.now+.051)
    a.feedback = replace(a.feedback, timestamp=a.clock.now-.008,
                         left_forward_rpm=pair[0], right_forward_rpm=pair[1])
    a.action._steering_feedback = a.feedback
    return a


@pytest.mark.parametrize("path", ["snapshot", "direct"])
@pytest.mark.parametrize("pair", [(-1., -2.), (-3., 2.), (-3., -3.), (0., -3.)])
def test_small_all_wheel_tail_produces_forward_without_zero(execution_case, path, pair):
    a = forward_case(execution_case, pair)
    before = len(a.backend.driver.pairs)
    original_feedback = a.feedback
    if path == "snapshot":
        a.action._service_follow_wheels()
    else:
        a.action.config.follow_wheel_period_sec = 0.
        with a.owner.motor_io_lock:
            a.action._send_follow_wheel_targets(46, -46, "STEER", visible_required=True)
    assert a.backend.driver.pairs[before:] == [(46, -46)]
    assert not a.backend.driver.stops
    assert a.feedback is original_feedback
    assert (a.feedback.left_forward_rpm, a.feedback.right_forward_rpm) == pair
    assert a.action._visible_wheel_guard.quiet_count == 0
    assert not a.action._visible_wheel_guard.pending_full_reverse
    assert a.action._continuation_executed_speed_history[-1].receipt is a.backend.last_speed_receipt
    assert a.owner._depth30_linear_timing.braking_assessment is a.model
    assert a.owner._depth30_linear_timing.depth_expires_at == a.timing.depth_expires_at
    assert a.timing.depth_expires_at == pytest.approx(a.stamp+.30)


def test_new_quiet_tail_cache_at_final_commit_does_not_invalidate_forward(execution_case, monkeypatch):
    a = forward_case(execution_case, (2., 1.))
    original = a.action._begin_follow_commit
    tail = replace(a.feedback, timestamp=a.clock.now-.001,
                   left_forward_rpm=-1., right_forward_rpm=-2.)
    calls = []

    def commit():
        result = original()
        calls.append(result)
        a.feedback = tail
        a.action._steering_feedback = tail
        return result

    monkeypatch.setattr(a.action, "_begin_follow_commit", commit)
    before = len(a.backend.driver.pairs)
    a.action._service_follow_wheels()
    assert calls and all(calls)
    assert a.backend.driver.pairs[before:] == [(46, -46)]
    assert a.owner._fresh_depth_linear_snapshot(1) is not None


def test_first_real_new_grant_can_send_with_quiet_tail_feedback(execution_case):
    a = forward_case(execution_case, prime=False)
    before = len(a.backend.driver.pairs)
    a.action._service_follow_wheels()
    assert a.backend.driver.pairs[before:] == [(46, -46)]
    assert not a.backend.driver.stops
    assert a.owner._depth30_linear_timing is a.timing


@pytest.mark.parametrize("fault", [
    "spinning_inner", "large_reverse", "feedback_old", "feedback_error",
    "commanded_reverse", "pending_cross", "pending_reverse", "previous_pivot",
    "uid", "identity_rejected", "depth_expired", "stop", "danger", "brake_owner",
])
def test_tail_exception_cannot_bypass_direction_identity_or_stop(execution_case, fault):
    a = forward_case(execution_case)
    guard = a.action._visible_wheel_guard
    if fault == "spinning_inner":
        a.feedback = replace(a.feedback, left_forward_rpm=-3., right_forward_rpm=7.)
    elif fault == "large_reverse":
        a.feedback = replace(a.feedback, left_forward_rpm=-3.01)
    elif fault == "feedback_old":
        a.feedback = replace(a.feedback, timestamp=a.clock.now-.151)
    elif fault == "feedback_error":
        a.feedback = replace(a.feedback, left_error=1)
    elif fault == "commanded_reverse":
        guard.commanded_reverse = True
    elif fault == "pending_cross":
        guard.pending_signs, guard.pending_wheels = (-1, 1), (0,)
    elif fault == "pending_reverse":
        guard.pending_full_reverse = True
    elif fault == "previous_pivot":
        guard.note_sent((-7, 7), a.clock.now-.02)
    elif fault == "uid":
        a.controller.active_target_id = 2
    elif fault == "identity_rejected":
        a.owner._revoke_depth_linear_authority("identity_rejected")
    elif fault == "depth_expired":
        advance(a, a.stamp+.300001)
    elif fault == "stop":
        a.backend.send_stop("test_safety", mode="emergency", preserve_zero=True)
        a.owner._explicit_stop_requested = True
    elif fault == "danger":
        a.owner.person_detected_flag = True
    else:
        a.owner._brake_hold_active = True
    a.action._steering_feedback = a.feedback
    assert not a.action._ordinary_forward_feedback_eligible(1, a.feedback, a.clock.now)
    before = len(a.backend.driver.pairs)
    a.action._service_follow_wheels()
    assert not any(left > 0 > right for left, right in a.backend.driver.pairs[before:])
    if fault == "stop":
        assert a.backend.driver.pairs[before:] == []  # No speed zero overrides STOP.
    assert a.timing.depth_expires_at == pytest.approx(a.stamp+.30)


def test_stop_arriving_at_quiet_tail_final_commit_owns_hardware(execution_case, monkeypatch):
    a = forward_case(execution_case)
    original = a.action._begin_follow_commit
    calls = []

    def commit():
        result = original()
        if not calls:
            calls.append(True)
            a.owner._explicit_stop_requested = True
            a.backend.send_stop("late_stop", mode="emergency", preserve_zero=True)
        return result

    monkeypatch.setattr(a.action, "_begin_follow_commit", commit)
    before = len(a.backend.driver.pairs)
    a.action._service_follow_wheels()
    assert calls and a.backend.driver.stops
    assert a.backend.driver.pairs[before:] == []
    assert a.backend.last_speed_receipt is None


def test_guard_tail_is_opt_in_not_a_stationary_or_reverse_permission(execution_case):
    a = forward_case(execution_case)
    guard = WheelZeroCrossGuard()
    assert guard.limit((46, 46), a.feedback, a.clock.now)[0] == (0, 0)
    assert guard.pending_signs == (1, 1) and guard.pending_wheels == (1,)
    fresh = WheelZeroCrossGuard()
    assert fresh.limit((46, 46), a.feedback, a.clock.now,
                       allow_quiet_forward_tail=True) == ((46, 46), "quiet_forward_tail")
    assert fresh.quiet_count == 0
    assert fresh.limit((-7, 7), a.feedback, a.clock.now,
                       allow_quiet_forward_tail=True)[0] == (0, 0)
