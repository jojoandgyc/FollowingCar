"""Visible-follow execution tests; fake serial driver only, no hardware."""
from types import SimpleNamespace

import pytest

from test_depth_drive_rpm import make_runtime
from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard


def feedback(stamp, left=0, right=0, trustworthy=True):
    return SimpleNamespace(timestamp=stamp, left_forward_rpm=left,
                           right_forward_rpm=right, trustworthy=trustworthy)


def visible_runtime(monkeypatch):
    runtime, owner, driver, symbols = make_runtime()
    clock = [10.0]
    monkeypatch.setattr("car_control_modular.action_runtime.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("car_control_modular.action_runtime.time.time", lambda: clock[0])
    owner.running = True
    owner.search_state = "none"
    owner._follow_controller = SimpleNamespace(search_state="none", active_target_id=1)
    owner._vision_control_state = "target_visible_depth_valid"
    owner._depth_longitudinal_authority_enabled = lambda: True
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 24)
    owner._has_fresh_lateral_yaw = lambda uid: True
    owner._lateral_yaw_revision = 1
    runtime.get_steering_feedback = lambda: feedback(clock[0])
    return runtime, owner, driver, symbols, clock


def test_same_direction_wheels_do_not_pause(monkeypatch):
    runtime, owner, driver, s, clock = visible_runtime(monkeypatch)
    assert not runtime.needs_transition_stop(s.steer_right, s.rotate_right)
    assert not runtime.needs_transition_stop(s.rotate_right, s.forward)
    for left, right in ((24, 24), (32, 16), (30, 18)):
        runtime.get_steering_feedback = lambda: feedback(clock[0], 24, 24)
        with owner.motor_io_lock:
            runtime._send_follow_wheel_targets(left, -right, "STEER")
        clock[0] += .05
    assert driver.pairs == [(24, -24), (32, -16), (30, -18)]
    assert not driver.stops


def test_reversal_zeroes_then_releases_on_two_new_quiet_samples(monkeypatch):
    runtime, owner, driver, s, clock = visible_runtime(monkeypatch)
    runtime.get_steering_feedback = lambda: feedback(clock[0], 24, 24)
    runtime.send_yaw_only(8)
    assert driver.pairs[-1] == (0, 0)
    clock[0] += .05
    runtime.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
    runtime.send_yaw_only(8)
    assert driver.pairs[-1] == (0, 0)
    runtime.send_yaw_only(8)  # same encoder sample cannot confirm twice
    assert driver.pairs[-1] == (0, 0)
    clock[0] += .05
    runtime.send_yaw_only(8)
    assert driver.pairs[-1] == (8, 8)  # physical forward +8 / -8
    assert not driver.stops


def test_depth_expiry_removes_base_but_preserves_yaw(monkeypatch):
    runtime, owner, driver, _, _ = visible_runtime(monkeypatch)
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(32, -16, "STEER")
    assert driver.pairs == [(8, 8)]


def test_yaw_zero_preserves_approved_base(monkeypatch):
    runtime, owner, driver, _, _ = visible_runtime(monkeypatch)
    owner._has_fresh_lateral_yaw = lambda uid: False
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(32, -16, "STEER")
    assert driver.pairs == [(24, -24)]


@pytest.mark.parametrize("base,correction,expected", [
    (24, 8, (32, -16)), (4, 8, (12, 4)), (0, 8, (8, 8)),
])
def test_real_steer_dispatch_preserves_signed_base_and_yaw(monkeypatch, base, correction, expected):
    runtime, owner, driver, s, _ = visible_runtime(monkeypatch)
    owner._current_steer_base_percent = base
    owner._current_steer_correction_rpm = correction
    owner.current_command = s.steer_right
    runtime.send_robot_command(s.steer_right)
    assert driver.pairs == [expected]


def test_real_turn_dispatch_also_gates_reversing_wheel(monkeypatch):
    runtime, owner, driver, s, clock = visible_runtime(monkeypatch)
    runtime.get_steering_feedback = lambda: feedback(clock[0], 24, 24)
    owner.current_command = s.rotate_right
    runtime.send_robot_command(s.rotate_right)
    assert driver.pairs == [(0, 0)]
    assert runtime._visible_wheel_waiting


def test_authority_change_during_feedback_read_cancels_write(monkeypatch):
    runtime, owner, driver, _, clock = visible_runtime(monkeypatch)
    def read():
        owner._lateral_yaw_revision += 1
        return feedback(clock[0])
    runtime.get_steering_feedback = read
    runtime.send_percent_drive(24, allow_below_min=True)
    assert not driver.pairs


def test_signed_visible_pair_cannot_fall_back_to_search_after_state_change(monkeypatch):
    runtime, owner, driver, _, _ = visible_runtime(monkeypatch)
    owner.search_state = "searching"
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(12, 4, "STEER", visible_required=True)
    assert not driver.pairs


@pytest.mark.parametrize("state", ["search", "low_quality", "pulse", "shutdown"])
def test_nonvisible_states_keep_legacy_transition_brake(monkeypatch, state):
    runtime, owner, _, s, _ = visible_runtime(monkeypatch)
    if state == "search":
        owner.search_state = "searching"
    elif state == "low_quality":
        owner._vision_control_state = "target_visible_low_quality"
    elif state == "pulse":
        runtime.config.rotate_pulse_brake_enable = True
        owner._current_rotate_pulse_enabled = True
    else:
        owner._runtime_shutdown_requested = True
    assert runtime.needs_transition_stop(s.steer_right, s.rotate_right)


def test_backward_still_requires_transition_stop(monkeypatch):
    runtime, _, _, s, _ = visible_runtime(monkeypatch)
    assert runtime.needs_transition_stop(s.forward, s.backward)
    assert runtime.needs_transition_stop(s.backward, s.steer_right)


def test_forward_reversal_retry_uses_new_feedback_without_half_second_wait(monkeypatch):
    runtime, owner, driver, s, clock = visible_runtime(monkeypatch)
    runtime.get_steering_feedback = lambda: feedback(clock[0], 8, -8)
    assert runtime.forward_like_refresh_due(s.forward, force=True)
    runtime.send_percent_drive(24, allow_below_min=True)
    assert driver.pairs[-1] == (0, 0)
    assert not runtime.forward_like_refresh_due(s.forward)
    for _ in range(2):
        clock[0] += .05
        runtime.get_steering_feedback = lambda: feedback(clock[0])
        assert runtime.forward_like_refresh_due(s.forward)
        runtime.send_percent_drive(24, allow_below_min=True)
    assert driver.pairs[-1] == (24, -24)
    assert not runtime._visible_wheel_waiting


@pytest.mark.parametrize("sample", [None, feedback(9), feedback(11),
    feedback(10, trustworthy=False), feedback(float("nan")),
    feedback(10, left=float("inf"))])
def test_bad_feedback_never_authorizes_motion(sample):
    guard = WheelZeroCrossGuard()
    assert guard.limit((24, 24), sample, 10) == ((0, 0), "feedback_unavailable")


def test_timeout_does_not_blindly_release_reversal():
    guard = WheelZeroCrossGuard()
    assert guard.limit((8, -8), feedback(10, 20, 20), 10)[0] == (0, 0)
    assert guard.limit((8, -8), feedback(10.3, 10, 10), 10.3) == (
        (0, 0), "cross_timeout_zero")


def test_changed_request_cancels_previous_reversal():
    guard = WheelZeroCrossGuard()
    guard.limit((8, -8), feedback(10, 20, 20), 10)
    assert guard.limit((24, 24), feedback(10.05, 20, 20), 10.05) == (
        (24, 24), "continuous")
    assert guard.pending_signs is None


def test_feedback_predating_last_opposite_command_requires_zero():
    guard = WheelZeroCrossGuard()
    guard.note_sent((8, -8), 10)
    assert guard.limit((24, 24), feedback(9.99), 10.01)[0] == (0, 0)


def test_explicit_zero_cancels_pending_even_without_feedback():
    guard = WheelZeroCrossGuard()
    guard.limit((8, -8), feedback(10, 20, 20), 10)
    assert guard.limit((0, 0), None, 10.1) == ((0, 0), "explicit_zero")
    assert guard.pending_signs is None
