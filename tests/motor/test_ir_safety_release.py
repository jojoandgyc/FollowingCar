"""Right-IR hold/release regression: real executor, fake clock/driver/sensors only.

No serial client is imported or opened.  The safety callback represents the
existing upstream hazard decision; these tests do not change IR thresholds.
"""
from dataclasses import replace
import queue
import threading
from types import SimpleNamespace

import pytest

from car_control_modular.action_command import ActionCommandSnapshot
from car_control_modular.control_types import SteeringFeedback
from test_follow_wheel_periodic import setup_periodic


@pytest.fixture
def ir_runtime(monkeypatch):
    rt, owner, driver, symbols, clock, axes = setup_periodic(monkeypatch)
    hazards = {"right_ir": True}
    checks = []

    def safety_check(action=None):
        checks.append(action)
        return any(hazards.values())

    rt.hard_stop_check = safety_check
    rt.get_steering_feedback = lambda: SteeringFeedback(timestamp=clock[0], trustworthy=True)
    owner.current_command = None
    owner.action_stop_event = threading.Event()
    owner.action_queue_lock = threading.Lock()
    owner.command_lock = threading.Lock()
    owner.stop_action_execution = owner.person_detected_flag = False
    owner._action_command_revision = 1
    owner._last_hard_stop_check_ts = 0.
    owner._hard_stop_check_interval_sec = .01
    owner._brake_hold_refresh_interval_sec = .75
    owner._rotate_follows_previous_rotate = False
    rt.config.enable_motor_rpm_feedback = False
    rt.config.action_intent_stale_sec = .45
    rt.config.follow_brake_distance_m = .8
    rt.config.rotate_chain_memory_sec = .5
    rt.config.rotate_turn_percent_chain = 10
    rt.config.rotate_pulse_settle_feedback_stale_sec = .3

    # Each invocation runs a bounded, real executor iteration; no background
    # thread or physical sleep is needed, including early safety-service exits.
    monkeypatch.setattr("car_control_modular.action_runtime.time.sleep",
                        lambda _seconds: owner.action_stop_event.set())
    monkeypatch.setattr(rt.logger, "error",
                        lambda message, *args: pytest.fail(message % args))
    return SimpleNamespace(runtime=rt, owner=owner, driver=driver, symbols=symbols,
                           clock=clock, axes=axes, hazards=hazards, checks=checks)


def run_once(case, command=None):
    owner = case.owner

    class OneIterationQueue(queue.Queue):
        def get_nowait(self):
            owner.action_stop_event.set()
            return super().get_nowait()

    owner.action_queue = OneIterationQueue()
    if command is not None:
        owner.action_queue.put(command)
    owner.action_stop_event.clear()
    case.runtime.run_loop()


def publish(case, action_name="forward"):
    """A new producer revision with fresh canonical axes, not a stale integer."""
    owner, s = case.owner, case.symbols
    action = getattr(s, action_name)
    yaw = -8 if action_name.endswith("left") else 8
    base = 0 if action in s.rotate_actions else 24
    case.axes[:] = [base, yaw if action != s.forward else 0,
                    case.clock[0] + .18, case.clock[0] + .18]
    owner._current_forward_percent = 24
    owner._current_steer_base_percent = base
    owner._current_steer_correction_rpm = abs(case.axes[1])
    owner._lateral_yaw_revision += 1
    owner._action_command_revision += 1
    owner._last_action_intent_ts = case.clock[0]
    return ActionCommandSnapshot(
        action=action, revision=owner._action_command_revision,
        enqueued_at=case.clock[0], control_frame=4, capture_frame_id=21,
        capture_timestamp=case.clock[0] - .02,
        reason="longitudinal_distance_pid" if base else "visual_pid_yaw",
        soft_stop=False,
    )


def stop_for_right_ir(case):
    case.runtime.send_stop_with_brake_hold("right_ir")
    assert case.owner._brake_hold_active
    assert case.owner._brake_hold_stop_mode == "emergency"
    assert case.owner._brake_hold_label == "safety_hold_right_ir"
    assert case.driver.stops == [1]
    assert case.driver.pairs == []


def test_right_ir_stop_clears_motion_and_cannot_be_softened(ir_runtime):
    case = ir_runtime
    case.owner._use_soft_stop_next = case.owner._soft_stop_active = True
    case.owner.is_forwarding = True
    stop_for_right_ir(case)
    assert case.owner._current_forward_percent == 0
    assert case.owner._current_steer_base_percent == 0
    assert not case.owner.is_forwarding
    assert not case.owner._use_soft_stop_next and not case.owner._soft_stop_active


@pytest.mark.parametrize("action", [
    "forward", "backward", "steer_left", "steer_right", "rotate_left", "rotate_right",
])
def test_active_right_ir_rejects_every_new_motion_in_real_queue(ir_runtime, action):
    case = ir_runtime
    stop_for_right_ir(case)
    command = publish(case, action)
    assert not case.runtime.can_release_brake_hold(command.action)
    run_once(case, command)
    case.runtime._service_follow_wheels()
    assert command.action in case.checks
    assert case.owner._brake_hold_active
    assert case.owner.current_command is None
    assert case.driver.pairs == []
    assert case.driver.stops == [1]
    # The rejected request is consumed, not saved for automatic replay when
    # the obstacle disappears (even though its numeric axes remain fresh).
    case.hazards["right_ir"] = False
    case.clock[0] += .02
    run_once(case)
    case.runtime._service_follow_wheels()
    assert case.owner._brake_hold_active
    assert case.owner.current_command is None
    assert case.driver.pairs == []


@pytest.mark.parametrize("action,expected", [
    ("forward", (24, -24)), ("steer_left", (16, -32)),
    ("steer_right", (32, -16)), ("rotate_left", (-8, -8)),
    ("rotate_right", (8, 8)),
])
def test_clear_ir_and_new_authorized_motion_releases_and_writes(ir_runtime, action, expected):
    case = ir_runtime
    stop_for_right_ir(case)
    case.hazards["right_ir"] = False
    command = publish(case, action)
    run_once(case, command)
    assert not case.owner._brake_hold_active
    assert case.owner._brake_hold_stop_mode is None
    assert case.owner.current_command == command.action
    assert case.runtime._current_action_snapshot == command
    case.runtime._service_follow_wheels()
    assert case.driver.pairs == [expected]
    assert case.driver.stops == [1]


def test_clearing_ir_alone_never_replays_old_current_action_or_axes(ir_runtime):
    case = ir_runtime
    old = publish(case)
    case.owner.current_command = old.action
    case.runtime._current_action_snapshot = old
    stop_for_right_ir(case)
    case.hazards["right_ir"] = False
    # Even a still-fresh old canonical axis is not a new release decision.
    for delta in (.02, .05, 1.0):
        case.clock[0] += delta
        run_once(case)
        case.runtime._service_follow_wheels()
    assert case.owner._brake_hold_active
    # The old intent can itself expire into another STOP, never into motion.
    assert all(pair == (0, 0) for pair in case.driver.pairs)
    assert (case.driver.left, case.driver.right) == (0, 0)


@pytest.mark.parametrize("invalid", ["old_revision", "expired_intent"])
def test_clear_ir_does_not_adopt_obsolete_queued_motion(ir_runtime, invalid):
    case = ir_runtime
    stop_for_right_ir(case)
    case.hazards["right_ir"] = False
    command = publish(case)
    if invalid == "old_revision":
        command = replace(command, revision=command.revision - 1)
    else:
        command = replace(command, enqueued_at=case.clock[0] - 1.)
    run_once(case, command)
    assert case.owner._brake_hold_active
    assert case.owner.current_command is None
    assert case.driver.pairs == []


def test_new_queue_intent_without_live_axes_cannot_restore_old_speed(ir_runtime):
    case = ir_runtime
    stop_for_right_ir(case)
    case.hazards["right_ir"] = False
    command = publish(case)
    case.axes[2:] = [case.clock[0] - .01, case.clock[0] - .01]
    run_once(case, command)
    case.runtime._service_follow_wheels()
    assert all(pair == (0, 0) for pair in case.driver.pairs)
    assert (case.driver.left, case.driver.right) == (0, 0)


@pytest.mark.parametrize("remaining_hazard", ["front_ir", "left_ir", "distance", "visual_hazard"])
def test_clear_right_ir_cannot_release_another_active_hazard(ir_runtime, remaining_hazard):
    case = ir_runtime
    stop_for_right_ir(case)
    case.hazards.update(right_ir=False, **{remaining_hazard: True})
    command = publish(case)
    run_once(case, command)
    assert command.action in case.checks
    assert case.owner._brake_hold_active
    assert case.owner.current_command is None
    assert case.driver.pairs == []


def test_safety_check_failure_cannot_release_ir_hold(ir_runtime):
    case = ir_runtime
    stop_for_right_ir(case)
    case.hazards["right_ir"] = False

    def unavailable(_action=None):
        raise OSError("offline simulated sensor failure")

    case.runtime.hard_stop_check = unavailable
    run_once(case, publish(case))
    assert case.owner._brake_hold_active
    assert case.driver.pairs == []
