#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.control_types import DistanceState, ObstacleState


class FakeDriver:
    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.stops = 0
        self.emergency_stops = 0
        self.free_stops = 0

    def set_left_speed(self, value: int) -> None:
        self.left = int(value)

    def set_right_speed(self, value: int) -> None:
        self.right = int(value)

    def stop_all(self, mode: int = 0) -> None:
        self.left = 0
        self.right = 0
        mode_value = int(mode)
        if mode_value == 1:
            self.emergency_stops += 1
        elif mode_value == 2:
            self.free_stops += 1
        else:
            self.stops += 1


def _load_request_module(config: str):
    old_argv = list(sys.argv)
    sys.argv = ["test_mssd_mapping.py", "--config", config]
    try:
        import request_0513_modular as mod
    finally:
        sys.argv = old_argv
    return mod


def _make_tracker_shell(mod):
    tracker = object.__new__(mod.PersonTracker)
    backend = mod.MssdMotorBackend(
        replace(mod.MSSD_MOTOR_CONFIG, stop_zero_delay_sec=0.0),
        logger=mod.logger,
    )
    backend.driver = FakeDriver()
    backend.motion_armed = True
    tracker._motor_backend = backend
    tracker.motor_io_lock = backend.io_lock
    tracker.current_command = None
    tracker.command_start_time = None
    tracker._last_rotate_pulse_refresh_ts = 0.0
    runtime_config = replace(
        mod.ACTION_RUNTIME_CONFIG,
        motor_rs485_transition_stop_delay_sec=0.0,
        motor_rs485_transition_stop_repeat=2,
    )
    tracker._action_runtime = mod.MotionActionRuntime(
        tracker,
        backend,
        runtime_config,
        mod.ACTION_RUNTIME_SYMBOLS,
        hard_stop_check=lambda _action=None: False,
        logger=mod.logger,
    )
    return tracker


class FakeBunkerRuntime:
    def __init__(self, state=None) -> None:
        self.state = state

    def current_split_state(self):
        return self.state


class FakeSensorRuntime:
    def __init__(self, *, front: bool = False, left: bool = False, right: bool = False) -> None:
        self.obstacles = ObstacleState(front=front, left=left, right=right)

    def get_obstacle_status(self) -> ObstacleState:
        return self.obstacles


class FakeDistanceRuntime:
    def __init__(self, distance_m=None) -> None:
        self.state = DistanceState(source="fake", used_distance_m=distance_m)

    def get_recent_vision_mmwave_state(self, **_kwargs) -> DistanceState:
        return self.state

    def get_sensor_distance_state(self, **_kwargs) -> DistanceState:
        return self.state


def _set_safety_stubs(tracker, *, front: bool = False, left: bool = False, right: bool = False, distance_m=None) -> None:
    tracker._bunker_runtime = FakeBunkerRuntime()
    tracker._sensor_runtime = FakeSensorRuntime(front=front, left=left, right=right)
    tracker._distance_runtime = FakeDistanceRuntime(distance_m)
    tracker.frame_index = 0
    tracker._last_distance_stop_log_key = None
    tracker._last_distance_hard_stop_log_ts = 0.0


def _assert_hard_stop_policy(mod, tracker) -> None:
    actions = (
        mod.ACTION_FORWARD,
        mod.ACTION_STEER_LEFT,
        mod.ACTION_STEER_RIGHT,
        mod.ACTION_ROTATE_LEFT,
        mod.ACTION_ROTATE_RIGHT,
    )
    for sensor_name, sensor_state in (
        ("front", {"front": True}),
        ("left", {"left": True}),
        ("right", {"right": True}),
    ):
        _set_safety_stubs(tracker, **sensor_state)
        for action in actions:
            if not tracker._should_hard_stop_now(action):
                raise AssertionError(f"{sensor_name} IR should hard-stop action={action}")

    _set_safety_stubs(tracker, distance_m=max(0.0, float(mod.FOLLOW_BRAKE_DISTANCE_M) - 0.01))
    if not tracker._should_hard_stop_now(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("too-close distance should hard-stop rotate_left")

    _set_safety_stubs(tracker)
    if tracker._should_hard_stop_now(mod.ACTION_FORWARD):
        raise AssertionError("clear sensors should not hard-stop forward")


def _assert_pair(label: str, actual, expected) -> None:
    print(label, actual)
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def _expected_send_diff(backend, m1_percent: int, m1_state: int, m2_percent: int, m2_state: int) -> tuple[int, int]:
    if backend.config.m1_is_left_wheel:
        left_target = backend.wheel_state_to_target("left", m1_percent, m1_state)
        right_target = backend.wheel_state_to_target("right", m2_percent, m2_state)
    else:
        right_target = backend.wheel_state_to_target("right", m1_percent, m1_state)
        left_target = backend.wheel_state_to_target("left", m2_percent, m2_state)
    return left_target, right_target


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run LZ30EMA wheel RPM mapping.")
    parser.add_argument("--config", default=str(ROOT / "car_control_modular/config/reid_runtime.ini"))
    args = parser.parse_args()

    mod = _load_request_module(args.config)
    tracker = _make_tracker_shell(mod)
    backend = tracker._motor_backend
    _assert_hard_stop_policy(mod, tracker)

    backend.send_diff(20, 0x01, 20, 0x01, "forward")
    _assert_pair("forward", (backend.driver.left, backend.driver.right), _expected_send_diff(backend, 20, 0x01, 20, 0x01))

    backend.send_diff(20, 0x01, 20, 0x02, "left")
    _assert_pair("left", (backend.driver.left, backend.driver.right), _expected_send_diff(backend, 20, 0x01, 20, 0x02))

    backend.send_diff(20, 0x02, 20, 0x01, "right")
    _assert_pair("right", (backend.driver.left, backend.driver.right), _expected_send_diff(backend, 20, 0x02, 20, 0x01))

    runtime = tracker._action_runtime
    if not runtime.needs_transition_stop(mod.ACTION_FORWARD, mod.ACTION_ROTATE_LEFT):
        raise AssertionError("forward -> rotate_left should require transition stop")
    if runtime.needs_transition_stop(mod.ACTION_FORWARD, mod.ACTION_STEER_LEFT):
        raise AssertionError("forward -> steer_left should update wheel speed without transition stop")
    if runtime.needs_transition_stop(mod.ACTION_STEER_LEFT, mod.ACTION_FORWARD):
        raise AssertionError("steer_left -> forward should update wheel speed without transition stop")
    if runtime.needs_transition_stop(mod.ACTION_STEER_LEFT, mod.ACTION_STEER_RIGHT):
        raise AssertionError("steer_left -> steer_right should update wheel speed without transition stop")
    if not runtime.needs_transition_stop(mod.ACTION_STEER_LEFT, mod.ACTION_ROTATE_RIGHT):
        raise AssertionError("steer_left -> rotate_right should require transition stop")
    if not runtime.needs_transition_stop(mod.ACTION_ROTATE_LEFT, mod.ACTION_FORWARD):
        raise AssertionError("rotate_left -> forward should require transition stop")
    if not runtime.needs_transition_stop(mod.ACTION_ROTATE_LEFT, mod.ACTION_ROTATE_RIGHT):
        raise AssertionError("rotate_left -> rotate_right should require transition stop")
    if runtime.needs_transition_stop(mod.ACTION_ROTATE_LEFT, mod.ACTION_ROTATE_LEFT):
        raise AssertionError("same rotate action should not require transition stop")
    tracker._current_steer_base_percent = max(35, int(mod.MIN_FORWARD_PERCENT))
    tracker._current_steer_inner_ratio_percent = 100
    tracker._current_steer_outer_ratio_percent = 130
    tracker._current_rotate_turn_percent = int(mod.ROTATE_TURN_PERCENT_FROM_FORWARD)
    runtime.send_robot_command(mod.ACTION_STEER_LEFT)
    if int(mod.MOTOR_STEER_RAW_TARGET) > 0:
        steer_base = int(mod.MOTOR_STEER_RAW_TARGET)
        steer_inner = max(steer_base, int(math.floor(steer_base * 100 / 100.0)))
        steer_outer = max(steer_base, int(math.ceil(steer_base * 130 / 100.0)))
        expected_left = backend.wheel_raw_state_to_target("left", steer_inner, 0x01)
        expected_right = backend.wheel_raw_state_to_target("right", steer_outer, 0x01)
        _assert_pair("steer_left_raw", (backend.driver.left, backend.driver.right), (expected_left, expected_right))
    else:
        steer_cap = max(mod.MIN_FORWARD_PERCENT, min(100, mod.STEER_PERCENT_LIMIT))
        base_cap = min(mod.MAX_FORWARD_PERCENT, steer_cap)
        steer_base = max(0, min(base_cap, tracker._current_steer_base_percent))
        steer_inner = int(math.floor(steer_base * 100 / 100.0))
        steer_outer = int(math.ceil(steer_base * 130 / 100.0))
        steer_inner = max(mod.MIN_FORWARD_PERCENT, min(steer_cap, steer_inner))
        steer_outer = max(mod.MIN_FORWARD_PERCENT, min(steer_cap, steer_outer))
        expected_left = backend.wheel_state_to_target("left", steer_inner, 0x01)
        expected_right = backend.wheel_state_to_target("right", steer_outer, 0x01)
        _assert_pair("steer_left_percent", (backend.driver.left, backend.driver.right), (expected_left, expected_right))

    tracker.current_command = mod.ACTION_ROTATE_LEFT
    runtime.send_robot_command(mod.ACTION_ROTATE_LEFT)
    if int(mod.MOTOR_ROTATE_RAW_TARGET) > 0:
        expected_left = backend.wheel_raw_state_to_target("left", int(mod.MOTOR_ROTATE_RAW_TARGET), 0x02)
        expected_right = backend.wheel_raw_state_to_target("right", int(mod.MOTOR_ROTATE_RAW_TARGET), 0x01)
    else:
        expected_left = backend.wheel_state_to_target("left", tracker._current_rotate_turn_percent, 0x02)
        expected_right = backend.wheel_state_to_target("right", tracker._current_rotate_turn_percent, 0x01)
    _assert_pair("rotate_left", (backend.driver.left, backend.driver.right), (expected_left, expected_right))
    runtime.send_motion_transition_stop(mod.ACTION_FORWARD, mod.ACTION_ROTATE_LEFT)
    _assert_pair("transition_stop", (backend.driver.left, backend.driver.right), (0, 0))
    if backend.driver.emergency_stops != 2:
        raise AssertionError(f"expected two emergency stops, got {backend.driver.emergency_stops}")
    backend.motion_armed = True

    tracker._current_forward_percent = 30
    tracker._current_steer_base_percent = 12
    tracker.is_forwarding = True
    runtime.send_stop_with_brake_hold("search_to_follow")
    if tracker._current_forward_percent != 30 or tracker._current_steer_base_percent != 12:
        raise AssertionError(
            "search_to_follow transition stop should preserve the just-decided motion parameters"
        )
    backend.motion_armed = True

    backend.send_diff(99, 0x01, 99, 0x01, "limit")
    _assert_pair("limit", (backend.driver.left, backend.driver.right), _expected_send_diff(backend, 99, 0x01, 99, 0x01))

    backend.send_stop("test_stop")
    _assert_pair("stop", (backend.driver.left, backend.driver.right), (0, 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
