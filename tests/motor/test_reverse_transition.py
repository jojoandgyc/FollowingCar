#!/usr/bin/env python3
from __future__ import annotations

import argparse

from test_mssd_mapping import _load_request_module, _make_tracker_shell


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify queued reverse RPM preservation.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    mod = _load_request_module(args.config)
    if abs(float(mod.FOLLOW_REVERSE_IMMEDIATE_DISTANCE_M) - 1.30) > 1e-9:
        raise AssertionError(
            "runtime did not load reverse_immediate_distance_m=1.30"
        )
    tracker = _make_tracker_shell(mod)
    runtime = tracker._action_runtime
    backend = tracker._motor_backend

    tracker._current_forward_percent = 32
    tracker._current_steer_base_percent = 0
    tracker.is_forwarding = False
    runtime.send_stop_with_brake_hold(
        "queued_action_stop_signal",
        preserve_motion_params=True,
    )
    if tracker._current_forward_percent != 32:
        raise AssertionError("queued transition stop erased the requested reverse RPM")

    backend.motion_armed = True
    tracker._last_control_decision_reason = "target_approaching_reverse"
    tracker._current_steer_correction_rpm = 0
    straight_signature = tracker._action_signature(mod.ACTION_BACKWARD)
    runtime.send_robot_command(mod.ACTION_BACKWARD)
    left = int(backend.driver.left)
    right = int(backend.driver.right)
    if left >= 0 or right <= 0:
        raise AssertionError(f"first reverse write was not nonzero reverse: {left}/{right}")

    base_raw = round(int(mod.MOTOR_FORWARD_MAX_TARGET_RPM) * 32 / 100.0)
    tracker._current_steer_correction_rpm = 8
    right_signature = tracker._action_signature(mod.ACTION_BACKWARD)
    if right_signature == straight_signature:
        raise AssertionError("reverse PID correction must change the action queue signature")
    runtime.send_robot_command(mod.ACTION_BACKWARD)
    right_turn = (int(backend.driver.left), int(backend.driver.right))
    expected_right_turn = (
        backend.wheel_raw_state_to_target("left", base_raw - 8, 0x02),
        backend.wheel_raw_state_to_target("right", base_raw + 8, 0x02),
    )
    if right_turn != expected_right_turn:
        raise AssertionError(
            f"reverse right correction mapped to wrong wheels: {right_turn} != {expected_right_turn}"
        )

    tracker._current_steer_correction_rpm = -8
    runtime.send_robot_command(mod.ACTION_BACKWARD)
    left_turn = (int(backend.driver.left), int(backend.driver.right))
    expected_left_turn = (
        backend.wheel_raw_state_to_target("left", base_raw + 8, 0x02),
        backend.wheel_raw_state_to_target("right", base_raw - 8, 0x02),
    )
    if left_turn != expected_left_turn:
        raise AssertionError(
            f"reverse left correction mapped to wrong wheels: {left_turn} != {expected_left_turn}"
        )

    print(f"queued_reverse_first_write_nonzero: PASS ({left}/{right} rpm)")
    print(f"reverse_visual_pid_mapping: PASS (right={right_turn}, left={left_turn})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
