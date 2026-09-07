#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.action_runtime import MotionActionRuntime
from car_control_modular.control_types import SteeringFeedback


class FakeBackend:
    def __init__(self) -> None:
        self.config = SimpleNamespace(max_target=100, m1_is_left_wheel=True)
        self.targets = []
        self.motion_armed = False

    @staticmethod
    def wheel_raw_state_to_target(_side: str, raw: int, state: int) -> int:
        return int(raw) if int(state) == 0x01 else -int(raw) if int(state) == 0x02 else 0

    def send_targets(
        self,
        left: int,
        right: int,
        label: str,
        *,
        max_target_override=None,
    ) -> None:
        self.targets.append((int(left), int(right), str(label), max_target_override))
        self.motion_armed = bool(left or right)


def main() -> int:
    owner = SimpleNamespace(
        action_queue_lock=threading.Lock(),
        motor_io_lock=threading.Lock(),
        _last_motor_dispatch_source="action_queue",
        frame_index=1,
    )
    backend = FakeBackend()
    config = SimpleNamespace(
        steering_feedback_median_window=3,
        motor_forward_max_target_rpm=100,
        motor_forward_raw_target=0,
        rotation_only=True,
        rotation_only_yaw_pulse_rpm=30,
        rotation_only_yaw_pulse_min_sec=0.18,
        rotation_only_yaw_pulse_max_sec=0.32,
        rotation_only_yaw_brake_sec=0.18,
        rotation_only_yaw_response_dps=4.0,
        rotation_only_yaw_zero_gap_sec=0.03,
        rotate_pulse_settle_feedback_stale_sec=0.30,
        rotate_pulse_settle_max_yaw_rate_dps=3.0,
        rotate_pulse_active_brake_enable=True,
        rotate_pulse_active_brake_rpm=30,
        rotate_pulse_active_brake_sec=0.18,
    )
    symbols = SimpleNamespace(
        forward=1,
        backward=2,
        rotate_left=3,
        rotate_right=4,
        steer_left=5,
        steer_right=6,
        action_names={3: "rotate_left", 4: "rotate_right"},
    )
    runtime = MotionActionRuntime(
        owner,
        backend,
        config,
        symbols,
        hard_stop_check=lambda _action=None: False,
        logger=logging.getLogger("rotation-pulse-test"),
    )

    owner._follow_controller = SimpleNamespace(target_stop_latched=True)
    owner._last_control_decision_reason = "near_distance_rotation_only"
    owner._current_forward_percent = 0
    owner._current_steer_base_percent = 0
    owner._current_steer_correction_rpm = 0
    owner._brake_hold_stop_mode = None
    owner._brake_hold_label = "brake"
    if not runtime.can_release_brake_hold(symbols.rotate_left):
        raise AssertionError("visible near-distance yaw must release an ordinary brake hold")
    owner._last_control_decision_reason = "target_distance_hold"
    if runtime.can_release_brake_hold(symbols.rotate_left):
        raise AssertionError("an unrelated parked rotation must not release brake hold")

    if not runtime.request_rotation_only_yaw_pulse(8):
        raise AssertionError("a fresh visual yaw request must start one pulse")
    if backend.targets[-1][:3] != (30, -30, "YAW_ONLY"):
        raise AssertionError(f"visible pulse must use fixed 30 RPM: {backend.targets[-1]}")
    first_call_count = len(backend.targets)
    if runtime.request_rotation_only_yaw_pulse(12):
        raise AssertionError("same-direction refresh must not restart an active pulse")
    if len(backend.targets) != first_call_count:
        raise AssertionError("same-direction refresh must not write another motor target")

    with runtime._steering_feedback_lock:
        runtime._steering_feedback = SteeringFeedback(
            timestamp=time.monotonic(),
            yaw_rate_right_dps=0.0,
            raw_yaw_rate_right_dps=5.0,
            trustworthy=True,
        )
    runtime._service_yaw_pulses()
    if backend.targets[-1][:3] != (0, 0, "TURN_ZERO"):
        raise AssertionError("first raw encoder response must end the startup pulse")
    if runtime._visible_yaw_pulse_direction != 0:
        raise AssertionError("encoder-released pulse state must be idle")

    runtime._visible_yaw_pulse_last_end_monotonic -= 1.0
    if not runtime.request_rotation_only_yaw_pulse(8):
        raise AssertionError("a later visual frame must be able to start another pulse")
    if not runtime.request_rotation_only_yaw_pulse(-20):
        raise AssertionError("direction reversal must immediately become a brake pulse")
    if runtime._visible_yaw_pulse_kind != "brake":
        raise AssertionError("opposite encoder motion must be represented as a short brake pulse")
    if backend.targets[-1][:3] != (-30, 30, "YAW_ONLY"):
        raise AssertionError(f"brake pulse must reverse at fixed 30 RPM: {backend.targets[-1]}")
    runtime._visible_yaw_pulse_deadline_monotonic = time.monotonic() - 0.01
    runtime._service_yaw_pulses()
    if runtime._visible_yaw_pulse_direction != 0 or backend.targets[-1][:3] != (0, 0, "TURN_ZERO"):
        raise AssertionError("brake pulse must have a hard deadline and end at zero target")

    if not runtime._start_search_active_brake(symbols.rotate_left):
        raise AssertionError("a completed search pulse must start active reverse braking")
    if backend.targets[-1][:3] != (30, -30, "YAW_ONLY"):
        raise AssertionError(f"search brake direction or RPM is wrong: {backend.targets[-1]}")
    runtime._search_brake_started_monotonic -= 0.07
    with runtime._steering_feedback_lock:
        runtime._steering_feedback = SteeringFeedback(
            timestamp=time.monotonic(),
            yaw_rate_right_dps=0.0,
            raw_yaw_rate_right_dps=0.0,
            trustworthy=True,
        )
    runtime._service_yaw_pulses()
    if runtime._search_brake_direction != 0 or backend.targets[-1][:3] != (0, 0, "TURN_ZERO"):
        raise AssertionError("search brake must stop as soon as fresh encoder feedback settles")

    print("rotation_only_pulse_runtime_ok", len(backend.targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
