#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.config_loader import load_config_to_env


def main() -> int:
    config_path = ROOT / "car_control_modular/config/reid_runtime_rotation_only.ini"
    if not config_path.exists():
        # The compact deployment work copy stores the downloaded config one
        # directory higher; the board uses the normal config/ location.
        config_path = ROOT / "car_control_modular/reid_runtime_rotation_only.ini"
    loaded = load_config_to_env(str(config_path))
    if loaded is None:
        raise AssertionError(f"failed to load config: {config_path}")

    expected = {
        "FOLLOW_ROTATION_ONLY": "1",
        "ASTRA_DEPTH_LONGITUDINAL_CONTROL_ENABLE": "0",
        "MODULE_ASTRA_DEPTH_ENABLE": "0",
        "MODULE_ULTRASONIC_ENABLE": "0",
        "MODULE_MMWAVE_ENABLE": "0",
        "MODULE_IMU_ENABLE": "0",
        "DISTANCE_SOURCE": "none",
        "RKNN_CAMERA_RAW_OUTPUT": "auto",
        "FOLLOW_CENTER_LEFT_RATIO": "0.40",
        "FOLLOW_CENTER_RIGHT_RATIO": "0.60",
        "VISIBLE_STEERING_PID_ENABLE": "1",
        "VISIBLE_STEERING_PID_FALLBACK_MAX_CORRECTION_RPM": "40.0",
        "VISIBLE_STEERING_PID_MIN_EFFECTIVE_ERROR_DEG": "0.0",
        "VISIBLE_STEERING_PID_MIN_EFFECTIVE_CORRECTION_RPM": "0.0",
        "VISIBLE_STEERING_PID_MECHANICAL_TIER2_ERROR_DEG": "0.0",
        "VISIBLE_STEERING_PID_MECHANICAL_TIER2_CORRECTION_RPM": "0.0",
        "VISIBLE_STEERING_PID_MECHANICAL_TIER3_ERROR_DEG": "0.0",
        "VISIBLE_STEERING_PID_MECHANICAL_TIER3_CORRECTION_RPM": "0.0",
        "VISIBLE_STEERING_PID_MECHANICAL_FLOOR_RELEASE_RATIO": "0.0",
        "VISIBLE_STEERING_PID_STARTUP_KICK_ERROR_DEG": "6.0",
        "VISIBLE_STEERING_PID_STARTUP_KICK_RPM": "30.0",
        "VISIBLE_STEERING_PID_STARTUP_KICK_MAX_SEC": "0.60",
        "VISIBLE_STEERING_PID_STARTUP_KICK_RELEASE_YAW_RATE_DPS": "4.0",
        "VISIBLE_STEERING_PID_ACTIVE_BRAKE_YAW_THRESHOLD_DPS": "3.0",
        "VISIBLE_STEERING_PID_ACTIVE_BRAKE_MIN_CORRECTION_RPM": "20.0",
        "PARKED_RECENTER_MIN_RPM": "2",
        "PARKED_RECENTER_MAX_RPM": "40",
        "ROTATION_ONLY_YAW_PULSE_RPM": "30",
        "ROTATION_ONLY_YAW_PULSE_MIN_SEC": "0.18",
        "ROTATION_ONLY_YAW_PULSE_MAX_SEC": "0.32",
        "ROTATION_ONLY_YAW_BRAKE_SEC": "0.18",
        "ROTATION_ONLY_YAW_RESPONSE_DPS": "4.0",
        "ROTATION_ONLY_YAW_ZERO_GAP_SEC": "0.03",
        "ROTATE_PULSE_ACTIVE_BRAKE_ENABLE": "1",
        "ROTATE_PULSE_ACTIVE_BRAKE_RPM": "30",
        "ROTATE_PULSE_ACTIVE_BRAKE_SEC": "0.18",
        "ROTATE_RAW_TARGET_SEARCH": "30",
        "MOTOR_RS485_MAX_TARGET": "40",
    }
    actual = {key: os.environ.get(key) for key in expected}
    if actual != expected:
        raise AssertionError(f"rotation-only config mismatch: expected={expected} actual={actual}")

    if os.environ.get("ROTATION_ONLY_TEST_IMPORT_RUNTIME", "0") == "1":
        import request_0513_modular as runtime

        runtime_actual = {
            "FOLLOW_ROTATION_ONLY": runtime.FOLLOW_ROTATION_ONLY,
            "ASTRA_DEPTH_LONGITUDINAL_CONTROL_ENABLE": (
                runtime.ASTRA_DEPTH_LONGITUDINAL_CONTROL_ENABLE
            ),
            "MODULE_ASTRA_DEPTH_ENABLE": runtime.MODULE_ASTRA_DEPTH_ENABLE,
            "MODULE_ULTRASONIC_ENABLE": runtime.MODULE_ULTRASONIC_ENABLE,
            "MODULE_IMU_ENABLE": runtime.MODULE_IMU_ENABLE,
            "DISTANCE_SOURCE": runtime.DISTANCE_SOURCE,
            "ACTION_ROTATION_ONLY": runtime.ACTION_RUNTIME_CONFIG.rotation_only,
            "ACTION_YAW_PULSE_RPM": runtime.ACTION_RUNTIME_CONFIG.rotation_only_yaw_pulse_rpm,
            "ACTION_YAW_PULSE_RANGE": (
                runtime.ACTION_RUNTIME_CONFIG.rotation_only_yaw_pulse_min_sec,
                runtime.ACTION_RUNTIME_CONFIG.rotation_only_yaw_pulse_max_sec,
            ),
            "ACTION_ACTIVE_BRAKE": (
                runtime.ACTION_RUNTIME_CONFIG.rotate_pulse_active_brake_enable,
                runtime.ACTION_RUNTIME_CONFIG.rotate_pulse_active_brake_rpm,
                runtime.ACTION_RUNTIME_CONFIG.rotate_pulse_active_brake_sec,
            ),
        }
        runtime_expected = {
            "FOLLOW_ROTATION_ONLY": True,
            "ASTRA_DEPTH_LONGITUDINAL_CONTROL_ENABLE": False,
            "MODULE_ASTRA_DEPTH_ENABLE": False,
            "MODULE_ULTRASONIC_ENABLE": False,
            "MODULE_IMU_ENABLE": False,
            "DISTANCE_SOURCE": "none",
            "ACTION_ROTATION_ONLY": True,
            "ACTION_YAW_PULSE_RPM": 30,
            "ACTION_YAW_PULSE_RANGE": (0.18, 0.32),
            "ACTION_ACTIVE_BRAKE": (True, 30, 0.18),
        }
        if runtime_actual != runtime_expected:
            raise AssertionError(
                "rotation-only runtime mismatch: "
                f"expected={runtime_expected} actual={runtime_actual}"
            )
    print("rotation_only_config_ok", actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
