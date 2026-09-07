#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.sensor_modules import SensorRuntime, SensorRuntimeConfig


def _runtime() -> SensorRuntime:
    return SensorRuntime(
        SensorRuntimeConfig(
            ir_enable=False,
            ultrasonic_enable=False,
            mmwave_enable=False,
            imu_enable=False,
            imu_fail_soft=True,
            imu_log_enable=False,
            imu_log_every_sec=1.0,
            side_ir_blocks_rotation=True,
            side_ir_confirm_sec=0.10,
            side_ir_release_sec=0.20,
        )
    )


def main() -> int:
    runtime = _runtime()
    if runtime._filter_side_ir("left", True, 10.00):
        raise AssertionError("single side-IR sample must not trigger")
    if runtime._filter_side_ir("left", True, 10.09):
        raise AssertionError("side IR must wait for the confirmation interval")
    if not runtime._filter_side_ir("left", True, 10.11):
        raise AssertionError("continuous side IR must latch after 0.10s")
    if not runtime._filter_side_ir("left", False, 10.12):
        raise AssertionError("a latched side IR must not release on one clear sample")
    if not runtime._filter_side_ir("left", False, 10.31):
        raise AssertionError("side IR must retain the 0.20s clear hysteresis")
    if runtime._filter_side_ir("left", False, 10.33):
        raise AssertionError("side IR should release after 0.20s continuously clear")
    if runtime._filter_side_ir("right", True, 20.00):
        raise AssertionError("left and right filters must have independent state")
    print("side_ir_filter: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
