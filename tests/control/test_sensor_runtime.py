#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.sensor_modules import SensorRuntime, SensorRuntimeConfig


class _FakeIR:
    IDX_0 = 0
    IDX_1 = 1
    IDX_2 = 2
    calls = []

    @classmethod
    def init(cls):
        cls.calls.append("init")
        return 0

    @classmethod
    def deinit(cls):
        cls.calls.append("deinit")

    @classmethod
    def is_triggered(cls, idx):
        return idx in (cls.IDX_1, cls.IDX_2)


class _FakeUltrasonic:
    @classmethod
    def init(cls):
        return 0

    @classmethod
    def deinit(cls):
        pass

    @classmethod
    def get_distance(cls):
        return 123.0


class _FakeMmWave:
    @classmethod
    def init(cls):
        return 0

    @classmethod
    def deinit(cls):
        pass

    @classmethod
    def get_distance(cls):
        return 456.0

    @classmethod
    def get_targets(cls):
        return [{"index": 1, "angle": 0.0, "distance": 4.56}]


class _FakeIMU:
    @classmethod
    def init(cls):
        return 0

    @classmethod
    def deinit(cls):
        pass

    @classmethod
    def info(cls):
        return "fake-imu"

    @classmethod
    def poll(cls, _timeout):
        return {
            "accel": {"raw": (1, 2, 3), "g": (0.1, 0.2, 0.3)},
            "gyro": {"raw": (4, 5, 6), "dps": (1.0, 2.0, 3.0)},
        }


def _install_fake_hals() -> None:
    sys.modules["ir_hal"] = types.SimpleNamespace(IR=_FakeIR)
    sys.modules["utrasonic_hal"] = types.SimpleNamespace(Utrasonic=_FakeUltrasonic)
    sys.modules["mmwave_hal"] = types.SimpleNamespace(MmWaveRadar=_FakeMmWave)
    sys.modules["imu_hal"] = types.SimpleNamespace(IMU=_FakeIMU)


def main() -> int:
    _install_fake_hals()
    runtime = SensorRuntime(
        SensorRuntimeConfig(
            ir_enable=True,
            ultrasonic_enable=True,
            mmwave_enable=True,
            imu_enable=True,
            imu_fail_soft=False,
            imu_log_enable=True,
            imu_log_every_sec=0.01,
            side_ir_blocks_rotation=True,
        )
    )
    runtime.start()
    obstacles = runtime.get_obstacle_status()
    if not (obstacles.front and obstacles.left and not obstacles.right):
        raise AssertionError(f"unexpected obstacles: {obstacles}")
    runtime.config = SensorRuntimeConfig(
        ir_enable=True,
        ultrasonic_enable=True,
        mmwave_enable=True,
        imu_enable=True,
        imu_fail_soft=False,
        imu_log_enable=True,
        imu_log_every_sec=0.01,
        side_ir_blocks_rotation=False,
    )
    obstacles = runtime.get_obstacle_status()
    if not (obstacles.front and obstacles.left and not obstacles.right):
        raise AssertionError(f"side IR safety must not depend on rotation policy: {obstacles}")
    if runtime.get_ultrasonic_distance_cm() != 123.0:
        raise AssertionError("ultrasonic distance mismatch")
    if runtime.get_mmwave_distance_cm() != 456.0:
        raise AssertionError("mmwave distance mismatch")
    if not runtime.get_mmwave_targets():
        raise AssertionError("mmwave targets missing")
    runtime.maybe_log_imu_sample(1)
    runtime.close()
    print("sensor_runtime ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
