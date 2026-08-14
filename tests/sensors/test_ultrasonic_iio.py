#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.config_loader import load_config_to_env


def _reload_ultrasonic_hal():
    if "utrasonic_hal" in sys.modules:
        return importlib.reload(sys.modules["utrasonic_hal"])
    return importlib.import_module("utrasonic_hal")


def _write_sr04_device(base: Path, raw: int, scale: float) -> Path:
    device = base / "sr04-device"
    device.mkdir(parents=True, exist_ok=True)
    (device / "name").write_text("hcsr04", encoding="ascii")
    (device / "in_distance_raw").write_text(str(int(raw)), encoding="ascii")
    (device / "in_distance_scale").write_text(f"{float(scale):.6f}", encoding="ascii")
    return device


def _run_fake() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        device = _write_sr04_device(Path(tmp), raw=1400, scale=0.001)
        os.environ.update(
            {
                "ULTRASONIC_BACKEND": "iio",
                "ULTRASONIC_IIO_DEVICE": str(device),
                "ULTRASONIC_IIO_DEVICE_NAME": "hcsr04",
                "ULTRASONIC_MIN_DISTANCE_M": "0.02",
                "ULTRASONIC_MAX_DISTANCE_M": "8.0",
            }
        )
        hal = _reload_ultrasonic_hal()
        ret = hal.Utrasonic.init()
        distance_cm = hal.Utrasonic.get_distance()
        print(f"init={ret} distance_cm={distance_cm}")
        if ret != 0:
            raise AssertionError("fake ultrasonic init failed")
        if distance_cm is None or abs(float(distance_cm) - 140.0) > 0.001:
            raise AssertionError(f"unexpected fake ultrasonic distance: {distance_cm}")
    return 0


def _run_live(config: str) -> int:
    load_config_to_env(config)
    hal = _reload_ultrasonic_hal()
    ret = hal.Utrasonic.init()
    distance_cm = hal.Utrasonic.get_distance()
    distance_m = None if distance_cm is None else float(distance_cm) / 100.0
    print(f"init={ret} distance_cm={distance_cm} distance_m={distance_m}")
    if ret != 0:
        raise AssertionError("live ultrasonic init failed")
    if distance_cm is None:
        raise AssertionError("live ultrasonic returned no distance")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test RK3588 SR04 ultrasonic IIO distance.")
    parser.add_argument("--config", default=str(ROOT / "car_control_modular/config/reid_runtime.ini"))
    parser.add_argument("--fake", action="store_true", help="Run against a temporary fake IIO tree.")
    parser.add_argument("--live", action="store_true", help="Read the board's configured SR04 IIO device.")
    args = parser.parse_args()

    if args.live:
        return _run_live(args.config)
    return _run_fake()


if __name__ == "__main__":
    raise SystemExit(main())
