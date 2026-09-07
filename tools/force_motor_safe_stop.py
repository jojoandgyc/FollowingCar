#!/usr/bin/env python3
"""Best-effort motor cleanup used when the main runtime cannot shut down."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from car_control_modular.config_loader import load_config_to_env


def _resolve_library_path(value: str) -> str:
    path = Path(value or "/home/topeet/lianzhan")
    if not path.is_absolute():
        path = ROOT / path
    if (path / "src" / "lz30ema_rs485" / "__init__.py").is_file():
        path = path / "src"
    if not (path / "lz30ema_rs485" / "__init__.py").is_file():
        raise RuntimeError(f"找不到 lz30ema_rs485: {path}")
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Force LZ30EMA speed/parking current to a safe state.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    load_config_to_env(args.config)
    lib_path = _resolve_library_path(os.environ.get("MOTOR_RS485_LIB_DIR", "/home/topeet/lianzhan"))
    sys.path.insert(0, lib_path)
    from lz30ema_rs485 import LZ30EMAClient, StopMode

    port = os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0")
    slave = int(os.environ.get("MOTOR_RS485_SLAVE_ID", "1"))
    baudrate = int(os.environ.get("MOTOR_RS485_BAUDRATE", "115200"))
    timeout = float(os.environ.get("MOTOR_RS485_TIMEOUT", "0.15"))
    driver = LZ30EMAClient.from_serial(port, slave=slave, baudrate=baudrate, timeout=timeout)
    try:
        driver.set_right_speed(0)
        driver.set_left_speed(0)
        driver.stop_all(StopMode.EMERGENCY)
        driver.write_register("right_parking_current", 0.0, persist=True)
        driver.write_register("left_parking_current", 0.0, persist=True)
        right = float(driver.read_register("right_parking_current"))
        left = float(driver.read_register("left_parking_current"))
        print(f"force motor cleanup complete: right_parking_current={right:g}A left_parking_current={left:g}A")
        if abs(right) > 0.005 or abs(left) > 0.005:
            raise RuntimeError(f"驻车电流清零校验失败: right={right:g} left={left:g}")
    finally:
        try:
            driver.close()
        except Exception as exc:
            print(f"驱动器关闭告警: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
