#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.config_loader import load_config_to_env


def _percent_to_target(percent: int, max_target: int) -> int:
    if percent < 0 or percent > 20:
        raise ValueError("live motor tests must keep --percent in 0..20")
    return round(int(max_target) * int(percent) / 100.0)


def _targets(direction: str, target: int, left_sign: int, right_sign: int) -> tuple[int, int]:
    forward_sign = -1 if int(os.environ.get("MOTOR_FORWARD_TARGET_SIGN", "-1")) < 0 else 1
    if direction == "forward":
        raw_left, raw_right = forward_sign * target, forward_sign * target
    elif direction == "left":
        raw_left, raw_right = -forward_sign * target, forward_sign * target
    elif direction == "right":
        raw_left, raw_right = forward_sign * target, -forward_sign * target
    elif direction == "stop":
        raw_left, raw_right = 0, 0
    else:
        raise ValueError("direction must be forward, left, right, or stop")
    return raw_left * int(left_sign), raw_right * int(right_sign)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live LZ-30EMA motor smoke test.")
    parser.add_argument("--config", default=str(ROOT / "car_control_modular/config/reid_runtime.ini"))
    parser.add_argument("--direction", choices=["forward", "left", "right", "stop"], default="stop")
    parser.add_argument("--percent", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.2)
    parser.add_argument("--execute", action="store_true", help="Actually send the motor command.")
    args = parser.parse_args()

    load_config_to_env(args.config)
    lib_dir = os.environ.get("MOTOR_RS485_LIB_DIR", "/home/topeet/lianzhan")
    port = os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0")
    slave_id = int(os.environ.get("MOTOR_RS485_SLAVE_ID", "1"))
    baudrate = int(os.environ.get("MOTOR_RS485_BAUDRATE", "9600"))
    timeout = float(os.environ.get("MOTOR_RS485_TIMEOUT", "0.3"))
    max_target = int(os.environ.get("MOTOR_RS485_MAX_TARGET", "100"))
    left_sign = int(os.environ.get("MOTOR_LEFT_SIGN", "-1"))
    right_sign = int(os.environ.get("MOTOR_RIGHT_SIGN", "1"))
    stop_mode = os.environ.get("MOTOR_RS485_STOP_MODE", "normal").strip().lower()

    target = _percent_to_target(args.percent, max_target)
    left_target, right_target = _targets(args.direction, target, left_sign, right_sign)
    print(
        f"plan execute={args.execute} direction={args.direction} percent={args.percent} "
        f"left={left_target} right={right_target} seconds={args.seconds}"
    )
    if not args.execute:
        print("dry-run only; add --execute to send this command")
        return 0

    lib_path = Path(lib_dir)
    if not lib_path.is_absolute():
        lib_path = ROOT / lib_path
    if (lib_path / "src" / "lz30ema_rs485" / "__init__.py").is_file():
        lib_path = lib_path / "src"
    if not (lib_path / "lz30ema_rs485" / "__init__.py").is_file():
        raise RuntimeError(f"MOTOR_RS485_LIB_DIR does not contain lz30ema_rs485: {lib_path}")
    if str(lib_path) not in sys.path:
        sys.path.insert(0, str(lib_path))
    from lz30ema_rs485 import LZ30EMAClient, StopMode

    stop_value = {
        "normal": StopMode.NORMAL,
        "emergency": StopMode.EMERGENCY,
        "free": StopMode.FREE,
    }.get(stop_mode, StopMode.NORMAL)
    driver = LZ30EMAClient.from_serial(
        port,
        slave=slave_id,
        baudrate=baudrate,
        timeout=timeout,
    )
    try:
        if args.direction == "stop" or args.percent == 0:
            driver.set_right_speed(0)
            driver.set_left_speed(0)
            driver.stop_all(stop_value)
            return 0
        driver.set_right_speed(int(right_target))
        driver.set_left_speed(int(left_target))
        time.sleep(max(0.0, float(args.seconds)))
    finally:
        try:
            driver.set_right_speed(0)
            driver.set_left_speed(0)
            driver.stop_all(stop_value)
        finally:
            driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
