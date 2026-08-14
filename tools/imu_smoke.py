#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from car_control_modular.config_loader import load_config_to_env


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the ICM20600 IMU HAL.")
    parser.add_argument("--config", default="", help="Optional INI config to preload.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--poll-sec", type=float, default=0.2)
    args = parser.parse_args()

    if args.config:
        load_config_to_env(args.config)

    from imu_hal import IMU

    ret = IMU.init()
    print(json.dumps({"init": ret, "info": IMU.info()}, ensure_ascii=False))
    if ret != 0:
        return 1

    produced = 0
    deadline = time.time() + max(0.1, args.timeout_sec)
    try:
        while produced < args.count and time.time() < deadline:
            snapshot = IMU.poll(args.poll_sec)
            if snapshot.get("accel") or snapshot.get("gyro"):
                print(json.dumps(snapshot, ensure_ascii=False))
                produced += 1
        return 0 if produced > 0 else 2
    finally:
        IMU.deinit()


if __name__ == "__main__":
    raise SystemExit(main())
