#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mmwave_hal


def main() -> int:
    original_read = mmwave_hal._read_at2410_frames
    original_open = mmwave_hal._open_at2410
    original_close = mmwave_hal._close_at2410
    original_verify = mmwave_hal.MMWAVE_AT2410_VERIFY_ON_INIT
    original_reset_on_init = mmwave_hal.MMWAVE_AT2410_USB_RESET_ON_INIT
    original_targets = mmwave_hal._last_targets
    original_targets_ts = mmwave_hal._last_targets_ts
    closed = []
    try:
        mmwave_hal._open_at2410 = lambda: 123
        mmwave_hal._close_at2410 = lambda: closed.append(True)
        mmwave_hal.MMWAVE_AT2410_USB_RESET_ON_INIT = False

        mmwave_hal._read_at2410_frames = lambda _timeout: [[{"target_id": 1}]]
        if not mmwave_hal.MmWaveRadar.wait_for_valid_frame(0.1):
            raise AssertionError("a parsed AT2410 frame must pass the health check")

        mmwave_hal._read_at2410_frames = lambda _timeout: []
        if mmwave_hal.MmWaveRadar.wait_for_valid_frame(0.1):
            raise AssertionError("an empty serial interval must fail the health check")

        mmwave_hal.MMWAVE_AT2410_VERIFY_ON_INIT = True
        if mmwave_hal.MmWaveRadar.init() == 0:
            raise AssertionError("verified initialization must fail without a valid frame")
        if not closed:
            raise AssertionError("failed verified initialization must close the serial port")

        mmwave_hal._read_at2410_frames = lambda _timeout: [[{"target_id": 7}]]
        if mmwave_hal.MmWaveRadar.init() != 0:
            raise AssertionError("verified initialization must pass with a valid frame")
        if mmwave_hal._last_targets != [{"target_id": 7}]:
            raise AssertionError("the verified frame must be retained for the runtime reader")
    finally:
        mmwave_hal._read_at2410_frames = original_read
        mmwave_hal._open_at2410 = original_open
        mmwave_hal._close_at2410 = original_close
        mmwave_hal.MMWAVE_AT2410_VERIFY_ON_INIT = original_verify
        mmwave_hal.MMWAVE_AT2410_USB_RESET_ON_INIT = original_reset_on_init
        mmwave_hal._last_targets = original_targets
        mmwave_hal._last_targets_ts = original_targets_ts

    print("mmwave health gate ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
