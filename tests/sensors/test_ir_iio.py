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


def _write_iio(base: Path, device: int, value: int) -> Path:
    path = base / f"device{device}"
    path.mkdir(parents=True, exist_ok=True)
    raw_path = path / "in_proximity_raw"
    raw_path.write_text(str(int(value)), encoding="ascii")
    return raw_path


def _reload_ir_hal():
    if "ir_hal" in sys.modules:
        return importlib.reload(sys.modules["ir_hal"])
    return importlib.import_module("ir_hal")


def _run_fake() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        right_path = _write_iio(base, 4, 0)
        left_path = _write_iio(base, 3, 1)
        front_path = _write_iio(base, 5, 0)
        os.environ.update(
            {
                "IR_BACKEND": "iio",
                "IR_IIO_BASE_DIR": str(base),
                "IR_IIO_RIGHT_DEVICE": "4",
                "IR_IIO_LEFT_DEVICE": "3",
                "IR_IIO_FRONT_DEVICE": "5",
                "IR_TRIGGER_VALUE": "0",
            }
        )
        ir_hal = _reload_ir_hal()
        ir_hal._IDX_TO_IIO_PATH = {0: str(right_path), 1: str(front_path), 2: str(left_path)}
        if ir_hal.IR.init() != 0:
            raise AssertionError("fake IIO init failed")
        got = {
            "right": ir_hal.IR.is_triggered(ir_hal.IR.IDX_0),
            "front": ir_hal.IR.is_triggered(ir_hal.IR.IDX_1),
            "left": ir_hal.IR.is_triggered(ir_hal.IR.IDX_2),
        }
        print(got)
        if got != {"right": True, "front": True, "left": False}:
            raise AssertionError(f"unexpected fake IIO mapping: {got}")
    return 0


def _run_live(config: str) -> int:
    load_config_to_env(config)
    ir_hal = _reload_ir_hal()
    ret = ir_hal.IR.init()
    got = {
        "right": ir_hal.IR.is_triggered(ir_hal.IR.IDX_0),
        "front": ir_hal.IR.is_triggered(ir_hal.IR.IDX_1),
        "left": ir_hal.IR.is_triggered(ir_hal.IR.IDX_2),
    }
    print(f"init={ret} status={got}")
    if ret != 0:
        raise AssertionError("live IR init failed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test IR IIO mapping.")
    parser.add_argument("--config", default=str(ROOT / "car_control_modular/config/reid_runtime.ini"))
    parser.add_argument("--fake", action="store_true", help="Run against a temporary fake IIO tree.")
    parser.add_argument("--live", action="store_true", help="Read the board's configured IIO devices.")
    args = parser.parse_args()

    if args.live:
        return _run_live(args.config)
    return _run_fake()


if __name__ == "__main__":
    raise SystemExit(main())
