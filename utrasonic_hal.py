#!/usr/bin/env python3
from __future__ import annotations

import glob
import math
import os
from pathlib import Path
from typing import Optional


ULTRASONIC_BACKEND = os.environ.get("ULTRASONIC_BACKEND", "iio").strip().lower()
ULTRASONIC_IIO_BASE_DIR = os.environ.get("ULTRASONIC_IIO_BASE_DIR", "/sys/bus/iio/devices").strip()
ULTRASONIC_IIO_DEVICE = os.environ.get("ULTRASONIC_IIO_DEVICE", "").strip()
ULTRASONIC_IIO_DEVICE_NAME = os.environ.get("ULTRASONIC_IIO_DEVICE_NAME", "hcsr04").strip()
ULTRASONIC_MIN_DISTANCE_M = float(os.environ.get("ULTRASONIC_MIN_DISTANCE_M", "0.02"))
ULTRASONIC_MAX_DISTANCE_M = float(os.environ.get("ULTRASONIC_MAX_DISTANCE_M", "8.0"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="ascii").strip()


def _device_path_from_config() -> Optional[Path]:
    raw = ULTRASONIC_IIO_DEVICE
    if not raw:
        return None
    path = Path(raw)
    if raw.isdigit():
        return Path(ULTRASONIC_IIO_BASE_DIR) / f"iio:device{raw}"
    return path


def _find_iio_device(expected_name: str) -> Optional[Path]:
    configured = _device_path_from_config()
    if configured is not None:
        return configured

    pattern = str(Path(ULTRASONIC_IIO_BASE_DIR) / "iio:device*")
    for raw_path in sorted(glob.glob(pattern)):
        path = Path(raw_path)
        try:
            if _read_text(path / "name") == expected_name:
                return path
        except OSError:
            continue
    return None


class Utrasonic:
    _device_dir: Optional[Path] = None
    _raw_path: Optional[Path] = None
    _scale_path: Optional[Path] = None

    @staticmethod
    def init() -> int:
        if ULTRASONIC_BACKEND not in {"iio", "sysfs"}:
            return -1

        device_dir = _find_iio_device(ULTRASONIC_IIO_DEVICE_NAME)
        if device_dir is None:
            return -1

        raw_path = device_dir / "in_distance_raw"
        scale_path = device_dir / "in_distance_scale"
        if not raw_path.exists() or not scale_path.exists():
            return -1

        try:
            _ = _read_text(raw_path)
            _ = _read_text(scale_path)
        except OSError:
            return -1

        Utrasonic._device_dir = device_dir
        Utrasonic._raw_path = raw_path
        Utrasonic._scale_path = scale_path
        return 0

    @staticmethod
    def get_distance():
        """Return distance in centimeters for compatibility with request_0428_modular.py."""
        if Utrasonic._raw_path is None or Utrasonic._scale_path is None:
            if Utrasonic.init() != 0:
                return None

        try:
            raw = float(_read_text(Utrasonic._raw_path))
            scale = float(_read_text(Utrasonic._scale_path))
        except (OSError, TypeError, ValueError):
            return None

        distance_m = raw * scale
        if not math.isfinite(distance_m):
            return None
        if not (ULTRASONIC_MIN_DISTANCE_M < distance_m < ULTRASONIC_MAX_DISTANCE_M):
            return None
        return distance_m * 100.0

    @staticmethod
    def deinit() -> int:
        Utrasonic._device_dir = None
        Utrasonic._raw_path = None
        Utrasonic._scale_path = None
        return 0
