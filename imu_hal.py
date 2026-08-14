#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import glob
import math
import os
import selectors
import struct
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


IMU_BACKEND = os.environ.get("IMU_BACKEND", "icm20600").strip().lower()
IMU_EVENT_BASE_DIR = os.environ.get("IMU_EVENT_BASE_DIR", "/dev/input").strip()
IMU_SYS_INPUT_BASE_DIR = os.environ.get("IMU_SYS_INPUT_BASE_DIR", "/sys/class/input").strip()
IMU_ACCEL_EVENT = os.environ.get("IMU_ACCEL_EVENT", "").strip()
IMU_GYRO_EVENT = os.environ.get("IMU_GYRO_EVENT", "").strip()
IMU_ACCEL_NAME = os.environ.get("IMU_ACCEL_NAME", "gsensor").strip()
IMU_GYRO_NAME = os.environ.get("IMU_GYRO_NAME", "gyro").strip()
IMU_ACCEL_MISC_DEV = os.environ.get("IMU_ACCEL_MISC_DEV", "/dev/mma8452_daemon").strip()
IMU_GYRO_MISC_DEV = os.environ.get("IMU_GYRO_MISC_DEV", "/dev/gyrosensor").strip()
IMU_ENABLE_ON_INIT = os.environ.get("IMU_ENABLE_ON_INIT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
    "enable",
    "enabled",
}
IMU_DISABLE_ON_DEINIT = os.environ.get("IMU_DISABLE_ON_DEINIT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
    "enable",
    "enabled",
}
IMU_ACCEL_RATE_MS = int(os.environ.get("IMU_ACCEL_RATE_MS", "30"))
IMU_GYRO_RATE_MS = int(os.environ.get("IMU_GYRO_RATE_MS", "30"))
IMU_ACCEL_LSB_PER_G = float(os.environ.get("IMU_ACCEL_LSB_PER_G", "16384.0"))
IMU_GYRO_LSB_PER_DPS = float(os.environ.get("IMU_GYRO_LSB_PER_DPS", "16.4"))
IMU_STANDARD_GRAVITY = float(os.environ.get("IMU_STANDARD_GRAVITY", "9.80665"))

MIN_RATE_MS = 5
MAX_RATE_MS = 200

EV_SYN = 0x00
EV_REL = 0x02
EV_ABS = 0x03
SYN_REPORT = 0x00

ABS_X = 0x00
ABS_Y = 0x01
ABS_Z = 0x02
REL_RX = 0x03
REL_RY = 0x04
REL_RZ = 0x05

EVENT_STRUCT = struct.Struct("llHHi")


def _ioc(direction: int, kind: int, number: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (kind << 8) | number


def _io(kind: int, number: int) -> int:
    return _ioc(0, kind, number, 0)


def _iow(kind: int, number: int, size: int) -> int:
    return _ioc(1, kind, number, size)


GSENSOR_IOCTL_START = _io(ord("a"), 0x03)
GSENSOR_IOCTL_CLOSE = _io(ord("a"), 0x02)
GSENSOR_IOCTL_APP_SET_RATE = _iow(ord("a"), 0x10, struct.calcsize("h"))

L3G4200D_IOCTL_BASE = 77
L3G4200D_IOCTL_SET_DELAY = _iow(L3G4200D_IOCTL_BASE, 0x00, struct.calcsize("i"))
L3G4200D_IOCTL_SET_ENABLE = _iow(L3G4200D_IOCTL_BASE, 0x02, struct.calcsize("i"))

ACCEL_CODE_TO_AXIS = {ABS_X: "x", ABS_Y: "y", ABS_Z: "z"}
GYRO_CODE_TO_AXIS = {REL_RX: "x", REL_RY: "y", REL_RZ: "z"}


@dataclass
class AxisState:
    x: int = 0
    y: int = 0
    z: int = 0

    def tuple(self) -> Tuple[int, int, int]:
        return (int(self.x), int(self.y), int(self.z))


@dataclass
class SensorSample:
    sensor: str
    timestamp: float
    raw: AxisState

    def physical(self) -> Dict[str, Tuple[float, float, float]]:
        if self.sensor == "accel":
            gx = self.raw.x / IMU_ACCEL_LSB_PER_G
            gy = self.raw.y / IMU_ACCEL_LSB_PER_G
            gz = self.raw.z / IMU_ACCEL_LSB_PER_G
            return {
                "g": (gx, gy, gz),
                "mps2": (
                    gx * IMU_STANDARD_GRAVITY,
                    gy * IMU_STANDARD_GRAVITY,
                    gz * IMU_STANDARD_GRAVITY,
                ),
            }

        dx = self.raw.x / IMU_GYRO_LSB_PER_DPS
        dy = self.raw.y / IMU_GYRO_LSB_PER_DPS
        dz = self.raw.z / IMU_GYRO_LSB_PER_DPS
        return {
            "dps": (dx, dy, dz),
            "radps": (math.radians(dx), math.radians(dy), math.radians(dz)),
        }

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sensor": self.sensor,
            "timestamp": float(self.timestamp),
            "raw": asdict(self.raw),
        }
        out.update(self.physical())
        return out


class IMUError(RuntimeError):
    pass


class _EventReader:
    def __init__(self, sensor: str, device: str) -> None:
        self.sensor = sensor
        self.device = device
        self.fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
        self.current = AxisState()
        self.dirty = False
        self.last_ts = 0.0

    def close(self) -> None:
        os.close(self.fd)

    def fileno(self) -> int:
        return self.fd

    def _axis_map(self) -> Dict[int, str]:
        return ACCEL_CODE_TO_AXIS if self.sensor == "accel" else GYRO_CODE_TO_AXIS

    def _event_type(self) -> int:
        return EV_ABS if self.sensor == "accel" else EV_REL

    def read_available(self) -> List[SensorSample]:
        samples: List[SensorSample] = []
        try:
            payload = os.read(self.fd, EVENT_STRUCT.size * 128)
        except BlockingIOError:
            return samples

        for offset in range(0, len(payload), EVENT_STRUCT.size):
            chunk = payload[offset : offset + EVENT_STRUCT.size]
            if len(chunk) != EVENT_STRUCT.size:
                continue
            sec, usec, ev_type, code, value = EVENT_STRUCT.unpack(chunk)
            timestamp = float(sec) + float(usec) / 1_000_000.0
            if ev_type == self._event_type():
                axis_name = self._axis_map().get(code)
                if axis_name is None:
                    continue
                setattr(self.current, axis_name, int(value))
                self.last_ts = timestamp
                self.dirty = True
            elif ev_type == EV_SYN and code == SYN_REPORT and self.dirty:
                samples.append(
                    SensorSample(
                        sensor=self.sensor,
                        timestamp=self.last_ts or timestamp,
                        raw=AxisState(self.current.x, self.current.y, self.current.z),
                    )
                )
                self.dirty = False
        return samples


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _find_event_device_by_name(expected_name: str) -> Optional[str]:
    pattern = os.path.join(IMU_SYS_INPUT_BASE_DIR, "event*", "device", "name")
    for name_path in sorted(glob.glob(pattern)):
        try:
            if _read_text(name_path) != expected_name:
                continue
        except OSError:
            continue
        event_dir = os.path.basename(os.path.dirname(os.path.dirname(name_path)))
        return os.path.join(IMU_EVENT_BASE_DIR, event_dir)
    return None


def _resolve_event_device(configured: str, expected_name: str) -> Optional[str]:
    if configured:
        if configured.startswith("/"):
            return configured
        if configured.startswith("event"):
            return os.path.join(IMU_EVENT_BASE_DIR, configured)
        if configured.isdigit():
            return os.path.join(IMU_EVENT_BASE_DIR, f"event{configured}")
    return _find_event_device_by_name(expected_name)


def _clamp_rate_ms(rate_ms: int) -> int:
    return max(MIN_RATE_MS, min(MAX_RATE_MS, int(rate_ms)))


def _ioctl_no_arg(device: str, command: int) -> None:
    fd = os.open(device, os.O_RDWR)
    try:
        fcntl.ioctl(fd, command)
    finally:
        os.close(fd)


def _ioctl_int(device: str, command: int, value: int) -> None:
    fd = os.open(device, os.O_RDWR)
    try:
        fcntl.ioctl(fd, command, struct.pack("i", int(value)))
    finally:
        os.close(fd)


def _ioctl_short(device: str, command: int, value: int) -> None:
    fd = os.open(device, os.O_RDWR)
    try:
        fcntl.ioctl(fd, command, struct.pack("h", int(value)))
    finally:
        os.close(fd)


def _enable_accel() -> None:
    _ioctl_short(IMU_ACCEL_MISC_DEV, GSENSOR_IOCTL_APP_SET_RATE, _clamp_rate_ms(IMU_ACCEL_RATE_MS))
    _ioctl_no_arg(IMU_ACCEL_MISC_DEV, GSENSOR_IOCTL_START)


def _disable_accel() -> None:
    _ioctl_no_arg(IMU_ACCEL_MISC_DEV, GSENSOR_IOCTL_CLOSE)


def _enable_gyro(enable: bool) -> None:
    _ioctl_int(IMU_GYRO_MISC_DEV, L3G4200D_IOCTL_SET_DELAY, _clamp_rate_ms(IMU_GYRO_RATE_MS))
    _ioctl_int(IMU_GYRO_MISC_DEV, L3G4200D_IOCTL_SET_ENABLE, 1 if enable else 0)


class IMU:
    _selector: Optional[selectors.BaseSelector] = None
    _readers: List[_EventReader] = []
    _latest: Dict[str, SensorSample] = {}
    _accel_event: Optional[str] = None
    _gyro_event: Optional[str] = None
    _initialized = False

    @staticmethod
    def init() -> int:
        if IMU_BACKEND not in {"icm20600", "input", "linux_input"}:
            return -1
        try:
            if IMU_ENABLE_ON_INIT:
                _enable_accel()
                _enable_gyro(True)

            IMU._accel_event = _resolve_event_device(IMU_ACCEL_EVENT, IMU_ACCEL_NAME)
            IMU._gyro_event = _resolve_event_device(IMU_GYRO_EVENT, IMU_GYRO_NAME)
            if not IMU._accel_event or not IMU._gyro_event:
                raise IMUError(f"missing event device accel={IMU._accel_event} gyro={IMU._gyro_event}")

            selector = selectors.DefaultSelector()
            readers = [
                _EventReader("accel", IMU._accel_event),
                _EventReader("gyro", IMU._gyro_event),
            ]
            for reader in readers:
                selector.register(reader.fileno(), selectors.EVENT_READ, reader)

            IMU._selector = selector
            IMU._readers = readers
            IMU._latest = {}
            IMU._initialized = True
            return 0
        except Exception:
            IMU.deinit()
            return -1

    @staticmethod
    def info() -> Dict[str, Any]:
        return {
            "backend": IMU_BACKEND,
            "accel_event": IMU._accel_event or _resolve_event_device(IMU_ACCEL_EVENT, IMU_ACCEL_NAME),
            "gyro_event": IMU._gyro_event or _resolve_event_device(IMU_GYRO_EVENT, IMU_GYRO_NAME),
            "accel_misc_dev": IMU_ACCEL_MISC_DEV,
            "gyro_misc_dev": IMU_GYRO_MISC_DEV,
            "accel_lsb_per_g": IMU_ACCEL_LSB_PER_G,
            "gyro_lsb_per_dps": IMU_GYRO_LSB_PER_DPS,
            "accel_rate_ms": _clamp_rate_ms(IMU_ACCEL_RATE_MS),
            "gyro_rate_ms": _clamp_rate_ms(IMU_GYRO_RATE_MS),
            "initialized": IMU._initialized,
        }

    @staticmethod
    def poll(timeout_sec: float = 0.0) -> Dict[str, Optional[Dict[str, Any]]]:
        if not IMU._initialized or IMU._selector is None:
            if IMU.init() != 0:
                return {"accel": None, "gyro": None}

        events = IMU._selector.select(timeout=max(0.0, float(timeout_sec)))
        for key, _ in events:
            reader = key.data
            for sample in reader.read_available():
                IMU._latest[sample.sensor] = sample
        return IMU.get_latest()

    @staticmethod
    def get_latest() -> Dict[str, Optional[Dict[str, Any]]]:
        return {
            "accel": IMU._latest.get("accel").to_dict() if IMU._latest.get("accel") else None,
            "gyro": IMU._latest.get("gyro").to_dict() if IMU._latest.get("gyro") else None,
        }

    @staticmethod
    def deinit() -> int:
        selector = IMU._selector
        for reader in IMU._readers:
            try:
                if selector is not None:
                    selector.unregister(reader.fileno())
            except Exception:
                pass
            try:
                reader.close()
            except Exception:
                pass
        if selector is not None:
            try:
                selector.close()
            except Exception:
                pass
        IMU._selector = None
        IMU._readers = []
        IMU._latest = {}
        IMU._initialized = False
        if IMU_DISABLE_ON_DEINIT:
            try:
                _enable_gyro(False)
            except Exception:
                pass
            try:
                _disable_accel()
            except Exception:
                pass
        return 0
