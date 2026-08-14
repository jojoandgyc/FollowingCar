import ctypes
import fcntl
import math
import os
import select
import struct
import termios
import time
from ctypes import Structure, c_char_p, c_float, c_int, c_int16, c_uint8, c_uint64
from typing import Any, Dict, List, Optional


MMWAVE_BACKEND = os.environ.get("MMWAVE_BACKEND", "at2410").strip().lower()

# AT2410 UART stream backend.  This follows the board-side reference:
# /home/topeet/devtest/golfcatdevtest/at2410/at2410_uart_test.py
MMWAVE_AT2410_PORT = os.environ.get(
    "MMWAVE_AT2410_PORT",
    "/dev/serial/by-id/usb-SIPEED_UARTx4_HS_FactoryAIOT_Prog_Serial-if00",
).strip()
MMWAVE_AT2410_BAUDRATE = int(os.environ.get("MMWAVE_AT2410_BAUDRATE", "9600"))
MMWAVE_AT2410_READ_TIMEOUT_SEC = max(0.0, float(os.environ.get("MMWAVE_AT2410_READ_TIMEOUT_SEC", "0.04")))
MMWAVE_AT2410_READ_SIZE = max(1, int(os.environ.get("MMWAVE_AT2410_READ_SIZE", "256")))
MMWAVE_AT2410_VERIFY_ON_INIT = os.environ.get("MMWAVE_AT2410_VERIFY_ON_INIT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Legacy ctypes/C HAL backend.
MMWAVE_RADAR_IDX = int(os.environ.get("MMWAVE_RADAR_IDX", "0"))
MMWAVE_RADAR_DEV_PATH = os.environ.get("MMWAVE_RADAR_DEV_PATH", "/dev/ttyCH943X3")

# The legacy C HAL currently exposes 3 targets; AT2410 also reports up to 3
# slots, with ID=0 or distance=0 representing an invalid slot.
MMWAVE_HAL_MAX_TARGETS = 3
MMWAVE_TARGET_LIMIT = max(1, min(MMWAVE_HAL_MAX_TARGETS, int(os.environ.get("MMWAVE_TARGET_LIMIT", "3"))))

# Selection modes for get_distance:
# - nearest: closest valid target inside the angle window
# - first: first valid target reported by the HAL
# - center: target with the smallest absolute angle, then nearest distance
MMWAVE_TARGET_MODE = os.environ.get("MMWAVE_TARGET_MODE", "nearest").strip().lower()
MMWAVE_FRONT_ANGLE_DEG = float(os.environ.get("MMWAVE_FRONT_ANGLE_DEG", "60"))

# MmWaveRadar.get_distance returns centimeters for compatibility with the old
# ultrasonic-style interface. get_targets returns meters.
MMWAVE_DISTANCE_BIAS_M = float(os.environ.get("MMWAVE_DISTANCE_BIAS_M", "0.0"))
MMWAVE_MIN_DISTANCE_M = float(os.environ.get("MMWAVE_MIN_DISTANCE_M", "0.50"))
MMWAVE_MIN_OUTPUT_DISTANCE_M = float(os.environ.get("MMWAVE_MIN_OUTPUT_DISTANCE_M", "0.03"))
MMWAVE_MAX_DISTANCE_M = float(os.environ.get("MMWAVE_MAX_DISTANCE_M", "10.0"))

# Radar data can lag behind actual motion. Keep a short conservative window and
# prefer the closest recent value so the car does not continue forward on stale,
# overly far readings.
MMWAVE_STALE_HOLD_SEC = float(os.environ.get("MMWAVE_STALE_HOLD_SEC", "0.35"))
MMWAVE_CONSERVATIVE_WINDOW_SEC = float(os.environ.get("MMWAVE_CONSERVATIVE_WINDOW_SEC", "0.60"))

AT2410_HEAD = 0x5A
AT2410_TARGET_REPORT_CMD = 0x0A
AT2410_MAX_TARGETS = 3
AT2410_OBJECT_SIZE = 8
AT2410_REPORT_PREFIX_SIZE = 4
AT2410_VALID_OBJECT_COUNTS = {1, 2, 3}

_libradar = None
_at2410_fd: Optional[int] = None
_at2410_buffer = bytearray()
_last_distance_cm = None
_last_distance_ts = 0.0
_recent_distances: List[tuple[float, float]] = []
_last_targets: List[Dict[str, Any]] = []
_last_targets_ts = 0.0


class RadarTarget(Structure):
    _fields_ = [
        ("x", c_int16),
        ("y", c_int16),
        ("speed", c_int16),
        ("distance", c_float),
        ("angle", c_float),
        ("valid", c_uint8),
    ]


class RadarData(Structure):
    _fields_ = [
        ("target", RadarTarget * MMWAVE_HAL_MAX_TARGETS),
        ("timestamp", c_uint64),
    ]


def _baud_constant(baudrate: int) -> int:
    value = getattr(termios, f"B{int(baudrate)}", None)
    if value is None:
        raise ValueError(f"unsupported baudrate: {baudrate}")
    return value


def _configure_serial(fd: int, baudrate: int) -> None:
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] &= ~(termios.PARENB | termios.CSTOPB | termios.CSIZE | termios.HUPCL)
    attrs[2] |= termios.CLOCAL | termios.CREAD | termios.CS8
    if hasattr(termios, "CRTSCTS"):
        attrs[2] &= ~termios.CRTSCTS
    attrs[3] = 0
    speed = _baud_constant(baudrate)
    attrs[4] = speed
    attrs[5] = speed
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    for line in (termios.TIOCM_DTR, termios.TIOCM_RTS):
        try:
            fcntl.ioctl(fd, termios.TIOCMBIS, struct.pack("I", line))
        except OSError:
            pass


def _calc_checksum(frame_without_checksum: bytes) -> int:
    return sum(frame_without_checksum) & 0xFF


def _verify_checksum(frame: bytes) -> bool:
    return len(frame) >= 4 and _calc_checksum(frame[:-1]) == frame[-1]


def _frame_size_from_header(buffer: bytes) -> Optional[int]:
    if len(buffer) < 4:
        return None
    if buffer[0] != AT2410_HEAD:
        raise ValueError(f"invalid AT2410 frame head: 0x{buffer[0]:02X}")
    if buffer[2] != AT2410_TARGET_REPORT_CMD:
        raise ValueError(f"unsupported AT2410 command: 0x{buffer[2]:02X}")
    object_count = buffer[3]
    if object_count not in AT2410_VALID_OBJECT_COUNTS:
        raise ValueError(f"invalid AT2410 object count: {object_count}")
    expected_length = 1 + AT2410_REPORT_PREFIX_SIZE + object_count * AT2410_OBJECT_SIZE
    if buffer[1] != expected_length:
        raise ValueError(f"invalid AT2410 frame length: 0x{buffer[1]:02X}, expected 0x{expected_length:02X}")
    return expected_length + 3


def _parse_at2410_frame(frame: bytes) -> List[Dict[str, Any]]:
    if len(frame) < 4:
        raise ValueError("AT2410 frame is too short")
    if frame[0] != AT2410_HEAD or frame[2] != AT2410_TARGET_REPORT_CMD:
        raise ValueError("invalid AT2410 frame")
    object_count = frame[3]
    expected_length = 1 + AT2410_REPORT_PREFIX_SIZE + object_count * AT2410_OBJECT_SIZE
    expected_size = expected_length + 3
    if object_count not in AT2410_VALID_OBJECT_COUNTS or len(frame) != expected_size:
        raise ValueError("invalid AT2410 target frame size")
    if not _verify_checksum(frame):
        raise ValueError("AT2410 checksum mismatch")

    payload = frame[3:-1]
    object_payload = payload[AT2410_REPORT_PREFIX_SIZE:]
    targets: List[Dict[str, Any]] = []
    for slot, offset in enumerate(range(0, len(object_payload), AT2410_OBJECT_SIZE)):
        chunk = object_payload[offset:offset + AT2410_OBJECT_SIZE]
        distance_cm, angle_deg, speed_cm_s, target_id, reserved = struct.unpack("<HhhBB", chunk)
        if int(target_id) <= 0 or int(distance_cm) <= 0:
            continue
        targets.append(
            {
                "index": slot,
                "target_id": int(target_id),
                "distance": float(distance_cm) / 100.0,
                "angle": float(angle_deg),
                "speed": int(speed_cm_s),
                "speed_m_s": float(speed_cm_s) / 100.0,
                "reserved": int(reserved),
                "timestamp": int(time.time() * 1000),
                "source": "at2410",
            }
        )
    return targets


def _feed_at2410(chunk: bytes) -> List[List[Dict[str, Any]]]:
    _at2410_buffer.extend(chunk)
    frames: List[List[Dict[str, Any]]] = []

    while True:
        if len(_at2410_buffer) < 4:
            break
        try:
            start = _at2410_buffer.index(AT2410_HEAD)
        except ValueError:
            _at2410_buffer.clear()
            break
        if start:
            del _at2410_buffer[:start]
        if len(_at2410_buffer) < 4:
            break

        try:
            frame_size = _frame_size_from_header(_at2410_buffer[:4])
        except ValueError:
            del _at2410_buffer[0]
            continue
        if frame_size is None or len(_at2410_buffer) < frame_size:
            break

        raw_frame = bytes(_at2410_buffer[:frame_size])
        try:
            frames.append(_parse_at2410_frame(raw_frame))
            del _at2410_buffer[:frame_size]
        except ValueError:
            del _at2410_buffer[0]

    return frames


def _open_at2410() -> int:
    global _at2410_fd
    if _at2410_fd is not None:
        return _at2410_fd
    fd = os.open(MMWAVE_AT2410_PORT, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    _configure_serial(fd, MMWAVE_AT2410_BAUDRATE)
    _at2410_fd = fd
    return fd


def _close_at2410() -> None:
    global _at2410_fd, _at2410_buffer
    if _at2410_fd is not None:
        os.close(_at2410_fd)
        _at2410_fd = None
    _at2410_buffer.clear()


def _read_at2410_frames(timeout_sec: float) -> List[List[Dict[str, Any]]]:
    fd = _open_at2410()
    frames: List[List[Dict[str, Any]]] = []
    deadline = time.monotonic() + max(0.0, timeout_sec)

    while True:
        remaining = max(0.0, deadline - time.monotonic())
        wait_sec = remaining if not frames else 0.0
        readable, _, _ = select.select([fd], [], [], wait_sec)
        if not readable:
            break
        chunk = os.read(fd, MMWAVE_AT2410_READ_SIZE)
        if not chunk:
            break
        frames.extend(_feed_at2410(chunk))
        if frames and time.monotonic() >= deadline:
            break
    return frames


def _load_radar_lib():
    global _libradar
    if _libradar is not None:
        return _libradar

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "libcvi_hal_radar.so"),
        os.path.join(base_dir, "lib_musl_riscv64", "libcvi_hal_radar.so"),
        "/mnt/system/runtime/zkwl-runtime/lib_musl_riscv64/libcvi_hal_radar.so",
        "libcvi_hal_radar.so",
    ]
    last_error = None
    for path in candidates:
        try:
            _libradar = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError as e:
            last_error = e
    if _libradar is None:
        raise last_error

    _libradar.radar_init.argtypes = [c_int, c_char_p]
    _libradar.radar_init.restype = c_int
    _libradar.radar_get_data.argtypes = [c_int, ctypes.POINTER(RadarData)]
    _libradar.radar_get_data.restype = None
    return _libradar


def _valid_target_dict(target: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    distance = float(target.get("distance", 0.0))
    angle = float(target.get("angle", 0.0))
    if not math.isfinite(distance) or not math.isfinite(angle):
        return None
    if not (MMWAVE_MIN_DISTANCE_M < distance < MMWAVE_MAX_DISTANCE_M):
        return None
    if abs(angle) > MMWAVE_FRONT_ANGLE_DEG:
        return None
    return target


def _valid_legacy_target(index, target) -> Optional[Dict[str, Any]]:
    if not target.valid:
        return None
    item = {
        "index": index,
        "x": int(target.x),
        "y": int(target.y),
        "speed": int(target.speed),
        "distance": float(target.distance),
        "angle": float(target.angle),
    }
    return _valid_target_dict(item)


def _select_distance_m(targets: List[Dict[str, Any]]) -> Optional[float]:
    candidates = []
    for item in targets[:MMWAVE_TARGET_LIMIT]:
        candidate = _valid_target_dict(item)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None

    if MMWAVE_TARGET_MODE == "first":
        selected = min(candidates, key=lambda item: int(item.get("index", 0)))
    elif MMWAVE_TARGET_MODE == "center":
        selected = min(candidates, key=lambda item: (abs(float(item.get("angle", 0.0))), float(item["distance"])))
    else:
        selected = min(candidates, key=lambda item: float(item["distance"]))
    return float(selected["distance"])


def _compensate_distance_m(distance_m):
    corrected = max(MMWAVE_MIN_OUTPUT_DISTANCE_M, distance_m - MMWAVE_DISTANCE_BIAS_M)
    now = time.monotonic()

    global _recent_distances
    _recent_distances = [
        (ts, d) for ts, d in _recent_distances
        if now - ts <= MMWAVE_CONSERVATIVE_WINDOW_SEC
    ]
    _recent_distances.append((now, corrected))
    return min(d for _, d in _recent_distances)


def _get_legacy_targets() -> List[Dict[str, Any]]:
    lib = _load_radar_lib()
    data = RadarData()
    lib.radar_get_data(MMWAVE_RADAR_IDX, ctypes.byref(data))
    targets = []
    for i in range(MMWAVE_TARGET_LIMIT):
        item = _valid_legacy_target(i, data.target[i])
        if item is None:
            continue
        item["timestamp"] = int(data.timestamp)
        targets.append(item)
    return targets


def _get_at2410_targets() -> List[Dict[str, Any]]:
    global _last_targets, _last_targets_ts

    frames = _read_at2410_frames(MMWAVE_AT2410_READ_TIMEOUT_SEC)
    now = time.monotonic()
    if frames:
        _last_targets = frames[-1]
        _last_targets_ts = now
        return list(_last_targets)

    if _last_targets and now - _last_targets_ts <= MMWAVE_STALE_HOLD_SEC:
        return list(_last_targets)
    return []


class MmWaveRadar:
    @staticmethod
    def init():
        try:
            if MMWAVE_BACKEND in {"at2410", "uart", "serial"}:
                _open_at2410()
                if MMWAVE_AT2410_VERIFY_ON_INIT:
                    _read_at2410_frames(MMWAVE_AT2410_READ_TIMEOUT_SEC)
                return 0
            lib = _load_radar_lib()
            return lib.radar_init(MMWAVE_RADAR_IDX, MMWAVE_RADAR_DEV_PATH.encode("utf-8"))
        except Exception:
            return -1

    @staticmethod
    def get_distance():
        """Return compensated front target distance in centimeters, or None."""
        global _last_distance_cm, _last_distance_ts

        try:
            distance_m = _select_distance_m(MmWaveRadar.get_targets())
        except Exception:
            distance_m = None

        now = time.monotonic()
        if distance_m is None:
            if _last_distance_cm is not None and now - _last_distance_ts <= MMWAVE_STALE_HOLD_SEC:
                return _last_distance_cm
            return None

        compensated_m = _compensate_distance_m(distance_m)
        _last_distance_cm = compensated_m * 100.0
        _last_distance_ts = now
        return _last_distance_cm

    @staticmethod
    def get_targets():
        """Return raw valid targets for logging/debugging."""
        try:
            if MMWAVE_BACKEND in {"at2410", "uart", "serial"}:
                return _get_at2410_targets()
            return _get_legacy_targets()
        except Exception:
            return []

    @staticmethod
    def deinit():
        global _last_distance_cm, _last_distance_ts, _recent_distances, _last_targets, _last_targets_ts

        _last_distance_cm = None
        _last_distance_ts = 0.0
        _recent_distances = []
        _last_targets = []
        _last_targets_ts = 0.0

        if MMWAVE_BACKEND in {"at2410", "uart", "serial"}:
            try:
                _close_at2410()
                return 0
            except Exception:
                return -1

        try:
            lib = _load_radar_lib()
            radar_deinit = getattr(lib, "radar_deinit")
        except AttributeError:
            return 0
        except Exception:
            return -1

        try:
            radar_deinit.argtypes = [c_int]
            radar_deinit.restype = c_int
            return radar_deinit(MMWAVE_RADAR_IDX)
        except Exception:
            return -1
