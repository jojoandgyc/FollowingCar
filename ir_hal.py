import os
import logging
import threading
import time
from typing import Dict

try:
    import requests
    _REQUESTS_IMPORT_ERROR = None
except Exception as exc:
    requests = None
    _REQUESTS_IMPORT_ERROR = exc


IR_BACKEND = os.environ.get("IR_BACKEND", "http").strip().lower()
IR_HTTP_HOST = os.environ.get("IR_HTTP_HOST", "192.168.33.103")
IR_HTTP_PORT = int(os.environ.get("IR_HTTP_PORT", "8888"))
IR_HTTP_TIMEOUT_SEC = float(os.environ.get("IR_HTTP_TIMEOUT_SEC", "0.25"))
IR_HTTP_CACHE_SEC = float(os.environ.get("IR_HTTP_CACHE_SEC", "0.05"))
IR_HTTP_STALE_SEC = float(os.environ.get("IR_HTTP_STALE_SEC", "0.50"))
IR_IIO_RIGHT_DEVICE = int(os.environ.get("IR_IIO_RIGHT_DEVICE", "4"))
IR_IIO_LEFT_DEVICE = int(os.environ.get("IR_IIO_LEFT_DEVICE", "3"))
IR_IIO_FRONT_DEVICE = int(os.environ.get("IR_IIO_FRONT_DEVICE", "5"))
IR_IIO_BASE_DIR = os.environ.get("IR_IIO_BASE_DIR", "/sys/bus/iio/devices").strip()
try:
    IR_TRIGGER_VALUE = int(os.environ.get("IR_TRIGGER_VALUE", "0"))
except ValueError:
    IR_TRIGGER_VALUE = 0
if IR_TRIGGER_VALUE not in (0, 1):
    IR_TRIGGER_VALUE = 0
IR_RAW_LOG_ENABLE = os.environ.get("IR_RAW_LOG_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
IR_RAW_LOG_EVERY_SEC = max(0.05, float(os.environ.get("IR_RAW_LOG_EVERY_SEC", "1.0")))

# Old request-side index mapping:
#   IR0 = right, IR1 = front, IR2 = left
# New board HTTP protocol:
#   /ir_status -> {"ir1": ..., "ir2": ..., "ir3": ...}
# Required mapping:
#   protocol ir1 -> old IR2 = left
#   protocol ir2 -> old IR1 = front
#   protocol ir3 -> old IR0 = right
_IDX_TO_PROTOCOL_KEY = {
    0: "ir3",
    1: "ir2",
    2: "ir1",
}

_lock = threading.Lock()
_last_status: Dict[str, int] = {}
_last_read_ts = 0.0
_last_log_ts = 0.0
_last_log_key = None
_iio_log_state = {}
_session = requests.Session() if requests is not None else None
_logger = logging.getLogger("PersonTracker")


def _iio_path(device_index: int) -> str:
    return os.path.join(IR_IIO_BASE_DIR, f"iio:device{int(device_index)}", "in_proximity_raw")


_IDX_TO_IIO_PATH = {
    0: _iio_path(IR_IIO_RIGHT_DEVICE),
    1: _iio_path(IR_IIO_FRONT_DEVICE),
    2: _iio_path(IR_IIO_LEFT_DEVICE),
}


def _read_iio_value(path: str) -> int:
    with open(path, "r", encoding="ascii") as fh:
        value = int(fh.read().strip())
    if value not in (0, 1):
        raise ValueError(f"IR IIO raw value must be 0 or 1: {value!r}")
    return value


def _maybe_log_iio(idx, path, *, value=None, error=None, force=False) -> None:
    """Report the existing read, without another device access or a cache.

    The HTTP diagnostics do not run for IIO. Keep per-channel state so that a
    clear front sensor cannot hide a blocked right sensor's change/release.
    This state is diagnostic only and never decides whether motion is allowed.
    """
    if not IR_RAW_LOG_ENABLE:
        return
    triggered = error is not None or value == IR_TRIGGER_VALUE
    error_text = None if error is None else f"{type(error).__name__}: {error}"
    key = (path, value, error_text, IR_TRIGGER_VALUE)
    with _lock:
        now = time.monotonic()
        previous = _iio_log_state.get(idx)
        previous_since = previous[2] if previous is not None else None
        blocked_since = (previous_since if previous_since is not None else now) if triggered else None
        previous_blocked_ms = 0.0 if previous_since is None else max(0.0, now - previous_since) * 1000.0
        changed = previous is None or previous[0] != key
        if not force and not changed and now - previous[1] < IR_RAW_LOG_EVERY_SEC:
            return
        _iio_log_state[idx] = (key, now, blocked_since)
    log = _logger.warning if triggered else _logger.info
    try:
        log(
            "IR IIO status side=%s idx=%s path=%s raw=%s trigger_value=%s "
            "triggered=%s fail_closed=%s event=%s blocked_observed_ms=%.1f "
            "previous_blocked_observed_ms=%.1f error=%s",
            {0: "right", 1: "front", 2: "left"}.get(idx, "unknown"),
            idx, path, value, IR_TRIGGER_VALUE, triggered, error is not None,
            "initial" if previous is None else ("changed" if changed else "held"),
            0.0 if blocked_since is None else max(0.0, now - blocked_since) * 1000.0,
            previous_blocked_ms if not triggered else 0.0, error_text,
        )
    except Exception:
        # A broken output handler must not replace a real hazard reading with
        # an exception that callers may interpret differently from a trigger.
        pass


def _status_url() -> str:
    return f"http://{IR_HTTP_HOST}:{IR_HTTP_PORT}/ir_status"


def _parse_status(obj) -> Dict[str, int]:
    if not isinstance(obj, dict):
        raise ValueError(f"IR status response is not an object: {obj!r}")

    status = {}
    for key in ("ir1", "ir2", "ir3"):
        if key not in obj:
            raise ValueError(f"IR status missing field: {key}")
        value = int(obj[key])
        if value not in (0, 1):
            raise ValueError(f"IR status {key} must be 0 or 1: {value!r}")
        status[key] = value
    return status


def _mapped_triggered(status: Dict[str, int]) -> Dict[str, bool]:
    return {
        "front": status.get("ir2") == IR_TRIGGER_VALUE,
        "left": status.get("ir1") == IR_TRIGGER_VALUE,
        "right": status.get("ir3") == IR_TRIGGER_VALUE,
    }


def _maybe_log_status(status: Dict[str, int], source: str, force: bool = False) -> None:
    global _last_log_ts, _last_log_key

    if not IR_RAW_LOG_ENABLE:
        return

    mapped = _mapped_triggered(status)
    key = (
        source,
        status.get("ir1"),
        status.get("ir2"),
        status.get("ir3"),
        mapped["front"],
        mapped["left"],
        mapped["right"],
    )
    now = time.monotonic()
    with _lock:
        if not force and key == _last_log_key and now - _last_log_ts < IR_RAW_LOG_EVERY_SEC:
            return
        _last_log_key = key
        _last_log_ts = now

    _logger.info(
        "IR raw status source=%s url=%s raw(ir1=%s,ir2=%s,ir3=%s) mapped(front=%s,left=%s,right=%s) protocol(triggered=%s)",
        source,
        _status_url(),
        status.get("ir1"),
        status.get("ir2"),
        status.get("ir3"),
        mapped["front"],
        mapped["left"],
        mapped["right"],
        IR_TRIGGER_VALUE,
    )


def _maybe_log_error(exc: Exception) -> None:
    global _last_log_ts, _last_log_key

    if not IR_RAW_LOG_ENABLE:
        return

    key = ("error", exc.__class__.__name__, str(exc))
    now = time.monotonic()
    with _lock:
        if key == _last_log_key and now - _last_log_ts < IR_RAW_LOG_EVERY_SEC:
            return
        _last_log_key = key
        _last_log_ts = now

    _logger.warning(
        "IR raw status read failed url=%s error=%s: %s; fail_closed=True",
        _status_url(),
        exc.__class__.__name__,
        exc,
    )


def _read_status(force: bool = False) -> Dict[str, int]:
    global _last_status, _last_read_ts

    now = time.monotonic()
    cached_status = None
    with _lock:
        if not force and _last_status and now - _last_read_ts <= IR_HTTP_CACHE_SEC:
            cached_status = dict(_last_status)
    if cached_status is not None:
        _maybe_log_status(cached_status, "cache")
        return cached_status

    try:
        if _session is None:
            raise RuntimeError(f"requests unavailable for IR HTTP backend: {_REQUESTS_IMPORT_ERROR}")
        response = _session.get(_status_url(), timeout=IR_HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        status = _parse_status(response.json())
    except Exception as exc:
        stale_status = None
        with _lock:
            if _last_status and now - _last_read_ts <= IR_HTTP_STALE_SEC:
                stale_status = dict(_last_status)
        if stale_status is not None:
            _maybe_log_status(stale_status, "stale_after_error")
            return stale_status
        _maybe_log_error(exc)
        raise

    with _lock:
        _last_status = dict(status)
        _last_read_ts = time.monotonic()
        status = dict(_last_status)
    _maybe_log_status(status, "http", force=force)
    return status


class IR:
    IDX_0 = 0
    IDX_1 = 1
    IDX_2 = 2

    @staticmethod
    def init():
        if IR_BACKEND in {"iio", "sysfs"}:
            try:
                for idx, path in _IDX_TO_IIO_PATH.items():
                    value = _read_iio_value(path)
                    _maybe_log_iio(idx, path, value=value, force=True)
                return 0
            except Exception as exc:
                if IR_RAW_LOG_ENABLE:
                    _logger.warning("IR IIO init failed paths=%s error=%s", _IDX_TO_IIO_PATH, exc)
                return -1
        try:
            _read_status(force=True)
            return 0
        except Exception:
            return -1

    @staticmethod
    def is_triggered(idx):
        if IR_BACKEND in {"iio", "sysfs"}:
            channel = None
            path = None
            try:
                channel = int(idx)
                path = _IDX_TO_IIO_PATH[channel]
                value = _read_iio_value(path)
            except Exception as exc:
                _maybe_log_iio(channel, path, error=exc)
                return True
            _maybe_log_iio(channel, path, value=value)
            return value == IR_TRIGGER_VALUE
        try:
            key = _IDX_TO_PROTOCOL_KEY[int(idx)]
            status = _read_status()
            return status[key] == IR_TRIGGER_VALUE
        except Exception:
            # Fail closed: if IR cannot be read and no fresh cached value exists,
            # treat the requested sensor as triggered so the car does not keep moving.
            return True

    @staticmethod
    def deinit():
        with _lock:
            _last_status.clear()
            _iio_log_state.clear()
        return 0
