"""Bounded, single-request Modbus RTU transactions for the motor client.

This adapter never opens a device, retransmits a request, or restores motion
authority. A failed transaction leaves response ownership uncertain until a
new client is constructed; only emergency STOP, parking-current clearing and
untrusted diagnostic reads remain possible on that client.
"""
from __future__ import annotations

from collections import deque
import logging
import math
import threading
import time


class RtuProtocolError(OSError):
    pass


class RtuLinkUncertain(RuntimeError):
    pass


def _crc(data: bytes) -> int:
    value = 0xFFFF
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0xA001 if value & 1 else 0)
    return value


def _check_crc(frame: bytes) -> None:
    if len(frame) < 4 or _crc(frame[:-2]) != int.from_bytes(frame[-2:], "little"):
        raise RtuProtocolError("invalid RTU response CRC")


def _safe_when_uncertain(request: bytes) -> bool:
    function = request[1]
    address = int.from_bytes(request[2:4], "big")
    if function == 0x03:
        return True  # Data is collected for diagnostics, never returned to control.
    if function == 0x06:
        value = int.from_bytes(request[4:6], "big")
        return ((address in (0x0040, 0x0044) and value == 1)
                or (address in (0x006D, 0x0076) and value == 0))
    return (function == 0x10 and address in (0x006D, 0x0076)
            and request[4:7] == b"\x00\x01\x02" and request[7:9] == b"\x00\x00"
            and len(request) == 11)


class MotorRtuGuard:
    """Serialize complete transactions and preserve the first ambiguous result.

    The deadline includes lock acquisition, RTU quiet time and reads. Where
    supported, the serial write timeout is reduced to the remaining budget.
    A transport without write_timeout (the external POSIX fallback uses
    tcdrain) cannot be forcibly interrupted; overruns are detected on return.
    """

    def __init__(self, transport, *, baudrate: int, timeout: float,
                 logger=None, clock=None, sleep=None):
        if not math.isfinite(float(timeout)) or float(timeout) <= 0:
            raise ValueError("motor RTU timeout must be finite and positive")
        if int(baudrate) <= 0:
            raise ValueError("motor RTU baudrate must be positive")
        self.transport = transport
        self.timeout = float(timeout)
        self.quiet_sec = 0.00175 if int(baudrate) > 19200 else 3.5 * 11 / int(baudrate)
        self.logger = logger or logging.getLogger(__name__)
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._lock = threading.Lock()
        self._last_activity = None
        self._sequence = 0
        self.rx_uncertain = False
        self.fault_reason = None
        self.last_error = None
        self.last_transaction = None
        self.recent = deque(maxlen=12)
        serial = getattr(transport, "_serial", None)
        self._write_timeout_owner = next((obj for obj in (transport, serial)
                                          if obj is not None and hasattr(obj, "write_timeout")), None)
        self._reset_owner = next((obj for obj in (transport, serial)
                                  if callable(getattr(obj, "reset_input_buffer", None))), None)

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise TimeoutError("motor RTU transaction deadline exceeded")
        return remaining

    def _read_to(self, response: bytearray, count: int, deadline: float, details: dict) -> None:
        while len(response) < count:
            remaining = self._remaining(deadline)
            self.transport.timeout = remaining
            details["read_calls"] += 1
            chunk = self.transport.read(count - len(response))
            received_at = self._clock()
            self._last_activity = received_at
            if chunk and details["first_rx_ms"] is None:
                details["first_rx_ms"] = (received_at - details["started"]) * 1000
            if not chunk:
                raise TimeoutError("no response from driver" if not response
                                   else "incomplete response from driver")
            response.extend(chunk)
            if len(response) > count:
                raise RtuProtocolError("transport returned more bytes than requested")
            self._remaining(deadline)

    def _read_response(self, response: bytearray, deadline: float, details: dict) -> None:
        self._read_to(response, 2, deadline, details)
        function = response[1]
        if function & 0x80:
            length = 5
        elif function == 0x03:
            self._read_to(response, 3, deadline, details)
            if response[2] > 250 or response[2] % 2:
                raise RtuProtocolError("invalid RTU read byte count")
            length = 5 + response[2]
        elif function in (0x06, 0x10):
            length = 8
        else:
            raise RtuProtocolError("unexpected RTU response function 0x%02X" % function)
        self._read_to(response, length, deadline, details)

    @staticmethod
    def _validate(request: bytes, response: bytes, expected_length: int) -> None:
        _check_crc(response)
        if response[0] != request[0]:
            raise RtuProtocolError("RTU response slave mismatch")
        if response[1] == (request[1] | 0x80):
            raise RtuProtocolError("Modbus exception 0x%02X" % response[2])
        if response[1] != request[1]:
            raise RtuProtocolError("RTU response function mismatch")
        if len(response) != expected_length:
            raise RtuProtocolError("RTU response length mismatch")
        if request[1] in (0x06, 0x10) and response[:6] != request[:6]:
            raise RtuProtocolError("RTU response register/value/count mismatch")
        if request[1] == 0x03 and response[2] != 2 * int.from_bytes(request[4:6], "big"):
            raise RtuProtocolError("RTU read response count mismatch")

    def _reset_received_input(self, details: dict) -> None:
        """Discard only bytes already received, without claiming resynchronization."""
        details["rx_reset"] = "unsupported"
        if self._reset_owner is None:
            return
        try:
            details["rx_discard_available"] = getattr(self._reset_owner, "in_waiting", None)
            self._reset_owner.reset_input_buffer()
            # A drained byte may have arrived just before this call. Require a
            # new quiet interval, without treating the drain as resynchronizing.
            self._last_activity = self._clock()
            details["rx_reset"] = "performed"
        except Exception as exc:
            # An unavailable cleanup must not prevent an emergency STOP write.
            details["rx_reset"] = "failed:%s:%s" % (type(exc).__name__, str(exc)[:120])

    def _record(self, details: dict, response: bytearray, error) -> None:
        details["elapsed_ms"] = (self._clock() - details["started"]) * 1000
        details["rx_hex"] = bytes(response[:64]).hex()
        details["rx_length"] = len(response)
        if error is not None:
            self.rx_uncertain = True
            reason = ("seq=%s stage=%s function=%s register=%s %s:%s" % (
                details["seq"], details["stage"], details["function"], details["register"],
                type(error).__name__, str(error)[:240]))
            self.last_error = reason
            if self.fault_reason is None:
                self.fault_reason = reason
            details["error"] = reason
        details["rx_uncertain"] = self.rx_uncertain
        self.last_transaction = details
        recent = list(self.recent) if error is not None else None
        self.recent.append({key: details[key] for key in (
            "seq", "function", "register", "stage", "elapsed_ms", "rx_length", "rx_uncertain")})
        try:
            if error is not None:
                self.logger.error("motor_rtu_transaction_failed details=%s recent=%s "
                                  "motion_authorized=False physical_stillness=unverified",
                                  details, recent)
            elif self.rx_uncertain:
                self.logger.warning("motor_rtu_safe_response details=%s "
                                    "link_uncertain=True motion_authorized=False "
                                    "physical_stillness=unverified", details)
        except Exception:
            # Broken logging must not hide the original wire failure or keep
            # the other wheel's emergency STOP from being attempted.
            pass

    def transact(self, request: bytes, expected_response_length: int) -> bytes:
        request = bytes(request)
        started = self._clock()
        deadline = started + self.timeout
        details = {"started": started, "seq": None, "stage": "lock",
                   "thread": threading.current_thread().name,
                   "function": "0x%02X" % request[1] if len(request) >= 2 else "missing",
                   "register": "0x%04X" % int.from_bytes(request[2:4], "big") if len(request) >= 4 else "missing",
                   "tx_hex": request[:64].hex(), "tx_length": len(request), "tx_written": 0,
                   "expected_rx_length": expected_response_length, "read_calls": 0,
                   "first_rx_ms": None, "quiet_wait_ms": 0.0,
                   "write_timeout_supported": self._write_timeout_owner is not None,
                   "rx_reset": "not_needed"}
        response = bytearray()
        acquired = self._lock.acquire(timeout=self.timeout)
        if not acquired:
            error = TimeoutError("motor RTU transaction lock deadline exceeded")
            self._record(details, response, error)
            raise error
        self._sequence += 1
        details["seq"] = self._sequence
        details["lock_wait_ms"] = (self._clock() - started) * 1000
        saved_timeout = None
        saved_write_timeout = None
        timeout_saved = False
        write_timeout_saved = False
        error = None
        try:
            details["stage"] = "request"
            if len(request) < 8 or request[1] not in (0x03, 0x06, 0x10):
                raise RtuProtocolError("unsupported motor RTU request")
            _check_crc(request)
            details["stage"] = "uncertain_policy"
            if self.rx_uncertain and not _safe_when_uncertain(request):
                raise RtuLinkUncertain("motor RTU motion/parameter write blocked: " + self.fault_reason)
            saved_timeout = self.transport.timeout
            timeout_saved = True
            if self._write_timeout_owner is not None:
                saved_write_timeout = self._write_timeout_owner.write_timeout
                write_timeout_saved = True
            if self.rx_uncertain:
                self._reset_received_input(details)
            details["stage"] = "quiet"
            wait = (0.0 if self._last_activity is None else
                    max(0.0, self.quiet_sec - (self._clock() - self._last_activity)))
            remaining = self._remaining(deadline)
            if wait > 0:
                before = self._clock()
                self._sleep(min(wait, remaining))
                details["quiet_wait_ms"] = (self._clock() - before) * 1000
                self._remaining(deadline)
            details["stage"] = "write"
            if self._write_timeout_owner is not None:
                self._write_timeout_owner.write_timeout = self._remaining(deadline)
            details["tx_written"] = self.transport.write(request)
            self._last_activity = self._clock()
            if details["tx_written"] != len(request):
                raise RtuProtocolError("short serial write: %r of %d bytes" %
                                       (details["tx_written"], len(request)))
            self._remaining(deadline)
            details["stage"] = "read"
            self._read_response(response, deadline, details)
            details["stage"] = "validate"
            self._validate(request, bytes(response), expected_response_length)
            if self.rx_uncertain and request[1] == 0x03:
                raise RtuLinkUncertain("diagnostic response received on uncertain motor RTU link")
            self._remaining(deadline)
            details["stage"] = "complete"
        except Exception as exc:
            if details["stage"] in ("write", "read", "validate"):
                # A serial exception can follow a partial transfer without
                # returning its bytes/count. Preserve quiet before safety I/O.
                self._last_activity = self._clock()
            error = exc
        finally:
            try:
                if timeout_saved:
                    self.transport.timeout = saved_timeout
                if write_timeout_saved:
                    self._write_timeout_owner.write_timeout = saved_write_timeout
            except Exception as restore_exc:
                details["timeout_restore_error"] = str(restore_exc)[:120]
                if error is None:
                    details["stage"] = "restore_timeout"
                    error = restore_exc
            try:
                self._record(details, response, error)
            finally:
                self._lock.release()
        if error is not None:
            raise error
        return bytes(response)


def install_motor_rtu_guard(driver, *, baudrate: int, timeout: float, logger=None):
    """Wrap an existing real client; test adapters without transport are untouched."""
    existing = getattr(driver, "_motor_rtu_guard", None)
    if isinstance(existing, MotorRtuGuard):
        return existing
    transport = getattr(driver, "transport", None)
    if transport is None or not callable(getattr(driver, "_transact", None)):
        return None
    guard = MotorRtuGuard(transport, baudrate=baudrate, timeout=timeout, logger=logger)
    driver._motor_rtu_guard = guard
    driver._transact = guard.transact
    guard.logger.info("motor_rtu_guard_installed transport=%s serial=%s timeout_sec=%.3f "
                      "quiet_ms=%.3f write_timeout_supported=%s rx_reset_supported=%s "
                      "request_retries=0",
                      type(transport).__name__, type(getattr(transport, "_serial", None)).__name__,
                      guard.timeout, guard.quiet_sec * 1000,
                      guard._write_timeout_owner is not None, guard._reset_owner is not None)
    return guard
