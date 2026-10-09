"""Serial transport for real RS485 communication."""

from __future__ import annotations

import os
import select
import time
from typing import Optional

if os.name == "posix":
    import fcntl
    import struct
    import termios
    import tty
else:
    fcntl = None  # type: ignore[assignment]
    struct = None  # type: ignore[assignment]
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]


class SerialTransport:
    """Serial wrapper.

    pyserial is used when available. On POSIX systems, a small termios based
    fallback is used so embedded boards can run without installing packages.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 0.5,
        parity: str = "N",
        stopbits: int = 1,
        bytesize: int = 8,
        rs485_mode: str = "auto",
    ) -> None:
        try:
            import serial
        except ImportError as exc:
            if os.name != "posix":
                raise RuntimeError("pyserial is required for SerialTransport on this platform") from exc
            self._serial = _PosixSerial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                parity=parity,
                stopbits=stopbits,
                bytesize=bytesize,
                rs485_mode=rs485_mode,
            )
            return

        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
            timeout=timeout,
        )
        if rs485_mode not in {"auto", "none"}:
            raise RuntimeError("manual RTS RS485 modes require the POSIX serial fallback")

    def write(self, data: bytes) -> int:
        return self._serial.write(data)

    def read(self, size: int) -> bytes:
        return self._serial.read(size)

    def close(self) -> None:
        self._serial.close()

    @property
    def timeout(self) -> Optional[float]:
        return self._serial.timeout

    @timeout.setter
    def timeout(self, value: Optional[float]) -> None:
        self._serial.timeout = value


class _PosixSerial:
    BAUD_RATES = {
        9600: termios.B9600 if termios else 13,
        19200: termios.B19200 if termios else 14,
        38400: termios.B38400 if termios else 15,
        57600: termios.B57600 if termios else 4097,
        115200: termios.B115200 if termios else 4098,
    }

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 0.5,
        parity: str = "N",
        stopbits: int = 1,
        bytesize: int = 8,
        rs485_mode: str = "auto",
    ) -> None:
        if bytesize != 8:
            raise ValueError("the POSIX fallback currently supports 8 data bits only")
        if termios is None or tty is None:
            raise RuntimeError("POSIX serial fallback is unavailable on this platform")
        if baudrate not in self.BAUD_RATES:
            raise ValueError(f"unsupported baudrate for POSIX fallback: {baudrate}")
        rs485_mode = rs485_mode.lower()
        if rs485_mode not in {"auto", "none", "rts-high", "rts-low"}:
            raise ValueError("rs485_mode must be auto, none, rts-high, or rts-low")
        self._fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        self.timeout: Optional[float] = timeout
        self._manual_rts_tx_level: Optional[bool] = None
        if rs485_mode == "auto":
            self._enable_rs485_if_supported()
        elif rs485_mode == "rts-high":
            self._manual_rts_tx_level = True
        elif rs485_mode == "rts-low":
            self._manual_rts_tx_level = False
        self._configure(baudrate, parity, stopbits)
        if self._manual_rts_tx_level is not None:
            self._set_rts(not self._manual_rts_tx_level)

    def _configure(self, baudrate: int, parity: str, stopbits: int) -> None:
        attrs = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        attrs = termios.tcgetattr(self._fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8
        attrs[3] = 0
        attrs[4] = self.BAUD_RATES[baudrate]
        attrs[5] = self.BAUD_RATES[baudrate]
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0

        parity = parity.upper()
        if parity == "E":
            attrs[2] |= termios.PARENB
        elif parity == "O":
            attrs[2] |= termios.PARENB | termios.PARODD
        elif parity != "N":
            raise ValueError("parity must be N, E, or O")

        if stopbits == 2:
            attrs[2] |= termios.CSTOPB
        elif stopbits != 1:
            raise ValueError("stopbits must be 1 or 2")

        termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
        termios.tcflush(self._fd, termios.TCIOFLUSH)

    def _enable_rs485_if_supported(self) -> None:
        if fcntl is None or struct is None:
            return
        tiocgrs485 = 0x542E
        tiocsrs485 = 0x542F
        ser_rs485_enabled = 1 << 0
        ser_rs485_rts_on_send = 1 << 1
        try:
            current = bytearray(32)
            fcntl.ioctl(self._fd, tiocgrs485, current, True)
            values = list(struct.unpack("8I", bytes(current)))
            values[0] |= ser_rs485_enabled | ser_rs485_rts_on_send
            packed = struct.pack("8I", *values)
            fcntl.ioctl(self._fd, tiocsrs485, packed)
        except OSError:
            # Plain UARTs do not support this ioctl. They still work as normal
            # serial ports, so keep going.
            return

    def _set_rts(self, enabled: bool) -> None:
        if fcntl is None or struct is None:
            raise RuntimeError("manual RTS control is unavailable on this platform")
        tiocmbis = 0x5416
        tiocmbic = 0x5417
        tiocm_rts = 0x004
        request = tiocmbis if enabled else tiocmbic
        fcntl.ioctl(self._fd, request, struct.pack("I", tiocm_rts))

    def write(self, data: bytes) -> int:
        if self._manual_rts_tx_level is None:
            written = os.write(self._fd, data)
            termios.tcdrain(self._fd)
            return written

        self._set_rts(self._manual_rts_tx_level)
        time.sleep(0.001)
        try:
            written = os.write(self._fd, data)
            termios.tcdrain(self._fd)
        finally:
            time.sleep(0.001)
            self._set_rts(not self._manual_rts_tx_level)
        return written

    def read(self, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        deadline = None if self.timeout is None else time.monotonic() + float(self.timeout)
        while remaining > 0:
            wait = None
            if deadline is not None:
                wait = max(0.0, deadline - time.monotonic())
                if wait <= 0:
                    break
            readable, _, _ = select.select([self._fd], [], [], wait)
            if not readable:
                break
            chunk = os.read(self._fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
