"""Modbus RTU frame helpers for the LZ-30EMA_2EC_N driver."""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable, Optional, Sequence

from .exceptions import CRCError, ModbusException, ProtocolError, ResponseMismatch


class FunctionCode(IntEnum):
    """Function codes supported by the driver."""

    READ_HOLDING_REGISTERS = 0x03
    WRITE_SINGLE_REGISTER = 0x06
    WRITE_MULTIPLE_REGISTERS = 0x10


EXCEPTION_CODES = {
    0x01: "非法功能码",
    0x02: "非法数据地址",
    0x03: "非法数据值",
    0x04: "从站设备故障",
    0x05: "请求已被确认，但需要较长时间来处理请求",
    0x06: "从设备忙",
    0x08: "存储奇偶性差错",
    0x0A: "不可用的网关",
    0x0B: "网关目标设备响应失败",
}


def crc16(data: bytes | bytearray | Iterable[int]) -> int:
    """Return the Modbus CRC16 value for *data*.

    The returned integer is the normal Modbus value. When appended to a frame,
    the low byte must be sent first.
    """

    crc = 0xFFFF
    for byte in bytes(data):
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
            crc &= 0xFFFF
    return crc


def crc_bytes(data: bytes | bytearray | Iterable[int]) -> bytes:
    """Return CRC bytes in Modbus wire order: low byte, then high byte."""

    value = crc16(data)
    return bytes((value & 0xFF, (value >> 8) & 0xFF))


def append_crc(payload: bytes | bytearray | Iterable[int]) -> bytes:
    """Return *payload* with the Modbus CRC appended."""

    body = bytes(payload)
    return body + crc_bytes(body)


def validate_crc(frame: bytes | bytearray | Iterable[int]) -> None:
    """Raise :class:`CRCError` if *frame* has an invalid CRC."""

    raw = bytes(frame)
    if len(raw) < 4:
        raise ProtocolError(f"frame is too short: {len(raw)} bytes")
    actual = raw[-2:]
    expected = crc_bytes(raw[:-2])
    if actual != expected:
        raise CRCError(
            f"bad CRC: got {actual.hex(' ').upper()}, expected {expected.hex(' ').upper()}"
        )


def _check_byte(value: int, name: str) -> int:
    if not 0 <= int(value) <= 0xFF:
        raise ValueError(f"{name} must be between 0 and 255")
    return int(value)


def _check_u16(value: int, name: str) -> int:
    if not 0 <= int(value) <= 0xFFFF:
        raise ValueError(f"{name} must be between 0 and 65535")
    return int(value)


def build_read_holding_request(slave: int, start_address: int, count: int) -> bytes:
    """Build a 0x03 read-holding-registers request."""

    slave = _check_byte(slave, "slave")
    start_address = _check_u16(start_address, "start_address")
    count = _check_u16(count, "count")
    if count < 1:
        raise ValueError("count must be at least 1")
    body = bytes(
        (
            slave,
            FunctionCode.READ_HOLDING_REGISTERS,
            (start_address >> 8) & 0xFF,
            start_address & 0xFF,
            (count >> 8) & 0xFF,
            count & 0xFF,
        )
    )
    return append_crc(body)


def build_write_single_request(slave: int, address: int, value: int) -> bytes:
    """Build a 0x06 write-single-register request."""

    slave = _check_byte(slave, "slave")
    address = _check_u16(address, "address")
    value = _check_u16(value, "value")
    body = bytes(
        (
            slave,
            FunctionCode.WRITE_SINGLE_REGISTER,
            (address >> 8) & 0xFF,
            address & 0xFF,
            (value >> 8) & 0xFF,
            value & 0xFF,
        )
    )
    return append_crc(body)


def build_write_multiple_request(slave: int, start_address: int, values: Sequence[int]) -> bytes:
    """Build a 0x10 write-multiple-registers request."""

    slave = _check_byte(slave, "slave")
    start_address = _check_u16(start_address, "start_address")
    if not values:
        raise ValueError("values must not be empty")
    if len(values) > 0x7B:
        raise ValueError("Modbus write-multiple-registers supports at most 123 registers")
    checked = [_check_u16(value, f"values[{i}]") for i, value in enumerate(values)]
    byte_count = len(checked) * 2
    data = bytearray()
    for value in checked:
        data.extend(((value >> 8) & 0xFF, value & 0xFF))
    body = bytes(
        (
            slave,
            FunctionCode.WRITE_MULTIPLE_REGISTERS,
            (start_address >> 8) & 0xFF,
            start_address & 0xFF,
            (len(checked) >> 8) & 0xFF,
            len(checked) & 0xFF,
            byte_count,
        )
    ) + bytes(data)
    return append_crc(body)


def _parse_common(
    frame: bytes | bytearray | Iterable[int],
    expected_slave: Optional[int],
    expected_function: FunctionCode,
) -> bytes:
    raw = bytes(frame)
    if len(raw) < 5:
        raise ProtocolError(f"response is too short: {len(raw)} bytes")
    validate_crc(raw)
    if expected_slave is not None and raw[0] != expected_slave:
        raise ResponseMismatch(f"unexpected slave {raw[0]}, expected {expected_slave}")
    function = raw[1]
    if function == (int(expected_function) | 0x80):
        code = raw[2]
        meaning = EXCEPTION_CODES.get(code, "未知异常码")
        raise ModbusException(function, code, f"Modbus exception 0x{code:02X}: {meaning}")
    if function != int(expected_function):
        raise ResponseMismatch(
            f"unexpected function 0x{function:02X}, expected 0x{int(expected_function):02X}"
        )
    return raw


def parse_read_holding_response(
    frame: bytes | bytearray | Iterable[int],
    expected_slave: Optional[int] = None,
    expected_count: Optional[int] = None,
) -> list[int]:
    """Parse a 0x03 response and return register values."""

    raw = _parse_common(frame, expected_slave, FunctionCode.READ_HOLDING_REGISTERS)
    byte_count = raw[2]
    expected_length = 3 + byte_count + 2
    if len(raw) != expected_length:
        raise ProtocolError(f"read response length {len(raw)} != expected {expected_length}")
    if byte_count % 2:
        raise ProtocolError(f"read response byte count is odd: {byte_count}")
    values = []
    data = raw[3:-2]
    for i in range(0, len(data), 2):
        values.append((data[i] << 8) | data[i + 1])
    if expected_count is not None and len(values) != expected_count:
        raise ResponseMismatch(f"read returned {len(values)} registers, expected {expected_count}")
    return values


def parse_write_single_response(
    frame: bytes | bytearray | Iterable[int],
    expected_slave: Optional[int] = None,
    expected_address: Optional[int] = None,
    expected_value: Optional[int] = None,
) -> tuple[int, int]:
    """Parse a 0x06 response and return ``(address, value)``."""

    raw = _parse_common(frame, expected_slave, FunctionCode.WRITE_SINGLE_REGISTER)
    if len(raw) != 8:
        raise ProtocolError(f"write-single response length {len(raw)} != expected 8")
    address = (raw[2] << 8) | raw[3]
    value = (raw[4] << 8) | raw[5]
    if expected_address is not None and address != expected_address:
        raise ResponseMismatch(f"write address 0x{address:04X} != expected 0x{expected_address:04X}")
    if expected_value is not None and value != expected_value:
        raise ResponseMismatch(f"write value 0x{value:04X} != expected 0x{expected_value:04X}")
    return address, value


def parse_write_multiple_response(
    frame: bytes | bytearray | Iterable[int],
    expected_slave: Optional[int] = None,
    expected_start_address: Optional[int] = None,
    expected_count: Optional[int] = None,
) -> tuple[int, int]:
    """Parse a 0x10 response and return ``(start_address, count)``."""

    raw = _parse_common(frame, expected_slave, FunctionCode.WRITE_MULTIPLE_REGISTERS)
    if len(raw) != 8:
        raise ProtocolError(f"write-multiple response length {len(raw)} != expected 8")
    start_address = (raw[2] << 8) | raw[3]
    count = (raw[4] << 8) | raw[5]
    if expected_start_address is not None and start_address != expected_start_address:
        raise ResponseMismatch(
            f"write start address 0x{start_address:04X} != expected 0x{expected_start_address:04X}"
        )
    if expected_count is not None and count != expected_count:
        raise ResponseMismatch(f"write count {count} != expected {expected_count}")
    return start_address, count


def registers_to_int(values: Sequence[int], signed: bool = False) -> int:
    """Combine one or more 16-bit registers into an integer."""

    if not values:
        raise ValueError("values must not be empty")
    result = 0
    for i, value in enumerate(values):
        result = (result << 16) | _check_u16(value, f"values[{i}]")
    bit_count = len(values) * 16
    if signed and result & (1 << (bit_count - 1)):
        result -= 1 << bit_count
    return result


def int_to_registers(value: int, words: int = 2, signed: bool = False) -> list[int]:
    """Split an integer into big-endian 16-bit register words."""

    if words < 1:
        raise ValueError("words must be at least 1")
    bit_count = words * 16
    if signed:
        min_value = -(1 << (bit_count - 1))
        max_value = (1 << (bit_count - 1)) - 1
    else:
        min_value = 0
        max_value = (1 << bit_count) - 1
    value = int(value)
    if not min_value <= value <= max_value:
        raise ValueError(f"value {value} outside {bit_count}-bit range [{min_value}, {max_value}]")
    if value < 0:
        value = (1 << bit_count) + value
    return [(value >> shift) & 0xFFFF for shift in range(bit_count - 16, -1, -16)]


def decode_scaled(raw: int, scale: Optional[float], signed: bool, words: int = 1) -> int | float:
    """Decode a raw register integer using signedness and optional scale."""

    bit_count = words * 16
    value = int(raw)
    if not 0 <= value <= (1 << bit_count) - 1:
        raise ValueError(f"raw value {value} outside {bit_count}-bit range")
    if signed and value & (1 << (bit_count - 1)):
        value -= 1 << bit_count
    if scale is None:
        return value
    return value * scale


def encode_scaled(value: int | float, scale: Optional[float]) -> int:
    """Encode a logical value into a raw integer using optional scale."""

    if scale is None:
        return int(value)
    return int(round(float(value) / scale))
