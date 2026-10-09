"""High-level client for LZ-30EMA_2EC_N motor drivers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Optional, Protocol

from .exceptions import ProtocolError
from .protocol import (
    build_read_holding_request,
    build_write_multiple_request,
    build_write_single_request,
    encode_scaled,
    int_to_registers,
    parse_read_holding_response,
    parse_write_multiple_response,
    parse_write_single_response,
    registers_to_int,
)
from .registers import (
    BAUD_RATE_CODES,
    CAN_BAUD_RATE_CODES,
    ERROR_STATES,
    PARITY_CODES,
    Register,
    ValueField,
    get_field,
    get_register,
)
from .transport import SerialTransport


class BinaryTransport(Protocol):
    def write(self, data: bytes) -> Any:
        ...

    def read(self, size: int) -> bytes:
        ...


class MotorSide(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class StopMode(IntEnum):
    NORMAL = 0
    EMERGENCY = 1
    FREE = 2


class SystemMode(IntEnum):
    INDEPENDENT_CLOSED_LOOP = 1
    SAME_SOURCE_CLOSED_LOOP = 3
    DIFFERENTIAL_CLOSED_LOOP = 4


@dataclass(frozen=True)
class MotorStatus:
    side: MotorSide
    phase_current_a: float
    speed_rpm: int
    error_code: int
    error_name: str
    pwm_percent: float
    board_temperature_c: int
    position_degree: int
    reverse: bool


@dataclass(frozen=True)
class RealtimeSnapshot:
    device_info: dict[str, int | float | tuple[int, int]]
    bus_status: dict[str, int]
    left_motor: MotorStatus
    right_motor: MotorStatus


def _side(value: str | MotorSide) -> MotorSide:
    if isinstance(value, MotorSide):
        return value
    try:
        return MotorSide(value.lower())
    except ValueError as exc:
        raise ValueError("side must be 'right' or 'left'") from exc


def _to_signed_u16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def _decode_register_value(register: Register, raw: int) -> int | float:
    value = _to_signed_u16(raw) if register.signed else raw
    if register.scale is None:
        return value
    return value * register.scale


def _encode_register_value(register: Register, value: int | float) -> int:
    raw = encode_scaled(value, register.scale)
    if register.signed and raw < 0:
        raw = (1 << 16) + raw
    if not 0 <= raw <= 0xFFFF:
        raise ValueError(f"{register.name} encoded value outside 16-bit range: {raw}")
    return raw


def _decode_field_value(field: ValueField, values: list[int]) -> int | float:
    raw = registers_to_int(values, signed=field.signed)
    if field.scale is None:
        return raw
    return raw * field.scale


def _encode_field_value(field: ValueField, value: int | float) -> list[int]:
    raw = encode_scaled(value, field.scale)
    return int_to_registers(raw, words=field.words, signed=field.signed)


class LZ30EMAClient:
    """Client for one LZ-30EMA_2EC_N slave address."""

    def __init__(self, transport: Optional[BinaryTransport], slave: int = 1) -> None:
        if not 0 <= slave <= 255:
            raise ValueError("slave must be between 0 and 255")
        self.transport = transport
        self.slave = slave

    @classmethod
    def from_serial(
        cls,
        port: str,
        slave: int = 1,
        baudrate: int = 9600,
        timeout: float = 0.5,
        parity: str = "N",
        stopbits: int = 1,
        rs485_mode: str = "auto",
    ) -> "LZ30EMAClient":
        return cls(
            SerialTransport(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                parity=parity,
                stopbits=stopbits,
                rs485_mode=rs485_mode,
            ),
            slave=slave,
        )

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if close is not None:
            close()

    def _require_transport(self) -> BinaryTransport:
        if self.transport is None:
            raise RuntimeError("this operation requires a transport")
        return self.transport

    def _transact(self, request: bytes, expected_response_length: int) -> bytes:
        transport = self._require_transport()
        transport.write(request)
        response = transport.read(expected_response_length)
        if not response:
            raise TimeoutError("no response from driver")
        return response

    def read_holding_registers(self, start_address: int, count: int) -> list[int]:
        request = build_read_holding_request(self.slave, start_address, count)
        response = self._transact(request, 5 + count * 2)
        return parse_read_holding_response(response, self.slave, count)

    def write_single_register(self, address: int, value: int) -> tuple[int, int]:
        request = build_write_single_request(self.slave, address, value)
        response = self._transact(request, 8)
        return parse_write_single_response(response, self.slave, address, value)

    def write_multiple_registers(self, start_address: int, values: list[int] | tuple[int, ...]) -> tuple[int, int]:
        request = build_write_multiple_request(self.slave, start_address, values)
        response = self._transact(request, 8)
        return parse_write_multiple_response(response, self.slave, start_address, len(values))

    def read_register(self, register: str | int | Register, decode: bool = True) -> int | float:
        spec = get_register(register)
        raw = self.read_holding_registers(spec.address, 1)[0]
        return _decode_register_value(spec, raw) if decode else raw

    def write_register(
        self,
        register: str | int | Register,
        value: int | float,
        *,
        persist: bool = False,
        encode: bool = True,
    ) -> tuple[int, int]:
        spec = get_register(register)
        raw = _encode_register_value(spec, value) if encode else int(value)
        if persist or 0x06 not in spec.functions:
            if 0x10 not in spec.functions:
                raise ProtocolError(f"register {spec.name} does not support 0x10 writes")
            start, count = self.write_multiple_registers(spec.address, [raw])
            return start, count
        if 0x06 not in spec.functions:
            raise ProtocolError(f"register {spec.name} does not support 0x06 writes")
        return self.write_single_register(spec.address, raw)

    def read_field(self, field: str | ValueField) -> int | float:
        spec = get_field(field)
        values = self.read_holding_registers(spec.address, spec.words)
        return _decode_field_value(spec, values)

    def write_field(self, field: str | ValueField, value: int | float) -> tuple[int, int]:
        spec = get_field(field)
        if 0x10 not in spec.functions:
            raise ProtocolError(f"field {spec.name} does not support 0x10 writes")
        values = _encode_field_value(spec, value)
        return self.write_multiple_registers(spec.address, values)

    def read_device_info(self) -> dict[str, int | float | tuple[int, int]]:
        device_id, version, max_current = self.read_holding_registers(0x0000, 3)
        return {
            "device_id": device_id,
            "version": ((version >> 8) & 0xFF, version & 0xFF),
            "max_current_a": max_current * 0.01,
        }

    def read_motor_status(self, side: str | MotorSide) -> MotorStatus:
        motor = _side(side)
        start = 0x0008 if motor is MotorSide.RIGHT else 0x0012
        values = self.read_holding_registers(start, 10)
        speed = registers_to_int(values[1:3], signed=True)
        position = registers_to_int(values[7:9], signed=True)
        temperature = _to_signed_u16(values[5])
        error_code = values[3]
        phase_current = round(abs(_to_signed_u16(values[0])) * 0.01, 2)
        pwm_percent = round(abs(_to_signed_u16(values[4])) * 0.1, 1)
        return MotorStatus(
            side=motor,
            phase_current_a=phase_current,
            speed_rpm=speed,
            error_code=error_code,
            error_name=ERROR_STATES.get(error_code, "未知错误"),
            pwm_percent=pwm_percent,
            board_temperature_c=temperature,
            position_degree=position,
            reverse=bool(values[9]),
        )

    def read_bus_status(self) -> dict[str, int]:
        supply_voltage, runtime_mode, control_mode, in1, in2 = self.read_holding_registers(0x001C, 5)
        return {
            "supply_voltage_v": supply_voltage,
            "runtime_system_mode": runtime_mode,
            "control_mode": control_mode,
            "in1_level": in1,
            "in2_level": in2,
        }

    def read_realtime_snapshot(self) -> RealtimeSnapshot:
        """Read the protocol-supported values used by the realtime status view."""

        return RealtimeSnapshot(
            device_info=self.read_device_info(),
            bus_status=self.read_bus_status(),
            left_motor=self.read_motor_status(MotorSide.LEFT),
            right_motor=self.read_motor_status(MotorSide.RIGHT),
        )

    def set_speed(self, side: str | MotorSide, rpm: int) -> tuple[int, int]:
        motor = _side(side)
        field = "right_speed_target" if motor is MotorSide.RIGHT else "left_speed_target"
        return self.write_field(field, rpm)

    def set_right_speed(self, rpm: int) -> tuple[int, int]:
        return self.set_speed(MotorSide.RIGHT, rpm)

    def set_left_speed(self, rpm: int) -> tuple[int, int]:
        return self.set_speed(MotorSide.LEFT, rpm)

    def set_differential_speed(self, forward_rpm: int, turn_rpm: int) -> tuple[tuple[int, int], tuple[int, int]]:
        first = self.write_field("right_speed_target", forward_rpm)
        second = self.write_field("left_speed_target", turn_rpm)
        return first, second

    def stop(self, side: str | MotorSide, mode: StopMode | int = StopMode.NORMAL) -> tuple[int, int]:
        motor = _side(side)
        register = "right_stop" if motor is MotorSide.RIGHT else "left_stop"
        return self.write_register(register, int(mode))

    def stop_all(self, mode: StopMode | int = StopMode.NORMAL) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.stop(MotorSide.RIGHT, mode), self.stop(MotorSide.LEFT, mode)

    def set_brake(self, side: str | MotorSide, enabled: bool) -> tuple[int, int]:
        motor = _side(side)
        register = "right_brake_control" if motor is MotorSide.RIGHT else "left_brake_control"
        return self.write_register(register, 1 if enabled else 0)

    def restore_factory(self) -> tuple[int, int]:
        return self.write_register("factory_reset_command", 1)

    def start_phase_learning(self, side: str | MotorSide) -> tuple[int, int]:
        motor = _side(side)
        return self.write_register("test_control", 2 if motor is MotorSide.RIGHT else 1)

    def cancel_test(self) -> tuple[int, int]:
        return self.write_register("test_control", 0)

    def set_system_mode(self, mode: SystemMode | int, *, persist: bool = True) -> tuple[int, int]:
        return self.write_register("system_mode", int(mode), persist=persist)

    def set_rs485_address(self, address: int, *, persist: bool = True) -> tuple[int, int]:
        if not 0 <= address <= 255:
            raise ValueError("RS485 address must be between 0 and 255")
        return self.write_register("rs485_address", address, persist=persist)

    def set_rs485_baud_rate(self, baudrate: int, *, persist: bool = True) -> tuple[int, int]:
        try:
            code = BAUD_RATE_CODES[baudrate]
        except KeyError as exc:
            raise ValueError("baudrate must be one of 9600, 19200, 38400, 57600, 115200") from exc
        return self.write_register("rs485_baud_rate_code", code, persist=persist)

    def set_rs485_parity(self, parity: str, *, persist: bool = True) -> tuple[int, int]:
        code = _parity_code(parity)
        return self.write_register("rs485_parity_code", code, persist=persist)

    def set_ttl_address(self, address: int, *, persist: bool = True) -> tuple[int, int]:
        if not 0 <= address <= 255:
            raise ValueError("TTL address must be between 0 and 255")
        return self.write_register("ttl_address", address, persist=persist)

    def set_ttl_baud_rate(self, baudrate: int, *, persist: bool = True) -> tuple[int, int]:
        try:
            code = BAUD_RATE_CODES[baudrate]
        except KeyError as exc:
            raise ValueError("baudrate must be one of 9600, 19200, 38400, 57600, 115200") from exc
        return self.write_register("ttl_baud_rate_code", code, persist=persist)

    def set_ttl_parity(self, parity: str, *, persist: bool = True) -> tuple[int, int]:
        code = _parity_code(parity)
        return self.write_register("ttl_parity_code", code, persist=persist)

    def set_can_node_id(self, node_id: int, *, persist: bool = True) -> tuple[int, int]:
        if not 0 <= node_id <= 127:
            raise ValueError("CAN node id must be between 0 and 127")
        return self.write_register("can_node_id", node_id, persist=persist)

    def set_can_baud_rate(self, baudrate: int, *, persist: bool = True) -> tuple[int, int]:
        try:
            code = CAN_BAUD_RATE_CODES[baudrate]
        except KeyError as exc:
            raise ValueError("unsupported CAN baudrate") from exc
        return self.write_register("can_baud_rate_code", code, persist=persist)

    def set_pid(self, name: str, actual_value: float) -> tuple[int, int]:
        return self.write_field(name, actual_value)


def _parity_code(value: str) -> int:
    normalized = value.upper().replace("+", "").replace(" ", "")
    if normalized in {"NONE1", "N"}:
        normalized = "N1"
    elif normalized in {"NONE2"}:
        normalized = "N2"
    elif normalized in {"EVEN1", "E"}:
        normalized = "E1"
    elif normalized in {"ODD1", "O"}:
        normalized = "O1"
    try:
        return PARITY_CODES[normalized]
    except KeyError as exc:
        raise ValueError("parity must be one of N1, E1, O1, N2") from exc
