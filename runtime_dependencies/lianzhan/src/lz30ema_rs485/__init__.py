"""LZ-30EMA_2EC_N RS485 Modbus RTU client library."""

from .client import (
    LZ30EMAClient,
    MotorSide,
    MotorStatus,
    RealtimeSnapshot,
    StopMode,
    SystemMode,
)
from .exceptions import CRCError, LZ30EMAError, ModbusException, ProtocolError, ResponseMismatch
from .protocol import (
    FunctionCode,
    append_crc,
    build_read_holding_request,
    build_write_multiple_request,
    build_write_single_request,
    crc16,
)
from .registers import FIELD_BY_NAME, REGISTER_BY_ADDRESS, REGISTER_BY_NAME, RESERVED_RANGES
from .status import format_realtime_snapshot, protocol_field_coverage_text

__all__ = [
    "CRCError",
    "FIELD_BY_NAME",
    "FunctionCode",
    "LZ30EMAClient",
    "LZ30EMAError",
    "ModbusException",
    "MotorSide",
    "MotorStatus",
    "ProtocolError",
    "REGISTER_BY_ADDRESS",
    "REGISTER_BY_NAME",
    "RESERVED_RANGES",
    "RealtimeSnapshot",
    "ResponseMismatch",
    "StopMode",
    "SystemMode",
    "append_crc",
    "build_read_holding_request",
    "build_write_multiple_request",
    "build_write_single_request",
    "crc16",
    "format_realtime_snapshot",
    "protocol_field_coverage_text",
]
