"""Exceptions raised by the LZ-30EMA RS485 library."""


class LZ30EMAError(Exception):
    """Base exception for this package."""


class CRCError(LZ30EMAError):
    """A Modbus RTU frame failed CRC validation."""


class ProtocolError(LZ30EMAError):
    """A frame is malformed or violates the expected Modbus protocol."""


class ResponseMismatch(ProtocolError):
    """A response does not match the request that was sent."""


class ModbusException(ProtocolError):
    """The driver returned a Modbus exception response."""

    def __init__(self, function_code: int, exception_code: int, message: str) -> None:
        self.function_code = function_code
        self.exception_code = exception_code
        super().__init__(message)
