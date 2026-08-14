from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import serial


class ModbusException(Exception):
    pass


class ModbusCRCError(ModbusException):
    pass


class ModbusResponseError(ModbusException):
    pass


class ModbusDeviceException(ModbusException):
    EXCEPTION_CODES = {
        0x01: "非法功能码",
        0x02: "非法数据地址",
        0x03: "非法数据值",
        0x04: "从站设备故障",
        0x05: "请求已被确认，但需要较长时间处理",
        0x06: "从设备忙",
        0x08: "存储奇偶性差错",
        0x0A: "不可用的网关",
        0x0B: "网关目标设备响应失败",
    }

    def __init__(self, function_code: int, exception_code: int) -> None:
        self.function_code = function_code
        self.exception_code = exception_code
        message = self.EXCEPTION_CODES.get(exception_code, f"未知异常码 0x{exception_code:02X}")
        super().__init__(f"设备返回异常: 功能码=0x{function_code:02X}, 异常码=0x{exception_code:02X} ({message})")


class Register(IntEnum):
    DEVICE_ID = 0x0000
    DEVICE_VERSION = 0x0001
    DEVICE_MAX_CURRENT = 0x0002

    RIGHT_BUS_CURRENT = 0x0007
    RIGHT_PHASE_CURRENT = 0x0008
    RIGHT_SPEED_H = 0x0009
    RIGHT_SPEED_L = 0x000A
    RIGHT_ERROR = 0x000B
    RIGHT_SIGNAL_VOLTAGE = 0x000C
    RIGHT_LOAD_RATE = 0x000D
    RIGHT_PPM_PULSE_WIDTH = 0x000E
    RIGHT_TEMPERATURE = 0x000F
    RIGHT_COMMUTATION_COUNT_H = 0x0010
    RIGHT_COMMUTATION_COUNT_L = 0x0011

    LEFT_BUS_CURRENT = 0x0012
    LEFT_PHASE_CURRENT = 0x0013
    LEFT_SPEED_H = 0x0014
    LEFT_SPEED_L = 0x0015
    LEFT_ERROR = 0x0016
    LEFT_SIGNAL_VOLTAGE = 0x0017
    LEFT_LOAD_RATE = 0x0018
    LEFT_PPM_PULSE_WIDTH = 0x0019
    LEFT_TEMPERATURE = 0x001A
    LEFT_COMMUTATION_COUNT_H = 0x001B
    LEFT_COMMUTATION_COUNT_L = 0x001C

    SUPPLY_VOLTAGE = 0x001D
    RUNTIME_SYSTEM_MODE = 0x001E
    CONTROL_MODE_WORD = 0x001F
    IN1_LEVEL = 0x0020
    IN2_LEVEL = 0x0021
    IN3_LEVEL = 0x0022
    IN4_LEVEL = 0x0023
    IN5_LEVEL = 0x0024
    IN6_LEVEL = 0x0025
    IN7_LEVEL = 0x0026
    IN8_LEVEL = 0x0027
    RIGHT_RECOVERY_PROTECT = 0x002A
    LEFT_RECOVERY_PROTECT = 0x002B

    STOP_1 = 0x0038
    TARGET_1_H = 0x0039
    TARGET_1_L = 0x003A
    RIGHT_BRAKE_CONTROL = 0x003B
    STOP_2 = 0x003C
    TARGET_2_H = 0x003D
    TARGET_2_L = 0x003E
    LEFT_BRAKE_CONTROL = 0x003F
    RESTORE_FACTORY = 0x0040
    TEST_CONTROL = 0x0041
    OUT1_CONTROL = 0x0042
    OUT2_CONTROL = 0x0043

    MAX_OUTPUT_CURRENT = 0x0050
    MAX_FORWARD_SPEED_H = 0x0051
    MAX_FORWARD_SPEED_L = 0x0052
    MAX_REVERSE_SPEED_H = 0x0053
    MAX_REVERSE_SPEED_L = 0x0054
    MIN_SPEED_H = 0x0055
    MIN_SPEED_L = 0x0056
    CLOSED_LOOP_ACCEL = 0x0057
    CLOSED_LOOP_DECEL = 0x0058
    OPEN_LOOP_ACCEL = 0x0059
    OPEN_LOOP_DECEL = 0x005A
    DISABLE_BRAKE_MODE = 0x005B
    AUTO_REVERSAL_MODE = 0x005C
    HALL_ELECTRIC_ANGLE = 0x005D
    RIGHT_POLE_PAIRS = 0x005E
    RIGHT_PHASE_12 = 0x005F
    RIGHT_PHASE_34 = 0x0060
    RIGHT_PHASE_56 = 0x0061
    RIGHT_PHASE_LEARNED = 0x0062
    RIGHT_PARKING_MAX_CURRENT = 0x0063
    RIGHT_FOC_PHASE_COMP = 0x0064
    RIGHT_FOC_ADVANCE_ANGLE = 0x0065
    LEFT_POLE_PAIRS = 0x0066
    LEFT_PHASE_12 = 0x0067
    LEFT_PHASE_34 = 0x0068
    LEFT_PHASE_56 = 0x0069
    LEFT_PHASE_LEARNED = 0x006A
    LEFT_PARKING_MAX_CURRENT = 0x006B
    LEFT_FOC_PHASE_COMP = 0x006C
    LEFT_FOC_ADVANCE_ANGLE = 0x006D

    OVER_VOLTAGE = 0x0078
    UNDER_VOLTAGE = 0x0079
    VOLTAGE_PROTECT_ENABLE = 0x007A
    SYSTEM_MODE = 0x007B
    CONTROL_SIGNAL = 0x007C
    DRIVER_MODE = 0x007D
    FOC_CLOSED_LOOP_MODE = 0x007E
    DIFF_TURN_MAX_SPEED = 0x007F
    STALL_STOP_TIME = 0x0080
    PHASE_OVERCURRENT_CUTOFF = 0x0081
    BUS_PROTECTION_CUTOFF = 0x0082
    ENERGY_RECOVERY_PROTECT_VOLTAGE = 0x0083
    ENERGY_RECOVERY_PROTECT_MODE = 0x0084

    IN1_TYPE = 0x0091
    IN2_TYPE = 0x0092
    IN3_TYPE = 0x0093
    IN4_TYPE = 0x0094
    IN5_TYPE = 0x0095
    IN6_TYPE = 0x0096
    IN7_TYPE = 0x0097
    IN8_TYPE = 0x0098
    OUT1_TYPE = 0x009B
    OUT2_TYPE = 0x009C

    RIGHT_ANALOG_MAX = 0x00A8
    RIGHT_ANALOG_MIN = 0x00A9
    RIGHT_ANALOG_DEADZONE = 0x00AA
    RIGHT_BRAKE_OUTPUT_TYPE = 0x00AB
    RIGHT_STOP_BRAKE_ENABLE = 0x00AC
    LEFT_ANALOG_MAX = 0x00B0
    LEFT_ANALOG_MIN = 0x00B1
    LEFT_ANALOG_DEADZONE = 0x00B2
    LEFT_BRAKE_OUTPUT_TYPE = 0x00B3
    LEFT_STOP_BRAKE_ENABLE = 0x00B4

    RS485_DEVICE_ADDRESS = 0x00C0
    RS485_BAUDRATE = 0x00C1
    RS485_PARITY = 0x00C2
    RS485_BRAKE_TIMEOUT = 0x00C3
    CAN_NODE_ID = 0x00C5
    CAN_BAUDRATE = 0x00C6
    CAN_BRAKE_TIMEOUT = 0x00C7

    SPEED_PID_P_H = 0x00D0
    SPEED_PID_P_L = 0x00D1
    SPEED_PID_I_H = 0x00D2
    SPEED_PID_I_L = 0x00D3
    SPEED_PID_D_H = 0x00D4
    SPEED_PID_D_L = 0x00D5
    D_AXIS_PID_P_H = 0x00D6
    D_AXIS_PID_P_L = 0x00D7
    D_AXIS_PID_I_H = 0x00D8
    D_AXIS_PID_I_L = 0x00D9
    D_AXIS_PID_D_H = 0x00DA
    D_AXIS_PID_D_L = 0x00DB
    Q_AXIS_PID_P_H = 0x00DC
    Q_AXIS_PID_P_L = 0x00DD
    Q_AXIS_PID_I_H = 0x00DE
    Q_AXIS_PID_I_L = 0x00DF
    Q_AXIS_PID_D_H = 0x00E0
    Q_AXIS_PID_D_L = 0x00E1
    PARK_SPEED_PID_P_H = 0x00E8
    PARK_SPEED_PID_P_L = 0x00E9
    PARK_SPEED_PID_I_H = 0x00EA
    PARK_SPEED_PID_I_L = 0x00EB
    PARK_SPEED_PID_D_H = 0x00EC
    PARK_SPEED_PID_D_L = 0x00ED
    PARK_D_AXIS_PID_P_H = 0x00EE
    PARK_D_AXIS_PID_P_L = 0x00EF
    PARK_D_AXIS_PID_I_H = 0x00F0
    PARK_D_AXIS_PID_I_L = 0x00F1
    PARK_D_AXIS_PID_D_H = 0x00F2
    PARK_D_AXIS_PID_D_L = 0x00F3
    PARK_Q_AXIS_PID_P_H = 0x00F4
    PARK_Q_AXIS_PID_P_L = 0x00F5
    PARK_Q_AXIS_PID_I_H = 0x00F6
    PARK_Q_AXIS_PID_I_L = 0x00F7
    PARK_Q_AXIS_PID_D_H = 0x00F8
    PARK_Q_AXIS_PID_D_L = 0x00F9


class StopMode(IntEnum):
    NORMAL = 0
    EMERGENCY = 1
    FREE = 2


class MotorChannel(str, Enum):
    RIGHT = "right"
    LEFT = "left"
    BOTH = "both"


class SystemMode(IntEnum):
    INDEPENDENT_OPEN_LOOP = 0
    INDEPENDENT_CLOSED_LOOP = 1
    SAME_SOURCE_OPEN_LOOP = 2
    SAME_SOURCE_CLOSED_LOOP = 3
    DIFFERENTIAL_CLOSED_LOOP = 4


class ControlSignal(IntEnum):
    RS485 = 0
    MODEL_PWM = 1
    JOYSTICK = 2


class DriverMode(IntEnum):
    SQUARE_WAVE = 0
    FOC = 1


class FocClosedLoopMode(IntEnum):
    SPEED_ONLY = 0
    SPEED_CURRENT_DUAL_LOOP = 1


class OutputLevel(IntEnum):
    LOW = 0
    HIGH = 1


class TestControl(IntEnum):
    CANCEL = 0
    LEARN_LEFT_PHASE = 1
    LEARN_RIGHT_PHASE = 2


class InputType(IntEnum):
    NONE = 0
    ENABLE = 1
    EMERGENCY_STOP = 2
    RESET = 3
    PARK_ENABLE = 4
    LEFT_FORWARD_LIMIT = 5
    LEFT_REVERSE_LIMIT = 6
    RIGHT_FORWARD_LIMIT = 7
    RIGHT_REVERSE_LIMIT = 8
    MODEL_PWM_CONTROL = 9
    JOYSTICK_CONTROL = 10


class OutputType(IntEnum):
    DISABLED = 0
    ALWAYS_ON = 1
    ALARM = 2
    FG_FEEDBACK = 3
    STOP_SIGNAL = 4
    RS485_CONTROL = 5


class BrakeOutputType(IntEnum):
    DISABLED = 0
    AUTO_BRAKE_WHEN_STOP = 1
    RS485_CONTROL = 2


class RS485Baudrate(IntEnum):
    B9600 = 0
    B19200 = 1
    B38400 = 2
    B57600 = 3
    B115200 = 4


class RS485Parity(IntEnum):
    NONE_1_STOP = 0
    EVEN_1_STOP = 1
    ODD_1_STOP = 2
    NONE_2_STOP = 3


ERROR_CODES = {
    0: "无错误",
    1: "堵转停机",
    2: "相序未学习",
    3: "相位异常",
    4: "相线过流",
    5: "母线过流",
    6: "电压过高",
    7: "电压过低",
    8: "相位学习异常",
    9: "过载异常",
    10: "485 通信中断",
    11: "CAN 通信中断",
    12: "母线电压过高",
    13: "温度过高",
    14: "无感启动失败",
    15: "相电流异常",
    16: "上电大电流异常",
    17: "同步超差异常",
    18: "硬件异常",
}


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def append_crc(frame: bytes) -> bytes:
    crc = crc16_modbus(frame)
    return frame + struct.pack("<H", crc)


def _coerce_stop_mode(mode: Union[StopMode, int, str]) -> StopMode:
    if isinstance(mode, StopMode):
        return mode
    if isinstance(mode, int):
        return StopMode(mode)
    text = str(mode).strip().lower()
    aliases = {
        "0": StopMode.NORMAL,
        "normal": StopMode.NORMAL,
        "normal_stop": StopMode.NORMAL,
        "normal-stop": StopMode.NORMAL,
        "正常": StopMode.NORMAL,
        "正常停止": StopMode.NORMAL,
        "1": StopMode.EMERGENCY,
        "emergency": StopMode.EMERGENCY,
        "emergency_stop": StopMode.EMERGENCY,
        "emergency-stop": StopMode.EMERGENCY,
        "紧急": StopMode.EMERGENCY,
        "紧急停止": StopMode.EMERGENCY,
        "急停": StopMode.EMERGENCY,
        "2": StopMode.FREE,
        "free": StopMode.FREE,
        "free_stop": StopMode.FREE,
        "free-stop": StopMode.FREE,
        "自由": StopMode.FREE,
        "自由停止": StopMode.FREE,
        "自由停": StopMode.FREE,
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(f"未知停机模式: {mode}")


def _coerce_motor_channel(channel: Union[MotorChannel, str]) -> MotorChannel:
    if isinstance(channel, MotorChannel):
        return channel
    text = str(channel).strip().lower()
    aliases = {
        "right": MotorChannel.RIGHT,
        "r": MotorChannel.RIGHT,
        "右": MotorChannel.RIGHT,
        "右电机": MotorChannel.RIGHT,
        "left": MotorChannel.LEFT,
        "l": MotorChannel.LEFT,
        "左": MotorChannel.LEFT,
        "左电机": MotorChannel.LEFT,
        "both": MotorChannel.BOTH,
        "all": MotorChannel.BOTH,
        "双": MotorChannel.BOTH,
        "双电机": MotorChannel.BOTH,
        "全部": MotorChannel.BOTH,
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(f"未知电机通道: {channel}")


def _to_u16(value: int) -> int:
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"16位无符号值越界: {value}")
    return value


def _to_i16(value: int) -> int:
    if not -0x8000 <= value <= 0x7FFF:
        raise ValueError(f"16位有符号值越界: {value}")
    return value & 0xFFFF


def _to_u32_words(value: int) -> List[int]:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"32位无符号值越界: {value}")
    return [(value >> 16) & 0xFFFF, value & 0xFFFF]


def _to_i32_words(value: int) -> List[int]:
    if not -0x80000000 <= value <= 0x7FFFFFFF:
        raise ValueError(f"32位有符号值越界: {value}")
    value &= 0xFFFFFFFF
    return [(value >> 16) & 0xFFFF, value & 0xFFFF]


def _u32_from_words(high_word: int, low_word: int) -> int:
    return ((high_word & 0xFFFF) << 16) | (low_word & 0xFFFF)


def _i32_from_words(high_word: int, low_word: int) -> int:
    value = _u32_from_words(high_word, low_word)
    if value & 0x80000000:
        value -= 0x100000000
    return value


@dataclass(frozen=True)
class FieldDef:
    getter: Callable[["MSSD60EHB"], Union[int, float]]
    setter: Optional[Callable[["MSSD60EHB", Union[int, float]], None]] = None


class MSSD60EHB:
    FUNCTION_READ = 0x03
    FUNCTION_WRITE_SINGLE = 0x06
    FUNCTION_WRITE_MULTI = 0x10

    def __init__(
        self,
        port: str,
        slave_id: int = 1,
        baudrate: int = 9600,
        bytesize: int = serial.EIGHTBITS,
        parity: str = serial.PARITY_NONE,
        stopbits: float = serial.STOPBITS_ONE,
        timeout: float = 0.2,
        write_timeout: float = 0.2,
        serial_port: Optional[serial.Serial] = None,
    ) -> None:
        if not 0 <= slave_id <= 255:
            raise ValueError("slave_id 必须在 0~255 之间")
        self.slave_id = slave_id
        self.serial = serial_port or serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
            timeout=timeout,
            write_timeout=write_timeout,
        )

    def close(self) -> None:
        self.serial.close()

    def __enter__(self) -> "MSSD60EHB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _send_request(self, payload: bytes, expected_len: Optional[int] = None) -> bytes:
        frame = append_crc(payload)
        self.serial.reset_input_buffer()
        self.serial.write(frame)
        self.serial.flush()

        if self.slave_id == 0:
            return b""

        if expected_len is None:
            header = self.serial.read(3)
            if len(header) != 3:
                raise ModbusResponseError(f"响应长度不足，收到 {header.hex(' ')}")
            if header[1] & 0x80:
                tail = self.serial.read(2)
                response = header + tail
            elif header[1] == self.FUNCTION_READ:
                response = header + self.serial.read(header[2] + 2)
            else:
                response = header + self.serial.read(5)
        else:
            response = self.serial.read(expected_len)

        if len(response) < 5:
            raise ModbusResponseError(f"响应过短: {response.hex(' ')}")
        self._validate_response(response)
        return response

    def _validate_response(self, response: bytes) -> None:
        body = response[:-2]
        recv_crc = struct.unpack("<H", response[-2:])[0]
        calc_crc = crc16_modbus(body)
        if recv_crc != calc_crc:
            raise ModbusCRCError(
                f"CRC 校验失败: recv=0x{recv_crc:04X}, calc=0x{calc_crc:04X}, raw={response.hex(' ')}"
            )
        if response[0] != self.slave_id:
            raise ModbusResponseError(f"从站地址不匹配: got={response[0]}, expected={self.slave_id}")
        if response[1] & 0x80:
            raise ModbusDeviceException(response[1] & 0x7F, response[2])

    def read_holding_registers(self, start_address: int, quantity: int) -> List[int]:
        if not 1 <= quantity <= 125:
            raise ValueError("quantity 必须在 1~125 之间")
        payload = struct.pack(">BBHH", self.slave_id, self.FUNCTION_READ, start_address, quantity)
        response = self._send_request(payload)
        byte_count = response[2]
        if byte_count != quantity * 2:
            raise ModbusResponseError(f"返回字节数异常: {byte_count}, expected={quantity * 2}")
        return list(struct.unpack(f">{quantity}H", response[3 : 3 + byte_count]))

    def write_single_register(self, address: int, value: int, broadcast: bool = False) -> None:
        slave = 0 if broadcast else self.slave_id
        payload = struct.pack(">BBHH", slave, self.FUNCTION_WRITE_SINGLE, address, _to_u16(value))
        response = self._send_request(payload, expected_len=0 if broadcast else 8)
        if not broadcast and response[:6] != payload[:6]:
            raise ModbusResponseError(f"写单寄存器回显不一致: {response.hex(' ')}")

    def build_write_single_register_request(self, address: Union[int, Register], value: int, broadcast: bool = False) -> bytes:
        slave = 0 if broadcast else self.slave_id
        payload = struct.pack(">BBHH", slave, self.FUNCTION_WRITE_SINGLE, int(address), _to_u16(value))
        return append_crc(payload)

    def write_multiple_registers(self, start_address: int, values: Sequence[int], broadcast: bool = False) -> None:
        if not values:
            raise ValueError("values 不能为空")
        if len(values) > 123:
            raise ValueError("一次最多写 123 个寄存器")
        slave = 0 if broadcast else self.slave_id
        clean = [_to_u16(v) for v in values]
        data = struct.pack(f">{len(clean)}H", *clean)
        payload = struct.pack(">BBHHB", slave, self.FUNCTION_WRITE_MULTI, start_address, len(clean), len(data)) + data
        response = self._send_request(payload, expected_len=0 if broadcast else 8)
        if not broadcast:
            expected_prefix = struct.pack(">BBHH", self.slave_id, self.FUNCTION_WRITE_MULTI, start_address, len(clean))
            if response[:6] != expected_prefix:
                raise ModbusResponseError(f"写多寄存器回显不一致: {response.hex(' ')}")

    def read_u16(self, address: Union[int, Register]) -> int:
        return self.read_holding_registers(int(address), 1)[0]

    def read_i16(self, address: Union[int, Register]) -> int:
        value = self.read_u16(address)
        return value - 0x10000 if value & 0x8000 else value

    def write_u16(self, address: Union[int, Register], value: int, broadcast: bool = False) -> None:
        self.write_single_register(int(address), _to_u16(value), broadcast=broadcast)

    def write_i16(self, address: Union[int, Register], value: int, broadcast: bool = False) -> None:
        self.write_single_register(int(address), _to_i16(value), broadcast=broadcast)

    def read_u32(self, high_address: Union[int, Register]) -> int:
        high_word, low_word = self.read_holding_registers(int(high_address), 2)
        return _u32_from_words(high_word, low_word)

    def read_i32(self, high_address: Union[int, Register]) -> int:
        high_word, low_word = self.read_holding_registers(int(high_address), 2)
        return _i32_from_words(high_word, low_word)

    def write_u32(self, high_address: Union[int, Register], value: int, broadcast: bool = False) -> None:
        self.write_multiple_registers(int(high_address), _to_u32_words(value), broadcast=broadcast)

    def write_i32(self, high_address: Union[int, Register], value: int, broadcast: bool = False) -> None:
        self.write_multiple_registers(int(high_address), _to_i32_words(value), broadcast=broadcast)

    def read_named(self, name: str) -> Union[int, float]:
        field = FIELD_DEFS[name]
        return field.getter(self)

    def write_named(self, name: str, value: Union[int, float]) -> None:
        field = FIELD_DEFS[name]
        if field.setter is None:
            raise AttributeError(f"{name} 是只读字段")
        field.setter(self, value)

    def get_error_text(self, code: int) -> str:
        return ERROR_CODES.get(code, f"未知错误 {code}")

    def get_device_info(self) -> Dict[str, Union[int, float, str]]:
        version = self.read_u16(Register.DEVICE_VERSION)
        return {
            "device_id": self.read_u16(Register.DEVICE_ID),
            "device_version_raw": version,
            "device_version": f"V{(version >> 8) & 0xFF}.{version & 0xFF}",
            "device_max_current_amps": self.read_u16(Register.DEVICE_MAX_CURRENT) * 0.01,
        }

    def get_runtime_status(self) -> Dict[str, Union[int, float, str]]:
        return {
            "right_phase_current_amps": self.get_right_phase_current_amps(),
            "right_speed_rpm": self.get_right_speed_rpm(),
            "right_error_code": self.get_right_error_code(),
            "right_error_text": self.get_error_text(self.get_right_error_code()),
            "left_phase_current_amps": self.get_left_phase_current_amps(),
            "left_speed_rpm": self.get_left_speed_rpm(),
            "left_error_code": self.get_left_error_code(),
            "left_error_text": self.get_error_text(self.get_left_error_code()),
            "system_mode": self.read_u16(Register.RUNTIME_SYSTEM_MODE),
            "control_mode_word": self.read_u16(Register.CONTROL_MODE_WORD),
            "supply_voltage_v": self.read_u16(Register.SUPPLY_VOLTAGE),
        }

    def build_stop_request(
        self,
        channel: Union[MotorChannel, str],
        mode: Union[StopMode, int, str] = StopMode.NORMAL,
        broadcast: bool = False,
    ) -> List[bytes]:
        stop_mode = _coerce_stop_mode(mode)
        motor = _coerce_motor_channel(channel)
        if motor == MotorChannel.RIGHT:
            return [self.build_write_single_register_request(Register.STOP_1, int(stop_mode), broadcast=broadcast)]
        if motor == MotorChannel.LEFT:
            return [self.build_write_single_register_request(Register.STOP_2, int(stop_mode), broadcast=broadcast)]
        return [
            self.build_write_single_register_request(Register.STOP_1, int(stop_mode), broadcast=broadcast),
            self.build_write_single_register_request(Register.STOP_2, int(stop_mode), broadcast=broadcast),
        ]

    def build_brake_request(self, channel: Union[MotorChannel, str], enabled: bool, broadcast: bool = False) -> List[bytes]:
        motor = _coerce_motor_channel(channel)
        value = 1 if enabled else 0
        if motor == MotorChannel.RIGHT:
            return [self.build_write_single_register_request(Register.RIGHT_BRAKE_CONTROL, value, broadcast=broadcast)]
        if motor == MotorChannel.LEFT:
            return [self.build_write_single_register_request(Register.LEFT_BRAKE_CONTROL, value, broadcast=broadcast)]
        return [
            self.build_write_single_register_request(Register.RIGHT_BRAKE_CONTROL, value, broadcast=broadcast),
            self.build_write_single_register_request(Register.LEFT_BRAKE_CONTROL, value, broadcast=broadcast),
        ]

    def build_parking_current_request(
        self,
        channel: Union[MotorChannel, str],
        current_amps: float,
        broadcast: bool = False,
    ) -> List[bytes]:
        motor = _coerce_motor_channel(channel)
        raw = int(round(float(current_amps) * 100.0))
        value = _to_u16(raw)
        if motor == MotorChannel.RIGHT:
            return [self.build_write_single_register_request(Register.RIGHT_PARKING_MAX_CURRENT, value, broadcast=broadcast)]
        if motor == MotorChannel.LEFT:
            return [self.build_write_single_register_request(Register.LEFT_PARKING_MAX_CURRENT, value, broadcast=broadcast)]
        return [
            self.build_write_single_register_request(Register.RIGHT_PARKING_MAX_CURRENT, value, broadcast=broadcast),
            self.build_write_single_register_request(Register.LEFT_PARKING_MAX_CURRENT, value, broadcast=broadcast),
        ]

    def stop_motor(self, channel: Union[MotorChannel, str], mode: Union[StopMode, int, str] = StopMode.NORMAL) -> None:
        motor = _coerce_motor_channel(channel)
        stop_mode = _coerce_stop_mode(mode)
        if motor == MotorChannel.RIGHT:
            self.write_u16(Register.STOP_1, int(stop_mode))
            return
        if motor == MotorChannel.LEFT:
            self.write_u16(Register.STOP_2, int(stop_mode))
            return
        self.stop_both_motors(stop_mode)

    def stop_right_motor(self, mode: Union[StopMode, int, str] = StopMode.NORMAL) -> None:
        self.write_u16(Register.STOP_1, int(_coerce_stop_mode(mode)))

    def stop_left_motor(self, mode: Union[StopMode, int, str] = StopMode.NORMAL) -> None:
        self.write_u16(Register.STOP_2, int(_coerce_stop_mode(mode)))

    def stop_both_motors(self, mode: Union[StopMode, int, str] = StopMode.NORMAL) -> None:
        stop_mode = _coerce_stop_mode(mode)
        self.stop_right_motor(stop_mode)
        self.stop_left_motor(stop_mode)

    def normal_stop_right_motor(self) -> None:
        self.stop_right_motor(StopMode.NORMAL)

    def normal_stop_left_motor(self) -> None:
        self.stop_left_motor(StopMode.NORMAL)

    def normal_stop_both_motors(self) -> None:
        self.stop_both_motors(StopMode.NORMAL)

    def emergency_stop_right_motor(self) -> None:
        self.stop_right_motor(StopMode.EMERGENCY)

    def emergency_stop_left_motor(self) -> None:
        self.stop_left_motor(StopMode.EMERGENCY)

    def emergency_stop_both_motors(self) -> None:
        self.stop_both_motors(StopMode.EMERGENCY)

    def free_stop_right_motor(self) -> None:
        self.stop_right_motor(StopMode.FREE)

    def free_stop_left_motor(self) -> None:
        self.stop_left_motor(StopMode.FREE)

    def free_stop_both_motors(self) -> None:
        self.stop_both_motors(StopMode.FREE)

    def set_right_target(self, value: int) -> None:
        self.write_i32(Register.TARGET_1_H, value)

    def set_left_target(self, value: int) -> None:
        self.write_i32(Register.TARGET_2_H, value)

    def set_same_source_target(self, value: int) -> None:
        self.set_right_target(value)

    def set_differential_target(self, straight_speed: int, turn_speed: int) -> None:
        self.set_right_target(straight_speed)
        self.set_left_target(turn_speed)

    def set_right_brake(self, enabled: bool) -> None:
        self.write_u16(Register.RIGHT_BRAKE_CONTROL, 1 if enabled else 0)

    def set_left_brake(self, enabled: bool) -> None:
        self.write_u16(Register.LEFT_BRAKE_CONTROL, 1 if enabled else 0)

    def set_parking(
        self,
        channel: Union[MotorChannel, str],
        enabled: bool,
        stop_mode: Union[StopMode, int, str] = StopMode.NORMAL,
    ) -> None:
        motor = _coerce_motor_channel(channel)
        if enabled:
            self.stop_motor(motor, stop_mode)
        if motor == MotorChannel.RIGHT:
            self.set_right_brake(enabled)
            return
        if motor == MotorChannel.LEFT:
            self.set_left_brake(enabled)
            return
        self.set_right_brake(enabled)
        self.set_left_brake(enabled)

    def set_parking_current(self, channel: Union[MotorChannel, str], current_amps: float) -> None:
        motor = _coerce_motor_channel(channel)
        current = float(current_amps)
        if current < 0:
            raise ValueError("驻车电流不能为负数")
        if motor == MotorChannel.RIGHT:
            self.set_right_parking_max_current_amps(current)
            return
        if motor == MotorChannel.LEFT:
            self.set_left_parking_max_current_amps(current)
            return
        self.set_right_parking_max_current_amps(current)
        self.set_left_parking_max_current_amps(current)

    def enter_parking_mode(
        self,
        channel: Union[MotorChannel, str] = MotorChannel.BOTH,
        current_amps: float = 5.0,
        stop_mode: Union[StopMode, int, str] = StopMode.NORMAL,
    ) -> None:
        self.stop_motor(channel, stop_mode)
        self.set_parking_current(channel, current_amps)

    def exit_parking_mode(self, channel: Union[MotorChannel, str] = MotorChannel.BOTH) -> None:
        self.set_parking_current(channel, 0.0)

    def engage_parking(
        self,
        channel: Union[MotorChannel, str] = MotorChannel.BOTH,
        stop_mode: Union[StopMode, int, str] = StopMode.NORMAL,
    ) -> None:
        self.enter_parking_mode(channel, 5.0, stop_mode=stop_mode)

    def release_parking(self, channel: Union[MotorChannel, str] = MotorChannel.BOTH) -> None:
        self.exit_parking_mode(channel)

    def park_right_motor(self, stop_mode: StopMode = StopMode.NORMAL) -> None:
        self.stop_right_motor(stop_mode)
        self.set_right_brake(True)

    def park_left_motor(self, stop_mode: StopMode = StopMode.NORMAL) -> None:
        self.stop_left_motor(stop_mode)
        self.set_left_brake(True)

    def park_both_motors(self, stop_mode: StopMode = StopMode.NORMAL) -> None:
        self.stop_both_motors(stop_mode)
        self.set_right_brake(True)
        self.set_left_brake(True)

    def release_right_park(self) -> None:
        self.set_right_brake(False)

    def release_left_park(self) -> None:
        self.set_left_brake(False)

    def release_both_park(self) -> None:
        self.set_right_brake(False)
        self.set_left_brake(False)

    def set_out1_level(self, level: OutputLevel) -> None:
        self.write_u16(Register.OUT1_CONTROL, int(level))

    def set_out2_level(self, level: OutputLevel) -> None:
        self.write_u16(Register.OUT2_CONTROL, int(level))

    def restore_factory_settings(self) -> None:
        self.write_u16(Register.RESTORE_FACTORY, 1)

    def start_left_phase_learning(self) -> None:
        self.write_u16(Register.TEST_CONTROL, int(TestControl.LEARN_LEFT_PHASE))

    def start_right_phase_learning(self) -> None:
        self.write_u16(Register.TEST_CONTROL, int(TestControl.LEARN_RIGHT_PHASE))

    def cancel_test(self) -> None:
        self.write_u16(Register.TEST_CONTROL, int(TestControl.CANCEL))


def _read_scaled_u16(register: Register, scale: float = 1.0) -> Callable[[MSSD60EHB], Union[int, float]]:
    def getter(dev: MSSD60EHB) -> Union[int, float]:
        value = dev.read_u16(register)
        return value if scale == 1.0 else value * scale

    return getter


def _write_scaled_u16(register: Register, scale: float = 1.0) -> Callable[[MSSD60EHB, Union[int, float]], None]:
    def setter(dev: MSSD60EHB, value: Union[int, float]) -> None:
        raw = int(round(float(value) / scale)) if scale != 1.0 else int(value)
        dev.write_u16(register, raw)

    return setter


def _read_scaled_i16(register: Register, scale: float = 1.0) -> Callable[[MSSD60EHB], Union[int, float]]:
    def getter(dev: MSSD60EHB) -> Union[int, float]:
        value = dev.read_i16(register)
        return value if scale == 1.0 else value * scale

    return getter


def _write_scaled_i16(register: Register, scale: float = 1.0) -> Callable[[MSSD60EHB, Union[int, float]], None]:
    def setter(dev: MSSD60EHB, value: Union[int, float]) -> None:
        raw = int(round(float(value) / scale)) if scale != 1.0 else int(value)
        dev.write_i16(register, raw)

    return setter


def _read_u32_field(register: Register) -> Callable[[MSSD60EHB], int]:
    return lambda dev: dev.read_u32(register)


def _write_u32_field(register: Register) -> Callable[[MSSD60EHB, Union[int, float]], None]:
    return lambda dev, value: dev.write_u32(register, int(value))


def _read_i32_field(register: Register) -> Callable[[MSSD60EHB], int]:
    return lambda dev: dev.read_i32(register)


def _write_i32_field(register: Register) -> Callable[[MSSD60EHB, Union[int, float]], None]:
    return lambda dev, value: dev.write_i32(register, int(value))


FIELD_DEFS: Dict[str, FieldDef] = {
    "right_bus_current_amps": FieldDef(_read_scaled_u16(Register.RIGHT_BUS_CURRENT, 0.01)),
    "right_phase_current_amps": FieldDef(_read_scaled_u16(Register.RIGHT_PHASE_CURRENT, 0.01)),
    "right_speed_rpm": FieldDef(_read_i32_field(Register.RIGHT_SPEED_H)),
    "right_error_code": FieldDef(_read_scaled_u16(Register.RIGHT_ERROR)),
    "right_signal_voltage_mv": FieldDef(_read_scaled_u16(Register.RIGHT_SIGNAL_VOLTAGE)),
    "right_load_percent": FieldDef(_read_scaled_u16(Register.RIGHT_LOAD_RATE, 0.1)),
    "right_ppm_pulse_width_0p1us": FieldDef(_read_scaled_u16(Register.RIGHT_PPM_PULSE_WIDTH)),
    "right_temperature_c": FieldDef(_read_scaled_i16(Register.RIGHT_TEMPERATURE)),
    "right_commutation_count": FieldDef(_read_i32_field(Register.RIGHT_COMMUTATION_COUNT_H)),
    "left_bus_current_amps": FieldDef(_read_scaled_u16(Register.LEFT_BUS_CURRENT, 0.01)),
    "left_phase_current_amps": FieldDef(_read_scaled_u16(Register.LEFT_PHASE_CURRENT, 0.01)),
    "left_speed_rpm": FieldDef(_read_i32_field(Register.LEFT_SPEED_H)),
    "left_error_code": FieldDef(_read_scaled_u16(Register.LEFT_ERROR)),
    "left_signal_voltage_mv": FieldDef(_read_scaled_u16(Register.LEFT_SIGNAL_VOLTAGE)),
    "left_load_percent": FieldDef(_read_scaled_u16(Register.LEFT_LOAD_RATE, 0.1)),
    "left_ppm_pulse_width_0p1us": FieldDef(_read_scaled_u16(Register.LEFT_PPM_PULSE_WIDTH)),
    "left_temperature_c": FieldDef(_read_scaled_i16(Register.LEFT_TEMPERATURE)),
    "left_commutation_count": FieldDef(_read_i32_field(Register.LEFT_COMMUTATION_COUNT_H)),
    "supply_voltage_v": FieldDef(_read_scaled_u16(Register.SUPPLY_VOLTAGE)),
    "runtime_system_mode": FieldDef(_read_scaled_u16(Register.RUNTIME_SYSTEM_MODE)),
    "control_mode_word": FieldDef(_read_scaled_u16(Register.CONTROL_MODE_WORD)),
    "max_output_current_amps": FieldDef(
        _read_scaled_u16(Register.MAX_OUTPUT_CURRENT, 0.01),
        _write_scaled_u16(Register.MAX_OUTPUT_CURRENT, 0.01),
    ),
    "max_forward_speed_rpm": FieldDef(_read_u32_field(Register.MAX_FORWARD_SPEED_H), _write_u32_field(Register.MAX_FORWARD_SPEED_H)),
    "max_reverse_speed_rpm": FieldDef(_read_u32_field(Register.MAX_REVERSE_SPEED_H), _write_u32_field(Register.MAX_REVERSE_SPEED_H)),
    "min_speed_rpm": FieldDef(_read_u32_field(Register.MIN_SPEED_H), _write_u32_field(Register.MIN_SPEED_H)),
    "closed_loop_accel_rpm_s": FieldDef(_read_scaled_u16(Register.CLOSED_LOOP_ACCEL), _write_scaled_u16(Register.CLOSED_LOOP_ACCEL)),
    "closed_loop_decel_rpm_s": FieldDef(_read_scaled_u16(Register.CLOSED_LOOP_DECEL), _write_scaled_u16(Register.CLOSED_LOOP_DECEL)),
    "open_loop_accel_percent_s": FieldDef(_read_scaled_u16(Register.OPEN_LOOP_ACCEL), _write_scaled_u16(Register.OPEN_LOOP_ACCEL)),
    "open_loop_decel_percent_s": FieldDef(_read_scaled_u16(Register.OPEN_LOOP_DECEL), _write_scaled_u16(Register.OPEN_LOOP_DECEL)),
    "disable_brake_mode": FieldDef(_read_scaled_u16(Register.DISABLE_BRAKE_MODE), _write_scaled_u16(Register.DISABLE_BRAKE_MODE)),
    "auto_reversal_mode": FieldDef(_read_scaled_u16(Register.AUTO_REVERSAL_MODE), _write_scaled_u16(Register.AUTO_REVERSAL_MODE)),
    "hall_electric_angle": FieldDef(_read_scaled_u16(Register.HALL_ELECTRIC_ANGLE), _write_scaled_u16(Register.HALL_ELECTRIC_ANGLE)),
    "right_pole_pairs": FieldDef(_read_scaled_u16(Register.RIGHT_POLE_PAIRS), _write_scaled_u16(Register.RIGHT_POLE_PAIRS)),
    "right_phase_12": FieldDef(_read_scaled_u16(Register.RIGHT_PHASE_12), _write_scaled_u16(Register.RIGHT_PHASE_12)),
    "right_phase_34": FieldDef(_read_scaled_u16(Register.RIGHT_PHASE_34), _write_scaled_u16(Register.RIGHT_PHASE_34)),
    "right_phase_56": FieldDef(_read_scaled_u16(Register.RIGHT_PHASE_56), _write_scaled_u16(Register.RIGHT_PHASE_56)),
    "right_phase_learned": FieldDef(_read_scaled_u16(Register.RIGHT_PHASE_LEARNED), _write_scaled_u16(Register.RIGHT_PHASE_LEARNED)),
    "right_parking_max_current_amps": FieldDef(
        _read_scaled_u16(Register.RIGHT_PARKING_MAX_CURRENT, 0.01),
        _write_scaled_u16(Register.RIGHT_PARKING_MAX_CURRENT, 0.01),
    ),
    "right_foc_phase_comp_deg": FieldDef(
        _read_scaled_i16(Register.RIGHT_FOC_PHASE_COMP, 0.01),
        _write_scaled_i16(Register.RIGHT_FOC_PHASE_COMP, 0.01),
    ),
    "right_foc_advance_deg": FieldDef(
        _read_scaled_i16(Register.RIGHT_FOC_ADVANCE_ANGLE, 0.01),
        _write_scaled_i16(Register.RIGHT_FOC_ADVANCE_ANGLE, 0.01),
    ),
    "left_pole_pairs": FieldDef(_read_scaled_u16(Register.LEFT_POLE_PAIRS), _write_scaled_u16(Register.LEFT_POLE_PAIRS)),
    "left_phase_12": FieldDef(_read_scaled_u16(Register.LEFT_PHASE_12), _write_scaled_u16(Register.LEFT_PHASE_12)),
    "left_phase_34": FieldDef(_read_scaled_u16(Register.LEFT_PHASE_34), _write_scaled_u16(Register.LEFT_PHASE_34)),
    "left_phase_56": FieldDef(_read_scaled_u16(Register.LEFT_PHASE_56), _write_scaled_u16(Register.LEFT_PHASE_56)),
    "left_phase_learned": FieldDef(_read_scaled_u16(Register.LEFT_PHASE_LEARNED), _write_scaled_u16(Register.LEFT_PHASE_LEARNED)),
    "left_parking_max_current_amps": FieldDef(
        _read_scaled_u16(Register.LEFT_PARKING_MAX_CURRENT, 0.01),
        _write_scaled_u16(Register.LEFT_PARKING_MAX_CURRENT, 0.01),
    ),
    "left_foc_phase_comp_deg": FieldDef(
        _read_scaled_i16(Register.LEFT_FOC_PHASE_COMP, 0.01),
        _write_scaled_i16(Register.LEFT_FOC_PHASE_COMP, 0.01),
    ),
    "left_foc_advance_deg": FieldDef(
        _read_scaled_i16(Register.LEFT_FOC_ADVANCE_ANGLE, 0.01),
        _write_scaled_i16(Register.LEFT_FOC_ADVANCE_ANGLE, 0.01),
    ),
    "over_voltage_v": FieldDef(_read_scaled_u16(Register.OVER_VOLTAGE), _write_scaled_u16(Register.OVER_VOLTAGE)),
    "under_voltage_v": FieldDef(_read_scaled_u16(Register.UNDER_VOLTAGE), _write_scaled_u16(Register.UNDER_VOLTAGE)),
    "voltage_protect_enable": FieldDef(_read_scaled_u16(Register.VOLTAGE_PROTECT_ENABLE), _write_scaled_u16(Register.VOLTAGE_PROTECT_ENABLE)),
    "system_mode": FieldDef(_read_scaled_u16(Register.SYSTEM_MODE), _write_scaled_u16(Register.SYSTEM_MODE)),
    "control_signal": FieldDef(_read_scaled_u16(Register.CONTROL_SIGNAL), _write_scaled_u16(Register.CONTROL_SIGNAL)),
    "driver_mode": FieldDef(_read_scaled_u16(Register.DRIVER_MODE), _write_scaled_u16(Register.DRIVER_MODE)),
    "foc_closed_loop_mode": FieldDef(_read_scaled_u16(Register.FOC_CLOSED_LOOP_MODE), _write_scaled_u16(Register.FOC_CLOSED_LOOP_MODE)),
    "diff_turn_max_speed_rpm": FieldDef(_read_scaled_u16(Register.DIFF_TURN_MAX_SPEED), _write_scaled_u16(Register.DIFF_TURN_MAX_SPEED)),
    "stall_stop_time_ms": FieldDef(_read_scaled_u16(Register.STALL_STOP_TIME), _write_scaled_u16(Register.STALL_STOP_TIME)),
    "phase_overcurrent_cutoff_amps": FieldDef(
        _read_scaled_u16(Register.PHASE_OVERCURRENT_CUTOFF, 0.01),
        _write_scaled_u16(Register.PHASE_OVERCURRENT_CUTOFF, 0.01),
    ),
    "bus_protection_cutoff_amps": FieldDef(
        _read_scaled_u16(Register.BUS_PROTECTION_CUTOFF, 0.01),
        _write_scaled_u16(Register.BUS_PROTECTION_CUTOFF, 0.01),
    ),
    "energy_recovery_protect_voltage_v": FieldDef(
        _read_scaled_u16(Register.ENERGY_RECOVERY_PROTECT_VOLTAGE),
        _write_scaled_u16(Register.ENERGY_RECOVERY_PROTECT_VOLTAGE),
    ),
    "energy_recovery_protect_mode": FieldDef(
        _read_scaled_u16(Register.ENERGY_RECOVERY_PROTECT_MODE),
        _write_scaled_u16(Register.ENERGY_RECOVERY_PROTECT_MODE),
    ),
    "rs485_device_address": FieldDef(
        _read_scaled_u16(Register.RS485_DEVICE_ADDRESS),
        _write_scaled_u16(Register.RS485_DEVICE_ADDRESS),
    ),
    "rs485_baudrate_code": FieldDef(_read_scaled_u16(Register.RS485_BAUDRATE), _write_scaled_u16(Register.RS485_BAUDRATE)),
    "rs485_parity_code": FieldDef(_read_scaled_u16(Register.RS485_PARITY), _write_scaled_u16(Register.RS485_PARITY)),
    "rs485_brake_timeout_s": FieldDef(_read_scaled_u16(Register.RS485_BRAKE_TIMEOUT), _write_scaled_u16(Register.RS485_BRAKE_TIMEOUT)),
    "can_node_id": FieldDef(_read_scaled_u16(Register.CAN_NODE_ID), _write_scaled_u16(Register.CAN_NODE_ID)),
    "can_baudrate_code": FieldDef(_read_scaled_u16(Register.CAN_BAUDRATE), _write_scaled_u16(Register.CAN_BAUDRATE)),
    "can_brake_timeout_s": FieldDef(_read_scaled_u16(Register.CAN_BRAKE_TIMEOUT), _write_scaled_u16(Register.CAN_BRAKE_TIMEOUT)),
}


def _install_dynamic_methods() -> None:
    def make_getter(name: str) -> Callable[[MSSD60EHB], Union[int, float]]:
        def getter(self: MSSD60EHB) -> Union[int, float]:
            return self.read_named(name)

        getter.__name__ = f"get_{name}"
        getter.__doc__ = f"读取字段 {name}"
        return getter

    def make_setter(name: str) -> Callable[[MSSD60EHB, Union[int, float]], None]:
        def setter(self: MSSD60EHB, value: Union[int, float]) -> None:
            self.write_named(name, value)

        setter.__name__ = f"set_{name}"
        setter.__doc__ = f"写入字段 {name}"
        return setter

    for name, field in FIELD_DEFS.items():
        setattr(MSSD60EHB, f"get_{name}", make_getter(name))
        if field.setter is not None:
            setattr(MSSD60EHB, f"set_{name}", make_setter(name))


_install_dynamic_methods()


if __name__ == "__main__":
    import pprint

    PORT = "COM3"

    with MSSD60EHB(PORT, slave_id=1, baudrate=9600, timeout=0.3) as driver:
        pprint.pprint(driver.get_device_info())
        pprint.pprint(driver.get_runtime_status())
        driver.set_system_mode(SystemMode.INDEPENDENT_CLOSED_LOOP)
        driver.set_control_signal(ControlSignal.RS485)
        driver.set_driver_mode(DriverMode.FOC)
        driver.set_right_target(1000)
        driver.set_left_target(-500)
