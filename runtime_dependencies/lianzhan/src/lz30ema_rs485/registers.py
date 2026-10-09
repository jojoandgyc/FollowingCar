"""Register definitions from the LZ-30EMA_2EC_N RS485 manual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

F03 = (0x03,)
F06 = (0x06,)
F10 = (0x10,)
F03_06_10 = (0x03, 0x06, 0x10)
F03_10 = (0x03, 0x10)


@dataclass(frozen=True)
class Register:
    name: str
    address: int
    description: str
    functions: tuple[int, ...]
    value_range: str = ""
    unit: str = ""
    scale: Optional[float] = None
    signed: bool = False
    notes: str = ""


@dataclass(frozen=True)
class ValueField:
    name: str
    address: int
    words: int
    description: str
    functions: tuple[int, ...]
    signed: bool = False
    scale: Optional[float] = None
    unit: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ReservedRange:
    start: int
    end: int
    description: str = "保留"


REGISTERS: tuple[Register, ...] = (
    Register("device_id", 0x0000, "设备标识号", F03),
    Register("device_version", 0x0001, "设备版本", F03, notes="高字节主版本号，低字节副版本号"),
    Register("device_max_current", 0x0002, "最大电流", F03, unit="A", scale=0.01),
    Register("right_phase_current", 0x0008, "右电机实时相电流", F03, "0~65535", "A", 0.01),
    Register("right_speed_high", 0x0009, "右电机实时速度高半字", F03, unit="RPM"),
    Register("right_speed_low", 0x000A, "右电机实时速度低半字", F03, unit="RPM"),
    Register("right_error_state", 0x000B, "右电机错误状态", F03, "0~18"),
    Register("right_pwm", 0x000C, "右电机实时 PWM", F03, "0~10000", "%", 0.1),
    Register("right_board_temperature", 0x000D, "右电路板温度", F03, "-200~200", "C", None, True),
    Register("right_position_high", 0x000F, "右电机位移高半字", F03, unit="degree"),
    Register("right_position_low", 0x0010, "右电机位移低半字", F03, unit="degree"),
    Register("right_reverse_state", 0x0011, "右电机反向状态", F03, "0,1"),
    Register("left_phase_current", 0x0012, "左电机实时相电流", F03, "0~65535", "A", 0.01),
    Register("left_speed_high", 0x0013, "左电机实时速度高半字", F03, unit="RPM"),
    Register("left_speed_low", 0x0014, "左电机实时速度低半字", F03, unit="RPM"),
    Register("left_error_state", 0x0015, "左电机错误状态", F03, "0~18"),
    Register("left_pwm", 0x0016, "左电机实时 PWM", F03, "0~10000", "%", 0.1),
    Register("left_board_temperature", 0x0017, "左电路板温度", F03, "-200~200", "C", None, True),
    Register("left_position_high", 0x0019, "左电机位移高半字", F03, unit="degree"),
    Register("left_position_low", 0x001A, "左电机位移低半字", F03, unit="degree"),
    Register("left_reverse_state", 0x001B, "左电机反向状态", F03, "0,1"),
    Register("supply_voltage", 0x001C, "电源电压", F03, "0~1000", "V"),
    Register("runtime_system_mode", 0x001D, "实时系统模式", F03, "0~4"),
    Register("control_mode", 0x001E, "控制模式", F03),
    Register("in1_level", 0x001F, "IN1 电平", F03, "0,1"),
    Register("in2_level", 0x0020, "IN2 电平", F03, "0,1"),
    Register("right_stop", 0x0040, "右电机停止", F06, "0,1,2"),
    Register("right_speed_target_high", 0x0041, "右电机速度目标高半字", (0x06, 0x10), unit="RPM"),
    Register("right_speed_target_low", 0x0042, "右电机速度目标低半字", (0x06, 0x10), unit="RPM"),
    Register("right_brake_control", 0x0043, "右电机抱闸控制", F06, "0,1"),
    Register("left_stop", 0x0044, "左电机停止", F06, "0,1,2"),
    Register("left_speed_target_high", 0x0045, "左电机速度目标高半字", (0x06, 0x10), unit="RPM"),
    Register("left_speed_target_low", 0x0046, "左电机速度目标低半字", (0x06, 0x10), unit="RPM"),
    Register("left_brake_control", 0x0047, "左电机抱闸控制", F06, "0,1"),
    Register("factory_reset_command", 0x0048, "参数恢复出厂命令", F06, "0,1"),
    Register("test_control", 0x0049, "测试控制", F06, "0,1,2", notes="0 取消测试，1 左电机相序学习，2 右电机相序学习"),
    Register("max_output_current", 0x0058, "最大输出电流(单路)", F03_06_10, "50~6000", "A", 0.01),
    Register("max_forward_speed_high", 0x0059, "电机正转最大转速高半字", F03_06_10, unit="RPM"),
    Register("max_forward_speed_low", 0x005A, "电机正转最大转速低半字", F03_06_10, unit="RPM"),
    Register("max_reverse_speed_high", 0x005B, "电机反转最大转速高半字", F03_06_10, unit="RPM"),
    Register("max_reverse_speed_low", 0x005C, "电机反转最大转速低半字", F03_06_10, unit="RPM"),
    Register("min_speed_high", 0x005D, "电机最小转速高半字", F03_06_10, unit="RPM"),
    Register("min_speed_low", 0x005E, "电机最小转速低半字", F03_06_10, unit="RPM"),
    Register("closed_loop_acceleration", 0x005F, "闭环加速加速度", F03_06_10, "10~65535", "RPM/S"),
    Register("closed_loop_deceleration", 0x0060, "闭环减速加速度", F03_06_10, "10~65535", "RPM/S"),
    Register("reserved_0x0061", 0x0061, "保留", F03_06_10, "4~500"),
    Register("reserved_0x0062", 0x0062, "保留", F03_06_10, "4~500"),
    Register("reserved_0x0063", 0x0063, "保留", F03_06_10, "0,1,2"),
    Register("auto_reverse_mode", 0x0064, "自动换向模式", F03_06_10, "0,1"),
    Register("hall_electrical_angle", 0x0065, "电机霍尔电角度", F03_06_10, "0,1", notes="0: 60 degree, 1: 120 degree"),
    Register("sensor_type", 0x0066, "传感器类型", F03_06_10, "0,1", notes="0: 霍尔, 1: 霍尔加编码器 AB"),
    Register("right_pole_pairs", 0x0068, "右电机极对数", F03_06_10, "0~65535"),
    Register("right_phase_sequence_12", 0x0069, "右电机 1/2 相相序数据", F03_06_10),
    Register("right_phase_sequence_34", 0x006A, "右电机 3/4 相相序数据", F03_06_10),
    Register("right_phase_sequence_56", 0x006B, "右电机 5/6 相相序数据", F03_06_10),
    Register("right_phase_learning_state", 0x006C, "右电机相序学习状态", F03_06_10, "0,1"),
    Register("right_parking_current", 0x006D, "右电机电子驻车最大电流", F03_06_10, unit="A", scale=0.01),
    Register("right_foc_phase_compensation", 0x006E, "右电机 FOC 相位补偿角度", F03_06_10, "-18000~18000", "degree", 0.01, True),
    Register("right_foc_lead_angle", 0x006F, "右电机 FOC 超前角度", F03_06_10, "-18000~18000", "degree", 0.01, True),
    Register("right_reverse_enabled", 0x0070, "右电机是否反向", F03_06_10, "0,1"),
    Register("left_pole_pairs", 0x0071, "左电机极对数", F03_06_10, "0~65535"),
    Register("left_phase_sequence_12", 0x0072, "左电机 1/2 相相序数据", F03_06_10),
    Register("left_phase_sequence_34", 0x0073, "左电机 3/4 相相序数据", F03_06_10),
    Register("left_phase_sequence_56", 0x0074, "左电机 5/6 相相序数据", F03_06_10),
    Register("left_phase_learning_state", 0x0075, "左电机相序学习状态", F03_06_10, "0,1"),
    Register("left_parking_current", 0x0076, "左电机电子驻车最大电流", F03_06_10, unit="A", scale=0.01),
    Register("left_foc_phase_compensation", 0x0077, "左电机 FOC 相位补偿角度", F03_06_10, "-18000~18000", "degree", 0.01, True),
    Register("left_foc_lead_angle", 0x0078, "左电机 FOC 超前角度", F03_06_10, "-18000~18000", "degree", 0.01, True),
    Register("left_reverse_enabled", 0x0079, "左电机是否反向", F03_06_10, "0,1"),
    Register("over_voltage_limit", 0x0084, "过压保护电压", F03_06_10, "8~60", "V"),
    Register("under_voltage_limit", 0x0085, "欠压保护电压", F03_06_10, "8~60", "V"),
    Register("voltage_protection_enabled", 0x0086, "是否进行电压保护", F03_06_10, "0,1"),
    Register("system_mode", 0x0087, "系统模式", F03_06_10, "0~4"),
    Register("foc_loop_mode", 0x0089, "FOC 闭环模式", F03_06_10, "0,1"),
    Register("differential_turn_max_speed", 0x008A, "电机差速转弯最大速度", F03_06_10, unit="RPM"),
    Register("stall_stop_time", 0x008B, "堵转停机时间", F03_06_10, unit="ms"),
    Register("phase_overcurrent_shutdown_current", 0x008C, "相过流关闭电流", F03_06_10, "100~8000", "A", 0.01),
    Register("right_brake_output_type", 0x00B0, "右电机抱闸输出类型", F03_06_10, "0,1,2"),
    Register("right_stop_brake_enabled", 0x00B1, "右电机停车刹车设置", F03_06_10, "0,1"),
    Register("left_brake_output_type", 0x00B2, "左电机抱闸输出类型", F03_06_10, "0,1,2"),
    Register("left_stop_brake_enabled", 0x00B3, "左电机停车刹车设置", F03_06_10, "0,1"),
    Register("rs485_address", 0x00C8, "485 通讯设备地址", F03_06_10, "0~255"),
    Register("rs485_baud_rate_code", 0x00C9, "485 通讯串口波特率", F03_06_10, "0~4"),
    Register("rs485_parity_code", 0x00CA, "485 通讯串口校验方式", F03_06_10, "0~3"),
    Register("rs485_timeout_s", 0x00CB, "485 通讯中断制动时间", F03_06_10, unit="s"),
    Register("can_node_id", 0x00CD, "CAN 通讯节点 ID", F03_06_10, "0~127"),
    Register("can_baud_rate_code", 0x00CE, "CAN 通讯串口波特率", F03_06_10, "0~13"),
    Register("can_timeout_s", 0x00CF, "CAN 通讯中断制动时间", F03_06_10, unit="s"),
    Register("ttl_address", 0x00D2, "TTL 设备地址", F03_06_10, "0~255"),
    Register("ttl_baud_rate_code", 0x00D3, "TTL 通讯串口波特率", F03_06_10, "0~4"),
    Register("ttl_parity_code", 0x00D4, "TTL 通讯串口校验方式", F03_06_10, "0~3"),
    Register("ttl_timeout_s", 0x00D5, "TTL 通讯中断制动时间", F03_06_10, unit="s"),
    Register("speed_loop_p_high", 0x00E0, "速度闭环 P 系数高半字", F03_10),
    Register("speed_loop_p_low", 0x00E1, "速度闭环 P 系数低半字", F03_10),
    Register("speed_loop_i_high", 0x00E2, "速度闭环 I 系数高半字", F03_10),
    Register("speed_loop_i_low", 0x00E3, "速度闭环 I 系数低半字", F03_10),
    Register("speed_loop_d_high", 0x00E4, "速度闭环 D 系数高半字", F03_10),
    Register("speed_loop_d_low", 0x00E5, "速度闭环 D 系数低半字", F03_10),
    Register("d_axis_p_high", 0x00E6, "D 轴 P 系数高半字", F03_10),
    Register("d_axis_p_low", 0x00E7, "D 轴 P 系数低半字", F03_10),
    Register("d_axis_i_high", 0x00E8, "D 轴 I 系数高半字", F03_10),
    Register("d_axis_i_low", 0x00E9, "D 轴 I 系数低半字", F03_10),
    Register("d_axis_d_high", 0x00EA, "D 轴 D 系数高半字", F03_10),
    Register("d_axis_d_low", 0x00EB, "D 轴 D 系数低半字", F03_10),
    Register("q_axis_p_high", 0x00EC, "Q 轴 P 系数高半字", F03_10),
    Register("q_axis_p_low", 0x00ED, "Q 轴 P 系数低半字", F03_10),
    Register("q_axis_i_high", 0x00EE, "Q 轴 I 系数高半字", F03_10),
    Register("q_axis_i_low", 0x00EF, "Q 轴 I 系数低半字", F03_10),
    Register("q_axis_d_high", 0x00F0, "Q 轴 D 系数高半字", F03_10),
    Register("q_axis_d_low", 0x00F1, "Q 轴 D 系数低半字", F03_10),
    Register("encoder_parking_speed_p_high", 0x00F2, "编码器驻车速度 PI P 高半字", F03_10),
    Register("encoder_parking_speed_p_low", 0x00F3, "编码器驻车速度 PI P 低半字", F03_10),
    Register("encoder_parking_speed_i_high", 0x00F4, "编码器驻车速度 PI I 高半字", F03_10),
    Register("encoder_parking_speed_i_low", 0x00F5, "编码器驻车速度 PI I 低半字", F03_10),
    Register("encoder_parking_speed_d_high", 0x00F6, "编码器驻车速度 PI D 高半字", F03_10),
    Register("encoder_parking_speed_d_low", 0x00F7, "编码器驻车速度 PI D 低半字", F03_10),
    Register("encoder_parking_d_axis_p_high", 0x00F8, "编码器驻车 D 轴 PI P 高半字", F03_10),
    Register("encoder_parking_d_axis_p_low", 0x00F9, "编码器驻车 D 轴 PI P 低半字", F03_10),
    Register("encoder_parking_d_axis_i_high", 0x00FA, "编码器驻车 D 轴 PI I 高半字", F03_10),
    Register("encoder_parking_d_axis_i_low", 0x00FB, "编码器驻车 D 轴 PI I 低半字", F03_10),
    Register("encoder_parking_d_axis_d_high", 0x00FC, "编码器驻车 D 轴 PI D 高半字", F03_10),
    Register("encoder_parking_d_axis_d_low", 0x00FD, "编码器驻车 D 轴 PI D 低半字", F03_10),
    Register("encoder_parking_q_axis_p_high", 0x00FE, "编码器驻车 Q 轴 PI P 高半字", F03_10),
    Register("encoder_parking_q_axis_p_low", 0x00FF, "编码器驻车 Q 轴 PI P 低半字", F03_10),
    Register("encoder_parking_q_axis_i_high", 0x0100, "编码器驻车 Q 轴 PI I 高半字", F03_10),
    Register("encoder_parking_q_axis_i_low", 0x0101, "编码器驻车 Q 轴 PI I 低半字", F03_10),
    Register("encoder_parking_q_axis_d_high", 0x0102, "编码器驻车 Q 轴 PI D 高半字", F03_10),
    Register("encoder_parking_q_axis_d_low", 0x0103, "编码器驻车 Q 轴 PI D 低半字", F03_10),
    Register("hall_parking_speed_p_high", 0x0104, "霍尔驻车速度 PI P 高半字", F03_10),
    Register("hall_parking_speed_p_low", 0x0105, "霍尔驻车速度 PI P 低半字", F03_10),
)


RESERVED_RANGES: tuple[ReservedRange, ...] = (
    ReservedRange(0x0003, 0x0007),
    ReservedRange(0x000E, 0x000E),
    ReservedRange(0x0018, 0x0018),
    ReservedRange(0x0021, 0x003F),
    ReservedRange(0x004A, 0x0057),
    ReservedRange(0x0067, 0x0067),
    ReservedRange(0x007A, 0x0083),
    ReservedRange(0x0088, 0x0088),
    ReservedRange(0x008D, 0x009A),
    ReservedRange(0x009B, 0x00AF),
    ReservedRange(0x00B4, 0x00C7),
    ReservedRange(0x00CC, 0x00CC),
    ReservedRange(0x00D0, 0x00D1),
    ReservedRange(0x00D6, 0x00DF),
    ReservedRange(0x0106, 0x0109),
)


PID_SCALE = 1.0 / 32768.0

VALUE_FIELDS: tuple[ValueField, ...] = (
    ValueField("right_speed_actual", 0x0009, 2, "右电机实时速度", F03, True, unit="RPM"),
    ValueField("right_position", 0x000F, 2, "右电机位移", F03, True, unit="degree"),
    ValueField("left_speed_actual", 0x0013, 2, "左电机实时速度", F03, True, unit="RPM"),
    ValueField("left_position", 0x0019, 2, "左电机位移", F03, True, unit="degree"),
    ValueField("right_speed_target", 0x0041, 2, "右电机速度目标/同源速度目标/差速直行速度", (0x06, 0x10), True, unit="RPM"),
    ValueField("left_speed_target", 0x0045, 2, "左电机速度目标/差速转弯速度", (0x06, 0x10), True, unit="RPM"),
    ValueField("max_forward_speed", 0x0059, 2, "电机正转最大转速", F03_06_10, False, unit="RPM"),
    ValueField("max_reverse_speed", 0x005B, 2, "电机反转最大转速", F03_06_10, False, unit="RPM"),
    ValueField("min_speed", 0x005D, 2, "电机最小转速", F03_06_10, False, unit="RPM"),
    ValueField("speed_loop_p", 0x00E0, 2, "速度闭环 P 系数", F03_10, False, PID_SCALE),
    ValueField("speed_loop_i", 0x00E2, 2, "速度闭环 I 系数", F03_10, False, PID_SCALE),
    ValueField("speed_loop_d", 0x00E4, 2, "速度闭环 D 系数", F03_10, False, PID_SCALE),
    ValueField("d_axis_p", 0x00E6, 2, "D 轴 P 系数", F03_10, False, PID_SCALE),
    ValueField("d_axis_i", 0x00E8, 2, "D 轴 I 系数", F03_10, False, PID_SCALE),
    ValueField("d_axis_d", 0x00EA, 2, "D 轴 D 系数", F03_10, False, PID_SCALE),
    ValueField("q_axis_p", 0x00EC, 2, "Q 轴 P 系数", F03_10, False, PID_SCALE),
    ValueField("q_axis_i", 0x00EE, 2, "Q 轴 I 系数", F03_10, False, PID_SCALE),
    ValueField("q_axis_d", 0x00F0, 2, "Q 轴 D 系数", F03_10, False, PID_SCALE),
    ValueField("encoder_parking_speed_p", 0x00F2, 2, "编码器驻车速度 PI P 参数", F03_10, False, PID_SCALE),
    ValueField("encoder_parking_speed_i", 0x00F4, 2, "编码器驻车速度 PI I 参数", F03_10, False, PID_SCALE),
    ValueField("encoder_parking_speed_d", 0x00F6, 2, "编码器驻车速度 PI D 参数", F03_10, False, PID_SCALE),
    ValueField("encoder_parking_d_axis_p", 0x00F8, 2, "编码器驻车 D 轴 PI P 参数", F03_10, False, PID_SCALE),
    ValueField("encoder_parking_d_axis_i", 0x00FA, 2, "编码器驻车 D 轴 PI I 参数", F03_10, False, PID_SCALE),
    ValueField("encoder_parking_d_axis_d", 0x00FC, 2, "编码器驻车 D 轴 PI D 参数", F03_10, False, PID_SCALE),
    ValueField("encoder_parking_q_axis_p", 0x00FE, 2, "编码器驻车 Q 轴 PI P 参数", F03_10, False, PID_SCALE),
    ValueField("encoder_parking_q_axis_i", 0x0100, 2, "编码器驻车 Q 轴 PI I 参数", F03_10, False, PID_SCALE),
    ValueField("encoder_parking_q_axis_d", 0x0102, 2, "编码器驻车 Q 轴 PI D 参数", F03_10, False, PID_SCALE),
    ValueField("hall_parking_speed_p", 0x0104, 2, "霍尔驻车速度 PI P 参数", F03_10, False),
)


REGISTER_BY_NAME = {register.name: register for register in REGISTERS}
REGISTER_BY_ADDRESS = {register.address: register for register in REGISTERS}
FIELD_BY_NAME = {field.name: field for field in VALUE_FIELDS}

ERROR_STATES = {
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
    11: "CAN 通讯中断",
    12: "母线电压过高",
    13: "温度过高",
    14: "无感启动失败",
    15: "相电流异常",
    16: "上电大电流异常",
    17: "同步超差异常",
    18: "硬件异常",
}

BAUD_RATE_CODES = {
    9600: 0,
    19200: 1,
    38400: 2,
    57600: 3,
    115200: 4,
}

CAN_BAUD_RATE_CODES = {
    10000: 0,
    20000: 1,
    40000: 2,
    50000: 3,
    80000: 4,
    100000: 5,
    125000: 6,
    200000: 7,
    250000: 8,
    400000: 9,
    500000: 10,
    600000: 11,
    800000: 12,
    1000000: 13,
}

PARITY_CODES = {
    "N1": 0,
    "E1": 1,
    "O1": 2,
    "N2": 3,
}


def get_register(register: str | int | Register) -> Register:
    if isinstance(register, Register):
        return register
    if isinstance(register, str):
        try:
            return REGISTER_BY_NAME[register]
        except KeyError as exc:
            raise KeyError(f"unknown register name: {register}") from exc
    try:
        return REGISTER_BY_ADDRESS[int(register)]
    except KeyError as exc:
        raise KeyError(f"unknown register address: 0x{int(register):04X}") from exc


def get_field(field: str | ValueField) -> ValueField:
    if isinstance(field, ValueField):
        return field
    try:
        return FIELD_BY_NAME[field]
    except KeyError as exc:
        raise KeyError(f"unknown field name: {field}") from exc
