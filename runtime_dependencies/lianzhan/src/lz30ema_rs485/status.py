"""Realtime status formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .client import MotorStatus, RealtimeSnapshot


SYSTEM_MODE_NAMES = {
    0: "独立速度开环",
    1: "独立速度闭环",
    2: "同源速度开环",
    3: "同源速度闭环",
    4: "差速闭环",
}

CONTROL_SOURCE_NAMES = {
    0: "通讯",
    1: "航模PWM",
    2: "摇杆",
    7: "相序学习",
}


@dataclass(frozen=True)
class ScreenshotFieldCoverage:
    available: tuple[str, ...]
    unavailable: tuple[str, ...]


SCREENSHOT_FIELD_COVERAGE = ScreenshotFieldCoverage(
    available=(
        "最大电流(A)",
        "系统模式",
        "电压(V)",
        "控制模式",
        "IN1电平",
        "IN2电平",
        "左/右电机相电流(A)",
        "左/右电机速度(RPM)",
        "左/右电机位置",
        "左/右电机实时PWM(%)",
        "左/右电机温度",
        "左/右电机异常状态",
        "左/右电机反向状态",
    ),
    unavailable=(
        "IN3~IN8电平",
        "遥控急停状态",
        "遥控挡位状态",
        "遥控驻车状态",
        "母线电流(A)",
        "负载率(%)",
        "控制电压(mV)",
        "航模PWM(us)",
        "回充保护状态",
    ),
)


def format_system_mode(value: int) -> str:
    return SYSTEM_MODE_NAMES.get(value, f"未知({value})")


def format_control_mode(value: int) -> str:
    drive = "FOC" if value & 0x20 else "方波"
    source = CONTROL_SOURCE_NAMES.get(value & 0x07, f"未知控制源({value & 0x07})")
    return f"{drive}{source}"


def format_level(value: int) -> str:
    return "高" if value else "低"


def format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def format_realtime_snapshot(
    snapshot: RealtimeSnapshot,
    *,
    include_unsupported: bool = True,
) -> str:
    """Format realtime values in a layout similar to the vendor tool."""

    info = snapshot.device_info
    bus = snapshot.bus_status
    lines: list[str] = []
    lines.append(f"最大电流(A): {format_number(float(info['max_current_a']))}")
    lines.append("")
    lines.append("实时信息")
    lines.append(
        _pair(
            "系统模式",
            format_system_mode(int(bus["runtime_system_mode"])),
            "电压(V)",
            bus["supply_voltage_v"],
        )
    )
    lines.append(f"控制模式: {format_control_mode(int(bus['control_mode']))}")
    lines.append(_pair("IN1电平", format_level(int(bus["in1_level"])), "IN2电平", format_level(int(bus["in2_level"]))))
    if include_unsupported:
        lines.append(_pair("IN3电平", "协议未提供", "IN4电平", "协议未提供"))
        lines.append(_pair("IN5电平", "协议未提供", "IN6电平", "协议未提供"))
        lines.append(_pair("IN7电平", "协议未提供", "IN8电平", "协议未提供"))
        lines.append("")
        lines.append(_pair("遥控急停状态", "协议未提供", "遥控挡位状态", "协议未提供"))
        lines.append(f"遥控驻车状态: 协议未提供")
    lines.append("")
    lines.extend(_format_motor("左电机", snapshot.left_motor, include_unsupported))
    lines.append("")
    lines.extend(_format_motor("右电机", snapshot.right_motor, include_unsupported))
    return "\n".join(lines)


def protocol_field_coverage_text() -> str:
    lines = ["协议能获取："]
    lines.extend(f"- {item}" for item in SCREENSHOT_FIELD_COVERAGE.available)
    lines.append("协议未提供对应实时寄存器：")
    lines.extend(f"- {item}" for item in SCREENSHOT_FIELD_COVERAGE.unavailable)
    return "\n".join(lines)


def _format_motor(title: str, status: MotorStatus, include_unsupported: bool) -> list[str]:
    lines = [title]
    if include_unsupported:
        lines.append("母线电流(A): 协议未提供")
    lines.append(f"相电流(A): {format_number(status.phase_current_a)}")
    lines.append(f"速度(RPM): {status.speed_rpm}")
    lines.append(f"电机位置: {status.position_degree}")
    if include_unsupported:
        lines.append("负载率(%): 协议未提供")
        lines.append("控制电压(mV): 协议未提供")
        lines.append("航模PWM(us): 协议未提供")
    lines.append(f"实时PWM(%): {format_number(status.pwm_percent)}")
    lines.append(f"温度(°): {status.board_temperature_c}")
    lines.append(f"异常状态: {status.error_name}")
    lines.append(f"反向状态: {'反向' if status.reverse else '无'}")
    if include_unsupported:
        lines.append("回充保护状态: 协议未提供")
    return lines


def _pair(left_label: str, left_value: object, right_label: str, right_value: object) -> str:
    left = f"{left_label}: {left_value}"
    right = f"{right_label}: {right_value}"
    return f"{left:<24}{right}"
