"""LZ-30EMA startup ramp diagnostics and an explicit volatile acceleration trial.

The caller owns the motor lock and must already have stopped the controller.
This module never opens a port, changes control mode, issues STOP, retries an
operation, or restores a previous setting. Register values are not measured
wheel acceleration. The legacy MSSD register map must not be used here.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, TypedDict


class RampConfigurationError(RuntimeError):
    """An acceleration write was attempted but its result is not verified."""

    def __init__(self, message: str, *, diagnostics: dict):
        super().__init__(message)
        self.write_attempted = True
        self.diagnostics = dict(diagnostics)


class MotorRampDiagnostics(TypedDict):
    before_acceleration_rpm_s: int
    acceleration_rpm_s: int
    deceleration_rpm_s: int
    requested_acceleration_rpm_s: Optional[int]
    changed: bool
    persist: bool
    mode_policy: Optional[str]


def validate_acceleration_rpm_s(value: Optional[int]) -> Optional[int]:
    """Accept an absent override or an exact integer in the LZ manual's range."""
    if value is not None and (type(value) is not int or not 10 <= value <= 65535):
        raise ValueError("closed-loop acceleration must be an integer from 10 to 65535 RPM/s, or None")
    return value


def validate_closed_loop_mode(bus, configured_mode) -> str:
    """Check existing independent command semantics without changing mode.

    Mode 0/control 0x38 is a specifically observed compatibility combination;
    do not treat other mode-0, open-loop or differential modes as equivalent.
    """
    runtime_mode = bus.get("runtime_system_mode") if isinstance(bus, Mapping) else None
    control = bus.get("control_mode") if isinstance(bus, Mapping) else None
    observed = f"runtime={runtime_mode!r} configured={configured_mode!r} control={control!r}"
    if (type(runtime_mode) is not int or type(configured_mode) is not int
            or type(control) is not int):
        raise RuntimeError(f"控制器模式数据缺失/非法：{observed}；未改变模式")
    if control not in (0x08, 0x18, 0x28, 0x38):
        raise RuntimeError(f"需要闭环通讯控制，拒绝外部控制源/开环/未知状态：{observed}；未改变模式")
    if runtime_mode == configured_mode == 1:
        return "independent_closed_loop_1"
    if runtime_mode == configured_mode == 0 and control == 0x38:
        return "compat_mode_0_control_0x38"
    raise RuntimeError(f"不支持的系统模式组合：{observed}；拒绝自动改变控制器模式")


def _read_ramp_pair(driver) -> tuple[int, int]:
    values = driver.read_holding_registers(0x005F, 2)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("LZ ramp readback must contain exactly two registers at 0x005F/0x0060")
    for name, value in zip(("acceleration", "deceleration"), values):
        if type(value) is not int or not 10 <= value <= 65535:
            raise ValueError(f"invalid LZ {name} readback: {value!r}; expected 10..65535 RPM/s")
    return values[0], values[1]


def configure_closed_loop_ramp(driver, acceleration_rpm_s: Optional[int] = None) -> MotorRampDiagnostics:
    """Read both ramps; optionally write only acceleration using volatile 0x06.

    Initial read/mode failures propagate unchanged, before any parameter write.
    Once a write is attempted, every write/readback failure becomes a
    RampConfigurationError so the backend can latch a motion prohibition.
    """
    requested = validate_acceleration_rpm_s(acceleration_rpm_s)
    before_acceleration, before_deceleration = _read_ramp_pair(driver)
    result: MotorRampDiagnostics = {
        "before_acceleration_rpm_s": before_acceleration,
        "acceleration_rpm_s": before_acceleration,
        "deceleration_rpm_s": before_deceleration,
        "requested_acceleration_rpm_s": requested,
        "changed": False,
        "persist": False,
        "mode_policy": None,
    }
    if requested is None:
        return result
    result["mode_policy"] = validate_closed_loop_mode(
        driver.read_bus_status(), driver.read_register("system_mode"),
    )
    if requested == before_acceleration:
        return result
    actual_acceleration = actual_deceleration = None
    try:
        driver.write_register("closed_loop_acceleration", requested, persist=False)
        actual_acceleration, actual_deceleration = _read_ramp_pair(driver)
        if actual_acceleration != requested or actual_deceleration != before_deceleration:
            raise ValueError(
                "LZ ramp readback mismatch: "
                f"requested_acceleration={requested} actual_acceleration={actual_acceleration} "
                f"before_deceleration={before_deceleration} actual_deceleration={actual_deceleration} RPM/s"
            )
    except Exception as exc:
        raise RampConfigurationError(
            f"LZ volatile acceleration write not verified: requested={requested} RPM/s; "
            f"{type(exc).__name__}: {exc}; no retry or restore performed",
            diagnostics=dict(
                result, acceleration_rpm_s=actual_acceleration,
                deceleration_rpm_s=actual_deceleration,
                before_deceleration_rpm_s=before_deceleration,
                changed=None, write_attempted=True, verified=False,
                error=f"{type(exc).__name__}: {exc}",
            ),
        ) from exc
    result.update(acceleration_rpm_s=actual_acceleration,
                  deceleration_rpm_s=actual_deceleration, changed=True)
    return result
