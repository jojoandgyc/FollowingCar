from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional


_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)
_DEFAULT_LIANZHAN_ROOT = "/home/topeet/lianzhan"
_STARTUP_PARKING_PRELOCK_CURRENT_A = 1.0
_STARTUP_PARKING_SETTLE_SEC = 0.10


def normalize_mssd_stop_mode(value: str, default: str = "normal") -> str:
    mode = str(value or default).strip().lower()
    if mode in {"normal", "emergency", "free"}:
        return mode
    return default


@dataclass(frozen=True)
class MssdMotorConfig:
    port: str
    slave_id: int
    baudrate: int
    timeout: float
    lib_dir: str
    max_target: int
    percent_limit: int
    left_sign: int
    right_sign: int
    forward_target_sign: int
    m1_is_left_wheel: bool
    exit_parking_mode_on_arm: bool
    stop_mode: str
    stop_zero_delay_sec: float
    parking_current_a: float = 5.0
    startup_parking_enabled: bool = True


class MssdMotorBackend:
    """Compatibility adapter for the LZ-30EMA_2EC_N RS485 controller.

    The class and config names stay unchanged so the person-follow runtime does
    not need a broad migration. Targets exposed to the rest of the project are
    now signed wheel RPM values consumed by ``lz30ema_rs485``.
    """

    def __init__(
        self,
        config: MssdMotorConfig,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.io_lock = threading.Lock()
        self.driver = None
        self.driver_ctx = None
        self.classes = None
        self.motion_armed = False
        self.parking_current_a = 0.0

    def clip_percent(self, percent: int) -> int:
        return max(0, min(int(self.config.percent_limit), int(percent)))

    def _resolve_lib_dir(self) -> str:
        lib_dir = str(self.config.lib_dir or "").strip()
        if not lib_dir:
            lib_dir = _DEFAULT_LIANZHAN_ROOT
        elif not os.path.isabs(lib_dir):
            lib_dir = os.path.abspath(os.path.join(_REPO_ROOT, lib_dir))

        candidates = (lib_dir, os.path.join(lib_dir, "src"))
        for candidate in candidates:
            package_init = os.path.join(candidate, "lz30ema_rs485", "__init__.py")
            if os.path.isfile(package_init):
                return candidate
        raise RuntimeError(
            "MOTOR_RS485_LIB_DIR does not contain lz30ema_rs485: "
            f"{lib_dir} (checked the path itself and its src directory)"
        )

    def ensure_driver(self):
        if self.driver is not None:
            return self.driver
        lib_dir = self._resolve_lib_dir()
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        from lz30ema_rs485 import LZ30EMAClient, StopMode

        driver = LZ30EMAClient.from_serial(
            self.config.port,
            slave=self.config.slave_id,
            baudrate=self.config.baudrate,
            timeout=self.config.timeout,
        )
        self.driver = driver
        self.classes = (LZ30EMAClient, StopMode)
        try:
            if self.config.startup_parking_enabled:
                self.enable_startup_parking()
            self.arm_for_motion(driver, force=True)
        except Exception:
            try:
                self.set_parking_current(0.0)
            except Exception as cleanup_exc:
                self.logger.warning("LZ30EMA 初始化失败后清零驻车电流失败: %s", cleanup_exc)
            driver.close()
            self.driver = None
            self.classes = None
            self.parking_current_a = 0.0
            raise
        self.logger.info(
            "LZ30EMA motor backend ready: port=%s slave=%d baud=%d lib_dir=%s percent_limit=%d max_rpm=%d signs(left=%d,right=%d,forward=%d) stop_zero_delay=%.3fs parking_current=%.1fA",
            self.config.port,
            self.config.slave_id,
            self.config.baudrate,
            lib_dir,
            self.config.percent_limit,
            self.config.max_target,
            self.config.left_sign,
            self.config.right_sign,
            self.config.forward_target_sign,
            self.config.stop_zero_delay_sec,
            self.parking_current_a,
        )
        return driver

    def enable_startup_parking(self) -> None:
        if self.driver is None:
            raise RuntimeError("LZ30EMA driver is not initialized")

        target_current_a = float(self.config.parking_current_a)
        if not 0.0 <= target_current_a <= 30.0:
            raise ValueError("parking current must be between 0 A and 30 A")

        # 先清掉驱动器可能保留的速度目标和旧驻车电流，避免启动时直接带着
        # 上一次状态进入锁相。stop_all 会依次操作右、左轮，因此先用急停消除运动。
        self.driver.set_right_speed(0)
        self.driver.set_left_speed(0)
        emergency_value = self.classes[1].EMERGENCY if self.classes else 1
        self.driver.stop_all(emergency_value)
        self.set_parking_current(0.0, persist=False)
        time.sleep(_STARTUP_PARKING_SETTLE_SEC)

        if target_current_a <= 0.0:
            self.logger.info("LZ30EMA 启动驻车未启用: 双轮已清零并急停")
            return

        # 先以较小电流进入锁相，再提升到配置电流。这样即使左右轮的正常停止
        # 不能同时下发，短暂的不对称锁相力矩也不足以让车身猛转。
        prelock_current_a = min(_STARTUP_PARKING_PRELOCK_CURRENT_A, target_current_a)
        self.set_parking_current(prelock_current_a, persist=False)
        normal_value = self.classes[1].NORMAL if self.classes else 0
        self.driver.stop_all(normal_value)
        time.sleep(_STARTUP_PARKING_SETTLE_SEC)
        self.set_parking_current(target_current_a, persist=True)
        self.logger.info("LZ30EMA 启动驻车已开启: 电流=%.1fA 模式=normal", self.parking_current_a)

    def set_parking_current(self, current_a: float, *, persist: bool = True) -> None:
        if self.driver is None:
            raise RuntimeError("LZ30EMA driver is not initialized")
        current_a = float(current_a)
        if not 0.0 <= current_a <= 30.0:
            raise ValueError("parking current must be between 0 A and 30 A")

        for register in ("right_parking_current", "left_parking_current"):
            self.driver.write_register(register, current_a, persist=persist)

        actual_right = float(self.driver.read_register("right_parking_current"))
        actual_left = float(self.driver.read_register("left_parking_current"))
        tolerance_a = 0.005
        if (
            abs(actual_right - current_a) > tolerance_a
            or abs(actual_left - current_a) > tolerance_a
        ):
            raise RuntimeError(
                "parking current readback mismatch: "
                f"requested={current_a:g} A right={actual_right:g} A left={actual_left:g} A"
            )
        self.parking_current_a = current_a
        self.logger.info(
            "LZ30EMA 驻车电流已确认: 右轮=%.1fA 左轮=%.1fA",
            actual_right,
            actual_left,
        )

    def arm_for_motion(self, driver, force: bool = False) -> None:
        if self.motion_armed and not force:
            return
        # LZ-30EMA speed commands do not require the old MSSD arm/parking
        # sequence. Keep this method as a compatibility hook for callers.
        self.motion_armed = True

    def percent_to_target(self, percent: int) -> int:
        return round(int(self.config.max_target) * self.clip_percent(percent) / 100.0)

    def clip_target_rpm(self, target: int, limit: Optional[int] = None) -> int:
        limit = max(
            0,
            int(self.config.max_target) if limit is None else int(limit),
        )
        return max(-limit, min(limit, int(target)))

    def wheel_state_to_target(self, wheel: str, percent: int, state: int) -> int:
        return self.wheel_raw_state_to_target(wheel, self.percent_to_target(percent), state)

    def wheel_raw_state_to_target(self, wheel: str, raw_target: int, state: int) -> int:
        target = max(0, int(raw_target))
        state = int(state) & 0xFF
        if state == 0x01:
            raw = int(self.config.forward_target_sign) * target
        elif state == 0x02:
            raw = -int(self.config.forward_target_sign) * target
        else:
            raw = 0
        sign = int(self.config.left_sign) if wheel == "left" else int(self.config.right_sign)
        return int(raw * sign)

    def send_targets(
        self,
        left_target: int,
        right_target: int,
        label: str,
        *,
        max_target_override: Optional[int] = None,
    ) -> None:
        driver = self.ensure_driver()
        requested_left = int(left_target)
        requested_right = int(right_target)
        target_limit = (
            int(self.config.max_target)
            if max_target_override is None
            else max(0, int(max_target_override))
        )
        left_target = self.clip_target_rpm(requested_left, target_limit)
        right_target = self.clip_target_rpm(requested_right, target_limit)
        if (left_target, right_target) != (requested_left, requested_right):
            self.logger.warning(
                "LZ30EMA 转速已限幅: 标签=%s 请求=(%d,%d)转/分 实际=(%d,%d)转/分 上限=%d转/分",
                label,
                requested_left,
                requested_right,
                left_target,
                right_target,
                target_limit,
            )
        if int(left_target) != 0 or int(right_target) != 0:
            self.arm_for_motion(driver)
        driver.set_right_speed(int(right_target))
        driver.set_left_speed(int(left_target))
        self.logger.info("LZ30EMA 电机命令: 标签=%s 左轮=%d转/分 右轮=%d转/分", label, int(left_target), int(right_target))

    def send_diff(self, m1_percent: int, m1_state: int, m2_percent: int, m2_state: int, label: str) -> None:
        if self.config.m1_is_left_wheel:
            left_target = self.wheel_state_to_target("left", m1_percent, m1_state)
            right_target = self.wheel_state_to_target("right", m2_percent, m2_state)
        else:
            right_target = self.wheel_state_to_target("right", m1_percent, m1_state)
            left_target = self.wheel_state_to_target("left", m2_percent, m2_state)
        self.send_targets(left_target, right_target, label)

    def send_stop(self, label: str = "stop", mode: Optional[str] = None) -> None:
        driver = self.ensure_driver()
        stop_mode = normalize_mssd_stop_mode(mode or self.config.stop_mode, self.config.stop_mode)
        try:
            driver.set_right_speed(0)
            driver.set_left_speed(0)
            if self.config.stop_zero_delay_sec > 0:
                time.sleep(self.config.stop_zero_delay_sec)
        except Exception as exc:
            self.logger.warning("LZ30EMA 停车前双轮清零失败: %s", exc)
        stop_value = {"normal": 0, "emergency": 1, "free": 2}[stop_mode]
        if self.classes:
            stop_value = self.classes[1](stop_value)
        driver.stop_all(stop_value)
        if stop_mode != "normal":
            try:
                driver.set_right_speed(0)
                driver.set_left_speed(0)
            except Exception as exc:
                self.logger.warning("LZ30EMA 停车后双轮清零失败: %s", exc)
        else:
            # NORMAL stop 会让驱动器进入编码器锁相驻车。此后再次写 0 RPM 会把
            # 控制器切回速度闭环，造成驻车速度环持续来回修正，不能再补发零速。
            self.logger.debug("LZ30EMA 正常驻车已保持锁相，跳过停车后零速写入")
        self.motion_armed = False
        self.logger.info("LZ30EMA 停车命令: 标签=%s 模式=%s 清零延时=%.3f秒", label, stop_mode, self.config.stop_zero_delay_sec)

    def close(self) -> None:
        if self.driver is None and self.driver_ctx is None:
            return
        try:
            if self.driver is not None:
                self.send_stop("close")
        except Exception as exc:
            self.logger.warning("LZ30EMA 关闭时停车失败: %s", exc)
        try:
            if self.driver is not None:
                self.set_parking_current(0.0)
        except Exception as exc:
            self.logger.warning("LZ30EMA 关闭时清零驻车电流失败: %s", exc)
        try:
            if self.driver is not None and hasattr(self.driver, "close"):
                self.driver.close()
        finally:
            self.driver = None
            self.driver_ctx = None
            self.classes = None
            self.motion_armed = False
            self.parking_current_a = 0.0
