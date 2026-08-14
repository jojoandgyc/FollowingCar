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
        self.arm_for_motion(driver, force=True)
        self.logger.info(
            "LZ30EMA motor backend ready: port=%s slave=%d baud=%d lib_dir=%s percent_limit=%d max_rpm=%d signs(left=%d,right=%d,forward=%d) stop_zero_delay=%.3fs",
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
        )
        return driver

    def arm_for_motion(self, driver, force: bool = False) -> None:
        if self.motion_armed and not force:
            return
        # LZ-30EMA speed commands do not require the old MSSD arm/parking
        # sequence. Keep this method as a compatibility hook for callers.
        self.motion_armed = True

    def percent_to_target(self, percent: int) -> int:
        return round(int(self.config.max_target) * self.clip_percent(percent) / 100.0)

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

    def send_targets(self, left_target: int, right_target: int, label: str) -> None:
        driver = self.ensure_driver()
        if int(left_target) != 0 or int(right_target) != 0:
            self.arm_for_motion(driver)
        driver.set_right_speed(int(right_target))
        driver.set_left_speed(int(left_target))
        self.logger.info("LZ30EMA command %s left=%dRPM right=%dRPM", label, int(left_target), int(right_target))

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
            self.logger.warning("LZ30EMA zero speeds before stop failed: %s", exc)
        stop_value = {"normal": 0, "emergency": 1, "free": 2}[stop_mode]
        if self.classes:
            stop_value = self.classes[1](stop_value)
        driver.stop_all(stop_value)
        try:
            driver.set_right_speed(0)
            driver.set_left_speed(0)
        except Exception as exc:
            self.logger.warning("LZ30EMA zero speeds after stop failed: %s", exc)
        self.motion_armed = False
        self.logger.info("LZ30EMA stop %s mode=%s zero_delay=%.3fs", label, stop_mode, self.config.stop_zero_delay_sec)

    def close(self) -> None:
        if self.driver is None and self.driver_ctx is None:
            return
        try:
            if self.driver is not None:
                self.send_stop("close")
        except Exception as exc:
            self.logger.warning("LZ30EMA stop on close failed: %s", exc)
        try:
            if self.driver is not None and hasattr(self.driver, "close"):
                self.driver.close()
        finally:
            self.driver = None
            self.driver_ctx = None
            self.classes = None
            self.motion_armed = False
