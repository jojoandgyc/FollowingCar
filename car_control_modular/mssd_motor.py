from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .motor_rtu import install_motor_rtu_guard
from .motor_ramp import (
    RampConfigurationError, configure_closed_loop_ramp, validate_acceleration_rpm_s,
)


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
class MotorSpeedReceipt:
    """Completed dual-wheel speed ACKs, not measured motion or authority.

    Wheel RPMs retain the raw protocol signs. Consumers must bind their own
    target identity and reject a receipt once it is no longer current.
    """

    sequence: int
    left_rpm: int
    right_rpm: int
    completed_at: float


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
    ramp_diagnostics_enable: bool = False
    closed_loop_acceleration_rpm_s: Optional[int] = None

    def __post_init__(self) -> None:
        validate_acceleration_rpm_s(self.closed_loop_acceleration_rpm_s)
        if self.closed_loop_acceleration_rpm_s is not None and not self.startup_parking_enabled:
            raise ValueError("acceleration override requires startup parking/STOP before parameter writes")


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
        # Opt-in transition brake: a zero-speed keepalive must not exit the
        # controller's NORMAL encoder-position hold. A new nonzero command
        # (validated by the runtime) or another explicit stop supersedes it.
        self.normal_zero_hold = False
        self.parking_current_a = 0.0
        self._parking_current_uncertain = False
        self.parking_release_fault = None  # Latched until a new backend/runtime is constructed.
        # A failed speed transaction may already have changed either wheel.
        # Never automatically retry motion after such an ambiguous write.
        self.motion_write_fault = None
        self._rtu_guard = None
        self.ramp_diagnostics = None
        self.last_speed_receipt: Optional[MotorSpeedReceipt] = None
        self._speed_receipt_sequence = 0

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
        # Preserve a failed initialization's transaction state before any
        # attempt to replace its closed client with a fresh, unpoisoned one.
        self.sync_transaction_fault()
        if self.driver is not None:
            return self.driver
        if self.motion_write_fault:
            raise RuntimeError("motor write fault: restart required before reconnect")
        self.last_speed_receipt = None
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
        # A newly connected controller may retain a previous parking current.
        self._parking_current_uncertain = True
        try:
            # Wrap the vendor transaction primitive before startup I/O. All
            # status reads, speed writes and STOPs share the same RTU boundary.
            self._rtu_guard = install_motor_rtu_guard(
                driver, baudrate=self.config.baudrate, timeout=self.config.timeout,
                logger=self.logger,
            )
            if self.config.startup_parking_enabled:
                self.enable_startup_parking()
                # Keep startup parking engaged until an actual speed-mode exit.
                self.motion_armed = False
            else:
                self.arm_for_motion(driver, force=True)
            if (self.config.ramp_diagnostics_enable
                    or self.config.closed_loop_acceleration_rpm_s is not None):
                self._initialize_closed_loop_ramp()
        except Exception:
            self.sync_transaction_fault()
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

    def _initialize_closed_loop_ramp(self) -> None:
        """Startup only: no workers yet; any override follows startup STOP.

        A read-only diagnostic timeout is still an uncertain RTU transaction:
        propagate it through ensure_driver's existing fault/close handling.
        Never add periodic reads to the motor/vision hot path.
        """
        started = time.monotonic()
        try:
            result = configure_closed_loop_ramp(
                self.driver, self.config.closed_loop_acceleration_rpm_s,
            )
        except RampConfigurationError as exc:
            # An acknowledged but unverified parameter write is also unsafe
            # to retry implicitly on a fresh connection to the same backend.
            self.ramp_diagnostics = exc.diagnostics
            self._record_motion_write_fault("startup_ramp_configuration", exc)
            raise
        self.ramp_diagnostics = result
        self.logger.info(
            "LZ30EMA ramp_readback accel_register=0x005F decel_register=0x0060 "
            "before_acceleration_rpm_s=%s acceleration_rpm_s=%s deceleration_rpm_s=%s "
            "requested_acceleration_rpm_s=%s changed=%s persist=False elapsed_ms=%.1f "
            "scope=shared_wheels_and_turns measured_acceleration=False",
            result["before_acceleration_rpm_s"], result["acceleration_rpm_s"],
            result["deceleration_rpm_s"], result["requested_acceleration_rpm_s"],
            result["changed"], (time.monotonic() - started) * 1000,
        )

    def enable_startup_parking(self) -> None:
        if self.driver is None:
            raise RuntimeError("LZ30EMA driver is not initialized")

        target_current_a = float(self.config.parking_current_a)
        if not 0.0 <= target_current_a <= 30.0:
            raise ValueError("parking current must be between 0 A and 30 A")

        # 先清掉驱动器可能保留的速度目标和旧驻车电流，避免启动时直接带着
        # 上一次状态进入锁相。两轮急停分别尝试，不能因一轮无应答跳过另一轮。
        self._write_speed_pair(0, 0, "startup_zero")
        self._checked_stop_wheels("startup_prelock", 1)
        self.set_parking_current(0.0, persist=False)
        time.sleep(_STARTUP_PARKING_SETTLE_SEC)

        if target_current_a <= 0.0:
            self.logger.info("LZ30EMA 启动驻车未启用: 双轮已清零并急停")
            return

        if self.config.stop_mode == "emergency":
            self.set_parking_current(target_current_a, persist=True)
            self._checked_stop_wheels("startup_parking", 1)
            self.logger.info("LZ30EMA 启动驻车已设置: 电流=%.1fA 模式=emergency", self.parking_current_a)
            return

        # 先以较小电流进入锁相，再提升到配置电流。这样即使左右轮的正常停止
        # 不能同时下发，短暂的不对称锁相力矩也不足以让车身猛转。
        prelock_current_a = min(_STARTUP_PARKING_PRELOCK_CURRENT_A, target_current_a)
        self.set_parking_current(prelock_current_a, persist=False)
        self._checked_stop_wheels("startup_parking", 0)
        time.sleep(_STARTUP_PARKING_SETTLE_SEC)
        self.set_parking_current(target_current_a, persist=True)
        self.logger.info("LZ30EMA 启动驻车已开启: 电流=%.1fA 模式=normal", self.parking_current_a)

    def set_parking_current(self, current_a: float, *, persist: bool = True) -> None:
        self.last_speed_receipt = None
        if self.driver is None:
            raise RuntimeError("LZ30EMA driver is not initialized")
        current_a = float(current_a)
        if not 0.0 <= current_a <= 30.0:
            raise ValueError("parking current must be between 0 A and 30 A")

        self._parking_current_uncertain = True
        for register in ("right_parking_current", "left_parking_current"):
            self.driver.write_register(register, current_a, persist=persist)

        actual_right = float(self.driver.read_register("right_parking_current"))
        actual_left = float(self.driver.read_register("left_parking_current"))
        tolerance_a = 0.005
        if (
            not math.isfinite(actual_right) or not math.isfinite(actual_left)
            or abs(actual_right - current_a) > tolerance_a
            or abs(actual_left - current_a) > tolerance_a
        ):
            raise RuntimeError(
                "parking current readback mismatch: "
                f"requested={current_a:g} A right={actual_right:g} A left={actual_left:g} A"
            )
        self.parking_current_a = current_a
        self._parking_current_uncertain = False
        self.logger.info(
            "LZ30EMA 驻车电流已确认: 右轮=%.1fA 左轮=%.1fA",
            actual_right,
            actual_left,
        )

    def arm_for_motion(self, driver, force: bool = False) -> None:
        if self.motion_write_fault:
            self.last_speed_receipt = None
            raise RuntimeError("motor write fault: restart required before motion")
        if getattr(self, "parking_release_fault", None):
            self.last_speed_receipt = None
            raise RuntimeError("parking release fault: restart required before motion")
        self.prepare_speed_mode()
        if self.motion_armed and not force:
            return
        # LZ-30EMA speed commands do not require the old MSSD arm/parking
        # sequence. Keep this method as a compatibility hook for callers.
        self.motion_armed = True

    def prepare_speed_mode(self) -> None:
        """Caller owns motor lock; clear both parking currents before speed I/O.

        Transition-only, volatile writes. Failure leaves motion prohibited and
        is retried, including a partial write with an unchanged cached value.
        Follow executor calls this BEFORE its final authority/TTL checks.
        """
        if self.motion_write_fault:
            self.last_speed_receipt = None
            raise RuntimeError("motor write fault: speed mode prohibited")
        if self._clear_parking_current("speed"):
            self.normal_zero_hold = False

    def release_parking_current_only(self) -> None:
        """End a bounded current hold without writing speed or another STOP.

        Caller owns the motor lock and the ordinary parking episode. This
        does not authorize motion or certify stillness. Suppress zero-speed
        keepalives until a separately authorized nonzero command is sent.
        """
        self.last_speed_receipt = None
        if self.motion_write_fault:
            raise RuntimeError("motor write fault: ordinary parking release prohibited")
        self.motion_armed = False
        self.normal_zero_hold = True
        self._clear_parking_current("park_current_released")

    def _clear_parking_current(self, transition: str) -> bool:
        """Volatile current writes plus dual readback; no speed-mode write."""
        if self.parking_current_a != 0.0 or self._parking_current_uncertain:
            started = time.monotonic()
            try:
                self.set_parking_current(0.0, persist=False)
            except Exception:
                self.motion_armed = False
                self._checked_stop_wheels("parking_current_release_failed", 1)
                raise
            self.logger.info("parking_current_transition state=%s current_a=0 "
                             "elapsed_ms=%.1f persist=False", transition, (time.monotonic()-started)*1000)
            return True
        return False

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

    def _try_stop_wheels(self, label: str, mode: int):
        """Attempt BOTH wheel stops even when one acknowledgement fails.

        Caller owns motor I/O exclusion. The production driver exposes stop
        per side; the all-wheel fallback supports older test/adapter drivers.
        No speed-mode/current transition is permitted on a fault refresh.
        """
        self.last_speed_receipt = None
        self.motion_armed = False
        stop = getattr(self.driver, "stop", None)
        stop_value = self.classes[1](mode) if self.classes else mode
        failures = []
        for side in ("right", "left") if callable(stop) else ("all",):
            try:
                if callable(stop):
                    stop(side, stop_value)
                else:
                    self.driver.stop_all(stop_value)
            except Exception as exc:
                failures.append((side, exc))
                self.logger.error("motor_stop_side_failed label=%s side=%s error=%s "
                                  "motion_authorized=False", label, side, exc)
        return failures

    def _record_motion_write_fault(self, label: str, exc: Exception) -> bool:
        self.last_speed_receipt = None
        if self.motion_write_fault is not None:
            return False
        self.motion_write_fault = "%s:%s:%s" % (label, type(exc).__name__, exc)
        self.motion_armed = False
        self.normal_zero_hold = False
        self.logger.error("motor_write_fault label=%s error=%s recovery=manual_restart "
                          "motion_authorized=False", label, exc)
        return True

    def sync_transaction_fault(self) -> None:
        """Surface even a feedback-read link fault to the independent executor.

        No I/O here: the caller may be outside motor_io_lock. A successful
        STOP or diagnostic read cannot make an uncertain RTU stream healthy.
        """
        guard = self._rtu_guard
        if guard is not None and guard.rx_uncertain and not self.motion_write_fault:
            self._record_motion_write_fault(
                "rs485_transaction", RuntimeError(guard.fault_reason or "uncertain_response_stream"),
            )

    def _write_speed_pair(self, left_target: int, right_target: int, label: str) -> None:
        # Invalidate before the first side can change. A partial transfer or
        # an in-flight write must never leave the preceding pair usable.
        self.last_speed_receipt = None
        started = time.monotonic()
        side = "right"
        right_acknowledged = False
        try:
            self.driver.set_right_speed(int(right_target))
            right_acknowledged = True
            side = "left"
            self.driver.set_left_speed(int(left_target))
        except Exception as exc:
            self.logger.error(
                "motor_pair_write_failed label=%s failed_side=%s requested_left_rpm=%d "
                "requested_right_rpm=%d right_acknowledged=%s pair_elapsed_ms=%.3f "
                "physical_execution=unknown error=%s",
                label, side, left_target, right_target, right_acknowledged,
                (time.monotonic() - started) * 1000.0, exc,
            )
            self._latch_motion_write_fault(label, exc)
            raise
        if not self.motion_write_fault and not self.parking_release_fault:
            self._speed_receipt_sequence += 1
            self.last_speed_receipt = MotorSpeedReceipt(
                self._speed_receipt_sequence, int(left_target), int(right_target),
                time.monotonic(),
            )

    def _checked_stop_wheels(self, label: str, mode: int) -> None:
        failures = self._try_stop_wheels(label, mode)
        if failures:
            # A STOP acknowledgement failure is not a reason to enter speed
            # mode with 0RPM. Preserve the original exception after trying BOTH
            # sides. Failed NORMAL/FREE falls back to dual EMERGENCY only.
            exc = failures[0][1]
            self._record_motion_write_fault(label + ":stop", exc)
            if mode != 1:
                self._fault_emergency_stop(label + ":stop_fallback")
            raise exc

    def _fault_emergency_stop(self, label: str) -> bool:
        failures = self._try_stop_wheels(label, 1)
        self.logger.error("motor_fault_stop_attempt label=%s failed_sides=%s "
                          "stop_writes_complete=%s physical_stillness=unverified "
                          "recovery=manual_restart", label,
                          [side for side, _ in failures], not failures)
        return not failures

    def _latch_motion_write_fault(self, label: str, exc: Exception) -> None:
        """One bounded rollback attempt; zero each side, then STOP each side."""
        if not self._record_motion_write_fault(label, exc):
            return
        for side in ("right", "left"):
            try:
                getattr(self.driver, "set_%s_speed" % side)(0)
            except Exception as zero_exc:
                self.logger.error("motor_fault_zero_failed side=%s error=%s "
                                  "continuing_other_wheel_and_stop=True", side, zero_exc)
        self._fault_emergency_stop(label)

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
        self.last_speed_receipt = None
        requested_left = int(left_target)
        requested_right = int(right_target)
        if self.motion_write_fault:
            if requested_left or requested_right:
                raise RuntimeError("motor write fault: nonzero wheel command blocked")
            self.send_stop(label, mode="emergency")
            return
        driver = self.ensure_driver()
        if getattr(self, "parking_release_fault", None):
            if requested_left or requested_right:
                raise RuntimeError("parking release fault: nonzero wheel command blocked")
            # Zero RPM is a speed-mode command too. A stale zero keepalive
            # must not supersede the latched fault STOP or clear its current.
            self.send_stop(label, mode="emergency")
            return
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
        if self.normal_zero_hold and left_target == right_target == 0:
            self.logger.debug("cross_brake_zero_preserved label=%s", label)
            return
        self.prepare_speed_mode()
        if int(left_target) != 0 or int(right_target) != 0:
            # A partial serial write may already release one wheel. Do not
            # claim NORMAL is still held if the following write then fails.
            self.normal_zero_hold = False
            self.arm_for_motion(driver)
        self._write_speed_pair(left_target, right_target, label)
        self.normal_zero_hold = False
        self.logger.info("LZ30EMA 电机命令: 标签=%s 左轮=%d转/分 右轮=%d转/分", label, int(left_target), int(right_target))

    def send_diff(self, m1_percent: int, m1_state: int, m2_percent: int, m2_state: int, label: str) -> None:
        if self.config.m1_is_left_wheel:
            left_target = self.wheel_state_to_target("left", m1_percent, m1_state)
            right_target = self.wheel_state_to_target("right", m2_percent, m2_state)
        else:
            right_target = self.wheel_state_to_target("right", m1_percent, m1_state)
            left_target = self.wheel_state_to_target("left", m2_percent, m2_state)
        self.send_targets(left_target, right_target, label)

    def send_stop(self, label: str = "stop", mode: Optional[str] = None,
                  *, preserve_zero: bool = False, prepare_parking_current: bool = False) -> None:
        self.last_speed_receipt = None
        driver = self.ensure_driver()
        if self.motion_write_fault:
            self._checked_stop_wheels(label, 1)
            return
        stop_mode = normalize_mssd_stop_mode(mode or self.config.stop_mode, self.config.stop_mode)
        if getattr(self, "parking_release_fault", None):
            stop_mode = "emergency"  # Never re-enter the faulty 5A hold.
        self.normal_zero_hold = False
        # Disarm even if either STOP write fails. Only log successful dispatch
        # after both acknowledgements; this is not proof of physical stillness.
        self.motion_armed = False
        pre_zero = stop_mode == "normal"
        if pre_zero:
            try:
                driver.set_right_speed(0)
                driver.set_left_speed(0)
                if self.config.stop_zero_delay_sec > 0:
                    time.sleep(self.config.stop_zero_delay_sec)
            except Exception as exc:
                self.logger.warning("LZ30EMA 停车前双轮清零失败: %s", exc)
        stop_value = {"normal": 0, "emergency": 1, "free": 2}[stop_mode]
        if stop_mode == "normal":
            # End the preceding speed-control sequence before preparing NORMAL.
            # Failure (including a partial dual-wheel write) must abort entry:
            # neither current setup nor NORMAL may follow an unacknowledged stop.
            # Held-NORMAL refreshes deliberately bypass this transition.
            emergency_started = time.monotonic()
            self._checked_stop_wheels(label + ":normal_pre_emergency", 1)
            self.logger.info(
                "normal_pre_emergency label=%s stop_registers_sent=0x0040,0x0044 "
                "stop_value=1 stop_write_ms=%.1f physical_stillness=unverified",
                label, (time.monotonic() - emergency_started) * 1000.,
            )
        # Only an ordinary parking entry opts in. Safety EMERGENCY must remain
        # STOP-only, without waiting for current register transactions.
        if stop_mode == "normal" or (prepare_parking_current
                                      and not getattr(self, "parking_release_fault", None)):
            try:
                if self.parking_current_a != self.config.parking_current_a or self._parking_current_uncertain:
                    started = time.monotonic()
                    self.set_parking_current(self.config.parking_current_a, persist=False)
                    self.logger.info("parking_current_transition state=park current_a=%.1f "
                                     "elapsed_ms=%.1f persist=False", self.parking_current_a,
                                     (time.monotonic()-started)*1000)
            except Exception:
                # Failed parking preparation must not silently become motion.
                self._checked_stop_wheels(label + ":parking_prepare_failed", 1)
                self.motion_armed = False
                raise
        started = time.monotonic()
        self._checked_stop_wheels(label, stop_value)
        # Legacy name: an explicitly preserved ordinary stop may also use
        # EMERGENCY or FREE. Zero keepalives must not re-enter speed mode.
        self.normal_zero_hold = bool(preserve_zero)
        # STOP is the final motor-mode write for every stop mode. In particular,
        # EMERGENCY/FREE refreshes are STOP-only: no temporary speed-mode entry,
        # no zero-speed delay, and no zero write after STOP. The stop registers
        # (0x0040/0x0044) are write-only; do not pretend to read back a latch.
        self.logger.info(
            "LZ30EMA 停车命令: 标签=%s 模式=%s 清零延时=%.3f秒 "
            "parking_current_a=%.1f current_source=transition_readback_cache "
            "current_uncertain=%s stop_registers_sent=0x0040,0x0044 stop_value=%d "
            "pre_zero=%s post_zero=False final_motor_command=stop "
            "stop_write_ms=%.1f physical_stillness=unverified",
            label, stop_mode, self.config.stop_zero_delay_sec if pre_zero else 0.,
            self.parking_current_a, self._parking_current_uncertain, int(stop_value),
            pre_zero, (time.monotonic() - started) * 1000.,
        )

    def refresh_normal_stop(self, label: str) -> None:
        """Refresh an already applied NORMAL hold without entering speed mode."""
        self.last_speed_receipt = None
        if self.motion_write_fault or getattr(self, "parking_release_fault", None):
            self.send_stop(label, mode="emergency")
            return
        self.ensure_driver()
        self._checked_stop_wheels(label, 0)
        self.motion_armed = False
        self.logger.info("LZ30EMA 驻车刷新: 标签=%s 模式=normal pre_zero=False", label)

    def close(self) -> None:
        self.last_speed_receipt = None
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
