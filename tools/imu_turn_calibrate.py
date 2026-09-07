#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "car_control_modular" / "config" / "reid_runtime.ini"
MAX_SAFE_RPM = 16
MAX_SAFE_PULSE_SEC = 0.80
MAX_SAFE_VERIFY_ANGLE_DEG = 30.0
MAX_GYRO_STALE_SEC = 0.18
MAX_STATIC_YAW_STD_DPS = 1.0
MAX_STATIC_YAW_PEAK_DPS = 3.0
ENCODER_DEGREES_PER_REV = 360.0
ENCODER_RESOLUTION_DEG = 1.0
# ABZ 编码器实车符号：左转双轮为负，右转双轮为正。
TURN_ENCODER_SIGN = {"left": -1, "right": 1}


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) >= 2 else 0.0


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _normalize(vector: Sequence[float]) -> Tuple[float, float, float]:
    length = _norm(vector)
    if length <= 1e-9:
        raise RuntimeError("无法从加速度计确定重力轴：向量长度接近 0")
    return tuple(float(value) / length for value in vector)  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _sign(value: float) -> int:
    return 1 if value >= 0.0 else -1


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _unwrap_i32_delta(after: int, before: int) -> int:
    """Return a signed encoder delta even if the 32-bit degree counter wrapped."""
    return int((int(after) - int(before) + (1 << 31)) % (1 << 32) - (1 << 31))


def _normalize_turn_encoder(
    direction: str,
    left_delta_deg: int,
    right_delta_deg: int,
) -> Tuple[int, float, float, float, float, bool]:
    """Normalize both ABZ encoders into positive progress for one turn direction."""
    if direction not in TURN_ENCODER_SIGN:
        raise ValueError("direction 必须是 left 或 right")
    expected_sign = int(TURN_ENCODER_SIGN[direction])
    left_progress = expected_sign * float(left_delta_deg)
    right_progress = expected_sign * float(right_delta_deg)
    mean_progress = (left_progress + right_progress) / 2.0
    denominator = max(1.0, (abs(left_progress) + abs(right_progress)) / 2.0)
    balance_error = abs(abs(left_progress) - abs(right_progress)) / denominator
    sign_valid = left_progress > 0.0 and right_progress > 0.0
    return (
        expected_sign,
        left_progress,
        right_progress,
        mean_progress,
        balance_error,
        sign_valid,
    )


def _parse_csv_ints(raw: str) -> List[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("至少需要一个 RPM")
    if any(value <= 0 or value > MAX_SAFE_RPM for value in values):
        raise ValueError(f"RPM 必须在 1..{MAX_SAFE_RPM} 范围内")
    return values


def _parse_csv_floats(raw: str) -> List[float]:
    values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("至少需要一个脉冲时长")
    if any(value <= 0.05 or value > MAX_SAFE_PULSE_SEC for value in values):
        raise ValueError(f"脉冲时长必须在 0.05..{MAX_SAFE_PULSE_SEC:.2f} 秒范围内")
    return values


def _ensure_follow_runtime_stopped() -> None:
    own_pid = os.getpid()
    conflicts: List[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (OSError, PermissionError):
            continue
        if "request_0513_modular.py" in cmdline or "run_request_0428_modular.sh" in cmdline:
            conflicts.append(f"pid={entry.name} {cmdline.strip()}")
    if conflicts:
        raise RuntimeError("主跟随程序仍在运行，拒绝占用电机和 IMU：\n" + "\n".join(conflicts))

    motor_port = Path(os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0"))
    port_owners: List[str] = []
    if motor_port.exists():
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == own_pid:
                continue
            fd_dir = entry / "fd"
            try:
                descriptors = list(fd_dir.iterdir())
            except (OSError, PermissionError):
                continue
            owns_port = False
            for descriptor in descriptors:
                try:
                    if os.path.samefile(descriptor, motor_port):
                        owns_port = True
                        break
                except (OSError, PermissionError):
                    continue
            if not owns_port:
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except (OSError, PermissionError):
                cmdline = "<无法读取命令行>"
            port_owners.append(f"pid={entry.name} {cmdline.strip()}")
    if port_owners:
        raise RuntimeError(
            f"电机串口 {motor_port} 已被其他进程占用，拒绝产生冲突命令：\n"
            + "\n".join(port_owners)
        )


def _require_motion_confirmation(args: argparse.Namespace, summary: str) -> None:
    if not bool(args.execute):
        raise RuntimeError("该模式会驱动车轮。确认场地安全后添加 --execute")
    print("\n安全确认：", summary)
    print("请让车轮落在正式使用的平整地面上，清空小车周围至少 1 米区域，并准备急停。")
    print("车轮架空时车身不会产生真实 yaw，只能检查电机方向，不能生成可信标定模型。")
    answer = input("输入 TURN 后按回车开始：").strip()
    if answer != "TURN":
        raise RuntimeError("未收到 TURN，已取消，电机不会启动")


@dataclass
class StaticCalibration:
    duration_sec: float
    accel_samples: int
    gyro_samples: int
    gravity_g: Tuple[float, float, float]
    gravity_norm_g: float
    vertical_axis_sensor: Tuple[float, float, float]
    gyro_bias_dps: Tuple[float, float, float]
    gyro_std_dps: Tuple[float, float, float]
    yaw_bias_dps: float
    yaw_std_dps: float
    max_abs_yaw_noise_dps: float
    trustworthy: bool
    reasons: List[str]


@dataclass
class MotorReading:
    timestamp: float
    left_position_deg: int
    right_position_deg: int
    left_speed_rpm: int
    right_speed_rpm: int
    left_error: int
    right_error: int


@dataclass
class TurnTrial:
    index: int
    direction: str
    command_rpm: int
    left_command_rpm: int
    right_command_rpm: int
    command_duration_sec: float
    imu_samples: int
    imu_gap_count: int
    yaw_at_stop_deg: float
    yaw_final_deg: float
    coast_yaw_deg: float
    settle_elapsed_sec: float
    stationary_confirmed: bool
    final_abs_yaw_rate_dps: float
    peak_abs_yaw_rate_dps: float
    encoder_expected_sign: int
    encoder_sign_valid: bool
    left_encoder_before_deg: int
    right_encoder_before_deg: int
    left_encoder_at_stop_deg: int
    right_encoder_at_stop_deg: int
    left_encoder_final_deg: int
    right_encoder_final_deg: int
    left_encoder_at_stop_delta_deg: int
    right_encoder_at_stop_delta_deg: int
    left_encoder_coast_delta_deg: int
    right_encoder_coast_delta_deg: int
    left_encoder_delta_deg: int
    right_encoder_delta_deg: int
    left_wheel_rev: float
    right_wheel_rev: float
    mean_abs_wheel_rev: float
    encoder_balance_error: float
    valid: bool
    invalid_reasons: List[str]


class YawIntegrator:
    def __init__(self, calibration: StaticCalibration) -> None:
        self.bias = calibration.gyro_bias_dps
        self.axis = calibration.vertical_axis_sensor
        self.last_timestamp: Optional[float] = None
        self.last_rate_dps = 0.0
        self.angle_deg = 0.0
        self.sample_count = 0
        self.gap_count = 0
        self.peak_abs_rate_dps = 0.0
        self.last_sample_wall_time: Optional[float] = None

    def feed(self, gyro: Optional[Dict[str, Any]]) -> bool:
        if not gyro:
            return False
        timestamp = float(gyro.get("timestamp", 0.0))
        if timestamp <= 0.0 or timestamp == self.last_timestamp:
            return False
        values = tuple(float(value) for value in gyro.get("dps", (0.0, 0.0, 0.0)))
        corrected = tuple(values[index] - self.bias[index] for index in range(3))
        rate_dps = _dot(corrected, self.axis)
        self.peak_abs_rate_dps = max(self.peak_abs_rate_dps, abs(rate_dps))

        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if 0.0 < dt <= 0.12:
                self.angle_deg += 0.5 * (self.last_rate_dps + rate_dps) * dt
            elif dt > 0.12:
                # 采样间断时不跨空白区积分，避免凭旧角速度虚构角度。
                self.gap_count += 1
        self.last_timestamp = timestamp
        self.last_rate_dps = rate_dps
        self.sample_count += 1
        self.last_sample_wall_time = time.monotonic()
        return True

    def sample_age_sec(self) -> float:
        if self.last_sample_wall_time is None:
            return math.inf
        return max(0.0, time.monotonic() - self.last_sample_wall_time)


class HardwareSession:
    def __init__(self) -> None:
        self.imu = None
        self.backend = None
        self.previous_parking_current: Optional[Tuple[float, float]] = None

    def open(self) -> None:
        from imu_hal import IMU
        from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig

        if IMU.init() != 0:
            raise RuntimeError(f"IMU 初始化失败：{IMU.info()}")
        self.imu = IMU
        config = MssdMotorConfig(
            port=os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0"),
            slave_id=int(os.environ.get("MOTOR_RS485_SLAVE_ID", "1")),
            baudrate=int(os.environ.get("MOTOR_RS485_BAUDRATE", "115200")),
            timeout=float(os.environ.get("MOTOR_RS485_TIMEOUT", "0.15")),
            lib_dir=os.environ.get("MOTOR_RS485_LIB_DIR", "/home/topeet/lianzhan"),
            max_target=int(os.environ.get("MOTOR_RS485_MAX_TARGET", "30")),
            percent_limit=int(os.environ.get("MOTOR_PERCENT_LIMIT", "100")),
            left_sign=int(os.environ.get("MOTOR_LEFT_SIGN", "-1")),
            right_sign=int(os.environ.get("MOTOR_RIGHT_SIGN", "1")),
            forward_target_sign=int(os.environ.get("MOTOR_FORWARD_TARGET_SIGN", "-1")),
            m1_is_left_wheel=_bool_env("M1_IS_LEFT_WHEEL", "0"),
            exit_parking_mode_on_arm=_bool_env("MOTOR_EXIT_PARKING_MODE_ON_ARM", "0"),
            stop_mode=os.environ.get("MOTOR_RS485_STOP_MODE", "normal"),
            stop_zero_delay_sec=float(os.environ.get("MOTOR_RS485_STOP_ZERO_DELAY_SEC", "0")),
        )
        self.backend = MssdMotorBackend(config)
        self.backend.ensure_driver()

    def _read_register_with_retry(self, name: str) -> float:
        if self.backend is None:
            raise RuntimeError("电机后端未初始化")
        driver = self.backend.ensure_driver()
        last_error: Optional[Exception] = None
        for attempt in range(4):
            try:
                return float(driver.read_register(name))
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(0.03 * (attempt + 1))
        raise RuntimeError(f"读取寄存器 {name} 连续4次失败：{last_error}") from last_error

    def _write_register_with_retry(self, name: str, value: float) -> None:
        if self.backend is None:
            raise RuntimeError("电机后端未初始化")
        driver = self.backend.ensure_driver()
        last_error: Optional[Exception] = None
        for attempt in range(4):
            try:
                driver.write_register(name, float(value), persist=False)
                return
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(0.03 * (attempt + 1))
        raise RuntimeError(f"写入寄存器 {name} 连续4次失败：{last_error}") from last_error

    def disable_parking_for_calibration(self) -> None:
        """Temporarily remove lock-phase current so it cannot vibrate the IMU at rest."""
        right = self._read_register_with_retry("right_parking_current")
        left = self._read_register_with_retry("left_parking_current")
        self.previous_parking_current = (right, left)
        self._write_register_with_retry("right_parking_current", 0.0)
        self._write_register_with_retry("left_parking_current", 0.0)
        right_readback = self._read_register_with_retry("right_parking_current")
        left_readback = self._read_register_with_retry("left_parking_current")
        if abs(right_readback) > 0.01 or abs(left_readback) > 0.01:
            raise RuntimeError(
                f"标定前关闭驻车电流失败，回读 right={right_readback}A left={left_readback}A"
            )
        print(f"标定期间驻车电流已临时关闭（原值 right={right:g}A left={left:g}A）")

    def restore_parking_current(self) -> None:
        previous = self.previous_parking_current
        # 测试/标定程序退出后始终关闭驻车电流，避免异常退出路径留下锁相扭矩。
        self._write_register_with_retry("right_parking_current", 0.0)
        self._write_register_with_retry("left_parking_current", 0.0)
        right = self._read_register_with_retry("right_parking_current")
        left = self._read_register_with_retry("left_parking_current")
        if abs(right) > 0.01 or abs(left) > 0.01:
            raise RuntimeError(
                f"退出时关闭驻车电流失败，回读 right={right:g}A left={left:g}A"
            )
        self.previous_parking_current = None
        if previous is None:
            print(f"测试结束，驻车电流已关闭 right={right:g}A left={left:g}A")
        else:
            previous_right, previous_left = previous
            print(
                "标定结束，驻车电流已关闭 "
                f"right={right:g}A left={left:g}A（原值 right={previous_right:g}A left={previous_left:g}A）"
            )

    def read_motors(self) -> MotorReading:
        if self.backend is None:
            raise RuntimeError("电机后端未初始化")
        driver = self.backend.ensure_driver()
        statuses: Dict[str, Any] = {}
        for side in ("left", "right"):
            last_error: Optional[Exception] = None
            for attempt in range(4):
                try:
                    statuses[side] = driver.read_motor_status(side)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        # RS485 偶发 CRC 坏帧只重试当前只读请求；运动命令不会在这里重发。
                        time.sleep(0.02 * (attempt + 1))
            if last_error is not None:
                raise RuntimeError(f"读取{side}电机状态连续4次失败：{last_error}") from last_error
        left = statuses["left"]
        right = statuses["right"]
        return MotorReading(
            timestamp=time.monotonic(),
            left_position_deg=int(left.position_degree),
            right_position_deg=int(right.position_degree),
            left_speed_rpm=int(left.speed_rpm),
            right_speed_rpm=int(right.speed_rpm),
            left_error=int(left.error_code),
            right_error=int(right.error_code),
        )

    def _send_targets_with_retry(self, left: int, right: int, label: str) -> None:
        if self.backend is None:
            raise RuntimeError("电机后端未初始化")
        last_error: Optional[Exception] = None
        for attempt in range(4):
            try:
                self.backend.send_targets(int(left), int(right), label)
                if attempt:
                    print(f"RS485 写入在第 {attempt + 1} 次尝试恢复：{label}")
                return
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    # 速度目标写入是幂等操作；重发相同目标不会叠加转速或位移。
                    time.sleep(0.03 * (attempt + 1))
        raise RuntimeError(f"RS485 目标写入连续4次失败({label})：{last_error}") from last_error

    def command_turn(self, direction: str, rpm: int) -> Tuple[int, int]:
        if self.backend is None:
            raise RuntimeError("电机后端未初始化")
        if not 1 <= int(rpm) <= MAX_SAFE_RPM:
            raise ValueError(f"转向 RPM 必须在 1..{MAX_SAFE_RPM}")
        if direction not in TURN_ENCODER_SIGN:
            raise ValueError("direction 必须是 left 或 right")
        # 直接按实车 ABZ 方向下发，避免 wheel state/安装符号再次把左右翻转：
        # 左转=(-RPM,-RPM)，右转=(+RPM,+RPM)。
        target = int(TURN_ENCODER_SIGN[direction]) * int(rpm)
        left = target
        right = target
        self._send_targets_with_retry(left, right, f"imu_calibrate_{direction}_{rpm}rpm")
        return int(left), int(right)

    def stop(self, label: str = "imu_calibrate_stop") -> None:
        if self.backend is None:
            return
        last_error: Optional[Exception] = None
        for attempt in range(4):
            try:
                self.backend.send_stop(label, mode="emergency")
                if attempt:
                    print(f"RS485 急停在第 {attempt + 1} 次尝试恢复：{label}")
                return
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(0.03 * (attempt + 1))
        raise RuntimeError(f"RS485 急停连续4次失败({label})：{last_error}") from last_error

    def zero_targets(self, label: str = "imu_calibrate_zero_targets") -> None:
        """Request zero speed without changing stop mode or resetting telemetry state."""
        if self.backend is None:
            raise RuntimeError("电机后端未初始化")
        self._send_targets_with_retry(0, 0, label)

    def close(self, ensure_stop: bool) -> None:
        if self.backend is not None:
            try:
                if ensure_stop:
                    self.stop("imu_calibrate_final_stop")
            finally:
                try:
                    self.restore_parking_current()
                except Exception as exc:
                    print(f"警告：退出时关闭驻车电流失败：{exc}", file=sys.stderr)
                finally:
                    driver = self.backend.driver
                    if driver is not None and hasattr(driver, "close"):
                        driver.close()
                    self.backend.driver = None
        if self.imu is not None:
            self.imu.deinit()


def _collect_static(session: HardwareSession, duration_sec: float) -> StaticCalibration:
    if session.imu is None:
        raise RuntimeError("IMU 未初始化")
    accel_rows: List[Tuple[float, float, float]] = []
    gyro_rows: List[Tuple[float, float, float]] = []
    last_accel_ts: Optional[float] = None
    last_gyro_ts: Optional[float] = None
    deadline = time.monotonic() + max(1.0, float(duration_sec))
    while time.monotonic() < deadline:
        snapshot = session.imu.poll(0.02)
        accel = snapshot.get("accel")
        gyro = snapshot.get("gyro")
        if accel and float(accel.get("timestamp", 0.0)) != last_accel_ts:
            last_accel_ts = float(accel["timestamp"])
            accel_rows.append(tuple(float(value) for value in accel.get("g", (0.0, 0.0, 0.0))))
        if gyro and float(gyro.get("timestamp", 0.0)) != last_gyro_ts:
            last_gyro_ts = float(gyro["timestamp"])
            gyro_rows.append(tuple(float(value) for value in gyro.get("dps", (0.0, 0.0, 0.0))))

    reasons: List[str] = []
    if len(accel_rows) < 30:
        reasons.append(f"加速度样本不足({len(accel_rows)})")
    if len(gyro_rows) < 30:
        reasons.append(f"陀螺仪样本不足({len(gyro_rows)})")
    gravity = tuple(_mean([row[index] for row in accel_rows]) for index in range(3))
    gyro_bias = tuple(_mean([row[index] for row in gyro_rows]) for index in range(3))
    gyro_std = tuple(_std([row[index] for row in gyro_rows]) for index in range(3))
    gravity_norm = _norm(gravity)
    if not 0.85 <= gravity_norm <= 1.15:
        reasons.append(f"静止重力模长异常({gravity_norm:.3f}g)，车可能在晃动")
    # 传感器在车上有明显倾角，不能把 gyro Z 直接当车体 yaw。
    # 静止时的重力方向就是车体竖直轴，将三轴角速度投影到该轴才是实际左右转角速度。
    vertical = _normalize(gravity)
    yaw_values = [_dot(tuple(row[index] - gyro_bias[index] for index in range(3)), vertical) for row in gyro_rows]
    yaw_std = _std(yaw_values)
    max_abs_yaw = max((abs(value) for value in yaw_values), default=0.0)
    # 实车当前静止噪声约 0.82°/s，但峰值稳定低于 1.6°/s；短时小角度
    # 积分误差仍在容差内。标准差门槛保留余量，峰值保护继续严格限制尖峰。
    if yaw_std > MAX_STATIC_YAW_STD_DPS:
        reasons.append(f"静止 yaw 噪声过大(std={yaw_std:.3f}°/s)")
    if max_abs_yaw > MAX_STATIC_YAW_PEAK_DPS:
        reasons.append(f"静止 yaw 峰值过大(max={max_abs_yaw:.3f}°/s)")
    return StaticCalibration(
        duration_sec=float(duration_sec),
        accel_samples=len(accel_rows),
        gyro_samples=len(gyro_rows),
        gravity_g=gravity,
        gravity_norm_g=gravity_norm,
        vertical_axis_sensor=vertical,
        gyro_bias_dps=gyro_bias,
        gyro_std_dps=gyro_std,
        yaw_bias_dps=_dot(gyro_bias, vertical),
        yaw_std_dps=yaw_std,
        max_abs_yaw_noise_dps=max_abs_yaw,
        trustworthy=not reasons,
        reasons=reasons,
    )


def _feed_for_duration(session: HardwareSession, integrator: YawIntegrator, duration_sec: float) -> None:
    if session.imu is None:
        raise RuntimeError("IMU 未初始化")
    deadline = time.monotonic() + max(0.0, float(duration_sec))
    while time.monotonic() < deadline:
        snapshot = session.imu.poll(0.01)
        integrator.feed(snapshot.get("gyro"))


def _prime_integrator(session: HardwareSession, integrator: YawIntegrator) -> None:
    _feed_for_duration(session, integrator, 0.08)


def _feed_until_stationary(
    session: HardwareSession,
    integrator: YawIntegrator,
    *,
    min_settle_sec: float,
    settle_timeout_sec: float,
    quiet_sec: float,
    quiet_yaw_rate_dps: float,
) -> Tuple[bool, float, MotorReading]:
    """Integrate coast motion until gyro and both encoder speeds are quiet."""
    if session.imu is None:
        raise RuntimeError("IMU 未初始化")
    started = time.monotonic()
    quiet_started_at: Optional[float] = None
    minimum = max(0.0, float(min_settle_sec))
    required_quiet = max(0.10, float(quiet_sec))
    timeout = max(minimum + required_quiet, float(settle_timeout_sec))
    rate_limit = max(0.10, float(quiet_yaw_rate_dps))

    while True:
        snapshot = session.imu.poll(0.01)
        integrator.feed(snapshot.get("gyro"))
        now = time.monotonic()
        elapsed = now - started
        rate_quiet = abs(float(integrator.last_rate_dps)) <= rate_limit
        if elapsed >= minimum and rate_quiet:
            if quiet_started_at is None:
                quiet_started_at = now
            elif now - quiet_started_at >= required_quiet:
                motors = session.read_motors()
                motor_quiet = abs(motors.left_speed_rpm) <= 1 and abs(motors.right_speed_rpm) <= 1
                if motor_quiet and not motors.left_error and not motors.right_error:
                    return True, elapsed, motors
                quiet_started_at = None
        else:
            quiet_started_at = None

        if elapsed >= timeout:
            return False, elapsed, session.read_motors()


def _run_trial(
    session: HardwareSession,
    calibration: StaticCalibration,
    index: int,
    direction: str,
    rpm: int,
    duration_sec: float,
    settle_sec: float,
    settle_timeout_sec: float,
    quiet_sec: float,
    quiet_yaw_rate_dps: float,
) -> TurnTrial:
    session.stop("imu_trial_prepare")
    time.sleep(0.20)
    before = session.read_motors()
    if before.left_error or before.right_error:
        raise RuntimeError(f"电机存在故障码：left={before.left_error} right={before.right_error}")
    if abs(before.left_speed_rpm) > 1 or abs(before.right_speed_rpm) > 1:
        raise RuntimeError(f"试验前车轮未停止：left={before.left_speed_rpm} right={before.right_speed_rpm}")

    integrator = YawIntegrator(calibration)
    _prime_integrator(session, integrator)
    left_command = 0
    right_command = 0
    at_stop: Optional[MotorReading] = None
    zero_targets_sent = False
    try:
        left_command, right_command = session.command_turn(direction, rpm)
        _feed_for_duration(session, integrator, duration_sec)
        yaw_at_stop = integrator.angle_deg
        # 先把速度目标清零，避免偶发 CRC 重试期间继续旋转；清零目标不会切换停止模式。
        session.zero_targets(f"imu_trial_{index}_zero_targets")
        zero_targets_sent = True
        at_stop = session.read_motors()
    finally:
        if not zero_targets_sent:
            yaw_at_stop = integrator.angle_deg
            try:
                session.zero_targets(f"imu_trial_{index}_zero_targets_finally")
            except Exception:
                session.stop(f"imu_trial_{index}_zero_failed_stop")
                raise
    if at_stop is None:
        raise RuntimeError("未取得停车前编码器读数")
    try:
        stationary, settle_elapsed, after = _feed_until_stationary(
            session,
            integrator,
            min_settle_sec=settle_sec,
            settle_timeout_sec=settle_timeout_sec,
            quiet_sec=quiet_sec,
            quiet_yaw_rate_dps=quiet_yaw_rate_dps,
        )
    finally:
        # 完成所有位置读取后再进入急停，避免停止模式影响本次编码器差值。
        session.stop(f"imu_trial_{index}_settled_stop")

    # 实车控制器从停止进入非零速度模式时，会将左右位移寄存器分别清零；
    # 清零还可能相差约 0.1 秒。因此命令后的 position 本身就是本次动作位移，
    # 不能再减去命令前旧位置，否则会把清零跳变误算成车轮运动。
    left_at_stop_delta = int(at_stop.left_position_deg)
    right_at_stop_delta = int(at_stop.right_position_deg)
    left_delta = int(after.left_position_deg)
    right_delta = int(after.right_position_deg)
    left_coast_delta = _unwrap_i32_delta(after.left_position_deg, at_stop.left_position_deg)
    right_coast_delta = _unwrap_i32_delta(after.right_position_deg, at_stop.right_position_deg)
    expected_sign, _left_progress, _right_progress, mean_progress, balance, sign_valid = (
        _normalize_turn_encoder(direction, left_delta, right_delta)
    )
    left_rev = left_delta / ENCODER_DEGREES_PER_REV
    right_rev = right_delta / ENCODER_DEGREES_PER_REV
    mean_rev = mean_progress / ENCODER_DEGREES_PER_REV
    invalid: List[str] = []
    expected_min_samples = max(4, int((duration_sec + settle_elapsed) / 0.03 * 0.55))
    if integrator.sample_count < expected_min_samples:
        invalid.append(f"IMU样本不足({integrator.sample_count}<{expected_min_samples})")
    if integrator.gap_count > 1:
        invalid.append(f"IMU采样间断({integrator.gap_count})")
    if not stationary:
        invalid.append(
            "停止后未确认静止(elapsed=%.2fs,rate=%.2f°/s,motor=%d/%dRPM)"
            % (
                settle_elapsed,
                abs(float(integrator.last_rate_dps)),
                after.left_speed_rpm,
                after.right_speed_rpm,
            )
        )
    if not sign_valid:
        invalid.append(
            "编码器方向错误(direction=%s,expected_sign=%+d,delta=%+d/%+d)"
            % (direction, expected_sign, left_delta, right_delta)
        )
    if mean_rev < 0.005:
        invalid.append(f"编码器位移过小({mean_rev:.4f}圈)")
    if abs(integrator.angle_deg) < 0.8:
        invalid.append(f"yaw角度过小({integrator.angle_deg:.3f}°)")
    if balance > 0.45:
        invalid.append(f"左右轮位移不平衡({balance:.1%})")
    if after.left_error or after.right_error:
        invalid.append(f"电机故障码(left={after.left_error},right={after.right_error})")
    return TurnTrial(
        index=int(index),
        direction=direction,
        command_rpm=int(rpm),
        left_command_rpm=int(left_command),
        right_command_rpm=int(right_command),
        command_duration_sec=float(duration_sec),
        imu_samples=int(integrator.sample_count),
        imu_gap_count=int(integrator.gap_count),
        yaw_at_stop_deg=float(yaw_at_stop),
        yaw_final_deg=float(integrator.angle_deg),
        coast_yaw_deg=float(integrator.angle_deg - yaw_at_stop),
        settle_elapsed_sec=float(settle_elapsed),
        stationary_confirmed=bool(stationary),
        final_abs_yaw_rate_dps=abs(float(integrator.last_rate_dps)),
        peak_abs_yaw_rate_dps=float(integrator.peak_abs_rate_dps),
        encoder_expected_sign=int(expected_sign),
        encoder_sign_valid=bool(sign_valid),
        left_encoder_before_deg=int(before.left_position_deg),
        right_encoder_before_deg=int(before.right_position_deg),
        left_encoder_at_stop_deg=int(at_stop.left_position_deg),
        right_encoder_at_stop_deg=int(at_stop.right_position_deg),
        left_encoder_final_deg=int(after.left_position_deg),
        right_encoder_final_deg=int(after.right_position_deg),
        left_encoder_at_stop_delta_deg=int(left_at_stop_delta),
        right_encoder_at_stop_delta_deg=int(right_at_stop_delta),
        left_encoder_coast_delta_deg=int(left_coast_delta),
        right_encoder_coast_delta_deg=int(right_coast_delta),
        left_encoder_delta_deg=int(left_delta),
        right_encoder_delta_deg=int(right_delta),
        left_wheel_rev=float(left_rev),
        right_wheel_rev=float(right_rev),
        mean_abs_wheel_rev=float(mean_rev),
        encoder_balance_error=float(balance),
        valid=not invalid,
        invalid_reasons=invalid,
    )


def _linear_fit(x_values: Sequence[float], y_values: Sequence[float]) -> Dict[str, float]:
    if len(x_values) != len(y_values) or not x_values:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "rmse": 999.0}
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    variance = sum((value - x_mean) ** 2 for value in x_values)
    slope = 0.0 if variance <= 1e-12 else sum(
        (x_values[index] - x_mean) * (y_values[index] - y_mean) for index in range(len(x_values))
    ) / variance
    intercept = y_mean - slope * x_mean
    predictions = [slope * value + intercept for value in x_values]
    residual_sum = sum((y_values[index] - predictions[index]) ** 2 for index in range(len(y_values)))
    total_sum = sum((value - y_mean) ** 2 for value in y_values)
    r2 = 1.0 if total_sum <= 1e-12 and residual_sum <= 1e-12 else (
        0.0 if total_sum <= 1e-12 else 1.0 - residual_sum / total_sum
    )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "rmse": math.sqrt(residual_sum / len(y_values)),
    }


def _fit_direction(
    trials: Sequence[TurnTrial],
    direction: str,
    calibration: Optional[StaticCalibration] = None,
) -> Dict[str, Any]:
    selected = [trial for trial in trials if trial.direction == direction and trial.valid]
    if not selected:
        return {"valid_trials": 0, "trustworthy": False, "reasons": ["没有有效试验"]}
    yaw_sign = _sign(_median([trial.yaw_final_deg for trial in selected]))
    positive_yaw = [yaw_sign * trial.yaw_final_deg for trial in selected]
    wheel_revs = [trial.mean_abs_wheel_rev for trial in selected]
    sign_consistency = sum(1 for value in positive_yaw if value > 0.0) / len(positive_yaw)
    # 物理上轮子不转时车体转角应为 0，因此最终控制模型使用过原点斜率；
    # 带截距拟合只用于 R2/RMSE 诊断死区、打滑和样本离散。
    through_origin_slope = sum(x * y for x, y in zip(wheel_revs, positive_yaw)) / max(
        1e-12,
        sum(x * x for x in wheel_revs),
    )
    individual_slopes = [y / x for x, y in zip(wheel_revs, positive_yaw) if x > 1e-6 and y > 0.0]
    slope_cv = _std(individual_slopes) / max(1e-9, abs(_mean(individual_slopes)))
    fit = _linear_fit(wheel_revs, positive_yaw)
    encoder_degrees = [value * ENCODER_DEGREES_PER_REV for value in wheel_revs]
    degrees_per_encoder_degree = through_origin_slope / ENCODER_DEGREES_PER_REV
    encoder_predictions = [degrees_per_encoder_degree * value for value in encoder_degrees]
    encoder_residuals = [
        positive_yaw[index] - encoder_predictions[index] for index in range(len(positive_yaw))
    ]
    encoder_rmse = math.sqrt(_mean([value * value for value in encoder_residuals]))
    # 陀螺仪负责短时真实转角，编码器负责低漂移预测。按两者噪声倒方差分配权重，
    # 同时限制编码器最多占 35%，避免轮胎打滑时编码器错误主导车体转角。
    median_measurement_sec = _median(
        [trial.command_duration_sec + trial.settle_elapsed_sec for trial in selected]
    )
    gyro_sigma = 0.35
    if calibration is not None:
        gyro_sigma = max(
            0.25,
            calibration.yaw_std_dps * max(0.5, median_measurement_sec),
            calibration.max_abs_yaw_noise_dps * 0.08,
        )
    encoder_quantization_sigma = abs(degrees_per_encoder_degree) * ENCODER_RESOLUTION_DEG / math.sqrt(2.0)
    encoder_sigma = max(0.35, encoder_rmse, encoder_quantization_sigma)
    gyro_reliability = 1.0 / (gyro_sigma * gyro_sigma)
    encoder_reliability = 1.0 / (encoder_sigma * encoder_sigma)
    encoder_weight = min(
        0.35,
        max(0.05, encoder_reliability / (gyro_reliability + encoder_reliability)),
    )
    gyro_weight = 1.0 - encoder_weight
    coast_progress = [yaw_sign * trial.coast_yaw_deg for trial in selected]
    encoder_balance = [trial.encoder_balance_error for trial in selected]
    reasons: List[str] = []
    warnings: List[str] = []
    if len(selected) < 3:
        reasons.append(f"有效试验不足({len(selected)}<3)")
    if sign_consistency < 0.90:
        reasons.append(f"yaw方向一致率不足({sign_consistency:.1%})")
    if slope_cv > 0.25:
        reasons.append(f"每圈角度离散过大(CV={slope_cv:.1%})")
    if len(selected) >= 3 and fit["r2"] < 0.80:
        reasons.append(f"线性拟合度偏低(R2={fit['r2']:.3f})")
    mean_balance = _mean(encoder_balance)
    if mean_balance > 0.30:
        reasons.append(f"左右轮平均不平衡({mean_balance:.1%})")
    elif mean_balance > 0.25:
        # 单组超过 45% 已在采集阶段剔除。平均 25%~30% 反映左右机械差异，
        # 若编码器与 gyro 仍保持高相关，可保留模型但必须显式告警。
        warnings.append(f"左右轮存在系统性不平衡({mean_balance:.1%})")
    mean_yaw = max(1e-9, _mean(positive_yaw))
    if encoder_rmse > max(2.0, mean_yaw * 0.20):
        reasons.append(f"编码器预测与陀螺仪偏差过大(RMSE={encoder_rmse:.3f}°)")
    return {
        "valid_trials": len(selected),
        "yaw_sign": yaw_sign,
        "degrees_per_mean_wheel_rev": float(through_origin_slope),
        "body_degrees_per_encoder_degree": float(degrees_per_encoder_degree),
        "encoder_degree_per_body_degree": float(
            0.0 if abs(degrees_per_encoder_degree) <= 1e-12 else 1.0 / degrees_per_encoder_degree
        ),
        "fit_with_intercept": fit,
        "encoder_prediction_rmse_deg": float(encoder_rmse),
        "encoder_prediction_residuals_deg": encoder_residuals,
        "fusion": {
            "method": "inverse_variance_weighted_gyro_encoder",
            "gyro_weight": float(gyro_weight),
            "encoder_weight": float(encoder_weight),
            "estimated_gyro_sigma_deg": float(gyro_sigma),
            "estimated_encoder_sigma_deg": float(encoder_sigma),
        },
        "individual_degrees_per_rev": individual_slopes,
        "slope_cv": float(slope_cv),
        "yaw_sign_consistency": float(sign_consistency),
        "median_coast_deg": float(max(0.0, _median(coast_progress))),
        "mean_encoder_balance_error": float(_mean(encoder_balance)),
        "mean_peak_yaw_rate_dps": float(_mean([trial.peak_abs_yaw_rate_dps for trial in selected])),
        "trustworthy": not reasons,
        "reasons": reasons,
        "warnings": warnings,
    }


def _build_model(calibration: StaticCalibration, trials: Sequence[TurnTrial]) -> Dict[str, Any]:
    left = _fit_direction(trials, "left", calibration)
    right = _fit_direction(trials, "right", calibration)
    reasons: List[str] = []
    warnings: List[str] = []
    if not calibration.trustworthy:
        reasons.extend("静态校准：" + reason for reason in calibration.reasons)
    if not left.get("trustworthy"):
        reasons.extend("左转：" + reason for reason in left.get("reasons", []))
    if not right.get("trustworthy"):
        reasons.extend("右转：" + reason for reason in right.get("reasons", []))
    warnings.extend("左转：" + warning for warning in left.get("warnings", []))
    warnings.extend("右转：" + warning for warning in right.get("warnings", []))
    if left.get("yaw_sign") == right.get("yaw_sign"):
        reasons.append("左右转陀螺仪符号没有相反，轴向或电机方向可能错误")
    left_slope = float(left.get("degrees_per_mean_wheel_rev", 0.0))
    right_slope = float(right.get("degrees_per_mean_wheel_rev", 0.0))
    slope_mean = (abs(left_slope) + abs(right_slope)) / 2.0
    symmetry_error = abs(abs(left_slope) - abs(right_slope)) / max(1e-9, slope_mean)
    if symmetry_error > 0.30:
        reasons.append(f"左右转模型不对称({symmetry_error:.1%})")
    valid_count = sum(1 for trial in trials if trial.valid)
    score = 100
    score -= 25 if not calibration.trustworthy else 0
    score -= min(25, 5 * len(left.get("reasons", [])))
    score -= min(25, 5 * len(right.get("reasons", [])))
    score -= min(25, int(symmetry_error * 50.0))
    return {
        "model_version": 3,
        "created_at": datetime.now().astimezone().isoformat(),
        "description": "LZ30EMA ABZ编码器与ICM20600重力轴yaw角度的原地转向融合标定",
        "encoder": {
            "type": "ABZ",
            "degrees_per_revolution": ENCODER_DEGREES_PER_REV,
            "resolution_degree": ENCODER_RESOLUTION_DEG,
            "physical_signs": {
                "forward": {"left": 1, "right": -1},
                "backward": {"left": -1, "right": 1},
                "turn_left": {"left": -1, "right": -1},
                "turn_right": {"left": 1, "right": 1},
            },
        },
        "fusion_policy": {
            "primary": "ICM20600 gyro integration",
            "secondary": "normalized mean ABZ encoder progress",
            "note": "短时陀螺仪主导，编码器用于预测、融合和打滑/方向异常检测",
        },
        "static_calibration": asdict(calibration),
        "directions": {"left": left, "right": right},
        "symmetry_error": float(symmetry_error),
        "valid_trial_count": int(valid_count),
        "total_trial_count": len(trials),
        "trust_score": max(0, int(score)),
        "trustworthy": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "trials": [asdict(trial) for trial in trials],
        "limitations": [
            "模型只描述原地左右转，不描述直线前后位移",
            "地面材质、轮胎、载荷和电池状态改变后应重新标定",
            "小角度控制应继续使用陀螺仪闭环，编码器模型用于预测和交叉校验",
        ],
    }


def _write_trial_csv(path: Path, trials: Sequence[TurnTrial]) -> None:
    rows = [asdict(trial) for trial in trials]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _print_static(calibration: StaticCalibration) -> None:
    print(
        "静态校准：samples(accel/gyro)=%d/%d gravity=%s |g|=%.4f"
        % (
            calibration.accel_samples,
            calibration.gyro_samples,
            tuple(round(value, 5) for value in calibration.gravity_g),
            calibration.gravity_norm_g,
        )
    )
    print(
        "动态车体竖直轴(sensor xyz)=%s gyro_bias=%s yaw_bias=%.4f°/s yaw_std=%.4f°/s trust=%s"
        % (
            tuple(round(value, 5) for value in calibration.vertical_axis_sensor),
            tuple(round(value, 5) for value in calibration.gyro_bias_dps),
            calibration.yaw_bias_dps,
            calibration.yaw_std_dps,
            calibration.trustworthy,
        )
    )
    for reason in calibration.reasons:
        print("  不可信原因：", reason)


def _run_check(args: argparse.Namespace) -> int:
    _ensure_follow_runtime_stopped()
    session = HardwareSession()
    session.open()
    try:
        first = session.read_motors()
        calibration = _collect_static(session, args.stationary_sec)
        second = session.read_motors()
        left_drift = _unwrap_i32_delta(second.left_position_deg, first.left_position_deg)
        right_drift = _unwrap_i32_delta(second.right_position_deg, first.right_position_deg)
        motor_reasons: List[str] = []
        if first.left_error or first.right_error or second.left_error or second.right_error:
            motor_reasons.append("电机控制器存在故障码")
        if max(abs(first.left_speed_rpm), abs(first.right_speed_rpm), abs(second.left_speed_rpm), abs(second.right_speed_rpm)) > 1:
            motor_reasons.append("静止检查期间电机速度不为 0")
        if abs(left_drift) > 2 or abs(right_drift) > 2:
            motor_reasons.append(f"静止编码器漂移 left={left_drift}° right={right_drift}°")
        payload = {
            "mode": "check",
            "created_at": datetime.now().astimezone().isoformat(),
            "imu": asdict(calibration),
            "motor_first": asdict(first),
            "motor_second": asdict(second),
            "encoder_static_delta_deg": {"left": left_drift, "right": right_drift},
            "motor_trustworthy": not motor_reasons,
            "motor_reasons": motor_reasons,
            "overall_trustworthy": calibration.trustworthy and not motor_reasons,
        }
        _print_static(calibration)
        print(
            f"编码器静止检查：left={first.left_position_deg}->{second.left_position_deg} "
            f"right={first.right_position_deg}->{second.right_position_deg} "
            f"delta=({left_drift},{right_drift})° trust={not motor_reasons}"
        )
        if args.output:
            output = Path(args.output).expanduser().resolve()
            _save_json(output, payload)
            print("检查结果：", output)
        return 0 if payload["overall_trustworthy"] else 2
    finally:
        session.close(ensure_stop=False)


def _run_direction_check(args: argparse.Namespace) -> int:
    """Use two minimal pulses to verify motor, encoder and gyro direction together."""
    _ensure_follow_runtime_stopped()
    rpm = int(args.rpm)
    duration = float(args.duration)
    if not 1 <= rpm <= MAX_SAFE_RPM:
        raise ValueError(f"RPM 必须在 1..{MAX_SAFE_RPM}")
    if not 0.10 <= duration <= MAX_SAFE_PULSE_SEC:
        raise ValueError(f"duration 必须在 0.10..{MAX_SAFE_PULSE_SEC:.2f}s")
    _require_motion_confirmation(
        args,
        f"将以 {rpm} RPM 分别左转和右转 {duration:.2f} 秒，只检查方向，不生成正式模型",
    )

    session = HardwareSession()
    session.open()
    trials: List[TurnTrial] = []
    try:
        session.disable_parking_for_calibration()
        session.stop("imu_direction_check_start")
        time.sleep(0.30)
        calibration = _collect_static(session, args.stationary_sec)
        _print_static(calibration)
        if not calibration.trustworthy and not args.allow_noisy_imu:
            raise RuntimeError("静态 IMU 检查不可信，拒绝执行方向检查")

        for index, direction in enumerate(("left", "right"), start=1):
            trial = _run_trial(
                session,
                calibration,
                index=index,
                direction=direction,
                rpm=rpm,
                duration_sec=duration,
                settle_sec=args.settle_sec,
                settle_timeout_sec=args.settle_timeout_sec,
                quiet_sec=args.quiet_sec,
                quiet_yaw_rate_dps=args.quiet_yaw_rate_dps,
            )
            trials.append(trial)
            print(
                "%s：command=%+d/%+dRPM encoder=%+d/%+d° yaw=%+.3f° "
                "stationary=%s sign=%s valid=%s"
                % (
                    direction,
                    trial.left_command_rpm,
                    trial.right_command_rpm,
                    trial.left_encoder_delta_deg,
                    trial.right_encoder_delta_deg,
                    trial.yaw_final_deg,
                    trial.stationary_confirmed,
                    trial.encoder_sign_valid,
                    trial.valid,
                )
            )
            if trial.invalid_reasons:
                print("  无效原因：", "; ".join(trial.invalid_reasons))
            time.sleep(max(0.50, float(args.pause_sec)))

        yaw_opposite = (
            len(trials) == 2
            and abs(trials[0].yaw_final_deg) >= 0.8
            and abs(trials[1].yaw_final_deg) >= 0.8
            and _sign(trials[0].yaw_final_deg) == -_sign(trials[1].yaw_final_deg)
        )
        trustworthy = calibration.trustworthy and all(trial.valid for trial in trials) and yaw_opposite
        reasons: List[str] = []
        if not calibration.trustworthy:
            reasons.extend(calibration.reasons)
        if not yaw_opposite:
            reasons.append("左右转的陀螺仪 yaw 符号未形成方向相反的有效角度")
        for trial in trials:
            reasons.extend(f"{trial.direction}：{reason}" for reason in trial.invalid_reasons)
        payload = {
            "mode": "direction-check",
            "created_at": datetime.now().astimezone().isoformat(),
            "encoder_mapping": {
                "turn_left": [-1, -1],
                "turn_right": [1, 1],
                "degrees_per_revolution": ENCODER_DEGREES_PER_REV,
                "resolution_degree": ENCODER_RESOLUTION_DEG,
            },
            "static_calibration": asdict(calibration),
            "trials": [asdict(trial) for trial in trials],
            "gyro_yaw_signs_opposite": bool(yaw_opposite),
            "trustworthy": bool(trustworthy),
            "reasons": reasons,
        }
        if args.output:
            output = Path(args.output).expanduser().resolve()
            _save_json(output, payload)
            print("方向检查结果：", output)
        print("方向联合检查：", "通过" if trustworthy else "不通过")
        return 0 if trustworthy else 2
    finally:
        try:
            session.stop("imu_direction_check_abort_or_finish")
        finally:
            session.close(ensure_stop=True)


def _run_calibrate(args: argparse.Namespace) -> int:
    _ensure_follow_runtime_stopped()
    rpms = _parse_csv_ints(args.rpms)
    durations = _parse_csv_floats(args.durations)
    repeats = max(1, min(5, int(args.repeats)))
    trial_count = len(rpms) * len(durations) * 2 * repeats
    _require_motion_confirmation(
        args,
        f"将执行 {trial_count} 次左右交替短转，最大 {max(rpms)} RPM，最长 {max(durations):.2f} 秒",
    )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_json = output_dir / f"imu_turn_progress_{stamp}.json"
    progress_csv = output_dir / f"imu_turn_progress_trials_{stamp}.csv"

    session = HardwareSession()
    session.open()
    trials: List[TurnTrial] = []
    try:
        session.disable_parking_for_calibration()
        session.stop("imu_calibration_start")
        time.sleep(0.30)
        calibration = _collect_static(session, args.stationary_sec)
        _print_static(calibration)
        if not calibration.trustworthy and not args.allow_noisy_imu:
            raise RuntimeError("静态 IMU 检查不可信，已拒绝驱动车轮；排除震动后重试，或明确添加 --allow-noisy-imu")

        plan: List[Tuple[str, int, float]] = []
        for _repeat in range(repeats):
            for rpm in rpms:
                for duration in durations:
                    plan.extend((("left", rpm, duration), ("right", rpm, duration)))

        for index, (direction, rpm, duration) in enumerate(plan, start=1):
            print(f"\n试验 {index}/{len(plan)}：{direction} {rpm}RPM {duration:.2f}s")
            try:
                trial = _run_trial(
                    session,
                    calibration,
                    index=index,
                    direction=direction,
                    rpm=rpm,
                    duration_sec=duration,
                    settle_sec=args.settle_sec,
                    settle_timeout_sec=args.settle_timeout_sec,
                    quiet_sec=args.quiet_sec,
                    quiet_yaw_rate_dps=args.quiet_yaw_rate_dps,
                )
            except Exception as exc:
                try:
                    session.stop(f"imu_calibration_trial_{index}_error")
                except Exception as stop_exc:
                    print(f"试验异常后的急停通信也失败：{stop_exc}", file=sys.stderr)
                partial_json = output_dir / f"imu_turn_partial_{stamp}.json"
                partial_csv = output_dir / f"imu_turn_partial_trials_{stamp}.csv"
                _save_json(
                    partial_json,
                    {
                        "mode": "calibrate_partial",
                        "created_at": datetime.now().astimezone().isoformat(),
                        "failed_trial": int(index),
                        "error": str(exc),
                        "static_calibration": asdict(calibration),
                        "completed_trials": [asdict(item) for item in trials],
                    },
                )
                _write_trial_csv(partial_csv, trials)
                print(f"标定中断，已保存 {len(trials)} 组完整样本：{partial_json}", file=sys.stderr)
                raise
            trials.append(trial)
            # 每完成一组立即落盘，断电、串口异常或后处理失败时也不会丢掉已完成样本。
            _save_json(
                progress_json,
                {
                    "mode": "calibrate_progress",
                    "created_at": datetime.now().astimezone().isoformat(),
                    "completed_trial_count": len(trials),
                    "planned_trial_count": len(plan),
                    "static_calibration": asdict(calibration),
                    "trials": [asdict(item) for item in trials],
                },
            )
            _write_trial_csv(progress_csv, trials)
            print(
                "  yaw(stop/final/coast)=%.3f/%.3f/%.3f° encoder(L/R)=%.4f/%.4f圈 balance=%.1f%% valid=%s"
                % (
                    trial.yaw_at_stop_deg,
                    trial.yaw_final_deg,
                    trial.coast_yaw_deg,
                    trial.left_wheel_rev,
                    trial.right_wheel_rev,
                    trial.encoder_balance_error * 100.0,
                    trial.valid,
                )
            )
            if trial.invalid_reasons:
                print("  无效原因：", "; ".join(trial.invalid_reasons))
            time.sleep(max(0.30, float(args.pause_sec)))

        model = _build_model(calibration, trials)
        json_path = output_dir / f"imu_turn_calibration_{stamp}.json"
        csv_path = output_dir / f"imu_turn_trials_{stamp}.csv"
        _save_json(json_path, model)
        _write_trial_csv(csv_path, trials)
        print("\n标定完成：")
        print(
            json.dumps(
                {
                    key: model[key]
                    for key in ("trustworthy", "trust_score", "symmetry_error", "reasons", "warnings")
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        for direction in ("left", "right"):
            fitted = model["directions"][direction]
            print(
                "%s: sign=%s body_deg/wheel_rev=%.3f body_deg/encoder_deg=%.5f "
                "encoder_RMSE=%.3f° weights(gyro/encoder)=%.2f/%.2f coast=%.3f° CV=%.1f%% R2=%.3f trust=%s"
                % (
                    direction,
                    fitted.get("yaw_sign"),
                    float(fitted.get("degrees_per_mean_wheel_rev", 0.0)),
                    float(fitted.get("body_degrees_per_encoder_degree", 0.0)),
                    float(fitted.get("encoder_prediction_rmse_deg", 0.0)),
                    float(fitted.get("fusion", {}).get("gyro_weight", 1.0)),
                    float(fitted.get("fusion", {}).get("encoder_weight", 0.0)),
                    float(fitted.get("median_coast_deg", 0.0)),
                    float(fitted.get("slope_cv", 0.0)) * 100.0,
                    float(fitted.get("fit_with_intercept", {}).get("r2", 0.0)),
                    fitted.get("trustworthy"),
                )
            )
        print("模型：", json_path)
        print("原始试验：", csv_path)
        return 0 if model["trustworthy"] else 3
    finally:
        try:
            session.stop("imu_calibration_abort_or_finish")
        finally:
            session.close(ensure_stop=True)


def _load_model(path: str) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if int(model.get("model_version", 0)) != 3:
        raise RuntimeError(
            "该验证命令只支持融合模型 v3；v2 错误地用命令前旧位置计算位移，"
            "请用修正后的脚本重新执行 calibrate"
        )
    return model


def _gyro_control_eligibility(fitted: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Allow gyro closed-loop control even when wheel slip invalidates angle fitting."""
    reasons: List[str] = []
    valid_trials = int(fitted.get("valid_trials", 0))
    yaw_sign = int(fitted.get("yaw_sign", 0))
    sign_consistency = float(fitted.get("yaw_sign_consistency", 0.0))
    peak_rate = float(fitted.get("mean_peak_yaw_rate_dps", 0.0))
    if valid_trials < 3:
        reasons.append(f"有效试验不足({valid_trials}<3)")
    if yaw_sign not in (-1, 1):
        reasons.append(f"yaw方向符号无效({yaw_sign})")
    if sign_consistency < 0.90:
        reasons.append(f"yaw方向一致率不足({sign_consistency:.1%})")
    if peak_rate < 2.0:
        reasons.append(f"平均yaw峰值过小({peak_rate:.2f}°/s)")
    return not reasons, reasons


def _rpm_matched_coast_deg(
    model: Dict[str, Any],
    direction: str,
    rpm: int,
    yaw_sign: int,
    fallback: float,
) -> Tuple[float, str, int]:
    """Use coast samples from the same RPM so low-speed turns are not over-braked."""
    matching: List[float] = []
    for raw in model.get("trials", []):
        if not raw.get("valid", False):
            continue
        if raw.get("direction") != direction or int(raw.get("command_rpm", 0)) != int(rpm):
            continue
        coast = yaw_sign * float(raw.get("coast_yaw_deg", 0.0))
        if coast >= 0.0:
            matching.append(coast)
    if len(matching) >= 2:
        return float(_median(matching)), "same_direction_same_rpm_median", len(matching)
    return max(0.0, float(fallback)), "direction_median_fallback", len(matching)


def _run_closed_loop_verify(
    session: HardwareSession,
    calibration: StaticCalibration,
    model: Dict[str, Any],
    direction: str,
    target_angle_deg: float,
    rpm: int,
    settle_sec: float,
    settle_timeout_sec: float,
    quiet_sec: float,
    quiet_yaw_rate_dps: float,
    max_turn_sec: float,
) -> Dict[str, Any]:
    fitted = model.get("directions", {}).get(direction, {})
    gyro_control_ok, gyro_control_reasons = _gyro_control_eligibility(fitted)
    if not gyro_control_ok:
        raise RuntimeError(
            f"模型中的 {direction} 方向不能用于陀螺仪闭环：{gyro_control_reasons}"
        )
    encoder_angle_model_trustworthy = bool(fitted.get("trustworthy"))
    yaw_sign = int(fitted["yaw_sign"])
    degrees_per_encoder_degree = float(fitted["body_degrees_per_encoder_degree"])
    # 反向制动命令会再次清零位移寄存器，无法把两个速度段的最终编码器位置
    # 直接拼成角度。因此最终角度由 gyro 积分，编码器在换向前校验方向、位移和轮差。
    gyro_weight = 1.0
    encoder_weight = 0.0
    control_mode = "gyro_dynamic_brake_encoder_segment_guard"
    expected_coast, coast_source, coast_sample_count = _rpm_matched_coast_deg(
        model,
        direction,
        rpm,
        yaw_sign,
        float(fitted.get("median_coast_deg", 0.0)),
    )
    stop_progress = max(target_angle_deg * 0.30, target_angle_deg - expected_coast * 1.40)

    session.stop("imu_verify_prepare")
    time.sleep(0.25)
    before = session.read_motors()
    integrator = YawIntegrator(calibration)
    _prime_integrator(session, integrator)
    started = time.monotonic()
    stop_reason = "target"
    at_stop: Optional[MotorReading] = None
    drive_end: Optional[MotorReading] = None
    yaw_at_brake_start = 0.0
    brake_elapsed = 0.0
    brake_stop_reason = "not_started"
    try:
        session.command_turn(direction, rpm)
        while True:
            if session.imu is None:
                raise RuntimeError("IMU 未初始化")
            snapshot = session.imu.poll(0.01)
            integrator.feed(snapshot.get("gyro"))
            elapsed = time.monotonic() - started
            progress = yaw_sign * integrator.angle_deg
            # gyro 停止出样时继续盲转会造成严重过冲，必须在常规超时前立即停车。
            if integrator.sample_age_sec() > MAX_GYRO_STALE_SEC:
                stop_reason = "gyro_stale"
                raise RuntimeError(
                    f"陀螺仪数据中断：连续 {integrator.sample_age_sec():.3f}s 没有新样本"
                )
            if progress >= stop_progress:
                break
            if elapsed > max_turn_sec:
                stop_reason = "timeout"
                raise RuntimeError(
                    f"闭环转向超时：elapsed={elapsed:.2f}s progress={progress:.2f}° target={target_angle_deg:.2f}°"
                )
            if elapsed > 0.45 and progress < -1.0:
                stop_reason = "wrong_direction"
                raise RuntimeError(f"陀螺仪检测到反向旋转：progress={progress:.2f}°")
            if elapsed > 0.60 and integrator.peak_abs_rate_dps < 2.0:
                stop_reason = "no_rotation"
                raise RuntimeError("电机已下发但陀螺仪未检测到可信旋转")
        drive_end = session.read_motors()
        yaw_at_brake_start = integrator.angle_deg
        opposite = "right" if direction == "left" else "left"
        session.command_turn(opposite, rpm)
        brake_started = time.monotonic()
        while True:
            if session.imu is None:
                raise RuntimeError("IMU 未初始化")
            snapshot = session.imu.poll(0.01)
            integrator.feed(snapshot.get("gyro"))
            brake_elapsed = time.monotonic() - brake_started
            progress = yaw_sign * integrator.angle_deg
            signed_rate = yaw_sign * integrator.last_rate_dps
            if integrator.sample_age_sec() > MAX_GYRO_STALE_SEC:
                stop_reason = "gyro_stale_during_brake"
                raise RuntimeError("反向制动期间陀螺仪数据中断")
            if brake_elapsed >= 0.15 and signed_rate <= 4.0:
                brake_stop_reason = "yaw_rate_reduced"
                break
            if progress >= target_angle_deg + max(1.0, target_angle_deg * 0.15):
                brake_stop_reason = "angle_guard"
                break
            if brake_elapsed >= 0.50:
                brake_stop_reason = "brake_timeout"
                break
    finally:
        yaw_at_stop = integrator.angle_deg
        try:
            session.stop(f"imu_verify_{stop_reason}_reverse_then_emergency")
            at_stop = session.read_motors()
        except Exception:
            session.stop(f"imu_verify_{stop_reason}_emergency_brake_failed_stop")
            raise
    if at_stop is None:
        raise RuntimeError("闭环验证未取得反向制动结束时的编码器读数")
    if drive_end is None:
        raise RuntimeError("闭环验证未取得换向前的编码器读数")
    try:
        stationary, settle_elapsed, after = _feed_until_stationary(
            session,
            integrator,
            min_settle_sec=settle_sec,
            settle_timeout_sec=settle_timeout_sec,
            quiet_sec=quiet_sec,
            quiet_yaw_rate_dps=quiet_yaw_rate_dps,
        )
    finally:
        session.stop(f"imu_verify_{stop_reason}_settled_stop")
    # 非零速度命令会把位移寄存器清零，命令后的原始 position 就是本次位移。
    # before 仅保留用于发现控制器清零行为，不参与动作位移计算。
    left_delta = int(after.left_position_deg)
    right_delta = int(after.right_position_deg)
    left_at_stop_delta = int(drive_end.left_position_deg)
    right_at_stop_delta = int(drive_end.right_position_deg)
    (
        _at_stop_expected_sign,
        left_at_stop_progress,
        right_at_stop_progress,
        mean_at_stop_encoder_deg,
        at_stop_balance,
        at_stop_sign_valid,
    ) = _normalize_turn_encoder(direction, left_at_stop_delta, right_at_stop_delta)
    expected_sign, left_progress, right_progress, mean_encoder_deg, balance, sign_valid = (
        _normalize_turn_encoder(direction, left_delta, right_delta)
    )
    mean_rev = mean_at_stop_encoder_deg / ENCODER_DEGREES_PER_REV
    gyro_progress = yaw_sign * integrator.angle_deg
    encoder_angle = degrees_per_encoder_degree * mean_at_stop_encoder_deg
    fused_angle = gyro_progress
    gyro_error = gyro_progress - target_angle_deg
    fused_error = fused_angle - target_angle_deg
    encoder_cross_reference = yaw_sign * yaw_at_brake_start
    cross_error = encoder_angle - encoder_cross_reference
    tolerance = max(1.5, target_angle_deg * 0.20)
    encoder_cross_limit = max(3.0, target_angle_deg * 0.30)
    encoder_cross_check_ok = (
        abs(cross_error) <= encoder_cross_limit
        if encoder_angle_model_trustworthy
        else True
    )
    encoder_cross_check_warning = bool(
        encoder_angle_model_trustworthy
        and abs(cross_error) > max(2.5, target_angle_deg * 0.25)
    )
    encoder_guard_sign_valid = at_stop_sign_valid
    encoder_guard_progress_deg = mean_at_stop_encoder_deg
    encoder_guard_balance = at_stop_balance
    encoder_guard_source = "before_reverse_brake_position"
    trustworthy = (
        abs(fused_error) <= tolerance
        and encoder_cross_check_ok
        and integrator.gap_count <= 1
        and stationary
        and encoder_guard_sign_valid
        and encoder_guard_progress_deg >= 1.0
        and encoder_guard_balance <= 0.45
    )
    return {
        "mode": "verify",
        "created_at": datetime.now().astimezone().isoformat(),
        "direction": direction,
        "command_rpm": int(rpm),
        "brake_mode": "dynamic_reverse_then_emergency",
        "brake_duration_sec": float(brake_elapsed),
        "brake_stop_reason": brake_stop_reason,
        "target_angle_deg": float(target_angle_deg),
        "control_mode": control_mode,
        "encoder_angle_model_trustworthy": encoder_angle_model_trustworthy,
        "encoder_angle_model_reasons": list(fitted.get("reasons", [])),
        "gyro_control_eligible": gyro_control_ok,
        "gyro_control_reasons": gyro_control_reasons,
        "stop_progress_deg": float(stop_progress),
        "expected_coast_deg": float(expected_coast),
        "expected_coast_source": coast_source,
        "expected_coast_sample_count": int(coast_sample_count),
        "gyro_yaw_at_brake_start_deg": float(yaw_sign * yaw_at_brake_start),
        "gyro_yaw_at_stop_deg": float(yaw_sign * yaw_at_stop),
        "gyro_yaw_final_deg": float(gyro_progress),
        "gyro_target_error_deg": float(gyro_error),
        "encoder_angle_estimate_deg": float(encoder_angle),
        "encoder_angle_estimate_basis": "before_reverse_brake_segment_only",
        "fused_angle_estimate_deg": float(fused_angle),
        "fused_target_error_deg": float(fused_error),
        "fusion_weights": {"gyro": float(gyro_weight), "encoder": float(encoder_weight)},
        "encoder_vs_gyro_error_deg": float(cross_error),
        "encoder_cross_reference_yaw_deg": float(encoder_cross_reference),
        "encoder_angle_cross_check_ok": bool(encoder_cross_check_ok),
        "encoder_angle_cross_check_limit_deg": float(encoder_cross_limit),
        "encoder_angle_cross_check_warning": encoder_cross_check_warning,
        "encoder_expected_sign": int(expected_sign),
        "encoder_sign_valid": bool(sign_valid),
        "encoder_at_stop_sign_valid": bool(at_stop_sign_valid),
        "encoder_guard_source": encoder_guard_source,
        "encoder_guard_sign_valid": bool(encoder_guard_sign_valid),
        "encoder_guard_progress_deg": float(encoder_guard_progress_deg),
        "encoder_guard_balance_error": float(encoder_guard_balance),
        "left_encoder_at_stop_delta_deg": int(left_at_stop_delta),
        "right_encoder_at_stop_delta_deg": int(right_at_stop_delta),
        "left_encoder_at_stop_progress_deg": float(left_at_stop_progress),
        "right_encoder_at_stop_progress_deg": float(right_at_stop_progress),
        "mean_at_stop_encoder_progress_deg": float(mean_at_stop_encoder_deg),
        "left_encoder_delta_deg": int(left_delta),
        "right_encoder_delta_deg": int(right_delta),
        "left_encoder_progress_deg": float(left_progress),
        "right_encoder_progress_deg": float(right_progress),
        "mean_abs_wheel_rev": float(mean_rev),
        "encoder_balance_error": float(balance),
        "imu_samples": int(integrator.sample_count),
        "imu_gap_count": int(integrator.gap_count),
        "peak_abs_yaw_rate_dps": float(integrator.peak_abs_rate_dps),
        "settle_elapsed_sec": float(settle_elapsed),
        "stationary_confirmed": bool(stationary),
        "final_motor_speed_rpm": {
            "left": int(after.left_speed_rpm),
            "right": int(after.right_speed_rpm),
        },
        "final_abs_yaw_rate_dps": abs(float(integrator.last_rate_dps)),
        "tolerance_deg": float(tolerance),
        "trustworthy": bool(trustworthy),
    }


def _run_verify(args: argparse.Namespace) -> int:
    _ensure_follow_runtime_stopped()
    model = _load_model(args.model)
    target = float(args.angle)
    rpm = int(args.rpm)
    if not 1.0 <= target <= MAX_SAFE_VERIFY_ANGLE_DEG:
        raise ValueError(f"验证角度必须在 1..{MAX_SAFE_VERIFY_ANGLE_DEG:.0f}°")
    if not 1 <= rpm <= MAX_SAFE_RPM:
        raise ValueError(f"RPM 必须在 1..{MAX_SAFE_RPM}")
    _require_motion_confirmation(args, f"将以 {rpm} RPM 闭环执行 {args.direction} {target:.1f}°")

    session = HardwareSession()
    session.open()
    try:
        session.disable_parking_for_calibration()
        session.stop("imu_verify_start")
        time.sleep(0.30)
        current = _collect_static(session, args.stationary_sec)
        _print_static(current)
        if not current.trustworthy:
            raise RuntimeError("当前静止 IMU 校准不可信，拒绝执行角度验证")
        saved_static = model.get("static_calibration", {})
        saved_axis = tuple(float(value) for value in saved_static.get("vertical_axis_sensor", (0.0, 0.0, 1.0)))
        axis_alignment = _dot(saved_axis, current.vertical_axis_sensor)
        if axis_alignment < 0.95:
            raise RuntimeError(f"IMU安装姿态与标定时不一致：axis_alignment={axis_alignment:.3f}")
        result = _run_closed_loop_verify(
            session,
            current,
            model,
            direction=args.direction,
            target_angle_deg=target,
            rpm=rpm,
            settle_sec=args.settle_sec,
            settle_timeout_sec=args.settle_timeout_sec,
            quiet_sec=args.quiet_sec,
            quiet_yaw_rate_dps=args.quiet_yaw_rate_dps,
            max_turn_sec=args.max_turn_sec,
        )
        result["axis_alignment"] = axis_alignment
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output:
            output = Path(args.output).expanduser().resolve()
            _save_json(output, result)
            print("验证结果：", output)
        return 0 if result["trustworthy"] else 4
    finally:
        try:
            session.stop("imu_verify_abort_or_finish")
        finally:
            session.close(ensure_stop=True)


def _run_self_test() -> int:
    checks: List[Tuple[str, bool]] = []
    checks.append(("unwrap_positive", _unwrap_i32_delta(-2147483600, 2147483600) == 96))
    checks.append(("unwrap_negative", _unwrap_i32_delta(2147483600, -2147483600) == -96))
    vertical = _normalize((-0.08, -0.74, 0.67))
    checks.append(("gravity_normalized", abs(_norm(vertical) - 1.0) < 1e-9))
    left_normalized = _normalize_turn_encoder("left", -36, -34)
    right_normalized = _normalize_turn_encoder("right", 36, 34)
    checks.append(("left_encoder_sign", left_normalized[5] and abs(left_normalized[3] - 35.0) < 1e-9))
    checks.append(("right_encoder_sign", right_normalized[5] and abs(right_normalized[3] - 35.0) < 1e-9))
    checks.append(("reject_wrong_encoder_sign", not _normalize_turn_encoder("left", 36, 34)[5]))
    synthetic: List[TurnTrial] = []
    for index, wheel_rev in enumerate((0.03, 0.05, 0.08, 0.10), start=1):
        for direction, yaw_sign in (("left", -1), ("right", 1)):
            yaw = yaw_sign * 180.0 * wheel_rev
            encoder_delta = int(TURN_ENCODER_SIGN[direction] * wheel_rev * ENCODER_DEGREES_PER_REV)
            synthetic.append(
                TurnTrial(
                    index=index,
                    direction=direction,
                    command_rpm=10,
                    left_command_rpm=yaw_sign * 10,
                    right_command_rpm=yaw_sign * 10,
                    command_duration_sec=0.4,
                    imu_samples=30,
                    imu_gap_count=0,
                    yaw_at_stop_deg=yaw * 0.9,
                    yaw_final_deg=yaw,
                    coast_yaw_deg=yaw * 0.1,
                    settle_elapsed_sec=0.8,
                    stationary_confirmed=True,
                    final_abs_yaw_rate_dps=0.2,
                    peak_abs_yaw_rate_dps=20.0,
                    encoder_expected_sign=TURN_ENCODER_SIGN[direction],
                    encoder_sign_valid=True,
                    left_encoder_before_deg=100,
                    right_encoder_before_deg=-100,
                    left_encoder_at_stop_deg=100 + int(encoder_delta * 0.9),
                    right_encoder_at_stop_deg=-100 + int(encoder_delta * 0.9),
                    left_encoder_final_deg=100 + encoder_delta,
                    right_encoder_final_deg=-100 + encoder_delta,
                    left_encoder_at_stop_delta_deg=int(encoder_delta * 0.9),
                    right_encoder_at_stop_delta_deg=int(encoder_delta * 0.9),
                    left_encoder_coast_delta_deg=encoder_delta - int(encoder_delta * 0.9),
                    right_encoder_coast_delta_deg=encoder_delta - int(encoder_delta * 0.9),
                    left_encoder_delta_deg=encoder_delta,
                    right_encoder_delta_deg=encoder_delta,
                    left_wheel_rev=encoder_delta / ENCODER_DEGREES_PER_REV,
                    right_wheel_rev=encoder_delta / ENCODER_DEGREES_PER_REV,
                    mean_abs_wheel_rev=wheel_rev,
                    encoder_balance_error=0.0,
                    valid=True,
                    invalid_reasons=[],
                )
            )
    left = _fit_direction(synthetic, "left")
    right = _fit_direction(synthetic, "right")
    checks.append(("left_fit", abs(float(left["degrees_per_mean_wheel_rev"]) - 180.0) < 1e-9))
    checks.append(("right_fit", abs(float(right["degrees_per_mean_wheel_rev"]) - 180.0) < 1e-9))
    checks.append(("encoder_degree_fit", abs(float(left["body_degrees_per_encoder_degree"]) - 0.5) < 1e-9))
    checks.append(("opposite_signs", int(left["yaw_sign"]) == -int(right["yaw_sign"])))
    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    return 0 if not failed else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="独立标定 ICM20600 yaw、LZ30EMA 编码器圈数与小车左右原地转角的关系。"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="运行配置 INI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("self-test", help="纯软件自测，不访问硬件")

    check = subparsers.add_parser("check", help="只读静态体检，不驱动车轮")
    check.add_argument("--stationary-sec", type=float, default=3.0)
    check.add_argument("--output", default="")

    direction_check = subparsers.add_parser(
        "direction-check",
        help="低速左右各转一次，联合检查命令、编码器和陀螺仪方向",
    )
    direction_check.add_argument("--execute", action="store_true", help="确认允许脚本驱动车轮")
    direction_check.add_argument("--rpm", type=int, default=8)
    direction_check.add_argument("--duration", type=float, default=0.50)
    direction_check.add_argument("--stationary-sec", type=float, default=3.0)
    direction_check.add_argument("--settle-sec", type=float, default=0.60)
    direction_check.add_argument("--settle-timeout-sec", type=float, default=4.0)
    direction_check.add_argument("--quiet-sec", type=float, default=0.60)
    direction_check.add_argument("--quiet-yaw-rate-dps", type=float, default=1.0)
    direction_check.add_argument("--pause-sec", type=float, default=2.0)
    direction_check.add_argument("--allow-noisy-imu", action="store_true")
    direction_check.add_argument("--output", default="")

    calibrate = subparsers.add_parser("calibrate", help="执行左右短脉冲并生成标定模型")
    calibrate.add_argument("--execute", action="store_true", help="确认允许脚本驱动车轮")
    calibrate.add_argument("--rpms", default="8,10,12", help=f"逗号分隔，最大 {MAX_SAFE_RPM}")
    calibrate.add_argument("--durations", default="0.25,0.45", help=f"逗号分隔，最大 {MAX_SAFE_PULSE_SEC:.2f}s")
    calibrate.add_argument("--repeats", type=int, default=1)
    calibrate.add_argument("--stationary-sec", type=float, default=3.0)
    calibrate.add_argument("--settle-sec", type=float, default=0.60)
    calibrate.add_argument("--settle-timeout-sec", type=float, default=4.0)
    calibrate.add_argument("--quiet-sec", type=float, default=0.60)
    calibrate.add_argument("--quiet-yaw-rate-dps", type=float, default=1.0)
    calibrate.add_argument("--pause-sec", type=float, default=0.70)
    calibrate.add_argument("--output-dir", default="calibration")
    calibrate.add_argument("--allow-noisy-imu", action="store_true")

    verify = subparsers.add_parser("verify", help="使用模型闭环验证指定小角度")
    verify.add_argument("--execute", action="store_true", help="确认允许脚本驱动车轮")
    verify.add_argument("--model", required=True)
    verify.add_argument("--direction", choices=("left", "right"), required=True)
    verify.add_argument("--angle", type=float, required=True)
    verify.add_argument("--rpm", type=int, default=8)
    verify.add_argument("--stationary-sec", type=float, default=2.5)
    verify.add_argument("--settle-sec", type=float, default=0.70)
    verify.add_argument("--settle-timeout-sec", type=float, default=4.0)
    verify.add_argument("--quiet-sec", type=float, default=0.60)
    verify.add_argument("--quiet-yaw-rate-dps", type=float, default=1.0)
    verify.add_argument("--max-turn-sec", type=float, default=3.0)
    verify.add_argument("--output", default="")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.mode == "self-test":
        return _run_self_test()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.is_file():
        parser.error(f"配置文件不存在：{config_path}")
    from car_control_modular.config_loader import load_config_to_env

    load_config_to_env(str(config_path))
    try:
        if args.mode == "check":
            return _run_check(args)
        if args.mode == "direction-check":
            return _run_direction_check(args)
        if args.mode == "calibrate":
            return _run_calibrate(args)
        if args.mode == "verify":
            return _run_verify(args)
        parser.error(f"未知模式：{args.mode}")
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停车并退出。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
