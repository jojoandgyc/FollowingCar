#!/usr/bin/env python3
"""Forward-only controller step-response experiment; defaults to a hardware-free plan.

Only --execute plus the interactive STEP confirmation opens the motor port. The
controller owns acceleration/deceleration: this tool sends 0 -> 60/100 -> 0 RPM,
without a software ramp. Zero-target stopping is measured; emergency stopping is
reserved for startup, watchdog failures and final cleanup. No IMU is required.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.imu_turn_calibrate import (  # noqa: E402
    DEFAULT_CONFIG, HardwareSession, _ensure_follow_runtime_stopped, _unwrap_i32_delta,
)

MAX_RPM = 100
FEEDBACK_MAX_AGE = 0.20
SERIAL_TIMEOUT = 0.06
QUIET_RPM = 2.0
QUIET_SEC = 0.25
SAMPLE_PERIOD = 0.02


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true", help="实际测试；仍需现场输入 STEP")
    parser.add_argument("--rpms", default="60,100", help="仅允许 60、100；例如先用 --rpms 60")
    parser.add_argument("--duration", type=float, default=0.8, help="每次阶跃保持秒数，0.2..1.5")
    parser.add_argument("--repeats", type=int, default=1, help="每档重复次数，1..3")
    parser.add_argument("--settle-timeout", type=float, default=3.0, help="零速后等待停止秒数，0.5..4")
    parser.add_argument("--max-travel", type=float, default=2.0, help="单次各轮编码器累计路程上限 m，0.2..3")
    parser.add_argument("--max-total-travel", type=float, default=4.0, help="整组各轮累计路程上限 m，0.2..6")
    parser.add_argument("--wheel-diameter", type=float, default=0.26, help="轮径 m，默认 26 cm")
    parser.add_argument("--output-dir", default=str(ROOT / "calibration"))
    return parser


def validate_args(args: argparse.Namespace) -> list[int]:
    try:
        rpms = [int(value.strip()) for value in args.rpms.split(",")]
    except ValueError as exc:
        raise ValueError("rpms 仅允许 60 或 100 的逗号分隔列表") from exc
    if not rpms or len(rpms) > 2 or any(rpm not in (60, 100) for rpm in rpms):
        raise ValueError("rpms 仅允许 60、100，最多两档；测试上限固定为 100 RPM")
    if len(set(rpms)) != len(rpms):
        raise ValueError("rpms 不应重复；重复测试请用 --repeats")
    for name, lower, upper in (("duration", 0.2, 1.5), ("settle_timeout", 0.5, 4.0),
                               ("max_travel", 0.2, 3.0), ("max_total_travel", 0.2, 6.0),
                               ("wheel_diameter", 0.10, 0.50)):
        value = getattr(args, name)
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} 必须在 {lower}..{upper} 范围内且为有限值")
    if not 1 <= args.repeats <= 3:
        raise ValueError("repeats 必须在 1..3 范围内")
    if args.max_total_travel < args.max_travel:
        raise ValueError("max_total_travel 不得小于 max_travel")
    return rpms


def acquire_test_lock(port: str):
    """Hold an advisory per-port lock throughout confirmation/open/cleanup."""
    port_key = hashlib.sha256(os.path.realpath(port).encode()).hexdigest()[:20]
    lock_path = Path("/tmp") / f"rk_forward_step_{port_key}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    stream = os.fdopen(descriptor, "w")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise RuntimeError("已有阶跃测试占用此电机端口的测试锁")
    return stream


def validate_controller_mode(bus: dict[str, Any], configured_mode: int) -> str:
    """Check command semantics without reconfiguring the motor controller.

    This board reports runtime/config mode 0 while control_mode=0x38
    advertises sensored FOC closed-loop serial control. Accept that observed
    combination explicitly, not arbitrary open-loop/differential modes. Keep
    the discrepancy visible in results rather than relabelling mode 0 as 1.
    """
    runtime_mode = bus.get("runtime_system_mode")
    control = bus.get("control_mode")
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


class ForwardSession(HardwareSession):
    """Reuse motor backend/register helpers without HardwareSession's IMU setup."""

    def open(self) -> None:
        from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig

        signs = [int(os.environ.get(name, default)) for name, default in
                 (("MOTOR_LEFT_SIGN", "-1"), ("MOTOR_RIGHT_SIGN", "1"),
                  ("MOTOR_FORWARD_TARGET_SIGN", "-1"))]
        if any(value not in (-1, 1) for value in signs):
            raise ValueError("motor signs must each be -1 or +1")
        self.forward_signs = {"left": signs[0] * signs[2], "right": signs[1] * signs[2]}
        config = MssdMotorConfig(
            port=os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0"),
            slave_id=int(os.environ.get("MOTOR_RS485_SLAVE_ID", "1")),
            baudrate=int(os.environ.get("MOTOR_RS485_BAUDRATE", "115200")),
            timeout=SERIAL_TIMEOUT,
            lib_dir=os.environ.get("MOTOR_RS485_LIB_DIR", "/home/topeet/lianzhan"),
            max_target=MAX_RPM, percent_limit=100, left_sign=signs[0], right_sign=signs[1],
            forward_target_sign=signs[2], m1_is_left_wheel=False,
            exit_parking_mode_on_arm=False, stop_mode="emergency", stop_zero_delay_sec=0,
            parking_current_a=0, startup_parking_enabled=False,
        )
        # Opening creates the connection only. Cleanup can still access the driver
        # if any following startup write/read fails.
        self.backend = MssdMotorBackend(config)
        self.backend.ensure_driver()
        self.stop("step_startup")
        self.backend.set_parking_current(0.0, persist=False)
        # Preserve diagnostics even when validation/read fails before the
        # first trial. These are read-only; never change mode or ramp settings.
        self.controller_diagnostics = {}
        self.bus_status = self.backend.driver.read_bus_status()
        self.controller_diagnostics["bus_status"] = dict(self.bus_status)
        configured = self.backend.driver.read_register("system_mode")
        self.controller_diagnostics["configured_system_mode"] = configured
        policy = validate_controller_mode(self.bus_status, configured)
        self.controller_diagnostics["mode_policy"] = policy
        for name in ("foc_loop_mode", "closed_loop_acceleration", "closed_loop_deceleration"):
            try:
                self.controller_diagnostics[name] = self.backend.driver.read_register(name)
            except Exception as exc:
                self.controller_diagnostics[name] = {"read_error": str(exc)}
        print(f"控制器检查：runtime={self.bus_status['runtime_system_mode']} "
              f"configured={configured} control=0x{self.bus_status['control_mode']:02X} "
              f"policy={policy}（不修改模式）")
        print("驱动器加速/减速寄存器（非实测）："
              f"{self.controller_diagnostics['closed_loop_acceleration']} / "
              f"{self.controller_diagnostics['closed_loop_deceleration']} RPM/s")

    def read_wheel(self, side: str) -> dict[str, Any]:
        started = time.monotonic()
        status = self.backend.driver.read_motor_status(side)
        return {"side": side, "read_started": started, "timestamp": time.monotonic(),
                "position_deg": int(status.position_degree), "raw_rpm": int(status.speed_rpm),
                "forward_rpm": self.forward_signs[side] * int(status.speed_rpm),
                "error_code": int(status.error_code)}

    def command(self, rpm: int, trial: dict[str, Any], phase: str) -> None:
        if rpm not in (0, 60, 100):
            raise ValueError("Only 0, 60 and 100 forward RPM commands are permitted")
        for side in ("right", "left"):
            event = {"side": side, "phase": phase, "forward_rpm": rpm,
                     "raw_rpm": self.forward_signs[side] * rpm,
                     "started": time.monotonic(), "completed": None, "acknowledged": False}
            trial["commands"].append(event)
            # No retries or ramp: ambiguous/failed motion writes abort the run.
            getattr(self.backend.driver, f"set_{side}_speed")(event["raw_rpm"])
            event.update(completed=time.monotonic(), acknowledged=True)
            if event["completed"] - event["started"] > FEEDBACK_MAX_AGE:
                raise RuntimeError("速度命令应答超时")

    def stop(self, label: str = "step_emergency") -> None:
        if self.backend is None or self.backend.driver is None:
            return
        errors = []
        # Attempt both sides even if one write fails. Emergency is independent of
        # zero-target acknowledgements and does not introduce a parking lock.
        for side in ("right", "left"):
            for operation in (lambda s=side: getattr(self.backend.driver, f"set_{s}_speed")(0),
                              lambda s=side: self.backend.driver.stop(s, 1)):
                try:
                    operation()
                except Exception as exc:
                    errors.append(f"{side}: {exc}")
        if errors:
            raise RuntimeError(f"{label}: " + "; ".join(errors))

    def close(self, ensure_stop: bool = True) -> None:
        if self.backend is None or self.backend.driver is None:
            return
        errors = []
        try:
            if ensure_stop:
                try:
                    self.stop("step_final_stop")
                except Exception as exc:
                    errors.append(str(exc))
            for side in ("right", "left"):
                try:
                    self._write_register_with_retry(f"{side}_parking_current", 0.0)
                    if abs(self._read_register_with_retry(f"{side}_parking_current")) > 0.005:
                        raise RuntimeError(f"{side} parking current did not reach zero")
                except Exception as exc:
                    errors.append(str(exc))
        finally:
            try:
                self.backend.driver.close()
            finally:
                self.backend.driver = None
        if errors:
            raise RuntimeError("cleanup: " + "; ".join(errors))


class Watchdog:
    def __init__(self, args: argparse.Namespace, signs: dict[str, int], clock=time.monotonic):
        self.args, self.signs, self.clock = args, signs, clock
        self.last: dict[str, dict[str, Any]] = {}
        self.total = {"left": 0.0, "right": 0.0}
        self.trial_start = self.total.copy()
        self.started = clock()

    def begin_trial(self) -> None:
        self.trial_start = self.total.copy()

    def check(self, row: dict[str, Any]) -> None:
        now = self.clock()
        side = row["side"]
        previous = self.last.get(side)
        age = now - row["timestamp"]
        latency = row["timestamp"] - row["read_started"]
        if not all(math.isfinite(row[key]) for key in
                   ("timestamp", "read_started", "raw_rpm", "forward_rpm", "position_deg", "error_code")):
            raise RuntimeError("反馈包含非有限值")
        if not 0 <= age <= FEEDBACK_MAX_AGE or not 0 <= latency <= FEEDBACK_MAX_AGE:
            raise RuntimeError("反馈过期或读取耗时过长")
        if previous and not 0 < row["timestamp"] - previous["timestamp"] <= FEEDBACK_MAX_AGE:
            raise RuntimeError("反馈时间不递增或样本间隔过长")
        if row["error_code"] != 0:
            raise RuntimeError(f"{side} motor fault={row['error_code']}")
        if row["forward_rpm"] < -QUIET_RPM:
            raise RuntimeError(f"{side} 意外反转")
        if abs(row["raw_rpm"]) > MAX_RPM + 10:
            raise RuntimeError(f"{side} 实测速度超过 110 RPM 保护阈值")
        row["delta_deg"] = 0
        if previous:
            delta = _unwrap_i32_delta(row["position_deg"], previous["position_deg"])
            dt = row["timestamp"] - previous["timestamp"]
            if abs(delta) > (MAX_RPM + 10) * 6 * dt + 5:
                raise RuntimeError(f"{side} 编码器突跳")
            if delta * self.signs[side] < -2:
                raise RuntimeError(f"{side} 编码器反向位移")
            self.total[side] += abs(delta) / 360 * math.pi * self.args.wheel_diameter
            row["delta_deg"] = delta
        row["total_path_m"] = self.total[side]
        row["trial_path_m"] = self.total[side] - self.trial_start[side]
        self.last[side] = row
        if row["trial_path_m"] >= self.args.max_travel:
            raise RuntimeError(f"{side} 达到单次编码器路程上限")
        if row["total_path_m"] >= self.args.max_total_travel:
            raise RuntimeError(f"{side} 达到整组编码器路程上限")
        if now - self.started > 60:
            raise RuntimeError("整组测试超过 60 秒")


def _sample(session, trial, phase, watchdog, clock):
    for side in ("left", "right"):
        row = session.read_wheel(side)
        row.update(trial=trial["index"], phase=phase, trusted=False)
        trial["samples"].append(row)  # Preserve the rejected reading as raw evidence.
        watchdog.check(row)
        row["trusted"] = True


def _wait_quiet(session, trial, phase, watchdog, timeout, clock, sleep):
    deadline = clock() + timeout
    quiet_since = {"left": None, "right": None}
    while clock() < deadline:
        _sample(session, trial, phase, watchdog, clock)
        for row in trial["samples"][-2:]:
            side = row["side"]
            if abs(row["forward_rpm"]) <= QUIET_RPM and abs(row["delta_deg"]) <= 2:
                if quiet_since[side] is None:
                    quiet_since[side] = row["timestamp"]
            else:
                quiet_since[side] = None
        if all(value is not None and clock() - value >= QUIET_SEC for value in quiet_since.values()):
            return
        sleep(SAMPLE_PERIOD)
    raise RuntimeError(f"{phase}: 未在时限内确认双轮静止")


def run_experiment(session, args, rpms, payload, clock=time.monotonic, sleep=time.sleep):
    """Mutate payload as samples arrive so callers can save any partial failure."""
    watchdog = Watchdog(args, session.forward_signs, clock)
    for _ in range(args.repeats):
        for rpm in rpms:
            trial = {"index": len(payload["trials"]) + 1, "target_rpm": rpm,
                     "status": "running", "commands": [], "samples": []}
            payload["trials"].append(trial)
            watchdog.begin_trial()
            _wait_quiet(session, trial, "baseline", watchdog, args.settle_timeout, clock, sleep)
            session.command(rpm, trial, "step")
            deadline = min(event["started"] for event in trial["commands"]) + args.duration
            while clock() < deadline:
                _sample(session, trial, "step", watchdog, clock)
                sleep(min(SAMPLE_PERIOD, max(0, deadline - clock())))
            session.command(0, trial, "zero")
            _wait_quiet(session, trial, "stop", watchdog, args.settle_timeout, clock, sleep)
            trial["status"] = "complete"


def analyze_trial(trial: dict[str, Any], wheel_diameter: float) -> dict[str, Any]:
    """Use observed sample crossings only; never substitute commanded speed.

    Times are first observed threshold samples relative to each wheel's write
    start, with read/command intervals retained in raw data. No interpolation
    claims extra precision between sequential RS485 samples.
    """
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        rows = [row for row in trial["samples"] if row["side"] == side and row.get("trusted")]
        events = {event["phase"]: event for event in trial["commands"]
                  if event["side"] == side and event.get("acknowledged")}
        step, zero = events.get("step"), events.get("zero")
        metrics: dict[str, Any] = {"onset_sec": None, "t50_sec": None, "t90_sec": None,
                                  "average_acceleration_t50_m_s2": None,
                                  "average_acceleration_t90_m_s2": None,
                                  "stop_latency_sec": None, "stop_encoder_travel_m": None,
                                  "stop_travel_sample_start": None,
                                  "total_encoder_travel_m": None, "unknown_reasons": [],
                                  "peak_forward_rpm": None, "overshoot_percent": None,
                                  "max_sample_gap_sec": None, "max_read_duration_sec": None,
                                  "step_ack_duration_sec": None, "zero_ack_duration_sec": None}
        if rows:
            metrics["total_encoder_travel_m"] = rows[-1].get("trial_path_m")
            metrics["max_read_duration_sec"] = max(row["timestamp"] - row["read_started"] for row in rows)
            if len(rows) > 1:
                metrics["max_sample_gap_sec"] = max(b["timestamp"] - a["timestamp"] for a, b in zip(rows, rows[1:]))
        if step:
            metrics["step_ack_duration_sec"] = step["completed"] - step["started"]
            rising = [row for row in rows if row["timestamp"] >= step["completed"]
                      and (not zero or row["timestamp"] < zero["started"])]
            if rising:
                metrics["peak_forward_rpm"] = max(row["forward_rpm"] for row in rising)
                metrics["overshoot_percent"] = max(0.0, (metrics["peak_forward_rpm"] / trial["target_rpm"] - 1) * 100)
            for key, threshold in (("onset_sec", max(QUIET_RPM + 1, trial["target_rpm"] * .05)),
                                   ("t50_sec", trial["target_rpm"] * .5),
                                   ("t90_sec", trial["target_rpm"] * .9)):
                crossing = next((row for row in rising if row["forward_rpm"] >= threshold), None)
                if crossing:
                    metrics[key] = crossing["timestamp"] - step["started"]
                    if key != "onset_sec" and metrics[key] > 0:
                        metrics[f"average_acceleration_{key[:3]}_m_s2"] = (
                            threshold * math.pi * wheel_diameter / 60 / metrics[key])
                else:
                    metrics["unknown_reasons"].append(f"{key}: threshold not observed before zero command")
        else:
            metrics["unknown_reasons"].append("step command not acknowledged")
        if zero:
            metrics["zero_ack_duration_sec"] = zero["completed"] - zero["started"]
            stopping = [row for row in rows if row["read_started"] >= zero["completed"]]
            quiet_start = None
            stopped = None
            for row in stopping:
                if abs(row["forward_rpm"]) <= QUIET_RPM and abs(row.get("delta_deg", 0)) <= 2:
                    if quiet_start is None:
                        quiet_start = row
                    if row["timestamp"] - quiet_start["timestamp"] >= QUIET_SEC:
                        stopped = quiet_start
                        break
                else:
                    quiet_start = None
            before_zero = [row for row in rows if row["timestamp"] <= zero["started"]]
            if stopped:
                metrics["stop_latency_sec"] = stopped["timestamp"] - zero["started"]
                if before_zero and all("total_path_m" in row for row in (before_zero[-1], stopped)):
                    # Last pre-zero sample yields a conservative sampled interval,
                    # not an invented exact position at the command instant.
                    metrics["stop_encoder_travel_m"] = stopped["total_path_m"] - before_zero[-1]["total_path_m"]
                    metrics["stop_travel_sample_start"] = before_zero[-1]["timestamp"]
            else:
                metrics["unknown_reasons"].append("stop not observed with 0.25 s sustained quiet")
        else:
            metrics["unknown_reasons"].append("zero command not acknowledged")
        result[side] = metrics
    return result


def _json_safe(value):
    # Invalid samples remain diagnostic raw evidence, but must not make a fault
    # run impossible to serialize as standard JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite": str(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_results(directory: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = directory / f"forward_step_response_{stamp}_{os.getpid()}"
    json_path, csv_path = stem.with_suffix(".json"), stem.with_suffix(".csv")
    for trial in payload["trials"]:
        trial["analysis"] = analyze_trial(trial, payload["parameters"]["wheel_diameter"])
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    fields = ["trial", "phase", "side", "read_started", "timestamp", "raw_rpm", "forward_rpm",
              "position_deg", "error_code", "delta_deg", "trial_path_m", "total_path_m", "trusted"]
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in payload["trials"]:
            writer.writerows(trial["samples"])
    return json_path, csv_path


def print_summary(payload: dict[str, Any]) -> None:
    def seconds(value):
        return "n/a（未观测）" if value is None else f"{value:.3f}s"
    for trial in payload["trials"]:
        print(f"试验 {trial['index']} / {trial['target_rpm']} RPM / {trial['status']}：")
        for side, metrics in trial.get("analysis", {}).items():
            print(f"  {side}: onset={seconds(metrics['onset_sec'])} "
                  f"t50={seconds(metrics['t50_sec'])} t90={seconds(metrics['t90_sec'])} "
                  f"stop={seconds(metrics['stop_latency_sec'])}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    test_lock = None
    try:
        rpms = validate_args(args)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    print(f"{'执行计划' if args.execute else 'dry-run（不打开硬件）'}：{rpms} RPM，每档 {args.repeats} 次，"
          f"每次 {args.duration:.2f}s；0→目标→0，不加软件斜坡。")
    print(f"轮径 {args.wheel_diameter * 100:g}cm；单次/整组编码器路程上限 "
          f"{args.max_travel:g}/{args.max_total_travel:g}m；停车观测最长 {args.settle_timeout:g}s。")
    print("编码器路程并非地面位移真值；串口采样保护不是硬件急停，需平地、足够直线净空及现场急停。")
    if not args.execute:
        print("实际测试须添加 --execute 并现场输入 STEP；建议先 --rpms 60。")
        return 0
    try:
        from car_control_modular.config_loader import load_config_to_env
        config = Path(args.config).expanduser().resolve()
        if not config.is_file():
            raise ValueError(f"配置文件不存在：{config}")
        load_config_to_env(str(config))
        test_lock = acquire_test_lock(os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0"))
        _ensure_follow_runtime_stopped()
        if not sys.stdin.isatty():
            raise RuntimeError("--execute 必须在交互终端现场确认")
        clearance = max(6.0, args.max_total_travel + 2.0)
        print(f"请确认行驶方向，至少 {clearance:g}m 平坦直线净空，路径内无人、无需人体目标，并准备硬件急停；"
              "退出驻车电流保持 0A。")
        if input("输入 STEP 后回车开始：").strip() != "STEP":
            raise RuntimeError("未收到 STEP，测试取消")
        _ensure_follow_runtime_stopped()
        directory = Path(args.output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        # Verify output writability before opening the motor port.
        import tempfile
        with tempfile.TemporaryFile(dir=directory):
            pass
    except (Exception, KeyboardInterrupt) as exc:
        if test_lock is not None:
            test_lock.close()
        print(f"未启动电机：{exc}", file=sys.stderr)
        return 2
    session = ForwardSession()
    payload = {"created_at": datetime.now().astimezone().isoformat(), "mode": "forward_step_response",
               "status": "running", "parameters": vars(args), "trials": [], "cleanup_errors": [],
               "measurement_definitions": {"quiet_rpm": QUIET_RPM, "quiet_sec": QUIET_SEC,
                                           "onset_rpm": "max(3, 0.05 * target_rpm)",
                                           "feedback_max_age_sec": FEEDBACK_MAX_AGE,
                                           "serial_timeout_sec": SERIAL_TIMEOUT,
                                           "nominal_sample_period_sec": SAMPLE_PERIOD,
                                           "encoder_degrees_per_revolution": 360},
               "notes": ["encoder distance assumes 360 degrees/revolution and given wheel diameter; not ground truth",
                         "times are sampled threshold observations, relative to each wheel write start",
                         "acceleration is average threshold velocity / elapsed time, not peak acceleration",
                         "stop travel starts at last pre-zero sample; includes its gap to the zero command",
                         "missing crossings are null; normal stopping uses zero speed target",
                         "100 RPM command ceiling; controller ramp registers are not changed",
                         "parking current remains zero in RAM; previous current is never restored"]}
    previous_handler = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    result = 0
    try:
        session.open()
        payload["forward_signs"] = session.forward_signs
        payload["bus_status"] = session.bus_status
        run_experiment(session, args, rpms, payload)
        payload["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        result = 130 if isinstance(exc, KeyboardInterrupt) else 1
        payload.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        if payload["trials"] and payload["trials"][-1]["status"] == "running":
            payload["trials"][-1].update(status="aborted", error=payload["error"])
        print(f"测试中止，正在急停：{exc}", file=sys.stderr)
    finally:
        # A second Ctrl+C/SIGTERM must not skip the bounded cleanup operations.
        old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            payload["controller_diagnostics"] = getattr(session, "controller_diagnostics", {})
            if hasattr(session, "bus_status"):
                payload["bus_status"] = session.bus_status
            try:
                session.close(ensure_stop=True)
            except Exception as exc:
                payload["cleanup_errors"].append(str(exc))
                payload["status"] = "cleanup_failed"
                result = result or 1
                print(f"清理失败，请使用现场急停：{exc}", file=sys.stderr)
            payload["finished_at"] = datetime.now().astimezone().isoformat()
            try:
                paths = save_results(directory, payload)
                print_summary(payload)
                print(f"结果（含原始样本/部分失败数据）：{paths[0]}\nCSV：{paths[1]}")
            except Exception as exc:
                result = result or 1
                print(f"结果保存失败：{exc}", file=sys.stderr)
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, previous_handler)
            test_lock.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
