#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for candidate in (str(ROOT), str(TOOLS_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from imu_turn_calibrate import (  # noqa: E402
    HardwareSession,
    TURN_ENCODER_SIGN,
    YawIntegrator,
    _collect_static,
    _ensure_follow_runtime_stopped,
    _prime_integrator,
    _print_static,
    _save_json,
    _unwrap_i32_delta,
)


MAX_SAFE_TRACE_SEC = 3.0
MAX_SAFE_TRACE_RPM = 30
MIN_TRACE_SAMPLE_SEC = 0.02


def _sample_row(
    session: HardwareSession,
    integrator: YawIntegrator,
    *,
    started: float,
    phase: str,
    before_left: int,
    before_right: int,
) -> Dict[str, Any]:
    motors = session.read_motors()
    return {
        "elapsed_sec": time.monotonic() - started,
        "phase": phase,
        "yaw_deg": integrator.angle_deg,
        "yaw_rate_dps": integrator.last_rate_dps,
        "left_position_deg": motors.left_position_deg,
        "right_position_deg": motors.right_position_deg,
        "left_delta_deg": _unwrap_i32_delta(motors.left_position_deg, before_left),
        "right_delta_deg": _unwrap_i32_delta(motors.right_position_deg, before_right),
        "left_speed_rpm": motors.left_speed_rpm,
        "right_speed_rpm": motors.right_speed_rpm,
        "left_error": motors.left_error,
        "right_error": motors.right_error,
    }


def _trace_phase(
    session: HardwareSession,
    integrator: YawIntegrator,
    rows: List[Dict[str, Any]],
    *,
    started: float,
    phase: str,
    duration_sec: float,
    sample_sec: float,
    before_left: int,
    before_right: int,
) -> None:
    if session.imu is None:
        raise RuntimeError("IMU 未初始化")
    deadline = time.monotonic() + duration_sec
    next_motor_sample = time.monotonic()
    while time.monotonic() < deadline:
        snapshot = session.imu.poll(0.005)
        integrator.feed(snapshot.get("gyro"))
        now = time.monotonic()
        if now >= next_motor_sample:
            rows.append(
                _sample_row(
                    session,
                    integrator,
                    started=started,
                    phase=phase,
                    before_left=before_left,
                    before_right=before_right,
                )
            )
            next_motor_sample = now + sample_sec


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _command_trace_turn(
    session: HardwareSession,
    direction: str,
    rpm: int,
) -> tuple[int, int]:
    """Allow a bounded 30 RPM trace without relaxing formal calibration limits."""
    if session.backend is None:
        raise RuntimeError("电机后端未初始化")
    if not 1 <= int(rpm) <= MAX_SAFE_TRACE_RPM:
        raise ValueError(f"连续跟踪 RPM 必须在 1..{MAX_SAFE_TRACE_RPM}")
    if direction not in TURN_ENCODER_SIGN:
        raise ValueError("direction 必须是 left 或 right")
    target = int(TURN_ENCODER_SIGN[direction]) * int(rpm)
    session._send_targets_with_retry(  # noqa: SLF001 - shared hardware safety path
        target,
        target,
        f"encoder_trace_{direction}_{rpm}rpm",
    )
    return target, target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="连续记录左右编码器、实时转速和 ICM20600 yaw，定位低速转向位置跳变"
    )
    parser.add_argument("--execute", action="store_true", help="确认允许脚本驱动车轮")
    parser.add_argument("--direction", choices=("left", "right"), default="left")
    parser.add_argument("--rpm", type=int, default=8)
    parser.add_argument(
        "--drive-sec",
        type=float,
        default=0.60,
        help=f"持续转动时间，范围 0.10..{MAX_SAFE_TRACE_SEC:.2f} 秒",
    )
    parser.add_argument("--coast-sec", type=float, default=2.0)
    parser.add_argument("--sample-sec", type=float, default=0.10)
    parser.add_argument("--stationary-sec", type=float, default=6.0)
    parser.add_argument(
        "--stop-mode",
        choices=("zero", "emergency", "normal-parking", "reverse-brake"),
        default="zero",
        help="zero=清零，emergency=急停，normal-parking=5A锁相，reverse-brake=反向扭矩后急停",
    )
    parser.add_argument("--brake-sec", type=float, default=0.20)
    parser.add_argument("--output-dir", default="calibration/encoder_trace")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not args.execute:
        print("错误：该模式会驱动车轮；确认安全后添加 --execute", file=sys.stderr)
        return 1
    if not 1 <= args.rpm <= MAX_SAFE_TRACE_RPM:
        print(f"错误：RPM 必须在 1..{MAX_SAFE_TRACE_RPM}", file=sys.stderr)
        return 1
    if not 0.10 <= args.drive_sec <= MAX_SAFE_TRACE_SEC:
        print(f"错误：drive-sec 必须在 0.10..{MAX_SAFE_TRACE_SEC:.2f}", file=sys.stderr)
        return 1
    if not 0.5 <= args.coast_sec <= 5.0:
        print("错误：coast-sec 必须在 0.5..5.0", file=sys.stderr)
        return 1
    if not MIN_TRACE_SAMPLE_SEC <= args.sample_sec <= 0.25:
        print(
            f"错误：sample-sec 必须在 {MIN_TRACE_SAMPLE_SEC:.2f}..0.25",
            file=sys.stderr,
        )
        return 1
    if not 0.05 <= args.brake_sec <= 0.40:
        print("错误：brake-sec 必须在 0.05..0.40", file=sys.stderr)
        return 1

    _ensure_follow_runtime_stopped()
    print(
        f"将连续记录 {args.direction} {args.rpm}RPM × {args.drive_sec:.2f}s，"
        f"停车后继续记录 {args.coast_sec:.1f}s。"
    )
    if input("确认周围安全，输入 TRACE 后开始：").strip() != "TRACE":
        print("未收到 TRACE，已取消")
        return 1

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"imu_encoder_trace_{args.direction}_{stamp}.json"
    csv_path = output_dir / f"imu_encoder_trace_{args.direction}_{stamp}.csv"

    session = HardwareSession()
    session.open()
    rows: List[Dict[str, Any]] = []
    try:
        session.disable_parking_for_calibration()
        session.stop("imu_encoder_trace_prepare")
        time.sleep(0.30)
        calibration = _collect_static(session, args.stationary_sec)
        _print_static(calibration)
        if not calibration.trustworthy:
            raise RuntimeError(f"静态 IMU 不可信，拒绝动作：{calibration.reasons}")

        before = session.read_motors()
        integrator = YawIntegrator(calibration)
        _prime_integrator(session, integrator)
        if args.stop_mode == "normal-parking":
            if session.backend is None:
                raise RuntimeError("电机后端未初始化")
            # 驻车电流只是 NORMAL 锁相的最大电流，启用后仍可正常发送运动命令。
            session.backend.set_parking_current(5.0, persist=False)
        started = time.monotonic()
        rows.append(
            _sample_row(
                session,
                integrator,
                started=started,
                phase="before",
                before_left=before.left_position_deg,
                before_right=before.right_position_deg,
            )
        )
        left_command, right_command = _command_trace_turn(
            session,
            args.direction,
            args.rpm,
        )
        _trace_phase(
            session,
            integrator,
            rows,
            started=started,
            phase="drive",
            duration_sec=args.drive_sec,
            sample_sec=args.sample_sec,
            before_left=before.left_position_deg,
            before_right=before.right_position_deg,
        )
        if args.stop_mode == "reverse-brake":
            opposite = "right" if args.direction == "left" else "left"
            session.command_turn(opposite, args.rpm)
            _trace_phase(
                session,
                integrator,
                rows,
                started=started,
                phase="reverse_brake",
                duration_sec=args.brake_sec,
                sample_sec=max(0.05, min(args.sample_sec, 0.08)),
                before_left=before.left_position_deg,
                before_right=before.right_position_deg,
            )
            session.stop("imu_encoder_trace_reverse_brake_stop")
            stop_phase = "reverse_stop"
        elif args.stop_mode == "normal-parking":
            if session.backend is None:
                raise RuntimeError("电机后端未初始化")
            session.backend.send_stop("imu_encoder_trace_normal_parking", mode="normal")
            stop_phase = "normal_parking"
        elif args.stop_mode == "emergency":
            session.stop("imu_encoder_trace_emergency_stop")
            stop_phase = "emergency_stop"
        else:
            session.zero_targets("imu_encoder_trace_zero_targets")
            stop_phase = "zero_target"
        rows.append(
            _sample_row(
                session,
                integrator,
                started=started,
                phase=stop_phase,
                before_left=before.left_position_deg,
                before_right=before.right_position_deg,
            )
        )
        _trace_phase(
            session,
            integrator,
            rows,
            started=started,
            phase="coast",
            duration_sec=args.coast_sec,
            sample_sec=args.sample_sec,
            before_left=before.left_position_deg,
            before_right=before.right_position_deg,
        )
        session.stop("imu_encoder_trace_finish")
        payload = {
            "mode": "imu_encoder_trace",
            "created_at": datetime.now().astimezone().isoformat(),
            "direction": args.direction,
            "command_rpm": {"left": left_command, "right": right_command},
            "drive_sec": args.drive_sec,
            "coast_sec": args.coast_sec,
            "sample_sec": args.sample_sec,
            "stop_mode": args.stop_mode,
            "brake_sec": args.brake_sec if args.stop_mode == "reverse-brake" else 0.0,
            "encoder_expected_sign": TURN_ENCODER_SIGN[args.direction],
            "static_calibration": asdict(calibration),
            "imu_samples": integrator.sample_count,
            "imu_gap_count": integrator.gap_count,
            "rows": rows,
        }
        _save_json(json_path, payload)
        _write_csv(csv_path, rows)
        print("时间序列：", csv_path)
        print("完整结果：", json_path)
        for row in rows:
            print(
                "%6.3fs %-11s encoder=%+4d/%+4d° speed=%+3d/%+3dRPM yaw=%+7.3f° rate=%+7.2f°/s"
                % (
                    row["elapsed_sec"],
                    row["phase"],
                    row["left_delta_deg"],
                    row["right_delta_deg"],
                    row["left_speed_rpm"],
                    row["right_speed_rpm"],
                    row["yaw_deg"],
                    row["yaw_rate_dps"],
                )
            )
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    finally:
        try:
            session.stop("imu_encoder_trace_abort_or_finish")
        finally:
            session.close(ensure_stop=True)


if __name__ == "__main__":
    raise SystemExit(main())
