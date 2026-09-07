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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
LZ_ROOT = Path(os.environ.get("LZ30EMA_LIB_DIR", "/home/topeet/lianzhan"))
LZ_SRC = LZ_ROOT / "src"
if str(LZ_SRC) not in sys.path:
    sys.path.insert(0, str(LZ_SRC))

PARKING_PID_FIELDS = (
    "encoder_parking_speed_p",
    "encoder_parking_speed_i",
    "encoder_parking_speed_d",
    "encoder_parking_d_axis_p",
    "encoder_parking_d_axis_i",
    "encoder_parking_d_axis_d",
    "encoder_parking_q_axis_p",
    "encoder_parking_q_axis_i",
    "encoder_parking_q_axis_d",
)
TURN_SIGN = {"left": -1, "right": 1}


def _retry(label: str, action: Callable[[], Any], attempts: int = 4) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return action()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.03 * (attempt + 1))
    raise RuntimeError(f"{label} 连续 {attempts} 次失败：{last_error}") from last_error


def _ensure_port_available(port: str) -> None:
    own_pid = os.getpid()
    target = Path(port)
    owners: List[str] = []
    if not target.exists():
        raise RuntimeError(f"电机串口不存在：{target}")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            descriptors = list((entry / "fd").iterdir())
        except (OSError, PermissionError):
            continue
        found = False
        for descriptor in descriptors:
            try:
                if os.path.samefile(descriptor, target):
                    found = True
                    break
            except (OSError, PermissionError):
                continue
        if not found:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (OSError, PermissionError):
            command = "<无法读取命令行>"
        owners.append(f"pid={entry.name} {command.strip()}")
    if owners:
        raise RuntimeError(f"串口 {port} 被占用：\n" + "\n".join(owners))


def _open_client(args: argparse.Namespace):
    from lz30ema_rs485 import LZ30EMAClient

    _ensure_port_available(args.port)
    return LZ30EMAClient.from_serial(
        args.port,
        slave=args.slave,
        baudrate=args.baudrate,
        timeout=args.timeout,
        parity=args.parity,
        stopbits=args.stopbits,
        rs485_mode=args.rs485_mode,
    )


def _read_pid(client) -> Dict[str, float]:
    return {
        name: float(_retry(f"读取 {name}", lambda name=name: client.read_field(name)))
        for name in PARKING_PID_FIELDS
    }


def _read_parking_current(client) -> Dict[str, float]:
    return {
        "right": float(_retry("读取右驻车电流", lambda: client.read_register("right_parking_current"))),
        "left": float(_retry("读取左驻车电流", lambda: client.read_register("left_parking_current"))),
    }


def _set_parking_current(client, current_a: float) -> None:
    for name in ("right_parking_current", "left_parking_current"):
        _retry(
            f"写入 {name}",
            lambda name=name: client.write_register(name, float(current_a), persist=False),
        )
    readback = _read_parking_current(client)
    if any(abs(value - current_a) > 0.01 for value in readback.values()):
        raise RuntimeError(f"驻车电流回读不一致：目标={current_a}A 回读={readback}")


def _safe_stop(client, *, parking_current_a: float = 0.0) -> None:
    from lz30ema_rs485 import StopMode

    for label, action in (
        ("右轮清零", lambda: client.set_right_speed(0)),
        ("左轮清零", lambda: client.set_left_speed(0)),
        ("双轮急停", lambda: client.stop_all(StopMode.EMERGENCY)),
    ):
        try:
            _retry(label, action)
        except Exception as exc:
            print(f"警告：{label}失败：{exc}", file=sys.stderr)
    try:
        _set_parking_current(client, parking_current_a)
    except Exception as exc:
        print(f"警告：停车后设置驻车电流失败：{exc}", file=sys.stderr)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _output_dir(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_info(args: argparse.Namespace) -> int:
    client = _open_client(args)
    try:
        payload = {
            "created_at": datetime.now().astimezone().isoformat(),
            "parking_current_a": _read_parking_current(client),
            "parking_pid": _read_pid(client),
            "left_status": client.read_motor_status("left").__dict__,
            "right_status": client.read_motor_status("right").__dict__,
        }
        # Enum cannot be serialized directly and is redundant with the object key.
        payload["left_status"]["side"] = "left"
        payload["right_status"]["side"] = "right"
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.output:
            _save_json(Path(args.output).expanduser(), payload)
        return 0
    finally:
        client.close()


def _run_current(args: argparse.Namespace) -> int:
    from lz30ema_rs485 import StopMode

    _require_execute(
        args,
        f"将双轮清零并把驻车电流设置为 {args.current:g}A，停止模式={args.stop_mode}",
    )
    client = _open_client(args)
    try:
        _safe_stop(client, parking_current_a=0.0)
        _set_parking_current(client, float(args.current))
        mode = {
            "normal": StopMode.NORMAL,
            "emergency": StopMode.EMERGENCY,
            "free": StopMode.FREE,
        }[args.stop_mode]
        _retry("设置最终停止模式", lambda: client.stop_all(mode))
        payload = {
            "created_at": datetime.now().astimezone().isoformat(),
            "parking_current_a": _read_parking_current(client),
            "stop_mode": args.stop_mode,
            "left_status": client.read_motor_status("left").__dict__,
            "right_status": client.read_motor_status("right").__dict__,
        }
        payload["left_status"]["side"] = "left"
        payload["right_status"]["side"] = "right"
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    finally:
        # current 子命令的用途就是保留用户指定状态，关闭串口时不再次改写寄存器。
        client.close()


def _position_metrics(rows: Sequence[Dict[str, Any]], side: str) -> Dict[str, Any]:
    positions = [int(row[f"{side}_position_deg"]) for row in rows]
    speeds = [int(row[f"{side}_speed_rpm"]) for row in rows]
    currents = [float(row[f"{side}_current_a"]) for row in rows]
    if not positions:
        return {"samples": 0, "converged": False}
    deltas = [positions[index] - positions[index - 1] for index in range(1, len(positions))]
    nonzero_deltas = [value for value in deltas if value]
    direction_changes = sum(
        1
        for index in range(1, len(nonzero_deltas))
        if (nonzero_deltas[index] > 0) != (nonzero_deltas[index - 1] > 0)
    )
    tail_start = max(0, len(positions) // 2)
    tail_positions = positions[tail_start:]
    tail_speeds = speeds[tail_start:]
    tail_currents = currents[tail_start:]
    tail_deltas = [
        tail_positions[index] - tail_positions[index - 1]
        for index in range(1, len(tail_positions))
    ]
    center = statistics.median(positions)
    rms = math.sqrt(sum((value - center) ** 2 for value in positions) / len(positions))
    tail_range = max(tail_positions) - min(tail_positions)
    tail_travel = sum(abs(value) for value in tail_deltas)
    nonzero_speed_ratio = sum(1 for value in speeds if value != 0) / len(speeds)
    tail_nonzero_speed_ratio = sum(1 for value in tail_speeds if value != 0) / len(tail_speeds)
    converged = tail_range <= 2 and tail_travel <= 4 and tail_nonzero_speed_ratio <= 0.10
    return {
        "samples": len(positions),
        "start_position_deg": positions[0],
        "final_position_deg": positions[-1],
        "drift_deg": positions[-1] - positions[0],
        "peak_to_peak_deg": max(positions) - min(positions),
        "rms_about_median_deg": rms,
        "total_travel_deg": sum(abs(value) for value in deltas),
        "direction_changes": direction_changes,
        "nonzero_speed_ratio": nonzero_speed_ratio,
        "tail_nonzero_speed_ratio": tail_nonzero_speed_ratio,
        "tail_current_mean_a": statistics.fmean(tail_currents),
        "tail_current_std_a": statistics.pstdev(tail_currents) if len(tail_currents) >= 2 else 0.0,
        "tail_current_peak_a": max(tail_currents),
        "tail_current_peak_to_peak_a": max(tail_currents) - min(tail_currents),
        "tail_peak_to_peak_deg": tail_range,
        "tail_total_travel_deg": tail_travel,
        "converged": converged,
    }


def _observe_once(client, args: argparse.Namespace, direction: str) -> Dict[str, Any]:
    from lz30ema_rs485 import StopMode

    target = TURN_SIGN[direction] * int(args.rpm)
    _set_parking_current(client, float(args.parking_current))
    _retry("下发右轮转速", lambda: client.set_right_speed(target))
    _retry("下发左轮转速", lambda: client.set_left_speed(target))
    time.sleep(float(args.drive_sec))
    _retry("停车右轮清零", lambda: client.set_right_speed(0))
    _retry("停车左轮清零", lambda: client.set_left_speed(0))
    stopped_at = time.monotonic()
    _retry("进入正常驻车", lambda: client.stop_all(StopMode.NORMAL))
    if args.stop_sequence == "project":
        # 完整复现跟随项目 MssdMotorBackend.send_stop()：NORMAL 之后再次写入双轮 0 RPM。
        _retry("项目时序停车后右轮清零", lambda: client.set_right_speed(0))
        _retry("项目时序停车后左轮清零", lambda: client.set_left_speed(0))

    rows: List[Dict[str, Any]] = []
    deadline = stopped_at + float(args.observe_sec)
    while time.monotonic() < deadline:
        sample_started = time.monotonic()
        right = _retry("读取右轮状态", lambda: client.read_motor_status("right"))
        left = _retry("读取左轮状态", lambda: client.read_motor_status("left"))
        now = time.monotonic()
        rows.append(
            {
                "elapsed_sec": now - stopped_at,
                "left_position_deg": int(left.position_degree),
                "right_position_deg": int(right.position_degree),
                "left_speed_rpm": int(left.speed_rpm),
                "right_speed_rpm": int(right.speed_rpm),
                "left_current_a": float(left.phase_current_a),
                "right_current_a": float(right.phase_current_a),
            }
        )
        remaining = float(args.sample_sec) - (time.monotonic() - sample_started)
        if remaining > 0:
            time.sleep(remaining)

    left_metrics = _position_metrics(rows, "left")
    right_metrics = _position_metrics(rows, "right")
    return {
        "direction": direction,
        "command_rpm": {"left": target, "right": target},
        "drive_sec": float(args.drive_sec),
        "observe_sec": float(args.observe_sec),
        "parking_current_a": float(args.parking_current),
        "stop_sequence": args.stop_sequence,
        "parking_pid": _read_pid(client),
        "left": left_metrics,
        "right": right_metrics,
        "converged": bool(left_metrics.get("converged") and right_metrics.get("converged")),
        "rows": rows,
    }


def _write_rows(path: Path, observations: Sequence[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for observation in observations:
        for row in observation["rows"]:
            rows.append({"direction": observation["direction"], **row})
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _require_execute(args: argparse.Namespace, message: str) -> None:
    if not args.execute:
        raise RuntimeError(message + "；确认安全后添加 --execute")
    answer = input(f"{message}。输入 PARK 后开始：").strip()
    if answer != "PARK":
        raise RuntimeError("未收到 PARK，已取消")


def _run_observe(args: argparse.Namespace) -> int:
    _require_execute(
        args,
        f"将以 {args.rpm} RPM 左右短转并以 {args.parking_current:g}A 正常驻车",
    )
    directions = ("left", "right") if args.direction == "both" else (args.direction,)
    output_dir = _output_dir(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    client = _open_client(args)
    observations: List[Dict[str, Any]] = []
    try:
        _safe_stop(client, parking_current_a=0.0)
        for direction in directions:
            print(f"测试 {direction}: {args.rpm}RPM × {args.drive_sec:.2f}s，驻车观察 {args.observe_sec:.1f}s")
            observation = _observe_once(client, args, direction)
            observations.append(observation)
            print(
                "%s left(range/tail/travel)=%d/%d/%d° right=%d/%d/%d° converged=%s"
                % (
                    direction,
                    observation["left"]["peak_to_peak_deg"],
                    observation["left"]["tail_peak_to_peak_deg"],
                    observation["left"]["total_travel_deg"],
                    observation["right"]["peak_to_peak_deg"],
                    observation["right"]["tail_peak_to_peak_deg"],
                    observation["right"]["total_travel_deg"],
                    observation["converged"],
                )
            )
            if direction != directions[-1]:
                _safe_stop(client, parking_current_a=0.0)
                time.sleep(float(args.pause_sec))
        payload = {
            "mode": "parking_observe",
            "created_at": datetime.now().astimezone().isoformat(),
            "observations": observations,
            "overall_converged": all(item["converged"] for item in observations),
        }
        json_path = output_dir / f"parking_observe_{stamp}.json"
        csv_path = output_dir / f"parking_observe_{stamp}.csv"
        _save_json(json_path, payload)
        _write_rows(csv_path, observations)
        print("结果：", json_path)
        print("采样：", csv_path)
        return 0 if payload["overall_converged"] else 3
    finally:
        _safe_stop(client, parking_current_a=0.0)
        if args.leave_parking_on:
            print("警告：测试程序退出时始终关闭驻车电流，忽略 --leave-parking-on", file=sys.stderr)
        client.close()


def _run_apply(args: argparse.Namespace) -> int:
    _require_execute(args, "将修改编码器驻车速度 PID，原值会先备份")
    values = {
        "encoder_parking_speed_p": args.speed_p,
        "encoder_parking_speed_i": args.speed_i,
        "encoder_parking_speed_d": args.speed_d,
    }
    if all(value is None for value in values.values()):
        raise ValueError("至少提供 --speed-p/--speed-i/--speed-d 中的一项")
    client = _open_client(args)
    output_dir = _output_dir(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        before = _read_pid(client)
        backup = {
            "mode": "parking_pid_backup",
            "created_at": datetime.now().astimezone().isoformat(),
            "parking_current_a": _read_parking_current(client),
            "parking_pid": before,
        }
        backup_path = output_dir / f"parking_pid_backup_{stamp}.json"
        _save_json(backup_path, backup)
        _safe_stop(client, parking_current_a=0.0)
        for name, value in values.items():
            if value is None:
                continue
            if not 0.0 <= float(value) <= 131071.0:
                raise ValueError(f"{name} 超出 0..131071")
            _retry(f"写入 {name}", lambda name=name, value=value: client.write_field(name, float(value)))
        after = _read_pid(client)
        print("PID 备份：", backup_path)
        print(json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=2))
        return 0
    finally:
        _safe_stop(client, parking_current_a=0.0)
        client.close()


def _run_restore(args: argparse.Namespace) -> int:
    _require_execute(args, "将从备份恢复全部编码器驻车 PID")
    with Path(args.backup).expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("parking_pid", {})
    missing = [name for name in PARKING_PID_FIELDS if name not in values]
    if missing:
        raise RuntimeError(f"备份缺少字段：{missing}")
    client = _open_client(args)
    try:
        _safe_stop(client, parking_current_a=0.0)
        for name in PARKING_PID_FIELDS:
            _retry(f"恢复 {name}", lambda name=name: client.write_field(name, float(values[name])))
        readback = _read_pid(client)
        print(json.dumps(readback, ensure_ascii=False, indent=2))
        return 0
    finally:
        _safe_stop(client, parking_current_a=0.0)
        client.close()


def _add_serial_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", default="/dev/ttyS0")
    parser.add_argument("--slave", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--parity", choices=("N", "E", "O"), default="N")
    parser.add_argument("--stopbits", type=int, choices=(1, 2), default=1)
    parser.add_argument("--rs485-mode", choices=("auto", "none", "rts-high", "rts-low"), default="auto")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LZ30EMA 5A 编码器驻车 PID 振荡观测与调参工具")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    info = subparsers.add_parser("info", help="只读当前驻车电流、PID 和双轮状态")
    _add_serial_args(info)
    info.add_argument("--output", default="")

    current = subparsers.add_parser("current", help="安全设置并回读双轮驻车电流")
    _add_serial_args(current)
    current.add_argument("--execute", action="store_true")
    current.add_argument("--current", type=float, required=True)
    current.add_argument(
        "--stop-mode",
        choices=("emergency", "normal", "free"),
        default="emergency",
    )

    observe = subparsers.add_parser("observe", help="短转后用编码器观测 5A 驻车是否收敛")
    _add_serial_args(observe)
    observe.add_argument("--execute", action="store_true")
    observe.add_argument("--direction", choices=("left", "right", "both"), default="both")
    observe.add_argument("--rpm", type=int, default=8)
    observe.add_argument("--drive-sec", type=float, default=0.50)
    observe.add_argument("--observe-sec", type=float, default=8.0)
    observe.add_argument("--sample-sec", type=float, default=0.05)
    observe.add_argument("--pause-sec", type=float, default=2.0)
    observe.add_argument("--parking-current", type=float, default=5.0)
    observe.add_argument("--stop-sequence", choices=("project", "normal"), default="project")
    observe.add_argument("--leave-parking-on", action="store_true")
    observe.add_argument("--output-dir", default="calibration/parking_pid")

    apply_pid = subparsers.add_parser("apply", help="备份后修改编码器驻车速度 PID")
    _add_serial_args(apply_pid)
    apply_pid.add_argument("--execute", action="store_true")
    apply_pid.add_argument("--speed-p", type=float)
    apply_pid.add_argument("--speed-i", type=float)
    apply_pid.add_argument("--speed-d", type=float)
    apply_pid.add_argument("--output-dir", default="calibration/parking_pid")

    restore = subparsers.add_parser("restore", help="从 JSON 备份恢复全部驻车 PID")
    _add_serial_args(restore)
    restore.add_argument("--execute", action="store_true")
    restore.add_argument("--backup", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        if args.mode == "info":
            return _run_info(args)
        if args.mode == "current":
            if not 0.0 <= args.current <= 8.0:
                raise ValueError("驻车电流必须在 0..8A")
            return _run_current(args)
        if args.mode == "observe":
            if not 1 <= args.rpm <= 16:
                raise ValueError("RPM 必须在 1..16")
            if not 0.1 <= args.drive_sec <= 0.8:
                raise ValueError("drive-sec 必须在 0.1..0.8 秒")
            if not 2.0 <= args.observe_sec <= 30.0:
                raise ValueError("observe-sec 必须在 2..30 秒")
            if not 0.03 <= args.sample_sec <= 0.5:
                raise ValueError("sample-sec 必须在 0.03..0.5 秒")
            if not 0.0 <= args.parking_current <= 8.0:
                raise ValueError("parking-current 必须在 0..8A")
            return _run_observe(args)
        if args.mode == "apply":
            return _run_apply(args)
        if args.mode == "restore":
            return _run_restore(args)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停车退出。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
