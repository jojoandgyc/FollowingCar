#!/usr/bin/env python3
"""低速验证旋转指令、编码器符号和 IMU yaw 的方向映射。

默认只打印测试计划。只有显式传入 ``--execute``，并在现场输入 ``MAP`` 后才会
驱动车轮。测试不修改配置或标定模型，只把原始结果保存为 JSON，便于确认是否
应该反转编码器方向映射。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from car_control_modular.config_loader import load_config_to_env  # noqa: E402
from imu_turn_calibrate import (  # noqa: E402
    DEFAULT_CONFIG,
    ENCODER_DEGREES_PER_REV,
    HardwareSession,
    MAX_SAFE_PULSE_SEC,
    MAX_SAFE_RPM,
    TURN_ENCODER_SIGN,
    _collect_static,
    _ensure_follow_runtime_stopped,
    _require_motion_confirmation,
    _run_trial,
    _save_json,
)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="低速左右转，确认电机指令与编码器/IMU方向符号是否一致。"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true", help="确认安全后实际驱动车轮")
    parser.add_argument("--rpm", type=int, default=6, help="测试转速，范围 1..16 RPM")
    parser.add_argument("--duration", type=float, default=0.35, help="每次脉冲秒数，范围 0.10..0.80")
    parser.add_argument("--repeats", type=int, default=2, help="每个方向重复次数，范围 1..4")
    parser.add_argument("--stationary-sec", type=float, default=3.0)
    parser.add_argument("--settle-sec", type=float, default=0.60)
    parser.add_argument("--settle-timeout-sec", type=float, default=4.0)
    parser.add_argument("--quiet-sec", type=float, default=0.60)
    parser.add_argument("--quiet-yaw-rate-dps", type=float, default=1.0)
    parser.add_argument("--pause-sec", type=float, default=1.5)
    parser.add_argument("--allow-noisy-imu", action="store_true")
    parser.add_argument("--output", default="", help="JSON输出路径；留空则写入 calibration/")
    return parser


def _summarize(trials: List[Any]) -> Dict[str, Any]:
    by_direction: Dict[str, List[Any]] = {"left": [], "right": []}
    for trial in trials:
        by_direction.setdefault(trial.direction, []).append(trial)

    direction_summary: Dict[str, Any] = {}
    encoder_matches: List[bool] = []
    yaw_signs: Dict[str, List[int]] = {"left": [], "right": []}
    for direction, items in by_direction.items():
        expected = int(TURN_ENCODER_SIGN[direction])
        records = []
        for trial in items:
            left_sign = _sign(trial.left_encoder_delta_deg)
            right_sign = _sign(trial.right_encoder_delta_deg)
            encoder_match = left_sign == expected and right_sign == expected
            encoder_matches.append(encoder_match)
            yaw_signs[direction].append(_sign(trial.yaw_final_deg))
            records.append(
                {
                    "index": trial.index,
                    "command_rpm": trial.command_rpm,
                    "command_targets": [trial.left_command_rpm, trial.right_command_rpm],
                    "expected_encoder_sign": expected,
                    "encoder_delta_deg": [trial.left_encoder_delta_deg, trial.right_encoder_delta_deg],
                    "encoder_sign": [left_sign, right_sign],
                    "encoder_match": encoder_match,
                    "imu_yaw_deg": trial.yaw_final_deg,
                    "imu_yaw_sign": _sign(trial.yaw_final_deg),
                    "peak_yaw_rate_dps": trial.peak_abs_yaw_rate_dps,
                    "valid": trial.valid,
                    "invalid_reasons": trial.invalid_reasons,
                }
            )
        direction_summary[direction] = {
            "expected_encoder_sign": expected,
            "trials": records,
            "encoder_match_count": sum(item["encoder_match"] for item in records),
            "trial_count": len(records),
        }

    left_yaw = [value for value in yaw_signs["left"] if value]
    right_yaw = [value for value in yaw_signs["right"] if value]
    yaw_opposite = bool(left_yaw and right_yaw and all(a == -b for a in left_yaw for b in right_yaw))
    if not encoder_matches:
        verdict = "no_samples"
    elif all(encoder_matches):
        verdict = "mapping_consistent"
    elif not any(encoder_matches):
        verdict = "encoder_sign_likely_reversed"
    else:
        verdict = "encoder_sign_intermittent_or_wheel_mismatch"
    if not yaw_opposite:
        verdict += ";imu_yaw_not_opposite"

    return {
        "verdict": verdict,
        "current_mapping": {
            "left": [TURN_ENCODER_SIGN["left"]] * 2,
            "right": [TURN_ENCODER_SIGN["right"]] * 2,
            "encoder_degrees_per_revolution": ENCODER_DEGREES_PER_REV,
        },
        "directions": direction_summary,
        "imu_yaw_signs": yaw_signs,
        "imu_yaw_opposite": yaw_opposite,
    }


def main() -> int:
    args = _build_parser().parse_args()
    if not 1 <= args.rpm <= MAX_SAFE_RPM:
        print(f"错误：rpm 必须在 1..{MAX_SAFE_RPM}", file=sys.stderr)
        return 2
    if not 0.10 <= args.duration <= MAX_SAFE_PULSE_SEC:
        print(f"错误：duration 必须在 0.10..{MAX_SAFE_PULSE_SEC:.2f}s", file=sys.stderr)
        return 2
    if not 1 <= args.repeats <= 4:
        print("错误：repeats 必须在 1..4", file=sys.stderr)
        return 2
    if not args.execute:
        print(
            f"dry-run：将左右各测试 {args.repeats} 次，{args.rpm}RPM × {args.duration:.2f}s；"
            "添加 --execute 后，现场输入 MAP 才会驱动车轮。"
        )
        return 0

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"错误：配置文件不存在：{config_path}", file=sys.stderr)
        return 2
    load_config_to_env(str(config_path))
    _ensure_follow_runtime_stopped()
    _require_motion_confirmation(
        args,
        f"将在平整地面左右交替短转：{args.repeats}×，{args.rpm}RPM × {args.duration:.2f}s；旁边必须有人急停。",
    )

    session = HardwareSession()
    trials: List[Any] = []
    try:
        session.open()
        session.disable_parking_for_calibration()
        session.stop("encoder_mapping_test_start")
        time.sleep(0.30)
        calibration = _collect_static(session, args.stationary_sec)
        if not calibration.trustworthy and not args.allow_noisy_imu:
            raise RuntimeError(f"静态 IMU 不可信：{calibration.reasons}")
        index = 0
        for repeat in range(args.repeats):
            for direction in ("left", "right"):
                index += 1
                trial = _run_trial(
                    session,
                    calibration,
                    index=index,
                    direction=direction,
                    rpm=args.rpm,
                    duration_sec=args.duration,
                    settle_sec=args.settle_sec,
                    settle_timeout_sec=args.settle_timeout_sec,
                    quiet_sec=args.quiet_sec,
                    quiet_yaw_rate_dps=args.quiet_yaw_rate_dps,
                )
                trials.append(trial)
                print(
                    f"{direction:5s} cmd={trial.left_command_rpm:+d}/{trial.right_command_rpm:+d}RPM "
                    f"encoder={trial.left_encoder_delta_deg:+d}/{trial.right_encoder_delta_deg:+d}° "
                    f"yaw={trial.yaw_final_deg:+.2f}° match={trial.encoder_sign_valid}"
                )
                time.sleep(max(0.5, args.pause_sec))

        summary = _summarize(trials)
        payload = {
            "mode": "encoder_turn_mapping_test",
            "created_at": datetime.now().astimezone().isoformat(),
            "parameters": {
                "rpm": args.rpm,
                "duration_sec": args.duration,
                "repeats": args.repeats,
            },
            "static_calibration": asdict(calibration),
            "summary": summary,
            "trials": [asdict(trial) for trial in trials],
        }
        output = Path(args.output).expanduser().resolve() if args.output else ROOT / "calibration" / (
            "encoder_turn_mapping_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        )
        _save_json(output, payload)
        print(f"结论：{summary['verdict']}")
        print(f"结果：{output}")
        return 0 if summary["verdict"] == "mapping_consistent" else 2
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停车。", file=sys.stderr)
        return 130
    finally:
        try:
            session.stop("encoder_mapping_test_finish")
        finally:
            session.close(ensure_stop=True)


if __name__ == "__main__":
    raise SystemExit(main())
