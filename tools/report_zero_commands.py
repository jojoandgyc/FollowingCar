#!/usr/bin/env python3
"""Read-only report of motor_zero_audit records. No project/hardware imports."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


MARKER = "motor_zero_audit "
OUTCOMES = ("acknowledged", "failed", "skipped")
_LOG_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?")


def read_audit_events(path):
    """Parse only explicit structured evidence; legacy labels are not guesses."""
    events, malformed = [], 0
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if MARKER not in line:
                continue
            prefix, encoded = line.split(MARKER, 1)
            try:
                event = json.loads(encoded)
                if (not isinstance(event, dict)
                        or event.get("event_kind") not in {"zero_speed", "stop"}
                        or event.get("outcome") not in OUTCOMES
                        or not isinstance(event.get("context"), dict)
                        or not isinstance(event.get("source"), str)
                        or not isinstance(event.get("phase"), str)
                        or event.get("writes_complete") is not (event["outcome"] == "acknowledged")):
                    raise ValueError("invalid audit schema")
            except (ValueError, TypeError):
                malformed += 1
                continue
            event = dict(event)
            event["line_number"] = line_number
            matched_time = _LOG_TIME.search(prefix)
            event["log_time"] = matched_time.group(0) if matched_time else None
            events.append(event)
    return events, malformed


def _display(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return "/".join(_display(item) for item in value)
    return str(value)


def format_report(events, malformed=0, *, details=False):
    if not events:
        result = ["未发现有效 motor_zero_audit 结构化记录；旧日志不能可靠追溯每条零速的生成原因。",
                  "本工具不会根据相邻 CAP、旧电机标签或未确认的推测补填原因。"]
        if malformed:
            result.append(f"另有 {malformed} 条审计记录无法解析或字段不一致。")
        return "\n".join(result)
    counts = Counter((event["event_kind"], event["outcome"]) for event in events)
    lines = [f"审计事件 {len(events)} 条；无法解析或字段不一致 {malformed} 条。",
             "类型                  双侧写入已确认  写入失败  未写入/跳过"]
    for kind in ("zero_speed", "stop"):
        lines.append(f"{kind:<21} {counts[kind, 'acknowledged']:>8}"
                     f" {counts[kind, 'failed']:>10} {counts[kind, 'skipped']:>12}")
    lines.append("仅统计指令 ACK；不代表车轮已停稳，失败可能已部分执行。不能由这些记录单独计算停车时长。")
    grouped = Counter((event["event_kind"], event["source"], event["phase"],
                       _display(event["context"].get("decision_stage")),
                       _display(event["context"].get("zero_reason")), event["outcome"])
                      for event in events)
    keys = sorted({key[:-1] for key in grouped})
    lines.append("\n按来源 / 生成阶段 / 原因汇总（确认 / 失败 / 跳过）：")
    for kind, source, phase, stage, reason in keys:
        key = (kind, source, phase, stage, reason)
        values = "/".join(str(grouped[key + (outcome,)]) for outcome in OUTCOMES)
        lines.append(f"  {values} {kind} source={source} phase={phase} stage={stage} reason={reason}")
    if details:
        lines.append("\n逐条记录（command/control/observation CAP 分开，不把异步观测当成动作依据）：")
        for event in events:
            ctx = event["context"]
            clock = event["log_time"] or f"mono={_display(event.get('generated_at'))}"
            caps = "/".join(_display(ctx.get(key)) for key in
                            ("command_cap", "control_cap", "observation_cap"))
            lines.append(
                f"  line={event['line_number']} {clock} id={_display(event.get('event_id'))} "
                f"CAP(cmd/ctrl/obs)={caps} UID={_display(ctx.get('uid'))} "
                f"command_UID={_display(ctx.get('command_uid'))} {event['event_kind']} "
                f"{event['outcome']} source={event['source']} phase={event['phase']} "
                f"stage={_display(ctx.get('decision_stage'))} reason={_display(ctx.get('zero_reason'))}\n"
                f"    previous_raw={_display(event.get('previous_acknowledged_raw'))} "
                f"previous_forward/yaw={_display(event.get('previous_forward_yaw'))} "
                f"requested_raw={_display(event.get('requested_raw'))} "
                f"planned_axes={_display(ctx.get('planned_axes'))} "
                f"ACK={_display(event.get('acknowledged_sides'))} "
                f"failed={_display(event.get('failed_sides'))} "
                f"depth_age_ms={_display(ctx.get('depth_age_ms'))} "
                f"feedback_age_ms={_display(ctx.get('feedback_age_ms'))} "
                f"receipt={_display(event.get('receipt_after'))} "
                f"STOPgen={_display(event.get('stop_generation'))}\n"
                f"    command_reason={_display(ctx.get('command_reason'))} "
                f"control_reason={_display(ctx.get('control_reason'))} "
                f"vision_state={_display(ctx.get('vision_state'))} "
                f"depth_read_veto={_display(ctx.get('depth_read_veto'))} "
                f"packet_veto={_display(ctx.get('packet_veto'))} "
                f"generated_mono={_display(event.get('generated_at'))} "
                f"completed_mono={_display(event.get('completed_at'))}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="request_0513_modular.log 路径，只读")
    parser.add_argument("--details", action="store_true", help="逐条显示 CAP、原因、轮包与时效")
    args = parser.parse_args(argv)
    try:
        events, malformed = read_audit_events(args.log)
    except OSError as exc:
        print(f"无法读取日志：{exc}", file=sys.stderr)
        return 2
    print(format_report(events, malformed, details=args.details))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
