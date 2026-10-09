#!/usr/bin/env python3
"""Measure completed forward-command coverage, not physical wheel motion.

Uses the main runtime's monotonic speed receipts AND acknowledged STOP
register writes. Ordinary speed/yaw updates do not split a forward segment.
No runtime, model, sensor or motor modules are imported.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from statistics import median


SPEED = re.compile(
    r"LZ30EMA 电机命令: .*?左轮=(-?\d+)转/分 右轮=(-?\d+)转/分 .*?"
    r"write_started_at=([\d.]+) write_completed_at=([\d.]+)")


@dataclass(frozen=True)
class Event:
    timestamp: float
    state: str
    line: int


def read_events(lines, *, left_forward_sign=1, right_forward_sign=-1):
    if left_forward_sign not in (-1, 1) or right_forward_sign not in (-1, 1):
        raise ValueError("wheel signs must be -1 or 1")
    events = []
    for number, line in enumerate(lines, 1):
        matched = SPEED.search(line)
        if matched:
            left, right = (int(matched[1]) * left_forward_sign,
                           int(matched[2]) * right_forward_sign)
            started, completed = float(matched[3]), float(matched[4])
            if not (math.isfinite(completed) and 0 < started <= completed):
                continue
            total = left + right
            state = ("forward" if total > 0 else "reverse" if total < 0
                     else "rotation" if left or right else "stopped")
            events.append(Event(completed, state, number))
            continue
        if "motor_zero_audit " not in line:
            continue
        try:
            audit = json.loads(line.split("motor_zero_audit ", 1)[1])
            completed = audit["completed_at"]
            if (not isinstance(completed, (int, float)) or isinstance(completed, bool)
                    or not math.isfinite(completed) or completed <= 0):
                continue
            if audit.get("outcome") == "failed":
                # A partial write is not proof that the previous pair remains
                # applied. Do not count uncertain time as forward coverage.
                events.append(Event(completed, "unknown", number))
            elif (audit.get("event_kind") == "stop"
                    and audit.get("phase") == "stop_registers"
                    and audit.get("outcome") == "acknowledged"
                    and audit.get("writes_complete") is True
                    and set(audit.get("acknowledged_sides", ())) == {"left", "right"}):
                events.append(Event(completed, "stopped", number))
            # Speed zero ACKs are already represented by SPEED. Skipped
            # zero/STOP plans never alter the executed-command timeline.
        except (ValueError, TypeError, KeyError):
            continue
    return sorted(events, key=lambda event: (event.timestamp, event.line))


def summarize(events):
    """Window: first forward ACK through first nonforward after last forward.

    An unfinished final forward segment is reported as right-censored. Its
    duration is a lower bound through the last available receipt, not a guess
    through process exit or the next camera frame.
    """
    result = {"basis": "completed_command_not_physical_motion", "event_count": len(events)}
    forward = [index for index, event in enumerate(events) if event.state == "forward"]
    if not forward:
        return dict(result, forward_segments=[], window_sec=0.0, forward_sec=0.0)
    first, last = forward[0], forward[-1]
    ended = last + 1 < len(events)
    end = last + 1 if ended else last
    selected = events[first:end + 1]
    totals = dict.fromkeys(("forward", "rotation", "reverse", "stopped", "unknown"), 0.0)
    segments = []
    beginning = selected[0]
    for previous, current in zip(selected, selected[1:]):
        totals[previous.state] += current.timestamp - previous.timestamp
        if previous.state == "forward" and current.state != "forward":
            segments.append(dict(start_line=beginning.line, end_line=current.line,
                                 duration_ms=(current.timestamp - beginning.timestamp) * 1000.0,
                                 end_state=current.state, right_censored=False))
        elif previous.state != "forward" and current.state == "forward":
            beginning = current
    if not ended:
        segments.append(dict(start_line=beginning.line, end_line=selected[-1].line,
                             duration_ms=(selected[-1].timestamp - beginning.timestamp) * 1000.0,
                             end_state=None, right_censored=True))
    complete = [part["duration_ms"] for part in segments if not part["right_censored"]]
    window = selected[-1].timestamp - selected[0].timestamp
    return dict(result, window_sec=window, start_line=selected[0].line,
                end_line=selected[-1].line, right_censored=not ended,
                forward_sec=totals["forward"], state_seconds=totals,
                state_fraction={key: value / window if window else 0.0 for key, value in totals.items()},
                forward_segments=segments, completed_forward_segments=len(complete),
                mean_forward_ms=sum(complete) / len(complete) if complete else None,
                median_forward_ms=median(complete) if complete else None,
                max_forward_ms=max(complete) if complete else None,
                forward_segments_under_500ms=sum(value < 500 for value in complete))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--left-forward-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--right-forward-sign", type=int, choices=(-1, 1), default=-1)
    args = parser.parse_args(argv)
    with args.log.open(encoding="utf-8", errors="replace") as stream:
        result = summarize(read_events(stream, left_forward_sign=args.left_forward_sign,
                                       right_forward_sign=args.right_forward_sign))
    result["raw_forward_signs"] = [args.left_forward_sign, args.right_forward_sign]
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
