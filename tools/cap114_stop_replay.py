"""Extract a STOP-only CAP114 replay. No serial access in this module.

The source has command-group completion logs, not a wire-level trace. Offsets
are therefore approximate; never translate feedback RPM into speed commands.
"""
import csv
from datetime import datetime
import hashlib
from pathlib import Path
import re
from zoneinfo import ZoneInfo

DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "run_request_0428_modular_logs" / "run_20260923_185427_35591_90c3de65"


def extract_zero_entry(plan):
    """Verify the four pre-NORMAL zero command groups; never replay prior speed."""
    data = (Path(plan['source']) / 'request_0513_modular.log').read_bytes()
    if hashlib.sha256(data).hexdigest() != plan['log_sha256']:
        raise ValueError('source changed since STOP plan extraction')
    entries = []
    for number, line in enumerate(data.decode().splitlines(), 1):
        if number >= plan['entry_normal']['line']:
            break
        if 'LZ30EMA 电机命令:' not in line:
            continue
        timestamp = datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S,%f').replace(
            tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        entries.append(dict(timestamp=timestamp, line=number, source_text=line))
    entries = entries[-4:]
    if len(entries) != 4 or any(not re.search(r'左轮=0转/分 右轮=0转/分(?:\s|$)', e['source_text'])
                                for e in entries):
        raise ValueError('expected four zero-only entry commands')
    origin = entries[0]['timestamp']
    offsets = [round(e['timestamp']-origin, 6) for e in entries]
    normal_offset = round(plan['entry_normal']['timestamp']-origin, 6)
    if offsets != [0., .054, .087, .219] or normal_offset != .325:
        raise ValueError('unexpected zero/NORMAL timing; review source')
    return dict(events=[dict(e, offset=offset) for e,offset in zip(entries,offsets)],
                normal_offset=normal_offset,
                normal_prepare_budget_sec=.015,
                timing_basis='first zero completion; NORMAL preparation starts 15ms before its target completion')


def extract_plan(directory=DEFAULT_SOURCE):
    directory = Path(directory)
    log = directory / "request_0513_modular.log"
    video_csv = directory / "camera_raw.frames.csv"
    data = log.read_bytes()
    with video_csv.open() as stream:
        frames = list(csv.DictReader(stream))
    cap = next(r for r in frames if r["capture_frame_id"] == "114")
    epoch = float(cap["capture_unix_sec"])
    events, normal = [], None
    for number, line in enumerate(data.decode().splitlines(), 1):
        if not any(marker in line for marker in
                   ("LZ30EMA 电机命令:", "LZ30EMA 停车命令:", "LZ30EMA 驻车电流已确认:")):
            continue
        timestamp = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").replace(
            tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
        if timestamp < epoch:
            if "LZ30EMA 停车命令:" in line:
                normal = dict(timestamp=timestamp, line=number, text=line)
            continue
        if "LZ30EMA 电机命令:" in line:
            raise ValueError(f"CAP114之后出现速度写入，拒绝STOP-only回放: line {number}")
        event = dict(offset=round(timestamp-epoch, 6), line=number, source_text=line)
        if "LZ30EMA 驻车电流已确认:" in line:
            if "右轮=0.0A 左轮=0.0A" not in line:
                raise ValueError("回放仅允许清0A")
            event.update(kind="current", value=0)
        else:
            match = re.search(r"stop_value=(\d+)", line)
            if not match or int(match[1]) not in (1, 2) or "pre_zero=False post_zero=False" not in line:
                raise ValueError("unsupported STOP semantics in replay")
            event.update(kind="stop", value=int(match[1]))
        events.append(event)
    if not normal or not all(text in normal["text"] for text in
                             ("模式=normal", "parking_current_a=5.0", "pre_zero=True post_zero=False")):
        raise ValueError("missing verified 5A NORMAL entry state")
    gap = epoch-normal["timestamp"]
    if not 0 <= gap <= .05 or not events or events[-1]["offset"] > 8:
        raise ValueError("unexpected CAP114 entry time or replay duration")
    # This is deliberately a bounded case-specific diagnostic, not arbitrary log execution.
    if [(e["kind"], e["value"]) for e in events] != [
            ("current", 0), ("stop", 2), *[("stop", 1)]*5, ("current", 0)]:
        raise ValueError("unexpected command sequence; manually review source first")
    return dict(source=str(directory), log_sha256=hashlib.sha256(data).hexdigest(),
                frames_sha256=hashlib.sha256(video_csv.read_bytes()).hexdigest(),
                cap=114, cap_epoch=epoch, normal_to_cap_sec=gap, entry_normal=normal,
                events=events, timing_basis="group completion log approximated as dispatch deadline",
                limitation="Does not reconstruct pre-CAP114 motion or controller internal state")
