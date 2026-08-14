#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SEGMENTS: Dict[str, Dict[str, Any]] = {
    "switch_entry": {
        "label": "entry and first ID switch",
        "notes": "Person enters at the right edge; public track usually changes from 2 to 3.",
        "target_bank_uids": 1,
        "watch": "bank should keep one stable reid_uid across the short track break",
    },
    "hand_occlusion": {
        "label": "hand occlusion",
        "notes": "Hand/body fills the frame and can create bad person boxes.",
        "target_bank_uids": 1,
        "watch": "fragmentation after severe occlusion",
    },
    "two_people": {
        "label": "two people / overlap",
        "notes": "Two people overlap or swap foreground/background positions.",
        "target_bank_uids": 1,
        "multi_person": True,
        "watch": "over-merge risk when different people are present",
    },
    "late_reentry": {
        "label": "late re-entry",
        "notes": "Long empty section before a person re-enters.",
        "target_bank_uids": 1,
        "watch": "stable re-entry and no long stale output after exit",
    },
}

DEFAULT_MODES = ("bank", "nobank")
DEFAULT_MAX_OUTPUT_AGE = 5


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize tracker segment JSONL outputs.")
    parser.add_argument("--input-dir", default=".test_outputs/tracker_segment_smokes")
    parser.add_argument("--suite-name", default="")
    parser.add_argument("--segments", default="", help="Comma-separated segment names. Defaults to files found in input-dir.")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated mode suffixes.")
    parser.add_argument("--max-output-age", type=int, default=DEFAULT_MAX_OUTPUT_AGE)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    if not modes:
        raise SystemExit("--modes must contain at least one mode")

    if args.segments.strip():
        segments = [item.strip() for item in args.segments.split(",") if item.strip()]
    else:
        segments = infer_segments(input_dir, modes)
    if not segments:
        raise SystemExit(f"no segment JSONL files found in {input_dir}")

    report = build_report(
        input_dir=input_dir,
        suite_name=args.suite_name or input_dir.name,
        segments=segments,
        modes=modes,
        max_output_age=max(0, int(args.max_output_age)),
    )

    md = render_markdown(report)
    print(md)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
    return 0


def infer_segments(input_dir: Path, modes: Sequence[str]) -> List[str]:
    found = set()
    for path in sorted(input_dir.glob("*.jsonl")):
        parsed = parse_segment_mode(path.stem, modes)
        if parsed is None:
            continue
        segment, _mode = parsed
        found.add(segment)
    ordered = [name for name in DEFAULT_SEGMENTS if name in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def parse_segment_mode(stem: str, modes: Sequence[str]) -> Optional[Tuple[str, str]]:
    for mode in sorted(modes, key=len, reverse=True):
        suffix = f"_{mode}"
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)], mode
    return None


def build_report(
    *,
    input_dir: Path,
    suite_name: str,
    segments: Sequence[str],
    modes: Sequence[str],
    max_output_age: int,
) -> Dict[str, Any]:
    entries: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    for segment in segments:
        segment_info = DEFAULT_SEGMENTS.get(segment, {})
        entries[segment] = {
            "segment": segment,
            "label": segment_info.get("label", segment),
            "notes": segment_info.get("notes", ""),
            "watch": segment_info.get("watch", ""),
            "modes": {},
            "comparison": {},
            "flags": [],
        }
        for mode in modes:
            path = input_dir / f"{segment}_{mode}.jsonl"
            if not path.exists():
                warnings.append(f"missing {path}")
                entries[segment]["modes"][mode] = {"missing": True, "path": str(path)}
                continue
            rows = load_jsonl(path)
            log_summary = load_log_summary(path.with_suffix(".log"))
            metrics = summarize_rows(rows, log_summary=log_summary)
            metrics["path"] = str(path)
            entries[segment]["modes"][mode] = metrics
        entries[segment]["comparison"] = compare_modes(entries[segment]["modes"], modes)
        entries[segment]["flags"] = segment_flags(
            segment=segment,
            segment_info=segment_info,
            mode_metrics=entries[segment]["modes"],
            max_output_age=max_output_age,
        )
    return {
        "suite_name": suite_name,
        "input_dir": str(input_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "modes": list(modes),
        "segments": entries,
        "warnings": warnings,
        "max_output_age": int(max_output_age),
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_log_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    summary: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("summary "):
            continue
        try:
            summary = json.loads(line[len("summary ") :])
        except json.JSONDecodeError:
            continue
    return summary


def summarize_rows(rows: Sequence[Dict[str, Any]], *, log_summary: Dict[str, Any]) -> Dict[str, Any]:
    if not rows:
        return {
            "missing": False,
            "processed": 0,
            "frame_start": None,
            "frame_end": None,
            "frame_step": 1,
            "person_frames": 0,
            "track_frames": 0,
            "track_ids": [],
            "reid_uids": [],
            "track_to_uids": {},
            "uid_to_tracks": {},
            "flags": ["empty"],
        }

    frame_indices = [int(row["frame_index"]) for row in rows]
    frame_step = infer_frame_step(frame_indices)
    person_frames = [int(row["frame_index"]) for row in rows if int(row.get("persons", 0)) > 0]
    track_frames = [int(row["frame_index"]) for row in rows if row.get("tracks")]
    no_person_track_frames = [
        int(row["frame_index"])
        for row in rows
        if int(row.get("persons", 0)) == 0 and bool(row.get("tracks"))
    ]
    person_no_track_frames = [
        int(row["frame_index"])
        for row in rows
        if int(row.get("persons", 0)) > 0 and not bool(row.get("tracks"))
    ]
    multi_track_frames = [
        int(row["frame_index"])
        for row in rows
        if len(row.get("tracks") or []) > 1
    ]

    track_frames_by_id: Dict[int, List[int]] = defaultdict(list)
    uid_frames_by_id: Dict[int, List[int]] = defaultdict(list)
    track_to_uids: Dict[int, set] = defaultdict(set)
    uid_to_tracks: Dict[int, set] = defaultdict(set)
    first_track_frame: Dict[int, int] = {}
    first_uid_frame: Dict[int, int] = {}
    score_by_track: Dict[int, List[float]] = defaultdict(list)
    area_by_track: Dict[int, List[float]] = defaultdict(list)

    output_changes = []
    last_state: Optional[Tuple[Tuple[int, int], ...]] = None
    for row in rows:
        frame_index = int(row["frame_index"])
        tracks = row.get("tracks") or []
        state = tuple((int(track["track_id"]), int(track.get("reid_uid", 0))) for track in tracks)
        if state != last_state:
            output_changes.append({"frame": frame_index, "tracks": [list(item) for item in state]})
            last_state = state
        for track in tracks:
            track_id = int(track["track_id"])
            uid = int(track.get("reid_uid", 0))
            bbox = track.get("bbox") or [0, 0, 0, 0]
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
            track_frames_by_id[track_id].append(frame_index)
            uid_frames_by_id[uid].append(frame_index)
            track_to_uids[track_id].add(uid)
            uid_to_tracks[uid].add(track_id)
            first_track_frame.setdefault(track_id, frame_index)
            first_uid_frame.setdefault(uid, frame_index)
            score_by_track[track_id].append(float(track.get("score", 0.0)))
            area_by_track[track_id].append(float(area))

    timing = summarize_timing(rows, log_summary)
    first_person = min(person_frames) if person_frames else None
    first_track = min(track_frames) if track_frames else None
    max_no_person_track_run = max_run_length(no_person_track_frames, frame_step)
    max_person_no_track_run = max_run_length(person_no_track_frames, frame_step)

    return {
        "missing": False,
        "processed": len(rows),
        "frame_start": frame_indices[0],
        "frame_end": frame_indices[-1],
        "frame_step": frame_step,
        "person_frames": len(person_frames),
        "track_frames": len(track_frames),
        "no_person_track_frames": len(no_person_track_frames),
        "no_person_track_runs": make_runs(no_person_track_frames, frame_step),
        "max_no_person_track_run": max_no_person_track_run,
        "person_no_track_frames": len(person_no_track_frames),
        "person_no_track_runs": make_runs(person_no_track_frames, frame_step),
        "max_person_no_track_run": max_person_no_track_run,
        "multi_track_frames": len(multi_track_frames),
        "max_concurrent_tracks": max((len(row.get("tracks") or []) for row in rows), default=0),
        "track_ids": sorted(track_frames_by_id),
        "reid_uids": sorted(uid_to_tracks),
        "track_to_uids": {str(key): sorted(value) for key, value in sorted(track_to_uids.items())},
        "uid_to_tracks": {str(key): sorted(value) for key, value in sorted(uid_to_tracks.items())},
        "track_runs": {str(key): make_runs(value, frame_step) for key, value in sorted(track_frames_by_id.items())},
        "uid_runs": {str(key): make_runs(value, frame_step) for key, value in sorted(uid_frames_by_id.items())},
        "first_track_frame": {str(key): value for key, value in sorted(first_track_frame.items())},
        "first_uid_frame": {str(key): value for key, value in sorted(first_uid_frame.items())},
        "first_person_frame": first_person,
        "first_public_track_frame": first_track,
        "first_track_delay_frames": None if first_person is None or first_track is None else int(first_track - first_person),
        "output_change_count": len(output_changes),
        "output_changes": output_changes,
        "track_avg_score": {str(key): round(avg(value), 4) for key, value in sorted(score_by_track.items())},
        "track_avg_area": {str(key): round(avg(value), 1) for key, value in sorted(area_by_track.items())},
        "timing": timing,
    }


def infer_frame_step(frames: Sequence[int]) -> int:
    diffs = [b - a for a, b in zip(frames, frames[1:]) if b > a]
    if not diffs:
        return 1
    counts = Counter(diffs)
    return max(1, int(counts.most_common(1)[0][0]))


def summarize_timing(rows: Sequence[Dict[str, Any]], log_summary: Dict[str, Any]) -> Dict[str, Any]:
    timing = {
        "processed_fps": log_summary.get("processed_fps"),
        "elapsed_sec": log_summary.get("elapsed_sec"),
        "avg_timing_ms": log_summary.get("avg_timing_ms") or {},
        "max_timing_ms": log_summary.get("max_timing_ms") or {},
    }
    if timing["avg_timing_ms"]:
        return timing
    keys = sorted({key for row in rows for key in (row.get("timing_ms") or {})})
    avg_timing = {}
    max_timing = {}
    for key in keys:
        values = [float((row.get("timing_ms") or {}).get(key, 0.0)) for row in rows]
        avg_timing[key] = round(avg(values), 3)
        max_timing[key] = round(max(values) if values else 0.0, 3)
    timing["avg_timing_ms"] = avg_timing
    timing["max_timing_ms"] = max_timing
    return timing


def compare_modes(mode_metrics: Dict[str, Dict[str, Any]], modes: Sequence[str]) -> Dict[str, Any]:
    if "bank" not in mode_metrics or "nobank" not in mode_metrics:
        return {}
    bank = mode_metrics.get("bank") or {}
    nobank = mode_metrics.get("nobank") or {}
    if bank.get("missing") or nobank.get("missing"):
        return {}
    bank_uid_count = len(bank.get("reid_uids") or [])
    nobank_uid_count = len(nobank.get("reid_uids") or [])
    return {
        "bank_uid_count": bank_uid_count,
        "nobank_uid_count": nobank_uid_count,
        "uid_count_delta": bank_uid_count - nobank_uid_count,
        "bank_track_count": len(bank.get("track_ids") or []),
        "nobank_track_count": len(nobank.get("track_ids") or []),
        "bank_merged_uid_groups": merged_uid_groups(bank),
        "nobank_merged_uid_groups": merged_uid_groups(nobank),
    }


def segment_flags(
    *,
    segment: str,
    segment_info: Dict[str, Any],
    mode_metrics: Dict[str, Dict[str, Any]],
    max_output_age: int,
) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    for mode, metrics in sorted(mode_metrics.items()):
        if metrics.get("missing"):
            flags.append({"level": "missing", "mode": mode, "message": f"{segment}_{mode}.jsonl is missing"})
            continue
        if int(metrics.get("max_no_person_track_run") or 0) > max_output_age:
            flags.append(
                {
                    "level": "watch",
                    "mode": mode,
                    "message": f"stale public track run exceeds max_output_age={max_output_age}",
                    "value": metrics.get("max_no_person_track_run"),
                }
            )
        if int(metrics.get("max_person_no_track_run") or 0) > 6:
            flags.append(
                {
                    "level": "watch",
                    "mode": mode,
                    "message": "person detections exist but no public track for a long run",
                    "value": metrics.get("max_person_no_track_run"),
                }
            )

    bank = mode_metrics.get("bank") or {}
    if not bank.get("missing"):
        target = segment_info.get("target_bank_uids")
        if target is not None:
            uid_count = len(bank.get("reid_uids") or [])
            if uid_count > int(target):
                flags.append(
                    {
                        "level": "risk",
                        "mode": "bank",
                        "message": "identity fragmentation in bank mode",
                        "uid_count": uid_count,
                        "target_uid_count": int(target),
                    }
                )
        if segment_info.get("multi_person") and merged_uid_groups(bank):
            flags.append(
                {
                    "level": "watch",
                    "mode": "bank",
                    "message": "multi-person segment has merged track IDs under one uid; inspect visually",
                    "groups": merged_uid_groups(bank),
                }
            )
    return flags


def merged_uid_groups(metrics: Dict[str, Any]) -> Dict[str, List[int]]:
    out = {}
    for uid, tracks in (metrics.get("uid_to_tracks") or {}).items():
        if len(tracks) > 1:
            out[str(uid)] = list(tracks)
    return out


def make_runs(frames: Iterable[int], step: int) -> List[List[int]]:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        return []
    runs: List[List[int]] = []
    start = prev = ordered[0]
    max_gap = max(1, int(step))
    for frame in ordered[1:]:
        if frame - prev <= max_gap:
            prev = frame
            continue
        runs.append([start, prev])
        start = prev = frame
    runs.append([start, prev])
    return runs


def max_run_length(frames: Iterable[int], step: int) -> int:
    runs = make_runs(frames, step)
    if not runs:
        return 0
    step = max(1, int(step))
    return max(((end - start) // step) + 1 for start, end in runs)


def avg(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        f"# Tracker Segment Report: {report['suite_name']}",
        "",
        f"- input_dir: `{report['input_dir']}`",
        f"- generated_at: `{report['generated_at']}`",
        f"- modes: `{', '.join(report['modes'])}`",
        "",
    ]
    if report.get("warnings"):
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in report["warnings"])
        lines.append("")

    lines.extend(
        [
            "## Summary",
            "",
            "| segment | mode | frames | persons | tracks | no-person+track | track ids | uid groups | changes | fps | yolo ms | reid ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for segment, entry in report["segments"].items():
        for mode in report["modes"]:
            metrics = entry["modes"].get(mode, {})
            if metrics.get("missing"):
                lines.append(f"| {segment} | {mode} | missing |  |  |  |  |  |  |  |  |  |")
                continue
            timing = metrics.get("timing") or {}
            avg_timing = timing.get("avg_timing_ms") or {}
            frame_span = f"{metrics.get('frame_start')}-{metrics.get('frame_end')}"
            uid_groups = format_uid_groups(metrics.get("uid_to_tracks") or {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        segment,
                        mode,
                        frame_span,
                        str(metrics.get("person_frames", 0)),
                        str(metrics.get("track_frames", 0)),
                        str(metrics.get("no_person_track_frames", 0)),
                        format_list(metrics.get("track_ids") or []),
                        uid_groups,
                        str(metrics.get("output_change_count", 0)),
                        format_float(timing.get("processed_fps")),
                        format_float(avg_timing.get("yolo_inference")),
                        format_float(avg_timing.get("reid_inference")),
                    ]
                )
                + " |"
            )
    lines.append("")

    lines.extend(["## Flags", ""])
    any_flags = False
    for segment, entry in report["segments"].items():
        flags = entry.get("flags") or []
        if not flags:
            continue
        any_flags = True
        lines.append(f"### {segment}")
        for flag in flags:
            details = {key: value for key, value in flag.items() if key not in {"level", "message"}}
            suffix = f" `{json.dumps(details, ensure_ascii=False, sort_keys=True)}`" if details else ""
            lines.append(f"- {flag.get('level', 'info')}: {flag.get('message', '')}{suffix}")
        lines.append("")
    if not any_flags:
        lines.append("- none")
        lines.append("")

    lines.extend(["## Details", ""])
    for segment, entry in report["segments"].items():
        lines.append(f"### {segment}: {entry.get('label', segment)}")
        if entry.get("notes"):
            lines.append(f"- notes: {entry['notes']}")
        if entry.get("watch"):
            lines.append(f"- watch: {entry['watch']}")
        comparison = entry.get("comparison") or {}
        if comparison:
            lines.append(f"- comparison: `{json.dumps(comparison, ensure_ascii=False, sort_keys=True)}`")
        for mode in report["modes"]:
            metrics = entry["modes"].get(mode, {})
            if metrics.get("missing"):
                continue
            changes = metrics.get("output_changes") or []
            lines.append(f"- {mode} track_to_uids: `{json.dumps(metrics.get('track_to_uids', {}), sort_keys=True)}`")
            lines.append(f"- {mode} uid_to_tracks: `{json.dumps(metrics.get('uid_to_tracks', {}), sort_keys=True)}`")
            lines.append(f"- {mode} track_runs: `{json.dumps(metrics.get('track_runs', {}), sort_keys=True)}`")
            lines.append(f"- {mode} changes: `{json.dumps(abbrev_changes(changes), sort_keys=True)}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_list(values: Sequence[Any]) -> str:
    if not values:
        return "-"
    return ",".join(str(value) for value in values)


def format_uid_groups(groups: Dict[str, Sequence[int]]) -> str:
    if not groups:
        return "-"
    parts = []
    for uid, tracks in sorted(groups.items(), key=lambda item: int(item[0])):
        parts.append(f"{uid}:{','.join(str(track) for track in tracks)}")
    return "; ".join(parts)


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def abbrev_changes(changes: Sequence[Dict[str, Any]], limit: int = 14) -> List[Dict[str, Any]]:
    if len(changes) <= limit:
        return list(changes)
    head = list(changes[: limit // 2])
    tail = list(changes[-(limit // 2) :])
    return head + [{"frame": "...", "tracks": "..."}] + tail


if __name__ == "__main__":
    raise SystemExit(main())
