"""Read-only, offline timing audit; no motor/control replay or authority claims.

Usage: python3 analysis/depth_result_gap_statistics.py RUN_DIR [CAP_START CAP_END]
Physical coverage assumes results were available at capture (counterfactual).
Delivered coverage begins only at logged fresh-result completion. Neither
coverage implies identity, obstacle, feedback, or braking checks passed.
"""
import csv
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path


def fields(line):
    return dict(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", line))


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summary(values):
    return {"count": len(values), "median": percentile(values, .5),
            "p95": percentile(values, .95), "max": max(values, default=None),
            "total": sum(values)}


def gaps(samples, begin, end, ttl, physical=False):
    intervals = sorted((max(begin, s["sample"] if physical else s["available"]),
                        min(end, s["sample"] + ttl)) for s in samples)
    cursor = begin
    result = []
    for start, stop in intervals:
        if stop <= start or stop <= begin or start >= end:
            continue
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < end:
        result.append((cursor, end))
    return result


def audit(run, first, last):
    rows = list(csv.DictReader((run / "camera_raw.frames.csv").open()))
    captures = {int(r["capture_frame_id"]): float(r["capture_monotonic_sec"]) for r in rows}
    offsets = [float(r["capture_unix_sec"]) - float(r["capture_monotonic_sec"]) for r in rows]
    offset = statistics.median(offsets)
    begin, end = captures[first], captures[last]
    sources = {"fresh_depth_timeline": {}, "fresh_depth_control_entry": {},
               "fresh_depth_result_publication_log": {}}
    timings = {}
    counts = {}
    examples = []
    for number, line in enumerate((run / "request_0513_modular.log").open(), 1):
        try:
            wall = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp()
        except ValueError:
            continue
        logged = wall - offset
        f = fields(line)
        sample = None
        if "Depth timeline:" in line and f.get("sample_decision") == "fresh" and f.get("used") != "None":
            try:
                sample = float(f["sample_ts"])
                available = sample + float(f["sample_age_ms"]) / 1000
            except (ValueError, KeyError):
                continue
            key = "fresh_depth_timeline"
        elif "depth_linear_limit " in line and f.get("fresh") == "True":
            try:
                sample = float(f["sample_ts"])
                available = float(f["authority_sample_ts"]) + (float(f["physical_depth_ttl_ms"]) - float(f["remaining_ms"])) / 1000
            except (ValueError, KeyError):
                continue
            # remaining_ms uses the commit function's entry clock, not the
            # final publication clock: this is a lower-bound completion proxy.
            key = "fresh_depth_control_entry"
        elif "depth_linear_authority " in line and f.get("fresh") == "True":
            try:
                sample = float(f["sample_ts"])
            except (ValueError, KeyError):
                continue
            # Log completion is millisecond wall time, mapped via camera CSV.
            # A fresh zero decision is still a range result, not a motion grant.
            available = logged
            key = "fresh_depth_result_publication_log"
        if sample is not None:
            event = {"line": number, "sample": sample, "available": available,
                     "age_ms": (available - sample) * 1000,
                     "capture": f.get("capture_frame_id"), "wall": line[:23]}
            existing = sources[key].get(sample)
            if existing is None or available < existing["available"]:
                sources[key][sample] = event
        if not begin <= logged <= end:
            continue
        if "pipeline_timing:" in line:
            for name in ("vision_total_ms", "control_ms", "capture_to_pipeline_ms", "capture_result_age_ms", "capture_decision_age_ms"):
                if name in f:
                    timings.setdefault(name, []).append(float(f[name]))
        if "control_lock_stage" in line:
            for name in ("depth_compute_outside_lock_ms", "lock_hold_ms", "lock_wait_ms"):
                if f.get(name) not in (None, "None"):
                    timings.setdefault(f["source"] + "_" + name, []).append(float(f[name]))
        if "depth30_prepared_discard" in line:
            reason = "depth30_discard_same_context_" + f.get("same_context", "unknown")
            counts[reason] = counts.get(reason, 0) + 1
            timings.setdefault("depth30_discard_compute_ms", []).append(float(f["compute_ms"]))
        if "stale_result_discarded=True" in line:
            counts["visual_stale_result_discarded"] = counts.get("visual_stale_result_discarded", 0) + 1
        if "visible_wheel_dispatch" in line and "'visibility_expired'" in line:
            counts["dispatch_visibility_expired"] = counts.get("dispatch_visibility_expired", 0) + 1
            examples.append({"line": number, "reason": "visibility_expired", "capture": f.get("evidence_capture_frame_id")})
    report = {"run": str(run), "capture_window": [first, last], "duration_ms": (end-begin)*1000,
              "offset_median_sec": offset, "counts": counts,
              "timings_ms": {name: summary(values) for name, values in timings.items()},
              "sources": {}, "visual_expiry_examples": examples}
    for name, data in sources.items():
        samples = sorted(data.values(), key=lambda s: s["sample"])
        inside = [s for s in samples if begin <= s["available"] <= end]
        intervals = [(b["sample"]-a["sample"])*1000 for a,b in zip(samples,samples[1:]) if begin <= b["available"] <= end]
        result = {"samples_delivered_in_window": len(inside), "age_ms": summary([s["age_ms"] for s in inside]),
                  "sample_intervals_ms": summary(intervals), "gaps": {}}
        for ttl in (.25, .5):
            for physical in (True, False):
                gaplist = gaps(samples, begin, end, ttl, physical)
                key = ("capture_only_" if physical else "delivered_") + str(int(ttl*1000))
                result["gaps"][key] = {"summary_ms": summary([(b-a)*1000 for a,b in gaplist]),
                    "episodes": [{"start": a, "end": b, "duration_ms": (b-a)*1000,
                                  "next_result_line": min((s["line"] for s in samples if abs(s["available"]-b)<.00001),default=None)} for a,b in gaplist]}
        report["sources"][name] = result
    return report


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv)>2 else 205,
                           int(sys.argv[3]) if len(sys.argv)>3 else 454), ensure_ascii=False, indent=2))
