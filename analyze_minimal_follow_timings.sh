#!/usr/bin/env bash
# Summarize per-frame latency records emitted by minimal_follow_runtime.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: ./analyze_minimal_follow_timings.sh [PATH/TO/minimal_follow.log]"
    echo "Without an argument, analyzes the newest minimal-follow run log."
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    LOG_FILE="$1"
else
    LOG_FILE="$(find "$ROOT/run_minimal_follow_logs" -type f -name minimal_follow.log -print 2>/dev/null | sort | tail -n 1 || true)"
fi

if [[ -z "$LOG_FILE" || ! -f "$LOG_FILE" ]]; then
    echo "minimal-follow log not found; pass its full path explicitly." >&2
    exit 2
fi

python3 - "$LOG_FILE" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

path = Path(sys.argv[1])
marker = "minimal_timing "
records = []
for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    offset = line.find(marker)
    if offset < 0:
        continue
    try:
        records.append(json.loads(line[offset + len(marker):]))
    except json.JSONDecodeError:
        continue

if not records:
    raise SystemExit("No minimal_timing records found. Run the new runtime version first.")

metrics = (
    "cycle_ms", "capture_ms", "detect_ms", "detect_preprocess_ms",
    "detect_inference_ms", "detect_decode_ms", "detect_nms_ms", "select_ms",
    "ir_ms", "depth_ms", "decision_ms", "search_policy_ms", "dispatch_ms", "total_ms",
)

def percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

def numeric_values(metric):
    return [float(row[metric]) for row in records
            if (metric != "depth_ms" or bool(row.get("depth_attempted")))
            and isinstance(row.get(metric), (int, float)) and math.isfinite(float(row[metric]))]

print(f"log={path}")
print(f"processed_frames={len(records)}")
timestamps = [float(row["monotonic_s"]) for row in records if isinstance(row.get("monotonic_s"), (int, float))]
duration = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0
person_frames = sum(bool(row.get("person_selected")) for row in records)
depth_attempts = sum(bool(row.get("depth_attempted")) for row in records)
if duration > 0:
    loop_hz = (len(records) - 1) / duration
    print(f"observed_duration_s={duration:.3f} loop_hz={loop_hz:.2f} "
          f"person_hz={loop_hz * person_frames / len(records):.2f} "
          f"depth_hz={loop_hz * depth_attempts / len(records):.2f}")
else:
    print("observed_duration_s=0.000 (need at least two timing records for Hz)")
print(f"person_frames={person_frames} depth_attempts={depth_attempts}")
print()
print(f"{'metric':24} {'n':>7} {'avg_ms':>10} {'p50_ms':>10} {'p90_ms':>10} {'p99_ms':>10} {'min_ms':>10} {'max_ms':>10}")
for metric in metrics:
    values = numeric_values(metric)
    if not values:
        continue
    print(f"{metric:24} {len(values):7d} {statistics.fmean(values):10.3f} "
          f"{percentile(values, .50):10.3f} {percentile(values, .90):10.3f} "
          f"{percentile(values, .99):10.3f} {min(values):10.3f} {max(values):10.3f}")
PY
