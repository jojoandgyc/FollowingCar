#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH="${1:-/home/topeet/Desktop/followme/output/camera_record_20s_20260511_130819.mp4}"
OUT_DIR="${2:-.test_outputs/followme_tracker/reid_compare_$(basename "$VIDEO_PATH" .mp4)}"

mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  "$VIDEO_PATH"
  --yolo-model models/yolo11s.rknn
  --backend lite2
  --max-frames 10000
  --max-output-age 5
  --verify-predicted-reid
  --predicted-reid-verify-threshold 0.30
  --predicted-reid-duplicate-iou-threshold 0.30
  --predicted-reid-duplicate-overlap-threshold 0.45
  --debug-tracker-state
)

run_case() {
  local name="$1"
  shift

  echo "== ${name} =="
  CONFIDENCE_THRESHOLD=0.7 python3 tools/rknn_video_smoke.py \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --jsonl "${OUT_DIR}/${name}.jsonl" \
    --save-video "${OUT_DIR}/${name}.mp4" \
    > "${OUT_DIR}/${name}.log" 2>&1
}

run_case osnet_x05 \
  --reid-model models/osnet_x0_5_msmt17_combineall_b1.rknn \
  --reid-input-width 128 \
  --reid-input-height 256 \
  --reid-input-format RGB \
  --reid-input-dtype float32 \
  --reid-input-layout NCHW \
  --reid-normalize imagenet

run_case osnet_x025 \
  --reid-model models/osnet_x0_25_msmt17_b1.rknn \
  --reid-input-width 128 \
  --reid-input-height 256 \
  --reid-input-format RGB \
  --reid-input-dtype float32 \
  --reid-input-layout NCHW \
  --reid-normalize imagenet

run_case fastreid_r50 \
  --reid-model models/fastreid_market_bot_R50_b1.rknn \
  --reid-input-width 128 \
  --reid-input-height 256 \
  --reid-input-format RGB \
  --reid-input-dtype float32 \
  --reid-input-layout NCHW \
  --reid-normalize none

python3 - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path


def summarize(path: Path) -> dict:
    frames = 0
    track_frames = 0
    track_ids = set()
    reid_uids = set()
    no_person_with_track = []
    transitions = []
    last_state = None
    verify_pass = 0
    verify_fail = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            frame_index = int(row.get("frame_index", frames))
            frames += 1
            tracks = row.get("tracks") or []
            persons = row.get("persons") or []
            if tracks:
                track_frames += 1
            if tracks and not persons:
                no_person_with_track.append(frame_index)

            state = []
            for track in tracks:
                tid = int(track.get("track_id", 0))
                uid = int(track.get("reid_uid", 0))
                track_ids.add(tid)
                if uid > 0:
                    reid_uids.add(uid)
                state.append((tid, uid))
            state = tuple(state)
            if state and state != last_state:
                transitions.append((frame_index, state))
            if state:
                last_state = state

            for item in row.get("predicted_reid_verifications") or []:
                if item.get("passed"):
                    verify_pass += 1
                else:
                    verify_fail += 1

    return {
        "frames": frames,
        "track_frames": track_frames,
        "track_ids": sorted(track_ids),
        "reid_uids": sorted(reid_uids),
        "uid_count": len(reid_uids),
        "transitions": transitions[:12],
        "no_person_with_track": no_person_with_track,
        "verify_pass": verify_pass,
        "verify_fail": verify_fail,
    }


out_dir = Path(sys.argv[1])
print("\nsummary")
print("| case | frames | track_frames | track_ids | reid_uids | no-person+track | verify pass/fail | transitions |")
print("| --- | ---: | ---: | --- | --- | --- | --- | --- |")
for path in sorted(out_dir.glob("*.jsonl")):
    s = summarize(path)
    no_person = ",".join(map(str, s["no_person_with_track"][:20]))
    if len(s["no_person_with_track"]) > 20:
        no_person += ",..."
    trans = "; ".join(
        f"f{frame}:{list(state)}" for frame, state in s["transitions"][:8]
    )
    print(
        f"| {path.stem} | {s['frames']} | {s['track_frames']} | "
        f"{s['track_ids']} | {s['reid_uids']} | {no_person or '-'} | "
        f"{s['verify_pass']}/{s['verify_fail']} | {trans or '-'} |"
    )
PY

echo "wrote ${OUT_DIR}"
