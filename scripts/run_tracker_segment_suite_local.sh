#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rk-car-rknn-toolkit2:2.3.2}"
OUT_DIR="${OUT_DIR:-.test_outputs/tracker_segment_smokes}"
REPORT_DIR="${REPORT_DIR:-.test_outputs/tracker_reports}"
VIDEO="${VIDEO:-.test_car/test_video/camera_record_1000s_20260511_132259.mp4}"
YOLO_MODEL="${YOLO_MODEL:-models/source/yolo11s.onnx}"
REID_MODEL="${REID_MODEL:-models/source/osnet_x0_5_msmt17_combineall_b1.onnx}"
RUN_MODES="${RUN_MODES:-bank nobank}"
FEATURE_UPDATE_INTERVAL="${FEATURE_UPDATE_INTERVAL:-3}"
MAX_OUTPUT_AGE="${MAX_OUTPUT_AGE:-5}"
BBOX_EXPAND_SCALE="${BBOX_EXPAND_SCALE:-}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.25}"
VERIFY_PREDICTED_REID="${VERIFY_PREDICTED_REID:-0}"
PREDICTED_REID_VERIFY_THRESHOLD="${PREDICTED_REID_VERIFY_THRESHOLD:-0.30}"
PLATFORM="${PLATFORM:-linux/amd64}"

mkdir -p "$OUT_DIR" "$REPORT_DIR"

CASES=(
  "switch_entry:60:45:1"
  "hand_occlusion:224:69:2"
  "two_people:378:83:1"
  "late_reentry:1489:56:2"
)

COMMON=(
  python3 tools/rknn_video_smoke.py "$VIDEO"
  --backend onnxruntime
  --yolo-model "$YOLO_MODEL"
  --reid-model "$REID_MODEL"
  --reid-input-width 128
  --reid-input-height 256
  --reid-input-format RGB
  --reid-input-dtype float32
  --reid-input-layout NCHW
  --reid-normalize imagenet
  --feature-update-interval "$FEATURE_UPDATE_INTERVAL"
  --max-output-age "$MAX_OUTPUT_AGE"
  --debug-tracker-state
)

if [[ -n "$BBOX_EXPAND_SCALE" ]]; then
  COMMON+=(--bbox-expand-scale "$BBOX_EXPAND_SCALE")
fi

if [[ "$VERIFY_PREDICTED_REID" != "0" ]]; then
  COMMON+=(--verify-predicted-reid --predicted-reid-verify-threshold "$PREDICTED_REID_VERIFY_THRESHOLD")
fi

run_case() {
  local name="$1"
  local start="$2"
  local max_frames="$3"
  local stride="$4"
  local mode="$5"
  local jsonl="$OUT_DIR/${name}_${mode}.jsonl"
  local log="$OUT_DIR/${name}_${mode}.log"
  local extra=()
  if [[ "$mode" == "nobank" ]]; then
    extra+=(--disable-identity-bank)
  fi

  echo "== local $name $mode start=$start max=$max_frames stride=$stride"
  docker run --rm --platform "$PLATFORM" \
    -e CONFIDENCE_THRESHOLD="$CONFIDENCE_THRESHOLD" \
    -e OMP_NUM_THREADS=4 \
    -e OPENBLAS_NUM_THREADS=4 \
    -e MKL_NUM_THREADS=4 \
    -v "$PWD:/workspace" \
    -w /workspace \
    "$IMAGE" \
    "${COMMON[@]}" \
    --start-frame "$start" \
    --max-frames "$max_frames" \
    --frame-stride "$stride" \
    --jsonl "$jsonl" \
    ${extra[@]+"${extra[@]}"} > "$log" 2>&1
  grep '^summary ' "$log" | tail -n 1 || true
}

for case_spec in "${CASES[@]}"; do
  IFS=: read -r name start max_frames stride <<< "$case_spec"
  for mode in $RUN_MODES; do
    run_case "$name" "$start" "$max_frames" "$stride" "$mode"
  done
done

python3 tools/tracker_segment_eval.py \
  --input-dir "$OUT_DIR" \
  --suite-name "local-onnx-conf${CONFIDENCE_THRESHOLD}" \
  --max-output-age "$MAX_OUTPUT_AGE" \
  --output-json "$REPORT_DIR/local_onnx_segment_report.json" \
  --output-md "$REPORT_DIR/local_onnx_segment_report.md"
