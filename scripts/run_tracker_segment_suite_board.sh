#!/usr/bin/env bash
set -euo pipefail

BOARD_HOST="${BOARD_HOST:-rk}"
BOARD_ROOT="${BOARD_ROOT:-/home/topeet/Desktop/rk_car_runtime}"
REMOTE_OUT="${REMOTE_OUT:-smoke_frames/board_segment_smokes}"
LOCAL_OUT="${LOCAL_OUT:-.test_outputs/board_segment_smokes}"
REPORT_DIR="${REPORT_DIR:-.test_outputs/tracker_reports}"
RUN_MODES="${RUN_MODES:-bank nobank}"
FEATURE_UPDATE_INTERVAL="${FEATURE_UPDATE_INTERVAL:-3}"
MAX_OUTPUT_AGE="${MAX_OUTPUT_AGE:-5}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.25}"
VERIFY_PREDICTED_REID="${VERIFY_PREDICTED_REID:-0}"
PREDICTED_REID_VERIFY_THRESHOLD="${PREDICTED_REID_VERIFY_THRESHOLD:-0.30}"
COPY_BACK="${COPY_BACK:-1}"

mkdir -p "$LOCAL_OUT" "$REPORT_DIR"

ssh "$BOARD_HOST" "bash -s" <<REMOTE
set -euo pipefail
cd "$BOARD_ROOT"
OUT="$REMOTE_OUT"
RUN_MODES="$RUN_MODES"
FEATURE_UPDATE_INTERVAL="$FEATURE_UPDATE_INTERVAL"
MAX_OUTPUT_AGE="$MAX_OUTPUT_AGE"
CONFIDENCE_THRESHOLD="$CONFIDENCE_THRESHOLD"
VERIFY_PREDICTED_REID="$VERIFY_PREDICTED_REID"
PREDICTED_REID_VERIFY_THRESHOLD="$PREDICTED_REID_VERIFY_THRESHOLD"
mkdir -p "\$OUT"

CASES=(
  "switch_entry:60:45:1"
  "hand_occlusion:224:69:2"
  "two_people:378:83:1"
  "late_reentry:1489:56:2"
)

COMMON=(
  python3 tools/rknn_video_smoke.py test_video/camera_record_1000s_20260511_132259.mp4
  --backend lite2
  --yolo-model models/yolo11s.rknn
  --reid-model models/osnet_x0_5_msmt17_combineall_b1.rknn
  --reid-input-width 128
  --reid-input-height 256
  --reid-input-format RGB
  --reid-input-dtype float32
  --reid-input-layout NCHW
  --reid-normalize imagenet
  --feature-update-interval "\$FEATURE_UPDATE_INTERVAL"
  --max-output-age "\$MAX_OUTPUT_AGE"
  --debug-tracker-state
)

if [[ "\$VERIFY_PREDICTED_REID" != "0" ]]; then
  COMMON+=(--verify-predicted-reid --predicted-reid-verify-threshold "\$PREDICTED_REID_VERIFY_THRESHOLD")
fi

run_case() {
  local name="\$1"
  local start="\$2"
  local max_frames="\$3"
  local stride="\$4"
  local mode="\$5"
  local jsonl="\$OUT/\${name}_\${mode}.jsonl"
  local log="\$OUT/\${name}_\${mode}.log"
  local extra=()
  if [[ "\$mode" == "nobank" ]]; then
    extra+=(--disable-identity-bank)
  fi
  echo "== board \$name \$mode start=\$start max=\$max_frames stride=\$stride"
  CONFIDENCE_THRESHOLD="\$CONFIDENCE_THRESHOLD" "\${COMMON[@]}" \
    --start-frame "\$start" \
    --max-frames "\$max_frames" \
    --frame-stride "\$stride" \
    --jsonl "\$jsonl" \
    "\${extra[@]}" > "\$log" 2>&1
  grep '^summary ' "\$log" | tail -n 1 || true
}

for case_spec in "\${CASES[@]}"; do
  IFS=: read -r name start max_frames stride <<< "\$case_spec"
  for mode in \$RUN_MODES; do
    run_case "\$name" "\$start" "\$max_frames" "\$stride" "\$mode"
  done
done
REMOTE

if [[ "$COPY_BACK" != "0" ]]; then
  scp "$BOARD_HOST:$BOARD_ROOT/$REMOTE_OUT/*" "$LOCAL_OUT/"
fi

python3 tools/tracker_segment_eval.py \
  --input-dir "$LOCAL_OUT" \
  --suite-name "board-lite2-conf${CONFIDENCE_THRESHOLD}" \
  --max-output-age "$MAX_OUTPUT_AGE" \
  --output-json "$REPORT_DIR/board_lite2_segment_report.json" \
  --output-md "$REPORT_DIR/board_lite2_segment_report.md"
