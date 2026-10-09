#!/usr/bin/env bash
# Minimal, isolated single-person follow loop.
#
# This deliberately does not change or invoke run_request_0428_modular.sh.
# It uses the same board configuration and hardware adapters, but has its own
# small controller and log directory.

set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="$ROOT/car_control_modular/config/reid_runtime.ini"
CONFIG="$DEFAULT_CONFIG"
ENABLE_MOTOR=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./run_minimal_follow.sh [--config PATH] [--enable-motor]

Without --enable-motor, the runtime is dry-run: it runs camera, person
detection, depth and IR, but only logs wheel commands.

--enable-motor enables real LZ30EMA RS485 output. Test with the drive wheels
raised first. Do not run this together with run_request_0428_modular.sh.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            [[ $# -ge 2 ]] || { echo "--config requires a path" >&2; exit 2; }
            CONFIG="$2"
            shift 2
            ;;
        --config=*)
            CONFIG="${1#--config=}"
            shift
            ;;
        --enable-motor)
            ENABLE_MOTOR=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }

# Two programs must never write the same camera/IR/RS485 devices at once.
if pgrep -f "$ROOT/request_0513_modular.py" >/dev/null 2>&1 \
    || pgrep -f "$ROOT/request_0428_modular.py" >/dev/null 2>&1 \
    || pgrep -f "$ROOT/minimal_follow_runtime.py" >/dev/null 2>&1; then
    echo "another follow runtime is already running; stop it before starting minimal_follow" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_ROOT="$ROOT/run_minimal_follow_logs"
mkdir -p "$LOG_ROOT"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="$LOG_ROOT/$RUN_ID"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/minimal_follow.log"

# Keep numerical libraries from competing with RKNN on the board.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export FOLLOW_LOG_DIR="$LOG_DIR"

CMD=("$PYTHON_BIN" -u "$ROOT/minimal_follow_runtime.py" --config "$CONFIG")
if [[ "$ENABLE_MOTOR" -eq 1 ]]; then
    CMD+=(--enable-motor)
fi

echo "root=$ROOT"
echo "config=$CONFIG"
echo "motor_enabled=$ENABLE_MOTOR"
echo "log_dir=$LOG_DIR"
echo "log_file=$LOG_FILE"
echo "command=${CMD[*]}"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
RC=${PIPESTATUS[0]}
echo "minimal_follow rc=$RC log_dir=$LOG_DIR log_file=$LOG_FILE"
exit "$RC"
