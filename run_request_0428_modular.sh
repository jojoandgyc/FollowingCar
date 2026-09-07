#!/bin/sh
set -u

ROOT="$(CDPATH= cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1

SCRIPT_NAME="$(basename "$0")"
SCRIPT_BASE="${SCRIPT_NAME%.*}"
REQUEST_SCRIPT="$ROOT/request_0513_modular.py"
PREFLIGHT_SCRIPT="$ROOT/car_control_modular/peripheral_preflight.py"
LOG_DIR="$ROOT/${SCRIPT_BASE}_logs"
LOG_FILE="$LOG_DIR/request_0513_modular.log"
DEFAULT_CONFIG="car_control_modular/config/reid_runtime.ini"
CONFIG="$DEFAULT_CONFIG"

usage() {
  cat <<EOF
Usage:
  ./run_request_0428_modular.sh [config.ini] [optional_model_path]
  ./run_request_0428_modular.sh --config config.ini [optional_model_path]

Environment:
  PY=/path/to/python3
  MMWAVE_AT2410_PORT=/dev/serial/by-id/usb-SIPEED_UARTx4_HS_FactoryAIOT_Prog_Serial-if00
  PERIPHERAL_PREFLIGHT=1  # verify enabled peripherals before control starts
  # AT2410 USB reset/verification runs inside the long-lived Python runtime.
  REQUEST_LOOPBACK_UP=1  # optional: run "ip link set lo up" before startup

Logs:
  ./${SCRIPT_BASE}_logs/ is cleared at startup and reused for each run.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -gt 0 ]; then
  case "$1" in
    --config)
      shift
      if [ "$#" -eq 0 ]; then
        echo "missing value for --config" >&2
        exit 2
      fi
      CONFIG="$1"
      shift
      ;;
    --config=*)
      CONFIG="${1#--config=}"
      shift
      ;;
    --*)
      ;;
    *)
      CONFIG="$1"
      shift
      ;;
  esac
fi

if [ ! -f "$CONFIG" ]; then
  echo "config not found: $CONFIG" >&2
  exit 2
fi

if [ ! -f "$REQUEST_SCRIPT" ]; then
  echo "request script not found: $REQUEST_SCRIPT" >&2
  exit 2
fi

PY="${PY:-python3}"
if [ ! -x "$PY" ]; then
  PY="${PYTHON:-python3}"
fi

# Keep one explicit AT2410 override visible in the launcher environment. The
# config loader treats this variable as higher priority than [mmwave].port.
export MMWAVE_AT2410_PORT="${MMWAVE_AT2410_PORT:-/dev/serial/by-id/usb-SIPEED_UARTx4_HS_FactoryAIOT_Prog_Serial-if00}"

# RKNNLite is installed for the board user rather than system-wide. Add that
# site-packages directory explicitly so the same launcher also works from a
# root maintenance shell.
PY_VERSION="$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
BOARD_USER_SITE="/home/topeet/.local/lib/python${PY_VERSION}/site-packages"
if [ -d "$BOARD_USER_SITE" ]; then
  case ":${PYTHONPATH:-}:" in
    *":$BOARD_USER_SITE:"*) ;;
    *) PYTHONPATH="$BOARD_USER_SITE${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
  export PYTHONPATH
fi

if [ -z "$SCRIPT_BASE" ] || [ "$LOG_DIR" = "$ROOT" ] || [ "$LOG_DIR" = "/" ]; then
  echo "unsafe log dir: $LOG_DIR" >&2
  exit 2
fi

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"
export FOLLOW_LOG_DIR="$LOG_DIR"

if [ "${PERIPHERAL_PREFLIGHT:-1}" != "0" ]; then
  if [ ! -f "$PREFLIGHT_SCRIPT" ]; then
    echo "外设自检程序不存在: $PREFLIGHT_SCRIPT" >&2
    exit 5
  fi
  if pgrep -f "$REQUEST_SCRIPT" >/dev/null 2>&1; then
    echo "已有跟随车进程正在运行，拒绝复位USB或重复启动" >&2
    exit 5
  fi

  PREFLIGHT_LOG="$LOG_DIR/peripheral_preflight.log"
  PREFLIGHT_RC_FILE="$LOG_DIR/.peripheral_preflight.rc"
  echo "外设启动基础自检" | tee -a "$PREFLIGHT_LOG"
  rm -f "$PREFLIGHT_RC_FILE"
  (
    "$PY" -u "$PREFLIGHT_SCRIPT" --config "$CONFIG"
    echo "$?" > "$PREFLIGHT_RC_FILE"
  ) 2>&1 | tee -a "$PREFLIGHT_LOG"
  if [ -f "$PREFLIGHT_RC_FILE" ]; then
    PREFLIGHT_RC="$(cat "$PREFLIGHT_RC_FILE")"
    rm -f "$PREFLIGHT_RC_FILE"
  else
    PREFLIGHT_RC=1
  fi
  if [ "$PREFLIGHT_RC" -ne 0 ]; then
    echo "外设自检未通过，禁止启动跟随控制 rc=$PREFLIGHT_RC" >&2
    exit "$PREFLIGHT_RC"
  fi
fi

OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
OLD_LD_PRELOAD="${LD_PRELOAD:-}"
LD_LIBRARY_PATH=""
unset LD_PRELOAD

append_path() {
  if [ -d "$1" ]; then
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
      LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$1"
    else
      LD_LIBRARY_PATH="$1"
    fi
  fi
}

append_path "$ROOT"
append_path "/mnt/system/lib"
append_path "/mnt/system/usr/lib"
append_path "/mnt/system/usr/lib/3rd"
append_path "/mnt/data/lib"

if [ "${REQUEST_KEEP_PARENT_LD_LIBRARY_PATH:-0}" != "0" ] && [ -n "$OLD_LD_LIBRARY_PATH" ]; then
  if [ -n "$LD_LIBRARY_PATH" ]; then
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$OLD_LD_LIBRARY_PATH"
  else
    LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
  fi
fi
export LD_LIBRARY_PATH

if [ "${REQUEST_KEEP_PARENT_LD_PRELOAD:-0}" != "0" ] && [ -n "$OLD_LD_PRELOAD" ]; then
  LD_PRELOAD="$OLD_LD_PRELOAD"
  export LD_PRELOAD
fi

RC_FILE="$LOG_FILE.rc"
PIPE_FILE="$LOG_DIR/.request_output.pipe"
rm -f "$RC_FILE" "$PIPE_FILE"
mkfifo "$PIPE_FILE"

tee "$LOG_FILE" < "$PIPE_FILE" &
TEE_PID="$!"

(
  PY_PID=""
  forward_signal() {
    if [ -n "$PY_PID" ] && kill -0 "$PY_PID" 2>/dev/null; then
      kill -TERM "$PY_PID" 2>/dev/null || true
    fi
  }
  trap forward_signal INT TERM HUP

  echo "root=$ROOT"
  echo "config=$CONFIG"
  echo "python=$PY"
  echo "python_path=${PYTHONPATH:-}"
  echo "request_script=$REQUEST_SCRIPT"
  echo "log_dir=$LOG_DIR"
  echo "log_file=$LOG_FILE"
  echo "extra_args=$*"
  echo "parent_LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "parent_LD_PRELOAD=${LD_PRELOAD:-}"
  if [ "${REQUEST_LOOPBACK_UP:-0}" != "0" ]; then
    if command -v ip >/dev/null 2>&1; then
      if ip link set lo up; then
        echo "ip link set lo up rc=0"
      else
        rc_ip="$?"
        echo "ip link set lo up failed rc=$rc_ip"
      fi
    else
      echo "ip command not found; skip loopback setup"
    fi
  fi
  "$PY" -u "$REQUEST_SCRIPT" --config "$CONFIG" "$@" &
  PY_PID="$!"
  PY_RC=0
  while kill -0 "$PY_PID" 2>/dev/null; do
    wait "$PY_PID"
    PY_RC="$?"
  done
  wait "$PY_PID" 2>/dev/null
  LAST_WAIT_RC="$?"
  if [ "$LAST_WAIT_RC" -ne 127 ]; then
    PY_RC="$LAST_WAIT_RC"
  fi
  echo "$PY_RC" > "$RC_FILE"
) > "$PIPE_FILE" 2>&1 &
RUNNER_PID="$!"

SHUTDOWN_GRACE_SEC="${REQUEST_SHUTDOWN_GRACE_SEC:-8}"
case "$SHUTDOWN_GRACE_SEC" in
  ''|*[!0-9]*) SHUTDOWN_GRACE_SEC=8 ;;
esac
RUNNER_STOP_REQUESTED=0
SHUTDOWN_DEADLINE=0
FORCE_TERMINATED=0

forward_runner_signal() {
  if kill -0 "$RUNNER_PID" 2>/dev/null; then
    kill -TERM "$RUNNER_PID" 2>/dev/null || true
  fi
  # 只在第一次收到退出信号时开始计时，避免重复 SIGINT/SIGTERM 无限延长
  # 等待时间。到期后由父脚本接管安全停车和子进程清理。
  if [ "$RUNNER_STOP_REQUESTED" -eq 0 ]; then
    RUNNER_STOP_REQUESTED=1
    SHUTDOWN_DEADLINE=$(( $(date +%s) + SHUTDOWN_GRACE_SEC ))
    echo "收到退出信号，等待子进程安全清理 ${SHUTDOWN_GRACE_SEC}s" >&2
  fi
}
trap forward_runner_signal INT TERM HUP

collect_descendants() {
  for child_pid in $(pgrep -P "$1" 2>/dev/null || true); do
    echo "$child_pid"
    collect_descendants "$child_pid"
  done
}

force_motor_safe_stop() {
  cleanup_log="$LOG_DIR/force_motor_safe_stop.log"
  echo "执行独立电机安全收尾，日志=$cleanup_log" >&2
  "$PY" -u "$ROOT/tools/force_motor_safe_stop.py" --config "$CONFIG" >"$cleanup_log" 2>&1 &
  cleanup_pid="$!"
  cleanup_deadline=$(( $(date +%s) + 3 ))
  while kill -0 "$cleanup_pid" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$cleanup_deadline" ]; then
      echo "独立电机安全收尾超时，终止清理进程 pid=$cleanup_pid" >&2
      kill -KILL "$cleanup_pid" 2>/dev/null || true
      break
    fi
    sleep 0.1
  done
}

force_runner_shutdown() {
  echo "退出清理超过 ${SHUTDOWN_GRACE_SEC}s，升级终止跟随进程" >&2
  # 先终止 Python 及其后代，释放 /dev/ttyS0，随后再用独立脚本把
  # 双轮速度和驻车电流写回安全值。
  runner_children="$(collect_descendants "$RUNNER_PID")"
  for child_pid in $runner_children; do
    kill -TERM "$child_pid" 2>/dev/null || true
  done
  kill -TERM "$RUNNER_PID" 2>/dev/null || true
  sleep 1
  for child_pid in $runner_children; do
    kill -KILL "$child_pid" 2>/dev/null || true
  done
  kill -KILL "$RUNNER_PID" 2>/dev/null || true
  force_motor_safe_stop
  kill -KILL "$TEE_PID" 2>/dev/null || true
  FORCE_TERMINATED=1
}

while kill -0 "$RUNNER_PID" 2>/dev/null; do
  if [ "$RUNNER_STOP_REQUESTED" -ne 0 ] && [ "$(date +%s)" -ge "$SHUTDOWN_DEADLINE" ]; then
    force_runner_shutdown
    break
  fi
  sleep 0.1
done
wait "$RUNNER_PID" 2>/dev/null || true
if [ "$FORCE_TERMINATED" -eq 0 ]; then
  wait "$TEE_PID" 2>/dev/null || true
fi
rm -f "$PIPE_FILE"
trap - INT TERM HUP

if [ "$FORCE_TERMINATED" -ne 0 ]; then
  RC=124
elif [ -f "$RC_FILE" ]; then
  RC="$(cat "$RC_FILE")"
  rm -f "$RC_FILE"
else
  RC=255
fi

echo "request_0513_modular rc=$RC log_dir=$LOG_DIR log_file=$LOG_FILE"
exit "$RC"
