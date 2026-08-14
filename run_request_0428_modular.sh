#!/bin/sh
set -u

ROOT="$(CDPATH= cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1

SCRIPT_NAME="$(basename "$0")"
SCRIPT_BASE="${SCRIPT_NAME%.*}"
REQUEST_SCRIPT="$ROOT/request_0513_modular.py"
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
  MMWAVE_AT2410_USB_RESET=1  # reset SIPEED UARTx4 once before startup; set 0 to skip
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

# The SIPEED UARTx4 can keep stale cdc_acm state after reconnect. Reset it once
# before any project process opens a channel. This briefly resets all 4 ports.
if [ "${MMWAVE_AT2410_USB_RESET:-1}" != "0" ]; then
  if command -v usbreset >/dev/null 2>&1; then
    if sudo -n usbreset 359f:3101; then
      echo "SIPEED UARTx4 reset before startup: ok"
      sleep 1
    else
      echo "warning: SIPEED UARTx4 reset failed; continuing without reset" >&2
    fi
  else
    echo "warning: usbreset not found; continuing without SIPEED reset" >&2
  fi
fi

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

forward_runner_signal() {
  if kill -0 "$RUNNER_PID" 2>/dev/null; then
    kill -TERM "$RUNNER_PID" 2>/dev/null || true
  fi
}
trap forward_runner_signal INT TERM HUP

while kill -0 "$RUNNER_PID" 2>/dev/null; do
  wait "$RUNNER_PID" || true
done
wait "$TEE_PID"
rm -f "$PIPE_FILE"
trap - INT TERM HUP

if [ -f "$RC_FILE" ]; then
  RC="$(cat "$RC_FILE")"
  rm -f "$RC_FILE"
else
  RC=255
fi

echo "request_0513_modular rc=$RC log_dir=$LOG_DIR log_file=$LOG_FILE"
exit "$RC"
