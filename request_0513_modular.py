# -*- coding: utf-8 -*-
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RK3588 person-follow runtime.

The current runtime uses RKNN vision, IIO IR sensors, and the direct
LZ-30EMA_2EC_N RS485 motor driver on /dev/ttyS*.
"""

import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Tuple, Optional

from car_control_modular.config_loader import preload_config_from_argv
from car_control_modular.control_types import (
    ControlAction,
    ControlDecision,
    DistanceState,
    HazardState,
    LateralCandidateEvidence,
    ObstacleState,
    PersonTarget,
    SensorFrame,
)
from car_control_modular.controllers import (
    GEOMETRY_FALLBACK_TARGET_ID,
    FollowPolicyConfig,
    FollowSafetyController,
)
from car_control_modular.action_runtime import ActionRuntimeConfig, ActionRuntimeSymbols, MotionActionRuntime
from car_control_modular.action_queue_policy import merge_pending_actions
from car_control_modular.distance_runtime import DistanceRuntime, DistanceRuntimeConfig
from car_control_modular.hazard_runtime import BunkerHazardRuntime, BunkerHazardRuntimeConfig
from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig, normalize_mssd_stop_mode
from car_control_modular.sensor_modules import SensorRuntime, SensorRuntimeConfig
from car_control_modular.search_diagnostics import (
    DetectionObservation,
    MotionObservation,
    RecognitionObservation,
    SearchControlObservation,
    SearchDiagnosticSample,
    SearchDiagnosticsConfig,
    SearchDiagnosticsObserver,
    TransportObservation,
)
from car_control_modular.search_candidate_gate import (
    CandidateObservation,
    SearchCandidateGate,
    SearchCandidateGateConfig,
    SearchCandidateGateDecision,
)
from car_control_modular.lateral_intent import (
    LateralControlIntent,
    LateralIntentStore,
    slew_signed_rpm,
)
from car_control_modular.video_recorder import (
    AsyncVideoRecorder,
    VideoControlOverlay,
    VideoRecorderConfig,
    VideoTrackOverlay,
)

try:
    import cv2
    _CV2_IMPORT_ERROR = None
except Exception as e:
    cv2 = None
    _CV2_IMPORT_ERROR = e

# 先config读取变量到 环境变量
LOADED_CONFIG = preload_config_from_argv()
FOLLOW_ROTATION_ONLY = os.environ.get("FOLLOW_ROTATION_ONLY", "0").strip() != "0"

# 传感读取默认对齐 read_ir_sr04_fast.py：
# - IR 走本地 IIO，对齐 read_ir_sr04_fast.py：right=device4、left=device3、front=device5。
# - SR04 走本地 IIO（自动按 name=hcsr04 查找）
# 环境变量若已由配置/外部传入则优先使用外部值。
os.environ.setdefault("IR_BACKEND", "iio")
os.environ.setdefault("IR_IIO_BASE_DIR", "/sys/bus/iio/devices")
os.environ.setdefault("IR_IIO_RIGHT_DEVICE", "4")
os.environ.setdefault("IR_IIO_LEFT_DEVICE", "3")
os.environ.setdefault("IR_IIO_FRONT_DEVICE", "5")
os.environ.setdefault("ULTRASONIC_BACKEND", "iio")
os.environ.setdefault("ULTRASONIC_IIO_BASE_DIR", "/sys/bus/iio/devices")
os.environ.setdefault("ULTRASONIC_IIO_DEVICE_NAME", "hcsr04")

# Board motor output uses direct LZ-30EMA RS485; vision uses RKNN Runtime.
from track_first_person2 import FirstPersonTracker

try:
    from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline
    _RKNN_VISION_IMPORT_ERROR = None
except Exception as e:
    RKNNVisionConfig = None
    RKNNVisionPipeline = None
    _RKNN_VISION_IMPORT_ERROR = e

try:
    from rk_vision.gstreamer_capture import GstMjpegTeeCapture, GstMjpegTeeConfig
    _RKNN_GST_CAPTURE_IMPORT_ERROR = None
except Exception as e:
    GstMjpegTeeCapture = None
    GstMjpegTeeConfig = None
    _RKNN_GST_CAPTURE_IMPORT_ERROR = e

try:
    from rk_vision.direction_worker import DirectionEvidence, DirectionInferencePool
    from rk_vision.yolo11 import YOLO11Config
    _RKNN_DIRECTION_IMPORT_ERROR = None
except Exception as e:
    DirectionEvidence = None
    DirectionInferencePool = None
    YOLO11Config = None
    _RKNN_DIRECTION_IMPORT_ERROR = e

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 退出阶段的设备释放可能进入驱动/GStreamer 的 C 调用，不能让单个设备
# 无限期阻塞整个进程。超时后释放线程作为 daemon 留在后台，外层启动脚本
# 会继续执行最终的电机安全收尾和强制退出。
SHUTDOWN_STEP_TIMEOUT_SEC = max(
    0.5,
    float(os.environ.get("FOLLOW_SHUTDOWN_STEP_TIMEOUT_SEC", "3.0")),
)

# RK3588 motor backend: direct /dev/ttyS* LZ-30EMA_2EC_N RS485 driver.
# Legacy MSSD names remain accepted so older INI files still start safely.
MOTOR_BACKEND = os.environ.get("MOTOR_BACKEND", "rs485_lz30ema").strip().lower()
if MOTOR_BACKEND not in {
    "rs485",
    "rs485_lz30ema",
    "lz30ema",
    "lianzhan",
    "rs485_mssd",
    "mssd",
    "mssd60ehb",
}:
    raise RuntimeError(f"Unsupported MOTOR_BACKEND={MOTOR_BACKEND!r}; RK3588 runtime requires rs485_lz30ema")


MOTOR_RS485_PORT = os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0").strip()
MOTOR_RS485_SLAVE_ID = int(os.environ.get("MOTOR_RS485_SLAVE_ID", "1"))
MOTOR_RS485_BAUDRATE = int(os.environ.get("MOTOR_RS485_BAUDRATE", "9600"))
MOTOR_RS485_TIMEOUT = float(os.environ.get("MOTOR_RS485_TIMEOUT", "0.3"))
MOTOR_RS485_LIB_DIR = os.environ.get("MOTOR_RS485_LIB_DIR", "/home/topeet/lianzhan").strip()
MOTOR_RS485_MAX_TARGET = int(os.environ.get("MOTOR_RS485_MAX_TARGET", "100"))
MOTOR_RS485_TARGET_MIN_INTERVAL_SEC = max(
    0.0,
    float(os.environ.get("MOTOR_RS485_TARGET_MIN_INTERVAL_SEC", "0.05")),
)
MOTOR_FORWARD_LIKE_KEEPALIVE_SEC = float(os.environ.get("MOTOR_FORWARD_LIKE_KEEPALIVE_SEC", "1.0"))
MOTOR_FORWARD_RAW_TARGET = max(0, int(os.environ.get("MOTOR_FORWARD_RAW_TARGET", "0")))
MOTOR_FORWARD_MAX_TARGET_RPM = max(0, int(os.environ.get("MOTOR_FORWARD_MAX_TARGET_RPM", "100")))
MOTOR_FORWARD_UPDATE_MIN_DELTA_RPM = max(
    0,
    int(os.environ.get("MOTOR_FORWARD_UPDATE_MIN_DELTA_RPM", "5")),
)
MOTOR_STEER_RAW_TARGET = max(0, int(os.environ.get("MOTOR_STEER_RAW_TARGET", "0")))
MOTOR_ROTATE_RAW_TARGET = max(0, int(os.environ.get("MOTOR_ROTATE_RAW_TARGET", "0")))
ROTATE_RAW_TARGET_VISIBLE = max(0, int(os.environ.get("ROTATE_RAW_TARGET_VISIBLE", str(MOTOR_ROTATE_RAW_TARGET))))
ROTATE_RAW_TARGET_LOST_WAIT = max(0, int(os.environ.get("ROTATE_RAW_TARGET_LOST_WAIT", str(MOTOR_ROTATE_RAW_TARGET))))
ROTATE_RAW_TARGET_SEARCH = max(0, int(os.environ.get("ROTATE_RAW_TARGET_SEARCH", str(MOTOR_ROTATE_RAW_TARGET))))
MOTOR_LEFT_SIGN = int(os.environ.get("MOTOR_LEFT_SIGN", "-1"))
MOTOR_RIGHT_SIGN = int(os.environ.get("MOTOR_RIGHT_SIGN", "1"))
MOTOR_FORWARD_TARGET_SIGN = -1 if int(os.environ.get("MOTOR_FORWARD_TARGET_SIGN", "1")) < 0 else 1
MOTOR_PARKING_CURRENT_A = max(0.0, min(30.0, float(os.environ.get("MOTOR_PARKING_CURRENT_A", "5.0"))))
MOTOR_EXIT_PARKING_MODE_ON_ARM = os.environ.get("MOTOR_EXIT_PARKING_MODE_ON_ARM", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MOTOR_RS485_STOP_MODE = normalize_mssd_stop_mode(os.environ.get("MOTOR_RS485_STOP_MODE", "normal"), "normal")
SAFETY_STOP_MODE = normalize_mssd_stop_mode(os.environ.get("SAFETY_STOP_MODE", "emergency"), "emergency")
MOTOR_RS485_STOP_ZERO_DELAY_SEC = max(0.0, float(os.environ.get("MOTOR_RS485_STOP_ZERO_DELAY_SEC", "0.03")))
MOTOR_RS485_TRANSITION_STOP_MODE = normalize_mssd_stop_mode(
    os.environ.get("MOTOR_RS485_TRANSITION_STOP_MODE", "emergency"),
    "emergency",
)
MOTOR_RS485_TRANSITION_STOP_DELAY_SEC = max(
    0.0,
    float(os.environ.get("MOTOR_RS485_TRANSITION_STOP_DELAY_SEC", "0.05")),
)
MOTOR_RS485_TRANSITION_STOP_REPEAT = max(1, int(os.environ.get("MOTOR_RS485_TRANSITION_STOP_REPEAT", "1")))
SAFETY_STOP_REASONS = {
    "front_ir",
    "left_ir",
    "right_ir",
    "distance_too_close",
    "person_too_close_no_rotate",
    "search_both_sides_blocked",
    "hard_stop",
    "reverse_radar_unstable_stop",
    "reverse_visual_lost_stop",
    "bunker",
    "bunker_hazard",
    "split_bunker",
    "merged_bunker",
}

def _parse_int_set_env(name: str, default: str = "") -> set:
    raw = os.environ.get(name, default).strip()
    out = set()
    if not raw:
        return out
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.add(int(part, 0))
    return out


def _parse_class_names_env(name: str, default: str = "") -> dict:
    raw = os.environ.get(name, default).strip()
    out = {}
    if not raw:
        return out
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            out[int(key, 0)] = value
    return out


def _resolve_existing_or_workdir_path(path: str, workdir: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.abspath(os.path.join(workdir, path))


def _tracker_state_name(state: int) -> str:
    return {0: "NEW", 1: "UNSTABLE", 2: "STABLE"}.get(int(state), f"UNKNOWN({state})")

# ==================== 配置区域 ====================
MODULE_VISION_ENABLE = os.environ.get("MODULE_VISION_ENABLE", "1").strip() != "0"
MODULE_IR_ENABLE = os.environ.get("MODULE_IR_ENABLE", "1").strip() != "0"
# Keep the mmWave implementation and config intact, but disable it for the
# current Astra Depth runtime so no stale radar safety path can affect motion.
MMWAVE_RUNTIME_DISABLED = os.environ.get("MMWAVE_RUNTIME_DISABLED", "1").strip() != "0"
_MODULE_MMWAVE_REQUESTED = os.environ.get("MODULE_MMWAVE_ENABLE", "0").strip() != "0"
MODULE_MMWAVE_ENABLE = bool(_MODULE_MMWAVE_REQUESTED and not MMWAVE_RUNTIME_DISABLED)
MODULE_ASTRA_DEPTH_ENABLE = bool(
    os.environ.get("MODULE_ASTRA_DEPTH_ENABLE", "0").strip() != "0"
    and not FOLLOW_ROTATION_ONLY
)
MODULE_ULTRASONIC_ENABLE = bool(
    os.environ.get("MODULE_ULTRASONIC_ENABLE", "0").strip() != "0"
    and not FOLLOW_ROTATION_ONLY
)
MODULE_IMU_ENABLE = os.environ.get("MODULE_IMU_ENABLE", "0").strip() != "0"
IMU_FAIL_SOFT = os.environ.get("IMU_FAIL_SOFT", "1").strip() != "0"
IMU_LOG_ENABLE = os.environ.get("IMU_LOG_ENABLE", "0").strip() != "0"
IMU_LOG_EVERY_SEC = max(0.05, float(os.environ.get("IMU_LOG_EVERY_SEC", "1.0")))
MMWAVE_ASYNC_CACHE_ENABLE = os.environ.get("MMWAVE_ASYNC_CACHE_ENABLE", "1").strip() != "0"
MMWAVE_CACHE_INTERVAL_SEC = max(0.01, float(os.environ.get("MMWAVE_CACHE_INTERVAL_SEC", "0.04")))
MMWAVE_CACHE_WINDOW_SEC = max(0.10, float(os.environ.get("MMWAVE_CACHE_WINDOW_SEC", "0.80")))
_DISTANCE_SOURCE_REQUESTED = os.environ.get("DISTANCE_SOURCE", "mmwave").strip().lower()
VISION_MMWAVE_SOURCE_ALIASES = {"vision_mmwave", "vision-mmwave", "vision+mmwave", "mmwave_vision"}
VISION_DEPTH_SOURCE_ALIASES = {"vision_depth", "vision-depth", "astra_depth", "astra-depth"}
# With mmWave disabled, never leave a radar source selected by an old shell
# environment. Depth is the active range source when Astra is enabled.
if FOLLOW_ROTATION_ONLY:
    DISTANCE_SOURCE = "none"
elif MMWAVE_RUNTIME_DISABLED:
    DISTANCE_SOURCE = "vision_depth"
else:
    DISTANCE_SOURCE = _DISTANCE_SOURCE_REQUESTED
VISION_MMWAVE_ENABLED = DISTANCE_SOURCE in VISION_MMWAVE_SOURCE_ALIASES and MODULE_MMWAVE_ENABLE
VISION_DEPTH_ENABLED = DISTANCE_SOURCE in VISION_DEPTH_SOURCE_ALIASES
ASTRA_DEPTH_OPENNI_PATH = os.environ.get(
    "ASTRA_DEPTH_OPENNI_PATH", "/home/topeet/AstraSDK/arm64_openni2"
).strip()
ASTRA_DEPTH_WIDTH = max(1, int(os.environ.get("ASTRA_DEPTH_WIDTH", "640")))
ASTRA_DEPTH_HEIGHT = max(1, int(os.environ.get("ASTRA_DEPTH_HEIGHT", "480")))
ASTRA_DEPTH_FPS = max(1, int(os.environ.get("ASTRA_DEPTH_FPS", "30")))
ASTRA_DEPTH_MIN_DISTANCE_M = max(0.05, float(os.environ.get("ASTRA_DEPTH_MIN_DISTANCE_M", "0.35")))
ASTRA_DEPTH_MAX_DISTANCE_M = max(
    ASTRA_DEPTH_MIN_DISTANCE_M + 0.1,
    float(os.environ.get("ASTRA_DEPTH_MAX_DISTANCE_M", "8.0")),
)
ASTRA_DEPTH_MAX_FRAME_AGE_SEC = max(
    0.05, float(os.environ.get("ASTRA_DEPTH_MAX_FRAME_AGE_SEC", "0.25"))
)
ASTRA_DEPTH_HOLD_SEC = max(0.0, float(os.environ.get("ASTRA_DEPTH_HOLD_SEC", "0.20")))
ASTRA_DEPTH_RGB_PROCESSING_DELAY_SEC = max(
    0.0,
    min(0.50, float(os.environ.get("ASTRA_DEPTH_RGB_PROCESSING_DELAY_SEC", "0.05"))),
)
ASTRA_DEPTH_ROI_LEFT_RATIO = float(os.environ.get("ASTRA_DEPTH_ROI_LEFT_RATIO", "0.32"))
ASTRA_DEPTH_ROI_RIGHT_RATIO = float(os.environ.get("ASTRA_DEPTH_ROI_RIGHT_RATIO", "0.68"))
ASTRA_DEPTH_ROI_TOP_RATIO = float(os.environ.get("ASTRA_DEPTH_ROI_TOP_RATIO", "0.22"))
ASTRA_DEPTH_ROI_BOTTOM_RATIO = float(os.environ.get("ASTRA_DEPTH_ROI_BOTTOM_RATIO", "0.68"))
ASTRA_DEPTH_MIN_VALID_PIXELS = max(1, int(os.environ.get("ASTRA_DEPTH_MIN_VALID_PIXELS", "80")))
ASTRA_DEPTH_DYNAMIC_MIN_VALID_FLOOR = max(
    1, int(os.environ.get("ASTRA_DEPTH_DYNAMIC_MIN_VALID_FLOOR", "20"))
)
ASTRA_DEPTH_DYNAMIC_MIN_VALID_FRACTION = max(
    0.0,
    min(1.0, float(os.environ.get("ASTRA_DEPTH_DYNAMIC_MIN_VALID_FRACTION", "0.03"))),
)
ASTRA_DEPTH_MEDIAN_WINDOW = max(1, int(os.environ.get("ASTRA_DEPTH_MEDIAN_WINDOW", "3")))
ASTRA_DEPTH_FOREGROUND_CLUSTER_SPAN_M = max(
    0.05,
    float(os.environ.get("ASTRA_DEPTH_FOREGROUND_CLUSTER_SPAN_M", "0.40")),
)
ASTRA_DEPTH_FOREGROUND_CLUSTER_MIN_FRACTION = max(
    0.01,
    min(
        1.0,
        float(os.environ.get("ASTRA_DEPTH_FOREGROUND_CLUSTER_MIN_FRACTION", "0.06")),
    ),
)
ASTRA_DEPTH_FOREGROUND_SPATIAL_SUPPORT_FRACTION = max(
    0.0,
    min(
        1.0,
        float(os.environ.get("ASTRA_DEPTH_FOREGROUND_SPATIAL_SUPPORT_FRACTION", "0.55")),
    ),
)
ASTRA_DEPTH_TORSO_REGION_MIN_SIZE_PX = max(
    4, int(os.environ.get("ASTRA_DEPTH_TORSO_REGION_MIN_SIZE_PX", "16"))
)
ASTRA_DEPTH_TORSO_REGION_MAX_SIZE_PX = max(
    ASTRA_DEPTH_TORSO_REGION_MIN_SIZE_PX,
    int(os.environ.get("ASTRA_DEPTH_TORSO_REGION_MAX_SIZE_PX", "64")),
)
ASTRA_DEPTH_CENTER_PATCH_SIZE = max(
    4,
    int(os.environ.get("ASTRA_DEPTH_CENTER_PATCH_SIZE", "16")),
)
ASTRA_DEPTH_CENTER_PATCH_KEEP_COUNT = max(
    1,
    int(os.environ.get("ASTRA_DEPTH_CENTER_PATCH_KEEP_COUNT", "4")),
)
ASTRA_DEPTH_CENTER_PATCH_MIN_VALID_FRACTION = max(
    0.0,
    min(
        1.0,
        float(os.environ.get("ASTRA_DEPTH_CENTER_PATCH_MIN_VALID_FRACTION", "0.25")),
    ),
)
ASTRA_DEPTH_LARGE_BBOX_GUARD_AREA_RATIO = max(
    0.05,
    min(1.0, float(os.environ.get("ASTRA_DEPTH_LARGE_BBOX_GUARD_AREA_RATIO", "0.35"))),
)
ASTRA_DEPTH_LARGE_BBOX_GUARD_HEIGHT_RATIO = max(
    0.05,
    min(1.0, float(os.environ.get("ASTRA_DEPTH_LARGE_BBOX_GUARD_HEIGHT_RATIO", "0.90"))),
)
ASTRA_DEPTH_LARGE_BBOX_GUARD_MAX_DISTANCE_M = max(
    ASTRA_DEPTH_MIN_DISTANCE_M,
    float(os.environ.get("ASTRA_DEPTH_LARGE_BBOX_GUARD_MAX_DISTANCE_M", "2.50")),
)
ASTRA_DEPTH_MAX_DISTANCE_JUMP_M = max(
    0.0, float(os.environ.get("ASTRA_DEPTH_MAX_DISTANCE_JUMP_M", "0.80"))
)
ASTRA_DEPTH_JUMP_CONFIRM_FRAMES = max(
    1, int(os.environ.get("ASTRA_DEPTH_JUMP_CONFIRM_FRAMES", "2"))
)
ASTRA_DEPTH_NEAR_GUARD_DISTANCE_M = max(
    ASTRA_DEPTH_MIN_DISTANCE_M,
    float(os.environ.get("ASTRA_DEPTH_NEAR_GUARD_DISTANCE_M", "1.80")),
)
ASTRA_DEPTH_NEAR_FAR_JUMP_CONFIRM_FRAMES = max(
    ASTRA_DEPTH_JUMP_CONFIRM_FRAMES,
    int(os.environ.get("ASTRA_DEPTH_NEAR_FAR_JUMP_CONFIRM_FRAMES", "5")),
)
ASTRA_DEPTH_ANCHOR_STRICT_AGE_SEC = max(
    0.0, float(os.environ.get("ASTRA_DEPTH_ANCHOR_STRICT_AGE_SEC", "0.60"))
)
ASTRA_DEPTH_ANCHOR_EXPIRE_AGE_SEC = max(
    ASTRA_DEPTH_ANCHOR_STRICT_AGE_SEC,
    float(os.environ.get("ASTRA_DEPTH_ANCHOR_EXPIRE_AGE_SEC", "1.50")),
)
ASTRA_DEPTH_REANCHOR_CONFIRM_FRAMES = max(
    1, int(os.environ.get("ASTRA_DEPTH_REANCHOR_CONFIRM_FRAMES", "3"))
)
ASTRA_DEPTH_MOTION_CONFIRM_FRAMES = max(
    1, int(os.environ.get("ASTRA_DEPTH_MOTION_CONFIRM_FRAMES", "3"))
)
ASTRA_DEPTH_MOTION_REVERSE_MIN_M = max(
    0.0, float(os.environ.get("ASTRA_DEPTH_MOTION_REVERSE_MIN_M", "0.08"))
)
ASTRA_DEPTH_ENCODER_WHEEL_CIRCUMFERENCE_M = max(
    0.0,
    float(os.environ.get("VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M", "0.60")),
)
ASTRA_DEPTH_NEAR_FAR_JUMP_MAX_BBOX_RATIO = max(
    0.05,
    min(1.0, float(os.environ.get("ASTRA_DEPTH_NEAR_FAR_JUMP_MAX_BBOX_RATIO", "0.90"))),
)
ASTRA_DEPTH_NEAR_FAR_JUMP_EDGE_MARGIN_RATIO = max(
    0.0,
    min(0.20, float(os.environ.get("ASTRA_DEPTH_NEAR_FAR_JUMP_EDGE_MARGIN_RATIO", "0.02"))),
)
ASTRA_DEPTH_LOG_EVERY_SEC = max(0.1, float(os.environ.get("ASTRA_DEPTH_LOG_EVERY_SEC", "1.0")))
# Depth原始流是30 FPS；纵向监督线程在两个YOLO结果之间复用最近人物框，
# 但人物框超过120ms后绝不再生成新运动命令。
ASTRA_DEPTH_LONGITUDINAL_CONTROL_ENABLE = os.environ.get(
    "ASTRA_DEPTH_LONGITUDINAL_CONTROL_ENABLE",
    "1",
).strip() != "0" and not FOLLOW_ROTATION_ONLY
ASTRA_DEPTH_LONGITUDINAL_CONTROL_HZ = max(
    1.0,
    min(60.0, float(os.environ.get("ASTRA_DEPTH_LONGITUDINAL_CONTROL_HZ", "30.0"))),
)
ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC = max(
    0.03,
    min(
        0.30,
        float(os.environ.get("ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", "0.12")),
    ),
)
ASTRA_DEPTH_VISION_DEDUPE_SEC = max(
    0.02,
    float(os.environ.get("ASTRA_DEPTH_VISION_DEDUPE_SEC", "0.08")),
)
VISION_CONTROL_MAX_RESULT_AGE_SEC = max(
    0.05,
    min(0.50, float(os.environ.get("VISION_CONTROL_MAX_RESULT_AGE_SEC", "0.12"))),
)

# 跟踪参数
DETECTION_CONFIRM_FRAMES = 3
LOST_CONFIRM_FRAMES = 3   # 防抖：连续 3 帧无人才确认丢人并开始旋转，避免检测偶发漏检
ACTION_COOLDOWN = max(1, int(os.environ.get("FOLLOW_ACTION_COOLDOWN_FRAMES", "1")))
LOST_CONFIRM_FRAMES = int(os.environ.get("FOLLOW_LOST_CONFIRM_FRAMES", str(LOST_CONFIRM_FRAMES)))
FOLLOW_LOST_CONFIRM_SEC = max(0.0, float(os.environ.get("FOLLOW_LOST_CONFIRM_SEC", "0.0")))
FOLLOW_STALE_DIRECTION_RECOVERY_ENABLE = os.environ.get(
    "FOLLOW_STALE_DIRECTION_RECOVERY_ENABLE",
    "0",
).strip().lower() in ("1", "true", "yes", "on")
FOLLOW_DIRECTION_HISTORY_ENABLE = os.environ.get(
    "FOLLOW_DIRECTION_HISTORY_ENABLE",
    "1",
).strip().lower() in ("1", "true", "yes", "on")
FOLLOW_STALE_DIRECTION_OBSERVE_FRAMES = max(
    1,
    int(os.environ.get("FOLLOW_STALE_DIRECTION_OBSERVE_FRAMES", "2")),
)
FOLLOW_STALE_DIRECTION_OBSERVE_MAX_SEC = max(
    0.05,
    float(os.environ.get("FOLLOW_STALE_DIRECTION_OBSERVE_MAX_SEC", "0.20")),
)
FOLLOW_STALE_DIRECTION_SETTLE_YAW_RATE_DPS = max(
    0.0,
    float(os.environ.get("FOLLOW_STALE_DIRECTION_SETTLE_YAW_RATE_DPS", "3.0")),
)
FOLLOW_STALE_DIRECTION_PROBE_ENABLE = os.environ.get(
    "FOLLOW_STALE_DIRECTION_PROBE_ENABLE",
    "0",
).strip().lower() in ("1", "true", "yes", "on")
FOLLOW_STALE_DIRECTION_PROBE_ANGLE_DEG = max(
    2.0,
    float(os.environ.get("FOLLOW_STALE_DIRECTION_PROBE_ANGLE_DEG", "12.0")),
)
FOLLOW_STALE_DIRECTION_PROBE_OBSERVE_FRAMES = max(
    1,
    int(os.environ.get("FOLLOW_STALE_DIRECTION_PROBE_OBSERVE_FRAMES", "2")),
)
FOLLOW_STALE_DIRECTION_PROBE_RETURN_TOLERANCE_DEG = max(
    0.5,
    float(
        os.environ.get(
            "FOLLOW_STALE_DIRECTION_PROBE_RETURN_TOLERANCE_DEG",
            "2.0",
        )
    ),
)
FOLLOW_STALE_DIRECTION_PROBE_RAW_TARGET = max(
    1,
    int(os.environ.get("FOLLOW_STALE_DIRECTION_PROBE_RAW_TARGET", "8")),
)
FOLLOW_STEER_MIN_HOLD_SEC = max(0.0, float(os.environ.get("FOLLOW_STEER_MIN_HOLD_SEC", "0.25")))
FOLLOW_STEER_LOST_HOLD_FRAMES = max(0, int(os.environ.get("FOLLOW_STEER_LOST_HOLD_FRAMES", "2")))
FOLLOW_STEER_LOST_HOLD_MAX_SEC = max(0.0, float(os.environ.get("FOLLOW_STEER_LOST_HOLD_MAX_SEC", "0.40")))
FOLLOW_LOST_FORWARD_HOLD_RPM = max(0, int(os.environ.get("FOLLOW_LOST_FORWARD_HOLD_RPM", "20")))
FOLLOW_LOST_FORWARD_HOLD_MAX_SEC = max(
    0.0,
    float(os.environ.get("FOLLOW_LOST_FORWARD_HOLD_MAX_SEC", "0.80")),
)
FOLLOW_LOST_FORWARD_HOLD_MIN_DISTANCE_M = max(
    0.0,
    float(os.environ.get("FOLLOW_LOST_FORWARD_HOLD_MIN_DISTANCE_M", "1.50")),
)
FOLLOW_DEPTH_MEDIUM_CONFIDENCE_HOLD_SEC = max(
    0.20,
    float(os.environ.get("FOLLOW_DEPTH_MEDIUM_CONFIDENCE_HOLD_SEC", "0.60")),
)
FOLLOW_DEPTH_MEDIUM_CONFIDENCE_RPM = max(
    0, int(os.environ.get("FOLLOW_DEPTH_MEDIUM_CONFIDENCE_RPM", "20"))
)
FOLLOW_DEPTH_RECOVERY_STAGE1_SEC = max(
    0.0, float(os.environ.get("FOLLOW_DEPTH_RECOVERY_STAGE1_SEC", "0.20"))
)
FOLLOW_DEPTH_RECOVERY_STAGE2_SEC = max(
    FOLLOW_DEPTH_RECOVERY_STAGE1_SEC,
    float(os.environ.get("FOLLOW_DEPTH_RECOVERY_STAGE2_SEC", "0.40")),
)
FOLLOW_DEPTH_RECOVERY_STAGE1_RPM = max(
    0, int(os.environ.get("FOLLOW_DEPTH_RECOVERY_STAGE1_RPM", "25"))
)
FOLLOW_DEPTH_RECOVERY_STAGE2_RPM = max(
    FOLLOW_DEPTH_RECOVERY_STAGE1_RPM,
    int(os.environ.get("FOLLOW_DEPTH_RECOVERY_STAGE2_RPM", "45")),
)
SEARCH_COOLDOWN = 1
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))
FOLLOW_CENTER_LEFT_RATIO = float(os.environ.get("FOLLOW_CENTER_LEFT_RATIO", str(1.0 / 6.0)))
FOLLOW_CENTER_RIGHT_RATIO = float(os.environ.get("FOLLOW_CENTER_RIGHT_RATIO", str(4.0 / 6.0)))
FOLLOW_CENTER_DEADZONE_RATIO = max(
    0.01,
    min(0.45, float(os.environ.get("FOLLOW_CENTER_DEADZONE_RATIO", "0.08"))),
)
FOLLOW_STEER_ENTER_LEFT_RATIO = float(os.environ.get("FOLLOW_STEER_ENTER_LEFT_RATIO", str(FOLLOW_CENTER_LEFT_RATIO)))
FOLLOW_STEER_ENTER_RIGHT_RATIO = float(os.environ.get("FOLLOW_STEER_ENTER_RIGHT_RATIO", str(FOLLOW_CENTER_RIGHT_RATIO)))
FOLLOW_STEER_RELEASE_LEFT_RATIO = float(os.environ.get("FOLLOW_STEER_RELEASE_LEFT_RATIO", str(FOLLOW_CENTER_LEFT_RATIO)))
FOLLOW_STEER_RELEASE_RIGHT_RATIO = float(os.environ.get("FOLLOW_STEER_RELEASE_RIGHT_RATIO", str(FOLLOW_CENTER_RIGHT_RATIO)))
FOLLOW_VISIBLE_ROTATE_LEFT_RATIO = float(os.environ.get("FOLLOW_VISIBLE_ROTATE_LEFT_RATIO", "0.0"))
FOLLOW_VISIBLE_ROTATE_RIGHT_RATIO = float(os.environ.get("FOLLOW_VISIBLE_ROTATE_RIGHT_RATIO", "1.0"))
FOLLOW_USE_VERTICAL_CENTER_GATE = os.environ.get("FOLLOW_USE_VERTICAL_CENTER_GATE", "0").strip() != "0"
FOLLOW_SEARCH_BEFORE_FIRST_PERSON = os.environ.get("FOLLOW_SEARCH_BEFORE_FIRST_PERSON", "0").strip() != "0"
FOLLOW_STARTUP_SEARCH_DELAY_SEC = float(os.environ.get("FOLLOW_STARTUP_SEARCH_DELAY_SEC", "0.0"))
FOLLOW_SEARCH_TIMEOUT_SEC = max(0.0, float(os.environ.get("FOLLOW_SEARCH_TIMEOUT_SEC", "5.0")))
FOLLOW_SEARCH_TIMEOUT_EXIT_PROGRAM = os.environ.get(
    "FOLLOW_SEARCH_TIMEOUT_EXIT_PROGRAM", "0"
).strip() != "0"
FOLLOW_EXIT_ON_TARGET_LOSS = os.environ.get(
    "FOLLOW_EXIT_ON_TARGET_LOSS", "0"
).strip() != "0"
FOLLOW_SEARCH_REVOLUTION_DEG = max(
    90.0,
    float(os.environ.get("FOLLOW_SEARCH_REVOLUTION_DEG", "360.0")),
)
FOLLOW_SEARCH_REVOLUTION_FEEDBACK_STALE_SEC = max(
    0.10,
    float(os.environ.get("FOLLOW_SEARCH_REVOLUTION_FEEDBACK_STALE_SEC", "0.80")),
)
FOLLOW_INITIAL_TARGET_CONFIRM_FRAMES = max(1, int(os.environ.get("FOLLOW_INITIAL_TARGET_CONFIRM_FRAMES", "1")))
SEARCH_CONFIRMED_REACQUIRE_FRAMES = max(
    1,
    int(os.environ.get("SEARCH_CONFIRMED_REACQUIRE_FRAMES", "1")),
)
SEARCH_EVIDENCE_GATE_ENABLE = (
    os.environ.get("SEARCH_EVIDENCE_GATE_ENABLE", "1").strip() != "0"
)
SEARCH_EVIDENCE_HOLD_FRAMES = max(
    1, int(os.environ.get("SEARCH_EVIDENCE_HOLD_FRAMES", "1"))
)
SEARCH_EVIDENCE_MAX_HOLD_SEC = max(
    0.10,
    min(0.50, float(os.environ.get("SEARCH_EVIDENCE_MAX_HOLD_SEC", "0.30"))),
)
SEARCH_EVIDENCE_REARM_MISSING_FRAMES = max(
    1, int(os.environ.get("SEARCH_EVIDENCE_REARM_MISSING_FRAMES", "8"))
)
SEARCH_EVIDENCE_PROBE_CONFIRM_FRAMES = max(
    1, int(os.environ.get("SEARCH_EVIDENCE_PROBE_CONFIRM_FRAMES", "1"))
)
SEARCH_EVIDENCE_PROBE_MIN_SCORE = max(
    0.01,
    min(
        CONFIDENCE_THRESHOLD,
        float(os.environ.get("SEARCH_EVIDENCE_PROBE_MIN_SCORE", "0.10")),
    ),
)
SEARCH_EVIDENCE_MIN_AREA_RATIO = max(
    0.001, min(0.90, float(os.environ.get("SEARCH_EVIDENCE_MIN_AREA_RATIO", "0.01")))
)
SEARCH_EVIDENCE_MAX_AREA_RATIO = max(
    SEARCH_EVIDENCE_MIN_AREA_RATIO,
    min(1.0, float(os.environ.get("SEARCH_EVIDENCE_MAX_AREA_RATIO", "0.75"))),
)
SEARCH_EVIDENCE_CONSISTENCY_IOU = max(
    0.0, min(1.0, float(os.environ.get("SEARCH_EVIDENCE_CONSISTENCY_IOU", "0.20")))
)
SEARCH_CANDIDATE_UNTRACKED_MIN_SCORE = max(
    0.01,
    min(
        0.95,
        float(os.environ.get("SEARCH_CANDIDATE_UNTRACKED_MIN_SCORE", "0.60")),
    ),
)
SEARCH_CANDIDATE_APPROACH_MARGIN_RATIO = max(
    0.0,
    min(
        0.25,
        float(os.environ.get("SEARCH_CANDIDATE_APPROACH_MARGIN_RATIO", "0.08")),
    ),
)
SEARCH_CANDIDATE_ACQUIRE_RAW_RPM = max(
    1,
    min(18, int(os.environ.get("SEARCH_CANDIDATE_ACQUIRE_RAW_RPM", "6"))),
)
SEARCH_ROTATE_CONTINUOUS_ENABLE = (
    os.environ.get("SEARCH_ROTATE_CONTINUOUS_ENABLE", "1").strip() != "0"
)
SEARCH_CANDIDATE_CONTINUE_ROTATE_MAX_REID_DISTANCE = max(
    0.0,
    min(
        1.0,
        float(os.environ.get("SEARCH_CANDIDATE_CONTINUE_ROTATE_MAX_REID_DISTANCE", "0.20")),
    ),
)
SEARCH_GEOMETRY_REACQUIRE_FRAMES = max(
    2,
    int(os.environ.get("SEARCH_GEOMETRY_REACQUIRE_FRAMES", "2")),
)
SEARCH_GEOMETRY_REACQUIRE_MAX_GAP_FRAMES = max(
    1,
    int(os.environ.get("SEARCH_GEOMETRY_REACQUIRE_MAX_GAP_FRAMES", "2")),
)
FOLLOW_RELEASE_TARGET_ON_LOST = os.environ.get("FOLLOW_RELEASE_TARGET_ON_LOST", "1").strip() != "0"
SIDE_IR_BLOCKS_ROTATION = os.environ.get("SIDE_IR_BLOCKS_ROTATION", "1").strip() != "0"
SIDE_IR_CONFIRM_SEC = max(0.0, float(os.environ.get("SIDE_IR_CONFIRM_SEC", "0.10")))
SIDE_IR_RELEASE_SEC = max(0.0, float(os.environ.get("SIDE_IR_RELEASE_SEC", "0.20")))
SEARCH_ROTATE_FRONT_BLOCK_ENABLE = os.environ.get("SEARCH_ROTATE_FRONT_BLOCK_ENABLE", "1").strip() != "0"
SEARCH_ROTATE_DISTANCE_BLOCK_ENABLE = os.environ.get("SEARCH_ROTATE_DISTANCE_BLOCK_ENABLE", "1").strip() != "0"
SENSOR_RUNTIME_CONFIG = SensorRuntimeConfig(
    ir_enable=MODULE_IR_ENABLE,
    ultrasonic_enable=MODULE_ULTRASONIC_ENABLE,
    mmwave_enable=MODULE_MMWAVE_ENABLE,
    astra_depth_enable=MODULE_ASTRA_DEPTH_ENABLE,
    astra_depth_openni_path=ASTRA_DEPTH_OPENNI_PATH,
    astra_depth_width=ASTRA_DEPTH_WIDTH,
    astra_depth_height=ASTRA_DEPTH_HEIGHT,
    astra_depth_fps=ASTRA_DEPTH_FPS,
    astra_depth_min_distance_m=ASTRA_DEPTH_MIN_DISTANCE_M,
    astra_depth_max_distance_m=ASTRA_DEPTH_MAX_DISTANCE_M,
    astra_depth_max_frame_age_sec=ASTRA_DEPTH_MAX_FRAME_AGE_SEC,
    astra_depth_hold_sec=ASTRA_DEPTH_HOLD_SEC,
    astra_depth_rgb_processing_delay_sec=ASTRA_DEPTH_RGB_PROCESSING_DELAY_SEC,
    astra_depth_roi_left_ratio=ASTRA_DEPTH_ROI_LEFT_RATIO,
    astra_depth_roi_right_ratio=ASTRA_DEPTH_ROI_RIGHT_RATIO,
    astra_depth_roi_top_ratio=ASTRA_DEPTH_ROI_TOP_RATIO,
    astra_depth_roi_bottom_ratio=ASTRA_DEPTH_ROI_BOTTOM_RATIO,
    astra_depth_min_valid_pixels=ASTRA_DEPTH_MIN_VALID_PIXELS,
    astra_depth_dynamic_min_valid_floor=ASTRA_DEPTH_DYNAMIC_MIN_VALID_FLOOR,
    astra_depth_dynamic_min_valid_fraction=ASTRA_DEPTH_DYNAMIC_MIN_VALID_FRACTION,
    astra_depth_median_window=ASTRA_DEPTH_MEDIAN_WINDOW,
    astra_depth_foreground_cluster_span_m=ASTRA_DEPTH_FOREGROUND_CLUSTER_SPAN_M,
    astra_depth_foreground_cluster_min_fraction=ASTRA_DEPTH_FOREGROUND_CLUSTER_MIN_FRACTION,
    astra_depth_foreground_spatial_support_fraction=(
        ASTRA_DEPTH_FOREGROUND_SPATIAL_SUPPORT_FRACTION
    ),
    astra_depth_torso_region_min_size_px=ASTRA_DEPTH_TORSO_REGION_MIN_SIZE_PX,
    astra_depth_torso_region_max_size_px=ASTRA_DEPTH_TORSO_REGION_MAX_SIZE_PX,
    astra_depth_center_patch_size=ASTRA_DEPTH_CENTER_PATCH_SIZE,
    astra_depth_center_patch_keep_count=ASTRA_DEPTH_CENTER_PATCH_KEEP_COUNT,
    astra_depth_center_patch_min_valid_fraction=ASTRA_DEPTH_CENTER_PATCH_MIN_VALID_FRACTION,
    astra_depth_large_bbox_guard_area_ratio=ASTRA_DEPTH_LARGE_BBOX_GUARD_AREA_RATIO,
    astra_depth_large_bbox_guard_height_ratio=ASTRA_DEPTH_LARGE_BBOX_GUARD_HEIGHT_RATIO,
    astra_depth_large_bbox_guard_max_distance_m=ASTRA_DEPTH_LARGE_BBOX_GUARD_MAX_DISTANCE_M,
    astra_depth_max_distance_jump_m=ASTRA_DEPTH_MAX_DISTANCE_JUMP_M,
    astra_depth_jump_confirm_frames=ASTRA_DEPTH_JUMP_CONFIRM_FRAMES,
    astra_depth_near_guard_distance_m=ASTRA_DEPTH_NEAR_GUARD_DISTANCE_M,
    astra_depth_near_far_jump_confirm_frames=ASTRA_DEPTH_NEAR_FAR_JUMP_CONFIRM_FRAMES,
    astra_depth_anchor_strict_age_sec=ASTRA_DEPTH_ANCHOR_STRICT_AGE_SEC,
    astra_depth_anchor_expire_age_sec=ASTRA_DEPTH_ANCHOR_EXPIRE_AGE_SEC,
    astra_depth_reanchor_confirm_frames=ASTRA_DEPTH_REANCHOR_CONFIRM_FRAMES,
    astra_depth_motion_confirm_frames=ASTRA_DEPTH_MOTION_CONFIRM_FRAMES,
    astra_depth_motion_reverse_min_m=ASTRA_DEPTH_MOTION_REVERSE_MIN_M,
    astra_depth_near_far_jump_max_bbox_ratio=ASTRA_DEPTH_NEAR_FAR_JUMP_MAX_BBOX_RATIO,
    astra_depth_near_far_jump_edge_margin_ratio=ASTRA_DEPTH_NEAR_FAR_JUMP_EDGE_MARGIN_RATIO,
    astra_depth_encoder_wheel_circumference_m=(
        ASTRA_DEPTH_ENCODER_WHEEL_CIRCUMFERENCE_M
    ),
    astra_depth_log_every_sec=ASTRA_DEPTH_LOG_EVERY_SEC,
    imu_enable=MODULE_IMU_ENABLE,
    imu_fail_soft=IMU_FAIL_SOFT,
    imu_log_enable=IMU_LOG_ENABLE,
    imu_log_every_sec=IMU_LOG_EVERY_SEC,
    side_ir_blocks_rotation=SIDE_IR_BLOCKS_ROTATION,
    side_ir_confirm_sec=SIDE_IR_CONFIRM_SEC,
    side_ir_release_sec=SIDE_IR_RELEASE_SEC,
    mmwave_async_enable=MMWAVE_ASYNC_CACHE_ENABLE,
    mmwave_poll_interval_sec=MMWAVE_CACHE_INTERVAL_SEC,
    mmwave_cache_window_sec=MMWAVE_CACHE_WINDOW_SEC,
)

# YOLO 模型路径（相对运行目录，与板子 zkwl-runtime 目录结构一致）
YOLO_MODEL_PATH = os.environ.get(
    "VISION_MODEL_PATH",
    os.path.join(SCRIPT_DIR, "models", "yolo11s.rknn"),
)

# Vision engine: request_0513 is the RK3588 RKNN Runtime path.
VISION_REID_ENABLE = os.environ.get("VISION_REID_ENABLE", "0").strip() != "0"
VISION_ENGINE = os.environ.get(
    "VISION_ENGINE",
    "rknn",
).strip().lower()
VISION_REID_MODEL_PATH = os.environ.get(
    "VISION_REID_MODEL_PATH",
    os.path.join(SCRIPT_DIR, "models", "deepsort.rknn"),
).strip()
VISION_FRAME_WIDTH = int(os.environ.get("VISION_FRAME_WIDTH", "1920"))
VISION_FRAME_HEIGHT = int(os.environ.get("VISION_FRAME_HEIGHT", "1080"))
VISION_HFOV_DEG = float(os.environ.get("VISION_HFOV_DEG", "90.0"))
VISION_EFFECTIVE_FPS = max(0.1, float(os.environ.get("VISION_EFFECTIVE_FPS", "4.0")))
VISION_TRACK_LOG_ENABLE = os.environ.get("VISION_TRACK_LOG_ENABLE", "1").strip() != "0"
VISION_TRACK_LOG_EMPTY_EVERY = int(os.environ.get("VISION_TRACK_LOG_EMPTY_EVERY", "20"))
VISION_CONTROL_USE_PREDICTED_TRACKS = os.environ.get("VISION_CONTROL_USE_PREDICTED_TRACKS", "0").strip() != "0"
SINGLE_PERSON_GEOMETRY_FALLBACK_ENABLE = os.environ.get(
    "SINGLE_PERSON_GEOMETRY_FALLBACK_ENABLE", "0"
).strip() != "0"
SINGLE_PERSON_GEOMETRY_CONFIRM_FRAMES = max(
    1, int(os.environ.get("SINGLE_PERSON_GEOMETRY_CONFIRM_FRAMES", "2"))
)
SINGLE_PERSON_GEOMETRY_MAX_GAP_FRAMES = max(
    1, int(os.environ.get("SINGLE_PERSON_GEOMETRY_MAX_GAP_FRAMES", "6"))
)
SINGLE_PERSON_GEOMETRY_MIN_IOU = max(
    0.0, min(1.0, float(os.environ.get("SINGLE_PERSON_GEOMETRY_MIN_IOU", "0.35")))
)
SINGLE_PERSON_GEOMETRY_MAX_CENTER_JUMP_RATIO = max(
    0.01,
    min(1.0, float(os.environ.get("SINGLE_PERSON_GEOMETRY_MAX_CENTER_JUMP_RATIO", "0.22"))),
)
VISIBLE_LOW_QUALITY_RECOVERY_MAX_GAP_SEC = max(
    0.10,
    float(os.environ.get("VISIBLE_LOW_QUALITY_RECOVERY_MAX_GAP_SEC", "0.50")),
)
VISIBLE_LOW_QUALITY_STEER_MAX_CORRECTION_RPM = max(
    0.0,
    float(os.environ.get("VISIBLE_LOW_QUALITY_STEER_MAX_CORRECTION_RPM", "4.0")),
)
VISIBLE_LOW_QUALITY_STEER_EDGE_MAX_CORRECTION_RPM = max(
    VISIBLE_LOW_QUALITY_STEER_MAX_CORRECTION_RPM,
    float(os.environ.get("VISIBLE_LOW_QUALITY_STEER_EDGE_MAX_CORRECTION_RPM", "18.0")),
)
VISIBLE_NEAR_CAMERA_OCCLUSION_AREA_RATIO = max(
    0.70,
    min(1.0, float(os.environ.get("VISIBLE_NEAR_CAMERA_OCCLUSION_AREA_RATIO", "0.82"))),
)
VISIBLE_NEAR_CAMERA_OCCLUSION_WIDTH_RATIO = max(
    0.80,
    min(1.0, float(os.environ.get("VISIBLE_NEAR_CAMERA_OCCLUSION_WIDTH_RATIO", "0.92"))),
)
VISIBLE_NEAR_CAMERA_OCCLUSION_HEIGHT_RATIO = max(
    0.75,
    min(1.0, float(os.environ.get("VISIBLE_NEAR_CAMERA_OCCLUSION_HEIGHT_RATIO", "0.88"))),
)
SEARCH_REID_DIAGNOSTIC_ENABLE = os.environ.get("SEARCH_REID_DIAGNOSTIC_ENABLE", "1").strip() != "0"
SEARCH_FRAME_DIAGNOSTIC_ENABLE = (
    os.environ.get("SEARCH_FRAME_DIAGNOSTIC_ENABLE", "1").strip() != "0"
)
SEARCH_FRAME_DIAGNOSTIC_INTERVAL_SEC = max(
    0.0,
    float(os.environ.get("SEARCH_FRAME_DIAGNOSTIC_INTERVAL_SEC", "0.0")),
)
SEARCH_DIAGNOSTIC_CHECKPOINT_DEG = max(
    10.0,
    float(os.environ.get("SEARCH_DIAGNOSTIC_CHECKPOINT_DEG", "30.0")),
)
SEARCH_DIAGNOSTIC_SNAPSHOT_ENABLE = (
    os.environ.get("SEARCH_DIAGNOSTIC_SNAPSHOT_ENABLE", "1").strip() != "0"
)
SEARCH_DIAGNOSTIC_SNAPSHOT_MAX = max(
    0,
    int(os.environ.get("SEARCH_DIAGNOSTIC_SNAPSHOT_MAX", "24")),
)
SEARCH_DIAGNOSTIC_JPEG_QUALITY = max(
    50,
    min(100, int(os.environ.get("SEARCH_DIAGNOSTIC_JPEG_QUALITY", "88"))),
)
SEARCH_DIAGNOSTIC_BASELINE_EVERY_FRAMES = max(
    1,
    int(os.environ.get("SEARCH_DIAGNOSTIC_BASELINE_EVERY_FRAMES", "10")),
)
SEARCH_DIAGNOSTIC_ROOT = os.path.abspath(
    os.path.join(
        os.environ.get("FOLLOW_LOG_DIR", os.path.join(SCRIPT_DIR, "search_diagnostics")),
        "search_frames",
    )
)
FOLLOW_LOG_DIR = os.path.abspath(
    os.environ.get("FOLLOW_LOG_DIR", os.path.join(SCRIPT_DIR, "run_request_0428_modular_logs"))
)
RKNN_CAMERA_ENABLE = os.environ.get("RKNN_CAMERA_ENABLE", "1").strip() != "0"
RKNN_CAMERA_DEVICE = os.environ.get("RKNN_CAMERA_DEVICE", "/dev/video1").strip()
RKNN_CAMERA_WIDTH = int(os.environ.get("RKNN_CAMERA_WIDTH", str(VISION_FRAME_WIDTH)))
RKNN_CAMERA_HEIGHT = int(os.environ.get("RKNN_CAMERA_HEIGHT", str(VISION_FRAME_HEIGHT)))
RKNN_CAMERA_FPS = float(os.environ.get("RKNN_CAMERA_FPS", "30.0"))
RKNN_CAMERA_FOURCC = os.environ.get("RKNN_CAMERA_FOURCC", "MJPG").strip()
RKNN_CAMERA_CAPTURE_MODE = os.environ.get("RKNN_CAMERA_CAPTURE_MODE", "gstreamer_mjpeg").strip().lower()
_RKNN_CAMERA_RAW_OUTPUT_SETTING = os.environ.get("RKNN_CAMERA_RAW_OUTPUT", "").strip()
if _RKNN_CAMERA_RAW_OUTPUT_SETTING.lower() in {"auto", "default"}:
    RKNN_CAMERA_RAW_OUTPUT = os.path.join(FOLLOW_LOG_DIR, "camera_raw.avi")
elif _RKNN_CAMERA_RAW_OUTPUT_SETTING and not os.path.isabs(_RKNN_CAMERA_RAW_OUTPUT_SETTING):
    RKNN_CAMERA_RAW_OUTPUT = os.path.abspath(
        os.path.join(SCRIPT_DIR, _RKNN_CAMERA_RAW_OUTPUT_SETTING)
    )
else:
    RKNN_CAMERA_RAW_OUTPUT = _RKNN_CAMERA_RAW_OUTPUT_SETTING
RKNN_CAMERA_RETRY_INTERVAL_SEC = max(0.2, float(os.environ.get("RKNN_CAMERA_RETRY_INTERVAL_SEC", "2.0")))
RKNN_CAMERA_LATEST_DRAIN_MAX = max(0, int(os.environ.get("RKNN_CAMERA_LATEST_DRAIN_MAX", "8")))
RKNN_DIRECTION_WORKERS = max(0, int(os.environ.get("RKNN_DIRECTION_WORKERS", "2")))
# Zero keeps an unbounded backlog so every captured frame eventually receives
# visible/missing/unknown direction evidence. Set a positive value only when
# a deployment needs a hard memory ceiling and accepts explicit unknown slots.
RKNN_DIRECTION_QUEUE_SIZE = max(0, int(os.environ.get("RKNN_DIRECTION_QUEUE_SIZE", "0")))
RKNN_CAPTURE_QUEUE_SIZE = max(2, int(os.environ.get("RKNN_CAPTURE_QUEUE_SIZE", "4")))
TRACK_MEMORY_ENABLED = os.environ.get("TRACK_MEMORY_ENABLED", "0").strip() != "0"
TRACK_MEMORY_AUTO_FROM_DEEPSORT = os.environ.get("TRACK_MEMORY_AUTO_FROM_DEEPSORT", "0").strip() != "0"
TRACK_MEMORY_DERIVED_MEMORY_RATIO = max(
    0.1,
    min(1.0, float(os.environ.get("TRACK_MEMORY_DERIVED_MEMORY_RATIO", "0.75"))),
)
TRACK_MEMORY_ALLOW_NEW_EXTRA_SEC = max(0.0, float(os.environ.get("TRACK_MEMORY_ALLOW_NEW_EXTRA_SEC", "0.5")))
DEEPSORT_MAX_UNMATCHED_NUM = max(0, int(os.environ.get("Y8_DEEPSORT_MAX_UNMATCHED_NUM", "10")))
TRACK_MEMORY_DEEPSORT_HOLD_SEC = float(DEEPSORT_MAX_UNMATCHED_NUM) / VISION_EFFECTIVE_FPS
if TRACK_MEMORY_AUTO_FROM_DEEPSORT:
    TRACK_MEMORY_SEC = max(0.1, TRACK_MEMORY_DEEPSORT_HOLD_SEC * TRACK_MEMORY_DERIVED_MEMORY_RATIO)
    TRACK_MEMORY_ALLOW_NEW_TARGET_AFTER_SEC = max(
        TRACK_MEMORY_SEC,
        TRACK_MEMORY_DEEPSORT_HOLD_SEC + TRACK_MEMORY_ALLOW_NEW_EXTRA_SEC,
    )
else:
    TRACK_MEMORY_SEC = max(0.1, float(os.environ.get("TRACK_MEMORY_SEC", "1.5")))
    TRACK_MEMORY_ALLOW_NEW_TARGET_AFTER_SEC = max(
        TRACK_MEMORY_SEC,
        float(os.environ.get("TRACK_MEMORY_ALLOW_NEW_TARGET_AFTER_SEC", "3.0")),
    )
TRACK_MEMORY_REACQUIRE_SAME_ID = os.environ.get("TRACK_MEMORY_REACQUIRE_SAME_ID", "1").strip() != "0"
TRACK_MEMORY_REACQUIRE_BY_GEOMETRY = os.environ.get("TRACK_MEMORY_REACQUIRE_BY_GEOMETRY", "0").strip() != "0"
TRACK_MEMORY_REACQUIRE_CONFIRM_FRAMES = int(os.environ.get("TRACK_MEMORY_REACQUIRE_CONFIRM_FRAMES", "3"))
TRACK_MEMORY_MAX_CENTER_JUMP_RATIO = float(os.environ.get("TRACK_MEMORY_MAX_CENTER_JUMP_RATIO", "0.6"))
TRACK_MEMORY_MIN_AREA_SIMILARITY = float(os.environ.get("TRACK_MEMORY_MIN_AREA_SIMILARITY", "0.35"))
TRACK_MEMORY_MAX_ASPECT_DIFF = float(os.environ.get("TRACK_MEMORY_MAX_ASPECT_DIFF", "0.90"))
TRACK_MEMORY_LOG_ENABLE = os.environ.get("TRACK_MEMORY_LOG_ENABLE", "1").strip() != "0"

# Vision-gated mmwave: only use radar distance after a visual target is present.
VISION_MMWAVE_ANGLE_MARGIN_DEG = float(os.environ.get("VISION_MMWAVE_ANGLE_MARGIN_DEG", "12.0"))
VISION_MMWAVE_ANGLE_OFFSET_DEG = float(os.environ.get("VISION_MMWAVE_ANGLE_OFFSET_DEG", "0.0"))
VISION_MMWAVE_ANGLE_SIGN = float(os.environ.get("VISION_MMWAVE_ANGLE_SIGN", "1.0"))
VISION_MMWAVE_MATCH_MODE = os.environ.get("VISION_MMWAVE_MATCH_MODE", "nearest").strip().lower()
VISION_MMWAVE_ANGLE_TIE_MARGIN_DEG = max(0.0, float(os.environ.get("VISION_MMWAVE_ANGLE_TIE_MARGIN_DEG", "0.0")))
VISION_MMWAVE_DISTANCE_BIAS_M = float(os.environ.get("VISION_MMWAVE_DISTANCE_BIAS_M", os.environ.get("MMWAVE_DISTANCE_BIAS_M", "0.30")))
VISION_MMWAVE_MIN_DISTANCE_M = float(os.environ.get("VISION_MMWAVE_MIN_DISTANCE_M", os.environ.get("MMWAVE_MIN_DISTANCE_M", "0.02")))
VISION_MMWAVE_MAX_DISTANCE_M = float(os.environ.get("VISION_MMWAVE_MAX_DISTANCE_M", os.environ.get("MMWAVE_MAX_DISTANCE_M", "8.0")))
VISION_MMWAVE_MIN_OUTPUT_DISTANCE_M = float(os.environ.get("VISION_MMWAVE_MIN_OUTPUT_DISTANCE_M", os.environ.get("MMWAVE_MIN_OUTPUT_DISTANCE_M", "0.03")))
VISION_MMWAVE_HARD_STOP_TTL_SEC = float(os.environ.get("VISION_MMWAVE_HARD_STOP_TTL_SEC", "0.30"))
VISION_MMWAVE_LOG_EVERY_FRAMES = int(os.environ.get("VISION_MMWAVE_LOG_EVERY_FRAMES", "10"))
VISION_MMWAVE_LATENCY_SEC = max(0.0, float(os.environ.get("VISION_MMWAVE_LATENCY_SEC", "0.10")))
VISION_MMWAVE_CACHE_MAX_AGE_SEC = max(0.05, float(os.environ.get("VISION_MMWAVE_CACHE_MAX_AGE_SEC", "0.50")))
VISION_MMWAVE_USE_ASYNC_CACHE = os.environ.get("VISION_MMWAVE_USE_ASYNC_CACHE", "1").strip() != "0"
# 同一 ReID 仍可见时，毫米波短暂失配可沿用上次可信距离，但必须低速前进。
VISION_MMWAVE_UNMATCHED_HOLD_SEC = max(
    0.0, float(os.environ.get("VISION_MMWAVE_UNMATCHED_HOLD_SEC", "2.50"))
)
VISION_MMWAVE_UNMATCHED_HOLD_MIN_DISTANCE_M = max(
    0.0, float(os.environ.get("VISION_MMWAVE_UNMATCHED_HOLD_MIN_DISTANCE_M", "1.40"))
)
VISION_MMWAVE_FAR_MARGIN_START_M = max(
    0.0, float(os.environ.get("VISION_MMWAVE_FAR_MARGIN_START_M", "2.50"))
)
VISION_MMWAVE_FAR_ANGLE_MARGIN_EXTRA_DEG = max(
    0.0, float(os.environ.get("VISION_MMWAVE_FAR_ANGLE_MARGIN_EXTRA_DEG", "4.0"))
)
VISION_MMWAVE_HOLD_FORWARD_PERCENT = max(
    0, min(100, int(os.environ.get("VISION_MMWAVE_HOLD_FORWARD_PERCENT", "50")))
)
VISION_MMWAVE_HOLD_DECEL_STEP_PERCENT = max(
    1, min(100, int(os.environ.get("VISION_MMWAVE_HOLD_DECEL_STEP_PERCENT", "8")))
)
VISION_MMWAVE_HOLD_RECOVER_STEP_PERCENT = max(
    1, min(100, int(os.environ.get("VISION_MMWAVE_HOLD_RECOVER_STEP_PERCENT", "6")))
)
VISION_MMWAVE_FUSION_LOW_CONFIDENCE_FORWARD_PERCENT = max(
    0,
    min(100, int(os.environ.get("VISION_MMWAVE_FUSION_LOW_CONFIDENCE_FORWARD_PERCENT", "60"))),
)
# 摄像头/ReID 仍负责锁定具体的人；下面参数只约束毫米波点不能随意换绑。
VISION_MMWAVE_MAX_CENTER_ANGLE_DIFF_DEG = max(
    0.0, float(os.environ.get("VISION_MMWAVE_MAX_CENTER_ANGLE_DIFF_DEG", "26.0"))
)
VISION_MMWAVE_MAX_DISTANCE_JUMP_M = max(
    0.0, float(os.environ.get("VISION_MMWAVE_MAX_DISTANCE_JUMP_M", "0.60"))
)
VISION_MMWAVE_MAX_ANGLE_JUMP_DEG = max(
    0.0, float(os.environ.get("VISION_MMWAVE_MAX_ANGLE_JUMP_DEG", "15.0"))
)
VISION_MMWAVE_SWITCH_CONFIRM_FRAMES = max(
    1, int(os.environ.get("VISION_MMWAVE_SWITCH_CONFIRM_FRAMES", "3"))
)
VISION_MMWAVE_CONTINUITY_MEMORY_SEC = max(
    0.0, float(os.environ.get("VISION_MMWAVE_CONTINUITY_MEMORY_SEC", "2.50"))
)
VISION_MMWAVE_NEAR_DISTANCE_LOCK_M = max(
    0.0, float(os.environ.get("VISION_MMWAVE_NEAR_DISTANCE_LOCK_M", "2.00"))
)
VISION_MMWAVE_DISTANCE_SCORE_WEIGHT = max(
    0.0, float(os.environ.get("VISION_MMWAVE_DISTANCE_SCORE_WEIGHT", "6.0"))
)
VISION_MMWAVE_MOTION_MIN_ANGLE_DELTA_DEG = max(
    0.0, float(os.environ.get("VISION_MMWAVE_MOTION_MIN_ANGLE_DELTA_DEG", "3.0"))
)
VISION_MMWAVE_MOTION_HINT_TTL_SEC = max(
    0.0, float(os.environ.get("VISION_MMWAVE_MOTION_HINT_TTL_SEC", "0.80"))
)
VISION_MMWAVE_FUSION_ENABLE = os.environ.get("VISION_MMWAVE_FUSION_ENABLE", "1").strip() != "0"
VISION_MMWAVE_FUSION_RADAR_MEDIAN_WINDOW = max(
    1, int(os.environ.get("VISION_MMWAVE_FUSION_RADAR_MEDIAN_WINDOW", "3"))
)
VISION_MMWAVE_FUSION_VISUAL_WEIGHT = max(
    0.0, min(1.0, float(os.environ.get("VISION_MMWAVE_FUSION_VISUAL_WEIGHT", "0.80")))
)
VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M = max(
    0.0, float(os.environ.get("VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M", "0.60"))
)
VISION_MMWAVE_FUSION_ENCODER_MAX_STEP_M = max(
    0.0, float(os.environ.get("VISION_MMWAVE_FUSION_ENCODER_MAX_STEP_M", "0.12"))
)
VISION_MMWAVE_FUSION_BBOX_MIN_HEIGHT_RATIO = max(
    0.01, min(0.50, float(os.environ.get("VISION_MMWAVE_FUSION_BBOX_MIN_HEIGHT_RATIO", "0.08")))
)
VISION_MMWAVE_FUSION_BBOX_MAX_HEIGHT_RATIO = max(
    VISION_MMWAVE_FUSION_BBOX_MIN_HEIGHT_RATIO,
    min(0.99, float(os.environ.get("VISION_MMWAVE_FUSION_BBOX_MAX_HEIGHT_RATIO", "0.99"))),
)
VISION_MMWAVE_FUSION_RADAR_RECOVERY_ALPHA = max(
    0.05, min(1.0, float(os.environ.get("VISION_MMWAVE_FUSION_RADAR_RECOVERY_ALPHA", "0.35")))
)
VISION_MMWAVE_FUSION_MAX_DISTANCE_INCREASE_MPS = max(
    0.0, float(os.environ.get("VISION_MMWAVE_FUSION_MAX_DISTANCE_INCREASE_MPS", "2.0"))
)
VISION_MMWAVE_FUSION_MIN_CONFIDENCE = max(
    0.0, min(0.90, float(os.environ.get("VISION_MMWAVE_FUSION_MIN_CONFIDENCE", "0.40")))
)
_default_missing_forward = "0" if (VISION_MMWAVE_ENABLED or VISION_DEPTH_ENABLED) else "50"
DISTANCE_MISSING_FORWARD_PERCENT = int(os.environ.get("DISTANCE_MISSING_FORWARD_PERCENT", _default_missing_forward))
ULTRASONIC_MIN_DISTANCE_M = float(os.environ.get("ULTRASONIC_MIN_DISTANCE_M", "0.02"))
ULTRASONIC_MAX_DISTANCE_M = float(os.environ.get("ULTRASONIC_MAX_DISTANCE_M", "8.0"))
ULTRASONIC_FILTER_WINDOW = max(1, int(os.environ.get("ULTRASONIC_FILTER_WINDOW", "3")))
ULTRASONIC_TARGET_CONFIRM_FRAMES = max(1, int(os.environ.get("ULTRASONIC_TARGET_CONFIRM_FRAMES", "2")))
ULTRASONIC_BRAKE_CONFIRM_FRAMES = max(1, int(os.environ.get("ULTRASONIC_BRAKE_CONFIRM_FRAMES", "2")))
ULTRASONIC_HYSTERESIS_M = max(0.0, float(os.environ.get("ULTRASONIC_HYSTERESIS_M", "0.25")))
ULTRASONIC_IMMEDIATE_BRAKE_M = max(0.0, float(os.environ.get("ULTRASONIC_IMMEDIATE_BRAKE_M", "0.35")))
DISTANCE_RUNTIME_CONFIG = DistanceRuntimeConfig(
    distance_source=DISTANCE_SOURCE,
    vision_mmwave_source_aliases=frozenset(VISION_MMWAVE_SOURCE_ALIASES),
    module_mmwave_enable=MODULE_MMWAVE_ENABLE,
    module_ultrasonic_enable=MODULE_ULTRASONIC_ENABLE,
    module_astra_depth_enable=MODULE_ASTRA_DEPTH_ENABLE,
    vision_depth_source_aliases=frozenset(VISION_DEPTH_SOURCE_ALIASES),
    vision_hfov_deg=VISION_HFOV_DEG,
    vision_mmwave_angle_margin_deg=VISION_MMWAVE_ANGLE_MARGIN_DEG,
    vision_mmwave_angle_offset_deg=VISION_MMWAVE_ANGLE_OFFSET_DEG,
    vision_mmwave_angle_sign=VISION_MMWAVE_ANGLE_SIGN,
    vision_mmwave_match_mode=VISION_MMWAVE_MATCH_MODE,
    vision_mmwave_angle_tie_margin_deg=VISION_MMWAVE_ANGLE_TIE_MARGIN_DEG,
    vision_mmwave_distance_bias_m=VISION_MMWAVE_DISTANCE_BIAS_M,
    vision_mmwave_min_distance_m=VISION_MMWAVE_MIN_DISTANCE_M,
    vision_mmwave_max_distance_m=VISION_MMWAVE_MAX_DISTANCE_M,
    vision_mmwave_min_output_distance_m=VISION_MMWAVE_MIN_OUTPUT_DISTANCE_M,
    vision_mmwave_hard_stop_ttl_sec=VISION_MMWAVE_HARD_STOP_TTL_SEC,
    vision_mmwave_log_every_frames=VISION_MMWAVE_LOG_EVERY_FRAMES,
    vision_mmwave_unmatched_hold_sec=VISION_MMWAVE_UNMATCHED_HOLD_SEC,
    vision_mmwave_unmatched_hold_min_distance_m=VISION_MMWAVE_UNMATCHED_HOLD_MIN_DISTANCE_M,
    vision_mmwave_far_margin_start_m=VISION_MMWAVE_FAR_MARGIN_START_M,
    vision_mmwave_far_angle_margin_extra_deg=VISION_MMWAVE_FAR_ANGLE_MARGIN_EXTRA_DEG,
    vision_mmwave_latency_sec=VISION_MMWAVE_LATENCY_SEC,
    vision_mmwave_cache_max_age_sec=VISION_MMWAVE_CACHE_MAX_AGE_SEC,
    vision_mmwave_use_async_cache=VISION_MMWAVE_USE_ASYNC_CACHE,
    vision_mmwave_max_center_angle_diff_deg=VISION_MMWAVE_MAX_CENTER_ANGLE_DIFF_DEG,
    vision_mmwave_max_distance_jump_m=VISION_MMWAVE_MAX_DISTANCE_JUMP_M,
    vision_mmwave_max_angle_jump_deg=VISION_MMWAVE_MAX_ANGLE_JUMP_DEG,
    vision_mmwave_switch_confirm_frames=VISION_MMWAVE_SWITCH_CONFIRM_FRAMES,
    vision_mmwave_continuity_memory_sec=VISION_MMWAVE_CONTINUITY_MEMORY_SEC,
    vision_mmwave_near_distance_lock_m=VISION_MMWAVE_NEAR_DISTANCE_LOCK_M,
    vision_mmwave_distance_score_weight=VISION_MMWAVE_DISTANCE_SCORE_WEIGHT,
    vision_mmwave_motion_min_angle_delta_deg=VISION_MMWAVE_MOTION_MIN_ANGLE_DELTA_DEG,
    vision_mmwave_motion_hint_ttl_sec=VISION_MMWAVE_MOTION_HINT_TTL_SEC,
    vision_mmwave_fusion_enable=VISION_MMWAVE_FUSION_ENABLE,
    vision_mmwave_fusion_radar_median_window=VISION_MMWAVE_FUSION_RADAR_MEDIAN_WINDOW,
    vision_mmwave_fusion_visual_weight=VISION_MMWAVE_FUSION_VISUAL_WEIGHT,
    vision_mmwave_fusion_encoder_wheel_circumference_m=VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M,
    vision_mmwave_fusion_encoder_max_step_m=VISION_MMWAVE_FUSION_ENCODER_MAX_STEP_M,
    vision_mmwave_fusion_bbox_min_height_ratio=VISION_MMWAVE_FUSION_BBOX_MIN_HEIGHT_RATIO,
    vision_mmwave_fusion_bbox_max_height_ratio=VISION_MMWAVE_FUSION_BBOX_MAX_HEIGHT_RATIO,
    vision_mmwave_fusion_radar_recovery_alpha=VISION_MMWAVE_FUSION_RADAR_RECOVERY_ALPHA,
    vision_mmwave_fusion_max_distance_increase_mps=VISION_MMWAVE_FUSION_MAX_DISTANCE_INCREASE_MPS,
    vision_mmwave_fusion_min_confidence=VISION_MMWAVE_FUSION_MIN_CONFIDENCE,
    ultrasonic_min_distance_m=ULTRASONIC_MIN_DISTANCE_M,
    ultrasonic_max_distance_m=ULTRASONIC_MAX_DISTANCE_M,
    ultrasonic_filter_window=ULTRASONIC_FILTER_WINDOW,
    ultrasonic_target_confirm_frames=ULTRASONIC_TARGET_CONFIRM_FRAMES,
    ultrasonic_brake_confirm_frames=ULTRASONIC_BRAKE_CONFIRM_FRAMES,
    ultrasonic_hysteresis_m=ULTRASONIC_HYSTERESIS_M,
    ultrasonic_immediate_brake_m=ULTRASONIC_IMMEDIATE_BRAKE_M,
    vision_depth_max_distance_jump_m=ASTRA_DEPTH_MAX_DISTANCE_JUMP_M,
    vision_depth_jump_confirm_frames=max(
        3,
        ASTRA_DEPTH_JUMP_CONFIRM_FRAMES,
        ASTRA_DEPTH_NEAR_FAR_JUMP_CONFIRM_FRAMES,
    ),
)

# Motor RPM feedback is not connected to the action loop yet; keep it disabled.
MOTOR_FEEDBACK_POLL_INTERVAL_SEC = 3.0
ENABLE_MOTOR_RPM_FEEDBACK = False
BRAKE_HOLD_REFRESH_INTERVAL_SEC = float(os.environ.get("BRAKE_HOLD_REFRESH_INTERVAL_SEC", "0.10"))   # brake 保持态下重复下发间隔（秒）
DIRECT_STOP_REPEAT_INTERVAL_SEC = max(0.10, float(os.environ.get("DIRECT_STOP_REPEAT_INTERVAL_SEC", "0.75")))
# 前进速度：使用 doc 中“写入M1/M2运行状态和速度百分比”（0x0002/0x0003）
# golf_foll通信内容V6_20260304-6D加百分比调速.doc：正常前进调速不建议低于 MIN_FORWARD_PERCENT（默认 10%）；
# 需要停住时用 state=00(stop) / 03(brake)，避免长期用 1~9% 前进。
USE_PERCENT_SPEED = True
MIN_FORWARD_PERCENT = int(os.environ.get("MIN_FORWARD_PERCENT", "10"))  # 与协议一致：前进态百分比下限（0 表示停/滑行停，仍用 stop 而非“极低速前进”）
MAX_FORWARD_PERCENT = int(os.environ.get("MAX_FORWARD_PERCENT", "20"))  # 先限制为低速调试上限 20%
STEER_PERCENT_LIMIT = max(0, min(100, int(os.environ.get("STEER_PERCENT_LIMIT", "20"))))
MOTOR_PERCENT_LIMIT = max(0, min(100, int(os.environ.get("MOTOR_PERCENT_LIMIT", str(MAX_FORWARD_PERCENT)))))
FORWARD_SPEED_LE_1_3_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_1_3_PERCENT", "10"))
FORWARD_SPEED_LE_1_7_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_1_7_PERCENT", "12"))
FORWARD_SPEED_LE_2_1_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_2_1_PERCENT", "14"))
FORWARD_SPEED_LE_2_6_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_2_6_PERCENT", "16"))
FORWARD_SPEED_LE_3_2_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_3_2_PERCENT", "18"))
FORWARD_SPEED_LE_3_8_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_3_8_PERCENT", "19"))
FORWARD_SPEED_LE_4_5_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_4_5_PERCENT", "20"))
FORWARD_SPEED_FAR_PERCENT = int(os.environ.get("FORWARD_SPEED_FAR_PERCENT", "20"))
FORWARD_MIN_RPM = max(0, int(os.environ.get("FORWARD_MIN_RPM", "20")))
FORWARD_MAX_RPM = max(FORWARD_MIN_RPM, int(os.environ.get("FORWARD_MAX_RPM", "100")))
FORWARD_CURVE_MAX_DISTANCE_M = max(
    float(os.environ.get("TARGET_DISTANCE", "1.0")) + 0.01,
    float(os.environ.get("FORWARD_CURVE_MAX_DISTANCE_M", "5.0")),
)
FORWARD_CURVE_EXPONENT = max(0.10, min(3.0, float(os.environ.get("FORWARD_CURVE_EXPONENT", "0.75"))))
# < FOLLOW_BRAKE_DISTANCE_M / 前方IR 触发时的紧急停止：优先用 percent 通道 brake(state=0x03)。
USE_PERCENT_BRAKE_FOR_EMERGENCY = True
EMERGENCY_STOP_GEAR = 3  # percent brake 失败时退回档位 stop（如 s3）

# 前进 -> 转向：
# - False（默认）：直接发差速旋转，不先发 brake / 不滑行降速
# - True：先线性降速滑行再转（ROTATE_PREP_COAST_*），更柔和
ROTATE_PREP_COAST_ENABLE = False
ROTATE_PREP_COAST_STEPS = 6
ROTATE_PREP_COAST_TOTAL_SEC = 0.50
# 单次旋转脉冲结束后按 ROTATE_PULSE_STOP_MODE 停车，再进入视觉观察窗口。
ROTATE_END_USE_SOFT_STOP = False
ROTATE_PULSE_PAUSE_SEC = max(0.0, float(os.environ.get("ROTATE_PULSE_PAUSE_SEC", "0.00")))  # 搜索脉冲结束后以 TURN_ZERO 过渡，固定暂停仅作兼容项
ROTATE_PULSE_OBSERVE_MIN_FRAMES = max(
    0,
    int(os.environ.get("ROTATE_PULSE_OBSERVE_MIN_FRAMES", "0")),
)
ROTATE_CHAIN_MEMORY_SEC = 1.00  # 旋转记忆窗口（秒）：该窗口内再次旋转按 CHAIN 力度，避免偶发大角度
ROTATE_PULSE_BRAKE_ENABLE = os.environ.get("ROTATE_PULSE_BRAKE_ENABLE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ROTATE_PULSE_SETTLE_ENABLE = os.environ.get("ROTATE_PULSE_SETTLE_ENABLE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ROTATE_PULSE_SETTLE_QUIET_SEC = max(
    0.05,
    float(os.environ.get("ROTATE_PULSE_SETTLE_QUIET_SEC", "0.05")),
)
ROTATE_PULSE_SETTLE_TIMEOUT_SEC = max(
    ROTATE_PULSE_SETTLE_QUIET_SEC,
    float(os.environ.get("ROTATE_PULSE_SETTLE_TIMEOUT_SEC", "0.25")),
)
ROTATE_PULSE_SETTLE_FEEDBACK_STALE_SEC = max(
    0.05,
    float(os.environ.get("ROTATE_PULSE_SETTLE_FEEDBACK_STALE_SEC", "0.30")),
)
ROTATE_PULSE_SETTLE_MAX_WHEEL_RPM = max(
    0,
    int(os.environ.get("ROTATE_PULSE_SETTLE_MAX_WHEEL_RPM", "1")),
)
ROTATE_PULSE_SETTLE_MAX_YAW_RATE_DPS = max(
    0.0,
    float(os.environ.get("ROTATE_PULSE_SETTLE_MAX_YAW_RATE_DPS", "3.0")),
)
ROTATE_PULSE_TRANSITION_RPM = max(
    0,
    int(os.environ.get("ROTATE_PULSE_TRANSITION_RPM", "1")),
)
ROTATE_PULSE_STOP_MODE = os.environ.get("ROTATE_PULSE_STOP_MODE", "zero").strip().lower()
if ROTATE_PULSE_STOP_MODE in {"target_zero", "zero_target", "coast"}:
    ROTATE_PULSE_STOP_MODE = "zero"
elif ROTATE_PULSE_STOP_MODE not in {"zero", "brake"}:
    ROTATE_PULSE_STOP_MODE = "zero"
ROTATE_HOLD_STALE_SEC = max(
    0.20,
    float(os.environ.get("ROTATE_HOLD_STALE_SEC", "0.80")),
)
ROTATION_ONLY_YAW_PULSE_RPM = max(
    1, int(os.environ.get("ROTATION_ONLY_YAW_PULSE_RPM", "30"))
)
ROTATION_ONLY_YAW_PULSE_MIN_SEC = max(
    0.04, float(os.environ.get("ROTATION_ONLY_YAW_PULSE_MIN_SEC", "0.18"))
)
ROTATION_ONLY_YAW_PULSE_MAX_SEC = max(
    ROTATION_ONLY_YAW_PULSE_MIN_SEC,
    float(os.environ.get("ROTATION_ONLY_YAW_PULSE_MAX_SEC", "0.32")),
)
ROTATION_ONLY_YAW_BRAKE_SEC = max(
    0.04, float(os.environ.get("ROTATION_ONLY_YAW_BRAKE_SEC", "0.18"))
)
ROTATION_ONLY_YAW_RESPONSE_DPS = max(
    0.0, float(os.environ.get("ROTATION_ONLY_YAW_RESPONSE_DPS", "4.0"))
)
ROTATION_ONLY_YAW_ZERO_GAP_SEC = max(
    0.0, float(os.environ.get("ROTATION_ONLY_YAW_ZERO_GAP_SEC", "0.03"))
)
ROTATE_PULSE_ACTIVE_BRAKE_ENABLE = os.environ.get(
    "ROTATE_PULSE_ACTIVE_BRAKE_ENABLE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
ROTATE_PULSE_ACTIVE_BRAKE_RPM = max(
    1, int(os.environ.get("ROTATE_PULSE_ACTIVE_BRAKE_RPM", "30"))
)
ROTATE_PULSE_ACTIVE_BRAKE_SEC = max(
    0.04, float(os.environ.get("ROTATE_PULSE_ACTIVE_BRAKE_SEC", "0.18"))
)

# 旋转（差速）参数：一个轮正转、另一个轮反转（百分比）
# 直行后首轮转向用较高力度；连续转向（上一动已是旋转，或刚结束一次旋转后的下一拍）用较低力度——轮姿不同所需力度不同
ROTATE_TURN_PERCENT_FROM_FORWARD = int(os.environ.get("ROTATE_TURN_PERCENT_FROM_FORWARD", "15"))  # 上一动不是旋转（如前进、首次、长时间停后）
ROTATE_TURN_PERCENT_CHAIN = int(os.environ.get("ROTATE_TURN_PERCENT_CHAIN", "10"))          # 上一动是旋转（左/右切换）或刚结束旋转后的下一拍转向
VISIBLE_STEER_FINE_INNER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_FINE_INNER_RATIO_PERCENT", "90"))
VISIBLE_STEER_FINE_OUTER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_FINE_OUTER_RATIO_PERCENT", "100"))
VISIBLE_STEER_INNER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_INNER_RATIO_PERCENT", "80"))
VISIBLE_STEER_OUTER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_OUTER_RATIO_PERCENT", "105"))
VISIBLE_STEER_STRONG_INNER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_STRONG_INNER_RATIO_PERCENT", "70"))
VISIBLE_STEER_STRONG_OUTER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_STRONG_OUTER_RATIO_PERCENT", "110"))
VISIBLE_STEER_STRONG_MARGIN_RATIO = float(os.environ.get("VISIBLE_STEER_STRONG_MARGIN_RATIO", "0.20"))
# 可见目标最近几帧的横向轨迹用于提前纠偏：人还在中心区内但正快速外移时，不等贴边就开始转向。
VISIBLE_MOTION_HISTORY_FRAMES = max(2, int(os.environ.get("VISIBLE_MOTION_HISTORY_FRAMES", "4")))
VISIBLE_MOTION_LOOKBACK_SEC = max(0.04, float(os.environ.get("VISIBLE_MOTION_LOOKBACK_SEC", "0.10")))
VISIBLE_MOTION_RATE_FILTER_ALPHA = max(0.05, min(1.0, float(os.environ.get("VISIBLE_MOTION_RATE_FILTER_ALPHA", "0.70"))))
VISIBLE_MOTION_MIN_RATIO = max(0.0, float(os.environ.get("VISIBLE_MOTION_MIN_RATIO", "0.03")))
VISIBLE_MOTION_PROJECTION_GAIN = max(0.0, float(os.environ.get("VISIBLE_MOTION_PROJECTION_GAIN", "0.35")))
VISIBLE_MOTION_STRONG_RATIO = max(0.0, float(os.environ.get("VISIBLE_MOTION_STRONG_RATIO", "0.10")))
VISIBLE_STEERING_PID_ENABLE = os.environ.get("VISIBLE_STEERING_PID_ENABLE", "0").strip() != "0"
VISIBLE_STEERING_PID_CAMERA_HFOV_DEG = float(os.environ.get("VISIBLE_STEERING_PID_CAMERA_HFOV_DEG", "90.0"))
VISIBLE_STEERING_PID_CAMERA_LATENCY_SEC = float(os.environ.get("VISIBLE_STEERING_PID_CAMERA_LATENCY_SEC", "0.13"))
VISIBLE_STEERING_PID_DEADBAND_DEG = float(os.environ.get("VISIBLE_STEERING_PID_DEADBAND_DEG", "1.2"))
VISIBLE_STEERING_PID_OUTER_KP_PER_SEC = float(os.environ.get("VISIBLE_STEERING_PID_OUTER_KP_PER_SEC", "1.55"))
VISIBLE_STEERING_PID_OUTER_KD_SEC = float(os.environ.get("VISIBLE_STEERING_PID_OUTER_KD_SEC", "0.06"))
VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_GAIN = max(0.0, float(os.environ.get("VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_GAIN", "0.30")))
VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_MAX_DPS = max(0.0, float(os.environ.get("VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_MAX_DPS", "12.0")))
VISIBLE_STEERING_PID_TARGET_SPEED_MATCH_MAX_CLOSING_DPS = max(
    0.0,
    float(os.environ.get("VISIBLE_STEERING_PID_TARGET_SPEED_MATCH_MAX_CLOSING_DPS", "0.0")),
)
VISIBLE_STEERING_PID_MAX_YAW_RATE_DPS = float(os.environ.get("VISIBLE_STEERING_PID_MAX_YAW_RATE_DPS", "34.0"))
VISIBLE_STEERING_PID_RATE_KP_RPM_PER_DPS = float(os.environ.get("VISIBLE_STEERING_PID_RATE_KP_RPM_PER_DPS", "0.32"))
VISIBLE_STEERING_PID_RATE_KI_RPM_PER_DEG = float(os.environ.get("VISIBLE_STEERING_PID_RATE_KI_RPM_PER_DEG", "0.015"))
VISIBLE_STEERING_PID_INTEGRAL_LIMIT_DEG = float(os.environ.get("VISIBLE_STEERING_PID_INTEGRAL_LIMIT_DEG", "25.0"))
VISIBLE_STEERING_PID_MAX_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_MAX_CORRECTION_RPM", "12.0"))
VISIBLE_STEERING_PID_DYNAMIC_SMALL_ERROR_DEG = float(os.environ.get("VISIBLE_STEERING_PID_DYNAMIC_SMALL_ERROR_DEG", "6.0"))
VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_DEG = float(os.environ.get("VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_DEG", "18.0"))
VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_YAW_RATE_DPS = float(os.environ.get("VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_YAW_RATE_DPS", "18.0"))
VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_CORRECTION_RPM", "5.0"))
VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_BASE_CAP_RPM = float(os.environ.get("VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_BASE_CAP_RPM", "35.0"))
VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_THRESHOLD_DPS = float(os.environ.get("VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_THRESHOLD_DPS", "6.0"))
VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_BOOST_RPM = float(os.environ.get("VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_BOOST_RPM", "6.0"))
VISIBLE_STEERING_PID_BRAKING_MAX_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_BRAKING_MAX_CORRECTION_RPM", "6.0"))
VISIBLE_STEERING_PID_FAST_COUNTERSTEER_MAX_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_FAST_COUNTERSTEER_MAX_CORRECTION_RPM", "10.0"))
VISIBLE_STEERING_PID_FAST_COUNTERSTEER_GAIN_RPM_PER_DPS = float(os.environ.get("VISIBLE_STEERING_PID_FAST_COUNTERSTEER_GAIN_RPM_PER_DPS", "0.08"))
VISIBLE_STEERING_PID_SAME_DIRECTION_OVERSPEED_THRESHOLD_DPS = float(os.environ.get("VISIBLE_STEERING_PID_SAME_DIRECTION_OVERSPEED_THRESHOLD_DPS", "8.0"))
VISIBLE_STEERING_PID_SAME_DIRECTION_OVERSPEED_BRAKE_GAIN_RPM_PER_DPS = float(os.environ.get("VISIBLE_STEERING_PID_SAME_DIRECTION_OVERSPEED_BRAKE_GAIN_RPM_PER_DPS", "0.25"))
VISIBLE_STEERING_PID_VISUAL_DIRECTION_GUARD_ENABLE = (
    os.environ.get("VISIBLE_STEERING_PID_VISUAL_DIRECTION_GUARD_ENABLE", "0")
    .strip()
    .lower()
    not in {"0", "false", "no", "off"}
)
VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_DECEL_DPS2 = max(
    0.0,
    float(os.environ.get("VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_DECEL_DPS2", "0.0")),
)
VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_MARGIN_DEG = max(
    0.0,
    float(os.environ.get("VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_MARGIN_DEG", "0.0")),
)
VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_RESPONSE_SEC = max(
    0.0,
    float(os.environ.get("VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_RESPONSE_SEC", "0.0")),
)
VISIBLE_STEERING_PID_MIN_EFFECTIVE_ERROR_DEG = float(os.environ.get("VISIBLE_STEERING_PID_MIN_EFFECTIVE_ERROR_DEG", "0.0"))
VISIBLE_STEERING_PID_MIN_EFFECTIVE_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_MIN_EFFECTIVE_CORRECTION_RPM", "0.0"))
VISIBLE_STEERING_PID_MECHANICAL_TIER2_ERROR_DEG = float(os.environ.get("VISIBLE_STEERING_PID_MECHANICAL_TIER2_ERROR_DEG", "0.0"))
VISIBLE_STEERING_PID_MECHANICAL_TIER2_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_MECHANICAL_TIER2_CORRECTION_RPM", "0.0"))
VISIBLE_STEERING_PID_MECHANICAL_TIER3_ERROR_DEG = float(os.environ.get("VISIBLE_STEERING_PID_MECHANICAL_TIER3_ERROR_DEG", "0.0"))
VISIBLE_STEERING_PID_MECHANICAL_TIER3_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_MECHANICAL_TIER3_CORRECTION_RPM", "0.0"))
VISIBLE_STEERING_PID_MECHANICAL_FLOOR_RELEASE_RATIO = float(os.environ.get("VISIBLE_STEERING_PID_MECHANICAL_FLOOR_RELEASE_RATIO", "0.0"))
VISIBLE_STEERING_PID_STARTUP_KICK_ERROR_DEG = float(os.environ.get("VISIBLE_STEERING_PID_STARTUP_KICK_ERROR_DEG", "0.0"))
VISIBLE_STEERING_PID_STARTUP_KICK_RPM = float(os.environ.get("VISIBLE_STEERING_PID_STARTUP_KICK_RPM", "0.0"))
VISIBLE_STEERING_PID_STARTUP_KICK_MAX_SEC = float(os.environ.get("VISIBLE_STEERING_PID_STARTUP_KICK_MAX_SEC", "0.0"))
VISIBLE_STEERING_PID_STARTUP_KICK_RELEASE_YAW_RATE_DPS = float(os.environ.get("VISIBLE_STEERING_PID_STARTUP_KICK_RELEASE_YAW_RATE_DPS", "0.0"))
VISIBLE_STEERING_PID_ACTIVE_BRAKE_YAW_THRESHOLD_DPS = float(os.environ.get("VISIBLE_STEERING_PID_ACTIVE_BRAKE_YAW_THRESHOLD_DPS", "0.0"))
VISIBLE_STEERING_PID_ACTIVE_BRAKE_MIN_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_ACTIVE_BRAKE_MIN_CORRECTION_RPM", "0.0"))
VISIBLE_STEERING_PID_EDGE_BOOST_START_ERROR_DEG = float(os.environ.get("VISIBLE_STEERING_PID_EDGE_BOOST_START_ERROR_DEG", "18.0"))
VISIBLE_STEERING_PID_AGGRESSIVE_INNER_WHEEL_MARGIN_RPM = float(os.environ.get("VISIBLE_STEERING_PID_AGGRESSIVE_INNER_WHEEL_MARGIN_RPM", "0.0"))
VISIBLE_STEERING_PID_FALLBACK_MAX_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_FALLBACK_MAX_CORRECTION_RPM", "12.0"))
VISIBLE_STEERING_PID_LOST_HOLD_MAX_CORRECTION_RPM = max(0, int(os.environ.get("VISIBLE_STEERING_PID_LOST_HOLD_MAX_CORRECTION_RPM", "8")))
VISIBLE_STEERING_PID_LEFT_BODY_DEG_PER_ENCODER_DEG = float(os.environ.get("VISIBLE_STEERING_PID_LEFT_BODY_DEG_PER_ENCODER_DEG", "0.5225"))
VISIBLE_STEERING_PID_RIGHT_BODY_DEG_PER_ENCODER_DEG = float(os.environ.get("VISIBLE_STEERING_PID_RIGHT_BODY_DEG_PER_ENCODER_DEG", "0.5424"))
VISIBLE_STEERING_PID_FEEDBACK_POLL_INTERVAL_SEC = max(0.05, float(os.environ.get("VISIBLE_STEERING_PID_FEEDBACK_POLL_INTERVAL_SEC", "0.10")))
VISIBLE_STEERING_PID_FEEDBACK_LOG_INTERVAL_SEC = max(0.10, float(os.environ.get("VISIBLE_STEERING_PID_FEEDBACK_LOG_INTERVAL_SEC", "0.50")))
VISIBLE_STEERING_PID_FEEDBACK_MEDIAN_WINDOW = max(
    1,
    min(9, int(os.environ.get("VISIBLE_STEERING_PID_FEEDBACK_MEDIAN_WINDOW", "3"))),
)
if VISIBLE_STEERING_PID_FEEDBACK_MEDIAN_WINDOW % 2 == 0:
    VISIBLE_STEERING_PID_FEEDBACK_MEDIAN_WINDOW += 1
VISIBLE_STEERING_PID_FEEDBACK_STALE_SEC = max(0.05, float(os.environ.get("VISIBLE_STEERING_PID_FEEDBACK_STALE_SEC", "0.30")))
VISIBLE_STEERING_PID_ERROR_FILTER_ALPHA = float(os.environ.get("VISIBLE_STEERING_PID_ERROR_FILTER_ALPHA", "0.60"))
VISIBLE_STEERING_PID_DERIVATIVE_FILTER_ALPHA = float(os.environ.get("VISIBLE_STEERING_PID_DERIVATIVE_FILTER_ALPHA", "0.25"))
VISIBLE_STEERING_PID_FALLBACK_BASE_RPM = max(1, int(os.environ.get("VISIBLE_STEERING_PID_FALLBACK_BASE_RPM", "15")))
PID_TRACE_INTERVAL_SEC = max(
    0.05,
    float(os.environ.get("VISIBLE_STEERING_PID_TRACE_INTERVAL_SEC", "0.20")),
)
LATERAL_INTENT_CONTROL_ENABLE = (
    os.environ.get("LATERAL_INTENT_CONTROL_ENABLE", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)
LATERAL_INTENT_CONTROL_RATE_HZ = max(
    10.0, min(50.0, float(os.environ.get("LATERAL_INTENT_CONTROL_RATE_HZ", "30.0")))
)
LATERAL_INTENT_TTL_SEC = max(
    0.08, min(0.30, float(os.environ.get("LATERAL_INTENT_TTL_SEC", "0.15")))
)
LATERAL_INTENT_MAX_PROJECTION_SEC = max(
    0.0,
    min(0.25, float(os.environ.get("LATERAL_INTENT_MAX_PROJECTION_SEC", "0.12"))),
)
LATERAL_INTENT_MAX_PROJECTION_RATIO = max(
    0.0,
    min(0.25, float(os.environ.get("LATERAL_INTENT_MAX_PROJECTION_RATIO", "0.10"))),
)
LATERAL_INTENT_RISE_RPM_PER_SEC = max(
    1.0, float(os.environ.get("LATERAL_INTENT_RISE_RPM_PER_SEC", "120.0"))
)
LATERAL_INTENT_BRAKE_RPM_PER_SEC = max(
    LATERAL_INTENT_RISE_RPM_PER_SEC,
    float(os.environ.get("LATERAL_INTENT_BRAKE_RPM_PER_SEC", "220.0")),
)
LATERAL_INTENT_MOTOR_PUBLISH_INTERVAL_SEC = max(
    0.03,
    min(
        0.15,
        float(os.environ.get("LATERAL_INTENT_MOTOR_PUBLISH_INTERVAL_SEC", "0.05")),
    ),
)
LATERAL_INTENT_LOG_INTERVAL_SEC = max(
    0.10, float(os.environ.get("LATERAL_INTENT_LOG_INTERVAL_SEC", "0.25"))
)
PARKED_RECENTER_MIN_RPM = max(1, int(os.environ.get("PARKED_RECENTER_MIN_RPM", "2")))
PARKED_RECENTER_MAX_RPM = max(
    PARKED_RECENTER_MIN_RPM,
    int(os.environ.get("PARKED_RECENTER_MAX_RPM", "10")),
)
# M1/M2 对应左右轮的映射（若方向不对可切换）
M1_IS_LEFT_WHEEL = os.environ.get("M1_IS_LEFT_WHEEL", "0").strip() in {"1", "true", "yes", "on"}
MSSD_MOTOR_CONFIG = MssdMotorConfig(
    port=MOTOR_RS485_PORT,
    slave_id=MOTOR_RS485_SLAVE_ID,
    baudrate=MOTOR_RS485_BAUDRATE,
    timeout=MOTOR_RS485_TIMEOUT,
    lib_dir=MOTOR_RS485_LIB_DIR,
    max_target=MOTOR_RS485_MAX_TARGET,
    percent_limit=MOTOR_PERCENT_LIMIT,
    left_sign=MOTOR_LEFT_SIGN,
    right_sign=MOTOR_RIGHT_SIGN,
    forward_target_sign=MOTOR_FORWARD_TARGET_SIGN,
    m1_is_left_wheel=M1_IS_LEFT_WHEEL,
    exit_parking_mode_on_arm=MOTOR_EXIT_PARKING_MODE_ON_ARM,
    stop_mode=MOTOR_RS485_STOP_MODE,
    stop_zero_delay_sec=MOTOR_RS485_STOP_ZERO_DELAY_SEC,
    parking_current_a=MOTOR_PARKING_CURRENT_A,
)

# 动作参数
ROTATE_DURATION = float(os.environ.get("ROTATE_DURATION", "0.10"))  # 搜索短脉冲时长；保持模式下仅作为日志/调试窗口
# 主循环不再固定等待 60ms。摄像头/推理本身会阻塞到下一帧；额外 sleep
# 只会把约 70ms 的视觉控制周期拖到约 130ms。需要限频时可临时通过环境变量设置。
PROCESS_FRAME_INTERVAL = max(
    0.0,
    float(os.environ.get("PROCESS_FRAME_INTERVAL", "0.0")),
)

# Distance uses Astra Depth at runtime; mmWave code remains available but disabled.
# Target distance feeds longitudinal speed/reverse; IR owns explicit parking.
TARGET_DISTANCE = float(os.environ.get("TARGET_DISTANCE", "1.5"))
DISTANCE_PARKING_ENABLE = os.environ.get("DISTANCE_PARKING_ENABLE", "0").strip() != "0"
# 目标距离停车释放回差：默认与目标距离相同，并要求连续帧和持续时间同时满足。
TARGET_DISTANCE_RELEASE_M = max(
    TARGET_DISTANCE,
    float(os.environ.get("TARGET_DISTANCE_RELEASE_M", str(TARGET_DISTANCE))),
)
TARGET_DISTANCE_RELEASE_HOLD_SEC = max(
    0.0,
    float(os.environ.get("TARGET_DISTANCE_RELEASE_HOLD_SEC", "0.50")),
)
TARGET_DISTANCE_RELEASE_CONFIRM_FRAMES = max(
    1,
    int(os.environ.get("TARGET_DISTANCE_RELEASE_CONFIRM_FRAMES", "3")),
)
TARGET_DISTANCE_RELEASE_VISUAL_SHRINK_RATIO = max(
    0.05,
    min(1.0, float(os.environ.get("TARGET_DISTANCE_RELEASE_VISUAL_SHRINK_RATIO", "0.90"))),
)
TARGET_DISTANCE_RELEASE_VISUAL_EDGE_MARGIN_RATIO = max(
    0.0,
    min(0.20, float(os.environ.get("TARGET_DISTANCE_RELEASE_VISUAL_EDGE_MARGIN_RATIO", "0.02"))),
)
# 小于此距离（米）发 brake / 硬停 / 边缘区「太近不转」（统一阈值）
FOLLOW_BRAKE_DISTANCE_M = float(os.environ.get("FOLLOW_BRAKE_DISTANCE_M", "0.5"))
# 受控倒车仅用于目标持续靠近时恢复 1.5m 间距。它只接受实时稳定毫米波，
# radar hold、视觉估距、编码器推算、目标丢失或任一路安全传感器均会立即停车。
FOLLOW_REVERSE_ENABLE = os.environ.get("FOLLOW_REVERSE_ENABLE", "0").strip() != "0"
FOLLOW_REVERSE_STOP_DISTANCE_M = max(
    FOLLOW_BRAKE_DISTANCE_M + 0.06,
    float(os.environ.get("FOLLOW_REVERSE_STOP_DISTANCE_M", "1.45")),
)
FOLLOW_REVERSE_START_DISTANCE_M = max(
    FOLLOW_BRAKE_DISTANCE_M + 0.05,
    min(
        FOLLOW_REVERSE_STOP_DISTANCE_M - 0.01,
        float(os.environ.get("FOLLOW_REVERSE_START_DISTANCE_M", str(TARGET_DISTANCE))),
    ),
)
FOLLOW_REVERSE_IMMEDIATE_DISTANCE_M = max(
    FOLLOW_BRAKE_DISTANCE_M + 0.05,
    min(
        FOLLOW_REVERSE_START_DISTANCE_M,
        float(os.environ.get("FOLLOW_REVERSE_IMMEDIATE_DISTANCE_M", "1.35")),
    ),
)
FOLLOW_REVERSE_FULL_SPEED_DISTANCE_M = max(
    FOLLOW_BRAKE_DISTANCE_M + 0.05,
    min(
        FOLLOW_REVERSE_START_DISTANCE_M - 0.01,
        float(os.environ.get("FOLLOW_REVERSE_FULL_SPEED_DISTANCE_M", "1.0")),
    ),
)
FOLLOW_REVERSE_MIN_RPM = max(1, int(os.environ.get("FOLLOW_REVERSE_MIN_RPM", "20")))
FOLLOW_REVERSE_MAX_RPM = max(
    FOLLOW_REVERSE_MIN_RPM,
    int(os.environ.get("FOLLOW_REVERSE_MAX_RPM", "100")),
)
FOLLOW_REVERSE_FEEDFORWARD_FLOOR_RPM = max(
    FOLLOW_REVERSE_MIN_RPM,
    int(os.environ.get("FOLLOW_REVERSE_FEEDFORWARD_FLOOR_RPM", "30")),
)
FOLLOW_REVERSE_FEEDFORWARD_GAIN_RPM_PER_M_S = max(
    0.0,
    float(os.environ.get("FOLLOW_REVERSE_FEEDFORWARD_GAIN_RPM_PER_M_S", "40.0")),
)
# 先把实车倒车限制在 60 RPM；绝对上限仍由 reverse_max_rpm=100 保护。
FOLLOW_REVERSE_RUNTIME_CAP_RPM = max(
    FOLLOW_REVERSE_MIN_RPM,
    min(
        FOLLOW_REVERSE_MAX_RPM,
        int(os.environ.get("FOLLOW_REVERSE_RUNTIME_CAP_RPM", "60")),
    ),
)
FOLLOW_REVERSE_APPROACH_SPEED_FILTER_ALPHA = max(
    0.05,
    min(
        1.0,
        float(os.environ.get("FOLLOW_REVERSE_APPROACH_SPEED_FILTER_ALPHA", "0.50")),
    ),
)
FOLLOW_REVERSE_MIN_APPROACH_DELTA_M = max(
    0.0,
    float(os.environ.get("FOLLOW_REVERSE_MIN_APPROACH_DELTA_M", "0.03")),
)
FOLLOW_REVERSE_CONFIRM_FRAMES = max(1, int(os.environ.get("FOLLOW_REVERSE_CONFIRM_FRAMES", "2")))
FOLLOW_REVERSE_RADAR_MAX_AGE_SEC = max(
    0.01,
    float(os.environ.get("FOLLOW_REVERSE_RADAR_MAX_AGE_SEC", "0.25")),
)
FOLLOW_REVERSE_DISTANCE_MISSING_HOLD_SEC = max(
    0.0,
    float(os.environ.get("FOLLOW_REVERSE_DISTANCE_MISSING_HOLD_SEC", "0.35")),
)
FOLLOW_REVERSE_VISUAL_GUARD_AREA_RATIO = max(
    0.05,
    min(1.0, float(os.environ.get("FOLLOW_REVERSE_VISUAL_GUARD_AREA_RATIO", "0.45"))),
)
FOLLOW_REVERSE_VISUAL_GUARD_HEIGHT_RATIO = max(
    0.05,
    min(1.0, float(os.environ.get("FOLLOW_REVERSE_VISUAL_GUARD_HEIGHT_RATIO", "0.92"))),
)
FOLLOW_REVERSE_VISUAL_GUARD_GROWTH_RATIO = max(
    1.01,
    float(os.environ.get("FOLLOW_REVERSE_VISUAL_GUARD_GROWTH_RATIO", "1.18")),
)
FOLLOW_REVERSE_VISUAL_GUARD_GROWTH_MIN_AREA_RATIO = max(
    0.01,
    min(
        1.0,
        float(os.environ.get("FOLLOW_REVERSE_VISUAL_GUARD_GROWTH_MIN_AREA_RATIO", "0.20")),
    ),
)
FOLLOW_REVERSE_VISUAL_GUARD_MAX_DISTANCE_M = max(
    FOLLOW_REVERSE_STOP_DISTANCE_M,
    float(os.environ.get("FOLLOW_REVERSE_VISUAL_GUARD_MAX_DISTANCE_M", "1.60")),
)
FOLLOW_REVERSE_VISUAL_GUARD_RPM = max(
    1,
    min(
        FOLLOW_REVERSE_RUNTIME_CAP_RPM,
        int(os.environ.get("FOLLOW_REVERSE_VISUAL_GUARD_RPM", "30")),
    ),
)
FOLLOW_FORWARD_START_DISTANCE_M = max(
    TARGET_DISTANCE + 0.01,
    float(os.environ.get("FOLLOW_FORWARD_START_DISTANCE_M", "1.80")),
)
FOLLOW_FORWARD_STOP_DISTANCE_M = max(
    TARGET_DISTANCE,
    min(
        FOLLOW_FORWARD_START_DISTANCE_M - 0.01,
        float(os.environ.get("FOLLOW_FORWARD_STOP_DISTANCE_M", "1.65")),
    ),
)
FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_ENABLE = (
    os.environ.get("FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_ENABLE", "1").strip() != "0"
)
FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_DISTANCE_M = max(
    TARGET_DISTANCE + 0.01,
    float(os.environ.get("FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_DISTANCE_M", "1.80")),
)
FOLLOW_NEAR_DISTANCE_ROTATION_ONLY_MAX_RPM = max(
    1,
    int(os.environ.get("FOLLOW_NEAR_DISTANCE_ROTATION_ONLY_MAX_RPM", "12")),
)
MMWAVE_OBSTACLE_THRESHOLD = float(os.environ.get("MMWAVE_OBSTACLE_THRESHOLD", "1.0"))  # 毫米波障碍物阈值（米），前方 1.0 m 内有障碍且未检测到人则停
ULTRASONIC_OBSTACLE_THRESHOLD = MMWAVE_OBSTACLE_THRESHOLD  # 保留旧变量名，后续超声融合时兼容
FORWARD_1M_DISTANCE = 1.0  # 绕障前进距离（米）

# 纵向距离串级外环：Depth 实际距离误差 -> 目标 RPM；LZ30EMA 负责内层速度闭环。
DISTANCE_PID_ENABLE = os.environ.get("DISTANCE_PID_ENABLE", "1").strip() != "0"
DISTANCE_PID_KP_RPM_PER_M = float(os.environ.get("DISTANCE_PID_KP_RPM_PER_M", "22.0"))
DISTANCE_PID_KI_RPM_PER_M_S = float(os.environ.get("DISTANCE_PID_KI_RPM_PER_M_S", "1.5"))
DISTANCE_PID_KD_RPM_S_PER_M = float(os.environ.get("DISTANCE_PID_KD_RPM_S_PER_M", "6.0"))
DISTANCE_PID_INTEGRAL_LIMIT_M_S = max(
    0.0,
    float(os.environ.get("DISTANCE_PID_INTEGRAL_LIMIT_M_S", "1.5")),
)
DISTANCE_PID_DEADBAND_M = max(
    0.0,
    float(os.environ.get("DISTANCE_PID_DEADBAND_M", "0.005")),
)
DISTANCE_PID_DERIVATIVE_FILTER_ALPHA = max(
    0.0,
    min(1.0, float(os.environ.get("DISTANCE_PID_DERIVATIVE_FILTER_ALPHA", "0.12"))),
)
DISTANCE_PID_MAX_MEASUREMENT_JUMP_M = max(
    0.0,
    float(os.environ.get("DISTANCE_PID_MAX_MEASUREMENT_JUMP_M", "0.80")),
)
DISTANCE_PID_OUTPUT_RISE_RPM_PER_SEC = max(
    0.0,
    float(os.environ.get("DISTANCE_PID_OUTPUT_RISE_RPM_PER_SEC", "180.0")),
)
DISTANCE_PID_OUTPUT_FALL_RPM_PER_SEC = max(
    0.0,
    float(os.environ.get("DISTANCE_PID_OUTPUT_FALL_RPM_PER_SEC", "300.0")),
)

# ==================== 常量定义 ====================
PERSON_CLASS_ID = int(os.environ.get("PERSON_CLASS_ID", "0"))  # YOLO中人的类别ID

# 沙坑/水坑安全停止：
# - split：当前双模型方案，主流程跑人员模型，后台另起沙坑/水坑模型。
# - merged：后续合并成一个模型后，直接从主 YOLO 输出中按危险类别判断。
# 默认关闭，避免普通 person/COCO 模型中的 class_id=0(person) 被误判为沙坑。
BUNKER_AVOID_ENABLE = os.environ.get("BUNKER_AVOID_ENABLE", "0").strip() != "0"
BUNKER_DETECT_MODE = os.environ.get("BUNKER_DETECT_MODE", "").strip().lower()
if BUNKER_AVOID_ENABLE and not BUNKER_DETECT_MODE:
    BUNKER_DETECT_MODE = "split"
elif not BUNKER_DETECT_MODE:
    BUNKER_DETECT_MODE = "merged"
BUNKER_WORKDIR = os.path.abspath(
    os.environ.get("BUNKER_WORKDIR", os.path.join(SCRIPT_DIR, "yolov8-track")).strip()
)
BUNKER_MODEL_PATH = os.environ.get(
    "BUNKER_MODEL_PATH",
    os.path.join(BUNKER_WORKDIR, "shakeng_yolov8n_cv181x_int8_640_compat12.cvimodel"),
).strip()
BUNKER_ENGINE = os.environ.get("BUNKER_ENGINE", "auto").strip().lower()
if BUNKER_ENGINE in {"", "auto"}:
    BUNKER_ENGINE = "rknn" if BUNKER_MODEL_PATH.lower().endswith(".rknn") else "sample"
BUNKER_STOP_CLASS_IDS = _parse_int_set_env("BUNKER_STOP_CLASS_IDS", "0,1")
BUNKER_STOP_SCORE_THRESHOLD = float(os.environ.get("BUNKER_STOP_SCORE_THRESHOLD", "0.0"))
BUNKER_STOP_AREA_RATIO = float(os.environ.get("BUNKER_STOP_AREA_RATIO", "0.05"))
BUNKER_CLASS_NAMES = _parse_class_names_env("BUNKER_CLASS_NAMES", "0:bunker,1:pond")
BUNKER_NUM_CLASSES = int(os.environ.get("BUNKER_NUM_CLASSES", "2"))
BUNKER_RKNN_INPUT_SIZE = int(os.environ.get("BUNKER_RKNN_INPUT_SIZE", "640"))
BUNKER_RKNN_NMS_THRESHOLD = float(os.environ.get("BUNKER_RKNN_NMS_THRESHOLD", "0.45"))
BUNKER_RKNN_BACKEND = os.environ.get("BUNKER_RKNN_BACKEND", os.environ.get("RKNN_BACKEND", "auto")).strip()
BUNKER_RKNN_CORE_MASK = os.environ.get("BUNKER_RKNN_CORE_MASK", os.environ.get("RKNN_CORE_MASK", "auto")).strip()
BUNKER_RKNN_INPUT_FORMAT = os.environ.get("BUNKER_RKNN_INPUT_FORMAT", "RGB").strip()
BUNKER_RKNN_BOX_FORMAT = os.environ.get("BUNKER_RKNN_BOX_FORMAT", "xywh").strip()
BUNKER_SAMPLE_DET_CONF = float(os.environ.get("BUNKER_SAMPLE_DET_CONF", "0.75"))
BUNKER_GET_FRAME_TIMEOUT_MS = int(os.environ.get("BUNKER_GET_FRAME_TIMEOUT_MS", "200"))
BUNKER_LOOP_PERIOD_MS = int(os.environ.get("BUNKER_LOOP_PERIOD_MS", "100"))
BUNKER_SPLIT_RESTART_DELAY = float(os.environ.get("BUNKER_SPLIT_RESTART_DELAY", "0.8"))
BUNKER_SPLIT_ACTIVE_HOLD_SEC = float(os.environ.get("BUNKER_SPLIT_ACTIVE_HOLD_SEC", "0.35"))
BUNKER_STOP_CONSEC_FRAMES = int(os.environ.get("BUNKER_STOP_CONSEC_FRAMES", "2"))
BUNKER_SPLIT_ECHO_RAW = os.environ.get("BUNKER_SPLIT_ECHO_RAW", "0").strip() != "0"
BUNKER_SAMPLE_BINARY = os.environ.get("BUNKER_SAMPLE_BINARY", "./sample_personv8_track").strip()
BUNKER_RUNTIME_CONFIG = BunkerHazardRuntimeConfig(
    enabled=BUNKER_AVOID_ENABLE,
    mode=BUNKER_DETECT_MODE,
    workdir=BUNKER_WORKDIR,
    model_path=BUNKER_MODEL_PATH,
    engine=BUNKER_ENGINE,
    class_ids=tuple(BUNKER_STOP_CLASS_IDS),
    class_names=BUNKER_CLASS_NAMES,
    score_threshold=BUNKER_STOP_SCORE_THRESHOLD,
    area_ratio_stop=BUNKER_STOP_AREA_RATIO,
    num_classes=BUNKER_NUM_CLASSES,
    rknn_input_size=BUNKER_RKNN_INPUT_SIZE,
    rknn_nms_threshold=BUNKER_RKNN_NMS_THRESHOLD,
    rknn_backend=BUNKER_RKNN_BACKEND,
    rknn_core_mask=BUNKER_RKNN_CORE_MASK,
    rknn_input_format=BUNKER_RKNN_INPUT_FORMAT,
    rknn_box_format=BUNKER_RKNN_BOX_FORMAT,
    sample_det_conf=BUNKER_SAMPLE_DET_CONF,
    get_frame_timeout_ms=BUNKER_GET_FRAME_TIMEOUT_MS,
    loop_period_ms=BUNKER_LOOP_PERIOD_MS,
    split_restart_delay=BUNKER_SPLIT_RESTART_DELAY,
    split_active_hold_sec=BUNKER_SPLIT_ACTIVE_HOLD_SEC,
    stop_consec_frames=BUNKER_STOP_CONSEC_FRAMES,
    split_echo_raw=BUNKER_SPLIT_ECHO_RAW,
    sample_binary=BUNKER_SAMPLE_BINARY,
    frame_width=int(os.environ.get("BUNKER_FRAME_WIDTH", "1920")),
    frame_height=int(os.environ.get("BUNKER_FRAME_HEIGHT", "1080")),
    runtime_base=os.environ.get("ZKWL_RUNTIME_BASE", "/mnt/system/runtime/zkwl-runtime").strip(),
    rknn_target=os.environ.get("RKNN_TARGET", "rk3588").strip(),
)

# 动作类型
ACTION_FORWARD = 0
ACTION_ROTATE_LEFT = 1
ACTION_ROTATE_RIGHT = 2
ACTION_STOP = 3
ACTION_STEER_LEFT = 4
ACTION_STEER_RIGHT = 5
ACTION_BACKWARD = 6
MOVEMENT_ACTIONS = {
    ACTION_FORWARD,
    ACTION_BACKWARD,
    ACTION_ROTATE_LEFT,
    ACTION_ROTATE_RIGHT,
    ACTION_STEER_LEFT,
    ACTION_STEER_RIGHT,
}
FORWARD_LIKE_ACTIONS = {
    ACTION_FORWARD,
    ACTION_STEER_LEFT,
    ACTION_STEER_RIGHT,
}
ROTATE_ACTIONS = {
    ACTION_ROTATE_LEFT,
    ACTION_ROTATE_RIGHT,
}
ACTION_NAMES = {
    ACTION_FORWARD: "forward",
    ACTION_BACKWARD: "backward",
    ACTION_ROTATE_LEFT: "rotate_left",
    ACTION_ROTATE_RIGHT: "rotate_right",
    ACTION_STOP: "stop",
    ACTION_STEER_LEFT: "steer_left",
    ACTION_STEER_RIGHT: "steer_right",
}
ACTION_RUNTIME_SYMBOLS = ActionRuntimeSymbols(
    forward=ACTION_FORWARD,
    backward=ACTION_BACKWARD,
    rotate_left=ACTION_ROTATE_LEFT,
    rotate_right=ACTION_ROTATE_RIGHT,
    stop=ACTION_STOP,
    steer_left=ACTION_STEER_LEFT,
    steer_right=ACTION_STEER_RIGHT,
    movement_actions=frozenset(MOVEMENT_ACTIONS),
    forward_like_actions=frozenset(FORWARD_LIKE_ACTIONS),
    rotate_actions=frozenset(ROTATE_ACTIONS),
    action_names=ACTION_NAMES,
    safety_stop_reasons=frozenset(SAFETY_STOP_REASONS),
)
ACTION_RUNTIME_CONFIG = ActionRuntimeConfig(
    enable_motor_rpm_feedback=ENABLE_MOTOR_RPM_FEEDBACK,
    motor_feedback_poll_interval_sec=MOTOR_FEEDBACK_POLL_INTERVAL_SEC,
    brake_hold_refresh_interval_sec=BRAKE_HOLD_REFRESH_INTERVAL_SEC,
    use_percent_speed=USE_PERCENT_SPEED,
    min_forward_percent=MIN_FORWARD_PERCENT,
    max_forward_percent=MAX_FORWARD_PERCENT,
    steer_percent_limit=STEER_PERCENT_LIMIT,
    mmwave_hold_forward_percent=VISION_MMWAVE_HOLD_FORWARD_PERCENT,
    visible_steer_inner_ratio_percent=VISIBLE_STEER_INNER_RATIO_PERCENT,
    visible_steer_outer_ratio_percent=VISIBLE_STEER_OUTER_RATIO_PERCENT,
    motor_forward_raw_target=MOTOR_FORWARD_RAW_TARGET,
    motor_forward_max_target_rpm=MOTOR_FORWARD_MAX_TARGET_RPM,
    motor_steer_raw_target=MOTOR_STEER_RAW_TARGET,
    motor_rotate_raw_target=MOTOR_ROTATE_RAW_TARGET,
    motor_forward_target_sign=MOTOR_FORWARD_TARGET_SIGN,
    motor_left_sign=MOTOR_LEFT_SIGN,
    motor_right_sign=MOTOR_RIGHT_SIGN,
    rotate_prep_coast_enable=ROTATE_PREP_COAST_ENABLE,
    rotate_prep_coast_steps=ROTATE_PREP_COAST_STEPS,
    rotate_prep_coast_total_sec=ROTATE_PREP_COAST_TOTAL_SEC,
    rotate_pulse_brake_enable=ROTATE_PULSE_BRAKE_ENABLE,
    rotate_pulse_stop_mode=ROTATE_PULSE_STOP_MODE,
    rotate_pulse_pause_sec=ROTATE_PULSE_PAUSE_SEC,
    rotate_pulse_observe_min_frames=ROTATE_PULSE_OBSERVE_MIN_FRAMES,
    rotate_pulse_settle_enable=ROTATE_PULSE_SETTLE_ENABLE,
    rotate_pulse_settle_quiet_sec=ROTATE_PULSE_SETTLE_QUIET_SEC,
    rotate_pulse_settle_timeout_sec=ROTATE_PULSE_SETTLE_TIMEOUT_SEC,
    rotate_pulse_settle_feedback_stale_sec=ROTATE_PULSE_SETTLE_FEEDBACK_STALE_SEC,
    rotate_pulse_settle_max_wheel_rpm=ROTATE_PULSE_SETTLE_MAX_WHEEL_RPM,
    rotate_pulse_settle_max_yaw_rate_dps=ROTATE_PULSE_SETTLE_MAX_YAW_RATE_DPS,
    rotate_pulse_transition_rpm=ROTATE_PULSE_TRANSITION_RPM,
    rotate_chain_memory_sec=ROTATE_CHAIN_MEMORY_SEC,
    rotate_hold_stale_sec=ROTATE_HOLD_STALE_SEC,
    rotate_turn_percent_from_forward=ROTATE_TURN_PERCENT_FROM_FORWARD,
    rotate_turn_percent_chain=ROTATE_TURN_PERCENT_CHAIN,
    rotate_duration=ROTATE_DURATION,
    motor_rs485_target_min_interval_sec=MOTOR_RS485_TARGET_MIN_INTERVAL_SEC,
    motor_forward_like_keepalive_sec=MOTOR_FORWARD_LIKE_KEEPALIVE_SEC,
    motor_rs485_stop_mode=MOTOR_RS485_STOP_MODE,
    safety_stop_mode=SAFETY_STOP_MODE,
    motor_rs485_transition_stop_mode=MOTOR_RS485_TRANSITION_STOP_MODE,
    motor_rs485_transition_stop_delay_sec=MOTOR_RS485_TRANSITION_STOP_DELAY_SEC,
    motor_rs485_transition_stop_repeat=MOTOR_RS485_TRANSITION_STOP_REPEAT,
    follow_brake_distance_m=FOLLOW_BRAKE_DISTANCE_M,
    visible_steering_pid_enable=VISIBLE_STEERING_PID_ENABLE,
    steering_feedback_poll_interval_sec=VISIBLE_STEERING_PID_FEEDBACK_POLL_INTERVAL_SEC,
    steering_feedback_log_interval_sec=VISIBLE_STEERING_PID_FEEDBACK_LOG_INTERVAL_SEC,
    steering_feedback_median_window=VISIBLE_STEERING_PID_FEEDBACK_MEDIAN_WINDOW,
    steering_feedback_left_body_deg_per_encoder_deg=VISIBLE_STEERING_PID_LEFT_BODY_DEG_PER_ENCODER_DEG,
    steering_feedback_right_body_deg_per_encoder_deg=VISIBLE_STEERING_PID_RIGHT_BODY_DEG_PER_ENCODER_DEG,
    rotation_only=FOLLOW_ROTATION_ONLY,
    rotation_only_yaw_pulse_rpm=ROTATION_ONLY_YAW_PULSE_RPM,
    rotation_only_yaw_pulse_min_sec=ROTATION_ONLY_YAW_PULSE_MIN_SEC,
    rotation_only_yaw_pulse_max_sec=ROTATION_ONLY_YAW_PULSE_MAX_SEC,
    rotation_only_yaw_brake_sec=ROTATION_ONLY_YAW_BRAKE_SEC,
    rotation_only_yaw_response_dps=ROTATION_ONLY_YAW_RESPONSE_DPS,
    rotation_only_yaw_zero_gap_sec=ROTATION_ONLY_YAW_ZERO_GAP_SEC,
    rotate_pulse_active_brake_enable=ROTATE_PULSE_ACTIVE_BRAKE_ENABLE,
    rotate_pulse_active_brake_rpm=ROTATE_PULSE_ACTIVE_BRAKE_RPM,
    rotate_pulse_active_brake_sec=ROTATE_PULSE_ACTIVE_BRAKE_SEC,
)

# 设置日志
logger = logging.getLogger("PersonTracker")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - 跟随车 - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class PersonTracker:
    """人员跟踪器类（避障版本）- 集成毫米波和红外传感器"""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the RK3588 tracker, vision pipeline, sensors, and LZ30EMA motor backend."""
        self._model_path = model_path if model_path is not None else YOLO_MODEL_PATH
        self._vision_engine = VISION_ENGINE
        if self._vision_engine in ("rknn", "rk", "rk3588", "rk_rknn", "rknn_runtime"):
            self._vision_engine = "rknn"
        else:
            raise RuntimeError(f"Unsupported VISION_ENGINE={VISION_ENGINE!r}; request_0513_modular.py only supports rknn")
        self._rknn_pipeline = None
        self._rknn_camera = None
        self._direction_pool = None
        self._capture_thread = None
        self._capture_stop_event = threading.Event()
        self._capture_queue = queue.Queue(maxsize=RKNN_CAPTURE_QUEUE_SIZE)
        self._capture_frame_id = 0
        self._capture_state_lock = threading.Lock()
        self._direction_result_backlog = []
        self._direction_evidence_merged_total = 0
        self._direction_evidence_last_log_ts = 0.0
        self._camera_video_recorder = None
        if RKNN_CAMERA_RAW_OUTPUT and RKNN_CAMERA_CAPTURE_MODE in {"opencv", "opencv_v4l2", "v4l2"}:
            if cv2 is None:
                raise RuntimeError(f"OpenCV(cv2) import failed: {_CV2_IMPORT_ERROR}")
            self._camera_video_recorder = AsyncVideoRecorder(
                VideoRecorderConfig(
                    output_path=RKNN_CAMERA_RAW_OUTPUT,
                    fps=RKNN_CAMERA_FPS,
                    fourcc="MJPG",
                    queue_capacity=60,
                ),
                cv2_module=cv2,
                logger=logger,
            )
        self._rknn_camera_next_retry_ts = 0.0
        self._last_rknn_no_frame_log_ts = 0.0
        self._sensor_runtime = SensorRuntime(SENSOR_RUNTIME_CONFIG, logger=logger)
        logger.info(
            "LZ30EMA RS485 motor backend enabled: port=%s limit=%d%% forward_rpm=%d steer_rpm=%d rotate_rpm=%d rotate_rpm_visible=%d rotate_rpm_lost_wait=%d rotate_rpm_search=%d signs(left=%d,right=%d,forward=%d) parking_current=%.1fA",
            MOTOR_RS485_PORT,
            MOTOR_PERCENT_LIMIT,
            MOTOR_FORWARD_RAW_TARGET,
            MOTOR_STEER_RAW_TARGET,
            MOTOR_ROTATE_RAW_TARGET,
            ROTATE_RAW_TARGET_VISIBLE,
            ROTATE_RAW_TARGET_LOST_WAIT,
            ROTATE_RAW_TARGET_SEARCH,
            MOTOR_LEFT_SIGN,
            MOTOR_RIGHT_SIGN,
            MOTOR_FORWARD_TARGET_SIGN,
            MOTOR_PARKING_CURRENT_A,
        )
        logger.info(
            "Rotate pulse scan: enabled=%s duration=%.3fs transition=%dRPM observe_pause=%.3fs observe_min_frames=%d stop_mode=%s settle=%s quiet=%.3fs timeout=%.3fs feedback_stale=%.3fs max_wheel=%dRPM max_yaw=%.2fdps",
            ROTATE_PULSE_BRAKE_ENABLE,
            ROTATE_DURATION,
            ROTATE_PULSE_TRANSITION_RPM,
            ROTATE_PULSE_PAUSE_SEC,
            ROTATE_PULSE_OBSERVE_MIN_FRAMES,
            ROTATE_PULSE_STOP_MODE,
            ROTATE_PULSE_SETTLE_ENABLE,
            ROTATE_PULSE_SETTLE_QUIET_SEC,
            ROTATE_PULSE_SETTLE_TIMEOUT_SEC,
            ROTATE_PULSE_SETTLE_FEEDBACK_STALE_SEC,
            ROTATE_PULSE_SETTLE_MAX_WHEEL_RPM,
            ROTATE_PULSE_SETTLE_MAX_YAW_RATE_DPS,
        )
        logger.info(
            "Rotation-only pulse control: enabled=%s fixed=%dRPM duration=%.0f-%.0fms "
            "response=%.1fdps zero_gap=%.0fms brake=%dRPM/%.0fms search_active_brake=%s",
            FOLLOW_ROTATION_ONLY,
            ROTATION_ONLY_YAW_PULSE_RPM,
            ROTATION_ONLY_YAW_PULSE_MIN_SEC * 1000.0,
            ROTATION_ONLY_YAW_PULSE_MAX_SEC * 1000.0,
            ROTATION_ONLY_YAW_RESPONSE_DPS,
            ROTATION_ONLY_YAW_ZERO_GAP_SEC * 1000.0,
            ROTATE_PULSE_ACTIVE_BRAKE_RPM,
            ROTATE_PULSE_ACTIVE_BRAKE_SEC * 1000.0,
            ROTATE_PULSE_ACTIVE_BRAKE_ENABLE,
        )
        if LOADED_CONFIG is not None:
            logger.info("已加载配置文件: %s", LOADED_CONFIG.path)
        logger.info(
            "模块开关: vision=%s, ir=%s, mmwave=%s, astra_depth=%s, ultrasonic=%s, imu=%s, distance_source=%s, bunker=%s(%s)",
            MODULE_VISION_ENABLE,
            MODULE_IR_ENABLE,
            MODULE_MMWAVE_ENABLE,
            MODULE_ASTRA_DEPTH_ENABLE,
            MODULE_ULTRASONIC_ENABLE,
            MODULE_IMU_ENABLE,
            DISTANCE_SOURCE,
            BUNKER_AVOID_ENABLE,
            BUNKER_DETECT_MODE,
        )
        logger.info(
            "运行策略: mmwave_runtime_disabled=%s distance_parking_enable=%s "
            "ir_only_explicit_stop=%s astra_rgb_depth_align_delay_ms=%.0f rotation_only=%s",
            MMWAVE_RUNTIME_DISABLED,
            DISTANCE_PARKING_ENABLE,
            not DISTANCE_PARKING_ENABLE,
            ASTRA_DEPTH_RGB_PROCESSING_DELAY_SEC * 1000.0,
            FOLLOW_ROTATION_ONLY,
        )
        if FOLLOW_ROTATION_ONLY:
            logger.warning(
                "仅旋转跟随测试已启用: 禁止前进/后退，目标可见时仅允许视觉PID原地偏航，红外硬停保留"
            )
        logger.info(
            "Safety config: side_ir_blocks_rotation=%s blocked_turn=stop "
            "safety_stop_mode=%s safety_stop_reasons=%s",
            SIDE_IR_BLOCKS_ROTATION,
            SAFETY_STOP_MODE,
            sorted(SAFETY_STOP_REASONS),
        )
        logger.info(
            "Follow search config: search_before_first=%s startup_delay=%.2f search_timeout=%.2f exit_on_timeout=%s exit_on_target_loss=%s revolution=%.1fdeg feedback_stale=%.2fs initial_confirm_frames=%d search_reacquire_confirm=%dframes search_rotate_continuous=%s lost_confirm_sec=%.2f lost_confirm_frames=%d visible_turn_cooldown=%dframes steer_hold=%.2fs/%dframes/max%.2fs release_target_on_lost=%s",
            FOLLOW_SEARCH_BEFORE_FIRST_PERSON,
            FOLLOW_STARTUP_SEARCH_DELAY_SEC,
            FOLLOW_SEARCH_TIMEOUT_SEC,
            FOLLOW_SEARCH_TIMEOUT_EXIT_PROGRAM,
            FOLLOW_EXIT_ON_TARGET_LOSS,
            FOLLOW_SEARCH_REVOLUTION_DEG,
            FOLLOW_SEARCH_REVOLUTION_FEEDBACK_STALE_SEC,
            FOLLOW_INITIAL_TARGET_CONFIRM_FRAMES,
            SEARCH_CONFIRMED_REACQUIRE_FRAMES,
            SEARCH_ROTATE_CONTINUOUS_ENABLE,
            FOLLOW_LOST_CONFIRM_SEC,
            LOST_CONFIRM_FRAMES,
            ACTION_COOLDOWN,
            FOLLOW_STEER_MIN_HOLD_SEC,
            FOLLOW_STEER_LOST_HOLD_FRAMES,
            FOLLOW_STEER_LOST_HOLD_MAX_SEC,
            FOLLOW_RELEASE_TARGET_ON_LOST,
        )
        logger.info(
            "Stale direction recovery: enabled=%s history_confirm=%dframes "
            "action=stop_until_history_side probe_disabled=True",
            FOLLOW_STALE_DIRECTION_RECOVERY_ENABLE,
            LOST_CONFIRM_FRAMES,
        )
        logger.info(
            "Capture direction history: enabled=%s depth=60frames lookback=6samples "
            "completion=fresh_missing_%dframes edge+motion_required",
            FOLLOW_DIRECTION_HISTORY_ENABLE,
            LOST_CONFIRM_FRAMES,
        )
        logger.info(
            "Search evidence gate: enabled=%s formal>=%.2f probe>=%.2f/%dframes "
            "area=[%.3f,%.3f] iou>=%.2f evidence_hold=%dframes rearm_missing=%dframes "
            "candidate_observe=%dfresh_frames untracked_direction>=%.2f "
            "candidate_acquire=%drpm aimline_approach<=%.3f/1rpm "
            "aimline_intersection=active_brake "
            "identity_claim=False direction=geometry_only",
            SEARCH_EVIDENCE_GATE_ENABLE,
            CONFIDENCE_THRESHOLD,
            SEARCH_EVIDENCE_PROBE_MIN_SCORE,
            SEARCH_EVIDENCE_PROBE_CONFIRM_FRAMES,
            SEARCH_EVIDENCE_MIN_AREA_RATIO,
            SEARCH_EVIDENCE_MAX_AREA_RATIO,
            SEARCH_EVIDENCE_CONSISTENCY_IOU,
            SEARCH_EVIDENCE_HOLD_FRAMES,
            SEARCH_EVIDENCE_REARM_MISSING_FRAMES,
            SEARCH_EVIDENCE_HOLD_FRAMES,
            SEARCH_CANDIDATE_UNTRACKED_MIN_SCORE,
            SEARCH_CANDIDATE_ACQUIRE_RAW_RPM,
            SEARCH_CANDIDATE_APPROACH_MARGIN_RATIO,
        )
        logger.info(
            "Single-direction search: direction=frozen_on_loss "
            "completion=encoder_heading_coverage_%.1fdeg",
            FOLLOW_SEARCH_REVOLUTION_DEG,
        )
        logger.info(
            "Visible steering PID: enabled=%s center=[%.3f,%.3f] deadband=%.2fdeg outer=(kp=%.3f,kd=%.3f,max=%.1fdps) rate=(kp=%.3f,ki=%.3f) correction_max=%.1frpm dynamic=[error %.1f->%.1fdeg,yaw %.1f->%.1fdps,correction %.1f->%.1frpm,base_cap %.1frpm] fallback_correction_max=%.1frpm countersteer=[threshold=%.1fdps,boost=%.1frpm,cap=%.1frpm] legacy_floor=[%.1fdeg/%.1frpm,%.1fdeg/%.1frpm,%.1fdeg/%.1frpm release=%.2f] startup_kick=[error>=%.1fdeg,%.1frpm,max=%.2fs,encoder>=%.1fdps] active_brake=[%.1fdps/%.1frpm] edge=[start=%.1fdeg,inner_margin=%.1frpm] feedback=%.3fs/stale%.3fs encoder_scale(left=%.4f,right=%.4f) trace_interval=%.2fs",
            VISIBLE_STEERING_PID_ENABLE,
            FOLLOW_STEER_ENTER_LEFT_RATIO,
            FOLLOW_STEER_ENTER_RIGHT_RATIO,
            VISIBLE_STEERING_PID_DEADBAND_DEG,
            VISIBLE_STEERING_PID_OUTER_KP_PER_SEC,
            VISIBLE_STEERING_PID_OUTER_KD_SEC,
            VISIBLE_STEERING_PID_MAX_YAW_RATE_DPS,
            VISIBLE_STEERING_PID_RATE_KP_RPM_PER_DPS,
            VISIBLE_STEERING_PID_RATE_KI_RPM_PER_DEG,
            VISIBLE_STEERING_PID_MAX_CORRECTION_RPM,
            VISIBLE_STEERING_PID_DYNAMIC_SMALL_ERROR_DEG,
            VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_DEG,
            VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_YAW_RATE_DPS,
            VISIBLE_STEERING_PID_MAX_YAW_RATE_DPS,
            VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_CORRECTION_RPM,
            VISIBLE_STEERING_PID_MAX_CORRECTION_RPM,
            VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_BASE_CAP_RPM,
            VISIBLE_STEERING_PID_FALLBACK_MAX_CORRECTION_RPM,
            VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_THRESHOLD_DPS,
            VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_BOOST_RPM,
            VISIBLE_STEERING_PID_BRAKING_MAX_CORRECTION_RPM,
            VISIBLE_STEERING_PID_MIN_EFFECTIVE_ERROR_DEG,
            VISIBLE_STEERING_PID_MIN_EFFECTIVE_CORRECTION_RPM,
            VISIBLE_STEERING_PID_MECHANICAL_TIER2_ERROR_DEG,
            VISIBLE_STEERING_PID_MECHANICAL_TIER2_CORRECTION_RPM,
            VISIBLE_STEERING_PID_MECHANICAL_TIER3_ERROR_DEG,
            VISIBLE_STEERING_PID_MECHANICAL_TIER3_CORRECTION_RPM,
            VISIBLE_STEERING_PID_MECHANICAL_FLOOR_RELEASE_RATIO,
            VISIBLE_STEERING_PID_STARTUP_KICK_ERROR_DEG,
            VISIBLE_STEERING_PID_STARTUP_KICK_RPM,
            VISIBLE_STEERING_PID_STARTUP_KICK_MAX_SEC,
            VISIBLE_STEERING_PID_STARTUP_KICK_RELEASE_YAW_RATE_DPS,
            VISIBLE_STEERING_PID_ACTIVE_BRAKE_YAW_THRESHOLD_DPS,
            VISIBLE_STEERING_PID_ACTIVE_BRAKE_MIN_CORRECTION_RPM,
            VISIBLE_STEERING_PID_EDGE_BOOST_START_ERROR_DEG,
            VISIBLE_STEERING_PID_AGGRESSIVE_INNER_WHEEL_MARGIN_RPM,
            VISIBLE_STEERING_PID_FEEDBACK_POLL_INTERVAL_SEC,
            VISIBLE_STEERING_PID_FEEDBACK_STALE_SEC,
            VISIBLE_STEERING_PID_LEFT_BODY_DEG_PER_ENCODER_DEG,
            VISIBLE_STEERING_PID_RIGHT_BODY_DEG_PER_ENCODER_DEG,
            PID_TRACE_INTERVAL_SEC,
        )
        logger.info(
            "Fast countersteer: max=%.1fRPM gain=%.3fRPM/dps baseline_cap=%.1fRPM",
            VISIBLE_STEERING_PID_FAST_COUNTERSTEER_MAX_CORRECTION_RPM,
            VISIBLE_STEERING_PID_FAST_COUNTERSTEER_GAIN_RPM_PER_DPS,
            VISIBLE_STEERING_PID_BRAKING_MAX_CORRECTION_RPM,
        )
        logger.info(
            "Visual aim guard: enabled=%s tracking_floor=%.1fRPM "
            "predictive_brake=%.1fdeg/s^2+%.2fdeg response=%.0fms "
            "target_rate_ff=%s max_closing=%.1fdps",
            VISIBLE_STEERING_PID_VISUAL_DIRECTION_GUARD_ENABLE,
            VISIBLE_STEERING_PID_MIN_EFFECTIVE_CORRECTION_RPM,
            VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_DECEL_DPS2,
            VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_MARGIN_DEG,
            VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_RESPONSE_SEC * 1000.0,
            VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_GAIN > 0.0,
            VISIBLE_STEERING_PID_TARGET_SPEED_MATCH_MAX_CLOSING_DPS,
        )
        logger.info(
            "Distance PID: enabled=%s kp=%.2f ki=%.2f kd=%.2f deadband=%.3fm "
            "d_filter_alpha=%.2f jump_guard=%.2fm rise=%.1fRPM/s fall=%.1fRPM/s "
            "output=%d..%dRPM",
            DISTANCE_PID_ENABLE,
            DISTANCE_PID_KP_RPM_PER_M,
            DISTANCE_PID_KI_RPM_PER_M_S,
            DISTANCE_PID_KD_RPM_S_PER_M,
            DISTANCE_PID_DEADBAND_M,
            DISTANCE_PID_DERIVATIVE_FILTER_ALPHA,
            DISTANCE_PID_MAX_MEASUREMENT_JUMP_M,
            DISTANCE_PID_OUTPUT_RISE_RPM_PER_SEC,
            DISTANCE_PID_OUTPUT_FALL_RPM_PER_SEC,
            FORWARD_MIN_RPM,
            FORWARD_MAX_RPM,
        )
        logger.info(
            "Parked recenter PID: in_place=True raw_rpm=%d..%d search_raw_rpm=%d/%d camera_encoder_loop=%s",
            PARKED_RECENTER_MIN_RPM,
            PARKED_RECENTER_MAX_RPM,
            ROTATE_RAW_TARGET_LOST_WAIT,
            ROTATE_RAW_TARGET_SEARCH,
            VISIBLE_STEERING_PID_ENABLE,
        )
        logger.info(
            "Lateral intent control: enabled=%s rate=%.1fHz ttl=%.0fms "
            "projection=%.0fms/%.3fratio slew=%.0f/%.0frpm_s motor_publish=%.0fms",
            LATERAL_INTENT_CONTROL_ENABLE,
            LATERAL_INTENT_CONTROL_RATE_HZ,
            LATERAL_INTENT_TTL_SEC * 1000.0,
            LATERAL_INTENT_MAX_PROJECTION_SEC * 1000.0,
            LATERAL_INTENT_MAX_PROJECTION_RATIO,
            LATERAL_INTENT_RISE_RPM_PER_SEC,
            LATERAL_INTENT_BRAKE_RPM_PER_SEC,
            LATERAL_INTENT_MOTOR_PUBLISH_INTERVAL_SEC * 1000.0,
        )
        logger.info(
            "Visible target-rate feedforward: lookback=%.3fs history=%d filter_alpha=%.2f "
            "gain=%.2f max=%.1fdps speed_match_closing=%.1fdps camera_latency=%.3fs",
            VISIBLE_MOTION_LOOKBACK_SEC,
            VISIBLE_MOTION_HISTORY_FRAMES,
            VISIBLE_MOTION_RATE_FILTER_ALPHA,
            VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_GAIN,
            VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_MAX_DPS,
            VISIBLE_STEERING_PID_TARGET_SPEED_MATCH_MAX_CLOSING_DPS,
            VISIBLE_STEERING_PID_CAMERA_LATENCY_SEC,
        )
        logger.info(
            "Distance/motion config: mmwave_match_mode=%s mmwave_angle_tie=%.1fdeg mmwave_latency=%.3fs mmwave_async=%s mmwave_continuity=[center<=%.1fdeg,distance_jump<=%.2fm,confirm=%d,memory=%.2fs] target=%.2fm brake=%.2fm speed_tiers=[<=1.3:%d,<=1.7:%d,<=2.1:%d,<=2.6:%d,<=3.2:%d,<=3.8:%d,<=4.5:%d,far:%d] max=%d fallback=%d forward_keepalive=%.3fs",
            VISION_MMWAVE_MATCH_MODE,
            VISION_MMWAVE_ANGLE_TIE_MARGIN_DEG,
            VISION_MMWAVE_LATENCY_SEC,
            VISION_MMWAVE_USE_ASYNC_CACHE,
            VISION_MMWAVE_MAX_CENTER_ANGLE_DIFF_DEG,
            VISION_MMWAVE_MAX_DISTANCE_JUMP_M,
            VISION_MMWAVE_SWITCH_CONFIRM_FRAMES,
            VISION_MMWAVE_CONTINUITY_MEMORY_SEC,
            TARGET_DISTANCE,
            FOLLOW_BRAKE_DISTANCE_M,
            FORWARD_SPEED_LE_1_3_PERCENT,
            FORWARD_SPEED_LE_1_7_PERCENT,
            FORWARD_SPEED_LE_2_1_PERCENT,
            FORWARD_SPEED_LE_2_6_PERCENT,
            FORWARD_SPEED_LE_3_2_PERCENT,
            FORWARD_SPEED_LE_3_8_PERCENT,
            FORWARD_SPEED_LE_4_5_PERCENT,
            FORWARD_SPEED_FAR_PERCENT,
            MAX_FORWARD_PERCENT,
            DISTANCE_MISSING_FORWARD_PERCENT,
            MOTOR_FORWARD_LIKE_KEEPALIVE_SEC,
        )
        logger.info(
            "Reverse control config: enable=%s start=%.2fm immediate=%.2fm stop=%.2fm "
            "confirm=%dframes output=%d..%drpm runtime_cap=%drpm "
            "visual_guard=[area>=%.2f,height>=%.2f+area,far_veto>%.2fm] "
            "steering_center=[%.2f,%.2f] correction_max=%.1frpm",
            FOLLOW_REVERSE_ENABLE,
            FOLLOW_REVERSE_START_DISTANCE_M,
            FOLLOW_REVERSE_IMMEDIATE_DISTANCE_M,
            FOLLOW_REVERSE_STOP_DISTANCE_M,
            FOLLOW_REVERSE_CONFIRM_FRAMES,
            FOLLOW_REVERSE_MIN_RPM,
            FOLLOW_REVERSE_MAX_RPM,
            FOLLOW_REVERSE_RUNTIME_CAP_RPM,
            FOLLOW_REVERSE_VISUAL_GUARD_AREA_RATIO,
            FOLLOW_REVERSE_VISUAL_GUARD_HEIGHT_RATIO,
            FOLLOW_REVERSE_VISUAL_GUARD_MAX_DISTANCE_M,
            FOLLOW_CENTER_LEFT_RATIO,
            FOLLOW_CENTER_RIGHT_RATIO,
            min(6.0, VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_CORRECTION_RPM),
        )
        if MODULE_ASTRA_DEPTH_ENABLE:
            logger.info(
                "Astra Depth config: %dx%d@%dFPS range=[%.2f,%.2f]m frame_age<=%.2fs hold=%.2fs RGB_align_delay=%.0fms median=%d jump<=%.2fm/%dframes",
                ASTRA_DEPTH_WIDTH,
                ASTRA_DEPTH_HEIGHT,
                ASTRA_DEPTH_FPS,
                ASTRA_DEPTH_MIN_DISTANCE_M,
                ASTRA_DEPTH_MAX_DISTANCE_M,
                ASTRA_DEPTH_MAX_FRAME_AGE_SEC,
                ASTRA_DEPTH_HOLD_SEC,
                ASTRA_DEPTH_RGB_PROCESSING_DELAY_SEC * 1000.0,
                ASTRA_DEPTH_MEDIAN_WINDOW,
                ASTRA_DEPTH_MAX_DISTANCE_JUMP_M,
                ASTRA_DEPTH_JUMP_CONFIRM_FRAMES,
            )
            logger.info(
                "Astra foreground safety: regions=5 dynamic_pixels=max(%d,area*%.0f%%) "
                "cluster=%.2fm spatial>=%.0f%% large_bbox area>=%.2f or height>=%.2f "
                "rejects>%.2fm near<=%.2fm anchor_age=%.1f/%.1fs confirms=%d/%d "
                "bbox_ratio<=%.2f edge=%.3f",
                ASTRA_DEPTH_DYNAMIC_MIN_VALID_FLOOR,
                ASTRA_DEPTH_DYNAMIC_MIN_VALID_FRACTION * 100.0,
                ASTRA_DEPTH_FOREGROUND_CLUSTER_SPAN_M,
                ASTRA_DEPTH_FOREGROUND_SPATIAL_SUPPORT_FRACTION * 100.0,
                ASTRA_DEPTH_LARGE_BBOX_GUARD_AREA_RATIO,
                ASTRA_DEPTH_LARGE_BBOX_GUARD_HEIGHT_RATIO,
                ASTRA_DEPTH_LARGE_BBOX_GUARD_MAX_DISTANCE_M,
                ASTRA_DEPTH_NEAR_GUARD_DISTANCE_M,
                ASTRA_DEPTH_ANCHOR_STRICT_AGE_SEC,
                ASTRA_DEPTH_ANCHOR_EXPIRE_AGE_SEC,
                ASTRA_DEPTH_NEAR_FAR_JUMP_CONFIRM_FRAMES,
                ASTRA_DEPTH_REANCHOR_CONFIRM_FRAMES,
                ASTRA_DEPTH_NEAR_FAR_JUMP_MAX_BBOX_RATIO,
                ASTRA_DEPTH_NEAR_FAR_JUMP_EDGE_MARGIN_RATIO,
            )
            logger.info(
                "Depth confidence control: estimate<=200ms medium<=%.0fms/%drpm "
                "recovery=%drpm@%.0fms -> %drpm@%.0fms -> PID",
                FOLLOW_DEPTH_MEDIUM_CONFIDENCE_HOLD_SEC * 1000.0,
                FOLLOW_DEPTH_MEDIUM_CONFIDENCE_RPM,
                FOLLOW_DEPTH_RECOVERY_STAGE1_RPM,
                FOLLOW_DEPTH_RECOVERY_STAGE1_SEC * 1000.0,
                FOLLOW_DEPTH_RECOVERY_STAGE2_RPM,
                FOLLOW_DEPTH_RECOVERY_STAGE2_SEC * 1000.0,
            )
            logger.info(
                "Control quality guard: rejected_bbox=lateral_only low_quality_forward=blocked "
                "low_quality_yaw=%.1f..%.1frpm "
                "near_camera_occlusion=stop(area>=%.2f,width>=%.2f,height>=%.2f) "
                "fresh_far_jump<=%.2fm confirm=%dframes depth30_visual_dedupe=%.0fms",
                VISIBLE_LOW_QUALITY_STEER_MAX_CORRECTION_RPM,
                VISIBLE_LOW_QUALITY_STEER_EDGE_MAX_CORRECTION_RPM,
                VISIBLE_NEAR_CAMERA_OCCLUSION_AREA_RATIO,
                VISIBLE_NEAR_CAMERA_OCCLUSION_WIDTH_RATIO,
                VISIBLE_NEAR_CAMERA_OCCLUSION_HEIGHT_RATIO,
                ASTRA_DEPTH_MAX_DISTANCE_JUMP_M,
                max(3, ASTRA_DEPTH_JUMP_CONFIRM_FRAMES, ASTRA_DEPTH_NEAR_FAR_JUMP_CONFIRM_FRAMES),
                ASTRA_DEPTH_VISION_DEDUPE_SEC * 1000.0,
            )
        logger.info(
            "Track memory: disabled in request_0513; target retention is handled by DeepSORT/IdentityBank"
        )
        logger.info(
            "Lost-target control: direction=last_reliable_center "
            "motion=in_place_rotation fixed_until_search_end=True"
        )
        logger.info("Vision engine: %s", self._vision_engine)
        if not MODULE_VISION_ENABLE:
            raise RuntimeError("当前 request_0513_modular.py 仍依赖视觉主循环，vision 模块不能关闭")

        self._sensor_runtime.start()

        model_path_abs = os.path.abspath(self._model_path)
        if self._vision_engine == "rknn" and not os.path.exists(model_path_abs):
            logger.warning("RKNN YOLO model not found yet: %s", model_path_abs)
        elif not os.path.exists(model_path_abs):
            raise RuntimeError(
                f"模型文件不存在: {model_path_abs} （当前工作目录: {os.getcwd()}）。"
                f"可传参: python3 track_person_follow_bizhang_car.py <模型路径>"
            )
        logger.info("RKNN vision model path: %s", model_path_abs)
        if RKNNVisionPipeline is None or RKNNVisionConfig is None:
            raise RuntimeError(f"RKNN vision import failed: {_RKNN_VISION_IMPORT_ERROR}")
        reid_model_abs = os.path.abspath(VISION_REID_MODEL_PATH)
        if VISION_REID_ENABLE and not os.path.exists(reid_model_abs):
            logger.warning("RKNN ReID model not found yet: %s", reid_model_abs)
        cfg = RKNNVisionConfig.from_env()
        cfg = RKNNVisionConfig(
            **{
                **cfg.__dict__,
                "yolo_model_path": model_path_abs,
                "reid_model_path": reid_model_abs,
                "reid_enable": VISION_REID_ENABLE,
                "frame_width": VISION_FRAME_WIDTH,
                "frame_height": VISION_FRAME_HEIGHT,
                "hfov_deg": VISION_HFOV_DEG,
            }
        )
        logger.info(
            "ReID appearance fusion: osnet=true color_hsv=%s weight=%.2f",
            cfg.reid_color_fusion_enable,
            cfg.reid_color_fusion_weight,
        )
        self._rknn_pipeline = RKNNVisionPipeline(cfg, logger=logger)
        if RKNN_DIRECTION_WORKERS > 0 and FOLLOW_DIRECTION_HISTORY_ENABLE:
            if DirectionInferencePool is None or YOLO11Config is None:
                raise RuntimeError(f"direction worker import failed: {_RKNN_DIRECTION_IMPORT_ERROR}")
            direction_config = YOLO11Config(
                model_path=model_path_abs,
                input_size=cfg.yolo_input_size,
                conf_threshold=min(float(cfg.conf_threshold), float(cfg.search_diagnostic_conf_threshold)),
                search_diagnostic_conf_threshold=min(
                    float(cfg.conf_threshold), float(cfg.search_diagnostic_conf_threshold)
                ),
                search_diagnostic_class_id=cfg.person_class_id,
                nms_threshold=cfg.nms_threshold,
                num_classes=cfg.yolo_num_classes,
                input_format=cfg.yolo_input_format,
                output_box_format=cfg.yolo_box_format,
                target=cfg.target,
                core_mask=cfg.core_mask,
                backend=cfg.backend,
            )
            self._direction_pool = DirectionInferencePool(
                direction_config,
                workers=RKNN_DIRECTION_WORKERS,
                queue_size=RKNN_DIRECTION_QUEUE_SIZE,
                min_area_ratio=SEARCH_EVIDENCE_MIN_AREA_RATIO,
                max_area_ratio=SEARCH_EVIDENCE_MAX_AREA_RATIO,
                logger=logger,
            )
            logger.info(
                "Capture direction workers enabled: workers=%d queue=%s",
                RKNN_DIRECTION_WORKERS,
                "unbounded" if RKNN_DIRECTION_QUEUE_SIZE == 0 else str(RKNN_DIRECTION_QUEUE_SIZE),
            )
        elif RKNN_DIRECTION_WORKERS > 0:
            logger.info("Capture direction workers disabled: FOLLOW_DIRECTION_HISTORY_ENABLE=0")
        logger.info(
            "RKNN vision ready: det_model=%s reid_model=%s reid=%s frame=%dx%d hfov=%.1f",
            model_path_abs,
            reid_model_abs,
            VISION_REID_ENABLE,
            VISION_FRAME_WIDTH,
            VISION_FRAME_HEIGHT,
            VISION_HFOV_DEG,
        )
        logger.info(
            "RKNN camera source: enabled=%s mode=%s device=%s request=%dx%d@%.1ffps fourcc=%s raw_output=%s latest_drain_max=%d",
            RKNN_CAMERA_ENABLE,
            RKNN_CAMERA_CAPTURE_MODE,
            RKNN_CAMERA_DEVICE,
            RKNN_CAMERA_WIDTH,
            RKNN_CAMERA_HEIGHT,
            RKNN_CAMERA_FPS,
            RKNN_CAMERA_FOURCC or "default",
            RKNN_CAMERA_RAW_OUTPUT or "disabled",
            RKNN_CAMERA_LATEST_DRAIN_MAX,
        )

        self._bunker_runtime = BunkerHazardRuntime(BUNKER_RUNTIME_CONFIG, logger=logger)
        self._bunker_runtime.start()

        # Motor command state (new commands can preempt old commands).
        self.current_command = None  # 当前执行的命令（ACTION_*）
        self.command_start_time = None  # 命令开始时间
        self.command_lock = threading.Lock()  # 保护命令状态的锁
        self._last_action_switch_ts = 0.0
        self._last_action_switch_frame = -1
        self._motor_backend = MssdMotorBackend(MSSD_MOTOR_CONFIG, logger=logger)
        self.motor_io_lock = self._motor_backend.io_lock

        # 跟踪状态
        self.last_person_center_x = None
        self.search_state = "none"
        self.search_direction = None
        self.lost_confirm_frames = 0
        self.last_action_frame = -1
        self.frame_index = 0
        # 单人场景下 ReID 暂时无 uid 时的几何连续性缓存。该兜底不参与
        # 多人选人，只用于保持已经锁定的人，避免一两帧低质量框触发搜索。
        self._single_person_geometry_bbox = None
        self._single_person_geometry_frame = -1
        self._single_person_geometry_track_id = None
        self._single_person_geometry_streak = 0
        self._single_person_geometry_anchor_uid = None
        self._single_person_geometry_unsteerable_uid = None
        self._confirmed_search_reacquire_uid = None
        self._confirmed_search_reacquire_track_id = None
        self._confirmed_search_reacquire_bbox = None
        self._confirmed_search_reacquire_last_frame = -1
        self._confirmed_search_reacquire_streak = 0
        self._search_candidate_gate = SearchCandidateGate(
            SearchCandidateGateConfig(
                enabled=SEARCH_EVIDENCE_GATE_ENABLE,
                formal_min_score=CONFIDENCE_THRESHOLD,
                probe_min_score=SEARCH_EVIDENCE_PROBE_MIN_SCORE,
                min_area_ratio=SEARCH_EVIDENCE_MIN_AREA_RATIO,
                max_area_ratio=SEARCH_EVIDENCE_MAX_AREA_RATIO,
                probe_confirm_frames=SEARCH_EVIDENCE_PROBE_CONFIRM_FRAMES,
                hold_frames=SEARCH_EVIDENCE_HOLD_FRAMES,
                max_hold_sec=SEARCH_EVIDENCE_MAX_HOLD_SEC,
                consistency_iou=SEARCH_EVIDENCE_CONSISTENCY_IOU,
                blocked_reset_missing_frames=SEARCH_EVIDENCE_REARM_MISSING_FRAMES,
            )
        )
        self._search_evidence_pause_current_frame = False
        self._search_evidence_observation_active = False
        self._search_evidence_observation_source = "none"
        self._search_evidence_observation_deadline = 0.0
        self._search_geometry_reacquire_track_id = None
        self._search_geometry_reacquire_last_frame = -1
        self._search_geometry_reacquire_frames = 0
        self._search_geometry_reacquire_bbox = None
        # 已绑定目标距离过近、人体框占满画面时，框中心不再可信，不能送入
        # 转向 PID；但真实检测仍在时也不能按“丢失目标”启动搜索。
        self._visible_unsteerable_uid = None
        self._visible_unsteerable_track_id = None
        self._visible_unsteerable_last_ts = None
        self._visible_unsteerable_bbox = None
        self._visible_unsteerable_recovery_frames = 0
        self.action_cooldown = ACTION_COOLDOWN
        self.search_cooldown = SEARCH_COOLDOWN
        self.is_forwarding = False  # 是否正在持续前进状态
        self.person_detected_flag = False  # 检测到人的标志，用于动作执行线程立即停止
        self.forward_1m_state = None  # 前进1米状态：None=未开始, "rotating"=正在旋转找方向, "forwarding"=正在前进, "completed"=已完成
        self.forward_1m_start_distance = None  # 开始前进1米时的距离传感器距离
        # 5 帧内未检测到人时为 True，主循环不发送停止，保持原运动
        self._waiting_lost_confirm = False
        self._explicit_stop_requested = False  # 仅障碍/人在停止距离等明确原因时为 True 才发 STOP
        # 根据距离调节的前进速度百分比（_process_detections 中设置，动作 runtime 使用）
        self._current_forward_percent = 0
        self._forward_speed_latched_percent: Optional[int] = None
        self._last_forward_speed_hysteresis_log_ts = 0.0
        self._current_steer_base_percent = 0
        self._current_steer_inner_ratio_percent = VISIBLE_STEER_INNER_RATIO_PERCENT
        self._current_steer_outer_ratio_percent = VISIBLE_STEER_OUTER_RATIO_PERCENT
        self._current_steer_correction_rpm = 0
        self._current_steer_limit_reason = "none"
        self._vision_control_state = "lost_confirming"
        # Only the vision loop updates this signed yaw correction. The 30Hz
        # Depth loop may reuse it briefly while changing longitudinal speed.
        self._last_vision_correction_rpm = 0
        self._last_vision_correction_at = 0.0
        self._last_vision_correction_target_id: Optional[int] = None
        self._last_vision_control_frame_index = -1
        self._last_vision_control_ts = 0.0
        self._last_depth30_dedupe_log_frame = -1
        self._last_depth30_preserve_yaw_log_ts = 0.0
        # 差速转向力度：由动作线程在切入旋转时按「上一动是否旋转」设定（见 ROTATE_TURN_PERCENT_*）
        self._current_rotate_turn_percent = ROTATE_TURN_PERCENT_FROM_FORWARD
        self._current_rotate_raw_target = MOTOR_ROTATE_RAW_TARGET
        self._current_rotate_raw_source = "default"
        # 完全丢失目标时使用脉冲扫描；近距离仍可见人物的原地居中会逐帧
        # 关闭该标志，改走短失效时间的连续低速闭环。
        self._current_rotate_pulse_enabled = True
        # 单次旋转时长到点后置 True，下一拍若再发旋转则用 CHAIN 力度（无需与上一指令无缝衔接）
        self._rotate_follows_previous_rotate = False
        # 保留首人跟踪器参数作备用；当前主流程由 FollowSafetyController 结合 ReID stable id 锁定目标。
        self._first_person_tracker = FirstPersonTracker(
            min_iou_threshold=0.08,
            max_lost_frames=3,
            frames_allow_new_id=600,
            max_center_dist_ratio=0.5,
            min_size_sim_for_center_match=0.1,
            velocity_smooth=0.4,
            max_reacquire_dist_ratio=0.9,
        )

        # 运行状态
        self.running = True
        self._runtime_shutdown_requested = False
        # 需容纳多步动作（如 [STOP, 旋转]）；maxsize=1 会丢第二个，表现为旋转怪/不转
        self.action_queue = queue.Queue(maxsize=32)
        self.action_queue_lock = threading.Lock()
        self.stop_action_execution = False
        self.action_stop_event = threading.Event()
        self.action_thread = None
        # 视觉主循环与30Hz纵向监督共用控制器状态，必须串行决策；电机仍只由
        # action_runtime线程写入，纵向线程只替换latest-action队列。
        self._control_update_lock = threading.RLock()
        self._lateral_intent_store = LateralIntentStore()
        self._lateral_intent_stop_event = threading.Event()
        self._lateral_intent_thread = None
        self._lateral_direction_intent = "hold"
        self._lateral_intent_last_target_id: Optional[int] = None
        self._lateral_intent_last_mode = "none"
        self._lateral_intent_last_sequence = -1
        self._lateral_intent_last_correction_rpm = 0
        self._lateral_intent_last_tick_ts = 0.0
        self._lateral_intent_last_publish_ts = 0.0
        self._lateral_intent_last_log_ts = 0.0
        self._lateral_intent_last_log_direction = "hold"
        self._lateral_intent_last_expired_sequence = -1
        self._lateral_intent_owned_frame = -1
        self._last_lateral_yaw_sign_mismatch_log_ts = 0.0
        self._longitudinal_context_lock = threading.Lock()
        self._longitudinal_context = None
        self._longitudinal_stop_event = threading.Event()
        self._longitudinal_thread = None
        self._last_longitudinal_stale_log_ts = 0.0
        # 动作线程“硬停”检查限频（避免每 10ms 读一次距离传感器导致抖动/卡顿）
        self._hard_stop_check_interval_sec = 0.05
        self._last_hard_stop_check_ts = 0.0
        # 动作线程“轮速反馈”检查限频：避免每 10ms 都发一次 0x03 读寄存器
        self._motor_feedback_poll_interval_sec = MOTOR_FEEDBACK_POLL_INTERVAL_SEC
        self._last_motor_feedback_check_ts = 0.0
        # 防滑锁轮保持态：触发后仅允许“旋转”或“前进且速度>0”来解除
        self._brake_hold_active = False
        self._brake_hold_stop_mode: Optional[str] = None
        self._brake_hold_label = "brake"
        self._brake_hold_refresh_interval_sec = BRAKE_HOLD_REFRESH_INTERVAL_SEC
        self._last_brake_hold_send_ts = 0.0
        # 脉冲旋转停顿：旋转到点后强制停一会儿再允许下一次旋转
        self._rotate_pause_until_ts = 0.0
        self._rotate_observe_until_frame = -1
        # Each lost-target search owns a distinct settle gate. A gate from an
        # earlier search must never delay a later search after target reacquire.
        self._search_epoch = 0
        self._rotate_settle_search_epoch = -1
        self._rotate_settle_pending = False
        self._rotate_settle_started_monotonic = 0.0
        self._rotate_settle_quiet_started_monotonic = 0.0
        self._rotate_settle_completed_monotonic = 0.0
        self._rotate_settle_completion_source = "idle"
        self._last_rotate_settle_log_ts = 0.0
        self._rotate_observation_last_defer_monotonic = 0.0
        self._last_rotate_pause_log_ts = 0.0
        self._last_rotate_pulse_active_log_ts = 0.0
        self._last_rotate_pulse_refresh_ts = 0.0
        self._last_rotate_hold_refresh_log_ts = 0.0
        self._last_rotate_visual_refresh_ts = 0.0
        self._last_rotate_hold_target_send_ts = 0.0
        self._last_rotate_dispatch_log_ts = 0.0
        # 最近一次旋转结束时间（用于判定短时间内再次旋转仍算“连续旋转”）
        self._last_rotate_end_ts = 0.0
        # 上一帧实际入队动作（用于搜索冷却等逻辑区分“前进保持”与“旋转脉冲”）
        self._last_dispatched_action = None  # type: Optional[int]
        self._last_action_queue_replace_ts = 0.0
        self._last_action_queue_replace_frame = -1
        self._action_queue_seq = 0
        self._last_action_queue_seq = 0
        self._last_action_queue_reason = ""
        # Provenance of the decision that created the current motor command.
        self._last_command_source_module = "unknown"
        self._last_command_control_frame = -1
        self._last_decision_capture_frame = -1
        self._last_command_capture_frame = -1
        self._last_command_capture_timestamp = 0.0
        self._last_action_queue_actions: List[int] = []
        self._last_action_queue_signature: Optional[Tuple] = None
        self._last_redundant_action_skip_log_ts = 0.0
        self._last_direct_stop_reason = ""
        self._last_direct_stop_ts = 0.0
        self._last_redundant_direct_stop_log_ts = 0.0
        self._last_distance_stop_log_key = None
        self._last_distance_hard_stop_log_ts = 0.0
        self._last_frame_distance_state = None
        self._last_tracker_action_kind = None  # type: Optional[int]
        self._last_tracker_action_change_ts = 0.0
        self._last_tracker_action_change_frame = -1
        self._last_forward_like_refresh_ts = 0.0
        self._last_motor_dispatch_ts = 0.0
        self._last_motor_dispatch_action = None  # type: Optional[int]
        # 仅转向结束停：下一拍 ACTION_STOP 走软停（0%%），不抢紧急 brake
        self._use_soft_stop_next = False
        self._soft_stop_active = False
        self._follow_controller = FollowSafetyController(
            FollowPolicyConfig(
                lost_confirm_frames=LOST_CONFIRM_FRAMES,
                lost_confirm_sec=FOLLOW_LOST_CONFIRM_SEC,
                stale_direction_recovery_enable=FOLLOW_STALE_DIRECTION_RECOVERY_ENABLE,
                direction_history_enable=FOLLOW_DIRECTION_HISTORY_ENABLE,
                stale_direction_observe_frames=FOLLOW_STALE_DIRECTION_OBSERVE_FRAMES,
                stale_direction_observe_max_sec=FOLLOW_STALE_DIRECTION_OBSERVE_MAX_SEC,
                stale_direction_settle_yaw_rate_dps=FOLLOW_STALE_DIRECTION_SETTLE_YAW_RATE_DPS,
                stale_direction_probe_enable=FOLLOW_STALE_DIRECTION_PROBE_ENABLE,
                stale_direction_probe_angle_deg=FOLLOW_STALE_DIRECTION_PROBE_ANGLE_DEG,
                stale_direction_probe_observe_frames=FOLLOW_STALE_DIRECTION_PROBE_OBSERVE_FRAMES,
                stale_direction_probe_return_tolerance_deg=FOLLOW_STALE_DIRECTION_PROBE_RETURN_TOLERANCE_DEG,
                steer_min_hold_sec=FOLLOW_STEER_MIN_HOLD_SEC,
                steer_lost_hold_frames=FOLLOW_STEER_LOST_HOLD_FRAMES,
                steer_lost_hold_max_sec=FOLLOW_STEER_LOST_HOLD_MAX_SEC,
                lost_forward_hold_rpm=FOLLOW_LOST_FORWARD_HOLD_RPM,
                lost_forward_hold_max_sec=FOLLOW_LOST_FORWARD_HOLD_MAX_SEC,
                lost_forward_hold_min_distance_m=FOLLOW_LOST_FORWARD_HOLD_MIN_DISTANCE_M,
                depth_medium_confidence_hold_sec=FOLLOW_DEPTH_MEDIUM_CONFIDENCE_HOLD_SEC,
                depth_medium_confidence_rpm=FOLLOW_DEPTH_MEDIUM_CONFIDENCE_RPM,
                depth_recovery_stage1_sec=FOLLOW_DEPTH_RECOVERY_STAGE1_SEC,
                depth_recovery_stage2_sec=FOLLOW_DEPTH_RECOVERY_STAGE2_SEC,
                depth_recovery_stage1_rpm=FOLLOW_DEPTH_RECOVERY_STAGE1_RPM,
                depth_recovery_stage2_rpm=FOLLOW_DEPTH_RECOVERY_STAGE2_RPM,
                action_cooldown=ACTION_COOLDOWN,
                search_cooldown=SEARCH_COOLDOWN,
                target_distance_m=TARGET_DISTANCE,
                target_distance_release_m=TARGET_DISTANCE_RELEASE_M,
                target_distance_release_hold_sec=TARGET_DISTANCE_RELEASE_HOLD_SEC,
                target_distance_release_confirm_frames=TARGET_DISTANCE_RELEASE_CONFIRM_FRAMES,
                target_distance_release_visual_shrink_ratio=TARGET_DISTANCE_RELEASE_VISUAL_SHRINK_RATIO,
                target_distance_release_visual_edge_margin_ratio=TARGET_DISTANCE_RELEASE_VISUAL_EDGE_MARGIN_RATIO,
                brake_distance_m=FOLLOW_BRAKE_DISTANCE_M,
                distance_parking_enable=DISTANCE_PARKING_ENABLE,
                reverse_enable=FOLLOW_REVERSE_ENABLE,
                near_distance_rotate_only_enable=FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_ENABLE,
                near_distance_rotate_only_distance_m=FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_DISTANCE_M,
                near_distance_rotation_only_max_rpm=FOLLOW_NEAR_DISTANCE_ROTATION_ONLY_MAX_RPM,
                reverse_start_distance_m=FOLLOW_REVERSE_START_DISTANCE_M,
                reverse_immediate_distance_m=FOLLOW_REVERSE_IMMEDIATE_DISTANCE_M,
                reverse_stop_distance_m=FOLLOW_REVERSE_STOP_DISTANCE_M,
                reverse_full_speed_distance_m=FOLLOW_REVERSE_FULL_SPEED_DISTANCE_M,
                reverse_min_rpm=FOLLOW_REVERSE_MIN_RPM,
                reverse_max_rpm=FOLLOW_REVERSE_MAX_RPM,
                reverse_feedforward_floor_rpm=FOLLOW_REVERSE_FEEDFORWARD_FLOOR_RPM,
                reverse_feedforward_gain_rpm_per_m_s=FOLLOW_REVERSE_FEEDFORWARD_GAIN_RPM_PER_M_S,
                reverse_runtime_cap_rpm=FOLLOW_REVERSE_RUNTIME_CAP_RPM,
                reverse_approach_speed_filter_alpha=FOLLOW_REVERSE_APPROACH_SPEED_FILTER_ALPHA,
                reverse_min_approach_delta_m=FOLLOW_REVERSE_MIN_APPROACH_DELTA_M,
                reverse_confirm_frames=FOLLOW_REVERSE_CONFIRM_FRAMES,
                reverse_radar_max_age_sec=FOLLOW_REVERSE_RADAR_MAX_AGE_SEC,
                reverse_distance_missing_hold_sec=FOLLOW_REVERSE_DISTANCE_MISSING_HOLD_SEC,
                reverse_visual_guard_area_ratio=FOLLOW_REVERSE_VISUAL_GUARD_AREA_RATIO,
                reverse_visual_guard_height_ratio=FOLLOW_REVERSE_VISUAL_GUARD_HEIGHT_RATIO,
                reverse_visual_guard_growth_ratio=FOLLOW_REVERSE_VISUAL_GUARD_GROWTH_RATIO,
                reverse_visual_guard_growth_min_area_ratio=FOLLOW_REVERSE_VISUAL_GUARD_GROWTH_MIN_AREA_RATIO,
                reverse_visual_guard_max_distance_m=FOLLOW_REVERSE_VISUAL_GUARD_MAX_DISTANCE_M,
                reverse_visual_guard_rpm=FOLLOW_REVERSE_VISUAL_GUARD_RPM,
                forward_start_distance_m=FOLLOW_FORWARD_START_DISTANCE_M,
                forward_stop_distance_m=FOLLOW_FORWARD_STOP_DISTANCE_M,
                obstacle_distance_m=MMWAVE_OBSTACLE_THRESHOLD,
                min_forward_percent=MIN_FORWARD_PERCENT,
                max_forward_percent=MAX_FORWARD_PERCENT,
                forward_speed_le_1_3_percent=FORWARD_SPEED_LE_1_3_PERCENT,
                forward_speed_le_1_7_percent=FORWARD_SPEED_LE_1_7_PERCENT,
                forward_speed_le_2_1_percent=FORWARD_SPEED_LE_2_1_PERCENT,
                forward_speed_le_2_6_percent=FORWARD_SPEED_LE_2_6_PERCENT,
                forward_speed_le_3_2_percent=FORWARD_SPEED_LE_3_2_PERCENT,
                forward_speed_le_3_8_percent=FORWARD_SPEED_LE_3_8_PERCENT,
                forward_speed_le_4_5_percent=FORWARD_SPEED_LE_4_5_PERCENT,
                forward_speed_far_percent=FORWARD_SPEED_FAR_PERCENT,
                distance_missing_forward_percent=DISTANCE_MISSING_FORWARD_PERCENT,
                mmwave_hold_forward_percent=VISION_MMWAVE_HOLD_FORWARD_PERCENT,
                mmwave_hold_decel_step_percent=VISION_MMWAVE_HOLD_DECEL_STEP_PERCENT,
                mmwave_hold_recover_step_percent=VISION_MMWAVE_HOLD_RECOVER_STEP_PERCENT,
                mmwave_fusion_low_confidence_forward_percent=VISION_MMWAVE_FUSION_LOW_CONFIDENCE_FORWARD_PERCENT,
                forward_min_rpm=FORWARD_MIN_RPM,
                forward_max_rpm=FORWARD_MAX_RPM,
                forward_curve_max_distance_m=FORWARD_CURVE_MAX_DISTANCE_M,
                forward_curve_exponent=FORWARD_CURVE_EXPONENT,
                distance_pid_enable=DISTANCE_PID_ENABLE,
                distance_pid_kp_rpm_per_m=DISTANCE_PID_KP_RPM_PER_M,
                distance_pid_ki_rpm_per_m_s=DISTANCE_PID_KI_RPM_PER_M_S,
                distance_pid_kd_rpm_s_per_m=DISTANCE_PID_KD_RPM_S_PER_M,
                distance_pid_integral_limit_m_s=DISTANCE_PID_INTEGRAL_LIMIT_M_S,
                distance_pid_deadband_m=DISTANCE_PID_DEADBAND_M,
                distance_pid_derivative_filter_alpha=DISTANCE_PID_DERIVATIVE_FILTER_ALPHA,
                distance_pid_max_measurement_jump_m=DISTANCE_PID_MAX_MEASUREMENT_JUMP_M,
                distance_pid_output_rise_rpm_per_sec=DISTANCE_PID_OUTPUT_RISE_RPM_PER_SEC,
                distance_pid_output_fall_rpm_per_sec=DISTANCE_PID_OUTPUT_FALL_RPM_PER_SEC,
                center_left_ratio=FOLLOW_CENTER_LEFT_RATIO,
                center_right_ratio=FOLLOW_CENTER_RIGHT_RATIO,
                center_deadzone_ratio=FOLLOW_CENTER_DEADZONE_RATIO,
                search_candidate_untracked_min_score=SEARCH_CANDIDATE_UNTRACKED_MIN_SCORE,
                search_candidate_approach_margin_ratio=SEARCH_CANDIDATE_APPROACH_MARGIN_RATIO,
                steer_enter_left_ratio=FOLLOW_STEER_ENTER_LEFT_RATIO,
                steer_enter_right_ratio=FOLLOW_STEER_ENTER_RIGHT_RATIO,
                steer_release_left_ratio=FOLLOW_STEER_RELEASE_LEFT_RATIO,
                steer_release_right_ratio=FOLLOW_STEER_RELEASE_RIGHT_RATIO,
                visible_rotate_left_ratio=FOLLOW_VISIBLE_ROTATE_LEFT_RATIO,
                visible_rotate_right_ratio=FOLLOW_VISIBLE_ROTATE_RIGHT_RATIO,
                use_vertical_center_gate=FOLLOW_USE_VERTICAL_CENTER_GATE,
                search_before_first_seen=FOLLOW_SEARCH_BEFORE_FIRST_PERSON,
                startup_search_delay_sec=FOLLOW_STARTUP_SEARCH_DELAY_SEC,
                search_timeout_sec=FOLLOW_SEARCH_TIMEOUT_SEC,
                search_timeout_exit_program=FOLLOW_SEARCH_TIMEOUT_EXIT_PROGRAM,
                search_revolution_deg=FOLLOW_SEARCH_REVOLUTION_DEG,
                search_revolution_feedback_stale_sec=FOLLOW_SEARCH_REVOLUTION_FEEDBACK_STALE_SEC,
                exit_on_target_loss=FOLLOW_EXIT_ON_TARGET_LOSS,
                initial_target_confirm_frames=FOLLOW_INITIAL_TARGET_CONFIRM_FRAMES,
                release_target_on_lost=FOLLOW_RELEASE_TARGET_ON_LOST,
                side_ir_blocks_rotation=SIDE_IR_BLOCKS_ROTATION,
                search_rotate_front_block_enable=SEARCH_ROTATE_FRONT_BLOCK_ENABLE,
                search_rotate_distance_block_enable=SEARCH_ROTATE_DISTANCE_BLOCK_ENABLE,
                visible_steer_fine_inner_ratio_percent=VISIBLE_STEER_FINE_INNER_RATIO_PERCENT,
                visible_steer_fine_outer_ratio_percent=VISIBLE_STEER_FINE_OUTER_RATIO_PERCENT,
                visible_steer_inner_ratio_percent=VISIBLE_STEER_INNER_RATIO_PERCENT,
                visible_steer_outer_ratio_percent=VISIBLE_STEER_OUTER_RATIO_PERCENT,
                visible_steer_strong_inner_ratio_percent=VISIBLE_STEER_STRONG_INNER_RATIO_PERCENT,
                visible_steer_strong_outer_ratio_percent=VISIBLE_STEER_STRONG_OUTER_RATIO_PERCENT,
                visible_steer_strong_margin_ratio=VISIBLE_STEER_STRONG_MARGIN_RATIO,
                visible_motion_history_frames=VISIBLE_MOTION_HISTORY_FRAMES,
                visible_motion_lookback_sec=VISIBLE_MOTION_LOOKBACK_SEC,
                visible_motion_rate_filter_alpha=VISIBLE_MOTION_RATE_FILTER_ALPHA,
                visible_motion_min_ratio=VISIBLE_MOTION_MIN_RATIO,
                visible_motion_projection_gain=VISIBLE_MOTION_PROJECTION_GAIN,
                visible_motion_strong_ratio=VISIBLE_MOTION_STRONG_RATIO,
                visible_steering_pid_enable=VISIBLE_STEERING_PID_ENABLE,
                visible_steering_pid_camera_hfov_deg=VISIBLE_STEERING_PID_CAMERA_HFOV_DEG,
                visible_steering_pid_camera_latency_sec=VISIBLE_STEERING_PID_CAMERA_LATENCY_SEC,
                visible_steering_pid_deadband_deg=VISIBLE_STEERING_PID_DEADBAND_DEG,
                visible_steering_pid_outer_kp_per_sec=VISIBLE_STEERING_PID_OUTER_KP_PER_SEC,
                visible_steering_pid_outer_kd_sec=VISIBLE_STEERING_PID_OUTER_KD_SEC,
                visible_steering_pid_target_rate_feedforward_gain=VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_GAIN,
                visible_steering_pid_target_rate_feedforward_max_dps=VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_MAX_DPS,
                visible_steering_pid_target_speed_match_max_closing_dps=VISIBLE_STEERING_PID_TARGET_SPEED_MATCH_MAX_CLOSING_DPS,
                visible_steering_pid_max_yaw_rate_dps=VISIBLE_STEERING_PID_MAX_YAW_RATE_DPS,
                visible_steering_pid_rate_kp_rpm_per_dps=VISIBLE_STEERING_PID_RATE_KP_RPM_PER_DPS,
                visible_steering_pid_rate_ki_rpm_per_deg=VISIBLE_STEERING_PID_RATE_KI_RPM_PER_DEG,
                visible_steering_pid_integral_limit_deg=VISIBLE_STEERING_PID_INTEGRAL_LIMIT_DEG,
                visible_steering_pid_max_correction_rpm=VISIBLE_STEERING_PID_MAX_CORRECTION_RPM,
                visible_steering_pid_dynamic_small_error_deg=VISIBLE_STEERING_PID_DYNAMIC_SMALL_ERROR_DEG,
                visible_steering_pid_dynamic_large_error_deg=VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_DEG,
                visible_steering_pid_dynamic_small_max_yaw_rate_dps=VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_YAW_RATE_DPS,
                visible_steering_pid_dynamic_small_max_correction_rpm=VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_CORRECTION_RPM,
                visible_steering_pid_dynamic_large_error_base_cap_rpm=VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_BASE_CAP_RPM,
                visible_steering_pid_opposite_yaw_brake_threshold_dps=VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_THRESHOLD_DPS,
                visible_steering_pid_opposite_yaw_brake_boost_rpm=VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_BOOST_RPM,
                visible_steering_pid_braking_max_correction_rpm=VISIBLE_STEERING_PID_BRAKING_MAX_CORRECTION_RPM,
                visible_steering_pid_fast_countersteer_max_correction_rpm=VISIBLE_STEERING_PID_FAST_COUNTERSTEER_MAX_CORRECTION_RPM,
                visible_steering_pid_fast_countersteer_gain_rpm_per_dps=VISIBLE_STEERING_PID_FAST_COUNTERSTEER_GAIN_RPM_PER_DPS,
                visible_steering_pid_same_direction_overspeed_threshold_dps=VISIBLE_STEERING_PID_SAME_DIRECTION_OVERSPEED_THRESHOLD_DPS,
                visible_steering_pid_same_direction_overspeed_brake_gain_rpm_per_dps=VISIBLE_STEERING_PID_SAME_DIRECTION_OVERSPEED_BRAKE_GAIN_RPM_PER_DPS,
                visible_steering_pid_visual_direction_guard_enabled=VISIBLE_STEERING_PID_VISUAL_DIRECTION_GUARD_ENABLE,
                visible_steering_pid_predictive_brake_decel_dps2=VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_DECEL_DPS2,
                visible_steering_pid_predictive_brake_margin_deg=VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_MARGIN_DEG,
                visible_steering_pid_predictive_brake_response_sec=VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_RESPONSE_SEC,
                visible_steering_pid_min_effective_error_deg=VISIBLE_STEERING_PID_MIN_EFFECTIVE_ERROR_DEG,
                visible_steering_pid_min_effective_correction_rpm=VISIBLE_STEERING_PID_MIN_EFFECTIVE_CORRECTION_RPM,
                visible_steering_pid_mechanical_tier2_error_deg=VISIBLE_STEERING_PID_MECHANICAL_TIER2_ERROR_DEG,
                visible_steering_pid_mechanical_tier2_correction_rpm=VISIBLE_STEERING_PID_MECHANICAL_TIER2_CORRECTION_RPM,
                visible_steering_pid_mechanical_tier3_error_deg=VISIBLE_STEERING_PID_MECHANICAL_TIER3_ERROR_DEG,
                visible_steering_pid_mechanical_tier3_correction_rpm=VISIBLE_STEERING_PID_MECHANICAL_TIER3_CORRECTION_RPM,
                visible_steering_pid_mechanical_floor_release_ratio=VISIBLE_STEERING_PID_MECHANICAL_FLOOR_RELEASE_RATIO,
                visible_steering_pid_startup_kick_error_deg=VISIBLE_STEERING_PID_STARTUP_KICK_ERROR_DEG,
                visible_steering_pid_startup_kick_rpm=VISIBLE_STEERING_PID_STARTUP_KICK_RPM,
                visible_steering_pid_startup_kick_max_sec=VISIBLE_STEERING_PID_STARTUP_KICK_MAX_SEC,
                visible_steering_pid_startup_kick_release_yaw_rate_dps=VISIBLE_STEERING_PID_STARTUP_KICK_RELEASE_YAW_RATE_DPS,
                visible_steering_pid_active_brake_yaw_threshold_dps=VISIBLE_STEERING_PID_ACTIVE_BRAKE_YAW_THRESHOLD_DPS,
                visible_steering_pid_active_brake_min_correction_rpm=VISIBLE_STEERING_PID_ACTIVE_BRAKE_MIN_CORRECTION_RPM,
                visible_steering_pid_edge_boost_start_error_deg=VISIBLE_STEERING_PID_EDGE_BOOST_START_ERROR_DEG,
                visible_steering_pid_aggressive_inner_wheel_margin_rpm=VISIBLE_STEERING_PID_AGGRESSIVE_INNER_WHEEL_MARGIN_RPM,
                visible_steering_pid_fallback_max_correction_rpm=VISIBLE_STEERING_PID_FALLBACK_MAX_CORRECTION_RPM,
                visible_steering_pid_lost_hold_max_correction_rpm=VISIBLE_STEERING_PID_LOST_HOLD_MAX_CORRECTION_RPM,
                visible_steering_pid_left_body_deg_per_encoder_deg=VISIBLE_STEERING_PID_LEFT_BODY_DEG_PER_ENCODER_DEG,
                visible_steering_pid_right_body_deg_per_encoder_deg=VISIBLE_STEERING_PID_RIGHT_BODY_DEG_PER_ENCODER_DEG,
                visible_steering_pid_feedback_stale_sec=VISIBLE_STEERING_PID_FEEDBACK_STALE_SEC,
                visible_steering_pid_error_filter_alpha=VISIBLE_STEERING_PID_ERROR_FILTER_ALPHA,
                visible_steering_pid_derivative_filter_alpha=VISIBLE_STEERING_PID_DERIVATIVE_FILTER_ALPHA,
                visible_steering_pid_fallback_base_rpm=VISIBLE_STEERING_PID_FALLBACK_BASE_RPM,
                parked_recenter_min_rpm=PARKED_RECENTER_MIN_RPM,
                parked_recenter_max_rpm=PARKED_RECENTER_MAX_RPM,
            )
        )
        self._last_control_decision_log_key = None
        self._last_control_decision_reason = ""
        self._last_explicit_stop_reason = ""
        self._last_pid_trace_log_ts = 0.0
        self._last_target_loss_trace_frame = -1
        self._last_rknn_latest_drain_log_ts = 0.0
        self._last_vision_frame_received_ts = 0.0
        self._search_diagnostics = SearchDiagnosticsObserver(
            SearchDiagnosticsConfig(
                enabled=SEARCH_FRAME_DIAGNOSTIC_ENABLE,
                interval_sec=SEARCH_FRAME_DIAGNOSTIC_INTERVAL_SEC,
                checkpoint_deg=SEARCH_DIAGNOSTIC_CHECKPOINT_DEG,
                snapshot_enabled=SEARCH_DIAGNOSTIC_SNAPSHOT_ENABLE,
                snapshot_max=SEARCH_DIAGNOSTIC_SNAPSHOT_MAX,
                jpeg_quality=SEARCH_DIAGNOSTIC_JPEG_QUALITY,
                output_dir=SEARCH_DIAGNOSTIC_ROOT,
                person_class_id=PERSON_CLASS_ID,
                formal_confidence=CONFIDENCE_THRESHOLD,
                probe_confidence=float(
                    os.environ.get("RKNN_SEARCH_DIAGNOSTIC_CONF_THRESHOLD", "0.10")
                ),
            ),
            logger,
        )
        self._distance_runtime = DistanceRuntime(
            self,
            DISTANCE_RUNTIME_CONFIG,
            sensor_runtime=self._sensor_runtime,
            logger=logger,
        )
        self._last_person_reid_debug_by_stable_id: Dict[int, List[Dict[str, Any]]] = {}
        if not hasattr(self, "motor_io_lock"):
            self.motor_io_lock = threading.Lock()
        self._action_runtime = MotionActionRuntime(
            self,
            self._motor_backend,
            ACTION_RUNTIME_CONFIG,
            ACTION_RUNTIME_SYMBOLS,
            hard_stop_check=self._should_hard_stop_now,
            logger=logger,
        )
        # 电机串口和启动驻车必须先初始化完成。动作/编码器线程在 run()
        # 中随后启动，避免反馈读取与驻车寄存器写入同时访问 /dev/ttyS0。
        self._action_runtime_started = False
        # Shared policy flags used by action_runtime to avoid stale radar holds.
        self._mmwave_runtime_enabled = bool(MODULE_MMWAVE_ENABLE)
        self._distance_source = str(DISTANCE_SOURCE)
        self._distance_parking_enable = bool(DISTANCE_PARKING_ENABLE)

    def _should_hard_stop_now(self, action: Optional[int] = None) -> bool:
        """
        最高优先级硬停条件：
        - split 沙坑/水坑模型触发危险面积阈值
        - 任意一路 IR 触发时，无条件打断所有运动并清零双轮转速
        - 当前运行不把 Depth/毫米波距离作为 hard-stop；距离只交给纵向控制。
        该检查会被动作 runtime 在持续动作时调用，用于打断危险动作避免“帧间空窗期撞人”。
        """
        try:
            if self._bunker_runtime.current_split_state() is not None:
                return True
        except Exception:
            pass
        try:
            raw_obstacles = self._sensor_runtime.get_raw_obstacle_status()
            if raw_obstacles.front or raw_obstacles.left or raw_obstacles.right:
                logger.warning(
                    "原始红外硬停: 前=%s 左=%s 右=%s action=%s",
                    raw_obstacles.front,
                    raw_obstacles.left,
                    raw_obstacles.right,
                    ACTION_NAMES.get(action, str(action)),
                )
                return True
        except Exception:
            pass
        try:
            obstacles = self._sensor_runtime.get_obstacle_status()
            if obstacles.front or obstacles.left or obstacles.right:
                return True
        except Exception:
            # 传感器异常时不在这里强行停，避免误停；真正安全策略以主流程为准
            pass
        # Distance-based hard-stop is intentionally disabled in the active
        # board policy. Depth/radar still feed the controller's speed/reverse
        # loop; only the IR gate above may interrupt an action here.
        return False

    def _maybe_log_imu_sample(self) -> None:
        self._sensor_runtime.maybe_log_imu_sample(self.frame_index)

    def _start_action_thread(self):
        if self._action_runtime_started:
            return None
        result = self._action_runtime.start()
        self._action_runtime_started = True
        return result

    def _clear_longitudinal_context(self) -> None:
        with self._longitudinal_context_lock:
            self._longitudinal_context = None

    def _clear_lateral_intent(self, reason: str) -> None:
        previous = self._lateral_intent_store.clear()
        self._lateral_direction_intent = "hold"
        self._lateral_intent_owned_frame = -1
        if previous is not None:
            logger.info(
                "lateral_intent_clear seq=%d target=%d frame=%d reason=%s age=%.0fms",
                int(previous.sequence),
                int(previous.target_id),
                int(previous.frame_index),
                str(reason or "unspecified"),
                previous.age_sec(time.monotonic()) * 1000.0,
            )

    def _handle_stale_vision_result(
        self,
        *,
        width: int,
        result_age_sec: float,
    ) -> bool:
        """Invalidate stale yaw and enter bounded direction recovery."""
        with self._control_update_lock:
            first_gap = self._follow_controller.note_stale_visual_result(
                frame_width=int(width),
                now=time.monotonic(),
                capture_frame_id=int(getattr(self, "_active_capture_frame_id", 0)),
                capture_timestamp=float(getattr(self, "_active_capture_timestamp", 0.0)),
            )
            recovery_active = self._follow_controller.stale_direction_recovery_active
            if recovery_active:
                self.search_state = self._follow_controller.search_state
                self.search_direction = self._follow_controller.search_direction
                self.lost_confirm_frames = self._follow_controller.lost_confirm_frames
                self._vision_control_state = "direction_uncertain"
            else:
                # Search/reacquire may already own the controller state, but a
                # stale result still must revoke the last high-RPM visible
                # intent. Keeping it until TTL is what allowed frame 696 to
                # continue the old 12 RPM command after vision had timed out.
                self._vision_control_state = "stale_visual_hold"
            self._clear_lateral_intent("stale_vision_result")
            self._clear_longitudinal_context()
            self._current_forward_percent = 0
            self._current_steer_base_percent = 0
            self._current_steer_correction_rpm = 0
            self._current_steer_limit_reason = "stale_vision_result"
            self._current_rotate_raw_target = 0
            self._current_rotate_raw_source = "stale_vision_zero_yaw"
            self._current_rotate_pulse_enabled = False
            self.is_forwarding = False

            with self.command_lock:
                current_action = self.current_command
            with self.action_queue_lock:
                pending_actions = self._action_queue_snapshot_locked()
            motion_active = bool(
                current_action in MOVEMENT_ACTIONS
                or self._last_dispatched_action in MOVEMENT_ACTIONS
                or any(action in MOVEMENT_ACTIONS for action in pending_actions)
            )
            soft_zero_published = bool(
                motion_active and not bool(getattr(self, "_brake_hold_active", False))
            )
            if soft_zero_published:
                reason = "stale_vision_zero_yaw"
                self._explicit_stop_requested = False
                self._last_control_decision_reason = reason
                self._use_soft_stop_next = True
                self._replace_action_queue([ACTION_STOP], reason)

            logger.info(
                "stale_vision_direction_invalidated frame=%d age=%.1fms "
                "first_gap=%s recovery_active=%s state=%s soft_zero=%s current=%s pending=%s",
                int(self.frame_index),
                max(0.0, float(result_age_sec)) * 1000.0,
                first_gap,
                recovery_active,
                self.search_state,
                soft_zero_published,
                ACTION_NAMES.get(current_action, str(current_action)),
                self._action_names_for_log(pending_actions),
            )
            return True

    def _publish_lateral_intent_from_decision(
        self,
        *,
        width: int,
        target: Optional[PersonTarget],
        runtime_actions: List[ControlAction],
        control_source: str,
        target_steerable: bool,
        low_quality_visible: bool,
    ) -> bool:
        if control_source != "vision":
            return False
        pid_result = self._follow_controller.last_steering_pid_result
        action = runtime_actions[0] if len(runtime_actions) == 1 else None
        limited_low_quality_yaw = bool(
            low_quality_visible
            and not target_steerable
            and action is not None
            and action.kind in ("rotate_left", "rotate_right")
            and pid_result is not None
            and float(pid_result.correction_limit_rpm) > 0.0
        )
        eligible = bool(
            LATERAL_INTENT_CONTROL_ENABLE
            and VISIBLE_STEERING_PID_ENABLE
            and target is not None
            and int(width) > 0
            and pid_result is not None
            and (target_steerable or limited_low_quality_yaw)
            and (not low_quality_visible or limited_low_quality_yaw)
            and self.search_state == "none"
            and self._vision_control_state.startswith("target_visible")
            and not self._explicit_stop_requested
            and not self._runtime_shutdown_requested
        )
        if not eligible or action is None:
            self._clear_lateral_intent("vision_not_eligible")
            return False
        if action.kind in ("rotate_left", "rotate_right"):
            mode = "yaw_only"
        elif action.kind == "backward":
            mode = "reverse"
        elif action.kind in ("forward", "steer_left", "steer_right"):
            mode = "forward"
        else:
            self._clear_lateral_intent("vision_non_motion_action")
            return False

        cx, _cy = target.center
        x_ratio = max(0.0, min(1.0, float(cx) / float(max(1, int(width)))))
        motion_samples = getattr(self._follow_controller, "_visible_motion_samples", [])
        motion_dx_ratio = 0.0
        if len(motion_samples) >= 2:
            motion_dx_ratio = float(motion_samples[-1][2]) - float(motion_samples[-2][2])
        target_rate = (
            float(pid_result.target_image_rate_dps)
            if bool(pid_result.target_rate_valid)
            else None
        )
        now = time.monotonic()
        published = self._lateral_intent_store.publish(
            LateralControlIntent(
                sequence=0,
                target_id=int(target.track_id),
                frame_index=int(self.frame_index),
                published_at=now,
                valid_until=now + LATERAL_INTENT_TTL_SEC,
                x_ratio=x_ratio,
                motion_dx_ratio=motion_dx_ratio,
                target_image_rate_dps=target_rate,
                mode=mode,
                base_percent=max(0, int(action.speed_percent)),
                base_rpm=max(0, int(pid_result.base_rpm)),
                initial_correction_rpm=int(pid_result.correction_rpm),
                correction_limit_rpm=max(0.0, float(pid_result.correction_limit_rpm)),
                confidence=max(0.0, min(1.0, float(target.confidence))),
                bbox_quality=(
                    "limited"
                    if low_quality_visible or not target_steerable
                    else "reliable"
                ),
                reason=str(self._last_control_decision_reason or action.reason or "vision"),
                capture_frame_id=int(getattr(self, "_last_command_capture_frame", 0)),
                capture_timestamp=float(getattr(self, "_last_command_capture_timestamp", 0.0)),
                decision_capture_frame_id=int(getattr(self, "_active_capture_frame_id", 0)),
            )
        )
        self._lateral_intent_owned_frame = int(self.frame_index)
        logger.debug(
            "lateral_intent_publish seq=%d control_frame_id=%d evidence_capture_frame_id=%d decision_capture_frame_id=%d target=%d mode=%s x=%.3f "
            "rate=%s base=%d/%drpm limit=%.1frpm confidence=%.2f quality=%s reason=%s",
            published.sequence,
            published.frame_index,
            published.capture_frame_id,
            published.decision_capture_frame_id,
            published.target_id,
            published.mode,
            published.x_ratio,
            "none" if target_rate is None else "%+.2f" % target_rate,
            published.base_percent,
            published.base_rpm,
            published.correction_limit_rpm,
            published.confidence,
            published.bbox_quality,
            published.reason,
        )
        return True

    def _start_lateral_intent_thread(self) -> None:
        if not LATERAL_INTENT_CONTROL_ENABLE or not VISIBLE_STEERING_PID_ENABLE:
            logger.info("30Hz横向意图控制未启用")
            return
        if not hasattr(self, "_lateral_intent_store"):
            # Some hardware-lifecycle tests construct a minimal tracker with
            # __new__ and intentionally skip normal control-state init.
            logger.info("横向意图状态未初始化，跳过线程启动")
            return
        if self._lateral_intent_thread is not None and self._lateral_intent_thread.is_alive():
            return
        self._lateral_intent_stop_event.clear()
        self._lateral_intent_thread = threading.Thread(
            target=self._lateral_intent_control_loop,
            name="lateral-intent-control",
            daemon=True,
        )
        self._lateral_intent_thread.start()
        logger.info(
            "横向意图控制已启动: rate=%.1fHz ttl=%.0fms motor_publish=%.0fms",
            LATERAL_INTENT_CONTROL_RATE_HZ,
            LATERAL_INTENT_TTL_SEC * 1000.0,
            LATERAL_INTENT_MOTOR_PUBLISH_INTERVAL_SEC * 1000.0,
        )

    def _stop_lateral_intent_thread(self) -> None:
        stop_event = getattr(self, "_lateral_intent_stop_event", None)
        if stop_event is None:
            return
        stop_event.set()
        thread = getattr(self, "_lateral_intent_thread", None)
        if thread is not None:
            thread.join(timeout=1.0)
            if not thread.is_alive():
                self._lateral_intent_thread = None
        if hasattr(self, "_lateral_intent_store"):
            self._clear_lateral_intent("runtime_stop")

    def _lateral_intent_action(self, mode: str, correction_rpm: int) -> Optional[int]:
        correction = int(correction_rpm)
        if mode == "yaw_only":
            if correction > 0:
                return ACTION_ROTATE_RIGHT
            if correction < 0:
                return ACTION_ROTATE_LEFT
            return ACTION_STOP
        if mode == "reverse":
            return ACTION_BACKWARD
        if mode == "forward":
            if correction > 0:
                return ACTION_STEER_RIGHT
            if correction < 0:
                return ACTION_STEER_LEFT
            return ACTION_FORWARD
        return None

    def _service_lateral_intent(self, now: float) -> None:
        intent = self._lateral_intent_store.snapshot()
        if intent is None:
            return
        if not intent.valid(now):
            self._lateral_direction_intent = "hold"
            if self._lateral_intent_last_expired_sequence != int(intent.sequence):
                self._lateral_intent_last_expired_sequence = int(intent.sequence)
                logger.info(
                    "lateral_intent_expired seq=%d frame=%d target=%d age=%.0fms ttl=%.0fms "
                    "action=stop_refresh_only recognizer_owns_lost_transition=True",
                    intent.sequence,
                    intent.frame_index,
                    intent.target_id,
                    intent.age_sec(now) * 1000.0,
                    LATERAL_INTENT_TTL_SEC * 1000.0,
                )
            return

        with self._control_update_lock:
            current = self._lateral_intent_store.snapshot()
            active_target_id = getattr(self._follow_controller, "active_target_id", None)
            if (
                current is None
                or int(current.sequence) != int(intent.sequence)
                or active_target_id is None
                or int(active_target_id) != int(intent.target_id)
                or self.search_state != "none"
                or not self._vision_control_state.startswith("target_visible")
                or self._explicit_stop_requested
                or self._runtime_shutdown_requested
                or not self.running
            ):
                return
            feedback = self._action_runtime.get_steering_feedback()
            projected_x, projection_sec = intent.projected_x_ratio(
                now,
                camera_hfov_deg=VISIBLE_STEERING_PID_CAMERA_HFOV_DEG,
                max_projection_sec=LATERAL_INTENT_MAX_PROJECTION_SEC,
                max_projection_ratio=LATERAL_INTENT_MAX_PROJECTION_RATIO,
            )
            new_sequence = int(intent.sequence) != int(self._lateral_intent_last_sequence)
            if new_sequence:
                # The vision decision already advanced the PID for this sample.
                # Reuse that output on its first control tick so one camera frame
                # cannot update the PID state twice.
                result = self._follow_controller.last_steering_pid_result
                requested = int(intent.initial_correction_rpm)
            else:
                refresh_pid = (
                    self._follow_controller.refresh_parked_lateral_pid
                    if intent.mode == "yaw_only"
                    else self._follow_controller.refresh_visible_lateral_pid
                )
                result = refresh_pid(
                    x_ratio=projected_x,
                    base_rpm=int(intent.base_rpm),
                    feedback=feedback,
                    now=now,
                    motion_dx_ratio=float(intent.motion_dx_ratio),
                    target_image_rate_dps=intent.target_image_rate_dps,
                    max_correction_rpm=float(intent.correction_limit_rpm),
                    visual_age_sec=(
                        None
                        if float(intent.capture_timestamp) <= 0.0
                        else max(0.0, now - float(intent.capture_timestamp))
                    ),
                )
                requested = int(result.correction_rpm)
            if intent.mode == "yaw_only" and requested != 0:
                target_side = 1 if projected_x > 0.5 else -1 if projected_x < 0.5 else 0
                if target_side != 0 and requested * target_side > 0:
                    requested = int(
                        target_side
                        * max(PARKED_RECENTER_MIN_RPM, min(PARKED_RECENTER_MAX_RPM, abs(requested)))
                    )

            if (
                self._lateral_intent_last_target_id != int(intent.target_id)
                or self._lateral_intent_last_tick_ts <= 0.0
            ):
                previous = 0
            else:
                previous = int(self._lateral_intent_last_correction_rpm)
            dt = (
                1.0 / LATERAL_INTENT_CONTROL_RATE_HZ
                if self._lateral_intent_last_tick_ts <= 0.0
                else max(0.0, min(0.20, now - self._lateral_intent_last_tick_ts))
            )
            correction = slew_signed_rpm(
                previous,
                requested,
                dt,
                rise_rpm_per_sec=LATERAL_INTENT_RISE_RPM_PER_SEC,
                brake_rpm_per_sec=LATERAL_INTENT_BRAKE_RPM_PER_SEC,
            )
            direction = "right" if correction > 0 else "left" if correction < 0 else "hold"
            action = self._lateral_intent_action(intent.mode, correction)
            if action is None:
                return

            # Keep hardware-direction anomalies visible without silently
            # changing the calibrated rotate_left/rotate_right mapping. A
            # brief opposite yaw can be inertia, but a sustained/significant
            # conflict must be traceable to this exact intent and capture.
            desired_yaw = 0.0 if result is None else float(result.desired_yaw_rate_dps)
            measured_yaw = 0.0 if result is None else float(result.measured_yaw_rate_dps)
            if (
                result is not None
                and abs(float(correction)) >= 2.0
                and abs(measured_yaw) >= 8.0
                and desired_yaw * measured_yaw < 0.0
                and now - self._last_lateral_yaw_sign_mismatch_log_ts >= 0.20
            ):
                self._last_lateral_yaw_sign_mismatch_log_ts = float(now)
                logger.warning(
                    "lateral_yaw_sign_mismatch seq=%d control_frame_id=%d evidence_capture_frame_id=%d decision_capture_frame_id=%d "
                    "source_module=lateral_intent_loop action=%s correction=%+drpm "
                    "desired_yaw=%+.2fdps measured_yaw=%+.2fdps age=%.0fms reason=%s",
                    int(intent.sequence),
                    int(intent.frame_index),
                    int(intent.capture_frame_id),
                    int(intent.decision_capture_frame_id),
                    ACTION_NAMES.get(action, str(action)),
                    int(correction),
                    desired_yaw,
                    measured_yaw,
                    intent.age_sec(now) * 1000.0,
                    str(intent.reason),
                )

            self._lateral_direction_intent = direction
            self._lateral_intent_last_target_id = int(intent.target_id)
            self._lateral_intent_last_mode = intent.mode
            self._lateral_intent_last_sequence = int(intent.sequence)
            self._lateral_intent_last_correction_rpm = int(correction)
            self._lateral_intent_last_tick_ts = float(now)

            if intent.mode == "yaw_only":
                self._current_rotate_pulse_enabled = False
                self._current_rotate_raw_target = abs(int(correction))
                self._current_rotate_raw_source = "lateral_intent_30hz"
            elif intent.mode == "reverse":
                self._current_forward_percent = int(intent.base_percent)
                self._current_steer_correction_rpm = int(correction)
            else:
                result_base_rpm = int(intent.base_rpm) if result is None else int(result.base_rpm)
                base_percent = max(
                    0,
                    min(
                        100,
                        int(round(100.0 * result_base_rpm / float(max(1, FORWARD_MAX_RPM)))),
                    ),
                )
                self._current_forward_percent = base_percent
                self._current_steer_base_percent = base_percent
                self._current_steer_inner_ratio_percent = 100
                self._current_steer_outer_ratio_percent = 100
                self._current_steer_correction_rpm = abs(int(correction))
            self._last_vision_correction_rpm = int(correction)
            self._last_vision_correction_at = float(now)
            self._last_vision_correction_target_id = int(intent.target_id)

            publish_due = bool(
                now - self._lateral_intent_last_publish_ts
                >= LATERAL_INTENT_MOTOR_PUBLISH_INTERVAL_SEC
            )
            if action != self._last_dispatched_action:
                # A STOP-to-1RPM start and any direction change are control
                # events, not periodic refreshes. Publish the first usable
                # command immediately instead of waiting another 50 ms slot.
                publish_due = True
            if (
                intent.mode == "yaw_only"
                and correction == 0
                and previous == 0
                and self._last_dispatched_action == ACTION_STOP
            ):
                publish_due = False
            if publish_due:
                self._last_command_source_module = "lateral_intent_loop"
                self._last_command_control_frame = int(intent.frame_index)
                self._last_decision_capture_frame = int(intent.decision_capture_frame_id)
                self._last_command_capture_frame = int(intent.capture_frame_id)
                self._last_command_capture_timestamp = float(intent.capture_timestamp)
                reason = "lateral_intent_30hz:" + str(intent.reason)
                if not self._should_skip_redundant_action_queue([action], reason):
                    if intent.mode == "yaw_only" and correction == 0:
                        # A PID zero crossing is a normal yaw target update.
                        # Keep ACTION_STOP for clear diagnostics, but execute it
                        # through STOP_SOFT (0 RPM), not the global brake latch.
                        self._use_soft_stop_next = True
                    self._replace_action_queue([action], reason)
                self._lateral_intent_last_publish_ts = float(now)

            log_due = bool(
                direction != self._lateral_intent_last_log_direction
                or now - self._lateral_intent_last_log_ts >= LATERAL_INTENT_LOG_INTERVAL_SEC
            )
            if log_due:
                self._lateral_intent_last_log_ts = float(now)
                self._lateral_intent_last_log_direction = direction
                logger.info(
                    "lateral_intent_tick seq=%d control_frame_id=%d evidence_capture_frame_id=%d decision_capture_frame_id=%d target=%d direction=%s mode=%s "
                    "age=%.0fms x=%.3f projected_x=%.3f projection=%.0fms "
                    "target_rate=%s desired_yaw=%+.2fdps measured_yaw=%+.2fdps "
                    "requested=%+drpm output=%+drpm limit=%.1frpm action=%s "
                    "confidence=%.2f quality=%s feedback=%s publish=%s",
                    intent.sequence,
                    intent.frame_index,
                    int(intent.capture_frame_id),
                    int(intent.decision_capture_frame_id),
                    intent.target_id,
                    direction,
                    intent.mode,
                    intent.age_sec(now) * 1000.0,
                    intent.x_ratio,
                    projected_x,
                    projection_sec * 1000.0,
                    "none" if intent.target_image_rate_dps is None else "%+.2f" % intent.target_image_rate_dps,
                    0.0 if result is None else result.desired_yaw_rate_dps,
                    0.0 if result is None else result.measured_yaw_rate_dps,
                    requested,
                    correction,
                    intent.correction_limit_rpm if result is None else result.correction_limit_rpm,
                    ACTION_NAMES.get(action, str(action)),
                    intent.confidence,
                    intent.bbox_quality,
                    False if result is None else result.feedback_used,
                    publish_due,
                )

    def _lateral_intent_control_loop(self) -> None:
        period = 1.0 / max(1.0, LATERAL_INTENT_CONTROL_RATE_HZ)
        next_tick = time.monotonic()
        while not self._lateral_intent_stop_event.is_set() and self.running:
            now = time.monotonic()
            if now >= next_tick:
                try:
                    self._service_lateral_intent(now)
                except Exception as exc:
                    logger.warning("横向意图控制单次刷新失败: %s", exc)
                next_tick = max(next_tick + period, now + 0.25 * period)
            self._lateral_intent_stop_event.wait(
                max(0.001, min(period, next_tick - time.monotonic()))
            )

    def _publish_longitudinal_context(
        self,
        width: int,
        height: int,
        persons: List[Tuple],
        *,
        target_steerable: bool = True,
    ) -> None:
        """Publish only the currently locked ReID target for short Depth reuse."""
        active_target_id = getattr(self._follow_controller, "active_target_id", None)
        if active_target_id is None or int(active_target_id) < 0:
            self._clear_longitudinal_context()
            return
        matching = [
            person
            for person in persons
            if len(person) >= 2 and int(person[1]) == int(active_target_id)
        ]
        if not matching:
            self._clear_longitudinal_context()
            return
        context = {
            "published_ts": time.monotonic(),
            "frame_index": int(self.frame_index),
            "width": int(width),
            "height": int(height),
            "persons": list(matching),
            "target_id": int(active_target_id),
            "target_steerable": bool(target_steerable),
        }
        with self._longitudinal_context_lock:
            self._longitudinal_context = context

    def _start_longitudinal_thread(self) -> None:
        if LATERAL_INTENT_CONTROL_ENABLE:
            logger.info(
                "30Hz Depth纵向监督未启动: 正常跟随由横向意图循环独占动作发布，"
                "距离仍在每个视觉决策帧参与纵向控制"
            )
            return
        if (
            not ASTRA_DEPTH_LONGITUDINAL_CONTROL_ENABLE
            or not MODULE_ASTRA_DEPTH_ENABLE
            or not VISION_DEPTH_ENABLED
        ):
            logger.info("30Hz Depth纵向监督未启用")
            return
        if self._longitudinal_thread is not None and self._longitudinal_thread.is_alive():
            return
        self._longitudinal_stop_event.clear()
        self._longitudinal_thread = threading.Thread(
            target=self._longitudinal_control_loop,
            name="depth-longitudinal-control",
            daemon=True,
        )
        self._longitudinal_thread.start()
        logger.info(
            "30Hz Depth纵向监督已启动: rate=%.1fHz bbox_ttl=%.0fms motor_writer=action_runtime",
            ASTRA_DEPTH_LONGITUDINAL_CONTROL_HZ,
            ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC * 1000.0,
        )

    def _stop_longitudinal_thread(self) -> None:
        self._longitudinal_stop_event.set()
        thread = self._longitudinal_thread
        if thread is not None:
            thread.join(timeout=0.50)
        self._longitudinal_thread = None
        self._clear_longitudinal_context()

    def _longitudinal_control_loop(self) -> None:
        period_sec = 1.0 / max(1.0, float(ASTRA_DEPTH_LONGITUDINAL_CONTROL_HZ))
        while not self._longitudinal_stop_event.is_set():
            cycle_started = time.monotonic()
            with self._longitudinal_context_lock:
                context = (
                    None
                    if self._longitudinal_context is None
                    else dict(self._longitudinal_context)
                )
            if context is not None and self.running and self._action_runtime_started:
                context_age = cycle_started - float(context["published_ts"])
                active_target_id = getattr(self._follow_controller, "active_target_id", None)
                context_valid = bool(
                    context_age <= ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC
                    and active_target_id is not None
                    and int(active_target_id) == int(context["target_id"])
                    and self.search_state == "none"
                    and not self._runtime_shutdown_requested
                )
                if context_valid:
                    try:
                        self._queue_actions_for_persons(
                            int(context["width"]),
                            int(context["height"]),
                            list(context["persons"]),
                            depth_use_latest=True,
                            control_source="depth30",
                            target_steerable=bool(context.get("target_steerable", True)),
                            expected_target_id=int(context["target_id"]),
                            context_published_ts=float(context["published_ts"]),
                            expected_frame_index=int(context["frame_index"]),
                        )
                    except Exception as exc:
                        logger.warning("30Hz Depth纵向监督单次更新失败: %s", exc)
                elif context_age > ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC:
                    now = time.monotonic()
                    if now - self._last_longitudinal_stale_log_ts >= 1.0:
                        self._last_longitudinal_stale_log_ts = now
                        logger.info(
                            "30Hz Depth纵向监督跳过旧人物框: frame=%d age=%.0fms ttl=%.0fms",
                            int(context["frame_index"]),
                            context_age * 1000.0,
                            ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC * 1000.0,
                        )
            elapsed = time.monotonic() - cycle_started
            self._longitudinal_stop_event.wait(max(0.001, period_sec - elapsed))

    def _action_names_for_log(self, actions: List[int]) -> List[str]:
        return [ACTION_NAMES.get(a, str(a)) for a in actions]

    def _action_queue_names_for_log(self, actions: List[int]) -> List[str]:
        return [ACTION_NAMES.get(a, str(a)) for a in actions]

    def _action_queue_snapshot_locked(self) -> List[int]:
        try:
            return list(self.action_queue.queue)
        except Exception:
            return []

    def _stabilize_forward_percent(self, requested_percent: int, reason: str) -> int:
        """Lock small DRIVE changes so keepalive cannot bypass RPM hysteresis."""
        requested = max(0, min(100, int(requested_percent)))
        previous = self._forward_speed_latched_percent
        if previous is None or MOTOR_FORWARD_UPDATE_MIN_DELTA_RPM <= 0:
            self._forward_speed_latched_percent = requested
            return requested

        max_rpm = max(1, int(MOTOR_FORWARD_MAX_TARGET_RPM))
        requested_rpm = round(max_rpm * requested / 100.0)
        previous_rpm = round(max_rpm * int(previous) / 100.0)
        delta_rpm = abs(requested_rpm - previous_rpm)

        # 漏检/测距保持是安全降速，必须立即生效，不能被普通速度滞回挡住。
        reason_code = str(reason or "")
        safety_hold = reason_code in {
            "lost_wait_hold_forward",
            "distance_missing_camera_hold",
        } or (
            reason_code.startswith("visual_pid_center_")
            and ("distance_missing" in reason_code or "mmwave_hold" in reason_code)
        )
        if safety_hold and requested_rpm < previous_rpm:
            self._forward_speed_latched_percent = requested
            return requested
        if delta_rpm >= MOTOR_FORWARD_UPDATE_MIN_DELTA_RPM:
            self._forward_speed_latched_percent = requested
            return requested

        now = time.monotonic()
        if now - self._last_forward_speed_hysteresis_log_ts >= 1.0:
            logger.info(
                "直行速度滞回保持: 原请求=%d%%/%drpm 当前=%d%%/%drpm 差值=%drpm 阈值=%drpm 原因代码=%s",
                requested,
                requested_rpm,
                int(previous),
                previous_rpm,
                delta_rpm,
                MOTOR_FORWARD_UPDATE_MIN_DELTA_RPM,
                reason,
            )
            self._last_forward_speed_hysteresis_log_ts = now
        return int(previous)

    def _action_signature(self, action: int) -> Tuple:
        if action == ACTION_BACKWARD:
            return (
                action,
                "percent",
                int(self._current_forward_percent),
                int(self._current_steer_correction_rpm),
            )
        if action == ACTION_FORWARD:
            if MOTOR_FORWARD_RAW_TARGET > 0:
                return (
                    action,
                    "raw",
                    int(MOTOR_FORWARD_RAW_TARGET),
                    int(MOTOR_FORWARD_TARGET_SIGN),
                    int(MOTOR_LEFT_SIGN),
                    int(MOTOR_RIGHT_SIGN),
                )
            return (action, "percent", int(self._current_forward_percent))
        if action in (ACTION_STEER_LEFT, ACTION_STEER_RIGHT):
            if MOTOR_STEER_RAW_TARGET > 0:
                mmwave_hold = "mmwave_hold" in str(self._last_control_decision_reason or "")
                direct_correction = int(getattr(self, "_current_steer_correction_rpm", 0))
                if direct_correction > 0:
                    return (
                        action,
                        "pid_raw",
                        int(self._current_steer_base_percent),
                        direct_correction,
                        mmwave_hold,
                    )
                return (
                    action,
                    "raw",
                    int(MOTOR_STEER_RAW_TARGET),
                    int(self._current_steer_inner_ratio_percent),
                    int(self._current_steer_outer_ratio_percent),
                    mmwave_hold,
                    int(VISION_MMWAVE_HOLD_FORWARD_PERCENT) if mmwave_hold else None,
                )
            return (
                action,
                "percent",
                int(self._current_steer_base_percent),
                int(self._current_steer_inner_ratio_percent),
                int(self._current_steer_outer_ratio_percent),
            )
        if action in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT):
            pulse_enabled = bool(
                ROTATE_PULSE_BRAKE_ENABLE
                and getattr(self, "_current_rotate_pulse_enabled", True)
            )
            rotate_raw_target = int(getattr(self, "_current_rotate_raw_target", MOTOR_ROTATE_RAW_TARGET))
            if rotate_raw_target > 0:
                return (
                    action,
                    "raw",
                    rotate_raw_target,
                    str(getattr(self, "_current_rotate_raw_source", "default")),
                    pulse_enabled,
                    float(ROTATE_DURATION),
                    float(ROTATE_PULSE_PAUSE_SEC),
                    int(ROTATE_PULSE_OBSERVE_MIN_FRAMES),
                )
            return (
                action,
                "percent",
                int(self._current_rotate_turn_percent),
                pulse_enabled,
                float(ROTATE_DURATION),
                float(ROTATE_PULSE_PAUSE_SEC),
                int(ROTATE_PULSE_OBSERVE_MIN_FRAMES),
            )
        return (action,)

    def _actions_signature(self, actions: List[int]) -> Tuple:
        return tuple(self._action_signature(action) for action in actions)

    def _should_skip_redundant_action_queue(self, actions: List[int], reason: str) -> bool:
        if len(actions) != 1:
            return False
        action = actions[0]
        if action == ACTION_STOP and self._brake_hold_active:
            with self.action_queue_lock:
                pending_actions = self._action_queue_snapshot_locked()
            if any(pending != ACTION_STOP for pending in pending_actions):
                return False
            self._last_action_intent_ts = time.monotonic()
            self._last_action_intent_frame = self.frame_index
            self._last_action_intent_reason = reason
            now = time.monotonic()
            if now - float(getattr(self, "_last_redundant_action_skip_log_ts", 0.0)) >= 1.0:
                logger.info(
                    "停车发布保持: 帧=%d 原因代码=%s brake_hold=True "
                    "动作线程按%.2f秒周期续发刹车",
                    self.frame_index,
                    reason,
                    float(getattr(self, "_brake_hold_refresh_interval_sec", 1.0)),
                )
                self._last_redundant_action_skip_log_ts = now
            return True
        if action not in MOVEMENT_ACTIONS:
            return False
        if self._brake_hold_active:
            return False
        signature = self._actions_signature(actions)
        if signature != self._last_action_queue_signature:
            return False
        with self.action_queue_lock:
            pending_actions = self._action_queue_snapshot_locked()
        if pending_actions:
            return False
        with self.command_lock:
            current_command = self.current_command
            stop_pending = bool(self.stop_action_execution or self.person_detected_flag)
        if stop_pending or current_command != action:
            return False
        # A skipped identical action is still a fresh controller heartbeat.
        self._last_action_intent_ts = time.monotonic()
        self._last_action_intent_frame = self.frame_index
        self._last_action_intent_reason = reason
        if action in ROTATE_ACTIONS and not bool(
            ROTATE_PULSE_BRAKE_ENABLE
            and getattr(self, "_current_rotate_pulse_enabled", True)
        ):
            # The visual loop is still asking for the same turn. Keep the
            # executor alive without re-enqueueing and flooding RS485.
            self._last_rotate_visual_refresh_ts = time.time()
        now = time.monotonic()
        if now - self._last_redundant_action_skip_log_ts >= 1.0:
            since_last_enqueue_ms = (
                (now - self._last_action_queue_replace_ts) * 1000.0
                if self._last_action_queue_replace_ts > 0
                else -1.0
            )
            logger.info(
                "动作队列跳过重复指令: control_frame_id=%d decision_capture_frame_id=%d evidence_capture_frame_id=%d 原因代码=%s 动作=%s 当前动作=%s 特征=%s 距上次入队=%.1f毫秒",
                self.frame_index,
                int(getattr(self, "_last_decision_capture_frame", -1)),
                int(getattr(self, "_last_command_capture_frame", -1)),
                reason,
                self._action_names_for_log(actions),
                ACTION_NAMES.get(current_command, str(current_command)),
                signature,
                since_last_enqueue_ms,
            )
            self._last_redundant_action_skip_log_ts = now
        return True

    def _prepare_direct_stop(self, reason: str) -> None:
        reason_text = str(reason or "direct_stop")
        self._last_command_source_module = (
            "safety_gate"
            if any(token in reason_text for token in ("hard_stop", "front_ir", "left_ir", "right_ir", "bunker", "hazard"))
            else "follow_controller"
        )
        self._last_command_control_frame = int(self.frame_index)
        self._last_decision_capture_frame = int(getattr(self, "_active_capture_frame_id", -1))
        self._last_command_capture_frame = int(getattr(self, "_active_capture_frame_id", -1))
        self._last_command_capture_timestamp = float(getattr(self, "_active_capture_timestamp", 0.0))
        with self.command_lock:
            old_command = self.current_command
            was_rotate = old_command in ROTATE_ACTIONS
            if was_rotate:
                self._rotate_follows_previous_rotate = True
                self._last_rotate_end_ts = time.time()
            self.current_command = None
            self.command_start_time = None
            self.stop_action_execution = False
            self.person_detected_flag = False
        self._last_dispatched_action = ACTION_STOP
        self._last_action_queue_reason = reason
        self._last_action_queue_replace_frame = self.frame_index
        self._last_action_queue_actions = []
        self._last_action_queue_signature = None
        self._forward_speed_latched_percent = None
        self._current_steer_correction_rpm = 0
        logger.info(
            "直接停车准备完成: 帧=%d 原因代码=%s 原命令=%s 原命令是否旋转=%s",
            self.frame_index,
            reason,
            ACTION_NAMES.get(old_command, str(old_command)),
            was_rotate,
        )

    def _mark_direct_stop_sent(self, reason: str) -> None:
        self._last_direct_stop_reason = str(reason or "")
        self._last_direct_stop_ts = time.monotonic()

    def _should_skip_redundant_direct_stop(self, reason: str) -> bool:
        with self.action_queue_lock:
            pending_actions = self._action_queue_snapshot_locked()
        if pending_actions:
            return False
        with self.command_lock:
            current_command = self.current_command
            stop_pending = bool(self.stop_action_execution or self.person_detected_flag)
        # Once brake hold is active, stale action-thread flags must not force
        # a fresh STOP on every vision frame. Safety transitions still clear
        # the queue and use their distinct reason codes.
        if (
            (current_command is not None or stop_pending)
            and not bool(getattr(self, "_brake_hold_active", False))
        ):
            return False
        if self._last_dispatched_action != ACTION_STOP:
            return False
        if str(reason or "") != self._last_direct_stop_reason:
            return False
        now = time.monotonic()
        if now - self._last_direct_stop_ts >= DIRECT_STOP_REPEAT_INTERVAL_SEC:
            return False
        if now - self._last_redundant_direct_stop_log_ts >= 1.0:
            logger.info(
                "直接停车跳过重复发送: 帧=%d 原因代码=%s 距上次发送=%.1f毫秒 重复间隔=%.2f秒 刹车保持=%s 队列=%s",
                self.frame_index,
                reason,
                (now - self._last_direct_stop_ts) * 1000.0,
                DIRECT_STOP_REPEAT_INTERVAL_SEC,
                self._brake_hold_active,
                self._action_queue_names_for_log(pending_actions),
            )
            self._last_redundant_direct_stop_log_ts = now
        return True

    def _clear_action_queue(self, reason: str) -> int:
        start = time.perf_counter()
        cleared = 0
        with self.action_queue_lock:
            before_actions = self._action_queue_snapshot_locked()
            while True:
                try:
                    self.action_queue.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
            after_actions = self._action_queue_snapshot_locked()
        clear_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "动作队列已清空: control_frame_id=%d decision_capture_frame_id=%d evidence_capture_frame_id=%d 原因代码=%s 清空前=%d 已清除=%d 清空后=%d 耗时=%.3f毫秒 原动作=%s 剩余动作=%s 当前动作=%s 停止标志=%s 目标标志=%s",
            self.frame_index,
            int(getattr(self, "_last_decision_capture_frame", -1)),
            int(getattr(self, "_last_command_capture_frame", -1)),
            reason,
            len(before_actions),
            cleared,
            len(after_actions),
            clear_ms,
            self._action_queue_names_for_log(before_actions),
            self._action_queue_names_for_log(after_actions),
            ACTION_NAMES.get(self.current_command, str(self.current_command)),
            self.stop_action_execution,
            self.person_detected_flag,
        )
        return cleared

    def _replace_action_queue(self, actions: List[int], reason: str) -> None:
        start = time.perf_counter()
        replace_ts = time.monotonic()
        self._last_action_intent_ts = replace_ts
        self._last_action_intent_frame = self.frame_index
        self._last_action_intent_reason = reason
        requested_actions = tuple(int(action) for action in actions)
        action_signature = self._actions_signature(list(requested_actions))
        if len(actions) == 1 and actions[0] in ROTATE_ACTIONS and not bool(
            ROTATE_PULSE_BRAKE_ENABLE
            and getattr(self, "_current_rotate_pulse_enabled", True)
        ):
            self._last_rotate_visual_refresh_ts = time.time()
        cleared = 0
        queued = 0
        with self.action_queue_lock:
            before_actions = self._action_queue_snapshot_locked()
            before = len(before_actions)
            merge = merge_pending_actions(
                before_actions,
                requested_actions,
                stop_action=ACTION_STOP,
                rotate_actions=frozenset(ROTATE_ACTIONS),
                current_action=self.current_command,
            )
            queued_actions = list(merge.actions)
            if merge.inserted_stop_barrier or merge.preserved_stop_barrier:
                logger.info(
                    "动作队列安全合并: reason=%s requested=%s pending=%s merged=%s inserted_stop=%s preserved_stop=%s",
                    reason,
                    self._action_names_for_log(requested_actions),
                    self._action_names_for_log(before_actions),
                    self._action_names_for_log(queued_actions),
                    merge.inserted_stop_barrier,
                    merge.preserved_stop_barrier,
                )
            clear_start = time.perf_counter()
            while True:
                try:
                    self.action_queue.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
            clear_ms = (time.perf_counter() - clear_start) * 1000.0
            for action in queued_actions:
                try:
                    self.action_queue.put_nowait(action)
                    queued += 1
                except queue.Full:
                    logger.warning(
                        "action queue full while replacing: control_frame_id=%d decision_capture_frame_id=%d evidence_capture_frame_id=%d reason=%s queued=%d actions=%s",
                        self.frame_index,
                        int(getattr(self, "_last_decision_capture_frame", -1)),
                        int(getattr(self, "_last_command_capture_frame", -1)),
                        reason,
                        queued,
                        self._action_names_for_log(queued_actions),
                    )
                    break
            after_actions = self._action_queue_snapshot_locked()
            after = len(after_actions)
        total_ms = (time.perf_counter() - start) * 1000.0
        new_primary = requested_actions[0] if requested_actions else None
        prev_primary = self._last_tracker_action_kind
        since_last_replace_ms = (
            (replace_ts - self._last_action_queue_replace_ts) * 1000.0
            if self._last_action_queue_replace_ts > 0
            else -1.0
        )
        tracker_switch_ms = (
            (replace_ts - self._last_tracker_action_change_ts) * 1000.0
            if self._last_tracker_action_change_ts > 0
            else -1.0
        )
        tracker_switch_frames = (
            self.frame_index - self._last_tracker_action_change_frame
            if self._last_tracker_action_change_frame >= 0
            else -1
        )
        if new_primary != prev_primary:
            logger.info(
                "action kind switch: control_frame_id=%d decision_capture_frame_id=%d evidence_capture_frame_id=%d reason=%s old=%s new=%s previous_kind_age_ms=%.1f control_frame_delta=%d since_last_enqueue_ms=%.1f",
                self.frame_index,
                int(getattr(self, "_last_decision_capture_frame", -1)),
                int(getattr(self, "_last_command_capture_frame", -1)),
                reason,
                ACTION_NAMES.get(prev_primary, str(prev_primary)),
                ACTION_NAMES.get(new_primary, str(new_primary)),
                tracker_switch_ms,
                tracker_switch_frames,
                since_last_replace_ms,
            )
            self._last_tracker_action_kind = new_primary
            self._last_tracker_action_change_ts = replace_ts
            self._last_tracker_action_change_frame = self.frame_index
        if requested_actions:
            self._action_queue_seq += 1
            self._last_dispatched_action = requested_actions[0]
            self._last_action_queue_replace_ts = replace_ts
            self._last_action_queue_replace_frame = self.frame_index
            self._last_action_queue_seq = self._action_queue_seq
            self._last_action_queue_reason = reason
            self._last_action_queue_actions = list(queued_actions)
            self._last_action_queue_signature = action_signature
        logger.info(
            "action queue replace: seq=%d control_frame_id=%d decision_capture_frame_id=%d evidence_capture_frame_id=%d source_module=%s reason=%s actions=%s signature=%s before=%d cleared=%d queued=%d after=%d clear_ms=%.3f total_ms=%.3f queue_before=%s queue_after=%s current=%s last_dispatched=%s since_last_enqueue_ms=%.1f search=%s/%s",
            self._last_action_queue_seq,
            self.frame_index,
            int(getattr(self, "_last_decision_capture_frame", -1)),
            int(getattr(self, "_last_command_capture_frame", -1)),
            str(getattr(self, "_last_command_source_module", "unknown")),
            reason,
            self._action_names_for_log(queued_actions),
            action_signature,
            before,
            cleared,
            queued,
            after,
            clear_ms,
            total_ms,
            self._action_queue_names_for_log(before_actions),
            self._action_queue_names_for_log(after_actions),
            ACTION_NAMES.get(self.current_command, str(self.current_command)),
            ACTION_NAMES.get(self._last_dispatched_action, str(self._last_dispatched_action)),
            since_last_replace_ms,
            self.search_state,
            self.search_direction,
        )

    def _get_obstacle_status(self) -> dict:
        """
        读取红外传感器状态（通过 SensorRuntime）

        Returns:
            dict: {"front": bool, "left": bool, "right": bool}
                  True表示有障碍物，False表示无障碍物
        """
        obstacles = self._sensor_runtime.get_obstacle_status()
        return {"front": obstacles.front, "left": obstacles.left, "right": obstacles.right}

    def _trigger_hazard_safety_stop(self, reason: str) -> None:
        """清空待执行动作，并对视觉危险触发刹车停止。"""
        self._explicit_stop_requested = True
        self.stop_action_execution = True
        self.is_forwarding = False
        self._current_forward_percent = 0
        self._use_soft_stop_next = False
        self._brake_hold_active = True
        self._brake_hold_stop_mode = SAFETY_STOP_MODE
        self._brake_hold_label = "safety_hold_hazard"
        self._last_brake_hold_send_ts = 0.0
        self._clear_action_queue(f"hazard:{reason}")
        try:
            self._action_runtime.send_percent_brake(mode=SAFETY_STOP_MODE, label=f"safety_{reason}")
        except Exception as e:
            logger.warning("视觉危险安全停止发送失败: %s, reason=%s", e, reason)

    def _handle_hazard_safety_state(self, state: Any) -> bool:
        if state is None or not getattr(state, "active", False):
            return False
        reason = state.reason() if hasattr(state, "reason") else "hazard"
        self._trigger_hazard_safety_stop(reason)
        return True

    def _action_kind_to_int(self, kind: str) -> Optional[int]:
        return {
            "forward": ACTION_FORWARD,
            "backward": ACTION_BACKWARD,
            "rotate_left": ACTION_ROTATE_LEFT,
            "rotate_right": ACTION_ROTATE_RIGHT,
            "steer_left": ACTION_STEER_LEFT,
            "steer_right": ACTION_STEER_RIGHT,
            "stop": ACTION_STOP,
        }.get(kind)

    def _action_int_to_kind(self, action: Optional[int]) -> Optional[str]:
        return {
            ACTION_FORWARD: "forward",
            ACTION_BACKWARD: "backward",
            ACTION_ROTATE_LEFT: "rotate_left",
            ACTION_ROTATE_RIGHT: "rotate_right",
            ACTION_STEER_LEFT: "steer_left",
            ACTION_STEER_RIGHT: "steer_right",
            ACTION_STOP: "stop",
        }.get(action)

    def clear_active_target(self, reason: str = "manual", *, stop_current: bool = True) -> None:
        """Clear the locked follow target so the next visible person can be selected."""
        self._follow_controller.clear_active_target(reason)
        self.search_state = self._follow_controller.search_state
        self.search_direction = self._follow_controller.search_direction
        self.lost_confirm_frames = self._follow_controller.lost_confirm_frames
        self._reset_search_geometry_reacquire()
        if stop_current:
            self._clear_action_queue(f"clear_active_target:{reason}")
            if not self._should_skip_redundant_direct_stop("clear_active_target"):
                self._prepare_direct_stop("clear_active_target")
                self._action_runtime.send_stop_with_brake_hold("clear_active_target")
                self._mark_direct_stop_sent("clear_active_target")

    def _persons_to_targets(self, persons: List[Tuple]) -> List[PersonTarget]:
        targets: List[PersonTarget] = []
        for bbox, track_id, conf, area in persons:
            targets.append(
                PersonTarget(
                    bbox=tuple(float(v) for v in bbox),
                    track_id=int(track_id),
                    confidence=float(conf),
                    area=float(area),
                )
            )
        return targets

    def _current_hazard_state_for_controller(self) -> HazardState:
        state = self._bunker_runtime.current_split_state()
        if state is None:
            return HazardState()
        return HazardState(
            active=True,
            reason=state.reason(),
            class_id=int(state.class_id),
            score=float(state.score),
            area_ratio=float(state.area_ratio),
        )

    @staticmethod
    def _format_distance_state(state: DistanceState) -> str:
        def fmt(value: Any) -> str:
            if value is None:
                return "none"
            try:
                return "%.2f" % float(value)
            except Exception:
                return str(value)

        base = (
            "来源=%s 原始距离=%s米 滤波距离=%s米 使用距离=%s米 触发代码=%s "
            "目标接近计数=%d 刹车接近计数=%d 锁存(目标=%s,刹车=%s)"
        ) % (
            state.source,
            fmt(state.raw_distance_m),
            fmt(state.filtered_distance_m),
            fmt(state.used_distance_m),
            state.trigger,
            int(state.target_close_count),
            int(state.brake_close_count),
            state.target_latched,
            state.brake_latched,
        )
        extras = []
        if state.source_detail:
            extras.append("详情代码=%s" % state.source_detail)
        if state.sample_age_sec is not None:
            extras.append("样本延迟=%.0f毫秒" % (float(state.sample_age_sec) * 1000.0))
        if state.target_angle_deg is not None:
            extras.append("目标角度=%.1f度" % float(state.target_angle_deg))
        if state.matched_angle_deg is not None:
            extras.append("匹配角度=%.1f度" % float(state.matched_angle_deg))
        if state.fusion_mode:
            extras.append(
                "融合=(模式=%s 置信度=%.2f 雷达=%s米 视觉=%s米 编码器步进=%.3f米)"
                % (
                    state.fusion_mode,
                    float(state.fusion_confidence),
                    fmt(state.fusion_radar_distance_m),
                    fmt(state.fusion_visual_distance_m),
                    float(state.fusion_encoder_delta_m),
                )
            )
        if extras:
            return base + " " + " ".join(extras)
        return base

    def _log_distance_stop_trigger(self, reason: str, state: DistanceState, *, safety: bool = False) -> None:
        if reason not in {
            "target_distance_reached",
            "distance_too_close",
            "person_too_close_no_rotate",
            "search_both_sides_blocked",
            "hard_stop",
        }:
            return
        log_key = (
            reason,
            state.source,
            state.trigger,
            None if state.used_distance_m is None else round(float(state.used_distance_m), 2),
            int(state.target_close_count),
            int(state.brake_close_count),
            bool(state.target_latched),
            bool(state.brake_latched),
        )
        now = time.monotonic()
        if log_key == self._last_distance_stop_log_key and now - self._last_distance_hard_stop_log_ts < 0.50:
            return
        self._last_distance_stop_log_key = log_key
        self._last_distance_hard_stop_log_ts = now
        logger.warning(
            "距离停车触发: 帧=%d 原因代码=%s 是否安全停车=%s %s 目标阈值=%s 刹车阈值=%s 回差=%.2f",
            self.frame_index,
            reason,
            safety,
            self._format_distance_state(state),
            "none" if state.target_threshold_m is None else "%.2fm" % float(state.target_threshold_m),
            "none" if state.brake_threshold_m is None else "%.2fm" % float(state.brake_threshold_m),
            float(state.hysteresis_m),
        )

    def _target_region_debug(self, target: Optional[PersonTarget], width: int, height: int) -> str:
        if target is None or width <= 0:
            return "none"
        cx, cy = target.center
        x_ratio = float(cx) / float(width)
        y_ratio = float(cy) / float(height) if height > 0 else 0.0
        center_left, center_right = self._follow_controller._active_center_band(width)  # noqa: SLF001 - runtime diagnostic
        center_left_ratio = center_left / float(width)
        center_right_ratio = center_right / float(width)
        rotate_edge = self._follow_controller._visible_rotate_edge_type(cx, width)  # noqa: SLF001 - runtime diagnostic
        edge = self._follow_controller._edge_type(cx, width)  # noqa: SLF001 - runtime diagnostic
        in_center = self._follow_controller._is_in_center_3x3(cx, cy, width, height)  # noqa: SLF001 - runtime diagnostic
        if rotate_edge != "none":
            region = "visible_rotate_%s" % rotate_edge
        elif in_center:
            region = "center"
        elif edge != "none":
            region = "steer_%s" % edge
        else:
            region = "unknown"
        rotate_left = self._follow_controller.cfg.visible_rotate_left_ratio
        rotate_right = self._follow_controller.cfg.visible_rotate_right_ratio
        return (
            "区域代码=%s 横向比例=%.3f 纵向比例=%.3f 中心范围=[%.2f,%.2f] "
            "旋转范围=[<%s,>%s]"
        ) % (
            region,
            x_ratio,
            y_ratio,
            center_left_ratio,
            center_right_ratio,
            "none" if rotate_left is None else "%.2f" % float(rotate_left),
            "none" if rotate_right is None else "%.2f" % float(rotate_right),
        )

    def _target_quality_debug(self, target: Optional[PersonTarget]) -> str:
        if target is None:
            return "none"
        candidates = self._last_person_reid_debug_by_stable_id.get(int(target.track_id), [])
        if not candidates:
            return "missing"
        cand = candidates[0]
        assign = cand.get("assignment") or {}
        quality_ok = assign.get("bbox_quality_ok")
        quality_reason = assign.get("bbox_quality_reason")
        reason = assign.get("reason", "missing")
        raw_track_id = cand.get("raw_track_id")
        reid_uid = cand.get("reid_uid")
        quality = str(quality_ok)
        if quality_reason:
            quality += "(%s)" % quality_reason
        return "原始轨迹=%s ReID编号=%s 分配原因代码=%s 边界框质量=%s" % (
            raw_track_id,
            reid_uid,
            reason,
            quality,
        )

    @staticmethod
    def _rotate_raw_target_for_reason(reason: str) -> Tuple[int, str]:
        reason = str(reason or "")
        if reason.startswith(
            ("stale_probe_", "stale_candidate_", "search_candidate_center_")
        ):
            return FOLLOW_STALE_DIRECTION_PROBE_RAW_TARGET, "stale_probe"
        if reason.startswith("search_candidate_approach_"):
            return max(1, int(PARKED_RECENTER_MIN_RPM)), "candidate_approach"
        if reason.startswith(
            ("search_current_candidate_", "current_candidate_near_target_")
        ):
            return int(SEARCH_CANDIDATE_ACQUIRE_RAW_RPM), "candidate_acquire"
        if reason.startswith(
            ("lost_current_candidate_hold_", "lost_history_hold_", "lost_wait_yaw_")
        ):
            # Loss confirmation is still part of visible tracking. Keep a
            # defensive cap here as well as in the decision translation so a
            # future classification change cannot promote it to scan speed.
            return (
                max(
                    1,
                    min(
                        int(VISIBLE_STEERING_PID_LOST_HOLD_MAX_CORRECTION_RPM),
                        int(ROTATE_RAW_TARGET_LOST_WAIT or MOTOR_ROTATE_RAW_TARGET),
                    ),
                ),
                "lost_confirm",
            )
        if reason.startswith("lost_wait_"):
            return ROTATE_RAW_TARGET_LOST_WAIT or MOTOR_ROTATE_RAW_TARGET, "lost_wait"
        if reason.startswith("search_") or reason.startswith("predict_"):
            return ROTATE_RAW_TARGET_SEARCH or MOTOR_ROTATE_RAW_TARGET, "search"
        if reason.startswith("person_") or reason.startswith("near_distance_rotation_only"):
            return ROTATE_RAW_TARGET_VISIBLE or MOTOR_ROTATE_RAW_TARGET, "visible"
        return MOTOR_ROTATE_RAW_TARGET, "default"

    def _remember_vision_steering(
        self,
        actions: List[ControlAction],
        target_id: Optional[int],
    ) -> None:
        if target_id is None or not actions:
            return
        action = actions[-1]
        if action.kind == "steer_right":
            signed_correction = abs(int(action.steer_correction_rpm))
        elif action.kind == "steer_left":
            signed_correction = -abs(int(action.steer_correction_rpm))
        elif action.kind == "backward":
            signed_correction = int(action.steer_correction_rpm)
        elif action.kind == "forward":
            signed_correction = 0
        else:
            return
        self._last_vision_correction_rpm = int(signed_correction)
        self._last_vision_correction_at = time.monotonic()
        self._last_vision_correction_target_id = int(target_id)

    def _merge_longitudinal_action_with_visual_steering(
        self,
        action: ControlAction,
        target_id: Optional[int],
    ) -> ControlAction:
        """Change longitudinal speed without recomputing or erasing visual yaw."""
        if action.kind not in ("forward", "backward"):
            return action
        correction_age = time.monotonic() - float(self._last_vision_correction_at)
        correction_fresh = bool(
            target_id is not None
            and self._last_vision_correction_target_id is not None
            and int(target_id) == int(self._last_vision_correction_target_id)
            and correction_age <= 0.20
        )
        correction = int(self._last_vision_correction_rpm) if correction_fresh else 0
        if action.kind == "backward":
            return ControlAction.backward(
                action.speed_percent,
                action.reason,
                correction_rpm=correction,
            )
        # The Depth supervisor may command zero longitudinal RPM after its
        # confidence timeout. Keep a fresh camera yaw request as a one-wheel
        # correction instead of replacing it with a two-wheel STOP.
        if correction > 0:
            return ControlAction.steer_right(
                action.speed_percent,
                100,
                100,
                action.reason,
                correction_rpm=correction,
            )
        if correction < 0:
            return ControlAction.steer_left(
                action.speed_percent,
                100,
                100,
                action.reason,
                correction_rpm=-correction,
            )
        return action

    def _depth_zero_longitudinal_preserves_visible_rotation(
        self,
        decision: ControlDecision,
    ) -> bool:
        """Keep a fresh visible-target yaw when Depth only requests zero speed."""
        if decision.reason not in {
            "longitudinal_distance_untrusted_hold",
            "longitudinal_distance_low_confidence_stop",
            "longitudinal_distance_hold",
        }:
            return False
        if not decision.actions or any(
            action.kind != "forward" or int(action.speed_percent) != 0
            for action in decision.actions
        ):
            return False
        if self.search_state != "none" or not str(self._vision_control_state).startswith(
            "target_visible"
        ):
            return False
        vision_age = time.monotonic() - float(self._last_vision_control_ts)
        if vision_age > max(0.05, float(ROTATE_HOLD_STALE_SEC)):
            return False
        with self.command_lock:
            current_command = self.current_command
        if current_command not in ROTATE_ACTIONS:
            return False
        now = time.monotonic()
        if now - float(getattr(self, "_last_depth30_preserve_yaw_log_ts", 0.0)) >= 0.50:
            self._last_depth30_preserve_yaw_log_ts = now
            logger.info(
                "depth30_zero_longitudinal_preserve_yaw frame=%d reason=%s "
                "current=%s vision_state=%s vision_age=%.0fms action=keep_current_rotation",
                int(self.frame_index),
                decision.reason,
                ACTION_NAMES.get(current_command, str(current_command)),
                self._vision_control_state,
                max(0.0, vision_age) * 1000.0,
            )
        return True

    @staticmethod
    def _rotation_only_action(action: ControlAction) -> ControlAction:
        """Remove longitudinal wheel speed while preserving camera yaw intent."""
        reason = f"rotation_only:{action.reason or action.kind}"
        if action.kind in ("rotate_left", "rotate_right", "stop", "idle"):
            return action

        if action.kind == "backward":
            correction = int(action.steer_correction_rpm)
            if correction > 0:
                return ControlAction.steer_right(
                    0, 100, 100, reason, correction_rpm=correction
                )
            if correction < 0:
                return ControlAction.steer_left(
                    0, 100, 100, reason, correction_rpm=-correction
                )
            return ControlAction.stop(reason)

        if action.kind in ("steer_left", "steer_right"):
            correction = abs(int(action.steer_correction_rpm))
            if correction <= 0:
                return ControlAction.stop(f"{reason}:missing_yaw_correction")
            factory = (
                ControlAction.steer_left
                if action.kind == "steer_left"
                else ControlAction.steer_right
            )
            return factory(0, 100, 100, reason, correction_rpm=correction)

        # A centered target normally produces forward/backward distance output.
        # In this test mode that output is a request to stop rotating in place.
        return ControlAction.stop(reason)

    def _process_detections_modular(
        self,
        width: int,
        height: int,
        persons: List[Tuple],
        *,
        depth_use_latest: bool = False,
        target_steerable: bool = True,
        target_steering_limit_rpm: Optional[float] = None,
        record_target_motion: bool = True,
        control_source: str = "vision",
        low_quality_visible: bool = False,
        lateral_candidate: Optional[LateralCandidateEvidence] = None,
    ) -> List[int]:
        """Use car_control_modular.controllers to decide actions."""
        self._explicit_stop_requested = False
        if control_source == "vision":
            # Set again only when this frame publishes a valid visible-target
            # intent. Search, safety and low-quality hold actions remain direct.
            self._lateral_intent_owned_frame = -1
        if FOLLOW_ROTATION_ONLY and control_source == "depth30":
            return []
        if control_source != "depth30":
            self._last_vision_control_frame_index = int(self.frame_index)
            self._last_vision_control_ts = time.monotonic()
        self._follow_controller.set_last_dispatched(self._action_int_to_kind(self._last_dispatched_action))

        obstacles_dict = self._get_obstacle_status()
        person_targets = self._persons_to_targets(persons)
        distance_target = self._distance_runtime.select_target(person_targets)
        steering_feedback = self._action_runtime.get_steering_feedback()
        if low_quality_visible:
            # Do not clear/re-read the depth fusion anchor with an invalid box.
            # The controller will issue a safe hold and wait for a complete box.
            distance_state = self._last_frame_distance_state or DistanceState(source="vision_depth")
        else:
            distance_state = self._distance_runtime.get_frame_distance_state(
                int(width),
                distance_target,
                frame_height=int(height),
                steering_feedback=steering_feedback,
                target_distance_m=TARGET_DISTANCE,
                brake_distance_m=FOLLOW_BRAKE_DISTANCE_M,
                depth_use_latest=bool(depth_use_latest),
            )
        self._last_frame_distance_state = distance_state
        distance_m = distance_state.used_distance_m
        frame = SensorFrame(
            width=int(width),
            height=int(height),
            persons=person_targets,
            hazard=self._current_hazard_state_for_controller(),
            obstacles=ObstacleState(
                front=bool(obstacles_dict.get("front", False)),
                left=bool(obstacles_dict.get("left", False)),
                right=bool(obstacles_dict.get("right", False)),
            ),
            distance_m=distance_m,
            distance_state=distance_state,
            steering_feedback=steering_feedback,
            module_status={
                "vision": MODULE_VISION_ENABLE,
                "ir": MODULE_IR_ENABLE,
                "mmwave": MODULE_MMWAVE_ENABLE,
                "astra_depth": MODULE_ASTRA_DEPTH_ENABLE,
                "ultrasonic": MODULE_ULTRASONIC_ENABLE,
                "imu": MODULE_IMU_ENABLE,
                "bunker": BUNKER_AVOID_ENABLE,
            },
            capture_frame_id=int(getattr(self, "_active_capture_frame_id", 0)),
            capture_timestamp=float(getattr(self, "_active_capture_timestamp", 0.0)),
            lateral_candidate=(
                lateral_candidate if control_source == "vision" else None
            ),
        )
        self._last_command_source_module = str(control_source or "vision")
        self._last_command_control_frame = int(self.frame_index)
        self._last_decision_capture_frame = int(frame.capture_frame_id)
        self._last_command_capture_frame = int(frame.capture_frame_id)
        self._last_command_capture_timestamp = float(frame.capture_timestamp)
        active_before = self._follow_controller.active_target_id
        search_before = self._follow_controller.search_state
        decision = self._follow_controller.decide(
            self.frame_index,
            frame,
            target_steerable=bool(target_steerable),
            target_steering_limit_rpm=target_steering_limit_rpm,
            record_target_motion=bool(record_target_motion),
            longitudinal_only=(control_source == "depth30"),
            rotation_only=bool(FOLLOW_ROTATION_ONLY),
            low_quality_visible=bool(low_quality_visible),
        )
        evidence_capture_frame = decision.evidence_capture_frame_id
        if evidence_capture_frame is not None and int(evidence_capture_frame) > 0:
            self._last_command_capture_frame = int(evidence_capture_frame)

        longitudinal_only = control_source == "depth30"
        if longitudinal_only and self._depth_zero_longitudinal_preserves_visible_rotation(
            decision
        ):
            # Rotation already has zero longitudinal wheel speed.  Depth has
            # fulfilled its safety responsibility without replacing the
            # camera PID command or clearing its action queue.
            return []
        decision_target = self._follow_controller.last_selected_target
        if not longitudinal_only:
            previous_control_state = self._vision_control_state
            if decision_target is not None:
                control_state = (
                    "target_visible_low_quality"
                    if not target_steerable
                    else "target_visible_depth_valid"
                    if distance_m is not None
                    else "target_visible_depth_missing"
                )
            elif decision.reason.startswith("target_visible_low_quality"):
                control_state = "target_visible_low_quality"
            elif decision.reason.startswith("search_candidate"):
                control_state = "target_reacquiring"
            elif self._follow_controller.search_state == "searching":
                control_state = "searching"
            elif self._follow_controller.search_state == "direction_unresolved":
                control_state = "direction_uncertain"
            else:
                control_state = "lost_confirming"
            if control_state != previous_control_state:
                logger.info(
                    "vision_control_state frame=%d old=%s new=%s reason=%s",
                    int(self.frame_index),
                    previous_control_state,
                    control_state,
                    decision.reason,
                )
            if control_state == "searching" and previous_control_state != "searching":
                desired_kind = decision.actions[0].kind if len(decision.actions) == 1 else None
                desired_side = (
                    "left"
                    if desired_kind in ("rotate_left", "steer_left")
                    else "right"
                    if desired_kind in ("rotate_right", "steer_right")
                    else None
                )
                with self.command_lock:
                    current_action = self.current_command
                previous_action = (
                    current_action
                    if current_action in MOVEMENT_ACTIONS
                    else self._last_dispatched_action
                )
                previous_side = (
                    "left"
                    if previous_action in (ACTION_ROTATE_LEFT, ACTION_STEER_LEFT)
                    else "right"
                    if previous_action in (ACTION_ROTATE_RIGHT, ACTION_STEER_RIGHT)
                    else None
                )
                opposite_yaw = bool(
                    desired_side is not None
                    and previous_side is not None
                    and desired_side != previous_side
                )
                self._action_runtime.begin_search_epoch(
                    decision.reason or "search_enter",
                    settle=opposite_yaw,
                )
                if opposite_yaw:
                    stop_reason = "search_handoff_reverse_brake"
                    self._clear_action_queue(stop_reason)
                    self._prepare_direct_stop(stop_reason)
                    # Search reversal is a normal control handoff, not a
                    # safety event. Clear both wheels with an explicit
                    # zero-RPM target so the old pulse cannot keep refreshing
                    # while the opposite search direction is being queued.
                    zero_start = time.monotonic()
                    try:
                        self._action_runtime.send_rotate_pulse_zero_stop()
                    except Exception as exc:
                        logger.warning("搜索方向切换双轮清零失败，退回安全停车: %s", exc)
                        self._action_runtime.send_stop_with_brake_hold(stop_reason)
                    else:
                        self._last_motor_dispatch_source = "search_handoff_zero"
                        logger.info(
                            "search direction handoff zero transition: control_frame_id=%d decision_capture_frame_id=%d reason=%s zero_rpm=0 elapsed_ms=%.1f",
                            int(self.frame_index),
                            int(getattr(self, "_last_decision_capture_frame", -1)),
                            stop_reason,
                            (time.monotonic() - zero_start) * 1000.0,
                        )
                    self._mark_direct_stop_sent(stop_reason)
                logger.info(
                    "search_motion_handoff frame=%d previous=%s previous_side=%s "
                    "desired=%s desired_side=%s reverse=%s settle=%s",
                    int(self.frame_index),
                    ACTION_NAMES.get(previous_action, str(previous_action)),
                    previous_side or "none",
                    desired_kind or "none",
                    desired_side or "none",
                    opposite_yaw,
                    opposite_yaw,
                )
            elif control_state.startswith("target_visible"):
                # A confirmed visible target transfers control back to normal
                # following. Cancel any post-pulse gate from the search that
                # found it, even if the visible state itself did not change.
                self._action_runtime.cancel_rotate_pulse_observation(
                    f"target_visible:{decision.reason or 'follow'}"
                )
            self._vision_control_state = control_state
        if not longitudinal_only:
            self.search_state = self._follow_controller.search_state
            self.search_direction = self._follow_controller.search_direction
            self.lost_confirm_frames = self._follow_controller.lost_confirm_frames
            self.last_action_frame = self._follow_controller.last_action_frame
            self.last_person_center_x = self._follow_controller.last_person_center_x
            self._waiting_lost_confirm = decision.waiting_lost_confirm
            self.person_detected_flag = decision.person_detected_flag
        self._explicit_stop_requested = decision.explicit_stop_requested
        self._last_explicit_stop_reason = decision.reason if decision.explicit_stop_requested else ""
        self._last_control_decision_reason = decision.reason or ""
        if not longitudinal_only:
            # The visible parked-target PID can legitimately settle at zero
            # yaw. Mark that queue stop as a coast/zero-target write so it
            # cannot enter the safety brake hold path and cause stop/restart
            # oscillation. Depth-only publications must not clear a pending
            # visual soft stop before the action thread consumes it.
            self._use_soft_stop_next = bool(
                getattr(decision, "soft_stop_requested", False)
            )
            if self._use_soft_stop_next:
                # A prior action switch may have left the asynchronous stop
                # flag set for one queue turn.  Soft PID settling owns this
                # transition and must not let that stale flag upgrade the
                # queued zero target into a brake-hold operation.
                self.stop_action_execution = False
                self.person_detected_flag = False
        if (
            longitudinal_only
            and not decision.actions
            and not decision.explicit_stop_requested
            and not decision.shutdown_requested
        ):
            # Missing/unchanged Depth is not a new motor target. Preserve the
            # latest visual speed, yaw correction and action queue.
            return []
        if decision.shutdown_requested:
            self._runtime_shutdown_requested = True
        self.is_forwarding = decision.is_forwarding
        self._current_forward_percent = decision.current_forward_percent
        if decision.stop_action_execution:
            self.stop_action_execution = True
        if decision.clear_action_queue:
            self._clear_action_queue(f"controller_clear:{decision.reason}")

        runtime_actions = list(decision.actions)
        selected_target = self._follow_controller.last_selected_target
        selected_target_id = None if selected_target is None else int(selected_target.track_id)
        if FOLLOW_ROTATION_ONLY:
            original_actions = runtime_actions
            runtime_actions = [
                self._rotation_only_action(action) for action in runtime_actions
            ]
            if runtime_actions != original_actions:
                logger.info(
                    "rotation_only action filter: frame=%d source=%s input=%s output=%s",
                    int(self.frame_index),
                    control_source,
                    [
                        (a.kind, int(a.speed_percent), int(a.steer_correction_rpm))
                        for a in original_actions
                    ],
                    [
                        (a.kind, int(a.speed_percent), int(a.steer_correction_rpm))
                        for a in runtime_actions
                    ],
                )
        elif control_source == "depth30":
            runtime_actions = [
                self._merge_longitudinal_action_with_visual_steering(action, selected_target_id)
                for action in runtime_actions
            ]
        else:
            self._remember_vision_steering(runtime_actions, selected_target_id)

        actions: List[int] = []
        for action in runtime_actions:
            if action.kind == "forward":
                self._current_forward_percent = self._stabilize_forward_percent(
                    int(action.speed_percent),
                    decision.reason,
                )
                self._current_steer_base_percent = 0
                self._current_steer_correction_rpm = 0
                self._current_steer_limit_reason = "none"
                self.is_forwarding = True
            elif action.kind == "backward":
                self._current_forward_percent = max(0, min(100, int(action.speed_percent)))
                self._forward_speed_latched_percent = None
                self._current_steer_base_percent = 0
                self._current_steer_correction_rpm = int(action.steer_correction_rpm)
                pid_result = self._follow_controller.last_steering_pid_result
                self._current_steer_limit_reason = (
                    "none" if pid_result is None else pid_result.correction_limit_reason
                )
                self.is_forwarding = False
            elif action.kind in ("steer_left", "steer_right"):
                self._current_steer_base_percent = int(action.speed_percent)
                self._current_steer_inner_ratio_percent = int(action.steer_inner_ratio_percent)
                self._current_steer_outer_ratio_percent = int(action.steer_outer_ratio_percent)
                self._current_steer_correction_rpm = int(action.steer_correction_rpm)
                pid_result = self._follow_controller.last_steering_pid_result
                self._current_steer_limit_reason = (
                    "none" if pid_result is None else pid_result.correction_limit_reason
                )
                self.is_forwarding = True
            elif action.kind in ("rotate_left", "rotate_right"):
                self._current_steer_correction_rpm = 0
                self._current_steer_limit_reason = "none"
                pid_result = self._follow_controller.last_steering_pid_result
                # A near-distance target is still visible, so its in-place yaw
                # must use the same camera-angle + encoder-rate PID as parked
                # recentering. Only fully lost search keeps a fixed scan RPM.
                parked_recenter = (
                    decision.reason.startswith("person_parked_recenter_")
                    or decision.reason.startswith("initial_candidate_centering_")
                    or decision.reason in (
                        "near_distance_rotation_only",
                        "target_visible_low_quality_yaw",
                    )
                )
                lost_confirm_hold = decision.reason.startswith(
                    (
                        "lost_current_candidate_hold_",
                        "lost_history_hold_",
                        "lost_wait_yaw_",
                    )
                )
                search_rotate = (
                    SEARCH_ROTATE_CONTINUOUS_ENABLE
                    and (
                        self._follow_controller.search_state
                        in ("searching", "timed_out")
                        or decision.reason.startswith(
                            ("search_", "predict_", "lost_wait_", "stale_probe_")
                        )
                    )
                )
                # Search rotation is a continuous hold. Visible near-distance
                # recentering also remains continuously refreshed by PID;
                # safety/target reacquisition still clears the command.
                self._current_rotate_pulse_enabled = (
                    not parked_recenter and not lost_confirm_hold and not search_rotate
                )
                if parked_recenter or lost_confirm_hold:
                    # 停车距离内的可见人物使用摄像头角度 + 编码器角速度闭环。
                    # 这里只禁止前进，原地转速按每帧 PID 输出限制在配置的
                    # 最小/最大 RPM；不使用搜索脉冲，视觉心跳超过 0.25 秒才
                    # 失效停车。完全丢失后的搜索仍走独立的固定低速配置。
                    if lost_confirm_hold:
                        # One or two fresh detector misses are not a search.
                        # Preserve only the last PID magnitude, bounded by the
                        # configured dropout hold limit. Previously this path
                        # fell through to the fixed 18 RPM search target.
                        hold_limit = max(
                            1,
                            int(VISIBLE_STEERING_PID_LOST_HOLD_MAX_CORRECTION_RPM),
                        )
                        previous_correction = (
                            0
                            if pid_result is None
                            else abs(int(pid_result.correction_rpm))
                        )
                        rotate_raw = min(hold_limit, max(1, previous_correction))
                        rotate_raw_source = "lost_confirm_pid_hold"
                    elif pid_result is None:
                        rotate_raw = PARKED_RECENTER_MIN_RPM
                        rotate_raw_source = "parked_recenter_fallback"
                    else:
                        rotate_raw = max(
                            PARKED_RECENTER_MIN_RPM,
                            min(PARKED_RECENTER_MAX_RPM, abs(int(pid_result.correction_rpm))),
                        )
                        rotate_raw_source = (
                            "parked_pid_encoder" if pid_result.feedback_used else "parked_pid_camera"
                        )
                else:
                    rotate_raw, rotate_raw_source = self._rotate_raw_target_for_reason(decision.reason)
                self._current_rotate_raw_target = int(rotate_raw)
                self._current_rotate_raw_source = rotate_raw_source
            action_int = self._action_kind_to_int(action.kind)
            if action_int is not None and action.kind != "idle":
                actions.append(action_int)

        lateral_intent_published = self._publish_lateral_intent_from_decision(
            width=int(width),
            target=selected_target,
            runtime_actions=runtime_actions,
            control_source=control_source,
            target_steerable=target_steerable,
            low_quality_visible=low_quality_visible,
        )
        if lateral_intent_published:
            logger.debug(
                "vision frame published to lateral action owner: frame=%d target=%s reason=%s",
                int(self.frame_index),
                selected_target_id,
                decision.reason,
            )

        if decision.reason:
            action_kinds = [a.kind for a in runtime_actions]
            target_dbg = self._follow_controller.last_selected_target
            largest_dbg = max(frame.persons, key=lambda p: p.area) if frame.persons else None
            target_id = None
            target_center = None
            target_area = "none"
            target_source = "none"
            target_region = "none"
            target_quality = "none"
            if target_dbg is not None:
                tx, ty = target_dbg.center
                target_id = int(target_dbg.track_id)
                target_center = (round(float(tx), 1), round(float(ty), 1))
                target_area = "%.0f" % float(target_dbg.area)
                target_region = self._target_region_debug(target_dbg, int(width), int(height))
                target_quality = self._target_quality_debug(target_dbg)
                if active_before is not None and int(target_id) == int(active_before):
                    target_source = "locked_id"
                elif search_before == "searching":
                    target_source = "search_visible_largest"
                elif active_before is None:
                    target_source = "new_visible_largest"
                else:
                    target_source = "free_search_visible_largest"
            largest_id = None
            largest_center = None
            if largest_dbg is not None:
                lx, ly = largest_dbg.center
                largest_id = int(largest_dbg.track_id)
                largest_center = (round(float(lx), 1), round(float(ly), 1))
            distance_dbg = "none" if frame.distance_m is None else "%.2fm" % float(frame.distance_m)
            log_key = (
                decision.reason,
                tuple(action_kinds),
                bool(decision.explicit_stop_requested),
                len(frame.persons),
                target_id,
                largest_id,
                target_source,
                None if active_before is None else int(active_before),
                None if self._follow_controller.active_target_id is None else int(self._follow_controller.active_target_id),
                self.search_state,
                self.search_direction,
            )
            if action_kinds or decision.explicit_stop_requested or log_key != self._last_control_decision_log_key:
                logger.info(
                    "控制决策: control_frame_id=%d capture_frame_id=%d 原因代码=%s 动作=%s 停车=%s 速度=%s 人数=%d 目标来源代码=%s 决策前活动目标=%s 决策后活动目标=%s 目标ID=%s 目标中心=%s 目标面积=%s 目标区域=(%s) 目标质量=(%s) 最大可见目标ID=%s 最大可见目标中心=%s 距离=%s 距离详情=(%s) 障碍物(前=%s,左=%s,右=%s) 搜索状态=%s/%s",
                    self.frame_index,
                    int(getattr(self, "_active_capture_frame_id", -1)),
                    decision.reason,
                    action_kinds,
                    decision.explicit_stop_requested,
                    self._current_forward_percent,
                    len(frame.persons),
                    target_source,
                    None if active_before is None else int(active_before),
                    None if self._follow_controller.active_target_id is None else int(self._follow_controller.active_target_id),
                    target_id,
                    target_center,
                    target_area,
                    target_region,
                    target_quality,
                    largest_id,
                    largest_center,
                    distance_dbg,
                    self._format_distance_state(frame.distance_state),
                    frame.obstacles.front,
                    frame.obstacles.left,
                    frame.obstacles.right,
                    self.search_state,
                    self.search_direction,
                )
                self._last_control_decision_log_key = log_key
            if decision.explicit_stop_requested:
                self._log_distance_stop_trigger(decision.reason, frame.distance_state)
            pid_result = self._follow_controller.last_steering_pid_result
            if pid_result is not None:
                logger.info(
                    "视觉转向PID: control_frame_id=%d capture_frame_id=%d 视觉误差=%.2f度 延迟补偿误差=%.2f度 滤波误差=%.2f度 误差变化率=%.2f度/秒 目标速度有效=%s 目标画面角速度=%+.2f度/秒 目标方位角速度=%+.2f度/秒 目标速度前馈=%+.2f度/秒 目标速度匹配=%s/%.2f度/秒 期望右转角速度=%.2f度/秒 当前角速度上限=%.2f度/秒 编码器右转角速度=%.2f度/秒 反馈年龄=%.0f毫秒 反馈有效=%s 换向制动=%s 同向超调制动=%s 视觉方向保护=%s 预测制动=%s 制动延迟=%.0f毫秒 剩余角=%.2f度 停车角=%.2f度 角速度超调=%.2f度/秒 超调制动=%.2fRPM 有效下限=%.1fRPM(%s) 启动补偿=%s/%.0f毫秒/%s 边缘增强=%s 原始前进=%dRPM 动态前进=%dRPM 当前轮差上限=%.2fRPM 轮差限幅原因=%s 前馈=%.2fRPM 速度P=%.2fRPM 速度I=%.2fRPM 输出轮差=%dRPM",
                    self.frame_index,
                    int(getattr(self, "_active_capture_frame_id", -1)),
                    pid_result.visual_error_deg,
                    pid_result.compensated_error_deg,
                    pid_result.filtered_error_deg,
                    pid_result.error_rate_dps,
                    pid_result.target_rate_valid,
                    pid_result.target_image_rate_dps,
                    pid_result.target_bearing_rate_dps,
                    pid_result.target_rate_feedforward_dps,
                    pid_result.target_speed_match_limited,
                    pid_result.target_speed_match_limit_dps,
                    pid_result.desired_yaw_rate_dps,
                    pid_result.yaw_rate_limit_dps,
                    pid_result.measured_yaw_rate_dps,
                    -1.0 if pid_result.feedback_age_sec is None else pid_result.feedback_age_sec * 1000.0,
                    pid_result.feedback_used,
                    pid_result.opposite_yaw_braking,
                    pid_result.same_direction_overspeed_braking,
                    pid_result.visual_direction_guarded,
                    pid_result.predictive_braking,
                    pid_result.prediction_latency_sec * 1000.0,
                    pid_result.remaining_error_deg,
                    pid_result.stopping_distance_deg,
                    pid_result.yaw_rate_overshoot_dps,
                    pid_result.overspeed_brake_rpm,
                    pid_result.output_floor_rpm,
                    pid_result.output_floor_reason,
                    pid_result.startup_kick_active,
                    pid_result.startup_kick_elapsed_sec * 1000.0,
                    pid_result.startup_kick_release_reason,
                    pid_result.edge_boost_active,
                    pid_result.requested_base_rpm,
                    pid_result.base_rpm,
                    pid_result.correction_limit_rpm,
                    pid_result.correction_limit_reason,
                    pid_result.feedforward_rpm,
                    pid_result.rate_p_rpm,
                    pid_result.rate_i_rpm,
                    pid_result.correction_rpm,
                )
                trace_target = target_dbg
                trace_action = "+".join(action_kinds) if action_kinds else "none"
                trace_reason = str(decision.reason or "")
                trace_relevant = bool(
                    trace_target is not None
                    and (
                        trace_reason.startswith((
                            "visual_pid_",
                            "near_distance_rotation_only",
                            "person_parked_recenter_",
                            "longitudinal_distance_pid",
                            "target_distance_",
                        ))
                        or any(k in ("steer_left", "steer_right", "rotate_left", "rotate_right") for k in action_kinds)
                    )
                )
                trace_now = time.monotonic()
                if trace_relevant and (
                    trace_now - self._last_pid_trace_log_ts >= PID_TRACE_INTERVAL_SEC
                ):
                    tx, ty = trace_target.center
                    x_ratio = float(tx) / float(max(1, width))
                    x1, y1, x2, y2 = (float(v) for v in trace_target.bbox)
                    motion_samples = getattr(self._follow_controller, "_visible_motion_samples", [])
                    dx_recent = 0.0
                    dx_history = 0.0
                    motion_dt_sec = 0.0
                    if len(motion_samples) >= 2:
                        dx_recent = float(motion_samples[-1][2]) - float(motion_samples[-2][2])
                        dx_history = float(motion_samples[-1][2]) - float(motion_samples[0][2])
                        motion_dt_sec = float(motion_samples[-1][1]) - float(motion_samples[-2][1])
                    feedback = frame.steering_feedback
                    feedback_raw = (
                        None
                        if feedback is None
                        else getattr(feedback, "raw_yaw_rate_right_dps", None)
                    )
                    distance_state = frame.distance_state
                    logger.info(
                        "pid_trace: control_frame_id=%d capture_frame_id=%d source=%s reason=%s action=%s target=%s "
                        "x=%.3f y=%.3f dx_recent=%+.3f dx_history=%+.3f motion_dt=%.3fs "
                        "bbox_w=%.1f bbox_h=%.1f area=%.0f area_ratio=%.3f quality=%s "
                        "distance=%s distance_source=%s distance_detail=%s "
                        "visual_error=%.2fdeg filtered=%.2fdeg target_rate_valid=%s target_image_rate=%+.2fdps "
                        "target_bearing_rate=%+.2fdps target_rate_ff=%+.2fdps "
                        "speed_match=%s/%.2fdps desired_yaw=%.2fdps "
                        "measured_yaw=%.2fdps raw_yaw=%s feedback_age_ms=%s feedback_used=%s "
                        "direction_guard=%s predictive_brake=%s brake_latency=%.0fms remaining=%.2fdeg stopping=%.2fdeg "
                        "correction=%+dRPM correction_limit=%.2fRPM limit_reason=%s floor_reason=%s "
                        "base=%dRPM rotate_raw=%d rotate_source=%s pulse=%s",
                        int(self.frame_index),
                        int(getattr(self, "_active_capture_frame_id", -1)),
                        control_source,
                        trace_reason,
                        trace_action,
                        int(trace_target.track_id),
                        x_ratio,
                        float(ty) / float(max(1, height)),
                        dx_recent,
                        dx_history,
                        motion_dt_sec,
                        max(0.0, x2 - x1),
                        max(0.0, y2 - y1),
                        float(trace_target.area),
                        float(trace_target.area) / float(max(1, width * height)),
                        target_quality,
                        "none" if frame.distance_m is None else "%.3f" % float(frame.distance_m),
                        str(getattr(distance_state, "source", "none")),
                        str(getattr(distance_state, "source_detail", "none")),
                        float(pid_result.visual_error_deg),
                        float(pid_result.filtered_error_deg),
                        bool(pid_result.target_rate_valid),
                        float(pid_result.target_image_rate_dps),
                        float(pid_result.target_bearing_rate_dps),
                        float(pid_result.target_rate_feedforward_dps),
                        bool(pid_result.target_speed_match_limited),
                        float(pid_result.target_speed_match_limit_dps),
                        float(pid_result.desired_yaw_rate_dps),
                        float(pid_result.measured_yaw_rate_dps),
                        "none" if feedback_raw is None else "%.2f" % float(feedback_raw),
                        "none" if pid_result.feedback_age_sec is None else "%.0f" % (float(pid_result.feedback_age_sec) * 1000.0),
                        bool(pid_result.feedback_used),
                        bool(pid_result.visual_direction_guarded),
                        bool(pid_result.predictive_braking),
                        float(pid_result.prediction_latency_sec) * 1000.0,
                        float(pid_result.remaining_error_deg),
                        float(pid_result.stopping_distance_deg),
                        int(pid_result.correction_rpm),
                        float(pid_result.correction_limit_rpm),
                        pid_result.correction_limit_reason,
                        pid_result.output_floor_reason,
                        int(pid_result.base_rpm),
                        int(self._current_rotate_raw_target),
                        self._current_rotate_raw_source,
                        bool(getattr(self, "_current_rotate_pulse_enabled", True)),
                    )
                    self._last_pid_trace_log_ts = trace_now
            if any(k in ("rotate_left", "rotate_right") for k in action_kinds):
                logger.info(
                    "旋转决策: control_frame_id=%d capture_frame_id=%d 原因代码=%s 动作=%s 目标ID=%s 目标中心=%s 目标区域=(%s) 目标质量=(%s) 最大目标ID=%s 搜索状态=%s/%s 脉冲启用=%s 持续=%.3f秒 停顿=%.3f秒 最少观察帧=%d 指令失效时间=%.3f秒 原始转速=%d 转速来源代码=%s 默认转速=%d",
                    self.frame_index,
                    int(getattr(self, "_active_capture_frame_id", -1)),
                    decision.reason,
                    action_kinds,
                    target_id,
                    target_center,
                    target_region,
                    target_quality,
                    largest_id,
                    self.search_state,
                    self.search_direction,
                    bool(
                        ROTATE_PULSE_BRAKE_ENABLE
                        and getattr(self, "_current_rotate_pulse_enabled", True)
                    ),
                    ROTATE_DURATION,
                    ROTATE_PULSE_PAUSE_SEC,
                    ROTATE_PULSE_OBSERVE_MIN_FRAMES,
                    ROTATE_HOLD_STALE_SEC,
                    int(self._current_rotate_raw_target),
                    self._current_rotate_raw_source,
                    MOTOR_ROTATE_RAW_TARGET,
                )
            if any(k in ("steer_left", "steer_right") for k in action_kinds):
                logger.info(
                    "steer decision: frame=%d reason=%s actions=%s target_id=%s target_center=%s largest_id=%s search=%s/%s base=%d correction_rpm=%d inner_ratio=%d outer_ratio=%d raw=%d",
                    self.frame_index,
                    decision.reason,
                    action_kinds,
                    target_id,
                    target_center,
                    largest_id,
                    self.search_state,
                    self.search_direction,
                    self._current_steer_base_percent,
                    self._current_steer_correction_rpm,
                    self._current_steer_inner_ratio_percent,
                    self._current_steer_outer_ratio_percent,
                    MOTOR_STEER_RAW_TARGET,
                )
        self._log_search_reid_diagnostics(frame, decision.reason)
        return actions

    @staticmethod
    def _fmt_optional_float(value: Any) -> str:
        if value is None:
            return "none"
        try:
            return "%.3f" % float(value)
        except Exception:
            return str(value)

    def _log_search_reid_diagnostics(self, frame: SensorFrame, reason: str) -> None:
        if not SEARCH_REID_DIAGNOSTIC_ENABLE:
            return
        if self.search_state != "searching":
            return
        if self._follow_controller.last_selected_target is not None:
            return
        if not frame.persons:
            return

        active_id = self._follow_controller.active_target_id
        parts = []
        for person in sorted(frame.persons, key=lambda p: p.area, reverse=True):
            candidates = self._last_person_reid_debug_by_stable_id.get(int(person.track_id), [])
            if not candidates:
                parts.append(
                    "stable=%s raw=none area=%.0f assign=missing"
                    % (int(person.track_id), float(person.area))
                )
                continue
            for cand in candidates:
                assign = cand.get("assignment") or {}
                dist = assign.get("distance")
                second = assign.get("second_distance")
                margin = None
                if dist is not None and second is not None:
                    try:
                        margin = float(second) - float(dist)
                    except Exception:
                        margin = None
                quality_ok = assign.get("bbox_quality_ok")
                quality_reason = assign.get("bbox_quality_reason")
                quality = str(quality_ok)
                if quality_reason:
                    quality += "(%s)" % quality_reason
                parts.append(
                    (
                        "stable=%s raw=%s out_uid=%s assign_uid=%s mapped_uid=%s reason=%s "
                        "best=%s dist=%s second=%s margin=%s gap=%s pending=%s quality=%s "
                        "score=%.3f area=%.0f bbox=%s"
                    )
                    % (
                        int(cand.get("stable_id", person.track_id)),
                        cand.get("raw_track_id"),
                        cand.get("reid_uid"),
                        assign.get("uid"),
                        assign.get("mapped_uid"),
                        assign.get("reason", "missing"),
                        assign.get("best_uid"),
                        self._fmt_optional_float(dist),
                        self._fmt_optional_float(second),
                        self._fmt_optional_float(margin),
                        assign.get("best_frame_gap"),
                        assign.get("pending_streak"),
                        quality,
                        float(cand.get("score", 0.0)),
                        float(cand.get("area", 0.0)),
                        cand.get("bbox"),
                    )
                )
        logger.info(
            "search_reid_diag frame=%d reason=%s active_target=%s search=%s/%s candidates=%s",
            self.frame_index,
            reason,
            "none" if active_id is None else int(active_id),
            self.search_state,
            self.search_direction,
            "; ".join(parts),
        )

    @staticmethod
    def _stable_id_from_track_record(rec: Any) -> int:
        reid_uid = int(rec.reid_uid)
        if reid_uid > 0:
            return reid_uid
        # Keep fallback DeepSORT track ids out of the positive ReID uid namespace.
        raw_track_id = int(rec.track_id)
        if raw_track_id >= 0:
            return -(raw_track_id + 1)
        return raw_track_id

    def _identity_assignment_debug_for_track(self, raw_track_id: int) -> Dict[str, Any]:
        pipeline = getattr(self, "_rknn_pipeline", None)
        tracker = getattr(pipeline, "tracker", None)
        identity_bank = getattr(tracker, "identity_bank", None)
        last_assignments = getattr(identity_bank, "last_assignments", None)
        if not isinstance(last_assignments, dict):
            return {}
        assignment = last_assignments.get(int(raw_track_id))
        if isinstance(assignment, dict):
            return dict(assignment)
        return {}

    @staticmethod
    def _track_record_bbox(rec: Any) -> Optional[Tuple[float, float, float, float]]:
        try:
            bbox = (
                float(rec.x1),
                float(rec.y1),
                float(rec.x2),
                float(rec.y2),
            )
        except (AttributeError, TypeError, ValueError):
            return None
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        return bbox

    def _active_target_track_record(
        self,
        records: Any,
        active_target_id: Optional[int],
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Resolve the fresh track still bound to the active identity.

        `best_uid` is intentionally excluded: it is only a nearest-neighbour
        hypothesis and may belong to a different person. `mapped_uid` is an
        established track-to-identity binding retained while a clipped or
        blurred bbox is temporarily withheld from normal control.
        """
        if active_target_id is None or int(active_target_id) <= 0:
            return None, {}
        active_uid = int(active_target_id)
        matches = []
        for rec in records or ():
            if int(getattr(rec, "class_id", -1)) != PERSON_CLASS_ID:
                continue
            if int(getattr(rec, "time_since_update", 0)) != 0:
                continue
            bbox = self._track_record_bbox(rec)
            if bbox is None:
                continue
            assignment = self._identity_assignment_debug_for_track(
                int(getattr(rec, "track_id", -1))
            )
            output_uid = int(getattr(rec, "reid_uid", 0))
            mapped_uid = int(assignment.get("mapped_uid") or 0)
            assignment_reason = str(assignment.get("reason") or "")
            rejected_mapping = assignment_reason in {
                "mapped_missing_identity",
                "mapped_verify_reject",
                "mapped_weak_distance_reject",
                "weak_preferred_reacquire_rejected",
            }
            match_level = (
                2
                if output_uid == active_uid
                else (
                    1
                    if mapped_uid == active_uid and not rejected_mapping
                    else 0
                )
            )
            if match_level <= 0:
                continue
            matches.append(
                (
                    match_level,
                    bool(assignment.get("bbox_quality_ok", True)),
                    float(getattr(rec, "score", 0.0)),
                    float(getattr(rec, "area", 0.0)),
                    rec,
                    assignment,
                )
            )
        if not matches:
            return None, {}
        _level, _quality, _score, _area, record, assignment = max(
            matches,
            key=lambda item: item[:4],
        )
        return record, dict(assignment)

    @staticmethod
    def _bbox_iou_xyxy(first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if intersection <= 0.0:
            return 0.0
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

    def _single_person_geometry_fallback_id(self, rec: Any, person_count: int) -> Optional[int]:
        """Return a safe stable id for one-person ReID gaps, otherwise None.

        A raw detector box is accepted only when it is a current (non-predicted)
        person and remains geometrically close to the previous single-person box.
        An already locked target may resume on the first continuous box; a new
        target needs the configured confirmation streak.
        """
        if not SINGLE_PERSON_GEOMETRY_FALLBACK_ENABLE or person_count != 1:
            self._single_person_geometry_streak = 0
            return None
        if int(getattr(rec, "time_since_update", 0)) > 0:
            self._single_person_geometry_streak = 0
            return None

        assignment = self._identity_assignment_debug_for_track(int(getattr(rec, "track_id", -1)))
        bbox = (float(rec.x1), float(rec.y1), float(rec.x2), float(rec.y2))
        current_frame = int(self.frame_index)
        previous_bbox = self._single_person_geometry_bbox
        previous_frame = int(self._single_person_geometry_frame)
        frame_gap = current_frame - previous_frame if previous_frame >= 0 else None
        continuous = False
        if previous_bbox is not None and frame_gap is not None and frame_gap <= SINGLE_PERSON_GEOMETRY_MAX_GAP_FRAMES:
            iou = self._bbox_iou_xyxy(previous_bbox, bbox)
            previous_center = (previous_bbox[0] + previous_bbox[2]) / 2.0
            current_center = (bbox[0] + bbox[2]) / 2.0
            width = max(1.0, float(VISION_FRAME_WIDTH))
            center_jump = abs(current_center - previous_center) / width
            continuous = iou >= SINGLE_PERSON_GEOMETRY_MIN_IOU or center_jump <= SINGLE_PERSON_GEOMETRY_MAX_CENTER_JUMP_RATIO

        self._single_person_geometry_bbox = bbox
        self._single_person_geometry_frame = current_frame
        self._single_person_geometry_track_id = int(getattr(rec, "track_id", -1))
        self._single_person_geometry_streak = self._single_person_geometry_streak + 1 if continuous else 1
        record_uid = int(getattr(rec, "reid_uid", 0))
        active_target_id = getattr(self._follow_controller, "active_target_id", None)
        held_uid = getattr(self, "_visible_unsteerable_uid", None)
        held_track_id = getattr(self, "_visible_unsteerable_track_id", None)
        held_last_ts = getattr(self, "_visible_unsteerable_last_ts", None)
        held_recent = bool(
            held_last_ts is not None
            and time.monotonic() - float(held_last_ts)
            <= VISIBLE_LOW_QUALITY_RECOVERY_MAX_GAP_SEC
        )
        recovering_same_track = bool(
            record_uid <= 0
            and active_target_id is not None
            and int(active_target_id) > 0
            and held_uid is not None
            and int(held_uid) == int(active_target_id)
            and held_track_id is not None
            and int(held_track_id) == int(getattr(rec, "track_id", -1))
            and held_recent
            and self._bound_reid_bbox_allowed_for_control(assignment)
        )
        if recovering_same_track:
            self._single_person_geometry_anchor_uid = int(active_target_id)
            self._single_person_geometry_unsteerable_uid = int(active_target_id)
            self._visible_unsteerable_recovery_frames += 1
            confirmed = (
                int(self._visible_unsteerable_recovery_frames)
                >= SINGLE_PERSON_GEOMETRY_CONFIRM_FRAMES
            )
            logger.info(
                "visible target geometry recovery: frame=%d track_id=%d reid_uid=%d "
                "confirmations=%d/%d confirmed=%s",
                current_frame,
                int(getattr(rec, "track_id", -1)),
                int(active_target_id),
                int(self._visible_unsteerable_recovery_frames),
                int(SINGLE_PERSON_GEOMETRY_CONFIRM_FRAMES),
                bool(confirmed),
            )
            return int(active_target_id) if confirmed else None
        if record_uid > 0:
            # A confirmed ReID frame starts a trusted continuity chain.
            self._single_person_geometry_anchor_uid = record_uid
        elif not continuous:
            # A locked formal target may start a two-frame geometry bridge when
            # ReID temporarily returns UID0. Low-quality fragments never reach
            # this method, so a 20x20 false box cannot establish this anchor.
            self._single_person_geometry_anchor_uid = (
                int(active_target_id)
                if (
                    previous_bbox is None
                    and active_target_id is not None
                    and int(active_target_id) > 0
                )
                else None
            )

        if not self._bound_reid_bbox_allowed_for_control(assignment):
            # Low-quality geometry must not become the next continuity anchor.
            # The caller may compare it read-only with the last reliable box to
            # prove visibility, but this cache remains high-quality-only.
            self._single_person_geometry_bbox = None
            self._single_person_geometry_frame = -1
            self._single_person_geometry_track_id = None
            self._single_person_geometry_streak = 0
            self._single_person_geometry_anchor_uid = None
            logger.info(
                "geometry_fallback_bbox_quality_rejected frame=%d track_id=%d reason=%s continuous=%s visible_uid=%s",
                int(self.frame_index),
                int(getattr(rec, "track_id", -1)),
                assignment.get("bbox_quality_reason", assignment.get("reason", "unknown")),
                bool(continuous),
                getattr(self, "_single_person_geometry_unsteerable_uid", None),
            )
            return None

        if record_uid > 0 or not continuous:
            return None
        if (
            active_target_id is not None
            and self._single_person_geometry_anchor_uid is not None
            and int(self._single_person_geometry_anchor_uid) == int(active_target_id)
        ):
            return int(active_target_id)
        if self._single_person_geometry_streak >= SINGLE_PERSON_GEOMETRY_CONFIRM_FRAMES:
            return self._stable_id_from_track_record(rec)
        return None

    @staticmethod
    def _bound_reid_bbox_allowed_for_control(assignment: Dict[str, Any]) -> bool:
        """Allow only complete, quality-approved boxes into normal control.

        Edge-cropped and distorted boxes are handled by the separate visible
        low-quality hold path, so they cannot reach the distance PID.
        """
        if assignment.get("bbox_quality_ok") is not False:
            return True
        reasons = [
            item.strip()
            for item in str(assignment.get("bbox_quality_reason") or "").split(",")
            if item.strip()
        ]
        return False

    @staticmethod
    def _bbox_quality_is_fragment(assignment: Dict[str, Any]) -> bool:
        reasons = [
            item.strip()
            for item in str(assignment.get("bbox_quality_reason") or "").split(",")
            if item.strip()
        ]
        fragment_prefixes = (
            "area<",
            "width<",
            "height<",
            "aspect<",
            "aspect>",
            "area_shrink<",
            "empty_bbox",
            "duplicate_person_box",
        )
        return any(item.startswith(fragment_prefixes) for item in reasons)

    @staticmethod
    def _bbox_quality_is_near_camera_occlusion(
        quality_reason: Optional[str],
        bbox: Optional[Tuple[float, float, float, float]],
        width: int,
        height: int,
    ) -> bool:
        """Recognize a locked person who is so close that the camera is nearly blocked.

        This is deliberately stricter than the normal large/cropped-box path. A
        single wide false positive must not stop the car permanently; the caller
        only invokes this after the box has already been mapped to the active UID.
        """
        if bbox is None or width <= 0 or height <= 0:
            return False
        reasons = {
            item.strip()
            for item in str(quality_reason or "").split(",")
            if item.strip()
        }
        if any(
            item.startswith((
                "area<",
                "width<",
                "height<",
                "aspect<",
                "aspect>",
                "area_shrink<",
                "duplicate_person_box",
            ))
            for item in reasons
        ):
            return False
        x1, y1, x2, y2 = (float(value) for value in bbox)
        bbox_width = max(0.0, x2 - x1)
        bbox_height = max(0.0, y2 - y1)
        area_ratio = (bbox_width * bbox_height) / max(1.0, float(width * height))
        width_ratio = bbox_width / max(1.0, float(width))
        height_ratio = bbox_height / max(1.0, float(height))
        edge_margin_x = 0.03 * float(width)
        edge_margin_y = 0.03 * float(height)
        edge_count = sum(
            (
                x1 <= edge_margin_x,
                y1 <= edge_margin_y,
                float(width) - x2 <= edge_margin_x,
                float(height) - y2 <= edge_margin_y,
            )
        )
        # Either dimension being nearly full is not enough on its own. Requiring
        # both large coverage and at least two clipped edges keeps ordinary
        # close-up, but still usable, detections on the normal coarse path.
        return bool(
            edge_count >= 2
            and (
                (
                    area_ratio >= VISIBLE_NEAR_CAMERA_OCCLUSION_AREA_RATIO
                    and height_ratio >= VISIBLE_NEAR_CAMERA_OCCLUSION_HEIGHT_RATIO
                )
                or (
                    width_ratio >= VISIBLE_NEAR_CAMERA_OCCLUSION_WIDTH_RATIO
                    and height_ratio >= max(0.82, VISIBLE_NEAR_CAMERA_OCCLUSION_HEIGHT_RATIO - 0.06)
                )
            )
        )

    def _reset_search_geometry_reacquire(self) -> None:
        self._search_geometry_reacquire_track_id = None
        self._search_geometry_reacquire_last_frame = -1
        self._search_geometry_reacquire_frames = 0
        self._search_geometry_reacquire_bbox = None

    def _search_geometry_reacquire_id(
        self,
        rec: Any,
        assignment: Dict[str, Any],
        *,
        person_count: int,
        width: int,
    ) -> Optional[int]:
        """Recover the locked UID from two unique, direction-consistent boxes."""
        if not SINGLE_PERSON_GEOMETRY_FALLBACK_ENABLE:
            self._reset_search_geometry_reacquire()
            return None
        active_target_id = getattr(self._follow_controller, "active_target_id", None)
        controller_search_state = getattr(self._follow_controller, "search_state", "none")
        search_active = bool(
            self.search_state == "searching"
            or controller_search_state == "searching"
        )
        fragment = self._bbox_quality_is_fragment(assignment)
        current_frame = int(self.frame_index)
        track_id = int(rec.track_id)
        bbox = (
            float(rec.x1),
            float(rec.y1),
            float(rec.x2),
            float(rec.y2),
        )
        if (
            active_target_id is None
            or int(active_target_id) <= 0
            or int(person_count) != 1
            or int(getattr(rec, "time_since_update", 0)) > 0
            or fragment
            or width <= 0
        ):
            self._reset_search_geometry_reacquire()
            return None

        mapped_uid = int(assignment.get("mapped_uid") or 0)
        best_uid = assignment.get("best_uid")
        best_distance = assignment.get("distance")
        if mapped_uid > 0 and mapped_uid != int(active_target_id):
            self._reset_search_geometry_reacquire()
            return None
        if (
            best_uid is not None
            and int(best_uid) > 0
            and int(best_uid) != int(active_target_id)
            and best_distance is not None
            and float(best_distance) <= SEARCH_CANDIDATE_CONTINUE_ROTATE_MAX_REID_DISTANCE
        ):
            self._reset_search_geometry_reacquire()
            return None

        previous_bbox = self._search_geometry_reacquire_bbox
        previous_frame = int(self._search_geometry_reacquire_last_frame)
        same_chain = bool(
            self._search_geometry_reacquire_track_id == track_id
            and previous_bbox is not None
            and 0 < current_frame - previous_frame <= SEARCH_GEOMETRY_REACQUIRE_MAX_GAP_FRAMES
            and (
                self._bbox_iou_xyxy(previous_bbox, bbox) >= SINGLE_PERSON_GEOMETRY_MIN_IOU
                or abs(
                    ((previous_bbox[0] + previous_bbox[2]) / 2.0)
                    - ((bbox[0] + bbox[2]) / 2.0)
                )
                / max(1.0, float(width))
                <= SINGLE_PERSON_GEOMETRY_MAX_CENTER_JUMP_RATIO
            )
        )
        if (
            not search_active
            and same_chain
            and self._search_geometry_reacquire_frames >= SEARCH_GEOMETRY_REACQUIRE_FRAMES
        ):
            self._search_geometry_reacquire_last_frame = current_frame
            self._search_geometry_reacquire_bbox = bbox
            logger.info(
                "search_geometry_reacquire_hold frame=%d track_id=%d active_target=%d "
                "gap=%d quality=%s",
                current_frame,
                track_id,
                int(active_target_id),
                current_frame - previous_frame,
                assignment.get("bbox_quality_reason", "ok"),
            )
            return int(active_target_id)
        if not search_active:
            self._reset_search_geometry_reacquire()
            return None

        center_ratio = (bbox[0] + bbox[2]) / (2.0 * float(width))
        direction = self.search_direction
        if direction not in ("left", "right"):
            direction = getattr(self._follow_controller, "search_direction", None)
        direction_consistent = bool(
            (direction == "left" and center_ratio <= 0.55)
            or (direction == "right" and center_ratio >= 0.45)
        )
        if not direction_consistent:
            self._reset_search_geometry_reacquire()
            return None

        self._search_geometry_reacquire_frames = (
            int(self._search_geometry_reacquire_frames) + 1 if same_chain else 1
        )
        self._search_geometry_reacquire_track_id = track_id
        self._search_geometry_reacquire_last_frame = current_frame
        self._search_geometry_reacquire_bbox = bbox
        confirmed = self._search_geometry_reacquire_frames >= SEARCH_GEOMETRY_REACQUIRE_FRAMES
        logger.info(
            "search_geometry_reacquire frame=%d track_id=%d active_target=%d "
            "direction=%s center=%.3f confirmations=%d/%d confirmed=%s quality=%s",
            current_frame,
            track_id,
            int(active_target_id),
            direction,
            center_ratio,
            int(self._search_geometry_reacquire_frames),
            int(SEARCH_GEOMETRY_REACQUIRE_FRAMES),
            bool(confirmed),
            assignment.get("bbox_quality_reason", "ok"),
        )
        return int(active_target_id) if confirmed else None

    def _reset_confirmed_search_reacquire(self) -> None:
        self._confirmed_search_reacquire_uid = None
        self._confirmed_search_reacquire_track_id = None
        self._confirmed_search_reacquire_bbox = None
        self._confirmed_search_reacquire_last_frame = -1
        self._confirmed_search_reacquire_streak = 0

    def _publish_observation_soft_zero(self, reason: str) -> None:
        """Publish a zero drive intent without entering motor brake hold."""
        self._last_control_decision_reason = str(reason)
        self._clear_lateral_intent(reason)
        self._clear_longitudinal_context()
        self._current_forward_percent = 0
        self._current_steer_base_percent = 0
        self._current_steer_correction_rpm = 0
        self._current_steer_limit_reason = str(reason)
        self._current_rotate_raw_target = 0
        self._current_rotate_raw_source = str(reason)
        self._current_rotate_pulse_enabled = False
        self._explicit_stop_requested = False
        self._use_soft_stop_next = False
        self.is_forwarding = False
        # ACTION_FORWARD at 0% is an ordinary publisher command. It stops yaw
        # without creating a brake latch that a later candidate yaw must unlock.
        self._replace_action_queue([ACTION_FORWARD], reason)

    def _hold_for_confirmed_search_reacquire(
        self,
        selected_candidates: List[Dict[str, Any]],
        *,
        width: int,
    ) -> bool:
        """Require a short stable UID chain before ending a search session."""
        status_getter = getattr(self._follow_controller, "search_status", None)
        status = status_getter(time.monotonic()) if callable(status_getter) else None
        search_state = (
            getattr(status, "state", None)
            if status is not None
            else getattr(self._follow_controller, "search_state", self.search_state)
        )
        search_active = search_state == "searching"
        active_uid = (
            getattr(status, "active_target_id", None)
            if status is not None
            else getattr(self._follow_controller, "active_target_id", None)
        )
        confirmed = [
            candidate
            for candidate in selected_candidates
            if int(candidate.get("stable_id", 0)) > 0
            and not bool(candidate.get("geometry_fallback", False))
            and (
                active_uid is None
                or int(candidate.get("stable_id", 0)) == int(active_uid)
            )
        ]
        if not search_active or len(confirmed) != 1:
            self._reset_confirmed_search_reacquire()
            return False

        candidate = confirmed[0]
        uid = int(candidate["stable_id"])
        rec = candidate["rec"]
        track_id = int(rec.track_id)
        bbox = tuple(float(value) for value in candidate["bbox"])
        previous_bbox = self._confirmed_search_reacquire_bbox
        frame_gap = int(self.frame_index) - int(self._confirmed_search_reacquire_last_frame)
        same_chain = bool(
            self._confirmed_search_reacquire_uid == uid
            and previous_bbox is not None
            and frame_gap == 1
            and (
                self._bbox_iou_xyxy(previous_bbox, bbox) >= SINGLE_PERSON_GEOMETRY_MIN_IOU
                or abs(
                    ((previous_bbox[0] + previous_bbox[2]) / 2.0)
                    - ((bbox[0] + bbox[2]) / 2.0)
                )
                / max(1.0, float(width))
                <= SINGLE_PERSON_GEOMETRY_MAX_CENTER_JUMP_RATIO
            )
        )
        streak = int(self._confirmed_search_reacquire_streak) + 1 if same_chain else 1
        self._confirmed_search_reacquire_uid = uid
        self._confirmed_search_reacquire_track_id = track_id
        self._confirmed_search_reacquire_bbox = bbox
        self._confirmed_search_reacquire_last_frame = int(self.frame_index)
        self._confirmed_search_reacquire_streak = streak
        required = int(SEARCH_CONFIRMED_REACQUIRE_FRAMES)
        if streak >= required:
            logger.info(
                "confirmed_search_reacquire frame=%d uid=%d track_id=%d "
                "confirmations=%d/%d result=resume_follow",
                int(self.frame_index),
                uid,
                track_id,
                streak,
                required,
            )
            self._reset_confirmed_search_reacquire()
            return False

        reason = "confirmed_search_reacquire_wait"
        self._follow_controller.set_search_observation_hold(True)
        self._publish_observation_soft_zero(reason)
        logger.info(
            "confirmed_search_reacquire frame=%d uid=%d track_id=%d "
            "confirmations=%d/%d result=hold_search bbox=%s",
            int(self.frame_index),
            uid,
            track_id,
            streak,
            required,
            bbox,
        )
        return True

    def _apply_search_candidate_gate_decision(
        self,
        decision: SearchCandidateGateDecision,
    ) -> None:
        if not decision.pause_rotation:
            return
        # With single-frame evidence the detector result is already decisive.
        # Do not insert even one observation-stop tick; the controller may
        # switch the frozen search direction and publish it immediately.
        if decision.completed and int(decision.hold_frames) <= 1:
            logger.info(
                "search_evidence_gate immediate_resume source=%s reason=%s",
                decision.source,
                decision.reason,
            )
            return
        reason = "search_candidate_evidence_observe"
        if decision.defer_sec > 0.0:
            self._follow_controller.defer_search_timeout(decision.defer_sec)
        self._publish_observation_soft_zero(reason)
        if decision.entered:
            # Candidate evidence is a detector-layer hint, not a confirmed
            # identity. Give the stationary detector a short bounded window;
            # never inherit the encoder settle gate, which can last over 1 s.
            self._action_runtime.cancel_rotate_pulse_observation(
                "search_candidate_bounded_observe"
            )
            self._search_evidence_observation_active = True
            self._search_evidence_observation_source = str(decision.source)
            self._search_evidence_observation_deadline = (
                time.monotonic() + SEARCH_EVIDENCE_MAX_HOLD_SEC
            )
        logger.info(
            "search_evidence_gate frame=%d source=%s reason=%s score=%.3f bbox=%s "
            "observe=%d/%d entered=%s completed=%s pause_rotation=True "
            "identity_claim=False reid_update=False pid_input=False defer=%.3fs "
            "max_hold=%.3fs",
            int(self.frame_index),
            decision.source,
            decision.reason,
            float(decision.score),
            decision.bbox,
            int(decision.hold_frame),
            int(decision.hold_frames),
            bool(decision.entered),
            bool(decision.completed),
            float(decision.defer_sec),
            SEARCH_EVIDENCE_MAX_HOLD_SEC,
        )

    def _hold_for_visible_unsteerable_target(
        self,
        *,
        width: int,
        height: int,
        track_id: Optional[int],
        reid_uid: Optional[int],
        bbox: Optional[Tuple[float, float, float, float]],
        score: Optional[float],
        area: Optional[float],
        quality_reason: Optional[str],
    ) -> bool:
        """Handle a mapped low-quality target without granting full control trust."""
        now_mono = time.monotonic()
        active_target_id = getattr(self._follow_controller, "active_target_id", None)
        eligible = bool(
            track_id is not None
            and reid_uid is not None
            and int(reid_uid) > 0
            and bbox is not None
            and score is not None
            and area is not None
            and active_target_id is not None
            and int(reid_uid) == int(active_target_id)
        )
        if not eligible:
            self._finish_visible_unsteerable_hold(now_mono, "target_not_visible")
            return False

        track_id = int(track_id)
        reid_uid = int(reid_uid)
        same_target = bool(
            getattr(self, "_visible_unsteerable_uid", None) == reid_uid
        )
        if same_target:
            last_ts = getattr(self, "_visible_unsteerable_last_ts", None)
            if last_ts is not None:
                self._follow_controller.defer_search_timeout(
                    max(0.0, now_mono - float(last_ts))
                )
        else:
            self._finish_visible_unsteerable_hold(now_mono, "target_changed")
            self._visible_unsteerable_uid = reid_uid
            logger.info(
                "visible unsteerable hold start: frame=%d track_id=%d reid_uid=%d quality=%s search=%s/%s",
                int(self.frame_index),
                track_id,
                reid_uid,
                quality_reason or "unknown",
                self.search_state,
                self.search_direction,
            )

        self._visible_unsteerable_track_id = track_id
        self._visible_unsteerable_last_ts = now_mono
        self._visible_unsteerable_bbox = tuple(float(value) for value in bbox)
        synthetic_person = (
            tuple(float(value) for value in bbox),
            reid_uid,
            float(score),
            float(area),
        )
        persons = [synthetic_person]
        near_camera_occlusion = PersonTracker._bbox_quality_is_near_camera_occlusion(
            quality_reason,
            bbox,
            int(width),
            int(height),
        )
        controller_search_state = str(
            getattr(self._follow_controller, "search_state", "none") or "none"
        )
        center_ratio = (
            float(bbox[0]) + float(bbox[2])
        ) / (2.0 * float(max(1, width)))
        center_half_width = max(
            0.01,
            0.5 * abs(FOLLOW_CENTER_RIGHT_RATIO - FOLLOW_CENTER_LEFT_RATIO),
        )
        edge_strength = max(
            0.0,
            min(
                1.0,
                (abs(center_ratio - 0.5) - center_half_width)
                / max(0.01, 0.40 - center_half_width),
            ),
        )
        yaw_limit_rpm = None
        if not near_camera_occlusion:
            yaw_limit_rpm = (
                VISIBLE_LOW_QUALITY_STEER_MAX_CORRECTION_RPM
                + edge_strength
                * (
                    VISIBLE_LOW_QUALITY_STEER_EDGE_MAX_CORRECTION_RPM
                    - VISIBLE_LOW_QUALITY_STEER_MAX_CORRECTION_RPM
                )
            )
        if controller_search_state == "searching":
            note_direction = getattr(
                self._follow_controller,
                "note_search_candidate_evidence",
                None,
            )
            if callable(note_direction):
                note_direction(
                    bbox,
                    frame_width=int(width),
                    confirmed=True,
                    source="low_quality_search",
                    candidate_score=float(score),
                    candidate_tracked=True,
                    now=now_mono,
                )
            logger.info(
                "search_low_quality_evidence frame=%d track_id=%d reid_uid=%d "
                "center=%.3f action=direction_only reason=quality_rejected",
                int(self.frame_index),
                track_id,
                reid_uid,
                center_ratio,
            )
            # A visible crop is current lateral evidence, even though it is
            # insufficient for identity or forward motion. Pause the 18 RPM
            # search command and use the same bounded camera/encoder yaw path
            # as other low-quality frames.
            self._queue_actions_for_persons(
                int(width),
                int(height),
                persons,
                target_steerable=False,
                target_steering_limit_rpm=yaw_limit_rpm,
                record_target_motion=True,
                low_quality_visible=True,
            )
            self._clear_longitudinal_context()
            return True
        # The crop remains excluded from Depth, ReID updates and longitudinal
        # control. A mapped non-occluding edge crop may publish bounded yaw and
        # contribute only to the short-horizon lateral motion estimate.
        self._queue_actions_for_persons(
            int(width),
            int(height),
            persons,
            target_steerable=False,
            target_steering_limit_rpm=yaw_limit_rpm,
            record_target_motion=True,
            low_quality_visible=True,
        )
        logger.info(
            "visible unsteerable hold: frame=%d track_id=%d reid_uid=%d quality=%s "
            "timeout_paused=True mode=%s action=%s center=%.3f yaw_limit=%s",
            int(self.frame_index),
            track_id,
            reid_uid,
            quality_reason or "unknown",
            "near_camera_occlusion" if near_camera_occlusion else "edge_or_large_box",
            "stop" if yaw_limit_rpm is None else "limited_yaw",
            center_ratio,
            "none" if yaw_limit_rpm is None else f"{yaw_limit_rpm:.1f}rpm",
        )
        return True

    def _finish_visible_unsteerable_hold(self, now_mono: float, reason: str) -> None:
        reid_uid = getattr(self, "_visible_unsteerable_uid", None)
        self._visible_unsteerable_recovery_frames = 0
        if reid_uid is None:
            return
        last_ts = getattr(self, "_visible_unsteerable_last_ts", None)
        deferred = 0.0
        if last_ts is not None:
            deferred = max(0.0, float(now_mono) - float(last_ts))
            self._follow_controller.defer_search_timeout(deferred)
        logger.info(
            "visible unsteerable hold end: frame=%d track_id=%s reid_uid=%d reason=%s timeout_deferred=%.3fs",
            int(self.frame_index),
            getattr(self, "_visible_unsteerable_track_id", None),
            int(reid_uid),
            reason,
            deferred,
        )
        self._visible_unsteerable_uid = None
        self._visible_unsteerable_track_id = None
        self._visible_unsteerable_last_ts = None
        self._visible_unsteerable_bbox = None

    def _consume_track_records(
        self,
        records,
        width: int,
        height: int,
        source_name: str,
        *,
        lateral_candidate: Optional[LateralCandidateEvidence] = None,
    ) -> None:
        # 新视觉结果到达后先撤销旧框发布；只有本帧通过ReID与质量门控后才重新发布。
        self._clear_longitudinal_context()
        all_dets = []
        person_candidates = []
        person_track_debug = []
        skipped_predicted_tracks = 0
        skipped_unconfirmed_reid = 0
        skipped_low_quality_reid = 0
        geometry_fallback_count = 0
        visible_unsteerable_candidate = None
        self._single_person_geometry_unsteerable_uid = None
        reid_debug_by_stable_id: Dict[int, List[Dict[str, Any]]] = {}
        single_person_records = [
            rec
            for rec in records
            if int(getattr(rec, "class_id", -1)) == PERSON_CLASS_ID
            and float(getattr(rec, "score", 0.0)) > CONFIDENCE_THRESHOLD
            and int(getattr(rec, "time_since_update", 0)) == 0
        ]
        reliable_person_records = [
            rec
            for rec in single_person_records
            if self._bound_reid_bbox_allowed_for_control(
                self._identity_assignment_debug_for_track(int(rec.track_id))
            )
        ]
        ignored_low_quality_persons = len(single_person_records) - len(reliable_person_records)
        search_reacquire_uid = None
        search_evidence_pause = bool(
            getattr(self, "_search_evidence_pause_current_frame", False)
        )
        if len(single_person_records) == 1 and not search_evidence_pause:
            search_rec = single_person_records[0]
            search_assignment = self._identity_assignment_debug_for_track(
                int(search_rec.track_id)
            )
            search_reacquire_uid = self._search_geometry_reacquire_id(
                search_rec,
                search_assignment,
                person_count=1,
                width=int(width),
            )
        else:
            self._reset_search_geometry_reacquire()
        if len(reliable_person_records) > 1:
            # More than one current person breaks the unique-person continuity
            # proof. A later UID0 frame must first reconnect through real ReID.
            self._single_person_geometry_bbox = None
            self._single_person_geometry_frame = -1
            self._single_person_geometry_track_id = None
            self._single_person_geometry_streak = 0
            self._single_person_geometry_anchor_uid = None
        geometry_fallback_rec = (
            reliable_person_records[0]
            if len(reliable_person_records) == 1
            else None
        )
        geometry_fallback_id = (
            self._single_person_geometry_fallback_id(geometry_fallback_rec, 1)
            if geometry_fallback_rec is not None and not search_evidence_pause
            else None
        )
        if VISION_TRACK_LOG_ENABLE:
            if records:
                track_parts = []
                for rec in records:
                    bbox_dbg = [round(float(v), 1) for v in (rec.x1, rec.y1, rec.x2, rec.y2)]
                    track_parts.append(
                        "轨迹ID=%s ReID编号=%s 状态=%s 距上次更新帧数=%s 类别=%s 置信度=%.3f 面积=%.0f 边界框=%s"
                        % (
                            int(rec.track_id),
                            int(rec.reid_uid),
                            _tracker_state_name(int(rec.tracker_state)),
                            int(getattr(rec, "time_since_update", 0)),
                            int(rec.class_id),
                            float(rec.score),
                            float(rec.area),
                            bbox_dbg,
                        )
                    )
                logger.info(
                    "%s 视觉轨迹: 帧=%d 数量=%d: %s",
                    source_name,
                    self.frame_index,
                    len(records),
                    "; ".join(track_parts),
                )
            elif VISION_TRACK_LOG_EMPTY_EVERY > 0 and self.frame_index % VISION_TRACK_LOG_EMPTY_EVERY == 0:
                logger.info("%s 视觉轨迹: 帧=%d 数量=0", source_name, self.frame_index)

        for rec in records:
            bbox = (float(rec.x1), float(rec.y1), float(rec.x2), float(rec.y2))
            all_dets.append({
                "class_id": int(rec.class_id),
                "score": float(rec.score),
                "bbox": bbox,
            })
            if int(rec.class_id) != PERSON_CLASS_ID:
                continue
            if float(rec.score) <= CONFIDENCE_THRESHOLD:
                continue
            time_since_update = int(getattr(rec, "time_since_update", 0))
            if time_since_update > 0 and not VISION_CONTROL_USE_PREDICTED_TRACKS:
                skipped_predicted_tracks += 1
                person_track_debug.append((float(rec.area), self._stable_id_from_track_record(rec), rec, None))
                continue
            area = float(rec.area)
            stable_id = self._stable_id_from_track_record(rec)
            reid_uid = int(rec.reid_uid)
            assignment = self._identity_assignment_debug_for_track(int(rec.track_id))
            debug_entry = {
                "stable_id": int(stable_id),
                "raw_track_id": int(rec.track_id),
                "reid_uid": int(reid_uid),
                "time_since_update": int(time_since_update),
                "score": float(rec.score),
                "area": float(area),
                "bbox": [round(float(v), 1) for v in bbox],
                "assignment": assignment,
            }
            reid_debug_by_stable_id.setdefault(int(stable_id), []).append(debug_entry)
            if VISION_REID_ENABLE and not self._bound_reid_bbox_allowed_for_control(assignment):
                # 已绑定身份也不能绕过正常控制质量门槛。裁剪型大框可在
                # 后续独立路径受限转向；其他拒绝框只保留诊断记录。
                skipped_low_quality_reid += 1
                person_track_debug.append((area, stable_id, rec, None))
                mapped_uid = int(assignment.get("mapped_uid") or reid_uid or 0)
                fragment = PersonTracker._bbox_quality_is_fragment(assignment)
                active_target_id = getattr(self._follow_controller, "active_target_id", None)
                if (
                    len(single_person_records) == 1
                    and active_target_id is not None
                    and mapped_uid > 0
                    and mapped_uid == int(active_target_id)
                    and not fragment
                ):
                    visible_unsteerable_candidate = {
                        "track_id": int(rec.track_id),
                        "reid_uid": mapped_uid,
                        "bbox": bbox,
                        "score": float(rec.score),
                        "area": area,
                        "quality_reason": assignment.get(
                            "bbox_quality_reason",
                            assignment.get("reason", "unknown"),
                        ),
                    }
                logger.info(
                    "control_bbox_quality_rejected frame=%d track_id=%d reid_uid=%d "
                    "mapped_uid=%d area=%.0f fragment=%s reason=%s "
                    "effects=normal_control_reid_update_depth_longitudinal_blocked,lateral_geometry_rate_allowed",
                    int(self.frame_index),
                    int(rec.track_id),
                    int(reid_uid),
                    mapped_uid,
                    area,
                    fragment,
                    assignment.get("bbox_quality_reason", assignment.get("reason", "unknown")),
                )
                continue
            if VISION_REID_ENABLE and reid_uid <= 0:
                fallback_id = (
                    int(search_reacquire_uid)
                    if search_reacquire_uid is not None
                    else geometry_fallback_id
                )
                if fallback_id is None or rec is not geometry_fallback_rec:
                    skipped_unconfirmed_reid += 1
                    person_track_debug.append((area, stable_id, rec, None))
                    continue
                stable_id = int(fallback_id)
                debug_entry["geometry_fallback"] = True
                reid_debug_by_stable_id.setdefault(int(stable_id), []).append(debug_entry)
                geometry_fallback_count += 1
            candidate = {
                "bbox": bbox,
                "stable_id": stable_id,
                "score": float(rec.score),
                "area": area,
                "rec": rec,
                "geometry_fallback": bool(debug_entry.get("geometry_fallback", False)),
                "debug": debug_entry,
            }
            person_candidates.append(candidate)
            person_track_debug.append((area, stable_id, rec, candidate))
        self._last_person_reid_debug_by_stable_id = reid_debug_by_stable_id

        selected_candidates = person_candidates
        persons = [
            (
                cand["bbox"],
                int(cand["stable_id"]),
                float(cand["score"]),
                float(cand["area"]),
            )
            for cand in selected_candidates
        ]

        if VISION_TRACK_LOG_ENABLE:
            if selected_candidates:
                selected_cand = max(selected_candidates, key=lambda cand: float(cand.get("area", 0.0)))
                selected_rec = selected_cand["rec"]
                selected_id = int(selected_cand["stable_id"])
                cx = (float(selected_rec.x1) + float(selected_rec.x2)) / 2.0
                cy = (float(selected_rec.y1) + float(selected_rec.y2)) / 2.0
                logger.info(
                    "%s 最大可见目标: 帧=%d 模式=最大面积 稳定ID=%s 轨迹ID=%s ReID编号=%s 状态=%s 距上次更新帧数=%s 置信度=%.3f 面积=%.0f 中心=(%.1f, %.1f) 几何兜底=%s",
                    source_name,
                    self.frame_index,
                    int(selected_id),
                    int(selected_rec.track_id),
                    int(selected_rec.reid_uid),
                    _tracker_state_name(int(selected_rec.tracker_state)),
                    int(getattr(selected_rec, "time_since_update", 0)),
                    float(selected_rec.score),
                    float(selected_rec.area),
                    cx,
                    cy,
                    bool(selected_cand.get("geometry_fallback", False)),
                )
            elif person_track_debug:
                logger.info(
                    "%s 最大可见目标: 帧=%d 模式=最大面积 候选=无 人员记录=%d 跳过预测轨迹=%d 跳过未确认ReID=%d 跳过低质量ReID=%d 控制使用预测轨迹=%s 要求ReID=%s",
                    source_name,
                    self.frame_index,
                    len(person_track_debug),
                    int(skipped_predicted_tracks),
                    int(skipped_unconfirmed_reid),
                    int(skipped_low_quality_reid),
                    bool(VISION_CONTROL_USE_PREDICTED_TRACKS),
                    bool(VISION_REID_ENABLE),
                )
            elif records:
                logger.info(
                    "%s 帧=%d 筛选后无有效人员 记录数=%d 人员类别=%d 置信度阈值=%.3f",
                    source_name,
                    self.frame_index,
                    len(records),
                    PERSON_CLASS_ID,
                    CONFIDENCE_THRESHOLD,
                )
            if (
                skipped_predicted_tracks > 0
                or skipped_unconfirmed_reid > 0
                or skipped_low_quality_reid > 0
            ) and selected_candidates:
                logger.info(
                    "%s 控制筛选: 帧=%d 跳过预测轨迹=%d 跳过未确认ReID=%d 跳过低质量ReID=%d 接受人员数=%d 控制使用预测轨迹=%s 要求ReID=%s",
                    source_name,
                    self.frame_index,
                    int(skipped_predicted_tracks),
                    int(skipped_unconfirmed_reid),
                    int(skipped_low_quality_reid),
                    len(selected_candidates),
                    bool(VISION_CONTROL_USE_PREDICTED_TRACKS),
                    bool(VISION_REID_ENABLE),
                )
            if geometry_fallback_count:
                logger.info(
                    "%s 单人几何兜底: 帧=%d 接受人数=%d 原因=ReID暂时无uid且检测框连续 稳定ID=%s",
                    source_name,
                    self.frame_index,
                    geometry_fallback_count,
                    geometry_fallback_id,
                )

        if self._handle_hazard_safety_state(
            self._bunker_runtime.check_merged_dets(all_dets, float(width * height))
        ):
            return

        if not selected_candidates and visible_unsteerable_candidate is not None:
            logger.info(
                "visible_active_low_quality_candidate frame=%d track_id=%d mapped_uid=%d "
                "area=%.0f quality=%s action=separate_hold_path",
                int(self.frame_index),
                int(visible_unsteerable_candidate["track_id"]),
                int(visible_unsteerable_candidate["reid_uid"]),
                float(visible_unsteerable_candidate["area"]),
                str(visible_unsteerable_candidate["quality_reason"] or "unknown"),
            )
            if self._hold_for_visible_unsteerable_target(
                width=int(width),
                height=int(height),
                **visible_unsteerable_candidate,
            ):
                # The hold path deliberately never publishes longitudinal
                # context. Returning here also prevents the empty-person path
                # from incrementing lost confirmation in the same frame.
                self._clear_longitudinal_context()
                return

        if getattr(self, "_visible_unsteerable_uid", None) is not None:
            self._finish_visible_unsteerable_hold(
                time.monotonic(),
                "complete_or_unmapped_frame",
            )

        candidate_track_id = (
            int(reliable_person_records[0].track_id)
            if len(reliable_person_records) == 1
            else None
        )
        candidate_assignment = (
            self._identity_assignment_debug_for_track(candidate_track_id)
            if candidate_track_id is not None
            else {}
        )
        candidate_best_uid = candidate_assignment.get("best_uid")
        candidate_reid_distance = candidate_assignment.get("distance")
        active_candidate_uid = getattr(self._follow_controller, "active_target_id", None)
        candidate_person = None
        if candidate_track_id is not None and active_candidate_uid is not None:
            candidate_record = reliable_person_records[0]
            candidate_bbox = (
                float(candidate_record.x1),
                float(candidate_record.y1),
                float(candidate_record.x2),
                float(candidate_record.y2),
            )
            candidate_person = (
                candidate_bbox,
                int(active_candidate_uid),
                float(candidate_record.score),
                float(candidate_record.area),
            )
        confirmed_person_count = sum(
            1
            for candidate in selected_candidates
            if not bool(candidate.get("geometry_fallback", False))
        )
        confirmed_control_person_count = sum(
            1
            for candidate in selected_candidates
            if not bool(candidate.get("geometry_fallback", False))
            and (
                active_candidate_uid is None
                or int(candidate.get("stable_id", 0)) == int(active_candidate_uid)
            )
        )
        if self._hold_for_confirmed_search_reacquire(
            selected_candidates,
            width=int(width),
        ):
            # A real UID is present, but one frame is not enough to terminate a
            # long-running search. Keep the chassis stopped and let the next
            # consecutive frame confirm the same visual chain.
            return
        if (
            active_candidate_uid is None
            and confirmed_person_count == 0
            and not selected_candidates
            and len(single_person_records) == 1
        ):
            # Do not steer from an unconfirmed or low-quality startup box.
            # Initial acquisition must complete the same formal ReID path as
            # reacquisition; otherwise a fragment can create a synthetic
            # geometry target before identity is known.
            startup_record = single_person_records[0]
            startup_assignment = self._identity_assignment_debug_for_track(
                int(startup_record.track_id)
            )
            logger.info(
                "initial candidate held: frame=%d track_id=%d reid_uid=%d "
                "quality=%s reason=await_formal_reid_confirmation",
                int(self.frame_index),
                int(startup_record.track_id),
                int(startup_record.reid_uid),
                str(startup_assignment.get("bbox_quality_reason") or "ok"),
            )
        if (
            candidate_track_id is not None
            and confirmed_control_person_count == 0
            and skipped_unconfirmed_reid > 0
        ):
            logger.info(
                "search_candidate_observation_only frame=%d track_id=%d "
                "action=keep_frozen_search identity_claim=False",
                int(self.frame_index),
                int(candidate_track_id),
            )

        if ignored_low_quality_persons > 0 and not selected_candidates:
            logger.info(
                "low_quality_person_only frame=%d ignored=%d state=observation_only "
                "control_target_present=False search_timer_reset=False "
                "exit_direction_update=False",
                int(self.frame_index),
                int(ignored_low_quality_persons),
            )

        self._queue_actions_for_persons(
            width,
            height,
            persons,
            low_quality_visible=False,
            lateral_candidate=lateral_candidate,
        )
        if selected_candidates:
            self._publish_longitudinal_context(width, height, persons)
        else:
            # Rejected boxes must never feed the independent Depth controller.
            self._clear_longitudinal_context()

    def _queue_actions_for_persons(
        self,
        width: int,
        height: int,
        persons: List[Tuple],
        *,
        depth_use_latest: bool = False,
        control_source: str = "vision",
        target_steerable: bool = True,
        target_steering_limit_rpm: Optional[float] = None,
        record_target_motion: bool = True,
        expected_target_id: Optional[int] = None,
        context_published_ts: Optional[float] = None,
        expected_frame_index: Optional[int] = None,
        low_quality_visible: bool = False,
        lateral_candidate: Optional[LateralCandidateEvidence] = None,
    ) -> None:
        with self._control_update_lock:
            if expected_target_id is not None:
                # 30Hz线程等待控制锁期间，视觉线程可能已切换目标或进入搜索。
                # 必须在锁内重新确认，禁止旧人物框覆盖刚产生的新视觉动作。
                active_target_id = getattr(self._follow_controller, "active_target_id", None)
                context_age = (
                    0.0
                    if context_published_ts is None
                    else max(0.0, time.monotonic() - float(context_published_ts))
                )
                if (
                    context_age > ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC
                    or active_target_id is None
                    or int(active_target_id) != int(expected_target_id)
                    or self.search_state != "none"
                    or self._runtime_shutdown_requested
                    or not self.running
                ):
                    return
            if control_source == "depth30":
                context_frame = (
                    int(expected_frame_index)
                    if expected_frame_index is not None
                    else int(self.frame_index)
                )
                vision_age = time.monotonic() - float(self._last_vision_control_ts)
                if (
                    self._last_vision_control_frame_index >= context_frame
                    and vision_age < ASTRA_DEPTH_VISION_DEDUPE_SEC
                ):
                    if self._last_depth30_dedupe_log_frame != context_frame:
                        self._last_depth30_dedupe_log_frame = context_frame
                        logger.info(
                            "30Hz Depth纵向监督跳过同帧/旧视觉结果: context_frame=%d "
                            "latest_vision_frame=%d vision_age=%.0fms dedupe=%.0fms",
                            context_frame,
                            int(self._last_vision_control_frame_index),
                            max(0.0, vision_age) * 1000.0,
                            ASTRA_DEPTH_VISION_DEDUPE_SEC * 1000.0,
                        )
                    return
            self._queue_actions_for_persons_locked(
                width,
                height,
                persons,
                depth_use_latest=depth_use_latest,
                control_source=control_source,
                target_steerable=target_steerable,
                target_steering_limit_rpm=target_steering_limit_rpm,
                record_target_motion=record_target_motion,
                low_quality_visible=low_quality_visible,
                lateral_candidate=lateral_candidate,
            )

    def _queue_actions_for_persons_locked(
        self,
        width: int,
        height: int,
        persons: List[Tuple],
        *,
        depth_use_latest: bool,
        control_source: str,
        target_steerable: bool,
        target_steering_limit_rpm: Optional[float] = None,
        record_target_motion: bool = True,
        low_quality_visible: bool = False,
        lateral_candidate: Optional[LateralCandidateEvidence] = None,
    ) -> None:
        was_searching_before = self.search_state == "searching"
        previous_target_id = getattr(self._follow_controller, "active_target_id", None)
        previous_search_state = getattr(self._follow_controller, "search_state", "none")
        previous_pid_result = getattr(self._follow_controller, "last_steering_pid_result", None)
        actions = self._process_detections_modular(
            width,
            height,
            persons,
            depth_use_latest=depth_use_latest,
            target_steerable=target_steerable,
            target_steering_limit_rpm=target_steering_limit_rpm,
            record_target_motion=record_target_motion,
            control_source=control_source,
            low_quality_visible=low_quality_visible,
            lateral_candidate=lateral_candidate,
        )

        selected_after = self._follow_controller.last_selected_target
        loss_started_at = getattr(self._follow_controller, "_lost_started_at", None)
        loss_age_sec = (
            0.0
            if loss_started_at is None
            else max(0.0, time.monotonic() - float(loss_started_at))
        )
        if (
            previous_target_id is not None
            and selected_after is None
            and self._last_target_loss_trace_frame != int(self.frame_index)
        ):
            feedback = getattr(previous_pid_result, "measured_yaw_rate_dps", None)
            logger.info(
                "target_loss_trace: frame=%d source=%s previous_target_id=%s "
                "previous_search=%s/%s decision=%s lost_frames=%d lost_sec=%.3f "
                "last_pid=(error=%.2fdeg desired=%.2fdps measured=%s correction=%sRPM limit=%sRPM) "
                "motor=(rotate_raw=%d/%s steer_base=%d steer_correction=%d)",
                int(self.frame_index),
                control_source,
                int(previous_target_id),
                previous_search_state,
                self._follow_controller.search_direction,
                self._last_control_decision_reason,
                int(self._follow_controller.lost_confirm_frames),
                loss_age_sec,
                0.0 if previous_pid_result is None else float(previous_pid_result.filtered_error_deg),
                0.0 if previous_pid_result is None else float(previous_pid_result.desired_yaw_rate_dps),
                "none" if feedback is None else "%.2f" % float(feedback),
                "none" if previous_pid_result is None else int(previous_pid_result.correction_rpm),
                "none" if previous_pid_result is None else "%.2f" % float(previous_pid_result.correction_limit_rpm),
                int(self._current_rotate_raw_target),
                self._current_rotate_raw_source,
                int(self._current_steer_base_percent),
                int(self._current_steer_correction_rpm),
            )
            self._last_target_loss_trace_frame = int(self.frame_index)

        search_target_recovered = (
            was_searching_before
            and self.search_state == "none"
            and self._follow_controller.last_selected_target is not None
        )
        if search_target_recovered:
            self._clear_action_queue("search_to_follow")
            if not actions and not self._should_skip_redundant_direct_stop("search_to_follow"):
                self._prepare_direct_stop("search_to_follow")
                self._action_runtime.send_stop_with_brake_hold("search_to_follow")
                self._mark_direct_stop_sent("search_to_follow")
                logger.info("搜索目标恢复但暂无跟随动作，已停车等待下一帧")
            else:
                # The latest-action queue replaces the old rotation directly;
                # do not insert an extra STOP before the first follow command.
                logger.info("已清空旧搜索动作，直接切换到恢复后的跟随动作")

        if actions:
            logger.info(
                "生成动作: control_frame_id=%d decision_capture_frame_id=%d evidence_capture_frame_id=%d source_module=%s source=%s reason=%s actions=%s",
                self.frame_index,
                int(getattr(self, "_last_decision_capture_frame", -1)),
                int(getattr(self, "_last_command_capture_frame", -1)),
                str(getattr(self, "_last_command_source_module", "unknown")),
                control_source,
                self._last_control_decision_reason,
                self._action_names_for_log(actions),
            )
            if any(a in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT) for a in actions):
                logger.info(
                    "rotate action queued: control_frame_id=%d decision_capture_frame_id=%d evidence_capture_frame_id=%d source_module=%s actions=%s search=%s/%s queue_size_before=%d last_dispatched=%s pulse_enable=%s duration=%.3fs pause=%.3fs observe_min_frames=%d hold_stale=%.3fs raw=%d raw_source=%s default_raw=%d",
                    self.frame_index,
                    int(getattr(self, "_last_decision_capture_frame", -1)),
                    int(getattr(self, "_last_command_capture_frame", -1)),
                    str(getattr(self, "_last_command_source_module", "unknown")),
                    self._action_names_for_log(actions),
                    self.search_state,
                    self.search_direction,
                    self.action_queue.qsize(),
                    ACTION_NAMES.get(self._last_dispatched_action, str(self._last_dispatched_action)),
                    bool(
                        ROTATE_PULSE_BRAKE_ENABLE
                        and getattr(self, "_current_rotate_pulse_enabled", True)
                    ),
                    ROTATE_DURATION,
                    ROTATE_PULSE_PAUSE_SEC,
                    ROTATE_PULSE_OBSERVE_MIN_FRAMES,
                    ROTATE_HOLD_STALE_SEC,
                    int(self._current_rotate_raw_target),
                    self._current_rotate_raw_source,
                    MOTOR_ROTATE_RAW_TARGET,
                )
            if any(a in (ACTION_STEER_LEFT, ACTION_STEER_RIGHT) for a in actions):
                logger.info(
                    "steer action queued: control_frame_id=%d decision_capture_frame_id=%d evidence_capture_frame_id=%d source_module=%s actions=%s search=%s/%s queue_size_before=%d last_dispatched=%s base=%d correction_rpm=%d inner_ratio=%d outer_ratio=%d raw=%d",
                    self.frame_index,
                    int(getattr(self, "_last_decision_capture_frame", -1)),
                    int(getattr(self, "_last_command_capture_frame", -1)),
                    str(getattr(self, "_last_command_source_module", "unknown")),
                    self._action_names_for_log(actions),
                    self.search_state,
                    self.search_direction,
                    self.action_queue.qsize(),
                    ACTION_NAMES.get(self._last_dispatched_action, str(self._last_dispatched_action)),
                    self._current_steer_base_percent,
                    self._current_steer_correction_rpm,
                    self._current_steer_inner_ratio_percent,
                    self._current_steer_outer_ratio_percent,
                    MOTOR_STEER_RAW_TARGET,
                )
        else:
            logger.debug("未生成动作")

        lateral_action_owned = bool(
            control_source == "vision"
            and LATERAL_INTENT_CONTROL_ENABLE
            and int(getattr(self, "_lateral_intent_owned_frame", -1))
            == int(self.frame_index)
        )
        if actions and lateral_action_owned:
            logger.info(
                "vision_action_delegated control_frame_id=%d decision_capture_frame_id=%d evidence_capture_frame_id=%d source_module=%s reason=%s actions=%s "
                "owner=lateral_intent_loop direct_queue=False",
                int(self.frame_index),
                int(getattr(self, "_last_decision_capture_frame", -1)),
                int(getattr(self, "_last_command_capture_frame", -1)),
                "lateral_intent_store",
                self._last_control_decision_reason or "decision",
                self._action_names_for_log(actions),
            )
        elif actions:
            reason = self._last_control_decision_reason or "decision"
            if not self._should_skip_redundant_action_queue(actions, reason):
                self._replace_action_queue(actions, reason)
        else:
            if self._explicit_stop_requested:
                reason = self._last_explicit_stop_reason or "explicit_stop"
                if self._runtime_shutdown_requested or not self._should_skip_redundant_direct_stop(reason):
                    self._clear_action_queue(f"explicit_stop:{reason}")
                    self._prepare_direct_stop(reason)
                    self._action_runtime.send_stop_with_brake_hold(reason)
                    self._mark_direct_stop_sent(reason)

        if self._runtime_shutdown_requested:
            logger.info(
                "目标丢失退出，安全停车已下发，主程序准备退出: frame=%d reason=%s timeout=%.2fs",
                self.frame_index,
                self._last_control_decision_reason or "target_lost_exit",
                FOLLOW_SEARCH_TIMEOUT_SEC,
            )
            self.running = False
            self.action_stop_event.set()

    def _ensure_rknn_camera_started(self) -> None:
        if not RKNN_CAMERA_ENABLE:
            return
        if self._rknn_camera is not None:
            return
        now = time.monotonic()
        if now < self._rknn_camera_next_retry_ts:
            return
        self._rknn_camera_next_retry_ts = now + RKNN_CAMERA_RETRY_INTERVAL_SEC
        if RKNN_CAMERA_CAPTURE_MODE in {"gst", "gstreamer", "gstreamer_mjpeg", "gstreamer_mjpeg_tee"}:
            if GstMjpegTeeCapture is None or GstMjpegTeeConfig is None:
                raise RuntimeError(f"GStreamer camera capture import failed: {_RKNN_GST_CAPTURE_IMPORT_ERROR}")
            if RKNN_CAMERA_FOURCC.strip().upper() != "MJPG":
                raise RuntimeError("GStreamer RKNN camera capture currently requires RKNN_CAMERA_FOURCC=MJPG")
            cap = GstMjpegTeeCapture(
                GstMjpegTeeConfig(
                    device=RKNN_CAMERA_DEVICE,
                    width=RKNN_CAMERA_WIDTH,
                    height=RKNN_CAMERA_HEIGHT,
                    fps=RKNN_CAMERA_FPS,
                    raw_output=RKNN_CAMERA_RAW_OUTPUT,
                )
            )
            cap.open()
            self._rknn_camera = cap
            logger.info(
                "RKNN camera started: %s (gstreamer_mjpeg raw_output=%s)",
                RKNN_CAMERA_DEVICE,
                RKNN_CAMERA_RAW_OUTPUT or "disabled",
            )
            logger.info("RKNN camera gst pipeline: %s", cap.pipeline_description)
            self._start_capture_thread()
            return
        if RKNN_CAMERA_CAPTURE_MODE not in {"opencv", "opencv_v4l2", "v4l2"}:
            raise RuntimeError(f"Unknown RKNN_CAMERA_CAPTURE_MODE: {RKNN_CAMERA_CAPTURE_MODE!r}")
        if cv2 is None:
            raise RuntimeError(f"OpenCV(cv2) import failed: {_CV2_IMPORT_ERROR}")
        candidates: List[Tuple[Any, int, str]] = [
            (RKNN_CAMERA_DEVICE, cv2.CAP_V4L2, "path+v4l2"),
            (RKNN_CAMERA_DEVICE, cv2.CAP_ANY, "path+any"),
        ]
        if RKNN_CAMERA_DEVICE.startswith("/dev/video"):
            suffix = RKNN_CAMERA_DEVICE.replace("/dev/video", "", 1)
            if suffix.isdigit():
                idx = int(suffix)
                candidates.extend(
                    [
                        (idx, cv2.CAP_V4L2, "index+v4l2"),
                        (idx, cv2.CAP_ANY, "index+any"),
                    ]
                )

        cap = None
        chosen = None
        for source, backend, tag in candidates:
            trial = cv2.VideoCapture(source, backend)
            if trial.isOpened():
                cap = trial
                chosen = tag
                break
            trial.release()
        if cap is None:
            if now - self._last_rknn_no_frame_log_ts >= 5.0:
                self._last_rknn_no_frame_log_ts = now
                logger.warning("RKNN camera open failed: device=%s tried=%s", RKNN_CAMERA_DEVICE, [c[2] for c in candidates])
            return

        if len(RKNN_CAMERA_FOURCC) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*RKNN_CAMERA_FOURCC))
        if RKNN_CAMERA_WIDTH > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, RKNN_CAMERA_WIDTH)
        if RKNN_CAMERA_HEIGHT > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RKNN_CAMERA_HEIGHT)
        if RKNN_CAMERA_FPS > 0:
            cap.set(cv2.CAP_PROP_FPS, RKNN_CAMERA_FPS)
        # Keep only the newest UVC frame so slow RKNN inference does not build
        # seconds of camera latency behind the person being followed.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._rknn_camera = cap
        logger.info("RKNN camera started: %s (%s)", RKNN_CAMERA_DEVICE, chosen)
        self._start_capture_thread()

    def _start_capture_thread(self) -> None:
        if self._direction_pool is None:
            return
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        self._capture_stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture-30fps",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info(
            "Camera capture queue started: fps=%.1f queue=%d",
            RKNN_CAMERA_FPS,
            RKNN_CAPTURE_QUEUE_SIZE,
        )

    def _capture_loop(self) -> None:
        while self.running and not self._capture_stop_event.is_set():
            camera = self._rknn_camera
            if camera is None:
                time.sleep(0.01)
                continue
            try:
                ok, frame = camera.read()
            except Exception as exc:
                logger.warning("RKNN capture thread read failed: %s", exc)
                time.sleep(0.05)
                continue
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            timestamp = time.monotonic()
            with self._capture_state_lock:
                self._capture_frame_id += 1
                capture_id = int(self._capture_frame_id)
            if self._camera_video_recorder is not None:
                self._camera_video_recorder.submit(
                    frame,
                    capture_frame_id=capture_id,
                    monotonic_sec=timestamp,
                    unix_sec=time.time(),
                )
            if self._direction_pool is not None:
                self._direction_pool.submit(capture_id, timestamp, frame, "BGR")
            packet = (capture_id, timestamp, frame)
            try:
                self._capture_queue.put_nowait(packet)
            except queue.Full:
                try:
                    self._capture_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._capture_queue.put_nowait(packet)
                except queue.Full:
                    pass

    def _drain_direction_results(self) -> None:
        pool = self._direction_pool
        note = getattr(self._follow_controller, "note_direction_classifier_evidence", None)
        if pool is None or not callable(note):
            return
        self._direction_result_backlog.extend(pool.drain_results())
        current_capture_id = int(getattr(self, "_camera_capture_frame_id", 0))
        ready = [
            evidence
            for evidence in self._direction_result_backlog
            if int(evidence.capture_frame_id) <= current_capture_id
        ]
        self._direction_result_backlog = [
            evidence
            for evidence in self._direction_result_backlog
            if int(evidence.capture_frame_id) > current_capture_id
        ]
        ready.sort(key=lambda item: item.capture_frame_id)
        for evidence in ready:
            note(
                evidence.capture_frame_id,
                evidence.timestamp,
                state=evidence.state,
                bbox=evidence.bbox,
                frame_width=int(evidence.frame_width or RKNN_CAMERA_WIDTH),
                confidence=evidence.score,
                reason=evidence.reason,
            )
            logger.debug(
                "direction evidence merged capture=%d state=%s side=%s age_ms=%.1f reason=%s",
                evidence.capture_frame_id,
                evidence.state,
                evidence.side,
                evidence.result_age_ms,
                evidence.reason,
            )
        if ready:
            self._direction_evidence_merged_total += len(ready)
            now = time.monotonic()
            if now - self._direction_evidence_last_log_ts >= 1.0:
                self._direction_evidence_last_log_ts = now
                latest = ready[-1]
                logger.info(
                    "capture_direction_progress merged=%d total=%d current_capture=%d "
                    "worker_pending=%d result_backlog=%d latest_state=%s latest_side=%s age=%.1fms",
                    len(ready),
                    self._direction_evidence_merged_total,
                    current_capture_id,
                    int(getattr(pool, "pending_count", 0)),
                    len(self._direction_result_backlog),
                    latest.state,
                    latest.side,
                    latest.result_age_ms,
                )

    def _process_frame_rknn_camera(self):
        if not RKNN_CAMERA_ENABLE:
            now = time.monotonic()
            if now - self._last_rknn_no_frame_log_ts >= 5.0:
                self._last_rknn_no_frame_log_ts = now
                logger.warning("RKNN camera disabled; enable RKNN_CAMERA_ENABLE=1 or call process_external_frame(frame)")
            return
        self._ensure_rknn_camera_started()
        if self._rknn_camera is None:
            return
        if self._direction_pool is not None:
            self._drain_direction_results()
            read_started = time.perf_counter()
            try:
                capture_id, capture_timestamp, frame = self._capture_queue.get(timeout=0.25)
            except queue.Empty:
                return
            while True:
                try:
                    capture_id, capture_timestamp, frame = self._capture_queue.get_nowait()
                except queue.Empty:
                    break
            self.process_external_frame(
                frame,
                "BGR",
                camera_read_ms=(time.perf_counter() - read_started) * 1000.0,
                camera_drained=0,
                capture_frame_id=int(capture_id),
                capture_timestamp=float(capture_timestamp),
            )
            self._drain_direction_results()
            return
        drained = 0
        read_latest = getattr(self._rknn_camera, "read_latest", None)
        if callable(read_latest):
            camera_read_started = time.perf_counter()
            ok, frame, drained = read_latest(max_drain=RKNN_CAMERA_LATEST_DRAIN_MAX)
            camera_read_ms = (time.perf_counter() - camera_read_started) * 1000.0
            if drained > 0:
                now = time.monotonic()
                if now - self._last_rknn_latest_drain_log_ts >= 2.0:
                    self._last_rknn_latest_drain_log_ts = now
                    logger.info(
                        "RKNN 摄像头使用队列中的最新画面: 已丢弃旧帧=%d 单次最大丢弃=%d",
                        drained,
                        RKNN_CAMERA_LATEST_DRAIN_MAX,
                    )
        else:
            camera_read_started = time.perf_counter()
            ok, frame = self._rknn_camera.read()
            camera_read_ms = (time.perf_counter() - camera_read_started) * 1000.0
        if not ok or frame is None:
            logger.warning("RKNN 摄像头读取失败: %s", RKNN_CAMERA_DEVICE)
            try:
                self._rknn_camera.release()
            except Exception:
                pass
            self._rknn_camera = None
            return
        if self._camera_video_recorder is not None:
            next_capture_id = int(getattr(self, "_camera_capture_frame_id", 0)) + max(0, int(drained)) + 1
            self._camera_video_recorder.submit(
                frame,
                capture_frame_id=next_capture_id,
                monotonic_sec=time.monotonic(),
                unix_sec=time.time(),
            )
        self.process_external_frame(
            frame,
            "BGR",
            camera_read_ms=camera_read_ms,
            camera_drained=drained,
        )

    def process_external_frame(
        self,
        frame,
        frame_format: str = "BGR",
        *,
        camera_read_ms: float = 0.0,
        camera_drained: int = 0,
        capture_frame_id: Optional[int] = None,
        capture_timestamp: Optional[float] = None,
    ):
        """Process one externally supplied frame through RKNN vision and control."""
        pipeline_started = time.perf_counter()
        frame_received_ts = (
            time.monotonic()
            if capture_timestamp is None
            else float(capture_timestamp)
        )
        previous_capture_id = int(getattr(self, "_camera_capture_frame_id", 0))
        drained_count = max(0, int(camera_drained))
        current_capture_id = (
            int(capture_frame_id)
            if capture_frame_id is not None and int(capture_frame_id) > 0
            else previous_capture_id + drained_count + 1
        )
        if capture_frame_id is None:
            for capture_id in range(previous_capture_id + 1, current_capture_id + 1):
                self._follow_controller.note_unknown_capture(
                    capture_id,
                    frame_received_ts,
                    "camera_drain_skipped" if capture_id < current_capture_id else "vision_pending",
                )
        self._camera_capture_frame_id = current_capture_id
        self._active_capture_frame_id = current_capture_id
        self._active_capture_timestamp = frame_received_ts
        frame_gap_ms = (
            None
            if self._last_vision_frame_received_ts <= 0.0
            else max(0.0, frame_received_ts - self._last_vision_frame_received_ts) * 1000.0
        )
        self._last_vision_frame_received_ts = frame_received_ts
        if self._vision_engine != "rknn":
            raise RuntimeError(f"process_external_frame is only available for rknn engine, got {self._vision_engine!r}")
        if self._rknn_pipeline is None:
            raise RuntimeError("RKNN vision pipeline is not initialized")
        self.frame_index += 1
        self._maybe_log_imu_sample()
        if self._handle_hazard_safety_state(self._bunker_runtime.check_split_monitor()):
            return []
        if self._handle_hazard_safety_state(self._bunker_runtime.check_split_frame(frame, frame_format)):
            return []

        controller_status_before = self._follow_controller.search_status(frame_received_ts)
        diagnostic_search_active = controller_status_before.state in (
            "searching",
            "direction_unresolved",
        )
        feedback_before = self._action_runtime.get_steering_feedback()
        measured_yaw_before = (
            0.0 if feedback_before is None else abs(float(feedback_before.yaw_rate_right_dps))
        )
        identity_context = getattr(
            self._rknn_pipeline,
            "set_identity_reacquire_context",
            None,
        )
        if callable(identity_context):
            identity_context(
                active_uid=controller_status_before.active_target_id,
                searching=diagnostic_search_active,
                direction=controller_status_before.direction,
            )
        diagnostic_mode = getattr(self._rknn_pipeline, "set_search_diagnostic_mode", None)
        if callable(diagnostic_mode):
            diagnostic_mode(diagnostic_search_active)
        detector_probe_only = bool(
            controller_status_before.state == "direction_unresolved"
            and not self._search_evidence_observation_active
            and controller_status_before.stage
            != "direction_unresolved"
        )
        probe_processor = getattr(
            self._rknn_pipeline,
            "process_search_probe_frame",
            None,
        )
        if detector_probe_only and callable(probe_processor):
            records = probe_processor(frame, frame_format)
        else:
            records = self._rknn_pipeline.process_frame(frame, frame_format)
        video_detections = []
        if self._camera_video_recorder is not None:
            video_detections = list(
                getattr(self._rknn_pipeline, "last_detections", []) or []
            )
            for probe_detection in (
                getattr(self._rknn_pipeline, "last_search_diagnostic_detections", [])
                or []
            ):
                if not any(
                    int(getattr(existing, "class_id", -1))
                    == int(getattr(probe_detection, "class_id", -1))
                    and tuple(getattr(existing, "bbox", ()))
                    == tuple(getattr(probe_detection, "bbox", ()))
                    and float(getattr(existing, "score", 0.0))
                    == float(getattr(probe_detection, "score", 0.0))
                    for existing in video_detections
                ):
                    video_detections.append(probe_detection)
        vision_finished = time.perf_counter()
        width = int(getattr(self._rknn_pipeline, "last_frame_width", VISION_FRAME_WIDTH))
        height = int(getattr(self._rknn_pipeline, "last_frame_height", VISION_FRAME_HEIGHT))
        control_started = time.perf_counter()
        vision_result_age_sec = max(0.0, vision_finished - pipeline_started)
        stale_result_discarded = vision_result_age_sec > VISION_CONTROL_MAX_RESULT_AGE_SEC
        evidence_getter = getattr(
            self._rknn_pipeline,
            "get_search_candidate_evidence",
            None,
        )
        evidence = evidence_getter() if callable(evidence_getter) else None
        search_active_before = diagnostic_search_active
        formal_evidence = ()
        probe_evidence = ()
        if evidence is not None and not stale_result_discarded:
            formal_evidence = tuple(
                CandidateObservation(
                    bbox=tuple(float(value) for value in item.bbox),
                    score=float(item.score),
                )
                for item in evidence.formal_persons
            )
            probe_evidence = tuple(
                CandidateObservation(
                    bbox=tuple(float(value) for value in item.bbox),
                    score=float(item.score),
                )
                for item in evidence.probe_persons
            )
        active_candidate_uid = getattr(self._follow_controller, "active_target_id", None)
        active_target_record, _active_target_assignment = self._active_target_track_record(
            records,
            active_candidate_uid,
        )
        preferred_candidate_bbox = (
            None
            if active_target_record is None
            else self._track_record_bbox(active_target_record)
        )
        current_candidate_decision = (
            SearchCandidateGateDecision(reason="stale_result_ignored")
            if stale_result_discarded
            else self._search_candidate_gate.select_current_candidate(
                width=width,
                height=height,
                formal_candidates=formal_evidence,
                probe_candidates=probe_evidence,
                preferred_bbox=preferred_candidate_bbox,
            )
        )

        def candidate_matches_active_target(candidate_bbox) -> bool:
            return bool(
                candidate_bbox is not None
                and preferred_candidate_bbox is not None
                and self._bbox_iou_xyxy(
                    preferred_candidate_bbox,
                    tuple(float(value) for value in candidate_bbox),
                ) >= 0.15
            )

        current_candidate_matches_target = candidate_matches_active_target(
            current_candidate_decision.bbox
        )
        lateral_candidate = (
            None
            if current_candidate_decision.bbox is None
            else LateralCandidateEvidence(
                capture_frame_id=int(current_capture_id),
                bbox=current_candidate_decision.bbox,
                score=float(current_candidate_decision.score),
                source=str(current_candidate_decision.source),
                active_target_match=current_candidate_matches_target,
            )
        )
        # A stale full-vision result must not advance candidate hold/probe
        # counters. It is evidence for neither the current search nor a new
        # candidate; preserving the gate state avoids delayed-frame pollution.
        gate_decision = (
            SearchCandidateGateDecision(reason="stale_result_ignored")
            if stale_result_discarded
            else self._search_candidate_gate.update(
                timestamp=frame_received_ts,
                search_active=search_active_before,
                width=width,
                height=height,
                formal_candidates=formal_evidence,
                probe_candidates=probe_evidence,
                preferred_bbox=preferred_candidate_bbox,
            )
        )
        note_candidate = getattr(
            self._follow_controller,
            "note_search_candidate_evidence",
            None,
        )
        if (
            not stale_result_discarded
            and gate_decision.bbox is not None
            and callable(note_candidate)
        ):
            candidate_tracked = candidate_matches_active_target(gate_decision.bbox)
            note_candidate(
                gate_decision.bbox,
                frame_width=width,
                confirmed=bool(gate_decision.completed),
                source=str(gate_decision.source),
                candidate_score=float(gate_decision.score),
                candidate_tracked=candidate_tracked,
                capture_frame_id=int(current_capture_id),
                now=time.monotonic(),
            )
        note_candidate_missing = getattr(
            self._follow_controller,
            "note_search_candidate_missing",
            None,
        )
        candidate_missing_hold = bool(
            not stale_result_discarded
            and gate_decision.bbox is None
            and callable(note_candidate_missing)
            and note_candidate_missing()
        )
        if gate_decision.pause_rotation:
            self._apply_search_candidate_gate_decision(gate_decision)
        observation_now = time.monotonic()
        bounded_observation_pending = bool(
            self._search_evidence_observation_active
            and observation_now < self._search_evidence_observation_deadline
        )
        effective_evidence_pause = bool(
            (gate_decision.pause_rotation and not gate_decision.completed)
            or bounded_observation_pending
            or candidate_missing_hold
        )
        if self._search_evidence_observation_active and not bounded_observation_pending:
            logger.info(
                "search_evidence_static_observe_complete frame=%d source=%s "
                "reason=bounded_timeout max_hold=%.3fs",
                int(self.frame_index),
                self._search_evidence_observation_source,
                SEARCH_EVIDENCE_MAX_HOLD_SEC,
            )
            self._search_evidence_observation_active = False
            self._search_evidence_observation_source = "none"
            self._search_evidence_observation_deadline = 0.0
        self._search_evidence_pause_current_frame = effective_evidence_pause
        self._follow_controller.set_search_observation_hold(effective_evidence_pause)
        if stale_result_discarded:
            # Tracker/ReID内部状态已经更新，但这个框对应的是旧画面，禁止再据此
            # 生成新的运动命令；同时撤销上一帧仍在发布的横向意图。
            self._handle_stale_vision_result(
                width=width,
                result_age_sec=vision_result_age_sec,
            )
            logger.warning(
                "视觉结果过期，跳过运动决策: control_frame_id=%d capture_frame_id=%d age=%.1fms limit=%.1fms records=%d",
                int(self.frame_index),
                int(getattr(self, "_active_capture_frame_id", 0)),
                vision_result_age_sec * 1000.0,
                VISION_CONTROL_MAX_RESULT_AGE_SEC * 1000.0,
                len(records),
            )
        else:
            try:
                self._consume_track_records(
                    records,
                    width,
                    height,
                    "rknn",
                    lateral_candidate=lateral_candidate,
                )
            finally:
                self._search_evidence_pause_current_frame = False
        if stale_result_discarded:
            self._search_evidence_pause_current_frame = False
        status_after_control = self._follow_controller.search_status(time.monotonic())
        direction_evidence = ()
        if status_after_control.state not in (
            "searching",
            "direction_unresolved",
        ):
            self._search_candidate_gate.reset()
            self._search_evidence_observation_active = False
            self._search_evidence_observation_source = "none"
            self._search_evidence_observation_deadline = 0.0
            self._reset_confirmed_search_reacquire()
            self._follow_controller.set_search_observation_hold(False)
        control_finished = time.perf_counter()
        timing = dict(getattr(self._rknn_pipeline, "last_timing_ms", {}) or {})
        logger.info(
            "pipeline_timing: control_frame_id=%d capture_frame_id=%d camera_read_ms=%.2f frame_wrap_ms=%.2f "
            "yolo_preprocess_ms=%.2f yolo_inference_ms=%.2f yolo_decode_ms=%.2f "
            "yolo_nms_ms=%.2f yolo_total_ms=%.2f reid_preprocess_ms=%.2f reid_inference_ms=%.2f "
            "reid_postprocess_ms=%.2f reid_total_ms=%.2f tracker_ms=%.2f "
            "predicted_reid_verify_ms=%.2f vision_total_ms=%.2f control_ms=%.2f "
            "end_to_end_ms=%.2f result_age_ms=%.2f stale_result_discarded=%s",
            int(self.frame_index),
            int(getattr(self, "_active_capture_frame_id", 0)),
            float(camera_read_ms or 0.0),
            float(timing.get("frame_wrap", 0.0)),
            float(timing.get("yolo_preprocess", 0.0)),
            float(timing.get("yolo_inference", 0.0)),
            float(timing.get("yolo_decode", 0.0)),
            float(timing.get("yolo_nms", 0.0)),
            float(timing.get("yolo_total", 0.0)),
            float(timing.get("reid_preprocess", 0.0)),
            float(timing.get("reid_inference", 0.0)),
            float(timing.get("reid_postprocess", 0.0)),
            float(timing.get("reid_total", 0.0)),
            float(timing.get("tracker", 0.0)),
            float(timing.get("predicted_reid_verify", 0.0)),
            (vision_finished - pipeline_started) * 1000.0,
            (control_finished - control_started) * 1000.0,
            (control_finished - pipeline_started) * 1000.0,
            vision_result_age_sec * 1000.0,
            stale_result_discarded,
        )
        diagnostic_now = time.monotonic()
        controller_status_after = self._follow_controller.search_status(diagnostic_now)
        if self._camera_video_recorder is not None:
            video_active_record, _video_active_assignment = self._active_target_track_record(
                records,
                controller_status_after.active_target_id,
            )
            video_tracks = []
            for record in records or ():
                bbox = self._track_record_bbox(record)
                if bbox is None:
                    continue
                assignment = self._identity_assignment_debug_for_track(
                    int(getattr(record, "track_id", -1))
                )
                distance = assignment.get("distance")
                try:
                    distance = None if distance is None else float(distance)
                except (TypeError, ValueError):
                    distance = None
                video_tracks.append(
                    VideoTrackOverlay(
                        bbox=bbox,
                        track_id=int(getattr(record, "track_id", -1)),
                        reid_uid=int(getattr(record, "reid_uid", 0)),
                        mapped_uid=int(assignment.get("mapped_uid") or 0),
                        best_uid=int(assignment.get("best_uid") or 0),
                        score=float(getattr(record, "score", 0.0)),
                        distance=distance,
                        assignment_reason=str(assignment.get("reason") or "none"),
                        quality_reason=str(assignment.get("bbox_quality_reason") or ""),
                        fresh=int(getattr(record, "time_since_update", 0)) == 0,
                        active_target=record is video_active_record,
                    )
                )
            feedback_for_video = self._action_runtime.get_steering_feedback()
            with self.command_lock:
                video_command = self.current_command
            video_action_name = ACTION_NAMES.get(video_command, "none")
            video_rpm = 0
            if video_action_name == "rotate_left":
                video_rpm = -abs(int(self._current_rotate_raw_target))
            elif video_action_name == "rotate_right":
                video_rpm = abs(int(self._current_rotate_raw_target))
            self._camera_video_recorder.update_overlay(
                current_capture_id,
                video_detections,
                tracks=video_tracks,
                control=VideoControlOverlay(
                    control_frame_id=int(self.frame_index),
                    active_target_id=controller_status_after.active_target_id,
                    selected_target_id=controller_status_after.selected_target_id,
                    candidate_bbox=current_candidate_decision.bbox,
                    candidate_score=float(current_candidate_decision.score),
                    candidate_source=str(current_candidate_decision.source),
                    candidate_matches_target=bool(current_candidate_matches_target),
                    action_name=video_action_name,
                    requested_rpm=video_rpm,
                    yaw_rate_dps=(
                        None
                        if feedback_for_video is None
                        else float(feedback_for_video.yaw_rate_right_dps)
                    ),
                    result_age_ms=vision_result_age_sec * 1000.0,
                    search_state=str(controller_status_after.state),
                    search_direction=controller_status_after.direction,
                    decision_reason=str(self._last_control_decision_reason or "none"),
                ),
            )
        search_diagnostic_relevant = bool(
            controller_status_before.state
            in ("searching", "direction_unresolved")
            or controller_status_after.state
            in ("searching", "direction_unresolved")
        )
        measure_quality = bool(
            search_diagnostic_relevant
            or int(self.frame_index) == 1
            or int(self.frame_index) % SEARCH_DIAGNOSTIC_BASELINE_EVERY_FRAMES == 0
        )
        frame_quality = (
            self._search_diagnostics.measure_frame_quality(
                frame,
                frame_format,
                update_low_yaw_baseline=(
                    not search_diagnostic_relevant and measured_yaw_before <= 3.0
                ),
            )
            if measure_quality
            else None
        )
        if search_diagnostic_relevant:
            feedback = self._action_runtime.get_steering_feedback()
            with self.command_lock:
                current_command = self.current_command
            formal_detections = tuple(
                DetectionObservation.from_object(item)
                for item in (getattr(self._rknn_pipeline, "last_detections", []) or [])
            )
            probe_detections = tuple(
                DetectionObservation.from_object(item)
                for item in (
                    getattr(
                        self._rknn_pipeline,
                        "last_search_diagnostic_detections",
                        [],
                    )
                    or []
                )
            )
            fresh_tracks = sum(
                1 for record in records if int(getattr(record, "time_since_update", 0)) == 0
            )
            predicted_tracks = len(records) - fresh_tracks
            assigned_uid_tracks = sum(
                1 for record in records if int(getattr(record, "reid_uid", 0)) > 0
            )
            self._search_diagnostics.observe(
                SearchDiagnosticSample(
                    frame_index=int(self.frame_index),
                    timestamp=diagnostic_now,
                    width=width,
                    height=height,
                    recognition=RecognitionObservation(
                        formal_detections=formal_detections,
                        probe_detections=probe_detections,
                        tracks_total=len(records),
                        fresh_tracks=fresh_tracks,
                        predicted_tracks=predicted_tracks,
                        assigned_uid_tracks=assigned_uid_tracks,
                        yolo_total_ms=float(timing.get("yolo_total", 0.0)),
                        yolo_inference_ms=float(timing.get("yolo_inference", 0.0)),
                        yolo_nms_ms=float(timing.get("yolo_nms", 0.0)),
                        reid_total_ms=float(timing.get("reid_total", 0.0)),
                        tracker_ms=float(timing.get("tracker", 0.0)),
                        stale_result_discarded=bool(stale_result_discarded),
                    ),
                    control=SearchControlObservation(
                        state_before=controller_status_before.state,
                        direction_before=controller_status_before.direction,
                        state_after=controller_status_after.state,
                        direction_after=controller_status_after.direction,
                        active_target_id=controller_status_after.active_target_id,
                        selected_target_id=controller_status_after.selected_target_id,
                        decision_reason=self._last_control_decision_reason,
                        progress_deg=max(
                            controller_status_before.progress_deg,
                            controller_status_after.progress_deg,
                        ),
                        target_deg=controller_status_after.target_deg,
                        elapsed_sec=(
                            controller_status_after.elapsed_sec
                            if controller_status_after.elapsed_sec is not None
                            else controller_status_before.elapsed_sec
                        ),
                        stage=(
                            controller_status_after.stage
                            if controller_status_after.stage != "inactive"
                            else controller_status_before.stage
                        ),
                        heading_from_loss_deg=(
                            controller_status_after.heading_from_loss_deg
                            if controller_status_after.stage != "inactive"
                            else controller_status_before.heading_from_loss_deg
                        ),
                        coverage_deg=max(
                            controller_status_before.coverage_deg,
                            controller_status_after.coverage_deg,
                        ),
                        travel_deg=max(
                            controller_status_before.travel_deg,
                            controller_status_after.travel_deg,
                        ),
                        hint_confidence=(
                            controller_status_after.hint_confidence
                            if controller_status_after.stage != "inactive"
                            else controller_status_before.hint_confidence
                        ),
                        hint_source=(
                            controller_status_after.hint_source
                            if controller_status_after.stage != "inactive"
                            else controller_status_before.hint_source
                        ),
                    ),
                    motion=MotionObservation(
                        command_name=ACTION_NAMES.get(current_command, str(current_command)),
                        requested_rotate_raw=int(self._current_rotate_raw_target),
                        requested_rotate_source=self._current_rotate_raw_source,
                        left_speed_rpm=(
                            None if feedback is None else int(feedback.left_speed_rpm)
                        ),
                        right_speed_rpm=(
                            None if feedback is None else int(feedback.right_speed_rpm)
                        ),
                        yaw_rate_dps=(
                            None if feedback is None else float(feedback.yaw_rate_right_dps)
                        ),
                        raw_yaw_rate_dps=(
                            None
                            if feedback is None
                            else getattr(feedback, "raw_yaw_rate_right_dps", None)
                        ),
                        integrated_yaw_deg=(
                            None
                            if feedback is None
                            else float(feedback.integrated_yaw_right_deg)
                        ),
                        feedback_age_ms=(
                            None
                            if feedback is None
                            else max(0.0, diagnostic_now - float(feedback.timestamp)) * 1000.0
                        ),
                        feedback_trustworthy=(
                            False if feedback is None else bool(feedback.trustworthy)
                        ),
                    ),
                    transport=TransportObservation(
                        frame_gap_ms=frame_gap_ms,
                        camera_read_ms=float(camera_read_ms or 0.0),
                        camera_drained=int(camera_drained),
                        result_age_ms=vision_result_age_sec * 1000.0,
                    ),
                    quality=frame_quality,
                    direction_evidence=direction_evidence,
                    image_frame=frame,
                    frame_format=frame_format,
                )
            )
        return records

    def process_frame(self):
        return self._process_frame_rknn_camera()

    def _run_shutdown_step(self, label: str, callback, timeout_sec: float = SHUTDOWN_STEP_TIMEOUT_SEC) -> bool:
        """Run one potentially blocking close operation with a hard timeout."""
        done = threading.Event()
        errors = []
        started = time.monotonic()

        def _close_worker() -> None:
            try:
                callback()
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        worker = threading.Thread(
            target=_close_worker,
            name=f"shutdown-{label}",
            daemon=True,
        )
        worker.start()
        if not done.wait(max(0.5, float(timeout_sec))):
            logger.error(
                "退出清理步骤超时: %s timeout=%.1fs，继续执行后续安全收尾",
                label,
                float(timeout_sec),
            )
            return False
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if errors:
            logger.warning("退出清理步骤失败: %s elapsed=%.1fms error=%s", label, elapsed_ms, errors[0])
            return False
        logger.info("退出清理步骤完成: %s elapsed=%.1fms", label, elapsed_ms)
        return True

    def _release_rknn_camera(self) -> None:
        self._capture_stop_event.set()
        capture_thread = self._capture_thread
        self._capture_thread = None
        if capture_thread is not None:
            capture_thread.join(timeout=SHUTDOWN_STEP_TIMEOUT_SEC)
        camera = self._rknn_camera
        # 先摘掉共享引用，超时后遗留的 daemon 线程不会再被主流程重复释放。
        self._rknn_camera = None
        if camera is None:
            return
        camera.release()
        logger.info("RKNN camera released")

    def _close_camera_video_recorder(self) -> None:
        recorder = self._camera_video_recorder
        self._camera_video_recorder = None
        if recorder is None:
            return
        recorder.close(timeout_sec=SHUTDOWN_STEP_TIMEOUT_SEC)

    def _close_rknn_pipeline(self) -> None:
        direction_pool = self._direction_pool
        self._direction_pool = None
        if direction_pool is not None:
            direction_pool.close(timeout_sec=SHUTDOWN_STEP_TIMEOUT_SEC)
        pipeline = self._rknn_pipeline
        self._rknn_pipeline = None
        if pipeline is None:
            return
        pipeline.close()
        logger.info("RKNN vision pipeline closed")

    def run(self):
        """Run the camera/frame processing loop."""
        logger.info("开始运行人员跟踪（避障版本，vision_engine=%s）...", self._vision_engine)
        try:
            # 先独占串口完成双轮清零和 5A 驻车，再启动动作与编码器反馈线程。
            # 反馈线程也使用 motor_io_lock，因此不会再与初始化寄存器事务交叉。
            with self.motor_io_lock:
                self._motor_backend.ensure_driver()
            self._start_action_thread()
            self._start_lateral_intent_thread()
            self._start_longitudinal_thread()
            while self.running:
                self.process_frame()
                if PROCESS_FRAME_INTERVAL > 0.0:
                    time.sleep(PROCESS_FRAME_INTERVAL)

        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭...")
        finally:
            self.running = False
            search_diagnostics = getattr(self, "_search_diagnostics", None)
            if search_diagnostics is not None and search_diagnostics.active:
                final_search_status = self._follow_controller.search_status()
                search_diagnostics.finish(
                    time.monotonic(),
                    outcome="runtime_shutdown_during_search",
                    progress_deg=final_search_status.progress_deg,
                )
            self._stop_lateral_intent_thread()
            self._stop_longitudinal_thread()
            self.action_stop_event.set()
            # 先发 STOP，确保 Ctrl+C 后车一定停（避免中断发生在 detect 后、发停前的空窗期）
            if self._motor_backend.driver is not None:
                try:
                    self._action_runtime.send_stop_with_brake_hold(self._last_explicit_stop_reason or "explicit_stop")
                except Exception as e:
                    logger.warning(f"finally 中发送 STOP 失败: {e}")
            else:
                logger.info("电机初始化未完成，退出时跳过 STOP，禁止隐式重新打开串口")
            if self.action_thread:
                self.action_thread.join(timeout=1.0)
            self._action_runtime.join_feedback(timeout=1.0)
            # 收尾前再发一次 STOP，确保车已停（应对中断发生在 detect 后、发停前的空窗期）
            if self._motor_backend.driver is not None:
                try:
                    self._action_runtime.send_robot_command(ACTION_STOP)
                except Exception as e:
                    logger.warning(f"finally 中再次发送 STOP 失败: {e}")

            # 电机必须在摄像头/RKNN之前关闭。这样即使视觉驱动释放卡住，
            # 5A驻车电流也已经独立清零，不会因为后续设备阻塞而一直保持。
            self._run_shutdown_step("LZ30EMA motor backend", self._motor_backend.close)
            self._run_shutdown_step("camera video recorder", self._close_camera_video_recorder)
            self._run_shutdown_step("RKNN camera", self._release_rknn_camera)
            self._run_shutdown_step("RKNN vision pipeline", self._close_rknn_pipeline)
            self._run_shutdown_step("bunker runtime", self._bunker_runtime.close)
            self._run_shutdown_step("sensor runtime", self._sensor_runtime.close)
            logger.info("人员跟踪（避障版本）已停止")


def main():
    # 模型路径支持命令行第一个参数，不传则用配置/默认值。
    model_path = sys.argv[1] if len(sys.argv) >= 2 else None
    if model_path is None:
        logger.info(f"未传模型路径，使用默认: {YOLO_MODEL_PATH}")
    tracker = PersonTracker(model_path=model_path)
    def _request_shutdown(signum, _frame):
        logger.info("收到系统信号 %s，准备安全停车并退出", signal.Signals(signum).name)
        tracker.running = False
        tracker.action_stop_event.set()

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)
    tracker.run()


if __name__ == "__main__":
    main()
