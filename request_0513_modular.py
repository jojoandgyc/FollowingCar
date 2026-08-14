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
from car_control_modular.control_types import DistanceState, HazardState, ObstacleState, PersonTarget, SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.action_runtime import ActionRuntimeConfig, ActionRuntimeSymbols, MotionActionRuntime
from car_control_modular.distance_runtime import DistanceRuntime, DistanceRuntimeConfig
from car_control_modular.hazard_runtime import BunkerHazardRuntime, BunkerHazardRuntimeConfig
from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig, normalize_mssd_stop_mode
from car_control_modular.sensor_modules import SensorRuntime, SensorRuntimeConfig

try:
    import cv2
    _CV2_IMPORT_ERROR = None
except Exception as e:
    cv2 = None
    _CV2_IMPORT_ERROR = e

# 先config读取变量到 环境变量
LOADED_CONFIG = preload_config_from_argv()

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
MOTOR_STEER_RAW_TARGET = max(0, int(os.environ.get("MOTOR_STEER_RAW_TARGET", "0")))
MOTOR_ROTATE_RAW_TARGET = max(0, int(os.environ.get("MOTOR_ROTATE_RAW_TARGET", "0")))
ROTATE_RAW_TARGET_VISIBLE = max(0, int(os.environ.get("ROTATE_RAW_TARGET_VISIBLE", str(MOTOR_ROTATE_RAW_TARGET))))
ROTATE_RAW_TARGET_LOST_WAIT = max(0, int(os.environ.get("ROTATE_RAW_TARGET_LOST_WAIT", str(MOTOR_ROTATE_RAW_TARGET))))
ROTATE_RAW_TARGET_SEARCH = max(0, int(os.environ.get("ROTATE_RAW_TARGET_SEARCH", str(MOTOR_ROTATE_RAW_TARGET))))
MOTOR_LEFT_SIGN = int(os.environ.get("MOTOR_LEFT_SIGN", "-1"))
MOTOR_RIGHT_SIGN = int(os.environ.get("MOTOR_RIGHT_SIGN", "1"))
MOTOR_FORWARD_TARGET_SIGN = -1 if int(os.environ.get("MOTOR_FORWARD_TARGET_SIGN", "1")) < 0 else 1
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
MODULE_MMWAVE_ENABLE = os.environ.get("MODULE_MMWAVE_ENABLE", "0").strip() != "0"
MODULE_ULTRASONIC_ENABLE = os.environ.get("MODULE_ULTRASONIC_ENABLE", "0").strip() != "0"
MODULE_IMU_ENABLE = os.environ.get("MODULE_IMU_ENABLE", "0").strip() != "0"
IMU_FAIL_SOFT = os.environ.get("IMU_FAIL_SOFT", "1").strip() != "0"
IMU_LOG_ENABLE = os.environ.get("IMU_LOG_ENABLE", "0").strip() != "0"
IMU_LOG_EVERY_SEC = max(0.05, float(os.environ.get("IMU_LOG_EVERY_SEC", "1.0")))
MMWAVE_ASYNC_CACHE_ENABLE = os.environ.get("MMWAVE_ASYNC_CACHE_ENABLE", "1").strip() != "0"
MMWAVE_CACHE_INTERVAL_SEC = max(0.01, float(os.environ.get("MMWAVE_CACHE_INTERVAL_SEC", "0.04")))
MMWAVE_CACHE_WINDOW_SEC = max(0.10, float(os.environ.get("MMWAVE_CACHE_WINDOW_SEC", "0.80")))
DISTANCE_SOURCE = os.environ.get("DISTANCE_SOURCE", "mmwave").strip().lower()
VISION_MMWAVE_SOURCE_ALIASES = {"vision_mmwave", "vision-mmwave", "vision+mmwave", "mmwave_vision"}
VISION_MMWAVE_ENABLED = DISTANCE_SOURCE in VISION_MMWAVE_SOURCE_ALIASES

# 跟踪参数
DETECTION_CONFIRM_FRAMES = 3
LOST_CONFIRM_FRAMES = 3   # 防抖：连续 3 帧无人才确认丢人并开始旋转，避免检测偶发漏检
ACTION_COOLDOWN = 1  # 降低冷却时间，使前进更流畅
LOST_CONFIRM_FRAMES = int(os.environ.get("FOLLOW_LOST_CONFIRM_FRAMES", str(LOST_CONFIRM_FRAMES)))
FOLLOW_LOST_CONFIRM_SEC = max(0.0, float(os.environ.get("FOLLOW_LOST_CONFIRM_SEC", "0.0")))
SEARCH_COOLDOWN = 1
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))
FOLLOW_CENTER_LEFT_RATIO = float(os.environ.get("FOLLOW_CENTER_LEFT_RATIO", str(1.0 / 6.0)))
FOLLOW_CENTER_RIGHT_RATIO = float(os.environ.get("FOLLOW_CENTER_RIGHT_RATIO", str(4.0 / 6.0)))
FOLLOW_STEER_ENTER_LEFT_RATIO = float(os.environ.get("FOLLOW_STEER_ENTER_LEFT_RATIO", str(FOLLOW_CENTER_LEFT_RATIO)))
FOLLOW_STEER_ENTER_RIGHT_RATIO = float(os.environ.get("FOLLOW_STEER_ENTER_RIGHT_RATIO", str(FOLLOW_CENTER_RIGHT_RATIO)))
FOLLOW_STEER_RELEASE_LEFT_RATIO = float(os.environ.get("FOLLOW_STEER_RELEASE_LEFT_RATIO", str(FOLLOW_CENTER_LEFT_RATIO)))
FOLLOW_STEER_RELEASE_RIGHT_RATIO = float(os.environ.get("FOLLOW_STEER_RELEASE_RIGHT_RATIO", str(FOLLOW_CENTER_RIGHT_RATIO)))
FOLLOW_VISIBLE_ROTATE_LEFT_RATIO = float(os.environ.get("FOLLOW_VISIBLE_ROTATE_LEFT_RATIO", "0.0"))
FOLLOW_VISIBLE_ROTATE_RIGHT_RATIO = float(os.environ.get("FOLLOW_VISIBLE_ROTATE_RIGHT_RATIO", "1.0"))
FOLLOW_USE_VERTICAL_CENTER_GATE = os.environ.get("FOLLOW_USE_VERTICAL_CENTER_GATE", "0").strip() != "0"
FOLLOW_SEARCH_BEFORE_FIRST_PERSON = os.environ.get("FOLLOW_SEARCH_BEFORE_FIRST_PERSON", "0").strip() != "0"
FOLLOW_STARTUP_SEARCH_DELAY_SEC = float(os.environ.get("FOLLOW_STARTUP_SEARCH_DELAY_SEC", "0.0"))
FOLLOW_SEARCH_TIMEOUT_SEC = max(0.0, float(os.environ.get("FOLLOW_SEARCH_TIMEOUT_SEC", "4.0")))
FOLLOW_INITIAL_TARGET_CONFIRM_FRAMES = max(1, int(os.environ.get("FOLLOW_INITIAL_TARGET_CONFIRM_FRAMES", "1")))
FOLLOW_RELEASE_TARGET_ON_LOST = os.environ.get("FOLLOW_RELEASE_TARGET_ON_LOST", "1").strip() != "0"
SIDE_IR_BLOCKS_ROTATION = os.environ.get("SIDE_IR_BLOCKS_ROTATION", "1").strip() != "0"
SEARCH_ROTATE_FRONT_BLOCK_ENABLE = os.environ.get("SEARCH_ROTATE_FRONT_BLOCK_ENABLE", "1").strip() != "0"
SEARCH_ROTATE_DISTANCE_BLOCK_ENABLE = os.environ.get("SEARCH_ROTATE_DISTANCE_BLOCK_ENABLE", "1").strip() != "0"
BLOCKED_TURN_FORWARD_ENABLE = os.environ.get("BLOCKED_TURN_FORWARD_ENABLE", "1").strip() != "0"
BLOCKED_TURN_FORWARD_MAX_STEPS = max(0, int(os.environ.get("BLOCKED_TURN_FORWARD_MAX_STEPS", "2")))
SENSOR_RUNTIME_CONFIG = SensorRuntimeConfig(
    ir_enable=MODULE_IR_ENABLE,
    ultrasonic_enable=MODULE_ULTRASONIC_ENABLE,
    mmwave_enable=MODULE_MMWAVE_ENABLE,
    imu_enable=MODULE_IMU_ENABLE,
    imu_fail_soft=IMU_FAIL_SOFT,
    imu_log_enable=IMU_LOG_ENABLE,
    imu_log_every_sec=IMU_LOG_EVERY_SEC,
    side_ir_blocks_rotation=SIDE_IR_BLOCKS_ROTATION,
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
SEARCH_REID_DIAGNOSTIC_ENABLE = os.environ.get("SEARCH_REID_DIAGNOSTIC_ENABLE", "1").strip() != "0"
RKNN_CAMERA_ENABLE = os.environ.get("RKNN_CAMERA_ENABLE", "1").strip() != "0"
RKNN_CAMERA_DEVICE = os.environ.get("RKNN_CAMERA_DEVICE", "/dev/video1").strip()
RKNN_CAMERA_WIDTH = int(os.environ.get("RKNN_CAMERA_WIDTH", str(VISION_FRAME_WIDTH)))
RKNN_CAMERA_HEIGHT = int(os.environ.get("RKNN_CAMERA_HEIGHT", str(VISION_FRAME_HEIGHT)))
RKNN_CAMERA_FPS = float(os.environ.get("RKNN_CAMERA_FPS", "30.0"))
RKNN_CAMERA_FOURCC = os.environ.get("RKNN_CAMERA_FOURCC", "MJPG").strip()
RKNN_CAMERA_CAPTURE_MODE = os.environ.get("RKNN_CAMERA_CAPTURE_MODE", "gstreamer_mjpeg").strip().lower()
RKNN_CAMERA_RAW_OUTPUT = os.environ.get("RKNN_CAMERA_RAW_OUTPUT", "").strip()
RKNN_CAMERA_RETRY_INTERVAL_SEC = max(0.2, float(os.environ.get("RKNN_CAMERA_RETRY_INTERVAL_SEC", "2.0")))
RKNN_CAMERA_LATEST_DRAIN_MAX = max(0, int(os.environ.get("RKNN_CAMERA_LATEST_DRAIN_MAX", "8")))
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

PREDICTIVE_SEARCH_ENABLED = os.environ.get("PREDICTIVE_SEARCH_ENABLED", "0").strip() != "0"
PREDICTIVE_SEARCH_HISTORY_SEC = max(0.2, float(os.environ.get("PREDICTIVE_SEARCH_HISTORY_SEC", "1.2")))
_predictive_search_sec_raw = float(os.environ.get("PREDICTIVE_SEARCH_SEC", "0.0"))
if _predictive_search_sec_raw > 0:
    PREDICTIVE_SEARCH_SEC = _predictive_search_sec_raw
else:
    PREDICTIVE_SEARCH_SEC = max(0.6, min(1.5, TRACK_MEMORY_DEEPSORT_HOLD_SEC * 0.33))
PREDICTIVE_SEARCH_EDGE_MARGIN_RATIO = max(
    0.02,
    min(0.45, float(os.environ.get("PREDICTIVE_SEARCH_EDGE_MARGIN_RATIO", "0.12"))),
)
PREDICTIVE_SEARCH_MIN_X_MOTION_RATIO = max(
    0.0,
    min(0.5, float(os.environ.get("PREDICTIVE_SEARCH_MIN_X_MOTION_RATIO", "0.03"))),
)
PREDICTIVE_SEARCH_FAR_AREA_DROP_RATIO = max(
    0.1,
    min(0.95, float(os.environ.get("PREDICTIVE_SEARCH_FAR_AREA_DROP_RATIO", "0.60"))),
)
PREDICTIVE_SEARCH_FAR_CENTER_RATIO = max(
    0.05,
    min(0.5, float(os.environ.get("PREDICTIVE_SEARCH_FAR_CENTER_RATIO", "0.25"))),
)
PREDICTIVE_SEARCH_FAR_FORWARD_PERCENT = int(os.environ.get("PREDICTIVE_SEARCH_FAR_FORWARD_PERCENT", "25"))
PREDICTIVE_SEARCH_LOG_ENABLE = os.environ.get("PREDICTIVE_SEARCH_LOG_ENABLE", "1").strip() != "0"

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
_default_missing_forward = "0" if VISION_MMWAVE_ENABLED else "50"
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
    vision_mmwave_latency_sec=VISION_MMWAVE_LATENCY_SEC,
    vision_mmwave_cache_max_age_sec=VISION_MMWAVE_CACHE_MAX_AGE_SEC,
    vision_mmwave_use_async_cache=VISION_MMWAVE_USE_ASYNC_CACHE,
    ultrasonic_min_distance_m=ULTRASONIC_MIN_DISTANCE_M,
    ultrasonic_max_distance_m=ULTRASONIC_MAX_DISTANCE_M,
    ultrasonic_filter_window=ULTRASONIC_FILTER_WINDOW,
    ultrasonic_target_confirm_frames=ULTRASONIC_TARGET_CONFIRM_FRAMES,
    ultrasonic_brake_confirm_frames=ULTRASONIC_BRAKE_CONFIRM_FRAMES,
    ultrasonic_hysteresis_m=ULTRASONIC_HYSTERESIS_M,
    ultrasonic_immediate_brake_m=ULTRASONIC_IMMEDIATE_BRAKE_M,
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
# < FOLLOW_BRAKE_DISTANCE_M / 前方IR 触发时的紧急停止：优先用 percent 通道 brake(state=0x03)。
USE_PERCENT_BRAKE_FOR_EMERGENCY = True
EMERGENCY_STOP_GEAR = 3  # percent brake 失败时退回档位 stop（如 s3）

# 前进 -> 转向：
# - False（默认）：直接发差速旋转，不先发 brake / 不滑行降速
# - True：先线性降速滑行再转（ROTATE_PREP_COAST_*），更柔和
ROTATE_PREP_COAST_ENABLE = False
ROTATE_PREP_COAST_STEPS = 6
ROTATE_PREP_COAST_TOTAL_SEC = 0.50
# 单次旋转脉冲结束：一律 brake(state=03) + 进入 brake 保持态（周期补发），坡上防溜；不再用 0%% 滑行停结束旋转
ROTATE_END_USE_SOFT_STOP = False
ROTATE_PULSE_PAUSE_SEC = max(0.0, float(os.environ.get("ROTATE_PULSE_PAUSE_SEC", "0.06")))  # 脉冲旋转：每次旋转到点后强制停顿（秒），避免连续转出大圈
ROTATE_CHAIN_MEMORY_SEC = 1.00  # 旋转记忆窗口（秒）：该窗口内再次旋转按 CHAIN 力度，避免偶发大角度
ROTATE_PULSE_BRAKE_ENABLE = os.environ.get("ROTATE_PULSE_BRAKE_ENABLE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ROTATE_PULSE_STOP_MODE = os.environ.get("ROTATE_PULSE_STOP_MODE", "zero").strip().lower()
if ROTATE_PULSE_STOP_MODE in {"target_zero", "zero_target", "coast"}:
    ROTATE_PULSE_STOP_MODE = "zero"
elif ROTATE_PULSE_STOP_MODE not in {"zero", "brake"}:
    ROTATE_PULSE_STOP_MODE = "zero"
ROTATE_HOLD_STALE_SEC = max(
    0.20,
    float(os.environ.get("ROTATE_HOLD_STALE_SEC", "0.80")),
)

# 旋转（差速）参数：一个轮正转、另一个轮反转（百分比）
# 直行后首轮转向用较高力度；连续转向（上一动已是旋转，或刚结束一次旋转后的下一拍）用较低力度——轮姿不同所需力度不同
ROTATE_TURN_PERCENT_FROM_FORWARD = int(os.environ.get("ROTATE_TURN_PERCENT_FROM_FORWARD", "15"))  # 上一动不是旋转（如前进、首次、长时间停后）
ROTATE_TURN_PERCENT_CHAIN = int(os.environ.get("ROTATE_TURN_PERCENT_CHAIN", "10"))          # 上一动是旋转（左/右切换）或刚结束旋转后的下一拍转向
VISIBLE_STEER_INNER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_INNER_RATIO_PERCENT", "80"))
VISIBLE_STEER_OUTER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_OUTER_RATIO_PERCENT", "105"))
VISIBLE_STEER_STRONG_INNER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_STRONG_INNER_RATIO_PERCENT", "70"))
VISIBLE_STEER_STRONG_OUTER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_STRONG_OUTER_RATIO_PERCENT", "110"))
VISIBLE_STEER_STRONG_MARGIN_RATIO = float(os.environ.get("VISIBLE_STEER_STRONG_MARGIN_RATIO", "0.12"))
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
)

# 动作参数
ROTATE_DURATION = float(os.environ.get("ROTATE_DURATION", "0.15"))  # 脉冲模式下单次旋转时长；保持模式下仅作为日志/调试窗口
# 主循环取帧间隔（秒），越大越不占摄像头/编码管线，可减轻与板子录像 VENC 冲突导致的 drop audio / 卡死
PROCESS_FRAME_INTERVAL = 0.06   # 约16Hz，跟踪够用且减轻与录像争用（原 0.03 约33Hz 易与 CVI_RECORDER 抢 VENC）

# 距离参数（中间3×3 + 毫米波跟随时）
# 人在此距离以内不前进（速度视为 0）；超过后由 FollowSafetyController 按距离分档。
TARGET_DISTANCE = float(os.environ.get("TARGET_DISTANCE", "1.0"))
# 小于此距离（米）发 brake / 硬停 / 边缘区「太近不转」（统一阈值）
FOLLOW_BRAKE_DISTANCE_M = float(os.environ.get("FOLLOW_BRAKE_DISTANCE_M", "0.5"))
MMWAVE_OBSTACLE_THRESHOLD = float(os.environ.get("MMWAVE_OBSTACLE_THRESHOLD", "1.0"))  # 毫米波障碍物阈值（米），前方 1.0 m 内有障碍且未检测到人则停
ULTRASONIC_OBSTACLE_THRESHOLD = MMWAVE_OBSTACLE_THRESHOLD  # 保留旧变量名，后续超声融合时兼容
FORWARD_1M_DISTANCE = 1.0  # 绕障前进距离（米）

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
MOVEMENT_ACTIONS = {
    ACTION_FORWARD,
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
    ACTION_ROTATE_LEFT: "rotate_left",
    ACTION_ROTATE_RIGHT: "rotate_right",
    ACTION_STOP: "stop",
    ACTION_STEER_LEFT: "steer_left",
    ACTION_STEER_RIGHT: "steer_right",
}
ACTION_RUNTIME_SYMBOLS = ActionRuntimeSymbols(
    forward=ACTION_FORWARD,
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
    visible_steer_inner_ratio_percent=VISIBLE_STEER_INNER_RATIO_PERCENT,
    visible_steer_outer_ratio_percent=VISIBLE_STEER_OUTER_RATIO_PERCENT,
    motor_forward_raw_target=MOTOR_FORWARD_RAW_TARGET,
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
)

# 设置日志
logger = logging.getLogger("PersonTracker")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class TargetMotionHistory:
    """Keep recent selected-target motion for short lost-target search hints."""

    def __init__(
        self,
        history_sec: float,
        edge_margin_ratio: float,
        min_x_motion_ratio: float,
        far_area_drop_ratio: float,
        far_center_ratio: float,
    ) -> None:
        self.history_sec = max(0.2, float(history_sec))
        self.edge_margin_ratio = max(0.02, min(0.45, float(edge_margin_ratio)))
        self.min_x_motion_ratio = max(0.0, min(0.5, float(min_x_motion_ratio)))
        self.far_area_drop_ratio = max(0.1, min(0.95, float(far_area_drop_ratio)))
        self.far_center_ratio = max(0.05, min(0.5, float(far_center_ratio)))
        self._records: List[Dict[str, Any]] = []

    def record(self, frame_index: int, target: PersonTarget, distance_m: Optional[float]) -> None:
        x1, y1, x2, y2 = target.bbox
        cx, cy = target.center
        now = time.monotonic()
        self._records.append(
            {
                "ts": now,
                "frame": int(frame_index),
                "track_id": int(target.track_id),
                "cx": float(cx),
                "cy": float(cy),
                "area": float(target.area),
                "height": max(0.0, float(y2) - float(y1)),
                "distance_m": None if distance_m is None else float(distance_m),
            }
        )
        cutoff = now - max(self.history_sec * 2.0, self.history_sec + 0.5)
        self._records = [rec for rec in self._records if float(rec["ts"]) >= cutoff]

    def classify(self, width: int, height: int) -> Tuple[str, float, Dict[str, Any]]:
        if width <= 0 or height <= 0 or not self._records:
            return "unknown", 999.0, {}

        now = time.monotonic()
        last = self._records[-1]
        last_ts = float(last["ts"])
        age = now - last_ts
        recent = [rec for rec in self._records if last_ts - float(rec["ts"]) <= self.history_sec]
        if len(recent) < 2:
            return "unknown", age, {"samples": len(recent)}

        first = recent[0]
        last = recent[-1]
        dx_ratio = (float(last["cx"]) - float(first["cx"])) / float(width)
        last_x_ratio = float(last["cx"]) / float(width)
        center_offset_ratio = abs(last_x_ratio - 0.5)
        first_area = max(1.0, float(first["area"]))
        last_area = max(1.0, float(last["area"]))
        area_ratio = last_area / first_area

        first_distance = first.get("distance_m")
        last_distance = last.get("distance_m")
        distance_delta = None
        if first_distance is not None and last_distance is not None:
            distance_delta = float(last_distance) - float(first_distance)

        debug = {
            "samples": len(recent),
            "dx_ratio": dx_ratio,
            "last_x_ratio": last_x_ratio,
            "area_ratio": area_ratio,
            "distance_delta": distance_delta,
        }

        if (
            last_x_ratio <= self.edge_margin_ratio
            and dx_ratio <= -self.min_x_motion_ratio
        ):
            return "left_exit", age, debug
        if (
            last_x_ratio >= 1.0 - self.edge_margin_ratio
            and dx_ratio >= self.min_x_motion_ratio
        ):
            return "right_exit", age, debug

        far_by_area = area_ratio <= self.far_area_drop_ratio
        far_by_distance = distance_delta is not None and distance_delta >= 0.25
        if center_offset_ratio <= self.far_center_ratio and (far_by_area or far_by_distance):
            return "far_exit", age, debug

        return "unknown", age, debug


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
        self._rknn_camera_next_retry_ts = 0.0
        self._last_rknn_no_frame_log_ts = 0.0
        self._sensor_runtime = SensorRuntime(SENSOR_RUNTIME_CONFIG, logger=logger)
        logger.info(
            "LZ30EMA RS485 motor backend enabled: port=%s limit=%d%% forward_rpm=%d steer_rpm=%d rotate_rpm=%d rotate_rpm_visible=%d rotate_rpm_lost_wait=%d rotate_rpm_search=%d signs(left=%d,right=%d,forward=%d)",
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
        )
        if LOADED_CONFIG is not None:
            logger.info("已加载配置文件: %s", LOADED_CONFIG.path)
        logger.info(
            "模块开关: vision=%s, ir=%s, mmwave=%s, ultrasonic=%s, imu=%s, distance_source=%s, bunker=%s(%s)",
            MODULE_VISION_ENABLE,
            MODULE_IR_ENABLE,
            MODULE_MMWAVE_ENABLE,
            MODULE_ULTRASONIC_ENABLE,
            MODULE_IMU_ENABLE,
            DISTANCE_SOURCE,
            BUNKER_AVOID_ENABLE,
            BUNKER_DETECT_MODE,
        )
        logger.info(
            "Safety config: side_ir_blocks_rotation=%s blocked_turn_forward=%s max_steps=%d safety_stop_mode=%s safety_stop_reasons=%s",
            SIDE_IR_BLOCKS_ROTATION,
            BLOCKED_TURN_FORWARD_ENABLE,
            BLOCKED_TURN_FORWARD_MAX_STEPS,
            SAFETY_STOP_MODE,
            sorted(SAFETY_STOP_REASONS),
        )
        logger.info(
            "Follow search config: search_before_first=%s startup_delay=%.2f search_timeout=%.2f initial_confirm_frames=%d lost_confirm_sec=%.2f lost_confirm_frames=%d release_target_on_lost=%s",
            FOLLOW_SEARCH_BEFORE_FIRST_PERSON,
            FOLLOW_STARTUP_SEARCH_DELAY_SEC,
            FOLLOW_SEARCH_TIMEOUT_SEC,
            FOLLOW_INITIAL_TARGET_CONFIRM_FRAMES,
            FOLLOW_LOST_CONFIRM_SEC,
            LOST_CONFIRM_FRAMES,
            FOLLOW_RELEASE_TARGET_ON_LOST,
        )
        logger.info(
            "Distance/motion config: mmwave_match_mode=%s mmwave_angle_tie=%.1fdeg mmwave_latency=%.3fs mmwave_async=%s target=%.2fm brake=%.2fm speed_tiers=[<=1.3:%d,<=1.7:%d,<=2.1:%d,<=2.6:%d,<=3.2:%d,<=3.8:%d,<=4.5:%d,far:%d] max=%d fallback=%d forward_keepalive=%.3fs",
            VISION_MMWAVE_MATCH_MODE,
            VISION_MMWAVE_ANGLE_TIE_MARGIN_DEG,
            VISION_MMWAVE_LATENCY_SEC,
            VISION_MMWAVE_USE_ASYNC_CACHE,
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
            "Track memory: disabled in request_0513; target retention is handled by DeepSORT/IdentityBank"
        )
        logger.info(
            "Predictive search: enabled=%s history=%.2fs predict=%.2fs edge=%.2f min_dx=%.2f far_area=%.2f far_forward=%d",
            PREDICTIVE_SEARCH_ENABLED,
            PREDICTIVE_SEARCH_HISTORY_SEC,
            PREDICTIVE_SEARCH_SEC,
            PREDICTIVE_SEARCH_EDGE_MARGIN_RATIO,
            PREDICTIVE_SEARCH_MIN_X_MOTION_RATIO,
            PREDICTIVE_SEARCH_FAR_AREA_DROP_RATIO,
            PREDICTIVE_SEARCH_FAR_FORWARD_PERCENT,
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
        self._rknn_pipeline = RKNNVisionPipeline(cfg, logger=logger)
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
        self._current_steer_base_percent = 0
        self._current_steer_inner_ratio_percent = VISIBLE_STEER_INNER_RATIO_PERCENT
        self._current_steer_outer_ratio_percent = VISIBLE_STEER_OUTER_RATIO_PERCENT
        # 差速转向力度：由动作线程在切入旋转时按「上一动是否旋转」设定（见 ROTATE_TURN_PERCENT_*）
        self._current_rotate_turn_percent = ROTATE_TURN_PERCENT_FROM_FORWARD
        self._current_rotate_raw_target = MOTOR_ROTATE_RAW_TARGET
        self._current_rotate_raw_source = "default"
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
        # 需容纳多步动作（如 [STOP, 旋转]）；maxsize=1 会丢第二个，表现为旋转怪/不转
        self.action_queue = queue.Queue(maxsize=32)
        self.action_queue_lock = threading.Lock()
        self.stop_action_execution = False
        self.action_stop_event = threading.Event()
        self.action_thread = None
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
        self._last_rotate_pause_log_ts = 0.0
        self._last_rotate_pulse_active_log_ts = 0.0
        self._last_rotate_pulse_refresh_ts = 0.0
        self._last_rotate_hold_refresh_log_ts = 0.0
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
        self._last_action_queue_actions: List[int] = []
        self._last_action_queue_signature: Optional[Tuple] = None
        self._last_redundant_action_skip_log_ts = 0.0
        self._last_direct_stop_reason = ""
        self._last_direct_stop_ts = 0.0
        self._last_redundant_direct_stop_log_ts = 0.0
        self._last_distance_stop_log_key = None
        self._last_distance_hard_stop_log_ts = 0.0
        self._last_tracker_action_kind = None  # type: Optional[int]
        self._last_tracker_action_change_ts = 0.0
        self._last_tracker_action_change_frame = -1
        self._last_forward_like_refresh_ts = 0.0
        self._last_motor_dispatch_ts = 0.0
        self._last_motor_dispatch_action = None  # type: Optional[int]
        # 仅转向结束停：下一拍 ACTION_STOP 走软停（0%%），不抢紧急 brake
        self._use_soft_stop_next = False
        self._follow_controller = FollowSafetyController(
            FollowPolicyConfig(
                lost_confirm_frames=LOST_CONFIRM_FRAMES,
                lost_confirm_sec=FOLLOW_LOST_CONFIRM_SEC,
                action_cooldown=ACTION_COOLDOWN,
                search_cooldown=SEARCH_COOLDOWN,
                target_distance_m=TARGET_DISTANCE,
                brake_distance_m=FOLLOW_BRAKE_DISTANCE_M,
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
                center_left_ratio=FOLLOW_CENTER_LEFT_RATIO,
                center_right_ratio=FOLLOW_CENTER_RIGHT_RATIO,
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
                initial_target_confirm_frames=FOLLOW_INITIAL_TARGET_CONFIRM_FRAMES,
                release_target_on_lost=FOLLOW_RELEASE_TARGET_ON_LOST,
                side_ir_blocks_rotation=SIDE_IR_BLOCKS_ROTATION,
                search_rotate_front_block_enable=SEARCH_ROTATE_FRONT_BLOCK_ENABLE,
                search_rotate_distance_block_enable=SEARCH_ROTATE_DISTANCE_BLOCK_ENABLE,
                blocked_turn_forward_enable=BLOCKED_TURN_FORWARD_ENABLE,
                blocked_turn_forward_max_steps=BLOCKED_TURN_FORWARD_MAX_STEPS,
                predictive_search_enabled=PREDICTIVE_SEARCH_ENABLED,
                predictive_search_sec=PREDICTIVE_SEARCH_SEC,
                predictive_far_forward_percent=PREDICTIVE_SEARCH_FAR_FORWARD_PERCENT,
                visible_steer_inner_ratio_percent=VISIBLE_STEER_INNER_RATIO_PERCENT,
                visible_steer_outer_ratio_percent=VISIBLE_STEER_OUTER_RATIO_PERCENT,
                visible_steer_strong_inner_ratio_percent=VISIBLE_STEER_STRONG_INNER_RATIO_PERCENT,
                visible_steer_strong_outer_ratio_percent=VISIBLE_STEER_STRONG_OUTER_RATIO_PERCENT,
                visible_steer_strong_margin_ratio=VISIBLE_STEER_STRONG_MARGIN_RATIO,
            )
        )
        self._last_control_decision_log_key = None
        self._last_control_decision_reason = ""
        self._last_explicit_stop_reason = ""
        self._last_rknn_latest_drain_log_ts = 0.0
        self._distance_runtime = DistanceRuntime(
            self,
            DISTANCE_RUNTIME_CONFIG,
            sensor_runtime=self._sensor_runtime,
            logger=logger,
        )
        self._last_person_reid_debug_by_stable_id: Dict[int, List[Dict[str, Any]]] = {}
        self._motion_history = (
            TargetMotionHistory(
                history_sec=PREDICTIVE_SEARCH_HISTORY_SEC,
                edge_margin_ratio=PREDICTIVE_SEARCH_EDGE_MARGIN_RATIO,
                min_x_motion_ratio=PREDICTIVE_SEARCH_MIN_X_MOTION_RATIO,
                far_area_drop_ratio=PREDICTIVE_SEARCH_FAR_AREA_DROP_RATIO,
                far_center_ratio=PREDICTIVE_SEARCH_FAR_CENTER_RATIO,
            )
            if PREDICTIVE_SEARCH_ENABLED
            else None
        )
        self._last_predictive_search_log_key = None
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
        self._start_action_thread()

    def _should_hard_stop_now(self, action: Optional[int] = None) -> bool:
        """
        最高优先级硬停条件：
        - split 沙坑/水坑模型触发危险面积阈值
        - 任意一路 IR 触发时，无条件打断所有运动并清零双轮转速
        - 距离传感器距离有效且 < FOLLOW_BRAKE_DISTANCE_M（与中间3×3 跟车 brake 阈值一致）
        该检查会被动作 runtime 在持续动作时调用，用于打断危险动作避免“帧间空窗期撞人”。
        """
        try:
            if self._bunker_runtime.current_split_state() is not None:
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
        try:
            if DISTANCE_SOURCE in VISION_MMWAVE_SOURCE_ALIASES:
                state = self._distance_runtime.get_recent_vision_mmwave_state(
                    target_distance_m=TARGET_DISTANCE,
                    brake_distance_m=FOLLOW_BRAKE_DISTANCE_M,
                )
                d = state.used_distance_m
            else:
                state = self._distance_runtime.get_sensor_distance_state(
                    target_distance_m=TARGET_DISTANCE,
                    brake_distance_m=FOLLOW_BRAKE_DISTANCE_M,
                )
                d = state.used_distance_m
            triggered = (d is not None) and (d < FOLLOW_BRAKE_DISTANCE_M)
            if triggered:
                self._log_distance_stop_trigger("hard_stop", state, safety=True)
            return triggered
        except Exception:
            return False

    def _maybe_log_imu_sample(self) -> None:
        self._sensor_runtime.maybe_log_imu_sample(self.frame_index)

    def _start_action_thread(self):
        return self._action_runtime.start()

    def _action_names_for_log(self, actions: List[int]) -> List[str]:
        return [ACTION_NAMES.get(a, str(a)) for a in actions]

    def _action_queue_names_for_log(self, actions: List[int]) -> List[str]:
        return [ACTION_NAMES.get(a, str(a)) for a in actions]

    def _action_queue_snapshot_locked(self) -> List[int]:
        try:
            return list(self.action_queue.queue)
        except Exception:
            return []

    def _action_signature(self, action: int) -> Tuple:
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
                return (
                    action,
                    "raw",
                    int(MOTOR_STEER_RAW_TARGET),
                    int(self._current_steer_inner_ratio_percent),
                    int(self._current_steer_outer_ratio_percent),
                )
            return (
                action,
                "percent",
                int(self._current_steer_base_percent),
                int(self._current_steer_inner_ratio_percent),
                int(self._current_steer_outer_ratio_percent),
            )
        if action in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT):
            rotate_raw_target = int(getattr(self, "_current_rotate_raw_target", MOTOR_ROTATE_RAW_TARGET))
            if rotate_raw_target > 0:
                return (
                    action,
                    "raw",
                    rotate_raw_target,
                    str(getattr(self, "_current_rotate_raw_source", "default")),
                    bool(ROTATE_PULSE_BRAKE_ENABLE),
                    float(ROTATE_DURATION),
                )
            return (
                action,
                "percent",
                int(self._current_rotate_turn_percent),
                bool(ROTATE_PULSE_BRAKE_ENABLE),
                float(ROTATE_DURATION),
            )
        return (action,)

    def _actions_signature(self, actions: List[int]) -> Tuple:
        return tuple(self._action_signature(action) for action in actions)

    def _should_skip_redundant_action_queue(self, actions: List[int], reason: str) -> bool:
        if len(actions) != 1:
            return False
        action = actions[0]
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
        now = time.monotonic()
        if now - self._last_redundant_action_skip_log_ts >= 1.0:
            since_last_enqueue_ms = (
                (now - self._last_action_queue_replace_ts) * 1000.0
                if self._last_action_queue_replace_ts > 0
                else -1.0
            )
            logger.info(
                "action queue skip redundant: frame=%d reason=%s actions=%s current=%s signature=%s since_last_enqueue_ms=%.1f",
                self.frame_index,
                reason,
                self._action_names_for_log(actions),
                ACTION_NAMES.get(current_command, str(current_command)),
                signature,
                since_last_enqueue_ms,
            )
            self._last_redundant_action_skip_log_ts = now
        return True

    def _prepare_direct_stop(self, reason: str) -> None:
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
        logger.info(
            "direct stop prepared: frame=%d reason=%s old_command=%s was_rotate=%s",
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
        if current_command is not None or stop_pending:
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
                "direct stop skip redundant: frame=%d reason=%s last_sent_ms=%.1f repeat_interval=%.2fs brake_hold=%s queue=%s",
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
            "action queue clear: frame=%d reason=%s before=%d cleared=%d after=%d clear_ms=%.3f before_actions=%s after_actions=%s current=%s stop_flag=%s person_flag=%s",
            self.frame_index,
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
        action_signature = self._actions_signature(actions)
        cleared = 0
        queued = 0
        with self.action_queue_lock:
            before_actions = self._action_queue_snapshot_locked()
            before = len(before_actions)
            clear_start = time.perf_counter()
            while True:
                try:
                    self.action_queue.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
            clear_ms = (time.perf_counter() - clear_start) * 1000.0
            for action in actions:
                try:
                    self.action_queue.put_nowait(action)
                    queued += 1
                except queue.Full:
                    logger.warning(
                        "action queue full while replacing: frame=%d reason=%s queued=%d actions=%s",
                        self.frame_index,
                        reason,
                        queued,
                        self._action_names_for_log(actions),
                    )
                    break
            after_actions = self._action_queue_snapshot_locked()
            after = len(after_actions)
        total_ms = (time.perf_counter() - start) * 1000.0
        new_primary = actions[0] if actions else None
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
                "action kind switch: frame=%d reason=%s old=%s new=%s previous_kind_age_ms=%.1f frame_delta=%d since_last_enqueue_ms=%.1f",
                self.frame_index,
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
        if actions:
            self._action_queue_seq += 1
            self._last_dispatched_action = actions[0]
            self._last_action_queue_replace_ts = replace_ts
            self._last_action_queue_replace_frame = self.frame_index
            self._last_action_queue_seq = self._action_queue_seq
            self._last_action_queue_reason = reason
            self._last_action_queue_actions = list(actions)
            self._last_action_queue_signature = action_signature
        logger.info(
            "action queue replace: seq=%d frame=%d reason=%s actions=%s signature=%s before=%d cleared=%d queued=%d after=%d clear_ms=%.3f total_ms=%.3f queue_before=%s queue_after=%s current=%s last_dispatched=%s since_last_enqueue_ms=%.1f search=%s/%s",
            self._last_action_queue_seq,
            self.frame_index,
            reason,
            self._action_names_for_log(actions),
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
            "rotate_left": ACTION_ROTATE_LEFT,
            "rotate_right": ACTION_ROTATE_RIGHT,
            "steer_left": ACTION_STEER_LEFT,
            "steer_right": ACTION_STEER_RIGHT,
            "stop": ACTION_STOP,
        }.get(kind)

    def _action_int_to_kind(self, action: Optional[int]) -> Optional[str]:
        return {
            ACTION_FORWARD: "forward",
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
            "source=%s raw=%sm filtered=%sm used=%sm trigger=%s "
            "target_count=%d brake_count=%d latched(target=%s,brake=%s)"
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
            extras.append("detail=%s" % state.source_detail)
        if state.sample_age_sec is not None:
            extras.append("sample_age=%.0fms" % (float(state.sample_age_sec) * 1000.0))
        if state.target_angle_deg is not None:
            extras.append("target_angle=%.1fdeg" % float(state.target_angle_deg))
        if state.matched_angle_deg is not None:
            extras.append("matched_angle=%.1fdeg" % float(state.matched_angle_deg))
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
            "distance stop trigger: frame=%d reason=%s safety=%s %s target_threshold=%s brake_threshold=%s hysteresis=%.2f",
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
            "region=%s x_ratio=%.3f y_ratio=%.3f center_band=[%.2f,%.2f] "
            "rotate_band=[<%s,>%s]"
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
        return "raw=%s reid_uid=%s assign_reason=%s bbox_quality=%s" % (
            raw_track_id,
            reid_uid,
            reason,
            quality,
        )

    @staticmethod
    def _rotate_raw_target_for_reason(reason: str) -> Tuple[int, str]:
        reason = str(reason or "")
        if reason.startswith("lost_wait_"):
            return ROTATE_RAW_TARGET_LOST_WAIT or MOTOR_ROTATE_RAW_TARGET, "lost_wait"
        if reason.startswith("search_") or reason.startswith("predict_"):
            return ROTATE_RAW_TARGET_SEARCH or MOTOR_ROTATE_RAW_TARGET, "search"
        if reason.startswith("person_"):
            return ROTATE_RAW_TARGET_VISIBLE or MOTOR_ROTATE_RAW_TARGET, "visible"
        return MOTOR_ROTATE_RAW_TARGET, "default"

    def _process_detections_modular(self, width: int, height: int, persons: List[Tuple]) -> List[int]:
        """Use car_control_modular.controllers to decide actions."""
        self._explicit_stop_requested = False
        self._follow_controller.set_last_dispatched(self._action_int_to_kind(self._last_dispatched_action))

        obstacles_dict = self._get_obstacle_status()
        person_targets = self._persons_to_targets(persons)
        distance_target = self._distance_runtime.select_target(person_targets)
        distance_state = self._distance_runtime.get_frame_distance_state(
            int(width),
            distance_target,
            target_distance_m=TARGET_DISTANCE,
            brake_distance_m=FOLLOW_BRAKE_DISTANCE_M,
        )
        distance_m = distance_state.used_distance_m
        lost_intent = "unknown"
        lost_intent_age_sec = 0.0
        if self._motion_history is not None:
            if distance_target is not None:
                self._motion_history.record(self.frame_index, distance_target, distance_m)
            else:
                lost_intent, lost_intent_age_sec, intent_debug = self._motion_history.classify(int(width), int(height))
                if PREDICTIVE_SEARCH_LOG_ENABLE and lost_intent != "unknown":
                    log_key = (
                        lost_intent,
                        int(lost_intent_age_sec * 10),
                        int(float(intent_debug.get("samples", 0))),
                    )
                    if log_key != self._last_predictive_search_log_key:
                        self._last_predictive_search_log_key = log_key
                        logger.info(
                            "predictive_search hint: frame=%d intent=%s age=%.2fs samples=%s dx=%.3f x=%.3f area_ratio=%.3f dist_delta=%s",
                            self.frame_index,
                            lost_intent,
                            lost_intent_age_sec,
                            intent_debug.get("samples"),
                            float(intent_debug.get("dx_ratio", 0.0)),
                            float(intent_debug.get("last_x_ratio", 0.0)),
                            float(intent_debug.get("area_ratio", 0.0)),
                            intent_debug.get("distance_delta"),
                        )
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
            lost_intent=lost_intent,
            lost_intent_age_sec=lost_intent_age_sec,
            module_status={
                "vision": MODULE_VISION_ENABLE,
                "ir": MODULE_IR_ENABLE,
                "mmwave": MODULE_MMWAVE_ENABLE,
                "ultrasonic": MODULE_ULTRASONIC_ENABLE,
                "imu": MODULE_IMU_ENABLE,
                "bunker": BUNKER_AVOID_ENABLE,
            },
        )
        active_before = self._follow_controller.active_target_id
        search_before = self._follow_controller.search_state
        decision = self._follow_controller.decide(self.frame_index, frame)

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
        self.is_forwarding = decision.is_forwarding
        self._current_forward_percent = decision.current_forward_percent
        if decision.stop_action_execution:
            self.stop_action_execution = True
        if decision.clear_action_queue:
            self._clear_action_queue(f"controller_clear:{decision.reason}")

        actions: List[int] = []
        for action in decision.actions:
            if action.kind == "forward":
                self._current_forward_percent = int(action.speed_percent)
                self._current_steer_base_percent = 0
                self.is_forwarding = True
            elif action.kind in ("steer_left", "steer_right"):
                self._current_steer_base_percent = int(action.speed_percent)
                self._current_steer_inner_ratio_percent = int(action.steer_inner_ratio_percent)
                self._current_steer_outer_ratio_percent = int(action.steer_outer_ratio_percent)
                self.is_forwarding = True
            elif action.kind in ("rotate_left", "rotate_right"):
                rotate_raw, rotate_raw_source = self._rotate_raw_target_for_reason(decision.reason)
                self._current_rotate_raw_target = int(rotate_raw)
                self._current_rotate_raw_source = rotate_raw_source
            action_int = self._action_kind_to_int(action.kind)
            if action_int is not None and action.kind != "idle":
                actions.append(action_int)

        if decision.reason:
            action_kinds = [a.kind for a in decision.actions]
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
                elif search_before in ("searching", "predictive"):
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
                    "control decision: frame=%d reason=%s actions=%s stop=%s speed=%s persons=%d target_source=%s active_before=%s active_after=%s target_id=%s target_center=%s target_area=%s target_region=(%s) target_quality=(%s) visible_largest_id=%s visible_largest_center=%s distance=%s distance_detail=(%s) obstacles(front=%s,left=%s,right=%s) search=%s/%s",
                    self.frame_index,
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
            if any(k in ("rotate_left", "rotate_right") for k in action_kinds):
                logger.info(
                    "rotate decision: frame=%d reason=%s actions=%s target_id=%s target_center=%s target_region=(%s) target_quality=(%s) largest_id=%s search=%s/%s pulse_enable=%s duration=%.3fs pause=%.3fs hold_stale=%.3fs raw=%d raw_source=%s default_raw=%d",
                    self.frame_index,
                    decision.reason,
                    action_kinds,
                    target_id,
                    target_center,
                    target_region,
                    target_quality,
                    largest_id,
                    self.search_state,
                    self.search_direction,
                    ROTATE_PULSE_BRAKE_ENABLE,
                    ROTATE_DURATION,
                    ROTATE_PULSE_PAUSE_SEC,
                    ROTATE_HOLD_STALE_SEC,
                    int(self._current_rotate_raw_target),
                    self._current_rotate_raw_source,
                    MOTOR_ROTATE_RAW_TARGET,
                )
            if any(k in ("steer_left", "steer_right") for k in action_kinds):
                logger.info(
                    "steer decision: frame=%d reason=%s actions=%s target_id=%s target_center=%s largest_id=%s search=%s/%s base=%d inner_ratio=%d outer_ratio=%d raw=%d",
                    self.frame_index,
                    decision.reason,
                    action_kinds,
                    target_id,
                    target_center,
                    largest_id,
                    self.search_state,
                    self.search_direction,
                    self._current_steer_base_percent,
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
        if self.search_state not in ("searching", "predictive"):
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

    def _consume_track_records(self, records, width: int, height: int, source_name: str) -> None:
        all_dets = []
        person_candidates = []
        person_track_debug = []
        skipped_predicted_tracks = 0
        skipped_unconfirmed_reid = 0
        reid_debug_by_stable_id: Dict[int, List[Dict[str, Any]]] = {}
        if VISION_TRACK_LOG_ENABLE:
            if records:
                track_parts = []
                for rec in records:
                    bbox_dbg = [round(float(v), 1) for v in (rec.x1, rec.y1, rec.x2, rec.y2)]
                    track_parts.append(
                        "track_id=%s reid_uid=%s state=%s tsu=%s class=%s score=%.3f area=%.0f bbox=%s"
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
                    "%s tracks frame=%d count=%d: %s",
                    source_name,
                    self.frame_index,
                    len(records),
                    "; ".join(track_parts),
                )
            elif VISION_TRACK_LOG_EMPTY_EVERY > 0 and self.frame_index % VISION_TRACK_LOG_EMPTY_EVERY == 0:
                logger.info("%s tracks frame=%d count=0", source_name, self.frame_index)

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
            debug_entry = {
                "stable_id": int(stable_id),
                "raw_track_id": int(rec.track_id),
                "reid_uid": int(reid_uid),
                "time_since_update": int(time_since_update),
                "score": float(rec.score),
                "area": float(area),
                "bbox": [round(float(v), 1) for v in bbox],
                "assignment": self._identity_assignment_debug_for_track(int(rec.track_id)),
            }
            reid_debug_by_stable_id.setdefault(int(stable_id), []).append(debug_entry)
            if VISION_REID_ENABLE and reid_uid <= 0:
                skipped_unconfirmed_reid += 1
                person_track_debug.append((area, stable_id, rec, None))
                continue
            candidate = {
                "bbox": bbox,
                "stable_id": stable_id,
                "score": float(rec.score),
                "area": area,
                "rec": rec,
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
                    "%s visible_largest frame=%d mode=largest_area stable_id=%s track_id=%s reid_uid=%s state=%s tsu=%s score=%.3f area=%.0f center=(%.1f, %.1f)",
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
                )
            elif person_track_debug:
                logger.info(
                    "%s visible_largest frame=%d mode=largest_area candidate=none person_records=%d skipped_predicted=%d skipped_unconfirmed_reid=%d control_use_predicted=%s reid_required=%s",
                    source_name,
                    self.frame_index,
                    len(person_track_debug),
                    int(skipped_predicted_tracks),
                    int(skipped_unconfirmed_reid),
                    bool(VISION_CONTROL_USE_PREDICTED_TRACKS),
                    bool(VISION_REID_ENABLE),
                )
            elif records:
                logger.info(
                    "%s frame=%d no valid person after filter records=%d person_class=%d threshold=%.3f",
                    source_name,
                    self.frame_index,
                    len(records),
                    PERSON_CLASS_ID,
                    CONFIDENCE_THRESHOLD,
                )
            if (skipped_predicted_tracks > 0 or skipped_unconfirmed_reid > 0) and selected_candidates:
                logger.info(
                    "%s control filter frame=%d skipped_predicted=%d skipped_unconfirmed_reid=%d accepted_persons=%d control_use_predicted=%s reid_required=%s",
                    source_name,
                    self.frame_index,
                    int(skipped_predicted_tracks),
                    int(skipped_unconfirmed_reid),
                    len(selected_candidates),
                    bool(VISION_CONTROL_USE_PREDICTED_TRACKS),
                    bool(VISION_REID_ENABLE),
                )

        if self._handle_hazard_safety_state(
            self._bunker_runtime.check_merged_dets(all_dets, float(width * height))
        ):
            return

        self._queue_actions_for_persons(width, height, persons)

    def _queue_actions_for_persons(self, width: int, height: int, persons: List[Tuple]) -> None:
        was_searching_before = self.search_state in ("searching", "predictive")
        actions = self._process_detections_modular(width, height, persons)

        if was_searching_before and self.search_state not in ("searching", "predictive"):
            self._clear_action_queue("search_to_follow")
            if not self._should_skip_redundant_direct_stop("search_to_follow"):
                self._prepare_direct_stop("search_to_follow")
                self._action_runtime.send_stop_with_brake_hold("search_to_follow")
                self._mark_direct_stop_sent("search_to_follow")
            logger.info("已清空搜索动作队列并停止，开始跟踪")

        if actions:
            logger.info(
                "生成动作: frame=%d reason=%s actions=%s",
                self.frame_index,
                self._last_control_decision_reason,
                self._action_names_for_log(actions),
            )
            if any(a in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT) for a in actions):
                logger.info(
                    "rotate action queued: frame=%d actions=%s search=%s/%s queue_size_before=%d last_dispatched=%s pulse_enable=%s duration=%.3fs pause=%.3fs hold_stale=%.3fs raw=%d raw_source=%s default_raw=%d",
                    self.frame_index,
                    self._action_names_for_log(actions),
                    self.search_state,
                    self.search_direction,
                    self.action_queue.qsize(),
                    ACTION_NAMES.get(self._last_dispatched_action, str(self._last_dispatched_action)),
                    ROTATE_PULSE_BRAKE_ENABLE,
                    ROTATE_DURATION,
                    ROTATE_PULSE_PAUSE_SEC,
                    ROTATE_HOLD_STALE_SEC,
                    int(self._current_rotate_raw_target),
                    self._current_rotate_raw_source,
                    MOTOR_ROTATE_RAW_TARGET,
                )
            if any(a in (ACTION_STEER_LEFT, ACTION_STEER_RIGHT) for a in actions):
                logger.info(
                    "steer action queued: frame=%d actions=%s search=%s/%s queue_size_before=%d last_dispatched=%s base=%d inner_ratio=%d outer_ratio=%d raw=%d",
                    self.frame_index,
                    self._action_names_for_log(actions),
                    self.search_state,
                    self.search_direction,
                    self.action_queue.qsize(),
                    ACTION_NAMES.get(self._last_dispatched_action, str(self._last_dispatched_action)),
                    self._current_steer_base_percent,
                    self._current_steer_inner_ratio_percent,
                    self._current_steer_outer_ratio_percent,
                    MOTOR_STEER_RAW_TARGET,
                )
        else:
            logger.debug("未生成动作")

        if actions:
            reason = self._last_control_decision_reason or "decision"
            if not self._should_skip_redundant_action_queue(actions, reason):
                self._replace_action_queue(actions, reason)
        else:
            if self._explicit_stop_requested:
                reason = self._last_explicit_stop_reason or "explicit_stop"
                if not self._should_skip_redundant_direct_stop(reason):
                    self._clear_action_queue(f"explicit_stop:{reason}")
                    self._prepare_direct_stop(reason)
                    self._action_runtime.send_stop_with_brake_hold(reason)
                    self._mark_direct_stop_sent(reason)

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
        self._rknn_camera = cap
        logger.info("RKNN camera started: %s (%s)", RKNN_CAMERA_DEVICE, chosen)

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
        read_latest = getattr(self._rknn_camera, "read_latest", None)
        if callable(read_latest):
            ok, frame, drained = read_latest(max_drain=RKNN_CAMERA_LATEST_DRAIN_MAX)
            if drained > 0:
                now = time.monotonic()
                if now - self._last_rknn_latest_drain_log_ts >= 2.0:
                    self._last_rknn_latest_drain_log_ts = now
                    logger.info(
                        "RKNN camera using newest queued frame: drained=%d max_drain=%d",
                        drained,
                        RKNN_CAMERA_LATEST_DRAIN_MAX,
                    )
        else:
            ok, frame = self._rknn_camera.read()
        if not ok or frame is None:
            logger.warning("RKNN camera read failed: %s", RKNN_CAMERA_DEVICE)
            try:
                self._rknn_camera.release()
            except Exception:
                pass
            self._rknn_camera = None
            return
        self.process_external_frame(frame, "BGR")

    def process_external_frame(self, frame, frame_format: str = "BGR"):
        """Process one externally supplied frame through RKNN vision and control."""
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

        records = self._rknn_pipeline.process_frame(frame, frame_format)
        width = int(getattr(self._rknn_pipeline, "last_frame_width", VISION_FRAME_WIDTH))
        height = int(getattr(self._rknn_pipeline, "last_frame_height", VISION_FRAME_HEIGHT))
        self._consume_track_records(records, width, height, "rknn")
        return records

    def process_frame(self):
        return self._process_frame_rknn_camera()

    def run(self):
        """Run the camera/frame processing loop."""
        logger.info("开始运行人员跟踪（避障版本，vision_engine=%s）...", self._vision_engine)
        try:
            while self.running:
                self.process_frame()
                time.sleep(PROCESS_FRAME_INTERVAL)

        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭...")
        finally:
            self.running = False
            self.action_stop_event.set()
            # 先发 STOP，确保 Ctrl+C 后车一定停（避免中断发生在 detect 后、发停前的空窗期）
            try:
                self._action_runtime.send_stop_with_brake_hold(self._last_explicit_stop_reason or "explicit_stop")
            except Exception as e:
                logger.warning(f"finally 中发送 STOP 失败: {e}")
            if self.action_thread:
                self.action_thread.join(timeout=1.0)
            try:
                if self._rknn_camera is not None:
                    self._rknn_camera.release()
                    self._rknn_camera = None
                    logger.info("RKNN camera released")
            except Exception as e:
                logger.warning("RKNN camera release failed: %s", e)
            try:
                if self._rknn_pipeline is not None:
                    self._rknn_pipeline.close()
                    self._rknn_pipeline = None
                    logger.info("RKNN vision pipeline closed")
            except Exception as e:
                logger.warning("RKNN vision pipeline close failed: %s", e)
            # 收尾前再发一次 STOP，确保车已停（应对中断发生在 detect 后、发停前的空窗期）
            try:
                self._action_runtime.send_robot_command(ACTION_STOP)
            except Exception as e:
                logger.warning(f"finally 中再次发送 STOP 失败: {e}")
            self._bunker_runtime.close()
            self._sensor_runtime.close()
            try:
                self._motor_backend.close()
            except Exception as e:
                logger.warning("MSSD motor backend close failed: %s", e)
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
