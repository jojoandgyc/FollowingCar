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
import math
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from typing import Any, Dict, List, Tuple, Optional

from car_control_modular.config_loader import preload_config_from_argv
from car_control_modular.control_types import (
    SteeringFeedback,
    ControlAction,
    ControlDecision,
    DistanceState,
    DepthLinearTiming,
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
from car_control_modular.async_console_logging import BoundedAsyncConsoleHandler
from car_control_modular.deferred_diagnostics import deferred_diagnostics
from car_control_modular.capture_pipeline_timing import capture_pipeline_timing
from car_control_modular.detector_identity_lease import (
    from_assignment as detector_identity_lease_from_assignment, motion_identity_live,
    detector_execution_supported, MAX_FULL_RESULT_AGE_SEC, MAX_VALIDATED_VISIBILITY_AGE_SEC,
    ValidatedVisualObservation, validated_visual_observation,
    publish_visual_identity_evidence,
)
from car_control_modular.action_queue_policy import merge_pending_actions
from car_control_modular.search_identity_evidence import confirmed_search_candidate_uid
from car_control_modular.action_command import ActionCommandSnapshot
from car_control_modular.search_reacquire_braking import (search_brake_reason, handoff_yaw_limit,
    limit_handoff_yaw, handoff_retirement_reason, observe_moving_handoff, moving_handoff_yaw)
from car_control_modular.distance_runtime import DistanceRuntime, DistanceRuntimeConfig
from car_control_modular.depth_async_scheduler import DepthAsyncScheduler
from car_control_modular.depth_authority_timing import MAX_FORWARD_DEPTH_TTL_SEC
from car_control_modular.depth_target_geometry import resolve_depth_target_observation, resolve_braking_target_observation
from car_control_modular.depth_continuation import (
    continuation_feedback_speeds, continuation_speed_cap,
    relative_continuation_speed_cap,
    RELATIVE_CONTINUATION_FEEDBACK_MAX_AGE_SEC,
    RELATIVE_CONTINUATION_REVERSE_TAIL_RPM,
)
from car_control_modular.startup_identity import resolve_initial_identity_confirmation
from car_control_modular.cropped_depth_observation import resolve_cropped_depth_observation
from car_control_modular.follow_distance_hold import is_follow_distance_hold
from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController
from car_control_modular.short_follow_adapter import ShortFollowAdapter
from car_control_modular.hazard_runtime import BunkerHazardRuntime, BunkerHazardRuntimeConfig
from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig, normalize_mssd_stop_mode
from car_control_modular.motor_ramp import validate_acceleration_rpm_s
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
from car_control_modular.search_observation_retry import (
    confirmed_observation_for_release,
    SearchObservationRetry, DetectorObservationSettlement, retry_evidence_source,
    DEFERRED_EDGE_COVERAGE, incomparable_edge_observation,
)
from car_control_modular.historical_direction_backfill import (
    HistoricalDirectionBackfill,
    HistoricalDirectionCandidate,
)
from car_control_modular.visual_steering_evidence import matching_control_x
from car_control_modular.steering_limits import effective_correction_limit, clamp_correction
from car_control_modular.turn_buildup import steering_buildup_allowed
from car_control_modular.predictive_turn_brake import qualified_countersteer, MAX_CAPTURE_AGE_SEC
from car_control_modular.lateral_intent import (
    with_forward_continuation,
    LateralControlIntent,
    LateralIntentStore,
    slew_signed_rpm,
)
from car_control_modular.low_quality_lateral import MultiPersonLateralGate
from car_control_modular.search_brake_observation import (
    SearchBrakeObservation, provisional_reacquire_direction,
)
from car_control_modular.near_yaw_parking import (
    NearYawParkRequest, newer_visual_evidence, predictive_or_center_stop,
    low_speed_turn_stop_reason,
)
from car_control_modular.video_recorder import (
    AsyncVideoRecorder,
    VideoControlOverlay,
    VideoRecorderConfig,
    VideoTrackOverlay,
)
from car_control_modular.video_follow_telemetry import build_follow_snapshot, recording_authority

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
MOTOR_RAMP_DIAGNOSTICS_ENABLE = os.environ.get("MOTOR_RAMP_DIAGNOSTICS_ENABLE", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
_motor_acceleration_text = os.environ.get("MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S", "").strip()
MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S = validate_acceleration_rpm_s(
    int(_motor_acceleration_text) if _motor_acceleration_text else None,
)
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
ASTRA_DEPTH_MAX_UNCONFIRMED_JUMP_RATE_M_S = max(
    0.1,
    float(os.environ.get("ASTRA_DEPTH_MAX_UNCONFIRMED_JUMP_RATE_M_S", "3.0")),
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
# Depth原始流是30 FPS；纵向监督线程按独立ROI年龄门复用最近人物框。
# ROI取样、<=180ms新控制、原前进授权的有界延续分别校验。
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
        0.50,
        float(os.environ.get("ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", "0.18")),
    ),
)
# Separate clock: changing the RGB sampling window must never extend a
# previously accepted physical Depth motor lease (including at actual write).
ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC = max(
    0.03, min(MAX_FORWARD_DEPTH_TTL_SEC, float(os.environ.get("ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC", "0.18")))
)
# New control samples and all reverse grants retain their original 180ms gate.
ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC = 0.18
ASTRA_DEPTH_CONTINUATION_SPEED_CAP_ENABLE = os.environ.get("ASTRA_DEPTH_CONTINUATION_SPEED_CAP_ENABLE", "0") == "1"
ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE = os.environ.get("ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE", "0") == "1"
ASTRA_DEPTH_CONTINUATION_OVERSHOOT_M = float(os.environ.get("ASTRA_DEPTH_CONTINUATION_OVERSHOOT_M", "0.20"))
if (not math.isfinite(ASTRA_DEPTH_CONTINUATION_OVERSHOOT_M)
        or not 0 <= ASTRA_DEPTH_CONTINUATION_OVERSHOOT_M <= .30):
    raise ValueError("ASTRA_DEPTH_CONTINUATION_OVERSHOOT_M must be finite and within [0, 0.30]")
# Stage-2 guard: the Depth supervisor may assist longitudinal control, but it
# must remain a low-speed authority independent of the global follow cap.
ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT = max(
    0,
    min(
        100,
        int(os.environ.get("ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT", "20")),
    ),
)
# Keep the conservative cap near the setpoint, but allow a reliable far target
# to be approached faster. The cap is applied after the distance PID and does
# not limit the lateral differential correction.
ASTRA_DEPTH_LONGITUDINAL_FAR_DISTANCE_M = max(
    ASTRA_DEPTH_NEAR_GUARD_DISTANCE_M,
    float(os.environ.get("ASTRA_DEPTH_LONGITUDINAL_FAR_DISTANCE_M", "2.20")),
)
ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT = max(
    ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT,
    min(
        100,
        int(os.environ.get("ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT", "60")),
    ),
)
# Compatibility symbol only: deduplication now uses physical Depth timestamps.
ASTRA_DEPTH_VISION_DEDUPE_SEC = 0.0
VISION_CONTROL_MAX_RESULT_AGE_SEC = max(
    0.05,
    min(0.50, float(os.environ.get("VISION_CONTROL_MAX_RESULT_AGE_SEC", "0.18"))),
)
# Historical direction evidence only. This is NOT a motion/Depth lease and
# must never extend VISION_CONTROL_MAX_RESULT_AGE_SEC or renew an old command.
SEARCH_BRAKE_DIRECTION_OBSERVATION_MAX_AGE_SEC = 0.50

# 跟踪参数
DETECTION_CONFIRM_FRAMES = 3
LOST_CONFIRM_FRAMES = 3   # 防抖：连续 3 帧无人才确认丢人并开始旋转，避免检测偶发漏检
ACTION_COOLDOWN = max(1, int(os.environ.get("FOLLOW_ACTION_COOLDOWN_FRAMES", "1")))
LOST_CONFIRM_FRAMES = int(os.environ.get("FOLLOW_LOST_CONFIRM_FRAMES", str(LOST_CONFIRM_FRAMES)))
FOLLOW_LOST_CONFIRM_SEC = max(0.0, float(os.environ.get("FOLLOW_LOST_CONFIRM_SEC", "0.0")))
TARGET_DISTANCE = float(os.environ.get("TARGET_DISTANCE", "1.5"))
FOLLOW_DISTANCE_AUTO_TUNE = os.environ.get(
    "FOLLOW_DISTANCE_AUTO_TUNE", "1"
).strip().lower() in ("1", "true", "yes", "on")
FOLLOW_STALE_DIRECTION_RECOVERY_ENABLE = os.environ.get(
    "FOLLOW_STALE_DIRECTION_RECOVERY_ENABLE",
    "0",
).strip().lower() in ("1", "true", "yes", "on")
FOLLOW_DIRECTION_HISTORY_ENABLE = os.environ.get(
    "FOLLOW_DIRECTION_HISTORY_ENABLE",
    "1",
).strip().lower() in ("1", "true", "yes", "on")
HISTORICAL_DIRECTION_BACKFILL_ENABLE = os.environ.get(
    "HISTORICAL_DIRECTION_BACKFILL_ENABLE", "1"
).strip().lower() in ("1", "true", "yes", "on")
HISTORICAL_DIRECTION_BACKFILL_MAX_AGE_SEC = max(
    0.20, float(os.environ.get("HISTORICAL_DIRECTION_BACKFILL_MAX_AGE_SEC", "0.70"))
)
HISTORICAL_DIRECTION_BACKFILL_MIN_SAMPLES = max(
    2, int(os.environ.get("HISTORICAL_DIRECTION_BACKFILL_MIN_SAMPLES", "2"))
)
HISTORICAL_DIRECTION_BACKFILL_MAX_CAPTURE_GAP = max(
    1, int(os.environ.get("HISTORICAL_DIRECTION_BACKFILL_MAX_CAPTURE_GAP", "6"))
)
HISTORICAL_DIRECTION_BACKFILL_MAX_CENTER_JUMP_RATIO = max(
    0.05,
    min(0.50, float(os.environ.get("HISTORICAL_DIRECTION_BACKFILL_MAX_CENTER_JUMP_RATIO", "0.30"))),
)
HISTORICAL_DIRECTION_BACKFILL_MIN_AREA_SIMILARITY = max(
    0.05,
    min(1.0, float(os.environ.get("HISTORICAL_DIRECTION_BACKFILL_MIN_AREA_SIMILARITY", "0.45"))),
)
HISTORICAL_DIRECTION_BACKFILL_CONFIDENCE_CAP = max(
    0.20,
    min(0.70, float(os.environ.get("HISTORICAL_DIRECTION_BACKFILL_CONFIDENCE_CAP", "0.70"))),
)
HISTORICAL_DIRECTION_BACKFILL_MIN_SCORE = max(
    0.0,
    min(1.0, float(os.environ.get("HISTORICAL_DIRECTION_BACKFILL_MIN_SCORE", "0.20"))),
)
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
    TARGET_DISTANCE
    if FOLLOW_DISTANCE_AUTO_TUNE
    else float(os.environ.get("FOLLOW_LOST_FORWARD_HOLD_MIN_DISTANCE_M", "1.50")),
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
    2,
    int(os.environ.get("SEARCH_CONFIRMED_REACQUIRE_FRAMES", "2")),
)
SEARCH_REACQUIRE_DEPTH_GATE_ENABLE = os.environ.get(
    "SEARCH_REACQUIRE_DEPTH_GATE_ENABLE", "1"
).strip() != "0"
SEARCH_REACQUIRE_DEPTH_CONFIRM_FRAMES = max(
    2,
    int(os.environ.get("SEARCH_REACQUIRE_DEPTH_CONFIRM_FRAMES", "2")),
)
SEARCH_REACQUIRE_DEPTH_MAX_AGE_SEC = max(
    0.12,
    float(os.environ.get("SEARCH_REACQUIRE_DEPTH_MAX_AGE_SEC", "0.20")),
)
VISUAL_REACQUIRE_HOLD_ENABLE = os.environ.get(
    "VISUAL_REACQUIRE_HOLD_ENABLE", "1"
).strip() != "0"
VISUAL_REACQUIRE_HOLD_SEC = max(
    0.15,
    min(0.50, float(os.environ.get("VISUAL_REACQUIRE_HOLD_SEC", "0.30"))),
)
VISUAL_REACQUIRE_HOLD_MAX_CENTER_JUMP_RATIO = max(
    0.08,
    min(
        0.45,
        float(os.environ.get("VISUAL_REACQUIRE_HOLD_MAX_CENTER_JUMP_RATIO", "0.30")),
    ),
)
VISUAL_REACQUIRE_HOLD_MIN_AREA_SIMILARITY = max(
    0.20,
    min(
        0.95,
        float(os.environ.get("VISUAL_REACQUIRE_HOLD_MIN_AREA_SIMILARITY", "0.45")),
    ),
)
SEARCH_EVIDENCE_GATE_ENABLE = (
    os.environ.get("SEARCH_EVIDENCE_GATE_ENABLE", "1").strip() != "0"
)
SEARCH_EVIDENCE_RETRY_ENABLE = os.environ.get("SEARCH_EVIDENCE_RETRY_ENABLE", "0").strip() != "0"
SEARCH_EVIDENCE_HOLD_FRAMES = max(
    1, int(os.environ.get("SEARCH_EVIDENCE_HOLD_FRAMES", "2"))
)
SEARCH_EVIDENCE_MAX_HOLD_SEC = max(
    0.10,
    min(0.50, float(os.environ.get("SEARCH_EVIDENCE_MAX_HOLD_SEC", "0.30"))),
)
SEARCH_EVIDENCE_REARM_MISSING_FRAMES = max(
    1, int(os.environ.get("SEARCH_EVIDENCE_REARM_MISSING_FRAMES", "8"))
)
SEARCH_EVIDENCE_PROBE_CONFIRM_FRAMES = max(
    1, int(os.environ.get("SEARCH_EVIDENCE_PROBE_CONFIRM_FRAMES", "2"))
)
SEARCH_EVIDENCE_PROBE_EDGE_MIN_SCORE = max(0.0, min(1.0, float(
    os.environ.get("SEARCH_EVIDENCE_PROBE_EDGE_MIN_SCORE", "0.20"))))
SEARCH_EVIDENCE_PROBE_EDGE_CENTER_RATIO = max(0.0, min(0.49, float(
    os.environ.get("SEARCH_EVIDENCE_PROBE_EDGE_CENTER_RATIO", "0.15"))))
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
    min(18, int(os.environ.get("SEARCH_CANDIDATE_ACQUIRE_RAW_RPM", "5"))),
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
    astra_depth_max_unconfirmed_jump_rate_m_s=(
        ASTRA_DEPTH_MAX_UNCONFIRMED_JUMP_RATE_M_S
    ),
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
SINGLE_PERSON_GEOMETRY_MAX_AREA_CHANGE_RATIO = max(
    0.05,
    min(0.95, float(os.environ.get("SINGLE_PERSON_GEOMETRY_MAX_AREA_CHANGE_RATIO", "0.60"))),
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
RKNN_CAMERA_DEVICE = os.environ.get(
    "RKNN_CAMERA_DEVICE",
    "/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0",
).strip()
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
    # Production Depth must range the current detector crop, never silently
    # fall back to DeepSORT's expanded steering/display box.
    vision_depth_require_detector_bbox=True,
    vision_depth_detector_bbox_max_age_sec=ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC,
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
# 单次旋转脉冲结束后按同方向过渡或 ROTATE_PULSE_STOP_MODE 进入视觉观察窗口。
ROTATE_END_USE_SOFT_STOP = False
ROTATE_PULSE_PAUSE_SEC = max(0.0, float(os.environ.get("ROTATE_PULSE_PAUSE_SEC", "0.00")))  # 固定暂停仅作兼容项；搜索优先使用同方向过渡
ROTATE_PULSE_OBSERVE_MIN_FRAMES = max(
    0,
    int(os.environ.get("ROTATE_PULSE_OBSERVE_MIN_FRAMES", "1")),
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
    float(os.environ.get("ROTATE_PULSE_SETTLE_TIMEOUT_SEC", "0.18")),
)
ROTATE_PULSE_SETTLE_FEEDBACK_STALE_SEC = max(
    0.05,
    float(os.environ.get("ROTATE_PULSE_SETTLE_FEEDBACK_STALE_SEC", "0.30")),
)
ROTATE_PULSE_SETTLE_MAX_WHEEL_RPM = max(
    0,
    int(os.environ.get("ROTATE_PULSE_SETTLE_MAX_WHEEL_RPM", "5")),
)
ROTATE_PULSE_SETTLE_MAX_YAW_RATE_DPS = max(
    0.0,
    float(os.environ.get("ROTATE_PULSE_SETTLE_MAX_YAW_RATE_DPS", "10.0")),
)
ROTATE_PULSE_TRANSITION_RPM = max(
    0,
    int(os.environ.get("ROTATE_PULSE_TRANSITION_RPM", "4")),
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
    1, int(os.environ.get("ROTATE_PULSE_ACTIVE_BRAKE_RPM", "4"))
)
ROTATE_PULSE_ACTIVE_BRAKE_SEC = max(
    0.04, float(os.environ.get("ROTATE_PULSE_ACTIVE_BRAKE_SEC", "0.08"))
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
VISIBLE_STEERING_PID_IMAGE_ERROR_ONLY = os.environ.get("VISIBLE_STEERING_PID_IMAGE_ERROR_ONLY", "0").strip() != "0"
VISIBLE_STEERING_PID_IMAGE_BRAKE_ASSIST = os.environ.get("VISIBLE_STEERING_PID_IMAGE_BRAKE_ASSIST", "0").strip() != "0"
VISIBLE_STEERING_PID_IMAGE_CAPTURE_MOTION = os.environ.get("VISIBLE_STEERING_PID_IMAGE_CAPTURE_MOTION", "0").strip() != "0"
VISIBLE_STEERING_PID_IMAGE_SLOW_BRAKE_CONTINUITY_SEC = float(os.environ.get("VISIBLE_STEERING_PID_IMAGE_SLOW_BRAKE_CONTINUITY_SEC", "0"))
VISIBLE_STEERING_PID_OUTWARD_LEAD_ENABLE = os.environ.get("VISIBLE_STEERING_PID_OUTWARD_LEAD_ENABLE", "0").lower() in ("1", "true", "yes", "on")
VISIBLE_STEERING_PID_IMAGE_MOTION_RESPONSE_SEC = float(os.environ.get("VISIBLE_STEERING_PID_IMAGE_MOTION_RESPONSE_SEC", "0.18"))
VISIBLE_STEERING_PID_EXECUTION_RESPONSE_TRIAL_SEC = float(os.environ.get("VISIBLE_STEERING_PID_EXECUTION_RESPONSE_TRIAL_SEC", "0"))
VISIBLE_STEERING_PID_IMAGE_CENTER_RELEASE_MARGIN_DEG = float(os.environ.get("VISIBLE_STEERING_PID_IMAGE_CENTER_RELEASE_MARGIN_DEG", "1.8"))
VISIBLE_STEERING_PID_FORWARD_TRACKING_ENABLE = os.environ.get("VISIBLE_STEERING_PID_FORWARD_TRACKING_ENABLE", "0").strip() != "0"
VISIBLE_STEERING_PID_CAMERA_HFOV_DEG = float(os.environ.get("VISIBLE_STEERING_PID_CAMERA_HFOV_DEG", "90.0"))
VISIBLE_STEERING_PID_CAMERA_LATENCY_SEC = float(os.environ.get("VISIBLE_STEERING_PID_CAMERA_LATENCY_SEC", "0.13"))
VISIBLE_STEERING_PID_DEADBAND_DEG = float(os.environ.get("VISIBLE_STEERING_PID_DEADBAND_DEG", "1.2"))
VISIBLE_STEERING_PID_OUTER_KP_PER_SEC = float(os.environ.get("VISIBLE_STEERING_PID_OUTER_KP_PER_SEC", "1.50"))
VISIBLE_STEERING_PID_OUTER_KD_SEC = float(os.environ.get("VISIBLE_STEERING_PID_OUTER_KD_SEC", "0.06"))
VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_GAIN = max(0.0, float(os.environ.get("VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_GAIN", "0.30")))
VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_MAX_DPS = max(0.0, float(os.environ.get("VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_MAX_DPS", "12.0")))
VISIBLE_STEERING_PID_TARGET_SPEED_MATCH_MAX_CLOSING_DPS = max(
    0.0,
    float(os.environ.get("VISIBLE_STEERING_PID_TARGET_SPEED_MATCH_MAX_CLOSING_DPS", "0.0")),
)
VISIBLE_STEERING_PID_MAX_YAW_RATE_DPS = float(os.environ.get("VISIBLE_STEERING_PID_MAX_YAW_RATE_DPS", "35.0"))
VISIBLE_STEERING_PID_RATE_KP_RPM_PER_DPS = float(os.environ.get("VISIBLE_STEERING_PID_RATE_KP_RPM_PER_DPS", "0.32"))
VISIBLE_STEERING_PID_RATE_KI_RPM_PER_DEG = float(os.environ.get("VISIBLE_STEERING_PID_RATE_KI_RPM_PER_DEG", "0.015"))
VISIBLE_STEERING_PID_INTEGRAL_LIMIT_DEG = float(os.environ.get("VISIBLE_STEERING_PID_INTEGRAL_LIMIT_DEG", "25.0"))
VISIBLE_STEERING_PID_MAX_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_MAX_CORRECTION_RPM", "10.0"))
VISIBLE_STEERING_PID_DYNAMIC_SMALL_ERROR_DEG = float(os.environ.get("VISIBLE_STEERING_PID_DYNAMIC_SMALL_ERROR_DEG", "6.0"))
VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_DEG = float(os.environ.get("VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_DEG", "18.0"))
VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_YAW_RATE_DPS = float(os.environ.get("VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_YAW_RATE_DPS", "18.0"))
VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_CORRECTION_RPM", "5.0"))
VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_BASE_CAP_RPM = float(os.environ.get("VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_BASE_CAP_RPM", "35.0"))
VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_THRESHOLD_DPS = float(os.environ.get("VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_THRESHOLD_DPS", "3.0"))
VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_BOOST_RPM = float(os.environ.get("VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_BOOST_RPM", "5.0"))
VISIBLE_STEERING_PID_BRAKING_MAX_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_BRAKING_MAX_CORRECTION_RPM", "5.0"))
VISIBLE_STEERING_PID_FAST_COUNTERSTEER_MAX_CORRECTION_RPM = float(os.environ.get("VISIBLE_STEERING_PID_FAST_COUNTERSTEER_MAX_CORRECTION_RPM", "5.0"))
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
VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_MAX_CORRECTION_RPM = max(
    0.0,
    float(os.environ.get("VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_MAX_CORRECTION_RPM", "0.0")),
)
VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_MIN_YAW_RATE_DPS = max(
    0.0,
    float(os.environ.get("VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_MIN_YAW_RATE_DPS", "4.0")),
)
VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_MIN_CORRECTION_RPM = max(
    0.0, float(os.environ.get("VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_MIN_CORRECTION_RPM", "0.0")),
)
VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_GAIN_RPM_PER_DPS = max(
    0.0, float(os.environ.get("VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_GAIN_RPM_PER_DPS", "0.10")),
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
VISIBLE_STEERING_PID_LOST_HOLD_MIN_CORRECTION_RPM = max(
    1,
    min(
        VISIBLE_STEERING_PID_LOST_HOLD_MAX_CORRECTION_RPM,
        int(os.environ.get("VISIBLE_STEERING_PID_LOST_HOLD_MIN_CORRECTION_RPM", "5")),
    ),
)
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
    ramp_diagnostics_enable=MOTOR_RAMP_DIAGNOSTICS_ENABLE,
    closed_loop_acceleration_rpm_s=MOTOR_CLOSED_LOOP_ACCELERATION_RPM_S,
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
_auto_reverse_stop_m = max(FOLLOW_BRAKE_DISTANCE_M + 0.06, TARGET_DISTANCE - 0.05)
FOLLOW_REVERSE_STOP_DISTANCE_M = max(
    FOLLOW_BRAKE_DISTANCE_M + 0.06,
    _auto_reverse_stop_m
    if FOLLOW_DISTANCE_AUTO_TUNE
    else float(os.environ.get("FOLLOW_REVERSE_STOP_DISTANCE_M", "1.45")),
)
_auto_reverse_start_m = max(FOLLOW_BRAKE_DISTANCE_M + 0.05, TARGET_DISTANCE - 0.20)
FOLLOW_REVERSE_START_DISTANCE_M = max(
    FOLLOW_BRAKE_DISTANCE_M + 0.05,
    min(
        FOLLOW_REVERSE_STOP_DISTANCE_M - 0.01,
        _auto_reverse_start_m
        if FOLLOW_DISTANCE_AUTO_TUNE
        else float(os.environ.get("FOLLOW_REVERSE_START_DISTANCE_M", str(TARGET_DISTANCE))),
    ),
)
_auto_reverse_immediate_m = _auto_reverse_start_m
FOLLOW_REVERSE_IMMEDIATE_DISTANCE_M = max(
    FOLLOW_BRAKE_DISTANCE_M + 0.05,
    min(
        FOLLOW_REVERSE_START_DISTANCE_M,
        _auto_reverse_immediate_m
        if FOLLOW_DISTANCE_AUTO_TUNE
        else float(os.environ.get("FOLLOW_REVERSE_IMMEDIATE_DISTANCE_M", "1.35")),
    ),
)
_auto_reverse_full_speed_m = max(FOLLOW_BRAKE_DISTANCE_M + 0.05, TARGET_DISTANCE - 0.50)
FOLLOW_REVERSE_FULL_SPEED_DISTANCE_M = max(
    FOLLOW_BRAKE_DISTANCE_M + 0.05,
    min(
        FOLLOW_REVERSE_START_DISTANCE_M - 0.01,
        _auto_reverse_full_speed_m
        if FOLLOW_DISTANCE_AUTO_TUNE
        else float(os.environ.get("FOLLOW_REVERSE_FULL_SPEED_DISTANCE_M", "1.0")),
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
_auto_reverse_visual_guard_m = max(FOLLOW_REVERSE_STOP_DISTANCE_M, TARGET_DISTANCE + 0.10)
FOLLOW_REVERSE_VISUAL_GUARD_MAX_DISTANCE_M = max(
    FOLLOW_REVERSE_STOP_DISTANCE_M,
    _auto_reverse_visual_guard_m
    if FOLLOW_DISTANCE_AUTO_TUNE
    else float(os.environ.get("FOLLOW_REVERSE_VISUAL_GUARD_MAX_DISTANCE_M", "1.60")),
)
FOLLOW_REVERSE_VISUAL_GUARD_RPM = max(
    1,
    min(
        FOLLOW_REVERSE_RUNTIME_CAP_RPM,
        int(os.environ.get("FOLLOW_REVERSE_VISUAL_GUARD_RPM", "30")),
    ),
)
_auto_forward_start_m = TARGET_DISTANCE + 0.08
FOLLOW_FORWARD_START_DISTANCE_M = max(
    TARGET_DISTANCE + 0.01,
    _auto_forward_start_m
    if FOLLOW_DISTANCE_AUTO_TUNE
    else float(os.environ.get("FOLLOW_FORWARD_START_DISTANCE_M", "1.80")),
)
_auto_forward_stop_m = TARGET_DISTANCE + 0.03
FOLLOW_FORWARD_STOP_DISTANCE_M = max(
    TARGET_DISTANCE,
    min(
        FOLLOW_FORWARD_START_DISTANCE_M - 0.01,
        _auto_forward_stop_m
        if FOLLOW_DISTANCE_AUTO_TUNE
        else float(os.environ.get("FOLLOW_FORWARD_STOP_DISTANCE_M", "1.65")),
    ),
)
FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_ENABLE = (
    os.environ.get("FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_ENABLE", "1").strip() != "0"
)
FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_DISTANCE_M = max(
    TARGET_DISTANCE + 0.01,
    _auto_forward_stop_m
    if FOLLOW_DISTANCE_AUTO_TUNE
    else float(os.environ.get("FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_DISTANCE_M", "1.80")),
)
FOLLOW_NEAR_DISTANCE_ROTATION_ONLY_MAX_RPM = max(
    1,
    int(os.environ.get("FOLLOW_NEAR_DISTANCE_ROTATION_ONLY_MAX_RPM", "10")),
)
FOLLOW_NEAR_DISTANCE_SETTLE_CONFIRM_FRAMES = max(
    1,
    int(os.environ.get("FOLLOW_NEAR_DISTANCE_SETTLE_CONFIRM_FRAMES", "2")),
)
FOLLOW_NEAR_DISTANCE_SETTLE_HOLD_SEC = max(
    0.05,
    float(os.environ.get("FOLLOW_NEAR_DISTANCE_SETTLE_HOLD_SEC", "0.35")),
)
FOLLOW_NEAR_DISTANCE_SETTLE_RELEASE_MARGIN_RATIO = max(
    0.01,
    float(os.environ.get("FOLLOW_NEAR_DISTANCE_SETTLE_RELEASE_MARGIN_RATIO", "0.03")),
)
FOLLOW_NEAR_DISTANCE_SETTLE_RELEASE_FRAMES = max(
    1,
    int(os.environ.get("FOLLOW_NEAR_DISTANCE_SETTLE_RELEASE_FRAMES", "2")),
)
FOLLOW_NEAR_DISTANCE_DISABLE_RATE_FEEDFORWARD = (
    os.environ.get("FOLLOW_NEAR_DISTANCE_DISABLE_RATE_FEEDFORWARD", "1").strip() != "0"
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
    0.03 if FOLLOW_DISTANCE_AUTO_TUNE else float(os.environ.get("DISTANCE_PID_DEADBAND_M", "0.005")),
)
DISTANCE_PID_FORWARD_INTEGRAL_LIMIT_M_S = max(
    0.0, float(os.environ.get("DISTANCE_PID_FORWARD_INTEGRAL_LIMIT_M_S", "0")),
)
DISTANCE_FEEDFORWARD_ENABLE = os.environ.get("DISTANCE_FEEDFORWARD_ENABLE", "0").strip() != "0"
DISTANCE_APPROACH_ENABLE = os.environ.get("DISTANCE_APPROACH_ENABLE", "0").strip() != "0"
DISTANCE_CONTROL_MODE = (os.environ.get("FOLLOW_DISTANCE_CONTROL_MODE", "").strip()
                         or os.environ.get("DISTANCE_CONTROL_MODE", "").strip()
                         or ("approach" if DISTANCE_APPROACH_ENABLE else "legacy"))
if DISTANCE_CONTROL_MODE not in {"distance_pi", "approach", "legacy"}:
    raise ValueError("FOLLOW_DISTANCE_CONTROL_MODE / DISTANCE_CONTROL_MODE must be distance_pi, approach or legacy")
# Explicit legacy must also disable an INI's retained approach rollback branch.
DISTANCE_APPROACH_ENABLE = DISTANCE_CONTROL_MODE == "approach"
_target_motion_setting = os.environ.get("DISTANCE_TARGET_MOTION_CONTROL_ENABLE",
    "0" if DISTANCE_CONTROL_MODE == "distance_pi" else "1").strip().lower()
if _target_motion_setting not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
    raise ValueError("DISTANCE_TARGET_MOTION_CONTROL_ENABLE must be a boolean")
DISTANCE_TARGET_MOTION_CONTROL_ENABLE = _target_motion_setting in {"1", "true", "yes", "on"}
if not DISTANCE_TARGET_MOTION_CONTROL_ENABLE and DISTANCE_CONTROL_MODE != "distance_pi":
    raise ValueError(
        "DISTANCE_TARGET_MOTION_CONTROL_ENABLE=false requires distance_pi; "
        "set DISTANCE_TARGET_MOTION_CONTROL_ENABLE=1 explicitly for approach/legacy rollback")
DISTANCE_PI_KP_PER_SEC = float(os.environ.get("DISTANCE_PI_KP_PER_SEC", "1.0"))
DISTANCE_PI_KI_PER_SEC2 = float(os.environ.get("DISTANCE_PI_KI_PER_SEC2", "0.4"))
DISTANCE_PI_INTEGRAL_MAX_M_S = float(os.environ.get("DISTANCE_PI_INTEGRAL_MAX_M_S", "0.8"))
DISTANCE_PI_MEMORY_SEC = float(os.environ.get("DISTANCE_PI_MEMORY_SEC", "0.35"))
DISTANCE_PI_MOTION_MEMORY_SEC = float(os.environ.get("DISTANCE_PI_MOTION_MEMORY_SEC", "0"))
if not math.isfinite(DISTANCE_PI_MOTION_MEMORY_SEC) or not 0 <= DISTANCE_PI_MOTION_MEMORY_SEC <= .35:
    raise ValueError("DISTANCE_PI_MOTION_MEMORY_SEC must be finite and within 0..0.35")
DISTANCE_PI_LAUNCH_REQUEST_RPM = float(os.environ.get("DISTANCE_PI_LAUNCH_REQUEST_RPM", "0"))
if not math.isfinite(DISTANCE_PI_LAUNCH_REQUEST_RPM) or not 0 <= DISTANCE_PI_LAUNCH_REQUEST_RPM <= 200:
    raise ValueError("DISTANCE_PI_LAUNCH_REQUEST_RPM must be finite and within 0..200")
DISTANCE_PI_LAUNCH_FULL_ERROR_M = float(os.environ.get("DISTANCE_PI_LAUNCH_FULL_ERROR_M", "0"))
if not math.isfinite(DISTANCE_PI_LAUNCH_FULL_ERROR_M) or not 0 <= DISTANCE_PI_LAUNCH_FULL_ERROR_M <= 2:
    raise ValueError("DISTANCE_PI_LAUNCH_FULL_ERROR_M must be finite and within 0..2")
_pi_stop_preview_setting = os.environ.get(
    "DISTANCE_PI_STATIONARY_STOP_PREVIEW_ENABLED", "0").strip().lower()
if _pi_stop_preview_setting not in {"0", "1", "false", "true", "off", "on"}:
    raise ValueError("DISTANCE_PI_STATIONARY_STOP_PREVIEW_ENABLED must be a boolean")
DISTANCE_PI_STATIONARY_STOP_PREVIEW_ENABLED = (
    _pi_stop_preview_setting in {"1", "true", "on"})
_feedback_reserve_setting = os.environ.get("DISTANCE_PI_OBSERVED_FEEDBACK_RESERVE", "false").strip().lower()
if _feedback_reserve_setting not in {"0", "1", "false", "true", "off", "on", "yes", "no"}:
    raise ValueError("DISTANCE_PI_OBSERVED_FEEDBACK_RESERVE must be a boolean")
DISTANCE_PI_OBSERVED_FEEDBACK_RESERVE = _feedback_reserve_setting in {"1", "true", "on", "yes"}
# Remove overlapping feedback age only with complete physical-interval history.
# Keep >=50ms extra reserve and the selected 1.1m/.15s/1m/s² braking profile;
# this neither extends an old authorization nor certifies stopping capability.
_feedback_interval_deduplication_setting = os.environ.get(
    "DISTANCE_PI_FEEDBACK_INTERVAL_DEDUPLICATION", "false").strip().lower()
if _feedback_interval_deduplication_setting not in {"0", "1", "false", "true", "off", "on", "yes", "no"}:
    raise ValueError("DISTANCE_PI_FEEDBACK_INTERVAL_DEDUPLICATION must be a boolean")
DISTANCE_PI_FEEDBACK_INTERVAL_DEDUPLICATION = (
    _feedback_interval_deduplication_setting in {"1", "true", "on", "yes"})
VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC = float(os.environ.get("VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC", "0.50"))
if (not math.isfinite(VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC)
        or not .05 <= VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC <= MAX_VALIDATED_VISIBILITY_AGE_SEC):
    raise ValueError("VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC must be within 0.05..0.50")
# A forward stopping-model boundary is not the reverse-start threshold.
# Keep legacy callers unchanged unless the runtime profile selects a trial.
DISTANCE_PI_BRAKING_STOP_DISTANCE_M = float(os.environ.get(
    "DISTANCE_PI_BRAKING_STOP_DISTANCE_M",
    str(max(FOLLOW_REVERSE_START_DISTANCE_M, FOLLOW_BRAKE_DISTANCE_M+.05,
            TARGET_DISTANCE-.2))))
if (not math.isfinite(DISTANCE_PI_BRAKING_STOP_DISTANCE_M)
        or not FOLLOW_BRAKE_DISTANCE_M+.05 <= DISTANCE_PI_BRAKING_STOP_DISTANCE_M <= TARGET_DISTANCE):
    raise ValueError("DISTANCE_PI_BRAKING_STOP_DISTANCE_M must be between brake distance + 0.05 and target distance")
for _pi_parameter_name in ("DISTANCE_PI_KP_PER_SEC", "DISTANCE_PI_KI_PER_SEC2",
                           "DISTANCE_PI_INTEGRAL_MAX_M_S", "DISTANCE_PI_MEMORY_SEC"):
    if not math.isfinite(globals()[_pi_parameter_name]) or globals()[_pi_parameter_name] < 0:
        raise ValueError(f"{_pi_parameter_name} must be finite and nonnegative")
if (LOADED_CONFIG is None and DISTANCE_CONTROL_MODE == "distance_pi"
        and os.environ.get("FOLLOW_DISTANCE_P_TRIAL", "").strip()):
    raise ValueError("FOLLOW_DISTANCE_P_TRIAL cannot tune distance_pi; use FOLLOW_DISTANCE_CONTROL_MODE=legacy")
DISTANCE_APPROACH_MATCHING_ENABLE = os.environ.get("DISTANCE_APPROACH_MATCHING_ENABLE", "1").strip() != "0"
DISTANCE_APPROACH_GAIN_PER_SEC = float(os.environ.get("DISTANCE_APPROACH_GAIN_PER_SEC", "1.0"))
DISTANCE_APPROACH_MAX_CATCHUP_M_S = float(os.environ.get("DISTANCE_APPROACH_MAX_CATCHUP_M_S", "0.60"))
DISTANCE_APPROACH_DECELERATION_M_S2 = float(os.environ.get("DISTANCE_APPROACH_DECELERATION_M_S2", "0.40"))
DISTANCE_APPROACH_RESPONSE_DELAY_SEC = float(os.environ.get("DISTANCE_APPROACH_RESPONSE_DELAY_SEC", "0.20"))
DISTANCE_APPROACH_NO_MATCHING_MAX_RPM = float(os.environ.get("DISTANCE_APPROACH_NO_MATCHING_MAX_RPM", "0"))
DISTANCE_TURN_COMPENSATION_ENABLE = os.environ.get("DISTANCE_TURN_COMPENSATION_ENABLE", "0").strip() != "0"
DEPTH_MEASURED_RECOVERY_ENABLE = os.environ.get("DEPTH_MEASURED_RECOVERY_ENABLE", "0").strip() != "0"
DISTANCE_FEEDFORWARD_MAX_RPM = max(0.0, min(20.0, float(os.environ.get("DISTANCE_FEEDFORWARD_MAX_RPM", "20"))))
DISTANCE_MATCHING_BASE_MAX_RPM = max(0.0, min(100.0, float(os.environ.get("DISTANCE_MATCHING_BASE_MAX_RPM", "0"))))
DISTANCE_MATCHING_TEST_BIAS_RPM = float(os.environ.get("DISTANCE_MATCHING_TEST_BIAS_RPM", "0"))
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
    follow_wheel_period_sec=float(os.environ.get("FOLLOW_WHEEL_PERIOD_SEC", "0")),
    follow_forward_handoff_enable=os.environ.get("FOLLOW_FORWARD_HANDOFF_ENABLE", "0") == "1",
    follow_residual_reverse_max_rpm=float(os.environ.get("FOLLOW_RESIDUAL_REVERSE_MAX_RPM", "0")),
    follow_forward_loss_handoff_enable=os.environ.get("FOLLOW_FORWARD_LOSS_HANDOFF_ENABLE", "0") == "1",
    follow_turn_residual_max_rpm=float(os.environ.get("FOLLOW_TURN_RESIDUAL_MAX_RPM", "0")),
    follow_turn_acceleration_priority_enable=os.environ.get("FOLLOW_TURN_ACCELERATION_PRIORITY_ENABLE", "0") == "1",
    follow_turn_response_assist_enable=os.environ.get("FOLLOW_TURN_RESPONSE_ASSIST_ENABLE", "0") == "1",
    follow_cross_brake_enable=os.environ.get("FOLLOW_CROSS_BRAKE_ENABLE", "0") == "1",
    follow_cross_brake_mode=os.environ.get("FOLLOW_CROSS_BRAKE_MODE", "zero").strip().lower(),
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
        self._capture_started_ts = 0.0
        self._capture_last_frame_ts = 0.0
        self._capture_failure_reason = None
        self._capture_recorder_disabled = False
        self._capture_direction_disabled = False
        self._direction_result_backlog = []
        # DirectionEvidence is tiny (bbox/score/timestamp), so retain a short
        # capture-ordered ring instead of copying full 1080p frames into RAM.
        # The raw recorder already owns the complete RGB stream for any later
        # stateless ReID extension.
        self._direction_evidence_ring = deque(maxlen=30)
        self._capture_metadata_ring = deque(maxlen=30)
        self._historical_backfill = HistoricalDirectionBackfill(
            max_age_sec=HISTORICAL_DIRECTION_BACKFILL_MAX_AGE_SEC,
            min_samples=HISTORICAL_DIRECTION_BACKFILL_MIN_SAMPLES,
            max_capture_gap=HISTORICAL_DIRECTION_BACKFILL_MAX_CAPTURE_GAP,
            max_center_jump_ratio=HISTORICAL_DIRECTION_BACKFILL_MAX_CENTER_JUMP_RATIO,
            min_area_similarity=HISTORICAL_DIRECTION_BACKFILL_MIN_AREA_SIMILARITY,
            confidence_cap=HISTORICAL_DIRECTION_BACKFILL_CONFIDENCE_CAP,
            min_score=HISTORICAL_DIRECTION_BACKFILL_MIN_SCORE,
        )
        self._historical_backfill_pending = None
        self._historical_backfill_episode = 0
        self._historical_backfill_last_start_capture = -1
        self._direction_evidence_merged_total = 0
        self._direction_evidence_last_log_ts = 0.0
        self._camera_video_recorder = None
        self._video_follow_snapshot = None
        if RKNN_CAMERA_RAW_OUTPUT and RKNN_CAMERA_CAPTURE_MODE in {"opencv", "opencv_v4l2", "v4l2"}:
            if cv2 is None:
                raise RuntimeError(f"OpenCV(cv2) import failed: {_CV2_IMPORT_ERROR}")
            self._camera_video_recorder = AsyncVideoRecorder(
                VideoRecorderConfig(
                    output_path=RKNN_CAMERA_RAW_OUTPUT,
                    fps=RKNN_CAMERA_FPS,
                    fourcc="MJPG",
                    queue_capacity=max(1, min(60, int(os.environ.get("FOLLOW_VIDEO_QUEUE_CAPACITY", "12")))),
                    worker_nice=max(0, min(19, int(os.environ.get("FOLLOW_VIDEO_WORKER_NICE", "5")))),
                    fast_mjpeg=os.environ.get("FOLLOW_VIDEO_FAST_MJPEG", "1").lower()
                        not in {"0", "false", "off", "no"},
                ),
                cv2_module=cv2,
                logger=logger,
                depth_sample_provider=self._recording_depth_sample,
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
            "candidate_acquire=%drpm aimline_approach<=%.3f/%drpm "
            "aimline_intersection=active_brake "
            "identity_claim=False direction=geometry_only "
            "edge_probe_skip_below=%.2f edge_center_margin=%.2f "
            "unmatched_probe_observation_min_score=0.15",
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
            PARKED_RECENTER_MIN_RPM,
            SEARCH_EVIDENCE_PROBE_EDGE_MIN_SCORE,
            SEARCH_EVIDENCE_PROBE_EDGE_CENTER_RATIO,
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
            "Distance control: mode=%s pi_kp=%.3f/s pi_ki=%.3f/s2 pi_integral_max=%.3fm/s "
            "pi_memory=%.3fs target_motion_control=%s feedforward_diagnostic_only=%s "
            "rotation_only=%s launch_request_rpm=%.1f "
            "launch_full_error_m=%.3f",
            DISTANCE_CONTROL_MODE, DISTANCE_PI_KP_PER_SEC, DISTANCE_PI_KI_PER_SEC2,
            DISTANCE_PI_INTEGRAL_MAX_M_S, DISTANCE_PI_MEMORY_SEC,
            DISTANCE_TARGET_MOTION_CONTROL_ENABLE,
            DISTANCE_TARGET_MOTION_CONTROL_ENABLE and DISTANCE_CONTROL_MODE == "distance_pi",
            FOLLOW_ROTATION_ONLY,
            DISTANCE_PI_LAUNCH_REQUEST_RPM,
            DISTANCE_PI_LAUNCH_FULL_ERROR_M,
        )
        logger.info(
            "Distance PID: enabled=%s kp=%.2f ki=%.2f kd=%.2f deadband=%.3fm "
            "d_filter_alpha=%.2f jump_guard=%.2fm rise=%.1fRPM/s fall=%.1fRPM/s "
            "output=%d..%dRPM",
            DISTANCE_PID_ENABLE,
            DISTANCE_PID_KP_RPM_PER_M,
            DISTANCE_PID_KI_RPM_PER_M_S,
            DISTANCE_PID_KD_RPM_S_PER_M if DISTANCE_TARGET_MOTION_CONTROL_ENABLE else 0.,
            DISTANCE_PID_DEADBAND_M,
            DISTANCE_PID_DERIVATIVE_FILTER_ALPHA,
            DISTANCE_PID_MAX_MEASUREMENT_JUMP_M,
            DISTANCE_PID_OUTPUT_RISE_RPM_PER_SEC,
            DISTANCE_PID_OUTPUT_FALL_RPM_PER_SEC,
            FORWARD_MIN_RPM,
            FORWARD_MAX_RPM,
        )
        logger.info(
            "Longitudinal tracking profile: target=%.2fm start=%.2fm stop=%.2fm "
            "stationary_rotate_only<=%.2fm feedforward=%s extra<=%.1fRPM "
            "wheel_circumference=%.3fm physical_depth_dedupe=True matching_base_max_rpm=%.1f "
            "matching_rise_rpm_s=80 matching_fall_rpm_s=80",
            TARGET_DISTANCE, FOLLOW_FORWARD_START_DISTANCE_M, FOLLOW_FORWARD_STOP_DISTANCE_M,
            FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_DISTANCE_M,
            DISTANCE_TARGET_MOTION_CONTROL_ENABLE and DISTANCE_FEEDFORWARD_ENABLE,
            DISTANCE_FEEDFORWARD_MAX_RPM if DISTANCE_TARGET_MOTION_CONTROL_ENABLE else 0.,
            VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M,
            DISTANCE_MATCHING_BASE_MAX_RPM if DISTANCE_TARGET_MOTION_CONTROL_ENABLE else 0.,
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
                "Astra Depth config: %dx%d@%dFPS range=[%.2f,%.2f]m frame_age<=%.2fs hold=%.2fs RGB_align_delay=%.0fms median=%d jump<=%.2fm/%dframes rate<=%.1fm/s",
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
                ASTRA_DEPTH_MAX_UNCONFIRMED_JUMP_RATE_M_S,
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
                "fresh_far_jump<=%.2fm confirm=%dframes depth30_visual_dedupe=%.0fms "
                "depth30_dedupe=physical_timestamp",
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
            "ReID appearance fusion: osnet=true color_hsv=%s weight=%.2f partial_torso=%s threshold=%.2f search_conf_floor=%.2f",
            cfg.reid_color_fusion_enable,
            cfg.reid_color_fusion_weight,
            cfg.identity_partial_appearance_enable,
            cfg.identity_partial_match_threshold,
            cfg.identity_preferred_search_reacquire_min_confidence,
        )
        self._rknn_pipeline = RKNNVisionPipeline(cfg, logger=logger)
        from rk_vision.numeric_runtime import log_numeric_runtime
        log_numeric_runtime(logger, "vision_initialized")
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
        self._confirmed_search_reacquire_depth_last_frame = -1
        self._confirmed_search_reacquire_depth_streak = 0
        # A visually confirmed reacquisition gets a short grace window.  It
        # keeps the UID and latest bbox through one clipped/blurred frame while
        # Depth reacquires its range; it is never used to authorize motion by
        # itself.
        self._visual_reacquire_hold_uid = None
        self._visual_reacquire_hold_bbox = None
        self._visual_reacquire_hold_started_at = 0.0
        self._visual_reacquire_hold_until = 0.0
        self._reacquire_depth_pending = False
        self._reacquire_depth_pending_uid = None
        self._search_candidate_gate = SearchCandidateGate(
            SearchCandidateGateConfig(
                enabled=SEARCH_EVIDENCE_GATE_ENABLE,
                formal_min_score=CONFIDENCE_THRESHOLD,
                probe_min_score=SEARCH_EVIDENCE_PROBE_MIN_SCORE,
                min_area_ratio=SEARCH_EVIDENCE_MIN_AREA_RATIO,
                max_area_ratio=SEARCH_EVIDENCE_MAX_AREA_RATIO,
                probe_confirm_frames=SEARCH_EVIDENCE_PROBE_CONFIRM_FRAMES,
                probe_edge_min_score=SEARCH_EVIDENCE_PROBE_EDGE_MIN_SCORE,
                # CAP900/901's 0.140/0.111 central background probe must not
                # stop search. Diagnostics retain it; current UID evidence
                # and formal detections keep their bounded observation path.
                probe_observation_min_score=0.15,
                probe_edge_center_ratio=SEARCH_EVIDENCE_PROBE_EDGE_CENTER_RATIO,
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
        self._search_observation_retry = SearchObservationRetry(SEARCH_EVIDENCE_MAX_HOLD_SEC)
        self._search_retry_zero_requested_at = 0.0
        self._search_retry_zero_sent_at = None
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
        self._current_forward_allow_below_min = False
        self._forward_speed_latched_percent: Optional[int] = None
        self._last_forward_speed_hysteresis_log_ts = 0.0
        self._current_steer_base_percent = 0
        self._current_steer_inner_ratio_percent = VISIBLE_STEER_INNER_RATIO_PERCENT
        self._current_steer_outer_ratio_percent = VISIBLE_STEER_OUTER_RATIO_PERCENT
        self._current_steer_correction_rpm = 0
        self._current_steer_limit_reason = "none"
        self._vision_control_state = "lost_confirming"
        # Last *formally confirmed* visual owner of an existing drive grant.
        # A later UID0 observation may read this provenance for a short
        # recheck, but cannot move or refresh its capture/depth deadlines.
        self._confirmed_identity_execution_anchor = None
        # Only the vision loop updates this signed yaw correction. The 30Hz
        # Depth loop may reuse it briefly while changing longitudinal speed.
        self._last_vision_correction_rpm = 0
        self._last_vision_correction_at = 0.0
        self._last_vision_correction_target_id: Optional[int] = None
        self._last_vision_control_frame_index = -1
        self._last_vision_control_ts = 0.0
        publish_visual_identity_evidence(self, observation=None, lease=None)
        self._last_depth30_dedupe_log_frame = -1
        self._last_depth30_preserve_yaw_log_ts = 0.0
        # Timestamp of the most recent Depth30 translational command.  A
        # single delayed visual miss must not replace that fresh command with
        # a lost-history rotation; once the bounded window expires, the
        # normal visual/search path regains control.
        self._last_depth30_translation_ts = 0.0
        self._last_depth30_translation_kind = None
        self._depth30_linear_snapshot = None
        self._depth30_linear_sample_watermark = None
        self._depth30_linear_timing = None
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
        self._runtime_shutdown_reason = None
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
        self._lateral_intent_zero_sequence = -1
        self._lateral_yaw_revision = 0
        self._lateral_intent_owned_frame = -1
        self._last_lateral_yaw_sign_mismatch_log_ts = 0.0
        self._longitudinal_context_lock = threading.Lock()
        self._longitudinal_context = None
        self._depth_async_scheduler = DepthAsyncScheduler()
        # Read-only admission hint for early ROI publication, NOT safety or
        # motion authority. Normal SensorFrame handling refreshes it; Depth
        # and the final writer still run current safety checks independently.
        self._depth_roi_safety_clear = False
        self._longitudinal_stop_event = threading.Event()
        self._longitudinal_wake_event = threading.Event()
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
        self._near_yaw_park_request = None
        self._near_yaw_park_evidence = (0, 0.0)
        self._near_yaw_park_generation = 0
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
                historical_direction_backfill_enable=HISTORICAL_DIRECTION_BACKFILL_ENABLE,
                historical_direction_backfill_max_age_sec=HISTORICAL_DIRECTION_BACKFILL_MAX_AGE_SEC,
                historical_direction_backfill_min_samples=HISTORICAL_DIRECTION_BACKFILL_MIN_SAMPLES,
                historical_direction_backfill_max_capture_gap=HISTORICAL_DIRECTION_BACKFILL_MAX_CAPTURE_GAP,
                historical_direction_backfill_max_center_jump_ratio=HISTORICAL_DIRECTION_BACKFILL_MAX_CENTER_JUMP_RATIO,
                historical_direction_backfill_min_area_similarity=HISTORICAL_DIRECTION_BACKFILL_MIN_AREA_SIMILARITY,
                historical_direction_backfill_confidence_cap=HISTORICAL_DIRECTION_BACKFILL_CONFIDENCE_CAP,
                historical_direction_backfill_min_score=HISTORICAL_DIRECTION_BACKFILL_MIN_SCORE,
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
                near_distance_settle_confirm_frames=FOLLOW_NEAR_DISTANCE_SETTLE_CONFIRM_FRAMES,
                near_distance_settle_hold_sec=FOLLOW_NEAR_DISTANCE_SETTLE_HOLD_SEC,
                near_distance_settle_release_margin_ratio=FOLLOW_NEAR_DISTANCE_SETTLE_RELEASE_MARGIN_RATIO,
                near_distance_settle_release_frames=FOLLOW_NEAR_DISTANCE_SETTLE_RELEASE_FRAMES,
                near_distance_disable_rate_feedforward=FOLLOW_NEAR_DISTANCE_DISABLE_RATE_FEEDFORWARD,
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
                depth_longitudinal_sample_max_age_sec=ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC,
                distance_control_mode=DISTANCE_CONTROL_MODE if not FOLLOW_ROTATION_ONLY else "legacy",
                distance_target_motion_control_enable=DISTANCE_TARGET_MOTION_CONTROL_ENABLE,
                distance_pi_kp_per_sec=DISTANCE_PI_KP_PER_SEC,
                distance_pi_ki_per_sec2=DISTANCE_PI_KI_PER_SEC2,
                distance_pi_integral_max_m_s=DISTANCE_PI_INTEGRAL_MAX_M_S,
                distance_pi_memory_sec=DISTANCE_PI_MEMORY_SEC,
                distance_pi_motion_memory_sec=DISTANCE_PI_MOTION_MEMORY_SEC,
                distance_pi_launch_request_rpm=DISTANCE_PI_LAUNCH_REQUEST_RPM,
                distance_pi_launch_full_error_m=DISTANCE_PI_LAUNCH_FULL_ERROR_M,
                distance_pi_stationary_stop_preview_enabled=(
                    DISTANCE_PI_STATIONARY_STOP_PREVIEW_ENABLED),
                distance_pi_braking_stop_distance_m=DISTANCE_PI_BRAKING_STOP_DISTANCE_M,
                distance_pi_observed_feedback_reserve=DISTANCE_PI_OBSERVED_FEEDBACK_RESERVE,
                distance_pi_feedback_interval_deduplication=DISTANCE_PI_FEEDBACK_INTERVAL_DEDUPLICATION,
                distance_approach_enable=DISTANCE_APPROACH_ENABLE and not FOLLOW_ROTATION_ONLY,
                distance_approach_matching_enable=DISTANCE_APPROACH_MATCHING_ENABLE,
                distance_approach_gain_per_sec=DISTANCE_APPROACH_GAIN_PER_SEC,
                distance_approach_max_catchup_m_s=DISTANCE_APPROACH_MAX_CATCHUP_M_S,
                distance_approach_deceleration_m_s2=DISTANCE_APPROACH_DECELERATION_M_S2,
                distance_approach_response_delay_sec=DISTANCE_APPROACH_RESPONSE_DELAY_SEC,
                distance_approach_no_matching_max_rpm=DISTANCE_APPROACH_NO_MATCHING_MAX_RPM,
                distance_pid_kp_rpm_per_m=DISTANCE_PID_KP_RPM_PER_M,
                distance_pid_ki_rpm_per_m_s=DISTANCE_PID_KI_RPM_PER_M_S,
                distance_pid_kd_rpm_s_per_m=DISTANCE_PID_KD_RPM_S_PER_M,
                distance_pid_integral_limit_m_s=DISTANCE_PID_INTEGRAL_LIMIT_M_S,
                distance_pid_forward_integral_limit_m_s=DISTANCE_PID_FORWARD_INTEGRAL_LIMIT_M_S,
                distance_pid_deadband_m=DISTANCE_PID_DEADBAND_M,
                distance_feedforward_enable=DISTANCE_FEEDFORWARD_ENABLE,
                distance_turn_compensation_enable=DISTANCE_TURN_COMPENSATION_ENABLE and not FOLLOW_ROTATION_ONLY,
                depth_measured_recovery_enable=DEPTH_MEASURED_RECOVERY_ENABLE and not FOLLOW_ROTATION_ONLY,
                distance_feedforward_max_rpm=DISTANCE_FEEDFORWARD_MAX_RPM,
                distance_matching_base_max_rpm=DISTANCE_MATCHING_BASE_MAX_RPM,
                distance_matching_test_bias_rpm=DISTANCE_MATCHING_TEST_BIAS_RPM,
                distance_feedforward_wheel_circumference_m=VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M,
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
                initial_target_max_age_sec=VISION_CONTROL_MAX_RESULT_AGE_SEC,
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
                visible_steering_pid_forward_tracking_enable=VISIBLE_STEERING_PID_FORWARD_TRACKING_ENABLE,
                visible_steering_pid_image_error_only=VISIBLE_STEERING_PID_IMAGE_ERROR_ONLY,
                visible_steering_pid_image_brake_assist=VISIBLE_STEERING_PID_IMAGE_BRAKE_ASSIST,
                visible_steering_pid_image_capture_motion=VISIBLE_STEERING_PID_IMAGE_CAPTURE_MOTION,
                visible_steering_pid_image_motion_response_sec=VISIBLE_STEERING_PID_IMAGE_MOTION_RESPONSE_SEC,
                visible_steering_pid_execution_response_trial_sec=VISIBLE_STEERING_PID_EXECUTION_RESPONSE_TRIAL_SEC,
                visible_steering_pid_image_slow_brake_continuity_sec=VISIBLE_STEERING_PID_IMAGE_SLOW_BRAKE_CONTINUITY_SEC,
                visible_steering_pid_outward_lead_enable=VISIBLE_STEERING_PID_OUTWARD_LEAD_ENABLE,
                visible_steering_pid_image_center_release_margin_deg=VISIBLE_STEERING_PID_IMAGE_CENTER_RELEASE_MARGIN_DEG,
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
                visible_steering_pid_predictive_countersteer_max_correction_rpm=VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_MAX_CORRECTION_RPM,
                visible_steering_pid_predictive_countersteer_min_correction_rpm=VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_MIN_CORRECTION_RPM,
                visible_steering_pid_predictive_countersteer_gain_rpm_per_dps=VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_GAIN_RPM_PER_DPS,
                visible_steering_pid_predictive_countersteer_min_yaw_rate_dps=VISIBLE_STEERING_PID_PREDICTIVE_COUNTERSTEER_MIN_YAW_RATE_DPS,
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
        # Read-only recovery check under the existing control owner. No serial
        # I/O, new lock, timer or independent motor lease is introduced.
        self._follow_controller._live_longitudinal_authority_reader = (
            lambda uid: PersonTracker._fresh_depth_linear_snapshot(self, uid)
        )
        self._follow_controller._longitudinal_restart_identity_reader = (
            lambda uid, sample_ts, now: PersonTracker._longitudinal_restart_identity_live(
                self, uid, sample_ts, now)
        )
        self._follow_controller._longitudinal_execution_reader = (
            lambda uid, stamp, now: self._action_runtime.forward_execution_anchor(uid, stamp, now)
        )
        self._follow_controller._recent_longitudinal_execution_reader = (
            lambda uid, now: self._action_runtime.recent_forward_execution_anchor(uid, now)
        )
        self._follow_controller._longitudinal_recovery_reader = (
            lambda uid, now: self._action_runtime.recovery_forward_execution_anchor(uid, now)
        )
        self._follow_controller._longitudinal_recovery_validator = (
            lambda uid, proof, now: self._action_runtime.forward_recovery_anchor_valid(uid, proof, now)
        )
        self._follow_controller._braking_execution_bound_reader = (
            (lambda uid, now: self._action_runtime.continuation_executed_speed_bound_rpm(uid, now))
            if (ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE
                or not DISTANCE_TARGET_MOTION_CONTROL_ENABLE) else None
        )
        self._follow_controller._braking_interval_speed_bound_reader = (
            (lambda uid, sample_timestamp, now:
                self._action_runtime.braking_interval_speed_bound_rpm(uid, sample_timestamp, now))
            if not DISTANCE_TARGET_MOTION_CONTROL_ENABLE else None
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
            follow_axes_reader=lambda now: self._follow_wheel_axes(now, quiet=True),
            depth_linear_reader=lambda uid, *, now=None: self._fresh_depth_linear_snapshot(
                uid, now=now, quiet=True),
            logger=logger,
        )
        # 电机串口和启动驻车必须先初始化完成。动作/编码器线程在 run()
        # 中随后启动，避免反馈读取与驻车寄存器写入同时访问 /dev/ttyS0。
        self._action_runtime_started = False
        short_config = ShortFollowConfig.from_env()
        self._short_follow = ShortFollowController(short_config)
        self._short_follow_adapter = (
            ShortFollowAdapter(self, self._short_follow, logger)
            if short_config.enabled and not FOLLOW_ROTATION_ONLY else None
        )
        self._short_follow_handled_frame = False
        logger.info(
            "normal_follow_config mode=%s paired_longitudinal=distance_pi max_rpm=%d "
            "kp_per_sec=%.3f ki_per_sec2=%.3f integral_max_m_s=%.3f "
            "stop_distance_m=%.2f restart_distance_m=%.2f braking_stop_distance_m=%.2f "
            "depth_ttl_ms=%.0f visual_ttl_ms=%.0f writer_period_ms=%.0f "
            "yaw_max_delta_rpm=%d yaw_full_error_ratio=%.3f pivot_max_rpm=%d "
            "legacy_pi_yaw_authority=%s single_complete_wheel_plan=%s "
            "paired_launch_boost=False paired_target_speed=False",
            "paired" if self._short_follow_adapter is not None else "legacy",
            short_config.max_rpm, short_config.kp_per_sec, short_config.ki_per_sec2,
            short_config.integral_max_m_s,
            short_config.target_distance_m + short_config.stop_margin_m,
            short_config.target_distance_m + short_config.restart_margin_m,
            short_config.braking_stop_distance_m,
            short_config.depth_ttl_sec * 1000., short_config.visual_ttl_sec * 1000.,
            short_config.write_period_sec * 1000.,
            short_config.yaw_max_delta_rpm, short_config.yaw_full_error_ratio,
            short_config.pivot_limit_rpm,
            self._short_follow_adapter is None, self._short_follow_adapter is not None,
        )
        # Shared policy flags used by action_runtime to avoid stale radar holds.
        self._mmwave_runtime_enabled = bool(MODULE_MMWAVE_ENABLE)
        self._distance_source = str(DISTANCE_SOURCE)
        self._distance_parking_enable = bool(DISTANCE_PARKING_ENABLE)
        self._depth_track_observer = None
        # Passive worker only: no feedback/control locks, serial reads or new
        # permissions. A normal run with a log directory records the experiment.
        if os.environ.get("FOLLOW_DEPTH_TRACK_SHADOW_ENABLE", "1").lower() in {"1", "true", "yes"}:
            try:
                depth = getattr(self._sensor_runtime, "astra_depth", None)
                log_dir = os.environ.get("FOLLOW_LOG_DIR")
                if depth is not None and log_dir:
                    from car_control_modular.depth_track_online import DepthTrackOnlineObserver
                    self._depth_track_observer = DepthTrackOnlineObserver(
                        depth, self._action_runtime.get_recording_feedback,
                        os.path.join(log_dir, "depth_track_shadow"), logger,
                        hfov_deg=VISION_HFOV_DEG,
                        circumference=SENSOR_RUNTIME_CONFIG.astra_depth_encoder_wheel_circumference_m,
                    )
            except Exception:
                logger.exception("depth_track_shadow startup skipped; control unchanged")

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

    def _clear_longitudinal_context(
        self, *, revoke_translation: bool = True, reason: str = "context_invalid",
    ) -> None:
        scheduler = getattr(self, "_depth_async_scheduler", None)
        if scheduler is not None and reason == "visual_frame_begin" and not revoke_translation:
            # A new inference is not an identity verdict. Keep the accepted
            # ROI and in-flight physical sample until this frame actually
            # confirms or rejects the UID. Neither clock is renewed here.
            return
        # Lock order matches the Depth publisher. Revoke a copied old ROI,
        # but starting a visual observation is NOT a zero-speed decision.
        observer = getattr(self, "_depth_track_observer", None)
        if observer is not None and reason != "stale_vision_result":
            observer.revoke()
        with self._control_update_lock:
            if scheduler is not None:
                scheduler.revoke(reason)
            with self._longitudinal_context_lock:
                self._longitudinal_context = None
            if revoke_translation:
                self._revoke_depth_linear_authority(reason)
            else:
                linear = getattr(self, "_depth30_linear_snapshot", None)
                if linear is not None:
                    logger.info(
                        "depth_linear_preserved capture_frame_id=%s reason=%s "
                        "uid=%s linear=%s/%s sample_ts=%.6f age_ms=%.1f "
                        "deadline_renewed=False roi_context=revoked remaining_ms=%.1f",
                        getattr(self, "_active_capture_frame_id", None), reason,
                        linear[2], linear[0], linear[1], linear[3],
                        (time.monotonic() - linear[3]) * 1000.0,
                        max(0.0, linear[3] + PersonTracker._depth_linear_max_age_sec(linear[0])
                            - time.monotonic()) * 1000.0,
                    )

    def _revoke_depth_linear_authority(self, reason: str) -> None:
        """Withdraw translation, not yaw; caller owns the control lock."""
        linear = getattr(self, "_depth30_linear_snapshot", None)
        self._depth30_linear_snapshot = None
        self._depth30_linear_timing = None
        self._current_forward_allow_below_min = False
        self._last_depth30_translation_kind = None
        self._last_depth30_translation_ts = 0.0
        suspend_authority = getattr(self._follow_controller, "suspend_longitudinal_authority", None)
        if linear is None:
            # A feedback withdrawal already removed the tuple, but its
            # completed motor receipt can remain as bounded internal memory.
            # A later safety/identity revocation must invalidate that memory
            # too, even before a zero/STOP packet has reached the driver.
            prior = getattr(self._follow_controller, "_distance_pi_grant_withdrawal", None)
            admitted = getattr(self._follow_controller, "_distance_pi_admitted_grant", None)
            if (isinstance(prior, tuple) and len(prior) == 3
                    and prior[:2] == admitted and prior[2] != reason
                    and callable(suspend_authority)):
                suspend_authority(time.monotonic(), reason)
            return
        if callable(suspend_authority):
            # Early revocation can precede the old physical sample's expiry.
            # Preserve PI memory, not motion permission. Only an independently
            # qualified NEW sample may use a still-current completed packet
            # after a feedback-only withdrawal. This is NOT zero antiwindup.
            suspend_authority(time.monotonic(), reason)
        # Revoke a DRIVE/STEER packet already waiting at the motor lock. Keep
        # the sample watermark: older positive observations cannot revive it.
        self._current_forward_percent = self._current_steer_base_percent = 0
        self._forward_speed_latched_percent = None
        self.is_forwarding = False
        self._lateral_yaw_revision = int(getattr(self, "_lateral_yaw_revision", 0)) + 1
        logger.info(
            "depth_linear_revoked capture_frame_id=%s reason=%s decision=%s "
            "uid=%s previous=%s/%s sample_ts=%.6f age_ms=%.1f revision=%d",
            getattr(self, "_active_capture_frame_id", None), reason,
            getattr(self, "_last_control_decision_reason", ""),
            linear[2], linear[0], linear[1], linear[3],
            (time.monotonic() - linear[3]) * 1000.0, self._lateral_yaw_revision,
        )

    def _longitudinal_target_motion_enabled(self) -> bool:
        # Production config always supplies this flag. Small legacy adapters
        # without the new policy retain their existing explicitly tested path.
        return bool(getattr(getattr(self._follow_controller, "cfg", None),
                            "distance_target_motion_control_enable", True))

    def _depth_shared_braking_required(self) -> bool:
        return bool(
            not PersonTracker._longitudinal_target_motion_enabled(self)
            and getattr(self._follow_controller, "distance_pi_enabled", False))

    def _fresh_depth_linear_snapshot(self, target_id: Optional[int], *, now: Optional[float] = None,
                                     quiet: bool = False, _candidate=None, _candidate_timing=None):
        """Read physical Depth authority; quiet motor reads do no audit logging.

        Quiet means observational work is omitted, not that any UID, clock or
        safety gate is omitted. The stopped-grant veto is safety state and is
        still retained. A cached encoder read can wait; recheck its completed
        clock and the original authority before doing the pure cap calculation.
        """
        # A publication preview uses the SAME qualification as the motor
        # reader, but never exposes the candidate or poisons a live lease.
        # The control owner publishes only after this preview has succeeded.
        preview = _candidate is not None
        quiet = bool(quiet or preview)
        linear = _candidate if preview else getattr(self, "_depth30_linear_snapshot", None)
        def remember_quiet_veto(reason, feedback=None):
            if preview:
                return
            # Retain the actual physical-sample cause for the control owner.
            # A quiet motor read and a normal lateral read must not turn a
            # visibility/feedback timeout into an indistinguishable brake.
            live_linear = getattr(self, "_depth30_linear_snapshot", None)
            if (isinstance(linear, tuple) and len(linear) == 4
                    and isinstance(live_linear, tuple) and len(live_linear) == 4
                    and linear[2:] == live_linear[2:]):
                previous = getattr(self, "_depth30_read_veto", None)
                same_sample = bool(isinstance(previous, tuple) and len(previous) == 3
                                   and previous[:2] == (linear[2], linear[3]))
                continuity = {"visibility_expired", "continuation_feedback_stale",
                              "continuation_feedback_unavailable", "physical_depth_expired"}
                established_stop = bool(same_sample and previous[2] not in continuity
                    and getattr(self, "_depth30_continuation_veto", None) == previous[:2])
                if not (same_sample and (reason == "sticky_continuation_veto"
                        or (established_stop and reason in continuity))):
                    self._depth30_read_veto = (linear[2], linear[3], reason)
            # The periodic motor reader is intentionally silent. Keep only a
            # scalar snapshot so its existing post-write log can explain an
            # otherwise mysterious depth_fresh=False (CAP498/646), without a
            # new logger call or serial read inside the motor lock.
            if quiet:
                stamp = (linear[3] if isinstance(linear, tuple) and len(linear) == 4
                         else None)
                feedback_ts = getattr(feedback, "timestamp", None)
                self._last_quiet_depth_veto = (
                    target_id, stamp, reason, time.monotonic(), feedback_ts)
        def scope_valid():
            return bool(
                self._depth_longitudinal_authority_enabled() and not FOLLOW_ROTATION_ONLY
                and self.search_state == "none"
                and getattr(self._follow_controller, "search_state", "none") == "none"
                and self._vision_control_state.startswith("target_visible")
                and self._vision_control_state != "target_visible_low_quality"
                and not self._explicit_stop_requested and not self._runtime_shutdown_requested
                and self.running and not bool(getattr(self, "_brake_hold_active", False)))
        if not scope_valid():
            remember_quiet_veto("scope_invalid")
            if not preview:
                PersonTracker._veto_depth_continuation(self, linear, now)
            return None
        if linear is None:
            remember_quiet_veto("missing_snapshot")
            return None
        try:
            kind, percent, uid, stamp = linear
            current = time.monotonic() if now is None else float(now)
            if quiet:
                current = max(current, time.monotonic())
            if (
                kind not in {"forward", "backward"}
                or isinstance(percent, bool) or not 0 < int(percent) <= 100 or int(percent) != percent
                or target_id is None or int(uid) <= 0 or int(uid) != uid or uid != int(target_id)
                or getattr(self._follow_controller, "active_target_id", None) != uid
                or not math.isfinite(float(stamp)) or float(stamp) <= 0.0
            ):
                remember_quiet_veto("grant_identity_or_age_invalid")
                if not preview:
                    PersonTracker._veto_depth_continuation(self, linear, now)
                return None
            if not motion_identity_live(self, uid, current):
                remember_quiet_veto("motion_identity_invalid")
                if not preview:
                    PersonTracker._veto_depth_continuation(self, linear, now)
                return None
            if not 0.0 <= current - float(stamp) <= PersonTracker._depth_linear_max_age_sec(kind):
                remember_quiet_veto("physical_depth_expired" if current > float(stamp)
                                    else "invalid_sample_clock")
                if not preview:
                    PersonTracker._veto_depth_continuation(self, linear, now)
                return None
        except (ValueError, TypeError, OverflowError):
            remember_quiet_veto("invalid_snapshot")
            return None
        prepared = None if preview else getattr(self, "_depth30_prepared_timing", None)
        timing = _candidate_timing if preview else getattr(self, "_depth30_linear_timing", None)
        if prepared is not None and prepared.snapshot == linear:
            # Metadata is prepared before the positive tuple is published.
            # Readers see either the old complete grant or the new complete
            # grant, never a new high request with the old braking model.
            timing = prepared
        motion_enabled = PersonTracker._longitudinal_target_motion_enabled(self)
        shared_required = PersonTracker._depth_shared_braking_required(self)
        if (((motion_enabled and ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE) or shared_required)
                and getattr(self._follow_controller, "distance_pi_enabled", False)
                and (timing is None or timing.snapshot != linear)):
            # Only possible if the reader was overtaken by a complete newer
            # publication (or corrupt metadata). Do not fall back to an
            # unreviewed positive request, nor poison the newer grant's veto.
            remember_quiet_veto("timing_mismatch")
            return None
        relative_policy = bool(
            motion_enabled and ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE
            and getattr(self._follow_controller, "distance_pi_enabled", False)
            and timing is not None and timing.snapshot == linear
            and timing.continuation_motion is not None and kind == "forward")
        continuation_required = bool(
            relative_policy
            or (kind == "forward" and (shared_required or (
                timing is not None and timing.snapshot == linear
                and timing.braking_assessment is not None)))
            or current-float(stamp) > ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC)
        feedback = None
        if quiet and continuation_required:
            if getattr(self, "_depth30_continuation_veto", None) == (uid, stamp):
                previous_veto = getattr(self, "_last_quiet_depth_veto", None)
                if not (isinstance(previous_veto, tuple) and
                        previous_veto[:2] == (uid, stamp)):
                    remember_quiet_veto("sticky_continuation_veto")
                return None
            reader = getattr(getattr(self, "_action_runtime", None), "get_steering_feedback", None)
            feedback = reader() if callable(reader) else None
            current = max(current, time.monotonic())
            # A newer publication must be read as a whole on the next plan;
            # it cannot borrow this older grant's metadata or feedback work.
            if not preview and getattr(self, "_depth30_linear_snapshot", None) is not linear:
                remember_quiet_veto("superseded_during_feedback", feedback)
                return None
            latest_prepared = getattr(self, "_depth30_prepared_timing", None)
            latest_timing = getattr(self, "_depth30_linear_timing", None)
            if latest_prepared is not None and latest_prepared.snapshot == linear:
                latest_timing = latest_prepared
            if not preview and latest_timing is not timing:
                remember_quiet_veto("timing_changed_during_feedback", feedback)
                return None
            if (not scope_valid()
                    or getattr(self._follow_controller, "active_target_id", None) != uid
                    or not motion_identity_live(self, uid, current)
                    or not 0.0 <= current-float(stamp) <= PersonTracker._depth_linear_max_age_sec(kind)):
                remember_quiet_veto("scope_or_deadline_after_feedback", feedback)
                if not preview:
                    PersonTracker._veto_depth_continuation(self, linear, current)
                return None
        if timing is not None and timing.snapshot == linear:
            if current > timing.depth_expires_at:
                remember_quiet_veto("physical_depth_expired", feedback)
                return None
            if timing.feedforward_expires_at is not None and current >= timing.feedforward_expires_at:
                percent = min(int(percent), timing.distance_only_percent)
                if not quiet and getattr(self, "_last_depth_ff_fallback_timing", None) != timing:
                    self._last_depth_ff_fallback_timing = timing
                    logger.info(
                        "depth_ff_fallback uid=%s accepted_sample_ts=%s ff_origin_ts=%s "
                        "depth_expires_at=%s ff_expires_at=%s before_percent=%s after_percent=%s "
                        "pid_updated=False deadline_renewed=False",
                        uid, timing.accepted_depth_timestamp, timing.feedforward_timestamp,
                        timing.depth_expires_at, timing.feedforward_expires_at, linear[1], percent,
                    )
                if percent <= 0:
                    remember_quiet_veto("feedforward_expired", feedback)
                    return None
        # Use one continuous envelope from admission, not only AFTER180ms.
        # This is still an upper bound on an already approved request, never
        # extra PI demand. Missing motion evidence retains the old policy.
        if continuation_required:
            if getattr(self, "_depth30_continuation_veto", None) == (uid, stamp):
                previous_veto = getattr(self, "_last_quiet_depth_veto", None)
                if not (isinstance(previous_veto, tuple) and
                        previous_veto[:2] == (uid, stamp)):
                    remember_quiet_veto("sticky_continuation_veto", feedback)
                return None
            if quiet:
                cap_percent, reason = PersonTracker._depth_forward_continuation_limit(
                    self, linear, timing, current, feedback=feedback, quiet=True)
            else:
                # Preserve the public/legacy helper contract outside the
                # motor writer, including existing injectable test readers.
                cap_percent, reason = PersonTracker._depth_forward_continuation_limit(self, linear, timing, current)
            percent = min(int(percent), cap_percent)
            allowed = percent > 0
            audit = (linear, allowed, reason, percent)
            if not quiet and getattr(self, "_last_depth_continuation_audit", None) != audit:
                self._last_depth_continuation_audit = audit
                logger.info(
                    "depth_forward_continuation uid=%s sample_ts=%s age_ms=%.1f allowed=%s "
                    "reason=%s speed_percent=%s deadline=%s max_age_ms=%.0f "
                    "pid_updated=False acceleration_allowed=False deadline_renewed=False "
                    "original_percent=%s braking_distance_m=%s original_speed_bound_m_s=%s "
                    "relative_policy=%s motion_origin_ts=%s target_speed_origin_m_s=%s",
                    uid, stamp, (current-stamp)*1000., allowed, reason, percent,
                    stamp+PersonTracker._depth_linear_max_age_sec(kind), PersonTracker._depth_linear_max_age_sec(kind)*1000.,
                    linear[1], getattr(timing, "continuation_distance_m", None),
                    getattr(timing, "continuation_speed_bound_m_s", None),
                    relative_policy,
                    getattr(getattr(timing, "continuation_motion", None), "sample_timestamp", None),
                    getattr(getattr(timing, "continuation_motion", None), "target_speed_bound_m_s", None),
                )
            if not allowed:
                remember_quiet_veto(reason, feedback)
                if not preview:
                    PersonTracker._veto_depth_continuation(self, linear, now)
                return None
        if quiet and not preview:
            self._last_quiet_depth_veto = None
        previous_veto = getattr(self, "_depth30_read_veto", None)
        if not preview and isinstance(previous_veto, tuple) and previous_veto[:2] == (uid, stamp):
            self._depth30_read_veto = None
        return kind, int(percent), int(uid), float(stamp)

    def _depth_linear_withdrawal_reason(self, linear, now, fallback):
        """Carry one sample's actual veto; unknown/safety causes stay closed.

        This is provenance for a later NEW-sample recovery, not permission to
        reuse this tuple. Only the fresh reader/final writer authorize motion.
        """
        if not isinstance(linear, tuple) or len(linear) != 4:
            return fallback
        veto = getattr(self, "_depth30_read_veto", None)
        if isinstance(veto, tuple) and len(veto) == 3 and veto[:2] == (linear[2], linear[3]):
            # Preserve an established safety veto even when its old clock
            # subsequently expires. Time cannot turn braking into continuity.
            return ("physical_depth_expired" if veto[2] == "physical_depth_expired"
                    else "lateral_depth:" + str(veto[2]))
        if now-linear[3] > PersonTracker._depth_linear_max_age_sec(linear[0]):
            return "physical_depth_expired"
        return fallback

    def _veto_depth_continuation(self, linear, now):
        """A read-time veto cannot later revive the same stopped old grant.

        Motor readers never mutate the control-owned snapshot or PI. This
        marker only vetoes its UID/physical timestamp; a new fresh grant has a
        different timestamp and follows the normal approval path.
        """
        try:
            current = time.monotonic() if now is None else float(now)
            prepared = getattr(self, "_depth30_prepared_timing", None)
            timing = getattr(self, "_depth30_linear_timing", None)
            if prepared is not None and prepared.snapshot == linear:
                timing = prepared
            relative_policy = bool(
                PersonTracker._longitudinal_target_motion_enabled(self) and ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE
                and timing is not None and timing.snapshot == linear
                and timing.continuation_motion is not None)
            shared_policy = bool(PersonTracker._depth_shared_braking_required(self) or (
                timing is not None and timing.snapshot == linear
                and timing.braking_assessment is not None))
            if (linear is not None and linear[0] == "forward"
                    and 0 <= current-linear[3] <= ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC
                    and (relative_policy or shared_policy
                         or current-linear[3] > ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC)):
                self._depth30_continuation_veto = (linear[2], linear[3])
        except (ValueError, TypeError, OverflowError):
            pass

    def _publish_depth_linear_pair(self, linear, timing):
        """Make all inputs available before publishing a positive authority.

        The control lock owns publication. The motor reader can run between
        these assignments, so it accepts prepared metadata ONLY for its exact
        tuple; an old UID/time cannot borrow the next grant's motion evidence.
        A revoke still works simply by setting the canonical tuple to None.
        """
        self._depth30_prepared_timing = timing
        self._depth30_linear_snapshot = linear
        self._depth30_linear_timing = timing
        self._depth30_prepared_timing = None

    @staticmethod
    def _depth_linear_max_age_sec(kind: str) -> float:
        return (ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC if kind == "forward" else
                min(ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC,
                    ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC))

    def _pending_forward_depth_lease(self, target_id: int, now: float):
        """Identify a transiently unreadable Depth lease, without authorizing motion.

        A metadata/read race can make _fresh_depth_linear_snapshot return None
        before the physical deadline. An explicit continuation veto or zero
        braking cap instead keeps the original revoke path, never STEER hold.
        Only the fresh reader and final motor writer authorize positive wheels.
        """
        linear = getattr(self, "_depth30_linear_snapshot", None)
        try:
            if not isinstance(linear, tuple) or len(linear) != 4:
                return None
            kind, percent, uid, stamp = linear
            if (kind != "forward" or type(percent) is not int or percent <= 0
                    or type(uid) is not int or uid <= 0 or uid != target_id
                    or getattr(self._follow_controller, "active_target_id", None) != uid
                    or not all(math.isfinite(value) for value in (now, stamp))
                    or not 0 <= now-stamp <= PersonTracker._depth_linear_max_age_sec(kind)
                    or not self._depth_longitudinal_authority_enabled()
                    or self.search_state != "none"
                    or getattr(self._follow_controller, "search_state", "none") != "none"
                    or self._vision_control_state not in {
                        "target_visible", "target_visible_depth_valid",
                        "target_visible_depth_missing",
                    }
                    or not self.running or self._explicit_stop_requested
                    or self._runtime_shutdown_requested
                    or getattr(self, "_brake_hold_active", False)
                    or getattr(self, "_soft_stop_active", False)
                    or getattr(self, "_near_yaw_park_request", None) is not None
                    or getattr(self, "stop_action_execution", False)
                    or getattr(self, "person_detected_flag", False)
                    or getattr(self, "_search_handoff_uid", None) == uid
                    or not motion_identity_live(self, uid, now)):
                return None
            # A same-grant read-time veto is sticky until a NEW Depth sample.
            # Zero-axis STEER has not been proven equivalent to the original
            # STOP path when braking margin or feedback safety has failed.
            if getattr(self, "_depth30_continuation_veto", None) == (uid, stamp):
                return None
            timing = getattr(self, "_depth30_linear_timing", None)
            if timing is not None and timing.snapshot == linear:
                if now > timing.depth_expires_at:
                    return None
                # Feedforward expiry can explicitly reduce the SAME grant's
                # distance-only permission to zero without setting the veto.
                if (timing.feedforward_expires_at is not None
                        and now >= timing.feedforward_expires_at
                        and timing.distance_only_percent <= 0):
                    return None
            for audit_name in ("_last_depth_relative_cap_audit",
                               "_last_depth_continuation_cap_audit"):
                audit = getattr(self, audit_name, None)
                if (isinstance(audit, tuple) and len(audit) == 3
                        and audit[0] == linear and audit[2] <= 0):
                    return None
            continuation_audit = getattr(self, "_last_depth_continuation_audit", None)
            if (isinstance(continuation_audit, tuple) and len(continuation_audit) == 4
                    and continuation_audit[0] == linear and not continuation_audit[1]):
                return None
            return linear
        except (TypeError, ValueError, OverflowError, AttributeError):
            return None

    def _depth_forward_continuation_safe(self, linear, timing, now: float, *, feedback=...,
                                         relative_feedback=False, vision_stamp=..., audit=True,
                                         shared_braking=False):
        """Qualify an old grant; shared callers own the sole braking budget.

        The legacy default additionally evaluates its original stop model.
        Shared-assessment callers only consume the time/encoder qualification
        here, then evaluate the assessment bound to their physical depth sample.
        Neither branch creates a new sample or extends its lifetime.
        """
        if linear[0] != "forward" or timing is None or timing.snapshot != linear:
            return False, "missing_original_forward_grant"
        distance, speed = timing.continuation_distance_m, timing.continuation_speed_bound_m_s
        visible = getattr(self, "_validated_visual_observation", None)
        if visible is not None:
            # A completed full identity check publishes before slow ranging.
            # Old yaw/control publication cannot erase this newer observation.
            # False means a current rejected result, never a legacy fallback.
            if not isinstance(visible, ValidatedVisualObservation):
                return False, "visibility_rejected"
            if visible.uid != linear[2]:
                return False, "visibility_identity_mismatch"
            if (visible.continuation_sample_timestamp is not None
                    and visible.continuation_sample_timestamp != linear[3]):
                return False, "visibility_sample_mismatch"
            if now < visible.validated_at:
                return False, "visibility_invalid_clock"
            if not visible.permits_depth(linear[2], linear[3], now):
                return False, "visibility_expired"
        else:
            if vision_stamp is ...:
                vision_stamp = getattr(self, "_last_vision_control_ts", None)
            if (vision_stamp is None or not math.isfinite(vision_stamp)
                    or not 0 <= now-vision_stamp <= min(.25, ROTATE_HOLD_STALE_SEC)):
                return False, "visibility_expired"
        if (distance is None or speed is None or not math.isfinite(distance)
                or not math.isfinite(speed) or distance <= 0 or speed < 0):
            return False, "missing_braking_evidence"
        if not shared_braking:
            deceleration = float(DISTANCE_APPROACH_DECELERATION_M_S2)
            delay = float(DISTANCE_APPROACH_RESPONSE_DELAY_SEC)
            if not math.isfinite(deceleration) or deceleration <= 0 or not math.isfinite(delay) or delay < 0:
                return False, "invalid_braking_model"
        age = now-linear[3]
        if not math.isfinite(age) or age < 0:
            return False, "invalid_braking_model"
        if age > min(MAX_FORWARD_DEPTH_TTL_SEC, PersonTracker._depth_linear_max_age_sec("forward")):
            return False, "physical_depth_expired"
        # Both the full-speed and reduced-speed paths consume the same current
        # encoder evidence. A far target must not bypass wheel/feedback checks.
        if feedback is ...:
            reader = getattr(getattr(self, "_action_runtime", None), "get_steering_feedback", None)
            feedback = reader() if callable(reader) else None
        _body, outer, feedback_reason = continuation_feedback_speeds(
            feedback=feedback, now=now,
            circumference=VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M,
            max_rpm=FORWARD_MAX_RPM, original_speed_bound=speed,
            feedback_max_age_sec=RELATIVE_CONTINUATION_FEEDBACK_MAX_AGE_SEC if relative_feedback else .10,
            reverse_tail_rpm=RELATIVE_CONTINUATION_REVERSE_TAIL_RPM if relative_feedback else 0.)
        if outer is None:
            if feedback is None:
                feedback_reason = "continuation_feedback_unavailable"
            elif (feedback_reason == "continuation_feedback_invalid"
                    and getattr(feedback, "trustworthy", False)
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) for v in (
                                feedback.timestamp, feedback.left_forward_rpm,
                                feedback.right_forward_rpm))
                    and min(feedback.left_forward_rpm, feedback.right_forward_rpm) >= 0
                    and max(feedback.left_forward_rpm, feedback.right_forward_rpm) <= FORWARD_MAX_RPM
                    and now-feedback.timestamp > (
                        RELATIVE_CONTINUATION_FEEDBACK_MAX_AGE_SEC
                        if relative_feedback else .10)):
                feedback_reason = "continuation_feedback_stale"
            audit_key = (linear, feedback_reason)
            if audit and getattr(self, "_last_depth_continuation_feedback_veto", None) != audit_key:
                self._last_depth_continuation_feedback_veto = audit_key
                exceeded = feedback_reason == "continuation_speed_exceeds_bound"
                logger.info(
                    "depth_continuation_feedback_veto uid=%s sample_ts=%s reason=%s "
                    "feedback_ts=%s body_rpm=%s outer_rpm=%s original_speed_bound_m_s=%s "
                    "bound_scope=longitudinal_body braking_scope=outer_wheel deadline_renewed=False",
                    linear[2], linear[3], feedback_reason, getattr(feedback, "timestamp", None),
                    .5*(feedback.left_forward_rpm+feedback.right_forward_rpm) if exceeded else None,
                    max(feedback.left_forward_rpm, feedback.right_forward_rpm) if exceeded else None,
                    speed)
            return False, feedback_reason
        if shared_braking:
            return True, "same_grant_evidence_qualified"
        # Assume the target stops moving, reserve all travel since the ORIGINAL
        # capture plus modeled braking. Approved speed is not a target-speed FF.
        # Preserve the original full-bound test; newer outer-wheel momentum may
        # only tighten it, never replace the old bound by a smaller body speed.
        remaining = distance - max(TARGET_DISTANCE+DISTANCE_PID_DEADBAND_M, FOLLOW_BRAKE_DISTANCE_M)
        braking_speed = max(speed, outer)
        needed = speed*age + braking_speed*delay + braking_speed*braking_speed/(2*deceleration) + .02
        return (True, "same_grant_braking_margin") if remaining >= needed else (False, "braking_margin")

    def _depth_forward_continuation_required(self, linear, timing, now):
        """Pure predicate shared by the normal reader and last motor write."""
        try:
            motion_enabled = PersonTracker._longitudinal_target_motion_enabled(self)
            shared_required = PersonTracker._depth_shared_braking_required(self)
            if (linear[0] == "forward" and (shared_required or (
                    motion_enabled and ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE))
                    and getattr(self._follow_controller, "distance_pi_enabled", False)
                    and (timing is None or timing.snapshot != linear)):
                return True  # Reader would reject missing atomic grant metadata.
            return bool(
                (linear[0] == "forward"
                 and getattr(self._follow_controller, "distance_pi_enabled", False)
                 and timing is not None and timing.snapshot == linear
                 and (shared_required or timing.braking_assessment is not None
                      or (motion_enabled and ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE
                          and timing.continuation_motion is not None)))
                or float(now)-float(linear[3]) > ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC)
        except (TypeError, ValueError, OverflowError, IndexError):
            return True  # Invalid metadata may not bypass the write-time check.

    def _depth_forward_continuation_limit(self, linear, timing, now, *, feedback=..., quiet=False):
        """One policy for motor reads AND later evidence tightening the lease."""
        if feedback is ...:
            if quiet:
                return 0, "feedback_not_supplied_at_write"
            reader = getattr(getattr(self, "_action_runtime", None), "get_steering_feedback", None)
            feedback = reader() if callable(reader) else None
        vision_stamp = getattr(self, "_last_vision_control_ts", None)
        # The cache read can yield to the encoder worker. Validate against the
        # time AFTER that read, not a stale pre-lock clock which makes a real
        # new report appear to come from the future. Never alter its timestamp.
        # The same time also rechecks visibility and the original Depth TTL.
        now = float(now) if quiet else max(float(now), time.monotonic())
        motion_enabled = PersonTracker._longitudinal_target_motion_enabled(self)
        motion = getattr(timing, "continuation_motion", None) if motion_enabled else None
        assessment = getattr(timing, "braking_assessment", None)
        relative_policy = bool(motion_enabled and ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE and motion is not None
                               and getattr(self._follow_controller, "distance_pi_enabled", False))
        if PersonTracker._depth_shared_braking_required(self) and assessment is None:
            return 0, "missing_shared_braking_evidence"
        if assessment is not None:
            from car_control_modular.sample_braking import SampleBrakingAssessment
            if (not isinstance(assessment, SampleBrakingAssessment)
                    or not assessment.valid_for(linear[2], linear[3])
                    or (motion_enabled and not relative_policy)
                    or (not motion_enabled and assessment.target_speed_m_s != 0.)):
                return 0, "invalid_shared_braking_evidence"
        allowed, reason = PersonTracker._depth_forward_continuation_safe(
            self, linear, timing, now, feedback=feedback,
            relative_feedback=relative_policy or assessment is not None,
            vision_stamp=vision_stamp, audit=not quiet,
            shared_braking=assessment is not None)
        if assessment is not None:
            if not allowed:
                return 0, reason
            reader = getattr(getattr(self, "_action_runtime", None),
                             "continuation_executed_speed_bound_rpm", None)
            executed = reader(linear[2], now) if callable(reader) else None
            if assessment.feedback_interval_covered:
                # The reduced uncertainty reserve depends on completed-command
                # coverage, not merely the last tracked follow speed. Another
                # writer's high->low sequence must not disappear behind the
                # same low current RPM. Normal lower follow writes retain the
                # continuous receipt chain and remain legal contractions.
                interval_reader = getattr(getattr(self, "_action_runtime", None),
                    "braking_interval_continuation_bound_rpm", None)
                interval_bound = (interval_reader(linear[2], linear[3], now)
                                  if callable(interval_reader) else None)
                if (not isinstance(interval_bound, (int, float))
                        or isinstance(interval_bound, bool)
                        or not math.isfinite(interval_bound) or interval_bound < 0):
                    return 0, "shared_braking_interval_lost"
                if executed is None:
                    executed = interval_bound
                elif (isinstance(executed, (int, float)) and not isinstance(executed, bool)
                      and math.isfinite(executed) and executed >= 0):
                    executed = max(executed, interval_bound)
            # This bound belongs to the ORIGINAL admitted sample, not to the
            # latest (possibly reduced) speed command. Previously a legal
            # 80 -> 50 RPM reduction reclassified the completed 80 RPM write
            # as an unauthorized new command and stopped both wheels.
            execution_bound_rpm = (
                timing.continuation_speed_bound_m_s * 60. / assessment.circumference_m)
            budget = assessment.budget(
                now, max(abs(feedback.left_forward_rpm), abs(feedback.right_forward_rpm)),
                authorized_rpm=linear[1]*FORWARD_MAX_RPM/100., completed_rpm=executed,
                execution_bound_rpm=execution_bound_rpm,
                tightened_distance_m=timing.continuation_distance_m)
            cap_rpm, cap_reason = budget.cap_rpm, budget.reason
            # Later closer range only tightens this assessment's margin; it
            # must not switch to the legacy 100 ms encoder policy, renew the
            # sample, or re-admit its original speed at the closer distance.
            percent = min(int(linear[1]), max(0, int(cap_rpm*100./FORWARD_MAX_RPM)))
            audit = (linear, cap_reason, percent)
            if not quiet and getattr(self, "_last_depth_shared_cap_audit", None) != audit:
                self._last_depth_shared_cap_audit = audit
                logger.info(
                    "depth_shared_braking_cap uid=%s sample_ts=%s assessment_ts=%s age_ms=%.1f "
                    "cap_rpm=%.2f approved_percent=%s reason=%s margin_m=%.4f required_stop_m=%.4f "
                    "past_travel_bound_rpm=%.2f future_outer_allowance_rpm=%.2f "
                    "original_execution_bound_rpm=%.2f current_authorized_rpm=%.2f "
                    "target_speed_bound_m_s=%.4f tightened_distance_m=%s "
                    "feedback_reserved_ms=%.1f feedback_reserve_observed=%s "
                    "feedback_interval_covered=%s "
                    "same_sample_assessment=True deadline_renewed=False",
                    linear[2], linear[3], assessment.checked_at, (now-linear[3])*1000.,
                    cap_rpm, percent, cap_reason, budget.margin_m, budget.required_stop_m,
                    assessment.travel_bound_rpm, assessment.outer_allowance_rpm,
                    execution_bound_rpm, linear[1]*FORWARD_MAX_RPM/100.,
                    assessment.target_speed_m_s, timing.continuation_distance_m,
                    assessment.effective_feedback_reserve_sec*1000., assessment.observed_feedback_reserve,
                    assessment.feedback_interval_covered)
            return percent, cap_reason
        if relative_policy:
            # Use the fresh controller's 150ms encoder budget continuously;
            # the relative model reserves that ENTIRE latency in its margin.
            # Near-zero signed tails still go through the final reversal guard.
            # No identity/visibility/physical deadline or momentum veto is lost.
            if not allowed and reason != "braking_margin":
                return 0, reason
            stop_distance = max(TARGET_DISTANCE-ASTRA_DEPTH_CONTINUATION_OVERSHOOT_M,
                                FOLLOW_REVERSE_START_DISTANCE_M, FOLLOW_BRAKE_DISTANCE_M+.05)
            cap, reason = relative_continuation_speed_cap(
                distance=timing.continuation_distance_m, stop_distance=stop_distance,
                original_speed_bound=timing.continuation_speed_bound_m_s,
                sample_age=now-linear[3], feedback=feedback, now=now,
                circumference=VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M,
                max_rpm=FORWARD_MAX_RPM, deceleration=DISTANCE_APPROACH_DECELERATION_M_S2,
                response_delay=DISTANCE_APPROACH_RESPONSE_DELAY_SEC,
                motion=motion, target_id=linear[2], sample_timestamp=linear[3])
            percent = min(int(linear[1]), max(0, int(cap*100./FORWARD_MAX_RPM)))
            audit = (linear, reason, percent)
            if not quiet and getattr(self, "_last_depth_relative_cap_audit", None) != audit:
                self._last_depth_relative_cap_audit = audit
                logger.info(
                    "depth_relative_continuation_cap uid=%s sample_ts=%.6f age_ms=%.1f "
                    "original_percent=%s cap_rpm=%.2f approved_percent=%s reason=%s "
                    "target_speed_origin_m_s=%s motion_span_ms=%s stop_model_distance_m=%.3f "
                    "overshoot_allowance_m=%.3f distance_m=%s original_speed_bound_m_s=%s "
                    "deceleration_model_m_s2=%.3f response_delay_sec=%.3f "
                    "feedback_ts=%s feedback_forward_rpm=%s deadline_renewed=False "
                    "acceleration_allowed=False stationary_stop_guaranteed=False",
                    linear[2], linear[3], (now-linear[3])*1000., linear[1], cap, percent, reason,
                    getattr(motion, "target_speed_bound_m_s", None),
                    None if getattr(motion, "span_sec", None) is None else motion.span_sec*1000.,
                    stop_distance, ASTRA_DEPTH_CONTINUATION_OVERSHOOT_M,
                    timing.continuation_distance_m, timing.continuation_speed_bound_m_s,
                    DISTANCE_APPROACH_DECELERATION_M_S2, DISTANCE_APPROACH_RESPONSE_DELAY_SEC,
                    getattr(feedback, "timestamp", None),
                    None if feedback is None else (feedback.left_forward_rpm, feedback.right_forward_rpm))
            return percent, reason
        if allowed:
            return int(linear[1]), reason
        if reason != "braking_margin" or not ASTRA_DEPTH_CONTINUATION_SPEED_CAP_ENABLE:
            return 0, reason
        cap, reason = continuation_speed_cap(
            distance=timing.continuation_distance_m,
            stop_distance=max(TARGET_DISTANCE+DISTANCE_PID_DEADBAND_M, FOLLOW_BRAKE_DISTANCE_M),
            original_speed_bound=timing.continuation_speed_bound_m_s,
            sample_age=now-linear[3], feedback=feedback, now=now,
            circumference=VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M,
            max_rpm=FORWARD_MAX_RPM, deceleration=DISTANCE_APPROACH_DECELERATION_M_S2,
            response_delay=DISTANCE_APPROACH_RESPONSE_DELAY_SEC,
        )
        percent = min(int(linear[1]), int(cap*100./FORWARD_MAX_RPM))
        audit = (linear, reason, percent)
        if not quiet and getattr(self, "_last_depth_continuation_cap_audit", None) != audit:
            self._last_depth_continuation_cap_audit = audit
            logger.info(
                "depth_continuation_cap uid=%s sample_ts=%.6f age_ms=%.1f "
                "original_percent=%s cap_rpm=%.2f approved_percent=%s reason=%s "
                "feedback_ts=%s feedback_forward_rpm=%s distance_m=%s "
                "original_speed_bound_m_s=%s body_rpm=%s outer_rpm=%s "
                "bound_scope=longitudinal_body braking_scope=outer_wheel deadline_renewed=False",
                linear[2], linear[3], (now-linear[3])*1000., linear[1], cap, percent, reason,
                getattr(feedback, "timestamp", None),
                None if feedback is None else (feedback.left_forward_rpm, feedback.right_forward_rpm),
                timing.continuation_distance_m, timing.continuation_speed_bound_m_s,
                None if feedback is None else .5*(feedback.left_forward_rpm+feedback.right_forward_rpm),
                None if feedback is None else max(feedback.left_forward_rpm, feedback.right_forward_rpm),
            )
        return percent, reason

    def _depth_continuation_evidence(self, frame, target_id, percent, now):
        """Capture a conservative bound only while the original grant is fresh."""
        # Diagnostic only. A rejected encoder pair is not a missing braking
        # model, and neither diagnosis creates authority. Reset on EVERY call
        # so an earlier sample's veto cannot be attributed to the next one.
        self._depth_continuation_evidence_veto = None
        def reject(reason):
            self._depth_continuation_evidence_veto = reason
            return None, None

        state, feedback = frame.distance_state, frame.steering_feedback
        from car_control_modular.sample_braking import (
            SampleBrakingAssessment, sample_feedback_time_valid)
        assessment = getattr(getattr(self._follow_controller, "last_distance_pid_result", None),
                             "pi_braking_assessment", None)
        motion_enabled = PersonTracker._longitudinal_target_motion_enabled(self)
        motion = (self._depth_continuation_motion_evidence(
            frame, target_id, state.sample_timestamp, now) if motion_enabled else None)
        shared_valid = bool(
            isinstance(assessment, SampleBrakingAssessment)
            and assessment.valid_for(target_id, state.sample_timestamp)
            and getattr(self._follow_controller, "_distance_pid_last_sample_timestamp", None)
                == state.sample_timestamp
            and (motion_enabled or assessment.target_speed_m_s == 0.))
        if ((PersonTracker._depth_shared_braking_required(self) and not shared_valid)
                or (assessment is not None and not shared_valid)):
            return reject("shared_braking_missing_or_mismatched")
        from car_control_modular.depth_continuation import RELATIVE_CONTINUATION_REVERSE_TAIL_RPM
        tail = RELATIVE_CONTINUATION_REVERSE_TAIL_RPM if motion is not None or shared_valid else 0.
        values = (frame.distance_m, state.raw_distance_m,
                  None if feedback is None else feedback.timestamp,
                  None if feedback is None else feedback.left_forward_rpm,
                  None if feedback is None else feedback.right_forward_rpm)
        # Pure distance control uses CURRENT wheel feedback to bound braking,
        # not to differentiate target motion at the Depth capture instant.
        # Reuse only the exact independently-timed assessment already checked
        # by PI; keep the legacy paired-sample rule for every other caller.
        current_feedback = bool(
            shared_valid and not motion_enabled
            and getattr(assessment, "current_feedback_independent", False) is True
            and feedback is not None
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in values)
            and assessment.feedback_timestamp == feedback.timestamp
            and assessment.outer_rpm == max(abs(feedback.left_forward_rpm),
                                            abs(feedback.right_forward_rpm))
            and sample_feedback_time_valid(state.sample_timestamp, feedback.timestamp, now,
                                           current_feedback_independent=True))
        if any(v is None or not math.isfinite(v) for v in values):
            return reject("distance_or_feedback_unavailable")
        if feedback is None or not feedback.trustworthy:
            return reject("feedback_untrustworthy")
        if not 0 <= now-feedback.timestamp <= .15:
            return reject("feedback_expired_or_future")
        if (state.sample_timestamp is None
                or (not current_feedback and abs(feedback.timestamp-state.sample_timestamp) > .15)):
            return reject("feedback_sample_pair_mismatch")
        if min(frame.distance_m, state.raw_distance_m) <= 0:
            return reject("distance_invalid")
        if (min(feedback.left_forward_rpm, feedback.right_forward_rpm) < 0
                and max(abs(feedback.left_forward_rpm), abs(feedback.right_forward_rpm)) > tail):
            return reject("feedback_reverse_exceeds_tail")
        if max(feedback.left_forward_rpm, feedback.right_forward_rpm) > FORWARD_MAX_RPM+5:
            return reject("feedback_overspeed")
        if frame.hazard.active or any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right)):
            return reject("hazard_or_obstacle")
        if state.safety_distance_m is not None or state.brake_latched:
            return reject("depth_safety_or_brake_latched")
        if not any(p.track_id == target_id for p in frame.persons):
            return reject("target_not_present")
        if shared_valid:
            # The new request bounds FUTURE travel/feedback, not the elapsed
            # time since Depth capture. The immutable assessment retains the
            # independently captured physical travel bound for that past.
            return (assessment.distance_m, max(assessment.travel_bound_rpm,
                    percent*FORWARD_MAX_RPM/100.+assessment.outer_allowance_rpm)
                    *assessment.circumference_m/60.)
        # A newly lowered request cannot erase the momentum of an earlier
        # completed higher motor write. This history is a physical cost only:
        # never a command, identity proof, feedback substitute or renewed lease.
        bound_rpm = max(percent*FORWARD_MAX_RPM/100., abs(feedback.left_forward_rpm),
                        abs(feedback.right_forward_rpm))
        if motion is not None:
            reader = getattr(getattr(self, "_action_runtime", None),
                             "continuation_executed_speed_bound_rpm", None)
            # A completed motor write may arrive while this Depth decision is
            # being computed. Its physical cost is not tied to the older PI
            # entry clock; final publication still enforces the original TTL.
            executed = reader(target_id, max(now, time.monotonic())) if callable(reader) else None
            if (isinstance(executed, (float, int)) and not isinstance(executed, bool)
                    and math.isfinite(executed) and executed >= 0):
                if executed > bound_rpm:
                    logger.info(
                        "depth_continuation_execution_budget uid=%s sample_ts=%s "
                        "request_rpm=%.2f feedback_rpm=%s captured_bound_rpm=%.2f "
                        "executed_bound_rpm=%.2f authority_renewed=False request_increased=False",
                        target_id, state.sample_timestamp, percent*FORWARD_MAX_RPM/100.,
                        (feedback.left_forward_rpm, feedback.right_forward_rpm), bound_rpm, executed)
                bound_rpm = max(bound_rpm, executed)
        speed = bound_rpm * VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M/60.
        result = getattr(self._follow_controller, "last_distance_pid_result", None) if motion_enabled else None
        closing = getattr(result, "approach_closing_m_s", None) if motion_enabled else None
        # Static legacy policy retains its old conservative closure budget.
        # Relative policy already subtracts target velocity: mixing closure
        # into ego speed here would count an approaching target twice.
        if motion is None and closing is not None and math.isfinite(closing):
            speed = max(speed, closing)
        return min(frame.distance_m, state.raw_distance_m), speed

    def _depth_continuation_motion_evidence(self, frame, target_id, stamp, now):
        """Freeze this sample's verified motion, never a held/FF prior.

        Called ONLY at fresh grant publication. Late/replayed data may inherit
        the original object, not manufacture a new motion clock or identity.
        """
        if not PersonTracker._longitudinal_target_motion_enabled(self) or not ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE:
            return None
        from car_control_modular.longitudinal_approach import RawDepthMotionEvidence
        from car_control_modular.depth_continuation import ContinuationMotionEvidence
        controller = self._follow_controller
        evidence = getattr(controller, "_braking_motion_evidence", None)
        result = getattr(controller, "last_distance_pid_result", None)
        if (not getattr(controller, "distance_pi_enabled", False)
                or getattr(controller, "active_target_id", None) != target_id
                or getattr(controller, "_distance_pid_last_sample_timestamp", None) != stamp
                or not isinstance(evidence, RawDepthMotionEvidence)
                or not evidence.valid_for(stamp, getattr(controller, "_braking_range_rate", None))
                or getattr(controller, "_braking_rate_source", None) != "raw_depth_window"
                or not getattr(result, "pi_motion_window_used", False)
                or getattr(result, "pi_brake_source", None) != "raw_relative_motion"
                or getattr(result, "measurement_jump_clamped", False)
                or frame.distance_state.sample_timestamp != stamp
                or not 0 <= now-stamp <= ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC):
            return None
        motion = ContinuationMotionEvidence(
            uid=target_id, sample_timestamp=stamp,
            target_speed_bound_m_s=evidence.target_speed_bound_m_s,
            range_rate_m_s=evidence.range_rate_m_s,
            span_sec=evidence.span_sec, sample_count=evidence.sample_count)
        return motion if motion.valid_for(target_id, stamp) else None

    def _continue_late_depth_linear_decision(self, decision, frame, target_id, now):
        """Late evidence can reduce one live grant; it cannot grant or renew one."""
        stamp = frame.distance_state.sample_timestamp
        if (not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
                or not ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC < now-stamp
                <= ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC):
            return None
        return self._tighten_existing_depth_linear_decision(decision, frame, target_id, now)

    def _tighten_existing_depth_linear_decision(self, decision, frame, target_id, now):
        """Apply rejected/late evidence only to an independently live old grant."""
        stamp = frame.distance_state.sample_timestamp
        previous = self._fresh_depth_linear_snapshot(target_id, now=now)
        state = frame.distance_state
        unsafe = (decision.explicit_stop_requested or decision.shutdown_requested or decision.soft_stop_requested
                  or any(a.kind == "stop" for a in decision.actions)
                  or frame.hazard.active or any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
                  or state.safety_distance_m is not None or state.brake_latched
                  or not any(p.track_id == target_id for p in frame.persons))
        if unsafe:
            self._revoke_depth_linear_authority("late_depth_safety")
            return [ControlAction.forward(0, "late_depth_safety")], True
        if previous is None:
            # A failed owner/UID/visibility/stop check cannot be reversed by
            # toggling that state back while the old tuple is still present.
            self._revoke_depth_linear_authority("late_depth_no_live_authority")
            return [], False
        if previous[0] != "forward":
            return [], False
        if stamp < previous[3]-1e-9:
            return [], False  # Historical depth must not replace a newer grant.
        untrusted = getattr(self._follow_controller, "_distance_longitudinally_untrusted", None)
        distances = (frame.distance_m, state.raw_distance_m)
        if (any(d is None or not math.isfinite(d) for d in distances)
                or min(distances) <= TARGET_DISTANCE+DISTANCE_PID_DEADBAND_M
                or (callable(untrusted) and untrusted(frame))
                or any(a.kind not in {"forward", "idle"} for a in decision.actions)):
            self._revoke_depth_linear_authority("late_depth_near_or_untrusted")
            return [ControlAction.forward(0, "late_depth_near_or_untrusted")], True
        timing = getattr(self, "_depth30_linear_timing", None)
        if timing is None or timing.snapshot != getattr(self, "_depth30_linear_snapshot", None):
            self._revoke_depth_linear_authority("late_depth_missing_original_evidence")
            return [], False
        # A later close reading can only tighten the old braking margin. Its
        # timestamp is not installed as either an accepted sample or deadline.
        if timing.continuation_distance_m is not None:
            timing = replace(timing, continuation_distance_m=min(timing.continuation_distance_m, *distances))
        cap_percent, reason = PersonTracker._depth_forward_continuation_limit(self, timing.snapshot, timing, now)
        if cap_percent <= 0:
            self._revoke_depth_linear_authority("late_depth_"+reason)
            return [ControlAction.forward(0, "late_depth_"+reason)], True
        percent = min([previous[1], cap_percent] + [max(0, int(a.speed_percent)) for a in decision.actions if a.kind == "forward"])
        if percent <= 0:
            self._revoke_depth_linear_authority("late_depth_requested_zero")
            return [ControlAction.forward(0, "late_depth_requested_zero")], True
        committed = ("forward", percent, previous[2], previous[3])
        self._publish_depth_linear_pair(committed, replace(timing, snapshot=committed))
        if percent < previous[1]:
            self._lateral_yaw_revision = int(getattr(self, "_lateral_yaw_revision", 0)) + 1
        logger.info(
            "depth_late_sample_continuation uid=%s observation_ts=%s original_sample_ts=%s "
            "old_percent=%s approved_percent=%s deadline=%s pid_updated=False "
            "acceleration_allowed=False deadline_renewed=False watermark_updated=False",
            target_id, stamp, previous[3], previous[1], percent, timing.depth_expires_at,
        )
        return [ControlAction.forward(percent, "depth_late_sample_continuation")], True

    @staticmethod
    def _depth_frame_sample_timestamp(frame: SensorFrame, now: float) -> Optional[float]:
        state = frame.distance_state
        stamp = getattr(state, "sample_timestamp", None)
        try:
            if stamp is None:
                # Compatibility for existing backends/tests without the new
                # field. Never substitute callback time for a missing age.
                age = float(getattr(state, "sample_age_sec", None))
                if not math.isfinite(age) or age < 0.0:
                    return None
                stamp = float(now) - age
            stamp = float(stamp)
            if not math.isfinite(stamp) or stamp <= 0.0 or not (
                0.0 <= float(now) - stamp <= min(ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC,
                                               ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC)
            ):
                return None
        except (ValueError, TypeError, OverflowError):
            return None
        return stamp

    def _preserve_depth_replay(self, frame: SensorFrame, target_id: Optional[int]) -> bool:
        """No new observation is not a zero command; retain the original lease."""
        now = time.monotonic()
        previous = self._fresh_depth_linear_snapshot(target_id, now=now)
        timing = getattr(self, "_depth30_linear_timing", None)
        accepted_stamp = (timing.accepted_depth_timestamp if timing is not None
                          and timing.snapshot == getattr(self, "_depth30_linear_snapshot", None)
                          else None if previous is None else previous[3])
        if (
            previous is None or not frame.distance_state.is_replay_of(accepted_stamp)
            or frame.hazard.active
            or any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
            or not any(p.track_id == target_id for p in frame.persons)
        ):
            return False
        logger.info(
            "depth30_replay_preserve capture_frame_id=%s uid=%s temporal=%s "
            "observation_ts=%s approved_ts=%.6f age_ms=%.1f remaining_ms=%.1f "
            "kind=%s speed=%s accepted_sample_ts=%s deadline_renewed=False pid_updated=False",
            frame.capture_frame_id, target_id, frame.distance_state.temporal_status,
            frame.distance_state.observation_timestamp, previous[3],
            (now - previous[3]) * 1000.0,
            (PersonTracker._depth_linear_max_age_sec(previous[0]) - (now - previous[3])) * 1000.0,
            previous[0], previous[1], accepted_stamp,
        )
        return True

    @staticmethod
    def _depth_update_has_adverse_feedback(frame):
        """Missing feedback is inconclusive; contrary wheel evidence is not."""
        feedback = frame.steering_feedback
        if feedback is None:
            return False
        from car_control_modular.depth_continuation import RELATIVE_CONTINUATION_REVERSE_TAIL_RPM
        values = (feedback.left_forward_rpm, feedback.right_forward_rpm)
        return bool(not feedback.trustworthy
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in values)
            or getattr(feedback, "left_error", 0) or getattr(feedback, "right_error", 0)
            or (min(values) < 0 and max(map(abs, values)) > RELATIVE_CONTINUATION_REVERSE_TAIL_RPM)
            or max(values) > FORWARD_MAX_RPM + 5)

    def _retain_depth_after_update_rejection(self, previous, decision, frame, target_id, reason):
        """A failed positive update is not a stop for an independent live lease.

        Deliberately narrow: closer observations use the existing tightening
        or stop path, never this hold. The old immutable evidence, clock and
        speed are not replaced; even a retained lease must pass every motor
        reader check and its original deadline. Caller owns the control lock.
        """
        if (previous is None or previous[0] != "forward"
                or previous[2] != target_id
                or getattr(self, "_depth30_linear_snapshot", None) is not previous
                or decision.explicit_stop_requested or decision.shutdown_requested
                or decision.soft_stop_requested or not decision.actions
                or any(a.kind != "forward" or a.speed_percent <= 0
                       or a.speed_percent < previous[1] for a in decision.actions)):
            return False
        state = frame.distance_state
        timing = getattr(self, "_depth30_linear_timing", None)
        distances = (frame.distance_m, state.raw_distance_m)
        if (frame.hazard.active
                or any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
                or state.safety_distance_m is not None or state.brake_latched
                or state.target_latched
                or not any(p.track_id == target_id for p in frame.persons)
                or timing is None or timing.snapshot != previous
                or timing.continuation_distance_m is None
                or any(d is None or not math.isfinite(d) for d in distances)
                or min(distances) <= TARGET_DISTANCE + DISTANCE_PID_DEADBAND_M
                or min(distances) < timing.continuation_distance_m):
            return False
        untrusted = getattr(self._follow_controller, "_distance_longitudinally_untrusted", None)
        if callable(untrusted) and untrusted(frame):
            return False
        # This helper also runs BEFORE a new braking assessment is built.
        # A missing distance budget must not hide adverse wheel evidence in
        # the incoming frame behind a different, still-positive old cache.
        # Absent/late feedback may defer an update; explicit reverse motion,
        # faults or overspeed may not. Use the same bounded reverse tail as
        # the physical continuation evidence gate, never a new relaxed limit.
        if PersonTracker._depth_update_has_adverse_feedback(frame):
            return False
        live = PersonTracker._fresh_depth_linear_snapshot(self, target_id, quiet=True)
        if live is None or live[0] != "forward" or live[2:] != previous[2:]:
            return False
        logger.info(
            "depth_update_rejected_preserve uid=%s capture_frame_id=%s reason=%s "
            "rejected_sample_ts=%s retained_sample_ts=%s retained_percent=%s "
            "deadline=%s deadline_renewed=False new_motion_authorized=False",
            target_id, frame.capture_frame_id, reason, state.sample_timestamp,
            previous[3], live[1], timing.depth_expires_at)
        return True

    def _commit_depth_linear_decision(
        self, decision: ControlDecision, frame: SensorFrame, target_id: Optional[int],
        *, is_fresh_depth: bool,
    ) -> Tuple[List[ControlAction], bool]:
        """Cap/record a canonical range decision before either yaw publisher.

        A zero also advances the sample watermark so an older positive range
        cannot restore translation. Nonfresh holds can only reduce an existing
        speed and retain its original deadline.
        """
        actions = list(decision.actions)
        previous_linear = getattr(self, "_depth30_linear_snapshot", None)
        previous_timing = getattr(self, "_depth30_linear_timing", None)
        if is_fresh_depth:
            continuation = self._continue_late_depth_linear_decision(decision, frame, target_id, time.monotonic())
            if continuation is not None:
                return continuation
        if decision.reason == "longitudinal_near_rotation_hold" and not actions:
            actions = [ControlAction.forward(0, decision.reason)]
        if not actions:
            return actions, False
        now = time.monotonic()
        stamp = self._depth_frame_sample_timestamp(frame, now) if is_fresh_depth else None
        watermark = getattr(self, "_depth30_linear_sample_watermark", None)
        pi_enabled = bool(getattr(self._follow_controller, "distance_pi_enabled", False))
        pi_forward_request = pi_enabled and any(a.kind == "forward" for a in actions)
        reject_sample = getattr(self._follow_controller, "reject_longitudinal_sample", None)
        visible = getattr(self, "_validated_visual_observation", None)
        if (is_fresh_depth and isinstance(visible, ValidatedVisualObservation)
                and visible.continuation_sample_timestamp is not None
                and stamp != visible.continuation_sample_timestamp
                and any(a.kind == "forward" and a.speed_percent > 0 for a in actions)):
            # A grey identity frame can only finish the existing grant. A
            # concurrent new depth computation cannot renew it, nor convert
            # this deferred positive computation into an artificial STOP.
            if stamp is not None and callable(reject_sample):
                reject_sample(stamp, reason="identity_recheck_new_depth")
            return [], False
        if is_fresh_depth:
            if stamp is None or target_id is None or (
                watermark is not None and watermark[0] == target_id and stamp <= watermark[1] + 1e-9
            ):
                raw_stamp = frame.distance_state.sample_timestamp
                if (pi_forward_request and stamp is None and callable(reject_sample)
                        and raw_stamp is not None and (watermark is None
                            or watermark[0] != target_id or raw_stamp > watermark[1] + 1e-9)):
                    reject_sample(raw_stamp, reason="depth_admission_expired")
                return [], False
            self._depth30_linear_sample_watermark = (int(target_id), stamp)
        tracking_base = None
        motion_enabled = PersonTracker._longitudinal_target_motion_enabled(self)
        base_getter = getattr(self._follow_controller, "_tracking_base_rpm", None)
        if (motion_enabled and is_fresh_depth and not pi_enabled
                and callable(base_getter) and frame.distance_m is not None):
            tracking_base = base_getter(float(frame.distance_m), now)
        distance_control_percent = None
        ordinary_getter = getattr(self._follow_controller, "distance_only_forward_percent", None)
        motion = (getattr(self._follow_controller, "_longitudinal_motion_evidence", None)
                  if motion_enabled else None)
        pid_result = getattr(self._follow_controller, "last_distance_pid_result", None)
        independent_distance = bool(
            is_fresh_depth and stamp is not None and pid_result is not None
            and getattr(pid_result, "approach_mode", "legacy_pid") != "legacy_pid"
            and getattr(self._follow_controller, "_distance_pid_last_sample_timestamp", None) == stamp
        )
        if (is_fresh_depth and stamp is not None
                and callable(ordinary_getter)
                and (not pi_enabled or pi_forward_request)
                and (pi_enabled or independent_distance or (tracking_base is None
                     and (motion is None or motion.status != "transient_bridge")))):
            # Exact current physical PID sample only; this getter rejects
            # hazards, held/jumping depth and old PID results. PI alone may
            # retain learned integral speed in the zero-error target band.
            distance_control_percent = ordinary_getter(frame, stamp)
        pi_sample_rejected = False
        if pi_forward_request and is_fresh_depth:
            runtime_allowed = bool(
                self._depth_longitudinal_authority_enabled() and not FOLLOW_ROTATION_ONLY
                and self.search_state == "none"
                and getattr(self._follow_controller, "search_state", "none") == "none"
                and getattr(self._follow_controller, "active_target_id", None) == target_id
                and self._vision_control_state.startswith("target_visible")
                and self._vision_control_state != "target_visible_low_quality"
                and not self._explicit_stop_requested and not self._runtime_shutdown_requested
                and self.running and not bool(getattr(self, "_brake_hold_active", False))
                and (visible is None or (isinstance(visible, ValidatedVisualObservation)
                     and visible.permits_depth(target_id, stamp, now)))
            )
            pi_sample_rejected = distance_control_percent is None or not runtime_allowed
            admission_reason = "depth_admission_unqualified"
            if any(a.kind == "forward" and a.speed_percent > 0 for a in actions):
                shared = getattr(pid_result, "pi_braking_assessment", None)
                if not motion_enabled and (shared is not None or PersonTracker._depth_shared_braking_required(self)):
                    from car_control_modular.sample_braking import SampleBrakingAssessment
                    valid_shared = bool(
                        isinstance(shared, SampleBrakingAssessment)
                        and shared.valid_for(target_id, stamp) and shared.target_speed_m_s == 0.
                        and getattr(self._follow_controller, "_distance_pid_last_sample_timestamp", None) == stamp)
                    if not valid_shared:
                        pi_sample_rejected = True
                        admission_reason = ("missing_shared_braking_evidence" if shared is None
                                            else "invalid_shared_braking_evidence")
            if pi_sample_rejected:
                distance_control_percent = 0
                if callable(reject_sample):
                    reject_sample(stamp, reason=admission_reason)
                if PersonTracker._retain_depth_after_update_rejection(
                        self, previous_linear, decision, frame, target_id, admission_reason):
                    return [], False
        pi_budget = None
        if pi_forward_request:
            if is_fresh_depth:
                pi_budget = distance_control_percent or 0
            else:
                # A held/replayed decision can only reduce a still-live grant;
                # it neither integrates PI nor applies a new distance ceiling.
                live = PersonTracker._fresh_depth_linear_snapshot(self, target_id, now=now)
                pi_budget = live[1] if live is not None and live[0] == "forward" else 0
        actions = self._cap_depth_longitudinal_actions(
            actions, distance_m=frame.distance_m, tracking_base_rpm=tracking_base,
            distance_control_percent=distance_control_percent,
            distance_pi_percent=pi_budget,
        )
        motion = (getattr(self._follow_controller, "_longitudinal_motion_evidence", None)
                  if motion_enabled else None)
        authority_stamp = stamp
        ff_origin_stamp = None
        ff_prior_age = getattr(self._follow_controller, "longitudinal_prior_max_age_sec", .18)
        distance_only_percent = 0
        if is_fresh_depth and not pi_enabled and motion is not None and motion.status == "transient_bridge":
            # A bridge is not permission to restart or restore a revoked speed.
            previous = PersonTracker._fresh_depth_linear_snapshot(self, target_id, now=now)
            cap = previous[1] if previous is not None and previous[0] == "forward" else 0
            bridge = getattr(self._follow_controller, "_longitudinal_bridge", None)
            origin = getattr(bridge, "origin", None)
            origin_stamp = getattr(origin, "sample_timestamp", None)
            origin_known = bool(origin_stamp is not None and math.isfinite(origin_stamp)
                                and stamp is not None and now >= origin_stamp)
            if previous is None or previous[0] != "forward" or not origin_known:
                # Old FF cannot resurrect its expired grant. Independently
                # admit a NEW range-only, measured-speed-bounded decision.
                recovery = getattr(self._follow_controller, "fresh_distance_recovery_percent", None)
                recovery_allowed = bool(
                    callable(recovery) and self._depth_longitudinal_authority_enabled()
                    and not FOLLOW_ROTATION_ONLY and self.search_state == "none"
                    and getattr(self._follow_controller, "search_state", "none") == "none"
                    and getattr(self._follow_controller, "active_target_id", None) == target_id
                    and self._vision_control_state.startswith("target_visible")
                    and self._vision_control_state != "target_visible_low_quality"
                    and not self._explicit_stop_requested and not self._runtime_shutdown_requested
                    and self.running and not bool(getattr(self, "_brake_hold_active", False))
                    and not (previous_linear is not None and previous_linear[0] == "backward")
                )
                cap = recovery(frame, stamp, now) if recovery_allowed else 0
                # Respect an existing live lower grant (e.g. a recent slowdown).
                if previous is not None:
                    cap = min(cap, previous[1])
                fallback_actions = self._cap_depth_longitudinal_actions(
                    [ControlAction.forward(cap, "fresh_distance_only_recovery")],
                    distance_m=frame.distance_m, tracking_base_rpm=None,
                    distance_control_percent=cap,
                )
                cap = fallback_actions[0].speed_percent
                tracking_base = None
                distance_control_percent = cap
                logger.info(
                    "depth_ff_detached uid=%s sample_ts=%s reason=%s "
                    "approved_percent=%s old_grant_reused=False new_depth_only=True",
                    target_id, stamp, "old_grant_unavailable" if previous is None else "prior_expired",
                    cap,
                )
            else:
                fallback = getattr(self._follow_controller, "distance_only_forward_percent", None)
                if is_fresh_depth and callable(fallback):
                    fallback_percent = fallback(frame, stamp)
                    fallback_actions = self._cap_depth_longitudinal_actions(
                        [ControlAction.forward(fallback_percent, "distance_only_fallback")],
                        distance_m=frame.distance_m, tracking_base_rpm=None,
                        distance_control_percent=fallback_percent if independent_distance else None,
                    )
                    distance_only_percent = fallback_actions[0].speed_percent
                if now - origin_stamp >= ff_prior_age:
                    # Expiry between PID computation and commit only removes
                    # matching speed, not this new accepted range observation.
                    cap = distance_only_percent if independent_distance else min(cap, distance_only_percent)
                    logger.info(
                        "depth_ff_admission_fallback uid=%s sample_ts=%s ff_origin_ts=%s "
                        "approved_cap_percent=%s pid_updated=False",
                        target_id, stamp, origin_stamp, cap,
                    )
                else:
                    ff_origin_stamp = origin_stamp
                    bridge_getter = getattr(self._follow_controller, "fresh_bridge_forward_percent", None)
                    if independent_distance and callable(bridge_getter):
                        combined_cap = bridge_getter(frame, stamp, now, previous[1]*FORWARD_MAX_RPM/100.)
                        if combined_cap is not None:
                            cap = combined_cap
                            logger.info(
                                "depth_bridge_components uid=%s sample_ts=%s previous_percent=%s "
                                "prior_base_rpm=%s distance_only_percent=%s combined_cap_percent=%s "
                                "cap_scope=prior_only pid_updated=False depth_ttl_ms=%.0f fresh_control_ms=180",
                                target_id, stamp, previous[1], tracking_base, distance_only_percent, cap,
                                ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC*1000.,
                            )
                # Depth remains valid independently of the matching prior.
                # Its extra speed expires separately in the read/write guards.
            actions = [ControlAction.forward(min(a.speed_percent, cap), a.reason)
                       if a.kind == "forward" else a for a in actions]
        linear = actions[0] if len(actions) == 1 else None
        # CAP834: a 4ms lease spent 9ms in the queue and could only write zero.
        # Reserve one actuator cadence (at least 30ms) for positive admission.
        # This is NOT an extension of either physical Depth or detector TTL.
        dispatch_budget = max(0.03, float(MOTOR_RS485_TARGET_MIN_INTERVAL_SEC))
        remaining = None if authority_stamp is None else (
            PersonTracker._depth_linear_max_age_sec(linear.kind if linear is not None else "forward")
            - (time.monotonic() - authority_stamp)
        )
        admission_limited = bool(
            is_fresh_depth and linear is not None
            and linear.kind in {"forward", "backward"} and linear.speed_percent > 0
            and remaining is not None and remaining < dispatch_budget
        )
        if admission_limited:
            previous = PersonTracker._fresh_depth_linear_snapshot(self, target_id)
            can_reduce = bool(previous is not None and previous[0] == linear.kind
                              and linear.speed_percent < previous[1])
            direction_changed = bool(previous is not None and previous[0] != linear.kind)
            logger.info(
                "depth_linear_admission capture_frame_id=%s uid=%s remaining_ms=%.1f "
                "required_ms=%.1f action=%s requested=%s previous=%s deadline_renewed=False",
                frame.capture_frame_id, target_id, remaining * 1000.0,
                dispatch_budget * 1000.0,
                "stop_direction_change" if direction_changed else "reduce_only" if can_reduce else "defer_positive",
                (linear.kind, linear.speed_percent), previous,
            )
            limiter_feedback = getattr(self._follow_controller, "accept_longitudinal_limit", None)
            if pi_forward_request and callable(reject_sample):
                reject_sample(stamp, reason="depth_dispatch_budget")
            if pi_enabled:
                # A late direction change becomes zero below, but that zero
                # is lost authority, not a fresh PI braking approval.
                pi_sample_rejected = True
            if (linear.kind == "forward" and callable(limiter_feedback)
                    and (not pi_enabled or can_reduce)):
                applied_percent = min(linear.speed_percent, previous[1]) if (
                    previous is not None and previous[0] == "forward"
                ) else 0
                limiter_feedback(stamp, applied_percent * FORWARD_MAX_RPM / 100.0)
            if direction_changed:
                # A newly observed approach/reverse request must withdraw old
                # forward motion, even when too late to authorize the reverse.
                linear = ControlAction.forward(0, "depth_dispatch_deadline_direction_change")
                actions = [linear]
            elif not can_reduce:
                return [], False
            else:
                # A safety slowdown remains useful; it inherits the ORIGINAL
                # authorized deadline instead of granting another brief lease.
                is_fresh_depth = False
        pid_result = getattr(self._follow_controller, "last_distance_pid_result", None)
        pid_requested = (
            float(pid_result.output_rpm) if pid_result is not None and stamp is not None
            and getattr(self._follow_controller, "_distance_pid_last_sample_timestamp", None) == stamp
            and linear is not None and linear.kind == "forward" else None
        )
        approved_rpm = (linear.speed_percent * FORWARD_MAX_RPM / 100.0
                        if linear is not None and linear.kind == "forward" else None)
        logger.info(
            "depth_linear_limit capture_frame_id=%s sample_ts=%s fresh=%s distance=%s "
            "tracking_base_rpm=%s requested=%s approved=%s total_cap_percent=%s "
            "sample_gap_ms=%s remaining_ms=%s authority_sample_ts=%s distance_control_percent=%s "
            "forward_scale_rpm=%s approved_forward_rpm=%s physical_depth_ttl_ms=%.0f "
            "uid=%s pid_requested_rpm=%s pid_to_approved_loss_rpm=%s",
            frame.capture_frame_id, stamp, is_fresh_depth, frame.distance_m, tracking_base,
            [(a.kind, a.speed_percent) for a in decision.actions],
            [(a.kind, a.speed_percent) for a in actions],
            ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT,
            None if stamp is None or watermark is None or watermark[0] != target_id
            else round((stamp - watermark[1]) * 1000.0, 1),
            None if authority_stamp is None else round((PersonTracker._depth_linear_max_age_sec(
                linear.kind if linear is not None else "forward") - (now - authority_stamp)) * 1000.0, 1),
            authority_stamp,
            distance_control_percent,
            FORWARD_MAX_RPM,
            int(round(linear.speed_percent * FORWARD_MAX_RPM / 100.0))
            if linear is not None and linear.kind == "forward" else None,
            PersonTracker._depth_linear_max_age_sec(linear.kind if linear is not None else "forward") * 1000.0,
            target_id, pid_requested,
            None if pid_requested is None or approved_rpm is None else max(0.0, pid_requested-approved_rpm),
        )
        limiter_feedback = getattr(self._follow_controller, "accept_longitudinal_limit", None)
        if (is_fresh_depth and not pi_sample_rejected
                and linear is not None and linear.kind == "forward"
                and callable(limiter_feedback)):
            limiter_feedback(stamp, float(linear.speed_percent) * FORWARD_MAX_RPM / 100.0)
        positive = bool(linear is not None and linear.kind in {"forward", "backward"}
                        and int(linear.speed_percent) > 0 and target_id is not None)

        continuation_evidence_veto = None
        def timing_for(candidate):
            nonlocal continuation_evidence_veto
            # Build with locals while the previous complete authority remains
            # readable. No I/O, locks, new PID step, or publication happens here.
            accepted_stamp = stamp
            origin_stamp = ff_origin_stamp
            only_percent = distance_only_percent
            if accepted_stamp is None:
                accepted_stamp = (previous_timing.accepted_depth_timestamp
                                  if previous_timing is not None and previous_timing.snapshot == previous_linear
                                  else candidate[3])
                if previous_timing is not None and previous_timing.snapshot == previous_linear:
                    origin_stamp = previous_timing.feedforward_timestamp
                    only_percent = previous_timing.distance_only_percent
            continuation_distance, continuation_speed, continuation_motion = None, None, None
            braking_assessment = None
            if candidate[0] == "forward" and is_fresh_depth:
                self._depth_continuation_evidence_veto = None
                continuation_distance, continuation_speed = self._depth_continuation_evidence(
                    frame, target_id, candidate[1], now)
                continuation_evidence_veto = getattr(self, "_depth_continuation_evidence_veto", None)
                if continuation_distance is not None and continuation_speed is not None:
                    if motion_enabled:
                        continuation_motion = self._depth_continuation_motion_evidence(
                            frame, target_id, candidate[3], now)
                    from car_control_modular.sample_braking import SampleBrakingAssessment
                    shared = getattr(getattr(self._follow_controller, "last_distance_pid_result", None),
                                     "pi_braking_assessment", None)
                    if (isinstance(shared, SampleBrakingAssessment)
                            and shared.valid_for(target_id, candidate[3])
                            and (continuation_motion is not None
                                 or (not motion_enabled and shared.target_speed_m_s == 0.))):
                        braking_assessment = shared
            elif previous_timing is not None and previous_timing.snapshot == previous_linear:
                continuation_distance = previous_timing.continuation_distance_m
                continuation_speed = previous_timing.continuation_speed_bound_m_s
                continuation_motion = previous_timing.continuation_motion if motion_enabled else None
                braking_assessment = previous_timing.braking_assessment
            return DepthLinearTiming(
                snapshot=candidate, accepted_depth_timestamp=accepted_stamp,
                depth_expires_at=candidate[3] + PersonTracker._depth_linear_max_age_sec(candidate[0]),
                feedforward_timestamp=origin_stamp,
                feedforward_expires_at=None if origin_stamp is None else origin_stamp + ff_prior_age,
                distance_only_percent=min(candidate[1], max(0, int(only_percent))),
                continuation_distance_m=continuation_distance,
                continuation_speed_bound_m_s=continuation_speed,
                continuation_motion=continuation_motion,
                braking_assessment=braking_assessment)

        execution_check = getattr(self._follow_controller, "longitudinal_execution_proof_valid", None)
        publication_preview = None
        if positive and is_fresh_depth:
            candidate = (linear.kind, int(linear.speed_percent), int(target_id), authority_stamp)
            preview_requested_percent = candidate[1]
            candidate_timing = timing_for(candidate)
            continuity_proof = getattr(self._follow_controller, "_distance_pi_execution_anchor_proof", None)
            expiry_proof = getattr(self._follow_controller, "_distance_pi_expiry_execution_proof", None)
            recovery_proof = getattr(self._follow_controller, "_distance_pi_recovery_execution_proof", None)
            proof_pending = bool(pi_forward_request and any(
                proof is not None and proof[0] == stamp
                for proof in (continuity_proof, expiry_proof, recovery_proof)))
            backend = getattr(getattr(self, "_action_runtime", None), "backend", None)
            preview_stop_generation = getattr(backend, "stop_write_generation", None)
            # Feedback/cache access and braking arithmetic stay OUTSIDE the
            # motor lock. The short publication section below rechecks scope,
            # identity, clocks and (when required) the executed receipt.
            publication_preview = PersonTracker._fresh_depth_linear_snapshot(
                self, target_id, now=time.monotonic(), quiet=True,
                _candidate=candidate, _candidate_timing=candidate_timing)
            # Existing order is control lock -> motor lock. Only scalar
            # checks/publication run here, never feedback reads or braking
            # arithmetic. A concurrent STOP cannot slip between them.
            admission_veto_reason = None
            continuation = None
            publication_lock = getattr(self, "motor_io_lock", None)
            with publication_lock if publication_lock is not None else nullcontext():
                publish_now = time.monotonic()
                # Every NEW positive grant must still fit the fresh sample
                # and dispatch windows after evidence/approval work. The
                # The extended reader only continues an already admitted grant.
                # Receipt checks remain conditional and run after time gates;
                # they can mutate PI rejection state as well as return a bool.
                if self._depth_frame_sample_timestamp(frame, publish_now) != stamp:
                    admission_veto_reason = "physical_sample_changed_or_expired"
                elif (candidate[0] == "forward" and PersonTracker._depth_shared_braking_required(self)
                      and candidate_timing.braking_assessment is None):
                    admission_veto_reason = "missing_shared_braking_evidence"
                # When no qualified braking evidence was captured, the motor
                # reader can use this new forward sample for only 180ms. Do
                # not publish it with an extended dispatch budget that disappears
                # immediately at the first motor read (CAP750).
                elif not (stamp + (min(
                          ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC,
                          PersonTracker._depth_linear_max_age_sec(candidate[0]))
                          if candidate[0] == "forward" and (
                              candidate_timing.continuation_distance_m is None or
                              candidate_timing.continuation_speed_bound_m_s is None)
                          else PersonTracker._depth_linear_max_age_sec(candidate[0]))
                          - publish_now >= dispatch_budget):
                    admission_veto_reason = "dispatch_budget"
                elif proof_pending and not (callable(execution_check) and execution_check(stamp, publish_now)):
                    admission_veto_reason = "execution_receipt_invalid"
                elif getattr(backend, "stop_write_generation", None) != preview_stop_generation:
                    admission_veto_reason = "execution_stop_changed"
                elif publication_preview is None:
                    admission_veto_reason = "candidate_qualification"
                elif (not self.running or self._explicit_stop_requested
                      or self._runtime_shutdown_requested
                      or getattr(self, "_brake_hold_active", False)
                      or self.search_state != "none"
                      or getattr(self._follow_controller, "search_state", "none") != "none"
                      or not self._vision_control_state.startswith("target_visible")
                      or self._vision_control_state == "target_visible_low_quality"
                      or getattr(self._follow_controller, "active_target_id", None) != target_id
                      or not motion_identity_live(self, target_id, publish_now)):
                    admission_veto_reason = "candidate_scope_changed"
                if admission_veto_reason is None:
                    candidate = publication_preview
                    candidate_timing = replace(candidate_timing, snapshot=candidate)
                    if candidate[1] < linear.speed_percent:
                        linear = replace(linear, speed_percent=candidate[1])
                        actions = [linear]
                    self._publish_depth_linear_pair(candidate, candidate_timing)
                    consume_execution = getattr(self._follow_controller, "consume_longitudinal_execution_proof", None)
                    if pi_forward_request and callable(consume_execution):
                        consume_execution(stamp)
            if admission_veto_reason is not None:
                publication_preview = None
                rejection_reason = ("execution_packet_changed_before_admission" if proof_pending else
                                    "execution_stop_changed" if admission_veto_reason == "execution_stop_changed" else
                                    "depth_admission_scope_changed" if admission_veto_reason == "candidate_scope_changed" else
                                    "missing_shared_braking_evidence" if admission_veto_reason == "missing_shared_braking_evidence" else
                                    ("relative_depth_admission_veto" if motion_enabled else
                                     "distance_only_depth_admission_veto") if admission_veto_reason == "candidate_qualification" else
                                    "depth_dispatch_budget" if admission_veto_reason == "dispatch_budget" else
                                    "depth_admission_expired")
                if callable(reject_sample):
                    reject_sample(stamp, reason=rejection_reason)
                # Missing/late metadata is not a new stop. A real adverse
                # wheel/obstacle observation is different: it cannot be
                # hidden by an older still-readable cache during fallback.
                soft_evidence_veto = continuation_evidence_veto in {
                    None, "shared_braking_missing_or_mismatched",
                    "distance_or_feedback_unavailable", "feedback_expired_or_future",
                    "feedback_sample_pair_mismatch",
                }
                retain_allowed = (not proof_pending and soft_evidence_veto
                    and not PersonTracker._depth_update_has_adverse_feedback(frame)
                    and admission_veto_reason not in {
                        "execution_stop_changed", "candidate_scope_changed"})
                if (retain_allowed and PersonTracker._retain_depth_after_update_rejection(
                        self, previous_linear, decision, frame, target_id, rejection_reason)):
                    return [], False
                if (retain_allowed and candidate[0] == "forward"
                        and previous_linear is not None and previous_linear[0] == "forward"
                        and frame.distance_state.sample_timestamp == stamp):
                    # A closer observation can tighten a still-live OLD
                    # lease, never renew it. This also runs outside motor I/O.
                    continuation = self._tighten_existing_depth_linear_decision(
                        replace(decision, actions=actions), frame, target_id, time.monotonic())
                if continuation is None or not continuation[1]:
                    self._depth30_linear_snapshot = None
                linear = ControlAction.forward(0, rejection_reason)
                actions = [linear]
                # Format/write after releasing the motor lock. Receipt-based
                # handoffs retain their existing diagnostic and PI reasons.
                logger.info(
                    "%s capture_frame_id=%s uid=%s "
                    "sample_ts=%s reason=%s sample_age_ms=%.1f remaining_ms=%.1f "
                    "budget_ms=%.1f evidence_veto=%s deadline_renewed=False",
                    "depth_execution_admission_veto" if proof_pending else "depth_linear_admission_veto",
                    frame.capture_frame_id, target_id, stamp, admission_veto_reason,
                    (publish_now-stamp)*1000.,
                    (stamp+PersonTracker._depth_linear_max_age_sec(candidate[0])-publish_now)*1000.,
                    dispatch_budget*1000., continuation_evidence_veto,
                )
                if continuation is not None and continuation[1]:
                    return continuation
            elif (candidate[0] == "forward" and candidate[1] < preview_requested_percent
                  and callable(limiter_feedback)):
                limiter_feedback(stamp, float(candidate[1]) * FORWARD_MAX_RPM / 100.)
        elif positive:
            previous = PersonTracker._fresh_depth_linear_snapshot(self, target_id, now=now)
            if previous is not None and previous[0] == linear.kind:
                percent = min(previous[1], int(linear.speed_percent))
                inherited_stamp = previous[3]
                candidate = (linear.kind, percent, int(target_id), inherited_stamp)
                self._publish_depth_linear_pair(candidate, timing_for(candidate))
                actions = [ControlAction.forward(percent, linear.reason) if linear.kind == "forward"
                           else ControlAction.backward(percent, linear.reason, correction_rpm=linear.steer_correction_rpm)]
            else:
                self._depth30_linear_snapshot = None
                actions = [ControlAction.forward(0, "longitudinal_distance_untrusted_hold")]
        else:
            self._depth30_linear_snapshot = None
        committed = self._depth30_linear_snapshot
        if committed is None:
            self._depth30_linear_timing = None
        # A NEW grant was fully checked before publication. Re-reading here
        # used to expose an unqualified candidate and then roll it back to
        # zero. The sole motor writer still rechecks at actual execution time.
        snapshot = (publication_preview if publication_preview is not None else
                    PersonTracker._fresh_depth_linear_snapshot(self, target_id, now=now))
        if (is_fresh_depth and snapshot is None and committed is not None
                and committed[0] == "forward" and self._depth30_linear_timing is not None
                and ((motion_enabled and self._depth30_linear_timing.continuation_motion is not None)
                     or self._depth30_linear_timing.braking_assessment is not None)):
            # Nothing was authorized by the final envelope. Do not leave a
            # positive action/PI execution anchor merely because an earlier
            # arithmetic request was positive. This is admission rejection,
            # not repeated zero-speed anti-windup during feedback loss.
            veto_reason = ("relative_depth_admission_veto" if motion_enabled
                           else "distance_only_depth_admission_veto")
            if callable(reject_sample):
                reject_sample(stamp, reason=veto_reason)
            logger.info("%s uid=%s sample_ts=%s "
                        "requested_percent=%s approved_percent=0 "
                        "limit_scope=final_depth_authority deadline_renewed=False",
                        "depth_relative_admission_veto" if motion_enabled else "depth_distance_only_admission_veto",
                        target_id, stamp, committed[1])
            actions = [ControlAction.forward(0, veto_reason)]
            self._depth30_linear_snapshot = None
            self._depth30_linear_timing = None
            committed = None
        elif (is_fresh_depth and snapshot is not None and committed is not None
                and snapshot[0] == "forward" and snapshot[1] < committed[1]
                and self._depth30_linear_timing is not None
                and ((motion_enabled and self._depth30_linear_timing.continuation_motion is not None)
                     or self._depth30_linear_timing.braking_assessment is not None)):
            # The continuous envelope also applies at admission. Publish its
            # actual initial cap back to PI rather than approving116 while the
            # sole writer can only execute100. Keep the ORIGINAL higher speed
            # bound for past travel; a lowered request cannot rewrite history.
            requested_percent = committed[1]
            committed = snapshot
            self._publish_depth_linear_pair(committed, replace(self._depth30_linear_timing, snapshot=committed))
            actions = [ControlAction.forward(snapshot[1], "relative_depth_admission_limit" if motion_enabled
                                            else "distance_only_depth_admission_limit")]
            if not pi_sample_rejected and callable(limiter_feedback):
                limiter_feedback(stamp, float(snapshot[1])*FORWARD_MAX_RPM/100.)
            logger.info("%s uid=%s sample_ts=%s requested_percent=%s "
                        "approved_percent=%s original_speed_bound_m_s=%s "
                        "limit_scope=final_depth_authority deadline_renewed=False",
                        "depth_relative_admission_limit" if motion_enabled else "depth_distance_only_admission_limit",
                        target_id, stamp, requested_percent, snapshot[1],
                        self._depth30_linear_timing.continuation_speed_bound_m_s)
        self._current_forward_allow_below_min = snapshot is not None
        if snapshot is not None:
            # A TURN packet can already be waiting at the motor lock when a
            # Depth update arrives. Revoke that old pure-yaw authority now,
            # before the next lateral tick has a chance to compose both axes.
            self._current_rotate_raw_target = 0
            self._current_rotate_turn_percent = 0
            self._current_rotate_raw_source = "lateral_intent_revoked"
            self._current_rotate_pulse_enabled = False
        else:
            # Revoking the range authority must invalidate a DRIVE/STEER
            # packet already prepared by the motor thread, not just the next
            # lateral publication. Only withdraw translation here: the fresh
            # yaw intent and its correction remain owned by the visual loop.
            self._current_forward_percent = 0
            self._current_steer_base_percent = 0
            self._forward_speed_latched_percent = None
            self.is_forwarding = False
        # Duplicate/older physical samples returned above without committing;
        # every accepted axis change, including zero, revokes earlier packets.
        self._lateral_yaw_revision = int(getattr(self, "_lateral_yaw_revision", 0)) + 1
        self._last_depth30_translation_kind = None if snapshot is None else snapshot[0]
        self._last_depth30_translation_ts = 0.0 if snapshot is None else snapshot[3]
        logger.info(
            "depth_linear_authority capture_frame_id=%s uid=%s fresh=%s sample_ts=%s "
            "actions=%s snapshot=%s previous=%s reason=%s accepted_sample_ts=%s "
            "ff_origin_ts=%s depth_expires_at=%s ff_expires_at=%s fallback_percent=%s "
            "temporal_status=%s observation_ts=%s previous_remaining_ms=%s",
            frame.capture_frame_id, target_id, is_fresh_depth, stamp,
            [(a.kind, a.speed_percent) for a in actions], self._depth30_linear_snapshot,
            previous_linear, decision.reason,
            None if committed is None else self._depth30_linear_timing.accepted_depth_timestamp,
            None if committed is None else self._depth30_linear_timing.feedforward_timestamp,
            None if committed is None else self._depth30_linear_timing.depth_expires_at,
            None if committed is None else self._depth30_linear_timing.feedforward_expires_at,
            None if committed is None else self._depth30_linear_timing.distance_only_percent,
            frame.distance_state.temporal_status or "unspecified",
            frame.distance_state.observation_timestamp,
            None if previous_linear is None else round(1000.0 * (
                previous_linear[3] + PersonTracker._depth_linear_max_age_sec(previous_linear[0]) - now), 1),
        )
        return actions, True

    def _observe_follow_distance_hold(self, frame, target, *, fresh, steerable, low_quality):
        """Observe while parked; only a typed ordinary hold can be released.

        Called under control-update lock, before longitudinal PID/authorization.
        Never emits a motor packet or renews an old depth lease.
        """
        if FOLLOW_ROTATION_ONLY or not is_follow_distance_hold(self):
            return False
        hold = self._follow_distance_hold
        state = frame.distance_state
        now = time.monotonic()
        obs = None if target is None else target.depth_observation
        feedback = frame.steering_feedback
        qualified = bool(
            steerable and not low_quality
            and target is not None and target.track_id == hold.uid
            and sum(p.track_id == hold.uid for p in frame.persons) == 1
            and obs is not None and obs.source == "yolo_detector" and obs.target_id == hold.uid
            and obs.capture_frame_id == frame.capture_frame_id
            and obs.capture_timestamp == frame.capture_timestamp
            and 0 <= now-obs.capture_timestamp <= ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC
            and not frame.hazard.active
            and not any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
            and not state.brake_latched and not state.target_latched
            and state.safety_distance_m is None
            and feedback is not None and feedback.trustworthy
            and math.isfinite(feedback.timestamp) and 0 <= now-feedback.timestamp <= .15
            and all(math.isfinite(v) and v >= -1 for v in
                    (feedback.left_forward_rpm, feedback.right_forward_rpm))
        )
        if (qualified and not fresh and state.is_replay_of(hold.last_stamp)
                and 0 <= now-hold.last_stamp <= .18):
            status = "duplicate"
        else:
            status = hold.observe(stamp=state.sample_timestamp, raw=state.raw_distance_m,
                                  used=state.used_distance_m, now=now, target=TARGET_DISTANCE,
                                  brake=FOLLOW_BRAKE_DISTANCE_M, qualified=qualified and fresh)
        logger.info(
            "follow_distance_hold_observe capture_frame_id=%s uid=%s status=%s count=%d "
            "sample_ts=%s raw=%s used=%s held_ms=%.1f motor_authorized=False",
            frame.capture_frame_id, hold.uid, status, hold.count, state.sample_timestamp,
            state.raw_distance_m, state.used_distance_m, (now-hold.started)*1000,
        )
        if status != "ready":
            return False
        # Recheck the actual safety gate; failure must not unlock anything.
        try:
            if self._action_runtime.hard_stop_check(ACTION_FORWARD):
                hold.reject()
                return False
        except Exception:
            hold.reject()
            return False
        with self.motor_io_lock:
            if (getattr(self, "_follow_distance_hold", None) is not hold
                    or not is_follow_distance_hold(self)
                    or not 0 <= time.monotonic()-state.sample_timestamp <= .18):
                return False
            self._brake_hold_active = False
            self._follow_distance_hold = None
            self._brake_hold_stop_mode = None
            self._brake_hold_label = "brake"
            self._last_brake_hold_send_ts = 0.0
            self.stop_action_execution = False
            self.person_detected_flag = False
            # Old motor authorization is not the recovery evidence.
            self._depth30_linear_snapshot = None
            self._depth30_linear_timing = None
            self._lateral_yaw_revision = getattr(self, "_lateral_yaw_revision", 0)+1
        logger.info(
            "follow_distance_hold_released capture_frame_id=%s uid=%s sample_ts=%s "
            "held_ms=%.1f next=canonical_depth_pid old_authority_restored=False",
            frame.capture_frame_id, hold.uid, state.sample_timestamp, (now-hold.started)*1000,
        )
        return True

    def _refresh_visual_depth_linear_authority(
        self, frame: SensorFrame, target: Optional[PersonTarget], decision: ControlDecision,
        *, is_fresh_depth: bool, target_steerable: bool, low_quality_visible: bool,
    ) -> bool:
        """Use a new RGB-aligned Depth sample without replaying visual decisions."""
        if (
            not self._depth_longitudinal_authority_enabled() or FOLLOW_ROTATION_ONLY
            or not is_fresh_depth or target is None or not target_steerable or low_quality_visible
            or getattr(self._follow_controller, "active_target_id", None) != target.track_id
            or self.search_state != "none"
            or getattr(self._follow_controller, "search_state", "none") != "none"
            or not self._vision_control_state.startswith("target_visible")
            or decision.explicit_stop_requested or decision.shutdown_requested
            or self._explicit_stop_requested or self._runtime_shutdown_requested or not self.running
            or getattr(self, "_near_yaw_park_request", None) is not None
            or frame.hazard.active or frame.obstacles.front or frame.obstacles.left or frame.obstacles.right
            or bool(getattr(frame.distance_state, "brake_latched", False))
        ):
            return False
        stamp = self._depth_frame_sample_timestamp(frame, time.monotonic())
        watermark = getattr(self, "_depth30_linear_sample_watermark", None)
        if stamp is None or (watermark is not None and watermark[0] == target.track_id
                             and stamp <= watermark[1] + 1e-9):
            return False
        canonical = getattr(self._follow_controller, "_longitudinal_only_decision", None)
        if not callable(canonical):
            return False
        if bool(getattr(self, "_brake_hold_active", False)):
            # Search NORMAL used to wait for a forward command while this
            # admission gate waited for NORMAL to release (CAP357 onward).
            # Only this independently qualified NEW observation may bridge
            # the handoff; every other parking owner keeps its old semantics.
            observation = target.depth_observation
            state = frame.distance_state
            distances = (frame.distance_m, state.raw_distance_m)
            trusted = getattr(self._follow_controller, "_distance_longitudinally_untrusted", None)
            eligible = bool(
                getattr(self, "_brake_hold_label", "") == "search_reacquire_brake"
                and sum(p.track_id == target.track_id for p in frame.persons) == 1
                and observation is not None and observation.source == "yolo_detector"
                and observation.target_id == target.track_id
                and observation.capture_frame_id == frame.capture_frame_id
                and observation.capture_timestamp == frame.capture_timestamp
                and state.sample_timestamp is not None
                and not state.target_latched and state.safety_distance_m is None
                and all(d is not None and math.isfinite(d)
                        and d > TARGET_DISTANCE + DISTANCE_PID_DEADBAND_M for d in distances)
                and not (callable(trusted) and trusted(frame))
                and any(a.kind in {"forward", "steer_left", "steer_right"}
                        and a.speed_percent > 0 for a in decision.actions)
            )
            release = getattr(getattr(self, "_action_runtime", None),
                              "release_settled_search_brake_for_depth", None)
            if not eligible or not callable(release) or not release(
                uid=target.track_id, capture_id=frame.capture_frame_id,
                capture_timestamp=frame.capture_timestamp, sample_timestamp=stamp,
                depth_max_age=min(ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC,
                                  ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC),
                image_max_age=VISION_CONTROL_MAX_RESULT_AGE_SEC,
            ):
                return False
        last_action_frame = getattr(self._follow_controller, "last_action_frame", -1)
        visual_pid = getattr(self._follow_controller, "last_steering_pid_result", None)
        try:
            longitudinal = canonical(self.frame_index, frame, target, target_steerable=True)
        finally:
            self._follow_controller.last_action_frame = last_action_frame
            self._follow_controller.last_steering_pid_result = visual_pid
        _actions, committed = self._commit_depth_linear_decision(
            longitudinal, frame, target.track_id, is_fresh_depth=True,
        )
        return committed

    def _clear_lateral_intent(self, reason: str) -> None:
        previous = self._lateral_intent_store.clear()
        if self._lateral_intent_store.snapshot() is None:
            self._lateral_turn_response_policy = None
        self._lateral_direction_intent = "hold"
        self._lateral_intent_owned_frame = -1
        if previous is not None:
            # Handoffs to search/safety keep their own action policy. In the
            # visible state, withdrawing ownership must also withdraw its yaw.
            self._publish_lateral_zero(previous, "revoke:" + str(reason))
            logger.info(
                "lateral_intent_clear seq=%d target=%d frame=%d reason=%s age=%.0fms",
                int(previous.sequence),
                int(previous.target_id),
                int(previous.frame_index),
                str(reason or "unspecified"),
                previous.age_sec(time.monotonic()) * 1000.0,
            )

    def _visible_rotate_command_allowed(self, action: int) -> bool:
        """Last-moment motor gate for visible yaw, never for search pulses.

        Called with motor_io_lock held; do not acquire the control/command
        locks here (the producer may be waiting for the motor thread).
        """
        if getattr(self._follow_controller, "active_target_id", None) is None:
            return False  # No startup search, including legacy/default packets.
        source = str(getattr(self, "_current_rotate_raw_source", ""))
        if source == "stale_vision_zero_yaw":
            return False
        if source not in {"lateral_intent_30hz", "lateral_intent_revoked"}:
            return True
        intent = self._lateral_intent_store.snapshot()
        correction = int(getattr(self, "_lateral_intent_last_correction_rpm", 0))
        return bool(
            intent is not None
            and intent.valid(time.monotonic())
            and self._has_fresh_lateral_yaw(intent.target_id)
            and not intent.hold_zero
            and int(getattr(self, "_lateral_intent_last_sequence", -1)) == intent.sequence
            and getattr(self._follow_controller, "active_target_id", None) == intent.target_id
            and self.search_state == "none"
            and self._vision_control_state.startswith("target_visible")
            and not self._explicit_stop_requested
            and not self._runtime_shutdown_requested
            and self.running
            and int(getattr(self, "_current_rotate_raw_target", 0)) > 0
            and ((action == ACTION_ROTATE_RIGHT and correction > 0)
                 or (action == ACTION_ROTATE_LEFT and correction < 0))
        )

    def _retire_near_yaw_park_commands(self) -> None:
        """Caller holds command -> motor locks; retire even popped old work by epoch."""
        self.current_command = None
        self.command_start_time = None
        with getattr(self, "action_queue_lock", nullcontext()):
            self._near_yaw_park_generation = int(getattr(self, "_near_yaw_park_generation", 0)) + 1
            pending = getattr(self, "action_queue", None)
            if pending is not None:
                while True:
                    try:
                        pending.get_nowait()
                    except queue.Empty:
                        break
        self._last_dispatched_action = ACTION_STOP
        self._lateral_intent_owned_frame = -1

    def _request_near_yaw_park(self, intent: LateralControlIntent, reason: str) -> None:
        """Latch intent only. The action thread owns the NORMAL stop write."""
        now = time.monotonic()
        with getattr(self, "command_lock", nullcontext()), getattr(self, "motor_io_lock", nullcontext()):
            request = getattr(self, "_near_yaw_park_request", None)
            if (getattr(self, "_brake_hold_active", False)
                    and getattr(self, "_brake_hold_label", "") != "near_yaw_park"):
                return  # Never downgrade a safety/other parking owner.
            if request is None:
                self._retire_near_yaw_park_commands()
                request = NearYawParkRequest(
                    intent.target_id, intent.capture_frame_id,
                    intent.capture_timestamp, now, str(reason),
                    countersteer_rpm=int(intent.countersteer_rpm),
                    visual_error_deg=float(intent.visual_error_deg or 0.0),
                    image_rate_dps=float(intent.target_image_rate_dps or 0.0),
                    countersteer_until=min(intent.valid_until,
                                           intent.capture_timestamp + MAX_CAPTURE_AGE_SEC),
                )
                self._near_yaw_park_request = request
                parking = getattr(self._follow_controller, "set_normal_parking", None)
                if callable(parking):
                    parking(True, request.uid)
                self._near_yaw_park_evidence = (intent.capture_frame_id, intent.capture_timestamp)
                logger.info(
                    "near_yaw_park_requested uid=%s capture_frame_id=%s reason=%s "
                    "base_rpm=0 yaw_rpm=0 mode=runtime_selected",
                    intent.target_id, intent.capture_frame_id, reason,
                )
            else:
                reference = getattr(self, "_near_yaw_park_evidence",
                                    (request.capture_frame_id, request.capture_timestamp))
                if (intent.capture_frame_id > reference[0]
                        and intent.capture_timestamp > reference[1]):
                    # Advance only the release barrier, not the stop episode.
                    self._near_yaw_park_evidence = (intent.capture_frame_id, intent.capture_timestamp)
            self._use_soft_stop_next = False
            self.stop_action_execution = False
            self.person_detected_flag = False
            self._lateral_yaw_revision = int(getattr(self, "_lateral_yaw_revision", 0)) + 1

    def _release_near_yaw_park(
        self, *, capture_id: int, capture_timestamp: float, reason: str,
        target_id: Optional[int] = None, qualified: bool = False,
        handoff: bool = False,
        x_ratio: Optional[float] = None, translation_requested: bool = False,
        outward_lead=None,
        forward_resume_sample_ts=None,
    ) -> bool:
        """Only newer physical visual evidence can release this ordinary hold.

        Explicit search/identity handoff keeps its original motion policy.
        Release grants no longitudinal authorization and performs no motor I/O.
        """
        request = getattr(self, "_near_yaw_park_request", None)
        if request is None:
            return False
        with getattr(self, "command_lock", nullcontext()), getattr(self, "motor_io_lock", nullcontext()):
            reference = getattr(self, "_near_yaw_park_evidence",
                                (request.capture_frame_id, request.capture_timestamp))
            if (getattr(self, "_near_yaw_park_request", None) is not request
                    or not newer_visual_evidence(capture_id, capture_timestamp, reference,
                                                 time.monotonic(), VISION_CONTROL_MAX_RESULT_AGE_SEC)
                    or not self.running or self._explicit_stop_requested
                    or self._runtime_shutdown_requested
                    or not (handoff or (qualified and target_id == request.uid))):
                return False
            if (getattr(self, "_brake_hold_active", False)
                    and getattr(self, "_brake_hold_label", "") != "near_yaw_park"):
                return False
            trajectory_release = False
            if not handoff and not translation_requested:
                cfg = getattr(self._follow_controller, "cfg", None)
                margin = max(0.0, float(getattr(cfg, "visible_steering_pid_image_center_release_margin_deg", 1.8)))
                hfov = max(1.0, float(getattr(cfg, "visible_steering_pid_camera_hfov_deg", 60.0)))
                left = float(getattr(cfg, "center_left_ratio", .45)) - margin / hfov
                right = float(getattr(cfg, "center_right_ratio", .55)) + margin / hfov
                if x_ratio is None or not math.isfinite(x_ratio):
                    return False
                if left <= x_ratio <= right:
                    # Only exact new capture evidence can take this narrow
                    # early-release route. Require the full quiet/minimum-hold gate.
                    evidence = getattr(getattr(self, "_action_runtime", None),
                                       "_near_yaw_park_settling", None)
                    trajectory_release = bool(
                        qualified and outward_lead is not None
                        and getattr(cfg, "visible_steering_pid_outward_lead_enable", False)
                        and outward_lead.matches(target_id, capture_id, capture_timestamp, time.monotonic())
                        and abs(x_ratio-outward_lead.x) <= .06
                        and (x_ratio-.5)*outward_lead.correction_rpm > 0
                        and evidence is not None and evidence.request is request
                        and outward_lead.first_timestamp > evidence.sent_at)
                if left <= x_ratio <= right and not trajectory_release:
                    logger.info("near_yaw_park_release_blocked reason=center_release_band x=%s band=%.3f/%.3f", x_ratio, left, right)
                    return False
            # Pure recentering must finish the previous stop before restarting
            # either direction. Qualified translation retains its independent
            # handoff so centered forward recovery does not deadlock on this.
            resume = getattr(getattr(self, "_action_runtime", None), "request_near_yaw_forward_resume", None)
            if (not handoff and qualified and translation_requested
                    and forward_resume_sample_ts is not None and callable(resume)):
                resume(request, capture_timestamp, forward_resume_sample_ts, time.monotonic())
            ready = getattr(getattr(self, "_action_runtime", None),
                            "near_yaw_park_release_ready" if handoff or not translation_requested else "near_yaw_park_motion_ready", None)
            if not callable(ready) or not ready(request, capture_timestamp, time.monotonic()):
                return False
            if trajectory_release:
                reason += ":outward_trajectory"
            settling = getattr(getattr(self, "_action_runtime", None),
                               "_near_yaw_park_settling", None)
            if (not handoff and not translation_requested and settling is not None
                    and settling.request is request and settling.ready_at is not None):
                note_settled = getattr(self._follow_controller, "note_post_park_settled", None)
                if callable(note_settled):
                    note_settled(request.uid, settling.ready_at,
                                 settling.ready_yaw_right_deg)
            self._retire_near_yaw_park_commands()
            self._lateral_intent_store.clear()
            self._near_yaw_park_request = None
            parking = getattr(self._follow_controller, "set_normal_parking", None)
            if callable(parking):
                parking(False, request.uid)
            if getattr(self, "_brake_hold_label", "") == "near_yaw_park":
                self._brake_hold_active = False
                self._brake_hold_stop_mode = None
                self._brake_hold_label = "brake"
                self._last_brake_hold_send_ts = 0.0
                self.stop_action_execution = False
                self.person_detected_flag = False
                self._soft_stop_active = False
            self._lateral_yaw_revision = int(getattr(self, "_lateral_yaw_revision", 0)) + 1
        logger.info(
            "near_yaw_park_released uid=%s capture_frame_id=%s parked_capture=%s "
            "held_ms=%.1f reason=%s handoff=%s old_authority_restored=False",
            request.uid, capture_id, reference[0],
            (time.monotonic()-request.requested_at)*1000, reason, handoff,
        )
        return True

    def _release_near_yaw_park_for_decision(
        self, frame: SensorFrame, target: Optional[PersonTarget], decision: ControlDecision,
        *, control_source: str, target_steerable: bool, low_quality_visible: bool,
    ) -> bool:
        """Release before committing NEW Depth, including straight-ahead restart."""
        if getattr(getattr(self._follow_controller, "last_steering_pid_result", None),
                   "output_floor_reason", "") == "predictive_countersteer":
            return False  # Braking evidence cannot unlock an existing NORMAL.
        if (getattr(self, "_near_yaw_park_request", None) is None
                or control_source != "vision" or target is None
                or decision.explicit_stop_requested or decision.shutdown_requested
                or getattr(decision, "near_yaw_park_requested", False)
                or frame.hazard.active
                or any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
                or frame.distance_state.brake_latched
                or frame.distance_state.safety_distance_m is not None
                or self.search_state != "none"):
            return False
        motion = any(
            action.kind in ("rotate_left", "rotate_right")
            or (action.kind in ("forward", "backward", "steer_left", "steer_right")
                and (action.speed_percent > 0 or action.steer_correction_rpm != 0))
            for action in decision.actions
        )
        translation = any(
            action.kind in ("forward", "backward", "steer_left", "steer_right")
            and action.speed_percent > 0 for action in decision.actions)
        c = self._follow_controller
        state = frame.distance_state
        stamp = state.sample_timestamp
        preview_demand = 0.0
        resume_demand = getattr(c, "parked_forward_resume_demand_rpm", None)
        if (not translation and not FOLLOW_ROTATION_ONLY
                and target_steerable and not low_quality_visible
                and callable(resume_demand)):
            # CAP2627/3299: a parked PI has zero executable base, even when
            # its fresh, brake-bounded forward demand is positive. A lateral
            # correction must not hide that independent demand and force the
            # whole chassis through the pure-recentering dwell/center gate.
            # Existing positive translation (especially reverse) keeps its
            # original path. This hint releases only ordinary parking; a new
            # physical sample is still required for forward motor authority.
            preview_demand = resume_demand(frame, target.track_id)
        if not motion and preview_demand <= 0:
            return False
        forward_requested = bool(preview_demand > 0 or any(
            a.kind in ("forward", "steer_left", "steer_right") and a.speed_percent > 0
            for a in decision.actions))
        # Pure-distance control deliberately disables the target-speed
        # estimator. A NEW brake-qualified parked PI demand can release an
        # ordinary yaw hold without requiring that unavailable estimator.
        # This only requests current release; post-stop evidence, 0A/FREE,
        # new forward authorization and the wheel reversal guard still apply.
        distance_only_demand = bool(
            preview_demand > 0
            and getattr(getattr(c, "cfg", None),
                        "distance_target_motion_control_enable", True) is False)
        early_forward = bool(
            target_steerable and not low_quality_visible
            and (distance_only_demand or (
                getattr(c, "_braking_rate_source", None) == "raw_depth_window"
                and getattr(c, "_braking_range_rate", 0.) > 0.))
            and isinstance(stamp, (int, float)) and math.isfinite(stamp)
            and 0 <= time.monotonic()-stamp <= .18 and c._is_fresh_depth_state(frame)
            and not c._distance_longitudinally_untrusted(frame)
            and all(v is not None and math.isfinite(v)
                    and v > c.cfg.target_distance_m+c.cfg.distance_pid_deadband_m
                    for v in (frame.distance_m, state.raw_distance_m))
            and forward_requested)
        released = self._release_near_yaw_park(
            capture_id=int(getattr(self, "_last_command_capture_frame", frame.capture_frame_id)),
            capture_timestamp=frame.capture_timestamp,
            target_id=target.track_id,
            qualified=bool(target_steerable or (low_quality_visible and target_steerable is False
                                                and any(a.kind in ("rotate_left", "rotate_right")
                                                        and a.speed_percent > 0 for a in decision.actions))),
            reason="new_visual_decision:" + str(decision.reason),
            x_ratio=target.center[0] / max(1, frame.width),
            translation_requested=bool(preview_demand > 0 or any(
                a.kind in ("forward", "backward", "steer_left", "steer_right")
                and a.speed_percent > 0 for a in decision.actions)),
            outward_lead=(getattr(self._follow_controller.last_steering_pid_result, "outward_lead", None)
                          if target_steerable and not low_quality_visible else None),
            forward_resume_sample_ts=stamp if early_forward else None,
        )
        if preview_demand > 0:
            logger.info(
                "near_yaw_park_resume_demand uid=%s capture_frame_id=%s sample_ts=%s "
                "brake_limited_demand_rpm=%.2f preview_output_rpm=%s released=%s "
                "motion_authorized=False reason=fresh_parked_pi_demand "
                "lateral_motion_present=%s translation_present=%s early_forward=%s",
                target.track_id, frame.capture_frame_id, stamp, preview_demand,
                getattr(getattr(self._follow_controller, "last_distance_pid_result", None), "output_rpm", None), released,
                motion, translation, early_forward,
            )
        return released

    def _publish_lateral_zero(
        self, intent: LateralControlIntent, reason: str, *, park_requested: bool = False,
        pid_result=None,
    ) -> bool:
        """Revoke yaw through the single action executor, preserving Depth v.

        Caller holds _control_update_lock. No motor I/O is done here. Search,
        shutdown and hard safety retain their existing direct-stop authority.
        """
        short = getattr(self, "_short_follow_adapter", None)
        if short is not None and short.owned:
            return False  # An obsolete split-axis update cannot own the pair.
        if (
            self.search_state != "none"
            or not self._vision_control_state.startswith("target_visible")
            or self._explicit_stop_requested
            or self._runtime_shutdown_requested
            or not self.running
        ):
            return False
        # Legacy/direct adapters do not publish an intent store. They can
        # still revoke their own yaw; canonical runtime checks the current
        # immutable generation when the store is available.
        intent_store = getattr(self, "_lateral_intent_store", None)
        current_intent = intent_store.snapshot() if intent_store is not None else None
        if (getattr(self._follow_controller, "active_target_id", None) != intent.target_id
                or (current_intent is not None and current_intent.sequence > intent.sequence)):
            # A delayed cancellation belongs to its original yaw generation,
            # not to a newer target/intent (and never to its depth grant).
            logger.info("lateral_zero_superseded seq=%s capture_frame_id=%s uid=%s "
                        "current_sequence=%s current_uid=%s axes_unchanged=True",
                        intent.sequence, intent.capture_frame_id, intent.target_id,
                        None if current_intent is None else current_intent.sequence,
                        getattr(self._follow_controller, "active_target_id", None))
            return False
        now = time.monotonic()
        self._last_vision_correction_rpm = 0
        self._last_vision_correction_at = now
        self._last_vision_correction_target_id = int(intent.target_id)
        self._lateral_intent_last_correction_rpm = 0
        self._lateral_intent_zero_sequence = int(intent.sequence)
        self._lateral_turn_response_policy = (
            None if current_intent is None else (int(intent.sequence), False))
        self._lateral_direction_intent = "hold"
        self._current_steer_correction_rpm = 0
        self._current_steer_inner_ratio_percent = 100
        self._current_steer_outer_ratio_percent = 100
        self._current_rotate_raw_target = 0
        self._current_rotate_turn_percent = 0
        self._current_rotate_raw_source = "lateral_intent_revoked"
        self._current_rotate_pulse_enabled = False

        linear = PersonTracker._fresh_depth_linear_snapshot(self, intent.target_id, now=now)
        visual = getattr(self, "_validated_visual_observation", None)
        newer_depth_context = bool(
            linear is not None and isinstance(visual, ValidatedVisualObservation)
            and visual.permits_depth(intent.target_id, linear[3], time.monotonic())
            and visual.capture > intent.capture_frame_id
            and visual.timestamp > intent.capture_timestamp
            # RGB-aligned Depth may physically precede its RGB exposure by
            # a small amount. It was independently admitted for this UID;
            # it must supersede the cancelled yaw's evidence, not be newer
            # than the unrelated RGB sensor clock. Grey rechecks remain
            # bound to their original physical sample by permits_depth.
            and linear[3] > intent.capture_timestamp)
        # A center/predictive yaw stop is only an axis update during ordinary
        # forward tracking. Residual pivot feedback must not convert a NEW,
        # independently qualified small forward grant into a NORMAL parking
        # latch just because its request is <=10 RPM. The final wheel writer
        # still owns actual reversal/zero-crossing protection. Explicit near
        # parking, an existing parking owner and unqualified Depth stay on
        # their original stop paths.
        forward_axis_only = bool(
            intent.mode == "forward" and not intent.near_distance_mode
            and intent.bbox_quality == "reliable"
            and linear is not None and linear[0] == "forward")
        stop_turn = False
        if ((intent.park_requested or park_requested)
                and not intent.distance_control_deferred
                and not newer_depth_context and not forward_axis_only
                and getattr(self, "_near_yaw_park_request", None) is None):
            reader = getattr(getattr(self, "_action_runtime", None), "get_steering_feedback", None)
            feedback = reader() if callable(reader) else None  # existing encoder cache only
            linear_rpm = (0.0 if linear is None else
                          float(linear[1]) * FORWARD_MAX_RPM / 100.
                          * (1 if linear[0] == "forward" else -1))
            stop_reason = low_speed_turn_stop_reason(
                intent, feedback, now=now,
                active_uid=getattr(self._follow_controller, "active_target_id", None),
                linear_rpm=linear_rpm, max_image_age=VISION_CONTROL_MAX_RESULT_AGE_SEC,
            )
            stop_turn = stop_reason == "low_speed_turn_stop"
            logger.info(
                "lateral_turn_stop_gate capture_frame_id=%s uid=%s reason=%s "
                "near_mode=%s linear_rpm=%.1f feedback_left=%s feedback_right=%s "
                "feedback_age_ms=%s",
                intent.capture_frame_id, intent.target_id, stop_reason,
                intent.near_distance_mode, linear_rpm,
                getattr(feedback, "left_forward_rpm", None),
                getattr(feedback, "right_forward_rpm", None),
                (now-feedback.timestamp)*1000 if feedback is not None else None,
            )
            if stop_turn:
                # Latch before another depth/periodic writer can restore a
                # tiny translation. The executor owns the actual NORMAL I/O.
                self._request_near_yaw_park(intent, "low_speed_turn_stop:" + str(reason))
        preserve = bool(
            linear is not None and (intent.bbox_quality == "reliable" or newer_depth_context)
            and getattr(self, "_near_yaw_park_request", None) is None
        )
        # Only a transient read/metadata gap may use zero-axis STEER. A
        # same-grant continuation veto or explicit zero braking cap keeps the
        # original STOP/revoke path. The periodic motor writer must still pass
        # _fresh on every positive packet; the canonical tuple cannot move it.
        pending_depth = (PersonTracker._pending_forward_depth_lease(
            self, intent.target_id, now) if not preserve and not stop_turn
            and intent.bbox_quality == "reliable"
            and not (intent.near_distance_mode and
                     (intent.park_requested or park_requested)) else None)
        if (not preserve and pending_depth is None
                and getattr(self, "_depth30_linear_snapshot", None) is not None):
            withdrawal_reason = (("lateral_low_speed_turn_stop:" if stop_turn else
                                  "lateral_zero_no_qualified_depth:") + str(reason))
            if not stop_turn:
                withdrawal_reason = PersonTracker._depth_linear_withdrawal_reason(
                    self, self._depth30_linear_snapshot, now, withdrawal_reason)
            self._revoke_depth_linear_authority(
                withdrawal_reason)
        self._current_forward_allow_below_min = preserve
        if preserve:
            kind, percent, _target_id, _stamp = linear
            self._current_forward_percent = percent
            self._current_steer_base_percent = percent if kind == "forward" else 0
            self.is_forwarding = kind == "forward"
            # Equal-wheel STEER preserves even a below-launch-floor Depth
            # speed. Converting that to DRIVE could raise it to the min RPM.
            action = ACTION_STEER_RIGHT if kind == "forward" else ACTION_BACKWARD
            self._use_soft_stop_next = False
        else:
            self._current_forward_percent = 0
            self._current_steer_base_percent = 0
            self.is_forwarding = False
            # STEER with zero axes retains the visible follow writer. Its
            # read-time Depth gate decides whether zero remains or a later
            # valid continuation can move; STOP would switch motor mode and
            # force the next fresh grant through a full restart.
            action = ACTION_STEER_RIGHT if pending_depth is not None else ACTION_STOP
            self._use_soft_stop_next = (False if pending_depth is not None else
                                        not bool(getattr(self, "_brake_hold_active", False)))
            if not stop_turn and intent.near_distance_mode and (intent.park_requested or park_requested):
                self._request_near_yaw_park(intent, reason)
            if getattr(self, "_near_yaw_park_request", None) is not None:
                self._use_soft_stop_next = False
        if pending_depth is not None:
            logger.info(
                "lateral_depth_read_unavailable capture_frame_id=%d uid=%d "
                "reason=%s sample_ts=%.6f age_ms=%.1f action=steer_zero "
                "grant_retained=True motion_authorized=False deadline_renewed=False "
                "continuation_veto=%s",
                intent.capture_frame_id, intent.target_id, reason,
                pending_depth[3], (now-pending_depth[3])*1000.,
                getattr(self, "_depth30_continuation_veto", None)
                    == (pending_depth[2], pending_depth[3]),
            )
        # Invalidate packets prepared before the axis fields above changed.
        # The executor compares this immediately before the backend write.
        self._lateral_yaw_revision = int(getattr(self, "_lateral_yaw_revision", 0)) + 1
        self._last_command_source_module = "lateral_intent_loop"
        self._last_command_control_frame = int(intent.frame_index)
        self._last_command_capture_frame = int(intent.capture_frame_id)
        self._last_command_capture_timestamp = float(intent.capture_timestamp)
        self._last_decision_capture_frame = int(intent.decision_capture_frame_id)
        zero_reason = "lateral_zero:" + str(reason)
        if not self._should_skip_redundant_action_queue([action], zero_reason):
            self._replace_action_queue([action], zero_reason)
        self._lateral_intent_last_publish_ts = now
        logger.info(
            "lateral_zero_publish seq=%d capture_frame_id=%d reason=%s "
            "yaw_rpm=0 preserve_depth=%s linear_percent=%d action=%s "
            "brake_continuity_reason=%s brake_continuity_elapsed_ms=%.1f "
            "brake_continuity_remaining_ms=%.1f",
            intent.sequence, intent.capture_frame_id, reason, preserve,
            int(linear[1]) if preserve else 0, ACTION_NAMES.get(action, str(action)),
            getattr(pid_result, "brake_continuity_reason", "not_evaluated"),
            1000. * getattr(pid_result, "brake_continuity_elapsed_sec", 0.),
            1000. * getattr(pid_result, "brake_continuity_remaining_sec", 0.),
        )
        return True

    def _publish_depth_composed_lateral(
        self, intent: LateralControlIntent, correction_rpm: int, reason: str,
    ) -> bool:
        """Publish nonzero yaw with the current Depth speed, never the old mode.

        A former yaw-only intent does not own a permanent zero-linear axis.
        Revoke its prepared pure-rotation packets when Depth later authorizes
        translation. All execution still passes through the normal queue.
        """
        now = time.monotonic()
        linear = PersonTracker._fresh_depth_linear_snapshot(self, intent.target_id, now=now)
        current = self._lateral_intent_store.snapshot()
        if (
            linear is None or intent.bbox_quality != "reliable"
            or current is None or current.sequence != intent.sequence
            or not intent.valid(now) or intent.hold_zero
            or int(getattr(self, "_lateral_intent_zero_sequence", -1)) == intent.sequence
            or int(correction_rpm) == 0
        ):
            return False
        kind, percent, _uid, _stamp = linear
        correction = int(correction_rpm)
        self._current_forward_percent = percent
        self._current_forward_allow_below_min = True
        self._current_steer_base_percent = percent if kind == "forward" else 0
        self._current_steer_correction_rpm = abs(correction) if kind == "forward" else correction
        self._current_steer_inner_ratio_percent = self._current_steer_outer_ratio_percent = 100
        self._current_rotate_raw_target = self._current_rotate_turn_percent = 0
        self._current_rotate_raw_source = "lateral_intent_revoked"
        self._current_rotate_pulse_enabled = False
        self._use_soft_stop_next = False
        self.is_forwarding = kind == "forward"
        self._last_vision_correction_rpm = correction
        self._last_vision_correction_at = now
        self._last_vision_correction_target_id = intent.target_id
        self._lateral_yaw_revision = int(getattr(self, "_lateral_yaw_revision", 0)) + 1
        action = (ACTION_BACKWARD if kind == "backward" else
                  ACTION_STEER_RIGHT if correction > 0 else ACTION_STEER_LEFT)
        self._last_command_source_module = "lateral_intent_loop"
        self._last_command_control_frame = int(intent.frame_index)
        self._last_command_capture_frame = int(intent.capture_frame_id)
        self._last_command_capture_timestamp = float(intent.capture_timestamp)
        self._last_decision_capture_frame = int(intent.decision_capture_frame_id)
        command_reason = "lateral_depth_composed:" + str(reason)
        if not self._should_skip_redundant_action_queue([action], command_reason):
            self._replace_action_queue([action], command_reason)
        self._lateral_intent_last_publish_ts = now
        logger.info(
            "lateral_depth_composed seq=%d capture_frame_id=%d old_mode=%s "
            "linear=%s/%d%% sample_age_ms=%.1f yaw=%+drpm action=%s",
            intent.sequence, intent.capture_frame_id, intent.mode, kind, percent,
            (now - linear[3]) * 1000.0, correction, ACTION_NAMES.get(action, str(action)),
        )
        return True

    def _handle_stale_vision_result(
        self,
        *,
        width: int,
        result_age_sec: float,
    ) -> bool:
        """Invalidate stale yaw and enter bounded direction recovery."""
        with self._control_update_lock:
            short = getattr(self, "_short_follow_adapter", None)
            if short is not None and short.owned:
                # A late processing attempt is not a new observation or a
                # command. The paired writer checks the original finite
                # depth/visual deadlines; do not enter legacy yaw recovery.
                if getattr(self, "_validated_visual_observation", None) is False:
                    short.controller.revoke("late_identity_rejected", time.monotonic())
                return True
            preserved = getattr(self, "_late_visual_preserved_evidence", None)
            self._late_visual_preserved_evidence = None
            identity_only = getattr(self, "_late_visual_identity_only", False)
            self._late_visual_identity_only = False
            identity_epoch = getattr(self, "_late_visual_identity_epoch", None)
            self._late_visual_identity_epoch = None
            identity_roi_published = getattr(self, "_late_visual_identity_roi_published", False)
            self._late_visual_identity_roi_published = False
            if preserved is not None:
                cap, stamp, visual, identity = preserved
                now = time.monotonic()
                uid = getattr(self._follow_controller, "active_target_id", None)
                # A processing-time miss is not evidence that an already
                # admitted independent Depth sample changed identity. Recheck
                # all live gates. The legacy branch retains only original
                # evidence. A current FULL identity-only result can instead
                # retain its own capture-bound proof/ROI, never a new yaw or
                # physical range lease. Executor TTL checks still own motion.
                scope = bool(
                    cap == getattr(self, "_active_capture_frame_id", None)
                    and stamp == getattr(self, "_active_capture_timestamp", None)
                    and getattr(self, "_validated_visual_observation", None) is visual
                    and getattr(self, "_detector_identity_lease", None) is identity
                    and visual.live(uid, now)
                    and self.search_state == "none"
                    and getattr(self._follow_controller, "search_state", "none") == "none"
                    and self._vision_control_state.startswith("target_visible")
                    and not self._explicit_stop_requested and not self._runtime_shutdown_requested
                    and self.running and not getattr(self, "_brake_hold_active", False)
                    and not getattr(self, "_reacquire_depth_pending", False))
                if identity_only:
                    scheduler = getattr(self, "_depth_async_scheduler", None)
                    scope = bool(scope and identity_roi_published
                                 and scheduler is not None and identity_epoch is not None
                                 and scheduler.publication_snapshot()[0] == identity_epoch)
                if scope:
                    hazards = self._current_hazard_state_for_controller()
                    obstacles = self._get_obstacle_status()
                    scope = not hazards.active and not any(obstacles.values())
                linear = (PersonTracker._fresh_depth_linear_snapshot(self, uid, now=now)
                          if scope else None)
                if identity_only and scope:
                    logger.info("processing_late_identity_retained capture_frame_id=%s uid=%s "
                                "processing_age_ms=%.1f capture_age_ms=%.1f visual_deadline=%s "
                                "depth_sample_ts=%s depth_deadline=%s "
                                "new_yaw_authorized=False new_depth_authorized=False",
                                cap, uid, result_age_sec*1000., (now-stamp)*1000.,
                                visual.expires_at, None if linear is None else linear[3],
                                None if linear is None else linear[3]+PersonTracker._depth_linear_max_age_sec(linear[0]))
                    # Absence/expiry of Depth is still enforced by the motor
                    # reader. Do not erase this independently verified new
                    # identity or its ROI while waiting for NEW ranging.
                    return True
                if linear is not None and linear[0] == "forward":
                    logger.info("stale_vision_existing_depth_preserved capture_frame_id=%s uid=%s "
                                "processing_age_ms=%.1f previous_visual_capture=%s "
                                "visual_deadline=%s depth_sample_ts=%s depth_deadline=%s "
                                "late_frame_authorized=False prior_axes_only=True deadline_renewed=False",
                                cap, uid, result_age_sec*1000., visual.capture, visual.expires_at,
                                linear[3], linear[3]+PersonTracker._depth_linear_max_age_sec(linear[0]))
                    return True
            # Audit only: preserving this lease also requires separating the
            # direction-recovery/identity state from visible-target execution.
            # Do not silently authorize translation by merely keeping a tuple.
            linear = getattr(self, "_depth30_linear_snapshot", None)
            timing = getattr(self, "_depth30_linear_timing", None)
            if linear is not None:
                audit_now = time.monotonic()
                logger.info(
                    "stale_vision_depth_audit capture_frame_id=%s uid=%s "
                    "sample_ts=%s age_ms=%.1f remaining_ms=%.1f "
                    "ff_expires_at=%s policy=revoke_with_direction_recovery",
                    getattr(self, "_active_capture_frame_id", None), linear[2], linear[3],
                    (audit_now-linear[3])*1000,
                    max(0.0, linear[3]+PersonTracker._depth_linear_max_age_sec(linear[0])-audit_now)*1000,
                    None if timing is None else timing.feedforward_expires_at,
                )
            lateral_gate = getattr(self, "_multi_person_lateral_gate", None)
            if lateral_gate is not None:
                lateral_gate.reset()
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
            self._clear_longitudinal_context(reason="stale_vision_result")
            self._current_forward_percent = 0
            self._current_steer_base_percent = 0
            self._current_steer_correction_rpm = 0
            self._current_steer_limit_reason = "stale_vision_result"
            self._current_rotate_raw_target = 0
            self._current_rotate_raw_source = "stale_vision_zero_yaw"
            self._current_rotate_pulse_enabled = False
            self._lateral_yaw_revision = int(getattr(self, "_lateral_yaw_revision", 0)) + 1
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
                and not self._explicit_stop_requested and not self._runtime_shutdown_requested
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
        publish_depth_immediately: bool = False,
        near_yaw_park_requested: bool = False,
        distance_control_deferred: bool = False,
    ) -> bool:
        if control_source != "vision":
            return False
        pid_result = self._follow_controller.last_steering_pid_result
        action = runtime_actions[0] if len(runtime_actions) == 1 else None
        zero_requested = bool(action is not None and action.kind == "stop")
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
            and getattr(self._follow_controller, "active_target_id", None) == target.track_id
            and int(width) > 0
            and (pid_result is not None or zero_requested)
            and (target_steerable or limited_low_quality_yaw or zero_requested)
            and (not low_quality_visible or limited_low_quality_yaw or zero_requested)
            and self.search_state == "none"
            and self._vision_control_state.startswith("target_visible")
            and not self._explicit_stop_requested
            and not self._runtime_shutdown_requested
        )
        if not eligible or action is None:
            self._clear_lateral_intent("vision_not_eligible")
            return False
        motion_requested = bool(
            not zero_requested
            and getattr(pid_result, "output_floor_reason", "") != "predictive_countersteer"
            and (int(getattr(pid_result, "correction_rpm", 0)) != 0
                 or (action.kind in ("forward", "backward", "steer_left", "steer_right")
                     and int(action.speed_percent) > 0))
        )
        if motion_requested:
            self._release_near_yaw_park(
                capture_id=int(getattr(self, "_last_command_capture_frame", 0)),
                capture_timestamp=float(getattr(self, "_last_command_capture_timestamp", 0.0)),
                target_id=int(target.track_id),
                # A mapped low-quality target may release a NORMAL pivot hold
                # for bounded yaw only. It never releases into translation.
                qualified=bool((target_steerable or limited_low_quality_yaw)
                               and not (low_quality_visible and not limited_low_quality_yaw)),
                reason="new_visible_motion",
                x_ratio=target.center[0] / max(1, width),
                translation_requested=bool(action.kind in ("forward", "backward", "steer_left", "steer_right")
                                           and action.speed_percent > 0),
                outward_lead=(getattr(pid_result, "outward_lead", None)
                              if target_steerable and not low_quality_visible else None),
            )
        if not zero_requested and getattr(self, "_near_yaw_park_request", None) is not None:
            # No new owner from old/low-quality motion evidence. The last
            # parked intent and its frame provenance remain authoritative.
            return False
        if action.kind in ("rotate_left", "rotate_right", "stop"):
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
        x_ratio = matching_control_x(
            getattr(self._follow_controller, "last_capture_steering_observation", None),
            int(target.track_id), int(getattr(self, "_last_command_capture_frame", 0)),
            float(getattr(self, "_last_command_capture_timestamp", 0.0)), x_ratio)
        motion_samples = getattr(self._follow_controller, "_visible_motion_samples", [])
        motion_dx_ratio = 0.0
        if len(motion_samples) >= 2:
            motion_dx_ratio = float(motion_samples[-1][2]) - float(motion_samples[-2][2])
        target_rate = (
            float(pid_result.target_image_rate_dps)
            if not zero_requested and bool(pid_result.target_rate_valid)
            else None
        )
        # Only ordinary qualified forward tracking may recompute the dynamic
        # limit on encoder refresh. Preserve explicit quality/park/reverse
        # ceilings; a transient 5RPM brake cap must not become a whole-frame
        # policy cap after the chassis has stopped rotating the wrong way.
        policy_limit = getattr(pid_result, "correction_policy_limit_rpm", None)
        if policy_limit is None or mode != "forward" or low_quality_visible or not target_steerable:
            policy_limit = getattr(pid_result, "correction_limit_rpm", 0.0)
        handoff_limit = handoff_yaw_limit(self, int(target.track_id))
        if handoff_limit is not None:
            policy_limit = min(float(policy_limit), handoff_limit)
        now = time.monotonic()
        previous_intent = self._lateral_intent_store.snapshot()
        capture_id = int(getattr(self, "_last_command_capture_frame", 0))
        watermark_uid, watermark_cap = getattr(self, "_lateral_observation_watermark", (None, 0))
        previous_cap = previous_intent.capture_frame_id if (
            previous_intent is not None and previous_intent.target_id == int(target.track_id)) else 0
        if watermark_uid == int(target.track_id):
            previous_cap = max(previous_cap, watermark_cap)
        if 0 < capture_id <= previous_cap and not zero_requested:
            return False  # Replaying one observation cannot renew yaw authority.
        brake_observation = getattr(self._follow_controller, "last_capture_steering_observation", None)
        turnaround_rate = getattr(pid_result, "braking_image_rate_dps", None)
        crop_outward_rate = getattr(pid_result, "outward_continuity_rate_dps", None)
        if not (not zero_requested and mode == "yaw_only" and limited_low_quality_yaw
                and brake_observation is not None
                and brake_observation.target_id == int(target.track_id)
                and brake_observation.capture_frame_id == capture_id
                and brake_observation.capture_timestamp == float(
                    getattr(self, "_last_command_capture_timestamp", 0.0))
                and crop_outward_rate == brake_observation.outward_continuity_rate_dps):
            crop_outward_rate = None
        if not (not zero_requested and mode == "yaw_only" and brake_observation is not None
                and brake_observation.target_id == int(target.track_id)
                and brake_observation.capture_frame_id == capture_id
                and brake_observation.capture_timestamp == float(
                    getattr(self, "_last_command_capture_timestamp", 0.0))
                and turnaround_rate == brake_observation.inward_turnaround_rate_dps):
            turnaround_rate = None
        intent = LateralControlIntent(
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
                base_rpm=max(0, int(getattr(pid_result, "base_rpm", 0))),
                initial_correction_rpm=0 if zero_requested else int(limit_handoff_yaw(
                    self, int(target.track_id), pid_result.correction_rpm)),
                correction_limit_rpm=max(0.0, float(policy_limit)),
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
                hold_zero=zero_requested,
                near_distance_mode=str(self._last_control_decision_reason).startswith("near_distance_rotation_only"),
                park_requested=bool(not distance_control_deferred
                    and (near_yaw_park_requested or predictive_or_center_stop(pid_result))),
                distance_control_deferred=bool(distance_control_deferred),
                visual_error_deg=getattr(pid_result, "visual_error_deg", None),
                outward_lead=(getattr(pid_result, "outward_lead", None)
                              if not zero_requested and mode in {"yaw_only", "forward"} and target_steerable
                              and not low_quality_visible else None),
                braking_image_rate_dps=turnaround_rate,
                outward_continuity_rate_dps=crop_outward_rate,
                forward_countersteer=bool(mode == "forward" and getattr(
                    pid_result, "output_floor_reason", "") == "predictive_countersteer"
                    and getattr(getattr(self._follow_controller, "cfg", None),
                                "visible_steering_pid_execution_response_trial_sec", 0) > 0),
                response_boost_allowed=bool(
                    not zero_requested and mode == "forward" and target_steerable
                    and not low_quality_visible and handoff_limit is None
                    and steering_buildup_allowed(pid_result)),
        )
        if VISIBLE_STEERING_PID_FORWARD_TRACKING_ENABLE:
            get_feedback = getattr(getattr(self, "_action_runtime", None), "get_steering_feedback", None)
            intent = replace(intent, image_error_only=bool(getattr(
                getattr(self._follow_controller, "cfg", None), "visible_steering_pid_image_error_only", False)))
            intent = with_forward_continuation(intent, get_feedback() if callable(get_feedback) else None)
        published = self._lateral_intent_store.publish(intent)
        self._lateral_turn_response_policy = (
            int(published.sequence), bool(published.response_boost_allowed))
        if capture_id > 0:
            self._lateral_observation_watermark = (int(target.track_id), max(previous_cap, capture_id))
        self._lateral_intent_owned_frame = int(self.frame_index)
        if self._request_predictive_turn_brake(published, pid_result):
            return True
        if zero_requested or (published.initial_correction_rpm == 0 and published.park_requested):
            # Do not wait for the next lateral tick or a 250ms motor timeout.
            self._publish_lateral_zero(published, published.reason, pid_result=pid_result)
        elif publish_depth_immediately:
            # New range and yaw become one wheel target, including a range
            # zero which must withdraw an earlier forward target immediately.
            self._lateral_intent_last_publish_ts = 0.0
            self._service_lateral_intent(now)
        elif (published.mode == "yaw_only" and published.near_distance_mode
              and published.initial_correction_rpm != 0 and target_steerable
              and not low_quality_visible
              and callable(getattr(getattr(self, "_action_runtime", None),
                                   "get_steering_feedback", None))):
            # CAP361: a new confirmed same-UID near-turn was already published
            # while the previous 150ms yaw lease expired. Service this NEW
            # capture now instead of waiting for the 30Hz tick and inserting a
            # needless zero. The service still checks freshness, identity and
            # all ordinary stop/parking gates; no old lease is extended.
            self._lateral_intent_last_publish_ts = 0.0
            self._service_lateral_intent(now)
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
            if thread.is_alive():
                thread.join(timeout=1.0)
            if thread.is_alive():
                logger.error("runtime_shutdown worker=lateral_intent still_alive=True cleanup_deferred=True")
                return
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

    @staticmethod
    def _depth_longitudinal_authority_enabled() -> bool:
        """Return whether Depth30 owns all forward/backward decisions.

        The visual PID remains responsible for producing a lateral correction,
        but its translational action must not be dispatched independently.
        Otherwise the 30 Hz Depth cap can be overwritten by the slower visual
        loop before the next Depth tick.
        """
        return bool(
            ASTRA_DEPTH_LONGITUDINAL_CONTROL_ENABLE
            and MODULE_ASTRA_DEPTH_ENABLE
            and VISION_DEPTH_ENABLED
        )

    def _depth30_translation_is_fresh(self) -> bool:
        """Return whether Depth30 still owns a recent forward/backward command."""
        if not self._depth_longitudinal_authority_enabled():
            return False
        kind = str(getattr(self, "_last_depth30_translation_kind", "") or "")
        if kind not in {"forward", "backward"}:
            return False
        stamp = float(getattr(self, "_last_depth30_translation_ts", 0.0) or 0.0)
        if stamp <= 0.0:
            return False
        age = time.monotonic()-stamp
        if 0 <= age <= min(ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC,
                           ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC):
            # Preserve the pre-existing short-window lateral arbitration
            # metadata contract. It is not itself a motor authorization.
            return True
        uid = getattr(getattr(self, "_follow_controller", None), "active_target_id", None)
        return uid is not None and PersonTracker._fresh_depth_linear_snapshot(self, uid) is not None

    def _remember_depth30_translation_decision(
        self,
        decision: ControlDecision,
        *,
        is_fresh_depth: bool,
    ) -> None:
        """Remember a fresh positive Depth30 translation before action merging.

        A forward decision can later become a ``steer_*`` action when the
        latest visual yaw is merged into it.  Recording ownership from the
        final integer action therefore misses exactly the cases that need to
        be protected from a delayed visual lost-frame rotation.
        """
        if decision.actions and not any(
            action.kind in {"forward", "backward"} and int(action.speed_percent) > 0
            for action in decision.actions
        ):
            self._depth30_linear_snapshot = None
        if not is_fresh_depth or not self._depth_longitudinal_authority_enabled():
            return
        for action in decision.actions:
            if action.kind not in {"forward", "backward"}:
                continue
            if int(getattr(action, "speed_percent", 0)) <= 0:
                continue
            self._last_depth30_translation_kind = str(action.kind)
            self._last_depth30_translation_ts = time.monotonic()
            return

    def _should_preserve_depth30_translation_for_visual_loss(self, reason: str) -> bool:
        """Keep a fresh Depth translation through a short visual miss only."""
        controller_state = str(
            getattr(self._follow_controller, "search_state", self.search_state)
            or "none"
        )
        return bool(
            controller_state not in ("searching", "timed_out", "direction_unresolved")
            and str(reason or "").startswith(
                ("lost_current_candidate_hold_", "lost_history_hold_", "lost_wait_yaw_")
            )
            and self._depth30_translation_is_fresh()
        )

    @staticmethod
    def _visual_stop_may_override_depth(reason: str, frame: SensorFrame) -> bool:
        """Allow only hard safety stops from the visual loop to preempt Depth30."""
        if bool(frame.hazard.active) or bool(frame.obstacles.front) or bool(
            frame.obstacles.left or frame.obstacles.right
        ):
            return True
        text = str(reason or "")
        return text.startswith((
            "hazard",
            "front_ir",
            "left_ir",
            "right_ir",
            "distance_too_close",
        ))

    def _request_predictive_turn_brake(self, intent, result) -> bool:
        if intent.distance_control_deferred:
            # A lateral-only result may countersteer, but missing a live
            # range grant is not evidence for a persistent chassis park.
            # The range owner and explicit safety paths decide translation.
            return False
        cfg = getattr(self._follow_controller, "cfg", None)
        if (intent.mode == "forward" and getattr(
                cfg, "visible_steering_pid_execution_response_trial_sec", 0) > 0):
            return False  # A forward brake never degrades into a parked pulse.
        if (intent.hold_zero or intent.bbox_quality != "reliable"
                or not intent.valid(time.monotonic())
                or not qualified_countersteer(result, intent.x_ratio, getattr(
                    cfg, "visible_steering_pid_predictive_countersteer_max_correction_rpm", 0.0))):
            return False
        # Never turn a normal forward curve into a whole-chassis brake pulse.
        if PersonTracker._fresh_depth_linear_snapshot(self, intent.target_id, now=time.monotonic()) is not None:
            return False
        braking = replace(intent, countersteer_rpm=int(result.correction_rpm),
                          visual_error_deg=float(result.visual_error_deg),
                          target_image_rate_dps=float(result.target_image_rate_dps),
                          park_requested=True)
        self._request_near_yaw_park(braking, "predictive_countersteer_then_normal")
        return True

    def _service_lateral_intent(self, now: float) -> None:
        short = getattr(self, "_short_follow_adapter", None)
        if short is not None and short.owned:
            return  # Both axes belong to one immutable normal-follow plan.
        if getattr(self, "_near_yaw_park_request", None) is not None:
            return  # A control tick/replayed image can never release parking.
        intent = self._lateral_intent_store.snapshot()
        if intent is None:
            return
        if not intent.valid(now):
            with self._control_update_lock:
                current = self._lateral_intent_store.snapshot()
                if current is None or current.sequence != intent.sequence:
                    return
                self._lateral_intent_last_expired_sequence = int(intent.sequence)
                self._clear_lateral_intent("expired")
                logger.info(
                    "lateral_intent_expired seq=%d frame=%d target=%d age=%.0fms ttl=%.0fms "
                    "action=revoke_yaw recognizer_owns_lost_transition=True "
                    "capture_frame_id=%d capture_age_ms=%.1f overdue_ms=%.1f",
                    intent.sequence,
                    intent.frame_index,
                    intent.target_id,
                    intent.age_sec(now) * 1000.0,
                    (intent.valid_until - intent.published_at) * 1000.0,
                    int(intent.capture_frame_id),
                    max(0.0, now - intent.capture_timestamp) * 1000.0 if intent.capture_timestamp > 0 else -1.0,
                    max(0.0, now - intent.valid_until) * 1000.0,
                )
            return

        with self._control_update_lock:
            if short is not None and short.owned:
                return
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
            # Acquiring the control lock may itself take a visual cycle.
            # Never refresh an intent using the pre-lock tick timestamp.
            now = max(float(now), time.monotonic())
            if not intent.valid(now):
                self._clear_lateral_intent("expired_waiting_control_lock")
                return
            if intent.hold_zero or int(getattr(self, "_lateral_intent_zero_sequence", -1)) == intent.sequence:
                # The synchronous publisher already sent zero. A reused frame
                # cannot re-arm startup or image-rate feedforward.
                return
            feedback = self._action_runtime.get_steering_feedback()
            if not intent.continuation_allowed(now, feedback):
                self._clear_lateral_intent("continuation_evidence_invalid")
                logger.info("lateral_continuation_rejected capture_frame_id=%d age_ms=%.1f",
                            intent.capture_frame_id, intent.age_sec(now) * 1000.0)
                return
            projected_x, projection_sec = intent.projected_x_ratio(
                now,
                camera_hfov_deg=VISIBLE_STEERING_PID_CAMERA_HFOV_DEG,
                max_projection_sec=LATERAL_INTENT_MAX_PROJECTION_SEC,
                max_projection_ratio=LATERAL_INTENT_MAX_PROJECTION_RATIO,
            )
            if getattr(getattr(self._follow_controller, "cfg", None), "visible_steering_pid_image_error_only", False):
                projected_x, projection_sec = intent.x_ratio, 0.0
            cfg = getattr(self._follow_controller, "cfg", None)
            effective_mode = intent.mode
            effective_base = intent.base_rpm
            effective_near = intent.near_distance_mode
            if (getattr(cfg, "visible_steering_pid_image_error_only", False)
                    and self._depth_longitudinal_authority_enabled()):
                live_linear = (PersonTracker._fresh_depth_linear_snapshot(
                    self, intent.target_id, now=now) if intent.bbox_quality == "reliable" else None)
                if live_linear is None:
                    effective_mode, effective_base, effective_near = "yaw_only", 0, True
                else:
                    effective_mode = "reverse" if live_linear[0] == "backward" else "forward"
                    effective_base = live_linear[1] * FORWARD_MAX_RPM / 100.0
            effective_limit = effective_correction_limit(
                cfg, effective_base, near_distance=effective_near,
                policy_limit=intent.correction_limit_rpm)
            recenter = getattr(self._follow_controller, "post_park_recenter_limit", None)
            if effective_mode == "yaw_only" and callable(recenter):
                recenter_limit = recenter(intent.target_id)
                if recenter_limit is not None:
                    effective_limit = min(effective_limit, recenter_limit)
            elif effective_mode in {"forward", "reverse"} and effective_base > 0:
                clear_recenter = getattr(self._follow_controller, "clear_post_park_recenter", None)
                if callable(clear_recenter):
                    clear_recenter("fresh_translation")
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
                    if effective_mode == "yaw_only"
                    else self._follow_controller.refresh_visible_lateral_pid
                )
                refresh_kwargs = {}
                if effective_mode == "yaw_only":
                    refresh_kwargs["near_distance_mode"] = effective_near
                    refresh_kwargs["braking_image_rate_dps"] = intent.braking_image_rate_dps
                    refresh_kwargs["outward_continuity_rate_dps"] = intent.outward_continuity_rate_dps
                elif effective_mode == "reverse":
                    refresh_kwargs["allow_forward_tracking"] = False
                if (effective_mode in {"yaw_only", "forward"}
                        and intent.outward_lead is not None and intent.bbox_quality == "reliable"
                        and intent.outward_lead.matches(intent.target_id, intent.capture_frame_id,
                                                       intent.capture_timestamp, now)):
                    refresh_kwargs["outward_lead"] = intent.outward_lead
                result = refresh_pid(
                    x_ratio=projected_x,
                    base_rpm=int(effective_base),
                    feedback=feedback,
                    now=now,
                    motion_dx_ratio=float(intent.motion_dx_ratio),
                    target_image_rate_dps=intent.target_image_rate_dps,
                    max_correction_rpm=float(effective_limit),
                    visual_age_sec=(
                        None
                        if float(intent.capture_timestamp) <= 0.0
                        else max(0.0, now - float(intent.capture_timestamp))
                    ),
                    **refresh_kwargs,
                )
                requested = int(result.correction_rpm)

            if not new_sequence and self._request_predictive_turn_brake(intent, result):
                return

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
            if (effective_mode == "forward" and getattr(
                    cfg, "visible_steering_pid_execution_response_trial_sec", 0) > 0
                    and qualified_countersteer(result, intent.x_ratio, getattr(
                        cfg, "visible_steering_pid_predictive_countersteer_max_correction_rpm", 0))):
                # This reverses the differential, NOT either wheel. The final
                # writer rechecks current feedback and the depth grant.
                correction = requested
            if requested == 0:
                # Zero is an immediate axis cancellation, not a ramp that can
                # keep the last turn alive while stopping is already required.
                correction = 0
            if (getattr(result, "brake_continuity_reason", "") in {
                    "recent_outward_deceleration", "fresh_position_deceleration"}
                    and getattr(result, "brake_continuity_remaining_sec", 0.) > 0.):
                # This is a bounded braking target, not a new acceleration
                # request. Slew must not keep an old -10 above the new -2 cap,
                # nor carry an opposite residual direction through the hold.
                correction = (int(math.copysign(min(abs(correction), abs(requested)), requested))
                              if correction * requested > 0 else 0)
            correction = int(limit_handoff_yaw(self, int(intent.target_id), correction))
            # Also clamp the first reused visual result and the slew output.
            # A policy decrease must not decay slowly from an obsolete 15RPM.
            correction = int(clamp_correction(correction, effective_limit))
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
            # Publish the fast loop's current braking veto before its output.
            # An untapered visual frame is not permission to amplify a later
            # encoder/visual predictive taper of that same frame.
            self._lateral_turn_response_policy = (int(intent.sequence), steering_buildup_allowed(result))
            self._lateral_intent_last_target_id = int(intent.target_id)
            self._lateral_intent_last_mode = intent.mode
            self._lateral_intent_last_sequence = int(intent.sequence)
            self._lateral_intent_last_correction_rpm = int(correction)
            self._lateral_intent_last_tick_ts = float(now)

            if correction == 0 and (
                intent.mode == "yaw_only" or self._depth_longitudinal_authority_enabled()
            ):
                self._publish_lateral_zero(
                    intent, "pid_zero:" + intent.reason,
                    park_requested=predictive_or_center_stop(result),
                    pid_result=result,
                )
                return

            depth_longitudinal_authority = self._depth_longitudinal_authority_enabled()
            depth_linear = (
                PersonTracker._fresh_depth_linear_snapshot(self, intent.target_id, now=now)
                if intent.bbox_quality == "reliable" else None
            )
            if depth_linear is not None:
                action = (ACTION_BACKWARD if depth_linear[0] == "backward" else
                          ACTION_STEER_RIGHT if correction > 0 else ACTION_STEER_LEFT)
            elif depth_longitudinal_authority:
                # A metadata/read race may be temporary while the physical
                # lease is live. Explicit braking or continuation vetoes are
                # not pending leases: they retain the old revoke/STOP path.
                pending_depth = (PersonTracker._pending_forward_depth_lease(
                    self, intent.target_id, now)
                    if intent.bbox_quality == "reliable" else None)
                if pending_depth is not None:
                    self._publish_lateral_zero(
                        intent, "depth_read_unavailable:" + str(intent.reason),
                    )
                    return
                # Once the physical lease or identity is gone, fresh visual
                # yaw cannot revive its old forward base.
                snapshot = getattr(self, "_depth30_linear_snapshot", None)
                if snapshot is not None:
                    self._revoke_depth_linear_authority(
                        PersonTracker._depth_linear_withdrawal_reason(
                            self, snapshot, now, "lateral_depth_ineligible")
                    )
                action = ACTION_ROTATE_RIGHT if correction > 0 else ACTION_ROTATE_LEFT
            visual_translation_suppressed = False

            if depth_linear is not None:
                pass  # the composer writes both axes together immediately before queueing
            elif intent.mode == "yaw_only" or depth_longitudinal_authority:
                self._current_forward_allow_below_min = False
                if depth_longitudinal_authority:
                    self._current_forward_percent = 0
                    self._current_steer_base_percent = 0
                    self.is_forwarding = False
                self._current_rotate_pulse_enabled = False
                self._current_rotate_raw_target = abs(int(correction))
                self._current_rotate_raw_source = "lateral_intent_30hz"
            elif intent.mode == "reverse" and not depth_longitudinal_authority:
                self._current_forward_percent = int(intent.base_percent)
                self._current_steer_correction_rpm = int(correction)
            elif intent.mode == "forward" and not depth_longitudinal_authority:
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
            if visual_translation_suppressed:
                # Depth30 is the sole owner of longitudinal motor commands.
                # _queue_actions_for_persons() will merge the saved visual
                # correction into its next 30 Hz action.  Do not enqueue a
                # visual forward/backward action here, or it can overwrite the
                # Depth speed cap between two depth ticks.
                publish_due = False
            if not visual_translation_suppressed and action != self._last_dispatched_action:
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
                if depth_linear is not None:
                    self._publish_depth_composed_lateral(intent, correction, str(intent.reason))
                elif not self._should_skip_redundant_action_queue([action], reason):
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
                    "requested=%+drpm output=%+drpm limit=%.1frpm floor=%s action=%s "
                    "confidence=%.2f quality=%s feedback=%s publish=%s "
                    "policy_limit_rpm=%.1f limit_reason=%s remaining_ms=%.1f forward_phase=%s continuation=%s capture_age_ms=%.1f "
                    "effective_mode=%s effective_limit_rpm=%.1f "
                    "position_demand_rpm=%s brake_reduction_rpm=%s buildup_pid_allowed=%s "
                    "brake_continuity_reason=%s brake_continuity_elapsed_ms=%.1f "
                    "brake_continuity_remaining_ms=%.1f",
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
                    "none" if result is None else result.output_floor_reason,
                    (
                        "depth30_owned"
                        if visual_translation_suppressed
                        else ACTION_NAMES.get(action, str(action))
                    ),
                    intent.confidence,
                    intent.bbox_quality,
                    False if result is None else result.feedback_used,
                    publish_due,
                    intent.correction_limit_rpm,
                    getattr(result, "correction_limit_reason", "none"),
                    max(0.0, intent.valid_until - now) * 1000.0,
                    getattr(result, "forward_phase", "legacy"),
                    bool(intent.nominal_valid_until > 0 and now > intent.nominal_valid_until),
                    max(0.0, now - intent.capture_timestamp) * 1000.0 if intent.capture_timestamp > 0 else -1.0,
                    effective_mode,
                    effective_limit,
                    getattr(result, "position_demand_rpm", None),
                    getattr(result, "brake_reduction_rpm", None),
                    steering_buildup_allowed(result),
                    getattr(result, "brake_continuity_reason", "not_evaluated"),
                    1000. * getattr(result, "brake_continuity_elapsed_sec", 0.),
                    1000. * getattr(result, "brake_continuity_remaining_sec", 0.),
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
        person_targets: Optional[Tuple[PersonTarget, ...]] = None,
    ) -> None:
        """Publish only the currently locked ReID target for short Depth reuse."""
        active_target_id = getattr(self._follow_controller, "active_target_id", None)
        if active_target_id is None or int(active_target_id) < 0:
            self._clear_longitudinal_context(reason="no_active_uid")
            return
        matching = [
            person
            for person in persons
            if len(person) >= 2 and int(person[1]) == int(active_target_id)
        ]
        if not matching:
            self._clear_longitudinal_context(reason="active_uid_missing")
            return
        target_snapshot = (
            tuple(PersonTracker._persons_to_targets(
                self, matching, width=int(width), height=int(height),
            )) if person_targets is None else tuple(
                target for target in person_targets if target.track_id == active_target_id
            )
        )
        if [(tuple(t.bbox), t.track_id) for t in target_snapshot] != [
            (tuple(p[0]), int(p[1])) for p in matching
        ]:
            self._clear_longitudinal_context(reason="depth_snapshot_person_mismatch")
            return
        if len(target_snapshot) != 1 or target_snapshot[0].depth_observation is None:
            # Missing or ambiguous detector provenance must not refresh a
            # Depth context using only the expanded display box.
            self._clear_longitudinal_context(reason="detector_provenance_missing_or_ambiguous")
            return
        observation = target_snapshot[0].depth_observation
        context = {
            "published_ts": time.monotonic(),
            "frame_index": int(self.frame_index),
            "width": int(width),
            "height": int(height),
            "persons": list(matching),
            "person_targets": target_snapshot,
            "capture_frame_id": observation.capture_frame_id,
            "capture_timestamp": observation.capture_timestamp,
            "target_id": int(active_target_id),
            "target_steerable": bool(target_steerable),
        }
        scheduler = getattr(self, "_depth_async_scheduler", None)
        if scheduler is not None:
            with self._control_update_lock:
                if (getattr(self, "_explicit_stop_requested", False)
                        or getattr(self, "_runtime_shutdown_requested", False)
                        or not self.running or self.search_state != "none"):
                    scheduler.revoke("context_publication_not_visible")
                    return
                context = scheduler.submit(context, now=time.monotonic()).as_dict()
                with self._longitudinal_context_lock:
                    self._longitudinal_context = context
        else:
            with self._longitudinal_context_lock:
                self._longitudinal_context = context
        observer = getattr(self, "_depth_track_observer", None)
        if observer is not None:
            if target_steerable and self.search_state == "none":
                observer.publish(observation, width, height)
            else:
                observer.revoke()
        # A fresh visual ROI has a short remaining lifetime. Do not spend it
        # waiting for the next periodic tick. Wakeups coalesce (no frame queue).
        wake = getattr(self, "_longitudinal_wake_event", None)
        if wake is not None:
            wake.set()

    def _longitudinal_restart_identity_live(self, uid, sample_timestamp, now) -> bool:
        """Read current identity for a NEW range restart, never the old grant.

        Called under the control owner. Do not query the old grant (which is
        necessarily withdrawn), read hardware, or use visual-control completion
        as identity proof. The caller separately validates this sample, wheel
        feedback and braking; final admission/writing repeats the live checks.
        """
        ctl = self._follow_controller
        proof = getattr(self, "_validated_visual_observation", None)
        if (type(uid) is not int or uid <= 0
                or not isinstance(proof, ValidatedVisualObservation)
                or proof.continuation_sample_timestamp is not None
                or not isinstance(sample_timestamp, (int, float))
                or isinstance(sample_timestamp, bool)
                or not math.isfinite(sample_timestamp)
                or not isinstance(now, (int, float)) or isinstance(now, bool)
                or not math.isfinite(now) or not 0 < sample_timestamp <= now
                or getattr(ctl, "active_target_id", None) != uid
                or not proof.permits_depth(uid, sample_timestamp, now)
                or not motion_identity_live(self, uid, now)
                or getattr(ctl, "search_state", None) != "none"
                or getattr(self, "search_state", None) != "none"
                or not getattr(self, "running", False)
                or getattr(self, "_depth_roi_safety_clear", False) is not True
                or any(getattr(self, flag, False) for flag in (
                    "_explicit_stop_requested", "_runtime_shutdown_requested",
                    "_brake_hold_active", "_reacquire_depth_pending"))
                or getattr(self, "_near_yaw_park_request", None) is not None
                or getattr(ctl, "_normal_parking_uid", None) is not None):
            return False
        return True

    def _publish_validated_depth_observation(
        self, records, width, height, capture_id, capture_ts, *, expected_epoch,
    ) -> bool:
        """Publish a stable detector ROI BEFORE waiting for controller work.

        Only the scheduler's short mailbox mutex is used. This does not select
        a range, change compatibility/control state, release parking, or grant
        motion. The Depth worker still rechecks its ticket, current UID, safety
        and the fixed physical sample under the normal control mutex.
        """
        scheduler = getattr(self, "_depth_async_scheduler", None)
        proof = getattr(self, "_validated_visual_observation", None)
        if scheduler is None or expected_epoch is None or not isinstance(proof, ValidatedVisualObservation):
            return False
        now = time.monotonic()
        epoch, previous = scheduler.publication_snapshot()
        ctl = self._follow_controller
        uid = proof.uid
        # Bootstrap, reacquisition and grey-frame bridges retain their existing
        # synchronized paths. This path cannot promote a candidate to a UID.
        if (epoch != expected_epoch or previous is None or previous.target_id != uid
                or capture_id <= previous.capture_frame_id or capture_ts <= previous.capture_timestamp
                or proof.capture != capture_id or proof.timestamp != capture_ts
                or proof.continuation_sample_timestamp is not None
                or not proof.live(uid, now)
                or not 0 <= now-capture_ts <= ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC
                or getattr(getattr(ctl, "last_selected_target", None), "track_id", None) != uid):
            return False
        worker = getattr(self, "_longitudinal_thread", None)
        if not scheduler.health(now=now, worker_alive=worker is not None and worker.is_alive()).healthy:
            return False

        def current_publication_allowed():
            # Pure reads only: runs under the scheduler mutex, never waits for
            # motor, sensor or control locks. Commit repeats the full checks.
            checked_at = time.monotonic()
            return bool(
                self._validated_visual_observation is proof
                and proof.live(uid, checked_at)
                and 0 <= checked_at-capture_ts <= ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC
                and getattr(ctl, "active_target_id", None) == uid
                and getattr(ctl, "search_state", None) == "none"
                and self.search_state == "none" and self.running
                and getattr(self, "_action_runtime_started", False)
                and getattr(self, "_depth_roi_safety_clear", False) is True
                and not any(getattr(self, flag, False) for flag in (
                    "_explicit_stop_requested", "_runtime_shutdown_requested",
                    "_brake_hold_active", "_reacquire_depth_pending"))
                and getattr(self, "_near_yaw_park_request", None) is None
                and getattr(self, "_vision_control_state", "") in {
                    "target_visible", "target_visible_depth_valid", "target_visible_depth_missing"}
                and self._active_capture_frame_id == capture_id
                and self._active_capture_timestamp == capture_ts)

        if not current_publication_allowed():
            return False
        hazard_config = getattr(getattr(self, "_bunker_runtime", None), "config", None)
        if (getattr(hazard_config, "enabled", False)
                and getattr(hazard_config, "mode", None) == "merged"
                and any(getattr(r, "class_id", None) in hazard_config.class_ids for r in records)):
            # The normal frame path has not evaluated merged danger boxes yet.
            # Leave that path in charge, without counting/confirming them twice.
            return False
        matches = [r for r in records if getattr(r, "reid_uid", None) == uid
                   and getattr(r, "time_since_update", 1) == 0]
        if len(matches) != 1:
            return False
        rec = matches[0]
        assignment = self._identity_assignment_debug_for_track(rec.track_id)
        if (rec.track_id != proof.track_id or rec.class_id != PERSON_CLASS_ID
                or not math.isfinite(rec.score) or rec.score <= CONFIDENCE_THRESHOLD
                or assignment.get("uid") != uid
                or not PersonTracker._bound_reid_bbox_allowed_for_control(assignment)
                or assignment.get("identity_recheck_pending")
                or assignment.get("search_excluded")
                or rec.track_id in PersonTracker._search_excluded_tracks(self)):
            return False
        bbox = PersonTracker._track_record_bbox(rec)
        if bbox is None:
            return False
        tracker = self._rknn_pipeline.tracker
        geometry_args = dict(target_id=uid, display_bbox=bbox, capture_frame_id=capture_id,
            capture_timestamp=capture_ts,
            observations=tuple(getattr(tracker, "last_identity_observations", ()) or ()),
            width=width, height=height, expected_raw_track_id=rec.track_id)
        observation = resolve_depth_target_observation(**geometry_args)
        if observation is None:
            return False
        # Do not call the sensor/hazard getters here: they may read IR hardware
        # and advance debounce counters. This is only a mailbox publication;
        # the Depth consumer must perform its own current safety checks.
        now = time.monotonic()
        if not proof.live(uid, now) or not 0 <= now-capture_ts <= ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC:
            return False
        target = PersonTarget(bbox=tuple(bbox), track_id=uid, confidence=float(rec.score),
            area=float(rec.area), depth_observation=observation,
            initial_identity_confirmed=resolve_initial_identity_confirmation(**geometry_args),
            braking_observation=resolve_braking_target_observation(**geometry_args))
        result = scheduler.submit(dict(
            published_ts=now, frame_index=int(self.frame_index), width=int(width), height=int(height),
            person_targets=(target,), capture_frame_id=capture_id, capture_timestamp=capture_ts,
            target_id=uid, target_steerable=True), now=now, expected_epoch=expected_epoch,
            publication_guard=current_publication_allowed)
        if result is None or result.capture_frame_id != capture_id or result.capture_timestamp != capture_ts:
            return False
        wake = getattr(self, "_longitudinal_wake_event", None)
        if wake is not None:
            wake.set()
        logger.info("depth_roi_early_publish capture_frame_id=%s uid=%s raw_track_id=%s "
                    "capture_age_ms=%.1f identity_to_publish_ms=%.1f epoch=%s "
                    "control_lock_waited=False motion_authorized=False deadline_renewed=False",
                    capture_id, uid, rec.track_id, (now-capture_ts)*1000.,
                    (now-proof.validated_at)*1000., result.epoch)
        return True

    def _depth_roi_age_allowed(self, capture_ts, now):
        from car_control_modular.depth_roi_policy import roi_age_allowed, bounded_roi_history_allowed
        feedback = None
        if now - capture_ts > .18:
            reader = getattr(getattr(self, "_action_runtime", None), "get_steering_feedback", None)
            try:
                feedback = reader() if callable(reader) else None
            except Exception:
                feedback = None
        if roi_age_allowed(capture_ts, now, ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC, feedback):
            return True
        # This only permits a ranging attempt. DistanceRuntime and the actual
        # camera must select/recheck a real sample inside the original 180ms
        # association window; no old ROI receives unrestricted latest access.
        return bool(getattr(getattr(self, "_sensor_runtime", None), "supports_bounded_depth_roi", False) is True
                    and bounded_roi_history_allowed(capture_ts, now,
                                                    ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC))

    def _visible_latest_depth_eligible(
        self, targets, selected, capture_id, capture_ts, *,
        control_source, target_steerable, low_quality_visible,
    ) -> bool:
        """Only an already-followed, identity-proven current detector ROI.

        Reacquisition/weak observations still use historical RGB alignment.
        Choosing latest here does not relax the sensor's spatial association,
        detector age, physical sample age or downstream safety checks.
        """
        ctl = self._follow_controller
        uid = getattr(ctl, "active_target_id", None)
        observation = getattr(selected, "depth_observation", None)
        previous = getattr(ctl, "last_selected_target", None)
        now = time.monotonic()
        return bool(
            control_source == "vision" and self._depth_longitudinal_authority_enabled()
            and not FOLLOW_ROTATION_ONLY and target_steerable and not low_quality_visible
            and getattr(self, "running", False)
            and not any(getattr(self, key, False) for key in (
                "_explicit_stop_requested", "_runtime_shutdown_requested",
                "_reacquire_depth_pending",
            ))
            and (not getattr(self, "_brake_hold_active", False) or is_follow_distance_hold(self)
                 or (getattr(self, "_near_yaw_park_request", None) is not None
                     and getattr(self, "_brake_hold_label", "") == "near_yaw_park"))
            and self.search_state == "none" and getattr(ctl, "search_state", None) == "none"
            and getattr(self, "_vision_control_state", "") in {
                "target_visible", "target_visible_depth_valid", "target_visible_depth_missing"
            }
            and selected is not None and uid is not None and selected.track_id == uid
            and previous is not None and previous.track_id == uid
            and sum(t.track_id == uid for t in targets) == 1
            and observation is not None and observation.source == "yolo_detector"
            and observation.target_id == uid and observation.capture_frame_id == capture_id
            and capture_id > 0 and observation.capture_timestamp == capture_ts
            and math.isfinite(capture_ts) and capture_ts > 0.0
            and self._depth_roi_age_allowed(capture_ts, now)
        )

    def _start_longitudinal_thread(self) -> None:
        if (
            not ASTRA_DEPTH_LONGITUDINAL_CONTROL_ENABLE
            or not MODULE_ASTRA_DEPTH_ENABLE
            or not VISION_DEPTH_ENABLED
        ):
            logger.info("30Hz Depth纵向监督未启用")
            return
        if self._longitudinal_thread is not None and self._longitudinal_thread.is_alive():
            return
        logger.info(
            "visual_depth_visibility_config capture_ttl_ms=%.0f "
            "new_full_result_max_age_ms=%.0f "
            "yaw_and_roi_clocks_independent=True repeat_capture_renews=False",
            VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC * 1000.0,
            MAX_FULL_RESULT_AGE_SEC * 1000.0,
        )
        logger.info(
            "depth_clock_config roi_max_ms=%.0f roi_normal_ms=180 live_latest_max_ms=250 "
            "extended_max_yaw_dps=5 roi_over_250=bounded_history_only "
            "physical_depth_ttl_ms=%.0f forward_max_rpm=%d forward_motor_max_rpm=%d "
            "bounded_roi_history=%s bounded_association_max_ms=180 "
            "bounded_sample_max_ms=180 depth_timestamp_kind=arrival",
            ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC * 1000,
            ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC * 1000,
            FORWARD_MAX_RPM, MOTOR_FORWARD_MAX_TARGET_RPM,
            getattr(getattr(self, "_sensor_runtime", None), "supports_bounded_depth_roi", False) is True,
        )
        logger.info(
            "depth_async_config enabled=%s stable_visual=lateral_only "
            "range_owner=depth30 max_inflight=1 pending_roi=latest "
            "same_uid_replacement_revokes=False "
            "fallback=exclusive_sync physical_depth_ttl_ms=%.0f",
            getattr(self, "_depth_async_scheduler", None) is not None,
            ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC * 1000.0,
        )
        self._longitudinal_stop_event.clear()
        self._longitudinal_thread = threading.Thread(
            target=self._longitudinal_control_loop,
            name="depth-longitudinal-control",
            daemon=True,
        )
        self._longitudinal_thread.start()
        logger.info(
            "30Hz Depth纵向监督已启动: rate=%.1fHz bbox_ttl=%.0fms "
            "max_forward=%d%% far_forward=%d%% far_distance=%.2fm "
            "motor_writer=action_runtime lateral_owner=%s",
            ASTRA_DEPTH_LONGITUDINAL_CONTROL_HZ,
            ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC * 1000.0,
            ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT,
            ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT,
            ASTRA_DEPTH_LONGITUDINAL_FAR_DISTANCE_M,
            bool(LATERAL_INTENT_CONTROL_ENABLE),
        )

    def _stop_longitudinal_thread(self) -> None:
        self._longitudinal_stop_event.set()
        scheduler = getattr(self, "_depth_async_scheduler", None)
        if scheduler is not None:
            scheduler.revoke("depth_worker_stopping")
        wake = getattr(self, "_longitudinal_wake_event", None)
        if wake is not None:
            wake.set()
        thread = self._longitudinal_thread
        if thread is not None:
            if thread.is_alive():
                thread.join(timeout=0.50)
            if thread.is_alive():
                # Do not lose ownership of a live worker or wait indefinitely
                # on the very control lock that prevented it from exiting.
                logger.error("runtime_shutdown worker=depth30 still_alive=True cleanup_deferred=True")
                return
        self._longitudinal_thread = None
        if not self._control_update_lock.acquire(blocking=False):
            logger.warning("runtime_shutdown worker=depth30 cleanup_deferred=True reason=control_lock_busy")
            return
        context_acquired = False
        try:
            context_acquired = self._longitudinal_context_lock.acquire(blocking=False)
            if not context_acquired:
                logger.warning("runtime_shutdown worker=depth30 cleanup_deferred=True reason=context_lock_busy")
                return
            # The latched shutdown gate already forbids all motion. Cleanup
            # need not call controller/logging callbacks while holding locks.
            self._longitudinal_context = None
            self._depth30_linear_snapshot = None
            self._depth30_linear_timing = None
        finally:
            if context_acquired:
                self._longitudinal_context_lock.release()
            self._control_update_lock.release()

    def _longitudinal_control_loop(self) -> None:
        period_sec = 1.0 / max(1.0, float(ASTRA_DEPTH_LONGITUDINAL_CONTROL_HZ))
        while not self._longitudinal_stop_event.is_set():
            wake = getattr(self, "_longitudinal_wake_event", None)
            if wake is not None:
                wake.clear()
            cycle_started = time.monotonic()
            scheduler = getattr(self, "_depth_async_scheduler", None)
            if scheduler is not None:
                scheduler.worker_tick(now=cycle_started)
            control_deferred = False
            with self._longitudinal_context_lock:
                context = (
                    None
                    if self._longitudinal_context is None
                    else dict(self._longitudinal_context)
                )
            if scheduler is not None and context is not None:
                # Identity may already have supplied a newer immutable ROI
                # while ordinary visual/lateral control waits for the mutex.
                # Keep the compatibility None veto used by explicit clearing.
                _, pending_roi = scheduler.publication_snapshot()
                if pending_roi is not None:
                    context = pending_roi.as_dict()
            if context is not None and self.running and self._action_runtime_started:
                context_age = cycle_started - float(context["capture_timestamp"])
                active_target_id = getattr(self._follow_controller, "active_target_id", None)
                context_valid = bool(
                    0.0 <= context_age <= ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC
                    and active_target_id is not None
                    and int(active_target_id) == int(context["target_id"])
                    and self.search_state == "none"
                    and not self._runtime_shutdown_requested
                )
                if context_valid:
                    ticket = None
                    failed = False
                    try:
                        if scheduler is not None:
                            ticket = scheduler.begin(
                                now=time.monotonic(),
                                max_capture_age_sec=ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC,
                            )
                            if ticket is None:
                                # Another scan (including sync bootstrap) owns
                                # the slot. Coalesce, never queue pixel work.
                                self._longitudinal_stop_event.wait(min(period_sec, .005))
                                continue
                            context = ticket.context.as_dict()
                        processed = self._queue_actions_for_persons(
                            int(context["width"]),
                            int(context["height"]),
                            list(context["persons"]),
                            depth_use_latest=True,
                            control_source="depth30",
                            target_steerable=bool(context.get("target_steerable", True)),
                            expected_target_id=int(context["target_id"]),
                            context_published_ts=float(context["published_ts"]),
                            expected_frame_index=int(context["frame_index"]),
                            depth_target_snapshot=context["person_targets"],
                            evidence_capture_frame_id=int(context["capture_frame_id"]),
                            evidence_capture_timestamp=float(context["capture_timestamp"]),
                            **({} if ticket is None else {"depth_work_ticket": ticket}),
                        )
                        control_deferred = processed is False
                    except Exception as exc:
                        failed = True
                        logger.warning("30Hz Depth纵向监督单次更新失败: %s", exc)
                    finally:
                        if ticket is not None:
                            scheduler.finish(ticket, now=time.monotonic(), error=failed)
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
            waiter = wake if wake is not None else self._longitudinal_stop_event
            # A busy control lock is not a failed depth sample. Retry from the
            # *latest* published context, rather than queueing this ROI behind
            # a visual decision or sleeping a complete 30 Hz period. No
            # ranging, PID update, queue write or authority renewal took place.
            # A completed preflight with no eligible physical sample consumes
            # its tick instead: poll at the normal rate, or wake for a new ROI.
            retry_sec = min(period_sec, 0.005) if control_deferred else period_sec
            waiter.wait(max(0.001, retry_sec - elapsed))

    def _action_names_for_log(self, actions: List[int]) -> List[str]:
        return [ACTION_NAMES.get(a, str(a)) for a in actions]

    def _action_queue_names_for_log(self, actions: List[int]) -> List[str]:
        return [ACTION_NAMES.get(a, str(a)) for a in actions]

    def _action_queue_snapshot_locked(self) -> List[int]:
        try:
            return [int(item) for item in self.action_queue.queue]
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
                    int(self._current_steer_base_percent),
                    "distance_missing" in str(self._last_control_decision_reason or ""),
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
        scheduler = getattr(self, "_depth_async_scheduler", None)
        if scheduler is not None:
            scheduler.revoke("direct_stop:" + reason_text)
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
        coalesce = getattr(getattr(self, "_action_runtime", None),
                           "can_coalesce_follow_queue_refresh", None)
        if callable(coalesce) and coalesce(requested_actions):
            # The producer has already published new canonical Depth/yaw
            # state. The periodic wheel writer reads it directly and checks
            # authority again at send time. Do not enqueue the same adopted
            # action under the serial lock merely to refresh its heartbeat.
            # In particular, leave queue revisions and STOP barriers intact.
            logger.info(
                "follow_queue_refresh_coalesced control_frame_id=%d evidence_capture_frame_id=%d "
                "source=%s action=%s reason=%s motor_wait_ms=0.000 queue_revision_unchanged=True",
                self.frame_index, int(getattr(self, "_last_command_capture_frame", -1)),
                getattr(self, "_last_command_source_module", "unknown"),
                ACTION_NAMES.get(requested_actions[0], str(requested_actions[0])), reason,
            )
            return
        action_signature = self._actions_signature(list(requested_actions))
        if len(actions) == 1 and actions[0] in ROTATE_ACTIONS and not bool(
            ROTATE_PULSE_BRAKE_ENABLE
            and getattr(self, "_current_rotate_pulse_enabled", True)
        ):
            self._last_rotate_visual_refresh_ts = time.time()
        cleared = 0
        queued = 0
        # Serialize publication with the final motor write. A popped ordinary
        # STOP cannot cross this commit and override a newer steering intent.
        motor_wait_started = time.perf_counter()
        with ExitStack() as publish_locks:
            publish_locks.enter_context(getattr(self, "motor_io_lock", nullcontext()))
            motor_acquired = time.perf_counter()
            publish_locks.enter_context(self.action_queue_lock)
            queue_acquired = time.perf_counter()
            self._action_command_revision = int(getattr(self, "_action_command_revision", 0)) + 1
            revision = self._action_command_revision
            old_commands = tuple(self.action_queue.queue)
            protected_commands = [item for item in old_commands
                                  if isinstance(item, ActionCommandSnapshot) and item.protected_stop]
            before_actions = self._action_queue_snapshot_locked()
            before = len(before_actions)
            merge = merge_pending_actions(
                [int(item) for item in old_commands
                 if not (isinstance(item, ActionCommandSnapshot)
                         and int(item) == ACTION_STOP and not item.protected_stop
                         and requested_actions and requested_actions[0] != ACTION_STOP)],
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
                    if action == ACTION_STOP and protected_commands:
                        self.action_queue.put_nowait(protected_commands.pop(0))
                        queued += 1
                        continue
                    stop_provenance = getattr(self, "_explicit_stop_provenance", None)
                    controller_state_stop = stop_provenance == ("controller_state", str(reason))
                    protected = bool(action == ACTION_STOP and (
                        getattr(self, "_runtime_shutdown_requested", False)
                        or (getattr(self, "_explicit_stop_requested", False) and not controller_state_stop)
                        or any(token in reason for token in
                               ("hard_stop", "hazard", "bunker", "front_ir", "left_ir", "right_ir", "manual", "user_stop"))))
                    stop_origin = (stop_provenance[0] if isinstance(stop_provenance, tuple)
                        and len(stop_provenance) == 2 and stop_provenance[1] == str(reason)
                        else "unknown" if protected else "controller_state")
                    self.action_queue.put_nowait(ActionCommandSnapshot(
                        action=int(action), revision=revision, enqueued_at=replace_ts,
                        control_frame=int(self.frame_index),
                        capture_frame_id=int(getattr(self, "_last_command_capture_frame", -1)),
                        capture_timestamp=float(getattr(self, "_last_command_capture_timestamp", 0.0)),
                        reason=str(reason), soft_stop=bool(getattr(self, "_use_soft_stop_next", False)) and not protected,
                        protected_stop=protected,
                        source_module=str(getattr(self, "_last_command_source_module", "unknown")),
                        uid=getattr(getattr(self, "_follow_controller", None), "active_target_id", None),
                        stop_origin=stop_origin,
                    ))
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
            publish_finished = time.perf_counter()
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
            "action queue replace: seq=%d control_frame_id=%d decision_capture_frame_id=%d evidence_capture_frame_id=%d source_module=%s reason=%s actions=%s signature=%s before=%d cleared=%d queued=%d after=%d clear_ms=%.3f total_ms=%.3f queue_before=%s queue_after=%s current=%s last_dispatched=%s since_last_enqueue_ms=%.1f search=%s/%s motor_lock_wait_ms=%.3f queue_lock_wait_ms=%.3f locked_publish_ms=%.3f",
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
            (motor_acquired - motor_wait_started) * 1000.0,
            (queue_acquired - motor_acquired) * 1000.0,
            (publish_finished - queue_acquired) * 1000.0,
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
        self._last_explicit_stop_reason = str(reason)
        self._explicit_stop_provenance = ("sensor_safety", str(reason))
        self._follow_distance_hold = None
        if getattr(self, "_depth30_linear_snapshot", None) is not None:
            self._clear_longitudinal_context(reason="hazard:" + str(reason))
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
        """Release the control lock; identity enrollment is reset only on restart."""
        with self._control_update_lock:
            self._follow_controller.clear_active_target(reason)
            self.search_state = self._follow_controller.search_state
            self.search_direction = self._follow_controller.search_direction
            self.lost_confirm_frames = self._follow_controller.lost_confirm_frames
            self._vision_control_state = "initial_wait"
            self._clear_longitudinal_context(reason="active_target_cleared")
            self._clear_lateral_intent("active_target_cleared")
            self._current_forward_percent = self._current_steer_base_percent = 0
            self._current_steer_correction_rpm = self._current_rotate_raw_target = 0
            self._current_rotate_raw_source = "initial_lock_required"
            self._current_rotate_pulse_enabled = False
            self._lateral_yaw_revision = int(getattr(self, "_lateral_yaw_revision", 0)) + 1
            self.is_forwarding = False
            # Draining alone cannot retire a packet already popped by the
            # action thread. Invalidate it under the final motor-write lock.
            with self.motor_io_lock, self.action_queue_lock:
                self._action_command_revision = int(getattr(self, "_action_command_revision", 0)) + 1
            self._reset_search_geometry_reacquire()
            self._clear_visual_reacquire_hold("active_target_cleared")
            self._reacquire_depth_pending = False
            self._reacquire_depth_pending_uid = None
        if stop_current:
            self._clear_action_queue(f"clear_active_target:{reason}")
            if not self._should_skip_redundant_direct_stop("clear_active_target"):
                self._prepare_direct_stop("clear_active_target")
                self._action_runtime.send_stop_with_brake_hold("clear_active_target")
                self._mark_direct_stop_sent("clear_active_target")

    def _persons_to_targets(
        self, persons: List[Tuple], *, width: int = 0, height: int = 0,
    ) -> List[PersonTarget]:
        tracker = getattr(getattr(self, "_rknn_pipeline", None), "tracker", None)
        observations = getattr(tracker, "last_identity_observations", ()) or ()
        initial_uid = getattr(getattr(tracker, "identity_bank", None), "_initial_enrolled_uid", None)
        startup_pending = not getattr(getattr(self, "_follow_controller", None), "_has_seen_person", False)
        capture_id = int(getattr(self, "_active_capture_frame_id", 0))
        capture_ts = float(getattr(self, "_active_capture_timestamp", 0.0))
        targets: List[PersonTarget] = []
        for bbox, track_id, conf, area in persons:
            if startup_pending and initial_uid is not None and int(track_id) != initial_uid:
                # Enrollment may have finished on a late frame. Do not let a
                # later sole/larger UID become the first control lock instead.
                continue
            initial_confirmed = resolve_initial_identity_confirmation(
                target_id=int(track_id), display_bbox=bbox,
                capture_frame_id=capture_id, capture_timestamp=capture_ts,
                observations=observations, width=int(width), height=int(height),
            )
            if startup_pending and initial_uid is not None and not initial_confirmed:
                # The official enrollment path must not fall back to counting
                # cached mapped UIDs when current appearance proof is missing.
                continue
            targets.append(
                PersonTarget(
                    bbox=tuple(float(v) for v in bbox),
                    track_id=int(track_id),
                    confidence=float(conf),
                    area=float(area),
                    initial_identity_confirmed=initial_confirmed,
                    depth_observation=resolve_depth_target_observation(
                        target_id=int(track_id), display_bbox=bbox,
                        capture_frame_id=capture_id, capture_timestamp=capture_ts,
                        observations=observations, width=int(width), height=int(height),
                    ),
                    braking_observation=(resolve_braking_target_observation(
                        target_id=int(track_id), display_bbox=bbox,
                        capture_frame_id=capture_id, capture_timestamp=capture_ts,
                        observations=observations, width=int(width), height=int(height),
                    ) if int(track_id) == getattr(getattr(self, "_follow_controller", None), "active_target_id", None) else None),
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
            "来源=%s 原始采样距离=%s米 滤波距离=%s米 控制距离=%s米 触发代码=%s "
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
            and (
                not LATERAL_INTENT_CONTROL_ENABLE
                or not hasattr(self, "_lateral_intent_store")
                or self._has_fresh_lateral_yaw(target_id)
            )
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

    def _follow_wheel_axes(self, now: float, *, quiet: bool = False):
        """Canonical normal-follow axes, independent of the last action name."""
        uid = getattr(self._follow_controller, "active_target_id", None)
        revision = self._lateral_yaw_revision
        linear = (self._fresh_depth_linear_snapshot(uid, now=now, quiet=True) if quiet
                  else self._fresh_depth_linear_snapshot(uid, now=now))
        if linear is not None and linear[0] == "backward":
            return None  # reverse retains its dedicated safety/zero-cross path
        base = 0.0 if linear is None else linear[1] * MOTOR_FORWARD_MAX_TARGET_RPM / 100.0
        if getattr(self, "_vision_control_state", "") == "target_visible_low_quality":
            base = 0.0
        yaw = 0.0
        if self._has_fresh_lateral_yaw(uid):
            intent = self._lateral_intent_store.snapshot()
            if intent is not None and intent.valid(now) and intent.target_id == uid:
                correction = (self._lateral_intent_last_correction_rpm
                              if self._lateral_intent_last_sequence == intent.sequence
                              else intent.initial_correction_rpm)
                limit = max(0.0, float(intent.correction_limit_rpm))
                yaw = max(-limit, min(limit, float(correction)))
        if (revision != self._lateral_yaw_revision
                or quiet and uid != getattr(self._follow_controller, "active_target_id", None)):
            return None
        if not motion_identity_live(self, uid, max(now, time.monotonic())):
            return uid, revision, 0.0, 0.0
        return uid, revision, base, yaw

    def _has_fresh_lateral_yaw(self, target_id: Optional[int] = None) -> bool:
        """Neither a queued rotate nor bbox geometry can revive a revoked yaw."""
        intent = self._lateral_intent_store.snapshot()
        now = time.monotonic()
        if (
            intent is None
            or not intent.valid(now)
            or intent.hold_zero
            or int(getattr(self, "_lateral_intent_zero_sequence", -1)) == intent.sequence
            or (target_id is not None and intent.target_id != int(target_id))
            or getattr(self._follow_controller, "active_target_id", None) != intent.target_id
            or not motion_identity_live(self, intent.target_id, now)
            or self.search_state != "none"
            or not self._vision_control_state.startswith("target_visible")
            or self._explicit_stop_requested
            or self._runtime_shutdown_requested
            or not self.running
        ):
            return False
        if intent.nominal_valid_until > 0.0 and now > intent.nominal_valid_until:
            get_feedback = getattr(getattr(self, "_action_runtime", None), "get_steering_feedback", None)
            if not intent.continuation_allowed(now, get_feedback() if callable(get_feedback) else None):
                return False  # Last-moment writer gate, even if the control loop is delayed.
        if not motion_identity_live(self, intent.target_id, time.monotonic()):
            return False  # A feedback read must not cross the full identity deadline.
        if int(getattr(self, "_lateral_intent_last_sequence", -1)) == intent.sequence:
            return int(getattr(self, "_lateral_intent_last_correction_rpm", 0)) != 0
        return intent.initial_correction_rpm != 0

    @staticmethod
    def _depth_longitudinal_cap_percent(distance_m: Optional[float] = None) -> int:
        near_cap = int(ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT)
        far_cap = int(ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT)
        cap = near_cap
        if distance_m is not None:
            try:
                distance = float(distance_m)
            except (TypeError, ValueError):
                distance = float("nan")
            if math.isfinite(distance):
                near_distance = float(ASTRA_DEPTH_NEAR_GUARD_DISTANCE_M)
                far_distance = max(
                    near_distance + 0.01,
                    float(ASTRA_DEPTH_LONGITUDINAL_FAR_DISTANCE_M),
                )
                progress = max(
                    0.0,
                    min(1.0, (distance - near_distance) / (far_distance - near_distance)),
                )
                cap = int(round(near_cap + (far_cap - near_cap) * progress))
        return max(0, min(100, int(cap)))

    @staticmethod
    def _cap_depth_longitudinal_actions(
        actions: List[ControlAction],
        distance_m: Optional[float] = None,
        tracking_base_rpm: Optional[float] = None,
        distance_control_percent: Optional[float] = None,
        distance_pi_percent: Optional[float] = None,
    ) -> List[ControlAction]:
        """Cap Depth speed, ramping up only when a valid target is far away."""
        cap = PersonTracker._depth_longitudinal_cap_percent(distance_m)
        forward_cap = cap
        # No speed prior is needed for bounded distance correction. This is
        # permission for an already calculated fresh PID request, NOT a speed
        # floor or an estimate of target velocity. Callers without qualified
        # current evidence retain the original distance-only cap.
        if distance_control_percent is not None and distance_m is not None:
            try:
                budget = float(distance_control_percent)
                distance = float(distance_m)
                if (math.isfinite(budget) and budget > 0 and math.isfinite(distance)
                        and distance >= max(FOLLOW_FORWARD_START_DISTANCE_M,
                                            TARGET_DISTANCE + DISTANCE_PID_DEADBAND_M)):
                    ordinary_max = (max(0.0, float(FORWARD_MIN_RPM)) + 20.0) / max(1.0, float(FORWARD_MAX_RPM)) * 100.0
                    if (DISTANCE_APPROACH_ENABLE or DISTANCE_CONTROL_MODE == "distance_pi") and not FOLLOW_ROTATION_ONLY:
                        ordinary_max = max(ordinary_max, DISTANCE_APPROACH_NO_MATCHING_MAX_RPM /
                                           max(1.0, float(FORWARD_MAX_RPM)) * 100.0)
                    forward_cap = max(cap, int(min(
                        float(ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT), ordinary_max, budget,
                    )))
            except (ValueError, TypeError, OverflowError):
                pass
        if tracking_base_rpm is not None:
            try:
                base = float(tracking_base_rpm)
                if math.isfinite(base) and base > 0.0:
                    rpm_max = max(1.0, float(FORWARD_MAX_RPM))
                    extra_limit = float(ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT) + 20.0 / rpm_max * 100.0
                    if DISTANCE_MATCHING_BASE_MAX_RPM > 0.0:
                        # The caller supplies a qualified matching estimate,
                        # not an arbitrary speed floor. Do not re-apply the
                        # legacy extra-FF budget to the whole matching base.
                        extra_limit = min(100.0, DISTANCE_MATCHING_BASE_MAX_RPM, rpm_max) / rpm_max * 100.0
                    tracking_cap = min(
                        float(ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT),
                        extra_limit, base / rpm_max * 100.0,
                    )
                    # Matching speed is a baseline, not the total catch-up
                    # ceiling. Permit up to 20 RPM of *requested* correction
                    # outside the setpoint deadband, within the same far cap.
                    correction_percent = 0.0
                    if distance_m is not None and math.isfinite(float(distance_m)) and (
                        float(distance_m) > TARGET_DISTANCE + DISTANCE_PID_DEADBAND_M
                    ):
                        requested = max((a.speed_percent for a in actions if a.kind == "forward"), default=0)
                        correction_rpm = 20.0
                        if (DISTANCE_APPROACH_ENABLE or DISTANCE_CONTROL_MODE == "distance_pi") and not FOLLOW_ROTATION_ONLY:
                            correction_rpm = (60.0 * DISTANCE_APPROACH_MAX_CATCHUP_M_S /
                                              VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M)
                        correction_percent = min(correction_rpm / rpm_max * 100.0,
                                                 max(0.0, requested - tracking_cap))
                    forward_cap = max(forward_cap, int(round(min(
                        float(ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT),
                        tracking_cap + correction_percent,
                    ))))
            except (ValueError, TypeError, OverflowError):
                pass
        if distance_pi_percent is not None:
            # Only the exact new qualified PI sample supplies this budget.
            # I is a learned forward speed, so e=0 is not permission to erase
            # it with the legacy near-distance/no-matching ceilings. Neither
            # a diagnostic FF estimate nor its lease participates in this cap.
            forward_cap = 0
            try:
                budget = float(distance_pi_percent)
                if not FOLLOW_ROTATION_ONLY and math.isfinite(budget):
                    forward_cap = max(0, int(min(100., float(ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT), budget)))
            except (ValueError, TypeError, OverflowError):
                pass
        # Reverse distances are on their own (lower) side of the setpoint.
        # Do not let a forward ceiling change create permission to reverse
        # fast from an inconsistent far-distance action.
        reverse_cap = min(cap, int(round(100.0 * FOLLOW_REVERSE_RUNTIME_CAP_RPM / max(1, FORWARD_MAX_RPM))))
        capped: List[ControlAction] = []
        for action in actions:
            speed = int(getattr(action, "speed_percent", 0))
            if action.kind == "forward" and speed > forward_cap:
                capped.append(ControlAction.forward(forward_cap, action.reason))
            elif action.kind == "backward" and speed > reverse_cap:
                capped.append(
                    ControlAction.backward(
                        reverse_cap,
                        action.reason,
                        correction_rpm=int(action.steer_correction_rpm),
                    )
                )
            else:
                capped.append(action)
        return capped

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
        if LATERAL_INTENT_CONTROL_ENABLE and hasattr(self, "_lateral_intent_store"):
            if not self._has_fresh_lateral_yaw():
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

    def _depth30_should_preserve_lateral_motion(
        self,
        target: Optional[PersonTarget],
        actions: List[ControlAction],
        *,
        width: int,
    ) -> bool:
        """Keep a fresh visible yaw owner from being replaced by Depth30.

        Depth30 owns forward/backward speed, but its zero-speed hold is not a
        lateral command.  When a visible target is already asking for a yaw
        correction, queuing ``forward(0)`` creates a stop/turn race and adds a
        full visual-cycle of dead time.  Only zero-forward Depth decisions
        are suppressed here; positive forward/reverse commands and all hard
        safety stops retain their existing behavior.
        """
        if not self._depth_longitudinal_authority_enabled():
            return False
        if target is None or not actions:
            return False
        if any(
            action.kind != "forward" or int(getattr(action, "speed_percent", 0)) != 0
            for action in actions
        ):
            return False
        target_id = int(target.track_id)
        if self.search_state != "none" or not str(self._vision_control_state).startswith(
            "target_visible"
        ):
            return False
        vision_age = max(0.0, time.monotonic() - float(self._last_vision_control_ts))
        # The handoff is only valid while the camera-loop command is fresh.
        # This preserves the existing stale-vision safety behavior instead of
        # keeping an old yaw command alive through repeated Depth ticks.
        if vision_age > min(0.25, max(0.05, float(ROTATE_HOLD_STALE_SEC))):
            return False
        if LATERAL_INTENT_CONTROL_ENABLE and hasattr(self, "_lateral_intent_store"):
            # No fallback to an old queued turn or off-centre rectangle after
            # a zero/expired intent. That fallback used to undo PID braking.
            return self._has_fresh_lateral_yaw(target_id)

        # A queued or active yaw command is already the latest lateral state.
        with self.command_lock:
            current_command = self.current_command
        if current_command in ROTATE_ACTIONS:
            return True
        with self.action_queue_lock:
            pending_actions = self._action_queue_snapshot_locked()
        if any(action in ROTATE_ACTIONS for action in pending_actions):
            return True

        # The camera frame may have published its target before the lateral
        # worker has executed the first tick.  Use its geometry as a bounded
        # one-frame handoff hint, never as an identity or search decision.
        center_left = float(getattr(self._follow_controller.cfg, "center_left_ratio", 0.45))
        center_right = float(getattr(self._follow_controller.cfg, "center_right_ratio", 0.55))
        x_ratio = float(target.center[0]) / float(max(1, int(width)))
        if not math.isfinite(x_ratio):
            return False
        if not (x_ratio < center_left or x_ratio > center_right):
            return False
        pid_result = getattr(self._follow_controller, "last_steering_pid_result", None)
        pid_correction = 0 if pid_result is None else int(getattr(pid_result, "correction_rpm", 0))
        last_correction_target = getattr(self, "_last_vision_correction_target_id", None)
        if pid_correction == 0 and last_correction_target != target_id:
            return False
        return bool(pid_correction != 0 or x_ratio < center_left or x_ratio > center_right)

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

    def _finish_limited_yaw_handoff(self, uid, capture_frame_id, lateral_intent_published):
        """Finish a queue-only mode switch without clearing any real stop."""
        runtime = getattr(self, "_action_runtime", None)
        check = getattr(runtime, "can_handoff_limited_yaw", None)
        accepted = bool(lateral_intent_published and callable(check)
                        and check(uid, capture_frame_id))
        if not accepted:
            self.stop_action_execution = True
        logger.info("limited_yaw_decision_handoff uid=%s capture_frame_id=%s accepted=%s "
                    "queue_cancelled=True motor_stop_requested=%s "
                    "forward_authorized=False reversal_guard=preserved",
                    uid, capture_frame_id, accepted, not accepted)
        return accepted

    def _publish_follow_recording(self, frame, decision, source):
        """Diagnostics only, called under the existing control-update lock."""
        if getattr(self, '_camera_video_recorder', None) is None:
            return
        try:
            short = getattr(self, '_short_follow', None)
            paired_snapshot = (short.snapshot() if short is not None and short.config.enabled else None)
            self._video_follow_snapshot = build_follow_snapshot(
                self._follow_controller, frame, decision, source, time.monotonic(),
                **({'paired_snapshot': paired_snapshot} if paired_snapshot is not None else {}))
        except Exception as exc:
            # Recording must never prevent a motor safety/stop decision.
            self._video_follow_snapshot = None
            if not getattr(self, '_video_follow_error_logged', False):
                self._video_follow_error_logged = True
                logger.warning('Follow recording diagnostics unavailable: %s', exc)

    def _refresh_distance_control_feedback(self, previous, distance_state, capture_id):
        """Re-read the existing encoder cache AFTER potentially slow ranging.

        No serial reads or motor writes. Geometry/ranging retains its original
        feedback; decision feedback retains its real timestamp, and the usual
        age/skew/trust gates still apply. Never conceal a newer fault by falling
        back to an older healthy sample. Legacy/search-only policy is unchanged.
        """
        if (not getattr(self._follow_controller, "distance_pi_enabled", False)
                or getattr(self._follow_controller, "search_state", "none") != "none"):
            return previous
        try:
            latest = self._action_runtime.get_steering_feedback()
        except Exception as exc:
            logger.warning("distance_feedback_refresh capture_frame_id=%s status=read_failed error=%s",
                           capture_id, exc)
            return None
        if latest is None:
            # No new observation: existing evidence can only retain its
            # original age, never manufacture a fresh wheel-speed sample.
            return previous
        if not isinstance(latest, SteeringFeedback) or not math.isfinite(latest.timestamp):
            logger.warning("distance_feedback_refresh capture_frame_id=%s status=invalid_snapshot", capture_id)
            return None
        now = time.monotonic()
        if latest.timestamp > now:
            logger.warning("distance_feedback_refresh capture_frame_id=%s status=future_snapshot", capture_id)
            return None
        if (previous is not None and math.isfinite(previous.timestamp)
                and latest.timestamp < previous.timestamp):
            return previous
        if latest is not previous:
            stamp = distance_state.sample_timestamp
            logger.info(
                "distance_feedback_refresh capture_frame_id=%s status=cache_resampled "
                "old_age_ms=%s new_age_ms=%.1f feedback_ts=%.6f depth_sample_ts=%s "
                "depth_skew_ms=%s trustworthy=%s serial_read=False deadline_renewed=False",
                capture_id, None if previous is None else (now-previous.timestamp)*1000.,
                (now-latest.timestamp)*1000., latest.timestamp, stamp,
                None if stamp is None else (latest.timestamp-stamp)*1000., latest.trustworthy,
            )
        return latest

    def _observe_cropped_target_depth(self, persons, width, height, capture_id, capture_ts, feedback):
        """Low-quality follow branch may measure, never grant longitudinal trust."""
        runtime = getattr(self, '_distance_runtime', None)
        measure = getattr(runtime, 'get_cropped_observation_state', None)
        if not callable(measure):
            return None
        empty = DistanceState(source='cropped_depth_observation', source_detail='crop_observation:unqualified')
        tracker = getattr(getattr(self, '_rknn_pipeline', None), 'tracker', None)
        bank = getattr(tracker, 'identity_bank', None)
        rows = tuple(getattr(tracker, 'last_identity_observations', ()) or ())
        active_uid = getattr(self._follow_controller, 'active_target_id', None)
        if bank is None or len(persons) != 1 or len(rows) != 1 or persons[0][1] != active_uid:
            return empty
        observation, reason = resolve_cropped_depth_observation(
            bank=bank, observations=rows, uid=active_uid, raw_track_id=rows[0].get('raw_track_id'),
            display_bbox=persons[0][0], capture_id=capture_id, capture_timestamp=capture_ts,
            width=width, height=height, now=time.monotonic())
        key = (active_uid, capture_id, capture_ts)
        if observation is None or getattr(self, '_cropped_depth_last_capture', None) == key:
            return replace(empty, source_detail='crop_observation:' + (reason if observation is None else 'duplicate_capture'))
        self._cropped_depth_last_capture = key
        started = time.perf_counter()
        state = measure(width, height, observation, steering_feedback=feedback)
        logger.info('cropped_depth_observation cap=%s uid=%s raw_track=%s distance=%s '
                    'sample_ts=%s age_ms=%s detail=%s processing_ms=%.2f '
                    'identity_confirmed=False template_updated=False motion_authorized=False',
                    capture_id, active_uid, observation.raw_track_id, state.used_distance_m,
                    state.sample_timestamp, None if state.sample_age_sec is None else state.sample_age_sec*1000,
                    state.source_detail, (time.perf_counter()-started)*1000)
        return state

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
        depth_target_snapshot: Optional[Tuple[PersonTarget, ...]] = None,
        evidence_capture_frame_id: Optional[int] = None,
        evidence_capture_timestamp: Optional[float] = None,
        prepared_depth=None,
        depth_state_replay: Optional[DistanceState] = None,
        visual_lateral_only: bool = False,
        range_deferred: bool = False,
    ) -> List[int]:
        """Use car_control_modular.controllers to decide actions."""
        self._short_follow_handled_frame = False
        short = getattr(self, "_short_follow_adapter", None)
        if short is not None and short.hold_explicit_stop(
                (getattr(self, "_active_capture_timestamp", 0.)
                 if evidence_capture_timestamp is None else evidence_capture_timestamp),
                time.monotonic()):
            # Let the single executor complete and acknowledge STOP. An
            # ordinary frame may not erase an unconsumed explicit request.
            self._last_control_decision_reason = self._last_explicit_stop_reason or "explicit_stop"
            return [ACTION_STOP]
        self._explicit_stop_requested = False
        self._explicit_stop_provenance = None
        if control_source == "vision":
            # Set again only when this frame publishes a valid visible-target
            # intent. Search, safety and low-quality hold actions remain direct.
            self._lateral_intent_owned_frame = -1
        if FOLLOW_ROTATION_ONLY and control_source == "depth30":
            return []
        if control_source != "depth30":
            self._last_vision_control_frame_index = int(self.frame_index)
            self._last_vision_control_ts = time.monotonic()
        parked = getattr(self, "_follow_distance_hold", None)
        if parked is not None and (
            self.search_state != "none"
            or getattr(self._follow_controller, "search_state", "none") != "none"
            or getattr(self._follow_controller, "active_target_id", None) != parked.uid
        ):
            self._follow_distance_hold = None
        self._follow_controller.set_last_dispatched(self._action_int_to_kind(self._last_dispatched_action))

        obstacles_dict = self._get_obstacle_status()
        if depth_target_snapshot is not None:
            # The inference thread clears/rebuilds its observations while
            # Depth30 runs. Consume the immutable published capture instead.
            if [(tuple(p.bbox), int(p.track_id)) for p in depth_target_snapshot] != [
                (tuple(person[0]), int(person[1])) for person in persons
            ]:
                raise ValueError("Depth target snapshot does not match control persons")
            person_targets = list(depth_target_snapshot)
        else:
            person_targets = self._persons_to_targets(persons, width=width, height=height)
        capture_id = (
            int(getattr(self, "_active_capture_frame_id", 0))
            if evidence_capture_frame_id is None else int(evidence_capture_frame_id)
        )
        capture_ts = (
            float(getattr(self, "_active_capture_timestamp", 0.0))
            if evidence_capture_timestamp is None else float(evidence_capture_timestamp)
        )
        distance_target = self._distance_runtime.select_target(person_targets)
        steering_feedback = self._action_runtime.get_steering_feedback()
        priority_latest = self._visible_latest_depth_eligible(
            person_targets, distance_target, capture_id, capture_ts,
            control_source=control_source, target_steerable=target_steerable,
            low_quality_visible=low_quality_visible,
        )
        if priority_latest and not range_deferred:
            # An optimistic visual scan already published this exact context
            # before releasing the lock. Do not replace its identity token or
            # invalidate a Depth30 transaction that used it in the meantime.
            if prepared_depth is None and depth_state_replay is None:
                self._publish_longitudinal_context(
                    width, height, persons, person_targets=tuple(person_targets),
                )
            depth_use_latest = True
        ranging_started = time.perf_counter()
        ranging_cpu_started = time.thread_time()
        if range_deferred:
            # Display-only copy: no raw sample or sample_timestamp, no fusion
            # mutation, and no permission to renew the motor's depth clock.
            distance_state = self._async_visual_distance_display(distance_target)
            if not visual_lateral_only:
                distance_state = DistanceState(source="vision_depth", source_detail="depth_async_pending")
        elif depth_state_replay is not None:
            # Depth30 won the optimistic commit. This is an observation-only
            # copy of that same target's committed sample, never another scan
            # or a new physical sample/longitudinal authorization.
            distance_state = depth_state_replay
        elif low_quality_visible:
            # Explicit observation permission, not a normal target or motion
            # ROI. The normal fusion anchor and Depth30 context remain closed.
            observed = PersonTracker._observe_cropped_target_depth(
                self, persons, width, height, capture_id, capture_ts, steering_feedback)
            distance_state = observed if observed is not None else (
                self._last_frame_distance_state or DistanceState(source="vision_depth"))
        else:
            distance_state = self._distance_runtime.get_frame_distance_state(
                int(width),
                distance_target,
                frame_height=int(height),
                steering_feedback=steering_feedback,
                target_distance_m=TARGET_DISTANCE,
                brake_distance_m=FOLLOW_BRAKE_DISTANCE_M,
                depth_use_latest=bool(depth_use_latest),
                capture_timestamp=(
                    None
                    if depth_use_latest
                    else capture_ts
                ),
                **({} if prepared_depth is None else {"prepared_depth": prepared_depth}),
            )
        # Diagnostic state only, written/read under _control_update_lock.
        # Depth30/visible tracking scan a private filter snapshot off-lock. Only a validated
        # commit and the fusion anchor are published under this mutex.
        self._last_control_ranging_timing = (
            control_source, capture_id,
            (time.perf_counter() - ranging_started) * 1000.0,
            (time.thread_time() - ranging_cpu_started) * 1000.0,
        )
        if priority_latest and not range_deferred:
            logger.info(
                "vision_depth_priority capture_frame_id=%s uid=%s roi_age_ms=%.1f "
                "request=latest_or_bounded_history sample_ts=%s sample_age_ms=%s "
                "capture_aligned_replay=False deadline_renewed=False",
                capture_id, distance_target.track_id,
                (time.monotonic() - capture_ts) * 1000.0,
                distance_state.sample_timestamp,
                None if distance_state.sample_age_sec is None else distance_state.sample_age_sec * 1000.0,
            )
        # CAP269: an 84ms depth calculation aged the pre-ranging wheel snapshot
        # past150ms despite newer encoder samples already being cached. Do not
        # turn that stale *copy* into a stationary-target brake fallback.
        steering_feedback = self._refresh_distance_control_feedback(
            steering_feedback, distance_state, capture_id,
        )
        if not range_deferred:
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
            capture_frame_id=capture_id,
            capture_timestamp=capture_ts,
            lateral_candidate=(
                lateral_candidate if control_source == "vision" else None
            ),
        )
        self._depth_roi_safety_clear = not (
            frame.hazard.active or frame.obstacles.front or frame.obstacles.left or frame.obstacles.right)
        self._last_command_source_module = str(control_source or "vision")
        self._last_command_control_frame = int(self.frame_index)
        self._last_decision_capture_frame = int(frame.capture_frame_id)
        self._last_command_capture_frame = int(frame.capture_frame_id)
        self._last_command_capture_timestamp = float(frame.capture_timestamp)
        depth_authority = self._depth_longitudinal_authority_enabled()
        fresh_depth_checker = getattr(
            self._follow_controller,
            "_is_fresh_depth_state",
            None,
        )
        is_fresh_depth = not range_deferred and (bool(fresh_depth_checker(frame)) if callable(fresh_depth_checker) else bool(
            str(getattr(frame.distance_state, "source", "")) == "vision_depth"
            and getattr(frame.distance_state, "raw_distance_m", None) is not None
            and str(getattr(frame.distance_state, "source_detail", "")).startswith("depth_")
            and not str(getattr(frame.distance_state, "source_detail", "")).endswith("_hold")
        ))
        short = getattr(self, "_short_follow_adapter", None)
        if short is not None and short.handle(
                frame, distance_target, is_fresh_depth=is_fresh_depth,
                control_source=control_source, target_steerable=target_steerable,
                low_quality_visible=low_quality_visible, now=time.monotonic()):
            return []
        if not range_deferred and is_follow_distance_hold(self):
            self._observe_follow_distance_hold(
                frame, distance_target, fresh=is_fresh_depth,
                steerable=target_steerable, low_quality=low_quality_visible,
            )
            if control_source == "depth30" and self._brake_hold_active:
                # Measuring during parking must not run PID, publish motor
                # authorization or mutate the action queue before release.
                return []
        if control_source == "depth30" and depth_authority and not is_fresh_depth:
            # Keep the already-approved axes and estimator untouched on a
            # discarded observation. Safety/state/UID/physical TTL checks are
            # evaluated first; no timestamp, speed or queue is refreshed.
            if target_steerable and not low_quality_visible and self._preserve_depth_replay(
                frame, self._follow_controller.active_target_id,
            ):
                return []
            # A fused/held distance is useful for diagnostics and lateral
            # tracking, but it must not restart the longitudinal PID.  The
            # Depth loop will issue its own zero/hold decision below.
            if frame.distance_m is not None:
                logger.info(
                    "depth30_reject_nonfresh_distance frame=%d capture_frame_id=%d "
                    "distance=%.3fm detail=%s age=%s",
                    int(self.frame_index),
                    int(frame.capture_frame_id),
                    float(frame.distance_m),
                    str(getattr(frame.distance_state, "source_detail", "")),
                    "none"
                    if getattr(frame.distance_state, "sample_age_sec", None) is None
                    else "%.0fms" % (float(frame.distance_state.sample_age_sec) * 1000.0),
                )
            frame = SensorFrame(
                width=frame.width,
                height=frame.height,
                persons=frame.persons,
                hazard=frame.hazard,
                obstacles=frame.obstacles,
                distance_m=None,
                distance_state=frame.distance_state,
                steering_feedback=frame.steering_feedback,
                module_status=frame.module_status,
                capture_frame_id=frame.capture_frame_id,
                capture_timestamp=frame.capture_timestamp,
                lateral_candidate=frame.lateral_candidate,
            )
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
            **({"lateral_only": True} if visual_lateral_only else {}),
        )
        # Startup/search still choose and validate the UID. Intercept their
        # first successful normal frame BEFORE publishing any legacy PI/yaw
        # action: paired PI must not briefly inherit the legacy 180RPM launch
        # boost or its separately managed longitudinal/yaw authorization.
        if (short is not None and (not decision.explicit_stop_requested
                or decision.reason.startswith("target_distance_")
                or decision.reason in {"distance_too_close", "person_too_close_no_rotate"})
                and not decision.shutdown_requested and short.handle(
                    frame, self._follow_controller.last_selected_target,
                    is_fresh_depth=is_fresh_depth, control_source=control_source,
                    target_steerable=target_steerable,
                    low_quality_visible=low_quality_visible, now=time.monotonic())):
            return []
        initial_lock_pending = self._follow_controller.active_target_id is None
        if initial_lock_pending:
            # Runtime invariant, independent of legacy controller reasons:
            # an unbound candidate never owns either motion axis.
            initial_reason = decision.reason if (
                decision.explicit_stop_requested or decision.shutdown_requested
                or decision.reason.startswith(("initial_", "wait_first_person"))
            ) else "initial_lock_required"
            safety_stop = bool(decision.explicit_stop_requested or decision.shutdown_requested)
            decision = replace(
                decision,
                actions=[ControlAction.stop(initial_reason, brake_hold=safety_stop)],
                reason=initial_reason, clear_action_queue=True,
                stop_action_execution=True, soft_stop_requested=not safety_stop,
                current_forward_percent=0, is_forwarding=False,
                near_yaw_park_requested=False,
            )
            self._vision_control_state = "initial_wait"
            self._clear_longitudinal_context(reason="initial_lock_required")
            self._clear_lateral_intent("initial_lock_required")
            self._current_forward_percent = self._current_steer_base_percent = 0
            self._current_steer_correction_rpm = self._current_rotate_raw_target = 0
            self._current_rotate_raw_source = "initial_lock_required"
            self._current_rotate_pulse_enabled = False
            self.is_forwarding = False
        depth_linear_actions = list(decision.actions)
        if control_source == "depth30" and depth_authority and not initial_lock_pending:
            if decision.explicit_stop_requested or decision.shutdown_requested:
                self._revoke_depth_linear_authority("depth_safety:" + str(decision.reason))
            else:
                # A fixed older ROI may complete after a newer visual CAP.
                # Admission belongs to the ROI that selected this physical
                # depth, never to the newer visual bbox/selection record.
                current_target = distance_target
                depth_linear_actions, _committed = self._commit_depth_linear_decision(
                    decision, frame,
                    None if current_target is None else current_target.track_id,
                    is_fresh_depth=is_fresh_depth,
                )
        self._publish_follow_recording(frame, decision, control_source)
        evidence_capture_frame = decision.evidence_capture_frame_id
        if evidence_capture_frame is not None and int(evidence_capture_frame) > 0:
            self._last_command_capture_frame = int(evidence_capture_frame)

        longitudinal_only = control_source == "depth30"
        decision_target = (None if initial_lock_pending else distance_target
                           if longitudinal_only else self._follow_controller.last_selected_target)
        preserve_depth_lateral = bool(
            not initial_lock_pending and longitudinal_only
            and self._depth30_should_preserve_lateral_motion(
                decision_target,
                depth_linear_actions,
                width=int(width),
            )
        )
        # Depth30 may have just issued a valid translation while this slower
        # visual frame reports a single missed detection.  Keep that command
        # for the bounded handoff window; otherwise the visual lost-history
        # rotate action clears the Depth queue and creates a same-frame
        # forward-versus-rotate race.  Once the window expires (or search
        # becomes active), the normal lost/search action is allowed through.
        preserve_depth_translation = bool(
            not initial_lock_pending and control_source == "vision"
            and depth_authority
            and self._should_preserve_depth30_translation_for_visual_loss(
                decision.reason
            )
        )
        if preserve_depth_translation:
            logger.info(
                "vision_lost_hold_preserve_depth30 frame=%d reason=%s "
                "depth30_kind=%s age=%.0fms action=keep_translation",
                int(self.frame_index),
                str(decision.reason),
                str(self._last_depth30_translation_kind),
                max(
                    0.0,
                    time.monotonic() - float(self._last_depth30_translation_ts),
                )
                * 1000.0,
            )
        if not initial_lock_pending and longitudinal_only and self._depth_zero_longitudinal_preserves_visible_rotation(
            decision
        ):
            # Rotation already has zero longitudinal wheel speed.  Depth has
            # fulfilled its safety responsibility without replacing the
            # camera PID command or clearing its action queue.
            return []
        if not longitudinal_only:
            previous_control_state = self._vision_control_state
            if initial_lock_pending:
                control_state = "initial_wait"
            elif decision_target is not None:
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
            if (not control_state.startswith("target_visible")
                    or (decision_target is not None
                        and getattr(self, "_near_yaw_park_request", None) is not None
                        and decision_target.track_id != self._near_yaw_park_request.uid)):
                self._release_near_yaw_park(
                    capture_id=frame.capture_frame_id, capture_timestamp=frame.capture_timestamp,
                    reason="visual_state_handoff:" + control_state, handoff=True,
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
            self._maybe_schedule_historical_direction_backfill(
                int(frame.capture_frame_id),
                active_before,
            )
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
        self._explicit_stop_provenance = ((
            "sensor_safety" if (frame.hazard.active or frame.obstacles.front
                or frame.obstacles.left or frame.obstacles.right) else
            "shutdown" if decision.shutdown_requested else "controller_state",
            decision.reason) if decision.explicit_stop_requested else None)
        self._last_control_decision_reason = decision.reason or ""
        if not getattr(decision, "lateral_only", False):
            self._release_near_yaw_park_for_decision(
                frame, decision_target, decision, control_source=control_source,
                target_steerable=target_steerable, low_quality_visible=low_quality_visible,
            )
        if (
            self.search_state != "none" or low_quality_visible or not target_steerable
            or decision.explicit_stop_requested or decision.shutdown_requested
            or (getattr(self, "_depth30_linear_snapshot", None) is not None
                and (decision_target is None
                     or decision_target.track_id != self._depth30_linear_snapshot[2]))
        ):
            self._revoke_depth_linear_authority("decision_safety_or_target:" + str(decision.reason))
        visual_depth_committed = bool(
            not initial_lock_pending and not longitudinal_only and not range_deferred
            and self._refresh_visual_depth_linear_authority(
                frame, decision_target, decision,
                is_fresh_depth=is_fresh_depth, target_steerable=target_steerable,
                low_quality_visible=low_quality_visible,
            )
        )
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
            and not depth_linear_actions
            and not decision.explicit_stop_requested
            and not decision.shutdown_requested
        ):
            # Missing/unchanged Depth is not a new motor target. Preserve the
            # latest visual speed, yaw correction and action queue.
            return []
        if decision.shutdown_requested:
            self._runtime_shutdown_requested = True
        visual_translation_suppressed = bool(
            control_source == "vision" and depth_authority
        )
        visual_stop_safety_override = bool(
            not initial_lock_pending and control_source == "vision"
            and depth_authority
            and decision.explicit_stop_requested
            and not self._visual_stop_may_override_depth(decision.reason, frame)
        )
        if not visual_translation_suppressed or (
            decision.explicit_stop_requested and not visual_stop_safety_override
        ):
            self.is_forwarding = decision.is_forwarding
            self._current_forward_percent = decision.current_forward_percent
        # A clipped-but-qualified target can retain yaw, not translation.
        # Cancel the old queued action now, but defer its generic motor-stop
        # flag until the NEW bounded yaw intent has been published and checked.
        # Never clear an already pending stop, or apply this to legacy writers.
        defer_limited_yaw_interrupt = bool(
            control_source == "vision" and not longitudinal_only
            and LATERAL_INTENT_CONTROL_ENABLE and depth_authority
            and decision.reason == "target_visible_low_quality_yaw"
            and decision.stop_action_execution and decision.clear_action_queue
            and not decision.explicit_stop_requested and not decision.shutdown_requested
            and not getattr(decision, "near_yaw_park_requested", False)
            and not getattr(decision, "soft_stop_requested", False)
            and active_before is not None and decision_target is not None
            and active_before == decision_target.track_id
            and getattr(getattr(getattr(self, "_action_runtime", None), "config", None),
                        "follow_wheel_period_sec", 0.0) >= .05)
        if defer_limited_yaw_interrupt:
            # Invalidate already-dequeued old snapshots BEFORE publishing the
            # replacement, not afterwards (which would invalidate it too).
            with self.action_queue_lock:
                self._action_command_revision = int(getattr(self, "_action_command_revision", 0)) + 1
        if (decision.stop_action_execution and not visual_stop_safety_override
                and not defer_limited_yaw_interrupt
                and not (LATERAL_INTENT_CONTROL_ENABLE
                         and getattr(decision, "near_yaw_park_requested", False))):
            self.stop_action_execution = True
        if visual_stop_safety_override:
            # The visual loop may request a translational hold at the same
            # time that Depth30 owns the longitudinal axis. Keep the request
            # for diagnostics, but do not let the later direct-stop branch
            # clear a valid Depth30 command.
            self._explicit_stop_requested = False
            self._last_explicit_stop_reason = ""
        if (
            decision.clear_action_queue
            and not visual_stop_safety_override
            and not preserve_depth_lateral
            and not preserve_depth_translation
        ):
            self._clear_action_queue(f"controller_clear:{decision.reason}")

        intent_runtime_actions = list(decision.actions)
        runtime_actions = list(depth_linear_actions if longitudinal_only else intent_runtime_actions)
        if preserve_depth_translation:
            runtime_actions = []
        if visual_translation_suppressed:
            # The camera loop still publishes its yaw intent below.  It must
            # not enqueue a forward/steer action or overwrite the Depth30
            # speed cap; safety stops and lost-target rotation remain direct.
            suppressed_actions = runtime_actions
            allow_direct = bool(
                initial_lock_pending
                or (decision.explicit_stop_requested and not visual_stop_safety_override)
                or self.search_state != "none"
                or decision.reason.startswith(("lost_", "search_"))
            )
            runtime_actions = [
                action
                for action in runtime_actions
                if action.kind in ("stop", "rotate_left", "rotate_right") and allow_direct
            ]
            if suppressed_actions and suppressed_actions != runtime_actions:
                logger.info(
                    "vision_longitudinal_action_suppressed frame=%d reason=%s "
                    "input=%s output=%s depth30_owner=True",
                    int(self.frame_index),
                    decision.reason,
                    [action.kind for action in suppressed_actions],
                    [action.kind for action in runtime_actions],
                )
        if visual_stop_safety_override:
            logger.info(
                "vision_stop_suppressed_depth30_owner frame=%d reason=%s "
                "depth30_owner=True safety_override=False",
                int(self.frame_index),
                str(decision.reason or "none"),
            )
        selected_target = None if initial_lock_pending else self._follow_controller.last_selected_target
        if control_source == "depth30" and not initial_lock_pending:
            selected_target = distance_target
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
            original_depth_actions = list(decision.actions)
            if runtime_actions != original_depth_actions:
                logger.info(
                    "30Hz Depth纵向速度限幅: frame=%d distance=%s cap=%d%% "
                    "near_cap=%d%% far_cap=%d%% far_distance=%.2fm input=%s output=%s",
                    int(self.frame_index),
                    "none" if frame.distance_m is None else "%.3f" % float(frame.distance_m),
                    int(self._depth_longitudinal_cap_percent(frame.distance_m)),
                    int(ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT),
                    int(ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT),
                    float(ASTRA_DEPTH_LONGITUDINAL_FAR_DISTANCE_M),
                    [
                        (action.kind, int(action.speed_percent))
                        for action in original_depth_actions
                    ],
                    [
                        (action.kind, int(action.speed_percent))
                        for action in runtime_actions
                    ],
                )
            runtime_actions = [
                self._merge_longitudinal_action_with_visual_steering(action, selected_target_id)
                for action in runtime_actions
            ]
            if preserve_depth_lateral and all(
                action.kind == "forward" and int(getattr(action, "speed_percent", 0)) == 0
                for action in runtime_actions
            ):
                with self.command_lock:
                    current_command_for_log = self.current_command
                with self.action_queue_lock:
                    pending_actions_for_log = self._action_queue_snapshot_locked()
                logger.info(
                    "depth30_zero_longitudinal_preserve_lateral frame=%d target=%s "
                    "reason=%s action=skip_queue current=%s pending=%s",
                    int(self.frame_index),
                    "none" if selected_target_id is None else int(selected_target_id),
                    str(decision.reason),
                    ACTION_NAMES.get(
                        current_command_for_log, str(current_command_for_log)
                    ),
                    self._action_queue_names_for_log(
                        pending_actions_for_log
                    ),
                )
                return []
        else:
            self._remember_vision_steering(runtime_actions, selected_target_id)

        actions: List[int] = []
        for action in runtime_actions:
            if action.kind == "forward":
                if longitudinal_only and depth_authority:
                    self._current_forward_percent = max(0, min(100, int(action.speed_percent)))
                    self._forward_speed_latched_percent = None
                else:
                    self._current_forward_percent = self._stabilize_forward_percent(
                        int(action.speed_percent), decision.reason,
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
                self._current_forward_percent = int(action.speed_percent)
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
                self._current_forward_allow_below_min = False
                self._current_steer_correction_rpm = 0
                self._current_steer_limit_reason = "none"
                pid_result = self._follow_controller.last_steering_pid_result
                # A near-distance target is still visible, so its in-place yaw
                # must use the same camera-angle + encoder-rate PID as parked
                # recentering. Only fully lost search keeps a fixed scan RPM.
                parked_recenter = (
                    decision.reason.startswith("person_parked_recenter_")
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
                        hold_min = min(
                            hold_limit,
                            int(VISIBLE_STEERING_PID_LOST_HOLD_MIN_CORRECTION_RPM),
                        )
                        rotate_raw = min(hold_limit, max(hold_min, previous_correction))
                        rotate_raw_source = "lost_confirm_pid_hold"
                    elif pid_result is None:
                        rotate_raw = PARKED_RECENTER_MIN_RPM
                        rotate_raw_source = "parked_recenter_fallback"
                    elif getattr(pid_result, "post_park_recenter", False):
                        # Use the same evidence-bounded ceiling as the lateral
                        # PID and final writer; do not reintroduce a fixed 4.
                        recenter = getattr(self._follow_controller,
                                           "post_park_recenter_limit", None)
                        ceiling = (recenter(getattr(decision_target, "track_id", None))
                                   if callable(recenter) else 4)
                        rotate_raw = min(4 if ceiling is None else ceiling,
                                         PARKED_RECENTER_MAX_RPM,
                                         abs(int(pid_result.correction_rpm)))
                        rotate_raw_source = "post_park_recenter"
                    elif getattr(pid_result, "outward_lead", None) is not None:
                        # The early trajectory lead deliberately sits below
                        # the ordinary 5RPM startup floor, including Depth30
                        # copies of this decision before the lateral tick.
                        rotate_raw = min(4, PARKED_RECENTER_MAX_RPM,
                                         abs(int(pid_result.correction_rpm)))
                        rotate_raw_source = "parked_outward_trajectory"
                    elif getattr(pid_result, "forward_phase", "") in {
                            "image_crop_outward_continuity", "image_outward_continuity"}:
                        # A brake-relief ceiling, never the normal startup
                        # floor. Keep the camera path and fast tick identical.
                        ceiling = (2 if pid_result.forward_phase == "image_crop_outward_continuity" else 4)
                        rotate_raw = min(ceiling, PARKED_RECENTER_MAX_RPM,
                                         abs(int(pid_result.correction_rpm)))
                        rotate_raw_source = pid_result.forward_phase
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
            runtime_actions=intent_runtime_actions,
            control_source=control_source,
            target_steerable=target_steerable,
            low_quality_visible=low_quality_visible,
            publish_depth_immediately=visual_depth_committed,
            near_yaw_park_requested=bool(getattr(decision, "near_yaw_park_requested", False)),
            distance_control_deferred=bool(getattr(decision, "lateral_only", False)),
        )
        if defer_limited_yaw_interrupt:
            self._finish_limited_yaw_handoff(
                selected_target_id, int(getattr(self, "_last_command_capture_frame", 0)),
                lateral_intent_published)
        if lateral_intent_published:
            logger.debug(
                "vision frame published to lateral action owner: frame=%d target=%s reason=%s",
                int(self.frame_index),
                selected_target_id,
                decision.reason,
            )

        if decision.reason:
            action_kinds = [a.kind for a in runtime_actions]
            target_dbg = distance_target if control_source == "depth30" else self._follow_controller.last_selected_target
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
                logger.info(
                    "control_direction_provenance processed_capture_frame_id=%s "
                    "latest_observation_capture_frame_id=%s action_evidence_capture_frame_id=%s "
                    "direction_source=%s",
                    frame.capture_frame_id,
                    getattr(self._follow_controller, "_direction_latest_visible_capture_id", None),
                    decision.evidence_capture_frame_id,
                    getattr(self._follow_controller, "_lost_hint_source", "none"),
                )
            if decision.explicit_stop_requested:
                self._log_distance_stop_trigger(decision.reason, frame.distance_state)
            pid_result = self._follow_controller.last_steering_pid_result
            if pid_result is not None:
                logger.info(
                    "视觉转向PID: control_frame_id=%d capture_frame_id=%d 视觉误差=%.2f度 延迟补偿误差=%.2f度 滤波误差=%.2f度 误差变化率=%.2f度/秒 目标速度有效=%s 目标画面角速度=%+.2f度/秒 目标方位角速度=%+.2f度/秒 目标速度前馈=%+.2f度/秒 目标速度匹配=%s/%.2f度/秒 期望右转角速度=%.2f度/秒 当前角速度上限=%.2f度/秒 编码器右转角速度=%.2f度/秒 反馈年龄=%.0f毫秒 反馈有效=%s 换向制动=%s 同向超调制动=%s 视觉方向保护=%s 预测制动=%s 制动延迟=%.0f毫秒 剩余角=%.2f度 停车角=%.2f度 角速度超调=%.2f度/秒 超调制动=%.2fRPM 有效下限=%.1fRPM(%s) 启动补偿=%s/%.0f毫秒/%s 边缘增强=%s 原始前进=%dRPM 动态前进=%dRPM 当前单轮修正上限=%.2fRPM 轮差限幅原因=%s 前馈=%.2fRPM 速度P=%.2fRPM 速度I=%.2fRPM 输出单轮修正=%dRPM 请求双轮差=%dRPM",
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
                    2 * pid_result.correction_rpm,
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
    def _search_candidate_bbox_diagnostic(
        bbox: Any,
        *,
        width: int,
        height: int,
        edge_margin_ratio: float,
        preferred_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
        """Return compact, JSON-safe geometry for search diagnostics.

        Keep this separate from control quality classification.  In
        particular, the raw detector box and expanded DeepSORT box must be
        visible side by side even when one of them is rejected for touching
        the image edge.
        """
        try:
            values = tuple(float(value) for value in bbox)
            if len(values) != 4:
                raise ValueError
            x1, y1, x2, y2 = values
        except (TypeError, ValueError):
            return {"bbox": None}
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        frame_width = max(1, int(width))
        frame_height = max(1, int(height))
        margin_x = float(frame_width) * max(0.0, float(edge_margin_ratio))
        margin_y = float(frame_height) * max(0.0, float(edge_margin_ratio))
        edge_touch_count = int(x1 <= margin_x) + int(frame_width - x2 <= margin_x)
        edge_touch_count += int(y1 <= margin_y) + int(frame_height - y2 <= margin_y)
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        result: Dict[str, Any] = {
            "bbox": [round(value, 1) for value in values],
            "center_x_ratio": round(center_x / float(frame_width), 4),
            "center_y_ratio": round(center_y / float(frame_height), 4),
            "area_ratio": round((box_width * box_height) / float(frame_width * frame_height), 5),
            "aspect_ratio": round(box_width / max(1.0, box_height), 4),
            "edge_touch_count": int(edge_touch_count),
        }
        if preferred_bbox is not None:
            result["preferred_iou"] = round(
                float(PersonTracker._bbox_iou_xyxy(values, preferred_bbox)),
                4,
            )
        return result

    def _log_search_candidate_frame_diagnostics(
        self,
        *,
        capture_frame_id: int,
        width: int,
        height: int,
        formal_evidence: Tuple[CandidateObservation, ...],
        probe_evidence: Tuple[CandidateObservation, ...],
        records: Any,
        preferred_bbox: Optional[Tuple[float, float, float, float]],
        current_candidate_decision: SearchCandidateGateDecision,
        gate_decision: SearchCandidateGateDecision,
        current_candidate_matches_target: bool,
        candidate_evidence_usable: bool,
        lateral_candidate_deferred: bool,
        stale_result_discarded: bool,
        search_settling_pending: bool,
    ) -> None:
        """Log one structured search frame to explain candidate gate outcomes."""
        if not SEARCH_REID_DIAGNOSTIC_ENABLE:
            return
        controller_search_state = str(
            getattr(self._follow_controller, "search_state", self.search_state)
            or self.search_state
        )
        if not (
            self.search_state == "searching"
            or controller_search_state == "searching"
            or search_settling_pending
        ):
            return
        # A tracker output keeps its expanded/display box in TrackRecord while
        # the raw YOLO box is retained by the identity observation metadata.
        # Join them by raw track id so edge-touch decisions are auditable.
        tracker = getattr(getattr(self, "_rknn_pipeline", None), "tracker", None)
        edge_margin_ratio = float(
            getattr(getattr(tracker, "config", None), "identity_edge_margin_ratio", 0.02)
        )
        observations_by_track: Dict[int, Dict[str, Any]] = {}
        for observation in getattr(tracker, "last_identity_observations", ()) or ():
            try:
                observations_by_track[int(observation.get("raw_track_id"))] = observation
            except (AttributeError, TypeError, ValueError):
                continue

        def detection_item(item: CandidateObservation, source: str) -> Dict[str, Any]:
            return {
                "source": str(source),
                "score": round(float(item.score), 4),
                **self._search_candidate_bbox_diagnostic(
                    item.bbox,
                    width=width,
                    height=height,
                    edge_margin_ratio=edge_margin_ratio,
                    preferred_bbox=preferred_bbox,
                ),
            }

        def json_value(value: Any) -> Any:
            """Normalize assignment scalars before serializing one-line JSON."""
            if value is None or isinstance(value, (bool, int, str)):
                return value
            try:
                number = float(value)
            except (TypeError, ValueError):
                return str(value)
            return round(number, 6) if math.isfinite(number) else None

        raw_candidates = [
            detection_item(item, "formal") for item in formal_evidence
        ] + [detection_item(item, "probe") for item in probe_evidence]
        raw_candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

        track_candidates = []
        assignment_reasons = []
        rejection_reasons = []
        for record in records or ():
            if int(getattr(record, "class_id", -1)) != PERSON_CLASS_ID:
                continue
            track_id = int(getattr(record, "track_id", -1))
            observation = observations_by_track.get(track_id, {})
            assignment = dict(
                observation.get("assignment")
                or self._identity_assignment_debug_for_track(track_id)
                or {}
            )
            sample_metadata = dict(observation.get("sample_metadata") or {})
            tracker_bbox = self._track_record_bbox(record)
            detector_bbox = observation.get("detector_bbox")
            track_item = {
                "track_id": track_id,
                "output_uid": int(getattr(record, "reid_uid", 0)),
                "score": round(float(getattr(record, "score", 0.0)), 4),
                "time_since_update": int(getattr(record, "time_since_update", 0)),
                "tracker": self._search_candidate_bbox_diagnostic(
                    tracker_bbox,
                    width=width,
                    height=height,
                    edge_margin_ratio=edge_margin_ratio,
                    preferred_bbox=preferred_bbox,
                ),
                "detector": self._search_candidate_bbox_diagnostic(
                    detector_bbox,
                    width=width,
                    height=height,
                    edge_margin_ratio=edge_margin_ratio,
                    preferred_bbox=preferred_bbox,
                )
                if detector_bbox is not None
                else None,
                "quality": {
                    key: json_value(sample_metadata.get(key))
                    for key in (
                        "quality_bbox_source", "quality_bbox", "quality_bbox_ok",
                        "quality_bbox_reason", "display_bbox_quality_ok",
                        "display_bbox_quality_reason", "detector_edge_touch_count",
                        "edge_touch_count", "partial_observation", "candidate_count",
                        "candidate_score_gap",
                    )
                    if key in sample_metadata
                },
                "assignment": {
                    key: json_value(assignment.get(key))
                    for key in (
                        "uid", "mapped_uid", "best_uid", "distance", "partial_distance",
                        "second_distance", "match_source", "reason", "pending_streak",
                        "handoff_streak", "late_candidate_streak", "bbox_quality_ok",
                        "preferred_search_identity_competition_override",
                        "bbox_quality_tier", "bbox_quality_reason", "reacquire_geometry_ok",
                        "reacquire_geometry_reason", "best_frame_gap",
                        "partial_aggregate_distance", "partial_aggregate_threshold",
                        "partial_aggregate_override", "late_candidate_rejection",
                    )
                    if key in assignment
                },
            }
            track_candidates.append(track_item)
            reason = assignment.get("reason")
            if reason:
                reason = str(reason)
                assignment_reasons.append(reason)
                if reason not in {
                    "mapped", "updated_diverse", "skip_update_redundant",
                    "skip_update_distance", "created", "created_confirmed",
                }:
                    rejection_reasons.append(reason)
            geometry_reason = assignment.get("reacquire_geometry_reason")
            if geometry_reason:
                rejection_reasons.append(str(geometry_reason))
            quality_reason = assignment.get("bbox_quality_reason")
            if quality_reason:
                rejection_reasons.extend(
                    item.strip() for item in str(quality_reason).split(",") if item.strip()
                )

        if stale_result_discarded:
            rejection_reasons.append("stale_result_discarded")
        if search_settling_pending:
            rejection_reasons.append("rotate_settling")
        if not raw_candidates:
            rejection_reasons.append("no_detector_candidate")
        if raw_candidates and not candidate_evidence_usable:
            rejection_reasons.append("candidate_below_control_score")
        if lateral_candidate_deferred:
            rejection_reasons.append("candidate_gate_incomplete")
        # Preserve ordering while avoiding repeated text from a compound
        # identity-bank quality reason.
        unique_reasons = list(dict.fromkeys(rejection_reasons))
        gate = {
            "source": str(gate_decision.source),
            "reason": str(gate_decision.reason),
            "pause_rotation": bool(gate_decision.pause_rotation),
            "entered": bool(gate_decision.entered),
            "completed": bool(gate_decision.completed),
            "probe_streak": int(gate_decision.probe_streak),
            "hold_frame": int(gate_decision.hold_frame),
            "hold_frames": int(gate_decision.hold_frames),
            "defer_sec": round(float(gate_decision.defer_sec), 4),
            "ignored_edge_probes": [detection_item(item, "probe") for item in
                                    getattr(self._search_candidate_gate, "last_ignored_edge_probes", ())],
            "ignored_weak_probes": [detection_item(item, "probe") for item in
                                    getattr(self._search_candidate_gate, "last_ignored_weak_probes", ())],
            "probe_observation_min_score": self._search_candidate_gate.config.probe_observation_min_score,
            "edge_probe_min_score": SEARCH_EVIDENCE_PROBE_EDGE_MIN_SCORE,
            "edge_probe_center_ratio": SEARCH_EVIDENCE_PROBE_EDGE_CENTER_RATIO,
            "internal_probe_streak": int(
                getattr(self._search_candidate_gate, "_probe_streak", 0)
            ),
            "internal_hold_remaining": int(
                getattr(self._search_candidate_gate, "_hold_remaining", 0)
            ),
        }
        current = {
            "source": str(current_candidate_decision.source),
            "reason": str(current_candidate_decision.reason),
            "score": round(float(current_candidate_decision.score), 4),
            "bbox": self._search_candidate_bbox_diagnostic(
                current_candidate_decision.bbox,
                width=width,
                height=height,
                edge_margin_ratio=edge_margin_ratio,
                preferred_bbox=preferred_bbox,
            )
            if current_candidate_decision.bbox is not None
            else None,
            "preferred_target_match": bool(
                current_candidate_decision.preferred_target_match
            ),
        }
        payload = {
            "control_frame_id": int(self.frame_index),
            "capture_frame_id": int(capture_frame_id),
            "search": {
                "state": controller_search_state,
                "direction": str(
                    getattr(
                        self._follow_controller,
                        "search_direction",
                        self.search_direction,
                    )
                    or self.search_direction
                    or "none"
                ),
                "active_uid": getattr(self._follow_controller, "active_target_id", None),
            },
            "raw_candidates": raw_candidates,
            "probe_cluster_diagnostics": [
                dict(item)
                for item in (
                    getattr(
                        getattr(self, "_rknn_pipeline", None),
                        "last_search_probe_cluster_diagnostics",
                        (),
                    )
                    or ()
                )
            ],
            "competition": {
                "candidate_count": len(raw_candidates),
                "top_score_gap": (
                    None
                    if len(raw_candidates) < 2
                    else round(
                        float(raw_candidates[0].get("score", 0.0))
                        - float(raw_candidates[1].get("score", 0.0)),
                        4,
                    )
                ),
            },
            "track_candidates": track_candidates,
            "current": current,
            "gate": gate,
            "outcome": {
                "matches_active_target": bool(current_candidate_matches_target),
                "evidence_usable": bool(candidate_evidence_usable),
                "deferred": bool(lateral_candidate_deferred),
                "assignment_reasons": list(dict.fromkeys(assignment_reasons)),
                "rejection_reasons": unique_reasons,
            },
        }
        try:
            encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning(
                "search_candidate_frame_diag_serialization_failed frame=%d error=%s",
                int(self.frame_index), type(exc).__name__,
            )
            return
        logger.info("search_candidate_frame_diag %s", encoded)

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
        current_reader = getattr(tracker, "control_assignment_for_track", None)
        current = current_reader(int(raw_track_id)) if callable(current_reader) else None
        if isinstance(current, dict):
            return dict(current)
        identity_bank = getattr(tracker, "identity_bank", None)
        last_assignments = getattr(identity_bank, "last_assignments", None)
        if not isinstance(last_assignments, dict):
            return {}
        assignment = last_assignments.get(int(raw_track_id))
        if isinstance(assignment, dict):
            return dict(assignment)
        return {}

    def _search_exclusion_bindings(self) -> Tuple[Dict[str, Any], ...]:
        """Join exclusions to *this capture's* detector boxes, never predictions.

        An exclusion concerns one raw track and one UID. A nearby unknown
        person or an unassociated probe must not inherit it through geometry.
        Keep non-excluded bindings too, so overlapping matches are ambiguous.
        """
        tracker = getattr(getattr(self, "_rknn_pipeline", None), "tracker", None)
        bank = getattr(tracker, "identity_bank", None)
        query = getattr(bank, "search_exclusion_for", None)
        try:
            uid = int(getattr(getattr(self, "_follow_controller", None), "active_target_id", 0) or 0)
            capture_id = int(getattr(self, "_active_capture_frame_id", -1))
            timestamp = float(getattr(self, "_active_capture_timestamp", 0.0))
        except (TypeError, ValueError):
            return ()
        if uid <= 0 or capture_id < 0 or not math.isfinite(timestamp) or timestamp <= 0:
            return ()
        bindings = []
        for observation in getattr(tracker, "last_identity_observations", ()) or ():
            try:
                metadata = observation.get("sample_metadata") or {}
                raw_track_id = int(observation["raw_track_id"])
                bbox = tuple(float(v) for v in observation["detector_bbox"])
                if (
                    metadata.get("is_fresh") is not True
                    or int(metadata.get("capture_frame_id", -1)) != capture_id
                    or float(metadata.get("capture_timestamp", -1.0)) != timestamp
                    or len(bbox) != 4 or not all(math.isfinite(v) for v in bbox)
                    or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]
                ):
                    continue
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            exclusion = None
            if callable(query):
                # The bank owns expiry. A None response must not be replaced
                # with a stale last_assignments exclusion from an older frame.
                exclusion = query(
                    raw_track_id, uid,
                    frame_index=int(getattr(tracker, "_frame_index", getattr(self, "frame_index", 0))),
                    capture_timestamp=timestamp,
                )
            else:
                assignment = PersonTracker._identity_assignment_debug_for_track(self, raw_track_id)
                if assignment.get("search_excluded") is True and assignment.get("excluded_uid") == uid:
                    exclusion = assignment.get("search_exclusion")
            if not isinstance(exclusion, dict) or exclusion.get("reason") != "co_visible_distinct_person":
                exclusion = None
            bindings.append({"raw_track_id": raw_track_id, "bbox": bbox, "exclusion": exclusion})
        counts = {}
        for item in bindings:
            counts[item["raw_track_id"]] = counts.get(item["raw_track_id"], 0) + 1
        for item in bindings:
            if counts[item["raw_track_id"]] > 1:
                item["exclusion"] = None
        return tuple(bindings)

    @staticmethod
    def _search_binding_for_bbox(bbox, bindings) -> Optional[Dict[str, Any]]:
        """Require a unique detector-box association; loose IoU is unsafe here."""
        try:
            bbox = tuple(float(v) for v in bbox)
            if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
                return None
        except (TypeError, ValueError):
            return None
        matches = sorted(
            ((PersonTracker._bbox_iou_xyxy(bbox, item["bbox"]), index, item)
             for index, item in enumerate(bindings)),
            key=lambda item: item[0], reverse=True,
        )
        if not matches or matches[0][0] < 0.85:
            return None
        if len(matches) > 1 and matches[0][0] - matches[1][0] < 0.10:
            return None
        return matches[0][2]

    def _search_excluded_tracks(self, bindings=None) -> Dict[int, Dict[str, Any]]:
        if bindings is None:
            bindings = PersonTracker._search_exclusion_bindings(self)
        excluded = {}
        for item in bindings:
            match = PersonTracker._search_binding_for_bbox(item["bbox"], bindings)
            if match is item and item["exclusion"] is not None:
                excluded[int(item["raw_track_id"])] = item["exclusion"]
        return excluded

    def _filter_search_excluded_evidence(self, candidates, bindings, *, source: str):
        kept = []
        for candidate in candidates:
            match = PersonTracker._search_binding_for_bbox(candidate.bbox, bindings)
            if match is None or match["exclusion"] is None:
                kept.append(candidate)
                continue
            logger.info(
                "search_candidate_excluded control_frame_id=%d capture_frame_id=%d "
                "raw_track_id=%d excluded_uid=%s source=%s bbox=%s reason=%s "
                "reference_track_id=%s source_capture_frame_id=%s effects=search_observation_only",
                int(self.frame_index), int(getattr(self, "_active_capture_frame_id", -1)),
                match["raw_track_id"], getattr(self._follow_controller, "active_target_id", None),
                source, candidate.bbox, match["exclusion"]["reason"],
                match["exclusion"].get("reference_track_id"),
                match["exclusion"].get("source_capture_frame_id"),
            )
        return tuple(kept)

    def _cancel_excluded_search_observation(self, bindings) -> bool:
        """Withdraw only the excluded candidate's hold; preserve search direction."""
        excluded = PersonTracker._search_excluded_tracks(self, bindings)
        if not excluded:
            return False
        gate = getattr(self, "_search_candidate_gate", None)
        held_track = getattr(self, "_search_evidence_observation_track_id", None)
        hold_match = PersonTracker._search_binding_for_bbox(getattr(gate, "_hold_bbox", None), bindings)
        cancel_hold = held_track in excluded or bool(hold_match and hold_match["exclusion"])
        if cancel_hold:
            if gate is not None:
                gate._hold_remaining = 0
                gate._hold_bbox = None
                gate._hold_source = "none"
                gate._hold_score = 0.0
                gate._hold_started_at = None
            PersonTracker._release_search_observation_for_control(self, "co_visible_distinct_person")
        # Do not reset the whole gate: another candidate may own its probe or
        # completed-observation state and must retain that independent budget.
        for field in ("_probe_bbox", "_blocked_bbox"):
            match = PersonTracker._search_binding_for_bbox(getattr(gate, field, None), bindings)
            if match and match["exclusion"]:
                setattr(gate, field, None)
                setattr(gate, "_probe_streak" if field == "_probe_bbox" else "_blocked_missing_frames", 0)
        if getattr(self, "_search_last_noted_candidate_track_id", None) in excluded:
            reset = getattr(self._follow_controller, "_reset_stale_direction_recovery", None)
            if callable(reset):
                # This clears observation/centering counters only, not the
                # controller's search_direction or _lost_exit_direction.
                reset("co_visible_distinct_person")
            self._search_last_noted_candidate_track_id = None
        return cancel_hold

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
        excluded_tracks = PersonTracker._search_excluded_tracks(self)
        matches = []
        for rec in records or ():
            if int(getattr(rec, "class_id", -1)) != PERSON_CLASS_ID:
                continue
            if int(getattr(rec, "time_since_update", 0)) != 0:
                continue
            if int(getattr(rec, "track_id", -1)) in excluded_tracks:
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
            rejected_mapping = bool(assignment.get("identity_control_rejected")) or assignment_reason in {
                "mapped_missing_identity",
                "mapped_verify_reject",
                "mapped_weak_distance_reject",
                "weak_preferred_reacquire_rejected",
                "identity_center_jump_reject",
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
        if assignment.get("identity_control_rejected"):
            return False
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
        if assignment.get("identity_control_rejected"):
            self._reset_search_geometry_reacquire()
            return None
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
        self._confirmed_search_reacquire_depth_last_frame = -1
        self._confirmed_search_reacquire_depth_streak = 0

    def _observe_search_settling_reid(
        self,
        records: Any,
        *,
        width: int,
        height: int,
    ) -> bool:
        """Accumulate strong identity evidence while post-rotation is quiet.

        ``process_external_frame`` intentionally withholds records from the
        normal controller while ActionRuntime is settling the chassis.  That
        used to discard the only fresh ReID frames in this interval, so the
        same person had to start the reacquisition streak again afterwards.
        This observer keeps a small, geometry-checked streak only; it never
        publishes an action or changes the frozen search direction.  Once the
        existing two-frame visual proof is complete, search is released and
        the following frame resumes normal control.
        """
        status_getter = getattr(self._follow_controller, "search_status", None)
        status = status_getter(time.monotonic()) if callable(status_getter) else None
        search_active = bool(
            getattr(status, "state", None) == "searching"
            or getattr(self._follow_controller, "search_state", self.search_state)
            == "searching"
        )
        active_uid = (
            getattr(status, "active_target_id", None)
            if status is not None
            else getattr(self._follow_controller, "active_target_id", None)
        )
        try:
            active_uid = None if active_uid is None else int(active_uid)
        except (TypeError, ValueError):
            active_uid = None
        if not search_active or active_uid is None or active_uid <= 0:
            self._reset_confirmed_search_reacquire()
            return False

        fresh_person_records = []
        candidates = []
        for rec in records or ():
            if int(getattr(rec, "class_id", -1)) != PERSON_CLASS_ID:
                continue
            if float(getattr(rec, "score", 0.0)) <= CONFIDENCE_THRESHOLD:
                continue
            if int(getattr(rec, "time_since_update", 0)) != 0:
                continue
            fresh_person_records.append(rec)
            assignment = self._identity_assignment_debug_for_track(
                int(getattr(rec, "track_id", -1))
            )
            best_uid = assignment.get("best_uid")
            try:
                best_uid = None if best_uid is None else int(best_uid)
            except (TypeError, ValueError):
                best_uid = None
            evidence = assignment.get("match_evidence")
            if not isinstance(evidence, dict):
                evidence = {}
            distance = assignment.get("distance")
            if distance is None:
                distance = evidence.get("distance")
            try:
                distance = None if distance is None else float(distance)
            except (TypeError, ValueError):
                distance = None
            match_source = str(
                assignment.get("match_source")
                or evidence.get("match_source")
                or ""
            ).strip().lower()
            if (
                best_uid != active_uid
                or match_source != "strong"
                or distance is None
                or not math.isfinite(distance)
                or distance > SEARCH_CANDIDATE_CONTINUE_ROTATE_MAX_REID_DISTANCE
                or assignment.get("bbox_quality_ok") is False
            ):
                continue
            bbox = self._track_record_bbox(rec)
            if bbox is None:
                continue
            candidates.append((rec, bbox, assignment, distance))

        if len(fresh_person_records) != 1 or len(candidates) != 1:
            # A gap or a competing person invalidates the local observation;
            # the normal search gate remains responsible for future evidence.
            previous_frame = int(self._confirmed_search_reacquire_last_frame)
            if previous_frame >= 0 and int(self.frame_index) - previous_frame > 1:
                self._reset_confirmed_search_reacquire()
            logger.info(
                "search_settling_reid_observation frame=%d active_uid=%d "
                "candidates=%d formal_persons=%d result=discard",
                int(self.frame_index),
                int(active_uid),
                len(candidates),
                len(fresh_person_records),
            )
            return False

        rec, bbox, assignment, distance = candidates[0]
        direction = str(
            getattr(status, "direction", None)
            or getattr(self._follow_controller, "search_direction", None)
            or self.search_direction
            or ""
        ).strip().lower()
        center_ratio = (float(bbox[0]) + float(bbox[2])) / max(1.0, float(width) * 2.0)
        direction_compatible = bool(
            direction not in {"left", "right"}
            or (direction == "left" and center_ratio <= 0.55)
            or (direction == "right" and center_ratio >= 0.45)
        )
        if not direction_compatible:
            self._reset_confirmed_search_reacquire()
            logger.info(
                "search_settling_reid_observation frame=%d active_uid=%d "
                "track_id=%d result=discard reason=search_direction_incompatible "
                "direction=%s center=%.3f",
                int(self.frame_index),
                int(active_uid),
                int(getattr(rec, "track_id", -1)),
                direction,
                float(center_ratio),
            )
            return False
        previous_bbox = self._confirmed_search_reacquire_bbox
        previous_frame = int(self._confirmed_search_reacquire_last_frame)
        frame_gap = int(self.frame_index) - previous_frame
        previous_area = (
            max(0.0, float(previous_bbox[2] - previous_bbox[0]))
            * max(0.0, float(previous_bbox[3] - previous_bbox[1]))
            if previous_bbox is not None
            else 0.0
        )
        current_area = max(0.0, float(bbox[2] - bbox[0])) * max(
            0.0, float(bbox[3] - bbox[1])
        )
        area_ratio = (
            min(previous_area, current_area) / max(previous_area, current_area)
            if previous_area > 0.0 and current_area > 0.0
            else 0.0
        )
        area_continuous = bool(
            previous_bbox is not None
            and area_ratio >= (1.0 - SINGLE_PERSON_GEOMETRY_MAX_AREA_CHANGE_RATIO)
        )
        center_jump_ratio = None
        if previous_bbox is not None and width > 0:
            previous_center = (previous_bbox[0] + previous_bbox[2]) * 0.5
            current_center = (bbox[0] + bbox[2]) * 0.5
            center_jump_ratio = abs(current_center - previous_center) / float(width)
        same_chain = bool(
            self._confirmed_search_reacquire_uid == active_uid
            and int(self._confirmed_search_reacquire_track_id or -1)
            == int(getattr(rec, "track_id", -1))
            and frame_gap == 1
            and area_continuous
            and (
                self._bbox_iou_xyxy(previous_bbox, bbox) >= SINGLE_PERSON_GEOMETRY_MIN_IOU
                or (center_jump_ratio is not None and center_jump_ratio <= SINGLE_PERSON_GEOMETRY_MAX_CENTER_JUMP_RATIO)
            )
        )
        streak = int(self._confirmed_search_reacquire_streak) + 1 if same_chain else 1
        self._confirmed_search_reacquire_uid = active_uid
        self._confirmed_search_reacquire_track_id = int(getattr(rec, "track_id", -1))
        self._confirmed_search_reacquire_bbox = bbox
        self._confirmed_search_reacquire_last_frame = int(self.frame_index)
        self._confirmed_search_reacquire_streak = streak
        # A very strong full-body match to the already locked UID is enough to
        # finish identity confirmation in the settling frame itself.  The
        # encoder/yaw settling gate still owns motor control, so this shortcut
        # cannot move the chassis or change search direction on its own.  Read
        # the threshold from the tracker config so the INI remains the single
        # source of truth; partial/weak matches retain the normal two-frame
        # requirement.
        tracker_cfg = getattr(
            getattr(getattr(self, "_rknn_pipeline", None), "tracker", None),
            "config",
            None,
        )
        instant_threshold = float(
            getattr(
                tracker_cfg,
                "identity_preferred_search_reacquire_instant_threshold",
                0.15,
            )
        )
        instant_reid = bool(
            instant_threshold > 0.0
            and match_source == "strong"
            and assignment.get("reacquire_geometry_ok") is True
            and assignment.get("bbox_quality_ok") is True
            and assignment.get("instant_reacquire_allowed") is True
            and distance <= min(
                instant_threshold,
                SEARCH_CANDIDATE_CONTINUE_ROTATE_MAX_REID_DISTANCE,
            )
        )
        # The instant path is still subordinate to the identity-bank
        # geometry result.  A low distance alone is not enough when the old
        # anchor rejected a large jump or a raw-track swap; those candidates
        # stay on the normal two-frame observation path.
        instant_geometry_ok = bool(assignment.get("reacquire_geometry_ok") is True)
        required = (
            1
            if instant_reid
            else max(2, int(SEARCH_CONFIRMED_REACQUIRE_FRAMES))
        )
        observation_result = "confirm" if streak >= required else "observe_only"
        logger.info(
            "search_settling_reid_observation frame=%d active_uid=%d track_id=%d "
            "distance=%.3f instant=%s instant_threshold=%.3f geometry_ok=%s confirmations=%d/%d same_chain=%s area_ratio=%.3f "
            "result=%s",
            int(self.frame_index),
            int(active_uid),
            int(getattr(rec, "track_id", -1)),
            float(distance),
            bool(instant_reid),
            float(instant_threshold),
            bool(instant_geometry_ok),
            int(streak),
            int(required),
            bool(same_chain),
            float(area_ratio),
            observation_result,
        )
        if streak < required:
            return True

        # Identity is proven, but the current frame is still under the
        # ActionRuntime settle gate. Release search state without publishing
        # lateral/longitudinal motion; the next fresh frame owns control.
        releaser = getattr(
            self._follow_controller,
            "release_search_on_confirmed_target",
            None,
        )
        if callable(releaser):
            releaser("settling_visual_reacquire")
        self.search_state = "none"
        self.search_direction = None
        self._reacquire_depth_pending = True
        self._reacquire_depth_pending_uid = active_uid
        hold_starter = getattr(self, "_start_visual_reacquire_hold", None)
        if callable(hold_starter):
            hold_starter(active_uid, bbox, reason="settling_visual_reacquire")
        logger.info(
            "search_settling_reid_confirmed frame=%d uid=%d track_id=%d "
            "confirmations=%d/%d action=none next_frame=normal_control",
            int(self.frame_index),
            int(active_uid),
            int(getattr(rec, "track_id", -1)),
            int(streak),
            int(required),
        )
        self._reset_confirmed_search_reacquire()
        return True

    def _clear_visual_reacquire_hold(self, reason: str) -> None:
        uid = getattr(self, "_visual_reacquire_hold_uid", None)
        if uid is not None:
            logger.info(
                "visual_reacquire_hold_end frame=%d uid=%s reason=%s",
                int(self.frame_index),
                int(uid),
                str(reason),
            )
        self._visual_reacquire_hold_uid = None
        self._visual_reacquire_hold_bbox = None
        self._visual_reacquire_hold_started_at = 0.0
        self._visual_reacquire_hold_until = 0.0

    def _start_visual_reacquire_hold(
        self,
        uid: int,
        bbox: Tuple[float, float, float, float],
        *,
        reason: str,
    ) -> None:
        if not VISUAL_REACQUIRE_HOLD_ENABLE or int(uid) <= 0:
            return
        now = time.monotonic()
        self._visual_reacquire_hold_uid = int(uid)
        self._visual_reacquire_hold_bbox = tuple(float(value) for value in bbox)
        self._visual_reacquire_hold_started_at = now
        self._visual_reacquire_hold_until = now + VISUAL_REACQUIRE_HOLD_SEC
        logger.info(
            "visual_reacquire_hold_start frame=%d uid=%d duration=%.0fms reason=%s bbox=%s",
            int(self.frame_index),
            int(uid),
            VISUAL_REACQUIRE_HOLD_SEC * 1000.0,
            str(reason),
            tuple(round(float(value), 1) for value in bbox),
        )

    def _visual_reacquire_hold_geometry_ok(
        self,
        bbox: Tuple[float, float, float, float],
        *,
        width: int,
        height: int,
    ) -> bool:
        uid = getattr(self, "_visual_reacquire_hold_uid", None)
        reference = getattr(self, "_visual_reacquire_hold_bbox", None)
        if (
            not VISUAL_REACQUIRE_HOLD_ENABLE
            or uid is None
            or reference is None
            or time.monotonic() > float(getattr(self, "_visual_reacquire_hold_until", 0.0))
            or int(width) <= 0
            or int(height) <= 0
        ):
            return False
        try:
            current = tuple(float(value) for value in bbox)
            reference = tuple(float(value) for value in reference)
        except (TypeError, ValueError):
            return False
        if current[2] <= current[0] or current[3] <= current[1]:
            return False
        current_center = (current[0] + current[2]) * 0.5 / float(width)
        reference_center = (reference[0] + reference[2]) * 0.5 / float(width)
        center_jump = abs(current_center - reference_center)
        current_area = max(0.0, current[2] - current[0]) * max(0.0, current[3] - current[1])
        reference_area = max(0.0, reference[2] - reference[0]) * max(0.0, reference[3] - reference[1])
        area_similarity = (
            min(current_area, reference_area) / max(current_area, reference_area)
            if current_area > 0.0 and reference_area > 0.0
            else 0.0
        )
        return bool(
            center_jump <= VISUAL_REACQUIRE_HOLD_MAX_CENTER_JUMP_RATIO
            and area_similarity >= VISUAL_REACQUIRE_HOLD_MIN_AREA_SIMILARITY
        )

    def _visual_reacquire_hold_match(
        self,
        rec: Any,
        assignment: Dict[str, Any],
        *,
        person_count: int,
        width: int,
        height: int,
    ) -> Optional[int]:
        """Return the held UID for one strong, geometrically continuous frame."""
        if assignment.get("identity_control_rejected"):
            return None
        active_uid = getattr(self._follow_controller, "active_target_id", None)
        held_uid = getattr(self, "_visual_reacquire_hold_uid", None)
        if (
            active_uid is None
            or held_uid is None
            or int(active_uid) != int(held_uid)
            or int(person_count) != 1
            or int(getattr(rec, "time_since_update", 0)) != 0
        ):
            return None
        bbox = self._track_record_bbox(rec)
        if bbox is None or not self._visual_reacquire_hold_geometry_ok(
            bbox, width=int(width), height=int(height)
        ):
            return None
        match_evidence = assignment.get("match_evidence")
        if not isinstance(match_evidence, dict):
            match_evidence = {}
        matched_uid = assignment.get("best_uid")
        if matched_uid is None:
            matched_uid = match_evidence.get("matched_uid")
        try:
            matched_uid = None if matched_uid is None else int(matched_uid)
        except (TypeError, ValueError):
            matched_uid = None
        if matched_uid != int(held_uid):
            return None
        distance = assignment.get("distance")
        if distance is None:
            distance = match_evidence.get("distance")
        try:
            distance = None if distance is None else float(distance)
        except (TypeError, ValueError):
            distance = None
        partial_distance = assignment.get("partial_distance")
        try:
            partial_distance = (
                None if partial_distance is None else float(partial_distance)
            )
        except (TypeError, ValueError):
            partial_distance = None
        match_source = str(
            assignment.get("match_source")
            or match_evidence.get("match_source")
            or ""
        ).strip().lower()
        appearance_ok = bool(
            match_source == "strong"
            and distance is not None
            and math.isfinite(distance)
            and distance <= SEARCH_CANDIDATE_CONTINUE_ROTATE_MAX_REID_DISTANCE
        ) or bool(
            partial_distance is not None
            and math.isfinite(partial_distance)
            and partial_distance <= 0.34
        )
        if not appearance_ok:
            return None
        self._visual_reacquire_hold_bbox = bbox
        self._visual_reacquire_hold_until = time.monotonic() + VISUAL_REACQUIRE_HOLD_SEC
        logger.info(
            "visual_reacquire_hold_candidate frame=%d uid=%d track_id=%d distance=%s partial_distance=%s reason=%s",
            int(self.frame_index),
            int(held_uid),
            int(getattr(rec, "track_id", -1)),
            "none" if distance is None else f"{distance:.3f}",
            "none" if partial_distance is None else f"{partial_distance:.3f}",
            str(assignment.get("reason") or "quality_rejected"),
        )
        return int(held_uid)

    def _observe_search_reacquire_depth(
        self,
        *,
        bbox: Tuple[float, float, float, float],
        track_id: int,
        uid: int,
        score: float,
        area: float,
        width: int,
        height: int,
    ) -> Tuple[bool, DistanceState]:
        """Refresh depth for a reacquisition candidate without publishing motion."""
        tracker = getattr(getattr(self, "_rknn_pipeline", None), "tracker", None)
        raw_observation = resolve_depth_target_observation(
            target_id=int(uid), display_bbox=bbox, expected_raw_track_id=int(track_id),
            capture_frame_id=int(getattr(self, "_active_capture_frame_id", 0)),
            capture_timestamp=float(getattr(self, "_active_capture_timestamp", 0.0)),
            observations=getattr(tracker, "last_identity_observations", ()) or (),
            width=int(width), height=int(height),
        )
        target = PersonTarget(
            bbox=tuple(float(value) for value in bbox),
            # Depth anchors follow the stable UID across raw-track handoffs.
            track_id=int(uid),
            confidence=float(score),
            area=float(area),
            depth_observation=raw_observation,
        )
        # This observation precedes _queue_actions_for_persons, so it must
        # reserve the same physical scan slot itself. Revoking an old Depth30
        # task does not mean that its pixel computation has already ended.
        scheduler = getattr(self, "_depth_async_scheduler", None)
        fallback_ticket = None
        pending = DistanceState(source="vision_depth", source_detail="depth_async_search_pending")
        if scheduler is not None:
            fallback_ticket = scheduler.begin_fallback(now=time.monotonic())
            if fallback_ticket is None:
                logger.info("search_reacquire_depth_deferred capture_frame_id=%s uid=%s "
                            "reason=scan_busy confirmation_preserved=True deadline_renewed=False",
                            getattr(self, "_active_capture_frame_id", 0), uid)
                return False, pending
        try:
            state = self._distance_runtime.get_frame_distance_state(
                int(width),
                target,
                frame_height=int(height),
                steering_feedback=self._action_runtime.get_steering_feedback(),
                target_distance_m=TARGET_DISTANCE,
                brake_distance_m=FOLLOW_BRAKE_DISTANCE_M,
                # This candidate belongs to the completed RGB capture, not to
                # whichever depth frame arrived while ReID was running. Identity
                # confirmation and the later Depth30 authority are unchanged.
                depth_use_latest=False,
                capture_timestamp=float(getattr(self, "_active_capture_timestamp", 0.0)),
            )
            observation_current = (fallback_ticket is None or scheduler.valid(fallback_ticket))
        finally:
            if fallback_ticket is not None:
                scheduler.finish(fallback_ticket, now=time.monotonic())
        if not observation_current:
            # A synchronous observation is not a motion grant. An intervening
            # identity/safety revocation also forbids it from advancing the
            # outer reacquisition counter or replacing the display cache.
            return False, pending
        self._last_frame_distance_state = state
        detail = str(getattr(state, "source_detail", "") or "")
        sample_age = getattr(state, "sample_age_sec", None)
        try:
            sample_age = None if sample_age is None else float(sample_age)
        except (TypeError, ValueError):
            sample_age = None
        fresh = bool(
            getattr(state, "used_distance_m", None) is not None
            and int(getattr(state, "sample_count", 0) or 0) > 0
            and detail.startswith("depth_")
            and "jump_pending" not in detail
            and sample_age is not None
            and sample_age <= SEARCH_REACQUIRE_DEPTH_MAX_AGE_SEC
        )
        if fresh:
            same_chain = bool(
                self._confirmed_search_reacquire_uid == int(uid)
                and self._confirmed_search_reacquire_depth_last_frame
                == int(self.frame_index) - 1
            )
            self._confirmed_search_reacquire_depth_streak = (
                int(self._confirmed_search_reacquire_depth_streak) + 1
                if same_chain
                else 1
            )
            self._confirmed_search_reacquire_depth_last_frame = int(self.frame_index)
        else:
            self._confirmed_search_reacquire_depth_streak = 0
            self._confirmed_search_reacquire_depth_last_frame = -1
        logger.info(
            "search_reacquire_depth_observation frame=%d uid=%d track_id=%d "
            "valid=%s streak=%d/%d distance=%s sample_age_ms=%s detail=%s",
            int(self.frame_index),
            int(uid),
            int(track_id),
            bool(fresh),
            int(self._confirmed_search_reacquire_depth_streak),
            int(SEARCH_REACQUIRE_DEPTH_CONFIRM_FRAMES),
            "none"
            if getattr(state, "used_distance_m", None) is None
            else f"{float(state.used_distance_m):.3f}",
            "none" if sample_age is None else f"{sample_age * 1000.0:.1f}",
            detail or "none",
        )
        return fresh, state

    def _remember_confirmed_identity_execution_anchor(
        self, candidate, person_count: int, width: int,
    ) -> None:
        """Record only a unique, current, formally identified control target."""
        rec = candidate["rec"]
        uid = getattr(self._follow_controller, "active_target_id", None)
        stamp = getattr(self, "_active_capture_timestamp", None)
        cap = getattr(self, "_active_capture_frame_id", None)
        now = time.monotonic()
        assignment = candidate["debug"]["assignment"]
        try:
            bbox = tuple(float(value) for value in candidate["bbox"])
            valid = (
                person_count == 1 and uid is not None and int(uid) > 0
                and int(width) == width and int(width) > 0
                and int(rec.reid_uid) == int(uid)
                and int(getattr(rec, "time_since_update", 0)) == 0
                and not candidate.get("geometry_fallback")
                and not assignment.get("identity_control_rejected")
                and assignment.get("uid") == int(uid)
                and assignment.get("bbox_quality_ok") is True
                and assignment.get("bbox_quality_tier") == "strong"
                and getattr(self, "search_state", "none") == "none"
                and getattr(self._follow_controller, "search_state", "none") == "none"
                and cap is not None and int(cap) == cap and int(cap) >= 0
                and stamp is not None and math.isfinite(float(stamp))
                and 0 <= now - float(stamp) <= VISION_CONTROL_MAX_RESULT_AGE_SEC
                and len(bbox) == 4 and all(math.isfinite(value) for value in bbox)
                and bbox[2] > bbox[0] and bbox[3] > bbox[1]
            )
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            self._confirmed_identity_execution_anchor = None
            return
        previous = getattr(self, "_confirmed_identity_execution_anchor", None)
        if (previous is not None
                and (int(cap) <= previous["capture_frame_id"]
                     or float(stamp) <= previous["capture_timestamp"])):
            # An old/reprocessed positive result cannot restart the window.
            self._confirmed_identity_execution_anchor = None
            return
        self._confirmed_identity_execution_anchor = {
            "uid": int(uid), "track_id": int(rec.track_id),
            "capture_frame_id": int(cap), "capture_timestamp": float(stamp),
            "bbox": bbox, "image_width": int(width),
        }

    def _identity_recheck_motion_evidence(self, uid, now):
        """Select existing motion evidence, never publish or renew it.

        A paired owner must not fall back to a retired legacy grant. Its
        immutable plan and the current identity proof must both remain live;
        an already narrowed recheck proof permits only its original sample.
        The legacy reader retains its existing full qualification below.
        """
        short = getattr(self, "_short_follow", None)
        if short is not None and short.config.enabled:
            state = short.snapshot()
            if state.active:
                plan = state.plan
                proof = getattr(self, "_validated_visual_observation", None)
                if (plan is None or state.uid != uid or plan.uid != uid
                        or plan.epoch != state.epoch or not plan.moving
                        or not plan.valid(now)
                        or not isinstance(proof, ValidatedVisualObservation)
                        or not proof.permits_depth(uid, plan.depth_timestamp, now)
                        or not motion_identity_live(self, uid, now)):
                    return None
                return ("paired", plan, plan.depth_timestamp, plan.expires_at)
        linear = getattr(self, "_depth30_linear_snapshot", None)
        if (not isinstance(linear, tuple) or len(linear) != 4
                or linear[0] != "forward" or linear[2] != uid):
            return None
        return ("legacy", linear, linear[3], None)

    def _existing_drive_grant_during_identity_recheck(
        self, rec, assignment, now: float, width: int,
    ) -> bool:
        """Permit only the old physical-time grant to finish a grey-frame gap."""
        anchor = getattr(self, "_confirmed_identity_execution_anchor", None)
        if not isinstance(anchor, dict):
            return False
        uid = anchor["uid"]
        if type(width) is not int or width <= 0 or anchor["image_width"] != width:
            return False
        competition = assignment.get("identity_competition")
        proof = assignment.get("identity_continuation")
        bank = getattr(getattr(getattr(self, "_rknn_pipeline", None), "tracker", None),
                       "identity_bank", None)
        binding = getattr(bank, "track_to_uid", None)
        conflicts = getattr(bank, "_mapped_geometry_conflicts", None)
        revoked = getattr(bank, "_geometry_revoked_uids", None)
        suspects = getattr(bank, "_reacquire_control_suspects", None)
        if (not all(isinstance(item, dict) for item in (binding, conflicts, revoked, suspects))
                or binding.get(anchor["track_id"]) != uid
                or anchor["track_id"] in conflicts or uid in revoked or uid in suspects):
            return False
        if (uid != getattr(self._follow_controller, "active_target_id", None)
                or anchor["track_id"] != int(rec.track_id)
                or getattr(self, "_vision_control_state", "") not in {
                    "target_visible", "target_visible_depth_valid", "target_visible_depth_missing"
                }
                or getattr(self, "search_state", "none") != "none"
                or getattr(self._follow_controller, "search_state", "none") != "none"
                or getattr(self, "_explicit_stop_requested", False)
                or getattr(self, "stop_action_execution", False)
                or getattr(self, "person_detected_flag", False)
                or getattr(self, "_runtime_shutdown_requested", False)
                or getattr(self, "_brake_hold_active", False)
                or getattr(self, "_near_yaw_park_request", None) is not None
                or getattr(self, "_use_soft_stop_next", False)
                or assignment.get("search_excluded") is True
                or assignment.get("candidate_geometry_conflict") is True
                or assignment.get("search_contradiction_retained") is True
                or not isinstance(competition, dict)
                or competition.get("passed") is not True
                or type(competition.get("uid")) is not int
                or competition.get("uid") != uid
                or type(competition.get("frame_index")) is not int
                or competition.get("frame_index") != int(self.frame_index)
                or type(competition.get("candidate_count")) is not int
                or competition.get("candidate_count") != 1
                or type(competition.get("source_detection_index")) is not int
                or competition["source_detection_index"] < 0
                or competition.get("observation_only") is True
                or (assignment.get("reason") == "verified_continuation_recheck"
                    and (not isinstance(proof, dict) or proof.get("status") != "hold"
                         or proof.get("source") not in {"strong", "partial"}))):
            return False
        cap = getattr(self, "_active_capture_frame_id", None)
        stamp = getattr(self, "_active_capture_timestamp", None)
        try:
            if (cap is None or int(cap) != cap or int(cap) <= anchor["capture_frame_id"]
                    or stamp is None or not math.isfinite(float(stamp))
                    or not anchor["capture_timestamp"] < float(stamp) <= now
                    or not 0 <= now-anchor["capture_timestamp"] <= .35):
                return False
            current = tuple(float(value) for value in (rec.x1, rec.y1, rec.x2, rec.y2))
            previous = anchor["bbox"]
            if (not all(math.isfinite(value) for value in current)
                    or current[2] <= current[0] or current[3] <= current[1]):
                return False
            old_area = (previous[2]-previous[0])*(previous[3]-previous[1])
            new_area = (current[2]-current[0])*(current[3]-current[1])
            area_similarity = min(old_area, new_area)/max(old_area, new_area)
            center_jump = abs(current[0]+current[2]-previous[0]-previous[2]) / (2.*width)
            if (area_similarity < .65
                    or (self._bbox_iou_xyxy(previous, current) < SINGLE_PERSON_GEOMETRY_MIN_IOU
                        and center_jump > min(.15, SINGLE_PERSON_GEOMETRY_MAX_CENTER_JUMP_RATIO))):
                return False
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            return False
        # Only the motion-evidence source differs. All identity, competition,
        # crop geometry and fixed-anchor checks above apply to both modes.
        evidence = PersonTracker._identity_recheck_motion_evidence(self, uid, now)
        if evidence is None:
            return False
        if evidence[0] == "paired":
            latest = PersonTracker._identity_recheck_motion_evidence(self, uid, now)
            return bool(latest is not None and latest[0] == "paired"
                        and latest[1] is evidence[1])
        # The legacy reader applies its existing depth/feedback/braking gate.
        # Neither mode publishes new evidence from an unconfirmed UID0 frame.
        linear = evidence[1]
        granted = PersonTracker._fresh_depth_linear_snapshot(self, uid, now=now)
        return bool(granted is not None and granted[0] == "forward"
                    and granted[1] > 0 and getattr(self, "_depth30_linear_snapshot", None) is linear)

    def _hold_for_identity_boundary_recheck(self, records, *, width: Optional[int] = None) -> bool:
        """Pause qualified borderline evidence within a fixed capture deadline.

        UID remains unconfirmed. No distance/steering anchor, search clock,
        gallery or motor deadline is renewed; existing safety has priority.
        """
        controller = getattr(self, "_follow_controller", None)
        uid = getattr(controller, "active_target_id", None)
        if (uid is None or len(records) != 1
                or getattr(controller, "search_state", "none") != "none"):
            return False
        rec = records[0]
        assignment = self._identity_assignment_debug_for_track(int(rec.track_id))
        deadline = assignment.get("identity_recheck_deadline")
        cap = getattr(self, "_active_capture_frame_id", None)
        stamp = getattr(self, "_active_capture_timestamp", None)
        now = time.monotonic()
        if (assignment.get("identity_recheck_pending") is not True
                or assignment.get("reason") not in ("partial_boundary_recheck", "verified_continuation_recheck")
                or assignment.get("mapped_uid") != uid
                or assignment.get("uid") != 0
                or int(getattr(rec, "reid_uid", 0)) != 0
                or assignment.get("identity_control_rejected") is not True
                or assignment.get("identity_recheck_capture") != cap
                or deadline is None or not math.isfinite(deadline)
                or stamp is None or not math.isfinite(stamp)
                or not stamp <= now <= deadline
                or int(getattr(rec, "time_since_update", 0)) != 0):
            return False
        if PersonTracker._existing_drive_grant_during_identity_recheck(
            self, rec, assignment, now, VISION_FRAME_WIDTH if width is None else width
        ):
            # Depth may revoke the old tuple on its own thread immediately
            # after the read. Diagnostics must not turn that normal revoke
            # into a vision-loop exception (or imply a renewed lease).
            evidence = PersonTracker._identity_recheck_motion_evidence(self, uid, now)
            existing_stamp = None if evidence is None else evidence[2]
            logger.info("identity_boundary_recheck capture=%s uid=%s deadline=%.6f "
                        "action=existing_grant_only identity_confirmed=False "
                        "authority_renewed=False sample_ts=%s search_started=False motion_source=%s",
                        cap, uid, deadline, existing_stamp,
                        None if evidence is None else evidence[0])
            return True
        self._confirmed_identity_execution_anchor = None
        self._publish_observation_soft_zero("identity_boundary_recheck", use_stop_action=True)
        logger.info("identity_boundary_recheck capture=%s uid=%s deadline=%.6f "
                    "action=soft_stop identity_confirmed=False authority_renewed=False "
                    "search_started=False", cap, uid, deadline)
        return True

    def _publish_observation_soft_zero(
        self,
        reason: str,
        *,
        use_stop_action: bool = False,
    ) -> None:
        """Publish a zero drive intent without entering motor brake hold.

        Detector evidence must not leave a ``forward=0`` action between a
        search pulse and the controller's STOP decision.  That intermediate
        action can race the pulse executor and clear its settling state.  The
        candidate gate therefore requests an explicit soft STOP, while the
        older confirmed-reacquire paths retain their publisher-zero behavior.
        """
        # Serialize against the Depth/lateral publishers. Revoking a visible
        # intent via _clear_lateral_intent would itself enqueue another action
        # (possibly preserving translation). Observation owns one all-zero
        # action, so withdraw the store directly before committing that zero.
        with getattr(self, "_control_update_lock", nullcontext()):
            PersonTracker._commit_observation_soft_zero(
                self, reason, use_stop_action=use_stop_action
            )

    def _commit_observation_soft_zero(self, reason: str, *, use_stop_action: bool) -> None:
        self._last_control_decision_reason = str(reason)
        self._last_command_source_module = "vision"
        self._last_command_control_frame = int(self.frame_index)
        self._last_decision_capture_frame = int(getattr(self, "_active_capture_frame_id", -1))
        self._last_command_capture_frame = self._last_decision_capture_frame
        self._last_command_capture_timestamp = float(getattr(self, "_active_capture_timestamp", 0.0))
        store = getattr(self, "_lateral_intent_store", None)
        if store is not None:
            store.clear()
        self._lateral_direction_intent = "hold"
        self._lateral_intent_owned_frame = -1
        self._lateral_intent_last_correction_rpm = 0
        self._clear_longitudinal_context(reason="observation:" + str(reason))
        self._current_forward_percent = 0
        self._current_steer_base_percent = 0
        self._current_steer_correction_rpm = 0
        self._current_steer_limit_reason = str(reason)
        self._current_rotate_raw_target = 0
        self._current_rotate_raw_source = str(reason)
        self._current_rotate_turn_percent = 0
        self._current_rotate_pulse_enabled = False
        # Invalidate a search TURN already computed by the motor thread too.
        self._lateral_yaw_revision = int(getattr(self, "_lateral_yaw_revision", 0)) + 1
        self._explicit_stop_requested = False
        self._use_soft_stop_next = bool(use_stop_action)
        self.is_forwarding = False
        if use_stop_action:
            # Candidate evidence needs a genuinely still frame.  Queueing a
            # soft STOP alone leaves a small race in which the action thread
            # can refresh the previous search rotation before it consumes the
            # STOP.  Cancel that in-flight rotation atomically, while keeping
            # the queued STOP for the normal controller diagnostics.
            cancel_rotation = getattr(
                self._action_runtime,
                "cancel_active_rotate_for_observation",
                None,
            )
            if callable(cancel_rotation):
                cancel_rotation(reason)
            # ACTION_STOP is dispatched as STOP_SOFT because the flag above is
            # set. It clears both wheel targets but does not create a brake
            # latch, and unlike ACTION_FORWARD it cannot be mistaken for a
            # translational command by the action queue.
            self._replace_action_queue([ACTION_STOP], reason)
        else:
            # Keep the legacy publisher-zero behavior for confirmed
            # reacquire/depth-wait paths, which deliberately avoid a brake
            # transition while preserving their existing control semantics.
            self._replace_action_queue([ACTION_FORWARD], reason)

    def _hold_search_reacquire_brake(self, *, bbox, width, eligible, confirmed=False,
                                    raw_track_id=None) -> bool:
        """Brake before search early returns. No UID, forward grant or template mutation."""
        runtime = getattr(self, "_action_runtime", None)
        pending = getattr(runtime, "search_reacquire_brake_pending", None)
        policy = getattr(self._follow_controller, "cfg", None)
        if not callable(pending) or policy is None:
            return False
        now = time.monotonic()
        stamp = float(getattr(self, "_active_capture_timestamp", 0.0))
        if not 0 < stamp <= now or now-stamp > VISION_CONTROL_MAX_RESULT_AGE_SEC:
            return bool(getattr(runtime, "_search_reacquire_brake_request", None))
        if pending():
            return True
        search_active = getattr(self._follow_controller, "search_state", self.search_state) == "searching"
        handoff = bool(confirmed and getattr(self, "_search_handoff_uid", None) is not None
                       and getattr(self._follow_controller, "active_target_id", None) == self._search_handoff_uid)
        if not search_active and not handoff:
            self._search_candidate_brake_episode = None
            return False
        direction = (getattr(self, "_search_handoff_direction", None) if handoff and not search_active
                     else getattr(self._follow_controller, "search_direction", self.search_direction))
        feedback = runtime.get_steering_feedback()
        now = time.monotonic()  # cache acquisition may cross the image/feedback deadline
        if not 0 <= now-stamp <= VISION_CONTROL_MAX_RESULT_AGE_SEC:
            if handoff:
                self._search_handoff_moving_evidence = None
            return False
        dispatch_delay = max(.05, min(.15, float(getattr(runtime, "_search_brake_dispatch_delay_sec", .05))))
        reason = search_brake_reason(
            bbox=bbox, width=width, direction=direction, eligible=eligible,
            now=now, capture_timestamp=stamp, max_age=VISION_CONTROL_MAX_RESULT_AGE_SEC,
            feedback=feedback, policy=policy, execution_delay_sec=dispatch_delay)
        moving = None
        if handoff and not search_active:
            evidence = observe_moving_handoff(
                self, eligible=eligible, uid=self._search_handoff_uid,
                raw_track_id=raw_track_id, cap=int(getattr(self, "_active_capture_frame_id", 0)),
                stamp=stamp, bbox=bbox, width=width, direction=direction,
                now=now, max_age=VISION_CONTROL_MAX_RESULT_AGE_SEC)
            fresh_linear = getattr(self, "_fresh_depth_linear_snapshot", None)
            linear = fresh_linear(self._search_handoff_uid, now=now) if callable(fresh_linear) else None
            if linear and linear[0] == "forward" and linear[1] > 0:
                moving = moving_handoff_yaw(self, evidence=evidence, uid=self._search_handoff_uid,
                    base=float(linear[1])*MOTOR_FORWARD_MAX_TARGET_RPM/100., yaw=0., feedback=feedback, now=now,
                    policy=policy, execution_delay_sec=dispatch_delay)
            retired = handoff_retirement_reason(
                self, eligible=eligible, uid=self._search_handoff_uid,
                raw_track_id=raw_track_id,
                cap=int(getattr(self, "_active_capture_frame_id", 0)),
                stamp=stamp, bbox=bbox, width=width, direction=direction)
            raw_yaw = getattr(feedback, "raw_yaw_rate_right_dps", None)
            feedback_current = bool(feedback is not None and feedback.trustworthy
                and raw_yaw is not None and math.isfinite(raw_yaw)
                and 0 <= now-feedback.timestamp <= min(.10, policy.visible_steering_pid_feedback_stale_sec))
            residual_pending = bool(
                (moving is not None and moving[1] == "residual_counter_differential")
                or (getattr(self, "_search_handoff_moving_active", False)
                    and (not feedback_current or (reason and abs(raw_yaw) > 2.))))
            if retired is not None and not residual_pending:
                logger.info("search_handoff_retired capture_frame_id=%s uid=%s reason=%s "
                            "normal_steering_required=True motion_authorized=False",
                            getattr(self, "_active_capture_frame_id", 0),
                            self._search_handoff_uid, retired)
                self._search_handoff_uid = None
                self._search_handoff_observation = None
                self._search_handoff_moving_evidence = None
                return False
        if moving is not None:
            # This merely selects the execution constraint. Only the writer's
            # current Depth/yaw leases authorize motion, rechecked at each tick.
            self._search_handoff_moving_active = True
            if reason is not None:
                logger.info("search_handoff_moving_convergence capture_frame_id=%s uid=%s "
                            "reason=%s phase=%s parking_requested=False deadline_renewed=False",
                            getattr(self, "_active_capture_frame_id", 0), self._search_handoff_uid,
                            reason, moving[1])
            return False
        if reason is None:
            self._search_candidate_brake_episode = None
            return False
        episode = getattr(self, "_search_candidate_brake_episode", None)
        if episode is not None:
            # A completed physical stop is not re-armed on every central box.
            # Unconfirmed evidence gets only the existing bounded observation
            # window; confirmed identity may immediately resume lateral control.
            if not confirmed and now-episode < SEARCH_EVIDENCE_MAX_HOLD_SEC:
                self._publish_observation_soft_zero("search_reacquire_brake_observe", use_stop_action=True)
                return True
            return False
        self._search_candidate_brake_episode = now
        # Install the execution veto BEFORE revoking axes. Revoking yaw can
        # otherwise publish a forward-only packet while STOP is being built.
        accepted = runtime.request_search_reacquire_brake(
            int(getattr(self, "_active_capture_frame_id", 0)), stamp, reason)
        if accepted is False:
            self._search_candidate_brake_episode = None
            return False
        self._search_handoff_uid = None  # one stop per acquisition, not normal-follow braking
        self._clear_lateral_intent("search_reacquire_brake")
        self._clear_longitudinal_context(reason="search_reacquire_brake")
        self._current_forward_percent = self._current_steer_base_percent = 0
        self._current_steer_correction_rpm = self._current_rotate_raw_target = 0
        self._use_soft_stop_next = False
        logger.info("search_reacquire_brake_gate capture_frame_id=%s reason=%s "
                    "confirmed=%s identity_claim=False longitudinal_allowed=False",
                    getattr(self, "_active_capture_frame_id", 0), reason, confirmed)
        return True

    def _publish_search_candidate_task_evidence(self, bbox=None, **evidence) -> bool:
        """Keep candidate task changes out of a physical search-brake hold.

        The normal brake-observation consumer still records current trusted
        positions. The release barrier, not this preliminary gate, chooses
        their effect on search direction after the chassis has settled.
        """
        runtime = getattr(self, "_action_runtime", None)
        if getattr(runtime, "_search_reacquire_brake_request", None) is not None:
            return False
        controller = getattr(self, "_follow_controller", None)
        method = getattr(controller, "note_search_candidate_missing" if bbox is None
                         else "note_search_candidate_evidence", None)
        if not callable(method):
            return False
        return bool(method() if bbox is None else method(bbox, **evidence))

    def _publish_search_reacquire_direction_hold(self, reason: str) -> bool:
        """Keep a bounded search yaw while a candidate waits for fresh Depth.

        A confirmed UID with an invalid depth sample is not safe for follow
        motion, but replacing the active search turn with ``forward=0`` leaves
        the chassis stationary and can make the reacquisition look stalled.
        The search direction is already frozen by the loss controller, so a
        low raw RPM turn is safe to refresh until the depth gate resolves.
        """
        reader = getattr(getattr(self, "_action_runtime", None), "get_steering_feedback", None)
        feedback = reader() if callable(reader) else None
        now = time.monotonic()
        status_getter = getattr(self._follow_controller, "search_status", None)
        status = status_getter(now) if callable(status_getter) else None
        direction = getattr(status, "direction", None) if status is not None else None
        if direction not in ("left", "right"):
            direction = getattr(self._follow_controller, "search_direction", None)
        if direction not in ("left", "right"):
            direction = getattr(self, "search_direction", None)
        budget_check = getattr(self._follow_controller, "search_wait_direction", None)
        if callable(budget_check):
            direction = budget_check(now, feedback)
        if direction not in ("left", "right"):
            logger.warning(
                "confirmed_search_reacquire_direction_hold unavailable: reason=%s",
                str(reason),
            )
            self._publish_observation_soft_zero(reason)
            return False

        action = ACTION_ROTATE_LEFT if direction == "left" else ACTION_ROTATE_RIGHT
        self._last_control_decision_reason = str(reason)
        self._clear_lateral_intent(reason)
        self._clear_longitudinal_context()
        self._current_forward_percent = 0
        self._current_steer_base_percent = 0
        self._current_steer_correction_rpm = 0
        self._current_steer_limit_reason = str(reason)
        self._current_rotate_raw_target = max(1, int(SEARCH_CANDIDATE_ACQUIRE_RAW_RPM))
        self._current_rotate_raw_source = "reacquire_depth_wait"
        # Refresh a continuous low-speed turn; the normal action stale timeout
        # still stops it if the next visual result does not arrive.
        self._current_rotate_pulse_enabled = False
        self._explicit_stop_requested = False
        self._use_soft_stop_next = False
        self.is_forwarding = False
        self._replace_action_queue([action], reason)
        logger.info(
            "confirmed_search_reacquire_direction_hold direction=%s raw=%d reason=%s",
            direction,
            int(self._current_rotate_raw_target),
            str(reason),
        )
        return True

    def _hold_for_confirmed_search_reacquire(
        self,
        selected_candidates: List[Dict[str, Any]],
        *,
        width: int,
        height: int = 480,
        depth_valid: bool = True,
    ) -> bool:
        """Require stable geometry and depth before ending a search session."""
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
            if not search_active and len(confirmed) == 1:
                current = confirmed[0]
                if getattr(self, "_search_handoff_uid", None) == int(current["stable_id"]):
                    if PersonTracker._hold_search_reacquire_brake(
                            self, bbox=current["bbox"], width=width,
                            eligible=int(getattr(current["rec"], "time_since_update", 0)) == 0,
                            confirmed=True, raw_track_id=getattr(current["rec"], "track_id", None)):
                        return True
            elif len(confirmed) != 1:
                self._search_handoff_uid = None
            self._reset_confirmed_search_reacquire()
            return False

        candidate = confirmed[0]
        uid = int(candidate["stable_id"])
        rec = candidate["rec"]
        track_id = int(rec.track_id)
        bbox = tuple(float(value) for value in candidate["bbox"])
        assignment = candidate.get("debug", {}).get("assignment", {})
        try:
            reid_distance = (
                None
                if assignment.get("distance") is None
                else float(assignment.get("distance"))
            )
        except (TypeError, ValueError):
            reid_distance = None
        assignment_reason = str(assignment.get("reason") or "").strip()
        assignment_uid = assignment.get("uid")
        try:
            assignment_uid = None if assignment_uid is None else int(assignment_uid)
        except (TypeError, ValueError):
            assignment_uid = None
        assignment_confirmed = bool(
            not assignment
            or (
                assignment_uid == uid
                and assignment_reason
                not in {
                    "controlled_handoff_wait",
                    "preferred_search_reacquire_wait",
                    "preferred_search_late_candidate_wait",
                    "preferred_search_reacquire_rejected",
                    "handoff_geometry_reject",
                }
            )
        )
        late_visual_confirmed = bool(
            assignment_confirmed
            and assignment_reason in {"preferred_search_late_reacquire", "preferred_search_soft_reacquire"}
        )
        identity_geometry_blocked = bool(
            assignment.get("reacquire_geometry_ok") is False
            or assignment.get("bbox_quality_ok") is False
            or int(getattr(rec, "time_since_update", 0)) != 0
        )
        if PersonTracker._hold_search_reacquire_brake(
                self, bbox=bbox, width=width,
                eligible=assignment_confirmed and not identity_geometry_blocked,
                confirmed=assignment_confirmed):
            return True
        # A small appearance distance alone is not identity proof. The bank
        # must also verify the locked UID's recent geometry; Depth still gates
        # longitudinal follow even when visual confirmation needs one frame.
        instant_reacquire = bool(
            active_uid is not None
            and int(uid) == int(active_uid)
            and reid_distance is not None
            and math.isfinite(reid_distance)
            and reid_distance <= 0.15
            and assignment.get("instant_reacquire_allowed") is True
            and assignment.get("reacquire_geometry_ok") is True
            and assignment.get("bbox_quality_ok") is True
            and not identity_geometry_blocked
        )
        previous_bbox = self._confirmed_search_reacquire_bbox
        frame_gap = int(self.frame_index) - int(self._confirmed_search_reacquire_last_frame)
        previous_area = (
            max(0.0, float(previous_bbox[2] - previous_bbox[0]))
            * max(0.0, float(previous_bbox[3] - previous_bbox[1]))
            if previous_bbox is not None
            else 0.0
        )
        current_area = max(0.0, float(bbox[2] - bbox[0])) * max(
            0.0, float(bbox[3] - bbox[1])
        )
        area_ratio = (
            min(previous_area, current_area) / max(previous_area, current_area)
            if previous_area > 0.0 and current_area > 0.0
            else 0.0
        )
        area_continuous = area_ratio >= (1.0 - SINGLE_PERSON_GEOMETRY_MAX_AREA_CHANGE_RATIO)
        same_chain = bool(
            self._confirmed_search_reacquire_uid == uid
            and previous_bbox is not None
            and frame_gap == 1
            and area_continuous
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
        # The identity bank's late-candidate path has already completed its
        # own two-frame visual confirmation. Do not make the motor wait for a
        # second, unrelated three-frame counter before releasing search.
        if late_visual_confirmed:
            streak = max(streak, 2)
        if identity_geometry_blocked:
            streak = 0
        self._confirmed_search_reacquire_uid = uid
        self._confirmed_search_reacquire_track_id = track_id
        self._confirmed_search_reacquire_bbox = bbox
        self._confirmed_search_reacquire_last_frame = int(self.frame_index)
        self._confirmed_search_reacquire_streak = streak
        required = (
            2
            if late_visual_confirmed
            else 1
            if instant_reacquire
            else max(2, int(SEARCH_CONFIRMED_REACQUIRE_FRAMES))
        )
        depth_gate_pending = False
        if (
            SEARCH_REACQUIRE_DEPTH_GATE_ENABLE
            and MODULE_ASTRA_DEPTH_ENABLE
            and getattr(self, "_distance_runtime", None) is not None
            and not identity_geometry_blocked
        ):
            depth_valid, _ = self._observe_search_reacquire_depth(
                bbox=bbox,
                track_id=track_id,
                uid=uid,
                score=float(candidate.get("score", 0.0)),
                area=float(candidate.get("area", 0.0)),
                width=int(width),
                height=int(height),
            )
            depth_gate_pending = bool(
                not depth_valid
                or int(self._confirmed_search_reacquire_depth_streak)
                < int(SEARCH_REACQUIRE_DEPTH_CONFIRM_FRAMES)
            )
        if (
            streak >= required
            and assignment_confirmed
            and not identity_geometry_blocked
        ):
            # Visual confirmation ends frozen search immediately. A missing
            # Depth sample only suppresses longitudinal context for this
            # frame; it must not keep applying the old search direction.
            releaser = getattr(
                self._follow_controller,
                "release_search_on_confirmed_target",
                None,
            )
            if callable(releaser):
                # Carry the search-to-follow braking obligation across early
                # identity release, until the first center approach is stopped.
                # It never supplies identity or motion authorization itself.
                if getattr(self, "_search_candidate_brake_episode", None) is None:
                    self._search_handoff_uid = uid
                    self._search_handoff_started_capture_ts = float(self._active_capture_timestamp)
                    self._search_handoff_observation = None
                    self._search_handoff_last_capture = None
                    self._search_handoff_moving_active = False
                    self._search_handoff_moving_evidence = None
                    # Capture the old search budget BEFORE visible PID can
                    # overwrite it with a saturated normal-follow correction.
                    old_search_rpm = abs(float(getattr(self, "_current_rotate_raw_target", 0)))
                    self._search_handoff_cap_rpm = min(7., old_search_rpm) if old_search_rpm > 0 else 7.
                    logger.info("search_handoff_yaw_armed capture_frame_id=%s uid=%s "
                                "search_rpm=%.1f cap_rpm=%.1f release=stop_or_fresh_follow_750ms "
                                "identity_changed=False deadline_renewed=False",
                                getattr(self, "_active_capture_frame_id", 0), uid,
                                old_search_rpm, self._search_handoff_cap_rpm)
                    self._search_handoff_direction = getattr(
                        self._follow_controller, "search_direction", self.search_direction)
                    observe_moving_handoff(self, eligible=True, uid=uid, raw_track_id=track_id,
                        cap=int(getattr(self, "_active_capture_frame_id", 0)),
                        stamp=float(self._active_capture_timestamp), bbox=bbox, width=width,
                        direction=self._search_handoff_direction, now=time.monotonic(),
                        max_age=VISION_CONTROL_MAX_RESULT_AGE_SEC)
                releaser(
                    "visual_reacquire_depth_pending"
                    if depth_gate_pending
                    else "visual_reacquire_confirmed"
                )
            self.search_state = "none"
            self.search_direction = None
            self._reacquire_depth_pending = bool(depth_gate_pending)
            self._reacquire_depth_pending_uid = uid if depth_gate_pending else None
            hold_starter = getattr(self, "_start_visual_reacquire_hold", None)
            if callable(hold_starter):
                hold_starter(
                    uid,
                    bbox,
                    reason=(
                        "visual_confirmed_depth_pending"
                        if depth_gate_pending
                        else "visual_confirmed"
                    ),
                )
            logger.info(
                "confirmed_search_reacquire frame=%d uid=%d track_id=%d "
                "confirmations=%d/%d depth_valid=%s geometry_ok=%s instant=%s "
                "depth_pending=%s result=resume_lateral",
                int(self.frame_index),
                uid,
                track_id,
                streak,
                required,
                bool(depth_valid),
                assignment.get("reacquire_geometry_ok"),
                instant_reacquire,
                bool(depth_gate_pending),
            )
            self._reset_confirmed_search_reacquire()
            return False

        reason = (
            "confirmed_search_reacquire_geometry_wait"
            if identity_geometry_blocked
            else
            "confirmed_search_reacquire_depth_wait"
            if depth_gate_pending
            else "confirmed_search_reacquire_wait"
        )
        self._follow_controller.set_search_observation_hold(True)
        if depth_gate_pending or identity_geometry_blocked:
            self._publish_search_reacquire_direction_hold(reason)
        else:
            self._publish_observation_soft_zero(reason)
        logger.info(
            "confirmed_search_reacquire frame=%d uid=%d track_id=%d "
            "confirmations=%d/%d depth_valid=%s area_ratio=%.3f geometry_ok=%s "
            "geometry_reason=%s instant=%s result=hold_search bbox=%s",
            int(self.frame_index),
            uid,
            track_id,
            streak,
            required,
            bool(depth_valid),
            area_ratio,
            assignment.get("reacquire_geometry_ok"),
            assignment.get("reacquire_geometry_reason", "missing_evidence"),
            instant_reacquire,
            bbox,
        )
        return True

    def _deferred_search_observation_bboxes(
        self, *, search_active, width, height, formal_candidates,
        capture_id, capture_timestamp, stale=False,
    ):
        """Current incomparable edge crops defer STOP only, never claim UID."""
        uid = getattr(self._follow_controller, "active_target_id", None)
        now = time.monotonic()
        if (not search_active or stale or uid is None
                or not math.isfinite(capture_timestamp)
                or not 0 <= now-capture_timestamp <= .19):
            return ()
        valid = self._search_candidate_gate._valid_candidates(
            formal_candidates, width=width, height=height,
            min_score=self._search_candidate_gate.config.formal_min_score)
        unique = []
        for item in valid:
            if not any(SearchCandidateGate._iou(item.bbox, old.bbox) >= .85 for old in unique):
                unique.append(item)
        if len(unique) != 1:
            return ()
        item = unique[0]
        tracker = getattr(getattr(self, "_rknn_pipeline", None), "tracker", None)
        matches = []
        for observation in getattr(tracker, "last_identity_observations", ()) or ():
            try:
                metadata = observation.get("sample_metadata") or {}
                bbox = tuple(float(v) for v in observation["detector_bbox"])
                if (metadata.get("is_fresh") is not True
                        or metadata.get("capture_frame_id") != capture_id
                        or metadata.get("capture_timestamp") != capture_timestamp
                        or len(bbox) != 4 or not all(math.isfinite(v) for v in bbox)
                        or SearchCandidateGate._iou(item.bbox, bbox) < .85):
                    continue
                assignment = self._identity_assignment_debug_for_track(int(observation["raw_track_id"]))
                matches.append(incomparable_edge_observation(
                    assignment, metadata, observation.get("uid"), uid, bbox, width))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        if matches != [True]:
            return ()
        logger.info("search_observation_deferred capture_frame_id=%s uid=%s bbox=%s "
                    "reason=incomparable_side_crop identity_claim=False "
                    "direction_change=False longitudinal_allowed=False", capture_id, uid, item.bbox)
        return (item.bbox,)

    def _retry_search_candidate_observation(
        self, decision, *, search_active, width, height, formal_candidates,
        capture_id, capture_timestamp, stale=False,
    ):
        """Shared observation budget; bank confirmation returns normal control."""
        retry = getattr(self, "_search_observation_retry", None)
        if retry is None:
            retry = self._search_observation_retry = SearchObservationRetry(SEARCH_EVIDENCE_MAX_HOLD_SEC)
        uid = getattr(self._follow_controller, "active_target_id", None)
        enabled = SEARCH_EVIDENCE_GATE_ENABLE and SEARCH_EVIDENCE_RETRY_ENABLE
        session = (int(getattr(self, "_search_epoch", 0)), uid) if enabled and search_active and uid else None
        now = time.monotonic()
        reader = getattr(getattr(self, "_action_runtime", None), "get_steering_feedback", None)
        feedback = reader() if callable(reader) else None
        settlement = getattr(self, "_search_detector_settlement", None)
        if settlement is None:
            settlement = self._search_detector_settlement = DetectorObservationSettlement()
        if not retry.active:
            if (search_active and not stale and decision.entered and decision.pause_rotation
                    and not settlement.active):
                retry.spend_detector_budget(session)
                self._search_retry_zero_requested_at = now
                self._search_retry_zero_sent_at = None
            decision = settlement.update(
                decision, now=now, capture_timestamp=capture_timestamp, capture_id=capture_id,
                search_active=search_active and SEARCH_EVIDENCE_GATE_ENABLE,
                zero_sent_at=getattr(self, "_search_retry_zero_sent_at", None),
                feedback=feedback, max_hold_sec=SEARCH_EVIDENCE_MAX_HOLD_SEC,
                stale=stale,
            )
        candidate = None
        confirmed_candidate = False
        evidence_reason = "no_unique_formal_candidate"
        # Use existing detector geometry limits. Overlapping probe/formal
        # duplicates cannot create a second person, but distinct people veto
        # this extra stop. The normal identity competition logic is unchanged.
        valid = self._search_candidate_gate._valid_candidates(
            formal_candidates, width=width, height=height,
            min_score=self._search_candidate_gate.config.formal_min_score,
        )
        unique = []
        for item in valid:
            if not any(SearchCandidateGate._iou(item.bbox, old.bbox) >= .85 for old in unique):
                unique.append(item)
        if (len(unique) == 1 and session is not None and not stale
                and decision.reason != DEFERRED_EDGE_COVERAGE):
            item = unique[0]
            tracker = getattr(getattr(self, "_rknn_pipeline", None), "tracker", None)
            matches = []
            for observation in getattr(tracker, "last_identity_observations", ()) or ():
                try:
                    metadata = observation.get("sample_metadata") or {}
                    bbox = tuple(float(v) for v in observation["detector_bbox"])
                    if (metadata.get("is_fresh") is not True
                            or int(metadata.get("capture_frame_id", -1)) != capture_id
                            or float(metadata.get("capture_timestamp", -1)) != capture_timestamp
                            or len(bbox) != 4 or not all(math.isfinite(v) for v in bbox)
                            or SearchCandidateGate._iou(item.bbox, bbox) < .85):
                        continue
                    assignment = self._identity_assignment_debug_for_track(int(observation["raw_track_id"]))
                    confirmed = confirmed_observation_for_release(
                        assignment, metadata, observation.get("uid"), uid)
                    source = retry_evidence_source(assignment, metadata, uid)
                    if source is None and not confirmed:
                        evidence_reason = "identity_evidence_insufficient"
                        continue
                    matches.append((item, source or "confirmed", confirmed))
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
            if len(matches) == 1:
                candidate, source, confirmed_candidate = matches[0]
                evidence_reason = "unique_" + source + "_observation"
        if (confirmed_candidate and (retry.active or settlement.active or decision.pause_rotation
                or getattr(self, "_search_evidence_observation_active", False))
                and not stale and 0 <= now-capture_timestamp <= .19
                and (retry.last_capture is None or (
                    capture_id > retry.last_capture[0] and capture_timestamp > retry.last_capture[1]))):
            # End only the observation pause. UID/depth/yaw and motor reversal
            # are rechecked by normal control; no old packet is restored.
            settlement.active = False
            result = retry.confirmed_release(session, candidate.bbox, candidate.score)
            retry.last_capture = (capture_id, capture_timestamp)
            logger.info("search_observation_confirmed_release capture_frame_id=%s uid=%s "
                        "normal_control_required=True motion_authorized=False", capture_id, uid)
            return result
        override = retry.update(
            now=now, session=session, capture_id=capture_id,
            capture_timestamp=capture_timestamp, eligible=candidate is not None and not stale,
            bbox=None if candidate is None else candidate.bbox,
            score=0.0 if candidate is None else candidate.score,
            blocked=decision.reason == "candidate_already_observed",
            zero_sent_at=getattr(self, "_search_retry_zero_sent_at", None),
            feedback=feedback,
        )
        if override is not None and override.entered:
            self._search_retry_zero_requested_at = retry.started_at
            self._search_retry_zero_sent_at = None
        if search_active and (override is not None or decision.reason == "candidate_already_observed"):
            logger.info(
                "search_observation_retry capture_frame_id=%d uid=%s reason=%s evidence=%s "
                "active=%s spent=%s capture_ts=%.6f zero_sent_at=%s deadline=%.6f "
                "quiet_samples=%d settled_at=%s "
                "identity_claim=False direction_change=False",
                capture_id, uid, retry.reason, evidence_reason, retry.active, retry.spent,
                capture_timestamp, getattr(self, "_search_retry_zero_sent_at", None), retry.deadline,
                retry.settled.quiet_count, retry.settled.ready_at,
            )
        return decision if override is None else override

    def _apply_search_candidate_gate_decision(
        self,
        decision: SearchCandidateGateDecision,
        *,
        prepare_only: bool = False,
    ) -> None:
        # Completion ends the observation budget; max_hold is only a timeout,
        # not an additional mandatory wait after the second image arrives.
        # Detector completion alone still does not grant identity or motion.
        if decision.completed:
            PersonTracker._release_search_observation_for_control(
                self, "candidate_observation_completed"
            )
            logger.info(
                "search_evidence_gate immediate_resume source=%s reason=%s",
                decision.source,
                decision.reason,
            )
            return
        if not decision.pause_rotation:
            return
        reason = "search_candidate_evidence_observe"
        if decision.defer_sec > 0.0:
            self._follow_controller.defer_search_timeout(decision.defer_sec)
        # Formal and probe detector evidence both get a soft STOP observation.
        # This is a zero-RPM confirmation pause, not an emergency brake; it
        # also leaves any encoder settling gate owned by ActionRuntime intact.
        if not prepare_only:
            self._publish_observation_soft_zero(reason, use_stop_action=True)
        if decision.entered:
            # Candidate evidence is a detector-layer hint, not a confirmed
            # identity. Give the stationary detector a short bounded window.
            # Do not cancel an encoder settle gate here: the next image must
            # be captured after the chassis is actually quiet.
            self._search_evidence_observation_active = True
            self._search_evidence_observation_source = str(decision.source)
            binding = PersonTracker._search_binding_for_bbox(
                decision.bbox, PersonTracker._search_exclusion_bindings(self)
            )
            self._search_evidence_observation_track_id = (
                None if binding is None else int(binding["raw_track_id"])
            )
            self._search_evidence_observation_deadline = (
                time.monotonic() + SEARCH_EVIDENCE_MAX_HOLD_SEC
            )
        logger.info(
            "search_evidence_gate frame=%d source=%s reason=%s score=%.3f bbox=%s "
            "observe=%d/%d entered=%s completed=%s pause_rotation=True "
            "action_policy=%s identity_claim=False reid_update=False "
            "pid_input=False defer=%.3fs max_hold=%.3fs",
            int(self.frame_index),
            decision.source,
            decision.reason,
            float(decision.score),
            decision.bbox,
            int(decision.hold_frame),
            int(decision.hold_frames),
            bool(decision.entered),
            bool(decision.completed),
            "pending_arbitration" if prepare_only else "soft_stop",
            float(decision.defer_sec),
            SEARCH_EVIDENCE_MAX_HOLD_SEC,
        )

    def _release_search_observation_for_control(self, reason: str) -> None:
        """Release only the detector observation hold, not identity/depth gates."""
        was_active = bool(
            getattr(self, "_search_evidence_observation_active", False)
            or getattr(self, "_search_evidence_pause_current_frame", False)
        )
        self._search_evidence_observation_active = False
        self._search_evidence_observation_source = "none"
        self._search_evidence_observation_deadline = 0.0
        self._search_evidence_observation_track_id = None
        self._search_evidence_pause_current_frame = False
        retry = getattr(self, "_search_observation_retry", None)
        if retry is not None:
            retry.release()
        self._search_retry_zero_requested_at = 0.0
        settlement = getattr(self, "_search_detector_settlement", None)
        if settlement is not None:
            settlement.active = False
        setter = getattr(self._follow_controller, "set_search_observation_hold", None)
        if callable(setter):
            setter(False)
        if was_active:
            logger.info(
                "search_observation_arbitration frame=%d capture_frame_id=%d "
                "reason=%s observation_released=True preliminary_stop=False",
                int(self.frame_index),
                int(getattr(self, "_active_capture_frame_id", 0)),
                reason,
            )

    def _multi_person_low_quality_lateral_track(self, records, width: int, height: int):
        """Select an existing UID for yaw only; never create a control identity."""
        gate = getattr(self, "_multi_person_lateral_gate", None)
        if gate is None:
            gate = self._multi_person_lateral_gate = MultiPersonLateralGate()
        if len(records) <= 1:
            gate.reset()
            return None
        tracker = getattr(getattr(self, "_rknn_pipeline", None), "tracker", None)
        result = gate.update(
            active_uid=getattr(self._follow_controller, "active_target_id", None),
            capture_frame_id=int(getattr(self, "_active_capture_frame_id", 0)),
            capture_timestamp=float(getattr(self, "_active_capture_timestamp", 0.0)),
            width=int(width),
            height=int(height),
            observations=getattr(tracker, "last_identity_observations", ()) or (),
        )
        logger.info(
            "low_quality_lateral_selection frame=%d capture_frame_id=%d track_id=%s "
            "reason=%s streak=%d distance=%s runner_up_distance=%s "
            "identity_claim=False longitudinal_allowed=False bank_update=False",
            int(self.frame_index), int(getattr(self, "_active_capture_frame_id", 0)),
            result.track_id, result.reason, result.streak,
            result.distance, result.runner_up_distance,
        )
        if result.track_id in PersonTracker._search_excluded_tracks(self):
            gate.reset()
            return None
        return result.track_id

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
        visual_hold_active = bool(
            getattr(self, "_visual_reacquire_hold_uid", None) == reid_uid
            and self._visual_reacquire_hold_geometry_ok(
                tuple(float(value) for value in bbox),
                width=int(width),
                height=int(height),
            )
        )
        if visual_hold_active:
            # The candidate has already passed the visual handoff gate. A
            # clipped follow-up frame may use bounded lateral yaw, but it must
            # never re-enter the frozen search-direction path.
            releaser = getattr(
                self._follow_controller,
                "release_search_on_confirmed_target",
                None,
            )
            if callable(releaser):
                releaser("visual_reacquire_hold_candidate")
            self.search_state = "none"
            self.search_direction = None
            self._follow_controller.set_search_observation_hold(False)
            logger.info(
                "visual_reacquire_hold_release_search frame=%d uid=%d track_id=%d",
                int(self.frame_index),
                reid_uid,
                track_id,
            )
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
            if (
                SEARCH_REACQUIRE_DEPTH_GATE_ENABLE
                and MODULE_ASTRA_DEPTH_ENABLE
                and getattr(self, "_distance_runtime", None) is not None
            ):
                self._observe_search_reacquire_depth(
                    bbox=tuple(float(value) for value in bbox),
                    track_id=track_id,
                    uid=reid_uid,
                    score=float(score),
                    area=float(area),
                    width=int(width),
                    height=int(height),
                )
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
                    # This path retains a mapping for bounded observation;
                    # a rejected crop is not a current formal UID decision.
                    candidate_confirmed_uid=0,
                    now=now_mono,
                )
            logger.info(
                "search_low_quality_evidence frame=%d track_id=%d reid_uid=%d "
                "center=%.3f search_direction_action=observation_only "
                "trusted_direction_update=False identity_claim=False "
                "longitudinal_allowed=False reason=quality_rejected",
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

    def _observe_target_during_search_brake(self, records, width: int, height: int) -> str:
        """Observation-only path: no PID, depth refresh, assignment or action queue."""
        controller = getattr(self, "_follow_controller", None)
        uid = getattr(controller, "active_target_id", None)
        cap = int(getattr(self, "_active_capture_frame_id", 0))
        stamp = float(getattr(self, "_active_capture_timestamp", 0.0))
        age = time.monotonic() - stamp
        self._search_brake_observation_age_sec = age
        if uid is None or int(uid) <= 0 or width <= 0 or height <= 0:
            return "no_active_target"
        if (not math.isfinite(age) or stamp <= 0
                or not 0 <= age <= SEARCH_BRAKE_DIRECTION_OBSERVATION_MAX_AGE_SEC):
            return "stale_capture"
        if cap <= int(getattr(controller, "_direction_latest_visible_capture_id", 0)):
            return "duplicate_or_old_capture"
        # Require a unique, fresh, formally assigned record. Detector-only and
        # geometry fallback candidates must not steer the loss-direction history.
        owned = [r for r in records if int(getattr(r, "reid_uid", 0)) == int(uid)
                 and int(getattr(r, "class_id", -1)) == PERSON_CLASS_ID
                 and int(getattr(r, "time_since_update", 0)) == 0
                 and float(getattr(r, "score", 0.0)) > CONFIDENCE_THRESHOLD]
        if len(owned) != 1:
            return "no_unique_confirmed_record"
        rec = owned[0]
        tracker = getattr(getattr(self, "_rknn_pipeline", None), "tracker", None)
        observations = [o for o in getattr(tracker, "last_identity_observations", ())
                        if o.get("raw_track_id") == int(rec.track_id)
                        and o.get("uid") == int(uid)
                        and (o.get("sample_metadata") or {}).get("capture_frame_id") == cap
                        and (o.get("sample_metadata") or {}).get("capture_timestamp") == stamp]
        if len(observations) != 1:
            return "missing_current_identity_provenance"
        observation = observations[0]
        metadata = observation.get("sample_metadata") or {}
        assignment = observation.get("assignment") or {}
        competition = metadata.get("identity_competition") or {}
        if (metadata.get("is_fresh") is not True
                or assignment.get("bbox_quality_ok") is not True
                or assignment.get("reacquire_geometry_ok") is False
                or assignment.get("search_excluded")
                or assignment.get("identity_control_rejected")
                or competition.get("passed") is False):
            return "identity_or_quality_rejected"
        if provisional_reacquire_direction(assignment):
            # A first, quarantined late binding may justify bounded stopped
            # observation, but cannot replace the trusted search direction.
            # Subsequent current identity verification can still publish it;
            # gallery quarantine itself does not require a prolonged stop.
            return "provisional_reacquire_observation"
        bindings = PersonTracker._search_exclusion_bindings(self)
        if any(b["raw_track_id"] == int(rec.track_id) and b["exclusion"] is not None
               for b in bindings):
            return "identity_excluded"
        bbox = observation.get("detector_bbox")
        if bbox is None or len(bbox) != 4:
            return "missing_detector_bbox"
        bbox = tuple(float(v) for v in bbox)
        if (not all(math.isfinite(v) for v in bbox)
                or not 0 <= bbox[0] < bbox[2] <= width
                or not 0 <= bbox[1] < bbox[3] <= height):
            return "invalid_detector_bbox"
        yaw = metadata.get("integrated_yaw_deg")
        feedback = (SteeringFeedback(timestamp=stamp, integrated_yaw_right_deg=float(yaw))
                    if yaw is not None and math.isfinite(float(yaw)) else None)
        target = PersonTarget(bbox=bbox, track_id=int(uid), confidence=float(rec.score),
                              area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        frame = SensorFrame(width=width, height=height, persons=[target],
                            capture_frame_id=cap, capture_timestamp=stamp,
                            steering_feedback=feedback)
        if not controller.note_brake_hold_observation(frame, target):
            return "history_not_updated"
        self._search_brake_latest_observation = (int(uid), cap)
        return "trusted_observation_recorded"

    def _record_search_brake_hold_observation(self, records, width: int, height: int) -> None:
        """Record verified capture history without releasing the actuator hold."""
        if not getattr(self, "_search_brake_observation_active", False):
            self._search_brake_observation_active = True
            self._search_brake_latest_observation = None
            self._search_brake_candidate_observation = SearchBrakeObservation()
            invalidate = getattr(getattr(self, "_follow_controller", None),
                                 "invalidate_pre_brake_steering", None)
            if callable(invalidate):
                invalidate()
        reason = PersonTracker._observe_target_during_search_brake(self, records, width, height)
        tracker = getattr(getattr(self, "_rknn_pipeline", None), "tracker", None)
        self._search_brake_center_candidate = self._search_brake_candidate_observation.centered_candidate(
            getattr(tracker, "last_identity_observations", ()),
            getattr(self._follow_controller, "active_target_id", None),
            int(getattr(self, "_active_capture_frame_id", 0)),
            float(getattr(self, "_active_capture_timestamp", 0.0)), width, height, time.monotonic())
        cap = getattr(self, "_active_capture_frame_id", 0)
        timing = getattr(self, "_brake_observation_pipeline_timing", {})
        if (timing.get("capture_frame_id") != cap
                or timing.get("capture_timestamp") != getattr(self, "_active_capture_timestamp", None)):
            timing = {}
        request = getattr(getattr(self, "_action_runtime", None), "_search_reacquire_brake_request", None)
        logger.info(
            "search_reacquire_brake_wait capture_frame_id=%s "
            "latest_observation_capture_frame_id=%s action_evidence_capture_frame_id=%s "
            "observation_reason=%s control=held longitudinal_allowed=False "
            "capture_age_ms=%.1f capture_to_pipeline_ms=%s vision_processing_ms=%s "
            "history_max_age_ms=%.1f motion_max_age_ms=%.1f motion_authorized=False",
            cap, getattr(getattr(self, "_follow_controller", None),
                         "_direction_latest_visible_capture_id", None),
            getattr(request, "capture_frame_id", None), reason,
            self._search_brake_observation_age_sec * 1000.,
            timing.get("capture_to_pipeline_ms"), timing.get("vision_processing_ms"),
            SEARCH_BRAKE_DIRECTION_OBSERVATION_MAX_AGE_SEC * 1000.,
            VISION_CONTROL_MAX_RESULT_AGE_SEC * 1000.,
        )

    def _finish_search_brake_observation_hold(self) -> None:
        runtime = getattr(self, "_action_runtime", None)
        lock_owner = getattr(runtime, "owner", self)
        with getattr(lock_owner, "motor_io_lock", nullcontext()):
            # A new brake request may have arrived after the caller checked
            # pending(). Never consume its context or retire its direction as
            # if it were the preceding completed observation episode.
            if getattr(runtime, "_search_reacquire_brake_request", None) is not None:
                return
            PersonTracker._finish_search_brake_observation_hold_locked(self)

    def _finish_search_brake_observation_hold_locked(self) -> None:
        if not getattr(self, "_search_brake_observation_active", False):
            return
        self._search_brake_observation_active = False
        evidence = getattr(self, "_search_brake_latest_observation", None)
        self._search_brake_latest_observation = None
        controller = getattr(self, "_follow_controller", None)
        runtime = getattr(self, "_action_runtime", None)
        sent_at = float(getattr(runtime, "_search_reacquire_brake_sent_at", 0.0))
        resume_context = getattr(runtime, "_search_reacquire_resume_context", None)
        resume_request = getattr(runtime, "_search_reacquire_resume_request", None)
        completed_request = getattr(runtime, "_search_reacquire_brake_applied", None)
        released_request = getattr(runtime, "_search_reacquire_resume_released_request", None)
        if runtime is not None:
            # One consumption per completed request; retries/new episodes do
            # not inherit the previous sweep's permission to resume searching.
            runtime._search_reacquire_resume_context = None
            runtime._search_reacquire_resume_request = None
            runtime._search_reacquire_resume_released_request = None
        resume_receipt_valid = not (resume_request is None or completed_request is not resume_request
                or released_request is not resume_request
                or getattr(runtime, "_search_reacquire_brake_request", None) is not None
                or any(getattr(self, name, False) for name in (
                    "_explicit_stop_requested", "_runtime_shutdown_requested", "_near_yaw_park_request"))
                or not getattr(self, "running", True)
                or str(getattr(self, "_brake_hold_label", "")).startswith("safety")
                or any(getattr(getattr(runtime, "backend", None), name, False)
                       for name in ("motion_write_fault", "parking_release_fault")))
        # Invalid proof of a saved pause is not the legacy "no saved task"
        # case. Falling back to unconditional retirement here would erase a
        # newer scan while bypassing the controller's episode-token check.
        context_current = getattr(controller, "search_brake_resume_context_current", None)
        stale_completion = resume_context is not None and (
            not resume_receipt_valid or not callable(context_current)
            or not context_current(resume_context))
        if stale_completion:
            evidence = None
        search_resumed = False
        if sent_at > 0 and not stale_completion:
            if resume_context is None:
                controller.retire_pre_search_brake_direction(sent_at)
            else:
                reader = getattr(runtime, "get_steering_feedback", None)
                search_resumed = controller.retire_pre_search_brake_direction(
                    sent_at, resume_context=resume_context,
                    steering_feedback=reader() if callable(reader) else None)
            visible = controller._target_direction_history.latest_visible_evidence()
            if visible is None or not evidence or visible.capture_frame_id != evidence[1]:
                evidence = None
            self.search_state, self.search_direction = controller.search_state, controller.search_direction
        reset = bool(evidence and controller.resume_direction_after_brake_hold(*evidence))
        if reset:
            self.search_state = controller.search_state
            self.search_direction = controller.search_direction
        logger.info("search_reacquire_brake_resume capture_frame_id=%s "
                    "latest_observation_capture_frame_id=%s direction_recomputed_required=%s "
                    "search_resumed=%s search_direction=%s stale_completion_ignored=%s "
                    "motion_authorized=False",
                    getattr(self, "_active_capture_frame_id", 0),
                    None if evidence is None else evidence[1], reset,
                    search_resumed, getattr(controller, "search_direction", None), stale_completion)

    def _consume_track_records(
        self,
        records,
        width: int,
        height: int,
        source_name: str,
        *,
        lateral_candidate: Optional[LateralCandidateEvidence] = None,
    ) -> None:
        runtime = getattr(self, "_action_runtime", None)
        pending = getattr(runtime, "search_reacquire_brake_pending", None)
        if callable(pending) and pending():
            # New vision may finish identity processing, but it cannot restore
            # forward/yaw while the search NORMAL stop is still settling.
            PersonTracker._record_search_brake_hold_observation(self, records, width, height)
            # Process THIS frame before releasing; CAP159 previously consumed
            # CAP156's direction because the current identity was still pending.
            settling = getattr(runtime, "_search_reacquire_settling", None)
            ready_at = getattr(settling, "ready_at", None)
            hold = bool(getattr(self, "_search_brake_center_candidate", False)
                        and ready_at is not None
                        and time.monotonic()-ready_at < SEARCH_EVIDENCE_MAX_HOLD_SEC)
            if settling is None or pending(
                    capture_timestamp=float(getattr(self, "_active_capture_timestamp", 0.0)),
                    keep_observing=hold):
                return
        PersonTracker._finish_search_brake_observation_hold(self)
        # CAP1786->1790: clearing the accepted speed here made a fresh
        # STEER become TURN + transition STOP on every older RGB observation.
        self._clear_longitudinal_context(
            revoke_translation=False, reason="visual_frame_begin",
        )
        # This is a one-frame longitudinal veto set by the reacquisition gate.
        # A new visual frame must earn the veto again; the UID grace window is
        # independently bounded by its wall-clock expiry.
        self._reacquire_depth_pending = False
        self._reacquire_depth_pending_uid = None
        if (
            getattr(self, "_visual_reacquire_hold_uid", None) is not None
            and time.monotonic()
            > float(getattr(self, "_visual_reacquire_hold_until", 0.0))
        ):
            self._clear_visual_reacquire_hold("expired")
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
        exclusion_bindings = PersonTracker._search_exclusion_bindings(self)
        excluded_tracks = PersonTracker._search_excluded_tracks(self, exclusion_bindings)
        if lateral_candidate is not None:
            lateral_binding = PersonTracker._search_binding_for_bbox(lateral_candidate.bbox, exclusion_bindings)
            if lateral_binding is not None and lateral_binding["exclusion"] is not None:
                lateral_candidate = None
        single_person_records = [
            rec
            for rec in records
            if int(getattr(rec, "class_id", -1)) == PERSON_CLASS_ID
            and float(getattr(rec, "score", 0.0)) > CONFIDENCE_THRESHOLD
            and int(getattr(rec, "time_since_update", 0)) == 0
            and int(getattr(rec, "track_id", -1)) not in excluded_tracks
        ]
        multi_person_lateral_track = PersonTracker._multi_person_low_quality_lateral_track(
            self, single_person_records, width, height
        ) if VISION_REID_ENABLE else None
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
            if int(rec.track_id) in excluded_tracks:
                # all_dets and original records still include this person for
                # hazard checks and video/debug; only target control excludes it.
                logger.info(
                    "search_excluded_control_skip frame=%d capture_frame_id=%d raw_track_id=%d "
                    "excluded_uid=%s reason=%s hazard_input_preserved=True",
                    int(self.frame_index), int(getattr(self, "_active_capture_frame_id", -1)),
                    int(rec.track_id), getattr(self._follow_controller, "active_target_id", None),
                    excluded_tracks[int(rec.track_id)]["reason"],
                )
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
            if VISION_REID_ENABLE and assignment.get("identity_control_rejected"):
                skipped_unconfirmed_reid += 1
                person_track_debug.append((area, stable_id, rec, None))
                logger.info("identity_control_rejected capture_frame_id=%s raw_track_id=%s "
                            "reason=%s geometry_fallback=False longitudinal_allowed=False",
                            getattr(self, "_active_capture_frame_id", None), rec.track_id,
                            assignment.get("reason"))
                continue
            if VISION_REID_ENABLE and not self._bound_reid_bbox_allowed_for_control(assignment):
                # 已绑定身份也不能绕过正常控制质量门槛。裁剪型大框可在
                # 后续独立路径受限转向；其他拒绝框只保留诊断记录。
                skipped_low_quality_reid += 1
                person_track_debug.append((area, stable_id, rec, None))
                mapped_uid = int(assignment.get("mapped_uid") or reid_uid or 0)
                fragment = PersonTracker._bbox_quality_is_fragment(assignment)
                active_target_id = getattr(self._follow_controller, "active_target_id", None)
                held_uid = None
                if not fragment:
                    hold_matcher = getattr(self, "_visual_reacquire_hold_match", None)
                    if callable(hold_matcher):
                        held_uid = hold_matcher(
                            rec,
                            assignment,
                            person_count=len(single_person_records),
                            width=int(width),
                            height=int(height),
                        )
                if held_uid is not None:
                    mapped_uid = int(held_uid)
                if (
                    (len(single_person_records) == 1
                     or int(rec.track_id) == multi_person_lateral_track)
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
                        "quality_reason": (
                            "visual_reacquire_hold"
                            if held_uid is not None
                            else assignment.get(
                                "bbox_quality_reason",
                                assignment.get("reason", "unknown"),
                            )
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
            self._confirmed_identity_execution_anchor = None
            return

        if (not selected_candidates
                and PersonTracker._hold_for_identity_boundary_recheck(
                    self, single_person_records, width=int(width))):
            return
        if not selected_candidates:
            self._confirmed_identity_execution_anchor = None
        linear = getattr(self, "_depth30_linear_snapshot", None)
        context = getattr(self, "_longitudinal_context", None)
        context_uid = (linear[2] if linear is not None else
                       context.get("target_id") if context is not None else None)
        if context_uid is not None:
            trusted_matches = [
                cand for cand in selected_candidates
                if int(cand["stable_id"]) == int(context_uid)
                and not bool(cand.get("geometry_fallback", False))
            ]
            if len(trusted_matches) != 1:
                self._clear_longitudinal_context(reason="visual_target_missing_or_ambiguous")

        active_uid = getattr(self._follow_controller, "active_target_id", None)
        retry = getattr(self, "_search_observation_retry", None)
        if (retry is not None and retry.active
                and time.monotonic() < retry.deadline
                and bool(getattr(self, "_search_evidence_pause_current_frame", False))
                and str(getattr(self._follow_controller, "search_state", "none"))
                in ("searching", "direction_unresolved")):
            # Only the explicit one-shot retry owns this pause. Ordinary
            # observed/confirmed targets retain their existing early handoff.
            self._publish_observation_soft_zero("search_candidate_retry_observe", use_stop_action=True)
            return
        normal_active_candidate = any(
            active_uid is not None and int(cand["stable_id"]) == int(active_uid)
            and not bool(cand.get("geometry_fallback", False))
            for cand in selected_candidates
        )
        if normal_active_candidate:
            PersonTracker._release_search_observation_for_control(self, "active_uid_control")
        if normal_active_candidate and len(selected_candidates) == 1:
            PersonTracker._remember_confirmed_identity_execution_anchor(
                self, selected_candidates[0], len(single_person_records), int(width)
            )
        else:
            self._confirmed_identity_execution_anchor = None
        if not normal_active_candidate and visible_unsteerable_candidate is not None:
            logger.info(
                "visible_active_low_quality_candidate frame=%d track_id=%d mapped_uid=%d "
                "area=%.0f quality=%s action=separate_hold_path",
                int(self.frame_index),
                int(visible_unsteerable_candidate["track_id"]),
                int(visible_unsteerable_candidate["reid_uid"]),
                float(visible_unsteerable_candidate["area"]),
                str(visible_unsteerable_candidate["quality_reason"] or "unknown"),
            )
            PersonTracker._release_search_observation_for_control(self, "mapped_low_quality_yaw")
            if self._hold_for_visible_unsteerable_target(
                width=int(width),
                height=int(height),
                **visible_unsteerable_candidate,
            ):
                # The hold path deliberately never publishes longitudinal
                # context. Returning here also prevents the empty-person path
                # from incrementing lost confirmation in the same frame.
                self._clear_longitudinal_context(reason="visible_low_quality_hold")
                return

        if getattr(self, "_visible_unsteerable_uid", None) is not None:
            self._finish_visible_unsteerable_hold(
                time.monotonic(),
                "complete_or_unmapped_frame",
            )

        if (
            bool(getattr(self, "_search_evidence_pause_current_frame", False))
            and str(getattr(self._follow_controller, "search_state", "none"))
            in ("searching", "direction_unresolved")
        ):
            # Final frame ownership: an unconfirmed detector candidate cannot
            # enqueue a rotate after its observation STOP in this same frame.
            self._publish_observation_soft_zero(
                "search_candidate_evidence_observe", use_stop_action=True
            )
            logger.info(
                "search_observation_arbitration frame=%d capture_frame_id=%d "
                "action_policy=soft_stop reason=await_candidate_evidence "
                "identity_claim=False longitudinal_allowed=False",
                int(self.frame_index), int(getattr(self, "_active_capture_frame_id", 0)),
            )
            return

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
        reacquire_depth_valid = True
        if (
            SEARCH_REACQUIRE_DEPTH_GATE_ENABLE
            and MODULE_ASTRA_DEPTH_ENABLE
            and getattr(self, "_distance_runtime", None) is not None
        ):
            latest_distance_state = getattr(self, "_last_frame_distance_state", None)
            reacquire_depth_valid = bool(
                latest_distance_state is not None
                and getattr(latest_distance_state, "used_distance_m", None) is not None
            )
        if self._hold_for_confirmed_search_reacquire(
            selected_candidates,
            width=int(width),
            height=int(height),
            depth_valid=reacquire_depth_valid,
        ):
            # A real UID is present, but one frame is not enough to terminate a
            # long-running search. Keep the chassis stopped until the same
            # visual chain and an accepted depth sample are both available.
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

        processed = self._queue_actions_for_persons(
            width,
            height,
            persons,
            low_quality_visible=False,
            lateral_candidate=lateral_candidate,
        )
        if processed is False:
            # A safety/identity/context transition may have invalidated the
            # visual computation while its private pixel scan was off-lock.
            # In particular, never republish that old ROI after STOP/reset.
            return
        if selected_candidates:
            if self._reacquire_depth_pending:
                # Visual identity is already confirmed, but this exact frame
                # did not produce a fresh accepted range. Keep lateral control
                # active while withholding the Depth30 bbox context so a stale
                # distance cannot restart forward motion.
                logger.info(
                    "visual_reacquire_depth_pending frame=%d uid=%s action=lateral_only",
                    int(self.frame_index),
                    "none"
                    if self._reacquire_depth_pending_uid is None
                    else int(self._reacquire_depth_pending_uid),
                )
                self._clear_longitudinal_context(reason="reacquire_depth_pending")
            else:
                already_published = False
                if getattr(self, "_longitudinal_context", None) is not None:
                    with self._longitudinal_context_lock:
                        context = self._longitudinal_context
                        already_published = bool(
                            context is not None
                            and context.get("frame_index") == self.frame_index
                            and context.get("capture_frame_id") == getattr(self, "_active_capture_frame_id", 0)
                            and context.get("target_id") == self._follow_controller.active_target_id
                        )
                if not already_published:
                    self._publish_longitudinal_context(width, height, persons)
        else:
            # Rejected boxes must never feed the independent Depth controller.
            self._clear_longitudinal_context(reason="visual_no_candidate")

    def _committed_visual_depth_replay(
        self, width, height, target, *, sample_max_age_sec=.18,
    ) -> Optional[DistanceState]:
        """Read a concurrent same-ROI commit without manufacturing freshness.

        Caller holds the control mutex. Exact target/geometry matching and the
        original physical clock are required; the copy cannot renew a grant,
        update the fusion anchor, or clear a committed safety indication.
        """
        distance = self._distance_runtime
        state = distance.last_distance_state
        stamp = getattr(state, "sample_timestamp", None)
        if (stamp is None and getattr(state, "temporal_status", None) == "duplicate"
                and getattr(state, "raw_distance_m", None) is None):
            # A subsequent Depth30 duplicate may already have downgraded its
            # latest state. Keep its original observation clock as-is.
            stamp = state.observation_timestamp
        if (distance._last_vision_depth_target != target
                or distance._last_vision_depth_frame_size != (int(width), int(height))
                or getattr(state, "source", None) != "vision_depth"
                or not isinstance(stamp, (int, float)) or isinstance(stamp, bool)
                or not math.isfinite(stamp) or stamp <= 0):
            return None
        age = time.monotonic() - stamp
        if not 0 <= age <= min(.18, ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC, sample_max_age_sec):
            return None
        return replace(
            state, raw_distance_m=None, sample_timestamp=None,
            observation_timestamp=stamp, temporal_status="duplicate",
            source_detail=str(state.source_detail) + "_concurrent_commit_hold",
            sample_age_sec=age,
        )

    def _async_visual_distance_display(self, target) -> DistanceState:
        """Same-UID held display data, not a new measurement or PI input."""
        runtime = self._distance_runtime
        state = runtime.last_distance_state
        previous = getattr(runtime, "_last_vision_depth_target", None)
        stamp = getattr(state, "sample_timestamp", None)
        if stamp is None:
            stamp = getattr(state, "observation_timestamp", None)
        if (target is None or previous is None or previous.track_id != target.track_id
                or not isinstance(stamp, (int, float)) or isinstance(stamp, bool)
                or not math.isfinite(stamp) or not 0 <= time.monotonic()-stamp
                    <= PersonTracker._depth_linear_max_age_sec("forward")):
            return DistanceState(source="vision_depth", source_detail="depth_async_pending")
        return replace(state, raw_distance_m=None, sample_timestamp=None,
                       observation_timestamp=stamp, temporal_status="deferred",
                       source_detail="depth_async_display_hold",
                       sample_age_sec=time.monotonic()-stamp)

    def _async_visual_lateral_eligible(self, width, height, targets, selected,
                                       capture_id, capture_ts, *,
                                       target_steerable, low_quality_visible) -> bool:
        ctl = self._follow_controller
        eligible = getattr(ctl, "can_decide_lateral_only", None)
        if (not callable(eligible) or not LATERAL_INTENT_CONTROL_ENABLE
                or not getattr(self, "_action_runtime_started", False)
                or getattr(self, "_brake_hold_active", False)
                or getattr(self, "_near_yaw_park_request", None) is not None
                or getattr(self, "_follow_distance_hold", None) is not None
                or getattr(ctl, "_normal_parking_uid", None) is not None
                or not self._visible_latest_depth_eligible(
                    targets, selected, capture_id, capture_ts,
                    control_source="vision", target_steerable=target_steerable,
                    low_quality_visible=low_quality_visible)):
            return False
        obstacles = self._get_obstacle_status()
        frame = SensorFrame(width=width, height=height, persons=list(targets),
            hazard=self._current_hazard_state_for_controller(),
            obstacles=ObstacleState(front=bool(obstacles.get("front", False)),
                                    left=bool(obstacles.get("left", False)),
                                    right=bool(obstacles.get("right", False))),
            capture_frame_id=capture_id, capture_timestamp=capture_ts)
        return eligible(frame, target_steerable=target_steerable,
                        low_quality_visible=low_quality_visible, rotation_only=FOLLOW_ROTATION_ONLY)

    @deferred_diagnostics(logger)
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
        depth_target_snapshot: Optional[Tuple[PersonTarget, ...]] = None,
        evidence_capture_frame_id: Optional[int] = None,
        evidence_capture_timestamp: Optional[float] = None,
        depth_work_ticket=None,
    ) -> bool:
        """False requests fast Depth retry / discard this visual result.

        For Depth30, True means this tick was consumed, not that a measurement
        or motion grant succeeded. Waiting for new sensor data is a normal
        completed tick; it must not take the 5 ms lock-contention retry path.

        Mutable live filtering/decision state stays under the same mutex.
        Normal visible tracking and Depth30 compute on private snapshots
        off-lock, then revalidate context identity and sensor revision before
        committing. Safety and newer Depth work never wait for pixel scans.
        """
        if getattr(self, "_runtime_shutdown_requested", False) or not getattr(self, "running", True):
            return control_source == "depth30"
        requested_at = time.monotonic()
        wait_started = time.perf_counter()
        acquired = self._control_update_lock.acquire(blocking=control_source != "depth30")
        if not acquired:
            pending = getattr(self, "_depth30_deferred_schedule", None)
            if pending is None or pending[0] != evidence_capture_frame_id:
                self._depth30_deferred_schedule = (evidence_capture_frame_id, requested_at, 1)
                logger.info(
                    "depth30_schedule_deferred capture_frame_id=%s reason=control_busy "
                    "retry_latest_context=True retry_max_ms=5 measurement_started=False "
                    "deadline_renewed=False",
                    evidence_capture_frame_id,
                )
            else:
                self._depth30_deferred_schedule = (pending[0], pending[1], pending[2] + 1)
            return False
        acquired_counter = time.perf_counter()
        acquired_cpu = time.thread_time()
        initial_wait_ms = (acquired_counter - wait_started) * 1000.0
        lock_held = True
        prepared_depth = None
        depth_state_replay = None
        visual_prepared = False
        range_deferred = visual_lateral_only = False
        scheduler = getattr(self, "_depth_async_scheduler", None)
        fallback_ticket = None
        prepare_lock_ms = prepare_cpu_ms = depth_compute_ms = commit_wait_ms = 0.0
        self._control_lock_stage_snapshot = None
        try:
            # A shutdown may have been latched while vision waited for Depth.
            if getattr(self, "_runtime_shutdown_requested", False) or not getattr(self, "running", True):
                return control_source == "depth30"
            if control_source == "vision" and scheduler is not None:
                targets = (tuple(depth_target_snapshot) if depth_target_snapshot is not None
                           else tuple(self._persons_to_targets(persons, width=width, height=height)))
                capture_id = (int(getattr(self, "_active_capture_frame_id", 0))
                              if evidence_capture_frame_id is None else int(evidence_capture_frame_id))
                capture_ts = (float(getattr(self, "_active_capture_timestamp", 0.))
                              if evidence_capture_timestamp is None else float(evidence_capture_timestamp))
                selected = self._distance_runtime.select_target(list(targets))
                eligible = self._async_visual_lateral_eligible(
                    width, height, targets, selected, capture_id, capture_ts,
                    target_steerable=target_steerable, low_quality_visible=low_quality_visible)
                worker = getattr(self, "_longitudinal_thread", None)
                health = scheduler.health(now=time.monotonic(),
                    worker_alive=worker is not None and worker.is_alive())
                if eligible and health.healthy:
                    visual_lateral_only = range_deferred = True
                else:
                    # Bootstrap/reacquisition/worker failure owns the same
                    # scan slot. A revoked running scan must finish before a
                    # fallback scan starts. Safety still runs without ranging.
                    fallback_ticket = scheduler.begin_fallback(now=time.monotonic())
                    if fallback_ticket is None:
                        range_deferred = True
                        visual_lateral_only = eligible
                if visual_lateral_only:
                    self._publish_longitudinal_context(width, height, persons, person_targets=targets)
                if range_deferred:
                    depth_target_snapshot = targets
                    evidence_capture_frame_id, evidence_capture_timestamp = capture_id, capture_ts
                logger.info("vision_depth_schedule capture_frame_id=%s uid=%s "
                    "mode=%s worker_health=%s scan_started=%s deadline_renewed=False",
                    capture_id, getattr(selected, "track_id", None),
                    "lateral_only" if visual_lateral_only else "fallback_wait" if range_deferred else "synchronous_fallback",
                    health.reason, not range_deferred)
            if control_source == "depth30":
                acquired_at = time.monotonic()
                capture_age = None if evidence_capture_timestamp is None else acquired_at - evidence_capture_timestamp
                deferred = getattr(self, "_depth30_deferred_schedule", None)
                self._depth30_deferred_schedule = None
                # A newer CAP supersedes the old pending ROI; its wait is not
                # the new frame's scheduling latency.
                superseded = deferred if deferred is not None and deferred[0] != evidence_capture_frame_id else None
                if superseded is not None:
                    deferred = None
                logger.info(
                    "depth30_schedule capture_frame_id=%s lock_wait_ms=%.1f "
                    "roi_age_ms=%s roi_remaining_ms=%s publication_age_ms=%s "
                    "deferred_capture_frame_id=%s deferred_count=%d deferred_elapsed_ms=%.1f "
                    "superseded_capture_frame_id=%s superseded_count=%d",
                    evidence_capture_frame_id, (acquired_at - requested_at) * 1000.0,
                    None if capture_age is None else round(capture_age * 1000.0, 1),
                    None if capture_age is None else round((ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC - capture_age) * 1000.0, 1),
                    None if context_published_ts is None else round((acquired_at - context_published_ts) * 1000.0, 1),
                    None if deferred is None else deferred[0],
                    0 if deferred is None else deferred[2],
                    0.0 if deferred is None else (acquired_at - deferred[1]) * 1000.0,
                    None if superseded is None else superseded[0],
                    0 if superseded is None else superseded[2],
                )
            if control_source == "depth30" and depth_target_snapshot is not None:
                # The loop may have copied this context before a visual
                # rejection cleared/replaced it. UID and age alone cannot
                # revoke that copy (lost_confirming still keeps the UID).
                with self._longitudinal_context_lock:
                    current_context = self._longitudinal_context
                    snapshot_current = bool(
                        current_context is not None
                        and current_context.get("person_targets") is depth_target_snapshot
                        and current_context.get("capture_frame_id") == evidence_capture_frame_id
                        and current_context.get("capture_timestamp") == evidence_capture_timestamp
                    )
                    if depth_work_ticket is not None and scheduler is not None:
                        snapshot_current = bool(scheduler.valid(depth_work_ticket)
                            and depth_work_ticket.context is not None
                            and current_context is not None
                            and current_context.get("target_id") == expected_target_id
                            and depth_work_ticket.context.person_targets is depth_target_snapshot
                            and depth_work_ticket.context.capture_frame_id == evidence_capture_frame_id
                            and depth_work_ticket.context.capture_timestamp == evidence_capture_timestamp)
                        current_context = depth_work_ticket.context.as_dict() if snapshot_current else None
                if not snapshot_current:
                    logger.info(
                        "depth30_context_revoked capture_frame_id=%s target=%s",
                        evidence_capture_frame_id, expected_target_id,
                    )
                    return True
            if expected_target_id is not None:
                # 30Hz线程等待控制锁期间，视觉线程可能已切换目标或进入搜索。
                # 必须在锁内重新确认，禁止旧人物框覆盖刚产生的新视觉动作。
                active_target_id = getattr(self._follow_controller, "active_target_id", None)
                context_age = (
                    0.0
                    if context_published_ts is None and evidence_capture_timestamp is None
                    else time.monotonic() - float(
                        evidence_capture_timestamp
                        if evidence_capture_timestamp is not None else context_published_ts
                    )
                )
                if (
                    not math.isfinite(context_age)
                    or context_age < 0.0
                    or context_age > ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC
                    or active_target_id is None
                    or int(active_target_id) != int(expected_target_id)
                    or self.search_state != "none"
                    or getattr(self._follow_controller, "search_state", "none") != "none"
                    or getattr(self, "_explicit_stop_requested", False)
                    or (getattr(self, "_brake_hold_active", False) and not is_follow_distance_hold(self))
                    or self._runtime_shutdown_requested
                    or not self.running
                ):
                    return control_source == "depth30"
            prepare = getattr(getattr(self, "_distance_runtime", None), "prepare_depth_measurement", None)
            if (control_source == "vision" and not range_deferred and persons and not low_quality_visible
                    and isinstance(getattr(self, "_distance_runtime", None), DistanceRuntime)
                    and (not getattr(self, "_brake_hold_active", False) or is_follow_distance_hold(self))):
                # Freeze detector/identity provenance before the inference
                # thread's next frame can alter its mutable observations.
                targets = (tuple(depth_target_snapshot) if depth_target_snapshot is not None
                           else tuple(self._persons_to_targets(persons, width=width, height=height)))
                capture_id = (int(getattr(self, "_active_capture_frame_id", 0))
                              if evidence_capture_frame_id is None else int(evidence_capture_frame_id))
                capture_ts = (float(getattr(self, "_active_capture_timestamp", 0.))
                              if evidence_capture_timestamp is None else float(evidence_capture_timestamp))
                selected = self._distance_runtime.select_target(list(targets))
                if self._visible_latest_depth_eligible(
                        targets, selected, capture_id, capture_ts,
                        control_source=control_source, target_steerable=target_steerable,
                        low_quality_visible=low_quality_visible):
                    if not callable(prepare):
                        logger.error("vision_prepare_unavailable capture_frame_id=%s "
                                     "locked_scan_fallback=False", capture_id)
                        return False
                    self._publish_longitudinal_context(
                        width, height, persons, person_targets=targets,
                    )
                    with self._longitudinal_context_lock:
                        current_context = self._longitudinal_context
                    if current_context is None:
                        return False
                    visual_prepared = True
                    depth_use_latest = True
                    depth_target_snapshot = targets
                    expected_target_id = selected.track_id
                    evidence_capture_frame_id, evidence_capture_timestamp = capture_id, capture_ts
            if (control_source == "depth30"
                    and isinstance(getattr(self, "_distance_runtime", None), DistanceRuntime)
                    and (not callable(prepare) or depth_target_snapshot is None or not depth_use_latest)):
                logger.error("depth30_prepare_unavailable capture_frame_id=%s "
                             "locked_scan_fallback=False", evidence_capture_frame_id)
                return True
            if ((control_source == "depth30" or visual_prepared) and depth_use_latest
                    and depth_target_snapshot is not None and callable(prepare)):
                target = self._distance_runtime.select_target(list(depth_target_snapshot))
                if target is None or target.track_id != expected_target_id:
                    return control_source == "depth30"
                prepared_depth = prepare(
                    width, height, target,
                    steering_feedback=self._action_runtime.get_steering_feedback(),
                )
                if prepared_depth is None:
                    # A busy sensor lock or an invalid ROI does not justify
                    # waiting under the control lock or clearing valid state.
                    return False
                context_token = current_context
                context_targets = current_context.get("person_targets")
                prepare_lock_ms = (time.perf_counter() - acquired_counter) * 1000.0
                prepare_cpu_ms = (time.thread_time() - acquired_cpu) * 1000.0
                self._control_update_lock.release()
                lock_held = False
                computation_started = time.perf_counter()
                prepared_depth.transaction.run()
                depth_compute_ms = (time.perf_counter() - computation_started) * 1000.0
                if prepared_depth.transaction.result is None:
                    transaction = prepared_depth.transaction
                    logger.info("%s_prepared_skip capture_frame_id=%s reason=%s "
                                "sample_ts=%s remaining_ms=%s "
                                "pixel_scan_started=%s failure_stage=%s compute_ms=%s compute_cpu_ms=%s "
                                "live_state_updated=False deadline_renewed=False",
                                control_source, evidence_capture_frame_id, transaction.reject_reason,
                                getattr(transaction, "selected_sample_timestamp", transaction.preflight_sample_timestamp),
                                None if transaction.preflight_remaining_sec is None else
                                transaction.preflight_remaining_sec * 1000.,
                                getattr(transaction, "pixel_scan_started", None),
                                getattr(transaction, "failure_stage", None),
                                getattr(transaction, "compute_wall_ms", None),
                                getattr(transaction, "compute_cpu_ms", None))
                    # The immutable history has been checked successfully;
                    # no eligible sample is data absence, not lock contention.
                    # Leave every live filter/authority untouched and let the
                    # normal period or a new-ROI wake start the next attempt.
                    # Vision retains its existing "result discarded" contract.
                    return control_source == "depth30"
                commit_wait_started = time.perf_counter()
                commit_acquired = self._control_update_lock.acquire(blocking=False)
                if not commit_acquired:
                    # Pixel work is already complete. A brief lateral/queue
                    # publication must not discard it and restart the scan.
                    # Wait holding no live lock; safety, Depth30 and the motor
                    # writer are never waiting on this pixel computation.
                    # This bounded wait grants nothing: all context/revision,
                    # identity, STOP and physical-age checks below still run.
                    commit_acquired = self._control_update_lock.acquire(timeout=.020)
                commit_wait_ms = (time.perf_counter() - commit_wait_started) * 1000.0
                if not commit_acquired:
                    logger.info("%s_prepared_discard capture_frame_id=%s reason=commit_control_busy "
                                "compute_ms=%.2f commit_wait_ms=%.2f retry_budget_ms=20 "
                                "sample_ts=%s live_state_updated=False",
                                control_source, evidence_capture_frame_id, depth_compute_ms, commit_wait_ms,
                                prepared_depth.transaction.result.observation_sample_timestamp)
                    return False
                lock_held = True
                acquired_counter, acquired_cpu = time.perf_counter(), time.thread_time()
                self._control_lock_stage_snapshot = None
                with self._longitudinal_context_lock:
                    current = self._longitudinal_context
                    same_context = bool(
                        current is context_token
                        and current is not None
                        and current.get("person_targets") is context_targets
                        and current.get("capture_frame_id") == evidence_capture_frame_id
                        and current.get("capture_timestamp") == evidence_capture_timestamp
                    )
                    if depth_work_ticket is not None and scheduler is not None:
                        same_context = bool(scheduler.valid(depth_work_ticket)
                            and current is not None and current.get("target_id") == expected_target_id
                            and depth_work_ticket.context.person_targets is context_targets)
                    if fallback_ticket is not None and scheduler is not None:
                        same_context = same_context and scheduler.valid(fallback_ticket)
                now = time.monotonic()
                age = now - float(evidence_capture_timestamp)
                still_valid = bool(
                    same_context and math.isfinite(age)
                    # ROI admission was checked when the fixed physical
                    # frame was selected. Completing that same sample is not
                    # a new ROI request; commit validates its selection proof
                    # and current physical age, including bounded history.
                    and 0 <= age
                    and getattr(self._follow_controller, "active_target_id", None) == expected_target_id
                    and self.search_state == "none"
                    and getattr(self._follow_controller, "search_state", "none") == "none"
                    and not getattr(self, "_explicit_stop_requested", False)
                    and not (getattr(self, "_brake_hold_active", False) and not is_follow_distance_hold(self))
                    and not self._runtime_shutdown_requested and self.running
                    and (not visual_prepared or (
                        int(getattr(self, "_active_capture_frame_id", 0)) == evidence_capture_frame_id
                        and float(getattr(self, "_active_capture_timestamp", 0.)) == evidence_capture_timestamp
                        and not getattr(self, "_reacquire_depth_pending", False)))
                )
                current_target = (self._distance_runtime.select_target(list(depth_target_snapshot))
                                  if still_valid else None)
                committed = bool(still_valid and self._distance_runtime.commit_prepared_depth(
                    prepared_depth, target=current_target,
                    steering_feedback=self._action_runtime.get_steering_feedback()))
                if not committed:
                    transaction = prepared_depth.transaction
                    # A same-context Depth30 winner may be newer than our
                    # speculative scan. Keep visual yaw/visibility processing
                    # via a read-only duplicate, never retry pixel work under
                    # this lock or overwrite the winning filter history.
                    if (visual_prepared and still_valid
                            and transaction.runtime._measurement_revision > transaction.revision
                            and not transaction.runtime._released
                            and not transaction.runtime._stop_event.is_set()):
                        depth_state_replay = self._committed_visual_depth_replay(
                            width, height, current_target,
                            sample_max_age_sec=transaction.runtime.config.max_frame_age_sec,
                        )
                    logger.info("%s_prepared_discard capture_frame_id=%s reason=context_or_sample_changed "
                                "same_context=%s compute_ms=%.2f live_state_updated=False "
                                "concurrent_commit_replay=%s deadline_renewed=False "
                                "transaction_reject=%s runtime_revision=%s selected_revision=%s "
                                "sample_ts=%s sample_age_ms=%s",
                                control_source, evidence_capture_frame_id, same_context, depth_compute_ms,
                                depth_state_replay is not None, transaction.reject_reason,
                                transaction.runtime._measurement_revision, transaction.revision,
                                getattr(transaction.result, "observation_sample_timestamp", None),
                                None if getattr(transaction.result, "observation_sample_timestamp", None) is None else
                                (time.monotonic()-transaction.result.observation_sample_timestamp)*1000.)
                    if depth_state_replay is None:
                        return not visual_prepared
            # RGB frame numbers cannot deduplicate independent Depth frames.
            # Process the current qualified ROI immediately; the sensor's
            # physical timestamp gate skips duplicate clustering, and replay
            # handling above preserves the original motor lease without PID.
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
                depth_target_snapshot=depth_target_snapshot,
                evidence_capture_frame_id=evidence_capture_frame_id,
                evidence_capture_timestamp=evidence_capture_timestamp,
                **({} if prepared_depth is None else {"prepared_depth": prepared_depth}),
                **({} if depth_state_replay is None else {"depth_state_replay": depth_state_replay}),
                **({"visual_lateral_only": visual_lateral_only, "range_deferred": True}
                   if range_deferred else {}),
            )
            return True
        finally:
            finished_counter = time.perf_counter()
            finished_cpu = time.thread_time()
            stage = self._control_lock_stage_snapshot if lock_held else None
            if lock_held:
                self._control_update_lock.release()
            if fallback_ticket is not None:
                scheduler.finish(fallback_ticket, now=time.monotonic())
            if prepared_depth is not None:
                prepared_depth.transaction.flush()
                transaction = prepared_depth.transaction
                # Stage formatting and logging occur after the control mutex
                # is released. Wall time is not CPU use: record both, including
                # scans discarded before commit, without a new sensor read.
                logger.info(
                    "depth_compute_profile capture_frame_id=%s source=%s sample_ts=%s "
                    "pixel_scan_started=%s stage=%s failure_stage=%s committed=%s "
                    "compute_ms=%s compute_cpu_ms=%s stages_ms=%s",
                    evidence_capture_frame_id, control_source,
                    getattr(transaction, "selected_sample_timestamp", None),
                    getattr(transaction, "pixel_scan_started", None),
                    getattr(transaction, "compute_stage", None),
                    getattr(transaction, "failure_stage", None),
                    getattr(transaction, "committed", None),
                    getattr(transaction, "compute_wall_ms", None),
                    getattr(transaction, "compute_cpu_ms", None),
                    {name: {kind: round(value, 3) for kind, value in values.items()}
                     for name, values in getattr(transaction, "stage_timings_ms", {}).items()},
                )
            # Format/write this aggregate after releasing the shared mutex.
            # The mutable ranging/fusion/controller commit remains atomic.
            wait_ms = initial_wait_ms
            hold_ms = (prepare_lock_ms +
                       ((finished_counter - acquired_counter) * 1000.0 if lock_held else 0.0))
            if stage is not None or max(wait_ms, hold_ms, depth_compute_ms) >= 10.0:
                if stage is None:
                    stage = (evidence_capture_frame_id, control_source, hold_ms, None, None, None, None)
                logger.info(
                    "control_lock_stage capture_frame_id=%s source=%s total_ms=%.2f "
                    "process_ms=%s ranging_ms=%s ranging_cpu_ms=%s queue_ms=%s "
                    "ranging_in_process=%s lock_wait_ms=%.2f lock_hold_ms=%.2f "
                    "lock_hold_cpu_ms=%.2f log_after_unlock=True prepare_lock_ms=%.2f "
                    "depth_compute_outside_lock_ms=%.2f optimistic_depth=%s "
                    "commit_wait_ms=%.2f commit_retry_budget_ms=20 "
                    "range_deferred=%s visual_lateral_only=%s",
                    *stage, prepared_depth is None and not range_deferred, wait_ms, hold_ms,
                    prepare_cpu_ms + ((finished_cpu - acquired_cpu) * 1000.0 if lock_held else 0.0),
                    prepare_lock_ms, depth_compute_ms, prepared_depth is not None, commit_wait_ms,
                    range_deferred, visual_lateral_only,
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
        depth_target_snapshot: Optional[Tuple[PersonTarget, ...]] = None,
        evidence_capture_frame_id: Optional[int] = None,
        evidence_capture_timestamp: Optional[float] = None,
        prepared_depth=None,
        depth_state_replay: Optional[DistanceState] = None,
        visual_lateral_only: bool = False,
        range_deferred: bool = False,
    ) -> None:
        started = time.perf_counter()
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
            depth_target_snapshot=depth_target_snapshot,
            evidence_capture_frame_id=evidence_capture_frame_id,
            evidence_capture_timestamp=evidence_capture_timestamp,
            **({} if prepared_depth is None else {"prepared_depth": prepared_depth}),
            **({} if depth_state_replay is None else {"depth_state_replay": depth_state_replay}),
            **({"visual_lateral_only": visual_lateral_only, "range_deferred": True}
               if range_deferred else {}),
        )
        decision_finished = time.perf_counter()
        if getattr(self, "_short_follow_handled_frame", False):
            # [] here means "paired owner handled", not "no follow action".
            # In particular search-to-follow must not append a legacy STOP.
            return
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
        finished = time.perf_counter()
        if finished - started >= .01:
            # Report only material control-lock holds; routine fast ticks do
            # not generate another high-rate line. Ranging CPU/wall separates
            # actual processing from scheduling/measurement-lock contention.
            cap = (getattr(self, "_active_capture_frame_id", 0)
                   if evidence_capture_frame_id is None else evidence_capture_frame_id)
            timing = getattr(self, "_last_control_ranging_timing", None)
            timing = timing if timing is not None and timing[:2] == (control_source, cap) else None
            self._control_lock_stage_snapshot = (
                cap, control_source, (finished - started) * 1000.0,
                round((decision_finished - started) * 1000.0, 2),
                None if timing is None else round(timing[2], 2),
                None if timing is None else round(timing[3], 2),
                round((finished - decision_finished) * 1000.0, 2),
            )

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
        if getattr(self, "_runtime_shutdown_requested", False):
            return
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        if self._capture_thread is not None or getattr(self, "_capture_failure_reason", None):
            # An unexpectedly dead reader is a fault, not permission to open
            # another concurrent camera reader or silently restart hardware.
            self._send_runtime_shutdown_stop("runtime_fault_capture_thread_dead")
            return
        self._capture_stop_event.clear()
        self._capture_started_ts = time.monotonic()
        self._capture_last_frame_ts = 0.0
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture-30fps",
            daemon=True,
        )
        try:
            self._capture_thread.start()
        except Exception as exc:
            self._capture_failure_reason = f"thread_start:{type(exc).__name__}:{exc}"
            self._send_runtime_shutdown_stop("runtime_fault_capture")
            logger.exception("Camera capture startup failed; shutdown latched, automatic_restart=False")
            return
        logger.info(
            "Camera capture queue started: fps=%.1f queue=%d",
            RKNN_CAMERA_FPS,
            RKNN_CAPTURE_QUEUE_SIZE,
        )

    def _capture_loop(self) -> None:
        try:
            self._capture_frames()
        except Exception as exc:
            self._capture_failure_reason = f"{type(exc).__name__}:{exc}"
            self._send_runtime_shutdown_stop("runtime_fault_capture")
            logger.exception("Camera capture worker failed; shutdown latched, automatic_restart=False")

    def _capture_frames(self) -> None:
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
            self._capture_last_frame_ts = timestamp
            with self._capture_state_lock:
                self._capture_frame_id += 1
                capture_id = int(self._capture_frame_id)
            shape = getattr(frame, "shape", ())
            frame_height = int(shape[0]) if len(shape) >= 1 else 0
            frame_width = int(shape[1]) if len(shape) >= 2 else 0
            self._capture_metadata_ring.append(
                (capture_id, float(timestamp), frame_width, frame_height)
            )
            # The control RGB is published before optional copies, encoding
            # metadata or direction-worker submission. Those consumers must
            # never make a successfully acquired control image disappear.
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
            self._submit_capture_sidecars(capture_id, timestamp, frame)

    def _submit_capture_sidecars(self, capture_id, timestamp, frame) -> None:
        self._record_depth_diagnostic_rgb(capture_id, timestamp, frame)
        recorder = self._camera_video_recorder
        if recorder is not None and not getattr(self, "_capture_recorder_disabled", False):
            try:
                recorder.submit(
                    frame, capture_frame_id=capture_id, monotonic_sec=timestamp,
                    unix_sec=time.time(), wheel_feedback=self._action_runtime.get_recording_feedback(),
                    follow_snapshot=getattr(self, "_video_follow_snapshot", None),
                    linear_timing=recording_authority(self),
                )
            except Exception:
                self._capture_recorder_disabled = True
                # Keep the reference for normal shutdown/partial-file cleanup.
                logger.exception("Camera sidecar disabled: recorder; control_capture_continues=True")
        pool = self._direction_pool
        if pool is not None and not getattr(self, "_capture_direction_disabled", False):
            try:
                pool.submit(capture_id, timestamp, frame, "BGR")
            except Exception:
                self._capture_direction_disabled = True
                # Do not set _direction_pool=None: that selects synchronous
                # camera.read in the main loop, creating a second reader.
                logger.exception("Camera sidecar disabled: direction; control_capture_continues=True")

    def _capture_health_ok(self) -> bool:
        """Fail closed on a dead/stalled reader; no automatic camera reopen.

        One second is a liveness alarm, NOT a motion-evidence TTL. Existing
        Separate configured image/depth deadlines still withdraw old motion
        authorization; this alarm never extends either evidence lifetime.
        """
        if getattr(self, "_runtime_shutdown_requested", False) or not self.running:
            return False
        thread = self._capture_thread
        if thread is None:
            return True  # Synchronous camera mode has no worker to supervise.
        reason = None
        if not thread.is_alive():
            reason = "runtime_fault_capture_thread_dead"
        else:
            last = (getattr(self, "_capture_last_frame_ts", 0.0)
                    or getattr(self, "_capture_started_ts", 0.0))
            if last > 0.0 and time.monotonic() - last > 1.0:
                reason = "runtime_fault_capture_stalled"
        if reason is None:
            return True
        self._capture_failure_reason = self._capture_failure_reason or reason
        self._send_runtime_shutdown_stop(reason)
        logger.error("Camera capture unhealthy: reason=%s automatic_restart=False", reason)
        return False

    def _recording_depth_sample(self, capture_id, timestamp):
        # Invoked only by the video writer. Reads bounded diagnostics, not
        # sensors, controller state, measurement locks or a serial port.
        depth = getattr(getattr(self, "_sensor_runtime", None), "astra_depth", None)
        diagnostics = getattr(depth, "diagnostics", None)
        return None if diagnostics is None else diagnostics.video_sample(capture_id, timestamp)

    def _record_depth_diagnostic_rgb(self, capture_id, timestamp, frame) -> None:
        depth = getattr(getattr(self, "_sensor_runtime", None), "astra_depth", None)
        recorder = getattr(depth, "diagnostics", None)
        if recorder is not None:
            try:
                recorder.add_rgb(capture_id, timestamp, frame)
            except Exception as exc:
                logger.warning("Depth diagnostic RGB skipped: %s", exc)

    def _drain_direction_results(self) -> None:
        pool = self._direction_pool
        note = getattr(self._follow_controller, "note_direction_classifier_evidence", None)
        if pool is None or not callable(note):
            return
        drained_results = pool.drain_results()
        for evidence in drained_results:
            self._direction_evidence_ring.append(evidence)
        self._direction_result_backlog.extend(drained_results)
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
                "direction evidence merged capture=%d state=%s side=%s age_ms=%.1f candidates=%d reason=%s",
                evidence.capture_frame_id,
                evidence.state,
                evidence.side,
                evidence.result_age_ms,
                int(getattr(evidence, "candidate_count", 0)),
                evidence.reason,
            )
        if ready:
            self._direction_evidence_merged_total += len(ready)
            now = time.monotonic()
            if now - self._direction_evidence_last_log_ts >= 1.0:
                self._direction_evidence_last_log_ts = now
                latest = ready[-1]
                history = getattr(self._follow_controller, "_target_direction_history", None)
                history_decision = (
                    history.latest_reliable_side()
                    if history is not None
                    else None
                )
                logger.info(
                    "capture_direction_progress merged=%d total=%d current_capture=%d "
                    "worker_pending=%d result_backlog=%d worker_latest_state=%s "
                    "worker_latest_side=%s history_side=%s history_capture=%s age=%.1fms",
                    len(ready),
                    self._direction_evidence_merged_total,
                    current_capture_id,
                    int(getattr(pool, "pending_count", 0)),
                    len(self._direction_result_backlog),
                    latest.state,
                    latest.side,
                    "none" if history_decision is None else history_decision.direction or "none",
                    "none"
                    if history_decision is None
                    or history_decision.last_visible_capture_frame_id is None
                    else int(history_decision.last_visible_capture_frame_id),
                    latest.result_age_ms,
                )
        self._service_historical_direction_backfill()

    def _maybe_schedule_historical_direction_backfill(
        self,
        loss_capture_frame_id: int,
        active_target_id: Optional[int],
    ) -> None:
        """Start one bounded, non-blocking direction evidence backfill.

        This is deliberately called after the normal visual decision.  The
        current decision therefore remains authoritative for this tick; a
        completed hint can only be consumed by a later loss/search decision.
        """
        if not FOLLOW_DIRECTION_HISTORY_ENABLE or not HISTORICAL_DIRECTION_BACKFILL_ENABLE:
            return
        if active_target_id is None or int(active_target_id) <= 0:
            return
        controller = self._follow_controller
        if getattr(controller, "_lost_exit_direction", None) in ("left", "right"):
            return
        if getattr(controller, "search_direction", None) in ("left", "right"):
            return
        state = str(getattr(self, "_vision_control_state", "") or "")
        search_state = str(getattr(controller, "search_state", "") or "")
        if state not in ("lost_confirming", "direction_uncertain") and search_state not in (
            "direction_unresolved",
        ):
            return
        # All asynchronous work in one loss belongs to its FIRST capture, not
        # whichever subsequent frame happened to schedule the worker.
        capture_id = int(getattr(controller, "_direction_loss_capture_id", None) or 0)
        if capture_id <= 0 or capture_id == self._historical_backfill_last_start_capture:
            return
        pending = self._historical_backfill_pending
        if isinstance(pending, dict) and int(pending.get("active_target_id", -1)) == int(active_target_id):
            return
        self._historical_backfill_episode += 1
        self._historical_backfill_last_start_capture = capture_id
        self._historical_backfill_pending = {
            "episode": int(self._historical_backfill_episode),
            "loss_capture_frame_id": capture_id,
            "active_target_id": int(active_target_id),
            "started_at": time.monotonic(),
            "deadline": time.monotonic() + 0.18,
        }
        logger.info(
            "historical_backfill_start episode=%d loss_capture=%d active_uid=%d evidence_ring=%d capture_ring=%d state=%s search_state=%s",
            int(self._historical_backfill_episode),
            capture_id,
            int(active_target_id),
            len(self._direction_evidence_ring),
            len(self._capture_metadata_ring),
            state,
            search_state,
        )
        self._service_historical_direction_backfill()

    def _service_historical_direction_backfill(self) -> None:
        pending = self._historical_backfill_pending
        if not isinstance(pending, dict):
            return
        controller = self._follow_controller
        if int(pending.get("loss_capture_frame_id", 0)) != getattr(controller, "_direction_loss_capture_id", None):
            self._historical_backfill_pending = None
            logger.info("historical_backfill_skipped episode=%d reason=loss_episode_changed loss_capture=%s current_loss=%s",
                        int(pending.get("episode", 0)), pending.get("loss_capture_frame_id"),
                        getattr(controller, "_direction_loss_capture_id", None))
            return
        if (
            getattr(controller, "_lost_exit_direction", None) in ("left", "right")
            or getattr(controller, "search_direction", None) in ("left", "right")
        ):
            self._historical_backfill_pending = None
            logger.info(
                "historical_backfill_skipped episode=%d reason=main_direction_resolved direction=%s",
                int(pending.get("episode", 0)),
                str(
                    getattr(controller, "search_direction", None)
                    or getattr(controller, "_lost_exit_direction", None)
                    or "none"
                ),
            )
            return
        now = time.monotonic()
        history = getattr(self._follow_controller, "_target_direction_history", None)
        anchor = None if history is None else history.latest_visible_evidence()
        anchor_id = None if anchor is None else int(anchor.capture_frame_id)
        evidence = []
        for item in list(self._direction_evidence_ring):
            if anchor_id is not None and int(item.capture_frame_id) <= anchor_id:
                continue
            if float(getattr(item, "result_age_ms", 0.0)) > 500.0:
                continue
            evidence.append(
                HistoricalDirectionCandidate(
                    capture_frame_id=int(item.capture_frame_id),
                    timestamp=float(item.timestamp),
                    state=str(item.state),
                    bbox=item.bbox,
                    score=float(item.score),
                    frame_width=int(item.frame_width),
                    candidate_count=max(0, int(getattr(item, "candidate_count", 0))) or 1,
                )
            )
        result = self._historical_backfill.evaluate(
            evidence,
            loss_capture_frame_id=int(pending["loss_capture_frame_id"]),
            now=now,
            anchor_capture_frame_id=anchor_id,
            anchor_center_ratio=(None if anchor is None else anchor.center_x_ratio),
        )
        deadline_reached = now >= float(pending["deadline"])
        if result.direction is None and not deadline_reached:
            return
        episode = int(pending["episode"])
        self._historical_backfill_pending = None
        window_ids = [
            int(item.capture_frame_id)
            for item in self._direction_evidence_ring
            if int(item.capture_frame_id) < int(pending["loss_capture_frame_id"])
            and int(item.capture_frame_id) >= int(pending["loss_capture_frame_id"]) - HISTORICAL_DIRECTION_BACKFILL_MAX_CAPTURE_GAP
        ]
        window_visible = sum(
            1 for item in self._direction_evidence_ring
            if int(item.capture_frame_id) in window_ids and str(item.state) == "visible"
        )
        window_unknown = sum(
            1 for item in self._direction_evidence_ring
            if int(item.capture_frame_id) in window_ids and str(item.state) == "unknown"
        )
        window_missing = sum(
            1 for item in self._direction_evidence_ring
            if int(item.capture_frame_id) in window_ids and str(item.state) == "missing"
        )
        if result.direction is None:
            logger.info(
                "historical_backfill_end episode=%d loss_capture=%d result=reject reason=%s candidate_frames=%s selected_frames=%s visible=%d unknown=%d missing=%d evidence_age_ms=%.1f",
                episode,
                int(pending["loss_capture_frame_id"]),
                result.reason,
                ",".join(str(value) for value in window_ids) or "none",
                ",".join(str(value) for value in result.selected_capture_frame_ids) or "none",
                window_visible,
                window_unknown,
                window_missing,
                max(0.0, (now - float(pending["started_at"])) * 1000.0),
            )
            return
        accepted = controller.note_historical_direction_hint(
            result.direction,
            active_target_id=int(pending["active_target_id"]),
            first_capture_frame_id=int(result.first_capture_frame_id or 0),
            last_capture_frame_id=int(result.last_capture_frame_id or 0),
            selected_capture_frame_ids=result.selected_capture_frame_ids,
            confidence=float(result.confidence),
            loss_capture_frame_id=int(pending["loss_capture_frame_id"]),
            evidence_timestamp=min(
                (float(item.timestamp) for item in evidence
                 if item.capture_frame_id in result.selected_capture_frame_ids),
                default=0.0,
            ),
            reason=result.reason,
        )
        logger.info(
            "historical_backfill_end episode=%d loss_capture=%d result=%s direction=%s confidence=%.2f candidate_frames=%s selected=%s visible=%d unknown=%d missing=%d accepted=%s evidence_age_ms=%.1f",
            episode,
            int(pending["loss_capture_frame_id"]),
            "accepted" if accepted else "rejected_by_controller",
            result.direction,
            float(result.confidence),
            ",".join(str(value) for value in window_ids) or "none",
            ",".join(str(value) for value in result.selected_capture_frame_ids),
            window_visible,
            window_unknown,
            window_missing,
            bool(accepted),
            max(0.0, (now - float(pending["started_at"])) * 1000.0),
        )

    def _update_detector_identity_lease(self, records, capture_id, capture_timestamp, *,
                                        now, stale, expected_epoch=None):
        """Bind fast control to the original full-verification deadline.

        Publishing a new detector box is not another identity verification.
        All final motor readers consult this immutable lease, including after
        cached feedback waits. Full results may replace it only on a newer
        real capture with current appearance features and accepted identity.
        """
        pipeline = self._rknn_pipeline
        self._late_visual_preserved_evidence = None
        self._late_visual_identity_only = False
        self._late_visual_identity_epoch = None
        self._late_visual_identity_roi_published = False
        processing = getattr(pipeline, "last_identity_processing", {})
        mode = processing.get("mode")
        if mode is None:  # legacy/full-only pipeline adapter
            return True
        previous = getattr(self, "_identity_processing_watermark", None)
        if previous is not None and (capture_id <= previous[0] or capture_timestamp <= previous[1]):
            return False
        self._identity_processing_watermark = (capture_id, capture_timestamp)
        uid = getattr(self._follow_controller, "active_target_id", None)
        matches = [r for r in records if r.reid_uid == uid and r.time_since_update == 0]
        visibility_window = VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC
        previous_visual = getattr(self, "_validated_visual_observation", None)
        previous_depth = getattr(self, "_depth30_linear_snapshot", None)
        hazard_config = getattr(getattr(self, "_bunker_runtime", None), "config", None)
        current_danger = bool(
            getattr(hazard_config, "enabled", False)
            and getattr(hazard_config, "mode", None) == "merged"
            and any(getattr(r, "class_id", None) in hazard_config.class_ids for r in records))
        if (stale and not current_danger and mode == "full" and processing.get("full_features_current")
                and processing.get("capture_frame_id") == capture_id
                and processing.get("capture_timestamp") == capture_timestamp
                and math.isfinite(capture_timestamp) and math.isfinite(now)
                and 0 < capture_timestamp <= now < capture_timestamp + MAX_FULL_RESULT_AGE_SEC
                and isinstance(previous_visual, ValidatedVisualObservation)
                and previous_visual.live(uid, now)
                and previous_visual.continuation_sample_timestamp is None
                and isinstance(previous_depth, tuple) and len(previous_depth) == 4
                and previous_depth[0] == "forward" and previous_depth[2] == uid
                and isinstance(previous_depth[3], (float, int)) and math.isfinite(previous_depth[3])
                and 0 <= now-previous_depth[3] < PersonTracker._depth_linear_max_age_sec("forward")
                and len(matches) == 1 and matches[0].track_id == previous_visual.track_id):
            assignment = self._identity_assignment_debug_for_track(matches[0].track_id)
            if (assignment.get("uid") == uid
                    and PersonTracker._bound_reid_bbox_allowed_for_control(assignment)
                    and not assignment.get("identity_control_rejected")
                    and not assignment.get("identity_recheck_pending")
                    and not assignment.get("search_excluded")
                    and assignment.get("identity_evidence_kind") != "detector_continuation"):
                # Do not publish the late frame as fresh identity or refresh
                # an old lease. The stale handler will independently verify
                # the existing physical Depth grant under the control lock.
                self._late_visual_preserved_evidence = (
                    capture_id, capture_timestamp, previous_visual,
                    getattr(self, "_detector_identity_lease", None))
                return False
        if mode == "detector_continuation":
            lease = None
            if not stale and len(matches) == 1:
                rec = matches[0]
                lease = detector_identity_lease_from_assignment(
                    self._identity_assignment_debug_for_track(rec.track_id),
                    uid=uid, track_id=rec.track_id, capture=capture_id,
                    timestamp=capture_timestamp, now=now,
                )
            observation = (validated_visual_observation(
                uid=uid, track_id=lease.track_id, capture=capture_id,
                timestamp=capture_timestamp, now=now,
                visibility_window=visibility_window, identity_lease=lease,
                capture_max_age_sec=VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC)
                if lease is not None else None)
            publish_visual_identity_evidence(
                self, observation=observation if observation is not None else False,
                lease=lease if lease is not None else False)
            return lease is not None
        observation = None
        full_source_current = bool(mode == "full" and processing.get("full_features_current")
                and processing.get("capture_frame_id") == capture_id
                and processing.get("capture_timestamp") == capture_timestamp
                and math.isfinite(capture_timestamp) and math.isfinite(now)
                and 0 < capture_timestamp <= now < capture_timestamp + MAX_FULL_RESULT_AGE_SEC)
        scheduler = getattr(self, "_depth_async_scheduler", None)
        # CAP97: this exposure occurred while the prior identity was live;
        # the full verification finished after that proof expired. A 210ms
        # processing-time miss forbids new yaw, but does not invalidate a
        # current (<350ms capture age) accepted identity check. This is NOT a
        # lost-target/recheck bootstrap: require the same stable raw track,
        # original live-at-exposure proof, and unchanged safe-follow context.
        identity_only = bool(stale and full_source_current
            and expected_epoch is not None and scheduler is not None
            and scheduler.publication_snapshot()[0] == expected_epoch
            and not current_danger
            and isinstance(previous_visual, ValidatedVisualObservation)
            and previous_visual.uid == uid
            and previous_visual.continuation_sample_timestamp is None
            and previous_visual.capture < capture_id
            and previous_visual.timestamp < capture_timestamp < previous_visual.expires_at
            and previous_visual.validated_at <= capture_timestamp
            and len(matches) == 1 and matches[0].track_id == previous_visual.track_id
            and getattr(self, "running", False)
            and getattr(self, "_depth_roi_safety_clear", False) is True
            and getattr(self, "search_state", None) == "none"
            and getattr(self._follow_controller, "search_state", None) == "none"
            and getattr(self, "_vision_control_state", "").startswith("target_visible")
            and not any(getattr(self, flag, False) for flag in (
                "_explicit_stop_requested", "_runtime_shutdown_requested",
                "_brake_hold_active", "_reacquire_depth_pending"))
            and getattr(self, "_near_yaw_park_request", None) is None)
        full_current = full_source_current and (not stale or identity_only)
        short = getattr(self, "_short_follow_adapter", None)
        if (full_current and not records and not current_danger
                and short is not None and short.owned
                and isinstance(previous_visual, ValidatedVisualObservation)
                and previous_visual.live(uid, now) and motion_identity_live(self, uid, now)):
            # No detector observation is not an identity contradiction. Keep
            # the exact prior proof for its original finite lifetime; no ROI,
            # depth clock, or identity deadline is renewed by an empty frame.
            return True
        if full_current and len(matches) == 1:
            for rec in matches:
                assignment = self._identity_assignment_debug_for_track(rec.track_id)
                if (assignment.get("uid") == uid
                        # Use normal full-control eligibility, not the
                        # stricter gallery/detector-only enrollment policy.
                        and PersonTracker._bound_reid_bbox_allowed_for_control(assignment)
                        and not assignment.get("identity_control_rejected")
                        and not assignment.get("identity_recheck_pending")
                        and not assignment.get("search_excluded")
                        and assignment.get("identity_evidence_kind") != "detector_continuation"):
                    # Gallery quarantine does not revoke a fresh full result's
                    # existing control eligibility. It still prevents creating
                    # another detector-only proof inside the tracker.
                    observation = validated_visual_observation(
                        uid=uid, track_id=rec.track_id, capture=capture_id,
                        timestamp=capture_timestamp, now=now,
                        visibility_window=visibility_window,
                        capture_max_age_sec=VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC)
        full_accepted = observation is not None
        if observation is None and full_current:
            observation = PersonTracker._retain_visual_proof_for_identity_recheck(
                self, records, capture_id, capture_timestamp, now)
        # Publish the completed identity check before ranging/controller work.
        # A rejected new result is not permission to fall back to an old
        # control-completion clock. Startup without a selected UID retains its
        # existing enrollment policy until the next bound visual observation.
        next_lease = getattr(self, "_detector_identity_lease", None)
        if next_lease is not None:
            if full_accepted:
                next_lease = None
            elif observation is None:
                next_lease = False
        publish_visual_identity_evidence(
            self, observation=(observation if observation is not None
                               else False if uid is not None else None),
            lease=next_lease)
        if identity_only and full_accepted:
            self._late_visual_identity_only = True
            self._late_visual_identity_epoch = expected_epoch
            self._late_visual_preserved_evidence = (
                capture_id, capture_timestamp, observation,
                getattr(self, "_detector_identity_lease", None))
            return False  # Still skip this late frame's yaw/control decisions.
        return True  # normal full-result rejection/observation logic still runs

    def _retain_visual_proof_for_identity_recheck(self, records, cap, stamp, now):
        """Preserve the existing narrow recheck bridge, never authorize UID0.

        The normal full-control source and original physical sample remain
        fixed. Reuse ALL existing bridge checks before the current rejected
        observation marker is published; subsequent readers can finish only
        this exact grant, not a new Depth result arriving during the recheck.
        """
        previous = getattr(self, "_validated_visual_observation", None)
        uid = getattr(self._follow_controller, "active_target_id", None)
        if (not isinstance(previous, ValidatedVisualObservation)
                or not previous.live(uid, now) or len(records) != 1
                or getattr(self._follow_controller, "search_state", "none") != "none"
                or getattr(self, "_active_capture_frame_id", None) != cap
                or getattr(self, "_active_capture_timestamp", None) != stamp):
            return None
        rec = records[0]
        assignment = self._identity_assignment_debug_for_track(int(rec.track_id))
        deadline = assignment.get("identity_recheck_deadline")
        if (assignment.get("identity_recheck_pending") is not True
                or assignment.get("reason") not in {"partial_boundary_recheck", "verified_continuation_recheck"}
                or assignment.get("mapped_uid") != uid or assignment.get("uid") != 0
                or rec.reid_uid != 0 or rec.time_since_update != 0
                or previous.track_id != rec.track_id
                or assignment.get("identity_control_rejected") is not True
                or assignment.get("identity_recheck_capture") != cap
                or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)
                or not stamp <= now <= deadline):
            return None
        evidence = PersonTracker._identity_recheck_motion_evidence(self, uid, now)
        width = int(getattr(self._rknn_pipeline, "last_frame_width", VISION_FRAME_WIDTH))
        if (evidence is None or not PersonTracker._existing_drive_grant_during_identity_recheck(
                self, rec, assignment, now, width)
                ):
            return None
        latest = PersonTracker._identity_recheck_motion_evidence(self, uid, now)
        if (latest is None or latest[0] != evidence[0]
                or latest[1] is not evidence[1]):
            return None
        expires = min(previous.expires_at, deadline)
        if evidence[3] is not None:
            expires = min(expires, evidence[3])
        return replace(previous, continuation_sample_timestamp=evidence[2],
                       expires_at=expires)

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
        if not self._capture_health_ok():
            return
        if self._direction_pool is not None:
            self._drain_direction_results()
            read_started = time.perf_counter()
            try:
                capture_id, capture_timestamp, frame = self._capture_queue.get(timeout=0.25)
            except queue.Empty:
                self._capture_health_ok()
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
            recording_timestamp = time.monotonic()
            recording_feedback = self._action_runtime.get_recording_feedback()
            recording_follow = getattr(self, '_video_follow_snapshot', None)
            recording_timing = recording_authority(self)
            self._camera_video_recorder.submit(
                frame,
                capture_frame_id=next_capture_id,
                monotonic_sec=recording_timestamp,
                unix_sec=time.time(),
                wheel_feedback=recording_feedback,
                follow_snapshot=recording_follow,
                linear_timing=recording_timing,
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
        pipeline_started_mono = time.monotonic()
        depth_scheduler = getattr(self, "_depth_async_scheduler", None)
        depth_publication_epoch = (None if depth_scheduler is None else
                                   depth_scheduler.publication_snapshot()[0])
        frame_received_ts = (
            pipeline_started_mono
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
        if capture_frame_id is None and frame_format.upper() in {"BGR", "RGB"}:
            PersonTracker._record_depth_diagnostic_rgb(
                self, current_capture_id, frame_received_ts,
                frame if frame_format.upper() == "BGR" else frame[..., ::-1],
            )
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
        # Arm the locked-UID handoff one state earlier.  A candidate can
        # reappear during the short lost-confirming window, before the
        # controller formally enters search; leaving the identity context
        # disabled here forces that first valid frame through the normal
        # tracker path and loses the beginning of the two-frame handoff.
        reacquire_direction = controller_status_before.direction
        if reacquire_direction not in ("left", "right"):
            fallback_direction = getattr(
                self._follow_controller, "_lost_exit_direction", None
            )
            if fallback_direction in ("left", "right"):
                reacquire_direction = fallback_direction
        # ``search_status`` intentionally reports only the controller's
        # search state.  During the first lost-confirming frame that state can
        # still be ``none`` even though the visual loop has already lost the
        # active target.  Arm the locked-UID identity context from that local
        # state as well, otherwise the first strong replacement track is sent
        # through the normal handoff gate and cannot use the search-only
        # two-frame confirmation path.
        visual_loss_pending = str(
            getattr(self, "_vision_control_state", "") or ""
        ) == "lost_confirming"
        identity_reacquire_active = bool(
            controller_status_before.active_target_id is not None
            and reacquire_direction in ("left", "right")
            and (
                diagnostic_search_active
                or controller_status_before.state == "lost_confirming"
                or (
                    controller_status_before.state == "none"
                    and visual_loss_pending
                )
            )
        )
        # Do not let a frame captured during the post-rotation settling gate
        # enter ReID/candidate confirmation.  The frame may still be useful
        # for diagnostics, but residual chassis yaw can make its appearance
        # and embedding unreliable.
        search_settling_pending = bool(
            diagnostic_search_active
            and self._action_runtime.search_observation_pending()
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
                searching=identity_reacquire_active,
                direction=reacquire_direction,
            )
        diagnostic_mode = getattr(self._rknn_pipeline, "set_search_diagnostic_mode", None)
        if callable(diagnostic_mode):
            diagnostic_mode(diagnostic_search_active)
        fast_context = getattr(self._rknn_pipeline, "set_detector_continuation_context", None)
        if callable(fast_context):
            current_linear = getattr(self, "_depth30_linear_snapshot", None)
            fast_context(
                active_uid=controller_status_before.active_target_id,
                allowed=bool(
                    detector_execution_supported(getattr(self, "_action_runtime", None))
                    and controller_status_before.state == "none" and not identity_reacquire_active
                    and str(getattr(self, "_vision_control_state", "")).startswith("target_visible")
                    and getattr(self, "_vision_control_state", "") != "target_visible_low_quality"
                    and getattr(self, "current_command", None) != ACTION_BACKWARD
                    and not (current_linear and current_linear[0] == "backward")
                    and not getattr(self, "_brake_hold_active", False)
                    and not getattr(self, "_explicit_stop_requested", False)
                    and not getattr(self, "_runtime_shutdown_requested", False)),
            )
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
        frame_context_setter = getattr(self._rknn_pipeline, "set_frame_context", None)
        if callable(frame_context_setter):
            frame_context_setter(
                control_frame_id=int(self.frame_index),
                capture_frame_id=current_capture_id,
                capture_timestamp=frame_received_ts,
                yaw_rate_dps=(
                    None
                    if feedback_before is None
                    else float(feedback_before.yaw_rate_right_dps)
                ),
                integrated_yaw_deg=(
                    None
                    if feedback_before is None
                    else float(feedback_before.integrated_yaw_right_deg)
                ),
            )
        if detector_probe_only and callable(probe_processor):
            records = probe_processor(frame, frame_format)
        else:
            records = self._rknn_pipeline.process_frame(frame, frame_format)
        if not getattr(self, "_numeric_runtime_first_frame_logged", False):
            from rk_vision.numeric_runtime import log_numeric_runtime
            log_numeric_runtime(logger, "first_frame")
            self._numeric_runtime_first_frame_logged = True
        if (not getattr(self, "_numeric_runtime_first_match_logged", False)
                and getattr(self._rknn_pipeline, "last_timing_ms", {}).get("deepsort_matched_count", 0) > 0):
            # SciPy's matcher can load a second BLAS library lazily. Report it
            # once after the first real association, not on every frame.
            from rk_vision.numeric_runtime import log_numeric_runtime
            log_numeric_runtime(logger, "first_track_match")
            self._numeric_runtime_first_match_logged = True
        video_detections = []
        if self._camera_video_recorder is not None:
            video_detections = list(
                getattr(self._rknn_pipeline, "last_detections", []) or []
            )
            # Keep raw probe boxes in structured diagnostics, but draw only
            # clustered representatives in the video overlay. This avoids
            # turning one low-confidence person hypothesis into a misleading
            # wall of fragmented boxes.
            probe_overlay = getattr(
                self._rknn_pipeline, "last_search_probe_clusters", None
            )
            if probe_overlay is None:
                probe_overlay = getattr(
                    self._rknn_pipeline, "last_search_diagnostic_detections", []
                )
            for probe_detection in (probe_overlay or []):
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
        vision_finished_mono = time.monotonic()
        width = int(getattr(self._rknn_pipeline, "last_frame_width", VISION_FRAME_WIDTH))
        height = int(getattr(self._rknn_pipeline, "last_frame_height", VISION_FRAME_HEIGHT))
        control_started = time.perf_counter()
        vision_result_age_sec = max(0.0, vision_finished - pipeline_started)
        self._brake_observation_pipeline_timing = {
            "capture_frame_id": current_capture_id,
            "capture_timestamp": frame_received_ts,
            "capture_to_pipeline_ms": max(0.0, pipeline_started_mono - frame_received_ts) * 1000.,
            "vision_processing_ms": vision_result_age_sec * 1000.,
        }
        stale_result_discarded = vision_result_age_sec > VISION_CONTROL_MAX_RESULT_AGE_SEC
        if not PersonTracker._update_detector_identity_lease(
                self, records, current_capture_id, frame_received_ts,
                now=vision_finished_mono, stale=stale_result_discarded,
                expected_epoch=depth_publication_epoch):
            stale_result_discarded = True
        if not stale_result_discarded or getattr(self, "_late_visual_identity_only", False):
            roi_published = PersonTracker._publish_validated_depth_observation(
                self, records, width, height, current_capture_id, frame_received_ts,
                expected_epoch=depth_publication_epoch)
            self._late_visual_identity_roi_published = bool(
                getattr(self, "_late_visual_identity_only", False) and roi_published)
            if getattr(self, "_late_visual_identity_only", False) and not roi_published:
                # A safety/epoch/geometry rejection of this off-lock mailbox
                # publication cannot leave a new identity-only exception live.
                self._late_visual_identity_only = False
                self._late_visual_identity_epoch = None
                self._late_visual_preserved_evidence = None
                publish_visual_identity_evidence(self, observation=False, lease=False)
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
        diagnostic_formal_evidence = formal_evidence
        diagnostic_probe_evidence = probe_evidence
        exclusion_bindings = PersonTracker._search_exclusion_bindings(self)
        excluded_tracks = PersonTracker._search_excluded_tracks(self, exclusion_bindings)
        if not stale_result_discarded:
            formal_evidence = PersonTracker._filter_search_excluded_evidence(
                self, formal_evidence, exclusion_bindings, source="formal"
            )
            probe_evidence = PersonTracker._filter_search_excluded_evidence(
                self, probe_evidence, exclusion_bindings, source="probe"
            )
            PersonTracker._cancel_excluded_search_observation(self, exclusion_bindings)
        candidate_records = [
            record for record in records
            if int(getattr(record, "track_id", -1)) not in excluded_tracks
        ]
        if search_settling_pending:
            # The frame remains observation-only while the chassis settles,
            # but a strong ReID match must still be allowed to build the same
            # two-frame reacquisition chain used by normal search handling.
            self._observe_search_settling_reid(
                candidate_records,
                width=int(width),
                height=int(height),
            )
        control_records = [] if search_settling_pending else records
        active_target_record, _active_target_assignment = self._active_target_track_record(
            [] if search_settling_pending else candidate_records,
            active_candidate_uid,
        )
        active_target_bbox = (
            None
            if active_target_record is None
            else self._track_record_bbox(active_target_record)
        )

        def preferred_reid_candidate_bbox() -> Optional[Tuple[float, float, float, float]]:
            """Find an unbound detector box that strongly matches the locked UID.

            After a target re-enters the frame, DeepSORT may expose a fresh
            track whose output UID is still zero.  The identity bank has the
            useful evidence already, so use that box only to steer the search
            candidate gate.  It remains an observation until the normal
            two-frame handoff completes; it is never treated as an active
            mapped track here.
            """
            if not diagnostic_search_active or active_candidate_uid is None:
                return None
            try:
                locked_uid = int(active_candidate_uid)
            except (TypeError, ValueError):
                return None
            if locked_uid <= 0:
                return None
            tracker = getattr(self._rknn_pipeline, "tracker", None)
            tracker_config = getattr(tracker, "config", None)
            distance_limit = float(
                getattr(
                    tracker_config,
                    "identity_preferred_search_reacquire_instant_threshold",
                    0.15,
                )
            )
            observations_by_track = {}
            for observation in getattr(tracker, "last_identity_observations", ()) or ():
                try:
                    observations_by_track[int(observation.get("raw_track_id"))] = observation
                except (AttributeError, TypeError, ValueError):
                    continue
            matches = []
            for record in candidate_records or ():
                if int(getattr(record, "class_id", -1)) != PERSON_CLASS_ID:
                    continue
                if int(getattr(record, "time_since_update", 0)) != 0:
                    continue
                assignment = self._identity_assignment_debug_for_track(
                    int(getattr(record, "track_id", -1))
                )
                if int(assignment.get("best_uid") or 0) != locked_uid:
                    continue
                if str(assignment.get("match_source") or "") != "strong":
                    continue
                try:
                    distance = float(assignment.get("distance"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(distance) or distance > distance_limit:
                    continue
                if assignment.get("bbox_quality_ok") is False:
                    continue
                observation = observations_by_track.get(int(getattr(record, "track_id", -1)), {})
                sample_metadata = observation.get("sample_metadata") or {}
                if sample_metadata.get("search_reacquire_context_active") is not True:
                    continue
                bbox = observation.get("detector_bbox") or self._track_record_bbox(record)
                if bbox is None:
                    continue
                try:
                    candidate_bbox = tuple(float(value) for value in bbox)
                except (TypeError, ValueError):
                    continue
                if len(candidate_bbox) != 4 or candidate_bbox[2] <= candidate_bbox[0] or candidate_bbox[3] <= candidate_bbox[1]:
                    continue
                matches.append(
                    (
                        distance,
                        -float(getattr(record, "score", 0.0)),
                        candidate_bbox,
                        int(getattr(record, "track_id", -1)),
                    )
                )
            if not matches:
                return None
            matches.sort(key=lambda item: (item[0], item[1]))
            distance, _score, bbox, track_id = matches[0]
            logger.info(
                "search_candidate_preferred_reid_bbox track_id=%d active_uid=%d "
                "distance=%.3f threshold=%.3f bbox=%s reason=strong_identity_priority",
                int(track_id),
                int(locked_uid),
                float(distance),
                float(distance_limit),
                bbox,
            )
            return bbox

        preferred_candidate_bbox = active_target_bbox
        if preferred_candidate_bbox is None and not search_settling_pending and not stale_result_discarded:
            preferred_candidate_bbox = preferred_reid_candidate_bbox()
        current_candidate_decision = (
            SearchCandidateGateDecision(reason="rotate_settling")
            if search_settling_pending
            else
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
                and active_target_bbox is not None
                and self._bbox_iou_xyxy(
                    active_target_bbox,
                    tuple(float(value) for value in candidate_bbox),
                ) >= 0.15
            )

        def candidate_has_active_reid_evidence(candidate_bbox) -> bool:
            """Recognize an observation-only ReID hint before formal UID confirmation.

            Search reacquisition commonly creates a fresh DeepSORT track.  In
            that short interval ``reid_uid`` can still be zero despite a strong
            feature match. Freshness, matching geometry, and low distance make
            the hint useful for observation diagnostics, not trusted search
            direction. Only candidate_confirmed_identity_uid supplies that
            permission; this helper never assigns UID or authorizes motion.
            """
            active_uid = getattr(self._follow_controller, "active_target_id", None)
            if candidate_bbox is None or active_uid is None or int(active_uid) <= 0:
                return False
            try:
                candidate_box = tuple(float(value) for value in candidate_bbox)
            except (TypeError, ValueError):
                return False
            for record in candidate_records or ():
                if int(getattr(record, "class_id", -1)) != PERSON_CLASS_ID:
                    continue
                if int(getattr(record, "time_since_update", 0)) != 0:
                    continue
                try:
                    record_box = (
                        float(record.x1),
                        float(record.y1),
                        float(record.x2),
                        float(record.y2),
                    )
                except (AttributeError, TypeError, ValueError):
                    continue
                if self._bbox_iou_xyxy(candidate_box, record_box) < 0.15:
                    continue
                assignment = self._identity_assignment_debug_for_track(
                    int(getattr(record, "track_id", -1))
                )
                best_uid = assignment.get("best_uid")
                try:
                    distance = (
                        None
                        if assignment.get("distance") is None
                        else float(assignment.get("distance"))
                    )
                except (TypeError, ValueError):
                    distance = None
                if (
                    best_uid is not None
                    and int(best_uid) == int(active_uid)
                    and distance is not None
                    and math.isfinite(distance)
                    and distance <= SEARCH_CANDIDATE_CONTINUE_ROTATE_MAX_REID_DISTANCE
                    and str(assignment.get("match_source") or "") == "strong"
                    # A ReID match with failed/stale geometry is only an
                    # observation. It must not be used to change the frozen
                    # search direction before the handoff gate accepts it.
                    and assignment.get("reacquire_geometry_ok") is True
                    and assignment.get("bbox_quality_ok") is not False
                ):
                    logger.info(
                        "search_candidate_identity_evidence track_id=%d active_uid=%d "
                        "distance=%.3f threshold=%.3f output_uid=%d",
                        int(getattr(record, "track_id", -1)),
                        int(active_uid),
                        float(distance),
                        float(SEARCH_CANDIDATE_CONTINUE_ROTATE_MAX_REID_DISTANCE),
                        int(getattr(record, "reid_uid", 0)),
                    )
                    return True
            return False

        def candidate_confirmed_identity_uid(candidate_bbox) -> int:
            # Keep the low-quality mapped-UID and strong-ReID hints for
            # observation only. CAP2284 retained mapped_uid=1 after returning
            # output_uid=0; that must not overwrite the right-search anchor.
            binding = PersonTracker._search_binding_for_bbox(candidate_bbox, exclusion_bindings)
            if binding is None:
                return 0
            tracker = getattr(getattr(self, "_rknn_pipeline", None), "tracker", None)
            return confirmed_search_candidate_uid(
                active_uid=active_candidate_uid,
                raw_track_id=binding["raw_track_id"],
                capture_frame_id=current_capture_id,
                capture_timestamp=frame_received_ts,
                records=candidate_records,
                observations=getattr(tracker, "last_identity_observations", ()) or (),
                excluded=binding["exclusion"] is not None,
                person_class_id=PERSON_CLASS_ID,
            )

        current_candidate_matches_target = bool(
            candidate_confirmed_identity_uid(current_candidate_decision.bbox)
        )
        candidate_min_score = max(
            0.01,
            float(
                getattr(
                    getattr(self._follow_controller, "cfg", None),
                    "search_candidate_untracked_min_score",
                    SEARCH_CANDIDATE_UNTRACKED_MIN_SCORE,
                )
            ),
        )
        candidate_evidence_usable = bool(
            current_candidate_decision.bbox is not None
            and float(current_candidate_decision.score) >= candidate_min_score
        )
        lateral_candidate = (
            None
            if not candidate_evidence_usable
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
        deferred_observation_bboxes = PersonTracker._deferred_search_observation_bboxes(
            self, search_active=search_active_before, width=width, height=height,
            formal_candidates=formal_evidence, capture_id=int(current_capture_id),
            capture_timestamp=frame_received_ts, stale=stale_result_discarded or search_settling_pending,
        )
        gate_decision = (
            SearchCandidateGateDecision(reason="rotate_settling")
            if search_settling_pending
            else
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
                deferred_observation_bboxes=deferred_observation_bboxes,
            )
        )
        gate_decision = PersonTracker._retry_search_candidate_observation(
            self, gate_decision, search_active=search_active_before,
            width=width, height=height, formal_candidates=formal_evidence,
            capture_id=int(current_capture_id), capture_timestamp=frame_received_ts,
            stale=stale_result_discarded,
        )
        gate_observation_confirmed = bool(gate_decision.completed and gate_decision.reason not in {
            "observation_timeout_unsettled", "observation_candidate_lost_or_changed",
            DEFERRED_EDGE_COVERAGE,
        })
        note_candidate = lambda bbox, **evidence: PersonTracker._publish_search_candidate_task_evidence(
            self, bbox, **evidence)
        if (
            not stale_result_discarded
            and not search_settling_pending
            and gate_decision.source != "credible_retry"
            and gate_decision.bbox is not None
            and float(gate_decision.score) >= candidate_min_score
            and callable(note_candidate)
        ):
            candidate_identity_match = candidate_has_active_reid_evidence(
                gate_decision.bbox
            )
            candidate_tracked = bool(
                candidate_matches_active_target(gate_decision.bbox)
                or candidate_identity_match
            )
            note_candidate(
                gate_decision.bbox,
                frame_width=width,
                confirmed=gate_observation_confirmed,
                source=str(gate_decision.source),
                candidate_score=float(gate_decision.score),
                candidate_tracked=candidate_tracked,
                candidate_identity_match=candidate_identity_match,
                candidate_confirmed_uid=candidate_confirmed_identity_uid(gate_decision.bbox),
                capture_frame_id=int(current_capture_id),
                now=time.monotonic(),
            )
            noted_binding = PersonTracker._search_binding_for_bbox(gate_decision.bbox, exclusion_bindings)
            self._search_last_noted_candidate_track_id = (
                None if noted_binding is None else int(noted_binding["raw_track_id"])
            )
        note_candidate_missing = lambda: PersonTracker._publish_search_candidate_task_evidence(self)
        candidate_missing_hold = bool(
            not stale_result_discarded
            and not search_settling_pending
            and gate_decision.bbox is None
            and callable(note_candidate_missing)
            and note_candidate_missing()
        )
        if gate_decision.pause_rotation or gate_decision.completed:
            self._apply_search_candidate_gate_decision(gate_decision, prepare_only=True)
        observation_now = time.monotonic()
        bounded_observation_pending = bool(
            self._search_evidence_observation_active
            and observation_now < self._search_evidence_observation_deadline
        )
        # Encoder settling is an action-runtime gate, not a visual evidence
        # hold.  The controller must keep publishing the frozen-direction
        # search pulse while the runtime briefly waits/requeues it; otherwise
        # the next empty detector frame emits STOP and overwrites the
        # same-direction transition hold. Candidate evidence and safety holds
        # still pause rotation here.
        effective_evidence_pause = bool(
            (gate_decision.pause_rotation and not gate_decision.completed)
            or bounded_observation_pending
            or candidate_missing_hold
        )
        # Detector-only boxes are observation evidence, not control evidence.
        # Do not let a single edge/blurred frame reverse the frozen search
        # direction before the candidate gate has completed its streak.
        lateral_candidate_deferred = False
        if lateral_candidate is not None and not stale_result_discarded:
            binding = PersonTracker._search_binding_for_bbox(lateral_candidate.bbox, exclusion_bindings)
            assignment = ({} if binding is None else
                          self._identity_assignment_debug_for_track(binding["raw_track_id"]))
            distance = assignment.get("distance")
            stop_only_evidence = bool(
                binding is not None and binding["exclusion"] is None
                and len(formal_evidence) == 1
                and assignment.get("best_uid", assignment.get("uid")) == active_candidate_uid
                and assignment.get("match_source") == "strong"
                and isinstance(distance, (float, int)) and math.isfinite(distance)
                and distance <= 0.30
                and assignment.get("reacquire_geometry_ok") is True
                and assignment.get("bbox_quality_ok") is True
            )
            PersonTracker._hold_search_reacquire_brake(
                self, bbox=lateral_candidate.bbox, width=width,
                eligible=bool(lateral_candidate.active_target_match or stop_only_evidence),
                confirmed=bool(lateral_candidate.active_target_match),
            )
        if (
            lateral_candidate is not None
            and not bool(lateral_candidate.active_target_match)
            and not bool(gate_observation_confirmed and gate_decision.source != "credible_retry")
        ):
            lateral_candidate_deferred = True
            logger.info(
                "search_candidate_control_deferred capture_frame_id=%d source=%s "
                "score=%.3f hold=%d/%d reason=identity_not_confirmed",
                int(lateral_candidate.capture_frame_id),
                str(lateral_candidate.source),
                float(lateral_candidate.score),
                int(gate_decision.hold_frame),
                int(gate_decision.hold_frames),
            )
            lateral_candidate = None
        self._log_search_candidate_frame_diagnostics(
            capture_frame_id=int(current_capture_id),
            width=width,
            height=height,
            formal_evidence=diagnostic_formal_evidence,
            probe_evidence=diagnostic_probe_evidence,
            records=control_records,
            preferred_bbox=preferred_candidate_bbox,
            current_candidate_decision=current_candidate_decision,
            gate_decision=gate_decision,
            current_candidate_matches_target=current_candidate_matches_target,
            candidate_evidence_usable=candidate_evidence_usable,
            lateral_candidate_deferred=lateral_candidate_deferred,
            stale_result_discarded=stale_result_discarded,
            search_settling_pending=search_settling_pending,
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
            # No new yaw/control decision from a processing-late image. A
            # separately accepted identity-only result may retain its ROI;
            # otherwise the stale handler withdraws the prior motion context.
            identity_only_accepted = bool(getattr(self, "_late_visual_identity_only", False))
            self._handle_stale_vision_result(
                width=width,
                result_age_sec=vision_result_age_sec,
            )
            # Search braking history remains observation-only. Neither this
            # path nor identity-only publication runs normal motion control.
            pending = getattr(getattr(self, "_action_runtime", None),
                              "search_reacquire_brake_pending", None)
            if callable(pending) and pending():
                PersonTracker._record_search_brake_hold_observation(self, records, width, height)
            logger.warning(
                "视觉结果过期，跳过运动决策: control_frame_id=%d capture_frame_id=%d age=%.1fms limit=%.1fms records=%d identity_only=%s",
                int(self.frame_index),
                int(getattr(self, "_active_capture_frame_id", 0)),
                vision_result_age_sec * 1000.0,
                VISION_CONTROL_MAX_RESULT_AGE_SEC * 1000.0,
                len(records),
                identity_only_accepted,
            )
        else:
            try:
                self._consume_track_records(
                    control_records,
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
            retry = getattr(self, "_search_observation_retry", None)
            if retry is not None:
                retry.reset()
            self._search_evidence_observation_active = False
            self._search_evidence_observation_source = "none"
            self._search_evidence_observation_deadline = 0.0
            self._reset_confirmed_search_reacquire()
            self._follow_controller.set_search_observation_hold(False)
        control_finished = time.perf_counter()
        control_finished_mono = time.monotonic()
        capture_timing = capture_pipeline_timing(
            capture_timestamp=frame_received_ts, pipeline_started_at=pipeline_started_mono,
            vision_finished_at=vision_finished_mono, decision_finished_at=control_finished_mono,
        )
        timing = dict(getattr(self._rknn_pipeline, "last_timing_ms", {}) or {})
        logger.info(
            "pipeline_timing: control_frame_id=%d capture_frame_id=%d camera_read_ms=%.2f frame_wrap_ms=%.2f "
            "yolo_preprocess_ms=%.2f yolo_inference_ms=%.2f yolo_decode_ms=%.2f "
            "yolo_nms_ms=%.2f yolo_total_ms=%.2f reid_preprocess_ms=%.2f reid_inference_ms=%.2f "
            "reid_partial_inference_ms=%.2f "
            "reid_postprocess_ms=%.2f reid_total_ms=%.2f tracker_ms=%.2f "
            "predicted_reid_verify_ms=%.2f vision_total_ms=%.2f control_ms=%.2f "
            "end_to_end_ms=%.2f result_age_ms=%.2f stale_result_discarded=%s "
            "reid_color_ms=%.2f reid_postprocess_exclusive_ms=%.2f "
            "tracker_stages_wall_cpu_ms=%s kalman_counts=%s kalman_solver=%s "
            "capture_clock_valid=%s capture_to_pipeline_ms=%s "
            "capture_result_age_ms=%s capture_decision_age_ms=%s "
            "legacy_result_age_scope=processing_duration",
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
            float(timing.get("reid_partial_inference", 0.0)),
            float(timing.get("reid_postprocess", 0.0)),
            float(timing.get("reid_total", 0.0)),
            float(timing.get("tracker", 0.0)),
            float(timing.get("predicted_reid_verify", 0.0)),
            (vision_finished - pipeline_started) * 1000.0,
            (control_finished - control_started) * 1000.0,
            (control_finished - pipeline_started) * 1000.0,
            vision_result_age_sec * 1000.0,
            stale_result_discarded,
            float(timing.get("reid_color", 0.0)),
            float(timing.get("reid_postprocess_exclusive", 0.0)),
            ";".join(
                "%s=%.2f/%.2f" % (
                    key, float(timing.get(key, 0.0)), float(timing.get(key + "_cpu", 0.0))
                )
                for key in (
                    "tracker_association", "deepsort_match", "deepsort_kalman_update",
                    "deepsort_metric_update", "tracker_geometry", "tracker_competition",
                    "tracker_evidence", "tracker_records", "tracker_probe",
                    "identity_match_evidence", "identity_decision",
                    "identity_quarantine", "identity_logging",
                    "deepsort_kf_predict", "deepsort_kf_project", "deepsort_kf_cholesky",
                    "deepsort_kf_solve_lower", "deepsort_kf_solve_upper", "deepsort_kf_fallback",
                )
            ),
            json.dumps({key: int(timing.get("deepsort_" + key, 0)) for key in (
                "tracks_count", "detections_count", "matched_count", "initiated_count",
                "kf_predict_calls", "kf_project_calls", "kf_cholesky_calls",
                "kf_cholesky_retries", "kf_solve_lower_calls", "kf_solve_upper_calls", "kf_fallback_calls",
            )}, separators=(",", ":")),
            os.environ.get("FOLLOW_KALMAN_SOLVER", "numpy"),
            capture_timing["capture_clock_valid"],
            *(None if capture_timing[key] is None else round(capture_timing[key], 2)
              for key in ("capture_to_pipeline_ms", "capture_result_age_ms", "capture_decision_age_ms")),
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
                        initial_enrollment_detail=str(
                            (assignment.get("initial_enrollment_evidence") or {}).get("reason") or ""),
                        initial_enrollment_blocker=str(
                            (assignment.get("initial_enrollment_state") or {}).get("bridge_blocker") or ""),
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
            video_distance_state = self._last_frame_distance_state
            video_distance_m = None
            video_distance_source = "none"
            video_distance_detail = ""
            if video_distance_state is not None:
                video_distance_m = getattr(video_distance_state, "used_distance_m", None)
                try:
                    video_distance_m = (
                        None if video_distance_m is None else float(video_distance_m)
                    )
                except (TypeError, ValueError):
                    video_distance_m = None
                video_distance_source = str(
                    getattr(video_distance_state, "source", "none") or "none"
                )
                video_distance_detail = str(
                    getattr(video_distance_state, "source_detail", "") or ""
                )
            self._camera_video_recorder.update_overlay(
                current_capture_id,
                video_detections,
                tracks=video_tracks,
                control=VideoControlOverlay(
                    control_frame_id=int(self.frame_index),
                    active_target_id=controller_status_after.active_target_id,
                    selected_target_id=controller_status_after.selected_target_id,
                    candidate_bbox=(
                        None if lateral_candidate is None else lateral_candidate.bbox
                    ),
                    candidate_score=(
                        0.0 if lateral_candidate is None else float(lateral_candidate.score)
                    ),
                    candidate_source=(
                        "none" if lateral_candidate is None else str(lateral_candidate.source)
                    ),
                    candidate_matches_target=(
                        False
                        if lateral_candidate is None
                        else bool(lateral_candidate.active_target_match)
                    ),
                    action_name=video_action_name,
                    requested_rpm=video_rpm,
                    yaw_rate_dps=(
                        None
                        if feedback_for_video is None
                        else float(feedback_for_video.yaw_rate_right_dps)
                    ),
                    result_age_ms=vision_result_age_sec * 1000.0,
                    target_distance_m=video_distance_m,
                    distance_source=video_distance_source,
                    distance_detail=video_distance_detail,
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
                        left_forward_rpm=None if feedback is None else feedback.left_forward_rpm,
                        right_forward_rpm=None if feedback is None else feedback.right_forward_rpm,
                        feedback_timestamp=None if feedback is None else feedback.timestamp,
                        left_position_deg=None if feedback is None else feedback.left_position_deg,
                        right_position_deg=None if feedback is None else feedback.right_position_deg,
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

    def _latch_runtime_shutdown(self, reason: str = "runtime_shutdown") -> None:
        """Signal-safe state latch: no control/motor lock, logging or I/O.

        Leave the action worker alive until a direct STOP has been attempted.
        It also observes this latch, so a blocked main/vision thread cannot
        prevent its independent shutdown-stop service from running.
        """
        if not getattr(self, "_runtime_shutdown_reason", None):
            self._runtime_shutdown_reason = str(reason or "runtime_shutdown")
        self._runtime_shutdown_requested = True
        self._explicit_stop_requested = True
        self._explicit_stop_provenance = ("shutdown", self._runtime_shutdown_reason)
        self.running = False
        self.stop_action_execution = True
        self._use_soft_stop_next = False

    def _send_runtime_shutdown_stop(self, reason: str = "runtime_shutdown") -> bool:
        """Stop before worker cleanup, independently of the control mutex.

        Serial I/O still uses its own transaction locks; this deliberately
        does not promise recovery from a blocked serial driver. No driver is
        opened implicitly when initialization failed.
        """
        self._latch_runtime_shutdown(reason)
        backend = getattr(self, "_motor_backend", None)
        if backend is None or backend.driver is None:
            logger.info("runtime_shutdown stop_skipped=True reason=motor_not_initialized implicit_open=False")
            return False
        try:
            self._action_runtime.send_stop_with_brake_hold(self._runtime_shutdown_reason)
            return True
        except Exception:
            logger.exception("runtime_shutdown direct_stop_failed=True reason=%s", self._runtime_shutdown_reason)
            return False

    def _run_shutdown_step(self, label: str, callback, timeout_sec: float = SHUTDOWN_STEP_TIMEOUT_SEC) -> bool:
        """Run one potentially blocking close operation with a hard timeout."""
        done = threading.Event()
        errors = []
        results = []
        started = time.monotonic()

        def _close_worker() -> None:
            try:
                results.append(callback())
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
        if results and results[0] is False:
            logger.warning("退出清理步骤未完成: %s elapsed=%.1fms cleanup_deferred=True", label, elapsed_ms)
            return False
        logger.info("退出清理步骤完成: %s elapsed=%.1fms", label, elapsed_ms)
        return True

    def _release_rknn_camera(self) -> bool:
        self._capture_stop_event.set()
        capture_thread = self._capture_thread
        if capture_thread is not None and capture_thread.is_alive():
            capture_thread.join(timeout=SHUTDOWN_STEP_TIMEOUT_SEC)
            if capture_thread.is_alive():
                # A blocked native read must not race release() on that same
                # camera. Keep ownership for diagnostics/process teardown;
                # no hardware restart is permitted after the shutdown latch.
                logger.error("RKNN camera release_deferred=True capture_worker_alive=True "
                             "native_read_not_interrupted=True")
                return False
        self._capture_thread = None
        camera = self._rknn_camera
        # The reader is confirmed stopped; it is now safe to close its owner.
        self._rknn_camera = None
        if camera is None:
            return True
        camera.release()
        logger.info("RKNN camera released")
        return True

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
            # Revoke movement and issue the first STOP BEFORE diagnostics,
            # worker joins or control-state cleanup can wait on any mutex.
            self._send_runtime_shutdown_stop("runtime_shutdown")
            self.action_stop_event.set()
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
            observer = getattr(self, "_depth_track_observer", None)
            if observer is not None:
                self._run_shutdown_step("depth track shadow", observer.close)
            self._run_shutdown_step("camera video recorder", self._close_camera_video_recorder)
            self._run_shutdown_step("RKNN camera", self._release_rknn_camera)
            self._run_shutdown_step("RKNN vision pipeline", self._close_rknn_pipeline)
            self._run_shutdown_step("bunker runtime", self._bunker_runtime.close)
            self._run_shutdown_step("sensor runtime", self._sensor_runtime.close)
            logger.info("人员跟踪（避障版本）已停止")


def main():
    # Install only for the application, not module import/offline tests.
    # Formatting/output retains the originating record timestamp; a stalled
    # console/tee must not stall a controller while it holds a state/I/O lock.
    async_console = BoundedAsyncConsoleHandler(
        stream=console_handler.stream, formatter=console_handler.formatter,
        capacity=4096, shutdown_timeout_sec=0.5,
    )
    async_console.setLevel(console_handler.level)
    had_console = console_handler in logger.handlers
    logger.removeHandler(console_handler)
    logger.addHandler(async_console)
    try:
        logger.info("async_console_started capacity=4096 producer_io=False shutdown_budget_ms=500")
        # 模型路径支持命令行第一个参数，不传则用配置/默认值。
        model_path = sys.argv[1] if len(sys.argv) >= 2 else None
        if model_path is None:
            logger.info(f"未传模型路径，使用默认: {YOLO_MODEL_PATH}")
        tracker = PersonTracker(model_path=model_path)
        def _request_shutdown(signum, _frame):
            # A Python signal can interrupt this thread while it owns the
            # command lock. Never do motor I/O/logging in that callback.
            tracker._latch_runtime_shutdown("runtime_shutdown_signal")

        signal.signal(signal.SIGINT, _request_shutdown)
        signal.signal(signal.SIGTERM, _request_shutdown)
        tracker.run()
    finally:
        # run() stops/releases the motors before reaching this diagnostic
        # drain. Never synchronously fall back to a possibly blocked stream.
        logger.info("async_console_stopping stats=%s", async_console.stats())
        async_console.close()
        logger.removeHandler(async_console)
        if had_console:
            logger.addHandler(console_handler)


if __name__ == "__main__":
    main()
