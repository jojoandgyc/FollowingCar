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
MSSD60EHB RS485 motor driver on /dev/ttyS*.
"""

import json
import logging
import math
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Tuple, Optional

from car_control_modular.config_loader import preload_config_from_argv
from car_control_modular.control_types import HazardState, ObstacleState, PersonTarget, SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController

# 先config读取变量到 环境变量
LOADED_CONFIG = preload_config_from_argv()

# Board HAL imports; motor output uses direct MSSD60EHB RS485.
try:
    from yolov_hal import Yolov, cvtdl_object_t
    _YOLOV_IMPORT_ERROR = None
except Exception as e:
    Yolov = None
    cvtdl_object_t = None
    _YOLOV_IMPORT_ERROR = e
# from utrasonic_hal import Utrasonic  # 保留：后续需要超声融合时再打开
from track_first_person2 import FirstPersonTracker
from bunker_hazard_detector import BunkerHazardMonitor, RKNNBunkerHazardDetector, check_hazard_from_dets

try:
    from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline
    _RKNN_VISION_IMPORT_ERROR = None
except Exception as e:
    RKNNVisionConfig = None
    RKNNVisionPipeline = None
    _RKNN_VISION_IMPORT_ERROR = e

try:
    from ir_hal import IR
    _IR_IMPORT_ERROR = None
except Exception as e:
    IR = None
    _IR_IMPORT_ERROR = e

try:
    from mmwave_hal import MmWaveRadar
    _MMWAVE_IMPORT_ERROR = None
except Exception as e:
    MmWaveRadar = None
    _MMWAVE_IMPORT_ERROR = e

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Y8_WORKDIR_DEFAULT = os.path.join(SCRIPT_DIR, "yolov8-track")

SamplePersonV8Wrapper = None
TrackRecord = None
_SAMPLE_WRAPPER_IMPORT_ERROR = None
try:
    _y8_import_dir = os.path.abspath(Y8_WORKDIR_DEFAULT)
    if _y8_import_dir not in sys.path:
        sys.path.insert(0, _y8_import_dir)
    from sample_personv8_track_wrapper import SamplePersonV8Wrapper, TrackRecord
except Exception as e:
    _SAMPLE_WRAPPER_IMPORT_ERROR = e

# RK3588 motor backend: direct /dev/ttyS* MSSD60EHB RS485 driver.
MOTOR_BACKEND = os.environ.get("MOTOR_BACKEND", "rs485_mssd").strip().lower()
if MOTOR_BACKEND not in {"rs485", "rs485_mssd", "mssd", "mssd60ehb"}:
    raise RuntimeError(f"Unsupported MOTOR_BACKEND={MOTOR_BACKEND!r}; RK3588 runtime requires rs485_mssd")
MOTOR_RS485_PORT = os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0").strip()
MOTOR_RS485_SLAVE_ID = int(os.environ.get("MOTOR_RS485_SLAVE_ID", "1"))
MOTOR_RS485_BAUDRATE = int(os.environ.get("MOTOR_RS485_BAUDRATE", "9600"))
MOTOR_RS485_TIMEOUT = float(os.environ.get("MOTOR_RS485_TIMEOUT", "0.3"))
MOTOR_RS485_LIB_DIR = os.environ.get("MOTOR_RS485_LIB_DIR", "car_control_modular/vendor").strip()
MOTOR_RS485_MAX_TARGET = int(os.environ.get("MOTOR_RS485_MAX_TARGET", "11000"))
MOTOR_LEFT_SIGN = int(os.environ.get("MOTOR_LEFT_SIGN", "-1"))
MOTOR_RIGHT_SIGN = int(os.environ.get("MOTOR_RIGHT_SIGN", "1"))
MOTOR_RS485_STOP_MODE = os.environ.get("MOTOR_RS485_STOP_MODE", "normal").strip().lower()

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
MODULE_MMWAVE_ENABLE = os.environ.get("MODULE_MMWAVE_ENABLE", "1").strip() != "0"
MODULE_ULTRASONIC_ENABLE = os.environ.get("MODULE_ULTRASONIC_ENABLE", "0").strip() != "0"
DISTANCE_SOURCE = os.environ.get("DISTANCE_SOURCE", "mmwave").strip().lower()
VISION_MMWAVE_SOURCE_ALIASES = {"vision_mmwave", "vision-mmwave", "vision+mmwave", "mmwave_vision"}
VISION_MMWAVE_ENABLED = DISTANCE_SOURCE in VISION_MMWAVE_SOURCE_ALIASES

# 跟踪参数
DETECTION_CONFIRM_FRAMES = 3
LOST_CONFIRM_FRAMES = 3   # 防抖：连续 3 帧无人才确认丢人并开始旋转，避免检测偶发漏检
ACTION_COOLDOWN = 1  # 降低冷却时间，使前进更流畅
LOST_CONFIRM_FRAMES = int(os.environ.get("FOLLOW_LOST_CONFIRM_FRAMES", str(LOST_CONFIRM_FRAMES)))
SEARCH_COOLDOWN = 1
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))
FOLLOW_CENTER_LEFT_RATIO = float(os.environ.get("FOLLOW_CENTER_LEFT_RATIO", str(1.0 / 6.0)))
FOLLOW_CENTER_RIGHT_RATIO = float(os.environ.get("FOLLOW_CENTER_RIGHT_RATIO", str(4.0 / 6.0)))
FOLLOW_USE_VERTICAL_CENTER_GATE = os.environ.get("FOLLOW_USE_VERTICAL_CENTER_GATE", "0").strip() != "0"
FOLLOW_SEARCH_BEFORE_FIRST_PERSON = os.environ.get("FOLLOW_SEARCH_BEFORE_FIRST_PERSON", "1").strip() != "0"
FOLLOW_STARTUP_SEARCH_DELAY_SEC = float(os.environ.get("FOLLOW_STARTUP_SEARCH_DELAY_SEC", "0.0"))
SIDE_IR_BLOCKS_ROTATION = os.environ.get("SIDE_IR_BLOCKS_ROTATION", "1").strip() != "0"

# YOLO 模型路径（相对运行目录，与板子 zkwl-runtime 目录结构一致）
YOLO_MODEL_PATH = os.environ.get(
    "VISION_MODEL_PATH",
    os.path.join(SCRIPT_DIR, "models", "yolo11s.rknn"),
)

# Vision engine:
# - rknn: RK3588 RKNN Runtime path; frames must be supplied externally.
# - hal/sample_reid: legacy HD05075A/CVI compatibility only.
VISION_REID_ENABLE = os.environ.get("VISION_REID_ENABLE", "0").strip() != "0"
VISION_ENGINE = os.environ.get(
    "VISION_ENGINE",
    "rknn",
).strip().lower()
VISION_SAMPLE_WORKDIR = os.path.abspath(
    os.environ.get("VISION_SAMPLE_WORKDIR", Y8_WORKDIR_DEFAULT).strip()
)
VISION_SAMPLE_BINARY = os.environ.get("VISION_SAMPLE_BINARY", "./sample_personv8_track").strip()
VISION_REID_MODEL_PATH = os.environ.get(
    "VISION_REID_MODEL_PATH",
    os.path.join(SCRIPT_DIR, "models", "deepsort.rknn"),
).strip()
VISION_FRAME_WIDTH = int(os.environ.get("VISION_FRAME_WIDTH", "1920"))
VISION_FRAME_HEIGHT = int(os.environ.get("VISION_FRAME_HEIGHT", "1080"))
VISION_HFOV_DEG = float(os.environ.get("VISION_HFOV_DEG", "90.0"))
VISION_EFFECTIVE_FPS = max(0.1, float(os.environ.get("VISION_EFFECTIVE_FPS", "4.0")))
VISION_SAMPLE_SINGLE_PERSON = os.environ.get("VISION_SAMPLE_SINGLE_PERSON", "1").strip() != "0"
VISION_SAMPLE_ECHO_RAW = os.environ.get("VISION_SAMPLE_ECHO_RAW", "0").strip() != "0"
VISION_SAMPLE_DET_CONF = os.environ.get("VISION_SAMPLE_DET_CONF", "0.55").strip()
VISION_SAMPLE_LOG_PATH = os.environ.get(
    "VISION_SAMPLE_LOG_PATH",
    os.path.join(SCRIPT_DIR, "logs", "sample_person_reid_runtime.log"),
).strip()
VISION_TRACK_LOG_ENABLE = os.environ.get("VISION_TRACK_LOG_ENABLE", "1").strip() != "0"
VISION_TRACK_LOG_EMPTY_EVERY = int(os.environ.get("VISION_TRACK_LOG_EMPTY_EVERY", "20"))
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
VISION_MMWAVE_DISTANCE_BIAS_M = float(os.environ.get("VISION_MMWAVE_DISTANCE_BIAS_M", os.environ.get("MMWAVE_DISTANCE_BIAS_M", "0.30")))
VISION_MMWAVE_MIN_DISTANCE_M = float(os.environ.get("VISION_MMWAVE_MIN_DISTANCE_M", os.environ.get("MMWAVE_MIN_DISTANCE_M", "0.02")))
VISION_MMWAVE_MAX_DISTANCE_M = float(os.environ.get("VISION_MMWAVE_MAX_DISTANCE_M", os.environ.get("MMWAVE_MAX_DISTANCE_M", "8.0")))
VISION_MMWAVE_MIN_OUTPUT_DISTANCE_M = float(os.environ.get("VISION_MMWAVE_MIN_OUTPUT_DISTANCE_M", os.environ.get("MMWAVE_MIN_OUTPUT_DISTANCE_M", "0.03")))
VISION_MMWAVE_HARD_STOP_TTL_SEC = float(os.environ.get("VISION_MMWAVE_HARD_STOP_TTL_SEC", "0.30"))
VISION_MMWAVE_LOG_EVERY_FRAMES = int(os.environ.get("VISION_MMWAVE_LOG_EVERY_FRAMES", "10"))
_default_missing_forward = "15" if VISION_MMWAVE_ENABLED else "50"
DISTANCE_MISSING_FORWARD_PERCENT = int(os.environ.get("DISTANCE_MISSING_FORWARD_PERCENT", _default_missing_forward))

# MSSD motor RPM feedback is not wired yet; keep disabled until the RS485 driver exposes a read path.
MOTOR_FEEDBACK_POLL_INTERVAL_SEC = 3.0
ENABLE_MOTOR_RPM_FEEDBACK = False
BRAKE_HOLD_REFRESH_INTERVAL_SEC = float(os.environ.get("BRAKE_HOLD_REFRESH_INTERVAL_SEC", "0.10"))   # brake 保持态下重复下发间隔（秒）
# 前进速度：使用 doc 中“写入M1/M2运行状态和速度百分比”（0x0002/0x0003）
# golf_foll通信内容V6_20260304-6D加百分比调速.doc：正常前进调速不建议低于 MIN_FORWARD_PERCENT（默认 10%）；
# 需要停住时用 state=00(stop) / 03(brake)，避免长期用 1~9% 前进。
USE_PERCENT_SPEED = True
MIN_FORWARD_PERCENT = int(os.environ.get("MIN_FORWARD_PERCENT", "10"))  # 与协议一致：前进态百分比下限（0 表示停/滑行停，仍用 stop 而非“极低速前进”）
MAX_FORWARD_PERCENT = int(os.environ.get("MAX_FORWARD_PERCENT", "100"))  # 上限按百分比协议允许到 100%
MOTOR_PERCENT_LIMIT = max(0, min(100, int(os.environ.get("MOTOR_PERCENT_LIMIT", str(MAX_FORWARD_PERCENT)))))
FORWARD_SPEED_LE_1_3_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_1_3_PERCENT", "30"))
FORWARD_SPEED_LE_1_7_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_1_7_PERCENT", "35"))
FORWARD_SPEED_LE_2_1_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_2_1_PERCENT", "42"))
FORWARD_SPEED_LE_2_6_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_2_6_PERCENT", "50"))
FORWARD_SPEED_LE_3_2_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_3_2_PERCENT", "60"))
FORWARD_SPEED_LE_3_8_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_3_8_PERCENT", "70"))
FORWARD_SPEED_LE_4_5_PERCENT = int(os.environ.get("FORWARD_SPEED_LE_4_5_PERCENT", "75"))
FORWARD_SPEED_FAR_PERCENT = int(os.environ.get("FORWARD_SPEED_FAR_PERCENT", "80"))
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
ROTATE_PULSE_PAUSE_SEC = 0.06  # 脉冲旋转：每次旋转到点后强制停顿（秒），避免连续转出大圈
ROTATE_CHAIN_MEMORY_SEC = 1.00  # 旋转记忆窗口（秒）：该窗口内再次旋转按 CHAIN 力度，避免偶发大角度

# 旋转（差速）参数：一个轮正转、另一个轮反转（百分比）
# 直行后首轮转向用较高力度；连续转向（上一动已是旋转，或刚结束一次旋转后的下一拍）用较低力度——轮姿不同所需力度不同
ROTATE_TURN_PERCENT_FROM_FORWARD = int(os.environ.get("ROTATE_TURN_PERCENT_FROM_FORWARD", "70"))  # 上一动不是旋转（如前进、首次、长时间停后）
ROTATE_TURN_PERCENT_CHAIN = int(os.environ.get("ROTATE_TURN_PERCENT_CHAIN", "55"))          # 上一动是旋转（左/右切换）或刚结束旋转后的下一拍转向
VISIBLE_STEER_INNER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_INNER_RATIO_PERCENT", "80"))
VISIBLE_STEER_OUTER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_OUTER_RATIO_PERCENT", "105"))
VISIBLE_STEER_STRONG_INNER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_STRONG_INNER_RATIO_PERCENT", "70"))
VISIBLE_STEER_STRONG_OUTER_RATIO_PERCENT = int(os.environ.get("VISIBLE_STEER_STRONG_OUTER_RATIO_PERCENT", "110"))
VISIBLE_STEER_STRONG_MARGIN_RATIO = float(os.environ.get("VISIBLE_STEER_STRONG_MARGIN_RATIO", "0.12"))
# M1/M2 对应左右轮的映射（若方向不对可切换）
M1_IS_LEFT_WHEEL = os.environ.get("M1_IS_LEFT_WHEEL", "0").strip() in {"1", "true", "yes", "on"}

# 动作参数
ROTATE_DURATION = float(os.environ.get("ROTATE_DURATION", "0.15"))  # 单次旋转最多执行时长（秒），到点即 brake 锁轮（见 _brake_lock_after_rotate_pulse）
# 主循环取帧间隔（秒），越大越不占摄像头/编码管线，可减轻与板子录像 VENC 冲突导致的 drop audio / 卡死
PROCESS_FRAME_INTERVAL = 0.06   # 约16Hz，跟踪够用且减轻与录像争用（原 0.03 约33Hz 易与 CVI_RECORDER 抢 VENC）
# get_frame 超时（秒）：>0 时主循环用线程+join(timeout) 调用 process_frame，超时则跳过本帧避免卡死；0=不启用超时
GET_FRAME_TIMEOUT = 0.35  # 稍短以便卡住时更快跳过本帧，减轻“停住没反应”的感觉

# 距离参数（中间3×3 + 毫米波跟随时）
# 人在此距离以内不前进（速度视为 0）；超过后才按 _forward_percent_for_distance 分档。
TARGET_DISTANCE = float(os.environ.get("TARGET_DISTANCE", "1.0"))
# 小于此距离（米）发 brake / 硬停 / 边缘区「太近不转」（统一阈值）
FOLLOW_BRAKE_DISTANCE_M = float(os.environ.get("FOLLOW_BRAKE_DISTANCE_M", "0.5"))
MMWAVE_OBSTACLE_THRESHOLD = float(os.environ.get("MMWAVE_OBSTACLE_THRESHOLD", "1.0"))  # 毫米波障碍物阈值（米），前方 1.0 m 内有障碍且未检测到人则停
ULTRASONIC_OBSTACLE_THRESHOLD = MMWAVE_OBSTACLE_THRESHOLD  # 保留旧变量名，后续超声融合时兼容
FORWARD_1M_DISTANCE = 1.0  # 绕障前进距离（米）

# ==================== 常量定义 ====================
PERSON_CLASS_ID = int(os.environ.get("PERSON_CLASS_ID", "0"))  # YOLO中人的类别ID

# 沙坑/水坑安全停止：
# - split：当前双模型方案，request_0428.py 跑人员模型，后台另起沙坑/水坑模型。
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

# 动作类型
ACTION_FORWARD = 0
ACTION_ROTATE_LEFT = 1
ACTION_ROTATE_RIGHT = 2
ACTION_STOP = 3
ACTION_STEER_LEFT = 4
ACTION_STEER_RIGHT = 5

# 设置日志
logger = logging.getLogger("PersonTracker")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class TrackMemorySelector:
    def __init__(
        self,
        memory_sec: float,
        allow_new_target_after_sec: float,
        reacquire_same_id: bool,
        reacquire_by_geometry: bool,
        reacquire_confirm_frames: int,
        max_center_jump_ratio: float,
        min_area_similarity: float,
        max_aspect_diff: float,
        log_enable: bool = True,
    ) -> None:
        self.memory_sec = max(0.1, float(memory_sec))
        self.allow_new_target_after_sec = max(
            self.memory_sec,
            float(allow_new_target_after_sec),
        )
        self.reacquire_same_id = bool(reacquire_same_id)
        self.reacquire_by_geometry = bool(reacquire_by_geometry)
        self.reacquire_confirm_frames = max(1, int(reacquire_confirm_frames))
        self.max_center_jump_ratio = max(0.05, float(max_center_jump_ratio))
        self.min_area_similarity = max(0.0, min(1.0, float(min_area_similarity)))
        self.max_aspect_diff = max(0.0, float(max_aspect_diff))
        self.log_enable = bool(log_enable)

        self.target_id: Optional[int] = None
        self.target_bbox: Optional[Tuple[float, float, float, float]] = None
        self.target_center: Optional[Tuple[float, float]] = None
        self.target_area: float = 0.0
        self.last_seen_ts: float = 0.0
        self.last_seen_frame: int = 0
        self._pending_id: Optional[int] = None
        self._pending_streak: int = 0
        self._last_log_key: Optional[Tuple[Any, ...]] = None

    @staticmethod
    def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _area(bbox: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _aspect(bbox: Tuple[float, float, float, float]) -> Optional[float]:
        x1, y1, x2, y2 = bbox
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 1.0 or h <= 1.0:
            return None
        return w / h

    def _log(self, key: Tuple[Any, ...], message: str, *args: Any) -> None:
        if not self.log_enable or key == self._last_log_key:
            return
        self._last_log_key = key
        logger.info(message, *args)

    def _candidate_id(self, cand: Dict[str, Any]) -> int:
        return int(cand["stable_id"])

    def _accept(self, cand: Dict[str, Any], frame_index: int, reason: str) -> List[Dict[str, Any]]:
        bbox = cand["bbox"]
        sid = self._candidate_id(cand)
        self.target_id = sid
        self.target_bbox = bbox
        self.target_center = self._center(bbox)
        self.target_area = float(cand.get("area") or self._area(bbox))
        self.last_seen_ts = time.monotonic()
        self.last_seen_frame = int(frame_index)
        self._pending_id = None
        self._pending_streak = 0
        self._log(
            ("accept", reason, sid),
            "track_memory accept frame=%d reason=%s target_id=%s area=%.0f center=(%.1f,%.1f)",
            frame_index,
            reason,
            sid,
            self.target_area,
            self.target_center[0],
            self.target_center[1],
        )
        return [cand]

    def _geometry_match(self, cand: Dict[str, Any]) -> bool:
        if self.target_bbox is None or self.target_center is None or self.target_area <= 1.0:
            return False
        bbox = cand["bbox"]
        cx, cy = self._center(bbox)
        tx, ty = self.target_center
        dist = math.hypot(cx - tx, cy - ty)
        ref_size = math.sqrt(max(1.0, self.target_area))
        if dist > max(40.0, ref_size * self.max_center_jump_ratio):
            return False

        area = float(cand.get("area") or self._area(bbox))
        area_sim = min(area, self.target_area) / max(area, self.target_area, 1.0)
        if area_sim < self.min_area_similarity:
            return False

        ref_ar = self._aspect(self.target_bbox)
        cand_ar = self._aspect(bbox)
        if ref_ar is not None and cand_ar is not None:
            aspect_diff = abs(cand_ar - ref_ar) / max(ref_ar, cand_ar, 1e-6)
            if aspect_diff > self.max_aspect_diff:
                return False
        return True

    def _best_geometry_candidate(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        matched = [cand for cand in candidates if self._geometry_match(cand)]
        if not matched or self.target_center is None:
            return None
        tx, ty = self.target_center
        return min(
            matched,
            key=lambda cand: (
                math.hypot(self._center(cand["bbox"])[0] - tx, self._center(cand["bbox"])[1] - ty),
                -float(cand.get("area", 0.0)),
            ),
        )

    def select(self, frame_index: int, candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        now = time.monotonic()
        candidates = sorted(candidates, key=lambda cand: float(cand.get("area", 0.0)), reverse=True)
        if not candidates:
            if self.target_id is not None:
                age = now - self.last_seen_ts
                if age < self.allow_new_target_after_sec:
                    reason = "lost_hold" if age <= self.memory_sec else "lost_protect"
                    self._log(
                        (reason, self.target_id),
                        "track_memory %s frame=%d target_id=%s age=%.2fs memory=%.2fs allow_after=%.2fs",
                        reason,
                        frame_index,
                        self.target_id,
                        age,
                        self.memory_sec,
                        self.allow_new_target_after_sec,
                    )
                    return [], reason
                self._log(
                    ("release_empty", self.target_id),
                    "track_memory release frame=%d target_id=%s reason=no_candidates age=%.2fs allow_after=%.2fs",
                    frame_index,
                    self.target_id,
                    age,
                    self.allow_new_target_after_sec,
                )
                self.target_id = None
                self.target_bbox = None
                self.target_center = None
                self.target_area = 0.0
            return [], "no_candidates"

        if self.target_id is None:
            return self._accept(candidates[0], frame_index, "new_largest"), "new_largest"

        same_id = [cand for cand in candidates if self._candidate_id(cand) == self.target_id]
        if self.reacquire_same_id and same_id:
            return self._accept(same_id[0], frame_index, "same_id"), "same_id"

        age = now - self.last_seen_ts
        if age <= self.memory_sec:
            if self.reacquire_by_geometry:
                cand = self._best_geometry_candidate(candidates)
                if cand is not None:
                    sid = self._candidate_id(cand)
                    if self._pending_id != sid:
                        self._pending_id = sid
                        self._pending_streak = 1
                    else:
                        self._pending_streak += 1
                    if self._pending_streak >= self.reacquire_confirm_frames:
                        return self._accept(cand, frame_index, "geometry_reacquire"), "geometry_reacquire"
                    self._log(
                        ("pending", self.target_id, sid, self._pending_streak),
                        "track_memory pending frame=%d old_id=%s cand_id=%s streak=%d/%d",
                        frame_index,
                        self.target_id,
                        sid,
                        self._pending_streak,
                        self.reacquire_confirm_frames,
                    )
                    return [], "geometry_pending"
            self._log(
                ("hold_other", self.target_id),
                "track_memory hold frame=%d target_id=%s age=%.2fs candidates=%d",
                frame_index,
                self.target_id,
                age,
                len(candidates),
            )
            return [], "hold_target_missing"

        if age < self.allow_new_target_after_sec:
            self._log(
                ("wait_new", self.target_id),
                "track_memory wait_new frame=%d target_id=%s age=%.2fs allow_after=%.2fs",
                frame_index,
                self.target_id,
                age,
                self.allow_new_target_after_sec,
            )
            return [], "wait_new_target"

        return self._accept(candidates[0], frame_index, "memory_expired_new_largest"), "memory_expired_new_largest"


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
        """Initialize the RK3588 tracker, vision pipeline, sensors, and MSSD motor backend."""
        self._model_path = model_path if model_path is not None else YOLO_MODEL_PATH
        self._Utrasonic = None
        self._vision_engine = VISION_ENGINE
        if self._vision_engine in ("rknn", "rk", "rk3588", "rk_rknn", "rknn_runtime"):
            self._vision_engine = "rknn"
        elif self._vision_engine in ("reid", "sample", "sample_reid", "sample_personv8", "sample-personv8"):
            self._vision_engine = "sample_reid"
        elif self._vision_engine in ("hal", "yolov", "yolov_hal", "cvi_yolo_hal"):
            self._vision_engine = "hal"
        else:
            raise RuntimeError(f"Unknown VISION_ENGINE: {VISION_ENGINE!r}")
        if self._vision_engine == "sample_reid" and model_path is None and "VISION_MODEL_PATH" not in os.environ:
            self._model_path = os.path.join(VISION_SAMPLE_WORKDIR, "person_v8n_cv181x_int8_640.cvimodel")
        self._vision_wrapper = None
        self._vision_iter = None
        self._rknn_pipeline = None
        self._last_rknn_no_frame_log_ts = 0.0
        self._yolov_hal_active = False
        logger.info("MSSD60EHB RS485 motor backend enabled: port=%s, limit=%d%%", MOTOR_RS485_PORT, MOTOR_PERCENT_LIMIT)
        if LOADED_CONFIG is not None:
            logger.info("已加载配置文件: %s", LOADED_CONFIG.path)
        logger.info(
            "模块开关: vision=%s, ir=%s, mmwave=%s, ultrasonic=%s, distance_source=%s, bunker=%s(%s)",
            MODULE_VISION_ENABLE,
            MODULE_IR_ENABLE,
            MODULE_MMWAVE_ENABLE,
            MODULE_ULTRASONIC_ENABLE,
            DISTANCE_SOURCE,
            BUNKER_AVOID_ENABLE,
            BUNKER_DETECT_MODE,
        )
        logger.info("Safety config: side_ir_blocks_rotation=%s", SIDE_IR_BLOCKS_ROTATION)
        logger.info(
            "Distance/motion config: mmwave_match_mode=%s target=%.2fm brake=%.2fm speed_tiers=[<=1.3:%d,<=1.7:%d,<=2.1:%d,<=2.6:%d,<=3.2:%d,<=3.8:%d,<=4.5:%d,far:%d] max=%d fallback=%d",
            VISION_MMWAVE_MATCH_MODE,
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
        )
        logger.info(
            "Track memory: enabled=%s auto=%s deepsort_hold=%.2fs(max_unmatched=%d/fps=%.2f) memory=%.2fs allow_new_after=%.2fs same_id=%s geometry=%s confirm=%d",
            TRACK_MEMORY_ENABLED,
            TRACK_MEMORY_AUTO_FROM_DEEPSORT,
            TRACK_MEMORY_DEEPSORT_HOLD_SEC,
            DEEPSORT_MAX_UNMATCHED_NUM,
            VISION_EFFECTIVE_FPS,
            TRACK_MEMORY_SEC,
            TRACK_MEMORY_ALLOW_NEW_TARGET_AFTER_SEC,
            TRACK_MEMORY_REACQUIRE_SAME_ID,
            TRACK_MEMORY_REACQUIRE_BY_GEOMETRY,
            TRACK_MEMORY_REACQUIRE_CONFIRM_FRAMES,
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
            raise RuntimeError("当前 request_0428_modular.py 仍依赖视觉主循环，vision 模块不能关闭")
       
        # 初始化 IR 传感器
        if MODULE_IR_ENABLE:
            if IR is None:
                raise RuntimeError(f"IR HAL 导入失败: {_IR_IMPORT_ERROR}")
            ret = IR.init()
            if ret != 0:
                raise RuntimeError("IR传感器初始化失败")
            logger.info("IR传感器初始化成功（前方、左侧、右侧）")
        else:
            logger.info("IR传感器模块已关闭")
       
        if MODULE_ULTRASONIC_ENABLE:
            from utrasonic_hal import Utrasonic

            self._Utrasonic = Utrasonic
            ret = self._Utrasonic.init()
            if ret != 0:
                raise RuntimeError("Utrasonic传感器初始化失败")
            logger.info("Utrasonic传感器初始化成功")
        else:
            logger.info("Utrasonic传感器模块已关闭")

        # 初始化毫米波雷达
        if MODULE_MMWAVE_ENABLE:
            if MmWaveRadar is None:
                raise RuntimeError(f"毫米波雷达 HAL 导入失败: {_MMWAVE_IMPORT_ERROR}")
            ret = MmWaveRadar.init()
            if ret != 0:
                raise RuntimeError("毫米波雷达初始化失败")
            logger.info("毫米波雷达初始化成功")
        else:
            logger.info("毫米波雷达模块已关闭")
       
        # YOLO 调用方式与 yolov_test.py 一致（板子稳定运行）
        if self._vision_engine in ("sample_reid", "rknn"):
            logger.info("%s mode: skip legacy Yolov HAL init", self._vision_engine)
            ret = 0
        else:
            if Yolov is None:
                raise RuntimeError(f"legacy Yolov HAL import failed: {_YOLOV_IMPORT_ERROR}")
            ret = Yolov.init()
        if ret != 0:
            raise RuntimeError(f"CVI_HAL_Yolov_Init failed with {hex(ret)}")
        if self._vision_engine == "hal":
            logger.info("Yolov HAL 初始化成功")
        # 转为绝对路径再传给 HAL，避免 C 库不解析相对路径导致 0xc0010101
        model_path_abs = os.path.abspath(self._model_path)
        if self._vision_engine == "rknn" and not os.path.exists(model_path_abs):
            logger.warning("RKNN YOLO model not found yet: %s", model_path_abs)
        elif not os.path.exists(model_path_abs):
            raise RuntimeError(
                f"模型文件不存在: {model_path_abs} （当前工作目录: {os.getcwd()}）。"
                f"可传参: python3 track_person_follow_bizhang_car.py <模型路径>"
            )
        if self._vision_engine == "sample_reid":
            logger.info("sample_reid detection model: %s", model_path_abs)
            logger.info("sample_reid mode: skip Yolov HAL open_model; child process will load model")
            ret = 0
        elif self._vision_engine == "rknn":
            logger.info("RKNN vision model path: %s", model_path_abs)
            ret = 0
        else:
            logger.info(f"打开模型: {model_path_abs}")
            ret = Yolov.open_model(model_path_abs)
        if ret != 0:
            raise RuntimeError(
                f"CVI_HAL_OpenModel failed with {hex(ret)}. "
                f"模型路径: {model_path_abs} （可传参: python3 track_person_follow_bizhang_car.py <模型路径>）"
            )
        if self._vision_engine == "hal":
            logger.info(f"YOLO 模型加载完成: {model_path_abs}")
        self._yolov_hal_active = self._vision_engine == "hal"
        if self._vision_engine == "sample_reid":
            logger.info("Yolov HAL remains inactive for sample_reid vision engine")
            if SamplePersonV8Wrapper is None:
                raise RuntimeError(f"sample_personv8_track_wrapper import failed: {_SAMPLE_WRAPPER_IMPORT_ERROR}")
            if not os.path.exists(VISION_SAMPLE_WORKDIR):
                raise RuntimeError(f"VISION_SAMPLE_WORKDIR not found: {VISION_SAMPLE_WORKDIR}")
            sample_binary_abs = _resolve_existing_or_workdir_path(VISION_SAMPLE_BINARY, VISION_SAMPLE_WORKDIR)
            if not os.path.exists(sample_binary_abs):
                raise RuntimeError(f"VISION_SAMPLE_BINARY not found: {sample_binary_abs}")
            reid_model_abs = _resolve_existing_or_workdir_path(VISION_REID_MODEL_PATH, VISION_SAMPLE_WORKDIR)
            if not os.path.exists(reid_model_abs):
                raise RuntimeError(f"VISION_REID_MODEL_PATH not found: {reid_model_abs}")
            self._vision_wrapper = SamplePersonV8Wrapper(
                model_path=model_path_abs,
                reid_model_path=reid_model_abs,
                image_width=VISION_FRAME_WIDTH,
                image_height=VISION_FRAME_HEIGHT,
                hfov_deg=VISION_HFOV_DEG,
                runtime_base=os.environ.get("ZKWL_RUNTIME_BASE", "/mnt/system/runtime/zkwl-runtime").strip(),
                echo_raw=VISION_SAMPLE_ECHO_RAW,
                single_person_mode=VISION_SAMPLE_SINGLE_PERSON,
                log_path=VISION_SAMPLE_LOG_PATH,
                workdir=VISION_SAMPLE_WORKDIR,
                binary_path=VISION_SAMPLE_BINARY,
                det_conf=VISION_SAMPLE_DET_CONF,
            )
            logger.info(
                "sample_reid vision ready: workdir=%s, binary=%s, det_model=%s, reid_model=%s",
                VISION_SAMPLE_WORKDIR,
                sample_binary_abs,
                model_path_abs,
                reid_model_abs,
            )

        if self._vision_engine == "rknn":
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

        self._bunker_monitor: Optional[BunkerHazardMonitor] = None
        self._bunker_rknn_detector: Optional[RKNNBunkerHazardDetector] = None
        self._last_bunker_stop_log_ts = 0.0
        self._bunker_active_frames = 0
        self._last_bunker_state_ts = 0.0
        if BUNKER_AVOID_ENABLE:
            if BUNKER_DETECT_MODE == "split":
                bunker_model_abs = _resolve_existing_or_workdir_path(BUNKER_MODEL_PATH, BUNKER_WORKDIR)
                if not os.path.exists(bunker_model_abs):
                    raise RuntimeError(
                        f"沙坑/水坑模型文件不存在: {bunker_model_abs}。"
                        f"可设置 BUNKER_MODEL_PATH 或关闭 BUNKER_AVOID_ENABLE。"
                    )
                if BUNKER_ENGINE in {"rknn", "rk", "rk3588", "lite", "lite2", "rknnlite"}:
                    self._bunker_rknn_detector = RKNNBunkerHazardDetector(
                        model_path=bunker_model_abs,
                        class_ids=tuple(BUNKER_STOP_CLASS_IDS),
                        class_names=BUNKER_CLASS_NAMES,
                        score_threshold=BUNKER_STOP_SCORE_THRESHOLD,
                        area_ratio_stop=BUNKER_STOP_AREA_RATIO,
                        num_classes=BUNKER_NUM_CLASSES,
                        det_conf=BUNKER_SAMPLE_DET_CONF,
                        input_size=BUNKER_RKNN_INPUT_SIZE,
                        nms_threshold=BUNKER_RKNN_NMS_THRESHOLD,
                        input_format=BUNKER_RKNN_INPUT_FORMAT,
                        box_format=BUNKER_RKNN_BOX_FORMAT,
                        target=RKNN_TARGET,
                        core_mask=BUNKER_RKNN_CORE_MASK,
                        backend=BUNKER_RKNN_BACKEND,
                    )
                    self._bunker_rknn_detector.start()
                    logger.info(
                        "沙坑/水坑检测: split RKNN 模型已启动（model=%s, class_ids=%s, area_ratio_stop=%.3f, conf=%.3f）",
                        bunker_model_abs,
                        sorted(BUNKER_STOP_CLASS_IDS),
                        BUNKER_STOP_AREA_RATIO,
                        BUNKER_SAMPLE_DET_CONF,
                    )
                else:
                    bunker_binary_abs = _resolve_existing_or_workdir_path(BUNKER_SAMPLE_BINARY, BUNKER_WORKDIR)
                    if not os.path.exists(BUNKER_WORKDIR):
                        raise RuntimeError(f"沙坑/水坑检测工作目录不存在: {BUNKER_WORKDIR}")
                    if not os.path.exists(bunker_binary_abs):
                        raise RuntimeError(
                            f"sample_personv8_track 不存在: {bunker_binary_abs}。"
                            f"可设置 BUNKER_SAMPLE_BINARY/BUNKER_WORKDIR。"
                        )
                    self._bunker_monitor = BunkerHazardMonitor(
                        model_path=bunker_model_abs,
                        frame_width=int(os.environ.get("BUNKER_FRAME_WIDTH", "1920")),
                        frame_height=int(os.environ.get("BUNKER_FRAME_HEIGHT", "1080")),
                        runtime_base=os.environ.get("ZKWL_RUNTIME_BASE", "/mnt/system/runtime/zkwl-runtime").strip(),
                        class_ids=tuple(BUNKER_STOP_CLASS_IDS),
                        class_names=BUNKER_CLASS_NAMES,
                        score_threshold=BUNKER_STOP_SCORE_THRESHOLD,
                        area_ratio_stop=BUNKER_STOP_AREA_RATIO,
                        num_classes=BUNKER_NUM_CLASSES,
                        det_conf=BUNKER_SAMPLE_DET_CONF,
                        get_frame_timeout_ms=BUNKER_GET_FRAME_TIMEOUT_MS,
                        loop_period_ms=BUNKER_LOOP_PERIOD_MS,
                        binary_path=bunker_binary_abs,
                        workdir=BUNKER_WORKDIR,
                        restart_delay=BUNKER_SPLIT_RESTART_DELAY,
                        active_hold_sec=BUNKER_SPLIT_ACTIVE_HOLD_SEC,
                        echo_raw=BUNKER_SPLIT_ECHO_RAW,
                        logger=logger,
                    )
                    self._bunker_monitor.start()
                    logger.info(
                        "沙坑/水坑检测: split 独立 sample 模型已启动（model=%s, class_ids=%s, area_ratio_stop=%.3f）",
                        bunker_model_abs,
                        sorted(BUNKER_STOP_CLASS_IDS),
                        BUNKER_STOP_AREA_RATIO,
                    )
            elif BUNKER_DETECT_MODE == "merged":
                logger.info(
                    "沙坑/水坑检测: merged 主模型模式（class_ids=%s, area_ratio_stop=%.3f）",
                    sorted(BUNKER_STOP_CLASS_IDS),
                    BUNKER_STOP_AREA_RATIO,
                )
            elif BUNKER_DETECT_MODE == "off":
                logger.info("沙坑/水坑检测: BUNKER_DETECT_MODE=off，已关闭")
            else:
                raise RuntimeError(f"未知 BUNKER_DETECT_MODE: {BUNKER_DETECT_MODE!r}")
       
        # Motor command state (new commands can preempt old commands).
        self.current_command = None  # 当前执行的命令（ACTION_*）
        self.command_start_time = None  # 命令开始时间
        self.command_lock = threading.Lock()  # 保护命令状态的锁
        self.motor_io_lock = threading.Lock()
        self._mssd_driver = None
        self._mssd_driver_ctx = None
        self._mssd_classes = None
       
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
        # 当前帧 obj_meta，用于 Ctrl+C 时若刚好在 detect 之后、stop 之前则 finally 里补 stop 释放资源
        self._current_obj_meta = None
        # 根据距离调节的前进速度百分比（_process_detections 中设置，_send_robot_command 使用）
        self._current_forward_percent = 0
        self._current_steer_base_percent = 0
        self._current_steer_inner_ratio_percent = VISIBLE_STEER_INNER_RATIO_PERCENT
        self._current_steer_outer_ratio_percent = VISIBLE_STEER_OUTER_RATIO_PERCENT
        # 差速转向力度：由动作线程在切入旋转时按「上一动是否旋转」设定（见 ROTATE_TURN_PERCENT_*）
        self._current_rotate_turn_percent = ROTATE_TURN_PERCENT_FROM_FORWARD
        # 单次旋转时长到点后置 True，下一拍若再发旋转则用 CHAIN 力度（无需与上一指令无缝衔接）
        self._rotate_follows_previous_rotate = False
        # 保留首人跟踪器参数作备用；当前主流程按每帧最大面积人员跟随，不再锁定特定身份。
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
        self._brake_hold_refresh_interval_sec = BRAKE_HOLD_REFRESH_INTERVAL_SEC
        self._last_brake_hold_send_ts = 0.0
        # 脉冲旋转停顿：旋转到点后强制停一会儿再允许下一次旋转
        self._rotate_pause_until_ts = 0.0
        # 最近一次旋转结束时间（用于判定短时间内再次旋转仍算“连续旋转”）
        self._last_rotate_end_ts = 0.0
        # 上一帧实际入队动作（用于搜索冷却等逻辑区分“前进保持”与“旋转脉冲”）
        self._last_dispatched_action = None  # type: Optional[int]
        # 仅转向结束停：下一拍 ACTION_STOP 走软停（0%%），不抢紧急 brake
        self._use_soft_stop_next = False
        self._follow_controller = FollowSafetyController(
            FollowPolicyConfig(
                lost_confirm_frames=LOST_CONFIRM_FRAMES,
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
                use_vertical_center_gate=FOLLOW_USE_VERTICAL_CENTER_GATE,
                search_before_first_seen=FOLLOW_SEARCH_BEFORE_FIRST_PERSON,
                startup_search_delay_sec=FOLLOW_STARTUP_SEARCH_DELAY_SEC,
                side_ir_blocks_rotation=SIDE_IR_BLOCKS_ROTATION,
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
        self._last_explicit_stop_reason = ""
        self._last_vision_mmwave_distance_m: Optional[float] = None
        self._last_vision_mmwave_ts = 0.0
        self._last_vision_mmwave_log_key = None
        self._track_memory = (
            TrackMemorySelector(
                memory_sec=TRACK_MEMORY_SEC,
                allow_new_target_after_sec=TRACK_MEMORY_ALLOW_NEW_TARGET_AFTER_SEC,
                reacquire_same_id=TRACK_MEMORY_REACQUIRE_SAME_ID,
                reacquire_by_geometry=TRACK_MEMORY_REACQUIRE_BY_GEOMETRY,
                reacquire_confirm_frames=TRACK_MEMORY_REACQUIRE_CONFIRM_FRAMES,
                max_center_jump_ratio=TRACK_MEMORY_MAX_CENTER_JUMP_RATIO,
                min_area_similarity=TRACK_MEMORY_MIN_AREA_SIMILARITY,
                max_aspect_diff=TRACK_MEMORY_MAX_ASPECT_DIFF,
                log_enable=TRACK_MEMORY_LOG_ENABLE,
            )
            if TRACK_MEMORY_ENABLED
            else None
        )
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
        self._start_action_thread()

    def _can_release_brake_hold(self, action: int) -> bool:
        """
        brake 保持态释放条件：
        - 旋转命令（左/右）可释放
        - 前进命令且前进百分比 > 0 可释放
        其余（STOP / 前进0%）不释放。
        """
        if action in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT):
            return True
        if action in (ACTION_STEER_LEFT, ACTION_STEER_RIGHT):
            return int(getattr(self, "_current_steer_base_percent", 0)) > 0
        if action == ACTION_FORWARD:
            return int(getattr(self, "_current_forward_percent", 0)) > 0
        return False

    def _coast_down_before_rotate(self):
        """从前进切到转向前：线性降低前进百分比到 0（滑行停），再开始差速转向。"""
        if not USE_PERCENT_SPEED or not ROTATE_PREP_COAST_ENABLE:
            return
        p0 = max(0, min(100, int(getattr(self, "_current_forward_percent", 0))))
        if p0 <= 0:
            return
        steps = max(1, int(ROTATE_PREP_COAST_STEPS))
        total = max(0.05, float(ROTATE_PREP_COAST_TOTAL_SEC))
        for i in range(steps):
            pct = max(0, int(p0 * (1.0 - (i + 1) / float(steps))))
            # 协议不建议 1~9% 前进：中间步若落在该区间则抬到 MIN，最后一步仍为 0（stop）
            if pct > 0 and pct < MIN_FORWARD_PERCENT:
                pct = MIN_FORWARD_PERCENT
            try:
                self._send_percent_drive(pct)
            except Exception as e:
                logger.warning("转向前滑行降速失败: %s", e)
            time.sleep(total / steps)
        logger.debug("转向前滑行降速完成: %d%% -> 0%%", p0)

    def _should_hard_stop_now(self) -> bool:
        """
        最高优先级硬停条件：
        - split 沙坑/水坑模型触发危险面积阈值
        - 前方 IR 触发
        - 距离传感器距离有效且 < FOLLOW_BRAKE_DISTANCE_M（与中间3×3 跟车 brake 阈值一致）
        该检查会被动作线程在持续前进时调用，用于打断前进避免“帧间空窗期撞人”。
        """
        try:
            if self._get_split_bunker_stop_state() is not None:
                return True
        except Exception:
            pass
        if MODULE_IR_ENABLE and IR is not None:
            try:
                if IR.is_triggered(IR.IDX_1):  # IR1 = 前方
                    return True
            except Exception:
                # 传感器异常时不在这里强行停，避免误停；真正安全策略以主流程为准
                pass
        try:
            if DISTANCE_SOURCE in VISION_MMWAVE_SOURCE_ALIASES:
                d = self._get_recent_vision_mmwave_distance()
            else:
                d = self._get_distance()
            return (d is not None) and (d < FOLLOW_BRAKE_DISTANCE_M)
        except Exception:
            return False
   
    def _start_action_thread(self):
        """启动动作执行线程"""
        self.action_stop_event.clear()
        self.action_thread = threading.Thread(target=self._action_executor, daemon=True)
        self.action_thread.start()
        logger.info("动作执行线程已启动")
   
    def _action_executor(self):
        """动作执行线程：新指令优先，可被新指令打断；无命令时不发 STOP，只在超时或收到停止信号时发"""
        while not self.action_stop_event.is_set():
            try:
                # brake 保持态：即便没有新命令，也要持续补发 brake，避免坡上松开
                if self._brake_hold_active:
                    now = time.time()
                    if now - self._last_brake_hold_send_ts >= self._brake_hold_refresh_interval_sec:
                        self._last_brake_hold_send_ts = now
                        try:
                            self._send_percent_brake()
                        except Exception as e:
                            logger.warning("brake 保持态下重复锁轮失败: %s", e)

                # 轮速反馈（可选）：与运动控制在同一线程，读寄存器会长时间阻塞，默认关闭
                if ENABLE_MOTOR_RPM_FEEDBACK:
                    now = time.time()
                    if now - self._last_motor_feedback_check_ts >= self._motor_feedback_poll_interval_sec:
                        self._last_motor_feedback_check_ts = now
                        rpm_pair = self._read_motor_feedback_rpm()
                        if rpm_pair is not None:
                            rpm_m1_raw, rpm_m2_raw, rpm_m1_s, rpm_m2_s = rpm_pair
                            if rpm_m1_s < 0 and rpm_m2_s < 0:
                                if not self._brake_hold_active:
                                    logger.warning(
                                        "检测到 M1/M2 同时倒转，进入 brake 保持态: signed=%d,%d raw=%d,%d",
                                        rpm_m1_s,
                                        rpm_m2_s,
                                        rpm_m1_raw,
                                        rpm_m2_raw,
                                    )
                                self._brake_hold_active = True
                                self._use_soft_stop_next = False
                                self._last_brake_hold_send_ts = 0.0  # 立刻刷新一次 brake

                # 尝试获取新动作（非阻塞）
                try:
                    action = self.action_queue.get_nowait()
                    with self.command_lock:
                        if action in (ACTION_FORWARD, ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT, ACTION_STEER_LEFT, ACTION_STEER_RIGHT):
                            self.stop_action_execution = False
                            self.person_detected_flag = False
                        # 脉冲旋转停顿期：旋转命令延后执行（重新入队），避免 continue 丢指令 + 忙等占满 CPU
                        if action in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT) and time.time() < float(
                            getattr(self, "_rotate_pause_until_ts", 0.0)
                        ):
                            logger.debug("脉冲旋转停顿中，延后旋转命令: %s", action)
                            try:
                                self._send_percent_brake()
                            except Exception as e:
                                logger.warning("停顿期 brake 失败: %s", e)
                            try:
                                self.action_queue.put_nowait(action)
                            except queue.Full:
                                logger.warning("停顿期旋转命令回队失败(queue满)")
                            time.sleep(0.02)
                            continue
                        if self._brake_hold_active:
                            if self._can_release_brake_hold(action):
                                self._brake_hold_active = False
                                logger.info("收到可释放命令，解除 brake 保持态: %s", action)
                            else:
                                logger.info("brake 保持态生效，忽略命令: %s（仅旋转或前进>0可解除）", action)
                                try:
                                    self._send_percent_brake()
                                except Exception as e:
                                    logger.warning("brake 保持态下再次锁轮失败: %s", e)
                                continue
                        if action == self.current_command:
                            # 连续同指令：
                            # - 前进：允许刷新计时，保持持续前进流畅
                            # - 旋转：不刷新计时，否则主线程每帧投递同一旋转会导致 ROTATE_DURATION 永远到不了、越转越大圈
                            if action in (ACTION_FORWARD, ACTION_STEER_LEFT, ACTION_STEER_RIGHT):
                                self.command_start_time = time.time()
                                logger.debug("连续同指令(前进/轮差)，刷新计时: %s", action)
                            else:
                                logger.debug("连续同指令(非前进)，不刷新计时: %s", action)
                        else:
                            # 新指令不同，切换命令（左转/右转/停止时再发 STOP 或转向）
                            old_cmd = self.current_command
                            if (
                                USE_PERCENT_SPEED
                                and ROTATE_PREP_COAST_ENABLE
                                and old_cmd in (ACTION_FORWARD, ACTION_STEER_LEFT, ACTION_STEER_RIGHT)
                                and action in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT)
                            ):
                                self._coast_down_before_rotate()
                            # 前进→旋转：默认不先发 brake，直接由下方 _send_robot_command(旋转) 发差速
                            # 旋转力度：上一动是旋转 → CHAIN；否则 → FROM_FORWARD
                            if action in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT):
                                now_ts = time.time()
                                if old_cmd in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT):
                                    self._current_rotate_turn_percent = ROTATE_TURN_PERCENT_CHAIN
                                elif self._rotate_follows_previous_rotate:
                                    self._current_rotate_turn_percent = ROTATE_TURN_PERCENT_CHAIN
                                    self._rotate_follows_previous_rotate = False
                                elif (now_ts - float(getattr(self, "_last_rotate_end_ts", 0.0))) <= float(ROTATE_CHAIN_MEMORY_SEC):
                                    self._current_rotate_turn_percent = ROTATE_TURN_PERCENT_CHAIN
                                else:
                                    self._current_rotate_turn_percent = ROTATE_TURN_PERCENT_FROM_FORWARD
                            if action in (ACTION_FORWARD, ACTION_STEER_LEFT, ACTION_STEER_RIGHT):
                                self._rotate_follows_previous_rotate = False
                            self.current_command = action
                            self.command_start_time = time.time()
                            if action in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT):
                                logger.info(
                                    "收到新指令: %s, 开始执行，差速转向力度 %s%%",
                                    action,
                                    self._current_rotate_turn_percent,
                                )
                            else:
                                logger.info("收到新指令: %s, 开始执行", action)
                except queue.Empty:
                    # 没有新动作，检查停止信号或旋转时长
                    with self.command_lock:
                        if self.current_command is not None and self.command_start_time is not None:
                            if self.stop_action_execution or self.person_detected_flag:
                                # 收到停止信号，立即停止
                                logger.info("收到停止信号，立即停止当前命令")
                                was_rotate = self.current_command in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT)
                                if was_rotate:
                                    self._rotate_follows_previous_rotate = True
                                self.current_command = None
                                self.command_start_time = None
                                if was_rotate:
                                    self._brake_lock_after_rotate_pulse()
                                else:
                                    self._send_stop_with_brake_hold("stop_signal")
                                self.stop_action_execution = False
                                self.person_detected_flag = False
                            else:
                                # 仅对旋转做时长限制，到点即停，避免单次转太大
                                elapsed = time.time() - self.command_start_time
                                if self.current_command in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT) and elapsed >= ROTATE_DURATION:
                                    logger.debug("单次旋转时长到 %.2fs，停止旋转", ROTATE_DURATION)
                                    self._rotate_follows_previous_rotate = True
                                    self._last_rotate_end_ts = time.time()
                                    self.current_command = None
                                    self.command_start_time = None
                                    self._rotate_pause_until_ts = time.time() + float(ROTATE_PULSE_PAUSE_SEC)
                                    self._brake_lock_after_rotate_pulse()
                
                # 如果有当前命令且未收到停止信号，持续发送命令
                with self.command_lock:
                    if self.current_command is not None and not self.stop_action_execution and not self.person_detected_flag:
                        # 旋转超时保护：若已明显超过单次旋转时长，优先强制 brake，避免网络抖动导致单次转角过大
                        if (
                            self.current_command in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT)
                            and self.command_start_time is not None
                            and (time.time() - self.command_start_time) > (ROTATE_DURATION + 0.05)
                        ):
                            logger.warning("旋转超时保护触发，强制 brake 防止大角度")
                            self._rotate_follows_previous_rotate = True
                            self._last_rotate_end_ts = time.time()
                            self.current_command = None
                            self.command_start_time = None
                            self._rotate_pause_until_ts = time.time() + float(ROTATE_PULSE_PAUSE_SEC)
                            self._brake_lock_after_rotate_pulse()
                            time.sleep(0.01)
                            continue
                        # brake 保持态下不发送其他运动命令（等待可释放命令解除）
                        if self._brake_hold_active:
                            time.sleep(0.01)
                            continue
                        # 最高优先级：持续前进时做“硬停”检查，避免帧间空窗期撞人
                        if self.current_command in (ACTION_FORWARD, ACTION_STEER_LEFT, ACTION_STEER_RIGHT):
                            now = time.time()
                            if now - self._last_hard_stop_check_ts >= self._hard_stop_check_interval_sec:
                                self._last_hard_stop_check_ts = now
                                if self._should_hard_stop_now():
                                    logger.warning(
                                        "硬停触发：沙坑/水坑视觉危险、前方IR触发或毫米波距离<%.2fm，立刻发送STOP并打断前进",
                                        FOLLOW_BRAKE_DISTANCE_M,
                                    )
                                    self.current_command = None
                                    self.command_start_time = None
                                    self.stop_action_execution = False
                                    self.person_detected_flag = False
                                    self._send_stop_with_brake_hold("hard_stop")
                                    time.sleep(0.01)
                                    continue
                        self._send_robot_command(self.current_command)
                
                time.sleep(0.01)  # 每10ms检查一次
            except Exception as e:
                logger.error(f"动作执行异常: {e}")
                # 出错时也要停止
                with self.command_lock:
                    self.current_command = None
                    self.command_start_time = None
                self._send_stop_with_brake_hold("action_executor_exception")
   
    def _read_motor_feedback_rpm(self) -> Optional[Tuple[int, int, int, int]]:
        """MSSD RS485 RPM feedback is not implemented in the current board driver."""
        return None

    def _clip_motor_percent(self, percent: int) -> int:
        return max(0, min(MOTOR_PERCENT_LIMIT, int(percent)))

    def _ensure_mssd_driver(self):
        if self._mssd_driver is not None:
            return self._mssd_driver
        if not MOTOR_RS485_LIB_DIR:
            raise RuntimeError("MOTOR_RS485_LIB_DIR is empty")
        if MOTOR_RS485_LIB_DIR not in sys.path:
            sys.path.insert(0, MOTOR_RS485_LIB_DIR)
        from mssd_60ehb_rs485 import ControlSignal, DriverMode, MSSD60EHB, SystemMode

        ctx = MSSD60EHB(
            MOTOR_RS485_PORT,
            slave_id=MOTOR_RS485_SLAVE_ID,
            baudrate=MOTOR_RS485_BAUDRATE,
            timeout=MOTOR_RS485_TIMEOUT,
        )
        driver = ctx.__enter__() if hasattr(ctx, "__enter__") else ctx
        driver.set_system_mode(SystemMode.INDEPENDENT_CLOSED_LOOP)
        driver.set_control_signal(ControlSignal.RS485)
        driver.set_driver_mode(DriverMode.FOC)
        self._mssd_driver_ctx = ctx
        self._mssd_driver = driver
        self._mssd_classes = (ControlSignal, DriverMode, MSSD60EHB, SystemMode)
        logger.info(
            "MSSD60EHB motor backend ready: port=%s slave=%d baud=%d percent_limit=%d max_target=%d signs(left=%d,right=%d)",
            MOTOR_RS485_PORT,
            MOTOR_RS485_SLAVE_ID,
            MOTOR_RS485_BAUDRATE,
            MOTOR_PERCENT_LIMIT,
            MOTOR_RS485_MAX_TARGET,
            MOTOR_LEFT_SIGN,
            MOTOR_RIGHT_SIGN,
        )
        return driver

    def _percent_to_mssd_target(self, percent: int) -> int:
        return round(MOTOR_RS485_MAX_TARGET * self._clip_motor_percent(percent) / 100.0)

    def _wheel_state_to_mssd_target(self, wheel: str, percent: int, state: int) -> int:
        target = self._percent_to_mssd_target(percent)
        state = int(state) & 0xFF
        if state == 0x01:
            raw = target
        elif state == 0x02:
            raw = -target
        else:
            raw = 0
        sign = MOTOR_LEFT_SIGN if wheel == "left" else MOTOR_RIGHT_SIGN
        return int(raw * sign)

    def _send_mssd_targets(self, left_target: int, right_target: int, label: str) -> None:
        driver = self._ensure_mssd_driver()
        driver.set_right_target(int(right_target))
        driver.set_left_target(int(left_target))
        logger.info("MSSD command %s left=%d right=%d", label, int(left_target), int(right_target))

    def _send_mssd_diff(self, m1_percent: int, m1_state: int, m2_percent: int, m2_state: int, label: str) -> None:
        if M1_IS_LEFT_WHEEL:
            left_target = self._wheel_state_to_mssd_target("left", m1_percent, m1_state)
            right_target = self._wheel_state_to_mssd_target("right", m2_percent, m2_state)
        else:
            right_target = self._wheel_state_to_mssd_target("right", m1_percent, m1_state)
            left_target = self._wheel_state_to_mssd_target("left", m2_percent, m2_state)
        self._send_mssd_targets(left_target, right_target, label)

    def _send_mssd_stop(self, label: str = "stop") -> None:
        driver = self._ensure_mssd_driver()
        if MOTOR_RS485_STOP_MODE == "emergency":
            driver.emergency_stop_both_motors()
        elif MOTOR_RS485_STOP_MODE == "free":
            driver.free_stop_both_motors()
        else:
            driver.normal_stop_both_motors()
        logger.info("MSSD stop %s mode=%s", label, MOTOR_RS485_STOP_MODE)

    def _close_mssd_driver(self) -> None:
        if self._mssd_driver is None and self._mssd_driver_ctx is None:
            return
        try:
            if self._mssd_driver is not None:
                self._send_mssd_stop("close")
        except Exception as exc:
            logger.warning("MSSD stop on close failed: %s", exc)
        try:
            if self._mssd_driver_ctx is not None and hasattr(self._mssd_driver_ctx, "__exit__"):
                self._mssd_driver_ctx.__exit__(None, None, None)
            elif self._mssd_driver is not None and hasattr(self._mssd_driver, "close"):
                self._mssd_driver.close()
        finally:
            self._mssd_driver = None
            self._mssd_driver_ctx = None

    def _send_percent_drive(self, percent: int):
        """
        百分比调速：写 0x0002(M1) / 0x0003(M2)。
        doc: 高8位=速度百分比(0~100)，低8位=运行状态(00 stop, 01 forward, 02 back, 03 brake)
        前进态(01)时百分比按协议钳制到 [MIN_FORWARD_PERCENT, 100]；0 表示 stop(00)，非“1~9% 前进”。
        """
        p = self._clip_motor_percent(percent)
        if p <= 0:
            state = 0x00  # stop (free stop)
        else:
            p = max(min(MIN_FORWARD_PERCENT, MOTOR_PERCENT_LIMIT), p)
            state = 0x01  # forward
        with self.motor_io_lock:
            self._send_mssd_diff(p, state, p, state, "DRIVE")

    def _send_percent_diff(self, m1_percent: int, m1_state: int, m2_percent: int, m2_state: int, label: str):
        """差速控制：分别给 M1/M2 写 percent+state（01 forward, 02 back, 00 stop, 03 brake）。"""
        p1 = self._clip_motor_percent(m1_percent)
        p2 = self._clip_motor_percent(m2_percent)
        with self.motor_io_lock:
            self._send_mssd_diff(p1, m1_state, p2, m2_state, label)

    def _send_percent_brake(self):
        """
        百分比通道“锁轮刹车”：state=0x03(brake)，percent=0。
        用于紧急停、旋转脉冲结束、以及 brake 保持态周期补发。
        """
        with self.motor_io_lock:
            self._send_mssd_stop("brake")

    def _brake_lock_after_rotate_pulse(self) -> None:
        """
        单次旋转脉冲结束（约 ROTATE_DURATION）：强制 percent brake(state=03) 并进入 brake 保持态。
        保持态下动作线程会周期补发刹车（见 BRAKE_HOLD_REFRESH_INTERVAL_SEC），避免坡上溜车。
        下一帧若收到左/右转或有效前进，由 _can_release_brake_hold 解除保持。
        """
        self._use_soft_stop_next = False
        if USE_PERCENT_SPEED:
            try:
                self._send_percent_brake()
            except Exception as e:
                logger.warning("旋转脉冲结束 brake 失败，退回 ACTION_STOP: %s", e)
                try:
                    self._send_robot_command(ACTION_STOP)
                except Exception as e2:
                    logger.warning("后备 STOP 失败: %s", e2)
        else:
            self._send_robot_command(ACTION_STOP)
        self._brake_hold_active = True
        self._last_brake_hold_send_ts = 0.0  # 下一循环立刻补发一次 brake

    def _send_robot_command(self, action: int):
        """Send a high-level action through the RK3588 MSSD RS485 motor backend."""
        if action == ACTION_FORWARD:
            percent = getattr(self, "_current_forward_percent", 0)
            try:
                self._send_percent_drive(percent)
            except Exception as e:
                logger.warning("百分比调速发送失败: %s", e)
            return

        if action in (ACTION_STEER_LEFT, ACTION_STEER_RIGHT):
            base = max(0, min(MAX_FORWARD_PERCENT, int(getattr(self, "_current_steer_base_percent", 0))))
            inner_ratio = max(
                0,
                min(100, int(getattr(self, "_current_steer_inner_ratio_percent", VISIBLE_STEER_INNER_RATIO_PERCENT))),
            )
            outer_ratio = max(
                0,
                min(150, int(getattr(self, "_current_steer_outer_ratio_percent", VISIBLE_STEER_OUTER_RATIO_PERCENT))),
            )
            if base <= 0:
                try:
                    self._send_percent_drive(0)
                except Exception as e:
                    logger.warning("轮差前进零速停止失败: %s", e)
                return
            inner = int(math.floor(base * inner_ratio / 100.0))
            outer = int(math.ceil(base * outer_ratio / 100.0))
            inner = max(MIN_FORWARD_PERCENT, min(MAX_FORWARD_PERCENT, inner))
            outer = max(MIN_FORWARD_PERCENT, min(MAX_FORWARD_PERCENT, outer))
            FWD = 0x01
            if action == ACTION_STEER_LEFT:
                left_percent, right_percent = inner, outer
            else:
                left_percent, right_percent = outer, inner

            if M1_IS_LEFT_WHEEL:
                m1_percent, m1_state = left_percent, FWD
                m2_percent, m2_state = right_percent, FWD
            else:
                m1_percent, m1_state = right_percent, FWD
                m2_percent, m2_state = left_percent, FWD

            try:
                self._send_percent_diff(m1_percent, m1_state, m2_percent, m2_state, label="STEER")
            except Exception as e:
                logger.warning("轮差前进发送失败: %s", e)
            return

        if action in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT):
            # 差速旋转：一侧正转一侧反转（力度由动作线程按上一动是否旋转写入 _current_rotate_turn_percent）
            p = max(
                0,
                min(
                    100,
                    int(
                        getattr(
                            self,
                            "_current_rotate_turn_percent",
                            ROTATE_TURN_PERCENT_FROM_FORWARD,
                        )
                    ),
                ),
            )
            FWD = 0x01
            BACK = 0x02
            if action == ACTION_ROTATE_LEFT:
                # 左转：左轮后退，右轮前进
                left_state, right_state = BACK, FWD
            else:
                # 右转：左轮前进，右轮后退
                left_state, right_state = FWD, BACK

            if M1_IS_LEFT_WHEEL:
                m1_percent, m1_state = p, left_state
                m2_percent, m2_state = p, right_state
            else:
                # M1/M2 对调
                m1_percent, m1_state = p, right_state
                m2_percent, m2_state = p, left_state

            try:
                self._send_percent_diff(m1_percent, m1_state, m2_percent, m2_state, label="TURN")
            except Exception as e:
                logger.warning("差速旋转发送失败: %s", e)
            else:
                return
            return

        if action == ACTION_STOP:
            # 若 _use_soft_stop_next：0%% 滑行停；否则 brake。旋转脉冲结束不走此处，用 _brake_lock_after_rotate_pulse。
            if getattr(self, "_use_soft_stop_next", False):
                self._use_soft_stop_next = False
                try:
                    self._send_percent_drive(0)
                except Exception as e:
                    logger.warning("转向结束软停失败: %s", e)
                return
            try:
                self._send_percent_brake()
            except Exception as e:
                logger.warning("百分比刹车发送失败: %s", e)
            return

        logger.warning("未知动作类型（仅支持百分比通道）: %s", action)
   
    def _send_stop_with_brake_hold(self, reason: str = "") -> None:
        """Send brake once and keep refreshing brake until a movement command releases it."""
        hold_brake = reason not in ("target_distance_reached", "search_to_follow")
        self._use_soft_stop_next = False
        self._brake_hold_active = hold_brake
        self._last_brake_hold_send_ts = 0.0
        self.is_forwarding = False
        self._current_forward_percent = 0
        self._current_steer_base_percent = 0
        if reason:
            logger.info("进入 brake 保持态: %s", reason)
        self._send_robot_command(ACTION_STOP)

    def _get_obstacle_status(self) -> dict:
        """
        读取红外传感器状态（使用 IR HAL）
       
        Returns:
            dict: {"front": bool, "left": bool, "right": bool}
                  True表示有障碍物，False表示无障碍物
        """
        if not MODULE_IR_ENABLE or IR is None:
            return {"front": False, "left": False, "right": False}
        front = IR.is_triggered(IR.IDX_1)
        if not SIDE_IR_BLOCKS_ROTATION:
            return {"front": front, "left": False, "right": False}
        return {
            "front": front,  # IR1 = 前方
            "left": IR.is_triggered(IR.IDX_2),   # IR2 = 左侧
            "right": IR.is_triggered(IR.IDX_0)   # IR0 = 右侧
        }

    def _trigger_bunker_safety_stop(self, reason: str) -> None:
        """清空待执行动作，并对沙坑/水坑视觉危险触发刹车停止。"""
        self._explicit_stop_requested = True
        self.stop_action_execution = True
        self.is_forwarding = False
        self._current_forward_percent = 0
        self._use_soft_stop_next = False
        self._brake_hold_active = True
        self._last_brake_hold_send_ts = 0.0
        try:
            while True:
                self.action_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._send_robot_command(ACTION_STOP)
        except Exception as e:
            logger.warning("沙坑/水坑安全停止发送失败: %s, reason=%s", e, reason)

    def _get_split_bunker_stop_state(self):
        if not BUNKER_AVOID_ENABLE or BUNKER_DETECT_MODE != "split" or self._bunker_monitor is None:
            return None
        state = self._bunker_monitor.get_state()
        if not state.active:
            self._bunker_active_frames = 0
            return None
        if state.updated_ts != self._last_bunker_state_ts:
            self._last_bunker_state_ts = state.updated_ts
            self._bunker_active_frames += 1
        if self._bunker_active_frames < max(1, BUNKER_STOP_CONSEC_FRAMES):
            return None
        return state

    def _check_bunker_safety_from_split_model(self) -> bool:
        state = self._get_split_bunker_stop_state()
        if state is None:
            return False

        now = time.time()
        if now - self._last_bunker_stop_log_ts >= 0.5:
            self._last_bunker_stop_log_ts = now
            logger.warning(
                "独立沙坑/水坑模型触发安全停止: %s(class_id=%s, score=%.3f, area_ratio=%.4f >= %.4f, bbox=%s)",
                state.class_name,
                state.class_id,
                state.score,
                state.area_ratio,
                BUNKER_STOP_AREA_RATIO,
                [round(float(v), 1) for v in state.bbox],
            )
        self._trigger_bunker_safety_stop(state.reason())
        return True

    def _check_bunker_safety_from_rknn_frame(self, frame, frame_format: str = "BGR") -> bool:
        if (
            not BUNKER_AVOID_ENABLE
            or BUNKER_DETECT_MODE != "split"
            or self._bunker_rknn_detector is None
        ):
            return False

        try:
            state = self._bunker_rknn_detector.detect_state(frame, frame_format)
        except Exception as e:
            now = time.time()
            if now - self._last_bunker_stop_log_ts >= 2.0:
                self._last_bunker_stop_log_ts = now
                logger.warning("沙坑/水坑 split RKNN 检测失败: %s", e)
            return False

        if not state.active:
            self._bunker_active_frames = 0
            return False
        if state.updated_ts != self._last_bunker_state_ts:
            self._last_bunker_state_ts = state.updated_ts
            self._bunker_active_frames += 1
        if self._bunker_active_frames < max(1, BUNKER_STOP_CONSEC_FRAMES):
            return False

        timing = getattr(self._bunker_rknn_detector, "last_timing_ms", {})
        now = time.time()
        if now - self._last_bunker_stop_log_ts >= 0.5:
            self._last_bunker_stop_log_ts = now
            logger.warning(
                "独立沙坑/水坑 RKNN 模型触发安全停止: %s(class_id=%s, score=%.3f, area_ratio=%.4f >= %.4f, bbox=%s, infer=%.1fms)",
                state.class_name,
                state.class_id,
                state.score,
                state.area_ratio,
                BUNKER_STOP_AREA_RATIO,
                [round(float(v), 1) for v in state.bbox],
                float(timing.get("total", 0.0)),
            )
        self._trigger_bunker_safety_stop(state.reason())
        return True

    def _check_bunker_safety_from_dets(self, dets: List[dict], frame_area: float) -> bool:
        """merged 单模型模式：从主 YOLO 输出里按危险类别面积占比判断是否停止。"""
        if not BUNKER_AVOID_ENABLE or BUNKER_DETECT_MODE != "merged" or not BUNKER_STOP_CLASS_IDS:
            return False
        if frame_area <= 0:
            return False

        state = check_hazard_from_dets(
            dets=dets,
            frame_area=frame_area,
            class_ids=tuple(BUNKER_STOP_CLASS_IDS),
            score_threshold=BUNKER_STOP_SCORE_THRESHOLD,
            area_ratio_stop=BUNKER_STOP_AREA_RATIO,
            class_names=BUNKER_CLASS_NAMES,
            source="merged",
        )
        if not state.active:
            return False

        logger.warning(
            "检测到安全危险类别 %s(class_id=%s, score=%.3f, area_ratio=%.4f >= %.4f, bbox=%s)，立即停止",
            state.class_name,
            state.class_id,
            state.score,
            state.area_ratio,
            BUNKER_STOP_AREA_RATIO,
            [round(float(v), 1) for v in state.bbox],
        )
        self._trigger_bunker_safety_stop(state.reason())
        return True
   
    def _forward_percent_for_distance(self, distance: float) -> int:
        """
        根据距离返回前进速度百分比（0 或 MIN_FORWARD_PERCENT~MAX_FORWARD_PERCENT）。
        返回 0 表示本帧不要发前进（由主逻辑停/等），不是让电机用 1~9% 蠕行；真正前进时不低于 MIN_FORWARD_PERCENT。
        """
        # < FOLLOW_BRAKE_DISTANCE_M：由主逻辑 brake，这里返回 0 兜底
        if distance < FOLLOW_BRAKE_DISTANCE_M:
            return 0
        # d <= TARGET_DISTANCE：已够近，不追近
        if distance <= TARGET_DISTANCE:
            return 0

        # 仅当 distance > TARGET_DISTANCE(1.0m) 时调用本函数：
        # (1.0, 255] → 10% ；更远保持原高档分档
        if distance <= 1.3:
            p = FORWARD_SPEED_LE_1_3_PERCENT
        elif distance <= 1.7:
            p = FORWARD_SPEED_LE_1_7_PERCENT
        elif distance <= 2.1:
            p = FORWARD_SPEED_LE_2_1_PERCENT
        elif distance <= 2.6:
            p = FORWARD_SPEED_LE_2_6_PERCENT
        elif distance <= 3.2:
            p = FORWARD_SPEED_LE_3_2_PERCENT
        elif distance <= 3.8:
            p = FORWARD_SPEED_LE_3_8_PERCENT
        elif distance <= 4.5:
            p = FORWARD_SPEED_LE_4_5_PERCENT
        else:
            p = FORWARD_SPEED_FAR_PERCENT
        p = min(MAX_FORWARD_PERCENT, int(p))
        if p > 0:
            p = max(MIN_FORWARD_PERCENT, p)
        return max(0, p)

    def _get_distance(self) -> Optional[float]:
        """
        获取距离传感器距离（按 DISTANCE_SOURCE 选择毫米波/超声，HAL 返回厘米，这里转换为米）
       
        Returns:
            Optional[float]: 距离（米），如果无效则返回None
        """
        source = DISTANCE_SOURCE
        if source == "auto":
            if MODULE_MMWAVE_ENABLE:
                source = "mmwave"
            elif MODULE_ULTRASONIC_ENABLE:
                source = "ultrasonic"
            else:
                return None
        if source in ("none", "off", "disabled"):
            return None
        if source in VISION_MMWAVE_SOURCE_ALIASES:
            return None
        if source not in ("mmwave", "ultrasonic"):
            logger.warning("未知距离来源 DISTANCE_SOURCE=%s，跳过距离读取", source)
            return None
        if source == "mmwave" and not MODULE_MMWAVE_ENABLE:
            return None
        if source == "ultrasonic" and not MODULE_ULTRASONIC_ENABLE:
            return None
        if source == "mmwave" and MmWaveRadar is None:
            logger.warning("毫米波雷达 HAL 不可用，跳过距离读取: %s", _MMWAVE_IMPORT_ERROR)
            return None
        try:
            if source == "ultrasonic":
                if self._Utrasonic is None:
                    return None
                distance_cm = self._Utrasonic.get_distance()
            else:
                distance_cm = MmWaveRadar.get_distance()
            if distance_cm is None or distance_cm <= 0:
                return None
            distance_m = distance_cm / 100.0  # 转换为米
            # 过滤无效值（有效范围0.02-8.0m，即2-800cm）
            if 0.02 < distance_m < 8.0:
                return distance_m
            else:
                return None
        except Exception as e:
            logger.warning("距离传感器读取失败(source=%s): %s", source, e)
            return None
   
    def _select_distance_target(self, targets: List[PersonTarget]) -> Optional[PersonTarget]:
        if not targets:
            return None
        return max(targets, key=lambda p: p.area)

    def _pixel_x_to_angle_deg(self, x: float, width: int) -> float:
        if width <= 0:
            return 0.0
        norm = (float(x) / float(width)) - 0.5
        return norm * VISION_HFOV_DEG * VISION_MMWAVE_ANGLE_SIGN + VISION_MMWAVE_ANGLE_OFFSET_DEG

    def _compensate_vision_mmwave_distance_m(self, distance_m: float) -> float:
        return max(VISION_MMWAVE_MIN_OUTPUT_DISTANCE_M, float(distance_m) - VISION_MMWAVE_DISTANCE_BIAS_M)

    def _get_recent_vision_mmwave_distance(self) -> Optional[float]:
        distance_m = self._last_vision_mmwave_distance_m
        if distance_m is None:
            return None
        if time.monotonic() - self._last_vision_mmwave_ts > VISION_MMWAVE_HARD_STOP_TTL_SEC:
            return None
        return distance_m

    def _log_vision_mmwave_match(
        self,
        reason: str,
        target: PersonTarget,
        center_angle: float,
        angle_range: Tuple[float, float],
        raw_targets: List[Dict[str, Any]],
        selected: Optional[Dict[str, Any]],
    ) -> None:
        selected_idx = None if selected is None else selected.get("index")
        selected_dist = None if selected is None else selected.get("distance_m")
        log_key = (
            reason,
            selected_idx,
            None if selected_dist is None else round(float(selected_dist), 2),
            len(raw_targets),
        )
        should_log = log_key != self._last_vision_mmwave_log_key
        if VISION_MMWAVE_LOG_EVERY_FRAMES > 0 and self.frame_index % VISION_MMWAVE_LOG_EVERY_FRAMES == 0:
            should_log = True
        if not should_log:
            return

        cx, cy = target.center
        raw_dbg = []
        for item in raw_targets:
            try:
                raw_dbg.append(
                    "%s:%.1fdeg/%.2fm" % (
                        item.get("index"),
                        float(item.get("angle")),
                        float(item.get("distance")),
                    )
                )
            except Exception:
                raw_dbg.append(str(item))
        selected_dbg = "none"
        if selected is not None:
            selected_dbg = "%s:%.1fdeg raw=%.2fm used=%.2fm" % (
                selected.get("index"),
                float(selected.get("angle")),
                float(selected.get("raw_distance_m")),
                float(selected.get("distance_m")),
            )
        logger.info(
            "vision_mmwave: frame=%d reason=%s person_center=(%.1f,%.1f) angle=%.1f range=[%.1f,%.1f] targets=%s selected=%s",
            self.frame_index,
            reason,
            float(cx),
            float(cy),
            float(center_angle),
            float(angle_range[0]),
            float(angle_range[1]),
            raw_dbg,
            selected_dbg,
        )
        self._last_vision_mmwave_log_key = log_key

    def _get_vision_mmwave_distance(self, width: int, target: Optional[PersonTarget]) -> Optional[float]:
        if target is None:
            return None
        if not MODULE_MMWAVE_ENABLE or MmWaveRadar is None or width <= 0:
            self._last_vision_mmwave_distance_m = None
            self._last_vision_mmwave_ts = 0.0
            return None

        try:
            raw_targets = MmWaveRadar.get_targets()
        except Exception as e:
            logger.warning("vision_mmwave read targets failed: %s", e)
            self._last_vision_mmwave_distance_m = None
            self._last_vision_mmwave_ts = 0.0
            return None

        x1, _y1, x2, _y2 = target.bbox
        center_x, _center_y = target.center
        left_angle = self._pixel_x_to_angle_deg(max(0.0, min(float(width), float(x1))), width)
        right_angle = self._pixel_x_to_angle_deg(max(0.0, min(float(width), float(x2))), width)
        center_angle = self._pixel_x_to_angle_deg(center_x, width)
        angle_lo, angle_hi = sorted((left_angle, right_angle))
        angle_lo -= VISION_MMWAVE_ANGLE_MARGIN_DEG
        angle_hi += VISION_MMWAVE_ANGLE_MARGIN_DEG

        candidates: List[Dict[str, Any]] = []
        for item in raw_targets:
            try:
                angle = float(item.get("angle"))
                distance_m = float(item.get("distance"))
            except Exception:
                continue
            if not (math.isfinite(angle) and math.isfinite(distance_m)):
                continue
            if not (VISION_MMWAVE_MIN_DISTANCE_M < distance_m < VISION_MMWAVE_MAX_DISTANCE_M):
                continue
            if not (angle_lo <= angle <= angle_hi):
                continue
            candidates.append(
                {
                    "index": int(item.get("index", -1)),
                    "angle": angle,
                    "raw_distance_m": distance_m,
                    "distance_m": self._compensate_vision_mmwave_distance_m(distance_m),
                    "target": item,
                }
            )

        selected: Optional[Dict[str, Any]]
        if not candidates:
            selected = None
            self._last_vision_mmwave_distance_m = None
            self._last_vision_mmwave_ts = 0.0
            self._log_vision_mmwave_match(
                "unmatched",
                target,
                center_angle,
                (angle_lo, angle_hi),
                raw_targets,
                selected,
            )
            return None

        mode = VISION_MMWAVE_MATCH_MODE
        if mode in ("first", "id"):
            selected = min(candidates, key=lambda item: item["index"])
        elif mode in ("angle", "center", "angle_closest"):
            selected = min(candidates, key=lambda item: (abs(item["angle"] - center_angle), item["distance_m"]))
        else:
            selected = min(candidates, key=lambda item: (item["distance_m"], abs(item["angle"] - center_angle)))

        distance_m = float(selected["distance_m"])
        self._last_vision_mmwave_distance_m = distance_m
        self._last_vision_mmwave_ts = time.monotonic()
        self._log_vision_mmwave_match(
            "matched",
            target,
            center_angle,
            (angle_lo, angle_hi),
            raw_targets,
            selected,
        )
        return distance_m

    def _get_frame_distance(self, width: int, person_targets: List[PersonTarget]) -> Optional[float]:
        if DISTANCE_SOURCE in VISION_MMWAVE_SOURCE_ALIASES:
            target = self._select_distance_target(person_targets)
            return self._get_vision_mmwave_distance(width, target)
        return self._get_distance()

    def _check_obstacle_for_action(self, action: int) -> bool:
        """
        检查指定动作是否会撞到障碍物
       
        Args:
            action: 动作类型
           
        Returns:
            bool: True表示会撞到障碍物，False表示安全
        """
        obstacles = self._get_obstacle_status()
       
        if action in (ACTION_FORWARD, ACTION_STEER_LEFT, ACTION_STEER_RIGHT):
            return obstacles["front"]
        elif action == ACTION_ROTATE_LEFT:
            return obstacles["left"]
        elif action == ACTION_ROTATE_RIGHT:
            return obstacles["right"]
        else:  # ACTION_STOP
            return False
   
    def _get_person_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """获取人员边界框的中心点"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y
   
    def _is_in_center_3x3(self, center_x: float, center_y: float,
                          img_width: int, img_height: int) -> bool:
        """
        判断目标是否在中间3x3区域（6x6网格的第2-4列，第2-4行）
       
        Args:
            center_x: 中心点x坐标
            center_y: 中心点y坐标
            img_width: 图像宽度
            img_height: 图像高度
           
        Returns:
            bool: True表示在中间3x3区域
        """
        grid_w = img_width / 6
        grid_h = img_height / 6
       
        # 中间3列：第2-4列（索引1-3），即 grid_w*1 到 grid_w*4
        in_center_cols = grid_w * 1 <= center_x < grid_w * 4
        # 中间3行：第2-4行（索引1-3），即 grid_h*1 到 grid_h*4
        in_center_rows = grid_h * 1 <= center_y < grid_h * 4
       
        return in_center_cols and in_center_rows
   
    def _get_edge_type(self, center_x: float, center_y: float,
                       img_width: int, img_height: int) -> str:
        """获取目标在哪个边缘（只判断左右）：最左两列→'left'，最右两列→'right'，否则'none'"""
        grid_w = img_width / 6

        if center_x < grid_w * 2:
            return 'left'   # 最左两列（列0、列1）触发左转
        elif center_x >= grid_w * 4:
            return 'right'  # 最右两列（列4、列5）触发右转
        else:
            return 'none'
   
    def _calculate_rotation_actions(self, edge_type: str) -> List[int]:
        """根据边缘类型计算需要旋转的动作"""
        if edge_type == 'right':
            return [ACTION_ROTATE_RIGHT]
        elif edge_type == 'left':
            return [ACTION_ROTATE_LEFT]
        else:
            return []

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
        state = self._get_split_bunker_stop_state()
        if state is None:
            return HazardState()
        return HazardState(
            active=True,
            reason=state.reason(),
            class_id=int(state.class_id),
            score=float(state.score),
            area_ratio=float(state.area_ratio),
        )

    def _process_detections_modular(self, width: int, height: int, persons: List[Tuple]) -> List[int]:
        """Use car_control_modular.controllers to decide actions."""
        self._explicit_stop_requested = False
        self._follow_controller.set_last_dispatched(self._action_int_to_kind(self._last_dispatched_action))

        obstacles_dict = self._get_obstacle_status()
        person_targets = self._persons_to_targets(persons)
        distance_m = self._get_frame_distance(int(width), person_targets)
        lost_intent = "unknown"
        lost_intent_age_sec = 0.0
        if self._motion_history is not None:
            if person_targets:
                target_for_history = max(person_targets, key=lambda p: p.area)
                self._motion_history.record(self.frame_index, target_for_history, distance_m)
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
            lost_intent=lost_intent,
            lost_intent_age_sec=lost_intent_age_sec,
            module_status={
                "vision": MODULE_VISION_ENABLE,
                "ir": MODULE_IR_ENABLE,
                "mmwave": MODULE_MMWAVE_ENABLE,
                "ultrasonic": MODULE_ULTRASONIC_ENABLE,
                "bunker": BUNKER_AVOID_ENABLE,
            },
        )
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
        self.is_forwarding = decision.is_forwarding
        self._current_forward_percent = decision.current_forward_percent
        if decision.stop_action_execution:
            self.stop_action_execution = True
        if decision.clear_action_queue:
            try:
                while True:
                    self.action_queue.get_nowait()
            except queue.Empty:
                pass

        actions: List[int] = []
        for action in decision.actions:
            if action.kind == "forward":
                self._current_forward_percent = int(action.speed_percent)
                self.is_forwarding = True
            elif action.kind in ("steer_left", "steer_right"):
                self._current_steer_base_percent = int(action.speed_percent)
                self._current_steer_inner_ratio_percent = int(action.steer_inner_ratio_percent)
                self._current_steer_outer_ratio_percent = int(action.steer_outer_ratio_percent)
                self._current_forward_percent = int(action.speed_percent)
                self.is_forwarding = True
            action_int = self._action_kind_to_int(action.kind)
            if action_int is not None and action.kind != "idle":
                actions.append(action_int)

        if decision.reason:
            action_kinds = [a.kind for a in decision.actions]
            target_dbg = max(frame.persons, key=lambda p: p.area) if frame.persons else None
            target_center = None
            if target_dbg is not None:
                tx, ty = target_dbg.center
                target_center = (round(float(tx), 1), round(float(ty), 1))
            distance_dbg = "none" if frame.distance_m is None else "%.2fm" % float(frame.distance_m)
            log_key = (
                decision.reason,
                tuple(action_kinds),
                bool(decision.explicit_stop_requested),
                len(frame.persons),
                self.search_state,
                self.search_direction,
            )
            if action_kinds or decision.explicit_stop_requested or log_key != self._last_control_decision_log_key:
                logger.info(
                    "control decision: frame=%d reason=%s actions=%s stop=%s speed=%s persons=%d center=%s distance=%s obstacles(front=%s,left=%s,right=%s) search=%s/%s",
                    self.frame_index,
                    decision.reason,
                    action_kinds,
                    decision.explicit_stop_requested,
                    self._current_forward_percent,
                    len(frame.persons),
                    target_center,
                    distance_dbg,
                    frame.obstacles.front,
                    frame.obstacles.left,
                    frame.obstacles.right,
                    self.search_state,
                    self.search_direction,
                )
                self._last_control_decision_log_key = log_key
        return actions
   
    def _process_detections(self, width: int, height: int, persons: List[Tuple]) -> List[int]:
        """处理检测结果，生成动作序列（集成避障逻辑）。persons: [((x1,y1,x2,y2), track_id, conf, area), ...]"""
        return self._process_detections_modular(width, height, persons)
        actions = []
        self._explicit_stop_requested = False  # 仅在有明确原因（障碍、人在停止距离内等）时为 True 才发 STOP
       
        # 如果没有检测到人
        self.person_detected_flag = False  # 重置检测标志
       
        if not persons:
            # 防抖：连续 LOST_CONFIRM_FRAMES 帧没有检测到人才确认丢失并开始搜索
            self.lost_confirm_frames += 1
           
            if self.lost_confirm_frames < LOST_CONFIRM_FRAMES:
                # 5 帧内未确认丢失，保持原运动，不发送停止
                self._waiting_lost_confirm = True
                logger.debug(f"未检测到人，等待确认 ({self.lost_confirm_frames}/{LOST_CONFIRM_FRAMES}帧)，保持原运动")
                return []
           
            # 已经确认丢失（连续LOST_CONFIRM_FRAMES帧没有检测到人）
            self._waiting_lost_confirm = False
            if self.lost_confirm_frames == LOST_CONFIRM_FRAMES:
                logger.info(f"未检测到人（连续{self.lost_confirm_frames}帧），开始搜索")
                # 重置计数（避免重复日志）
                self.lost_confirm_frames = LOST_CONFIRM_FRAMES + 1
           
            # 进入搜索状态
            if self.search_state != "searching":
                if self.last_person_center_x is not None:
                    img_center_x = width / 2
                    if self.last_person_center_x < img_center_x:
                        self.search_direction = 'left'
                        logger.info(f"最后位置在左边，开始左旋转寻找")
                    else:
                        self.search_direction = 'right'
                        logger.info(f"最后位置在右边，开始右旋转寻找")
                else:
                    self.search_direction = 'right'
                    logger.info("没有历史位置信息，默认右旋转寻找")
                self.search_state = "searching"
           
            # 在搜索状态，继续旋转（有障碍的方向不转，两边都有障则停止）
            frames_since_last_action = self.frame_index - self.last_action_frame if self.last_action_frame >= 0 else 999
            if frames_since_last_action >= self.search_cooldown:
                obstacles = self._get_obstacle_status()
                if self.search_direction == 'left':
                    if obstacles["left"]:
                        if not obstacles["right"]:
                            self.search_direction = 'right'
                            actions.append(ACTION_ROTATE_RIGHT)
                        else:
                            logger.warning("搜索时左右都有障碍物，不旋转，停止")
                            self._explicit_stop_requested = True
                            return []
                    else:
                        actions.append(ACTION_ROTATE_LEFT)
                else:  # right
                    if obstacles["right"]:
                        if not obstacles["left"]:
                            self.search_direction = 'left'
                            actions.append(ACTION_ROTATE_LEFT)
                        else:
                            logger.warning("搜索时左右都有障碍物，不旋转，停止")
                            self._explicit_stop_requested = True
                            return []
                    else:
                        actions.append(ACTION_ROTATE_RIGHT)
                if actions:
                    self.last_action_frame = self.frame_index
                    return actions
            else:
                # 搜索冷却中：若上一段是旋转脉冲，才显式 STOP，打断旋转；若仍是前进（漏检前几帧未确认丢失），
                # 不要发 STOP，避免误刹断前进。
                if self._last_dispatched_action in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT):
                    self._explicit_stop_requested = True
                return []
       
        # 检测到人了！
        # 如果之前在搜索状态，立即退出搜索状态并停止当前动作
        if self.search_state == "searching":
            logger.info("搜索中发现人，立即停止搜索，开始跟踪")
            self.search_state = "none"
            self.search_direction = None
            # 设置标志，用于动作执行线程立即停止当前搜索动作
            self.person_detected_flag = True
            # 清空动作队列并停止当前动作
            while not self.action_queue.empty():
                try:
                    self.action_queue.get_nowait()
                except queue.Empty:
                    break
            self.stop_action_execution = True
        else:
            # 如果人本来就在视野中（非搜索状态），不需要停止，重置标志
            self.person_detected_flag = False
       
        # 重置丢失确认计数与标志（因为检测到人了）
        self.lost_confirm_frames = 0
        self._waiting_lost_confirm = False
        # 重置前进状态（检测到人后重新判断）
        self.is_forwarding = False
        # 重置前进1米状态（检测到人后重新判断）
        self.forward_1m_state = None
        self.forward_1m_start_distance = None
       
        # 人数已由 FirstPersonTracker 限定为 0 或 1（首人跟踪）
        # 检查当前唯一目标（若有）
        person_in_center_3x3 = None
        person_need_rotate = None
       
        for bbox, track_id, conf, area in persons:
            center_x, center_y = self._get_person_center(bbox)
           
            # 记录最后检测到的人的center_x位置
            self.last_person_center_x = center_x
           
            in_center_3x3 = self._is_in_center_3x3(center_x, center_y, width, height)
           
            if in_center_3x3:
                # 在中间3x3区域，使用毫米波距离
                person_in_center_3x3 = (bbox, track_id, conf, center_x, center_y)
            else:
                # 不在中心，需要旋转调整
                edge_type = self._get_edge_type(center_x, center_y, width, height)
                person_need_rotate = (bbox, track_id, conf, center_x, center_y, edge_type)
       
        # 动作冷却仅用于旋转，避免旋转过频；前进不受冷却限制，每帧可投递以保持流畅
        frames_since_last_action = self.frame_index - self.last_action_frame if self.last_action_frame >= 0 else 999
       
        # ========== 避障优先级最高：检查毫米波 ==========
        # 毫米波检查只在不检测到人时进行，检测到人时在距离调整部分统一处理
        if person_in_center_3x3 is None and person_need_rotate is None:
            # 没有检测到人时，检查毫米波1m内障碍物
            distance_sensor_distance = self._get_distance()
            if distance_sensor_distance is not None and distance_sensor_distance < MMWAVE_OBSTACLE_THRESHOLD:
                logger.warning(f"毫米波检测到近距离障碍物（{distance_sensor_distance:.2f}m < {MMWAVE_OBSTACLE_THRESHOLD}m），且没有检测到人，停止")
                self._explicit_stop_requested = True
                return []
       
        # ========== 避障检查：前方 IR 有障时仅停，不绕开（不做避障旋转） ==========
        obstacles = self._get_obstacle_status()
        if obstacles["front"]:
            logger.warning("前方 IR 触发：不绕开，原地停止等待")
            self._explicit_stop_requested = True
            return []

        # ========== 位置调整：人不在中间 3×3 时旋转对准中心（前方 IR 已清，允许转） ==========
        # 与「前方有障绕开」不同：这里是把人从画面左/右缘旋回中间，仅在 obstacles["front"] 为 False 时才会走到这里。
        if person_need_rotate is not None:
            # 1米内（与 FOLLOW_BRAKE_DISTANCE_M 一致）直接停下，不再旋转
            distance_near = self._get_distance()
            if distance_near is not None and distance_near < FOLLOW_BRAKE_DISTANCE_M:
                self.is_forwarding = False
                logger.info(
                    f"跟的人太近（{distance_near:.2f}m < {FOLLOW_BRAKE_DISTANCE_M}m），直接停下，不旋转"
                )
                self._explicit_stop_requested = True
                return []
            bbox, track_id, conf, center_x, center_y, edge_type = person_need_rotate
            rotation_actions = self._calculate_rotation_actions(edge_type)
            if rotation_actions:
                action = rotation_actions[0]
                obstacles = self._get_obstacle_status()
                if self._check_obstacle_for_action(action):
                    # 该旋转方向有障碍物，尝试另一侧；若两侧都有障则不转，停止
                    if edge_type == 'left':
                        if not obstacles["right"]:
                            action = ACTION_ROTATE_RIGHT
                            logger.warning("左侧有障碍物，改为右转调整")
                        else:
                            logger.warning("左右都有障碍物，不旋转，停止")
                            self._explicit_stop_requested = True
                            return []
                    else:
                        if not obstacles["left"]:
                            action = ACTION_ROTATE_LEFT
                            logger.warning("右侧有障碍物，改为左转调整")
                        else:
                            logger.warning("左右都有障碍物，不旋转，停止")
                            self._explicit_stop_requested = True
                            return []
                actions.append(action)
                if frames_since_last_action < self.action_cooldown:
                    # 旋转冷却期间也必须停住：否则动作线程可能继续沿用上一条前进指令，导致“该转向调整却还在前进”
                    self._explicit_stop_requested = True
                    return []
                self.last_action_frame = self.frame_index
                edge_names = {'left': '左边', 'right': '右边'}
                edge_name = edge_names.get(edge_type, edge_type)
                action_names = {ACTION_ROTATE_LEFT: '左转', ACTION_ROTATE_RIGHT: '右转'}
                action_name = action_names.get(action, '未知')
                logger.info(f"检测到人不在中间3x3 ({edge_name})，{action_name}一次调整")
                return actions
       
        # ========== 距离调整：如果人在中间3x3区域，使用毫米波距离控制 ==========
        if person_in_center_3x3 is not None:
            bbox, track_id, conf, center_x, center_y = person_in_center_3x3
            distance = self._get_distance()
           
            if distance is not None:
                logger.info(f"人在中间3x3区域，毫米波距离: {distance:.2f}m")
               
                # 距离控制：<0.5m brake；[0.5,1.0] 不前进；>1.0 按分档前进
                if distance < FOLLOW_BRAKE_DISTANCE_M:
                    self.is_forwarding = False
                    logger.info(f"距离过近（{distance:.2f}m < {FOLLOW_BRAKE_DISTANCE_M}m），brake")
                    self._explicit_stop_requested = True
                    return []
                elif distance > TARGET_DISTANCE:
                    # 仅当距离 > TARGET_DISTANCE 才允许前进
                    # 前方 IR 已在上面处理：有障则已停；此处仅「人在中心且距离够远」时前进
                    self.is_forwarding = True
                    self._current_forward_percent = self._forward_percent_for_distance(distance)
                    # 避免出现“距离>1.5m但速度=0%仍发前进动作”的冲突：0% 直接视为不前进
                    if self._current_forward_percent <= 0:
                        self.is_forwarding = False
                        logger.info(f"距离: {distance:.2f}m，但速度=0%，不前进")
                        return []
                    actions.append(ACTION_FORWARD)
                    self.last_action_frame = self.frame_index
                    logger.info(f"距离: {distance:.2f}m，速度 {self._current_forward_percent}%，持续前进")
                    return actions
                else:
                    # [FOLLOW_BRAKE_DISTANCE_M, TARGET_DISTANCE]：不前进、等待（0.5～1.0m 速度视为 0）
                    self.is_forwarding = False
                    self._current_forward_percent = 0
                    logger.info(f"距离: {distance:.2f}m（<= {TARGET_DISTANCE}m），不前进")
                    return []
            else:
                # 毫米波距离无效（超出范围或没有正常值），但人在中心区域
                # 毫米波无效时：若前方 IR 未触发（前方无近距离障碍/人），允许用保守低速继续前进；否则停止
                obstacles_now = self._get_obstacle_status()
                if not obstacles_now["front"]:
                    self.is_forwarding = True
                    self._current_forward_percent = 50  # 毫米波无效但前方IR未触发时，允许较快前进
                    actions.append(ACTION_FORWARD)
                    self.last_action_frame = self.frame_index
                    logger.warning("人在中间3x3区域，但毫米波距离无效且前方IR未触发：前进(50%)")
                    return actions
                logger.warning("人在中间3x3区域，但毫米波距离无效且前方IR触发：停止等待")
                self.is_forwarding = False
                self._current_forward_percent = 0
                self._explicit_stop_requested = True
                return []
       
        # 默认情况：保持不动
        return []
   
    def _ensure_sample_reid_started(self) -> None:
        if self._vision_wrapper is None:
            raise RuntimeError("sample_reid vision wrapper is not initialized")
        if self._vision_iter is not None:
            return
        self._vision_wrapper.start()
        self._vision_iter = iter(self._vision_wrapper.iter_records())
        logger.info("sample_reid vision wrapper started")

    def _consume_track_records(self, records, width: int, height: int, source_name: str) -> None:
        all_dets = []
        person_candidates = []
        person_track_debug = []
        if VISION_TRACK_LOG_ENABLE:
            if records:
                track_parts = []
                for rec in records:
                    bbox_dbg = [round(float(v), 1) for v in (rec.x1, rec.y1, rec.x2, rec.y2)]
                    track_parts.append(
                        "track_id=%s reid_uid=%s state=%s class=%s score=%.3f area=%.0f bbox=%s"
                        % (
                            int(rec.track_id),
                            int(rec.reid_uid),
                            _tracker_state_name(int(rec.tracker_state)),
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
            area = float(rec.area)
            stable_id = int(rec.reid_uid) if int(rec.reid_uid) > 0 else int(rec.track_id)
            candidate = {
                "bbox": bbox,
                "stable_id": stable_id,
                "score": float(rec.score),
                "area": area,
                "rec": rec,
            }
            person_candidates.append(candidate)
            person_track_debug.append((area, stable_id, rec, candidate))

        track_memory_reason = "disabled"
        selected_candidates = person_candidates
        if self._track_memory is not None:
            selected_candidates, track_memory_reason = self._track_memory.select(
                self.frame_index,
                person_candidates,
            )
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
                selected_cand = (
                    selected_candidates[0]
                    if self._track_memory is not None
                    else max(selected_candidates, key=lambda cand: float(cand.get("area", 0.0)))
                )
                selected_rec = selected_cand["rec"]
                selected_id = int(selected_cand["stable_id"])
                cx = (float(selected_rec.x1) + float(selected_rec.x2)) / 2.0
                cy = (float(selected_rec.y1) + float(selected_rec.y2)) / 2.0
                logger.info(
                    "%s selected frame=%d mode=%s memory_reason=%s stable_id=%s track_id=%s reid_uid=%s state=%s score=%.3f area=%.0f center=(%.1f, %.1f)",
                    source_name,
                    self.frame_index,
                    "track_memory" if self._track_memory is not None else "largest_area",
                    track_memory_reason,
                    int(selected_id),
                    int(selected_rec.track_id),
                    int(selected_rec.reid_uid),
                    _tracker_state_name(int(selected_rec.tracker_state)),
                    float(selected_rec.score),
                    float(selected_rec.area),
                    cx,
                    cy,
                )
            elif person_track_debug:
                target_id = None if self._track_memory is None else self._track_memory.target_id
                logger.info(
                    "%s selected frame=%d mode=%s memory_reason=%s selected=none valid_persons=%d memory_target=%s",
                    source_name,
                    self.frame_index,
                    "track_memory" if self._track_memory is not None else "largest_area",
                    track_memory_reason,
                    len(person_track_debug),
                    target_id,
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

        if self._check_bunker_safety_from_dets(all_dets, float(width * height)):
            return

        self._queue_actions_for_persons(width, height, persons)

    def _queue_actions_for_persons(self, width: int, height: int, persons: List[Tuple]) -> None:
        if persons:
            bbox, track_id, conf, area = max(persons, key=lambda p: p[3])
            logger.info(
                "sample_reid persons=%d largest_candidate id=%s, conf=%.3f, area=%.0f, bbox=%s",
                len(persons),
                track_id,
                float(conf),
                float(area),
                [round(float(v), 1) for v in bbox],
            )

        was_searching_before = self.search_state in ("searching", "predictive")
        actions = self._process_detections(width, height, persons)

        if was_searching_before and self.search_state not in ("searching", "predictive"):
            self.stop_action_execution = True
            try:
                while True:
                    self.action_queue.get_nowait()
            except queue.Empty:
                pass
            self._send_stop_with_brake_hold("search_to_follow")
            logger.info("已清空搜索动作队列并停止，开始跟踪")

        if actions:
            action_names = {ACTION_FORWARD: "前进", ACTION_ROTATE_LEFT: "左转", ACTION_ROTATE_RIGHT: "右转", ACTION_STEER_LEFT: "左轮差", ACTION_STEER_RIGHT: "右轮差", ACTION_STOP: "停止"}
            logger.info("生成动作: %s", [action_names.get(a, "未知") for a in actions])
        else:
            logger.debug("未生成动作")

        if actions:
            try:
                while True:
                    self.action_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                for a in actions:
                    self.action_queue.put_nowait(a)
                self._last_dispatched_action = actions[0] if actions else None
            except queue.Full:
                pass
        else:
            if self._explicit_stop_requested:
                self.stop_action_execution = True
                try:
                    while True:
                        self.action_queue.get_nowait()
                except queue.Empty:
                    pass
                self._send_stop_with_brake_hold(self._last_explicit_stop_reason or "explicit_stop")

    def _process_frame_sample_reid(self):
        self.frame_index += 1
        if self._check_bunker_safety_from_split_model():
            return

        self._ensure_sample_reid_started()
        try:
            records = next(self._vision_iter)
        except StopIteration:
            logger.warning("sample_reid vision stream ended; restarting wrapper")
            self._vision_iter = None
            if self._vision_wrapper is not None:
                self._vision_wrapper.stop()
            return

        self._consume_track_records(records, int(VISION_FRAME_WIDTH), int(VISION_FRAME_HEIGHT), "sample_reid")

    def process_external_frame(self, frame, frame_format: str = "BGR"):
        """Process one externally supplied frame through RKNN vision and control."""
        if self._vision_engine != "rknn":
            raise RuntimeError(f"process_external_frame is only available for rknn engine, got {self._vision_engine!r}")
        if self._rknn_pipeline is None:
            raise RuntimeError("RKNN vision pipeline is not initialized")
        self.frame_index += 1
        if self._check_bunker_safety_from_split_model():
            return []
        if self._check_bunker_safety_from_rknn_frame(frame, frame_format):
            return []

        records = self._rknn_pipeline.process_frame(frame, frame_format)
        width = int(getattr(self._rknn_pipeline, "last_frame_width", VISION_FRAME_WIDTH))
        height = int(getattr(self._rknn_pipeline, "last_frame_height", VISION_FRAME_HEIGHT))
        self._consume_track_records(records, width, height, "rknn")
        return records

    def process_frame(self):
        if self._vision_engine == "rknn":
            now = time.monotonic()
            if now - self._last_rknn_no_frame_log_ts >= 5.0:
                self._last_rknn_no_frame_log_ts = now
                logger.warning("RKNN engine is decoupled from camera; call process_external_frame(frame) with BGR/RGB frames")
            return
        if self._vision_engine == "sample_reid":
            return self._process_frame_sample_reid()

        """从 HAL 取一帧、做 YOLO 检测（调用顺序与 yolov_test.py 一致）。"""
        self.frame_index += 1
        if self._check_bunker_safety_from_split_model():
            return
       
        # 与 yolov_test.py 一致：get_frame() -> cvtdl_object_t() -> detect() -> 用结果 -> stop()
        ret = Yolov.get_frame()
        if ret != 0:
            logger.warning(f"CVI_HAL_GetFrame failed with {hex(ret)}")
            return
       
        obj_meta = cvtdl_object_t()
        self._current_obj_meta = obj_meta
        Yolov.detect(obj_meta)
       
        width = int(obj_meta.width)
        height = int(obj_meta.height)
        if width <= 0 or height <= 0:
            logger.warning(f"HAL 返回无效宽高: {width}x{height}，跳过本帧")
            Yolov.stop(obj_meta)
            self._current_obj_meta = None
            return
        # 与 yolov_test 相同方式读结果：obj_meta.size, obj_meta.info[i].classes/bbox/unique_id/score
        all_dets = []
        persons = []
        for i in range(obj_meta.size):
            info = obj_meta.info[i]
            x1, y1 = float(info.bbox.x1), float(info.bbox.y1)
            x2, y2 = float(info.bbox.x2), float(info.bbox.y2)
            score = float(info.bbox.score)
            class_id = int(info.classes)
            all_dets.append({
                "class_id": class_id,
                "score": score,
                "bbox": (x1, y1, x2, y2),
            })
            if class_id != PERSON_CLASS_ID:
                continue
            if score <= CONFIDENCE_THRESHOLD:
                continue
            area = (x2 - x1) * (y2 - y1)
            persons.append(((x1, y1, x2, y2), int(info.unique_id), float(score), area))

        Yolov.stop(obj_meta)
        self._current_obj_meta = None

        if self._check_bunker_safety_from_dets(all_dets, float(width * height)):
            return

        # 最大面积优先：每帧重新选择当前画面面积最大的 person，不再锁定特定身份。
        if persons:
            persons = [max(persons, key=lambda p: p[3])]  # 跟当前画面里面积最大的人
            bbox, track_id, conf, area = persons[0]
            logger.info(
                "跟随当前画面最大面积人员: id=%s, conf=%.3f, area=%.0f, bbox=%s",
                track_id,
                float(conf),
                float(area),
                [round(float(v), 1) for v in bbox],
            )

        # 记录之前的搜索状态（在调用_process_detections之前）
        was_searching_before = self.search_state in ("searching", "predictive")

        # 生成动作
        actions = self._process_detections(width, height, persons)
       
        # 如果之前在搜索状态，现在检测到人了（search_state不再是searching），立即清空队列停止搜索
        if was_searching_before and self.search_state not in ("searching", "predictive"):
            # 从搜索状态切换到跟踪状态，立即清空动作队列
            self.stop_action_execution = True
            try:
                while True:
                    self.action_queue.get_nowait()
            except queue.Empty:
                pass
            # 先发送停止命令，停止之前的搜索旋转
            self._send_robot_command(ACTION_STOP)
            logger.info("已清空搜索动作队列并停止，开始跟踪")
       
        # 调试日志：显示生成的动作
        if actions:
            action_names = {ACTION_FORWARD: "前进", ACTION_ROTATE_LEFT: "左转", ACTION_ROTATE_RIGHT: "右转", ACTION_STEER_LEFT: "左轮差", ACTION_STEER_RIGHT: "右轮差", ACTION_STOP: "停止"}
            logger.info(f"生成动作: {[action_names.get(a, '未知') for a in actions]}")
        else:
            logger.debug("未生成动作（保持停止）")
       
        # 执行动作
        if actions:
            # 清空队列并按顺序放入本帧动作（允许“先 brake 再旋转”等组合动作）
            try:
                while True:
                    self.action_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                for a in actions:
                    self.action_queue.put_nowait(a)
                self._last_dispatched_action = actions[0] if actions else None
            except queue.Full:
                pass
        else:
            # 仅在有明确原因（毫米波/红外障碍、人在停止距离内等）时才发 STOP，其他情况保持原运动
            if self._explicit_stop_requested:
                self.stop_action_execution = True
                try:
                    while True:
                        self.action_queue.get_nowait()
                except queue.Empty:
                    pass
                self._send_stop_with_brake_hold(self._last_explicit_stop_reason or "explicit_stop")
            # 否则（如等待丢失确认、冷却、距离合适等）不清队列、不发 STOP
   
    def run(self):
        """运行跟踪主循环（每帧由板子 Yolov HAL get_frame + detect 获取检测结果）"""
        logger.info("开始运行人员跟踪（避障版本，vision_engine=%s）...", self._vision_engine)
        timeout_sec = 0 if self._vision_engine == "sample_reid" else GET_FRAME_TIMEOUT
        try:
            while self.running:
                if timeout_sec > 0:
                    # 带超时：避免 get_frame 与录像争用 VENC 时无限阻塞导致卡死
                    done = threading.Event()
                    exc_holder = []
                    def _run():
                        try:
                            self.process_frame()
                        except Exception as e:
                            exc_holder.append(e)
                        finally:
                            done.set()
                    t = threading.Thread(target=_run, daemon=True)
                    t.start()
                    if not done.wait(timeout=timeout_sec):
                        logger.warning("process_frame 超时(%.1fs)，跳过本帧，减轻与录像争用", timeout_sec)
                    if exc_holder:
                        raise exc_holder[0]
                else:
                    self.process_frame()
                time.sleep(PROCESS_FRAME_INTERVAL)
       
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭...")
        finally:
            self.running = False
            self.action_stop_event.set()
            # 先发 STOP，确保 Ctrl+C 后车一定停（避免中断发生在 detect 后、发停前的空窗期）
            try:
                self._send_stop_with_brake_hold(self._last_explicit_stop_reason or "explicit_stop")
            except Exception as e:
                logger.warning(f"finally 中发送 STOP 失败: {e}")
            if self.action_thread:
                self.action_thread.join(timeout=1.0)
            try:
                if self._vision_wrapper is not None:
                    self._vision_wrapper.stop()
                    self._vision_iter = None
                    logger.info("sample_reid vision wrapper stopped")
            except Exception as e:
                logger.warning("sample_reid vision wrapper stop failed: %s", e)
            try:
                if self._rknn_pipeline is not None:
                    self._rknn_pipeline.close()
                    self._rknn_pipeline = None
                    logger.info("RKNN vision pipeline closed")
            except Exception as e:
                logger.warning("RKNN vision pipeline close failed: %s", e)
            # Ctrl+C 若刚好在 detect 之后、Yolov.stop 之前，此处补一次 stop 释放本帧资源
            if self._current_obj_meta is not None:
                try:
                    if Yolov is not None:
                        Yolov.stop(self._current_obj_meta)
                except Exception as e:
                    logger.warning(f"finally 中 Yolov.stop 出错: {e}")
                self._current_obj_meta = None
            # 收尾前再发一次 STOP，确保车已停（应对中断发生在 detect 后、发停前的空窗期）
            try:
                self._send_robot_command(ACTION_STOP)
            except Exception as e:
                logger.warning(f"finally 中再次发送 STOP 失败: {e}")
            try:
                if self._bunker_monitor is not None:
                    self._bunker_monitor.stop()
                    self._bunker_monitor = None
                    logger.info("沙坑/水坑 split 检测已关闭")
            except Exception as e:
                logger.warning("沙坑/水坑 split 检测关闭出错: %s", e)
            try:
                if self._bunker_rknn_detector is not None:
                    self._bunker_rknn_detector.stop()
                    self._bunker_rknn_detector = None
                    logger.info("沙坑/水坑 split RKNN 检测已关闭")
            except Exception as e:
                logger.warning("沙坑/水坑 split RKNN 检测关闭出错: %s", e)
            # 关闭 YOLO HAL
            try:
                if self._yolov_hal_active:
                    if Yolov is not None:
                        Yolov.deinit()
                    self._yolov_hal_active = False
                    logger.info("Yolov HAL 已关闭")
            except Exception as e:
                logger.warning(f"Yolov.deinit 出错: {e}")
            # Close HAL sensors; motor driver is closed separately.
            try:
                if MODULE_IR_ENABLE and IR is not None:
                    IR.deinit()
                if MODULE_ULTRASONIC_ENABLE and self._Utrasonic is not None:
                    self._Utrasonic.deinit()
                if MODULE_MMWAVE_ENABLE and MmWaveRadar is not None:
                    MmWaveRadar.deinit()
                logger.info("HAL传感器已关闭")
            except Exception as e:
                logger.warning("关闭HAL传感器时出错: %s", e)
            try:
                self._close_mssd_driver()
            except Exception as e:
                logger.warning("MSSD motor backend close failed: %s", e)
            logger.info("人员跟踪（避障版本）已停止")


def main():
    # 模型路径：与 yolov_test 一致，支持命令行第一个参数，不传则用默认
    model_path = sys.argv[1] if len(sys.argv) >= 2 else None
    if model_path is None:
        logger.info(f"未传模型路径，使用默认: {YOLO_MODEL_PATH}")
    tracker = PersonTracker(model_path=model_path)
    tracker.run()


if __name__ == "__main__":
    main()
