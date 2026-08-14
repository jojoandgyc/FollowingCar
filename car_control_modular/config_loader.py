#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INI configuration loader for the modular request entrypoint.

The current request script reads most knobs from environment variables at
import time.  To keep this migration low-risk, this loader maps INI values into
those environment variables before the copied request constants are evaluated.
"""

from __future__ import annotations

import configparser
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


def _as_bool_env(value: str) -> str:
    return "1" if str(value).strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"} else "0"


def _set_env_if_present(parser: configparser.ConfigParser, section: str, option: str, env_name: str) -> None:
    if parser.has_option(section, option):
        os.environ[env_name] = parser.get(section, option).strip()


def _set_env_if_unset(parser: configparser.ConfigParser, section: str, option: str, env_name: str) -> None:
    """Load an INI fallback without overriding an explicit process setting."""
    if not os.environ.get(env_name, "").strip() and parser.has_option(section, option):
        os.environ[env_name] = parser.get(section, option).strip()


def _set_bool_env_if_present(parser: configparser.ConfigParser, section: str, option: str, env_name: str) -> None:
    if parser.has_option(section, option):
        os.environ[env_name] = _as_bool_env(parser.get(section, option))


@dataclass(frozen=True)
class LoadedConfig:
    path: str
    values: Dict[str, Dict[str, str]]


def find_config_arg(argv: Iterable[str]) -> Optional[str]:
    args = list(argv)
    for idx, arg in enumerate(args):
        if arg == "--config" and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def strip_config_arg(argv: Iterable[str]) -> list:
    args = list(argv)
    out = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--config":
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        out.append(arg)
    return out


def load_config_to_env(config_path: Optional[str]) -> Optional[LoadedConfig]:
    if not config_path:
        return None

    path = os.path.abspath(config_path)
    parser = configparser.ConfigParser()
    read_files = parser.read(path, encoding="utf-8")
    if not read_files:
        raise RuntimeError(f"配置文件不存在或无法读取: {path}")

    # Module switches.  Keep both generic MODULE_* names and existing BUNKER_*.
    _set_bool_env_if_present(parser, "modules", "vision", "MODULE_VISION_ENABLE")
    _set_bool_env_if_present(parser, "modules", "ir", "MODULE_IR_ENABLE")
    _set_bool_env_if_present(parser, "modules", "mmwave", "MODULE_MMWAVE_ENABLE")
    _set_bool_env_if_present(parser, "modules", "ultrasonic", "MODULE_ULTRASONIC_ENABLE")
    _set_bool_env_if_present(parser, "modules", "imu", "MODULE_IMU_ENABLE")
    _set_bool_env_if_present(parser, "modules", "bunker", "BUNKER_AVOID_ENABLE")

    # IR HTTP client.  The board endpoint returns only triggered/idle state;
    # distance thresholds are owned by the board-side IR service or hardware.
    _set_env_if_present(parser, "ir", "backend", "IR_BACKEND")
    _set_env_if_present(parser, "ir", "host", "IR_HTTP_HOST")
    _set_env_if_present(parser, "ir", "port", "IR_HTTP_PORT")
    _set_env_if_present(parser, "ir", "timeout_sec", "IR_HTTP_TIMEOUT_SEC")
    _set_env_if_present(parser, "ir", "cache_sec", "IR_HTTP_CACHE_SEC")
    _set_env_if_present(parser, "ir", "stale_sec", "IR_HTTP_STALE_SEC")
    _set_env_if_present(parser, "ir", "trigger_value", "IR_TRIGGER_VALUE")
    _set_env_if_present(parser, "ir", "iio_base_dir", "IR_IIO_BASE_DIR")
    _set_env_if_present(parser, "ir", "iio_right_device", "IR_IIO_RIGHT_DEVICE")
    _set_env_if_present(parser, "ir", "iio_left_device", "IR_IIO_LEFT_DEVICE")
    _set_env_if_present(parser, "ir", "iio_front_device", "IR_IIO_FRONT_DEVICE")
    _set_bool_env_if_present(parser, "ir", "raw_log_enable", "IR_RAW_LOG_ENABLE")
    _set_env_if_present(parser, "ir", "raw_log_every_sec", "IR_RAW_LOG_EVERY_SEC")

    # AT2410 mmWave radar on RK3588 UART.  A legacy ctypes backend remains
    # available through MMWAVE_BACKEND=ctypes for older board images.
    _set_env_if_present(parser, "mmwave", "backend", "MMWAVE_BACKEND")
    # MMWAVE_AT2410_PORT is the single runtime override for the AT2410 node.
    # The INI value remains the default, but must not replace an exported value.
    _set_env_if_unset(parser, "mmwave", "port", "MMWAVE_AT2410_PORT")
    _set_env_if_unset(parser, "mmwave", "device", "MMWAVE_AT2410_PORT")
    _set_env_if_present(parser, "mmwave", "baudrate", "MMWAVE_AT2410_BAUDRATE")
    _set_env_if_present(parser, "mmwave", "read_timeout_sec", "MMWAVE_AT2410_READ_TIMEOUT_SEC")
    _set_env_if_present(parser, "mmwave", "read_size", "MMWAVE_AT2410_READ_SIZE")
    _set_bool_env_if_present(parser, "mmwave", "verify_on_init", "MMWAVE_AT2410_VERIFY_ON_INIT")
    _set_env_if_present(parser, "mmwave", "target_limit", "MMWAVE_TARGET_LIMIT")
    _set_env_if_present(parser, "mmwave", "target_mode", "MMWAVE_TARGET_MODE")
    _set_env_if_present(parser, "mmwave", "front_angle_deg", "MMWAVE_FRONT_ANGLE_DEG")
    _set_env_if_present(parser, "mmwave", "distance_bias_m", "MMWAVE_DISTANCE_BIAS_M")
    _set_env_if_present(parser, "mmwave", "min_distance_m", "MMWAVE_MIN_DISTANCE_M")
    _set_env_if_present(parser, "mmwave", "max_distance_m", "MMWAVE_MAX_DISTANCE_M")
    _set_env_if_present(parser, "mmwave", "min_output_distance_m", "MMWAVE_MIN_OUTPUT_DISTANCE_M")
    _set_env_if_present(parser, "mmwave", "stale_hold_sec", "MMWAVE_STALE_HOLD_SEC")
    _set_env_if_present(parser, "mmwave", "conservative_window_sec", "MMWAVE_CONSERVATIVE_WINDOW_SEC")
    _set_bool_env_if_present(parser, "mmwave", "async_cache_enable", "MMWAVE_ASYNC_CACHE_ENABLE")
    _set_env_if_present(parser, "mmwave", "cache_interval_sec", "MMWAVE_CACHE_INTERVAL_SEC")
    _set_env_if_present(parser, "mmwave", "cache_window_sec", "MMWAVE_CACHE_WINDOW_SEC")

    # Ultrasonic SR04 IIO driver on RK3588.
    _set_env_if_present(parser, "ultrasonic", "backend", "ULTRASONIC_BACKEND")
    _set_env_if_present(parser, "ultrasonic", "iio_base_dir", "ULTRASONIC_IIO_BASE_DIR")
    _set_env_if_present(parser, "ultrasonic", "iio_device", "ULTRASONIC_IIO_DEVICE")
    _set_env_if_present(parser, "ultrasonic", "iio_device_name", "ULTRASONIC_IIO_DEVICE_NAME")
    _set_env_if_present(parser, "ultrasonic", "min_distance_m", "ULTRASONIC_MIN_DISTANCE_M")
    _set_env_if_present(parser, "ultrasonic", "max_distance_m", "ULTRASONIC_MAX_DISTANCE_M")
    _set_env_if_present(parser, "ultrasonic", "filter_window", "ULTRASONIC_FILTER_WINDOW")
    _set_env_if_present(parser, "ultrasonic", "target_confirm_frames", "ULTRASONIC_TARGET_CONFIRM_FRAMES")
    _set_env_if_present(parser, "ultrasonic", "brake_confirm_frames", "ULTRASONIC_BRAKE_CONFIRM_FRAMES")
    _set_env_if_present(parser, "ultrasonic", "hysteresis_m", "ULTRASONIC_HYSTERESIS_M")
    _set_env_if_present(parser, "ultrasonic", "immediate_brake_m", "ULTRASONIC_IMMEDIATE_BRAKE_M")

    # ICM20600 accel/gyro through Rockchip sensor misc devices and Linux input.
    _set_env_if_present(parser, "imu", "backend", "IMU_BACKEND")
    _set_bool_env_if_present(parser, "imu", "fail_soft", "IMU_FAIL_SOFT")
    _set_bool_env_if_present(parser, "imu", "enable_on_init", "IMU_ENABLE_ON_INIT")
    _set_bool_env_if_present(parser, "imu", "disable_on_deinit", "IMU_DISABLE_ON_DEINIT")
    _set_bool_env_if_present(parser, "imu", "log_enable", "IMU_LOG_ENABLE")
    _set_env_if_present(parser, "imu", "log_every_sec", "IMU_LOG_EVERY_SEC")
    _set_env_if_present(parser, "imu", "event_base_dir", "IMU_EVENT_BASE_DIR")
    _set_env_if_present(parser, "imu", "sys_input_base_dir", "IMU_SYS_INPUT_BASE_DIR")
    _set_env_if_present(parser, "imu", "accel_event", "IMU_ACCEL_EVENT")
    _set_env_if_present(parser, "imu", "gyro_event", "IMU_GYRO_EVENT")
    _set_env_if_present(parser, "imu", "accel_name", "IMU_ACCEL_NAME")
    _set_env_if_present(parser, "imu", "gyro_name", "IMU_GYRO_NAME")
    _set_env_if_present(parser, "imu", "accel_misc_dev", "IMU_ACCEL_MISC_DEV")
    _set_env_if_present(parser, "imu", "gyro_misc_dev", "IMU_GYRO_MISC_DEV")
    _set_env_if_present(parser, "imu", "accel_rate_ms", "IMU_ACCEL_RATE_MS")
    _set_env_if_present(parser, "imu", "gyro_rate_ms", "IMU_GYRO_RATE_MS")
    _set_env_if_present(parser, "imu", "accel_lsb_per_g", "IMU_ACCEL_LSB_PER_G")
    _set_env_if_present(parser, "imu", "gyro_lsb_per_dps", "IMU_GYRO_LSB_PER_DPS")

    # Vision/person.
    _set_env_if_present(parser, "vision", "model_path", "VISION_MODEL_PATH")
    _set_env_if_present(parser, "vision", "engine", "VISION_ENGINE")
    _set_bool_env_if_present(parser, "vision", "reid_enable", "VISION_REID_ENABLE")
    _set_env_if_present(parser, "vision", "reid_model_path", "VISION_REID_MODEL_PATH")
    _set_env_if_present(parser, "vision", "sample_workdir", "VISION_SAMPLE_WORKDIR")
    _set_env_if_present(parser, "vision", "sample_binary", "VISION_SAMPLE_BINARY")
    _set_bool_env_if_present(parser, "vision", "sample_single_person", "VISION_SAMPLE_SINGLE_PERSON")
    _set_bool_env_if_present(parser, "vision", "sample_echo_raw", "VISION_SAMPLE_ECHO_RAW")
    _set_env_if_present(parser, "vision", "sample_det_conf", "VISION_SAMPLE_DET_CONF")
    _set_env_if_present(parser, "vision", "sample_log_path", "VISION_SAMPLE_LOG_PATH")
    _set_bool_env_if_present(parser, "vision", "track_log_enable", "VISION_TRACK_LOG_ENABLE")
    _set_env_if_present(parser, "vision", "track_log_empty_every", "VISION_TRACK_LOG_EMPTY_EVERY")
    _set_env_if_present(parser, "vision", "frame_width", "VISION_FRAME_WIDTH")
    _set_env_if_present(parser, "vision", "frame_height", "VISION_FRAME_HEIGHT")
    _set_env_if_present(parser, "vision", "hfov_deg", "VISION_HFOV_DEG")
    _set_env_if_present(parser, "vision", "effective_fps", "VISION_EFFECTIVE_FPS")
    _set_env_if_present(parser, "vision", "person_class_id", "PERSON_CLASS_ID")
    _set_env_if_present(parser, "vision", "confidence_threshold", "CONFIDENCE_THRESHOLD")
    _set_env_if_present(parser, "vision", "target_select", "VISION_TARGET_SELECT")
    _set_bool_env_if_present(parser, "vision", "control_use_predicted_tracks", "VISION_CONTROL_USE_PREDICTED_TRACKS")
    _set_env_if_present(parser, "vision", "rknn_target", "RKNN_TARGET")
    _set_env_if_present(parser, "vision", "rknn_core_mask", "RKNN_CORE_MASK")
    _set_env_if_present(parser, "vision", "rknn_backend", "RKNN_BACKEND")
    _set_env_if_present(parser, "vision", "input_size", "RKNN_YOLO_INPUT_SIZE")
    _set_env_if_present(parser, "vision", "num_classes", "RKNN_YOLO_NUM_CLASSES")
    _set_env_if_present(parser, "vision", "nms_threshold", "RKNN_YOLO_NMS_THRESHOLD")
    _set_env_if_present(parser, "vision", "yolo_box_format", "RKNN_YOLO_BOX_FORMAT")
    _set_env_if_present(parser, "vision", "yolo_input_format", "RKNN_YOLO_INPUT_FORMAT")
    _set_env_if_present(parser, "vision", "reid_input_width", "RKNN_REID_INPUT_WIDTH")
    _set_env_if_present(parser, "vision", "reid_input_height", "RKNN_REID_INPUT_HEIGHT")
    _set_env_if_present(parser, "vision", "reid_input_format", "RKNN_REID_INPUT_FORMAT")
    _set_env_if_present(parser, "vision", "reid_input_dtype", "RKNN_REID_INPUT_DTYPE")
    _set_env_if_present(parser, "vision", "reid_input_layout", "RKNN_REID_INPUT_LAYOUT")
    _set_env_if_present(parser, "vision", "reid_normalize", "RKNN_REID_NORMALIZE")

    # RKNN camera source. request_0513_modular.py can use a GStreamer MJPEG
    # capture path here while request_0512_modular.py keeps the OpenCV path.
    _set_bool_env_if_present(parser, "camera", "enable", "RKNN_CAMERA_ENABLE")
    _set_env_if_present(parser, "camera", "device", "RKNN_CAMERA_DEVICE")
    _set_env_if_present(parser, "camera", "width", "RKNN_CAMERA_WIDTH")
    _set_env_if_present(parser, "camera", "height", "RKNN_CAMERA_HEIGHT")
    _set_env_if_present(parser, "camera", "fps", "RKNN_CAMERA_FPS")
    _set_env_if_present(parser, "camera", "fourcc", "RKNN_CAMERA_FOURCC")
    _set_env_if_present(parser, "camera", "capture_mode", "RKNN_CAMERA_CAPTURE_MODE")
    _set_env_if_present(parser, "camera", "raw_output", "RKNN_CAMERA_RAW_OUTPUT")
    _set_env_if_present(parser, "camera", "retry_interval_sec", "RKNN_CAMERA_RETRY_INTERVAL_SEC")
    _set_env_if_present(parser, "camera", "latest_drain_max", "RKNN_CAMERA_LATEST_DRAIN_MAX")

    # Target memory sits above DeepSORT/ReID output: keep following the
    # selected stable id for a short window before allowing a new largest person.
    _set_bool_env_if_present(parser, "track_memory", "enabled", "TRACK_MEMORY_ENABLED")
    _set_bool_env_if_present(parser, "track_memory", "auto_from_deepsort", "TRACK_MEMORY_AUTO_FROM_DEEPSORT")
    _set_env_if_present(parser, "track_memory", "memory_ratio", "TRACK_MEMORY_DERIVED_MEMORY_RATIO")
    _set_env_if_present(parser, "track_memory", "allow_new_extra_sec", "TRACK_MEMORY_ALLOW_NEW_EXTRA_SEC")
    _set_env_if_present(parser, "track_memory", "memory_sec", "TRACK_MEMORY_SEC")
    _set_env_if_present(parser, "track_memory", "allow_new_target_after_sec", "TRACK_MEMORY_ALLOW_NEW_TARGET_AFTER_SEC")
    _set_bool_env_if_present(parser, "track_memory", "reacquire_same_id", "TRACK_MEMORY_REACQUIRE_SAME_ID")
    _set_bool_env_if_present(parser, "track_memory", "reacquire_by_geometry", "TRACK_MEMORY_REACQUIRE_BY_GEOMETRY")
    _set_env_if_present(parser, "track_memory", "reacquire_confirm_frames", "TRACK_MEMORY_REACQUIRE_CONFIRM_FRAMES")
    _set_env_if_present(parser, "track_memory", "max_center_jump_ratio", "TRACK_MEMORY_MAX_CENTER_JUMP_RATIO")
    _set_env_if_present(parser, "track_memory", "min_area_similarity", "TRACK_MEMORY_MIN_AREA_SIMILARITY")
    _set_env_if_present(parser, "track_memory", "max_aspect_diff", "TRACK_MEMORY_MAX_ASPECT_DIFF")
    _set_bool_env_if_present(parser, "track_memory", "log_enable", "TRACK_MEMORY_LOG_ENABLE")

    # Short-window search behavior after the selected target disappears.  This
    # only steers the search action; it does not accept a new identity.
    _set_bool_env_if_present(parser, "predictive_search", "enabled", "PREDICTIVE_SEARCH_ENABLED")
    _set_env_if_present(parser, "predictive_search", "history_sec", "PREDICTIVE_SEARCH_HISTORY_SEC")
    _set_env_if_present(parser, "predictive_search", "predict_sec", "PREDICTIVE_SEARCH_SEC")
    _set_env_if_present(parser, "predictive_search", "edge_margin_ratio", "PREDICTIVE_SEARCH_EDGE_MARGIN_RATIO")
    _set_env_if_present(parser, "predictive_search", "min_x_motion_ratio", "PREDICTIVE_SEARCH_MIN_X_MOTION_RATIO")
    _set_env_if_present(parser, "predictive_search", "far_area_drop_ratio", "PREDICTIVE_SEARCH_FAR_AREA_DROP_RATIO")
    _set_env_if_present(parser, "predictive_search", "far_center_ratio", "PREDICTIVE_SEARCH_FAR_CENTER_RATIO")
    _set_env_if_present(parser, "predictive_search", "far_forward_percent", "PREDICTIVE_SEARCH_FAR_FORWARD_PERCENT")
    _set_bool_env_if_present(parser, "predictive_search", "log_enable", "PREDICTIVE_SEARCH_LOG_ENABLE")

    # DeepSORT identity/lost-track tuning.
    _set_env_if_present(parser, "deepsort", "max_unmatched_num", "Y8_DEEPSORT_MAX_UNMATCHED_NUM")
    _set_env_if_present(parser, "deepsort", "max_output_age", "Y8_DEEPSORT_MAX_OUTPUT_AGE")
    _set_env_if_present(parser, "deepsort", "accreditation_threshold", "Y8_DEEPSORT_ACCREDITATION_THRESHOLD")
    _set_env_if_present(
        parser,
        "deepsort",
        "max_unmatched_times_for_bbox_matching",
        "Y8_DEEPSORT_MAX_UNMATCHED_TIMES_FOR_BBOX_MATCHING",
    )
    _set_env_if_present(parser, "deepsort", "max_distance_iou", "Y8_DEEPSORT_MAX_DISTANCE_IOU")
    _set_env_if_present(parser, "deepsort", "max_distance_cosine", "Y8_DEEPSORT_MAX_DISTANCE_CONSINE")
    _set_env_if_present(parser, "deepsort", "feature_budget_size", "Y8_DEEPSORT_FEATURE_BUDGET_SIZE")
    _set_env_if_present(parser, "deepsort", "feature_update_interval", "Y8_DEEPSORT_FEATURE_UPDATE_INTERVAL")
    _set_env_if_present(parser, "deepsort", "min_confidence", "Y8_DEEPSORT_MIN_CONFIDENCE")
    _set_env_if_present(parser, "deepsort", "nms_max_overlap", "Y8_DEEPSORT_NMS_MAX_OVERLAP")
    _set_env_if_present(parser, "deepsort", "bbox_expand_scale", "Y8_DEEPSORT_BBOX_EXPAND_SCALE")

    _set_bool_env_if_present(parser, "identity_bank", "enabled", "Y8_IDENTITY_BANK_ENABLE")
    _set_env_if_present(parser, "identity_bank", "match_threshold", "Y8_IDENTITY_MATCH_THRESHOLD")
    _set_env_if_present(parser, "identity_bank", "match_margin", "Y8_IDENTITY_MATCH_MARGIN")
    _set_env_if_present(parser, "identity_bank", "update_threshold", "Y8_IDENTITY_UPDATE_THRESHOLD")
    _set_env_if_present(parser, "identity_bank", "update_interval", "Y8_IDENTITY_UPDATE_INTERVAL")
    _set_env_if_present(parser, "identity_bank", "max_features", "Y8_IDENTITY_MAX_FEATURES")
    _set_env_if_present(parser, "identity_bank", "min_confidence", "Y8_IDENTITY_MIN_CONFIDENCE")
    _set_env_if_present(parser, "identity_bank", "min_area", "Y8_IDENTITY_MIN_AREA")
    _set_env_if_present(parser, "identity_bank", "max_area_ratio", "Y8_IDENTITY_MAX_AREA_RATIO")
    _set_env_if_present(parser, "identity_bank", "max_width_ratio", "Y8_IDENTITY_MAX_WIDTH_RATIO")
    _set_env_if_present(parser, "identity_bank", "max_height_ratio", "Y8_IDENTITY_MAX_HEIGHT_RATIO")
    _set_env_if_present(parser, "identity_bank", "min_aspect_ratio", "Y8_IDENTITY_MIN_ASPECT_RATIO")
    _set_env_if_present(parser, "identity_bank", "max_aspect_ratio", "Y8_IDENTITY_MAX_ASPECT_RATIO")
    _set_env_if_present(parser, "identity_bank", "max_edge_touch_count", "Y8_IDENTITY_MAX_EDGE_TOUCH_COUNT")
    _set_env_if_present(parser, "identity_bank", "edge_margin_ratio", "Y8_IDENTITY_EDGE_MARGIN_RATIO")
    _set_env_if_present(parser, "identity_bank", "reacquire_threshold", "Y8_IDENTITY_REACQUIRE_THRESHOLD")
    _set_env_if_present(parser, "identity_bank", "reacquire_max_frames", "Y8_IDENTITY_REACQUIRE_MAX_FRAMES")
    _set_env_if_present(parser, "identity_bank", "reacquire_margin", "Y8_IDENTITY_REACQUIRE_MARGIN")
    _set_bool_env_if_present(
        parser,
        "identity_bank",
        "reacquire_single_candidate_only",
        "Y8_IDENTITY_REACQUIRE_SINGLE_CANDIDATE_ONLY",
    )
    _set_bool_env_if_present(
        parser,
        "identity_bank",
        "reacquire_multi_candidate_enable",
        "Y8_IDENTITY_REACQUIRE_MULTI_CANDIDATE_ENABLE",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "reacquire_multi_candidate_threshold",
        "Y8_IDENTITY_REACQUIRE_MULTI_CANDIDATE_THRESHOLD",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "reacquire_multi_candidate_margin",
        "Y8_IDENTITY_REACQUIRE_MULTI_CANDIDATE_MARGIN",
    )
    _set_env_if_present(parser, "identity_bank", "new_identity_confirm_frames", "Y8_IDENTITY_NEW_CONFIRM_FRAMES")
    _set_bool_env_if_present(parser, "identity_bank", "mapped_verify_enable", "Y8_IDENTITY_MAPPED_VERIFY_ENABLE")
    _set_env_if_present(parser, "identity_bank", "mapped_verify_threshold", "Y8_IDENTITY_MAPPED_VERIFY_THRESHOLD")
    _set_bool_env_if_present(
        parser,
        "identity_bank",
        "mapped_bad_quality_returns_unassigned",
        "Y8_IDENTITY_MAPPED_BAD_QUALITY_RETURNS_UNASSIGNED",
    )
    _set_bool_env_if_present(
        parser,
        "identity_bank",
        "exclusive_uid_claim_enable",
        "Y8_IDENTITY_EXCLUSIVE_UID_CLAIM_ENABLE",
    )
    _set_env_if_present(parser, "identity_bank", "exclusive_uid_claim_frames", "Y8_IDENTITY_EXCLUSIVE_UID_CLAIM_FRAMES")
    _set_bool_env_if_present(parser, "identity_bank", "controlled_handoff_enable", "Y8_IDENTITY_CONTROLLED_HANDOFF_ENABLE")
    _set_env_if_present(
        parser,
        "identity_bank",
        "controlled_handoff_confirm_frames",
        "Y8_IDENTITY_CONTROLLED_HANDOFF_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "controlled_handoff_threshold",
        "Y8_IDENTITY_CONTROLLED_HANDOFF_THRESHOLD",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "controlled_handoff_min_old_track_gap_frames",
        "Y8_IDENTITY_CONTROLLED_HANDOFF_MIN_OLD_TRACK_GAP_FRAMES",
    )
    _set_bool_env_if_present(parser, "identity_bank", "suppress_duplicate_uids", "Y8_IDENTITY_SUPPRESS_DUPLICATE_UIDS")

    _set_bool_env_if_present(parser, "predicted_reid", "verify_enable", "Y8_PREDICTED_REID_VERIFY_ENABLE")
    _set_env_if_present(parser, "predicted_reid", "verify_threshold", "Y8_PREDICTED_REID_VERIFY_THRESHOLD")
    _set_env_if_present(
        parser,
        "predicted_reid",
        "duplicate_iou_threshold",
        "Y8_PREDICTED_REID_DUPLICATE_IOU_THRESHOLD",
    )
    _set_env_if_present(
        parser,
        "predicted_reid",
        "duplicate_overlap_threshold",
        "Y8_PREDICTED_REID_DUPLICATE_OVERLAP_THRESHOLD",
    )

    # Bunker split/merged detector.
    _set_env_if_present(parser, "bunker", "mode", "BUNKER_DETECT_MODE")
    _set_env_if_present(parser, "bunker", "engine", "BUNKER_ENGINE")
    _set_env_if_present(parser, "bunker", "workdir", "BUNKER_WORKDIR")
    _set_env_if_present(parser, "bunker", "model_path", "BUNKER_MODEL_PATH")
    _set_env_if_present(parser, "bunker", "class_ids", "BUNKER_STOP_CLASS_IDS")
    _set_env_if_present(parser, "bunker", "class_names", "BUNKER_CLASS_NAMES")
    _set_env_if_present(parser, "bunker", "score_threshold", "BUNKER_STOP_SCORE_THRESHOLD")
    _set_env_if_present(parser, "bunker", "area_ratio_stop", "BUNKER_STOP_AREA_RATIO")
    _set_env_if_present(parser, "bunker", "num_classes", "BUNKER_NUM_CLASSES")
    _set_env_if_present(parser, "bunker", "input_size", "BUNKER_RKNN_INPUT_SIZE")
    _set_env_if_present(parser, "bunker", "nms_threshold", "BUNKER_RKNN_NMS_THRESHOLD")
    _set_env_if_present(parser, "bunker", "rknn_backend", "BUNKER_RKNN_BACKEND")
    _set_env_if_present(parser, "bunker", "rknn_core_mask", "BUNKER_RKNN_CORE_MASK")
    _set_env_if_present(parser, "bunker", "input_format", "BUNKER_RKNN_INPUT_FORMAT")
    _set_env_if_present(parser, "bunker", "box_format", "BUNKER_RKNN_BOX_FORMAT")
    _set_env_if_present(parser, "bunker", "sample_det_conf", "BUNKER_SAMPLE_DET_CONF")
    _set_env_if_present(parser, "bunker", "get_frame_timeout_ms", "BUNKER_GET_FRAME_TIMEOUT_MS")
    _set_env_if_present(parser, "bunker", "loop_period_ms", "BUNKER_LOOP_PERIOD_MS")
    _set_env_if_present(parser, "bunker", "active_hold_sec", "BUNKER_SPLIT_ACTIVE_HOLD_SEC")
    _set_env_if_present(parser, "bunker", "stop_consec_frames", "BUNKER_STOP_CONSEC_FRAMES")
    _set_bool_env_if_present(parser, "bunker", "echo_raw", "BUNKER_SPLIT_ECHO_RAW")
    _set_env_if_present(parser, "bunker", "sample_binary", "BUNKER_SAMPLE_BINARY")

    # Distance/safety.
    _set_env_if_present(parser, "distance", "target_distance_m", "TARGET_DISTANCE")
    _set_env_if_present(parser, "distance", "brake_distance_m", "FOLLOW_BRAKE_DISTANCE_M")
    _set_env_if_present(parser, "distance", "mmwave_obstacle_threshold_m", "MMWAVE_OBSTACLE_THRESHOLD")
    _set_env_if_present(parser, "distance", "source", "DISTANCE_SOURCE")
    _set_env_if_present(parser, "distance", "fallback_forward_percent", "DISTANCE_MISSING_FORWARD_PERCENT")

    # Vision-gated mmwave matching.  Radar is only used after a visual person
    # target has been selected.
    _set_env_if_present(parser, "mmwave_match", "angle_margin_deg", "VISION_MMWAVE_ANGLE_MARGIN_DEG")
    _set_env_if_present(parser, "mmwave_match", "angle_offset_deg", "VISION_MMWAVE_ANGLE_OFFSET_DEG")
    _set_env_if_present(parser, "mmwave_match", "angle_sign", "VISION_MMWAVE_ANGLE_SIGN")
    _set_env_if_present(parser, "mmwave_match", "match_mode", "VISION_MMWAVE_MATCH_MODE")
    _set_env_if_present(parser, "mmwave_match", "angle_tie_margin_deg", "VISION_MMWAVE_ANGLE_TIE_MARGIN_DEG")
    _set_env_if_present(parser, "mmwave_match", "distance_bias_m", "VISION_MMWAVE_DISTANCE_BIAS_M")
    _set_env_if_present(parser, "mmwave_match", "min_distance_m", "VISION_MMWAVE_MIN_DISTANCE_M")
    _set_env_if_present(parser, "mmwave_match", "max_distance_m", "VISION_MMWAVE_MAX_DISTANCE_M")
    _set_env_if_present(parser, "mmwave_match", "min_output_distance_m", "VISION_MMWAVE_MIN_OUTPUT_DISTANCE_M")
    _set_env_if_present(parser, "mmwave_match", "hard_stop_ttl_sec", "VISION_MMWAVE_HARD_STOP_TTL_SEC")
    _set_env_if_present(parser, "mmwave_match", "log_every_frames", "VISION_MMWAVE_LOG_EVERY_FRAMES")
    _set_env_if_present(parser, "mmwave_match", "latency_sec", "VISION_MMWAVE_LATENCY_SEC")
    _set_env_if_present(parser, "mmwave_match", "cache_max_age_sec", "VISION_MMWAVE_CACHE_MAX_AGE_SEC")
    _set_bool_env_if_present(parser, "mmwave_match", "use_async_cache", "VISION_MMWAVE_USE_ASYNC_CACHE")

    # Motion.
    _set_env_if_present(parser, "motor", "backend", "MOTOR_BACKEND")
    _set_env_if_present(parser, "motor", "percent_limit", "MOTOR_PERCENT_LIMIT")
    _set_env_if_present(parser, "motor", "rs485_port", "MOTOR_RS485_PORT")
    _set_env_if_present(parser, "motor", "rs485_slave_id", "MOTOR_RS485_SLAVE_ID")
    _set_env_if_present(parser, "motor", "rs485_baudrate", "MOTOR_RS485_BAUDRATE")
    _set_env_if_present(parser, "motor", "rs485_timeout", "MOTOR_RS485_TIMEOUT")
    _set_env_if_present(parser, "motor", "rs485_lib_dir", "MOTOR_RS485_LIB_DIR")
    _set_env_if_present(parser, "motor", "rs485_max_target", "MOTOR_RS485_MAX_TARGET")
    _set_env_if_present(parser, "motor", "forward_raw_target", "MOTOR_FORWARD_RAW_TARGET")
    _set_env_if_present(parser, "motor", "steer_raw_target", "MOTOR_STEER_RAW_TARGET")
    _set_env_if_present(parser, "motor", "rotate_raw_target", "MOTOR_ROTATE_RAW_TARGET")
    _set_env_if_present(parser, "motor", "left_sign", "MOTOR_LEFT_SIGN")
    _set_env_if_present(parser, "motor", "right_sign", "MOTOR_RIGHT_SIGN")
    _set_env_if_present(parser, "motor", "forward_target_sign", "MOTOR_FORWARD_TARGET_SIGN")
    _set_bool_env_if_present(parser, "motor", "exit_parking_mode_on_arm", "MOTOR_EXIT_PARKING_MODE_ON_ARM")
    _set_env_if_present(parser, "motor", "stop_mode", "MOTOR_RS485_STOP_MODE")
    _set_env_if_present(parser, "motor", "stop_zero_delay_sec", "MOTOR_RS485_STOP_ZERO_DELAY_SEC")
    _set_env_if_present(parser, "motor", "transition_stop_mode", "MOTOR_RS485_TRANSITION_STOP_MODE")
    _set_env_if_present(parser, "motor", "transition_stop_delay_sec", "MOTOR_RS485_TRANSITION_STOP_DELAY_SEC")
    _set_env_if_present(parser, "motor", "transition_stop_repeat", "MOTOR_RS485_TRANSITION_STOP_REPEAT")
    _set_env_if_present(parser, "motor", "target_min_interval_sec", "MOTOR_RS485_TARGET_MIN_INTERVAL_SEC")
    _set_env_if_present(parser, "motor", "forward_like_keepalive_sec", "MOTOR_FORWARD_LIKE_KEEPALIVE_SEC")
    _set_bool_env_if_present(parser, "motor", "m1_is_left_wheel", "M1_IS_LEFT_WHEEL")
    _set_env_if_present(parser, "motion", "min_forward_percent", "MIN_FORWARD_PERCENT")
    _set_env_if_present(parser, "motion", "max_forward_percent", "MAX_FORWARD_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_1_3_percent", "FORWARD_SPEED_LE_1_3_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_1_7_percent", "FORWARD_SPEED_LE_1_7_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_2_1_percent", "FORWARD_SPEED_LE_2_1_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_2_6_percent", "FORWARD_SPEED_LE_2_6_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_3_2_percent", "FORWARD_SPEED_LE_3_2_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_3_8_percent", "FORWARD_SPEED_LE_3_8_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_4_5_percent", "FORWARD_SPEED_LE_4_5_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_far_percent", "FORWARD_SPEED_FAR_PERCENT")
    _set_env_if_present(parser, "motion", "steer_percent_limit", "STEER_PERCENT_LIMIT")
    _set_env_if_present(parser, "motion", "rotate_duration", "ROTATE_DURATION")
    _set_bool_env_if_present(parser, "motion", "rotate_pulse_brake_enable", "ROTATE_PULSE_BRAKE_ENABLE")
    _set_env_if_present(parser, "motion", "rotate_pulse_stop_mode", "ROTATE_PULSE_STOP_MODE")
    _set_env_if_present(parser, "motion", "rotate_pulse_pause_sec", "ROTATE_PULSE_PAUSE_SEC")
    _set_env_if_present(parser, "motion", "rotate_hold_stale_sec", "ROTATE_HOLD_STALE_SEC")
    _set_env_if_present(parser, "motion", "rotate_turn_percent_from_forward", "ROTATE_TURN_PERCENT_FROM_FORWARD")
    _set_env_if_present(parser, "motion", "rotate_turn_percent_chain", "ROTATE_TURN_PERCENT_CHAIN")
    _set_env_if_present(parser, "motion", "rotate_raw_target_visible", "ROTATE_RAW_TARGET_VISIBLE")
    _set_env_if_present(parser, "motion", "rotate_raw_target_lost_wait", "ROTATE_RAW_TARGET_LOST_WAIT")
    _set_env_if_present(parser, "motion", "rotate_raw_target_search", "ROTATE_RAW_TARGET_SEARCH")
    _set_env_if_present(parser, "motion", "visible_steer_inner_ratio_percent", "VISIBLE_STEER_INNER_RATIO_PERCENT")
    _set_env_if_present(parser, "motion", "visible_steer_outer_ratio_percent", "VISIBLE_STEER_OUTER_RATIO_PERCENT")
    _set_env_if_present(parser, "motion", "visible_steer_strong_inner_ratio_percent", "VISIBLE_STEER_STRONG_INNER_RATIO_PERCENT")
    _set_env_if_present(parser, "motion", "visible_steer_strong_outer_ratio_percent", "VISIBLE_STEER_STRONG_OUTER_RATIO_PERCENT")
    _set_env_if_present(parser, "motion", "visible_steer_strong_margin_ratio", "VISIBLE_STEER_STRONG_MARGIN_RATIO")
    _set_env_if_present(parser, "motion", "brake_hold_refresh_sec", "BRAKE_HOLD_REFRESH_INTERVAL_SEC")

    # Follow policy.  These are separate from motion speed knobs because they
    # control the target-selection/search state machine.
    _set_env_if_present(parser, "follow", "center_left_ratio", "FOLLOW_CENTER_LEFT_RATIO")
    _set_env_if_present(parser, "follow", "center_right_ratio", "FOLLOW_CENTER_RIGHT_RATIO")
    _set_env_if_present(parser, "follow", "steer_enter_left_ratio", "FOLLOW_STEER_ENTER_LEFT_RATIO")
    _set_env_if_present(parser, "follow", "steer_enter_right_ratio", "FOLLOW_STEER_ENTER_RIGHT_RATIO")
    _set_env_if_present(parser, "follow", "steer_release_left_ratio", "FOLLOW_STEER_RELEASE_LEFT_RATIO")
    _set_env_if_present(parser, "follow", "steer_release_right_ratio", "FOLLOW_STEER_RELEASE_RIGHT_RATIO")
    _set_env_if_present(parser, "follow", "visible_rotate_left_ratio", "FOLLOW_VISIBLE_ROTATE_LEFT_RATIO")
    _set_env_if_present(parser, "follow", "visible_rotate_right_ratio", "FOLLOW_VISIBLE_ROTATE_RIGHT_RATIO")
    _set_bool_env_if_present(parser, "follow", "use_vertical_center_gate", "FOLLOW_USE_VERTICAL_CENTER_GATE")
    _set_bool_env_if_present(parser, "follow", "search_before_first_person", "FOLLOW_SEARCH_BEFORE_FIRST_PERSON")
    _set_env_if_present(parser, "follow", "startup_search_delay_sec", "FOLLOW_STARTUP_SEARCH_DELAY_SEC")
    _set_env_if_present(parser, "follow", "search_timeout_sec", "FOLLOW_SEARCH_TIMEOUT_SEC")
    _set_env_if_present(parser, "follow", "initial_target_confirm_frames", "FOLLOW_INITIAL_TARGET_CONFIRM_FRAMES")
    _set_bool_env_if_present(parser, "follow", "release_target_on_lost", "FOLLOW_RELEASE_TARGET_ON_LOST")
    _set_env_if_present(parser, "follow", "lost_confirm_sec", "FOLLOW_LOST_CONFIRM_SEC")
    _set_env_if_present(parser, "follow", "lost_confirm_frames", "FOLLOW_LOST_CONFIRM_FRAMES")
    _set_bool_env_if_present(parser, "safety", "side_ir_blocks_rotation", "SIDE_IR_BLOCKS_ROTATION")
    _set_bool_env_if_present(parser, "safety", "search_rotate_front_block_enable", "SEARCH_ROTATE_FRONT_BLOCK_ENABLE")
    _set_bool_env_if_present(parser, "safety", "search_rotate_distance_block_enable", "SEARCH_ROTATE_DISTANCE_BLOCK_ENABLE")
    _set_bool_env_if_present(parser, "safety", "blocked_turn_forward_enable", "BLOCKED_TURN_FORWARD_ENABLE")
    _set_env_if_present(parser, "safety", "blocked_turn_forward_max_steps", "BLOCKED_TURN_FORWARD_MAX_STEPS")
    _set_env_if_present(parser, "safety", "safety_stop_mode", "SAFETY_STOP_MODE")
    _set_env_if_present(parser, "safety", "direct_stop_repeat_interval_sec", "DIRECT_STOP_REPEAT_INTERVAL_SEC")

    values = {section: dict(parser.items(section)) for section in parser.sections()}
    return LoadedConfig(path=path, values=values)


def preload_config_from_argv() -> Optional[LoadedConfig]:
    config_path = find_config_arg(sys.argv[1:])
    loaded = load_config_to_env(config_path)
    if config_path:
        sys.argv[:] = strip_config_arg(sys.argv)
    return loaded
