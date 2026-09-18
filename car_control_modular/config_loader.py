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
import warnings
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

    # This explicit process override is also the rollback switch. Older INIs
    # keep their approach/legacy behaviour when no control_mode is present.
    control_mode = os.environ.get("FOLLOW_DISTANCE_CONTROL_MODE", "").strip()
    if not control_mode:
        control_mode = parser.get("distance_pid", "control_mode", fallback="").strip()
    if not control_mode:
        approach = parser.get("distance_pid", "approach_enable", fallback="false")
        control_mode = "approach" if _as_bool_env(approach) == "1" else "legacy"
    if control_mode not in {"distance_pi", "approach", "legacy"}:
        raise ValueError("FOLLOW_DISTANCE_CONTROL_MODE / [distance_pid] control_mode must be distance_pi, approach or legacy")

    # Narrow, opt-in A/B experiment. Normal DISTANCE_PID_* environment values
    # are overwritten by INI below; this explicit switch changes ONLY P.
    p_trial = os.environ.get("FOLLOW_DISTANCE_P_TRIAL", "").strip()
    if p_trial:
        if p_trial not in {"24", "27", "36"} or not parser.has_section("distance_pid"):
            raise ValueError("FOLLOW_DISTANCE_P_TRIAL must be 24, 27 or 36 with [distance_pid]")
        parser.set("distance_pid", "kp_rpm_per_m", p_trial)
        if control_mode == "distance_pi":
            warnings.warn(
                "FOLLOW_DISTANCE_P_TRIAL does not tune distance_pi forward control; "
                "it only changes the legacy/reverse PID. Use "
                "FOLLOW_DISTANCE_CONTROL_MODE=legacy for the legacy P experiment.",
                RuntimeWarning, stacklevel=2,
            )
    bias_trial = os.environ.get("FOLLOW_MATCHING_BIAS_TRIAL", "0").strip() or "0"
    if bias_trial not in {"0", "5", "10"}:
        raise ValueError("FOLLOW_MATCHING_BIAS_TRIAL must be 0, 5 or 10 (RPM)")
    os.environ["DISTANCE_MATCHING_TEST_BIAS_RPM"] = bias_trial
    matching_mode = os.environ.get("FOLLOW_MATCHING_MODE", "").strip()
    if matching_mode:
        if matching_mode not in {"optional", "distance_only"} or not parser.has_section("distance_pid"):
            raise ValueError("FOLLOW_MATCHING_MODE must be optional or distance_only with [distance_pid]")
        parser.set("distance_pid", "approach_matching_enable", "true" if matching_mode == "optional" else "false")

    # Module switches.  Keep both generic MODULE_* names and existing BUNKER_*.
    _set_bool_env_if_present(parser, "modules", "vision", "MODULE_VISION_ENABLE")
    _set_bool_env_if_present(parser, "modules", "ir", "MODULE_IR_ENABLE")
    _set_bool_env_if_present(parser, "modules", "mmwave", "MODULE_MMWAVE_ENABLE")
    _set_bool_env_if_present(parser, "modules", "astra_depth", "MODULE_ASTRA_DEPTH_ENABLE")
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
    _set_env_if_present(parser, "mmwave", "verify_timeout_sec", "MMWAVE_AT2410_VERIFY_TIMEOUT_SEC")
    _set_bool_env_if_present(parser, "mmwave", "verify_on_init", "MMWAVE_AT2410_VERIFY_ON_INIT")
    _set_bool_env_if_present(parser, "mmwave", "usb_reset_on_init", "MMWAVE_AT2410_USB_RESET_ON_INIT")
    _set_env_if_present(parser, "mmwave", "usb_reset_attempts", "MMWAVE_AT2410_USB_RESET_ATTEMPTS")
    _set_env_if_present(parser, "mmwave", "usb_id", "MMWAVE_AT2410_USB_ID")
    _set_env_if_present(parser, "mmwave", "usb_reconnect_timeout_sec", "MMWAVE_AT2410_USB_RECONNECT_TIMEOUT_SEC")
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

    # Astra Pro uses one OpenNI device for synchronized RGB and registered depth.
    _set_env_if_present(parser, "astra_depth", "openni_path", "ASTRA_DEPTH_OPENNI_PATH")
    _set_env_if_present(parser, "astra_depth", "width", "ASTRA_DEPTH_WIDTH")
    _set_env_if_present(parser, "astra_depth", "height", "ASTRA_DEPTH_HEIGHT")
    _set_env_if_present(parser, "astra_depth", "fps", "ASTRA_DEPTH_FPS")
    _set_env_if_present(parser, "astra_depth", "min_distance_m", "ASTRA_DEPTH_MIN_DISTANCE_M")
    _set_env_if_present(parser, "astra_depth", "max_distance_m", "ASTRA_DEPTH_MAX_DISTANCE_M")
    _set_env_if_present(parser, "astra_depth", "max_frame_age_sec", "ASTRA_DEPTH_MAX_FRAME_AGE_SEC")
    _set_env_if_present(parser, "astra_depth", "hold_sec", "ASTRA_DEPTH_HOLD_SEC")
    _set_env_if_present(
        parser,
        "astra_depth",
        "rgb_processing_delay_sec",
        "ASTRA_DEPTH_RGB_PROCESSING_DELAY_SEC",
    )
    _set_env_if_present(parser, "astra_depth", "roi_left_ratio", "ASTRA_DEPTH_ROI_LEFT_RATIO")
    _set_env_if_present(parser, "astra_depth", "roi_right_ratio", "ASTRA_DEPTH_ROI_RIGHT_RATIO")
    _set_env_if_present(parser, "astra_depth", "roi_top_ratio", "ASTRA_DEPTH_ROI_TOP_RATIO")
    _set_env_if_present(parser, "astra_depth", "roi_bottom_ratio", "ASTRA_DEPTH_ROI_BOTTOM_RATIO")
    _set_env_if_present(parser, "astra_depth", "min_valid_pixels", "ASTRA_DEPTH_MIN_VALID_PIXELS")
    _set_env_if_present(
        parser,
        "astra_depth",
        "dynamic_min_valid_floor",
        "ASTRA_DEPTH_DYNAMIC_MIN_VALID_FLOOR",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "dynamic_min_valid_fraction",
        "ASTRA_DEPTH_DYNAMIC_MIN_VALID_FRACTION",
    )
    _set_env_if_present(parser, "astra_depth", "median_window", "ASTRA_DEPTH_MEDIAN_WINDOW")
    _set_env_if_present(
        parser,
        "astra_depth",
        "foreground_cluster_span_m",
        "ASTRA_DEPTH_FOREGROUND_CLUSTER_SPAN_M",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "foreground_cluster_min_fraction",
        "ASTRA_DEPTH_FOREGROUND_CLUSTER_MIN_FRACTION",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "foreground_spatial_support_fraction",
        "ASTRA_DEPTH_FOREGROUND_SPATIAL_SUPPORT_FRACTION",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "torso_region_min_size_px",
        "ASTRA_DEPTH_TORSO_REGION_MIN_SIZE_PX",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "torso_region_max_size_px",
        "ASTRA_DEPTH_TORSO_REGION_MAX_SIZE_PX",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "center_patch_size",
        "ASTRA_DEPTH_CENTER_PATCH_SIZE",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "center_patch_keep_count",
        "ASTRA_DEPTH_CENTER_PATCH_KEEP_COUNT",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "center_patch_min_valid_fraction",
        "ASTRA_DEPTH_CENTER_PATCH_MIN_VALID_FRACTION",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "large_bbox_guard_area_ratio",
        "ASTRA_DEPTH_LARGE_BBOX_GUARD_AREA_RATIO",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "large_bbox_guard_height_ratio",
        "ASTRA_DEPTH_LARGE_BBOX_GUARD_HEIGHT_RATIO",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "large_bbox_guard_max_distance_m",
        "ASTRA_DEPTH_LARGE_BBOX_GUARD_MAX_DISTANCE_M",
    )
    _set_env_if_present(parser, "astra_depth", "max_distance_jump_m", "ASTRA_DEPTH_MAX_DISTANCE_JUMP_M")
    _set_env_if_present(parser, "astra_depth", "jump_confirm_frames", "ASTRA_DEPTH_JUMP_CONFIRM_FRAMES")
    _set_env_if_present(
        parser,
        "astra_depth",
        "max_unconfirmed_jump_rate_m_s",
        "ASTRA_DEPTH_MAX_UNCONFIRMED_JUMP_RATE_M_S",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "near_guard_distance_m",
        "ASTRA_DEPTH_NEAR_GUARD_DISTANCE_M",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "near_far_jump_confirm_frames",
        "ASTRA_DEPTH_NEAR_FAR_JUMP_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "anchor_strict_age_sec",
        "ASTRA_DEPTH_ANCHOR_STRICT_AGE_SEC",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "anchor_expire_age_sec",
        "ASTRA_DEPTH_ANCHOR_EXPIRE_AGE_SEC",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "reanchor_confirm_frames",
        "ASTRA_DEPTH_REANCHOR_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "motion_confirm_frames",
        "ASTRA_DEPTH_MOTION_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "motion_reverse_min_m",
        "ASTRA_DEPTH_MOTION_REVERSE_MIN_M",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "near_far_jump_max_bbox_ratio",
        "ASTRA_DEPTH_NEAR_FAR_JUMP_MAX_BBOX_RATIO",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "near_far_jump_edge_margin_ratio",
        "ASTRA_DEPTH_NEAR_FAR_JUMP_EDGE_MARGIN_RATIO",
    )
    _set_env_if_present(parser, "astra_depth", "log_every_sec", "ASTRA_DEPTH_LOG_EVERY_SEC")
    _set_bool_env_if_present(
        parser,
        "astra_depth",
        "longitudinal_control_enable",
        "ASTRA_DEPTH_LONGITUDINAL_CONTROL_ENABLE",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "longitudinal_control_hz",
        "ASTRA_DEPTH_LONGITUDINAL_CONTROL_HZ",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "longitudinal_bbox_max_age_sec",
        "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "longitudinal_max_forward_percent",
        "ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT",
    )
    # Forward grant TTL only; fresh PI integration/ROI/reverse clocks stay separate.
    _set_env_if_present(parser, "astra_depth", "longitudinal_sample_max_age_sec",
                        "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC")
    _set_env_if_present(
        parser,
        "astra_depth",
        "longitudinal_far_distance_m",
        "ASTRA_DEPTH_LONGITUDINAL_FAR_DISTANCE_M",
    )
    _set_env_if_present(
        parser,
        "astra_depth",
        "longitudinal_far_forward_percent",
        "ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT",
    )
    _set_env_if_present(
        parser,
        "vision",
        "control_max_result_age_sec",
        "VISION_CONTROL_MAX_RESULT_AGE_SEC",
    )

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
    _set_env_if_present(
        parser,
        "vision",
        "search_diagnostic_conf_threshold",
        "RKNN_SEARCH_DIAGNOSTIC_CONF_THRESHOLD",
    )
    _set_bool_env_if_present(
        parser,
        "vision",
        "search_probe_cluster_enable",
        "RKNN_SEARCH_PROBE_CLUSTER_ENABLE",
    )
    _set_env_if_present(
        parser,
        "vision",
        "search_probe_cluster_iou_threshold",
        "RKNN_SEARCH_PROBE_CLUSTER_IOU_THRESHOLD",
    )
    _set_env_if_present(
        parser,
        "vision",
        "search_probe_cluster_center_distance_ratio",
        "RKNN_SEARCH_PROBE_CLUSTER_CENTER_DISTANCE_RATIO",
    )
    _set_env_if_present(
        parser,
        "vision",
        "search_probe_cluster_min_score_gap",
        "RKNN_SEARCH_PROBE_CLUSTER_MIN_SCORE_GAP",
    )
    _set_env_if_present(parser, "vision", "target_select", "VISION_TARGET_SELECT")
    _set_bool_env_if_present(parser, "vision", "control_use_predicted_tracks", "VISION_CONTROL_USE_PREDICTED_TRACKS")
    _set_bool_env_if_present(
        parser,
        "vision",
        "single_person_geometry_fallback_enable",
        "SINGLE_PERSON_GEOMETRY_FALLBACK_ENABLE",
    )
    _set_env_if_present(
        parser,
        "vision",
        "single_person_geometry_confirm_frames",
        "SINGLE_PERSON_GEOMETRY_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "vision",
        "single_person_geometry_max_gap_frames",
        "SINGLE_PERSON_GEOMETRY_MAX_GAP_FRAMES",
    )
    _set_env_if_present(
        parser,
        "vision",
        "single_person_geometry_min_iou",
        "SINGLE_PERSON_GEOMETRY_MIN_IOU",
    )
    _set_env_if_present(
        parser,
        "vision",
        "single_person_geometry_max_center_jump_ratio",
        "SINGLE_PERSON_GEOMETRY_MAX_CENTER_JUMP_RATIO",
    )
    _set_env_if_present(
        parser,
        "vision",
        "visible_low_quality_steer_max_correction_rpm",
        "VISIBLE_LOW_QUALITY_STEER_MAX_CORRECTION_RPM",
    )
    _set_env_if_present(
        parser,
        "vision",
        "visible_low_quality_steer_edge_max_correction_rpm",
        "VISIBLE_LOW_QUALITY_STEER_EDGE_MAX_CORRECTION_RPM",
    )
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
    _set_bool_env_if_present(parser, "vision", "reid_color_fusion_enable", "RKNN_REID_COLOR_FUSION_ENABLE")
    _set_env_if_present(parser, "vision", "reid_color_fusion_weight", "RKNN_REID_COLOR_FUSION_WEIGHT")
    _set_bool_env_if_present(parser, "vision", "reid_partial_appearance_enable", "RKNN_REID_PARTIAL_APPEARANCE_ENABLE")
    _set_bool_env_if_present(parser, "vision", "reid_partial_osnet_enable", "RKNN_REID_PARTIAL_OSNET_ENABLE")
    _set_bool_env_if_present(parser, "vision", "reid_diagnostics_enable", "RKNN_REID_DIAGNOSTICS_ENABLE")
    _set_env_if_present(parser, "vision", "reid_diagnostics_max_samples", "RKNN_REID_DIAGNOSTICS_MAX_SAMPLES")
    _set_env_if_present(parser, "vision", "reid_diagnostics_queue_capacity", "RKNN_REID_DIAGNOSTICS_QUEUE_CAPACITY")
    _set_env_if_present(parser, "vision", "reid_diagnostics_mapped_interval", "RKNN_REID_DIAGNOSTICS_MAPPED_INTERVAL")

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
    _set_env_if_present(parser, "identity_bank", "max_weak_features", "Y8_IDENTITY_MAX_WEAK_FEATURES")
    _set_env_if_present(
        parser,
        "identity_bank",
        "diversity_min_distance",
        "Y8_IDENTITY_DIVERSITY_MIN_DISTANCE",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "diversity_replace_margin",
        "Y8_IDENTITY_DIVERSITY_REPLACE_MARGIN",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "weak_update_threshold",
        "Y8_IDENTITY_WEAK_UPDATE_THRESHOLD",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "weak_update_interval",
        "Y8_IDENTITY_WEAK_UPDATE_INTERVAL",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "weak_match_penalty",
        "Y8_IDENTITY_WEAK_MATCH_PENALTY",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "weak_reacquire_threshold",
        "Y8_IDENTITY_WEAK_REACQUIRE_THRESHOLD",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "weak_reacquire_confirm_frames",
        "Y8_IDENTITY_WEAK_REACQUIRE_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "weak_quality_weight",
        "Y8_IDENTITY_WEAK_QUALITY_WEIGHT",
    )
    _set_env_if_present(parser, "identity_bank", "min_confidence", "Y8_IDENTITY_MIN_CONFIDENCE")
    _set_env_if_present(parser, "identity_bank", "min_area", "Y8_IDENTITY_MIN_AREA")
    _set_env_if_present(parser, "identity_bank", "min_width_px", "Y8_IDENTITY_MIN_WIDTH_PX")
    _set_env_if_present(parser, "identity_bank", "min_height_px", "Y8_IDENTITY_MIN_HEIGHT_PX")
    _set_env_if_present(
        parser,
        "identity_bank",
        "max_single_frame_area_shrink_ratio",
        "Y8_IDENTITY_MAX_SINGLE_FRAME_AREA_SHRINK_RATIO",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "area_shrink_max_gap_frames",
        "Y8_IDENTITY_AREA_SHRINK_MAX_GAP_FRAMES",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "max_center_jump_ratio",
        "Y8_IDENTITY_MAX_CENTER_JUMP_RATIO",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "center_jump_max_gap_frames",
        "Y8_IDENTITY_CENTER_JUMP_MAX_GAP_FRAMES",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "swap_min_mapped_jump_ratio",
        "Y8_IDENTITY_SWAP_MIN_MAPPED_JUMP_RATIO",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "swap_max_replacement_distance_ratio",
        "Y8_IDENTITY_SWAP_MAX_REPLACEMENT_DISTANCE_RATIO",
    )
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
    _set_env_if_present(parser, "identity_bank", "handoff_geometry_max_gap_frames", "Y8_IDENTITY_HANDOFF_GEOMETRY_MAX_GAP_FRAMES")
    _set_env_if_present(parser, "identity_bank", "handoff_geometry_max_center_jump_ratio", "Y8_IDENTITY_HANDOFF_GEOMETRY_MAX_CENTER_JUMP_RATIO")
    _set_env_if_present(parser, "identity_bank", "handoff_geometry_min_area_similarity", "Y8_IDENTITY_HANDOFF_GEOMETRY_MIN_AREA_SIMILARITY")
    _set_env_if_present(
        parser,
        "identity_bank",
        "controlled_handoff_confirm_frames",
        "Y8_IDENTITY_CONTROLLED_HANDOFF_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "controlled_handoff_instant_threshold",
        "Y8_IDENTITY_CONTROLLED_HANDOFF_INSTANT_THRESHOLD",
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
    _set_bool_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_reacquire_enable",
        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_ENABLE",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_reacquire_threshold",
        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_THRESHOLD",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_reacquire_max_disadvantage",
        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_MAX_DISADVANTAGE",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_reacquire_min_confidence",
        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_MIN_CONFIDENCE",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_reacquire_observation_min_confidence",
        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_OBSERVATION_MIN_CONFIDENCE",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_reacquire_max_age_sec",
        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_MAX_AGE_SEC",
    )
    _set_bool_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_reacquire_late_candidate_enable",
        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_LATE_CANDIDATE_ENABLE",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_reacquire_confirm_frames",
        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_reacquire_instant_threshold",
        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_INSTANT_THRESHOLD",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_reacquire_min_score_gap",
        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_MIN_SCORE_GAP",
    )
    _set_bool_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_soft_candidate_enable",
        "Y8_IDENTITY_PREFERRED_SEARCH_SOFT_CANDIDATE_ENABLE",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_soft_candidate_threshold",
        "Y8_IDENTITY_PREFERRED_SEARCH_SOFT_CANDIDATE_THRESHOLD",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_soft_min_score_gap",
        "Y8_IDENTITY_PREFERRED_SEARCH_SOFT_MIN_SCORE_GAP",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_soft_min_area_ratio",
        "Y8_IDENTITY_PREFERRED_SEARCH_SOFT_MIN_AREA_RATIO",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_soft_min_confidence",
        "Y8_IDENTITY_PREFERRED_SEARCH_SOFT_MIN_CONFIDENCE",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "preferred_search_reacquire_side_ratio",
        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_SIDE_RATIO",
    )
    _set_bool_env_if_present(
        parser,
        "identity_bank",
        "partial_appearance_enable",
        "Y8_IDENTITY_PARTIAL_APPEARANCE_ENABLE",
    )
    _set_env_if_present(parser, "identity_bank", "partial_match_threshold", "Y8_IDENTITY_PARTIAL_MATCH_THRESHOLD")
    _set_env_if_present(parser, "identity_bank", "partial_max_features", "Y8_IDENTITY_PARTIAL_MAX_FEATURES")
    _set_env_if_present(parser, "identity_bank", "partial_update_threshold", "Y8_IDENTITY_PARTIAL_UPDATE_THRESHOLD")
    _set_bool_env_if_present(
        parser,
        "identity_bank",
        "duplicate_box_suppression_enable",
        "Y8_IDENTITY_DUPLICATE_BOX_SUPPRESSION_ENABLE",
    )
    _set_env_if_present(parser, "identity_bank", "duplicate_iou_threshold", "Y8_IDENTITY_DUPLICATE_IOU_THRESHOLD")
    _set_env_if_present(
        parser,
        "identity_bank",
        "duplicate_vertical_overlap_threshold",
        "Y8_IDENTITY_DUPLICATE_VERTICAL_OVERLAP_THRESHOLD",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "duplicate_horizontal_overlap_threshold",
        "Y8_IDENTITY_DUPLICATE_HORIZONTAL_OVERLAP_THRESHOLD",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "duplicate_large_height_ratio",
        "Y8_IDENTITY_DUPLICATE_LARGE_HEIGHT_RATIO",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "duplicate_large_width_ratio",
        "Y8_IDENTITY_DUPLICATE_LARGE_WIDTH_RATIO",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "duplicate_max_area_ratio",
        "Y8_IDENTITY_DUPLICATE_MAX_AREA_RATIO",
    )
    _set_env_if_present(
        parser,
        "identity_bank",
        "duplicate_bottom_gap_ratio",
        "Y8_IDENTITY_DUPLICATE_BOTTOM_GAP_RATIO",
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
    _set_bool_env_if_present(
        parser,
        "distance",
        "auto_tune_follow_distance",
        "FOLLOW_DISTANCE_AUTO_TUNE",
    )
    _set_env_if_present(parser, "distance", "target_distance_release_m", "TARGET_DISTANCE_RELEASE_M")
    _set_env_if_present(
        parser,
        "distance",
        "target_distance_release_hold_sec",
        "TARGET_DISTANCE_RELEASE_HOLD_SEC",
    )
    _set_env_if_present(
        parser,
        "distance",
        "target_distance_release_confirm_frames",
        "TARGET_DISTANCE_RELEASE_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "distance",
        "target_distance_release_visual_shrink_ratio",
        "TARGET_DISTANCE_RELEASE_VISUAL_SHRINK_RATIO",
    )
    _set_env_if_present(
        parser,
        "distance",
        "target_distance_release_visual_edge_margin_ratio",
        "TARGET_DISTANCE_RELEASE_VISUAL_EDGE_MARGIN_RATIO",
    )
    _set_env_if_present(parser, "distance", "brake_distance_m", "FOLLOW_BRAKE_DISTANCE_M")
    _set_bool_env_if_present(parser, "distance", "reverse_enable", "FOLLOW_REVERSE_ENABLE")
    _set_bool_env_if_present(
        parser,
        "distance",
        "near_distance_rotate_only_enable",
        "FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_ENABLE",
    )
    _set_env_if_present(
        parser,
        "distance",
        "near_distance_rotate_only_distance_m",
        "FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_DISTANCE_M",
    )
    _set_env_if_present(
        parser,
        "distance",
        "near_distance_rotation_only_max_rpm",
        "FOLLOW_NEAR_DISTANCE_ROTATION_ONLY_MAX_RPM",
    )
    _set_env_if_present(
        parser,
        "distance",
        "near_distance_settle_confirm_frames",
        "FOLLOW_NEAR_DISTANCE_SETTLE_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "distance",
        "near_distance_settle_hold_sec",
        "FOLLOW_NEAR_DISTANCE_SETTLE_HOLD_SEC",
    )
    _set_env_if_present(
        parser,
        "distance",
        "near_distance_settle_release_margin_ratio",
        "FOLLOW_NEAR_DISTANCE_SETTLE_RELEASE_MARGIN_RATIO",
    )
    _set_env_if_present(
        parser,
        "distance",
        "near_distance_settle_release_frames",
        "FOLLOW_NEAR_DISTANCE_SETTLE_RELEASE_FRAMES",
    )
    _set_bool_env_if_present(
        parser,
        "distance",
        "near_distance_disable_rate_feedforward",
        "FOLLOW_NEAR_DISTANCE_DISABLE_RATE_FEEDFORWARD",
    )
    _set_env_if_present(parser, "distance", "reverse_start_distance_m", "FOLLOW_REVERSE_START_DISTANCE_M")
    _set_env_if_present(
        parser,
        "distance",
        "reverse_immediate_distance_m",
        "FOLLOW_REVERSE_IMMEDIATE_DISTANCE_M",
    )
    _set_env_if_present(parser, "distance", "reverse_stop_distance_m", "FOLLOW_REVERSE_STOP_DISTANCE_M")
    _set_env_if_present(
        parser,
        "distance",
        "reverse_full_speed_distance_m",
        "FOLLOW_REVERSE_FULL_SPEED_DISTANCE_M",
    )
    _set_env_if_present(parser, "distance", "reverse_min_rpm", "FOLLOW_REVERSE_MIN_RPM")
    _set_env_if_present(parser, "distance", "reverse_max_rpm", "FOLLOW_REVERSE_MAX_RPM")
    _set_env_if_present(
        parser,
        "distance",
        "reverse_feedforward_floor_rpm",
        "FOLLOW_REVERSE_FEEDFORWARD_FLOOR_RPM",
    )
    _set_env_if_present(
        parser,
        "distance",
        "reverse_feedforward_gain_rpm_per_m_s",
        "FOLLOW_REVERSE_FEEDFORWARD_GAIN_RPM_PER_M_S",
    )
    _set_env_if_present(
        parser,
        "distance",
        "reverse_runtime_cap_rpm",
        "FOLLOW_REVERSE_RUNTIME_CAP_RPM",
    )
    _set_env_if_present(
        parser,
        "distance",
        "reverse_approach_speed_filter_alpha",
        "FOLLOW_REVERSE_APPROACH_SPEED_FILTER_ALPHA",
    )
    _set_env_if_present(
        parser,
        "distance",
        "reverse_min_approach_delta_m",
        "FOLLOW_REVERSE_MIN_APPROACH_DELTA_M",
    )
    _set_env_if_present(parser, "distance", "reverse_confirm_frames", "FOLLOW_REVERSE_CONFIRM_FRAMES")
    _set_env_if_present(
        parser,
        "distance",
        "reverse_radar_max_age_sec",
        "FOLLOW_REVERSE_RADAR_MAX_AGE_SEC",
    )
    _set_env_if_present(
        parser,
        "distance",
        "reverse_distance_missing_hold_sec",
        "FOLLOW_REVERSE_DISTANCE_MISSING_HOLD_SEC",
    )
    _set_env_if_present(
        parser,
        "distance",
        "reverse_visual_guard_area_ratio",
        "FOLLOW_REVERSE_VISUAL_GUARD_AREA_RATIO",
    )
    _set_env_if_present(
        parser,
        "distance",
        "reverse_visual_guard_height_ratio",
        "FOLLOW_REVERSE_VISUAL_GUARD_HEIGHT_RATIO",
    )
    _set_env_if_present(
        parser,
        "distance",
        "reverse_visual_guard_growth_ratio",
        "FOLLOW_REVERSE_VISUAL_GUARD_GROWTH_RATIO",
    )
    _set_env_if_present(
        parser,
        "distance",
        "reverse_visual_guard_growth_min_area_ratio",
        "FOLLOW_REVERSE_VISUAL_GUARD_GROWTH_MIN_AREA_RATIO",
    )
    _set_env_if_present(
        parser,
        "distance",
        "reverse_visual_guard_max_distance_m",
        "FOLLOW_REVERSE_VISUAL_GUARD_MAX_DISTANCE_M",
    )
    _set_env_if_present(
        parser,
        "distance",
        "reverse_visual_guard_rpm",
        "FOLLOW_REVERSE_VISUAL_GUARD_RPM",
    )
    _set_env_if_present(
        parser,
        "distance",
        "forward_start_distance_m",
        "FOLLOW_FORWARD_START_DISTANCE_M",
    )
    _set_env_if_present(
        parser,
        "distance",
        "forward_stop_distance_m",
        "FOLLOW_FORWARD_STOP_DISTANCE_M",
    )
    _set_env_if_present(parser, "distance", "mmwave_obstacle_threshold_m", "MMWAVE_OBSTACLE_THRESHOLD")
    _set_env_if_present(parser, "distance", "source", "DISTANCE_SOURCE")
    _set_bool_env_if_present(parser, "distance", "parking_enable", "DISTANCE_PARKING_ENABLE")
    _set_env_if_present(parser, "distance", "fallback_forward_percent", "DISTANCE_MISSING_FORWARD_PERCENT")

    # Longitudinal cascade outer loop: depth distance error -> target wheel RPM.
    os.environ["DISTANCE_CONTROL_MODE"] = control_mode
    for key in ("kp_per_sec", "ki_per_sec2", "integral_max_m_s", "memory_sec", "launch_request_rpm", "motion_memory_sec"):
        _set_env_if_present(parser, "distance_pid", "pi_" + key, "DISTANCE_PI_" + key.upper())
    _set_bool_env_if_present(parser, "distance_pid", "enable", "DISTANCE_PID_ENABLE")
    _set_env_if_present(parser, "distance_pid", "kp_rpm_per_m", "DISTANCE_PID_KP_RPM_PER_M")
    _set_env_if_present(parser, "distance_pid", "ki_rpm_per_m_s", "DISTANCE_PID_KI_RPM_PER_M_S")
    _set_env_if_present(parser, "distance_pid", "forward_integral_limit_m_s", "DISTANCE_PID_FORWARD_INTEGRAL_LIMIT_M_S")
    _set_env_if_present(parser, "distance_pid", "kd_rpm_s_per_m", "DISTANCE_PID_KD_RPM_S_PER_M")
    _set_env_if_present(
        parser,
        "distance_pid",
        "integral_limit_m_s",
        "DISTANCE_PID_INTEGRAL_LIMIT_M_S",
    )
    _set_env_if_present(parser, "distance_pid", "deadband_m", "DISTANCE_PID_DEADBAND_M")
    _set_bool_env_if_present(parser, "distance_pid", "feedforward_enable", "DISTANCE_FEEDFORWARD_ENABLE")
    _set_bool_env_if_present(parser, "distance_pid", "approach_enable", "DISTANCE_APPROACH_ENABLE")
    _set_bool_env_if_present(parser, "distance_pid", "approach_matching_enable", "DISTANCE_APPROACH_MATCHING_ENABLE")
    for key in ("gain_per_sec", "max_catchup_m_s", "deceleration_m_s2", "response_delay_sec", "no_matching_max_rpm"):
        _set_env_if_present(parser, "distance_pid", "approach_" + key, "DISTANCE_APPROACH_" + key.upper())
    _set_bool_env_if_present(parser, "distance_pid", "turn_compensation_enable", "DISTANCE_TURN_COMPENSATION_ENABLE")
    _set_bool_env_if_present(parser, "distance_pid", "measured_recovery_enable", "DEPTH_MEASURED_RECOVERY_ENABLE")
    _set_env_if_present(parser, "distance_pid", "feedforward_max_rpm", "DISTANCE_FEEDFORWARD_MAX_RPM")
    _set_env_if_present(parser, "distance_pid", "matching_base_max_rpm", "DISTANCE_MATCHING_BASE_MAX_RPM")
    _set_env_if_present(
        parser,
        "distance_pid",
        "derivative_filter_alpha",
        "DISTANCE_PID_DERIVATIVE_FILTER_ALPHA",
    )
    _set_env_if_present(
        parser,
        "distance_pid",
        "max_measurement_jump_m",
        "DISTANCE_PID_MAX_MEASUREMENT_JUMP_M",
    )
    _set_env_if_present(
        parser,
        "distance_pid",
        "output_rise_rpm_per_sec",
        "DISTANCE_PID_OUTPUT_RISE_RPM_PER_SEC",
    )
    _set_env_if_present(
        parser,
        "distance_pid",
        "output_fall_rpm_per_sec",
        "DISTANCE_PID_OUTPUT_FALL_RPM_PER_SEC",
    )

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
    _set_env_if_present(parser, "mmwave_match", "unmatched_hold_sec", "VISION_MMWAVE_UNMATCHED_HOLD_SEC")
    _set_env_if_present(
        parser,
        "mmwave_match",
        "unmatched_hold_min_distance_m",
        "VISION_MMWAVE_UNMATCHED_HOLD_MIN_DISTANCE_M",
    )
    _set_env_if_present(parser, "mmwave_match", "far_margin_start_m", "VISION_MMWAVE_FAR_MARGIN_START_M")
    _set_env_if_present(
        parser,
        "mmwave_match",
        "far_angle_margin_extra_deg",
        "VISION_MMWAVE_FAR_ANGLE_MARGIN_EXTRA_DEG",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "max_center_angle_diff_deg",
        "VISION_MMWAVE_MAX_CENTER_ANGLE_DIFF_DEG",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "max_distance_jump_m",
        "VISION_MMWAVE_MAX_DISTANCE_JUMP_M",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "max_angle_jump_deg",
        "VISION_MMWAVE_MAX_ANGLE_JUMP_DEG",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "switch_confirm_frames",
        "VISION_MMWAVE_SWITCH_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "continuity_memory_sec",
        "VISION_MMWAVE_CONTINUITY_MEMORY_SEC",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "distance_score_weight",
        "VISION_MMWAVE_DISTANCE_SCORE_WEIGHT",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "motion_min_angle_delta_deg",
        "VISION_MMWAVE_MOTION_MIN_ANGLE_DELTA_DEG",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "motion_hint_ttl_sec",
        "VISION_MMWAVE_MOTION_HINT_TTL_SEC",
    )
    _set_bool_env_if_present(parser, "mmwave_match", "fusion_enable", "VISION_MMWAVE_FUSION_ENABLE")
    _set_env_if_present(
        parser,
        "mmwave_match",
        "fusion_radar_median_window",
        "VISION_MMWAVE_FUSION_RADAR_MEDIAN_WINDOW",
    )
    _set_env_if_present(parser, "mmwave_match", "fusion_visual_weight", "VISION_MMWAVE_FUSION_VISUAL_WEIGHT")
    _set_env_if_present(
        parser,
        "mmwave_match",
        "fusion_encoder_wheel_circumference_m",
        "VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "fusion_encoder_max_step_m",
        "VISION_MMWAVE_FUSION_ENCODER_MAX_STEP_M",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "fusion_bbox_min_height_ratio",
        "VISION_MMWAVE_FUSION_BBOX_MIN_HEIGHT_RATIO",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "fusion_bbox_max_height_ratio",
        "VISION_MMWAVE_FUSION_BBOX_MAX_HEIGHT_RATIO",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "fusion_radar_recovery_alpha",
        "VISION_MMWAVE_FUSION_RADAR_RECOVERY_ALPHA",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "fusion_max_distance_increase_mps",
        "VISION_MMWAVE_FUSION_MAX_DISTANCE_INCREASE_MPS",
    )
    _set_env_if_present(
        parser,
        "mmwave_match",
        "fusion_min_confidence",
        "VISION_MMWAVE_FUSION_MIN_CONFIDENCE",
    )

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
    _set_env_if_present(parser, "motor", "forward_max_target_rpm", "MOTOR_FORWARD_MAX_TARGET_RPM")
    _set_env_if_present(
        parser,
        "motor",
        "forward_update_min_delta_rpm",
        "MOTOR_FORWARD_UPDATE_MIN_DELTA_RPM",
    )
    _set_env_if_present(parser, "motor", "steer_raw_target", "MOTOR_STEER_RAW_TARGET")
    _set_env_if_present(parser, "motor", "rotate_raw_target", "MOTOR_ROTATE_RAW_TARGET")
    _set_env_if_present(parser, "motor", "left_sign", "MOTOR_LEFT_SIGN")
    _set_env_if_present(parser, "motor", "right_sign", "MOTOR_RIGHT_SIGN")
    _set_env_if_present(parser, "motor", "forward_target_sign", "MOTOR_FORWARD_TARGET_SIGN")
    _set_env_if_present(parser, "motor", "parking_current_a", "MOTOR_PARKING_CURRENT_A")
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
    _set_env_if_present(parser, "motion", "mmwave_hold_forward_percent", "VISION_MMWAVE_HOLD_FORWARD_PERCENT")
    _set_env_if_present(
        parser,
        "motion",
        "mmwave_hold_decel_step_percent",
        "VISION_MMWAVE_HOLD_DECEL_STEP_PERCENT",
    )
    _set_env_if_present(
        parser,
        "motion",
        "mmwave_hold_recover_step_percent",
        "VISION_MMWAVE_HOLD_RECOVER_STEP_PERCENT",
    )
    _set_env_if_present(
        parser,
        "motion",
        "mmwave_fusion_low_confidence_forward_percent",
        "VISION_MMWAVE_FUSION_LOW_CONFIDENCE_FORWARD_PERCENT",
    )
    _set_env_if_present(parser, "motion", "forward_speed_le_1_3_percent", "FORWARD_SPEED_LE_1_3_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_1_7_percent", "FORWARD_SPEED_LE_1_7_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_2_1_percent", "FORWARD_SPEED_LE_2_1_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_2_6_percent", "FORWARD_SPEED_LE_2_6_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_3_2_percent", "FORWARD_SPEED_LE_3_2_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_3_8_percent", "FORWARD_SPEED_LE_3_8_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_le_4_5_percent", "FORWARD_SPEED_LE_4_5_PERCENT")
    _set_env_if_present(parser, "motion", "forward_speed_far_percent", "FORWARD_SPEED_FAR_PERCENT")
    _set_env_if_present(parser, "motion", "forward_min_rpm", "FORWARD_MIN_RPM")
    _set_env_if_present(parser, "motion", "forward_max_rpm", "FORWARD_MAX_RPM")
    _set_env_if_present(parser, "motion", "forward_curve_max_distance_m", "FORWARD_CURVE_MAX_DISTANCE_M")
    _set_env_if_present(parser, "motion", "forward_curve_exponent", "FORWARD_CURVE_EXPONENT")
    _set_env_if_present(parser, "motion", "steer_percent_limit", "STEER_PERCENT_LIMIT")
    _set_env_if_present(parser, "motion", "rotate_duration", "ROTATE_DURATION")
    _set_bool_env_if_present(parser, "motion", "rotate_pulse_brake_enable", "ROTATE_PULSE_BRAKE_ENABLE")
    _set_env_if_present(parser, "motion", "rotate_pulse_stop_mode", "ROTATE_PULSE_STOP_MODE")
    _set_env_if_present(parser, "motion", "rotate_pulse_transition_rpm", "ROTATE_PULSE_TRANSITION_RPM")
    _set_env_if_present(parser, "motion", "rotate_pulse_pause_sec", "ROTATE_PULSE_PAUSE_SEC")
    _set_env_if_present(
        parser,
        "motion",
        "rotate_pulse_observe_min_frames",
        "ROTATE_PULSE_OBSERVE_MIN_FRAMES",
    )
    _set_bool_env_if_present(
        parser,
        "motion",
        "rotate_pulse_settle_enable",
        "ROTATE_PULSE_SETTLE_ENABLE",
    )
    _set_env_if_present(
        parser,
        "motion",
        "rotate_pulse_settle_quiet_sec",
        "ROTATE_PULSE_SETTLE_QUIET_SEC",
    )
    _set_env_if_present(
        parser,
        "motion",
        "rotate_pulse_settle_timeout_sec",
        "ROTATE_PULSE_SETTLE_TIMEOUT_SEC",
    )
    _set_env_if_present(
        parser,
        "motion",
        "rotate_pulse_settle_feedback_stale_sec",
        "ROTATE_PULSE_SETTLE_FEEDBACK_STALE_SEC",
    )
    _set_env_if_present(
        parser,
        "motion",
        "rotate_pulse_settle_max_wheel_rpm",
        "ROTATE_PULSE_SETTLE_MAX_WHEEL_RPM",
    )
    _set_env_if_present(
        parser,
        "motion",
        "rotate_pulse_settle_max_yaw_rate_dps",
        "ROTATE_PULSE_SETTLE_MAX_YAW_RATE_DPS",
    )
    _set_env_if_present(parser, "motion", "rotation_only_yaw_pulse_rpm", "ROTATION_ONLY_YAW_PULSE_RPM")
    _set_env_if_present(parser, "motion", "rotation_only_yaw_pulse_min_sec", "ROTATION_ONLY_YAW_PULSE_MIN_SEC")
    _set_env_if_present(parser, "motion", "rotation_only_yaw_pulse_max_sec", "ROTATION_ONLY_YAW_PULSE_MAX_SEC")
    _set_env_if_present(parser, "motion", "rotation_only_yaw_brake_sec", "ROTATION_ONLY_YAW_BRAKE_SEC")
    _set_env_if_present(parser, "motion", "rotation_only_yaw_response_dps", "ROTATION_ONLY_YAW_RESPONSE_DPS")
    _set_env_if_present(parser, "motion", "rotation_only_yaw_zero_gap_sec", "ROTATION_ONLY_YAW_ZERO_GAP_SEC")
    _set_bool_env_if_present(
        parser,
        "motion",
        "rotate_pulse_active_brake_enable",
        "ROTATE_PULSE_ACTIVE_BRAKE_ENABLE",
    )
    _set_env_if_present(parser, "motion", "rotate_pulse_active_brake_rpm", "ROTATE_PULSE_ACTIVE_BRAKE_RPM")
    _set_env_if_present(parser, "motion", "rotate_pulse_active_brake_sec", "ROTATE_PULSE_ACTIVE_BRAKE_SEC")
    _set_env_if_present(parser, "motion", "rotate_hold_stale_sec", "ROTATE_HOLD_STALE_SEC")
    _set_env_if_present(parser, "motion", "rotate_turn_percent_from_forward", "ROTATE_TURN_PERCENT_FROM_FORWARD")
    _set_env_if_present(parser, "motion", "rotate_turn_percent_chain", "ROTATE_TURN_PERCENT_CHAIN")
    _set_env_if_present(parser, "motion", "rotate_raw_target_visible", "ROTATE_RAW_TARGET_VISIBLE")
    _set_env_if_present(parser, "motion", "rotate_raw_target_lost_wait", "ROTATE_RAW_TARGET_LOST_WAIT")
    _set_env_if_present(parser, "motion", "rotate_raw_target_search", "ROTATE_RAW_TARGET_SEARCH")
    _set_env_if_present(parser, "motion", "visible_steer_fine_inner_ratio_percent", "VISIBLE_STEER_FINE_INNER_RATIO_PERCENT")
    _set_env_if_present(parser, "motion", "visible_steer_fine_outer_ratio_percent", "VISIBLE_STEER_FINE_OUTER_RATIO_PERCENT")
    _set_env_if_present(parser, "motion", "visible_steer_inner_ratio_percent", "VISIBLE_STEER_INNER_RATIO_PERCENT")
    _set_env_if_present(parser, "motion", "visible_steer_outer_ratio_percent", "VISIBLE_STEER_OUTER_RATIO_PERCENT")
    _set_env_if_present(parser, "motion", "visible_steer_strong_inner_ratio_percent", "VISIBLE_STEER_STRONG_INNER_RATIO_PERCENT")
    _set_env_if_present(parser, "motion", "visible_steer_strong_outer_ratio_percent", "VISIBLE_STEER_STRONG_OUTER_RATIO_PERCENT")
    _set_env_if_present(parser, "motion", "visible_steer_strong_margin_ratio", "VISIBLE_STEER_STRONG_MARGIN_RATIO")
    _set_env_if_present(parser, "motion", "visible_motion_history_frames", "VISIBLE_MOTION_HISTORY_FRAMES")
    _set_env_if_present(parser, "motion", "visible_motion_lookback_sec", "VISIBLE_MOTION_LOOKBACK_SEC")
    _set_env_if_present(parser, "motion", "visible_motion_rate_filter_alpha", "VISIBLE_MOTION_RATE_FILTER_ALPHA")
    _set_env_if_present(parser, "motion", "visible_motion_min_ratio", "VISIBLE_MOTION_MIN_RATIO")
    _set_env_if_present(parser, "motion", "visible_motion_projection_gain", "VISIBLE_MOTION_PROJECTION_GAIN")
    _set_env_if_present(parser, "motion", "visible_motion_strong_ratio", "VISIBLE_MOTION_STRONG_RATIO")
    _set_env_if_present(parser, "motion", "brake_hold_refresh_sec", "BRAKE_HOLD_REFRESH_INTERVAL_SEC")

    # Follow policy.  These are separate from motion speed knobs because they
    # control the target-selection/search state machine.
    _set_bool_env_if_present(parser, "follow", "rotation_only", "FOLLOW_ROTATION_ONLY")
    _set_env_if_present(parser, "follow", "center_left_ratio", "FOLLOW_CENTER_LEFT_RATIO")
    _set_env_if_present(parser, "follow", "center_right_ratio", "FOLLOW_CENTER_RIGHT_RATIO")
    _set_env_if_present(parser, "follow", "center_deadzone_ratio", "FOLLOW_CENTER_DEADZONE_RATIO")
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
    _set_env_if_present(parser, "follow", "search_revolution_deg", "FOLLOW_SEARCH_REVOLUTION_DEG")
    _set_env_if_present(
        parser,
        "follow",
        "search_revolution_feedback_stale_sec",
        "FOLLOW_SEARCH_REVOLUTION_FEEDBACK_STALE_SEC",
    )
    _set_bool_env_if_present(
        parser,
        "follow",
        "search_timeout_exit_program",
        "FOLLOW_SEARCH_TIMEOUT_EXIT_PROGRAM",
    )
    _set_bool_env_if_present(
        parser,
        "follow",
        "exit_on_target_loss",
        "FOLLOW_EXIT_ON_TARGET_LOSS",
    )
    _set_env_if_present(parser, "follow", "initial_target_confirm_frames", "FOLLOW_INITIAL_TARGET_CONFIRM_FRAMES")
    _set_env_if_present(
        parser,
        "follow",
        "search_confirmed_reacquire_frames",
        "SEARCH_CONFIRMED_REACQUIRE_FRAMES",
    )
    _set_bool_env_if_present(
        parser,
        "follow",
        "visual_reacquire_hold_enable",
        "VISUAL_REACQUIRE_HOLD_ENABLE",
    )
    _set_env_if_present(
        parser,
        "follow",
        "visual_reacquire_hold_sec",
        "VISUAL_REACQUIRE_HOLD_SEC",
    )
    _set_env_if_present(
        parser,
        "follow",
        "visual_reacquire_hold_max_center_jump_ratio",
        "VISUAL_REACQUIRE_HOLD_MAX_CENTER_JUMP_RATIO",
    )
    _set_env_if_present(
        parser,
        "follow",
        "visual_reacquire_hold_min_area_similarity",
        "VISUAL_REACQUIRE_HOLD_MIN_AREA_SIMILARITY",
    )
    _set_bool_env_if_present(
        parser,
        "follow",
        "search_evidence_gate_enable",
        "SEARCH_EVIDENCE_GATE_ENABLE",
    )
    _set_bool_env_if_present(parser, "follow", "search_evidence_retry_enable", "SEARCH_EVIDENCE_RETRY_ENABLE")
    _set_env_if_present(
        parser,
        "follow",
        "search_evidence_hold_frames",
        "SEARCH_EVIDENCE_HOLD_FRAMES",
    )
    _set_env_if_present(
        parser,
        "follow",
        "search_evidence_max_hold_sec",
        "SEARCH_EVIDENCE_MAX_HOLD_SEC",
    )
    _set_env_if_present(
        parser,
        "follow",
        "search_evidence_probe_confirm_frames",
        "SEARCH_EVIDENCE_PROBE_CONFIRM_FRAMES",
    )
    _set_env_if_present(
        parser,
        "follow",
        "search_evidence_probe_min_score",
        "SEARCH_EVIDENCE_PROBE_MIN_SCORE",
    )
    _set_env_if_present(
        parser,
        "follow",
        "search_evidence_min_area_ratio",
        "SEARCH_EVIDENCE_MIN_AREA_RATIO",
    )
    _set_env_if_present(
        parser,
        "follow",
        "search_evidence_max_area_ratio",
        "SEARCH_EVIDENCE_MAX_AREA_RATIO",
    )
    _set_env_if_present(
        parser,
        "follow",
        "search_evidence_consistency_iou",
        "SEARCH_EVIDENCE_CONSISTENCY_IOU",
    )
    _set_env_if_present(
        parser,
        "follow",
        "search_candidate_untracked_min_score",
        "SEARCH_CANDIDATE_UNTRACKED_MIN_SCORE",
    )
    _set_env_if_present(
        parser,
        "follow",
        "search_candidate_approach_margin_ratio",
        "SEARCH_CANDIDATE_APPROACH_MARGIN_RATIO",
    )
    _set_env_if_present(
        parser,
        "follow",
        "search_candidate_acquire_raw_rpm",
        "SEARCH_CANDIDATE_ACQUIRE_RAW_RPM",
    )
    _set_bool_env_if_present(
        parser,
        "follow",
        "search_rotate_continuous_enable",
        "SEARCH_ROTATE_CONTINUOUS_ENABLE",
    )
    _set_bool_env_if_present(parser, "follow", "release_target_on_lost", "FOLLOW_RELEASE_TARGET_ON_LOST")
    _set_env_if_present(parser, "follow", "lost_confirm_sec", "FOLLOW_LOST_CONFIRM_SEC")
    _set_env_if_present(parser, "follow", "lost_confirm_frames", "FOLLOW_LOST_CONFIRM_FRAMES")
    _set_bool_env_if_present(
        parser,
        "follow",
        "stale_direction_recovery_enable",
        "FOLLOW_STALE_DIRECTION_RECOVERY_ENABLE",
    )
    _set_bool_env_if_present(
        parser,
        "follow",
        "direction_history_enable",
        "FOLLOW_DIRECTION_HISTORY_ENABLE",
    )
    _set_bool_env_if_present(
        parser,
        "follow",
        "historical_direction_backfill_enable",
        "HISTORICAL_DIRECTION_BACKFILL_ENABLE",
    )
    _set_env_if_present(
        parser,
        "follow",
        "historical_direction_backfill_max_age_sec",
        "HISTORICAL_DIRECTION_BACKFILL_MAX_AGE_SEC",
    )
    _set_env_if_present(
        parser,
        "follow",
        "historical_direction_backfill_min_samples",
        "HISTORICAL_DIRECTION_BACKFILL_MIN_SAMPLES",
    )
    _set_env_if_present(
        parser,
        "follow",
        "historical_direction_backfill_max_capture_gap",
        "HISTORICAL_DIRECTION_BACKFILL_MAX_CAPTURE_GAP",
    )
    _set_env_if_present(
        parser,
        "follow",
        "historical_direction_backfill_max_center_jump_ratio",
        "HISTORICAL_DIRECTION_BACKFILL_MAX_CENTER_JUMP_RATIO",
    )
    _set_env_if_present(
        parser,
        "follow",
        "historical_direction_backfill_min_area_similarity",
        "HISTORICAL_DIRECTION_BACKFILL_MIN_AREA_SIMILARITY",
    )
    _set_env_if_present(
        parser,
        "follow",
        "historical_direction_backfill_confidence_cap",
        "HISTORICAL_DIRECTION_BACKFILL_CONFIDENCE_CAP",
    )
    _set_env_if_present(
        parser,
        "follow",
        "historical_direction_backfill_min_score",
        "HISTORICAL_DIRECTION_BACKFILL_MIN_SCORE",
    )
    _set_env_if_present(
        parser,
        "follow",
        "stale_direction_observe_frames",
        "FOLLOW_STALE_DIRECTION_OBSERVE_FRAMES",
    )
    _set_env_if_present(
        parser,
        "follow",
        "stale_direction_observe_max_sec",
        "FOLLOW_STALE_DIRECTION_OBSERVE_MAX_SEC",
    )
    _set_env_if_present(
        parser,
        "follow",
        "stale_direction_settle_yaw_rate_dps",
        "FOLLOW_STALE_DIRECTION_SETTLE_YAW_RATE_DPS",
    )
    _set_bool_env_if_present(
        parser,
        "follow",
        "stale_direction_probe_enable",
        "FOLLOW_STALE_DIRECTION_PROBE_ENABLE",
    )
    _set_env_if_present(
        parser,
        "follow",
        "stale_direction_probe_angle_deg",
        "FOLLOW_STALE_DIRECTION_PROBE_ANGLE_DEG",
    )
    _set_env_if_present(
        parser,
        "follow",
        "stale_direction_probe_observe_frames",
        "FOLLOW_STALE_DIRECTION_PROBE_OBSERVE_FRAMES",
    )
    _set_env_if_present(
        parser,
        "follow",
        "stale_direction_probe_return_tolerance_deg",
        "FOLLOW_STALE_DIRECTION_PROBE_RETURN_TOLERANCE_DEG",
    )
    _set_env_if_present(
        parser,
        "follow",
        "stale_direction_probe_raw_target",
        "FOLLOW_STALE_DIRECTION_PROBE_RAW_TARGET",
    )
    _set_env_if_present(parser, "follow", "action_cooldown_frames", "FOLLOW_ACTION_COOLDOWN_FRAMES")
    _set_env_if_present(parser, "follow", "steer_min_hold_sec", "FOLLOW_STEER_MIN_HOLD_SEC")
    _set_env_if_present(parser, "follow", "steer_lost_hold_frames", "FOLLOW_STEER_LOST_HOLD_FRAMES")
    _set_env_if_present(parser, "follow", "steer_lost_hold_max_sec", "FOLLOW_STEER_LOST_HOLD_MAX_SEC")
    _set_env_if_present(parser, "follow", "lost_forward_hold_rpm", "FOLLOW_LOST_FORWARD_HOLD_RPM")
    _set_env_if_present(
        parser,
        "follow",
        "lost_forward_hold_max_sec",
        "FOLLOW_LOST_FORWARD_HOLD_MAX_SEC",
    )
    _set_env_if_present(
        parser,
        "follow",
        "lost_forward_hold_min_distance_m",
        "FOLLOW_LOST_FORWARD_HOLD_MIN_DISTANCE_M",
    )
    _set_env_if_present(
        parser,
        "follow",
        "depth_medium_confidence_hold_sec",
        "FOLLOW_DEPTH_MEDIUM_CONFIDENCE_HOLD_SEC",
    )
    _set_env_if_present(
        parser,
        "follow",
        "depth_medium_confidence_rpm",
        "FOLLOW_DEPTH_MEDIUM_CONFIDENCE_RPM",
    )
    _set_env_if_present(
        parser,
        "follow",
        "depth_recovery_stage1_sec",
        "FOLLOW_DEPTH_RECOVERY_STAGE1_SEC",
    )
    _set_env_if_present(
        parser,
        "follow",
        "depth_recovery_stage2_sec",
        "FOLLOW_DEPTH_RECOVERY_STAGE2_SEC",
    )
    _set_env_if_present(
        parser,
        "follow",
        "depth_recovery_stage1_rpm",
        "FOLLOW_DEPTH_RECOVERY_STAGE1_RPM",
    )
    _set_env_if_present(
        parser,
        "follow",
        "depth_recovery_stage2_rpm",
        "FOLLOW_DEPTH_RECOVERY_STAGE2_RPM",
    )

    # Visible-target steering cascade: camera angle outer loop plus ABZ
    # encoder-derived yaw-rate feedback. Search rotation remains independent.
    _set_bool_env_if_present(parser, "steering_pid", "enable", "VISIBLE_STEERING_PID_ENABLE")
    _set_env_if_present(parser, "steering_pid", "camera_hfov_deg", "VISIBLE_STEERING_PID_CAMERA_HFOV_DEG")
    _set_env_if_present(parser, "steering_pid", "camera_latency_sec", "VISIBLE_STEERING_PID_CAMERA_LATENCY_SEC")
    _set_env_if_present(parser, "steering_pid", "deadband_deg", "VISIBLE_STEERING_PID_DEADBAND_DEG")
    _set_env_if_present(parser, "steering_pid", "outer_kp_per_sec", "VISIBLE_STEERING_PID_OUTER_KP_PER_SEC")
    _set_env_if_present(parser, "steering_pid", "outer_kd_sec", "VISIBLE_STEERING_PID_OUTER_KD_SEC")
    _set_env_if_present(parser, "steering_pid", "target_rate_feedforward_gain", "VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_GAIN")
    _set_env_if_present(parser, "steering_pid", "target_rate_feedforward_max_dps", "VISIBLE_STEERING_PID_TARGET_RATE_FEEDFORWARD_MAX_DPS")
    _set_env_if_present(parser, "steering_pid", "target_speed_match_max_closing_dps", "VISIBLE_STEERING_PID_TARGET_SPEED_MATCH_MAX_CLOSING_DPS")
    _set_env_if_present(parser, "steering_pid", "max_yaw_rate_dps", "VISIBLE_STEERING_PID_MAX_YAW_RATE_DPS")
    _set_env_if_present(parser, "steering_pid", "rate_kp_rpm_per_dps", "VISIBLE_STEERING_PID_RATE_KP_RPM_PER_DPS")
    _set_env_if_present(parser, "steering_pid", "rate_ki_rpm_per_deg", "VISIBLE_STEERING_PID_RATE_KI_RPM_PER_DEG")
    _set_env_if_present(parser, "steering_pid", "integral_limit_deg", "VISIBLE_STEERING_PID_INTEGRAL_LIMIT_DEG")
    _set_env_if_present(parser, "steering_pid", "max_correction_rpm", "VISIBLE_STEERING_PID_MAX_CORRECTION_RPM")
    _set_env_if_present(parser, "steering_pid", "dynamic_small_error_deg", "VISIBLE_STEERING_PID_DYNAMIC_SMALL_ERROR_DEG")
    _set_env_if_present(parser, "steering_pid", "dynamic_large_error_deg", "VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_DEG")
    _set_env_if_present(parser, "steering_pid", "dynamic_small_max_yaw_rate_dps", "VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_YAW_RATE_DPS")
    _set_env_if_present(parser, "steering_pid", "dynamic_small_max_correction_rpm", "VISIBLE_STEERING_PID_DYNAMIC_SMALL_MAX_CORRECTION_RPM")
    _set_env_if_present(parser, "steering_pid", "dynamic_large_error_base_cap_rpm", "VISIBLE_STEERING_PID_DYNAMIC_LARGE_ERROR_BASE_CAP_RPM")
    _set_env_if_present(parser, "steering_pid", "opposite_yaw_brake_threshold_dps", "VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_THRESHOLD_DPS")
    _set_env_if_present(parser, "steering_pid", "opposite_yaw_brake_boost_rpm", "VISIBLE_STEERING_PID_OPPOSITE_YAW_BRAKE_BOOST_RPM")
    _set_env_if_present(parser, "steering_pid", "braking_max_correction_rpm", "VISIBLE_STEERING_PID_BRAKING_MAX_CORRECTION_RPM")
    _set_env_if_present(parser, "steering_pid", "fast_countersteer_max_correction_rpm", "VISIBLE_STEERING_PID_FAST_COUNTERSTEER_MAX_CORRECTION_RPM")
    _set_env_if_present(parser, "steering_pid", "fast_countersteer_gain_rpm_per_dps", "VISIBLE_STEERING_PID_FAST_COUNTERSTEER_GAIN_RPM_PER_DPS")
    _set_env_if_present(parser, "steering_pid", "same_direction_overspeed_threshold_dps", "VISIBLE_STEERING_PID_SAME_DIRECTION_OVERSPEED_THRESHOLD_DPS")
    _set_env_if_present(parser, "steering_pid", "same_direction_overspeed_brake_gain_rpm_per_dps", "VISIBLE_STEERING_PID_SAME_DIRECTION_OVERSPEED_BRAKE_GAIN_RPM_PER_DPS")
    _set_bool_env_if_present(parser, "steering_pid", "visual_direction_guard_enable", "VISIBLE_STEERING_PID_VISUAL_DIRECTION_GUARD_ENABLE")
    _set_env_if_present(parser, "steering_pid", "predictive_brake_decel_dps2", "VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_DECEL_DPS2")
    _set_env_if_present(parser, "steering_pid", "predictive_brake_margin_deg", "VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_MARGIN_DEG")
    _set_env_if_present(parser, "steering_pid", "predictive_brake_response_sec", "VISIBLE_STEERING_PID_PREDICTIVE_BRAKE_RESPONSE_SEC")
    _set_env_if_present(parser, "steering_pid", "min_effective_error_deg", "VISIBLE_STEERING_PID_MIN_EFFECTIVE_ERROR_DEG")
    _set_env_if_present(parser, "steering_pid", "min_effective_correction_rpm", "VISIBLE_STEERING_PID_MIN_EFFECTIVE_CORRECTION_RPM")
    _set_env_if_present(parser, "steering_pid", "mechanical_tier2_error_deg", "VISIBLE_STEERING_PID_MECHANICAL_TIER2_ERROR_DEG")
    _set_env_if_present(parser, "steering_pid", "mechanical_tier2_correction_rpm", "VISIBLE_STEERING_PID_MECHANICAL_TIER2_CORRECTION_RPM")
    _set_env_if_present(parser, "steering_pid", "mechanical_tier3_error_deg", "VISIBLE_STEERING_PID_MECHANICAL_TIER3_ERROR_DEG")
    _set_env_if_present(parser, "steering_pid", "mechanical_tier3_correction_rpm", "VISIBLE_STEERING_PID_MECHANICAL_TIER3_CORRECTION_RPM")
    _set_env_if_present(parser, "steering_pid", "mechanical_floor_release_ratio", "VISIBLE_STEERING_PID_MECHANICAL_FLOOR_RELEASE_RATIO")
    _set_env_if_present(parser, "steering_pid", "startup_kick_error_deg", "VISIBLE_STEERING_PID_STARTUP_KICK_ERROR_DEG")
    _set_env_if_present(parser, "steering_pid", "startup_kick_rpm", "VISIBLE_STEERING_PID_STARTUP_KICK_RPM")
    _set_env_if_present(parser, "steering_pid", "startup_kick_max_sec", "VISIBLE_STEERING_PID_STARTUP_KICK_MAX_SEC")
    _set_env_if_present(parser, "steering_pid", "startup_kick_release_yaw_rate_dps", "VISIBLE_STEERING_PID_STARTUP_KICK_RELEASE_YAW_RATE_DPS")
    _set_env_if_present(parser, "steering_pid", "active_brake_yaw_threshold_dps", "VISIBLE_STEERING_PID_ACTIVE_BRAKE_YAW_THRESHOLD_DPS")
    _set_env_if_present(parser, "steering_pid", "active_brake_min_correction_rpm", "VISIBLE_STEERING_PID_ACTIVE_BRAKE_MIN_CORRECTION_RPM")
    _set_env_if_present(parser, "steering_pid", "edge_boost_start_error_deg", "VISIBLE_STEERING_PID_EDGE_BOOST_START_ERROR_DEG")
    _set_env_if_present(parser, "steering_pid", "aggressive_inner_wheel_margin_rpm", "VISIBLE_STEERING_PID_AGGRESSIVE_INNER_WHEEL_MARGIN_RPM")
    _set_env_if_present(parser, "steering_pid", "fallback_max_correction_rpm", "VISIBLE_STEERING_PID_FALLBACK_MAX_CORRECTION_RPM")
    _set_env_if_present(parser, "steering_pid", "lost_hold_max_correction_rpm", "VISIBLE_STEERING_PID_LOST_HOLD_MAX_CORRECTION_RPM")
    _set_env_if_present(parser, "steering_pid", "left_body_deg_per_encoder_deg", "VISIBLE_STEERING_PID_LEFT_BODY_DEG_PER_ENCODER_DEG")
    _set_env_if_present(parser, "steering_pid", "right_body_deg_per_encoder_deg", "VISIBLE_STEERING_PID_RIGHT_BODY_DEG_PER_ENCODER_DEG")
    _set_env_if_present(parser, "steering_pid", "feedback_poll_interval_sec", "VISIBLE_STEERING_PID_FEEDBACK_POLL_INTERVAL_SEC")
    _set_env_if_present(parser, "steering_pid", "feedback_log_interval_sec", "VISIBLE_STEERING_PID_FEEDBACK_LOG_INTERVAL_SEC")
    _set_env_if_present(parser, "steering_pid", "feedback_median_window", "VISIBLE_STEERING_PID_FEEDBACK_MEDIAN_WINDOW")
    _set_env_if_present(parser, "steering_pid", "feedback_stale_sec", "VISIBLE_STEERING_PID_FEEDBACK_STALE_SEC")
    _set_env_if_present(parser, "steering_pid", "error_filter_alpha", "VISIBLE_STEERING_PID_ERROR_FILTER_ALPHA")
    _set_env_if_present(parser, "steering_pid", "derivative_filter_alpha", "VISIBLE_STEERING_PID_DERIVATIVE_FILTER_ALPHA")
    _set_env_if_present(parser, "steering_pid", "fallback_base_rpm", "VISIBLE_STEERING_PID_FALLBACK_BASE_RPM")
    _set_env_if_present(parser, "steering_pid", "parked_recenter_min_rpm", "PARKED_RECENTER_MIN_RPM")
    _set_env_if_present(parser, "steering_pid", "parked_recenter_max_rpm", "PARKED_RECENTER_MAX_RPM")
    _set_env_if_present(
        parser,
        "steering_pid",
        "lost_hold_min_correction_rpm",
        "VISIBLE_STEERING_PID_LOST_HOLD_MIN_CORRECTION_RPM",
    )
    _set_bool_env_if_present(parser, "lateral_intent", "enable", "LATERAL_INTENT_CONTROL_ENABLE")
    _set_env_if_present(parser, "lateral_intent", "control_rate_hz", "LATERAL_INTENT_CONTROL_RATE_HZ")
    _set_env_if_present(parser, "lateral_intent", "ttl_sec", "LATERAL_INTENT_TTL_SEC")
    _set_env_if_present(parser, "lateral_intent", "max_projection_sec", "LATERAL_INTENT_MAX_PROJECTION_SEC")
    _set_env_if_present(parser, "lateral_intent", "max_projection_ratio", "LATERAL_INTENT_MAX_PROJECTION_RATIO")
    _set_env_if_present(parser, "lateral_intent", "rise_rpm_per_sec", "LATERAL_INTENT_RISE_RPM_PER_SEC")
    _set_env_if_present(parser, "lateral_intent", "brake_rpm_per_sec", "LATERAL_INTENT_BRAKE_RPM_PER_SEC")
    _set_env_if_present(parser, "lateral_intent", "motor_publish_interval_sec", "LATERAL_INTENT_MOTOR_PUBLISH_INTERVAL_SEC")
    _set_env_if_present(parser, "lateral_intent", "follow_wheel_period_sec", "FOLLOW_WHEEL_PERIOD_SEC")
    _set_bool_env_if_present(parser, "lateral_intent", "follow_forward_handoff_enable", "FOLLOW_FORWARD_HANDOFF_ENABLE")
    _set_env_if_present(parser, "lateral_intent", "log_interval_sec", "LATERAL_INTENT_LOG_INTERVAL_SEC")
    _set_bool_env_if_present(parser, "safety", "side_ir_blocks_rotation", "SIDE_IR_BLOCKS_ROTATION")
    _set_env_if_present(parser, "safety", "side_ir_confirm_sec", "SIDE_IR_CONFIRM_SEC")
    _set_env_if_present(parser, "safety", "side_ir_release_sec", "SIDE_IR_RELEASE_SEC")
    _set_bool_env_if_present(parser, "safety", "search_rotate_front_block_enable", "SEARCH_ROTATE_FRONT_BLOCK_ENABLE")
    _set_bool_env_if_present(parser, "safety", "search_rotate_distance_block_enable", "SEARCH_ROTATE_DISTANCE_BLOCK_ENABLE")
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
