#!/usr/bin/env python3
"""Standalone minimal person-follow closed loop for RK3588.

This is intentionally independent from request_0513_modular.py.  It reuses
only stable adapters: RKNN vision, registered Astra depth, front IR, and the
LZ30EMA motor backend.  The normal loop has no per-frame ReID: an optional
appearance worker runs only for bounded initial enrollment and loss recovery.
It does not reuse the legacy action queue, target memory, or motion predictor.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from car_control_modular.config_loader import preload_config_from_argv

LOADED_CONFIG = preload_config_from_argv()

from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig
from minimal_follow.reid_v2 import (
    ReidCandidate,
    ReidConfig,
    ReidDecision,
    ReidPolicy,
    ReidWorker,
    ReidWorkerConfig,
)
from minimal_follow import (
    LostPersonSearchConfig,
    LostPersonSearchPolicy,
    MinimalFollowConfig,
    MinimalFollowController,
)
from rk_vision.yolo11 import YOLO11Config, YOLO11RKNNDetector

try:
    import cv2
except Exception as exc:  # pragma: no cover - board dependency
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    from ir_hal import IR
except Exception as exc:  # pragma: no cover - board dependency
    IR = None
    _IR_IMPORT_ERROR = exc
else:
    _IR_IMPORT_ERROR = None


LOG = logging.getLogger("minimal_follow")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _bool_env(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


@dataclass(frozen=True)
class RuntimeConfig:
    camera_device: str
    camera_width: int
    camera_height: int
    camera_fps: float
    camera_fourcc: str
    camera_retry_sec: float
    confidence_threshold: float
    target_distance_m: float
    distance_deadband_m: float
    distance_kp_percent_per_m: float
    min_forward_percent: int
    max_forward_percent: int
    center_deadband_ratio: float
    steering_delta_max_percent: int
    search_enabled: bool
    search_lost_confirm_frames: int
    search_turn_memory_sec: float
    search_timeout_sec: float
    search_turn_percent: int
    reid_enabled: bool
    reid_stable_enroll_frames: int
    reid_enroll_interval_sec: float
    reid_reacquire_interval_sec: float
    reid_result_max_age_sec: float
    reid_full_match_threshold: float
    reid_torso_match_threshold: float
    reid_confirm_hits: int
    motor_enabled: bool
    front_ir_enabled: bool

    @classmethod
    def from_env(cls, *, motor_enabled: bool) -> "RuntimeConfig":
        return cls(
            camera_device=os.environ.get("RKNN_CAMERA_DEVICE", "/dev/video0").strip(),
            camera_width=_int_env("RKNN_CAMERA_WIDTH", 640),
            camera_height=_int_env("RKNN_CAMERA_HEIGHT", 480),
            camera_fps=_float_env("RKNN_CAMERA_FPS", 30.0),
            camera_fourcc=os.environ.get("RKNN_CAMERA_FOURCC", "MJPG").strip(),
            camera_retry_sec=max(0.2, _float_env("RKNN_CAMERA_RETRY_INTERVAL_SEC", 2.0)),
            confidence_threshold=_float_env("CONFIDENCE_THRESHOLD", 0.25),
            target_distance_m=_float_env("MINIMAL_TARGET_DISTANCE_M", 1.4),
            distance_deadband_m=_float_env("MINIMAL_DISTANCE_DEADBAND_M", 0.10),
            distance_kp_percent_per_m=_float_env("MINIMAL_DISTANCE_KP_PERCENT_PER_M", 30.0),
            min_forward_percent=_int_env("MINIMAL_MIN_FORWARD_PERCENT", 8),
            max_forward_percent=_int_env("MINIMAL_MAX_FORWARD_PERCENT", 20),
            center_deadband_ratio=_float_env("MINIMAL_CENTER_DEADBAND_RATIO", 0.08),
            steering_delta_max_percent=_int_env("MINIMAL_STEERING_DELTA_MAX_PERCENT", 8),
            search_enabled=_bool_env("MINIMAL_SEARCH_ENABLE", True),
            search_lost_confirm_frames=max(1, _int_env("MINIMAL_SEARCH_LOST_CONFIRM_FRAMES", 1)),
            search_turn_memory_sec=max(0.0, _float_env("MINIMAL_SEARCH_TURN_MEMORY_SEC", 1.0)),
            search_timeout_sec=max(0.0, _float_env("MINIMAL_SEARCH_TIMEOUT_SEC", 1.5)),
            search_turn_percent=max(0, _int_env("MINIMAL_SEARCH_TURN_PERCENT", 8)),
            reid_enabled=_bool_env("MINIMAL_REID_ENABLE", True),
            reid_stable_enroll_frames=max(1, _int_env("MINIMAL_REID_STABLE_ENROLL_FRAMES", 10)),
            reid_enroll_interval_sec=max(0.0, _float_env("MINIMAL_REID_ENROLL_INTERVAL_SEC", 1.0)),
            reid_reacquire_interval_sec=max(0.0, _float_env("MINIMAL_REID_REACQUIRE_INTERVAL_SEC", 0.20)),
            reid_result_max_age_sec=max(0.01, _float_env("MINIMAL_REID_RESULT_MAX_AGE_SEC", 0.30)),
            reid_full_match_threshold=_float_env("MINIMAL_REID_FULL_THRESHOLD", 0.78),
            reid_torso_match_threshold=_float_env("MINIMAL_REID_TORSO_THRESHOLD", 0.84),
            reid_confirm_hits=max(1, _int_env("MINIMAL_REID_CONFIRM_HITS", 2)),
            motor_enabled=bool(motor_enabled),
            front_ir_enabled=_bool_env("MINIMAL_FRONT_IR_ENABLE", True),
        )


class MinimalFollowRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.running = True
        self.camera = None
        self.next_camera_retry_at = 0.0
        self.last_camera_warning_at = 0.0
        self.last_command_key = None
        self.last_stop_at = 0.0
        self.frame_index = 0
        self.previous_frame_started_at: Optional[float] = None
        self.motor: Optional[MssdMotorBackend] = None
        self.ir_started = False
        self.reid_worker: Optional[ReidWorker] = None

        model_value = os.environ.get(
            "VISION_MODEL_PATH", "models/yolo11n_int8_person_val2017.rknn"
        ).strip()
        yolo_model = model_value if os.path.isabs(model_value) else os.path.join(SCRIPT_DIR, model_value)
        self.detector = YOLO11RKNNDetector(
            YOLO11Config(
                model_path=os.path.abspath(yolo_model),
                input_size=_int_env("RKNN_YOLO_INPUT_SIZE", 640),
                conf_threshold=config.confidence_threshold,
                nms_threshold=_float_env("RKNN_YOLO_NMS_THRESHOLD", 0.45),
                num_classes=_int_env("RKNN_YOLO_NUM_CLASSES", 80),
                input_format=os.environ.get("RKNN_YOLO_INPUT_FORMAT", "RGB").strip(),
                output_box_format=os.environ.get("RKNN_YOLO_BOX_FORMAT", "xywh").strip(),
                target=os.environ.get("RKNN_TARGET", "rk3588").strip(),
                core_mask=os.environ.get("RKNN_CORE_MASK", "auto").strip(),
                backend=os.environ.get("RKNN_BACKEND", "auto").strip(),
            )
        )

        if not _bool_env("MODULE_ASTRA_DEPTH_ENABLE", True):
            raise RuntimeError("minimal runtime requires [modules] astra_depth=true")
        self.depth = AstraDepthRuntime(
            AstraDepthConfig(
                openni_path=os.environ.get(
                    "ASTRA_DEPTH_OPENNI_PATH",
                    os.path.join(SCRIPT_DIR, "runtime_dependencies", "arm64_openni2"),
                ),
                width=_int_env("ASTRA_DEPTH_WIDTH", config.camera_width),
                height=_int_env("ASTRA_DEPTH_HEIGHT", config.camera_height),
                fps=_int_env("ASTRA_DEPTH_FPS", int(config.camera_fps)),
                min_distance_m=_float_env("ASTRA_DEPTH_MIN_DISTANCE_M", 0.35),
                max_distance_m=_float_env("ASTRA_DEPTH_MAX_DISTANCE_M", 8.0),
                max_frame_age_sec=_float_env("ASTRA_DEPTH_MAX_FRAME_AGE_SEC", 0.25),
                hold_sec=_float_env("ASTRA_DEPTH_HOLD_SEC", 0.20),
                rgb_processing_delay_sec=0.0,
                median_window=_int_env("ASTRA_DEPTH_MEDIAN_WINDOW", 3),
            ),
            logger=LOG,
        )
        self.depth.start()

        if config.front_ir_enabled:
            if IR is None:
                raise RuntimeError(f"front IR import failed: {_IR_IMPORT_ERROR}")
            if IR.init() != 0:
                raise RuntimeError("front IR initialization failed")
            self.ir_started = True

        self.controller = MinimalFollowController(
            MinimalFollowConfig(
                target_distance_m=config.target_distance_m,
                distance_deadband_m=config.distance_deadband_m,
                distance_kp_percent_per_m=config.distance_kp_percent_per_m,
                min_forward_percent=config.min_forward_percent,
                max_forward_percent=config.max_forward_percent,
                center_deadband_ratio=config.center_deadband_ratio,
                steering_delta_max_percent=config.steering_delta_max_percent,
            )
        )
        self.search_policy = LostPersonSearchPolicy(
            LostPersonSearchConfig(
                enabled=config.search_enabled,
                lost_confirm_frames=config.search_lost_confirm_frames,
                turn_memory_sec=config.search_turn_memory_sec,
                timeout_sec=config.search_timeout_sec,
                turn_percent=config.search_turn_percent,
            )
        )
        self.reid_policy = self._make_reid_policy()
        if config.motor_enabled:
            self.motor = self._make_motor()
        LOG.info(
            "minimal follow ready motor_enabled=%s target=%.2fm deadband=%.2fm camera=%s %dx%d@%.1f "
            "search(enabled=%s memory=%.2fs timeout=%.2fs turn=%d%%)",
            config.motor_enabled, config.target_distance_m, config.distance_deadband_m,
            config.camera_device, config.camera_width, config.camera_height, config.camera_fps,
            config.search_enabled, config.search_turn_memory_sec,
            config.search_timeout_sec, config.search_turn_percent,
        )

    def _make_reid_policy(self) -> ReidPolicy:
        cfg = self.config
        policy_config = ReidConfig(
            enabled=cfg.reid_enabled,
            stable_frames=cfg.reid_stable_enroll_frames,
            enroll_interval_sec=cfg.reid_enroll_interval_sec,
            reacquire_interval_sec=cfg.reid_reacquire_interval_sec,
            result_max_age_sec=cfg.reid_result_max_age_sec,
            full_threshold=cfg.reid_full_match_threshold,
            torso_threshold=cfg.reid_torso_match_threshold,
            confirm_hits=cfg.reid_confirm_hits,
            confirm_window=max(cfg.reid_confirm_hits, _int_env("MINIMAL_REID_CONFIRM_WINDOW", 3)),
            min_full_templates=max(1, _int_env("MINIMAL_REID_MIN_FULL_TEMPLATES", 1)),
            min_torso_templates=max(1, _int_env("MINIMAL_REID_MIN_TORSO_TEMPLATES", 2)),
            max_full_templates=max(1, _int_env("MINIMAL_REID_MAX_FULL_TEMPLATES", 8)),
            max_torso_templates=max(1, _int_env("MINIMAL_REID_MAX_TORSO_TEMPLATES", 4)),
            min_iou=_float_env("MINIMAL_REID_ASSOC_MIN_IOU", 0.12),
            max_center_distance_ratio=_float_env("MINIMAL_REID_ASSOC_MAX_CENTER_RATIO", 0.16),
            min_confidence=_float_env("MINIMAL_REID_MIN_CONFIDENCE", 0.65),
            min_height_px=_float_env("MINIMAL_REID_MIN_HEIGHT_PX", 120.0),
            min_area_px=_float_env("MINIMAL_REID_MIN_AREA_PX", 8000.0),
            edge_margin_ratio=_float_env("MINIMAL_REID_EDGE_MARGIN_RATIO", 0.015),
        )
        if not cfg.reid_enabled:
            LOG.info("minimal ReID v2 disabled")
            return ReidPolicy(policy_config, None)
        model_value = os.environ.get(
            "VISION_REID_MODEL_PATH", "models/osnet_x0_25_msmt17_b1.rknn"
        ).strip()
        model_path = model_value if os.path.isabs(model_value) else os.path.join(SCRIPT_DIR, model_value)
        if not os.path.isfile(model_path):
            LOG.warning("minimal ReID v2 disabled: OSNet model not found: %s", model_path)
            disabled = ReidConfig(**{**policy_config.__dict__, "enabled": False})
            return ReidPolicy(disabled, None)
        self.reid_worker = ReidWorker(ReidWorkerConfig(
            model_path=model_path,
            input_width=_int_env("RKNN_REID_INPUT_WIDTH", 128),
            input_height=_int_env("RKNN_REID_INPUT_HEIGHT", 256),
            input_format=os.environ.get("RKNN_REID_INPUT_FORMAT", "RGB").strip(),
            input_dtype=os.environ.get("RKNN_REID_INPUT_DTYPE", "float32").strip(),
            input_layout=os.environ.get("RKNN_REID_INPUT_LAYOUT", "NCHW").strip(),
            normalize=os.environ.get("RKNN_REID_NORMALIZE", "imagenet").strip(),
            target=os.environ.get("RKNN_TARGET", "rk3588").strip(),
            core_mask=os.environ.get("MINIMAL_REID_RKNN_CORE_MASK", os.environ.get("RKNN_CORE_MASK", "auto")).strip(),
            backend=os.environ.get("RKNN_BACKEND", "auto").strip(),
        ), logger=LOG)
        self.reid_worker.start()
        LOG.info(
            "minimal ReID v2 ready model=%s enroll_every=%.2fs reacquire_every=%.2fs confirm=%d",
            model_path, cfg.reid_enroll_interval_sec,
            cfg.reid_reacquire_interval_sec, cfg.reid_confirm_hits,
        )
        return ReidPolicy(policy_config, self.reid_worker)

    def _make_motor(self) -> MssdMotorBackend:
        backend = os.environ.get("MOTOR_BACKEND", "rs485_lz30ema").strip().lower()
        if backend not in {"rs485", "rs485_lz30ema", "lz30ema", "lianzhan"}:
            raise RuntimeError(f"minimal runtime requires LZ30EMA backend, got {backend!r}")
        return MssdMotorBackend(
            MssdMotorConfig(
                port=os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0").strip(),
                slave_id=_int_env("MOTOR_RS485_SLAVE_ID", 1),
                baudrate=_int_env("MOTOR_RS485_BAUDRATE", 115200),
                timeout=_float_env("MOTOR_RS485_TIMEOUT", 0.15),
                lib_dir=os.environ.get("MOTOR_RS485_LIB_DIR", "runtime_dependencies/lianzhan").strip(),
                max_target=_int_env("MOTOR_RS485_MAX_TARGET", 200),
                percent_limit=_int_env("MOTOR_PERCENT_LIMIT", 100),
                left_sign=_int_env("MOTOR_LEFT_SIGN", -1),
                right_sign=_int_env("MOTOR_RIGHT_SIGN", 1),
                forward_target_sign=_int_env("MOTOR_FORWARD_TARGET_SIGN", -1),
                m1_is_left_wheel=_bool_env("M1_IS_LEFT_WHEEL", False),
                exit_parking_mode_on_arm=_bool_env("MOTOR_EXIT_PARKING_MODE_ON_ARM", False),
                stop_mode=os.environ.get("MOTOR_RS485_STOP_MODE", "emergency").strip(),
                stop_zero_delay_sec=_float_env("MOTOR_RS485_STOP_ZERO_DELAY_SEC", 0.0),
                parking_current_a=_float_env("MOTOR_PARKING_CURRENT_A", 10.0),
            ),
            logger=LOG,
        )

    def _warn_camera(self, message: str, *args) -> None:
        now = time.monotonic()
        if now - self.last_camera_warning_at >= 5.0:
            self.last_camera_warning_at = now
            LOG.warning(message, *args)

    def _open_camera(self) -> bool:
        if self.camera is not None:
            return True
        if cv2 is None:
            raise RuntimeError(f"OpenCV import failed: {_CV2_IMPORT_ERROR}")
        candidates = [(self.config.camera_device, getattr(cv2, "CAP_V4L2", 0), "v4l2")]
        if self.config.camera_device.startswith("/dev/video"):
            candidates.append((self.config.camera_device, getattr(cv2, "CAP_ANY", 0), "any"))
        for device, backend, label in candidates:
            camera = cv2.VideoCapture(device, backend)
            if not camera.isOpened():
                camera.release()
                continue
            if len(self.config.camera_fourcc) == 4:
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.camera_fourcc))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            self.camera = camera
            LOG.info("camera opened device=%s backend=%s", device, label)
            return True
        self._warn_camera("camera open failed: %s", self.config.camera_device)
        return False

    def _front_ir_triggered(self) -> bool:
        if not self.ir_started:
            return False
        try:
            return bool(IR.is_triggered(IR.IDX_1))
        except Exception as exc:
            LOG.warning("front IR read failed; fail closed: %s", exc)
            return True

    @staticmethod
    def _person_candidates(detections, threshold: float) -> list[ReidCandidate]:
        candidates = []
        for detection in detections:
            if int(detection.class_id) != 0 or float(detection.score) < threshold:
                continue
            bbox = tuple(float(value) for value in detection.bbox)
            candidates.append(ReidCandidate(bbox, 1, float(detection.area), float(detection.score)))
        return candidates

    def _measure_distance(self, bbox, track_id: int, width: int, height: int) -> Optional[float]:
        measurement = self.depth.measure_target(
            bbox, width, height, target_id=track_id, use_latest_depth=True,
        )
        return measurement.distance_m

    def _dispatch(self, command) -> bool:
        key = (
            command.left_percent, command.right_percent,
            command.left_state, command.right_state, command.reason,
        )
        now = time.monotonic()
        changed = key != self.last_command_key
        if changed:
            LOG.info(
                "minimal command reason=%s left=%d%%/state=%02x right=%d%%/state=%02x motor_enabled=%s",
                command.reason, command.left_percent, command.left_state,
                command.right_percent, command.right_state, self.config.motor_enabled,
            )
            self.last_command_key = key
        if self.motor is None:
            return True
        if not command.moving:
            # Every new STOP reason must be sent immediately. Repeated held
            # stops are rate-limited only to avoid needless RS485 traffic.
            if changed or now - self.last_stop_at >= 1.0:
                self.motor.send_stop("minimal_" + command.reason, mode="emergency")
                self.last_stop_at = now
            return True
        left = self.motor.wheel_state_to_target("left", command.left_percent, command.left_state)
        right = self.motor.wheel_state_to_target("right", command.right_percent, command.right_state)
        self.motor.send_targets(left, right, "minimal_" + command.reason)
        return True

    def _log_frame_timing(
        self,
        *,
        frame_started_at: float,
        capture_ms: float,
        detect_ms: float,
        select_ms: float,
        ir_ms: float,
        depth_ms: float,
        decision_ms: float,
        dispatch_ms: float,
        search_policy_ms: float,
        search_status,
        appearance_policy_ms: float,
        appearance_decision,
        detection_count: int,
        selected: bool,
        depth_attempted: bool,
        distance_m: Optional[float],
        front_blocked: bool,
        command,
    ) -> None:
        """Write one machine-readable latency record for every processed frame."""
        finished_at = time.perf_counter()
        previous = self.previous_frame_started_at
        self.previous_frame_started_at = frame_started_at
        self.frame_index += 1
        detector_timing = getattr(self.detector, "last_timing_ms", {}) or {}
        appearance_timing = appearance_decision.worker_timing_ms or {}

        def _appearance_timing_value(name: str):
            value = appearance_timing.get(name)
            return round(float(value), 3) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None

        record = {
            "frame": self.frame_index,
            "monotonic_s": round(time.monotonic(), 6),
            "cycle_ms": None if previous is None else round((frame_started_at - previous) * 1000.0, 3),
            "capture_ms": round(capture_ms, 3),
            "detect_ms": round(detect_ms, 3),
            "detect_preprocess_ms": round(float(detector_timing.get("preprocess", 0.0)), 3),
            "detect_inference_ms": round(float(detector_timing.get("inference", 0.0)), 3),
            "detect_decode_ms": round(float(detector_timing.get("decode", 0.0)), 3),
            "detect_nms_ms": round(float(detector_timing.get("nms", 0.0)), 3),
            "select_ms": round(select_ms, 3),
            "ir_ms": round(ir_ms, 3),
            "depth_ms": round(depth_ms, 3),
            "decision_ms": round(decision_ms, 3),
            "dispatch_ms": round(dispatch_ms, 3),
            "search_policy_ms": round(search_policy_ms, 3),
            "appearance_policy_ms": round(appearance_policy_ms, 3),
            "total_ms": round((finished_at - frame_started_at) * 1000.0, 3),
            "detections": int(detection_count),
            "person_selected": bool(selected),
            "depth_attempted": bool(depth_attempted),
            "distance_m": None if distance_m is None else round(float(distance_m), 4),
            "front_blocked": bool(front_blocked),
            "action": command.reason,
            "left_percent": int(command.left_percent),
            "right_percent": int(command.right_percent),
            "search_state": search_status.state,
            "search_direction": search_status.direction,
            "lost_frames": int(search_status.lost_frames),
            "search_elapsed_ms": (
                None if search_status.search_elapsed_ms is None
                else round(float(search_status.search_elapsed_ms), 3)
            ),
            "appearance_state": appearance_decision.state,
            "appearance_reason": appearance_decision.reason,
            "appearance_match_score": appearance_decision.match_score,
            "appearance_match_source": appearance_decision.match_source,
            "appearance_result_age_ms": appearance_decision.result_age_ms,
            "appearance_worker_preprocess_ms": _appearance_timing_value("preprocess"),
            "appearance_worker_inference_ms": _appearance_timing_value("inference"),
            "appearance_worker_partial_inference_ms": _appearance_timing_value("partial_inference"),
            "appearance_worker_total_ms": _appearance_timing_value("total"),
            "reid_state": appearance_decision.state,
            "reid_reason": appearance_decision.reason,
            "reid_match_score": appearance_decision.match_score,
            "reid_match_source": appearance_decision.match_source,
            "reid_result_age_ms": appearance_decision.result_age_ms,
            "reid_full_templates": appearance_decision.full_templates,
            "reid_torso_templates": appearance_decision.torso_templates,
            "reid_worker_preprocess_ms": _appearance_timing_value("preprocess"),
            "reid_worker_inference_ms": _appearance_timing_value("inference"),
            "reid_worker_partial_inference_ms": _appearance_timing_value("partial_inference"),
            "reid_worker_total_ms": _appearance_timing_value("total"),
        }
        LOG.info("minimal_timing %s", json.dumps(record, separators=(",", ":"), sort_keys=True))

    def run(self) -> None:
        while self.running:
            if self.camera is None:
                if time.monotonic() < self.next_camera_retry_at:
                    time.sleep(0.05)
                    continue
                if not self._open_camera():
                    self.next_camera_retry_at = time.monotonic() + self.config.camera_retry_sec
                    continue
            frame_started_at = time.perf_counter()
            capture_started_at = frame_started_at
            ok, frame = self.camera.read()
            capture_ms = (time.perf_counter() - capture_started_at) * 1000.0
            if not ok or frame is None:
                self._warn_camera("camera read failed; reconnecting")
                self.camera.release()
                self.camera = None
                self.next_camera_retry_at = time.monotonic() + self.config.camera_retry_sec
                continue
            detect_started_at = time.perf_counter()
            detections = self.detector.detect(frame, "BGR")
            detect_ms = (time.perf_counter() - detect_started_at) * 1000.0
            select_started_at = time.perf_counter()
            candidates = self._person_candidates(detections, self.config.confidence_threshold)
            select_ms = (time.perf_counter() - select_started_at) * 1000.0
            ir_started_at = time.perf_counter()
            front_blocked = self._front_ir_triggered()
            ir_ms = (time.perf_counter() - ir_started_at) * 1000.0
            depth_ms = 0.0
            depth_attempted = False
            distance = None
            policy_now = time.monotonic()
            appearance_started_at = time.perf_counter()
            reid_decision = self.reid_policy.observe(
                frame=frame, candidates=candidates, frame_id=self.frame_index + 1, now=policy_now,
                frame_width=frame.shape[1], frame_height=frame.shape[0],
            )
            appearance_policy_ms = (time.perf_counter() - appearance_started_at) * 1000.0
            selected = reid_decision.candidate if reid_decision.accepted else None
            if selected is None:
                search_started_at = time.perf_counter()
                command, search_status = self.search_policy.target_missing(
                    now=policy_now, front_obstacle=front_blocked,
                )
                search_policy_ms = (time.perf_counter() - search_started_at) * 1000.0
                decision_ms = 0.0
            else:
                bbox = selected.bbox
                track_id = selected.track_id
                search_started_at = time.perf_counter()
                search_status = self.search_policy.visible(policy_now)
                search_policy_ms = (time.perf_counter() - search_started_at) * 1000.0
                if search_status.state == "search_reacquire_transition_stop":
                    decision_started_at = time.perf_counter()
                    command = self.controller.step(
                        frame_width=frame.shape[1], bbox=bbox, distance_m=None, front_obstacle=front_blocked,
                    )
                    if not front_blocked:
                        command = command.stop("search_reacquire_transition_stop")
                    decision_ms = (time.perf_counter() - decision_started_at) * 1000.0
                else:
                    depth_attempted = True
                    depth_started_at = time.perf_counter()
                    try:
                        distance = self._measure_distance(bbox, track_id, frame.shape[1], frame.shape[0])
                    except Exception as exc:
                        LOG.warning("depth measure failed; stopping: %s", exc)
                        distance = None
                    depth_ms = (time.perf_counter() - depth_started_at) * 1000.0
                    decision_started_at = time.perf_counter()
                    command = self.controller.step(
                        frame_width=frame.shape[1], bbox=bbox, distance_m=distance, front_obstacle=front_blocked,
                    )
                    decision_ms = (time.perf_counter() - decision_started_at) * 1000.0
            dispatch_started_at = time.perf_counter()
            dispatched = self._dispatch(command)
            dispatch_ms = (time.perf_counter() - dispatch_started_at) * 1000.0
            if dispatched and selected is not None:
                self.search_policy.record_executed_follow_command(command, time.monotonic())
            self._log_frame_timing(
                frame_started_at=frame_started_at,
                capture_ms=capture_ms,
                detect_ms=detect_ms,
                select_ms=select_ms,
                ir_ms=ir_ms,
                depth_ms=depth_ms,
                decision_ms=decision_ms,
                dispatch_ms=dispatch_ms,
                search_policy_ms=search_policy_ms,
                search_status=search_status,
                appearance_policy_ms=appearance_policy_ms,
                appearance_decision=reid_decision,
                detection_count=len(detections),
                selected=selected is not None,
                depth_attempted=depth_attempted,
                distance_m=distance,
                front_blocked=front_blocked,
                command=command,
            )

    def close(self) -> None:
        self.running = False
        if self.motor is not None:
            try:
                self.motor.send_stop("minimal_shutdown", mode="emergency")
            except Exception:
                LOG.exception("minimal shutdown stop failed")
            self.motor.close()
            self.motor = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.ir_started:
            try:
                IR.deinit()
            except Exception:
                LOG.exception("front IR close failed")
        if self.reid_worker is not None:
            self.reid_worker.close()
            self.reid_worker = None
        self.depth.close()
        self.detector.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RK3588 single-person follow runtime")
    parser.add_argument("--enable-motor", action="store_true", help="allow real LZ30EMA motor output")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    runtime = MinimalFollowRuntime(RuntimeConfig.from_env(motor_enabled=bool(args.enable_motor)))

    def _stop(_signum, _frame) -> None:
        runtime.running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        runtime.run()
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    sys.exit(main())
