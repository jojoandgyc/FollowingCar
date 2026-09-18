#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Astra Pro RGB/depth runtime used by the follow-car control loop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import logging
import math
import threading
import time
from typing import Optional, Tuple

from .control_types import DepthJumpConfirmation
from .depth_orientation import configure_orientation, read_orientation
from .depth_torso_selection import allow_sparse_torso_continuation, select_torso_candidate_group
from .depth_torso_recovery import assess_torso_recovery, torso_evidence_continuous
from .depth_temporal_filter import append_depth_sample, reset_depth_window


@dataclass(frozen=True)
class AstraDepthConfig:
    openni_path: str = "/home/topeet/AstraSDK/arm64_openni2"
    width: int = 640
    height: int = 480
    fps: int = 30
    min_distance_m: float = 0.35
    max_distance_m: float = 8.0
    max_frame_age_sec: float = 0.25
    hold_sec: float = 0.20
    # RGB 需要经过采集、RKNN 和 ReID 后才得到人物框。测距时应回看同一
    # 时刻的 Depth，而不是拿约 130ms 之后的最新深度去套旧人物框。
    rgb_processing_delay_sec: float = 0.13
    roi_left_ratio: float = 0.32
    roi_right_ratio: float = 0.68
    roi_top_ratio: float = 0.22
    roi_bottom_ratio: float = 0.68
    min_valid_pixels: int = 80
    dynamic_min_valid_floor: int = 20
    dynamic_min_valid_fraction: float = 0.03
    median_window: int = 3
    foreground_cluster_span_m: float = 0.40
    foreground_cluster_min_fraction: float = 0.06
    foreground_spatial_support_fraction: float = 0.55
    torso_region_min_size_px: int = 16
    torso_region_max_size_px: int = 64
    # 距离控制取目标 ROI 中心 16x16 深度像素，排序后只保留最中间的
    # 2x2=4 个代表样本求平均。允许少量深度孔洞，但至少需要 25% 有效点。
    center_patch_size: int = 16
    center_patch_keep_count: int = 4
    center_patch_min_valid_fraction: float = 0.25
    large_bbox_guard_area_ratio: float = 0.35
    large_bbox_guard_height_ratio: float = 0.90
    large_bbox_guard_max_distance_m: float = 2.50
    max_distance_jump_m: float = 0.80
    jump_confirm_frames: int = 2
    # A near-to-far jump must also be physically plausible relative to the
    # last accepted sample. Encoder-supported chassis motion may override
    # this guard; otherwise an impossible jump remains held until re-anchor.
    max_unconfirmed_jump_rate_m_s: float = 3.0
    near_guard_distance_m: float = 1.80
    near_far_jump_confirm_frames: int = 5
    anchor_strict_age_sec: float = 0.60
    anchor_expire_age_sec: float = 1.50
    reanchor_confirm_frames: int = 3
    motion_confirm_frames: int = 3
    motion_reverse_min_m: float = 0.08
    near_far_jump_max_bbox_ratio: float = 0.90
    near_far_jump_edge_margin_ratio: float = 0.02
    encoder_wheel_circumference_m: float = 0.60
    log_every_sec: float = 1.0
    diagnostics_dir: str = ""


@dataclass(frozen=True)
class AstraDepthMeasurement:
    distance_m: Optional[float]
    raw_distance_m: Optional[float]
    sample_age_sec: Optional[float]
    valid_pixels: int
    detail: str
    anchor_age_sec: Optional[float] = None
    candidate_distance_m: Optional[float] = None
    required_valid_pixels: int = 0
    bbox_clipped: bool = False
    bbox_area_ratio: Optional[float] = None
    bbox_area_change_ratio: Optional[float] = None
    required_confirm_frames: int = 0
    confirm_count: int = 0
    rejection_reason: str = ""
    # A rejected near candidate may still be safety-relevant.  It is kept
    # separate from distance_m so the longitudinal PID continues using the
    # last trusted anchor while the caller can apply an immediate brake gate.
    safety_distance_m: Optional[float] = None
    region_count: int = 0
    jump_confirmation: Optional[DepthJumpConfirmation] = None
    # Only a newly accepted sample carries its source timestamp. Runtime may
    # recompute age after ROI processing without guessing from an earlier now.
    sample_timestamp: Optional[float] = None
    observation_sample_timestamp: Optional[float] = None
    observation_source: str = "unknown"
    temporal_status: str = ""
    roi_valid_pixels: int = 0
    roi_required_valid_pixels: int = 0
    sparse_torso_continuation: bool = False
    torso_recovery_status: str = "not_evaluated"
    filter_expired_count: int = 0
    filter_reset_count: int = 0
    filter_window_count: int = 0


@dataclass(frozen=True)
class _DepthClusterCandidate:
    distance_m: float
    pixels: int
    valid_pixels: int
    required_pixels: int
    spatial_support_fraction: float
    region_name: str


class AstraDepthRuntime:
    """Own synchronized OpenNI RGB/depth streams and target ROI ranging."""

    def __init__(
        self,
        config: AstraDepthConfig,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._openni2 = None
        self._np = None
        self._device = None
        self._color_stream = None
        self._depth_stream = None
        self._depth_orientation = None
        self._depth_lock = threading.Lock()
        self._measurement_lock = threading.RLock()
        self._latest_depth = None
        self._latest_depth_ts = 0.0
        self._last_depth_arrival_ts = 0.0
        self._last_depth_period_ms = 0.0
        history_sec = (
            max(0.05, float(config.max_frame_age_sec))
            + max(0.0, float(config.rgb_processing_delay_sec))
            + 0.10
        )
        self._depth_history = deque(
            maxlen=max(8, int(math.ceil(history_sec * max(1, int(config.fps)))))
        )
        self._depth_thread = None
        self._stop_event = threading.Event()
        self._started = False
        self._released = False
        self._last_target_id: Optional[int] = None
        self._distance_history = deque(maxlen=max(1, int(config.median_window)))
        self._distance_history_timestamps = deque(maxlen=self._distance_history.maxlen)
        self._last_accepted_distance_m: Optional[float] = None
        self._last_accepted_ts = 0.0
        self._pending_jump_distance_m: Optional[float] = None
        self._pending_jump_count = 0
        self._pending_jump_required_confirms = 0
        self._pending_jump_timestamp = 0.0
        self._pending_jump_kind = ""
        self._pending_jump_region_count = 0
        self._pending_torso_recovery_evidence = None
        self._last_torso_recovery_evidence = None
        self._last_processed_depth_ts = 0.0
        # A failed latest-frame attempt is not a trusted-state watermark.
        # Bound memory independently of stream uptime and deduplicate across
        # latest and RGB-aligned callers using the physical sample timestamp.
        self._attempted_depth_samples = deque(maxlen=128)
        self._measurement_sample_ts = None
        self._measurement_temporal_status = ""
        self._last_torso_selection = None
        self._measurement_roi_valid = 0
        self._measurement_roi_required = 0
        self._measurement_sparse = False
        self._measurement_torso_recovery_status = "not_evaluated"
        self._measurement_filter_expired_count = 0
        self._measurement_filter_reset_count = 0
        self._near_reference_bbox_area_ratio: Optional[float] = None
        self._last_accepted_bbox_area_ratio: Optional[float] = None
        self._last_accepted_bbox = None
        self._last_accepted_region_count = 0
        self._last_feedback_ts: Optional[float] = None
        self._encoder_distance_change_since_accept_m = 0.0
        self._last_log_ts = 0.0
        self._last_region_log_ts = 0.0
        self._last_diagnostic_log_ts = 0.0
        self._last_diagnostic_log_key = None
        self._measurement_regions = []
        # Immutable passive evidence; never read by ranging/control.
        self._shadow_range_evidence = None
        self.diagnostics = None
        if config.diagnostics_dir:
            from .depth_diagnostics import DepthDiagnostics
            self.diagnostics = DepthDiagnostics(config.diagnostics_dir, self.logger)

    def start(self) -> None:
        if self._started:
            return
        import numpy as np
        from openni import openni2

        c = self.config
        self._np = np
        self._openni2 = openni2
        openni2.initialize(c.openni_path)
        try:
            self._device = openni2.Device.open_any()
            self._color_stream = self._device.create_color_stream()
            self._depth_stream = self._device.create_depth_stream()
            self._color_stream.set_video_mode(
                openni2.VideoMode(
                    pixelFormat=openni2.PIXEL_FORMAT_RGB888,
                    resolutionX=int(c.width),
                    resolutionY=int(c.height),
                    fps=int(c.fps),
                )
            )
            self._depth_stream.set_video_mode(
                openni2.VideoMode(
                    pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
                    resolutionX=int(c.width),
                    resolutionY=int(c.height),
                    fps=int(c.fps),
                )
            )
            registration_mode = openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
            if not self._device.is_image_registration_mode_supported(registration_mode):
                raise RuntimeError("Astra不支持Depth到RGB硬件配准，禁止使用未对齐深度控制小车")
            self._device.set_image_registration_mode(registration_mode)
            if self._device.get_image_registration_mode() != registration_mode:
                raise RuntimeError("Astra Depth到RGB硬件配准未生效")
            configure_orientation(self._depth_stream, self._color_stream, self.logger)
            # Astra Pro exposes RGB through its UVC node. The OpenNI color stream
            # is created only to provide calibration for depth-to-color mapping;
            # starting it yields no frames on this hardware revision.
            self._depth_stream.start()
            # Some drivers reset properties on start. Verify before any frame
            # reaches the reader, and derive the ONE ingest transform here.
            self._depth_orientation = read_orientation(
                self._depth_stream, self._color_stream, self.logger, stage="started",
            )
        except Exception:
            self.close()
            raise

        self._started = True
        self._released = False
        self._stop_event.clear()
        self._depth_thread = threading.Thread(
            target=self._depth_loop,
            name="astra-depth-reader",
            daemon=True,
        )
        self._depth_thread.start()
        info = self._device.get_device_info()
        self.logger.info(
            "Astra配准Depth已启动: device=%s %dx%d@%dFPS registration=depth_to_color RGB=external_UVC",
            info,
            int(c.width),
            int(c.height),
            int(c.fps),
        )

    def _depth_loop(self) -> None:
        np = self._np
        stream = self._depth_stream
        while not self._stop_event.is_set() and stream is not None:
            try:
                if self._openni2.wait_for_any_stream([stream], 0.20) is None:
                    continue
                frame = stream.read_frame()
                now = time.monotonic()
                width = int(frame.width)
                height = int(frame.height)
                depth = np.frombuffer(
                    frame.get_buffer_as_uint16(), dtype=np.uint16
                ).reshape(height, width).copy()
                depth = self._depth_orientation.normalize(depth)
                previous_arrival = self._last_depth_arrival_ts
                period_ms = (
                    max(0.0, now - previous_arrival) * 1000.0
                    if previous_arrival > 0.0
                    else 0.0
                )
                with self._depth_lock:
                    self._latest_depth = depth
                    self._latest_depth_ts = now
                    self._last_depth_arrival_ts = now
                    self._last_depth_period_ms = period_ms
                    self._depth_history.append((now, depth))
                if self.diagnostics is not None:
                    self.diagnostics.add_depth(now, depth)
            except Exception as exc:
                if not self._stop_event.is_set():
                    self.logger.warning("Astra Depth读取失败: %s", exc)
                    self._stop_event.wait(0.05)

    def read(self) -> Tuple[bool, object]:
        """Color is exposed by Astra's configured UVC node, not OpenNI."""
        return False, None

    def release(self) -> None:
        """Camera adapter hook; SensorRuntime owns the actual device close."""
        self._released = True

    def latest_depth_ready(self) -> bool:
        with self._depth_lock:
            return self._latest_depth is not None and self._latest_depth_ts > 0.0

    def copy_depth_history(self, *, after_timestamp: float, max_frames: int = 16, nonblocking: bool = False):
        """Owned, read-only copies for offline/shadow tracking; no ranging state.

        Physical arrival stamps retain their existing meaning (not exposure
        timestamps). Copy arrays outside the acquisition lock: published depth
        arrays are never modified by the producer. A consumer must detect gaps
        and must not substitute the copy time for a source timestamp.
        """
        if not math.isfinite(after_timestamp) or after_timestamp < 0:
            raise ValueError("after_timestamp must be finite and nonnegative")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= 16:
            raise ValueError("max_frames must be in 1..16")
        if not self._depth_lock.acquire(blocking=not nonblocking):
            return ()
        try:
            samples = [(float(t), d) for t, d in self._depth_history if t > after_timestamp]
            samples = samples[-max_frames:]
        finally:
            self._depth_lock.release()
        owned = []
        for stamp, depth in samples:
            copy = depth.copy()
            copy.setflags(write=False)
            owned.append((stamp, copy))
        return tuple(owned)

    def wait_until_ready(self, timeout_sec: float = 2.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while time.monotonic() < deadline:
            if self.latest_depth_ready():
                return True
            time.sleep(0.01)
        return self.latest_depth_ready()

    def _aligned_depth_locked(
        self,
        now: float,
        *,
        reference_timestamp: Optional[float] = None,
        use_latest_depth: bool = False,
    ):
        """Return the Depth frame closest to the RGB capture time.

        Caller must hold ``_depth_lock``. The latest-frame fallback keeps
        direct unit-test injection and old camera backends compatible. When a
        real RGB capture timestamp is available, it is authoritative; the
        configured delay is only a fallback for older callers.
        """
        if use_latest_depth:
            return (
                self._latest_depth,
                float(self._latest_depth_ts),
                0.0,
            )
        target_ts = None
        if reference_timestamp is not None:
            try:
                candidate_ts = float(reference_timestamp)
            except (TypeError, ValueError):
                candidate_ts = 0.0
            if math.isfinite(candidate_ts) and candidate_ts > 0.0:
                target_ts = candidate_ts
        if target_ts is None:
            target_ts = float(now) - max(0.0, float(self.config.rgb_processing_delay_sec))
        if self._depth_history:
            sample_ts, depth = min(
                self._depth_history,
                key=lambda item: abs(float(item[0]) - target_ts),
            )
            return depth, float(sample_ts), abs(float(sample_ts) - target_ts)
        return (
            self._latest_depth,
            float(self._latest_depth_ts),
            abs(float(self._latest_depth_ts) - target_ts),
        )

    @staticmethod
    def _scaled_target_roi(
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        depth_width: int,
        depth_height: int,
        config: AstraDepthConfig,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        roi_x1 = x1 + box_width * float(config.roi_left_ratio)
        roi_x2 = x1 + box_width * float(config.roi_right_ratio)
        roi_y1 = y1 + box_height * float(config.roi_top_ratio)
        roi_y2 = y1 + box_height * float(config.roi_bottom_ratio)
        scale_x = float(depth_width) / float(max(1, int(frame_width)))
        scale_y = float(depth_height) / float(max(1, int(frame_height)))
        left = max(0, min(depth_width - 1, int(round(roi_x1 * scale_x))))
        right = max(left + 1, min(depth_width, int(round(roi_x2 * scale_x))))
        top = max(0, min(depth_height - 1, int(round(roi_y1 * scale_y))))
        bottom = max(top + 1, min(depth_height, int(round(roi_y2 * scale_y))))
        return left, top, right, bottom

    def _anchor_age(self, now: float) -> Optional[float]:
        if self._last_accepted_distance_m is None or self._last_accepted_ts <= 0.0:
            return None
        return max(0.0, float(now) - float(self._last_accepted_ts))

    def _reset_pending_jump(self) -> None:
        self._pending_jump_distance_m = None
        self._pending_jump_count = 0
        self._pending_jump_required_confirms = 0
        self._pending_jump_timestamp = 0.0
        self._pending_jump_kind = ""
        self._pending_jump_region_count = 0
        self._pending_torso_recovery_evidence = None

    def _clear_distance_history(self) -> None:
        self._measurement_filter_reset_count += reset_depth_window(
            self._distance_history, self._distance_history_timestamps,
        )

    def _advance_pending_jump(
        self, distance_m: float, sample_timestamp: float, *,
        required_confirms: int, region_count: int, kind: str,
    ) -> None:
        """Count distinct, temporally adjacent samples of the same surface."""
        tolerance = max(0.15, float(self.config.max_distance_jump_m) * 0.35)
        same_surface = bool(
            self._pending_jump_distance_m is not None
            and self._pending_jump_kind == kind
            and abs(distance_m - float(self._pending_jump_distance_m)) <= tolerance
            and 0.0 < sample_timestamp - self._pending_jump_timestamp
            <= max(0.01, float(self.config.max_frame_age_sec))
        )
        if same_surface:
            self._pending_jump_count += 1
            self._pending_jump_region_count = min(
                self._pending_jump_region_count, int(region_count)
            )
        else:
            self._pending_jump_distance_m = float(distance_m)
            self._pending_jump_count = 1
            self._pending_jump_region_count = int(region_count)
        self._pending_jump_timestamp = float(sample_timestamp)
        self._pending_jump_kind = str(kind)
        self._pending_jump_required_confirms = int(required_confirms)

    def _log_depth_diagnostic(self, now: float, measurement: AstraDepthMeasurement) -> None:
        key = (
            str(measurement.detail),
            int(measurement.confirm_count),
            int(measurement.required_confirm_frames),
            int(measurement.region_count),
            measurement.jump_confirmation,
        )
        interval = max(0.10, float(self.config.log_every_sec))
        if key == self._last_diagnostic_log_key and now - self._last_diagnostic_log_ts < interval:
            return
        self._last_diagnostic_log_key = key
        self._last_diagnostic_log_ts = now
        self.logger.info(
            "Astra depth diagnostic: target=%s detail=%s rejection=%s "
            "anchor_age_ms=%s candidate_m=%s valid=%d required=%d clipped=%s "
            "bbox_area=%s bbox_area_change=%s confirm=%d/%d safety_candidate_m=%s "
            "encoder_distance_change=%+.3fm region_count=%d jump_confirmation=%s",
            "none" if self._last_target_id is None else int(self._last_target_id),
            measurement.detail,
            measurement.rejection_reason or "none",
            "none"
            if measurement.anchor_age_sec is None
            else f"{measurement.anchor_age_sec * 1000.0:.0f}",
            "none"
            if measurement.candidate_distance_m is None
            else f"{measurement.candidate_distance_m:.3f}",
            int(measurement.valid_pixels),
            int(measurement.required_valid_pixels),
            bool(measurement.bbox_clipped),
            "none"
            if measurement.bbox_area_ratio is None
            else f"{measurement.bbox_area_ratio:.3f}",
            "none"
            if measurement.bbox_area_change_ratio is None
            else f"{measurement.bbox_area_change_ratio:.3f}",
            int(measurement.confirm_count),
            int(measurement.required_confirm_frames),
            "none"
            if measurement.safety_distance_m is None
            else f"{measurement.safety_distance_m:.3f}",
            float(self._encoder_distance_change_since_accept_m),
            int(measurement.region_count),
            measurement.jump_confirmation or "none",
        )

    def _held_measurement(
        self,
        now: float,
        detail: str,
        valid_pixels: int = 0,
        *,
        candidate_distance_m: Optional[float] = None,
        required_valid_pixels: int = 0,
        bbox_clipped: bool = False,
        bbox_area_ratio: Optional[float] = None,
        bbox_area_change_ratio: Optional[float] = None,
        required_confirm_frames: int = 0,
        confirm_count: int = 0,
        rejection_reason: Optional[str] = None,
        safety_distance_m: Optional[float] = None,
        region_count: int = 0,
    ) -> AstraDepthMeasurement:
        c = self.config
        age = self._anchor_age(now)
        if (
            self._last_accepted_distance_m is not None
            and age is not None
            and age <= max(0.0, float(c.hold_sec))
        ):
            measurement = AstraDepthMeasurement(
                distance_m=float(self._last_accepted_distance_m),
                raw_distance_m=None,
                sample_age_sec=age,
                valid_pixels=int(valid_pixels),
                detail=f"{detail}_hold",
                anchor_age_sec=age,
                candidate_distance_m=candidate_distance_m,
                required_valid_pixels=int(required_valid_pixels),
                bbox_clipped=bool(bbox_clipped),
                bbox_area_ratio=bbox_area_ratio,
                bbox_area_change_ratio=bbox_area_change_ratio,
                required_confirm_frames=int(required_confirm_frames),
                confirm_count=int(confirm_count),
                rejection_reason=rejection_reason or detail,
                safety_distance_m=safety_distance_m,
                region_count=int(region_count),
            )
        else:
            measurement = AstraDepthMeasurement(
                None,
                None,
                None,
                int(valid_pixels),
                detail,
                anchor_age_sec=age,
                candidate_distance_m=candidate_distance_m,
                required_valid_pixels=int(required_valid_pixels),
                bbox_clipped=bool(bbox_clipped),
                bbox_area_ratio=bbox_area_ratio,
                bbox_area_change_ratio=bbox_area_change_ratio,
                required_confirm_frames=int(required_confirm_frames),
                confirm_count=int(confirm_count),
                rejection_reason=rejection_reason or detail,
                safety_distance_m=safety_distance_m,
                region_count=int(region_count),
            )
        self._log_depth_diagnostic(now, measurement)
        return measurement

    def _dynamic_required_pixels(self, visible_area_pixels: int) -> int:
        return max(
            1,
            int(self.config.dynamic_min_valid_floor),
            int(math.ceil(
                max(0, int(visible_area_pixels))
                * max(0.0, min(1.0, float(self.config.dynamic_min_valid_fraction)))
            )),
        )

    @staticmethod
    def _shifted_region_bounds(
        center_x: float,
        center_y: float,
        region_width: int,
        region_height: int,
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:
        width = max(1, min(int(image_width), int(region_width)))
        height = max(1, min(int(image_height), int(region_height)))
        left = int(round(float(center_x) - width / 2.0))
        top = int(round(float(center_y) - height / 2.0))
        left = max(0, min(int(image_width) - width, left))
        top = max(0, min(int(image_height) - height, top))
        return left, top, left + width, top + height

    def _torso_sampling_regions(
        self,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        depth_width: int,
        depth_height: int,
    ) -> Tuple[list, bool]:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        scale_x = float(depth_width) / float(max(1, int(frame_width)))
        scale_y = float(depth_height) / float(max(1, int(frame_height)))
        box_width = max(1.0, (x2 - x1) * scale_x)
        box_height = max(1.0, (y2 - y1) * scale_y)
        min_size = max(4, int(self.config.torso_region_min_size_px))
        max_size = max(min_size, int(self.config.torso_region_max_size_px))
        region_width = max(min_size, min(max_size, int(round(box_width * 0.22))))
        region_height = max(min_size, min(max_size, int(round(box_height * 0.16))))
        specs = (
            ("chest_center", 0.50, 0.30),
            ("abdomen_center", 0.50, 0.50),
            ("left_torso", 0.36, 0.42),
            ("right_torso", 0.64, 0.42),
            ("lower_abdomen", 0.50, 0.68),
        )
        regions = []
        for name, x_ratio, y_ratio in specs:
            center_x = (x1 + (x2 - x1) * x_ratio) * scale_x
            center_y = (y1 + (y2 - y1) * y_ratio) * scale_y
            bounds = self._shifted_region_bounds(
                center_x,
                center_y,
                region_width,
                region_height,
                depth_width,
                depth_height,
            )
            regions.append((name, *bounds))
        margin_x = max(1.0, float(frame_width) * float(self.config.near_far_jump_edge_margin_ratio))
        margin_y = max(1.0, float(frame_height) * float(self.config.near_far_jump_edge_margin_ratio))
        clipped = bool(
            x1 <= margin_x
            or y1 <= margin_y
            or x2 >= float(frame_width) - margin_x
            or y2 >= float(frame_height) - margin_y
        )
        return regions, clipped

    @staticmethod
    def _spatial_support_mask(mask, np):
        supported = np.zeros_like(mask, dtype=np.bool_)
        supported[1:, :] |= mask[1:, :] & mask[:-1, :]
        supported[:-1, :] |= mask[:-1, :] & mask[1:, :]
        supported[:, 1:] |= mask[:, 1:] & mask[:, :-1]
        supported[:, :-1] |= mask[:, :-1] & mask[:, 1:]
        return supported

    def _region_cluster_candidates(self, patch, region_name: str) -> list:
        np = self._np
        min_mm = int(round(max(0.0, float(self.config.min_distance_m)) * 1000.0))
        max_mm = int(round(max(float(self.config.min_distance_m), float(self.config.max_distance_m)) * 1000.0))
        valid_mask = (patch >= min_mm) & (patch <= max_mm)
        valid = patch[valid_mask]
        total_valid = int(valid.size)
        required = self._dynamic_required_pixels(int(patch.size))
        if total_valid < required:
            return []
        values = np.sort(valid.reshape(-1).astype(np.int32, copy=False))
        span_mm = max(
            50,
            int(round(max(0.05, float(self.config.foreground_cluster_span_m)) * 1000.0)),
        )
        candidates = []
        index = 0
        while index < total_valid:
            upper = int(values[index]) + span_mm
            end = int(np.searchsorted(values, upper, side="right"))
            if end - index < required:
                index += 1
                continue
            low_mm = int(values[index])
            high_mm = int(values[end - 1])
            band_mask = valid_mask & (patch >= low_mm) & (patch <= high_mm)
            band_pixels = int(np.count_nonzero(band_mask))
            spatial_mask = self._spatial_support_mask(band_mask, np)
            spatial_pixels = int(np.count_nonzero(spatial_mask))
            spatial_fraction = float(spatial_pixels) / float(max(1, band_pixels))
            if (
                spatial_pixels >= required
                and spatial_fraction >= max(
                    0.0,
                    min(1.0, float(self.config.foreground_spatial_support_fraction)),
                )
            ):
                supported_values = patch[spatial_mask]
                candidates.append(
                    _DepthClusterCandidate(
                        distance_m=float(np.median(supported_values)) / 1000.0,
                        pixels=spatial_pixels,
                        valid_pixels=total_valid,
                        required_pixels=required,
                        spatial_support_fraction=spatial_fraction,
                        region_name=str(region_name),
                    )
                )
            index = max(index + 1, end)
        return candidates

    def _select_multiregion_distance(
        self,
        depth,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        anchor_age_sec: Optional[float],
    ) -> Tuple[Optional[float], int, int, int, str, bool]:
        self._last_torso_selection = None
        regions, clipped = self._torso_sampling_regions(
            bbox,
            frame_width,
            frame_height,
            int(depth.shape[1]),
            int(depth.shape[0]),
        )
        candidates = []
        region_diagnostics = []
        self._measurement_depth_size = [int(depth.shape[1]), int(depth.shape[0])]
        self._measurement_regions = []
        region_valid_total = 0
        required_max = 0
        for name, left, top, right, bottom in regions:
            patch = depth[top:bottom, left:right]
            region_candidates = self._region_cluster_candidates(patch, name)
            if region_candidates:
                region_valid_total += max(item.valid_pixels for item in region_candidates)
                required_max = max(required_max, max(item.required_pixels for item in region_candidates))
                candidates.extend(region_candidates)
            else:
                min_mm = int(round(max(0.0, float(self.config.min_distance_m)) * 1000.0))
                max_mm = int(round(max(float(self.config.min_distance_m), float(self.config.max_distance_m)) * 1000.0))
                region_valid_total += int(self._np.count_nonzero(
                    (patch >= min_mm) & (patch <= max_mm)
                ))
                required_max = max(required_max, self._dynamic_required_pixels(int(patch.size)))
            region_valid = int(self._np.count_nonzero(
                (patch >= int(round(max(0.0, float(self.config.min_distance_m)) * 1000.0)))
                & (patch <= int(round(max(float(self.config.min_distance_m), float(self.config.max_distance_m)) * 1000.0)))
            ))
            region_required = self._dynamic_required_pixels(int(patch.size))
            self._measurement_regions.append({
                "name": name, "roi": [left, top, right, bottom],
                "pixels": int(patch.size), "valid": region_valid,
                "required": region_required,
                "nonzero_below_0_4m": int(self._np.count_nonzero((patch > 0) & (patch < 400))),
                "cluster_count": len(region_candidates),
                "clusters": [dict(distance_m=float(c.distance_m), pixels=int(c.pixels),
                                  spatial_support=float(c.spatial_support_fraction))
                             for c in region_candidates[:8]],
                "clusters_omitted": max(0, len(region_candidates)-8),
            })
            best_distance = (
                None
                if not region_candidates
                else max(region_candidates, key=lambda item: int(item.pixels)).distance_m
            )
            region_diagnostics.append(
                "%s(valid=%d required=%d clusters=%d best=%s)" % (
                    name,
                    region_valid,
                    region_required,
                    len(region_candidates),
                    "none" if best_distance is None else "%.3fm" % float(best_distance),
                )
            )
        if not candidates:
            now = time.monotonic()
            if now - self._last_region_log_ts >= max(0.1, float(self.config.log_every_sec)):
                self._last_region_log_ts = now
                self.logger.info(
                    "Astra depth regions: selected=none selected_distance=none "
                    "selected_pixels=0 clipped=%s details=%s",
                    bool(clipped),
                    "; ".join(region_diagnostics),
                )
            return None, 0, region_valid_total, required_max, "none", clipped

        selected = select_torso_candidate_group(
            candidates,
            anchor_distance_m=self._last_accepted_distance_m,
            anchor_age_sec=anchor_age_sec,
            cluster_span_m=self.config.foreground_cluster_span_m,
            max_distance_jump_m=self.config.max_distance_jump_m,
            anchor_strict_age_sec=self.config.anchor_strict_age_sec,
            anchor_expire_age_sec=self.config.anchor_expire_age_sec,
            minimum_spatial_support_fraction=self.config.foreground_spatial_support_fraction,
        )
        self._last_torso_selection = selected
        if selected is None:
            return None, 0, region_valid_total, required_max, "none", clipped
        region_names = "+".join(selected.region_names)
        now = time.monotonic()
        if now - self._last_region_log_ts >= max(0.1, float(self.config.log_every_sec)):
            self._last_region_log_ts = now
            self.logger.info(
                "Astra depth regions: selected=%s selected_distance=%.3fm "
                "selected_pixels=%d clipped=%s details=%s selection_reason=%s groups=%s",
                region_names,
                selected.distance_m,
                selected.pixels,
                bool(clipped),
                "; ".join(region_diagnostics),
                selected.selection_reason,
                selected.candidate_summary,
            )
        return (
            selected.distance_m,
            selected.pixels,
            selected.valid_pixels,
            selected.required_pixels,
            region_names,
            clipped,
        )

    def _update_encoder_motion(self, steering_feedback) -> None:
        if steering_feedback is None or not bool(getattr(steering_feedback, "trustworthy", False)):
            return
        sample_ts = float(getattr(steering_feedback, "timestamp", 0.0))
        previous_ts = self._last_feedback_ts
        self._last_feedback_ts = sample_ts
        if previous_ts is None or sample_ts <= previous_ts:
            return
        dt = sample_ts - previous_ts
        if dt <= 0.0 or dt > 0.50:
            return
        forward_rpm = 0.5 * (
            float(getattr(steering_feedback, "left_forward_rpm", 0.0))
            + float(getattr(steering_feedback, "right_forward_rpm", 0.0))
        )
        distance_change_m = -(
            forward_rpm
            * max(0.0, float(self.config.encoder_wheel_circumference_m))
            * dt
            / 60.0
        )
        self._encoder_distance_change_since_accept_m = max(
            -2.0,
            min(2.0, self._encoder_distance_change_since_accept_m + distance_change_m),
        )

    def _select_foreground_cluster(self, valid):
        """Return the closest coherent depth surface in the person torso ROI."""
        values = self._np.sort(valid.reshape(-1).astype(self._np.int32, copy=False))
        total = int(values.size)
        min_fraction = max(
            0.01,
            min(1.0, float(self.config.foreground_cluster_min_fraction)),
        )
        required = max(
            self._dynamic_required_pixels(total),
            int(math.ceil(total * min_fraction)),
        )
        required = min(total, max(1, required))
        span_mm = max(
            50,
            int(round(max(0.05, float(self.config.foreground_cluster_span_m)) * 1000.0)),
        )

        # The whole rectangular person box often contains more wall pixels
        # than body pixels.  Select the nearest coherent band instead of the
        # median of all valid pixels; isolated near noise cannot meet required.
        spans = values[required - 1 :] - values[: total - required + 1]
        starts = self._np.flatnonzero(spans <= span_mm)
        if starts.size <= 0:
            return None, 0, total

        start = int(starts[0])
        upper = int(values[start]) + span_mm
        end = int(self._np.searchsorted(values, upper, side="right"))
        cluster = values[start:end]
        return float(self._np.median(cluster)) / 1000.0, int(cluster.size), total

    def _select_reference_cluster(self, valid, reference_distance_m: Optional[float]):
        """Prefer a coherent torso surface close to the previous trusted range."""
        if reference_distance_m is None:
            return None, 0, int(valid.size)
        values = valid.reshape(-1).astype(self._np.int32, copy=False)
        total = int(values.size)
        if total <= 0:
            return None, 0, 0
        reference_mm = float(reference_distance_m) * 1000.0
        tolerance_mm = max(
            150.0,
            min(
                max(0.15, float(self.config.max_distance_jump_m)) * 1000.0,
                max(0.15, float(self.config.foreground_cluster_span_m)) * 1500.0,
            ),
        )
        nearby = values[self._np.abs(values.astype(self._np.float64) - reference_mm) <= tolerance_mm]
        required = max(
            self._dynamic_required_pixels(total),
            int(math.ceil(total * max(
                0.0,
                min(1.0, float(self.config.dynamic_min_valid_fraction)),
            ))),
        )
        if int(nearby.size) < required:
            return None, int(nearby.size), total
        return float(self._np.median(nearby)) / 1000.0, int(nearby.size), total

    def _select_center_patch_distance(
        self,
        depth,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> Tuple[Optional[float], int, int, int]:
        """Measure the target from robust central samples in a 16x16 patch."""
        np = self._np
        patch_size = max(4, int(self.config.center_patch_size))
        keep_count = max(1, int(self.config.center_patch_keep_count))
        center_x = (int(left) + int(right) - 1) // 2
        center_y = (int(top) + int(bottom) - 1) // 2
        half = patch_size // 2
        x0 = max(0, min(int(depth.shape[1]) - patch_size, center_x - half + 1))
        y0 = max(0, min(int(depth.shape[0]) - patch_size, center_y - half + 1))
        patch = depth[y0 : y0 + patch_size, x0 : x0 + patch_size]
        min_mm = int(round(max(0.0, float(self.config.min_distance_m)) * 1000.0))
        max_mm = int(
            round(max(float(self.config.min_distance_m), float(self.config.max_distance_m)) * 1000.0)
        )
        valid = patch[(patch >= min_mm) & (patch <= max_mm)]
        valid_count = int(valid.size)
        total_count = int(patch_size * patch_size)
        required_valid = max(
            keep_count,
            int(math.ceil(total_count * max(
                0.0,
                min(1.0, float(self.config.center_patch_min_valid_fraction)),
            ))),
        )
        if valid_count < required_valid:
            return None, valid_count, total_count, min(keep_count, valid_count)
        ordered = np.sort(valid.reshape(-1).astype(np.float64, copy=False))
        keep_count = min(keep_count, valid_count)
        keep_start = max(0, (valid_count - keep_count) // 2)
        kept = ordered[keep_start : keep_start + keep_count]
        return float(np.mean(kept)) / 1000.0, valid_count, total_count, int(kept.size)

    def _bbox_depth_evidence(
        self,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, bool]:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        frame_area = float(max(1, int(frame_width)) * max(1, int(frame_height)))
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / frame_area
        margin = max(
            2.0,
            float(max(1, int(frame_width)))
            * max(0.0, float(self.config.near_far_jump_edge_margin_ratio)),
        )
        horizontally_clipped = bool(
            x1 <= margin or x2 >= float(max(1, int(frame_width))) - margin
        )
        return area_ratio, horizontally_clipped

    def _near_far_jump_guard_detail(
        self,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Optional[str]:
        area_ratio, horizontally_clipped = self._bbox_depth_evidence(
            bbox,
            frame_width,
            frame_height,
        )
        reference = self._near_reference_bbox_area_ratio
        max_bbox_ratio = max(
            0.05,
            min(1.0, float(self.config.near_far_jump_max_bbox_ratio)),
        )
        if horizontally_clipped:
            return "far_background_guard_edge"
        if reference is None or area_ratio > float(reference) * max_bbox_ratio:
            return "far_background_guard_bbox"
        return None

    def measure_target(
        self,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        *,
        target_id: Optional[int] = None,
        use_latest_depth: bool = False,
        steering_feedback=None,
        reference_timestamp: Optional[float] = None,
        evidence_capture_frame_id: Optional[int] = None,
    ) -> AstraDepthMeasurement:
        # Search reacquisition can also measure outside the main control
        # mutex. Keep sample deduplication, filtering and confirmation atomic.
        with self._measurement_lock:
            started = time.monotonic()
            self._measurement_sample_ts = None
            self._measurement_temporal_status = "no_sample"
            self._last_torso_selection = None
            self._measurement_regions = []
            self._measurement_depth_size = None
            self._measurement_roi_valid = self._measurement_roi_required = 0
            self._measurement_sparse = False
            self._measurement_torso_recovery_status = "not_evaluated"
            self._measurement_filter_expired_count = 0
            self._measurement_filter_reset_count = 0
            result = self._measure_target_locked(
                bbox, frame_width, frame_height, target_id=target_id,
                use_latest_depth=use_latest_depth, steering_feedback=steering_feedback,
                reference_timestamp=reference_timestamp,
            )
            # Only a genuinely new failed observation breaks the accepted
            # torso shortcut. Old/duplicate reads cannot revoke newer evidence.
            if result.raw_distance_m is None and (
                self._measurement_temporal_status == "new_sample"
                or self._measurement_temporal_status == "expired_during_sampling"
                or (use_latest_depth and self._measurement_temporal_status == "no_sample")
            ):
                self._last_torso_recovery_evidence = None
            source = "latest" if use_latest_depth else "rgb_aligned"
            result = replace(
                result, observation_sample_timestamp=self._measurement_sample_ts,
                observation_source=source, temporal_status=self._measurement_temporal_status,
                roi_valid_pixels=self._measurement_roi_valid,
                roi_required_valid_pixels=self._measurement_roi_required,
                sparse_torso_continuation=self._measurement_sparse,
                torso_recovery_status=self._measurement_torso_recovery_status,
                filter_expired_count=self._measurement_filter_expired_count,
                filter_reset_count=self._measurement_filter_reset_count,
                filter_window_count=len(self._distance_history),
            )
            finished = time.monotonic()
            if (result.sample_timestamp is not None and result.raw_distance_m is not None
                    and target_id is not None and not result.rejection_reason
                    and result.sample_timestamp == self._measurement_sample_ts):
                previous_shadow = self._shadow_range_evidence
                if previous_shadow is None or result.sample_timestamp > previous_shadow[1]:
                    self._shadow_range_evidence = (
                        int(target_id), float(result.sample_timestamp), float(result.raw_distance_m),
                    )
            self.logger.info(
                "Astra depth timeline: target=%s source=%s reference_ts=%s sample_ts=%s "
                "attempt_watermark=%.6f accepted_ts=%.6f pending_ts=%.6f "
                "temporal=%s detail=%s candidate=%s raw=%s distance=%s "
                "sample_age_ms=%s anchor_age_ms=%s processing_ms=%.1f "
                "roi_valid=%d roi_required=%d selected_regions=%s selection_reason=%s "
                "sparse_continuation=%s torso_recovery=%s "
                "filter_expired=%d filter_reset=%d filter_window=%d evidence_capture_frame_id=%s "
                "roi_scan_skipped=%s",
                target_id, source, reference_timestamp, self._measurement_sample_ts,
                self._last_processed_depth_ts, self._last_accepted_ts,
                self._pending_jump_timestamp, self._measurement_temporal_status,
                result.detail, result.candidate_distance_m, result.raw_distance_m,
                result.distance_m,
                None if self._measurement_sample_ts is None
                else round((finished - self._measurement_sample_ts) * 1000.0, 1),
                None if self._anchor_age(finished) is None
                else round(self._anchor_age(finished) * 1000.0, 1),
                (finished - started) * 1000.0,
                self._measurement_roi_valid, self._measurement_roi_required,
                "none" if self._last_torso_selection is None
                else "+".join(self._last_torso_selection.region_names),
                "none" if self._last_torso_selection is None
                else self._last_torso_selection.selection_reason,
                self._measurement_sparse,
                self._measurement_torso_recovery_status,
                self._measurement_filter_expired_count, self._measurement_filter_reset_count,
                len(self._distance_history),
                evidence_capture_frame_id,
                result.temporal_status in {"duplicate", "older_than_anchor", "older_than_pending"},
            )
            if self.diagnostics is not None:
                # Diagnostic failures must never turn a usable measurement
                # into a controller failure or alter any confirmation state.
                try:
                    metadata = dict(
                        target_id=target_id, source=source, reference_timestamp=reference_timestamp,
                        evidence_capture_frame_id=evidence_capture_frame_id,
                        sample_timestamp=self._measurement_sample_ts, observed_at=finished,
                        bbox=[float(v) for v in bbox], frame_size=[frame_width, frame_height],
                        temporal=result.temporal_status, detail=result.detail,
                        candidate_m=result.candidate_distance_m, accepted_raw_m=result.raw_distance_m,
                        filtered_or_held_m=result.distance_m, accepted_timestamp=self._last_accepted_ts,
                        roi_valid=self._measurement_roi_valid, roi_required=self._measurement_roi_required,
                        region_count=result.region_count, regions=self._measurement_regions,
                        depth_size=self._measurement_depth_size,
                        selected_region_distances=({} if self._last_torso_selection is None else {
                            r.region_name: float(r.distance_m)
                            for r in self._last_torso_selection._region_evidence
                        }),
                        selected_regions=([] if self._last_torso_selection is None
                                          else list(self._last_torso_selection.region_names)),
                        rejection_reason=result.rejection_reason,
                        confirm_count=result.confirm_count, required_confirms=result.required_confirm_frames,
                        orientation=(None if self._depth_orientation is None
                                     else self._depth_orientation.metadata()),
                        roi_scan_skipped=result.temporal_status in {
                            "duplicate", "older_than_anchor", "older_than_pending",
                        },
                    )
                    self.diagnostics.observe(
                        metadata, anomaly=result.raw_distance_m is None and result.temporal_status
                        in {"new_sample", "historical_after_later_attempt", "expired_during_sampling"},
                        now=finished, sample_stamp=self._measurement_sample_ts,
                    )
                except Exception as exc:
                    self.logger.warning("Depth diagnostic observation skipped: %s", exc)
            return result

    def _measure_target_locked(
        self,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        *,
        target_id: Optional[int] = None,
        use_latest_depth: bool = False,
        steering_feedback=None,
        reference_timestamp: Optional[float] = None,
    ) -> AstraDepthMeasurement:
        now = time.monotonic()
        current_target_id = None if target_id is None else int(target_id)
        if current_target_id != self._last_target_id:
            self._last_target_id = current_target_id
            self._clear_distance_history()
            self._last_torso_recovery_evidence = None
            self._last_accepted_distance_m = None
            self._last_accepted_ts = 0.0
            self._reset_pending_jump()
            self._last_processed_depth_ts = 0.0
            self._attempted_depth_samples.clear()
            self._near_reference_bbox_area_ratio = None
            self._last_accepted_bbox_area_ratio = None
            self._last_accepted_bbox = None
            self._last_accepted_region_count = 0
            self._last_feedback_ts = None
            self._encoder_distance_change_since_accept_m = 0.0
        with self._depth_lock:
            depth, sample_ts, alignment_error_sec = self._aligned_depth_locked(
                now,
                reference_timestamp=reference_timestamp,
                use_latest_depth=bool(use_latest_depth),
            )
        if depth is None or not math.isfinite(float(sample_ts)) or sample_ts <= 0.0:
            if use_latest_depth:
                self._reset_pending_jump()
            return self._held_measurement(now, "no_depth_frame")
        self._measurement_sample_ts = float(sample_ts)
        previous_attempt_ts = self._last_processed_depth_ts
        sample_age = now - sample_ts
        if sample_age < 0.0 or sample_age > max(0.01, float(self.config.max_frame_age_sec)):
            self._measurement_temporal_status = "stale_or_future"
            if 0.0 <= sample_age and (use_latest_depth or sample_ts >= previous_attempt_ts):
                self._reset_pending_jump()
            return self._held_measurement(now, "stale_depth_frame")
        attempted = any(abs(sample_ts - stamp) <= 1e-9 for stamp in self._attempted_depth_samples)
        if not attempted:
            self._attempted_depth_samples.append(float(sample_ts))
        self._last_processed_depth_ts = max(previous_attempt_ts, float(sample_ts))
        older_than_anchor = sample_ts <= self._last_accepted_ts + 1e-9
        older_than_pending = bool(
            self._pending_jump_count > 0 and sample_ts <= self._pending_jump_timestamp + 1e-9
        )
        out_of_order = sample_ts < previous_attempt_ts - 1e-9
        is_new_depth = not attempted and not older_than_anchor and not older_than_pending
        self._measurement_temporal_status = (
            "duplicate" if attempted else "older_than_anchor" if older_than_anchor
            else "older_than_pending" if older_than_pending
            else "historical_after_later_attempt" if out_of_order else "new_sample"
        )

        if not is_new_depth:
            # A physical sample cannot become new evidence by changing callers
            # or bboxes. Avoid expensive ROI clustering and do not renew the
            # accepted anchor, pending count, or PID authority. An unseen sample
            # after a newer FAILED attempt remains eligible below.
            if self._pending_jump_count > 0:
                confirms = max(1, int(self._pending_jump_required_confirms),
                               int(self.config.jump_confirm_frames))
                return self._held_measurement(
                    now, f"distance_jump_pending_{self._pending_jump_count}_of_{confirms}",
                    candidate_distance_m=self._pending_jump_distance_m,
                    required_confirm_frames=confirms, confirm_count=self._pending_jump_count,
                    rejection_reason="reused_depth_frame",
                    safety_distance_m=(
                        self._pending_jump_distance_m
                        if self._pending_jump_distance_m is not None
                        and self._last_accepted_distance_m is not None
                        and self._pending_jump_distance_m < self._last_accepted_distance_m
                        else None
                    ),
                )
            return self._held_measurement(
                now, "depth_observation_reused" if self._last_accepted_distance_m is not None
                else "reused_depth_frame", rejection_reason="reused_depth_frame",
            )

        self._update_encoder_motion(steering_feedback)

        left, top, right, bottom = self._scaled_target_roi(
            bbox,
            frame_width,
            frame_height,
            int(depth.shape[1]),
            int(depth.shape[0]),
            self.config,
        )
        roi = depth[top:bottom, left:right]
        min_mm = int(round(max(0.0, float(self.config.min_distance_m)) * 1000.0))
        max_mm = int(round(max(float(self.config.min_distance_m), float(self.config.max_distance_m)) * 1000.0))
        valid = roi[(roi >= min_mm) & (roi <= max_mm)]
        valid_pixels = int(valid.size)
        required_valid_pixels = self._dynamic_required_pixels(int(roi.size))
        self._measurement_roi_valid = valid_pixels
        self._measurement_roi_required = required_valid_pixels
        bbox_area_ratio, _ = self._bbox_depth_evidence(
            bbox,
            frame_width,
            frame_height,
        )
        sampling_regions, bbox_clipped = self._torso_sampling_regions(
            bbox,
            frame_width,
            frame_height,
            int(depth.shape[1]),
            int(depth.shape[0]),
        )
        previous_bbox_area = self._last_accepted_bbox_area_ratio
        bbox_area_change_ratio = (
            None
            if previous_bbox_area is None or previous_bbox_area <= 0.0
            else float(bbox_area_ratio) / float(previous_bbox_area)
        )
        anchor_age_sec = self._anchor_age(now)
        (
            raw_distance_m,
            foreground_pixels,
            total_valid_pixels,
            region_required_pixels,
            selected_regions,
            bbox_clipped,
        ) = self._select_multiregion_distance(
            depth,
            bbox,
            frame_width,
            frame_height,
            anchor_age_sec,
        )
        if valid_pixels < required_valid_pixels:
            sparse_continuation = allow_sparse_torso_continuation(
                self._last_torso_selection,
                anchor_distance_m=self._last_accepted_distance_m,
                anchor_age_sec=anchor_age_sec,
                strict_age_sec=self.config.anchor_strict_age_sec,
                max_near_distance_m=self.config.large_bbox_guard_max_distance_m,
                cluster_span_m=self.config.foreground_cluster_span_m,
                max_distance_jump_m=self.config.max_distance_jump_m,
                minimum_spatial_support_fraction=self.config.foreground_spatial_support_fraction,
            )
            if not sparse_continuation:
                if is_new_depth:
                    self._reset_pending_jump()
                return self._held_measurement(
                    now, "insufficient_depth_pixels", valid_pixels,
                    candidate_distance_m=raw_distance_m,
                    required_valid_pixels=required_valid_pixels,
                    bbox_clipped=bbox_clipped, bbox_area_ratio=bbox_area_ratio,
                    bbox_area_change_ratio=bbox_area_change_ratio,
                    region_count=(0 if self._last_torso_selection is None
                                  else self._last_torso_selection.region_count),
                )
            # The original central ROI remains in diagnostics. Measurement
            # support now describes the actually selected, validated regions.
            self._measurement_sparse = True
            valid_pixels = foreground_pixels
            required_valid_pixels = region_required_pixels
        required_valid_pixels = max(required_valid_pixels, region_required_pixels)
        _center_patch_distance, center_patch_valid, center_patch_pixels, center_patch_kept = (
            self._select_center_patch_distance(depth, left, top, right, bottom)
        )
        measurement_detail = (
            "depth_torso_continuation" if self._measurement_sparse else "depth_multiregion"
        )
        if raw_distance_m is None:
            # Multi-region sampling is authoritative. The whole torso ROI is a
            # last-resort coherent foreground fallback, never a plain median.
            raw_distance_m, foreground_pixels, total_valid_pixels = (
                self._select_foreground_cluster(valid)
            )
            if raw_distance_m is None:
                if is_new_depth:
                    self._reset_pending_jump()
                return self._held_measurement(
                    now,
                    "multiregion_and_foreground_insufficient",
                    valid_pixels,
                    required_valid_pixels=required_valid_pixels,
                    bbox_clipped=bbox_clipped,
                    bbox_area_ratio=bbox_area_ratio,
                    bbox_area_change_ratio=bbox_area_change_ratio,
                )
            measurement_detail = "depth_foreground_fallback_multiregion_hole"
            selected_regions = "whole_torso"
        large_bbox_guard_area = max(
            0.05,
            min(1.0, float(self.config.large_bbox_guard_area_ratio)),
        )
        bbox_height_ratio = max(0.0, float(bbox[3]) - float(bbox[1])) / float(
            max(1, int(frame_height))
        )
        large_bbox_guard_height = max(
            0.05,
            min(1.0, float(self.config.large_bbox_guard_height_ratio)),
        )
        large_bbox = bool(
            bbox_area_ratio >= large_bbox_guard_area
            or bbox_height_ratio >= large_bbox_guard_height
        )
        consensus_regions = tuple(
            item.strip()
            for item in str(selected_regions or "").split("+")
            if item.strip() and item.strip() not in ("whole_torso", "none")
        )
        region_count = len(set(consensus_regions))
        torso_recovery_evidence = None
        if (
            bbox_clipped and 1 <= region_count <= 2
            and raw_distance_m > float(self.config.large_bbox_guard_max_distance_m)
            and current_target_id is not None and current_target_id > 0
        ):
            torso_recovery_evidence, recovery_reason = assess_torso_recovery(
                selection=self._last_torso_selection, bbox=bbox,
                frame_width=frame_width, frame_height=frame_height,
                depth_width=int(depth.shape[1]), depth_height=int(depth.shape[0]),
                regions=sampling_regions,
                anchor_distance_m=self._last_accepted_distance_m,
                anchor_age_sec=anchor_age_sec, sample_timestamp=sample_ts,
                max_distance_jump_m=self.config.max_distance_jump_m,
                max_anchor_age_sec=min(3.0, 2.0 * self.config.anchor_expire_age_sec),
                edge_margin_ratio=self.config.near_far_jump_edge_margin_ratio,
                large_bbox_area_ratio=self.config.large_bbox_guard_area_ratio,
                large_bbox_height_ratio=self.config.large_bbox_guard_height_ratio,
                min_spatial_support_fraction=self.config.foreground_spatial_support_fraction,
            )
            self._measurement_torso_recovery_status = recovery_reason
        # A box touching the image boundary can expose a large background
        # surface even when its area is below the normal large-box threshold.
        # For ranges beyond the near-field guard, a single torso region is not
        # enough evidence to advance the distance anchor.
        clipped_far_weak_consensus = bool(
            bbox_clipped
            and raw_distance_m > float(self.config.large_bbox_guard_max_distance_m)
            and region_count < 2
            and torso_recovery_evidence is None
        )
        if clipped_far_weak_consensus:
            if is_new_depth:
                self._reset_pending_jump()
            edge_guard_detail = self._near_far_jump_guard_detail(
                bbox,
                frame_width,
                frame_height,
            )
            return self._held_measurement(
                now,
                edge_guard_detail or "far_background_guard_clipped_multiregion",
                valid_pixels,
                candidate_distance_m=raw_distance_m,
                required_valid_pixels=required_valid_pixels,
                bbox_clipped=bbox_clipped,
                bbox_area_ratio=bbox_area_ratio,
                bbox_area_change_ratio=bbox_area_change_ratio,
                rejection_reason="clipped_bbox_single_region",
                region_count=region_count,
            )
        multi_region_consensus = bool(
            region_count >= 3
            and int(foreground_pixels) >= int(required_valid_pixels)
        )
        if large_bbox and raw_distance_m > float(self.config.large_bbox_guard_max_distance_m):
            # 人靠近摄像头时，中心区域可能恰好落在腋下、衣服深度孔洞或
            # 露出的背景墙上。先在躯干ROI里找与上一可信距离连续的深度簇；
            # 没有历史时再找最近的连续前景簇。两者都失败才保持旧距离。
            visually_smaller = bool(
                bbox_area_change_ratio is not None
                and bbox_area_change_ratio
                <= max(0.05, min(1.0, float(self.config.near_far_jump_max_bbox_ratio)))
                and not bbox_clipped
            )
            anchor_relaxed = bool(
                anchor_age_sec is not None
                and anchor_age_sec >= float(self.config.anchor_strict_age_sec)
                and visually_smaller
            )
            if anchor_relaxed:
                anchored_m, anchored_pixels, anchored_total = None, 0, int(valid.size)
                measurement_detail = "depth_multiregion_reanchor_candidate"
            else:
                anchored_m, anchored_pixels, anchored_total = self._select_reference_cluster(
                    valid,
                    self._last_accepted_distance_m
                    if anchor_age_sec is not None
                    and anchor_age_sec <= float(self.config.anchor_expire_age_sec)
                    else None,
                )
            if anchored_m is not None and (
                float(anchored_m) <= float(self.config.large_bbox_guard_max_distance_m)
                or not multi_region_consensus
            ):
                raw_distance_m = float(anchored_m)
                foreground_pixels = int(anchored_pixels)
                total_valid_pixels = int(anchored_total)
                measurement_detail = "depth_foreground_fallback_large_bbox_anchor"
                # Whole-ROI fallback is not independent torso-region proof.
                region_count = 0
                multi_region_consensus = False
            else:
                nearest_m, nearest_pixels, nearest_total = self._select_foreground_cluster(valid)
                if nearest_m is not None and float(nearest_m) <= float(
                    self.config.large_bbox_guard_max_distance_m
                ):
                    raw_distance_m = float(nearest_m)
                    foreground_pixels = int(nearest_pixels)
                    total_valid_pixels = int(nearest_total)
                    measurement_detail = "depth_foreground_fallback_large_bbox_nearest"
                    region_count = 0
                    multi_region_consensus = False
                elif not multi_region_consensus:
                    if is_new_depth:
                        self._reset_pending_jump()
                    return self._held_measurement(
                        now,
                        "far_background_guard_large_bbox",
                        valid_pixels,
                        candidate_distance_m=raw_distance_m,
                        required_valid_pixels=required_valid_pixels,
                        bbox_clipped=bbox_clipped,
                        bbox_area_ratio=bbox_area_ratio,
                        bbox_area_change_ratio=bbox_area_change_ratio,
                        region_count=region_count,
                    )
                else:
                    measurement_detail = "depth_multiregion_large_bbox_consensus"
        high_risk_far = bool(
            (large_bbox or bbox_clipped)
            and raw_distance_m > float(self.config.large_bbox_guard_max_distance_m)
        )
        if high_risk_far and not multi_region_consensus and torso_recovery_evidence is None:
            # A reference ROI fallback may continue an already trusted far
            # anchor, but cannot bootstrap/re-anchor a clipped far surface.
            far_anchor_continuation = bool(
                measurement_detail == "depth_foreground_fallback_large_bbox_anchor"
                and self._last_accepted_distance_m is not None
                and self._last_accepted_distance_m > float(self.config.large_bbox_guard_max_distance_m)
                and anchor_age_sec is not None
                and anchor_age_sec <= float(self.config.anchor_expire_age_sec)
            )
            if not far_anchor_continuation:
                if is_new_depth:
                    self._reset_pending_jump()
                return self._held_measurement(
                    now, "far_background_guard_clipped_multiregion", valid_pixels,
                    candidate_distance_m=raw_distance_m,
                    required_valid_pixels=required_valid_pixels,
                    bbox_clipped=bbox_clipped,
                    bbox_area_ratio=bbox_area_ratio,
                    bbox_area_change_ratio=bbox_area_change_ratio,
                    rejection_reason="insufficient_multiregion_consensus",
                    region_count=region_count,
                )
        sampled_now = time.monotonic()
        if not 0.0 <= sampled_now - sample_ts <= max(0.01, float(self.config.max_frame_age_sec)):
            self._measurement_temporal_status = "expired_during_sampling"
            if is_new_depth and not out_of_order:
                self._reset_pending_jump()
            return self._held_measurement(
                sampled_now, "stale_depth_frame", valid_pixels,
                candidate_distance_m=raw_distance_m, required_valid_pixels=required_valid_pixels,
                bbox_clipped=bbox_clipped, bbox_area_ratio=bbox_area_ratio,
                region_count=region_count,
            )
        last = self._last_accepted_distance_m
        jump_limit = max(0.0, float(self.config.max_distance_jump_m))
        farther_jump = bool(
            last is not None
            and raw_distance_m > float(last)
            and raw_distance_m - float(last) > jump_limit
        )
        closer_jump = bool(
            last is not None
            and raw_distance_m < float(last)
            and float(last) - raw_distance_m > jump_limit
        )
        expire_age = max(
            float(self.config.anchor_strict_age_sec),
            float(self.config.anchor_expire_age_sec),
        )
        # A continuous torso moving across 2.5m is not a new background
        # surface. This narrow boundary path requires proof on BOTH endpoints;
        # it does not bootstrap identity, renew a stale anchor or admit jumps.
        boundary = float(self.config.large_bbox_guard_max_distance_m)
        boundary_dt = float(sample_ts) - self._last_accepted_ts
        old_bbox = self._last_accepted_bbox
        boundary_continuous = bool(
            high_risk_far and multi_region_consensus and not out_of_order
            and current_target_id is not None and current_target_id > 0
            and last is not None and boundary-.10 <= float(last) <= boundary
            and boundary < raw_distance_m <= boundary+.10
            and self._last_accepted_region_count >= 3
            and anchor_age_sec is not None and 0 <= anchor_age_sec <= .18
            and 0 <= sampled_now-sample_ts <= .18
            and 0 < boundary_dt <= .18
            and abs(raw_distance_m-float(last)) <= min(.08, jump_limit)
            and abs(raw_distance_m-float(last))/boundary_dt <= min(
                1.5, float(self.config.max_unconfirmed_jump_rate_m_s))
            and bbox_area_change_ratio is not None and .85 <= bbox_area_change_ratio <= 1.15
            and old_bbox is not None
            and .85 <= (bbox[2]-bbox[0])/max(1.,old_bbox[2]-old_bbox[0]) <= 1.15
            and .85 <= (bbox[3]-bbox[1])/max(1.,old_bbox[3]-old_bbox[1]) <= 1.15
            and abs((bbox[0]+bbox[2]-old_bbox[0]-old_bbox[2])*.5) <= frame_width*.05
            and abs((bbox[1]+bbox[3]-old_bbox[1]-old_bbox[3])*.5) <= frame_height*.05
        )
        if boundary_continuous:
            self.logger.info(
                "depth_boundary_continuity uid=%s sample_ts=%.6f anchor_ts=%.6f "
                "candidate_m=%.3f anchor_m=%.3f region_count=%d previous_regions=%d "
                "pending_bypassed=True identity_changed=False",
                current_target_id, sample_ts, self._last_accepted_ts,
                raw_distance_m, last, region_count, self._last_accepted_region_count,
            )
        needs_far_consensus_confirmation = bool(
            high_risk_far
            and multi_region_consensus
            and not boundary_continuous
            and (
                last is None
                or float(last) <= float(self.config.large_bbox_guard_max_distance_m)
                or (anchor_age_sec is not None and anchor_age_sec > expire_age)
            )
        )
        if out_of_order and (
            farther_jump or closer_jump or needs_far_consensus_confirmation
            or torso_recovery_evidence is not None
        ):
            # A late observation can restore a continuous trusted range, but
            # cannot backfill confirmations across a newer failed sample.
            self._measurement_temporal_status = "out_of_order_jump_observation"
            return self._held_measurement(
                now, "depth_out_of_order_jump_observation", valid_pixels,
                candidate_distance_m=raw_distance_m,
                required_valid_pixels=required_valid_pixels,
                bbox_clipped=bbox_clipped, bbox_area_ratio=bbox_area_ratio,
                bbox_area_change_ratio=bbox_area_change_ratio,
                rejection_reason="out_of_order_jump_observation", region_count=region_count,
            )
        jump_confirmation = None
        accepted_confirm_count = 0
        accepted_required_confirms = 0
        reanchoring_after_timeout = False
        if torso_recovery_evidence is not None:
            # A foot touching the lower image edge is not itself evidence that
            # the chest/abdomen sampling area is clipped. This narrow exception
            # still requires an existing nearby anchor and two physical frames;
            # it is never a replacement for three-region far-jump proof.
            rate_limit = max(0.1, float(self.config.max_unconfirmed_jump_rate_m_s))
            anchor_dt = max(0.033, float(sample_ts) - self._last_accepted_ts)
            if abs(float(raw_distance_m) - float(last)) / anchor_dt > rate_limit:
                self._reset_pending_jump()
                self._measurement_torso_recovery_status = "anchor_rate_rejected"
                return self._held_measurement(
                    now, "distance_jump_rate_guard", valid_pixels,
                    candidate_distance_m=raw_distance_m,
                    required_valid_pixels=required_valid_pixels,
                    bbox_clipped=bbox_clipped, bbox_area_ratio=bbox_area_ratio,
                    bbox_area_change_ratio=bbox_area_change_ratio,
                    rejection_reason="torso_recovery_rate_guard", region_count=region_count,
                )
            continuity = dict(
                max_gap_sec=max(0.01, float(self.config.max_frame_age_sec)),
                max_distance_delta_m=min(0.28, max(0.0, self.config.max_distance_jump_m)),
                max_rate_m_s=rate_limit,
            )
            continuing = torso_evidence_continuous(
                self._last_torso_recovery_evidence, torso_recovery_evidence, **continuity,
            )
            if not continuing:
                if self._pending_jump_kind != "torso_recovery" or not torso_evidence_continuous(
                    self._pending_torso_recovery_evidence, torso_recovery_evidence, **continuity,
                ):
                    self._reset_pending_jump()
                self._advance_pending_jump(
                    raw_distance_m, sample_ts, required_confirms=2,
                    region_count=region_count, kind="torso_recovery",
                )
                self._pending_torso_recovery_evidence = torso_recovery_evidence
                if self._pending_jump_count < 2:
                    self._measurement_torso_recovery_status = "pending_1_of_2"
                    return self._held_measurement(
                        now, "distance_jump_pending_torso_1_of_2", valid_pixels,
                        candidate_distance_m=raw_distance_m,
                        required_valid_pixels=required_valid_pixels,
                        bbox_clipped=bbox_clipped, bbox_area_ratio=bbox_area_ratio,
                        bbox_area_change_ratio=bbox_area_change_ratio,
                        required_confirm_frames=2, confirm_count=1,
                        rejection_reason="bottom_only_torso_confirmation", region_count=region_count,
                    )
                accepted_confirm_count = accepted_required_confirms = 2
                # A locally reconfirmed surface starts a new median window.
                # Do not blend it with the pre-loss surface, even if nearby.
                self._clear_distance_history()
            self._measurement_torso_recovery_status = "continued" if continuing else "confirmed_2_of_2"
            measurement_detail = "depth_torso_continuation" if continuing else "depth_torso_recovery_confirmed"
        elif farther_jump or needs_far_consensus_confirmation:
            confirms = max(1, int(self.config.jump_confirm_frames))
            far_from_near = bool(
                last is not None and float(last) <= float(self.config.near_guard_distance_m)
            )
            strict_age = max(0.0, float(self.config.anchor_strict_age_sec))
            expire_age = max(strict_age, float(self.config.anchor_expire_age_sec))
            effective_anchor_age = 0.0 if anchor_age_sec is None else float(anchor_age_sec)
            motion_supported = bool(
                bbox_area_change_ratio is not None
                and bbox_area_change_ratio
                <= max(0.05, min(1.0, float(self.config.near_far_jump_max_bbox_ratio)))
                and self._encoder_distance_change_since_accept_m
                >= max(0.0, float(self.config.motion_reverse_min_m))
                and not bbox_clipped
            )
            if far_from_near:
                if effective_anchor_age < strict_age:
                    guard_detail = self._near_far_jump_guard_detail(
                        bbox,
                        frame_width,
                        frame_height,
                    )
                    if (
                        guard_detail is not None
                        and not motion_supported
                        and not (high_risk_far and multi_region_consensus)
                    ):
                        self._reset_pending_jump()
                        return self._held_measurement(
                            now,
                            guard_detail,
                            valid_pixels,
                            candidate_distance_m=raw_distance_m,
                            required_valid_pixels=required_valid_pixels,
                            bbox_clipped=bbox_clipped,
                            bbox_area_ratio=bbox_area_ratio,
                            bbox_area_change_ratio=bbox_area_change_ratio,
                            required_confirm_frames=int(self.config.near_far_jump_confirm_frames),
                            rejection_reason=guard_detail,
                            region_count=region_count,
                        )
                    confirms = max(
                        confirms,
                        int(
                            self.config.motion_confirm_frames
                            if motion_supported
                            else self.config.near_far_jump_confirm_frames
                        ),
                    )
                elif effective_anchor_age <= expire_age:
                    confirms = max(confirms, int(self.config.reanchor_confirm_frames))
                else:
                    confirms = max(confirms, int(self.config.reanchor_confirm_frames))
                    reanchoring_after_timeout = True
            elif last is not None and effective_anchor_age > expire_age:
                confirms = max(confirms, int(self.config.reanchor_confirm_frames))
                reanchoring_after_timeout = True
            if high_risk_far:
                confirms = max(2, confirms, int(self.config.reanchor_confirm_frames))

            # A person cannot move from 1.8m to 3.7m in a few 30Hz samples.
            # Do not let a stable background surface become authoritative just
            # because it satisfies the frame-count confirmation. Once the
            # anchor is old enough, the normal re-anchor confirmation is used.
            elapsed_since_accept = max(
                0.033,
                float(sample_ts) - float(self._last_accepted_ts),
            )
            jump_rate = (
                0.0 if last is None
                else (float(raw_distance_m) - float(last)) / elapsed_since_accept
            )
            rate_limit = max(0.1, float(self.config.max_unconfirmed_jump_rate_m_s))
            if (
                jump_rate > rate_limit
                and farther_jump
                and effective_anchor_age <= expire_age
                and not motion_supported
            ):
                self._reset_pending_jump()
                rate_guard_confirms = max(
                    int(confirms), int(self.config.near_far_jump_confirm_frames)
                )
                return self._held_measurement(
                    now,
                    "distance_jump_rate_guard",
                    valid_pixels,
                    candidate_distance_m=raw_distance_m,
                    required_valid_pixels=required_valid_pixels,
                    bbox_clipped=bbox_clipped,
                    bbox_area_ratio=bbox_area_ratio,
                    bbox_area_change_ratio=bbox_area_change_ratio,
                    required_confirm_frames=rate_guard_confirms,
                    confirm_count=0,
                    rejection_reason="physically_impossible_far_jump",
                    region_count=region_count,
                )

            confirmation_kind = (
                "reanchor" if reanchoring_after_timeout
                else "far_jump" if farther_jump
                else "large_clipped_consensus"
            )
            self._advance_pending_jump(
                raw_distance_m, sample_ts, required_confirms=confirms,
                region_count=region_count if multi_region_consensus else 0,
                kind=confirmation_kind,
            )
            if self._pending_jump_count < confirms:
                if reanchoring_after_timeout:
                    rejection_reason = "reanchor_after_anchor_timeout"
                elif not farther_jump:
                    rejection_reason = "large_clipped_multiregion_confirmation"
                elif effective_anchor_age < strict_age:
                    rejection_reason = "strict_near_to_far_confirmation"
                else:
                    rejection_reason = "aged_anchor_confirmation"
                return self._held_measurement(
                    now,
                    f"distance_jump_pending_{self._pending_jump_count}_of_{confirms}",
                    valid_pixels,
                    candidate_distance_m=raw_distance_m,
                    required_valid_pixels=required_valid_pixels,
                    bbox_clipped=bbox_clipped,
                    bbox_area_ratio=bbox_area_ratio,
                    bbox_area_change_ratio=bbox_area_change_ratio,
                    required_confirm_frames=confirms,
                    confirm_count=self._pending_jump_count,
                    rejection_reason=rejection_reason,
                    region_count=region_count,
                )

            # A confirmed new surface replaces the old median history instead
            # of being blended with the background/person surface it replaced.
            self._clear_distance_history()
            accepted_confirm_count = self._pending_jump_count
            accepted_required_confirms = confirms
            if (
                current_target_id is not None and current_target_id > 0
                and math.isfinite(float(sample_ts)) and 0.0 < float(sample_ts) <= now
                and self._pending_jump_region_count >= 3
                and accepted_confirm_count >= confirms >= 2
            ):
                jump_confirmation = DepthJumpConfirmation(
                    target_id=current_target_id,
                    sample_timestamp=float(sample_ts),
                    distance_m=float(raw_distance_m),
                    kind=confirmation_kind,
                    confirm_count=accepted_confirm_count,
                    required_confirm_frames=confirms,
                    region_count=self._pending_jump_region_count,
                )
            measurement_detail = (
                "depth_reanchored_after_timeout"
                if reanchoring_after_timeout
                else "depth_large_clipped_consensus_confirmed"
                if not farther_jump
                else "depth_multiregion_after_jump_confirm"
            )

        elif closer_jump:
            # A sudden closer return is normally safety-relevant and is
            # accepted immediately.  A clipped large box backed by only one
            # torso region is the exception: CAP569-like samples commonly hit
            # a nearby background/object and must not replace the longitudinal
            # anchor until the same surface is seen again.
            single_region_closer = bool(
                large_bbox
                and bbox_clipped
                and region_count < 2
            )
            if single_region_closer:
                confirms = max(2, int(self.config.jump_confirm_frames))
                self._advance_pending_jump(
                    raw_distance_m, sample_ts, required_confirms=confirms,
                    region_count=region_count, kind="closer_jump",
                )
                if self._pending_jump_count < confirms:
                    return self._held_measurement(
                        now,
                        f"closer_jump_pending_{self._pending_jump_count}_of_{confirms}",
                        valid_pixels,
                        candidate_distance_m=raw_distance_m,
                        required_valid_pixels=required_valid_pixels,
                        bbox_clipped=bbox_clipped,
                        bbox_area_ratio=bbox_area_ratio,
                        bbox_area_change_ratio=bbox_area_change_ratio,
                        required_confirm_frames=confirms,
                        confirm_count=self._pending_jump_count,
                        rejection_reason="single_region_closer_jump",
                        safety_distance_m=raw_distance_m,
                        region_count=region_count,
                    )
                measurement_detail = "depth_multiregion_after_closer_jump_confirm"
            self._clear_distance_history()

        self._reset_pending_jump()
        temporal_filter = append_depth_sample(
            self._distance_history, self._distance_history_timestamps,
            distance_m=raw_distance_m, sample_timestamp=sample_ts, now=time.monotonic(),
            max_age_sec=max(
                0.01, float(self.config.max_frame_age_sec), float(self.config.anchor_strict_age_sec),
            ),
        )
        self._measurement_filter_expired_count = temporal_filter.expired_count
        if not temporal_filter.accepted:
            self._measurement_temporal_status = "out_of_order_jump_observation"
            return self._held_measurement(
                time.monotonic(), "depth_filter_observation_rejected", valid_pixels,
                candidate_distance_m=raw_distance_m, region_count=region_count,
                rejection_reason=temporal_filter.reason,
            )
        filtered_distance_m = float(temporal_filter.distance_m)
        self._last_accepted_distance_m = filtered_distance_m
        self._last_accepted_ts = sample_ts
        self._last_torso_recovery_evidence = torso_recovery_evidence
        self._last_accepted_bbox_area_ratio = bbox_area_ratio
        self._last_accepted_bbox = tuple(bbox)
        self._last_accepted_region_count = region_count
        self._encoder_distance_change_since_accept_m = 0.0
        if filtered_distance_m <= float(self.config.near_guard_distance_m):
            previous_area = self._near_reference_bbox_area_ratio
            self._near_reference_bbox_area_ratio = (
                bbox_area_ratio
                if previous_area is None
                else max(float(previous_area), bbox_area_ratio)
            )
        else:
            self._near_reference_bbox_area_ratio = None
        if now - self._last_log_ts >= max(0.1, float(self.config.log_every_sec)):
            self._last_log_ts = now
            patch_size = max(4, int(self.config.center_patch_size))
            patch_cx = (left + right - 1) // 2
            patch_cy = (top + bottom - 1) // 2
            patch_x0 = max(0, min(int(depth.shape[1]) - patch_size, patch_cx - patch_size // 2 + 1))
            patch_y0 = max(0, min(int(depth.shape[0]) - patch_size, patch_cy - patch_size // 2 + 1))
            log_patch = depth[patch_y0 : patch_y0 + patch_size, patch_x0 : patch_x0 + patch_size]
            log_valid = log_patch[(log_patch >= min_mm) & (log_patch <= max_mm)]
            if int(log_valid.size) > 0:
                p10_mm, p25_mm, p50_mm = self._np.percentile(
                    self._np.sort(log_valid.reshape(-1)),
                    (10.0, 25.0, 50.0),
                )
            else:
                p10_mm = p25_mm = p50_mm = float("nan")
            self.logger.info(
                "Astra depth sample: target=%s mode=%s regions=%s samples=%d/%d "
                "required=%d center_keep=%d p10/p25/p50=%.3f/%.3f/%.3fm "
                "region_count=%d jump_confirmation=%s",
                "none" if target_id is None else int(target_id),
                measurement_detail,
                selected_regions,
                foreground_pixels,
                total_valid_pixels,
                required_valid_pixels,
                center_patch_kept,
                float(p10_mm) / 1000.0,
                float(p25_mm) / 1000.0,
                float(p50_mm) / 1000.0,
                region_count,
                jump_confirmation or "none",
            )
            self.logger.info(
                "Astra目标深度: target=%s control_distance=%.3fm raw_sample=%.3fm mode=%s "
                "regions=%s valid_pixels=%d required=%d clipped=%s bbox_area=%.3f "
                "bbox_area_change=%s previous_anchor_age_ms=%s center_patch=%dx%d "
                "middle_keep=%d center_valid=%d/%d roi=(%d,%d)-(%d,%d) age=%.0fms "
                "align_delay=%.0fms align_error=%.0fms stream_period=%.1fms",
                "none" if target_id is None else int(target_id),
                filtered_distance_m,
                raw_distance_m,
                measurement_detail,
                selected_regions,
                valid_pixels,
                required_valid_pixels,
                bool(bbox_clipped),
                bbox_area_ratio,
                "none"
                if bbox_area_change_ratio is None
                else f"{bbox_area_change_ratio:.3f}",
                "none"
                if anchor_age_sec is None
                else f"{anchor_age_sec * 1000.0:.0f}",
                patch_size,
                patch_size,
                center_patch_kept,
                center_patch_valid,
                center_patch_pixels,
                left,
                top,
                right,
                bottom,
                sample_age * 1000.0,
                max(0.0, float(self.config.rgb_processing_delay_sec)) * 1000.0,
                alignment_error_sec * 1000.0,
                float(self._last_depth_period_ms),
            )
        measurement = AstraDepthMeasurement(
            distance_m=filtered_distance_m,
            raw_distance_m=raw_distance_m,
            sample_age_sec=sample_age,
            valid_pixels=valid_pixels,
            detail=measurement_detail,
            anchor_age_sec=0.0,
            candidate_distance_m=raw_distance_m,
            required_valid_pixels=required_valid_pixels,
            bbox_clipped=bbox_clipped,
            bbox_area_ratio=bbox_area_ratio,
            bbox_area_change_ratio=bbox_area_change_ratio,
            confirm_count=accepted_confirm_count,
            required_confirm_frames=accepted_required_confirms,
            region_count=region_count,
            jump_confirmation=jump_confirmation,
            sample_timestamp=float(sample_ts),
        )
        if jump_confirmation is not None:
            self._log_depth_diagnostic(now, measurement)
        return measurement

    def close(self) -> None:
        self._stop_event.set()
        thread = self._depth_thread
        if thread is not None:
            thread.join(timeout=0.30)
        self._depth_thread = None
        if self.diagnostics is not None:
            self.diagnostics.close()
        for stream in (self._depth_stream, self._color_stream):
            if stream is None:
                continue
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        self._depth_stream = None
        self._color_stream = None
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
        self._device = None
        if self._openni2 is not None:
            try:
                self._openni2.unload()
            except Exception:
                pass
        self._openni2 = None
        self._started = False
        with self._depth_lock:
            self._latest_depth = None
            self._latest_depth_ts = 0.0
            self._depth_history.clear()
        self.logger.info("Astra RGB+Depth已关闭")
