#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Astra Pro RGB/depth runtime used by the follow-car control loop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Optional, Tuple


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
        self._depth_lock = threading.Lock()
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
        self._last_accepted_distance_m: Optional[float] = None
        self._last_accepted_ts = 0.0
        self._pending_jump_distance_m: Optional[float] = None
        self._pending_jump_count = 0
        self._pending_jump_required_confirms = 0
        self._last_processed_depth_ts = 0.0
        self._near_reference_bbox_area_ratio: Optional[float] = None
        self._last_accepted_bbox_area_ratio: Optional[float] = None
        self._last_feedback_ts: Optional[float] = None
        self._encoder_distance_change_since_accept_m = 0.0
        self._last_log_ts = 0.0
        self._last_region_log_ts = 0.0
        self._last_diagnostic_log_ts = 0.0
        self._last_diagnostic_log_key = None

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
            # Astra Pro exposes RGB through /dev/video3. The OpenNI color stream
            # is created only to provide calibration for depth-to-color mapping;
            # starting it yields no frames on this hardware revision.
            self._depth_stream.start()
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
            "Astra配准Depth已启动: device=%s %dx%d@%dFPS registration=depth_to_color RGB=/dev/video3",
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
                width = int(frame.width)
                height = int(frame.height)
                depth = np.frombuffer(
                    frame.get_buffer_as_uint16(), dtype=np.uint16
                ).reshape(height, width).copy()
                now = time.monotonic()
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
            except Exception as exc:
                if not self._stop_event.is_set():
                    self.logger.warning("Astra Depth读取失败: %s", exc)
                    self._stop_event.wait(0.05)

    def read(self) -> Tuple[bool, object]:
        """Color is exposed by Astra's UVC /dev/video3 node, not OpenNI."""
        return False, None

    def release(self) -> None:
        """Camera adapter hook; SensorRuntime owns the actual device close."""
        self._released = True

    def latest_depth_ready(self) -> bool:
        with self._depth_lock:
            return self._latest_depth is not None and self._latest_depth_ts > 0.0

    def wait_until_ready(self, timeout_sec: float = 2.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while time.monotonic() < deadline:
            if self.latest_depth_ready():
                return True
            time.sleep(0.01)
        return self.latest_depth_ready()

    def _aligned_depth_locked(self, now: float, *, use_latest_depth: bool = False):
        """Return the Depth frame closest to the RGB capture time.

        Caller must hold ``_depth_lock``. The latest-frame fallback keeps
        direct unit-test injection and old camera backends compatible.
        """
        if use_latest_depth:
            return (
                self._latest_depth,
                float(self._latest_depth_ts),
                0.0,
            )
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

    def _log_depth_diagnostic(self, now: float, measurement: AstraDepthMeasurement) -> None:
        key = (
            str(measurement.detail),
            int(measurement.confirm_count),
            int(measurement.required_confirm_frames),
        )
        interval = max(0.10, float(self.config.log_every_sec))
        if key == self._last_diagnostic_log_key and now - self._last_diagnostic_log_ts < interval:
            return
        self._last_diagnostic_log_key = key
        self._last_diagnostic_log_ts = now
        self.logger.info(
            "Astra depth diagnostic: target=%s detail=%s rejection=%s "
            "anchor_age_ms=%s candidate_m=%s valid=%d required=%d clipped=%s "
            "bbox_area=%s bbox_area_change=%s confirm=%d/%d encoder_distance_change=%+.3fm",
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
            float(self._encoder_distance_change_since_accept_m),
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
        regions, clipped = self._torso_sampling_regions(
            bbox,
            frame_width,
            frame_height,
            int(depth.shape[1]),
            int(depth.shape[0]),
        )
        candidates = []
        region_diagnostics = []
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

        tolerance = max(0.15, float(self.config.foreground_cluster_span_m) * 0.75)
        groups = []
        for candidate in sorted(candidates, key=lambda item: item.distance_m):
            matching = None
            for group in groups:
                if abs(candidate.distance_m - group["center"]) <= tolerance:
                    matching = group
                    break
            if matching is None:
                matching = {"items": [], "center": float(candidate.distance_m)}
                groups.append(matching)
            matching["items"].append(candidate)
            weights = [max(1, item.pixels) for item in matching["items"]]
            matching["center"] = sum(
                item.distance_m * weight
                for item, weight in zip(matching["items"], weights)
            ) / float(sum(weights))

        reference = self._last_accepted_distance_m
        strict_age = max(0.0, float(self.config.anchor_strict_age_sec))
        expire_age = max(strict_age, float(self.config.anchor_expire_age_sec))
        for group in groups:
            items = group["items"]
            region_count = len({item.region_name for item in items})
            support = sum(item.pixels for item in items)
            coherence = sum(item.spatial_support_fraction for item in items) / float(len(items))
            continuity = 0.0
            if reference is not None and anchor_age_sec is not None and anchor_age_sec < expire_age:
                age_weight = 1.0 if anchor_age_sec < strict_age else 0.45
                continuity = age_weight * max(
                    0.0,
                    1.0 - abs(float(group["center"]) - float(reference)) / max(0.20, float(self.config.max_distance_jump_m)),
                )
            group["score"] = (
                2.0 * float(region_count)
                + math.log1p(max(0, support))
                + coherence
                + 4.0 * continuity
            )
            group["support"] = support
            group["region_count"] = region_count

        selected = max(groups, key=lambda group: float(group["score"]))
        closest = min(groups, key=lambda group: float(group["center"]))
        # A coherent nearer surface is safety-relevant even if fewer torso
        # regions see it. Farther surfaces still pass the age-based guard below.
        if (
            float(closest["center"]) + max(0.15, float(self.config.max_distance_jump_m))
            < float(selected["center"])
            and int(closest["support"]) >= max(
                item.required_pixels for item in closest["items"]
            )
        ):
            selected = closest
        selected_items = selected["items"]
        selected_pixels = sum(item.pixels for item in selected_items)
        selected_valid = sum(item.valid_pixels for item in selected_items)
        selected_required = max(item.required_pixels for item in selected_items)
        region_names = "+".join(sorted({item.region_name for item in selected_items}))
        now = time.monotonic()
        if now - self._last_region_log_ts >= max(0.1, float(self.config.log_every_sec)):
            self._last_region_log_ts = now
            self.logger.info(
                "Astra depth regions: selected=%s selected_distance=%.3fm "
                "selected_pixels=%d clipped=%s details=%s",
                region_names,
                float(selected["center"]),
                selected_pixels,
                bool(clipped),
                "; ".join(region_diagnostics),
            )
        return (
            float(selected["center"]),
            int(selected_pixels),
            int(selected_valid),
            int(selected_required),
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
    ) -> AstraDepthMeasurement:
        now = time.monotonic()
        with self._depth_lock:
            depth, sample_ts, alignment_error_sec = self._aligned_depth_locked(
                now,
                use_latest_depth=bool(use_latest_depth),
            )
        if depth is None or sample_ts <= 0.0:
            return self._held_measurement(now, "no_depth_frame")
        sample_age = max(0.0, now - sample_ts)
        if sample_age > max(0.01, float(self.config.max_frame_age_sec)):
            return self._held_measurement(now, "stale_depth_frame")

        current_target_id = None if target_id is None else int(target_id)
        if current_target_id != self._last_target_id:
            self._last_target_id = current_target_id
            self._distance_history.clear()
            self._last_accepted_distance_m = None
            self._last_accepted_ts = 0.0
            self._pending_jump_distance_m = None
            self._pending_jump_count = 0
            self._pending_jump_required_confirms = 0
            self._last_processed_depth_ts = 0.0
            self._near_reference_bbox_area_ratio = None
            self._last_accepted_bbox_area_ratio = None
            self._last_feedback_ts = None
            self._encoder_distance_change_since_accept_m = 0.0

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
        bbox_area_ratio, _ = self._bbox_depth_evidence(
            bbox,
            frame_width,
            frame_height,
        )
        _, bbox_clipped = self._torso_sampling_regions(
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
        if valid_pixels < required_valid_pixels:
            return self._held_measurement(
                now,
                "insufficient_depth_pixels",
                valid_pixels,
                required_valid_pixels=required_valid_pixels,
                bbox_clipped=bbox_clipped,
                bbox_area_ratio=bbox_area_ratio,
                bbox_area_change_ratio=bbox_area_change_ratio,
            )

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
        required_valid_pixels = max(required_valid_pixels, region_required_pixels)
        _center_patch_distance, center_patch_valid, center_patch_pixels, center_patch_kept = (
            self._select_center_patch_distance(depth, left, top, right, bottom)
        )
        measurement_detail = "depth_multiregion"
        if raw_distance_m is None:
            # Multi-region sampling is authoritative. The whole torso ROI is a
            # last-resort coherent foreground fallback, never a plain median.
            raw_distance_m, foreground_pixels, total_valid_pixels = (
                self._select_foreground_cluster(valid)
            )
            if raw_distance_m is None:
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
            if anchored_m is not None:
                raw_distance_m = float(anchored_m)
                foreground_pixels = int(anchored_pixels)
                total_valid_pixels = int(anchored_total)
                measurement_detail = "depth_foreground_fallback_large_bbox_anchor"
            elif not anchor_relaxed:
                nearest_m, nearest_pixels, nearest_total = self._select_foreground_cluster(valid)
                if (
                    nearest_m is None
                    or float(nearest_m) > float(self.config.large_bbox_guard_max_distance_m)
                ):
                    return self._held_measurement(
                        now,
                        "far_background_guard_large_bbox",
                        valid_pixels,
                        candidate_distance_m=raw_distance_m,
                        required_valid_pixels=required_valid_pixels,
                        bbox_clipped=bbox_clipped,
                        bbox_area_ratio=bbox_area_ratio,
                        bbox_area_change_ratio=bbox_area_change_ratio,
                    )
                raw_distance_m = float(nearest_m)
                foreground_pixels = int(nearest_pixels)
                total_valid_pixels = int(nearest_total)
                measurement_detail = "depth_foreground_fallback_large_bbox_nearest"
        if sample_ts <= self._last_processed_depth_ts + 1e-9:
            if self._pending_jump_count > 0:
                confirms = max(
                    1,
                    int(self._pending_jump_required_confirms),
                    int(self.config.jump_confirm_frames),
                )
                return self._held_measurement(
                    now,
                    f"distance_jump_pending_{self._pending_jump_count}_of_{confirms}",
                    valid_pixels,
                    candidate_distance_m=self._pending_jump_distance_m,
                    required_valid_pixels=required_valid_pixels,
                    bbox_clipped=bbox_clipped,
                    bbox_area_ratio=bbox_area_ratio,
                    bbox_area_change_ratio=bbox_area_change_ratio,
                    required_confirm_frames=confirms,
                    confirm_count=self._pending_jump_count,
                    rejection_reason="reused_depth_frame",
                )
            if self._last_accepted_distance_m is not None:
                return AstraDepthMeasurement(
                    distance_m=float(self._last_accepted_distance_m),
                    raw_distance_m=None,
                    sample_age_sec=sample_age,
                    valid_pixels=valid_pixels,
                    detail=f"{measurement_detail}_reused_hold",
                    anchor_age_sec=anchor_age_sec,
                    required_valid_pixels=required_valid_pixels,
                    bbox_clipped=bbox_clipped,
                    bbox_area_ratio=bbox_area_ratio,
                    bbox_area_change_ratio=bbox_area_change_ratio,
                )
        self._last_processed_depth_ts = sample_ts
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
        reanchoring_after_timeout = False
        if farther_jump:
            confirms = max(1, int(self.config.jump_confirm_frames))
            far_from_near = bool(
                float(last) <= float(self.config.near_guard_distance_m)
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
                    if guard_detail is not None and not motion_supported:
                        self._pending_jump_distance_m = None
                        self._pending_jump_count = 0
                        self._pending_jump_required_confirms = 0
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
            elif effective_anchor_age > expire_age:
                confirms = max(confirms, int(self.config.reanchor_confirm_frames))
                reanchoring_after_timeout = True

            pending = self._pending_jump_distance_m
            if pending is not None and abs(raw_distance_m - float(pending)) <= max(0.15, jump_limit * 0.35):
                self._pending_jump_count += 1
            else:
                self._pending_jump_distance_m = raw_distance_m
                self._pending_jump_count = 1
            self._pending_jump_required_confirms = int(confirms)
            if self._pending_jump_count < confirms:
                if reanchoring_after_timeout:
                    rejection_reason = "reanchor_after_anchor_timeout"
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
                )

            # A confirmed new surface replaces the old median history instead
            # of being blended with the background/person surface it replaced.
            self._distance_history.clear()
            measurement_detail = (
                "depth_reanchored_after_timeout"
                if reanchoring_after_timeout
                else "depth_multiregion_after_jump_confirm"
            )

        elif closer_jump:
            # Any coherent sudden closer return is safety-relevant. Accept it
            # immediately; only farther jumps wait for confirmation.
            self._distance_history.clear()

        self._pending_jump_distance_m = None
        self._pending_jump_count = 0
        self._pending_jump_required_confirms = 0
        self._distance_history.append(raw_distance_m)
        filtered_distance_m = float(self._np.median(tuple(self._distance_history)))
        self._last_accepted_distance_m = filtered_distance_m
        self._last_accepted_ts = sample_ts
        self._last_accepted_bbox_area_ratio = bbox_area_ratio
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
                "required=%d center_keep=%d p10/p25/p50=%.3f/%.3f/%.3fm",
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
            )
            self.logger.info(
                "Astra目标深度: target=%s distance=%.3fm raw=%.3fm mode=%s "
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
        return AstraDepthMeasurement(
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
        )

    def close(self) -> None:
        self._stop_event.set()
        thread = self._depth_thread
        if thread is not None:
            thread.join(timeout=0.30)
        self._depth_thread = None
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
