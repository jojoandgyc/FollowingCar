#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensor lifecycle and read helpers for the follow-car runtime.

The concrete board HALs are imported lazily inside module classes so this
package can be imported on a development machine without the board libraries.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, List, Optional, Tuple

from .astra_depth import AstraDepthConfig, AstraDepthMeasurement, AstraDepthRuntime
from .control_types import ObstacleState


@dataclass(frozen=True)
class SensorRuntimeConfig:
    ir_enable: bool
    ultrasonic_enable: bool
    mmwave_enable: bool
    imu_enable: bool
    imu_fail_soft: bool
    imu_log_enable: bool
    imu_log_every_sec: float
    side_ir_blocks_rotation: bool
    astra_depth_enable: bool = False
    astra_depth_openni_path: str = "/home/topeet/AstraSDK/arm64_openni2"
    astra_depth_width: int = 640
    astra_depth_height: int = 480
    astra_depth_fps: int = 30
    astra_depth_min_distance_m: float = 0.35
    astra_depth_max_distance_m: float = 8.0
    astra_depth_max_frame_age_sec: float = 0.25
    astra_depth_hold_sec: float = 0.20
    astra_depth_rgb_processing_delay_sec: float = 0.13
    astra_depth_roi_left_ratio: float = 0.32
    astra_depth_roi_right_ratio: float = 0.68
    astra_depth_roi_top_ratio: float = 0.22
    astra_depth_roi_bottom_ratio: float = 0.68
    astra_depth_min_valid_pixels: int = 80
    astra_depth_dynamic_min_valid_floor: int = 20
    astra_depth_dynamic_min_valid_fraction: float = 0.03
    astra_depth_median_window: int = 3
    astra_depth_foreground_cluster_span_m: float = 0.40
    astra_depth_foreground_cluster_min_fraction: float = 0.06
    astra_depth_foreground_spatial_support_fraction: float = 0.55
    astra_depth_torso_region_min_size_px: int = 16
    astra_depth_torso_region_max_size_px: int = 64
    astra_depth_center_patch_size: int = 16
    astra_depth_center_patch_keep_count: int = 4
    astra_depth_center_patch_min_valid_fraction: float = 0.25
    astra_depth_large_bbox_guard_area_ratio: float = 0.35
    astra_depth_large_bbox_guard_height_ratio: float = 0.90
    astra_depth_large_bbox_guard_max_distance_m: float = 2.50
    astra_depth_max_distance_jump_m: float = 0.80
    astra_depth_jump_confirm_frames: int = 2
    astra_depth_near_guard_distance_m: float = 1.80
    astra_depth_near_far_jump_confirm_frames: int = 5
    astra_depth_anchor_strict_age_sec: float = 0.60
    astra_depth_anchor_expire_age_sec: float = 1.50
    astra_depth_reanchor_confirm_frames: int = 3
    astra_depth_motion_confirm_frames: int = 3
    astra_depth_motion_reverse_min_m: float = 0.08
    astra_depth_near_far_jump_max_bbox_ratio: float = 0.90
    astra_depth_near_far_jump_edge_margin_ratio: float = 0.02
    astra_depth_encoder_wheel_circumference_m: float = 0.60
    astra_depth_log_every_sec: float = 1.0
    side_ir_confirm_sec: float = 0.10
    side_ir_release_sec: float = 0.20
    mmwave_async_enable: bool = True
    mmwave_poll_interval_sec: float = 0.04
    mmwave_cache_window_sec: float = 0.80


class SensorRuntime:
    """Own board sensor lifecycle and lightweight read helpers.

    This runtime keeps HAL imports and start/stop calls out of the request
    entrypoint.  Higher-level policy still lives in controllers and distance
    matching still lives in distance_runtime.
    """

    def __init__(
        self,
        config: SensorRuntimeConfig,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.ir = None
        self.ultrasonic = None
        self.mmwave_radar = None
        self.imu = None
        self.astra_depth: Optional[AstraDepthRuntime] = None
        self.imu_runtime_enabled = False
        self.last_imu_log_ts = 0.0
        self._mmwave_samples = deque()
        self._mmwave_lock = threading.Lock()
        self._mmwave_stop = threading.Event()
        self._mmwave_thread = None
        self._last_mmwave_warn_ts = 0.0
        self._side_ir_raw_since = {"left": None, "right": None}
        self._side_ir_clear_since = {"left": None, "right": None}
        self._side_ir_latched = {"left": False, "right": False}
        self._side_ir_lock = threading.Lock()

    def start(self) -> None:
        c = self.config
        if c.astra_depth_enable:
            self.astra_depth = AstraDepthRuntime(
                AstraDepthConfig(
                    openni_path=c.astra_depth_openni_path,
                    width=int(c.astra_depth_width),
                    height=int(c.astra_depth_height),
                    fps=int(c.astra_depth_fps),
                    min_distance_m=float(c.astra_depth_min_distance_m),
                    max_distance_m=float(c.astra_depth_max_distance_m),
                    max_frame_age_sec=float(c.astra_depth_max_frame_age_sec),
                    hold_sec=float(c.astra_depth_hold_sec),
                    rgb_processing_delay_sec=float(c.astra_depth_rgb_processing_delay_sec),
                    roi_left_ratio=float(c.astra_depth_roi_left_ratio),
                    roi_right_ratio=float(c.astra_depth_roi_right_ratio),
                    roi_top_ratio=float(c.astra_depth_roi_top_ratio),
                    roi_bottom_ratio=float(c.astra_depth_roi_bottom_ratio),
                    min_valid_pixels=int(c.astra_depth_min_valid_pixels),
                    dynamic_min_valid_floor=int(c.astra_depth_dynamic_min_valid_floor),
                    dynamic_min_valid_fraction=float(
                        c.astra_depth_dynamic_min_valid_fraction
                    ),
                    median_window=int(c.astra_depth_median_window),
                    foreground_cluster_span_m=float(
                        c.astra_depth_foreground_cluster_span_m
                    ),
                    foreground_cluster_min_fraction=float(
                        c.astra_depth_foreground_cluster_min_fraction
                    ),
                    foreground_spatial_support_fraction=float(
                        c.astra_depth_foreground_spatial_support_fraction
                    ),
                    torso_region_min_size_px=int(c.astra_depth_torso_region_min_size_px),
                    torso_region_max_size_px=int(c.astra_depth_torso_region_max_size_px),
                    center_patch_size=int(c.astra_depth_center_patch_size),
                    center_patch_keep_count=int(c.astra_depth_center_patch_keep_count),
                    center_patch_min_valid_fraction=float(
                        c.astra_depth_center_patch_min_valid_fraction
                    ),
                    large_bbox_guard_area_ratio=float(
                        c.astra_depth_large_bbox_guard_area_ratio
                    ),
                    large_bbox_guard_height_ratio=float(
                        c.astra_depth_large_bbox_guard_height_ratio
                    ),
                    large_bbox_guard_max_distance_m=float(
                        c.astra_depth_large_bbox_guard_max_distance_m
                    ),
                    max_distance_jump_m=float(c.astra_depth_max_distance_jump_m),
                    jump_confirm_frames=int(c.astra_depth_jump_confirm_frames),
                    near_guard_distance_m=float(c.astra_depth_near_guard_distance_m),
                    near_far_jump_confirm_frames=int(
                        c.astra_depth_near_far_jump_confirm_frames
                    ),
                    anchor_strict_age_sec=float(c.astra_depth_anchor_strict_age_sec),
                    anchor_expire_age_sec=float(c.astra_depth_anchor_expire_age_sec),
                    reanchor_confirm_frames=int(c.astra_depth_reanchor_confirm_frames),
                    motion_confirm_frames=int(c.astra_depth_motion_confirm_frames),
                    motion_reverse_min_m=float(c.astra_depth_motion_reverse_min_m),
                    near_far_jump_max_bbox_ratio=float(
                        c.astra_depth_near_far_jump_max_bbox_ratio
                    ),
                    near_far_jump_edge_margin_ratio=float(
                        c.astra_depth_near_far_jump_edge_margin_ratio
                    ),
                    encoder_wheel_circumference_m=float(
                        c.astra_depth_encoder_wheel_circumference_m
                    ),
                    log_every_sec=float(c.astra_depth_log_every_sec),
                ),
                logger=self.logger,
            )
            self.astra_depth.start()
        else:
            self.logger.info("Astra Depth模块已关闭")

        if c.ir_enable:
            from ir_hal import IR

            ret = IR.init()
            if ret != 0:
                raise RuntimeError("IR传感器初始化失败")
            self.ir = IR
            self.logger.info("IR传感器初始化成功（前方、左侧、右侧）")
        else:
            self.logger.info("IR传感器模块已关闭")

        if c.ultrasonic_enable:
            from utrasonic_hal import Utrasonic

            ret = Utrasonic.init()
            if ret != 0:
                raise RuntimeError("Utrasonic传感器初始化失败")
            self.ultrasonic = Utrasonic
            self.logger.info("Utrasonic传感器初始化成功")
        else:
            self.logger.info("Utrasonic传感器模块已关闭")

        if c.mmwave_enable:
            from mmwave_hal import MmWaveRadar

            ret = MmWaveRadar.init()
            if ret != 0:
                raise RuntimeError("毫米波雷达初始化失败")
            self.mmwave_radar = MmWaveRadar
            self.logger.info("毫米波雷达初始化成功")
            self._start_mmwave_cache_thread()
        else:
            self.logger.info("毫米波雷达模块已关闭")

        if c.imu_enable:
            try:
                from imu_hal import IMU
            except Exception as exc:
                if c.imu_fail_soft:
                    self.logger.warning("IMU HAL 导入失败，已按 fail_soft 跳过: %s", exc)
                else:
                    raise RuntimeError(f"IMU HAL 导入失败: {exc}") from exc
            else:
                ret = IMU.init()
                if ret != 0:
                    if c.imu_fail_soft:
                        self.logger.warning("ICM20600 IMU 初始化失败，已按 fail_soft 跳过: %s", IMU.info())
                    else:
                        raise RuntimeError("ICM20600 IMU 初始化失败")
                else:
                    self.imu = IMU
                    self.imu_runtime_enabled = True
                    self.logger.info("ICM20600 IMU 初始化成功: %s", IMU.info())
        else:
            self.logger.info("ICM20600 IMU 模块已关闭")

    def close(self) -> None:
        self._stop_mmwave_cache_thread()
        if self.astra_depth is not None:
            try:
                self.astra_depth.close()
            except Exception as exc:
                self.logger.warning("关闭Astra RGB+Depth时出错: %s", exc)
            self.astra_depth = None
        try:
            if self.config.ir_enable and self.ir is not None:
                self.ir.deinit()
            if self.config.ultrasonic_enable and self.ultrasonic is not None:
                self.ultrasonic.deinit()
            if self.config.mmwave_enable and self.mmwave_radar is not None:
                self.mmwave_radar.deinit()
            if self.config.imu_enable and self.imu is not None:
                self.imu.deinit()
            self.logger.info("传感器已关闭")
        except Exception as exc:
            self.logger.warning("关闭传感器时出错: %s", exc)

    def get_astra_camera(self) -> Optional[AstraDepthRuntime]:
        if not self.config.astra_depth_enable:
            return None
        return self.astra_depth

    def get_astra_target_distance(
        self,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        *,
        target_id: Optional[int] = None,
        use_latest_depth: bool = False,
        steering_feedback=None,
    ) -> AstraDepthMeasurement:
        runtime = self.astra_depth
        if not self.config.astra_depth_enable or runtime is None:
            return AstraDepthMeasurement(None, None, None, 0, "disabled")
        return runtime.measure_target(
            bbox,
            frame_width,
            frame_height,
            target_id=target_id,
            use_latest_depth=bool(use_latest_depth),
            steering_feedback=steering_feedback,
        )

    def _start_mmwave_cache_thread(self) -> None:
        if not self.config.mmwave_async_enable:
            return
        if self.mmwave_radar is None:
            return
        if self._mmwave_thread is not None and self._mmwave_thread.is_alive():
            return
        self._mmwave_stop.clear()
        self._mmwave_thread = threading.Thread(target=self._mmwave_cache_loop, daemon=True)
        self._mmwave_thread.start()
        self.logger.info(
            "毫米波异步缓存已启动 interval=%.3fs window=%.3fs",
            float(self.config.mmwave_poll_interval_sec),
            float(self.config.mmwave_cache_window_sec),
        )

    def _stop_mmwave_cache_thread(self) -> None:
        thread = self._mmwave_thread
        if thread is None:
            return
        self._mmwave_stop.set()
        thread.join(timeout=1.0)
        self._mmwave_thread = None

    def _mmwave_cache_loop(self) -> None:
        interval = max(0.01, float(self.config.mmwave_poll_interval_sec))
        window = max(interval, float(self.config.mmwave_cache_window_sec))
        while not self._mmwave_stop.is_set():
            start = time.monotonic()
            try:
                targets = list(self.mmwave_radar.get_targets()) if self.mmwave_radar is not None else []
                ts = time.monotonic()
                with self._mmwave_lock:
                    self._mmwave_samples.append((ts, targets))
                    while self._mmwave_samples and ts - self._mmwave_samples[0][0] > window:
                        self._mmwave_samples.popleft()
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_mmwave_warn_ts >= 2.0:
                    self._last_mmwave_warn_ts = now
                    self.logger.warning("毫米波异步读取失败: %s", exc)
            elapsed = time.monotonic() - start
            self._mmwave_stop.wait(max(0.0, interval - elapsed))

    def get_obstacle_status(self) -> ObstacleState:
        ir = self.ir
        if not self.config.ir_enable or ir is None:
            return ObstacleState()
        now = time.monotonic()
        raw_left = bool(ir.is_triggered(ir.IDX_2))
        raw_right = bool(ir.is_triggered(ir.IDX_0))
        return ObstacleState(
            front=bool(ir.is_triggered(ir.IDX_1)),
            left=self._filter_side_ir("left", raw_left, now),
            right=self._filter_side_ir("right", raw_right, now),
        )

    def get_raw_obstacle_status(self) -> ObstacleState:
        """Read the instantaneous IR state for the hard-stop safety path.

        The controller keeps a short side-IR debounce to reject electrical
        glitches.  The action executor must still be able to stop immediately
        when a raw side sensor goes active while the car is moving.
        """
        ir = self.ir
        if not self.config.ir_enable or ir is None:
            return ObstacleState()
        return ObstacleState(
            front=bool(ir.is_triggered(ir.IDX_1)),
            left=bool(ir.is_triggered(ir.IDX_2)),
            right=bool(ir.is_triggered(ir.IDX_0)),
        )

    def _filter_side_ir(self, side: str, raw_triggered: bool, now: float) -> bool:
        """Debounce side IR while keeping the front IR path immediate."""
        with self._side_ir_lock:
            return self._filter_side_ir_locked(side, raw_triggered, now)

    def _filter_side_ir_locked(self, side: str, raw_triggered: bool, now: float) -> bool:
        if raw_triggered:
            self._side_ir_clear_since[side] = None
            if self._side_ir_latched[side]:
                return True
            started_at = self._side_ir_raw_since[side]
            if started_at is None:
                self._side_ir_raw_since[side] = float(now)
                started_at = float(now)
            if float(now) - float(started_at) >= max(0.0, float(self.config.side_ir_confirm_sec)):
                self._side_ir_latched[side] = True
                self.logger.info(
                    "侧红外确认触发: 方向=%s 连续时间=%.3f秒 阈值=%.3f秒",
                    side,
                    float(now) - float(started_at),
                    float(self.config.side_ir_confirm_sec),
                )
            return bool(self._side_ir_latched[side])

        self._side_ir_raw_since[side] = None
        if not self._side_ir_latched[side]:
            self._side_ir_clear_since[side] = None
            return False
        clear_started_at = self._side_ir_clear_since[side]
        if clear_started_at is None:
            self._side_ir_clear_since[side] = float(now)
            return True
        if float(now) - float(clear_started_at) < max(0.0, float(self.config.side_ir_release_sec)):
            return True
        self._side_ir_latched[side] = False
        self._side_ir_clear_since[side] = None
        self.logger.info(
            "侧红外确认释放: 方向=%s 连续清除时间=%.3f秒 阈值=%.3f秒",
            side,
            float(now) - float(clear_started_at),
            float(self.config.side_ir_release_sec),
        )
        return False

    def is_front_ir_triggered(self) -> bool:
        ir = self.ir
        if not self.config.ir_enable or ir is None:
            return False
        return bool(ir.is_triggered(ir.IDX_1))

    def get_ultrasonic_distance_cm(self) -> Optional[float]:
        if not self.config.ultrasonic_enable or self.ultrasonic is None:
            return None
        return self.ultrasonic.get_distance()

    def get_mmwave_distance_cm(self) -> Optional[float]:
        if not self.config.mmwave_enable or self.mmwave_radar is None:
            return None
        return self.mmwave_radar.get_distance()

    def get_mmwave_targets(self) -> list:
        if not self.config.mmwave_enable or self.mmwave_radar is None:
            return []
        if self.config.mmwave_async_enable:
            targets, _ts = self.get_mmwave_targets_at(None)
            return targets
        return self.mmwave_radar.get_targets()

    def get_mmwave_targets_at(self, target_ts: Optional[float], max_age_sec: Optional[float] = None) -> Tuple[List[Any], Optional[float]]:
        """Return cached mmwave targets closest to target_ts, plus sample time.

        target_ts uses time.monotonic().  If target_ts is None, the newest cached
        sample is returned.  When async cache is disabled or empty, this falls
        back to a direct read so smoke tests and older configs still work.
        """
        if not self.config.mmwave_enable or self.mmwave_radar is None:
            return [], None
        if self.config.mmwave_async_enable:
            now = time.monotonic()
            max_age = float(max_age_sec) if max_age_sec is not None else float(self.config.mmwave_cache_window_sec)
            with self._mmwave_lock:
                samples = list(self._mmwave_samples)
            if samples:
                if target_ts is None:
                    sample_ts, targets = samples[-1]
                else:
                    sample_ts, targets = min(samples, key=lambda item: abs(float(item[0]) - float(target_ts)))
                if now - float(sample_ts) <= max_age:
                    return list(targets), float(sample_ts)
        try:
            return list(self.mmwave_radar.get_targets()), time.monotonic()
        except Exception:
            return [], None

    def maybe_log_imu_sample(self, frame_index: int) -> None:
        c = self.config
        imu = self.imu
        if not c.imu_enable or not self.imu_runtime_enabled or not c.imu_log_enable or imu is None:
            return
        now = time.time()
        if now - self.last_imu_log_ts < c.imu_log_every_sec:
            return
        try:
            snapshot: Any = imu.poll(0.0)
        except Exception as exc:
            self.last_imu_log_ts = now
            self.logger.warning("IMU 读取失败: %s", exc)
            return

        accel = snapshot.get("accel") if isinstance(snapshot, dict) else None
        gyro = snapshot.get("gyro") if isinstance(snapshot, dict) else None
        if not accel and not gyro:
            return

        self.last_imu_log_ts = now
        accel_raw = None if not accel else accel.get("raw")
        accel_g = None if not accel else tuple(round(float(v), 4) for v in accel.get("g", ()))
        gyro_raw = None if not gyro else gyro.get("raw")
        gyro_dps = None if not gyro else tuple(round(float(v), 3) for v in gyro.get("dps", ()))
        self.logger.info(
            "IMU 样本: 帧=%d 加速度原始值=%s 加速度(g)=%s 陀螺仪原始值=%s 角速度(度/秒)=%s",
            frame_index,
            accel_raw,
            accel_g,
            gyro_raw,
            gyro_dps,
        )
