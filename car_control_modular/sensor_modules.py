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
        self.imu_runtime_enabled = False
        self.last_imu_log_ts = 0.0
        self._mmwave_samples = deque()
        self._mmwave_lock = threading.Lock()
        self._mmwave_stop = threading.Event()
        self._mmwave_thread = None
        self._last_mmwave_warn_ts = 0.0

    def start(self) -> None:
        c = self.config
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
        return ObstacleState(
            front=bool(ir.is_triggered(ir.IDX_1)),
            left=bool(ir.is_triggered(ir.IDX_2)),
            right=bool(ir.is_triggered(ir.IDX_0)),
        )

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
            self.logger.warning("IMU read failed: %s", exc)
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
            "IMU sample frame=%d accel_raw=%s accel_g=%s gyro_raw=%s gyro_dps=%s",
            frame_index,
            accel_raw,
            accel_g,
            gyro_raw,
            gyro_dps,
        )
