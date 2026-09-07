from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from .control_types import DistanceState, PersonTarget, SteeringFeedback
from .distance_fusion import DistanceFusionConfig, VisionRadarEncoderDistanceFusion


@dataclass(frozen=True)
class DistanceRuntimeConfig:
    distance_source: str
    vision_mmwave_source_aliases: frozenset
    module_mmwave_enable: bool
    module_ultrasonic_enable: bool
    vision_hfov_deg: float
    vision_mmwave_angle_margin_deg: float
    vision_mmwave_angle_offset_deg: float
    vision_mmwave_angle_sign: float
    vision_mmwave_match_mode: str
    vision_mmwave_distance_bias_m: float
    vision_mmwave_min_distance_m: float
    vision_mmwave_max_distance_m: float
    vision_mmwave_min_output_distance_m: float
    vision_mmwave_hard_stop_ttl_sec: float
    vision_mmwave_log_every_frames: int
    vision_mmwave_unmatched_hold_sec: float = 2.50
    vision_mmwave_unmatched_hold_min_distance_m: float = 1.40
    vision_mmwave_far_margin_start_m: float = 2.50
    vision_mmwave_far_angle_margin_extra_deg: float = 4.0
    vision_mmwave_latency_sec: float = 0.10
    vision_mmwave_cache_max_age_sec: float = 0.50
    vision_mmwave_use_async_cache: bool = True
    vision_mmwave_angle_tie_margin_deg: float = 0.0
    vision_mmwave_max_center_angle_diff_deg: float = 26.0
    vision_mmwave_max_distance_jump_m: float = 0.60
    vision_mmwave_max_angle_jump_deg: float = 15.0
    vision_mmwave_switch_confirm_frames: int = 3
    vision_mmwave_continuity_memory_sec: float = 2.50
    # 近距离关联在视觉轨迹号短暂变化时继续保留，防止远雷达点抢占目标。
    vision_mmwave_near_distance_lock_m: float = 2.00
    vision_mmwave_distance_score_weight: float = 6.0
    vision_mmwave_motion_min_angle_delta_deg: float = 3.0
    vision_mmwave_motion_hint_ttl_sec: float = 0.80
    ultrasonic_min_distance_m: float = 0.02
    ultrasonic_max_distance_m: float = 8.0
    ultrasonic_filter_window: int = 3
    ultrasonic_target_confirm_frames: int = 2
    ultrasonic_brake_confirm_frames: int = 2
    ultrasonic_hysteresis_m: float = 0.25
    ultrasonic_immediate_brake_m: float = 0.35
    vision_mmwave_fusion_enable: bool = False
    vision_mmwave_fusion_radar_median_window: int = 3
    vision_mmwave_fusion_visual_weight: float = 0.80
    vision_mmwave_fusion_encoder_wheel_circumference_m: float = 0.60
    vision_mmwave_fusion_encoder_max_step_m: float = 0.12
    vision_mmwave_fusion_bbox_min_height_ratio: float = 0.08
    vision_mmwave_fusion_bbox_max_height_ratio: float = 0.90
    vision_mmwave_fusion_radar_recovery_alpha: float = 0.35
    vision_mmwave_fusion_max_distance_increase_mps: float = 2.0
    vision_mmwave_fusion_min_confidence: float = 0.40
    module_astra_depth_enable: bool = False
    vision_depth_source_aliases: frozenset = frozenset(
        {"vision_depth", "vision-depth", "astra_depth", "astra-depth"}
    )
    vision_depth_max_distance_jump_m: float = 0.0
    vision_depth_jump_confirm_frames: int = 3


class DistanceRuntime:
    """Distance sensor selection and vision-gated mmwave matching."""

    def __init__(
        self,
        owner,
        config: DistanceRuntimeConfig,
        *,
        sensor_runtime,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.owner = owner
        self.config = config
        self.sensor_runtime = sensor_runtime
        self.logger = logger or logging.getLogger(__name__)
        self.last_vision_mmwave_distance_m: Optional[float] = None
        self.last_vision_mmwave_ts = 0.0
        self.last_vision_mmwave_log_key = None
        self.last_vision_mmwave_detail = ""
        self.last_vision_mmwave_sample_age_sec: Optional[float] = None
        self.last_vision_mmwave_target_angle_deg: Optional[float] = None
        self.last_vision_mmwave_matched_angle_deg: Optional[float] = None
        self.last_vision_mmwave_aligned_target_angle_deg: Optional[float] = None
        self.last_vision_mmwave_motion_direction = "unknown"
        self._vision_angle_history = deque(maxlen=24)
        self._vision_angle_history_track_id: Optional[int] = None
        self._mmwave_associated_track_id: Optional[int] = None
        self._mmwave_accepted_angle_deg: Optional[float] = None
        self._mmwave_accepted_distance_m: Optional[float] = None
        self._mmwave_accepted_ts = 0.0
        self._mmwave_pending_candidate: Optional[Dict[str, Any]] = None
        self._mmwave_pending_count = 0
        self._mmwave_angle_history = deque(maxlen=16)
        self._distance_fusion = VisionRadarEncoderDistanceFusion(
            DistanceFusionConfig(
                enabled=bool(config.vision_mmwave_fusion_enable),
                radar_median_window=int(config.vision_mmwave_fusion_radar_median_window),
                visual_weight=float(config.vision_mmwave_fusion_visual_weight),
                encoder_wheel_circumference_m=float(
                    config.vision_mmwave_fusion_encoder_wheel_circumference_m
                ),
                encoder_max_step_m=float(config.vision_mmwave_fusion_encoder_max_step_m),
                bbox_min_height_ratio=float(config.vision_mmwave_fusion_bbox_min_height_ratio),
                bbox_max_height_ratio=float(config.vision_mmwave_fusion_bbox_max_height_ratio),
                radar_recovery_alpha=float(config.vision_mmwave_fusion_radar_recovery_alpha),
                max_distance_increase_mps=float(
                    config.vision_mmwave_fusion_max_distance_increase_mps
                ),
                hold_max_sec=float(config.vision_mmwave_unmatched_hold_sec),
                min_confidence=float(config.vision_mmwave_fusion_min_confidence),
                min_distance_m=float(config.vision_mmwave_min_output_distance_m),
                max_distance_m=float(config.vision_mmwave_max_distance_m),
            )
        )
        # Astra短时丢点只允许保持200ms；期间由人物框尺度和编码器共同更新，
        # Depth恢复后重新作为绝对距离锚点。毫米波融合配置与此实例互不影响。
        self._vision_depth_fusion = VisionRadarEncoderDistanceFusion(
            DistanceFusionConfig(
                enabled=True,
                radar_median_window=1,
                visual_weight=0.78,
                encoder_wheel_circumference_m=float(
                    config.vision_mmwave_fusion_encoder_wheel_circumference_m
                ),
                encoder_max_step_m=float(config.vision_mmwave_fusion_encoder_max_step_m),
                bbox_min_height_ratio=0.05,
                bbox_max_height_ratio=0.98,
                radar_recovery_alpha=0.60,
                max_distance_increase_mps=2.5,
                hold_max_sec=0.20,
                min_confidence=0.45,
                min_distance_m=0.03,
                max_distance_m=8.0,
                fresh_far_jump_m=float(config.vision_depth_max_distance_jump_m),
                fresh_far_jump_confirm_frames=int(config.vision_depth_jump_confirm_frames),
            )
        )
        sample_window = max(1, int(config.ultrasonic_filter_window))
        self._ultrasonic_samples = deque(maxlen=sample_window)
        self._ultrasonic_target_close_count = 0
        self._ultrasonic_brake_close_count = 0
        self._ultrasonic_target_latched = False
        self._ultrasonic_brake_latched = False
        self._last_vision_depth_target: Optional[PersonTarget] = None
        self._last_vision_depth_frame_size: Tuple[int, int] = (0, 0)
        self.last_distance_state = DistanceState()

    def _sensor_source(self) -> Optional[str]:
        c = self.config
        source = c.distance_source
        if source == "auto":
            if c.module_mmwave_enable:
                source = "mmwave"
            elif c.module_ultrasonic_enable:
                source = "ultrasonic"
            else:
                return None
        if source in ("none", "off", "disabled"):
            return None
        if source in c.vision_mmwave_source_aliases:
            return None
        if source in c.vision_depth_source_aliases:
            return None
        if source not in ("mmwave", "ultrasonic"):
            self.logger.warning("未知距离来源 DISTANCE_SOURCE=%s，跳过距离读取", source)
            return None
        if source == "mmwave" and not c.module_mmwave_enable:
            return None
        if source == "ultrasonic" and not c.module_ultrasonic_enable:
            return None
        return source

    def _read_sensor_distance_raw(self, source: str) -> Optional[float]:
        c = self.config
        try:
            if source == "ultrasonic":
                distance_cm = self.sensor_runtime.get_ultrasonic_distance_cm()
            else:
                distance_cm = self.sensor_runtime.get_mmwave_distance_cm()
            if distance_cm is None or distance_cm <= 0:
                return None
            distance_m = float(distance_cm) / 100.0
            if source == "ultrasonic":
                min_m = float(c.ultrasonic_min_distance_m)
                max_m = float(c.ultrasonic_max_distance_m)
            else:
                min_m = 0.02
                max_m = 8.0
            if min_m < distance_m < max_m:
                return distance_m
            return None
        except Exception as exc:
            self.logger.warning("距离传感器读取失败(source=%s): %s", source, exc)
            return None

    def get_sensor_distance(self) -> Optional[float]:
        """Read the configured raw distance source and return meters."""
        source = self._sensor_source()
        if source is None:
            return None
        return self._read_sensor_distance_raw(source)

    def _distance_trigger(self, distance_m: Optional[float], target_distance_m: Optional[float], brake_distance_m: Optional[float]) -> str:
        if distance_m is None:
            return "none"
        if brake_distance_m is not None and distance_m < float(brake_distance_m):
            return "brake"
        if target_distance_m is not None and distance_m <= float(target_distance_m):
            return "target"
        return "clear"

    def _ultrasonic_state(
        self,
        raw_distance_m: Optional[float],
        target_distance_m: Optional[float],
        brake_distance_m: Optional[float],
    ) -> DistanceState:
        c = self.config
        if raw_distance_m is None:
            state = DistanceState(
                source="ultrasonic",
                trigger="invalid",
                target_close_count=self._ultrasonic_target_close_count,
                brake_close_count=self._ultrasonic_brake_close_count,
                target_latched=self._ultrasonic_target_latched,
                brake_latched=self._ultrasonic_brake_latched,
                target_threshold_m=target_distance_m,
                brake_threshold_m=brake_distance_m,
                hysteresis_m=float(c.ultrasonic_hysteresis_m),
                sample_count=len(self._ultrasonic_samples),
            )
            self.last_distance_state = state
            return state

        self._ultrasonic_samples.append(float(raw_distance_m))
        filtered = float(median(self._ultrasonic_samples))
        target_m = None if target_distance_m is None else float(target_distance_m)
        brake_m = None if brake_distance_m is None else float(brake_distance_m)
        hysteresis = max(0.0, float(c.ultrasonic_hysteresis_m))
        target_confirm = max(1, int(c.ultrasonic_target_confirm_frames))
        brake_confirm = max(1, int(c.ultrasonic_brake_confirm_frames))
        immediate_brake_m = max(0.0, float(c.ultrasonic_immediate_brake_m))
        immediate_brake = bool(brake_m is not None and immediate_brake_m > 0.0 and filtered <= immediate_brake_m)

        if brake_m is not None:
            if immediate_brake or filtered < brake_m:
                self._ultrasonic_brake_close_count += 1
            elif filtered >= brake_m + hysteresis:
                self._ultrasonic_brake_close_count = 0
                self._ultrasonic_brake_latched = False
            if immediate_brake or self._ultrasonic_brake_close_count >= brake_confirm:
                self._ultrasonic_brake_latched = True

        if target_m is not None:
            if immediate_brake or filtered <= target_m:
                self._ultrasonic_target_close_count += 1
            elif filtered >= target_m + hysteresis:
                self._ultrasonic_target_close_count = 0
                self._ultrasonic_target_latched = False
            if immediate_brake or self._ultrasonic_target_close_count >= target_confirm:
                self._ultrasonic_target_latched = True

        if self._ultrasonic_brake_latched:
            self._ultrasonic_target_latched = True

        trigger = "clear"
        used_distance = filtered
        if self._ultrasonic_brake_latched:
            trigger = "brake_immediate" if immediate_brake else "brake_confirmed"
        elif self._ultrasonic_target_latched:
            trigger = "target_confirmed"
        elif brake_m is not None and filtered < brake_m:
            trigger = "brake_unconfirmed"
            if target_m is not None:
                used_distance = max(filtered, target_m + 0.01)
        elif target_m is not None and filtered <= target_m:
            trigger = "target_unconfirmed"
            used_distance = max(filtered, target_m + 0.01)

        state = DistanceState(
            source="ultrasonic",
            raw_distance_m=float(raw_distance_m),
            filtered_distance_m=filtered,
            used_distance_m=used_distance,
            trigger=trigger,
            target_close_count=self._ultrasonic_target_close_count,
            brake_close_count=self._ultrasonic_brake_close_count,
            target_latched=self._ultrasonic_target_latched,
            brake_latched=self._ultrasonic_brake_latched,
            target_threshold_m=target_m,
            brake_threshold_m=brake_m,
            hysteresis_m=hysteresis,
            sample_count=len(self._ultrasonic_samples),
        )
        self.last_distance_state = state
        return state

    def get_sensor_distance_state(
        self,
        *,
        target_distance_m: Optional[float] = None,
        brake_distance_m: Optional[float] = None,
    ) -> DistanceState:
        source = self._sensor_source()
        if source is None:
            state = DistanceState(source="none")
            self.last_distance_state = state
            return state
        raw_distance_m = self._read_sensor_distance_raw(source)
        if source == "ultrasonic":
            return self._ultrasonic_state(raw_distance_m, target_distance_m, brake_distance_m)

        trigger = self._distance_trigger(raw_distance_m, target_distance_m, brake_distance_m)
        state = DistanceState(
            source=source,
            raw_distance_m=raw_distance_m,
            filtered_distance_m=raw_distance_m,
            used_distance_m=raw_distance_m,
            trigger=trigger,
            target_threshold_m=target_distance_m,
            brake_threshold_m=brake_distance_m,
            sample_count=1 if raw_distance_m is not None else 0,
        )
        self.last_distance_state = state
        return state

    def select_target(self, targets: List[PersonTarget]) -> Optional[PersonTarget]:
        return self.owner._follow_controller.select_target_for_current_state(targets)

    def pixel_x_to_angle_deg(self, x: float, width: int) -> float:
        c = self.config
        if width <= 0:
            return 0.0
        norm = (float(x) / float(width)) - 0.5
        return norm * c.vision_hfov_deg * c.vision_mmwave_angle_sign + c.vision_mmwave_angle_offset_deg

    def compensate_vision_mmwave_distance_m(self, distance_m: float) -> float:
        c = self.config
        return max(c.vision_mmwave_min_output_distance_m, float(distance_m) - c.vision_mmwave_distance_bias_m)

    def _clear_mmwave_association(self) -> None:
        self._mmwave_associated_track_id = None
        self._mmwave_accepted_angle_deg = None
        self._mmwave_accepted_distance_m = None
        self._mmwave_accepted_ts = 0.0
        self._mmwave_pending_candidate = None
        self._mmwave_pending_count = 0
        self._mmwave_angle_history.clear()
        self.last_vision_mmwave_motion_direction = "unknown"

    def _clear_mmwave_pending_candidate(self) -> None:
        self._mmwave_pending_candidate = None
        self._mmwave_pending_count = 0

    def _record_visual_angles(
        self,
        *,
        target: PersonTarget,
        center_angle: float,
        angle_range: Tuple[float, float],
        now: float,
    ) -> None:
        """保存视觉角度历史，供延迟毫米波样本按时间对齐。"""
        track_id = int(target.track_id)
        if self._vision_angle_history_track_id != track_id:
            self._vision_angle_history.clear()
            self._vision_angle_history_track_id = track_id
            if self._mmwave_associated_track_id != track_id:
                accepted_distance = self._mmwave_accepted_distance_m
                accepted_angle = self._mmwave_accepted_angle_deg
                accepted_age = max(0.0, float(now) - float(self._mmwave_accepted_ts))
                near_lock = max(
                    float(self.config.vision_mmwave_unmatched_hold_min_distance_m),
                    float(self.config.vision_mmwave_near_distance_lock_m),
                )
                angle_limit = max(
                    float(self.config.vision_mmwave_max_center_angle_diff_deg),
                    float(self.config.vision_mmwave_max_angle_jump_deg),
                ) + float(self.config.vision_mmwave_angle_margin_deg)
                preserve_near_association = bool(
                    accepted_distance is not None
                    and float(accepted_distance) <= near_lock
                    and accepted_angle is not None
                    and abs(float(accepted_angle) - float(center_angle)) <= angle_limit
                    and accepted_age <= max(
                        0.0,
                        float(self.config.vision_mmwave_continuity_memory_sec),
                    )
                )
                if preserve_near_association:
                    old_associated_track = self._mmwave_associated_track_id
                    self._mmwave_associated_track_id = track_id
                    self._clear_mmwave_pending_candidate()
                    self.logger.info(
                        "毫米波近距离关联跨视觉轨迹保持: old_track=%s new_track=%d distance=%.2fm age=%.3fs",
                        "none" if old_associated_track is None else int(old_associated_track),
                        int(track_id),
                        float(accepted_distance),
                        accepted_age,
                    )
                else:
                    self._clear_mmwave_association()
        self._vision_angle_history.append(
            {
                "ts": float(now),
                "track_id": track_id,
                "center_angle": float(center_angle),
                "angle_lo": float(angle_range[0]),
                "angle_hi": float(angle_range[1]),
            }
        )
        keep_sec = max(
            1.0,
            float(self.config.vision_mmwave_latency_sec)
            + float(self.config.vision_mmwave_cache_max_age_sec)
            + 0.3,
        )
        cutoff = now - keep_sec
        while self._vision_angle_history and float(self._vision_angle_history[0]["ts"]) < cutoff:
            self._vision_angle_history.popleft()

    def _aligned_visual_angles(
        self,
        sample_ts: Optional[float],
        current_center_angle: float,
        current_angle_range: Tuple[float, float],
    ) -> Tuple[float, Tuple[float, float]]:
        if sample_ts is None or not self._vision_angle_history:
            return current_center_angle, current_angle_range
        record = min(
            self._vision_angle_history,
            key=lambda item: abs(float(item["ts"]) - float(sample_ts)),
        )
        return (
            float(record["center_angle"]),
            (float(record["angle_lo"]), float(record["angle_hi"])),
        )

    def _association_is_fresh(self, track_id: int, now: float) -> bool:
        return bool(
            self._mmwave_associated_track_id == int(track_id)
            and self._mmwave_accepted_angle_deg is not None
            and self._mmwave_accepted_distance_m is not None
            and now - self._mmwave_accepted_ts
            <= max(0.0, float(self.config.vision_mmwave_continuity_memory_sec))
        )

    def _select_mmwave_candidate(
        self,
        candidates: List[Dict[str, Any]],
        aligned_center_angle: float,
        *,
        association_fresh: bool,
    ) -> Optional[Dict[str, Any]]:
        c = self.config
        if association_fresh:
            accepted_angle = float(self._mmwave_accepted_angle_deg)
            accepted_distance = float(self._mmwave_accepted_distance_m)
            for item in candidates:
                item["center_angle_delta_deg"] = abs(float(item["angle"]) - aligned_center_angle)
                item["previous_angle_delta_deg"] = abs(float(item["angle"]) - accepted_angle)
                item["previous_distance_delta_m"] = abs(float(item["distance_m"]) - accepted_distance)
                item["continuity_score"] = (
                    item["center_angle_delta_deg"]
                    + item["previous_angle_delta_deg"]
                    + max(0.0, float(c.vision_mmwave_distance_score_weight))
                    * item["previous_distance_delta_m"]
                )

            # 短时丢点后只允许原角度轨迹续接；距离变远仍交给三帧确认。
            angle_continuous = [
                item
                for item in candidates
                if item["previous_angle_delta_deg"]
                <= max(0.0, float(c.vision_mmwave_max_angle_jump_deg))
            ]
            if angle_continuous:
                return min(
                    angle_continuous,
                    key=lambda item: (
                        float(item["continuity_score"]),
                        float(item["distance_m"]),
                    ),
                )

            # 完全不连续的新点不得抢占原目标；但紧急近点仍立即采用以保留防撞优先级。
            emergency_close = [
                item
                for item in candidates
                if float(item["distance_m"])
                <= max(0.0, float(c.vision_mmwave_unmatched_hold_min_distance_m))
            ]
            if emergency_close:
                return min(emergency_close, key=lambda item: float(item["distance_m"]))
            return None

        mode = c.vision_mmwave_match_mode
        if mode in ("first", "id"):
            return min(candidates, key=lambda item: item["index"])
        if mode in ("angle", "center", "angle_closest"):
            tie_margin = max(0.0, float(c.vision_mmwave_angle_tie_margin_deg))
            if tie_margin > 0:
                scored = [(abs(item["angle"] - aligned_center_angle), item) for item in candidates]
                best_delta = min(delta for delta, _item in scored)
                close_angle_candidates = [
                    item for delta, item in scored if delta <= best_delta + tie_margin
                ]
                return min(
                    close_angle_candidates,
                    key=lambda item: (item["distance_m"], abs(item["angle"] - aligned_center_angle)),
                )
            return min(
                candidates,
                key=lambda item: (abs(item["angle"] - aligned_center_angle), item["distance_m"]),
            )
        return min(
            candidates,
            key=lambda item: (item["distance_m"], abs(item["angle"] - aligned_center_angle)),
        )

    def _pending_candidate_matches(self, track_id: int, selected: Dict[str, Any]) -> bool:
        pending = self._mmwave_pending_candidate
        if pending is None or int(pending.get("track_id", -999999)) != int(track_id):
            return False
        distance_tolerance = max(
            0.15,
            min(0.30, float(self.config.vision_mmwave_max_distance_jump_m) * 0.5),
        )
        return bool(
            abs(float(pending["distance_m"]) - float(selected["distance_m"])) <= distance_tolerance
            and abs(float(pending["angle"]) - float(selected["angle"]))
            <= max(1.0, float(self.config.vision_mmwave_max_angle_jump_deg))
        )

    def _record_mmwave_motion(self, track_id: int, angle: float, sample_ts: Optional[float], now: float) -> None:
        record = {
            "track_id": int(track_id),
            "angle": float(angle),
            "sample_ts": float(now if sample_ts is None else sample_ts),
            "accepted_ts": float(now),
        }
        if self._mmwave_angle_history and record["sample_ts"] <= float(self._mmwave_angle_history[-1]["sample_ts"]):
            self._mmwave_angle_history[-1] = record
        else:
            self._mmwave_angle_history.append(record)
        cutoff = now - max(0.1, float(self.config.vision_mmwave_continuity_memory_sec))
        while self._mmwave_angle_history and float(self._mmwave_angle_history[0]["accepted_ts"]) < cutoff:
            self._mmwave_angle_history.popleft()

    def get_recent_vision_mmwave_motion_hint(self) -> Tuple[str, float, Dict[str, Any]]:
        """返回已确认雷达轨迹的短时左右趋势；只供视觉丢失方向兜底。"""
        now = time.monotonic()
        if not self._mmwave_angle_history:
            return "unknown", 999.0, {"samples": 0}
        latest = self._mmwave_angle_history[-1]
        age = max(0.0, now - float(latest["accepted_ts"]))
        if age > max(0.0, float(self.config.vision_mmwave_motion_hint_ttl_sec)):
            return "unknown", age, {"samples": len(self._mmwave_angle_history)}
        recent = [
            item
            for item in self._mmwave_angle_history
            if int(item["track_id"]) == int(latest["track_id"])
        ]
        if len(recent) < 2:
            return "unknown", age, {"samples": len(recent)}
        delta_angle = float(recent[-1]["angle"]) - float(recent[0]["angle"])
        debug = {
            "samples": len(recent),
            "delta_angle_deg": delta_angle,
            "track_id": int(latest["track_id"]),
        }
        if abs(delta_angle) < max(0.0, float(self.config.vision_mmwave_motion_min_angle_delta_deg)):
            return "unknown", age, debug
        # angle_sign 把摄像头像素方向映射到雷达角度；这里做逆映射得到画面左右方向。
        pixel_direction_delta = delta_angle * float(self.config.vision_mmwave_angle_sign)
        direction = "right" if pixel_direction_delta > 0.0 else "left"
        self.last_vision_mmwave_motion_direction = direction
        return direction, age, debug

    def _accept_mmwave_candidate(
        self,
        *,
        track_id: int,
        selected: Dict[str, Any],
        current_center_angle: float,
        aligned_center_angle: float,
        sample_ts: Optional[float],
        sample_age: Optional[float],
        now: float,
        detail: str,
    ) -> float:
        distance_m = float(selected["distance_m"])
        angle = float(selected["angle"])
        self._mmwave_associated_track_id = int(track_id)
        self._mmwave_accepted_angle_deg = angle
        self._mmwave_accepted_distance_m = distance_m
        self._mmwave_accepted_ts = now
        self._mmwave_pending_candidate = None
        self._mmwave_pending_count = 0
        self._record_mmwave_motion(track_id, angle, sample_ts, now)
        self.last_vision_mmwave_distance_m = distance_m
        self.last_vision_mmwave_ts = now
        self.last_vision_mmwave_detail = detail
        self.last_vision_mmwave_sample_age_sec = sample_age
        self.last_vision_mmwave_target_angle_deg = current_center_angle
        self.last_vision_mmwave_aligned_target_angle_deg = aligned_center_angle
        self.last_vision_mmwave_matched_angle_deg = angle
        return distance_m

    def _reset_vision_mmwave_match(self, detail: str) -> None:
        self.last_vision_mmwave_distance_m = None
        self.last_vision_mmwave_ts = 0.0
        self.last_vision_mmwave_detail = detail
        self.last_vision_mmwave_sample_age_sec = None
        self.last_vision_mmwave_target_angle_deg = None
        self.last_vision_mmwave_matched_angle_deg = None
        self.last_vision_mmwave_aligned_target_angle_deg = None

    def _hold_distance_for_temporary_match_loss(
        self,
        *,
        track_id: int,
        now: float,
        detail: str,
        current_center_angle: float,
        aligned_center_angle: float,
    ) -> Optional[float]:
        hold_sec = max(0.0, float(self.config.vision_mmwave_unmatched_hold_sec))
        accepted_distance = self._mmwave_accepted_distance_m
        accepted_angle = self._mmwave_accepted_angle_deg
        age = max(0.0, float(now) - float(self._mmwave_accepted_ts))
        can_hold = bool(
            hold_sec > 0.0
            and self._mmwave_associated_track_id == int(track_id)
            and accepted_distance is not None
            and accepted_angle is not None
            and float(accepted_distance)
            > max(0.0, float(self.config.vision_mmwave_unmatched_hold_min_distance_m))
            and age <= hold_sec
        )
        if not can_hold:
            return None

        # 保持的是上次已确认距离，不刷新 accepted_ts，避免丢点时间被无限延长。
        self.last_vision_mmwave_distance_m = float(accepted_distance)
        self.last_vision_mmwave_ts = float(self._mmwave_accepted_ts)
        self.last_vision_mmwave_detail = "%s_hold" % detail
        self.last_vision_mmwave_sample_age_sec = age
        self.last_vision_mmwave_target_angle_deg = float(current_center_angle)
        self.last_vision_mmwave_aligned_target_angle_deg = float(aligned_center_angle)
        self.last_vision_mmwave_matched_angle_deg = float(accepted_angle)
        return float(accepted_distance)

    def get_recent_vision_mmwave_distance(self) -> Optional[float]:
        c = self.config
        distance_m = self.last_vision_mmwave_distance_m
        if distance_m is None:
            return None
        if time.monotonic() - self.last_vision_mmwave_ts > c.vision_mmwave_hard_stop_ttl_sec:
            return None
        return distance_m

    def get_recent_vision_mmwave_state(self, *, target_distance_m: Optional[float] = None, brake_distance_m: Optional[float] = None) -> DistanceState:
        distance_m = self.get_recent_vision_mmwave_distance()
        trigger = self._distance_trigger(distance_m, target_distance_m, brake_distance_m)
        return DistanceState(
            source="vision_mmwave",
            raw_distance_m=distance_m,
            filtered_distance_m=distance_m,
            used_distance_m=distance_m,
            trigger=trigger,
            source_detail=self.last_vision_mmwave_detail,
            sample_age_sec=self.last_vision_mmwave_sample_age_sec,
            target_angle_deg=self.last_vision_mmwave_target_angle_deg,
            matched_angle_deg=self.last_vision_mmwave_matched_angle_deg,
            target_threshold_m=target_distance_m,
            brake_threshold_m=brake_distance_m,
            sample_count=1 if distance_m is not None else 0,
        )

    def _read_vision_mmwave_targets(self) -> Tuple[List[Dict[str, Any]], Optional[float], Optional[float]]:
        c = self.config
        now = time.monotonic()
        sample_ts: Optional[float] = None
        target_ts = now - max(0.0, float(c.vision_mmwave_latency_sec))
        try:
            if c.vision_mmwave_use_async_cache and hasattr(self.sensor_runtime, "get_mmwave_targets_at"):
                raw_targets, sample_ts = self.sensor_runtime.get_mmwave_targets_at(
                    target_ts,
                    max_age_sec=max(float(c.vision_mmwave_cache_max_age_sec), float(c.vision_mmwave_latency_sec)),
                )
            else:
                raw_targets = self.sensor_runtime.get_mmwave_targets()
                sample_ts = time.monotonic()
        except Exception as exc:
            self.logger.warning("视觉与毫米波融合读取目标失败: %s", exc)
            return [], None, None
        sample_age = None if sample_ts is None else max(0.0, now - float(sample_ts))
        return list(raw_targets), sample_ts, sample_age

    def log_vision_mmwave_match(
        self,
        reason: str,
        target: PersonTarget,
        current_center_angle: float,
        aligned_center_angle: float,
        angle_range: Tuple[float, float],
        raw_targets: List[Dict[str, Any]],
        selected: Optional[Dict[str, Any]],
        sample_age_sec: Optional[float],
    ) -> None:
        c = self.config
        selected_idx = None if selected is None else selected.get("index")
        selected_dist = None if selected is None else selected.get("distance_m")
        log_key = (
            reason,
            selected_idx,
            None if selected_dist is None else round(float(selected_dist), 2),
            len(raw_targets),
            None if sample_age_sec is None else round(float(sample_age_sec), 1),
        )
        should_log = log_key != self.last_vision_mmwave_log_key
        if c.vision_mmwave_log_every_frames > 0 and self.owner.frame_index % c.vision_mmwave_log_every_frames == 0:
            should_log = True
        if not should_log:
            return

        cx, cy = target.center
        raw_dbg = []
        for item in raw_targets:
            try:
                raw_dbg.append(
                    "%s:%.1f度/%.2f米"
                    % (
                        item.get("index"),
                        float(item.get("angle")),
                        float(item.get("distance")),
                    )
                )
            except Exception:
                raw_dbg.append(str(item))
        selected_dbg = "none"
        if selected is not None:
            selected_dbg = "%s:%.1f度 原始距离=%.2f米 使用距离=%.2f米 连续性分数=%s" % (
                selected.get("index"),
                float(selected.get("angle")),
                float(selected.get("raw_distance_m")),
                float(selected.get("distance_m")),
                "none"
                if selected.get("continuity_score") is None
                else "%.2f" % float(selected.get("continuity_score")),
            )
        self.logger.info(
            "视觉与毫米波匹配: 帧=%d 原因代码=%s 人员中心=(%.1f,%.1f) 当前视觉角度=%.1f度 对齐视觉角度=%.1f度 匹配角度范围=[%.1f,%.1f] 样本延迟=%s 延迟补偿=%.3f秒 雷达目标=%s 已选目标=%s",
            self.owner.frame_index,
            reason,
            float(cx),
            float(cy),
            float(current_center_angle),
            float(aligned_center_angle),
            float(angle_range[0]),
            float(angle_range[1]),
            "none" if sample_age_sec is None else "%.0f毫秒" % (float(sample_age_sec) * 1000.0),
            float(c.vision_mmwave_latency_sec),
            raw_dbg,
            selected_dbg,
        )
        self.last_vision_mmwave_log_key = log_key

    def get_vision_mmwave_distance(self, width: int, target: Optional[PersonTarget]) -> Optional[float]:
        c = self.config
        if target is None:
            # 远点换绑必须是连续视觉帧；视觉空帧不能累计确认次数。
            self._clear_mmwave_pending_candidate()
            return None
        if not c.module_mmwave_enable or width <= 0:
            self._reset_vision_mmwave_match("disabled")
            return None

        now = time.monotonic()
        track_id = int(target.track_id)
        x1, _y1, x2, _y2 = target.bbox
        center_x, _center_y = target.center
        left_angle = self.pixel_x_to_angle_deg(max(0.0, min(float(width), float(x1))), width)
        right_angle = self.pixel_x_to_angle_deg(max(0.0, min(float(width), float(x2))), width)
        center_angle = self.pixel_x_to_angle_deg(center_x, width)
        base_angle_lo, base_angle_hi = sorted((left_angle, right_angle))
        angle_margin_deg = max(0.0, float(c.vision_mmwave_angle_margin_deg))
        if (
            self._association_is_fresh(track_id, now)
            and self._mmwave_accepted_distance_m is not None
            and float(self._mmwave_accepted_distance_m)
            >= max(0.0, float(c.vision_mmwave_far_margin_start_m))
        ):
            # 远距离人体框的角度宽度更小，只对同一已确认雷达轨迹扩大少量余量。
            angle_margin_deg += max(0.0, float(c.vision_mmwave_far_angle_margin_extra_deg))

        self._record_visual_angles(
            target=target,
            center_angle=center_angle,
            # 历史中只保存原始人体框；毫米波时间对齐后再统一叠加当前安全余量。
            angle_range=(base_angle_lo, base_angle_hi),
            now=now,
        )

        raw_targets, sample_ts, sample_age = self._read_vision_mmwave_targets()
        aligned_center_angle, aligned_angle_range = self._aligned_visual_angles(
            sample_ts,
            center_angle,
            (base_angle_lo, base_angle_hi),
        )
        aligned_angle_lo = float(aligned_angle_range[0]) - angle_margin_deg
        aligned_angle_hi = float(aligned_angle_range[1]) + angle_margin_deg
        aligned_angle_range = (aligned_angle_lo, aligned_angle_hi)
        self.last_vision_mmwave_target_angle_deg = center_angle
        self.last_vision_mmwave_aligned_target_angle_deg = aligned_center_angle
        if not raw_targets:
            selected = None
            self._clear_mmwave_pending_candidate()
            reason = "no_radar_targets"
            distance_m = self._hold_distance_for_temporary_match_loss(
                track_id=track_id,
                now=now,
                detail=reason,
                current_center_angle=center_angle,
                aligned_center_angle=aligned_center_angle,
            )
            if distance_m is None:
                self._reset_vision_mmwave_match(reason)
                self.last_vision_mmwave_target_angle_deg = center_angle
                self.last_vision_mmwave_aligned_target_angle_deg = aligned_center_angle
                self.last_vision_mmwave_sample_age_sec = sample_age
            self.log_vision_mmwave_match(
                self.last_vision_mmwave_detail, target, center_angle, aligned_center_angle,
                aligned_angle_range, [], selected, sample_age,
            )
            return distance_m

        candidates: List[Dict[str, Any]] = []
        center_angle_rejected = False
        for item in raw_targets:
            try:
                angle = float(item.get("angle"))
                distance_m = float(item.get("distance"))
            except Exception:
                continue
            if not (math.isfinite(angle) and math.isfinite(distance_m)):
                continue
            if not (c.vision_mmwave_min_distance_m < distance_m < c.vision_mmwave_max_distance_m):
                continue
            if not (aligned_angle_lo <= angle <= aligned_angle_hi):
                continue
            center_delta = abs(angle - aligned_center_angle)
            if center_delta > max(0.0, float(c.vision_mmwave_max_center_angle_diff_deg)):
                center_angle_rejected = True
                continue
            candidates.append(
                {
                    "index": int(item.get("index", -1)),
                    "angle": angle,
                    "raw_distance_m": distance_m,
                    "distance_m": self.compensate_vision_mmwave_distance_m(distance_m),
                    "center_angle_delta_deg": center_delta,
                    "target": item,
                }
            )

        selected: Optional[Dict[str, Any]]
        if not candidates:
            selected = None
            self._clear_mmwave_pending_candidate()
            reason = "center_angle_rejected" if center_angle_rejected else "unmatched"
            distance_m = self._hold_distance_for_temporary_match_loss(
                track_id=track_id,
                now=now,
                detail=reason,
                current_center_angle=center_angle,
                aligned_center_angle=aligned_center_angle,
            )
            if distance_m is None:
                self._reset_vision_mmwave_match(reason)
                self.last_vision_mmwave_target_angle_deg = center_angle
                self.last_vision_mmwave_aligned_target_angle_deg = aligned_center_angle
                self.last_vision_mmwave_sample_age_sec = sample_age
            self.log_vision_mmwave_match(
                self.last_vision_mmwave_detail, target, center_angle, aligned_center_angle,
                aligned_angle_range, raw_targets, selected, sample_age,
            )
            return distance_m

        association_fresh = self._association_is_fresh(track_id, now)
        if not association_fresh and self._mmwave_associated_track_id == track_id:
            self._clear_mmwave_association()
        selected = self._select_mmwave_candidate(
            candidates,
            aligned_center_angle,
            association_fresh=association_fresh,
        )
        if selected is None:
            self._clear_mmwave_pending_candidate()
            reason = "continuity_rejected"
            distance_m = self._hold_distance_for_temporary_match_loss(
                track_id=track_id,
                now=now,
                detail=reason,
                current_center_angle=center_angle,
                aligned_center_angle=aligned_center_angle,
            )
            if distance_m is None:
                self._reset_vision_mmwave_match(reason)
                self.last_vision_mmwave_target_angle_deg = center_angle
                self.last_vision_mmwave_aligned_target_angle_deg = aligned_center_angle
                self.last_vision_mmwave_sample_age_sec = sample_age
            self.log_vision_mmwave_match(
                self.last_vision_mmwave_detail, target, center_angle, aligned_center_angle,
                aligned_angle_range, raw_targets, selected, sample_age,
            )
            return distance_m

        reason = "matched"
        if association_fresh:
            farther_jump_m = float(selected["distance_m"]) - float(self._mmwave_accepted_distance_m)
            if farther_jump_m > max(0.0, float(c.vision_mmwave_max_distance_jump_m)):
                if self._pending_candidate_matches(track_id, selected):
                    self._mmwave_pending_count += 1
                else:
                    self._mmwave_pending_candidate = {
                        "track_id": track_id,
                        "angle": float(selected["angle"]),
                        "distance_m": float(selected["distance_m"]),
                    }
                    self._mmwave_pending_count = 1
                confirm_frames = max(1, int(c.vision_mmwave_switch_confirm_frames))
                if self._mmwave_pending_count < confirm_frames:
                    reason = "distance_jump_pending_%d_of_%d" % (
                        self._mmwave_pending_count,
                        confirm_frames,
                    )
                    distance_m = self._hold_distance_for_temporary_match_loss(
                        track_id=track_id,
                        now=now,
                        detail=reason,
                        current_center_angle=center_angle,
                        aligned_center_angle=aligned_center_angle,
                    )
                    if distance_m is None:
                        self._reset_vision_mmwave_match(reason)
                        self.last_vision_mmwave_target_angle_deg = center_angle
                        self.last_vision_mmwave_aligned_target_angle_deg = aligned_center_angle
                        self.last_vision_mmwave_sample_age_sec = sample_age
                    self.log_vision_mmwave_match(
                        self.last_vision_mmwave_detail, target, center_angle, aligned_center_angle,
                        aligned_angle_range, raw_targets, selected, sample_age,
                    )
                    return distance_m
                reason = "matched_after_continuity_confirm"

        distance_m = self._accept_mmwave_candidate(
            track_id=track_id,
            selected=selected,
            current_center_angle=center_angle,
            aligned_center_angle=aligned_center_angle,
            sample_ts=sample_ts,
            sample_age=sample_age,
            now=now,
            detail=reason,
        )
        self.log_vision_mmwave_match(
            reason, target, center_angle, aligned_center_angle,
            aligned_angle_range, raw_targets, selected, sample_age,
        )
        return distance_m

    def get_frame_distance_state(
        self,
        width: int,
        target: Optional[PersonTarget],
        *,
        frame_height: int = 0,
        steering_feedback: Optional[SteeringFeedback] = None,
        target_distance_m: Optional[float] = None,
        brake_distance_m: Optional[float] = None,
        depth_use_latest: bool = False,
    ) -> DistanceState:
        if self.config.distance_source in self.config.vision_depth_source_aliases:
            return self.get_vision_depth_state(
                width,
                frame_height,
                target,
                steering_feedback=steering_feedback,
                target_distance_m=target_distance_m,
                brake_distance_m=brake_distance_m,
                use_latest_depth=bool(depth_use_latest),
            )
        if self.config.distance_source in self.config.vision_mmwave_source_aliases:
            radar_distance_m = self.get_vision_mmwave_distance(width, target)
            is_held_distance = str(self.last_vision_mmwave_detail).endswith("_hold")
            fusion = self._distance_fusion.update(
                target=target,
                frame_height=int(frame_height),
                radar_distance_m=radar_distance_m,
                radar_fresh=bool(radar_distance_m is not None and not is_held_distance),
                sample_age_sec=self.last_vision_mmwave_sample_age_sec,
                steering_feedback=steering_feedback,
                now=time.monotonic(),
            )
            distance_m = fusion.distance_m
            trigger = self._distance_trigger(distance_m, target_distance_m, brake_distance_m)
            state = DistanceState(
                source="vision_mmwave",
                raw_distance_m=None if is_held_distance else radar_distance_m,
                filtered_distance_m=distance_m,
                used_distance_m=distance_m,
                trigger=trigger,
                source_detail=self.last_vision_mmwave_detail,
                sample_age_sec=self.last_vision_mmwave_sample_age_sec,
                target_angle_deg=self.last_vision_mmwave_target_angle_deg,
                matched_angle_deg=self.last_vision_mmwave_matched_angle_deg,
                target_threshold_m=target_distance_m,
                brake_threshold_m=brake_distance_m,
                sample_count=0 if is_held_distance or distance_m is None else 1,
                fusion_mode=fusion.mode,
                fusion_confidence=fusion.confidence,
                fusion_radar_distance_m=fusion.radar_distance_m,
                fusion_visual_distance_m=fusion.visual_distance_m,
                fusion_encoder_delta_m=fusion.encoder_delta_m,
            )
            self.last_distance_state = state
            return state
        return self.get_sensor_distance_state(
            target_distance_m=target_distance_m,
            brake_distance_m=brake_distance_m,
        )

    def _build_vision_depth_state(
        self,
        measurement,
        *,
        target: PersonTarget,
        frame_height: int,
        steering_feedback: Optional[SteeringFeedback],
        target_distance_m: Optional[float],
        brake_distance_m: Optional[float],
    ) -> DistanceState:
        now = time.monotonic()
        detail = str(measurement.detail)
        measurement_distance = measurement.distance_m
        fresh_depth = bool(
            measurement_distance is not None
            and measurement.raw_distance_m is not None
            and not detail.endswith("_hold")
            and detail.startswith("depth_")
        )
        fusion = self._vision_depth_fusion.update(
            target=target,
            frame_height=int(frame_height),
            radar_distance_m=measurement_distance,
            radar_fresh=fresh_depth,
            sample_age_sec=measurement.sample_age_sec,
            steering_feedback=steering_feedback,
            now=now,
        )
        distance_m = fusion.distance_m
        source_detail = detail
        sample_age_sec = measurement.sample_age_sec
        raw_distance_m = measurement.raw_distance_m if fresh_depth else None
        if not fresh_depth and distance_m is not None:
            source_detail = f"{detail}_fused_{fusion.mode}_hold"
            sample_age_sec = fusion.anchor_age_sec
        elif fresh_depth and fusion.mode == "radar_jump_pending":
            source_detail = (
                f"{detail}_distance_jump_pending_"
                f"{self._vision_depth_fusion._pending_far_count}_of_"
                f"{max(2, int(self.config.vision_depth_jump_confirm_frames))}"
            )
        state = DistanceState(
            source="vision_depth",
            raw_distance_m=raw_distance_m,
            filtered_distance_m=distance_m,
            used_distance_m=distance_m,
            trigger=self._distance_trigger(distance_m, target_distance_m, brake_distance_m),
            source_detail=source_detail,
            sample_age_sec=sample_age_sec,
            target_threshold_m=target_distance_m,
            brake_threshold_m=brake_distance_m,
            sample_count=int(measurement.valid_pixels) if fresh_depth else 0,
            fusion_mode=f"depth_{fusion.mode}",
            fusion_confidence=float(fusion.confidence),
            fusion_radar_distance_m=fusion.radar_distance_m,
            fusion_visual_distance_m=fusion.visual_distance_m,
            fusion_encoder_delta_m=fusion.encoder_delta_m,
        )
        self.last_distance_state = state
        return state

    def get_vision_depth_state(
        self,
        width: int,
        height: int,
        target: Optional[PersonTarget],
        *,
        steering_feedback: Optional[SteeringFeedback] = None,
        target_distance_m: Optional[float] = None,
        brake_distance_m: Optional[float] = None,
        use_latest_depth: bool = False,
    ) -> DistanceState:
        if not self.config.module_astra_depth_enable or target is None or width <= 0 or height <= 0:
            if target is None:
                self._last_vision_depth_target = None
                self._last_vision_depth_frame_size = (0, 0)
                self._vision_depth_fusion.reset()
            state = DistanceState(
                source="vision_depth",
                source_detail="disabled" if not self.config.module_astra_depth_enable else "no_visual_target",
                target_threshold_m=target_distance_m,
                brake_threshold_m=brake_distance_m,
            )
            self.last_distance_state = state
            return state

        self._last_vision_depth_target = target
        self._last_vision_depth_frame_size = (int(width), int(height))
        measurement_kwargs = {"target_id": int(target.track_id)}
        if use_latest_depth:
            measurement_kwargs["use_latest_depth"] = True
        if steering_feedback is not None:
            measurement_kwargs["steering_feedback"] = steering_feedback
        measurement = self.sensor_runtime.get_astra_target_distance(
            target.bbox,
            int(width),
            int(height),
            **measurement_kwargs,
        )
        return self._build_vision_depth_state(
            measurement,
            target=target,
            frame_height=int(height),
            steering_feedback=steering_feedback,
            target_distance_m=target_distance_m,
            brake_distance_m=brake_distance_m,
        )

    def get_recent_vision_depth_state(
        self,
        *,
        target_distance_m: Optional[float] = None,
        brake_distance_m: Optional[float] = None,
    ) -> DistanceState:
        target = self._last_vision_depth_target
        state = self.last_distance_state
        if target is None or str(getattr(state, "source", "")) != "vision_depth":
            return DistanceState(
                source="vision_depth",
                source_detail="no_visual_target",
                target_threshold_m=target_distance_m,
                brake_threshold_m=brake_distance_m,
            )
        # 动作线程在下一帧视觉推理期间会高频调用本方法做硬停复查。此时只有
        # 上一帧人物框，若用它读取最新 Depth，会把背景距离写进滤波器并抢先
        # 消耗新深度帧。这里只读主视觉循环已经配准好的缓存；红外仍逐动作检查。
        return state

    def get_frame_distance(
        self,
        width: int,
        target: Optional[PersonTarget],
        *,
        target_distance_m: Optional[float] = None,
        brake_distance_m: Optional[float] = None,
    ) -> Optional[float]:
        return self.get_frame_distance_state(
            width,
            target,
            target_distance_m=target_distance_m,
            brake_distance_m=brake_distance_m,
        ).used_distance_m
