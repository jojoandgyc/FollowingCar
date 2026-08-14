from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from .control_types import DistanceState, PersonTarget


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
    vision_mmwave_latency_sec: float = 0.10
    vision_mmwave_cache_max_age_sec: float = 0.50
    vision_mmwave_use_async_cache: bool = True
    vision_mmwave_angle_tie_margin_deg: float = 0.0
    ultrasonic_min_distance_m: float = 0.02
    ultrasonic_max_distance_m: float = 8.0
    ultrasonic_filter_window: int = 3
    ultrasonic_target_confirm_frames: int = 2
    ultrasonic_brake_confirm_frames: int = 2
    ultrasonic_hysteresis_m: float = 0.25
    ultrasonic_immediate_brake_m: float = 0.35


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
        sample_window = max(1, int(config.ultrasonic_filter_window))
        self._ultrasonic_samples = deque(maxlen=sample_window)
        self._ultrasonic_target_close_count = 0
        self._ultrasonic_brake_close_count = 0
        self._ultrasonic_target_latched = False
        self._ultrasonic_brake_latched = False
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

    def _reset_vision_mmwave_match(self, detail: str) -> None:
        self.last_vision_mmwave_distance_m = None
        self.last_vision_mmwave_ts = 0.0
        self.last_vision_mmwave_detail = detail
        self.last_vision_mmwave_sample_age_sec = None
        self.last_vision_mmwave_target_angle_deg = None
        self.last_vision_mmwave_matched_angle_deg = None

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
            self.logger.warning("vision_mmwave read targets failed: %s", exc)
            return [], None, None
        sample_age = None if sample_ts is None else max(0.0, now - float(sample_ts))
        return list(raw_targets), sample_ts, sample_age

    def log_vision_mmwave_match(
        self,
        reason: str,
        target: PersonTarget,
        center_angle: float,
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
                    "%s:%.1fdeg/%.2fm"
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
            selected_dbg = "%s:%.1fdeg raw=%.2fm used=%.2fm" % (
                selected.get("index"),
                float(selected.get("angle")),
                float(selected.get("raw_distance_m")),
                float(selected.get("distance_m")),
            )
        self.logger.info(
            "vision_mmwave: frame=%d reason=%s person_center=(%.1f,%.1f) angle=%.1f range=[%.1f,%.1f] sample_age=%s latency=%.3fs targets=%s selected=%s",
            self.owner.frame_index,
            reason,
            float(cx),
            float(cy),
            float(center_angle),
            float(angle_range[0]),
            float(angle_range[1]),
            "none" if sample_age_sec is None else "%.0fms" % (float(sample_age_sec) * 1000.0),
            float(c.vision_mmwave_latency_sec),
            raw_dbg,
            selected_dbg,
        )
        self.last_vision_mmwave_log_key = log_key

    def get_vision_mmwave_distance(self, width: int, target: Optional[PersonTarget]) -> Optional[float]:
        c = self.config
        if target is None:
            return None
        if not c.module_mmwave_enable or width <= 0:
            self._reset_vision_mmwave_match("disabled")
            return None

        x1, _y1, x2, _y2 = target.bbox
        center_x, _center_y = target.center
        left_angle = self.pixel_x_to_angle_deg(max(0.0, min(float(width), float(x1))), width)
        right_angle = self.pixel_x_to_angle_deg(max(0.0, min(float(width), float(x2))), width)
        center_angle = self.pixel_x_to_angle_deg(center_x, width)
        angle_lo, angle_hi = sorted((left_angle, right_angle))
        angle_lo -= c.vision_mmwave_angle_margin_deg
        angle_hi += c.vision_mmwave_angle_margin_deg

        raw_targets, _sample_ts, sample_age = self._read_vision_mmwave_targets()
        if not raw_targets:
            selected = None
            self._reset_vision_mmwave_match("no_radar_targets")
            self.last_vision_mmwave_target_angle_deg = center_angle
            self.last_vision_mmwave_sample_age_sec = sample_age
            self.log_vision_mmwave_match("no_radar_targets", target, center_angle, (angle_lo, angle_hi), [], selected, sample_age)
            return None

        candidates: List[Dict[str, Any]] = []
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
            if not (angle_lo <= angle <= angle_hi):
                continue
            candidates.append(
                {
                    "index": int(item.get("index", -1)),
                    "angle": angle,
                    "raw_distance_m": distance_m,
                    "distance_m": self.compensate_vision_mmwave_distance_m(distance_m),
                    "target": item,
                }
            )

        selected: Optional[Dict[str, Any]]
        if not candidates:
            selected = None
            self._reset_vision_mmwave_match("unmatched")
            self.last_vision_mmwave_target_angle_deg = center_angle
            self.last_vision_mmwave_sample_age_sec = sample_age
            self.log_vision_mmwave_match("unmatched", target, center_angle, (angle_lo, angle_hi), raw_targets, selected, sample_age)
            return None

        mode = c.vision_mmwave_match_mode
        if mode in ("first", "id"):
            selected = min(candidates, key=lambda item: item["index"])
        elif mode in ("angle", "center", "angle_closest"):
            tie_margin = max(0.0, float(c.vision_mmwave_angle_tie_margin_deg))
            if tie_margin > 0:
                scored = [(abs(item["angle"] - center_angle), item) for item in candidates]
                best_delta = min(delta for delta, _item in scored)
                close_angle_candidates = [
                    item for delta, item in scored if delta <= best_delta + tie_margin
                ]
                selected = min(
                    close_angle_candidates,
                    key=lambda item: (item["distance_m"], abs(item["angle"] - center_angle)),
                )
            else:
                selected = min(candidates, key=lambda item: (abs(item["angle"] - center_angle), item["distance_m"]))
        else:
            selected = min(candidates, key=lambda item: (item["distance_m"], abs(item["angle"] - center_angle)))

        distance_m = float(selected["distance_m"])
        self.last_vision_mmwave_distance_m = distance_m
        self.last_vision_mmwave_ts = time.monotonic()
        self.last_vision_mmwave_detail = "matched"
        self.last_vision_mmwave_sample_age_sec = sample_age
        self.last_vision_mmwave_target_angle_deg = center_angle
        self.last_vision_mmwave_matched_angle_deg = float(selected["angle"])
        self.log_vision_mmwave_match("matched", target, center_angle, (angle_lo, angle_hi), raw_targets, selected, sample_age)
        return distance_m

    def get_frame_distance_state(
        self,
        width: int,
        target: Optional[PersonTarget],
        *,
        target_distance_m: Optional[float] = None,
        brake_distance_m: Optional[float] = None,
    ) -> DistanceState:
        if self.config.distance_source in self.config.vision_mmwave_source_aliases:
            distance_m = self.get_vision_mmwave_distance(width, target)
            trigger = self._distance_trigger(distance_m, target_distance_m, brake_distance_m)
            state = DistanceState(
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
            self.last_distance_state = state
            return state
        return self.get_sensor_distance_state(
            target_distance_m=target_distance_m,
            brake_distance_m=brake_distance_m,
        )

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
