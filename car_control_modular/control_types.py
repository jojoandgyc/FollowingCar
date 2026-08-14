#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared data structures for modular vehicle control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class PersonTarget:
    bbox: BBox
    track_id: int
    confidence: float
    area: float

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass(frozen=True)
class HazardState:
    active: bool = False
    reason: str = ""
    class_id: int = -1
    score: float = 0.0
    area_ratio: float = 0.0


@dataclass(frozen=True)
class ObstacleState:
    front: bool = False
    left: bool = False
    right: bool = False


@dataclass(frozen=True)
class DistanceState:
    source: str = "none"
    raw_distance_m: Optional[float] = None
    filtered_distance_m: Optional[float] = None
    used_distance_m: Optional[float] = None
    trigger: str = "none"
    source_detail: str = ""
    sample_age_sec: Optional[float] = None
    target_angle_deg: Optional[float] = None
    matched_angle_deg: Optional[float] = None
    target_close_count: int = 0
    brake_close_count: int = 0
    target_latched: bool = False
    brake_latched: bool = False
    target_threshold_m: Optional[float] = None
    brake_threshold_m: Optional[float] = None
    hysteresis_m: float = 0.0
    sample_count: int = 0


@dataclass(frozen=True)
class SensorFrame:
    width: int = 0
    height: int = 0
    persons: List[PersonTarget] = field(default_factory=list)
    hazard: HazardState = field(default_factory=HazardState)
    obstacles: ObstacleState = field(default_factory=ObstacleState)
    distance_m: Optional[float] = None
    distance_state: DistanceState = field(default_factory=DistanceState)
    lost_intent: str = "unknown"
    lost_intent_age_sec: float = 0.0
    module_status: Dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlAction:
    kind: str
    speed_percent: int = 0
    steer_inner_ratio_percent: int = 100
    steer_outer_ratio_percent: int = 100
    reason: str = ""
    brake_hold: bool = False

    @staticmethod
    def stop(reason: str, brake_hold: bool = True) -> "ControlAction":
        return ControlAction(kind="stop", speed_percent=0, reason=reason, brake_hold=brake_hold)

    @staticmethod
    def forward(speed_percent: int, reason: str) -> "ControlAction":
        return ControlAction(kind="forward", speed_percent=int(speed_percent), reason=reason)

    @staticmethod
    def rotate_left(reason: str) -> "ControlAction":
        return ControlAction(kind="rotate_left", reason=reason)

    @staticmethod
    def rotate_right(reason: str) -> "ControlAction":
        return ControlAction(kind="rotate_right", reason=reason)

    @staticmethod
    def steer_left(
        base_speed_percent: int,
        inner_ratio_percent: int,
        outer_ratio_percent: int,
        reason: str,
    ) -> "ControlAction":
        return ControlAction(
            kind="steer_left",
            speed_percent=int(base_speed_percent),
            steer_inner_ratio_percent=int(inner_ratio_percent),
            steer_outer_ratio_percent=int(outer_ratio_percent),
            reason=reason,
        )

    @staticmethod
    def steer_right(
        base_speed_percent: int,
        inner_ratio_percent: int,
        outer_ratio_percent: int,
        reason: str,
    ) -> "ControlAction":
        return ControlAction(
            kind="steer_right",
            speed_percent=int(base_speed_percent),
            steer_inner_ratio_percent=int(inner_ratio_percent),
            steer_outer_ratio_percent=int(outer_ratio_percent),
            reason=reason,
        )

    @staticmethod
    def idle(reason: str = "idle") -> "ControlAction":
        return ControlAction(kind="idle", reason=reason)


@dataclass(frozen=True)
class ControlDecision:
    actions: List[ControlAction] = field(default_factory=list)
    explicit_stop_requested: bool = False
    person_detected_flag: bool = False
    waiting_lost_confirm: bool = False
    is_forwarding: bool = False
    current_forward_percent: int = 0
    clear_action_queue: bool = False
    stop_action_execution: bool = False
    reason: str = ""
