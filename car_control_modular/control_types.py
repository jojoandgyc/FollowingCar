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
class LateralCandidateEvidence:
    """Fresh YOLO geometry that may steer, but never claims target identity."""

    capture_frame_id: int
    bbox: BBox
    score: float
    source: str = "none"
    active_target_match: bool = False

    @property
    def center_x(self) -> float:
        return (float(self.bbox[0]) + float(self.bbox[2])) / 2.0


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
    fusion_mode: str = ""
    fusion_confidence: float = 0.0
    fusion_radar_distance_m: Optional[float] = None
    fusion_visual_distance_m: Optional[float] = None
    fusion_encoder_delta_m: float = 0.0


@dataclass(frozen=True)
class SteeringFeedback:
    timestamp: float
    left_position_deg: int = 0
    right_position_deg: int = 0
    left_speed_rpm: int = 0
    right_speed_rpm: int = 0
    left_forward_rpm: float = 0.0
    right_forward_rpm: float = 0.0
    yaw_rate_right_dps: float = 0.0
    # Unfiltered single encoder sample. Normal PID feedback keeps using the
    # median-filtered value above; startup/brake pulse gates may use this value
    # to react to the first real wheel response without waiting for 3 samples.
    raw_yaw_rate_right_dps: Optional[float] = None
    integrated_yaw_right_deg: float = 0.0
    left_error: int = 0
    right_error: int = 0
    trustworthy: bool = False


@dataclass(frozen=True)
class SearchControlStatus:
    state: str = "none"
    direction: Optional[str] = None
    active_target_id: Optional[int] = None
    selected_target_id: Optional[int] = None
    progress_deg: float = 0.0
    target_deg: float = 360.0
    elapsed_sec: Optional[float] = None
    stage: str = "inactive"
    heading_from_loss_deg: float = 0.0
    coverage_deg: float = 0.0
    travel_deg: float = 0.0
    hint_confidence: float = 0.0
    hint_source: str = "none"


@dataclass(frozen=True)
class SensorFrame:
    width: int = 0
    height: int = 0
    persons: List[PersonTarget] = field(default_factory=list)
    hazard: HazardState = field(default_factory=HazardState)
    obstacles: ObstacleState = field(default_factory=ObstacleState)
    distance_m: Optional[float] = None
    distance_state: DistanceState = field(default_factory=DistanceState)
    steering_feedback: Optional[SteeringFeedback] = None
    module_status: Dict[str, bool] = field(default_factory=dict)
    capture_frame_id: int = 0
    capture_timestamp: float = 0.0
    lateral_candidate: Optional[LateralCandidateEvidence] = None


@dataclass(frozen=True)
class ControlAction:
    kind: str
    speed_percent: int = 0
    steer_inner_ratio_percent: int = 100
    steer_outer_ratio_percent: int = 100
    steer_correction_rpm: int = 0
    reason: str = ""
    brake_hold: bool = False

    @staticmethod
    def stop(reason: str, brake_hold: bool = True) -> "ControlAction":
        return ControlAction(kind="stop", speed_percent=0, reason=reason, brake_hold=brake_hold)

    @staticmethod
    def forward(speed_percent: int, reason: str) -> "ControlAction":
        return ControlAction(kind="forward", speed_percent=int(speed_percent), reason=reason)

    @staticmethod
    def backward(
        speed_percent: int,
        reason: str,
        correction_rpm: int = 0,
    ) -> "ControlAction":
        return ControlAction(
            kind="backward",
            speed_percent=int(speed_percent),
            steer_correction_rpm=int(correction_rpm),
            reason=reason,
        )

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
        correction_rpm: int = 0,
    ) -> "ControlAction":
        return ControlAction(
            kind="steer_left",
            speed_percent=int(base_speed_percent),
            steer_inner_ratio_percent=int(inner_ratio_percent),
            steer_outer_ratio_percent=int(outer_ratio_percent),
            steer_correction_rpm=max(0, int(correction_rpm)),
            reason=reason,
        )

    @staticmethod
    def steer_right(
        base_speed_percent: int,
        inner_ratio_percent: int,
        outer_ratio_percent: int,
        reason: str,
        correction_rpm: int = 0,
    ) -> "ControlAction":
        return ControlAction(
            kind="steer_right",
            speed_percent=int(base_speed_percent),
            steer_inner_ratio_percent=int(inner_ratio_percent),
            steer_outer_ratio_percent=int(outer_ratio_percent),
            steer_correction_rpm=max(0, int(correction_rpm)),
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
    # A controller-owned zero-yaw update.  Unlike an explicit safety stop,
    # this clears the motor target without entering the action runtime's
    # brake-hold state (used when the visual PID is already settled).
    soft_stop_requested: bool = False
    shutdown_requested: bool = False
    reason: str = ""
    # Capture slot whose geometry/history evidence caused this decision. The
    # runtime uses it for command provenance; normal live decisions leave it
    # unset and therefore use the current capture slot.
    evidence_capture_frame_id: Optional[int] = None
