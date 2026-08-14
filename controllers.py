#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Follow-car control policies.

This module is intentionally independent from board HALs and motor I/O.  It
accepts a SensorFrame and returns control decisions.

The active request_0513_modular.py path uses FollowSafetyController.  The
earlier SafetyController/FollowController/DecisionPipeline classes are kept as
staged refactor candidates until the old entrypoints are retired or updated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
import logging
from typing import List, Optional

from .control_types import ControlAction, ControlDecision, PersonTarget, SensorFrame

logger = logging.getLogger("PersonTracker")


@dataclass(frozen=True)
class ControlConfig:
    target_distance_m: float = 1.0
    brake_distance_m: float = 0.5
    distance_missing_forward_percent: int = 50
    min_forward_percent: int = 10
    max_forward_percent: int = 100
    forward_speed_le_1_3_percent: int = 30
    forward_speed_le_1_7_percent: int = 35
    forward_speed_le_2_1_percent: int = 42
    forward_speed_le_2_6_percent: int = 50
    forward_speed_le_3_2_percent: int = 60
    forward_speed_le_3_8_percent: int = 70
    forward_speed_le_4_5_percent: int = 75
    forward_speed_far_percent: int = 80


class SafetyController:
    def decide(self, frame: SensorFrame, cfg: ControlConfig) -> Optional[ControlAction]:
        if frame.hazard.active:
            return ControlAction.stop(frame.hazard.reason or "hazard", brake_hold=True)
        if frame.obstacles.front:
            return ControlAction.stop("front_ir", brake_hold=True)
        if frame.distance_m is not None and frame.distance_m < cfg.brake_distance_m:
            return ControlAction.stop("distance_too_close", brake_hold=True)
        return None


class FollowController:
    def _clamp_forward_percent(self, percent: int, cfg: ControlConfig) -> int:
        p = min(int(cfg.max_forward_percent), int(percent))
        if p > 0:
            p = max(int(cfg.min_forward_percent), p)
        return max(0, p)

    def _forward_percent_for_distance(self, distance_m: float, cfg: ControlConfig) -> int:
        if distance_m <= cfg.target_distance_m:
            return 0
        if distance_m <= 1.3:
            return self._clamp_forward_percent(cfg.forward_speed_le_1_3_percent, cfg)
        if distance_m <= 1.7:
            return self._clamp_forward_percent(cfg.forward_speed_le_1_7_percent, cfg)
        if distance_m <= 2.1:
            return self._clamp_forward_percent(cfg.forward_speed_le_2_1_percent, cfg)
        if distance_m <= 2.6:
            return self._clamp_forward_percent(cfg.forward_speed_le_2_6_percent, cfg)
        if distance_m <= 3.2:
            return self._clamp_forward_percent(cfg.forward_speed_le_3_2_percent, cfg)
        if distance_m <= 3.8:
            return self._clamp_forward_percent(cfg.forward_speed_le_3_8_percent, cfg)
        if distance_m <= 4.5:
            return self._clamp_forward_percent(cfg.forward_speed_le_4_5_percent, cfg)
        return self._clamp_forward_percent(cfg.forward_speed_far_percent, cfg)

    def _largest_person(self, frame: SensorFrame) -> Optional[PersonTarget]:
        if not frame.persons:
            return None
        return max(frame.persons, key=lambda p: p.area)

    def decide(self, frame: SensorFrame, cfg: ControlConfig) -> Optional[ControlAction]:
        target = self._largest_person(frame)
        if target is None or frame.width <= 0 or frame.height <= 0:
            return None

        cx, _cy = target.center
        grid_w = frame.width / 6.0
        if cx < grid_w * 2:
            return ControlAction.rotate_left("person_left")
        if cx >= grid_w * 4:
            return ControlAction.rotate_right("person_right")

        if frame.distance_m is None:
            fallback_speed = max(0, min(cfg.max_forward_percent, int(cfg.distance_missing_forward_percent)))
            if 0 < fallback_speed < cfg.min_forward_percent:
                fallback_speed = cfg.min_forward_percent
            return ControlAction.forward(fallback_speed, "distance_missing_front_clear")
        speed = self._forward_percent_for_distance(frame.distance_m, cfg)
        if speed <= 0:
            return ControlAction.idle("distance_ok")
        return ControlAction.forward(speed, "follow_distance")


class DecisionPipeline:
    def __init__(self, cfg: ControlConfig) -> None:
        self.cfg = cfg
        self.safety = SafetyController()
        self.follow = FollowController()

    def decide(self, frame: SensorFrame) -> ControlAction:
        for controller in (self.safety, self.follow):
            action = controller.decide(frame, self.cfg)
            if action is not None:
                return action
        return ControlAction.idle("no_target")


@dataclass(frozen=True)
class FollowPolicyConfig:
    lost_confirm_frames: int = 3
    lost_confirm_sec: float = 0.0
    action_cooldown: int = 1
    search_cooldown: int = 1
    target_distance_m: float = 1.0
    brake_distance_m: float = 0.5
    distance_missing_forward_percent: int = 50
    obstacle_distance_m: float = 1.0
    min_forward_percent: int = 10
    max_forward_percent: int = 100
    forward_speed_le_1_3_percent: int = 30
    forward_speed_le_1_7_percent: int = 35
    forward_speed_le_2_1_percent: int = 42
    forward_speed_le_2_6_percent: int = 50
    forward_speed_le_3_2_percent: int = 60
    forward_speed_le_3_8_percent: int = 70
    forward_speed_le_4_5_percent: int = 75
    forward_speed_far_percent: int = 80
    center_left_ratio: float = 1.0 / 6.0
    center_right_ratio: float = 4.0 / 6.0
    steer_enter_left_ratio: Optional[float] = None
    steer_enter_right_ratio: Optional[float] = None
    steer_release_left_ratio: Optional[float] = None
    steer_release_right_ratio: Optional[float] = None
    visible_rotate_left_ratio: Optional[float] = None
    visible_rotate_right_ratio: Optional[float] = None
    use_vertical_center_gate: bool = False
    search_before_first_seen: bool = True
    startup_search_delay_sec: float = 0.0
    side_ir_blocks_rotation: bool = True
    search_rotate_front_block_enable: bool = True
    search_rotate_distance_block_enable: bool = True
    predictive_search_enabled: bool = False
    predictive_search_sec: float = 1.0
    predictive_far_forward_percent: int = 25
    visible_steer_inner_ratio_percent: int = 80
    visible_steer_outer_ratio_percent: int = 105
    visible_steer_strong_inner_ratio_percent: int = 70
    visible_steer_strong_outer_ratio_percent: int = 110
    visible_steer_strong_margin_ratio: float = 0.12


class FollowSafetyController:
    """Stateful controller matching the current request_0428 follow policy.

    It keeps search/lost state inside the controller and returns a decision
    object.  The request script remains responsible for motor I/O and queues.
    """

    def __init__(self, cfg: FollowPolicyConfig) -> None:
        self.cfg = cfg
        self.search_state = "none"
        self.search_direction: Optional[str] = None
        self.lost_confirm_frames = 0
        self.last_action_frame = -1
        self.last_person_center_x: Optional[float] = None
        self.last_dispatched_kind: Optional[str] = None
        self.active_target_id: Optional[int] = None
        self.last_selected_target: Optional[PersonTarget] = None
        self._has_seen_person = False
        self._startup_search_started_at = time.monotonic()
        self._lost_started_at: Optional[float] = None

    def set_last_dispatched(self, action_kind: Optional[str]) -> None:
        self.last_dispatched_kind = action_kind

    def _retag_action(self, action: ControlAction, reason: str) -> ControlAction:
        return ControlAction(
            kind=action.kind,
            speed_percent=action.speed_percent,
            steer_inner_ratio_percent=action.steer_inner_ratio_percent,
            steer_outer_ratio_percent=action.steer_outer_ratio_percent,
            reason=reason,
            brake_hold=action.brake_hold,
        )

    def _lost_wait_action_from_history(self, frame: SensorFrame) -> Optional[ControlAction]:
        cfg = self.cfg
        if not cfg.predictive_search_enabled:
            return None

        action = self._predictive_search_action(frame)
        if action is not None:
            return self._retag_action(action, "lost_wait_" + action.reason)

        if frame.lost_intent_age_sec > cfg.predictive_search_sec:
            return None
        if self.last_person_center_x is None or frame.width <= 0:
            return None

        left, right = self._active_center_band(frame.width)
        if self.last_person_center_x < left:
            return ControlAction.rotate_left("lost_wait_last_left")
        if self.last_person_center_x > right:
            return ControlAction.rotate_right("lost_wait_last_right")
        return None

    def _lost_confirm_wait_decision(self, first_lost_frame: bool, frame: SensorFrame) -> ControlDecision:
        action = self._lost_wait_action_from_history(frame)
        if action is not None:
            if self._is_action_blocked(action, frame):
                return ControlDecision(
                    waiting_lost_confirm=True,
                    explicit_stop_requested=True,
                    reason="lost_confirm_wait_predict_blocked",
                )
            return ControlDecision(
                actions=[action],
                waiting_lost_confirm=True,
                reason=action.reason,
                is_forwarding=action.kind == "forward",
                current_forward_percent=action.speed_percent if action.kind == "forward" else 0,
            )

        keep_current_rotate = self.last_dispatched_kind in ("rotate_left", "rotate_right")
        explicit_stop = bool(first_lost_frame and not keep_current_rotate)
        if keep_current_rotate:
            reason = "lost_confirm_wait_keep_rotate"
        else:
            reason = "lost_confirm_wait_stop" if first_lost_frame else "lost_confirm_wait"
        return ControlDecision(
            waiting_lost_confirm=True,
            explicit_stop_requested=explicit_stop,
            reason=reason,
        )

    def _center(self, person: PersonTarget) -> tuple:
        return person.center

    def _ratio_or_default(self, value: Optional[float], fallback: float) -> float:
        if value is None:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _band_from_ratios(self, width: int, left_ratio: float, right_ratio: float) -> tuple:
        left_ratio = max(0.0, min(1.0, float(left_ratio)))
        right_ratio = max(0.0, min(1.0, float(right_ratio)))
        if right_ratio <= left_ratio:
            left_ratio = 1.0 / 6.0
            right_ratio = 4.0 / 6.0
        return width * left_ratio, width * right_ratio

    def _center_band(self, width: int) -> tuple:
        return self._band_from_ratios(width, self.cfg.center_left_ratio, self.cfg.center_right_ratio)

    def _active_center_band(self, width: int) -> tuple:
        center_left = float(self.cfg.center_left_ratio)
        center_right = float(self.cfg.center_right_ratio)
        if self.last_dispatched_kind in ("steer_left", "steer_right"):
            left_ratio = self._ratio_or_default(self.cfg.steer_release_left_ratio, center_left)
            right_ratio = self._ratio_or_default(self.cfg.steer_release_right_ratio, center_right)
        else:
            left_ratio = self._ratio_or_default(self.cfg.steer_enter_left_ratio, center_left)
            right_ratio = self._ratio_or_default(self.cfg.steer_enter_right_ratio, center_right)
        return self._band_from_ratios(width, left_ratio, right_ratio)

    def _is_in_center_3x3(self, cx: float, cy: float, width: int, height: int) -> bool:
        left, right = self._active_center_band(width)
        in_center_cols = left <= cx < right
        if not self.cfg.use_vertical_center_gate:
            return in_center_cols

        grid_h = height / 6.0
        in_center_rows = grid_h * 1 <= cy < grid_h * 4
        return in_center_cols and in_center_rows

    def _edge_type(self, cx: float, width: int) -> str:
        left, right = self._active_center_band(width)
        if cx < left:
            return "left"
        if cx >= right:
            return "right"
        return "none"

    def _forward_percent_for_distance(self, distance_m: float) -> int:
        cfg = self.cfg
        if distance_m < cfg.brake_distance_m:
            return 0
        if distance_m <= cfg.target_distance_m:
            return 0
        if distance_m <= 1.3:
            p = cfg.forward_speed_le_1_3_percent
        elif distance_m <= 1.7:
            p = cfg.forward_speed_le_1_7_percent
        elif distance_m <= 2.1:
            p = cfg.forward_speed_le_2_1_percent
        elif distance_m <= 2.6:
            p = cfg.forward_speed_le_2_6_percent
        elif distance_m <= 3.2:
            p = cfg.forward_speed_le_3_2_percent
        elif distance_m <= 3.8:
            p = cfg.forward_speed_le_3_8_percent
        elif distance_m <= 4.5:
            p = cfg.forward_speed_le_4_5_percent
        else:
            p = cfg.forward_speed_far_percent
        p = min(int(cfg.max_forward_percent), int(p))
        if p > 0:
            p = max(int(cfg.min_forward_percent), p)
        return max(0, p)

    def _rotate_action_for_edge(self, edge_type: str) -> Optional[ControlAction]:
        if edge_type == "left":
            return ControlAction.rotate_left("person_left_rotate")
        if edge_type == "right":
            return ControlAction.rotate_right("person_right_rotate")
        return None

    def _visible_rotate_edge_type(self, cx: float, width: int) -> str:
        if width <= 0:
            return "none"
        left_ratio = self.cfg.visible_rotate_left_ratio
        right_ratio = self.cfg.visible_rotate_right_ratio
        if left_ratio is None and right_ratio is None:
            return "none"
        left = 0.0 if left_ratio is None else max(0.0, min(1.0, float(left_ratio)))
        right = 1.0 if right_ratio is None else max(0.0, min(1.0, float(right_ratio)))
        if left > 0.0 and cx <= width * left:
            return "left"
        if right < 1.0 and cx >= width * right:
            return "right"
        return "none"

    def _visible_base_forward_percent(self, frame: SensorFrame) -> int:
        cfg = self.cfg
        if frame.distance_m is not None:
            speed = self._forward_percent_for_distance(float(frame.distance_m))
            if speed <= 0 and float(frame.distance_m) >= float(cfg.brake_distance_m):
                speed = int(cfg.min_forward_percent)
            return speed
        speed = max(0, min(int(cfg.max_forward_percent), int(cfg.distance_missing_forward_percent)))
        if 0 < speed < cfg.min_forward_percent:
            speed = cfg.min_forward_percent
        return speed

    def _turn_action_for_visible_target(
        self,
        edge_type: str,
        cx: float,
        width: int,
        frame: SensorFrame,
    ) -> Optional[ControlAction]:
        if edge_type not in ("left", "right") or width <= 0:
            return None

        base_speed = self._visible_base_forward_percent(frame)
        if base_speed <= 0:
            return None
        margin = max(0.02, min(0.45, float(self.cfg.visible_steer_strong_margin_ratio)))
        strong = cx <= width * margin or cx >= width * (1.0 - margin)
        reason = "person_%s_strong" % edge_type if strong else "person_%s" % edge_type
        if strong:
            inner_ratio = int(self.cfg.visible_steer_strong_inner_ratio_percent)
            outer_ratio = int(self.cfg.visible_steer_strong_outer_ratio_percent)
        else:
            inner_ratio = int(self.cfg.visible_steer_inner_ratio_percent)
            outer_ratio = int(self.cfg.visible_steer_outer_ratio_percent)
        if edge_type == "left":
            return ControlAction.steer_left(base_speed, inner_ratio, outer_ratio, reason)
        return ControlAction.steer_right(base_speed, inner_ratio, outer_ratio, reason)

    def _is_action_blocked(self, action: ControlAction, frame: SensorFrame) -> bool:
        if action.kind == "forward":
            return frame.obstacles.front
        if action.kind in ("steer_left", "steer_right"):
            return frame.obstacles.front
        if action.kind in ("rotate_left", "rotate_right"):
            if self.cfg.search_rotate_front_block_enable and frame.obstacles.front:
                return True
            if self.cfg.search_rotate_distance_block_enable:
                distance_close = frame.distance_m is not None and frame.distance_m < self.cfg.brake_distance_m
                distance_latched = bool(getattr(frame.distance_state, "brake_latched", False))
                used_distance = getattr(frame.distance_state, "used_distance_m", None)
                used_close = used_distance is not None and used_distance < self.cfg.brake_distance_m
                if distance_close or distance_latched or used_close:
                    return True
            if not self.cfg.side_ir_blocks_rotation:
                return False
            if action.kind == "rotate_left":
                return frame.obstacles.left
            return frame.obstacles.right
        return False

    def _select_person(self, frame: SensorFrame) -> Optional[PersonTarget]:
        if not frame.persons:
            return None
        if self.active_target_id is not None:
            matches = [p for p in frame.persons if int(p.track_id) == int(self.active_target_id)]
            if matches:
                return max(matches, key=lambda p: p.area)
            if self._has_seen_person:
                return None
        return max(frame.persons, key=lambda p: p.area)

    def select_target_for_current_state(self, persons: List[PersonTarget]) -> Optional[PersonTarget]:
        """Return the target this controller would use before running decision logic."""
        return self._select_person(SensorFrame(persons=list(persons)))

    def _predictive_search_action(self, frame: SensorFrame) -> Optional[ControlAction]:
        cfg = self.cfg
        if not cfg.predictive_search_enabled:
            return None
        if frame.lost_intent_age_sec > cfg.predictive_search_sec:
            return None

        intent = (frame.lost_intent or "unknown").strip().lower()
        if intent == "left_exit":
            return ControlAction.rotate_left("predict_left_exit")
        if intent == "right_exit":
            return ControlAction.rotate_right("predict_right_exit")
        if intent == "far_exit":
            speed = max(0, min(int(cfg.max_forward_percent), int(cfg.predictive_far_forward_percent)))
            if 0 < speed < cfg.min_forward_percent:
                speed = cfg.min_forward_percent
            if speed > 0:
                return ControlAction.forward(speed, "predict_far_exit")
        return None

    def _fallback_search_action(self, frame: SensorFrame, reason_prefix: str) -> Optional[ControlAction]:
        if self.search_direction == "left":
            candidate = ControlAction.rotate_left(f"{reason_prefix}_left")
            if self._is_action_blocked(candidate, frame):
                opposite = ControlAction.rotate_right(f"{reason_prefix}_left_blocked")
                if not self._is_action_blocked(opposite, frame):
                    self.search_direction = "right"
                    return opposite
                return None
            return candidate

        candidate = ControlAction.rotate_right(f"{reason_prefix}_right")
        if self._is_action_blocked(candidate, frame):
            opposite = ControlAction.rotate_left(f"{reason_prefix}_right_blocked")
            if not self._is_action_blocked(opposite, frame):
                self.search_direction = "left"
                return opposite
            return None
        return candidate

    def _ensure_search_state(self, frame: SensorFrame) -> None:
        if self.search_state in ("searching", "predictive"):
            return
        if self.last_person_center_x is not None and frame.width > 0:
            self.search_direction = "left" if self.last_person_center_x < frame.width / 2.0 else "right"
        else:
            self.search_direction = "right"
        self.search_state = "searching"

    def decide(self, frame_index: int, frame: SensorFrame) -> ControlDecision:
        cfg = self.cfg
        now = time.monotonic()
        target = self._select_person(frame)
        self.last_selected_target = target

        if target is None:
            if not self._has_seen_person:
                if not cfg.search_before_first_seen:
                    self.lost_confirm_frames = 0
                    self.search_state = "none"
                    self.search_direction = None
                    self._lost_started_at = None
                    return ControlDecision(reason="wait_first_person")
                elapsed = now - self._startup_search_started_at
                if cfg.startup_search_delay_sec > 0 and elapsed < cfg.startup_search_delay_sec:
                    self.lost_confirm_frames = 0
                    self.search_state = "none"
                    self.search_direction = None
                    self._lost_started_at = None
                    return ControlDecision(reason="startup_wait_before_search")
            else:
                self.lost_confirm_frames += 1
                first_lost_frame = self._lost_started_at is None
                if first_lost_frame:
                    self._lost_started_at = now

                lost_confirmed = False
                lost_elapsed_sec = now - self._lost_started_at
                if float(cfg.lost_confirm_sec) > 0.0:
                    if lost_elapsed_sec < float(cfg.lost_confirm_sec):
                        return self._lost_confirm_wait_decision(first_lost_frame, frame)
                    lost_confirmed = True
                else:
                    if self.lost_confirm_frames < cfg.lost_confirm_frames:
                        return self._lost_confirm_wait_decision(first_lost_frame, frame)
                    lost_confirmed = True

                if lost_confirmed:
                    self._ensure_search_state(frame)
                    if self.active_target_id is not None:
                        old_target_id = int(self.active_target_id)
                        self.active_target_id = None
                        logger.info(
                            "free_search_enter frame=%d cleared_target=%d lost_frames=%d lost_sec=%.2f threshold_frames=%d threshold_sec=%.2f",
                            int(frame_index),
                            old_target_id,
                            int(self.lost_confirm_frames),
                            float(lost_elapsed_sec),
                            int(cfg.lost_confirm_frames),
                            float(cfg.lost_confirm_sec),
                        )
                        target = self._select_person(frame)
                        self.last_selected_target = target
                    if float(cfg.lost_confirm_sec) <= 0.0 and self.lost_confirm_frames == cfg.lost_confirm_frames:
                        self.lost_confirm_frames = cfg.lost_confirm_frames + 1

            if target is None:
                self._ensure_search_state(frame)

        if target is None:
            frames_since_last_action = frame_index - self.last_action_frame if self.last_action_frame >= 0 else 999
            if frames_since_last_action >= cfg.search_cooldown:
                action = self._predictive_search_action(frame)
                if action is not None and not self._is_action_blocked(action, frame):
                    self.search_state = "predictive"
                    if action.kind == "rotate_left":
                        self.search_direction = "left"
                    elif action.kind == "rotate_right":
                        self.search_direction = "right"
                else:
                    self.search_state = "searching"
                    action = self._fallback_search_action(frame, "search")
                    if action is None:
                        return ControlDecision(explicit_stop_requested=True, reason="search_both_sides_blocked")
                self.last_action_frame = frame_index
                decision = ControlDecision(
                    actions=[action],
                    reason=action.reason,
                    is_forwarding=(action.kind == "forward"),
                    current_forward_percent=action.speed_percent if action.kind == "forward" else 0,
                )
                return decision

            if self.last_dispatched_kind in ("rotate_left", "rotate_right"):
                return ControlDecision(explicit_stop_requested=True, reason="search_cooldown_stop_rotate")
            return ControlDecision(reason="search_cooldown_keep_motion")

        person_detected_flag = False
        clear_queue = False
        stop_execution = False
        if self.search_state in ("searching", "predictive"):
            self.search_state = "none"
            self.search_direction = None
            person_detected_flag = True
            clear_queue = True
            stop_execution = True
        if self.active_target_id is None or int(self.active_target_id) != int(target.track_id):
            self.active_target_id = int(target.track_id)

        self.lost_confirm_frames = 0
        self._lost_started_at = None
        self._has_seen_person = True
        cx, cy = self._center(target)
        self.last_person_center_x = cx

        in_center = self._is_in_center_3x3(cx, cy, frame.width, frame.height)
        frames_since_last_action = frame_index - self.last_action_frame if self.last_action_frame >= 0 else 999

        # Safety gates.
        if frame.hazard.active:
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=person_detected_flag,
                clear_action_queue=clear_queue,
                stop_action_execution=stop_execution,
                reason=frame.hazard.reason or "hazard",
            )
        if frame.obstacles.front:
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=person_detected_flag,
                clear_action_queue=clear_queue,
                stop_action_execution=stop_execution,
                reason="front_ir",
            )

        if not in_center:
            if frame.distance_m is not None and frame.distance_m < cfg.brake_distance_m:
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="person_too_close_no_rotate",
                )
            edge = self._edge_type(cx, frame.width)
            rotate_edge = self._visible_rotate_edge_type(cx, frame.width)
            if rotate_edge != "none":
                action = self._rotate_action_for_edge(rotate_edge)
            else:
                action = self._turn_action_for_visible_target(edge, cx, frame.width, frame)
            if action is None:
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="turn_edge_unknown",
                )
            if self._is_action_blocked(action, frame):
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="turn_side_blocked",
                )
            if frames_since_last_action < cfg.action_cooldown:
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="rotate_cooldown",
                )
            self.last_action_frame = frame_index
            return ControlDecision(
                actions=[action],
                person_detected_flag=person_detected_flag,
                is_forwarding=False,
                current_forward_percent=0,
                clear_action_queue=clear_queue,
                stop_action_execution=stop_execution,
                reason=action.reason,
            )

        # Centered: distance follow.
        if frame.distance_m is not None:
            if frame.distance_m < cfg.brake_distance_m:
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="distance_too_close",
                )
            if frame.distance_m > cfg.target_distance_m:
                speed = self._forward_percent_for_distance(frame.distance_m)
                if speed <= 0:
                    return ControlDecision(
                        person_detected_flag=person_detected_flag,
                        clear_action_queue=clear_queue,
                        stop_action_execution=stop_execution,
                        reason="distance_speed_zero",
                    )
                self.last_action_frame = frame_index
                return ControlDecision(
                    actions=[ControlAction.forward(speed, "follow_distance")],
                    person_detected_flag=person_detected_flag,
                    is_forwarding=True,
                    current_forward_percent=speed,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="follow_distance",
                )
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=person_detected_flag,
                clear_action_queue=clear_queue,
                stop_action_execution=stop_execution,
                reason="target_distance_reached",
            )

        if not frame.obstacles.front:
            fallback_speed = max(0, min(cfg.max_forward_percent, int(cfg.distance_missing_forward_percent)))
            if 0 < fallback_speed < cfg.min_forward_percent:
                fallback_speed = cfg.min_forward_percent
            self.last_action_frame = frame_index
            return ControlDecision(
                actions=[ControlAction.forward(fallback_speed, "distance_missing_front_clear")],
                person_detected_flag=person_detected_flag,
                is_forwarding=True,
                current_forward_percent=fallback_speed,
                clear_action_queue=clear_queue,
                stop_action_execution=stop_execution,
                reason="distance_missing_front_clear",
            )

        return ControlDecision(
            explicit_stop_requested=True,
            person_detected_flag=person_detected_flag,
            clear_action_queue=clear_queue,
            stop_action_execution=stop_execution,
            reason="distance_missing_front_blocked",
        )
