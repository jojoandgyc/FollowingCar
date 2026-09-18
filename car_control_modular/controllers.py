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

import math
import time
from dataclasses import dataclass, replace
import logging
from typing import List, Optional, Tuple

from .control_types import (
    ControlAction,
    ControlDecision,
    LateralCandidateEvidence,
    PersonTarget,
    SearchControlStatus,
    SensorFrame,
)
from .steering_pid import (
    DistancePidConfig,
    DistancePidResult,
    LongitudinalDistancePid,
    VisualSteeringPid,
    VisualSteeringPidConfig,
    VisualSteeringPidResult,
)
from .target_direction_history import TargetDirectionHistory
from .longitudinal_approach import (
    ApproachConfig, RawDepthClosingWindow, closure_rotation_bound, bounded_encoder_fallback,
)
from .distance_pi import DistancePiConfig
from .longitudinal_feedforward import (
    LongitudinalFeedforwardConfig, LongitudinalFeedforwardEstimator, LongitudinalFeedforwardBridge,
    bounded_disagreeing_yaw_rotation,
    depth_rotation_rate,
    far_closure_consistent,
)
logger = logging.getLogger("PersonTracker")

# 视觉层使用 -2 表示“只有单人几何兜底，尚未锁定正式 ReID”。
# 这个 ID 不能进入丢失目标搜索，否则启动时一次几何候选漏帧就会误触发原地旋转。
GEOMETRY_FALLBACK_TARGET_ID = -2


@dataclass(frozen=True)
class _CurrentLateralCandidate:
    position: str
    center_ratio: float
    aimline_gap_ratio: float
    aimline_intersects: bool
    evidence: LateralCandidateEvidence


@dataclass(frozen=True)
class ControlConfig:
    target_distance_m: float = 1.5
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
        if frame.obstacles.front or frame.obstacles.left or frame.obstacles.right:
            if frame.obstacles.front:
                reason = "front_ir"
            elif frame.obstacles.left:
                reason = "left_ir"
            else:
                reason = "right_ir"
            return ControlAction.stop(reason, brake_hold=True)
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
    # A stale vision result creates an information gap. Do not promote the
    # previous image side directly into a full single-direction search.
    stale_direction_recovery_enable: bool = False
    stale_direction_observe_frames: int = 2
    stale_direction_observe_max_sec: float = 0.20
    stale_direction_settle_yaw_rate_dps: float = 3.0
    # Kept for config compatibility; stale recovery no longer probes or
    # returns to origin using encoder yaw.
    stale_direction_probe_enable: bool = False
    stale_direction_probe_angle_deg: float = 12.0
    stale_direction_probe_observe_frames: int = 2
    stale_direction_probe_return_tolerance_deg: float = 2.0
    direction_history_enable: bool = False
    direction_history_max_frames: int = 60
    direction_history_lookback_samples: int = 6
    direction_history_max_capture_gap: int = 12
    direction_history_edge_margin_ratio: float = 0.04
    direction_history_outer_center_ratio: float = 0.60
    direction_history_min_motion_ratio: float = 0.015
    direction_history_min_consistent_steps: int = 1
    # Detector-only capture evidence may provide a bounded advisory search
    # direction after a loss. It never becomes target-owned history.
    historical_direction_backfill_enable: bool = True
    historical_direction_backfill_max_age_sec: float = 0.70
    historical_direction_backfill_min_samples: int = 2
    historical_direction_backfill_max_capture_gap: int = 6
    historical_direction_backfill_max_center_jump_ratio: float = 0.30
    historical_direction_backfill_min_area_similarity: float = 0.45
    historical_direction_backfill_confidence_cap: float = 0.70
    historical_direction_backfill_min_score: float = 0.20
    # This threshold admits detector-only boxes as lateral evidence. An
    # opposite-side box still needs an active-target match or target geometry
    # continuity before it may reverse the current search direction.
    search_candidate_untracked_min_score: float = 0.10
    # An unconfirmed detector box may keep the current search side, but it may
    # reverse that side only when its geometry is continuous with the last
    # target-owned box. This rejects another person appearing across the image.
    search_candidate_continuity_max_capture_gap: int = 6
    search_candidate_continuity_max_center_jump_ratio: float = 0.25
    search_candidate_continuity_min_horizontal_overlap: float = 0.08
    # Slow a search when a fresh detector-only person box approaches the
    # camera aimline. Once the aimline intersects that box, actively stop
    # without waiting for Tracker/ReID identity confirmation.
    search_candidate_approach_margin_ratio: float = 0.08
    steer_min_hold_sec: float = 0.50
    steer_lost_hold_frames: int = 3
    # 配置允许时，视觉短暂漏检可继续保持上一条低速差速动作。
    steer_lost_hold_max_sec: float = 0.0
    # 摄像头偶发漏检时，只允许沿用上一条直行，并降到固定安全转速。
    lost_forward_hold_rpm: int = 20
    # 目标从视觉列表消失后先停车确认，不盲目前进。
    lost_forward_hold_max_sec: float = 0.0
    lost_forward_hold_min_distance_m: float = 1.50
    depth_medium_confidence_hold_sec: float = 0.60
    depth_medium_confidence_rpm: int = 20
    depth_recovery_stage1_sec: float = 0.20
    depth_recovery_stage2_sec: float = 0.40
    depth_recovery_stage1_rpm: int = 25
    depth_recovery_stage2_rpm: int = 45
    action_cooldown: int = 1
    search_cooldown: int = 1
    target_distance_m: float = 1.5
    # 目标距离停车回差：进入目标距离后先锁存，必须连续多帧超过更远的释放距离
    # 并保持一段时间，才能再次前进，避免毫米波在阈值附近抖动导致反复启停。
    target_distance_release_m: float = 1.5
    target_distance_release_hold_sec: float = 0.50
    target_distance_release_confirm_frames: int = 3
    target_distance_release_visual_shrink_ratio: float = 0.90
    target_distance_release_visual_edge_margin_ratio: float = 0.02
    brake_distance_m: float = 0.5
    # Runtime board policy: Depth is for speed/reverse; IR owns explicit parking.
    distance_parking_enable: bool = True
    # 倒车只允许由当前人物框的实时稳定Depth或已匹配毫米波触发；任何hold均无权倒车。
    reverse_enable: bool = False
    # 近距离横向修正只允许原地旋转，禁止把倒车纵向分量叠加到转向动作。
    # 默认关闭以保持库调用方的既有策略；板端运行配置显式开启。
    near_distance_rotate_only_enable: bool = False
    near_distance_rotate_only_distance_m: float = 1.80
    # Keep near-target in-place yaw below the normal steering authority. The
    # detector cadence is slower than the motor loop, so a large correction can
    # accumulate chassis yaw before the next visual update arrives.
    near_distance_rotation_only_max_rpm: int = 10
    # Near-target lateral settling: require a few fresh, low-yaw samples in
    # the center band before allowing another correction.
    near_distance_settle_confirm_frames: int = 2
    near_distance_settle_hold_sec: float = 0.25
    near_distance_settle_release_margin_ratio: float = 0.03
    near_distance_settle_release_frames: int = 2
    near_distance_disable_rate_feedforward: bool = True
    reverse_start_distance_m: float = 1.50
    # 新鲜 Depth 已明显小于目标距离时立即倒车，不再等待上一帧趋势确认。
    reverse_immediate_distance_m: float = 1.35
    reverse_stop_distance_m: float = 1.55
    reverse_full_speed_distance_m: float = 1.00
    reverse_min_rpm: int = 20
    reverse_max_rpm: int = 100
    # 倒车启动不能只靠距离误差慢慢爬升：目标正在快速靠近时加入速度前馈。
    # runtime_cap 是当前实车验证上限，reverse_max_rpm 仍是系统绝对上限。
    reverse_feedforward_floor_rpm: int = 30
    reverse_feedforward_gain_rpm_per_m_s: float = 40.0
    reverse_runtime_cap_rpm: int = 60
    reverse_approach_speed_filter_alpha: float = 0.50
    reverse_min_approach_delta_m: float = 0.03
    reverse_confirm_frames: int = 1
    reverse_radar_max_age_sec: float = 0.25
    reverse_distance_missing_hold_sec: float = 0.35
    reverse_visual_guard_area_ratio: float = 0.45
    reverse_visual_guard_height_ratio: float = 0.92
    reverse_visual_guard_growth_ratio: float = 1.18
    reverse_visual_guard_growth_min_area_ratio: float = 0.20
    # A recent trusted Depth reading beyond this distance vetoes visual-only
    # reverse. This prevents a tall, distant person box from looking "near".
    reverse_visual_guard_max_distance_m: float = 1.60
    reverse_visual_guard_rpm: int = 30
    # A stopped vehicle must move clearly beyond the hold band before forward
    # motion can restart. Once moving, it stops at the nearer release point.
    forward_start_distance_m: float = 1.80
    forward_stop_distance_m: float = 1.65
    distance_missing_forward_percent: int = 50
    mmwave_hold_forward_percent: int = 50
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
    # Optional symmetric dead zone around the optical center. Explicit
    # enter/release hysteresis bands take precedence when configured.
    center_deadzone_ratio: Optional[float] = None
    steer_enter_left_ratio: Optional[float] = None
    steer_enter_right_ratio: Optional[float] = None
    steer_release_left_ratio: Optional[float] = None
    steer_release_right_ratio: Optional[float] = None
    visible_rotate_left_ratio: Optional[float] = None
    visible_rotate_right_ratio: Optional[float] = None
    use_vertical_center_gate: bool = False
    search_before_first_seen: bool = False
    startup_search_delay_sec: float = 0.0
    search_timeout_sec: float = 4.0
    search_timeout_exit_program: bool = False
    # A confirmed lost target gets one complete in-place scan before runtime
    # shutdown. Encoder-integrated yaw is preferred; timeout remains only a
    # fallback when feedback is unavailable or stale.
    search_revolution_deg: float = 360.0
    search_revolution_feedback_stale_sec: float = 0.80
    # Board runtime can terminate immediately after confirmed target loss
    # instead of entering the lost-target rotation/search state machine.
    exit_on_target_loss: bool = False
    initial_target_confirm_frames: int = 1
    release_target_on_lost: bool = True
    side_ir_blocks_rotation: bool = True
    search_rotate_front_block_enable: bool = True
    search_rotate_distance_block_enable: bool = True
    mmwave_hold_decel_step_percent: int = 8
    mmwave_hold_recover_step_percent: int = 6
    # 融合距离置信度下降时的最低速度上限；不会覆盖近距离/红外硬停。
    mmwave_fusion_low_confidence_forward_percent: int = 60
    # 直行距离曲线：目标距离外从最低 RPM 平滑加速到最大 RPM。
    forward_min_rpm: int = 20
    forward_max_rpm: int = 100
    forward_curve_max_distance_m: float = 5.0
    forward_curve_exponent: float = 0.75
    # 纵向距离外环 PID：实际距离 - 目标距离 -> 前进目标 RPM。
    # Runtime config enables this explicitly; keep direct legacy/test callers
    # on the old curve unless they opt into the cascade.
    distance_pid_enable: bool = False
    # Explicit opt-in; callers without this field retain their legacy/profile policy.
    distance_control_mode: str = "legacy"
    distance_pi_kp_per_sec: float = 1.0
    distance_pi_ki_per_sec2: float = 0.4
    distance_pi_integral_max_m_s: float = 0.8
    distance_pi_memory_sec: float = 0.35
    distance_pi_motion_memory_sec: float = 0.0
    distance_pi_launch_request_rpm: float = 0.0
    # Forward authority only; measurement/PI/velocity updates stay at180ms.
    depth_longitudinal_sample_max_age_sec: float = 0.18
    distance_approach_enable: bool = False
    distance_approach_gain_per_sec: float = 1.0
    distance_approach_max_catchup_m_s: float = 0.60
    distance_approach_deceleration_m_s2: float = 0.40
    distance_approach_response_delay_sec: float = 0.20
    distance_approach_no_matching_max_rpm: float = 0.0
    distance_approach_matching_enable: bool = True
    distance_pid_kp_rpm_per_m: float = 22.0
    distance_pid_ki_rpm_per_m_s: float = 1.5
    distance_pid_kd_rpm_s_per_m: float = 6.0
    distance_pid_integral_limit_m_s: float = 1.5
    distance_pid_forward_integral_limit_m_s: float = 0.0
    distance_pid_deadband_m: float = 0.005
    distance_feedforward_enable: bool = False
    distance_turn_compensation_enable: bool = False
    depth_measured_recovery_enable: bool = False
    distance_feedforward_max_rpm: float = 20.0
    distance_matching_base_max_rpm: float = 0.0
    distance_matching_test_bias_rpm: float = 0.0
    distance_feedforward_wheel_circumference_m: float = 0.60
    distance_pid_derivative_filter_alpha: float = 0.25
    # Optional runtime guards for noisy depth. Zero keeps the outer loop
    # behavior unchanged for callers that do not explicitly enable them.
    distance_pid_max_measurement_jump_m: float = 0.0
    distance_pid_output_rise_rpm_per_sec: float = 0.0
    distance_pid_output_fall_rpm_per_sec: float = 0.0
    visible_steer_fine_inner_ratio_percent: int = 90
    visible_steer_fine_outer_ratio_percent: int = 100
    visible_steer_inner_ratio_percent: int = 80
    visible_steer_outer_ratio_percent: int = 105
    visible_steer_strong_inner_ratio_percent: int = 70
    visible_steer_strong_outer_ratio_percent: int = 110
    visible_steer_strong_margin_ratio: float = 0.12
    visible_motion_history_frames: int = 4
    visible_motion_lookback_sec: float = 0.10
    visible_motion_rate_filter_alpha: float = 0.55
    visible_motion_min_ratio: float = 0.015
    visible_motion_projection_gain: float = 0.80
    visible_motion_strong_ratio: float = 0.05
    visible_steering_pid_enable: bool = False
    visible_steering_pid_camera_hfov_deg: float = 90.0
    visible_steering_pid_camera_latency_sec: float = 0.13
    visible_steering_pid_deadband_deg: float = 1.2
    visible_steering_pid_outer_kp_per_sec: float = 1.85
    visible_steering_pid_outer_kd_sec: float = 0.08
    visible_steering_pid_target_rate_feedforward_gain: float = 0.0
    visible_steering_pid_target_rate_feedforward_max_dps: float = 0.0
    visible_steering_pid_target_speed_match_max_closing_dps: float = 0.0
    visible_steering_pid_max_yaw_rate_dps: float = 46.0
    visible_steering_pid_rate_kp_rpm_per_dps: float = 0.16
    visible_steering_pid_rate_ki_rpm_per_deg: float = 0.015
    visible_steering_pid_integral_limit_deg: float = 25.0
    visible_steering_pid_max_correction_rpm: float = 16.0
    visible_steering_pid_dynamic_small_error_deg: float = 3.5
    visible_steering_pid_dynamic_large_error_deg: float = 14.0
    visible_steering_pid_dynamic_small_max_yaw_rate_dps: float = 22.0
    visible_steering_pid_dynamic_small_max_correction_rpm: float = 6.0
    visible_steering_pid_dynamic_large_error_base_cap_rpm: float = 28.0
    visible_steering_pid_opposite_yaw_brake_threshold_dps: float = 6.0
    visible_steering_pid_opposite_yaw_brake_boost_rpm: float = 6.0
    visible_steering_pid_braking_max_correction_rpm: float = 12.0
    # Runtime enables this explicitly; keep standalone controller callers on
    # the established fixed opposite-yaw braking cap.
    visible_steering_pid_fast_countersteer_max_correction_rpm: float = 0.0
    visible_steering_pid_fast_countersteer_gain_rpm_per_dps: float = 0.0
    visible_steering_pid_same_direction_overspeed_threshold_dps: float = 8.0
    visible_steering_pid_same_direction_overspeed_brake_gain_rpm_per_dps: float = 0.25
    visible_steering_pid_visual_direction_guard_enabled: bool = False
    visible_steering_pid_predictive_brake_decel_dps2: float = 0.0
    visible_steering_pid_predictive_brake_margin_deg: float = 0.0
    visible_steering_pid_predictive_brake_response_sec: float = 0.0
    visible_steering_pid_min_effective_error_deg: float = 0.0
    visible_steering_pid_min_effective_correction_rpm: float = 0.0
    visible_steering_pid_mechanical_tier2_error_deg: float = 0.0
    visible_steering_pid_mechanical_tier2_correction_rpm: float = 0.0
    visible_steering_pid_mechanical_tier3_error_deg: float = 0.0
    visible_steering_pid_mechanical_tier3_correction_rpm: float = 0.0
    visible_steering_pid_mechanical_floor_release_ratio: float = 0.0
    visible_steering_pid_startup_kick_error_deg: float = 0.0
    visible_steering_pid_startup_kick_rpm: float = 0.0
    visible_steering_pid_startup_kick_max_sec: float = 0.0
    visible_steering_pid_startup_kick_release_yaw_rate_dps: float = 0.0
    visible_steering_pid_active_brake_yaw_threshold_dps: float = 0.0
    visible_steering_pid_active_brake_min_correction_rpm: float = 0.0
    visible_steering_pid_edge_boost_start_error_deg: float = 18.0
    visible_steering_pid_aggressive_inner_wheel_margin_rpm: float = 0.0
    # 距离缺失只限制前进速度，不应把摄像头横向闭环也压到无法追人的 3 RPM。
    visible_steering_pid_fallback_max_correction_rpm: float = 16.0
    visible_steering_pid_lost_hold_max_correction_rpm: int = 8
    visible_steering_pid_left_body_deg_per_encoder_deg: float = 0.5225
    visible_steering_pid_right_body_deg_per_encoder_deg: float = 0.5424
    visible_steering_pid_feedback_stale_sec: float = 0.30
    visible_steering_pid_error_filter_alpha: float = 0.60
    visible_steering_pid_derivative_filter_alpha: float = 0.25
    visible_steering_pid_fallback_base_rpm: int = 15
    # 到达跟随停车距离后只做低速原地居中；与 16-20 RPM 的丢失搜索分开。
    parked_recenter_min_rpm: int = 2
    parked_recenter_max_rpm: int = 5


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
        self._search_rotation_started_at: Optional[float] = None
        self._search_rotation_accumulated_deg = 0.0
        self._search_rotation_last_integrated_yaw_deg: Optional[float] = None
        self._search_rotation_origin_integrated_yaw_deg: Optional[float] = None
        self._search_heading_from_loss_deg = 0.0
        self._search_heading_min_deg = 0.0
        self._search_heading_max_deg = 0.0
        self._search_rotation_feedback_last_ts: Optional[float] = None
        self._search_rotation_feedback_seen = False
        self._last_search_rotation_log_deg = 0.0
        self._search_observation_hold = False
        self._lost_started_at: Optional[float] = None
        self._lost_exit_direction: Optional[str] = None
        self._lost_hint_confidence = 0.0
        self._lost_hint_source = "none"
        self._historical_direction_hint = None
        self._direction_loss_capture_id: Optional[int] = None
        self._direction_latest_visible_capture_id = 0
        self._candidate_geometry_anchor_bbox: Optional[Tuple[float, float, float, float]] = None
        self._candidate_geometry_anchor_capture_frame_id = -1
        self._target_direction_history = TargetDirectionHistory(
            max_frames=cfg.direction_history_max_frames,
            lookback_samples=cfg.direction_history_lookback_samples,
            max_capture_gap=cfg.direction_history_max_capture_gap,
            edge_margin_ratio=cfg.direction_history_edge_margin_ratio,
            outer_center_ratio=cfg.direction_history_outer_center_ratio,
            min_motion_ratio=cfg.direction_history_min_motion_ratio,
            min_consistent_steps=cfg.direction_history_min_consistent_steps,
        )
        self._stale_direction_recovery_active = False
        self._stale_direction_recovery_stage = "none"
        self._stale_direction_recovery_started_at: Optional[float] = None
        self._stale_direction_prior_hint: Optional[str] = None
        self._stale_direction_observe_frames = 0
        self._stale_direction_probe_observe_frames = 0
        self._stale_candidate_center_ratio: Optional[float] = None
        self._stale_candidate_side: Optional[str] = None
        self._stale_candidate_center_source = "none"
        self._stale_candidate_center_anchor_heading_deg: Optional[float] = None
        self._candidate_center_reason_prefix = "stale_candidate_center"
        self._candidate_center_missing_frames = 0
        self._initial_candidate_id: Optional[int] = None
        self._initial_candidate_frames = 0
        self._last_target_distance_m: Optional[float] = None
        self._last_target_distance_at: Optional[float] = None
        self._last_mmwave_motion_speed_percent: Optional[int] = None
        self._mmwave_speed_recovery_active = False
        self._distance_pid = LongitudinalDistancePid(
            DistancePidConfig(
                kp_rpm_per_m=float(cfg.distance_pid_kp_rpm_per_m),
                ki_rpm_per_m_s=float(cfg.distance_pid_ki_rpm_per_m_s),
                kd_rpm_s_per_m=float(cfg.distance_pid_kd_rpm_s_per_m),
                integral_limit_m_s=float(cfg.distance_pid_integral_limit_m_s),
                forward_integral_limit_m_s=float(cfg.distance_pid_forward_integral_limit_m_s),
                deadband_m=float(cfg.distance_pid_deadband_m),
                min_forward_output_rpm=float(cfg.forward_min_rpm),
                max_forward_output_rpm=float(cfg.forward_max_rpm),
                min_reverse_output_rpm=float(cfg.reverse_min_rpm),
                max_reverse_output_rpm=float(cfg.reverse_max_rpm),
                derivative_filter_alpha=float(cfg.distance_pid_derivative_filter_alpha),
                max_measurement_jump_m=float(cfg.distance_pid_max_measurement_jump_m),
                output_rise_rpm_per_sec=float(cfg.distance_pid_output_rise_rpm_per_sec),
                output_fall_rpm_per_sec=float(cfg.distance_pid_output_fall_rpm_per_sec),
                approach_profile=ApproachConfig(
                    gain_per_sec=cfg.distance_approach_gain_per_sec,
                    max_catchup_m_s=cfg.distance_approach_max_catchup_m_s,
                    deceleration_m_s2=cfg.distance_approach_deceleration_m_s2,
                    response_delay_sec=cfg.distance_approach_response_delay_sec,
                    no_matching_max_rpm=cfg.distance_approach_no_matching_max_rpm,
                    wheel_circumference_m=cfg.distance_feedforward_wheel_circumference_m,
                ) if cfg.distance_approach_enable and not self.distance_pi_enabled else None,
                pi_profile=DistancePiConfig(
                    kp_per_sec=cfg.distance_pi_kp_per_sec,
                    ki_per_sec2=cfg.distance_pi_ki_per_sec2,
                    integral_max_m_s=cfg.distance_pi_integral_max_m_s,
                    retain_integral_sec=cfg.distance_pi_memory_sec,
                    motion_memory_sec=cfg.distance_pi_motion_memory_sec,
                    launch_request_rpm=cfg.distance_pi_launch_request_rpm,
                    physical_ttl_sec=cfg.depth_longitudinal_sample_max_age_sec,
                    wheel_circumference_m=cfg.distance_feedforward_wheel_circumference_m,
                    deceleration_m_s2=cfg.distance_approach_deceleration_m_s2,
                    response_delay_sec=cfg.distance_approach_response_delay_sec,
                ) if self.distance_pi_enabled else None,
            )
        )
        self.last_distance_pid_result: Optional[DistancePidResult] = None
        if self.distance_pi_enabled:
            logger.info(
                "distance_control_mode mode=distance_pi kp_per_sec=%.3f ki_per_sec2=%.3f "
                "integral_max_m_s=%.3f memory_ms=%.0f matching_output=False "
                "matching_diagnostics=True depth_ttl_ms=%.0f fresh_update_ms=180 recovery_policy=unified "
                "deceleration_assumed=True launch_request_rpm=%.1f motion_memory_ms=%.0f",
                cfg.distance_pi_kp_per_sec, cfg.distance_pi_ki_per_sec2,
                cfg.distance_pi_integral_max_m_s, 1000 * cfg.distance_pi_memory_sec,
                1000 * cfg.depth_longitudinal_sample_max_age_sec,
                cfg.distance_pi_launch_request_rpm,
                cfg.distance_pi_motion_memory_sec*1000.,
            )
        elif cfg.distance_approach_enable:
            logger.info(
                "Longitudinal approach profile enabled gain_per_sec=%.3f max_catchup_m_s=%.3f "
                "deceleration_m_s2=%.3f delay_sec=%.3f wheel_circumference_m=%.6f "
                "forward_pid_replaced=True bias_trial_ignored=True deceleration_assumed=True "
                "no_matching_max_rpm=%.1f braking_source=raw_depth_window",
                cfg.distance_approach_gain_per_sec, cfg.distance_approach_max_catchup_m_s,
                cfg.distance_approach_deceleration_m_s2, cfg.distance_approach_response_delay_sec,
                cfg.distance_feedforward_wheel_circumference_m,
                cfg.distance_approach_no_matching_max_rpm,
            )
        self._distance_pid_last_input_m: Optional[float] = None
        self._distance_pid_last_update_at: Optional[float] = None
        self._distance_pid_sample_timestamp: Optional[float] = None
        self._distance_approach_sample_trusted = False
        self._raw_closing_window = RawDepthClosingWindow(max_gap_sec=(
            min(.30, max(.18, cfg.distance_pi_motion_memory_sec))
            if self.distance_pi_enabled else .18))
        self._distance_pi_motion_memory_allowed = False
        self._distance_pi_raw_distance_m = None
        self._closure_rejected_observation = None
        # Same physical input stream, separate qualifications. FF rejection
        # must not erase already-qualified closure evidence.
        self._matching_motion_window = RawDepthClosingWindow()
        self._braking_range_rate = 0.0
        self._braking_rate_source = "encoder_fallback"
        self._distance_pi_ego_forward_rpm = None
        self._distance_pi_feedback_timestamp = None
        self._distance_pi_pause_key = None
        self._live_longitudinal_authority_reader = None
        self._distance_pid_last_sample_timestamp: Optional[float] = None
        self._distance_pid_last_forward_control = True
        self._longitudinal_motion_uid: Optional[int] = None
        self._longitudinal_motion_stamp: Optional[float] = None
        self._longitudinal_motion_evidence = None
        self._longitudinal_bridge = LongitudinalFeedforwardBridge()
        self._longitudinal_bridge_output_cap = None
        if cfg.distance_matching_test_bias_rpm not in (0., 5., 10.):
            raise ValueError("distance_matching_test_bias_rpm must be 0, 5 or 10")
        self._bias_audit_key = None
        feedback_rpm_limit = min(105.0, max(100.0, float(cfg.forward_max_rpm)))
        if self.distance_pi_enabled:
            # PI is not subject to the old 100 RPM human-speed qualification.
            # This accepts feedback, not permission to command more speed.
            feedback_rpm_limit = max(0., float(cfg.forward_max_rpm)) + 5.
        elif cfg.distance_approach_enable:
            # Do not request base80+chase44 and then invalidate that same
            # legitimate measured speed at the old105RPM boundary. Bound
            # acceptance to this mode's command budget plus5RPM tolerance;
            # this does not raise a command, wheel or matching-speed limit.
            profile = self._distance_pid.config.approach_profile
            matching_budget = (cfg.distance_matching_base_max_rpm if cfg.distance_matching_base_max_rpm > 0
                               else cfg.forward_min_rpm + cfg.distance_feedforward_max_rpm)
            feedback_rpm_limit = min(float(cfg.forward_max_rpm) + 5., max(
                feedback_rpm_limit, matching_budget + profile.correction_max_rpm + 5.))
        self._longitudinal_feedforward = LongitudinalFeedforwardEstimator(
            LongitudinalFeedforwardConfig(
                wheel_circumference_m=float(cfg.distance_feedforward_wheel_circumference_m),
                baseline_rpm=float(cfg.forward_min_rpm),
                max_feedforward_rpm=float(cfg.distance_feedforward_max_rpm),
                max_tracking_base_rpm=min(float(cfg.forward_max_rpm), float(cfg.distance_matching_base_max_rpm)),
                max_abs_ego_rpm=feedback_rpm_limit,
                min_distance_m=max(float(cfg.brake_distance_m), float(cfg.target_distance_m) - 0.03),
            ), shared_window=self._matching_motion_window if self._uses_range_controller else None,
        )
        if cfg.distance_approach_enable and not self.distance_pi_enabled:
            logger.info("distance_control_mode matching_enabled=%s closure_independent=True "
                        "depth_ttl_ms=%.0f fresh_update_ms=180",
                        cfg.distance_approach_matching_enable, cfg.depth_longitudinal_sample_max_age_sec * 1000.)
        logger.info(
            "longitudinal_feedback_config kp_rpm_per_m=%.2f ego_limit_rpm=%.1f "
            "yaw_uncertainty_max_m_s=0.04 disagreement_skew_ms=50 depth_ttl_ms=%.0f "
            "fresh_update_ms=180 bias_trial_rpm=%.1f "
            "shared_motion_window=%s",
            cfg.distance_pid_kp_rpm_per_m, self._longitudinal_feedforward.config.max_abs_ego_rpm,
            cfg.depth_longitudinal_sample_max_age_sec * 1000.,
            cfg.distance_matching_test_bias_rpm,
            self._uses_range_controller,
        )
        self._last_visible_steer_action: Optional[ControlAction] = None
        self._last_visible_steer_started_at: Optional[float] = None
        self._visible_motion_target_id: Optional[int] = None
        self._visible_motion_samples: List[tuple] = []
        self._visible_motion_filtered_rate_ratio_s = 0.0
        self._target_stop_latched = False
        self._target_release_started_at: Optional[float] = None
        self._target_release_confirm_frames = 0
        self._target_stop_visual_track_id: Optional[int] = None
        self._target_stop_peak_bbox_area_ratio: Optional[float] = None
        self._reverse_active = False
        self._reverse_target_id: Optional[int] = None
        self._reverse_last_radar_distance_m: Optional[float] = None
        self._reverse_last_distance_at: Optional[float] = None
        self._reverse_filtered_approach_speed_m_s = 0.0
        self._reverse_last_base_rpm = 0
        self._reverse_last_feedforward_rpm = 0
        self._reverse_last_output_rpm = 0
        self._reverse_approach_confirm_frames = 0
        self._reverse_last_approach_at: Optional[float] = None
        self._reverse_release_confirm_frames = 0
        self._reverse_release_last_confirm_at: Optional[float] = None
        self._reverse_missing_started_at: Optional[float] = None
        self._reverse_visual_target_id: Optional[int] = None
        self._reverse_visual_area_ratio: Optional[float] = None
        self._reverse_visual_height_ratio: Optional[float] = None
        self._reverse_visual_sample_at: Optional[float] = None
        self._reverse_visual_frame_index = -1
        self._reverse_visual_guard_detail = "none"
        self._reverse_visual_candidate_detail = "none"
        self._reverse_visual_candidate_frames = 0
        self._reverse_visual_guard_logged = False
        self._reverse_visual_guard_blocked_logged = False
        self._near_distance_rotation_only_active = False
        self._near_distance_rotation_only_last_distance_m: Optional[float] = None
        self._near_settle_target_id: Optional[int] = None
        self._near_settle_confirm_frames = 0
        self._near_settle_release_frames = 0
        self._near_settle_until = 0.0
        self._longitudinal_missing_started_at: Optional[float] = None
        self._forward_active = False
        self._depth_quality_degraded = True
        self._depth_recovery_started_at: Optional[float] = None
        self._depth_recovery_anchor = None
        self._depth_schedule_recovery = None
        self._depth_last_approved_forward_rpm = None
        self._depth_recovery_pending_gap = False
        steering_pid_config = VisualSteeringPidConfig(
                enabled=bool(cfg.visible_steering_pid_enable),
                camera_hfov_deg=float(cfg.visible_steering_pid_camera_hfov_deg),
                camera_latency_sec=float(cfg.visible_steering_pid_camera_latency_sec),
                deadband_deg=float(cfg.visible_steering_pid_deadband_deg),
                outer_kp_per_sec=float(cfg.visible_steering_pid_outer_kp_per_sec),
                outer_kd_sec=float(cfg.visible_steering_pid_outer_kd_sec),
                target_rate_feedforward_gain=float(
                    cfg.visible_steering_pid_target_rate_feedforward_gain
                ),
                target_rate_feedforward_max_dps=float(
                    cfg.visible_steering_pid_target_rate_feedforward_max_dps
                ),
                target_speed_match_max_closing_dps=float(
                    cfg.visible_steering_pid_target_speed_match_max_closing_dps
                ),
                max_yaw_rate_dps=float(cfg.visible_steering_pid_max_yaw_rate_dps),
                rate_kp_rpm_per_dps=float(cfg.visible_steering_pid_rate_kp_rpm_per_dps),
                rate_ki_rpm_per_deg=float(cfg.visible_steering_pid_rate_ki_rpm_per_deg),
                rate_integral_limit_deg=float(cfg.visible_steering_pid_integral_limit_deg),
                max_correction_rpm=float(cfg.visible_steering_pid_max_correction_rpm),
                dynamic_small_error_deg=float(cfg.visible_steering_pid_dynamic_small_error_deg),
                dynamic_large_error_deg=float(cfg.visible_steering_pid_dynamic_large_error_deg),
                dynamic_small_max_yaw_rate_dps=float(
                    cfg.visible_steering_pid_dynamic_small_max_yaw_rate_dps
                ),
                dynamic_small_max_correction_rpm=float(
                    cfg.visible_steering_pid_dynamic_small_max_correction_rpm
                ),
                dynamic_large_error_base_cap_rpm=float(
                    cfg.visible_steering_pid_dynamic_large_error_base_cap_rpm
                ),
                opposite_yaw_brake_threshold_dps=float(
                    cfg.visible_steering_pid_opposite_yaw_brake_threshold_dps
                ),
                opposite_yaw_brake_boost_rpm=float(
                    cfg.visible_steering_pid_opposite_yaw_brake_boost_rpm
                ),
                braking_max_correction_rpm=float(
                    cfg.visible_steering_pid_braking_max_correction_rpm
                ),
                fast_countersteer_max_correction_rpm=float(
                    cfg.visible_steering_pid_fast_countersteer_max_correction_rpm
                ),
                fast_countersteer_gain_rpm_per_dps=float(
                    cfg.visible_steering_pid_fast_countersteer_gain_rpm_per_dps
                ),
                same_direction_overspeed_threshold_dps=float(
                    cfg.visible_steering_pid_same_direction_overspeed_threshold_dps
                ),
                same_direction_overspeed_brake_gain_rpm_per_dps=float(
                    cfg.visible_steering_pid_same_direction_overspeed_brake_gain_rpm_per_dps
                ),
                visual_direction_guard_enabled=bool(
                    cfg.visible_steering_pid_visual_direction_guard_enabled
                ),
                predictive_brake_decel_dps2=float(
                    cfg.visible_steering_pid_predictive_brake_decel_dps2
                ),
                predictive_brake_margin_deg=float(
                    cfg.visible_steering_pid_predictive_brake_margin_deg
                ),
                predictive_brake_response_sec=float(
                    cfg.visible_steering_pid_predictive_brake_response_sec
                ),
                min_effective_error_deg=float(
                    cfg.visible_steering_pid_min_effective_error_deg
                ),
                min_effective_correction_rpm=float(
                    cfg.visible_steering_pid_min_effective_correction_rpm
                ),
                mechanical_tier2_error_deg=float(
                    cfg.visible_steering_pid_mechanical_tier2_error_deg
                ),
                mechanical_tier2_correction_rpm=float(
                    cfg.visible_steering_pid_mechanical_tier2_correction_rpm
                ),
                mechanical_tier3_error_deg=float(
                    cfg.visible_steering_pid_mechanical_tier3_error_deg
                ),
                mechanical_tier3_correction_rpm=float(
                    cfg.visible_steering_pid_mechanical_tier3_correction_rpm
                ),
                mechanical_floor_release_ratio=float(
                    cfg.visible_steering_pid_mechanical_floor_release_ratio
                ),
                startup_kick_error_deg=float(
                    cfg.visible_steering_pid_startup_kick_error_deg
                ),
                startup_kick_rpm=float(
                    cfg.visible_steering_pid_startup_kick_rpm
                ),
                startup_kick_max_sec=float(
                    cfg.visible_steering_pid_startup_kick_max_sec
                ),
                startup_kick_release_yaw_rate_dps=float(
                    cfg.visible_steering_pid_startup_kick_release_yaw_rate_dps
                ),
                active_brake_yaw_threshold_dps=float(
                    cfg.visible_steering_pid_active_brake_yaw_threshold_dps
                ),
                active_brake_min_correction_rpm=float(
                    cfg.visible_steering_pid_active_brake_min_correction_rpm
                ),
                edge_boost_start_error_deg=float(
                    cfg.visible_steering_pid_edge_boost_start_error_deg
                ),
                aggressive_inner_wheel_margin_rpm=float(
                    cfg.visible_steering_pid_aggressive_inner_wheel_margin_rpm
                ),
                left_body_deg_per_encoder_deg=float(
                    cfg.visible_steering_pid_left_body_deg_per_encoder_deg
                ),
                right_body_deg_per_encoder_deg=float(
                    cfg.visible_steering_pid_right_body_deg_per_encoder_deg
                ),
                feedback_stale_sec=float(cfg.visible_steering_pid_feedback_stale_sec),
                error_filter_alpha=float(cfg.visible_steering_pid_error_filter_alpha),
                derivative_filter_alpha=float(
                    cfg.visible_steering_pid_derivative_filter_alpha
                ),
        )
        self._visual_steering_pid = VisualSteeringPid(steering_pid_config)
        self._parked_recenter_pid = VisualSteeringPid(steering_pid_config)
        self.last_steering_pid_result: Optional[VisualSteeringPidResult] = None

    def set_last_dispatched(self, action_kind: Optional[str]) -> None:
        self.last_dispatched_kind = action_kind

    def set_search_observation_hold(self, enabled: bool) -> None:
        """Pause unconfirmed search motion without claiming a visual target."""
        self._search_observation_hold = bool(enabled)

    def note_unknown_capture(self, capture_frame_id: int, timestamp: float, reason: str) -> None:
        if self.cfg.direction_history_enable:
            self._target_direction_history.record_unknown(capture_frame_id, timestamp, reason)

    def clear_historical_direction_hint(self, reason: str = "clear") -> None:
        if self._historical_direction_hint is not None:
            logger.info(
                "historical_direction_hint_clear reason=%s loss_capture=%s evidence_last=%s latest_visible=%s",
                str(reason), self._historical_direction_hint.get("loss_capture_frame_id"),
                self._historical_direction_hint.get("last_capture_frame_id"),
                self._direction_latest_visible_capture_id,
            )
        self._historical_direction_hint = None

    def _historical_hint_rejection(self, hint: dict) -> Optional[str]:
        if self.active_target_id is not None and int(hint.get("active_target_id", -1)) != int(self.active_target_id):
            return "active_uid_changed"
        if (
            self._direction_loss_capture_id is None
            or hint.get("loss_capture_frame_id") != self._direction_loss_capture_id
        ):
            return "loss_episode_changed"
        if self._direction_latest_visible_capture_id >= int(hint.get("last_capture_frame_id", 0)):
            return "newer_target_visible"
        stamp = float(hint.get("evidence_timestamp", 0.0))
        age = time.monotonic() - stamp
        if (
            not math.isfinite(age) or stamp <= 0.0
            or not 0.0 <= age <= max(0.10, float(self.cfg.historical_direction_backfill_max_age_sec))
        ):
            return "evidence_expired_or_future"
        return None

    def note_historical_direction_hint(
        self,
        direction: str,
        *,
        active_target_id: Optional[int],
        first_capture_frame_id: int,
        last_capture_frame_id: int,
        selected_capture_frame_ids: Tuple[int, ...],
        confidence: float,
        loss_capture_frame_id: int,
        evidence_timestamp: float,
        reason: str = "historical_direction_evidence",
    ) -> bool:
        """Store a non-target-owned hint for this specific loss episode.

        This method deliberately does not modify ``search_direction`` or emit
        an action.  The normal controller state machine consumes the hint only
        when its capture timeline has no trusted side. ``evidence_timestamp``
        is the oldest contributing capture's monotonic timestamp, so queueing
        or resubmission cannot renew the lifetime of the evidence chain.
        """
        side = str(direction or "").lower()
        if side not in ("left", "right"):
            return False
        if not bool(self.cfg.historical_direction_backfill_enable):
            return False
        if active_target_id is None:
            return False
        if self.active_target_id is not None and int(active_target_id) != int(self.active_target_id):
            return False
        if self.active_target_id is None and self.search_state not in (
            "none", "direction_unresolved", "searching", "timed_out"
        ):
            return False
        if self.active_target_id is None and not self._has_seen_person:
            return False
        ids = tuple(int(value) for value in selected_capture_frame_ids if int(value) > 0)
        if len(ids) < max(2, int(self.cfg.historical_direction_backfill_min_samples)):
            return False
        if (
            len(set(ids)) != len(ids)
            or min(ids) != int(first_capture_frame_id)
            or max(ids) != int(last_capture_frame_id)
            or max(ids) >= int(loss_capture_frame_id)
        ):
            return False
        hint = {
            "direction": side,
            "active_target_id": int(active_target_id),
            "first_capture_frame_id": int(first_capture_frame_id),
            "last_capture_frame_id": int(last_capture_frame_id),
            "selected_capture_frame_ids": ids,
            "confidence": max(0.0, min(float(self.cfg.historical_direction_backfill_confidence_cap), float(confidence))),
            "reason": str(reason),
            "loss_capture_frame_id": int(loss_capture_frame_id),
            "evidence_timestamp": float(evidence_timestamp),
        }
        rejection = self._historical_hint_rejection(hint)
        if rejection is not None:
            logger.info("historical_direction_hint_rejected reason=%s loss_capture=%s current_loss=%s evidence_last=%s latest_visible=%s",
                        rejection, loss_capture_frame_id, self._direction_loss_capture_id,
                        last_capture_frame_id, self._direction_latest_visible_capture_id)
            return False
        self._historical_direction_hint = hint
        logger.info(
            "historical_direction_hint_ready direction=%s active_uid=%d captures=%s confidence=%.2f reason=%s loss_capture=%d evidence_timestamp=%.6f",
            side,
            int(active_target_id),
            ",".join(str(value) for value in ids),
            float(self._historical_direction_hint["confidence"]),
            str(reason),
            int(loss_capture_frame_id), float(evidence_timestamp),
        )
        return True

    def _historical_hint_for_current_target(self) -> Optional[dict]:
        hint = self._historical_direction_hint
        if not isinstance(hint, dict):
            return None
        rejection = self._historical_hint_rejection(hint)
        if rejection is not None:
            self.clear_historical_direction_hint(rejection)
            return None
        direction = str(hint.get("direction", ""))
        if direction not in ("left", "right"):
            return None
        return hint

    def note_direction_classifier_evidence(
        self,
        capture_frame_id: int,
        timestamp: float,
        *,
        state: str,
        bbox: Optional[tuple] = None,
        frame_width: int = 0,
        confidence: float = 0.0,
        reason: str = "direction_classifier",
    ) -> None:
        """Merge asynchronous detector-only evidence into capture history.

        A classifier result is intentionally weaker than the main tracked
        target.  It may fill an audit slot, but it must not become a visible
        target vote: this worker has no ReID/DeepSORT identity and can select
        a bystander while the real target is briefly missing.  In particular,
        an unverified right-side box must not replace a trusted left-side exit
        direction and reverse the search motor.
        """
        if not self.cfg.direction_history_enable or int(capture_frame_id) <= 0:
            return
        existing = next(
            (
                item
                for item in self._target_direction_history.entries
                if int(item.capture_frame_id) == int(capture_frame_id)
            ),
            None,
        )
        if existing is not None and existing.state == "visible" and existing.reason != "direction_classifier":
            return
        normalized_state = str(state or "unknown").strip().lower()
        if normalized_state == "visible" and bbox is not None and int(frame_width) > 0:
            # Keep the capture slot ordered for diagnostics, but classify the
            # detector-only result as unknown.  Only the main control path,
            # which has the active UID and quality gates, is allowed to add a
            # ``visible`` entry used by latest_reliable_side()/resolve().
            self._target_direction_history.record_unknown(
                int(capture_frame_id),
                float(timestamp),
                "direction_classifier_unverified:%s" % str(
                    reason or "direction_classifier"
                ),
            )
        elif normalized_state == "missing":
            self._target_direction_history.record_missing(int(capture_frame_id), float(timestamp))
        else:
            self._target_direction_history.record_unknown(
                int(capture_frame_id), float(timestamp), str(reason or "direction_classifier_unknown")
            )

    def _record_target_direction_evidence(
        self,
        frame: SensorFrame,
        target: Optional[PersonTarget],
        *,
        reliable: bool,
    ) -> None:
        if not self.cfg.direction_history_enable or int(frame.capture_frame_id) <= 0:
            return
        if target is not None:
            # Only main-path, target-owned observations reach this method;
            # detector-only candidates cannot close a loss episode. Mapped
            # low-quality crops already qualify for direction geometry here.
            if int(frame.capture_frame_id) > self._direction_latest_visible_capture_id:
                self._direction_latest_visible_capture_id = int(frame.capture_frame_id)
                self._direction_loss_capture_id = None
                self.clear_historical_direction_hint("newer_target_visible")
            feedback = frame.steering_feedback
            self._target_direction_history.record_visible(
                frame.capture_frame_id,
                frame.capture_timestamp,
                target_id=int(target.track_id),
                bbox=target.bbox,
                frame_width=int(frame.width),
                confidence=float(target.confidence),
                vehicle_yaw_deg=(
                    None
                    if feedback is None
                    else float(feedback.integrated_yaw_right_deg)
                ),
                reason=(
                    "reliable_target"
                    if reliable
                    else "geometry_only_low_quality_target"
                ),
            )
        elif target is None and reliable:
            if self._direction_loss_capture_id is None:
                self._direction_loss_capture_id = int(frame.capture_frame_id)
            self._target_direction_history.record_missing(
                frame.capture_frame_id,
                frame.capture_timestamp,
            )
        else:
            self._target_direction_history.record_unknown(
                frame.capture_frame_id,
                frame.capture_timestamp,
                "low_quality_or_unsteerable_target",
            )

    @property
    def stale_direction_recovery_active(self) -> bool:
        return bool(self._stale_direction_recovery_active)

    def note_stale_visual_result(
        self,
        *,
        frame_width: int,
        now: Optional[float] = None,
        capture_frame_id: int = 0,
        capture_timestamp: float = 0.0,
    ) -> bool:
        """Invalidate an uncommitted loss direction after a stale vision gap."""
        stale_now = time.monotonic() if now is None else float(now)
        if self.cfg.direction_history_enable and int(capture_frame_id) > 0:
            self._target_direction_history.record_stale(
                int(capture_frame_id),
                float(capture_timestamp or stale_now),
            )
        if (
            not self.cfg.stale_direction_recovery_enable
            or not self._has_seen_person
            or self.active_target_id is None
            or self.search_state in ("searching", "timed_out")
        ):
            return False
        first_gap = not self._stale_direction_recovery_active
        if first_gap:
            history_hint = self._target_direction_history.resolve(
                required_missing_frames=1,
            ) if self.cfg.direction_history_enable else None
            if history_hint is not None and history_hint.direction in ("left", "right"):
                self._stale_direction_prior_hint = history_hint.direction
            elif self.last_person_center_x is not None and int(frame_width) > 0:
                self._stale_direction_prior_hint = (
                    "left"
                    if float(self.last_person_center_x) < 0.5 * float(frame_width)
                    else "right"
                )
            else:
                self._stale_direction_prior_hint = None
            logger.info(
                "stale_direction_recovery_start active_target=%s prior_hint=%s",
                int(self.active_target_id),
                self._stale_direction_prior_hint or "none",
            )
        self._stale_direction_recovery_started_at = stale_now
        self._stale_direction_observe_frames = 0
        self._stale_direction_probe_observe_frames = 0
        self._stale_candidate_center_ratio = None
        self._stale_candidate_side = None
        self._stale_candidate_center_source = "none"
        self._stale_candidate_center_anchor_heading_deg = None
        self._reset_search_timeout()
        self._stale_direction_recovery_active = True
        self._stale_direction_recovery_stage = "observe"
        # A stale result starts one bounded loss-confirmation window.  The
        # capture timeline, not an encoder sweep, owns direction selection.
        self.search_state = "none"
        self.search_direction = None
        self._lost_exit_direction = None
        self._lost_hint_confidence = 0.0
        self._lost_hint_source = "stale_result_gap"
        self.lost_confirm_frames = 0
        self._lost_started_at = None
        return first_gap

    def note_stale_candidate_evidence(
        self,
        bbox: tuple,
        *,
        frame_width: int,
        confirmed: bool,
        source: str,
        candidate_score: Optional[float] = None,
        candidate_tracked: bool = True,
        now: Optional[float] = None,
    ) -> bool:
        """Use confirmed detector evidence for bounded yaw without claiming identity."""
        return self._note_candidate_centering_evidence(
            bbox,
            frame_width=frame_width,
            confirmed=confirmed,
            source=source,
            candidate_score=candidate_score,
            candidate_tracked=candidate_tracked,
            now=now,
            allow_active_search=False,
        )

    def note_search_candidate_evidence(
        self,
        bbox: tuple,
        *,
        frame_width: int,
        confirmed: bool,
        source: str,
        candidate_score: Optional[float] = None,
        candidate_tracked: bool = True,
        candidate_identity_match: bool = False,
        capture_frame_id: int = 0,
        now: Optional[float] = None,
    ) -> bool:
        """Use bounded candidate evidence without giving up identity ownership.

        ``candidate_identity_match`` is a strong ReID hint for a fresh
        DeepSORT track that has not received the positive UID yet.  It can
        redirect a frozen search only after the normal candidate hold, while
        detector-only boxes remain unable to reverse the search direction.
        """
        return self._note_candidate_centering_evidence(
            bbox,
            frame_width=frame_width,
            confirmed=confirmed,
            source=source,
            candidate_score=candidate_score,
            candidate_tracked=candidate_tracked,
            candidate_identity_match=candidate_identity_match,
            capture_frame_id=capture_frame_id,
            now=now,
            allow_active_search=True,
        )

    def _note_candidate_centering_evidence(
        self,
        bbox: tuple,
        *,
        frame_width: int,
        confirmed: bool,
        source: str,
        now: Optional[float],
        allow_active_search: bool,
        candidate_score: Optional[float],
        candidate_tracked: bool,
        candidate_identity_match: bool = False,
        capture_frame_id: int = 0,
    ) -> bool:
        active_search = bool(
            allow_active_search
            and self.search_state == "searching"
            and self.search_direction in ("left", "right")
        )
        if (
            not self._stale_direction_recovery_active
            and not active_search
        ) or int(frame_width) <= 0:
            return False
        try:
            x1, _y1, x2, _y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(x1) and math.isfinite(x2)) or x2 <= x1:
            return False

        center_ratio = max(
            0.0,
            min(1.0, ((x1 + x2) * 0.5) / float(frame_width)),
        )
        already_centering = self._stale_direction_recovery_stage == "candidate_centering"
        score = None
        if candidate_score is not None:
            try:
                score = float(candidate_score)
            except (TypeError, ValueError):
                score = None
        source_name = str(source or "detector")
        # Class confidence says that this is probably a person, not that it is
        # the locked person. Opposite-side redirection therefore requires an
        # explicit match to the active target identity.
        direction_switch_quality = bool(candidate_tracked or candidate_identity_match)
        if (
            not confirmed
            and not already_centering
            and not (active_search and str(source) == "blocked")
        ):
            return False

        if not already_centering:
            left_ratio = max(0.0, min(1.0, float(self.cfg.center_left_ratio)))
            right_ratio = max(left_ratio, min(1.0, float(self.cfg.center_right_ratio)))
            if active_search and self.search_direction in ("left", "right"):
                observed_side = (
                    "left" if center_ratio < left_ratio
                    else "right" if center_ratio > right_ratio
                    else None
                )
                if observed_side is not None and observed_side != self.search_direction:
                    if not direction_switch_quality:
                        logger.info(
                            "search_candidate_direction_ignored observed=%s previous=%s "
                            "center=%.3f score=%s tracked=%s source=%s reason=weak_or_untracked",
                            observed_side,
                            self.search_direction,
                            center_ratio,
                            "none" if score is None else "%.3f" % score,
                            bool(candidate_tracked),
                            source_name,
                        )
                        return False
                    # An opposite-side box may reverse the sweep only after it
                    # is matched to the active target UID. Geometry alone is
                    # intentionally insufficient: another person can move
                    # smoothly through the same image region.
                    previous_direction = str(self.search_direction)
                    self._reset_stale_direction_recovery("opposite_search_candidate")
                    self.search_state = "searching"
                    self.search_direction = observed_side
                    self._lost_exit_direction = observed_side
                    self._lost_hint_confidence = 0.90
                    self._lost_hint_source = "search_candidate_opposite_side"
                    self._reset_search_timeout()
                    logger.info(
                        "search_candidate_direction_switch observed=%s previous=%s "
                        "center=%.3f score=%s tracked=%s identity_match=%s confirmations=1",
                        observed_side,
                        previous_direction,
                        center_ratio,
                        "none" if score is None else "%.3f" % score,
                        bool(candidate_tracked),
                        bool(candidate_identity_match),
                    )
                    # Keep the controller in the active search state. The
                    # caller can publish the new direction on this same tick.
                    return False
                else:
                    candidate_side = str(self.search_direction)
                # A centered candidate may pause/center the scan, but it cannot
                # erase a previously selected side.
                if not confirmed:
                    return False
                if str(source) == "low_quality_search":
                    # Edge/cropped geometry is directional evidence only. Keep
                    # the current sweep alive when it agrees with the side;
                    # an opposite side was handled above as an immediate flip.
                    return False
            elif center_ratio < left_ratio:
                candidate_side = "left"
            elif center_ratio > right_ratio:
                candidate_side = "right"
            else:
                candidate_side = "left" if center_ratio <= 0.5 else "right"
            self._stale_candidate_side = candidate_side
            if active_search and confirmed and candidate_side in ("left", "right"):
                # A fresh one-frame candidate is sufficient for directional
                # reacquisition. Do not route it through the old multi-frame
                # centering/probe path, which could leave the controller in
                # direction_probe with no motor direction.
                self._reset_stale_direction_recovery("candidate_direction_confirmed")
                self.search_state = "searching"
                self.search_direction = candidate_side
                self._lost_exit_direction = candidate_side
                self._lost_hint_confidence = 0.90
                self._lost_hint_source = "search_candidate_side"
                self._reset_search_timeout()
                logger.info(
                    "search_candidate_direction_promoted side=%s center=%.3f confirmations=1",
                    candidate_side,
                    center_ratio,
                )
                return False
        elif self._stale_candidate_side in ("left", "right"):
            candidate_side = str(self._stale_candidate_side)
            left_ratio = max(0.0, min(1.0, float(self.cfg.center_left_ratio)))
            right_ratio = max(left_ratio, min(1.0, float(self.cfg.center_right_ratio)))
            observed_side = (
                "left" if center_ratio < left_ratio
                else "right" if center_ratio > right_ratio
                else None
            )
            if observed_side is not None and observed_side != candidate_side:
                if not direction_switch_quality:
                    logger.info(
                        "candidate_centering_direction_ignored observed=%s previous=%s "
                        "center=%.3f score=%s tracked=%s source=%s reason=weak_or_untracked",
                        observed_side,
                        candidate_side,
                        center_ratio,
                        "none" if score is None else "%.3f" % score,
                        bool(candidate_tracked),
                        source_name,
                    )
                    return True
                # A single fresh opposite-side frame supersedes the stale
                # centering window and immediately resumes the new sweep.
                previous_side = str(candidate_side)
                self._reset_stale_direction_recovery("opposite_centering_candidate")
                self.search_state = "searching"
                self.search_direction = observed_side
                self._lost_exit_direction = observed_side
                self._lost_hint_confidence = 0.90
                self._lost_hint_source = "search_candidate_opposite_side"
                self._reset_search_timeout()
                logger.info(
                    "candidate_centering_direction_switch observed=%s previous=%s "
                    "center=%.3f score=%s tracked=%s confirmations=1",
                    observed_side,
                    previous_side,
                    center_ratio,
                    "none" if score is None else "%.3f" % score,
                    bool(candidate_tracked),
                )
                return True
        self._stale_candidate_center_ratio = center_ratio
        self._candidate_center_missing_frames = 0
        if str(source) != "blocked":
            self._stale_candidate_center_source = str(source or "detector")
        if already_centering:
            return True

        # A same-side candidate is already directional evidence.  The gate
        # supplies the bounded two-frame observation pause; after that pause
        # keep the frozen side and let normal target selection take over.
        if active_search:
            self._stale_direction_prior_hint = str(self.search_direction)
        else:
            self._stale_direction_prior_hint = str(candidate_side)
        candidate_source = str(self._stale_candidate_center_source or source or "detector")
        self._stale_candidate_center_ratio = center_ratio
        self._stale_candidate_side = candidate_side
        self._lost_exit_direction = candidate_side
        self.search_direction = candidate_side
        self.search_state = "searching"
        self._lost_hint_confidence = 0.75 if str(source) == "formal" else 0.50
        self._lost_hint_source = "search_candidate_%s" % candidate_source
        self._reset_search_timeout()
        self._reset_stale_direction_recovery("candidate_observation_complete")
        logger.info(
            "search_candidate_observation_complete context=%s source=%s center=%.3f "
            "direction=%s identity_claim=False observations=2",
            "search" if active_search else "stale_gap",
            candidate_source,
            center_ratio,
            candidate_side,
        )
        return False

    def note_search_candidate_missing(self) -> bool:
        """Bound a detector-only centering state when its candidate disappears.

        Returns True while the publisher should hold zero yaw. Once the short
        gap is confirmed, the frozen side from the first candidate-centering
        window becomes the search direction. A centered candidate still uses
        the prior search side when one exists; it must not create a new side.
        """
        if (
            not self._stale_direction_recovery_active
            or self._stale_direction_recovery_stage != "candidate_centering"
        ):
            return False
        self._candidate_center_missing_frames += 1
        required = max(1, int(self.cfg.lost_confirm_frames))
        if self._candidate_center_missing_frames < required:
            return True

        center_ratio = self._stale_candidate_center_ratio
        frozen_side = self._stale_candidate_side
        left_ratio = max(0.0, min(1.0, float(self.cfg.center_left_ratio)))
        right_ratio = max(left_ratio, min(1.0, float(self.cfg.center_right_ratio)))
        direction = frozen_side if frozen_side in ("left", "right") else None
        if direction is None:
            if center_ratio is not None and center_ratio < left_ratio:
                direction = "left"
            elif center_ratio is not None and center_ratio > right_ratio:
                direction = "right"

        if direction is None:
            self._stale_direction_recovery_stage = "unresolved"
            self.search_state = "direction_unresolved"
            self.search_direction = None
            self._lost_hint_confidence = 0.0
            self._lost_hint_source = "search_candidate_center_missing"
            logger.info(
                "candidate_centering_missing result=direction_unresolved "
                "last_center=%s frozen_side=%s missing=%d/%d",
                "none" if center_ratio is None else "%.3f" % center_ratio,
                frozen_side or "none",
                self._candidate_center_missing_frames,
                required,
            )
            return False

        context = self._candidate_center_reason_prefix
        self._reset_stale_direction_recovery("candidate_missing_latest_side")
        self.search_state = "none"
        self.search_direction = None
        self._lost_exit_direction = direction
        self._lost_hint_confidence = 0.75
        self._lost_hint_source = "search_candidate_last_%s" % direction
        self._reset_search_timeout()
        logger.info(
            "candidate_centering_missing result=resume_search direction=%s "
                "source=frozen_candidate_side context=%s missing=%d/%d",
            direction,
            context,
            required,
            required,
        )
        return False

    def _reset_stale_direction_recovery(self, reason: str = "reset") -> None:
        if self._stale_direction_recovery_active:
            logger.info(
                "stale_direction_recovery_end reason=%s stage=%s prior_hint=%s",
                str(reason),
                self._stale_direction_recovery_stage,
                self._stale_direction_prior_hint or "none",
            )
        self._stale_direction_recovery_active = False
        self._stale_direction_recovery_stage = "none"
        self._stale_direction_recovery_started_at = None
        self._stale_direction_prior_hint = None
        self._stale_direction_observe_frames = 0
        self._stale_direction_probe_observe_frames = 0
        self._stale_candidate_center_ratio = None
        self._stale_candidate_side = None
        self._stale_candidate_center_source = "none"
        self._stale_candidate_center_anchor_heading_deg = None
        self._candidate_center_reason_prefix = "stale_candidate_center"
        self._candidate_center_missing_frames = 0

    def search_status(self, now: Optional[float] = None) -> SearchControlStatus:
        """Return an immutable search snapshot without exposing controller internals."""
        current = time.monotonic() if now is None else float(now)
        selected_id = (
            None
            if self.last_selected_target is None
            else int(self.last_selected_target.track_id)
        )
        elapsed = (
            None
            if self._search_rotation_started_at is None
            else max(0.0, current - float(self._search_rotation_started_at))
        )
        coverage_deg = max(
            0.0,
            float(self._search_heading_max_deg) - float(self._search_heading_min_deg),
        )
        use_coverage = bool(self._search_rotation_feedback_seen)
        search_active = self.search_state in (
            "searching",
            "timed_out",
            "direction_unresolved",
        )
        if self.search_state == "direction_unresolved":
            stage = "direction_unresolved"
        else:
            stage = "single_direction" if search_active else "inactive"
        return SearchControlStatus(
            state=str(self.search_state or "none"),
            direction=self.search_direction,
            active_target_id=(
                None if self.active_target_id is None else int(self.active_target_id)
            ),
            selected_target_id=selected_id,
            progress_deg=(
                float(coverage_deg)
                if use_coverage
                else float(self._search_rotation_accumulated_deg)
            ),
            target_deg=max(90.0, float(self.cfg.search_revolution_deg)),
            elapsed_sec=elapsed,
            stage=stage,
            heading_from_loss_deg=float(self._search_heading_from_loss_deg),
            coverage_deg=float(coverage_deg),
            travel_deg=float(self._search_rotation_accumulated_deg),
            hint_confidence=float(self._lost_hint_confidence),
            hint_source=str(self._lost_hint_source),
        )

    @property
    def target_stop_latched(self) -> bool:
        """Whether distance safety currently forbids all motion release."""
        return bool(self._target_stop_latched)

    def release_search_on_confirmed_target(self, reason: str = "strong_reid") -> None:
        """Release a frozen search immediately after a strong UID match.

        This does not assign an identity or choose a target; the caller must
        already have passed the formal ReID/selection gate. It only clears the
        stale search-direction state so a <=0.15 ReID match can resume control
        without waiting for the ordinary reacquisition streak.
        """
        if self.search_state != "none" or self.search_direction is not None:
            logger.info(
                "search_direction_released_strong_reid uid=%s reason=%s prior_state=%s prior_direction=%s",
                "none" if self.active_target_id is None else int(self.active_target_id),
                str(reason),
                str(self.search_state),
                str(self.search_direction or "none"),
            )
        self.search_state = "none"
        self.search_direction = None
        self._lost_started_at = None
        self._lost_exit_direction = None
        self.lost_confirm_frames = 0
        self._search_observation_hold = False
        self.clear_historical_direction_hint("confirmed_target")
        self._direction_loss_capture_id = None
        self._reset_stale_direction_recovery("strong_reid")
        self._reset_search_timeout()

    def clear_active_target(self, reason: str = "manual") -> None:
        old_target_id = self.active_target_id
        self.active_target_id = None
        self.last_selected_target = None
        self._has_seen_person = False
        self.search_state = "none"
        self.search_direction = None
        self._search_observation_hold = False
        self.lost_confirm_frames = 0
        self._lost_started_at = None
        self._lost_exit_direction = None
        self.clear_historical_direction_hint("active_target_cleared")
        self._direction_loss_capture_id = None
        self._reset_stale_direction_recovery("clear_active_target")
        self._startup_search_started_at = time.monotonic()
        self._reset_search_timeout()
        self._reset_initial_target_confirm()
        self._depth_schedule_recovery = None
        self._depth_gap_resume_hint = None
        self._depth_last_approved_forward_rpm = None
        self._last_target_distance_m = None
        self._last_target_distance_at = None
        self._last_mmwave_motion_speed_percent = None
        self._mmwave_speed_recovery_active = False
        self._distance_pid.reset()
        self.last_distance_pid_result = None
        self._distance_pid_last_input_m = None
        self._distance_pid_last_update_at = None
        self._forward_active = False
        self._depth_quality_degraded = True
        self._depth_recovery_started_at = None
        self._depth_recovery_anchor = None
        self._depth_recovery_pending_gap = False
        self._reset_visible_steer_memory()
        self._reset_visible_motion()
        self._target_stop_latched = False
        self._target_release_started_at = None
        self._target_release_confirm_frames = 0
        self._clear_target_stop_visual_reference()
        self._reset_reverse_control("clear_active_target")
        self._near_distance_rotation_only_active = False
        self._near_distance_rotation_only_last_distance_m = None
        self._reset_near_settle()
        self._visual_steering_pid.reset()
        self._parked_recenter_pid.reset()
        self.last_steering_pid_result = None
        logger.info(
            "active_target_cleared reason=%s old_target=%s",
            reason,
            "none" if old_target_id is None else int(old_target_id),
        )

    def _reset_initial_target_confirm(self) -> None:
        self._initial_candidate_id = None
        self._initial_candidate_frames = 0

    def _reset_reverse_control(self, reason: str, *, keep_last_distance: bool = False) -> None:
        was_active = bool(self._reverse_active)
        self._reverse_active = False
        self._reverse_target_id = None
        self._reverse_approach_confirm_frames = 0
        self._reverse_last_approach_at = None
        self._reverse_release_confirm_frames = 0
        self._reverse_release_last_confirm_at = None
        self._reverse_missing_started_at = None
        if not keep_last_distance:
            self._reverse_last_radar_distance_m = None
            self._reverse_last_distance_at = None
            self._reverse_filtered_approach_speed_m_s = 0.0
        self._reverse_last_base_rpm = 0
        self._reverse_last_feedforward_rpm = 0
        self._reverse_last_output_rpm = 0
        if was_active:
            logger.info("target_reverse_stopped reason=%s", reason)

    def _stable_reverse_radar_distance(self, frame: SensorFrame) -> Optional[float]:
        """Return a fresh target-associated distance that is safe to reverse from."""
        if not self.cfg.reverse_enable or frame.distance_m is None:
            return None
        state = frame.distance_state
        source = str(getattr(state, "source", ""))
        detail = str(getattr(state, "source_detail", ""))
        if source == "vision_depth":
            # Depth is sampled inside the currently locked ReID person's torso
            # box. Held, stale, and jump-pending values cannot initiate reverse.
            if (
                not detail.startswith("depth_")
                or detail.endswith("_hold")
            ) or getattr(state, "raw_distance_m", None) is None:
                return None
        elif source == "vision_mmwave":
            if str(getattr(state, "fusion_mode", "")) != "radar":
                return None
            if detail not in {"matched", "matched_after_continuity_confirm"}:
                return None
        else:
            return None
        sample_age = getattr(state, "sample_age_sec", None)
        if sample_age is None or float(sample_age) > max(0.01, float(self.cfg.reverse_radar_max_age_sec)):
            return None
        return float(frame.distance_m)

    @property
    def distance_pi_enabled(self) -> bool:
        return bool(self.cfg.distance_pid_enable and self.cfg.distance_control_mode == "distance_pi")

    @property
    def _uses_range_controller(self) -> bool:
        return self.distance_pi_enabled or self.cfg.distance_approach_enable

    def _reset_longitudinal_motion(self) -> None:
        if self.distance_pi_enabled:
            self._reset_distance_pid()
            self._distance_pi_ego_forward_rpm = None
            self._distance_pi_feedback_timestamp = None
        self._raw_closing_window.reset()
        self._closure_rejected_observation = None
        self._depth_gap_resume_hint = None
        self._depth_recovery_resume_base_rpm = 0.0
        self._depth_schedule_recovery = None
        self._depth_last_approved_forward_rpm = None
        self._longitudinal_feedforward.reset()
        self._longitudinal_bridge.reset()
        self._longitudinal_bridge_output_cap = None
        self._longitudinal_motion_evidence = None
        self._longitudinal_motion_uid = None
        self._longitudinal_motion_stamp = None
        self._distance_pid_sample_timestamp = None

    def _older_depth_observation(self, frame: SensorFrame) -> bool:
        stamp = getattr(frame.distance_state, "sample_timestamp", None)
        previous = self._distance_pid_last_sample_timestamp
        return bool(
            str(getattr(frame.distance_state, "source", "")) == "vision_depth"
            and isinstance(stamp, (int, float)) and math.isfinite(stamp)
            and previous is not None and stamp < previous
        )

    def _clear_longitudinal_velocity_evidence(self, frame: SensorFrame, reason: str,
                                             *, yaw=None, bearing=None) -> None:
        evidence = self._longitudinal_motion_evidence
        if evidence is not None or getattr(self, "_last_velocity_reject_reason", None) != reason:
            logger.info(
                "longitudinal_motion_reset capture_frame_id=%s uid=%s reason=%s "
                "sample_ts=%s previous_ts=%s samples=%s yaw_dps=%s bearing_deg=%s "
                "depth_detail=%s",
                frame.capture_frame_id, self.active_target_id, reason,
                frame.distance_state.sample_timestamp, self._longitudinal_motion_stamp,
                None if evidence is None else evidence.sample_count, yaw, bearing,
                frame.distance_state.source_detail,
            )
        self._last_velocity_reject_reason = reason
        self._longitudinal_feedforward.reset()
        self._longitudinal_bridge.reset()
        self._longitudinal_bridge_output_cap = None
        self._longitudinal_motion_evidence = None
        self._longitudinal_motion_stamp = None

    @property
    def longitudinal_prior_max_age_sec(self):
        # Estimate memory only. Physical Depth/motor authority remains 180ms.
        return .35 if self.cfg.distance_matching_base_max_rpm > 0 else .18

    def _bridge_longitudinal_motion(self, frame, uid, now, stamp, yaw, bearing, reason):
        """Fresh depth, bounded prior speed; never renew the prior's clock.

        The prior cannot increase its contribution. The approach profile may
        independently increase distance correction from a NEW accepted Depth.
        """
        if self.distance_pi_enabled or (self.cfg.distance_approach_enable and not self.cfg.distance_approach_matching_enable):
            return False
        if (frame.distance_m is None or frame.distance_state.safety_distance_m is not None
                or frame.distance_m < max(self.cfg.brake_distance_m, self.cfg.target_distance_m - .03)):
            return False
        # A prior may span estimator rebuilds, not restart expired translation.
        # Runtime independently requires a still-live previous motor grant.
        origin = self._longitudinal_bridge.origin
        previous_stamp = self._longitudinal_motion_stamp
        if previous_stamp is None and origin is not None:
            previous_stamp = origin.sample_timestamp
        if (previous_stamp is None
                or not 0 <= stamp-previous_stamp <= .18
                or frame.steering_feedback is None
                or not all(math.isfinite(v) and v >= 0 for v in (
                    frame.steering_feedback.left_forward_rpm,
                    frame.steering_feedback.right_forward_rpm))):
            return False
        cap = max(0.0, self._distance_pid._last_output_rpm or 0.0)
        evidence = self._longitudinal_bridge.evaluate(
            now=now, stamp=stamp, uid=uid, distance=frame.distance_state.raw_distance_m,
            yaw=yaw, bearing=bearing, previous_output=cap, baseline=self.cfg.forward_min_rpm,
            fall_rate_rpm_per_sec=(80.0 if self.cfg.distance_matching_base_max_rpm > 0 else None),
            near_distance=self.cfg.target_distance_m + .20,
            prior_max_age_sec=self.longitudinal_prior_max_age_sec,
            ego_speed=(.5 * (frame.steering_feedback.left_forward_rpm +
                            frame.steering_feedback.right_forward_rpm) *
                       self.cfg.distance_feedforward_wheel_circumference_m / 60.
                       if frame.steering_feedback is not None else None),
        )
        if evidence is None:
            return False
        self._longitudinal_motion_evidence = evidence
        self._longitudinal_motion_stamp = stamp
        self._longitudinal_bridge_output_cap = cap
        logger.info(
            "longitudinal_motion_bridge capture_frame_id=%s uid=%s reason=%s sample_ts=%s "
            "origin_ts=%s remaining_ms=%.1f tracking_base=%.2frpm output_cap_rpm=%.2f "
            "acceleration_allowed=False origin_renewed=False decay_policy=%s chain_reset_reason=%s "
            "cap_scope=%s fresh_distance_acceleration_allowed=%s",
            frame.capture_frame_id, uid, reason, stamp,
            self._longitudinal_bridge.origin.sample_timestamp,
            1000.0 * (self.longitudinal_prior_max_age_sec - (now - self._longitudinal_bridge.origin.sample_timestamp)),
            evidence.target_rpm, cap,
            "bounded_80rpm_s" if self.cfg.distance_matching_base_max_rpm > 0 else "legacy_to_baseline",
            self._longitudinal_feedforward.last_result.chain_reset_reason if reason == "warming_up" else None,
            "prior_only" if self.cfg.distance_approach_enable else "whole_output",
            self.cfg.distance_approach_enable,
        )
        return True

    def _low_confidence_replay_can_retain_velocity(self, frame: SensorFrame, previous) -> bool:
        """Only the known duplicate-to-hold downgrade may retain old evidence.

        This is NOT a fresh-distance check. Caller still checks UID, deadlines,
        near/hazard/encoder/turn state and must not update PID or motor authority.
        """
        s = frame.distance_state
        confidence = s.fusion_confidence
        return bool(
            s.sample_timestamp is None and s.is_replay_of(previous)
            and ((s.temporal_status == "duplicate" and s.observation_timestamp == previous)
                 or (self.distance_pi_enabled and s.temporal_status == "older_than_anchor"
                     and s.observation_timestamp <= previous))
            and s.source_detail == "depth_sample_observation_discarded_fused_radar_hold_hold"
            and s.fusion_mode == "depth_radar_hold"
            and isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            and math.isfinite(confidence) and .45 <= confidence < .50
            and s.used_distance_m is not None and math.isfinite(s.used_distance_m)
            and not self._distance_longitudinally_untrusted(frame, check_hold_confidence=False)
        )

    def _observe_distance_closure(self, frame, target, uid, stamp, now, trusted):
        """Distance/braking evidence independent of target-speed eligibility."""
        w, fb = self._raw_closing_window, frame.steering_feedback
        def reject_geometry():
            w.reset()
            self._distance_pid.invalidate_motion_memory()
        if not trusted:
            if (frame.distance_state.raw_distance_m is not None
                    or (w.samples and now-w.samples[-1][0] > w.max_gap_sec)):
                reject_geometry()
            return
        if self._closure_rejected_observation == (uid, stamp):
            # Multiple callers cannot turn the same rejected raw jump into
            # a fresh first sample after the window's outlier reset.
            self._braking_range_rate = min(self._braking_range_rate, -3.)
            self._braking_rate_source = 'raw_jump_protection'
            return
        if (fb is None or not fb.trustworthy or not math.isfinite(fb.timestamp)
                or not 0 <= now-fb.timestamp <= .15 or abs(fb.timestamp-stamp) > .15
                or not all(math.isfinite(v) and abs(v) <= self._longitudinal_feedforward.config.max_abs_ego_rpm
                           for v in (fb.left_forward_rpm,fb.right_forward_rpm))):
            reject_geometry()
            return
        yaw_values = [fb.yaw_rate_right_dps]
        if fb.raw_yaw_rate_right_dps is not None:
            yaw_values.append(fb.raw_yaw_rate_right_dps)
        if not all(math.isfinite(v) for v in yaw_values) or frame.width <= 0:
            reject_geometry()
            return
        yaw = max(yaw_values,key=abs)
        obs = getattr(target,'depth_observation',None)
        age = 0.
        cx = target.center[0]
        if obs is not None:
            if obs.target_id != uid or obs.source != 'yolo_detector':
                reject_geometry()
                return
            age = stamp-obs.capture_timestamp
            cx = .5*(obs.bbox[0]+obs.bbox[2])
        elif abs(yaw) > 5:
            reject_geometry()
            return
        fx = .5*frame.width/math.tan(math.radians(self.cfg.visible_steering_pid_camera_hfov_deg)*.5)
        bearing = math.degrees(math.atan((cx-.5*frame.width)/fx))
        raw = frame.distance_state.raw_distance_m
        bound = None if raw is None else closure_rotation_bound(
            depth=raw,bearing_deg=bearing,yaw_dps=yaw,geometry_age=age)
        if bound is None:
            reject_geometry()
            return
        is_new = not w.samples or stamp > w.samples[-1][0]
        rate = w.update(uid=uid,stamp=stamp,raw=raw,rotation=bound)
        if w.samples and w.samples[-1][0] == stamp and rate is not None:
            # Subtract a positive rotation upper bound: conservative closure,
            # not a claim about target world speed or a new identity.
            self._braking_range_rate = rate
            self._braking_rate_source = 'raw_depth_window'
        elif w.status == 'raw_rate_out_of_bounds':
            self._distance_pid.invalidate_motion_memory()
            self._closure_rejected_observation = (uid, stamp)
            self._braking_range_rate = min(self._braking_range_rate,-3.)
            self._braking_rate_source = 'raw_jump_protection'
        # Only accepted geometry+fresh encoder and a genuinely warming window
        # may use bounded memory. Faults, yaw/geometry rejection, raw jumps,
        # old depth and untrusted observations never qualify.
        self._distance_pi_motion_memory_allowed = bool(
            w.status == 'warming_up' and w.samples and w.samples[-1][0] == stamp)
        if is_new:
            logger.info(
                'distance_closure capture_frame_id=%s uid=%s sample_ts=%s source=%s samples=%s '
                'span_ms=%.1f rate=%s rotation_bound=%s bearing=%s max_gap_ms=%.0f ff_required=False deadline_renewed=False',
                frame.capture_frame_id,uid,stamp,self._braking_rate_source,len(w.samples),
                w.span*1000,rate,bound,bearing,w.max_gap_sec*1000.)

    def _observe_longitudinal_motion(
        self, frame: SensorFrame, target: Optional[PersonTarget], *, allowed: bool = True
    ) -> None:
        """Consume physical observations, never repeated processing ticks."""
        now = time.monotonic()
        self._distance_approach_sample_trusted = False
        self._distance_pi_motion_memory_allowed = False
        self._distance_pi_raw_distance_m = None
        stamp = getattr(frame.distance_state, "sample_timestamp", None)
        try:
            stamp = float(stamp)
            if not math.isfinite(stamp) or stamp <= 0.0:
                stamp = None
        except (TypeError, ValueError, OverflowError):
            stamp = None
        self._distance_pid_sample_timestamp = stamp
        uid = None if target is None else int(target.track_id)
        valid_target = bool(
            allowed and target is not None and uid == self.active_target_id
            and self._has_seen_person and self.search_state == "none"
            and not frame.hazard.active and not any((
                frame.obstacles.front, frame.obstacles.left, frame.obstacles.right
            ))
        )
        if not valid_target:
            self._depth_quality_degraded = True
            self._depth_recovery_started_at = None
            self._depth_recovery_anchor = None
            self._depth_recovery_pending_gap = False
            self._clear_longitudinal_velocity_evidence(frame, "target_or_safety_rejected")
            self._reset_longitudinal_motion()
            return
        if self._longitudinal_motion_uid != uid:
            self._reset_longitudinal_motion()
            self._reset_distance_pid()
            self._longitudinal_motion_uid = uid
            self._distance_pid_sample_timestamp = stamp
        trusted = bool(
            self._is_fresh_depth_state(frame) and stamp is not None
            and 0.0 <= now - stamp <= 0.18
            and frame.distance_m is not None
            and not self._distance_longitudinally_untrusted(frame)
            and not getattr(frame.distance_state, "brake_latched", False)
        )
        self._distance_approach_sample_trusted = trusted
        self._distance_pi_raw_distance_m = frame.distance_state.raw_distance_m if trusted else None
        self._distance_pi_ego_forward_rpm = None
        self._distance_pi_feedback_timestamp = None
        pi_feedback = frame.steering_feedback
        if (self.distance_pi_enabled and trusted and pi_feedback is not None
                and pi_feedback.trustworthy and math.isfinite(pi_feedback.timestamp)
                and 0 <= now - pi_feedback.timestamp <= .15
                and abs(pi_feedback.timestamp - stamp) <= .15
                and all(math.isfinite(v) and abs(v) <= self._longitudinal_feedforward.config.max_abs_ego_rpm
                        for v in (pi_feedback.left_forward_rpm, pi_feedback.right_forward_rpm))):
            self._distance_pi_ego_forward_rpm = .5 * (
                pi_feedback.left_forward_rpm + pi_feedback.right_forward_rpm)
            self._distance_pi_feedback_timestamp = pi_feedback.timestamp
        # Fail closed for the new higher no-matching budget until raw closure
        # has been verified. This fallback assumes closure at measured car
        # speed, not that an unavailable human-speed estimate equals zero.
        self._braking_rate_source = "encoder_fallback"
        fb = frame.steering_feedback
        fallback_rpm = float(self.cfg.forward_max_rpm)
        if fb is not None and self._uses_range_controller:
            prior = self.last_distance_pid_result
            fallback_rpm, self._braking_rate_source = bounded_encoder_fallback(
                now=now,stamp=fb.timestamp,left=fb.left_forward_rpm,right=fb.right_forward_rpm,
                trustworthy=fb.trustworthy,max_rpm=float(self.cfg.forward_max_rpm),
                feedback_limit=self._longitudinal_feedforward.config.max_abs_ego_rpm,
                last_request=0. if prior is None else float(prior.output_rpm),
                rise_rpm_s=float(self.cfg.distance_pid_output_rise_rpm_per_sec))
            if self._braking_rate_source == 'encoder_age_bound':
                logger.info('distance_brake_fallback capture_frame_id=%s uid=%s sample_ts=%s '
                            'feedback_age_ms=%.1f bound_rpm=%.1f source=encoder_age_bound deadline_renewed=False',
                            frame.capture_frame_id,uid,stamp,(now-fb.timestamp)*1000,fallback_rpm)
        elif (fb is not None and fb.trustworthy and math.isfinite(fb.timestamp)
                and 0 <= now-fb.timestamp <= .15
                and all(math.isfinite(v) for v in (fb.left_forward_rpm, fb.right_forward_rpm))):
            fallback_rpm = max(0., .5*(fb.left_forward_rpm+fb.right_forward_rpm))
        self._braking_range_rate = -fallback_rpm*self.cfg.distance_feedforward_wheel_circumference_m/60.
        if self._uses_range_controller:
            self._observe_distance_closure(frame,target,uid,stamp,now,trusted)
        if not self.cfg.distance_feedforward_enable:
            return
        feedback = frame.steering_feedback
        cx = target.center[0]
        bearing = (cx / frame.width - 0.5) * self.cfg.visible_steering_pid_camera_hfov_deg if frame.width > 0 else 90.0
        yaw = None if feedback is None else (
            feedback.yaw_rate_right_dps if feedback.raw_yaw_rate_right_dps is None
            else max((feedback.yaw_rate_right_dps, feedback.raw_yaw_rate_right_dps), key=abs)
        )
        # A failed ROI attempt supplies no observation. Retain the old chain
        # for the NEXT sample, without supplying a PID timestamp or renewing
        # either the physical-depth or feedforward deadline.
        previous = self._longitudinal_motion_stamp
        state = frame.distance_state
        origin = self._longitudinal_bridge.origin
        scheduling_only = (
            state.source_detail == "depth_detector_bbox_stale"
            or (previous is not None and state.is_replay_of(previous))
        )
        low_confidence_replay = self._low_confidence_replay_can_retain_velocity(frame, previous)
        if (stamp is None and state.sample_timestamp is None and state.raw_distance_m is None
                and state.source == "vision_depth" and scheduling_only
                and (not self._distance_longitudinally_untrusted(frame) or low_confidence_replay)
                and previous is not None and 0 <= now - previous <= self.longitudinal_prior_max_age_sec
                and (origin is None or 0 <= now - origin.sample_timestamp < self.longitudinal_prior_max_age_sec)
                and state.safety_distance_m is None and not state.brake_latched
                and not self._target_stop_latched
                and (frame.distance_m is None or frame.distance_m >= max(
                    self.cfg.brake_distance_m, self.cfg.target_distance_m - .03))
                and feedback is not None and feedback.trustworthy
                and math.isfinite(feedback.timestamp) and 0 <= now-feedback.timestamp <= .15
                and yaw is not None and math.isfinite(yaw) and abs(yaw) <= 15
                and math.isfinite(bearing) and abs(bearing) <= 10
                and all(math.isfinite(v) and 0 <= v <= self._longitudinal_feedforward.config.max_abs_ego_rpm for v in (
                    feedback.left_forward_rpm, feedback.right_forward_rpm))):
            # No measurement now. A previous compensated baseline survives
            # only a short stable turn; the next endpoint must verify it again.
            stable_compensated_gap = (
                (self.cfg.distance_turn_compensation_enable or self._uses_range_controller)
                and self._longitudinal_feedforward.preserve_compensated_gap(now=now, yaw=yaw)
            )
            reset_derivative = not stable_compensated_gap and (
                abs(yaw) > 5 or self._longitudinal_feedforward.has_compensated_baseline
                or now - previous > .18
            )
            if reset_derivative:
                self._longitudinal_feedforward.reset()
            logger.info(
                "longitudinal_motion_observation_skipped capture_frame_id=%s uid=%s "
                "reason=%s original_sample_ts=%s remaining_ms=%.1f "
                "derivative_reset=%s motion_authorized=False deadline_renewed=False "
                "compensated_gap_preserved=%s prior_remaining_ms=%.1f",
                frame.capture_frame_id, uid, state.source_detail, previous,
                max(0., 1000 * (.18 - (now - previous))), reset_derivative, stable_compensated_gap,
                max(0., 1000 * (self.longitudinal_prior_max_age_sec - (now - (
                    origin.sample_timestamp if origin is not None else previous)))),
            )
            if low_confidence_replay:
                logger.info(
                    "longitudinal_replay_retained capture_frame_id=%s uid=%s "
                    "observation_ts=%s origin_ts=%s remaining_ms=%.1f hold_confidence=%s "
                    "derivative_reset=%s pid_updated=False motion_authorized=False deadline_renewed=False",
                    frame.capture_frame_id, uid, state.observation_timestamp, previous,
                    1000*(.18-(now-previous)), state.fusion_confidence, reset_derivative,
                )
            return
        compensation = None
        compensation_status = "unavailable"
        observation = getattr(target, "depth_observation", None)
        if (self.cfg.distance_turn_compensation_enable and trusted and feedback is not None
                and observation is not None and observation.target_id == uid
                and observation.source == "yolo_detector"):
            geometry = dict(
                depth=frame.distance_state.raw_distance_m, bbox=observation.bbox,
                width=frame.width, hfov_deg=self.cfg.visible_steering_pid_camera_hfov_deg,
                capture_stamp=observation.capture_timestamp, depth_stamp=stamp,
                low_yaw_max_age_sec=.25,
            )
            if feedback.yaw_rate_right_dps * yaw >= 0:
                compensation = depth_rotation_rate(**geometry, yaw=yaw)
                compensation_status = "aligned" if compensation is not None else "geometry_rejected"
            else:
                compensation_status = "yaw_disagreement_rejected"
                if (feedback.trustworthy and math.isfinite(feedback.timestamp)
                        and 0 <= now-feedback.timestamp <= .15
                        and abs(feedback.timestamp-stamp) <= .05
                        and frame.distance_m > self.cfg.target_distance_m + .20
                        and frame.distance_state.raw_distance_m > self.cfg.target_distance_m + .20):
                    compensation = bounded_disagreeing_yaw_rotation(
                        **geometry, raw_yaw=feedback.raw_yaw_rate_right_dps,
                        filtered_yaw=feedback.yaw_rate_right_dps,
                    )
                    if compensation is not None:
                        compensation_status = "yaw_disagreement_bounded"
        if trusted and feedback is not None and (previous is None or stamp > previous):
            logger.info(
                "longitudinal_feedback_audit capture_frame_id=%s uid=%s sample_ts=%s "
                "compensation=%s raw_yaw=%s filtered_yaw=%s feedback_depth_skew_ms=%.1f "
                "rotation_rate=%s left_rpm=%s right_rpm=%s ego_limit_rpm=%.1f",
                frame.capture_frame_id, uid, stamp, compensation_status,
                feedback.raw_yaw_rate_right_dps, feedback.yaw_rate_right_dps,
                1000*(feedback.timestamp-stamp), None if compensation is None else compensation[0],
                feedback.left_forward_rpm, feedback.right_forward_rpm,
                self._longitudinal_feedforward.config.max_abs_ego_rpm,
            )
        yaw_limit = 15.0 if compensation is not None else 5.0
        if compensation is not None:
            bearing = compensation[1]
        feedback_ok = bool(
            feedback is not None and feedback.trustworthy
            and math.isfinite(feedback.timestamp) and 0.0 <= now - feedback.timestamp <= 0.15
            and math.isfinite(yaw) and abs(yaw) <= yaw_limit and abs(bearing) <= 10.0
        )
        if not feedback_ok:
            reason = (
                "encoder_missing" if feedback is None else
                "encoder_untrusted" if not feedback.trustworthy else
                "encoder_timestamp" if not math.isfinite(feedback.timestamp) or not 0 <= now-feedback.timestamp <= .15 else
                "yaw_limit" if yaw is None or not math.isfinite(yaw) or abs(yaw) > yaw_limit else
                "bearing_limit"
            )
            origin = self._longitudinal_bridge.origin
            if (reason == "yaw_limit" and origin is not None and abs(yaw) <= 15
                    and abs(bearing) <= 10 and 0 <= now - origin.sample_timestamp < .18
                    and not self._distance_longitudinally_untrusted(frame)
                    and stamp is None and frame.distance_state.is_replay_of(self._longitudinal_motion_stamp)):
                # A rejected duplicate cannot use FF now (stamp stays None),
                # but need not destroy the bounded prior for the next new Depth.
                self._longitudinal_feedforward.reset()
                return
            if (reason == "yaw_limit" and trusted
                    and feedback is not None and abs(feedback.timestamp - stamp) <= .15
                    and all(math.isfinite(v) and abs(v) <= self._longitudinal_feedforward.config.max_abs_ego_rpm for v in (
                        feedback.left_forward_rpm, feedback.right_forward_rpm))
                    and self._bridge_longitudinal_motion(frame, uid, now, stamp, yaw, bearing, reason)):
                # Do not differentiate across an uncompensated turn. The old
                # speed's fixed deadline survives, but the estimator restarts.
                self._longitudinal_feedforward.reset()
                return
            self._clear_longitudinal_velocity_evidence(frame, reason, yaw=yaw, bearing=bearing)
            return
        if (
            frame.distance_state.is_replay_of(self._longitudinal_motion_stamp)
            and not self._distance_longitudinally_untrusted(frame)
            and self._longitudinal_motion_stamp is not None
            and 0.0 <= now - self._longitudinal_motion_stamp <= 0.18
        ):
            # Keep evidence for the NEXT distinct measurement. sample_timestamp
            # stays None, so a hold cannot use this estimate to accelerate PID.
            return
        if not trusted:
            reason = ("depth_timestamp_missing" if stamp is None else
                      "depth_expired_or_future" if not 0 <= now-stamp <= .18 else
                      "depth_rejected")
            self._clear_longitudinal_velocity_evidence(frame, reason, yaw=yaw, bearing=bearing)
            return
        # A changed encoder safety condition above is never hidden by the
        # same-sample shortcut. Historical alignment cannot advance evidence.
        if self._longitudinal_motion_stamp is not None and stamp <= self._longitudinal_motion_stamp:
            # Re-reading the SAME accepted sample may reuse its closure, but
            # never advance the estimator or its authority deadline.
            window = self._raw_closing_window
            if (self._uses_range_controller and window.uid == uid
                    and window.samples and window.samples[-1][0] == stamp
                    and window.rate is not None):
                self._braking_range_rate = window.rate
                self._braking_rate_source = "raw_depth_window"
            return
        evidence = self._longitudinal_feedforward.update(
            # The range derivative must use the accepted sample at this exact
            # timestamp, not a median/fused value lagging ego displacement.
            now=now, sample_timestamp=stamp, distance_m=frame.distance_state.raw_distance_m,
            target_id=uid, trusted=bool(feedback_ok),
            feedback_timestamp=None if feedback is None else feedback.timestamp,
            ego_forward_rpm=None if feedback is None else 0.5 * (feedback.left_forward_rpm + feedback.right_forward_rpm),
            yaw_rate_dps=yaw,
            target_bearing_deg=bearing,
            rotation_rate_m_s=None if compensation is None else compensation[0],
        )
        if self._uses_range_controller:
            window = self._matching_motion_window
            logger.info(
                "longitudinal_shared_window capture_frame_id=%s uid=%s sample_ts=%s "
                "status=%s samples=%s span_ms=%.1f range_rate=%s window_target_speed=%s "
                "matching_rpm=%.2f yaw_dps=%s reset_reason=%s matching_timeline_shared=True "
                "braking_window_independent=True deadline_renewed=False",
                frame.capture_frame_id, uid, stamp, evidence.status, len(window.samples), window.span*1000,
                window.rate, window.target_speed, evidence.target_rpm, yaw, evidence.chain_reset_reason,
            )
        if (evidence.status in {"warming_up", "interval_too_short"}
                and self._bridge_longitudinal_motion(frame, uid, now, stamp, yaw, bearing, evidence.status)):
            return
        self._longitudinal_bridge.remember(evidence, frame.distance_state.raw_distance_m)
        self._longitudinal_bridge_output_cap = None
        sample_gap = None if self._longitudinal_motion_stamp is None else stamp - self._longitudinal_motion_stamp
        self._longitudinal_motion_evidence = evidence
        self._longitudinal_motion_stamp = stamp
        self._last_velocity_reject_reason = None
        logger.info(
            "longitudinal_motion target=%s sample_ts=%.6f status=%s samples=%d "
            "range_rate=%s target_speed=%s tracking_base=%.2frpm extra_ff=%.2frpm "
            "sample_gap_ms=%s encoder_depth_skew_ms=%s raw_range_rate=%s rotation_rate=%s compensated=%s "
            "unbounded_target_rpm=%.2f matching_cap_rpm=%.2f matching_rate_limited=%s "
            "chain_reset_reason=%s instant_target_speed=%s window_target_speed=%s "
            "speed_window_ms=%.1f speed_window_samples=%d decline_policy=%s",
            uid, stamp, evidence.status, evidence.sample_count, evidence.range_rate_m_s,
            evidence.target_speed_m_s, evidence.target_rpm, evidence.feedforward_rpm,
            None if sample_gap is None else round(sample_gap * 1000.0, 1),
            None if feedback is None else round((feedback.timestamp - stamp) * 1000.0, 1),
            evidence.uncompensated_range_rate_m_s, evidence.rotation_rate_m_s, compensation is not None,
            evidence.unbounded_target_rpm, evidence.matching_cap_rpm, evidence.matching_rate_limited,
            evidence.chain_reset_reason,
            evidence.instantaneous_target_speed_m_s, evidence.window_target_speed_m_s,
            evidence.speed_window_sec * 1000., evidence.speed_window_samples, evidence.decline_policy,
        )

    def accept_longitudinal_limit(self, sample_timestamp: float, approved_rpm: float) -> None:
        if (sample_timestamp != self._distance_pid_last_sample_timestamp
                or self.last_distance_pid_result is None):
            return
        pi_before = (self._distance_pid._distance_pi.integral_m_s
                     if self.distance_pi_enabled else None)
        if self._distance_pid.accept_output_limit(sample_timestamp, approved_rpm):
            if self.distance_pi_enabled:
                logger.info(
                    "distance_pi_limit uid=%s sample_ts=%s requested_rpm=%s approved_rpm=%.2f "
                    "integral_before_m_s=%.5f integral_after_m_s=%.5f quantization_rpm=%.2f "
                    "measured_feedback=False deadline_renewed=False",
                    self.active_target_id, sample_timestamp, self.last_distance_pid_result.output_rpm,
                    approved_rpm, pi_before, self._distance_pid._distance_pi.integral_m_s,
                    self.cfg.forward_max_rpm / 100.,
                )
                return
            logger.info(
                "distance_pid_antiwindup sample_ts=%s requested_rpm=%s approved_rpm=%.2f "
                "integral_before=%.4f integral_after=%.4f measured_feedback=False",
                sample_timestamp, self.last_distance_pid_result.output_rpm, approved_rpm,
                self.last_distance_pid_result.integral_m_s, self._distance_pid._integral_m_s,
            )

    def reject_longitudinal_sample(self, sample_timestamp: float, reason: str = "admission_rejected") -> None:
        """No motor grant: undo new windup, not the saved same-UID PI memory."""
        if self.distance_pi_enabled and sample_timestamp == self._distance_pid_last_sample_timestamp:
            self._distance_pid.reject_output(sample_timestamp)
            logger.info("distance_pi_admission_rejected sample_ts=%s reason=%s deadline_renewed=False",
                        sample_timestamp, reason)

    def suspend_longitudinal_authority(self, now: float, reason: str) -> None:
        """An actually revoked grant cannot remain the next acceleration anchor.

        Keep bounded PI memory; this is neither a new measurement nor a zero
        output anti-windup update. A new grant still needs fresh depth and the
        resumed ramp must start from trustworthy measured wheel speed.
        """
        if self.distance_pi_enabled:
            self._distance_pid.suspend(now=now, reason=reason, retain=True,
                                       reset_execution=True)
            logger.info(
                "distance_pi_authority_suspended uid=%s sample_ts=%s reason=%s "
                "execution_suspended=True deadline_renewed=False",
                self.active_target_id, self._distance_pid_last_sample_timestamp, reason,
            )

    def _tracking_base_rpm(self, distance_m: float, now: float) -> Optional[float]:
        if self.distance_pi_enabled or (self.cfg.distance_approach_enable and not self.cfg.distance_approach_matching_enable):
            return None
        evidence = self._longitudinal_motion_evidence
        if evidence is not None and evidence.status == "transient_bridge":
            origin = self._longitudinal_bridge.origin
            if origin is None or not 0 <= now - origin.sample_timestamp < self.longitudinal_prior_max_age_sec:
                return None
        if (
            self.cfg.distance_feedforward_enable and evidence is not None and evidence.eligible
            and evidence.target_id == self.active_target_id
            and evidence.sample_timestamp is not None
            and self._distance_pid_sample_timestamp == evidence.sample_timestamp
            and 0.0 <= now - evidence.sample_timestamp <= 0.18
            and distance_m >= max(self.cfg.brake_distance_m, self.cfg.target_distance_m - 0.03)
        ):
            base = float(evidence.target_rpm)
            if evidence.status == "transient_bridge" and self._longitudinal_bridge._last_bridge_rpm is not None:
                base = min(base, self._longitudinal_bridge._last_bridge_rpm)
            configured = self.cfg.distance_matching_test_bias_rpm
            if configured > 0 and not self.cfg.distance_approach_enable:
                # Experimental command bias, NOT a calibrated target speed.
                # Never add it to warming/held/bridge/stop-suspect evidence.
                bias = 0.0
                if (evidence.status in {"ready", "ready_capped"}
                        and evidence.decline_policy == "bounded_far_positive"
                        and evidence.speed_window_samples >= 3 and evidence.speed_window_sec >= .06
                        and evidence.instantaneous_target_speed_m_s is not None
                        and evidence.instantaneous_target_speed_m_s > .03
                        and evidence.window_target_speed_m_s is not None
                        and evidence.window_target_speed_m_s > .10
                        and self.cfg.distance_matching_base_max_rpm > 0):
                    weight = min(1., max(0., (distance_m-self.cfg.target_distance_m-.20)/.30))
                    bias = max(0., min(configured*weight,
                        self.cfg.distance_matching_base_max_rpm-base, self.cfg.forward_max_rpm-base))
                audit_key = (self.active_target_id, evidence.sample_timestamp, round(base, 2), round(bias, 2))
                if audit_key != self._bias_audit_key:
                    self._bias_audit_key = audit_key
                    logger.info(
                        "longitudinal_bias_trial uid=%s sample_ts=%s configured_rpm=%.1f "
                        "measured_base_rpm=%.2f applied_bias_rpm=%.2f result_base_rpm=%.2f "
                        "distance_m=%.3f policy=%s estimator_changed=False deadline_renewed=False",
                        self.active_target_id, evidence.sample_timestamp, configured,
                        base, bias, base+bias, distance_m, evidence.decline_policy,
                    )
                base += bias
            return base
        return None

    def distance_only_forward_percent(self, frame: SensorFrame, sample_timestamp: float) -> Optional[int]:
        """Cache a conservative no-FF fallback from the SAME PID update.

        No reintegration, derivative recomputation or new motion evidence. Use
        the ordinary start band (not the matching-speed exception at setpoint).
        Runtime additionally applies ordinary depth caps and the approved cap.
        """
        result = self.last_distance_pid_result
        if self.distance_pi_enabled:
            if (result is None or self._distance_pid_last_sample_timestamp != sample_timestamp
                    or sample_timestamp != frame.distance_state.sample_timestamp
                    or not self._distance_pi_frame_qualified(frame, time.monotonic())
                    or result.measurement_jump_clamped):
                return None
            return self._forward_percent_for_rpm(max(0, result.output_rpm), allow_below_min=True)
        if (not self.cfg.distance_pid_enable or result is None
                or self._distance_pid_last_sample_timestamp != sample_timestamp
                or not self._is_fresh_depth_state(frame)
                or self._distance_longitudinally_untrusted(frame)
                or frame.distance_m is None
                or frame.distance_m < max(self.cfg.forward_start_distance_m,
                                          self.cfg.target_distance_m + .01,
                                          self.cfg.brake_distance_m)
                or frame.distance_state.safety_distance_m is not None
                or frame.distance_state.brake_latched
                or frame.hazard.active
                or any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
                or result.measurement_jump_clamped):
            return 0
        fallback_rpm = (result.distance_only_rpm if result.distance_only_rpm is not None else
                        self.cfg.forward_min_rpm + result.p_rpm + result.i_rpm + result.d_rpm)
        fallback_rpm = (math.floor(fallback_rpm + 1e-9) if result.distance_only_rpm is not None
                        else round(fallback_rpm))
        rpm = max(0, min(result.output_rpm, int(fallback_rpm)))
        return self._forward_percent_for_rpm(rpm, allow_below_min=True)

    def fresh_distance_recovery_percent(self, frame: SensorFrame, sample_timestamp: float,
                                        now: float) -> int:
        """New range-only authority after a walking prior can no longer be used.

        Read the cached PID once; do not integrate again or revive the prior.
        Only a far, fresh same-UID range and measured forward wheel speeds may
        resume, bounded to one normal 50ms acceleration step above those wheels.
        """
        s, fb = frame.distance_state, frame.steering_feedback
        values = (now, sample_timestamp, s.raw_distance_m, frame.distance_m,
                  None if fb is None else fb.timestamp,
                  None if fb is None else fb.left_forward_rpm,
                  None if fb is None else fb.right_forward_rpm,
                  None if fb is None else fb.yaw_rate_right_dps)
        if (any(v is None or not math.isfinite(v) for v in values)
                or sample_timestamp != s.sample_timestamp
                or not 0 <= now-sample_timestamp <= .18
                or self.active_target_id is None
                or not any(p.track_id == self.active_target_id for p in frame.persons)
                or self.search_state != "none" or self._target_stop_latched
                or min(s.raw_distance_m, frame.distance_m) <= self.cfg.target_distance_m+.30
                or fb is None or not fb.trustworthy
                or not 0 <= now-fb.timestamp <= .15
                or abs(fb.timestamp-sample_timestamp) > .15
                or abs(fb.yaw_rate_right_dps) > 15
                or (fb.raw_yaw_rate_right_dps is not None and (
                    not math.isfinite(fb.raw_yaw_rate_right_dps) or abs(fb.raw_yaw_rate_right_dps) > 15))
                or not all(0 <= v <= self._longitudinal_feedforward.config.max_abs_ego_rpm
                           for v in (fb.left_forward_rpm, fb.right_forward_rpm))):
            return 0
        # Includes fresh-depth classification, obstacle, hazard, jump and latch checks.
        requested = self.distance_only_forward_percent(frame, sample_timestamp)
        measured = .5*(fb.left_forward_rpm+fb.right_forward_rpm)
        rise = float(self.cfg.distance_pid_output_rise_rpm_per_sec)
        rise = min(240., rise) if rise > 0 else 240.
        cap_rpm = measured + rise*.05
        approved = min(requested, max(0, int(math.floor(100*cap_rpm/self.cfg.forward_max_rpm))))
        logger.info(
            "depth_fresh_distance_recovery capture_frame_id=%s uid=%s sample_ts=%s "
            "distance_m=%.3f requested_percent=%s approved_percent=%s measured_rpm=%.2f "
            "measured_cap_rpm=%.2f prior_used=False pid_recomputed=False depth_ttl_ms=180",
            frame.capture_frame_id, self.active_target_id, sample_timestamp, frame.distance_m,
            requested, approved, measured, cap_rpm,
        )
        return approved

    def fresh_bridge_forward_percent(self, frame: SensorFrame, sample_timestamp: float,
                                     now: float, previous_approved_rpm: float) -> Optional[int]:
        """Bound only old matching speed, not this sample's distance response.

        Requires an independently validated LIVE forward grant at the caller.
        An expired/revoked grant must use fresh_distance_recovery_percent instead.
        No PID update, new estimate, or deadline extension is performed here.
        None preserves the legacy PID bridge policy.
        """
        if not self.cfg.distance_approach_enable:
            return None
        result = self.last_distance_pid_result
        if (result is None or result.approach_mode == "legacy_pid"
                or self._distance_pid_last_sample_timestamp != sample_timestamp
                or sample_timestamp != frame.distance_state.sample_timestamp
                or not math.isfinite(sample_timestamp) or not 0 <= now-sample_timestamp <= .18
                or not math.isfinite(previous_approved_rpm) or previous_approved_rpm <= 0
                or not self._is_fresh_depth_state(frame)
                or self._distance_longitudinally_untrusted(frame)
                or frame.distance_m is None or result.measurement_jump_clamped
                or frame.distance_state.safety_distance_m is not None
                or frame.distance_state.brake_latched or frame.hazard.active
                or any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
                or not any(p.track_id == self.active_target_id for p in frame.persons)):
            return 0
        distance_percent = self.distance_only_forward_percent(frame, sample_timestamp)
        base = self._tracking_base_rpm(frame.distance_m, now)
        # A matching prior may have expired between PID and commit. It cannot
        # borrow the fresh range timestamp to survive that expiry.
        if base is None:
            return distance_percent
        retained_base = min(result.tracking_base_rpm, max(0., base), previous_approved_rpm)
        # A live grant may shrink AFTER the PID was calculated. Persist the
        # prior-only reduction, or the next fresh distance approval could
        # accidentally allow that old matching speed to grow back.
        retained_base = self._longitudinal_bridge.limit_prior_rpm(retained_base)
        withdrawn_base = max(0., result.tracking_base_rpm-retained_base)
        mixed_rpm = max(0., result.output_rpm-withdrawn_base)
        mixed_percent = int(math.floor(100.*mixed_rpm/self.cfg.forward_max_rpm + 1e-9))
        # The pure-distance alternative already obeys the SAME update's
        # braking envelope, acceleration bound and output ceiling.
        return max(distance_percent, mixed_percent)

    def _update_distance_pid(self, distance_m: float, *, now: Optional[float] = None,
                             forward_control: bool = True) -> DistancePidResult:
        sample_now = time.monotonic() if now is None else float(now)
        physical_stamp = self._distance_pid_sample_timestamp
        if ((self._uses_range_controller and not self._distance_approach_sample_trusted)
                or (physical_stamp is not None and not 0.0 <= sample_now - physical_stamp <= (
                    .18 if self._uses_range_controller else .25))
                or (self.distance_pi_enabled and physical_stamp is None)):
            # Invalid physical time is not permission to substitute wall time.
            # Return zero without contaminating the last accepted PID sample.
            return DistancePidResult(
                float(distance_m), float(distance_m), float(self.cfg.target_distance_m),
                float(distance_m) - float(self.cfg.target_distance_m),
                0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, False, False,
            )
        if (
            physical_stamp is not None and self._distance_pid_last_sample_timestamp is not None
            and physical_stamp <= self._distance_pid_last_sample_timestamp
            and self.last_distance_pid_result is not None
            and (not self.distance_pi_enabled or forward_control == self._distance_pid_last_forward_control)
        ):
            if (
                self.last_distance_pid_result.tracking_base_rpm > 0.0
                and self._tracking_base_rpm(float(distance_m), sample_now) is None
            ):
                previous = self.last_distance_pid_result
                fallback = (int(math.floor(previous.distance_only_rpm))
                            if previous.distance_only_rpm is not None else
                            0 if abs(previous.error_m) <= self.cfg.distance_pid_deadband_m else max(
                                0, int(round(self.cfg.forward_min_rpm + previous.p_rpm + previous.i_rpm + previous.d_rpm))))
                # Revoking FF cannot add speed or integrate this sample again.
                self.last_distance_pid_result = replace(
                    previous, tracking_base_rpm=0.0,
                    output_rpm=min(max(0, previous.output_rpm), fallback),
                )
            return self.last_distance_pid_result
        if (
            self.last_distance_pid_result is not None
            and self._distance_pid_last_input_m is not None
            and self._distance_pid_last_update_at is not None
            and not self.distance_pi_enabled
            and abs(float(distance_m) - float(self._distance_pid_last_input_m)) <= 1e-9
            and sample_now - float(self._distance_pid_last_update_at) < 0.005
        ):
            return self.last_distance_pid_result
        if (self.distance_pi_enabled and forward_control
                and self.last_distance_pid_result is not None
                and self.last_distance_pid_result.output_rpm > 0
                and callable(self._live_longitudinal_authority_reader)):
            # The runtime callback takes only UID and checks its own current
            # monotonic clock. Keep the same contract as the recovery reader.
            live = self._live_longitudinal_authority_reader(self.active_target_id)
            if (live is None or live[0] != "forward" or live[1] <= 0
                    or live[2] != self.active_target_id
                    or live[3] != self._distance_pid_last_sample_timestamp):
                # Read-time braking/identity vetoes need not have mutated the
                # canonical tuple. Never continue a ramp solely on its clock.
                self.suspend_longitudinal_authority(sample_now, "no_live_grant_before_pi")
        result = self._distance_pid.update(
            float(distance_m),
            float(self.cfg.target_distance_m),
            now=sample_now if physical_stamp is None else physical_stamp,
            tracking_base_rpm=self._tracking_base_rpm(float(distance_m), sample_now),
            measurement_age_sec=0. if physical_stamp is None else max(0., sample_now - physical_stamp),
            braking_range_rate_m_s=self._braking_range_rate if self._uses_range_controller else None,
            raw_closure_valid=self._braking_rate_source == "raw_depth_window",
            allow_motion_memory=self._distance_pi_motion_memory_allowed,
            braking_raw_distance_m=self._distance_pi_raw_distance_m,
            ego_forward_rpm=(self._distance_pi_ego_forward_rpm
                if self._distance_pi_feedback_timestamp is not None
                and 0 <= sample_now - self._distance_pi_feedback_timestamp <= .15 else None),
            execution_now=sample_now,
            forward_control=forward_control,
        )
        if (not self.cfg.distance_approach_enable
                and self._longitudinal_bridge_output_cap is not None and result.tracking_base_rpm > 0):
            cap = max(0, int(self._longitudinal_bridge_output_cap))
            if result.output_rpm > cap:
                self._distance_pid.accept_output_limit(physical_stamp, cap)
                result = replace(result, output_rpm=cap)
        self.last_distance_pid_result = result
        self._distance_pid_last_input_m = float(distance_m)
        self._distance_pid_last_update_at = sample_now
        self._distance_pid_last_sample_timestamp = physical_stamp
        self._distance_pid_last_forward_control = forward_control
        if self.distance_pi_enabled and forward_control:
            logger.info(
                "distance_pi uid=%s sample_ts=%s error_m=%.4f p_rpm=%.2f i_rpm=%.2f "
                "request_rpm=%s unslewed_rpm=%.2f envelope_rpm=%s status=%s "
                "closure_source=%s ego_rpm=%s matching_output=False recovery_policy=unified "
                "sample_dt_sec=%.4f integral_m_s=%.5f integral_frozen=%s brake_source=%s "
                "pi_demand_rpm=%.2f launch_floor_rpm=%.2f total_demand_rpm=%.2f "
                "software_rise_bypassed=%s demand_limit_reason=%s feedback_age_ms=%s "
                "feedback_ts=%s depth_feedback_skew_ms=%s kp_per_sec=%.3f "
                "launch_masks_pi=%s brake_limit_loss_rpm=%.2f motion_origin_ts=%s "
                "motion_uncertainty_m_s=%.3f deadline_renewed=False",
                self.active_target_id, physical_stamp, result.error_m, result.p_rpm,
                result.i_rpm, result.output_rpm, result.unslewed_output_rpm,
                result.approach_cap_rpm, result.pi_status, self._braking_rate_source,
                self._distance_pi_ego_forward_rpm, result.pi_sample_dt_sec,
                result.pi_integral_m_s, result.pi_integral_frozen, result.pi_brake_source,
                result.pi_demand_rpm, result.pi_launch_floor_rpm, result.pi_total_demand_rpm,
                result.pi_software_rise_bypassed, result.pi_demand_limit_reason,
                None if self._distance_pi_feedback_timestamp is None else
                round((sample_now-self._distance_pi_feedback_timestamp)*1000., 1),
                self._distance_pi_feedback_timestamp,
                None if self._distance_pi_feedback_timestamp is None or physical_stamp is None else
                round((self._distance_pi_feedback_timestamp-physical_stamp)*1000., 1),
                self.cfg.distance_pi_kp_per_sec,
                result.pi_launch_floor_rpm > result.pi_demand_rpm,
                max(0., result.pi_total_demand_rpm-result.approach_cap_rpm),
                result.pi_motion_origin_ts, result.pi_motion_uncertainty_m_s,
            )
        elif result.approach_mode != "legacy_pid":
            logger.info(
                "longitudinal_approach uid=%s sample_ts=%s mode=%s distance_m=%.3f "
                "target_m=%.3f base_rpm=%.2f correction_rpm=%.2f request_rpm=%s "
                "envelope_rpm=%.2f closing_m_s=%.3f braking_distance_m=%.3f "
                "distance_only_rpm=%.2f response_budget_ms=%.1f deceleration_assumed=True deadline_renewed=False "
                "closing_source=%s closing_window_samples=%s closing_window_ms=%.1f "
                "closing_window_status=%s no_matching_budget_rpm=%.1f no_matching_reason=%s",
                self.active_target_id, physical_stamp, result.approach_mode, result.actual_distance_m,
                result.target_distance_m, result.tracking_base_rpm, result.p_rpm, result.output_rpm,
                result.approach_cap_rpm, result.approach_closing_m_s,
                result.approach_braking_distance_m, result.distance_only_rpm, result.approach_delay_sec*1000.,
                self._braking_rate_source, len(self._raw_closing_window.samples),
                self._raw_closing_window.span*1000., self._raw_closing_window.status,
                self.cfg.distance_approach_no_matching_max_rpm,
                "trial_disabled" if not self.cfg.distance_approach_matching_enable else
                "not_applicable" if result.tracking_base_rpm > 0 else (
                    self._longitudinal_motion_evidence.status if self._longitudinal_motion_evidence is not None
                    else getattr(self, "_last_velocity_reject_reason", None) or "no_evidence"),
            )
        logger.info(
            "distance_pid actual=%.3fm raw=%.3fm target=%.3fm error=%+.3fm rate=%+.3fm/s "
            "p=%+.2frpm i=%+.2frpm d=%+.2frpm output=%+drpm raw_output=%+.1frpm "
            "jump_clamped=%s slew_limited=%s sample_ts=%s tracking_base=%.2frpm matching_source=%s "
            "integral_cap_rpm=%.2f integral_at_cap=%s uid=%s",
            result.actual_distance_m,
            result.raw_actual_distance_m,
            result.target_distance_m,
            result.error_m,
            result.error_rate_m_s,
            result.p_rpm,
            result.i_rpm,
            result.d_rpm,
            result.output_rpm,
            result.unslewed_output_rpm,
            result.measurement_jump_clamped,
            result.output_slew_limited,
            physical_stamp,
            result.tracking_base_rpm,
            "none" if result.tracking_base_rpm <= 0 else
            "bridge" if self._longitudinal_bridge_output_cap is not None else "fresh",
            result.integral_cap_rpm,
            result.integral_cap_rpm > 0 and abs(result.i_rpm) >= result.integral_cap_rpm - 1e-6,
            self.active_target_id,
        )
        return result

    def _reset_distance_pid(self) -> None:
        self._distance_pid.reset()
        self.last_distance_pid_result = None
        self._distance_pid_last_input_m = None
        self._distance_pid_last_update_at = None
        self._distance_pid_last_sample_timestamp = None
        self._distance_pid_last_forward_control = True

    def _distance_pi_frame_qualified(self, frame: SensorFrame, now: float, *,
                                     allow_active_reverse: bool = False,
                                     continuation_only: bool = False) -> bool:
        """PI memory is never a replacement for fresh, same-UID measurements."""
        state = frame.distance_state
        stamp = state.sample_timestamp
        age_limit = (self.cfg.depth_longitudinal_sample_max_age_sec
                     if continuation_only else min(.18, self.cfg.depth_longitudinal_sample_max_age_sec))
        return bool(
            self._is_fresh_depth_state(frame)
            and isinstance(stamp, (int, float)) and math.isfinite(stamp)
            and 0 <= now - stamp <= age_limit
            and self.active_target_id is not None and self._has_seen_person
            and any(p.track_id == self.active_target_id for p in frame.persons)
            and self.search_state == "none"
            and (not self._target_stop_latched or (allow_active_reverse and self._reverse_active))
            and frame.distance_m is not None and math.isfinite(frame.distance_m)
            and state.raw_distance_m is not None and math.isfinite(state.raw_distance_m)
            and min(frame.distance_m, state.raw_distance_m) > 0
            and state.safety_distance_m is None and not state.brake_latched
            and not frame.hazard.active
            and not any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
            and not self._distance_longitudinally_untrusted(frame)
        )

    def _pause_distance_pi(self, frame: SensorFrame, now: float, reason: str) -> bool:
        """Retain bounded memory for absent observations, not new bad evidence.

        Returns whether the old sample may still have a live lease. The caller
        does NOT issue a new action or change its original sample timestamp.
        """
        state = frame.distance_state
        previous = self._distance_pid_last_sample_timestamp
        same_target = bool(
            self.active_target_id is not None and self._has_seen_person
            and any(p.track_id == self.active_target_id for p in frame.persons)
            and self.search_state == "none" and not self._target_stop_latched
        )
        safe = bool(same_target and not frame.hazard.active
                    and not any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
                    and state.safety_distance_m is None and not state.brake_latched)
        no_observation = bool(state.raw_distance_m is None and state.sample_timestamp is None
                              and (not self._distance_longitudinally_untrusted(frame, check_hold_confidence=False)
                                   or (previous is not None and self._low_confidence_replay_can_retain_velocity(frame, previous))))
        expired_accepted = bool(
            self._is_fresh_depth_state(frame) and not self._distance_longitudinally_untrusted(frame)
            and state.sample_timestamp is not None and math.isfinite(state.sample_timestamp)
            and state.sample_timestamp <= now and previous is not None
            and state.sample_timestamp <= previous
        )
        deferred_measurement = bool(
            self.cfg.depth_longitudinal_sample_max_age_sec > .18
            and self._distance_pi_frame_qualified(frame, now, continuation_only=True)
            and now - state.sample_timestamp > .18
            and min(frame.distance_m, state.raw_distance_m) >= self.cfg.target_distance_m
        )
        retain = bool(safe and (no_observation or expired_accepted or deferred_measurement)
                      and previous is not None and 0 <= now - previous <= self.cfg.distance_pi_memory_sec)
        # This is only a possible OLD grant; runtime checks the actual grant,
        # its original deadline and braking margin before each motor write.
        # No new timestamp or forward action is created by late observations.
        lease_limit = (.18 if self._reverse_active or not self._distance_pid_last_forward_control
                       else self.cfg.depth_longitudinal_sample_max_age_sec)
        lease_live = bool(retain and now - previous <= lease_limit)
        if retain:
            self._distance_pid.suspend(now=now, reason=reason, retain=True,
                                       reset_execution=not lease_live)
        else:
            self._reset_distance_pid()
        key = (previous, reason, retain, lease_live)
        if self._distance_pi_pause_key != key:
            self._distance_pi_pause_key = key
            logger.info(
                "distance_pi_pause capture_frame_id=%s uid=%s sample_ts=%s reason=%s "
                "memory_retained=%s old_lease_may_be_live=%s depth_ttl_ms=%.0f "
                "fresh_update_ms=180 pid_updated=False deadline_renewed=False",
                frame.capture_frame_id, self.active_target_id, previous, reason, retain, lease_live,
                1000 * lease_limit,
            )
        return lease_live

    def _distance_pi_longitudinal_decision(self, frame_index: int, frame: SensorFrame,
                                          target: PersonTarget, target_steerable: bool,
                                          now: float) -> ControlDecision:
        if (not self._reverse_active and not self.cfg.distance_parking_enable
                and self._target_stop_latched):
            # Use the existing IR-only parking policy on the depth-only path
            # too. A completed reverse must not leave a permanent forward veto.
            self._target_distance_lock_decision(frame, now, off_center=False, target=target)
        if ((self._reverse_active or not target_steerable)
                and self._distance_pi_frame_qualified(frame, now, allow_active_reverse=True)
                and (self._distance_approach_sample_trusted or not target_steerable)):
            # The reverse controller owns its own latch/release hysteresis.
            # Do not trap it behind the new forward-only PI qualification.
            # An unsteerable, same-UID near target can retain the old protected
            # reverse policy. Its fresh range is NOT permission to run forward.
            reverse = self._reverse_control_decision(
                frame_index, frame, target, force_immediate_close=not target_steerable,
                target_steering_limit_rpm=None, target_steerable=target_steerable)
            if reverse is not None:
                return reverse
            if not self._reverse_active and not self.cfg.distance_parking_enable:
                self._target_distance_lock_decision(frame, now, off_center=False, target=target)
        if not self._distance_pi_frame_qualified(frame, now) or not self._distance_approach_sample_trusted:
            lease_live = self._pause_distance_pi(frame, now, "depth_not_qualified")
            if lease_live:
                return ControlDecision(reason="longitudinal_distance_pi_observation_skipped")
            self._forward_active = False
            return ControlDecision(
                actions=[ControlAction.forward(0, "longitudinal_distance_untrusted_hold")],
                current_forward_percent=0, clear_action_queue=True,
                reason="longitudinal_distance_untrusted_hold",
            )
        self._distance_pi_pause_key = None
        reverse = self._reverse_control_decision(
            frame_index, frame, target, force_immediate_close=not target_steerable,
            target_steering_limit_rpm=None, target_steerable=target_steerable,
        )
        if reverse is not None:
            return reverse
        self._remember_target_distance(frame, now)
        speed = self._forward_percent_for_distance(float(frame.distance_m), now=now)
        speed = self._limit_depth_quality_forward_percent(frame, speed, now)
        reason = "longitudinal_distance_pid" if speed > 0 else "longitudinal_distance_hold"
        return ControlDecision(
            actions=[ControlAction.forward(speed, reason)], current_forward_percent=speed,
            is_forwarding=speed > 0, clear_action_queue=speed <= 0, reason=reason,
        )

    def _reverse_percent_for_distance(
        self,
        distance_m: float,
        *,
        approach_speed_m_s: float = 0.0,
        now: Optional[float] = None,
        frame: Optional[SensorFrame] = None,
    ) -> int:
        cfg = self.cfg
        # Reverse and forward are mutually exclusive longitudinal states.
        self._forward_active = False
        if cfg.distance_pid_enable:
            # Lateral quality can forbid steering without invalidating the
            # locked target's fresh near range. Preserve only the existing
            # protected reverse path in both visual and depth callers.
            reverse_only_sample = bool(
                self.distance_pi_enabled and frame is not None
                and self._distance_pi_frame_qualified(
                    frame, time.monotonic() if now is None else now, allow_active_reverse=True))
            previous_trusted = self._distance_approach_sample_trusted
            previous_stamp = self._distance_pid_sample_timestamp
            if reverse_only_sample:
                self._distance_approach_sample_trusted = True
                self._distance_pid_sample_timestamp = frame.distance_state.sample_timestamp
            try:
                base_rpm = abs(min(0, int(self._update_distance_pid(
                    distance_m, now=now, forward_control=False).output_rpm)))
            finally:
                if reverse_only_sample:
                    self._distance_approach_sample_trusted = previous_trusted
                    self._distance_pid_sample_timestamp = previous_stamp
            if base_rpm <= 0:
                return 0
        else:
            min_rpm = max(1, int(cfg.reverse_min_rpm))
            max_rpm = max(min_rpm, int(cfg.reverse_max_rpm))
            start_m = max(float(cfg.brake_distance_m) + 0.05, float(cfg.reverse_start_distance_m))
            full_speed_m = max(
                float(cfg.brake_distance_m) + 0.05,
                min(start_m - 0.01, float(cfg.reverse_full_speed_distance_m)),
            )
            span = max(0.01, start_m - full_speed_m)
            progress = max(0.0, min(1.0, (start_m - float(distance_m)) / span))
            base_rpm = int(round(min_rpm + (max_rpm - min_rpm) * progress))

        absolute_cap = max(1, int(cfg.reverse_max_rpm))
        runtime_cap = max(1, min(absolute_cap, int(cfg.reverse_runtime_cap_rpm)))
        approach_speed = max(0.0, float(approach_speed_m_s))
        feedforward_floor = max(
            int(cfg.reverse_min_rpm),
            int(cfg.reverse_feedforward_floor_rpm),
        )
        feedforward_rpm = int(
            round(
                feedforward_floor
                + max(0.0, float(cfg.reverse_feedforward_gain_rpm_per_m_s))
                * approach_speed
            )
        )
        rpm = max(int(base_rpm), feedforward_rpm)
        rpm = max(1, min(runtime_cap, rpm))
        self._reverse_last_base_rpm = int(base_rpm)
        self._reverse_last_feedforward_rpm = int(feedforward_rpm)
        self._reverse_last_output_rpm = int(rpm)
        return max(
            1,
            min(100, int(round(100.0 * rpm / max(1, int(cfg.forward_max_rpm))))),
        )

    def _update_reverse_approach_speed(self, distance_m: float, now: float) -> float:
        """Estimate target approach speed from consecutive fresh target distances."""
        previous_distance = self._reverse_last_radar_distance_m
        previous_ts = self._reverse_last_distance_at
        raw_speed = 0.0
        if previous_distance is not None and previous_ts is not None:
            dt = float(now) - float(previous_ts)
            if 0.015 <= dt <= 0.50:
                raw_speed = max(
                    0.0,
                    min(3.0, (float(previous_distance) - float(distance_m)) / dt),
                )
        alpha = max(
            0.05,
            min(1.0, float(self.cfg.reverse_approach_speed_filter_alpha)),
        )
        self._reverse_filtered_approach_speed_m_s = (
            alpha * raw_speed
            + (1.0 - alpha) * self._reverse_filtered_approach_speed_m_s
        )
        self._reverse_last_radar_distance_m = float(distance_m)
        self._reverse_last_distance_at = float(now)
        return float(self._reverse_filtered_approach_speed_m_s)

    def _clear_target_stop_visual_reference(self) -> None:
        self._target_stop_visual_track_id = None
        self._target_stop_peak_bbox_area_ratio = None

    @staticmethod
    def _target_bbox_area_ratio(frame: SensorFrame, target: PersonTarget) -> float:
        x1, y1, x2, y2 = (float(v) for v in target.bbox)
        frame_area = float(max(1, int(frame.width)) * max(1, int(frame.height)))
        return max(0.0, (x2 - x1) * (y2 - y1)) / frame_area

    def _visual_reverse_guard(
        self,
        frame_index: int,
        frame: SensorFrame,
        target: PersonTarget,
        now: float,
        fresh_distance_m: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Use target scale only as a short-range fallback when Depth is missing."""
        depth_veto_distance_m = max(
            float(self.cfg.brake_distance_m) + 0.05,
            min(
                float(self.cfg.reverse_start_distance_m),
                float(self.cfg.reverse_visual_guard_max_distance_m),
            ),
        )
        if int(frame_index) == int(self._reverse_visual_frame_index):
            detail = str(self._reverse_visual_guard_detail)
            if (
                detail != "none"
                and fresh_distance_m is not None
                and float(fresh_distance_m)
                >= depth_veto_distance_m
            ):
                # Depth30 and vision can evaluate the same frame in either
                # order. A fresh far sample must invalidate an earlier cached
                # visual-near result from that frame.
                self._reverse_visual_guard_detail = "none"
                self._reverse_visual_candidate_detail = "none"
                self._reverse_visual_candidate_frames = 0
                return False, "fresh_far_depth"
            return detail != "none", detail
        area_ratio = self._target_bbox_area_ratio(frame, target)
        _x1, y1, _x2, y2 = (float(v) for v in target.bbox)
        height_ratio = max(0.0, y2 - y1) / float(max(1, int(frame.height)))
        same_target = bool(
            self._reverse_visual_target_id == int(target.track_id)
            and self._reverse_visual_sample_at is not None
            and float(now) - float(self._reverse_visual_sample_at) <= 0.50
        )
        previous_area = self._reverse_visual_area_ratio if same_target else None
        growth_ratio = max(1.01, float(self.cfg.reverse_visual_guard_growth_ratio))
        rapid_growth = bool(
            previous_area is not None
            and area_ratio >= float(self.cfg.reverse_visual_guard_growth_min_area_ratio)
            and area_ratio >= float(previous_area) * growth_ratio
        )
        area_threshold = float(self.cfg.reverse_visual_guard_area_ratio)
        growth_area_threshold = float(self.cfg.reverse_visual_guard_growth_min_area_ratio)
        height_area_threshold = max(growth_area_threshold, area_threshold * 0.75)
        if area_ratio >= area_threshold:
            candidate_detail = "large_area"
        elif (
            height_ratio >= float(self.cfg.reverse_visual_guard_height_ratio)
            and area_ratio >= height_area_threshold
        ):
            # Height alone is ambiguous: a tall, narrow full-body box can still
            # be several metres away. Require meaningful width/area as well.
            candidate_detail = "near_full_height_and_area"
        elif rapid_growth:
            candidate_detail = "rapid_growth"
        else:
            candidate_detail = "none"

        recent_distance = (
            float(fresh_distance_m)
            if fresh_distance_m is not None
            else self._reverse_last_radar_distance_m
        )
        recent_distance_age = (
            0.0
            if fresh_distance_m is not None
            else (
                None
                if self._reverse_last_distance_at is None
                else max(0.0, float(now) - float(self._reverse_last_distance_at))
            )
        )
        recent_far_depth = bool(
            candidate_detail != "none"
            and recent_distance is not None
            and recent_distance_age is not None
            and recent_distance_age <= max(0.05, float(self.cfg.reverse_radar_max_age_sec))
            and float(recent_distance)
            >= depth_veto_distance_m
        )
        if recent_far_depth:
            if not self._reverse_visual_guard_blocked_logged:
                logger.info(
                    "reverse_visual_near_guard_blocked detail=%s distance=%.2fm "
                    "age=%.0fms limit=%.2fm area=%.3f height=%.3f",
                    candidate_detail,
                    float(recent_distance),
                    float(recent_distance_age) * 1000.0,
                    depth_veto_distance_m,
                    area_ratio,
                    height_ratio,
                )
            self._reverse_visual_guard_blocked_logged = True
            candidate_detail = "none"
        else:
            self._reverse_visual_guard_blocked_logged = False

        required_frames = max(1, int(self.cfg.reverse_confirm_frames))
        if candidate_detail == "none":
            candidate_frames = 0
        elif same_target and self._reverse_visual_candidate_detail != "none":
            candidate_frames = int(self._reverse_visual_candidate_frames) + 1
        else:
            candidate_frames = 1
        guard_detail = candidate_detail if candidate_frames >= required_frames else "none"

        self._reverse_visual_target_id = int(target.track_id)
        self._reverse_visual_area_ratio = float(area_ratio)
        self._reverse_visual_height_ratio = float(height_ratio)
        self._reverse_visual_sample_at = float(now)
        self._reverse_visual_frame_index = int(frame_index)
        self._reverse_visual_guard_detail = guard_detail
        self._reverse_visual_candidate_detail = candidate_detail
        self._reverse_visual_candidate_frames = int(candidate_frames)
        active = guard_detail != "none"
        if active and not self._reverse_visual_guard_logged:
            logger.info(
                "reverse_visual_near_guard target=%d detail=%s confirmations=%d/%d "
                "area=%.3f height=%.3f previous_area=%s",
                int(target.track_id),
                guard_detail,
                int(candidate_frames),
                int(required_frames),
                area_ratio,
                height_ratio,
                "none" if previous_area is None else f"{float(previous_area):.3f}",
            )
        self._reverse_visual_guard_logged = active
        return active, guard_detail

    def _update_target_stop_visual_reference(
        self,
        frame: SensorFrame,
        target: Optional[PersonTarget],
    ) -> None:
        if str(getattr(frame.distance_state, "source", "")) != "vision_depth":
            return
        if target is None or frame.width <= 0 or frame.height <= 0:
            return
        track_id = int(target.track_id)
        if self._target_stop_visual_track_id is None:
            self._target_stop_visual_track_id = track_id
        if self._target_stop_visual_track_id != track_id:
            return
        area_ratio = self._target_bbox_area_ratio(frame, target)
        previous = self._target_stop_peak_bbox_area_ratio
        self._target_stop_peak_bbox_area_ratio = (
            area_ratio if previous is None else max(float(previous), area_ratio)
        )

    def _vision_depth_release_evidence(
        self,
        frame: SensorFrame,
        target: Optional[PersonTarget],
    ) -> Tuple[bool, str]:
        if str(getattr(frame.distance_state, "source", "")) != "vision_depth":
            return True, "not_depth"
        depth_detail = str(getattr(frame.distance_state, "source_detail", ""))
        if (
            not depth_detail.startswith("depth_")
            or depth_detail.endswith("_hold")
            or getattr(frame.distance_state, "raw_distance_m", None) is None
        ):
            return False, "depth_not_fresh"
        if target is None or frame.width <= 0 or frame.height <= 0:
            return False, "target_missing"
        if (
            self._target_stop_visual_track_id is not None
            and int(target.track_id) != int(self._target_stop_visual_track_id)
        ):
            # DeepSORT can briefly assign a new track number while the same
            # locked ReID person remains continuously visible. The selected
            # target has already passed _select_person's active-target gate;
            # accept that stable handoff instead of holding release forever.
            if (
                self.active_target_id is None
                or int(target.track_id) != int(self.active_target_id)
            ):
                return False, "target_changed"
            self._target_stop_visual_track_id = int(target.track_id)

        peak_area = self._target_stop_peak_bbox_area_ratio
        current_area = self._target_bbox_area_ratio(frame, target)
        if peak_area is None:
            self._update_target_stop_visual_reference(frame, target)
            return False, "bbox_reference_missing"

        shrink_ratio = max(
            0.05,
            min(1.0, float(self.cfg.target_distance_release_visual_shrink_ratio)),
        )
        if current_area > float(peak_area) * shrink_ratio:
            return False, "bbox_not_smaller"

        x1, _, x2, _ = (float(v) for v in target.bbox)
        edge_margin = max(
            2.0,
            float(frame.width)
            * max(
                0.0,
                float(self.cfg.target_distance_release_visual_edge_margin_ratio),
            ),
        )
        if x1 <= edge_margin or x2 >= float(frame.width) - edge_margin:
            return False, "bbox_edge_clipped"
        return True, "ready"

    def _reverse_control_decision(
        self,
        frame_index: int,
        frame: SensorFrame,
        target: Optional[PersonTarget],
        *,
        force_immediate_close: bool = False,
        target_steering_limit_rpm: Optional[float] = None,
        target_steerable: bool = True,
    ) -> Optional[ControlDecision]:
        """Maintain 1.5 m only when the locked camera target has a fresh range."""
        cfg = self.cfg
        if target is not None and int(target.track_id) == self.active_target_id and self._older_depth_observation(frame):
            return None
        if not cfg.reverse_enable and not cfg.near_distance_rotate_only_enable:
            if self._reverse_active:
                self._reset_reverse_control("reverse_disabled")
            return None
        visual_locked = bool(
            target is not None
            and self._has_seen_person
            and self.active_target_id is not None
            and int(self.active_target_id) >= 0
            and int(target.track_id) == int(self.active_target_id)
        )
        if not visual_locked:
            # Keep the near-distance latch through the short lost-confirming
            # window. A detector dropout must be allowed to hold the previous
            # in-place yaw command instead of being converted into an
            # immediate STOP by the next branch.
            preserve_near_loss_hold = bool(
                target is None
                and cfg.near_distance_rotate_only_enable
                and self._near_distance_rotation_only_active
                and self._near_distance_rotation_only_last_distance_m is not None
            )
            if not preserve_near_loss_hold:
                self._near_distance_rotation_only_active = False
                self._near_distance_rotation_only_last_distance_m = None
            if self._reverse_active:
                self._reset_reverse_control("visual_target_unavailable")
                if cfg.distance_parking_enable:
                    return ControlDecision(
                        explicit_stop_requested=True,
                        clear_action_queue=True,
                        stop_action_execution=True,
                        reason="reverse_visual_lost_stop",
                    )
                # Let the visual lost/search state take over without a distance STOP.
            self._reset_reverse_control("visual_target_unavailable")
            return None

        near_distance = None
        if (self.distance_pi_enabled and self._distance_approach_sample_trusted
                and frame.distance_m is not None
                and frame.distance_m >= cfg.target_distance_m - cfg.distance_pid_deadband_m):
            self._near_distance_rotation_only_active = False
            self._near_distance_rotation_only_last_distance_m = None
        if cfg.near_distance_rotate_only_enable and (
                not self.distance_pi_enabled
                or (frame.distance_m is not None
                    and frame.distance_m < cfg.target_distance_m - cfg.distance_pid_deadband_m)):
            near_distance = frame.distance_m
            if near_distance is None:
                near_distance = getattr(frame.distance_state, "used_distance_m", None)
            if near_distance is not None:
                near_distance = float(near_distance)
                near_limit = max(
                    float(cfg.brake_distance_m),
                    float(cfg.near_distance_rotate_only_distance_m),
                )
                if near_distance <= near_limit and self._tracking_base_rpm(near_distance, time.monotonic()) is None:
                    self._near_distance_rotation_only_active = True
                    self._near_distance_rotation_only_last_distance_m = near_distance
                    return self._near_distance_rotation_only_decision(
                        frame_index,
                        frame,
                        target,
                        near_distance,
                        near_limit,
                        target_steering_limit_rpm=target_steering_limit_rpm,
                        target_steerable=target_steerable,
                    )
                self._near_distance_rotation_only_active = False
                self._near_distance_rotation_only_last_distance_m = None
            elif self._near_distance_rotation_only_active:
                near_limit = max(
                    float(cfg.brake_distance_m),
                    float(cfg.near_distance_rotate_only_distance_m),
                )
                remembered_distance = self._near_distance_rotation_only_last_distance_m
                if remembered_distance is not None:
                    return self._near_distance_rotation_only_decision(
                        frame_index,
                        frame,
                        target,
                        float(remembered_distance),
                        near_limit,
                        target_steering_limit_rpm=target_steering_limit_rpm,
                        target_steerable=target_steerable,
                    )

        if not cfg.reverse_enable:
            if self._reverse_active:
                self._reset_reverse_control("reverse_disabled")
            return None

        if self._target_stop_latched:
            self._update_target_stop_visual_reference(frame, target)

        distance_now = time.monotonic()
        distance = self._stable_reverse_radar_distance(frame)
        visual_near_guard, visual_near_detail = self._visual_reverse_guard(
            frame_index,
            frame,
            target,
            distance_now,
            fresh_distance_m=distance,
        )
        fresh_reverse_distance = distance is not None
        if distance is None and self._reverse_active and frame.distance_m is not None:
            state = frame.distance_state
            source = str(getattr(state, "source", ""))
            detail = str(getattr(state, "source_detail", ""))
            sample_age = getattr(state, "sample_age_sec", None)
            if (
                source == "vision_depth"
                and detail.endswith("_hold")
                and sample_age is not None
                and float(sample_age) <= max(0.01, float(cfg.reverse_radar_max_age_sec))
            ):
                # 已经确认目标正在靠近并进入倒车后，允许沿用一帧新鲜的
                # Depth hold。若立即退出倒车，横向分支可能在同一近距离下
                # 改发向前差速，正是实车日志中 31RPM 倒车下一帧变 44/28RPM
                # 向前的原因。这里只维持既有倒车，hold 不能单独触发倒车。
                distance = float(frame.distance_m)
                fresh_reverse_distance = False
                logger.info(
                    "reverse_depth_hold_continue distance=%.2fm detail=%s age=%.0fms",
                    distance,
                    detail,
                    float(sample_age) * 1000.0,
                )
        if distance is None:
            if self._reverse_active:
                if self._reverse_missing_started_at is None:
                    self._reverse_missing_started_at = distance_now
                missing_age = max(0.0, distance_now - float(self._reverse_missing_started_at))
                hold_sec = max(0.0, float(cfg.reverse_distance_missing_hold_sec))
                if missing_age <= hold_sec:
                    hold_rpm = max(
                        int(cfg.reverse_visual_guard_rpm),
                        int(self._reverse_last_output_rpm),
                    )
                    hold_rpm = min(
                        int(cfg.reverse_max_rpm),
                        int(cfg.reverse_runtime_cap_rpm),
                        hold_rpm,
                    )
                    speed = self._forward_percent_for_rpm(hold_rpm, allow_below_min=True)
                    self.last_action_frame = int(frame_index)
                    return ControlDecision(
                        actions=[ControlAction.backward(speed, "reverse_distance_missing_hold")],
                        current_forward_percent=speed,
                        reason="reverse_distance_missing_hold",
                    )

                # Do not unlock reverse on an invalid Depth frame. Replace the
                # last motor target with zero RPM and wait for a fresh distance
                # that explicitly reaches reverse_stop_distance_m.
                self.last_action_frame = int(frame_index)
                return ControlDecision(
                    actions=[ControlAction.backward(0, "reverse_distance_missing_wait")],
                    current_forward_percent=0,
                    clear_action_queue=True,
                    reason="reverse_distance_missing_wait",
                )

            if visual_near_guard:
                # When the camera already proves that the target is very near,
                # keep one lateral mode while range is temporarily unavailable.
                # Starting backward and switching to parked rotation on the next
                # Depth frame creates a needless stop/reverse/rotate jerk.
                if cfg.near_distance_rotate_only_enable:
                    near_limit = max(
                        float(cfg.brake_distance_m),
                        float(cfg.near_distance_rotate_only_distance_m),
                    )
                    self._near_distance_rotation_only_active = True
                    self._near_distance_rotation_only_last_distance_m = near_limit
                    logger.info(
                        "near_distance_rotation_only_visual_latch frame=%d target=%d "
                        "detail=%s assumed_distance=%.2fm",
                        int(frame_index),
                        int(target.track_id),
                        visual_near_detail,
                        near_limit,
                    )
                    return self._near_distance_rotation_only_decision(
                        frame_index,
                        frame,
                        target,
                        near_limit,
                        near_limit,
                        target_steering_limit_rpm=target_steering_limit_rpm,
                        target_steerable=target_steerable,
                    )
                self._reverse_active = True
                self._reverse_target_id = int(target.track_id)
                self._reverse_missing_started_at = distance_now
                self._target_stop_latched = True
                guard_rpm = min(
                    int(cfg.reverse_max_rpm),
                    int(cfg.reverse_runtime_cap_rpm),
                    max(int(cfg.reverse_min_rpm), int(cfg.reverse_visual_guard_rpm)),
                )
                self._reverse_last_base_rpm = int(guard_rpm)
                self._reverse_last_feedforward_rpm = int(guard_rpm)
                self._reverse_last_output_rpm = int(guard_rpm)
                speed = self._forward_percent_for_rpm(guard_rpm, allow_below_min=True)
                self.last_action_frame = int(frame_index)
                logger.info(
                    "target_reverse_started frame=%d target=%d mode=visual_%s speed=%d%% output=%drpm",
                    int(frame_index),
                    int(target.track_id),
                    visual_near_detail,
                    speed,
                    guard_rpm,
                )
                return ControlDecision(
                    actions=[ControlAction.backward(speed, "visual_near_guard_reverse")],
                    current_forward_percent=speed,
                    clear_action_queue=True,
                    stop_action_execution=True,
                    reason="visual_near_guard_reverse",
                )

            self._reverse_approach_confirm_frames = 0
            self._reverse_last_approach_at = None
            # Keep the most recent trusted Depth briefly. The visual fallback
            # needs it to veto a tall-box false positive on intervening hold
            # frames; freshness is still bounded by reverse_radar_max_age_sec.
            return None

        brake_m = float(cfg.brake_distance_m)
        # Reverse hysteresis is independent from the 1.50m target center. It
        # may stop below target so coasting does not immediately overshoot into
        # forward motion, while still remaining above the IR-only brake range.
        stop_m = max(brake_m + 0.06, float(cfg.reverse_stop_distance_m))
        start_m = max(brake_m + 0.05, min(stop_m - 0.01, float(cfg.reverse_start_distance_m)))
        previous_distance = self._reverse_last_radar_distance_m
        self._reverse_missing_started_at = None
        if fresh_reverse_distance:
            approach_speed_m_s = self._update_reverse_approach_speed(
                float(distance),
                distance_now,
            )
        else:
            approach_speed_m_s = float(self._reverse_filtered_approach_speed_m_s)

        # A rejected near candidate is safety evidence only; do not use the
        # old anchor to start reverse. Stop until the candidate is confirmed.
        if str(getattr(frame.distance_state, "trigger", "")) == "brake_candidate":
            self._reset_reverse_control("unconfirmed_depth_safety_candidate", keep_last_distance=True)
            return ControlDecision(
                explicit_stop_requested=True,
                clear_action_queue=True,
                stop_action_execution=True,
                reason="depth_safety_candidate",
            )

        # 硬刹车优先级高于倒车。让现有距离锁存逻辑生成 distance_too_close。
        if distance < brake_m or bool(getattr(frame.distance_state, "brake_latched", False)):
            if not cfg.distance_parking_enable and cfg.reverse_enable:
                # Depth may command reverse below the nominal brake distance.
                # The IR gate remains the only hard stop in the board runtime.
                self._reverse_active = True
                self._reverse_target_id = int(target.track_id)
                speed = self._reverse_percent_for_distance(
                    distance,
                    approach_speed_m_s=approach_speed_m_s,
                    now=distance_now,
                    frame=frame,
                )
                self.last_action_frame = int(frame_index)
                return ControlDecision(
                    actions=[ControlAction.backward(speed, "target_approaching_reverse")],
                    current_forward_percent=speed,
                    clear_action_queue=True,
                    stop_action_execution=True,
                    reason="target_approaching_reverse",
                )
            self._reset_reverse_control("hard_close", keep_last_distance=True)
            return None

        if self._reverse_active:
            if self._reverse_target_id != int(target.track_id):
                self._reset_reverse_control("target_changed", keep_last_distance=True)
                if cfg.distance_parking_enable:
                    return ControlDecision(
                        explicit_stop_requested=True,
                        clear_action_queue=True,
                        stop_action_execution=True,
                        reason="reverse_target_changed_stop",
                    )
                # Target handoff is not an IR parking event. Let the new visual
                # target go through the normal steering/longitudinal decision.
                return None
            if distance >= stop_m:
                min_confirm_interval_sec = 0.02
                if (
                    self._reverse_release_last_confirm_at is None
                    or distance_now - float(self._reverse_release_last_confirm_at)
                    >= min_confirm_interval_sec
                ):
                    self._reverse_release_confirm_frames += 1
                    self._reverse_release_last_confirm_at = distance_now
                required_frames = max(1, int(cfg.reverse_confirm_frames))
                if self._reverse_release_confirm_frames >= required_frames:
                    self._reset_reverse_control("target_distance_restored", keep_last_distance=True)
                    if cfg.distance_parking_enable:
                        return ControlDecision(
                            explicit_stop_requested=True,
                            clear_action_queue=True,
                            stop_action_execution=True,
                            reason="reverse_target_distance_restored",
                        )
                    # Let the normal longitudinal branch take over on this frame.
                    return None
                logger.info(
                    "target_reverse_release_wait distance=%.2fm threshold=%.2fm "
                    "confirmations=%d/%d",
                    float(distance),
                    stop_m,
                    int(self._reverse_release_confirm_frames),
                    int(required_frames),
                )
            else:
                self._reverse_release_confirm_frames = 0
                self._reverse_release_last_confirm_at = None
            speed = self._reverse_percent_for_distance(
                distance,
                approach_speed_m_s=approach_speed_m_s,
                now=distance_now,
                frame=frame,
            )
            self.last_action_frame = int(frame_index)
            return ControlDecision(
                actions=[ControlAction.backward(speed, "target_approaching_reverse")],
                current_forward_percent=speed,
                reason="target_approaching_reverse",
            )

        required_frames = max(1, int(cfg.reverse_confirm_frames))
        immediate_m = max(
            brake_m + 0.05,
            min(start_m, float(cfg.reverse_immediate_distance_m)),
        )
        # 人物框过大时中心点不可信，但锁定人物的实时 Depth 仍可用于纵向
        # 安全退距。明显近距不再要求距离继续下降，但仍需要配置数量的独立
        # 新鲜样本确认换向；横向转向会在不可转向分支中封锁。
        immediate_close = bool(
            distance <= immediate_m
            or (force_immediate_close and distance < start_m)
        )
        approach_delta = (
            0.0
            if previous_distance is None
            else float(previous_distance) - float(distance)
        )
        if distance > start_m:
            self._reverse_approach_confirm_frames = 0
            self._reverse_last_approach_at = None
            return None

        min_delta = max(0.0, float(cfg.reverse_min_approach_delta_m))
        approach_now = time.monotonic()
        # Visual and Depth30 loops can evaluate one sensor sample close together.
        # Count confirmations at least 20ms apart so one Depth frame cannot be
        # mistaken for two independent direction-change confirmations.
        confirmation_is_new = bool(
            self._reverse_last_approach_at is None
            or approach_now - float(self._reverse_last_approach_at) >= 0.02
        )
        if (
            self._reverse_approach_confirm_frames > 0
            and self._reverse_last_approach_at is not None
            and approach_now - self._reverse_last_approach_at > 0.80
        ):
            self._reverse_approach_confirm_frames = 0
            self._reverse_last_approach_at = None
            confirmation_is_new = True

        if immediate_close:
            if confirmation_is_new:
                self._reverse_approach_confirm_frames += 1
                self._reverse_last_approach_at = approach_now
        else:
            if previous_distance is None:
                return None
            if approach_delta >= min_delta and confirmation_is_new:
                self._reverse_approach_confirm_frames += 1
                self._reverse_last_approach_at = approach_now
            elif approach_delta <= -min_delta:
                self._reverse_approach_confirm_frames = 0
                self._reverse_last_approach_at = None
        if self._reverse_approach_confirm_frames < required_frames:
            return None
        confirmation_count = required_frames

        self._reverse_active = True
        self._reverse_target_id = int(target.track_id)
        self._reverse_approach_confirm_frames = 0
        # 倒车是在目标距离停车锁存内受控执行；恢复到带回差的停止距离后
        # 立即交回普通距离控制，避免 1.50m 附近前进/后退快速反复切换。
        self._target_stop_latched = True
        self._target_release_started_at = None
        self._target_release_confirm_frames = 0
        self._update_target_stop_visual_reference(frame, target)
        speed = self._reverse_percent_for_distance(
            distance,
            approach_speed_m_s=approach_speed_m_s,
            now=distance_now,
            frame=frame,
        )
        self.last_action_frame = int(frame_index)
        logger.info(
            "target_reverse_started frame=%d target=%d distance=%.2fm delta=%.2fm "
            "approach=%.2fm/s confirmations=%d speed=%d%% output=%drpm base=%drpm "
            "feedforward=%drpm cap=%drpm mode=%s immediate_threshold=%.2fm",
            int(frame_index),
            int(target.track_id),
            float(distance),
            float(approach_delta),
            float(approach_speed_m_s),
            confirmation_count,
            speed,
            int(self._reverse_last_output_rpm),
            int(self._reverse_last_base_rpm),
            int(self._reverse_last_feedforward_rpm),
            min(int(cfg.reverse_max_rpm), int(cfg.reverse_runtime_cap_rpm)),
            "immediate_close" if immediate_close else "approach_confirmed",
            immediate_m,
        )
        return ControlDecision(
            actions=[ControlAction.backward(speed, "target_approaching_reverse")],
            current_forward_percent=speed,
            clear_action_queue=True,
            stop_action_execution=True,
            reason="target_approaching_reverse",
        )

    def _near_distance_rotation_only_decision(
        self,
        frame_index: int,
        frame: SensorFrame,
        target: PersonTarget,
        distance_m: float,
        distance_limit_m: float,
        target_steering_limit_rpm: Optional[float] = None,
        target_steerable: bool = True,
    ) -> ControlDecision:
        """Hold longitudinal speed at zero and recenter with in-place yaw only."""
        self._reset_reverse_control("near_distance_rotation_only", keep_last_distance=True)
        self._reset_distance_pid()
        limited_steering = bool(
            target_steering_limit_rpm is not None
            and float(target_steering_limit_rpm) > 0.0
        )
        steering_allowed = bool(target_steerable or limited_steering)
        cx, _cy = target.center
        edge = self._edge_type(cx, frame.width) if steering_allowed else "none"
        now = time.monotonic()
        motion_dx_ratio = 0.0
        projected_x_ratio = float(cx) / float(max(1, frame.width))
        target_image_rate_dps = None
        if target_steerable:
            (
                motion_dx_ratio,
                projected_x_ratio,
                target_image_rate_dps,
                motion_dt_sec,
            ) = self._record_visible_motion(
                frame_index,
                target,
                frame.width,
                now,
            )
            if motion_dt_sec <= 0.0:
                target_image_rate_dps = None
        settling = self._near_settle_hold_active(
            target,
            frame,
            now,
            motion_dx_ratio=motion_dx_ratio,
            target_image_rate_dps=target_image_rate_dps,
        )
        soft_zero_hold = False
        near_yaw_park_requested = False
        park_reason = "none"
        if settling:
            # Do not run the PID while the center hold is active. Running it
            # here would re-arm startup_kick on every small image crossing.
            self._parked_recenter_pid.reset()
            self._visual_steering_pid.reset()
            self.last_steering_pid_result = None
            action = ControlAction.stop("near_distance_center_settle", brake_hold=True)
            near_yaw_park_requested = True
            park_reason = "near_distance_center_settle"
        else:
            action = self._pid_action_for_parked_target(
                target,
                frame,
                now,
                edge,
                motion_dx_ratio=motion_dx_ratio,
                projected_x_ratio=projected_x_ratio,
                target_image_rate_dps=target_image_rate_dps,
                max_correction_rpm=(
                    target_steering_limit_rpm
                    if limited_steering
                    else None
                    if target_steerable
                    else 0.0
                ),
                near_distance_mode=True,
            )
        # A zero PID result is an intentional coast/brake decision when the
        # measured yaw rate is already sufficient. Do not replace it with the
        # legacy minimum-RPM edge fallback, or the chassis keeps accelerating
        # after the inner loop has asked it to slow down.
        pid_produced_zero = bool(
            self.cfg.visible_steering_pid_enable
            and self.last_steering_pid_result is not None
            and int(self.last_steering_pid_result.correction_rpm) == 0
        )
        if action is None and pid_produced_zero:
            # This branch has already set the longitudinal target to zero.
            # An explicit predicted stop / center hold therefore means park
            # the chassis, not repeatedly re-enter the zero-speed loop. Other
            # PID zeroes are still transient yaw updates, not parking events.
            result = self.last_steering_pid_result
            floor_reason = str(result.output_floor_reason)
            near_yaw_park_requested = bool(
                result.predictive_braking
                or floor_reason in {"predictive_brake_coast", "center_hold"}
            )
            if near_yaw_park_requested:
                park_reason = (
                    "center_hold"
                    if floor_reason == "center_hold"
                    else "predictive_brake"
                )
            action = ControlAction.stop(
                "person_parked_pid_zero_hold",
                brake_hold=near_yaw_park_requested,
            )
            soft_zero_hold = not near_yaw_park_requested
        elif (
            action is not None
            and action.kind == "stop"
            and action.reason == "person_parked_direction_guard_hold"
        ):
            # Direction protection is a visual-control guard, not a hazard.
            # Coast to zero while waiting for the next fresh frame; do not
            # engage the global brake latch for a recoverable PID decision.
            action = ControlAction.stop(
                "person_parked_direction_guard_hold",
                brake_hold=False,
            )
            soft_zero_hold = True
        elif action is None and edge in ("left", "right"):
            action = self._rotate_action_for_edge(edge)
        if action is None:
            self._parked_recenter_pid.reset()
            self._visual_steering_pid.reset()
            self.last_steering_pid_result = None
            action = ControlAction.stop("near_distance_rotation_only_hold")
            # A disabled steering branch can also have edge="none" while its
            # bbox is off-center. Do not turn that quality guard into a
            # center-parking intent merely because correction was disallowed.
            if self._edge_type(cx, frame.width) == "none":
                near_yaw_park_requested = True
                park_reason = "center_fallback_hold"

        interrupt_existing = self.last_dispatched_kind != action.kind
        # A visible near-distance PID update is a continuous signed yaw
        # command.  Reversing its sign must replace the queued target without
        # raising stop_action_execution, otherwise the action thread inserts a
        # TURN_ZERO between every correction and the chassis rocks in place.
        continuous_rotate_switch = bool(
            interrupt_existing
            and self.last_dispatched_kind in ("rotate_left", "rotate_right")
            and action.kind in ("rotate_left", "rotate_right")
        )
        # A soft PID zero leaves the executor in an ordinary zero-output
        # state.  Resuming a near-distance rotate from that state must not
        # synthesize a queued stop/brake transition.
        near_rotate_resume = bool(
            interrupt_existing
            and action.kind in ("rotate_left", "rotate_right")
            and self.last_dispatched_kind == "stop"
        )
        self.last_action_frame = int(frame_index)
        logger.info(
            "near_distance_rotation_only distance=%.2fm limit=%.2fm brake=%.2fm "
            "steerable=%s limited=%s correction_limit=%s edge=%s action=%s soft_zero=%s "
            "park_requested=%s park_reason=%s",
            float(distance_m),
            float(distance_limit_m),
            float(self.cfg.brake_distance_m),
            bool(target_steerable),
            limited_steering,
            "none" if target_steering_limit_rpm is None else "%.1f" % float(target_steering_limit_rpm),
            edge,
            action.kind,
            soft_zero_hold,
            near_yaw_park_requested,
            park_reason,
        )
        return ControlDecision(
            actions=[self._retag_action(action, "near_distance_rotation_only")],
            current_forward_percent=0,
            clear_action_queue=interrupt_existing,
            stop_action_execution=(
                interrupt_existing
                and not continuous_rotate_switch
                and not near_rotate_resume
                and not soft_zero_hold
            ),
            soft_stop_requested=soft_zero_hold,
            reason="near_distance_rotation_only",
            near_yaw_park_requested=near_yaw_park_requested,
        )

    def _reset_near_settle(self) -> None:
        self._near_settle_target_id = None
        self._near_settle_confirm_frames = 0
        self._near_settle_release_frames = 0
        self._near_settle_until = 0.0

    def _near_settle_hold_active(
        self,
        target: PersonTarget,
        frame: SensorFrame,
        now: float,
        *,
        motion_dx_ratio: float = 0.0,
        target_image_rate_dps: Optional[float] = None,
    ) -> bool:
        """Hold zero yaw briefly after a stable near-target center crossing.

        A center hold is only meant to absorb residual chassis yaw.  If the
        target is already moving out of the center corridor, waiting for the
        full release hysteresis adds a visible dead time.  Release early only
        when position and image motion agree on the same outward direction;
        static detector jitter therefore keeps the original hold behavior.
        """
        target_id = int(target.track_id)
        if self._near_settle_target_id != target_id:
            self._reset_near_settle()
            self._near_settle_target_id = target_id

        x_ratio = float(target.center[0]) / float(max(1, frame.width))
        left = float(self.cfg.center_left_ratio)
        right = float(self.cfg.center_right_ratio)
        margin = max(0.01, float(self.cfg.near_distance_settle_release_margin_ratio))
        outside_release = x_ratio < left - margin or x_ratio > right + margin

        if self._near_settle_until > now:
            outward_motion = bool(
                (x_ratio > right and motion_dx_ratio >= 0.004)
                or (x_ratio < left and motion_dx_ratio <= -0.004)
                or (
                    target_image_rate_dps is not None
                    and math.isfinite(float(target_image_rate_dps))
                    and (
                        (x_ratio > right and float(target_image_rate_dps) >= 2.0)
                        or (x_ratio < left and float(target_image_rate_dps) <= -2.0)
                    )
                )
            )
            if outward_motion:
                self._near_settle_until = 0.0
                self._near_settle_confirm_frames = 0
                self._near_settle_release_frames = 0
                logger.info(
                    "near_distance_center_settle_release_motion target=%d x=%.3f "
                    "dx=%+.4f image_rate=%s",
                    target_id,
                    x_ratio,
                    float(motion_dx_ratio),
                    "none"
                    if target_image_rate_dps is None
                    else f"{float(target_image_rate_dps):+.2f}dps",
                )
                return False
            if outside_release:
                self._near_settle_release_frames += 1
                if self._near_settle_release_frames >= max(
                    1, int(self.cfg.near_distance_settle_release_frames)
                ):
                    self._near_settle_until = 0.0
                    self._near_settle_confirm_frames = 0
                    self._near_settle_release_frames = 0
                    return False
            else:
                self._near_settle_release_frames = 0
            return True

        if outside_release:
            self._near_settle_confirm_frames = 0
            self._near_settle_release_frames = 0
            return False

        feedback = frame.steering_feedback
        feedback_ok = bool(
            feedback is not None
            and feedback.trustworthy
            and max(0.0, now - float(feedback.timestamp))
            <= max(0.05, float(self.cfg.visible_steering_pid_feedback_stale_sec))
            and math.isfinite(float(feedback.yaw_rate_right_dps))
            and abs(float(feedback.yaw_rate_right_dps)) <= 3.0
        )
        if left <= x_ratio <= right and feedback_ok:
            self._near_settle_confirm_frames += 1
        else:
            self._near_settle_confirm_frames = 0
        if self._near_settle_confirm_frames >= max(
            1, int(self.cfg.near_distance_settle_confirm_frames)
        ):
            self._near_settle_until = now + max(
                0.05, float(self.cfg.near_distance_settle_hold_sec)
            )
            self._near_settle_release_frames = 0
            logger.info(
                "near_distance_center_settle target=%d x=%.3f hold_ms=%.0f",
                target_id,
                x_ratio,
                float(self.cfg.near_distance_settle_hold_sec) * 1000.0,
            )
            return True
        return False

    def _longitudinal_only_decision(
        self,
        frame_index: int,
        frame: SensorFrame,
        target: Optional[PersonTarget],
        *,
        target_steerable: bool,
    ) -> ControlDecision:
        """Update only forward/reverse speed for the independent Depth loop."""
        if target is not None and int(target.track_id) == self.active_target_id and self._older_depth_observation(frame):
            return ControlDecision(reason="longitudinal_old_depth_observation")
        self._observe_longitudinal_motion(frame, target, allowed=target_steerable)
        if target is None:
            return ControlDecision(reason="longitudinal_target_unavailable")

        now = time.monotonic()

        if self.distance_pi_enabled:
            return self._distance_pi_longitudinal_decision(
                frame_index, frame, target, bool(target_steerable), now)

        # The 30Hz Depth supervisor owns longitudinal speed only.  When the
        # parked/near-distance policy is active, running the visual yaw PID a
        # second time with a stale bbox/encoder sample makes the two loops
        # alternate rotate_left/rotate_right.  Leave the latest camera-loop
        # rotation untouched; the next visual frame remains the sole yaw owner.
        if self.cfg.near_distance_rotate_only_enable:
            near_distance = frame.distance_m
            # Held `used_distance_m` is not fresh enough to activate the
            # near-distance rotation policy.
            fresh_depth = self._is_fresh_depth_state(frame)
            if near_distance is None and fresh_depth:
                near_distance = getattr(frame.distance_state, "used_distance_m", None)
            if near_distance is not None and (frame.distance_m is not None or fresh_depth):
                near_limit = max(
                    float(self.cfg.brake_distance_m),
                    float(self.cfg.near_distance_rotate_only_distance_m),
                )
                if (
                    float(self.cfg.brake_distance_m) <= float(near_distance) <= near_limit
                    and self._tracking_base_rpm(float(near_distance), now) is None
                ):
                    self._forward_active = False
                    self._reset_distance_pid()
                    return ControlDecision(
                        actions=[ControlAction.forward(0, "longitudinal_near_rotation_hold")],
                        current_forward_percent=0,
                        reason="longitudinal_near_rotation_hold",
                    )

        # A fresh Depth sample can be produced after the target-distance
        # anchor expires.  The sensor runtime labels this re-anchor and any
        # jump-confirmation sample explicitly; neither is safe for the
        # longitudinal PID because it may be a far background return.  Stop
        # the previous longitudinal command and let the camera loop reacquire
        # a trustworthy range before allowing forward motion again.
        if self._distance_longitudinally_untrusted(frame):
            self._note_depth_quality_failure(frame, now)
            # A fused Depth hold can contain a short encoder prediction even
            # though the current ROI had too few valid pixels. Keep the car
            # moving very slowly for this bounded window; a stale/background
            # re-anchor, jump-pending sample, or expired sample still stops.
            short_hold = self._distance_missing_camera_hold_action(
                frame,
                now,
                max_depth_hold_sec=0.18,
            )
            if short_hold is not None:
                self.last_action_frame = frame_index
                return ControlDecision(
                    actions=[short_hold],
                    is_forwarding=True,
                    current_forward_percent=short_hold.speed_percent,
                    reason="longitudinal_distance_short_hold",
                )
            self._forward_active = False
            self._reset_distance_pid()
            return ControlDecision(
                # This controller invocation owns the longitudinal axis only.
                # A zero-forward action stops forward/reverse motion without
                # claiming that visible-target yaw must also be stopped.
                actions=[ControlAction.forward(0, "longitudinal_distance_untrusted_hold")],
                current_forward_percent=0,
                clear_action_queue=True,
                reason="longitudinal_distance_untrusted_hold",
            )

        reverse_decision = self._reverse_control_decision(
            frame_index,
            frame,
            target,
            force_immediate_close=not bool(target_steerable),
            target_steering_limit_rpm=None,
            target_steerable=bool(target_steerable),
        )
        if reverse_decision is not None:
            return reverse_decision

        distance = frame.distance_m
        if distance is None:
            if self._longitudinal_missing_started_at is None:
                self._longitudinal_missing_started_at = now
            anchor_age = (
                None
                if self._last_target_distance_at is None
                else max(0.0, now - float(self._last_target_distance_at))
            )
            self._note_depth_quality_failure(frame, now)
            if (
                anchor_age is not None
                and anchor_age <= max(0.20, float(self.cfg.depth_medium_confidence_hold_sec))
            ):
                return ControlDecision(reason="longitudinal_distance_missing_keep")
            self._forward_active = False
            self._reset_distance_pid()
            return ControlDecision(
                actions=[ControlAction.forward(0, "longitudinal_distance_low_confidence_stop")],
                current_forward_percent=0,
                clear_action_queue=True,
                reason="longitudinal_distance_low_confidence_stop",
            )

        self._longitudinal_missing_started_at = None
        self._remember_target_distance(frame, now)
        speed = self._forward_percent_for_distance(float(distance), now=now)
        speed = self._limit_depth_quality_forward_percent(frame, speed, now)
        if speed <= 0:
            return ControlDecision(
                actions=[ControlAction.forward(0, "longitudinal_distance_hold")],
                current_forward_percent=0,
                clear_action_queue=True,
                reason="longitudinal_distance_hold",
            )
        return ControlDecision(
            actions=[ControlAction.forward(speed, "longitudinal_distance_pid")],
            is_forwarding=True,
            current_forward_percent=speed,
            reason="longitudinal_distance_pid",
        )

    def _target_distance_lock_decision(
        self,
        frame: SensorFrame,
        now: float,
        *,
        off_center: bool,
        target: Optional[PersonTarget] = None,
        person_detected_flag: bool = False,
        clear_action_queue: bool = True,
        stop_action_execution: bool = True,
    ) -> Optional[ControlDecision]:
        """Apply target-distance stop latching and release hysteresis.

        A single reading above the follow threshold is not sufficient to
        restart the vehicle.  This protects against a near radar return being
        replaced for one frame by a distant return or a stale cached sample.
        """
        cfg = self.cfg
        if not cfg.distance_parking_enable:
            # Do not let distance create STOP/brake-hold in the active board path.
            if self._target_stop_latched:
                logger.info("distance parking latch bypassed: policy=ir_only")
            self._target_stop_latched = False
            self._target_release_started_at = None
            self._target_release_confirm_frames = 0
            self._clear_target_stop_visual_reference()
            return None
        distance = frame.distance_m
        if distance is None:
            distance = getattr(frame.distance_state, "used_distance_m", None)
        target_m = float(cfg.target_distance_m)
        brake_m = float(cfg.brake_distance_m)
        # 停车后的恢复距离由配置直接决定；目标距离为 1.5m 时，达到 1.5m 即可进入释放确认。
        release_m = max(target_m, float(cfg.target_distance_release_m))
        suffix = "_off_center" if off_center else ""

        if not self._target_stop_latched:
            if distance is None or float(distance) > target_m:
                return None
            self._clear_target_stop_visual_reference()
            self._target_stop_latched = True
            self._target_release_started_at = None
            self._target_release_confirm_frames = 0
            logger.info(
                "target_distance_stop_latched distance=%.2fm threshold=%.2fm release=%.2fm",
                float(distance),
                target_m,
                release_m,
            )

        self._update_target_stop_visual_reference(frame, target)

        if distance is None:
            self._reset_distance_pid()
            self._target_release_started_at = None
            self._target_release_confirm_frames = 0
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=person_detected_flag,
                clear_action_queue=clear_action_queue,
                stop_action_execution=stop_action_execution,
                reason="target_distance_hold_missing",
            )

        distance = float(distance)
        if distance < brake_m:
            self._reset_distance_pid()
            self._target_release_started_at = None
            self._target_release_confirm_frames = 0
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=person_detected_flag,
                clear_action_queue=clear_action_queue,
                stop_action_execution=stop_action_execution,
                reason="person_too_close_no_rotate" if off_center else "distance_too_close",
            )

        if distance <= target_m:
            self._reset_distance_pid()
            self._target_release_started_at = None
            self._target_release_confirm_frames = 0
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=person_detected_flag,
                clear_action_queue=clear_action_queue,
                stop_action_execution=stop_action_execution,
                reason="target_distance_reached" + suffix,
            )

        if distance < release_m:
            self._reset_distance_pid()
            self._target_release_started_at = None
            self._target_release_confirm_frames = 0
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=person_detected_flag,
                clear_action_queue=clear_action_queue,
                stop_action_execution=stop_action_execution,
                reason="target_distance_hold" + suffix,
            )

        visual_release_ready, visual_release_detail = self._vision_depth_release_evidence(
            frame,
            target,
        )
        if not visual_release_ready:
            self._reset_distance_pid()
            self._target_release_started_at = None
            self._target_release_confirm_frames = 0
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=person_detected_flag,
                clear_action_queue=clear_action_queue,
                stop_action_execution=stop_action_execution,
                reason=(
                    "target_distance_release_visual_wait_"
                    + visual_release_detail
                    + suffix
                ),
            )

        if self._target_release_started_at is None:
            self._target_release_started_at = float(now)
        self._target_release_confirm_frames += 1
        required_frames = max(1, int(cfg.target_distance_release_confirm_frames))
        elapsed = max(0.0, float(now) - float(self._target_release_started_at))
        required_sec = max(0.0, float(cfg.target_distance_release_hold_sec))
        if self._target_release_confirm_frames < required_frames or elapsed < required_sec:
            self._reset_distance_pid()
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=person_detected_flag,
                clear_action_queue=clear_action_queue,
                stop_action_execution=stop_action_execution,
                reason="target_distance_release_wait" + suffix,
            )

        self._target_stop_latched = False
        self._target_release_started_at = None
        self._target_release_confirm_frames = 0
        self._clear_target_stop_visual_reference()
        logger.info(
            "target_distance_stop_released distance=%.2fm release=%.2fm hold=%.2fs frames=%d",
            distance,
            release_m,
            elapsed,
            required_frames,
        )
        return None

    def _reset_search_timeout(self) -> None:
        self._search_rotation_started_at = None
        self._search_rotation_accumulated_deg = 0.0
        self._search_rotation_last_integrated_yaw_deg = None
        self._search_rotation_origin_integrated_yaw_deg = None
        self._search_heading_from_loss_deg = 0.0
        self._search_heading_min_deg = 0.0
        self._search_heading_max_deg = 0.0
        self._search_rotation_feedback_last_ts = None
        self._search_rotation_feedback_seen = False
        self._last_search_rotation_log_deg = 0.0

    def _begin_search_rotation_measurement(self, frame: SensorFrame) -> None:
        """Take the encoder yaw at target loss as the zero of this scan."""
        feedback = frame.steering_feedback
        integrated = None if feedback is None else getattr(
            feedback, "integrated_yaw_right_deg", None
        )
        try:
            integrated = float(integrated)
        except (TypeError, ValueError):
            integrated = None
        if integrated is None or not math.isfinite(integrated):
            self._search_rotation_last_integrated_yaw_deg = None
            self._search_rotation_origin_integrated_yaw_deg = None
            self._search_rotation_feedback_last_ts = None
            self._search_rotation_feedback_seen = False
            logger.info("search rotation measurement unavailable: using timeout fallback")
            return
        self._search_rotation_accumulated_deg = 0.0
        self._search_rotation_last_integrated_yaw_deg = integrated
        self._search_rotation_origin_integrated_yaw_deg = integrated
        self._search_heading_from_loss_deg = 0.0
        self._search_heading_min_deg = 0.0
        self._search_heading_max_deg = 0.0
        self._search_rotation_feedback_last_ts = time.monotonic()
        self._search_rotation_feedback_seen = True
        self._last_search_rotation_log_deg = 0.0
        logger.info(
            "search rotation measurement start: yaw_zero=%.2fdeg target=%.1fdeg",
            integrated,
            max(90.0, float(self.cfg.search_revolution_deg)),
        )

    def _update_search_rotation_progress(self, frame: SensorFrame) -> float:
        """Accumulate absolute encoder yaw travelled during the current scan."""
        if not self._search_rotation_feedback_seen:
            return float(self._search_rotation_accumulated_deg)
        feedback = frame.steering_feedback
        integrated = None if feedback is None else getattr(
            feedback, "integrated_yaw_right_deg", None
        )
        try:
            integrated = float(integrated)
        except (TypeError, ValueError):
            integrated = None
        if integrated is None or not math.isfinite(integrated):
            return float(self._search_rotation_accumulated_deg)
        previous = self._search_rotation_last_integrated_yaw_deg
        self._search_rotation_last_integrated_yaw_deg = integrated
        self._search_rotation_feedback_last_ts = time.monotonic()
        if previous is None:
            return float(self._search_rotation_accumulated_deg)
        delta = integrated - float(previous)
        # A reconnect or invalid encoder sample can jump by hundreds of degrees;
        # never let one bad sample falsely complete the revolution.
        if abs(delta) <= 90.0:
            self._search_rotation_accumulated_deg += abs(delta)
            origin = self._search_rotation_origin_integrated_yaw_deg
            if origin is not None:
                self._search_heading_from_loss_deg = integrated - float(origin)
                self._search_heading_min_deg = min(
                    float(self._search_heading_min_deg),
                    float(self._search_heading_from_loss_deg),
                )
                self._search_heading_max_deg = max(
                    float(self._search_heading_max_deg),
                    float(self._search_heading_from_loss_deg),
                )
        else:
            logger.warning(
                "search rotation encoder jump ignored: previous=%.2f current=%.2f delta=%.2f",
                float(previous),
                integrated,
                delta,
            )
        if self._search_rotation_accumulated_deg - self._last_search_rotation_log_deg >= 45.0:
            self._last_search_rotation_log_deg = self._search_rotation_accumulated_deg
            logger.info(
                "search rotation progress: travel=%.1fdeg coverage=%.1f/%.1fdeg "
                "heading_from_loss=%.1fdeg feedback_yaw=%.2fdeg direction=%s mode=single_direction",
                self._search_rotation_accumulated_deg,
                max(0.0, self._search_heading_max_deg - self._search_heading_min_deg),
                max(90.0, float(self.cfg.search_revolution_deg)),
                self._search_heading_from_loss_deg,
                integrated,
                self.search_direction or self._lost_exit_direction or "unknown",
            )
        return float(self._search_rotation_accumulated_deg)

    def defer_search_timeout(self, duration_sec: float) -> None:
        """Exclude a short ReID candidate observation from lost-search time."""
        duration = max(0.0, float(duration_sec))
        if duration <= 0.0:
            return
        now = time.monotonic()
        if self._lost_started_at is not None:
            self._lost_started_at = min(now, float(self._lost_started_at) + duration)
        if self._search_rotation_started_at is not None:
            self._search_rotation_started_at = min(
                now,
                float(self._search_rotation_started_at) + duration,
            )

    def _mark_search_rotation_started(self, now: float) -> None:
        if self._search_rotation_started_at is None:
            self._search_rotation_started_at = float(now)

    def _search_timeout_decision(
        self,
        now: float,
        steering_feedback: Optional[object] = None,
    ) -> Optional[ControlDecision]:
        timeout_sec = max(0.0, float(self.cfg.search_timeout_sec))
        # 总搜索时间从原目标第一次消失开始计算，5 帧确认窗口也包含在内，
        # 避免“先等待确认，再额外搜索 timeout_sec”导致实际退出明显超时。
        started_at = self._lost_started_at
        if started_at is None:
            started_at = self._search_rotation_started_at
        if timeout_sec <= 0.0 or started_at is None:
            return None
        # With fresh encoder feedback, the revolution gate owns termination.
        # Do not let the legacy wall-clock timeout cut a slow 360-degree scan
        # short; stale feedback falls back to the timeout below.
        feedback_fresh = False
        if steering_feedback is not None:
            try:
                feedback_fresh = bool(getattr(steering_feedback, "trustworthy", False)) and (
                    now - float(getattr(steering_feedback, "timestamp"))
                    <= max(0.10, float(self.cfg.search_revolution_feedback_stale_sec))
                )
            except (TypeError, ValueError):
                feedback_fresh = False
        if feedback_fresh:
            self._search_rotation_feedback_last_ts = now
            self._search_rotation_feedback_seen = True
            return None
        if (
            self._search_rotation_feedback_seen
            and self._search_rotation_feedback_last_ts is not None
            and now - float(self._search_rotation_feedback_last_ts)
            <= max(0.10, float(self.cfg.search_revolution_feedback_stale_sec))
        ):
            return None
        elapsed = max(0.0, float(now) - float(started_at))
        if elapsed < timeout_sec:
            return None
        first_stop = self.search_state != "timed_out"
        self.search_state = "timed_out"
        self.search_direction = None
        if first_stop:
            logger.info(
                "search_timeout_%s elapsed=%.2fs threshold=%.2fs active_target=%s",
                "exit" if self.cfg.search_timeout_exit_program else "stop",
                elapsed,
                timeout_sec,
                "none" if self.active_target_id is None else int(self.active_target_id),
            )
        exit_program = bool(self.cfg.search_timeout_exit_program)
        return ControlDecision(
            explicit_stop_requested=True,
            clear_action_queue=first_stop,
            stop_action_execution=first_stop,
            shutdown_requested=exit_program,
            reason="search_timeout_exit" if exit_program else "search_timeout_stop",
        )

    def _search_revolution_complete_decision(self, now: float) -> Optional[ControlDecision]:
        required = max(90.0, float(self.cfg.search_revolution_deg))
        travelled = float(self._search_rotation_accumulated_deg)
        coverage = max(
            0.0,
            float(self._search_heading_max_deg) - float(self._search_heading_min_deg),
        )
        measured = coverage if self._search_rotation_feedback_seen else travelled
        if measured < required:
            return None
        first_stop = self.search_state != "timed_out"
        self.search_state = "timed_out"
        self.search_direction = None
        if first_stop:
            logger.info(
                "search_revolution_complete: coverage=%.1fdeg travelled=%.1fdeg "
                "target=%.1fdeg elapsed=%.2fs mode=single_direction",
                coverage,
                travelled,
                required,
                0.0
                if self._search_rotation_started_at is None
                else max(0.0, float(now) - float(self._search_rotation_started_at)),
            )
        return ControlDecision(
            explicit_stop_requested=True,
            clear_action_queue=first_stop,
            stop_action_execution=first_stop,
            shutdown_requested=True,
            reason="search_revolution_complete",
        )

    def _confirm_initial_target(self, target: PersonTarget, frame_index: int) -> bool:
        confirm_frames = max(1, int(self.cfg.initial_target_confirm_frames))
        target_id = int(target.track_id)
        if self._initial_candidate_id == target_id:
            self._initial_candidate_frames += 1
        else:
            self._initial_candidate_id = target_id
            self._initial_candidate_frames = 1

        if self._initial_candidate_frames < confirm_frames:
            logger.info(
                "initial_target_confirm_wait frame=%d candidate=%d streak=%d/%d",
                int(frame_index),
                target_id,
                int(self._initial_candidate_frames),
                int(confirm_frames),
            )
            return False

        logger.info(
            "initial_target_confirmed frame=%d candidate=%d streak=%d/%d",
            int(frame_index),
            target_id,
            int(self._initial_candidate_frames),
            int(confirm_frames),
        )
        self._reset_initial_target_confirm()
        return True

    def _retag_action(self, action: ControlAction, reason: str) -> ControlAction:
        return ControlAction(
            kind=action.kind,
            speed_percent=action.speed_percent,
            steer_inner_ratio_percent=action.steer_inner_ratio_percent,
            steer_outer_ratio_percent=action.steer_outer_ratio_percent,
            steer_correction_rpm=action.steer_correction_rpm,
            reason=reason,
            brake_hold=action.brake_hold,
        )

    def _reset_visible_steer_memory(self) -> None:
        self._last_visible_steer_action = None
        self._last_visible_steer_started_at = None

    def _reset_visible_motion(self) -> None:
        self._visible_motion_target_id = None
        self._visible_motion_samples = []
        self._visible_motion_filtered_rate_ratio_s = 0.0

    def _record_visible_motion(
        self,
        frame_index: int,
        target: PersonTarget,
        width: int,
        now: float,
    ) -> tuple:
        """Estimate timestamped image motion for edge prediction and yaw feedforward."""
        if width <= 0:
            return 0.0, 0.5, 0.0, 0.0

        target_id = int(target.track_id)
        history_frames = max(2, int(self.cfg.visible_motion_history_frames))
        lookback_sec = max(0.04, min(0.30, float(self.cfg.visible_motion_lookback_sec)))
        max_sample_gap_sec = max(0.35, 3.0 * lookback_sec)
        if self._visible_motion_target_id != target_id:
            self._visible_motion_target_id = target_id
            self._visible_motion_samples = []
            self._visible_motion_filtered_rate_ratio_s = 0.0
        elif self._visible_motion_samples:
            last_frame = int(self._visible_motion_samples[-1][0])
            last_ts = float(self._visible_motion_samples[-1][1])
            if (
                int(frame_index) - last_frame > history_frames
                or float(now) - last_ts > max_sample_gap_sec
            ):
                # 长时间漏检后的第一帧不能和旧位置计算速度，否则会产生虚假的大幅预转向。
                self._visible_motion_samples = []
                self._visible_motion_filtered_rate_ratio_s = 0.0

        cx, _cy = target.center
        x_ratio = max(0.0, min(1.0, float(cx) / float(width)))
        sample = (int(frame_index), float(now), x_ratio)
        if self._visible_motion_samples and int(self._visible_motion_samples[-1][0]) == int(frame_index):
            self._visible_motion_samples[-1] = sample
        else:
            self._visible_motion_samples.append(sample)
        self._visible_motion_samples = self._visible_motion_samples[-history_frames:]

        if len(self._visible_motion_samples) < 2:
            return 0.0, x_ratio, 0.0, 0.0

        candidates = []
        for old_sample in self._visible_motion_samples[:-1]:
            sample_age = float(now) - float(old_sample[1])
            if 0.025 <= sample_age <= max_sample_gap_sec:
                candidates.append((abs(sample_age - lookback_sec), old_sample, sample_age))
        if not candidates:
            return 0.0, x_ratio, 0.0, 0.0
        _distance, reference, sample_dt_sec = min(candidates, key=lambda item: item[0])
        first_x_ratio = float(reference[2])
        dx_ratio = x_ratio - first_x_ratio
        min_motion = max(0.0, float(self.cfg.visible_motion_min_ratio))
        raw_rate_ratio_s = 0.0 if abs(dx_ratio) < min_motion else dx_ratio / sample_dt_sec
        hfov_deg = max(1.0, float(self.cfg.visible_steering_pid_camera_hfov_deg))
        # This is a measurement limit, not the feedforward-output limit. When
        # the chassis yaws at 35-45 dps, a stationary target moves through the
        # image at the same rate. Capping that measurement at 15 dps makes
        # image_rate + encoder_yaw look like target motion and sustains yaw.
        max_rate_dps = max(
            15.0,
            float(self.cfg.visible_steering_pid_max_yaw_rate_dps)
            + float(self.cfg.visible_steering_pid_target_rate_feedforward_max_dps),
        )
        max_rate_ratio_s = max_rate_dps / hfov_deg
        raw_rate_ratio_s = max(-max_rate_ratio_s, min(max_rate_ratio_s, raw_rate_ratio_s))
        rate_alpha = max(0.05, min(1.0, float(self.cfg.visible_motion_rate_filter_alpha)))
        self._visible_motion_filtered_rate_ratio_s = (
            rate_alpha * raw_rate_ratio_s
            + (1.0 - rate_alpha) * self._visible_motion_filtered_rate_ratio_s
        )
        projection_gain = max(0.0, min(3.0, float(self.cfg.visible_motion_projection_gain)))
        # Predict far enough to cover the age of the camera result plus one
        # motion-estimation horizon. This projection only selects an edge state;
        # the PID itself still receives the unmodified current x position.
        projection_sec = max(
            0.0,
            float(self.cfg.visible_steering_pid_camera_latency_sec),
        ) + lookback_sec * projection_gain
        projected_x_ratio = max(
            0.0,
            min(
                1.0,
                x_ratio + self._visible_motion_filtered_rate_ratio_s * projection_sec,
            ),
        )
        image_rate_dps = self._visible_motion_filtered_rate_ratio_s * hfov_deg
        return dx_ratio, projected_x_ratio, image_rate_dps, sample_dt_sec

    def _visible_motion_edge_type(
        self,
        dx_ratio: float,
        current_x_ratio: float,
        projected_x_ratio: float,
        width: int,
    ) -> str:
        min_motion = max(0.0, float(self.cfg.visible_motion_min_ratio))
        if width <= 0 or abs(float(dx_ratio)) < min_motion:
            return "none"
        left, right = self._active_center_band(width)
        left_ratio = float(left) / float(width)
        right_ratio = float(right) / float(width)
        # 趋势预判最多提前一个 min_motion 区间介入。车身刚把人物从一侧拉回中心时，
        # 画面位移会很大；若允许从中心直接反向强修，就会形成左右来回摆动。
        if dx_ratio < 0.0 and current_x_ratio <= left_ratio + min_motion and projected_x_ratio < left_ratio:
            return "left"
        if dx_ratio > 0.0 and current_x_ratio >= right_ratio - min_motion and projected_x_ratio >= right_ratio:
            return "right"
        return "none"

    def _remember_visible_steer(self, action: ControlAction, now: float) -> None:
        if action.kind not in ("steer_left", "steer_right"):
            self._reset_visible_steer_memory()
            return
        previous = self._last_visible_steer_action
        if (
            previous is None
            or previous.kind != action.kind
            or self.last_dispatched_kind != action.kind
        ):
            self._last_visible_steer_started_at = float(now)
        self._last_visible_steer_action = action

    def _held_steer_action_for_lost(
        self,
        frame: SensorFrame,
        now: float,
        *,
        rotation_only: bool = False,
    ) -> Optional[ControlAction]:
        action = self._last_visible_steer_action
        started_at = self._last_visible_steer_started_at
        if action is None or started_at is None:
            return None
        if action.kind not in ("steer_left", "steer_right"):
            return None
        if self.last_dispatched_kind != action.kind:
            return None

        lost_started_at = self._lost_started_at
        if lost_started_at is None:
            return None
        # lost_confirm_frames 是硬上限；达到确认帧时必须先停车，不能因为
        # steer_min_hold_sec 尚未到期而继续盲目差速转向。
        if self.lost_confirm_frames >= max(1, int(self.cfg.lost_confirm_frames)):
            return None
        lost_elapsed_sec = max(0.0, float(now) - float(lost_started_at))
        max_lost_hold_sec = max(0.0, float(self.cfg.steer_lost_hold_max_sec))
        if rotation_only:
            max_lost_hold_sec = min(max_lost_hold_sec, 0.20)
        if max_lost_hold_sec <= 0.0 or lost_elapsed_sec > max_lost_hold_sec:
            return None

        steer_elapsed_sec = max(0.0, float(now) - float(started_at))
        within_min_hold = steer_elapsed_sec < max(0.0, float(self.cfg.steer_min_hold_sec))
        frame_hold_limit = max(0, int(self.cfg.steer_lost_hold_frames))
        if rotation_only:
            frame_hold_limit = min(frame_hold_limit, 2)
            within_min_hold = False
        within_frame_hold = self.lost_confirm_frames <= frame_hold_limit
        if not within_min_hold and not within_frame_hold:
            return None

        if frame.hazard.active or self._is_action_blocked(action, frame):
            return None
        distance = frame.distance_m
        if distance is None:
            distance = getattr(frame.distance_state, "used_distance_m", None)
        if distance is None:
            distance = self._last_target_distance_m
        stop_distance = max(
            float(self.cfg.brake_distance_m),
            float(self.cfg.target_distance_m),
        )
        if (
            not rotation_only
            and self.cfg.distance_parking_enable
            and (distance is None or float(distance) <= stop_distance)
        ):
            return None

        # 漏检窗口只保持方向，不沿用原来的高速前进基准。15 RPM 足够抵消
        # 车身惯性，又不会让两帧误检期间继续高速冲向目标。
        hold_rpm = max(1, int(self.cfg.visible_steering_pid_fallback_base_rpm))
        hold_speed = min(
            int(action.speed_percent),
            self._forward_percent_for_rpm(hold_rpm, allow_below_min=True),
        )
        hold_correction_rpm = min(
            max(0, int(action.steer_correction_rpm)),
            max(0, int(self.cfg.visible_steering_pid_lost_hold_max_correction_rpm)),
        )
        return ControlAction(
            kind=action.kind,
            speed_percent=hold_speed,
            steer_inner_ratio_percent=action.steer_inner_ratio_percent,
            steer_outer_ratio_percent=action.steer_outer_ratio_percent,
            steer_correction_rpm=hold_correction_rpm,
            reason="lost_wait_hold_" + action.kind,
            brake_hold=action.brake_hold,
        )

    def _forward_percent_for_rpm(self, rpm: int, *, allow_below_min: bool = False) -> int:
        max_rpm = max(1, int(self.cfg.forward_max_rpm))
        value = 100.0 * max(0, int(rpm)) / float(max_rpm)
        # A percentage quantum must never round a new PI request UP through
        # its braking envelope. Its anti-windup separately tolerates this
        # known quantization; it is not a new recovery speed cap.
        percent = math.floor(value + 1e-9) if self.distance_pi_enabled and allow_below_min else round(value)
        percent = min(int(self.cfg.max_forward_percent), percent)
        if percent > 0 and not allow_below_min:
            percent = max(int(self.cfg.min_forward_percent), percent)
        return max(0, percent)

    def _held_forward_action_for_lost(self, frame: SensorFrame, now: float) -> Optional[ControlAction]:
        """Keep a short camera-dropout moving window without trusting a new target."""
        if self.last_dispatched_kind != "forward":
            return None
        if self.lost_confirm_frames >= max(1, int(self.cfg.lost_confirm_frames)):
            return None
        lost_started_at = self._lost_started_at
        if lost_started_at is None:
            return None
        max_hold_sec = max(0.0, float(self.cfg.lost_forward_hold_max_sec))
        if max_hold_sec <= 0.0 or float(now) - float(lost_started_at) > max_hold_sec:
            return None
        if frame.hazard.active or frame.obstacles.front or frame.obstacles.left or frame.obstacles.right:
            return None
        if bool(getattr(frame.distance_state, "brake_latched", False)):
            return None

        distance = frame.distance_m
        if distance is None:
            distance = getattr(frame.distance_state, "used_distance_m", None)
        if distance is None:
            distance = self._last_target_distance_m
        min_distance = max(
            float(self.cfg.brake_distance_m),
            float(self.cfg.target_distance_m),
            float(self.cfg.lost_forward_hold_min_distance_m),
        )
        if distance is None or float(distance) <= min_distance:
            return None

        speed = self._forward_percent_for_rpm(self.cfg.lost_forward_hold_rpm)
        if speed <= 0:
            return None
        return ControlAction.forward(speed, "lost_wait_hold_forward")

    def _lost_search_rotate_action(self, exit_direction: str, reason: str) -> Optional[ControlAction]:
        # Search compatibility path uses the same physical direction as the
        # camera exit direction; ordinary steer_left/steer_right is unchanged.
        if exit_direction == "left":
            return ControlAction.rotate_left(reason)
        if exit_direction == "right":
            return ControlAction.rotate_right(reason)
        return None

    def _capture_lost_exit_direction(self, frame: SensorFrame, *, search_entry: bool = False) -> None:
        if self._direction_loss_capture_id is None and int(frame.capture_frame_id) > 0:
            self._direction_loss_capture_id = int(frame.capture_frame_id)
        if (
            self._lost_exit_direction in ("left", "right")
            and self._lost_hint_source.startswith("search_candidate_last_")
        ):
            logger.info(
                "target_direction_latest_candidate_preserved direction=%s source=%s",
                self._lost_exit_direction,
                self._lost_hint_source,
            )
            return
        self._lost_exit_direction = None
        self._lost_hint_confidence = 0.0
        self._lost_hint_source = "none"
        if self._stale_direction_recovery_active:
            self._lost_hint_source = "stale_result_gap"
            return
        if (
            self.cfg.direction_history_enable
            and self._stale_direction_recovery_stage != "candidate_centering"
        ):
            decision = self._target_direction_history.resolve(
                required_missing_frames=max(1, int(self.cfg.lost_confirm_frames)),
            )
            if (
                decision.direction not in ("left", "right")
                and (search_entry or self.lost_confirm_frames >= max(1, int(self.cfg.lost_confirm_frames)))
            ):
                decision = self._target_direction_history.latest_reliable_side()
            if decision.direction in ("left", "right"):
                self._lost_exit_direction = decision.direction
                self._lost_hint_confidence = float(decision.confidence)
                self._lost_hint_source = str(decision.reason)
            elif (
                decision.reason == "missing_confirmation_pending" and not search_entry
                and self.lost_confirm_frames < max(1, int(self.cfg.lost_confirm_frames))
            ):
                # Pending is NOT evidence that history lacks a side. The
                # existing wait path uses latest_reliable_side for bounded yaw;
                # defer the search choice until loss is actually confirmed.
                self._lost_hint_source = "missing_confirmation_pending"
            else:
                hint = self._historical_hint_for_current_target()
                if hint is not None:
                    self._lost_exit_direction = str(hint["direction"])
                    self._lost_hint_confidence = float(hint["confidence"])
                    self._lost_hint_source = "historical_direction_evidence"
                    logger.info(
                        "target_direction_history_historical_fallback capture=%d direction=%s confidence=%.2f captures=%s",
                        int(frame.capture_frame_id),
                        self._lost_exit_direction,
                        float(self._lost_hint_confidence),
                        ",".join(str(value) for value in hint["selected_capture_frame_ids"]),
                    )
                else:
                    self._lost_exit_direction = None
                    self._lost_hint_confidence = 0.0
                    self._lost_hint_source = str(decision.reason)
            logger.info(
                "target_direction_history_resolve capture=%d direction=%s confidence=%.2f "
                "reason=%s missing=%d visible_samples=%d last_visible_capture=%s "
                "history_checked=%s selected_direction=%s selected_source=%s loss_capture=%s search_entry=%s",
                int(frame.capture_frame_id),
                decision.direction or "none",
                float(decision.confidence),
                decision.reason,
                int(decision.missing_frames),
                int(decision.visible_samples),
                "none" if decision.last_visible_capture_frame_id is None else int(decision.last_visible_capture_frame_id),
                decision.reason != "missing_confirmation_pending", self._lost_exit_direction or "none",
                self._lost_hint_source, self._direction_loss_capture_id, search_entry,
            )
            return
        if self.last_person_center_x is None or frame.width <= 0:
            return
        self._lost_exit_direction = (
            "left"
            if float(self.last_person_center_x) < float(frame.width) / 2.0
            else "right"
        )
        self._lost_hint_confidence = 1.0
        self._lost_hint_source = "last_reliable_center"

    @staticmethod
    def _stale_direction_zero_decision(reason: str) -> ControlDecision:
        return ControlDecision(
            actions=[ControlAction.stop(reason, brake_hold=False)],
            waiting_lost_confirm=True,
            clear_action_queue=False,
            # This is an observation/search hold, not a safety stop. Mark it
            # explicitly so the runtime dispatches STOP_SOFT instead of
            # entering a persistent brake latch.
            soft_stop_requested=True,
            reason=reason,
        )

    def _stale_probe_feedback_usable(self, frame: SensorFrame, now: float) -> bool:
        feedback = frame.steering_feedback
        if feedback is None or not bool(feedback.trustworthy):
            return False
        try:
            timestamp = float(feedback.timestamp)
            integrated = float(feedback.integrated_yaw_right_deg)
        except (TypeError, ValueError):
            return False
        return bool(
            math.isfinite(integrated)
            and math.isfinite(timestamp)
            and float(now) - timestamp
            <= max(
                0.10,
                min(0.30, float(self.cfg.search_revolution_feedback_stale_sec)),
            )
        )

    def _stale_probe_rotate_decision(
        self,
        direction: str,
        frame: SensorFrame,
        reason: str,
    ) -> ControlDecision:
        action = self._lost_search_rotate_action(direction, reason)
        if action is None:
            self.search_state = "direction_unresolved"
            return self._stale_direction_zero_decision("stale_probe_direction_unresolved")
        block_reason = self._action_block_reason(action, frame)
        if block_reason is not None:
            return ControlDecision(
                explicit_stop_requested=True,
                clear_action_queue=True,
                stop_action_execution=True,
                reason=block_reason,
            )
        return ControlDecision(
            actions=[action],
            waiting_lost_confirm=True,
            reason=reason,
        )

    def _stale_direction_recovery_decision(
        self,
        frame_index: int,
        frame: SensorFrame,
        now: float,
    ) -> Optional[ControlDecision]:
        if not self._stale_direction_recovery_active:
            return None

        if self._lost_started_at is None:
            self._lost_started_at = float(now)
        self.lost_confirm_frames += 1
        # A direction resolved from an earlier capture slot is authoritative
        # for this loss episode. Delayed worker results must not erase it.
        if self.search_direction not in ("left", "right"):
            self.search_direction = None
        if self._lost_exit_direction not in ("left", "right"):
            self._lost_exit_direction = None
        self._reset_visible_steer_memory()
        self._visual_steering_pid.reset()
        self._parked_recenter_pid.reset()
        self._reset_distance_pid()
        self.last_steering_pid_result = None

        # Stale vision is a loss-confirmation event, not a command to probe
        # both sides.  Wait for the configured number of missing decisions,
        # then resolve the latest reliable capture-side and freeze that side
        # for the search episode.  This keeps the control path single-owner
        # and prevents an encoder-based direction_probe from going idle.
        required_missing = max(1, int(self.cfg.lost_confirm_frames))
        if self.lost_confirm_frames < required_missing:
            self.search_state = "none"
            return self._stale_direction_zero_decision(
                "stale_loss_confirm_%d/%d" % (self.lost_confirm_frames, required_missing)
            )
        history_decision = None
        if self.cfg.direction_history_enable:
            history_decision = self._target_direction_history.resolve(
                required_missing_frames=required_missing,
            )
            if history_decision.direction not in ("left", "right"):
                history_decision = self._target_direction_history.latest_reliable_side()
        direction = (
            None
            if history_decision is None
            else history_decision.direction
        )
        if direction not in ("left", "right"):
            direction = self._lost_exit_direction
        if direction not in ("left", "right"):
            self.search_state = "direction_unresolved"
            self.search_direction = None
            self._lost_hint_confidence = 0.0
            self._lost_hint_source = "stale_loss_no_reliable_side"
            logger.warning(
                "stale_loss_direction_unresolved frame=%d capture=%d missing=%d",
                int(frame_index),
                int(frame.capture_frame_id),
                int(self.lost_confirm_frames),
            )
            return self._stale_direction_zero_decision(
                "stale_loss_direction_unresolved"
            )
        self._lost_exit_direction = str(direction)
        if history_decision is not None:
            self._lost_hint_confidence = float(history_decision.confidence)
            self._lost_hint_source = str(history_decision.reason)
        else:
            self._lost_hint_confidence = max(0.50, float(self._lost_hint_confidence))
            self._lost_hint_source = "latest_reliable_side"
        self.search_direction = str(direction)
        self.search_state = "searching"
        self._reset_stale_direction_recovery("history_side_resolved")
        self._begin_search_rotation_measurement(frame)
        self._ensure_search_state(frame)
        action = self._fallback_search_action(frame, "search_history")
        if action is None:
            return self._stale_direction_zero_decision("search_history_direction_blocked")
        self._mark_search_rotation_started(now)
        self.last_action_frame = int(frame_index)
        logger.info(
            "stale_loss_search_started frame=%d capture=%d direction=%s source=%s "
            "missing=%d action=%s",
            int(frame_index),
            int(frame.capture_frame_id),
            str(direction),
            self._lost_hint_source,
            int(self.lost_confirm_frames),
            action.kind,
        )
        return ControlDecision(
            actions=[action],
            clear_action_queue=True,
            reason=action.reason,
            evidence_capture_frame_id=(
                None
                if history_decision is None
                else history_decision.last_visible_capture_frame_id
            ),
        )

        if (
            self.cfg.direction_history_enable
            and self._stale_direction_recovery_stage != "candidate_centering"
        ):
            history_decision = self._target_direction_history.resolve(
                required_missing_frames=max(1, int(self.cfg.lost_confirm_frames)),
            )
            if (
                history_decision.direction not in ("left", "right")
                and self.lost_confirm_frames >= max(1, int(self.cfg.lost_confirm_frames))
            ):
                history_decision = self._target_direction_history.latest_reliable_side()
            if history_decision.direction in ("left", "right"):
                self._lost_exit_direction = history_decision.direction
                self._lost_hint_confidence = float(history_decision.confidence)
                self._lost_hint_source = str(history_decision.reason)
                self._reset_stale_direction_recovery("capture_timeline_resolved")
                self.search_state = "none"
                self._begin_search_rotation_measurement(frame)
                self._ensure_search_state(frame)
                action = self._fallback_search_action(frame, "search_history")
                if action is None:
                    return self._stale_direction_zero_decision(
                        "search_history_direction_blocked"
                    )
                self._mark_search_rotation_started(now)
                self.last_action_frame = int(frame_index)
                logger.info(
                    "stale_gap_history_promoted frame=%d capture=%d direction=%s "
                    "confidence=%.2f source=%s missing=%d last_visible_capture=%s",
                    int(frame_index),
                    int(frame.capture_frame_id),
                    history_decision.direction,
                    float(history_decision.confidence),
                    history_decision.reason,
                    int(history_decision.missing_frames),
                    "none" if history_decision.last_visible_capture_frame_id is None else int(history_decision.last_visible_capture_frame_id),
                )
                return ControlDecision(
                    actions=[action],
                    clear_action_queue=True,
                    reason=action.reason,
                )
            if history_decision.missing_frames < max(
                1, int(self.cfg.lost_confirm_frames)
            ):
                self.search_state = "direction_probe"
                return self._stale_direction_zero_decision(
                    "stale_history_missing_confirm"
                )

        if self._search_observation_hold:
            self.search_state = "direction_probe"
            reason = (
                "search_candidate_center_observe_hold"
                if self._candidate_center_reason_prefix == "search_candidate_center"
                else "stale_probe_candidate_evidence_observe"
            )
            return self._stale_direction_zero_decision(
                reason
            )

        stage = self._stale_direction_recovery_stage
        recovery_started_at = self._stale_direction_recovery_started_at
        recovery_elapsed = (
            0.0
            if recovery_started_at is None
            else max(0.0, float(now) - float(recovery_started_at))
        )
        recovery_timeout = max(0.0, float(self.cfg.search_timeout_sec))
        if (
            stage not in ("observe", "unresolved")
            and recovery_timeout > 0.0
            and recovery_elapsed >= recovery_timeout
        ):
            self._stale_direction_recovery_stage = "unresolved"
            self.search_state = "direction_unresolved"
            self._lost_hint_confidence = 0.0
            self._lost_hint_source = "stale_probe_timeout"
            logger.info(
                "stale_direction_probe_aborted frame=%d stage=%s "
                "reason=timeout elapsed=%.3fs limit=%.3fs",
                int(frame_index),
                stage,
                recovery_elapsed,
                recovery_timeout,
            )
            return self._stale_direction_zero_decision("stale_probe_timeout")
        if stage == "observe":
            self.search_state = "direction_probe"
            self._stale_direction_observe_frames += 1
            elapsed = recovery_elapsed
            feedback = frame.steering_feedback
            feedback_usable = self._stale_probe_feedback_usable(frame, now)
            yaw_rate = (
                0.0
                if feedback is None
                else abs(float(feedback.yaw_rate_right_dps))
            )
            settled = bool(
                not feedback_usable
                or yaw_rate <= max(0.0, float(self.cfg.stale_direction_settle_yaw_rate_dps))
                or elapsed >= max(0.05, float(self.cfg.stale_direction_observe_max_sec))
            )
            if (
                self._stale_direction_observe_frames
                >= max(1, int(self.cfg.stale_direction_observe_frames))
                and settled
            ):
                self._reset_search_timeout()
                self._begin_search_rotation_measurement(frame)
                if (
                    self.cfg.stale_direction_probe_enable
                    and self._stale_direction_prior_hint in ("left", "right")
                    and self._stale_probe_feedback_usable(frame, now)
                ):
                    self._stale_direction_recovery_stage = "primary_out"
                    self._lost_hint_confidence = 0.25
                    self._lost_hint_source = (
                        "stale_probe_prior_" + str(self._stale_direction_prior_hint)
                    )
                    logger.info(
                        "stale_direction_probe_ready frame=%d prior_hint=%s "
                        "observe_frames=%d elapsed=%.3fs angle=%.1fdeg",
                        int(frame_index),
                        self._stale_direction_prior_hint,
                        int(self._stale_direction_observe_frames),
                        elapsed,
                        float(self.cfg.stale_direction_probe_angle_deg),
                    )
                else:
                    self._stale_direction_recovery_stage = "unresolved"
                    self.search_state = "direction_unresolved"
                    self._lost_hint_confidence = 0.0
                    self._lost_hint_source = "stale_probe_encoder_unavailable"
                    logger.info(
                        "stale_direction_probe_unavailable frame=%d prior_hint=%s encoder=%s",
                        int(frame_index),
                        self._stale_direction_prior_hint or "none",
                        bool(self._stale_probe_feedback_usable(frame, now)),
                    )
            return self._stale_direction_zero_decision("stale_direction_observe")

        if stage == "unresolved":
            self.search_state = "direction_unresolved"
            return self._stale_direction_zero_decision("stale_probe_unresolved_hold")

        if not self._stale_probe_feedback_usable(frame, now):
            self._stale_direction_recovery_stage = "unresolved"
            self.search_state = "direction_unresolved"
            self._lost_hint_confidence = 0.0
            self._lost_hint_source = "stale_probe_encoder_lost"
            logger.info(
                "stale_direction_probe_aborted frame=%d stage=%s reason=encoder_unavailable",
                int(frame_index),
                stage,
            )
            return self._stale_direction_zero_decision("stale_probe_encoder_lost")

        self.search_state = "direction_probe"
        if (
            stage == "candidate_centering"
            and self._search_rotation_origin_integrated_yaw_deg is None
        ):
            self._begin_search_rotation_measurement(frame)
        self._update_search_rotation_progress(frame)
        heading = float(self._search_heading_from_loss_deg)
        angle = max(2.0, float(self.cfg.stale_direction_probe_angle_deg))
        tolerance = max(
            0.5,
            min(angle * 0.5, float(self.cfg.stale_direction_probe_return_tolerance_deg)),
        )
        primary = str(self._stale_direction_prior_hint)
        secondary = "left" if primary == "right" else "right"
        primary_target = angle if primary == "right" else -angle
        secondary_target = -primary_target

        if stage == "candidate_centering":
            center_ratio = self._stale_candidate_center_ratio
            if center_ratio is None:
                self._stale_direction_recovery_stage = "unresolved"
                self.search_state = "direction_unresolved"
                self._lost_hint_confidence = 0.0
                self._lost_hint_source = "stale_candidate_geometry_missing"
                return self._stale_direction_zero_decision(
                    "stale_candidate_geometry_missing"
                )
            if self._stale_candidate_center_anchor_heading_deg is None:
                self._stale_candidate_center_anchor_heading_deg = heading
            relative_heading = heading - float(
                self._stale_candidate_center_anchor_heading_deg
            )
            left_ratio = max(0.0, min(1.0, float(self.cfg.center_left_ratio)))
            right_ratio = max(left_ratio, min(1.0, float(self.cfg.center_right_ratio)))
            if center_ratio < left_ratio:
                direction = "left"
            elif center_ratio > right_ratio:
                direction = "right"
            else:
                self._lost_hint_confidence = 0.75
                self._lost_hint_source = self._candidate_center_reason_prefix + "ed"
                return self._stale_direction_zero_decision(
                    self._candidate_center_reason_prefix + "_observe"
                )
            limit_reached = (
                relative_heading <= -angle
                if direction == "left"
                else relative_heading >= angle
            )
            if limit_reached:
                self._lost_hint_source = self._candidate_center_reason_prefix + "_limit"
                logger.info(
                    "stale_candidate_centering_limit frame=%d direction=%s "
                    "center=%.3f relative_heading=%.2fdeg limit=%.1fdeg action=hold",
                    int(frame_index),
                    direction,
                    center_ratio,
                    relative_heading,
                    angle,
                )
                return self._stale_direction_zero_decision(
                    self._candidate_center_reason_prefix + "_limit_hold"
                )
            reason = "%s_%s" % (self._candidate_center_reason_prefix, direction)
            self._lost_hint_source = reason
            return self._stale_probe_rotate_decision(
                direction,
                frame,
                reason,
            )

        if stage == "primary_out":
            reached = heading >= primary_target if primary == "right" else heading <= primary_target
            if not reached:
                return self._stale_probe_rotate_decision(
                    primary,
                    frame,
                    "stale_probe_primary_" + primary,
                )
            self._stale_direction_recovery_stage = "primary_observe"
            self._stale_direction_probe_observe_frames = 0
            logger.info(
                "stale_direction_probe_endpoint frame=%d side=%s heading=%.2fdeg",
                int(frame_index),
                primary,
                heading,
            )
            return self._stale_direction_zero_decision("stale_probe_primary_observe")

        if stage == "primary_observe":
            self._stale_direction_probe_observe_frames += 1
            if self._stale_direction_probe_observe_frames >= max(
                1, int(self.cfg.stale_direction_probe_observe_frames)
            ):
                self._stale_direction_recovery_stage = "return_origin"
            return self._stale_direction_zero_decision("stale_probe_primary_observe")

        if stage == "return_origin":
            if abs(heading) <= tolerance:
                self._stale_direction_recovery_stage = "secondary_out"
                logger.info(
                    "stale_direction_probe_origin frame=%d heading=%.2fdeg secondary=%s",
                    int(frame_index),
                    heading,
                    secondary,
                )
                return self._stale_direction_zero_decision("stale_probe_origin_observe")
            return self._stale_probe_rotate_decision(
                "left" if heading > 0.0 else "right",
                frame,
                "stale_probe_return_origin",
            )

        if stage == "secondary_out":
            reached = (
                heading >= secondary_target
                if secondary == "right"
                else heading <= secondary_target
            )
            if not reached:
                return self._stale_probe_rotate_decision(
                    secondary,
                    frame,
                    "stale_probe_secondary_" + secondary,
                )
            self._stale_direction_recovery_stage = "secondary_observe"
            self._stale_direction_probe_observe_frames = 0
            logger.info(
                "stale_direction_probe_endpoint frame=%d side=%s heading=%.2fdeg",
                int(frame_index),
                secondary,
                heading,
            )
            return self._stale_direction_zero_decision("stale_probe_secondary_observe")

        if stage == "secondary_observe":
            self._stale_direction_probe_observe_frames += 1
            if self._stale_direction_probe_observe_frames >= max(
                1, int(self.cfg.stale_direction_probe_observe_frames)
            ):
                self._stale_direction_recovery_stage = "final_return_origin"
            return self._stale_direction_zero_decision("stale_probe_secondary_observe")

        if stage == "final_return_origin":
            if abs(heading) > tolerance:
                return self._stale_probe_rotate_decision(
                    "left" if heading > 0.0 else "right",
                    frame,
                    "stale_probe_final_return_origin",
                )
            self._stale_direction_recovery_stage = "unresolved"
            self.search_state = "direction_unresolved"
            self._lost_hint_confidence = 0.0
            self._lost_hint_source = "stale_probe_both_sides_empty"
            logger.info(
                "stale_direction_probe_complete frame=%d outcome=both_sides_empty "
                "heading=%.2fdeg action=hold",
                int(frame_index),
                heading,
            )
            return self._stale_direction_zero_decision("stale_probe_unresolved_hold")

        self._stale_direction_recovery_stage = "unresolved"
        self.search_state = "direction_unresolved"
        return self._stale_direction_zero_decision("stale_probe_unknown_stage")

    def _current_lateral_candidate(
        self,
        frame: SensorFrame,
    ) -> Optional[_CurrentLateralCandidate]:
        candidate = frame.lateral_candidate
        if candidate is None or int(frame.width) <= 0:
            return None
        if (
            int(candidate.capture_frame_id) <= 0
            or int(candidate.capture_frame_id) != int(frame.capture_frame_id)
            or float(candidate.score)
            < max(0.0, float(self.cfg.search_candidate_untracked_min_score))
        ):
            return None
        try:
            x1, y1, x2, y2 = (float(value) for value in candidate.bbox)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        width = float(frame.width)
        aimline_x = width * 0.5
        center_ratio = (x1 + x2) / (2.0 * width)
        aimline_intersects = bool(x1 <= aimline_x <= x2)
        if aimline_intersects:
            aimline_gap_ratio = 0.0
        elif x2 < aimline_x:
            aimline_gap_ratio = (aimline_x - x2) / width
        else:
            aimline_gap_ratio = (x1 - aimline_x) / width
        center_left = max(0.0, min(1.0, float(self.cfg.center_left_ratio)))
        center_right = max(center_left, min(1.0, float(self.cfg.center_right_ratio)))
        if center_ratio < center_left:
            position = "left"
        elif center_ratio > center_right:
            position = "right"
        else:
            position = "center"
        current = _CurrentLateralCandidate(
            position=position,
            center_ratio=center_ratio,
            aimline_gap_ratio=max(0.0, float(aimline_gap_ratio)),
            aimline_intersects=aimline_intersects,
            evidence=candidate,
        )
        if position == "center" or bool(candidate.active_target_match):
            return current

        latest_side = self._target_direction_history.latest_reliable_side()
        previous_direction = (
            self.search_direction
            if self.search_direction in ("left", "right")
            else self._lost_exit_direction
            if self._lost_exit_direction in ("left", "right")
            else latest_side.direction
        )
        if previous_direction not in ("left", "right") or position == previous_direction:
            return current
        logger.info(
            "current_lateral_candidate_rejected capture_frame_id=%d source=%s "
            "score=%.3f center=%.3f previous=%s position=%s "
            "identity_match=False reason=opposite_side_identity_unconfirmed",
            int(candidate.capture_frame_id),
            str(candidate.source),
            float(candidate.score),
            float(center_ratio),
            str(previous_direction),
            str(position),
        )
        return None

    @staticmethod
    def _normalize_candidate_bbox(
        bbox: tuple,
        frame_width: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        if int(frame_width) <= 0:
            return None
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            return None
        width = float(frame_width)
        return (x1 / width, y1, x2 / width, y2)

    def _candidate_geometry_reference(
        self,
    ) -> Tuple[Optional[Tuple[float, float, float, float]], int]:
        history = self._target_direction_history.latest_visible_evidence()
        history_bbox = None if history is None else history.bbox
        history_capture = -1 if history is None else int(history.capture_frame_id)
        if self._candidate_geometry_anchor_capture_frame_id > history_capture:
            return (
                self._candidate_geometry_anchor_bbox,
                int(self._candidate_geometry_anchor_capture_frame_id),
            )
        return history_bbox, history_capture

    def _candidate_geometry_is_continuous(
        self,
        bbox: tuple,
        *,
        frame_width: int,
        capture_frame_id: int,
    ) -> bool:
        current = self._normalize_candidate_bbox(bbox, frame_width)
        reference, reference_capture = self._candidate_geometry_reference()
        if current is None or reference is None or int(capture_frame_id) <= 0:
            return False
        gap = int(capture_frame_id) - int(reference_capture)
        if gap <= 0 or gap > max(
            1, int(self.cfg.search_candidate_continuity_max_capture_gap)
        ):
            return False
        current_center = (float(current[0]) + float(current[2])) * 0.5
        reference_center = (float(reference[0]) + float(reference[2])) * 0.5
        center_jump = abs(current_center - reference_center)
        max_jump = max(
            0.01,
            min(0.50, float(self.cfg.search_candidate_continuity_max_center_jump_ratio)),
        )
        horizontal_intersection = max(
            0.0,
            min(float(current[2]), float(reference[2]))
            - max(float(current[0]), float(reference[0])),
        )
        min_box_width = max(
            1e-6,
            min(
                float(current[2]) - float(current[0]),
                float(reference[2]) - float(reference[0]),
            ),
        )
        horizontal_overlap = horizontal_intersection / min_box_width
        return bool(
            center_jump <= max_jump
            or horizontal_overlap
            >= max(
                0.0,
                min(1.0, float(self.cfg.search_candidate_continuity_min_horizontal_overlap)),
            )
        )

    def _remember_candidate_geometry(
        self,
        candidate: LateralCandidateEvidence,
        frame_width: int,
    ) -> None:
        # A detector-only C0 box is not target-owned geometry. Caching it here
        # would let a second person establish a false continuity chain and
        # reverse the frozen search direction on the next frame.
        if not bool(candidate.active_target_match):
            return
        normalized = self._normalize_candidate_bbox(candidate.bbox, frame_width)
        if normalized is None:
            return
        self._candidate_geometry_anchor_bbox = normalized
        self._candidate_geometry_anchor_capture_frame_id = int(candidate.capture_frame_id)

    def _apply_current_lateral_candidate(
        self,
        frame: SensorFrame,
    ) -> Optional[_CurrentLateralCandidate]:
        current = self._current_lateral_candidate(frame)
        if current is None:
            return None
        direction = current.position
        center_ratio = current.center_ratio
        candidate = current.evidence
        if self._stale_direction_recovery_active:
            self._reset_stale_direction_recovery("fresh_current_lateral_candidate")
        previous_direction = (
            self.search_direction
            if self.search_direction in ("left", "right")
            else self._lost_exit_direction
        )
        self._lost_hint_confidence = float(candidate.score)
        if direction == "center":
            # Keep the last search side until ReID confirms identity, while
            # the current detector geometry independently stops lateral yaw.
            self._lost_hint_source = "search_candidate_aimline_intersection"
        else:
            self.search_direction = direction
            self._lost_exit_direction = direction
            self._lost_hint_source = "search_candidate_last_current_frame"
            if previous_direction in ("left", "right") and previous_direction != direction:
                self._reset_search_timeout()
                self._begin_search_rotation_measurement(frame)
        self._remember_candidate_geometry(candidate, int(frame.width))
        logger.info(
            "current_lateral_candidate_applied capture_frame_id=%d source=%s "
            "score=%.3f center=%.3f aimline_gap=%.3f aimline_intersects=%s "
            "previous=%s position=%s "
            "identity_claim=False longitudinal_allowed=False",
            int(candidate.capture_frame_id),
            str(candidate.source),
            float(candidate.score),
            float(center_ratio),
            float(current.aimline_gap_ratio),
            bool(current.aimline_intersects),
            previous_direction or "none",
            direction,
        )
        return current

    def _lost_confirm_wait_decision(self, first_lost_frame: bool, frame: SensorFrame) -> ControlDecision:
        if self.cfg.direction_history_enable:
            # During a fresh detector dropout, preserve only the latest
            # reliable *lateral* side and continue bounded yaw. This keeps a
            # fast target near the edge in view while the capture timeline
            # accumulates the configured missing-frame confirmation. A stale
            # processing gap is handled earlier and remains zero-yaw because
            # it has no fresh evidence.
            current = self._current_lateral_candidate(frame)
            latest_side = self._target_direction_history.latest_reliable_side()
            if current is not None and current.position in ("left", "right"):
                direction = current.position
                candidate = current.evidence
                evidence_capture_frame_id = int(candidate.capture_frame_id)
                confidence = float(candidate.score)
                evidence_source = "current_candidate"
            else:
                direction = latest_side.direction
                evidence_capture_frame_id = latest_side.last_visible_capture_frame_id
                confidence = float(latest_side.confidence)
                evidence_source = "history"
            hold_action = self._lost_search_rotate_action(
                direction,
                "lost_%s_hold_%s" % (evidence_source, direction),
            )
            if (
                hold_action is not None
                and self.lost_confirm_frames
                < max(1, int(self.cfg.lost_confirm_frames))
                and not frame.hazard.active
                and self._action_block_reason(hold_action, frame) is None
            ):
                logger.info(
                    "lost lateral yaw hold: capture_frame_id=%d missing=%d/%d "
                    "direction=%s evidence_source=%s evidence_capture_frame_id=%s "
                    "confidence=%.2f",
                    int(frame.capture_frame_id),
                    int(self.lost_confirm_frames),
                    int(self.cfg.lost_confirm_frames),
                    direction,
                    evidence_source,
                    "none"
                    if evidence_capture_frame_id is None
                    else int(evidence_capture_frame_id),
                    confidence,
                )
                return ControlDecision(
                    actions=[hold_action],
                    waiting_lost_confirm=True,
                    reason=hold_action.reason,
                    evidence_capture_frame_id=evidence_capture_frame_id,
                )
            # No reliable side is available. Cancel yaw through the publisher
            # without entering brake hold until history can resolve a side.
            return self._stale_direction_zero_decision(
                "lost_direction_history_confirm"
            )
        # A missing frame immediately removes longitudinal speed, but it does
        # not discard the last reliable yaw direction.  Continue bounded
        # in-place yaw through the short confirmation window so a fast target
        # is not lost while DRIVE -> STOP -> SEARCH is being sequenced.
        direction = self._lost_exit_direction
        hold_action = self._lost_search_rotate_action(
            direction,
            "lost_wait_yaw_%s" % direction,
        )
        if (
            hold_action is not None
            and self.lost_confirm_frames < max(1, int(self.cfg.lost_confirm_frames))
            and not frame.hazard.active
            and self._action_block_reason(hold_action, frame) is None
        ):
            return ControlDecision(
                actions=[hold_action],
                waiting_lost_confirm=True,
                reason=hold_action.reason,
            )

        # No trustworthy lateral hint remains. Stop once instead of guessing.
        motion_still_active = self.last_dispatched_kind in (
            "forward",
            "steer_left",
            "steer_right",
            "rotate_left",
            "rotate_right",
        )
        should_stop = bool(first_lost_frame or motion_still_active)
        reason = "lost_confirm_wait_stop" if should_stop else "lost_confirm_wait"
        return ControlDecision(
            waiting_lost_confirm=True,
            explicit_stop_requested=should_stop,
            clear_action_queue=should_stop,
            stop_action_execution=should_stop,
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
        if self.cfg.center_deadzone_ratio is not None:
            half = max(0.01, min(0.45, float(self.cfg.center_deadzone_ratio)))
            return width * (0.5 - half), width * (0.5 + half)
        return self._band_from_ratios(width, self.cfg.center_left_ratio, self.cfg.center_right_ratio)

    def _active_center_band(self, width: int) -> tuple:
        center_left = float(self.cfg.center_left_ratio)
        center_right = float(self.cfg.center_right_ratio)
        if self.last_dispatched_kind in (
            "steer_left",
            "steer_right",
            "rotate_left",
            "rotate_right",
        ):
            configured_left = self.cfg.steer_release_left_ratio
            configured_right = self.cfg.steer_release_right_ratio
        else:
            configured_left = self.cfg.steer_enter_left_ratio
            configured_right = self.cfg.steer_enter_right_ratio
        if configured_left is not None or configured_right is not None:
            # 显式 enter/release 回差必须优先，否则 center_deadzone 会把快速居中参数完全遮蔽。
            left_ratio = self._ratio_or_default(configured_left, center_left)
            right_ratio = self._ratio_or_default(configured_right, center_right)
        elif self.cfg.center_deadzone_ratio is not None:
            return self._center_band(width)
        else:
            left_ratio = center_left
            right_ratio = center_right
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

    def _forward_percent_for_distance(
        self,
        distance_m: float,
        *,
        now: Optional[float] = None,
    ) -> int:
        cfg = self.cfg
        if distance_m < cfg.brake_distance_m:
            self._forward_active = False
            if self.distance_pi_enabled:
                self._reset_distance_pid()
            return 0

        sample_now = time.monotonic() if now is None else float(now)
        if self.distance_pi_enabled:
            result = self._update_distance_pid(float(distance_m), now=sample_now)
            self._forward_active = result.output_rpm > 0
            return self._forward_percent_for_rpm(max(0, result.output_rpm), allow_below_min=True)
        tracking_base = self._tracking_base_rpm(float(distance_m), sample_now)
        if cfg.distance_pid_enable and tracking_base is not None:
            # Matching a moving target is not a stop/restart hysteresis event.
            # Two trusted physical observations have already authorized it.
            if not self._forward_active:
                self._forward_active = True
                self._reset_distance_pid()
            rpm = max(0, self._update_distance_pid(float(distance_m), now=sample_now).output_rpm)
            return self._forward_percent_for_rpm(rpm, allow_below_min=True)

        forward_start_m = max(
            float(cfg.target_distance_m) + 0.01,
            float(cfg.forward_start_distance_m),
        )
        forward_stop_m = max(
            float(cfg.target_distance_m),
            min(forward_start_m - 0.01, float(cfg.forward_stop_distance_m)),
        )
        if self._forward_active:
            no_matching_near = bool(
                cfg.distance_pid_enable and cfg.distance_feedforward_enable
                and not cfg.distance_approach_enable
                and float(distance_m) < forward_start_m
            )
            if float(distance_m) <= forward_stop_m or no_matching_near:
                self._forward_active = False
                self._reset_distance_pid()
                if no_matching_near:
                    logger.info(
                        "near_no_matching_stop uid=%s sample_ts=%s distance_m=%.3f "
                        "restart_m=%.3f launch_bias_suppressed=True",
                        self.active_target_id, self._distance_pid_sample_timestamp,
                        float(distance_m), forward_start_m,
                    )
                logger.info(
                    "forward_hysteresis_stopped distance=%.2fm stop=%.2fm restart=%.2fm",
                    float(distance_m),
                    forward_stop_m,
                    forward_start_m,
                )
                return 0
        elif float(distance_m) < forward_start_m:
            return 0
        else:
            self._forward_active = True
            self._reset_distance_pid()
            logger.info(
                "forward_hysteresis_started distance=%.2fm start=%.2fm stop=%.2fm",
                float(distance_m),
                forward_start_m,
                forward_stop_m,
            )
        if cfg.distance_pid_enable:
            rpm = max(0, int(self._update_distance_pid(distance_m, now=now).output_rpm))
            return self._forward_percent_for_rpm(rpm, allow_below_min=True)
        min_rpm = max(0, int(cfg.forward_min_rpm))
        max_rpm = max(min_rpm, int(cfg.forward_max_rpm))
        max_distance = max(float(cfg.target_distance_m) + 0.01, float(cfg.forward_curve_max_distance_m))
        progress = (float(distance_m) - float(cfg.target_distance_m)) / (
            max_distance - float(cfg.target_distance_m)
        )
        progress = max(0.0, min(1.0, progress))
        exponent = max(0.10, min(3.0, float(cfg.forward_curve_exponent)))
        rpm = min_rpm + (max_rpm - min_rpm) * (progress**exponent)
        # The DRIVE path converts this percentage against its independent
        # forward RPM limit, so 20..50 rpm remains independent of STEER/TURN.
        p = round(100.0 * rpm / float(max(1, max_rpm)))
        p = min(int(cfg.max_forward_percent), int(p))
        if p > 0:
            p = max(int(cfg.min_forward_percent), p)
        return max(0, p)

    def _rotate_action_for_edge(self, edge_type: str) -> Optional[ControlAction]:
        if edge_type == "left":
            return self._lost_search_rotate_action("left", "person_left_rotate")
        if edge_type == "right":
            return self._lost_search_rotate_action("right", "person_right_rotate")
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

    @staticmethod
    def _is_fresh_depth_state(frame: SensorFrame) -> bool:
        state = frame.distance_state
        detail = str(getattr(state, "source_detail", ""))
        return bool(
            str(getattr(state, "source", "")) == "vision_depth"
            and getattr(state, "raw_distance_m", None) is not None
            and detail.startswith("depth_")
            and not detail.endswith("_hold")
        )

    def _distance_longitudinally_untrusted(self, frame: SensorFrame, *, check_hold_confidence=True) -> bool:
        """Return True for Depth values that must never start forward PID.

        These labels describe a newly re-anchored, jump-pending, expired, or
        otherwise background-prone sample.  Camera position remains usable;
        only the distance-driven longitudinal branch is blocked.
        """
        state = frame.distance_state
        if str(getattr(state, "source", "")) != "vision_depth":
            return False
        detail = str(getattr(state, "source_detail", "")).lower()
        mode = str(getattr(state, "fusion_mode", "")).lower()
        blocked_tokens = (
            "reanchored_after_timeout",
            "distance_jump_pending",
            "distance_jump_rate_guard",
            "far_background_guard",
            "depth_expired",
            "depth_unavailable",
            "no_valid_depth",
        )
        if any(token in detail or token in mode for token in blocked_tokens):
            return True

        # The explicit re-anchor label is present for only the first sample;
        # following hold/fresh samples may be reported as plain ``radar``.
        # Keep rejecting a large upward jump relative to the last trusted
        # target distance until a new target lock resets that anchor.
        current = getattr(state, "used_distance_m", None)
        previous = getattr(self, "_last_target_distance_m", None)
        if current is not None and previous is not None:
            try:
                jump = float(current) - float(previous)
                jump_limit = max(1.0, abs(float(previous)) * 0.75)
                if jump > jump_limit:
                    return True
            except (TypeError, ValueError):
                pass
        # Only the strictly identified replay-retention helper skips this last
        # test. All PID, recovery and motion eligibility callers keep it.
        if (check_hold_confidence and
            ("hold" in detail or mode.endswith("_hold"))
            and float(getattr(state, "fusion_confidence", 1.0)) < 0.50
        ):
            return True
        return False

    def _note_depth_quality_failure(self, frame: SensorFrame, now: float) -> None:
        """A scheduling gap may pause a ramp, but cannot authorize motion."""
        if self.distance_pi_enabled:
            self._pause_distance_pi(frame, now, "measurement_attempt_failed")
            return
        active = self._depth_schedule_recovery
        state = frame.distance_state
        anchor = self._depth_recovery_anchor
        reference = active if active is not None else anchor
        scheduling_only = (
            state.source_detail == "depth_detector_bbox_stale"
            or (reference is not None and state.is_replay_of(reference[1]))
        )
        safe_gap = False
        if self.cfg.depth_measured_recovery_enable:
            hint = getattr(self, "_depth_gap_resume_hint", None)
            source = reference[:3] if reference is not None else (None if hint is None else hint[:3])
            safe_gap = bool(
                source is not None and source[0] == self.active_target_id
                and (scheduling_only or state.source_detail == "depth_detector_bbox_stale")
                and any(p.track_id == source[0] for p in frame.persons)
                and self.search_state == "none" and not frame.hazard.active
                and not any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
                and state.safety_distance_m is None and not state.brake_latched
                and not self._target_stop_latched
                and state.raw_distance_m is None and 0 <= now-source[1] <= .50
                and not self._distance_longitudinally_untrusted(frame, check_hold_confidence=False)
            )
            if safe_gap and hint is None:
                self._depth_gap_resume_hint = (
                    *source, max(0.0, self._depth_last_approved_forward_rpm or 0.0),
                    self._depth_recovery_started_at, self._depth_quality_degraded,
                )
            elif not safe_gap:
                self._depth_gap_resume_hint = None
                self._depth_recovery_resume_base_rpm = 0.0
        if active is not None:
            # Retain only recovery bookkeeping, not a Depth/motor lease. Neither
            # repeated attempts nor wall-clock waiting advance its sample clock.
            if safe_gap:
                if not self._depth_recovery_pending_gap:
                    logger.info(
                        "depth_scheduling_recovery_gap capture_frame_id=%s uid=%s "
                        "action=pause original_sample_ts=%s ramp_end=%s "
                        "cap_rpm=%.2f motion_authorized=False deadline_renewed=False",
                        frame.capture_frame_id, active[0], active[1], active[5], active[4],
                    )
                self._depth_recovery_pending_gap = True
                return
            logger.info(
                "depth_scheduling_recovery_gap capture_frame_id=%s uid=%s "
                "action=discard reason=unsafe_or_reference_expired motion_authorized=False",
                frame.capture_frame_id, active[0],
            )
            self._depth_schedule_recovery = None
        resumable = bool(
            scheduling_only and anchor is not None and anchor[0] == self.active_target_id
            and any(p.track_id == anchor[0] for p in frame.persons)
            and self.search_state == "none" and not frame.hazard.active
            and not any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
            and state.safety_distance_m is None and not state.brake_latched
            and not self._target_stop_latched
            and not self._distance_longitudinally_untrusted(frame, check_hold_confidence=False)
            and state.raw_distance_m is None and 0 <= now - anchor[1] <= .18
        )
        if resumable:
            if not self._depth_recovery_pending_gap:
                logger.info(
                    "depth_recovery_gap capture_frame_id=%s uid=%s reason=%s "
                    "action=pause original_sample_ts=%s motion_authorized=False",
                    frame.capture_frame_id, self.active_target_id, state.source_detail, anchor[1],
                )
            self._depth_recovery_pending_gap = True
            return
        self._depth_quality_degraded = True
        self._depth_recovery_started_at = None
        self._depth_recovery_pending_gap = False
        self._depth_recovery_anchor = None

    def _apply_depth_recovery_cap(self, requested_speed: int, now: float) -> int:
        requested = max(0, int(requested_speed))
        recovery_started = self._depth_recovery_started_at
        if recovery_started is None:
            return requested
        elapsed = max(0.0, float(now) - float(recovery_started))
        stage1_sec = max(0.0, float(self.cfg.depth_recovery_stage1_sec))
        stage2_sec = max(stage1_sec, float(self.cfg.depth_recovery_stage2_sec))
        if elapsed < stage1_sec:
            cap_rpm = max(0, int(self.cfg.depth_recovery_stage1_rpm))
        elif elapsed < stage2_sec:
            cap_rpm = max(0, int(self.cfg.depth_recovery_stage2_rpm))
        else:
            self._depth_recovery_started_at = None
            logger.info("depth_speed_recovery_completed elapsed=%.0fms", elapsed * 1000.0)
            return requested
        cap_rpm = max(cap_rpm, getattr(self, "_depth_recovery_resume_base_rpm", 0.0))
        return min(
            requested,
            self._forward_percent_for_rpm(cap_rpm, allow_below_min=True),
        )

    def _far_closing_recovery_continuous(self, frame: SensorFrame, hint, now: float) -> bool:
        """A bounded recovery reference, never permission during a depth gap.

        Keep existing strict near/jump checks. Only a modest closure consistent
        with ongoing forward travel may avoid restarting the launch ramp.
        """
        state, fb = frame.distance_state, frame.steering_feedback
        if (hint is None or fb is None or not fb.trustworthy
                or self._target_stop_latched or self._distance_longitudinally_untrusted(frame)):
            return False
        values = (state.sample_timestamp, state.raw_distance_m, frame.distance_m,
                  hint[1], hint[2], fb.timestamp, fb.left_forward_rpm,
                  fb.right_forward_rpm, fb.yaw_rate_right_dps)
        if any(v is None or not math.isfinite(v) for v in values):
            return False
        gap = state.sample_timestamp-hint[1]
        delta = state.raw_distance_m-hint[2]
        near = self.cfg.target_distance_m + .30
        if not (0 < gap <= .18 and 0 <= now-state.sample_timestamp <= .18
                and 0 <= now-hint[1] <= .35 and 0 <= now-fb.timestamp <= .15
                and abs(fb.timestamp-state.sample_timestamp) <= .15
                and min(state.raw_distance_m, frame.distance_m, hint[2]) > near
                and -.12 <= delta < -.03 and -delta/gap <= 1.0
                and (state.raw_distance_m-near)/(-delta/gap) > .4
                and abs(fb.yaw_rate_right_dps) <= 15
                and all(0 < v <= self._longitudinal_feedforward.config.max_abs_ego_rpm
                        for v in (fb.left_forward_rpm, fb.right_forward_rpm))):
            return False
        ego_speed = .5*(fb.left_forward_rpm+fb.right_forward_rpm) * (
            self._longitudinal_feedforward.config.wheel_circumference_m/60.)
        # Reject closure beyond measured ego travel (+3cm noise allowance),
        # which may instead be an approaching person or a new depth surface.
        return -delta <= ego_speed*gap + .03

    def _scheduling_recovery_cap(self, frame, requested, now, hint):
        """Fresh-only, measured-wheel ramp after a scheduling gap, NOT a lease.

        Up to 500ms of reference history is retained without authorizing motion.
        The next physical sample still expires in 180ms. The old approved speed
        seeds the ramp, but is not a permanent ceiling. Every increase is bounded
        by new-sample time, current request and feedback + 150ms acceleration.
        Near/closing/unsafe recovery stays on the original strict path.
        """
        active = self._depth_schedule_recovery
        if not self.cfg.depth_measured_recovery_enable or (active is None and hint is None):
            return None
        state, fb = frame.distance_state, frame.steering_feedback
        # (uid, sample timestamp, raw distance, initial approval, cap, end time)
        ref = active if active is not None else (hint[0], hint[1], hint[2], hint[3], 0., now+.4)
        stamp, raw = state.sample_timestamp, state.raw_distance_m
        numbers = (now, stamp, raw, frame.distance_m, *ref[1:],
                   None if fb is None else fb.timestamp,
                   None if fb is None else fb.left_forward_rpm,
                   None if fb is None else fb.right_forward_rpm,
                   None if fb is None else fb.yaw_rate_right_dps)
        reason = None
        if any(v is None or not math.isfinite(v) for v in numbers):
            reason = "invalid_feedback_or_depth"
        elif (requested <= 0 or ref[0] != self.active_target_id or ref[0] is None
              or not any(p.track_id == ref[0] for p in frame.persons)
              or self.search_state != "none" or frame.hazard.active
              or any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
              or state.safety_distance_m is not None or state.brake_latched
              or self._target_stop_latched or self._distance_longitudinally_untrusted(frame)):
            reason = "identity_or_safety"
        elif not (0 <= now-stamp <= .18 and 0 <= now-fb.timestamp <= .15
                  and abs(fb.timestamp-stamp) <= .15 and fb.trustworthy
                  and abs(fb.yaw_rate_right_dps) <= 15
                  and all(0 <= v <= self._longitudinal_feedforward.config.max_abs_ego_rpm
                          for v in (fb.left_forward_rpm, fb.right_forward_rpm))):
            reason = "freshness_or_motion"
        elif not (min(raw, frame.distance_m, ref[2]) > self.cfg.target_distance_m+.30
                  and (-.03 <= raw-ref[2] <= .30 or far_closure_consistent(
                      distance=raw, previous_distance=ref[2], interval=stamp-ref[1],
                      ego_speed=.5*(fb.left_forward_rpm+fb.right_forward_rpm)*
                          self.cfg.distance_feedforward_wheel_circumference_m/60.,
                      near=self.cfg.target_distance_m+.30)) and ref[3] > 0):
            reason = "near_or_distance_change"
        elif (stamp < ref[1] or now-ref[1] > .50
              or (active is None and stamp == ref[1])):
            reason = "reference_expired_or_replayed"
        if reason is not None:
            self._depth_schedule_recovery = None
            if active is not None:
                self._depth_quality_degraded = True
                self._depth_recovery_started_at = None
                self._depth_recovery_resume_base_rpm = 0.
            logger.info(
                "depth_scheduling_recovery_rejected capture_frame_id=%s uid=%s reason=%s "
                "left_rpm=%s right_rpm=%s feedback_limit_rpm=%.3f",
                frame.capture_frame_id, self.active_target_id, reason,
                None if fb is None else fb.left_forward_rpm,
                None if fb is None else fb.right_forward_rpm,
                self._longitudinal_feedforward.config.max_abs_ego_rpm,
            )
            return None
        if active is not None and stamp == ref[1]:
            # A replay may reduce output, never restore a previously higher cap.
            cap = min(ref[4], requested*self.cfg.forward_max_rpm/100.)
            if cap < ref[4]:
                self._depth_schedule_recovery = (*ref[:4], cap, ref[5])
            return min(requested, self._forward_percent_for_rpm(cap, allow_below_min=True))
        measured = .5*(fb.left_forward_rpm+fb.right_forward_rpm)
        if raw-ref[2] < -.03:
            logger.info(
                "depth_far_closing_recovery capture_frame_id=%s uid=%s sample_ts=%s "
                "distance_m=%.3f delta_m=%.4f gap_ms=%.1f measured_rpm=%.2f "
                "policy=measured_bounded depth_ttl_ms=180",
                frame.capture_frame_id, ref[0], stamp, raw, raw-ref[2],
                1000*(stamp-ref[1]), measured,
            )
        rise = float(self.cfg.distance_pid_output_rise_rpm_per_sec)
        rise = min(240., rise) if rise > 0 else 240.
        dt = .05 if active is None else min(.10, stamp-ref[1])
        previous_cap = min(ref[3], measured) if active is None else ref[4]
        requested_rpm = requested*self.cfg.forward_max_rpm/100.
        cap = min(previous_cap+rise*dt, measured+rise*.15, requested_rpm)
        self._depth_schedule_recovery = (ref[0], stamp, raw, ref[3], cap, ref[5])
        self._depth_quality_degraded = False
        self._depth_recovery_started_at = None
        self._depth_recovery_pending_gap = False
        self._depth_recovery_resume_base_rpm = 0.
        # Elapsed time alone must not release a stopped car to a large request.
        # Complete only on a new safe sample whose measured envelope allows it.
        completed = now >= ref[5] and cap >= requested_rpm
        if completed:
            self._depth_schedule_recovery = None
        logger.info(
            "depth_scheduling_recovery capture_frame_id=%s uid=%s sample_ts=%s event=%s "
            "previous_approved_rpm=%.2f measured_rpm=%.2f cap_rpm=%.2f "
            "sample_gap_ms=%.1f depth_age_ms=%.1f deadline_renewed=False "
            "ramp_end=%.6f ramp_completed=%s requested_rpm=%.2f "
            "measured_envelope_rpm=%.2f",
            frame.capture_frame_id, ref[0], stamp, "start" if active is None else "advance",
            ref[3], measured, cap, 1000*(stamp-ref[1]), 1000*(now-stamp),
            ref[5], completed, requested_rpm, measured+rise*.15,
        )
        return min(requested, self._forward_percent_for_rpm(cap, allow_below_min=True))

    def _limit_depth_quality_forward_percent(
        self, frame: SensorFrame, requested_speed: int, now: float,
    ) -> int:
        approved = self._limit_depth_quality_forward_percent_impl(frame, requested_speed, now)
        # Only fresh approved samples can establish the next recovery starting point.
        if (self._is_fresh_depth_state(frame) and frame.distance_state.sample_timestamp is not None
                and 0 <= now-frame.distance_state.sample_timestamp <= .18
                and not self._distance_longitudinally_untrusted(frame)):
            self._depth_last_approved_forward_rpm = approved*self.cfg.forward_max_rpm/100.
        return approved

    def _limit_depth_quality_forward_percent_impl(
        self,
        frame: SensorFrame,
        requested_speed: int,
        now: float,
    ) -> int:
        requested = max(0, int(requested_speed))
        state = frame.distance_state
        if self.distance_pi_enabled:
            # The PI's single accelerator/brake already bounded this sample.
            # Do not re-enter the legacy 25/45 RPM or scheduling recovery ramp.
            if not self._distance_pi_frame_qualified(frame, now):
                self._pause_distance_pi(frame, now, "quality_limit_no_measurement")
                return 0
            result = self.last_distance_pid_result
            if result is None or self._distance_pid_last_sample_timestamp != state.sample_timestamp:
                return 0
            self._depth_quality_degraded = False
            self._depth_recovery_started_at = None
            self._depth_schedule_recovery = None
            self._depth_recovery_pending_gap = False
            self._depth_gap_resume_hint = None
            approved = min(requested, self._forward_percent_for_rpm(max(0, result.output_rpm), allow_below_min=True))
            self.accept_longitudinal_limit(state.sample_timestamp, approved * self.cfg.forward_max_rpm / 100.)
            return approved
        if str(getattr(state, "source", "")) != "vision_depth":
            return requested

        if self._is_fresh_depth_state(frame):
            stamp = state.sample_timestamp
            anchor = self._depth_recovery_anchor
            resume_rpm = 0.0
            continuity_restored = False
            hint = getattr(self, "_depth_gap_resume_hint", None)
            if self.cfg.depth_measured_recovery_enable and hint is not None:
                # This retains a recovery reference, NOT permission to move
                # during the gap. Current physical Depth must authorize anew.
                self._depth_gap_resume_hint = None
                fb = frame.steering_feedback
                far_closing = self._far_closing_recovery_continuous(frame, hint, now)
                if (stamp is not None and hint[0] == self.active_target_id
                        and hint[1] < stamp <= now and 0 <= now-stamp <= .18
                        and now-hint[1] <= .35
                        and self.search_state == "none" and not frame.hazard.active
                        and not any((frame.obstacles.front, frame.obstacles.left, frame.obstacles.right))
                        and any(p.track_id == hint[0] for p in frame.persons)
                        and state.safety_distance_m is None and not state.brake_latched
                        and state.raw_distance_m is not None
                        and (-.03 <= state.raw_distance_m-hint[2] <= .30 or far_closing)
                        and frame.distance_m is not None and frame.distance_m > self.cfg.target_distance_m + .03
                        and fb is not None and fb.trustworthy
                        and 0 <= now-fb.timestamp <= .15 and abs(fb.timestamp-stamp) <= .15
                        and all(math.isfinite(v) and 0 <= v <= self._longitudinal_feedforward.config.max_abs_ego_rpm for v in (
                            fb.left_forward_rpm, fb.right_forward_rpm))):
                    resume_rpm = min(hint[3], .5*(fb.left_forward_rpm+fb.right_forward_rpm))
                    # A failed measurement attempt is not a physical stop.
                    # Read the actual runtime grant, including its original
                    # deadline, revocation and FF-expiry guards. No cached
                    # controller request can stand in for live authorization.
                    reader = self._live_longitudinal_authority_reader
                    live = reader(self.active_target_id) if callable(reader) else None
                    normal_continuation = bool(
                        self.cfg.distance_approach_enable and live is not None
                        and live[0] == "forward" and live[1] > 0 and live[2] == hint[0]
                        and 0 <= now-live[3] <= .18
                        and self._depth_schedule_recovery is None
                        and hint[4] is None and not hint[5]
                        and min(fb.left_forward_rpm, fb.right_forward_rpm) > 1.
                    )
                    # Compare physical sample times, not processing time to
                    # the OLD lease deadline. No motion was permitted in an
                    # expired gap. Only a NEW valid Depth and measured wheels
                    # still near the previous approval may retain ramp progress.
                    continuity_restored = bool(
                        len(hint) >= 6 and not hint[5] and hint[3] > 0
                        and not self._target_stop_latched
                        and not self._distance_longitudinally_untrusted(frame)
                        and 0 < stamp - hint[1] <= .18
                        and (resume_rpm >= .9 * hint[3] or normal_continuation)
                    )
                    if continuity_restored:
                        if normal_continuation:
                            rise = max(0., min(240., self.cfg.distance_pid_output_rise_rpm_per_sec))
                            previous_rpm = live[1]*self.cfg.forward_max_rpm/100.
                            normal_cap = previous_rpm + rise*max(0., min(.10, stamp-live[3]))
                            requested = min(requested, int(math.floor(
                                100.*normal_cap/self.cfg.forward_max_rpm)))
                        self._depth_quality_degraded = False
                        self._depth_recovery_started_at = hint[4]
                        self._depth_recovery_resume_base_rpm = resume_rpm
                        logger.info(
                            "depth_recovery_continuity_restored capture_frame_id=%s uid=%s "
                            "sample_ts=%s physical_gap_ms=%.1f processing_gap_ms=%.1f "
                            "previous_approved_rpm=%.1f measured_base_rpm=%.1f "
                            "original_ramp_started=%s deadline_renewed=False normal_live_continuation=%s",
                            frame.capture_frame_id, hint[0], stamp, (stamp-hint[1])*1000,
                            (now-hint[1])*1000, hint[3], .5*(fb.left_forward_rpm+fb.right_forward_rpm),
                            hint[4], normal_continuation,
                        )
                    logger.info(
                        "depth_measured_recovery capture_frame_id=%s uid=%s gap_ms=%.1f "
                        "previous_approved_rpm=%.1f measured_base_rpm=%.1f resume_cap_rpm=%.1f "
                        "depth_age_ms=%.1f sample_ts=%s deadline_renewed=False distance_policy=%s "
                        "distance_delta_m=%.4f physical_gap_ms=%.1f continuity_restored=%s "
                        "left_rpm=%.2f right_rpm=%.2f feedback_limit_rpm=%.3f",
                        frame.capture_frame_id, hint[0], (now-hint[1])*1000,
                        hint[3], .5*(fb.left_forward_rpm+fb.right_forward_rpm), resume_rpm,
                        (now-stamp)*1000, stamp,
                        "far_closing_measured" if far_closing else "nonclosing",
                        state.raw_distance_m-hint[2], (stamp-hint[1])*1000, continuity_restored,
                        fb.left_forward_rpm, fb.right_forward_rpm,
                        self._longitudinal_feedforward.config.max_abs_ego_rpm,
                    )
            if self._depth_recovery_pending_gap:
                resume = continuity_restored or bool(
                    anchor is not None and anchor[0] == self.active_target_id
                    and stamp is not None and 0 <= now - anchor[1] <= .18
                    and anchor[1] < stamp <= now
                    and state.raw_distance_m is not None
                    and abs(state.raw_distance_m - anchor[2]) <= .30
                    and state.safety_distance_m is None and not state.brake_latched
                )
                logger.info(
                    "depth_recovery_gap capture_frame_id=%s uid=%s action=%s sample_ts=%s "
                    "original_sample_ts=%s original_ramp_started=%s",
                    frame.capture_frame_id, self.active_target_id, "resume" if resume else "restart",
                    stamp, None if anchor is None else anchor[1], self._depth_recovery_started_at,
                )
                if not resume:
                    self._depth_quality_degraded = True
                    self._depth_recovery_started_at = None
                self._depth_recovery_pending_gap = False
            if stamp is not None and state.raw_distance_m is not None and 0 <= now - stamp <= .18:
                self._depth_recovery_anchor = (self.active_target_id, stamp, state.raw_distance_m)
            # Do not weaken the existing direct-continuity or far-closing path.
            # This alternative handles measured deceleration after a real gap.
            if not continuity_restored or self._depth_schedule_recovery is not None:
                scheduled = self._scheduling_recovery_cap(frame, requested, now, hint)
                if scheduled is not None:
                    if scheduled < requested:
                        scale = self.cfg.forward_max_rpm/100.
                        logger.info(
                            "depth_speed_recovery_limit capture_frame_id=%s uid=%s sample_ts=%s "
                            "requested_percent=%s approved_percent=%s requested_rpm=%.2f "
                            "approved_rpm=%.2f lost_rpm=%.2f recovery_policy=scheduling",
                            frame.capture_frame_id, self.active_target_id, stamp, requested, scheduled,
                            requested*scale, scheduled*scale, (requested-scheduled)*scale,
                        )
                    return scheduled
            if self._depth_quality_degraded:
                self._depth_quality_degraded = False
                self._depth_recovery_resume_base_rpm = resume_rpm
                self._depth_recovery_started_at = float(now)
                logger.info(
                    "depth_speed_recovery_started distance=%s stage1=%drpm/%.0fms "
                    "stage2=%drpm/%.0fms",
                    "none" if frame.distance_m is None else f"{float(frame.distance_m):.2f}m",
                    int(self.cfg.depth_recovery_stage1_rpm),
                    float(self.cfg.depth_recovery_stage1_sec) * 1000.0,
                    int(self.cfg.depth_recovery_stage2_rpm),
                    float(self.cfg.depth_recovery_stage2_sec) * 1000.0,
                )
            approved = self._apply_depth_recovery_cap(requested, now)
            if approved < requested:
                logger.info(
                    "depth_speed_recovery_limit capture_frame_id=%s uid=%s sample_ts=%s "
                    "requested_percent=%s approved_percent=%s elapsed_ms=%s "
                    "requested_rpm=%.2f approved_rpm=%.2f lost_rpm=%.2f",
                    frame.capture_frame_id, self.active_target_id, stamp, requested, approved,
                    None if self._depth_recovery_started_at is None else
                    round(1000.0 * (now - self._depth_recovery_started_at), 1),
                    requested * self.cfg.forward_max_rpm / 100.0,
                    approved * self.cfg.forward_max_rpm / 100.0,
                    (requested - approved) * self.cfg.forward_max_rpm / 100.0,
                )
            return approved

        anchor_age = None
        if self._last_target_distance_at is not None:
            anchor_age = max(0.0, float(now) - float(self._last_target_distance_at))
        short_estimate = bool(
            frame.distance_m is not None
            and anchor_age is not None
            and anchor_age <= 0.20
        )
        if short_estimate:
            # The 30Hz Depth loop alternates fresh samples and short fused-hold
            # samples. Preserve an active recovery ramp across those hold frames
            # instead of allowing the second control tick to jump to full speed.
            return self._apply_depth_recovery_cap(requested, now)

        self._note_depth_quality_failure(frame, now)
        medium_sec = max(0.20, float(self.cfg.depth_medium_confidence_hold_sec))
        if anchor_age is not None and anchor_age <= medium_sec:
            cap_rpm = max(0, int(self.cfg.depth_medium_confidence_rpm))
            return min(
                requested,
                self._forward_percent_for_rpm(cap_rpm, allow_below_min=True),
            )
        return 0

    def _visible_base_forward_percent(
        self,
        frame: SensorFrame,
        *,
        now: Optional[float] = None,
    ) -> int:
        cfg = self.cfg
        if self._distance_longitudinally_untrusted(frame):
            return 0
        if frame.distance_m is not None:
            speed = self._forward_percent_for_distance(float(frame.distance_m), now=now)
        else:
            speed = max(0, min(int(cfg.max_forward_percent), int(cfg.distance_missing_forward_percent)))
            if 0 < speed < cfg.min_forward_percent:
                speed = cfg.min_forward_percent
        speed = self._limit_mmwave_hold_forward_percent(frame, speed)
        return self._limit_depth_quality_forward_percent(
            frame,
            speed,
            time.monotonic() if now is None else float(now),
        )

    @staticmethod
    def _is_mmwave_hold(frame: SensorFrame) -> bool:
        state = frame.distance_state
        return bool(
            frame.distance_m is not None
            and str(getattr(state, "source", "")) == "vision_mmwave"
            and str(getattr(state, "source_detail", "")).endswith("_hold")
        )

    def _limit_mmwave_hold_forward_percent(self, frame: SensorFrame, speed: int) -> int:
        requested = max(0, int(speed))
        previous = self._last_mmwave_motion_speed_percent
        if not self._is_mmwave_hold(frame):
            if self._mmwave_speed_recovery_active and previous is not None and requested > previous:
                step = max(1, int(self.cfg.mmwave_hold_recover_step_percent))
                output = min(requested, previous + step)
                logger.info(
                    "mmwave_speed_slew phase=recover requested=%d previous=%d output=%d step=%d",
                    requested,
                    previous,
                    output,
                    step,
                )
            else:
                output = requested
            self._last_mmwave_motion_speed_percent = output
            self._mmwave_speed_recovery_active = output != requested
            return output
        hold_cap = max(
            0,
            min(int(self.cfg.max_forward_percent), int(self.cfg.mmwave_hold_forward_percent)),
        )
        fusion_mode = str(getattr(frame.distance_state, "fusion_mode", ""))
        fusion_confidence = float(getattr(frame.distance_state, "fusion_confidence", 0.0))
        if fusion_mode and fusion_mode not in ("disabled", "unavailable"):
            low_cap = max(
                0,
                min(hold_cap, int(self.cfg.mmwave_fusion_low_confidence_forward_percent)),
            )
            # 融合器置信度从 1.0 平滑降到配置下限，避免异常期间突然刹停。
            confidence = max(0.0, min(1.0, fusion_confidence))
            hold_cap = max(low_cap, int(round(low_cap + (hold_cap - low_cap) * confidence)))
        target = min(requested, hold_cap)
        if previous is None or previous <= target:
            output = target
        else:
            step = max(1, int(self.cfg.mmwave_hold_decel_step_percent))
            output = max(target, previous - step)
            logger.info(
                "mmwave_speed_slew phase=hold requested=%d previous=%d output=%d cap=%d step=%d",
                requested,
                previous,
                output,
                hold_cap,
                step,
            )
        self._last_mmwave_motion_speed_percent = output
        self._mmwave_speed_recovery_active = True
        return output

    def _fallback_forward_percent(self) -> int:
        cfg = self.cfg
        speed = max(0, min(int(cfg.max_forward_percent), int(cfg.distance_missing_forward_percent)))
        if 0 < speed < cfg.min_forward_percent:
            speed = cfg.min_forward_percent
        return speed

    def _remember_target_distance(self, frame: SensorFrame, now: float) -> None:
        distance = frame.distance_m
        if distance is None:
            distance = getattr(frame.distance_state, "used_distance_m", None)
        if distance is None:
            return
        state = frame.distance_state
        source = str(getattr(state, "source", ""))
        source_detail = str(getattr(state, "source_detail", ""))
        raw_distance = getattr(state, "raw_distance_m", None)
        # Held/fused samples are predictions, not new measurements. They must
        # never replace the last trusted anchor or extend its freshness window;
        # otherwise a drifting visual/depth hold can feed the PID indefinitely.
        range_source = source in ("vision_mmwave", "vision_depth")
        fresh_range = raw_distance is not None and not source_detail.endswith("_hold")
        if range_source and not fresh_range:
            return
        self._last_target_distance_m = float(distance)
        if not range_source or fresh_range:
            self._last_target_distance_at = float(now)

    def _distance_missing_camera_hold_action(
        self,
        frame: SensorFrame,
        now: float,
        *,
        max_depth_hold_sec: Optional[float] = None,
    ) -> Optional[ControlAction]:
        """Use the last fresh range briefly while the same visual target remains visible."""
        last_distance = self._last_target_distance_m
        last_distance_at = self._last_target_distance_at
        if last_distance is None or last_distance_at is None:
            return None
        if str(getattr(frame.distance_state, "source", "")) == "vision_depth":
            max_hold_sec = max(
                0.0,
                min(
                    0.18 if max_depth_hold_sec is None else float(max_depth_hold_sec),
                    float(self.cfg.depth_medium_confidence_hold_sec),
                ),
            )
            hold_rpm = int(self.cfg.depth_medium_confidence_rpm)
            detail = str(getattr(frame.distance_state, "source_detail", "")).lower()
            mode = str(getattr(frame.distance_state, "fusion_mode", "")).lower()
            # Only a fused/held sample may use this path. Re-anchors, far
            # background guards and jump confirmation are intentionally hard
            # rejects even when an old distance is still available.
            if frame.distance_m is None or not ("hold" in detail or mode.endswith("_hold")):
                return None
            if any(
                token in detail
                for token in (
                    "reanchored_after_timeout",
                    "distance_jump_pending",
                    "far_background_guard",
                    "depth_expired",
                    "no_valid_depth",
                )
            ):
                return None
        else:
            max_hold_sec = max(0.0, float(self.cfg.lost_forward_hold_max_sec))
            hold_rpm = int(self.cfg.lost_forward_hold_rpm)
        if max_hold_sec <= 0.0 or float(now) - float(last_distance_at) > max_hold_sec:
            return None
        min_distance = max(
            float(self.cfg.brake_distance_m),
            float(self.cfg.target_distance_m),
            float(self.cfg.lost_forward_hold_min_distance_m),
        )
        if float(last_distance) <= min_distance:
            return None
        if frame.hazard.active or frame.obstacles.front:
            return None
        if bool(getattr(frame.distance_state, "brake_latched", False)):
            return None
        speed = self._forward_percent_for_rpm(hold_rpm, allow_below_min=True)
        if speed <= 0:
            return None
        return ControlAction.forward(speed, "distance_missing_camera_hold")

    def _turn_action_for_visible_target(
        self,
        edge_type: str,
        cx: float,
        width: int,
        frame: SensorFrame,
        motion_dx_ratio: float = 0.0,
        projected_x_ratio: Optional[float] = None,
        motion_led: bool = False,
    ) -> Optional[ControlAction]:
        if edge_type not in ("left", "right") or width <= 0:
            return None

        base_speed = self._visible_base_forward_percent(frame, now=time.monotonic())
        if base_speed <= 0 and frame.distance_m is not None and not self.cfg.distance_parking_enable:
            # Inside the longitudinal hold band, replace any stale forward or
            # steer target with zero instead of restoring the old minimum RPM.
            return ControlAction.forward(0, "target_distance_hold_off_center")
        if base_speed <= 0 and not self.cfg.distance_parking_enable:
            # In IR-only parking mode a close reading may not create STOP, but
            # stale forward motion must still be replaced.
            base_speed = max(1, min(int(self.cfg.max_forward_percent), int(self.cfg.min_forward_percent)))
        if base_speed <= 0 and frame.distance_m is None:
            last_distance = getattr(frame.distance_state, "used_distance_m", None)
            if last_distance is None:
                last_distance = self._last_target_distance_m
            distance_brake_latched = bool(getattr(frame.distance_state, "brake_latched", False))
            if (
                not distance_brake_latched
                and last_distance is not None
                and float(last_distance) >= float(self.cfg.brake_distance_m)
            ):
                # 毫米波短暂失配时只给 STEER 一个非零控制门槛。当前 raw 模式仍按
                # steer_raw_target=15rpm 下发，不会恢复普通直行或使用前进速度曲线。
                base_speed = max(1, min(int(self.cfg.max_forward_percent), int(self.cfg.min_forward_percent)))
        if base_speed <= 0:
            return None
        margin = max(0.02, min(0.45, float(self.cfg.visible_steer_strong_margin_ratio)))
        x_ratio = max(0.0, min(1.0, float(cx) / float(width)))
        projected = x_ratio if projected_x_ratio is None else float(projected_x_ratio)
        moving_outward = (
            (edge_type == "left" and motion_dx_ratio < 0.0)
            or (edge_type == "right" and motion_dx_ratio > 0.0)
        )
        left_enter = self._ratio_or_default(self.cfg.steer_enter_left_ratio, self.cfg.center_left_ratio)
        right_enter = self._ratio_or_default(self.cfg.steer_enter_right_ratio, self.cfg.center_right_ratio)
        if edge_type == "left":
            # 向左外移时使用当前位置和短期外推中更靠左的一个，提前加大修正。
            correction_position = min(x_ratio, projected)
            position_span = max(0.01, left_enter - margin)
            correction_strength = (left_enter - correction_position) / position_span
        else:
            # 向右外移时使用更靠右的一个；回到中心的轨迹不会凭空放大轮差。
            correction_position = max(x_ratio, projected)
            position_span = max(0.01, (1.0 - margin) - right_enter)
            correction_strength = (correction_position - right_enter) / position_span
        correction_strength = max(0.0, min(1.0, float(correction_strength)))

        strong_motion_threshold = max(0.0, float(self.cfg.visible_motion_strong_ratio))
        if moving_outward and abs(float(motion_dx_ratio)) > 0.0:
            if strong_motion_threshold <= 0.0:
                motion_strength = 1.0
            else:
                motion_strength = min(1.0, abs(float(motion_dx_ratio)) / strong_motion_threshold)
            correction_strength = max(correction_strength, motion_strength)

        # 三个控制点做分段线性插值。靠近中心只给很小轮差，越靠边轮差越大；
        # raw RPM 最终会自然量化成多个稳定档位，避免原先只在 12/16 与 10/17 间跳变。
        fine_inner = max(0, min(100, int(self.cfg.visible_steer_fine_inner_ratio_percent)))
        fine_outer = max(0, min(150, int(self.cfg.visible_steer_fine_outer_ratio_percent)))
        normal_inner = max(0, min(100, int(self.cfg.visible_steer_inner_ratio_percent)))
        normal_outer = max(0, min(150, int(self.cfg.visible_steer_outer_ratio_percent)))
        strong_inner = max(0, min(100, int(self.cfg.visible_steer_strong_inner_ratio_percent)))
        strong_outer = max(0, min(150, int(self.cfg.visible_steer_strong_outer_ratio_percent)))
        if correction_strength <= 0.5:
            blend = correction_strength * 2.0
            inner_ratio = int(round(fine_inner + (normal_inner - fine_inner) * blend))
            outer_ratio = int(round(fine_outer + (normal_outer - fine_outer) * blend))
        else:
            blend = (correction_strength - 0.5) * 2.0
            inner_ratio = int(round(normal_inner + (strong_inner - normal_inner) * blend))
            outer_ratio = int(round(normal_outer + (strong_outer - normal_outer) * blend))

        strong = correction_strength >= 0.999
        reason = "person_%s_strong" % edge_type if strong else "person_%s" % edge_type
        if motion_led or (moving_outward and correction_strength >= 0.999):
            reason += "_motion"
        if self._is_mmwave_hold(frame):
            reason += "_mmwave_hold"
        elif frame.distance_m is None:
            reason += "_distance_missing"
        if edge_type == "left":
            return ControlAction.steer_left(base_speed, inner_ratio, outer_ratio, reason)
        return ControlAction.steer_right(base_speed, inner_ratio, outer_ratio, reason)

    def _pid_direction_guard_reason(
        self,
        current_x_ratio: float,
        motion_dx_ratio: float,
        correction_rpm: int,
    ) -> Optional[str]:
        """Block powered yaw away from the target's current visual side."""
        correction = int(correction_rpm)
        if correction == 0 or current_x_ratio == 0.5:
            return None

        target_side = 1 if current_x_ratio > 0.5 else -1
        correction_side = 1 if correction > 0 else -1
        if target_side == correction_side:
            return None

        return "target_still_outside_center"

    @staticmethod
    def _visual_age_sec(frame: SensorFrame, now: float) -> Optional[float]:
        capture_timestamp = float(getattr(frame, "capture_timestamp", 0.0) or 0.0)
        if capture_timestamp <= 0.0:
            return None
        return max(0.0, float(now) - capture_timestamp)

    def _visible_pid_center_hold(self, x_ratio: float) -> bool:
        if (
            self.cfg.steer_release_left_ratio is None
            or self.cfg.steer_release_right_ratio is None
        ):
            return False
        left = max(0.0, min(1.0, float(self.cfg.center_left_ratio)))
        right = max(left, min(1.0, float(self.cfg.center_right_ratio)))
        return left + 1e-6 < float(x_ratio) < right - 1e-6

    def _update_lateral_pid(
        self,
        pid: VisualSteeringPid,
        *,
        x_ratio: float,
        base_rpm: int,
        feedback,
        now: float,
        target_image_rate_dps: Optional[float] = None,
        max_correction_rpm: Optional[float] = None,
        visual_age_sec: Optional[float] = None,
        near_distance_mode: bool = False,
        hold_zero: bool = False,
    ) -> VisualSteeringPidResult:
        """Apply the same yaw constraints on camera and encoder refresh ticks."""
        disable_rate_feedforward = bool(
            near_distance_mode and self.cfg.near_distance_disable_rate_feedforward
        )
        result = pid.update(
            max(0.0, min(1.0, float(x_ratio))),
            max(0, int(base_rpm)),
            feedback,
            now=float(now),
            target_image_rate_dps=target_image_rate_dps,
            max_correction_override_rpm=0.0 if hold_zero else max_correction_rpm,
            visual_age_sec=visual_age_sec,
            target_rate_feedforward_max_dps_override=(
                0.0 if disable_rate_feedforward or hold_zero else None
            ),
            target_speed_match_max_closing_dps_override=(
                0.0 if disable_rate_feedforward or hold_zero else None
            ),
        )
        near_center_zero = bool(
            disable_rate_feedforward
            and abs(result.visual_error_deg) <= max(0.0, float(pid.config.deadband_deg))
            and abs(result.desired_yaw_rate_dps) <= 1e-6
        )
        if hold_zero or near_center_zero:
            # Startup assistance needs a nonzero tracking request. In near
            # mode an outward bbox rate can select a side inside the deadband
            # even though feedforward is disabled and desired yaw is zero.
            # Preserve that zero and rearm only for a future tracking sample.
            pid.reset()
            result = replace(
                result,
                correction_rpm=0,
                desired_yaw_rate_dps=0.0,
                feedforward_rpm=0.0,
                rate_p_rpm=0.0,
                rate_i_rpm=0.0,
                unsaturated_rpm=0.0,
                output_floor_rpm=0.0,
                output_floor_reason="center_hold",
                startup_kick_active=False,
                startup_kick_elapsed_sec=0.0,
                startup_kick_release_reason="center_hold",
            )
            pid.last_result = result
        return result

    def _parked_startup_floor(
        self, result: VisualSteeringPidResult, x_ratio: float
    ) -> VisualSteeringPidResult:
        correction = int(result.correction_rpm)
        startup_floor = min(
            max(1, int(self.cfg.parked_recenter_min_rpm)),
            max(0, int(math.floor(result.correction_limit_rpm))),
        )
        if (
            result.startup_kick_active
            and correction * (float(x_ratio) - 0.5) > 0.0
            and abs(correction) < startup_floor
        ):
            return replace(
                result,
                correction_rpm=startup_floor if correction > 0 else -startup_floor,
                output_floor_rpm=float(startup_floor),
            )
        return result

    def refresh_visible_lateral_pid(
        self,
        *,
        x_ratio: float,
        base_rpm: int,
        feedback,
        now: float,
        motion_dx_ratio: float = 0.0,
        target_image_rate_dps: Optional[float] = None,
        max_correction_rpm: Optional[float] = None,
        visual_age_sec: Optional[float] = None,
        hold_zero: bool = False,
    ) -> VisualSteeringPidResult:
        """Refresh the existing visible-target PID between detector results.

        The caller owns target identity, intent freshness, motor mode and all
        safety gates. This method only advances the same PID used by the vision
        decision path, keeping the controller state single-owned.
        """
        result = self._update_lateral_pid(
            self._visual_steering_pid,
            x_ratio=x_ratio,
            base_rpm=base_rpm,
            feedback=feedback,
            now=float(now),
            target_image_rate_dps=target_image_rate_dps,
            max_correction_rpm=max_correction_rpm,
            visual_age_sec=visual_age_sec,
            hold_zero=hold_zero or self._visible_pid_center_hold(x_ratio),
        )
        guard_reason = self._pid_direction_guard_reason(
            float(x_ratio),
            float(motion_dx_ratio),
            int(result.correction_rpm),
        )
        if guard_reason is not None:
            result = replace(result, correction_rpm=0)
        self.last_steering_pid_result = result
        return result

    def refresh_parked_lateral_pid(
        self,
        *,
        x_ratio: float,
        base_rpm: int,
        feedback,
        now: float,
        motion_dx_ratio: float = 0.0,
        target_image_rate_dps: Optional[float] = None,
        max_correction_rpm: Optional[float] = None,
        visual_age_sec: Optional[float] = None,
        near_distance_mode: bool = False,
        hold_zero: bool = False,
    ) -> VisualSteeringPidResult:
        """Refresh the in-place yaw PID between detector results."""
        result = self._update_lateral_pid(
            self._parked_recenter_pid,
            x_ratio=x_ratio,
            base_rpm=base_rpm,
            feedback=feedback,
            now=float(now),
            target_image_rate_dps=target_image_rate_dps,
            max_correction_rpm=max_correction_rpm,
            visual_age_sec=visual_age_sec,
            near_distance_mode=near_distance_mode,
            hold_zero=hold_zero,
        )
        guard_reason = self._pid_direction_guard_reason(
            float(x_ratio),
            float(motion_dx_ratio),
            int(result.correction_rpm),
        )
        if guard_reason is not None:
            result = replace(result, correction_rpm=0)
        result = self._parked_startup_floor(result, x_ratio)
        self.last_steering_pid_result = result
        return result

    def _pid_action_for_visible_target(
        self,
        target: PersonTarget,
        frame: SensorFrame,
        now: float,
        motion_dx_ratio: float = 0.0,
        target_image_rate_dps: Optional[float] = None,
        max_correction_rpm: Optional[float] = None,
    ) -> Optional[ControlAction]:
        if not self.cfg.visible_steering_pid_enable or frame.width <= 0:
            self.last_steering_pid_result = None
            return None

        distance_fallback = frame.distance_m is None
        vision_depth_fallback = bool(
            distance_fallback
            and str(getattr(frame.distance_state, "source", "")) == "vision_depth"
        )
        if distance_fallback:
            if self.distance_pi_enabled:
                self._pause_distance_pi(frame, now, "visual_distance_missing")
            else:
                self._reset_distance_pid()
            if vision_depth_fallback:
                # Depth confidence controls only longitudinal speed. The
                # camera/encoder yaw loop keeps running even after the >600ms
                # longitudinal timeout has reduced base speed to zero.
                base_speed = self._visible_base_forward_percent(frame, now=now)
                base_rpm = max(
                    0,
                    int(round(int(self.cfg.forward_max_rpm) * int(base_speed) / 100.0)),
                )
            else:
                # Non-Depth callers do not carry the staged Astra confidence
                # state. Preserve the established visible-target fallback.
                base_rpm = max(
                    1,
                    min(
                        int(self.cfg.forward_max_rpm),
                        int(self.cfg.visible_steering_pid_fallback_base_rpm),
                    ),
                )
                base_speed = self._forward_percent_for_rpm(
                    base_rpm,
                    allow_below_min=True,
                )
        else:
            base_speed = self._visible_base_forward_percent(frame, now=now)

        if not distance_fallback:
            base_rpm = max(
                1,
                int(round(int(self.cfg.forward_max_rpm) * int(base_speed) / 100.0)),
            )
        cx, _cy = target.center
        current_x_ratio = float(cx) / float(max(1, frame.width))
        # Position error and target velocity are separate inputs. The old path
        # added a multi-frame displacement directly to x, which changed gain
        # whenever detector cadence changed and duplicated the PID derivative.
        pid_x_ratio = current_x_ratio
        correction_override = (
            float(self.cfg.visible_steering_pid_fallback_max_correction_rpm)
            if distance_fallback
            else None
        )
        if max_correction_rpm is not None:
            limited_override = max(0.0, float(max_correction_rpm))
            correction_override = (
                limited_override
                if correction_override is None
                else min(correction_override, limited_override)
            )
        result = self._update_lateral_pid(
            self._visual_steering_pid,
            x_ratio=pid_x_ratio,
            base_rpm=base_rpm,
            feedback=frame.steering_feedback,
            now=now,
            target_image_rate_dps=target_image_rate_dps,
            max_correction_rpm=correction_override,
            visual_age_sec=self._visual_age_sec(frame, now),
            hold_zero=self._visible_pid_center_hold(current_x_ratio),
        )
        self.last_steering_pid_result = result
        if target_image_rate_dps is not None and abs(target_image_rate_dps) >= 1.0:
            logger.info(
                "visual_pid_target_rate current_x=%.3f dx=%+.3f image_rate=%+.2fdps "
                "bearing_rate=%+.2fdps feedforward=%+.2fdps",
                current_x_ratio,
                float(motion_dx_ratio),
                float(result.target_image_rate_dps),
                float(result.target_bearing_rate_dps),
                float(result.target_rate_feedforward_dps),
            )
        # PID 会在人物明显偏离中心时主动压低直行基准，让增大的轮差真正
        # 转化为车身角速度；接近中心时保持原距离曲线速度。
        base_rpm = int(result.base_rpm)
        base_speed = self._forward_percent_for_rpm(base_rpm, allow_below_min=True)
        correction = int(result.correction_rpm)
        direction_guard_reason = self._pid_direction_guard_reason(
            current_x_ratio,
            motion_dx_ratio,
            correction,
        )
        if direction_guard_reason is not None:
            result = replace(result, correction_rpm=0)
            self.last_steering_pid_result = result
            logger.info(
                "visual_pid_direction_guard current_x=%.3f pid_x=%.3f dx=%+.3f "
                "raw=%+drpm output=0rpm reason=%s measured_yaw=%+.2fdps "
                "desired_yaw=%+.2fdps",
                current_x_ratio,
                pid_x_ratio,
                float(motion_dx_ratio),
                correction,
                direction_guard_reason,
                float(result.measured_yaw_rate_dps),
                float(result.desired_yaw_rate_dps),
            )
            return ControlAction.forward(base_speed, "visual_pid_direction_guard_hold")

        # The latest camera position is the final authority inside the center
        # band.  Filtered error and encoder yaw may still contain the previous
        # turn, but counter-steering at this point makes the chassis cross the
        # center and start a left/right limit cycle.  Coast with zero
        # differential until the target leaves the band again.
        center_left = max(
            0.0,
            min(
                1.0,
                float(self.cfg.center_left_ratio),
            ),
        )
        center_right = max(
            center_left,
            min(
                1.0,
                float(self.cfg.center_right_ratio),
            ),
        )
        # Keep the configured edges available for the normal PID hysteresis;
        # only the interior of the band is an unconditional center hold.
        if self._visible_pid_center_hold(current_x_ratio):
            suppressed_correction = int(result.correction_rpm)
            if int(result.correction_rpm) != 0 or result.output_floor_reason != "center_hold":
                result = replace(
                    result,
                    correction_rpm=0,
                    output_floor_rpm=0.0,
                    output_floor_reason="center_hold",
                )
                self.last_steering_pid_result = result
            logger.info(
                "visual_pid_center_hold current_x=%.3f center=[%.3f,%.3f] "
                "suppressed_correction=%+drpm measured_yaw=%+.2fdps",
                current_x_ratio,
                center_left,
                center_right,
                suppressed_correction,
                float(result.measured_yaw_rate_dps),
            )
            return ControlAction.forward(base_speed, "visual_pid_center_hold")

        if correction == 0:
            correction = self._pid_zero_guard_correction(
                result,
                current_x_ratio=current_x_ratio,
                center_left_ratio=float(self.cfg.center_left_ratio),
                center_right_ratio=float(self.cfg.center_right_ratio),
                min_correction_rpm=int(self.cfg.parked_recenter_min_rpm),
            )
            if correction == 0:
                return None
            logger.info(
                "visual_pid_zero_guard current_x=%.3f center=[%.3f,%.3f] "
                "desired=%+.1fdps measured=%+.1fdps correction=%+drpm",
                current_x_ratio,
                float(self.cfg.center_left_ratio),
                float(self.cfg.center_right_ratio),
                float(result.desired_yaw_rate_dps),
                float(result.measured_yaw_rate_dps),
                correction,
            )

        feedback_tag = "encoder" if result.feedback_used else "camera"
        reason = f"visual_pid_{'right' if correction > 0 else 'left'}_{feedback_tag}"
        if self._is_mmwave_hold(frame):
            reason += "_mmwave_hold"
        elif frame.distance_m is None:
            reason += "_distance_missing"
            if base_rpm <= 0:
                reason += "_yaw_only"
        if correction > 0:
            return ControlAction.steer_right(
                base_speed,
                100,
                100,
                reason,
                correction_rpm=correction,
            )
        return ControlAction.steer_left(
            base_speed,
            100,
            100,
            reason,
            correction_rpm=-correction,
        )

    @staticmethod
    def _pid_zero_guard_correction(
        result: VisualSteeringPidResult,
        *,
        current_x_ratio: float,
        center_left_ratio: float,
        center_right_ratio: float,
        min_correction_rpm: int,
    ) -> int:
        """Keep a PID zero crossing from masquerading as settled yaw."""
        if int(result.correction_rpm) != 0:
            return int(result.correction_rpm)

        # A same-side overspeed zero is intentional. Reconstructing an
        # opposite minimum-RPM command here would undo the PID's damping and
        # recreate the left/right limit cycle.
        if result.same_direction_overspeed_braking or result.output_floor_reason in {
            "same_direction_overspeed_coast",
            "predictive_brake_coast",
            "visual_direction_guard",
            "center_hold",
            "yaw_damping_coast",
        }:
            return 0

        center_left = min(float(center_left_ratio), float(center_right_ratio))
        center_right = max(float(center_left_ratio), float(center_right_ratio))
        in_center_band = center_left <= float(current_x_ratio) <= center_right
        measured_rate = float(result.measured_yaw_rate_dps)
        feedback_unsettled = bool(
            result.feedback_used and abs(measured_rate) >= 2.0
        )
        if in_center_band and not feedback_unsettled:
            return 0

        desired_rate = float(result.desired_yaw_rate_dps)
        visual_error = float(result.visual_error_deg)
        same_direction_overspeed = bool(
            result.feedback_used
            and desired_rate * measured_rate >= 0.0
            and abs(measured_rate) > abs(desired_rate)
        )
        if feedback_unsettled and (in_center_band or same_direction_overspeed):
            direction = -1 if measured_rate > 0.0 else 1
            brake_evidence_rpm = max(
                abs(float(result.overspeed_brake_rpm)),
                abs(float(result.rate_p_rpm)),
            )
        else:
            steering_evidence = visual_error if abs(visual_error) > 1e-6 else desired_rate
            if abs(steering_evidence) <= 1e-6:
                return 0
            direction = 1 if steering_evidence > 0.0 else -1
            brake_evidence_rpm = 0.0

        correction_limit = max(0, int(round(float(result.correction_limit_rpm))))
        if correction_limit <= 0:
            return 0
        minimum = max(1, int(min_correction_rpm))
        magnitude = max(minimum, int(round(brake_evidence_rpm)))
        return int(direction * min(correction_limit, magnitude))

    def _reverse_decision_with_visual_steering(
        self,
        decision: ControlDecision,
        target: PersonTarget,
        frame: SensorFrame,
        now: float,
        motion_dx_ratio: float,
        target_image_rate_dps: Optional[float] = None,
        max_correction_rpm: Optional[float] = None,
    ) -> ControlDecision:
        """Overlay camera/encoder yaw control on an active reverse command."""
        # Near-distance in-place rotation is already computed by the parked
        # PID before this helper is reached. Do not erase that result merely
        # because this reverse overlay only applies to backward actions.
        if len(decision.actions) != 1 or decision.actions[0].kind != "backward":
            return decision
        if not self.cfg.visible_steering_pid_enable or frame.width <= 0:
            self.last_steering_pid_result = None
            return decision

        action = decision.actions[0]
        base_rpm = max(
            1,
            int(round(int(self.cfg.forward_max_rpm) * int(action.speed_percent) / 100.0)),
        )
        cx, _cy = target.center
        current_x_ratio = float(cx) / float(max(1, frame.width))
        center_left = max(0.0, min(1.0, float(self.cfg.center_left_ratio)))
        center_right = max(center_left, min(1.0, float(self.cfg.center_right_ratio)))
        if center_left <= current_x_ratio <= center_right:
            self._visual_steering_pid.reset()
            self.last_steering_pid_result = None
            logger.info(
                "reverse_visual_pid_center_hold current_x=%.3f center=[%.3f,%.3f] correction=0rpm",
                current_x_ratio,
                center_left,
                center_right,
            )
            return replace(
                decision,
                actions=[
                    ControlAction.backward(
                        action.speed_percent,
                        action.reason,
                        correction_rpm=0,
                    )
                ],
            )

        # Reverse reacts opposite to forward steering and has much less room for
        # inertia. Do not project camera motion ahead while backing up.
        motion_lead_ratio = 0.0
        pid_x_ratio = current_x_ratio
        reverse_center_limit = max(
            0.0,
            min(6.0, float(self.cfg.visible_steering_pid_dynamic_small_max_correction_rpm)),
        )
        reverse_edge_limit = max(
            reverse_center_limit,
            min(10.0, float(self.cfg.visible_steering_pid_max_correction_rpm)),
        )
        center_error_ratio = abs(current_x_ratio - 0.5)
        center_half_width = max(
            0.01,
            min(0.45, 0.5 * abs(center_right - center_left)),
        )
        edge_blend = max(
            0.0,
            min(
                1.0,
                (center_error_ratio - center_half_width)
                / max(0.01, 0.35 - center_half_width),
            ),
        )
        reverse_correction_limit = reverse_center_limit + edge_blend * (
            reverse_edge_limit - reverse_center_limit
        )
        if max_correction_rpm is not None:
            reverse_correction_limit = min(
                reverse_correction_limit,
                max(0.0, float(max_correction_rpm)),
            )
        result = self._visual_steering_pid.update(
            pid_x_ratio,
            base_rpm,
            frame.steering_feedback,
            now=now,
            target_image_rate_dps=target_image_rate_dps,
            max_correction_override_rpm=reverse_correction_limit,
            visual_age_sec=self._visual_age_sec(frame, now),
        )
        self.last_steering_pid_result = result
        correction = int(result.correction_rpm)
        logger.info(
            "reverse_visual_pid current_x=%.3f dx=%.3f lead=%+.3f pid_x=%.3f "
            "base=%drpm correction=%+drpm limit=%.1frpm feedback=%s",
            current_x_ratio,
            float(motion_dx_ratio),
            motion_lead_ratio,
            pid_x_ratio,
            base_rpm,
            correction,
            reverse_correction_limit,
            "encoder" if result.feedback_used else "camera",
        )
        return replace(
            decision,
            actions=[
                ControlAction.backward(
                    action.speed_percent,
                    action.reason,
                    correction_rpm=correction,
                )
            ],
        )

    def _pid_action_for_parked_target(
        self,
        target: PersonTarget,
        frame: SensorFrame,
        now: float,
        edge: str,
        motion_dx_ratio: float = 0.0,
        projected_x_ratio: Optional[float] = None,
        target_image_rate_dps: Optional[float] = None,
        max_correction_rpm: Optional[float] = None,
        near_distance_mode: bool = False,
    ) -> Optional[ControlAction]:
        """Use the camera/encoder loop for low-speed in-place recentering."""
        initial_fallback = self._rotate_action_for_edge(edge)
        if not self.cfg.visible_steering_pid_enable or frame.width <= 0:
            self.last_steering_pid_result = None
            if initial_fallback is None:
                return None
            return self._retag_action(initial_fallback, f"person_parked_recenter_{edge}")

        min_rpm = max(1, int(self.cfg.parked_recenter_min_rpm))
        max_rpm = max(min_rpm, int(self.cfg.parked_recenter_max_rpm))
        cx, _cy = target.center
        x_ratio = float(cx) / float(max(1, frame.width))
        pid_x_ratio = x_ratio
        pid_edge = edge
        motion_led = False
        if pid_edge not in ("left", "right") and projected_x_ratio is not None:
            pid_edge = self._visible_motion_edge_type(
                motion_dx_ratio,
                x_ratio,
                float(projected_x_ratio),
                frame.width,
            )
            motion_led = pid_edge in ("left", "right")
        if (
            pid_edge not in ("left", "right")
            and target_image_rate_dps is not None
            and math.isfinite(float(target_image_rate_dps))
            and abs(float(target_image_rate_dps)) >= 2.0
            and abs(x_ratio - 0.5) > 1e-6
            and (x_ratio - 0.5) * float(target_image_rate_dps) > 0.0
        ):
            # A target that has just crossed the aim line but remains inside
            # the coarse center band must not wait until x reaches 0.45/0.55.
            # Feed it into the PID now; its deadband-motion branch will issue
            # the bounded 1-2 RPM lead requested by the image velocity.
            pid_edge = "right" if x_ratio > 0.5 else "left"
            motion_led = True
        if pid_edge not in ("left", "right"):
            if target_image_rate_dps is not None and abs(target_image_rate_dps) >= 5.0:
                logger.info(
                    "parked_pid_center_hold current_x=%.3f dx=%+.3f "
                    "target_image_rate=%+.2fdps center=[%.3f,%.3f] "
                    "reason=raw_target_inside_center",
                    x_ratio,
                    float(motion_dx_ratio),
                    float(target_image_rate_dps),
                    float(self.cfg.center_left_ratio),
                    float(self.cfg.center_right_ratio),
                )
            self._parked_recenter_pid.reset()
            self.last_steering_pid_result = None
            return None
        if motion_led:
            logger.info(
                "parked_pid_motion_entry current_x=%.3f projected_x=%.3f "
                "dx=%+.3f target_image_rate=%s edge=%s",
                x_ratio,
                x_ratio if projected_x_ratio is None else float(projected_x_ratio),
                float(motion_dx_ratio),
                "none"
                if target_image_rate_dps is None
                else "%+.2fdps" % float(target_image_rate_dps),
                pid_edge,
            )
        fallback = self._rotate_action_for_edge(pid_edge)
        if fallback is None:
            self.last_steering_pid_result = None
            return None

        center_error_ratio = abs(pid_x_ratio - 0.5)
        if self.cfg.center_deadzone_ratio is not None:
            center_half_width = max(0.01, min(0.45, float(self.cfg.center_deadzone_ratio)))
        else:
            center_half_width = max(
                0.01,
                min(0.45, 0.5 * abs(float(self.cfg.center_right_ratio) - float(self.cfg.center_left_ratio))),
            )
        # 刚离开中心区时只允许 2-3 RPM；横向偏差继续增大后才逐步放宽。
        # 这样每个视觉帧仍能快速更新方向，但不会一出 0.45~0.55 就打满原地转速。
        full_strength_error_ratio = max(center_half_width + 0.01, 0.25)
        strength = max(
            0.0,
            min(
                1.0,
                (center_error_ratio - center_half_width)
                / max(0.01, full_strength_error_ratio - center_half_width),
            ),
        )
        dynamic_max_rpm = max(
            min_rpm,
            min(max_rpm, int(round(min_rpm + (max_rpm - min_rpm) * strength))),
        )
        near_max_rpm = max(min_rpm, int(self.cfg.near_distance_rotation_only_max_rpm))
        dynamic_max_rpm = min(dynamic_max_rpm, near_max_rpm)
        if max_correction_rpm is not None:
            dynamic_max_rpm = min(
                dynamic_max_rpm,
                max(0, int(round(float(max_correction_rpm)))),
            )
        # VisualSteeringPid 把 base_rpm-3 作为轮差上限。这里不产生前进速度，
        # 只是借用同一个摄像头角度外环和编码器角速度内环计算原地转速。
        result = self._update_lateral_pid(
            self._parked_recenter_pid,
            x_ratio=pid_x_ratio,
            base_rpm=max_rpm + 3,
            feedback=frame.steering_feedback,
            now=now,
            target_image_rate_dps=target_image_rate_dps,
            max_correction_rpm=float(dynamic_max_rpm),
            visual_age_sec=self._visual_age_sec(frame, now),
            near_distance_mode=near_distance_mode,
        )
        if target_image_rate_dps is not None and abs(target_image_rate_dps) >= 1.0:
            logger.info(
                "parked_pid_target_rate current_x=%.3f dx=%+.3f image_rate=%+.2fdps "
                "bearing_rate=%+.2fdps feedforward=%+.2fdps "
                "speed_match=%s/%.2fdps edge=%s pid_edge=%s",
                x_ratio,
                float(motion_dx_ratio),
                float(result.target_image_rate_dps),
                float(result.target_bearing_rate_dps),
                float(result.target_rate_feedforward_dps),
                result.target_speed_match_limited,
                float(result.target_speed_match_limit_dps),
                edge,
                pid_edge,
            )
        correction = int(result.correction_rpm)
        raw_correction = correction

        guard_reason = self._pid_direction_guard_reason(
            x_ratio,
            motion_dx_ratio,
            correction,
        )
        if guard_reason is not None:
            result = replace(result, correction_rpm=0)
            self.last_steering_pid_result = result
            logger.info(
                "parked_pid_direction_guard current_x=%.3f pid_x=%.3f dx=%+.3f "
                "raw=%+drpm output=0rpm reason=%s center=[%.3f,%.3f] "
                "measured_yaw=%+.2fdps desired_yaw=%+.2fdps",
                x_ratio,
                pid_x_ratio,
                float(motion_dx_ratio),
                raw_correction,
                guard_reason,
                float(self.cfg.center_left_ratio),
                float(self.cfg.center_right_ratio),
                float(result.measured_yaw_rate_dps),
                float(result.desired_yaw_rate_dps),
            )
            return ControlAction.stop(
                "person_parked_direction_guard_hold",
                brake_hold=False,
            )

        result = self._parked_startup_floor(result, x_ratio)
        correction = int(result.correction_rpm)
        if correction != raw_correction:
            logger.info(
                "parked_pid_output_floor current_x=%.3f pid_x=%.3f raw=%+drpm "
                "output=%+drpm dynamic_limit=%drpm measured_yaw=%+.2fdps",
                x_ratio,
                pid_x_ratio,
                raw_correction,
                correction,
                dynamic_max_rpm,
                float(result.measured_yaw_rate_dps),
            )
        self.last_steering_pid_result = result
        if correction == 0:
            return None

        # PID 的正号表示车身需要产生右正角速度。实车编码器回放确认：
        # rotate_right 的物理角速度为正，rotate_left 为负；不能复用
        # “人物从哪一侧离开”到搜索动作的反向映射，否则 PID 会越修越偏。
        direction = "right" if correction > 0 else "left"
        action = (
            ControlAction.rotate_right(f"person_parked_recenter_{direction}")
            if correction > 0
            else ControlAction.rotate_left(f"person_parked_recenter_{direction}")
        )
        return action

    def _action_block_reason(self, action: ControlAction, frame: SensorFrame) -> Optional[str]:
        if frame.obstacles.front:
            return "front_ir"
        if frame.obstacles.left:
            return "left_ir"
        if frame.obstacles.right:
            return "right_ir"
        if action.kind in ("rotate_left", "rotate_right"):
            if self.cfg.distance_parking_enable and self.cfg.search_rotate_distance_block_enable:
                distance_close = frame.distance_m is not None and frame.distance_m < self.cfg.brake_distance_m
                distance_latched = bool(getattr(frame.distance_state, "brake_latched", False))
                used_distance = getattr(frame.distance_state, "used_distance_m", None)
                used_close = used_distance is not None and used_distance < self.cfg.brake_distance_m
                if distance_close or distance_latched or used_close:
                    return "distance_too_close"
                # 跟随停车距离只禁止继续前进，不禁止原地改变朝向。否则人物在
                # 1.2m 附近横向离开画面后，停车锁存会让小车永远无法重新居中。
                # < brake_distance 的极近距离与三路红外仍在上面无条件拦截。
        return None

    def _is_action_blocked(self, action: ControlAction, frame: SensorFrame) -> bool:
        return self._action_block_reason(action, frame) is not None

    def _visible_turn_action(
        self,
        edge: str,
        rotate_edge: str,
        cx: float,
        width: int,
        frame: SensorFrame,
        motion_dx_ratio: float = 0.0,
        projected_x_ratio: Optional[float] = None,
        motion_led: bool = False,
    ) -> Optional[ControlAction]:
        # 目标仍在摄像头画面内时始终使用两轮差速修正。即使人物进入画面
        # 最外侧，也不能提前切换成大角度原地旋转；只有 target=None 且
        # 丢失确认完成后，搜索阶段才允许 _lost_search_rotate_action。
        return self._turn_action_for_visible_target(
            edge,
            cx,
            width,
            frame,
            motion_dx_ratio=motion_dx_ratio,
            projected_x_ratio=projected_x_ratio,
            motion_led=motion_led,
        )

    def _select_person(self, frame: SensorFrame) -> Optional[PersonTarget]:
        if not frame.persons:
            return None
        if self.active_target_id == GEOMETRY_FALLBACK_TARGET_ID:
            geometry_matches = [
                p for p in frame.persons
                if int(p.track_id) == GEOMETRY_FALLBACK_TARGET_ID
            ]
            if geometry_matches:
                return max(geometry_matches, key=lambda p: p.area)

            # 几何兜底期间如果视觉层已经给出正式（非 -2）稳定 ID，
            # 允许它接管；否则仍按“未确认目标”处理，不能把候选漏帧当成丢失。
            reid_matches = [
                p for p in frame.persons
                if int(p.track_id) >= 0
            ]
            if reid_matches:
                return max(reid_matches, key=lambda p: p.area)
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

    def _fallback_search_action(self, frame: SensorFrame, reason_prefix: str) -> Optional[ControlAction]:
        direction = self.search_direction
        if direction not in ("left", "right"):
            return None
        candidate = self._lost_search_rotate_action(
            direction,
            f"{reason_prefix}_{direction}",
        )
        if candidate is None:
            return None
        return None if self._action_block_reason(candidate, frame) is not None else candidate

    def _ensure_search_state(self, frame: SensorFrame) -> None:
        if self.search_state in ("searching", "timed_out"):
            return
        if self.cfg.direction_history_enable and (
            self._lost_exit_direction not in ("left", "right")
            or self._lost_hint_source == "historical_direction_evidence"
        ):
            # Revalidate fallback evidence at the actual search transition.
            # Do not overwrite a confirmed candidate's authorized direction or
            # change an already-running search's direction/rotation coverage.
            self._capture_lost_exit_direction(frame, search_entry=True)
        if self._lost_exit_direction in ("left", "right"):
            self.search_direction = self._lost_exit_direction
        elif self.cfg.direction_history_enable:
            self.search_direction = None
            self.search_state = "direction_unresolved"
            logger.warning(
                "single_direction_search_unresolved hint_source=%s; hold zero yaw",
                str(self._lost_hint_source),
            )
            return
        elif self.last_person_center_x is not None and frame.width > 0:
            self.search_direction = "left" if self.last_person_center_x < frame.width / 2.0 else "right"
        else:
            self.search_direction = "right"
        self.search_state = "searching"
        if not self._search_rotation_feedback_seen and self._search_rotation_started_at is None:
            self._begin_search_rotation_measurement(frame)
        logger.info(
            "single_direction_search_start direction=%s hint_conf=%.2f "
            "hint_source=%s completion=encoder_heading_coverage_%.1fdeg encoder=%s",
            self.search_direction,
            float(self._lost_hint_confidence),
            str(self._lost_hint_source),
            max(90.0, float(self.cfg.search_revolution_deg)),
            bool(self._search_rotation_feedback_seen),
        )

    def _near_target_lost_rotate_decision(
        self,
        frame_index: int,
        frame: SensorFrame,
        now: float,
    ) -> Optional[ControlDecision]:
        """Confirm a parked-target loss before using timeline search direction."""
        if not self._target_stop_latched:
            return None

        first_lost_frame = self._lost_started_at is None
        if first_lost_frame:
            self._lost_started_at = float(now)
            self._capture_lost_exit_direction(frame)
            self._begin_search_rotation_measurement(frame)

        current_candidate = self._current_lateral_candidate(frame)
        self.lost_confirm_frames += 1
        if (
            current_candidate is None
            and self.lost_confirm_frames < max(1, int(self.cfg.lost_confirm_frames))
        ):
            return self._stale_direction_zero_decision(
                "near_target_lost_direction_confirm"
            )
        if current_candidate is not None:
            self.lost_confirm_frames = max(
                self.lost_confirm_frames,
                max(1, int(self.cfg.lost_confirm_frames)),
            )
        self._ensure_search_state(frame)
        if self.search_direction not in ("left", "right"):
            return self._stale_direction_zero_decision(
                "near_target_lost_direction_unresolved"
            )
        self._update_search_rotation_progress(frame)
        revolution_decision = self._search_revolution_complete_decision(now)
        if revolution_decision is not None:
            return revolution_decision
        timeout_decision = self._search_timeout_decision(now, frame.steering_feedback)
        if timeout_decision is not None:
            return timeout_decision
        direction = self.search_direction
        if direction not in ("left", "right"):
            return self._stale_direction_zero_decision(
                "near_target_lost_direction_unresolved"
            )

        if (
            current_candidate is not None
            and current_candidate.aimline_gap_ratio
            <= max(
                0.0,
                float(self.cfg.search_candidate_approach_margin_ratio),
            )
        ):
            reason = f"search_candidate_approach_{direction}"
        elif current_candidate is not None:
            reason = f"current_candidate_near_target_{direction}"
        else:
            reason = f"lost_wait_near_target_{direction}"
        action = self._lost_search_rotate_action(direction, reason)
        if action is None:
            return None
        block_reason = self._action_block_reason(action, frame)
        if block_reason is not None:
            return ControlDecision(
                explicit_stop_requested=True,
                clear_action_queue=True,
                stop_action_execution=True,
                reason=block_reason,
            )

        self.search_state = "searching"
        self._reset_visible_steer_memory()
        self._visual_steering_pid.reset()
        self._parked_recenter_pid.reset()
        self.last_steering_pid_result = None
        self._mark_search_rotation_started(now)
        self.last_action_frame = int(frame_index)
        return ControlDecision(
            actions=[action],
            clear_action_queue=first_lost_frame,
            reason=reason,
            evidence_capture_frame_id=(
                None
                if current_candidate is None
                else int(current_candidate.evidence.capture_frame_id)
            ),
        )

    def decide(
        self,
        frame_index: int,
        frame: SensorFrame,
        *,
        target_steerable: bool = True,
        target_steering_limit_rpm: Optional[float] = None,
        record_target_motion: bool = True,
        longitudinal_only: bool = False,
        rotation_only: bool = False,
        low_quality_visible: bool = False,
    ) -> ControlDecision:
        cfg = self.cfg
        now = time.monotonic()

        if frame.hazard.active:
            self._reset_longitudinal_motion()
            self._reset_visible_steer_memory()
            self._visual_steering_pid.reset()
            self._parked_recenter_pid.reset()
            self._reset_distance_pid()
            self.last_steering_pid_result = None
            return ControlDecision(
                explicit_stop_requested=True,
                clear_action_queue=True,
                stop_action_execution=True,
                reason=frame.hazard.reason or "hazard",
            )

        # IR is a direction-independent emergency gate. Handle it before target
        # selection/search so no current or newly generated motion can survive.
        ir_reason = self._action_block_reason(ControlAction.idle("ir_gate"), frame)
        if ir_reason is not None:
            self._reset_longitudinal_motion()
            self._visual_steering_pid.reset()
            self._parked_recenter_pid.reset()
            self._reset_distance_pid()
            self.last_steering_pid_result = None
            return ControlDecision(
                explicit_stop_requested=True,
                clear_action_queue=True,
                stop_action_execution=True,
                reason=ir_reason,
            )

        # A mapped crop can keep supplying bounded yaw geometry without being
        # trusted for identity updates, Depth, or longitudinal motion. Fragments
        # and crops inside the center corridor still use the stopped hold path.
        if (
            low_quality_visible
            and self._has_seen_person
            and self.active_target_id is not None
        ):
            self._reset_longitudinal_motion()
            search_was_active = bool(
                self.search_state in (
                    "searching",
                    "direction_unresolved",
                )
                or self._stale_direction_recovery_active
            )
            # Capture-time geometry remains useful while searching. It does
            # not grant identity, Depth, or forward-control permissions, but
            # it must pause the blind high-RPM scan and steer through the
            # bounded visible-target yaw path.
            history_target = self._select_person(frame)
            limited_target = None
            if (
                history_target is not None
                and target_steering_limit_rpm is not None
                and float(target_steering_limit_rpm) > 0.0
            ):
                limited_target = history_target
            self.last_selected_target = limited_target
            # A cropped/edge-touching box is unsuitable for identity, Depth,
            # and forward control, but its capture-time geometry is still the
            # strongest evidence for the direction in which the target left.
            if history_target is not None:
                self._record_target_direction_evidence(
                    frame,
                    history_target,
                    reliable=False,
                )
            motion_dx_ratio = 0.0
            projected_x_ratio = None
            target_image_rate_dps = None
            # A mapped, non-occluding low-quality box is still usable for
            # short-horizon image motion. Keep identity/depth/forward gates
            # closed, but do not throw away the only timely lateral evidence.
            if history_target is not None and record_target_motion:
                (
                    motion_dx_ratio,
                    projected_x_ratio,
                    target_image_rate_dps,
                    motion_dt_sec,
                ) = self._record_visible_motion(
                    frame_index,
                    history_target,
                    frame.width,
                    now,
                )
                if motion_dt_sec <= 0.0:
                    target_image_rate_dps = None
            if search_was_active:
                # A fragment seen during a scan is not reacquisition. Preserve
                # the direction, loss timer, and accumulated revolution so a
                # single bad box cannot restart the 360-degree search.
                self.lost_confirm_frames = max(
                    int(self.lost_confirm_frames),
                    max(1, int(self.cfg.lost_confirm_frames)),
                )
            else:
                self.lost_confirm_frames = 0
                self._lost_started_at = None
                self.search_state = "none"
                self.search_direction = None
                self._reset_search_timeout()
            self._reset_visible_steer_memory()
            self._visual_steering_pid.reset()
            self._reset_distance_pid()
            self.last_steering_pid_result = None
            if limited_target is not None:
                cx, _cy = limited_target.center
                edge = self._edge_type(cx, frame.width)
                action = self._pid_action_for_parked_target(
                    limited_target,
                    frame,
                    now,
                    edge,
                    motion_dx_ratio=motion_dx_ratio,
                    projected_x_ratio=projected_x_ratio,
                    target_image_rate_dps=target_image_rate_dps,
                    max_correction_rpm=float(target_steering_limit_rpm),
                )
                if action is not None and action.kind in ("rotate_left", "rotate_right"):
                    interrupt_existing = self.last_dispatched_kind != action.kind
                    continuous_rotate_switch = bool(
                        interrupt_existing
                        and self.last_dispatched_kind in ("rotate_left", "rotate_right")
                    )
                    self.last_action_frame = int(frame_index)
                    return ControlDecision(
                        actions=[self._retag_action(action, "target_visible_low_quality_yaw")],
                        current_forward_percent=0,
                        clear_action_queue=interrupt_existing,
                        stop_action_execution=interrupt_existing and not continuous_rotate_switch,
                        reason="target_visible_low_quality_yaw",
                    )

            self._parked_recenter_pid.reset()
            hold_reason = (
                "target_visible_low_quality_search_hold"
                if search_was_active
                else "target_visible_low_quality_hold"
            )
            if self._stale_direction_recovery_active:
                if self._stale_direction_recovery_stage == "candidate_centering":
                    candidate_decision = self._stale_direction_recovery_decision(
                        frame_index,
                        frame,
                        now,
                    )
                    if candidate_decision is not None:
                        return candidate_decision
                return self._stale_direction_zero_decision(
                    "stale_probe_low_quality_observe"
                )
            return ControlDecision(
                actions=[ControlAction.stop(hold_reason, brake_hold=False)],
                person_detected_flag=True,
                soft_stop_requested=True,
                reason=hold_reason,
            )

        target = self._select_person(frame)
        self.last_selected_target = target
        self._observe_longitudinal_motion(
            frame, target, allowed=bool(target_steerable and not low_quality_visible and not rotation_only)
        )
        current_lateral_candidate = None
        if target is None and not longitudinal_only and self._has_seen_person:
            # Fresh detector geometry has priority over historical/frozen search
            # direction. It only owns lateral motion; target identity and
            # longitudinal control remain unavailable until Tracker/ReID agree.
            current_lateral_candidate = self._apply_current_lateral_candidate(frame)
        if not longitudinal_only:
            mapped_low_quality_target = bool(
                target is not None
                and low_quality_visible
                and self.active_target_id is not None
                and int(target.track_id) == int(self.active_target_id)
            )
            self._record_target_direction_evidence(
                frame,
                target,
                # A mapped active UID with a cropped/large box is not safe for
                # Depth or full steering, but its current side is still valid
                # direction evidence when the target disappears immediately
                # afterwards. Unmapped candidates remain unknown.
                reliable=bool(
                    (target_steerable and not low_quality_visible)
                    or mapped_low_quality_target
                ),
            )
        if (
            current_lateral_candidate is not None
            and current_lateral_candidate.position == "center"
        ):
            candidate = current_lateral_candidate.evidence
            self._reset_distance_pid()
            self._visual_steering_pid.reset()
            self._parked_recenter_pid.reset()
            self.last_steering_pid_result = None
            logger.info(
                "search_candidate_aimline_brake capture_frame_id=%d source=%s "
                "score=%.3f bbox=%s center=%.3f action=active_stop "
                "identity_claim=False",
                int(candidate.capture_frame_id),
                str(candidate.source),
                float(candidate.score),
                tuple(round(float(value), 1) for value in candidate.bbox),
                float(current_lateral_candidate.center_ratio),
            )
            return ControlDecision(
                explicit_stop_requested=True,
                clear_action_queue=True,
                stop_action_execution=True,
                reason="search_candidate_aimline_brake",
                evidence_capture_frame_id=int(candidate.capture_frame_id),
            )
        initial_target_confirmed_now = False

        if target is not None and self._stale_direction_recovery_active:
            self._reset_stale_direction_recovery("target_reacquired")
            self.search_direction = None
            self._lost_started_at = None
            self._lost_exit_direction = None
            self.lost_confirm_frames = 0
            self._reset_search_timeout()

        if longitudinal_only:
            # The 30Hz Depth supervisor must not advance visual motion history,
            # search state, steering PID, or visual action cooldown.
            last_action_frame = self.last_action_frame
            try:
                return self._longitudinal_only_decision(
                    frame_index,
                    frame,
                    target,
                    target_steerable=bool(target_steerable),
                )
            finally:
                self.last_action_frame = last_action_frame

        limited_steering = bool(
            target_steering_limit_rpm is not None
            and float(target_steering_limit_rpm) > 0.0
        )
        steering_allowed = bool(target_steerable or limited_steering)
        recovered_from_search_unsteerable = bool(
            target is not None
            and not target_steerable
            and self.search_state in (
                "searching",
                "timed_out",
                "direction_unresolved",
            )
        )
        reverse_decision = None
        if not rotation_only:
            reverse_decision = self._reverse_control_decision(
                frame_index,
                frame,
                target,
                force_immediate_close=bool(target is not None and not target_steerable),
                target_steering_limit_rpm=target_steering_limit_rpm,
                target_steerable=bool(target_steerable),
            )
        if reverse_decision is not None:
            near_rotation_pid_active = bool(
                reverse_decision.reason == "near_distance_rotation_only"
                and self.last_steering_pid_result is not None
            )
            reverse_pid_active = False
            if target is not None:
                if steering_allowed:
                    target_cx, _target_cy = self._center(target)
                    self.last_person_center_x = target_cx
                    target_image_rate_dps = None
                    if record_target_motion:
                        (
                            motion_dx_ratio,
                            _projected_x,
                            target_image_rate_dps,
                            motion_dt_sec,
                        ) = self._record_visible_motion(
                            frame_index,
                            target,
                            frame.width,
                            now,
                        )
                        if motion_dt_sec <= 0.0:
                            target_image_rate_dps = None
                    else:
                        motion_dx_ratio = 0.0
                    reverse_decision = self._reverse_decision_with_visual_steering(
                        reverse_decision,
                        target,
                        frame,
                        now,
                        motion_dx_ratio,
                        target_image_rate_dps=target_image_rate_dps,
                        max_correction_rpm=target_steering_limit_rpm,
                    )
                    reverse_pid_active = self.last_steering_pid_result is not None
                self._remember_target_distance(frame, now)
                self.active_target_id = int(target.track_id)
                self._has_seen_person = True
                self.search_state = "none"
                self.search_direction = None
                self.lost_confirm_frames = 0
                self._lost_started_at = None
                self._lost_exit_direction = None
                self._reset_search_timeout()
            self._reset_visible_steer_memory()
            if not reverse_pid_active:
                self._visual_steering_pid.reset()
            # The near-distance rotation decision is produced by the parked
            # camera/encoder PID. Preserve that result for the motor runtime;
            # only actual reverse decisions should reset the parked PID state.
            if not near_rotation_pid_active:
                self._parked_recenter_pid.reset()
            # 纵向 PID 必须跨 Depth 帧保留 D 项和积分；此前这里每帧清零，
            # 导致倒车永远只有 20~35 RPM，完全看不到目标接近速度。
            if not reverse_pid_active and not near_rotation_pid_active:
                self.last_steering_pid_result = None
            if recovered_from_search_unsteerable:
                return ControlDecision(
                    actions=list(reverse_decision.actions),
                    explicit_stop_requested=reverse_decision.explicit_stop_requested,
                    person_detected_flag=True,
                    waiting_lost_confirm=reverse_decision.waiting_lost_confirm,
                    is_forwarding=reverse_decision.is_forwarding,
                    current_forward_percent=reverse_decision.current_forward_percent,
                    clear_action_queue=True,
                    stop_action_execution=True,
                    shutdown_requested=reverse_decision.shutdown_requested,
                    reason=reverse_decision.reason,
                )
            return reverse_decision

        if target is not None and not steering_allowed:
            # YOLO仍清晰看到唯一人物，但过大/裁切框的中心点不能用于转向。
            # 保留身份并持续清除丢失状态；Depth未要求倒车时只停车，绝不
            # 让这个框进入横向PID、预测离开方向或丢失搜索。
            self.active_target_id = int(target.track_id)
            self._has_seen_person = True
            self._remember_target_distance(frame, now)
            self.search_state = "none"
            self.search_direction = None
            self.lost_confirm_frames = 0
            self._lost_started_at = None
            self._lost_exit_direction = None
            self._reset_search_timeout()
            self._reset_visible_steer_memory()
            self._visual_steering_pid.reset()
            self._parked_recenter_pid.reset()
            self.last_steering_pid_result = None
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=recovered_from_search_unsteerable,
                clear_action_queue=True,
                stop_action_execution=True,
                reason="target_visible_unsteerable_hold",
            )

        if target is None and self._stale_direction_recovery_active:
            stale_recovery = self._stale_direction_recovery_decision(
                frame_index,
                frame,
                now,
            )
            if stale_recovery is not None:
                return stale_recovery

        # 停车距离内丢失目标时，前进锁存继续有效，但立即允许按最后视觉方向
        # 原地找人。极近距离刹车和红外仍由 locked_decision/action gate 拦截。
        if target is None and self._target_stop_latched and not cfg.exit_on_target_loss:
            locked_decision = self._target_distance_lock_decision(
                frame,
                now,
                off_center=False,
                person_detected_flag=True,
                clear_action_queue=True,
                stop_action_execution=True,
            )
            if locked_decision is not None:
                if locked_decision.reason in ("distance_too_close", "person_too_close_no_rotate"):
                    return locked_decision
                near_lost_rotate = self._near_target_lost_rotate_decision(
                    frame_index,
                    frame,
                    now,
                )
                if near_lost_rotate is not None:
                    return near_lost_rotate
                return locked_decision

        if target is None:
            if self.active_target_id == GEOMETRY_FALLBACK_TARGET_ID:
                # 启动阶段只有几何兜底目标时，任何短暂漏帧都只能停车等待。
                # 在正式 ReID 锁定前禁止 lost_confirm/search，避免开机误旋转。
                self.lost_confirm_frames = 0
                self._lost_started_at = None
                self._lost_exit_direction = None
                self.search_state = "none"
                self.search_direction = None
                self._reset_search_timeout()
                return ControlDecision(
                    explicit_stop_requested=True,
                    clear_action_queue=True,
                    stop_action_execution=True,
                    reason="unconfirmed_target_wait",
                )
            if not self._has_seen_person:
                self._reset_initial_target_confirm()
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
                    self._capture_lost_exit_direction(frame)
                    self._begin_search_rotation_measurement(frame)

                # A detector dropout invalidates longitudinal target distance
                # immediately. Do not carry forward/steer wheel speed into the
                # confirmation window; only `_lost_confirm_wait_decision` may
                # retain bounded in-place yaw from the last reliable side.

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
                    if cfg.exit_on_target_loss:
                        # The target was previously locked and has now passed
                        # the loss-confirmation window. Stop and terminate the
                        # runtime; do not enter search rotation.
                        self.search_state = "timed_out"
                        self.search_direction = None
                        self._reset_search_timeout()
                        return ControlDecision(
                            explicit_stop_requested=True,
                            clear_action_queue=True,
                            stop_action_execution=True,
                            shutdown_requested=True,
                            reason="target_lost_exit",
                        )
                    self._ensure_search_state(frame)
                    if self.active_target_id is not None:
                        old_target_id = int(self.active_target_id)
                        if cfg.release_target_on_lost and current_lateral_candidate is None:
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
                        else:
                            logger.info(
                                "locked_search_enter frame=%d kept_target=%d lost_frames=%d lost_sec=%.2f threshold_frames=%d threshold_sec=%.2f",
                                int(frame_index),
                                old_target_id,
                                int(self.lost_confirm_frames),
                                float(lost_elapsed_sec),
                                int(cfg.lost_confirm_frames),
                                float(cfg.lost_confirm_sec),
                            )
                    # Search motion is generated below in this same control
                    # decision.  The motor layer performs a bounded transition
                    # only when the requested yaw direction actually reverses.

            if target is None:
                self._ensure_search_state(frame)

        if target is not None and not self._has_seen_person:
            self.lost_confirm_frames = 0
            self.search_state = "none"
            self.search_direction = None
            self._lost_started_at = None
            self._lost_exit_direction = None
            self.last_person_center_x = self._center(target)[0]
            if not self._confirm_initial_target(target, frame_index):
                # Initial identity confirmation must not create a visual blind
                # spot.  Keep the vehicle stationary longitudinally, but use
                # the same camera/encoder yaw PID as parked recentering so a
                # candidate at the edge is kept in view while ReID confirms it.
                self._visual_steering_pid.reset()
                self.last_steering_pid_result = None
                motion_dx_ratio = 0.0
                projected_x_ratio = None
                target_image_rate_dps = None
                if record_target_motion:
                    (
                        motion_dx_ratio,
                        projected_x_ratio,
                        target_image_rate_dps,
                        motion_dt_sec,
                    ) = self._record_visible_motion(
                        frame_index,
                        target,
                        frame.width,
                        now,
                    )
                    if motion_dt_sec <= 0.0:
                        target_image_rate_dps = None
                candidate_edge = self._edge_type(target.center[0], frame.width)
                candidate_action = self._pid_action_for_parked_target(
                    target,
                    frame,
                    now,
                    candidate_edge,
                    motion_dx_ratio=motion_dx_ratio,
                    projected_x_ratio=projected_x_ratio,
                    target_image_rate_dps=target_image_rate_dps,
                    max_correction_rpm=float(
                        self.cfg.near_distance_rotation_only_max_rpm
                    ),
                )
                if candidate_action is not None and candidate_action.kind in (
                    "rotate_left",
                    "rotate_right",
                ):
                    candidate_reason = "initial_candidate_centering_%s" % (
                        candidate_action.kind.removeprefix("rotate_")
                    )
                    candidate_action = self._retag_action(
                        candidate_action,
                        candidate_reason,
                    )
                    logger.info(
                        "initial candidate yaw preview: control_frame_id=%d "
                        "capture_frame_id=%d track_id=%d center_x=%.3f edge=%s "
                        "action=%s correction=%sRPM identity_confirm=%d/%d",
                        int(frame_index),
                        int(getattr(frame, "capture_frame_id", -1)),
                        int(target.track_id),
                        float(target.center[0]) / float(max(1, frame.width)),
                        candidate_edge,
                        candidate_action.kind,
                        "none"
                        if self.last_steering_pid_result is None
                        else int(self.last_steering_pid_result.correction_rpm),
                        int(self._initial_candidate_frames),
                        int(self.cfg.initial_target_confirm_frames),
                    )
                    return ControlDecision(
                        actions=[candidate_action],
                        current_forward_percent=0,
                        clear_action_queue=True,
                        waiting_lost_confirm=True,
                        reason=candidate_reason,
                    )

                # A centered candidate needs no yaw command. Preserve the
                # enrollment hold semantics for that case; a PID guard is
                # still represented as a soft zero-yaw update so residual
                # motion is damped without entering a new brake hold.
                if candidate_action is None:
                    return ControlDecision(
                        actions=[ControlAction.stop(
                            "initial_candidate_confirmation_hold",
                            brake_hold=False,
                        )],
                        waiting_lost_confirm=True,
                        soft_stop_requested=True,
                        reason="initial_candidate_confirmation_hold",
                    )
                hold_reason = "initial_candidate_yaw_guard_hold"
                return ControlDecision(
                    actions=[ControlAction.stop(hold_reason, brake_hold=False)],
                    waiting_lost_confirm=True,
                    soft_stop_requested=True,
                    reason=hold_reason,
                )
            initial_target_confirmed_now = True

        if target is not None:
            target_cx, target_cy = self._center(target)
            target_off_center = not self._is_in_center_3x3(
                target_cx,
                target_cy,
                frame.width,
                frame.height,
            )
            distance_lock = None
            if not rotation_only:
                distance_lock = self._target_distance_lock_decision(
                    frame,
                    now,
                    off_center=target_off_center,
                    target=target,
                    person_detected_flag=False,
                    clear_action_queue=True,
                    stop_action_execution=True,
                )
            if distance_lock is not None:
                # 即使已经到达停车距离，也持续记录画面中心和移动趋势；目标下
                # 一帧消失时，原地搜索才能沿正确方向立即开始。
                self.last_person_center_x = target_cx
                motion_dx_ratio = 0.0
                target_image_rate_dps = None
                if record_target_motion:
                    (
                        motion_dx_ratio,
                        _projected_x_ratio,
                        target_image_rate_dps,
                        motion_dt_sec,
                    ) = self._record_visible_motion(
                        frame_index,
                        target,
                        frame.width,
                        now,
                    )
                    if motion_dt_sec <= 0.0:
                        target_image_rate_dps = None
                self._remember_target_distance(frame, now)
                self.active_target_id = int(target.track_id)
                self._has_seen_person = True
                recovered_from_search = self.search_state in (
                    "searching",
                    "timed_out",
                    "direction_unresolved",
                )
                self.search_state = "none"
                self.search_direction = None
                self.lost_confirm_frames = 0
                self._lost_started_at = None
                self._lost_exit_direction = None
                self._reset_search_timeout()
                # 停车居中使用独立 PID；清掉行驶微调状态，避免恢复前进时
                # 带入停车前积累的积分和误差历史。
                self._visual_steering_pid.reset()
                if (
                    target_off_center
                    and distance_lock.reason not in ("distance_too_close", "person_too_close_no_rotate")
                ):
                    edge = self._edge_type(target_cx, frame.width)
                    recenter_action = self._pid_action_for_parked_target(
                        target,
                        frame,
                        now,
                        edge,
                        motion_dx_ratio=motion_dx_ratio,
                        target_image_rate_dps=target_image_rate_dps,
                        max_correction_rpm=target_steering_limit_rpm,
                    )
                    if recenter_action is not None:
                        block_reason = self._action_block_reason(recenter_action, frame)
                        if block_reason is None:
                            reason = recenter_action.reason
                            self._reset_visible_steer_memory()
                            self.last_action_frame = int(frame_index)
                            return ControlDecision(
                                actions=[recenter_action],
                                person_detected_flag=recovered_from_search,
                                clear_action_queue=recovered_from_search,
                                reason=reason,
                            )
                self._parked_recenter_pid.reset()
                self.last_steering_pid_result = None
                return distance_lock

        if target is None:
            self._reset_distance_pid()
            self._visual_steering_pid.reset()
            self._parked_recenter_pid.reset()
            self.last_steering_pid_result = None
            self._update_search_rotation_progress(frame)
            if (
                self._search_observation_hold
                and self.search_state == "searching"
                and current_lateral_candidate is None
            ):
                return self._stale_direction_zero_decision(
                    "search_candidate_evidence_observe"
                )
            revolution_decision = self._search_revolution_complete_decision(now)
            if revolution_decision is not None:
                return revolution_decision
            timeout_decision = self._search_timeout_decision(now, frame.steering_feedback)
            if timeout_decision is not None:
                return timeout_decision
            if self.search_direction not in ("left", "right"):
                self.search_state = "direction_unresolved"
                return self._stale_direction_zero_decision(
                    "search_direction_unresolved_hold"
                )
            frames_since_last_action = frame_index - self.last_action_frame if self.last_action_frame >= 0 else 999
            if frames_since_last_action >= cfg.search_cooldown:
                self.search_state = "searching"
                if (
                    current_lateral_candidate is not None
                    and current_lateral_candidate.aimline_gap_ratio
                    <= max(
                        0.0,
                        float(cfg.search_candidate_approach_margin_ratio),
                    )
                ):
                    search_reason_prefix = "search_candidate_approach"
                elif current_lateral_candidate is not None:
                    search_reason_prefix = "search_current_candidate"
                else:
                    search_reason_prefix = "search"
                action = self._fallback_search_action(
                    frame,
                    search_reason_prefix,
                )
                if action is None:
                    return ControlDecision(explicit_stop_requested=True, reason="search_both_sides_blocked")
                if action.kind in ("rotate_left", "rotate_right"):
                    self._mark_search_rotation_started(now)
                self.last_action_frame = frame_index
                decision = ControlDecision(
                    actions=[action],
                    reason=action.reason,
                    is_forwarding=(action.kind == "forward"),
                    current_forward_percent=action.speed_percent if action.kind == "forward" else 0,
                    evidence_capture_frame_id=(
                        None
                        if current_lateral_candidate is None
                        else int(current_lateral_candidate.evidence.capture_frame_id)
                    ),
                )
                return decision

            if self.last_dispatched_kind in ("rotate_left", "rotate_right"):
                return ControlDecision(explicit_stop_requested=True, reason="search_cooldown_stop_rotate")
            return ControlDecision(reason="search_cooldown_keep_motion")

        was_waiting_lost_confirm = self._lost_started_at is not None or self.lost_confirm_frames > 0
        person_detected_flag = False
        clear_queue = False
        stop_execution = False
        if self.search_state in (
            "searching",
            "timed_out",
            "direction_unresolved",
        ):
            self.search_state = "none"
            self.search_direction = None
            person_detected_flag = True
            clear_queue = True
            stop_execution = True
        elif was_waiting_lost_confirm and self.last_dispatched_kind in ("rotate_left", "rotate_right"):
            # 人物在确认窗口内重新出现时，先打断可能仍在运行的搜索脉冲，再下发视觉动作。
            person_detected_flag = True
            clear_queue = True
            stop_execution = True
        self._reset_search_timeout()
        target_id_changed = (
            self.active_target_id is None
            or int(self.active_target_id) != int(target.track_id)
        )
        if target_id_changed:
            # ReID 交接后不能把旧目标的视觉误差、D 项和积分带给新目标，
            # 否则新目标出现的第一帧可能仍沿旧目标方向修正。
            # Initial two-frame enrollment is different: its first frame has
            # already used this same bbox chain for temporary centering, so
            # keep that encoder-loop state and avoid a duplicate startup kick.
            if not initial_target_confirmed_now:
                self._visual_steering_pid.reset()
                self._parked_recenter_pid.reset()
            self._reset_distance_pid()
            self._forward_active = False
            self.last_steering_pid_result = None
            self.active_target_id = int(target.track_id)

        self.lost_confirm_frames = 0
        self._lost_started_at = None
        self._lost_exit_direction = None
        self._has_seen_person = True
        cx, cy = self._center(target)
        target_image_rate_dps = None
        motion_sample_dt_sec = 0.0
        if record_target_motion:
            (
                motion_dx_ratio,
                projected_x_ratio,
                target_image_rate_dps,
                motion_sample_dt_sec,
            ) = self._record_visible_motion(
                frame_index,
                target,
                frame.width,
                now,
            )
            if motion_sample_dt_sec <= 0.0:
                target_image_rate_dps = None
        else:
            motion_dx_ratio = 0.0
            projected_x_ratio = float(cx) / float(max(1, frame.width))
        self.last_person_center_x = cx
        self._remember_target_distance(frame, now)

        position_edge = self._edge_type(cx, frame.width)
        current_x_ratio = float(cx) / float(max(1, frame.width))
        motion_edge = self._visible_motion_edge_type(
            motion_dx_ratio,
            current_x_ratio,
            projected_x_ratio,
            frame.width,
        )
        edge = position_edge if position_edge != "none" else motion_edge
        motion_led = position_edge == "none" and motion_edge != "none"
        in_center = self._is_in_center_3x3(cx, cy, frame.width, frame.height) and not motion_led
        if motion_led:
            logger.info(
                "visible_motion_lead frame=%d direction=%s x=%.3f dx=%+.3f "
                "dt=%.3fs image_rate=%+.2fdps projected_x=%.3f",
                int(frame_index),
                motion_edge,
                float(cx) / float(max(1, frame.width)),
                float(motion_dx_ratio),
                float(motion_sample_dt_sec),
                float(target_image_rate_dps or 0.0),
                float(projected_x_ratio),
            )
        frames_since_last_action = frame_index - self.last_action_frame if self.last_action_frame >= 0 else 999

        self._parked_recenter_pid.reset()
        pid_action = self._pid_action_for_visible_target(
            target,
            frame,
            now,
            motion_dx_ratio=motion_dx_ratio,
            target_image_rate_dps=target_image_rate_dps,
            max_correction_rpm=target_steering_limit_rpm,
        )
        if pid_action is not None:
            block_reason = self._action_block_reason(pid_action, frame)
            if block_reason is not None:
                self._visual_steering_pid.reset()
                self.last_steering_pid_result = None
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason=block_reason,
                )
            self._remember_visible_steer(pid_action, now)
            self.last_action_frame = frame_index
            return ControlDecision(
                actions=[pid_action],
                person_detected_flag=person_detected_flag,
                is_forwarding=True,
                current_forward_percent=pid_action.speed_percent,
                clear_action_queue=clear_queue,
                stop_action_execution=stop_execution,
                reason=pid_action.reason,
            )

        # 只有目标确实处于中心带、编码器也确认车身基本静止时，零输出才
        # 表示已经居中。PID 零交叉或高速制动过程不能覆盖成直行/STOP。
        pid_result = self.last_steering_pid_result
        pid_zero_is_settled = bool(
            cfg.visible_steering_pid_enable
            and pid_result is not None
            and in_center
            and (
                not pid_result.feedback_used
                or abs(float(pid_result.measured_yaw_rate_dps)) < 2.0
            )
        )
        if pid_zero_is_settled:
            in_center = True
            edge = "none"
            motion_led = False

        if in_center:
            self._reset_visible_steer_memory()

        # Safety gates.
        if frame.hazard.active:
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=person_detected_flag,
                clear_action_queue=clear_queue,
                stop_action_execution=stop_execution,
                reason=frame.hazard.reason or "hazard",
            )

        if not in_center:
            if cfg.distance_parking_enable and frame.distance_m is not None and frame.distance_m < cfg.brake_distance_m:
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="person_too_close_no_rotate",
                )
            if cfg.distance_parking_enable and frame.distance_m is not None and frame.distance_m <= cfg.target_distance_m:
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="target_distance_reached_off_center",
                )
            rotate_edge = self._visible_rotate_edge_type(cx, frame.width)
            action = self._visible_turn_action(
                edge,
                rotate_edge,
                cx,
                frame.width,
                frame,
                motion_dx_ratio=motion_dx_ratio,
                projected_x_ratio=projected_x_ratio,
                motion_led=motion_led,
            )
            if action is None:
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="turn_edge_unknown",
                )
            block_reason = self._action_block_reason(action, frame)
            if block_reason is not None:
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason=block_reason,
                )
            if frames_since_last_action < cfg.action_cooldown:
                held_action = self._last_visible_steer_action
                if held_action is not None and held_action.kind == action.kind:
                    held_action = self._retag_action(held_action, "visible_turn_hold")
                    return ControlDecision(
                        actions=[held_action],
                        person_detected_flag=person_detected_flag,
                        is_forwarding=False,
                        current_forward_percent=0,
                        clear_action_queue=clear_queue,
                        stop_action_execution=stop_execution,
                        reason="visible_turn_hold",
                    )
            self._remember_visible_steer(action, now)
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
        if frame.obstacles.front:
            return ControlDecision(
                explicit_stop_requested=True,
                person_detected_flag=person_detected_flag,
                clear_action_queue=clear_queue,
                stop_action_execution=stop_execution,
                reason="front_ir",
            )

        side_ir_active = bool(frame.obstacles.left or frame.obstacles.right)

        if self._distance_longitudinally_untrusted(frame):
            self._forward_active = False
            if self.distance_pi_enabled:
                self._pause_distance_pi(frame, now, "visual_distance_untrusted")
            else:
                self._reset_distance_pid()
            return ControlDecision(
                actions=[ControlAction.stop("distance_untrusted_hold")],
                person_detected_flag=person_detected_flag,
                explicit_stop_requested=True,
                current_forward_percent=0,
                clear_action_queue=True,
                stop_action_execution=True,
                reason="distance_untrusted_hold",
            )

        pid_distance_fallback = frame.distance_m is None
        if (
            cfg.visible_steering_pid_enable
            and self.last_steering_pid_result is not None
            and pid_distance_fallback
            and not self.distance_pi_enabled
            and not bool(getattr(frame.distance_state, "brake_latched", False))
        ):
            # PID 输出为零表示人物已经回到中心。真正没有任何可用距离时
            # 仍以 15 RPM 基准直行；毫米波短时 hold 则在下方继续走距离曲线。
            speed = self._forward_percent_for_rpm(
                cfg.visible_steering_pid_fallback_base_rpm,
                allow_below_min=True,
            )
            feedback_tag = "encoder" if self.last_steering_pid_result.feedback_used else "camera"
            reason = f"visual_pid_center_{feedback_tag}"
            if self._is_mmwave_hold(frame):
                reason += "_mmwave_hold"
            else:
                reason += "_distance_missing"
            self.last_action_frame = frame_index
            return ControlDecision(
                actions=[ControlAction.forward(speed, reason)],
                person_detected_flag=person_detected_flag,
                is_forwarding=True,
                current_forward_percent=speed,
                clear_action_queue=clear_queue,
                stop_action_execution=stop_execution,
                reason=reason,
            )

        if frame.distance_m is not None:
            if cfg.distance_parking_enable and frame.distance_m < cfg.brake_distance_m:
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="distance_too_close",
                )
            if frame.distance_m > cfg.target_distance_m or self.distance_pi_enabled:
                if side_ir_active:
                    speed = self._fallback_forward_percent()
                    reason = "side_ir_escape_forward"
                else:
                    speed = self._forward_percent_for_distance(frame.distance_m, now=now)
                    reason = "follow_distance_pid" if cfg.distance_pid_enable else "follow_distance"
                pid_result = self.last_steering_pid_result
                if cfg.visible_steering_pid_enable and pid_result is not None:
                    pid_speed_cap = self._forward_percent_for_rpm(
                        int(pid_result.base_rpm),
                        allow_below_min=True,
                    )
                    if pid_speed_cap < speed:
                        speed = pid_speed_cap
                        reason += "_pid_base_cap"
                speed = self._limit_mmwave_hold_forward_percent(frame, speed)
                if self._is_mmwave_hold(frame):
                    reason += "_mmwave_hold"
                if speed <= 0:
                    return ControlDecision(
                        actions=[ControlAction.forward(0, "target_distance_hold")],
                        person_detected_flag=person_detected_flag,
                        clear_action_queue=True,
                        stop_action_execution=stop_execution,
                        reason="target_distance_hold",
                    )
                self.last_action_frame = frame_index
                return ControlDecision(
                    actions=[ControlAction.forward(speed, reason)],
                    person_detected_flag=person_detected_flag,
                    is_forwarding=True,
                    current_forward_percent=speed,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason=reason,
                )
            if cfg.distance_parking_enable:
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="target_distance_reached",
                )
            # Distance remains a speed/reverse input; it cannot create a STOP.
            # Replace stale forward motion with a zero DRIVE frame.
            self._forward_active = False
            self._reset_distance_pid()
            return ControlDecision(
                actions=[ControlAction.forward(0, "target_distance_speed_zero")],
                person_detected_flag=person_detected_flag,
                is_forwarding=False,
                current_forward_percent=0,
                clear_action_queue=True,
                reason="target_distance_speed_zero",
            )

        if not frame.obstacles.front:
            camera_hold = self._distance_missing_camera_hold_action(frame, now)
            if camera_hold is not None:
                self.last_action_frame = frame_index
                return ControlDecision(
                    actions=[camera_hold],
                    person_detected_flag=person_detected_flag,
                    is_forwarding=True,
                    current_forward_percent=camera_hold.speed_percent,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason=camera_hold.reason,
                )
            if cfg.distance_parking_enable:
                return ControlDecision(
                    explicit_stop_requested=True,
                    person_detected_flag=person_detected_flag,
                    clear_action_queue=clear_queue,
                    stop_action_execution=stop_execution,
                    reason="distance_missing_stop",
                )
            # No distance sample is not a parking condition in IR-only mode.
            # Replace stale motion with a zero DRIVE until range recovers.
            return ControlDecision(
                actions=[ControlAction.forward(0, "distance_missing_camera_hold_unavailable")],
                person_detected_flag=person_detected_flag,
                is_forwarding=False,
                current_forward_percent=0,
                clear_action_queue=True,
                reason="distance_missing_camera_hold_unavailable",
            )

        return ControlDecision(
            explicit_stop_requested=True,
            person_detected_flag=person_detected_flag,
            clear_action_queue=clear_queue,
            stop_action_execution=stop_execution,
            reason="distance_missing_front_blocked",
        )
