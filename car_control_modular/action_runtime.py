from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, FrozenSet, Mapping, Optional

from .control_types import SteeringFeedback
from .action_queue_policy import should_drop_queued_action
from .mssd_motor import MssdMotorBackend
from .steering_pid import encoder_yaw_rate_right_dps
from .wheel_zero_cross import WheelZeroCrossGuard, wheel_feedback_valid
from .forward_loss_handoff import ForwardLossHandoff
from .longitudinal_execution import ForwardExecutionAnchor
from .steering_limits import effective_correction_limit, clamp_correction
from .turn_response_assist import TurnResponseAssist
from .turn_buildup import TurnBuildup
from .turn_response_trial import TurnResponseTrial, forward_brake_allowed
from .wheel_response import WheelDifferentialResponse
from .follow_wheel_clock import FollowWheelClock
from .follow_distance_hold import FollowDistanceHold
from .final_yaw_coalescing import contract_forward_yaw
from .near_yaw_parking import ParkSettlingEvidence
from .predictive_turn_brake import pulse_feedback_qualified, MAX_PULSE_SEC
from .action_command import ActionCommandSnapshot, SearchReacquireBrakeRequest
from .search_reacquire_braking import (
    limit_handoff_yaw, moving_handoff_yaw, handoff_straight_allowed,
)


@dataclass(frozen=True)
class ActionRuntimeSymbols:
    forward: int
    backward: int
    rotate_left: int
    rotate_right: int
    stop: int
    steer_left: int
    steer_right: int
    movement_actions: FrozenSet[int]
    forward_like_actions: FrozenSet[int]
    rotate_actions: FrozenSet[int]
    action_names: Mapping[int, str]
    safety_stop_reasons: FrozenSet[str]


@dataclass(frozen=True)
class ActionRuntimeConfig:
    enable_motor_rpm_feedback: bool
    motor_feedback_poll_interval_sec: float
    brake_hold_refresh_interval_sec: float
    use_percent_speed: bool
    min_forward_percent: int
    max_forward_percent: int
    steer_percent_limit: int
    mmwave_hold_forward_percent: int
    visible_steer_inner_ratio_percent: int
    visible_steer_outer_ratio_percent: int
    motor_forward_raw_target: int
    motor_forward_max_target_rpm: int
    motor_steer_raw_target: int
    motor_rotate_raw_target: int
    motor_forward_target_sign: int
    motor_left_sign: int
    motor_right_sign: int
    rotate_prep_coast_enable: bool
    rotate_prep_coast_steps: int
    rotate_prep_coast_total_sec: float
    rotate_pulse_brake_enable: bool
    rotate_pulse_stop_mode: str
    rotate_pulse_pause_sec: float
    rotate_pulse_observe_min_frames: int
    rotate_pulse_settle_enable: bool
    rotate_pulse_settle_quiet_sec: float
    rotate_pulse_settle_timeout_sec: float
    rotate_pulse_settle_feedback_stale_sec: float
    rotate_pulse_settle_max_wheel_rpm: int
    rotate_pulse_settle_max_yaw_rate_dps: float
    # Low-speed same-direction handoff between lost-target search pulses.
    # A value of zero preserves the hard-zero transition behavior.
    rotate_pulse_transition_rpm: int
    rotate_chain_memory_sec: float
    rotate_hold_stale_sec: float
    rotate_turn_percent_from_forward: int
    rotate_turn_percent_chain: int
    rotate_duration: float
    motor_rs485_target_min_interval_sec: float
    motor_forward_like_keepalive_sec: float
    motor_rs485_stop_mode: str
    safety_stop_mode: str
    motor_rs485_transition_stop_mode: str
    motor_rs485_transition_stop_delay_sec: float
    motor_rs485_transition_stop_repeat: int
    follow_brake_distance_m: float
    visible_steering_pid_enable: bool = False
    steering_feedback_poll_interval_sec: float = 0.10
    steering_feedback_log_interval_sec: float = 0.50
    steering_feedback_median_window: int = 3
    steering_feedback_left_body_deg_per_encoder_deg: float = 0.5225
    steering_feedback_right_body_deg_per_encoder_deg: float = 0.5424
    # Stop movement when the controller stops refreshing its intent.
    # This prevents stale motion during a camera/RKNN stall.
    action_intent_stale_sec: float = 0.45
    rotation_only: bool = False
    rotation_only_yaw_pulse_rpm: int = 30
    rotation_only_yaw_pulse_min_sec: float = 0.18
    rotation_only_yaw_pulse_max_sec: float = 0.32
    rotation_only_yaw_brake_sec: float = 0.18
    rotation_only_yaw_response_dps: float = 4.0
    rotation_only_yaw_zero_gap_sec: float = 0.03
    rotate_pulse_active_brake_enable: bool = False
    rotate_pulse_active_brake_rpm: int = 30
    rotate_pulse_active_brake_sec: float = 0.18
    follow_wheel_period_sec: float = 0.0
    # Opt-in: motor controller handles residual one-wheel reversal on a live
    # forward grant. Whole-car reverse/search retain their existing guards.
    follow_forward_handoff_enable: bool = False
    follow_residual_reverse_max_rpm: float = 0.0
    follow_forward_loss_handoff_enable: bool = False
    follow_turn_residual_max_rpm: float = 0.0
    follow_turn_acceleration_priority_enable: bool = False
    follow_turn_response_assist_enable: bool = False
    follow_cross_brake_enable: bool = False
    # Normal-follow handoffs default to speed-zero, NOT position-lock parking.
    follow_cross_brake_mode: str = "zero"


class MotionActionRuntime:
    """Run motor actions for the follow-car runtime.

    The owner is intentionally still the source of dynamic control state during
    this transition.  That keeps behavior stable while moving the bulky motor
    execution code out of request_0513_modular.py.
    """

    def __init__(
        self,
        owner,
        backend: MssdMotorBackend,
        config: ActionRuntimeConfig,
        symbols: ActionRuntimeSymbols,
        *,
        hard_stop_check: Callable[[Optional[int]], bool],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.owner = owner
        self._follow_wheel_clock = FollowWheelClock(getattr(config, "follow_wheel_period_sec", 0.05))
        self._periodic_follow_writing = False
        self._visible_wheel_guard = WheelZeroCrossGuard()
        self._forward_loss_handoff = ForwardLossHandoff()
        self._turn_response_assist = TurnResponseAssist()
        self._turn_buildup = TurnBuildup()
        self.backend = backend
        self.config = config
        self.symbols = symbols
        self.hard_stop_check = hard_stop_check
        self.logger = logger or logging.getLogger(__name__)
        if getattr(config, "follow_wheel_period_sec", 0.0) >= .05:
            self.logger.info("follow_wheel_config period_ms=%.1f safety_preempt=True scope=normal_visible_follow",
                             self._follow_wheel_clock.period * 1000)
        self.logger.info("follow_forward_handoff_config enabled=%s scope=authorized_forward "
                         "commanded_reverse_guard=True feedback_required=True residual_max_rpm=%.1f residual_samples=2",
                         getattr(config, "follow_forward_handoff_enable", False),
                         min(8.0, max(0.0, getattr(config, "follow_residual_reverse_max_rpm", 0.0))))
        self.logger.info("follow_cross_brake_config enabled=%s mode=%s "
                         "quiet_or_aligned_samples=2 timeout_never_authorizes_motion=True",
                         getattr(config, "follow_cross_brake_enable", False),
                         getattr(config, "follow_cross_brake_mode", "zero"))
        self.logger.info("forward_loss_handoff_config enabled=%s mode=zero quiet_samples=2 "
                         "turn_residual_max_rpm=%.1f response_window_ms=500 lease_extended=False "
                         "turn_confirmation_owner=wheel_guard aligned_outer_need_not_stop=True",
                         getattr(config, "follow_forward_loss_handoff_enable", False),
                         min(4.0, max(0.0, getattr(config, "follow_turn_residual_max_rpm", 0.0))))
        self._steering_feedback_lock = threading.Lock()
        self.logger.info("turn_response_assist_config enabled=%s gain=1.4 legacy_internal_max_diff_rpm=40 "
                         "final_limit=effective_visible_policy "
                         "max_duration_ms=350 motor_response_observation_ms=500 scope=qualified_forward "
                         "image_mode=two_feedback_buildup correction_units=half_wheel_diff "
                         "common_acceleration_cap=disabled",
                         getattr(config, "follow_turn_response_assist_enable", False))
        self.logger.info("turn_acceleration_priority_config legacy_requested=%s enabled=False "
                         "normal_base_preserved=True bounded_image_buildup=turn_response_assist "
                         "depth_lease_unchanged=True",
                         getattr(config, "follow_turn_acceleration_priority_enable", False))
        self._steering_feedback: Optional[SteeringFeedback] = None
        self._steering_feedback_thread: Optional[threading.Thread] = None
        self._steering_feedback_integrated_yaw_deg = 0.0
        self._steering_feedback_last_ts: Optional[float] = None
        self._steering_feedback_high_yaw_sign = 0
        self._steering_feedback_high_yaw_count = 0
        self._last_steering_feedback_warn_ts = 0.0
        self._last_steering_feedback_log_ts = 0.0
        filter_window = max(1, int(config.steering_feedback_median_window))
        self._steering_feedback_yaw_samples = deque(maxlen=filter_window)
        self._yaw_pulse_lock = threading.Lock()
        self._visible_yaw_pulse_direction = 0
        self._visible_yaw_pulse_started_monotonic = 0.0
        self._visible_yaw_pulse_deadline_monotonic = 0.0
        self._visible_yaw_pulse_planned_sec = 0.0
        self._visible_yaw_pulse_kind = "idle"
        self._visible_yaw_pulse_requested_rpm = 0
        self._visible_yaw_pulse_last_end_monotonic = 0.0
        self._visible_yaw_pulse_last_skip_log_monotonic = 0.0
        self._search_brake_direction = 0
        self._search_brake_started_monotonic = 0.0
        self._search_brake_deadline_monotonic = 0.0
        self._search_brake_source_action: Optional[int] = None
        self._rotate_cancel_revision = 0
        # A transition may clear the producer's authorization before the old
        # forward command has coasted down. This snapshot is for that ramp
        # only; it never authorizes a new forward command or keepalive.
        self._forward_coast_snapshot: Optional[tuple[int, bool]] = None
        self._near_yaw_park_applied = None
        self._near_yaw_park_settling = None
        self._dispatch_context = threading.local()
        self._current_action_snapshot = None
        self._search_reacquire_brake_request = None
        self._search_reacquire_brake_applied = None
        self._search_reacquire_brake_uid = None
        self._search_reacquire_brake_sent_at = 0.0
        self._search_reacquire_settling = None
        if not hasattr(owner, "action_queue_lock"):
            owner.action_queue_lock = threading.Lock()

    def start(self) -> None:
        owner = self.owner
        owner.action_stop_event.clear()
        self._steering_feedback_yaw_samples.clear()
        self._steering_feedback_integrated_yaw_deg = 0.0
        self._steering_feedback_last_ts = None
        self._steering_feedback_high_yaw_sign = 0
        self._steering_feedback_high_yaw_count = 0
        owner.action_thread = threading.Thread(target=self.run_loop, daemon=True)
        owner.action_thread.start()
        if self.config.visible_steering_pid_enable:
            self._steering_feedback_thread = threading.Thread(
                target=self._steering_feedback_loop,
                name="steering-encoder-feedback",
                daemon=True,
            )
            self._steering_feedback_thread.start()
            self.logger.info(
                "编码器转向反馈已启动: interval=%.3fs median_window=%d left_scale=%.4f right_scale=%.4f",
                float(self.config.steering_feedback_poll_interval_sec),
                int(self._steering_feedback_yaw_samples.maxlen or 1),
                float(self.config.steering_feedback_left_body_deg_per_encoder_deg),
                float(self.config.steering_feedback_right_body_deg_per_encoder_deg),
            )
        self.logger.info("动作执行线程已启动")

    def read_motor_feedback_rpm(self):
        feedback = self.get_steering_feedback()
        if feedback is None:
            return None
        return (
            int(feedback.left_speed_rpm),
            int(feedback.right_speed_rpm),
            int(round(feedback.left_forward_rpm)),
            int(round(feedback.right_forward_rpm)),
        )

    def get_steering_feedback(self) -> Optional[SteeringFeedback]:
        with self._steering_feedback_lock:
            return self._steering_feedback

    def get_recording_feedback(self) -> Optional[SteeringFeedback]:
        """Diagnostics only: snapshot the immutable published cache, no I/O/wait.

        The feedback worker replaces (never mutates) this frozen dataclass.
        Recording must not contend for the motor/feedback locks. Control keeps
        using get_steering_feedback; this accessor grants no motion authority.
        """
        return self._steering_feedback

    def join_feedback(self, timeout: float = 1.0) -> None:
        thread = self._steering_feedback_thread
        if thread is None:
            return
        thread.join(timeout=max(0.0, float(timeout)))
        if not thread.is_alive():
            self._steering_feedback_thread = None

    @staticmethod
    def _sign(value: int) -> int:
        return 1 if int(value) >= 0 else -1

    @staticmethod
    def _median_rate(values) -> float:
        ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
        if not ordered:
            return 0.0
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    def _steering_feedback_loop(self) -> None:
        owner = self.owner
        c = self.config
        interval = max(0.05, float(c.steering_feedback_poll_interval_sec))
        log_interval = max(interval, float(c.steering_feedback_log_interval_sec))
        left_forward_sign = self._sign(
            self.backend.wheel_raw_state_to_target("left", 1, 0x01)
        )
        right_forward_sign = self._sign(
            self.backend.wheel_raw_state_to_target("right", 1, 0x01)
        )
        while not owner.action_stop_event.is_set():
            started = time.monotonic()
            driver = self.backend.driver
            if driver is None:
                owner.action_stop_event.wait(min(0.10, interval))
                continue
            try:
                with owner.motor_io_lock:
                    left_read_started = time.monotonic()
                    left = driver.read_motor_status("left")
                    left_read_finished = time.monotonic()
                    right_read_started = time.monotonic()
                    right = driver.read_motor_status("right")
                    # Timestamp the physical read before another writer can
                    # issue NORMAL. A pre-stop read published late must not
                    # masquerade as post-stop settling evidence.
                    sample_ts = time.monotonic()
                left_speed = int(left.speed_rpm)
                right_speed = int(right.speed_rpm)
                left_forward_rpm, right_forward_rpm, raw_yaw_rate_right_dps = (
                    encoder_yaw_rate_right_dps(
                        left_speed,
                        right_speed,
                        left_forward_sign,
                        right_forward_sign,
                        c.steering_feedback_left_body_deg_per_encoder_deg,
                        c.steering_feedback_right_body_deg_per_encoder_deg,
                    )
                )
                # 电机状态寄存器在动作切换边界偶尔只返回一个周期的高转速。
                # 3 点中值会滤掉这种孤立尖峰；真实旋转连续出现时，第二个采样
                # 起仍会进入闭环，因此不会把持续的车身角速度误当成噪声。
                self._steering_feedback_yaw_samples.append(raw_yaw_rate_right_dps)
                yaw_rate_right_dps = self._median_rate(self._steering_feedback_yaw_samples)
                high_limit = max(35.0, float(getattr(c, "steering_feedback_max_yaw_dps", 35.0)))
                high_yaw = abs(raw_yaw_rate_right_dps) > high_limit
                high_sign = 1 if raw_yaw_rate_right_dps > 0 else -1 if raw_yaw_rate_right_dps < 0 else 0
                if high_yaw and high_sign == self._steering_feedback_high_yaw_sign:
                    self._steering_feedback_high_yaw_count += 1
                elif high_yaw:
                    self._steering_feedback_high_yaw_sign = high_sign
                    self._steering_feedback_high_yaw_count = 1
                else:
                    self._steering_feedback_high_yaw_sign = 0
                    self._steering_feedback_high_yaw_count = 0
                yaw_rate_confirmed = (not high_yaw) or self._steering_feedback_high_yaw_count >= 2
                previous_ts = self._steering_feedback_last_ts
                if previous_ts is not None:
                    dt = sample_ts - previous_ts
                    if 0.0 < dt <= 0.50:
                        self._steering_feedback_integrated_yaw_deg += yaw_rate_right_dps * dt
                self._steering_feedback_last_ts = sample_ts
                left_error = int(left.error_code)
                right_error = int(right.error_code)
                feedback = SteeringFeedback(
                    timestamp=sample_ts,
                    left_position_deg=int(left.position_degree),
                    right_position_deg=int(right.position_degree),
                    left_speed_rpm=left_speed,
                    right_speed_rpm=right_speed,
                    left_forward_rpm=left_forward_rpm,
                    right_forward_rpm=right_forward_rpm,
                    yaw_rate_right_dps=yaw_rate_right_dps,
                    raw_yaw_rate_right_dps=raw_yaw_rate_right_dps,
                    yaw_rate_confirmed=yaw_rate_confirmed,
                    integrated_yaw_right_deg=self._steering_feedback_integrated_yaw_deg,
                    left_error=left_error,
                    right_error=right_error,
                    trustworthy=(left_error == 0 and right_error == 0),
                    left_read_started=left_read_started,
                    left_read_finished=left_read_finished,
                    right_read_started=right_read_started,
                    right_read_finished=sample_ts,
                )
                with self._steering_feedback_lock:
                    self._steering_feedback = feedback
                if sample_ts - self._last_steering_feedback_log_ts >= log_interval:
                    self._last_steering_feedback_log_ts = sample_ts
                    self.logger.info(
                        "编码器转向反馈: 左位置=%d度 右位置=%d度 原始转速=%d/%dRPM 前进归一转速=%.1f/%.1fRPM 原始右转角速度=%.2f度/秒 滤波右转角速度=%.2f度/秒 累计角=%.2f度 错误码=%d/%d 可信=%s 高角速度确认=%s",
                        feedback.left_position_deg,
                        feedback.right_position_deg,
                        feedback.left_speed_rpm,
                        feedback.right_speed_rpm,
                        feedback.left_forward_rpm,
                        feedback.right_forward_rpm,
                        raw_yaw_rate_right_dps,
                        feedback.yaw_rate_right_dps,
                        feedback.integrated_yaw_right_deg,
                        feedback.left_error,
                        feedback.right_error,
                        feedback.trustworthy,
                        feedback.yaw_rate_confirmed,
                    )
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_steering_feedback_warn_ts >= 2.0:
                    self._last_steering_feedback_warn_ts = now
                    self.logger.warning("编码器转向反馈读取失败，PID降级为摄像头PD: %s", exc)
            elapsed = time.monotonic() - started
            owner.action_stop_event.wait(max(0.0, interval - elapsed))

    def _fresh_raw_yaw_rate(self, now_monotonic: float) -> tuple[Optional[float], Optional[float]]:
        feedback = self.get_steering_feedback()
        if feedback is None or not feedback.trustworthy:
            return None, None
        age = max(0.0, float(now_monotonic) - float(feedback.timestamp))
        if age > max(0.05, float(self.config.rotate_pulse_settle_feedback_stale_sec)):
            return None, age
        raw_rate = getattr(feedback, "raw_yaw_rate_right_dps", None)
        if raw_rate is None or not math.isfinite(float(raw_rate)):
            raw_rate = float(feedback.yaw_rate_right_dps)
        return float(raw_rate), age

    def _visible_yaw_pulse_duration(self, requested_rpm: int, *, braking: bool) -> float:
        c = self.config
        if braking:
            return max(0.04, float(c.rotation_only_yaw_brake_sec))
        minimum = max(0.04, float(c.rotation_only_yaw_pulse_min_sec))
        maximum = max(minimum, float(c.rotation_only_yaw_pulse_max_sec))
        pulse_rpm = max(1, int(c.rotation_only_yaw_pulse_rpm))
        demand = max(0.0, min(1.0, abs(int(requested_rpm)) / float(pulse_rpm)))
        return minimum + (maximum - minimum) * demand

    def _finish_visible_yaw_pulse_locked(
        self,
        reason: str,
        now_monotonic: float,
        *,
        send_zero: bool,
    ) -> None:
        if self._visible_yaw_pulse_direction == 0:
            return
        elapsed = max(
            0.0,
            float(now_monotonic) - float(self._visible_yaw_pulse_started_monotonic),
        )
        direction = self._visible_yaw_pulse_direction
        kind = self._visible_yaw_pulse_kind
        requested_rpm = self._visible_yaw_pulse_requested_rpm
        planned_sec = self._visible_yaw_pulse_planned_sec
        raw_rate, feedback_age = self._fresh_raw_yaw_rate(now_monotonic)
        self._visible_yaw_pulse_direction = 0
        self._visible_yaw_pulse_started_monotonic = 0.0
        self._visible_yaw_pulse_deadline_monotonic = 0.0
        self._visible_yaw_pulse_planned_sec = 0.0
        self._visible_yaw_pulse_kind = "idle"
        self._visible_yaw_pulse_requested_rpm = 0
        self._visible_yaw_pulse_last_end_monotonic = float(now_monotonic)
        if send_zero:
            self.send_rotate_pulse_zero_stop()
        self.logger.info(
            "rotation_only yaw pulse end: kind=%s direction=%s requested=%+drpm "
            "planned_ms=%.0f actual_ms=%.0f reason=%s raw_yaw=%s feedback_age_ms=%s zero=%s",
            kind,
            "right" if direction > 0 else "left",
            requested_rpm,
            planned_sec * 1000.0,
            elapsed * 1000.0,
            reason,
            "none" if raw_rate is None else f"{raw_rate:+.2f}",
            "none" if feedback_age is None else f"{feedback_age * 1000.0:.0f}",
            send_zero,
        )

    def request_rotation_only_yaw_pulse(
        self, correction_rpm: int, *, yaw_revision: Optional[int] = None
    ) -> bool:
        """Quantize visible yaw demand into one fixed-RPM, encoder-gated pulse."""
        c = self.config
        requested = int(correction_rpm)
        if not c.rotation_only or requested == 0:
            return False
        # Only a fresh visual action may start the next pulse. A motor keepalive
        # must not replay a stale pulse while inference is no longer updating.
        if str(getattr(self.owner, "_last_motor_dispatch_source", "")) != "action_queue":
            return False
        direction = 1 if requested > 0 else -1
        now_monotonic = time.monotonic()
        with self._yaw_pulse_lock:
            if self._visible_yaw_pulse_direction == direction:
                return False
            direction_changed = self._visible_yaw_pulse_direction != 0
            if self._visible_yaw_pulse_direction != 0:
                self._finish_visible_yaw_pulse_locked(
                    "direction_change",
                    now_monotonic,
                    send_zero=False,
                )
            zero_gap = max(0.0, float(c.rotation_only_yaw_zero_gap_sec))
            if (
                not direction_changed
                and now_monotonic - self._visible_yaw_pulse_last_end_monotonic < zero_gap
            ):
                if (
                    now_monotonic - self._visible_yaw_pulse_last_skip_log_monotonic
                    >= 0.20
                ):
                    self._visible_yaw_pulse_last_skip_log_monotonic = now_monotonic
                    self.logger.info(
                        "rotation_only yaw pulse deferred: requested=%+drpm remaining_gap_ms=%.0f",
                        requested,
                        max(
                            0.0,
                            zero_gap
                            - (now_monotonic - self._visible_yaw_pulse_last_end_monotonic),
                        )
                        * 1000.0,
                    )
                return False

            raw_rate, _feedback_age = self._fresh_raw_yaw_rate(now_monotonic)
            braking = bool(
                raw_rate is not None
                and abs(raw_rate) >= max(0.0, float(c.rotate_pulse_settle_max_yaw_rate_dps))
                and direction * raw_rate < 0.0
            )
            planned_sec = self._visible_yaw_pulse_duration(requested, braking=braking)
            fixed_rpm = max(1, int(c.rotation_only_yaw_pulse_rpm))
            self._visible_yaw_pulse_direction = direction
            self._visible_yaw_pulse_started_monotonic = now_monotonic
            self._visible_yaw_pulse_deadline_monotonic = now_monotonic + planned_sec
            self._visible_yaw_pulse_planned_sec = planned_sec
            self._visible_yaw_pulse_kind = "brake" if braking else "tracking"
            self._visible_yaw_pulse_requested_rpm = requested
            if not self.send_yaw_only(direction * fixed_rpm, yaw_revision=yaw_revision):
                self._finish_visible_yaw_pulse_locked(
                    "yaw_revision_revoked", now_monotonic, send_zero=False
                )
                return False
            self.logger.info(
                "rotation_only yaw pulse start: kind=%s direction=%s requested=%+drpm "
                "fixed=%drpm planned_ms=%.0f raw_yaw=%s",
                self._visible_yaw_pulse_kind,
                "right" if direction > 0 else "left",
                requested,
                fixed_rpm,
                planned_sec * 1000.0,
                "none" if raw_rate is None else f"{raw_rate:+.2f}",
            )
            return True

    def _start_search_active_brake(self, ended_action: Optional[int]) -> bool:
        c = self.config
        s = self.symbols
        if (
            not c.rotation_only
            or not c.rotate_pulse_active_brake_enable
            or ended_action not in (s.rotate_left, s.rotate_right)
        ):
            return False
        # rotate_right is positive/right yaw and rotate_left is negative/left
        # yaw, matching send_yaw_only() and the encoder feedback convention.
        previous_direction = 1 if ended_action == s.rotate_right else -1
        brake_direction = -previous_direction
        now_monotonic = time.monotonic()
        with self._yaw_pulse_lock:
            if self._visible_yaw_pulse_direction != 0:
                self._finish_visible_yaw_pulse_locked(
                    "search_brake_takeover", now_monotonic, send_zero=False
                )
            brake_rpm = max(1, int(c.rotate_pulse_active_brake_rpm))
            brake_sec = max(0.04, float(c.rotate_pulse_active_brake_sec))
            self._search_brake_direction = brake_direction
            self._search_brake_started_monotonic = now_monotonic
            self._search_brake_deadline_monotonic = now_monotonic + brake_sec
            self._search_brake_source_action = ended_action
            self.send_yaw_only(brake_direction * brake_rpm)
            self.logger.info(
                "search active brake start: ended_action=%s direction=%s rpm=%d max_ms=%.0f",
                s.action_names.get(ended_action, str(ended_action)),
                "right" if brake_direction > 0 else "left",
                brake_rpm,
                brake_sec * 1000.0,
            )
            return True

    def _finish_search_active_brake_locked(
        self,
        reason: str,
        now_monotonic: float,
        *,
        send_zero: bool,
    ) -> None:
        if self._search_brake_direction == 0:
            return
        elapsed = max(
            0.0,
            float(now_monotonic) - float(self._search_brake_started_monotonic),
        )
        direction = self._search_brake_direction
        source_action = self._search_brake_source_action
        raw_rate, feedback_age = self._fresh_raw_yaw_rate(now_monotonic)
        self._search_brake_direction = 0
        self._search_brake_started_monotonic = 0.0
        self._search_brake_deadline_monotonic = 0.0
        self._search_brake_source_action = None
        if send_zero:
            self.send_rotate_pulse_zero_stop()
        self.logger.info(
            "search active brake end: source=%s direction=%s actual_ms=%.0f reason=%s "
            "raw_yaw=%s feedback_age_ms=%s zero=%s",
            self.symbols.action_names.get(source_action, str(source_action)),
            "right" if direction > 0 else "left",
            elapsed * 1000.0,
            reason,
            "none" if raw_rate is None else f"{raw_rate:+.2f}",
            "none" if feedback_age is None else f"{feedback_age * 1000.0:.0f}",
            send_zero,
        )

    def _service_yaw_pulses(self) -> None:
        now_monotonic = time.monotonic()
        with self._yaw_pulse_lock:
            raw_rate, _feedback_age = self._fresh_raw_yaw_rate(now_monotonic)
            feedback = self.get_steering_feedback()
            if self._visible_yaw_pulse_direction != 0:
                elapsed = now_monotonic - self._visible_yaw_pulse_started_monotonic
                fresh_for_pulse = bool(
                    feedback is not None
                    and float(feedback.timestamp)
                    >= float(self._visible_yaw_pulse_started_monotonic)
                )
                response_limit = max(
                    0.0, float(self.config.rotation_only_yaw_response_dps)
                )
                response_seen = bool(
                    fresh_for_pulse
                    and raw_rate is not None
                    and self._visible_yaw_pulse_direction * raw_rate > 0.0
                    and abs(raw_rate) >= response_limit
                )
                brake_settled = bool(
                    self._visible_yaw_pulse_kind == "brake"
                    and elapsed >= 0.06
                    and fresh_for_pulse
                    and raw_rate is not None
                    and (
                        abs(raw_rate)
                        <= max(
                            0.0,
                            float(self.config.rotate_pulse_settle_max_yaw_rate_dps),
                        )
                        or response_seen
                    )
                )
                if brake_settled:
                    self._finish_visible_yaw_pulse_locked(
                        "encoder_brake_settled", now_monotonic, send_zero=True
                    )
                elif self._visible_yaw_pulse_kind == "tracking" and response_seen:
                    self._finish_visible_yaw_pulse_locked(
                        "encoder_first_response", now_monotonic, send_zero=True
                    )
                elif now_monotonic >= self._visible_yaw_pulse_deadline_monotonic:
                    self._finish_visible_yaw_pulse_locked(
                        "planned_duration", now_monotonic, send_zero=True
                    )

            if self._search_brake_direction != 0:
                elapsed = now_monotonic - self._search_brake_started_monotonic
                fresh_for_brake = bool(
                    feedback is not None
                    and float(feedback.timestamp)
                    >= float(self._search_brake_started_monotonic)
                )
                settled = bool(
                    elapsed >= 0.06
                    and fresh_for_brake
                    and raw_rate is not None
                    and (
                        abs(raw_rate)
                        <= max(
                            0.0,
                            float(self.config.rotate_pulse_settle_max_yaw_rate_dps),
                        )
                        or self._search_brake_direction * raw_rate
                        >= max(0.0, float(self.config.rotation_only_yaw_response_dps))
                    )
                )
                if settled:
                    self._finish_search_active_brake_locked(
                        "encoder_settled", now_monotonic, send_zero=True
                    )
                elif now_monotonic >= self._search_brake_deadline_monotonic:
                    self._finish_search_active_brake_locked(
                        "max_duration", now_monotonic, send_zero=True
                    )

    def cancel_yaw_pulses(self, reason: str, *, send_zero: bool = False) -> None:
        now_monotonic = time.monotonic()
        self.owner._rotate_transition_hold_active = False
        with self._yaw_pulse_lock:
            self._finish_visible_yaw_pulse_locked(
                f"cancelled:{reason}", now_monotonic, send_zero=send_zero
            )
            self._finish_search_active_brake_locked(
                f"cancelled:{reason}", now_monotonic, send_zero=send_zero
            )

    def cancel_active_rotate_for_observation(self, reason: str) -> bool:
        """Atomically stop a search turn before a candidate observation.

        Candidate evidence asks the camera to hold the chassis still for a
        fresh image.  Replacing the queue with ``STOP`` alone is racy: the
        action thread may still see the old ``current_command`` and refresh a
        rotate target between queue replacement and STOP consumption.  Clear
        that command under the same lock used by the executor, cancel any
        auxiliary pulse, then send one zero-target write.  The regular queued
        soft STOP remains responsible for the controller-level state and
        diagnostics.

        Returns ``True`` when a rotation command or auxiliary yaw pulse was
        cancelled, otherwise ``False``.
        """
        owner = self.owner
        s = self.symbols
        now_monotonic = time.monotonic()
        active_action: Optional[int] = None
        command_lock = getattr(owner, "command_lock", None)
        if command_lock is None:
            # Runtime owners always expose command_lock; keeping this fallback
            # makes the helper safe for lightweight dry-run owners.
            command_context = threading.Lock()
        else:
            command_context = command_lock
        with command_context:
            # A prepared TURN may already be waiting for motor I/O. Its
            # packet must stay revoked even after a new search turn is armed.
            self._rotate_cancel_revision += 1
            current = getattr(owner, "current_command", None)
            if current in (s.rotate_left, s.rotate_right):
                active_action = int(current)
                owner.current_command = None
                owner.command_start_time = None
                owner._rotate_follows_previous_rotate = True
                owner._last_rotate_end_ts = time.time()

        aux_active = self.yaw_aux_pulse_active()
        # Reset pulse bookkeeping even when current_command was already
        # cleared by a concurrent transition; this prevents a late service
        # tick from treating the old pulse as still active.
        self.cancel_yaw_pulses(f"observation:{reason}", send_zero=False)
        cancelled = active_action is not None or aux_active
        if not cancelled:
            return False
        try:
            self.send_rotate_pulse_zero_stop()
        except Exception:
            # Keep the state transition complete; the caller's soft STOP path
            # will retry through the normal motor-stop mechanism.
            self.logger.exception(
                "candidate observation zero transition failed: reason=%s action=%s",
                reason,
                s.action_names.get(active_action, str(active_action)),
            )
        self.logger.info(
            "candidate observation cancelled active rotation: reason=%s action=%s "
            "aux_pulse=%s zero_rpm=0 frame=%d",
            reason,
            s.action_names.get(active_action, "none"),
            aux_active,
            int(getattr(owner, "frame_index", -1)),
        )
        return True

    def yaw_aux_pulse_active(self) -> bool:
        with self._yaw_pulse_lock:
            return bool(
                self._visible_yaw_pulse_direction != 0
                or self._search_brake_direction != 0
            )

    def can_release_brake_hold(self, action: int) -> bool:
        owner = self.owner
        s = self.symbols
        if (getattr(self.backend, "motion_write_fault", None)
                or getattr(owner, "_runtime_shutdown_requested", False)):
            return False
        if self._search_reacquire_brake_request is not None:
            return False
        if getattr(owner, "_near_yaw_park_request", None) is not None:
            # Only the observation producer may release this hold, after it
            # validates newer evidence. A queued action is not new evidence.
            return False
        safety_hold_mode = getattr(owner, "_brake_hold_stop_mode", None)
        safety_hold_label = str(getattr(owner, "_brake_hold_label", "") or "")
        if safety_hold_mode:
            # Never release an emergency hold while an instantaneous IR or
            # distance hard-stop condition is still active.
            try:
                if self.hard_stop_check(action):
                    self.logger.info(
                        "安全刹车保持中拒绝释放: action=%s label=%s",
                        s.action_names.get(action, str(action)),
                        safety_hold_label,
                    )
                    return False
            except Exception as exc:
                self.logger.warning("安全刹车释放检查失败，保持停车: %s", exc)
                return False
            # A radar-unstable reverse stop may only resume visible recentering
            # after a fresh radar-backed distance sample.  Encoder hold alone
            # must not turn the vehicle again near the person.
            if safety_hold_label == "safety_hold_reverse_radar_unstable_stop":
                # The board runtime currently uses Astra Depth and disables mmWave.
                # Do not let the legacy radar-unavailable hold block visual search.
                mmwave_enabled = bool(getattr(owner, "_mmwave_runtime_enabled", False))
                distance_source = str(getattr(owner, "_distance_source", "") or "")
                if mmwave_enabled and distance_source not in {"vision_depth", "astra_depth"}:
                    distance_state = getattr(owner, "_last_frame_distance_state", None)
                    fusion_mode = str(getattr(distance_state, "fusion_mode", "") or "")
                    if fusion_mode not in {"radar", "mmwave", "direct"}:
                        self.logger.info(
                            "radar-unavailable brake hold blocks action=%s fusion_mode=%s",
                            s.action_names.get(action, str(action)),
                            fusion_mode or "none",
                        )
                        return False
                else:
                    self.logger.info(
                        "mmWave disabled at runtime; bypass radar-unstable brake hold: action=%s source=%s",
                        s.action_names.get(action, str(action)),
                        distance_source or "none",
                    )
        if safety_hold_label == "aimline_brake":
            # A detector box intersecting the camera aim line is an explicit
            # request to stop the scan. Do not let the next empty/blurred frame
            # release that stop while the chassis still carries search yaw.
            # Once fresh encoder feedback confirms that both wheels and body
            # yaw are quiet, the latest visual/search decision may take over.
            feedback = self.get_steering_feedback()
            now = time.monotonic()
            feedback_age = (
                float("inf")
                if feedback is None
                else max(0.0, now - float(feedback.timestamp))
            )
            feedback_fresh = bool(
                feedback is not None
                and feedback.trustworthy
                and feedback_age
                <= max(0.05, float(self.config.rotate_pulse_settle_feedback_stale_sec))
            )
            max_wheel_rpm = max(0, int(self.config.rotate_pulse_settle_max_wheel_rpm))
            max_yaw_rate = max(
                0.0,
                float(self.config.rotate_pulse_settle_max_yaw_rate_dps),
            )
            raw_yaw_rate = None if feedback is None else feedback.raw_yaw_rate_right_dps
            if raw_yaw_rate is None and feedback is not None:
                raw_yaw_rate = feedback.yaw_rate_right_dps
            settled = bool(
                feedback_fresh
                and abs(int(feedback.left_speed_rpm)) <= max_wheel_rpm
                and abs(int(feedback.right_speed_rpm)) <= max_wheel_rpm
                and raw_yaw_rate is not None
                and math.isfinite(float(raw_yaw_rate))
                and abs(float(raw_yaw_rate)) <= max_yaw_rate
            )
            if not settled:
                self.logger.info(
                    "aimline brake hold blocks action=%s feedback_fresh=%s age_ms=%s "
                    "wheels=%s/%sRPM raw_yaw=%sdps limits=%dRPM/%.1fdps",
                    s.action_names.get(action, str(action)),
                    feedback_fresh,
                    "none" if feedback is None else "%.0f" % (feedback_age * 1000.0),
                    "none" if feedback is None else feedback.left_speed_rpm,
                    "none" if feedback is None else feedback.right_speed_rpm,
                    "none" if raw_yaw_rate is None else "%.2f" % float(raw_yaw_rate),
                    max_wheel_rpm,
                    max_yaw_rate,
                )
                return False
            self.logger.info(
                "aimline brake settled; release allowed: action=%s age_ms=%.0f "
                "wheels=%d/%dRPM raw_yaw=%.2fdps",
                s.action_names.get(action, str(action)),
                feedback_age * 1000.0,
                int(feedback.left_speed_rpm),
                int(feedback.right_speed_rpm),
                float(raw_yaw_rate),
            )
        # 跟随停车距离锁存期间，普通前进和差速转向仍不能解锁刹车。
        # 但控制器明确生成“停车原地居中/近距丢失找回”动作时，允许旋转；
        # 该动作产生前已经再次通过极近距离、三路红外和视觉危险检查。
        controller = getattr(owner, "_follow_controller", None)
        forward_percent = int(getattr(owner, "_current_forward_percent", 0))
        steer_base_percent = int(getattr(owner, "_current_steer_base_percent", 0))
        steer_correction_rpm = int(
            getattr(owner, "_current_steer_correction_rpm", 0)
        )
        yaw_only_command = bool(
            steer_correction_rpm != 0
            and (
                (action == s.backward and forward_percent <= 0)
                or (
                    action in (s.steer_left, s.steer_right)
                    and steer_base_percent <= 0
                )
            )
        )
        if bool(getattr(controller, "target_stop_latched", False)):
            decision_reason = str(getattr(owner, "_last_control_decision_reason", "") or "")
            controlled_parked_rotate = decision_reason.startswith(
                (
                    "person_parked_recenter_",
                    "lost_wait_near_target_",
                    "lost_current_candidate_hold_",
                    "current_candidate_near_target_",
                    "search_candidate_approach_",
                )
            ) or decision_reason in (
                "near_distance_rotation_only",
                "target_visible_low_quality_yaw",
            )
            controlled_reverse = bool(
                action == s.backward and decision_reason == "target_approaching_reverse"
            )
            release_allowed = bool(
                (action in (s.rotate_left, s.rotate_right) and controlled_parked_rotate)
                or controlled_reverse
                or yaw_only_command
            )
            if release_allowed:
                self.logger.info(
                    "target-stop brake release allowed: action=%s reason=%s "
                    "parked_rotate=%s yaw_only=%s reverse=%s",
                    s.action_names.get(action, str(action)),
                    decision_reason or "none",
                    controlled_parked_rotate,
                    yaw_only_command,
                    controlled_reverse,
                )
            return release_allowed
        if action in (s.rotate_left, s.rotate_right):
            return True
        if action in (s.steer_left, s.steer_right):
            return steer_base_percent > 0 or yaw_only_command
        if action == s.backward:
            return forward_percent > 0 or yaw_only_command
        if action == s.forward:
            return forward_percent > 0
        return False

    def has_pending_actions(self) -> bool:
        owner = self.owner
        try:
            with owner.action_queue_lock:
                return not owner.action_queue.empty()
        except Exception:
            return False

    def rotate_pulse_enabled(self, action: Optional[int] = None) -> bool:
        """Return whether the current rotate intent uses lost-target pulses."""
        s = self.symbols
        if not self.config.rotate_pulse_brake_enable:
            return False
        if action is not None and action not in (s.rotate_left, s.rotate_right):
            return False
        # 近距离可见人物居中需要逐视觉帧连续刷新，不能继承搜索旋转的
        # 固定持续/停车观察周期。未设置该字段的旧调用方仍沿用脉冲模式。
        return bool(getattr(self.owner, "_current_rotate_pulse_enabled", True))

    def rotate_pulse_observation_pending(self, action: int) -> bool:
        """Gate the next pulse until the chassis settles and fresh frames arrive."""
        if not self.rotate_pulse_enabled(action):
            return False
        return self.search_observation_pending()

    def search_observation_pending(self) -> bool:
        """Return whether a stopped search still waits for quiet, fresh vision.

        Recognition owns the decision to start an evidence observation; this
        runtime owns only the encoder/time/frame gate. Keeping this query free
        of candidate or ReID state prevents motor settling from claiming an
        identity decision.
        """
        if self._rotate_pulse_settle_pending():
            self._defer_search_timeout_during_observation()
            return True
        time_pending = time.time() < float(getattr(self.owner, "_rotate_pause_until_ts", 0.0))
        current_frame = int(getattr(self.owner, "frame_index", -1))
        observe_until_frame = int(getattr(self.owner, "_rotate_observe_until_frame", -1))
        frame_pending = observe_until_frame >= 0 and current_frame < observe_until_frame
        pending = time_pending or frame_pending
        if pending:
            self._defer_search_timeout_during_observation()
        else:
            self.owner._rotate_observation_last_defer_monotonic = 0.0
        return pending

    def cancel_rotate_pulse_observation(self, reason: str) -> None:
        """Discard a settle/observation gate that no longer owns the control state."""
        owner = self.owner
        # A reacquired target owns yaw immediately. Do not let the short search
        # brake continue rotating underneath temporary/confirmed visual control.
        with self._yaw_pulse_lock:
            self._finish_search_active_brake_locked(
                f"target_visible:{reason}", time.monotonic(), send_zero=True
            )
        now_monotonic = time.monotonic()
        had_gate = bool(
            getattr(owner, "_rotate_settle_pending", False)
            or float(getattr(owner, "_rotate_pause_until_ts", 0.0)) > time.time()
            or int(getattr(owner, "_rotate_observe_until_frame", -1)) >= 0
        )
        old_epoch = int(getattr(owner, "_rotate_settle_search_epoch", -1))
        started = float(getattr(owner, "_rotate_settle_started_monotonic", 0.0) or 0.0)
        elapsed = max(0.0, now_monotonic - started) if started > 0.0 else 0.0
        owner._rotate_settle_pending = False
        owner._rotate_transition_hold_active = False
        owner._rotate_pause_until_ts = 0.0
        owner._rotate_observe_until_frame = -1
        owner._rotate_settle_search_epoch = -1
        owner._rotate_settle_started_monotonic = 0.0
        owner._rotate_settle_quiet_started_monotonic = 0.0
        owner._rotate_settle_completed_monotonic = now_monotonic
        owner._rotate_settle_completion_source = f"cancelled:{reason}"
        owner._rotate_observation_last_defer_monotonic = 0.0
        owner._last_rotate_settle_log_ts = 0.0
        if had_gate:
            self.logger.info(
                "rotate pulse settle cancelled: reason=%s old_epoch=%d current_epoch=%d elapsed=%.3fs frame=%d",
                reason,
                old_epoch,
                int(getattr(owner, "_search_epoch", 0)),
                elapsed,
                int(getattr(owner, "frame_index", -1)),
            )

    def _begin_rotate_settle_gate(self, source: str, reason: str) -> tuple[float, int]:
        owner = self.owner
        c = self.config
        now_wall = time.time()
        now_monotonic = time.monotonic()
        pause_until = now_wall + max(0.0, float(c.rotate_pulse_pause_sec))
        owner._rotate_pause_until_ts = pause_until
        owner._rotate_settle_started_monotonic = now_monotonic
        owner._rotate_settle_quiet_started_monotonic = 0.0
        owner._rotate_settle_completed_monotonic = 0.0
        owner._rotate_settle_completion_source = "pending"
        owner._rotate_settle_search_epoch = int(getattr(owner, "_search_epoch", 0))
        owner._last_rotate_settle_log_ts = 0.0
        owner._rotate_observation_last_defer_monotonic = now_monotonic
        owner._rotate_settle_pending = bool(c.rotate_pulse_settle_enable)
        if owner._rotate_settle_pending:
            # Observation frames before the wheels stop are likely blurred and
            # must not count toward the visual gate.
            owner._rotate_observe_until_frame = -1
        else:
            owner._rotate_observe_until_frame = int(getattr(owner, "frame_index", -1)) + max(
                0, int(c.rotate_pulse_observe_min_frames)
            )
        self.logger.info(
            "rotate pulse settle start: source=%s reason=%s epoch=%d enabled=%s "
            "pause_until=%.3f quiet_required=%.3fs timeout=%.3fs "
            "feedback_stale=%.3fs max_wheel=%dRPM max_yaw=%.2fdps observe_until_frame=%d",
            source,
            reason,
            int(owner._rotate_settle_search_epoch),
            owner._rotate_settle_pending,
            pause_until,
            c.rotate_pulse_settle_quiet_sec,
            c.rotate_pulse_settle_timeout_sec,
            c.rotate_pulse_settle_feedback_stale_sec,
            c.rotate_pulse_settle_max_wheel_rpm,
            c.rotate_pulse_settle_max_yaw_rate_dps,
            owner._rotate_observe_until_frame,
        )
        return pause_until, int(owner._rotate_observe_until_frame)

    def begin_search_epoch(self, reason: str, *, settle: bool = True) -> int:
        """Start a new lost-target search and isolate it from prior pulse state."""
        owner = self.owner
        self.cancel_rotate_pulse_observation(f"new_search:{reason}")
        owner._search_epoch = int(getattr(owner, "_search_epoch", 0)) + 1
        if settle:
            self._begin_rotate_settle_gate("search_entry", reason)
        else:
            owner._rotate_settle_search_epoch = int(owner._search_epoch)
            owner._rotate_settle_pending = False
            owner._rotate_pause_until_ts = 0.0
            owner._rotate_observe_until_frame = -1
            owner._rotate_settle_completion_source = "continuous_handoff"
        self.logger.info(
            "search epoch started: epoch=%d reason=%s frame=%d settle=%s",
            int(owner._search_epoch),
            reason,
            int(getattr(owner, "frame_index", -1)),
            bool(settle),
        )
        return int(owner._search_epoch)

    def _defer_search_timeout_during_observation(self) -> None:
        owner = self.owner
        now = time.monotonic()
        last = float(
            getattr(owner, "_rotate_observation_last_defer_monotonic", 0.0) or 0.0
        )
        owner._rotate_observation_last_defer_monotonic = now
        if last <= 0.0:
            return
        controller = getattr(owner, "_follow_controller", None)
        defer = getattr(controller, "defer_search_timeout", None)
        if callable(defer):
            defer(max(0.0, now - last))

    def begin_rotate_pulse_observation(self) -> tuple[float, int]:
        """Start the post-pulse observation gate.

        Search pulses may leave a small same-direction holding RPM so the
        motor does not repeatedly fall below its start threshold. In that
        mode there is no quiet encoder condition to wait for; only the
        configured fresh-frame observation gate remains.
        """
        owner = self.owner
        if bool(getattr(owner, "_rotate_transition_hold_active", False)):
            owner._rotate_transition_hold_active = False
            owner._rotate_settle_pending = False
            owner._rotate_pause_until_ts = 0.0
            owner._rotate_settle_completed_monotonic = time.monotonic()
            owner._rotate_settle_completion_source = "transition_hold"
            owner._rotate_observe_until_frame = int(
                getattr(owner, "frame_index", -1)
            ) + max(0, int(self.config.rotate_pulse_observe_min_frames))
            self.logger.info(
                "rotate pulse transition hold: rpm=%d frame=%d observe_until_frame=%d",
                int(self.config.rotate_pulse_transition_rpm),
                int(getattr(owner, "frame_index", -1)),
                int(owner._rotate_observe_until_frame),
            )
            return 0.0, int(owner._rotate_observe_until_frame)
        return self._begin_rotate_settle_gate("pulse_stop", "rotate_pulse_complete")

    def _complete_rotate_pulse_settle(
        self,
        source: str,
        *,
        feedback: Optional[SteeringFeedback],
        feedback_age_sec: float,
        quiet_sec: float,
    ) -> None:
        owner = self.owner
        now_monotonic = time.monotonic()
        current_frame = int(getattr(owner, "frame_index", -1))
        observe_until_frame = current_frame + max(
            0, int(self.config.rotate_pulse_observe_min_frames)
        )
        owner._rotate_settle_pending = False
        owner._rotate_settle_completed_monotonic = now_monotonic
        owner._rotate_settle_completion_source = str(source)
        owner._rotate_observe_until_frame = observe_until_frame
        started = float(getattr(owner, "_rotate_settle_started_monotonic", now_monotonic))
        self.logger.info(
            "rotate pulse settle complete: source=%s elapsed=%.3fs quiet=%.3fs feedback_age=%.3fs wheels=%s/%sRPM yaw=%sdps frame=%d observe_min_frames=%d observe_until_frame=%d",
            source,
            max(0.0, now_monotonic - started),
            max(0.0, quiet_sec),
            feedback_age_sec,
            "none" if feedback is None else feedback.left_speed_rpm,
            "none" if feedback is None else feedback.right_speed_rpm,
            "none" if feedback is None else f"{feedback.yaw_rate_right_dps:.2f}",
            current_frame,
            self.config.rotate_pulse_observe_min_frames,
            observe_until_frame,
        )

    def _rotate_pulse_settle_pending(self) -> bool:
        owner = self.owner
        c = self.config
        if not c.rotate_pulse_settle_enable or not bool(
            getattr(owner, "_rotate_settle_pending", False)
        ):
            return False

        settle_epoch = int(getattr(owner, "_rotate_settle_search_epoch", -1))
        search_epoch = int(getattr(owner, "_search_epoch", 0))
        if settle_epoch != search_epoch:
            self.logger.warning(
                "stale rotate settle epoch rejected: settle_epoch=%d current_epoch=%d frame=%d",
                settle_epoch,
                search_epoch,
                int(getattr(owner, "frame_index", -1)),
            )
            self.cancel_rotate_pulse_observation("stale_search_epoch")
            return False

        now = time.monotonic()
        started = float(getattr(owner, "_rotate_settle_started_monotonic", now) or now)
        elapsed = max(0.0, now - started)
        feedback = self.get_steering_feedback()
        feedback_age_sec = float("inf")
        feedback_usable = False
        if feedback is not None:
            feedback_age_sec = max(0.0, now - float(feedback.timestamp))
            feedback_usable = bool(
                feedback.trustworthy
                and feedback_age_sec <= max(0.05, float(c.rotate_pulse_settle_feedback_stale_sec))
            )

        quiet_started = float(
            getattr(owner, "_rotate_settle_quiet_started_monotonic", 0.0) or 0.0
        )
        quiet_elapsed = 0.0
        wheels_quiet = False
        yaw_quiet = False
        if feedback_usable and feedback is not None:
            wheels_quiet = bool(
                abs(int(feedback.left_speed_rpm)) <= int(c.rotate_pulse_settle_max_wheel_rpm)
                and abs(int(feedback.right_speed_rpm)) <= int(c.rotate_pulse_settle_max_wheel_rpm)
            )
            yaw_quiet = bool(
                abs(float(feedback.yaw_rate_right_dps))
                <= float(c.rotate_pulse_settle_max_yaw_rate_dps)
            )
            if wheels_quiet and yaw_quiet:
                if quiet_started <= 0.0:
                    quiet_started = now
                    owner._rotate_settle_quiet_started_monotonic = quiet_started
                quiet_elapsed = max(0.0, now - quiet_started)
                if quiet_elapsed >= max(0.0, float(c.rotate_pulse_settle_quiet_sec)):
                    self._complete_rotate_pulse_settle(
                        "feedback_quiet",
                        feedback=feedback,
                        feedback_age_sec=feedback_age_sec,
                        quiet_sec=quiet_elapsed,
                    )
                    return False
            else:
                # A post-stop rebound restarts the continuous quiet interval.
                owner._rotate_settle_quiet_started_monotonic = 0.0
        else:
            owner._rotate_settle_quiet_started_monotonic = 0.0

        # A trustworthy encoder can still report a small residual speed from
        # drivetrain quantization or brake rebound. Do not hold the search
        # indefinitely waiting for an exact quiet sample: after the bounded
        # timeout, resume the same-direction pulse and let the next fresh
        # frame close the loop. Direction reversals and safety stops still
        # use their explicit zero-RPM handoff.
        if elapsed >= max(0.0, float(c.rotate_pulse_settle_timeout_sec)):
            timeout_source = (
                "feedback_motion_timeout" if feedback_usable else "feedback_timeout"
            )
            self.logger.warning(
                "rotate pulse settle bounded timeout: elapsed=%.3fs source=%s feedback=%s age=%.3fs trustworthy=%s wheels=%s/%sRPM yaw=%sdps",
                elapsed,
                timeout_source,
                feedback is not None,
                feedback_age_sec,
                None if feedback is None else feedback.trustworthy,
                "none" if feedback is None else feedback.left_speed_rpm,
                "none" if feedback is None else feedback.right_speed_rpm,
                "none" if feedback is None else f"{feedback.yaw_rate_right_dps:.2f}",
            )
            self._complete_rotate_pulse_settle(
                timeout_source,
                feedback=feedback,
                feedback_age_sec=feedback_age_sec,
                quiet_sec=quiet_elapsed,
            )
            return False

        if now - float(getattr(owner, "_last_rotate_settle_log_ts", 0.0)) >= 0.20:
            owner._last_rotate_settle_log_ts = now
            self.logger.info(
                "rotate pulse settle wait: elapsed=%.3fs usable=%s feedback_age=%.3fs wheels=%s/%sRPM wheels_quiet=%s yaw=%sdps yaw_quiet=%s quiet=%.3f/%.3fs timeout=%.3fs",
                elapsed,
                feedback_usable,
                feedback_age_sec,
                "none" if feedback is None else feedback.left_speed_rpm,
                "none" if feedback is None else feedback.right_speed_rpm,
                wheels_quiet,
                "none" if feedback is None else f"{feedback.yaw_rate_right_dps:.2f}",
                yaw_quiet,
                quiet_elapsed,
                c.rotate_pulse_settle_quiet_sec,
                c.rotate_pulse_settle_timeout_sec,
            )
        return True

    def coast_down_before_rotate(self) -> None:
        owner = self.owner
        c = self.config
        if not c.use_percent_speed or not c.rotate_prep_coast_enable:
            return
        p0 = max(0, min(100, int(getattr(owner, "_current_forward_percent", 0))))
        if p0 <= 0:
            return
        coast_action = getattr(owner, "current_command", None)
        if coast_action not in self.symbols.forward_like_actions:
            return
        allow_below_min = self._forward_allow_below_min()
        snapshot = self._forward_coast_snapshot
        if (
            snapshot is not None
            and snapshot[1]
            and getattr(owner, "current_command", None) in self.symbols.forward_like_actions
            and getattr(owner, "_last_motor_dispatch_action", None)
            in self.symbols.forward_like_actions
        ):
            # Never ramp above the last successfully dispatched, approved
            # speed if a new decision already changed the live fields.
            p0 = min(p0, snapshot[0])
            allow_below_min = True
        steps = max(1, int(c.rotate_prep_coast_steps))
        total = max(0.05, float(c.rotate_prep_coast_total_sec))
        for i in range(steps):
            pct = max(0, int(p0 * (1.0 - (i + 1) / float(steps))))
            if not allow_below_min and 0 < pct < c.min_forward_percent:
                pct = c.min_forward_percent
            try:
                if not self.send_percent_drive(
                    pct, allow_below_min=allow_below_min, coast_action=coast_action
                ):
                    return
            except Exception as exc:
                self.logger.warning("转向前滑行降速失败: %s", exc)
            time.sleep(total / steps)
        self.logger.debug("转向前滑行降速完成: %d%% -> 0%%", p0)

    def run_loop(self) -> None:
        owner = self.owner
        c = self.config
        s = self.symbols
        while not owner.action_stop_event.is_set():
            try:
                self._dispatch_context.command = None
                if self._service_motion_write_fault() or self._service_runtime_shutdown():
                    time.sleep(0.01)
                    continue
                action_from_queue = False
                self._service_follow_wheels()
                self._service_yaw_pulses()
                if self.yaw_aux_pulse_active():
                    now = time.time()
                    if now - owner._last_hard_stop_check_ts >= owner._hard_stop_check_interval_sec:
                        owner._last_hard_stop_check_ts = now
                        if self.hard_stop_check(None):
                            self.cancel_yaw_pulses("hard_stop", send_zero=False)
                            self.send_stop_with_brake_hold("hard_stop")
                            time.sleep(0.01)
                            continue
                if owner._brake_hold_active:
                    now = time.time()
                    if now - owner._last_brake_hold_send_ts >= owner._brake_hold_refresh_interval_sec:
                        owner._last_brake_hold_send_ts = now
                        try:
                            self.send_percent_brake(
                                mode=getattr(owner, "_brake_hold_stop_mode", None),
                                label=getattr(owner, "_brake_hold_label", "brake"),
                            )
                        except Exception as exc:
                            self.logger.warning("brake 保持态下重复锁轮失败: %s", exc)

                if c.enable_motor_rpm_feedback:
                    now = time.time()
                    if now - owner._last_motor_feedback_check_ts >= owner._motor_feedback_poll_interval_sec:
                        owner._last_motor_feedback_check_ts = now
                        rpm_pair = self.read_motor_feedback_rpm()
                        if rpm_pair is not None:
                            rpm_m1_raw, rpm_m2_raw, rpm_m1_s, rpm_m2_s = rpm_pair
                            if (
                                rpm_m1_s < 0
                                and rpm_m2_s < 0
                                and owner.current_command != s.backward
                            ):
                                if not owner._brake_hold_active:
                                    self.logger.warning(
                                        "检测到 M1/M2 同时倒转，进入 brake 保持态: signed=%d,%d raw=%d,%d",
                                        rpm_m1_s,
                                        rpm_m2_s,
                                        rpm_m1_raw,
                                        rpm_m2_raw,
                                    )
                                owner._brake_hold_active = True
                                owner._follow_distance_hold = None
                                owner._use_soft_stop_next = False
                                owner._last_brake_hold_send_ts = 0.0

                try:
                    dequeue_ts = time.monotonic()
                    with owner.action_queue_lock:
                        park_generation = int(getattr(owner, "_near_yaw_park_generation", 0))
                        queued_command = owner.action_queue.get_nowait()
                        action = int(queued_command)
                        command = queued_command if isinstance(queued_command, ActionCommandSnapshot) else None
                        queue_after_pop = list(owner.action_queue.queue)
                    self._dispatch_context.command = command
                    action_from_queue = True
                    queue_age_ms = (
                        (dequeue_ts - float(command.enqueued_at if command else getattr(owner, "_last_action_queue_replace_ts", 0.0))) * 1000.0
                        if float(command.enqueued_at if command else getattr(owner, "_last_action_queue_replace_ts", 0.0)) > 0
                        else -1.0
                    )
                    action_kind_age_ms = (
                        (dequeue_ts - float(getattr(owner, "_last_tracker_action_change_ts", 0.0))) * 1000.0
                        if float(getattr(owner, "_last_tracker_action_change_ts", 0.0)) > 0
                        else -1.0
                    )
                    self.logger.info(
                        "action queue pop: seq=%d frame=%d reason=%s enqueue_frame=%d action=%s remaining=%d queue_after=%s queue_wait_ms=%.1f action_kind_age_ms=%.1f",
                        command.revision if command else int(getattr(owner, "_last_action_queue_seq", 0)),
                        int(getattr(owner, "frame_index", -1)),
                        command.reason if command else getattr(owner, "_last_action_queue_reason", ""),
                        command.control_frame if command else int(getattr(owner, "_last_action_queue_replace_frame", -1)),
                        s.action_names.get(action, str(action)),
                        len(queue_after_pop),
                        [s.action_names.get(int(a), str(a)) for a in queue_after_pop],
                        queue_age_ms,
                        action_kind_age_ms,
                    )
                    if should_drop_queued_action(
                        action,
                        queue_age_ms / 1000.0,
                        ttl_sec=self.config.action_intent_stale_sec,
                        stop_action=s.stop,
                    ):
                        self.logger.warning(
                            "queued action expired; dropping stale motion: action=%s age=%.3fs threshold=%.3fs",
                            s.action_names.get(action, str(action)),
                            queue_age_ms / 1000.0,
                            max(0.20, float(self.config.action_intent_stale_sec)),
                        )
                        continue
                    with owner.command_lock:
                        if not self._command_revision_write_allowed("queue_adopt"):
                            continue
                        if not self._near_yaw_park_queue_action_allowed(action, park_generation):
                            if self.hard_stop_check(action):
                                self.send_stop_with_brake_hold("hard_stop")
                            continue
                        if (
                            (owner.stop_action_execution or owner.person_detected_flag)
                            and owner.current_command is not None
                        ):
                            interrupted_action = owner.current_command
                            was_rotate = interrupted_action in (s.rotate_left, s.rotate_right)
                            self.logger.info(
                                "新动作入队前执行停止信号: 旧动作=%s 新动作=%s 旧动作是否旋转=%s",
                                s.action_names.get(interrupted_action, str(interrupted_action)),
                                s.action_names.get(action, str(action)),
                                was_rotate,
                            )
                            owner.current_command = None
                            owner.command_start_time = None
                            if was_rotate:
                                owner._rotate_follows_previous_rotate = True
                                owner._last_rotate_end_ts = time.time()
                                self.brake_lock_after_rotate_pulse(interrupted_action)
                            else:
                                self.send_stop_with_brake_hold(
                                    "queued_action_stop_signal",
                                    preserve_motion_params=action in s.movement_actions,
                                )
                        if action in (
                            s.forward,
                            s.backward,
                            s.rotate_left,
                            s.rotate_right,
                            s.steer_left,
                            s.steer_right,
                        ):
                            owner.stop_action_execution = False
                            owner.person_detected_flag = False
                        if self.rotate_pulse_observation_pending(action):
                            now_ts = time.time()
                            current_frame = int(getattr(owner, "frame_index", -1))
                            observe_until_frame = int(getattr(owner, "_rotate_observe_until_frame", -1))
                            if now_ts - float(getattr(owner, "_last_rotate_pause_log_ts", 0.0)) >= 0.20:
                                owner._last_rotate_pause_log_ts = now_ts
                                self.logger.info(
                                    "rotate pulse observe: action=%s remaining=%.3fs frame=%d observe_until_frame=%d remaining_frames=%d pause=%.3fs duration=%.3fs raw=%d raw_source=%s percent=%d",
                                    s.action_names.get(action, str(action)),
                                    max(0.0, float(getattr(owner, "_rotate_pause_until_ts", 0.0)) - now_ts),
                                    current_frame,
                                    observe_until_frame,
                                    max(0, observe_until_frame - current_frame),
                                    c.rotate_pulse_pause_sec,
                                    c.rotate_duration,
                                    int(getattr(owner, "_current_rotate_raw_target", c.motor_rotate_raw_target)),
                                    str(getattr(owner, "_current_rotate_raw_source", "default")),
                                    int(getattr(owner, "_current_rotate_turn_percent", c.rotate_turn_percent_from_forward)),
                                )
                            if c.rotate_pulse_stop_mode == "brake":
                                try:
                                    self.send_percent_brake(
                                        mode=getattr(owner, "_brake_hold_stop_mode", None),
                                        label=getattr(owner, "_brake_hold_label", "brake"),
                                    )
                                except Exception as exc:
                                    self.logger.warning("停顿期 brake 失败: %s", exc)
                            try:
                                with owner.action_queue_lock:
                                    owner.action_queue.put_nowait(queued_command)
                            except queue.Full:
                                self.logger.warning("停顿期旋转命令回队失败(queue满)")
                            time.sleep(0.02)
                            continue
                        if owner._brake_hold_active:
                            if self.can_release_brake_hold(action):
                                owner._brake_hold_active = False
                                owner._follow_distance_hold = None
                                owner._brake_hold_stop_mode = None
                                owner._brake_hold_label = "brake"
                                self.logger.info("收到可释放命令，解除 brake 保持态: %s", action)
                            else:
                                self.logger.info(
                                    "brake 保持态生效，忽略命令: %s"
                                    "（仅安全检查通过的旋转、零速偏航或非零纵向运动可解除）",
                                    action,
                                )
                                now_ts = time.time()
                                if (
                                    now_ts - owner._last_brake_hold_send_ts
                                    >= owner._brake_hold_refresh_interval_sec
                                ):
                                    owner._last_brake_hold_send_ts = now_ts
                                    try:
                                        self.send_percent_brake(
                                            mode=getattr(owner, "_brake_hold_stop_mode", None),
                                            label=getattr(owner, "_brake_hold_label", "brake"),
                                        )
                                    except Exception as exc:
                                        self.logger.warning("brake 保持态下再次锁轮失败: %s", exc)
                                continue
                        # Same-action refreshes also carry a new revision;
                        # their subsequent keepalive must not retain the old one.
                        self._current_action_snapshot = command
                        if action == owner.current_command:
                            if action in (s.forward, s.backward, s.steer_left, s.steer_right) or (
                                action in (s.rotate_left, s.rotate_right)
                                and not self.rotate_pulse_enabled(action)
                            ):
                                if action in (s.rotate_left, s.rotate_right):
                                    now_ts = time.time()
                                    if now_ts - float(getattr(owner, "_last_rotate_hold_refresh_log_ts", 0.0)) >= 0.50:
                                        elapsed = 0.0 if owner.command_start_time is None else now_ts - owner.command_start_time
                                        owner._last_rotate_hold_refresh_log_ts = now_ts
                                        self.logger.info(
                                            "rotate hold refresh: action=%s elapsed_before_refresh=%.3fs hold_stale=%.3fs pulse_enable=%s raw=%d raw_source=%s percent=%d",
                                            s.action_names.get(action, str(action)),
                                            elapsed,
                                            c.rotate_hold_stale_sec,
                                            self.rotate_pulse_enabled(action),
                                            int(getattr(owner, "_current_rotate_raw_target", c.motor_rotate_raw_target)),
                                            str(getattr(owner, "_current_rotate_raw_source", "default")),
                                            int(getattr(owner, "_current_rotate_turn_percent", c.rotate_turn_percent_from_forward)),
                                        )
                                owner.command_start_time = time.time()
                                self.logger.debug("连续同指令，刷新计时: %s", action)
                            else:
                                now_ts = time.time()
                                if now_ts - float(getattr(owner, "_last_rotate_pulse_active_log_ts", 0.0)) >= 0.20:
                                    owner._last_rotate_pulse_active_log_ts = now_ts
                                    self.logger.info(
                                        "rotate pulse active: action=%s same-command does not refresh timer duration=%.3fs pause=%.3fs",
                                        s.action_names.get(action, str(action)),
                                        c.rotate_duration,
                                        c.rotate_pulse_pause_sec,
                                    )
                        else:
                            switch_start_ts = time.monotonic()
                            switch_start_perf = time.perf_counter()
                            old_cmd = owner.current_command
                            # A search direction change must invalidate the
                            # old pulse before any keepalive/refresh can run.
                            # The controller normally clears this at the
                            # handoff boundary; this executor-side guard also
                            # covers queue races between a vision decision and
                            # the action thread.
                            opposite_search_switch = bool(
                                old_cmd in (s.rotate_left, s.rotate_right)
                                and action in (s.rotate_left, s.rotate_right)
                                and old_cmd != action
                                and str(getattr(owner, "_last_control_decision_reason", "")).startswith("search_")
                            )
                            if opposite_search_switch:
                                try:
                                    self.send_rotate_pulse_zero_stop()
                                except Exception as exc:
                                    self.logger.warning(
                                        "search direction switch zero transition failed: old=%s new=%s error=%s",
                                        s.action_names.get(old_cmd, str(old_cmd)),
                                        s.action_names.get(action, str(action)),
                                        exc,
                                    )
                                owner.current_command = None
                                owner.command_start_time = None
                                owner._rotate_follows_previous_rotate = True
                                owner._last_rotate_end_ts = time.time()
                                self.logger.info(
                                    "search direction switch zero transition: old=%s new=%s zero_rpm=0 frame=%d",
                                    s.action_names.get(old_cmd, str(old_cmd)),
                                    s.action_names.get(action, str(action)),
                                    int(getattr(owner, "frame_index", -1)),
                                )
                                old_cmd = None
                            old_duration_ms = (
                                (time.time() - owner.command_start_time) * 1000.0
                                if owner.command_start_time is not None
                                else -1.0
                            )
                            since_last_switch_ms = (
                                (switch_start_ts - float(getattr(owner, "_last_action_switch_ts", 0.0))) * 1000.0
                                if float(getattr(owner, "_last_action_switch_ts", 0.0)) > 0
                                else -1.0
                            )
                            last_switch_frame = int(getattr(owner, "_last_action_switch_frame", -1))
                            switch_frame_delta = (
                                int(getattr(owner, "frame_index", -1)) - last_switch_frame
                                if last_switch_frame >= 0
                                else -1
                            )
                            queue_wait_ms = (
                                (switch_start_ts - float(getattr(owner, "_last_action_queue_replace_ts", 0.0))) * 1000.0
                                if float(getattr(owner, "_last_action_queue_replace_ts", 0.0)) > 0
                                else -1.0
                            )
                            action_kind_age_ms = (
                                (switch_start_ts - float(getattr(owner, "_last_tracker_action_change_ts", 0.0))) * 1000.0
                                if float(getattr(owner, "_last_tracker_action_change_ts", 0.0)) > 0
                                else -1.0
                            )
                            rotate_strength_source = "none"
                            transition_needed = self.needs_transition_stop(old_cmd, action)
                            self.logger.info(
                                "action switch timing: seq=%d frame=%d old=%s new=%s since_last_switch_ms=%.1f frame_delta=%d old_duration_ms=%.1f queue_wait_ms=%.1f action_kind_age_ms=%.1f transition_needed=%s queue_remaining=%d queue_after_pop=%s",
                                int(getattr(owner, "_last_action_queue_seq", 0)),
                                int(getattr(owner, "frame_index", -1)),
                                s.action_names.get(old_cmd, str(old_cmd)),
                                s.action_names.get(action, str(action)),
                                since_last_switch_ms,
                                switch_frame_delta,
                                old_duration_ms,
                                queue_wait_ms,
                                action_kind_age_ms,
                                transition_needed,
                                len(queue_after_pop),
                                [s.action_names.get(int(a), str(a)) for a in queue_after_pop],
                            )
                            transition_stop_ms = 0.0
                            if transition_needed:
                                transition_start = time.perf_counter()
                                self.send_motion_transition_stop(old_cmd, action)
                                transition_stop_ms = (time.perf_counter() - transition_start) * 1000.0
                            prep_coast_ms = 0.0
                            if (
                                c.use_percent_speed
                                and c.rotate_prep_coast_enable
                                and not self._visible_wheel_control_active()
                                and old_cmd in (s.forward, s.steer_left, s.steer_right)
                                and action in (s.rotate_left, s.rotate_right)
                            ):
                                prep_start = time.perf_counter()
                                self.coast_down_before_rotate()
                                prep_coast_ms = (time.perf_counter() - prep_start) * 1000.0
                            if action in (s.rotate_left, s.rotate_right):
                                now_ts = time.time()
                                if old_cmd in (s.rotate_left, s.rotate_right):
                                    owner._current_rotate_turn_percent = c.rotate_turn_percent_chain
                                    rotate_strength_source = "old_rotate"
                                elif owner._rotate_follows_previous_rotate:
                                    owner._current_rotate_turn_percent = c.rotate_turn_percent_chain
                                    owner._rotate_follows_previous_rotate = False
                                    rotate_strength_source = "previous_pulse_or_stale"
                                elif (now_ts - float(getattr(owner, "_last_rotate_end_ts", 0.0))) <= float(c.rotate_chain_memory_sec):
                                    owner._current_rotate_turn_percent = c.rotate_turn_percent_chain
                                    rotate_strength_source = "chain_memory"
                                else:
                                    owner._current_rotate_turn_percent = c.rotate_turn_percent_from_forward
                                    rotate_strength_source = "from_forward_or_idle"
                            if action in (s.forward, s.backward, s.steer_left, s.steer_right):
                                owner._rotate_follows_previous_rotate = False
                            owner.current_command = action
                            self._current_action_snapshot = command
                            if self.rotate_pulse_enabled(action):
                                owner.command_start_time = None
                            else:
                                owner.command_start_time = time.time()
                            owner._last_action_switch_ts = switch_start_ts
                            owner._last_action_switch_frame = int(getattr(owner, "frame_index", -1))
                            self.logger.info(
                                "action switch applied: seq=%d frame=%d old=%s new=%s transition_stop_ms=%.1f prep_coast_ms=%.1f assign_total_ms=%.1f command_timer=%s",
                                int(getattr(owner, "_last_action_queue_seq", 0)),
                                int(getattr(owner, "frame_index", -1)),
                                s.action_names.get(old_cmd, str(old_cmd)),
                                s.action_names.get(action, str(action)),
                                transition_stop_ms,
                                prep_coast_ms,
                                (time.perf_counter() - switch_start_perf) * 1000.0,
                                "after_motor_dispatch" if self.rotate_pulse_enabled(action) else "command_start",
                            )
                            if action in (s.rotate_left, s.rotate_right):
                                self.logger.info(
                                    "rotate start: action=%s old_action=%s strength_source=%s percent=%d raw=%d raw_source=%s default_raw=%d pulse_enable=%s duration=%.3fs pause=%.3fs hold_stale=%.3fs timer=%s",
                                    s.action_names.get(action, str(action)),
                                    s.action_names.get(old_cmd, str(old_cmd)),
                                    rotate_strength_source,
                                    owner._current_rotate_turn_percent,
                                    int(getattr(owner, "_current_rotate_raw_target", c.motor_rotate_raw_target)),
                                    str(getattr(owner, "_current_rotate_raw_source", "default")),
                                    c.motor_rotate_raw_target,
                                    self.rotate_pulse_enabled(action),
                                    c.rotate_duration,
                                    c.rotate_pulse_pause_sec,
                                    c.rotate_hold_stale_sec,
                                    "after_motor_dispatch" if self.rotate_pulse_enabled(action) else "command_start",
                                )
                            else:
                                self.logger.info("收到新指令: %s, 开始执行", action)
                except queue.Empty:
                    with owner.command_lock:
                        if owner.current_command is not None:
                            if owner.stop_action_execution or owner.person_detected_flag:
                                self.logger.info("收到停止信号，立即停止当前命令")
                                ended_action = owner.current_command
                                was_rotate = ended_action in (s.rotate_left, s.rotate_right)
                                if was_rotate:
                                    owner._rotate_follows_previous_rotate = True
                                owner.current_command = None
                                owner.command_start_time = None
                                if was_rotate:
                                    self.brake_lock_after_rotate_pulse(ended_action)
                                else:
                                    self.send_stop_with_brake_hold("stop_signal")
                                owner.stop_action_execution = False
                                owner.person_detected_flag = False
                            else:
                                # A movement command must have a recent controller intent.
                                # Stop instead of keeping an obsolete target alive.
                                intent_ts = float(getattr(owner, "_last_action_intent_ts", 0.0) or 0.0)
                                intent_stale_sec = max(0.20, float(self.config.action_intent_stale_sec))
                                if (
                                    owner.current_command in s.movement_actions
                                    and not self.has_pending_actions()
                                    and intent_ts > 0.0
                                    and time.monotonic() - intent_ts >= intent_stale_sec
                                ):
                                    stale_action = owner.current_command
                                    stale_age = time.monotonic() - intent_ts
                                    owner.current_command = None
                                    owner.command_start_time = None
                                    owner.stop_action_execution = False
                                    owner.person_detected_flag = False
                                    self.logger.warning(
                                        "action intent stale; stopping old action: action=%s age=%.3fs threshold=%.3fs",
                                        s.action_names.get(stale_action, str(stale_action)),
                                        stale_age,
                                        intent_stale_sec,
                                    )
                                    self.send_stop_with_brake_hold("action_intent_stale")
                                    time.sleep(0.01)
                                    continue
                                if owner.command_start_time is None:
                                    time.sleep(0.01)
                                    continue
                                elapsed = time.time() - owner.command_start_time
                                if (
                                    owner.current_command in (s.rotate_left, s.rotate_right)
                                    and not self.rotate_pulse_enabled(owner.current_command)
                                ):
                                    elapsed = self.rotate_hold_age_sec()
                                if (
                                    owner.current_command in (s.rotate_left, s.rotate_right)
                                    and self.rotate_pulse_enabled(owner.current_command)
                                    and elapsed >= c.rotate_duration
                                ):
                                    ended_action = owner.current_command
                                    owner._rotate_follows_previous_rotate = True
                                    owner._last_rotate_end_ts = time.time()
                                    owner.current_command = None
                                    owner.command_start_time = None
                                    self.brake_lock_after_rotate_pulse(
                                        ended_action,
                                        active_brake=True,
                                    )
                                    pause_until, observe_until_frame = self.begin_rotate_pulse_observation()
                                    self.logger.info(
                                        "rotate pulse duration reached: action=%s elapsed=%.3fs duration=%.3fs -> stop pause=%.3fs pause_until=%.3f settle_enable=%s observe_min_frames=%d observe_until_frame=%d",
                                        s.action_names.get(ended_action, str(ended_action)),
                                        elapsed,
                                        c.rotate_duration,
                                        c.rotate_pulse_pause_sec,
                                        pause_until,
                                        c.rotate_pulse_settle_enable,
                                        c.rotate_pulse_observe_min_frames,
                                        observe_until_frame,
                                    )
                                elif (
                                    owner.current_command in (s.rotate_left, s.rotate_right)
                                    and not self.rotate_pulse_enabled(owner.current_command)
                                    and elapsed >= c.rotate_hold_stale_sec
                                ):
                                    ended_action = owner.current_command
                                    self.logger.info(
                                        "rotate hold stale stop: action=%s elapsed=%.3fs hold_stale=%.3fs pulse_enable=%s",
                                        s.action_names.get(ended_action, str(ended_action)),
                                        elapsed,
                                        c.rotate_hold_stale_sec,
                                        self.rotate_pulse_enabled(ended_action),
                                    )
                                    owner._rotate_follows_previous_rotate = True
                                    owner._last_rotate_end_ts = time.time()
                                    owner.current_command = None
                                    owner.command_start_time = None
                                    self.send_stop_with_brake_hold("rotate_stale")

                with owner.command_lock:
                    if getattr(self._dispatch_context, "command", None) is None:
                        self._dispatch_context.command = self._current_action_snapshot
                    if owner.current_command is not None and not owner.stop_action_execution and not owner.person_detected_flag:
                        if (
                            owner.current_command in (s.rotate_left, s.rotate_right)
                            and owner.command_start_time is not None
                            and self.rotate_pulse_enabled(owner.current_command)
                            and (time.time() - owner.command_start_time) >= c.rotate_duration
                        ):
                            elapsed = time.time() - owner.command_start_time
                            ended_action = owner.current_command
                            owner._rotate_follows_previous_rotate = True
                            owner._last_rotate_end_ts = time.time()
                            owner.current_command = None
                            owner.command_start_time = None
                            self.brake_lock_after_rotate_pulse(
                                ended_action,
                                active_brake=True,
                            )
                            pause_until, observe_until_frame = self.begin_rotate_pulse_observation()
                            self.logger.warning(
                                "rotate pulse duration stop: action=%s elapsed=%.3fs duration=%.3fs pause=%.3fs pause_until=%.3f settle_enable=%s observe_min_frames=%d observe_until_frame=%d",
                                s.action_names.get(ended_action, str(ended_action)),
                                elapsed,
                                c.rotate_duration,
                                c.rotate_pulse_pause_sec,
                                pause_until,
                                c.rotate_pulse_settle_enable,
                                c.rotate_pulse_observe_min_frames,
                                observe_until_frame,
                            )
                            time.sleep(0.01)
                            continue
                        if owner._brake_hold_active:
                            time.sleep(0.01)
                            continue
                        if owner.current_command in (
                            s.forward,
                            s.backward,
                            s.steer_left,
                            s.steer_right,
                            s.rotate_left,
                            s.rotate_right,
                        ):
                            now = time.time()
                            if now - owner._last_hard_stop_check_ts >= owner._hard_stop_check_interval_sec:
                                owner._last_hard_stop_check_ts = now
                                current_action = owner.current_command
                                if self.hard_stop_check(current_action):
                                    self.logger.warning(
                                        "硬停触发：沙坑/水坑视觉危险、IR/距离安全条件触发，"
                                        "action=%s 距离阈值<%.2fm，立刻发送STOP并打断当前动作",
                                        s.action_names.get(current_action, str(current_action)),
                                        c.follow_brake_distance_m,
                                    )
                                    owner.current_command = None
                                    owner.command_start_time = None
                                    owner.stop_action_execution = False
                                    owner.person_detected_flag = False
                                    self.send_stop_with_brake_hold("hard_stop")
                                    time.sleep(0.01)
                                    continue
                        if (
                            owner.current_command in (s.rotate_left, s.rotate_right)
                            and self.rotate_pulse_enabled(owner.current_command)
                            and owner.command_start_time is not None
                            and not self.has_pending_actions()
                        ):
                            now_ts = time.time()
                            refresh_interval = max(0.02, float(c.motor_rs485_target_min_interval_sec))
                            if now_ts - float(getattr(owner, "_last_rotate_pulse_refresh_ts", 0.0)) >= refresh_interval:
                                self.logger.info(
                                    "rotate pulse target refresh: action=%s elapsed=%.3fs duration=%.3fs interval=%.3fs",
                                    s.action_names.get(owner.current_command, str(owner.current_command)),
                                    now_ts - owner.command_start_time,
                                    c.rotate_duration,
                                    refresh_interval,
                                )
                                owner._last_motor_dispatch_source = "rotate_refresh"
                                self.send_robot_command(owner.current_command)
                            time.sleep(0.01)
                            continue
                        if owner.current_command in (s.forward, s.backward, s.steer_left, s.steer_right) and self.has_pending_actions():
                            time.sleep(0.01)
                            continue
                        if self.forward_like_refresh_due(owner.current_command, force=action_from_queue):
                            owner._last_motor_dispatch_source = (
                                "action_queue" if action_from_queue else "keepalive_refresh"
                            )
                            self.send_robot_command(owner.current_command)

                time.sleep(0.01)
            except Exception as exc:
                self.logger.error("动作执行异常: %s", exc)
                with owner.command_lock:
                    owner.current_command = None
                    owner.command_start_time = None
                if not self._service_motion_write_fault():
                    self.send_stop_with_brake_hold("action_executor_exception")

    def _service_runtime_shutdown(self) -> bool:
        """Signal/main fault flags preempt motion even while vision is blocked."""
        if not getattr(self.owner, "_runtime_shutdown_requested", False):
            return False
        now = time.monotonic()
        if now - getattr(self, "_shutdown_stop_attempt_at", -float("inf")) >= max(
                .05, getattr(self.config, "brake_hold_refresh_interval_sec", 1.2)):
            self._shutdown_stop_attempt_at = now
            try:
                self.send_stop_with_brake_hold("runtime_shutdown")
            except Exception:
                self.logger.exception("runtime shutdown STOP failed; motion remains prohibited")
        return True

    def _service_motion_write_fault(self) -> bool:
        """A backend half-write fault cannot be cleared by a new visual target."""
        sync_fault = getattr(self.backend, "sync_transaction_fault", None)
        if callable(sync_fault):
            sync_fault()
        fault = getattr(self.backend, "motion_write_fault", None)
        if not fault:
            return False
        owner = self.owner
        owner._runtime_shutdown_requested = owner._explicit_stop_requested = True
        owner.running = False
        owner._brake_hold_active = True
        owner._brake_hold_stop_mode = "emergency"
        owner._brake_hold_label = "safety_hold_motion_write_fault"
        owner._current_forward_percent = owner._current_steer_base_percent = 0
        owner.is_forwarding = False
        owner.current_command = owner.command_start_time = None
        owner._use_soft_stop_next = owner._soft_stop_active = False
        self._follow_wheel_clock.reset()
        # The backend already attempted zero/STOP at the failing write. Retry
        # STOP-only here, at a bounded rate, including after shutdown is set.
        now = time.monotonic()
        if now - getattr(self, "_motion_fault_stop_attempt_at", -float("inf")) >= max(
                .05, getattr(self.config, "brake_hold_refresh_interval_sec", 1.2)):
            self._motion_fault_stop_attempt_at = now
            self.logger.error("motion_executor_fault reason=%s recovery=manual_restart", fault)
            try:
                with owner.motor_io_lock:
                    self.backend.send_stop("safety_motion_write_fault", mode="emergency")
            except Exception:
                self.logger.exception("motor fault STOP retry failed; motion remains prohibited")
        return True

    def note_rotate_pulse_target_sent(self, action: int) -> bool:
        owner = self.owner
        c = self.config
        s = self.symbols
        if not (self.rotate_pulse_enabled(action) and owner.current_command == action):
            return False
        if owner.command_start_time is None:
            owner.command_start_time = time.time()
            owner._last_rotate_pulse_refresh_ts = owner.command_start_time
            self.logger.info(
                "旋转脉冲计时开始: 动作=%s 电机下发后时间戳=%.3f 持续=%.3f秒 停顿=%.3f秒 最少观察帧=%d",
                s.action_names.get(action, str(action)),
                owner.command_start_time,
                c.rotate_duration,
                c.rotate_pulse_pause_sec,
                c.rotate_pulse_observe_min_frames,
            )
            return True

        owner._last_rotate_pulse_refresh_ts = time.time()
        self.logger.info(
            "旋转脉冲目标已刷新: 动作=%s 已持续=%.3f秒 设定时长=%.3f秒",
            s.action_names.get(action, str(action)),
            owner._last_rotate_pulse_refresh_ts - owner.command_start_time,
            c.rotate_duration,
        )
        return False

    def rotate_hold_age_sec(self) -> float:
        owner = self.owner
        command_start_ts = float(getattr(owner, "command_start_time", 0.0) or 0.0)
        visual_refresh_ts = float(getattr(owner, "_last_rotate_visual_refresh_ts", 0.0) or 0.0)
        return max(0.0, time.time() - max(command_start_ts, visual_refresh_ts))

    def forward_like_refresh_due(self, action: int, *, force: bool = False) -> bool:
        owner = self.owner
        s = self.symbols
        if action == s.stop:
            # A queued STOP is dispatched once through force=True. Replaying a
            # latched zero target every 10 ms monopolizes the shared RS485 bus
            # and delays the next fresh steering command. Brake hold has its
            # own low-rate refresh path, so no STOP keepalive is needed here.
            return bool(force)
        if action in (s.rotate_left, s.rotate_right) and not self.rotate_pulse_enabled(action):
            now_ts = time.time()
            refresh_sec = max(0.02, float(self.config.motor_rs485_target_min_interval_sec))
            last_ts = float(getattr(owner, "_last_rotate_hold_target_send_ts", 0.0))
            if force or now_ts - last_ts >= refresh_sec:
                owner._last_rotate_hold_target_send_ts = now_ts
                return True
            return False
        if action not in (s.forward, s.backward, s.steer_left, s.steer_right):
            return True
        now_ts = time.time()
        if force:
            owner._last_forward_like_refresh_ts = now_ts
            return True
        if (getattr(self, "_visible_wheel_waiting", False)
                and self._visible_wheel_control_active()):
            feedback = self.get_steering_feedback()
            last_ts = float(getattr(owner, "_last_forward_like_refresh_ts", 0.0))
            interval = max(0.02, float(self.config.motor_rs485_target_min_interval_sec))
            if (feedback is not None and feedback.trustworthy
                    and math.isfinite(feedback.timestamp)
                    and feedback.timestamp > getattr(self, "_visible_wheel_feedback_ts", 0.0)
                    and now_ts - last_ts >= interval):
                # Retry the current command, never a saved pre-brake wheel
                # pair. The write path rechecks UID, yaw version and Depth TTL.
                owner._last_forward_like_refresh_ts = now_ts
                return True
        # Some MSSD target-mode setups decay if targets are not refreshed.  Keep
        # this separate from rotate pulse timing so forward/steer can be tuned
        # without changing TURN refresh behavior.
        keepalive_sec = float(self.config.motor_forward_like_keepalive_sec)
        if keepalive_sec <= 0:
            keepalive_sec = max(1.0, float(self.config.motor_rs485_target_min_interval_sec) * 20.0)
        keepalive_sec = max(0.02, keepalive_sec)
        last_ts = float(getattr(owner, "_last_forward_like_refresh_ts", 0.0))
        if now_ts - last_ts >= keepalive_sec:
            owner._last_forward_like_refresh_ts = now_ts
            self.logger.debug(
                "前进类动作保活刷新: 动作=%s 刷新间隔=%.3f秒 已间隔=%.3f秒",
                s.action_names.get(action, str(action)),
                keepalive_sec,
                now_ts - last_ts,
            )
            return True
        return False

    def log_motor_dispatch_timing(self, action: int, label: str, send_start_ts: float) -> None:
        owner = self.owner
        s = self.symbols
        now_ts = time.monotonic()
        command = getattr(self._dispatch_context, "command", None)
        timing_source = str(getattr(owner, "_last_motor_dispatch_source", "action_queue"))
        enqueue_ts = float(getattr(owner, "_last_action_queue_replace_ts", 0.0))
        enqueue_frame = int(getattr(owner, "_last_action_queue_replace_frame", -1))
        if timing_source != "action_queue" and action != s.stop:
            # Keepalive refresh is not a new queue item; do not report its age as queue wait.
            enqueue_ts = now_ts
            enqueue_frame = -1
        if action == s.stop:
            # STOP 通常由控制线程直接发送，不在 action_queue 中排队。此前沿用
            # 上一条运动命令的入队时间，造成日志显示数秒甚至十几秒的假延迟。
            direct_stop_ts = float(getattr(owner, "_last_stop_command_prepare_ts", 0.0))
            if direct_stop_ts > 0.0:
                enqueue_ts = direct_stop_ts
                enqueue_frame = int(getattr(owner, "_last_stop_command_frame", -1))
                timing_source = "direct_stop"
        if command is not None:
            enqueue_ts, enqueue_frame = command.enqueued_at, command.control_frame
            timing_source = "command_snapshot"
        queue_to_dispatch_ms = (now_ts - enqueue_ts) * 1000.0 if enqueue_ts > 0 else -1.0
        last_dispatch_ts = float(getattr(owner, "_last_motor_dispatch_ts", 0.0))
        since_last_dispatch_ms = (now_ts - last_dispatch_ts) * 1000.0 if last_dispatch_ts > 0 else -1.0
        same_action_refresh_ms = (
            since_last_dispatch_ms
            if getattr(owner, "_last_motor_dispatch_action", None) == action
            else -1.0
        )
        owner._last_motor_dispatch_ts = now_ts
        owner._last_motor_dispatch_action = action
        if action not in (s.forward, s.steer_left, s.steer_right):
            self._forward_coast_snapshot = None
        self.logger.info(
            "电机下发时序: 序号=%d 动作=%s 标签=%s source_module=%s reason=%s 依据控制帧=%d 依据采集帧=%d 决策采集帧=%d 入队控制帧=%d 当前控制帧=%d 计时来源=%s 排队到下发=%.1f毫秒 发送耗时=%.1f毫秒 距上次下发=%.1f毫秒 同动作刷新间隔=%.1f毫秒",
            command.revision if command else int(getattr(owner, "_last_action_queue_seq", 0)),
            s.action_names.get(action, str(action)),
            label,
            command.source_module if command else str(getattr(owner, "_last_command_source_module", "unknown")),
            command.reason if command else getattr(owner, "_last_action_queue_reason", ""),
            command.control_frame if command else int(getattr(owner, "_last_command_control_frame", -1)),
            command.capture_frame_id if command else int(getattr(owner, "_last_command_capture_frame", -1)),
            command.capture_frame_id if command else int(getattr(owner, "_last_decision_capture_frame", -1)),
            enqueue_frame,
            int(getattr(owner, "frame_index", -1)),
            timing_source,
            queue_to_dispatch_ms,
            (now_ts - send_start_ts) * 1000.0,
            since_last_dispatch_ms,
            same_action_refresh_ms,
        )

    def needs_transition_stop(self, old_action: Optional[int], new_action: int) -> bool:
        s = self.symbols
        if old_action == new_action:
            return False
        continuous = s.forward_like_actions | s.rotate_actions
        if (old_action in continuous and new_action in continuous
                and self._visible_wheel_control_active()):
            # The final wheel writer gates real reversals using encoders.
            # Enum changes alone must not send emergency/reset commands.
            return False
        # Visible parked-target PID updates are continuous signed yaw targets.
        # Changing their sign is the braking command itself; inserting a
        # target-zero transition here creates a stop/restart oscillation.  Lost
        # target search pulses keep their normal stop behavior.  Use the
        # current decision reason as well as the pulse flag so a stale search
        # action cannot accidentally inherit the visible fast path.
        if (
            old_action in s.rotate_actions
            and new_action in s.rotate_actions
            and str(getattr(self.owner, "_last_control_decision_reason", ""))
            in {"near_distance_rotation_only"}
            and not self.rotate_pulse_enabled(new_action)
        ):
            return False
        # 前进/差速与倒车之间必须先双轮清零，禁止在轮子仍向前时直接反向。
        if s.backward in (old_action, new_action):
            return old_action in s.movement_actions and new_action in s.movement_actions
        if old_action in s.forward_like_actions and new_action in s.forward_like_actions:
            return False
        return old_action in s.movement_actions and new_action in s.movement_actions and (
            old_action in s.rotate_actions or new_action in s.rotate_actions
        )

    def send_motion_transition_stop(self, old_action: Optional[int], new_action: int) -> None:
        s = self.symbols
        old_name = s.action_names.get(old_action, str(old_action))
        new_name = s.action_names.get(new_action, str(new_action))
        if getattr(self.owner, "_vision_control_state", "") == "lost_confirming":
            feedback = self.get_steering_feedback()  # Existing cache; no serial I/O.
            self.logger.info("identity_unconfirmed_transition old=%s new=%s "
                "stop_mode=%s feedback_forward_rpm=%s identity_shortcut=False",
                old_name, new_name, self.config.motor_rs485_transition_stop_mode,
                None if feedback is None else (feedback.left_forward_rpm, feedback.right_forward_rpm))
        self.logger.info("运动模式切换急停: %s -> %s", old_name, new_name)
        self.send_transition_stop_sequence(f"transition_{old_name}_to_{new_name}")

    def _visible_wheel_control_active(self) -> bool:
        owner = self.owner
        enabled = getattr(owner, "_depth_longitudinal_authority_enabled", None)
        controller = getattr(owner, "_follow_controller", None)
        return bool(
            not getattr(self.backend, "motion_write_fault", None)
            and callable(enabled) and enabled()
            and getattr(self.config, "use_percent_speed", False)
            and self.config.motor_forward_max_target_rpm > 0
            and self.config.motor_steer_raw_target > 0
            and getattr(owner, "running", False)
            and getattr(owner, "search_state", None) == "none"
            and getattr(controller, "search_state", None) == "none"
            and getattr(controller, "active_target_id", None) is not None
            and (str(getattr(owner, "_vision_control_state", "")) in {
                "target_visible", "target_visible_depth_valid", "target_visible_depth_missing"
            } or (getattr(self.config, "follow_forward_loss_handoff_enable", False)
                  and getattr(owner, "_vision_control_state", "") == "target_visible_low_quality"))
            and not getattr(owner, "_explicit_stop_requested", False)
            and not getattr(owner, "_runtime_shutdown_requested", False)
            and not getattr(owner, "_brake_hold_active", False)
            and getattr(owner, "_near_yaw_park_request", None) is None
            and not self.config.rotation_only
            and not self.rotate_pulse_enabled(self.symbols.rotate_right)
            and callable(getattr(owner, "_fresh_depth_linear_snapshot", None))
            and callable(getattr(owner, "_has_fresh_lateral_yaw", None))
        )

    def _near_yaw_park_queue_action_allowed(self, action: int, generation: int) -> bool:
        """Recheck after command_lock; a popped action may predate arm/release.

        The producer changes the generation and clears the queue together
        under action_queue_lock. This guard also blocks stale pulse state
        transitions, not only their eventual wheel writes.
        """
        current = int(getattr(self.owner, "_near_yaw_park_generation", 0))
        pending = getattr(self.owner, "_near_yaw_park_request", None) is not None
        if generation == current and not pending:
            return True
        self.logger.info(
            "near_yaw_park_queue_veto action=%s popped_generation=%d current_generation=%d pending=%s",
            self.symbols.action_names.get(action, str(action)), generation, current, pending,
        )
        return False

    def _near_yaw_park_blocks_write(self, label: str) -> bool:
        """Check under motor_io_lock, including for zero-speed packets.

        NORMAL locks the encoder position; a subsequent zero-speed write
        exits that mode. Thus a stale zero is just as invalid as a stale turn.
        This check never releases a hold or performs nested motor I/O.
        """
        if (getattr(self.owner, "_runtime_shutdown_requested", False)
                or getattr(self.backend, "motion_write_fault", None)
                or getattr(self.backend, "parking_release_fault", None)):
            return True
        if not self._command_revision_write_allowed(label):
            return True
        if self._search_reacquire_brake_request is not None:
            return True
        request = getattr(self.owner, "_near_yaw_park_request", None)
        if request is None:
            return False
        now = time.monotonic()
        if now - getattr(self, "_near_yaw_park_last_veto_log", -float("inf")) >= .20:
            self._near_yaw_park_last_veto_log = now
            self.logger.info(
                "near_yaw_park_write_blocked capture_frame_id=%s uid=%s label=%s reason=%s",
                request.capture_frame_id, request.uid, label, request.reason,
            )
        return True

    def _command_revision_write_allowed(self, label: str) -> bool:
        command = getattr(self._dispatch_context, "command", None)
        if command is None or command.protected_stop:
            return True
        revision = int(getattr(self.owner, "_action_command_revision", 0))
        if command.revision == revision:
            return True
        self.logger.info(
            "action_revision_veto label=%s command_revision=%d current_revision=%d "
            "capture_frame_id=%d reason=%s", label, command.revision, revision,
            command.capture_frame_id, command.reason,
        )
        return False

    def request_search_reacquire_brake(self, capture_id, capture_timestamp, reason):
        """Publish a brake intent. Only the execution thread writes NORMAL."""
        owner = self.owner
        with owner.command_lock, owner.motor_io_lock, owner.action_queue_lock:
            if (getattr(owner, "_explicit_stop_requested", False)
                    or getattr(owner, "_runtime_shutdown_requested", False)
                    or getattr(owner, "_near_yaw_park_request", None) is not None
                    or str(getattr(owner, "_brake_hold_label", "")).startswith("safety")):
                return False
            if self._search_reacquire_brake_request is not None:
                return True
            self._search_reacquire_brake_request = SearchReacquireBrakeRequest(
                int(capture_id), float(capture_timestamp), time.monotonic(), str(reason))
            self._search_reacquire_brake_uid = getattr(
                getattr(owner, "_follow_controller", None), "active_target_id", None)
            self._search_reacquire_brake_applied = None
            self._search_reacquire_settling = None
            owner._action_command_revision = int(getattr(owner, "_action_command_revision", 0)) + 1
            owner._near_yaw_park_generation = int(getattr(owner, "_near_yaw_park_generation", 0)) + 1
            while True:
                try:
                    owner.action_queue.get_nowait()
                except queue.Empty:
                    break
            owner.current_command = None
            owner.command_start_time = None
        self.logger.info("search_reacquire_brake_requested capture_frame_id=%s reason=%s",
                         capture_id, reason)
        return True

    def search_reacquire_brake_pending(self, *, capture_timestamp=None, keep_observing=False):
        """Only a post-settle image can release this hold; polling cannot release it."""
        with self.owner.motor_io_lock:
            request = self._search_reacquire_brake_request
            if request is None:
                return False
            if (getattr(self.owner, "_explicit_stop_requested", False)
                    or getattr(self.owner, "_runtime_shutdown_requested", False)
                    or getattr(self.owner, "_near_yaw_park_request", None) is not None
                    or str(getattr(self.owner, "_brake_hold_label", "")).startswith("safety")):
                # A higher-priority owner replaces this episode. Never leave
                # an unapplied request preventing safety/identity processing.
                self._search_reacquire_brake_request = None
                return False
            if self._search_reacquire_brake_applied is not request:
                return True
            feedback = self.get_steering_feedback()  # cached, no serial read
            now = time.monotonic()
            settling = self._search_reacquire_settling
            if settling is None:
                return True
            stamp = capture_timestamp if isinstance(capture_timestamp, (int, float)) else float("nan")
            ready = settling.release_ready(stamp, feedback, now)
            fresh_image = bool(math.isfinite(stamp) and 0 <= now-stamp <= .25)
            if capture_timestamp is not None:
                self.logger.info("search_brake_release_check cap=%d image_ts=%s reason=%s "
                                 "fresh_image=%s candidate_hold=%s ready_at=%s motion_authorized=False",
                                 request.capture_frame_id, capture_timestamp, settling.reason,
                                 fresh_image, keep_observing, settling.ready_at)
            if not ready or not fresh_image or keep_observing:
                return True
            self._search_reacquire_brake_request = None
            # Retire work published during settling. Leave the software hold
            # until the executor adopts a NEW decision and releases the hold;
            # merely observing quiet encoders must not revive cached axes.
            self.owner._action_command_revision = int(getattr(self.owner, "_action_command_revision", 0)) + 1
            with self.owner.action_queue_lock:
                protected = []
                while True:
                    try:
                        item = self.owner.action_queue.get_nowait()
                        if isinstance(item, ActionCommandSnapshot) and item.protected_stop:
                            protected.append(item)
                    except queue.Empty:
                        break
                for item in protected:
                    self.owner.action_queue.put_nowait(item)
            self.logger.info("search_reacquire_brake_settled capture_frame_id=%d elapsed_ms=%.1f "
                             "stop_sent_ts=%.6f quiet_ts=%.6f release_image_ts=%.6f "
                             "old_authority_restored=False", request.capture_frame_id,
                             (now-request.requested_at)*1000, settling.sent_at, settling.ready_at, stamp)
            return False

    def release_settled_search_brake_for_depth(self, *, uid, capture_id,
                                             capture_timestamp, sample_timestamp,
                                             depth_max_age, image_max_age):
        """Qualified producer handoff, not a queued command or a motor write.

        Caller has checked identity, fresh associated depth and forward demand.
        Quiet completion alone cannot unlock this hold. All old axes/packets
        are retired; the caller must compute and admit a NEW distance grant.
        """
        owner = self.owner
        with owner.command_lock, owner.motor_io_lock:
            now = time.monotonic()
            request = self._search_reacquire_brake_applied
            settling = self._search_reacquire_settling
            reason = "ready"
            if (not getattr(owner, "_brake_hold_active", False)
                    or getattr(owner, "_brake_hold_label", "") != "search_reacquire_brake"
                    or getattr(owner, "_brake_hold_stop_mode", None) != self._ordinary_park_stop_mode()):
                reason = "not_search_hold"
            elif (self._search_reacquire_brake_request is not None or request is None
                  or settling is None or settling.request is not request):
                reason = "not_settled"
            elif (uid is None or uid != self._search_reacquire_brake_uid
                  or uid != getattr(owner._follow_controller, "active_target_id", None)):
                reason = "identity_changed"
            elif (getattr(owner, "_explicit_stop_requested", False)
                  or getattr(owner, "_runtime_shutdown_requested", False)
                  or not owner.running or owner.search_state != "none"
                  or getattr(owner._follow_controller, "search_state", "none") != "none"
                  or getattr(owner, "_near_yaw_park_request", None) is not None):
                reason = "state_changed"
            elif (capture_id <= request.capture_frame_id
                  or not all(math.isfinite(v) for v in (capture_timestamp, sample_timestamp))
                  or not 0 <= now-capture_timestamp <= image_max_age
                  or not 0 <= now-sample_timestamp <= depth_max_age
                  or sample_timestamp <= settling.sent_at):
                reason = "stale_or_pre_stop_evidence"
            elif not settling.release_ready(capture_timestamp, self.get_steering_feedback(), now):
                reason = settling.reason
            else:
                try:
                    if self.hard_stop_check(self.symbols.forward):
                        reason = "hard_stop"
                except Exception:
                    reason = "hard_stop_check_failed"
            if reason != "ready":
                self.logger.info("search_brake_depth_resume_rejected cap=%s uid=%s reason=%s",
                                 capture_id, uid, reason)
                return False
            with owner.action_queue_lock:
                # Do not discard or bypass a queued safety/explicit STOP.
                with owner.action_queue.mutex:
                    if any(isinstance(p, ActionCommandSnapshot) and p.protected_stop
                           for p in owner.action_queue.queue):
                        return False
                while True:
                    try:
                        owner.action_queue.get_nowait()
                    except queue.Empty:
                        break
            owner._action_command_revision = int(getattr(owner, "_action_command_revision", 0)) + 1
            owner._near_yaw_park_generation = int(getattr(owner, "_near_yaw_park_generation", 0)) + 1
            owner._lateral_yaw_revision = int(getattr(owner, "_lateral_yaw_revision", 0)) + 1
            owner.current_command = owner.command_start_time = None
            store = getattr(owner, "_lateral_intent_store", None)
            if store is not None:
                store.clear()
            owner._depth30_linear_snapshot = owner._depth30_linear_timing = None
            owner._current_forward_percent = owner._current_steer_base_percent = 0
            owner._current_steer_correction_rpm = owner._current_rotate_raw_target = 0
            owner._current_rotate_pulse_enabled = False
            owner._brake_hold_active = False
            owner._brake_hold_stop_mode = None
            owner._brake_hold_label = "brake"
            owner._use_soft_stop_next = False
            owner.stop_action_execution = owner.person_detected_flag = False
            self._follow_wheel_clock.reset()
            self._visible_wheel_guard.reset()
            self._forward_loss_handoff.reset()
            self._search_reacquire_brake_applied = None
            self._search_reacquire_brake_uid = None
            self.logger.info("search_brake_depth_resumed cap=%s uid=%s depth_ts=%.6f "
                             "stop_sent_ts=%.6f quiet_ts=%.6f next=new_depth_admission "
                             "old_authority_restored=False", capture_id, uid, sample_timestamp,
                             settling.sent_at, settling.ready_at)
            return True

    def _ordinary_park_stop_mode(self):
        """Wire mode does not determine ownership: an ordinary hold can use EMERGENCY."""
        mode = getattr(getattr(self.backend, "config", None), "stop_mode", "normal")
        return "emergency" if str(mode).strip().lower() == "emergency" else "normal"

    def _service_search_reacquire_brake(self):
        request = self._search_reacquire_brake_request
        if request is None:
            return False
        if self.hard_stop_check(getattr(self.owner, "current_command", None)):
            self.send_stop_with_brake_hold("hard_stop")
            return True
        if self._search_reacquire_brake_applied is request:
            # Gather quiet evidence at the existing motor tick, rather than
            # waiting for two slow vision completions. No new serial reads.
            with self.owner.motor_io_lock:
                settling = self._search_reacquire_settling
                if request is not self._search_reacquire_brake_request:
                    return True
                if settling is not None:
                    self._service_ordinary_park_exit(settling, "search_reacquire_brake")
                    feedback = self.get_steering_feedback()
                    observed_at = time.monotonic()
                    ready = settling.observe(feedback, observed_at)
                    if (feedback is not None and math.isfinite(feedback.timestamp)
                            and feedback.timestamp > settling.sent_at
                            and feedback.timestamp > getattr(settling, "response_sample_ts", 0.)
                            and observed_at-getattr(settling, "response_log_at", 0.) >= .10):
                        settling.response_sample_ts = feedback.timestamp
                        settling.response_log_at = observed_at
                        self.logger.info(
                            "search_brake_response_sample cap=%d elapsed_ms=%.1f "
                            "feedback_ts=%.6f feedback_age_ms=%.1f left_rpm=%.1f right_rpm=%.1f "
                            "raw_yaw_dps=%s filtered_yaw_dps=%s trustworthy=%s quiet_count=%d reason=%s",
                            request.capture_frame_id, (feedback.timestamp-settling.sent_at)*1000,
                            feedback.timestamp, (observed_at-feedback.timestamp)*1000,
                            feedback.left_forward_rpm, feedback.right_forward_rpm,
                            getattr(feedback, "raw_yaw_rate_right_dps", None),
                            getattr(feedback, "yaw_rate_right_dps", None),
                            feedback.trustworthy, settling.quiet_count, settling.reason)
                    if ready and not getattr(settling, "response_logged", False):
                        settling.response_logged = True
                        self.logger.info("search_brake_stop_response cap=%d stop_sent_ts=%.6f "
                                         "quiet_ts=%.6f stop_to_quiet_ms=%.1f",
                                         request.capture_frame_id, settling.sent_at, settling.ready_at,
                                         (settling.ready_at-settling.sent_at)*1000)
            return True
        self.cancel_yaw_pulses("search_reacquire_brake", send_zero=False)
        owner = self.owner
        with owner.motor_io_lock:
            if request is not self._search_reacquire_brake_request:
                return False
            if (getattr(owner, "_explicit_stop_requested", False)
                    or getattr(owner, "_runtime_shutdown_requested", False)
                    or str(getattr(owner, "_brake_hold_label", "")).startswith("safety")
                    or getattr(owner, "_near_yaw_park_request", None) is not None):
                return True
            self._follow_wheel_clock.reset()
            self._visible_wheel_guard.reset()
            owner._brake_hold_active = True
            owner._brake_hold_stop_mode = self._ordinary_park_stop_mode()
            owner._brake_hold_label = "search_reacquire_brake"
            owner._use_soft_stop_next = owner._soft_stop_active = False
            self.backend.send_stop("search_reacquire_brake", mode=owner._brake_hold_stop_mode,
                                   prepare_parking_current=True)
            self._search_reacquire_brake_sent_at = time.monotonic()
            self._search_brake_dispatch_delay_sec = min(.15, max(.05,
                self._search_reacquire_brake_sent_at-request.requested_at))
            self._search_reacquire_settling = ParkSettlingEvidence(
                request, self._search_reacquire_brake_sent_at, require_current_release=True)
            owner._last_brake_hold_send_ts = time.time()
            self._search_reacquire_brake_applied = request
            self.logger.info("search_reacquire_brake_applied capture_frame_id=%d mode=%s minimum_hold_ms=%.0f "
                             "request_age_ms=%.1f capture_to_stop_ms=%.1f next_dispatch_allowance_ms=%.1f",
                             request.capture_frame_id, owner._brake_hold_stop_mode, ParkSettlingEvidence.MIN_HOLD_SEC * 1000,
                             (self._search_reacquire_brake_sent_at-request.requested_at)*1000,
                             (self._search_reacquire_brake_sent_at-request.capture_timestamp)*1000,
                             self._search_brake_dispatch_delay_sec*1000)
        return True

    def _service_near_yaw_park(self) -> bool:
        """Consume an explicit yaw-parking intent once, never a generic zero.

        The producer owns intent validity and release. This executor merely
        applies the configured stop once, ends its bounded hold independently,
        and retains a software hold against queued/periodic motion writes.
        """
        owner = self.owner
        if self._service_motion_write_fault():
            return True
        if getattr(self.backend, "parking_release_fault", None):
            return True  # Latched until runtime restart; new targets cannot clear it.
        request = getattr(owner, "_near_yaw_park_request", None)
        if request is None:
            self._near_yaw_park_applied = None
            self._near_yaw_park_settling = None
            return self._service_search_reacquire_brake()
        self._search_reacquire_brake_request = None  # superseded, never replay after near park releases
        if self.hard_stop_check(getattr(owner, "current_command", None)):
            self.send_stop_with_brake_hold("hard_stop")
            self._near_yaw_park_applied = request
            return True
        if self._near_yaw_park_applied is request:
            with owner.motor_io_lock:
                evidence = self._near_yaw_park_settling
                if evidence is not None and evidence.request is request:
                    self._service_ordinary_park_exit(evidence, "near_yaw_park")
                    evidence.observe(self.get_steering_feedback(), time.monotonic())
                    self._log_near_yaw_stop_response(evidence)
            return True
        self.cancel_yaw_pulses("near_yaw_park", send_zero=False)
        with owner.motor_io_lock:
            if getattr(owner, "_near_yaw_park_request", None) is not request:
                return False
            self._follow_wheel_clock.reset()
            self._visible_wheel_guard.reset()
            self._visible_wheel_waiting = False
            owner._use_soft_stop_next = False
            owner._soft_stop_active = False
            owner._follow_distance_hold = None
            # A concurrently established safety hold has priority. Never
            # downgrade emergency/explicit stops to the normal yaw park.
            protected_hold = bool(
                (getattr(owner, "_brake_hold_active", False)
                 and ((getattr(owner, "_brake_hold_stop_mode", None) not in (None, "normal")
                       and not (getattr(owner, "_brake_hold_label", "") in
                                ("near_yaw_park", "search_reacquire_brake")
                                and getattr(owner, "_brake_hold_stop_mode", None) == self._ordinary_park_stop_mode()))
                      or str(getattr(owner, "_brake_hold_label", "")).startswith("safety")))
                or getattr(owner, "_explicit_stop_requested", False)
                or getattr(owner, "_runtime_shutdown_requested", False)
            )
            if (protected_hold
                    and getattr(self, "_predictive_turn_brake_request", None) is request
                    and getattr(self, "_predictive_turn_brake_started", None) is not None):
                self._stop_predictive_turn_brake_emergency(request, "predictive_countersteer_preempted")
                return True
            if not protected_hold:
                if self._service_predictive_turn_brake(request):
                    return True
                if getattr(owner, "_near_yaw_park_request", None) is not request:
                    return False
                owner._brake_hold_active = True
                owner._brake_hold_stop_mode = self._ordinary_park_stop_mode()
                owner._brake_hold_label = "near_yaw_park"
                owner._last_stop_command_prepare_ts = time.monotonic()
                owner._last_stop_command_frame = int(getattr(owner, "frame_index", -1))
                owner._last_stop_command_reason = "near_yaw_park:" + request.reason
                self.backend.send_stop("near_yaw_park", mode=owner._brake_hold_stop_mode,
                                       prepare_parking_current=True)
                self._near_yaw_park_settling = ParkSettlingEvidence(
                    request, time.monotonic(), require_current_release=True)
                owner._last_brake_hold_send_ts = time.time()
                self._forward_coast_snapshot = None
                self.logger.info(
                    "near_yaw_park_applied capture_frame_id=%s uid=%s reason=%s "
                    "mode=%s minimum_hold_ms=%.0f request_age_ms=%.1f stop_sent_ts=%.6f",
                    request.capture_frame_id, request.uid, request.reason, owner._brake_hold_stop_mode,
                    ParkSettlingEvidence.MIN_HOLD_SEC * 1000,
                    max(0., time.monotonic() - request.requested_at) * 1000.,
                    self._near_yaw_park_settling.sent_at,
                )
            self._near_yaw_park_applied = request
        return True

    def _ordinary_park_exit_fault(self, evidence, reason):
        """Motor lock already held. Persist independently of the visual request."""
        if getattr(self.backend, "parking_release_fault", None):
            return
        evidence.fault = evidence.reason = reason
        self.backend.parking_release_fault = reason
        self.backend.motion_armed = False
        self.owner._brake_hold_active = True
        self.owner._brake_hold_stop_mode = "emergency"
        self.owner._brake_hold_label = "safety_hold_parking_release_fault"
        self.logger.error("ordinary_park_exit_fault cap=%s reason=%s recovery=manual_restart "
                          "motion_authorized=False", evidence.request.capture_frame_id, reason)
        if getattr(self.backend, "motion_write_fault", None):
            # A failed FREE/NORMAL transaction already attempted both emergency
            # STOPs in the backend. The executor's fault service owns bounded
            # retries; do not burst-repeat them at this same exception boundary.
            return
        try:
            self.backend.send_stop("parking_release_fault", mode="emergency")
        except Exception:
            self.logger.exception("parking_release_fault emergency write failed; motion remains blocked")

    def _service_ordinary_park_exit(self, evidence, label):
        """Motor-lock protected bounded current hold, independent of motion release.

        Safety/explicit stops are never time-released. This ordinary episode
        clears current, then sends one FREE STOP, never a zero-speed packet.
        """
        owner = self.owner
        def owns_hold():
            request = (getattr(owner, "_near_yaw_park_request", None) if label == "near_yaw_park"
                       else self._search_reacquire_brake_request)
            return (request is evidence.request
                    and getattr(owner, "_brake_hold_label", "") == label
                    and getattr(owner, "_brake_hold_stop_mode", None) == self._ordinary_park_stop_mode()
                    and not getattr(owner, "_explicit_stop_requested", False)
                    and not getattr(owner, "_runtime_shutdown_requested", False))
        if evidence.fault or not owns_hold():
            return
        now = time.monotonic()
        early_forward = (label == "near_yaw_park" and evidence.forward_resume_live(now)
                         and getattr(owner, "search_state", "none") == "none"
                         and getattr(owner, "running", False)
                         and getattr(owner, "_vision_control_state", "") in {
                             "target_visible", "target_visible_depth_valid"}
                         and owner._follow_controller.search_state == "none"
                         and owner._follow_controller.active_target_id == evidence.request.uid)
        if not early_forward and not evidence.minimum_hold_complete(now):
            return
        if evidence.current_released_at is None:
            failure_reason = "current_release_failed"
            try:
                if self.hard_stop_check(getattr(owner, "current_command", None)):
                    owner._brake_hold_active = True
                    owner._brake_hold_stop_mode = "emergency"
                    owner._brake_hold_label = "safety_hold_hard_stop"
                    self.backend.send_stop("hard_stop", mode="emergency")
                    return  # In particular, a refresh may not clear 5A first.
                self.backend.release_parking_current_only()  # both 0A readbacks required; no speed write
                if (not owns_hold()
                        or self.hard_stop_check(getattr(owner, "current_command", None))):
                    # Authority/safety changed during serial I/O. A protected
                    # emergency owns the stop; no motion may follow the clear.
                    self._ordinary_park_exit_fault(evidence, "authority_changed_during_current_release")
                    return
                failure_reason = "free_stop_failed"
                self.backend.send_stop("ordinary_park_release_free", mode="free", preserve_zero=True)
                if (not owns_hold()
                        or self.hard_stop_check(getattr(owner, "current_command", None))):
                    self._ordinary_park_exit_fault(evidence, "authority_changed_during_free_stop")
                    return
                # Completion of both FREE writes, not merely the 0A readback,
                # starts the fresh feedback/image boundary. The ordinary hold
                # remains latched; its periodic service must not replay FREE.
                evidence.mark_current_released(time.monotonic())
                self.logger.info("ordinary_park_current_released cap=%s label=%s hold_ms=%.1f "
                                 "current_a=0 zero_rpm=False speed_write=False motion_authorized=False "
                                 "exit_stop_mode=free",
                                 evidence.request.capture_frame_id, label,
                                 (evidence.current_released_at-evidence.sent_at)*1000)
            except Exception:
                self.logger.exception("ordinary park current/free-stop release failed")
                self._ordinary_park_exit_fault(evidence, failure_reason)
            return
        # Slow settling alone is not a latched motor fault. Keep collecting
        # fresh evidence so existing release gates can recover after a long
        # coast; elapsed time never grants motion or replays an old command.
        # Current/STOP I/O failures and safety changes still fault above.
        evidence.observe(self.get_steering_feedback(), now)

    def _stop_predictive_turn_brake_emergency(self, request, reason):
        # Caller already holds the NON-reentrant motor lock. Do not call
        # send_stop_with_brake_hold here (it would acquire that lock again).
        self.owner._brake_hold_active = True
        self.owner._brake_hold_label = "safety_hold_" + reason
        self.owner._brake_hold_stop_mode = "emergency"
        self.backend.send_stop(reason, mode="emergency")
        self._near_yaw_park_applied = request

    def _service_predictive_turn_brake(self, request):
        """Motor-lock protected phase of a latched STOP, not a move command.

        The normal wheel reversal guard is unchanged. Only this evidence-
        bounded braking pulse can oppose existing pivot motion. All ordinary
        writes remain blocked by the park request until NORMAL and release.
        """
        if not request.countersteer_rpm:
            return False
        if getattr(self, "_predictive_turn_brake_failed", None) is request:
            return False
        owner = self.owner
        now = time.monotonic()
        started = getattr(self, "_predictive_turn_brake_started", None)
        if getattr(self, "_predictive_turn_brake_request", None) is not request:
            self._predictive_turn_brake_request = request
            self._predictive_turn_brake_started = started = None
            self._predictive_turn_brake_feedback_ts = None
        def scope_ok():
            return bool(
                getattr(owner._follow_controller, "active_target_id", None) == request.uid
                and getattr(owner, "search_state", "none") == "none"
                and getattr(owner, "_vision_control_state", "").startswith("target_visible")
                and owner.running and not getattr(owner, "_explicit_stop_requested", False)
                and not getattr(owner, "_runtime_shutdown_requested", False)
                and (started is not None or not getattr(owner, "_brake_hold_active", False)))
        def feedback_ok():
            feedback = self.get_steering_feedback()
            if not pulse_feedback_qualified(request, feedback, time.monotonic()):
                return False
            previous = self._predictive_turn_brake_feedback_ts
            if previous is not None and feedback.timestamp < previous:
                return False
            self._predictive_turn_brake_feedback_ts = feedback.timestamp
            return True
        qualified = scope_ok() and feedback_ok()
        if started is not None:
            if qualified and now < started + MAX_PULSE_SEC:
                return True  # Keep the one bounded packet; never renew its timer.
            self.logger.info("predictive_turn_brake_end cap=%s elapsed_ms=%.1f reason=%s",
                             request.capture_frame_id, (now-started)*1000,
                             "deadline" if qualified else "evidence_or_motion_changed")
            return False
        if not qualified:
            self.logger.info("predictive_turn_brake_skipped cap=%s fallback=%s",
                             request.capture_frame_id, self._ordinary_park_stop_mode())
            return False
        self.backend.prepare_speed_mode()
        # Register I/O can consume the lease. Check it again, and never
        # replace a new stop owner or continue through a hazard.
        now = time.monotonic()
        if self.hard_stop_check(getattr(owner, "current_command", None)):
            self._stop_predictive_turn_brake_emergency(request, "predictive_countersteer_hard_stop")
            return True
        if (getattr(owner, "_near_yaw_park_request", None) is not request
                or not scope_ok()
                or not feedback_ok()):
            return False
        correction = request.countersteer_rpm
        ls = self.backend.wheel_raw_state_to_target("left", 1, 0x01)
        rs = self.backend.wheel_raw_state_to_target("right", 1, 0x01)
        # A NORMAL request already retired queued work. No speed-mode action
        # is allowed to override this request during the pulse.
        try:
            self.backend.send_targets(correction*ls, -correction*rs, "PREDICTIVE_COUNTERSTEER")
        except Exception:
            # A partial write may have powered one wheel. Never retry this
            # pulse; stop immediately and let the normal stop path recover.
            self._predictive_turn_brake_failed = request
            self._stop_predictive_turn_brake_emergency(request, "predictive_countersteer_write_failed")
            raise
        self._predictive_turn_brake_started = now
        self.logger.info("predictive_turn_brake_applied cap=%s uid=%s yaw_rpm=%s "
                         "max_duration_ms=80 sent_ts=%.6f fallback=%s",
                         request.capture_frame_id, request.uid, correction, now, self._ordinary_park_stop_mode())
        # A slow serial write must not buy another scheduler interval.
        if self.hard_stop_check(getattr(owner, "current_command", None)):
            self._stop_predictive_turn_brake_emergency(request, "predictive_countersteer_hard_stop")
            return True
        if time.monotonic() >= now + MAX_PULSE_SEC or not scope_ok() or not feedback_ok():
            return False
        return True

    def _log_near_yaw_stop_response(self, evidence):
        # Cache-only, once per parking request. Encoder quiet is not proof that
        # the camera/chassis has stopped vibrating; retain visual/IMU logs too.
        if evidence.ready_at is not None and not getattr(evidence, "response_logged", False):
            evidence.response_logged = True
            self.logger.info("near_yaw_stop_response uid=%d cap=%d stop_sent_ts=%.6f "
                "quiet_ts=%.6f stop_to_quiet_ms=%.1f source=encoder_not_body",
                evidence.request.uid, evidence.request.capture_frame_id,
                evidence.sent_at, evidence.ready_at, (evidence.ready_at-evidence.sent_at)*1000)

    def near_yaw_park_release_ready(self, request, capture_timestamp, now):
        """Caller holds motor lock; cache-only feedback, no serial read."""
        evidence = self._near_yaw_park_settling
        ready = bool(evidence is not None and evidence.request is request
                     and self._near_yaw_park_applied is request
                     and evidence.release_ready(capture_timestamp, self.get_steering_feedback(), now))
        if evidence is not None and evidence.request is request:
            self._log_near_yaw_stop_response(evidence)
        self.logger.info(
            "near_yaw_park_release_check capture_ts=%.6f stop_sent_ts=%s quiet_at=%s "
            "quiet_samples=%s ready=%s reason=%s current_released_ts=%s",
            capture_timestamp, None if evidence is None else evidence.sent_at,
            None if evidence is None else evidence.ready_at,
            0 if evidence is None else evidence.quiet_count, ready,
            "stop_not_applied" if evidence is None or evidence.request is not request else evidence.reason,
            None if evidence is None else evidence.current_released_at,
        )
        return ready

    def near_yaw_park_motion_ready(self, request, capture_timestamp, now):
        """Ordinary qualified motion: transfer residual motion to wheel guards."""
        evidence = self._near_yaw_park_settling
        ready = bool(evidence is not None and evidence.request is request
                     and self._near_yaw_park_applied is request
                     and evidence.motion_handoff_ready(
                         capture_timestamp, self.get_steering_feedback(), now))
        self.logger.info(
            "near_yaw_park_motion_check capture_ts=%.6f ready=%s reason=%s "
            "quiet_required=False wheel_reversal_guard=preserved",
            capture_timestamp, ready,
            evidence.reason if evidence is not None and evidence.request is request
            else "stop_not_applied")
        return ready

    def request_near_yaw_forward_resume(self, request, capture_timestamp, sample_timestamp, now):
        """Producer is under locks; cache a bounded hint, never do serial I/O."""
        evidence = self._near_yaw_park_settling
        accepted = bool(evidence is not None and evidence.request is request
                    and self._near_yaw_park_applied is request
                    and evidence.request_forward_resume(capture_timestamp, sample_timestamp, now))
        if accepted:
            self.logger.info("near_yaw_forward_resume_requested uid=%s capture_ts=%s sample_ts=%s "
                             "expires_at=%s motion_authorized=False", request.uid,
                             capture_timestamp, sample_timestamp, evidence.forward_resume_until)
        return accepted

    def _request_follow_axes_rebuild(self, uid):
        """Discard an unsent plan, never renew a grant; caller retries once."""
        latest = self.owner._follow_wheel_axes(time.monotonic())
        if (getattr(self, "_follow_axes_rebuild_allowed", False)
                and latest is not None and latest[0] == uid
                and (latest[2] > 0 or (latest[2] == 0 and latest[3] != 0
                     and self.owner._has_fresh_lateral_yaw(uid)))
                and self.owner._follow_controller.active_target_id == uid
                and self._periodic_follow_active()):
            self._follow_axes_rebuild_requested = True
            self.logger.info("follow_wheel_retry uid=%s reason=new_axes_before_write "
                             "old_packet_discarded=True immediate_rebuild=True", uid)
            return True
        return False

    def forward_execution_anchor(self, uid, sample_timestamp, now):
        """Read completed I/O only; cannot keep an old depth grant alive.

        Used solely before PI processes a NEW qualified depth sample. A
        read-time continuation veto is not proof that a zero packet was sent.
        Any intervening speed transaction/STOP invalidates the backend receipt.
        """
        anchor = getattr(self, "_forward_execution_anchor", None)
        if (anchor is None or anchor.uid != uid
                or anchor.sample_timestamp != sample_timestamp
                or getattr(self.backend, "last_speed_receipt", None) is not anchor.receipt
                or not self._visible_wheel_control_active()
                or getattr(self.owner, "_vision_control_state", "") not in {
                    "target_visible", "target_visible_depth_valid"}
                or getattr(self.owner, "stop_action_execution", False)
                or getattr(self.owner, "person_detected_flag", False)
                or getattr(self.backend, "parking_release_fault", None)
                or self.owner._follow_controller.active_target_id != uid
                or not math.isfinite(now)
                or not 0 <= now-anchor.sent_at <= .10
                or not 0 <= now-sample_timestamp <= .25):
            return None
        # Explicit revocation clears this canonical tuple even before the
        # executor can physically send zero. Never resurrect that grant.
        linear = getattr(self.owner, "_depth30_linear_snapshot", None)
        if (linear is None or linear[0] != "forward" or linear[1] <= 0
                or linear[2] != uid or linear[3] != sample_timestamp):
            return None
        if (getattr(self.backend, "last_speed_receipt", None) is not anchor.receipt
                or getattr(self, "_forward_execution_anchor", None) is not anchor):
            return None
        return anchor

    def _send_follow_wheel_targets(
        self, left, right, label, *, max_target_override=None, visible_required=False
    ):
        """Called under motor_io_lock after action/version revalidation."""
        if self._near_yaw_park_blocks_write(label):
            return False
        if self._periodic_follow_active() and not self._periodic_follow_writing:
            # Producers only update canonical axes. Do not remember this
            # packet: a later tick must reconstruct BOTH current axes.
            return False
        if not self._visible_wheel_control_active():
            self._visible_wheel_guard.reset()
            self._forward_loss_handoff.reset()
            self._visible_wheel_waiting = False
            if visible_required:
                self.logger.info("visible_wheel_revoked label=%s reason=visible_state_changed", label)
                return
            if max_target_override is None:
                self.backend.send_targets(left, right, label)
            else:
                self.backend.send_targets(left, right, label, max_target_override=max_target_override)
            return
        # Exiting ordinary parking is a hardware mode transition, even when
        # the next guarded pair is zero. Do it before measuring authority age.
        # Explicit preserved NORMAL cross-brakes remain held until nonzero.
        if not self.backend.normal_zero_hold:
            parking_exit = bool(getattr(self.backend, "parking_current_a", 0.0)
                or getattr(self.backend, "_parking_current_uncertain", False))
            self.backend.prepare_speed_mode()
            if parking_exit and self.hard_stop_check(getattr(self.owner, "current_command", None)):
                self.backend.send_stop("parking_exit_hard_stop", mode="emergency")
                return False
        now = time.monotonic()
        uid = self.owner._follow_controller.active_target_id
        if self._periodic_follow_writing and self._periodic_follow_axes[0] != uid:
            self.backend.send_targets(0, 0, "FOLLOW20_UID_CHANGED")
            self._visible_wheel_guard.reset()
            self._forward_loss_handoff.reset()
            return False
        revision = getattr(self.owner, "_lateral_yaw_revision", None)
        if getattr(self, "_visible_wheel_uid", None) != uid:
            self._visible_wheel_guard.reset()
            self._forward_loss_handoff.reset()
            self._visible_wheel_uid = uid
        ls = self.backend.wheel_raw_state_to_target("left", 1, 0x01)
        rs = self.backend.wheel_raw_state_to_target("right", 1, 0x01)
        forward = (left * ls, right * rs)
        base, yaw = .5 * sum(forward), .5 * (forward[0] - forward[1])
        # Revalidate physical Depth TTL at the actual write, not only at queue
        # creation. A yaw-only guard must never create positive translation.
        linear = self.owner._fresh_depth_linear_snapshot(uid, now=now)
        if getattr(self.owner, "_vision_control_state", "") == "target_visible_low_quality":
            # Share the wheel handoff, NOT identity/forward permission.
            linear = None
            base = 0.0
        if base > 0:
            base = min(base, linear[1] * self.config.motor_forward_max_target_rpm / 100.0) if linear and linear[0] == "forward" else 0.0
        if not self.owner._has_fresh_lateral_yaw(uid):
            yaw = 0.0
        limited_yaw = limit_handoff_yaw(self.owner, uid, yaw)
        if limited_yaw != yaw:
            self.logger.info("search_handoff_yaw_limited uid=%s label=%s "
                             "requested_yaw_rpm=%.1f applied_yaw_rpm=%.1f base_rpm=%.1f",
                             uid, label, yaw, limited_yaw, base)
        yaw = limited_yaw
        requested = (int(round(base + yaw)), int(round(base - yaw)))
        feedback = self.get_steering_feedback()
        moving_evidence = getattr(self.owner, "_search_handoff_moving_evidence", None)
        moving_handoff = bool(getattr(self.owner, "_search_handoff_uid", None) == uid
            and ((base > 0 and moving_evidence is not None)
                 or getattr(self.owner, "_search_handoff_moving_active", False)))
        if (self._periodic_follow_writing
                and self.owner._follow_wheel_axes(time.monotonic()) != self._periodic_follow_axes):
            # Do not let an already superseded pivot start a reversal wait
            # before rebuilding the newly authorized forward curve.
            if not self._request_follow_axes_rebuild(uid):
                self.backend.send_targets(0, 0, "FOLLOW20_AUTHORITY_CHANGED")
                self._visible_wheel_guard.reset()
            return False
        original_base, original_yaw = base, yaw
        response_phase = "disabled"
        response_intent = None
        response_adjusted = False
        cfg = getattr(self.owner._follow_controller, "cfg", None)
        image_mode = getattr(cfg, "visible_steering_pid_image_error_only", False)
        response_policy = getattr(self.owner, "_lateral_turn_response_policy", None)
        if getattr(self.config, "follow_turn_response_assist_enable", False):
            store = getattr(self.owner, "_lateral_intent_store", None)
            response_intent = store.snapshot() if store is not None else None
            permitted = not (response_intent is not None and response_policy is not None
                and response_policy[0] == response_intent.sequence and not response_policy[1])
            if image_mode:
                limit = effective_correction_limit(cfg, base,
                    near_distance=bool(response_intent and response_intent.near_distance_mode),
                    policy_limit=response_intent.correction_limit_rpm if response_intent else None)
                yaw = clamp_correction(yaw, limit)
                self._turn_buildup.sync_writer(getattr(self._visible_wheel_guard, "last_sent", 0.))
                response_input = (base, yaw)
                base, yaw, response_phase = self._turn_buildup.adjust(
                    base, yaw, response_intent, feedback, now, uid,
                    limit=limit, permitted=permitted)
            else:
                response_input = (base, yaw)
                base, yaw, response_phase = self._turn_response_assist.adjust(
                    base, yaw, response_intent, feedback, now, uid, boost_permitted=permitted)
            # CAP384: the ordinary mode clamp (-10 -> -7) is not an
            # assistance boost. Only actual assist changes require its
            # shorter evidence lease; otherwise a legal reduction is lost.
            response_adjusted = (base, yaw) != response_input
            yaw = limit_handoff_yaw(self.owner, uid, yaw)
        # The current physical axes own the mode, not a saved visual intent.
        # Enforce after temporary boost so it cannot exceed the same ceiling.
        cfg = getattr(self.owner._follow_controller, "cfg", None)
        if getattr(cfg, "visible_steering_pid_image_error_only", False):
            store = getattr(self.owner, "_lateral_intent_store", None)
            current_intent = store.snapshot() if store is not None else None
            matching = bool(current_intent is not None and current_intent.target_id == uid
                            and current_intent.valid(now))
            limit = effective_correction_limit(
                cfg, base, near_distance=matching and current_intent.near_distance_mode,
                policy_limit=current_intent.correction_limit_rpm if matching else None)
            recenter = getattr(self.owner._follow_controller, "post_park_recenter_limit", None)
            if base == 0 and callable(recenter):
                recenter_limit = recenter(uid)
                if recenter_limit is not None:
                    limit = min(limit, recenter_limit)
            bounded_yaw = clamp_correction(yaw, limit)
            if bounded_yaw != yaw:
                self.logger.info("effective_yaw_limit uid=%s base_rpm=%.1f requested_yaw=%.1f "
                                 "applied_yaw=%.1f limit=%.1f", uid, base, yaw, bounded_yaw, limit)
            yaw = bounded_yaw
            requested = (int(round(base+yaw)), int(round(base-yaw)))
        # Normal tracking and buildup preserve approved translation.
        # At wheel saturation reduce yaw headroom, not the approved base.
        if base > 0 and linear and linear[0] == "forward":
            cap = self.config.motor_forward_max_target_rpm
            backend_cap = (getattr(getattr(self.backend, "config", None), "max_target", cap)
                           if max_target_override is None else max_target_override)
            cap = min(cap, backend_cap)
            base = min(base, max(0., cap))
            yaw = math.copysign(min(abs(yaw), max(0., cap-base)), yaw)
            requested = (int(round(base+yaw)), int(round(base-yaw)))
        guarded_request, handoff_reason = requested, None
        priority_removed = max(0., original_base-base)
        priority_reason = response_phase if priority_removed else "forward_base_preserved"
        turn_bound = min(4.0, max(0.0, getattr(self.config, "follow_turn_residual_max_rpm", 0.0)))
        if (turn_bound > 0 and base > 0 and linear and linear[0] == "forward"
                and min(requested) < 0 and feedback is not None and feedback.trustworthy
                and all(math.isfinite(v) for v in (feedback.timestamp, feedback.left_forward_rpm,
                                                   feedback.right_forward_rpm))
                and 0 <= now-feedback.timestamp <= .15
                and min(feedback.left_forward_rpm, feedback.right_forward_rpm) >= -1
                and max(feedback.left_forward_rpm, feedback.right_forward_rpm) > 1):
            # CAP676: base 8 + yaw -15 must not force a new wheel reversal.
            # Preserve approved translation, shrink yaw to a non-reversing arc.
            # Never raise base to keep a larger differential.
            arc_yaw = math.copysign(min(abs(yaw), base), yaw)
            guarded_request = (int(round(base+arc_yaw)), int(round(base-arc_yaw)))
            handoff_reason = "forward_curve_no_reverse"
        loss_input = guarded_request
        trial_horizon = getattr(cfg, "visible_steering_pid_execution_response_trial_sec", 0.)
        trial = None
        trial_intent = None
        trial_phase = "disabled"
        if trial_horizon > 0 and getattr(cfg, "visible_steering_pid_image_error_only", False):
            trial = getattr(self, "_turn_response_trial", None)
            if trial is None:
                trial = self._turn_response_trial = TurnResponseTrial()
            store = getattr(self.owner, "_lateral_intent_store", None)
            trial_intent = store.snapshot() if store is not None else None
            trial_base, trial_yaw, trial_phase = trial.adjust(
                .5*sum(guarded_request), .5*(guarded_request[0]-guarded_request[1]),
                trial_intent, feedback, now, uid, trial_horizon)
            guarded_request = (int(round(trial_base+trial_yaw)), int(round(trial_base-trial_yaw)))
            if trial_phase == "forward_brake":
                handoff_reason = trial_phase
            audit_key = (uid, trial_phase)
            if (getattr(self, "_turn_trial_log_key", None) != audit_key
                    or (trial_phase != "tracking" and now-getattr(self, "_turn_trial_log_at", 0.) >= .5)):
                self.logger.info("turn_response_trial uid=%s horizon_ms=%.0f phase=%s input=%s output=%s",
                                 uid, trial_horizon*1000, trial_phase, loss_input, guarded_request)
                self._turn_trial_log_key, self._turn_trial_log_at = audit_key, now
        # Last yaw constraint: neither PID, buildup nor a trial may bypass
        # residual search rotation convergence after forward has resumed.
        moving_phase = "inactive"
        moving_input = guarded_request
        if moving_handoff:
            moving = moving_handoff_yaw(self.owner, evidence=moving_evidence, uid=uid,
                base=.5*sum(guarded_request), yaw=.5*(guarded_request[0]-guarded_request[1]),
                feedback=feedback, now=time.monotonic(), policy=cfg,
                execution_delay_sec=getattr(self, "_search_brake_dispatch_delay_sec", .05))
            quiet_straight = handoff_straight_allowed(
                self.owner, moving=moving, evidence=moving_evidence, uid=uid,
                feedback=feedback, now=time.monotonic())
            if (moving is None or not linear or linear[0] != "forward"
                    or (not quiet_straight and not self.owner._has_fresh_lateral_yaw(uid))):
                guarded_request = (0, 0)
                moving_phase = "evidence_unavailable"
            else:
                moving_base = .5*sum(guarded_request)
                guarded_request = (int(round(moving_base+moving[0])), int(round(moving_base-moving[0])))
                moving_phase = moving[1]
            self.logger.info("search_handoff_execution uid=%s cap=%s phase=%s "
                             "input=%s output=%s depth_renewed=False",
                             uid, getattr(moving_evidence, "cap", None), moving_phase,
                             loss_input, guarded_request)
        loss_input = guarded_request
        loss_reason = None
        if getattr(self.config, "follow_forward_loss_handoff_enable", False):
            guarded_request, loss_reason = self._forward_loss_handoff.limit(
                guarded_request, feedback, now, residual_turn_max_rpm=turn_bound,
                allow_outer_deceleration=bool(
                    (getattr(self.owner, "_vision_control_state", "") == "target_visible_depth_valid"
                     or (getattr(self.owner, "_vision_control_state", "") == "target_visible_low_quality"
                         and self._current_limited_yaw_intent(uid, now) is not None))
                    and self.owner._has_fresh_lateral_yaw(uid)))
            handoff_reason = loss_reason or handoff_reason
        cross_started = self._visible_wheel_guard.started if self._visible_wheel_guard.pending_signs else None
        cross_brake_enabled = getattr(self.config, "follow_cross_brake_enable", False)
        applied, reason = self._visible_wheel_guard.limit(
            guarded_request, feedback, now,
            allow_forward_handoff=bool(
                getattr(self.config, "follow_forward_handoff_enable", False)
                and base > 0 and linear and linear[0] == "forward"
            ),
            allow_aligned_turn=cross_brake_enabled,
            residual_reverse_max_rpm=getattr(self.config, "follow_residual_reverse_max_rpm", 0.0),
            residual_turn_max_rpm=turn_bound,
            aligned_deceleration_max_rpm=(self.config.motor_forward_max_target_rpm
                if (cross_brake_enabled
                    and getattr(self.owner, "_vision_control_state", "") in {
                        "target_visible_depth_valid", "target_visible_low_quality"}
                    and self.owner._has_fresh_lateral_yaw(uid)) else 0.),
            preserve_wait_on_zero=bool(
                guarded_request == (0, 0) and loss_input != (0, 0)
                and sum(loss_input) == 0 and loss_reason
                and (self._visible_wheel_guard.pending_signs is None
                    or all(v*sign > 0 for v, sign in zip(
                        loss_input, self._visible_wheel_guard.pending_signs)))),
        )
        if handoff_reason and (guarded_request == (0, 0) or reason == "continuous"):
            reason = handoff_reason
        if self._near_yaw_park_blocks_write(label):
            return False
        if (applied != (0, 0) and response_adjusted and response_intent is not None
                and (self.owner._lateral_intent_store.snapshot() is not response_intent
                     or getattr(self.owner, "_lateral_turn_response_policy", None) != response_policy)):
            self.logger.info("turn_response_veto reason=intent_or_braking_policy_replaced_before_write")
            return False
        if (self._periodic_follow_writing
                and self.owner._follow_wheel_axes(time.monotonic()) != self._periodic_follow_axes):
            # Feedback/serial lock acquisition can consume the remaining TTL.
            # Never write the pair computed before an axis expired or changed.
            if self._request_follow_axes_rebuild(uid):
                return False
            self.backend.send_targets(0, 0, "FOLLOW20_AUTHORITY_CHANGED")
            self._visible_wheel_guard.reset()
            self.logger.info("follow_wheel_veto uid=%s reason=authority_changed_before_write", uid)
            return False
        if not self._periodic_follow_writing:
            # The direct writer can also wait for encoder/serial feedback.
            # Recheck after that wait, not merely before it: an old packet
            # cannot cross the physical deadline (including the 250ms hold).
            write_linear = self.owner._fresh_depth_linear_snapshot(uid, now=time.monotonic())
            applied_base = .5 * sum(applied)
            write_kind = "forward" if applied_base > 0 else "backward"
            if applied_base and (
                    write_linear is None or write_linear[0] != write_kind
                    or abs(applied_base) > write_linear[1] * self.config.motor_forward_max_target_rpm / 100.0 + 1e-9):
                self.backend.send_targets(0, 0, "FOLLOW_AUTHORITY_CHANGED")
                self._visible_wheel_guard.reset()
                self.logger.info("follow_wheel_veto uid=%s reason=authority_changed_before_direct_write", uid)
                return False
        if (revision != getattr(self.owner, "_lateral_yaw_revision", None)
                or not self._visible_wheel_control_active()
                or self.owner._follow_controller.active_target_id != uid):
            self.logger.info("visible_wheel_revoked uid=%s label=%s reason=authority_changed", uid, label)
            return
        if applied[0] != applied[1] and not self.owner._has_fresh_lateral_yaw(uid):
            self.backend.send_targets(0, 0, "FOLLOW_YAW_AUTHORITY_CHANGED")
            self._visible_wheel_guard.reset()
            self.logger.info("follow_wheel_veto uid=%s reason=yaw_expired_before_write", uid)
            return False
        if applied != (0, 0) and self.backend.normal_zero_hold:
            self.backend.prepare_speed_mode()
            if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                self.backend.send_stop("parking_exit_hard_stop", mode="emergency")
                return False
            # A preserved legacy cross-brake can require register I/O here.
            # Recheck after it, not only before releasing the parking current.
            current_linear = self.owner._fresh_depth_linear_snapshot(uid, now=time.monotonic())
            mean = .5 * sum(applied)
            invalid = (not self._visible_wheel_control_active()
                or self.owner._follow_controller.active_target_id != uid
                or revision != getattr(self.owner, "_lateral_yaw_revision", None)
                or (applied[0] != applied[1] and not self.owner._has_fresh_lateral_yaw(uid))
                or (mean != 0 and (not current_linear
                    or current_linear[0] != ("forward" if mean > 0 else "backward")
                    or abs(mean) > current_linear[1]*self.config.motor_forward_max_target_rpm/100.)))
            if invalid or self._near_yaw_park_blocks_write(label):
                # A newer positive grant/revision can supersede this packet
                # during 0A I/O, including an old yaw-only packet. Rebuild
                # both current axes once instead of reparking merely because
                # the old plan no longer owns authority. The retry rechecks
                # physical TTL, current identity, hazards and wheel reversal.
                if (self._periodic_follow_writing and invalid
                        and self._visible_wheel_control_active()
                        and self.owner._follow_controller.active_target_id == uid
                        and getattr(self.owner, "_near_yaw_park_request", None) is None
                        and current_linear and current_linear[0] == "forward"
                        and current_linear[1] > 0):
                    rebuild = self._request_follow_axes_rebuild(uid)
                    self.logger.info("follow_wheel_retry uid=%s reason=parking_exit_new_forward_grant "
                                     "old_packet_discarded=True parking_reapplied=False "
                                     "immediate_rebuild=%s", uid, rebuild)
                    return False
                self.backend.send_stop("parking_exit_authority_changed", mode=self._ordinary_park_stop_mode(),
                                       preserve_zero=True, prepare_parking_current=True)
                return False
        if trial_phase == "forward_brake":
            brake_feedback = self.get_steering_feedback()
            brake_now = time.monotonic()
            brake_linear = self.owner._fresh_depth_linear_snapshot(uid, now=brake_now)
            if self._near_yaw_park_blocks_write(label):
                return False
            if (self.owner._lateral_intent_store.snapshot() is not trial_intent
                    or brake_now-trial.brake_started >= .08
                    or not self._visible_wheel_control_active()
                    or self.owner._follow_controller.active_target_id != uid
                    or revision != getattr(self.owner, "_lateral_yaw_revision", None)
                    or not self.owner._has_fresh_lateral_yaw(uid)
                    or not brake_linear or brake_linear[0] != "forward"
                    or .5*sum(applied) > brake_linear[1]*self.config.motor_forward_max_target_rpm/100.
                    or not forward_brake_allowed(.5*sum(applied), .5*(applied[0]-applied[1]),
                        trial_intent, brake_feedback, brake_now, uid)):
                self.backend.send_targets(0, 0, "FORWARD_BRAKE_EVIDENCE_CHANGED")
                self._visible_wheel_guard.reset()
                return False
        # The base-preserving assist may leave the packet unchanged. Safety
        # must not depend on whether an assistance branch altered the pair.
        if applied != (0, 0) and self.hard_stop_check(getattr(self.owner, "current_command", None)):
            self.backend.send_stop("visible_wheel_hard_stop", mode="emergency")
            return False
        coalesced_yaw = None
        if applied != (0, 0):
            if self._near_yaw_park_blocks_write(label):
                return False
            # The last safety callback may itself consume the remaining
            # lease or publish a new UID/stop. Never reuse its pre-call age.
            final_now = time.monotonic()
            final_linear = self.owner._fresh_depth_linear_snapshot(uid, now=final_now)
            final_mean = .5*sum(applied)
            final_invalid = (
                not self._visible_wheel_control_active()
                or self.owner._follow_controller.active_target_id != uid
                or revision != getattr(self.owner, "_lateral_yaw_revision", None)
                or not wheel_feedback_valid(feedback, final_now)
                or (applied[0] != applied[1] and not self.owner._has_fresh_lateral_yaw(uid))
                or (final_mean != 0 and (not final_linear
                    or final_linear[0] != ("forward" if final_mean > 0 else "backward")
                    or abs(final_mean) > final_linear[1]*self.config.motor_forward_max_target_rpm/100.)))
            current_axes = (self.owner._follow_wheel_axes(final_now)
                            if self._periodic_follow_writing else None)
            axes_changed = self._periodic_follow_writing and current_axes != self._periodic_follow_axes
            if final_invalid or axes_changed:
                if axes_changed and self._request_follow_axes_rebuild(uid):
                    return False
                # A second equal/smaller yaw update need not revoke the
                # unchanged live forward axis. Coalesce only a continuous
                # pair, once; neither another calculation loop nor a wheel
                # reversal/response/parking handoff is admitted here.
                guard = self._visible_wheel_guard
                contracted = None
                coalescing_store = getattr(self.owner, "_lateral_intent_store", None)
                coalescing_intent = coalescing_store.snapshot() if coalescing_store is not None else None
                if (axes_changed and not getattr(self, "_follow_axes_rebuild_allowed", False)
                        and reason == "continuous" and not response_adjusted
                        and trial_phase in {"disabled", "tracking"} and not moving_handoff
                        and (trial_intent is None or coalescing_intent is trial_intent)
                        and (response_intent is None or coalescing_intent is response_intent)
                        and not any(getattr(coalescing_intent, name, False) for name in (
                            "park_requested", "forward_countersteer", "countersteer_rpm"))
                        and guard.pending_signs is None and guard.resume_signs is None
                        and not guard.pending_full_reverse and not guard.commanded_reverse
                        and guard.residual_turn_signs is None and guard.residual_forward_until <= 0
                        and not self.backend.normal_zero_hold
                        and not self.backend.parking_current_a and not self.backend._parking_current_uncertain
                        and wheel_feedback_valid(feedback, final_now)):
                    contracted = contract_forward_yaw(applied, self._periodic_follow_axes, current_axes)
                if contracted is not None:
                    if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                        self.backend.send_stop("final_yaw_coalescing_hard_stop", mode="emergency")
                        return False
                    if (self._near_yaw_park_blocks_write(label)
                            or getattr(self.owner, "_brake_hold_active", False)):
                        return False  # Its STOP owner must not be overwritten by a speed zero.
                    current_feedback = self.get_steering_feedback()  # Published cache, never serial I/O.
                    checked_at = time.monotonic()
                    checked_linear = self.owner._fresh_depth_linear_snapshot(uid, now=checked_at)
                    # Check after the safety callback and every lease reader.
                    # A third update or any lost evidence fails immediately.
                    valid_contraction = bool(
                        self._periodic_follow_active()
                        and self.owner._follow_controller.active_target_id == uid
                        and getattr(self.owner, "_vision_control_state", "") in {
                            "target_visible", "target_visible_depth_valid"}
                        and not self._near_yaw_park_blocks_write(label)
                        and not getattr(self.backend, "parking_release_fault", None)
                        and not self.backend.normal_zero_hold
                        and not self.backend.parking_current_a and not self.backend._parking_current_uncertain
                        and getattr(self.owner, "_search_handoff_uid", None) != uid
                        and guard.pending_signs is None and guard.resume_signs is None
                        and not guard.pending_full_reverse and not guard.commanded_reverse
                        and guard.residual_turn_signs is None and guard.residual_forward_until <= 0
                        and (coalescing_store is None or coalescing_store.snapshot() is coalescing_intent)
                        and checked_linear is not None and checked_linear[0] == "forward"
                        and final_mean <= checked_linear[1]*self.config.motor_forward_max_target_rpm/100.
                        and (contracted[0] == contracted[1] or self.owner._has_fresh_lateral_yaw(uid))
                        and self.owner._follow_wheel_axes(time.monotonic()) == current_axes
                        and getattr(self.owner, "_lateral_yaw_revision", None) == current_axes[1]
                        and wheel_feedback_valid(feedback, time.monotonic())
                        and wheel_feedback_valid(current_feedback, time.monotonic())
                        and min(current_feedback.left_forward_rpm, current_feedback.right_forward_rpm) >= 0
                        and not self._near_yaw_park_blocks_write(label)
                        and self._visible_wheel_control_active()
                        and self.owner._follow_controller.active_target_id == uid
                        and not getattr(self.owner, "stop_action_execution", False)
                        and not getattr(self.owner, "person_detected_flag", False))
                    if valid_contraction:
                        coalesced_yaw = (self._periodic_follow_axes, current_axes, applied, contracted)
                        applied = contracted
                        self._periodic_follow_axes = current_axes
                        revision, base, yaw = current_axes[1:]
                        final_linear = checked_linear
                        feedback = current_feedback
                        reason = "final_yaw_coalesced"
                    else:
                        contracted = None
                if contracted is None:
                    if (self._near_yaw_park_blocks_write(label)
                            or getattr(self.owner, "_brake_hold_active", False)):
                        return False
                    self.backend.send_targets(0, 0, "FOLLOW_FINAL_AUTHORITY_CHANGED")
                    self._visible_wheel_guard.reset()
                    self.logger.info(
                        "follow_wheel_veto uid=%s reason=final_authority_or_feedback_changed "
                        "axes_changed=%s revision_changed=%s feedback_valid=%s "
                        "feedback_age_ms=%s planned_axes=%s current_axes=%s linear=%s "
                        "retry_available=%s",
                        uid, axes_changed, revision != getattr(self.owner, "_lateral_yaw_revision", None),
                        wheel_feedback_valid(feedback, final_now),
                        None if feedback is None else (final_now-feedback.timestamp)*1000.,
                        getattr(self, "_periodic_follow_axes", None),
                        self.owner._follow_wheel_axes(final_now) if self._periodic_follow_writing else None,
                        final_linear, getattr(self, "_follow_axes_rebuild_allowed", False))
                    return False
        if image_mode and response_adjusted and response_intent is not None and applied != (0, 0):
            # Recheck after ALL potentially blocking safety callbacks, but
            # never let expired boost proof prevent an actual safety zero.
            write_now = time.monotonic()
            if (self.owner._lateral_intent_store.snapshot() is not response_intent
                    or getattr(self.owner, "_lateral_turn_response_policy", None) != response_policy
                    or not response_intent.valid(write_now)
                    or not 0 <= write_now-response_intent.capture_timestamp <= .25
                    or feedback is None or not 0 <= write_now-feedback.timestamp <= .10
                    or (self._turn_buildup.started is not None
                        and write_now-self._turn_buildup.started >= .35)):
                self.logger.info("turn_response_veto reason=buildup_evidence_expired_before_write")
                return False
        if moving_handoff and applied != (0, 0):
            # The evidence may expire or be replaced during parking-mode I/O
            # or the final safety callback. No cached geometry can continue a
            # counter-differential after losing its own or the Depth lease.
            moving_feedback = self.get_steering_feedback()
            moving_now = time.monotonic()
            moving_linear = self.owner._fresh_depth_linear_snapshot(uid, now=moving_now)
            moving_valid = moving_handoff_yaw(self.owner, evidence=moving_evidence, uid=uid,
                base=.5*sum(moving_input), yaw=.5*(moving_input[0]-moving_input[1]),
                feedback=moving_feedback, now=moving_now, policy=cfg,
                execution_delay_sec=getattr(self, "_search_brake_dispatch_delay_sec", .05))
            moving_pair = None if moving_valid is None else (
                int(round(.5*sum(moving_input)+moving_valid[0])),
                int(round(.5*sum(moving_input)-moving_valid[0])))
            quiet_straight = handoff_straight_allowed(
                self.owner, moving=moving_valid, evidence=moving_evidence, uid=uid,
                feedback=moving_feedback, now=moving_now)
            if (moving_valid is None or not moving_linear or moving_linear[0] != "forward"
                    or .5*sum(applied) > moving_linear[1]*self.config.motor_forward_max_target_rpm/100.
                    or self.owner._follow_controller.active_target_id != uid
                    or not self._visible_wheel_control_active()
                    or revision != getattr(self.owner, "_lateral_yaw_revision", None)
                    or (not quiet_straight and not self.owner._has_fresh_lateral_yaw(uid))):
                self.backend.send_targets(0, 0, "SEARCH_HANDOFF_EVIDENCE_CHANGED")
                self._visible_wheel_guard.reset()
                return False
            moving_wheels_changed = (
                moving_feedback.left_forward_rpm != feedback.left_forward_rpm
                or moving_feedback.right_forward_rpm != feedback.right_forward_rpm)
            if moving_pair != applied or moving_wheels_changed:
                # Re-evaluate the whole wheel guard once with the latest
                # encoder, not just patch its already-approved pair. Direct
                # writers and repeated changes fail closed until next tick.
                if self._periodic_follow_writing and self._request_follow_axes_rebuild(uid):
                    return False
                self.backend.send_targets(0, 0, "SEARCH_HANDOFF_CONSTRAINT_CHANGED")
                self._visible_wheel_guard.reset()
                return False
        # A zero planned against missing/older axes can itself become obsolete
        # during the guards above. Nonzero packets already get final lease
        # checks; zeros must also yield to a newly admitted pair, not erase it.
        # Rebuild at most once and rerun ALL reversal/braking checks. A genuine
        # safety zero or a still-current zero request is never delayed here.
        if (applied == (0, 0) and self._periodic_follow_writing
                and self.owner._follow_wheel_axes(time.monotonic()) != self._periodic_follow_axes
                and self._request_follow_axes_rebuild(uid)):
            self.logger.info("follow_wheel_retry uid=%s reason=zero_plan_superseded_before_write "
                             "old_packet_discarded=True motion_authorized=False", uid)
            return False
        previous_pair = getattr(self._visible_wheel_guard, "last_output", None)
        previous_sent = getattr(self._visible_wheel_guard, "last_sent", 0.0)
        cross_brake = "none"
        packet_written = True
        zero_held = bool(getattr(self.backend, "normal_zero_hold", False))
        if (cross_brake_enabled
                and getattr(self.config, "follow_cross_brake_mode", "zero") == "normal"
                and applied == (0, 0)
                and reason in {"cross_wait_zero", "cross_timeout_zero"}):
            if not zero_held:
                self.backend.send_stop("follow_cross_brake", mode=self._ordinary_park_stop_mode(),
                                       preserve_zero=True, prepare_parking_current=True)
                cross_brake = "applied"
            else:
                packet_written = False
                cross_brake = "held"
        else:
            packet_written = not (zero_held and applied == (0, 0))
            cross_brake = "released" if zero_held and any(applied) else "held" if zero_held else "none"
            self.backend.send_targets(applied[0] * ls, applied[1] * rs, label,
                                      max_target_override=max_target_override)
        self._forward_execution_anchor = None
        receipt = getattr(self.backend, "last_speed_receipt", None)
        # Bind only THIS normal-follow write to the exact admitted UID/sample.
        # All other backend writes replace/invalidate receipt identity, even
        # when they happen to request the same wheel speeds.
        if (packet_written and receipt is not None and min(applied) >= 0
                and sum(applied) > 0 and len(final_linear or ()) == 4
                and final_linear[0] == "forward" and final_linear[2] == uid
                and (receipt.left_rpm, receipt.right_rpm) == (applied[0]*ls, applied[1]*rs)):
            self._forward_execution_anchor = ForwardExecutionAnchor(
                uid, final_linear[3], .5*sum(applied), receipt.completed_at, receipt)
        sent_at = time.monotonic()
        if not hasattr(self, "_wheel_diff_response"):
            self._wheel_diff_response = WheelDifferentialResponse()
        response = self._wheel_diff_response.observe_and_note(
            uid, applied, sent_at, previous_sent, feedback, packet_written=packet_written)
        if packet_written:
            self._visible_wheel_guard.note_sent(applied, sent_at)
            self._forward_loss_handoff.note_sent(applied, sent_at)
            self._turn_buildup.note_sent(uid, applied, sent_at, previous_sent)
            if trial is not None:
                trial.record(uid, applied, sent_at)
        if coalesced_yaw is not None:
            # Log only after I/O; logging must not consume the newly checked
            # physical lease between final validation and the speed write.
            self.logger.info("follow_wheel_yaw_coalesced uid=%s old_axes=%s new_axes=%s "
                             "old_pair=%s applied_pair=%s base_preserved=True retries_added=0",
                             uid, *coalesced_yaw)
        self._visible_wheel_waiting = reason in {
            "cross_wait_zero", "cross_timeout_zero", "feedback_unavailable",
            "forward_loss_decelerating", "forward_loss_feedback_wait",
        }
        self._visible_wheel_feedback_ts = (
            feedback.timestamp if feedback is not None and math.isfinite(feedback.timestamp) else 0.0
        )
        cross_episode = (self._visible_wheel_guard.started
                         if self._visible_wheel_guard.pending_signs else cross_started)
        self.logger.info(
            "visible_wheel_dispatch uid=%s label=%s base=%.1f yaw=%.1f "
            "requested_forward_rpm=%s applied_forward_rpm=%s reason=%s depth_fresh=%s "
            "feedback_forward_rpm=%s feedback_ts=%s cross_pending=%s cross_quiet_count=%s "
            "cross_aligned_count=%s cross_full_reverse=%s execution_base_loss_rpm=%.1f "
            "forward_confirmed_handoff=%s obsolete_turn_handoff=%s "
            "requested_diff_rpm=%.1f applied_diff_rpm=%.1f feedback_diff_rpm=%s "
            "feedback_age_ms=%s feedback_trustworthy=%s previous_pair=%s "
            "previous_sent_ts=%.6f sent_ts=%.6f feedback_after_previous_send_ms=%s "
            "evidence_capture_frame_id=%s response_reference_diff_rpm=%s "
            "response_reference_sent_ts=%s response_measured_diff_rpm=%s "
            "response_sustained_ms=%s response_new_feedback=%s response_lagging=%s "
            "cross_brake=%s packet_written=%s cross_wait_ms=%.1f "
            "cross_episode_ts=%s cross_finished=%s forward_loss_state=%s "
            "forward_loss_wait_ms=%.1f forward_loss_quiet_count=%s "
            "zero_owner=%s forward_loss_input_rpm=%s wheel_guard_input_rpm=%s "
            "turn_priority=%s common_accel_removed_rpm=%.1f zero_since=%s "
            "response_phase=%s normal_base_rpm=%.1f normal_yaw_rpm=%.1f "
            "response_extra_yaw_rpm=%.1f response_episode_ms=%.1f "
            "feedback_raw_rpm=%s feedback_position_deg=%s feedback_errors=%s "
            "feedback_read_intervals=%s feedback_yaw_confirmed=%s "
            "cross_pending_wheels=%s cross_decelerating_count=%s cross_residual_count=%s "
            "response_adjusted=%s ordinary_mode_limited=%s",
            uid, label, base, yaw, requested, applied, reason, linear is not None,
            None if feedback is None else (feedback.left_forward_rpm, feedback.right_forward_rpm),
            None if feedback is None else feedback.timestamp,
            getattr(self._visible_wheel_guard, "pending_signs", None),
            getattr(self._visible_wheel_guard, "quiet_count", None),
            getattr(self._visible_wheel_guard, "aligned_count", None),
            getattr(self._visible_wheel_guard, "pending_full_reverse", None),
            max(0., original_base - .5 * sum(applied)),
            reason == "cross_confirmed_forward_handoff",
            reason == "cross_obsolete_turn_forward_handoff",
            requested[0] - requested[1], applied[0] - applied[1],
            None if feedback is None else feedback.left_forward_rpm - feedback.right_forward_rpm,
            None if feedback is None else round((now - feedback.timestamp) * 1000., 1),
            False if feedback is None else feedback.trustworthy,
            previous_pair, previous_sent,
            sent_at,
            None if feedback is None or previous_sent <= 0 else round((feedback.timestamp - previous_sent) * 1000., 1),
            getattr(self.owner, "_last_command_capture_frame", 0),
            response["reference_diff"], response["reference_sent"], response["measured_diff"],
            response["sustained_ms"], response["new_feedback"], response["lagging"],
            cross_brake, packet_written,
            max(0., now - cross_episode) * 1000. if cross_episode is not None else 0.,
            cross_episode,
            bool(cross_started is not None and self._visible_wheel_guard.pending_signs is None and any(applied)),
            handoff_reason or "none",
            (max(0., now-self._forward_loss_handoff.started)*1000.
             if self._forward_loss_handoff.started is not None else 0.),
            self._forward_loss_handoff.quiet_count,
            ("none" if applied != (0, 0) else
             "requested_zero" if requested == (0, 0) else
             "forward_loss_handoff" if guarded_request == (0, 0) else "wheel_zero_cross"),
            loss_input, guarded_request, priority_reason, priority_removed,
            getattr(self._visible_wheel_guard, "zero_since", None),
            response_phase, original_base, original_yaw, abs(yaw)-abs(original_yaw),
            (max(0., now-self._turn_buildup.started)*1000
             if self._turn_buildup.started is not None else 0.) if image_mode else
            (max(0., now-self._turn_response_assist.started)*1000 if self._turn_response_assist.sign else 0.),
            None if feedback is None else (getattr(feedback, "left_speed_rpm", None),
                                          getattr(feedback, "right_speed_rpm", None)),
            None if feedback is None else (getattr(feedback, "left_position_deg", None),
                                          getattr(feedback, "right_position_deg", None)),
            None if feedback is None else (getattr(feedback, "left_error", None),
                                          getattr(feedback, "right_error", None)),
            None if feedback is None else tuple(getattr(feedback, name, None) for name in (
                "left_read_started", "left_read_finished", "right_read_started", "right_read_finished")),
            None if feedback is None else getattr(feedback, "yaw_rate_confirmed", None),
            getattr(self._visible_wheel_guard, "pending_wheels", ()),
            getattr(self._visible_wheel_guard, "decelerating_count", 0),
            getattr(self._visible_wheel_guard, "residual_turn_count", 0),
            response_adjusted,
            not response_adjusted and (base != original_base or yaw != original_yaw),
        )
        return True

    def _current_limited_yaw_intent(self, uid, now):
        store = getattr(self.owner, "_lateral_intent_store", None)
        intent = store.snapshot() if store is not None else None
        if (intent is not None and intent.target_id == uid and intent.valid(now)
                and intent.capture_frame_id > 0
                and intent.mode == "yaw_only" and intent.bbox_quality == "limited"
                and not intent.hold_zero and not intent.park_requested
                and self.owner._follow_controller.active_target_id == uid
                and self.owner._has_fresh_lateral_yaw(uid)):
            return intent
        return None

    def can_handoff_limited_yaw(self, uid, capture_frame_id):
        """Queue interruption is not a stop when the sole writer has new yaw.

        This grants nothing and clears no stop flags. The producer calls it
        AFTER publishing its new limited observation; every physical packet
        still rechecks identity, leases, feedback and wheel reversal.
        """
        if (not self._periodic_follow_active()
                or getattr(self.owner, "_vision_control_state", "") != "target_visible_low_quality"
                or self.owner._follow_controller.active_target_id != uid):
            return False
        now = time.monotonic()
        intent = self._current_limited_yaw_intent(uid, now)
        axes = self.owner._follow_wheel_axes(now)
        return bool(
            intent is not None
            and capture_frame_id > 0 and intent.capture_frame_id == capture_frame_id
            and self.owner._fresh_depth_linear_snapshot(uid, now=now) is None
            and axes is not None and axes[0] == uid and axes[2] == 0
            and math.isfinite(axes[3]) and axes[3] != 0)

    def _periodic_follow_active(self):
        return bool(
            getattr(self.config, "follow_wheel_period_sec", 0.0) >= .05
            and self._visible_wheel_control_active()
            and not getattr(self.owner, "stop_action_execution", False)
            and not getattr(self.owner, "person_detected_flag", False)
            and callable(getattr(self.owner, "_follow_wheel_axes", None))
            and self.owner._follow_wheel_axes(time.monotonic()) is not None
        )

    def _service_follow_wheels(self):
        """Action-thread sole normal-follow writer; safety is checked each loop."""
        if self._service_runtime_shutdown():
            return
        if self._service_near_yaw_park():
            return
        clock = self._follow_wheel_clock
        if not self._periodic_follow_active():
            if clock.last_axes is None:
                return
            # A fresh Depth/identity result can resume follow while this
            # executor waits for motor I/O. An obsolete pre-lock revocation
            # must not insert zero after those new axes have been admitted.
            with self.owner.motor_io_lock:
                if not self._periodic_follow_active():
                    # Revoke once before search/reverse/stop takes over.
                    if getattr(self.owner, "_runtime_shutdown_requested", False):
                        # Shutdown won the lock after the entry check. A zero
                        # target would re-enter speed mode after another STOP.
                        self._shutdown_stop_attempt_at = time.monotonic()
                        self.backend.send_stop("runtime_shutdown", mode=self.config.safety_stop_mode)
                    elif not self._near_yaw_park_blocks_write("FOLLOW20_REVOKED"):
                        self.backend.send_targets(0, 0, "FOLLOW20_REVOKED")
                    clock.reset()
                    self._visible_wheel_guard.reset()
                    self._forward_loss_handoff.reset()
                    return
                self.logger.info("follow_wheel_retry reason=revocation_superseded_before_write "
                                 "old_packet_discarded=True immediate_rebuild=True")
            # Continue through the ordinary UID, hard-stop, TTL and encoder
            # checks; observing a restored state is not motor authorization.
        if self.hard_stop_check(getattr(self.owner, "current_command", None)):
            clock.reset()
            self.send_stop_with_brake_hold("follow20_hard_stop")
            return
        with self.owner.motor_io_lock:
            if not self._periodic_follow_active():
                return
            now = time.monotonic()
            axes = self.owner._follow_wheel_axes(now)
            if axes is None:
                return
            reason = clock.reason(now, axes)
            if reason is None:
                return
            uid, revision, base, yaw = axes
            if clock.last_axes is not None and uid != clock.last_axes[0]:
                if self._near_yaw_park_blocks_write("FOLLOW20_UID_ZERO"):
                    return
                self.backend.send_targets(0, 0, "FOLLOW20_UID_ZERO")
                clock.reset()
                self._visible_wheel_guard.reset()
                self._forward_loss_handoff.reset()
                return
            cap = int(self.config.motor_forward_max_target_rpm)
            ls = self.backend.wheel_raw_state_to_target("left", 1, 0x01)
            rs = self.backend.wheel_raw_state_to_target("right", 1, 0x01)
            self._periodic_follow_writing = True
            try:
                # Rebuild once under the same I/O exclusion when a fresh
                # grant supersedes a plan during calculation. Never replay
                # queued packets or spin while producers keep changing it.
                sent = False
                for attempt in range(2):
                    self._follow_axes_rebuild_allowed = attempt == 0
                    self._follow_axes_rebuild_requested = False
                    self._periodic_follow_axes = axes
                    uid, revision, base, yaw = axes
                    if base >= 0:
                        packet_base = min(base, cap)
                        packet_yaw = math.copysign(min(abs(yaw), cap-packet_base), yaw)
                        left, right = packet_base+packet_yaw, packet_base-packet_yaw
                    else:
                        left, right = base+yaw, base-yaw
                        overflow = max(0, max(left, right)-cap)
                        left, right = max(-cap, left-overflow), max(-cap, right-overflow)
                    sent = self._send_follow_wheel_targets(
                        int(round(left)) * ls, int(round(right)) * rs, "FOLLOW20",
                        max_target_override=cap, visible_required=True)
                    if sent:
                        # The final same-base yaw contraction may adopt one
                        # newer revision without another full calculation.
                        axes = self._periodic_follow_axes
                        uid, revision, base, yaw = axes
                    if not self._follow_axes_rebuild_requested:
                        break
                    axes = self.owner._follow_wheel_axes(time.monotonic())
                    if (axes is None or axes[0] != uid
                            or not (axes[2] > 0 or (axes[2] == 0 and axes[3] != 0
                                    and self.owner._has_fresh_lateral_yaw(uid)))
                            or not self._periodic_follow_active()):
                        self.backend.send_targets(0, 0, "FOLLOW20_REBUILD_REVOKED")
                        self._visible_wheel_guard.reset()
                        break
                    if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                        self.backend.send_stop("follow20_rebuild_hard_stop", mode="emergency")
                        break
            finally:
                self._periodic_follow_writing = False
                self._follow_axes_rebuild_allowed = False
            if sent:
                previous_sent = clock.last_sent
                clock.sent(time.monotonic(), axes)
                self.logger.info(
                    "follow_wheel_tick capture_frame_id=%s uid=%s revision=%s "
                    "base_rpm=%.1f yaw_rpm=%.1f reason=%s interval_ms=%s period_ms=%.1f",
                    getattr(self.owner, "_last_command_capture_frame", None), uid, revision,
                    base, yaw, reason, None if previous_sent is None else round((now-previous_sent)*1000, 2),
                    clock.period * 1000,
                )

    def send_transition_stop_sequence(self, label: str) -> None:
        c = self.config
        try:
            repeat = max(1, int(c.motor_rs485_transition_stop_repeat))
            transition_mode = c.motor_rs485_transition_stop_mode.strip().lower()
            for idx in range(repeat):
                stop_label = label if repeat <= 1 else f"{label}#{idx + 1}"
                if transition_mode in {"zero", "target_zero", "zero_target", "coast"}:
                    with self.owner.motor_io_lock:
                        if self._near_yaw_park_blocks_write(stop_label):
                            return
                        self.backend.send_targets(0, 0, f"{stop_label}_zero")
                    self.backend.motion_armed = False
                    self.logger.info("MSSD 模式切换双轮清零: 标签=%s", stop_label)
                else:
                    with self.owner.motor_io_lock:
                        if self._near_yaw_park_blocks_write(stop_label):
                            return
                        self.backend.send_stop(stop_label, mode=c.motor_rs485_transition_stop_mode)
                if c.motor_rs485_transition_stop_delay_sec > 0:
                    time.sleep(c.motor_rs485_transition_stop_delay_sec)
        except Exception as exc:
            self.logger.warning("运动模式切换停稳失败，继续尝试新动作: %s", exc)

    def _forward_allow_below_min(self) -> bool:
        """Keep approved closed-loop speeds below the legacy launch floor."""
        if bool(getattr(self.owner, "_current_forward_allow_below_min", False)):
            return True
        control_reason = str(getattr(self.owner, "_last_control_decision_reason", ""))
        return bool(
            control_reason.startswith("visual_pid_center_")
            and ("distance_missing" in control_reason or "mmwave_hold" in control_reason)
        )

    def _coast_write_allowed(self, action: int) -> bool:
        """Check under motor I/O lock so a ramp cannot follow a new stop."""
        owner = self.owner
        return bool(
            getattr(owner, "current_command", None) == action
            and getattr(owner, "_last_motor_dispatch_action", action)
            in self.symbols.forward_like_actions
            and not any(
                bool(getattr(owner, name, False))
                for name in (
                    "_brake_hold_active", "_explicit_stop_requested",
                    "_use_soft_stop_next", "_soft_stop_active",
                    "stop_action_execution", "person_detected_flag",
                )
            )
        )

    def send_percent_drive(
        self,
        percent: int,
        *,
        allow_below_min: bool = False,
        coast_action: Optional[int] = None,
        yaw_revision: Optional[int] = None,
    ) -> bool:
        c = self.config
        p = self.backend.clip_percent(percent)
        if p <= 0:
            state = 0x00
        else:
            if not allow_below_min:
                p = max(min(c.min_forward_percent, self.backend.config.percent_limit), p)
            state = 0x01
            if c.motor_forward_raw_target > 0 or c.motor_forward_max_target_rpm > 0:
                if c.motor_forward_raw_target > 0:
                    raw_target = int(c.motor_forward_raw_target)
                else:
                    raw_target = round(
                        int(c.motor_forward_max_target_rpm) * p / 100.0
                    )
                left_target = self.backend.wheel_raw_state_to_target("left", raw_target, state)
                right_target = self.backend.wheel_raw_state_to_target("right", raw_target, state)
                with self.owner.motor_io_lock:
                    if coast_action is not None and not self._coast_write_allowed(coast_action):
                        return False
                    if not self._yaw_revision_write_allowed(yaw_revision, "DRIVE"):
                        return False
                    self._send_follow_wheel_targets(
                        left_target,
                        right_target,
                        "DRIVE",
                        max_target_override=(
                            int(c.motor_forward_max_target_rpm)
                            if c.motor_forward_max_target_rpm > 0
                            else None
                        ),
                    )
                    self._forward_coast_snapshot = (p, bool(allow_below_min))
                self.logger.info(
                    "直行目标RPM下发: 百分比=%d 目标转速=%d rpm 前进上限=%d rpm",
                    p,
                    raw_target,
                    int(c.motor_forward_max_target_rpm),
                )
                return True
        with self.owner.motor_io_lock:
            if self._near_yaw_park_blocks_write("DRIVE"):
                return False
            if coast_action is not None and not self._coast_write_allowed(coast_action):
                return False
            if p > 0 and not self._yaw_revision_write_allowed(yaw_revision, "DRIVE"):
                return False
            self.backend.send_diff(p, state, p, state, "DRIVE")
            if p <= 0:
                self._note_search_retry_zero_sent()
            self._forward_coast_snapshot = (p, bool(allow_below_min)) if p > 0 else None
        return True

    def send_yaw_only(
        self, correction_rpm: int, *, yaw_revision: Optional[int] = None
    ) -> bool:
        """Apply signed camera yaw with zero longitudinal wheel speed."""
        c = self.config
        correction = int(correction_rpm)
        raw = abs(correction)
        if raw <= 0:
            self.send_percent_drive(0)
            return True
        raw_cap = max(
            1,
            int(c.motor_forward_max_target_rpm or self.backend.config.max_target),
        )
        raw = min(raw, raw_cap)
        turn_right = correction > 0
        left_state = 0x01 if turn_right else 0x02
        right_state = 0x02 if turn_right else 0x01
        if c.motor_forward_raw_target > 0 or c.motor_forward_max_target_rpm > 0:
            left_target = self.backend.wheel_raw_state_to_target(
                "left", raw, left_state
            )
            right_target = self.backend.wheel_raw_state_to_target(
                "right", raw, right_state
            )
            with self.owner.motor_io_lock:
                if not self._yaw_revision_write_allowed(yaw_revision, "YAW_ONLY"):
                    return False
                self._send_follow_wheel_targets(
                    left_target,
                    right_target,
                    "YAW_ONLY",
                    max_target_override=(
                        int(c.motor_forward_max_target_rpm)
                        if c.motor_forward_max_target_rpm > 0
                        else None
                    ),
                )
        else:
            percent = self.backend.clip_percent(
                int(round(100.0 * raw / float(raw_cap)))
            )
            left_percent = percent
            right_percent = percent
            if self.backend.config.m1_is_left_wheel:
                m1_percent, m1_state = left_percent, left_state
                m2_percent, m2_state = right_percent, right_state
            else:
                m1_percent, m1_state = right_percent, right_state
                m2_percent, m2_state = left_percent, left_state
            with self.owner.motor_io_lock:
                if not self._yaw_revision_write_allowed(yaw_revision, "YAW_ONLY"):
                    return False
                self.backend.send_diff(
                    m1_percent,
                    m1_state,
                    m2_percent,
                    m2_state,
                    "YAW_ONLY",
                )
        self.logger.info(
            "零纵向偏航下发: 方向=%s PID轮差=%+drpm 双轮原地转速=%drpm",
            "right" if turn_right else "left",
            correction,
            raw,
        )
        return True

    def send_percent_backward(
        self, percent: int, *, yaw_revision: Optional[int] = None
    ) -> Optional[str]:
        """Send reverse with an optional signed visual-PID wheel differential."""
        c = self.config
        p = self.backend.clip_percent(percent)
        correction_rpm = int(getattr(self.owner, "_current_steer_correction_rpm", 0))
        if p <= 0 and correction_rpm != 0:
            return (
                "YAW_ONLY"
                if self.send_yaw_only(correction_rpm, yaw_revision=yaw_revision)
                else None
            )
        state = 0x02 if p > 0 or correction_rpm != 0 else 0x00
        if (p > 0 or correction_rpm != 0) and (
            c.motor_forward_raw_target > 0 or c.motor_forward_max_target_rpm > 0
        ):
            if p <= 0:
                raw_target = 0
            elif c.motor_forward_raw_target > 0:
                raw_target = int(c.motor_forward_raw_target)
            else:
                raw_target = round(int(c.motor_forward_max_target_rpm) * p / 100.0)
            raw_cap = (
                int(c.motor_forward_max_target_rpm)
                if c.motor_forward_max_target_rpm > 0
                else int(self.backend.config.max_target)
            )
            # Positive correction means the target is to the camera's right.
            # While reversing, the right wheel must run faster backwards and
            # the left wheel slower; the magnitude mapping is the opposite of
            # a forward steer-right command.
            left_raw = int(raw_target) - correction_rpm
            right_raw = int(raw_target) + correction_rpm
            if left_raw > raw_cap:
                overflow = left_raw - raw_cap
                left_raw = raw_cap
                right_raw -= overflow
            elif right_raw > raw_cap:
                overflow = right_raw - raw_cap
                right_raw = raw_cap
                left_raw -= overflow
            left_raw = max(0, min(raw_cap, left_raw))
            right_raw = max(0, min(raw_cap, right_raw))
            left_target = self.backend.wheel_raw_state_to_target("left", left_raw, state)
            right_target = self.backend.wheel_raw_state_to_target("right", right_raw, state)
            with self.owner.motor_io_lock:
                if self._near_yaw_park_blocks_write("REVERSE"):
                    return None
                if left_raw != right_raw and not self._yaw_revision_write_allowed(
                    yaw_revision, "REVERSE"
                ):
                    return None
                self.backend.send_targets(
                    left_target,
                    right_target,
                    "REVERSE",
                    max_target_override=(
                        int(c.motor_forward_max_target_rpm)
                        if c.motor_forward_max_target_rpm > 0
                        else None
                    ),
                )
            self.logger.info(
                "倒车距离闭环下发: 百分比=%d 基础转速=%d rpm PID轮差=%+d rpm "
                "左轮转速=%d rpm 右轮转速=%d rpm 左轮目标=%d 右轮目标=%d "
                "轮差限幅原因=%s",
                p,
                raw_target,
                correction_rpm,
                left_raw,
                right_raw,
                left_target,
                right_target,
                str(getattr(self.owner, "_current_steer_limit_reason", "none")),
            )
            return "REVERSE"
        max_rpm = max(1, int(c.motor_forward_max_target_rpm or self.backend.config.max_target))
        correction_percent = int(round(100.0 * correction_rpm / float(max_rpm)))
        left_percent = self.backend.clip_percent(p - correction_percent)
        right_percent = self.backend.clip_percent(p + correction_percent)
        if self.backend.config.m1_is_left_wheel:
            m1_percent, m2_percent = left_percent, right_percent
        else:
            m1_percent, m2_percent = right_percent, left_percent
        with self.owner.motor_io_lock:
            if self._near_yaw_park_blocks_write("REVERSE"):
                return None
            if m1_percent != m2_percent and not self._yaw_revision_write_allowed(
                yaw_revision, "REVERSE"
            ):
                return None
            self.backend.send_diff(m1_percent, state, m2_percent, state, "REVERSE")
        return "REVERSE"

    def send_percent_diff(
        self,
        m1_percent: int,
        m1_state: int,
        m2_percent: int,
        m2_state: int,
        label: str,
        *,
        yaw_revision: Optional[int] = None,
    ) -> bool:
        p1 = self.backend.clip_percent(m1_percent)
        p2 = self.backend.clip_percent(m2_percent)
        positive_steer = bool(
            label == "STEER"
            and ((p1 > 0 and m1_state == 0x01) or (p2 > 0 and m2_state == 0x01))
        )
        with self.owner.motor_io_lock:
            if self._near_yaw_park_blocks_write(label):
                return False
            if (positive_steer or p1 != p2 or m1_state != m2_state) and not self._yaw_revision_write_allowed(
                yaw_revision, label
            ):
                return False
            self.backend.send_diff(p1, m1_state, p2, m2_state, label)
        return True

    def send_percent_brake(self, mode: Optional[str] = None, label: str = "brake") -> bool:
        with self.owner.motor_io_lock:
            ordinary_label = label in ("near_yaw_park", "search_reacquire_brake")
            safety = (mode == "emergency" and not ordinary_label) or label.startswith("safety_")
            if not safety and not self._command_revision_write_allowed(label):
                return False
            if (not safety and self._search_reacquire_brake_request is not None
                    and label != "search_reacquire_brake"):
                return False
            if label == "search_reacquire_brake" and (
                    self._search_reacquire_brake_request is None
                    or getattr(self.owner, "_brake_hold_label", "") != "search_reacquire_brake"):
                return False
            if label == "near_yaw_park" and (
                    getattr(self.owner, "_near_yaw_park_request", None) is None
                    or getattr(self.owner, "_brake_hold_label", "") != "near_yaw_park"):
                # A queued hold refresh cannot re-park after fresh evidence
                # released it, or replace a newer emergency stop.
                return False
            request = getattr(self.owner, "_near_yaw_park_request", None)
            if not safety and ordinary_label:
                evidence = (self._near_yaw_park_settling if label == "near_yaw_park"
                            else self._search_reacquire_settling)
                if evidence is not None:
                    # The latched stop is already applied. Refreshes must not
                    # re-enter a stop mode after the bounded hold ends.
                    self._service_ordinary_park_exit(evidence, label)
                    return True
                mode = self._ordinary_park_stop_mode()
            if (label == "near_yaw_park" and mode == "normal"
                    and request is self._near_yaw_park_applied and request is not None):
                self.backend.refresh_normal_stop(label)
            elif ordinary_label:
                self.backend.send_stop(label, mode=mode, prepare_parking_current=True)
            else:
                self.backend.send_stop(label, mode=mode)
            self._forward_coast_snapshot = None
            return True

    def _note_search_retry_zero_sent(self) -> None:
        """Acknowledge only a successful zero write, once per retry window."""
        requested = float(getattr(self.owner, "_search_retry_zero_requested_at", 0.0))
        if requested > 0 and getattr(self.owner, "_search_retry_zero_sent_at", None) is None:
            self.owner._search_retry_zero_sent_at = time.monotonic()
            self.logger.info(
                "search_observation_zero_sent requested_at=%.6f sent_at=%.6f",
                requested, self.owner._search_retry_zero_sent_at,
            )

    def send_rotate_pulse_zero_stop(self) -> None:
        self.owner._rotate_transition_hold_active = False
        with self.owner.motor_io_lock:
            if self._near_yaw_park_blocks_write("TURN_ZERO"):
                return
            self.backend.send_targets(0, 0, "TURN_ZERO")
            self._note_search_retry_zero_sent()
        self.backend.motion_armed = False

    def send_rotate_transition_hold(self, ended_action: int) -> None:
        """Keep a lost-target search pulse moving through its frame handoff."""
        owner = self.owner
        c = self.config
        s = self.symbols
        rpm = max(0, int(c.rotate_pulse_transition_rpm))
        if rpm <= 0 or ended_action not in (s.rotate_left, s.rotate_right):
            raise ValueError("invalid rotate transition hold")
        if ended_action == s.rotate_right:
            left_state, right_state = 0x01, 0x02
        else:
            left_state, right_state = 0x02, 0x01
        left_target = self.backend.wheel_raw_state_to_target("left", rpm, left_state)
        right_target = self.backend.wheel_raw_state_to_target("right", rpm, right_state)
        with owner.motor_io_lock:
            if self._near_yaw_park_blocks_write("TURN_TRANSITION"):
                return
            self.backend.send_targets(left_target, right_target, "TURN_TRANSITION")
        owner._rotate_transition_hold_active = True
        self.logger.info(
            "rotate transition hold sent: action=%s rpm=%d left=%d right=%d source=%s frame=%d",
            s.action_names.get(ended_action, str(ended_action)),
            rpm,
            left_target,
            right_target,
            str(getattr(owner, "_current_rotate_raw_source", "unknown")),
            int(getattr(owner, "frame_index", -1)),
        )

    def brake_lock_after_rotate_pulse(
        self,
        ended_action: Optional[int] = None,
        *,
        active_brake: bool = False,
    ) -> None:
        owner = self.owner
        if self._service_near_yaw_park():
            return
        owner._follow_distance_hold = None
        c = self.config
        self.logger.info(
            "旋转停止: 脉冲启用=%s 停车模式=%s 持续=%.3f秒 停顿=%.3f秒 最少观察帧=%d 刹车刷新间隔=%.3f秒",
            self.rotate_pulse_enabled(),
            c.rotate_pulse_stop_mode,
            c.rotate_duration,
            c.rotate_pulse_pause_sec,
            c.rotate_pulse_observe_min_frames,
            owner._brake_hold_refresh_interval_sec,
        )
        owner._use_soft_stop_next = False
        if active_brake and self._start_search_active_brake(ended_action):
            owner._brake_hold_active = False
            owner._brake_hold_stop_mode = None
            owner._brake_hold_label = "brake"
            owner._last_brake_hold_send_ts = 0.0
            return
        transition_source = str(getattr(owner, "_current_rotate_raw_source", ""))
        transition_search = transition_source in {"search", "lost_wait", "stale_probe"}
        if (
            transition_search
            and int(c.rotate_pulse_transition_rpm) > 0
            and ended_action in (self.symbols.rotate_left, self.symbols.rotate_right)
        ):
            try:
                self.send_rotate_transition_hold(ended_action)
            except Exception as exc:
                self.logger.warning("旋转脉冲过渡保持失败，退回0RPM: %s", exc)
            else:
                owner._brake_hold_active = False
                owner._brake_hold_stop_mode = None
                owner._brake_hold_label = "brake"
                owner._last_brake_hold_send_ts = 0.0
                return
        if c.rotate_pulse_stop_mode == "zero":
            try:
                self.send_rotate_pulse_zero_stop()
            except Exception as exc:
                self.logger.warning("旋转脉冲结束清零目标失败，退回刹车: %s", exc)
                self.send_percent_brake()
                owner._brake_hold_active = True
                owner._last_brake_hold_send_ts = 0.0
                return
            owner._brake_hold_active = False
            owner._brake_hold_stop_mode = None
            owner._brake_hold_label = "brake"
            owner._last_brake_hold_send_ts = 0.0
            return

        if c.use_percent_speed:
            try:
                self.send_percent_brake()
            except Exception as exc:
                self.logger.warning("旋转脉冲结束刹车失败，退回 ACTION_STOP: %s", exc)
                try:
                    self.send_robot_command(self.symbols.stop)
                except Exception as exc2:
                    self.logger.warning("后备 STOP 失败: %s", exc2)
        else:
            self.send_robot_command(self.symbols.stop)
        owner._brake_hold_active = True
        owner._brake_hold_stop_mode = None
        owner._brake_hold_label = "brake"
        owner._last_brake_hold_send_ts = 0.0

    def _yaw_revision_write_allowed(self, revision: Optional[int], label: str) -> bool:
        """Reject prepared yaw packets superseded while waiting for motor I/O."""
        if self._near_yaw_park_blocks_write(label):
            return False
        if revision is None or revision == getattr(self.owner, "_lateral_yaw_revision", None):
            return True
        self.logger.info(
            "yaw packet revision revoked: label=%s prepared_revision=%d "
            "current_revision=%s action_policy=skip_old_yaw",
            label,
            revision,
            getattr(self.owner, "_lateral_yaw_revision", None),
        )
        return False

    def _visible_rotate_write_allowed(self, action: int) -> bool:
        """Check visible-yaw ownership while the caller holds motor_io_lock."""
        guard = getattr(self.owner, "_visible_rotate_command_allowed", None)
        if guard is None or bool(guard(action)):
            return True
        self.logger.info(
            "visible rotate write revoked: action=%s source=%s frame=%d "
            "action_policy=skip_old_turn",
            self.symbols.action_names.get(action, str(action)),
            str(getattr(self.owner, "_current_rotate_raw_source", "default")),
            int(getattr(self.owner, "frame_index", -1)),
        )
        return False

    @staticmethod
    def _explicit_rotate_raw_source(source: str) -> bool:
        """Only the unnamed/default legacy path may select percent mode."""
        return str(source or "").strip() not in {"", "default"}

    def _rotate_write_allowed(
        self,
        action: int,
        *,
        raw_target: int,
        raw_source: str,
        yaw_revision: Optional[int],
        cancel_revision: int,
    ) -> bool:
        """Validate the prepared rotation packet with motor_io_lock held."""
        owner = self.owner
        if self._near_yaw_park_blocks_write("TURN"):
            return False
        current_source = str(getattr(owner, "_current_rotate_raw_source", "default"))
        current_raw = max(
            0,
            int(getattr(owner, "_current_rotate_raw_target", self.config.motor_rotate_raw_target)),
        )
        explicit_raw = self._explicit_rotate_raw_source(raw_source)
        current_explicit_raw = self._explicit_rotate_raw_source(current_source)
        rejection = None
        if cancel_revision != self._rotate_cancel_revision:
            rejection = "observation_cancelled"
        elif explicit_raw and (raw_target <= 0 or current_raw <= 0):
            # Explicit raw zero withdraws this turn; it never asks the
            # percent-speed backend to substitute its nonzero launch floor.
            rejection = "explicit_raw_zero"
        elif (explicit_raw or current_explicit_raw) and (
            raw_source != current_source or raw_target != current_raw
        ):
            rejection = "raw_command_replaced"
        elif explicit_raw and not self._yaw_revision_write_allowed(yaw_revision, "TURN"):
            return False
        if rejection is not None:
            self.logger.info(
                "rotate packet revoked: action=%s reason=%s source=%s current_source=%s "
                "raw=%d current_raw=%d action_policy=skip_old_turn",
                self.symbols.action_names.get(action, str(action)),
                rejection, raw_source, current_source, raw_target, current_raw,
            )
            return False
        return self._visible_rotate_write_allowed(action)

    def send_robot_command(self, action: int) -> None:
        owner = self.owner
        c = self.config
        s = self.symbols
        if self._service_motion_write_fault():
            return
        if not self._command_revision_write_allowed("dispatch"):
            return
        if self._service_near_yaw_park():
            return
        periodic_follow = self._periodic_follow_active()
        if not periodic_follow:
            # The legacy search/reverse/stop writer is taking ownership now;
            # a later follow tick must not zero its newly issued command.
            self._follow_wheel_clock.reset()
            if not self._visible_wheel_control_active():
                # Search/stop packets are not in the visible writer history.
                # Never attribute their feedback to an earlier forward pair.
                self._turn_buildup = TurnBuildup()
        if periodic_follow and action in (
            s.forward, s.steer_left, s.steer_right, s.rotate_left, s.rotate_right,
        ):
            owner._soft_stop_active = False
            return  # the action thread samples current axes at its next tick
        if (action == s.stop and periodic_follow
                and (getattr(owner, "_use_soft_stop_next", False)
                     or getattr(owner, "_soft_stop_active", False))):
            owner._use_soft_stop_next = False
            # Coalescing a soft stop into canonical axes must not erase its
            # provenance: a later queue interrupt is still not NORMAL park.
            owner._soft_stop_active = True
            return  # zero yaw/base are already explicit canonical axis values
        # Capture before reading any wheel targets. The producer commits a
        # new revision after its zero-yaw fields are ready; a packet prepared
        # before that commit cannot restore a withdrawn wheel difference or
        # a forward target revoked by a canonical near-distance zero.
        yaw_revision = getattr(owner, "_lateral_yaw_revision", None)
        rotate_cancel_revision = self._rotate_cancel_revision
        if action != s.stop:
            # A real motion command takes ownership back from a previous
            # persistent soft zero-yaw hold.
            owner._soft_stop_active = False
        if action == s.backward:
            percent = int(getattr(owner, "_current_forward_percent", 0))
            send_start = time.monotonic()
            try:
                dispatch_label = self.send_percent_backward(percent, yaw_revision=yaw_revision)
            except Exception as exc:
                self.logger.warning("倒车调速发送失败: %s", exc)
                self._service_motion_write_fault()
            else:
                if dispatch_label is not None:
                    self.log_motor_dispatch_timing(action, dispatch_label, send_start)
            return

        if action == s.forward:
            percent = getattr(owner, "_current_forward_percent", 0)
            allow_below_min = self._forward_allow_below_min()
            send_start = time.monotonic()
            try:
                if not self.send_percent_drive(
                    percent, allow_below_min=allow_below_min, yaw_revision=yaw_revision
                ):
                    return
            except Exception as exc:
                self.logger.warning("百分比调速发送失败: %s", exc)
                self._service_motion_write_fault()
            else:
                self.log_motor_dispatch_timing(action, "DRIVE", send_start)
            return

        if action in (s.steer_left, s.steer_right):
            allow_below_min = self._forward_allow_below_min()
            steer_cap = max(0, min(100, int(c.steer_percent_limit)))
            base_cap = min(c.max_forward_percent, steer_cap)
            control_reason = str(getattr(owner, "_last_control_decision_reason", ""))
            mmwave_hold = "mmwave_hold" in control_reason
            distance_missing = "distance_missing" in control_reason
            hold_percent_cap = max(0, min(100, int(c.mmwave_hold_forward_percent)))
            if mmwave_hold:
                base_cap = min(base_cap, hold_percent_cap)
            base = max(0, min(base_cap, int(getattr(owner, "_current_steer_base_percent", 0))))
            direct_correction = max(
                0,
                int(getattr(owner, "_current_steer_correction_rpm", 0)),
            )
            inner_ratio = max(0, min(100, int(getattr(owner, "_current_steer_inner_ratio_percent", c.visible_steer_inner_ratio_percent))))
            outer_ratio = max(0, min(150, int(getattr(owner, "_current_steer_outer_ratio_percent", c.visible_steer_outer_ratio_percent))))
            if base <= 0 and direct_correction > 0:
                send_start = time.monotonic()
                signed_correction = (
                    -direct_correction if action == s.steer_left else direct_correction
                )
                try:
                    if c.rotation_only:
                        dispatched = self.request_rotation_only_yaw_pulse(
                            signed_correction, yaw_revision=yaw_revision
                        )
                        if not dispatched:
                            return
                    else:
                        if not self.send_yaw_only(signed_correction, yaw_revision=yaw_revision):
                            return
                except Exception as exc:
                    self.logger.warning("零纵向偏航发送失败: %s", exc)
                    self._service_motion_write_fault()
                else:
                    self.log_motor_dispatch_timing(action, "YAW_ONLY", send_start)
                return
            if base <= 0 and direct_correction <= 0:
                send_start = time.monotonic()
                try:
                    sent = self.send_percent_drive(0)
                except Exception as exc:
                    self.logger.warning("轮差前进零速停止失败: %s", exc)
                    self._service_motion_write_fault()
                else:
                    if sent:
                        self.log_motor_dispatch_timing(action, "STEER_ZERO", send_start)
                return
            inner = int(math.floor(base * inner_ratio / 100.0))
            outer = int(math.ceil(base * outer_ratio / 100.0))
            # min_forward_percent is a straight-line launch floor. Applying it
            # to both wheels here erases the small differential requested by
            # the centering controller.
            inner = max(0, min(steer_cap, inner))
            outer = max(0, min(steer_cap, outer))
            if base <= 0 and direct_correction > 0 and c.motor_steer_raw_target <= 0:
                max_rpm = max(1, int(c.motor_forward_max_target_rpm or self.backend.config.max_target))
                outer = max(1, int(round(100.0 * direct_correction / float(max_rpm))))
            if mmwave_hold:
                # 毫米波短暂失配时，每个轮子的最终速度都不得超过保持期上限。
                inner = min(inner, hold_percent_cap)
                outer = min(outer, hold_percent_cap)
            fwd = 0x01
            if action == s.steer_left:
                left_percent, right_percent = inner, outer
            else:
                left_percent, right_percent = outer, inner

            if c.motor_steer_raw_target > 0:
                continuous_wheels = self._visible_wheel_control_active()
                # 可信距离和毫米波短时 hold 都共用距离速度曲线；hold 使用的
                # 是上一条已确认距离，摄像头仍负责方向，因此不能退回固定
                # 15 RPM。只有完全没有距离时才使用低速兜底。
                distance_curve_raw = bool(
                    c.motor_forward_max_target_rpm > 0
                    and not distance_missing
                )
                if base <= 0 and direct_correction > 0:
                    base_raw = 0
                    max_target_override = (
                        int(c.motor_forward_max_target_rpm)
                        if c.motor_forward_max_target_rpm > 0
                        else None
                    )
                    raw_mode = "纵向零速横向PID"
                elif distance_curve_raw:
                    base_raw = round(int(c.motor_forward_max_target_rpm) * base / 100.0)
                    max_target_override = int(c.motor_forward_max_target_rpm)
                    raw_mode = "距离曲线转速"
                else:
                    base_raw = max(0, int(c.motor_steer_raw_target))
                    max_target_override = None
                    raw_mode = "异常保持转速" if (mmwave_hold or distance_missing) else "固定转速"
                if c.motor_forward_max_target_rpm > 0:
                    hold_raw_cap = round(
                        int(c.motor_forward_max_target_rpm) * hold_percent_cap / 100.0
                    )
                else:
                    hold_raw_cap = self.backend.percent_to_target(hold_percent_cap)
                if mmwave_hold:
                    base_raw = min(base_raw, hold_raw_cap)
                if direct_correction > 0:
                    # PID 直接给出基础转速两侧的 RPM 差，不再重新量化成比例档。
                    # 轮差已经在视觉 PID 中按误差、基础转速和缺距上限约束。
                    # 这里不能再把 Depth 暂缺时的 5..12 RPM 输出压回 3 RPM，
                    # 否则人物横向快速移动时，日志计算值与电机实际下发不一致。
                    raw_cap = (
                        int(max_target_override)
                        if max_target_override is not None
                        else int(self.backend.config.max_target)
                    )
                    if mmwave_hold:
                        raw_cap = min(raw_cap, int(hold_raw_cap))
                    inner_floor = -raw_cap if continuous_wheels else 0
                    # Preserve base +/- yaw even when the inner wheel must
                    # reverse. The visible wheel writer gates its zero crossing.
                    inner_raw = max(inner_floor, int(base_raw) - direct_correction)
                    outer_raw = int(base_raw) + direct_correction
                    if outer_raw > raw_cap:
                        overflow = outer_raw - raw_cap
                        outer_raw = raw_cap
                        inner_raw = max(inner_floor, inner_raw - overflow)
                    raw_mode += "+编码器角速度PID"
                else:
                    inner_raw = int(math.floor(base_raw * inner_ratio / 100.0))
                    outer_raw = int(math.ceil(base_raw * outer_ratio / 100.0))
                    inner_raw = max(0, inner_raw)
                    outer_raw = max(0, outer_raw)
                    if mmwave_hold:
                        inner_raw = min(inner_raw, hold_raw_cap)
                        outer_raw = min(outer_raw, hold_raw_cap)
                if action == s.steer_left:
                    left_raw, right_raw = inner_raw, outer_raw
                else:
                    left_raw, right_raw = outer_raw, inner_raw
                left_target = self.backend.wheel_raw_state_to_target(
                    "left", abs(left_raw), fwd if left_raw >= 0 else 0x02
                )
                right_target = self.backend.wheel_raw_state_to_target(
                    "right", abs(right_raw), fwd if right_raw >= 0 else 0x02
                )
                try:
                    self.logger.info(
                        "转向电机下发: 动作=%s 模式=%s 基础转速=%d PID轮差=%d "
                        "内轮转速=%d 外轮转速=%d 左轮目标=%d 右轮目标=%d "
                        "内轮比例=%d 外轮比例=%d 毫米波保持限速=%s 轮差限幅原因=%s",
                        s.action_names.get(action, str(action)),
                        raw_mode,
                        base_raw,
                        direct_correction,
                        inner_raw,
                        outer_raw,
                        left_target,
                        right_target,
                        inner_ratio,
                        outer_ratio,
                        mmwave_hold,
                        str(getattr(owner, "_current_steer_limit_reason", "none")),
                    )
                    send_start = time.monotonic()
                    with owner.motor_io_lock:
                        # Equal wheels still carry a positive longitudinal
                        # target that a newer canonical zero can revoke.
                        if (left_raw != 0 or right_raw != 0) and not self._yaw_revision_write_allowed(
                            yaw_revision, "STEER"
                        ):
                            return
                        self._send_follow_wheel_targets(
                            left_target,
                            right_target,
                            "STEER",
                            max_target_override=max_target_override,
                            visible_required=continuous_wheels,
                        )
                        self._forward_coast_snapshot = (base, allow_below_min)
                except Exception as exc:
                    self.logger.warning("轮差 raw 前进发送失败: %s", exc)
                    self._service_motion_write_fault()
                else:
                    self.log_motor_dispatch_timing(action, "STEER", send_start)
                return

            if self.backend.config.m1_is_left_wheel:
                m1_percent, m1_state = left_percent, fwd
                m2_percent, m2_state = right_percent, fwd
            else:
                m1_percent, m1_state = right_percent, fwd
                m2_percent, m2_state = left_percent, fwd

            send_start = time.monotonic()
            try:
                self.logger.info(
                "转向电机下发: 动作=%s 模式=百分比 基础速度=%d 上限=%d 内轮速度=%d 外轮速度=%d 左轮百分比=%d 右轮百分比=%d 内轮比例=%d 外轮比例=%d",
                    s.action_names.get(action, str(action)),
                    base,
                    steer_cap,
                    inner,
                    outer,
                    left_percent,
                    right_percent,
                    inner_ratio,
                    outer_ratio,
                )
                if not self.send_percent_diff(
                    m1_percent, m1_state, m2_percent, m2_state,
                    label="STEER", yaw_revision=yaw_revision,
                ):
                    return
                self._forward_coast_snapshot = (base, allow_below_min)
            except Exception as exc:
                self.logger.warning("轮差前进发送失败: %s", exc)
                self._service_motion_write_fault()
            else:
                self.log_motor_dispatch_timing(action, "STEER", send_start)
            return

        if action in (s.rotate_left, s.rotate_right):
            p = max(0, min(100, int(getattr(owner, "_current_rotate_turn_percent", c.rotate_turn_percent_from_forward))))
            raw_target = max(0, int(getattr(owner, "_current_rotate_raw_target", c.motor_rotate_raw_target)))
            raw_source = str(getattr(owner, "_current_rotate_raw_source", "default"))
            if raw_target <= 0 and self._explicit_rotate_raw_source(raw_source):
                with owner.motor_io_lock:
                    self._rotate_write_allowed(
                        action, raw_target=raw_target, raw_source=raw_source,
                        yaw_revision=yaw_revision, cancel_revision=rotate_cancel_revision,
                    )
                return
            fwd = 0x01
            back = 0x02
            # Keep action names aligned with the signed encoder yaw convention:
            # rotate_right => positive/right yaw, rotate_left => negative/left yaw.
            # Straight and steer_left/steer_right mappings are unchanged.
            if action == s.rotate_right:
                left_state, right_state = fwd, back
            else:
                left_state, right_state = back, fwd

            if raw_target > 0:
                left_target = self.backend.wheel_raw_state_to_target("left", raw_target, left_state)
                right_target = self.backend.wheel_raw_state_to_target("right", raw_target, right_state)
                now_ts = time.time()
                if now_ts - float(getattr(owner, "_last_rotate_dispatch_log_ts", 0.0)) >= 0.25:
                    owner._last_rotate_dispatch_log_ts = now_ts
                    self.logger.info(
                        "旋转电机下发: 动作=%s 模式=原始转速 转速=%d 转速来源代码=%s 默认转速=%d 左轮目标=%d 右轮目标=%d 脉冲启用=%s 持续=%.3f秒 停顿=%.3f秒 指令失效时间=%.3f秒",
                        s.action_names.get(action, str(action)),
                        raw_target,
                        raw_source,
                        c.motor_rotate_raw_target,
                        left_target,
                        right_target,
                        self.rotate_pulse_enabled(action),
                        c.rotate_duration,
                        c.rotate_pulse_pause_sec,
                        c.rotate_hold_stale_sec,
                    )
                try:
                    send_start = time.monotonic()
                    with owner.motor_io_lock:
                        if not self._rotate_write_allowed(
                            action, raw_target=raw_target, raw_source=raw_source,
                            yaw_revision=yaw_revision, cancel_revision=rotate_cancel_revision,
                        ):
                            return
                        self._send_follow_wheel_targets(left_target, right_target, "TURN")
                except Exception as exc:
                    self.logger.warning("差速旋转 raw 发送失败: %s", exc)
                    self._service_motion_write_fault()
                else:
                    self.note_rotate_pulse_target_sent(action)
                    self.log_motor_dispatch_timing(action, "TURN", send_start)
                return

            if self.backend.config.m1_is_left_wheel:
                m1_percent, m1_state = p, left_state
                m2_percent, m2_state = p, right_state
            else:
                m1_percent, m1_state = p, right_state
                m2_percent, m2_state = p, left_state

            now_ts = time.time()
            if now_ts - float(getattr(owner, "_last_rotate_dispatch_log_ts", 0.0)) >= 0.25:
                owner._last_rotate_dispatch_log_ts = now_ts
                self.logger.info(
                "旋转电机下发: 动作=%s 模式=百分比 百分比=%d 脉冲启用=%s 持续=%.3f秒 停顿=%.3f秒 指令失效时间=%.3f秒",
                    s.action_names.get(action, str(action)),
                    p,
                    self.rotate_pulse_enabled(action),
                    c.rotate_duration,
                    c.rotate_pulse_pause_sec,
                    c.rotate_hold_stale_sec,
                )
            send_start = time.monotonic()
            try:
                with owner.motor_io_lock:
                    if not self._rotate_write_allowed(
                        action, raw_target=raw_target, raw_source=raw_source,
                        yaw_revision=yaw_revision, cancel_revision=rotate_cancel_revision,
                    ):
                        return
                    self.backend.send_diff(
                        self.backend.clip_percent(m1_percent),
                        m1_state,
                        self.backend.clip_percent(m2_percent),
                        m2_state,
                        "TURN",
                    )
            except Exception as exc:
                self.logger.warning("差速旋转发送失败: %s", exc)
                self._service_motion_write_fault()
            else:
                self.note_rotate_pulse_target_sent(action)
                self.log_motor_dispatch_timing(action, "TURN", send_start)
                return
            return

        if action == s.stop:
            command = getattr(self._dispatch_context, "command", None)
            self.cancel_yaw_pulses("stop_action", send_zero=False)
            owner._last_stop_command_prepare_ts = time.monotonic()
            owner._last_stop_command_frame = int(getattr(owner, "frame_index", -1))
            owner._last_stop_command_reason = str(getattr(owner, "_last_control_decision_reason", "action_stop"))
            soft_stop = (command.soft_stop if command is not None else
                         bool(getattr(owner, "_use_soft_stop_next", False)
                              or getattr(owner, "_soft_stop_active", False)))
            if soft_stop:
                owner._use_soft_stop_next = False
                owner._soft_stop_active = True
                send_start = time.monotonic()
                try:
                    sent = self.send_percent_drive(0)
                except Exception as exc:
                    self.logger.warning("转向结束软停失败: %s", exc)
                    self._service_motion_write_fault()
                else:
                    if sent:
                        self.log_motor_dispatch_timing(action, "STOP_SOFT", send_start)
                return
            send_start = time.monotonic()
            try:
                sent = self.send_percent_brake()
            except Exception as exc:
                self.logger.warning("百分比刹车发送失败: %s", exc)
            else:
                if sent:
                    self.log_motor_dispatch_timing(action, "STOP", send_start)
            return

        self.logger.warning("未知动作类型: %s", action)

    def send_stop_with_brake_hold(
        self,
        reason: str = "",
        *,
        preserve_motion_params: bool = False,
    ) -> None:
        owner = self.owner
        c = self.config
        reason_text = str(reason or "direct_stop")
        if reason_text in {"stop_signal", "queued_action_stop_signal"}:
            # A generic queue interrupt can race a newly arriving hazard.
            # Recheck the existing safety gate before skipping ordinary STOP.
            try:
                blocked = self.hard_stop_check(None)
            except Exception:
                self.logger.exception("visible_turn_interrupt safety check failed")
                blocked = True
            if blocked:
                reason = reason_text = "hard_stop"
        safety_request = (reason in self.symbols.safety_stop_reasons
            or reason_text.startswith(("runtime_fault_", "runtime_shutdown"))
            or any(token in reason_text for token in
                   ("hard_stop", "front_ir", "left_ir", "right_ir", "bunker", "hazard")))
        if not safety_request:
            if self._search_reacquire_brake_request is not None:
                self._service_search_reacquire_brake()
                return
            if not self._command_revision_write_allowed("stop_hold"):
                return
        if (getattr(owner, "_near_yaw_park_request", None) is not None
                and not safety_request):
            self._service_near_yaw_park()
            return
        # Interrupting a SOFT stop is an ordinary queue transition, not a new
        # parking request. Both the queued and queue-empty paths land here.
        # Keep zero speed without engaging NORMAL position lock / 5A parking.
        if reason_text in {"stop_signal", "queued_action_stop_signal"}:
            with owner.motor_io_lock:
                command = getattr(self._dispatch_context, "command", None)
                # A startup STOP may be adopted while stop_action_execution
                # is still set, so it is interrupted BEFORE its first write.
                # The immutable STOP provenance, not a naked producer flag,
                # lets that pending zero remain soft. Queue-empty execution
                # has not yet restored its adopted snapshot into the context.
                soft_command = command
                if soft_command is None and reason_text == "stop_signal":
                    soft_command = self._current_action_snapshot
                soft_snapshot = bool(
                    soft_command is not None
                    and soft_command.action == self.symbols.stop
                    and soft_command.soft_stop and not soft_command.protected_stop
                )
                pending_soft = soft_snapshot and bool(getattr(owner, "_use_soft_stop_next", False))
                if (soft_snapshot and not safety_request
                        and not getattr(owner, "_explicit_stop_requested", False)
                        and not getattr(owner, "_runtime_shutdown_requested", False)
                        and not getattr(owner, "_brake_hold_active", False)
                        and soft_command.revision != int(getattr(owner, "_action_command_revision", 0))):
                    # Do not let an obsolete pending zero install a generic
                    # brake hold over the new command either.
                    self.logger.info(
                        "soft_stop_interrupt_veto reason=%s command_revision=%d current_revision=%d",
                        reason_text, soft_command.revision,
                        int(getattr(owner, "_action_command_revision", 0)),
                    )
                    return
                if (not safety_request
                        and (self._search_reacquire_brake_request is not None
                             or getattr(owner, "_near_yaw_park_request", None) is not None)):
                    # Published while we waited for motor_io_lock. Its own
                    # executor services the request; do not overwrite it here.
                    return
                uid = getattr(getattr(owner, "_follow_controller", None), "active_target_id", None)
                if (getattr(c, "follow_turn_residual_max_rpm", 0) > 0
                        and self._visible_wheel_control_active()
                        and "lateral" in str(getattr(owner, "_last_action_queue_reason", ""))
                        and owner._has_fresh_lateral_yaw(uid)
                        and not getattr(owner, "_explicit_stop_requested", False)
                        and not getattr(owner, "_runtime_shutdown_requested", False)
                        and not getattr(owner, "_brake_hold_active", False)
                        and not safety_request):
                    # A producer changed steer_left -> rotate_left. The
                    # canonical wheel writer owns this handoff, not NORMAL
                    # parking. Its next tick still validates both axis TTLs.
                    self.logger.info("visible_turn_interrupt_handoff uid=%s reason=%s "
                                     "parking=False old_authority_restored=False", uid, reason_text)
                    return
                if ((getattr(owner, "_soft_stop_active", False) or pending_soft)
                        and not (soft_command is not None and soft_command.protected_stop)
                        and not getattr(owner, "_explicit_stop_requested", False)
                        and not getattr(owner, "_runtime_shutdown_requested", False)
                        and not getattr(owner, "_brake_hold_active", False)
                        and getattr(owner, "_near_yaw_park_request", None) is None
                        and self._search_reacquire_brake_request is None
                        and not safety_request):
                    if self._command_revision_write_allowed("soft_stop_interrupt"):
                        self.backend.send_targets(0, 0, "SOFT_STOP_INTERRUPT")
                        if (getattr(self.backend, "motion_write_fault", None)
                                or getattr(self.backend, "parking_release_fault", None)):
                            return  # Fault service, never a successful soft zero.
                        owner._use_soft_stop_next = False
                        owner._soft_stop_active = True
                        self.logger.info(
                            "soft_stop_interrupt_preserved reason=%s pending=%s capture_frame_id=%s "
                            "mode=zero parking=False", reason_text, pending_soft,
                            getattr(soft_command, "capture_frame_id", None),
                        )
                    return
        # Narrow provenance: only the normal near-distance queued zero may
        # create a measurement-driven recovery episode. Unknown stops remain
        # locked under their existing release policy.
        uid = getattr(getattr(owner, "_follow_controller", None), "active_target_id", None)
        ordinary_distance_hold = bool(
            reason_text == "queued_action_stop_signal"
            and getattr(owner, "_last_control_decision_reason", "") == "near_distance_rotation_only"
            and getattr(owner, "_last_action_queue_reason", "") == "lateral_zero:near_distance_rotation_only"
            and isinstance(uid, int) and not isinstance(uid, bool) and uid > 0
            and callable(getattr(owner, "_depth_longitudinal_authority_enabled", None))
            and owner._depth_longitudinal_authority_enabled()
            and getattr(owner, "search_state", None) == "none"
            and getattr(owner, "_vision_control_state", "").startswith("target_visible")
            and not c.rotation_only
            and not getattr(owner, "_explicit_stop_requested", False)
            and not getattr(owner, "_runtime_shutdown_requested", False)
            and not getattr(owner, "_brake_hold_stop_mode", None)
            and not str(getattr(owner, "_brake_hold_label", "")).startswith("safety")
        )
        owner._follow_distance_hold = (
            FollowDistanceHold(uid, time.monotonic()) if ordinary_distance_hold else None
        )
        if ordinary_distance_hold:
            self.logger.info(
                "follow_distance_hold_started capture_frame_id=%s uid=%s "
                "reason=%s recovery=two_fresh_depth_samples",
                getattr(owner, "_active_capture_frame_id", None), uid, reason_text,
            )
        self._follow_wheel_clock.reset()
        owner._last_command_source_module = (
            "safety_gate"
            if safety_request
            else "action_runtime_direct_stop"
        )
        owner._last_command_control_frame = int(getattr(owner, "frame_index", -1))
        owner._last_command_capture_frame = int(getattr(owner, "_active_capture_frame_id", -1))
        owner._last_command_capture_timestamp = float(getattr(owner, "_active_capture_timestamp", 0.0))
        self.cancel_yaw_pulses(reason or "direct_stop", send_zero=False)
        # 目标距离停车也必须保持刹车，不能因为距离一帧跳过 1.5m 就立即解锁。
        hold_brake = reason != "search_to_follow"
        safety_stop = safety_request
        # 运动模式切换时，停车只负责先把双轮清零；后面即将执行的新动作
        # 已经写入了目标 RPM，不能在这里把它清成 0。红外等真正的停车
        # 路径仍使用默认值并清空运动参数。
        preserve_motion_params = bool(
            preserve_motion_params or reason == "search_to_follow"
        )
        owner._use_soft_stop_next = False
        owner._soft_stop_active = False
        owner._brake_hold_active = hold_brake
        owner._brake_hold_stop_mode = c.safety_stop_mode if safety_stop and hold_brake else None
        owner._brake_hold_label = (
            "follow_distance_hold" if ordinary_distance_hold else f"safety_hold_{reason}"
            if safety_stop and hold_brake
            else "aimline_brake"
            if reason_text == "search_candidate_aimline_brake" and hold_brake
            else "brake"
        )
        owner._last_brake_hold_send_ts = 0.0
        owner._last_stop_command_prepare_ts = time.monotonic()
        owner._last_stop_command_frame = int(getattr(owner, "frame_index", -1))
        owner._last_stop_command_reason = str(reason or "direct_stop")
        if not preserve_motion_params:
            owner.is_forwarding = False
            owner._current_forward_percent = 0
            owner._current_steer_base_percent = 0
        if reason:
            self.logger.info(
                "进入刹车保持状态: 原因代码=%s 保持=%s 停车模式=%s 安全停车=%s 保留运动参数=%s",
                reason,
                hold_brake,
                c.safety_stop_mode if safety_stop else c.motor_rs485_stop_mode,
                safety_stop,
                preserve_motion_params,
            )
        if reason == "search_to_follow":
            self.send_transition_stop_sequence("search_to_follow")
            return
        if safety_stop:
            self.send_percent_brake(mode=c.safety_stop_mode, label=f"safety_{reason}")
            # The direct safety write is the first hold refresh; avoid an
            # immediate duplicate from the background refresh loop.
            owner._last_brake_hold_send_ts = time.time()
            return
        self.send_robot_command(self.symbols.stop)
        # send_robot_command(stop) already sent the brake command. Start the
        # periodic hold interval from that write rather than sending twice.
        owner._last_brake_hold_send_ts = time.time()
