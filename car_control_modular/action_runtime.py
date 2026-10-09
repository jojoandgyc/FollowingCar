from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, FrozenSet, Mapping, Optional

from .control_types import SteeringFeedback
from .sample_braking import SampleBrakingAssessment
from .depth_authority_timing import MAX_FORWARD_DEPTH_TTL_SEC
from .detector_identity_lease import motion_identity_live, ValidatedVisualObservation
from .action_queue_policy import should_drop_queued_action
from .mssd_motor import MssdMotorBackend, MotorSpeedReceipt, MotorSpeedWrite
from .steering_pid import encoder_yaw_rate_right_dps
from .wheel_zero_cross import WheelZeroCrossGuard, wheel_feedback_valid
from .forward_loss_handoff import ForwardLossHandoff
from .longitudinal_execution import ForwardExecutionAnchor, ForwardRecoveryAnchor
from .lateral_intent import LateralControlIntent, LateralIntentStore
from .executed_speed_budget import (
    record_completed_speed, executed_speed_bound_rpm, observe_completed_speed_response,
    executed_interval_speed_bound_rpm, submitted_execution_view,
)
from .steering_limits import effective_correction_limit, clamp_correction
from .turn_response_assist import TurnResponseAssist
from .turn_buildup import TurnBuildup
from .view_retention import ViewRetention
from .turn_response_trial import TurnResponseTrial, forward_brake_allowed
from .wheel_response import WheelDifferentialResponse
from .follow_wheel_clock import FollowWheelClock
from .follow_distance_hold import FollowDistanceHold
from .final_yaw_coalescing import (
    contract_forward_yaw, contract_forward_base, contract_forward_axes, contract_forward_speed,
    ordinary_intent_handoff, straight_park_candidate_contraction,
    contract_straight_forward_handoff, contract_fresh_forward_handoff,
    contract_fresh_forward_neutral_handoff,
)
from .deferred_diagnostics import deferred_diagnostics
from .near_yaw_parking import ParkSettlingEvidence
from .predictive_turn_brake import pulse_feedback_qualified, MAX_PULSE_SEC
from .action_command import ActionCommandSnapshot, SearchReacquireBrakeRequest
from .search_reacquire_braking import (
    limit_handoff_yaw, moving_handoff_yaw, handoff_straight_allowed, note_handoff_zero_write,
)


@dataclass(frozen=True)
class LinearWriteDecision:
    limit_rpm: float
    forward_pair: Optional[tuple[int, int]] = None


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
        follow_axes_reader=None,
        depth_linear_reader=None,
    ) -> None:
        self.owner = owner
        self._follow_axes_reader = follow_axes_reader
        self._depth_linear_reader = depth_linear_reader
        self._follow_wheel_clock = FollowWheelClock(getattr(config, "follow_wheel_period_sec", 0.05))
        self._follow_wheel_last_receipt = None
        self._follow_wheel_last_stop_generation = getattr(backend, "stop_write_generation", 0)
        self._periodic_follow_writing = False
        # Planning may read/control-compute for tens of milliseconds. It is
        # serialized independently from physical serial/encoder transactions.
        self._follow_planning_lock = threading.Lock()
        self._follow_planning_offlock = False
        self._follow_planning_thread_id = None
        self._follow_plan_token = None
        self._follow_commit_stack = None
        self._follow_commit_locked = False
        self._follow_commit_lock_times = []
        self._follow_snapshot_zero_state = None
        self._follow_snapshot_next_attempt = 0
        self._visible_wheel_guard = WheelZeroCrossGuard()
        self._forward_loss_handoff = ForwardLossHandoff()
        self._turn_response_assist = TurnResponseAssist()
        self._turn_buildup = TurnBuildup()
        self._view_retention = ViewRetention()
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
        self._executed_speed_history_lock = threading.Lock()
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
        self.logger.info("view_retention_config enabled=%s soft_side_margin_deg=5 "
                         "min_error_deg=16 capture_samples=3 feedback_samples=2 "
                         "causal_write_ms=100 max_duration_ms=350 max_diff_rpm=20 "
                         "wheel_demand_increase=False lease_extended=False",
                         getattr(config, "follow_turn_response_assist_enable", False))
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
        self.backend.zero_audit_context_provider = self._zero_audit_context
        self._search_reacquire_brake_request = None
        self._search_reacquire_brake_applied = None
        self._search_reacquire_brake_uid = None
        self._search_reacquire_brake_sent_at = 0.0
        self._search_reacquire_settling = None
        self._distance_brake_episode = None
        self._distance_brake_sample_floor = 0.0
        self._distance_brake_stop_generation = None
        if not hasattr(owner, "action_queue_lock"):
            owner.action_queue_lock = threading.Lock()

    @contextmanager
    def _zero_packet_decision(self, stage, reason, **details):
        """Thread-local provenance, never a motor permission or shared veto."""
        previous = getattr(self._dispatch_context, "motor_zero_decision", None)
        self._dispatch_context.motor_zero_decision = dict(
            decision_stage=stage, zero_reason=reason, **details)
        try:
            yield
        finally:
            self._dispatch_context.motor_zero_decision = previous

    def _zero_audit_context(self, label, event_kind):
        """Pure cache snapshot at the backend's actual zero/STOP entry point.

        No serial, control, feedback or intent-store lock. Immutable evidence
        is copied now; asynchronous formatting must not read later state.
        Command provenance and latest observation are deliberately distinct.
        """
        now, owner = time.monotonic(), self.owner
        command = (getattr(self._dispatch_context, "command", None)
                   or self._current_action_snapshot)
        raw = getattr(owner, "_depth30_linear_snapshot", None)
        timing = getattr(owner, "_depth30_prepared_timing", None)
        if timing is None or getattr(timing, "snapshot", None) != raw:
            timing = getattr(owner, "_depth30_linear_timing", None)
        timing_matches = timing is not None and getattr(timing, "snapshot", None) == raw
        if not timing_matches:
            timing = None  # Never label a new grant with an old grant's deadline.
        stamp = raw[3] if isinstance(raw, tuple) and len(raw) == 4 else None
        visual = getattr(owner, "_validated_visual_observation", None)
        feedback = self._steering_feedback
        controller = getattr(owner, "_follow_controller", None)
        intent = getattr(getattr(owner, "_lateral_intent_store", None), "_intent", None)
        def age_ms(value):
            return ((now-value)*1000. if isinstance(value, (int, float))
                    and math.isfinite(value) else None)
        values = dict(
            decision_stage="executor" if label.startswith("FOLLOW") else "command_or_direct_stop",
            zero_reason=label,
            uid=getattr(controller, "active_target_id", None),
            observation_cap=getattr(visual, "capture", None),
            command_sequence=getattr(command, "revision", None),
            command_uid=getattr(command, "uid", None),
            command_cap=getattr(command, "capture_frame_id", None),
            command_source=getattr(command, "source_module", None),
            command_reason=getattr(command, "reason", None),
            command_age_ms=age_ms(getattr(command, "enqueued_at", None)),
            control_cap=getattr(owner, "_last_command_capture_frame", None),
            control_reason=getattr(owner, "_last_control_decision_reason", None),
            planned_axes=getattr(self, "_periodic_follow_axes", None),
            grant=raw, depth_age_ms=age_ms(stamp),
            depth_deadline=getattr(timing, "depth_expires_at", None),
            depth_timing_matches=timing_matches,
            depth_veto=getattr(owner, "_depth30_continuation_veto", None),
            depth_read_veto=getattr(owner, "_depth30_read_veto", None),
            visual_age_ms=age_ms(getattr(visual, "timestamp", None)),
            visual_deadline=getattr(visual, "expires_at", None),
            vision_state=getattr(owner, "_vision_control_state", None),
            feedback_age_ms=age_ms(getattr(feedback, "timestamp", None)),
            feedback_rpm=(getattr(feedback, "left_forward_rpm", None),
                          getattr(feedback, "right_forward_rpm", None)),
            yaw_sequence=getattr(intent, "sequence", None),
            yaw_revision=getattr(owner, "_lateral_yaw_revision", None),
            yaw_policy=getattr(owner, "_lateral_turn_response_policy", None),
            wheel_guard_pending=self._visible_wheel_guard.pending_signs,
            wheel_guard_quiet_count=self._visible_wheel_guard.quiet_count,
            packet_veto=getattr(self, "_follow_write_veto_reason", None),
            explicit_stop=bool(getattr(owner, "_explicit_stop_requested", False)),
            shutdown=bool(getattr(owner, "_runtime_shutdown_requested", False)),
            brake_hold=bool(getattr(owner, "_brake_hold_active", False)),
            park_reason=getattr(getattr(owner, "_near_yaw_park_request", None), "reason", None),
        )
        decision = getattr(self._dispatch_context, "motor_zero_decision", None)
        if decision is not None:
            values.update(decision)
        # A prior queued STOP is useful context, not the cause of a later
        # safety/fault/periodic zero. Only an explicit decision scope may
        # override the actual entry label.
        return values

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

    def _publish_steering_feedback(self, feedback):
        """Publish one precomputed immutable sample between motor commits.

        Physical reads and filtering finish before this brief section. The
        timestamp remains the real read time, never the delayed publication
        time. Motor ownership prevents a late publisher from replacing the
        terminal wheel evidence between its validation and physical write.
        """
        with self.owner.motor_io_lock:
            with self._steering_feedback_lock:
                self._steering_feedback = feedback
        self._observe_executed_speed_response(feedback)

    def get_recording_feedback(self) -> Optional[SteeringFeedback]:
        """Diagnostics only: snapshot the immutable published cache, no I/O/wait.

        The feedback worker replaces (never mutates) this frozen dataclass.
        Recording must not contend for the motor/feedback locks. Control keeps
        using get_steering_feedback; this accessor grants no motion authority.
        """
        return self._steering_feedback

    def _observe_executed_speed_response(self, feedback):
        """Feedback updates physical history independently of motor traffic.

        No I/O or control decisions under this short copy-on-write lock. The
        frozen evidence on an existing Depth grant is deliberately unchanged.
        """
        with self._executed_speed_history_lock:
            history = getattr(self, "_continuation_executed_speed_history", ())
            receipt = getattr(self.backend, "last_speed_receipt", None)
            submission = getattr(self.backend, "last_speed_write", None)
            # A normal pair's I/O/ACK-to-ledger window is not a broken chain.
            # Defer retirement only; never infer a completed lower response.
            view, _, pending = submitted_execution_view(history,
                uid=getattr(self.owner._follow_controller, "active_target_id", None),
                now=time.monotonic(), receipt=receipt, submission=submission,
                stop_generation=getattr(self.backend, "stop_write_generation", None))
            if pending is not None or view is not history:
                return
            updated = observe_completed_speed_response(
                history, feedback=feedback, now=time.monotonic(), receipt=receipt)
            if getattr(self.backend, "last_speed_receipt", None) is receipt:
                self._continuation_executed_speed_history = updated

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
                self._publish_steering_feedback(feedback)
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
                if self._service_detector_reverse_guard():
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
        backend_logger = getattr(self.backend, "logger", self.logger)
        with deferred_diagnostics(self.logger), (
            deferred_diagnostics(backend_logger) if backend_logger is not self.logger else nullcontext()
        ):
            return self._service_runtime_shutdown_deferred()

    def _service_runtime_shutdown_deferred(self) -> bool:
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
        backend_logger = getattr(self.backend, "logger", self.logger)
        with deferred_diagnostics(self.logger), (
            deferred_diagnostics(backend_logger) if backend_logger is not self.logger else nullcontext()
        ):
            return self._service_motion_write_fault_deferred()

    def _service_motion_write_fault_deferred(self) -> bool:
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
        if self._distance_brake_episode is not None:
            # A yaw/zero/old queued packet must not cancel a physical braking
            # episode. Only its service can retire it using fresh evidence.
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
        backend_logger = getattr(self.backend, "logger", self.logger)
        with deferred_diagnostics(self.logger), (
            deferred_diagnostics(backend_logger) if backend_logger is not self.logger else nullcontext()
        ):
            return self._service_search_reacquire_brake_deferred()

    def _service_search_reacquire_brake_deferred(self):
        request = self._search_reacquire_brake_request
        if request is None:
            self._observe_settled_search_brake()
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

    def _observe_settled_search_brake(self):
        """Keep sampling one settled hold until a NEW decision releases it.

        Clearing the search request ends candidate observation, not the
        software brake hold. If we stop sampling here, slow visual decisions
        become the only encoder consumers: a >150 ms decision interval then
        repeatedly destroys quiet evidence despite continuous fresh feedback.
        This is observation only. It cannot release the hold, refresh a motor
        lease, or write anything to the driver.
        """
        evidence = self._search_reacquire_settling
        if evidence is None:
            return
        owner = self.owner

        def still_owns_hold():
            uid = self._search_reacquire_brake_uid
            return bool(
                self._search_reacquire_brake_request is None
                and self._search_reacquire_settling is evidence
                and self._search_reacquire_brake_applied is evidence.request
                and evidence.current_released_at is not None
                and uid is not None
                and uid == getattr(owner._follow_controller, "active_target_id", None)
                and getattr(owner, "_brake_hold_active", False)
                and getattr(owner, "_brake_hold_label", "") == "search_reacquire_brake"
                and getattr(owner, "_brake_hold_stop_mode", None) == self._ordinary_park_stop_mode()
                and getattr(owner, "_near_yaw_park_request", None) is None
                and not getattr(owner, "_explicit_stop_requested", False)
                and not getattr(owner, "_runtime_shutdown_requested", False)
                and owner.running)

        if not still_owns_hold():
            return
        # Serialize with the producer's hold release and any newer STOP.
        # Existing feedback cache only; this adds no serial transaction.
        with owner.motor_io_lock:
            if still_owns_hold():
                evidence.observe(self.get_steering_feedback(), time.monotonic())

    def _service_near_yaw_park(self) -> bool:
        backend_logger = getattr(self.backend, "logger", self.logger)
        with deferred_diagnostics(self.logger), (
            deferred_diagnostics(backend_logger) if backend_logger is not self.logger else nullcontext()
        ):
            return self._service_near_yaw_park_deferred()

    def _service_near_yaw_park_deferred(self) -> bool:
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
        latest = self._read_follow_axes(time.monotonic())
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

    def _request_follow_commit_rebuild(self, uid, publication, *, expired_forward_yaw=False,
                                       terminal_fallback=False, yaw_only_pair=None):
        """Discard a superseded unsent plan; admission runs again OFF I/O lock.

        This cache-only check grants no motion and never extends a lease. A
        publication racing the final serial-lock acquisition is not itself a
        STOP. The service loop releases that lock before reading latest axes
        and rerunning all wheel, braking, identity and deadline checks. Its
        single three-attempt budget also bounds repeated producer updates.
        """
        if (publication is None or not self._follow_offlock_owner()
                or not self._periodic_follow_writing
                or (not getattr(self, "_follow_commit_rebuild_allowed", False)
                    and (not terminal_fallback
                         or getattr(self, "_follow_terminal_forward_used", False)))
                or not self._follow_plan_owns_motor()):
            return False
        owner = self.owner
        if (owner._follow_controller.active_target_id != uid
                or not self._visible_wheel_control_active()
                or any(getattr(owner, name, False) for name in (
                    "_explicit_stop_requested", "_runtime_shutdown_requested",
                    "stop_action_execution", "person_detected_flag", "_brake_hold_active"))
                or getattr(owner, "_near_yaw_park_request", None) is not None
                or getattr(self, "_search_reacquire_brake_request", None) is not None
                or getattr(self.backend, "normal_zero_hold", False)
                or getattr(self.backend, "parking_current_a", 0)
                or getattr(self.backend, "_parking_current_uncertain", False)
                or getattr(self.backend, "parking_release_fault", None)
                or getattr(self.backend, "motion_write_fault", None)):
            return False
        source, revision, intent, policy = publication
        latest = self._follow_grant_marker()
        now = time.monotonic()
        store = getattr(owner, "_lateral_intent_store", None)
        latest_intent = self._follow_intent_snapshot(store)
        latest_policy = getattr(owner, "_lateral_turn_response_policy", None)
        # A mismatched policy is an incomplete/malformed publication, not a
        # new ordinary turn. Do not let a rebuild rebaseline that mismatch.
        if latest_policy is not None and (
                not isinstance(latest_policy, tuple) or len(latest_policy) != 2
                or not isinstance(latest_intent, LateralControlIntent)
                or latest_policy[0] != getattr(latest_intent, "sequence", None)
                or not isinstance(latest_policy[1], bool)):
            return False
        yaw_refresh = bool(yaw_only_pair is not None and source is None and latest is None)
        if yaw_refresh:
            # An equivalent visible pivot needs no Depth grant. Discard the
            # unsent packet and use the SAME bounded service retries to read
            # the new intent and run its complete wheel/TTL/feedback guards.
            # This is not permission to carry the old pivot across expiry,
            # braking evidence, a reversal, or a different motor owner.
            if (not getattr(self, "_follow_commit_rebuild_allowed", False)
                    or len(yaw_only_pair) != 2 or not any(yaw_only_pair)
                    or sum(yaw_only_pair) != 0
                    or not isinstance(intent, LateralControlIntent)
                    or not isinstance(latest_intent, LateralControlIntent)
                    or intent.target_id != uid or latest_intent.target_id != uid
                    or latest_intent.mode != intent.mode
                    or not all(math.isfinite(value) for value in (
                        *yaw_only_pair, latest_intent.initial_correction_rpm,
                        latest_intent.correction_limit_rpm, latest_intent.capture_timestamp,
                        latest_intent.published_at, latest_intent.valid_until, now))
                    or latest_intent.initial_correction_rpm != intent.initial_correction_rpm
                    or latest_intent.initial_correction_rpm * yaw_only_pair[0] <= 0
                    or abs(yaw_only_pair[0]) > latest_intent.correction_limit_rpm
                    or not 0 < latest_intent.capture_timestamp <= latest_intent.published_at <= now
                    or latest_intent.capture_timestamp < intent.capture_timestamp
                    or not latest_intent.valid(now)
                    or getattr(owner, "_lateral_intent_zero_sequence", -1) == latest_intent.sequence
                    or any(getattr(value, name) for value in (intent, latest_intent)
                           for name in ("hold_zero", "park_requested", "forward_countersteer", "countersteer_rpm"))
                    or (policy is not None and (not isinstance(policy, tuple)
                        or len(policy) != 2 or latest_policy is None
                        or policy[1] != latest_policy[1]))):
                return False
        else:
            if latest is None:
                return False  # Missing/half-published authority is not a new grant.
            raw = latest[0]
            try:
                if (len(raw) != 4 or raw[0] != "forward" or raw[2] != uid
                        or not all(math.isfinite(v) for v in (raw[1], raw[3], now))
                        or not 0 < raw[1] <= 100 or not 0 < raw[3] <= now
                        or (source is not None and (source[0][0] != "forward"
                            or source[0][2] != uid or raw[3] < source[0][3]))):
                    return False
            except (TypeError, ValueError, IndexError, OverflowError):
                return False
        if (not expired_forward_yaw
                and (yaw_refresh or self._same_follow_grant_marker(source, latest))
                and revision == getattr(owner, "_lateral_yaw_revision", None)
                and latest_intent is intent and latest_policy == policy):
            return False
        if getattr(self, "_follow_commit_rebuild_allowed", False):
            self._follow_axes_rebuild_requested = True
        else:
            # One straight-only admission remains after ordinary retries;
            # never restart the full retry loop or renew any physical lease.
            self._follow_terminal_forward_requested = True
        self.logger.info("follow_wheel_retry uid=%s reason=%s "
                         "old_packet_discarded=True immediate_rebuild=True "
                         "motion_authorized=False", uid,
                         "equivalent_yaw_published_at_commit" if yaw_refresh else
                         "forward_yaw_expired_at_commit" if expired_forward_yaw
                         else "publication_changed_at_commit")
        return True

    def _follow_grant_marker(self):
        """Identity of the published depth grant, not its age-dependent cap."""
        raw = getattr(self.owner, "_depth30_linear_snapshot", None)
        if raw is None:
            return None
        prepared = getattr(self.owner, "_depth30_prepared_timing", None)
        timing = getattr(self.owner, "_depth30_linear_timing", None)
        if prepared is not None and getattr(prepared, "snapshot", None) == raw:
            timing = prepared
        elif timing is not None and getattr(timing, "snapshot", None) != raw:
            return None  # Publication is in progress; do not coalesce it.
        return raw, timing

    @staticmethod
    def _same_follow_grant_marker(left, right):
        return bool(left is not None and right is not None
                    and left[0] is right[0] and left[1] is right[1])

    def _fresh_forward_grant_handoff(self, uid, planned, current, feedback, now,
                                     *, allow_neutral_yaw=False):
        """Qualify an overtaking same-UID Depth grant using cached evidence.

        This is only permission to rebuild/contract an *unsent* forward plan.
        It does not renew a lease or bypass the terminal brake, STOP, identity,
        encoder or physical deadline checks. In particular, no serial read is
        made while the motor I/O lock is held.
        """
        def reject(reason):
            self._fresh_forward_handoff_reject_reason = reason
            return None

        source = getattr(self, "_follow_source_grant", None)
        latest = self._follow_grant_marker()
        receipt = getattr(self, "_follow_base_receipt_reference", None)
        if (source is None or latest is None or receipt is None
                or self._same_follow_grant_marker(source, latest)):
            return reject("no_new_grant_or_receipt")
        previous, new = source[0], latest[0]
        if (not isinstance(previous, tuple) or len(previous) != 4
                or not isinstance(new, tuple) or len(new) != 4
                or previous[0] != new[0] or new[0] != "forward"
                or previous[2] != new[2] or new[2] != uid
                or not all(isinstance(v, (int, float)) and math.isfinite(v)
                           for v in (previous[3], new[3], new[1], now))
                or not previous[3] < new[3] <= now):
            return reject("not_new_same_uid_forward")
        if (planned is None or current is None or len(planned) != 4
                or len(current) != 4 or planned[0] != current[0]
                or current[0] != uid
                or current[1] != getattr(self.owner, "_lateral_yaw_revision", None)
                or not all(math.isfinite(v) for v in (*planned[2:], *current[2:]))
                or planned[2] <= 0 or current[2] <= 0
                or current[2] > new[1]*self.config.motor_forward_max_target_rpm/100.
                or abs(current[3]) > current[2]
                or (planned[3]*current[3] < 0 and not allow_neutral_yaw)):
            return reject("forward_axes_changed_incompatibly")
        store = getattr(self.owner, "_lateral_intent_store", None)
        intent = store.snapshot() if store is not None else None
        yaw_zeroed = bool(intent is not None
            and getattr(self.owner, "_lateral_intent_zero_sequence", -1) == intent.sequence)
        if (not isinstance(intent, LateralControlIntent)
                or intent.target_id != uid or intent.mode != "forward"
                or intent.bbox_quality != "reliable" or intent.hold_zero
                or intent.near_distance_mode or intent.forward_countersteer
                or intent.countersteer_rpm
                or not all(math.isfinite(v) for v in (
                    intent.capture_timestamp, intent.published_at, intent.valid_until))
                or not 0 < intent.capture_timestamp <= intent.published_at <= now
                or not 0 <= now-intent.capture_timestamp <= .25
                or not intent.valid(now)
                or (intent.park_requested and (planned[3] or current[3]))
                or (yaw_zeroed and current[3] != 0)
                or (current[3] != 0 and not self.owner._has_fresh_lateral_yaw(uid))):
            return reject("visual_handoff_unqualified")
        vision_ts = getattr(self.owner, "_last_vision_control_ts", None)
        if (not isinstance(vision_ts, (int, float)) or not math.isfinite(vision_ts)
                or not 0 <= now-vision_ts <= .25
                or not motion_identity_live(self.owner, uid, now)
                or not self._periodic_follow_scope_active()
                or self.owner._follow_controller.active_target_id != uid
                or getattr(self.owner, "_vision_control_state", "") not in {
                    "target_visible", "target_visible_depth_valid",
                    "target_visible_depth_missing"}
                or getattr(self.owner, "_brake_hold_active", False)
                or getattr(self.owner, "_near_yaw_park_request", None) is not None
                or getattr(self, "_search_reacquire_brake_request", None) is not None
                or getattr(self.owner, "_search_handoff_uid", None) == uid
                or getattr(self.owner, "_explicit_stop_requested", False)
                or getattr(self.owner, "_runtime_shutdown_requested", False)
                or getattr(self.owner, "stop_action_execution", False)
                or getattr(self.owner, "person_detected_flag", False)):
            return reject("identity_or_stop_owner")
        guard = self._visible_wheel_guard
        if (not self._ordinary_forward_feedback_eligible(uid, feedback, now)
                or guard.pending_signs is not None or guard.resume_signs is not None
                or guard.pending_full_reverse or guard.commanded_reverse
                or guard.residual_turn_signs is not None
                or guard.residual_forward_until > 0
                or getattr(self.backend, "last_speed_receipt", None) is not receipt
                or getattr(self.backend, "stop_write_generation", 0)
                    != getattr(self, "_follow_wheel_last_stop_generation", None)
                or self.backend.normal_zero_hold or self.backend.parking_current_a
                or self.backend._parking_current_uncertain
                or self.backend.parking_release_fault
                or getattr(self.backend, "motion_write_fault", None)):
            return reject("wheel_feedback_or_motor_owner")
        self._fresh_forward_handoff_reject_reason = None
        return latest, intent, yaw_zeroed

    @staticmethod
    def _contract_same_grant_forward_base(applied, planned, current):
        """Do the arithmetic for a smaller base when the yaw value is unchanged.

        Revision equality is checked by the live-axis gate, not by this
        arithmetic helper. A fresh lateral publication can advance its
        revision without changing its approved yaw; the caller must still
        prove fresh yaw, the same depth grant and every stop/feedback gate.
        """
        if (planned is None or current is None or len(planned) != 4
                or len(current) != 4 or planned[3] != current[3]):
            return None
        same_revision_plan = (planned[0], current[1], planned[2], planned[3])
        return contract_forward_base(applied, same_revision_plan, current)

    def _can_defer_follow_base_contraction(self, uid, planned, current, feedback, now,
                                          *, allow_yaw_update=False):
        """An unchanged grant's ordinary contraction may reach final review.

        No old packet is sent here. The final stage must rebuild a lower pair
        from current evidence and recheck the physical write deadline. A yaw
        update is optional and may only lower individual guarded wheel speeds.
        """
        def reject(reason):
            # The final veto is a physical zero write. Keep the failed gate
            # available for its post-write audit instead of collapsing every
            # benign cap change and genuine authority loss into one label.
            self._follow_base_contraction_reject_reason = reason
            return False

        source = getattr(self, "_follow_source_grant", None)
        receipt = getattr(self, "_follow_base_receipt_reference", None)
        if source is None or receipt is None:
            return reject("missing_source_or_receipt")
        if not self._same_follow_grant_marker(source, self._follow_grant_marker()):
            return reject("grant_republished")
        if getattr(self.backend, "last_speed_receipt", None) is not receipt:
            return reject("intervening_motor_write")
        if planned is None or current is None or len(planned) != 4 or len(current) != 4:
            return reject("axes_missing")
        if (planned[0] != current[0] or planned[0] != uid
                or current[1] != getattr(self.owner, "_lateral_yaw_revision", None)):
            return reject("uid_or_yaw_revision")
        if (not all(math.isfinite(value) for value in (*planned[2:], *current[2:]))
                or not 0 < current[2] <= planned[2]
                or (not allow_yaw_update and (current[2] == planned[2] or current[3] != planned[3]))
                or (allow_yaw_update and current[3] * planned[3] < 0)
                or abs(current[3]) > current[2]):
            return reject("not_pure_base_reduction")
        store = getattr(self.owner, "_lateral_intent_store", None)
        intent = store.snapshot() if store is not None else None
        if (allow_yaw_update and current[3] != planned[3]
                and any(getattr(intent, name, False) for name in (
                    "park_requested", "forward_countersteer", "countersteer_rpm"))):
            return reject("predictive_braking_intent")
        if current[3] != 0 and not self.owner._has_fresh_lateral_yaw(uid):
            return reject("yaw_evidence_expired")
        cfg = getattr(self.owner._follow_controller, "cfg", None)
        if allow_yaw_update and getattr(cfg, "visible_steering_pid_image_error_only", False):
            limit = effective_correction_limit(
                cfg, current[2], near_distance=bool(intent and intent.near_distance_mode),
                policy_limit=intent.correction_limit_rpm if intent else None)
            if abs(current[3]) > limit:
                return reject("current_yaw_mode_limit")
        if (not self._visible_wheel_control_active()
                or self.owner._follow_controller.active_target_id != uid
                or getattr(self.owner, "_vision_control_state", "") not in {
                    "target_visible", "target_visible_depth_valid",
                    "target_visible_depth_missing"}):
            return reject("visible_identity_state")
        if (getattr(self.owner, "stop_action_execution", False)
                or getattr(self.owner, "person_detected_flag", False)
                or getattr(self.owner, "_brake_hold_active", False)
                or getattr(self.owner, "_near_yaw_park_request", None) is not None
                or getattr(self, "_search_reacquire_brake_request", None) is not None
                or getattr(self.owner, "_search_handoff_uid", None) == uid):
            return reject("stop_or_handoff_owner")
        if (getattr(self.backend, "motion_write_fault", None)
                or getattr(self.backend, "parking_release_fault", None)
                or self.backend.normal_zero_hold or self.backend.parking_current_a
                or self.backend._parking_current_uncertain):
            return reject("motor_or_parking_owner")
        if not self._ordinary_forward_feedback_eligible(uid, feedback, now):
            return reject("wheel_feedback")
        guard = self._visible_wheel_guard
        if (guard.pending_signs is not None or guard.resume_signs is not None
                or guard.pending_full_reverse or guard.commanded_reverse
                or guard.residual_turn_signs is not None or guard.residual_forward_until > 0):
            return reject("wheel_reversal_guard")
        raw = source[0]
        if (len(raw) != 4 or raw[0] != "forward" or raw[2] != uid
                or not math.isfinite(raw[3]) or raw[3] <= 0):
            return reject("source_not_forward")
        effective = self._read_depth_linear(uid, now=now)
        if not (len(effective or ()) == 4 and effective[0] == "forward"
                and effective[1] > 0 and effective[2:] == raw[2:]):
            return reject("live_depth_authority")
        if (not self._same_follow_grant_marker(source, self._follow_grant_marker())
                or getattr(self.backend, "last_speed_receipt", None) is not receipt):
            return reject("grant_or_receipt_changed_during_check")
        self._follow_base_contraction_reject_reason = None
        return True

    def _second_same_grant_straight_axes(self, uid, planned, current, feedback):
        """Rebuild an unsent turn as straight travel on its own live grant.

        This runs before the wheel guard on the second attempt. The returned
        axes still pass that guard and the terminal physical write check; a
        predictive yaw-zero publication cannot turn into a whole-car zero
        merely because the one ordinary rebuild was already spent.
        """
        source = getattr(self, "_follow_source_grant", None)
        receipt = getattr(self, "_follow_base_receipt_reference", None)
        intent_store = getattr(self.owner, "_lateral_intent_store", None)
        intent = intent_store.snapshot() if intent_store is not None else None
        now = time.monotonic()
        stop_generation = getattr(self.backend, "stop_write_generation", 0)
        if (not self._periodic_follow_writing
                or getattr(self, "_follow_axes_rebuild_allowed", False)
                or source is None or receipt is None
                or receipt is not getattr(self, "_follow_wheel_last_receipt", None)
                or getattr(self.backend, "last_speed_receipt", None) is not receipt
                or stop_generation != getattr(self, "_follow_wheel_last_stop_generation", None)
                or not isinstance(source[0], tuple) or len(source[0]) != 4
                or source[0][0] != "forward" or source[0][2] != uid
                or not self._same_follow_grant_marker(source, self._follow_grant_marker())
                or planned is None or current is None
                or len(planned) != 4 or len(current) != 4
                or planned[0] != current[0] or current[0] != uid
                or not all(math.isfinite(value) for value in (*planned[2:], *current[2:]))
                or not 0 < current[2] <= planned[2]
                or not 0 < abs(planned[3]) <= planned[2]
                or current[3] != 0
                or not isinstance(intent, LateralControlIntent)
                or intent.target_id != uid or intent.mode != "forward"
                or intent.bbox_quality != "reliable" or intent.near_distance_mode
                or intent.hold_zero or not intent.valid(now)
                or getattr(self.owner, "_lateral_intent_zero_sequence", -1)
                    != intent.sequence
                or not 0 < intent.capture_timestamp <= intent.published_at <= now
                or now-intent.capture_timestamp > .25
                or not self._periodic_follow_active()
                or not motion_identity_live(self.owner, uid, now)
                or not self._ordinary_forward_feedback_eligible(uid, feedback, now)
                or getattr(self.owner, "_brake_hold_active", False)
                or getattr(self.owner, "_near_yaw_park_request", None) is not None
                or getattr(self, "_search_reacquire_brake_request", None) is not None
                or getattr(self.owner, "_search_handoff_uid", None) == uid
                or self.backend.normal_zero_hold or self.backend.parking_current_a
                or self.backend._parking_current_uncertain
                or self.backend.parking_release_fault
                or getattr(self.backend, "motion_write_fault", None)):
            return None
        ls = self.backend.wheel_raw_state_to_target("left", 1, 0x01)
        rs = self.backend.wheel_raw_state_to_target("right", 1, 0x01)
        previous_pair = (receipt.left_rpm*ls, receipt.right_rpm*rs)
        if (min(previous_pair) <= 0
                or current[2] >= min(previous_pair)):
            return None
        guard = self._visible_wheel_guard
        if (guard.pending_signs is not None or guard.resume_signs is not None
                or guard.pending_full_reverse or guard.commanded_reverse
                or guard.residual_turn_signs is not None or guard.residual_forward_until > 0):
            return None
        linear = self._read_depth_linear(uid, now=now)
        checked_at = time.monotonic()
        latest = self._read_follow_axes(checked_at)
        if (len(linear or ()) != 4 or linear[0] != "forward"
                or linear[2:] != source[0][2:]
                or current != latest
                or current[1] != getattr(self.owner, "_lateral_yaw_revision", None)
                or current[2] > linear[1]*self.config.motor_forward_max_target_rpm/100.
                or not wheel_feedback_valid(feedback, checked_at)
                or intent_store.snapshot() is not intent
                or not self._same_follow_grant_marker(source, self._follow_grant_marker())
                or getattr(self.backend, "last_speed_receipt", None) is not receipt
                or getattr(self.backend, "stop_write_generation", 0) != stop_generation):
            return None
        return current, linear, receipt, stop_generation

    def _final_lower_straight_grant_handoff(self, uid, applied, feedback):
        """Rebind one unsent straight pair to a newly published lower grant.

        It cannot increase either guarded wheel or the last completed motor
        target. The ordinary terminal check still authorizes the final write.
        """
        source = getattr(self, "_follow_source_grant", None)
        receipt = getattr(self, "_follow_base_receipt_reference", None)
        stop_generation = getattr(self.backend, "stop_write_generation", 0)
        if (not self._periodic_follow_writing or source is None or receipt is None
                or receipt is not getattr(self, "_follow_wheel_last_receipt", None)
                or getattr(self.backend, "last_speed_receipt", None) is not receipt
                or stop_generation != getattr(self, "_follow_wheel_last_stop_generation", None)
                or not isinstance(source[0], tuple) or len(source[0]) != 4
                or source[0][0] != "forward" or source[0][2] != uid
                or min(applied) <= 0 or applied[0] != applied[1]
                or self._visible_wheel_guard.pending_signs is not None
                or self._visible_wheel_guard.resume_signs is not None
                or self._visible_wheel_guard.pending_full_reverse
                or self._visible_wheel_guard.commanded_reverse
                or self._visible_wheel_guard.residual_turn_signs is not None
                or self._visible_wheel_guard.residual_forward_until > 0):
            return None
        now = time.monotonic()
        marker = self._follow_grant_marker()
        if (marker is None or self._same_follow_grant_marker(source, marker)
                or not isinstance(marker[0], tuple) or len(marker[0]) != 4
                or marker[0][0] != "forward" or marker[0][2] != uid
                or not all(isinstance(value, (int, float)) and math.isfinite(value)
                           for value in (source[0][3], marker[0][3], marker[0][1]))
                or marker[0][3] <= source[0][3]):
            return None
        store = getattr(self.owner, "_lateral_intent_store", None)
        intent = store.snapshot() if store is not None else None
        policy = getattr(self.owner, "_lateral_turn_response_policy", None)
        linear = self._read_depth_linear(uid, now=now)
        axes = self._read_follow_axes(time.monotonic())
        current_feedback = self.get_steering_feedback()
        checked_at = time.monotonic()
        if (len(linear or ()) != 4 or linear[0] != "forward"
                or linear[2:] != marker[0][2:]
                or axes is None or len(axes) != 4 or axes[0] != uid
                or axes[1] != getattr(self.owner, "_lateral_yaw_revision", None)
                or axes[2] <= 0 or axes[3] != 0
                or not self._periodic_follow_active()
                or not motion_identity_live(self.owner, uid, checked_at)
                or not wheel_feedback_valid(feedback, checked_at)
                or not self._ordinary_forward_feedback_eligible(uid, current_feedback, checked_at)
                or getattr(self.owner, "_brake_hold_active", False)
                or getattr(self.owner, "_near_yaw_park_request", None) is not None
                or getattr(self, "_search_reacquire_brake_request", None) is not None
                or getattr(self.owner, "_search_handoff_uid", None) == uid
                or self.backend.normal_zero_hold or self.backend.parking_current_a
                or self.backend._parking_current_uncertain
                or self.backend.parking_release_fault
                or getattr(self.backend, "motion_write_fault", None)):
            return None
        pair = contract_straight_forward_handoff(
            uid, intent, applied, self._periodic_follow_axes, axes,
            linear[1]*self.config.motor_forward_max_target_rpm/100., checked_at)
        ls = self.backend.wheel_raw_state_to_target("left", 1, 0x01)
        rs = self.backend.wheel_raw_state_to_target("right", 1, 0x01)
        previous_pair = (receipt.left_rpm*ls, receipt.right_rpm*rs)
        if (pair is None or min(previous_pair) <= 0
                or any(new >= old for new, old in zip(pair, previous_pair))
                or not isinstance(intent, LateralControlIntent)
                or not isinstance(getattr(self.owner, "_last_vision_control_ts", None), (int, float))
                or not 0 <= checked_at-self.owner._last_vision_control_ts <= .25
                or store.snapshot() is not intent
                or getattr(self.owner, "_lateral_turn_response_policy", None) != policy
                or self._read_follow_axes(time.monotonic()) != axes
                or not self._same_follow_grant_marker(marker, self._follow_grant_marker())
                or getattr(self.backend, "last_speed_receipt", None) is not receipt
                or getattr(self.backend, "stop_write_generation", 0) != stop_generation):
            return None
        return (pair, axes, linear, marker, current_feedback, intent, policy,
                receipt, stop_generation)

    def _final_live_forward_handoff(self, uid, applied, feedback):
        """Finish a superseded ordinary turn as a bounded straight packet.

        This is a one-shot contraction of an already guarded pair, not a
        retry or retention of the old command. Current Depth independently
        authorizes translation; expired/replaced yaw cannot authorize a turn.
        Neither wheel may exceed its guarded speed. The caller still checks
        the exact adopted grant, intent, receipt, STOP token and write clock.
        """
        source = getattr(self, "_follow_source_grant", None)
        receipt = getattr(self, "_follow_base_receipt_reference", None)
        stop_generation = getattr(self.backend, "stop_write_generation", 0)
        planned = getattr(self, "_periodic_follow_axes", None)
        guard = self._visible_wheel_guard
        if (not self._periodic_follow_writing or source is None or receipt is None
                or receipt is not getattr(self, "_follow_wheel_last_receipt", None)
                or getattr(self.backend, "last_speed_receipt", None) is not receipt
                or stop_generation != getattr(self, "_follow_wheel_last_stop_generation", None)
                or planned is None or len(planned) != 4 or planned[0] != uid
                or not isinstance(source[0], tuple) or len(source[0]) != 4
                or source[0][0] != "forward" or source[0][2] != uid
                or not all(math.isfinite(v) for v in (*applied, *planned[2:]))
                or min(applied) <= 0 or planned[2] <= 0
                or guard.pending_signs is not None or guard.resume_signs is not None
                or guard.pending_full_reverse or guard.commanded_reverse
                or guard.residual_turn_signs is not None or guard.residual_forward_until > 0):
            return None
        store = getattr(self.owner, "_lateral_intent_store", None)
        intent = store.snapshot() if store is not None else None
        policy = getattr(self.owner, "_lateral_turn_response_policy", None)
        marker = self._follow_grant_marker()
        if (marker is None or not isinstance(marker[0], tuple) or len(marker[0]) != 4
                or marker[0][0] != "forward" or marker[0][2] != uid
                or not all(isinstance(v, (int, float)) and math.isfinite(v)
                           for v in (source[0][3], marker[0][3]))
                or marker[0][3] <= source[0][3]
                or not isinstance(intent, LateralControlIntent)
                or intent.target_id != uid or intent.mode != "forward"
                or intent.bbox_quality != "reliable" or intent.hold_zero
                or intent.near_distance_mode):
            return None
        zeroed = getattr(self.owner, "_lateral_intent_zero_sequence", -1) == intent.sequence
        if (intent.forward_countersteer or intent.countersteer_rpm) and not zeroed:
            return None
        now = time.monotonic()
        linear = self._read_depth_linear(uid, now=now)
        axes = self._read_follow_axes(time.monotonic())
        current_feedback = self.get_steering_feedback()
        yaw_live = self.owner._has_fresh_lateral_yaw(uid)
        active = self._periodic_follow_active()
        checked_at = time.monotonic()
        if (len(linear or ()) != 4 or linear[0] != "forward"
                or linear[2:] != marker[0][2:]
                or axes is None or len(axes) != 4 or axes[0] != uid
                or axes[1] != getattr(self.owner, "_lateral_yaw_revision", None)
                or not all(math.isfinite(v) for v in (*axes[2:], linear[1], checked_at,
                       intent.capture_timestamp, intent.published_at, intent.valid_until))
                or axes[2] <= 0 or abs(axes[3]) > axes[2]
                or not 0 < intent.capture_timestamp <= intent.published_at <= checked_at
                or not 0 <= checked_at-intent.capture_timestamp <= .25
                or not active or not motion_identity_live(self.owner, uid, checked_at)
                or not wheel_feedback_valid(feedback, checked_at)
                or not self._ordinary_forward_feedback_eligible(uid, current_feedback, checked_at)
                or getattr(self.owner, "_brake_hold_active", False)
                or getattr(self.owner, "_near_yaw_park_request", None) is not None
                or getattr(self, "_search_reacquire_brake_request", None) is not None
                or getattr(self.owner, "_search_handoff_uid", None) == uid
                or self.backend.normal_zero_hold or self.backend.parking_current_a
                or self.backend._parking_current_uncertain
                or self.backend.parking_release_fault
                or getattr(self.backend, "motion_write_fault", None)):
            return None
        # A live ordinary turn should go through the existing curve path.
        # This terminal bridge only removes an expired/zeroed/superseded yaw;
        # it cannot ignore an active countersteer or invent a new turn.
        if (axes[3] != 0 and yaw_live and intent.valid(checked_at) and not zeroed
                and axes[3]*planned[3] >= 0):
            return None
        speed = math.floor(min(*applied, axes[2],
                               linear[1]*self.config.motor_forward_max_target_rpm/100.))
        if (speed <= 0 or store.snapshot() is not intent
                or getattr(self.owner, "_lateral_turn_response_policy", None) != policy
                or self._read_follow_axes(time.monotonic()) != axes
                or not self._same_follow_grant_marker(marker, self._follow_grant_marker())
                or getattr(self.backend, "last_speed_receipt", None) is not receipt
                or getattr(self.backend, "stop_write_generation", 0) != stop_generation):
            return None
        return ((speed, speed), axes, linear, marker, current_feedback,
                intent, policy, receipt, stop_generation)

    def continuation_executed_speed_bound_rpm(self, uid, now):
        """Recent completed/submitted outer speed for braking MAX only.

        Unlike execution continuity this survives a later low/zero write:
        issuing zero is not proof of stopped wheels.  It cannot grant motion,
        clear a veto, refresh a depth lease, or replace current feedback.
        """
        if getattr(self.owner._follow_controller, "active_target_id", None) != uid:
            return None
        return self._execution_budget_bound(uid, now, require_current_history=False)

    def _execution_budget_bound(self, uid, now, sample_timestamp=None, *,
                                require_current_history=True):
        """Cache-only accounting across normal dual-write transactions.

        No motor/control lock, no fabricated receipt and no lease extension.
        A possible in-flight speed is MAX cost, never permission to accelerate.
        Two bounded snapshot attempts cover ACK/ledger publication transitions.
        """
        controller = self.owner._follow_controller
        for _ in range(2):
            if getattr(controller, "active_target_id", None) != uid:
                return None
            history = getattr(self, "_continuation_executed_speed_history", ())
            receipt = getattr(self.backend, "last_speed_receipt", None)
            submission = getattr(self.backend, "last_speed_write", None)
            generation = getattr(self.backend, "stop_write_generation",
                                 0 if sample_timestamp is None else None)
            # A just-published ACK must not look as if it came from the future
            # merely because it arrived between the earlier clock and cache read.
            checked_at = max(now, time.monotonic())
            if (type(generation) is not int or generation < 0
                    or getattr(self.backend, "motion_write_fault", None)
                    or getattr(self.backend, "parking_release_fault", None)):
                return None
            view, anchor, pending = submitted_execution_view(history, uid=uid,
                now=checked_at, receipt=receipt, submission=submission,
                stop_generation=generation)
            if sample_timestamp is None:
                result = executed_speed_bound_rpm(view, uid=uid, now=checked_at)
                if pending is not None:
                    result = max(pending, result or 0.)
            else:
                result = executed_interval_speed_bound_rpm(view, uid=uid,
                    sample_timestamp=sample_timestamp, now=checked_at, receipt=anchor)
                if result is not None and pending is not None:
                    result = max(result, pending)
            if (getattr(controller, "active_target_id", None) == uid
                    and getattr(self.backend, "last_speed_receipt", None) is receipt
                    and getattr(self.backend, "last_speed_write", None) is submission
                    and getattr(self.backend, "stop_write_generation", 0) == generation
                    and not getattr(self.backend, "motion_write_fault", None)
                    and not getattr(self.backend, "parking_release_fault", None)
                    and (not require_current_history
                         or getattr(self, "_continuation_executed_speed_history", ()) is history)):
                return result
            # Retry only a known normal follow I/O/ACK transition. Do not
            # turn STOP, an untracked write, or a strict admission's changed
            # feedback/retirement snapshot into a second chance to qualify.
            latest = getattr(self.backend, "last_speed_write", None)
            if (getattr(controller, "active_target_id", None) != uid
                    or getattr(self.backend, "stop_write_generation", 0) != generation
                    or getattr(self.backend, "motion_write_fault", None)
                    or getattr(self.backend, "parking_release_fault", None)
                    or not isinstance(latest, MotorSpeedWrite) or latest.uid != uid
                    or (getattr(self.backend, "last_speed_receipt", None) is receipt
                        and latest is submission)):
                return None
        return None

    def braking_interval_speed_bound_rpm(self, uid, sample_timestamp, now):
        """Cache-only interval proof; missing coverage never means no motion.

        No encoder/serial read or lock acquisition. The immutable history and
        dual-write transaction are checked again after the pure query. Normal
        ACK/ledger transitions get bounded snapshot retries; STOP, untracked
        writes or missing coverage still lose this proof.
        """
        return self._braking_interval_speed_bound_rpm(uid, sample_timestamp, now,
                                                    require_current_history=True)

    def braking_interval_continuation_bound_rpm(self, uid, sample_timestamp, now):
        """Recheck an admitted covered interval without a retirement race.

        A feedback-only retirement may replace the immutable history while
        this reads it. Keeping the older, higher cost is conservative. Actual
        STOPs/untracked writes still fail closed. A submitted normal pair adds
        MAX cost; an actual dual ACK may bridge its brief ledger-publication
        gap without letting an unfinished write masquerade as a receipt.
        """
        return self._braking_interval_speed_bound_rpm(uid, sample_timestamp, now,
                                                    require_current_history=False)

    def _braking_interval_speed_bound_rpm(self, uid, sample_timestamp, now, *,
                                         require_current_history):
        return self._execution_budget_bound(uid, now, sample_timestamp,
            require_current_history=require_current_history)

    def _linear_packet_within_write_deadline(self, linear, uid, mean_rpm, *,
                                             source_grant=None, feedback=None):
        """Compatibility predicate; the writer also consumes a lower live cap."""
        limit = self._linear_packet_write_limit(
            linear, uid, mean_rpm, source_grant=source_grant, feedback=feedback)
        return limit is not None and abs(mean_rpm) <= limit.limit_rpm

    def _linear_packet_write_limit(self, linear, uid, mean_rpm, *,
                                   source_grant=None, feedback=None, forward_pair=None,
                                   planned_revision=None, yaw_required=False,
                                   adopted_intent=..., adopted_policy=...,
                                   expected_receipt=..., expected_stop_generation=...,
                                   independent_straight=False, commit_after_prepare=False):
        """Prepare an optional contraction, then check its physical lifetime.

        The ordinary authority reader may log or wait while evaluating its
        braking envelope. Do not re-enter it here: check its returned grant
        against the canonical tuple and deadlines AFTER that work instead.
        Legacy owners without the physical-depth clock retain their existing
        reader contract; PersonTracker always exposes this clock policy.
        """
        self._follow_write_veto_reason = None
        def reject(reason):
            self._follow_write_veto_reason = reason
            return None

        self._snapshot_used_feedback(feedback)
        if independent_straight and (yaw_required or mean_rpm <= 0
                or forward_pair is None or len(forward_pair) != 2
                or forward_pair[0] != forward_pair[1] or forward_pair[0] != mean_rpm):
            return reject("invalid_independent_straight_packet")

        if ((expected_receipt is not ...
                and getattr(self.backend, "last_speed_receipt", None) is not expected_receipt)
                or (expected_stop_generation is not ...
                    and getattr(self.backend, "stop_write_generation", 0)
                        != expected_stop_generation)):
            return reject("handoff_receipt_or_stop_changed")
        if mean_rpm == 0:
            return LinearWriteDecision(0.)  # This check grants no yaw authority.
        if (self._distance_brake_stop_generation is not None
                and self.backend.stop_write_generation != self._distance_brake_stop_generation):
            return reject("distance_brake_newer_stop_owner")
        if (mean_rpm > 0 and self._distance_brake_sample_floor > 0
                and (len(linear or ()) != 4 or linear[3] <= self._distance_brake_sample_floor)):
            return reject("distance_before_brake_write")
        ttl_reader = getattr(self.owner, "_depth_linear_max_age_sec", None)
        if not callable(ttl_reader):
            return LinearWriteDecision(abs(mean_rpm), forward_pair)
        # A quiet continuation-cap calculation below is still Python work:
        # STOP or a different speed writer may take ownership while it runs.
        # Retain the completed-write identity before that work and refuse to
        # turn a later STOP back into a speed-mode packet.
        limit_rpm = abs(mean_rpm)
        expected_revision = (getattr(self.owner, "_lateral_yaw_revision", None)
                             if planned_revision is None else planned_revision)
        intent_store = getattr(self.owner, "_lateral_intent_store", None)
        expected_intent = self._follow_intent_snapshot(intent_store)
        expected_policy = getattr(self.owner, "_lateral_turn_response_policy", None)
        neutral_straight = bool(not yaw_required and mean_rpm > 0
            and forward_pair is not None and len(forward_pair) == 2
            and forward_pair[0] == forward_pair[1] == mean_rpm
            and self._neutral_follow_yaw(uid, expected_intent, time.monotonic()))
        if (not independent_straight and not neutral_straight and (
                (adopted_intent is not ... and expected_intent is not adopted_intent)
                or (adopted_policy is not ... and expected_policy != adopted_policy))):
            return reject("adopted_intent_or_policy_changed")
        prior_receipt = getattr(self.backend, "last_speed_receipt", None)
        prior_stop_generation = getattr(self.backend, "stop_write_generation", 0)
        if ((expected_receipt is not ... and prior_receipt is not expected_receipt)
                or (expected_stop_generation is not ...
                    and prior_stop_generation != expected_stop_generation)):
            return reject("handoff_receipt_or_stop_changed")
        prior_zero_hold = bool(getattr(self.backend, "normal_zero_hold", False))
        current = getattr(self.owner, "_depth30_linear_snapshot", None)
        timing = getattr(self.owner, "_depth30_prepared_timing", None)
        if timing is None or timing.snapshot != current:
            timing = getattr(self.owner, "_depth30_linear_timing", None)
        if (source_grant is not None
                and (current is not source_grant[0] or timing is not source_grant[1])):
            return reject("source_grant_replaced")
        try:
            kind, percent, grant_uid, stamp = linear
            ttl = ttl_reader(kind)  # Pure configuration lookup, no I/O.
            # Capture time after metadata. Only the side-effect-free cap
            # calculation below may follow before the physical write.
            now = time.monotonic()
            if (kind != ("forward" if mean_rpm > 0 else "backward")
                    or grant_uid != uid or self.owner._follow_controller.active_target_id != uid
                    or current is None
                    or len(current) != 4
                    or (current[0], current[2], current[3]) != (kind, uid, stamp)
                    or not all(math.isfinite(v) for v in (stamp, ttl, percent, current[1], now))
                    or stamp <= 0 or ttl <= 0 or not 0 <= now-stamp <= ttl
                    or not 0 < percent <= current[1] <= 100
                    or abs(mean_rpm) > percent*self.config.motor_forward_max_target_rpm/100.
                    or getattr(self.owner, "_depth30_continuation_veto", None) == (uid, stamp)):
                return reject("physical_grant_or_initial_limit")
            if timing is not None and timing.snapshot == current:
                if not math.isfinite(timing.depth_expires_at) or now > timing.depth_expires_at:
                    return reject("physical_depth_deadline")
                ff_expiry = timing.feedforward_expires_at
                if ff_expiry is not None and (not math.isfinite(ff_expiry)
                    or (now >= ff_expiry and abs(mean_rpm)
                            > timing.distance_only_percent*self.config.motor_forward_max_target_rpm/100.)):
                    return reject("feedforward_deadline")
            # The ordinary reader may have evaluated an age-dependent brake
            # cap before other safety callbacks consumed time. Recompute that
            # SAME grant's cap at the final physical clock, using only the
            # already sampled encoder data. Neither logging nor serial/cache
            # reads are permitted between this check and the speed write.
            due = getattr(self.owner, "_depth_forward_continuation_required", None)
            cap_reader = getattr(self.owner, "_depth_forward_continuation_limit", None)
            if kind == "forward" and callable(due) and due(current, timing, now):
                if not callable(cap_reader):
                    return reject("missing_continuation_limit")
                cap_percent, cap_reason = cap_reader(current, timing, now, feedback=feedback, quiet=True)
                if (not isinstance(cap_percent, (int, float))
                        or not math.isfinite(cap_percent) or cap_percent <= 0):
                    return reject("continuation_limit:%s" % cap_reason)
                limit_rpm = min(limit_rpm, cap_percent*self.config.motor_forward_max_target_rpm/100.)
            # The pure cap calculation still consumes a little time. A
            # producer may also replace the grant concurrently with it.
            # Recheck the canonical objects and any newly crossed cap/TTL
            # boundary immediately before send_targets, without logging.
            if commit_after_prepare:
                # The first (potentially rescheduled) budget evaluation is
                # done without excluding encoder reads. Only acquire I/O for
                # the second current-clock budget and final cache-only gates.
                if not self._begin_follow_commit():
                    return reject("commit_motor_owner_changed")
                if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                    self._follow_motor_call("send_stop", "follow_snapshot_commit_hard_stop", mode="emergency")
                    return reject("commit_hard_stop")
                cached = getattr(self, "_steering_feedback", None)
                if cached is not None and (feedback is None or cached.timestamp > feedback.timestamp):
                    self._snapshot_used_feedback(cached)
                    # A forward plan cannot inherit its earlier wheel guard
                    # across newly opposing measured motion. Non-opposing
                    # feedback is still charged by the second budget below.
                    if not self._ordinary_forward_feedback_eligible(uid, cached, time.monotonic()):
                        return reject("terminal_feedback_direction_changed")
                    feedback = cached
            checked_at = time.monotonic()
            latest = getattr(self.owner, "_depth30_linear_snapshot", None)
            latest_timing = getattr(self.owner, "_depth30_prepared_timing", None)
            if latest_timing is None or latest_timing.snapshot != latest:
                latest_timing = getattr(self.owner, "_depth30_linear_timing", None)
            if (latest is not current or latest_timing is not timing
                    or self.owner._follow_controller.active_target_id != uid
                    or (kind == "forward" and not wheel_feedback_valid(feedback, checked_at))
                    or checked_at-stamp > ttl
                    or (timing is not None and timing.snapshot == current
                        and (checked_at > timing.depth_expires_at
                             or (timing.feedforward_expires_at is not None
                                 and checked_at >= timing.feedforward_expires_at
                                 and abs(mean_rpm) > timing.distance_only_percent
                                     *self.config.motor_forward_max_target_rpm/100.)))):
                return reject("authority_or_feedback_changed_during_limit")
            if kind == "forward" and callable(due) and due(current, timing, checked_at):
                if not callable(cap_reader):
                    return reject("missing_continuation_limit")
                cap_percent, cap_reason = cap_reader(
                    current, timing, checked_at, feedback=feedback, quiet=True)
                if (not isinstance(cap_percent, (int, float))
                        or not math.isfinite(cap_percent) or cap_percent <= 0):
                    return reject("continuation_limit:%s" % cap_reason)
                limit_rpm = min(limit_rpm, cap_percent*self.config.motor_forward_max_target_rpm/100.)
            # Prepare the smaller pair BEFORE the terminal clock/STOP check.
            # No helper, cache reader, or logger may consume a lease after
            # that last gate. A current cap of zero remains a real veto.
            prepared_pair = (contract_forward_speed(forward_pair, limit_rpm)
                             if forward_pair is not None else None)
            if forward_pair is not None and prepared_pair is None:
                return reject("no_forward_pair_within_limit")
            if yaw_required:
                if intent_store is None:
                    # Legacy adapters expose only this predicate. Run their
                    # callback before the terminal physical-clock checks.
                    if not self.owner._has_fresh_lateral_yaw(uid):
                        return reject("terminal_yaw_expired")
                elif expected_intent is None:
                    return reject("terminal_yaw_missing")
            # The last quiet cap computation can itself consume time or see a
            # concurrent publisher. Recheck canonical authority and physical
            # clocks after it, immediately before the STOP/receipt veto.
            # snapshot() takes the intent-store lock: account for that wait
            # before sampling the physical clock, not after it.
            terminal_intent = self._follow_intent_snapshot(intent_store)
            terminal_now = time.monotonic()
            terminal_raw = getattr(self.owner, "_depth30_linear_snapshot", None)
            terminal_timing = getattr(self.owner, "_depth30_prepared_timing", None)
            if terminal_timing is None or terminal_timing.snapshot != terminal_raw:
                terminal_timing = getattr(self.owner, "_depth30_linear_timing", None)
            if (terminal_raw is not current or terminal_timing is not timing
                    or self.owner._follow_controller.active_target_id != uid
                    or not motion_identity_live(self.owner, uid, terminal_now)
                    or not 0 <= terminal_now-stamp <= ttl
                    or (kind == "forward" and not wheel_feedback_valid(feedback, terminal_now))
                    or (timing is not None and timing.snapshot == current
                        and (terminal_now > timing.depth_expires_at
                             or (timing.feedforward_expires_at is not None
                                 and terminal_now >= timing.feedforward_expires_at
                                 and abs(mean_rpm) > timing.distance_only_percent
                                     *self.config.motor_forward_max_target_rpm/100.)))):
                return reject("terminal_authority_or_feedback_expired")
            # Full visual acceptance has its own physical capture deadline.
            # A grey-frame continuation may finish its already bound Depth
            # sample, but must not authorize a newer sample. This forward-only
            # check follows BOTH quiet cap calculations; yaw-only low-quality
            # observation policies remain independent.
            if kind == "forward" and not self._forward_visual_permits(uid, stamp, terminal_now):
                return reject("terminal_visual_proof_revoked")
            if (yaw_required and expected_intent is not None
                    and (getattr(expected_intent, "target_id", None) != uid
                         or not expected_intent.valid(terminal_now)
                         or expected_intent.hold_zero
                         or getattr(self.owner, "_lateral_intent_zero_sequence", -1) == expected_intent.sequence
                         or not expected_intent.continuation_allowed(terminal_now, feedback))):
                return reject("terminal_yaw_expired")
            # A center/zero-yaw refresh has no authority over an equal-wheel
            # forward packet. A newly nonzero correction still requires a
            # bounded replan; only the explicit terminal fallback may drop it.
            neutral_straight = bool(neutral_straight
                and self._neutral_follow_yaw(uid, terminal_intent, terminal_now))
            if ((not independent_straight and not neutral_straight and (
                    getattr(self.owner, "_lateral_yaw_revision", None) != expected_revision
                    or terminal_intent is not expected_intent
                    or getattr(self.owner, "_lateral_turn_response_policy", None) != expected_policy))
                    or getattr(self.owner, "_depth30_continuation_veto", None) == (uid, stamp)
                    or getattr(self.owner, "_explicit_stop_requested", False)
                    or getattr(self.owner, "_runtime_shutdown_requested", False)
                    or getattr(self.owner, "stop_action_execution", False)
                    or getattr(self.owner, "person_detected_flag", False)
                    or getattr(self.owner, "_brake_hold_active", False)
                    or getattr(self.owner, "_near_yaw_park_request", None) is not None
                    or getattr(self, "_search_reacquire_brake_request", None) is not None
                    or getattr(self.owner, "running", True) is False
                    or getattr(self.owner, "search_state", "none") != "none"
                    or getattr(self.owner._follow_controller, "search_state", "none") != "none"
                    or getattr(self.backend, "motion_write_fault", None)
                    or getattr(self.backend, "parking_release_fault", None)
                    or bool(getattr(self.backend, "normal_zero_hold", False)) != prior_zero_hold
                    or getattr(self.backend, "stop_write_generation", 0) != prior_stop_generation
                    or (expected_receipt is not ...
                        and getattr(self.backend, "last_speed_receipt", None) is not expected_receipt)
                    or (expected_stop_generation is not ...
                        and getattr(self.backend, "stop_write_generation", 0)
                            != expected_stop_generation)
                    or getattr(self.backend, "parking_current_a", 0)
                    or getattr(self.backend, "_parking_current_uncertain", False)
                    or getattr(self.backend, "last_speed_receipt", None) is not prior_receipt):
                return reject("terminal_stop_or_publisher_changed")
            self._follow_write_feedback = feedback
            return LinearWriteDecision(limit_rpm, prepared_pair)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return reject("invalid_write_metadata")

    def forward_execution_anchor(self, uid, sample_timestamp, now):
        """Read completed I/O only; cannot keep an old depth grant alive.

        Used solely before PI processes a NEW qualified depth sample. A
        read-time continuation veto is not proof that a zero packet was sent.
        Any intervening speed transaction/STOP invalidates the backend receipt.
        """
        anchor = getattr(self, "_forward_execution_anchor", None)
        source_deadline = getattr(getattr(self.owner, "_depth30_linear_timing", None),
                                  "depth_expires_at", None)
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
                or not 0 <= now-sample_timestamp <= MAX_FORWARD_DEPTH_TTL_SEC
                or (source_deadline is not None and (
                    not isinstance(source_deadline, (int, float))
                    or isinstance(source_deadline, bool) or not math.isfinite(source_deadline)
                    or now > source_deadline))):
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

    def recent_forward_execution_anchor(self, uid, now):
        """Recent completed speed for a NEW depth sample's recovery calculation.

        The old sample may have passed its physical TTL. This is evidence of
        what the motor was last told to do, never a renewal of that sample or
        permission to move. The caller must independently qualify a fresh
        depth sample and recheck this exact receipt before admitting its new
        positive grant. Any zero, STOP or intervening speed write replaces
        the backend receipt and invalidates this observation.
        """
        anchor = getattr(self, "_forward_execution_anchor", None)
        receipt = getattr(self.backend, "last_speed_receipt", None)
        source = getattr(self.owner, "_depth30_linear_snapshot", None)
        if (anchor is None or receipt is None or receipt is not anchor.receipt
                or anchor.uid != uid or anchor.rpm <= 0
                or not all(math.isfinite(value) for value in (
                    now, anchor.sample_timestamp, anchor.sent_at, anchor.rpm))
                or not 0 <= now-anchor.sent_at <= .10
                or not (isinstance(source, tuple) and len(source) == 4
                        and source[0] == "forward" and source[1] > 0
                        and source[2] == uid and source[3] == anchor.sample_timestamp)
                or self.owner._follow_controller.active_target_id != uid
                or getattr(self.owner, "_vision_control_state", "") not in {
                    "target_visible", "target_visible_depth_valid"}
                or not self._visible_wheel_control_active()
                or getattr(self.owner, "stop_action_execution", False)
                or getattr(self.owner, "person_detected_flag", False)
                or getattr(self.owner, "_soft_stop_active", False)
                or getattr(self.owner, "_brake_hold_active", False)
                or getattr(self.owner, "_near_yaw_park_request", None) is not None
                or getattr(self, "_search_reacquire_brake_request", None) is not None
                or getattr(self.owner, "_search_handoff_uid", None) == uid
                or getattr(self.backend, "motion_write_fault", None)
                or getattr(self.backend, "parking_release_fault", None)
                or self.backend.normal_zero_hold or self.backend.parking_current_a
                or self.backend._parking_current_uncertain):
            return None
        # Recheck after all state and mode readers, before returning a proof
        # that another speed writer could otherwise have replaced meanwhile.
        if (getattr(self, "_forward_execution_anchor", None) is not anchor
                or getattr(self.backend, "last_speed_receipt", None) is not receipt
                or getattr(self.owner, "_depth30_linear_snapshot", None) is not source):
            return None
        return anchor

    def recovery_forward_execution_anchor(self, uid, now):
        """Completed command memory for a separately qualified NEW depth.

        Unlike the 100ms live-execution reader this may span a short expired
        depth lease, bounded by the existing 350ms PI memory window. It is not
        motion authority. In particular it does NOT cross a zero, STOP, reverse
        packet or an identity transition. Only an exact feedback-only
        withdrawal may retain completed-command memory after source removal;
        no removed grant is made live again. The controller must
        assess the new physical sample, feedback and braking budget itself.
        """
        try:
            anchor = getattr(self, "_forward_execution_anchor", None)
            receipt = getattr(self.backend, "last_speed_receipt", None)
            source = getattr(self.owner, "_depth30_linear_snapshot", None)
            source_deadline = getattr(getattr(self.owner, "_depth30_linear_timing", None),
                                      "depth_expires_at", None)
            stop_generation = getattr(self.backend, "stop_write_generation", None)
            controller = self.owner._follow_controller
            admitted = getattr(controller, "_distance_pi_admitted_grant", None)
            withdrawal = getattr(controller, "_distance_pi_grant_withdrawal", None)
            feedback_only_withdrawal = bool(
                source is None and isinstance(anchor, ForwardExecutionAnchor)
                and admitted == (uid, anchor.sample_timestamp)
                and isinstance(withdrawal, tuple) and len(withdrawal) == 3
                and withdrawal[:2] == admitted
                and withdrawal[2] in {
                    "lateral_depth:continuation_feedback_stale",
                    "lateral_depth:continuation_feedback_unavailable"})
            if (not isinstance(anchor, ForwardExecutionAnchor)
                    or type(uid) is not int or uid <= 0 or type(anchor.uid) is not int
                    or anchor.uid != uid
                    or type(stop_generation) is not int or stop_generation < 0
                    or receipt is None or receipt is not anchor.receipt
                    or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                               and math.isfinite(value) for value in (
                                   now, anchor.rpm, anchor.sample_timestamp, anchor.sent_at))
                    or anchor.rpm <= 0 or anchor.sample_timestamp <= 0
                    or not 0 <= anchor.sent_at-anchor.sample_timestamp <= MAX_FORWARD_DEPTH_TTL_SEC
                    or (source_deadline is not None and (
                        not isinstance(source_deadline, (int, float))
                        or isinstance(source_deadline, bool) or not math.isfinite(source_deadline)
                        or anchor.sent_at > source_deadline))
                    or not 0 <= now-anchor.sample_timestamp <= .35
                    or not anchor.sample_timestamp <= anchor.sent_at <= now
                    or receipt.completed_at != anchor.sent_at
                    or not (feedback_only_withdrawal or (
                            isinstance(source, tuple) and len(source) == 4
                            and source[0] == "forward" and source[1] > 0
                            and source[2] == uid and source[3] == anchor.sample_timestamp))
                    or self.owner._follow_controller.active_target_id != uid
                    or getattr(self.owner, "_vision_control_state", "") not in {
                        "target_visible", "target_visible_depth_valid"}
                    or not self._visible_wheel_control_active()
                    or getattr(self.owner, "stop_action_execution", False)
                    or getattr(self.owner, "person_detected_flag", False)
                    or getattr(self.owner, "_soft_stop_active", False)
                    or getattr(self.owner, "_brake_hold_active", False)
                    or getattr(self.owner, "_near_yaw_park_request", None) is not None
                    or getattr(self, "_search_reacquire_brake_request", None) is not None
                    or getattr(self.owner, "_search_handoff_uid", None) == uid
                    or getattr(self.backend, "motion_write_fault", None)
                    or getattr(self.backend, "parking_release_fault", None)
                    or self.backend.normal_zero_hold or self.backend.parking_current_a
                    or self.backend._parking_current_uncertain):
                return None
            if (getattr(self, "_forward_execution_anchor", None) is not anchor
                    or getattr(self.backend, "last_speed_receipt", None) is not receipt
                    or getattr(self.backend, "stop_write_generation", None) != stop_generation
                    or getattr(self.owner, "_depth30_linear_snapshot", None) is not source
                    or (feedback_only_withdrawal and (
                        getattr(controller, "_distance_pi_admitted_grant", None) != admitted
                        or getattr(controller, "_distance_pi_grant_withdrawal", None) != withdrawal))):
                return None
            return ForwardRecoveryAnchor(anchor, receipt, stop_generation, now)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None

    def forward_recovery_anchor_valid(self, uid, proof, now):
        """Revalidate memory at PI completion and new-depth admission."""
        if (not isinstance(proof, ForwardRecoveryAnchor)
                or type(proof.stop_generation) is not int or proof.stop_generation < 0
                or not isinstance(now, (int, float)) or isinstance(now, bool)
                or not math.isfinite(now)
                or not isinstance(proof.checked_at, (int, float))
                or isinstance(proof.checked_at, bool) or not math.isfinite(proof.checked_at)
                or not proof.checked_at <= now):
            return False
        current = self.recovery_forward_execution_anchor(uid, now)
        return bool(current is not None and current.executed is proof.executed
                    and current.current_receipt is proof.current_receipt
                    and current.stop_generation == proof.stop_generation)

    def _adopt_follow_revision_only(self, uid, linear, feedback):
        """One metadata-only update after the existing rebuild; no new plan."""
        reference = getattr(self, "_follow_revision_reference", None)
        planned = self._periodic_follow_axes
        if (getattr(self, "_follow_axes_rebuild_allowed", False)
                or getattr(self, "_follow_revision_coalesced", None) is not None
                or reference is None or len(linear or ()) != 4
                or linear != reference[0] or linear[0] != "forward" or linear[2] != uid
                or reference[1] is None
                or getattr(self.backend, "last_speed_receipt", None) is not reference[1]):
            return False
        now = time.monotonic()
        current = self._read_follow_axes(now)
        guard = self._visible_wheel_guard
        if (current is None or planned is None or current[0] != uid
                or current[0] != planned[0] or current[1] == planned[1]
                or current[2:] != planned[2:]
                or not all(math.isfinite(v) for v in current[2:])
                or current[2] <= 0 or current[2] < abs(current[3])
                or current[2] != linear[1]*self.config.motor_forward_max_target_rpm/100.
                or not self._ordinary_forward_feedback_eligible(uid, feedback, now)
                or guard.pending_signs is not None or guard.resume_signs is not None
                or guard.pending_full_reverse or guard.commanded_reverse
                or guard.residual_turn_signs is not None or guard.residual_forward_until > 0
                or self.backend.normal_zero_hold or self.backend.parking_current_a
                or self.backend._parking_current_uncertain or self.backend.parking_release_fault
                or getattr(self.owner, "_brake_hold_active", False)
                or getattr(self.owner, "_near_yaw_park_request", None) is not None
                or getattr(self, "_search_reacquire_brake_request", None) is not None
                or getattr(self.owner, "_search_handoff_uid", None) == uid
                or not self._periodic_follow_active()
                or self._read_depth_linear(uid, now=time.monotonic()) != linear
                or self._read_follow_axes(time.monotonic()) != current
                or getattr(self.backend, "last_speed_receipt", None) is not reference[1]):
            return False
        # Keep the exact wheel pair and physical grant. All ordinary wheel,
        # STOP, feedback and final authority checks below still run.
        self._periodic_follow_axes = current
        self._follow_revision_coalesced = (linear, reference[1], planned, current)
        return True

    def _expired_straight_intent_coalescible(self, uid, applied, current_axes,
                                           current_intent, old_intents, now):
        """An expired yaw owner cannot revoke an unchanged straight Depth axis.

        This only admits the existing one-shot final contraction checks. It
        neither retries nor permits an increase, a new grant, or a yaw write.
        """
        planned = getattr(self, "_periodic_follow_axes", None)
        old_intents = tuple(intent for intent in old_intents if intent is not None)
        source = getattr(self, "_follow_source_grant", None)
        receipt = getattr(self, "_follow_base_receipt_reference", None)
        if (current_intent is not None or not old_intents
                or planned is None or current_axes is None
                or planned[0] != uid or current_axes[0] != uid
                or planned[3] != 0 or current_axes[3] != 0
                or not 0 < current_axes[2] <= planned[2]
                or applied[0] != applied[1] or applied[0] <= 0
                or applied[0] > planned[2]
                or not self._same_follow_grant_marker(source, self._follow_grant_marker())
                or receipt is None or self.backend.last_speed_receipt is not receipt):
            return False
        return all(
            getattr(intent, "target_id", None) == uid
            and isinstance(getattr(intent, "valid_until", None), (int, float))
            and math.isfinite(intent.valid_until) and now > intent.valid_until
            and not any(getattr(intent, name, False) for name in (
                "park_requested", "forward_countersteer", "countersteer_rpm"))
            for intent in old_intents)

    def _follow_offlock_owner(self):
        return (self._follow_planning_offlock
                and self._follow_planning_thread_id == threading.get_ident())

    def _follow_plan_owns_motor(self):
        token = self._follow_plan_token
        return (token is not None
                and getattr(self.backend, "last_speed_receipt", None) is token[0]
                and getattr(self.backend, "stop_write_generation", 0) == token[1])

    def _follow_motor_call(self, method, *args, **kwargs):
        """Serialize physical I/O only; a newer writer/STOP always wins."""
        history_uid = kwargs.pop("history_uid", None)
        offlock = self._follow_offlock_owner()
        context = (self.owner.motor_io_lock
                   if offlock and not self._follow_commit_locked else nullcontext())
        wait_started = time.monotonic()
        with context:
            acquired_at = time.monotonic()
            try:
                if offlock and not self._follow_plan_owns_motor():
                    return False
                if (method in {"send_targets", "prepare_speed_mode"}
                        and self._distance_brake_stop_generation is not None
                        and self.backend.stop_write_generation != self._distance_brake_stop_generation):
                    return False  # Even an obsolete zero would leave the newer STOP mode.
                if method == "send_targets" and len(args) >= 2 and args[0] == args[1] == 0:
                    if self._distance_brake_episode is not None:
                        return False
                    proof = self._distance_brake_evidence(time.monotonic())
                    if (proof is not None and proof[2] == "shared_braking_momentum"
                            and self._distance_brake_forward_motion()):
                        self._start_distance_brake(proof[0], time.monotonic())
                        return False  # STOP is not a completed speed receipt.
                    # A zero computed from missing Depth, an obsolete pivot,
                    # or its deceleration wait is not a permanent STOP owner.
                    # Inspect the CURRENT publication at every normal zero
                    # entry, not a whitelist of old reason labels. Replanning
                    # happens only after releasing I/O, with the same guard.
                    axes = getattr(self, "_periodic_follow_axes", None)
                    current_grant = self._follow_grant_marker()
                    assessment = (getattr(current_grant[1], "braking_assessment", None)
                                  if current_grant is not None else None)
                    if (offlock and self._periodic_follow_writing
                            and not getattr(self, "_follow_snapshot_planning", False)
                            and not getattr(self, "_follow_terminal_forward_used", False)
                            and axes is not None and axes[2] >= 0
                            and isinstance(assessment, SampleBrakingAssessment)
                            and assessment.valid_for(axes[0], current_grant[0][3])
                            and not self._visible_wheel_guard.pending_full_reverse
                            and not self._visible_wheel_guard.commanded_reverse
                            and self._request_follow_commit_rebuild(
                                axes[0], getattr(self, "_follow_attempt_publication", None),
                                terminal_fallback=True)):
                        return False
                if method == "send_targets" and isinstance(self.backend, MssdMotorBackend):
                    kwargs["history_uid"] = history_uid
                getattr(self.backend, method)(*args, **kwargs)
                return True
            finally:
                if offlock and not self._follow_commit_locked:
                    self._follow_commit_lock_times.append(
                        ((acquired_at-wait_started)*1000.,
                         (time.monotonic()-acquired_at)*1000.))

    def _begin_follow_commit(self):
        if not self._follow_offlock_owner():
            return True  # Legacy/direct caller already owns motor_io_lock.
        if self._follow_commit_locked:
            return self._follow_plan_owns_motor()
        wait_started = time.monotonic()
        self._follow_commit_stack.enter_context(self.owner.motor_io_lock)
        acquired_at = time.monotonic()
        self._follow_commit_locked = True
        # The stack callback runs before releasing the lock.
        self._follow_commit_stack.callback(lambda: self._follow_commit_lock_times.append(
            ((acquired_at-wait_started)*1000.,
             (time.monotonic()-acquired_at)*1000.)))
        return self._follow_plan_owns_motor()

    def _follow_intent_snapshot(self, store):
        # LateralIntentStore publishes one immutable reference under its own
        # lock. During the serial commit, compare that reference directly:
        # waiting on its publisher would exclude the encoder again.
        if self._follow_commit_locked and isinstance(store, LateralIntentStore):
            return store._intent
        return store.snapshot() if store is not None else None

    def _neutral_follow_yaw(self, uid, intent, now):
        """Cache-only semantic zero, never a grant for identity or translation.

        The lateral producer can publish a new object/revision for every
        centered observation. Use its actual correction/withdrawal state so
        those equivalent zeros cannot revoke independently checked straight
        motion. Do not call the axes reader under serial ownership.
        """
        if intent is None:
            return isinstance(getattr(self.owner, "_lateral_intent_store", None), LateralIntentStore)
        if not isinstance(intent, LateralControlIntent) or intent.target_id != uid:
            return False
        if (intent.hold_zero or not intent.valid(now)
                or getattr(self.owner, "_lateral_intent_zero_sequence", -1) == intent.sequence):
            return True
        correction = (getattr(self.owner, "_lateral_intent_last_correction_rpm", None)
            if getattr(self.owner, "_lateral_intent_last_sequence", -1) == intent.sequence
            else intent.initial_correction_rpm)
        return isinstance(correction, (int, float)) and correction == 0

    def _ordinary_forward_feedback_eligible(self, uid, feedback, now, *, hold_existing=False):
        """One forward feedback contract, including a narrowly bounded tail.

        Keep the wheel guard's existing 1 RPM quantization tolerance. A
        +/-3 RPM all-wheel tail matches the shared braking input contract,
        but only a live distance-only grant may use it. It never proves that
        the wheels are parked and cannot bypass actual reversal/STOP owners.
        Retaining a previous packet remains stricter than planning a new one:
        signed feedback there requires the all-wheel quiet-tail proof.
        """
        if not wheel_feedback_valid(feedback, now):
            return False
        if min(feedback.left_forward_rpm, feedback.right_forward_rpm) >= (0. if hold_existing else -1.):
            return True
        guard = self._visible_wheel_guard
        if not guard.quiet_forward_tail(feedback, now):
            return False
        if not all(hasattr(self.backend, name) for name in (
                "normal_zero_hold", "parking_current_a", "_parking_current_uncertain",
                "parking_release_fault")):
            # Legacy adapters have no proof of parking ownership. Keep their
            # old full guard instead of inventing eligibility for this tail.
            return False
        controller = self.owner._follow_controller
        marker = self._follow_grant_marker()
        if (marker is None or len(marker[0]) != 4
                or getattr(getattr(controller, "cfg", None), "distance_target_motion_control_enable", True)
                or controller.active_target_id != uid
                or not self._visible_wheel_control_active()
                or not motion_identity_live(self.owner, uid, now)
                or any(getattr(self.owner, name, False) for name in (
                    "stop_action_execution", "person_detected_flag"))
                or getattr(self.owner, "_search_handoff_uid", None) == uid
                or getattr(self.owner, "_search_handoff_moving_active", False)
                or getattr(self, "_search_reacquire_brake_request", None) is not None
                or getattr(self, "_distance_brake_episode", None) is not None
                or self.backend.normal_zero_hold or self.backend.parking_current_a
                or self.backend._parking_current_uncertain
                or self.backend.parking_release_fault
                or getattr(self.backend, "motion_write_fault", None)):
            return False
        raw, timing = marker
        assessment = getattr(timing, "braking_assessment", None)
        try:
            return bool(raw[0] == "forward" and raw[2] == uid
                and 0 < raw[1] <= 100
                and isinstance(assessment, SampleBrakingAssessment)
                and assessment.valid_for(uid, raw[3])
                and 0 < raw[3] <= now <= timing.depth_expires_at
                and now-raw[3] <= self.owner._depth_linear_max_age_sec("forward")
                and getattr(self.owner, "_depth30_continuation_veto", None) != (uid, raw[3])
                and tuple((getattr(self.owner, "_depth30_read_veto", None) or ())[:2]) != (uid, raw[3])
                and self._forward_visual_permits(uid, raw[3], now))
        except (TypeError, ValueError, AttributeError, OverflowError):
            return False

    def _ordinary_snapshot_scope(self, uid, pair, feedback, now):
        """Only unified-budget, ordinary forward motion can use short commit.

        Legacy adapters without a physical sample assessment, reverse/pivot
        and search handoff keep their existing full planner. A pending pivot
        or resume flag is not itself a revocation of a new forward curve:
        the SAME wheel guard still qualifies that curve before any write.
        Shared feedback qualification admits only a guarded all-wheel tail.
        This predicate grants no motion.
        """
        marker = self._follow_grant_marker()
        if (not self._periodic_follow_writing or not self._follow_offlock_owner()
                or marker is None or len(marker[0]) != 4
                or marker[0][0] != "forward" or marker[0][2] != uid
                or min(pair) < 0 or sum(pair) <= 0
                or not self._ordinary_forward_feedback_eligible(uid, feedback, now)
                or getattr(self.owner, "_search_handoff_uid", None) == uid
                or getattr(self.owner, "_search_handoff_moving_active", False)):
            return False
        assessment = getattr(marker[1], "braking_assessment", None)
        guard = self._visible_wheel_guard
        return bool(isinstance(assessment, SampleBrakingAssessment)
            and assessment.valid_for(uid, marker[0][3])
            and not guard.pending_full_reverse and not guard.commanded_reverse)

    def _release_follow_commit_for_refresh(self):
        """Release serial ownership BEFORE latest-state readers or wheel work."""
        if self._follow_commit_locked:
            self._follow_commit_stack.close()
            self._follow_commit_locked = False

    def _snapshot_zero_evidence(self):
        """Immutable/cache-only inputs which produced one snapshot attempt."""
        return (self._follow_grant_marker(),
                getattr(self.owner, "_lateral_yaw_revision", None),
                self._follow_intent_snapshot(getattr(self.owner, "_lateral_intent_store", None)),
                getattr(self.owner, "_lateral_turn_response_policy", None),
                getattr(self, "_steering_feedback", None))

    def _snapshot_used_feedback(self, feedback):
        """Remember the actual feedback checked, not just the entry cache.

        Encoder publication can change during off-lock planning. A later
        failure must retain the rejected sample so a fresh cache at zero
        commit can trigger bounded readmission. This does not change the
        attempt's grant/publication or turn feedback refresh into permission.
        """
        state = getattr(self, "_follow_snapshot_zero_state", None)
        if getattr(self, "_follow_snapshot_planning", False) and state is not None:
            self._follow_snapshot_zero_state = (*state[:4], feedback)

    def _snapshot_zero_superseded(self, uid):
        """A newer plausible forward state requires admission, not permission.

        Runs at zero commit without axes readers, serial reads or wheel work.
        All snapshot exits share the attempt's evidence; no call site may omit
        the grant/feedback which produced its unsent zero. A subsequent plan
        must independently check both wheels and every physical deadline.
        A released, still-owned hardware parking hold can also be readmitted;
        the new plan must perform and recheck its own speed-mode transition.
        Active braking, a newer STOP and parking faults cannot use this path.
        """
        old = self._follow_snapshot_zero_state
        if (old is None or not self._follow_plan_owns_motor()
                or not self._visible_wheel_control_active()
                or self.owner._follow_controller.active_target_id != uid
                or not motion_identity_live(self.owner, uid, time.monotonic())
                or any(getattr(self.owner, name, False) for name in (
                    "_brake_hold_active", "_explicit_stop_requested", "_runtime_shutdown_requested",
                    "stop_action_execution", "person_detected_flag"))
                or getattr(self.owner, "_near_yaw_park_request", None) is not None
                or getattr(self, "_search_reacquire_brake_request", None) is not None
                or self._distance_brake_episode is not None
                or (self._distance_brake_stop_generation is not None
                    and self.backend.stop_write_generation != self._distance_brake_stop_generation)
                or getattr(self.backend, "_parking_current_uncertain", False)
                or getattr(self.backend, "parking_release_fault", None)
                or getattr(self.backend, "motion_write_fault", None)
                or self._visible_wheel_guard.pending_full_reverse
                or self._visible_wheel_guard.commanded_reverse):
            return False
        current = self._snapshot_zero_evidence()
        marker, revision, intent, policy, feedback = current
        if marker is None:
            return False
        raw, timing = marker
        now = time.monotonic()
        assessment = getattr(timing, "braking_assessment", None)
        try:
            if (len(raw) != 4 or raw[0] != "forward" or raw[2] != uid
                    or not all(math.isfinite(v) for v in (raw[1], raw[3], now))
                    or not 0 < raw[1] <= 100 or not 0 < raw[3] <= now
                    or now > timing.depth_expires_at
                    or raw[3] <= self._distance_brake_sample_floor
                    or now-raw[3] > self.owner._depth_linear_max_age_sec("forward")
                    or not isinstance(assessment, SampleBrakingAssessment)
                    or not assessment.valid_for(uid, raw[3])
                    or not self._ordinary_forward_feedback_eligible(uid, feedback, now)
                    or (old[0] is not None and (old[0][0][2] != uid
                        or raw[3] < old[0][0][3]))):
                return False
        except (TypeError, ValueError, IndexError, AttributeError, OverflowError):
            return False
        return bool(not self._same_follow_grant_marker(old[0], marker)
            or old[1] != revision or old[2] is not intent or old[3] != policy
            or (old[4] is not feedback
                and not self._ordinary_forward_feedback_eligible(uid, old[4], now)
                and (not wheel_feedback_valid(old[4], now)
                     or feedback.timestamp > old[4].timestamp)))

    def _ordinary_snapshot_stop(self, uid, reason, *, allow_refresh=True):
        """Return None only to retry an obsolete zero within the caller's budget.

        False is terminal (stopped or a newer STOP owns I/O). Neither outcome
        grants motion; a refresh always reruns full independent admission.
        """
        # A real STOP/parking writer retains its mode. A normal same-owner
        # expired/invalid command must be revoked, never silently held.
        if not self._begin_follow_commit():
            return False
        if allow_refresh and self._snapshot_zero_superseded(uid):
            self.logger.info("follow_snapshot_zero_superseded uid=%s reason=%s "
                "next_attempt=%s straight_used=%s packet_written=False "
                "motion_authorized=False deadline_renewed=False", uid, reason,
                self._follow_snapshot_next_attempt, getattr(self, "_follow_terminal_forward_used", False))
            return None
        if not (getattr(self.owner, "_brake_hold_active", False)
                or getattr(self.owner, "_near_yaw_park_request", None) is not None
                or getattr(self.owner, "_explicit_stop_requested", False)
                or getattr(self.owner, "_runtime_shutdown_requested", False)
                or getattr(self.backend, "normal_zero_hold", False)):
            previous_receipt = getattr(self.backend, "last_speed_receipt", None)
            previous_sent = self._visible_wheel_guard.last_sent
            with self._zero_packet_decision("snapshot_admission", reason):
                written = self._follow_motor_call("send_targets", 0, 0, "FOLLOW_SNAPSHOT_REVOKED",
                                                   history_uid=uid)
            if written:
                # A real completed zero belongs to the same execution ledger
                # as a positive packet. Never synthesize a receipt when a
                # concurrent STOP owns I/O or the backend skipped the write.
                receipt = getattr(self.backend, "last_speed_receipt", None)
                if (isinstance(receipt, MotorSpeedReceipt) and receipt is not previous_receipt
                        and receipt.left_rpm == receipt.right_rpm == 0):
                    self._note_follow_packet(
                        uid, (0, 0), None, getattr(self, "_steering_feedback", None),
                        (1, 1), previous_receipt, previous_sent, packet_written=True,
                        trial=getattr(self, "_turn_response_trial", None))
        self._forward_execution_anchor = None
        self._visible_wheel_guard.reset()
        self.logger.info("follow_wheel_snapshot_veto uid=%s reason=%s", uid, reason)
        return False

    def _send_ordinary_snapshot_forward(self, uid, label, *, max_target_override=None,
                                        straight_fallback=False):
        previous = getattr(self, "_follow_snapshot_planning", False)
        previous_state = self._follow_snapshot_zero_state
        if not previous:
            self._follow_snapshot_next_attempt = getattr(self, "_follow_service_attempt", 0)
        self._follow_snapshot_planning = True
        try:
            while True:
                result = self._plan_ordinary_snapshot_forward(uid, label,
                    max_target_override=max_target_override, straight_fallback=straight_fallback)
                if result is not None:
                    return result
                # Every zero exit has the same None=readmit contract, including
                # direct terminal returns and the exhausted-loop stop. Never
                # restart the three-attempt budget or do planning under I/O.
                self._release_follow_commit_for_refresh()
                if not straight_fallback and self._follow_snapshot_next_attempt < 3:
                    continue
                if not straight_fallback and not getattr(self, "_follow_terminal_forward_used", False):
                    straight_fallback = True
                    continue
                # Finalize, without another retry: only a still-legal ACTUAL
                # completed packet may remain in force; otherwise revoke it.
                if self._current_receipt_survives_publication(uid, max_target_override=max_target_override):
                    return False
                return self._ordinary_snapshot_stop(uid, "snapshot_readmission_exhausted", allow_refresh=False)
        finally:
            self._follow_snapshot_planning = previous
            self._follow_snapshot_zero_state = previous_state

    def _plan_ordinary_snapshot_forward(self, uid, label, *, max_target_override=None,
                                        straight_fallback=False):
        """One current forward snapshot, one wheel guard, one terminal gate.

        Ordinary publications are refreshes, not revocations. They never run
        the full legacy handoff cascade again. Three fused planning attempts
        are shared with the service. If only the old command/yaw is obsolete,
        at most one independent straight admission can follow; it must pass
        all current authority checks and cannot extend a sample deadline.
        """
        first_attempt = 0 if straight_fallback else self._follow_snapshot_next_attempt
        last_reason = "publication_kept_changing"
        if straight_fallback:
            self._follow_terminal_forward_used = True
        for attempt in range(first_attempt, 1 if straight_fallback else 3):
            self._release_follow_commit_for_refresh()
            if not straight_fallback:
                self._follow_snapshot_next_attempt = attempt+1
            self._follow_snapshot_zero_state = self._snapshot_zero_evidence()
            if not self._follow_plan_owns_motor():
                return False
            if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                self._follow_motor_call("send_stop", "follow_snapshot_hard_stop", mode="emergency")
                self._visible_wheel_guard.reset()
                return False
            if (not self._visible_wheel_control_active()
                    or self.owner._follow_controller.active_target_id != uid
                    or not motion_identity_live(self.owner, uid, time.monotonic())):
                return self._ordinary_snapshot_stop(uid, "identity_or_control_revoked")
            marker = self._follow_grant_marker()
            axes = self._read_follow_axes(time.monotonic())
            linear = self._read_depth_linear(uid, now=time.monotonic())
            store = getattr(self.owner, "_lateral_intent_store", None)
            intent = self._follow_intent_snapshot(store)
            policy = getattr(self.owner, "_lateral_turn_response_policy", None)
            feedback = self.get_steering_feedback()
            self._snapshot_used_feedback(feedback)
            now = time.monotonic()
            if not self._same_follow_grant_marker(marker, self._follow_grant_marker()):
                last_reason = "depth_published_during_snapshot"
                continue
            if (axes is None or len(axes) != 4 or axes[0] != uid
                    or len(linear or ()) != 4 or linear[0] != "forward"
                    or linear[2] != uid or marker is None
                    or linear[2:] != marker[0][2:]
                    or not all(math.isfinite(v) for v in (*axes[2:], linear[1]))
                    or axes[2] <= 0 or linear[1] <= 0):
                stopped = self._ordinary_snapshot_stop(
                    uid, "forward_authority_revoked")
                if stopped is None:
                    last_reason = "revocation_superseded_before_commit"
                    continue
                return stopped
            if (not straight_fallback and axes[1] != getattr(self.owner, "_lateral_yaw_revision", None)
                    and not (axes[3] == 0 and self._neutral_follow_yaw(
                        uid, self._follow_intent_snapshot(store), now))):
                last_reason = "yaw_published_during_snapshot"
                continue
            policy_complete = (policy is None or (isinstance(policy, tuple) and len(policy) == 2
                and isinstance(intent, LateralControlIntent)
                and policy[0] == intent.sequence and isinstance(policy[1], bool)))
            base = min(axes[2], linear[1]*self.config.motor_forward_max_target_rpm/100.)
            cap = min(self.config.motor_forward_max_target_rpm,
                      self.config.motor_forward_max_target_rpm
                      if max_target_override is None else max_target_override)
            base = min(base, cap)
            yaw = axes[3]
            if not policy_complete or not self.owner._has_fresh_lateral_yaw(uid):
                # Clearing/centering yaw is a legal state even while its
                # auxiliary boost policy still names the previous generation.
                # An incomplete turn cannot authorize yaw, but cannot veto
                # independently qualified straight Depth motion either.
                yaw = 0.
            cfg = getattr(self.owner._follow_controller, "cfg", None)
            if getattr(cfg, "visible_steering_pid_image_error_only", False):
                yaw = clamp_correction(yaw, effective_correction_limit(
                    cfg, base, near_distance=bool(intent and intent.near_distance_mode),
                    policy_limit=intent.correction_limit_rpm if intent else None))
            yaw = math.copysign(min(abs(yaw), max(0., cap-base)), yaw)
            if straight_fallback:
                if (abs(yaw) > base and intent is not None and intent.target_id == uid
                        and not intent.hold_zero and intent.valid(now)
                        and intent.continuation_allowed(now, feedback)
                        and getattr(self.owner, "_lateral_intent_zero_sequence", -1) != intent.sequence):
                    # Actual single-wheel reversal is not ordinary yaw churn;
                    # keep its dedicated guard rather than substituting drive.
                    return self._ordinary_snapshot_stop(uid, "independent_straight_reversal_requested")
                yaw = 0.
            trial = getattr(self, "_turn_response_trial", None)
            if trial is not None and not straight_fallback:
                base, yaw, _ = trial.adjust(base, yaw, intent, feedback, now, uid,
                    getattr(cfg, "visible_steering_pid_execution_response_trial_sec", 0.))
            pair = (int(round(base+yaw)), int(round(base-yaw)))
            # Integer wheel packets cannot exceed the real-valued common
            # speed grant merely because round(94.6) == 95. Contract before
            # the guard/final admission; do not weaken the grant checker.
            if min(pair) >= 0 and .5*sum(pair) > base:
                pair = (math.floor(base+yaw), math.floor(base-yaw))
            if not self._ordinary_snapshot_scope(uid, pair, feedback, now):
                stopped = self._ordinary_snapshot_stop(
                    uid, "nonordinary_or_feedback_changed")
                if stopped is None:
                    last_reason = "revocation_superseded_before_commit"
                    continue
                return stopped
            # No stale visual turn is retained when only Depth is current.
            # park_requested by itself is a yaw-brake hint. Actual whole-car
            # parking is represented by _near_yaw_park_request/STOP ownership.
            if pair[0] != pair[1] and (intent is not None and (
                    intent.target_id != uid or intent.hold_zero
                    or not intent.valid(now)
                    or not intent.continuation_allowed(now, feedback)
                    or getattr(self.owner, "_lateral_intent_zero_sequence", -1) == intent.sequence)):
                pair = (int(math.floor(base)),) * 2
            guarded, guard_reason = self._visible_wheel_guard.limit(
                pair, feedback, now,
                allow_quiet_forward_tail=self._ordinary_forward_feedback_eligible(uid, feedback, now),
                allow_forward_handoff=bool(getattr(self.config, "follow_forward_handoff_enable", False)),
                residual_reverse_max_rpm=getattr(self.config, "follow_residual_reverse_max_rpm", 0.0))
            if min(guarded) < 0 or sum(guarded) <= 0:
                stopped = self._ordinary_snapshot_stop(uid, guard_reason)
                if stopped is None:
                    last_reason = "revocation_superseded_before_commit"
                    continue
                return stopped
            publication = (marker, axes[1], intent, policy)
            self._follow_source_grant = marker
            self._periodic_follow_axes = (uid, axes[1], .5*sum(guarded),
                                          .5*(guarded[0]-guarded[1]))
            previous_receipt = getattr(self.backend, "last_speed_receipt", None)
            stop_generation = getattr(self.backend, "stop_write_generation", 0)
            if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                self._follow_motor_call("send_stop", "follow_snapshot_commit_hard_stop", mode="emergency")
                self._visible_wheel_guard.reset()
                return False
            if (getattr(self.backend, "normal_zero_hold", False)
                    or getattr(self.backend, "parking_current_a", 0.)
                    or getattr(self.backend, "_parking_current_uncertain", False)):
                if not self._begin_follow_commit():
                    return False
                if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                    self._follow_motor_call("send_stop", "parking_exit_hard_stop", mode="emergency")
                    self._visible_wheel_guard.reset()
                    return False
                # The shortcut precedes the legacy planner's parking exit.
                # A released parking episode can leave its hardware 10A hold
                # behind; terminal admission must not veto that hold forever
                # without first performing the qualified 0A transition.
                # Clearing current alone never grants motion. Recheck every
                # physical deadline, publisher and STOP token after its I/O.
                parking_now = time.monotonic()
                if (not self._visible_wheel_control_active()
                        or self.owner._follow_controller.active_target_id != uid
                        or not motion_identity_live(self.owner, uid, parking_now)
                        or getattr(self.owner, "stop_action_execution", False)
                        or getattr(self.owner, "person_detected_flag", False)
                        or self._near_yaw_park_blocks_write(label)):
                    return False
                if not self._same_follow_grant_marker(marker, self._follow_grant_marker()):
                    last_reason = "depth_published_during_snapshot"
                    continue
                cached = getattr(self, "_steering_feedback", None)
                if cached is not None and (feedback is None or cached.timestamp > feedback.timestamp):
                    feedback = cached
                self._snapshot_used_feedback(feedback)
                if (not 0 <= parking_now-linear[3]
                            <= self.owner._depth_linear_max_age_sec("forward")
                        or parking_now > marker[1].depth_expires_at
                        or not self._forward_visual_permits(uid, linear[3], parking_now)
                        or not wheel_feedback_valid(feedback, parking_now)):
                    stopped = self._ordinary_snapshot_stop(uid, "parking_exit_authority_expired")
                    if stopped is None:
                        last_reason = "revocation_superseded_before_commit"
                        continue
                    return stopped
                if not self._follow_motor_call("prepare_speed_mode"):
                    return False
                if not self._follow_plan_owns_motor():
                    return False  # A STOP during release owns the motor mode.
                if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                    self._follow_motor_call("send_stop", "parking_exit_hard_stop", mode="emergency")
                    return False
            cached = getattr(self, "_steering_feedback", None)
            if cached is not None and cached.timestamp > feedback.timestamp:
                feedback = cached
            self._snapshot_used_feedback(feedback)
            if not self._ordinary_forward_feedback_eligible(uid, feedback, time.monotonic()):
                stopped = self._ordinary_snapshot_stop(
                    uid, "terminal_feedback_invalid")
                if stopped is None:
                    last_reason = "revocation_superseded_before_commit"
                    continue
                return stopped
            decision = self._linear_packet_write_limit(
                linear, uid, .5*sum(guarded), source_grant=marker,
                feedback=feedback, forward_pair=guarded, planned_revision=axes[1],
                yaw_required=guarded[0] != guarded[1], adopted_intent=intent,
                adopted_policy=policy, expected_receipt=previous_receipt,
                expected_stop_generation=stop_generation,
                independent_straight=straight_fallback, commit_after_prepare=True)
            if decision is None:
                latest = (self._follow_grant_marker(),
                    getattr(self.owner, "_lateral_yaw_revision", None),
                    self._follow_intent_snapshot(store),
                    getattr(self.owner, "_lateral_turn_response_policy", None))
                changed = (not self._same_follow_grant_marker(publication[0], latest[0])
                    or publication[1] != latest[1] or publication[2] is not latest[2]
                    or publication[3] != latest[3])
                if (changed or self._follow_write_veto_reason == "terminal_yaw_expired") \
                        and self._follow_plan_owns_motor():
                    # Pure time expiry has no publisher/revision change.
                    # Rebuild without stale yaw; Depth/visual/feedback and
                    # STOP still pass their own complete terminal gate.
                    last_reason = ("publication_changed_at_terminal" if changed
                                   else "yaw_expired_at_terminal")
                    continue
                return self._ordinary_snapshot_stop(uid, self._follow_write_veto_reason)
            applied = decision.forward_pair
            feedback = self._follow_write_feedback
            if applied is None or sum(applied) <= 0:
                return self._ordinary_snapshot_stop(uid, "no_positive_packet_within_budget")
            ls = self.backend.wheel_raw_state_to_target("left", 1, 0x01)
            rs = self.backend.wheel_raw_state_to_target("right", 1, 0x01)
            previous_sent = self._visible_wheel_guard.last_sent
            if not self._follow_motor_call("send_targets", applied[0]*ls, applied[1]*rs,
                                           label, max_target_override=max_target_override,
                                           history_uid=uid):
                return False
            sent_at, _ = self._note_follow_packet(
                uid, applied, linear, feedback, (ls, rs), previous_receipt,
                previous_sent, packet_written=True, trial=trial)
            self._periodic_follow_axes = (uid, axes[1], .5*sum(applied),
                                          .5*(applied[0]-applied[1]))
            self._visible_wheel_waiting = False
            self._visible_wheel_feedback_ts = feedback.timestamp
            self.logger.info("visible_wheel_dispatch uid=%s label=%s base=%.1f yaw=%.1f "
                "requested_forward_rpm=%s applied_forward_rpm=%s reason=snapshot_forward "
                "depth_fresh=True feedback_forward_rpm=%s feedback_ts=%s "
                "feedback_age_ms=%.1f evidence_capture_frame_id=%s packet_written=True "
                "sent_ts=%.6f snapshot_attempts=%d full_plan_repeated=False straight_fallback=%s",
                uid, label, .5*sum(applied), .5*(applied[0]-applied[1]), pair, applied,
                (feedback.left_forward_rpm, feedback.right_forward_rpm), feedback.timestamp,
                (sent_at-feedback.timestamp)*1000., getattr(self.owner, "_last_command_capture_frame", None),
                sent_at, attempt-first_attempt+1, straight_fallback)
            return True
        if (not straight_fallback and last_reason in {"publication_changed_at_terminal", "depth_published_during_snapshot",
                            "yaw_published_during_snapshot", "yaw_expired_at_terminal", "revocation_superseded_before_commit"}
                and self._current_receipt_survives_publication(
                    uid, max_target_override=max_target_override)):
            # No packet was sent and no execution/sample clock is renewed.
            # The next service remains due; it will check all clocks again.
            return False
        if (not straight_fallback and not getattr(self, "_follow_terminal_forward_used", False)
                and (getattr(self, "_follow_defer_reject_reason", None) == "forward_replan_needed"
                     or last_reason == "revocation_superseded_before_commit")
                and last_reason in {"publication_changed_at_terminal", "depth_published_during_snapshot",
                                    "yaw_published_during_snapshot", "yaw_expired_at_terminal", "revocation_superseded_before_commit"}):
            # Failure to retain an old receipt says nothing about admission
            # of the current forward grant. The old packet may be zero/a
            # pivot, or its execution clock may have been cleared on loss.
            # One final independent straight plan reruns every current gate,
            # including reversal/STOP ownership, and cannot loop again.
            return self._send_ordinary_snapshot_forward(uid, label,
                max_target_override=max_target_override, straight_fallback=True)
        return self._ordinary_snapshot_stop(uid, last_reason)

    def _current_receipt_survives_publication(self, uid, *, max_target_override=None):
        """One final check of an ALREADY sent pair, not a new wheel plan.

        Publication churn alone is not a STOP request. If the current grant
        independently still permits exactly the completed pair, leave it in
        force until the next due service. A lower cap, changed turn direction,
        expired measurement, wheel reversal or any STOP makes this fail. No
        deadline, receipt or acceleration history is updated here.
        """
        self._follow_defer_reject_reason = "evidence_unqualified"
        self._release_follow_commit_for_refresh()
        # Holding the receipt is itself a new assessment. Do not compare its
        # eventual zero against an older failed attempt and mislabel an owner
        # or feedback veto as a publication that has not yet been examined.
        self._follow_snapshot_zero_state = self._snapshot_zero_evidence()
        if not self._follow_plan_owns_motor():
            return False
        receipt = getattr(self.backend, "last_speed_receipt", None)
        last_axes = self._follow_wheel_clock.last_axes
        if (not isinstance(receipt, MotorSpeedReceipt)
                or (last_axes is not None and last_axes[0] != uid)
                or not self._visible_wheel_control_active()
                or self.owner._follow_controller.active_target_id != uid):
            return False
        ls = self.backend.wheel_raw_state_to_target("left", 1, 0x01)
        rs = self.backend.wheel_raw_state_to_target("right", 1, 0x01)
        pair = (receipt.left_rpm*ls, receipt.right_rpm*rs)
        submission = getattr(self.backend, "last_speed_write", None)
        known_uid = bool(last_axes is not None or (
            isinstance(submission, MotorSpeedWrite) and submission.uid == uid
            and submission.completed_receipt is receipt
            and submission.stop_generation == getattr(self.backend, "stop_write_generation", 0)))
        if (pair == (0, 0) or (receipt is self._follow_wheel_last_receipt and known_uid
                and sum(pair) >= 0 and (last_axes is None or sum(pair) == 0))):
            # There is no positive forward packet to retain after a zero,
            # pivot, or cleared execution clock. This is NOT a failed current
            # authority check: let the single independent admission assess
            # the new grant. A reverse packet, another UID or an unowned
            # nonzero receipt cannot acquire this recovery classification.
            # Zero carries no motion/UID permission; the plan's receipt/STOP
            # token still prevents overwriting a writer that intervenes now.
            self._follow_defer_reject_reason = "forward_replan_needed"
            return False
        if receipt is not self._follow_wheel_last_receipt or last_axes is None:
            return False
        marker = self._follow_grant_marker()
        axes = self._read_follow_axes(time.monotonic())
        linear = self._read_depth_linear(uid, now=time.monotonic())
        feedback = self.get_steering_feedback()
        self._snapshot_used_feedback(feedback)
        store = getattr(self.owner, "_lateral_intent_store", None)
        intent = self._follow_intent_snapshot(store)
        policy = getattr(self.owner, "_lateral_turn_response_policy", None)
        base, yaw = .5*sum(pair), .5*(pair[0]-pair[1])
        motor_cap = min(self.config.motor_forward_max_target_rpm,
                        self.config.motor_forward_max_target_rpm
                        if max_target_override is None else max_target_override)
        now = time.monotonic()
        if (marker is None or not self._same_follow_grant_marker(marker, self._follow_grant_marker())
                or axes is None or len(axes) != 4 or axes[0] != uid
                or len(linear or ()) != 4 or linear[0] != "forward" or linear[2:] != marker[0][2:]
                or not all(isinstance(v, (int, float)) and math.isfinite(v)
                           for v in (*axes[2:], linear[1]))
                or base <= 0 or axes[2] <= 0 or linear[1] <= 0
                or max(pair) > motor_cap
                or not self._ordinary_snapshot_scope(uid, pair, feedback, now)):
            return False
        if (base > min(axes[2], linear[1]*self.config.motor_forward_max_target_rpm/100.)
                or (yaw and (not self.owner._has_fresh_lateral_yaw(uid)
                    or not isinstance(intent, LateralControlIntent)
                    or (policy is not None and (not isinstance(policy, tuple)
                        or len(policy) != 2 or policy[0] != intent.sequence
                        or not isinstance(policy[1], bool)))
                    or axes[3]*yaw <= 0 or abs(yaw) > abs(axes[3])
                    or abs(yaw) > intent.correction_limit_rpm))):
            # Old yaw/cap cannot be retained, but an independently checked
            # new straight packet may be possible. Fault/owner/feedback
            # failures above and below never qualify this fallback.
            self._follow_defer_reject_reason = "forward_replan_needed"
            return False
        stop_generation = getattr(self.backend, "stop_write_generation", 0)
        if not self._begin_follow_commit():
            return False
        if self.hard_stop_check(getattr(self.owner, "current_command", None)):
            self._follow_motor_call("send_stop", "follow_defer_hard_stop", mode="emergency")
            return False
        cached = self._steering_feedback
        if cached is not None and cached.timestamp > feedback.timestamp:
            feedback = cached
        self._snapshot_used_feedback(feedback)
        if not self._ordinary_forward_feedback_eligible(uid, feedback, time.monotonic(), hold_existing=True):
            return False
        decision = self._linear_packet_write_limit(
            linear, uid, base, source_grant=marker, feedback=feedback,
            forward_pair=pair, planned_revision=axes[1], yaw_required=yaw != 0,
            adopted_intent=intent, adopted_policy=policy, expected_receipt=receipt,
            expected_stop_generation=stop_generation)
        latest_feedback = self._steering_feedback
        feedback_preserved = latest_feedback is feedback
        if not feedback_preserved and wheel_feedback_valid(latest_feedback, time.monotonic()):
            # The encoder publishes a new immutable object each poll. Equal
            # or lower non-opposing speed does not invalidate a stopping
            # budget already checked against the higher outer/body speed.
            # A faster, reversing, older, errored or stale report still vetoes
            # holding this old packet; never rebaseline its travel bound.
            old_wheels = (feedback.left_forward_rpm, feedback.right_forward_rpm)
            new_wheels = (latest_feedback.left_forward_rpm, latest_feedback.right_forward_rpm)
            feedback_preserved = bool(
                latest_feedback.timestamp >= feedback.timestamp
                and self._ordinary_forward_feedback_eligible(uid, latest_feedback, time.monotonic(), hold_existing=True)
                and max(map(abs, new_wheels)) <= max(map(abs, old_wheels))
                and max(0., sum(new_wheels)) <= max(0., sum(old_wheels))
                and (not yaw or (intent is not None
                    and intent.continuation_allowed(time.monotonic(), latest_feedback))))
        if (decision is None or not feedback_preserved
                or not self._follow_plan_owns_motor()):
            return False  # A required reduction cannot be silently postponed.
        if decision.forward_pair != pair:
            self._follow_defer_reject_reason = "forward_replan_needed"
            return False
        self.logger.info("follow_packet_deferred uid=%s reason=publication_churn "
            "held_forward_pair=%s receipt_sequence=%s checked_sample_ts=%.6f "
            "physical_deadline=%.6f packet_written=False receipt_renewed=False "
            "deadline_renewed=False next_service_due=True",
            uid, pair, receipt.sequence, linear[3], marker[1].depth_expires_at)
        return True

    @contextmanager
    def _follow_planning_attempt(self):
        with ExitStack() as stack:
            self._follow_commit_stack = stack
            try:
                yield
            finally:
                stack.close()
                self._follow_commit_stack = None
                self._follow_commit_locked = False

    def _send_follow_wheel_targets(self, *args, **kwargs):
        if self._follow_planning_offlock and not self._follow_offlock_owner():
            # Producers/direct callers cannot enter another thread's mutable
            # FOLLOW plan. STOP and raw safety writers remain independent.
            return False
        if not self._follow_offlock_owner() or self._follow_commit_stack is not None:
            return self._plan_follow_wheel_targets(*args, **kwargs)
        with ExitStack() as stack:
            self._follow_commit_stack = stack
            try:
                return self._plan_follow_wheel_targets(*args, **kwargs)
            finally:
                # ExitStack releases a possibly acquired commit lock after
                # the complete write receipt/guard accounting, on every exit.
                stack.close()
                self._follow_commit_stack = None
                self._follow_commit_locked = False

    def _plan_follow_wheel_targets(
        self, left, right, label, *, max_target_override=None, visible_required=False
    ):
        """Periodic planning is off-I/O-lock; legacy direct calls stay locked."""
        self._follow_base_contraction_reject_reason = "not_checked"
        if self._near_yaw_park_blocks_write(label):
            return False
        if not self._periodic_follow_writing and self._periodic_follow_active():
            # Producers only update canonical axes. Do not remember this
            # packet: a later tick must reconstruct BOTH current axes.
            return False
        if not self._visible_wheel_control_active():
            self._visible_wheel_guard.reset()
            self._forward_loss_handoff.reset()
            self._visible_wheel_waiting = False
            if visible_required:
                self.logger.info("visible_wheel_revoked label=%s reason=visible_state_changed", label)
                return False
            if (label in {"DRIVE", "STEER", "YAW_ONLY"} and (left != 0 or right != 0)
                    and getattr(self.owner, "_detector_identity_lease", None) is not None):
                # A detector-only result can withdraw periodic follow while
                # an older visible command is still waiting for motor I/O.
                # This fallback has no final Depth/identity deadline reader.
                self.logger.info("detector_identity_fallback_veto label=%s", label)
                return False
            if max_target_override is None:
                self._follow_motor_call("send_targets", left, right, label)
            else:
                self._follow_motor_call("send_targets", left, right, label, max_target_override=max_target_override)
            return
        # Exiting ordinary parking is a hardware mode transition, even when
        # the next guarded pair is zero. Do it before measuring authority age.
        # Explicit preserved NORMAL cross-brakes remain held until nonzero.
        if not self.backend.normal_zero_hold:
            parking_exit = bool(getattr(self.backend, "parking_current_a", 0.0)
                or getattr(self.backend, "_parking_current_uncertain", False))
            if not self._follow_motor_call("prepare_speed_mode"):
                return False
            if parking_exit and self.hard_stop_check(getattr(self.owner, "current_command", None)):
                self._follow_motor_call("send_stop", "parking_exit_hard_stop", mode="emergency")
                return False
        now = time.monotonic()
        uid = self.owner._follow_controller.active_target_id
        packet_entry_receipt = getattr(self.backend, "last_speed_receipt", None)
        packet_entry_stop_generation = getattr(self.backend, "stop_write_generation", 0)
        if self._periodic_follow_writing:
            self._follow_base_receipt_reference = getattr(self.backend, "last_speed_receipt", None)
        if self._periodic_follow_writing and self._periodic_follow_axes[0] != uid:
            self._follow_motor_call("send_targets", 0, 0, "FOLLOW20_UID_CHANGED")
            self._visible_wheel_guard.reset()
            self._forward_loss_handoff.reset()
            return False
        revision = getattr(self.owner, "_lateral_yaw_revision", None)
        # Keep the publication which produced a requested zero. A new Depth
        # grant may arrive while that zero waits for serial ownership, after
        # the last off-lock axes read. Comparing only before lock acquisition
        # allowed the obsolete zero to erase newly authorized forward motion.
        entry_store = getattr(self.owner, "_lateral_intent_store", None)
        zero_plan_publication = (
            getattr(self, "_follow_source_grant", None), revision,
            self._follow_intent_snapshot(entry_store),
            getattr(self.owner, "_lateral_turn_response_policy", None))
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
        linear = self._read_depth_linear(uid, now=now)
        final_linear = linear  # Zero/park branches still share receipt bookkeeping.
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
        if self._periodic_follow_writing and getattr(self, "_follow_axes_rebuild_allowed", False):
            self._follow_revision_reference = (linear, getattr(self.backend, "last_speed_receipt", None))
        feedback = self.get_steering_feedback()
        # The cache read can complete after the first clock sample. Guarding
        # that fresh feedback against the older instant falsely marks it as
        # coming from the future and inserts a zero-speed wait.
        now = time.monotonic()
        snapshot_forward = self._ordinary_snapshot_scope(uid, requested, feedback, now)
        moving_evidence = getattr(self.owner, "_search_handoff_moving_evidence", None)
        moving_handoff = bool(getattr(self.owner, "_search_handoff_uid", None) == uid
            and ((base > 0 and moving_evidence is not None)
                 or getattr(self.owner, "_search_handoff_moving_active", False)))
        early_fresh_handoff = None
        handoff_expected = None
        final_lower_adopted = False
        if (not snapshot_forward and self._periodic_follow_writing
                and self._read_follow_axes(time.monotonic()) != self._periodic_follow_axes):
            # Do not let an already superseded pivot start a reversal wait
            # before rebuilding the newly authorized forward curve.
            reference = getattr(self, "_follow_revision_reference", None)
            if (reference is not None and reference[1] is not None
                    and getattr(self.backend, "last_speed_receipt", None) is not reference[1]):
                return False  # Preserve a concurrent STOP/write, including its mode.
            if self._request_follow_axes_rebuild(uid):
                return False
            if self._adopt_follow_revision_only(uid, linear, feedback):
                revision = self._periodic_follow_axes[1]
            else:
                current_axes = self._read_follow_axes(time.monotonic())
                fresh = self._fresh_forward_grant_handoff(
                    uid, self._periodic_follow_axes, current_axes,
                    feedback, time.monotonic(), allow_neutral_yaw=True)
                if fresh is not None:
                    # Retry credit was already spent, but no wheel guard has
                    # run yet. Adopt this new, independently qualified Depth
                    # axis and let the normal wheel/STOP/final-write path run
                    # once. A later publication is still rejected at final.
                    old_axes = self._periodic_follow_axes
                    self._follow_source_grant = fresh[0]
                    self._periodic_follow_axes = current_axes
                    revision, base, yaw = current_axes[1:]
                    raw = fresh[0][0]
                    linear = ("forward", base*100./self.config.motor_forward_max_target_rpm,
                              uid, raw[3])
                    requested = (int(round(base+yaw)), int(round(base-yaw)))
                    early_fresh_handoff = (old_axes, current_axes)
                elif self._can_defer_follow_base_contraction(
                        uid, self._periodic_follow_axes, current_axes, feedback,
                        time.monotonic(), allow_yaw_update=True):
                # The old higher pair is still unsent. Let ordinary wheel
                # guards run; final review will construct the lower pair.
                    pass
                else:
                    straight = self._second_same_grant_straight_axes(
                        uid, self._periodic_follow_axes, current_axes, feedback)
                    if straight is not None:
                        adopted_axes, linear, expected_receipt, expected_stop = straight
                        handoff_expected = (expected_receipt, expected_stop)
                        self._periodic_follow_axes = adopted_axes
                        revision, base, yaw = adopted_axes[1:]
                        requested = (int(round(base)), int(round(base)))
                        self.logger.info("follow_wheel_second_straight_rebuild uid=%s "
                                         "axes=%s same_grant=True packet_written=False",
                                         uid, adopted_axes)
                    else:
                        if (getattr(self.backend, "last_speed_receipt", None)
                                is not packet_entry_receipt
                                or getattr(self.backend, "stop_write_generation", 0)
                                    != packet_entry_stop_generation):
                            return False  # A concurrent STOP/write keeps ownership.
                        self.logger.info("follow_wheel_veto uid=%s reason=axes_changed_before_guard "
                                         "planned_axes=%s current_axes=%s contraction_block=%s "
                                         "fresh_handoff_block=%s",
                                         uid, self._periodic_follow_axes, current_axes,
                                         self._follow_base_contraction_reject_reason,
                                         getattr(self, "_fresh_forward_handoff_reject_reason", None))
                        self._follow_motor_call("send_targets", 0, 0, "FOLLOW20_AUTHORITY_CHANGED")
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
        view_adjusted = False
        view_phase = "disabled"
        view_intent = None
        view_policy = getattr(self.owner, "_lateral_turn_response_policy", None)
        view_feedback = feedback
        view_input = requested
        view_receipt = getattr(self.backend, "last_speed_receipt", None)
        if image_mode and getattr(self.config, "follow_turn_response_assist_enable", False):
            store = getattr(self.owner, "_lateral_intent_store", None)
            view_intent = store.snapshot() if store is not None else None
            view_permitted = bool(view_intent is not None
                and view_policy == (view_intent.sequence, True)
                and not moving_handoff and not response_adjusted
                and linear and linear[0] == "forward"
                and self._visible_wheel_control_active())
            self._view_retention.sync_writer(getattr(self._visible_wheel_guard, "last_sent", 0.),
                                             getattr(self.backend, "last_speed_receipt", None))
            view_base, view_yaw, view_phase = self._view_retention.adjust(
                base, yaw, view_intent, feedback, now, uid, permitted=view_permitted,
                hfov=getattr(cfg, "visible_steering_pid_camera_hfov_deg", 0.))
            if (view_base, view_yaw) != (base, yaw):
                base, yaw = view_base, view_yaw
                requested = (int(round(base+yaw)), int(round(base-yaw)))
                view_adjusted = True
        guarded_request, handoff_reason = requested, None
        priority_removed = max(0., original_base-base)
        priority_reason = response_phase if priority_removed else "forward_base_preserved"
        if view_adjusted:
            priority_reason = view_phase
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
            allow_quiet_forward_tail=bool(base > 0 and linear and linear[0] == "forward"
                and self._ordinary_forward_feedback_eligible(uid, feedback, now)),
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
        if (snapshot_forward and min(applied) >= 0 and sum(applied) > 0
                and not response_adjusted and not view_adjusted and not moving_handoff
                and reason in {"continuous", "forward_brake", "forward_controller_handoff",
                               "cross_obsolete_turn_forward_handoff"}
                and trial_phase in {"disabled", "tracking", "forward_brake"}):
            return self._send_ordinary_snapshot_forward(
                uid, label, max_target_override=max_target_override)
        if (applied != (0, 0) and response_adjusted and response_intent is not None
                and (self.owner._lateral_intent_store.snapshot() is not response_intent
                     or getattr(self.owner, "_lateral_turn_response_policy", None) != response_policy)):
            self.logger.info("turn_response_veto reason=intent_or_braking_policy_replaced_before_write")
            return False
        if (self._periodic_follow_writing
                and self._read_follow_axes(time.monotonic()) != self._periodic_follow_axes):
            # Feedback/serial lock acquisition can consume the remaining TTL.
            # Never write the pair computed before an axis expired or changed.
            if self._request_follow_axes_rebuild(uid):
                return False
            current_axes = self._read_follow_axes(time.monotonic())
            ordinary_handoff = bool(
                reason == "continuous" and not response_adjusted and not view_adjusted
                and trial_phase in {"disabled", "tracking"} and not moving_handoff
                and contract_forward_axes(
                    applied, self._periodic_follow_axes, current_axes) is not None
                and self._can_defer_follow_base_contraction(
                    uid, self._periodic_follow_axes, current_axes, feedback, time.monotonic(),
                    allow_yaw_update=True))
            fresh_handoff = False
            if (not ordinary_handoff and reason == "continuous"
                    and not response_adjusted and not view_adjusted
                    and trial_phase in {"disabled", "tracking"} and not moving_handoff):
                # An overtaking Depth grant may arrive AFTER the wheel guard
                # but BEFORE final review. The same-grant predicate above
                # must reject it; do not turn that rejection into a zero
                # before the fresh-grant handoff can run at final review.
                handoff_now = time.monotonic()
                fresh = self._fresh_forward_grant_handoff(
                    uid, self._periodic_follow_axes, current_axes,
                    feedback, handoff_now, allow_neutral_yaw=True)
                if fresh is not None:
                    handoff_linear = self._read_depth_linear(uid, now=handoff_now)
                    handoff_now = time.monotonic()
                    if (len(handoff_linear or ()) == 4
                            and handoff_linear[0] == "forward"
                            and handoff_linear[2:] == fresh[0][0][2:]
                            and self._same_follow_grant_marker(
                                fresh[0], self._follow_grant_marker())):
                        candidate = contract_fresh_forward_handoff(
                            uid, fresh[1], applied, self._periodic_follow_axes,
                            current_axes,
                            handoff_linear[1]*self.config.motor_forward_max_target_rpm/100.,
                            handoff_now, yaw_zeroed=fresh[2])
                        if candidate is None:
                            candidate = contract_fresh_forward_neutral_handoff(
                                uid, fresh[1], applied, self._periodic_follow_axes,
                                current_axes,
                                handoff_linear[1]*self.config.motor_forward_max_target_rpm/100.,
                                handoff_now, yaw_zeroed=fresh[2])
                        fresh_handoff = candidate is not None
            if not (ordinary_handoff or fresh_handoff):
                self._follow_motor_call("send_targets", 0, 0, "FOLLOW20_AUTHORITY_CHANGED")
                self._visible_wheel_guard.reset()
                self.logger.info("follow_wheel_veto uid=%s reason=authority_changed_before_write", uid)
                return False
            # This only permits reaching final review. No new wheel pair has
            # been authorised yet; revision and current intent are rechecked
            # there, together with unchanged physical grant and STOP owner.
            revision = current_axes[1]
        if not self._periodic_follow_writing:
            # The direct writer can also wait for encoder/serial feedback.
            # Recheck after that wait, not merely before it: an old packet
            # cannot cross the configured physical depth deadline.
            write_linear = self._read_depth_linear(uid, now=time.monotonic())
            applied_base = .5 * sum(applied)
            write_kind = "forward" if applied_base > 0 else "backward"
            if applied_base and (
                    write_linear is None or write_linear[0] != write_kind
                    or abs(applied_base) > write_linear[1] * self.config.motor_forward_max_target_rpm / 100.0 + 1e-9):
                self._follow_motor_call("send_targets", 0, 0, "FOLLOW_AUTHORITY_CHANGED")
                self._visible_wheel_guard.reset()
                self.logger.info("follow_wheel_veto uid=%s reason=authority_changed_before_direct_write", uid)
                return False
        if (revision != getattr(self.owner, "_lateral_yaw_revision", None)
                or not self._visible_wheel_control_active()
                or self.owner._follow_controller.active_target_id != uid):
            self.logger.info("visible_wheel_revoked uid=%s label=%s reason=authority_changed", uid, label)
            return
        if applied[0] != applied[1] and not self.owner._has_fresh_lateral_yaw(uid):
            self._follow_motor_call("send_targets", 0, 0, "FOLLOW_YAW_AUTHORITY_CHANGED")
            self._visible_wheel_guard.reset()
            self.logger.info("follow_wheel_veto uid=%s reason=yaw_expired_before_write", uid)
            return False
        if applied != (0, 0) and self.backend.normal_zero_hold:
            parking_exit_stop_generation = getattr(self.backend, "stop_write_generation", 0)
            if not self._follow_motor_call("prepare_speed_mode"):
                return False
            if getattr(self.backend, "stop_write_generation", 0) != parking_exit_stop_generation:
                return False  # Never replace a STOP arriving during mode I/O.
            if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                self._follow_motor_call("send_stop", "parking_exit_hard_stop", mode="emergency")
                return False
            # A preserved legacy cross-brake can require register I/O here.
            # Recheck after it, not only before releasing the parking current.
            current_linear = self._read_depth_linear(uid, now=time.monotonic())
            mean = .5 * sum(applied)
            invalid = (not self._visible_wheel_control_active()
                or self.owner._follow_controller.active_target_id != uid
                or revision != getattr(self.owner, "_lateral_yaw_revision", None)
                or (applied[0] != applied[1] and not self.owner._has_fresh_lateral_yaw(uid))
                or (mean != 0 and (not current_linear
                    or current_linear[0] != ("forward" if mean > 0 else "backward")
                    or abs(mean) > current_linear[1]*self.config.motor_forward_max_target_rpm/100.)))
            if invalid or self._near_yaw_park_blocks_write(label):
                # A newer forward grant OR yaw publication can supersede this
                # packet during 0A I/O. Rebuild both axes within the existing
                # service budget; normal yaw-only updates must not reapply
                # parking current. A retry is not permission to send the old
                # packet: all identity, TTL, STOP and reversal checks rerun.
                new_forward = bool(current_linear and current_linear[0] == "forward"
                                   and current_linear[1] > 0)
                new_yaw = bool(revision != getattr(self.owner, "_lateral_yaw_revision", None)
                               and self.owner._has_fresh_lateral_yaw(uid))
                if (self._periodic_follow_writing and invalid
                        and self._visible_wheel_control_active()
                        and self.owner._follow_controller.active_target_id == uid
                        and motion_identity_live(self.owner, uid, time.monotonic())
                        and not getattr(self.owner, "stop_action_execution", False)
                        and not getattr(self.owner, "person_detected_flag", False)
                        and not self._near_yaw_park_blocks_write(label)
                        and (new_forward or new_yaw)):
                    rebuild = self._request_follow_axes_rebuild(uid)
                    self.logger.info("follow_wheel_retry uid=%s reason=%s "
                                     "old_packet_discarded=True parking_reapplied=False "
                                     "immediate_rebuild=%s", uid,
                                     "parking_exit_new_forward_grant" if new_forward
                                         else "parking_exit_new_yaw_intent", rebuild)
                    return False
                self._follow_motor_call("send_stop", "parking_exit_authority_changed", mode=self._ordinary_park_stop_mode(),
                                       preserve_zero=True, prepare_parking_current=True)
                return False
        if trial_phase == "forward_brake":
            brake_feedback = self.get_steering_feedback()
            brake_now = time.monotonic()
            brake_linear = self._read_depth_linear(uid, now=brake_now)
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
                self._follow_motor_call("send_targets", 0, 0, "FORWARD_BRAKE_EVIDENCE_CHANGED")
                self._visible_wheel_guard.reset()
                return False
        # The base-preserving assist may leave the packet unchanged. Safety
        # must not depend on whether an assistance branch altered the pair.
        if applied != (0, 0) and self.hard_stop_check(getattr(self.owner, "current_command", None)):
            self._follow_motor_call("send_stop", "visible_wheel_hard_stop", mode="emergency")
            return False
        coalesced_yaw = None
        coalesced_base = None
        coalesced_axes = None
        straight_handoff = None
        fresh_grant_handoff = None
        fresh_grant_neutral = False
        adopted_intent = adopted_policy = ...
        if applied != (0, 0):
            if self._near_yaw_park_blocks_write(label):
                return False
            # The last safety callback may itself consume the remaining
            # lease or publish a new UID/stop. Never reuse its pre-call age.
            final_now = time.monotonic()
            final_linear = self._read_depth_linear(uid, now=final_now)
            # The reader itself may log/wait. Feedback must not be checked
            # against the clock from before that work either.
            final_now = time.monotonic()
            final_mean = .5*sum(applied)
            revision_coalesced = getattr(self, "_follow_revision_coalesced", None)
            if (revision_coalesced is not None
                    and getattr(self.backend, "last_speed_receipt", None) is not revision_coalesced[1]):
                return False  # An intervening STOP/write retains ownership.
            final_invalid = (
                not self._visible_wheel_control_active()
                or self.owner._follow_controller.active_target_id != uid
                or revision != getattr(self.owner, "_lateral_yaw_revision", None)
                or not wheel_feedback_valid(feedback, final_now)
                or (revision_coalesced is not None and final_linear != revision_coalesced[0])
                or (applied[0] != applied[1] and not self.owner._has_fresh_lateral_yaw(uid))
                or (final_mean != 0 and (not final_linear
                    or final_linear[0] != ("forward" if final_mean > 0 else "backward")
                    or abs(final_mean) > final_linear[1]*self.config.motor_forward_max_target_rpm/100.)))
            current_axes = (self._read_follow_axes(final_now)
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
                contraction_kind = None
                valid_contraction = None
                coalescing_store = getattr(self.owner, "_lateral_intent_store", None)
                coalescing_intent = coalescing_store.snapshot() if coalescing_store is not None else None
                coalescing_policy = getattr(self.owner, "_lateral_turn_response_policy", None)
                park_candidate = bool(getattr(coalescing_intent, "park_requested", False))
                straight_park_candidate = straight_park_candidate_contraction(
                    uid, coalescing_intent, applied,
                    getattr(self, "_periodic_follow_axes", None), current_axes)
                contraction_first_reject = None

                def contraction_gate(allowed, detail):
                    nonlocal contraction_first_reject
                    if not allowed and contraction_first_reject is None:
                        contraction_first_reject = detail
                    return allowed

                expired_straight_intent = bool(
                    not response_adjusted and not view_adjusted
                    and self._expired_straight_intent_coalescible(
                        uid, applied, current_axes, coalescing_intent,
                        (trial_intent, response_intent), final_now))
                ordinary_axes = bool(
                    axes_changed and not response_adjusted and not view_adjusted
                    and all(not any(getattr(intent, name, False) for name in (
                        "park_requested", "forward_countersteer", "countersteer_rpm"))
                        for intent in (trial_intent, response_intent, coalescing_intent))
                    and self._can_defer_follow_base_contraction(
                        uid, self._periodic_follow_axes, current_axes, feedback, final_now,
                        allow_yaw_update=True))
                new_ordinary_intent = bool(ordinary_axes and ordinary_intent_handoff(
                    uid, coalescing_intent, (trial_intent, response_intent), final_now))
                if (contraction_gate(axes_changed, "axes_unchanged")
                        and contraction_gate(not getattr(self, "_follow_axes_rebuild_allowed", False),
                                             "rebuild_available")
                        and contraction_gate(revision_coalesced is None or ordinary_axes
                                             or straight_park_candidate, "revision_already_coalesced")
                        and contraction_gate(reason == "continuous", "wheel_guard:" + reason)
                        and contraction_gate(not response_adjusted, "response_adjusted")
                        and contraction_gate(trial_phase in {"disabled", "tracking"}, "trial:" + trial_phase)
                        and contraction_gate(not moving_handoff, "moving_handoff")
                        and contraction_gate(expired_straight_intent or new_ordinary_intent or (
                            (trial_intent is None or coalescing_intent is trial_intent)
                            and (response_intent is None or coalescing_intent is response_intent)),
                                             "lateral_intent_replaced")
                        # A stored predictive countersteer hint does not
                        # cancel a pure forward reduction when the current
                        # wheel pair is already continuous and its yaw value
                        # did not change. A zero-yaw parking candidate may
                        # also retain a smaller straight Depth base; actual
                        # parking owners are checked below and at the write.
                        and contraction_gate(not park_candidate or straight_park_candidate,
                                             "park_candidate_not_straight_contraction")
                        and contraction_gate(guard.pending_signs is None and guard.resume_signs is None
                            and not guard.pending_full_reverse and not guard.commanded_reverse
                            and guard.residual_turn_signs is None and guard.residual_forward_until <= 0,
                                             "wheel_reversal_guard")
                        and contraction_gate(not self.backend.normal_zero_hold
                            and not self.backend.parking_current_a and not self.backend._parking_current_uncertain,
                                             "motor_parking_owner")
                        and contraction_gate(wheel_feedback_valid(feedback, final_now), "wheel_feedback")):
                    braking_hint = any(getattr(coalescing_intent, name, False) for name in (
                        "forward_countersteer", "countersteer_rpm"))
                    # Braking intent may invalidate an older yaw plan. It
                    # cannot invalidate an otherwise identical yaw while the
                    # physical forward base simply contracts on one grant.
                    if new_ordinary_intent:
                        contracted = contract_forward_axes(
                            applied, self._periodic_follow_axes, current_axes)
                        if contracted is not None:
                            contraction_kind = "axes"
                    if contracted is None and not braking_hint and not new_ordinary_intent:
                        contracted = contract_forward_yaw(
                            applied, self._periodic_follow_axes, current_axes)
                        if contracted is not None:
                            contraction_kind = "yaw"
                    if (contracted is None and not view_adjusted
                            and self._can_defer_follow_base_contraction(
                                uid, self._periodic_follow_axes, current_axes, feedback, final_now)):
                        contracted = self._contract_same_grant_forward_base(
                            applied, self._periodic_follow_axes, current_axes)
                        if contracted is not None:
                            contraction_kind = "base"
                    if contracted is None and ordinary_axes:
                        contracted = contract_forward_axes(
                            applied, self._periodic_follow_axes, current_axes)
                        if contracted is not None:
                            contraction_kind = "axes"
                if contracted is not None:
                    if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                        self._follow_motor_call("send_stop", "final_yaw_coalescing_hard_stop", mode="emergency")
                        return False
                    if (self._near_yaw_park_blocks_write(label)
                            or getattr(self.owner, "_brake_hold_active", False)):
                        return False  # Its STOP owner must not be overwritten by a speed zero.
                    current_feedback = self.get_steering_feedback()  # Published cache, never serial I/O.
                    # This reader itself evaluates the age-dependent axes.
                    # Run it BEFORE the final cap/axes capture so a later
                    # natural reduction cannot escape the contracted pair.
                    active_contraction = self._periodic_follow_active()
                    checked_at = time.monotonic()
                    checked_linear = self._read_depth_linear(uid, now=checked_at)
                    if contraction_kind in {"base", "axes"}:
                        # Fresh readers may themselves advance the age-based
                        # cap. Rebuild the smaller pair after those callbacks.
                        latest_axes = self._read_follow_axes(time.monotonic())
                        latest_pair = (self._contract_same_grant_forward_base(
                            applied, self._periodic_follow_axes, latest_axes)
                            if contraction_kind == "base" else contract_forward_axes(
                                applied, self._periodic_follow_axes, latest_axes))
                        # A later yaw publication needs another full plan;
                        # only the same captured yaw may contract with age.
                        if (contraction_kind == "axes" and (latest_axes is None
                                or latest_axes[1] != current_axes[1]
                                or latest_axes[3] != current_axes[3])):
                            latest_pair = None
                        if latest_pair is None:
                            contracted = None
                        else:
                            current_axes, contracted = latest_axes, latest_pair
                    # Check after the safety callback and every lease reader.
                    # A third update or any lost evidence fails immediately.
                    valid_contraction = bool(
                        contraction_gate(contracted is not None, "latest_axes_not_monotone")
                        and contraction_gate(active_contraction, "periodic_authority_changed")
                        and contraction_gate(self.owner._follow_controller.active_target_id == uid
                            and getattr(self.owner, "_vision_control_state", "") in {
                            "target_visible", "target_visible_depth_valid",
                            "target_visible_depth_missing"}, "visible_identity_changed")
                        and contraction_gate(not self._near_yaw_park_blocks_write(label), "stop_owner")
                        and contraction_gate(not getattr(self.backend, "parking_release_fault", None)
                            and not self.backend.normal_zero_hold
                            and not self.backend.parking_current_a and not self.backend._parking_current_uncertain,
                                             "motor_parking_owner")
                        and contraction_gate(getattr(self.owner, "_search_handoff_uid", None) != uid,
                                             "search_handoff")
                        and contraction_gate(guard.pending_signs is None and guard.resume_signs is None
                            and not guard.pending_full_reverse and not guard.commanded_reverse
                            and guard.residual_turn_signs is None and guard.residual_forward_until <= 0,
                                             "wheel_reversal_guard")
                        and contraction_gate((coalescing_store is None or coalescing_store.snapshot() is coalescing_intent)
                            and getattr(self.owner, "_lateral_turn_response_policy", None) == coalescing_policy
                            and (not new_ordinary_intent or ordinary_intent_handoff(
                                uid, coalescing_intent, (trial_intent, response_intent), time.monotonic())),
                                             "lateral_intent_or_policy_changed")
                        and contraction_gate(checked_linear is not None and checked_linear[0] == "forward"
                            and .5*sum(contracted) <= checked_linear[1]*self.config.motor_forward_max_target_rpm/100.,
                                             "forward_authority_or_cap_changed")
                        and contraction_gate(contracted[0] == contracted[1] or self.owner._has_fresh_lateral_yaw(uid),
                                             "yaw_evidence_expired")
                        and contraction_gate((contraction_kind in {"base", "axes"}
                            or self._read_follow_axes(time.monotonic()) == current_axes)
                            and getattr(self.owner, "_lateral_yaw_revision", None) == current_axes[1],
                                             "axes_or_revision_changed")
                        and contraction_gate(wheel_feedback_valid(feedback, time.monotonic())
                            and self._ordinary_forward_feedback_eligible(uid, current_feedback, time.monotonic()),
                                             "wheel_feedback")
                        and contraction_gate(not self._near_yaw_park_blocks_write(label)
                            and self._visible_wheel_control_active()
                            and self.owner._follow_controller.active_target_id == uid
                            and not getattr(self.owner, "stop_action_execution", False)
                            and not getattr(self.owner, "person_detected_flag", False), "final_stop_or_identity")
                        and contraction_gate(not expired_straight_intent or self._expired_straight_intent_coalescible(
                            uid, contracted, current_axes, coalescing_intent,
                            (trial_intent, response_intent), time.monotonic()), "expired_intent_changed")
                        and contraction_gate((contraction_kind not in {"base", "axes"} and not expired_straight_intent) or (
                            self._same_follow_grant_marker(
                                getattr(self, "_follow_source_grant", None),
                                self._follow_grant_marker())
                            and getattr(self.backend, "last_speed_receipt", None)
                                is getattr(self, "_follow_base_receipt_reference", None)
                            and checked_linear[2:] == self._follow_source_grant[0][2:]), "grant_or_receipt_changed"))
                    if valid_contraction:
                        adopted_intent, adopted_policy = coalescing_intent, coalescing_policy
                        if contraction_kind in {"base", "axes"}:
                            details = (self._periodic_follow_axes, current_axes, applied, contracted)
                            if contraction_kind == "base":
                                coalesced_base = details
                            else:
                                coalesced_axes = details
                            # Record the pair actually sent. A prior fresh
                            # read may already have bounded it below the
                            # latest canonical axes; the wheel clock must not
                            # claim that higher base was executed.
                            adopted_axes = (uid, current_axes[1], .5*sum(contracted),
                                            .5*(contracted[0]-contracted[1]))
                        else:
                            coalesced_yaw = (self._periodic_follow_axes, current_axes, applied, contracted)
                            adopted_axes = current_axes
                        if expired_straight_intent or (straight_park_candidate and revision_coalesced is not None):
                            self._follow_revision_coalesced = (
                                checked_linear, self._follow_base_receipt_reference,
                                self._periodic_follow_axes, adopted_axes)
                        applied = contracted
                        self._periodic_follow_axes = adopted_axes
                        revision, base, yaw = adopted_axes[1:]
                        final_linear = checked_linear
                        feedback = current_feedback
                        reason = "final_%s_coalesced" % contraction_kind
                    else:
                        contraction_gate(False, "final_contraction_evidence_changed")
                        contracted = None
                handoff_candidate = (coalescing_intent if coalescing_intent is not None
                                     else trial_intent or response_intent)
                if (contracted is None and self._periodic_follow_writing
                        and min(applied) >= 0 and sum(applied) > 0
                        and isinstance(handoff_candidate, LateralControlIntent)):
                    # A completed rebuild can be overtaken by another ordinary
                    # visual publication or a newer Depth sample. The new
                    # grant may only inherit a non-increasing wheel pair;
                    # opposite yaw uses a straight bridge, not the old turn.
                    # A parking *candidate* is not a STOP owner; the actual
                    # park/STOP flags below still prohibit this path.
                    handoff_now = time.monotonic()
                    vision_ts = getattr(self.owner, "_last_vision_control_ts", None)
                    handoff_linear = self._read_depth_linear(uid, now=handoff_now)
                    handoff_now = time.monotonic()
                    handoff_marker = self._follow_grant_marker()
                    handoff_axes = self._read_follow_axes(handoff_now)
                    handoff_feedback = self.get_steering_feedback()
                    handoff_now = time.monotonic()
                    handoff_pair = None
                    handoff_kind = None
                    if handoff_linear is not None and len(handoff_linear) == 4:
                        handoff_cap = (handoff_linear[1]
                            *self.config.motor_forward_max_target_rpm/100.)
                        if applied[0] == applied[1]:
                            handoff_pair = contract_straight_forward_handoff(
                                uid, handoff_candidate, applied, self._periodic_follow_axes,
                                handoff_axes, handoff_cap, handoff_now,
                                cleared=coalescing_intent is None)
                            if handoff_pair is not None:
                                handoff_kind = "straight"
                        if handoff_pair is None:
                            fresh = self._fresh_forward_grant_handoff(
                                uid, self._periodic_follow_axes, handoff_axes,
                                handoff_feedback, handoff_now,
                                allow_neutral_yaw=True)
                            if fresh is not None and fresh[0] == handoff_marker:
                                handoff_pair = contract_fresh_forward_handoff(
                                    uid, fresh[1], applied, self._periodic_follow_axes,
                                    handoff_axes, handoff_cap, handoff_now,
                                    yaw_zeroed=fresh[2])
                                if handoff_pair is not None:
                                    handoff_kind = "fresh_grant"
                                else:
                                    handoff_pair = contract_fresh_forward_neutral_handoff(
                                        uid, fresh[1], applied,
                                        self._periodic_follow_axes, handoff_axes,
                                        handoff_cap, handoff_now,
                                        yaw_zeroed=fresh[2])
                                    if handoff_pair is not None:
                                        handoff_kind = "fresh_neutral"
                    if (reason == "continuous" and trial_phase in {"disabled", "tracking"}
                            and not response_adjusted and not view_adjusted and not moving_handoff
                            and revision_coalesced is None and handoff_pair is not None
                            and isinstance(vision_ts, (int, float)) and math.isfinite(vision_ts)
                            and 0 <= handoff_now-vision_ts <= .25
                            and self._periodic_follow_scope_active()
                            and self.owner._follow_controller.active_target_id == uid
                            and motion_identity_live(self.owner, uid, handoff_now)
                            and handoff_axes[1] == getattr(self.owner, "_lateral_yaw_revision", None)
                            and handoff_linear[0] == "forward" and handoff_linear[2] == uid
                            and handoff_marker is not None and handoff_marker[0][0] == "forward"
                            and handoff_marker[0][2:] == handoff_linear[2:]
                            and handoff_marker[0][1] >= handoff_linear[1]
                            # A cleared yaw owner can only finish the SAME
                            # physical grant. A new Depth grant needs a live
                            # visual publication of its own.
                            and (coalescing_intent is not None or
                                 self._same_follow_grant_marker(
                                     handoff_marker, self._follow_source_grant))
                            and coalescing_store is not None
                            and coalescing_store.snapshot() is coalescing_intent
                            and self._ordinary_forward_feedback_eligible(uid, handoff_feedback, handoff_now)
                            and guard.pending_signs is None and guard.resume_signs is None
                            and not guard.pending_full_reverse and not guard.commanded_reverse
                            and guard.residual_turn_signs is None and guard.residual_forward_until <= 0
                            and not getattr(self.owner, "_brake_hold_active", False)
                            and getattr(self.owner, "_near_yaw_park_request", None) is None
                            and getattr(self, "_search_reacquire_brake_request", None) is None
                            and getattr(self.owner, "_search_handoff_uid", None) != uid
                            and not self.backend.normal_zero_hold and not self.backend.parking_current_a
                            and not self.backend._parking_current_uncertain
                            and not self.backend.parking_release_fault
                            and getattr(self.backend, "last_speed_receipt", None)
                                is getattr(self, "_follow_base_receipt_reference", None)
                            and getattr(self.backend, "last_speed_receipt", None) is not None
                            and getattr(self.backend, "stop_write_generation", 0)
                                == getattr(self, "_follow_wheel_last_stop_generation", None)):
                        details = (self._periodic_follow_axes, handoff_axes,
                                   applied, handoff_pair)
                        if handoff_kind in {"fresh_grant", "fresh_neutral"}:
                            fresh_grant_handoff = details
                            fresh_grant_neutral = handoff_kind == "fresh_neutral"
                        else:
                            straight_handoff = details
                        applied = handoff_pair
                        revision = handoff_axes[1]
                        base = .5*sum(handoff_pair)
                        yaw = .5*(handoff_pair[0]-handoff_pair[1])
                        self._periodic_follow_axes = (uid, revision, base, yaw)
                        self._follow_source_grant = handoff_marker
                        final_linear = handoff_linear
                        feedback = handoff_feedback
                        adopted_intent, adopted_policy = coalescing_intent, coalescing_policy
                        reason = ("final_fresh_grant_handoff"
                                  if handoff_kind in {"fresh_grant", "fresh_neutral"}
                                  else "final_straight_handoff")
                        contracted = handoff_pair
                if contracted is None:
                    lower = (self._final_lower_straight_grant_handoff(uid, applied, feedback)
                             if reason == "continuous" and not response_adjusted
                             and not view_adjusted and not moving_handoff
                             and trial_phase in {"disabled", "tracking"} else None)
                    live_forward_bridge = False
                    if (lower is None and reason == "continuous" and not response_adjusted
                            and not view_adjusted and not moving_handoff
                            and trial_phase in {"disabled", "tracking"}):
                        lower = self._final_live_forward_handoff(uid, applied, feedback)
                        live_forward_bridge = lower is not None
                    if lower is not None:
                        if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                            self._follow_motor_call("send_stop", "final_straight_handoff_hard_stop", mode="emergency")
                            return False
                        (lower_pair, lower_axes, lower_linear, lower_marker,
                         lower_feedback, lower_intent, lower_policy,
                         expected_receipt, expected_stop) = lower
                        handoff_expected = (expected_receipt, expected_stop)
                        final_lower_adopted = True
                        fresh_grant_handoff = (self._periodic_follow_axes,
                                               lower_axes, applied, lower_pair)
                        applied = lower_pair
                        revision = lower_axes[1]
                        base, yaw = .5*sum(lower_pair), 0.
                        self._periodic_follow_axes = (uid, revision, base, yaw)
                        self._follow_source_grant = lower_marker
                        final_linear = lower_linear
                        feedback = lower_feedback
                        adopted_intent, adopted_policy = lower_intent, lower_policy
                        reason = ("final_live_forward_handoff" if live_forward_bridge
                                  else "final_fresh_grant_handoff")
                        contracted = lower_pair
                if contracted is None:
                    if (self._near_yaw_park_blocks_write(label)
                            or getattr(self.owner, "_brake_hold_active", False)):
                        return False
                    # Snapshot diagnostic facts BEFORE the zero write changes
                    # the completed-write receipt. This is deliberately
                    # read-only; it cannot grant or delay motion.
                    source_grant = getattr(self, "_follow_source_grant", None)
                    prior_receipt = getattr(self, "_follow_base_receipt_reference", None)
                    grant_stable = self._same_follow_grant_marker(
                        source_grant, self._follow_grant_marker())
                    receipt_stable = bool(prior_receipt is not None
                        and getattr(self.backend, "last_speed_receipt", None) is prior_receipt)
                    if (getattr(self.backend, "last_speed_receipt", None) is not packet_entry_receipt
                            or getattr(self.backend, "stop_write_generation", 0)
                                != packet_entry_stop_generation):
                        return False  # Rejected candidates cannot overwrite a newer STOP/write.
                    base_pair_possible = self._contract_same_grant_forward_base(
                        applied, getattr(self, "_periodic_follow_axes", None),
                        current_axes) is not None
                    contraction_block = getattr(
                        self, "_follow_base_contraction_reject_reason", "not_checked")
                    if contraction_first_reject is None:
                        contraction_first_reject = (contraction_block
                            if contraction_block not in (None, "not_checked") else "no_monotone_pair")
                    stop_owner = ",".join(name for name, active in (
                        ("near_yaw_park", getattr(self.owner, "_near_yaw_park_request", None) is not None),
                        ("brake_hold", getattr(self.owner, "_brake_hold_active", False)),
                        ("explicit_stop", getattr(self.owner, "_explicit_stop_requested", False)),
                        ("shutdown", getattr(self.owner, "_runtime_shutdown_requested", False)),
                        ("stop_action", getattr(self.owner, "stop_action_execution", False)),
                        ("person_stop", getattr(self.owner, "person_detected_flag", False)),
                        ("search_brake", self._search_reacquire_brake_request is not None),
                        ("motor_parking", self.backend.normal_zero_hold or self.backend.parking_current_a
                         or self.backend._parking_current_uncertain),
                    ) if active) or "none"
                    vision_state = getattr(self.owner, "_vision_control_state", "")
                    self._follow_motor_call("send_targets", 0, 0, "FOLLOW_FINAL_AUTHORITY_CHANGED")
                    self._visible_wheel_guard.reset()
                    self.logger.info(
                        "follow_wheel_veto uid=%s reason=final_authority_or_feedback_changed "
                        "axes_changed=%s revision_changed=%s feedback_valid=%s "
                        "feedback_age_ms=%s planned_axes=%s current_axes=%s linear=%s "
                        "retry_available=%s contraction_block=%s base_pair_possible=%s "
                        "grant_stable=%s receipt_stable=%s vision_state=%s "
                        "wheel_reason=%s trial_phase=%s response_adjusted=%s "
                        "view_adjusted=%s moving_handoff=%s final_invalid=%s "
                        "contraction_kind=%s contraction_valid=%s "
                        "trial_intent_same=%s response_intent_same=%s "
                        "expired_straight_intent=%s contraction_first_reject=%s "
                        "park_candidate=%s straight_park_candidate=%s stop_owner=%s "
                        "depth_final_quiet_veto=%s",
                        uid, axes_changed, revision != getattr(self.owner, "_lateral_yaw_revision", None),
                        wheel_feedback_valid(feedback, final_now),
                        None if feedback is None else (final_now-feedback.timestamp)*1000.,
                        getattr(self, "_periodic_follow_axes", None),
                        current_axes,
                        final_linear, getattr(self, "_follow_axes_rebuild_allowed", False),
                        contraction_block, base_pair_possible, grant_stable,
                        receipt_stable, vision_state, reason, trial_phase,
                        response_adjusted, view_adjusted, moving_handoff,
                        final_invalid, contraction_kind, valid_contraction,
                        trial_intent is None or coalescing_intent is trial_intent,
                        response_intent is None or coalescing_intent is response_intent,
                        expired_straight_intent, contraction_first_reject,
                        park_candidate, straight_park_candidate, stop_owner,
                        getattr(self.owner, "_last_quiet_depth_veto", None)
                        if final_linear is None else None)
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
            moving_linear = self._read_depth_linear(uid, now=moving_now)
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
                self._follow_motor_call("send_targets", 0, 0, "SEARCH_HANDOFF_EVIDENCE_CHANGED")
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
                self._follow_motor_call("send_targets", 0, 0, "SEARCH_HANDOFF_CONSTRAINT_CHANGED")
                self._visible_wheel_guard.reset()
                return False
        # A zero planned against missing/older axes can itself become obsolete
        # during the guards above. Nonzero packets already get final lease
        # checks; zeros must also yield to a newly admitted pair, not erase it.
        # Rebuild at most once and rerun ALL reversal/braking checks. A genuine
        # safety zero or a still-current zero request is never delayed here.
        if (applied == (0, 0) and self._periodic_follow_writing
                and self._read_follow_axes(time.monotonic()) != self._periodic_follow_axes
                and self._request_follow_axes_rebuild(uid)):
            self.logger.info("follow_wheel_retry uid=%s reason=zero_plan_superseded_before_write "
                             "old_packet_discarded=True motion_authorized=False", uid)
            return False
        previous_pair = getattr(self._visible_wheel_guard, "last_output", None)
        previous_sent = getattr(self._visible_wheel_guard, "last_sent", 0.0)
        revision_coalesced = getattr(self, "_follow_revision_coalesced", None)
        if (revision_coalesced is not None
                and getattr(self.backend, "last_speed_receipt", None) is not revision_coalesced[1]):
            return False
        cross_brake = "none"
        packet_written = True
        if view_adjusted and applied != (0, 0):
            # This final, non-I/O check follows all blocking guards. The
            # reduced pair has no right to survive a changed lease or PID veto.
            view_read_started = time.monotonic()
            view_linear = self._read_depth_linear(uid, now=view_read_started)
            # The grant reader can log, wait, and observe encoders itself.
            # Collect lock-taking accessors first; check the clock and caches
            # AFTER that work, without another reader call/retry loop.
            view_yaw_fresh = self.owner._has_fresh_lateral_yaw(uid)
            view_current_intent = self.owner._lateral_intent_store.snapshot()
            view_latest_feedback = self.get_steering_feedback()
            view_now = time.monotonic()
            view_current_linear = getattr(self.owner, "_depth30_linear_snapshot", view_linear)
            if (view_current_intent is not view_intent
                    or getattr(self.owner, "_lateral_turn_response_policy", None) != view_policy
                    or self.owner._follow_controller.active_target_id != uid
                    or not self._visible_wheel_control_active()
                    or not view_yaw_fresh
                    or revision != getattr(self.owner, "_lateral_yaw_revision", None)
                    or view_latest_feedback is not view_feedback
                    or getattr(self.backend, "last_speed_receipt", None) is not view_receipt
                    or not self._view_retention.grant_survives_read(
                        view_linear, view_current_linear, getattr(self.owner, "_depth30_linear_timing", None),
                        uid=uid, started=view_read_started, now=view_now,
                        ttl=getattr(cfg, "depth_longitudinal_sample_max_age_sec", .18), base=.5*sum(applied))
                    or .5*sum(applied) > view_linear[1]*self.config.motor_forward_max_target_rpm/100.
                    or self._view_retention.deadline is None or view_now >= self._view_retention.deadline
                    or not self._view_retention.evidence_valid(view_intent, view_feedback, view_now, uid,
                        permitted=True, hfov=getattr(cfg, "visible_steering_pid_camera_hfov_deg", 0.))
                    or min(applied) < 0 or abs(applied[0]-applied[1]) > 20
                    or any(a > r for a, r in zip(applied, view_input))):
                self.logger.info("view_retention_veto uid=%s reason=evidence_changed_before_write", uid)
                return False
        previous_speed_receipt = getattr(self.backend, "last_speed_receipt", None)
        previous_stop_generation = getattr(self.backend, "stop_write_generation", 0)
        zero_held = bool(getattr(self.backend, "normal_zero_hold", False))
        terminal_contraction = None
        commit_publication = None
        if (cross_brake_enabled
                and getattr(self.config, "follow_cross_brake_mode", "zero") == "normal"
                and applied == (0, 0)
                and reason in {"cross_wait_zero", "cross_timeout_zero"}):
            if not self._begin_follow_commit():
                return False
            if not zero_held:
                if not self._follow_motor_call(
                        "send_stop", "follow_cross_brake", mode=self._ordinary_park_stop_mode(),
                        preserve_zero=True, prepare_parking_current=True):
                    return False
                cross_brake = "applied"
            else:
                packet_written = False
                cross_brake = "held"
        else:
            packet_written = not (zero_held and applied == (0, 0))
            cross_brake = "released" if zero_held and any(applied) else "held" if zero_held else "none"
            if (handoff_expected is not None
                    and (getattr(self.backend, "last_speed_receipt", None)
                         is not handoff_expected[0]
                         or getattr(self.backend, "stop_write_generation", 0)
                            != handoff_expected[1])):
                return False  # Preserve a STOP/write which overtook either new helper.
            if final_lower_adopted:
                # The live reader may tighten the same raw Depth grant after
                # the final handoff candidate was built. It can block, so the
                # physical clock and STOP/receipt token are checked again below.
                final_linear = self._read_depth_linear(uid, now=time.monotonic())
                if (len(final_linear or ()) == 4 and final_linear[0] == "forward"
                        and final_linear[2:] == self._follow_source_grant[0][2:]
                        and isinstance(final_linear[1], (int, float))
                        and math.isfinite(final_linear[1]) and final_linear[1] > 0):
                    live_cap = (final_linear[1]
                                *self.config.motor_forward_max_target_rpm/100.)
                    if live_cap < .5*sum(applied):
                        reduced = contract_forward_speed(applied, live_cap)
                        if reduced is not None:
                            terminal_contraction = (applied, reduced, live_cap)
                            applied = reduced
                            base = .5*sum(reduced)
                            self._periodic_follow_axes = (uid, revision, base, 0.)
                            fresh_grant_handoff = (*fresh_grant_handoff[:3], reduced)
                if (getattr(self.backend, "last_speed_receipt", None)
                        is not handoff_expected[0]
                        or getattr(self.backend, "stop_write_generation", 0)
                            != handoff_expected[1]):
                    return False
            if self._follow_offlock_owner() and any(applied):
                # The encoder publishes immutable feedback by one reference
                # assignment. Reuse the guarded sample unless a newer cache
                # publication has arrived; do not call an accessor under I/O.
                commit_feedback = feedback
                commit_store = getattr(self.owner, "_lateral_intent_store", None)
                commit_intent = self._follow_intent_snapshot(commit_store)
                commit_policy = getattr(self.owner, "_lateral_turn_response_policy", None)
                commit_publication = (
                    getattr(self, "_follow_source_grant", None), revision,
                    commit_intent if adopted_intent is ... else adopted_intent,
                    commit_policy if adopted_policy is ... else adopted_policy)
                if not self._begin_follow_commit():
                    return False
                if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                    self._follow_motor_call("send_stop", "follow_commit_hard_stop", mode="emergency")
                    self._visible_wheel_guard.reset()
                    return False
                if self._request_follow_commit_rebuild(
                        uid, commit_publication,
                        yaw_only_pair=applied if sum(applied) == 0 else None):
                    return False
                cached = getattr(self, "_steering_feedback", None)
                if (cached is not None and (commit_feedback is None
                        or cached.timestamp > commit_feedback.timestamp)):
                    commit_feedback = cached
                commit_now = time.monotonic()
                if (self._near_yaw_park_blocks_write(label)
                        or getattr(self.owner, "_brake_hold_active", False)):
                    return False
                yaw_invalid = False
                if applied[0] != applied[1]:
                    if isinstance(commit_intent, LateralControlIntent):
                        yaw_invalid = (commit_intent.target_id != uid
                            or not commit_intent.valid(commit_now)
                            or commit_intent.hold_zero
                            or not commit_intent.continuation_allowed(commit_now, commit_feedback)
                            or getattr(self.owner, "_lateral_intent_zero_sequence", -1)
                                == commit_intent.sequence)
                    else:
                        yaw_invalid = not self.owner._has_fresh_lateral_yaw(uid)
                if (not wheel_feedback_valid(commit_feedback, commit_now)
                        or self.owner._follow_controller.active_target_id != uid
                        or not motion_identity_live(self.owner, uid, commit_now)
                        or not self._visible_wheel_control_active()
                        or revision != getattr(self.owner, "_lateral_yaw_revision", None)
                        or self._follow_intent_snapshot(commit_store) is not commit_intent
                        or getattr(self.owner, "_lateral_turn_response_policy", None) != commit_policy
                        or yaw_invalid
                        or any(getattr(self.owner, name, False) for name in (
                            "_explicit_stop_requested", "_runtime_shutdown_requested",
                            "stop_action_execution", "person_detected_flag"))):
                    if self._request_follow_commit_rebuild(
                            uid, commit_publication, expired_forward_yaw=bool(
                                yaw_invalid and min(applied) >= 0 and sum(applied) > 0),
                            yaw_only_pair=applied if sum(applied) == 0 else None):
                        return False
                    # The source lease is not prolonged by an off-lock plan.
                    # Preserve an intervening STOP/packet; otherwise revoke.
                    self._follow_motor_call("send_targets", 0, 0, "FOLLOW_COMMIT_EVIDENCE_EXPIRED")
                    return False
                if (feedback is not None and commit_feedback.timestamp > feedback.timestamp
                        and any(command*measured < 0 and abs(measured) > 1.
                                for command, measured in zip(applied, (
                                    commit_feedback.left_forward_rpm,
                                    commit_feedback.right_forward_rpm)))
                        and not (min(applied) > 0 and self._ordinary_forward_feedback_eligible(
                            uid, commit_feedback, commit_now))):
                    # A newer encoder sample cannot inherit a wheel-crossing
                    # guard which ran on different direction evidence.
                    self._follow_motor_call("send_targets", 0, 0, "FOLLOW_COMMIT_FEEDBACK_CHANGED")
                    return False
                feedback = commit_feedback
                if adopted_intent is ...:
                    adopted_intent = commit_intent
                if adopted_policy is ...:
                    adopted_policy = commit_policy
            ordinary_contraction = bool(
                self._periodic_follow_writing
                and reason in {"continuous", "final_base_coalesced", "final_axes_coalesced",
                               "final_yaw_coalesced", "final_straight_handoff",
                               "final_fresh_grant_handoff", "final_live_forward_handoff"}
                and not response_adjusted and not view_adjusted and not moving_handoff
                and trial_phase in {"disabled", "tracking"} and not zero_held
                and min(applied) >= 0 and sum(applied) > 0)
            write_decision = (LinearWriteDecision(.5*abs(sum(applied))) if not sum(applied) else
                self._linear_packet_write_limit(
                    final_linear, uid, .5*sum(applied),
                    source_grant=(getattr(self, "_follow_source_grant", None)
                                  if self._periodic_follow_writing else None),
                    feedback=feedback, forward_pair=applied if ordinary_contraction else None,
                    planned_revision=revision, yaw_required=applied[0] != applied[1],
                    adopted_intent=adopted_intent, adopted_policy=adopted_policy,
                    expected_receipt=(handoff_expected[0] if handoff_expected is not None else ...),
                    expected_stop_generation=(handoff_expected[1]
                        if handoff_expected is not None else ...)))
            write_limit = write_decision.limit_rpm if write_decision is not None else None
            if write_limit is not None and 0 < write_limit < .5*sum(applied):
                # Same physical grant, same yaw and all final STOP/feedback
                # checks already passed. A legal lower cap is a contraction,
                # not an instruction to stop. Never increase either wheel or
                # reuse an assist/reversal/parking plan here.
                pair = write_decision.forward_pair
                if pair is None:
                    write_limit = None
                else:
                    terminal_contraction = (applied, pair, write_limit)
                    applied = pair
                    base, yaw = .5*sum(applied), .5*(applied[0]-applied[1])
                    self._periodic_follow_axes = (uid, revision, base, yaw)
            if sum(applied) and (write_limit is None or .5*abs(sum(applied)) > write_limit):
                # A producer can also publish during the quiet final cap
                # calculation. Retry admission outside the serial lock, not
                # by patching the old pair or interpreting publication as STOP.
                if self._request_follow_commit_rebuild(
                        uid, commit_publication, expired_forward_yaw=bool(
                            self._follow_write_veto_reason == "terminal_yaw_expired"
                            and min(applied) >= 0 and sum(applied) > 0)):
                    return False
                # Preserve a newer STOP/parking owner; otherwise revoke this
                # expired packet with zero. Do not retry a blocking reader or
                # print diagnostics between the last clock and motor write.
                if not (getattr(self.owner, "_brake_hold_active", False)
                        or getattr(self.owner, "_near_yaw_park_request", None) is not None
                        or getattr(self.owner, "_runtime_shutdown_requested", False)
                        or getattr(self.owner, "_explicit_stop_requested", False)
                        or getattr(self.owner, "stop_action_execution", False)
                        or getattr(self.owner, "person_detected_flag", False)
                        or getattr(self.owner, "running", True) is False
                        or getattr(self.owner, "search_state", "none") != "none"
                        or getattr(self.owner._follow_controller, "search_state", "none") != "none"
                        or getattr(self, "_search_reacquire_brake_request", None) is not None
                        or getattr(self.backend, "last_speed_receipt", None) is not previous_speed_receipt
                        or getattr(self.backend, "stop_write_generation", 0) != previous_stop_generation
                        or (handoff_expected is not None
                            and (getattr(self.backend, "last_speed_receipt", None)
                                 is not handoff_expected[0]
                                 or getattr(self.backend, "stop_write_generation", 0)
                                    != handoff_expected[1]))
                        or getattr(self.backend, "normal_zero_hold", False)
                        or getattr(self.backend, "parking_current_a", 0)
                        or getattr(self.backend, "_parking_current_uncertain", False)
                        or getattr(self.backend, "parking_release_fault", None)
                        or getattr(self.backend, "motion_write_fault", None)):
                    self._follow_motor_call("send_targets", 0, 0, "FOLLOW_WRITE_DEADLINE_EXPIRED")
                self._visible_wheel_guard.reset()
                self.logger.info("follow_wheel_veto uid=%s reason=physical_deadline_or_grant_changed_at_write "
                                 "detail=%s planned_pair=%s write_limit_rpm=%s "
                                 "depth_final_quiet_veto=%s",
                                 uid, self._follow_write_veto_reason, applied, write_limit,
                                 getattr(self.owner, "_last_quiet_depth_veto", None)
                                 if final_linear is None else None)
                return False
            # A normal FOLLOW20 zero is not a safety STOP. The positive yaw
            # producer can publish while the old zero packet is being built.
            # Check again at the physical write boundary, after the earlier
            # zero-plan check and any intervening callback. Never suppress a
            # wheel-guard zero, an explicit stop, or a different UID's zero.
            # Forward-loss handoff also labels an originally requested zero
            # as feedback_wait; only that case (not a blocked pivot) shares
            # the ordinary zero-plan supersession check.
            if (applied == (0, 0) and label == "FOLLOW20"
                    and self._periodic_follow_writing
                    and (reason == "explicit_zero" or (
                        reason == "forward_loss_feedback_wait"
                        and requested == (0, 0) and loss_input == (0, 0)))
                    and self._periodic_follow_axes is not None
                    and self._periodic_follow_axes[2:] == (0, 0)):
                latest_axes = self._read_follow_axes(time.monotonic())
                newer_motion = bool(
                    latest_axes is not None and latest_axes != self._periodic_follow_axes
                    and latest_axes[0] == uid
                    and self.owner._follow_controller.active_target_id == uid
                    and self._periodic_follow_active()
                    and not getattr(self.owner, "_brake_hold_active", False)
                    and getattr(self.owner, "_near_yaw_park_request", None) is None
                    and (latest_axes[2] > 0 or (latest_axes[2] == 0 and latest_axes[3] != 0
                         and self.owner._has_fresh_lateral_yaw(uid))))
                if newer_motion:
                    if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                        self._follow_motor_call("send_stop", "follow20_zero_superseded_hard_stop", mode="emergency")
                        return False
                    rebuilt = self._request_follow_axes_rebuild(uid)
                    self.logger.info(
                        "follow_wheel_retry uid=%s reason=zero_superseded_at_write "
                        "old_packet_discarded=%s immediate_rebuild=%s motion_authorized=False",
                        uid, rebuilt, rebuilt)
                    if rebuilt:
                        return False
                    # No retry credit (or a lease changed during recheck):
                    # retain the safe zero instead of letting old motion run.
            if (sum(applied) > 0 and handoff_expected is not None
                    and (getattr(self.backend, "last_speed_receipt", None)
                         is not handoff_expected[0]
                         or getattr(self.backend, "stop_write_generation", 0)
                            != handoff_expected[1])):
                return False
            # Zero packets need the same atomic receipt/guard accounting as
            # positive packets. Keep heavy zero supersession readers above
            # outside the serial lock, then protect write through note_sent.
            if not self._begin_follow_commit():
                return False
            if (applied == (0, 0) and label == "FOLLOW20"
                    and self._periodic_follow_writing
                    and requested == loss_input == (0, 0)
                    and reason in {"explicit_zero", "forward_loss_feedback_wait"}):
                if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                    self._follow_motor_call(
                        "send_stop", "follow20_zero_commit_hard_stop", mode="emergency")
                    return False
                # Cache-only publication comparison under I/O ownership.
                # This does not authorize motion or mint a write receipt:
                # the shared three-attempt service budget releases the lock
                # before the new axes, brake cap and wheel guards are read.
                # A crossing/braking zero is deliberately outside this path.
                if self._request_follow_commit_rebuild(uid, zero_plan_publication):
                    self.logger.info("follow_wheel_retry uid=%s "
                        "reason=requested_zero_superseded_at_commit "
                        "old_packet_discarded=True immediate_rebuild=True "
                        "packet_written=False receipt_renewed=False", uid)
                    return False
            decision_scope = (self._zero_packet_decision(
                "wheel_composition", reason, requested_forward_rpm=requested,
                guarded_forward_rpm=guarded_request, applied_forward_rpm=applied,
                forward_loss_reason=loss_reason, handoff_reason=handoff_reason)
                if applied == (0, 0) else nullcontext())
            with decision_scope:
                if not self._follow_motor_call(
                        "send_targets", applied[0] * ls, applied[1] * rs, label,
                        max_target_override=max_target_override, history_uid=uid):
                    return False
        sent_at, response = self._note_follow_packet(
            uid, applied, final_linear, feedback, (ls, rs), previous_speed_receipt,
            previous_sent, packet_written=packet_written, trial=trial)
        if view_adjusted:
            self.logger.info("view_retention uid=%s capture_frame_id=%s phase=%s input=%s applied=%s "
                             "deadline=%s remaining_ms=%.1f common_removed_rpm=%.1f "
                             "motor_authority_renewed=False packet_written=%s",
                             uid, view_intent.capture_frame_id, view_phase, view_input, applied,
                             self._view_retention.deadline,
                             max(0., self._view_retention.deadline-sent_at)*1000.,
                             .5*sum(view_input)-.5*sum(applied), packet_written)
        if coalesced_yaw is not None:
            # Log only after I/O; logging must not consume the newly checked
            # physical lease between final validation and the speed write.
            self.logger.info("follow_wheel_yaw_coalesced uid=%s old_axes=%s new_axes=%s "
                             "old_pair=%s applied_pair=%s base_preserved=True retries_added=0",
                             uid, *coalesced_yaw)
        if coalesced_base is not None:
            self.logger.info("follow_wheel_base_coalesced uid=%s old_axes=%s new_axes=%s "
                             "old_pair=%s applied_pair=%s grant_renewed=False retries_added=0",
                             uid, *coalesced_base)
        if coalesced_axes is not None:
            self.logger.info("follow_wheel_axes_coalesced uid=%s old_axes=%s new_axes=%s "
                             "old_pair=%s applied_pair=%s wheels_increased=False grant_renewed=False",
                             uid, *coalesced_axes)
        if straight_handoff is not None:
            self.logger.info("follow_wheel_straight_handoff uid=%s old_axes=%s new_axes=%s "
                             "old_pair=%s applied_pair=%s wheels_increased=False "
                             "grant=%s packet_written=%s",
                             uid, *straight_handoff, getattr(self, "_follow_source_grant", None),
                             packet_written)
        if fresh_grant_handoff is not None:
            self.logger.info("follow_wheel_fresh_grant_handoff uid=%s old_axes=%s new_axes=%s "
                             "old_pair=%s applied_pair=%s wheels_increased=False "
                             "grant=%s packet_written=%s yaw_neutralized=%s",
                             uid, *fresh_grant_handoff,
                             getattr(self, "_follow_source_grant", None), packet_written,
                             fresh_grant_neutral)
        if early_fresh_handoff is not None and packet_written:
            self.logger.info("follow_wheel_early_grant_adopted uid=%s old_axes=%s new_axes=%s "
                             "grant=%s packet_written=True", uid, *early_fresh_handoff,
                             getattr(self, "_follow_source_grant", None))
        if terminal_contraction is not None:
            self.logger.info("follow_wheel_terminal_cap uid=%s old_pair=%s applied_pair=%s "
                             "limit_rpm=%.2f grant_renewed=False", uid, *terminal_contraction)
        if revision_coalesced is not None:
            self.logger.info("follow_wheel_revision_coalesced uid=%s sample_ts=%s "
                             "old_axes=%s new_axes=%s pair_unchanged=True retries_added=0",
                             uid, revision_coalesced[0][3], *revision_coalesced[2:])
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
            "response_adjusted=%s ordinary_mode_limited=%s "
            "view_retention_phase=%s view_retention_adjusted=%s view_retention_deadline=%s "
            "depth_initial_quiet_veto=%s",
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
            not response_adjusted and not view_adjusted and (base != original_base or yaw != original_yaw),
            view_phase, view_adjusted, self._view_retention.deadline,
            getattr(self.owner, "_last_quiet_depth_veto", None) if linear is None else None,
        )
        return True

    def _note_follow_packet(self, uid, applied, linear, feedback, signs,
                            previous_receipt, previous_sent, *, packet_written, trial):
        """Shared completed-I/O bookkeeping; never record an unsent plan."""
        self._forward_execution_anchor = None
        receipt = getattr(self.backend, "last_speed_receipt", None)
        if (packet_written and receipt is not None and min(applied) >= 0
                and sum(applied) > 0 and len(linear or ()) == 4
                and linear[0] == "forward" and linear[2] == uid
                and (receipt.left_rpm, receipt.right_rpm)
                    == (applied[0]*signs[0], applied[1]*signs[1])):
            self._forward_execution_anchor = ForwardExecutionAnchor(
                uid, linear[3], .5*sum(applied), receipt.completed_at, receipt)
        sent_at = time.monotonic()
        with self._executed_speed_history_lock:
            self._continuation_executed_speed_history = record_completed_speed(
                getattr(self, "_continuation_executed_speed_history", ()), uid=uid,
                applied=applied, signs=signs, receipt=receipt,
                previous_receipt=previous_receipt, now=sent_at,
                packet_written=packet_written, feedback=feedback)
        note_handoff_zero_write(self, uid=uid, previous_receipt=previous_receipt,
                                packet_written=packet_written, feedback=feedback, now=sent_at)
        if not hasattr(self, "_wheel_diff_response"):
            self._wheel_diff_response = WheelDifferentialResponse()
        response = self._wheel_diff_response.observe_and_note(
            uid, applied, sent_at, previous_sent, feedback, packet_written=packet_written)
        if packet_written:
            self._visible_wheel_guard.note_sent(applied, sent_at)
            self._forward_loss_handoff.note_sent(applied, sent_at)
            self._turn_buildup.note_sent(uid, applied, sent_at, previous_sent)
            self._view_retention.note_sent(uid, applied, sent_at, previous_sent, receipt)
            if trial is not None:
                trial.record(uid, applied, sent_at)
        return sent_at, response

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
        axes = self._read_follow_axes(now)
        return bool(
            intent is not None
            and capture_frame_id > 0 and intent.capture_frame_id == capture_frame_id
            and self._read_depth_linear(uid, now=now) is None
            and axes is not None and axes[0] == uid and axes[2] == 0
            and math.isfinite(axes[3]) and axes[3] != 0)

    def can_coalesce_follow_queue_refresh(self, actions):
        """Coalesce the adopted forward/steer stream; readers still own axes.

        This changes no command, revision, lease or motor state. In particular,
        a STOP already popped from the queue has advanced the publication
        revision but has not necessarily replaced ``current_command`` yet.
        Both locks are probes: a busy executor must not stall the producer.
        Forward and left/right steering are labels on the SAME periodic
        two-axis writer, not different motor modes. Re-enqueuing each label
        change used to wait on serial I/O while holding the control lock.
        Reverse, pivot, STOP and all lifecycle transitions are excluded.
        A zero ACK from a temporary evidence gap is not such a transition:
        the adopted writer still owns this stream, and checks NEW authority
        before resuming. Requiring a positive old ACK here made recovery
        publication wait for serial I/O while holding the control mutex.
        """
        owner, symbols = self.owner, self.symbols
        follow_actions = (symbols.forward, symbols.steer_left, symbols.steer_right)
        if (len(actions) != 1
                or actions[0] not in follow_actions
                or getattr(owner, "_last_command_source_module", None)
                    not in {"depth30", "lateral_intent_loop"}):
            return False
        command_lock = getattr(owner, "command_lock", None)
        queue_lock = getattr(owner, "action_queue_lock", None)
        if command_lock is None or queue_lock is None or not command_lock.acquire(blocking=False):
            return False
        try:
            if not queue_lock.acquire(blocking=False):
                return False
            try:
                command = getattr(self, "_current_action_snapshot", None)
                uid = getattr(getattr(owner, "_follow_controller", None), "active_target_id", None)
                clock_axes = self._follow_wheel_clock.last_axes
                if (not isinstance(command, ActionCommandSnapshot)
                        or command.action not in follow_actions
                        or owner.current_command != command.action
                        or command.protected_stop or command.soft_stop
                        or command.uid != uid or uid is None
                        or command.source_module not in {"depth30", "lateral_intent_loop"}
                        or command.revision != getattr(owner, "_action_command_revision", None)
                        or not owner.action_queue.empty()
                        or clock_axes is None or clock_axes[0] != uid
                        or not math.isfinite(clock_axes[2]) or clock_axes[2] < 0
                        or self._follow_wheel_clock.last_sent is None
                        or getattr(owner, "_vision_control_state", "") not in {
                            "target_visible", "target_visible_depth_valid"}
                        or any(getattr(owner, name, False) for name in (
                            "_use_soft_stop_next", "_soft_stop_active", "_explicit_stop_requested",
                            "_runtime_shutdown_requested", "_brake_hold_active",
                            "stop_action_execution", "person_detected_flag"))
                        or getattr(owner, "_near_yaw_park_request", None) is not None
                        or getattr(self, "_search_reacquire_brake_request", None) is not None
                        or getattr(owner, "_search_handoff_uid", None) == uid
                        or self.backend.normal_zero_hold or self.backend.parking_current_a
                        or self.backend._parking_current_uncertain
                        or self.backend.parking_release_fault or self.backend.motion_write_fault
                        or not self._periodic_follow_active()):
                    return False
                now = time.monotonic()
                linear = owner._fresh_depth_linear_snapshot(uid, now=now)
                axes = owner._follow_wheel_axes(now)
                return bool(
                    len(linear or ()) == 4 and linear[0] == "forward" and linear[2] == uid
                    and math.isfinite(linear[1]) and linear[1] > 0
                    and math.isfinite(linear[3]) and 0 <= now-linear[3] <= MAX_FORWARD_DEPTH_TTL_SEC
                    and axes is not None and axes[0] == uid
                    and axes[1] == getattr(owner, "_lateral_yaw_revision", None)
                    and all(math.isfinite(value) for value in axes[2:])
                    and axes[2] > 0
                    and axes[2] <= linear[1]*self.config.motor_forward_max_target_rpm/100.
                    and self._current_action_snapshot is command
                    and command.revision == getattr(owner, "_action_command_revision", None)
                    and owner.current_command == command.action
                    and owner._follow_controller.active_target_id == uid)
            finally:
                queue_lock.release()
        finally:
            command_lock.release()

    def _periodic_follow_scope_active(self):
        """Cheap owner/config scope only; this never reads or grants axes."""
        return bool(
            getattr(self.config, "follow_wheel_period_sec", 0.0) >= .05
            and self._visible_wheel_control_active()
            and not getattr(self.owner, "stop_action_execution", False)
            and not getattr(self.owner, "person_detected_flag", False)
            and callable(getattr(self.owner, "_follow_wheel_axes", None))
        )

    def _periodic_follow_active(self):
        # Preserve the public predicate's complete live-axis semantics.
        return bool(self._periodic_follow_scope_active()
                    and self._read_follow_axes(time.monotonic()) is not None)

    def _forward_visual_permits(self, uid, sample_timestamp, now):
        proof = getattr(self.owner, "_validated_visual_observation", None)
        return proof is None or (isinstance(proof, ValidatedVisualObservation)
            and proof.permits_depth(uid, sample_timestamp, now))

    def _follow_revocation_marker(self):
        """Cheap publication token; never evaluates a depth cap or takes a lock."""
        owner = self.owner
        controller = getattr(owner, "_follow_controller", None)
        store = getattr(owner, "_lateral_intent_store", None)
        raw = getattr(owner, "_depth30_linear_snapshot", None)
        sample_stamp = raw[3] if isinstance(raw, tuple) and len(raw) == 4 else None
        return (
            tuple(getattr(owner, name, None) for name in (
                "running", "search_state", "_vision_control_state",
                "_explicit_stop_requested", "_runtime_shutdown_requested",
                "_brake_hold_active", "stop_action_execution", "person_detected_flag",
                "_lateral_yaw_revision", "_action_command_revision")),
            getattr(controller, "active_target_id", None),
            getattr(controller, "search_state", None),
            tuple(id(getattr(owner, name, None)) for name in (
                "_depth30_linear_snapshot", "_depth30_linear_timing",
                "_depth30_prepared_timing", "_detector_identity_lease",
                "_validated_visual_observation",
                "_near_yaw_park_request", "_lateral_turn_response_policy")),
            id(getattr(store, "_intent", None)),
            self._forward_visual_permits(getattr(controller, "active_target_id", None),
                sample_stamp, time.monotonic()),
        )

    def _read_follow_axes(self, now):
        reader = getattr(self, "_follow_axes_reader", None) or self.owner._follow_wheel_axes
        return reader(now)

    def _read_depth_linear(self, uid, *, now=None):
        reader = getattr(self, "_depth_linear_reader", None) or self.owner._fresh_depth_linear_snapshot
        linear = reader(uid, now=now)
        if (self._distance_brake_sample_floor > 0 and linear is not None
                and linear[0] == "forward" and linear[3] <= self._distance_brake_sample_floor):
            return None
        return linear

    def _distance_brake_evidence(self, now):
        """Current immutable measurement only; an old veto string is not proof.

        Return (assessment, cap, reason, live_grant). PI can request braking
        without publishing a positive grant. It cannot authorize resumption.
        No lock acquisition, serial read or second braking model here.
        """
        controller = getattr(self.owner, "_follow_controller", None)
        uid = getattr(controller, "active_target_id", None)
        feedback = getattr(self, "_steering_feedback", None)
        if not wheel_feedback_valid(feedback, now):
            return None
        outer = max(abs(feedback.left_forward_rpm), abs(feedback.right_forward_rpm))
        marker = self._follow_grant_marker()
        raw, timing = marker if marker is not None else (None, None)
        assessment = getattr(timing, "braking_assessment", None)
        result = getattr(controller, "last_distance_pid_result", None)
        pi_assessment = getattr(result, "pi_braking_assessment", None)
        if (isinstance(pi_assessment, SampleBrakingAssessment)
                and pi_assessment.valid_for(uid, getattr(controller, "_distance_pid_last_sample_timestamp", None))
                and (not isinstance(assessment, SampleBrakingAssessment)
                     or pi_assessment.sample_timestamp > assessment.sample_timestamp)):
            assessment, raw, timing = pi_assessment, None, None
        if (not isinstance(assessment, SampleBrakingAssessment)
                or assessment.uid != uid
                or not assessment.checked_at <= now <= assessment.sample_timestamp + MAX_FORWARD_DEPTH_TTL_SEC):
            return None
        if (raw is not None and len(raw) == 4 and raw[0] == "forward" and raw[1] > 0
                and assessment.valid_for(uid, raw[3])):
            bound = getattr(timing, "continuation_speed_bound_m_s", None)
            if (not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound <= 0
                    or now > timing.depth_expires_at):
                return None
            budget = assessment.budget(now, outer,
                authorized_rpm=raw[1]*self.config.motor_forward_max_target_rpm/100.,
                execution_bound_rpm=bound*60./assessment.circumference_m,
                tightened_distance_m=timing.continuation_distance_m)
            return assessment, budget.cap_rpm, budget.reason, raw
        # A PI zero has no admitted execution budget. Its original fresh
        # assessment can prove braking, never a later positive permission.
        if (assessment is not pi_assessment or getattr(result, "output_rpm", None) != 0
                or getattr(result, "pi_stationary_preview_status", None) != "momentum_brake"):
            return None
        budget = assessment.budget(assessment.checked_at, max(assessment.outer_rpm, outer))
        return assessment, budget.cap_rpm, budget.reason, None

    def _distance_brake_forward_motion(self):
        feedback = getattr(self, "_steering_feedback", None)
        return bool(not self.config.rotation_only and self._visible_wheel_control_active()
            and feedback is not None
            and min(feedback.left_forward_rpm, feedback.right_forward_rpm) >= 0.
            and .5*(feedback.left_forward_rpm+feedback.right_forward_rpm) > 1.
            and not getattr(self.backend, "parking_current_a", 0.)
            and not getattr(self.backend, "_parking_current_uncertain", False))

    def _start_distance_brake(self, assessment, now):
        """Called with motor I/O ownership. No zero-speed preface or FREE.

        Emergency STOP is a driver mode, not a claimed deceleration. Do not
        introduce 10A parking current: its high-speed benefit is uncalibrated.
        """
        self.backend.send_stop("distance_momentum_brake", mode="emergency", preserve_zero=True)
        sent_at = time.monotonic()
        self._distance_brake_episode = (assessment, ParkSettlingEvidence(assessment, sent_at))
        self._distance_brake_sample_floor = sent_at
        self._distance_brake_stop_generation = self.backend.stop_write_generation
        self._forward_execution_anchor = None
        self._follow_wheel_clock.reset()
        self._visible_wheel_guard.reset()
        self.logger.info(
            "distance_brake_start uid=%s sample_ts=%.6f sent_at=%.6f "
            "stop_mode=emergency zero_keepalive_suppressed=True parking_current_increased=False "
            "physical_stillness=unverified",
            assessment.uid, assessment.sample_timestamp, sent_at)

    def _service_distance_brake(self):
        """Keep braking until observed quiet OR independently safe new motion.

        No fixed-duration release, no renewed depth/identity lease. Returning
        False only hands control back to the normal complete packet checks.
        """
        # Ordinary live following acquires NO extra motor lock and invokes no
        # publisher/continuation callback. Recheck evidence under I/O ownership
        # only when it actually asks for braking (or an episode already exists).
        if self._distance_brake_episode is None:
            if (self._distance_brake_stop_generation is not None
                    and self.backend.stop_write_generation != self._distance_brake_stop_generation):
                with self.owner.motor_io_lock:
                    # Covers a STOP between episode release and the normal
                    # planner, too. No speed/STOP write replaces that owner.
                    self._distance_brake_sample_floor = time.monotonic()
                    self._distance_brake_stop_generation = self.backend.stop_write_generation
                return True
            candidate = self._distance_brake_evidence(time.monotonic())
            if candidate is None or candidate[2] != "shared_braking_momentum":
                return False
        with self.owner.motor_io_lock:
            now = time.monotonic()
            episode = self._distance_brake_episode
            protected = (any(getattr(self.owner, name, False) for name in (
                "_explicit_stop_requested", "_runtime_shutdown_requested", "_brake_hold_active",
                "stop_action_execution", "person_detected_flag"))
                or getattr(self.owner, "_near_yaw_park_request", None) is not None
                or self._search_reacquire_brake_request is not None
                or getattr(self.backend, "motion_write_fault", None)
                or getattr(self.backend, "parking_release_fault", None))
            # Explicit safety/parking keeps its own lifecycle and stop mode.
            if protected:
                return False
            if (episode is not None and self.backend.stop_write_generation
                    != self._distance_brake_stop_generation):
                # STOP can reach hardware before its producer publishes its
                # Python flags. A newer writer owns that mode immediately.
                # Its exact completion time is not stored by the backend, so
                # conservatively start a NEW observation boundary now. Never
                # resume using a grant that only postdates our earlier STOP.
                assessment, _ = episode
                self._distance_brake_episode = (
                    assessment, ParkSettlingEvidence(assessment, now))
                self._distance_brake_sample_floor = now
                self._distance_brake_stop_generation = self.backend.stop_write_generation
                self.logger.info("distance_brake_stop_owner_changed uid=%s generation=%s "
                                 "new_sample_after=%.6f old_motion_discarded=True",
                                 assessment.uid, self._distance_brake_stop_generation, now)
                return True
            if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                self.backend.send_stop("distance_brake_hard_stop", mode="emergency")
                return True
            proof = self._distance_brake_evidence(now)
            feedback = getattr(self, "_steering_feedback", None)
            if episode is None:
                if (proof is None or proof[2] != "shared_braking_momentum"
                        or not self._distance_brake_forward_motion()):
                    return False
                self._start_distance_brake(proof[0], now)
                return True
            assessment, settling = episode
            quiet = settling.observe(feedback if wheel_feedback_valid(feedback, now) else None, now)
            new_motion = bool(proof is not None and proof[3] is not None
                and proof[0].uid == assessment.uid
                and proof[0].sample_timestamp > settling.sent_at
                and proof[2] == "shared_braking_cap" and proof[1] > 0
                and getattr(self.owner, "_depth30_continuation_veto", None)
                    != (proof[0].uid, proof[0].sample_timestamp)
                and tuple((getattr(self.owner, "_depth30_read_veto", None) or ())[:2])
                    != (proof[0].uid, proof[0].sample_timestamp)
                and self._visible_wheel_control_active()
                and motion_identity_live(self.owner, assessment.uid, now)
                and self._forward_visual_permits(assessment.uid, proof[0].sample_timestamp, now))
            if quiet or new_motion:
                self._distance_brake_episode = None
                self.logger.info("distance_brake_release uid=%s reason=%s elapsed_ms=%.1f "
                                 "motion_authorized=False full_packet_checks_required=True",
                                 assessment.uid, "fresh_independent_budget" if new_motion else "encoder_quiet",
                                 (now-settling.sent_at)*1000.)
                return False
            # Another STOP refresh can clear the legacy zero suppression flag.
            # Reassert only the driver STOP mode, never a speed-zero packet.
            if not getattr(self.backend, "normal_zero_hold", False):
                self.backend.send_stop("distance_momentum_brake_hold", mode="emergency", preserve_zero=True)
                self._distance_brake_stop_generation = self.backend.stop_write_generation
            return True

    def _service_follow_wheels(self):
        # Capture before handler locks/formatting, flush only after all motor
        # critical sections exit. The production sink remains asynchronous.
        with deferred_diagnostics(self.logger):
            # A second service caller must not interleave mutable wheel-guard
            # planning. It does not hold or wait for the motor/encoder lock.
            if not self._follow_planning_lock.acquire(blocking=False):
                return
            self._follow_planning_thread_id = threading.get_ident()
            self._follow_planning_offlock = True
            try:
                return self._service_follow_wheels_deferred()
            finally:
                self._follow_planning_offlock = False
                self._follow_planning_thread_id = None
                self._follow_plan_token = None
                self._follow_planning_lock.release()

    def _service_follow_wheels_deferred(self):
        """Action-thread sole normal-follow writer; safety is checked each loop."""
        if self._service_runtime_shutdown():
            return
        if self._service_distance_brake():
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
            revocation_marker = self._follow_revocation_marker()
            still_inactive = not self._periodic_follow_active()
            with self.owner.motor_io_lock:
                if (not self._periodic_follow_scope_active()
                        or (still_inactive and revocation_marker == self._follow_revocation_marker())):
                    # Revoke once before search/reverse/stop takes over.
                    if getattr(self.owner, "_runtime_shutdown_requested", False):
                        # Shutdown won the lock after the entry check. A zero
                        # target would re-enter speed mode after another STOP.
                        self._shutdown_stop_attempt_at = time.monotonic()
                        self.backend.send_stop("runtime_shutdown", mode=self.config.safety_stop_mode)
                    elif (getattr(self.backend, "last_speed_receipt", None)
                          is not self._follow_wheel_last_receipt
                          or getattr(self.backend, "stop_write_generation", 0)
                          != self._follow_wheel_last_stop_generation):
                        # A STOP or another motor packet already superseded the
                        # last follow write. A speed-mode zero would undo its
                        # mode or erase the new writer's command.
                        pass
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
        # Serialize planning separately from physical I/O. Each final commit
        # checks its original ownership token and current evidence again.
        sent = False
        tick = None
        lock_times = self._follow_commit_lock_times = []
        planning_times = []
        retry_receipt = None
        retry_stop_generation = None
        initial_uid = None
        reason = None
        tick_now = None
        self._follow_revision_reference = None
        self._follow_revision_coalesced = None
        self._follow_terminal_forward_used = False
        self._follow_terminal_forward_requested = False
        try:
            # One shared budget for ordinary and commit-boundary rebuilds;
            # every unsuccessful attempt exits _follow_planning_attempt (and
            # releases motor_io_lock) before the next complete admission.
            for attempt in range(3):
                wait_started = time.monotonic()
                # Capture motor ownership atomically, then release the serial
                # lock before axes readers, prediction and wheel guards.
                with self.owner.motor_io_lock:
                    token = (getattr(self.backend, "last_speed_receipt", None),
                             getattr(self.backend, "stop_write_generation", 0))
                    if attempt and (token[0] is not retry_receipt
                                    or token[1] != retry_stop_generation):
                        break
                    self._follow_plan_token = token
                with self._follow_planning_attempt():
                    acquired_at = time.monotonic()  # Planning, NOT serial ownership.
                    try:
                        if attempt:
                            # A STOP or another completed write owns the gap.
                            # Do not overwrite it, even with a speed-zero
                            # packet which would re-enter speed mode.
                            if (getattr(self.backend, "last_speed_receipt", None) is not retry_receipt
                                    or getattr(self.backend, "stop_write_generation", 0) != retry_stop_generation):
                                break
                            if (self._near_yaw_park_blocks_write("FOLLOW20_REBUILD")
                                    or getattr(self.owner, "_brake_hold_active", False)
                                    or getattr(self.owner, "_explicit_stop_requested", False)
                                    or getattr(self.owner, "_runtime_shutdown_requested", False)
                                    or getattr(self.owner, "stop_action_execution", False)
                                    or getattr(self.owner, "person_detected_flag", False)
                                    or getattr(self, "_search_reacquire_brake_request", None) is not None):
                                break
                        # The complete axes read immediately below is this
                        # attempt's admission. Do not compute its quiet Depth
                        # braking cap once for a boolean and again for the
                        # plan while holding the serial lock. None still
                        # follows the original first/rebuild revoke branches.
                        if not self._periodic_follow_scope_active():
                            if attempt:
                                self._follow_motor_call("send_targets", 0, 0, "FOLLOW20_REBUILD_REVOKED")
                                self._visible_wheel_guard.reset()
                            break
                        now = time.monotonic()
                        source_grant = self._follow_grant_marker()
                        axes = self._read_follow_axes(now)
                        if not self._same_follow_grant_marker(source_grant, self._follow_grant_marker()):
                            source_grant = None
                        if attempt:
                            if (axes is None or axes[0] != initial_uid
                                    or not (axes[2] > 0 or (axes[2] == 0 and axes[3] != 0
                                            and self.owner._has_fresh_lateral_yaw(initial_uid)))):
                                self._follow_motor_call("send_targets", 0, 0, "FOLLOW20_REBUILD_REVOKED")
                                self._visible_wheel_guard.reset()
                                break
                            if self.hard_stop_check(getattr(self.owner, "current_command", None)):
                                self._follow_motor_call("send_stop", "follow20_rebuild_hard_stop", mode="emergency")
                                break
                        else:
                            if axes is None:
                                break
                            reason = clock.reason(now, axes)
                            if reason is None:
                                break
                            tick_now = now
                            initial_uid = axes[0]
                            if clock.last_axes is not None and initial_uid != clock.last_axes[0]:
                                if self._near_yaw_park_blocks_write("FOLLOW20_UID_ZERO"):
                                    break
                                self._follow_motor_call("send_targets", 0, 0, "FOLLOW20_UID_ZERO")
                                clock.reset()
                                self._visible_wheel_guard.reset()
                                self._forward_loss_handoff.reset()
                                break
                        cap = int(self.config.motor_forward_max_target_rpm)
                        ls = self.backend.wheel_raw_state_to_target("left", 1, 0x01)
                        rs = self.backend.wheel_raw_state_to_target("right", 1, 0x01)
                        self._periodic_follow_writing = True
                        self._follow_service_attempt = attempt
                        self._follow_axes_rebuild_allowed = attempt == 0
                        self._follow_commit_rebuild_allowed = attempt < 2
                        self._follow_axes_rebuild_requested = False
                        self._periodic_follow_axes = axes
                        self._follow_source_grant = source_grant
                        self._follow_attempt_publication = (
                            source_grant, axes[1], self._follow_intent_snapshot(
                                getattr(self.owner, "_lateral_intent_store", None)),
                            getattr(self.owner, "_lateral_turn_response_policy", None))
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
                        if self._follow_terminal_forward_requested:
                            self._follow_terminal_forward_requested = False
                            sent = self._send_ordinary_snapshot_forward(
                                uid, "FOLLOW20", max_target_override=cap, straight_fallback=True)
                        if sent:
                            axes = self._periodic_follow_axes
                            uid, revision, base, yaw = axes
                            previous_sent = clock.last_sent
                            clock.sent(time.monotonic(), axes)
                            self._follow_wheel_last_receipt = getattr(
                                self.backend, "last_speed_receipt", None)
                            self._follow_wheel_last_stop_generation = getattr(
                                self.backend, "stop_write_generation", 0)
                            tick = (getattr(self.owner, "_last_command_capture_frame", None),
                                    uid, revision, base, yaw, reason,
                                    None if previous_sent is None else round((tick_now-previous_sent)*1000, 2),
                                    clock.period * 1000)
                        if not self._follow_axes_rebuild_requested:
                            break
                        if not self._follow_plan_owns_motor():
                            break  # A writer/STOP during planning owns the gap.
                        retry_receipt, retry_stop_generation = self._follow_plan_token
                    finally:
                        if not sent and not self._follow_plan_owns_motor():
                            # A STOP may reset a guard while an off-lock
                            # computation is still mutating it. Discard those
                            # uncommitted reversal/continuity states before
                            # another normal planner may enter. Search/STOP
                            # motor ownership itself is never changed here.
                            cleanup_lock = (nullcontext() if self._follow_commit_locked
                                            else self.owner.motor_io_lock)
                            with cleanup_lock:
                                self._visible_wheel_guard.reset()
                                self._forward_loss_handoff.reset()
                                self._forward_execution_anchor = None
                                clock.reset()
                        planning_times.append((time.monotonic()-acquired_at)*1000.)
        finally:
            self._periodic_follow_writing = False
            self._follow_axes_rebuild_allowed = False
            self._follow_commit_rebuild_allowed = False
            self._follow_revision_reference = None
            self._follow_revision_coalesced = None
            self._follow_source_grant = None
            self._follow_attempt_publication = None
            self._follow_base_receipt_reference = None
        # Execution receipts/guards/clock were committed while locked. These
        # observational messages must not delay encoder polling or STOP I/O.
        if tick is not None:
            self.logger.info(
                "follow_wheel_tick capture_frame_id=%s uid=%s revision=%s "
                "base_rpm=%.1f yaw_rpm=%.1f reason=%s interval_ms=%s period_ms=%.1f", *tick)
        if reason is not None:
            self.logger.info(
                "follow_wheel_lock_timing attempts=%d lock_wait_ms=%.2f lock_hold_ms=%.2f "
                "max_hold_ms=%.2f packet_written=%s retry_gap_released=%s "
                "planning_ms=%.2f planning_outside_motor_lock=True",
                len(planning_times), sum(v[0] for v in lock_times), sum(v[1] for v in lock_times),
                max((v[1] for v in lock_times), default=0.), bool(sent), len(planning_times) > 1,
                sum(planning_times))

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
                    sent = self._send_follow_wheel_targets(
                        left_target,
                        right_target,
                        "DRIVE",
                        max_target_override=(
                            int(c.motor_forward_max_target_rpm)
                            if c.motor_forward_max_target_rpm > 0
                            else None
                        ),
                    )
                    if sent is False:
                        return False
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
            if p > 0 and getattr(self.owner, "_detector_identity_lease", None) is not None:
                # The percent writer has no detector identity deadline gate.
                return False
            self.backend.send_diff(p, state, p, state, "DRIVE")
            if p <= 0:
                self._note_search_retry_zero_sent()
            self._forward_coast_snapshot = (p, bool(allow_below_min)) if p > 0 else None
        return True

    def send_yaw_only(
        self, correction_rpm: int, *, yaw_revision: Optional[int] = None,
        reverse_guard: bool = False,
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
                if reverse_guard and self._detector_reverse_write_blocked():
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
                if reverse_guard and self._detector_reverse_write_blocked():
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

    def _detector_reverse_write_blocked(self) -> bool:
        """Motor-lock check: detector continuation never authorizes reverse.

        None retains the existing full-verification reverse policy. A live,
        expired or rejected detector lease all require another full result.
        Do not replace a concurrent STOP/parking owner with speed-mode zero.
        """
        owner = self.owner
        if getattr(owner, "_detector_identity_lease", None) is None:
            return False
        if (self._near_yaw_park_blocks_write("REVERSE_DETECTOR_IDENTITY_GUARD")
                or any(getattr(owner, name, False) for name in (
                    "_explicit_stop_requested", "_runtime_shutdown_requested",
                    "_brake_hold_active", "stop_action_execution", "person_detected_flag"))
                or any(getattr(self.backend, name, False) for name in (
                    "normal_zero_hold", "parking_current_a", "_parking_current_uncertain"))):
            return True
        self.backend.send_targets(0, 0, "REVERSE_DETECTOR_IDENTITY_GUARD")
        return True

    def _service_detector_reverse_guard(self) -> bool:
        """Retire an already running reverse without waiting for keepalive.

        The action loop invokes this even while vision is blocked. Recheck
        after both locks: a completed full result or newer command may have
        replaced the reason for withdrawal while this service was waiting.
        """
        owner = self.owner
        if (getattr(owner, "_detector_identity_lease", None) is None
                or getattr(owner, "current_command", None) != self.symbols.backward):
            return False
        with owner.command_lock:
            if getattr(owner, "current_command", None) != self.symbols.backward:
                return False
            with owner.motor_io_lock:
                if not self._detector_reverse_write_blocked():
                    return False
                owner.current_command = None
                owner.command_start_time = None
        self.logger.info("detector_reverse_withdrawn reason=full_identity_required "
                         "old_reverse_retired=True identity_deadline_renewed=False")
        return True

    def send_percent_backward(
        self, percent: int, *, yaw_revision: Optional[int] = None
    ) -> Optional[str]:
        """Send reverse with an optional signed visual-PID wheel differential."""
        if getattr(self.owner, "_detector_identity_lease", None) is not None:
            with self.owner.motor_io_lock:
                if self._detector_reverse_write_blocked():
                    return None
        c = self.config
        p = self.backend.clip_percent(percent)
        correction_rpm = int(getattr(self.owner, "_current_steer_correction_rpm", 0))
        if p <= 0 and correction_rpm != 0:
            return (
                "YAW_ONLY"
                if self.send_yaw_only(correction_rpm, yaw_revision=yaw_revision,
                                      reverse_guard=True)
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
                if self._detector_reverse_write_blocked():
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
            if self._detector_reverse_write_blocked():
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
            if (label == "STEER" and (p1 > 0 or p2 > 0)
                    and getattr(self.owner, "_detector_identity_lease", None) is not None):
                # This percent writer cannot apply the detector deadline.
                return False
            self.backend.send_diff(p1, m1_state, p2, m2_state, label)
        return True

    def send_percent_brake(self, mode: Optional[str] = None, label: str = "brake") -> bool:
        # Cover both producers when an adapter uses a separate backend logger.
        # The scope must outlive this method's I/O lock, including failed ACKs.
        backend_logger = getattr(self.backend, "logger", self.logger)
        with deferred_diagnostics(self.logger), (
            deferred_diagnostics(backend_logger) if backend_logger is not self.logger else nullcontext()
        ):
            return self._send_percent_brake_deferred(mode=mode, label=label)

    def _send_percent_brake_deferred(self, mode: Optional[str] = None, label: str = "brake") -> bool:
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
                        sent = self._send_follow_wheel_targets(
                            left_target,
                            right_target,
                            "STEER",
                            max_target_override=max_target_override,
                            visible_required=continuous_wheels,
                        )
                        if sent is False:
                            return
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
