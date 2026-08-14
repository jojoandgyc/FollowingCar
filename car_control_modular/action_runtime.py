from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, FrozenSet, Mapping, Optional

from .mssd_motor import MssdMotorBackend


@dataclass(frozen=True)
class ActionRuntimeSymbols:
    forward: int
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
    visible_steer_inner_ratio_percent: int
    visible_steer_outer_ratio_percent: int
    motor_forward_raw_target: int
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
        self.backend = backend
        self.config = config
        self.symbols = symbols
        self.hard_stop_check = hard_stop_check
        self.logger = logger or logging.getLogger(__name__)
        if not hasattr(owner, "action_queue_lock"):
            owner.action_queue_lock = threading.Lock()

    def start(self) -> None:
        owner = self.owner
        owner.action_stop_event.clear()
        owner.action_thread = threading.Thread(target=self.run_loop, daemon=True)
        owner.action_thread.start()
        self.logger.info("动作执行线程已启动")

    def read_motor_feedback_rpm(self):
        return None

    def can_release_brake_hold(self, action: int) -> bool:
        owner = self.owner
        s = self.symbols
        if action in (s.rotate_left, s.rotate_right):
            return True
        if action in (s.steer_left, s.steer_right):
            return int(getattr(owner, "_current_steer_base_percent", 0)) > 0
        if action == s.forward:
            return int(getattr(owner, "_current_forward_percent", 0)) > 0
        return False

    def has_pending_actions(self) -> bool:
        owner = self.owner
        try:
            with owner.action_queue_lock:
                return not owner.action_queue.empty()
        except Exception:
            return False

    def coast_down_before_rotate(self) -> None:
        owner = self.owner
        c = self.config
        if not c.use_percent_speed or not c.rotate_prep_coast_enable:
            return
        p0 = max(0, min(100, int(getattr(owner, "_current_forward_percent", 0))))
        if p0 <= 0:
            return
        steps = max(1, int(c.rotate_prep_coast_steps))
        total = max(0.05, float(c.rotate_prep_coast_total_sec))
        for i in range(steps):
            pct = max(0, int(p0 * (1.0 - (i + 1) / float(steps))))
            if pct > 0 and pct < c.min_forward_percent:
                pct = c.min_forward_percent
            try:
                self.send_percent_drive(pct)
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
                action_from_queue = False
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
                            if rpm_m1_s < 0 and rpm_m2_s < 0:
                                if not owner._brake_hold_active:
                                    self.logger.warning(
                                        "检测到 M1/M2 同时倒转，进入 brake 保持态: signed=%d,%d raw=%d,%d",
                                        rpm_m1_s,
                                        rpm_m2_s,
                                        rpm_m1_raw,
                                        rpm_m2_raw,
                                    )
                                owner._brake_hold_active = True
                                owner._use_soft_stop_next = False
                                owner._last_brake_hold_send_ts = 0.0

                try:
                    dequeue_ts = time.monotonic()
                    with owner.action_queue_lock:
                        action = owner.action_queue.get_nowait()
                        queue_after_pop = list(owner.action_queue.queue)
                    action_from_queue = True
                    queue_age_ms = (
                        (dequeue_ts - float(getattr(owner, "_last_action_queue_replace_ts", 0.0))) * 1000.0
                        if float(getattr(owner, "_last_action_queue_replace_ts", 0.0)) > 0
                        else -1.0
                    )
                    action_kind_age_ms = (
                        (dequeue_ts - float(getattr(owner, "_last_tracker_action_change_ts", 0.0))) * 1000.0
                        if float(getattr(owner, "_last_tracker_action_change_ts", 0.0)) > 0
                        else -1.0
                    )
                    self.logger.info(
                        "action queue pop: seq=%d frame=%d reason=%s enqueue_frame=%d action=%s remaining=%d queue_after=%s queue_wait_ms=%.1f action_kind_age_ms=%.1f",
                        int(getattr(owner, "_last_action_queue_seq", 0)),
                        int(getattr(owner, "frame_index", -1)),
                        getattr(owner, "_last_action_queue_reason", ""),
                        int(getattr(owner, "_last_action_queue_replace_frame", -1)),
                        s.action_names.get(action, str(action)),
                        len(queue_after_pop),
                        [s.action_names.get(a, str(a)) for a in queue_after_pop],
                        queue_age_ms,
                        action_kind_age_ms,
                    )
                    with owner.command_lock:
                        if action in (s.forward, s.rotate_left, s.rotate_right, s.steer_left, s.steer_right):
                            owner.stop_action_execution = False
                            owner.person_detected_flag = False
                        if (
                            c.rotate_pulse_brake_enable
                            and action in (s.rotate_left, s.rotate_right)
                            and time.time() < float(getattr(owner, "_rotate_pause_until_ts", 0.0))
                        ):
                            now_ts = time.time()
                            if now_ts - float(getattr(owner, "_last_rotate_pause_log_ts", 0.0)) >= 0.20:
                                owner._last_rotate_pause_log_ts = now_ts
                                self.logger.info(
                                    "rotate pulse pause: action=%s remaining=%.3fs pause=%.3fs duration=%.3fs raw=%d raw_source=%s percent=%d",
                                    s.action_names.get(action, str(action)),
                                    max(0.0, float(getattr(owner, "_rotate_pause_until_ts", 0.0)) - now_ts),
                                    c.rotate_pulse_pause_sec,
                                    c.rotate_duration,
                                    int(getattr(owner, "_current_rotate_raw_target", c.motor_rotate_raw_target)),
                                    str(getattr(owner, "_current_rotate_raw_source", "default")),
                                    int(getattr(owner, "_current_rotate_turn_percent", c.rotate_turn_percent_from_forward)),
                                )
                            if c.rotate_pulse_stop_mode == "brake":
                                try:
                                    self.send_percent_brake()
                                except Exception as exc:
                                    self.logger.warning("停顿期 brake 失败: %s", exc)
                            try:
                                with owner.action_queue_lock:
                                    owner.action_queue.put_nowait(action)
                            except queue.Full:
                                self.logger.warning("停顿期旋转命令回队失败(queue满)")
                            time.sleep(0.02)
                            continue
                        if owner._brake_hold_active:
                            if self.can_release_brake_hold(action):
                                owner._brake_hold_active = False
                                owner._brake_hold_stop_mode = None
                                owner._brake_hold_label = "brake"
                                self.logger.info("收到可释放命令，解除 brake 保持态: %s", action)
                            else:
                                self.logger.info("brake 保持态生效，忽略命令: %s（仅旋转或前进>0可解除）", action)
                                try:
                                    self.send_percent_brake()
                                except Exception as exc:
                                    self.logger.warning("brake 保持态下再次锁轮失败: %s", exc)
                                continue
                        if action == owner.current_command:
                            if action in (s.forward, s.steer_left, s.steer_right) or (
                                action in (s.rotate_left, s.rotate_right) and not c.rotate_pulse_brake_enable
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
                                            c.rotate_pulse_brake_enable,
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
                                [s.action_names.get(a, str(a)) for a in queue_after_pop],
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
                            if action in (s.forward, s.steer_left, s.steer_right):
                                owner._rotate_follows_previous_rotate = False
                            owner.current_command = action
                            if action in (s.rotate_left, s.rotate_right) and c.rotate_pulse_brake_enable:
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
                                "after_motor_dispatch" if action in (s.rotate_left, s.rotate_right) and c.rotate_pulse_brake_enable else "command_start",
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
                                    c.rotate_pulse_brake_enable,
                                    c.rotate_duration,
                                    c.rotate_pulse_pause_sec,
                                    c.rotate_hold_stale_sec,
                                    "after_motor_dispatch" if c.rotate_pulse_brake_enable else "command_start",
                                )
                            else:
                                self.logger.info("收到新指令: %s, 开始执行", action)
                except queue.Empty:
                    with owner.command_lock:
                        if owner.current_command is not None:
                            if owner.stop_action_execution or owner.person_detected_flag:
                                self.logger.info("收到停止信号，立即停止当前命令")
                                was_rotate = owner.current_command in (s.rotate_left, s.rotate_right)
                                if was_rotate:
                                    owner._rotate_follows_previous_rotate = True
                                owner.current_command = None
                                owner.command_start_time = None
                                if was_rotate:
                                    self.brake_lock_after_rotate_pulse()
                                else:
                                    self.send_stop_with_brake_hold("stop_signal")
                                owner.stop_action_execution = False
                                owner.person_detected_flag = False
                            else:
                                if owner.command_start_time is None:
                                    time.sleep(0.01)
                                    continue
                                elapsed = time.time() - owner.command_start_time
                                if (
                                    owner.current_command in (s.rotate_left, s.rotate_right)
                                    and c.rotate_pulse_brake_enable
                                    and elapsed >= c.rotate_duration
                                ):
                                    ended_action = owner.current_command
                                    pause_until = time.time() + float(c.rotate_pulse_pause_sec)
                                    self.logger.info(
                                        "rotate pulse duration reached: action=%s elapsed=%.3fs duration=%.3fs -> brake pause=%.3fs pause_until=%.3f",
                                        s.action_names.get(ended_action, str(ended_action)),
                                        elapsed,
                                        c.rotate_duration,
                                        c.rotate_pulse_pause_sec,
                                        pause_until,
                                    )
                                    owner._rotate_follows_previous_rotate = True
                                    owner._last_rotate_end_ts = time.time()
                                    owner.current_command = None
                                    owner.command_start_time = None
                                    owner._rotate_pause_until_ts = pause_until
                                    self.brake_lock_after_rotate_pulse()
                                elif (
                                    owner.current_command in (s.rotate_left, s.rotate_right)
                                    and not c.rotate_pulse_brake_enable
                                    and elapsed >= c.rotate_hold_stale_sec
                                ):
                                    ended_action = owner.current_command
                                    self.logger.info(
                                        "rotate hold stale stop: action=%s elapsed=%.3fs hold_stale=%.3fs pulse_enable=%s",
                                        s.action_names.get(ended_action, str(ended_action)),
                                        elapsed,
                                        c.rotate_hold_stale_sec,
                                        c.rotate_pulse_brake_enable,
                                    )
                                    owner._rotate_follows_previous_rotate = True
                                    owner._last_rotate_end_ts = time.time()
                                    owner.current_command = None
                                    owner.command_start_time = None
                                    self.send_stop_with_brake_hold("rotate_stale")

                with owner.command_lock:
                    if owner.current_command is not None and not owner.stop_action_execution and not owner.person_detected_flag:
                        if (
                            owner.current_command in (s.rotate_left, s.rotate_right)
                            and owner.command_start_time is not None
                            and c.rotate_pulse_brake_enable
                            and (time.time() - owner.command_start_time) >= c.rotate_duration
                        ):
                            elapsed = time.time() - owner.command_start_time
                            ended_action = owner.current_command
                            pause_until = time.time() + float(c.rotate_pulse_pause_sec)
                            self.logger.warning(
                                "rotate pulse duration brake: action=%s elapsed=%.3fs duration=%.3fs pause=%.3fs pause_until=%.3f",
                                s.action_names.get(ended_action, str(ended_action)),
                                elapsed,
                                c.rotate_duration,
                                c.rotate_pulse_pause_sec,
                                pause_until,
                            )
                            owner._rotate_follows_previous_rotate = True
                            owner._last_rotate_end_ts = time.time()
                            owner.current_command = None
                            owner.command_start_time = None
                            owner._rotate_pause_until_ts = pause_until
                            self.brake_lock_after_rotate_pulse()
                            time.sleep(0.01)
                            continue
                        if owner._brake_hold_active:
                            time.sleep(0.01)
                            continue
                        if owner.current_command in (
                            s.forward,
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
                            and c.rotate_pulse_brake_enable
                            and owner.command_start_time is not None
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
                                self.send_robot_command(owner.current_command)
                            time.sleep(0.01)
                            continue
                        if owner.current_command in (s.forward, s.steer_left, s.steer_right) and self.has_pending_actions():
                            time.sleep(0.01)
                            continue
                        if self.forward_like_refresh_due(owner.current_command, force=action_from_queue):
                            self.send_robot_command(owner.current_command)

                time.sleep(0.01)
            except Exception as exc:
                self.logger.error("动作执行异常: %s", exc)
                with owner.command_lock:
                    owner.current_command = None
                    owner.command_start_time = None
                self.send_stop_with_brake_hold("action_executor_exception")

    def note_rotate_pulse_target_sent(self, action: int) -> bool:
        owner = self.owner
        c = self.config
        s = self.symbols
        if not (c.rotate_pulse_brake_enable and owner.current_command == action):
            return False
        if owner.command_start_time is None:
            owner.command_start_time = time.time()
            owner._last_rotate_pulse_refresh_ts = owner.command_start_time
            self.logger.info(
                "rotate pulse timer start: action=%s after_motor_dispatch_ts=%.3f duration=%.3fs pause=%.3fs",
                s.action_names.get(action, str(action)),
                owner.command_start_time,
                c.rotate_duration,
                c.rotate_pulse_pause_sec,
            )
            return True

        owner._last_rotate_pulse_refresh_ts = time.time()
        self.logger.info(
            "rotate pulse target refreshed: action=%s elapsed=%.3fs duration=%.3fs",
            s.action_names.get(action, str(action)),
            owner._last_rotate_pulse_refresh_ts - owner.command_start_time,
            c.rotate_duration,
        )
        return False

    def forward_like_refresh_due(self, action: int, *, force: bool = False) -> bool:
        owner = self.owner
        s = self.symbols
        if action not in (s.forward, s.steer_left, s.steer_right):
            return True
        now_ts = time.time()
        if force:
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
                "forward-like keepalive refresh: action=%s interval=%.3fs elapsed=%.3fs",
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
        enqueue_ts = float(getattr(owner, "_last_action_queue_replace_ts", 0.0))
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
        self.logger.info(
            "motor dispatch timing: seq=%d action=%s label=%s reason=%s enqueue_frame=%d frame=%d queue_to_dispatch_ms=%.1f send_ms=%.1f since_last_dispatch_ms=%.1f same_action_refresh_ms=%.1f",
            int(getattr(owner, "_last_action_queue_seq", 0)),
            s.action_names.get(action, str(action)),
            label,
            getattr(owner, "_last_action_queue_reason", ""),
            int(getattr(owner, "_last_action_queue_replace_frame", -1)),
            int(getattr(owner, "frame_index", -1)),
            queue_to_dispatch_ms,
            (now_ts - send_start_ts) * 1000.0,
            since_last_dispatch_ms,
            same_action_refresh_ms,
        )

    def needs_transition_stop(self, old_action: Optional[int], new_action: int) -> bool:
        s = self.symbols
        if old_action == new_action:
            return False
        if old_action in s.forward_like_actions and new_action in s.forward_like_actions:
            return False
        return old_action in s.movement_actions and new_action in s.movement_actions and (
            old_action in s.rotate_actions or new_action in s.rotate_actions
        )

    def send_motion_transition_stop(self, old_action: Optional[int], new_action: int) -> None:
        s = self.symbols
        old_name = s.action_names.get(old_action, str(old_action))
        new_name = s.action_names.get(new_action, str(new_action))
        self.logger.info("运动模式切换急停: %s -> %s", old_name, new_name)
        self.send_transition_stop_sequence(f"transition_{old_name}_to_{new_name}")

    def send_transition_stop_sequence(self, label: str) -> None:
        c = self.config
        try:
            repeat = max(1, int(c.motor_rs485_transition_stop_repeat))
            transition_mode = c.motor_rs485_transition_stop_mode.strip().lower()
            for idx in range(repeat):
                stop_label = label if repeat <= 1 else f"{label}#{idx + 1}"
                if transition_mode in {"zero", "target_zero", "zero_target", "coast"}:
                    with self.owner.motor_io_lock:
                        self.backend.send_targets(0, 0, f"{stop_label}_zero")
                    self.backend.motion_armed = False
                    self.logger.info("MSSD transition zero %s", stop_label)
                else:
                    with self.owner.motor_io_lock:
                        self.backend.send_stop(stop_label, mode=c.motor_rs485_transition_stop_mode)
                if c.motor_rs485_transition_stop_delay_sec > 0:
                    time.sleep(c.motor_rs485_transition_stop_delay_sec)
        except Exception as exc:
            self.logger.warning("运动模式切换停稳失败，继续尝试新动作: %s", exc)

    def send_percent_drive(self, percent: int) -> None:
        c = self.config
        p = self.backend.clip_percent(percent)
        if p <= 0:
            state = 0x00
        else:
            if c.motor_forward_raw_target > 0:
                raw = int(c.motor_forward_target_sign * c.motor_forward_raw_target)
                left_target = int(raw * c.motor_left_sign)
                right_target = int(raw * c.motor_right_sign)
                with self.owner.motor_io_lock:
                    self.backend.send_targets(left_target, right_target, "DRIVE")
                return
            p = max(min(c.min_forward_percent, self.backend.config.percent_limit), p)
            state = 0x01
        with self.owner.motor_io_lock:
            self.backend.send_diff(p, state, p, state, "DRIVE")

    def send_percent_diff(self, m1_percent: int, m1_state: int, m2_percent: int, m2_state: int, label: str) -> None:
        p1 = self.backend.clip_percent(m1_percent)
        p2 = self.backend.clip_percent(m2_percent)
        with self.owner.motor_io_lock:
            self.backend.send_diff(p1, m1_state, p2, m2_state, label)

    def send_percent_brake(self, mode: Optional[str] = None, label: str = "brake") -> None:
        with self.owner.motor_io_lock:
            self.backend.send_stop(label, mode=mode)

    def send_rotate_pulse_zero_stop(self) -> None:
        with self.owner.motor_io_lock:
            self.backend.send_targets(0, 0, "TURN_ZERO")
        self.backend.motion_armed = False

    def brake_lock_after_rotate_pulse(self) -> None:
        owner = self.owner
        c = self.config
        self.logger.info(
            "rotate pulse stop: pulse_enable=%s stop_mode=%s duration=%.3fs pause=%.3fs brake_refresh=%.3fs",
            c.rotate_pulse_brake_enable,
            c.rotate_pulse_stop_mode,
            c.rotate_duration,
            c.rotate_pulse_pause_sec,
            owner._brake_hold_refresh_interval_sec,
        )
        owner._use_soft_stop_next = False
        if c.rotate_pulse_stop_mode == "zero":
            try:
                self.send_rotate_pulse_zero_stop()
            except Exception as exc:
                self.logger.warning("旋转脉冲结束清零目标失败，退回 brake: %s", exc)
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
                self.logger.warning("旋转脉冲结束 brake 失败，退回 ACTION_STOP: %s", exc)
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

    def send_robot_command(self, action: int) -> None:
        owner = self.owner
        c = self.config
        s = self.symbols
        if action == s.forward:
            percent = getattr(owner, "_current_forward_percent", 0)
            send_start = time.monotonic()
            try:
                self.send_percent_drive(percent)
            except Exception as exc:
                self.logger.warning("百分比调速发送失败: %s", exc)
            else:
                self.log_motor_dispatch_timing(action, "DRIVE", send_start)
            return

        if action in (s.steer_left, s.steer_right):
            steer_cap = max(c.min_forward_percent, min(100, int(c.steer_percent_limit)))
            base_cap = min(c.max_forward_percent, steer_cap)
            base = max(0, min(base_cap, int(getattr(owner, "_current_steer_base_percent", 0))))
            inner_ratio = max(0, min(100, int(getattr(owner, "_current_steer_inner_ratio_percent", c.visible_steer_inner_ratio_percent))))
            outer_ratio = max(0, min(150, int(getattr(owner, "_current_steer_outer_ratio_percent", c.visible_steer_outer_ratio_percent))))
            if base <= 0:
                send_start = time.monotonic()
                try:
                    self.send_percent_drive(0)
                except Exception as exc:
                    self.logger.warning("轮差前进零速停止失败: %s", exc)
                else:
                    self.log_motor_dispatch_timing(action, "STEER_ZERO", send_start)
                return
            inner = int(math.floor(base * inner_ratio / 100.0))
            outer = int(math.ceil(base * outer_ratio / 100.0))
            inner = max(c.min_forward_percent, min(steer_cap, inner))
            outer = max(c.min_forward_percent, min(steer_cap, outer))
            fwd = 0x01
            if action == s.steer_left:
                left_percent, right_percent = inner, outer
            else:
                left_percent, right_percent = outer, inner

            if c.motor_steer_raw_target > 0:
                base_raw = max(0, int(c.motor_steer_raw_target))
                inner_raw = int(math.floor(base_raw * inner_ratio / 100.0))
                outer_raw = int(math.ceil(base_raw * outer_ratio / 100.0))
                inner_raw = max(base_raw, inner_raw)
                outer_raw = max(base_raw, outer_raw)
                if action == s.steer_left:
                    left_raw, right_raw = inner_raw, outer_raw
                else:
                    left_raw, right_raw = outer_raw, inner_raw
                left_target = self.backend.wheel_raw_state_to_target("left", left_raw, fwd)
                right_target = self.backend.wheel_raw_state_to_target("right", right_raw, fwd)
                try:
                    self.logger.info(
                        "steer motor dispatch: action=%s mode=raw base=%d inner_raw=%d outer_raw=%d left_target=%d right_target=%d inner_ratio=%d outer_ratio=%d",
                        s.action_names.get(action, str(action)),
                        base_raw,
                        inner_raw,
                        outer_raw,
                        left_target,
                        right_target,
                        inner_ratio,
                        outer_ratio,
                    )
                    send_start = time.monotonic()
                    with owner.motor_io_lock:
                        self.backend.send_targets(left_target, right_target, "STEER")
                except Exception as exc:
                    self.logger.warning("轮差 raw 前进发送失败: %s", exc)
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
                    "steer motor dispatch: action=%s mode=percent base=%d cap=%d inner=%d outer=%d left_percent=%d right_percent=%d inner_ratio=%d outer_ratio=%d",
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
                self.send_percent_diff(m1_percent, m1_state, m2_percent, m2_state, label="STEER")
            except Exception as exc:
                self.logger.warning("轮差前进发送失败: %s", exc)
            else:
                self.log_motor_dispatch_timing(action, "STEER", send_start)
            return

        if action in (s.rotate_left, s.rotate_right):
            p = max(0, min(100, int(getattr(owner, "_current_rotate_turn_percent", c.rotate_turn_percent_from_forward))))
            raw_target = max(0, int(getattr(owner, "_current_rotate_raw_target", c.motor_rotate_raw_target)))
            raw_source = str(getattr(owner, "_current_rotate_raw_source", "default"))
            fwd = 0x01
            back = 0x02
            if action == s.rotate_left:
                left_state, right_state = back, fwd
            else:
                left_state, right_state = fwd, back

            if raw_target > 0:
                left_target = self.backend.wheel_raw_state_to_target("left", raw_target, left_state)
                right_target = self.backend.wheel_raw_state_to_target("right", raw_target, right_state)
                now_ts = time.time()
                if now_ts - float(getattr(owner, "_last_rotate_dispatch_log_ts", 0.0)) >= 0.25:
                    owner._last_rotate_dispatch_log_ts = now_ts
                    self.logger.info(
                        "rotate motor dispatch: action=%s mode=raw raw=%d raw_source=%s default_raw=%d left_target=%d right_target=%d pulse_enable=%s duration=%.3fs pause=%.3fs hold_stale=%.3fs",
                        s.action_names.get(action, str(action)),
                        raw_target,
                        raw_source,
                        c.motor_rotate_raw_target,
                        left_target,
                        right_target,
                        c.rotate_pulse_brake_enable,
                        c.rotate_duration,
                        c.rotate_pulse_pause_sec,
                        c.rotate_hold_stale_sec,
                    )
                try:
                    send_start = time.monotonic()
                    with owner.motor_io_lock:
                        self.backend.send_targets(left_target, right_target, "TURN")
                except Exception as exc:
                    self.logger.warning("差速旋转 raw 发送失败: %s", exc)
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
                    "rotate motor dispatch: action=%s mode=percent percent=%d pulse_enable=%s duration=%.3fs pause=%.3fs hold_stale=%.3fs",
                    s.action_names.get(action, str(action)),
                    p,
                    c.rotate_pulse_brake_enable,
                    c.rotate_duration,
                    c.rotate_pulse_pause_sec,
                    c.rotate_hold_stale_sec,
                )
            send_start = time.monotonic()
            try:
                self.send_percent_diff(m1_percent, m1_state, m2_percent, m2_state, label="TURN")
            except Exception as exc:
                self.logger.warning("差速旋转发送失败: %s", exc)
            else:
                self.note_rotate_pulse_target_sent(action)
                self.log_motor_dispatch_timing(action, "TURN", send_start)
                return
            return

        if action == s.stop:
            if getattr(owner, "_use_soft_stop_next", False):
                owner._use_soft_stop_next = False
                send_start = time.monotonic()
                try:
                    self.send_percent_drive(0)
                except Exception as exc:
                    self.logger.warning("转向结束软停失败: %s", exc)
                else:
                    self.log_motor_dispatch_timing(action, "STOP_SOFT", send_start)
                return
            send_start = time.monotonic()
            try:
                self.send_percent_brake()
            except Exception as exc:
                self.logger.warning("百分比刹车发送失败: %s", exc)
            else:
                self.log_motor_dispatch_timing(action, "STOP", send_start)
            return

        self.logger.warning("未知动作类型（仅支持百分比通道）: %s", action)

    def send_stop_with_brake_hold(self, reason: str = "") -> None:
        owner = self.owner
        c = self.config
        hold_brake = reason not in ("target_distance_reached", "search_to_follow")
        safety_stop = reason in self.symbols.safety_stop_reasons
        preserve_motion_params = reason == "search_to_follow"
        owner._use_soft_stop_next = False
        owner._brake_hold_active = hold_brake
        owner._brake_hold_stop_mode = c.safety_stop_mode if safety_stop and hold_brake else None
        owner._brake_hold_label = f"safety_hold_{reason}" if safety_stop and hold_brake else "brake"
        owner._last_brake_hold_send_ts = 0.0
        if not preserve_motion_params:
            owner.is_forwarding = False
            owner._current_forward_percent = 0
            owner._current_steer_base_percent = 0
        if reason:
            self.logger.info(
                "进入 brake 保持态: %s hold=%s stop_mode=%s safety=%s preserve_motion_params=%s",
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
            return
        self.send_robot_command(self.symbols.stop)
