#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.control_types import DistanceState, ObstacleState, SteeringFeedback


class FakeDriver:
    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.stops = 0
        self.emergency_stops = 0
        self.free_stops = 0
        self.registers = {
            "right_parking_current": 0.0,
            "left_parking_current": 0.0,
        }
        self.register_writes = []
        self.calls = []
        self.closed = False

    def set_left_speed(self, value: int) -> None:
        self.left = int(value)
        self.calls.append(("left_speed", int(value)))

    def set_right_speed(self, value: int) -> None:
        self.right = int(value)
        self.calls.append(("right_speed", int(value)))

    def stop_all(self, mode: int = 0) -> None:
        self.left = 0
        self.right = 0
        mode_value = int(mode)
        self.calls.append(("stop", mode_value))
        if mode_value == 1:
            self.emergency_stops += 1
        elif mode_value == 2:
            self.free_stops += 1
        else:
            self.stops += 1

    def write_register(self, name: str, value: float, persist: bool = False) -> None:
        self.register_writes.append((name, float(value), bool(persist)))
        self.registers[name] = float(value)

    def read_register(self, name: str) -> float:
        return float(self.registers[name])

    def read_motor_status(self, side: str):
        speed = self.left if str(side) == "left" else self.right
        return SimpleNamespace(
            position_degree=0,
            speed_rpm=0 if speed is None else int(speed),
            error_code=0,
        )

    def close(self) -> None:
        self.closed = True


def _load_request_module(config: str):
    old_argv = list(sys.argv)
    sys.argv = ["test_mssd_mapping.py", "--config", config]
    try:
        try:
            import bunker_hazard_detector  # noqa: F401
        except ModuleNotFoundError:
            # The board-only hazard extension is not needed by this dry-run.
            # Keep the test runnable in the source checkout without masking a
            # real import error when the extension is installed.
            stub = types.ModuleType("bunker_hazard_detector")
            stub.BunkerHazardMonitor = object
            stub.RKNNBunkerHazardDetector = object
            stub.HazardState = types.SimpleNamespace
            stub.check_hazard_from_dets = lambda *_args, **_kwargs: None
            sys.modules["bunker_hazard_detector"] = stub
        try:
            import track_first_person2  # noqa: F401
        except ModuleNotFoundError:
            tracker_stub = types.ModuleType("track_first_person2")
            tracker_stub.FirstPersonTracker = object
            sys.modules["track_first_person2"] = tracker_stub
        import request_0513_modular as mod
    finally:
        sys.argv = old_argv
    return mod


def _make_tracker_shell(mod):
    tracker = object.__new__(mod.PersonTracker)
    backend = mod.MssdMotorBackend(
        replace(mod.MSSD_MOTOR_CONFIG, stop_zero_delay_sec=0.0),
        logger=mod.logger,
    )
    backend.driver = FakeDriver()
    backend.motion_armed = True
    tracker._motor_backend = backend
    tracker.motor_io_lock = backend.io_lock
    tracker.current_command = None
    tracker.command_start_time = None
    tracker._last_rotate_pulse_refresh_ts = 0.0
    tracker._current_rotate_pulse_enabled = True
    tracker._search_epoch = 0
    tracker._rotate_settle_search_epoch = -1
    tracker._forward_speed_latched_percent = None
    tracker._last_forward_speed_hysteresis_log_ts = 0.0
    runtime_config = replace(
        mod.ACTION_RUNTIME_CONFIG,
        motor_rs485_transition_stop_delay_sec=0.0,
        motor_rs485_transition_stop_repeat=2,
    )
    tracker._action_runtime = mod.MotionActionRuntime(
        tracker,
        backend,
        runtime_config,
        mod.ACTION_RUNTIME_SYMBOLS,
        hard_stop_check=lambda _action=None: False,
        logger=mod.logger,
    )
    return tracker


def _assert_depth_recovery_ramp_and_timeout_clamp(mod) -> None:
    controller = mod.FollowSafetyController(
        mod.FollowPolicyConfig(
            max_forward_percent=100,
            min_forward_percent=0,
            forward_max_rpm=100,
            depth_recovery_stage1_sec=0.20,
            depth_recovery_stage2_sec=0.40,
            depth_recovery_stage1_rpm=25,
            depth_recovery_stage2_rpm=45,
        )
    )
    fresh_state = mod.DistanceState(
        source="vision_depth",
        raw_distance_m=7.65,
        filtered_distance_m=7.05,
        used_distance_m=7.05,
        source_detail="depth_multiregion_after_jump_confirm",
    )
    hold_state = mod.DistanceState(
        source="vision_depth",
        raw_distance_m=None,
        filtered_distance_m=7.05,
        used_distance_m=7.05,
        source_detail="depth_multiregion_reused_hold_fused_visual_encoder_hold",
    )
    fresh_frame = mod.SensorFrame(distance_m=7.05, distance_state=fresh_state)
    hold_frame = mod.SensorFrame(distance_m=7.05, distance_state=hold_state)
    started_at = 100.0
    controller._remember_target_distance(fresh_frame, started_at)
    samples = [controller._limit_depth_quality_forward_percent(fresh_frame, 100, started_at)]
    samples.append(
        controller._limit_depth_quality_forward_percent(hold_frame, 100, started_at + 0.05)
    )
    controller._remember_target_distance(fresh_frame, started_at + 0.25)
    samples.append(
        controller._limit_depth_quality_forward_percent(fresh_frame, 100, started_at + 0.25)
    )
    samples.append(
        controller._limit_depth_quality_forward_percent(hold_frame, 100, started_at + 0.30)
    )
    controller._remember_target_distance(fresh_frame, started_at + 0.45)
    samples.append(
        controller._limit_depth_quality_forward_percent(fresh_frame, 100, started_at + 0.45)
    )
    if samples != [25, 25, 45, 45, 100]:
        raise AssertionError(f"Depth recovery must survive fused hold frames: {samples}")

    now = time.monotonic()
    controller._lost_started_at = now - 0.10
    controller._search_rotation_started_at = now - 0.05
    controller.defer_search_timeout(20.0)
    after = time.monotonic()
    if controller._lost_started_at > after or controller._search_rotation_started_at > after:
        raise AssertionError(
            "search timeout deferral must not push lost/search timestamps into the future"
        )
    print("depth_recovery_ramp_and_timeout_clamp: PASS")


class FakeBunkerRuntime:
    def __init__(self, state=None) -> None:
        self.state = state

    def current_split_state(self):
        return self.state


class FakeSensorRuntime:
    def __init__(self, *, front: bool = False, left: bool = False, right: bool = False) -> None:
        self.obstacles = ObstacleState(front=front, left=left, right=right)

    def get_obstacle_status(self) -> ObstacleState:
        return self.obstacles


class FakeDistanceRuntime:
    def __init__(self, distance_m=None) -> None:
        self.state = DistanceState(source="fake", used_distance_m=distance_m)

    def get_recent_vision_mmwave_state(self, **_kwargs) -> DistanceState:
        return self.state

    def get_sensor_distance_state(self, **_kwargs) -> DistanceState:
        return self.state


def _set_safety_stubs(tracker, *, front: bool = False, left: bool = False, right: bool = False, distance_m=None) -> None:
    tracker._bunker_runtime = FakeBunkerRuntime()
    tracker._sensor_runtime = FakeSensorRuntime(front=front, left=left, right=right)
    tracker._distance_runtime = FakeDistanceRuntime(distance_m)
    tracker.frame_index = 0
    tracker._last_distance_stop_log_key = None
    tracker._last_distance_hard_stop_log_ts = 0.0


def _make_geometry_fallback_tracker(mod, active_target_id=None):
    tracker = object.__new__(mod.PersonTracker)
    tracker.frame_index = 0
    tracker._single_person_geometry_bbox = None
    tracker._single_person_geometry_frame = -1
    tracker._single_person_geometry_track_id = None
    tracker._single_person_geometry_streak = 0
    tracker._follow_controller = SimpleNamespace(active_target_id=active_target_id)
    tracker._identity_assignment_debug_for_track = lambda _track_id: {}
    return tracker


def _geometry_record(*, x1, y1, x2, y2, track_id=7, reid_uid=0, time_since_update=0):
    return SimpleNamespace(
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        track_id=int(track_id),
        reid_uid=int(reid_uid),
        time_since_update=int(time_since_update),
    )


def _assert_single_person_geometry_fallback(mod) -> None:
    if not mod.SINGLE_PERSON_GEOMETRY_FALLBACK_ENABLE:
        print("single_person_geometry_fallback: DISABLED_BY_RUNTIME_POLICY")
        return
    locked = _make_geometry_fallback_tracker(mod, active_target_id=42)
    locked.frame_index = 100
    first = locked._single_person_geometry_fallback_id(
        _geometry_record(x1=200, y1=100, x2=360, y2=430),
        1,
    )
    if first is not None:
        raise AssertionError(f"first ReID-less frame must not bypass continuity, got {first}")

    locked.frame_index = 101
    resumed = locked._single_person_geometry_fallback_id(
        _geometry_record(x1=208, y1=101, x2=368, y2=431),
        1,
    )
    if resumed != 42:
        raise AssertionError(f"continuous single-person box should retain active target 42, got {resumed}")

    new_target = _make_geometry_fallback_tracker(mod)
    new_target.frame_index = 200
    if new_target._single_person_geometry_fallback_id(
        _geometry_record(x1=250, y1=110, x2=390, y2=430, track_id=3),
        1,
    ) is not None:
        raise AssertionError("a new ReID-less target must wait for geometric confirmation")
    new_target.frame_index = 201
    confirmed = new_target._single_person_geometry_fallback_id(
        _geometry_record(x1=256, y1=112, x2=396, y2=432, track_id=3),
        1,
    )
    if confirmed != -4:
        raise AssertionError(f"two stable single-person boxes should confirm raw track fallback -4, got {confirmed}")

    new_target.frame_index = 202
    rejected_multi = new_target._single_person_geometry_fallback_id(
        _geometry_record(x1=258, y1=112, x2=398, y2=432, track_id=3),
        2,
    )
    if rejected_multi is not None:
        raise AssertionError(f"geometry fallback must stay disabled in multi-person frames, got {rejected_multi}")

    rejected_quality = _make_geometry_fallback_tracker(mod, active_target_id=42)
    rejected_quality.frame_index = 300
    rejected_quality._single_person_geometry_bbox = (200.0, 100.0, 360.0, 430.0)
    rejected_quality._single_person_geometry_frame = 299
    rejected_quality._single_person_geometry_track_id = 7
    rejected_quality._single_person_geometry_streak = 2
    rejected_quality._identity_assignment_debug_for_track = lambda _track_id: {
        "bbox_quality_ok": False,
        "bbox_quality_reason": "area_ratio>0.75",
    }
    rejected = rejected_quality._single_person_geometry_fallback_id(
        _geometry_record(x1=0, y1=0, x2=630, y2=470),
        1,
    )
    if rejected is not None:
        raise AssertionError(f"explicitly low-quality bbox must not enter geometry fallback: {rejected}")
    if (
        rejected_quality._single_person_geometry_bbox is not None
        or rejected_quality._single_person_geometry_frame != -1
        or rejected_quality._single_person_geometry_streak != 0
    ):
        raise AssertionError("low-quality bbox must clear geometry fallback continuity state")
    print("single_person_geometry_fallback: PASS")


def _assert_unconfirmed_search_candidate_stops(mod) -> None:
    forbidden = (
        "_pause_for_unconfirmed_search_candidate",
        "_queue_search_low_quality_yaw",
    )
    present = [name for name in forbidden if hasattr(mod.PersonTracker, name)]
    if present:
        raise AssertionError(f"unconfirmed evidence still exposes motion helpers: {present}")
    print("unconfirmed_search_candidate_control_boundary: PASS")


def _assert_hard_stop_policy(mod, tracker) -> None:
    actions = (
        mod.ACTION_FORWARD,
        mod.ACTION_STEER_LEFT,
        mod.ACTION_STEER_RIGHT,
        mod.ACTION_ROTATE_LEFT,
        mod.ACTION_ROTATE_RIGHT,
    )
    for sensor_name, sensor_state in (
        ("front", {"front": True}),
        ("left", {"left": True}),
        ("right", {"right": True}),
    ):
        _set_safety_stubs(tracker, **sensor_state)
        for action in actions:
            if not tracker._should_hard_stop_now(action):
                raise AssertionError(f"{sensor_name} IR should hard-stop action={action}")

    _set_safety_stubs(tracker, distance_m=max(0.0, float(mod.FOLLOW_BRAKE_DISTANCE_M) - 0.01))
    if tracker._should_hard_stop_now(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("distance must not hard-stop rotate_left in IR-only parking mode")

    _set_safety_stubs(tracker)
    if tracker._should_hard_stop_now(mod.ACTION_FORWARD):
        raise AssertionError("clear sensors should not hard-stop forward")


def _assert_depth_zero_longitudinal_preserves_visible_yaw(mod) -> None:
    tracker = object.__new__(mod.PersonTracker)
    tracker.command_lock = threading.Lock()
    tracker.current_command = mod.ACTION_ROTATE_RIGHT
    tracker.search_state = "none"
    tracker._vision_control_state = "target_visible_depth_missing"
    tracker._last_vision_control_ts = time.monotonic()
    tracker.frame_index = 77
    decision = SimpleNamespace(
        reason="longitudinal_distance_untrusted_hold",
        actions=[mod.ControlAction.forward(0, "longitudinal_distance_untrusted_hold")],
    )
    if not tracker._depth_zero_longitudinal_preserves_visible_rotation(decision):
        raise AssertionError(
            "zero-longitudinal Depth hold must preserve a fresh visible-target rotation"
        )

    tracker.current_command = mod.ACTION_FORWARD
    if tracker._depth_zero_longitudinal_preserves_visible_rotation(decision):
        raise AssertionError("Depth hold must still zero an active forward command")

    tracker.current_command = mod.ACTION_ROTATE_RIGHT
    tracker._vision_control_state = "searching"
    if tracker._depth_zero_longitudinal_preserves_visible_rotation(decision):
        raise AssertionError("the longitudinal/visible-yaw exception must not apply to search")

    safety_stop = SimpleNamespace(
        reason="front_ir",
        actions=[mod.ControlAction.stop("front_ir")],
    )
    tracker._vision_control_state = "target_visible_depth_missing"
    if tracker._depth_zero_longitudinal_preserves_visible_rotation(safety_stop):
        raise AssertionError("IR safety stop must always be allowed to interrupt rotation")
    print("depth_zero_longitudinal_preserves_visible_yaw: PASS")


def _assert_pair(label: str, actual, expected) -> None:
    print(label, actual)
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def _expected_send_diff(backend, m1_percent: int, m1_state: int, m2_percent: int, m2_state: int) -> tuple[int, int]:
    if backend.config.m1_is_left_wheel:
        left_target = backend.wheel_state_to_target("left", m1_percent, m1_state)
        right_target = backend.wheel_state_to_target("right", m2_percent, m2_state)
    else:
        right_target = backend.wheel_state_to_target("right", m1_percent, m1_state)
        left_target = backend.wheel_state_to_target("left", m2_percent, m2_state)
    return left_target, right_target


def _assert_parking_lifecycle(mod) -> None:
    backend = mod.MssdMotorBackend(
        replace(mod.MSSD_MOTOR_CONFIG, stop_zero_delay_sec=0.0),
        logger=mod.logger,
    )
    driver = FakeDriver()
    backend.driver = driver

    backend.enable_startup_parking()
    expected_start = [
        ("right_parking_current", 0.0, False),
        ("left_parking_current", 0.0, False),
        ("right_parking_current", 1.0, False),
        ("left_parking_current", 1.0, False),
        ("right_parking_current", 5.0, True),
        ("left_parking_current", 5.0, True),
    ]
    if driver.register_writes != expected_start:
        raise AssertionError(f"startup parking writes mismatch: {driver.register_writes}")
    if not math.isclose(backend.parking_current_a, 5.0):
        raise AssertionError(f"startup parking state mismatch: {backend.parking_current_a}")
    if driver.stops != 1:
        raise AssertionError(f"startup parking must issue one normal stop, got {driver.stops}")
    if driver.emergency_stops != 1:
        raise AssertionError(
            f"startup parking must clear old motion with one emergency stop, got {driver.emergency_stops}"
        )
    if driver.left != 0 or driver.right != 0:
        raise AssertionError(f"startup parking must clear both speed targets: left={driver.left} right={driver.right}")

    backend.close()
    expected_close = expected_start + [
        ("right_parking_current", 0.0, True),
        ("left_parking_current", 0.0, True),
    ]
    if driver.register_writes != expected_close:
        raise AssertionError(f"shutdown parking writes mismatch: {driver.register_writes}")
    if driver.registers["right_parking_current"] != 0.0 or driver.registers["left_parking_current"] != 0.0:
        raise AssertionError(f"shutdown parking readback was not zero: {driver.registers}")
    if not driver.closed:
        raise AssertionError("motor serial driver must close after parking current is cleared")
    print("parking_lifecycle: PASS")


def _assert_normal_stop_keeps_parking_mode(mod) -> None:
    backend = mod.MssdMotorBackend(
        replace(mod.MSSD_MOTOR_CONFIG, stop_zero_delay_sec=0.0),
        logger=mod.logger,
    )
    driver = FakeDriver()
    backend.driver = driver
    backend.motion_armed = True

    backend.send_stop("normal_parking_test", mode="normal")
    expected_normal = [
        ("right_speed", 0),
        ("left_speed", 0),
        ("stop", 0),
    ]
    if driver.calls != expected_normal:
        raise AssertionError(
            "normal stop must end at NORMAL lock-phase command without post-stop RPM writes: "
            f"{driver.calls}"
        )

    driver.calls.clear()
    backend.motion_armed = True
    backend.send_stop("emergency_test", mode="emergency")
    expected_emergency = [
        ("right_speed", 0),
        ("left_speed", 0),
        ("stop", 1),
        ("right_speed", 0),
        ("left_speed", 0),
    ]
    if driver.calls != expected_emergency:
        raise AssertionError(f"emergency stop sequence changed unexpectedly: {driver.calls}")
    print("normal_stop_keeps_parking_mode: PASS")


class _LifecycleBackend:
    def __init__(self, events, *, fail_startup: bool = False) -> None:
        self.events = events
        self.fail_startup = bool(fail_startup)
        self.driver = None
        self.tracker = None

    def ensure_driver(self) -> None:
        lock_was_free = self.tracker.motor_io_lock.acquire(blocking=False)
        if lock_was_free:
            self.tracker.motor_io_lock.release()
            raise AssertionError("startup motor initialization must hold motor_io_lock")
        self.events.append("ensure_driver")
        if self.fail_startup:
            raise RuntimeError("simulated startup failure")
        self.driver = object()

    def close(self) -> None:
        self.events.append("backend_close")
        self.driver = None


class _LifecycleActionRuntime:
    def __init__(self, backend, events) -> None:
        self.backend = backend
        self.events = events

    def start(self) -> None:
        if self.backend.driver is None:
            raise AssertionError("action/feedback threads started before motor initialization")
        self.events.append("runtime_start")

    def send_stop_with_brake_hold(self, _reason: str) -> None:
        if self.backend.driver is None:
            raise AssertionError("STOP attempted to reopen a failed motor backend")
        self.events.append("stop_hold")

    def join_feedback(self, timeout: float = 1.0) -> None:
        self.events.append("feedback_join")

    def send_robot_command(self, _action: int) -> None:
        if self.backend.driver is None:
            raise AssertionError("final STOP attempted without an initialized driver")
        self.events.append("stop_final")


class _LifecycleCloseProbe:
    def __init__(self, events, label: str) -> None:
        self.events = events
        self.label = label

    def close(self) -> None:
        self.events.append(self.label)


def _make_lifecycle_tracker(mod, *, fail_startup: bool = False):
    events = []
    tracker = object.__new__(mod.PersonTracker)
    backend = _LifecycleBackend(events, fail_startup=fail_startup)
    backend.tracker = tracker
    runtime = _LifecycleActionRuntime(backend, events)
    tracker._vision_engine = "test"
    tracker._motor_backend = backend
    tracker._action_runtime = runtime
    tracker._action_runtime_started = False
    tracker.motor_io_lock = threading.Lock()
    tracker.running = True
    tracker.action_stop_event = threading.Event()
    tracker.action_thread = None
    tracker._last_explicit_stop_reason = ""
    tracker._rknn_camera = None
    tracker._rknn_pipeline = None
    tracker._bunker_runtime = _LifecycleCloseProbe(events, "bunker_close")
    tracker._sensor_runtime = _LifecycleCloseProbe(events, "sensor_close")
    # This fixture validates motor startup/shutdown ordering only. The real
    # PersonTracker constructor owns the independent Depth longitudinal thread.
    tracker._start_longitudinal_thread = lambda: None
    tracker._stop_longitudinal_thread = lambda: None
    tracker.process_frame = lambda: setattr(tracker, "running", False)
    return tracker, events


def _assert_motor_startup_thread_order(mod) -> None:
    tracker, events = _make_lifecycle_tracker(mod)
    tracker.run()
    if events[:2] != ["ensure_driver", "runtime_start"]:
        raise AssertionError(f"motor must initialize before runtime threads: {events}")
    if events.count("stop_hold") != 1 or events.count("stop_final") != 1:
        raise AssertionError(f"normal shutdown must keep both STOP stages: {events}")

    failed, failed_events = _make_lifecycle_tracker(mod, fail_startup=True)
    try:
        failed.run()
    except RuntimeError as exc:
        if "simulated startup failure" not in str(exc):
            raise
    else:
        raise AssertionError("simulated motor startup failure did not propagate")
    forbidden = {"runtime_start", "stop_hold", "stop_final"}
    if forbidden.intersection(failed_events):
        raise AssertionError(f"failed startup unexpectedly started threads or sent STOP: {failed_events}")
    if failed_events.count("ensure_driver") != 1:
        raise AssertionError(f"failed startup retried motor initialization: {failed_events}")
    print("motor_startup_thread_order: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run LZ30EMA wheel RPM mapping.")
    parser.add_argument("--config", default=str(ROOT / "car_control_modular/config/reid_runtime.ini"))
    args = parser.parse_args()

    mod = _load_request_module(args.config)
    _assert_single_person_geometry_fallback(mod)
    _assert_unconfirmed_search_candidate_stops(mod)
    _assert_parking_lifecycle(mod)
    _assert_normal_stop_keeps_parking_mode(mod)
    _assert_motor_startup_thread_order(mod)
    _assert_depth_recovery_ramp_and_timeout_clamp(mod)
    _assert_depth_zero_longitudinal_preserves_visible_yaw(mod)
    tracker = _make_tracker_shell(mod)
    backend = tracker._motor_backend
    _assert_hard_stop_policy(mod, tracker)

    motor_max_rpm = max(1, int(mod.MOTOR_FORWARD_MAX_TARGET_RPM))
    delta_threshold_rpm = max(1, int(mod.MOTOR_FORWARD_UPDATE_MIN_DELTA_RPM))

    def _percent_for_rpm(target_rpm: int) -> int:
        return min(
            range(101),
            key=lambda percent: abs(
                round(motor_max_rpm * percent / 100.0) - int(target_rpm)
            ),
        )

    first_request = _percent_for_rpm(round(motor_max_rpm * 0.80))
    first_rpm = round(motor_max_rpm * first_request / 100.0)
    held_request = _percent_for_rpm(first_rpm - max(0, delta_threshold_rpm - 1))
    changed_request = _percent_for_rpm(first_rpm - delta_threshold_rpm)
    first_percent = tracker._stabilize_forward_percent(first_request, "follow_distance")
    held_percent = tracker._stabilize_forward_percent(held_request, "follow_distance")
    changed_percent = tracker._stabilize_forward_percent(changed_request, "follow_distance")
    safe_hold_percent = tracker._stabilize_forward_percent(40, "lost_wait_hold_forward")
    tracker._forward_speed_latched_percent = 100
    pid_missing_hold_percent = tracker._stabilize_forward_percent(
        30,
        "visual_pid_center_camera_distance_missing",
    )
    if first_percent != first_request or held_percent != first_request:
        raise AssertionError(f"sub-5-rpm forward change should be held: {first_percent}, {held_percent}")
    if changed_percent != changed_request:
        raise AssertionError(f"a forward change of at least 5 rpm must apply: {changed_percent}")
    if safe_hold_percent != 40:
        raise AssertionError(f"dropout safety deceleration must bypass hysteresis: {safe_hold_percent}")
    if pid_missing_hold_percent != 30:
        raise AssertionError(
            f"visual PID distance dropout must immediately decelerate to 15rpm: {pid_missing_hold_percent}"
        )
    print("forward_rpm_hysteresis: PASS")

    backend.send_diff(20, 0x01, 20, 0x01, "forward")
    _assert_pair("forward", (backend.driver.left, backend.driver.right), _expected_send_diff(backend, 20, 0x01, 20, 0x01))

    backend.send_diff(20, 0x01, 20, 0x02, "left")
    _assert_pair("left", (backend.driver.left, backend.driver.right), _expected_send_diff(backend, 20, 0x01, 20, 0x02))

    backend.send_diff(20, 0x02, 20, 0x01, "right")
    _assert_pair("right", (backend.driver.left, backend.driver.right), _expected_send_diff(backend, 20, 0x02, 20, 0x01))

    runtime = tracker._action_runtime
    if runtime._median_rate((0.0, 61.83, -4.70)) != 0.0:
        raise AssertionError("encoder yaw median must reject an isolated transition spike")
    tracker._follow_controller = SimpleNamespace(target_stop_latched=True)
    tracker._last_control_decision_reason = "person_parked_recenter_right"
    if not runtime.can_release_brake_hold(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("controlled parked recenter must release brake hold for rotation")
    tracker._last_control_decision_reason = "target_visible_low_quality_yaw"
    if not runtime.can_release_brake_hold(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("bounded mapped-crop yaw must release a normal target stop")
    tracker._last_control_decision_reason = "search_candidate_approach_left"
    if not runtime.can_release_brake_hold(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("fresh candidate yaw must release the aimline brake hold")
    if runtime.can_release_brake_hold(mod.ACTION_FORWARD) or runtime.can_release_brake_hold(mod.ACTION_STEER_RIGHT):
        raise AssertionError("target-distance latch must continue blocking forward and differential steer")
    tracker._last_control_decision_reason = "target_approaching_reverse"
    tracker._current_forward_percent = 20
    if not runtime.can_release_brake_hold(mod.ACTION_BACKWARD):
        raise AssertionError("controller-approved distance recovery reverse must release brake hold")
    tracker._last_control_decision_reason = "reverse_distance_missing_wait"
    tracker._current_forward_percent = 0
    tracker._current_steer_correction_rpm = 6
    if not runtime.can_release_brake_hold(mod.ACTION_BACKWARD):
        raise AssertionError("zero-longitudinal visual yaw must release normal brake hold")
    tracker._brake_hold_stop_mode = "emergency"
    tracker._brake_hold_label = "safety_hold_front_ir"
    original_hard_stop_check = runtime.hard_stop_check
    runtime.hard_stop_check = lambda _action=None: True
    try:
        if runtime.can_release_brake_hold(mod.ACTION_BACKWARD):
            raise AssertionError("active IR safety hold must block zero-longitudinal yaw")
    finally:
        runtime.hard_stop_check = original_hard_stop_check
        tracker._brake_hold_stop_mode = None
        tracker._brake_hold_label = "brake"
    tracker._last_control_decision_reason = "distance_too_close"
    if runtime.can_release_brake_hold(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("hard-close stop must not be released by a rotate command")
    tracker._follow_controller.target_stop_latched = False

    tracker._brake_hold_label = "aimline_brake"
    with runtime._steering_feedback_lock:
        runtime._steering_feedback = SteeringFeedback(
            timestamp=time.monotonic(),
            left_speed_rpm=-12,
            right_speed_rpm=12,
            yaw_rate_right_dps=-37.0,
            raw_yaw_rate_right_dps=-37.0,
            trustworthy=True,
        )
    if runtime.can_release_brake_hold(mod.ACTION_ROTATE_LEFT):
        raise AssertionError(
            "an empty frame must not restart search while aimline braking still has yaw"
        )
    with runtime._steering_feedback_lock:
        runtime._steering_feedback = SteeringFeedback(
            timestamp=time.monotonic(),
            left_speed_rpm=0,
            right_speed_rpm=0,
            yaw_rate_right_dps=1.0,
            raw_yaw_rate_right_dps=1.0,
            trustworthy=True,
        )
    if not runtime.can_release_brake_hold(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("settled encoder feedback must release the aimline brake hold")
    tracker._brake_hold_label = "brake"
    print("parked_recenter_brake_release: PASS")

    tracker._brake_hold_active = True
    if not tracker._should_skip_redundant_action_queue(
        [mod.ACTION_STOP], "near_distance_rotation_only"
    ):
        raise AssertionError("an active brake hold must absorb repeated STOP publications")
    tracker._brake_hold_active = False
    print("stop_publication_idempotent: PASS")

    tracker.action_stop_event = threading.Event()
    backend.driver.left = 30
    backend.driver.right = -20
    feedback_thread = threading.Thread(target=runtime._steering_feedback_loop, daemon=True)
    feedback_thread.start()
    feedback_deadline = time.monotonic() + 0.60
    feedback = None
    while time.monotonic() < feedback_deadline:
        feedback = runtime.get_steering_feedback()
        if feedback is not None:
            break
        time.sleep(0.01)
    tracker.action_stop_event.set()
    feedback_thread.join(timeout=0.50)
    if feedback_thread.is_alive():
        raise AssertionError("encoder feedback thread must exit when action_stop_event is set")
    if feedback is None or not feedback.trustworthy or feedback.yaw_rate_right_dps <= 0.0:
        raise AssertionError(f"encoder feedback must report a trustworthy right yaw: {feedback}")
    tracker.action_stop_event.clear()
    print("encoder_feedback_thread: PASS")

    if not mod.ROTATE_PULSE_BRAKE_ENABLE:
        raise AssertionError("rotate pulse scan must be enabled")
    if not math.isclose(float(mod.ROTATE_DURATION), 0.10):
        raise AssertionError(f"unexpected rotate pulse duration: {mod.ROTATE_DURATION}")
    if not math.isclose(float(mod.ROTATE_PULSE_PAUSE_SEC), 0.00):
        raise AssertionError(f"unexpected rotate observation pause: {mod.ROTATE_PULSE_PAUSE_SEC}")
    if int(mod.ROTATE_PULSE_OBSERVE_MIN_FRAMES) != 1:
        raise AssertionError(f"unexpected rotate observation frame gate: {mod.ROTATE_PULSE_OBSERVE_MIN_FRAMES}")
    if not mod.ROTATE_PULSE_SETTLE_ENABLE:
        raise AssertionError("encoder settle gate must be enabled for search pulses")
    if mod.SEARCH_ROTATE_CONTINUOUS_ENABLE:
        raise AssertionError("search rotation should use short pulse mode")
    if not math.isclose(float(mod.ROTATE_PULSE_SETTLE_QUIET_SEC), 0.05):
        raise AssertionError(f"unexpected rotate settle quiet time: {mod.ROTATE_PULSE_SETTLE_QUIET_SEC}")
    if not math.isclose(float(mod.ROTATE_PULSE_SETTLE_TIMEOUT_SEC), 0.25):
        raise AssertionError(f"unexpected rotate settle timeout: {mod.ROTATE_PULSE_SETTLE_TIMEOUT_SEC}")
    if int(mod.ROTATE_PULSE_TRANSITION_RPM) != 1:
        raise AssertionError(
            f"search pulse transition must use 1 RPM: {mod.ROTATE_PULSE_TRANSITION_RPM}"
        )
    for reason, expected in (
        ("person_left_rotate", (int(mod.ROTATE_RAW_TARGET_VISIBLE), "visible")),
        ("near_distance_rotation_only", (int(mod.ROTATE_RAW_TARGET_VISIBLE), "visible")),
        (
            "lost_history_hold_right",
            (
                int(mod.VISIBLE_STEERING_PID_LOST_HOLD_MAX_CORRECTION_RPM),
                "lost_confirm",
            ),
        ),
        (
            "lost_current_candidate_hold_right",
            (
                int(mod.VISIBLE_STEERING_PID_LOST_HOLD_MAX_CORRECTION_RPM),
                "lost_confirm",
            ),
        ),
        (
            "search_candidate_approach_right",
            (max(1, int(mod.PARKED_RECENTER_MIN_RPM)), "candidate_approach"),
        ),
        (
            "search_current_candidate_right",
            (int(mod.SEARCH_CANDIDATE_ACQUIRE_RAW_RPM), "candidate_acquire"),
        ),
        ("lost_wait_last_left", (int(mod.ROTATE_RAW_TARGET_LOST_WAIT), "lost_wait")),
        ("search_left", (int(mod.ROTATE_RAW_TARGET_SEARCH), "search")),
    ):
        actual = mod.PersonTracker._rotate_raw_target_for_reason(reason)
        if actual != expected:
            raise AssertionError(f"rotate speed mapping for {reason}: expected {expected}, got {actual}")

    runtime.send_rotate_transition_hold(mod.ACTION_ROTATE_RIGHT)
    transition_expected = (
        backend.wheel_raw_state_to_target("left", 1, 0x01),
        backend.wheel_raw_state_to_target("right", 1, 0x02),
    )
    _assert_pair(
        "rotate_transition_one_rpm",
        (backend.driver.left, backend.driver.right),
        transition_expected,
    )
    tracker._rotate_transition_hold_active = False

    tracker.frame_index = 100
    tracker._rotate_pause_until_ts = time.time() + 1.0
    tracker._rotate_observe_until_frame = (
        tracker.frame_index + int(mod.ROTATE_PULSE_OBSERVE_MIN_FRAMES)
    )
    if not runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("time and frame observation gates should block the next pulse")
    tracker._rotate_pause_until_ts = time.time() - 1.0
    tracker.frame_index = tracker._rotate_observe_until_frame - 1
    if not runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("the next pulse must wait for all configured visual frames")
    tracker.frame_index = tracker._rotate_observe_until_frame
    if runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("the pulse gate should open after pause time and configured frames")

    def set_feedback(
        left_rpm: int,
        right_rpm: int,
        yaw_rate_dps: float,
        *,
        age_sec: float = 0.0,
        trustworthy: bool = True,
    ) -> None:
        feedback = SteeringFeedback(
            timestamp=time.monotonic() - max(0.0, float(age_sec)),
            left_speed_rpm=left_rpm,
            right_speed_rpm=right_rpm,
            yaw_rate_right_dps=yaw_rate_dps,
            trustworthy=trustworthy,
        )
        with runtime._steering_feedback_lock:
            runtime._steering_feedback = feedback

    tracker.frame_index = 200
    runtime.begin_rotate_pulse_observation()
    tracker._rotate_pause_until_ts = time.time() - 1.0
    set_feedback(4, -4, 8.0)
    if not runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("moving wheel feedback must keep the settle gate closed")
    if tracker._rotate_observe_until_frame != -1:
        raise AssertionError("moving frames must not arm the visual observation gate")

    set_feedback(0, 0, 0.0)
    if not runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("a single quiet sample must not release the settle gate")
    tracker._rotate_settle_quiet_started_monotonic = (
        time.monotonic() - float(mod.ROTATE_PULSE_SETTLE_QUIET_SEC) - 0.01
    )
    set_feedback(2, 0, 0.0)
    if not runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("post-stop wheel rebound must keep the settle gate closed")
    if tracker._rotate_settle_quiet_started_monotonic != 0.0:
        raise AssertionError("wheel rebound must reset the continuous quiet timer")

    set_feedback(0, 0, 0.0)
    tracker._rotate_settle_quiet_started_monotonic = (
        time.monotonic() - float(mod.ROTATE_PULSE_SETTLE_QUIET_SEC) - 0.01
    )
    if not runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("settled wheels must still wait for fresh visual frames")
    expected_settled_frame = tracker.frame_index + int(mod.ROTATE_PULSE_OBSERVE_MIN_FRAMES)
    if tracker._rotate_settle_pending or tracker._rotate_observe_until_frame != expected_settled_frame:
        raise AssertionError(
            "continuous quiet feedback must arm fresh post-settle frames: "
            f"pending={tracker._rotate_settle_pending} observe_until={tracker._rotate_observe_until_frame}"
        )
    tracker.frame_index = expected_settled_frame
    if runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("settle gate must open after the post-settle frame count")

    tracker.frame_index = 300
    runtime.begin_rotate_pulse_observation()
    tracker._rotate_pause_until_ts = time.time() - 1.0
    tracker._rotate_settle_started_monotonic = (
        time.monotonic() - float(mod.ROTATE_PULSE_SETTLE_TIMEOUT_SEC) - 0.01
    )
    set_feedback(
        0,
        0,
        0.0,
        age_sec=float(mod.ROTATE_PULSE_SETTLE_FEEDBACK_STALE_SEC) + 0.10,
    )
    if not runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("feedback timeout fallback must still require fresh visual frames")
    if tracker._rotate_settle_completion_source != "feedback_timeout":
        raise AssertionError(
            f"stale feedback must use timeout fallback: {tracker._rotate_settle_completion_source}"
        )
    tracker.frame_index = tracker._rotate_observe_until_frame
    if runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("timeout fallback must open after fresh visual frames")
    print("rotate_pulse_encoder_settle_gate: PASS")

    tracker.frame_index = 320
    tracker._follow_controller = SimpleNamespace(
        target_stop_latched=False,
        defer_search_timeout=lambda _seconds: None,
    )
    first_epoch = runtime.begin_search_epoch("lost_confirmed_stop_before_search")
    first_started = tracker._rotate_settle_started_monotonic
    if first_epoch != 1 or tracker._rotate_settle_search_epoch != first_epoch:
        raise AssertionError(
            "new search must own a fresh settle epoch: "
            f"search={first_epoch} settle={tracker._rotate_settle_search_epoch}"
        )
    if not tracker._rotate_settle_pending:
        raise AssertionError("new search must wait for the current STOP to settle")
    runtime.cancel_rotate_pulse_observation("target_visible:reacquired")
    if (
        tracker._rotate_settle_pending
        or tracker._rotate_settle_search_epoch != -1
        or tracker._rotate_observe_until_frame != -1
        or tracker._rotate_pause_until_ts != 0.0
    ):
        raise AssertionError("stable target reacquire must clear all previous search gates")

    second_epoch = runtime.begin_search_epoch("lost_confirmed_stop_before_search")
    if second_epoch != first_epoch + 1:
        raise AssertionError(f"search epoch must increase for each search: {second_epoch}")
    if tracker._rotate_settle_started_monotonic < first_started:
        raise AssertionError("new search settle timing must start from the current search")
    continuous_epoch = runtime.begin_search_epoch("same_direction_yaw_handoff", settle=False)
    if continuous_epoch != second_epoch + 1:
        raise AssertionError("continuous handoff must still create a fresh search epoch")
    if (
        tracker._rotate_settle_pending
        or tracker._rotate_pause_until_ts != 0.0
        or tracker._rotate_observe_until_frame != -1
        or tracker._rotate_settle_completion_source != "continuous_handoff"
    ):
        raise AssertionError("same-direction search handoff must not insert a settle gate")
    tracker._rotate_settle_search_epoch = first_epoch
    if runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("a stale settle epoch must not block the current search")
    if tracker._rotate_settle_pending:
        raise AssertionError("stale settle state must be cancelled immediately")
    print("search_epoch_settle_lifecycle: PASS")

    tracker._current_rotate_pulse_enabled = False
    tracker._rotate_pause_until_ts = time.time() + 1.0
    tracker._rotate_observe_until_frame = tracker.frame_index + 3
    if runtime.rotate_pulse_observation_pending(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("parked visual recenter must bypass search pulse observation gates")
    tracker._current_rotate_raw_target = mod.PARKED_RECENTER_MIN_RPM
    tracker._current_rotate_raw_source = "parked_pid_encoder"
    parked_signature = tracker._action_signature(mod.ACTION_ROTATE_LEFT)
    if parked_signature[4] is not False:
        raise AssertionError(f"parked recenter signature must disable pulses: {parked_signature}")

    tracker.current_command = mod.ACTION_ROTATE_LEFT
    tracker.command_start_time = time.time()
    tracker._last_rotate_pulse_refresh_ts = 0.0
    runtime.send_robot_command(mod.ACTION_ROTATE_LEFT)
    parked_expected = (
        backend.wheel_raw_state_to_target("left", mod.PARKED_RECENTER_MIN_RPM, 0x02),
        backend.wheel_raw_state_to_target("right", mod.PARKED_RECENTER_MIN_RPM, 0x01),
    )
    _assert_pair(
        "parked_recenter_continuous",
        (backend.driver.left, backend.driver.right),
        parked_expected,
    )
    if tracker._last_rotate_pulse_refresh_ts != 0.0:
        raise AssertionError("parked recenter must not start or refresh the search pulse timer")

    tracker._last_rotate_hold_target_send_ts = 0.0
    if not runtime.forward_like_refresh_due(mod.ACTION_ROTATE_LEFT, force=True):
        raise AssertionError("first visual-hold rotate command should dispatch immediately")
    if runtime.forward_like_refresh_due(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("visual-hold rotate refresh should respect the RS485 minimum interval")
    tracker._last_rotate_hold_target_send_ts -= float(mod.MOTOR_RS485_TARGET_MIN_INTERVAL_SEC)
    if not runtime.forward_like_refresh_due(mod.ACTION_ROTATE_LEFT):
        raise AssertionError("visual-hold rotate command should refresh after the RS485 interval")
    tracker.command_start_time = time.time() - 1.0
    tracker._last_rotate_visual_refresh_ts = time.time()
    if runtime.rotate_hold_age_sec() >= float(mod.ROTATE_HOLD_STALE_SEC):
        raise AssertionError("fresh visual rotate heartbeat should keep the turn active")
    tracker._last_rotate_visual_refresh_ts -= float(mod.ROTATE_HOLD_STALE_SEC) + 0.01
    if runtime.rotate_hold_age_sec() < float(mod.ROTATE_HOLD_STALE_SEC):
        raise AssertionError("stale visual rotate heartbeat should trigger the safety timeout")
    tracker.command_start_time = None
    tracker.current_command = None
    tracker._current_rotate_pulse_enabled = True
    tracker._current_rotate_raw_target = mod.MOTOR_ROTATE_RAW_TARGET
    tracker._current_rotate_raw_source = "default"
    print("visual_hold_refresh", float(mod.MOTOR_RS485_TARGET_MIN_INTERVAL_SEC))
    if not runtime.needs_transition_stop(mod.ACTION_FORWARD, mod.ACTION_ROTATE_LEFT):
        raise AssertionError("forward -> rotate_left should require transition stop")
    if runtime.needs_transition_stop(mod.ACTION_FORWARD, mod.ACTION_STEER_LEFT):
        raise AssertionError("forward -> steer_left should update wheel speed without transition stop")
    if runtime.needs_transition_stop(mod.ACTION_STEER_LEFT, mod.ACTION_FORWARD):
        raise AssertionError("steer_left -> forward should update wheel speed without transition stop")
    if not runtime.needs_transition_stop(mod.ACTION_FORWARD, mod.ACTION_BACKWARD):
        raise AssertionError("forward -> backward must stop both wheels before reversing")
    if not runtime.needs_transition_stop(mod.ACTION_BACKWARD, mod.ACTION_FORWARD):
        raise AssertionError("backward -> forward must stop both wheels before reversing direction")
    if not runtime.needs_transition_stop(mod.ACTION_BACKWARD, mod.ACTION_STEER_LEFT):
        raise AssertionError("backward -> steer must stop both wheels before forward differential drive")
    if runtime.needs_transition_stop(mod.ACTION_STEER_LEFT, mod.ACTION_STEER_RIGHT):
        raise AssertionError("steer_left -> steer_right should update wheel speed without transition stop")
    if not runtime.needs_transition_stop(mod.ACTION_STEER_LEFT, mod.ACTION_ROTATE_RIGHT):
        raise AssertionError("steer_left -> rotate_right should require transition stop")
    if not runtime.needs_transition_stop(mod.ACTION_ROTATE_LEFT, mod.ACTION_FORWARD):
        raise AssertionError("rotate_left -> forward should require transition stop")
    if not runtime.needs_transition_stop(mod.ACTION_ROTATE_LEFT, mod.ACTION_ROTATE_RIGHT):
        raise AssertionError("rotate_left -> rotate_right should require transition stop")
    if runtime.needs_transition_stop(mod.ACTION_ROTATE_LEFT, mod.ACTION_ROTATE_LEFT):
        raise AssertionError("same rotate action should not require transition stop")
    tracker._current_steer_base_percent = max(35, int(mod.MIN_FORWARD_PERCENT))
    tracker._current_steer_inner_ratio_percent = int(mod.VISIBLE_STEER_INNER_RATIO_PERCENT)
    tracker._current_steer_outer_ratio_percent = int(mod.VISIBLE_STEER_OUTER_RATIO_PERCENT)
    tracker._current_steer_correction_rpm = 0
    tracker._current_rotate_turn_percent = int(mod.ROTATE_TURN_PERCENT_FROM_FORWARD)
    tracker._last_control_decision_reason = "person_left"
    runtime.send_robot_command(mod.ACTION_STEER_LEFT)
    if int(mod.MOTOR_STEER_RAW_TARGET) > 0:
        steer_base = round(
            int(mod.MOTOR_FORWARD_MAX_TARGET_RPM)
            * tracker._current_steer_base_percent
            / 100.0
        )
        steer_inner = max(0, int(math.floor(steer_base * mod.VISIBLE_STEER_INNER_RATIO_PERCENT / 100.0)))
        steer_outer = max(0, int(math.ceil(steer_base * mod.VISIBLE_STEER_OUTER_RATIO_PERCENT / 100.0)))
        expected_left = backend.wheel_raw_state_to_target("left", steer_inner, 0x01)
        expected_right = backend.wheel_raw_state_to_target("right", steer_outer, 0x01)
        _assert_pair("steer_left_distance_curve_raw", (backend.driver.left, backend.driver.right), (expected_left, expected_right))
    else:
        steer_cap = max(0, min(100, mod.STEER_PERCENT_LIMIT))
        base_cap = min(mod.MAX_FORWARD_PERCENT, steer_cap)
        steer_base = max(0, min(base_cap, tracker._current_steer_base_percent))
        steer_inner = int(math.floor(steer_base * mod.VISIBLE_STEER_INNER_RATIO_PERCENT / 100.0))
        steer_outer = int(math.ceil(steer_base * mod.VISIBLE_STEER_OUTER_RATIO_PERCENT / 100.0))
        steer_inner = max(0, min(steer_cap, steer_inner))
        steer_outer = max(0, min(steer_cap, steer_outer))
        expected_left = backend.wheel_state_to_target("left", steer_inner, 0x01)
        expected_right = backend.wheel_state_to_target("right", steer_outer, 0x01)
        _assert_pair("steer_left_percent", (backend.driver.left, backend.driver.right), (expected_left, expected_right))

    if int(mod.MOTOR_STEER_RAW_TARGET) > 0:
        # 摄像头外环请求右转、编码器内环给出 3 RPM 修正时，左轮应比
        # 距离曲线基准快 3 RPM，右轮应慢 3 RPM；实车前进符号为左正右负。
        tracker._current_steer_base_percent = 60
        tracker._current_steer_correction_rpm = 3
        tracker._last_control_decision_reason = "visual_pid_right_encoder"
        runtime.send_robot_command(mod.ACTION_STEER_RIGHT)
        pid_base = round(int(mod.MOTOR_FORWARD_MAX_TARGET_RPM) * 60 / 100.0)
        expected_pid_left = backend.wheel_raw_state_to_target("left", pid_base + 3, 0x01)
        expected_pid_right = backend.wheel_raw_state_to_target("right", pid_base - 3, 0x01)
        _assert_pair(
            "steer_right_encoder_pid",
            (backend.driver.left, backend.driver.right),
            (expected_pid_left, expected_pid_right),
        )
        if backend.driver.left <= 0 or backend.driver.right >= 0:
            raise AssertionError(
                "PID differential must preserve real forward signs: left positive, right negative"
            )

        # 大偏差动态档按当前前进上限计算纵向基准，再独立叠加
        # 12 RPM 横向修正（实车右轮前进寄存器符号为负）。
        tracker._current_steer_base_percent = 70
        tracker._current_steer_correction_rpm = 12
        runtime.send_robot_command(mod.ACTION_STEER_RIGHT)
        dynamic_base = round(int(mod.MOTOR_FORWARD_MAX_TARGET_RPM) * 70 / 100.0)
        expected_dynamic_left = backend.wheel_raw_state_to_target(
            "left",
            dynamic_base + 12,
            0x01,
        )
        expected_dynamic_right = backend.wheel_raw_state_to_target(
            "right",
            max(0, dynamic_base - 12),
            0x01,
        )
        _assert_pair(
            "steer_right_dynamic_pid",
            (backend.driver.left, backend.driver.right),
            (expected_dynamic_left, expected_dynamic_right),
        )

        # Depth confidence may reduce longitudinal speed to zero while the
        # camera still sees a person at the edge. Preserve full pivot yaw.
        tracker._current_steer_base_percent = 0
        tracker._current_steer_correction_rpm = 10
        tracker._last_control_decision_reason = "visual_pid_left_camera_distance_missing_yaw_only"
        runtime.send_robot_command(mod.ACTION_STEER_LEFT)
        expected_yaw_only = (
            backend.wheel_raw_state_to_target("left", 10, 0x02),
            backend.wheel_raw_state_to_target("right", 10, 0x01),
        )
        _assert_pair(
            "steer_left_zero_longitudinal_yaw",
            (backend.driver.left, backend.driver.right),
            expected_yaw_only,
        )
        tracker._current_steer_correction_rpm = 0
        tracker._last_control_decision_reason = "person_left"

    tracker._current_steer_base_percent = 87
    runtime.send_robot_command(mod.ACTION_STEER_LEFT)
    if int(mod.MOTOR_STEER_RAW_TARGET) > 0:
        far_steer_base = round(int(mod.MOTOR_FORWARD_MAX_TARGET_RPM) * 87 / 100.0)
        far_steer_inner = max(
            0,
            int(math.floor(far_steer_base * mod.VISIBLE_STEER_INNER_RATIO_PERCENT / 100.0)),
        )
        far_steer_outer = max(
            0,
            int(math.ceil(far_steer_base * mod.VISIBLE_STEER_OUTER_RATIO_PERCENT / 100.0)),
        )
        expected_far_left = backend.wheel_raw_state_to_target("left", far_steer_inner, 0x01)
        expected_far_right = backend.wheel_raw_state_to_target("right", far_steer_outer, 0x01)
        _assert_pair(
            "steer_left_4m_distance_curve_raw",
            (backend.driver.left, backend.driver.right),
            (expected_far_left, expected_far_right),
        )
        if max(abs(backend.driver.left), abs(backend.driver.right)) <= 30:
            raise AssertionError("4m differential follow was still clipped by the old 30 rpm ceiling")

    normal_signature = tracker._action_signature(mod.ACTION_STEER_LEFT)
    tracker._last_control_decision_reason = "person_left_mmwave_hold"
    hold_signature = tracker._action_signature(mod.ACTION_STEER_LEFT)
    if hold_signature == normal_signature:
        raise AssertionError("entering mmwave hold must force a new steer motor dispatch")
    runtime.send_robot_command(mod.ACTION_STEER_LEFT)
    if int(mod.MOTOR_FORWARD_MAX_TARGET_RPM) > 0:
        hold_raw_cap = round(
            int(mod.MOTOR_FORWARD_MAX_TARGET_RPM)
            * int(mod.VISION_MMWAVE_HOLD_FORWARD_PERCENT)
            / 100.0
        )
    else:
        hold_raw_cap = backend.percent_to_target(mod.VISION_MMWAVE_HOLD_FORWARD_PERCENT)
    if int(mod.MOTOR_STEER_RAW_TARGET) > 0:
        steer_base = min(
            round(
                int(mod.MOTOR_FORWARD_MAX_TARGET_RPM)
                * min(
                    int(tracker._current_steer_base_percent),
                    int(mod.VISION_MMWAVE_HOLD_FORWARD_PERCENT),
                )
                / 100.0
            ),
            hold_raw_cap,
        )
        steer_inner = min(
            hold_raw_cap,
            max(0, int(math.floor(steer_base * mod.VISIBLE_STEER_INNER_RATIO_PERCENT / 100.0))),
        )
        steer_outer = min(
            hold_raw_cap,
            max(0, int(math.ceil(steer_base * mod.VISIBLE_STEER_OUTER_RATIO_PERCENT / 100.0))),
        )
        expected_left = backend.wheel_raw_state_to_target("left", steer_inner, 0x01)
        expected_right = backend.wheel_raw_state_to_target("right", steer_outer, 0x01)
    else:
        hold_percent_cap = int(mod.VISION_MMWAVE_HOLD_FORWARD_PERCENT)
        steer_inner = min(
            hold_percent_cap,
            max(0, int(math.floor(hold_percent_cap * mod.VISIBLE_STEER_INNER_RATIO_PERCENT / 100.0))),
        )
        steer_outer = min(
            hold_percent_cap,
            max(0, int(math.ceil(hold_percent_cap * mod.VISIBLE_STEER_OUTER_RATIO_PERCENT / 100.0))),
        )
        expected_left = backend.wheel_state_to_target("left", steer_inner, 0x01)
        expected_right = backend.wheel_state_to_target("right", steer_outer, 0x01)
    _assert_pair("steer_left_mmwave_hold", (backend.driver.left, backend.driver.right), (expected_left, expected_right))
    if max(abs(backend.driver.left), abs(backend.driver.right)) > hold_raw_cap:
        raise AssertionError("mmwave hold steer wheel exceeded the configured curve cap")
    tracker._last_control_decision_reason = "person_left"

    # DRIVE uses its own 50 rpm ceiling; the backend's 30 rpm ceiling remains
    # in force for direct/legacy targets and therefore does not change TURN.
    tracker._current_forward_percent = 40
    tracker._last_control_decision_reason = "follow_distance"
    runtime.send_robot_command(mod.ACTION_FORWARD)
    forward_min_raw = round(int(mod.MOTOR_FORWARD_MAX_TARGET_RPM) * 40 / 100.0)
    _assert_pair(
        "forward_curve_min",
        (backend.driver.left, backend.driver.right),
        (
            backend.wheel_raw_state_to_target("left", forward_min_raw, 0x01),
            backend.wheel_raw_state_to_target("right", forward_min_raw, 0x01),
        ),
    )
    tracker._current_forward_percent = 20
    tracker._last_control_decision_reason = "target_approaching_reverse"
    runtime.send_robot_command(mod.ACTION_BACKWARD)
    reverse_raw = round(int(mod.MOTOR_FORWARD_MAX_TARGET_RPM) * 20 / 100.0)
    expected_reverse = (
        backend.wheel_raw_state_to_target("left", reverse_raw, 0x02),
        backend.wheel_raw_state_to_target("right", reverse_raw, 0x02),
    )
    _assert_pair(
        "reverse_distance_control",
        (backend.driver.left, backend.driver.right),
        expected_reverse,
    )
    if not (backend.driver.left < 0 and backend.driver.right > 0):
        raise AssertionError(
            f"reverse encoder convention must be left negative/right positive, got "
            f"{backend.driver.left}/{backend.driver.right}"
        )
    tracker._current_steer_correction_rpm = 8
    runtime.send_robot_command(mod.ACTION_BACKWARD)
    expected_reverse_right = (
        backend.wheel_raw_state_to_target("left", reverse_raw - 8, 0x02),
        backend.wheel_raw_state_to_target("right", reverse_raw + 8, 0x02),
    )
    _assert_pair(
        "reverse_visual_pid_right",
        (backend.driver.left, backend.driver.right),
        expected_reverse_right,
    )
    tracker._current_steer_correction_rpm = -8
    runtime.send_robot_command(mod.ACTION_BACKWARD)
    expected_reverse_left = (
        backend.wheel_raw_state_to_target("left", reverse_raw + 8, 0x02),
        backend.wheel_raw_state_to_target("right", reverse_raw - 8, 0x02),
    )
    _assert_pair(
        "reverse_visual_pid_left",
        (backend.driver.left, backend.driver.right),
        expected_reverse_left,
    )
    tracker._current_forward_percent = 0
    tracker._current_steer_correction_rpm = 6
    runtime.send_robot_command(mod.ACTION_BACKWARD)
    expected_reverse_yaw_only = (
        backend.wheel_raw_state_to_target("left", 6, 0x01),
        backend.wheel_raw_state_to_target("right", 6, 0x02),
    )
    _assert_pair(
        "reverse_zero_longitudinal_yaw",
        (backend.driver.left, backend.driver.right),
        expected_reverse_yaw_only,
    )
    if not (backend.driver.left > 0 and backend.driver.right > 0):
        raise AssertionError(
            "right yaw-only command must use opposing wheel directions and "
            f"produce positive encoder signs, got {backend.driver.left}/{backend.driver.right}"
        )
    tracker._current_steer_correction_rpm = 0
    fallback_raw = int(mod.VISIBLE_STEERING_PID_FALLBACK_BASE_RPM)
    tracker._current_forward_percent = round(
        100.0 * fallback_raw / max(1, int(mod.MOTOR_FORWARD_MAX_TARGET_RPM))
    )
    tracker._last_control_decision_reason = "visual_pid_center_camera_distance_missing"
    runtime.send_robot_command(mod.ACTION_FORWARD)
    _assert_pair(
        "visual_pid_center_distance_missing_fallback",
        (backend.driver.left, backend.driver.right),
        (
            backend.wheel_raw_state_to_target("left", fallback_raw, 0x01),
            backend.wheel_raw_state_to_target("right", fallback_raw, 0x01),
        ),
    )
    tracker._current_forward_percent = 100
    tracker._last_control_decision_reason = "follow_distance"
    runtime.send_robot_command(mod.ACTION_FORWARD)
    forward_max_raw = int(mod.MOTOR_FORWARD_MAX_TARGET_RPM)
    _assert_pair(
        "forward_curve_max",
        (backend.driver.left, backend.driver.right),
        (
            backend.wheel_raw_state_to_target("left", forward_max_raw, 0x01),
            backend.wheel_raw_state_to_target("right", forward_max_raw, 0x01),
        ),
    )

    tracker.current_command = mod.ACTION_ROTATE_LEFT
    runtime.send_robot_command(mod.ACTION_ROTATE_LEFT)
    if int(mod.MOTOR_ROTATE_RAW_TARGET) > 0:
        expected_left = backend.wheel_raw_state_to_target("left", int(mod.MOTOR_ROTATE_RAW_TARGET), 0x02)
        expected_right = backend.wheel_raw_state_to_target("right", int(mod.MOTOR_ROTATE_RAW_TARGET), 0x01)
    else:
        expected_left = backend.wheel_state_to_target("left", tracker._current_rotate_turn_percent, 0x02)
        expected_right = backend.wheel_state_to_target("right", tracker._current_rotate_turn_percent, 0x01)
    _assert_pair("rotate_left", (backend.driver.left, backend.driver.right), (expected_left, expected_right))

    tracker.current_command = mod.ACTION_ROTATE_RIGHT
    tracker.command_start_time = None
    runtime.send_robot_command(mod.ACTION_ROTATE_RIGHT)
    if int(mod.MOTOR_ROTATE_RAW_TARGET) > 0:
        expected_left = backend.wheel_raw_state_to_target("left", int(mod.MOTOR_ROTATE_RAW_TARGET), 0x01)
        expected_right = backend.wheel_raw_state_to_target("right", int(mod.MOTOR_ROTATE_RAW_TARGET), 0x02)
    else:
        expected_left = backend.wheel_state_to_target("left", tracker._current_rotate_turn_percent, 0x01)
        expected_right = backend.wheel_state_to_target("right", tracker._current_rotate_turn_percent, 0x02)
    _assert_pair("rotate_right", (backend.driver.left, backend.driver.right), (expected_left, expected_right))

    runtime.send_motion_transition_stop(mod.ACTION_FORWARD, mod.ACTION_ROTATE_LEFT)
    _assert_pair("transition_stop", (backend.driver.left, backend.driver.right), (0, 0))
    if backend.driver.emergency_stops != 2:
        raise AssertionError(f"expected two emergency stops, got {backend.driver.emergency_stops}")
    backend.motion_armed = True

    tracker._current_forward_percent = 30
    tracker._current_steer_base_percent = 12
    tracker.is_forwarding = True
    runtime.send_stop_with_brake_hold("search_to_follow")
    if tracker._current_forward_percent != 30 or tracker._current_steer_base_percent != 12:
        raise AssertionError(
            "search_to_follow transition stop should preserve the just-decided motion parameters"
        )
    backend.motion_armed = True

    tracker._current_forward_percent = 32
    tracker._current_steer_base_percent = 0
    tracker.is_forwarding = False
    runtime.send_stop_with_brake_hold(
        "queued_action_stop_signal",
        preserve_motion_params=True,
    )
    if tracker._current_forward_percent != 32:
        raise AssertionError(
            "forward-to-reverse transition stop must preserve the queued reverse RPM"
        )
    backend.motion_armed = True
    runtime.send_robot_command(mod.ACTION_BACKWARD)
    expected_reverse = (
        backend.wheel_raw_state_to_target("left", 32, 0x02),
        backend.wheel_raw_state_to_target("right", 32, 0x02),
    )
    _assert_pair(
        "queued_reverse_first_write_nonzero",
        (backend.driver.left, backend.driver.right),
        expected_reverse,
    )
    if backend.driver.left == 0 or backend.driver.right == 0:
        raise AssertionError("first queued reverse write must be nonzero")
    backend.motion_armed = True

    backend.send_diff(99, 0x01, 99, 0x01, "limit")
    _assert_pair("limit", (backend.driver.left, backend.driver.right), _expected_send_diff(backend, 99, 0x01, 99, 0x01))

    backend.send_targets(999, -999, "absolute_rpm_limit")
    _assert_pair(
        "absolute_rpm_limit",
        (backend.driver.left, backend.driver.right),
        (int(mod.MOTOR_RS485_MAX_TARGET), -int(mod.MOTOR_RS485_MAX_TARGET)),
    )

    backend.send_stop("test_stop")
    _assert_pair("stop", (backend.driver.left, backend.driver.right), (0, 0))
    if not runtime.forward_like_refresh_due(mod.ACTION_STOP, force=True):
        raise AssertionError("a newly queued STOP must be dispatched once")
    if runtime.forward_like_refresh_due(mod.ACTION_STOP):
        raise AssertionError("STOP must not be replayed continuously on the RS485 bus")
    print("stop_no_busy_keepalive: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
