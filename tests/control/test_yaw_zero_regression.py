"""Dry-run regressions for revoking visible TURN writes at the motor boundary."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.action_runtime import MotionActionRuntime


class FakeBackend:
    def __init__(self):
        self.config = SimpleNamespace(m1_is_left_wheel=True, max_target=100)
        self.targets = []

    @staticmethod
    def wheel_raw_state_to_target(_side, raw, state):
        return int(raw) if state == 0x01 else -int(raw) if state == 0x02 else 0

    @staticmethod
    def clip_percent(percent):
        return max(0, min(100, int(percent)))

    def send_targets(self, left, right, label, **_kwargs):
        self.targets.append((int(left), int(right), label))

    def send_diff(self, left, left_state, right, right_state, label):
        self.send_targets(
            self.wheel_raw_state_to_target("left", left, left_state),
            self.wheel_raw_state_to_target("right", right, right_state),
            label,
        )


def make_runtime(*, raw=5, source="lateral_intent_30hz"):
    symbols = SimpleNamespace(
        forward=1,
        backward=2,
        rotate_left=3,
        rotate_right=4,
        stop=5,
        steer_left=6,
        steer_right=7,
        action_names={3: "rotate_left", 4: "rotate_right"},
    )
    owner = SimpleNamespace(
        motor_io_lock=threading.Lock(),
        current_command=symbols.rotate_right,
        frame_index=217,
        _current_rotate_raw_target=raw,
        _current_rotate_raw_source=source,
        _current_rotate_turn_percent=18,
        _current_forward_percent=17,
        _current_steer_base_percent=17,
        _current_steer_correction_rpm=5,
        _current_steer_inner_ratio_percent=100,
        _current_steer_outer_ratio_percent=100,
        _brake_hold_active=False,
        _last_motor_dispatch_source="keepalive_refresh",
    )
    config = SimpleNamespace(
        steering_feedback_median_window=3,
        motor_rotate_raw_target=18,
        rotate_turn_percent_from_forward=18,
        rotate_pulse_brake_enable=False,
        rotate_duration=0.18,
        rotate_pulse_pause_sec=0.0,
        rotate_hold_stale_sec=0.25,
        motor_forward_raw_target=0,
        motor_forward_max_target_rpm=100,
        motor_steer_raw_target=15,
        steer_percent_limit=100,
        max_forward_percent=100,
        mmwave_hold_forward_percent=20,
        visible_steer_inner_ratio_percent=100,
        visible_steer_outer_ratio_percent=100,
        rotation_only=False,
    )
    backend = FakeBackend()
    runtime = MotionActionRuntime(
        owner,
        backend,
        config,
        symbols,
        hard_stop_check=lambda _action=None: False,
        logger=logging.getLogger("yaw-zero-regression"),
    )
    return runtime, owner, backend, symbols


def test_revoked_raw_keepalive_cannot_overwrite_depth_translation():
    runtime, owner, backend, symbols = make_runtime()
    authorization = {"valid": True}
    guard_calls = []

    def allowed(action):
        assert owner.motor_io_lock.locked()
        guard_calls.append(action)
        return authorization["valid"]

    owner._visible_rotate_command_allowed = allowed
    runtime.send_robot_command(symbols.rotate_right)
    assert backend.targets == [(5, -5, "TURN")]

    # A newer Depth write has already established translation; an old yaw
    # refresh must neither overwrite it nor replace it with a two-wheel STOP.
    backend.send_targets(17, 17, "DRIVE")
    owner.current_command = symbols.forward
    authorization["valid"] = False
    runtime.send_robot_command(symbols.rotate_right)
    assert backend.targets == [(5, -5, "TURN"), (17, 17, "DRIVE")]
    assert owner.current_command == symbols.forward
    assert owner._current_forward_percent == 17
    assert owner._current_steer_base_percent == 17
    assert not owner._brake_hold_active
    assert guard_calls == [symbols.rotate_right, symbols.rotate_right]


@pytest.mark.parametrize("action_name", ["rotate_left", "rotate_right"])
def test_revoked_zero_raw_does_not_fall_back_to_percent_turn(action_name):
    runtime, owner, backend, symbols = make_runtime(
        raw=0, source="lateral_intent_revoked"
    )
    action = getattr(symbols, action_name)
    owner.current_command = action

    def allowed(checked_action):
        assert checked_action == action
        assert owner.motor_io_lock.locked()
        return False

    owner._visible_rotate_command_allowed = allowed
    runtime.send_robot_command(action)
    assert backend.targets == []
    assert owner.current_command == action
    assert not owner._brake_hold_active
    assert not hasattr(owner, "_last_motor_dispatch_ts")


@pytest.mark.parametrize(
    "raw, source, expected",
    [(5, "search", (5, -5)), (0, "default", (18, -18))],
)
def test_authorized_search_turn_keeps_raw_and_percent_paths(raw, source, expected):
    runtime, owner, backend, symbols = make_runtime(raw=raw, source=source)

    def allowed(_action):
        assert owner.motor_io_lock.locked()
        return True

    owner._visible_rotate_command_allowed = allowed
    runtime.send_robot_command(symbols.rotate_right)
    assert backend.targets == [(*expected, "TURN")]


def test_legacy_owner_without_visible_guard_keeps_search_turn():
    runtime, _owner, backend, symbols = make_runtime(source="search")
    runtime.send_robot_command(symbols.rotate_left)
    assert backend.targets == [(-5, 5, "TURN")]


def test_authorization_is_rechecked_after_waiting_for_motor_io():
    runtime, owner, backend, symbols = make_runtime()
    attempted_io = threading.Event()
    io_lock = threading.Lock()
    authorization = {"valid": True}
    guard_results = []
    failures = []

    class ContendedIoLock:
        def __enter__(self):
            attempted_io.set()
            io_lock.acquire()

        def __exit__(self, *_args):
            io_lock.release()

    owner.motor_io_lock = ContendedIoLock()

    def allowed(_action):
        assert io_lock.locked()
        guard_results.append(authorization["valid"])
        return authorization["valid"]

    owner._visible_rotate_command_allowed = allowed

    def delayed_turn():
        try:
            runtime.send_robot_command(symbols.rotate_right)
        except BaseException as exc:
            failures.append(exc)

    io_lock.acquire()
    worker = threading.Thread(target=delayed_turn, daemon=True)
    try:
        worker.start()
        assert attempted_io.wait(1.0), "TURN never reached the motor I/O boundary"
        # Simulate zero/TTL revocation while the old packet waits for RS485.
        authorization["valid"] = False
        backend.send_targets(17, 17, "DRIVE")
    finally:
        io_lock.release()
        worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert failures == []
    assert guard_results == [False]
    assert backend.targets == [(17, 17, "DRIVE")]


@pytest.mark.parametrize(
    "action_name, raw_mode, yaw_only",
    [
        ("steer_right", True, False),
        ("steer_left", False, False),
        ("backward", True, False),
        ("backward", False, False),
        ("steer_right", True, True),
        ("steer_left", False, True),
        ("backward", True, True),
        ("backward", False, True),
    ],
)
def test_prepared_yaw_packet_cannot_override_published_zero(
    monkeypatch, action_name, raw_mode, yaw_only
):
    import request_0513_modular as request
    from car_control_modular.lateral_intent import LateralControlIntent

    monkeypatch.setattr(request, "FOLLOW_ROTATION_ONLY", False)
    motor, owner, backend, symbols = make_runtime()
    owner._lateral_yaw_revision = 7
    owner._control_update_lock = threading.RLock()
    owner.search_state = "none"
    owner._vision_control_state = "target_visible_depth_valid"
    owner._explicit_stop_requested = False
    owner._runtime_shutdown_requested = False
    owner.running = True
    owner._follow_controller = SimpleNamespace(active_target_id=1)
    owner._depth_longitudinal_authority_enabled = lambda: True
    owner._should_skip_redundant_action_queue = lambda _actions, _reason: False
    queued = []
    owner._replace_action_queue = lambda actions, reason: queued.append((actions, reason))
    kind = "backward" if action_name == "backward" else "forward"
    owner._depth30_linear_snapshot = (kind, 17, 1, time.monotonic())
    owner.current_command = getattr(symbols, action_name)
    # The real root symbols must match the selected motor action for the
    # helper's queued zero-yaw action to be consumed by this fake executor.
    for name in ("forward", "backward", "rotate_left", "rotate_right", "stop", "steer_left", "steer_right"):
        setattr(symbols, name, getattr(request, "ACTION_" + name.upper()))
    owner.current_command = getattr(symbols, action_name)
    if not raw_mode:
        motor.config.motor_forward_max_target_rpm = 0
        motor.config.motor_steer_raw_target = 0
        owner._current_steer_inner_ratio_percent = 50
    if yaw_only:
        owner._current_forward_percent = 0
        owner._current_steer_base_percent = 0

    stamp = time.monotonic()
    intent = LateralControlIntent(
        sequence=22, target_id=1, frame_index=217,
        published_at=stamp, valid_until=stamp + 0.15,
        x_ratio=0.62, motion_dx_ratio=0.0, target_image_rate_dps=0.0,
        mode="reverse" if kind == "backward" else "forward",
        base_percent=17, base_rpm=17, initial_correction_rpm=5,
        correction_limit_rpm=10.0, confidence=0.9, bbox_quality="reliable",
        reason="test", capture_frame_id=560,
    )
    attempted_io = threading.Event()
    io_lock = threading.Lock()
    failures = []

    class ContendedIoLock:
        def __enter__(self):
            attempted_io.set()
            io_lock.acquire()

        def __exit__(self, *_args):
            io_lock.release()

    owner.motor_io_lock = ContendedIoLock()

    def prepared_packet():
        try:
            motor.send_robot_command(getattr(symbols, action_name))
        except BaseException as exc:
            failures.append(exc)

    io_lock.acquire()
    worker = threading.Thread(target=prepared_packet, daemon=True)
    expected = (-17, -17) if kind == "backward" else (17, 17)
    try:
        worker.start()
        assert attempted_io.wait(1.0), "prepared yaw never reached motor I/O"
        with owner._control_update_lock:
            assert request.PersonTracker._publish_lateral_zero(owner, intent, "test_zero")
        assert owner._lateral_yaw_revision == 8
        assert owner._current_steer_correction_rpm == 0
        assert queued[-1][0] == [
            symbols.backward if kind == "backward" else symbols.steer_right
        ]
        # The old packet was calculated before the real helper published
        # zero. Its captured targets must not replace a newer straight write.
        backend.send_targets(*expected, "DEPTH_ZERO_YAW")
    finally:
        io_lock.release()
        worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert failures == []
    assert backend.targets == [(*expected, "DEPTH_ZERO_YAW")]
    assert not hasattr(owner, "_last_motor_dispatch_ts")

    motor.send_robot_command(queued[-1][0][0])
    assert backend.targets[-1][:2] == expected
    assert not owner._brake_hold_active


def test_revoked_rotation_only_pulse_does_not_leave_an_active_timer():
    motor, owner, backend, _symbols = make_runtime()
    owner._lateral_yaw_revision = 2
    owner._last_motor_dispatch_source = "action_queue"
    motor.config.rotation_only = True
    motor.config.rotation_only_yaw_zero_gap_sec = 0.03
    motor.config.rotation_only_yaw_pulse_min_sec = 0.18
    motor.config.rotation_only_yaw_pulse_max_sec = 0.32
    motor.config.rotation_only_yaw_pulse_rpm = 30
    motor.config.rotate_pulse_settle_feedback_stale_sec = 0.3
    motor.config.rotate_pulse_settle_max_yaw_rate_dps = 3.0

    assert not motor.request_rotation_only_yaw_pulse(5, yaw_revision=1)
    assert backend.targets == []
    assert motor._visible_yaw_pulse_direction == 0
    assert motor._visible_yaw_pulse_deadline_monotonic == 0.0
