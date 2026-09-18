"""Offline coverage of approved Depth RPM all the way to a fake serial driver."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.action_runtime import ActionRuntimeSymbols, MotionActionRuntime
from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig


class FakeDriver:
    def __init__(self):
        self.left = self.right = 0
        self.pairs = []
        self.stops = []

    def set_right_speed(self, value):
        self.right = int(value)

    def set_left_speed(self, value):
        self.left = int(value)
        self.pairs.append((self.left, self.right))

    def stop_all(self, mode=0):
        self.left = self.right = 0
        self.stops.append(int(mode))


def make_runtime(*, authorized=True, percent=24, raw_mode=True, max_rpm=100):
    symbols = ActionRuntimeSymbols(
        forward=0, rotate_left=1, rotate_right=2, stop=3,
        steer_left=4, steer_right=5, backward=6,
        movement_actions=frozenset({0, 1, 2, 4, 5, 6}),
        forward_like_actions=frozenset({0, 4, 5}),
        rotate_actions=frozenset({1, 2}),
        action_names={0: "forward", 1: "rotate_left", 2: "rotate_right",
                      3: "stop", 4: "steer_left", 5: "steer_right", 6: "backward"},
        safety_stop_reasons=frozenset({"hard_stop"}),
    )
    backend = MssdMotorBackend(MssdMotorConfig(
        port="unused-offline", slave_id=1, baudrate=115200, timeout=0.1,
        lib_dir="unused-offline", max_target=max_rpm, percent_limit=100,
        left_sign=-1, right_sign=1, forward_target_sign=-1,
        m1_is_left_wheel=True, exit_parking_mode_on_arm=False,
        stop_mode="normal", stop_zero_delay_sec=0.0,
        startup_parking_enabled=False,
    ))
    # Never let this test import the serial client or open a port.
    backend.driver = FakeDriver()
    owner = SimpleNamespace(
        motor_io_lock=backend.io_lock,
        current_command=symbols.forward,
        command_start_time=None,
        _current_forward_percent=percent,
        _current_forward_allow_below_min=authorized,
        _current_steer_base_percent=percent,
        _current_steer_correction_rpm=0,
        _current_steer_inner_ratio_percent=100,
        _current_steer_outer_ratio_percent=100,
        _current_rotate_raw_target=8,
        _current_rotate_raw_source="search",
        _current_rotate_turn_percent=10,
        _current_rotate_pulse_enabled=False,
        _last_control_decision_reason="longitudinal_distance_pid",
        _last_motor_dispatch_source="action_queue",
        _brake_hold_active=False,
    )
    config = SimpleNamespace(
        steering_feedback_median_window=3,
        use_percent_speed=True,
        min_forward_percent=40, max_forward_percent=100,
        motor_forward_raw_target=0,
        motor_forward_max_target_rpm=max_rpm if raw_mode else 0,
        motor_steer_raw_target=15 if raw_mode else 0,
        steer_percent_limit=100,
        visible_steer_inner_ratio_percent=100,
        visible_steer_outer_ratio_percent=100,
        mmwave_hold_forward_percent=20,
        rotate_prep_coast_enable=True, rotate_prep_coast_steps=4,
        rotate_prep_coast_total_sec=0.08,
        motor_forward_like_keepalive_sec=0.5,
        motor_rs485_target_min_interval_sec=0.02,
        motor_rotate_raw_target=8, rotate_turn_percent_from_forward=10,
        rotate_pulse_brake_enable=False,
        rotate_duration=0.18, rotate_pulse_pause_sec=0.0, rotate_hold_stale_sec=0.25,
        rotation_only=False, safety_stop_mode="emergency",
        motor_rs485_stop_mode="normal",
        motor_rs485_transition_stop_mode="emergency",
        motor_rs485_transition_stop_delay_sec=0.0,
        motor_rs485_transition_stop_repeat=1,
    )
    runtime = MotionActionRuntime(
        owner, backend, config, symbols,
        hard_stop_check=lambda _action=None: False,
        logger=logging.getLogger("depth-drive-rpm"),
    )
    return runtime, owner, backend.driver, symbols


@pytest.mark.parametrize("raw_mode", [True, False])
@pytest.mark.parametrize("percent", [1, 15, 20, 24, 29, 32, 39, 40, 75, 100])
def test_approved_depth_forward_reaches_driver_without_launch_floor(raw_mode, percent):
    runtime, _owner, driver, symbols = make_runtime(percent=percent, raw_mode=raw_mode)
    runtime.send_robot_command(symbols.forward)
    assert driver.pairs == [(percent, -percent)]


@pytest.mark.parametrize("max_rpm", [50, 200])
def test_approved_percent_still_uses_configured_forward_rpm_scale(max_rpm):
    runtime, _owner, driver, symbols = make_runtime(percent=24, max_rpm=max_rpm)
    runtime.send_robot_command(symbols.forward)
    expected_rpm = round(max_rpm * 24 / 100)
    assert driver.pairs == [(expected_rpm, -expected_rpm)]


@pytest.mark.parametrize("raw_mode", [True, False])
def test_first_dispatch_and_keepalive_share_the_approved_speed(monkeypatch, raw_mode):
    runtime, owner, driver, symbols = make_runtime(percent=29, raw_mode=raw_mode)
    now = [100.0]
    monkeypatch.setattr("car_control_modular.action_runtime.time.time", lambda: now[0])
    assert runtime.forward_like_refresh_due(symbols.forward, force=True)
    runtime.send_robot_command(symbols.forward)
    now[0] = 100.49
    assert not runtime.forward_like_refresh_due(symbols.forward)
    now[0] = 100.51
    # Lateral diagnostics can replace the reason without changing Depth ownership.
    owner._last_control_decision_reason = "visual_pid_center_hold"
    owner._last_motor_dispatch_source = "keepalive_refresh"
    assert runtime.forward_like_refresh_due(symbols.forward)
    runtime.send_robot_command(symbols.forward)
    assert driver.pairs == [(29, -29), (29, -29)]


@pytest.mark.parametrize("raw_mode", [True, False])
def test_depth_slowdown_keeps_each_approved_step_and_zero(raw_mode):
    runtime, owner, driver, symbols = make_runtime(raw_mode=raw_mode)
    for percent in (32, 24, 15, 0):
        owner._current_forward_percent = percent
        runtime.send_robot_command(symbols.forward)
    assert driver.pairs == [(32, -32), (24, -24), (15, -15), (0, 0)]


@pytest.mark.parametrize("raw_mode", [True, False])
@pytest.mark.parametrize("reason", ["manual", "longitudinal_distance_pid", "visual_pid_center_hold"])
def test_legacy_forward_retains_launch_floor_without_explicit_authorization(raw_mode, reason):
    runtime, owner, driver, symbols = make_runtime(authorized=False, raw_mode=raw_mode)
    del owner._current_forward_allow_below_min
    owner._last_control_decision_reason = reason
    runtime.send_robot_command(symbols.forward)
    assert driver.pairs == [(40, -40)]


@pytest.mark.parametrize("reason", [
    "visual_pid_center_camera_distance_missing", "visual_pid_center_mmwave_hold",
])
def test_existing_safety_low_speed_exceptions_remain(reason):
    runtime, owner, driver, symbols = make_runtime(authorized=False, percent=15)
    owner._last_control_decision_reason = reason
    runtime.send_robot_command(symbols.forward)
    assert driver.pairs == [(15, -15)]


def test_direct_legacy_drive_does_not_implicitly_inherit_owner_authorization():
    runtime, _owner, driver, _symbols = make_runtime()
    runtime.send_percent_drive(24)
    runtime.send_percent_drive(24, allow_below_min=True)
    assert driver.pairs == [(40, -40), (24, -24)]


@pytest.mark.parametrize("raw_mode", [True, False])
@pytest.mark.parametrize("start_action", ["forward", "steer_left", "steer_right"])
def test_coast_preserves_dispatched_authorization_after_producer_revocation(
    monkeypatch, raw_mode, start_action,
):
    runtime, owner, driver, symbols = make_runtime(raw_mode=raw_mode)
    owner.current_command = getattr(symbols, start_action)
    runtime.send_robot_command(owner.current_command)
    assert driver.pairs == [(24, -24)]
    owner._current_forward_allow_below_min = False
    owner._last_control_decision_reason = "search_after_target_loss"
    owner._current_forward_percent = 99  # A new decision must not raise the old ramp.
    monkeypatch.setattr("car_control_modular.action_runtime.time.sleep", lambda _seconds: None)
    runtime.send_motion_transition_stop(owner.current_command, symbols.rotate_right)
    driver.pairs.clear()
    runtime.coast_down_before_rotate()
    assert driver.pairs == [(18, -18), (12, -12), (6, -6), (0, 0)]


def test_legacy_coast_retains_its_original_minimum(monkeypatch):
    runtime, _owner, driver, symbols = make_runtime(authorized=False)
    runtime.send_robot_command(symbols.forward)
    driver.pairs.clear()
    monkeypatch.setattr("car_control_modular.action_runtime.time.sleep", lambda _seconds: None)
    runtime.coast_down_before_rotate()
    assert driver.pairs == [(40, -40), (40, -40), (40, -40), (0, 0)]


def test_coast_snapshot_cannot_authorize_a_new_forward_keepalive():
    runtime, owner, driver, symbols = make_runtime()
    runtime.send_robot_command(symbols.forward)
    owner._current_forward_allow_below_min = False
    owner._last_motor_dispatch_source = "keepalive_refresh"
    runtime.send_robot_command(symbols.forward)
    assert driver.pairs == [(24, -24), (40, -40)]


@pytest.mark.parametrize("raw_mode", [True, False])
@pytest.mark.parametrize("stop_kind", ["hard", "soft", "requested", "command_changed"])
def test_new_stop_during_coast_cannot_be_followed_by_positive_rpm(
    monkeypatch, raw_mode, stop_kind,
):
    runtime, owner, driver, symbols = make_runtime(raw_mode=raw_mode)
    runtime.send_robot_command(symbols.forward)
    driver.pairs.clear()
    sleeps = []

    def stop_after_first_step(_seconds):
        sleeps.append(True)
        if stop_kind == "hard":
            runtime.send_stop_with_brake_hold("hard_stop")
        elif stop_kind == "soft":
            owner._use_soft_stop_next = True
            runtime.send_robot_command(symbols.stop)
        elif stop_kind == "requested":
            owner._explicit_stop_requested = True
        else:
            owner.current_command = symbols.stop

    monkeypatch.setattr("car_control_modular.action_runtime.time.sleep", stop_after_first_step)
    runtime.coast_down_before_rotate()
    assert sleeps == [True]
    assert driver.pairs[0] == (18, -18)
    assert all(pair == (0, 0) for pair in driver.pairs[1:])
    if stop_kind == "hard":
        assert driver.stops == [1]


@pytest.mark.parametrize("raw_mode", [True, False])
def test_coast_checks_a_new_safety_stop_after_waiting_for_motor_lock(monkeypatch, raw_mode):
    runtime, owner, driver, symbols = make_runtime(raw_mode=raw_mode)
    runtime.send_robot_command(symbols.forward)
    driver.pairs.clear()

    class StopBeforeLock:
        def __enter__(self):
            # Deterministically model safety taking ownership while this
            # prepared coast packet is waiting for the shared serial lock.
            owner._brake_hold_active = True
            runtime.backend.io_lock.acquire()

        def __exit__(self, *_args):
            runtime.backend.io_lock.release()

    owner.motor_io_lock = StopBeforeLock()
    monkeypatch.setattr("car_control_modular.action_runtime.time.sleep", lambda _seconds: None)
    runtime.coast_down_before_rotate()
    assert driver.pairs == []


@pytest.mark.parametrize("authorized", [False, True])
def test_zero_reverse_search_and_hard_stop_mappings_do_not_depend_on_depth_flag(authorized):
    runtime, owner, driver, symbols = make_runtime(authorized=authorized)
    owner._current_forward_percent = 0
    runtime.send_robot_command(symbols.forward)
    owner._current_forward_percent = 24
    runtime.send_robot_command(symbols.backward)
    runtime.send_robot_command(symbols.rotate_right)
    assert driver.pairs == [(0, 0), (-24, 24), (8, 8)]
    runtime.send_stop_with_brake_hold("hard_stop")
    assert driver.stops == [1]
    assert (driver.left, driver.right) == (0, 0)


@pytest.mark.parametrize("ending", ["stop", "backward", "rotate_right", "hard_stop"])
def test_finished_forward_authorization_cannot_leak_into_a_later_coast(ending):
    runtime, _owner, _driver, symbols = make_runtime()
    runtime.send_robot_command(symbols.forward)
    assert runtime._forward_coast_snapshot == (24, True)
    if ending == "hard_stop":
        runtime.send_stop_with_brake_hold("hard_stop")
    else:
        runtime.send_robot_command(getattr(symbols, ending))
    assert runtime._forward_coast_snapshot is None


def test_fixed_raw_legacy_configuration_keeps_its_existing_mapping():
    runtime, _owner, driver, symbols = make_runtime()
    runtime.config.motor_forward_raw_target = 12
    runtime.send_robot_command(symbols.forward)
    assert driver.pairs == [(12, -12)]


@pytest.mark.parametrize("raw_mode", [True, False])
@pytest.mark.parametrize("dispatch_source", ["action_queue", "keepalive_refresh"])
@pytest.mark.parametrize("action_name", ["forward", "steer_left", "steer_right"])
def test_prepared_forward_cannot_replay_after_new_zero_revision(
    monkeypatch, raw_mode, dispatch_source, action_name,
):
    runtime, owner, driver, symbols = make_runtime(raw_mode=raw_mode)
    action = getattr(symbols, action_name)
    owner.current_command = action
    owner._lateral_yaw_revision = 7
    owner._last_motor_dispatch_source = dispatch_source
    prepared = threading.Event()
    resume = threading.Event()
    errors = []

    class PausePreparedDrive:
        def __enter__(self):
            if threading.current_thread().name == "prepared-old-drive":
                prepared.set()
                assert resume.wait(2.0), "test did not release prepared forward packet"
            runtime.backend.io_lock.acquire()

        def __exit__(self, *_args):
            runtime.backend.io_lock.release()

    owner.motor_io_lock = PausePreparedDrive()
    original_guard = runtime._yaw_revision_write_allowed
    checked_revisions = []

    def guarded(revision, label):
        assert runtime.backend.io_lock.locked()
        checked_revisions.append((revision, label))
        return original_guard(revision, label)

    monkeypatch.setattr(runtime, "_yaw_revision_write_allowed", guarded)

    def old_drive():
        try:
            runtime.send_robot_command(action)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=old_drive, name="prepared-old-drive", daemon=True)
    try:
        worker.start()
        assert prepared.wait(1.0), "old forward packet did not reach the motor lock"
        owner._current_forward_percent = 0
        owner._current_steer_base_percent = 0
        owner._current_forward_allow_below_min = False
        owner._lateral_yaw_revision = 8
        assert runtime.send_percent_drive(0)
    finally:
        resume.set()
        worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert errors == []
    assert driver.pairs == [(0, 0)]
    assert checked_revisions == [(7, "DRIVE" if action_name == "forward" else "STEER")]
    assert runtime._forward_coast_snapshot is None
    assert not hasattr(owner, "_last_motor_dispatch_ts")
    assert not hasattr(owner, "_last_motor_dispatch_action")

    # The replacement revision remains usable by a genuinely new command.
    owner._current_forward_percent = 12
    owner._current_steer_base_percent = 12
    owner._current_forward_allow_below_min = True
    runtime.send_robot_command(action)
    assert driver.pairs == [(0, 0), (12, -12)]
    assert runtime._forward_coast_snapshot == (12, True)


@pytest.mark.parametrize("raw_mode", [True, False])
def test_direct_drive_default_stays_legacy_but_explicit_stale_revision_is_rejected(raw_mode):
    runtime, owner, driver, _symbols = make_runtime(raw_mode=raw_mode)
    owner._lateral_yaw_revision = 8
    assert runtime.send_percent_drive(24)
    assert driver.pairs == [(40, -40)]
    snapshot = runtime._forward_coast_snapshot
    assert not runtime.send_percent_drive(24, allow_below_min=True, yaw_revision=7)
    assert driver.pairs == [(40, -40)]
    assert runtime._forward_coast_snapshot == snapshot
    # Withdrawing motion must not be blocked by a stale positive authorization.
    assert runtime.send_percent_drive(0, yaw_revision=7)
    assert driver.pairs == [(40, -40), (0, 0)]


@pytest.mark.parametrize("raw_mode", [True, False])
def test_zero_steer_stays_allowed_when_revision_changes_before_motor_write(raw_mode):
    runtime, owner, driver, symbols = make_runtime(percent=0, raw_mode=raw_mode)
    owner._lateral_yaw_revision = 7

    class AdvanceRevisionBeforeLock:
        def __enter__(self):
            owner._lateral_yaw_revision = 8
            runtime.backend.io_lock.acquire()

        def __exit__(self, *_args):
            runtime.backend.io_lock.release()

    owner.motor_io_lock = AdvanceRevisionBeforeLock()
    runtime.send_robot_command(symbols.steer_right)
    assert driver.pairs == [(0, 0)]
    assert owner._last_motor_dispatch_action == symbols.steer_right
    assert runtime._forward_coast_snapshot is None


def test_percent_steer_zero_is_not_blocked_by_an_explicit_stale_revision():
    runtime, owner, driver, _symbols = make_runtime(raw_mode=False)
    owner._lateral_yaw_revision = 8
    assert runtime.send_percent_diff(0, 0x01, 0, 0x01, "STEER", yaw_revision=7)
    assert driver.pairs == [(0, 0)]
