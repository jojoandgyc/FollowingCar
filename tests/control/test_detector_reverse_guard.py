"""Real reverse writer/service with a recording backend; no device access."""
from contextlib import contextmanager
import threading
from types import SimpleNamespace

import pytest

from car_control_modular.action_runtime import MotionActionRuntime
from car_control_modular.detector_identity_lease import DetectorIdentityLease


def runtime(*, lease=None, raw=True):
    owner = SimpleNamespace(
        motor_io_lock=threading.RLock(), command_lock=threading.RLock(),
        _current_steer_correction_rpm=0, _detector_identity_lease=lease,
        current_command=2, command_start_time=99.,
        _follow_controller=SimpleNamespace(active_target_id=1))
    sent = []
    backend = SimpleNamespace(
        clip_percent=lambda p: max(0, min(100, p)),
        config=SimpleNamespace(max_target=100, m1_is_left_wheel=True),
        wheel_raw_state_to_target=lambda wheel, value, state: -value if wheel == "left" else value,
        send_targets=lambda l, r, label, **kw: sent.append((l, r, label)),
        send_diff=lambda *args: sent.append(args))
    result = object.__new__(MotionActionRuntime)
    result.owner, result.backend = owner, backend
    result.config = SimpleNamespace(motor_forward_raw_target=0,
                                    motor_forward_max_target_rpm=100 if raw else 0)
    result.symbols = SimpleNamespace(backward=2)
    result._near_yaw_park_blocks_write = lambda label: False
    result.logger = SimpleNamespace(info=lambda *args, **kw: None)
    return result, sent


@pytest.mark.parametrize("lease", [
    False,
    DetectorIdentityLease(1, 1, 2, 99.7, 4, 99.9, 100.05),
    DetectorIdentityLease(1, 1, 2, 10., 4, 10.1, 10.35),
])
@pytest.mark.parametrize("raw", [True, False])
def test_live_expired_or_rejected_fast_identity_cannot_dispatch_reverse(lease, raw):
    action, sent = runtime(lease=lease, raw=raw)
    assert action.send_percent_backward(20) is None
    assert sent == [(0, 0, "REVERSE_DETECTOR_IDENTITY_GUARD")]


@pytest.mark.parametrize("raw", [True, False])
def test_full_only_reverse_preserves_existing_behavior(raw):
    action, sent = runtime(raw=raw)
    assert action.send_percent_backward(20) == "REVERSE"
    assert sent == ([(-20, 20, "REVERSE")] if raw else [(20, 2, 20, 2, "REVERSE")])


def test_execution_tick_withdraws_existing_reverse_once_without_keepalive():
    action, sent = runtime(lease=False)
    assert action._service_detector_reverse_guard()
    assert action.owner.current_command is None
    assert action.owner.command_start_time is None
    assert sent == [(0, 0, "REVERSE_DETECTOR_IDENTITY_GUARD")]
    assert not action._service_detector_reverse_guard()
    assert len(sent) == 1


@pytest.mark.parametrize("raw", [True, False])
def test_final_lock_check_blocks_lease_created_after_packet_preparation(raw):
    action, sent = runtime(raw=raw)
    @contextmanager
    def enter():
        action.owner._detector_identity_lease = False
        yield
    action.owner.motor_io_lock = enter()
    assert action.send_percent_backward(20) is None
    assert sent == [(0, 0, "REVERSE_DETECTOR_IDENTITY_GUARD")]


def test_full_result_arriving_during_tick_lock_wait_is_not_zeroed():
    action, sent = runtime(lease=False)
    @contextmanager
    def enter():
        action.owner._detector_identity_lease = None
        yield
    action.owner.motor_io_lock = enter()
    assert not action._service_detector_reverse_guard()
    assert action.owner.current_command == 2
    assert not sent


def test_new_command_arriving_during_tick_lock_wait_is_not_retired():
    action, sent = runtime(lease=False)
    @contextmanager
    def enter():
        action.owner.current_command = 1
        yield
    action.owner.command_lock = enter()
    assert not action._service_detector_reverse_guard()
    assert action.owner.current_command == 1 and not sent


@pytest.mark.parametrize("owner_flag,backend_flag", [
    ("_explicit_stop_requested", None), ("_runtime_shutdown_requested", None),
    ("_brake_hold_active", None), (None, "normal_zero_hold"),
    (None, "parking_current_a"), (None, "_parking_current_uncertain"),
])
def test_reverse_guard_does_not_replace_stop_or_parking_with_speed_zero(owner_flag, backend_flag):
    action, sent = runtime(lease=False)
    if owner_flag:
        setattr(action.owner, owner_flag, True)
    if backend_flag:
        setattr(action.backend, backend_flag, True)
    assert action._service_detector_reverse_guard()
    assert action.owner.current_command is None and not sent


def test_full_mode_and_nonreverse_tick_are_noops():
    action, sent = runtime()
    assert not action._service_detector_reverse_guard()
    action.owner._detector_identity_lease = False
    action.owner.current_command = 1
    assert not action._service_detector_reverse_guard()
    assert not sent
