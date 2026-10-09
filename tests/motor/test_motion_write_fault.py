"""Offline fault injection: no serial, camera or physical motor access."""
import threading

import pytest

from car_control_modular.action_command import ActionCommandSnapshot
from test_depth_drive_rpm import FakeDriver, make_runtime


class FaultDriver(FakeDriver):
    def __init__(self, fail_write=2, *, zero_fail=(), stop_fail=()):
        super().__init__()
        self.events = []
        self.speed_calls = 0
        self.fail_write = fail_write
        self.zero_fail = set(zero_fail)
        self.stop_fail = set(stop_fail)
        self.closed = False

    def _speed(self, side, value):
        self.events.append(("speed", side, value))
        self.speed_calls += 1
        # An ACK failure does not prove the wheel ignored the write.
        setattr(self, side, value)
        if self.speed_calls == self.fail_write or (value == 0 and side in self.zero_fail):
            raise OSError("injected speed ACK failure")

    def set_right_speed(self, value):
        self._speed("right", value)

    def set_left_speed(self, value):
        self._speed("left", value)

    def stop(self, side, mode):
        self.events.append(("stop", side, int(mode)))
        if side in self.stop_fail:
            raise OSError("injected STOP ACK failure")
        setattr(self, side, 0)

    def stop_all(self, mode=0):
        self.stop("right", mode)
        self.stop("left", mode)

    def close(self):
        self.closed = True


def setup_fault(**kw):
    runtime, owner, _, symbols = make_runtime()
    driver = FaultDriver(**kw)
    runtime.backend.driver = driver
    return runtime, owner, driver, symbols


@pytest.mark.parametrize("fail_write", [1, 2])
@pytest.mark.parametrize("pair", [(7, 7), (-30, 30), (80, -60), (0, 0)])
def test_partial_speed_write_latches_and_attempts_each_zero_then_each_stop(fail_write, pair):
    rt, _, d, _ = setup_fault(fail_write=fail_write)
    with pytest.raises(OSError, match="speed ACK"):
        rt.backend.send_targets(*pair, "fault_injection")
    assert d.events[-4:] == [("speed", "right", 0), ("speed", "left", 0),
                             ("stop", "right", 1), ("stop", "left", 1)]
    assert rt.backend.motion_write_fault.startswith("fault_injection:OSError:")
    assert not rt.backend.motion_armed
    assert d.left == d.right == 0


@pytest.mark.parametrize("zero_fail", [("right",), ("left",), ("right", "left")])
@pytest.mark.parametrize("stop_fail", [("right",), ("left",), ("right", "left")])
def test_cleanup_failure_never_skips_other_wheel_or_original_error(zero_fail, stop_fail):
    rt, _, d, _ = setup_fault(zero_fail=zero_fail, stop_fail=stop_fail)
    with pytest.raises(OSError, match="speed ACK"):
        rt.backend.send_targets(7, 7, "TURN")
    assert d.events[-4:] == [("speed", "right", 0), ("speed", "left", 0),
                             ("stop", "right", 1), ("stop", "left", 1)]
    assert rt.backend.motion_write_fault and not rt.backend.motion_armed
    before = len(d.events)
    with pytest.raises(OSError, match="STOP ACK"):
        rt.backend.send_stop("refresh", mode="normal", prepare_parking_current=True)
    assert d.events[before:] == [("stop", "right", 1), ("stop", "left", 1)]


def test_repeated_motion_never_revives_fault_and_zero_refresh_never_enters_speed_mode():
    rt, _, d, _ = setup_fault()
    with pytest.raises(OSError):
        rt.backend.send_targets(10, -10, "initial")
    fault = rt.backend.motion_write_fault
    before = list(d.events)
    for pair in [(10, -10), (-10, 10), (0, 5)]:
        with pytest.raises(RuntimeError, match="nonzero wheel command blocked"):
            rt.backend.send_targets(*pair, "new_visual_grant")
    for operation in (rt.backend.prepare_speed_mode, rt.backend.release_parking_current_only,
                      lambda: rt.backend.arm_for_motion(d, force=True)):
        with pytest.raises(RuntimeError, match="motor write fault"):
            operation()
    assert d.events == before
    rt.backend.send_targets(0, 0, "stale_zero_keepalive")
    rt.backend.refresh_normal_stop("old_normal_refresh")
    assert d.events[len(before):] == [("stop", "right", 1), ("stop", "left", 1)] * 2
    assert rt.backend.motion_write_fault == fault and not rt.backend.motion_armed
    assert d.register_writes == []


@pytest.mark.parametrize("stop_fail", [(), ("right", "left")])
def test_close_attempts_both_stops_closes_and_cannot_reopen_faulted_backend(stop_fail):
    rt, _, d, _ = setup_fault(stop_fail=stop_fail)
    with pytest.raises(OSError):
        rt.backend.send_targets(8, 8, "initial")
    fault = rt.backend.motion_write_fault
    before = len(d.events)
    rt.backend.close()
    assert d.events[before:] == [("stop", "right", 1), ("stop", "left", 1)]
    assert d.closed and rt.backend.driver is None
    assert rt.backend.motion_write_fault == fault
    with pytest.raises(RuntimeError, match="restart required before reconnect"):
        rt.backend.ensure_driver()


@pytest.mark.parametrize("action", ["rotate_right", "rotate_left", "backward", "forward", "steer_left"])
def test_legacy_catches_still_escalate_backend_fault_to_runtime_shutdown(action):
    rt, owner, d, symbols = setup_fault()
    owner.running = True
    owner.current_command = getattr(symbols, action)
    rt.send_robot_command(owner.current_command)
    assert rt.backend.motion_write_fault
    assert owner._runtime_shutdown_requested and owner._explicit_stop_requested
    assert not owner.running and not owner.is_forwarding
    assert owner.current_command is None and owner._current_forward_percent == 0
    assert owner._brake_hold_label == "safety_hold_motion_write_fault"
    assert owner._brake_hold_stop_mode == "emergency"
    assert not rt.can_release_brake_hold(symbols.forward)
    before = list(d.events)
    rt.send_robot_command(symbols.forward)
    assert d.events == before  # no motion replay or unbounded retry burst


def test_percent_search_branch_has_same_fault_contract():
    rt, owner, _, symbols = setup_fault()
    owner._current_rotate_raw_target = 0
    owner._current_rotate_raw_source = "default"
    rt.send_robot_command(symbols.rotate_right)
    assert rt.backend.motion_write_fault
    assert owner._runtime_shutdown_requested and not owner.running


@pytest.mark.parametrize("reason", ["hard_stop", "follow20_hard_stop", "runtime_fault_capture",
                                     "runtime_shutdown", "runtime_shutdown_signal"])
def test_safety_classification_bypasses_stale_dispatch_and_ordinary_park(reason):
    rt, owner, d, symbols = make_runtime()
    owner._action_command_revision = 2
    rt._dispatch_context.command = ActionCommandSnapshot(
        symbols.forward, 1, 10., 1, 1, 10., "old", False)
    owner._near_yaw_park_request = object()
    rt.send_stop_with_brake_hold(reason)
    assert d.stops == [1]
    assert owner._brake_hold_stop_mode == "emergency"
    assert owner._brake_hold_label == "safety_hold_" + reason
    assert owner._last_command_source_module == "safety_gate"


def test_ordinary_stale_stop_still_cannot_replace_current_command():
    rt, owner, d, symbols = make_runtime()
    owner._action_command_revision = 2
    rt._dispatch_context.command = ActionCommandSnapshot(
        symbols.forward, 1, 10., 1, 1, 10., "old", False)
    rt.send_stop_with_brake_hold("queued_action_stop_signal")
    assert not d.stops and not owner._brake_hold_active


def test_runtime_shutdown_flag_stops_before_any_periodic_or_queue_motion(monkeypatch):
    rt, owner, d, _ = make_runtime()
    owner.action_stop_event = threading.Event()
    owner._runtime_shutdown_requested = True
    monkeypatch.setattr("car_control_modular.action_runtime.time.sleep",
                        lambda _: owner.action_stop_event.set())
    rt._service_follow_wheels = lambda: pytest.fail("motion service after shutdown")
    rt.run_loop()
    assert d.stops == [1] and not d.pairs
    assert owner._brake_hold_label == "safety_hold_runtime_shutdown"


@pytest.mark.parametrize("action", ["rotate_right", "backward", "forward", "steer_left"])
@pytest.mark.parametrize("raw_mode", [True, False])
def test_prepared_legacy_motion_cannot_write_after_shutdown_stop_while_waiting_lock(action, raw_mode):
    rt, owner, d, symbols = make_runtime(raw_mode=raw_mode)
    owner.current_command = getattr(symbols, action)
    if not raw_mode:
        owner._current_rotate_raw_target = 0
        owner._current_rotate_raw_source = "default"

    class ShutdownWinsLock:
        def __enter__(self):
            # Another owner completed STOP before this prepared packet acquired
            # motor exclusion. No actual lock/thread/serial is used here.
            owner._runtime_shutdown_requested = True
            d.stop_all(1)

        def __exit__(self, *args):
            pass

    owner.motor_io_lock = ShutdownWinsLock()
    rt.send_robot_command(owner.current_command)
    assert not d.pairs
    assert d.left == d.right == 0 and d.stops


def test_fault_stop_retry_is_bounded_and_fault_survives_missing_ack(monkeypatch):
    rt, owner, d, _ = setup_fault(stop_fail=("right", "left"))
    with pytest.raises(OSError):
        rt.backend.send_targets(7, 7, "initial")
    clock = [100.]
    monkeypatch.setattr("car_control_modular.action_runtime.time.monotonic", lambda: clock[0])
    assert rt._service_motion_write_fault()
    before = len(d.events)
    assert rt._service_motion_write_fault()
    assert len(d.events) == before
    clock[0] += 1.3
    assert rt._service_motion_write_fault()
    assert d.events[before:] == [("stop", "right", 1), ("stop", "left", 1)]
    assert owner._runtime_shutdown_requested and rt.backend.motion_write_fault


@pytest.mark.parametrize("stop_fail", [("right",), ("left",), ("right", "left")])
@pytest.mark.parametrize("entry", ["shutdown", "hard_stop", "follow20_hard_stop"])
def test_first_safety_stop_failure_attempts_both_and_latches_without_speed_writes(stop_fail, entry):
    rt, owner, d, _ = setup_fault(fail_write=999, stop_fail=stop_fail)
    d.left = d.right = 100
    assert rt.backend.motion_write_fault is None
    if entry == "shutdown":
        owner._runtime_shutdown_requested = True
        assert rt._service_runtime_shutdown()
    else:
        with pytest.raises(OSError, match="STOP ACK"):
            rt.send_stop_with_brake_hold(entry)
    assert d.events == [("stop", "right", 1), ("stop", "left", 1)]
    assert rt.backend.motion_write_fault and not rt.backend.motion_armed
    for side in {"right", "left"} - set(stop_fail):
        assert getattr(d, side) == 0
    before = list(d.events)
    with pytest.raises(RuntimeError, match="nonzero wheel command blocked"):
        rt.backend.send_targets(8, 8, "new_search")
    assert d.events == before


@pytest.mark.parametrize("failed_side", ["right", "left"])
@pytest.mark.parametrize("mode,value", [("normal", 0), ("free", 2)])
def test_failed_normal_or_free_attempts_other_side_then_only_emergency(monkeypatch, failed_side, mode, value):
    rt, _, d, _ = setup_fault(fail_write=999)
    original = d.stop

    def fail_selected_mode(side, stop_value):
        d.stop_fail = {failed_side} if int(stop_value) == value else set()
        original(side, stop_value)

    monkeypatch.setattr(d, "stop", fail_selected_mode)
    with pytest.raises(OSError, match="STOP ACK"):
        rt.backend.send_stop("ordinary_stop", mode=mode, preserve_zero=True)
    assert d.events[-4:] == [("stop", "right", value), ("stop", "left", value),
                             ("stop", "right", 1), ("stop", "left", 1)]
    first_stop = next(i for i, event in enumerate(d.events) if event[0] == "stop")
    assert all(event[0] == "stop" for event in d.events[first_stop:])
    assert rt.backend.motion_write_fault and not rt.backend.normal_zero_hold
    assert not rt.backend.motion_armed and d.left == d.right == 0


@pytest.mark.parametrize("fail_write", [1, 2])
def test_startup_zero_failure_still_attempts_both_wheels_and_both_emergency_stops(fail_write):
    rt, _, d, _ = setup_fault(fail_write=fail_write)
    d.left = d.right = 100  # retained physical-controller targets from an old run
    with pytest.raises(OSError, match="speed ACK"):
        rt.backend.enable_startup_parking()
    initial = [("speed", "right", 0)]
    if fail_write == 2:
        initial.append(("speed", "left", 0))
    assert d.events == initial + [("speed", "right", 0), ("speed", "left", 0),
                                  ("stop", "right", 1), ("stop", "left", 1)]
    assert rt.backend.motion_write_fault.startswith("startup_zero:")
    assert not rt.backend.motion_armed and d.left == d.right == 0
    assert not d.register_writes  # no startup parking-current setup after failure
