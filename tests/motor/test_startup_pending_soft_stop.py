"""CAP24->26: a queued startup zero must not become a latched parking stop."""
from dataclasses import replace
import queue
import threading

import pytest

from test_visible_wheel_continuity import visible_runtime
from car_control_modular.action_command import ActionCommandSnapshot


def pending(runtime, owner, symbols, *, queued=False):
    owner._use_soft_stop_next = True
    owner._soft_stop_active = False
    owner._action_command_revision = 7
    command = ActionCommandSnapshot(
        action=symbols.stop, revision=7, enqueued_at=10., control_frame=3,
        capture_frame_id=24, capture_timestamp=9.95, reason="wait_first_person",
        soft_stop=True,
    )
    runtime._current_action_snapshot = command
    if queued:
        runtime._dispatch_context.command = command
    return command


@pytest.mark.parametrize("queued", [False, True])
def test_pending_soft_interrupt_writes_zero_without_latching_park(monkeypatch, queued):
    r, owner, driver, s, _ = visible_runtime(monkeypatch)
    pending(r, owner, s, queued=queued)
    r.send_stop_with_brake_hold("queued_action_stop_signal" if queued else "stop_signal")
    assert driver.pairs == [(0, 0)] and not driver.stops
    assert owner._soft_stop_active and not owner._use_soft_stop_next
    assert not owner._brake_hold_active


def test_actual_queue_adoption_then_empty_interrupt_before_first_soft_dispatch(monkeypatch):
    r, owner, driver, s, _ = visible_runtime(monkeypatch)
    command = pending(r, owner, s)
    r._current_action_snapshot = None  # It must be adopted by the real loop.
    owner.current_command = None
    owner.stop_action_execution = True
    owner.person_detected_flag = False
    owner.action_stop_event = threading.Event()
    owner.action_queue_lock = threading.Lock()
    owner.command_lock = threading.Lock()
    owner._last_action_queue_seq = 7
    r.config.action_intent_stale_sec = .5
    r.config.enable_motor_rpm_feedback = False
    # Fail the test rather than endlessly retrying an incomplete fake owner.
    monkeypatch.setattr(r.logger, "error", lambda msg, *args: pytest.fail(msg % args))
    r._service_follow_wheels = lambda: None
    r._service_yaw_pulses = lambda: None
    class OneThenEmpty(queue.Queue):
        def get_nowait(self):
            try:
                return super().get_nowait()
            except queue.Empty:
                owner.action_stop_event.set()
                raise
    owner.action_queue = OneThenEmpty()
    owner.action_queue.put(command)
    r.run_loop()
    assert r._current_action_snapshot == command
    assert driver.pairs == [(0, 0)] and not driver.stops
    assert not owner._brake_hold_active
    assert owner.current_command is None and not owner.stop_action_execution


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("pending_flag", [False, True])
def test_obsolete_pending_stop_cannot_write_or_create_hold(monkeypatch, queued, pending_flag):
    r, owner, driver, s, _ = visible_runtime(monkeypatch)
    pending(r, owner, s, queued=queued)
    owner._action_command_revision += 1
    owner._use_soft_stop_next = pending_flag
    r.send_stop_with_brake_hold("queued_action_stop_signal" if queued else "stop_signal")
    assert not driver.pairs and not driver.stops
    assert not owner._brake_hold_active
    assert not owner._soft_stop_active


@pytest.mark.parametrize("case", ["missing", "nonsoft", "protected", "motion", "explicit", "shutdown", "hold"])
def test_pending_flag_is_not_enough_to_soften_other_stops(monkeypatch, case):
    r, owner, driver, s, _ = visible_runtime(monkeypatch)
    command = pending(r, owner, s)
    if case == "missing": r._current_action_snapshot = None
    elif case == "nonsoft": r._current_action_snapshot = replace(command, soft_stop=False)
    elif case == "protected": r._current_action_snapshot = replace(command, protected_stop=True)
    elif case == "motion": r._current_action_snapshot = replace(command, action=s.forward)
    elif case == "explicit": owner._explicit_stop_requested = True
    elif case == "shutdown": owner._runtime_shutdown_requested = True
    elif case == "hold": owner._brake_hold_active = True
    r.send_stop_with_brake_hold("stop_signal")
    assert driver.stops and owner._brake_hold_active
    assert not owner._soft_stop_active


@pytest.mark.parametrize("limit", [0, 8])
@pytest.mark.parametrize("queued", [False, True])
def test_new_hazard_is_not_softened_by_startup_wait(monkeypatch, limit, queued):
    r, owner, driver, s, _ = visible_runtime(monkeypatch)
    pending(r, owner, s, queued=queued)
    r.config.follow_turn_residual_max_rpm = limit
    r.hard_stop_check = lambda _: True
    r.send_stop_with_brake_hold("queued_action_stop_signal" if queued else "stop_signal")
    assert driver.stops and owner._brake_hold_active
    assert owner._brake_hold_label == "safety_hold_hard_stop"
    assert not owner._soft_stop_active


def test_pending_write_exception_does_not_claim_soft_stop_success(monkeypatch):
    r, owner, _, s, _ = visible_runtime(monkeypatch)
    pending(r, owner, s)
    def fail(*args):
        raise RuntimeError("offline write failure")
    r.backend.send_targets = fail
    with pytest.raises(RuntimeError, match="offline write failure"):
        r.send_stop_with_brake_hold("stop_signal")
    assert owner._use_soft_stop_next and not owner._soft_stop_active


def test_protected_queued_stop_cannot_use_old_pending_soft_snapshot(monkeypatch):
    r, owner, driver, s, _ = visible_runtime(monkeypatch)
    command = pending(r, owner, s)
    r._dispatch_context.command = replace(command, soft_stop=False, protected_stop=True)
    r.send_stop_with_brake_hold("queued_action_stop_signal")
    assert driver.stops and owner._brake_hold_active
    assert not owner._soft_stop_active


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("pending_flag", [False, True])
def test_new_revision_while_waiting_for_motor_lock_vetoes_old_pending_zero(monkeypatch, queued, pending_flag):
    r, owner, driver, s, _ = visible_runtime(monkeypatch)
    pending(r, owner, s, queued=queued)
    class ChangedOnLock:
        def __enter__(self):
            owner._action_command_revision += 1
            owner._use_soft_stop_next = pending_flag
        def __exit__(self, *args):
            pass
    owner.motor_io_lock = ChangedOnLock()
    r.send_stop_with_brake_hold("queued_action_stop_signal" if queued else "stop_signal")
    assert not driver.pairs and not driver.stops
    assert not owner._brake_hold_active and not owner._soft_stop_active


@pytest.mark.parametrize("kind", ["search", "near"])
def test_new_parking_request_cannot_be_overwritten_after_lock_wait(monkeypatch, kind):
    r, owner, driver, s, _ = visible_runtime(monkeypatch)
    pending(r, owner, s)
    request = object()
    class RequestOnLock:
        def __enter__(self):
            if kind == "search": r._search_reacquire_brake_request = request
            else: owner._near_yaw_park_request = request
        def __exit__(self, *args):
            pass
    owner.motor_io_lock = RequestOnLock()
    r.send_stop_with_brake_hold("stop_signal")
    assert not driver.pairs and not driver.stops
    assert not owner._soft_stop_active
    assert (r._search_reacquire_brake_request if kind == "search" else owner._near_yaw_park_request) is request


@pytest.mark.parametrize("fault", ["motion_write_fault", "parking_release_fault"])
def test_backend_fault_during_zero_does_not_claim_pending_stop_success(monkeypatch, fault):
    r, owner, _, s, _ = visible_runtime(monkeypatch)
    pending(r, owner, s)
    def faulted(*args):
        setattr(r.backend, fault, "offline fault")
    r.backend.send_targets = faulted
    r.send_stop_with_brake_hold("stop_signal")
    assert owner._use_soft_stop_next and not owner._soft_stop_active
