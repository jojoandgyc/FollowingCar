"""An adopted follow refresh need not wait for unrelated serial traffic."""
import queue
import threading
from dataclasses import replace

import pytest

from car_control_modular.action_command import ActionCommandSnapshot
from test_follow_wheel_periodic import setup_periodic


def following(monkeypatch, action_name="forward"):
    runtime, owner, driver, symbols, clock, state = setup_periodic(monkeypatch)
    owner.command_lock = threading.Lock()
    owner.action_queue_lock = threading.Lock()
    owner.action_queue = queue.Queue()
    owner._action_command_revision = 7
    owner._last_command_source_module = "depth30"
    owner.current_command = getattr(symbols, action_name)
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", state[0], uid, 10.) if clock[0] < state[2] else None)
    runtime._current_action_snapshot = ActionCommandSnapshot(
        action=owner.current_command, revision=7, enqueued_at=10., control_frame=1,
        capture_frame_id=1, capture_timestamp=9.9, reason="longitudinal_distance_pid",
        soft_stop=False, source_module="depth30", uid=1)
    runtime._service_follow_wheels()
    assert driver.pairs == [(24, -24)]
    driver.pairs.clear()
    return runtime, owner, driver, symbols, clock, state


@pytest.mark.parametrize("action_name", ["forward", "steer_left", "steer_right"])
def test_same_adopted_kind_can_refresh_without_touching_motor_lock(monkeypatch, action_name):
    runtime, owner, driver, _, _, _ = following(monkeypatch, action_name)
    snapshot = runtime._current_action_snapshot
    axes = runtime._follow_wheel_clock.last_axes
    # A separate producer returns while this thread still owns the motor
    # lock. It cannot pass this assertion by waiting for that lock to release.
    finished = threading.Event()
    results = []

    def producer():
        results.append(runtime.can_coalesce_follow_queue_refresh([owner.current_command]))
        finished.set()

    with owner.motor_io_lock:
        worker = threading.Thread(target=producer, daemon=True)
        worker.start()
        completed_while_busy = finished.wait(.5)
    worker.join(1.)
    assert completed_while_busy and results == [True]
    assert driver.pairs == [] and driver.stops == []
    assert owner._action_command_revision == 7 and owner.action_queue.empty()
    assert runtime._current_action_snapshot is snapshot
    assert runtime._follow_wheel_clock.last_axes is axes


@pytest.mark.parametrize("lock_name", ["command_lock", "action_queue_lock"])
def test_busy_state_lock_declines_without_waiting(monkeypatch, lock_name):
    runtime, owner, _, symbols, _, _ = following(monkeypatch)
    lock = getattr(owner, lock_name)
    with lock:
        assert not runtime.can_coalesce_follow_queue_refresh([symbols.forward])
    assert runtime.can_coalesce_follow_queue_refresh([symbols.forward])


@pytest.mark.parametrize("stage", ["queued", "popped", "adopted", "stale_protected"])
def test_stop_publication_cannot_hide_behind_previous_current_command(monkeypatch, stage):
    runtime, owner, _, symbols, _, _ = following(monkeypatch)
    stop = replace(runtime._current_action_snapshot, action=symbols.stop, revision=8,
                   protected_stop=stage == "stale_protected")
    owner._action_command_revision = 8
    owner.action_queue.put(stop)
    if stage != "queued":
        popped = owner.action_queue.get_nowait()
        runtime._dispatch_context.command = popped
    if stage == "adopted":
        runtime._current_action_snapshot = stop
    if stage == "stale_protected":
        # A protected STOP retained across a publication is still a barrier.
        owner.action_queue.put(stop)
        runtime._current_action_snapshot = replace(runtime._current_action_snapshot, revision=8)
    assert owner.current_command == symbols.forward
    assert not runtime.can_coalesce_follow_queue_refresh([symbols.forward])


@pytest.mark.parametrize("veto", [
    "adopted_pivot", "stop", "reverse", "rotate", "multiple", "no_snapshot", "old_revision",
    "old_uid", "new_uid", "source", "adopted_source", "expired_depth", "old_physical_sample",
    "future_sample", "reverse_depth", "zero_base", "nan_base", "no_prior_write", "prior_other_uid",
    "queued_motion", "search", "controller_search", "low_quality", "explicit_stop", "shutdown",
    "stop_signal", "person_stop", "soft_stop", "brake_hold", "park", "search_brake", "handoff",
    "parking_current", "normal_hold", "uncertain_current", "release_fault", "write_fault",
])
def test_authority_or_lifecycle_change_uses_normal_publication(monkeypatch, veto):
    runtime, owner, _, symbols, clock, state = following(monkeypatch)
    actions = [symbols.forward]
    if veto == "adopted_pivot":
        owner.current_command = symbols.rotate_right
        runtime._current_action_snapshot = replace(runtime._current_action_snapshot, action=symbols.rotate_right)
    elif veto == "stop": actions = [symbols.stop]
    elif veto == "reverse": actions = [symbols.backward]
    elif veto == "rotate": actions = [symbols.rotate_right]
    elif veto == "multiple": actions = [symbols.forward, symbols.steer_right]
    elif veto == "no_snapshot": runtime._current_action_snapshot = None
    elif veto == "old_revision": owner._action_command_revision += 1
    elif veto == "old_uid": runtime._current_action_snapshot = replace(runtime._current_action_snapshot, uid=2)
    elif veto == "new_uid": owner._follow_controller.active_target_id = 2
    elif veto == "source": owner._last_command_source_module = "vision"
    elif veto == "adopted_source": runtime._current_action_snapshot = replace(runtime._current_action_snapshot, source_module="vision")
    elif veto == "expired_depth": clock[0] = state[2]
    elif veto == "old_physical_sample": owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 24, uid, 9.699)
    elif veto == "future_sample": owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 24, uid, 10.001)
    elif veto == "reverse_depth": owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("backward", 24, uid, 10.)
    elif veto == "zero_base": state[0] = 0
    elif veto == "nan_base": state[0] = float("nan")
    elif veto == "no_prior_write": runtime._follow_wheel_clock.reset()
    elif veto == "prior_other_uid": runtime._follow_wheel_clock.last_axes = (2, 1, 24., 0.)
    elif veto == "queued_motion": owner.action_queue.put(runtime._current_action_snapshot)
    elif veto == "search": owner.search_state = "searching"
    elif veto == "controller_search": owner._follow_controller.search_state = "searching"
    elif veto == "low_quality": owner._vision_control_state = "target_visible_low_quality"
    elif veto == "explicit_stop": owner._explicit_stop_requested = True
    elif veto == "shutdown": owner._runtime_shutdown_requested = True
    elif veto == "stop_signal": owner.stop_action_execution = True
    elif veto == "person_stop": owner.person_detected_flag = True
    elif veto == "soft_stop": owner._use_soft_stop_next = True
    elif veto == "brake_hold": owner._brake_hold_active = True
    elif veto == "park": owner._near_yaw_park_request = object()
    elif veto == "search_brake": runtime._search_reacquire_brake_request = object()
    elif veto == "handoff": owner._search_handoff_uid = 1
    elif veto == "parking_current": runtime.backend.parking_current_a = 10.
    elif veto == "normal_hold": runtime.backend.normal_zero_hold = True
    elif veto == "uncertain_current": runtime.backend._parking_current_uncertain = True
    elif veto == "release_fault": runtime.backend.parking_release_fault = "fault"
    elif veto == "write_fault": runtime.backend.motion_write_fault = "fault"
    assert not runtime.can_coalesce_follow_queue_refresh(actions)


def test_changed_canonical_speeds_keep_periodic_writer_authority(monkeypatch):
    runtime, owner, driver, symbols, clock, state = following(monkeypatch)
    owner._last_command_source_module = "lateral_intent_loop"
    state[0], state[1] = 32., -6.
    assert runtime.can_coalesce_follow_queue_refresh([symbols.forward])
    assert driver.pairs == []
    clock[0] += .05
    runtime._service_follow_wheels()
    assert driver.pairs == [(26, -38)]
