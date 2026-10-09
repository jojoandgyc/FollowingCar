"""Forward/steer labels do not require a new serial-locked action adoption."""
import threading
from dataclasses import replace

import pytest

import request_0513_modular as runtime_module
from test_follow_queue_coalescing import following


@pytest.mark.parametrize("old", ["forward", "steer_left", "steer_right"])
@pytest.mark.parametrize("new,yaw", [("forward", 0.), ("steer_left", -6.), ("steer_right", 6.)])
def test_real_producer_label_change_is_nonblocking_and_writer_uses_current_axes(monkeypatch, old, new, yaw):
    rt, owner, driver, symbols, clock, state = following(monkeypatch, old)
    owner._action_runtime = rt
    owner.frame_index = 101
    owner._last_command_capture_frame = 317
    owner._last_action_queue_signature = "old"
    owner._last_action_queue_seq = 7
    owner._last_action_queue_replace_ts = 9.8
    owner._last_command_source_module = "lateral_intent_loop"
    adopted = rt._current_action_snapshot
    state[0], state[1] = 32., yaw
    done, errors = threading.Event(), []
    def publish():
        try:
            runtime_module.PersonTracker._replace_action_queue(owner, [getattr(symbols, new)], "new_axes")
        except Exception as error:
            errors.append(error)
        finally:
            done.set()
    with owner.motor_io_lock:
        thread = threading.Thread(target=publish, daemon=True)
        thread.start()
        no_wait = done.wait(.5)
    thread.join(1.)
    assert no_wait and not errors
    assert rt._current_action_snapshot is adopted
    assert owner.current_command == getattr(symbols, old)
    assert owner._action_command_revision == 7 and owner.action_queue.empty()
    assert owner._last_action_queue_signature == "old"
    assert not driver.pairs and not driver.stops
    clock[0] += .05
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (32+int(yaw), -(32-int(yaw)))
    # Coalescing did not extend either original lease.
    clock[0] = state[2] + .001
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (0, 0)


@pytest.mark.parametrize("stage", ["queued", "popped", "adopted"])
@pytest.mark.parametrize("new", ["steer_left", "steer_right"])
def test_name_change_never_bypasses_stop_barrier(monkeypatch, stage, new):
    rt, owner, _, symbols, _, _ = following(monkeypatch)
    stop = replace(rt._current_action_snapshot, action=symbols.stop,
                   protected_stop=True, revision=8)
    owner._action_command_revision = 8
    if stage == "queued": owner.action_queue.put(stop)
    if stage == "popped": rt._dispatch_context.command = stop
    if stage == "adopted": rt._current_action_snapshot = stop
    assert not rt.can_coalesce_follow_queue_refresh([getattr(symbols, new)])


@pytest.mark.parametrize("kind", ["expired", "zero_base", "uid", "revision", "search", "park"])
def test_new_label_with_changed_authority_uses_existing_serialized_path(monkeypatch, kind):
    rt, owner, _, symbols, clock, state = following(monkeypatch)
    if kind == "expired": clock[0] = state[2]
    if kind == "zero_base": state[0] = 0
    if kind == "uid": owner._follow_controller.active_target_id = 2
    if kind == "revision": owner._action_command_revision += 1
    if kind == "search": owner.search_state = "searching"
    if kind == "park": owner._near_yaw_park_request = object()
    assert not rt.can_coalesce_follow_queue_refresh([symbols.steer_right])
