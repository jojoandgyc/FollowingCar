"""A zero ACK is not an action-mode transition or a new STOP barrier."""
import threading
from dataclasses import replace

import pytest

import request_0513_modular as main
from test_follow_queue_coalescing import following


def expired_then_fresh(monkeypatch):
    rt, owner, driver, symbols, clock, state = following(monkeypatch)
    clock[0] = 10.181
    rt._service_follow_wheels()
    assert driver.pairs == [(0, 0)]
    assert rt._follow_wheel_clock.last_axes[2] == 0
    clock[0] = 10.20
    state[:] = [32., 0., 10.49, 10.49]
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", state[0], uid, 10.19) if clock[0] < state[2] else None)
    owner._action_runtime = rt
    owner.frame_index = 171
    owner._last_command_capture_frame = 171
    owner._actions_signature = lambda actions: tuple(actions)
    return rt, owner, driver, symbols, clock, state


@pytest.mark.parametrize("action,yaw", [("forward", 0.), ("steer_left", -6.), ("steer_right", 6.)])
def test_fresh_recovery_publication_does_not_wait_for_serial_after_zero(monkeypatch, action, yaw):
    rt, owner, driver, symbols, clock, state = expired_then_fresh(monkeypatch)
    state[1] = yaw
    snapshot = rt._current_action_snapshot
    done, errors = threading.Event(), []

    def publish():
        try:
            main.PersonTracker._replace_action_queue(owner, [getattr(symbols, action)], "fresh_recovery")
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
    assert rt._current_action_snapshot is snapshot
    assert owner._action_command_revision == 7 and owner.action_queue.empty()
    assert driver.pairs == [(0, 0)] and not driver.stops
    clock[0] += .05
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (32+int(yaw), -(32-int(yaw)))
    clock[0] = state[2]+.001
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (0, 0)  # Coalescing renews neither lease.


@pytest.mark.parametrize("change", ["stop", "popped_stop", "uid", "reverse", "unstarted", "park", "expired"])
def test_recovery_refresh_cannot_remove_barriers(monkeypatch, change):
    rt, owner, _, symbols, clock, state = expired_then_fresh(monkeypatch)
    if change == "stop":
        owner._explicit_stop_requested = True
    elif change == "popped_stop":
        owner._action_command_revision += 1
        rt._dispatch_context.command = replace(rt._current_action_snapshot,
            action=symbols.stop, revision=8, protected_stop=True)
    elif change == "uid":
        owner._follow_controller.active_target_id = 2
    elif change == "reverse":
        rt._follow_wheel_clock.last_axes = (1, 1, -10., 0.)
    elif change == "unstarted":
        rt._follow_wheel_clock.reset()
    elif change == "park":
        owner._brake_hold_active = True
    elif change == "expired":
        clock[0] = state[2]+.001
    assert not rt.can_coalesce_follow_queue_refresh([symbols.forward])
