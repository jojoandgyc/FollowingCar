"""Real producer integration, with fake motor state and no device I/O."""
import threading

import pytest

import request_0513_modular as runtime
from test_follow_queue_coalescing import following


@pytest.mark.parametrize("action_name", ["forward", "steer_left", "steer_right"])
def test_real_queue_producer_refreshes_only_heartbeat_while_serial_busy(monkeypatch, action_name):
    executor, owner, driver, _, clock, state = following(monkeypatch, action_name)
    owner._action_runtime = executor
    owner.frame_index = 350
    owner._last_command_capture_frame = 351
    owner._last_action_intent_ts = 9.8
    owner._last_action_queue_signature = "unchanged-signature"
    owner._last_action_queue_seq = 7
    owner._last_dispatched_action = owner.current_command
    owner._last_action_queue_replace_ts = 9.8
    snapshot = executor._current_action_snapshot
    done, errors = threading.Event(), []

    def publish():
        try:
            runtime.PersonTracker._replace_action_queue(owner, [owner.current_command], "fresh_depth")
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    with owner.motor_io_lock:
        thread = threading.Thread(target=publish, daemon=True)
        thread.start()
        no_wait = done.wait(.5)
    thread.join(1.)
    assert no_wait and not errors
    assert owner._last_action_intent_ts == clock[0]
    assert owner._last_action_intent_frame == 350
    assert owner._last_action_intent_reason == "fresh_depth"
    assert owner._last_action_queue_signature == "unchanged-signature"
    assert owner._last_action_queue_seq == owner._action_command_revision == 7
    assert owner._last_action_queue_replace_ts == 9.8
    assert owner.action_queue.empty() and executor._current_action_snapshot is snapshot
    assert not driver.pairs and not driver.stops
    # Queue coalescing cannot extend either lease. Expiry still sends zero.
    clock[0] = state[2] + .01
    executor._service_follow_wheels()
    assert driver.pairs[-1] == (0, 0)


@pytest.mark.parametrize("change", ["stop", "expired", "pivot", "popped_stop"])
def test_real_queue_producer_keeps_serialized_path_for_barriers(monkeypatch, change):
    executor, owner, _, symbols, clock, state = following(monkeypatch)
    owner._action_runtime = executor
    owner.frame_index = 1
    owner._actions_signature = lambda actions: "test"
    actions = [symbols.forward]
    if change == "stop":
        actions = [symbols.stop]
    elif change == "expired":
        clock[0] = state[2]
    elif change == "pivot":
        actions = [symbols.rotate_right]
    else:
        owner._action_command_revision += 1

    class PublicationReached(Exception):
        pass

    class SerialProbe:
        def __enter__(self):
            raise PublicationReached("normal serialized publication retained")

        def __exit__(self, *args):
            pass

    owner.motor_io_lock = SerialProbe()
    with pytest.raises(PublicationReached):
        runtime.PersonTracker._replace_action_queue(owner, actions, "new_intent")
