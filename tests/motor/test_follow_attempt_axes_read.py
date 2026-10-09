"""One planning axes admission per off-I/O-lock attempt; fake serial/time."""
import pytest

from test_follow_wheel_periodic import setup_periodic
from test_follow_rebuild_lock_gap import ObservedLock
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("rebuild", [False, True])
def test_each_attempt_reads_axes_once_before_entering_wheel_writer(monkeypatch, rebuild):
    runtime, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    lock = ObservedLock(owner.motor_io_lock)
    owner.motor_io_lock = lock
    read_axes = owner._follow_wheel_axes
    send = runtime._send_follow_wheel_targets
    entered_writer = set()
    planning_reads = {}
    reads_seen_at_send = []

    def axes(now):
        token = runtime._follow_plan_token
        if token is not None and id(token) not in entered_writer:
            assert lock.depth == 0
            planning_reads[id(token)] = planning_reads.get(id(token), 0)+1
        return read_axes(now)

    def write(*args, **kwargs):
        assert lock.depth == 0
        key = id(runtime._follow_plan_token)
        reads_seen_at_send.append(planning_reads[key])
        entered_writer.add(key)
        return send(*args, **kwargs)

    feedback_reads = [0]

    def read_feedback():
        feedback_reads[0] += 1
        if rebuild and feedback_reads[0] == 1:
            owner._lateral_yaw_revision += 1
        return feedback(clock[0], 20, 20)

    owner._follow_wheel_axes = axes
    runtime._send_follow_wheel_targets = write
    runtime.get_steering_feedback = read_feedback
    runtime._service_follow_wheels()

    assert reads_seen_at_send == ([1, 1] if rebuild else [1])
    assert driver.pairs == [(24, -24)]
    assert not driver.stops


@pytest.mark.parametrize("change", ["stop", "uid"])
def test_waiting_for_first_attempt_does_not_reuse_old_owner(monkeypatch, change):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    owner._follow_wheel_axes = lambda now: (
        owner._follow_controller.active_target_id, owner._lateral_yaw_revision,
        state[0], state[1])
    runtime._service_follow_wheels()
    assert driver.pairs == [(24, -24)]
    driver.pairs.clear()
    clock[0] += .05

    class WaitLock(ObservedLock):
        def __enter__(self):
            if self.entries == 0:
                assert self.depth == 0
                if change == "stop":
                    runtime.backend.send_stop("stop_during_wait", mode="emergency", preserve_zero=True)
                    owner._explicit_stop_requested = True
                else:
                    owner._follow_controller.active_target_id = 2
            return super().__enter__()

    owner.motor_io_lock = WaitLock(owner.motor_io_lock)
    runtime._service_follow_wheels()
    if change == "stop":
        assert driver.stops == [1]
        assert not driver.pairs  # No speed-zero write may release the STOP.
    else:
        assert driver.pairs == [(0, 0)]
        assert runtime._follow_wheel_clock.last_axes is None


def test_public_active_predicate_still_requires_full_live_axes(monkeypatch):
    runtime, owner, _, _, _, _ = setup_periodic(monkeypatch)
    reads = []
    owner._follow_wheel_axes = lambda now: reads.append(now)
    assert not runtime._periodic_follow_active()
    assert len(reads) == 1
    owner._explicit_stop_requested = True
    assert not runtime._periodic_follow_active()
    assert len(reads) == 1
