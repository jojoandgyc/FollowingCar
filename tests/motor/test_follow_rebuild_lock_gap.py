"""Off-lock wheel plans allow encoder/STOP access without losing ownership."""
import pytest

from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


class ObservedLock:
    """Deterministic lock-boundary hooks; no sleeps, motors or worker threads."""
    def __init__(self, lock, *, released=None, reacquiring=None):
        self.lock = lock
        self.depth = 0
        self.entries = 0
        self.released = released
        self.reacquiring = reacquiring
        self.before_acquire = None

    def __enter__(self):
        if self.before_acquire is not None:
            hook, self.before_acquire = self.before_acquire, None
            hook()
        if self.entries == 1 and self.reacquiring:
            assert self.depth == 0
            self.reacquiring()
        self.lock.acquire()
        self.entries += 1
        self.depth += 1
        return self

    def __exit__(self, *args):
        self.depth -= 1
        self.lock.release()
        if self.entries == 1 and self.released:
            assert self.depth == 0
            self.released()


def rebuild_gap(monkeypatch, change=None, *, during_wait=False):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [24., 0., 10.25, 10.25]
    owner._depth30_linear_snapshot = ("forward", 24., 1, 10.)
    owner._depth_linear_max_age_sec = lambda kind: .25
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", state[0], uid, owner._depth30_linear_snapshot[3])
        if clock[0] < state[2] else None)
    runtime._service_follow_wheels()
    assert driver.pairs == [(24, -24)]
    driver.pairs.clear()
    clock[0] = 10.05
    reads = []
    sample = [feedback(clock[0], 20, 20)]
    events = []

    def between():
        assert observed.depth == 0
        events.append("gap")
        if change:
            change(runtime, owner, clock, state, sample)

    observed = ObservedLock(owner.motor_io_lock)
    owner.motor_io_lock = observed
    rebuild = runtime._request_follow_axes_rebuild

    def request_rebuild(uid):
        requested = rebuild(uid)
        if requested:
            if during_wait:
                observed.before_acquire = between
            else:
                between()
        return requested

    runtime._request_follow_axes_rebuild = request_rebuild

    def read_feedback():
        assert observed.depth == 0
        reads.append((observed.entries, sample[0]))
        if len(reads) == 1:
            owner._lateral_yaw_revision += 1
        return sample[0]

    runtime.get_steering_feedback = read_feedback
    return runtime, owner, driver, clock, state, observed, reads, events


def test_rebuild_releases_lock_and_uses_feedback_published_in_gap(monkeypatch, caplog):
    published = []
    def update_feedback(_runtime, _owner, clock, _state, sample):
        clock[0] += .02
        sample[0] = feedback(clock[0], 18, 19)
        published.append(sample[0])
    runtime, _, driver, _, _, lock, reads, events = rebuild_gap(monkeypatch, update_feedback)
    info = runtime.logger.info
    def log(fmt, *args, **kwargs):
        if fmt.startswith(("follow_wheel_tick ", "follow_wheel_lock_timing ")):
            assert lock.depth == 0
        info(fmt, *args, **kwargs)
    monkeypatch.setattr(runtime.logger, "info", log)
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    assert driver.pairs == [(24, -24)]
    assert lock.entries >= 2 and lock.depth == 0 and events == ["gap"]
    assert reads[0][0] < reads[1][0]
    assert reads[1][1] is published[0]
    assert "retry_gap_released=True" in caplog.text
    assert "follow_wheel_lock_timing attempts=2" in caplog.text


@pytest.mark.parametrize("during_wait", [False, True])
@pytest.mark.parametrize("restore_receipt", [False, True])
def test_stop_in_gap_or_reacquire_wait_never_gets_speed_zero(monkeypatch, during_wait, restore_receipt):
    def stop(runtime, _owner, _clock, _state, _sample):
        receipt = runtime.backend.last_speed_receipt
        runtime.backend.send_stop("gap_stop", mode="emergency", preserve_zero=True)
        if restore_receipt:
            # Generation protects STOP even if a buggy adapter reuses a
            # receipt reference; normal backends invalidate it themselves.
            runtime.backend.last_speed_receipt = receipt
    runtime, _, driver, _, _, lock, reads, _ = rebuild_gap(monkeypatch, stop, during_wait=during_wait)
    runtime._service_follow_wheels()
    assert lock.entries >= 2 and lock.depth == 0
    assert driver.stops == [1] and not driver.pairs
    assert len(reads) == 1
    assert not runtime._periodic_follow_writing


def test_intervening_speed_write_owns_rebuild_gap(monkeypatch):
    def write(runtime, *_):
        runtime.backend.send_targets(5, -5, "gap_other_writer")
    runtime, _, driver, _, _, lock, reads, _ = rebuild_gap(monkeypatch, write)
    runtime._service_follow_wheels()
    assert driver.pairs == [(5, -5)]
    assert lock.entries >= 2 and len(reads) == 1


def test_new_grant_in_gap_gets_full_fresh_rebuild(monkeypatch):
    def new_grant(_runtime, owner, clock, state, sample):
        clock[0] = 10.06
        owner._depth30_linear_snapshot = ("forward", 28., 1, clock[0])
        state[0], state[2] = 28., clock[0] + .25
        sample[0] = feedback(clock[0], 20, 20)
    runtime, owner, driver, _, _, lock, reads, _ = rebuild_gap(monkeypatch, new_grant)
    runtime._service_follow_wheels()
    assert driver.pairs == [(28, -28)] and not driver.stops
    assert lock.entries >= 2 and len(reads) == 2
    assert runtime._forward_execution_anchor.sample_timestamp == owner._depth30_linear_snapshot[3]


@pytest.mark.parametrize("change", ["uid", "ttl", "feedback", "explicit_stop"])
def test_rebuild_gap_rechecks_identity_ttl_feedback_and_stop(monkeypatch, change):
    def invalidate(_runtime, owner, clock, state, sample):
        if change == "uid": owner._follow_controller.active_target_id = 2
        elif change == "ttl": clock[0] = 10.251
        elif change == "feedback": sample[0] = feedback(9., 20, 20)
        else: owner._explicit_stop_requested = True
    runtime, _, driver, _, _, lock, _, _ = rebuild_gap(monkeypatch, invalidate)
    runtime._service_follow_wheels()
    assert lock.entries >= 2 and lock.depth == 0
    assert all(pair == (0, 0) for pair in driver.pairs)
    if change == "explicit_stop":
        assert not driver.pairs  # Its own STOP path has priority over speed mode.
    assert not runtime._periodic_follow_writing


def test_lock_timing_records_wait_and_each_held_attempt(monkeypatch, caplog):
    runtime, _, driver, clock, _, lock, _, _ = rebuild_gap(monkeypatch)
    read = runtime.get_steering_feedback
    def costly_read():
        value = read()
        clock[0] += .01
        return value
    runtime.get_steering_feedback = costly_read
    original_enter = lock.reacquiring
    def waiting():
        if original_enter:
            original_enter()
        clock[0] += .007
    lock.reacquiring = waiting
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    assert driver.pairs == [(24, -24)]
    assert "lock_wait_ms=7.00" in caplog.text
    assert "lock_hold_ms=0.00" in caplog.text
    assert "max_hold_ms=0.00" in caplog.text
    assert "planning_outside_motor_lock=True" in caplog.text
