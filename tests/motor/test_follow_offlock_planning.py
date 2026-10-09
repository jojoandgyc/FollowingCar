"""Real-thread serial contention tests with fake driver, no hardware."""
import threading

import pytest

from test_follow_deferred_diagnostics import live_writer
from test_visible_wheel_continuity import feedback


def paused_plan(monkeypatch, *, phase="guard"):
    runtime, owner, driver, clock, _ = live_writer(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    errors = []
    if phase == "guard":
        original = runtime._visible_wheel_guard.limit

        def pause(*args, **kwargs):
            assert not owner.motor_io_lock.locked()
            entered.set()
            assert release.wait(2.), "test did not release wheel planning"
            result = original(*args, **kwargs)
            tail = getattr(runtime, "_offline_guard_tail", None)
            if tail:
                tail()
            return result

        monkeypatch.setattr(runtime._visible_wheel_guard, "limit", pause)
    else:
        original = runtime._begin_follow_commit

        def pause():
            assert not owner.motor_io_lock.locked()
            entered.set()
            assert release.wait(2.), "test did not release commit admission"
            return original()

        monkeypatch.setattr(runtime, "_begin_follow_commit", pause)

    def run():
        try:
            runtime._service_follow_wheels()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert entered.wait(1.), errors
    return runtime, owner, driver, clock, thread, release, errors


def finish(thread, release, errors):
    release.set()
    thread.join(2.)
    assert not thread.is_alive(), "FOLLOW worker did not exit"
    assert not errors


class CommitWaitLock:
    """Expose an actual blocked acquire without sleeps or scheduler guesses."""
    def __init__(self, lock):
        self.lock = lock
        self.waiting = threading.Event()

    def acquire(self, *args, **kwargs):
        return self.lock.acquire(*args, **kwargs)

    def release(self):
        self.lock.release()

    def locked(self):
        return self.lock.locked()

    def __enter__(self):
        if self.lock.locked():
            self.waiting.set()
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


def hold_until_commit_waits(owner, release):
    probe = CommitWaitLock(owner.motor_io_lock)
    owner.motor_io_lock = probe
    assert probe.acquire(timeout=.2)
    release.set()
    assert probe.waiting.wait(1.), "planner never waited for actual serial commit"
    return probe


@pytest.mark.parametrize("phase", ["guard", "commit"])
@pytest.mark.parametrize("intervening", ["stop", "zero", "speed"])
def test_cpu_plan_does_not_exclude_encoder_or_overwrite_new_motor_owner(
        monkeypatch, phase, intervening):
    runtime, owner, driver, clock, worker, release, errors = paused_plan(
        monkeypatch, phase=phase)
    try:
        assert owner.motor_io_lock.acquire(timeout=.2), "planning monopolized serial lock"
        try:
            clock[0] = 10.02
            runtime._steering_feedback = feedback(clock[0], 20, 20)
            if intervening == "stop":
                runtime.backend.send_stop("concurrent-safety", mode="emergency", preserve_zero=True)
            elif intervening == "zero":
                runtime.backend.send_targets(0, 0, "concurrent-zero")
            else:
                runtime.backend.send_targets(7, -7, "concurrent-command")
            expected_pairs, expected_stops = list(driver.pairs), list(driver.stops)
        finally:
            owner.motor_io_lock.release()
    finally:
        finish(worker, release, errors)
    assert driver.pairs == expected_pairs
    assert driver.stops == expected_stops


@pytest.mark.parametrize("opposed", [False, True])
def test_encoder_publication_during_planning_is_rechecked_at_commit(monkeypatch, opposed):
    runtime, owner, driver, clock, worker, release, errors = paused_plan(monkeypatch)
    committed = []
    final_limit = runtime._linear_packet_write_limit

    def inspect(*args, **kwargs):
        assert owner.motor_io_lock.locked()
        committed.append(kwargs["feedback"])
        return final_limit(*args, **kwargs)

    monkeypatch.setattr(runtime, "_linear_packet_write_limit", inspect)
    try:
        assert owner.motor_io_lock.acquire(timeout=.2)
        try:
            clock[0] = 10.02
            sample = feedback(clock[0], -10 if opposed else 20, 20)
            runtime._steering_feedback = sample
        finally:
            owner.motor_io_lock.release()
    finally:
        finish(worker, release, errors)
    if opposed:
        assert driver.pairs == [(0, 0)]
        assert not committed
    else:
        assert driver.pairs == [(24, -24)]
        assert committed == [sample]


@pytest.mark.parametrize("change", ["grant", "uid", "identity", "depth_expiry", "feedback_expiry"])
def test_current_authority_is_rechecked_after_commit_lock_wait(monkeypatch, change):
    runtime, owner, driver, clock, worker, release, errors = paused_plan(
        monkeypatch, phase="commit")
    # Hold the actual serial lock, let the planner reach acquisition, publish
    # while it is waiting, then release. No artificial callback grants motion.
    try:
        held = hold_until_commit_waits(owner, release)
        try:
            if change == "grant":
                owner._depth30_linear_snapshot = ("forward", 24., 1, 10.01)
            elif change == "uid":
                owner._follow_controller.active_target_id = 2
            elif change == "identity":
                owner._vision_control_state = "target_lost"
            elif change == "depth_expiry":
                clock[0] = 10.251
                runtime._steering_feedback = feedback(clock[0], 20, 20)
            else:
                clock[0] = 10.151
        finally:
            held.release()
    finally:
        finish(worker, release, errors)
    assert all(pair == (0, 0) for pair in driver.pairs)


def test_second_periodic_service_does_not_interleave_shared_wheel_plan(monkeypatch):
    runtime, owner, driver, clock, worker, release, errors = paused_plan(monkeypatch)
    try:
        runtime._service_follow_wheels()
        assert not driver.pairs
        assert owner.motor_io_lock.acquire(timeout=.2)
        owner.motor_io_lock.release()
    finally:
        finish(worker, release, errors)
    assert driver.pairs == [(24, -24)]


def test_hazard_appearing_while_waiting_for_serial_commit_stops_before_speed(monkeypatch):
    runtime, owner, driver, clock, worker, release, errors = paused_plan(
        monkeypatch, phase="commit")
    try:
        held = hold_until_commit_waits(owner, release)
        try:
            runtime.hard_stop_check = lambda action: True
        finally:
            held.release()
    finally:
        finish(worker, release, errors)
    assert driver.stops == [1]
    assert not driver.pairs


def test_superseded_plan_cannot_resurrect_reversal_state_after_stop_reset(monkeypatch):
    runtime, owner, driver, clock, worker, release, errors = paused_plan(monkeypatch)

    def finish_old_computation():
        runtime._visible_wheel_guard.pending_signs = (-1, 1)
        runtime._visible_wheel_guard.resume_signs = (-1, 1)
        runtime._visible_wheel_guard.residual_forward_until = 10.20

    runtime._offline_guard_tail = finish_old_computation
    try:
        assert owner.motor_io_lock.acquire(timeout=.2)
        try:
            runtime.backend.send_stop("safety-during-old-guard", mode="emergency", preserve_zero=True)
            runtime._visible_wheel_guard.reset()
        finally:
            owner.motor_io_lock.release()
    finally:
        finish(worker, release, errors)
    assert driver.stops == [1] and not driver.pairs
    guard = runtime._visible_wheel_guard
    assert guard.pending_signs is None and guard.resume_signs is None
    assert guard.residual_forward_until == 0
    assert runtime._follow_wheel_clock.last_axes is None


def test_zero_write_receipt_and_guard_accounting_are_in_one_short_commit(monkeypatch):
    runtime, owner, driver, clock, _ = live_writer(monkeypatch)
    owner._follow_wheel_axes = lambda now: (1, owner._lateral_yaw_revision, 0., 0.)
    noted = []
    original = runtime._visible_wheel_guard.note_sent

    def note(pair, now):
        assert owner.motor_io_lock.locked()
        assert driver.pairs == [(0, 0)]
        noted.append(runtime.backend.last_speed_receipt)
        return original(pair, now)

    monkeypatch.setattr(runtime._visible_wheel_guard, "note_sent", note)
    runtime._service_follow_wheels()
    assert len(noted) == 1
    assert runtime._follow_wheel_last_receipt is noted[0]
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("yaw_only", [False, True])
def test_late_filtered_encoder_publication_cannot_replace_terminal_sample(
        monkeypatch, yaw_only):
    runtime, owner, driver, clock, _ = live_writer(monkeypatch)
    old_sample = feedback(10., 0, 0)
    runtime._steering_feedback = old_sample
    runtime.get_steering_feedback = lambda: runtime._steering_feedback
    if yaw_only:
        owner._follow_wheel_axes = lambda now: (1, owner._lateral_yaw_revision, 0., 4.)
        owner._has_fresh_lateral_yaw = lambda uid: True
    entered, release, published = threading.Event(), threading.Event(), threading.Event()
    errors, observers = [], []
    original = runtime._follow_motor_call
    # This is after the terminal cap (or its pure-yaw bypass), before the
    # actual physical packet. The encoder already read but filtered late.
    def pause(method, *args, **kwargs):
        if method == "send_targets" and args[-1] == "FOLLOW20":
            assert owner.motor_io_lock.locked()
            entered.set()
            assert release.wait(2.)
            assert runtime._steering_feedback is old_sample
        return original(method, *args, **kwargs)

    runtime._follow_motor_call = pause
    probe = CommitWaitLock(owner.motor_io_lock)
    owner.motor_io_lock = probe

    def observe(sample):
        assert not owner.motor_io_lock.locked()
        assert not runtime._steering_feedback_lock.locked()
        observers.append(sample)

    runtime._observe_executed_speed_response = observe
    new_sample = feedback(10.01, -10, 20)

    def writer():
        try:
            runtime._service_follow_wheels()
        except BaseException as exc:
            errors.append(exc)

    def publisher():
        try:
            runtime._publish_steering_feedback(new_sample)
            published.set()
        except BaseException as exc:
            errors.append(exc)

    follow = threading.Thread(target=writer, daemon=True)
    encoder = threading.Thread(target=publisher, daemon=True)
    try:
        follow.start()
        assert entered.wait(1.), errors
        encoder.start()
        assert probe.waiting.wait(1.)
        assert not published.is_set()
        assert runtime.get_recording_feedback() is old_sample
        clock[0] = 10.02
    finally:
        release.set()
        follow.join(2.)
        if encoder.ident is not None:
            encoder.join(2.)
    assert not errors
    assert not follow.is_alive() and not encoder.is_alive()
    assert published.is_set() and runtime._steering_feedback is new_sample
    assert new_sample.timestamp == 10.01
    assert observers == [new_sample]
    assert driver.pairs == ([(4, 4)] if yaw_only else [(24, -24)])


def test_revocation_full_axes_check_runs_outside_serial_lock(monkeypatch):
    runtime, owner, driver, clock, _ = live_writer(monkeypatch)
    runtime._service_follow_wheels()
    clock[0] = 10.05
    calls = []

    def inactive_axes(now):
        assert not owner.motor_io_lock.locked()
        calls.append(now)
        return None

    owner._follow_wheel_axes = inactive_axes
    runtime._service_follow_wheels()
    assert len(calls) == 2
    assert driver.pairs == [(24, -24), (0, 0)]
