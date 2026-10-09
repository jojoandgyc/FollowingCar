"""Final serial-commit publication races, fake clocks and driver only."""
import pytest

from car_control_modular.control_types import DepthLinearTiming
from car_control_modular.lateral_intent import LateralIntentStore
from test_cap331_intent_handoff import ordinary_intent
from test_follow_deferred_diagnostics import live_writer
from test_follow_offlock_planning import paused_plan, hold_until_commit_waits, finish
from test_visible_wheel_continuity import feedback


def writer(monkeypatch):
    runtime, owner, driver, clock, _ = live_writer(monkeypatch)
    store = owner._lateral_intent_store = LateralIntentStore()
    yaw = [0.]

    def publish(base=24., correction=0., *, stamp=None):
        stamp = clock[0] if stamp is None else stamp
        raw = ("forward", float(base), 1, stamp)
        timing = DepthLinearTiming(raw, stamp, stamp + .25)
        owner._depth30_prepared_timing = timing
        owner._depth30_linear_snapshot = raw
        owner._depth30_linear_timing = timing
        yaw[0] = correction
        owner._lateral_yaw_revision += 1
        store.publish(ordinary_intent(
            capture=stamp, published=clock[0], initial_correction_rpm=correction,
            base_percent=base, base_rpm=base))
        runtime._steering_feedback = feedback(clock[0], 24., 24.)
        return raw, timing

    def depth(uid, now=None):
        assert not owner.motor_io_lock.locked(), "re-admission monopolized serial lock"
        raw = owner._depth30_linear_snapshot
        return raw if raw and raw[2] == uid and 0 <= clock[0]-raw[3] <= .25 else None

    def axes(now):
        raw = depth(owner._follow_controller.active_target_id, now)
        return (1, owner._lateral_yaw_revision, raw[1] if raw else 0., yaw[0])

    publish()
    owner._fresh_depth_linear_snapshot = depth
    owner._follow_wheel_axes = axes
    owner._has_fresh_lateral_yaw = lambda uid: yaw[0] != 0 and store.snapshot().valid(clock[0])
    runtime.get_steering_feedback = lambda: runtime._steering_feedback
    runtime._service_follow_wheels()
    assert driver.pairs == [(24, -24)]
    clock[0] = 10.05
    return runtime, owner, driver, clock, publish


@pytest.mark.parametrize("base", [12., 24., 40.])
@pytest.mark.parametrize("yaw", [-4., 0., 4.])
def test_new_depth_and_turn_after_planning_are_fully_rebuilt_without_zero(
        monkeypatch, base, yaw):
    rt, owner, driver, clock, publish = writer(monkeypatch)
    begin = rt._begin_follow_commit
    updates = []

    def change():
        if not updates:
            assert not owner.motor_io_lock.locked()
            updates.append(publish(base, yaw))
        return begin()

    monkeypatch.setattr(rt, "_begin_follow_commit", change)
    rt._service_follow_wheels()
    assert driver.pairs == [(24, -24), (int(base+yaw), -int(base-yaw))]
    assert owner._depth30_linear_timing is updates[0][1]
    assert rt._forward_execution_anchor.sample_timestamp == clock[0]
    assert rt._follow_wheel_clock.last_axes[2:] == (base, yaw)
    assert not driver.stops


@pytest.mark.parametrize("yaw", [-4., 0., 4.])
def test_only_lateral_publication_rebuilds_without_renewing_depth(monkeypatch, yaw):
    rt, owner, driver, clock, _ = writer(monkeypatch)
    raw = owner._depth30_linear_snapshot
    timing = owner._depth30_linear_timing
    begin = rt._begin_follow_commit
    updates = []

    def change():
        if not updates:
            owner._lateral_yaw_revision += 1
            intent = owner._lateral_intent_store.publish(ordinary_intent(
                capture=clock[0], published=clock[0], initial_correction_rpm=yaw))
            owner._lateral_turn_response_policy = (intent.sequence, False)
            owner._follow_wheel_axes = lambda now: (1, owner._lateral_yaw_revision, 24., yaw)
            owner._has_fresh_lateral_yaw = lambda uid: yaw != 0 and intent.valid(clock[0])
            updates.append(intent)
        return begin()

    monkeypatch.setattr(rt, "_begin_follow_commit", change)
    rt._service_follow_wheels()
    assert driver.pairs == [(24, -24), (int(24+yaw), -int(24-yaw))]
    assert owner._depth30_linear_snapshot is raw
    assert owner._depth30_linear_timing is timing
    assert timing.depth_expires_at == 10.25


def test_same_sample_tightening_at_commit_is_replanned_not_zeroed(monkeypatch):
    rt, owner, driver, clock, publish = writer(monkeypatch)
    stamp = owner._depth30_linear_snapshot[3]
    expiry = owner._depth30_linear_timing.depth_expires_at
    begin = rt._begin_follow_commit
    updates = []

    def change():
        if not updates:
            updates.append(publish(12., 0., stamp=stamp))
        return begin()

    monkeypatch.setattr(rt, "_begin_follow_commit", change)
    rt._service_follow_wheels()
    assert driver.pairs == [(24, -24), (12, -12)]
    assert owner._depth30_linear_timing.depth_expires_at == expiry


@pytest.mark.parametrize("phase", ["before_limit", "inside_quiet_cap"])
def test_publication_during_terminal_limit_retries_outside_io_lock(monkeypatch, phase):
    rt, owner, driver, clock, publish = writer(monkeypatch)
    updates = []

    def change():
        if not updates:
            assert owner.motor_io_lock.locked()
            updates.append(publish(20., -4.))

    if phase == "before_limit":
        limit = rt._linear_packet_write_limit

        def replaced(*args, **kwargs):
            change()
            return limit(*args, **kwargs)

        monkeypatch.setattr(rt, "_linear_packet_write_limit", replaced)
    else:
        owner._depth_forward_continuation_required = lambda *_: True

        def quiet_cap(*args, **kwargs):
            change()
            return 100., "test_valid"

        owner._depth_forward_continuation_limit = quiet_cap
    rt._service_follow_wheels()
    assert driver.pairs == [(24, -24), (16, -24)]
    assert len(updates) == 1 and not owner.motor_io_lock.locked()


@pytest.mark.parametrize("count", [1, 2, 4])
def test_repeated_publication_uses_one_bounded_service_budget(monkeypatch, count):
    rt, owner, driver, clock, publish = writer(monkeypatch)
    begin = rt._begin_follow_commit
    updates = []

    def change():
        if not rt._follow_commit_locked and len(updates) < count:
            assert not owner.motor_io_lock.locked()
            clock[0] += .005
            updates.append(publish(26.+len(updates), 2.))
        return begin()

    monkeypatch.setattr(rt, "_begin_follow_commit", change)
    rt._service_follow_wheels()
    assert len(updates) == min(count, 3)
    if count <= 2:
        base = 25+count
        assert driver.pairs == [(24, -24), (base+2, -(base-2))]
    else:
        # No unbounded retry or expired old-command hold if a producer keeps
        # invalidating every attempt, even at the final physical write gate.
        assert driver.pairs == [(24, -24), (0, 0)]
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("fault", ["stop", "uid", "identity", "expired", "future",
                                    "feedback", "reverse_feedback", "zero", "hazard"])
def test_new_publication_does_not_override_real_invalidation(monkeypatch, fault):
    rt, owner, driver, clock, publish = writer(monkeypatch)
    begin = rt._begin_follow_commit
    updated = []

    def change():
        if not updated:
            updated.append(publish(40., 3.))
            if fault == "stop":
                with owner.motor_io_lock:
                    rt.backend.send_stop("test-stop", mode="emergency", preserve_zero=True)
            elif fault == "uid":
                owner._follow_controller.active_target_id = 2
            elif fault == "identity":
                owner._vision_control_state = "target_lost"
            elif fault == "expired":
                clock[0] += .251
                rt._steering_feedback = feedback(clock[0], 24., 24.)
            elif fault == "future":
                publish(40., 3., stamp=clock[0]+.01)
            elif fault == "feedback":
                rt._steering_feedback = feedback(clock[0]-.151, 24., 24.)
            elif fault == "reverse_feedback":
                rt._steering_feedback = feedback(clock[0], -10., 24.)
            elif fault == "zero":
                publish(0., 0.)
            else:
                rt.hard_stop_check = lambda _action: True
        return begin()

    monkeypatch.setattr(rt, "_begin_follow_commit", change)
    rt._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs[1:])
    if fault in {"stop", "hazard"}:
        assert driver.stops == [1] and len(driver.pairs) == 1


def test_real_serial_wait_new_valid_grant_is_not_treated_as_future_or_invalid(monkeypatch):
    rt, owner, driver, clock, worker, release, errors = paused_plan(
        monkeypatch, phase="commit")
    # The blocked first wrapper remains on the worker stack; subsequent
    # normal begin calls may already own the commit lock and must not pause.
    rt._begin_follow_commit = type(rt)._begin_follow_commit.__get__(rt)
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: owner._depth30_linear_snapshot
    try:
        held = hold_until_commit_waits(owner, release)
        try:
            clock[0] = 10.02
            raw = ("forward", 24., 1, 10.01)
            timing = DepthLinearTiming(raw, 10.01, 10.26)
            owner._depth30_prepared_timing = timing
            owner._depth30_linear_snapshot = raw
            owner._depth30_linear_timing = timing
            rt._steering_feedback = feedback(clock[0], 24., 24.)
        finally:
            held.release()
    finally:
        finish(worker, release, errors)
    assert driver.pairs == [(24, -24)]


def test_stop_in_rebuild_gap_owns_motor_and_prevents_resurrected_speed(monkeypatch):
    rt, owner, driver, clock, publish = writer(monkeypatch)
    begin = rt._begin_follow_commit
    updated = []

    def change():
        if not updated:
            updated.append(publish())
        return begin()

    rebuild = rt._request_follow_commit_rebuild

    def stop_after_request(*args, **kwargs):
        result = rebuild(*args, **kwargs)
        if result:
            rt.backend.send_stop("test-stop-after-rebuild", mode="emergency", preserve_zero=True)
        return result

    monkeypatch.setattr(rt, "_begin_follow_commit", change)
    monkeypatch.setattr(rt, "_request_follow_commit_rebuild", stop_after_request)
    rt._service_follow_wheels()
    assert driver.stops == [1] and driver.pairs == [(24, -24)]
