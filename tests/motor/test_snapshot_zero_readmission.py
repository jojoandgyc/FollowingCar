"""A superseded ordinary stop is requalified at commit, using fake motor I/O.

The hooks only schedule real publication/clock/cache changes. Neither admission,
wheel guards nor the backend receipt ledger is replaced with a success stub.
"""
from dataclasses import replace
import threading

import pytest

from test_unified_forward_snapshot import feedback, writer


def snapshot_loss(monkeypatch, rt, owner, clock, publish, mode):
    """Make a real snapshot admission fail after its ordinary entry qualified."""
    changes = []
    if mode in {"missing_depth", "missing_feedback"}:
        original = rt._plan_ordinary_snapshot_forward

        def lose_before_snapshot(*args, **kwargs):
            if not changes and getattr(rt, "_follow_snapshot_planning", False):
                changes.append(mode)
                if mode == "missing_depth":
                    owner._depth30_linear_snapshot = None
                else:
                    rt._steering_feedback = replace(rt._steering_feedback, trustworthy=False)
            return original(*args, **kwargs)

        monkeypatch.setattr(rt, "_plan_ordinary_snapshot_forward", lose_before_snapshot)
    elif mode == "terminal_feedback":
        original = rt._visible_wheel_guard.limit

        def age_after_guard(*args, **kwargs):
            result = original(*args, **kwargs)
            if not changes and getattr(rt, "_follow_snapshot_planning", False):
                changes.append(mode)
                # The physical Depth lease is still live, but cached encoder
                # feedback really expires before the terminal feedback gate.
                clock[0] = 10.201
            return result

        monkeypatch.setattr(rt._visible_wheel_guard, "limit", age_after_guard)
    else:
        original = rt._linear_packet_write_limit
        original_budget = owner._depth_forward_continuation_limit
        pending = [False]

        def age_at_terminal(*args, **kwargs):
            if mode == "terminal_depth" and not changes:
                changes.append(mode)
                clock[0] = 10.251
            elif mode == "exhausted" and len(changes) < 3:
                pending[0] = True
            return original(*args, **kwargs)

        def publish_during_budget(*args, **kwargs):
            result = original_budget(*args, **kwargs)
            if pending[0]:
                pending[0] = False
                changes.append(mode)
                clock[0] += .005
                publish(58., -4.)
                if len(changes) == 3:
                    clock[0] += .251
                    rt._steering_feedback = feedback(clock[0], 24., 24.)
            return result

        monkeypatch.setattr(rt, "_linear_packet_write_limit", age_at_terminal)
        monkeypatch.setattr(owner, "_depth_forward_continuation_limit", publish_during_budget)
    return changes


def at_snapshot_zero_commit(monkeypatch, rt, change):
    """Publish only after the stop has been decided, at its I/O acquisition."""
    reasons, committed = [], []
    original_stop = rt._ordinary_snapshot_stop
    original_commit = rt._begin_follow_commit

    def stop(uid, reason, **kwargs):
        reasons.append(reason)
        return original_stop(uid, reason, **kwargs)

    def commit():
        if reasons and not committed:
            committed.append(True)
            change()
        return original_commit()

    monkeypatch.setattr(rt, "_ordinary_snapshot_stop", stop)
    monkeypatch.setattr(rt, "_begin_follow_commit", commit)
    return reasons, committed


@pytest.mark.parametrize("old_yaw,new_yaw", [(4., 0.), (-4., 0.), (4., -4.), (-4., 4.)])
@pytest.mark.parametrize("mode", ["terminal_depth", "terminal_feedback", "missing_depth",
                                 "missing_feedback", "exhausted"])
def test_new_forward_at_snapshot_zero_commit_never_inserts_obsolete_zero(
        monkeypatch, old_yaw, new_yaw, mode):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=old_yaw)
    original_receipt = rt.backend.last_speed_receipt
    changes = snapshot_loss(monkeypatch, rt, owner, clock, publish, mode)
    reasons, committed = at_snapshot_zero_commit(
        monkeypatch, rt, lambda: publish(54., new_yaw))
    rt._service_follow_wheels()
    assert changes and reasons and committed == [True]
    expected_reason = {
        "terminal_depth": "physical_grant_or_initial_limit",
        "terminal_feedback": "terminal_feedback_invalid",
        # None is deliberately not an equal, valid grant marker. The three
        # snapshot reads therefore exhaust before submitting the old zero.
        "missing_depth": "depth_published_during_snapshot",
        "missing_feedback": "nonordinary_or_feedback_changed",
        "exhausted": "publication_changed_at_terminal",
    }[mode]
    assert reasons[0] == expected_reason
    assert len(driver.pairs) == 2
    left, raw_right = driver.pairs[-1]
    assert left > 0 and raw_right < 0 and .5*(left-raw_right) == 54.
    # The bounded final attempt may deliberately drop yaw, but never retain
    # the obsolete/opposite yaw or invent a higher common speed.
    assert .5*(left+raw_right) in {0., new_yaw}
    assert not driver.stops
    assert rt.backend.last_speed_receipt is not original_receipt
    assert rt._forward_execution_anchor.sample_timestamp == clock[0]
    assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(clock[0]+.25)
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("fault", ["backend_stop", "explicit_stop", "hazard", "uid",
                                   "identity", "depth", "feedback", "reverse_feedback",
                                   "full_reverse", "other_writer", "single_wheel_reverse"])
def test_latest_safety_event_still_wins_over_new_forward_at_zero_commit(monkeypatch, fault):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=4.)
    snapshot_loss(monkeypatch, rt, owner, clock, publish, "terminal_depth")

    def change():
        publish(54., 0.)
        if fault == "backend_stop":
            rt.backend.send_stop("newer-stop", mode="emergency", preserve_zero=True)
        elif fault == "explicit_stop":
            owner._explicit_stop_requested = True
        elif fault == "hazard":
            rt.hard_stop_check = lambda _: True
        elif fault == "uid":
            owner._follow_controller.active_target_id = 2
        elif fault == "identity":
            owner._vision_control_state = "target_lost"
        elif fault == "depth":
            owner._depth30_linear_snapshot = None
        elif fault == "feedback":
            rt._steering_feedback = feedback(clock[0]-.151, 24., 24.)
        elif fault == "reverse_feedback":
            rt._steering_feedback = feedback(clock[0], -10., 24.)
        elif fault == "full_reverse":
            rt._visible_wheel_guard.pending_full_reverse = True
        elif fault == "other_writer":
            rt.backend.send_targets(20, -20, "newer-writer")
        else:
            publish(5., -10.)
            # The helper's normal execution bound assumes >=24 RPM. Correct
            # its low-base physical fixture, without weakening any checker.
            def budget(raw, timing, now, *, feedback, quiet):
                measured = max(feedback.left_forward_rpm, feedback.right_forward_rpm)
                result = timing.braking_assessment.budget(now, measured,
                    authorized_rpm=raw[1], execution_bound_rpm=max(measured, raw[1]+10.))
                return result.cap_rpm, result.reason

            owner._depth_forward_continuation_limit = budget

    reasons, committed = at_snapshot_zero_commit(monkeypatch, rt, change)
    rt._service_follow_wheels()
    assert reasons and committed == [True]
    own_packets = driver.pairs[2:] if fault == "other_writer" else driver.pairs[1:]
    assert not any(left > 0 and right < 0 for left, right in own_packets)
    if fault == "backend_stop":
        assert driver.stops == [1] and len(driver.pairs) == 1
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("yaw", [-4., 0., 4.])
def test_same_grant_new_encoder_feedback_can_supersede_temporary_feedback_zero(monkeypatch, yaw):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=yaw)
    source = owner._depth30_linear_snapshot
    timing = owner._depth30_linear_timing
    intent = owner._lateral_intent_store.snapshot()
    revision = owner._lateral_yaw_revision
    snapshot_loss(monkeypatch, rt, owner, clock, publish, "missing_feedback")

    def fresh_encoder_only():
        rt._steering_feedback = feedback(clock[0], 24., 24.)

    reasons, committed = at_snapshot_zero_commit(monkeypatch, rt, fresh_encoder_only)
    rt._service_follow_wheels()
    assert reasons[0] == "nonordinary_or_feedback_changed" and committed == [True]
    assert driver.pairs == [(int(60+yaw), -int(60-yaw))]*2
    assert owner._depth30_linear_snapshot is source
    assert owner._depth30_linear_timing is timing
    assert owner._lateral_intent_store.snapshot() is intent
    assert owner._lateral_yaw_revision == revision
    assert rt._forward_execution_anchor.sample_timestamp == 10.
    assert not driver.stops


@pytest.mark.parametrize("yaw", [-4., 0., 4.])
@pytest.mark.parametrize("stage,invalid_kind", [("getter", "untrustworthy"),
                                               ("getter", "unavailable"),
                                               ("after_guard", "untrustworthy")])
def test_feedback_lost_during_snapshot_and_restored_at_zero_commit_uses_actual_rejected_sample(
        monkeypatch, yaw, stage, invalid_kind):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=yaw)
    source = owner._depth30_linear_snapshot
    timing = owner._depth30_linear_timing
    intent = owner._lateral_intent_store.snapshot()
    revision = owner._lateral_yaw_revision
    entry_feedback = rt._steering_feedback
    changed, restored = [], []

    def invalidate():
        assert rt._steering_feedback is entry_feedback
        # This happens AFTER the snapshot attempt captured its inputs. The
        # actual getter/cache really sees failure; it is not a fabricated
        # return value at odds with the cache used at commit.
        rt._steering_feedback = (None if invalid_kind == "unavailable" else
            replace(feedback(clock[0], 24., 24.), trustworthy=False))
        changed.append(rt._steering_feedback)

    if stage == "getter":
        original = rt.get_steering_feedback

        def getter():
            if getattr(rt, "_follow_snapshot_planning", False) and not changed:
                invalidate()
            return original()

        monkeypatch.setattr(rt, "get_steering_feedback", getter)
    else:
        original = rt._visible_wheel_guard.limit

        def guard(*args, **kwargs):
            result = original(*args, **kwargs)
            if getattr(rt, "_follow_snapshot_planning", False) and not changed:
                invalidate()
            return result

        monkeypatch.setattr(rt._visible_wheel_guard, "limit", guard)

    def restore_encoder_only():
        assert changed and rt._steering_feedback is changed[0]
        rt._steering_feedback = feedback(clock[0], 24., 24.)
        restored.append(rt._steering_feedback)

    reasons, committed = at_snapshot_zero_commit(monkeypatch, rt, restore_encoder_only)
    rt._service_follow_wheels()
    assert len(changed) == len(restored) == 1 and committed == [True]
    assert reasons[0] == ("nonordinary_or_feedback_changed" if stage == "getter"
                          else "terminal_feedback_invalid")
    assert driver.pairs == [(int(60+yaw), -int(60-yaw))]*2
    assert owner._depth30_linear_snapshot is source
    assert owner._depth30_linear_timing is timing
    assert owner._lateral_intent_store.snapshot() is intent
    assert owner._lateral_yaw_revision == revision
    assert rt._forward_execution_anchor.sample_timestamp == 10.
    assert not driver.stops and not owner.motor_io_lock.locked()


def test_repeated_zero_readmission_is_bounded_and_does_not_replay_expired_motion(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=4.)
    original_limit = rt._linear_packet_write_limit
    original_stop = rt._ordinary_snapshot_stop
    calls, stops = [], []

    def expire_each_plan(*args, **kwargs):
        calls.append(True)
        assert len(calls) <= 4, "zero readmission reset the shared planning budget"
        clock[0] += .251
        return original_limit(*args, **kwargs)

    def publish_at_each_stop(uid, reason, **kwargs):
        stops.append(reason)
        assert len(stops) <= 5, "snapshot zero retry is unbounded"
        publish(54., 0.)
        return original_stop(uid, reason, **kwargs)

    monkeypatch.setattr(rt, "_linear_packet_write_limit", expire_each_plan)
    monkeypatch.setattr(rt, "_ordinary_snapshot_stop", publish_at_each_stop)
    rt._service_follow_wheels()
    assert len(calls) == 4
    assert driver.pairs == [(64, -56), (0, 0)]
    assert rt._forward_execution_anchor is None
    assert not owner.motor_io_lock.locked()


def test_real_serial_lock_wait_requalifies_a_new_grant_before_old_zero(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=4.)
    snapshot_loss(monkeypatch, rt, owner, clock, publish, "terminal_depth")
    reached_zero, release_zero, waiting_for_lock = (
        threading.Event(), threading.Event(), threading.Event())
    original_stop = rt._ordinary_snapshot_stop
    original_commit = rt._begin_follow_commit
    stopping = []
    errors = []

    def stop(uid, reason, **kwargs):
        stopping.append(reason)
        return original_stop(uid, reason, **kwargs)

    def commit():
        if stopping and not reached_zero.is_set():
            reached_zero.set()
            assert release_zero.wait(2.), "test did not release zero commit"
            waiting_for_lock.set()
        return original_commit()

    def run():
        try:
            rt._service_follow_wheels()
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(rt, "_ordinary_snapshot_stop", stop)
    monkeypatch.setattr(rt, "_begin_follow_commit", commit)
    worker = threading.Thread(target=run, daemon=True)
    try:
        worker.start()
        assert reached_zero.wait(1.)
        with owner.motor_io_lock:
            release_zero.set()
            assert waiting_for_lock.wait(1.), "worker did not reach the held serial lock"
            publish(54., 0.)
        worker.join(2.)
    finally:
        release_zero.set()
        worker.join(2.)
    assert not errors and not worker.is_alive()
    assert driver.pairs == [(64, -56), (54, -54)]
    assert not driver.stops and not owner.motor_io_lock.locked()
