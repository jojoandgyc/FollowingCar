"""Current forward authority survives equivalent yaw publications, fake I/O.

Exercise the real periodic planner, braking budget, wheel guard, motor driver
receipt and STOP lifecycle. Hooks only schedule publications/cache/clock
changes, never replace an admission result with success.
"""
import pytest

from test_distance_momentum_brake import start
from test_follow_parking_snapshot import parked_writer
from test_unified_forward_snapshot import feedback, writer
from test_zero_publication_lifecycle import arm_planning_owner


@pytest.mark.parametrize("stage", ["entry", "budget"])
def test_repeated_neutral_publications_keep_current_straight_packets(monkeypatch, stage):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=0.)
    source, timing = owner._depth30_linear_snapshot, owner._depth30_linear_timing
    target = rt if stage == "entry" else owner
    method = "_linear_packet_write_limit" if stage == "entry" else "_depth_forward_continuation_limit"
    original = getattr(target, method)
    changes = []

    def refresh(*args, **kwargs):
        changes.append(clock[0])
        publish(60., 0., new_depth=False)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, method, refresh)
    for _ in range(4):
        clock[0] += .04
        rt._steering_feedback = feedback(clock[0], 24., 24.)
        rt._service_follow_wheels()
    assert len(changes) >= 4
    assert driver.pairs == [(60, -60)] * 5
    assert not driver.stops and not owner.motor_io_lock.locked()
    assert owner._depth30_linear_snapshot is source
    assert owner._depth30_linear_timing is timing
    assert timing.depth_expires_at == 10.25


def test_neutral_publication_cannot_reject_already_acknowledged_straight_pair(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=46., yaw=0.)
    publish(86., 0.)
    receipt, execution = rt.backend.last_speed_receipt, rt._forward_execution_anchor
    history = tuple(rt._continuation_executed_speed_history)
    limit = owner._depth_forward_continuation_limit
    changes = []

    def refresh(*args, **kwargs):
        changes.append(True)
        publish(86., 0., new_depth=False)
        return limit(*args, **kwargs)

    monkeypatch.setattr(owner, "_depth_forward_continuation_limit", refresh)
    arm_planning_owner(monkeypatch, rt)
    with rt._follow_planning_attempt():
        assert rt._current_receipt_survives_publication(1)
    assert changes and driver.pairs == [(46, -46)]
    assert rt.backend.last_speed_receipt is receipt
    assert rt._forward_execution_anchor is execution
    assert tuple(rt._continuation_executed_speed_history) == history


@pytest.mark.parametrize("old_yaw", [0., -4.])
def test_new_depth_publications_are_adopted_before_retry_exhaustion(monkeypatch, old_yaw):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=46., yaw=old_yaw)
    original = rt._linear_packet_write_limit
    changes = []

    def refresh(*args, **kwargs):
        changes.append(True)
        clock[0] += .002
        publish(86., 0., new_depth=len(changes) <= 3)
        return original(*args, **kwargs)

    monkeypatch.setattr(rt, "_linear_packet_write_limit", refresh)
    rt._service_follow_wheels()
    assert len(changes) == (2 if old_yaw else 1)
    assert not driver.stops and all(left > 0 > right for left, right in driver.pairs)
    if old_yaw:
        # An obsolete curved receipt must be replaced, never silently held.
        assert driver.pairs == [(42, -50), (86, -86)]
    else:
        assert driver.pairs == [(46, -46), (46, -46)]
    assert rt._forward_execution_anchor.sample_timestamp == clock[0]
    assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(clock[0] + .25)
    # Adoption only retains the guarded pair; it is not a persistent cap.
    clock[0] += .05
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (86, -86)


@pytest.mark.parametrize("fault", ["uid", "identity", "stop", "explicit_stop", "receipt",
                                    "depth", "feedback", "budget"])
def test_neutral_refresh_preserves_every_physical_and_ownership_gate(monkeypatch, fault):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=46., yaw=0.)
    original = rt._linear_packet_write_limit
    changes = []

    def refresh(*args, **kwargs):
        if not changes:
            changes.append(True)
            publish(46., 0., new_depth=False)
            if fault == "uid":
                owner._follow_controller.active_target_id = 2
            elif fault == "identity":
                owner._detector_identity_lease = False
            elif fault == "stop":
                rt.backend.send_stop("new_stop", mode="emergency", preserve_zero=True)
            elif fault == "explicit_stop":
                owner._explicit_stop_requested = True
            elif fault == "receipt":
                rt.backend.send_targets(12, -12, "other_writer")
            elif fault == "depth":
                clock[0] = 10.251
                rt._steering_feedback = feedback(clock[0], 24., 24.)
            elif fault == "feedback":
                clock[0] += .151
                rt._steering_feedback = feedback(clock[0]-.151, 24., 24.)
            else:
                # Current real sample budget tightens to momentum braking.
                from dataclasses import replace
                timing = owner._depth30_linear_timing
                assessment = replace(timing.braking_assessment, distance_m=1.1)
                timing = replace(timing, braking_assessment=assessment)
                owner._depth30_prepared_timing = owner._depth30_linear_timing = timing
        return original(*args, **kwargs)

    monkeypatch.setattr(rt, "_linear_packet_write_limit", refresh)
    rt._service_follow_wheels()
    own_packets = driver.pairs[2:] if fault == "receipt" else driver.pairs[1:]
    assert changes and not any(left > 0 > right for left, right in own_packets)
    if fault == "stop":
        assert driver.pairs == [(46, -46)] and driver.stops == [1]


@pytest.mark.parametrize("stage", ["entry", "budget"])
def test_real_turn_arriving_during_neutral_plan_is_rebuilt(monkeypatch, stage):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=46., yaw=0.)
    target = rt if stage == "entry" else owner
    method = "_linear_packet_write_limit" if stage == "entry" else "_depth_forward_continuation_limit"
    original = getattr(target, method)
    changed = []

    def turn(*args, **kwargs):
        if not changed:
            changed.append(True)
            publish(46., 4., new_depth=False)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, method, turn)
    rt._service_follow_wheels()
    assert driver.pairs == [(46, -46), (50, -42)] and not driver.stops


def test_parking_exit_uses_encoder_published_while_waiting_for_commit(monkeypatch):
    rt, owner, driver, clock, _ = parked_writer(monkeypatch, base=34., yaw=0.)
    begin = rt._begin_follow_commit
    changed = []

    def wait():
        if not changed:
            changed.append(True)
            clock[0] += .16  # The proposal's feedback expires, its Depth does not.
            rt._steering_feedback = feedback(clock[0], 0., 0.)
        return begin()

    monkeypatch.setattr(rt, "_begin_follow_commit", wait)
    rt._service_follow_wheels()
    assert driver.pairs == [(34, -34), (34, -34)] and driver.stops == [1]
    assert not rt.backend.normal_zero_hold


@pytest.mark.parametrize("fault", [None, "uid", "stop", "feedback", "old_sample"])
def test_brake_exit_readmits_latest_same_uid_grant_without_reparking(monkeypatch, fault):
    rt, owner, driver, clock, sample, _ = start(monkeypatch)
    clock[0] += .03
    sample(3., base=30., outer=24.)
    begin, stop = rt._begin_follow_commit, rt._ordinary_snapshot_stop
    waited, reasons = [], []

    def wait():
        if not waited:
            waited.append(True)
            clock[0] += .251
            rt._steering_feedback = feedback(clock[0], 24., 24.)
        return begin()

    def supersede(uid, reason, **kwargs):
        reasons.append(reason)
        if len(reasons) == 1:
            sample(3., base=34., outer=24.)
            if fault == "uid":
                owner._follow_controller.active_target_id = 2
            elif fault == "stop":
                rt.backend.send_stop("new_owner", mode="emergency", preserve_zero=True)
            elif fault == "feedback":
                rt._steering_feedback = feedback(clock[0]-.151, 24., 24.)
            elif fault == "old_sample":
                sample(3., base=34., outer=24., stamp=rt._distance_brake_sample_floor)
        return stop(uid, reason, **kwargs)

    monkeypatch.setattr(rt, "_begin_follow_commit", wait)
    monkeypatch.setattr(rt, "_ordinary_snapshot_stop", supersede)
    rt._service_follow_wheels()
    assert waited and reasons[0] == "parking_exit_authority_expired"
    if fault is None:
        assert driver.pairs == [(60, -60), (34, -34)]
        assert driver.stops == [1] and not rt.backend.normal_zero_hold
        assert rt._forward_execution_anchor.sample_timestamp == clock[0]
    else:
        assert driver.pairs == [(60, -60)]
        assert rt.backend.normal_zero_hold
    assert not owner.motor_io_lock.locked()
