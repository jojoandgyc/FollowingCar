"""Ordinary axis expiry/renewal at commit: fake motor, real receipt ledger.

An expired yaw must not cancel independently live forward authority. Conversely,
replacing the observation while a zero plan waits for I/O must not hide a real
identity/STOP/depth revocation. No camera, serial port or physical motor is used.
"""
from dataclasses import replace

import pytest

from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from test_unified_forward_snapshot import feedback, inject, writer


def cross_yaw_deadline(monkeypatch, rt, owner, clock, *, cap_call=1, fault=None):
    """Cross only the original yaw deadline during a quiet terminal cap call."""
    clock[0] = 10.149
    rt._steering_feedback = feedback(clock[0], 24., 24.)
    proof = ValidatedVisualObservation(1, 3, 200, 10., 10., 10.25, "full")
    owner._validated_visual_observation = proof
    original = owner._depth_forward_continuation_limit
    calls = []

    def limit(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(True)
        if len(calls) == cap_call:
            clock[0] = 10.151
            if fault == "depth":
                clock[0] = 10.251
                rt._steering_feedback = feedback(clock[0], 24., 24.)
            elif fault == "visual":
                owner._validated_visual_observation = replace(proof, expires_at=10.15)
            elif fault == "identity":
                owner._detector_identity_lease = False
            elif fault == "uid":
                owner._follow_controller.active_target_id = 2
            elif fault == "stop":
                rt.backend.send_stop("audit_yaw_deadline_stop", mode="emergency",
                                     preserve_zero=True)
        return result

    monkeypatch.setattr(owner, "_depth_forward_continuation_limit", limit)
    return calls


@pytest.mark.parametrize("yaw", [-6., 6.])
@pytest.mark.parametrize("cap_call", [1, 2])
def test_yaw_expiry_during_either_terminal_cap_preserves_live_forward(
        monkeypatch, yaw, cap_call):
    rt, owner, driver, clock, _, _ = writer(monkeypatch, base=90., yaw=yaw)
    old_raw = owner._depth30_linear_snapshot
    old_timing = owner._depth30_linear_timing
    old_intent = owner._lateral_intent_store.snapshot()
    old_receipt = rt.backend.last_speed_receipt
    calls = cross_yaw_deadline(monkeypatch, rt, owner, clock, cap_call=cap_call)

    rt._service_follow_wheels()

    assert len(calls) >= cap_call
    assert driver.pairs == [(int(90+yaw), -int(90-yaw)), (90, -90)]
    assert not driver.stops
    assert owner._depth30_linear_snapshot is old_raw
    assert owner._depth30_linear_timing is old_timing
    assert owner._lateral_intent_store.snapshot() is old_intent
    assert not old_intent.valid(clock[0])  # Refresh never renews the old yaw.
    assert rt.backend.last_speed_receipt is not old_receipt
    assert rt._forward_execution_anchor.sample_timestamp == 10.
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("fault", ["depth", "visual", "identity", "uid", "stop"])
@pytest.mark.parametrize("cap_call", [1, 2])
def test_yaw_expiry_refresh_never_overrides_forward_revocations(
        monkeypatch, fault, cap_call):
    rt, owner, driver, clock, _, _ = writer(monkeypatch)
    calls = cross_yaw_deadline(
        monkeypatch, rt, owner, clock, cap_call=cap_call, fault=fault)

    rt._service_follow_wheels()

    assert len(calls) >= cap_call
    assert all(pair == (0, 0) for pair in driver.pairs[1:])
    assert not owner.motor_io_lock.locked()
    if fault == "stop":
        assert driver.stops == [1]
        assert len(driver.pairs) == 1  # Preserve the newer STOP's motor mode.


def expire_before_snapshot(monkeypatch, rt, clock):
    """Make the original grant expire after admission, before the short plan."""
    original = rt._send_ordinary_snapshot_forward
    entered = []

    def snapshot(*args, **kwargs):
        if not entered:
            entered.append(True)
            clock[0] = 10.251
            rt._steering_feedback = feedback(clock[0], 24., 24.)
        return original(*args, **kwargs)

    monkeypatch.setattr(rt, "_send_ordinary_snapshot_forward", snapshot)
    return entered


@pytest.mark.parametrize("yaw", [-6., 0., 6.])
def test_new_same_uid_grant_supersedes_old_zero_plan_while_acquiring_io(
        monkeypatch, yaw):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    old_receipt = rt.backend.last_speed_receipt
    entered = expire_before_snapshot(monkeypatch, rt, clock)
    updates = inject(monkeypatch, rt, "commit", lambda: publish(88., yaw))

    rt._service_follow_wheels()

    assert entered == updates == [True]
    assert driver.pairs == [(84, -96), (int(88+yaw), -int(88-yaw))]
    assert not driver.stops
    assert owner._depth30_linear_snapshot == ("forward", 88., 1, 10.251)
    assert rt._forward_execution_anchor.sample_timestamp == 10.251
    assert rt.backend.last_speed_receipt is not old_receipt
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("fault", ["uid", "identity", "visual", "depth", "hazard", "stop"])
def test_new_grant_at_zero_commit_cannot_launder_real_revocation(monkeypatch, fault):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    expire_before_snapshot(monkeypatch, rt, clock)

    def publish_with_fault():
        publish(88., -6.)
        if fault == "uid":
            owner._follow_controller.active_target_id = 2
        elif fault == "identity":
            owner._detector_identity_lease = False
        elif fault == "visual":
            owner._validated_visual_observation = False
        elif fault == "depth":
            clock[0] += .251
            rt._steering_feedback = feedback(clock[0], 24., 24.)
        elif fault == "hazard":
            rt.hard_stop_check = lambda _: True
        elif fault == "stop":
            rt.backend.send_stop("audit_zero_plan_stop", mode="emergency",
                                 preserve_zero=True)

    updates = inject(monkeypatch, rt, "commit", publish_with_fault)
    rt._service_follow_wheels()

    assert updates == [True]
    assert all(pair == (0, 0) for pair in driver.pairs[1:])
    assert not owner.motor_io_lock.locked()
    if fault == "stop":
        assert driver.stops == [1]
        assert len(driver.pairs) == 1


def test_already_expired_replacement_is_revoked_without_extra_retries(
        monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    expire_before_snapshot(monkeypatch, rt, clock)
    original = rt._begin_follow_commit
    commits = []

    def delayed_commit():
        commits.append(clock[0])
        if len(commits) <= 3:
            # Even the replacement expires before lock acquisition. It cannot
            # justify deferring revocation or spending further retry attempts.
            publish(88., -6.)
            clock[0] += .251
            rt._steering_feedback = feedback(clock[0], 24., 24.)
        return original()

    monkeypatch.setattr(rt, "_begin_follow_commit", delayed_commit)
    rt._service_follow_wheels()

    assert len(commits) == 1
    assert driver.pairs == [(84, -96), (0, 0)]
    assert rt._forward_execution_anchor is None
    assert rt._continuation_executed_speed_history[-1].outer_rpm == 0
    assert not owner.motor_io_lock.locked()


def test_real_stop_during_second_superseded_zero_attempt_keeps_stop_ownership(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    expire_before_snapshot(monkeypatch, rt, clock)
    original = rt._begin_follow_commit
    commits = []

    def delayed_commit():
        commits.append(clock[0])
        if len(commits) == 1:
            publish(88., -6.)
            # This replacement is genuinely live when readmission begins.
            clock[0] += .005
            rt._steering_feedback = feedback(clock[0], 24., 24.)
        elif len(commits) == 2:
            publish(88., -6.)
            rt.backend.send_stop("audit_second_zero_refresh_stop", mode="emergency",
                                 preserve_zero=True)
        return original()

    monkeypatch.setattr(rt, "_begin_follow_commit", delayed_commit)
    rt._service_follow_wheels()

    # The second commit encounters STOP; the final zero path only rechecks
    # ownership and must not write a speed-mode zero over that STOP.
    assert len(commits) == 3
    assert driver.pairs == [(84, -96)]
    assert driver.stops == [1]
    assert rt._forward_execution_anchor is None
    assert not owner.motor_io_lock.locked()
