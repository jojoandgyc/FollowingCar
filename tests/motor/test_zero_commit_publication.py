"""A requested zero waiting for serial ownership cannot erase a new grant.

CAP253 published positive Depth between an old zero plan and its write.
This reproduces that ordering with the complete periodic writer and fake I/O,
not a claim that old field feedback predicts the repaired physical trajectory.
"""
from dataclasses import replace

import pytest

from test_unified_forward_snapshot import feedback, inject, writer


def pending_zero(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    publish(0., 0.)
    owner._depth30_linear_snapshot = None
    owner._depth30_prepared_timing = owner._depth30_linear_timing = None
    return rt, owner, driver, clock, publish


@pytest.mark.parametrize("base,yaw", [(40., -7.), (40., 0.), (40., 7.), (20., 0.)])
@pytest.mark.parametrize("publication_stage", ["lock_wait", "locked"])
def test_new_forward_at_requested_zero_commit_is_readmitted_without_zero(
        monkeypatch, base, yaw, publication_stage):
    rt, owner, driver, clock, publish = pending_zero(monkeypatch)
    receipt = rt.backend.last_speed_receipt
    if publication_stage == "lock_wait":
        calls = inject(monkeypatch, rt, "commit", lambda: publish(base, yaw))
    else:
        original = rt._begin_follow_commit
        calls = []

        def commit():
            result = original()
            if not calls:
                calls.append(True)
                assert owner.motor_io_lock.locked()
                publish(base, yaw)
            return result

        monkeypatch.setattr(rt, "_begin_follow_commit", commit)

    rt._service_follow_wheels()

    assert calls == [True]
    assert driver.pairs == [(84, -96), (int(base+yaw), -int(base-yaw))]
    assert rt._forward_execution_anchor.sample_timestamp == clock[0]
    assert rt.backend.last_speed_receipt.sequence == receipt.sequence+1
    assert len(rt._continuation_executed_speed_history) == 2
    assert all(item.outer_rpm > 0 for item in rt._continuation_executed_speed_history)
    assert rt._follow_wheel_clock.last_axes[2:] == (base, yaw)
    assert not owner.motor_io_lock.locked() and not driver.stops


@pytest.mark.parametrize("fault", ["uid", "identity", "visual", "expired", "feedback",
                                    "reverse", "hazard", "stop", "brake", "near", "cap"])
def test_requested_zero_refresh_still_checks_every_current_authority(monkeypatch, fault):
    rt, owner, driver, clock, publish = pending_zero(monkeypatch)

    def change():
        publish(40., -7.)
        if fault == "uid":
            owner._follow_controller.active_target_id = 2
        elif fault == "identity":
            owner._detector_identity_lease = False
        elif fault == "visual":
            owner._validated_visual_observation = False
        elif fault == "expired":
            clock[0] += .251
            rt._steering_feedback = feedback(clock[0], 24., 24.)
        elif fault == "feedback":
            rt._steering_feedback = feedback(clock[0]-.151, 24., 24.)
        elif fault == "reverse":
            rt._steering_feedback = feedback(clock[0], -20., 24.)
        elif fault == "hazard":
            rt.hard_stop_check = lambda _: True
        elif fault == "stop":
            rt.backend.send_stop("offline_zero_commit_stop", mode="emergency", preserve_zero=True)
        elif fault == "brake":
            owner._brake_hold_active = True
        elif fault == "near":
            owner._near_yaw_park_request = object()
        else:
            # A newly published cap still runs the original brake check.
            timing = owner._depth30_linear_timing
            timing = replace(timing, braking_assessment=replace(
                timing.braking_assessment, distance_m=1.1))
            owner._depth30_prepared_timing = owner._depth30_linear_timing = timing

    calls = inject(monkeypatch, rt, "commit", change)
    rt._service_follow_wheels()

    assert calls == [True]
    assert all(pair == (0, 0) for pair in driver.pairs[1:])
    assert not owner.motor_io_lock.locked()
    if fault in {"hazard", "stop"}:
        assert driver.stops == [1]


def test_unchanged_requested_zero_is_not_suppressed(monkeypatch):
    rt, owner, driver, _, _ = pending_zero(monkeypatch)
    rt._service_follow_wheels()
    assert driver.pairs == [(84, -96), (0, 0)]
    assert not owner.motor_io_lock.locked()
