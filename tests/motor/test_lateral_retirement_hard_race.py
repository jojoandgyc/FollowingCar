"""A prequalified crop cannot turn a newer hard rejection into yaw ownership."""
from dataclasses import replace

import pytest

from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowObservation,
)
from test_cap1040_limited_yaw_owner import setup
from test_visible_wheel_continuity import feedback


def pending_handoff(monkeypatch):
    # This fixture has genuine current limited-yaw identity/publication and
    # the real adapter-independent consumer, writer and FakeDriver backend.
    rt, owner, driver, _, clock, _ = setup(monkeypatch)
    core = ShortFollowController(ShortFollowConfig(enabled=True))
    core.activate(1, 9.8)
    assert core.update(ShortFollowObservation(1, 1039, 9.9, 9.9, 2., .3), 9.9)
    owner._short_follow = core
    writer = rt._short_follow_executor_instance()
    writer._owned = True
    writer._controller = core
    rt.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
    return rt, owner, driver, clock, core, core.snapshot()


@pytest.mark.parametrize("operation", ["revoke", "deactivate"])
@pytest.mark.parametrize("reason", ["identity_rejected", "feedback_reverse", "hard_stop"])
@pytest.mark.parametrize("bind_snapshot", [False, True])
def test_hard_event_after_crop_precheck_is_not_relabelled_or_written(
        monkeypatch, operation, reason, bind_snapshot):
    rt, _, driver, clock, core, checked = pending_handoff(monkeypatch)
    clock[0] = 10.001
    getattr(core, operation)(reason, clock[0])
    hard = core.snapshot()
    floor = core._source_floor
    kwargs = {"expected_snapshot": checked} if bind_snapshot else {}
    assert core.retire_for_lateral_handoff(10., **kwargs) is hard
    assert core.snapshot() is hard and hard.reason == reason
    assert core._source_floor == floor == 10.001
    # The hard event need not have completed motor I/O yet. Old crop evidence
    # is deliberately still present, so source-floor checks alone cannot help.
    rt._service_short_follow()
    assert not driver.pairs and driver.stops


@pytest.mark.parametrize("new_plan", [False, True])
def test_cas_cannot_retire_a_new_uid_or_deliver_old_uid_crop(monkeypatch, new_plan):
    rt, owner, driver, clock, core, checked = pending_handoff(monkeypatch)
    clock[0] = 10.001
    core.activate(2, clock[0])
    owner._follow_controller.active_target_id = 2
    if new_plan:
        clock[0] = 10.004
        assert core.update(ShortFollowObservation(2, 1041, 10.002, 10.003, 2., .3), clock[0])
    newer = core.snapshot()
    assert core.retire_for_lateral_handoff(10., expected_snapshot=checked) is newer
    assert core.snapshot() is newer and newer.uid == 2 and newer.active
    rt._service_short_follow()
    assert not driver.pairs and driver.stops


def test_cas_preserves_new_same_uid_depth_plan_and_all_watermarks(monkeypatch):
    _, _, _, clock, core, checked = pending_handoff(monkeypatch)
    clock[0] = 10.01
    fresh = core.update(ShortFollowObservation(1, 1040, 10., 10.005, 2., .35), clock[0])
    assert fresh is not None
    newer = core.snapshot()
    counters = (core._source_floor, core._last_depth, core._last_capture,
                core._last_capture_id, core._integral_m_s, core._integral_stamp)
    assert core.retire_for_lateral_handoff(10., expected_snapshot=checked) is newer
    assert core.snapshot().plan is fresh
    assert (core._source_floor, core._last_depth, core._last_capture,
            core._last_capture_id, core._integral_m_s, core._integral_stamp) == counters


def test_cas_requires_identical_snapshot_not_an_equal_reconstruction(monkeypatch):
    _, _, _, _, core, checked = pending_handoff(monkeypatch)
    assert replace(checked) == checked and replace(checked) is not checked
    assert core.retire_for_lateral_handoff(10., expected_snapshot=replace(checked)) is checked


def test_current_populated_snapshot_still_retires_and_repeated_call_is_noop(monkeypatch):
    rt, _, driver, _, core, checked = pending_handoff(monkeypatch)
    floor = core._source_floor
    retired = core.retire_for_lateral_handoff(10., expected_snapshot=checked)
    assert retired is not checked and not retired.active and retired.plan is None
    assert retired.reason == "lateral_handoff" and retired.epoch == checked.epoch + 1
    assert core._source_floor == floor
    assert core.retire_for_lateral_handoff(10.01) is retired
    assert core.retire_for_lateral_handoff(10.01, expected_snapshot=checked) is retired
    rt._service_short_follow()
    assert driver.pairs == [(-7, -7)] and not driver.stops


@pytest.mark.parametrize("reason", ["awaiting", "expired", "brake_wait"])
def test_empty_mailbox_cannot_become_a_new_lateral_handoff(reason):
    core = ShortFollowController(ShortFollowConfig(enabled=True))
    core.activate(1, 9.8)
    if reason != "awaiting":
        plan = core.update(ShortFollowObservation(1, 1039, 9.9, 9.9, 2., .3), 9.9)
        if reason == "expired":
            core.expire_observation(plan.expires_at + .001)
        else:
            core.wait_for_existing_brake(10.)
    old = core.snapshot()
    assert old.plan is None
    assert core.retire_for_lateral_handoff(10.3, expected_snapshot=old) is old
    assert core.retire_for_lateral_handoff(10.4) is old
