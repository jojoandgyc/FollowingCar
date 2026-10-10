"""Ordinary expired-plan retirement is not a new identity/safety event.

Pure controller tests using the CAP1245 -> 1252 -> 1254 physical clock values.
Identity/depth qualification stays with the adapter; no hardware is opened.
"""
from dataclasses import replace
import threading

import pytest

from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowObservation,
)


EXPIRED_AT = 40864.780


def scene():
    core = ShortFollowController(ShortFollowConfig(enabled=True, depth_ttl_sec=.35))
    core.activate(1, 40864.2)
    old = core.update(ShortFollowObservation(1, 1245, 40864.276640677,
        40864.45296511, 2.6308, .6667), 40864.625)
    assert old.expires_at == pytest.approx(40864.776640677)
    return core, old


def cap1254(depth=40864.889033229):
    return ShortFollowObservation(1, 1254, 40864.770776138, depth, 2.64499, .4719)


@pytest.mark.parametrize("deactivate", [False, True])
@pytest.mark.parametrize("depth", [40864.777090272, 40864.889033229])
def test_cap1254_new_qualified_sample_survives_ordinary_expiry_processing(deactivate, depth):
    core, old = scene()
    floor = core._source_floor
    before = core.snapshot()
    retired = core.expire_observation(EXPIRED_AT, deactivate=deactivate)
    assert retired.plan is None and retired.epoch == before.epoch+1
    assert retired.active is not deactivate
    assert core._source_floor == floor
    assert not old.valid(EXPIRED_AT)
    if deactivate:
        assert core.update(cap1254(depth), 40864.986) is None
    core.activate(1, 40864.986)
    assert core._source_floor == floor
    # The original image predates the expiry handling instant; the independent
    # range is newer than the retired sample, not borrowed from its old plan.
    observation = cap1254(depth)
    assert observation.capture_timestamp < EXPIRED_AT
    plan = core.update(observation, 40864.986)
    assert plan is not None and plan is not old and plan.forwarding
    assert plan.epoch == core.snapshot().epoch and plan.epoch > old.epoch
    assert plan.expires_at == min(depth+.35, observation.capture_timestamp+.50)
    assert plan.capture_timestamp == observation.capture_timestamp
    assert plan.depth_timestamp == depth
    assert not core.acknowledge_output(old, old.base_rpm)
    assert core.snapshot().plan is plan


def test_repeated_expiry_keeps_watermarks_and_does_not_restore_retired_pair():
    core, old = scene()
    watermarks = core._source_floor, core._last_depth, core._last_capture, core._last_capture_id
    retired = core.expire_observation(EXPIRED_AT, deactivate=True)
    for now in (EXPIRED_AT, EXPIRED_AT+.1, EXPIRED_AT+.4):
        assert core.expire_observation(now, deactivate=True) is retired
        assert core.snapshot().plan is None
        assert (core._source_floor, core._last_depth, core._last_capture, core._last_capture_id) == watermarks
    assert old.expires_at == pytest.approx(40864.776640677)


def test_soft_expired_active_owner_can_retire_ownership_without_hardening_floor():
    core, _ = scene()
    stopped = core.expire_observation(EXPIRED_AT)
    retired = core.expire_observation(EXPIRED_AT+.03, deactivate=True)
    assert stopped.active and not retired.active
    assert retired.epoch == stopped.epoch+1 and retired.plan is None
    assert core._source_floor == 0.
    core.activate(1, 40864.986)
    assert core.update(cap1254(), 40864.986) is not None


@pytest.mark.parametrize("case", ["duplicate_depth", "older_depth", "older_capture_id", "older_capture_stamp", "future", "expired"])
def test_expiry_does_not_relax_sample_order_or_source_deadlines(case):
    core, old = scene()
    core.expire_observation(EXPIRED_AT, deactivate=True)
    core.activate(1, EXPIRED_AT+.01)
    sample = cap1254(40864.777090272)
    now = 40864.800
    if case == "duplicate_depth": sample = replace(sample, depth_timestamp=old.depth_timestamp)
    elif case == "older_depth": sample = replace(sample, depth_timestamp=old.depth_timestamp-.001)
    elif case == "older_capture_id": sample = replace(sample, capture_id=1244)
    elif case == "older_capture_stamp": sample = replace(sample, capture_timestamp=old.capture_timestamp-.001)
    elif case == "future": sample = replace(sample, depth_timestamp=now+.001)
    else: now = sample.depth_timestamp+.35
    before = core.snapshot()
    assert core.update(sample, now) is None
    assert core.snapshot() is before and before.plan is None


def test_duplicate_new_sample_after_recovery_cannot_renew_deadline():
    core, _ = scene()
    core.expire_observation(EXPIRED_AT, deactivate=True)
    core.activate(1, 40864.986)
    recovered = core.update(cap1254(), 40864.986)
    assert recovered is not None
    assert core.update(cap1254(), 40865.0) is None
    assert core.update(replace(cap1254(), capture_id=1255,
                               capture_timestamp=40864.98), 40865.0) is None
    assert core.snapshot().plan is recovered


@pytest.mark.parametrize("reason", [
    "identity_rejected", "identity_or_search_handoff", "depth_safety_observation",
    "safety_or_shutdown", "manual_emergency", "feedback_reverse",
])
@pytest.mark.parametrize("deactivate", [False, True])
def test_explicit_identity_or_safety_revoke_still_requires_post_event_sources(reason, deactivate):
    core, _ = scene()
    (core.deactivate if deactivate else core.revoke)(reason, EXPIRED_AT)
    hard_floor = core._source_floor
    assert core.expire_observation(EXPIRED_AT+.01, deactivate=True).reason == reason
    core.activate(1, 40864.986)
    assert core._source_floor == hard_floor == EXPIRED_AT
    assert core.update(cap1254(), 40864.986) is None
    fresh = replace(cap1254(), capture_id=1257, capture_timestamp=40864.913982939)
    assert core.update(fresh, 40864.986) is not None


def test_later_hard_event_cannot_be_softened_by_repeated_expiry():
    core, _ = scene()
    core.expire_observation(EXPIRED_AT, deactivate=True)
    hard = core.revoke("manual_emergency", EXPIRED_AT+.02)
    assert core.expire_observation(EXPIRED_AT+.03, deactivate=True) is hard
    core.activate(1, 40864.986)
    assert core.update(cap1254(), 40864.986) is None
    assert core._source_floor == EXPIRED_AT+.02


def test_changed_uid_after_soft_expiry_establishes_new_hard_floor():
    core, _ = scene()
    core.expire_observation(EXPIRED_AT, deactivate=True)
    core.activate(2, 40864.9)
    assert core._source_floor == 40864.9
    assert core.update(cap1254(), 40864.986) is None
    assert core.update(replace(cap1254(), uid=2), 40864.986) is None
    core.activate(1, 40864.95)
    assert core._source_floor == 40864.95
    assert core.update(cap1254(), 40864.986) is None


def test_stale_expiry_call_does_not_retire_new_valid_plan_same_epoch():
    core, old = scene()
    new = core.update(cap1254(40864.777090272), EXPIRED_AT)
    assert new is not None and new.epoch == old.epoch
    state = core.snapshot()
    assert core.expire_observation(EXPIRED_AT, deactivate=True) is state
    assert core.snapshot().plan is new and state.active


def test_prepared_recovery_cannot_cross_concurrent_hard_revoke():
    core, _ = scene()
    core.expire_observation(EXPIRED_AT, deactivate=True)
    core.activate(1, 40864.82)
    original_lock = core._lock
    prepared, proceed = threading.Event(), threading.Event()
    class Gate:
        def __enter__(self):
            if threading.current_thread().name == "prepared-depth":
                prepared.set()
                assert proceed.wait(2.)
            return original_lock.__enter__()
        def __exit__(self, *args):
            return original_lock.__exit__(*args)
    core._lock = Gate()
    results = []
    producer = threading.Thread(name="prepared-depth", target=lambda:
        results.append(core.update(cap1254(), 40864.986)))
    producer.start()
    try:
        assert prepared.wait(2.)
        hard = core.revoke("identity_rejected", 40864.9)
    finally:
        proceed.set()
        producer.join(2.)
    assert not producer.is_alive() and results == [None]
    assert core.snapshot() is hard and hard.plan is None


@pytest.mark.parametrize("now", [float("nan"), float("inf"), True, 0., -1.])
def test_invalid_expiry_clock_cannot_change_state(now):
    core, _ = scene()
    before = core.snapshot()
    with pytest.raises(ValueError): core.expire_observation(now)
    assert core.snapshot() is before
