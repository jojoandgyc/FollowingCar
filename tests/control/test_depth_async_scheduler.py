"""No hardware: immutable/coalesced ROI, physical scan ownership and health."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from car_control_modular.control_types import DepthTargetObservation, PersonTarget
from car_control_modular.depth_async_scheduler import DepthAsyncScheduler


def context(cap=100, stamp=99.9, uid=1):
    bbox = (100., 50., 300., 400.)
    observation = DepthTargetObservation(bbox, uid, 7, cap, stamp)
    target = PersonTarget(bbox, uid, .94, 70000., observation)
    return dict(target_id=uid, capture_frame_id=cap, capture_timestamp=stamp,
                published_ts=100., frame_index=80, width=640, height=480,
                persons=[(bbox, uid, .94, 70000.)], person_targets=(target,),
                target_steerable=True)


@pytest.fixture
def scheduler():
    return DepthAsyncScheduler()


def begin(scheduler, *, now=100.):
    return scheduler.begin(now=now, max_capture_age_sec=.25)


def test_same_uid_new_capture_does_not_cancel_fixed_inflight(scheduler):
    old = scheduler.submit(context(), now=100.)
    ticket = begin(scheduler)
    new = scheduler.submit(context(101, 99.95), now=100.01)
    assert new.epoch == old.epoch and new.sequence > old.sequence
    assert ticket.context is old
    assert ticket.context.capture_frame_id == 100
    assert ticket.context.capture_timestamp == 99.9
    assert scheduler.valid(ticket, now=100.02)
    assert begin(scheduler, now=100.02) is None
    assert scheduler.finish(ticket, now=100.03)
    assert begin(scheduler, now=100.03).context is new


def test_many_pending_frames_coalesce_to_newest_only(scheduler):
    scheduler.submit(context(), now=100.)
    ticket = begin(scheduler)
    for cap in range(101, 150):
        latest = scheduler.submit(context(cap, 99.9 + (cap - 100) / 1000.), now=100.01)
    assert scheduler.valid(ticket)
    scheduler.finish(ticket, now=100.02)
    next_ticket = begin(scheduler, now=100.02)
    assert next_ticket.context is latest
    assert next_ticket.context.capture_frame_id == 149


def test_latest_roi_can_be_reused_for_distinct_depth_samples(scheduler):
    original = scheduler.submit(context(), now=100.)
    first = begin(scheduler)
    scheduler.finish(first, now=100.01)
    second = begin(scheduler, now=100.02)
    assert first.sequence != second.sequence
    assert first.context is second.context is original
    assert not scheduler.valid(first)
    # Selecting/committing the actual different Depth samples is downstream.
    assert scheduler.valid(second)


@pytest.mark.parametrize("cap, stamp", [(100, 99.9), (99, 99.89), (101, 99.89), (99, 99.95)])
def test_duplicate_or_older_roi_never_refreshes_time_or_geometry(scheduler, cap, stamp):
    original = scheduler.submit(context(), now=100.)
    data = context(cap, stamp)
    data["published_ts"] = 100.1
    result = scheduler.submit(data, now=100.1)
    assert result is original
    assert result.published_ts == 100.
    assert begin(scheduler, now=100.151) is None


def test_frozen_context_does_not_alias_mutable_geometry(scheduler):
    data = context()
    old_target = data["person_targets"][0]
    mutable_box = list(old_target.bbox)
    mutable_obs_box = list(old_target.depth_observation.bbox)
    data["person_targets"] = [replace(old_target, bbox=mutable_box,
        depth_observation=replace(old_target.depth_observation, bbox=mutable_obs_box))]
    frozen = scheduler.submit(data, now=100.)
    mutable_box[0] = 999
    mutable_obs_box[0] = 999
    data["person_targets"].clear()
    data["target_id"] = 8
    assert frozen.person_targets[0].bbox[0] == 100
    assert frozen.person_targets[0].depth_observation.bbox[0] == 100
    copied = frozen.as_dict()
    copied["capture_timestamp"] = 100.1
    assert frozen.capture_timestamp == 99.9
    with pytest.raises(FrozenInstanceError):
        frozen.epoch = 2


@pytest.mark.parametrize("reason", ["identity_conflict", "stop", "shutdown", "hazard", "worker_timeout"])
def test_revoke_invalidates_commit_but_keeps_physical_slot(scheduler, reason):
    scheduler.submit(context(), now=100.)
    ticket = begin(scheduler)
    scheduler.revoke(reason)
    assert not scheduler.valid(ticket)
    scheduler.submit(context(101, 99.95), now=100.01)
    assert begin(scheduler, now=100.01) is None
    assert scheduler.begin_fallback(now=100.01) is None
    assert scheduler.finish(ticket, now=100.02)
    fallback = scheduler.begin_fallback(now=100.03)
    assert fallback is not None and fallback.owner == "vision_fallback"


def test_uid_change_invalidates_old_commit_without_starting_parallel_scan(scheduler):
    scheduler.submit(context(), now=100.)
    ticket = begin(scheduler)
    replacement = scheduler.submit(context(101, 99.95, uid=2), now=100.01)
    assert replacement.epoch > ticket.epoch
    assert not scheduler.valid(ticket)
    assert begin(scheduler, now=100.01) is None
    scheduler.finish(ticket, now=100.02)
    assert begin(scheduler, now=100.02).context.target_id == 2


def test_stop_clear_republish_same_uid_does_not_resurrect_old_ticket(scheduler):
    scheduler.submit(context(), now=100.)
    ticket = begin(scheduler)
    scheduler.revoke("stop")
    scheduler.submit(context(), now=100.01)
    assert not scheduler.valid(ticket)


def test_begin_checks_roi_age_commit_can_use_separately_verified_sample(scheduler):
    scheduler.submit(context(), now=100.)
    ticket = begin(scheduler, now=100.14)
    assert ticket is not None
    assert scheduler.valid(ticket, now=100.20)
    assert not scheduler.valid(ticket, now=100.20, max_capture_age_sec=.25)
    scheduler.finish(ticket, now=100.20)
    assert begin(scheduler, now=100.20) is None
    # No scheduler action alters the original capture time.
    assert ticket.context.capture_timestamp == 99.9


def test_fallback_before_uid_exists_reserves_only_scan_not_motion(scheduler):
    ticket = scheduler.begin_fallback(now=100.)
    assert ticket.context is None
    assert scheduler.valid(ticket)
    scheduler.submit(context(), now=100.01)
    assert begin(scheduler, now=100.01) is None
    assert scheduler.finish(ticket, now=100.02)
    assert begin(scheduler, now=100.02) is not None


def test_fallback_context_not_automatically_published_to_worker(scheduler):
    scheduler.submit(context(), now=100.)
    fallback = scheduler.begin_fallback(context(101, 99.95), now=100.01,
                                        max_capture_age_sec=.25)
    assert fallback.context.capture_frame_id == 101
    assert scheduler.valid(fallback)
    scheduler.finish(fallback, now=100.02)
    assert begin(scheduler, now=100.02) is None


def test_old_finish_cannot_release_new_flight(scheduler):
    scheduler.submit(context(), now=100.)
    first = begin(scheduler)
    scheduler.finish(first, now=100.01)
    second = begin(scheduler, now=100.02)
    assert not scheduler.finish(first, now=100.03)
    assert scheduler.valid(second)
    # A structurally identical dataclass is not possession of the lease.
    assert not scheduler.valid(replace(second))
    assert not scheduler.finish(replace(second), now=100.03)


@pytest.mark.parametrize("key, value", [
    ("capture_timestamp", float("nan")), ("capture_timestamp", 100.1),
    ("capture_timestamp", -1), ("capture_frame_id", 0), ("target_id", -1),
    ("width", 0), ("height", 0), ("published_ts", 100.1),
    ("person_targets", ()),
])
def test_invalid_context_is_rejected_without_revoking_good_flight(scheduler, key, value):
    scheduler.submit(context(), now=100.)
    flight = begin(scheduler)
    bad = context(101, 99.95)
    bad[key] = value
    with pytest.raises(ValueError):
        scheduler.submit(bad, now=100.)
    assert scheduler.valid(flight)


@pytest.mark.parametrize("change", [
    {"target_id": 2}, {"capture_frame_id": 99}, {"capture_timestamp": 99.8},
    {"source": "predicted"},
])
def test_context_proof_must_match_original_observation(scheduler, change):
    data = context()
    target = data["person_targets"][0]
    data["person_targets"] = (replace(target,
        depth_observation=replace(target.depth_observation, **change)),)
    with pytest.raises(ValueError, match="detector proof"):
        scheduler.submit(data, now=100.)


def test_stale_fallback_does_not_take_a_physical_slot(scheduler):
    assert scheduler.begin_fallback(context(), now=100.2, max_capture_age_sec=.25) is None
    assert scheduler.begin_fallback(now=100.2) is not None


def test_heartbeat_not_alive_and_not_started_are_unhealthy(scheduler):
    assert scheduler.health(now=100., worker_alive=True).reason == "worker_not_started"
    scheduler.worker_tick(now=100.)
    assert scheduler.health(now=100., worker_alive=False).reason == "worker_not_alive"
    assert scheduler.health(now=100., worker_alive=True).healthy
    assert scheduler.health(now=100.251, worker_alive=True).reason == "worker_silent"


def test_busy_timeout_cannot_be_masked_by_repeated_heartbeat(scheduler):
    scheduler.submit(context(), now=100.)
    flight = begin(scheduler)
    scheduler.worker_tick(now=100.26)
    state = scheduler.health(now=100.26, worker_alive=True)
    assert not state.healthy and state.reason == "worker_busy_timeout"
    assert state.busy and state.busy_age_sec == pytest.approx(.26)
    assert scheduler.valid(flight)
    # The owner decides fallback/revoke. Health never grants or revokes motion.


def test_normal_busy_does_not_require_heartbeat_from_inside_calculation(scheduler):
    scheduler.submit(context(), now=100.)
    begin(scheduler)
    state = scheduler.health(now=100.2, worker_alive=True, max_silence_sec=.1,
                             max_busy_sec=.25)
    assert state.healthy and state.reason == "working"


def test_repeated_new_roi_and_heartbeat_cannot_mask_worker_no_progress(scheduler):
    scheduler.worker_tick(now=100.)
    scheduler.submit(context(), now=100.)
    for index in range(1, 4):
        stamp = 100. + .1 * index
        scheduler.worker_tick(now=stamp)
        data = context(100 + index, stamp - .01)
        data["published_ts"] = stamp
        scheduler.submit(data, now=stamp)
    state = scheduler.health(now=100.3, worker_alive=True)
    assert not state.healthy and state.reason == "worker_no_progress"


def test_errors_and_success_are_progress_not_thread_liveness(scheduler):
    scheduler.submit(context(), now=100.)
    for now in (100., 100.02):
        flight = begin(scheduler, now=now)
        scheduler.finish(flight, now=now + .01, error=True)
    state = scheduler.health(now=100.03, worker_alive=True)
    assert state.reason == "worker_errors" and state.consecutive_errors == 2
    flight = begin(scheduler, now=100.04)
    scheduler.finish(flight, now=100.05)
    state = scheduler.health(now=100.05, worker_alive=True)
    assert state.healthy and state.consecutive_errors == 0
    assert state.progress_age_sec == 0


def test_fallback_finish_does_not_make_dead_worker_healthy(scheduler):
    fallback = scheduler.begin_fallback(now=100.)
    scheduler.finish(fallback, now=100.01)
    assert scheduler.health(now=100.01, worker_alive=True).reason == "worker_not_started"


def test_revoked_busy_and_fallback_busy_health(scheduler):
    scheduler.submit(context(), now=100.)
    flight = begin(scheduler)
    scheduler.revoke("hazard")
    assert scheduler.health(now=100.01, worker_alive=True).reason == "revoked_busy"
    scheduler.finish(flight, now=100.02)
    scheduler.begin_fallback(now=100.03)
    assert scheduler.health(now=100.03, worker_alive=True).reason == "fallback_busy"


@pytest.mark.parametrize("method, kwargs", [
    ("worker_tick", {"now": float("nan")}),
    ("health", {"now": 100., "worker_alive": True, "max_busy_sec": 0}),
    ("health", {"now": 100., "worker_alive": True, "max_errors": 0}),
    ("health", {"now": 100., "worker_alive": True, "max_errors": 1.5}),
    ("begin", {"now": 100., "max_capture_age_sec": float("inf")}),
])
def test_invalid_health_or_age_configuration_rejected(scheduler, method, kwargs):
    with pytest.raises(ValueError):
        getattr(scheduler, method)(**kwargs)


def test_time_reversal_cannot_validate_or_release_task(scheduler):
    scheduler.submit(context(), now=100.)
    flight = begin(scheduler)
    assert not scheduler.valid(flight, now=99.99)
    assert not scheduler.finish(flight, now=99.99)
    assert scheduler.health(now=99.99, worker_alive=True).reason == "clock_invalid"
    assert scheduler.valid(flight, now=100.01)


def test_threads_racing_to_start_have_one_physical_owner(scheduler):
    scheduler.submit(context(), now=100.)
    barrier = threading.Barrier(12)
    tickets = []
    result_lock = threading.Lock()

    def attempt():
        barrier.wait(timeout=2)
        ticket = begin(scheduler)
        with result_lock:
            tickets.append(ticket)

    workers = [threading.Thread(target=attempt) for _ in range(12)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert len([ticket for ticket in tickets if ticket is not None]) == 1


def test_actual_blocked_worker_revocation_never_starts_fallback_in_parallel(scheduler):
    scheduler.submit(context(), now=100.)
    scanning = threading.Event()
    allow_finish = threading.Event()
    finished = threading.Event()
    results = []

    def worker():
        ticket = begin(scheduler)
        scanning.set()
        assert allow_finish.wait(timeout=2)
        results.append(scheduler.valid(ticket, now=100.03))
        scheduler.finish(ticket, now=100.03)
        finished.set()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert scanning.wait(timeout=2)
        scheduler.submit(context(101, 99.95), now=100.01)
        assert scheduler.begin_fallback(now=100.02) is None
        assert begin(scheduler, now=100.02) is None
        allow_finish.set()
        assert finished.wait(timeout=2)
        fallback = scheduler.begin_fallback(now=100.04)
        assert fallback is not None
        assert results == [False]
    finally:
        allow_finish.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_none_or_forged_ticket_does_not_own_idle_scan(scheduler):
    assert not scheduler.valid(None)
    assert not scheduler.finish(None, now=100.)
    with pytest.raises(ValueError):
        scheduler.valid(None, max_capture_age_sec=.25)

