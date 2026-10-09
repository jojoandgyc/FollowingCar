"""Depth updates coalesce on control contention without queuing stale ROIs."""
from types import SimpleNamespace
import threading

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import DepthTargetObservation, PersonTarget


NOW = 100.0
BOX = (240., 40., 400., 440.)


class NonHangingLock:
    """Deterministic lock fixture: regressions fail instead of hanging pytest."""
    def __init__(self):
        self._lock = threading.Lock()

    def acquire(self, blocking=True):
        if self._lock.locked() and blocking:
            raise AssertionError("Depth30 attempted to block on the control lock")
        return self._lock.acquire(blocking=blocking)

    def release(self):
        self._lock.release()


def context(cap, stamp, *, uid=1):
    target = PersonTarget(
        BOX, uid, .95, 64000.,
        depth_observation=DepthTargetObservation(
            bbox=BOX, target_id=uid, raw_track_id=3,
            capture_frame_id=cap, capture_timestamp=stamp,
        ),
    )
    return dict(
        published_ts=NOW, frame_index=400, width=640, height=480,
        persons=[(BOX, uid, .95, 64000.)], person_targets=(target,),
        capture_frame_id=cap, capture_timestamp=stamp,
        target_id=uid, target_steerable=True,
    )


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", .25)
    obj = object.__new__(runtime.PersonTracker)
    obj._follow_controller = SimpleNamespace(active_target_id=1, search_state="none")
    obj.search_state = "none"
    obj.running = True
    obj._runtime_shutdown_requested = False
    obj._explicit_stop_requested = False
    obj._brake_hold_active = False
    obj._action_runtime_started = True
    obj._control_update_lock = NonHangingLock()
    obj._longitudinal_context_lock = threading.Lock()
    obj._longitudinal_context = context(1628, NOW - .1838)
    obj._last_longitudinal_stale_log_ts = NOW
    obj._longitudinal_stop_event = threading.Event()
    obj._depth30_linear_snapshot = ("forward", 32, 1, NOW - .15)
    obj._depth30_linear_timing = object()
    obj.calls = []
    obj._queue_actions_for_persons_locked = lambda *args, **kw: obj.calls.append(kw)
    return obj


def attempt(owner, ctx=None):
    ctx = owner._longitudinal_context if ctx is None else ctx
    return owner._queue_actions_for_persons(
        ctx["width"], ctx["height"], ctx["persons"], depth_use_latest=True,
        control_source="depth30", expected_target_id=ctx["target_id"],
        context_published_ts=ctx["published_ts"],
        depth_target_snapshot=ctx["person_targets"],
        evidence_capture_frame_id=ctx["capture_frame_id"],
        evidence_capture_timestamp=ctx["capture_timestamp"],
    )


def test_busy_control_never_waits_or_mutates_depth_motor_authority(owner):
    linear, timing = owner._depth30_linear_snapshot, owner._depth30_linear_timing
    owner._control_update_lock.acquire()
    try:
        # The deliberately held lock rejects any blocking request; the
        # depth worker must decline immediately without touching authority.
        assert attempt(owner) is False
        assert attempt(owner) is False
        assert owner.calls == []
        assert owner._depth30_linear_snapshot is linear
        assert owner._depth30_linear_timing is timing
        assert owner._depth30_deferred_schedule == (1628, NOW, 2)
    finally:
        owner._control_update_lock.release()
    assert attempt(owner) is True
    assert len(owner.calls) == 1
    assert owner._depth30_deferred_schedule is None


def test_loop_reloads_new_capture_after_contention_instead_of_old_waiting_roi(owner):
    owner._control_update_lock.acquire()
    waits = []

    def wait(duration):
        waits.append(duration)
        if len(waits) == 1:
            owner._longitudinal_context = context(1631, NOW - .08)
            owner._control_update_lock.release()
        else:
            owner._longitudinal_stop_event.set()

    owner._longitudinal_wake_event = SimpleNamespace(clear=lambda: None, wait=wait)
    owner._longitudinal_control_loop()
    assert waits[0] == pytest.approx(.005)
    assert waits[1] > .005
    assert len(owner.calls) == 1
    assert owner.calls[0]["evidence_capture_frame_id"] == 1631
    assert owner.calls[0]["evidence_capture_timestamp"] == NOW - .08


@pytest.mark.parametrize("revoke", ["clear", "replace", "uid", "search", "controller_search", "stop", "brake", "shutdown", "not_running", "expired", "future"])
def test_retry_rechecks_all_context_and_safety_conditions(owner, revoke):
    old = owner._longitudinal_context
    owner._control_update_lock.acquire()
    assert attempt(owner, old) is False
    owner._control_update_lock.release()
    if revoke == "clear":
        owner._longitudinal_context = None
    elif revoke == "replace":
        owner._longitudinal_context = context(1631, NOW - .08)
    elif revoke == "uid":
        owner._follow_controller.active_target_id = 2
    elif revoke == "search":
        owner.search_state = "searching"
    elif revoke == "controller_search":
        owner._follow_controller.search_state = "searching"
    elif revoke == "stop":
        owner._explicit_stop_requested = True
    elif revoke == "brake":
        owner._brake_hold_active = True
    elif revoke == "shutdown":
        owner._runtime_shutdown_requested = True
    elif revoke == "not_running":
        owner.running = False
    elif revoke in {"expired", "future"}:
        old = context(1628, NOW - .251 if revoke == "expired" else NOW + .001)
        owner._longitudinal_context = old
    assert attempt(owner, old) is True  # terminal reject; do not fast retry it
    assert owner.calls == []
    assert owner._depth30_linear_snapshot == ("forward", 32, 1, NOW - .15)


def test_duplicate_rgb_can_still_carry_new_physical_depth_sample(owner):
    # Scheduling does not deduplicate by CAP: the sensor handles physical
    # timestamp replay, retaining its original motion deadline.
    assert attempt(owner) is True
    assert attempt(owner) is True
    assert len(owner.calls) == 2


def test_control_lock_released_if_ranging_raises(owner):
    def fail(*args, **kwargs):
        raise ValueError("simulated depth failure")

    owner._queue_actions_for_persons_locked = fail
    with pytest.raises(ValueError, match="simulated depth"):
        attempt(owner)
    assert owner._control_update_lock.acquire(blocking=False)
    owner._control_update_lock.release()


def test_visual_safety_decision_uses_normal_blocking_serialization(owner):
    class LockProbe:
        def __init__(self):
            self.calls = []

        def acquire(self, *, blocking):
            self.calls.append(blocking)
            return True

        def release(self):
            self.calls.append("release")

    lock = LockProbe()
    owner._control_update_lock = lock
    assert owner._queue_actions_for_persons(640, 480, [], control_source="vision") is True
    assert lock.calls == [True, "release"]
    assert len(owner.calls) == 1


def test_stop_during_deferred_wait_does_not_start_ranging(owner):
    owner._control_update_lock.acquire()
    owner._longitudinal_wake_event = SimpleNamespace(
        clear=lambda: None,
        wait=lambda duration: owner._longitudinal_stop_event.set(),
    )
    try:
        owner._longitudinal_control_loop()
    finally:
        owner._control_update_lock.release()
    assert owner.calls == []
