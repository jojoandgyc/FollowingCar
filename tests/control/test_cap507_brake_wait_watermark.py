"""CAP505/507: settling waits must not reject already post-STOP captures.

Real paired adapter/controller, in-memory identity and physical-release stub;
no camera, NPU, serial port or motor command is opened by these tests.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import DepthTargetObservation
from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController, ShortFollowObservation
from test_short_follow_acceptance import _adapter_chain, _adapter_frame, _deliver, _writer_chain


STOP = 22563.359315
WAIT = 22564.15
CAP507 = 22564.140532126


@pytest.fixture
def handoff(monkeypatch):
    chain = _adapter_chain()
    chain.now = WAIT
    monkeypatch.setattr("car_control_modular.detector_identity_lease.time.monotonic", lambda: chain.now)
    chain.controller.config = replace(chain.controller.config, depth_ttl_sec=.35)
    chain.controller.activate(1, STOP-.1)
    chain.controller.deactivate("identity_or_search_handoff", STOP)
    chain.owner._brake_hold_active = True
    chain.owner._brake_hold_label = "search_reacquire_brake"
    chain.ready = False
    chain.release_calls = []

    def release(**values):
        chain.release_calls.append(values)
        if chain.ready:
            chain.owner._brake_hold_active = False
        return chain.ready

    chain.owner._action_runtime = SimpleNamespace(
        search_reacquire_brake_pending=lambda **_: False,
        release_settled_search_brake_for_depth=release)
    return chain


def frame_at(chain, cap, capture, sample, distance=2.575):
    frame, target = _adapter_frame(chain, capture_id=cap, distance=distance,
                                  center=.355, new_identity=False, sample_stamp=sample)
    frame.capture_timestamp = capture
    frame.height = 480
    target.bbox = (182., 10., 272., 411.)
    target.depth_observation = DepthTargetObservation(target.bbox, 1, 1, cap, capture)
    frame.persons = [target]
    chain.owner._validated_visual_observation = ValidatedVisualObservation(
        1, 1, cap, capture, chain.now, capture+.5, "full")
    return frame, target


def start_wait(chain):
    # CAP505: UID is back but re-anchoring Depth is still pending.
    frame, target = frame_at(chain, 505, 22564.039416731, None, distance=None)
    assert _deliver(chain, frame, target, fresh=False)
    state = chain.controller.snapshot()
    assert state.active and state.plan is None
    assert state.reason == "await_existing_brake_completion"
    return state


@pytest.mark.parametrize("sample,now", [
    (22564.129920391, 22564.286),
    (22564.277015306, 22564.325),
    (22564.308540478, 22564.371),
    (22564.340394351, 22564.396),
])
def test_cap507_post_stop_capture_is_not_rejected_by_later_wait_processing(handoff, sample, now):
    chain = handoff
    waiting = start_wait(chain)
    assert chain.controller._source_floor == STOP
    chain.now = now
    chain.ready = True
    frame, target = frame_at(chain, 507, CAP507, sample)
    assert CAP507 < WAIT  # Reproduces the original rejection boundary.
    assert _deliver(chain, frame, target)
    state = chain.controller.snapshot()
    plan = state.plan
    assert plan is not None and plan.forwarding
    assert state.epoch == waiting.epoch
    assert plan.capture_id == 507 and plan.capture_timestamp == CAP507
    assert plan.depth_timestamp == sample
    assert plan.expires_at == pytest.approx(min(sample+.35, CAP507+.5))
    assert plan.integral_dt_sec == 0
    assert not chain.owner._brake_hold_active
    assert len(chain.release_calls) == 1


def test_repeated_wait_is_idempotent_but_does_not_release_physical_brake(handoff):
    chain = handoff
    waiting = start_wait(chain)
    for index in range(3):
        chain.now = WAIT+.025*(index+1)
        frame, target = frame_at(chain, 507, CAP507, chain.now-.01)
        assert _deliver(chain, frame, target)
        assert chain.controller.snapshot() is waiting
        assert chain.controller._source_floor == STOP
        assert chain.owner._brake_hold_active
    chain.ready = True
    chain.now += .025
    frame, target = frame_at(chain, 507, CAP507, chain.now-.01)
    assert _deliver(chain, frame, target)
    assert chain.controller.snapshot().plan.forwarding


@pytest.mark.parametrize("hard_rejected", [False, True])
def test_real_writer_cap507_release_writes_fresh_pair_not_stale_await_stop(monkeypatch, hard_rejected):
    chain = _writer_chain(monkeypatch)
    chain.owner._action_runtime = chain.runtime
    chain.controller.config = replace(chain.controller.config, depth_ttl_sec=.35)
    chain.now = STOP
    chain.controller.activate(1, STOP-.1)
    chain.controller.deactivate("identity_or_search_handoff", STOP)
    # An inherited physical STOP was already acknowledged by the fake driver.
    chain.runtime.backend.send_stop("search_reacquire_brake", mode="emergency")
    chain.owner._brake_hold_active = True
    chain.owner._brake_hold_label = "search_reacquire_brake"
    chain.runtime.search_reacquire_brake_pending = lambda **_: False
    chain.runtime.release_settled_search_brake_for_depth = lambda **_: False
    chain.now = WAIT
    start_wait(chain)
    if hard_rejected:
        chain.controller.revoke("identity_conflict", WAIT+.01)

    def released(**_):
        chain.owner._brake_hold_active = False
        return True

    chain.runtime.release_settled_search_brake_for_depth = released
    chain.now = 22564.286
    chain.feedback = SimpleNamespace(timestamp=chain.now, left_forward_rpm=0.,
                                     right_forward_rpm=0., trustworthy=True)
    frame, target = frame_at(chain, 507, CAP507, 22564.129920391)
    assert _deliver(chain, frame, target)
    stops_before = list(chain.driver.stops)
    assert chain.runtime._service_short_follow()
    if hard_rejected:
        assert chain.controller.snapshot().plan is None
        assert not chain.driver.pairs
    else:
        plan = chain.controller.snapshot().plan
        assert plan is not None and plan.forwarding
        assert chain.driver.pairs == [(plan.left_rpm, -plan.right_rpm)]
        assert plan.left_rpm > 0 and plan.right_rpm > 0
        assert chain.driver.stops == stops_before
        assert all(left != 0 or right != 0 for left, right in chain.driver.pairs)


@pytest.mark.parametrize("hard_event", ["identity_conflict", "depth_safety_observation", "front_ir"])
def test_later_hard_event_still_rejects_capture_from_before_that_event(handoff, hard_event):
    chain = handoff
    start_wait(chain)
    hard_time = WAIT+.01
    chain.controller.revoke(hard_event, hard_time)
    chain.now = WAIT+.02
    frame, target = frame_at(chain, 507, CAP507, chain.now-.005)
    assert _deliver(chain, frame, target)  # Still awaiting actual release.
    assert chain.controller._source_floor == hard_time
    chain.ready = True
    chain.now += .03
    frame, target = frame_at(chain, 507, CAP507, chain.now-.005)
    assert _deliver(chain, frame, target)
    assert chain.controller.snapshot().plan is None
    # Fresh independent identity/capture/depth after the hard event can recover.
    chain.now += .05
    frame, target = frame_at(chain, 510, chain.now-.03, chain.now-.01)
    assert _deliver(chain, frame, target)
    assert chain.controller.snapshot().plan.forwarding


def test_unrelated_safety_hold_keeps_hard_revocation(handoff):
    chain = handoff
    chain.owner._brake_hold_label = "safety_hold_hazard"
    start_wait(chain)
    assert chain.controller._source_floor == WAIT
    chain.owner._brake_hold_active = False
    chain.now += .15
    frame, target = frame_at(chain, 507, CAP507, chain.now-.01)
    assert _deliver(chain, frame, target)
    assert chain.controller.snapshot().plan is None
    assert not chain.release_calls


@pytest.mark.parametrize("bad", ["stale_capture", "stale_depth", "future_depth", "rejected_identity"])
def test_soft_wait_does_not_make_invalid_evidence_legal(handoff, bad):
    chain = handoff
    start_wait(chain)
    chain.ready = True
    chain.now += .20
    capture = chain.now-.6 if bad == "stale_capture" else CAP507
    sample = chain.now-.4 if bad == "stale_depth" else chain.now+.01 if bad == "future_depth" else chain.now-.01
    frame, target = frame_at(chain, 507, capture, sample)
    if bad == "rejected_identity":
        chain.owner._validated_visual_observation = False
    _deliver(chain, frame, target)
    assert chain.controller.snapshot().plan is None


def test_wait_retires_old_pair_without_restoring_duplicate_depth_or_extending_lease():
    core = ShortFollowController(ShortFollowConfig(enabled=True))
    core.activate(1, 10.)
    obs = ShortFollowObservation(1, 1, 10.01, 10.02, 2., .5, 2.)
    old_plan = core.update(obs, 10.03)
    waiting = core.wait_for_existing_brake(10.04)
    assert waiting.epoch > old_plan.epoch and waiting.plan is None
    assert core.wait_for_existing_brake(10.05) is waiting
    assert core.update(obs, 10.06) is None
    assert not core.acknowledge_output(old_plan, old_plan.base_rpm)
    new = core.update(replace(obs, capture_id=2, capture_timestamp=10.035,
                              depth_timestamp=10.055), 10.06)
    assert new is not None and new is not old_plan
    assert new.expires_at == pytest.approx(10.355)
    assert new.integral_dt_sec == 0
    core.wait_for_existing_brake(10.07)
    core.deactivate("identity_conflict", 10.08)
    inactive = core.snapshot()
    assert core.wait_for_existing_brake(10.09) is inactive
    assert not inactive.active
    core.activate(2, 10.1)
    assert core.update(replace(obs, uid=2, capture_id=3, capture_timestamp=10.09,
                              depth_timestamp=10.11), 10.12) is None


@pytest.mark.parametrize("now", [None, 0., -1., float("nan"), float("inf"), True])
def test_wait_requires_valid_monotonic_clock(now):
    core = ShortFollowController(ShortFollowConfig(enabled=True))
    with pytest.raises(ValueError):
        core.wait_for_existing_brake(now)
