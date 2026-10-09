"""PI -> physical Depth admission -> final wheel commit, without hardware.

Publication is injected at the serial-commit boundary, after the ordinary
planner has finished. The oracle is the actual emitted wheel stream, not a
helper's successful return value.
"""
from dataclasses import replace

import pytest

from car_control_modular.controllers import FollowSafetyController
from car_control_modular.mssd_motor import MotorSpeedReceipt
from car_control_modular.sample_braking import SampleBrakingAssessment
from test_depth_authority_250 import authority, advance, decide_commit, writer
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


@pytest.fixture
def chain(authority):
    a = authority
    a.controller = FollowSafetyController(replace(
        a.controller.cfg,
        distance_target_motion_control_enable=False,
        distance_pi_stationary_stop_preview_enabled=False,
    ))
    a.controller.active_target_id = 1
    a.controller._has_seen_person = True
    a.owner._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    a.controller._braking_execution_bound_reader = lambda uid, now: 20.
    for _ in range(4):
        _, actions, accepted = decide_commit(a, a.frame(3.5, rpm=20.))
        assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
        advance(a, a.clock.now + .05)
    a.action, a.backend = writer(a)
    a.action.get_steering_feedback = lambda: a.feedback
    a.backend.last_speed_receipt = None
    a.backend.stop_write_generation = 0
    a.backend.stops = []
    send = a.backend.send_targets

    def recorded(left, right, label, **kwargs):
        send(left, right, label, **kwargs)
        a.backend.last_speed_receipt = MotorSpeedReceipt(
            len(a.backend.pairs), left, right, a.clock.now)

    a.backend.send_targets = recorded

    def stopped(label, **kwargs):
        a.backend.stops.append((label, kwargs))
        a.backend.stop_write_generation += 1
        a.backend.normal_zero_hold = True

    a.backend.send_stop = stopped
    a.controller._braking_execution_bound_reader = (
        a.action.continuation_executed_speed_bound_rpm)
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][0] > 0 and a.backend.pairs[-1][1] < 0
    return a


@pytest.mark.parametrize("distance", [3.0, 3.5, 4.0])
def test_new_real_pi_grant_at_commit_does_not_insert_zero(chain, monkeypatch, distance):
    a = chain
    original_timing = a.owner._depth30_linear_timing
    advance(a, a.clock.now + .05)
    begin = a.action._begin_follow_commit
    injected = []

    def publish_before_commit():
        if not injected:
            assert not a.owner.motor_io_lock.locked()
            # This is a genuinely new captured sample, not a tuple rewritten
            # to pretend an expired grant has become fresh.
            current = a.frame(distance, rpm=20.)
            _, actions, accepted = decide_commit(a, current)
            assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
            timing = a.owner._depth30_linear_timing
            assert timing is not original_timing
            assert isinstance(timing.braking_assessment, SampleBrakingAssessment)
            assert timing.braking_assessment.sample_timestamp == current.distance_state.sample_timestamp
            injected.append(timing)
        return begin()

    monkeypatch.setattr(a.action, "_begin_follow_commit", publish_before_commit)
    a.action._service_follow_wheels()
    assert len(injected) == 1
    assert len(a.backend.pairs) == 2
    assert all(left > 0 and right < 0 for left, right, _ in a.backend.pairs)
    latest = a.owner._fresh_depth_linear_snapshot(1, quiet=True)
    assert latest is not None
    left, raw_right, _ = a.backend.pairs[-1]
    assert .5 * (left - raw_right) <= latest[1] * 2.
    assert a.owner._depth30_linear_timing is injected[0]
    assert injected[0].depth_expires_at == pytest.approx(latest[3] + .25)

    # A successful handoff must not make its new grant immortal.
    advance(a, injected[0].depth_expires_at + .000001)
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][:2] == (0, 0)


@pytest.mark.parametrize("fault", ["near", "hazard", "identity", "expired"])
def test_real_authority_loss_at_commit_still_stops(chain, monkeypatch, fault):
    a = chain
    advance(a, a.clock.now + .05)
    begin = a.action._begin_follow_commit
    injected = []

    def invalidate_before_commit():
        if not injected:
            injected.append(fault)
            current = a.frame(1.0 if fault == "near" else 3.5, rpm=20.)
            if fault == "hazard":
                current = replace(current, hazard=replace(current.hazard, active=True, reason="test"))
                # The live hazard latch is independent of the queued Depth
                # decision. An emergency STOP, not a normal zero-speed
                # packet, is the correct physical outcome here.
                a.action.hard_stop_check = lambda _: True
            elif fault == "identity":
                a.controller.active_target_id = 2
                current = a.frame(3.5, rpm=20., uid=2)
            elif fault == "expired":
                advance(a, a.owner._depth30_linear_timing.depth_expires_at + .000001)
                return begin()
            decide_commit(a, current)
        return begin()

    monkeypatch.setattr(a.action, "_begin_follow_commit", invalidate_before_commit)
    a.action._service_follow_wheels()
    assert injected == [fault]
    if fault == "hazard":
        assert len(a.backend.pairs) == 1
        assert a.backend.stops[-1][1]["mode"] == "emergency"
    else:
        assert a.backend.pairs[-1][:2] == (0, 0)


def test_repeated_commit_handoffs_keep_a_nonzero_bounded_stream(chain, monkeypatch):
    a = chain
    begin = a.action._begin_follow_commit
    pending = []
    accepted_stamps = []

    def publish_once_per_tick():
        if pending:
            distance = pending.pop()
            old = a.owner._depth30_linear_timing
            _, actions, accepted = decide_commit(a, a.frame(distance, rpm=20.))
            assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
            timing = a.owner._depth30_linear_timing
            assert timing.accepted_depth_timestamp > old.accepted_depth_timestamp
            assert timing.depth_expires_at == pytest.approx(timing.accepted_depth_timestamp + .25)
            accepted_stamps.append(timing.accepted_depth_timestamp)
        return begin()

    monkeypatch.setattr(a.action, "_begin_follow_commit", publish_once_per_tick)
    for distance in (3.5, 3.2, 3.8, 3.1) * 3:
        advance(a, a.clock.now + .06)
        pending.append(distance)
        before = len(a.backend.pairs)
        a.action._service_follow_wheels()
        assert not pending
        assert len(a.backend.pairs) == before + 1
        left, raw_right, _ = a.backend.pairs[-1]
        assert left > 0 and raw_right < 0
        grant = a.owner._fresh_depth_linear_snapshot(1, quiet=True)
        assert .5 * (left - raw_right) <= grant[1] * 2.
    assert len(accepted_stamps) == 12
    assert len(set(accepted_stamps)) == 12
    assert not any(left == right == 0 for left, right, _ in a.backend.pairs)
