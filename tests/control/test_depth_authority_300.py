"""300 ms forward lease through real PI, admission and fake wheel writer.

Only the deadline for a previously admitted safe forward sample is extended.
These tests never mock the final authority or physical-age checks.
"""
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision
from test_depth_authority_250 import advance, decide_commit, writer
from test_distance_pi_controller import configured, integral
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


@pytest.fixture
def authority300(setup, owner, monkeypatch):
    clock, controller, frame = configured(
        setup, depth_longitudinal_sample_max_age_sec=.30,
        distance_target_motion_control_enable=False,
        distance_pi_braking_stop_distance_m=1.1,
    )
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC", .30)
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 200)
    monkeypatch.setattr(runtime, "MOTOR_FORWARD_MAX_TARGET_RPM", 200)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT", 100)
    monkeypatch.setattr(runtime, "MOTOR_RS485_TARGET_MIN_INTERVAL_SEC", .05)
    owner._follow_controller = controller
    owner._lateral_yaw_revision = 1
    controller._live_longitudinal_authority_reader = owner._fresh_depth_linear_snapshot
    a = SimpleNamespace(clock=clock, controller=controller, frame=frame, owner=owner,
                        feedback=None)
    a.action, a.backend = writer(a)
    a.action.get_steering_feedback = lambda: a.feedback
    controller._braking_execution_bound_reader = (
        lambda uid, now: a.action.continuation_executed_speed_bound_rpm(uid, now))
    return a


def seed300(a, *, distance=3., rpm=40.):
    current = a.frame(distance, rpm=rpm)
    _, actions, accepted = decide_commit(a, current)
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    stamp = current.distance_state.sample_timestamp
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .30)
    assert a.controller._distance_pid._distance_pi.config.physical_ttl_sec == .30
    return stamp, a.owner._depth30_linear_snapshot


def writer300(a):
    return a.action, a.backend


@pytest.mark.parametrize("rpm", [20., 40.])
@pytest.mark.parametrize("age, live", [(.249, True), (.251, True), (.275, True),
                                       (.299, True), (.301, False)])
def test_original_forward_grant_uses_300ms_without_renewal_or_acceleration(authority300, rpm, age, live):
    a = authority300
    stamp, original = seed300(a, rpm=rpm)
    initial_integral = integral(a.controller)
    action, backend = writer300(a)
    advance(a, stamp + age)
    current = a.owner._fresh_depth_linear_snapshot(1)
    assert (current is not None) is live
    if live:
        assert current[3] == stamp and 0 < current[1] <= original[1]
    action._service_follow_wheels()
    if live:
        assert backend.pairs[-1][0] > 0 and backend.pairs[-1][1] < 0
        assert max(abs(v) for v in backend.pairs[-1][:2]) <= original[1] * 2
    else:
        assert backend.pairs[-1][:2] == (0, 0)
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .30)
    assert integral(a.controller) == initial_integral


@pytest.mark.parametrize("offset", [0., -.001, .001])
def test_repeated_older_and_late_new_stamps_do_not_renew_existing_300ms_grant(authority300, offset):
    a = authority300
    stamp, original = seed300(a)
    original_watermark = a.owner._depth30_linear_sample_watermark
    original_integral = integral(a.controller)
    for age in (.249, .251, .275, .299):
        advance(a, stamp + age)
        decision, actions, _ = decide_commit(a, a.frame(3., rpm=40., stamp=stamp + offset))
        assert not any(x.kind == "forward" and x.speed_percent > original[1] for x in actions)
        current = a.owner._fresh_depth_linear_snapshot(1)
        assert current is not None and current[3] == stamp
        assert current[1] <= original[1]
        assert a.owner._depth30_linear_sample_watermark == original_watermark
        assert a.controller._distance_pid_last_sample_timestamp == stamp
        assert integral(a.controller) == original_integral
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .30)
    advance(a, stamp + .301)
    decide_commit(a, a.frame(3., rpm=40., stamp=stamp + offset))
    assert a.owner._fresh_depth_linear_snapshot(1) is None


@pytest.mark.parametrize("age", [.1801, .210, .249, .251, .275, .299])
def test_300ms_hold_does_not_relax_180ms_first_admission(authority300, age):
    a = authority300
    stamp = a.clock.now - age
    _, actions, _ = decide_commit(a, a.frame(3., rpm=40., stamp=stamp))
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert getattr(a.owner, "_depth30_linear_sample_watermark", None) is None
    assert a.controller._distance_pid._distance_pi._last_sample_ts is None


def test_178ms_first_measurement_still_admits_with_fixed_300ms_deadline(authority300):
    a = authority300
    stamp = a.clock.now - .178
    current = a.frame(3., rpm=40., stamp=stamp)
    current = replace(current, steering_feedback=replace(current.steering_feedback,
                                                         timestamp=stamp + .04))
    _, actions, accepted = decide_commit(a, current)
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .30)


def test_expiry_during_feedback_read_writes_no_positive_packet(authority300):
    a = authority300
    stamp, _ = seed300(a)
    action, backend = writer300(a)
    advance(a, stamp + .299)

    def delayed_feedback():
        advance(a, stamp + .301)
        return a.feedback

    action.get_steering_feedback = delayed_feedback
    action._service_follow_wheels()
    assert backend.pairs and all(pair[:2] == (0, 0) for pair in backend.pairs)


def test_expiry_while_waiting_for_serial_ownership_writes_zero(authority300):
    a = authority300
    stamp, _ = seed300(a)
    action, backend = writer300(a)
    advance(a, stamp + .299)
    waits = []

    class DelayedIoLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            self.lock.acquire()
            if not waits:
                waits.append(True)
                advance(a, stamp + .301)
            return self

        def __exit__(self, *args):
            self.lock.release()

        def locked(self):
            return self.lock.locked()

    a.owner.motor_io_lock = DelayedIoLock()
    action._service_follow_wheels()
    assert waits == [True]
    assert backend.pairs and all(pair[:2] == (0, 0) for pair in backend.pairs)
    assert not a.owner.motor_io_lock.locked()


@pytest.mark.parametrize("cause", ["uid", "stop", "hazard", "obstacle", "search",
                                  "visual_lost", "feedback_missing", "feedback_stale",
                                  "early_revoke"])
def test_300ms_grant_does_not_override_current_identity_or_safety(authority300, cause):
    a = authority300
    stamp, _ = seed300(a)
    action, backend = writer300(a)
    advance(a, stamp + .275)
    current = a.frame(3., rpm=40., stamp=stamp)
    if cause == "uid":
        a.controller.active_target_id = 2
        current = a.frame(3., rpm=40., stamp=stamp, uid=2)
    elif cause == "stop":
        a.owner._explicit_stop_requested = True
    elif cause == "hazard":
        current = replace(current, hazard=replace(current.hazard, active=True, reason="test"))
    elif cause == "obstacle":
        current = replace(current, obstacles=replace(current.obstacles, front=True))
    elif cause == "search":
        a.owner.search_state = a.controller.search_state = "searching"
    elif cause == "visual_lost":
        a.owner._vision_control_state = "target_lost"
        current = replace(current, persons=[])
    elif cause == "feedback_missing":
        a.feedback = None
    elif cause == "feedback_stale":
        a.feedback = replace(a.feedback, timestamp=a.clock.now - .151)
    else:
        a.owner._revoke_depth_linear_authority("identity_rejected")
    if cause in {"uid", "hazard", "obstacle", "search", "visual_lost"}:
        decision, actions, _ = decide_commit(a, current)
        assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
        if decision.explicit_stop_requested:
            a.owner._explicit_stop_requested = True
    assert a.owner._fresh_depth_linear_snapshot(a.controller.active_target_id) is None
    action._service_follow_wheels()
    assert not any(left > 0 and right < 0 for left, right, *_ in backend.pairs)


@pytest.mark.parametrize("offset", [0., .001])
def test_explicit_revoke_cannot_be_undone_by_late_observation(authority300, offset):
    a = authority300
    stamp, _ = seed300(a)
    watermark = a.owner._depth30_linear_sample_watermark
    advance(a, stamp + .05)
    a.owner._revoke_depth_linear_authority("identity_rejected")
    advance(a, stamp + .275)
    _, actions, _ = decide_commit(a, a.frame(3., rpm=40., stamp=stamp + offset))
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.owner._depth30_linear_sample_watermark == watermark


def test_reverse_still_expires_at_180ms_with_forward_300_configuration(authority300, setup):
    a = authority300
    _, a.controller, _ = configured(
        setup, depth_longitudinal_sample_max_age_sec=.30,
        distance_target_motion_control_enable=False, reverse_enable=True,
        near_distance_rotate_only_enable=False,
        reverse_start_distance_m=1.3, reverse_immediate_distance_m=1.3)
    a.owner._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    stamp = a.clock.now
    _, actions, accepted = decide_commit(a, a.frame(1.2, rpm=0.))
    assert accepted and any(x.kind == "backward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .18)
    advance(a, stamp + .178)
    assert a.owner._fresh_depth_linear_snapshot(1)[0] == "backward"
    advance(a, stamp + .181)
    assert a.owner._fresh_depth_linear_snapshot(1) is None


def test_new_fresh_sample_at_285ms_does_not_restart_a_still_live_300ms_lease(authority300):
    a = authority300
    stamp, original = seed300(a)
    action, backend = writer300(a)
    action._service_follow_wheels()
    assert backend.pairs[-1][0] > 0 and backend.pairs[-1][1] < 0
    advance(a, stamp + .285)
    current = a.frame(3., rpm=40., stamp=stamp + .245)
    _, actions, accepted = decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert result.pi_status != "recovering"
    assert result.pi_final_limit_reason not in {
        "execution_recovery", "zero_origin_restart", "depth_expiry_recovery_step"}
    assert not result.pi_depth_expiry_recovery_used
    assert result.output_rpm >= original[1] * 2
    assert a.owner._depth30_linear_snapshot[3] == stamp + .245
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .545)
    action._service_follow_wheels()
    assert backend.pairs[-1][0] > 0 and backend.pairs[-1][1] < 0


def test_aged_request_can_only_reduce_current_cap_not_accelerate_or_renew(authority300):
    a = authority300
    stamp, original = seed300(a)
    advance(a, stamp + .275)
    current = a.frame(3., rpm=40., stamp=stamp + .001)
    for request in (original[1] + 20, max(1, original[1] - 5), original[1] + 20):
        before = a.owner._fresh_depth_linear_snapshot(1)
        decision = ControlDecision(actions=[ControlAction.forward(request, "held")], reason="held")
        actions, _ = a.owner._commit_depth_linear_decision(decision, current, 1, is_fresh_depth=True)
        after = a.owner._fresh_depth_linear_snapshot(1)
        assert after is not None and after[3] == stamp
        assert after[1] <= min(before[1], request)
        assert all(x.speed_percent <= before[1] for x in actions)
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .30)
