"""250 ms forward leases through real PI/runtime/writer; fake clocks and I/O."""
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.action_runtime import MotionActionRuntime
from car_control_modular.control_types import ControlAction, ControlDecision
from test_distance_pi_controller import configured, integral
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


@pytest.fixture
def authority(setup, owner, monkeypatch):
    clock, controller, frame = configured(
        setup, depth_longitudinal_sample_max_age_sec=.25,
    )
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC", .25)
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 200)
    monkeypatch.setattr(runtime, "MOTOR_FORWARD_MAX_TARGET_RPM", 200)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT", 100)
    monkeypatch.setattr(runtime, "MOTOR_RS485_TARGET_MIN_INTERVAL_SEC", .05)
    owner._follow_controller = controller
    owner._lateral_yaw_revision = 1
    controller._live_longitudinal_authority_reader = owner._fresh_depth_linear_snapshot
    return SimpleNamespace(clock=clock, controller=controller, frame=frame, owner=owner)


def decide_commit(a, current, *, fresh=True):
    decision = a.controller.decide(10, current, longitudinal_only=True)
    actions, accepted = a.owner._commit_depth_linear_decision(
        decision, current, a.controller.active_target_id, is_fresh_depth=fresh,
    )
    return decision, actions, accepted


def advance(a, now):
    a.clock.now = now
    # The extension does not relax visual freshness: RGB continues to report
    # the same visible identity while physical Depth is delayed.
    a.owner._last_vision_control_ts = now - .01


def seed(a, *, distance=2.5, rpm=40.0):
    current = a.frame(distance, rpm=rpm)
    _, actions, accepted = decide_commit(a, current)
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    stamp = current.distance_state.sample_timestamp
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .25)
    return stamp, a.owner._depth30_linear_snapshot


@pytest.mark.parametrize("age,live", [(.178, True), (.210, True), (.249, True), (.251, False)])
def test_original_safe_forward_grant_uses_250ms_deadline(authority, age, live):
    a = authority
    stamp, original = seed(a)
    advance(a, stamp + age)
    current = a.owner._fresh_depth_linear_snapshot(1)
    assert (current is not None) is live
    if live:
        assert current[3] == stamp
        assert 0 < current[1] <= original[1]
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .25)


def test_repeated_frames_never_extend_deadline_or_reintegrate(authority):
    a = authority
    stamp, original = seed(a)
    watermark = a.owner._depth30_linear_sample_watermark
    before_i = integral(a.controller)
    for age in (.178, .210, .249):
        advance(a, stamp + age)
        decision, _, _ = decide_commit(a, a.frame(2.5, rpm=40, stamp=stamp))
        if age > .18:
            assert not decision.actions
        current = a.owner._fresh_depth_linear_snapshot(1)
        assert current is not None and current[3] == stamp
        assert current[1] <= original[1]
        assert a.owner._depth30_linear_sample_watermark == watermark
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .25)
        assert a.controller._distance_pid_last_sample_timestamp == stamp
        assert a.controller._distance_pid._distance_pi._last_sample_ts == stamp
        assert integral(a.controller) == pytest.approx(before_i)
    advance(a, stamp + .251)
    decision, _, _ = decide_commit(a, a.frame(2.5, rpm=40, stamp=stamp))
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in decision.actions)
    assert a.owner._fresh_depth_linear_snapshot(1) is None


def test_late_new_stamp_keeps_original_grant_not_new_measurement(authority):
    a = authority
    stamp, original = seed(a)
    watermark = a.owner._depth30_linear_sample_watermark
    before_i = integral(a.controller)
    for age, incoming_offset in ((.210, .001), (.249, .010)):
        advance(a, stamp + age)
        current = a.frame(2.5, rpm=40, stamp=stamp + incoming_offset)
        decision, _, _ = decide_commit(a, current)
        assert not decision.actions
        assert a.controller._distance_pid_last_sample_timestamp == stamp
        assert a.controller._distance_pid._distance_pi._last_sample_ts == stamp
        assert integral(a.controller) == pytest.approx(before_i)
        assert a.owner._depth30_linear_sample_watermark == watermark
        live = a.owner._fresh_depth_linear_snapshot(1)
        assert live is not None and live[3] == stamp and live[1] <= original[1]
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .25)


@pytest.mark.parametrize("incoming_offset", [0.0, .001])
def test_deferred_depth_preserves_real_nonzero_integral_without_integration(authority, setup, incoming_offset):
    a = authority
    _, a.controller, _ = configured(
        setup, depth_longitudinal_sample_max_age_sec=.25, distance_pi_kp_per_sec=.5,
    )
    a.owner._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    for _ in range(5):
        seed(a)
        advance(a, a.clock.now + .05)
    stamp = a.controller._distance_pid_last_sample_timestamp
    before_i = integral(a.controller)
    assert before_i > .05
    advance(a, stamp + .210)
    decision, _, _ = decide_commit(a, a.frame(2.5, rpm=40, stamp=stamp + incoming_offset))
    assert not decision.actions
    assert a.controller._distance_pid_last_sample_timestamp == stamp
    assert a.controller._distance_pid._distance_pi._last_sample_ts == stamp
    assert integral(a.controller) == pytest.approx(before_i)
    assert a.owner._fresh_depth_linear_snapshot(1)[3] == stamp


def test_no_measurement_at_210ms_keeps_only_original_safe_authority(authority):
    a = authority
    stamp, original = seed(a)
    advance(a, stamp + .210)
    missing = a.frame(None)
    missing = replace(missing, distance_state=replace(
        missing.distance_state, sample_timestamp=None, source_detail="depth_detector_bbox_stale",
    ))
    decision, _, _ = decide_commit(a, missing, fresh=False)
    assert not decision.actions
    assert a.owner._fresh_depth_linear_snapshot(1) == original
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .25)
    assert a.controller._distance_pid_last_sample_timestamp == stamp


@pytest.mark.parametrize("age", [.181, .210, .249])
def test_late_sample_cannot_create_first_forward_grant(authority, age):
    a = authority
    stamp = a.clock.now - age
    _, actions, _ = decide_commit(a, a.frame(2.5, rpm=40, stamp=stamp))
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert getattr(a.owner, "_depth30_linear_sample_watermark", None) is None
    assert a.controller._distance_pid._distance_pi._last_sample_ts is None


def test_178ms_new_sample_can_grant_but_210ms_cannot(authority):
    a = authority
    stamp = a.clock.now - .178
    current = a.frame(2.5, rpm=40, stamp=stamp)
    current = replace(current, steering_feedback=replace(
        current.steering_feedback, timestamp=stamp + .04,
    ))
    _, actions, accepted = decide_commit(a, current)
    assert accepted and any(x.speed_percent > 0 for x in actions)
    assert a.controller._distance_pid_last_sample_timestamp == stamp
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .25)


@pytest.mark.parametrize("cause", ["uid", "hazard", "obstacle", "search", "visual_lost"])
def test_original_grant_is_revoked_by_identity_and_safety_at_210ms(authority, cause):
    a = authority
    stamp, _ = seed(a)
    advance(a, stamp + .210)
    current = a.frame(2.5, rpm=40, stamp=stamp)
    if cause == "uid":
        a.controller.active_target_id = 2
        current = a.frame(2.5, rpm=40, stamp=stamp, uid=2)
    elif cause == "hazard":
        current = replace(current, hazard=replace(current.hazard, active=True, reason="test"))
    elif cause == "obstacle":
        current = replace(current, obstacles=replace(current.obstacles, front=True))
    elif cause == "search":
        a.owner.search_state = a.controller.search_state = "searching"
    else:
        a.owner._vision_control_state = "target_lost"
        current = replace(current, persons=[])
    decision, actions, _ = decide_commit(a, current)
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    # Runtime's actual stop handling withdraws the grant before motor dispatch.
    if decision.explicit_stop_requested:
        a.owner._explicit_stop_requested = True
    assert a.owner._fresh_depth_linear_snapshot(a.controller.active_target_id) is None


def test_new_late_near_measurement_must_not_preserve_far_grant(authority):
    a = authority
    stamp, _ = seed(a)
    advance(a, stamp + .210)
    _, actions, _ = decide_commit(a, a.frame(1.5, rpm=40, stamp=stamp + .001))
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._fresh_depth_linear_snapshot(1) is None


def test_near_braking_has_no_180_to_250_extension(authority):
    a = authority
    stamp, _ = seed(a, distance=1.60, rpm=20)
    advance(a, stamp + .178)
    assert a.owner._fresh_depth_linear_snapshot(1) is not None
    advance(a, stamp + .210)
    assert a.owner._fresh_depth_linear_snapshot(1) is None


def test_extended_depth_grant_does_not_extend_visual_freshness(authority):
    a = authority
    stamp, _ = seed(a)
    # Deliberately do not simulate another same-UID visual observation.
    a.clock.now = stamp + .210
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .25)


def test_reverse_authority_still_expires_at_180ms(authority, setup):
    a = authority
    _, a.controller, _ = configured(
        setup, depth_longitudinal_sample_max_age_sec=.25, reverse_enable=True,
        near_distance_rotate_only_enable=False,
        reverse_start_distance_m=1.3, reverse_immediate_distance_m=1.3,
    )
    a.owner._follow_controller = a.controller
    stamp = a.clock.now
    _, actions, accepted = decide_commit(a, a.frame(1.2, rpm=0.0))
    assert accepted and any(x.kind == "backward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .18)
    advance(a, stamp + .178)
    assert a.owner._fresh_depth_linear_snapshot(1)[0] == "backward"
    advance(a, stamp + .181)
    assert a.owner._fresh_depth_linear_snapshot(1) is None


@pytest.mark.parametrize("incoming_offset", [0.0, .001])
def test_explicitly_revoked_grant_cannot_be_restored_by_late_depth(authority, incoming_offset):
    a = authority
    stamp, _ = seed(a)
    watermark = a.owner._depth30_linear_sample_watermark
    advance(a, stamp + .05)
    a.owner._revoke_depth_linear_authority("test_early_revoke")
    advance(a, stamp + .210)
    _, actions, _ = decide_commit(a, a.frame(2.5, rpm=40, stamp=stamp + incoming_offset))
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.owner._depth30_linear_sample_watermark == watermark


def test_aged_decision_can_reduce_but_not_accelerate_old_grant(authority):
    a = authority
    stamp, original = seed(a)
    advance(a, stamp + .210)
    current = a.frame(2.5, rpm=40, stamp=stamp + .001)
    for requested in (original[1] + 20, max(1, original[1] - 5), original[1] + 20):
        before = a.owner._fresh_depth_linear_snapshot(1)
        decision = ControlDecision(actions=[ControlAction.forward(requested, "held")], reason="held")
        actions, _ = a.owner._commit_depth_linear_decision(decision, current, 1, is_fresh_depth=True)
        after = a.owner._fresh_depth_linear_snapshot(1)
        assert after is not None and after[3] == stamp
        assert after[1] <= min(before[1], requested)
        assert all(x.speed_percent <= before[1] for x in actions)
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .25)


class FakeBackend:
    def __init__(self):
        self.pairs = []

    def wheel_raw_state_to_target(self, wheel, rpm, state):
        return rpm if wheel == "left" else -rpm

    def send_targets(self, left, right, label, **kwargs):
        self.pairs.append((left, right, label))


def writer(a):
    backend = FakeBackend()
    a.owner.motor_io_lock = threading.Lock()
    a.owner.stop_action_execution = False
    a.owner.person_detected_flag = False
    config = SimpleNamespace(
        follow_wheel_period_sec=.05, steering_feedback_median_window=3,
        use_percent_speed=True, motor_forward_max_target_rpm=200,
        motor_steer_raw_target=15, rotation_only=False, rotate_pulse_brake_enable=False,
    )
    action = MotionActionRuntime(
        a.owner, backend, config, SimpleNamespace(rotate_right=runtime.ACTION_ROTATE_RIGHT),
        hard_stop_check=lambda _: False,
    )
    action.get_steering_feedback = lambda: a.frame(2.5, rpm=40).steering_feedback
    return action, backend


def test_periodic_motor_writer_obeys_250ms_not_old_180ms(authority):
    a = authority
    stamp, _ = seed(a)
    action, backend = writer(a)
    for age in (.178, .210, .249):
        advance(a, stamp + age)
        action._service_follow_wheels()
        assert backend.pairs[-1][0] > 0 and backend.pairs[-1][1] < 0
    advance(a, stamp + .251)
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (0, 0)


def test_expiry_during_feedback_read_cannot_write_positive_motor_pair(authority):
    a = authority
    stamp, _ = seed(a)
    action, backend = writer(a)
    advance(a, stamp + .249)

    def read_feedback():
        advance(a, stamp + .251)
        return a.frame(2.5, rpm=40).steering_feedback

    action.get_steering_feedback = read_feedback
    action._service_follow_wheels()
    assert backend.pairs and all(pair[:2] == (0, 0) for pair in backend.pairs)


def test_direct_motor_write_rechecks_expiry_after_feedback_read(authority):
    a = authority
    stamp, original = seed(a)
    action, backend = writer(a)
    action.config.follow_wheel_period_sec = 0.0
    advance(a, stamp + .249)

    def read_feedback():
        advance(a, stamp + .251)
        return a.frame(2.5, rpm=40).steering_feedback

    action.get_steering_feedback = read_feedback
    with a.owner.motor_io_lock:
        action._send_follow_wheel_targets(original[1] * 2, -original[1] * 2, "TEST", visible_required=True)
    assert not backend.pairs or all(pair[:2] == (0, 0) for pair in backend.pairs)
