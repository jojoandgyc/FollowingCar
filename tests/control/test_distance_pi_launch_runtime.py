"""120 RPM request trial through production grants/writer, with no hardware."""
from dataclasses import replace

import pytest

from test_depth_authority_250 import authority, advance, decide_commit, writer
from test_distance_pi_controller import configured, integral
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner
from test_live_authority_binding import bind_production_reader


@pytest.fixture
def launch(authority, setup):
    a = authority
    # A small P isolates the configurable request from the normal far-distance
    # PI demand. Production P may itself request MORE than 120, still capped.
    _, a.controller, _ = configured(
        setup, depth_longitudinal_sample_max_age_sec=.25,
        distance_pi_kp_per_sec=.1, distance_pi_launch_request_rpm=120.,
    )
    a.owner._follow_controller = a.controller
    bind_production_reader(a.owner)
    return a


@pytest.mark.parametrize("visual", [False, True], ids=["depth_fast_loop", "visual_loop"])
def test_fresh_far_cold_start_reaches_120_without_software_ramp(launch, visual):
    a = launch
    current = a.frame(5.5, rpm=0.)
    decision = a.controller.decide(10, current, longitudinal_only=not visual)
    actions, accepted = a.owner._commit_depth_linear_decision(
        decision, current, 1, is_fresh_depth=True,
    )
    assert accepted and any(x.kind == "forward" and x.speed_percent == 60 for x in actions)
    result = a.controller._distance_pid._distance_pi.last_result
    assert result.software_rise_bypassed and result.launch_floor_rpm == 120.
    assert result.output_rpm == 120.
    assert not result.slew_limited
    action, backend = writer(a)
    action.get_steering_feedback = lambda: current.steering_feedback
    action._service_follow_wheels()
    assert backend.pairs[-1] == (120, -120, "FOLLOW20")


@pytest.mark.parametrize("distance", [1.6, 1.8, 2.4])
def test_close_launch_is_limited_by_braking_before_motor_write(launch, distance):
    a = launch
    current = a.frame(distance, rpm=0.)
    _, actions, accepted = decide_commit(a, current)
    assert accepted
    result = a.controller._distance_pid._distance_pi.last_result
    assert result.software_rise_bypassed and result.launch_floor_rpm == 120.
    assert result.demand_limit_reason == "braking_envelope"
    approved = max(x.speed_percent for x in actions if x.kind == "forward") * 2
    assert 0 < approved <= result.cap_rpm < 120.
    action, backend = writer(a)
    action.get_steering_feedback = lambda: current.steering_feedback
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (approved, -approved)


def test_setpoint_stopped_target_never_restarts_the_120_request(launch):
    a = launch
    start = a.clock.now
    decide_commit(a, a.frame(1.6, rpm=0.))
    for index in range(1, 6):
        advance(a, start + index * .05)
        _, actions, _ = decide_commit(a, a.frame(1.5, rpm=0.))
        assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
        assert a.owner._fresh_depth_linear_snapshot(1) is None


@pytest.mark.parametrize("feedback", ["missing", "stale", "untrusted"])
def test_missing_encoder_evidence_does_not_enable_launch(launch, feedback):
    a = launch
    current = a.frame(6., rpm=0.)
    if feedback == "missing":
        current = replace(current, steering_feedback=None)
    else:
        current = replace(current, steering_feedback=replace(
            current.steering_feedback,
            timestamp=a.clock.now - .2 if feedback == "stale" else a.clock.now,
            trustworthy=feedback != "untrusted",
        ))
    _, actions, _ = decide_commit(a, current)
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    result = a.controller._distance_pid._distance_pi.last_result
    assert result is None or (not result.software_rise_bypassed and result.launch_floor_rpm == 0.)


@pytest.mark.parametrize("cause", ["uid", "hazard", "obstacle", "visual_lost"])
def test_launch_does_not_preserve_authority_after_identity_or_safety_failure(launch, cause):
    a = launch
    stamp = a.clock.now
    decide_commit(a, a.frame(6., rpm=0.))
    assert a.owner._fresh_depth_linear_snapshot(1) is not None
    advance(a, stamp + .05)
    current = a.frame(6., rpm=0.)
    if cause == "uid":
        a.controller.active_target_id = 2
        current = a.frame(6., rpm=0., uid=2)
        # A changed owner cannot read or dispatch UID1's grant. No new
        # measurement is committed as UID2 merely to manufacture a revocation.
        assert a.owner._fresh_depth_linear_snapshot(2) is None
        return
    if cause == "hazard":
        current = replace(current, hazard=replace(current.hazard, active=True, reason="test"))
    elif cause == "obstacle":
        current = replace(current, obstacles=replace(current.obstacles, front=True))
    else:
        a.owner._vision_control_state = "target_lost"
        current = replace(current, persons=[])
    decision, actions, _ = decide_commit(a, current)
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    if decision.explicit_stop_requested:
        a.owner._explicit_stop_requested = True
    assert a.owner._fresh_depth_linear_snapshot(a.controller.active_target_id) is None


def test_duplicate_launch_sample_cannot_reintegrate_or_renew_authority(launch):
    a = launch
    start = a.clock.now
    for index in range(4):
        advance(a, start + index * .05)
        current = a.frame(6., rpm=0.)
        decide_commit(a, current)
    stamp = current.distance_state.sample_timestamp
    before_i = integral(a.controller)
    assert before_i > 0
    watermark = a.owner._depth30_linear_sample_watermark
    for age in (.025, .1, .179):
        advance(a, stamp + age)
        decide_commit(a, current)
        assert integral(a.controller) == pytest.approx(before_i)
        assert a.controller._distance_pid_last_sample_timestamp == stamp
        assert a.owner._depth30_linear_sample_watermark == watermark
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .25)
        assert a.owner._fresh_depth_linear_snapshot(1)[3] == stamp


def test_expired_120_grant_stops_writer_and_old_frame_cannot_restart(launch):
    a = launch
    current = a.frame(6., rpm=0.)
    stamp = current.distance_state.sample_timestamp
    decide_commit(a, current)
    action, backend = writer(a)
    action.get_steering_feedback = lambda: a.frame(6., rpm=0.).steering_feedback
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (120, -120)
    advance(a, stamp + .251)
    _, actions, _ = decide_commit(a, current)
    action._service_follow_wheels()
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert backend.pairs[-1][:2] == (0, 0)
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp + .25)


@pytest.mark.parametrize("age", [.181, .21, .251])
def test_late_first_measurement_cannot_start_120_request(launch, age):
    a = launch
    _, actions, _ = decide_commit(a, a.frame(6., rpm=0., stamp=a.clock.now - age))
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.controller._distance_pid._distance_pi._last_sample_ts is None


def test_launch_does_not_bypass_wheel_reversal_guard(launch):
    a = launch
    decide_commit(a, a.frame(6., rpm=0.))
    action, backend = writer(a)
    # A positive authorized target must not skip an actual wheel still moving
    # backward, even though the PI's ordinary rise limiter was bypassed.
    action.get_steering_feedback = lambda: a.frame(6., rpm=-10.).steering_feedback
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (0, 0)
    assert action._visible_wheel_waiting


def test_launch_expiry_while_reading_feedback_prevents_positive_write(launch):
    a = launch
    current = a.frame(6., rpm=0.)
    stamp = current.distance_state.sample_timestamp
    decide_commit(a, current)
    action, backend = writer(a)
    advance(a, stamp + .249)

    def feedback_after_expiry():
        advance(a, stamp + .251)
        return a.frame(6., rpm=0.).steering_feedback

    action.get_steering_feedback = feedback_after_expiry
    action._service_follow_wheels()
    assert backend.pairs and all(pair[:2] == (0, 0) for pair in backend.pairs)
