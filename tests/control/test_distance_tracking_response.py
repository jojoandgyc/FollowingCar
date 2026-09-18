"""Offline longitudinal regressions: no camera, motor backend or runtime threads."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import DistanceState, PersonTarget, SensorFrame, SteeringFeedback, ObstacleState
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.steering_pid import DistancePidConfig, LongitudinalDistancePid


@pytest.fixture
def setup(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: clock.now)
    cfg = FollowPolicyConfig(
        distance_pid_enable=True, distance_feedforward_enable=True,
        target_distance_m=1.5, forward_start_distance_m=1.58,
        forward_stop_distance_m=1.53, near_distance_rotate_only_enable=True,
        near_distance_rotate_only_distance_m=1.53, reverse_enable=False,
        distance_pid_deadband_m=0.03, distance_pid_kp_rpm_per_m=24.0,
        distance_pid_ki_rpm_per_m_s=0.8, distance_pid_kd_rpm_s_per_m=2.0,
        forward_min_rpm=20, forward_max_rpm=100,
        visible_steering_pid_camera_hfov_deg=60.0,
        depth_recovery_stage1_sec=0.0, depth_recovery_stage2_sec=0.0,
    )
    controller = FollowSafetyController(cfg)
    controller.active_target_id = 1
    controller._has_seen_person = True
    target = PersonTarget((240.0, 50.0, 400.0, 460.0), 1, 0.95, 65600.0)

    def frame(distance, rpm=0.0, *, stamp=None, uid=1, yaw=0.0, detail="depth_multiregion", **extra):
        physical = clock.now if stamp is None else stamp
        state = DistanceState(
            source="vision_depth", raw_distance_m=distance, used_distance_m=distance,
            filtered_distance_m=distance, source_detail=detail, sample_timestamp=physical,
            sample_age_sec=max(0.0, clock.now - physical), fusion_confidence=1.0,
        )
        return SensorFrame(
            width=640, height=480, persons=[replace(target, track_id=uid)],
            distance_m=distance, distance_state=state,
            steering_feedback=SteeringFeedback(
                timestamp=clock.now, trustworthy=True, left_forward_rpm=rpm,
                right_forward_rpm=rpm, yaw_rate_right_dps=yaw,
            ), **extra,
        )

    return clock, controller, frame


def decide(controller, observation):
    return controller.decide(10, observation, longitudinal_only=True)


def test_forward_starts_at_158_not_180(setup):
    clock, controller, frame = setup
    first = decide(controller, frame(1.50))
    assert first.actions[0].speed_percent == 0
    clock.now += 0.1
    # No eligible motion estimate is necessary to enter the new start band.
    started = decide(controller, frame(1.59, yaw=20.0))
    assert started.reason == "longitudinal_distance_pid"
    assert started.actions[0].speed_percent > 0


def test_matching_moving_target_at_150_remains_forward(setup):
    clock, controller, frame = setup
    assert decide(controller, frame(1.50, rpm=30)).actions[0].speed_percent == 0
    clock.now += 0.1
    matched = decide(controller, frame(1.50, rpm=30))
    assert matched.reason == "longitudinal_distance_pid"
    assert matched.actions[0].speed_percent == 30
    assert controller.last_distance_pid_result.tracking_base_rpm == pytest.approx(30.0)


def test_stationary_at_setpoint_does_not_creep(setup):
    clock, controller, frame = setup
    for _ in range(8):
        response = decide(controller, frame(1.50))
        assert response.actions[0].speed_percent == 0
        clock.now += 0.05


def test_ego_motion_is_not_misread_as_target_motion(setup):
    clock, controller, frame = setup
    decide(controller, frame(1.60, rpm=30))
    clock.now += 0.1
    response = decide(controller, frame(1.57, rpm=30))
    evidence = controller._longitudinal_motion_evidence
    assert not evidence.eligible
    assert evidence.target_rpm == 0
    # Near target, losing matching now stops rather than re-injecting launch20.
    assert response.current_forward_percent == 0
    assert controller.last_distance_pid_result is None


def test_matching_stops_for_too_close_reading(setup):
    clock, controller, frame = setup
    decide(controller, frame(1.50, rpm=30))
    clock.now += 0.1
    assert decide(controller, frame(1.50, rpm=30)).actions[0].speed_percent == 30
    clock.now += 0.1
    response = decide(controller, frame(1.46, rpm=30))
    assert response.actions[0].speed_percent == 0
    assert controller._tracking_base_rpm(1.46, clock.now) is None


@pytest.mark.parametrize("condition", ["missing", "held", "jump", "yaw", "feedback", "uid", "ir"])
def test_invalid_evidence_never_keeps_matching_velocity(setup, condition):
    clock, controller, frame = setup
    decide(controller, frame(1.50, rpm=30))
    clock.now += 0.1
    decide(controller, frame(1.50, rpm=30))
    clock.now += 0.1
    observation = frame(1.50, rpm=30)
    if condition == "missing":
        observation = replace(observation, persons=[])
    elif condition == "held":
        observation = replace(observation, distance_state=replace(
            observation.distance_state, raw_distance_m=None, source_detail="depth_multiregion_reused_hold"
        ))
    elif condition == "jump":
        observation = replace(observation, distance_state=replace(
            observation.distance_state, source_detail="distance_jump_pending"
        ))
    elif condition == "yaw":
        observation = frame(1.50, rpm=30, yaw=20.0)
    elif condition == "feedback":
        observation = replace(observation, steering_feedback=replace(
            observation.steering_feedback, trustworthy=False
        ))
    elif condition == "uid":
        observation = frame(1.50, rpm=30, uid=2)
    else:
        observation = replace(observation, obstacles=ObstacleState(front=True))
    response = decide(controller, observation)
    assert controller._tracking_base_rpm(1.50, clock.now) is None
    assert not any(a.kind == "forward" and a.speed_percent > 0 for a in response.actions)


def test_same_physical_depth_does_not_integrate_twice(setup):
    clock, controller, frame = setup
    stamp = clock.now
    decide(controller, frame(1.90, stamp=stamp))
    result = controller.last_distance_pid_result
    clock.now += 0.04
    decide(controller, frame(1.90, stamp=stamp))
    assert controller.last_distance_pid_result is result
    assert controller.last_distance_pid_result.integral_m_s == result.integral_m_s


def test_raw_physical_range_not_lagging_median_drives_velocity_estimate(setup):
    clock, controller, frame = setup
    decide(controller, frame(2.00, rpm=30))
    clock.now += 0.1
    observation = frame(1.985, rpm=30)
    observation = replace(observation, distance_state=replace(
        observation.distance_state, raw_distance_m=1.970,
    ))
    decide(controller, observation)
    assert controller._longitudinal_motion_evidence.target_rpm == 0


@pytest.mark.parametrize("change", ["yaw", "untrusted", "stale"])
def test_same_depth_new_unsafe_feedback_revokes_ff_without_integrating(setup, change):
    clock, controller, frame = setup
    decide(controller, frame(1.90, rpm=30))
    clock.now += 0.1
    stamp = clock.now
    decide(controller, frame(1.90, rpm=30))
    previous = controller.last_distance_pid_result
    assert previous.tracking_base_rpm == pytest.approx(30)
    clock.now += 0.03
    observation = frame(1.90, rpm=30, stamp=stamp)
    feedback = observation.steering_feedback
    if change == "yaw":
        feedback = replace(feedback, raw_yaw_rate_right_dps=20.0)
    elif change == "stale":
        feedback = replace(feedback, timestamp=clock.now - 0.2)
    else:
        feedback = replace(feedback, trustworthy=False)
    decide(controller, replace(observation, steering_feedback=feedback))
    assert controller._tracking_base_rpm(1.90, clock.now) is None
    result = controller.last_distance_pid_result
    assert result.integral_m_s == previous.integral_m_s
    assert result.tracking_base_rpm == 0
    assert result.output_rpm <= previous.output_rpm


def test_older_aligned_depth_cannot_reset_new_pid(setup):
    clock, controller, frame = setup
    decide(controller, frame(1.90))
    result = controller.last_distance_pid_result
    clock.now += 0.05
    response = decide(controller, frame(1.50, stamp=99.98))
    assert response.reason == "longitudinal_old_depth_observation"
    assert controller.last_distance_pid_result is result


@pytest.mark.parametrize("age", [0.19, 0.22, 0.25])
def test_expired_motor_authority_does_not_drop_physical_pid_dedupe(setup, age):
    clock, controller, frame = setup
    controller.cfg = replace(controller.cfg, distance_feedforward_enable=False)
    stamp = clock.now
    decide(controller, frame(1.90, stamp=stamp))
    result = controller.last_distance_pid_result
    clock.now = stamp + age
    decide(controller, frame(1.90, stamp=stamp))
    assert controller.last_distance_pid_result is result
    assert controller._distance_pid_last_sample_timestamp == stamp


def test_future_physical_time_does_not_poison_pid_watermark(setup):
    clock, controller, frame = setup
    decide(controller, frame(1.90))
    result = controller.last_distance_pid_result
    clock.now += 0.05
    decide(controller, frame(2.00, stamp=clock.now + 10.0))
    assert controller.last_distance_pid_result is result
    assert controller._distance_pid_last_sample_timestamp == 100.0


@pytest.mark.parametrize("base", [5.0, 10.0, 30.0, 40.0])
def test_pid_tracking_base_replaces_launch_offset(base):
    pid = LongitudinalDistancePid(DistancePidConfig(min_forward_output_rpm=20))
    result = pid.update(1.5, 1.5, now=100.0, tracking_base_rpm=base)
    assert result.output_rpm == base


@pytest.mark.parametrize("base", [float("nan"), float("inf"), -1.0])
def test_pid_rejects_invalid_tracking_base(base):
    pid = LongitudinalDistancePid(DistancePidConfig())
    with pytest.raises(ValueError):
        pid.update(1.5, 1.5, now=100.0, tracking_base_rpm=base)


def test_pid_matching_can_decelerate_without_authorizing_reverse():
    pid = LongitudinalDistancePid(DistancePidConfig(kp_rpm_per_m=100))
    assert pid.update(1.3, 1.5, now=100.0, tracking_base_rpm=5).output_rpm == 0


def test_idealized_moving_target_does_not_need_large_steady_distance_error(setup):
    """Synthetic plant, not a claim about real chassis dynamics."""
    clock, template, frame = setup

    def simulate(enabled):
        controller = FollowSafetyController(replace(template.cfg, distance_feedforward_enable=enabled))
        controller.active_target_id = 1
        controller._has_seen_person = True
        distance, wheel_rpm = 1.50, 0.0
        distances = []
        for _ in range(160):
            response = decide(controller, frame(distance, rpm=wheel_rpm))
            command = next((float(a.speed_percent) for a in response.actions if a.kind == "forward"), 0.0)
            # 100ms wheel response, five 10ms substeps per depth observation.
            for _ in range(5):
                old_rpm = wheel_rpm
                wheel_rpm += (command - wheel_rpm) * 0.1
                distance += (0.30 - (old_rpm + wheel_rpm) * 0.005) * 0.01
                clock.now += 0.01
            distances.append(distance)
        return distances

    matching = simulate(True)
    proportional = simulate(False)
    assert max(matching) < 1.60
    assert abs(matching[-1] - 1.50) < 0.06
    assert proportional[-1] - 1.50 > 0.15
