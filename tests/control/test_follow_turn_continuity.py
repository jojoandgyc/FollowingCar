"""Analytic camera rotation and bounded recovery, without motor hardware."""
from dataclasses import replace
import math
import pytest

from test_distance_tracking_response import setup, decide
from car_control_modular.control_types import DepthTargetObservation
from car_control_modular.longitudinal_feedforward import LongitudinalFeedforwardEstimator, depth_rotation_rate


def bbox(x, z):
    cx = 320 + x/z * (320 / math.tan(math.radians(30)))
    return (cx-60, 60, cx+60, 400)


@pytest.mark.parametrize("yaw", [-15., -10., 10., 15.])
@pytest.mark.parametrize("x0", [-.2, .2])
def test_rotating_camera_does_not_invent_stationary_person_velocity(yaw, x0):
    estimator = LongitudinalFeedforwardEstimator()
    for i in range(3):
        delta = math.radians(yaw) * i*.1
        x, z = x0*math.cos(delta)-2*math.sin(delta), x0*math.sin(delta)+2*math.cos(delta)
        stamp = 100+i*.1
        correction = depth_rotation_rate(depth=z, bbox=bbox(x,z), width=640, hfov_deg=60,
            capture_stamp=stamp, depth_stamp=stamp, yaw=yaw)
        assert correction is not None
        result = estimator.update(now=stamp+.01, sample_timestamp=stamp, distance_m=z,
            target_id=1, feedback_timestamp=stamp, ego_forward_rpm=0, yaw_rate_dps=yaw,
            trusted=True, target_bearing_deg=correction[1], rotation_rate_m_s=correction[0])
        if i:
            assert abs(result.target_speed_m_s) < .0001
            assert not result.eligible and result.target_rpm == 0


def test_walking_person_keeps_matching_speed_through_turn():
    estimator = LongitudinalFeedforwardEstimator()
    for i in range(4):
        t = i*.05
        theta = math.radians(10)*t
        world_z = 2 + .3*t
        x, z = .1*math.cos(theta)-world_z*math.sin(theta), .1*math.sin(theta)+world_z*math.cos(theta)
        stamp = 100+t
        correction = depth_rotation_rate(depth=z, bbox=bbox(x,z), width=640, hfov_deg=60,
            capture_stamp=stamp, depth_stamp=stamp, yaw=10)
        result = estimator.update(now=stamp+.01, sample_timestamp=stamp, distance_m=z,
            target_id=1, feedback_timestamp=stamp, ego_forward_rpm=0, yaw_rate_dps=10,
            trusted=True, target_bearing_deg=correction[1], rotation_rate_m_s=correction[0])
        if i:
            assert result.eligible and result.target_rpm == pytest.approx(30, abs=.2)


@pytest.mark.parametrize("change", [{"yaw":16}, {"capture_stamp":99.7},
    {"depth":float("nan")}, {"bbox":(0,0,50,300)}, {"depth_stamp":99.9}])
def test_compensation_never_bypasses_geometry_time_or_yaw_limits(change):
    args = dict(depth=2, bbox=bbox(.1,2), width=640, hfov_deg=60,
                capture_stamp=100., depth_stamp=100., yaw=10)
    assert depth_rotation_rate(**{**args, **change}) is None


def test_rgb_bearing_is_projected_to_depth_capture_time():
    value = depth_rotation_rate(depth=2, bbox=bbox(0,2), width=640, hfov_deg=60,
        capture_stamp=100., depth_stamp=100.1, yaw=10)
    assert value[1] == pytest.approx(-1.)
    assert value[0] < 0


def test_controller_requires_raw_geometry_to_enable_turn_compensation(setup):
    clock, controller, frame = setup
    controller.cfg = replace(controller.cfg, distance_turn_compensation_enable=True)
    for i in range(3):
        current = frame(1.9, rpm=30, yaw=10)
        p = current.persons[0]
        observation = DepthTargetObservation(p.bbox, 1, 1, i+1, clock.now)
        current = replace(current, persons=[replace(p, depth_observation=observation)])
        decide(controller, current)
        if i: assert controller._longitudinal_motion_evidence.eligible
        clock.now += .05
    assert controller._longitudinal_motion_evidence.rotation_rate_m_s == 0.
    # Without raw provenance the original yaw gate still applies.
    clock.now += .2
    decide(controller, frame(1.9, rpm=30, yaw=10))
    assert controller._tracking_base_rpm(1.9, clock.now) is None


def recovery_case(setup, *, delay=.22, rpm=42., detail="depth_detector_bbox_stale", **changes):
    clock, controller, frame = setup
    controller.cfg = replace(controller.cfg, depth_measured_recovery_enable=True,
                             depth_recovery_stage1_sec=.2, depth_recovery_stage2_sec=.4)
    controller._depth_recovery_anchor = (1, clock.now, 2.)
    controller._depth_quality_degraded = False
    controller._distance_pid._last_output_rpm = 55.
    controller._depth_last_approved_forward_rpm = 55.
    old = frame(2., rpm=rpm)
    clock.now += .1
    missing = replace(old, distance_state=replace(old.distance_state,
        raw_distance_m=None, sample_timestamp=None, source_detail=detail))
    controller._note_depth_quality_failure(missing, clock.now)
    clock.now += delay-.1
    new = frame(2.05, rpm=rpm, **changes)
    return clock, controller, new


def test_short_scheduling_gap_recovers_from_measured_speed_not_fixed_25(setup):
    clock, controller, new = recovery_case(setup)
    # New fresh-sample ramp: measured 42 + one 50ms acceleration budget (12).
    assert controller._limit_depth_quality_forward_percent(new, 55, clock.now) == 54
    assert controller._depth_schedule_recovery[3] == 55


@pytest.mark.parametrize("condition", ["long_gap", "wrong_uid", "jump", "closing", "stale_feedback", "hazard", "reverse_wheel"])
def test_measured_recovery_requires_fresh_noncontradictory_evidence(setup, condition):
    clock, controller, new = recovery_case(setup, delay=.501 if condition=="long_gap" else .22)
    if condition=="wrong_uid": controller.active_target_id=2
    if condition=="jump": new=replace(new,distance_state=replace(new.distance_state,raw_distance_m=2.5))
    if condition=="closing": new=replace(new,distance_state=replace(new.distance_state,raw_distance_m=1.9))
    if condition=="stale_feedback": new=replace(new,steering_feedback=replace(new.steering_feedback,timestamp=clock.now-.2))
    if condition=="hazard": new=replace(new,hazard=replace(new.hazard,active=True))
    if condition=="reverse_wheel": new=replace(new,steering_feedback=replace(new.steering_feedback,left_forward_rpm=-10))
    assert controller._limit_depth_quality_forward_percent(new, 55, clock.now) == 25


def test_low_measured_speed_cannot_resume_old_high_command(setup):
    clock, controller, new = recovery_case(setup, rpm=10.)
    assert controller._limit_depth_quality_forward_percent(new, 55, clock.now) == 22
