"""PI closure can bound near-axis rotation without claiming human identity/speed."""
from dataclasses import replace
import math

import pytest

from car_control_modular.control_types import DepthTargetObservation
from car_control_modular.controllers import FollowSafetyController
from car_control_modular.longitudinal_approach import closure_rotation_bound
from test_distance_tracking_response import setup
from test_distance_pi_controller import configured, step


def observation(frame, clock, distance, *, x=.438, yaw=30.9168, raw_yaw=-12.54,
                age=.17, rpm=0.):
    f = frame(distance, rpm=rpm, yaw=yaw)
    bbox = (x*640-100, 30., x*640+100, 450.)
    target = replace(f.persons[0], bbox=bbox,
                     depth_observation=DepthTargetObservation(bbox, 1, 1, 437, clock.now-age))
    return replace(f, persons=[target], steering_feedback=replace(
        f.steering_feedback, raw_yaw_rate_right_dps=raw_yaw))


@pytest.mark.parametrize('yaw', [-35., -30.9168, -20., 20., 30.9168, 35.])
def test_near_axis_bound_preserves_uncertainty_budget(yaw):
    args = dict(depth=1.8, bearing_deg=4., yaw_dps=yaw, geometry_age=.17)
    assert closure_rotation_bound(**args) is None  # legacy unchanged
    bound = closure_rotation_bound(**args, max_yaw_dps=35.)
    assert 0 < bound <= .25
    assert bound >= abs(math.radians(yaw))*1.8*math.tan(math.radians(4.))


@pytest.mark.parametrize('changes', [dict(yaw_dps=35.01), dict(depth=5., bearing_deg=20.),
    dict(geometry_age=.181), dict(geometry_age=-.021), dict(bearing_deg=46.),
    dict(yaw_dps=float('nan')), dict(max_yaw_dps=36.)])
def test_bound_does_not_remove_geometry_or_freshness_protection(changes):
    args = dict(depth=1.8, bearing_deg=4., yaw_dps=30., geometry_age=.17, max_yaw_dps=35.)
    assert closure_rotation_bound(**dict(args, **changes)) is None


def test_new_raw_depth_can_raise_brake_cap_without_feedforward(setup):
    clock, c, frame = configured(setup, distance_pi_launch_request_rpm=180.,
        distance_pi_kp_per_sec=3., distance_feedforward_enable=False)
    for d in [1.677, 1.828]:
        step(c, observation(frame, clock, d))
        clock.now += .1275
    r = c.last_distance_pid_result
    assert c._braking_rate_source == 'raw_depth_window'
    assert len(c._raw_closing_window.samples) == 2
    assert c._braking_range_rate > .8  # observed receding, minus rotation bound
    assert r.tracking_base_rpm == 0  # not a new human-speed matching permission
    assert 60 < r.output_rpm <= r.approach_cap_rpm < 180
    assert r.pi_brake_source == 'raw_relative_motion'
    assert c._distance_pid_last_sample_timestamp == pytest.approx(clock.now-.1275)


def test_toward_car_motion_still_brakes_with_large_launch(setup):
    clock, c, frame = configured(setup, distance_pi_launch_request_rpm=180.,
        distance_pi_kp_per_sec=3., distance_feedforward_enable=False)
    for d in [1.8, 1.6]:
        step(c, observation(frame, clock, d, rpm=60.))
        clock.now += .1
    assert c._braking_rate_source == 'raw_depth_window'
    assert c.last_distance_pid_result.output_rpm == 0


@pytest.mark.parametrize('failure', ['old_box', 'wrong_uid', 'no_box', 'bad_encoder', 'large_yaw'])
def test_controller_rejects_unqualified_geometry_and_logs_reason(setup, caplog, failure):
    clock, c, frame = configured(setup, distance_pi_launch_request_rpm=180.)
    step(c, observation(frame, clock, 2.))
    clock.now += .1
    f = observation(frame, clock, 2.1)
    t = f.persons[0]
    if failure == 'old_box': t = replace(t, depth_observation=replace(t.depth_observation, capture_timestamp=clock.now-.3))
    if failure == 'wrong_uid': t = replace(t, depth_observation=replace(t.depth_observation, target_id=2))
    if failure == 'no_box': t = replace(t, depth_observation=None)
    if failure == 'bad_encoder': f = replace(f, steering_feedback=replace(f.steering_feedback, trustworthy=False))
    if failure == 'large_yaw': f = replace(f, steering_feedback=replace(f.steering_feedback, yaw_rate_right_dps=36.))
    step(c, replace(f, persons=[t]))
    assert not c._raw_closing_window.samples
    assert c._braking_rate_source != 'raw_depth_window'
    assert 'distance_closure_rejected' in caplog.text


def test_legacy_approach_still_rejects_yaw_above_fifteen(setup):
    clock, c, frame = configured(setup)
    c = FollowSafetyController(replace(c.cfg, distance_control_mode='approach'))
    c.active_target_id = 1
    c._has_seen_person = True
    for d in [1.677, 1.828]:
        step(c, observation(frame, clock, d))
        clock.now += .1275
    assert not c._raw_closing_window.samples
