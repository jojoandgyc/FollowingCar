"""Calculate a request with signed motion; executor owns transition handling."""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from test_depth_authority_250 import authority, advance, decide_commit, writer
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner
from test_live_authority_binding import bind_production_reader


@pytest.mark.parametrize("ego", [-1.5, -10., 0., 10.])
def test_signed_fresh_feedback_enables_launch_request(ego):
    c = DistancePiController(DistancePiConfig(launch_request_rpm=180))
    r = c.update(1.6038, 1.5, sample_timestamp=100, execution_now=100,
                 deadband_m=.03, max_output_rpm=200, rise_rpm_per_sec=1,
                 ego_forward_rpm=ego)
    assert r.launch_floor_rpm == 180 and r.software_rise_bypassed
    assert 0 < r.output_rpm <= r.cap_rpm < 180


def test_signed_ego_kept_in_raw_relative_braking_not_clamped_into_false_person_speed():
    c = DistancePiController(DistancePiConfig(launch_request_rpm=180))
    # Backing away from a stationary target: range growth equals backwards
    # vehicle speed. It is NOT evidence of a moving-away person.
    velocity = 10 * c.config.wheel_circumference_m / 60
    r = c.update(1.8, 1.5, sample_timestamp=100, execution_now=100,
                 deadband_m=.03, max_output_rpm=200, ego_forward_rpm=-10,
                 range_rate_m_s=velocity, raw_closure_valid=True)
    still = DistancePiController(c.config).update(
        1.8, 1.5, sample_timestamp=100, execution_now=100, deadband_m=.03,
        max_output_rpm=200, ego_forward_rpm=0, range_rate_m_s=0, raw_closure_valid=True)
    assert r.cap_rpm == pytest.approx(still.cap_rpm)


@pytest.mark.parametrize("loop", ["periodic", "direct"])
def test_cap99_residual_reverse_no_longer_suppresses_request_or_writer(authority, setup, loop):
    a = authority
    _, a.controller, _ = configured(setup, depth_longitudinal_sample_max_age_sec=.25,
        distance_pi_kp_per_sec=.1, distance_pi_launch_request_rpm=180.)
    a.owner._follow_controller = a.controller
    bind_production_reader(a.owner)
    current = a.frame(1.6038)
    current = replace(current, steering_feedback=replace(current.steering_feedback,
                      left_forward_rpm=0., right_forward_rpm=-3.))
    _, actions, accepted = decide_commit(a, current)
    assert accepted
    result = a.controller._distance_pid._distance_pi.last_result
    assert result.launch_floor_rpm == 180 and result.software_rise_bypassed
    approved = max(x.speed_percent for x in actions if x.kind == "forward") * 2
    assert approved > 0 and approved <= result.cap_rpm
    action, backend = writer(a)
    action.config.follow_forward_handoff_enable = True
    action.get_steering_feedback = lambda: current.steering_feedback
    if loop == "periodic":
        action._service_follow_wheels()
    else:
        action.config.follow_wheel_period_sec = 0
        action._send_follow_wheel_targets(approved, -approved, "DIRECT", visible_required=True)
    assert backend.pairs[-1][:2] == (approved, -approved)
    assert not action._visible_wheel_waiting


@pytest.mark.parametrize("loop", ["periodic", "direct"])
def test_handoff_still_rechecks_depth_expiry_at_write(authority, setup, loop):
    a = authority
    _, a.controller, _ = configured(setup, depth_longitudinal_sample_max_age_sec=.25,
                                    distance_pi_launch_request_rpm=180.)
    a.owner._follow_controller = a.controller
    bind_production_reader(a.owner)
    decide_commit(a, a.frame(5.5))
    stamp = a.owner._depth30_linear_snapshot[3]
    action, backend = writer(a)
    action.config.follow_forward_handoff_enable = True
    advance(a, stamp + .249)

    def delayed_feedback():
        advance(a, stamp + .251)
        # Valid relative to the writer's captured now; the WRITE-TIME depth
        # check, not a future-dated encoder rejection, must veto the request.
        return replace(a.frame(5.5).steering_feedback, timestamp=stamp + .249,
                       left_forward_rpm=0., right_forward_rpm=-3.)

    action.get_steering_feedback = delayed_feedback
    if loop == "periodic":
        action._service_follow_wheels()
    else:
        action.config.follow_wheel_period_sec = 0
        action._send_follow_wheel_targets(40, -40, "DIRECT", visible_required=True)
    assert not backend.pairs or all(pair[:2] == (0, 0) for pair in backend.pairs)
