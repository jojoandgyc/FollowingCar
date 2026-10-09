"""CAP193--370: far chase must not be a permanent near-distance180 request."""
import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from test_depth_authority_250 import authority, advance, decide_commit, writer
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_live_authority_binding import bind_production_reader
from test_lateral_zero_runtime import owner


def controller(**changes):
    return DistancePiController(DistancePiConfig(**dict(
        dict(kp_per_sec=3., launch_request_rpm=180., launch_full_error_m=.5,
             physical_ttl_sec=.25, motion_memory_sec=.35), **changes)))


def update(c, *, distance=1.8, stamp=100., ego=20., rate=0., rise=180., **changes):
    return c.update(distance, 1.4, **dict(dict(
        sample_timestamp=stamp, execution_now=stamp, deadband_m=.03,
        max_output_rpm=200., rise_rpm_per_sec=rise, ego_forward_rpm=ego,
        range_rate_m_s=rate, raw_closure_valid=True, raw_distance_m=distance), **changes))


@pytest.mark.parametrize("distance", [1.43, 1.48659, 1.53736, 1.57567, 1.7, 1.736, 1.8, 1.93, 2.4, 4.])
def test_same_state_never_increases_demand_or_brake_cap_over_old_trial(distance):
    new = update(controller(), distance=distance)
    old = update(controller(launch_full_error_m=0.), distance=distance)
    assert new.demand_rpm <= old.demand_rpm + 1e-9
    assert new.cap_rpm == old.cap_rpm
    assert new.output_rpm <= old.output_rpm
    assert not new.software_rise_bypassed


@pytest.mark.parametrize("distance", [1.48659, 1.53736, 1.57567, 1.47262])
def test_near_distance_examples_are_pi_led_not_180(distance):
    result = update(controller(), distance=distance)
    assert result.demand_rpm == result.pi_demand_rpm < 40
    assert result.launch_floor_rpm < result.pi_demand_rpm


@pytest.mark.parametrize("error", [.03, .03 + 3.*60/.816814/720., .53])
def test_no_demand_step_at_deadband_pi_intersection_or_full_boost(error):
    left = update(controller(), distance=1.4+error-1e-7)
    right = update(controller(), distance=1.4+error+1e-7)
    assert 0 <= right.demand_rpm-left.demand_rpm < .001


def test_far_180_request_and_normal_ramp_start_without_a_pulse():
    c = controller()
    outputs = [update(c, distance=2.5, stamp=100+i*.05, ego=0.).output_rpm
               for i in range(5)]
    assert outputs[0] == 0  # no artificial acceleration credit
    assert outputs[1] > 0
    assert all(0 < b-a <= 9 for a, b in zip(outputs, outputs[1:]))
    assert c.last_result.launch_floor_rpm == 180
    assert not c.last_result.software_rise_bypassed


def test_revocation_does_not_restart_180_or_use_the_blind_gap_as_ramp_credit():
    c = controller()
    update(c, distance=2.5, ego=40.)
    c.suspend(100.02, "test_expiry", reset_execution=True)
    r = update(c, distance=2.5, stamp=100.1, ego=10.)
    assert r.status == "recovering" and r.output_rpm <= 10
    r = update(c, distance=2.5, stamp=100.15, ego=10.)
    assert r.output_rpm <= 19 and not r.software_rise_bypassed


@pytest.mark.parametrize("distance,ego,rate", [(1.508, 50.5, -.8), (1.4, 40., -.5), (1.7, 80., -2.)])
def test_real_fast_approach_and_setpoint_protection_are_unchanged(distance, ego, rate):
    old = update(controller(launch_full_error_m=0.), distance=distance, ego=ego, rate=rate)
    new = update(controller(), distance=distance, ego=ego, rate=rate)
    assert new.cap_rpm == old.cap_rpm
    assert new.output_rpm == old.output_rpm == 0


def test_memory_components_are_diagnostics_not_a_relaxed_bound():
    c = controller()
    update(c, distance=1.8, ego=30., rate=-.1)
    r = update(c, distance=1.79, stamp=100.1, ego=30., rate=None,
               raw_closure_valid=False, allow_motion_memory=True,
               motion_memory_rotation_bound=.35)
    assert r.brake_source == "relative_motion_memory"
    assert r.memory_time_penalty_m_s == pytest.approx(.2)
    assert r.memory_rotation_penalty_m_s == pytest.approx(.1)
    assert r.motion_uncertainty_m_s == pytest.approx(.3)
    assert r.memory_retained_rate_m_s == pytest.approx(-.4)
    assert r.memory_endpoint_rate_m_s == pytest.approx(-.45)
    assert r.effective_range_rate_m_s == pytest.approx(-.45)
    assert r.target_velocity_bound_m_s == pytest.approx(30*.816814/60-.45)
    assert r.target_velocity_bound_m_s < 0  # never clip away approaching-person evidence


@pytest.mark.parametrize("value", [-.1, 2.01, float("nan"), float("inf"), True])
def test_invalid_taper_rejected(value):
    with pytest.raises(ValueError, match="launch_full_error_m"):
        controller(launch_full_error_m=value)


@pytest.fixture
def tapered(authority, setup):
    a = authority
    _, a.controller, _ = configured(
        setup, distance_pi_kp_per_sec=3., distance_pi_launch_request_rpm=180.,
        distance_pi_launch_full_error_m=.5, distance_pid_output_rise_rpm_per_sec=180.,
        depth_longitudinal_sample_max_age_sec=.25,
    )
    a.owner._follow_controller = a.controller
    bind_production_reader(a.owner)
    return a


@pytest.mark.parametrize("visual", [False, True])
def test_real_controller_commit_writer_starts_in_three_fresh_frames(tapered, visual):
    a = tapered
    start = a.clock.now
    approved = []
    for i in range(3):
        advance(a, start+i*.05)
        current = a.frame(2.5, rpm=0.)
        decision = a.controller.decide(10, current, longitudinal_only=not visual)
        actions, _ = a.owner._commit_depth_linear_decision(decision, current, 1, is_fresh_depth=True)
        approved.append(max([x.speed_percent*2 for x in actions if x.kind == "forward"] or [0]))
    assert approved[0] == 0 < approved[1] <= approved[2] <= 18
    r = a.controller.last_distance_pid_result
    assert r.pi_launch_floor_rpm == 180 and not r.pi_software_rise_bypassed
    action, backend = writer(a)
    action.get_steering_feedback = lambda: current.steering_feedback
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (approved[-1], -approved[-1])


def test_new_mode_duplicate_and_expired_samples_cannot_extend_authority(tapered):
    a = tapered
    start = a.clock.now
    for i in range(3):
        advance(a, start+i*.05)
        current = a.frame(2.5, rpm=0.)
        decide_commit(a, current)
    stamp = current.distance_state.sample_timestamp
    original = a.owner._depth30_linear_timing
    advance(a, stamp+.1)
    decide_commit(a, current)
    assert a.owner._depth30_linear_timing.depth_expires_at == original.depth_expires_at
    advance(a, stamp+.251)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    decide_commit(a, current)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
