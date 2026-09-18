"""Opt-in launch requests cannot become an unconditional motor speed floor."""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.steering_pid import DistancePidConfig, LongitudinalDistancePid


def sample(c, stamp=100., *, distance=5.5, ego=0., rate=0., **kwargs):
    return c.update(distance, 1.5, sample_timestamp=stamp, execution_now=stamp,
                    deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=1.,
                    ego_forward_rpm=ego, range_rate_m_s=rate, raw_closure_valid=True, **kwargs)


def controller(**kwargs):
    return DistancePiController(DistancePiConfig(kp_per_sec=.1, launch_request_rpm=120., **kwargs))


def test_launch_at_standstill_is_120_request_without_software_rise():
    result = sample(controller())
    assert result.demand_rpm == result.launch_floor_rpm == result.output_rpm == 120
    assert result.pi_demand_rpm < 120
    assert result.software_rise_bypassed and not result.slew_limited
    assert result.sample_dt_sec == 0


@pytest.mark.parametrize("distance", [1.55, 1.6, 1.8, 2., 2.4])
def test_brake_cap_not_launch_floor_limits_nearby_target(distance):
    result = sample(controller(), distance=distance)
    assert result.demand_rpm == 120
    assert 0 <= result.output_rpm <= result.cap_rpm < 120
    assert result.demand_limit_reason == "braking_envelope"


@pytest.mark.parametrize("distance", [1.5, 1.4])
def test_at_or_inside_setpoint_has_no_launch(distance):
    result = sample(controller(), distance=distance)
    assert result.output_rpm == result.launch_floor_rpm == 0
    assert not result.software_rise_bypassed


def test_fast_approach_can_stop_even_with_120_demand():
    result = sample(controller(), distance=1.6, ego=120., rate=-2.)
    assert result.demand_rpm == 120
    assert result.cap_rpm == result.output_rpm == 0


@pytest.mark.parametrize("ego", [None, float("nan"), float("inf"), -300., 300.])
def test_bad_feedback_cannot_bypass_software_launch(ego):
    result = sample(controller(), ego=ego, rate=None)
    assert result.launch_floor_rpm == result.output_rpm == 0
    assert not result.software_rise_bypassed


def test_duplicate_old_or_rejected_samples_cannot_create_new_launch():
    c = controller(physical_ttl_sec=.25)
    first = sample(c)
    before = (c._last_sample_ts, c._last_execution_ts, c.integral_m_s)
    assert sample(c).status == "duplicate"
    assert sample(c, 99.9).output_rpm == 0
    assert (c._last_sample_ts, c._last_execution_ts, c.integral_m_s) == before
    c.suspend(100.1, "revoked", reset_execution=True)
    assert sample(c).output_rpm == 0
    late = c.update(5.5, 1.5, sample_timestamp=100., execution_now=100.2,
                    deadband_m=.03, max_output_rpm=200, ego_forward_rpm=0.)
    assert late.status == "continuation_only" and late.output_rpm == 0
    assert late.launch_floor_rpm == 0


def test_disabling_trial_is_exact_old_profile():
    base = DistancePiConfig(kp_per_sec=.1)
    a, b = DistancePiController(base), DistancePiController(replace(base, launch_request_rpm=0.))
    for index, distance in enumerate((1.6, 1.8, 1.7, 1.5)):
        assert sample(a, 100.+index*.05, distance=distance) == sample(b, 100.+index*.05, distance=distance)


@pytest.mark.parametrize("value", [-1., 201., float("nan"), float("inf"), True])
def test_invalid_launch_configuration_rejected(value):
    with pytest.raises(ValueError, match="launch_request_rpm"):
        DistancePiConfig(launch_request_rpm=value)


def test_launch_respects_lower_total_budget():
    c = controller()
    r = c.update(8., 1.5, sample_timestamp=100., execution_now=100.,
                 deadband_m=.03, max_output_rpm=80., ego_forward_rpm=0.,
                 range_rate_m_s=0., raw_closure_valid=True)
    assert r.output_rpm == 80 < r.launch_floor_rpm
    assert r.demand_limit_reason == "total_rpm_cap"


def test_reverse_output_is_unchanged_by_launch_trial():
    controllers = [LongitudinalDistancePid(DistancePidConfig(
        pi_profile=DistancePiConfig(launch_request_rpm=rpm), output_rise_rpm_per_sec=240.))
        for rpm in (0., 120.)]
    for index in range(3):
        results = [c.update(1.2, 1.5, now=100.+index*.1, forward_control=False) for c in controllers]
        assert results[0] == results[1]
        assert results[1].output_rpm < 0 and results[1].approach_mode == "legacy_pid"


@pytest.mark.parametrize("distance", [1.8, 5.5, 10.])
def test_180_request_is_not_a_floor_on_braking(distance):
    c = DistancePiController(DistancePiConfig(kp_per_sec=.1, launch_request_rpm=180.))
    r = sample(c, distance=distance)
    assert r.launch_floor_rpm == r.demand_rpm == 180.
    assert 0 < r.output_rpm <= min(180., r.cap_rpm)
    assert r.software_rise_bypassed
    if distance == 1.8:
        assert r.output_rpm < 60
    if distance == 10.:
        # Pure synthetic range: proves the requested value is not still120,
        # not a claim that the real depth sensor has a10m usable range.
        assert r.output_rpm == 180
