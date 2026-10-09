"""Launch demand must not hide the PI's available execution headroom."""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig
from car_control_modular.steering_pid import DistancePidConfig, LongitudinalDistancePid


SCALE = 60. / .816814
PROFILE = DistancePidConfig(
    pi_profile=DistancePiConfig(
        kp_per_sec=3., ki_per_sec2=.4, integral_max_m_s=.8,
        launch_request_rpm=180., launch_full_error_m=.5,
        physical_ttl_sec=.25, motion_memory_sec=.35,
    ),
    deadband_m=.03, max_forward_output_rpm=200.,
    output_rise_rpm_per_sec=240., output_fall_rpm_per_sec=300.,
)


def controller(**changes):
    return LongitudinalDistancePid(replace(PROFILE, **changes))


def sample(c, index, *, distance=1.83, ego=90., rate=0.):
    stamp = 100. + index*.05
    return c.update(
        distance, 1.4, now=stamp, execution_now=stamp,
        ego_forward_rpm=ego, braking_range_rate_m_s=rate,
        raw_closure_valid=True, braking_raw_distance_m=distance,
    )


def test_launch_ramp_preserves_integral_when_it_can_execute_the_whole_pi():
    c = controller()
    first = sample(c, 0)
    assert first.output_rpm == 90
    ramping = sample(c, 1)
    assert ramping.pi_demand_rpm < ramping.pi_pre_quantization_rpm < ramping.pi_total_demand_rpm
    assert ramping.output_rpm == 102
    assert ramping.pi_final_limit_reason == "software_slew"
    assert ramping.pi_integral_m_s == pytest.approx(.008)
    assert not ramping.pi_integral_frozen


def test_launch_ramp_still_freezes_when_it_cannot_execute_the_pi():
    c = controller()
    sample(c, 0, ego=60.)
    ramping = sample(c, 1, ego=60.)
    assert ramping.output_rpm == 72 < ramping.pi_demand_rpm
    assert ramping.pi_integral_m_s == 0
    assert ramping.pi_integral_frozen


@pytest.mark.parametrize("approved", [90., 100.])
def test_repeated_launch_limits_learn_only_the_available_pi_headroom(approved):
    c = controller()
    for index in range(201):
        result = sample(c, index, ego=approved)
        c.accept_output_limit(100. + index*.05, min(approved, result.output_rpm))
        actual = c.last_result
        assert actual.output_rpm <= approved
        assert actual.output_rpm <= actual.approach_cap_rpm
        assert actual.p_rpm + actual.i_rpm <= approved + 1e-9
    # Previously every sample rolled its increment back to zero despite the
    # executed command being higher than P+I. It must learn up to that command
    # and freeze before the next integration step would exceed it.
    assert c.last_result.pi_integral_m_s > 0
    assert approved - (c.last_result.p_rpm + c.last_result.i_rpm) < .008*SCALE
    assert c.last_result.pi_integral_frozen


def test_external_launch_limit_preserves_increment_without_software_slew():
    c = controller(output_rise_rpm_per_sec=0.)
    sample(c, 0, ego=100.)
    result = sample(c, 1, ego=100.)
    assert result.pi_demand_rpm < 100 < result.output_rpm
    assert c.accept_output_limit(100.05, 100.)
    assert c.last_result.output_rpm == 100
    assert c.last_result.pi_integral_m_s == pytest.approx(.008)
    assert not c.last_result.pi_integral_frozen


def test_successively_tighter_approvals_rollback_only_when_pi_is_limited():
    c = controller(output_rise_rpm_per_sec=0.)
    sample(c, 0, ego=100.)
    sample(c, 1, ego=100.)
    for approved in (100., 90.):
        assert c.accept_output_limit(100.05, approved)
        assert c.last_result.pi_integral_m_s == pytest.approx(.008)
        assert not c.last_result.pi_integral_frozen
    assert not c.accept_output_limit(100.05, 90.)
    assert not c.accept_output_limit(100.05, 100.)
    assert c.accept_output_limit(100.05, 80.)
    assert c.last_result.output_rpm == 80
    assert c.last_result.pi_integral_m_s == 0
    assert c.last_result.pi_integral_frozen
    assert not c.accept_output_limit(100.05, 90.)
    assert c.accept_output_limit(100.05, 0.)
    assert c.last_result.output_rpm == c.last_result.pi_integral_m_s == 0
    assert not c.accept_output_limit(100., 0.)


def test_rejected_sample_rolls_back_learning_even_if_pi_fits_approved_launch():
    c = controller()
    sample(c, 0)
    sample(c, 1)
    learned = c.last_result.pi_integral_m_s
    sample(c, 2)
    assert c.last_result.pi_integral_m_s > learned > 0
    assert c.reject_output(100.1)
    assert c.last_result.pi_integral_m_s == learned
    assert c.last_result.pi_integral_frozen
    assert not c.reject_output(100.1)
    assert c.accept_output_limit(100.1, 0.)
    assert c.last_result.output_rpm == c.last_result.pi_integral_m_s == 0
    assert sample(c, 2).pi_status == "suspended_duplicate"


def test_braking_envelope_still_blocks_integral_and_overrides_launch():
    c = controller()
    sample(c, 0)
    sample(c, 1)
    assert c.last_result.pi_integral_m_s > 0
    braking = sample(c, 2, ego=100., rate=-3.)
    assert braking.pi_launch_floor_rpm > 100
    assert braking.approach_cap_rpm == braking.output_rpm == 0
    assert braking.pi_integral_m_s == 0
    assert braking.pi_integral_frozen


def test_launch_profile_retains_normal_two_rpm_quantization_learning():
    c = controller()
    for index in range(201):
        result = sample(c, index, distance=1.5, ego=40.)
        c.accept_output_limit(100. + index*.05, 2*(result.output_rpm//2))
    assert c.last_result.pi_integral_m_s == pytest.approx(.28)
    assert c.last_result.output_rpm == 34


def test_parked_preview_cannot_learn_from_unexecuted_launch_demand():
    c = controller()
    sample(c, 0)
    sample(c, 1)
    learned = c.last_result.pi_integral_m_s
    c.set_normal_parking(True)
    for index in range(2, 8):
        result = sample(c, index)
        assert result.pi_status == "parked_preview"
        assert result.pi_sample_dt_sec == 0
        assert result.pi_integral_m_s == learned
        assert result.pi_integral_frozen
