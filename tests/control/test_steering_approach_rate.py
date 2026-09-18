"""Error-dependent closing rate retains wheel caps and center braking."""
import pytest
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig


def pid():
    return VisualSteeringPid(VisualSteeringPidConfig(
        enabled=True, camera_hfov_deg=60, camera_latency_sec=0,
        target_speed_match_max_closing_dps=8, outer_kp_per_sec=1.5,
        max_yaw_rate_dps=35, max_correction_rpm=10,
        dynamic_small_max_correction_rpm=5,
    ))


def update(controller, x, **changes):
    return controller.update(x, 40, SteeringFeedback(timestamp=100, trustworthy=True),
                             now=100, target_image_rate_dps=0, **changes)


@pytest.mark.parametrize("x", [.15, .85])
def test_large_error_gets_bounded_approach_not_new_rpm_cap(x):
    result = update(pid(), x)
    assert result.target_speed_match_limit_dps == pytest.approx(12)
    assert abs(result.desired_yaw_rate_dps) <= 12
    assert abs(result.correction_rpm) <= 10
    assert result.correction_rpm * (x-.5) > 0


@pytest.mark.parametrize("x", [.46, .54])
def test_near_center_keeps_original_eight_dps_closing_limit(x):
    result = update(pid(), x)
    assert result.target_speed_match_limit_dps == pytest.approx(8)


def test_center_and_explicit_hold_are_not_overridden():
    controller = pid()
    update(controller, .15)
    result = update(controller, .5, max_correction_override_rpm=0,
                    target_speed_match_max_closing_dps_override=0)
    assert result.correction_rpm == 0


def test_explicit_closing_limit_not_increased_by_approach():
    result = update(pid(), .85, target_speed_match_max_closing_dps_override=4)
    assert result.target_speed_match_limit_dps == 4
