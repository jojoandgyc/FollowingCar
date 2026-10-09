"""Do not mistake body yaw for centering a far, outward-moving person.

Pure controller tests: no camera, motor, serial or changes to motion authority.
"""
from dataclasses import replace

import pytest

from car_control_modular.control_types import SteeringFeedback
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig


def config():
    return VisualSteeringPidConfig(enabled=True, image_error_only=True,
        image_brake_assist=True, image_capture_motion=True,
        execution_response_trial_sec=.35, camera_hfov_deg=66,
        deadband_deg=3, dynamic_large_error_deg=10, max_correction_rpm=10,
        max_yaw_rate_dps=35, predictive_brake_response_sec=.1,
        predictive_brake_margin_deg=1.25, predictive_brake_decel_dps2=60)


def feedback(yaw=25., **changes):
    return replace(SteeringFeedback(timestamp=9.98, trustworthy=True,
        yaw_rate_right_dps=yaw, raw_yaw_rate_right_dps=yaw,
        left_forward_rpm=48, right_forward_rpm=32), **changes)


def result(*, error=18.43, rate=6., yaw=25., age=.20, forward=True,
           base=60, override=None, cfg=None, fb=None):
    return VisualSteeringPid(cfg or config()).update(.5+error/66., base,
        feedback(yaw) if fb is None else fb, now=10.,
        visual_age_sec=age, target_image_rate_dps=rate,
        max_correction_override_rpm=override, forward_tracking=forward)


@pytest.mark.parametrize("sign", [-1, 1])
def test_fresh_far_outward_retains_position_demand_instead_of_four_rpm(sign):
    legacy = result(error=sign*18.43, rate=sign*6, yaw=sign*25, forward=False)
    fixed = result(error=sign*18.43, rate=sign*6, yaw=sign*25)
    assert legacy.correction_rpm == sign*4
    assert fixed.correction_rpm == sign*10
    assert fixed.base_rpm == fixed.requested_base_rpm == 60
    assert fixed.forward_phase == "image_forward_outward_tracking"
    assert fixed.forward_tracking_active and fixed.feedback_used
    assert fixed.brake_reduction_rpm == 0
    assert not fixed.predictive_braking


@pytest.mark.parametrize("changes", [dict(error=15.9), dict(rate=-6.),
    dict(rate=None), dict(rate=0.), dict(rate=2.9), dict(rate=61.),
    dict(rate=float("nan")), dict(age=.251), dict(age=-.01),
    dict(age=float("nan")), dict(yaw=35.1), dict(base=0), dict(base=1),
    dict(base=10), dict(forward=False)])
def test_unknown_inward_near_center_parked_and_overspeed_keep_original_brakes(changes):
    fixed = result(**changes)
    conservative = result(**{**changes, "forward": False})
    assert fixed.correction_rpm == conservative.correction_rpm
    assert fixed.brake_reduction_rpm == conservative.brake_reduction_rpm
    assert not fixed.forward_tracking_active


def test_cap294_actual_overspeed_is_still_braked_not_called_false_braking():
    r = result(error=27.89, rate=6., yaw=48.6, age=.191)
    assert r.correction_rpm == 0 and r.predictive_braking
    assert not r.forward_tracking_active


@pytest.mark.parametrize("changes", [dict(trustworthy=False), dict(timestamp=9.8),
    dict(timestamp=10.01), dict(raw_yaw_rate_right_dps=50.),
    dict(raw_yaw_rate_right_dps=None), dict(raw_yaw_rate_right_dps=-25.)])
def test_new_relative_motion_mode_requires_qualified_encoder(changes):
    r = result(fb=feedback(**changes))
    assert not r.forward_tracking_active


@pytest.mark.parametrize("pair", [(-32, -48), (8, -8), (-.1, 16)])
def test_reverse_or_pivot_is_not_forward_tracking(pair):
    r = result(fb=feedback(left_forward_rpm=pair[0], right_forward_rpm=pair[1]))
    assert not r.forward_tracking_active
    assert r.correction_rpm == 4


@pytest.mark.parametrize("limit", [0., 3., 7., 10.])
def test_position_demand_still_respects_callers_limit(limit):
    assert abs(result(override=limit).correction_rpm) <= limit


def test_partial_crop_cue_cannot_enter_full_outward_mode():
    r = VisualSteeringPid(config()).update(.5+18.43/66, 60, feedback(),
        now=10., visual_age_sec=.2, outward_continuity_rate_dps=6.,
        forward_tracking=True)
    assert not r.forward_tracking_active
    assert r.correction_rpm <= 2


def test_fast_refresh_loses_exception_as_original_capture_ages():
    pid = VisualSteeringPid(config())
    fresh = pid.update(.5+18.43/66, 60, feedback(), now=10.,
        visual_age_sec=.20, target_image_rate_dps=6, forward_tracking=True)
    stale = pid.update(.5+18.43/66, 60, feedback(timestamp=10.06), now=10.06,
        visual_age_sec=.26, target_image_rate_dps=6, forward_tracking=True)
    assert fresh.correction_rpm == 10
    # Existing non-renewable residual-deceleration policy may retain <=2 RPM;
    # the new full-position exception must not survive the capture's age.
    assert stale.correction_rpm <= 2 and stale.brake_reduction_rpm > 0
    assert not stale.forward_tracking_active


def test_real_controller_main_and_fast_use_same_mode_and_near_mode_does_not():
    controller = FollowSafetyController(FollowPolicyConfig(
        visible_steering_pid_enable=True, visible_steering_pid_forward_tracking_enable=True))
    controller._visual_steering_pid.config = config()
    kwargs = dict(x_ratio=.5+18.43/66, base_rpm=60, feedback=feedback(),
                  now=10., visual_age_sec=.2, target_image_rate_dps=6)
    main = controller._update_lateral_pid(controller._visual_steering_pid, **kwargs)
    fast = controller.refresh_visible_lateral_pid(**kwargs)
    near = controller._update_lateral_pid(controller._visual_steering_pid,
                                          near_distance_mode=True, **kwargs)
    assert main.correction_rpm == fast.correction_rpm == 10
    assert main.forward_tracking_active and fast.forward_tracking_active
    assert not near.forward_tracking_active
    assert near.correction_rpm <= 4


@pytest.mark.parametrize("distance,percent", [(2., 0), (None, 20)])
def test_actual_action_zero_or_missing_distance_cannot_enable_exception(monkeypatch, distance, percent):
    from car_control_modular.control_types import PersonTarget, SensorFrame
    from car_control_modular.visual_steering_evidence import SteeringObservation
    c = FollowSafetyController(FollowPolicyConfig(visible_steering_pid_enable=True,
        visible_steering_pid_forward_tracking_enable=True))
    c._visual_steering_pid.config = config()
    x = .5+18.43/66
    monkeypatch.setattr(c, "_visible_base_forward_percent", lambda *a, **kw: percent)
    obs = SteeringObservation(1, 379, 9.8, x, x, x, 6., "capture_rate_valid", .4)
    monkeypatch.setattr(c, "_capture_steering_observation", lambda *a: obs)
    target = PersonTarget((x*640-40, 50, x*640+40, 450), 1, .95, 32000)
    frame = SensorFrame(width=640, height=480, distance_m=distance,
        capture_frame_id=379, capture_timestamp=9.8, steering_feedback=feedback())
    c._pid_action_for_visible_target(target, frame, 10.)
    assert not c.last_steering_pid_result.forward_tracking_active
    assert c.last_steering_pid_result.correction_rpm <= 4
