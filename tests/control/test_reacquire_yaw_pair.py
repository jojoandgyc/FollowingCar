"""One wheel-pair owner can recenter without granting forward permission."""
from dataclasses import replace

import pytest

from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController, ShortFollowObservation
from car_control_modular.short_follow_yaw import ShortFollowYawObservation


def make_controller():
    controller = ShortFollowController(ShortFollowConfig(enabled=True))
    controller.activate(1, 9.9)
    return controller


@pytest.mark.parametrize('center,sign', [(.20, -1), (.50, 0), (.88, 1)])
@pytest.mark.parametrize('distance', [1.527, 2.2])
def test_first_accepted_range_yields_current_yaw_only_pair(center, sign, distance):
    ctl = make_controller()
    obs = ShortFollowObservation(1, 1542, 10., 10., distance, center)
    plan = ctl.update(obs, 10.14, longitudinal_allowed=False)
    assert plan is not None
    assert plan.base_rpm == plan.base_request_rpm == plan.i_rpm == plan.integral_dt_sec == 0
    assert plan.left_rpm + plan.right_rpm == 0
    assert plan.left_rpm * sign > 0 if sign else plan.left_rpm == plan.right_rpm == 0
    assert max(abs(plan.left_rpm), abs(plan.right_rpm)) <= 8
    assert plan.longitudinal_reason == 'reacquire_depth_pending'
    assert plan.distance_m == distance and plan.depth_timestamp == 10.
    assert plan.expires_at == pytest.approx(10.3)
    assert ctl._integral_m_s == 0


def test_new_position_changes_yaw_without_range_or_identity_renewal():
    ctl = make_controller()
    first = ctl.update(ShortFollowObservation(1, 1542, 10., 10., 1.527, .2),
                       10.14, longitudinal_allowed=False)
    assert first.pivot and first.left_rpm < 0
    next_plan = ctl.update_lateral(ShortFollowYawObservation(1, 1546, 10.20, .80), 10.24)
    assert next_plan.pivot and next_plan.left_rpm > 0
    assert next_plan.capture_id == first.capture_id and next_plan.yaw_capture_id == 1546
    assert next_plan.expires_at == first.expires_at
    assert next_plan.base_rpm == next_plan.i_rpm == 0
    assert ctl.update_lateral(ShortFollowYawObservation(1, 1548, 10.31, .85), 10.32) is None


@pytest.mark.parametrize('fault', ['expired', 'no_range', 'too_near', 'raw_near'])
def test_pending_permission_cannot_pivot_without_current_clearance(fault):
    ctl = make_controller()
    obs = ShortFollowObservation(1, 1542, 10., 10., 1.527, .8)
    now = 10.14
    if fault == 'expired': now = 10.31
    elif fault == 'no_range': obs = replace(obs, distance_m=None)
    elif fault == 'too_near': obs = replace(obs, distance_m=1.09)
    else: obs = replace(obs, raw_distance_m=1.09)
    plan = ctl.update(obs, now, longitudinal_allowed=False)
    assert plan is None or not plan.moving


def test_next_independent_range_promotes_same_owner_without_stop_epoch():
    ctl = make_controller()
    obs = ShortFollowObservation(1, 1542, 10., 10., 2.2, .2)
    first = ctl.update(obs, 10.14, longitudinal_allowed=False)
    # One physical range cannot be used twice to promote forward motion.
    assert ctl.update(obs, 10.15, longitudinal_allowed=True) is None
    next_plan = ctl.update(replace(obs, capture_id=1546, capture_timestamp=10.2,
        depth_timestamp=10.2), 10.24, longitudinal_allowed=True)
    assert next_plan.forwarding and next_plan.epoch == first.epoch
    assert next_plan.sequence == first.sequence + 1
    assert next_plan.base_rpm > 0 and next_plan.integral_dt_sec == 0


@pytest.mark.parametrize('invalid', [None, 0, 1, 'false'])
def test_longitudinal_permission_requires_explicit_boolean(invalid):
    ctl = make_controller()
    assert ctl.update(ShortFollowObservation(1, 1542, 10., 10., 2.2, .2), 10.14,
                      longitudinal_allowed=invalid) is None
