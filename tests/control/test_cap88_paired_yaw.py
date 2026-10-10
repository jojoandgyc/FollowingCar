"""CAP83..99: current visual yaw must not wait for another depth sample."""
from dataclasses import replace

import pytest

from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowObservation,
    ShortFollowYawObservation,
)
from car_control_modular.short_follow_yaw import tapered_center


def setup(distance=2., x=.294, **config):
    core = ShortFollowController(ShortFollowConfig(enabled=True, **config))
    core.activate(1, 99.)
    plan = core.update(ShortFollowObservation(1, 83, 99.98, 99.99,
                                             distance, x), 100.)
    assert plan is not None
    return core, plan


def yaw(cap=88, stamp=100.05, x=.430, **kwargs):
    return ShortFollowYawObservation(1, cap, stamp, x, **kwargs)


@pytest.mark.parametrize("distance", [1.18, 1.35, 2.])
def test_cap88_detector_center_removes_old_left_yaw_without_waiting_for_depth(distance):
    core, original = setup(distance)
    assert original.left_rpm < original.right_rpm
    live = core.update_lateral(yaw(), 100.06)
    assert live is not None
    assert live.left_rpm == live.right_rpm == original.base_rpm
    assert live.depth_timestamp == original.depth_timestamp
    assert live.capture_timestamp == original.capture_timestamp
    assert live.expires_at == original.expires_at
    assert live.sequence > original.sequence and live.epoch == original.epoch
    assert live.yaw_capture_id == 88
    assert live.yaw_center_x_ratio == .430
    assert live.longitudinal_reason == original.longitudinal_reason


def test_lateral_publication_cannot_change_pi_values_or_memory():
    core, original = setup(1.56)
    second = core.update(ShortFollowObservation(1, 84, 100.01, 100.02, 1.56, .29), 100.03)
    old_memory = core._integral_m_s, core._integral_stamp, core._parked
    refreshed = core.update_lateral(yaw(), 100.06)
    names = ("capture_id", "capture_timestamp", "depth_timestamp", "expires_at", "distance_m",
             "base_request_rpm", "base_rpm", "speed_cap_rpm", "p_rpm", "i_rpm", "integral_dt_sec", "limit_reason")
    for name in names:
        assert getattr(refreshed, name) == getattr(second, name), name
    assert old_memory == (core._integral_m_s, core._integral_stamp, core._parked)


def test_cap92_visual_can_center_old_pair_before_new_depth_arrives():
    core, _ = setup(1.35)
    left = core.update_lateral(yaw(88, 100.03, .373), 100.04)
    assert left.pivot and left.left_rpm < 0
    center = core.update_lateral(yaw(92, 100.08, .564), 100.09)
    assert not center.moving and center.reason == "target_distance_reached"
    old_frame_depth = core.update(ShortFollowObservation(1, 88, 100.03, 100.10, 1.35, .373), 100.11)
    assert old_frame_depth.yaw_capture_id == 92
    assert not old_frame_depth.moving


def test_new_depth_changes_base_without_rolling_back_newer_yaw():
    core, original = setup()
    core.update_lateral(yaw(92, 100.04, .75), 100.05)
    later = core.update(ShortFollowObservation(1, 88, 100.02, 100.08, 2.2, .2), 100.09)
    assert later.yaw_capture_id == 92 and later.reason == "steer_right"
    assert later.left_rpm > later.right_rpm > 0
    assert later.base_rpm != original.base_rpm


def test_first_lateral_observation_is_only_cached_until_depth_is_available():
    core = ShortFollowController(ShortFollowConfig(enabled=True))
    core.activate(1, 99.)
    assert core.update_lateral(yaw(), 100.06) is None
    assert core.snapshot().plan is None
    plan = core.update(ShortFollowObservation(1, 83, 99.98, 100.07, 2., .1), 100.08)
    assert plan.left_rpm == plan.right_rpm > 0
    assert plan.yaw_capture_id == 88


@pytest.mark.parametrize("bad", [
    {"uid": 2}, {"uid": True}, {"capture_id": 83}, {"capture_id": True},
    {"capture_timestamp": 99.90}, {"capture_timestamp": 100.2},
    {"center_x_ratio": float("nan")}, {"center_x_ratio": -1},
    {"capture_yaw_deg": float("inf")},
])
def test_invalid_duplicate_or_out_of_order_yaw_preserves_existing_plan(bad):
    core, original = setup()
    assert core.update_lateral(replace(yaw(), **bad), 100.06) is None
    assert core.snapshot().plan is original


def test_duplicate_visual_does_not_update_again_or_refresh_either_deadline():
    core, original = setup()
    first = core.update_lateral(yaw(), 100.06)
    assert core.update_lateral(yaw(), 100.15) is None
    assert core.snapshot().plan is first
    assert not first.valid(original.expires_at)
    assert core.update_lateral(yaw(99, 100.4, .8), 100.41) is None
    assert not core.snapshot().plan.valid(100.41)


def test_revocation_clears_old_yaw_and_cannot_be_reversed_by_visual_only():
    core, _ = setup()
    core.update_lateral(yaw(), 100.06)
    core.revoke("hazard", 100.1)
    assert core._latest_yaw is None
    assert core.update_lateral(yaw(95, 100.09, .8), 100.12) is None
    assert core.snapshot().plan is None
    assert core.update_lateral(yaw(96, 100.11, .8), 100.12) is None
    assert core.snapshot().plan is None


@pytest.mark.parametrize("guard", [lambda: False, lambda: 1/0])
def test_publication_guard_failure_does_not_publish_or_clear_old_yaw(guard):
    core, original = setup()
    prior = core._latest_yaw
    assert core.update_lateral(yaw(), 100.06, publication_guard=guard) is None
    assert core.snapshot().plan is original and core._latest_yaw is prior


@pytest.mark.parametrize("sign", [-1, 1])
def test_measured_capture_to_now_yaw_and_rate_reduce_same_side_only(sign):
    obs = yaw(x=.5+sign*.20, capture_yaw_deg=100.)
    center, reason = tapered_center(obs, current_yaw_deg=100.+sign*6., yaw_rate_deg_s=sign*12.)
    assert center == pytest.approx(.5+sign*.08)
    assert reason == "yaw_rate_taper"
    crossed, _ = tapered_center(obs, current_yaw_deg=100.+sign*30., yaw_rate_deg_s=sign*50.)
    assert crossed == .5
    away, reason = tapered_center(obs, current_yaw_deg=100.-sign*6., yaw_rate_deg_s=-sign*12.)
    assert away == obs.center_x_ratio and reason == "none"


@pytest.mark.parametrize("feedback", [
    {}, {"current_yaw_deg": None, "yaw_rate_deg_s": None},
    {"current_yaw_deg": float("nan"), "yaw_rate_deg_s": float("inf")},
    {"current_yaw_deg": 5000., "yaw_rate_deg_s": 5000.},
])
def test_missing_or_invalid_feedback_keeps_measured_visual_request(feedback):
    obs = yaw(x=.2, capture_yaw_deg=0.)
    assert tapered_center(obs, **feedback) == (.2, "none")


def test_no_capture_aligned_heading_cannot_use_current_heading_as_rotation():
    obs = yaw(x=.2)
    center, reason = tapered_center(obs, current_yaw_deg=-30.)
    assert center == .2 and reason == "none"


@pytest.mark.parametrize("distance", [1.35, 2.])
@pytest.mark.parametrize("sign", [-1, 1])
def test_write_time_feedback_tapers_yaw_without_changing_authority_or_base(distance, sign):
    core, _ = setup(distance, x=.5+sign*.2)
    plan = core.update_lateral(yaw(x=.5+sign*.2, capture_yaw_deg=0.), 100.06)
    executed = core.execution_plan(plan, 100.1, current_yaw_deg=sign*9., yaw_rate_deg_s=sign*12.)
    assert executed.left_rpm == executed.right_rpm == plan.base_rpm
    assert executed.sequence == plan.sequence
    assert executed.epoch == plan.epoch and executed.expires_at == plan.expires_at
    assert executed.base_rpm == plan.base_rpm
    assert core.snapshot().plan is plan
    assert executed.forwarding == (distance > 1.5)


def test_execution_cannot_regrow_an_already_tapered_plan_when_feedback_recedes():
    core, _ = setup(x=.8)
    plan = core.update_lateral(yaw(x=.8, capture_yaw_deg=0.), 100.06, current_yaw_deg=9.)
    proposal = core.execution_plan(plan, 100.10, current_yaw_deg=1.)
    assert proposal is plan


@pytest.mark.parametrize("distance", [1.35, 2.])
@pytest.mark.parametrize("feedback", [{}, {"current_yaw_deg": 0., "yaw_rate_deg_s": 0.}])
def test_same_visual_new_depth_cannot_regrow_taper_before_first_motor_write(distance, feedback):
    core, _ = setup(distance=distance)
    centered = core.update_lateral(yaw(x=.3, capture_yaw_deg=0.), 100.06,
                                   current_yaw_deg=-12., yaw_rate_deg_s=-20.)
    assert centered.left_rpm == centered.right_rpm
    # Deliberately no executor/ack between these producer publications.
    later = core.update(ShortFollowObservation(1, 88, 100.05, 100.07,
                                              distance, .3), 100.08, **feedback)
    assert later.left_rpm == later.right_rpm == later.base_rpm
    assert later.yaw_control_center_x_ratio == .5
    assert later.yaw_adjustment_reason == "retained_yaw_taper"
    assert later.yaw_capture_id == centered.yaw_capture_id
    renewed = core.update_lateral(yaw(92, 100.09, .3), 100.1)
    assert renewed.left_rpm < renewed.right_rpm


def test_cached_visual_taper_is_preserved_by_the_first_depth_publication():
    core = ShortFollowController(ShortFollowConfig(enabled=True))
    core.activate(1, 99.)
    assert core.update_lateral(yaw(x=.3, capture_yaw_deg=0.), 100.06,
                              current_yaw_deg=-12.) is None
    plan = core.update(ShortFollowObservation(1, 88, 100.05, 100.07, 1.35, .3), 100.08)
    assert plan.left_rpm == plan.right_rpm == 0


def test_same_yaw_geometric_bound_does_not_freeze_new_longitudinal_base():
    core, _ = setup(distance=1.35)
    core.update_lateral(yaw(x=.3, capture_yaw_deg=0.), 100.06, current_yaw_deg=-12.)
    farther = core.update(ShortFollowObservation(1, 88, 100.05, 100.07, 2., .3), 100.08)
    assert farther.left_rpm == farther.right_rpm > 0
    assert farther.base_rpm > 0 and farther.yaw_control_center_x_ratio == .5


def test_env_uses_the_same_camera_hfov_as_visual_geometry(monkeypatch):
    monkeypatch.setenv("VISION_HFOV_DEG", "61.5")
    cfg = ShortFollowConfig.from_env()
    assert cfg.yaw_camera_hfov_deg == 61.5


@pytest.mark.parametrize("bad", [{"yaw_camera_hfov_deg": 0}, {"yaw_camera_hfov_deg": float("nan")},
                                {"yaw_damping_sec": -.01}, {"yaw_damping_sec": .21}])
def test_yaw_parameters_are_finite_and_bounded(bad):
    with pytest.raises(ValueError):
        ShortFollowConfig(**bad)
