"""Near-target yaw parking intent; controller-only, with no hardware access."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.control_types import (
    ControlAction, ControlDecision, PersonTarget, SensorFrame, SteeringFeedback,
)
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: 100.0)
    cfg = FollowPolicyConfig(
        visible_steering_pid_enable=True,
        visible_steering_pid_camera_hfov_deg=60.0,
        visible_steering_pid_camera_latency_sec=0.0,
        visible_steering_pid_predictive_brake_decel_dps2=50.0,
        visible_steering_pid_predictive_brake_response_sec=0.10,
        visible_steering_pid_predictive_brake_margin_deg=1.0,
        visible_steering_pid_deadband_deg=3.0,
        visible_steering_pid_outer_kp_per_sec=1.5,
        visible_steering_pid_error_filter_alpha=1.0,
        center_left_ratio=0.45,
        center_right_ratio=0.55,
        parked_recenter_min_rpm=5,
        parked_recenter_max_rpm=10,
        near_distance_rotation_only_max_rpm=10,
    )
    instance = FollowSafetyController(cfg)
    instance.last_dispatched_kind = "rotate_right"
    monkeypatch.setattr(instance, "_near_settle_hold_active", lambda *a, **kw: False)
    return instance


def decision(controller, *, x=0.66, yaw=0.0, steerable=True):
    target = PersonTarget((640 * x - 40, 60, 640 * x + 40, 440), 1, 0.95, 30400)
    frame = SensorFrame(
        width=640, height=480, persons=[target], distance_m=1.28,
        steering_feedback=SteeringFeedback(
            timestamp=100.0, yaw_rate_right_dps=yaw, trustworthy=True,
        ),
    )
    return controller._near_distance_rotation_only_decision(
        46, frame, target, 1.28, 1.53, target_steerable=steerable,
    )


def pid_result(monkeypatch, controller, *, floor="none", predictive=False, action=None):
    def update(*a, **kw):
        controller.last_steering_pid_result = SimpleNamespace(
            correction_rpm=0, output_floor_reason=floor, predictive_braking=predictive,
        )
        return action
    monkeypatch.setattr(controller, "_pid_action_for_parked_target", update)


def assert_park(result):
    assert result.near_yaw_park_requested
    assert not result.soft_stop_requested
    assert not result.explicit_stop_requested  # normal parking, not a hazard latch
    assert result.current_forward_percent == 0
    assert result.actions[0].kind == "stop"
    assert result.actions[0].brake_hold
    assert result.actions[0].reason == result.reason == "near_distance_rotation_only"


def test_default_decision_does_not_change_other_control_paths():
    assert not ControlDecision().near_yaw_park_requested
    assert not ControlDecision(actions=[ControlAction.forward(0, "depth_zero")]).near_yaw_park_requested


def test_center_settle_is_normal_parking_not_soft_zero(controller, monkeypatch, caplog):
    monkeypatch.setattr(controller, "_near_settle_hold_active", lambda *a, **kw: True)
    with caplog.at_level(logging.INFO):
        result = decision(controller, x=0.51)
    assert_park(result)
    assert "park_requested=True park_reason=near_distance_center_settle" in caplog.text


@pytest.mark.parametrize("floor,predictive", [
    ("predictive_brake_coast", True),
    ("predictive_brake_coast", False),
    ("center_hold", False),
    ("none", True),
])
def test_explicit_pid_braking_or_center_hold_requests_parking(controller, monkeypatch, floor, predictive):
    pid_result(monkeypatch, controller, floor=floor, predictive=predictive)
    assert_park(decision(controller))


@pytest.mark.parametrize("floor", ["none", "yaw_damping_coast", "same_direction_overspeed_coast"])
def test_other_pid_zeroes_remain_soft_zero(controller, monkeypatch, floor):
    pid_result(monkeypatch, controller, floor=floor)
    result = decision(controller)
    assert not result.near_yaw_park_requested
    assert result.soft_stop_requested
    assert not result.actions[0].brake_hold
    assert not result.stop_action_execution


def test_direction_guard_is_not_promoted_by_stale_pid_brake_metadata(controller, monkeypatch):
    pid_result(
        monkeypatch, controller, floor="predictive_brake_coast", predictive=True,
        action=ControlAction.stop("person_parked_direction_guard_hold", brake_hold=False),
    )
    result = decision(controller)
    assert not result.near_yaw_park_requested
    assert result.soft_stop_requested
    assert not result.actions[0].brake_hold


def test_real_center_fallback_requests_parking(controller, caplog):
    with caplog.at_level(logging.INFO):
        result = decision(controller, x=0.50)
    assert_park(result)
    assert "park_requested=True park_reason=center_fallback_hold" in caplog.text


def test_quality_guard_off_center_does_not_masquerade_as_center_hold(controller, monkeypatch):
    monkeypatch.setattr(controller, "_pid_action_for_parked_target", lambda *a, **kw: None)
    result = decision(controller, x=0.80, steerable=False)
    assert result.actions[0].kind == "stop"  # preserve the pre-existing quality hold
    assert not result.near_yaw_park_requested


def test_cap46_style_real_predictive_pid_zero_requests_parking(controller):
    result = decision(controller, x=0.60, yaw=19.53)
    pid = controller.last_steering_pid_result
    assert pid is not None
    assert pid.predictive_braking and pid.correction_rpm == 0
    assert_park(result)


def test_distance_below_target_still_allows_off_center_rotation(controller):
    result = decision(controller, x=0.721)
    assert result.actions[0].kind == "rotate_right"
    assert not result.near_yaw_park_requested
    assert not result.soft_stop_requested


def test_fresh_off_center_follow_can_leave_center_parking(controller):
    assert_park(decision(controller, x=0.50))
    controller.last_dispatched_kind = "stop"
    result = decision(controller, x=0.72)
    assert result.actions[0].kind == "rotate_right"
    assert not result.near_yaw_park_requested
    assert not result.stop_action_execution  # runtime owns controlled release
