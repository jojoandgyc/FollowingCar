"""Paired execution restores the existing PI tuning, not a P-only speed table."""
import os
from pathlib import Path

import pytest

from car_control_modular.config_loader import load_config_to_env
from car_control_modular.short_follow import ShortFollowConfig


CONFIG = Path(__file__).resolve().parents[2] / "car_control_modular/config/reid_runtime.ini"


def test_production_ini_selects_paired_pi_with_shared_tuning_and_motor_ceiling(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    load_config_to_env(str(CONFIG))
    cfg = ShortFollowConfig.from_env()
    assert cfg.enabled and cfg.max_rpm == 200
    assert cfg.kp_per_sec == 3.0 and cfg.ki_per_sec2 == .4
    assert cfg.integral_max_m_s == .8 and cfg.memory_sec == .35
    assert cfg.deadband_m == .03
    assert cfg.wheel_circumference_m == .816814
    assert cfg.braking_stop_distance_m == 1.1
    assert cfg.deceleration_m_s2 == 1.0 and cfg.response_delay_sec == .15
    assert cfg.target_distance_m == 1.4
    assert cfg.depth_ttl_sec == .35 and cfg.visual_ttl_sec == .50
    assert cfg.max_integral_gap_sec == .30
    assert cfg.yaw_max_delta_rpm == 16 and cfg.yaw_full_error_ratio == .30
    assert cfg.yaw_response_exponent == .5 and cfg.yaw_understeer_reduction_rpm == 8
    assert cfg.yaw_camera_hfov_deg == 60 and cfg.yaw_damping_sec == .10
    assert cfg.pivot_max_rpm == 8 and cfg.center_deadband_ratio == .08
    assert os.environ["DISTANCE_CONTROL_MODE"] == "distance_pi"
    assert "SHORT_FOLLOW_MIN_RPM" not in os.environ
    assert "SHORT_FOLLOW_MAX_RPM" not in os.environ
    assert "SHORT_FOLLOW_KP_RPM_PER_M" not in os.environ


def test_paired_pi_reads_same_pi_tuning_and_does_not_restore_180rpm_launch_boost(monkeypatch):
    monkeypatch.setattr(os, "environ", {
        "FOLLOW_NORMAL_MODE": "paired", "DISTANCE_PI_KP_PER_SEC": "2.5",
        "DISTANCE_PI_KI_PER_SEC2": ".6", "DISTANCE_PI_INTEGRAL_MAX_M_S": ".7",
        "DISTANCE_PI_MEMORY_SEC": ".34", "DISTANCE_PI_LAUNCH_REQUEST_RPM": "180",
        "DISTANCE_PID_DEADBAND_M": ".02", "FORWARD_MAX_RPM": "180",
        "MOTOR_FORWARD_MAX_TARGET_RPM": "150",
    })
    cfg = ShortFollowConfig.from_env()
    assert cfg.kp_per_sec == 2.5 and cfg.ki_per_sec2 == .6
    assert cfg.integral_max_m_s == .7 and cfg.memory_sec == .34
    assert cfg.deadband_m == .02 and cfg.max_rpm == 150
    assert not hasattr(cfg, "launch_request_rpm")
    assert not hasattr(cfg, "min_rpm") and not hasattr(cfg, "kp_rpm_per_m")


def test_removed_p_only_environment_cannot_silently_restore_40rpm_ceiling(monkeypatch):
    monkeypatch.setattr(os, "environ", {
        "FOLLOW_NORMAL_MODE": "paired", "SHORT_FOLLOW_MIN_RPM": "16",
        "SHORT_FOLLOW_MAX_RPM": "40", "SHORT_FOLLOW_KP_RPM_PER_M": "60",
    })
    cfg = ShortFollowConfig.from_env()
    assert cfg.max_rpm == 200 and cfg.kp_per_sec == 3. and cfg.ki_per_sec2 == .4


def test_explicit_legacy_override_is_rollback_not_a_second_writer(monkeypatch):
    monkeypatch.setattr(os, "environ", {"FOLLOW_NORMAL_MODE": "legacy"})
    load_config_to_env(str(CONFIG))
    assert not ShortFollowConfig.from_env().enabled


def test_old_environment_without_mode_does_not_enable_new_path(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    assert not ShortFollowConfig.from_env().enabled


def test_default_forward_differential_is_16_without_changing_pivot_limit(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    for cfg in (ShortFollowConfig(), ShortFollowConfig.from_env()):
        assert cfg.yaw_max_delta_rpm == 16
        assert cfg.pivot_limit_rpm == 8


@pytest.mark.parametrize("change", [
    {"max_rpm": 201}, {"max_rpm": 0}, {"max_rpm": 39.5},
    {"depth_ttl_sec": .351}, {"visual_ttl_sec": .501},
    {"stop_margin_m": .20}, {"restart_margin_m": .1},
    {"yaw_max_delta_rpm": 25}, {"center_deadband_ratio": .5},
    {"yaw_full_error_ratio": .08}, {"yaw_full_error_ratio": .501},
    {"yaw_full_error_ratio": float("nan")},
    {"pivot_max_rpm": -1}, {"pivot_max_rpm": 9}, {"pivot_max_rpm": 3.5},
    {"kp_per_sec": float("nan")}, {"ki_per_sec2": -.1},
    {"integral_max_m_s": -.1}, {"write_period_sec": 0},
    {"target_distance_m": 0}, {"stop_refresh_sec": 2},
])
def test_invalid_pi_or_unbounded_configuration_fails_closed(change):
    with pytest.raises(ValueError):
        ShortFollowConfig(**change)


@pytest.mark.parametrize("mode", ["off", "fast", "", "mixed"])
def test_invalid_mode_cannot_silently_select_another_path(monkeypatch, mode):
    monkeypatch.setattr(os, "environ", {"FOLLOW_NORMAL_MODE": mode})
    with pytest.raises(ValueError):
        ShortFollowConfig.from_env()


def test_paired_turning_overrides_load_without_touching_distance_pi(monkeypatch):
    monkeypatch.setattr(os, "environ", {
        "SHORT_FOLLOW_YAW_MAX_DELTA_RPM": "20",
        "SHORT_FOLLOW_YAW_FULL_ERROR_RATIO": ".25",
        "SHORT_FOLLOW_PIVOT_MAX_RPM": "6",
    })
    load_config_to_env(str(CONFIG))
    cfg = ShortFollowConfig.from_env()
    assert (cfg.yaw_max_delta_rpm, cfg.yaw_full_error_ratio, cfg.pivot_max_rpm) == (20, .25, 6)
    assert (cfg.kp_per_sec, cfg.ki_per_sec2, cfg.max_rpm) == (3., .4, 200)


def test_delivery_trial_does_not_change_library_default_or_integral_memory(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    cfg = ShortFollowConfig.from_env()
    assert cfg.depth_ttl_sec == .30
    monkeypatch.setenv("SHORT_FOLLOW_DEPTH_TTL_SEC", ".35")
    trial = ShortFollowConfig.from_env()
    assert trial.depth_ttl_sec == .35
    assert trial.max_integral_gap_sec == .30 and trial.memory_sec == .35


def test_production_delivery_trial_can_be_overridden_to_original_watchdog(monkeypatch):
    monkeypatch.setattr(os, "environ", {"SHORT_FOLLOW_DEPTH_TTL_SEC": ".30"})
    load_config_to_env(str(CONFIG))
    cfg = ShortFollowConfig.from_env()
    assert cfg.depth_ttl_sec == .30


def test_yaw_damping_override_does_not_change_rpm_or_deadlines(monkeypatch):
    monkeypatch.setattr(os, "environ", {"SHORT_FOLLOW_YAW_DAMPING_SEC": "0"})
    load_config_to_env(str(CONFIG))
    cfg = ShortFollowConfig.from_env()
    assert cfg.yaw_damping_sec == 0
    assert cfg.yaw_max_delta_rpm == 16 and cfg.pivot_max_rpm == 8
    assert cfg.depth_ttl_sec == .35 and cfg.visual_ttl_sec == .50


def test_turn_response_trial_can_roll_back_without_changing_limits_or_distance_pi(monkeypatch):
    monkeypatch.setattr(os, "environ", {
        "SHORT_FOLLOW_YAW_RESPONSE_EXPONENT": "1",
        "SHORT_FOLLOW_YAW_UNDERSTEER_REDUCTION_RPM": "0",
    })
    load_config_to_env(str(CONFIG))
    cfg = ShortFollowConfig.from_env()
    assert cfg.yaw_response_exponent == 1 and cfg.yaw_understeer_reduction_rpm == 0
    assert cfg.yaw_max_delta_rpm == 16 and cfg.pivot_max_rpm == 8
    assert (cfg.kp_per_sec, cfg.ki_per_sec2, cfg.max_rpm) == (3., .4, 200)
    assert cfg.depth_ttl_sec == .35 and cfg.visual_ttl_sec == .50
