"""Runtime INI: 10 RPM per wheel means 20 RPM differential; no hardware."""
import configparser
import os
from pathlib import Path

import pytest

from car_control_modular.config_loader import load_config_to_env
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig


@pytest.fixture
def trial_config(monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'car_control_modular/config/reid_runtime.ini'
    monkeypatch.setattr(os, 'environ', {})
    load_config_to_env(str(path))
    assert float(os.environ['VISIBLE_STEERING_PID_MAX_CORRECTION_RPM']) == 10
    parser = configparser.ConfigParser()
    parser.read(path)
    section = parser['steering_pid']
    return VisualSteeringPidConfig(
        enabled=True,
        image_error_only=section.getboolean('image_error_only'),
        image_brake_assist=section.getboolean('image_brake_assist'),
        image_capture_motion=section.getboolean('image_capture_motion'),
        camera_hfov_deg=section.getfloat('camera_hfov_deg'),
        deadband_deg=section.getfloat('deadband_deg'),
        dynamic_large_error_deg=section.getfloat('dynamic_large_error_deg'),
        max_correction_rpm=float(os.environ['VISIBLE_STEERING_PID_MAX_CORRECTION_RPM']),
    )


@pytest.mark.parametrize('x,difference', [(.8, 20), (.2, -20), (.5, 0)])
def test_trial_wheel_difference(trial_config, x, difference):
    result = VisualSteeringPid(trial_config).update(x, 80, None, now=10)
    assert result.base_rpm == 80
    assert 2 * result.correction_rpm == difference


def test_trial_retains_lower_policy_cap(trial_config):
    result = VisualSteeringPid(trial_config).update(
        .8, 80, None, now=10, max_correction_override_rpm=5)
    assert result.correction_rpm == 5


def test_trial_retains_visual_predictive_stop(trial_config):
    result = VisualSteeringPid(trial_config).update(
        .65, 80, None, now=10, visual_age_sec=.15,
        target_image_rate_dps=-80)
    assert result.predictive_braking
    assert result.correction_rpm == 0
