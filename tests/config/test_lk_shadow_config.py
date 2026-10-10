import os
from pathlib import Path

from car_control_modular.config_loader import load_config_to_env
from rk_vision.pipeline import RKNNVisionConfig


ROOT = Path(__file__).resolve().parents[2]


def test_shadow_library_default_off_and_board_diagnostic_trial_on(monkeypatch):
    monkeypatch.setattr(os, 'environ', {})
    assert RKNNVisionConfig(yolo_model_path='unused').lk_shadow_enable is False
    assert RKNNVisionConfig.from_env().lk_shadow_enable is False
    load_config_to_env(str(ROOT / 'car_control_modular/config/reid_runtime.ini'))
    config = RKNNVisionConfig.from_env()
    assert config.lk_shadow_enable is True
    assert config.lk_shadow_width == 320
    assert config.lk_shadow_correction_interval_sec == .3


def test_explicit_environment_rollback_overrides_enabled_ini(monkeypatch):
    monkeypatch.setattr(os, 'environ', {'Y8_LK_SHADOW_ENABLE': '0'})
    load_config_to_env(str(ROOT / 'car_control_modular/config/reid_runtime.ini'))
    assert RKNNVisionConfig.from_env().lk_shadow_enable is False


def test_enabled_ini_and_environment_rollback(tmp_path, monkeypatch):
    config = tmp_path / 'shadow.ini'
    config.write_text('[lk_shadow]\nenabled=true\nwidth=240\ncorrection_interval_sec=.4\n')
    monkeypatch.setattr(os, 'environ', {})
    load_config_to_env(str(config))
    assert RKNNVisionConfig.from_env().lk_shadow_enable is True
    assert RKNNVisionConfig.from_env().lk_shadow_width == 240
    assert RKNNVisionConfig.from_env().lk_shadow_correction_interval_sec == .4
    monkeypatch.setenv('Y8_LK_SHADOW_ENABLE', '0')
    load_config_to_env(str(config))
    assert RKNNVisionConfig.from_env().lk_shadow_enable is False
