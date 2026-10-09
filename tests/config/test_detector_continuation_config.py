from configparser import ConfigParser
from pathlib import Path

import pytest

from car_control_modular.config_loader import _set_bool_env_if_present
from rk_vision.pipeline import RKNNVisionConfig


@pytest.mark.parametrize("enabled", [True, False])
def test_fast_path_switch_reaches_pipeline(monkeypatch, enabled):
    monkeypatch.setenv("Y8_DETECTOR_CONTINUATION_ENABLE", "0")
    parser = ConfigParser()
    parser.read_dict({"vision": {"detector_continuation_enable": str(enabled)}})
    _set_bool_env_if_present(parser, "vision", "detector_continuation_enable",
                             "Y8_DETECTOR_CONTINUATION_ENABLE")
    assert RKNNVisionConfig.from_env().detector_continuation_enable is enabled


def test_runtime_enables_fast_path_but_library_default_does_not():
    config = ConfigParser()
    config.read(Path(__file__).resolve().parents[2] / "car_control_modular/config/reid_runtime.ini")
    assert config.getboolean("vision", "detector_continuation_enable")
    assert not RKNNVisionConfig(yolo_model_path="unused").detector_continuation_enable
