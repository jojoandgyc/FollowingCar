"""The continuity trial must reach the real bank, not just an unused INI."""
import os
from pathlib import Path

import pytest

from car_control_modular.config_loader import load_config_to_env
from rk_vision.identity_bank import IdentityBankConfig
from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline
from rk_vision.tracker import DeepSortTrackerConfig


CONFIG = Path(__file__).resolve().parents[2] / "car_control_modular/config/reid_runtime.ini"


def test_runtime_similar_follow_settings_reach_real_identity_bank(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    load_config_to_env(str(CONFIG))
    config = RKNNVisionConfig.from_env()
    # Constructors do not load models or open hardware. Test the complete
    # configuration route, including the real tracker/bank constructors.
    pipeline = RKNNVisionPipeline(config)
    for item, prefix in ((config, "identity_"), (pipeline.tracker.config, "identity_"),
                         (pipeline.tracker.identity_bank.config, "")):
        assert getattr(item, prefix + "similar_follow_enable") is True
        assert getattr(item, prefix + "similar_follow_entry_threshold") == .50
        assert getattr(item, prefix + "similar_follow_retain_threshold") == .55
        assert getattr(item, prefix + "similar_follow_max_gap_sec") == .50


def test_library_defaults_keep_new_policy_opt_in(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    assert not IdentityBankConfig().similar_follow_enable
    assert not DeepSortTrackerConfig().identity_similar_follow_enable
    assert not RKNNVisionConfig(yolo_model_path="unused").identity_similar_follow_enable
    assert not RKNNVisionConfig.from_env().identity_similar_follow_enable


@pytest.mark.parametrize("enabled", [False, True])
def test_environment_supports_explicit_policy_selection(monkeypatch, enabled):
    monkeypatch.setattr(os, "environ", {
        "Y8_IDENTITY_SIMILAR_FOLLOW_ENABLE": str(int(enabled)),
        "Y8_IDENTITY_SIMILAR_FOLLOW_ENTRY_THRESHOLD": ".44",
        "Y8_IDENTITY_SIMILAR_FOLLOW_RETAIN_THRESHOLD": ".49",
        "Y8_IDENTITY_SIMILAR_FOLLOW_MAX_GAP_SEC": ".40",
    })
    bank = RKNNVisionPipeline(RKNNVisionConfig.from_env()).tracker.identity_bank
    assert bank.config.similar_follow_enable is enabled
    assert bank.config.similar_follow_entry_threshold == .44
    assert bank.config.similar_follow_retain_threshold == .49
    assert bank.config.similar_follow_max_gap_sec == .40
