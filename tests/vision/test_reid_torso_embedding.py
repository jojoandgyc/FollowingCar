from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rk_vision.reid import OSNetConfig, OSNetRKNNExtractor
from rk_vision.yolo11 import Detection


class _Session:
    def __init__(self):
        self.calls = []

    def inference(self, inputs):
        tensor = np.asarray(inputs[0])
        self.calls.append(tensor.copy())
        # A fixed 512-D output matches the OSNet model used on the board.
        value = float(len(self.calls))
        return [np.full((1, 512), value, dtype="float32")]


def _frame():
    y, x = np.indices((120, 200))
    return np.stack((x, y, x + y), axis=-1).astype("uint8")


def test_extract_runs_separate_learned_torso_pass():
    extractor = OSNetRKNNExtractor(
        OSNetConfig(
            model_path="unused.rknn",
            enabled=True,
            input_width=32,
            input_height=64,
            color_fusion_enable=False,
            partial_appearance_enable=True,
            partial_osnet_enable=True,
        )
    )
    session = _Session()
    extractor.session = session
    detections = [
        Detection((10.0, 10.0, 80.0, 110.0), 0.9, 0),
        Detection((100.0, 20.0, 180.0, 115.0), 0.9, 0),
    ]

    features = extractor.extract(_frame(), detections)

    assert len(features) == 2
    assert len(session.calls) == 4
    assert all(call.shape == (1, 3, 64, 32) for call in session.calls)
    assert extractor.last_partial_feature_sources == ["osnet_torso", "osnet_torso"]
    assert all(feature.shape == (512,) for feature in extractor.last_partial_features)
    assert extractor.last_timing_ms["partial_inference"] >= 0.0


def test_extract_can_disable_torso_pass_for_latency_budget():
    extractor = OSNetRKNNExtractor(
        OSNetConfig(
            model_path="unused.rknn",
            enabled=True,
            input_width=32,
            input_height=64,
            color_fusion_enable=False,
            partial_appearance_enable=True,
            partial_osnet_enable=True,
        )
    )
    session = _Session()
    extractor.session = session
    detections = [Detection((10.0, 10.0, 80.0, 110.0), 0.9, 0)]

    features = extractor.extract(_frame(), detections, compute_partial=False)

    assert len(features) == 1
    assert len(session.calls) == 1
    assert extractor.last_partial_features == [None]
    assert extractor.last_partial_feature_sources == [None]
    assert extractor.last_timing_ms["partial_inference"] == 0.0
