from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rk_vision.reid import _color_signature, OSNetConfig, OSNetRKNNExtractor
from rk_vision.stage_timing import StageTiming
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


def reference_color(crop):
    cv2 = pytest.importorskip("cv2")
    top = int(round(len(crop) * .15))
    bottom = max(top + 1, int(round(len(crop) * .85)))
    hsv = cv2.cvtColor(crop[top:bottom], cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[:, :, i].reshape(-1) for i in range(3))
    out = np.concatenate((
        np.histogram(h[s >= 24], bins=8, range=(0, 180))[0],
        np.histogram(s, bins=4, range=(0, 256))[0],
        np.histogram(v, bins=4, range=(0, 256))[0],
    )).astype("float32")
    return out / max(float(np.linalg.norm(out)), 1e-12)


@pytest.mark.parametrize("shape", [(4, 4, 3), (40, 20, 3), (479, 310, 3), (480, 640, 3)])
def test_color_descriptor_exactly_preserved(shape):
    crop = np.random.default_rng(12).integers(0, 256, shape, dtype=np.uint8)
    np.testing.assert_array_equal(_color_signature(crop), reference_color(crop))
    np.testing.assert_array_equal(_color_signature(crop[:, ::-1]), reference_color(crop[:, ::-1]))


@pytest.mark.parametrize("value", [0, 23, 24, 63, 64, 127, 128, 191, 192, 255])
def test_flat_gray_retains_no_hue_and_bin_edges(value):
    crop = np.full((40, 30, 3), value, dtype=np.uint8)
    np.testing.assert_array_equal(_color_signature(crop), reference_color(crop))


def test_all_hue_and_saturation_values_preserved():
    cv2 = pytest.importorskip("cv2")
    h, s = np.indices((180, 256))
    hsv = np.stack((h, s, np.full_like(h, 255)), axis=-1).astype(np.uint8)
    crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    np.testing.assert_array_equal(_color_signature(crop), reference_color(crop))


def test_timing_wall_and_cpu_are_separate(monkeypatch):
    import rk_vision.stage_timing as module
    wall = iter([1., 1.100, 1.150, 1.160])
    cpu = iter([2., 2.010, 2.030, 2.032])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(module.time, "thread_time", lambda: next(cpu))
    timer = StageTiming()
    timer.mark("association")
    timer.mark("records")
    result = timer.finish()
    assert result["association"] == pytest.approx(100)
    assert result["association_cpu"] == pytest.approx(10)
    assert result["records"] == pytest.approx(50)
    assert result["cpu"] == pytest.approx(32)
    assert result["noncpu"] == pytest.approx(128)


def test_tracker_timing_resets_on_empty_frame():
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1, min_confidence=.1))
    f = np.ones(512, dtype=np.float32)
    f /= np.linalg.norm(f)
    for _ in range(3):
        tracker.update([Detection((100, 50, 200, 400), .9, 0)], [f], image_width=640, image_height=480)
    assert tracker.last_timing_ms["competition"] >= 0
    assert tracker.last_timing_ms["records_cpu"] >= 0
    assert tracker.deepsort.tracker.last_timing_ms["kalman_update"] >= 0
    tracker.update([], [], image_width=640, image_height=480)
    assert "competition" not in tracker.last_timing_ms
    assert tracker.last_timing_ms["association"] >= 0


def test_reid_empty_clears_new_metrics():
    extractor = OSNetRKNNExtractor(OSNetConfig(model_path="", enabled=False))
    extractor.last_timing_ms.update(color=99., postprocess_exclusive=99.)
    extractor.extract(np.zeros((10, 10, 3), dtype=np.uint8), [])
    assert extractor.last_timing_ms["color"] == 0
    assert extractor.last_timing_ms["postprocess_exclusive"] == 0
