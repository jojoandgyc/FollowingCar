import time

import numpy as np

from rk_vision import direction_worker
from rk_vision.direction_worker import DirectionInferencePool
from rk_vision.yolo11 import Detection, YOLO11Config


class _FakeDetector:
    def __init__(self, config):
        self.config = config
        self.last_search_diagnostic_detections = []

    def set_search_diagnostic_active(self, active):
        self.active = active

    def detect(self, frame, frame_format="BGR"):
        marker = int(frame[0, 0, 0])
        if marker == 1:
            self.last_search_diagnostic_detections = [
                Detection((0.0, 0.0, 20.0, 20.0), 0.8, 0)
            ]
        elif marker == 2:
            self.last_search_diagnostic_detections = [
                Detection((80.0, 0.0, 100.0, 20.0), 0.8, 0)
            ]
        elif marker == 3:
            # Formal detection is on the left; the permissive diagnostic pass
            # reports a higher-scoring background blob on the right.
            self.last_search_diagnostic_detections = [
                Detection((80.0, 0.0, 100.0, 20.0), 0.95, 0)
            ]
            return [Detection((0.0, 0.0, 20.0, 20.0), 0.30, 0)]
        else:
            self.last_search_diagnostic_detections = []
        return list(self.last_search_diagnostic_detections)

    def release(self):
        pass


def _wait_for_results(pool, count):
    deadline = time.monotonic() + 2.0
    results = []
    while time.monotonic() < deadline and len(results) < count:
        results.extend(pool.drain_results())
        if len(results) < count:
            time.sleep(0.005)
    return results


def test_direction_pool_keeps_capture_ids_and_classifies_states(monkeypatch):
    monkeypatch.setattr(direction_worker, "YOLO11RKNNDetector", _FakeDetector)
    config = YOLO11Config(model_path="unused", backend="mock")
    pool = DirectionInferencePool(config, workers=2, queue_size=0)
    try:
        frames = [
            np.full((20, 100, 3), 1, dtype=np.uint8),
            np.zeros((20, 100, 3), dtype=np.uint8),
            np.full((20, 100, 3), 2, dtype=np.uint8),
        ]
        for capture_id, frame in enumerate(frames, start=10):
            assert pool.submit(capture_id, float(capture_id), frame)

        results = _wait_for_results(pool, len(frames))
        assert [item.capture_frame_id for item in results] == [10, 11, 12]
        assert [(item.state, item.side) for item in results] == [
            ("visible", "left"),
            ("missing", "none"),
            ("visible", "right"),
        ]
    finally:
        pool.close()


def test_direction_pool_prefers_formal_detection_over_diagnostic_blob(monkeypatch):
    monkeypatch.setattr(direction_worker, "YOLO11RKNNDetector", _FakeDetector)
    config = YOLO11Config(model_path="unused", backend="mock")
    pool = DirectionInferencePool(config, workers=1, queue_size=0)
    try:
        frame = np.full((20, 100, 3), 3, dtype=np.uint8)
        assert pool.submit(20, 20.0, frame)
        result = _wait_for_results(pool, 1)[0]
        assert result.state == "visible"
        assert result.side == "left"
        assert result.reason == "detector_formal_person_side"
    finally:
        pool.close()
