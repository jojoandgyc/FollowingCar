import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from rk_vision.frames import numpy_from_frame
from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline
from rk_vision.reid import OSNetConfig, OSNetRKNNExtractor
from rk_vision.yolo11 import Detection


class _Detector:
    def __init__(self, batches):
        self.batches = batches
        self.calls = 0
        self.last_search_diagnostic_detections = []
        self.last_timing_ms = {}
        self.released = False

    def detect(self, _frame, _frame_format):
        result = list(self.batches[self.calls])
        self.calls += 1
        return result

    def release(self):
        self.released = True


class _ReID:
    def __init__(self, feature_for_detection):
        self.feature_for_detection = feature_for_detection
        self.cropper = OSNetRKNNExtractor(OSNetConfig(model_path="", enabled=False))
        self.calls = []
        self.last_timing_ms = {}
        self.released = False

    def extract(self, frame, detections, frame_format):
        pixels, _, _, _ = numpy_from_frame(frame, frame_format)
        batch_index = len(self.calls)
        batch = []
        features = []
        for detection in detections:
            feature = self.feature_for_detection(batch_index, detection)
            crop = self.cropper._crop(pixels, detection.bbox)
            batch.append({
                "bbox": tuple(detection.bbox),
                "feature": feature,
                "pixels": None if crop is None else crop.copy(),
            })
            features.append(feature)
        self.calls.append(batch)
        return features

    def release(self):
        self.released = True


def _pipeline(directory, batches, feature_for_detection, **options):
    config = {
        "yolo_model_path": "unused-yolo.rknn",
        "reid_model_path": "unused-osnet.rknn",
        "backend": "mock",
        "accreditation_threshold": 2,
        "max_output_age": 2,
        "identity_new_confirm_frames": 1,
        "identity_update_interval": 1,
        "deepsort_bbox_expand_scale": 1.2,
        "deepsort_min_confidence": 0.5,
        "deepsort_nms_max_overlap": 0.5,
        "reid_diagnostics_enable": True,
        "reid_diagnostics_dir": str(directory),
        "reid_diagnostics_mapped_interval": 1,
    }
    config.update(options)
    pipeline = RKNNVisionPipeline(RKNNVisionConfig(**config))
    pipeline.detector.release()
    pipeline.reid.release()
    pipeline.detector = _Detector(batches)
    pipeline.reid = _ReID(feature_for_detection)
    return pipeline


def _frame(index):
    y, x = np.indices((120, 200))
    return np.stack((x + 17 * index, 2 * y + 31 * index, x + y + 13 * index), axis=-1).astype("uint8")


def _process(pipeline, index, *, control_id=None, capture_id=None):
    pipeline.set_frame_context(
        control_frame_id=100 + index if control_id is None else control_id,
        capture_frame_id=400 + index if capture_id is None else capture_id,
        capture_timestamp=10.0 + 0.1 * index,
    )
    return pipeline.process_frame(_frame(index))


def _records(directory):
    return [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]


def test_detector_crop_provenance_survives_filtering_nms_and_reordering(tmp_path):
    first_left = Detection((20.4, 18.4, 55.6, 100.6), 0.88, 0)
    first_right = Detection((126.4, 16.4, 162.6, 104.6), 0.96, 0)
    tracker_low = Detection((80.4, 20.4, 110.6, 90.6), 0.40, 0)
    pipeline_low = Detection((80.4, 20.4, 110.6, 90.6), 0.10, 0)
    other_class = Detection((80.4, 20.4, 110.6, 90.6), 0.99, 2)
    first_duplicate = Detection((21.4, 19.4, 54.6, 99.6), 0.70, 0)
    second_left = Detection((22.4, 18.4, 57.6, 100.6), 0.97, 0)
    second_right = Detection((125.4, 16.4, 161.6, 104.6), 0.82, 0)
    second_duplicate = Detection((23.4, 19.4, 56.6, 99.6), 0.70, 0)
    batches = [
        [other_class, first_left, tracker_low, first_duplicate, pipeline_low, first_right],
        [other_class, second_duplicate, second_right, tracker_low, pipeline_low, second_left],
    ]

    def feature_for_detection(_index, detection):
        if detection.score == 0.70:
            return np.array([0, 0, 1, 0], dtype="float32")
        if detection.score == 0.40:
            return np.array([0, 0, 0, 1], dtype="float32")
        return np.array([1, 0, 0, 0] if detection.bbox[0] < 80 else [0, 1, 0, 0], dtype="float32")

    pipeline = _pipeline(tmp_path, batches, feature_for_detection)
    writer = pipeline._reid_diagnostics
    try:
        assert _process(pipeline, 1) == []
        assert writer.accepted_samples == 0
        tracks = _process(pipeline, 2)
        assert len(tracks) == 2
        observations = pipeline.tracker.last_identity_observations
        assert [(item["raw_track_id"], item["sample_metadata"]["source_detection_index"])
                for item in observations] == [(1, 1), (2, 3)]
        assert pipeline.reid.calls[1][1]["bbox"] == second_right.bbox
        assert pipeline.reid.calls[1][3]["bbox"] == second_left.bbox
        assert len(pipeline.reid.calls[0]) == len(pipeline.reid.calls[1]) == 4
        assert len(pipeline.tracker.deepsort.tracker.tracks) == 2
        for observation in observations:
            assert observation["assignment"]["reason"] == "created"
            source = observation["sample_metadata"]["source_detection_index"]
            extracted = pipeline.reid.calls[1][source]
            entry = pipeline.tracker.identity_bank.identities[observation["uid"]]
            assert np.allclose(entry.features[0], extracted["feature"])
            assert tuple(entry.feature_metadata[0]["detector_bbox"]) == extracted["bbox"]
            assert not np.allclose(observation["display_bbox"], extracted["bbox"])
    finally:
        pipeline.close()

    records = _records(tmp_path)
    assert len(records) == 2
    assert writer.failed_samples == writer.dropped_samples == 0
    for record in records:
        source = record["sample_metadata"]["source_detection_index"]
        extracted = pipeline.reid.calls[1][source]
        assert record["raw_detector_bbox"] == list(extracted["bbox"])
        assert record["detector_bbox"] == list(extracted["bbox"])
        assert record["display_bbox"] != record["raw_detector_bbox"]
        assert np.array_equal(cv2.imread(str(tmp_path / record["sample_path"])), extracted["pixels"])
        assert record["control_frame_id"] == record["sample_metadata"]["control_frame_id"] == 102
        assert record["capture_frame_id"] == record["sample_metadata"]["capture_frame_id"] == 402
        assert record["frame_index"] == 2
        assert record["capture_timestamp"] == pytest.approx(10.2)
        assert record["assignment"]["match_evidence"] is None
        assert record["matched_template_sample_path"] is None
        assert record["anchor_sample_path"] is None


def test_gallery_snapshot_paths_and_prediction_do_not_reuse_old_crop(tmp_path):
    batches = [[Detection((50.4 + offset, 20.4, 90.6 + offset, 100.6), 0.95, 0)]
               for offset in (0, 1, 3, 6)] + [[]]
    angles = (0.0, 0.0, 0.35, 0.38, 0.38)

    def feature_for_detection(index, _detection):
        return np.array([math.cos(angles[index]), math.sin(angles[index]), 0, 0], dtype="float32")

    pipeline = _pipeline(tmp_path, batches, feature_for_detection)
    writer = pipeline._reid_diagnostics
    control_ids = [701, 805, 910, 1012, 1133]
    capture_ids = [1501, 1608, 1809, 1944, 1999]
    try:
        for index in range(4):
            _process(pipeline, index + 1, control_id=control_ids[index], capture_id=capture_ids[index])
        entry = pipeline.tracker.identity_bank.identities[1]
        assert len(entry.features) == 2
        assert [item["control_frame_id"] for item in entry.feature_metadata] == [805, 910]
        assert [item["capture_frame_id"] for item in entry.feature_metadata] == [1608, 1809]
        assert writer.accepted_samples == 3
        update_count = entry.update_count
        predicted = _process(pipeline, 5, control_id=control_ids[4], capture_id=capture_ids[4])
        assert len(predicted) == 1 and predicted[0].time_since_update == 1
        assert pipeline.tracker.last_identity_observations == []
        assert writer.accepted_samples == 3
        assert entry.update_count == update_count
        assert pipeline.reid.calls[-1] == []
    finally:
        pipeline.close()

    created, updated, mapped = _records(tmp_path)
    assert created["assignment"]["reason"] == "created"
    assert created["assignment"]["match_evidence"] is None
    assert updated["assignment"]["reason"] == "updated_diverse"
    assert updated["assignment"]["bank_updated"] is True
    assert len(updated["assignment"]["match_evidence"]["nearest_samples"]) == 1
    assert updated["matched_template_sample_path"] == created["sample_path"]
    assert updated["anchor_sample_path"] == created["sample_path"]
    assert mapped["assignment"]["reason"] == "skip_update_redundant"
    assert mapped["assignment"]["bank_updated"] is False
    evidence = mapped["assignment"]["match_evidence"]
    assert evidence["winner"]["index"] == 1
    assert evidence["winner"]["metadata"]["control_frame_id"] == 910
    assert evidence["anchor_metadata"]["control_frame_id"] == 805
    assert evidence["winner"]["weighted_distance"] < evidence["anchor_distance"]
    assert mapped["matched_template_sample_path"] == updated["sample_path"]
    assert mapped["anchor_sample_path"] == created["sample_path"]
    for record, batch_index in zip((created, updated, mapped), (1, 2, 3)):
        assert record["control_frame_id"] == control_ids[batch_index]
        assert record["capture_frame_id"] == capture_ids[batch_index]
        assert record["frame_index"] == batch_index + 1
        assert np.array_equal(
            cv2.imread(str(tmp_path / record["sample_path"])),
            pipeline.reid.calls[batch_index][0]["pixels"],
        )
    assert writer._thread is not None and not writer._thread.is_alive()
    assert pipeline.detector.released and pipeline.reid.released
    assert len(list(tmp_path.glob("*.png"))) == 3


def test_ordinary_mapping_is_sampled_but_gallery_updates_are_saved(tmp_path):
    batches = [[Detection((50.4, 20.4, 90.6, 100.6), 0.95, 0)] for _ in range(5)]
    angles = [0, 0, 0.35, 0.35, 0.35]

    def feature_for_detection(index, _detection):
        return np.array([math.cos(angles[index]), math.sin(angles[index]), 0, 0], dtype="float32")

    pipeline = _pipeline(tmp_path, batches, feature_for_detection, reid_diagnostics_mapped_interval=5)
    try:
        for index in range(1, 6):
            _process(pipeline, index)
    finally:
        pipeline.close()
    records = _records(tmp_path)
    assert [record["frame_index"] for record in records] == [2, 3, 5]
    assert [record["assignment"]["reason"] for record in records] == [
        "created", "updated_diverse", "skip_update_redundant",
    ]
    assert records[-1]["matched_template_sample_path"] == records[1]["sample_path"]


@pytest.mark.parametrize("limit,capacity,interval,expected", [(37, 3, 7, (37, 3, 7)), (-1, 0, 0, (0, 1, 1))])
def test_diagnostic_environment_config_reaches_writer(tmp_path, monkeypatch, limit, capacity, interval, expected):
    log_dir = tmp_path / "session logs"
    for key, value in {
        "RKNN_REID_DIAGNOSTICS_ENABLE": "1",
        "RKNN_REID_DIAGNOSTICS_MAX_SAMPLES": str(limit),
        "RKNN_REID_DIAGNOSTICS_QUEUE_CAPACITY": str(capacity),
        "RKNN_REID_DIAGNOSTICS_MAPPED_INTERVAL": str(interval),
        "RKNN_BACKEND": "mock",
        "FOLLOW_LOG_DIR": str(log_dir),
    }.items():
        monkeypatch.setenv(key, value)
    config = RKNNVisionConfig.from_env()
    assert config.reid_diagnostics_enable
    assert Path(config.reid_diagnostics_dir) == log_dir / "reid_diagnostics"
    assert (config.reid_diagnostics_max_samples, config.reid_diagnostics_queue_capacity,
            config.reid_diagnostics_mapped_interval) == expected
    pipeline = RKNNVisionPipeline(config)
    try:
        assert pipeline._reid_diagnostics.max_samples == expected[0]
        assert pipeline._reid_diagnostics._queue.maxsize == expected[1]
    finally:
        pipeline.close()
    assert not log_dir.exists()


def test_disabled_diagnostics_environment_does_not_create_writer(tmp_path, monkeypatch):
    monkeypatch.setenv("RKNN_REID_DIAGNOSTICS_ENABLE", "0")
    monkeypatch.setenv("FOLLOW_LOG_DIR", str(tmp_path / "disabled"))
    monkeypatch.setenv("RKNN_BACKEND", "mock")
    pipeline = RKNNVisionPipeline(RKNNVisionConfig.from_env())
    try:
        assert pipeline._reid_diagnostics is None
    finally:
        pipeline.close()
    assert not (tmp_path / "disabled").exists()
