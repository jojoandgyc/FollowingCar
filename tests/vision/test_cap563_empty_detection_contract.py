"""Real pipeline/tracker with fake inference; no RKNN runtime or hardware.

An empty detector result may preserve an existing bounded proof, but is never
itself fresh appearance/identity evidence. These tests exercise the producer
contract through process_frame, including failures after detection succeeds.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


BOX = (230., 70., 350., 400.)
CAP558_TS = 38715.837819393
CAP563_TS = 38716.102835918


@pytest.fixture
def pipeline_case(monkeypatch):
    p = RKNNVisionPipeline.__new__(RKNNVisionPipeline)
    p.config = RKNNVisionConfig(
        yolo_model_path="unused", detector_continuation_enable=True,
        predicted_reid_verify_enable=False, identity_suppress_duplicate_uids=False,
    )
    p._frame_context = {}
    p._reid_diagnostics = None
    p.logger = None
    p.tracker = DeepSortTracker(DeepSortTrackerConfig(
        n_init=1, identity_new_confirm_frames=1,
    ))
    detector = SimpleNamespace(
        detections=[], last_search_diagnostic_detections=[],
        last_timing_ms={}, calls=0, failure=None,
    )
    def detect(*_):
        detector.calls += 1
        if detector.failure is not None:
            raise detector.failure
        return list(detector.detections)
    detector.detect = detect
    p.detector = detector
    extractor = SimpleNamespace(
        calls=0, missing=False, failure=None, wrong_length=False,
        last_timing_ms={}, last_partial_features=[],
        last_partial_feature_sources=[], last_color_features=[],
    )
    def extract(packet, persons, fmt):
        extractor.calls += 1
        if extractor.failure is not None:
            raise extractor.failure
        extractor.last_partial_features = [None for _ in persons]
        extractor.last_partial_feature_sources = [None for _ in persons]
        extractor.last_color_features = [None for _ in persons]
        if extractor.wrong_length:
            return []
        return [None if extractor.missing else np.array([1., 0., 0.]) for _ in persons]
    extractor.extract = extract
    p.reid = extractor
    p.set_detector_continuation_context(active_uid=1, allowed=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    clock = SimpleNamespace(now=CAP558_TS)
    monkeypatch.setattr("rk_vision.pipeline.time.monotonic", lambda: clock.now)
    def step(cap=563, stamp=CAP563_TS, *, probe=False, image=None):
        clock.now = stamp + .06
        p.set_frame_context(control_frame_id=cap, capture_frame_id=cap,
                            capture_timestamp=stamp)
        process = p.process_search_probe_frame if probe else p.process_frame
        return process(frame if image is None else image, "BGR")
    return SimpleNamespace(pipeline=p, detector=detector, extractor=extractor,
                           frame=frame, step=step)


def assert_binding(p, cap=563, stamp=CAP563_TS):
    state = p.last_identity_processing
    assert state["capture_frame_id"] == cap
    assert state["capture_timestamp"] == stamp
    assert state["detector_result_capture_frame_id"] == cap
    assert state["detector_result_capture_timestamp"] == stamp
    return state


def test_cap563_real_empty_pipeline_publishes_detection_not_identity(pipeline_case):
    s = pipeline_case
    assert s.step() == []
    state = assert_binding(s.pipeline)
    assert state["mode"] == "full"
    assert state["detector_result_complete"] is True
    assert type(state["detector_person_count"]) is int
    assert state["detector_person_count"] == 0
    assert state["full_features_current"] is False
    assert "identity_valid_until" not in state
    assert s.pipeline.tracker.last_identity_observations == []
    assert s.detector.calls == s.extractor.calls == 1


@pytest.mark.parametrize("missing", [False, True])
def test_person_without_target_or_features_is_not_empty(pipeline_case, missing):
    s = pipeline_case
    s.detector.detections = [Detection(BOX, .9, 0)]
    s.extractor.missing = missing
    s.step()
    state = assert_binding(s.pipeline)
    assert state["detector_result_complete"] is True
    assert state["detector_person_count"] == 1
    assert state["full_features_current"] is (not missing)


def test_size_rejected_person_cannot_masquerade_as_no_person(pipeline_case):
    s = pipeline_case
    s.pipeline.config = replace(s.pipeline.config, identity_min_width_px=60.)
    s.detector.detections = [Detection((10., 10., 30., 100.), .9, 0)]
    assert s.step() == []
    state = assert_binding(s.pipeline)
    assert state["detector_result_complete"] is True
    assert state["detector_person_count"] == 1
    assert state["full_features_current"] is False
    assert s.pipeline.last_follow_size_rejections


def test_low_score_person_diagnostic_is_not_certified_empty(pipeline_case):
    s = pipeline_case
    s.detector.last_search_diagnostic_detections = [Detection(BOX, .12, 0)]
    assert s.step() == []
    state = assert_binding(s.pipeline)
    assert state["detector_result_complete"] is True
    assert state["detector_person_count"] == 1
    assert state["full_features_current"] is False


def test_formal_person_is_not_double_counted_in_diagnostics(pipeline_case):
    s = pipeline_case
    formal = Detection(BOX, .9, 0)
    s.detector.detections = [formal]
    s.detector.last_search_diagnostic_detections = [formal, Detection(BOX, .12, 0)]
    s.step()
    assert assert_binding(s.pipeline)["detector_person_count"] == 2


def test_non_person_background_is_a_successful_empty_person_result(pipeline_case):
    s = pipeline_case
    s.detector.detections = [Detection(BOX, .9, 2)]
    assert s.step() == []
    state = assert_binding(s.pipeline)
    assert state["detector_result_complete"] is True
    assert state["detector_person_count"] == 0
    assert state["full_features_current"] is False


def test_background_person_not_matching_active_uid_is_not_empty(pipeline_case):
    s = pipeline_case
    s.pipeline.set_detector_continuation_context(active_uid=99, allowed=True)
    s.detector.detections = [Detection(BOX, .9, 0)]
    s.step(558, CAP558_TS)
    records = s.step()
    assert records and all(record.reid_uid != 99 for record in records)
    state = assert_binding(s.pipeline)
    assert state["detector_result_complete"] is True
    assert state["detector_person_count"] == 1


def test_cap558_then_cap563_empty_does_not_reuse_features_or_identity(pipeline_case):
    s = pipeline_case
    s.detector.detections = [Detection(BOX, .9, 0)]
    s.step(558, CAP558_TS)
    assert s.pipeline.last_identity_processing["full_features_current"] is True
    s.detector.detections = []
    s.step()
    state = assert_binding(s.pipeline)
    assert state["detector_result_complete"] is True
    assert state["detector_person_count"] == 0
    assert state["full_features_current"] is False
    assert "identity_valid_until" not in state


def test_next_full_capture_replaces_empty_count_and_binding(pipeline_case):
    s = pipeline_case
    s.step()
    old = dict(s.pipeline.last_identity_processing)
    s.detector.detections = [Detection(BOX, .9, 0)]
    s.step(565, CAP563_TS + .1)
    state = assert_binding(s.pipeline, 565, CAP563_TS + .1)
    assert state["detector_result_complete"] is True
    assert state["detector_person_count"] == 1
    assert state["full_features_current"] is True
    assert old["detector_person_count"] == 0


def test_probe_cannot_reuse_previous_completed_empty_capture(pipeline_case):
    s = pipeline_case
    s.step()
    assert s.step(565, CAP563_TS + .1, probe=True) == []
    state = assert_binding(s.pipeline, 565, CAP563_TS + .1)
    assert state["mode"] == "probe"
    assert state["detector_result_complete"] is False
    assert state["detector_person_count"] is None
    assert state["full_features_current"] is False
    assert s.extractor.calls == 1


@pytest.mark.parametrize("failure", ["frame", "detector", "reid", "tracker", "diagnostics"])
def test_any_incomplete_next_frame_clears_previous_empty_contract(
        pipeline_case, monkeypatch, failure):
    s = pipeline_case
    s.step()
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic pipeline failure")
    if failure == "frame":
        monkeypatch.setattr("rk_vision.pipeline.numpy_from_frame", fail)
    elif failure == "detector":
        s.detector.failure = RuntimeError("synthetic pipeline failure")
    elif failure == "reid":
        s.extractor.failure = RuntimeError("synthetic pipeline failure")
    elif failure == "tracker":
        monkeypatch.setattr(s.pipeline.tracker, "update", fail)
    else:
        monkeypatch.setattr(s.pipeline, "_record_reid_diagnostics", fail)
    with pytest.raises(RuntimeError, match="synthetic pipeline failure"):
        s.step(565, CAP563_TS + .1)
    state = assert_binding(s.pipeline, 565, CAP563_TS + .1)
    assert state["detector_result_complete"] is False
    assert state["detector_person_count"] is None
    assert state["full_features_current"] is False


def test_wrong_feature_count_fails_closed_after_successful_detection(pipeline_case):
    s = pipeline_case
    s.step()
    s.detector.detections = [Detection(BOX, .9, 0)]
    s.extractor.wrong_length = True
    with pytest.raises(ValueError, match="features length must match"):
        s.step(565, CAP563_TS + .1)
    state = assert_binding(s.pipeline, 565, CAP563_TS + .1)
    assert state["detector_result_complete"] is False
    assert state["detector_person_count"] is None
    assert state["full_features_current"] is False


def test_missing_capture_context_does_not_invent_provenance(pipeline_case):
    s = pipeline_case
    s.pipeline.process_frame(s.frame, "BGR")
    state = s.pipeline.last_identity_processing
    assert state["detector_result_complete"] is True
    assert state["detector_person_count"] == 0
    assert state["detector_result_capture_frame_id"] is None
    assert state["detector_result_capture_timestamp"] is None
    assert state["full_features_current"] is False
