from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection
from car_control_modular.search_candidate_gate import (
    CandidateObservation, SearchCandidateGate, SearchCandidateGateConfig,
)


CAP_BOXES = (
    (309.4425, 188.4312, 347.4674, 237.8712),  # CAP822
    (273.1334, 189.1476, 312.6663, 240.9538),  # CAP825
    (208.0122, 187.9015, 249.1055, 243.3184),  # CAP833
    (203.6025, 187.4735, 245.7405, 244.2168),  # CAP834
)


def pipeline(probes=()):
    pipe = RKNNVisionPipeline.__new__(RKNNVisionPipeline)
    pipe.config = RKNNVisionConfig(
        yolo_model_path="unused.rknn",
        identity_min_area=3072, identity_min_width_px=32,
        identity_min_height_px=80,
    )
    pipe.detector = SimpleNamespace(last_search_diagnostic_detections=list(probes))
    pipe.logger = Mock()
    pipe._frame_context = {"capture_frame_id": 834}
    return pipe


@pytest.mark.parametrize("bbox", CAP_BOXES)
@pytest.mark.parametrize("score", [0.60, 0.99])
def test_cap_small_boxes_never_reach_identity_or_search(bbox, score):
    det = Detection(bbox, score, 0)
    pipe = pipeline()
    assert pipe._record_detector_output([det]) == []
    assert pipe.last_search_candidate_evidence.formal_persons == ()
    assert pipe.last_detections == [det]  # still visible to recording/diagnosis
    assert len(pipe.last_follow_size_rejections) == 1
    rejection = pipe.last_follow_size_rejections[0]
    assert rejection["height_px"] < 80
    assert "height<80" in rejection["reason"]
    assert pipe.logger.info.call_args.args[1] == 834


def test_small_probe_does_not_compete_with_valid_probe_or_request_hold():
    small = Detection(CAP_BOXES[-1], 0.20, 0)
    good = Detection((400, 100, 500, 350), 0.19, 0)
    pipe = pipeline([small, good])
    pipe._record_detector_output([])
    assert pipe.last_search_candidate_evidence.probe_persons == (good,)
    assert pipe.last_search_diagnostic_detections == [small, good]
    pipe.detector.last_search_diagnostic_detections = [small]
    pipe._record_detector_output([])
    assert pipe.last_search_candidate_evidence.probe_persons == ()
    assert len(pipe.last_follow_size_rejections) == 1  # per-frame, not cumulative


@pytest.mark.parametrize("bbox,allowed", [
    ((0, 0, 32, 96), True),       # width/area exact boundaries
    ((0, 0, 38.4, 80), True),     # height/area exact boundaries
    ((0, 0, 31, 200), False),     # narrow fragment
    ((0, 0, 100, 79), False),     # short fragment despite sufficient area
    ((0, 0, 32, 80), False),      # area still too small
    ((0, 0, 400, 479), True),     # don't add a new close/edge rejection
])
def test_size_boundary(bbox, allowed):
    pipe = pipeline()
    det = Detection(bbox, 0.9, 0)
    assert bool(pipe._record_detector_output([det])) is allowed


def test_other_classes_and_small_boxes_stay_in_raw_diagnostics_only():
    small = Detection(CAP_BOXES[-1], 0.95, 0)
    plant = Detection((100, 20, 300, 450), 0.99, 58)
    good = Detection((400, 100, 500, 350), 0.9, 0)
    pipe = pipeline()
    assert pipe._record_detector_output([small, plant, good]) == [good]
    assert pipe.last_detections == [small, plant, good]


def test_expanded_track_cannot_override_small_raw_crop_even_with_perfect_feature():
    tracker = DeepSortTracker(DeepSortTrackerConfig(
        identity_min_area=3072, identity_min_width_px=32,
        identity_min_height_px=80, identity_min_confidence=0.6,
    ))
    feature = np.ones(512, dtype=np.float32) / np.sqrt(512)
    uid = tracker.identity_bank.assign(
        track_id=14, feature=feature, confidence=0.95, area=40000,
        frame_index=1, bbox_quality_ok=True, bbox_quality_tier="strong",
    )
    assert uid > 0
    tracker._current_detections = (Detection(CAP_BOXES[-1], 0.99, 0),)
    tracker._frame_index = 2
    output = SimpleNamespace(
        track_id=14, reid_uid=uid, x1=150., y1=100., x2=350., y2=400.,
        class_id=0, confidence=0.99, feature=feature,
        state=2, time_since_update=0, source_detection_index=0,
    )
    rec = tracker._to_record(output, 640, 480, 1, partial_features=[feature])
    assert rec.reid_uid == 0
    assignment = tracker.identity_bank.last_assignments[14]
    assert assignment["reason"] == "mapped_bbox_quality_reject"
    assert not assignment.get("bank_updated", False)


def test_filtered_search_evidence_cannot_start_observation():
    pipe = pipeline()
    pipe._record_detector_output([Detection(CAP_BOXES[-1], 0.99, 0)])
    gate = SearchCandidateGate(SearchCandidateGateConfig(min_area_ratio=0.0))
    candidates = tuple(CandidateObservation(d.bbox, d.score)
                       for d in pipe.last_search_candidate_evidence.formal_persons)
    assert gate._valid_candidates(candidates, width=640, height=480, min_score=.1) == ()
    assert not gate.hold_active


def test_full_pipeline_repeated_small_frames_cannot_create_identity_or_track():
    pipe = pipeline()
    pipe.detector.detect = Mock(return_value=[Detection(CAP_BOXES[-1], .99, 0)])
    pipe.detector.last_timing_ms = {}
    pipe.reid = SimpleNamespace(extract=Mock(return_value=[]), last_timing_ms={})
    pipe.tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1, max_output_age=0))
    pipe._record_reid_diagnostics = Mock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        assert pipe.process_frame(frame, "BGR") == []
        assert pipe.reid.extract.call_args.args[1] == []
        assert pipe.tracker.identity_bank.track_to_uid == {}
        assert pipe.last_search_candidate_evidence.formal_persons == ()
