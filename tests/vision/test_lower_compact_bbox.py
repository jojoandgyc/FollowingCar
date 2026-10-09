"""CAP1945/1953/1958: reject control crops, preserve raw diagnostics."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rk_vision.follow_bbox_policy import lower_compact_bbox_reason
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection
from test_follow_bbox_size import pipeline


BOXES = [
    (0.757622, 272.268372, 134.583893, 407.026123),
    (77.487564, 211.638092, 244.356659, 405.905579),
    (147.913696, 213.667023, 308.106750, 404.364166),
]


@pytest.mark.parametrize("bbox", BOXES)
@pytest.mark.parametrize("scale", [1., 2.])
def test_logged_crops_rejected_at_both_resolutions(bbox, scale):
    assert lower_compact_bbox_reason(tuple(v*scale for v in bbox), 640*scale, 480*scale) == "lower_compact_bbox"


@pytest.mark.parametrize("bbox", [
    (0, 0, 400, 479),             # large near-camera partial person
    (0, 200, 100, 450),           # tall left-edge person
    (530, 200, 630, 450),         # tall right-edge person
    (300, 250, 355, 400),         # small but narrow standing person
    (100, 100, 270, 295),         # same dimensions, upper image
    (100, 300, 270, 479),         # bottom-clipped body
])
def test_not_a_blanket_small_or_edge_rejection(bbox):
    assert lower_compact_bbox_reason(bbox, 640, 480) == ""


@pytest.mark.parametrize("bbox", BOXES)
@pytest.mark.parametrize("probe", [False, True])
def test_formal_and_probe_never_reach_search_observation(bbox, probe):
    det = Detection(bbox, .2 if probe else .99, 0)
    pipe = pipeline([det] if probe else [])
    pipe.last_frame_width, pipe.last_frame_height = 640, 480
    assert pipe._record_detector_output([] if probe else [det]) == []
    assert pipe.last_search_candidate_evidence.formal_persons == ()
    assert pipe.last_search_candidate_evidence.probe_persons == ()
    assert pipe.last_follow_size_rejections[0]["reason"] == "lower_compact_bbox"
    assert det in (pipe.last_search_diagnostic_detections if probe else pipe.last_detections)


@pytest.mark.parametrize("bbox", BOXES)
def test_raw_crop_reject_not_rescued_by_expanded_track_or_perfect_reid(bbox):
    tracker = DeepSortTracker(DeepSortTrackerConfig(
        identity_min_area=3072, identity_min_width_px=32,
        identity_min_height_px=80, identity_min_confidence=.6))
    feature = np.ones(512, dtype=np.float32)/np.sqrt(512)
    uid = tracker.identity_bank.assign(track_id=12, feature=feature, confidence=.99,
        area=40000, frame_index=1, bbox_quality_ok=True, bbox_quality_tier="strong")
    tracker._current_detections = (Detection(bbox, .99, 0),)
    tracker._frame_index = 2
    output = SimpleNamespace(track_id=12, reid_uid=uid, x1=0., y1=0., x2=350., y2=479.,
        class_id=0, confidence=.99, feature=feature, state=2, time_since_update=0,
        source_detection_index=0)
    rec = tracker._to_record(output, 640, 480, 1, partial_features=[feature])
    assert rec.reid_uid == 0
    assert not tracker.identity_bank.last_assignments[12].get("bank_updated", False)


def test_full_pipeline_no_reid_or_track_created_for_repeated_crops():
    pipe = pipeline()
    pipe.detector.detect = Mock(return_value=[Detection(BOXES[1], .99, 0)])
    pipe.detector.last_timing_ms = {}
    pipe.reid = SimpleNamespace(extract=Mock(return_value=[]), last_timing_ms={})
    pipe.tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1, max_output_age=0))
    pipe._record_reid_diagnostics = Mock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        assert pipe.process_frame(frame, "BGR") == []
        assert pipe.reid.extract.call_args.args[1] == []
        assert pipe.tracker.identity_bank.track_to_uid == {}
