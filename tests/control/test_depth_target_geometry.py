"""Exact YOLO-to-confirmed-UID depth geometry, with no tracker or hardware."""
from __future__ import annotations

import copy
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.depth_target_geometry import resolve_depth_target_observation


# Actual CAP1015: tracker enlargement touched the bottom boundary even though
# the associated YOLO detection did not. Depth must use the latter unchanged.
DISPLAY = (242.54516241976836, 40.28864742651828, 415.25703503457976, 479.0)
DETECTOR = (255.99012756347656, 78.54582214355469, 389.80328369140625, 455.6407470703125)
OTHER = (20.0, 30.0, 100.0, 420.0)
CAPTURE = 1015
TIMESTAMP = 1234.56789


def observation():
    return {
        "raw_track_id": 11, "uid": 1,
        "detector_bbox": list(DETECTOR), "display_bbox": list(DISPLAY),
        "sample_metadata": {
            "capture_frame_id": CAPTURE, "capture_timestamp": TIMESTAMP,
            "is_fresh": True, "source_detection_index": 0,
        },
        "assignment": {
            "uid": 1, "reason": "mapped", "bbox_quality_ok": True,
            "bbox_quality_tier": "strong", "match_source": "strong",
        },
    }


def resolve(records=None, **changes):
    args = dict(target_id=1, display_bbox=DISPLAY, capture_frame_id=CAPTURE,
                capture_timestamp=TIMESTAMP, observations=[observation()] if records is None else records,
                width=640, height=480)
    args.update(changes)
    return resolve_depth_target_observation(**args)


def test_cap1015_uses_original_yolo_box_without_rewriting_display_or_uid():
    records = [observation()]
    original = copy.deepcopy(records)
    result = resolve(records)
    assert result is not None
    assert result.bbox == DETECTOR and result.bbox != DISPLAY
    assert result.target_id == 1 and result.raw_track_id == 11
    assert result.capture_frame_id == CAPTURE and result.capture_timestamp == TIMESTAMP
    assert result.source == "yolo_detector"
    assert records == original
    assert original[0]["display_bbox"][3] == 479.0 and result.bbox[3] < 456


def test_result_is_frozen_and_does_not_share_mutable_bbox_list():
    record = observation()
    result = resolve([record])
    record["detector_bbox"][0] = 0
    assert result.bbox == DETECTOR
    with pytest.raises(FrozenInstanceError):
        result.target_id = 2


def test_exact_display_match_can_ignore_an_unrelated_person():
    other = observation()
    other.update(raw_track_id=12, uid=2, display_bbox=OTHER, detector_bbox=OTHER)
    other["sample_metadata"]["source_detection_index"] = 1
    other["assignment"]["uid"] = 2
    assert resolve([other, observation()]).raw_track_id == 11


def test_no_nearest_box_or_high_score_reassignment_when_exact_geometry_missing():
    record = observation()
    record["display_bbox"][0] += 0.001
    record["score"] = 1.0
    assert resolve([record]) is None


def test_display_numeric_tolerance_only_handles_tiny_roundoff():
    record = observation()
    record["display_bbox"][0] += 0.0000001
    assert resolve([record]).bbox == DETECTOR


@pytest.mark.parametrize("overrides", [
    {"target_id": 0}, {"target_id": -1}, {"target_id": True}, {"target_id": 1.2},
    {"capture_frame_id": 0}, {"capture_frame_id": float("nan")},
    {"capture_timestamp": float("nan")}, {"capture_timestamp": float("inf")},
    {"capture_timestamp": 0.0}, {"width": 0}, {"height": -2}, {"width": 639.5},
    {"expected_raw_track_id": 12}, {"expected_raw_track_id": -2},
    {"display_bbox": None}, {"display_bbox": (0, 0, 640.1, 480)},
])
def test_invalid_query_or_expected_raw_track_rejected(overrides):
    assert resolve(**overrides) is None


@pytest.mark.parametrize("key,value", [
    ("is_fresh", False), ("is_fresh", 1), ("capture_frame_id", 1014),
    ("capture_timestamp", TIMESTAMP - .001), ("capture_timestamp", None),
    ("source_detection_index", -1), ("source_detection_index", None),
    ("source_detection_index", 1.2), ("search_observation_only", True),
])
def test_missing_stale_or_probe_capture_proof_rejected(key, value):
    record = observation()
    record["sample_metadata"][key] = value
    assert resolve([record]) is None


@pytest.mark.parametrize("key", ["assignment", "sample_metadata", "display_bbox", "detector_bbox", "uid", "raw_track_id"])
def test_required_observation_fields_cannot_be_inferred(key):
    record = observation()
    record.pop(key)
    assert resolve([record]) is None


@pytest.mark.parametrize("bbox", [
    (-1, 20, 100, 200), (10, -1, 100, 200), (10, 20, 641, 200),
    (10, 20, 100, 481), (10, 20, 10, 200), (10, 30, 100, 20),
    (10, 20, float("nan"), 200), (10, 20, 100, float("inf")), (10, 20, 100),
    "1234", {1: 0, 2: 0, 3: 0, 4: 0},
])
def test_invalid_detector_bbox_is_not_clamped_or_replaced_with_tracker_box(bbox):
    record = observation()
    record["detector_bbox"] = bbox
    assert resolve([record]) is None


@pytest.mark.parametrize("reason", [
    "controlled_handoff_wait", "preferred_search_reacquire_wait", "pending_new",
    "preferred_search_late_candidate_wait", "preferred_search_reacquire_geometry_reject",
    "mapped_verify_reject", "search_candidate_excluded", "mapped_weak_observed",
    "mapped_low_confidence_strong_observation", "weak_match_requires_handoff",
])
def test_positive_uid_cannot_override_pending_rejected_or_observation_assignment(reason):
    record = observation()
    record["assignment"]["reason"] = reason
    assert resolve([record], allow_confirmed_mapped=True) is None


@pytest.mark.parametrize("key,value", [
    ("uid", 2), ("uid", 0), ("mapped_uid", 2), ("bbox_quality_ok", False),
    ("bbox_quality_ok", None), ("bbox_quality_tier", "weak"),
    ("reacquire_geometry_ok", False), ("search_excluded", True),
    ("search_observation_only", True),
])
def test_conflicting_uid_quality_geometry_or_exclusion_rejected(key, value):
    record = observation()
    record["assignment"][key] = value
    assert resolve([record]) is None


def test_unbound_low_distance_best_uid_is_not_identity_confirmation():
    record = observation()
    record["uid"] = 0
    record["assignment"].update(uid=0, best_uid=1, distance=.01, reason="preferred_search_reacquire_wait")
    assert resolve([record], allow_confirmed_mapped=True) is None


@pytest.mark.parametrize("reason", ["controlled_handoff", "preferred_search_late_reacquire", "weak_preferred_reacquire_confirmed"])
def test_confirmed_mapped_requires_explicit_opt_in_and_completed_reason(reason):
    record = observation()
    record["uid"] = 0
    record["assignment"].update(uid=0, mapped_uid=1, reason=reason, reacquire_geometry_ok=True)
    assert resolve([record]) is None
    assert resolve([record], allow_confirmed_mapped=True).target_id == 1


def test_real_weak_confirmed_edge_path_remains_direction_only():
    record = observation()
    record["uid"] = 0
    record["assignment"].update(
        uid=0, mapped_uid=1, reason="weak_preferred_reacquire_confirmed",
        bbox_quality_ok=False, bbox_quality_tier="weak", bbox_quality_reason="edge_touch>2",
        partial_distance=.02, match_source="partial", reacquire_geometry_ok=True,
    )
    assert resolve([record], allow_confirmed_mapped=True) is None


def test_reserved_probe_with_confirmed_public_uid_can_supply_depth_before_deepsort():
    record = observation()
    record.update(raw_track_id=-1, display_bbox=DETECTOR)
    record["assignment"].update(reason="preferred_search_reacquire", reacquire_geometry_ok=True)
    result = resolve([record], display_bbox=DETECTOR, expected_raw_track_id=-1)
    assert result is not None and result.raw_track_id == -1 and result.target_id == 1


def test_real_search_probe_record_exports_resolvable_confirmed_geometry(monkeypatch):
    import numpy as np
    from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
    from rk_vision.yolo11 import Detection

    tracker = DeepSortTracker(DeepSortTrackerConfig())
    tracker._frame_context = {"capture_frame_id": CAPTURE, "capture_timestamp": TIMESTAMP}
    tracker._frame_index = 2
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction="left")

    def confirmed_assignment(**kwargs):
        assert kwargs["track_id"] == -1 and kwargs["bbox_quality_ok"] is True
        tracker.identity_bank.last_assignments[-1] = {
            "uid": 1, "reason": "preferred_search_reacquire",
            "bbox_quality_ok": True, "bbox_quality_tier": "strong",
            "reacquire_geometry_ok": True,
        }
        return 1

    monkeypatch.setattr(tracker.identity_bank, "assign", confirmed_assignment)
    record = tracker._search_probe_record(
        [Detection(DETECTOR, .9, 0)], [np.array([1., 0.], dtype=np.float32)],
        partial_features=[None], image_width=640, image_height=480,
    )
    assert record.track_id == -1 and record.reid_uid == 1
    result = resolve(tracker.last_identity_observations, display_bbox=DETECTOR, expected_raw_track_id=-1)
    assert result is not None and result.bbox == DETECTOR


@pytest.mark.parametrize("raw", [0, -2, -100])
def test_no_other_zero_or_negative_raw_id_is_accepted(raw):
    record = observation()
    record["raw_track_id"] = raw
    assert resolve([record]) is None


def test_unconfirmed_probe_mapping_is_not_promoted_even_with_mapped_opt_in():
    record = observation()
    record.update(raw_track_id=-1, uid=0)
    record["assignment"].update(uid=0, mapped_uid=1, reason="weak_preferred_reacquire_confirmed")
    assert resolve([record], allow_confirmed_mapped=True) is None


def test_duplicate_display_is_ambiguous_even_with_expected_raw_track():
    other = observation()
    other.update(raw_track_id=12, uid=2)
    other["assignment"]["uid"] = 2
    other["sample_metadata"]["source_detection_index"] = 1
    assert resolve([observation(), other], expected_raw_track_id=11) is None


@pytest.mark.parametrize("same_raw", [False, True])
def test_same_detection_or_raw_track_cannot_prove_two_different_records(same_raw):
    other = observation()
    other.update(display_bbox=OTHER, detector_bbox=OTHER)
    if same_raw:
        other["sample_metadata"]["source_detection_index"] = 1
    else:
        other["raw_track_id"] = 12
    assert resolve([observation(), other]) is None


def test_confirmed_uid_geometry_is_not_rejudged_by_reid_distance_or_wall_clock():
    record = observation()
    record["sample_metadata"]["capture_timestamp"] = 1.0
    record["assignment"].update(match_source=None, distance=None, reason="mapped")
    # Runtime owns freshness TTL and bank owns identity; no new ReID threshold.
    assert resolve([record], capture_timestamp=1.0).capture_timestamp == 1.0
