from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


def _unit(values):
    vector = np.asarray(values, dtype="float32")
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _feature_at_distance(distance: float):
    cosine = 1.0 - float(distance)
    return _unit([cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)), 0.0])


def _metadata(frame_index: int):
    return {
        "track_id": 7,
        "frame_index": frame_index,
        "control_frame_id": frame_index,
        "capture_frame_id": frame_index,
        "capture_timestamp": frame_index * 0.033,
        "bbox": [250.0, 40.0, 450.0, 440.0],
        "detector_bbox": [250.0, 40.0, 450.0, 440.0],
        "center_x_ratio": 0.547,
        "detector_center_x_ratio": 0.547,
        "area_ratio": 0.26,
        "detector_area_ratio": 0.26,
        "is_fresh": True,
        "search_reacquire_context_active": True,
        "search_direction_compatible": True,
    }


def _make_bank():
    return IdentityBank(
        IdentityBankConfig(
            min_confidence=0.60,
            preferred_search_reacquire_min_confidence=0.50,
            preferred_search_reacquire_threshold=0.20,
            preferred_search_reacquire_confirm_frames=2,
            controlled_handoff_enable=True,
            controlled_handoff_min_old_track_gap_frames=1,
            handoff_geometry_max_gap_frames=15,
        )
    )


def test_search_only_confidence_floor_allows_two_frame_confirmation_without_gallery_update():
    bank = _make_bank()
    anchor = _unit([1.0, 0.0, 0.0])
    uid = bank.assign(
        track_id=1,
        feature=anchor,
        confidence=0.90,
        area=80000,
        frame_index=1,
        bbox_quality_ok=True,
        bbox_quality_tier="strong",
        sample_metadata={
            **_metadata(1),
            "track_id": 1,
            "capture_timestamp": 0.033,
        },
    )
    query = _feature_at_distance(0.166)

    first = bank.assign(
        track_id=7,
        feature=query,
        confidence=0.573,
        area=80000,
        frame_index=2,
        preferred_uid=uid,
        preferred_candidate_ok=True,
        bbox_quality_ok=True,
        bbox_quality_tier="strong",
        sample_metadata=_metadata(2),
    )
    assignment = bank.last_assignments[7]
    assert first == 0
    assert assignment["reason"] == "preferred_search_reacquire_wait"
    assert assignment["search_quality_override"] is True
    assert assignment["quality_confidence_floor"] == 0.50
    assert assignment["preferred_search_low_confidence"] is True
    assert len(bank.identities[uid].features) == 1

    second = bank.assign(
        track_id=7,
        feature=query,
        confidence=0.573,
        area=80000,
        frame_index=3,
        preferred_uid=uid,
        preferred_candidate_ok=True,
        bbox_quality_ok=True,
        bbox_quality_tier="strong",
        sample_metadata=_metadata(3),
    )
    assert second == uid
    assert len(bank.identities[uid].features) == 1


def test_search_confidence_floor_does_not_change_normal_assignment_or_lower_bound():
    bank = _make_bank()
    anchor = _unit([1.0, 0.0, 0.0])
    uid = bank.assign(track_id=1, feature=anchor, confidence=0.90, area=80000, frame_index=1)

    normal = bank.assign(
        track_id=2,
        feature=anchor,
        confidence=0.573,
        area=80000,
        frame_index=2,
    )
    assert normal == 0
    assert bank.last_assignments[2]["reason"] == "low_quality"

    too_low = bank.assign(
        track_id=3,
        feature=anchor,
        confidence=0.49,
        area=80000,
        frame_index=3,
        preferred_uid=uid,
        preferred_candidate_ok=True,
        bbox_quality_ok=True,
        bbox_quality_tier="strong",
        sample_metadata=_metadata(3),
    )
    assert too_low == 0
    assert bank.last_assignments[3]["reason"] == "low_quality"


def test_detector_only_search_probe_reaches_identity_bank_at_search_floor():
    tracker = DeepSortTracker(
        DeepSortTrackerConfig(
            identity_min_confidence=0.60,
            identity_preferred_search_reacquire_min_confidence=0.50,
            identity_preferred_search_reacquire_threshold=0.20,
            identity_preferred_search_reacquire_confirm_frames=2,
            identity_min_area=900.0,
            identity_min_width_px=16.0,
            identity_min_height_px=28.0,
        )
    )
    anchor = _unit([1.0, 0.0, 0.0])
    uid = tracker.identity_bank.assign(
        track_id=1,
        feature=anchor,
        confidence=0.90,
        area=80000,
        frame_index=1,
    )
    tracker.set_search_reacquire_context(active_uid=uid, searching=True, direction="left")
    tracker._frame_index = 2
    record = tracker._search_probe_record(
        [Detection((260.0, 80.0, 380.0, 400.0), 0.573, 0)],
        [anchor],
        partial_features=[None],
        image_width=640,
        image_height=480,
    )
    assert record is not None
    assignment = tracker.identity_bank.last_assignments[tracker._search_probe_track_id]
    assert assignment["search_quality_override"] is True
    assert assignment["quality_confidence_floor"] == 0.50
    assert assignment["preferred_search_low_confidence"] is True


def test_large_low_score_search_probe_is_observation_only():
    """A person-sized low-score box reaches bounded ReID observation only."""
    tracker = DeepSortTracker(
        DeepSortTrackerConfig(
            identity_min_confidence=0.60,
            identity_preferred_search_reacquire_min_confidence=0.50,
            identity_preferred_search_reacquire_observation_min_confidence=0.25,
            identity_preferred_search_reacquire_threshold=0.20,
            identity_preferred_search_reacquire_confirm_frames=2,
            identity_min_area=900.0,
            identity_min_width_px=16.0,
            identity_min_height_px=28.0,
        )
    )
    anchor = _unit([1.0, 0.0, 0.0])
    uid = tracker.identity_bank.assign(
        track_id=1,
        feature=anchor,
        confidence=0.90,
        area=80000,
        frame_index=1,
    )
    tracker.set_search_reacquire_context(active_uid=uid, searching=True, direction="left")
    tracker._frame_index = 2
    record = tracker._search_probe_record(
        [Detection((40.0, 10.0, 520.0, 450.0), 0.252, 0)],
        [anchor],
        partial_features=[None],
        image_width=640,
        image_height=480,
    )
    assert record is not None
    assert record.reid_uid == 0
    assignment = tracker.identity_bank.last_assignments[tracker._search_probe_track_id]
    assert assignment["search_observation_override"] is True
    assert assignment["quality_confidence_floor"] == 0.25
    assert assignment["preferred_search_low_confidence"] is True
    assert assignment["reacquire_distance_limit"] == 0.30
    assert len(tracker.identity_bank.identities[uid].features) == 1
    assert assignment["reason"] in {
        "weak_preferred_reacquire_wait",
        "weak_handoff_geometry_reject",
        "preferred_search_late_candidate_wait",
        "preferred_search_reacquire_wait",
    }
