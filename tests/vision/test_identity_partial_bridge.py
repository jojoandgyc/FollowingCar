from __future__ import annotations

import numpy as np

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig


def _unit(values):
    vector = np.asarray(values, dtype="float32")
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def test_partial_reacquire_bridges_probe_and_new_track():
    bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.20,
            weak_reacquire_threshold=0.38,
            weak_reacquire_confirm_frames=2,
            min_confidence=0.60,
            min_area=900,
            controlled_handoff_min_old_track_gap_frames=2,
            preferred_search_reacquire_threshold=0.20,
            preferred_search_reacquire_max_disadvantage=0.05,
            handoff_geometry_max_center_jump_ratio=0.25,
        )
    )
    feature = _unit([1.0, 0.0, 0.0])
    uid = bank.assign(
        track_id=1,
        feature=feature,
        confidence=0.95,
        area=4000,
        frame_index=1,
        sample_metadata={"center_x_ratio": 0.50, "area_ratio": 0.20},
    )

    outputs = []
    for track_id, frame_index, center in ((-1, 7, 0.60), (3, 8, 0.62)):
        outputs.append(
            bank.assign(
                track_id=track_id,
                feature=feature,
                confidence=0.90,
                area=4000,
                frame_index=frame_index,
                candidate_count=1,
                bbox_quality_ok=False,
                bbox_quality_reason="edge_touch>2,detector_crop:edge_touch>2",
                bbox_quality_tier="weak",
                sample_metadata={"center_x_ratio": center, "area_ratio": 0.20},
                preferred_uid=uid,
                preferred_candidate_ok=True,
            )
        )

    assert outputs == [0, 0]
    assert bank.track_to_uid == {3: uid}
    assert bank.last_assignments[3]["reason"] == "weak_preferred_reacquire_confirmed"


def test_edge_touch_partial_observation_bridges_into_late_confirmation():
    """A clipped raw detector box may start the same two-frame chain as a late box."""
    bank = IdentityBank(
        IdentityBankConfig(
            min_confidence=0.60,
            min_area=900,
            new_identity_confirm_frames=1,
            controlled_handoff_enable=True,
            controlled_handoff_min_old_track_gap_frames=1,
            preferred_search_reacquire_enable=True,
            preferred_search_reacquire_late_candidate_enable=True,
            preferred_search_reacquire_confirm_frames=2,
            preferred_search_reacquire_threshold=0.20,
            partial_appearance_enable=True,
            partial_match_threshold=0.34,
            handoff_geometry_max_gap_frames=15,
        )
    )
    full = _unit([1.0, 0.0, 0.0])
    torso = _unit([0.9, 0.435, 0.0])
    uid = bank.assign(
        track_id=1,
        feature=full,
        partial_feature=torso,
        confidence=0.95,
        area=40000,
        frame_index=1,
        sample_metadata={
            "bbox": [400, 80, 800, 440],
            "detector_bbox": [400, 80, 800, 440],
            "detector_center_x_ratio": 0.50,
            "detector_area_ratio": 0.23,
            "center_x_ratio": 0.50,
            "area_ratio": 0.23,
            "is_fresh": True,
        },
    )
    edge_meta = {
        "bbox": [300, 0, 850, 480],
        "detector_bbox": [300, 0, 850, 480],
        "detector_center_x_ratio": 0.449,
        "detector_area_ratio": 0.985,
        "center_x_ratio": 0.449,
        "area_ratio": 0.985,
        "partial_observation": True,
        "is_fresh": True,
    }
    first = bank.assign(
        track_id=7,
        feature=full,
        partial_feature=torso,
        confidence=0.90,
        area=40000,
        frame_index=20,
        bbox_quality_ok=False,
        bbox_quality_tier="reject",
        bbox_quality_reason="edge_touch>2",
        preferred_uid=uid,
        preferred_candidate_ok=True,
        sample_metadata={**edge_meta, "capture_timestamp": 10.0},
    )
    assert first == 0
    assert bank.last_assignments[7]["reason"] == "weak_preferred_reacquire_wait"
    assert bank.last_assignments[7]["match_source"] == "partial"

    second = bank.assign(
        track_id=7,
        feature=full,
        partial_feature=torso,
        confidence=0.92,
        area=40000,
        frame_index=21,
        bbox_quality_ok=True,
        bbox_quality_tier="strong",
        preferred_uid=uid,
        preferred_candidate_ok=True,
        sample_metadata={
            **edge_meta,
            "bbox": [305, 0, 855, 480],
            "detector_bbox": [305, 0, 855, 480],
            "detector_center_x_ratio": 0.453,
            "center_x_ratio": 0.453,
            "capture_timestamp": 10.1,
        },
    )
    assert second == uid
    assert bank.last_assignments[7]["reason"] == "preferred_search_late_reacquire"
    assert bank.last_assignments[7]["late_candidate_streak"] == 2


def test_edge_touch_partial_reacquire_still_rejects_competing_candidates():
    """Partial evidence must not bypass the unique-candidate gate."""
    bank = IdentityBank(
        IdentityBankConfig(
            min_confidence=0.60,
            new_identity_confirm_frames=1,
            preferred_search_reacquire_threshold=0.20,
            partial_appearance_enable=True,
            partial_match_threshold=0.34,
            preferred_search_reacquire_min_score_gap=0.25,
        )
    )
    feature = _unit([1.0, 0.0, 0.0])
    uid = bank.assign(track_id=1, feature=feature, partial_feature=feature,
                      confidence=0.95, area=4000, frame_index=1)
    result = bank.assign(
        track_id=2,
        feature=feature,
        partial_feature=feature,
        confidence=0.90,
        area=4000,
        frame_index=3,
        candidate_count=2,
        bbox_quality_ok=False,
        bbox_quality_tier="reject",
        bbox_quality_reason="edge_touch>2",
        preferred_uid=uid,
        preferred_candidate_ok=True,
        sample_metadata={
            "partial_observation": True,
            "center_x_ratio": 0.5,
            "area_ratio": 0.2,
            "candidate_score_gap": 0.05,
            "is_fresh": True,
        },
    )
    assert result == 0
    assert bank.last_assignments[2]["reason"] == "weak_bbox_unassigned"
    assert 2 not in bank.track_to_uid


def test_strong_full_match_survives_partial_aggregate_mismatch():
    """A clipped torso descriptor must not erase a unique strong full match."""
    bank = IdentityBank(
        IdentityBankConfig(
            min_confidence=0.60,
            min_area=900,
            new_identity_confirm_frames=1,
            controlled_handoff_enable=True,
            preferred_search_reacquire_enable=True,
            preferred_search_reacquire_late_candidate_enable=True,
            preferred_search_reacquire_confirm_frames=2,
            preferred_search_reacquire_threshold=0.20,
            partial_appearance_enable=True,
            partial_match_threshold=0.34,
        )
    )
    full = _unit([1.0, 0.0, 0.0])
    gallery_torso = _unit([1.0, 0.0, 0.0])
    query_torso = _unit([0.0, 1.0, 0.0])

    def metadata(frame, timestamp):
        bbox = [300.0, 40.0, 700.0, 440.0]
        return {
            "bbox": bbox,
            "detector_bbox": bbox,
            "center_x_ratio": 0.50,
            "detector_center_x_ratio": 0.50,
            "area_ratio": 0.52,
            "detector_area_ratio": 0.52,
            "partial_observation": True,
            "is_fresh": True,
            "capture_timestamp": timestamp,
            "search_reacquire_context_active": frame > 1,
        }

    uid = bank.assign(
        track_id=1,
        feature=full,
        partial_feature=gallery_torso,
        confidence=0.95,
        area=40000,
        frame_index=1,
        sample_metadata=metadata(1, 1.0),
    )
    first = bank.assign(
        track_id=7,
        feature=full,
        partial_feature=query_torso,
        confidence=0.90,
        area=40000,
        frame_index=20,
        preferred_uid=uid,
        preferred_candidate_ok=True,
        sample_metadata=metadata(20, 10.0),
    )
    assert first == 0
    assert bank.last_assignments[7]["reason"] == "preferred_search_late_candidate_wait"

    second = bank.assign(
        track_id=7,
        feature=full,
        partial_feature=query_torso,
        confidence=0.90,
        area=40000,
        frame_index=21,
        preferred_uid=uid,
        preferred_candidate_ok=True,
        sample_metadata=metadata(21, 10.1),
    )
    assert second == uid
    assignment = bank.last_assignments[7]
    assert assignment["reason"] == "preferred_search_late_reacquire"
    assert assignment["partial_aggregate_override"] is True
    assert assignment["partial_aggregate_distance"] > 0.34
    assert assignment["reacquire_geometry"]["partial_aggregate_rejection"] == "overridden_full_strong"


def test_soft_search_candidate_bridges_strict_reid_distance():
    """A near-threshold search match must observe twice before UID takeover."""
    bank = IdentityBank(
        IdentityBankConfig(
            min_confidence=0.60,
            min_area=900,
            new_identity_confirm_frames=1,
            controlled_handoff_enable=True,
            preferred_search_reacquire_enable=True,
            preferred_search_reacquire_late_candidate_enable=True,
            preferred_search_reacquire_confirm_frames=2,
            preferred_search_reacquire_threshold=0.20,
            preferred_search_soft_candidate_threshold=0.30,
        )
    )
    reference = _unit([1.0, 0.0, 0.0])
    query = _unit([0.76, 0.65, 0.0])  # cosine distance ~= 0.24

    def metadata(center, timestamp, *, count=1, gap=0.60):
        bbox = [100.0 + center * 100.0, 40.0, 500.0 + center * 100.0, 440.0]
        return {
            "bbox": bbox,
            "detector_bbox": bbox,
            "center_x_ratio": center,
            "detector_center_x_ratio": center,
            "area_ratio": 0.50,
            "detector_area_ratio": 0.50,
            "detector_confidence": 0.88,
            "candidate_score_gap": gap,
            "candidate_count": count,
            "is_fresh": True,
            "search_reacquire_context_active": True,
            "capture_timestamp": timestamp,
            "integrated_yaw_deg": 0.0,
        }

    uid = bank.assign(
        track_id=1,
        feature=reference,
        confidence=0.95,
        area=4000,
        frame_index=1,
        sample_metadata={**metadata(0.50, 10.0), "search_reacquire_context_active": False},
    )
    assert uid > 0

    first = bank.assign(
        track_id=7,
        feature=query,
        confidence=0.88,
        area=4000,
        frame_index=2,
        preferred_uid=uid,
        preferred_candidate_ok=True,
        sample_metadata=metadata(0.25, 10.1),
    )
    assert first == 0
    assert bank.last_assignments[7]["reason"] == "preferred_search_soft_candidate_wait"
    assert bank.last_assignments[7]["soft_search_observation"] is True

    second = bank.assign(
        track_id=7,
        feature=query,
        confidence=0.88,
        area=4000,
        frame_index=3,
        preferred_uid=uid,
        preferred_candidate_ok=True,
        sample_metadata=metadata(0.30, 10.2),
    )
    assert second == uid
    assert bank.last_assignments[7]["reason"] == "preferred_search_soft_reacquire"


def test_soft_search_ignores_small_fragment_competition():
    bank = IdentityBank(IdentityBankConfig())
    assert bank._soft_candidate_competition_ok(
        2,
        {
            "candidate_score_gap": 0.194,
            "detector_area_ratio": 0.50,
            "detector_confidence": 0.844,
        },
    )
    assert not bank._soft_candidate_competition_ok(
        2,
        {
            "candidate_score_gap": 0.194,
            "detector_area_ratio": 0.017,
            "detector_confidence": 0.844,
        },
    )


def test_search_geometry_jump_seeds_rolling_anchor_instead_of_binding():
    """Camera motion may invalidate the old anchor, but never grants a UID alone."""
    bank = IdentityBank(
        IdentityBankConfig(
            min_confidence=0.60,
            min_area=900,
            new_identity_confirm_frames=1,
            controlled_handoff_enable=True,
            preferred_search_reacquire_enable=True,
            preferred_search_reacquire_late_candidate_enable=True,
            preferred_search_reacquire_confirm_frames=2,
            preferred_search_reacquire_threshold=0.20,
            handoff_geometry_max_center_jump_ratio=0.25,
        )
    )
    feature = _unit([1.0, 0.0, 0.0])

    def metadata(bbox, timestamp, yaw):
        x1, y1, x2, y2 = bbox
        return {
            "bbox": list(bbox),
            "detector_bbox": list(bbox),
            "center_x_ratio": (x1 + x2) / 1280.0,
            "detector_center_x_ratio": (x1 + x2) / 1280.0,
            "area_ratio": (x2 - x1) * (y2 - y1) / (640.0 * 480.0),
            "detector_area_ratio": (x2 - x1) * (y2 - y1) / (640.0 * 480.0),
            "capture_timestamp": timestamp,
            "integrated_yaw_deg": yaw,
            "is_fresh": True,
            "search_reacquire_context_active": True,
        }

    uid = bank.assign(
        track_id=1,
        feature=feature,
        confidence=0.95,
        area=80000,
        frame_index=1,
        sample_metadata=metadata([500, 40, 700, 440], 1.0, 0.0),
    )
    first = bank.assign(
        track_id=8,
        feature=feature,
        confidence=0.90,
        area=80000,
        frame_index=5,
        preferred_uid=uid,
        preferred_candidate_ok=True,
        sample_metadata=metadata([100, 40, 300, 440], 2.0, 10.0),
    )
    assert first == 0
    assert bank.last_assignments[8]["reason"] == "preferred_search_late_candidate_wait"
    assert bank.last_assignments[8]["reacquire_geometry"]["late_anchor_mode"] == "seed"
    assert bank.last_assignments[8]["reacquire_geometry"]["old_anchor_ignored"]

    second = bank.assign(
        track_id=8,
        feature=feature,
        confidence=0.90,
        area=80000,
        frame_index=6,
        preferred_uid=uid,
        preferred_candidate_ok=True,
        sample_metadata=metadata([110, 40, 310, 440], 2.1, 11.0),
    )
    assert second == uid
    assert bank.last_assignments[8]["reason"] == "preferred_search_late_reacquire"
    assert bank.last_assignments[8]["reacquire_geometry"]["yaw_compensated_center_jump_ratio"] < 0.25


def test_search_local_anchor_expires_after_a_frame_gap():
    """An abandoned candidate chain cannot be revived by a later isolated box."""
    bank = IdentityBank(
        IdentityBankConfig(
            min_confidence=0.60,
            min_area=900,
            new_identity_confirm_frames=1,
            controlled_handoff_enable=True,
            preferred_search_reacquire_late_candidate_enable=True,
            preferred_search_reacquire_confirm_frames=2,
        )
    )
    feature = _unit([1.0, 0.0, 0.0])
    base = {
        "bbox": [300, 40, 500, 440],
        "detector_bbox": [300, 40, 500, 440],
        "center_x_ratio": 0.3125,
        "detector_center_x_ratio": 0.3125,
        "area_ratio": 0.2604,
        "detector_area_ratio": 0.2604,
        "is_fresh": True,
        "search_reacquire_context_active": True,
    }
    uid = bank.assign(track_id=1, feature=feature, confidence=0.9, area=80000, frame_index=1,
                      sample_metadata={**base, "capture_timestamp": 1.0})
    bank.assign(track_id=2, feature=feature, confidence=0.9, area=80000, frame_index=5,
                preferred_uid=uid, preferred_candidate_ok=True,
                sample_metadata={**base, "capture_timestamp": 2.0})
    isolated = bank.assign(track_id=2, feature=feature, confidence=0.9, area=80000, frame_index=7,
                           preferred_uid=uid, preferred_candidate_ok=True,
                           sample_metadata={**base, "capture_timestamp": 2.2})
    assert isolated == 0
    assert bank.last_assignments[2]["reason"] == "preferred_search_late_candidate_wait"
    assert bank.last_assignments[2]["late_candidate_streak"] == 1
