from copy import deepcopy

import pytest

from car_control_modular.low_quality_lateral import MultiPersonLateralGate


def observations(capture_id=235, timestamp=10.0, *, target_track=1, active_uid=1):
    result = []
    for index, track_id in enumerate((target_track, 2)):
        result.append({
            "raw_track_id": track_id,
            "uid": 0,
            "detector_bbox": [1.5, 2.0, 247.9, 474.0] if index == 0 else [400, 50, 510, 470],
            "sample_metadata": {
                "is_fresh": True,
                "capture_frame_id": capture_id,
                "capture_timestamp": timestamp,
                "candidate_count": 2,
                "source_detection_index": index,
                "detector_confidence": 0.942 if index == 0 else 0.870,
                "candidate_score_gap": 0.072 if index == 0 else -0.072,
            },
            "assignment": {
                "uid": 0,
                "mapped_uid": active_uid,
                "reason": "mapped_weak_observed",
                "bbox_quality_ok": False,
                "bbox_quality_tier": "weak",
                "bbox_quality_reason": "edge_touch>2",
                "strong_distance": 0.079,
                "match_source": "strong",
            } if index == 0 else {
                "uid": 0,
                "best_uid": active_uid,
                "distance": 0.370,
                "match_source": "strong",
            },
        })
    return result


def update(gate, capture_id=235, timestamp=10.0, *, items=None, active_uid=1):
    return gate.update(
        active_uid=active_uid, capture_frame_id=capture_id, capture_timestamp=timestamp,
        width=640, height=480,
        observations=observations(capture_id, timestamp) if items is None else items,
    )


def test_same_mapped_crop_wins_by_identity_after_two_new_frames():
    gate = MultiPersonLateralGate()
    source = observations()
    original = deepcopy(source)
    first = update(gate, items=source)
    assert first.track_id is None and first.streak == 1
    assert source == original  # No UID/gallery/observation mutation.
    second = update(gate, 238, 10.10)
    assert second.track_id == 1 and second.streak == 2
    assert second.distance == 0.079 and second.runner_up_distance == 0.370


@pytest.mark.parametrize("distance", [0.10, 0.15, None, float("nan"), float("inf")])
def test_unknown_or_similar_competitor_resets_proof(distance):
    gate = MultiPersonLateralGate()
    update(gate)
    items = observations(238, 10.1)
    items[1]["assignment"]["distance"] = distance
    rejected = update(gate, 238, 10.1, items=items)
    assert rejected.track_id is None and rejected.streak == 0
    assert update(gate, 241, 10.2).streak == 1


def test_competitor_distance_must_refer_to_active_uid_and_strong_gallery():
    for assignment in (
        {"best_uid": 8, "distance": 0.7, "match_source": "strong"},
        {"best_uid": 1, "distance": 0.7, "match_source": "partial"},
        {"best_uid": 1, "distance": 0.7},
    ):
        items = observations()
        items[1]["assignment"] = assignment
        assert update(MultiPersonLateralGate(), items=items).reason == "unknown_competitor_distance"
    gate = MultiPersonLateralGate()
    for capture, timestamp in ((235, 10.0), (238, 10.1)):
        items = observations(capture, timestamp)
        items[1]["assignment"] = {"match_evidence": {"matched_uid": 1, "strong_distance": 0.370}}
        result = update(gate, capture, timestamp, items=items)
    assert result.track_id == 1


@pytest.mark.parametrize("match_source", [None, "weak", "partial"])
def test_target_requires_strong_winner_even_when_strong_distance_is_small(match_source):
    items = observations()
    items[0]["assignment"]["match_source"] = match_source
    result = update(MultiPersonLateralGate(), items=items)
    assert result.reason == "weak_identity_evidence" and result.track_id is None


@pytest.mark.parametrize("reason", [
    "area<900", "edge_touch>2,area_shrink<0.30", "aspect>1.3",
    "duplicate_person_box", "edge_touch>2,identity_center_jump>0.30",
    "edge_touch>2,identity_swap_competing_track",
    "detector_crop:edge_touch>2,detector_crop:area<900",
])
def test_fragment_and_hard_rejection_cannot_gain_lateral_permission(reason):
    items = observations()
    items[0]["assignment"]["bbox_quality_reason"] = reason
    assert update(MultiPersonLateralGate(), items=items).reason == "unsafe_crop_quality"


def test_prefixed_edge_reason_is_accepted_but_near_full_camera_is_not():
    items = observations()
    items[0]["assignment"]["bbox_quality_reason"] = "detector_crop:edge_touch>2"
    assert update(MultiPersonLateralGate(), items=items).streak == 1
    items[0]["detector_bbox"] = [1, 2, 639, 478]
    assert update(MultiPersonLateralGate(), items=items).reason == "near_camera_occlusion"


@pytest.mark.parametrize("active_uid", [None, 0, -1, float("nan"), float("inf"), "invalid"])
def test_invalid_active_identity_fails_closed(active_uid):
    assert update(MultiPersonLateralGate(), active_uid=active_uid).reason == "no_active_uid"


def test_missing_detection_evidence_or_duplicate_source_is_rejected():
    for change in ("missing", "duplicate", "uncovered", "stale", "wrong_capture"):
        items = observations()
        if change == "missing":
            items[1].pop("sample_metadata")
        elif change == "duplicate":
            items[1]["sample_metadata"]["source_detection_index"] = 0
        elif change == "uncovered":
            for item in items:
                item["sample_metadata"]["candidate_count"] = 3
        elif change == "stale":
            items[1]["sample_metadata"]["is_fresh"] = False
        else:
            items[1]["sample_metadata"]["capture_frame_id"] = 234
        result = update(MultiPersonLateralGate(), items=items)
        assert result.track_id is None and result.streak == 0, change


def test_multiple_mappings_and_conflicting_output_uid_are_rejected():
    items = observations()
    items[1]["assignment"]["mapped_uid"] = 1
    assert update(MultiPersonLateralGate(), items=items).reason == "ambiguous_mapping"
    items = observations()
    items[0]["uid"] = 9
    assert update(MultiPersonLateralGate(), items=items).reason == "conflicting_output_uid"


def test_track_change_requires_two_new_observations_and_repeat_cannot_confirm():
    gate = MultiPersonLateralGate()
    assert update(gate).streak == 1
    repeated = update(gate)
    assert repeated.reason == "duplicate_capture" and repeated.track_id is None and repeated.streak == 1
    items = observations(238, 10.1, target_track=3)
    changed = update(gate, 238, 10.1, items=items)
    assert changed.track_id is None and changed.streak == 1
    confirmed = update(gate, 241, 10.2, items=observations(241, 10.2, target_track=3))
    assert confirmed.track_id == 3
    gate.reset()
    assert update(gate).streak == 1


@pytest.mark.parametrize("bbox", [[350, 2, 595, 474], [1, 2, 50, 200]])
def test_center_or_area_discontinuity_resets_confirmation(bbox):
    gate = MultiPersonLateralGate()
    update(gate)
    items = observations(238, 10.1)
    items[0]["detector_bbox"] = bbox
    result = update(gate, 238, 10.1, items=items)
    assert result.reason == "geometry_discontinuity" and result.streak == 0


def test_time_gap_and_reordered_capture_reset_confirmation():
    gate = MultiPersonLateralGate()
    update(gate)
    assert update(gate, 238, 10.5).reason == "capture_gap"
    assert update(gate, 241, 10.6).streak == 1
    assert update(gate, 239, 10.55).reason == "out_of_order_capture"
    assert update(gate, 244, 10.7).streak == 1
