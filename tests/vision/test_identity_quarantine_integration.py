"""Exercise template quarantine through the real IdentityBank.assign paths."""

from copy import deepcopy
import math

import numpy as np
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig


def feature(distance=0.0):
    cosine = 1.0 - distance
    return np.asarray([cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)), 0.0], dtype="float32")


def metadata(capture_id, timestamp, *, center=0.408, area_ratio=0.033, fresh=True, search=False):
    box_width = 100.0
    box_height = area_ratio * 640.0 * 480.0 / box_width
    bbox = [center * 640.0 - box_width / 2.0, 160.0,
            center * 640.0 + box_width / 2.0, 160.0 + box_height]
    return {
        "capture_frame_id": capture_id, "capture_timestamp": timestamp,
        "source_detection_index": 0, "is_fresh": fresh,
        "bbox": bbox, "detector_bbox": list(bbox),
        "center_x_ratio": center, "detector_center_x_ratio": center,
        "area_ratio": area_ratio, "detector_area_ratio": area_ratio,
        "candidate_count": 1, "candidate_score_gap": 0.90,
        "detector_confidence": 0.90, "quality_bbox_ok": True,
        "bbox_quality_tier": "strong", "edge_touch_count": 0,
        "detector_edge_touch_count": 0, "integrated_yaw_deg": 0.0,
        "partial_observation": False,
        "search_reacquire_context_active": search,
        "search_direction_compatible": True if search else None,
    }


def make_bank(**overrides):
    config = dict(
        new_identity_confirm_frames=1, update_interval=1,
        update_threshold=0.30, min_confidence=0.65, min_area=0.0,
        diversity_min_distance=0.01, max_features=20,
        controlled_handoff_enable=True,
        controlled_handoff_instant_threshold=0.15,
        controlled_handoff_min_old_track_gap_frames=2,
        preferred_search_reacquire_instant_threshold=0.15,
        partial_appearance_enable=True,
    )
    config.update(overrides)
    bank = IdentityBank(IdentityBankConfig(**config))
    uid = assign(bank, 1, 1, 1565, 9.8, distance=0.0, partial_distance=0.0)
    assert uid == 1
    return bank


def assign(
    bank, track_id, frame, capture_id, timestamp, *, distance=0.10,
    partial_distance=0.08, quality="strong", confidence=0.90,
    missing_feature=False, fresh=True, search=False, center=0.408,
    area_ratio=0.033,
):
    evidence = metadata(capture_id, timestamp, center=center, area_ratio=area_ratio,
                        fresh=fresh, search=search)
    evidence["bbox_quality_tier"] = quality
    evidence["quality_bbox_ok"] = quality == "strong"
    return bank.assign(
        track_id=track_id, feature=None if missing_feature else feature(distance),
        partial_feature=feature(partial_distance), confidence=confidence,
        area=area_ratio * 640 * 480, frame_index=frame, candidate_count=1,
        bbox_quality_ok=quality == "strong", bbox_quality_tier=quality,
        bbox_quality_reason="edge_touch>2" if quality == "weak" else "",
        sample_metadata=evidence,
        preferred_uid=1 if search else None, preferred_candidate_ok=search,
    )


def gallery_snapshot(bank):
    entry = bank.identities[1]
    return (
        tuple(tuple(float(v) for v in item) for item in entry.features),
        tuple(tuple(float(v) for v in item) for item in entry.weak_features),
        tuple(tuple(float(v) for v in item) for item in entry.partial_features),
        deepcopy(entry.feature_metadata), deepcopy(entry.weak_feature_metadata),
        deepcopy(entry.partial_feature_metadata), entry.last_frame, entry.last_weak_frame,
        entry.update_count, entry.weak_update_count,
    )


def reacquired_bank():
    bank = make_bank()
    original = gallery_snapshot(bank)
    assert assign(bank, 35, 3, 1571, 10.0) == 1
    assert bank._reacquire_quarantine.is_held(1)
    assert gallery_snapshot(bank) == original
    return bank, original


def test_initial_identity_and_same_track_update_are_not_quarantined():
    bank = make_bank()
    assert not bank._reacquire_quarantine.is_held(1)
    initial = gallery_snapshot(bank)
    assert assign(bank, 1, 2, 1567, 9.9) == 1
    assert bank.last_assignments[1]["bank_updated"]
    assert gallery_snapshot(bank) != initial
    assert not bank._reacquire_quarantine.is_held(1)


@pytest.mark.parametrize("path", ["normal", "controlled", "preferred", "late", "weak"])
def test_each_cross_track_reacquire_keeps_uid_but_freezes_all_galleries(path):
    bank = make_bank(controlled_handoff_enable=path != "normal")
    original = gallery_snapshot(bank)
    if path == "late":
        assert assign(bank, 35, 3, 1571, 10.4, search=True) == 0
        assert assign(bank, 35, 4, 1573, 10.5, search=True) == 1
        assert bank.last_assignments[35]["reason"] == "preferred_search_late_reacquire"
        next_frame, next_capture, next_time = 5, 1575, 10.6
    elif path == "weak":
        assert assign(bank, 35, 3, 1571, 10.0, search=True, quality="weak") == 0
        assert assign(bank, 35, 4, 1573, 10.1, search=True, quality="weak") == 0
        assert bank.last_assignments[35]["reason"] == "weak_preferred_reacquire_confirmed"
        assert bank.track_to_uid[35] == 1
        next_frame, next_capture, next_time = 5, 1575, 10.2
    else:
        assert assign(bank, 35, 3, 1571, 10.0, search=path == "preferred") == 1
        next_frame, next_capture, next_time = 4, 1573, 10.1
    assert bank._reacquire_quarantine.is_held(1)
    assert not bank.last_assignments[35]["bank_updated"]
    assert gallery_snapshot(bank) == original
    assert assign(bank, 35, next_frame, next_capture, next_time) == 1
    assert not bank.last_assignments[35]["bank_updated"]
    assert gallery_snapshot(bank) == original


def test_release_requires_rgb_duration_and_three_stable_frames_then_allows_update():
    bank, original = reacquired_bank()
    for frame, timestamp in ((4, 10.2), (5, 10.4), (6, 10.6), (7, 10.8)):
        assert assign(bank, 35, frame, 1571 + 2 * (frame - 3), timestamp) == 1
        assert bank._reacquire_quarantine.is_held(1)
        assert gallery_snapshot(bank) == original
    assert assign(bank, 35, 8, 1581, 11.0) == 1
    assert not bank._reacquire_quarantine.is_held(1)
    assert bank.last_assignments[35]["bank_updated"]
    assert gallery_snapshot(bank) != original


@pytest.mark.parametrize("bad_observation", [
    {"quality": "weak"}, {"confidence": 0.40},
    {"missing_feature": True}, {"fresh": False},
])
def test_weak_low_quality_missing_feature_or_stale_breaks_release_chain(bad_observation):
    bank, original = reacquired_bank()
    assign(bank, 35, 4, 1573, 10.8)
    assign(bank, 35, 5, 1575, 11.0)
    assign(bank, 35, 6, 1577, 11.1, **bad_observation)
    assert bank._reacquire_quarantine.is_held(1)
    assert gallery_snapshot(bank) == original
    for frame, timestamp in ((7, 11.2), (8, 11.3)):
        assert assign(bank, 35, frame, 1571 + 2 * (frame - 3), timestamp) == 1
        assert bank._reacquire_quarantine.is_held(1)
        assert gallery_snapshot(bank) == original
    assert assign(bank, 35, 9, 1583, 11.4) == 1
    assert not bank._reacquire_quarantine.is_held(1)


def test_repeated_capture_cannot_release_or_refresh_gallery():
    bank, original = reacquired_bank()
    assign(bank, 35, 4, 1573, 11.0)
    for frame in (5, 6, 7):
        assert assign(bank, 35, frame, 1573, 11.0) == 1
        assert bank._reacquire_quarantine.is_held(1)
        assert gallery_snapshot(bank) == original
    assign(bank, 35, 8, 1575, 11.1)
    assert bank._reacquire_quarantine.is_held(1)
    assign(bank, 35, 9, 1577, 11.2)
    assert not bank._reacquire_quarantine.is_held(1)


@pytest.mark.parametrize("jump", [{"center": 0.65}, {"area_ratio": 0.014}])
def test_local_geometry_jump_breaks_quarantine_proof_without_changing_uid(jump):
    bank, original = reacquired_bank()
    assign(bank, 35, 4, 1573, 10.8)
    assign(bank, 35, 5, 1575, 11.0)
    # Both changes remain within the bank's ordinary mapped geometry limits,
    # but exceed the stricter quarantine limits (.20 center / .50 area).
    assert assign(bank, 35, 6, 1577, 11.1, **jump) == 1
    assert bank._reacquire_quarantine.is_held(1)
    assert gallery_snapshot(bank) == original
    for frame, timestamp in ((7, 11.2), (8, 11.3)):
        assert assign(bank, 35, frame, 1571 + 2 * (frame - 3), timestamp) == 1
        assert bank._reacquire_quarantine.is_held(1)
        assert gallery_snapshot(bank) == original
    assert assign(bank, 35, 9, 1583, 11.4) == 1
    assert not bank._reacquire_quarantine.is_held(1)


def test_frozen_strong_distance_cannot_be_bootstrapped_by_new_track_templates():
    bank, original = reacquired_bank()
    for frame in range(4, 12):
        assert assign(bank, 35, frame, 1571 + 2 * (frame - 3), 10.0 + 0.2 * (frame - 3), distance=0.25) == 1
        assert bank._reacquire_quarantine.is_held(1)
        assert gallery_snapshot(bank) == original


def test_bank_reset_discards_quarantine_for_removed_identities():
    bank, _ = reacquired_bank()
    bank.reset()
    assert not bank.identities and not bank.track_to_uid
    assert not bank._reacquire_quarantine.is_held(1)
    assert assign(bank, 1, 1, 1, 20.0, distance=0.0) == 1
    assert not bank._reacquire_quarantine.is_held(1)


@pytest.mark.parametrize("rejection", ["mapped_verify", "identity_swap"])
def test_third_strong_sample_cannot_release_if_identity_checks_reject_it(rejection):
    bank = make_bank(mapped_verify_threshold=0.05 if rejection == "mapped_verify" else 0.40)
    original = gallery_snapshot(bank)
    assert assign(bank, 35, 3, 1571, 10.0) == 1
    # Both samples pass mapped verification but must remain quarantined.
    assert assign(bank, 35, 4, 1573, 10.8, distance=0.01) == 1
    assert assign(bank, 35, 5, 1575, 11.0, distance=0.01) == 1
    if rejection == "mapped_verify":
        # .10 satisfies quarantine's <=.20 frozen-gallery gate, but fails
        # this bank's stricter mapped verifier. Verification must win first.
        assert assign(bank, 35, 6, 1577, 11.1, distance=0.10) == 0
        assert bank.last_assignments[35]["reason"] == "mapped_verify_reject"
    else:
        # An explicit identity-swap rejection must dominate even if a direct
        # caller still labels appearance and crop quality as strong.
        assert bank.assign(
            track_id=35, feature=feature(0.10), confidence=0.90, area=10137.6,
            frame_index=6, bbox_quality_ok=True, bbox_quality_tier="strong",
            bbox_quality_reason="identity_swap_competing_track",
            sample_metadata=metadata(1577, 11.1),
        ) == 0
        assert bank.last_assignments[35]["reason"] == "identity_center_jump_reject"
    assert bank._reacquire_quarantine.is_held(1)
    assert gallery_snapshot(bank) == original


def test_exclusion_from_uid1_does_not_block_unmatched_person_creating_uid2():
    bank = make_bank()
    frame_observations = []
    for track_id, mapped_uid, center, distance in ((1, 1, 0.408, 0.0), (35, 0, 0.8, 1.0)):
        evidence = metadata(1567, 9.9, center=center)
        frame_observations.append({
            "raw_track_id": track_id, "mapped_uid": mapped_uid,
            "feature": feature(distance), "confidence": 0.90,
            "is_fresh": True, "duplicate": False, "identity_swap": False,
            "detector_bbox": evidence["detector_bbox"],
            "capture_frame_id": 1567, "capture_timestamp": 9.9,
        })
    bank.observe_frame_evidence(frame_index=2, observations=frame_observations, width=640, height=480)
    assert bank.search_exclusion_for(35, 1, frame_index=2, capture_timestamp=9.9) is not None
    assert assign(bank, 35, 2, 1567, 9.9, distance=1.0, partial_distance=1.0, center=0.8) == 2
    assert bank.last_assignments[35]["reason"] == "created"
    assert not bank._reacquire_quarantine.is_held(2)
