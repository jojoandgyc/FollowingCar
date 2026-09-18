"""CAP334 -> CAP338 wrong raw-ID inheritance -> CAP341/346 true person.

Detector geometry/distances are fixed from the run; embeddings are synthetic
vectors at those cosine distances. No camera, model or motor is opened.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


ANCHOR = (204.404846, 34.199982, 407.408173, 475.147217)
WRONG = (5.026791, 192.736877, 82.208817, 344.089722)
TRUE_341 = (191.986740, 22.176178, 398.440857, 476.339905)
TRUE_346 = (171.394180, 9.686890, 386.263245, 476.742554)


def feature(distance=0.0):
    c = 1.0 - distance
    return np.array([c, np.sqrt(1.0 - c*c), 0.0], dtype=np.float32)


def metadata(bbox, cap=334, timestamp=10.0, yaw=-47.252393):
    x1, y1, x2, y2 = bbox
    return dict(
        bbox=bbox, detector_bbox=bbox, capture_frame_id=cap,
        capture_timestamp=timestamp, integrated_yaw_deg=yaw, is_fresh=True,
        detector_center_x_ratio=(x1+x2)/1280.0,
        detector_area_ratio=(x2-x1)*(y2-y1)/(640*480),
    )


def bank_with_anchor():
    bank = IdentityBank(IdentityBankConfig(
        new_identity_confirm_frames=1, update_interval=1,
        controlled_handoff_enable=True,
    ))
    assert bank.assign(track_id=1, feature=feature(), confidence=.946, area=89553,
                       frame_index=114, sample_metadata=metadata(ANCHOR)) == 1
    return bank


def assign(bank, track=1, frame=115, bbox=WRONG, distance=.225934, **kwargs):
    meta = metadata(bbox, 338+(frame-115)*3, 10.230411+(frame-115)*.165, -47.69142)
    meta.update(kwargs.pop("metadata_extra", {}))
    return bank.assign(
        track_id=track, feature=feature(distance), confidence=.946,
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1]), frame_index=frame,
        sample_metadata=meta, **kwargs,
    )


def observation(track, bbox, distance, mapped=0, cap=341):
    return dict(raw_track_id=track, mapped_uid=mapped, detector_bbox=bbox,
                confidence=.946, feature=feature(distance), is_fresh=True,
                capture_frame_id=cap, capture_timestamp=10.4,
                integrated_yaw_deg=-47.69142)


def test_cap338_same_id_strong_crop_cannot_keep_uid_or_train_gallery():
    bank = bank_with_anchor()
    reference = dict(bank.identities[1].last_strong_observation)
    assert assign(bank) == 0
    diag = bank.last_assignments[1]
    assert diag["reason"] == "mapped_geometry_reject"
    assert diag["reacquire_geometry_reason"] == "center_jump,area_change"
    assert not diag["bank_updated"]
    assert 1 not in bank.track_to_uid
    assert 1 not in bank.track_last_seen_frame
    assert bank.identities[1].last_strong_observation == reference
    assert len(bank.identities[1].features) == 1


@pytest.mark.parametrize("tier", ["weak", "reject", "strong"])
def test_severe_geometry_guard_precedes_quality_paths(tier):
    bank = bank_with_anchor()
    assert assign(bank, bbox_quality_tier=tier, bbox_quality_ok=tier == "strong",
                  bbox_quality_reason="edge_touch>2" if tier != "strong" else "") == 0
    assert bank.last_assignments[1]["reason"] == "mapped_geometry_reject"


def test_contradiction_cannot_age_into_late_reacquisition():
    bank = bank_with_anchor()
    assert assign(bank) == 0
    for frame in (116, 140, 200):
        assert assign(bank, frame=frame, distance=.037, preferred_uid=1,
                      preferred_candidate_ok=True,
                      metadata_extra={"search_reacquire_context_active": True}) == 0
        assert bank.last_assignments[1]["reason"] == "mapped_geometry_reject"
    assert bank.identities[1].last_strong_observation["capture_frame_id"] == 334


@pytest.mark.parametrize("reverse", [False, True])
def test_central_person_is_not_excluded_and_completes_two_frame_recovery(reverse):
    bank = bank_with_anchor()
    # Exercise the pre-assignment whole-frame path, not only assign().
    observations = [observation(1, WRONG, .19029, mapped=1),
                    observation(3, TRUE_341, .0506)]
    bank.observe_frame_evidence(frame_index=116, width=640, height=480,
                                observations=observations[::-1] if reverse else observations)
    assert bank.search_exclusion_for(3, 1, frame_index=116) is None
    assert 1 not in bank.track_to_uid
    competition = dict(uid=1, frame_index=116, passed=True,
                       distance=.0506, competitor_distance=.19029)
    assert assign(bank, track=3, frame=116, bbox=TRUE_341, distance=.0506,
                  candidate_count=3, metadata_extra={"identity_competition": competition}) == 0
    assert bank.last_assignments[3]["handoff_streak"] == 1
    competition["frame_index"] = 117
    assert assign(bank, track=3, frame=117, bbox=TRUE_346, distance=.0475,
                  candidate_count=3, metadata_extra={"identity_competition": competition}) == 1
    assert bank.last_assignments[3]["handoff_streak"] == 2
    assert bank.last_assignments[3]["template_update_quarantined"]
    assert len(bank.identities[1].features) == 1


def test_revoke_only_negative_evidence_created_after_trusted_anchor():
    bank = bank_with_anchor()
    # Model a pre-fix bad witness at CAP341, then discover its contradiction.
    memory = bank._identity_exclusion
    memory.observe_frame(frame_index=116, width=640, height=480, observations=[
        {**observation(1, WRONG, .19), "trusted_uid": 1},
        observation(3, TRUE_341, .05),
    ])
    assert bank.search_exclusion_for(3, 1, frame_index=116)
    assert assign(bank, frame=116, distance=.19) == 0
    assert bank.search_exclusion_for(3, 1, frame_index=116) is None


def test_matching_pair_review_is_pure_and_does_not_revoke_valid_owner():
    bank = bank_with_anchor()
    assert bank.review_mapped_geometry(1, metadata(WRONG, 338, 10.23), 115)["mapped_geometry_blocked"]
    assert bank.track_to_uid == {1: 1}
    assert not bank._mapped_geometry_conflicts
    assert not bank.review_mapped_geometry(1, metadata(ANCHOR, 338, 10.23), 115)["mapped_geometry_blocked"]


@pytest.mark.parametrize("bbox", [
    (250, 180, 325, 300),  # area change alone: partial/occluded body
    (0, 0, 407, 479),    # close clipped person, including three edges
    (200, 25, 410, 477), # ordinary jitter
])
def test_partial_or_clipped_box_does_not_trigger_combined_swap_rule(bbox):
    bank = bank_with_anchor()
    assert assign(bank, bbox=bbox, distance=.05) == 1
    assert not bank._mapped_geometry_conflicts


def test_camera_yaw_compensation_prevents_false_position_contradiction():
    bank = bank_with_anchor()
    delta = ((WRONG[0]+WRONG[2])-(ANCHOR[0]+ANCHOR[2]))/1280
    # Existing geometry convention: image displacement = -yaw / HFOV.
    meta = metadata(WRONG, 338, 10.23, -47.252393-delta*90)
    review = bank.review_mapped_geometry(1, meta, 115)
    assert review["reason"] == "area_change"
    assert not review["mapped_geometry_blocked"]


def test_stale_or_missing_geometry_cannot_clear_existing_contradiction():
    bank = bank_with_anchor()
    assert assign(bank) == 0
    assert assign(bank, frame=116, bbox=ANCHOR,
                  metadata_extra={"is_fresh": False}) == 0
    assert bank.review_mapped_geometry(1, {}, 117, commit=True)["mapped_geometry_blocked"]


def test_valid_witness_still_excludes_actual_other_person():
    bank = bank_with_anchor()
    bank.observe_frame_evidence(frame_index=115, width=640, height=480, observations=[
        observation(1, ANCHOR, .05, mapped=1), observation(3, WRONG, .19),
    ])
    exclusion = bank.search_exclusion_for(3, 1, frame_index=115)
    assert exclusion["witness_geometry_valid"] is True
    assert exclusion["witness_reference_capture_frame_id"] == 334


def test_missing_reference_or_ambiguous_reid_is_not_negative_evidence():
    bank = bank_with_anchor()
    ref = observation(1, ANCHOR, .05, mapped=1)
    ref["identity_competition"] = dict(uid=1, frame_index=115, passed=False)
    bank.observe_frame_evidence(frame_index=115, width=640, height=480,
                                observations=[ref, observation(3, WRONG, .06)])
    assert bank.search_exclusion_for(3, 1, frame_index=115) is None
    bank.identities[1].last_strong_observation = None
    ref.pop("identity_competition")
    bank.observe_frame_evidence(frame_index=116, width=640, height=480,
                                observations=[ref, observation(3, WRONG, .19)])
    assert bank.search_exclusion_for(3, 1, frame_index=116) is None


def test_wrapper_guard_uses_raw_detection_not_smoothed_track_box():
    tracker = DeepSortTracker(DeepSortTrackerConfig())
    tracker.identity_bank = bank_with_anchor()
    tracker._frame_index = 115
    tracker._frame_context = dict(capture_frame_id=338, capture_timestamp=10.23)
    tracker._current_detections = (Detection(WRONG, .893, 0), Detection(ANCHOR, .946, 0))
    assert not tracker._identity_match_allowed(1, 0, 640, 480)
    assert tracker._identity_match_allowed(1, 1, 640, 480)
    assert tracker.identity_bank.track_to_uid == {1: 1}


def test_wrapper_wires_guard_into_real_deepsort_and_preserves_good_track():
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1))
    # Use un-clipped original geometry so the baseline identity is strong.
    anchor = (204, 40, 407, 450)
    detections = [Detection(anchor, .946, 0)]
    for i in range(4):
        tracker.update(detections, [feature()], image_width=640, image_height=480,
                       frame_context=dict(capture_frame_id=i, capture_timestamp=10+i*.1))
    assert tracker.identity_bank.track_to_uid.get(1) == 1
    records = tracker.update(
        [Detection(WRONG, .893, 0), Detection(anchor, .946, 0)],
        [feature(.19), feature(.05)], image_width=640, image_height=480,
        frame_context=dict(capture_frame_id=338, capture_timestamp=10.5),
    )
    selected = [r for r in records if r.reid_uid == 1]
    assert len(selected) == 1
    assert abs(selected[0].cx/640-.477) < .03
    assert tracker.identity_bank.identities[1].last_strong_observation["bbox"][0] == 204


def test_reset_clears_conflicts_without_modifying_template_policy():
    bank = bank_with_anchor()
    assign(bank)
    bank.reset()
    assert not bank._mapped_geometry_conflicts and not bank._geometry_revoked_uids


@pytest.mark.parametrize("frame", [116, 125])
def test_ambiguous_candidate_cannot_fall_back_to_instant_handoff(frame):
    bank = bank_with_anchor()
    assert assign(bank) == 0
    competition = dict(uid=1, frame_index=frame, passed=False, distance_gap=.01)
    assert assign(bank, track=3, frame=frame, bbox=TRUE_341, distance=.01,
                  candidate_count=2, metadata_extra={"identity_competition": competition}) == 0
    assert bank.last_assignments[3]["reason"] == "revoked_owner_candidate_unqualified"
    assert 3 not in bank.pending_handoffs


def test_old_rejected_track_cannot_reopen_a_completed_uid_recovery():
    bank = bank_with_anchor()
    assert assign(bank) == 0
    assert assign(bank, track=3, frame=116, bbox=TRUE_341, distance=.05) == 0
    assert assign(bank, track=3, frame=117, bbox=TRUE_346, distance=.0475) == 1
    assert not bank._geometry_revoked_uids
    assert assign(bank, frame=118, distance=.03) == 0
    assert bank.track_to_uid == {3: 1}
    assert not bank._geometry_revoked_uids
