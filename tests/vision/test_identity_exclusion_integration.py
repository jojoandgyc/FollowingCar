"""CAP1550/1554 co-visibility must survive CAP1567/1571 late reacquire."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rk_vision.identity_bank import IdentityBank, IdentityBankConfig


def feature(distance=0.0):
    cosine = 1.0 - distance
    return np.array([cosine, np.sqrt(1.0 - cosine * cosine), 0.0], dtype=np.float32)


CORRECT = (361.645, 2.54, 637.908, 474.887)
WRONG = (211.984, 163.882, 310.605, 265.127)


def metadata(cap, timestamp, bbox=WRONG, **extra):
    x1, y1, x2, y2 = bbox
    return {
        "capture_frame_id": cap, "capture_timestamp": timestamp,
        "bbox": bbox, "detector_bbox": bbox,
        "detector_center_x_ratio": (x1 + x2) / 1280.0,
        "detector_area_ratio": (x2 - x1) * (y2 - y1) / (640 * 480),
        "is_fresh": True, "integrated_yaw_deg": 0.0, **extra,
    }


def observation(track, cap, timestamp, bbox, *, mapped=0, distance=0.0, **extra):
    return {
        "raw_track_id": track, "detector_bbox": bbox, "feature": feature(distance),
        "mapped_uid": mapped, "confidence": 0.9, "is_fresh": True,
        "capture_frame_id": cap, "capture_timestamp": timestamp,
        "integrated_yaw_deg": 0.0, **extra,
    }


def make_bank():
    bank = IdentityBank(IdentityBankConfig(
        new_identity_confirm_frames=1, min_confidence=0.60,
        controlled_handoff_enable=True,
        controlled_handoff_min_old_track_gap_frames=1,
        update_interval=1,
    ))
    uid = bank.assign(
        track_id=26, feature=feature(), confidence=0.95, area=100000,
        frame_index=1, sample_metadata=metadata(1545, 10.0, CORRECT),
    )
    assert uid == 1
    return bank


def co_visible(bank, **reference_extra):
    bank.observe_frame_evidence(
        frame_index=2, width=640, height=480, observations=[
            # Deliberately put the distractor first: order must not matter.
            observation(35, 1550, 10.2, WRONG, distance=0.21686),
            observation(26, 1550, 10.2, CORRECT, mapped=1, distance=0.15255,
                        **reference_extra),
        ],
    )


def assign_candidate(bank, frame, cap, timestamp, *, track=35, weak=False, preferred=True):
    bank.observe_frame_evidence(
        frame_index=frame, width=640, height=480,
        observations=[observation(track, cap, timestamp, WRONG, distance=0.192)],
    )
    return bank.assign(
        track_id=track, feature=feature(0.192), partial_feature=feature(),
        confidence=0.90, area=10000, frame_index=frame,
        bbox_quality_ok=not weak, bbox_quality_tier="weak" if weak else "strong",
        bbox_quality_reason="edge_touch>2" if weak else "",
        preferred_uid=1 if preferred else None, preferred_candidate_ok=preferred,
        sample_metadata=metadata(cap, timestamp, search_reacquire_context_active=preferred,
                                 search_direction="right", search_direction_compatible=False),
    )


@pytest.mark.parametrize("weak,preferred", [(False, True), (True, True), (False, False)])
def test_simultaneous_person_cannot_use_stale_anchor_or_two_local_frames(weak, preferred):
    bank = make_bank()
    co_visible(bank)
    # The incorrect person remains visible after the correct person exits.
    # Preserve its evidence beyond both the 350ms old anchor and 1s gap limit:
    # that limit measures candidate absence, not time since original sighting.
    for frame in range(3, 13):
        assert assign_candidate(bank, frame, 1550 + frame, 10.2 + (frame - 2) * .15,
                                weak=weak, preferred=preferred) == 0
        diag = bank.last_assignments[35]
        assert diag["reason"] == "search_candidate_excluded"
        assert diag["search_exclusion"]["source_capture_frame_id"] == 1550
        assert diag["excluded_uid"] == 1
        assert not diag["bank_updated"]
        assert 35 not in bank.pending_late_handoffs
        assert 35 not in bank.pending_handoffs
    assert bank.track_to_uid == {26: 1}
    assert len(bank.identities[1].features) == 1


def test_without_negative_evidence_normal_late_two_frame_confirmation_is_preserved():
    bank = make_bank()
    assert assign_candidate(bank, 10, 1567, 11.133) == 0
    assert bank.last_assignments[35]["reason"] == "preferred_search_late_candidate_wait"
    assert assign_candidate(bank, 11, 1571, 11.332) == 1
    assert bank.last_assignments[35]["reason"] == "preferred_search_late_reacquire"
    assert bank.last_assignments[35]["template_update_quarantined"]


@pytest.mark.parametrize("extra", [
    {"is_fresh": False}, {"duplicate": True}, {"identity_swap": True},
    {"mapped_uid": 0}, {"confidence": .50}, {"feature": None},
])
def test_untrusted_witness_cannot_exclude_another_person(extra):
    bank = make_bank()
    reference = observation(26, 1550, 10.2, CORRECT, mapped=1)
    reference.update(extra)
    bank.observe_frame_evidence(frame_index=2, width=640, height=480, observations=[
        reference, observation(35, 1550, 10.2, WRONG, distance=.192),
    ])
    assert bank.search_exclusion_for(35, 1, frame_index=2) is None


def test_weak_edge_mapped_reference_can_witness_but_quarantined_one_cannot():
    bank = make_bank()
    # The mapped target is publicly UID0 solely because its detector is clipped.
    assert bank.assign(
        track_id=26, feature=feature(.15255), confidence=.9, area=100000,
        frame_index=2, bbox_quality_ok=False, bbox_quality_tier="weak",
        bbox_quality_reason="edge_touch>2",
        sample_metadata=metadata(1550, 10.2, CORRECT),
    ) == 0
    co_visible(bank)
    assert bank.search_exclusion_for(35, 1, frame_index=2)
    bank.reset()
    bank = make_bank()
    bank._reacquire_quarantine.arm(1, 26, 1545, 10.0, 1)
    co_visible(bank)
    assert bank.search_exclusion_for(35, 1, frame_index=2) is None


def test_strict_new_track_continuity_inherits_evidence_not_reset_by_search_context():
    bank = make_bank()
    co_visible(bank)
    assert assign_candidate(bank, 3, 1554, 10.3, track=36) == 0
    assert bank.last_assignments[36]["reason"] == "search_candidate_excluded"
    # Pure time without candidate observations does eventually retire it.
    assert bank.search_exclusion_for(36, 1, frame_index=30, capture_timestamp=12.0) is None


def test_reset_and_other_uid_are_not_globally_blacklisted():
    bank = make_bank()
    co_visible(bank)
    assert bank.search_exclusion_for(35, 2, frame_index=2) is None
    bank.reset()
    assert bank.search_exclusion_for(35, 1, frame_index=2) is None


def test_exclusion_of_one_uid_does_not_prevent_distinct_new_person_enrollment():
    bank = make_bank()
    co_visible(bank)
    uid = bank.assign(
        track_id=35, feature=feature(1.0), confidence=.9, area=10000,
        frame_index=2, sample_metadata=metadata(1550, 10.2), candidate_count=2,
    )
    assert uid == 2
    assert bank.track_to_uid == {26: 1, 35: 2}
    assert bank.search_exclusion_for(35, 1, frame_index=2)
    assert bank.search_exclusion_for(35, 2, frame_index=2) is None


def test_multiple_current_claims_for_one_uid_cannot_exclude_each_other():
    bank = make_bank()
    bank.track_to_uid[35] = 1
    bank.observe_frame_evidence(frame_index=2, width=640, height=480, observations=[
        observation(26, 1550, 10.2, CORRECT, mapped=1),
        observation(35, 1550, 10.2, WRONG, mapped=1),
    ])
    assert bank.search_exclusion_for(35, 1, frame_index=2) is None
    assert bank.search_exclusion_for(26, 1, frame_index=2) is None


def test_direct_late_observer_cannot_overwrite_exclusion():
    bank = make_bank()
    co_visible(bank)
    result = bank._observe_late_search_candidate(
        track_id=35, candidate_uid=1, distance=.01, partial_feature=None,
        match_source="strong", candidate_count=1, bbox_quality_ok=True,
        sample_metadata=metadata(1554, 10.3), frame_index=3,
        geometry={"ok": False, "reason": "stale_reference"},
    )
    assert result[0] == 0
    assert result[2]["late_candidate_rejection"] == "search_candidate_excluded"
    assert 35 not in bank.pending_late_handoffs
