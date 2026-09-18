"""Revoking an unreliable witness must follow the original exclusion source."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rk_vision.identity_exclusion import IdentityExclusionMemory


REFERENCE = (450, 20, 620, 460)
CANDIDATE = (200, 150, 300, 300)
OTHER_CANDIDATE = (40, 320, 140, 470)


def observation(track_id, bbox=CANDIDATE, *, uid=0, capture=341, **extra):
    return {
        "raw_track_id": track_id,
        "detector_bbox": bbox,
        "trusted_uid": uid,
        "capture_frame_id": capture,
        "capture_timestamp": capture * 0.01,
        "integrated_yaw_deg": 0.0,
        "is_fresh": True,
        **extra,
    }


def observe(memory, frame, observations):
    memory.observe_frame(
        frame_index=frame, observations=observations, width=640, height=480,
    )


def test_invalidate_witness_removes_all_affected_candidates_once():
    memory = IdentityExclusionMemory()
    observe(memory, 101, [
        observation(26, REFERENCE, uid=1),
        observation(35),
        observation(36, OTHER_CANDIDATE),
    ])
    assert memory.exclusion_for(35, 1, frame_index=101)
    assert memory.exclusion_for(36, 1, frame_index=101)

    assert memory.invalidate_witness(1, 26, after_frame=100) == 2
    assert memory.exclusion_for(35, 1, frame_index=101) is None
    assert memory.exclusion_for(36, 1, frame_index=101) is None
    assert memory.invalidate_witness(1, 26, after_frame=100) == 0


def test_revocation_follows_evidence_through_multiple_geometry_transfers():
    memory = IdentityExclusionMemory()
    observe(memory, 101, [observation(26, REFERENCE, uid=1), observation(35)])
    observe(memory, 102, [observation(-1, capture=342)])
    observe(memory, 103, [observation(37, capture=343)])
    evidence = memory.exclusion_for(37, 1, frame_index=103)
    assert evidence["reference_track_id"] == 26
    assert evidence["source_frame"] == 101
    assert evidence["transferred_from_track_id"] == -1

    assert memory.invalidate_witness(1, 26, after_frame=100) == 1
    assert memory.exclusion_for(37, 1, frame_index=103) is None
    # Continued observations cannot bring back the revoked inherited label.
    observe(memory, 104, [observation(38, capture=344)])
    assert memory.exclusion_for(38, 1, frame_index=104) is None


@pytest.mark.parametrize("source_frame", [99, 100])
def test_revocation_preserves_sources_at_or_before_last_trusted_frame(source_frame):
    memory = IdentityExclusionMemory()
    observe(memory, source_frame, [
        observation(26, REFERENCE, uid=1, capture=338),
        observation(35, capture=338),
    ])
    # This candidate's source remains trustworthy even when another candidate
    # receives a later exclusion from the same witness.
    observe(memory, 101, [
        observation(26, REFERENCE, uid=1),
        observation(36, OTHER_CANDIDATE),
    ])
    assert memory.invalidate_witness(1, 26, after_frame=100) == 1
    assert memory.exclusion_for(35, 1, frame_index=101)["source_frame"] == source_frame
    assert memory.exclusion_for(36, 1, frame_index=101) is None


def test_revocation_preserves_other_uids_and_other_witnesses():
    memory = IdentityExclusionMemory()
    observe(memory, 101, [
        observation(26, REFERENCE, uid=1),
        observation(35),
    ])
    # The same candidate can carry an unrelated UID exclusion, and a separate
    # candidate can carry the same UID exclusion from a different witness.
    observe(memory, 102, [
        observation(26, REFERENCE, uid=2, capture=342),
        observation(35, capture=342),
    ])
    observe(memory, 103, [
        observation(28, (380, 20, 550, 460), uid=1, capture=343),
        observation(36, OTHER_CANDIDATE, capture=343),
    ])
    assert memory.exclusion_for(36, 1, frame_index=103)["reference_track_id"] == 28
    other_uid = memory.exclusion_for(35, 2, frame_index=103)
    other_witness = memory.exclusion_for(36, 1, frame_index=103)

    assert memory.invalidate_witness(1, 26, after_frame=100) == 1
    assert memory.exclusion_for(35, 1, frame_index=103) is None
    assert memory.exclusion_for(35, 2, frame_index=103) == other_uid
    assert memory.exclusion_for(36, 1, frame_index=103) == other_witness


def test_revocation_does_not_confuse_capture_or_transfer_frame_with_source_frame():
    memory = IdentityExclusionMemory()
    observe(memory, 100, [
        observation(26, REFERENCE, uid=1),
        observation(35),
    ])
    observe(memory, 101, [observation(36, capture=342)])
    assert memory.invalidate_witness(1, 26, after_frame=100) == 0
    evidence = memory.exclusion_for(36, 1, frame_index=101)
    assert evidence["source_frame"] == 100
    assert evidence["source_capture_frame_id"] == 341
    assert evidence["last_frame"] == 101


@pytest.mark.parametrize("valid", [True, False, None])
def test_witness_geometry_provenance_comes_from_reference_and_survives_transfer(valid):
    memory = IdentityExclusionMemory()
    observe(memory, 101, [
        observation(
            26, REFERENCE, uid=1,
            witness_reference_capture_frame_id=338,
            witness_reference_frame_index=100,
            witness_geometry_valid=valid,
        ),
        observation(
            35,
            witness_reference_capture_frame_id=111,
            witness_reference_frame_index=55,
            witness_geometry_valid=not valid,
        ),
    ])
    observe(memory, 102, [observation(36, capture=342)])
    evidence = memory.exclusion_for(36, 1, frame_index=102)
    assert evidence["witness_reference_capture_frame_id"] == 338
    assert evidence["witness_reference_frame_index"] == 100
    assert evidence["witness_geometry_valid"] is valid
    assert evidence["source_capture_frame_id"] == 341
    assert evidence["source_frame"] == 101


def test_legacy_observations_without_geometry_provenance_still_create_exclusions():
    memory = IdentityExclusionMemory()
    observe(memory, 101, [observation(26, REFERENCE, uid=1), observation(35)])
    evidence = memory.exclusion_for(35, 1, frame_index=101)
    assert evidence is not None
    assert evidence["witness_reference_capture_frame_id"] is None
    assert evidence["witness_reference_frame_index"] is None
    assert evidence["witness_geometry_valid"] is None


def test_unknown_witness_and_uid_are_noops():
    memory = IdentityExclusionMemory()
    observe(memory, 101, [observation(26, REFERENCE, uid=1), observation(35)])
    original = memory.exclusion_for(35, 1, frame_index=101)
    assert memory.invalidate_witness(1, 999, after_frame=100) == 0
    assert memory.invalidate_witness(999, 26, after_frame=100) == 0
    assert memory.exclusion_for(35, 1, frame_index=101) == original
