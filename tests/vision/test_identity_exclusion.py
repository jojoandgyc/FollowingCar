from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rk_vision.identity_exclusion import IdentityExclusionMemory


TARGET = (361.644867, 2.539612, 637.908325, 474.887024)
CANDIDATE = (211.983749, 163.882309, 310.605469, 265.126831)


def observation(track_id, bbox=CANDIDATE, *, uid=0, capture=1550, timestamp=10.0, yaw=0.0, **extra):
    return {
        "raw_track_id": track_id, "detector_bbox": bbox, "trusted_uid": uid,
        "capture_frame_id": capture, "capture_timestamp": timestamp,
        "integrated_yaw_deg": yaw, "is_fresh": True, **extra,
    }


def observe(memory, frame, items):
    memory.observe_frame(frame_index=frame, observations=items, width=640, height=480)


def seeded_memory():
    memory = IdentityExclusionMemory()
    observe(memory, 582, [observation(26, TARGET, uid=1), observation(35)])
    return memory


@pytest.mark.parametrize("reverse", [False, True])
def test_simultaneous_id_swap_moves_exclusion_not_identity(reverse):
    memory = seeded_memory()
    # Both raw IDs still exist, but now describe the opposite person.
    items = [observation(26, CANDIDATE, uid=1, capture=1554, timestamp=10.1,
                         identity_swap=True),
             observation(35, TARGET, capture=1554, timestamp=10.1, identity_swap=True)]
    observe(memory, 583, items[::-1] if reverse else items)
    evidence = memory.exclusion_for(26, 1, frame_index=583)
    assert evidence["source_capture_frame_id"] == 1550
    assert evidence["transferred_from_track_id"] == 35
    assert evidence["association_reason"] == "unique_geometry_track_transfer"
    assert memory.exclusion_for(35, 1, frame_index=583) is None
    # Correct target receives yet another ID; wrong person keeps its label.
    observe(memory, 584, [observation(26, CANDIDATE, capture=1558, timestamp=10.2),
                          observation(36, TARGET, capture=1558, timestamp=10.2)])
    assert memory.exclusion_for(26, 1, frame_index=584) is not None
    assert memory.exclusion_for(36, 1, frame_index=584) is None


def test_swapped_id_without_unique_geometry_never_inherits_or_seeds():
    memory = seeded_memory()
    observe(memory, 583, [
        observation(26, CANDIDATE, uid=1, capture=1554, timestamp=10.1, identity_swap=True),
        observation(36, CANDIDATE, capture=1554, timestamp=10.1),
        observation(35, TARGET, capture=1554, timestamp=10.1, identity_swap=True),
    ])
    assert memory.exclusion_for(26, 1, frame_index=583) is None
    assert memory.exclusion_for(36, 1, frame_index=583) is None
    assert memory.exclusion_for(35, 1, frame_index=583) is None


@pytest.mark.parametrize("reverse", [False, True])
def test_actual_capture_covisibility_is_order_independent(reverse):
    memory = IdentityExclusionMemory()
    items = [observation(26, TARGET, uid=1), observation(35)]
    observe(memory, 582, items[::-1] if reverse else items)
    evidence = memory.exclusion_for(35, 1, frame_index=582)
    assert evidence["reason"] == "co_visible_distinct_person"
    assert evidence["source_capture_frame_id"] == 1550
    assert evidence["reference_track_id"] == 26
    assert evidence["candidate_track_id"] == 35
    assert evidence["bbox"] == list(CANDIDATE)
    assert memory.exclusion_for(26, 1, frame_index=582) is None
    assert memory.exclusion_for(35, 2, frame_index=582) is None


def test_capture1554_refreshes_source_without_requiring_a_strong_edge_bbox():
    memory = seeded_memory()
    observe(memory, 583, [
        observation(26, (500.163513, 6.20079, 639.234863, 471.099609), uid=1,
                    capture=1554, timestamp=10.2, bbox_quality_ok=False),
        observation(35, (212.870758, 164.792297, 309.518524, 263.363708),
                    capture=1554, timestamp=10.2),
    ])
    assert memory.exclusion_for(35, 1, frame_index=583)["source_capture_frame_id"] == 1554


@pytest.mark.parametrize("source_change,candidate_change", [
    ({"trusted_uid": 0}, {}), ({"is_fresh": False}, {}),
    ({"duplicate": True}, {}), ({"identity_swap": True}, {}),
    ({}, {"capture_frame_id": 1554}), ({}, {"capture_frame_id": None}),
    ({}, {"is_fresh": False}), ({}, {"duplicate": True}),
    ({}, {"identity_swap": True}),
])
def test_untrusted_or_non_simultaneous_observations_do_not_create_exclusions(source_change, candidate_change):
    memory = IdentityExclusionMemory()
    observe(memory, 582, [
        {**observation(26, TARGET, uid=1), **source_change},
        {**observation(35), **candidate_change},
    ])
    assert memory.exclusion_for(35, 1, frame_index=582) is None


@pytest.mark.parametrize("bbox", [
    (400, 100, 500, 200),  # Contained torso/duplicate, not another person.
    (300, 100, 500, 300),  # Partial overlap.
    (260, 100, 355, 260),  # Only 6.6 px gap, below 2% image width.
])
def test_overlap_containment_or_insufficient_separation_do_not_prove_distinctness(bbox):
    memory = IdentityExclusionMemory()
    observe(memory, 1, [observation(26, TARGET, uid=1), observation(35, bbox)])
    assert memory.exclusion_for(35, 1, frame_index=1) is None


def test_vertical_separation_is_supported():
    memory = IdentityExclusionMemory()
    observe(memory, 1, [observation(26, (200, 10, 300, 100), uid=1), observation(35)])
    assert memory.exclusion_for(35, 1, frame_index=1) is not None


def test_continuous_candidate_keeps_exclusion_after_old_target_anchor_expires():
    memory = seeded_memory()
    for frame in range(583, 630):
        observe(memory, frame, [observation(35, capture=1550 + frame, timestamp=10 + (frame - 582) * 0.1)])
    evidence = memory.exclusion_for(35, 1, frame_index=629, capture_timestamp=14.7)
    assert evidence is not None
    assert evidence["source_capture_frame_id"] == 1550
    assert evidence["last_frame"] == 629


@pytest.mark.parametrize("frame,timestamp", [(598, 10.1), (583, 11.01)])
def test_unobserved_candidate_expires_by_either_frame_or_time(frame, timestamp):
    memory = seeded_memory()
    assert memory.exclusion_for(35, 1, frame_index=frame, capture_timestamp=timestamp) is None
    observe(memory, frame, [observation(35, capture=1600, timestamp=timestamp)])
    assert memory.exclusion_for(35, 1, frame_index=frame, capture_timestamp=timestamp) is None


def test_same_raw_track_can_bridge_a_short_gap_but_not_a_geometry_jump():
    memory = seeded_memory()
    observe(memory, 589, [observation(35, capture=1571, timestamp=10.7)])
    assert memory.exclusion_for(35, 1, frame_index=589) is not None
    observe(memory, 590, [observation(35, (480, 150, 580, 250), capture=1573, timestamp=10.8)])
    assert memory.exclusion_for(35, 1, frame_index=590) is None


@pytest.mark.parametrize("new_id", [-1, 36])
def test_unique_short_local_tracklet_inherits_across_real_and_probe_ids(new_id):
    memory = seeded_memory()
    observe(memory, 583, [observation(new_id, capture=1554, timestamp=10.1)])
    evidence = memory.exclusion_for(new_id, 1, frame_index=583)
    assert evidence["reference_track_id"] == 26
    assert evidence["candidate_track_id"] == new_id
    assert memory.exclusion_for(35, 1, frame_index=583) is None
    observe(memory, 584, [observation(37, capture=1558, timestamp=10.2)])
    assert memory.exclusion_for(37, 1, frame_index=584) is not None


@pytest.mark.parametrize("gap,delta", [(3, 0.1), (1, 0.351)])
def test_raw_id_reconstruction_requires_a_short_interval(gap, delta):
    memory = seeded_memory()
    observe(memory, 582 + gap, [observation(-1, capture=1558, timestamp=10 + delta)])
    assert memory.exclusion_for(-1, 1, frame_index=582 + gap) is None


@pytest.mark.parametrize("yaw", [9.0, None, -9.0])
def test_only_known_correctly_signed_yaw_can_explain_a_large_image_shift(yaw):
    memory = seeded_memory()
    moved = (CANDIDATE[0] - 64, CANDIDATE[1], CANDIDATE[2] - 64, CANDIDATE[3])
    observe(memory, 583, [observation(-1, moved, capture=1554, timestamp=10.1, yaw=yaw)])
    assert (memory.exclusion_for(-1, 1, frame_index=583) is not None) == (yaw == 9.0)


def test_two_current_candidates_matching_one_old_tracklet_do_not_inherit():
    memory = seeded_memory()
    observe(memory, 583, [observation(36, capture=1554, timestamp=10.1),
                          observation(37, capture=1554, timestamp=10.1)])
    assert memory.exclusion_for(36, 1, frame_index=583) is None
    assert memory.exclusion_for(37, 1, frame_index=583) is None


def test_two_old_tracklets_matching_one_new_candidate_do_not_inherit():
    memory = IdentityExclusionMemory()
    observe(memory, 582, [observation(26, TARGET, uid=1), observation(35), observation(34)])
    observe(memory, 583, [observation(-1, capture=1554, timestamp=10.1)])
    assert memory.exclusion_for(-1, 1, frame_index=583) is None


def test_existing_current_track_also_counts_as_a_transfer_competitor():
    memory = seeded_memory()
    # A second track was far away, then moves into the same local region as
    # the disappearing candidate. A new raw ID there is no longer unique.
    observe(memory, 583, [
        observation(35, capture=1554, timestamp=10.1),
        observation(40, (30, 160, 130, 260), capture=1554, timestamp=10.1),
    ])
    observe(memory, 584, [
        observation(40, capture=1558, timestamp=10.2),
        observation(36, capture=1558, timestamp=10.2),
    ])
    assert memory.exclusion_for(36, 1, frame_index=584) is None


def test_still_visible_old_candidate_does_not_donate_its_identity_exclusion():
    memory = seeded_memory()
    observe(memory, 583, [observation(35, capture=1554, timestamp=10.1),
                          observation(36, capture=1554, timestamp=10.1)])
    assert memory.exclusion_for(35, 1, frame_index=583) is not None
    assert memory.exclusion_for(36, 1, frame_index=583) is None


def test_no_capture_or_reacquire_geometry_alone_creates_an_exclusion():
    memory = IdentityExclusionMemory()
    observe(memory, 1, [observation(26, TARGET, uid=1)])
    observe(memory, 20, [observation(35, capture=1600, timestamp=12.0)])
    assert memory.exclusion_for(35, 1, frame_index=20) is None


def test_query_copy_and_reset_do_not_leak_mutable_or_reused_track_state():
    memory = seeded_memory()
    evidence = memory.exclusion_for(35, 1, frame_index=582)
    evidence["reference_bbox"][0] = -1000
    assert memory.exclusion_for(35, 1, frame_index=582)["reference_bbox"][0] == TARGET[0]
    memory.reset()
    observe(memory, 1, [observation(35, capture=1, timestamp=0.1)])
    assert memory.exclusion_for(35, 1, frame_index=1) is None
