"""First-person enrollment ownership; synthetic observations, no hardware."""
import numpy as np
import pytest

from rk_vision.initial_enrollment import InitialEnrollment
from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


BOX = (200., 40., 400., 460.)
FEATURE = np.array([1., 0., 0.], dtype=np.float32)
OTHER_FEATURE = np.array([0., 1., 0.], dtype=np.float32)


def metadata(frame, **changes):
    result = dict(capture_frame_id=100 + frame, capture_timestamp=10. + frame * .1,
                  frame_index=frame, control_frame_id=frame, candidate_count=1,
                  detector_bbox=BOX, bbox=BOX, image_width=640, image_height=480,
                  quality_bbox_ok=True, bbox_quality_tier='strong', is_fresh=True)
    result.update(changes)
    return result


def observe(enrollment, frame, *, track=1, feature=FEATURE, **changes):
    return enrollment.observe(track, feature, metadata(frame, **changes))


def test_first_unique_candidate_cannot_be_replaced_by_repeated_second_track():
    enrollment = InitialEnrollment()
    assert observe(enrollment, 1) == (False, 'pending_initial_identity')
    for frame in range(2, 7):
        assert observe(enrollment, frame, track=2) == (
            False, 'initial_candidate_track_changed')
    assert enrollment.candidate_track_id == 1
    assert observe(enrollment, 7) == (False, 'pending_initial_identity')
    assert observe(enrollment, 8) == (True, 'created_confirmed')


@pytest.mark.parametrize('interruption', [
    {'candidate_count': 2}, {'quality_bbox_ok': False}, {'is_fresh': False},
    {'detector_bbox': (0., 40., 150., 460.)}, {'capture_timestamp': None},
])
def test_interruption_restarts_proof_without_releasing_first_candidate(interruption):
    enrollment = InitialEnrollment()
    observe(enrollment, 1)
    assert not observe(enrollment, 2, **interruption)[0]
    assert enrollment.candidate_track_id == 1
    assert not observe(enrollment, 3, track=2)[0]
    assert not observe(enrollment, 4, track=2)[0]
    assert observe(enrollment, 5) == (False, 'pending_initial_identity')
    assert observe(enrollment, 6) == (True, 'created_confirmed')


@pytest.mark.parametrize('changes', [{}, {'capture_frame_id': 100},
                                    {'capture_timestamp': 10.}])
def test_replayed_evidence_cannot_replace_or_complete_first_candidate(changes):
    enrollment = InitialEnrollment()
    observe(enrollment, 1)
    for track in (2, 1, 2, 1):
        assert observe(enrollment, 1, track=track, **changes) == (
            False, 'initial_capture_not_new')
    assert enrollment.candidate_track_id == 1
    assert observe(enrollment, 2) == (True, 'created_confirmed')


def test_rejected_capture_cannot_be_replayed_with_qualified_metadata():
    enrollment = InitialEnrollment()
    observe(enrollment, 1)
    assert not observe(enrollment, 2, quality_bbox_ok=False)[0]
    assert observe(enrollment, 2) == (False, 'initial_capture_not_new')
    assert observe(enrollment, 3) == (False, 'pending_initial_identity')
    assert observe(enrollment, 4) == (True, 'created_confirmed')


def test_long_gap_keeps_first_candidate_and_requires_a_new_continuous_pair():
    enrollment = InitialEnrollment()
    observe(enrollment, 1)
    assert observe(enrollment, 100, track=2) == (
        False, 'initial_candidate_track_changed')
    assert observe(enrollment, 101) == (False, 'pending_initial_identity')
    assert observe(enrollment, 102) == (True, 'created_confirmed')


def test_long_gap_alone_cannot_complete_first_candidate():
    enrollment = InitialEnrollment()
    observe(enrollment, 1)
    assert observe(enrollment, 100) == (False, 'pending_initial_identity')
    assert observe(enrollment, 101) == (True, 'created_confirmed')


def test_same_track_cannot_replace_first_appearance_after_repeated_mismatch():
    enrollment = InitialEnrollment()
    observe(enrollment, 1)
    for frame in range(2, 6):
        assert observe(enrollment, frame, feature=OTHER_FEATURE) == (
            False, 'initial_candidate_appearance_mismatch')
    assert observe(enrollment, 6) == (False, 'pending_initial_identity')
    assert observe(enrollment, 7) == (True, 'created_confirmed')


def test_initial_multiple_people_do_not_reserve_either_track():
    enrollment = InitialEnrollment()
    for track in (1, 2):
        assert observe(enrollment, 1, track=track, candidate_count=2) == (
            False, 'initial_candidate_ambiguous')
    assert enrollment.candidate_track_id is None
    assert observe(enrollment, 2, track=2) == (False, 'pending_initial_identity')
    assert observe(enrollment, 3, track=2) == (True, 'created_confirmed')


def test_explicit_reset_allows_selection_of_a_different_first_candidate():
    enrollment = InitialEnrollment()
    observe(enrollment, 1)
    enrollment.reset()
    assert observe(enrollment, 1, track=2, feature=OTHER_FEATURE) == (
        False, 'pending_initial_identity')
    assert observe(enrollment, 2, track=2, feature=OTHER_FEATURE) == (
        True, 'created_confirmed')


def test_identity_bank_does_not_create_a_uid_for_the_second_track():
    bank = IdentityBank(IdentityBankConfig(appearance_region_safety_enable=True))
    for frame, track in ((1, 1), (2, 2), (3, 2), (4, 2), (5, 1), (6, 1)):
        uid = bank.assign(track_id=track, feature=FEATURE, confidence=.94,
                          area=84000., frame_index=frame, sample_metadata=metadata(frame),
                          candidate_count=1, bbox_quality_ok=True, bbox_quality_tier='strong')
        assert uid == (1 if frame == 6 else 0)
    assert bank.track_to_uid == {1: 1}


@pytest.mark.parametrize('max_age, detected_frames, expected', [
    (10, (True, True, False, True, True),
     [[], [(1, 0)], [], [(1, 0)], [(1, 1)]]),
    # Tentative tracks may be deleted on one miss, but no first candidate has
    # been selected yet because the wrapper has not exposed that track.
    (10, (True, False, True, True, True),
     [[], [], [], [(2, 0)], [(2, 1)]]),
    # Once selected, expiry cannot silently transfer ownership to raw track 2.
    (1, (True, True, False, False, True, True, True),
     [[], [(1, 0)], [], [], [], [(2, 0)], [(2, 0)]]),
])
def test_real_tracker_gap_and_raw_track_replacement_boundaries(max_age, detected_frames, expected):
    tracker = DeepSortTracker(DeepSortTrackerConfig(
        n_init=1, max_age=max_age, identity_appearance_region_safety_enable=True))
    observations = []
    for frame, detected in enumerate(detected_frames, 1):
        records = tracker.update(
            [Detection(BOX, .94, 0)] if detected else [],
            [FEATURE] if detected else [], image_width=640, image_height=480,
            frame_context=dict(control_frame_id=frame, capture_frame_id=100 + frame,
                               capture_timestamp=10. + frame * .1))
        observations.append([(record.track_id, record.reid_uid)
                             for record in records if record.time_since_update == 0])
    assert observations == expected
    if max_age == 1:
        assert not tracker.identity_bank.identities
        assert tracker.identity_bank._initial_enrollment.candidate_track_id == 1
        assert tracker.identity_bank.last_assignments[2]['reason'] == 'initial_candidate_track_changed'
