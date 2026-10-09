"""Identity proof consistency: real policy calls, synthetic descriptors, no hardware."""
from copy import deepcopy

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.reacquire_quarantine import ReacquireQuarantine
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection
from test_cap555_quarantine_exit import checkpoint, send
from test_cap874_identity_reacquire import feature, metadata


BOX = (150., 40., 400., 460.)
OTHER_BOX = (460., 60., 620., 455.)


def search_tracker():
    tracker = DeepSortTracker(DeepSortTrackerConfig())
    bank = IdentityBank(IdentityBankConfig(
        template_memory_enable=True, template_crosscheck_enable=True,
        appearance_region_safety_enable=True, controlled_handoff_enable=True,
        partial_match_threshold=.45, partial_confirm_threshold=.40,
        camera_hfov_deg=66.))
    tracker.identity_bank = bank
    anchor = metadata(100, 10., BOX, search=False)
    anchor.update(track_id=1, frame_index=1, image_width=640, image_height=480,
                  partial_feature_source='osnet_torso', partial_observation=True)
    bank._create_identity(feature(0), 1, anchor, feature(0))
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction='left')
    return tracker


def search_sample(tracker, index, *, other_score=.90, full=.10, other_full=.70,
                  partial=.10, opposite=False):
    """Generate the proof through the production same-frame competition producer."""
    frame = 21 + index
    detections = [Detection(BOX, .95, 0), Detection(OTHER_BOX, other_score, 0)]
    descriptors = [feature(full), feature(other_full)]
    sample = metadata(200 + index, 12. + index * .1, BOX, search=True, opposite=opposite)
    sample.update(track_id=2, frame_index=frame, image_width=640, image_height=480,
                  partial_feature_source='osnet_torso', partial_observation=True,
                  candidate_count=2, source_detection_index=0,
                  candidate_score_gap=tracker._candidate_score_gap(0, detections))
    tracker._frame_context, tracker._frame_index = sample, frame
    proof = tracker._frame_identity_competition(
        detections, descriptors, image_width=640, image_height=480)
    sample['identity_competition'] = proof[0]
    return sample, descriptors[0], feature(partial)


def assign_search(tracker, sample, full, partial):
    return tracker.identity_bank.assign(
        track_id=2, feature=full, partial_feature=partial, confidence=.95,
        area=105000, frame_index=sample['frame_index'], candidate_count=2,
        bbox_quality_ok=True, bbox_quality_tier='strong', sample_metadata=sample,
        preferred_uid=1, preferred_candidate_ok=sample['search_direction_compatible'] is not False)


def test_strong_quarantine_proof_is_not_reset_by_better_torso_matches():
    bank = checkpoint()
    for index, cap in enumerate((423, 426, 428)):
        assert send(bank, cap, full=.10, partial=.20 if index % 2 == 0 else .33) == 1
        assignment = bank.last_assignments[2]
        assert assignment['template_quarantine_streak'] == index + 1
        assert not assignment['bank_updated']
    assert not bank._reacquire_quarantine.is_held(1)
    assert bank.last_assignments[2]['template_quarantine_reason'] == 'released'


def quarantine_sample(quarantine, index, *, full=.10, paired=False, **changes):
    args = dict(uid=1, track_id=2, capture_frame_id=100 + index,
                capture_timestamp=10. + index * .1, frame_index=index,
                is_fresh=True, quality_ok=True, quality_tier='strong',
                match_source='strong', feature_available=True,
                strong_distance=full, center_x_ratio=.50, area_ratio=.30,
                region_pair_verified=paired)
    args.update(changes)
    return quarantine.observe(**args)


@pytest.mark.parametrize('pairs', [(True, True, True), (True, False, True), (False, True, False)])
def test_already_strong_samples_use_the_same_existing_three_frame_proof(pairs):
    quarantine = ReacquireQuarantine()
    quarantine.arm(1, 2, 1, 9., 0)
    for index, paired in enumerate(pairs, 1):
        result = quarantine_sample(quarantine, index, paired=paired)
        assert result.streak == index
    assert not result.hold
    assert result.reason == 'released'


def test_only_regional_samples_still_require_the_full_qualified_time_window():
    quarantine = ReacquireQuarantine()
    quarantine.arm(1, 2, 1, 9., 0)
    for index in range(1, 11):
        result = quarantine_sample(quarantine, index, full=.25, paired=True)
        assert result.hold
    result = quarantine_sample(quarantine, 11, full=.25, paired=True)
    assert not result.hold and result.reason == 'released_region_pair'


def test_a_regional_sample_cannot_complete_an_unfinished_strong_proof():
    quarantine = ReacquireQuarantine()
    quarantine.arm(1, 2, 1, 9., 0)
    assert quarantine_sample(quarantine, 1).streak == 1
    assert quarantine_sample(quarantine, 2).streak == 2
    result = quarantine_sample(quarantine, 3, full=.25, paired=True)
    assert result.hold and result.streak == 1
    result = quarantine_sample(quarantine, 4, full=.10, paired=True)
    assert result.hold and result.streak == 1


@pytest.mark.parametrize('change', [
    {'quality_ok': False}, {'is_fresh': False}, {'feature_available': False},
    {'full': .31}, {'center_x_ratio': .95},
])
def test_strong_priority_does_not_bypass_existing_rejections(change):
    quarantine = ReacquireQuarantine()
    quarantine.arm(1, 2, 1, 9., 0)
    quarantine_sample(quarantine, 1, paired=True)
    result = quarantine_sample(quarantine, 2, paired=True, **change)
    assert result.hold and result.streak == 0


@pytest.mark.parametrize('other_score', [.99, .90, .50])
@pytest.mark.parametrize('full', [.10, .25])
def test_search_reacquisition_uses_uid_competition_not_detector_score_gap(other_score, full):
    tracker = search_tracker()
    for index, expected in enumerate((0, 1, 1, 1)):
        sample, descriptor, partial = search_sample(
            tracker, index, other_score=other_score, full=full)
        proof = sample['identity_competition']
        assert proof['passed'] is True
        assert proof['distance_gap'] == pytest.approx(.70 - full)
        assert assign_search(tracker, sample, descriptor, partial) == expected
        assert not tracker.identity_bank.last_assignments[2]['bank_updated']


@pytest.mark.parametrize('kind', [
    'missing', 'empty', 'foreign_uid', 'old_frame', 'foreign_source', 'foreign_count',
    'missing_source', 'missing_count', 'missing_frame', 'failed', 'override_only',
])
@pytest.mark.parametrize('opposite', [False, True])
def test_invalid_competition_cannot_use_a_large_detector_gap_or_opposite_override(kind, opposite):
    tracker = search_tracker()
    for index in range(3):
        sample, descriptor, partial = search_sample(tracker, index, other_score=.05, opposite=opposite)
        proof = sample['identity_competition']
        if kind == 'missing':
            sample.pop('identity_competition')
        elif kind == 'empty':
            sample['identity_competition'] = {}
        elif kind == 'override_only':
            sample['identity_competition'] = {}
            sample['preferred_search_identity_competition_override'] = True
        elif kind.startswith('missing_'):
            proof.pop({'missing_source': 'source_detection_index', 'missing_count': 'candidate_count',
                       'missing_frame': 'frame_index'}[kind])
        else:
            proof.update({'foreign_uid': {'uid': 99}, 'old_frame': {'frame_index': 1},
                          'foreign_source': {'source_detection_index': 1},
                          'foreign_count': {'candidate_count': 3}, 'failed': {'passed': False}}[kind])
        assert assign_search(tracker, sample, descriptor, partial) == 0
        assert not tracker.identity_bank.last_assignments[2]['bank_updated']
    assert 2 not in tracker.identity_bank.track_to_uid
    assert not tracker.identity_bank._candidate_observations.rows


def test_genuinely_ambiguous_people_remain_rejected_despite_large_yolo_gap():
    tracker = search_tracker()
    for index in range(3):
        sample, descriptor, partial = search_sample(tracker, index, other_score=.05, other_full=.12)
        assert sample['identity_competition']['passed'] is False
        assert assign_search(tracker, sample, descriptor, partial) == 0


def test_uid_competition_does_not_allow_partial_only_opposite_side_reacquisition():
    tracker = search_tracker()
    for index in range(3):
        sample, descriptor, partial = search_sample(tracker, index, full=.25, opposite=True)
        assert sample['identity_competition']['passed'] is True
        assert assign_search(tracker, sample, descriptor, partial) == 0


def test_valid_competition_cannot_erase_a_known_geometry_contradiction():
    tracker = search_tracker()
    bank = tracker.identity_bank
    anchor = deepcopy(bank.identities[1].last_strong_observation)
    bank._mapped_geometry_conflicts[2] = dict(uid=1, search_contradiction=True,
                                             reference=anchor, candidate=anchor)
    sample, descriptor, partial = search_sample(tracker, 0)
    # A large opposite-side displacement is not this UID, even if its ReID wins.
    sample.update(detector_bbox=[500., 40., 630., 460.], bbox=[500., 40., 630., 460.],
                  detector_center_x_ratio=565./640, center_x_ratio=565./640)
    assert assign_search(tracker, sample, descriptor, partial) == 0
    assert not bank.last_assignments[2]['bank_updated']


def test_soft_search_and_late_observer_share_the_same_valid_uid_competition():
    tracker = search_tracker()
    for index, expected in enumerate((0, 1)):
        sample, descriptor, partial = search_sample(tracker, index, full=.25)
        sample['partial_observation'] = False
        assert tracker.identity_bank._preferred_search_reacquire_candidate(
            feature=descriptor, partial_feature=partial, preferred_uid=1,
            candidate_ok=True, candidate_count=2, sample_metadata=sample) is None
        selected = tracker.identity_bank._soft_search_reacquire_candidate(
            feature=descriptor, partial_feature=partial, preferred_uid=1,
            candidate_count=2, sample_metadata=sample, bbox_quality_ok=True,
            bbox_quality_tier='strong')
        assert selected is not None and selected[-1] == 'soft_strong'
        assert assign_search(tracker, sample, descriptor, partial) == expected
        assert not tracker.identity_bank.last_assignments[2]['bank_updated']


@pytest.mark.parametrize('change', [{'detector_area_ratio': .10}, {'detector_confidence': .79}])
def test_soft_multi_person_candidate_keeps_its_independent_crop_quality_requirements(change):
    tracker = search_tracker()
    sample, descriptor, partial = search_sample(tracker, 0, full=.25)
    sample.update(partial_observation=False, **change)
    assert sample['identity_competition']['passed']
    assert tracker.identity_bank._soft_search_reacquire_candidate(
        feature=descriptor, partial_feature=partial, preferred_uid=1,
        candidate_count=2, sample_metadata=sample, bbox_quality_ok=True,
        bbox_quality_tier='strong') is None


@pytest.mark.parametrize('source', ['strong', 'partial', 'soft_strong', 'soft_partial'])
def test_late_observer_cannot_trust_a_legacy_override_instead_of_current_proof(source):
    tracker = search_tracker()
    sample, descriptor, partial = search_sample(tracker, 0)
    sample.update(identity_competition=None, preferred_search_identity_competition_override=True)
    geometry = tracker.identity_bank._handoff_geometry(1, sample, sample['frame_index'])
    assert tracker.identity_bank._observe_late_search_candidate(
        track_id=2, candidate_uid=1, distance=.10, partial_feature=partial,
        match_source=source, candidate_count=2, bbox_quality_ok=True,
        sample_metadata=sample, frame_index=sample['frame_index'], geometry=geometry) is None
    assert not tracker.identity_bank.pending_late_handoffs


def test_weak_search_observation_also_uses_the_current_person_competition():
    tracker = search_tracker()
    sample, descriptor, partial = search_sample(tracker, 0)
    bank = tracker.identity_bank
    assert bank._assign_weak_search_candidate(
        track_id=2, feature=descriptor, partial_feature=partial,
        frame_index=sample['frame_index'], candidate_count=2,
        quality_reason='edge_touch>2', sample_metadata=sample,
        preferred_uid=1, preferred_candidate_ok=True, diagnostics={}) == 0
    assert bank.last_assignments[2]['reason'] == 'weak_preferred_reacquire_wait'
    assert 2 in bank.pending_weak_handoffs
    assert not bank.last_assignments[2]['bank_updated']


def detector_probe(tracker, index):
    tracker._frame_index = 21 + index
    tracker._frame_context = dict(capture_frame_id=200 + index,
                                 capture_timestamp=12. + index * .1,
                                 integrated_yaw_deg=0.)
    # Only the first person qualifies for the probe, but both people are part
    # of the UID competition. A non-person must not change either count.
    detections = [Detection(BOX, .95, 0), Detection(OTHER_BOX, .10, 0),
                  Detection((10., 10., 50., 50.), .95, 2)]
    return tracker._search_probe_record(
        detections, [feature(.10), feature(.70), feature(.10)],
        partial_features=[feature(.10), feature(.70), None],
        partial_feature_sources=['osnet_torso', 'osnet_torso', None],
        image_width=640, image_height=480)


def test_detector_probe_counts_every_competing_person_not_only_its_one_eligible_crop():
    tracker = search_tracker()
    for index, expected in enumerate((0, 1)):
        result = detector_probe(tracker, index)
        assert result is not None
        assert result.reid_uid == expected
        observation = tracker.last_identity_observations[-1]
        sample = observation['sample_metadata']
        assert sample['candidate_count'] == 2
        assert sample['identity_competition']['candidate_count'] == 2
        assert sample['identity_competition']['passed']
        assert not observation['assignment']['bank_updated']


def test_detector_probe_does_not_trust_a_falsified_count_from_competition_proof(monkeypatch):
    tracker = search_tracker()
    produce = tracker._frame_identity_competition

    def bad_count(*args, **kwargs):
        proof = produce(*args, **kwargs)
        proof[0]['candidate_count'] = 1
        return proof

    monkeypatch.setattr(tracker, '_frame_identity_competition', bad_count)
    for index in range(3):
        result = detector_probe(tracker, index)
        assert result.reid_uid == 0
        sample = tracker.last_identity_observations[-1]['sample_metadata']
        assert sample['candidate_count'] == 2
        assert sample['identity_competition']['candidate_count'] == 1
    assert not tracker.identity_bank.pending_late_handoffs


def test_single_candidate_without_a_proof_retains_the_existing_compatibility_contract():
    assert IdentityBank._reacquire_competition(1, 2, 1, {}) == (True, 'single_candidate')
    assert IdentityBank._reacquire_competition(1, 2, 2, {}) == (False, 'missing')
