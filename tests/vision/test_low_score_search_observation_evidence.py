"""Low-score conflict diagnostics do not grant UID, counts or gallery writes.

The normal/reliable-torso cases exercise assign's real evidence calculation.
The policy-unit cases additionally inject existing bank contradiction records
to reach the shared low-score branch, before assign's earlier geometry veto.
"""
from copy import deepcopy
from dataclasses import replace

import pytest

from car_control_modular.search_observation_retry import side_crop_search_observation
from test_cap1040_identity_continuity import make_bank, observe
from test_cap874_identity_reacquire import feature, metadata
from tools.replay_cap334_recovery import gallery_snapshot


BOX = (0., 10., 200., 470.)


def bank_and_metadata():
    bank = make_bank()
    assert observe(bank, 1, 1., BOX, track=1, partial=0.) == 1
    m = dict(metadata(3, 1.2, BOX, search=True), track_id=15,
        frame_index=3, control_frame_id=3, image_width=640, image_height=480,
        source_detection_index=0, detector_confidence=.433, partial_feature_source='osnet_torso',
        low_score_continuation=True, association_reason='low_score_existing_track',
        association_previous_capture_frame_id=2, association_previous_capture_timestamp=1.1,
        association_confidence_limit=.5,
        identity_competition=dict(uid=1, frame_index=3, candidate_count=1,
            source_detection_index=0, passed=True))
    return bank, m


@pytest.mark.parametrize('conflict', ['none', 'mapped_geometry', 'revoked_uid', 'reliable_partial'])
def test_shared_low_score_reason_preserves_actual_conflict_diagnostic(conflict):
    bank, m = bank_and_metadata()
    if conflict == 'mapped_geometry':
        bank._mapped_geometry_conflicts[15] = dict(uid=1, search_contradiction=True, rejected_frame=2)
    elif conflict == 'revoked_uid':
        bank._geometry_revoked_uids[1] = dict(rejected_frame=2, rejected_capture=2)
    gallery = gallery_snapshot(bank)
    counts = deepcopy(bank._similar_follow_states)
    mappings = deepcopy(bank.track_to_uid)
    protected = deepcopy(bank.identities[1].last_strong_observation)
    diagnostics = {}
    uid = bank._evaluate_similar_follow(uid=0, preferred_uid=1, track_id=15,
        feature=feature(.232), partial_feature=feature(.8 if conflict == 'reliable_partial' else .1),
        metadata=m, frame_index=3, candidate_count=1, confidence=.433,
        area=92000., diagnostics=diagnostics)
    assert uid == 0
    assert bank.last_assignments[15]['reason'] == 'low_score_observation_rejected'
    assert diagnostics['low_score_observation_blocked'] is (conflict != 'none')
    assert diagnostics['identity_control_rejected']
    assert not bank.last_assignments[15]['bank_updated']
    assert gallery_snapshot(bank) == gallery
    assert bank._similar_follow_states == counts and bank.track_to_uid == mappings
    assert bank.identities[1].last_strong_observation == protected


@pytest.mark.parametrize('partial,blocked', [(.1, False), (.8, True)])
def test_real_assign_exports_diagnostic_without_changing_uid_or_learning(partial, blocked):
    bank, m = bank_and_metadata()
    gallery, mapping = gallery_snapshot(bank), deepcopy(bank.track_to_uid)
    assert observe(bank, 3, 1.2, BOX, track=15, score=.433, full=.232,
                   partial=partial, search=True, extra=m) == 0
    result = bank.last_assignments[15]
    assert result['reason'] == 'low_score_observation_rejected'
    assert result['low_score_observation_blocked'] is blocked
    assert result['identity_control_rejected']
    assert result['match_evidence']['matched_uid'] == 1
    assert result['match_evidence']['strong_distance'] == pytest.approx(.232)
    assert result['partial_distance'] == pytest.approx(partial)
    assert side_crop_search_observation(result, m, 0, 1, BOX, 640, 'left') is (not blocked)
    assert not result['bank_updated'] and not result['recent_bank_updated']
    assert not result['learning_written_tiers']
    assert gallery_snapshot(bank) == gallery and bank.track_to_uid == mapping
    assert not bank._similar_follow_states


@pytest.mark.parametrize('partial', [.1, .8])
def test_fallback_without_full_conflict_review_does_not_publish_false(partial):
    bank, m = bank_and_metadata()
    bank.config = replace(bank.config, similar_follow_enable=False)
    gallery, mapping = gallery_snapshot(bank), deepcopy(bank.track_to_uid)
    assert observe(bank, 3, 1.2, BOX, track=15, score=.433, full=.232,
                   partial=partial, search=True, extra=m) == 0
    result = bank.last_assignments[15]
    assert result['reason'] == 'low_score_observation_only'
    assert 'low_score_observation_blocked' not in result
    assert result['identity_control_rejected'] and not result['bank_updated']
    assert not result['recent_bank_updated'] and not result['learning_written_tiers']
    assert gallery_snapshot(bank) == gallery and bank.track_to_uid == mapping
