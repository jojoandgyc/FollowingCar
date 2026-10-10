"""Follow-only observations must not become a rolling trusted feature gallery."""

from copy import deepcopy

import numpy as np
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from test_reacquire_crosscheck import feature, meta, send
from tools.replay_cap334_recovery import gallery_snapshot


def make_bank():
    bank = IdentityBank(IdentityBankConfig(
        template_memory_enable=True, template_crosscheck_enable=True,
        template_learning_guard_enable=True, similar_follow_enable=True,
        new_identity_confirm_frames=1, controlled_handoff_enable=True,
        update_interval=1, mapped_verify_threshold=.45,
        partial_match_threshold=.40, partial_confirm_threshold=.40))
    assert send(bank, 1, 1., extra={'image_width': 640, 'image_height': 480}) == 1
    return bank


def observe_reference(bank, cap=2, timestamp=2., extra=None):
    assert send(bank, cap, timestamp, full=.35, part=.334, extra=extra) == 1
    assert bank.last_assignments[1]['reason'] == 'skip_update_distance'
    return bank._follow_references[1][-1]


def activate(bank, *, full=.44, timestamp=3.):
    assert send(bank, 20, timestamp, full=full, part=None, track=3, search=True) == 0
    assert bank.last_assignments[3]['reason'] == 'similar_follow_observe'
    assert send(bank, 21, timestamp+.1, full=full, part=None, track=3, search=True) == 1
    assert bank.last_assignments[3]['reason'] == 'similar_follow_reacquire'


def test_normally_verified_skip_update_can_support_follow_without_gallery_write():
    bank = make_bank()
    before = gallery_snapshot(bank)
    row = observe_reference(bank)
    assert row['cap'] == 2
    assert gallery_snapshot(bank) == before
    assert bank.last_assignments[1]['follow_reference_updated']['permission'] == 'follow_only'
    # .65 fails both gallery thresholds; only the separately marked normal
    # tracking reference supports this deliberately weaker follow permission.
    activate(bank, full=.65)
    detail = bank.last_assignments[3]['similar_follow']
    assert detail['reference_cap'] == 2
    assert detail['gallery_distance'] == pytest.approx(.65, abs=1e-6)
    assert detail['full_distance'] < .50
    assert gallery_snapshot(bank) == before


def test_references_are_not_template_parents_or_template_evidence():
    bank = make_bank()
    observe_reference(bank)
    entry = bank.identities[1]
    assert {p.capture_frame_id for p in bank._learning_pairs(entry)} == {1}
    assert {m['capture_frame_id'] for m in bank._learning_source_metadata(entry)} == {1}
    assert entry.distance(feature(.65)) == pytest.approx(.65, abs=1e-6)
    activate(bank, full=.65)
    assert not bank._maybe_add_to_identity(1, feature(.65), feature(.65), 22, .1,
        {**meta(22, 3.2), 'track_id': 3})
    assert {p.capture_frame_id for p in bank._learning_pairs(entry)} == {1}


def test_soft_follow_never_appends_or_renews_normal_references():
    bank = make_bank()
    observe_reference(bank)
    original = deepcopy(bank._follow_references[1])
    activate(bank, full=.65)
    for cap in range(22, 40):
        assert send(bank, cap, 3.1+.1*(cap-21), full=.65, part=None, track=3) == 1
        assert bank.last_assignments[3]['reason'] == 'mapped_similar_follow'
        assert not bank.last_assignments[3]['bank_updated']
    rows = bank._follow_references[1]
    assert len(rows) == len(original) == 1
    assert rows[0]['timestamp'] == original[0]['timestamp'] == 2.
    assert rows[0]['cap'] == 2
    np.testing.assert_array_equal(rows[0]['feature'], original[0]['feature'])


def test_reference_expiry_cannot_be_renewed_by_soft_follow():
    bank = make_bank()
    observe_reference(bank, timestamp=2.)
    activate(bank, full=.65, timestamp=9.9)
    assert bank.last_assignments[3]['similar_follow']['reference_cap'] == 2
    assert send(bank, 22, 10.1, full=.65, part=None, track=3) == 0
    detail = bank.last_assignments[3]['similar_follow']
    assert detail['status'] == 'reject'
    assert detail['reference_cap'] is None
    assert not bank._follow_references.get(1)
    assert 1 in bank._similar_learning_fences


def test_reference_capacity_is_six_not_a_second_unbounded_gallery():
    bank = make_bank()
    before = gallery_snapshot(bank)
    for cap in range(2, 10):
        observe_reference(bank, cap, 1.+.1*cap)
    assert [r['cap'] for r in bank._follow_references[1]] == list(range(4, 10))
    assert gallery_snapshot(bank) == before


def test_reference_feature_and_nested_metadata_are_independent_snapshots():
    bank = make_bank()
    value = feature(.35)
    metadata = {**meta(2, 2.), 'track_id': 1, 'frame_index': 2,
                'nested': {'values': [1]}}
    diagnostic = {'reacquire_geometry_ok': True}
    bank._remember_follow_reference(1, 1, value, metadata,
        {'reason': 'skip_update_distance', 'match_source': 'strong'}, diagnostic)
    stored = bank._follow_references[1][-1]
    old_feature = stored['feature'].copy()
    old_bbox = list(metadata['detector_bbox'])
    value[:] = 0.
    metadata['detector_bbox'][0] = -999.
    metadata['nested']['values'].append(2)
    np.testing.assert_array_equal(stored['feature'], old_feature)
    assert stored['metadata']['detector_bbox'] == old_bbox
    assert stored['metadata']['nested']['values'] == [1]


def test_duplicate_normal_observation_cannot_refresh_reference_timestamp():
    bank = make_bank()
    row = observe_reference(bank)
    before = row['timestamp']
    diagnostic = {'reacquire_geometry_ok': True}
    bank._remember_follow_reference(1, 1, feature(.35),
        {**meta(2, 2.1), 'track_id': 1, 'frame_index': 3},
        {'reason': 'skip_update_distance', 'match_source': 'strong'}, diagnostic)
    assert len(bank._follow_references[1]) == 1
    assert bank._follow_references[1][0]['timestamp'] == before
    assert 'follow_reference_updated' not in diagnostic


def test_reset_clears_reference_follow_state_and_learning_fence():
    bank = make_bank()
    observe_reference(bank)
    activate(bank, full=.65)
    assert bank._follow_references and bank._similar_follow_states and bank._similar_learning_fences
    bank.reset()
    assert not bank._follow_references
    assert not bank._similar_follow_states
    assert not bank._similar_learning_fences


def test_active_candidate_crossing_survives_leaving_search_without_new_learning():
    bank = make_bank()
    before = gallery_snapshot(bank)
    activate(bank)
    for cap in range(22, 30):
        assert send(bank, cap, 3.1+.1*(cap-21), full=.513, part=None, track=3,
                    extra={'search_direction_compatible': False}) == 1
        result = bank.last_assignments[3]
        assert result['reason'] == 'mapped_similar_follow'
        assert result['similar_follow']['status'] == 'follow'
        assert not result['bank_updated']
    assert gallery_snapshot(bank) == before


@pytest.mark.parametrize('failure', ['quality', 'competition', 'explicit_geometry', 'exclusion'])
def test_active_similarity_cannot_override_current_negative_evidence(monkeypatch, failure):
    bank = make_bank()
    activate(bank)
    before = gallery_snapshot(bank)
    extra = {}
    if failure == 'quality':
        extra['quality_bbox_ok'] = False
    elif failure == 'competition':
        extra['identity_competition'] = dict(uid=1, frame_index=22, candidate_count=1,
                                             passed=False, reason='ambiguous')
    elif failure == 'explicit_geometry':
        extra['quality_bbox_reason'] = 'identity_swap_competing_track'
    else:
        monkeypatch.setattr(bank, 'search_exclusion_for', lambda *a, **k:
                            {'reason': 'previously_covisisible_different_person'})
    assert send(bank, 22, 3.2, full=.44, part=None, track=3, extra=extra) == 0
    assert (1, 3) not in bank._similar_follow_states
    assert 1 in bank._similar_learning_fences
    assert gallery_snapshot(bank) == before


def test_one_strong_frame_after_uid_zero_cannot_erase_learning_fence():
    bank = make_bank()
    activate(bank)
    before = gallery_snapshot(bank)
    assert send(bank, 22, 3.2, full=.44, part=None, track=3,
                extra={'quality_bbox_ok': False}) == 0
    send(bank, 23, 3.3, full=.1, part=.1, track=3)
    assert 1 in bank._similar_learning_fences
    assert gallery_snapshot(bank) == before
    assert not bank.last_assignments[3]['bank_updated']


def test_independent_existing_pair_commits_on_release_without_stopping_follow():
    bank = make_bank()
    activate(bank)
    before = gallery_snapshot(bank)
    quality = {'image_width': 640, 'image_height': 480,
               'template_learning_risk': {'observed': True, 'risky': False, 'reason': 'clear'}}
    for cap, timestamp in ((22, 3.35), (23, 3.6), (24, 3.85), (25, 4.1)):
        assert send(bank, cap, timestamp, full=.1, part=.1, track=3, extra=quality) == 1
        assert bank.last_assignments[3]['reason'] == 'mapped_similar_follow'
        if cap < 25:
            assert not bank.last_assignments[3]['bank_updated']
            assert gallery_snapshot(bank) == before
            assert 1 in bank._similar_learning_fences
        else:
            assert bank.last_assignments[3]['bank_updated']
            assert bank.last_assignments[3]['learning_written_tiers'] == ['recent_strong', 'recent_partial']
            assert bank.last_assignments[3]['template_learning']['parent_caps'] == [1]
    assert 1 not in bank._similar_learning_fences
    assert not bank._reacquire_quarantine.is_held(1)
    assert bank.last_assignments[3]['similar_follow']['independent_verification_recovered']
    assert not bank._follow_references.get(1)
    for cap, timestamp in ((26, 4.2), (27, 4.3)):
        assert send(bank, cap, timestamp, full=.1, part=.1, track=3, extra=quality) == 1
    assert bank.last_assignments[3]['reason'] != 'mapped_similar_follow'
    assert 27 in {m['capture_frame_id'] for m in bank._learning_source_metadata(bank.identities[1])}


def test_search_flag_lag_after_regional_release_never_restarts_uid_zero_observation():
    bank = make_bank()
    activate(bank)
    before = gallery_snapshot(bank)
    quality = {'image_width': 640, 'image_height': 480,
               'template_learning_risk': {'observed': True, 'risky': False, 'reason': 'clear'}}
    # Five paired .25/.20 observations over one second qualify the existing
    # regional review, without requiring the .20 full-only recovery entrance.
    for cap in range(22, 27):
        assert send(bank, cap, 3.35+.25*(cap-22), full=.25, part=.20,
                    track=3, search=True, extra=quality) == 1
        assert not bank.last_assignments[3]['bank_updated']
        assert gallery_snapshot(bank) == before
    assert bank.last_assignments[3]['similar_follow']['independent_verification_recovered']
    assert bank._similar_follow_states[(1, 3)]['independent_verified']
    # The control/search owner can receive identity updates later. Its old
    # search=True must not convert already accepted tracking into a new seed.
    for cap, timestamp in ((27, 4.45), (28, 4.55)):
        assert send(bank, cap, timestamp, full=.25, part=.20,
                    track=3, search=True, extra=quality) == 1
        assert not bank.last_assignments[3]['bank_updated']
        assert gallery_snapshot(bank) == before
        assert bank._similar_follow_states[(1, 3)]['independent_verified']
        assert not bank._reacquire_quarantine.is_held(1)
    assert send(bank, 29, 4.65, full=.25, part=.20, track=3, extra=quality) == 1
    assert bank.last_assignments[3]['reason'] != 'mapped_similar_follow'
    assert (1, 3) not in bank._similar_follow_states
    assert not bank.last_assignments[3]['bank_updated']


def test_elapsed_time_and_similarity_to_follow_reference_cannot_release_learning():
    bank = make_bank()
    observe_reference(bank)
    before = gallery_snapshot(bank)
    activate(bank, full=.65)
    for cap in range(22, 37):
        assert send(bank, cap, 3.1+.2*(cap-21), full=.65, part=.41, track=3,
                    extra={'image_width': 640, 'image_height': 480}) == 1
        result = bank.last_assignments[3]
        assert result['similar_follow']['reference_cap'] == 2
        assert result['similar_follow']['full_distance'] < .20
        assert 1 in bank._similar_learning_fences
        assert bank._reacquire_quarantine.is_held(1)
        assert not result['bank_updated']
        assert gallery_snapshot(bank) == before


def test_out_of_order_timestamp_does_not_delete_newer_reference():
    bank = make_bank()
    observe_reference(bank, timestamp=2.)
    bank._remember_follow_reference(0, 1, None, meta(1, 1.5), {}, {})
    assert bank._follow_references[1][0]['timestamp'] == 2.


@pytest.mark.parametrize('block', ['quarantine', 'learning_fence', 'pose_fence', 'suspect',
                                  'search', 'weak', 'competition', 'no_geometry'])
def test_unqualified_normal_claim_cannot_create_follow_reference(block):
    bank = make_bank()
    metadata = {**meta(2, 2.), 'track_id': 1, 'frame_index': 2}
    diagnostic = {'reacquire_geometry_ok': True}
    if block == 'quarantine':
        bank._reacquire_quarantine.arm(1, 1, 1, 1., 1)
    elif block == 'learning_fence':
        bank._similar_learning_fences.add(1)
    elif block == 'pose_fence':
        bank._pose_learning_fences[1] = {}
    elif block == 'suspect':
        bank._reacquire_control_suspects[1] = {}
    elif block == 'search':
        metadata['search_reacquire_context_active'] = True
    elif block == 'weak':
        metadata['bbox_quality_tier'] = 'weak'
    elif block == 'competition':
        metadata['identity_competition'] = dict(uid=1, frame_index=2, candidate_count=1, passed=False)
    elif block == 'no_geometry':
        diagnostic['reacquire_geometry_ok'] = None
    bank._remember_follow_reference(1, 1, feature(.35), metadata,
        {'reason': 'skip_update_distance', 'match_source': 'strong'}, diagnostic)
    assert not bank._follow_references
