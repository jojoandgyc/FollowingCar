"""CAP1161->1166's saved geometry/time, synthetic descriptor policy tests.

This is not a motor replay. The checkpoint is an already confirmed partial
handoff, as logged; no candidate may manufacture that internal provenance.
"""
from copy import deepcopy
from dataclasses import replace

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, _geometry_observation
from test_cap874_identity_reacquire import feature, metadata
from test_cap225_pose_continuity import gallery


ROWS = {
    944: (285, 29135.924107658, (238.3879089, 17.5693970, 474.7597961, 476.0383301), 9.9083798),
    1161: (420, 29147.210613977, (296.6357727, 112.7681427, 425.6234741, 478.0936279), 146.5000581),
    1166: (421, 29147.475808801, (316.3028564, 117.6648102, 432.4581909, 464.7012939), 146.0734049),
    1168: (422, 29147.574021901, (316.7655334, 121.9085999, 429.9288635, 467.0295715), 145.0538820),
}


def meta(cap, **changes):
    frame, stamp, box, yaw = ROWS[cap]
    m = dict(metadata(cap, stamp, box, yaw, search=False), frame_index=frame,
        control_frame_id=frame, track_id=2, image_width=640, image_height=480,
        partial_feature_source='osnet_torso', partial_observation=cap < 1166,
        detector_confidence=.92, source_detection_index=0)
    m.update(changes)
    x1, y1, x2, y2 = m['detector_bbox']
    m.update(bbox=list(m['detector_bbox']),
        center_x_ratio=(x1+x2)/1280., detector_center_x_ratio=(x1+x2)/1280.,
        area_ratio=(x2-x1)*(y2-y1)/(640*480), detector_area_ratio=(x2-x1)*(y2-y1)/(640*480))
    return m


def checkpoint():
    bank = IdentityBank(IdentityBankConfig(template_memory_enable=True,
        template_crosscheck_enable=True, appearance_region_safety_enable=True,
        partial_match_threshold=.45, partial_confirm_threshold=.40,
        mapped_verify_threshold=.45, controlled_handoff_enable=True,
        camera_hfov_deg=60., update_interval=1))
    bank._create_identity(feature(0), ROWS[944][0], meta(944, track_id=1), feature(0))
    m = meta(1161)
    bank._bind_reacquired_identity(1, 2, 420, m)
    bank.identities[1].last_strong_observation = _geometry_observation(m, 420)
    bank._remember_track_seen(2, 1, 420)
    bank._appearance_verified[1] = dict(metadata=m, comparable_caps=[944],
        comparison_mode='exact_coverage', pending_used=False,
        scale_started=m['capture_timestamp'], continuation_source='partial')
    return bank


def send(bank, cap=1166, *, full=.22, part=.31, **changes):
    m = meta(cap, **changes)
    x1, y1, x2, y2 = m['detector_bbox']
    return bank.assign(track_id=m['track_id'], feature=feature(full),
        partial_feature=None if part is None else feature(part),
        confidence=m['detector_confidence'], area=(x2-x1)*(y2-y1),
        frame_index=m['frame_index'], candidate_count=m['candidate_count'],
        bbox_quality_ok=m['quality_bbox_ok'], bbox_quality_tier=m['bbox_quality_tier'],
        sample_metadata=m)


def test_recorded_265ms_crop_transition_continues_known_source_without_learning():
    bank = checkpoint()
    before = gallery(bank)
    assert ROWS[1166][1]-ROWS[1161][1] == pytest.approx(.265194824)
    for cap in (1166, 1168):
        assert send(bank, cap) == 1
        a = bank.last_assignments[2]
        assert a['reason'] == 'skip_update_pose_retention'
        evidence = a['reacquire_recent_partial_evidence']
        assert evidence['comparison_mode'] == 'verified_pose_continuation'
        assert evidence['pose_bridge_caps'] == [944]
        assert not a['bank_updated']
        assert bank._appearance_verified[1]['pose_started'] == ROWS[1161][1]
        assert bank._reacquire_quarantine.is_held(1)
        assert gallery(bank) == before


@pytest.mark.parametrize('gap,allowed', [(.249, True), (.265194824, True), (.349, True), (.35, False), (.351, False)])
def test_confirmed_observation_budget_boundaries(gap, allowed):
    bank = checkpoint()
    assert send(bank, capture_timestamp=ROWS[1161][1]+gap) == int(allowed)


@pytest.mark.parametrize('config_gap,gap,allowed', [(.18,.20,False), (.25,.265,False),
    (.30,.265,True), (.30,.31,False), (.80,.36,False)])
def test_existing_config_can_tighten_but_not_extend_350ms(config_gap, gap, allowed):
    bank = checkpoint()
    bank.config = replace(bank.config, preferred_search_reacquire_max_age_sec=config_gap)
    assert send(bank, capture_timestamp=ROWS[1161][1] + gap) == int(allowed)


@pytest.mark.parametrize('case', ['no_source', 'no_proof', 'pending', 'search', 'other_track',
    'other_uid', 'conflict', 'revoked', 'suspect', 'competition', 'crowd', 'stale', 'weak',
    'bad_quality', 'duplicate_cap', 'old_timestamp', 'missing_yaw', 'side_crop', 'tiny',
    'jump', 'bad_full', 'bad_partial', 'missing_partial', 'expired_template'])
def test_extended_clock_never_bootstraps_or_overrides_negative_evidence(case):
    bank = checkpoint()
    before = gallery(bank)
    changes = {}
    if case == 'no_source': bank._appearance_verified[1].pop('continuation_source')
    if case == 'no_proof': bank._appearance_verified.clear()
    if case == 'pending': bank._appearance_verified[1]['pending_continuation_deadline'] = ROWS[1161][1]+.35
    if case == 'search': changes.update(search_reacquire_context_active=True, search_direction_compatible=True)
    if case == 'other_track': changes['track_id'] = 3
    if case == 'other_uid': bank.track_to_uid[2] = 9
    if case == 'conflict': bank._mapped_geometry_conflicts[2] = dict(uid=1,
        search_contradiction=True, reference=deepcopy(bank.identities[1].last_strong_observation))
    if case == 'revoked': bank._geometry_revoked_uids[1] = {}
    if case == 'suspect': bank._reacquire_control_suspects[1] = dict(track_id=2,
        streak=0, reason='appearance_conflict', capture=None, timestamp=None)
    if case == 'competition': changes['identity_competition'] = dict(uid=1, frame_index=421,
        candidate_count=1, source_detection_index=0, passed=False)
    if case == 'crowd': changes['candidate_count'] = 2
    if case == 'stale': changes['is_fresh'] = False
    if case == 'weak': changes['bbox_quality_tier'] = 'weak'
    if case == 'bad_quality': changes['quality_bbox_ok'] = False
    if case == 'duplicate_cap': changes['capture_frame_id'] = 1161
    if case == 'old_timestamp': changes['capture_timestamp'] = ROWS[1161][1]-.01
    if case == 'missing_yaw': changes['integrated_yaw_deg'] = None
    if case == 'side_crop': changes['detector_bbox'] = [0, 118, 116, 465]
    if case == 'tiny': changes['detector_bbox'] = [316, 260, 340, 340]
    if case == 'jump': changes['detector_bbox'] = [30, 118, 146, 465]
    if case == 'bad_full': changes['full'] = .31
    if case == 'bad_partial': changes['part'] = .46
    if case == 'missing_partial': changes['part'] = None
    if case == 'expired_template':
        bank.identities[1].template_memory.recent['partial'][0][1]['capture_timestamp'] = ROWS[1166][1]-30.1
        before = gallery(bank)
    assert send(bank, **changes) == 0
    assert not bank.last_assignments[changes.get('track_id', 2)]['bank_updated']
    if case == 'expired_template':
        assert bank.identities[1].template_memory.recent['partial'] == []
        assert gallery(bank)[-1] == before[-1]  # Pruning is not new learning.
    else:
        assert gallery(bank) == before


def test_source_bound_longer_frame_gap_does_not_renew_two_second_pose_epoch():
    bank = checkpoint()
    frozen = gallery(bank)
    for index in range(7):
        assert send(bank, capture_frame_id=1166+index, frame_index=421+index,
                    capture_timestamp=ROWS[1161][1]+.265*(index+1)) == 1
        assert bank._appearance_verified[1]['pose_started'] == ROWS[1161][1]
    assert send(bank, capture_frame_id=1174, frame_index=429,
                capture_timestamp=ROWS[1161][1]+2.01) == 0
    assert gallery(bank) == frozen


def test_independent_exact_coverage_can_end_pose_epoch_after_fixed_budget():
    bank = checkpoint()
    for index in range(7):
        assert send(bank, capture_frame_id=1166+index, frame_index=421+index,
                    capture_timestamp=ROWS[1161][1]+.265*(index+1)) == 1
    assert send(bank, capture_frame_id=1174, frame_index=429,
                capture_timestamp=ROWS[1161][1]+2.01,
                detector_bbox=ROWS[1161][2], part=.20, full=.10) == 1
    evidence = bank.last_assignments[2]['reacquire_recent_partial_evidence']
    assert evidence['comparison_mode'] == 'exact_coverage'
    assert evidence['pose_bridge_caps'] == []
    assert 'pose_started' not in bank._appearance_verified[1]


def test_incoming_metadata_cannot_supply_internal_continuation_source():
    bank = checkpoint()
    bank._appearance_verified[1].pop('continuation_source')
    assert send(bank, continuation_source='partial', pose_sample_gap_sec=.35) == 0
    assert not bank.last_assignments[2]['bank_updated']


@pytest.mark.parametrize('config_gap', [.18, .35])
def test_actual_pending_capture_cannot_recover_through_pose_fallback(config_gap):
    bank = checkpoint()
    bank.config = replace(bank.config, preferred_search_reacquire_max_age_sec=config_gap)
    frozen = gallery(bank)
    start = ROWS[1161][1]
    # A real independent pair produces a held observation; the accepted
    # reference remains CAP1161, with its original configured deadline.
    assert send(bank, capture_frame_id=1164, frame_index=421,
        capture_timestamp=start+.1, detector_bbox=ROWS[1161][2], part=.42) == 0
    assert bank.last_assignments[2]['reason'] == 'verified_continuation_recheck'
    assert bank._appearance_verified[1]['pending_continuation_deadline'] == pytest.approx(start+config_gap)
    assert send(bank, capture_frame_id=1166, frame_index=422,
        capture_timestamp=start+.2) == 0
    assignment = bank.last_assignments[2]
    assert assignment['identity_continuation']['status'] == 'reject'
    if config_gap == .18:
        assert assignment['identity_continuation']['reason'] == 'continuation_expired'
    assert assignment['reacquire_recent_partial_evidence']['pose_bridge_caps'] == []
    assert 1 not in bank._appearance_verified
    assert gallery(bank) == frozen


def test_actual_pending_capture_can_recover_with_independent_pair_before_deadline():
    bank = checkpoint()
    bank.config = replace(bank.config, preferred_search_reacquire_max_age_sec=.18)
    start = ROWS[1161][1]
    assert send(bank, capture_frame_id=1164, frame_index=421,
        capture_timestamp=start+.1, detector_bbox=ROWS[1161][2], part=.42) == 0
    assert send(bank, capture_frame_id=1166, frame_index=422,
        capture_timestamp=start+.15, detector_bbox=ROWS[1161][2], part=.31) == 1
    assignment = bank.last_assignments[2]
    assert assignment['identity_continuation']['status'] == 'accept'
    assert assignment['reacquire_recent_partial_evidence']['comparison_mode'] == 'exact_coverage'
    assert 'pending_continuation_deadline' not in bank._appearance_verified[1]
