"""Follow-only CAP1040/candidate-bridge policy; synthetic embeddings, no hardware.

CAP1078 was discarded before the identity bank. Its recorded rounded detector
box and *processing-time* yaw reproduce only a conditional geometry experiment,
not a recovered embedding or capture-aligned yaw measurement.
"""
from copy import deepcopy

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.similar_follow import candidate_motion_geometry
from test_cap874_identity_reacquire import metadata, feature
from tools.replay_cap334_recovery import gallery_snapshot


CROP1040 = (0., 3.480804443, 195.29133606, 475.88562012)
CROP1047 = (0., 4.169082642, 107.89083099, 473.28173828)
BOX1074 = (54.36293793, 4.49131775, 265.66314697, 477.59680176)
BOX1078 = (227.7, 2.6, 389.7, 406.9)
BOX1084 = (421.23187256, 21.33450317, 639.47979736, 472.77203369)
TIME1074, TIME1078, TIME1084 = 16078.070609859, 16078.306017, 16078.603474707


def make_bank():
    return IdentityBank(IdentityBankConfig(
        similar_follow_enable=True, template_memory_enable=True,
        template_crosscheck_enable=True, template_learning_guard_enable=True,
        new_identity_confirm_frames=1, controlled_handoff_enable=True, min_confidence=.60,
        camera_hfov_deg=60.,
        update_interval=1, mapped_verify_threshold=.45,
        partial_match_threshold=.40, partial_confirm_threshold=.40))


def observe(bank, cap, stamp, box, *, frame=None, score=.95, full=0., partial=None,
            crop=False, search=False, opposite=False, yaw=0., extra=None, track=16):
    m = metadata(cap, stamp, box, yaw=yaw, search=search, opposite=opposite)
    m.update(image_width=640, image_height=480, detector_confidence=score,
             partial_feature_source='osnet_torso')
    if crop:
        m.update(quality_bbox_ok=False, bbox_quality_tier='weak',
                 quality_bbox_reason='edge_touch>2', bbox_quality_reason='edge_touch>2',
                 detector_edge_touch_count=3, edge_touch_count=3)
    m.update(extra or {})
    return bank.assign(track_id=track, feature=None if full is None else feature(full),
        partial_feature=None if partial is None else feature(partial), confidence=score,
        area=(box[2]-box[0])*(box[3]-box[1]), frame_index=cap if frame is None else frame,
        bbox_quality_ok=m['quality_bbox_ok'], bbox_quality_tier=m['bbox_quality_tier'],
        bbox_quality_reason=m.get('quality_bbox_reason', ''), sample_metadata=m,
        preferred_uid=1 if search else None, preferred_candidate_ok=search and not opposite)


def ordinary_bank():
    bank = make_bank()
    assert observe(bank, 1035, 100., (20., 4., 220., 476.), frame=452, partial=0.) == 1
    return bank


def test_ordinary_verified_target_enters_same_follow_only_crop_policy():
    bank = ordinary_bank()
    gallery = gallery_snapshot(bank)
    anchor = deepcopy(bank.identities[1].last_strong_observation)
    assert observe(bank, 1040, 100.2, CROP1040, frame=454,
                   score=.8853573, full=.1541166, crop=True) == 1
    result = bank.last_assignments[16]
    assert result['reason'] == 'mapped_similar_follow'
    assert result['similar_follow']['mapped_crop_origin_cap'] == 1035
    assert result['similar_follow']['crop_continuation']
    assert result['reacquire_geometry_ok'] and not result['bank_updated']
    assert bank.identities[1].last_strong_observation == anchor
    assert gallery_snapshot(bank) == gallery
    # Visible area shrinks to .5486, but unchanged full-height SAME-side crop
    # demonstrates visibility loss. Do not globally lower the area threshold.
    assert observe(bank, 1047, 100.560008557, CROP1047, frame=457,
                   score=.6894485, full=.3015881, crop=True, yaw=-4.109) == 1
    assert bank.last_assignments[16]['reacquire_geometry']['crop_visible_area_similarity'] < .55
    assert bank._similar_follow_states[(1, 16)]['last_strong_timestamp'] == 100.
    assert observe(bank, 1050, 100.76, CROP1047, frame=460,
                   score=.689, full=.30, crop=True, yaw=-4.109) == 0
    assert gallery_snapshot(bank) == gallery


@pytest.mark.parametrize('failure', ['appearance', 'competition', 'exclusion', 'jump',
                                    'sliver', 'stale', 'expired', 'quarantine', 'conflict'])
def test_ordinary_crop_bridge_does_not_turn_track_id_into_proof(monkeypatch, failure):
    bank = ordinary_bank()
    before = gallery_snapshot(bank)
    kw = dict(full=.154, crop=True, frame=454)
    box, stamp = CROP1040, 100.2
    if failure == 'appearance': kw['full'] = .60
    elif failure == 'competition':
        kw['extra'] = dict(source_detection_index=0, identity_competition=dict(uid=1,
            frame_index=454, candidate_count=1, source_detection_index=0, passed=False))
    elif failure == 'exclusion':
        monkeypatch.setattr(bank, 'search_exclusion_for', lambda *a, **k: {'reason': 'different_person'})
    elif failure == 'jump': box = (450., 0., 639., 479.)
    elif failure == 'sliver': box = (0., 0., 50., 479.)
    elif failure == 'stale': kw['extra'] = dict(is_fresh=False)
    elif failure == 'expired': stamp = 100.8
    elif failure == 'quarantine': bank._reacquire_quarantine.arm(1, 16, 1035, 100., 452)
    else: bank._mapped_geometry_conflicts[16] = dict(uid=1, search_contradiction=True)
    assert observe(bank, 1040, stamp, box, **kw) == 0
    assert gallery_snapshot(bank) == before


def pending_bank():
    bank = make_bank()
    assert observe(bank, 1, 1., BOX1074, track=1, partial=0.) == 1
    assert observe(bank, 1074, TIME1074, BOX1074, frame=474, score=.6254,
                   full=.2152673, search=True, yaw=-180.96460148135935) == 0
    assert bank.last_assignments[16]['similar_follow']['count'] == 1
    return bank


def observe1078(bank, **changes):
    options = dict(frame=477, score=.3278, full=.23, search=True, yaw=-185.949)
    extra = dict(low_score_continuation=True, association_reason='low_score_existing_track',
        association_previous_capture_frame_id=1074,
        association_previous_capture_timestamp=TIME1074)
    extra.update(changes.pop('extra', {}))
    options.update(changes)
    return observe(bank, 1078, TIME1078, BOX1078, extra=extra, **options)


def test_low_score_position_does_not_confirm_identity_or_refresh_qualified_capture():
    bank = pending_bank()
    gallery = gallery_snapshot(bank)
    assert observe1078(bank) == 0
    state = bank._similar_follow_states[(1, 16)]
    assert state['count'] == 1 and not state['active']
    assert state['last_cap'] == 1078 and state['last_qualified_cap'] == 1074
    assert state['last_qualified_timestamp'] == TIME1074
    assert bank.last_assignments[16]['similar_follow']['reason'] == 'low_score_position_only'
    assert 16 not in bank.track_to_uid
    assert gallery_snapshot(bank) == gallery


def test_conditional_cap1074_1078_1084_motion_policy_keeps_direction_provenance():
    bank = pending_bank()
    gallery = gallery_snapshot(bank)
    assert observe1078(bank) == 0
    # Same real detection sequence, synthetic compatible embeddings; CAP1078
    # capture-time yaw is not saved, so this is NOT full historical replay.
    assert observe(bank, 1084, TIME1084, BOX1084, frame=481, score=.70075,
        full=.2326126, search=True, opposite=True, yaw=-189.85429946853276) == 1
    detail = bank.last_assignments[16]
    predicted = detail['similar_follow_motion_geometry']
    assert predicted['reference_caps'] == [1074, 1078]
    assert predicted['measured_center_jump'] > .20
    assert predicted['residual'] < .10
    assert detail['similar_follow']['count'] == 2
    assert not detail['bank_updated'] and gallery_snapshot(bank) == gallery
    assert bank._reacquire_quarantine.is_held(1)


@pytest.mark.parametrize('yaw_error', [-2., -1., 0., 1., 2.])
def test_conditional_motion_bridge_tolerates_small_midframe_yaw_variation(yaw_error):
    bank = pending_bank()
    assert observe1078(bank, yaw=-185.949+yaw_error) == 0
    assert observe(bank, 1084, TIME1084, BOX1084, frame=481, full=.2326,
        search=True, opposite=True, yaw=-189.85429946853276) == 1
    assert bank.last_assignments[16]['similar_follow_motion_geometry']['residual'] <= .20


def test_predicted_intermediate_box_keeps_but_does_not_renew_unbound_candidate():
    bank = pending_bank()
    before = deepcopy(bank._similar_follow_states[(1, 16)])
    assert observe(bank, 1076, TIME1074+.1, BOX1074, frame=476, full=None,
        search=True, extra={'is_fresh': False}) == 0
    assert bank._similar_follow_states[(1, 16)] == before
    assert observe1078(bank) == 0


def test_low_observation_cannot_erase_or_renew_existing_appearance_memory():
    bank = ordinary_bank()
    # Keep a real ordinary qualified torso proof with a bounded capture clock.
    assert observe(bank, 1036, 100.1, (20., 4., 220., 476.), frame=453, partial=0.) == 1
    prior = deepcopy(bank._appearance_verified[1])
    assert observe(bank, 1038, 100.2, (22., 4., 222., 476.), frame=454,
        score=.4, extra=dict(low_score_continuation=True,
            association_reason='low_score_existing_track', association_confidence_limit=.5,
            association_previous_capture_frame_id=1036,
            association_previous_capture_timestamp=100.1)) == 0
    assert bank.last_assignments[16]['reason'] == 'low_score_observation_only'
    assert bank._appearance_verified[1] == prior


def test_low_score_uses_producer_confidence_boundary_not_another_hardcoded_half():
    bank = pending_bank()
    assert observe1078(bank, score=.55, extra={'association_confidence_limit': .60}) == 0
    assert bank.last_assignments[16]['similar_follow']['reason'] == 'low_score_position_only'
    assert bank._similar_follow_states[(1, 16)]['count'] == 1


@pytest.mark.parametrize('failure', ['score', 'missing_feature', 'foreign_origin', 'competition',
                                    'prediction', 'unknown_yaw', 'appearance', 'expired', 'conflict'])
def test_position_bridge_never_bypasses_independent_checks(failure):
    bank = pending_bank()
    gallery = gallery_snapshot(bank)
    if failure in ('score', 'missing_feature', 'foreign_origin', 'competition'):
        changes = {}
        if failure == 'score': changes['score'] = .20
        elif failure == 'missing_feature': changes['full'] = None
        elif failure == 'foreign_origin': changes['extra'] = dict(association_previous_capture_frame_id=1072)
        else: changes['extra'] = dict(source_detection_index=0, identity_competition=dict(
            uid=1, frame_index=477, candidate_count=1, source_detection_index=0, passed=False))
        assert observe1078(bank, **changes) == 0
        assert (1, 16) not in bank._similar_follow_states
    else:
        assert observe1078(bank) == 0
        box, ts, yaw, full = BOX1084, TIME1084, -189.85429946853276, .2326
        if failure == 'prediction': yaw = -180.
        elif failure == 'unknown_yaw': yaw = None
        elif failure == 'appearance': full = .70
        elif failure == 'expired': ts = TIME1074 + .76
        else: bank._mapped_geometry_conflicts[16] = dict(uid=1, search_contradiction=True, rejected_frame=474)
        assert observe(bank, 1084, ts, box, frame=481, full=full,
                       search=True, opposite=True, yaw=yaw) == 0
    assert gallery_snapshot(bank) == gallery


def test_missing_low_score_bridge_cannot_cross_screen_on_raw_id_alone():
    bank = pending_bank()
    assert observe(bank, 1084, TIME1084, BOX1084, frame=481, full=.2326,
                   search=True, opposite=True, yaw=-189.85429946853276) == 0
    assert bank.last_assignments[16]['similar_follow']['reason'] == 'entry_direction_unverified'


def test_low_score_cannot_seed_identity_even_with_permissive_ordinary_configuration():
    bank = make_bank()
    assert observe1078(bank) == 0
    assert not bank.identities and not bank._similar_follow_states


@pytest.mark.parametrize('state', [None, {}, {'position_only': True},
                                  {'position_only': True, 'observation': {}}])
def test_motion_evidence_requires_well_formed_bounded_state(state):
    assert candidate_motion_geometry({}, state, {'ok': False}, camera_hfov_deg=60.) is None
