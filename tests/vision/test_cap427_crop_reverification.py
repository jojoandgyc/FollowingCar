"""CAP427 crop interruption: recorded geometry/clocks, synthetic embeddings.

No camera, NPU or motor is used. Current embeddings reproduce the recorded
gallery distances; this is an identity-policy regression, not a model replay.
"""
from copy import deepcopy

import pytest

from rk_vision.identity_bank import IdentityBank, _geometry_observation
from rk_vision.similar_follow import crop_reverification_eligible, cropped_follow_continuous
from test_cap1040_identity_continuity import make_bank, observe
from test_cap874_identity_reacquire import metadata
from tools.replay_cap334_recovery import gallery_snapshot


ORIGIN = 22559.047491216
ROWS = (
    (414, 127, 22559.286036809, -4.766641381429591, .9456369876861572, .12329983711242676,
     (0.744842529296875, 2.873992919921875, 235.53317260742188, 477.034423828125)),
    (416, 128, 22559.383429172, -4.138592286171241, .9267995953559875, .12686002254486084,
     (0., 5.3374481201171875, 215.332275390625, 477.69317626953125)),
    (421, 129, 22559.648476104, -5.576924980188878, .9230321049690247, .19056999683380127,
     (.0092315673828125, 3.043365478515625, 183.07835388183594, 477.70465087890625)),
    (423, 130, 22559.747236058, -7.790680465233799, .9192646145820618, .2000563144683838,
     (.7343521118164062, 1.90069580078125, 171.20187377929688, 477.1392822265625)),
    (427, 131, 22559.982266572, -9.648815574694499, .9192646145820618, .2364572286605835,
     (0., 1.77923583984375, 144.15896606445312, 477.0238037109375)),
)


def send_row(bank, row=ROWS[-1], **overrides):
    cap, frame, stamp, yaw, score, distance, box = row
    opts = dict(frame=frame, yaw=yaw, score=score, full=distance, crop=True, track=1)
    opts.update(overrides)
    return observe(bank, cap, stamp, box, **opts)


def before427():
    bank = make_bank()
    assert observe(bank, 409, ORIGIN,
                   (18.218650817871094, 2.602935791015625, 295.65289306640625, 476.6002197265625),
                   frame=126, track=1, yaw=-5.539503253969925, partial=0.) == 1
    gallery, anchor = gallery_snapshot(bank), deepcopy(bank.identities[1].last_strong_observation)
    for row in ROWS[:-1]:
        assert send_row(bank, row) == 1
        assert not bank.last_assignments[1]['bank_updated']
    return bank, gallery, anchor


def test_recorded_cap427_passes_without_moving_trusted_anchor_or_learning():
    bank, gallery, anchor = before427()
    assert send_row(bank) == 1
    result = bank.last_assignments[1]
    assert result['reason'] == 'mapped_similar_follow'
    detail = result['similar_follow']
    assert detail['crop_current_reverified']
    assert detail['crop_gallery_distance'] == pytest.approx(.23645723, abs=1e-6)
    assert detail['crop_reference_age_ms'] == pytest.approx(934.775356)
    assert detail['crop_observation_gap_ms'] == pytest.approx(235.030514)
    assert not result['bank_updated'] and result['template_update_quarantined']
    assert bank.identities[1].last_strong_observation == anchor
    assert bank._similar_follow_states[(1, 1)]['last_strong_timestamp'] == ORIGIN
    assert gallery_snapshot(bank) == gallery


def test_fresh_crops_do_not_hit_a_new_fixed_total_time_cliff():
    bank, gallery, anchor = before427()
    assert send_row(bank) == 1
    cap, frame, stamp, yaw, score, distance, box = ROWS[-1]
    for index in range(1, 51):
        assert send_row(bank, (cap+index, frame+index, stamp+.1*index, yaw, score, distance, box)) == 1
    state = bank._similar_follow_states[(1, 1)]
    assert state['last_strong_timestamp'] == ORIGIN
    assert state['last_cap'] == cap+50
    assert bank.identities[1].last_strong_observation == anchor
    assert gallery_snapshot(bank) == gallery


def test_unknown_crop_appearance_preserves_old_observation_without_renewing_it():
    bank, gallery, anchor = before427()
    prior = deepcopy(bank._similar_follow_states[(1, 1)])
    assert send_row(bank, full=.31) == 0
    assert bank.last_assignments[1]['similar_follow']['observation_retained']
    assert bank._similar_follow_states[(1, 1)] == prior
    assert gallery_snapshot(bank) == gallery
    cap, frame, stamp, yaw, score, distance, box = ROWS[-1]
    assert send_row(bank, (cap+1, frame+1, stamp+.05, yaw, score, distance, box)) == 1
    assert bank.identities[1].last_strong_observation == anchor


@pytest.mark.parametrize('distance', [.56, .70])
def test_hard_appearance_failure_is_not_retained(distance):
    bank, gallery, _ = before427()
    assert send_row(bank, full=distance) == 0
    assert (1, 1) not in bank._similar_follow_states
    assert gallery_snapshot(bank) == gallery


def test_follow_reference_alone_cannot_supply_current_gallery_recheck():
    bank, gallery, _ = before427()
    from test_cap874_identity_reacquire import feature
    # Force a very attractive follow-only reference, while unchanged trusted
    # templates still disagree beyond .30. The reference cannot self-authorize.
    bank._follow_references[1] = [dict(cap=423, timestamp=ROWS[-2][2], feature=feature(.4))]
    assert send_row(bank, full=.4) == 0
    detail = bank.last_assignments[1]['similar_follow']
    assert detail['full_distance'] < .01 and detail['gallery_distance'] > .30
    assert not detail.get('crop_current_reverified')
    assert gallery_snapshot(bank) == gallery


@pytest.mark.parametrize('failure', ['competition', 'geometry_conflict', 'excluded'])
def test_current_identity_conflict_cannot_be_overridden_or_retained(monkeypatch, failure):
    bank, gallery, _ = before427()
    changes = {}
    if failure == 'competition':
        changes['extra'] = dict(source_detection_index=0, identity_competition=dict(
            uid=1, frame_index=131, source_detection_index=0, candidate_count=2,
            passed=False, reason='reid_margin_insufficient'))
    elif failure == 'geometry_conflict':
        bank._mapped_geometry_conflicts[1] = dict(uid=1, search_contradiction=True)
    else:
        monkeypatch.setattr(bank, 'search_exclusion_for', lambda *a, **k: {'reason': 'different_person'})
    assert send_row(bank, **changes) == 0
    assert not bank.last_assignments[1].get('similar_follow', {}).get('crop_current_reverified')
    assert gallery_snapshot(bank) == gallery


def test_reliable_comparable_torso_conflict_blocks_crop_without_renewing_memory(monkeypatch):
    bank, gallery, anchor = before427()
    prior = deepcopy(bank._similar_follow_states[(1, 1)])
    memory = bank.identities[1].template_memory
    original_evidence = memory.evidence
    verified_calls = []

    def independent_comparable_conflict(feature, current, tier='strong', **options):
        result = original_evidence(feature, current, tier, **options)
        if tier == 'partial' and options.get('reliable_only') and options.get('comparable_only'):
            assert feature is not None
            verified_calls.append(current['capture_frame_id'])
            # Deliberately inject the *output of a successful comparability
            # review*. Merely passing a bad torso vector with this three-edge
            # crop would test unavailable evidence, not reliable opposition.
            result = dict(result, count=1, distance=.75, comparable_count=1,
                          comparable_distance=.75, comparable_caps=[409], winner_cap=409,
                          winner_coverage=result['query_coverage'], comparison_mode='exact_coverage')
        return result

    monkeypatch.setattr(memory, 'evidence', independent_comparable_conflict)
    assert send_row(bank, partial=.75) == 0
    assert 427 in verified_calls
    assignment = bank.last_assignments[1]
    assert assignment['similar_follow']['reason'] == 'reliable_partial_conflict'
    assert not assignment['similar_follow'].get('crop_current_reverified')
    assert not assignment['similar_follow'].get('observation_retained')
    assert (1, 1) not in bank._similar_follow_states
    assert prior['last_cap'] == 423 and prior['last_timestamp'] == ROWS[-2][2]
    assert bank.identities[1].last_strong_observation == anchor
    assert not assignment['bank_updated'] and gallery_snapshot(bank) == gallery


def test_successful_crop_recheck_does_not_hide_next_frame_cross_side_conflict():
    bank, gallery, anchor = before427()
    assert send_row(bank) == 1
    assert bank.last_assignments[1]['similar_follow']['crop_current_reverified']
    prior = deepcopy(bank._similar_follow_states[(1, 1)])
    cap, frame, stamp, yaw, score, distance, _ = ROWS[-1]
    # Same raw ID and independently low appearance score, but a near-instant
    # jump to the opposite border without camera rotation is a hard conflict.
    opposite = (496., 1.77923583984375, 640., 477.0238037109375)
    current = dict(metadata(cap+1, stamp+.1, opposite, yaw=yaw), track_id=1,
                   image_width=640, image_height=480)
    geometry = bank._handoff_geometry(1, current, frame+1,
        reference_override=_geometry_observation(prior['observation'], frame))
    assert geometry['ok'] is False and geometry['yaw_compensated_center_jump_ratio'] > .60
    # Supply the newly established negative evidence through the existing
    # association/geometry owner, rather than relying on the older protected
    # CAP409 anchor (which intentionally was not rewritten by follow-only).
    bank._mapped_geometry_conflicts[1] = dict(uid=1, search_contradiction=True,
        reference=geometry['reference'], candidate=geometry['current'], rejected_capture=cap+1)
    assert send_row(bank, (cap+1, frame+1, stamp+.1, yaw, score, distance, opposite)) == 0
    assignment = bank.last_assignments[1]
    assert assignment['reason'] == 'mapped_geometry_reject'
    assert not assignment.get('similar_follow', {}).get('crop_current_reverified')
    assert (1, 1) not in bank._similar_follow_states
    assert prior['last_cap'] == 427 and prior['last_timestamp'] == stamp
    assert bank.identities[1].last_strong_observation == anchor
    assert not assignment['bank_updated'] and gallery_snapshot(bank) == gallery


def pure_input():
    bank, _, _ = before427()
    state = deepcopy(bank._similar_follow_states[(1, 1)])
    cap, frame, stamp, yaw, score, _, box = ROWS[-1]
    current = metadata(cap, stamp, box, yaw=yaw)
    current.update(track_id=1, image_width=640, image_height=480, detector_confidence=score,
                   quality_bbox_ok=False, bbox_quality_tier='weak',
                   quality_bbox_reason='edge_touch>2', bbox_quality_reason='edge_touch>2')
    geometry = IdentityBank()._handoff_geometry(1, current, frame,
        reference_override=_geometry_observation(state['observation'], frame-1))
    return current, state, geometry


def test_actual_geometry_only_old_time_budget_rejected_current_crop():
    current, state, geometry = pure_input()
    assert geometry['ok'] and geometry['area_similarity'] == pytest.approx(.845678866)
    assert geometry['yaw_compensated_center_jump_ratio'] == pytest.approx(.0423469299)
    assert not cropped_follow_continuous(current, state, geometry)
    assert crop_reverification_eligible(current, state, geometry)


@pytest.mark.parametrize('failure', ['duplicate', 'out_of_order', 'gap', 'low_score', 'prediction',
    'position_only', 'raw_changed', 'opposite_side', 'no_previous_crop', 'confidence',
    'missing_confidence', 'sliver', 'other_quality', 'image_resize', 'jump', 'scale',
    'geometry_unknown', 'unknown_dimensions'])
def test_current_crop_reverification_requires_every_new_observation_condition(failure):
    current, state, geometry = pure_input()
    if failure == 'duplicate': current['capture_frame_id'] = state['last_cap']
    elif failure == 'out_of_order': current['capture_timestamp'] = state['last_timestamp']-.01
    elif failure == 'gap': current['capture_timestamp'] = state['last_timestamp']+.351
    elif failure == 'low_score': current['low_score_continuation'] = True
    elif failure == 'prediction': current['is_fresh'] = False
    elif failure == 'position_only': state['position_only'] = True
    elif failure == 'raw_changed': current['track_id'] = 2
    elif failure == 'opposite_side': current['detector_bbox'] = (496., 0., 640., 479.)
    elif failure == 'no_previous_crop':
        state['observation']['quality_bbox_reason'] = state['observation']['bbox_quality_reason'] = ''
    elif failure == 'confidence': current['detector_confidence'] = .749
    elif failure == 'missing_confidence': current.pop('detector_confidence')
    elif failure == 'sliver': current['detector_bbox'] = (0., 0., 79., 479.)
    elif failure == 'other_quality': current['bbox_quality_reason'] = 'aspect<0.18'
    elif failure == 'image_resize': state['observation']['image_width'] = 1280
    elif failure == 'jump': geometry['yaw_compensated_center_jump_ratio'] = .101
    elif failure == 'scale': geometry['area_similarity'] = .699
    elif failure == 'geometry_unknown': geometry['ok'] = None
    else: current.pop('image_width')
    assert not crop_reverification_eligible(current, state, geometry)


def test_failed_crop_cannot_extend_missing_observation_cleanup():
    bank, _, _ = before427()
    assert send_row(bank, full=.31) == 0
    cap, frame, _, yaw, score, distance, box = ROWS[-1]
    late = ROWS[-2][2]+.501
    assert send_row(bank, (cap+1, frame+1, late, yaw, score, distance, box)) == 0
    assert (1, 1) not in bank._similar_follow_states
