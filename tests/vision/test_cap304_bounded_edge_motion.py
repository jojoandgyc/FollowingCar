"""CAP298 -> 304 from run_20261010_234438: real metadata/scalar distances.

The run did not save embedding vectors. Synthetic 3-D vectors reproduce both
gallery and current-to-298 distances; CAP286 is an isolated enrollment seed,
not a claim to replay the entire historical gallery. No models/hardware run.
"""
from copy import deepcopy

import pytest

from rk_vision.identity_bank import _geometry_observation
from rk_vision.similar_follow import bounded_crop_follow_geometry
from test_cap1040_identity_continuity import make_bank
from test_cap738_edge_visibility import pair_vector
from test_cap874_identity_reacquire import feature, metadata
from tools.replay_cap334_recovery import gallery_snapshot


# control frame, capture time, yaw, score, gallery full/partial, detector bbox
ROWS = {
    286: (99, 47429.052626536, 44.15920436585398, .95, 0., 0.,
          (52.5550537109375, 2.56561279296875, 295.1783447265625, 478.77349853515625)),
    290: (100, 47429.253942696, 44.252544765399435, .9381020069122314,
          .06122756004333496, .11326605081558228,
          (.8044815063476562, 2.8109130859375, 224.8535614013672, 477.177978515625)),
    294: (101, 47429.453706936, 42.72586497030026, .9343345761299133,
          .08448004722595215, .11448655277490616,
          (0., 2.1222076416015625, 196.45462036132812, 475.87005615234375)),
    298: (102, 47429.652234249, 39.519387029849774, .896659791469574,
          .0724114179611206, .10265127569437027,
          (.00945281982421875, 3.6509246826171875, 199.13917541503906, 476.8804931640625)),
    304: (104, 47429.984474278, 33.415245397627, .9305670857429504,
          .07197815179824829, .09944748878479004,
          (.04235076904296875, 3.26922607421875, 192.241455078125, 476.99871826171875)),
}
LOCAL_FULL, LOCAL_PARTIAL = .07177495956420898, .06055188179016113


def sample(cap, *, mirror=False, **changes):
    frame, stamp, yaw, score, _, _, box = ROWS[cap]
    if mirror:
        box, yaw = (640-box[2], box[1], 640-box[0], box[3]), -yaw
    reason = '' if cap == 286 else 'edge_touch>2'
    m = dict(metadata(cap, stamp, box, yaw, search=False), track_id=2,
        frame_index=frame, control_frame_id=frame, image_width=640, image_height=480,
        source_detection_index=0, detector_confidence=score, partial_feature_source='osnet_torso',
        partial_observation=True, quality_bbox_ok=not reason,
        bbox_quality_tier='weak' if reason else 'strong',
        quality_bbox_reason=reason, bbox_quality_reason=reason,
        detector_edge_touch_count=3 if reason else 2, edge_touch_count=3 if reason else 2,
        identity_competition=dict(uid=1, frame_index=frame, source_detection_index=0,
            candidate_count=1, passed=True, reason='single_candidate'))
    m.update(changes)
    if 'identity_competition' not in changes:
        m['identity_competition']['frame_index'] = m['frame_index']
    return m


def send(bank, cap, *, mirror=False, full_vector=None, partial_vector=None, **changes):
    m = sample(cap, mirror=mirror, **changes)
    if full_vector is None:
        full_vector = (pair_vector(ROWS[298][4], ROWS[304][4], LOCAL_FULL)
                       if cap == 304 else feature(ROWS[cap][4]))
    if partial_vector is None:
        partial_vector = (pair_vector(ROWS[298][5], ROWS[304][5], LOCAL_PARTIAL)
                          if cap == 304 else feature(ROWS[cap][5]))
    box = m['detector_bbox']
    return bank.assign(track_id=m['track_id'], feature=full_vector, partial_feature=partial_vector,
        confidence=m['detector_confidence'], area=(box[2]-box[0])*(box[3]-box[1]),
        frame_index=m['frame_index'], bbox_quality_ok=m['quality_bbox_ok'],
        bbox_quality_tier=m['bbox_quality_tier'], bbox_quality_reason=m['quality_bbox_reason'],
        sample_metadata=m)


def before_304(*, mirror=False):
    bank = make_bank()
    for cap in (286, 290, 294, 298):
        assert send(bank, cap, mirror=mirror) == 1
    assert bank._similar_follow_states[(1, 2)]['crop_appearance_anchor']['capture'] == 298
    return bank


@pytest.mark.parametrize('mirror', [False, True])
def test_actual_cap304_follows_without_learning_or_refreshing_fixed_anchor(mirror):
    bank = before_304(mirror=mirror)
    gallery, refs = gallery_snapshot(bank), deepcopy(bank._follow_references)
    protected = deepcopy(bank.identities[1].last_strong_observation)
    anchor = deepcopy(bank._similar_follow_states[(1, 2)]['crop_appearance_anchor'])
    assert send(bank, 304, mirror=mirror) == 1
    result = bank.last_assignments[2]
    assert result['reason'] == 'mapped_similar_follow'
    assert result['reacquire_geometry']['yaw_compensated_center_jump_ratio'] == pytest.approx(.1070988364)
    proof = result['similar_follow']['bounded_crop_continuation']
    assert proof['fixed_anchor_motion_bridge'] and proof['compensated_jump_limit'] == .12
    assert proof['full_distance'] == pytest.approx(LOCAL_FULL, abs=2e-7)
    assert proof['partial_distance'] == pytest.approx(LOCAL_PARTIAL, abs=2e-7)
    assert proof['gallery_distance'] == pytest.approx(ROWS[304][4], abs=2e-7)
    assert proof['reference_age_ms'] == pytest.approx(332.240029)
    assert proof['expires_at'] == ROWS[298][1]+.5
    assert proof['permission'] == 'follow_only' and not proof['learning_allowed']
    assert result['similar_follow']['completed_confirmation'] and result['similar_follow']['count'] == 2
    assert not result['bank_updated'] and not result['recent_bank_updated']
    assert not result['learning_written_tiers']
    assert bank._similar_follow_states[(1, 2)]['crop_appearance_anchor'] == anchor
    assert bank.identities[1].last_strong_observation == protected
    assert gallery_snapshot(bank) == gallery and bank._follow_references == refs


def pure_input():
    bank = before_304()
    state, current = deepcopy(bank._similar_follow_states[(1, 2)]), sample(304)
    geometry = bank._handoff_geometry(1, current, 104,
        reference_override=_geometry_observation(state['observation'], 102))
    return dict(uid=1, track_id=2, current=current, state=state, geometry=geometry,
        gallery_distance=ROWS[304][4], local_full_distance=LOCAL_FULL,
        local_partial_distance=LOCAL_PARTIAL, competition_ok=True, blocked=False,
        partial_conflict=False)


@pytest.mark.parametrize('fault', ['gallery', 'anchor_gallery', 'local_full', 'local_partial',
    'residual', 'width', 'height', 'center', 'end_cut', 'opposite_edge', 'new_raw', 'new_uid',
    'competition', 'blocked', 'partial_conflict', 'geometry', 'anchor_age', 'gap', 'low_score'])
def test_motion_band_requires_strong_independent_evidence_and_bounded_shape(fault):
    args = pure_input()
    assert bounded_crop_follow_geometry(**args)
    m, state = args['current'], args['state']
    if fault == 'gallery': args['gallery_distance'] = .151
    elif fault == 'anchor_gallery': state['crop_appearance_anchor']['gallery_distance'] = .151
    elif fault == 'local_full': args['local_full_distance'] = .101
    elif fault == 'local_partial': args['local_partial_distance'] = .101
    elif fault == 'residual': args['geometry']['yaw_compensated_center_jump_ratio'] = .121
    elif fault == 'width': m['detector_bbox'] = (0., 3., 175., 477.)
    elif fault == 'height': m['detector_bbox'] = (0., 40., 192., 477.)
    elif fault == 'center': m['detector_bbox'] = (0., 3., 250., 477.)
    elif fault == 'end_cut': m['detector_bbox'] = (0., 12., 192., 477.)
    elif fault == 'opposite_edge': m['detector_bbox'] = (448., 3., 640., 477.)
    elif fault == 'new_raw': m['track_id'] = args['track_id'] = 3
    elif fault == 'new_uid': args['uid'] = 2
    elif fault == 'competition': args['competition_ok'] = False
    elif fault == 'blocked': args['blocked'] = True
    elif fault == 'partial_conflict': args['partial_conflict'] = True
    elif fault == 'geometry': args['geometry']['ok'] = False
    elif fault == 'anchor_age': m['capture_timestamp'] = ROWS[298][1]+.501
    elif fault == 'gap': state['last_timestamp'] -= .02
    else: m['low_score_continuation'] = True
    assert bounded_crop_follow_geometry(**args) is None


@pytest.mark.parametrize('fault', ['competition', 'geometry', 'wrong_person'])
def test_real_assign_keeps_negative_evidence_and_does_not_learn(fault):
    bank = before_304()
    gallery = gallery_snapshot(bank)
    changes = {}
    if fault == 'competition':
        changes['identity_competition'] = dict(uid=1, frame_index=104, source_detection_index=0,
            candidate_count=2, passed=False, reason='reid_margin_insufficient')
    elif fault == 'geometry':
        bank._mapped_geometry_conflicts[2] = dict(uid=1, search_contradiction=True)
    else:
        changes.update(full_vector=feature(.7), partial_vector=feature(.8))
    assert send(bank, 304, **changes) == 0
    assert not bank.last_assignments[2].get('similar_follow', {}).get('bounded_crop_continuation')
    assert gallery_snapshot(bank) == gallery


def test_accepted_motion_band_does_not_self_renew_at_fixed_anchor_deadline():
    bank = before_304()
    assert send(bank, 304) == 1
    gallery = gallery_snapshot(bank)
    # Another similarly compensated observation still needs the original 298
    # anchor. Its fresh timestamp cannot manufacture a rolling motion lease.
    assert send(bank, 304, capture_frame_id=308, frame_index=105,
        capture_timestamp=ROWS[298][1]+.51,
        integrated_yaw_deg=ROWS[304][2]-6.42) == 0
    assert gallery_snapshot(bank) == gallery
