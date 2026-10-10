"""CAP734 -> 738: recorded geometry, clocks, gallery AND local distances.

Run 20261010_223336 saved scalar OSNet distances, not vectors. Three-dimensional
synthetic vectors below reproduce both recorded gallery and 734->738 pairwise
distances. This exercises the real assign path without running a model/hardware;
the isolated enrollment/730 observation is a synthetic bootstrap, not a replay
of the entire historical bank. Subsequent CAP743 uses its logged gallery scores.
"""
from copy import deepcopy
import math

import numpy as np
import pytest

from rk_vision.identity_bank import _geometry_observation
from rk_vision.similar_follow import (bounded_crop_follow_geometry,
                                     bounded_crop_observation_retainable)
from test_cap1040_identity_continuity import make_bank
from test_cap874_identity_reacquire import feature, metadata
from tools.replay_cap334_recovery import gallery_snapshot


# frame, stamp, yaw, confidence, gallery full/torso, detector box, quality
ROWS = {
    730: (336, 43189.865239676, -223.68204310335076, .8401476144790649,
          .15119314193725586, .3062628209590912,
          (554.7018432617188, 60.2923583984375, 640., 475.7645263671875), ''),
    734: (337, 43190.066357411, -220.7328656704957, .7987053394317627,
          .1965421438217163, .35127100348472595,
          (562.7926635742188, 56.30287170410156, 640., 456.6064453125), ''),
    738: (338, 43190.268138088, -217.28509559303237, .8665199279785156,
          .23739498853683472, .30553945899009705,
          (564.517822265625, 10.35626220703125, 639.3312377929688, 456.78607177734375),
          'aspect<0.18'),
    743: (339, 43190.500556113, -211.68498601242965, .8928923010826111,
          .18104958534240723, .22238202393054962,
          (514.7612915039062, 1.6149139404296875, 639.2716674804688, 476.74249267578125),
          'edge_touch>2'),
}
LOCAL_FULL, LOCAL_PARTIAL = .12164616584777832, .13804179430007935


def pair_vector(anchor_gallery, current_gallery, local_distance):
    """e0 is gallery; anchor lies in e0/e1; solve the two measured dot products."""
    anchor = feature(anchor_gallery).astype(np.float64)
    x = 1-current_gallery
    y = (1-local_distance-anchor[0]*x)/anchor[1]
    square = 1-x*x-y*y
    assert square >= 0
    return np.asarray([x, y, math.sqrt(square)], dtype=np.float32)


def sample(cap, *, mirror=False, **changes):
    frame, stamp, yaw, score, _, _, box, reason = ROWS[cap]
    if mirror:
        box, yaw = (640-box[2], box[1], 640-box[0], box[3]), -yaw
    m = dict(metadata(cap, stamp, box, yaw, search=False), track_id=10,
        frame_index=frame, control_frame_id=frame, image_width=640, image_height=480,
        source_detection_index=0, detector_confidence=score, partial_feature_source='osnet_torso',
        partial_observation=True, quality_bbox_ok=not reason,
        bbox_quality_tier='weak' if reason else 'strong',
        quality_bbox_reason=reason, bbox_quality_reason=reason,
        detector_edge_touch_count=3 if cap == 743 else 1,
        identity_competition=dict(uid=1, frame_index=frame, source_detection_index=0,
            candidate_count=1, passed=True, reason='single_candidate'))
    m.update(changes)
    if 'identity_competition' not in changes:
        m['identity_competition']['frame_index'] = m['frame_index']
    return m


def send(bank, cap, *, mirror=False, full_vector=None, partial_vector=None,
         local_full=LOCAL_FULL, local_partial=LOCAL_PARTIAL, **changes):
    m = sample(cap, mirror=mirror, **changes)
    if full_vector is None:
        full_vector = (pair_vector(ROWS[734][4], ROWS[738][4], local_full)
                       if cap == 738 else feature(ROWS[cap][4]))
    if partial_vector is None:
        partial_vector = (pair_vector(ROWS[734][5], ROWS[738][5], local_partial)
                          if cap == 738 else feature(ROWS[cap][5]))
    box = m['detector_bbox']
    return bank.assign(track_id=m['track_id'], feature=full_vector, partial_feature=partial_vector,
        confidence=m['detector_confidence'], area=(box[2]-box[0])*(box[3]-box[1]),
        frame_index=m['frame_index'], bbox_quality_ok=m['quality_bbox_ok'],
        bbox_quality_tier=m['bbox_quality_tier'], bbox_quality_reason=m['quality_bbox_reason'],
        sample_metadata=m)


def before_738(*, mirror=False):
    bank = make_bank()
    assert send(bank, 730, mirror=mirror, capture_frame_id=700, frame_index=330,
        capture_timestamp=ROWS[730][1]-.5, full_vector=feature(0), partial_vector=feature(0)) == 1
    bank._similar_learning_fences.add(1)
    assert send(bank, 730, mirror=mirror) == 0
    assert send(bank, 734, mirror=mirror) == 1
    assert bank._similar_follow_states[(1, 10)]['crop_appearance_anchor']['capture'] == 734
    return bank


def recovered_738(*, mirror=False):
    """Shared integration fixture: actual 734/738 fields + constrained synthetic vectors."""
    bank = before_738(mirror=mirror)
    assert send(bank, 738, mirror=mirror) == 1
    return bank, sample(738, mirror=mirror), deepcopy(bank.last_assignments[10])


@pytest.mark.parametrize('mirror', [False, True])
def test_recorded_crop_retains_completed_identity_and_current_geometry_without_learning(mirror):
    bank = before_738(mirror=mirror)
    anchor = deepcopy(bank._similar_follow_states[(1, 10)]['crop_appearance_anchor'])
    gallery, refs = gallery_snapshot(bank), deepcopy(bank._follow_references)
    protected = deepcopy(bank.identities[1].last_strong_observation)
    assert send(bank, 738, mirror=mirror) == 1
    result = bank.last_assignments[10]
    review = result['bounded_crop_review']
    assert review['eligible'] and not review['observation_retainable']
    assert review['local_full_distance'] == pytest.approx(LOCAL_FULL, abs=2e-7)
    assert review['local_partial_distance'] == pytest.approx(LOCAL_PARTIAL, abs=2e-7)
    assert review['gallery_distance'] == pytest.approx(ROWS[738][4], abs=2e-7)
    proof = result['similar_follow']['bounded_crop_continuation']
    assert proof['visibility_mode'] == 'single_end_change'
    assert proof['height_similarity'] == pytest.approx(.896677518)
    assert proof['reference_age_ms'] == pytest.approx(201.780677)
    assert proof['permission'] == 'follow_only' and not proof['learning_allowed']
    assert proof['expires_at'] == anchor['timestamp']+.5 and not proof['anchor_renewed']
    assert result['similar_follow']['completed_confirmation'] and result['similar_follow']['count'] == 2
    assert result['reason'] == 'mapped_similar_follow' and not result['identity_control_rejected']
    assert result['reacquire_geometry']['ok']
    assert result['reacquire_geometry']['current']['capture_frame_id'] == 738
    assert bank._similar_follow_states[(1, 10)]['crop_appearance_anchor'] == anchor
    assert bank.identities[1].last_strong_observation == protected
    assert gallery_snapshot(bank) == gallery and bank._follow_references == refs
    assert not result['bank_updated'] and not result['recent_bank_updated']
    assert not result['learning_written_tiers']
    # The following recorded wider edge3 crop uses existing continuity, not a
    # new observe/park cycle. Its local pair distances were not logged.
    assert send(bank, 743, mirror=mirror) == 1
    assert bank.last_assignments[10]['similar_follow']['count'] == 2
    assert not bank.last_assignments[10]['learning_written_tiers']


def pure_input():
    bank = before_738()
    m = sample(738)
    state = deepcopy(bank._similar_follow_states[(1, 10)])
    geometry = bank._handoff_geometry(1, m, m['frame_index'],
        reference_override=_geometry_observation(state['observation'], 337))
    return dict(uid=1, track_id=10, current=m, state=state, geometry=geometry,
        gallery_distance=ROWS[738][4], local_full_distance=LOCAL_FULL,
        local_partial_distance=LOCAL_PARTIAL, competition_ok=True, blocked=False,
        partial_conflict=False)


@pytest.mark.parametrize('fault', ['both_ends', 'height', 'width', 'center', 'gallery',
    'local_full', 'local_partial', 'competition', 'blocked', 'partial_conflict',
    'new_raw', 'new_uid', 'inactive', 'anchor_age', 'gap', 'sliver', 'short', 'geometry',
    'opposite_edge', 'unknown_torso', 'nonfinite_full'])
def test_edge_visibility_exception_does_not_remove_other_guards(fault):
    args = pure_input()
    assert bounded_crop_follow_geometry(**args)
    m = args['current']
    if fault == 'both_ends': m['detector_bbox'] = [564.5, 10.3, 639.3, 480.]
    elif fault == 'height': m['detector_bbox'] = [564.5, 120., 639.3, 456.8]
    elif fault == 'width': m['detector_bbox'] = [540., 10.3, 639.3, 456.8]
    elif fault == 'center': m['detector_bbox'] = [600., 10.3, 675., 456.8]
    elif fault == 'gallery': args['gallery_distance'] = .301
    elif fault == 'local_full': args['local_full_distance'] = .161
    elif fault == 'local_partial': args['local_partial_distance'] = .161
    elif fault == 'competition': args['competition_ok'] = False
    elif fault == 'blocked': args['blocked'] = True
    elif fault == 'partial_conflict': args['partial_conflict'] = True
    elif fault == 'new_raw': m['track_id'] = 11
    elif fault == 'new_uid': args['uid'] = 2
    elif fault == 'inactive': args['state'].update(active=False, count=1)
    elif fault == 'anchor_age': m['capture_timestamp'] = ROWS[734][1]+.501
    elif fault == 'gap': m['capture_timestamp'] = ROWS[734][1]+.351
    elif fault == 'sliver': m['detector_bbox'] = [600., 10., 640., 456.]
    elif fault == 'short': m['detector_bbox'] = [564., 170., 640., 456.]
    elif fault == 'geometry': args['geometry']['ok'] = False
    elif fault == 'opposite_edge': m['detector_bbox'] = [0., 10.3, 75., 456.8]
    elif fault == 'unknown_torso': args['local_partial_distance'] = None
    else: args['local_full_distance'] = float('nan')
    assert bounded_crop_follow_geometry(**args) is None
    if fault not in ('local_full', 'local_partial'):
        assert not bounded_crop_observation_retainable(**args)


def test_identical_box_does_not_borrow_visibility_change_appearance_allowance():
    args = pure_input()
    args['current']['detector_bbox'] = deepcopy(args['state']['observation']['detector_bbox'])
    args.update(local_full_distance=.13, local_partial_distance=.14)
    assert bounded_crop_follow_geometry(**args) is None
    assert bounded_crop_observation_retainable(**args)  # no new authorization
    args.update(local_full_distance=.11, local_partial_distance=.11)
    result = bounded_crop_follow_geometry(**args)
    assert result['bounded_crop_confirmation']['visibility_mode'] == 'stable_height'
    assert result['bounded_crop_confirmation']['local_distance_limit'] == .12


@pytest.mark.parametrize('channel', ['full', 'partial'])
def test_wrong_local_person_with_same_gallery_score_is_not_laundered(channel):
    bank = before_738()
    before = gallery_snapshot(bank)
    vector = feature(ROWS[738][4 if channel == 'full' else 5])
    vector[1] *= -1
    assert send(bank, 738, **{channel+'_vector': vector}) == 0
    result = bank.last_assignments[10]
    assert not result['bounded_crop_review']['eligible']
    assert not result['bounded_crop_review']['observation_retainable']
    assert (1, 10) not in bank._similar_follow_states
    assert gallery_snapshot(bank) == before


@pytest.mark.parametrize('fault', ['competition', 'geometry', 'new_raw', 'wrong_source'])
def test_actual_assign_does_not_bypass_competition_conflict_or_new_binding(fault):
    bank = before_738()
    gallery = gallery_snapshot(bank)
    changes = {}
    if fault == 'competition':
        changes['identity_competition'] = dict(uid=1, frame_index=338,
            source_detection_index=0, candidate_count=2, passed=False)
    elif fault == 'geometry':
        bank._mapped_geometry_conflicts[10] = dict(uid=1, search_contradiction=True)
    elif fault == 'new_raw': changes['track_id'] = 11
    else: changes['partial_feature_source'] = 'different_region'
    assert send(bank, 738, **changes) == 0
    result = bank.last_assignments[changes.get('track_id', 10)]
    assert not result.get('similar_follow', {}).get('bounded_crop_continuation')
    assert not result.get('bounded_crop_review', {}).get('observation_retainable')
    assert gallery_snapshot(bank) == gallery


def test_inconclusive_frame_keeps_only_original_state_then_fresh_evidence_recovers():
    bank = before_738()
    state = deepcopy(bank._similar_follow_states[(1, 10)])
    gallery = gallery_snapshot(bank)
    assert send(bank, 738, local_partial=.19) == 0
    a = bank.last_assignments[10]
    assert a['identity_control_rejected'] and a['bounded_crop_review']['observation_retainable']
    assert a['similar_follow']['observation_retained']
    assert bank._similar_follow_states[(1, 10)] == state
    assert gallery_snapshot(bank) == gallery
    assert send(bank, 738, capture_frame_id=739, frame_index=339,
        capture_timestamp=ROWS[738][1]+.05) == 1
    assert bank.last_assignments[10]['similar_follow']['count'] == 2
    assert bank._similar_follow_states[(1, 10)]['crop_appearance_anchor'] == state['crop_appearance_anchor']


def test_repeated_inconclusive_frames_cannot_extend_any_observation_clock():
    bank = before_738()
    state = deepcopy(bank._similar_follow_states[(1, 10)])
    for offset in (0., .10):
        assert send(bank, 738, local_full=.19, capture_frame_id=738+int(offset*100),
            frame_index=338+int(offset*100), capture_timestamp=ROWS[738][1]+offset) == 0
        assert bank._similar_follow_states[(1, 10)] == state
    assert send(bank, 738, capture_frame_id=750, frame_index=345,
                capture_timestamp=state['last_timestamp']+.351) == 0
    assert (1, 10) not in bank._similar_follow_states


def test_accepted_crops_cannot_roll_fixed_anchor_or_keep_learning_alive():
    bank = before_738()
    anchor = deepcopy(bank._similar_follow_states[(1, 10)]['crop_appearance_anchor'])
    gallery = gallery_snapshot(bank)
    for offset in (0., .10, .20):
        assert send(bank, 738, capture_frame_id=738+int(offset*100),
            frame_index=338+int(offset*100), capture_timestamp=ROWS[738][1]+offset) == 1
        assert bank._similar_follow_states[(1, 10)]['crop_appearance_anchor'] == anchor
    assert send(bank, 738, capture_frame_id=769, frame_index=369,
                capture_timestamp=anchor['timestamp']+.501) == 0
    assert gallery_snapshot(bank) == gallery
