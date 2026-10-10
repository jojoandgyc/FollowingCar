"""Recorded CAP1428/1499 geometry, clocks and gallery scores; synthetic vectors.

The run saved distances, not OSNet vectors. The vectors below reproduce those
gallery scores and deliberately model locally consistent full/torso appearance;
this is an assign-path policy replay, NOT proof of the original pair distances.
No model, camera or motor is started. Prefixes seed an isolated synthetic bank.
"""
from copy import deepcopy

import numpy as np
import pytest

from rk_vision.identity_bank import _geometry_observation
from rk_vision.similar_follow import bounded_crop_follow_geometry
from test_cap1040_identity_continuity import make_bank
from test_cap874_identity_reacquire import feature, metadata
from tools.replay_cap334_recovery import gallery_snapshot


# frame, capture time, yaw, confidence, gallery full/torso, detector box, quality
ROWS = {
    1409: (701, 40872.829038045, -218.55020101192048, .8401476144790649,
           .2701603174209595, .3650808334350586,
           (394.8518371582031, 2.3660430908203125, 563.6314086914062, 463.18212890625), ''),
    1413: (703, 40873.026172959, -217.80392585217666, .8439151048660278,
           .27138662338256836, .3610669672489166,
           (441.3791198730469, 2.9640655517578125, 634.8395385742188, 475.89337158203125), 'edge_touch>2'),
    1415: (704, 40873.159062017, -217.63168072454985, .896659791469574,
           .2638871669769287, .36716142296791077,
           (458.47430419921875, 2.8946685791015625, 639.6993408203125, 476.4305419921875), 'edge_touch>2'),
    1417: (705, 40873.260375447, -217.63168072454985, .8326126337051392,
           .22159790992736816, .2313162237405777,
           (460.778564453125, 8.948684692382812, 638.5682373046875, 476.0533447265625), 'edge_touch>2'),
    1421: (707, 40873.458570835, -216.2674281850645, .8665199279785156,
           .2718247175216675, .24975253641605377,
           (468.63128662109375, 0., 640., 475.1292724609375), 'edge_touch>2'),
    1425: (708, 40873.659091914, -213.40732123458335, .8702874183654785,
           .22518163919448853, .300167053937912,
           (446.40850830078125, 0., 639.18701171875, 473.911376953125), 'edge_touch>2'),
    1428: (709, 40873.823730722, -209.11075316616308, .896659791469574,
           .3114711046218872, .301644891500473,
           (474.2320251464844, 2.6540069580078125, 639.2506103515625, 477.6357421875), 'edge_touch>2'),
    1488: (736, 40876.952696385, -196.7524740399959, .9154971837997437,
           .18020570278167725, .31198957562446594,
           (502.2977294921875, 37.46897888183594, 637.0608520507812, 408.5583190917969), ''),
    1491: (737, 40877.120328784, -197.90857042745486, .9154971837997437,
           .19059991836547852, .3025519549846649,
           (523.1144409179688, 40.5113525390625, 639.998046875, 419.9118347167969), ''),
    1494: (738, 40877.285436581, -198.83443292592872, .9154971837997437,
           .14436888694763184, .23262472450733185,
           (552.7103881835938, 33.54597473144531, 639.2647094726562, 430.2400207519531), ''),
    1499: (739, 40877.532656119, -195.19917350853294, .8815898895263672,
           .18292075395584106, .2188778966665268,
           (571.4631958007812, 31.128067016601562, 639.26904296875, 434.452880859375), 'aspect<0.18'),
    1502: (740, 40877.683968718, -192.64000434139692, .8815898895263672,
           .17918461561203003, .2368219494819641,
           (559.4888305664062, 26.072494506835938, 640., 446.21905517578125), ''),
}


def sample(cap, *, mirror=False, **changes):
    frame, stamp, yaw, score, _, _, box, reason = ROWS[cap]
    if mirror:
        box, yaw = (640-box[2], box[1], 640-box[0], box[3]), -yaw
    raw = 31 if cap < 1488 else 33
    m = dict(metadata(cap, stamp, box, yaw, search=False), track_id=raw,
        frame_index=frame, control_frame_id=frame, image_width=640, image_height=480,
        source_detection_index=0, detector_confidence=score, partial_feature_source='osnet_torso',
        partial_observation=True, quality_bbox_ok=not reason,
        bbox_quality_tier='weak' if reason else 'strong',
        quality_bbox_reason=reason, bbox_quality_reason=reason,
        detector_edge_touch_count=3 if reason == 'edge_touch>2' else 1,
        identity_competition=dict(uid=1, frame_index=frame, source_detection_index=0,
            candidate_count=1, passed=True, reason='single_candidate'))
    m.update(changes)
    if 'identity_competition' not in changes:
        m['identity_competition']['frame_index'] = m['frame_index']
    return m


def send(bank, cap, *, mirror=False, full=None, partial=None, full_vector=None,
         partial_vector=None, **changes):
    m = sample(cap, mirror=mirror, **changes)
    box = m['detector_bbox']
    return bank.assign(track_id=m['track_id'],
        feature=feature(ROWS[cap][4] if full is None else full) if full_vector is None else full_vector,
        partial_feature=feature(ROWS[cap][5] if partial is None else partial) if partial_vector is None else partial_vector,
        confidence=m['detector_confidence'], area=(box[2]-box[0])*(box[3]-box[1]),
        frame_index=m['frame_index'], bbox_quality_ok=m['quality_bbox_ok'],
        bbox_quality_tier=m['bbox_quality_tier'], bbox_quality_reason=m['quality_bbox_reason'],
        sample_metadata=m)


def before_crop(cap, *, mirror=False):
    bank = make_bank()
    if cap == 1428:
        assert send(bank, 1409, mirror=mirror, full=0., partial=0.) == 1
        for prior in (1413, 1415, 1417, 1421, 1425):
            assert send(bank, prior, mirror=mirror) == 1
        assert bank.last_assignments[31]['similar_follow']['crop_current_reverified']
        assert bank._similar_follow_states[(1, 31)]['last_strong_timestamp'] == ROWS[1409][1]
    else:
        assert send(bank, 1488, mirror=mirror, full=0., partial=0.) == 1
        # The real run was already in the provisional-follow learning fence.
        # Bootstrap that policy with the two recorded preceding strong boxes.
        bank._similar_learning_fences.add(1)
        assert send(bank, 1491, mirror=mirror) == 0
        assert send(bank, 1494, mirror=mirror) == 1
    return bank


@pytest.mark.parametrize('cap', [1428, 1499])
@pytest.mark.parametrize('mirror', [False, True])
def test_recorded_assign_crop_keeps_completed_follow_without_learning(cap, mirror):
    bank = before_crop(cap, mirror=mirror)
    raw = sample(cap)['track_id']
    before = deepcopy(bank._similar_follow_states[(1, raw)]['crop_appearance_anchor'])
    gallery, references = gallery_snapshot(bank), deepcopy(bank._follow_references)
    protected = deepcopy(bank.identities[1].last_strong_observation)
    assert send(bank, cap, mirror=mirror) == 1
    a = bank.last_assignments[raw]
    proof = a['similar_follow']['bounded_crop_continuation']
    assert a['reason'] == 'mapped_similar_follow' and not a['identity_control_rejected']
    assert a['similar_follow']['completed_confirmation'] and a['similar_follow']['count'] == 2
    assert a['bounded_crop_review']['eligible'] is True
    assert proof['reference_cap'] == (1425 if cap == 1428 else 1494)
    assert proof['reference_age_ms'] == pytest.approx(164.638808 if cap == 1428 else 247.219538)
    assert not proof['learning_allowed'] and not proof['anchor_renewed']
    assert not a['bank_updated'] and not a['recent_bank_updated'] and not a['learning_written_tiers']
    assert gallery_snapshot(bank) == gallery and bank._follow_references == references
    assert bank.identities[1].last_strong_observation == protected
    assert bank._similar_follow_states[(1, raw)]['crop_appearance_anchor'] == before
    if cap == 1499:
        assert send(bank, 1502, mirror=mirror) == 1  # no second observation/parking cycle
        assert bank.last_assignments[raw]['similar_follow']['count'] == 2


def pure_input(cap=1499):
    bank = before_crop(cap)
    m = sample(cap)
    state = deepcopy(bank._similar_follow_states[(1, m['track_id'])])
    geometry = bank._handoff_geometry(1, m, m['frame_index'],
        reference_override=_geometry_observation(state['observation'], m['frame_index']-1))
    return dict(uid=1, track_id=m['track_id'], current=m, state=state, geometry=geometry,
        gallery_distance=ROWS[cap][4], local_full_distance=.02, local_partial_distance=.02,
        competition_ok=True, blocked=False, partial_conflict=False)


@pytest.mark.parametrize('fault', ['raw', 'uid', 'inactive', 'position', 'missing_anchor',
    'anchor_source', 'anchor_gallery', 'anchor_age', 'gap', 'duplicate', 'low_score',
    'stale', 'weak_probe', 'competition', 'blocked', 'partial_conflict', 'full_gallery',
    'local_full', 'local_partial', 'partial_source', 'geometry', 'jump', 'area',
    'sliver', 'short', 'opposite', 'wrong_reason'])
def test_bounded_crop_rejects_missing_or_conflicting_evidence(fault):
    args = pure_input()
    assert bounded_crop_follow_geometry(**args) is not None
    m, s, g = args['current'], args['state'], args['geometry']
    if fault == 'raw': m['track_id'] += 1
    elif fault == 'uid': args['uid'] = 2
    elif fault == 'inactive': s.update(active=False, count=1)
    elif fault == 'position': s['position_only'] = True
    elif fault == 'missing_anchor': s.pop('crop_appearance_anchor')
    elif fault == 'anchor_source': s['crop_appearance_anchor']['source'] = 'follow_reference'
    elif fault == 'anchor_gallery': s['crop_appearance_anchor']['gallery_distance'] = .301
    elif fault == 'anchor_age': m['capture_timestamp'] = s['crop_appearance_anchor']['timestamp']+.501
    elif fault == 'gap': m['capture_timestamp'] = s['last_timestamp']+.351
    elif fault == 'duplicate': m['capture_frame_id'] = s['last_cap']
    elif fault == 'low_score': m['detector_confidence'] = .749
    elif fault == 'stale': m['is_fresh'] = False
    elif fault == 'weak_probe': m['observation_only'] = True
    elif fault in ('competition', 'blocked', 'partial_conflict'):
        args[dict(competition='competition_ok').get(fault, fault)] = fault != 'competition'
    elif fault == 'full_gallery': args['gallery_distance'] = .351
    elif fault == 'local_full': args['local_full_distance'] = .121
    elif fault == 'local_partial': args['local_partial_distance'] = None
    elif fault == 'partial_source': m['partial_feature_source'] = 'other_region'
    elif fault == 'geometry': g['ok'] = False
    elif fault == 'jump': g['yaw_compensated_center_jump_ratio'] = .101
    elif fault == 'area': g['area_similarity'] = .649
    elif fault == 'sliver': m['detector_bbox'] = [591., 31., 639., 434.]
    elif fault == 'short': m['detector_bbox'] = [571., 130., 639., 434.]
    elif fault == 'opposite': m['detector_bbox'] = [0., 31., 68., 434.]
    else: m['quality_bbox_reason'] += ',identity_swap_competing_track'
    assert bounded_crop_follow_geometry(**args) is None


@pytest.mark.parametrize('cap', [1428, 1499])
def test_repeated_crops_do_not_refresh_the_fixed_anchor(cap):
    bank = before_crop(cap)
    raw = sample(cap)['track_id']
    anchor = deepcopy(bank._similar_follow_states[(1, raw)]['crop_appearance_anchor'])
    gallery = gallery_snapshot(bank)
    for offset in (0., .10, .20):
        assert send(bank, cap, capture_frame_id=cap+int(offset*100),
            frame_index=ROWS[cap][0]+int(offset*100), capture_timestamp=ROWS[cap][1]+offset) == 1
        assert bank._similar_follow_states[(1, raw)]['crop_appearance_anchor'] == anchor
    assert send(bank, cap, capture_frame_id=cap+31, frame_index=ROWS[cap][0]+31,
                capture_timestamp=anchor['timestamp']+.501) == 0
    assert gallery_snapshot(bank) == gallery


@pytest.mark.parametrize('cap', [1428, 1499])
@pytest.mark.parametrize('channel', ['full', 'partial'])
def test_same_gallery_distance_but_wrong_local_person_is_not_laundered(cap, channel):
    bank = before_crop(cap)
    raw = sample(cap)['track_id']
    # Same cosine score to the untouched gallery, opposite local descriptor.
    vector = feature(ROWS[cap][4 if channel == 'full' else 5]).copy()
    vector[1] *= -1
    assert send(bank, cap, **{channel+'_vector': vector}) == 0
    assert bank.last_assignments[raw]['bounded_crop_review']['eligible'] is False
    assert bank.last_assignments[raw]['bounded_crop_review']['local_'+channel+'_distance'] > .12
    assert not bank.last_assignments[raw].get('similar_follow', {}).get('bounded_crop_continuation')
    assert not bank.last_assignments[raw]['bank_updated']


@pytest.mark.parametrize('fault', ['competition', 'geometry', 'new_raw', 'follow_reference'])
def test_assign_does_not_bypass_existing_negative_or_independent_binding(fault):
    bank = before_crop(1499)
    changes = {}
    if fault == 'competition':
        changes['identity_competition'] = dict(uid=1, frame_index=739,
            source_detection_index=0, candidate_count=2, passed=False)
    elif fault == 'geometry':
        bank._mapped_geometry_conflicts[33] = dict(uid=1, search_contradiction=True)
    elif fault == 'new_raw': changes['track_id'] = 34
    else:
        bank._follow_references[1] = [dict(cap=1494, timestamp=ROWS[1494][1], feature=feature(.4))]
        changes['full'] = .4
    assert send(bank, 1499, **changes) == 0
    assert not bank.last_assignments[changes.get('track_id', 33)].get('similar_follow', {}).get('bounded_crop_continuation')


def test_nonfinite_torso_is_not_a_perfect_local_match():
    bank = before_crop(1499)
    assert send(bank, 1499, partial_vector=np.array([np.nan, 0., 0.])) == 0
    assert not bank.last_assignments[33].get('similar_follow', {}).get('bounded_crop_continuation')
