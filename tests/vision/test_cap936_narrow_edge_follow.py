"""CAP932->936 recorded geometry/clock/score with synthetic descriptors.

The approved synthetic gallery is seeded at CAP927 to isolate the recorded
edge transition, not to replay OSNet or the full run's gallery history.
No camera, NPU or motor is used.
"""
from copy import deepcopy
import json

import pytest

from rk_vision.identity_bank import _geometry_observation
from rk_vision.similar_follow import narrow_edge_follow_geometry
from test_cap1040_identity_continuity import make_bank
from test_cap874_identity_reacquire import feature, metadata
from tools.replay_cap334_recovery import gallery_snapshot


# frame, timestamp, yaw, detector score, full distance, torso distance, bbox
ROWS = {
    927: (388, 35745.383288668, -145.19725705929807, .896659791469574,
          .12955021858215332, .23262588679790497,
          (307.73382568359375, 4.332427978515625, 489.90692138671875, 475.2752685546875)),
    932: (390, 35745.619783628, -151.1161392569253, .6743785738945007,
          .15073072910308838, .18572066724300385,
          (405.2136535644531, 3.1571197509765625, 638.5869750976562, 475.2843017578125)),
    936: (391, 35745.818275121, -153.07148927119525, .8137752413749695,
          .2079584002494812, .2887549102306366,
          (554.08544921875, 2.08880615234375, 639.2608642578125, 477.07073974609375)),
    833: (333, 35740.490865663, -73.62813165860504, .8928923010826111,
          .21271789073944092, .25301846861839294,
          (.002910614013671875, 2.600189208984375, 87.96466064453125, 472.5433349609375)),
    839: (335, 35740.793575689, -75.04189182916085, .6442387700080872,
          .32756471633911133, None,
          (.685175895690918, 120.77053833007812, 48.58641815185547, 475.84844970703125)),
}
PROTECTED = dict(frame_index=106, track_id=4, capture_frame_id=334,
    bbox=[42.841217041015625, 1.1620635986328125, 296.48822021484375, 477.41497802734375],
    center_x_ratio=.2651011228561401, area=.39322957193243535, area_units='ratio',
    geometry_source='detector', capture_timestamp=35714.271756765, edge_touch_count=2.,
    integrated_yaw_deg=87.39174315420235, yaw_rate_dps=-1.5675)


def sample(cap=936, *, mirror=False, **changes):
    frame, stamp, yaw, score, _, _, box = ROWS[cap]
    if mirror:
        box, yaw = (640-box[2], box[1], 640-box[0], box[3]), -yaw
    reason = ('' if cap == 927 else 'aspect<0.18,edge_touch>2' if cap == 936
              else 'aspect<0.18' if cap == 839 else 'edge_touch>2')
    m = dict(metadata(cap, stamp, box, yaw, search=False), track_id=11,
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


def send(bank, cap=936, *, mirror=False, full=None, partial=None, **changes):
    m = sample(cap, mirror=mirror, **changes)
    box = m['detector_bbox']
    full = ROWS[cap][4] if full is None else full
    part = ROWS[cap][5] if partial is None else partial
    return bank.assign(track_id=m['track_id'], feature=feature(full),
        partial_feature=None if part is None else feature(part),
        confidence=m['detector_confidence'], area=(box[2]-box[0])*(box[3]-box[1]),
        frame_index=m['frame_index'], bbox_quality_ok=m['quality_bbox_ok'],
        bbox_quality_tier=m['bbox_quality_tier'], bbox_quality_reason=m['quality_bbox_reason'],
        sample_metadata=m)


def before936(*, mirror=False):
    bank = make_bank()
    assert send(bank, 927, mirror=mirror, full=0., partial=0.) == 1
    assert send(bank, 932, mirror=mirror) == 1
    assert bank.last_assignments[11]['similar_follow']['crop_continuation']
    # The actual run still held CAP334 as its protected historical anchor.
    # Recovery must consume current raw-local evidence, not rewrite this anchor.
    protected = deepcopy(PROTECTED)
    if mirror:
        x1, y1, x2, y2 = protected['bbox']
        protected.update(bbox=[640-x2, y1, 640-x1, y2],
            center_x_ratio=1-protected['center_x_ratio'],
            integrated_yaw_deg=-protected['integrated_yaw_deg'])
    bank.identities[1].last_strong_observation = deepcopy(protected)
    bank._reacquire_search_anchors[1] = protected
    return bank


@pytest.mark.parametrize('mirror', [False, True])
def test_recorded_cap936_assign_preserves_follow_without_learning_or_quality_rewrite(mirror, caplog):
    caplog.set_level(20, logger='PersonTracker')
    bank = before936(mirror=mirror)
    gallery = gallery_snapshot(bank)
    anchor = deepcopy(bank.identities[1].last_strong_observation)
    references = deepcopy(bank._follow_references)
    assert send(bank, mirror=mirror) == 1
    a = bank.last_assignments[11]
    assert a['reason'] == 'mapped_similar_follow'
    assert a['similar_follow']['narrow_edge_continuation']
    assert a['similar_follow']['completed_confirmation']
    assert not a['similar_follow']['learning_allowed']
    assert a['similar_follow']['raw_track_id'] == 11
    assert a['similar_follow']['capture_frame_id'] == 936
    assert a['reacquire_geometry']['ok'] is True
    assert a['reacquire_geometry']['yaw_compensated_center_jump_ratio'] == pytest.approx(.084243399)
    assert a['reacquire_geometry']['area_similarity'] == pytest.approx(.367181792)
    assert a['reacquire_geometry']['crop_continuity_area_similarity'] > .99
    logged = [json.loads(record.message.split('reid_match_evidence ', 1)[1])
              for record in caplog.records if 'reid_match_evidence {' in record.message]
    m = logged[-1]['query_metadata']
    assert m['quality_bbox_ok'] is False and m['bbox_quality_tier'] == 'weak'
    assert m['quality_bbox_reason'] == 'aspect<0.18,edge_touch>2'
    assert not a['bank_updated'] and not a['recent_bank_updated']
    assert not a['learning_written_tiers']
    assert gallery_snapshot(bank) == gallery
    assert bank.identities[1].last_strong_observation == anchor
    assert anchor['capture_frame_id'] == 334
    assert bank._follow_references == references
    assert bank._similar_follow_states[(1, 11)]['last_strong_timestamp'] == ROWS[927][1]


def pure_input():
    bank = before936()
    state = deepcopy(bank._similar_follow_states[(1, 11)])
    current = sample()
    geometry = bank._handoff_geometry(1, current, current['frame_index'],
        reference_override=_geometry_observation(state['observation'], 390))
    return dict(uid=1, track_id=11, current=current, state=state, geometry=geometry,
                gallery_distance=ROWS[936][4], competition_ok=True, blocked=False,
                partial_conflict=False)


@pytest.mark.parametrize('fault', ['raw', 'uid', 'inactive', 'position_only', 'stale',
    'duplicate', 'gap', 'strong_age', 'missing_strong_age', 'appearance', 'competition',
    'blocked', 'partial_conflict', 'geometry', 'jump', 'area', 'opposite', 'short',
    'sliver', 'wrong_reason', 'inward', 'missing_dimensions', 'nonfresh_probe'])
def test_narrow_crop_pure_gate_cannot_invent_missing_proof(fault):
    args = pure_input()
    assert narrow_edge_follow_geometry(**args) is not None
    m, state, g = args['current'], args['state'], args['geometry']
    if fault == 'raw': m['track_id'] = 12
    elif fault == 'uid': args['uid'] = 2
    elif fault == 'inactive': state.update(active=False, count=1)
    elif fault == 'position_only': state['position_only'] = True
    elif fault == 'stale': m['is_fresh'] = False
    elif fault == 'duplicate': m['capture_frame_id'] = 932
    elif fault == 'gap': m['capture_timestamp'] = state['last_timestamp']+.351
    elif fault == 'strong_age': state['last_strong_timestamp'] = m['capture_timestamp']-.751
    elif fault == 'missing_strong_age': state.pop('last_strong_timestamp')
    elif fault == 'appearance': args['gallery_distance'] = .301
    elif fault == 'competition': args['competition_ok'] = False
    elif fault == 'blocked': args['blocked'] = True
    elif fault == 'partial_conflict': args['partial_conflict'] = True
    elif fault == 'geometry': g['ok'] = False
    elif fault == 'jump': g['yaw_compensated_center_jump_ratio'] = .121
    elif fault == 'area': g['area_similarity'] = .349
    elif fault == 'opposite': m['detector_bbox'] = [0., 2., 85., 477.]
    elif fault == 'short': m['detector_bbox'] = [554., 150., 639., 477.]
    elif fault == 'sliver': m['detector_bbox'] = [565., 2., 639., 477.]
    elif fault == 'wrong_reason': m['quality_bbox_reason'] += ',identity_swap_competing_track'
    elif fault == 'inward': m['detector_bbox'] = [390., 2., 639., 477.]
    elif fault == 'missing_dimensions': m.pop('image_width')
    else: m['search_observation_only'] = True
    assert narrow_edge_follow_geometry(**args) is None


def test_new_raw_id_cannot_use_the_existing_crop_permission():
    bank = before936()
    gallery = gallery_snapshot(bank)
    assert send(bank, track_id=12) == 0
    assert not bank.last_assignments[12].get('similar_follow', {}).get('narrow_edge_continuation')
    assert gallery_snapshot(bank) == gallery


def test_follow_reference_cannot_replace_current_independent_appearance():
    bank = before936()
    bank._follow_references[1] = [dict(cap=932, timestamp=ROWS[932][1], feature=feature(.4))]
    assert send(bank, full=.4) == 0
    detail = bank.last_assignments[11]['similar_follow']
    assert detail['full_distance'] < .01 and detail['gallery_distance'] > .30
    assert not detail.get('narrow_edge_continuation')


@pytest.mark.parametrize('conflict', ['competition', 'geometry', 'exclusion', 'partial'])
def test_current_negative_evidence_is_not_erased_by_narrow_crop(monkeypatch, conflict):
    bank = before936()
    gallery = gallery_snapshot(bank)
    changes = {}
    if conflict == 'competition':
        changes['identity_competition'] = dict(uid=1, frame_index=391,
            candidate_count=1, source_detection_index=0, passed=False)
    elif conflict == 'geometry':
        bank._mapped_geometry_conflicts[11] = dict(uid=1, search_contradiction=True)
    elif conflict == 'exclusion':
        monkeypatch.setattr(bank, 'search_exclusion_for', lambda *a, **k: {'reason': 'different_person'})
    else:
        memory = bank.identities[1].template_memory
        original = memory.evidence

        def evidence(feature_value, current, tier='strong', **options):
            result = original(feature_value, current, tier, **options)
            if tier == 'partial' and options.get('reliable_only') and options.get('comparable_only'):
                result = dict(result, count=1, distance=.75, comparable_count=1,
                              comparable_distance=.75, comparable_caps=[927])
            return result

        monkeypatch.setattr(memory, 'evidence', evidence)
    assert send(bank, **changes) == 0
    assert not bank.last_assignments[11].get('similar_follow', {}).get('narrow_edge_continuation')
    assert not bank.last_assignments[11]['bank_updated']
    if conflict == 'geometry':
        assert bank._mapped_geometry_conflicts[11]['search_contradiction']
        # Existing true-conflict isolation may remove this synthetic CAP927
        # seed because the protected CAP334 anchor predates it. Never undo it.
        after = gallery_snapshot(bank)
        assert all(tier['count'] <= gallery[uid][name]['count']
                   for uid, tiers in after.items() for name, tier in tiers.items())
    else:
        assert gallery_snapshot(bank) == gallery


def test_narrow_crops_do_not_roll_the_strong_observation_deadline():
    bank = before936()
    gallery = gallery_snapshot(bank)
    for i in range(4):
        assert send(bank, capture_frame_id=936+i, frame_index=391+i,
                    capture_timestamp=ROWS[936][1]+.1*i) == 1
    assert send(bank, capture_frame_id=940, frame_index=395,
                capture_timestamp=ROWS[936][1]+.4) == 0
    assert gallery_snapshot(bank) == gallery


def test_cap839_new_raw_and_48_pixel_sliver_still_have_no_follow_permission():
    args = pure_input()
    old, current = sample(833, track_id=9), sample(839, track_id=10)
    args['state'].update(track_id=9, last_cap=833, last_timestamp=old['capture_timestamp'],
        origin_cap=830, origin_timestamp=old['capture_timestamp']-.2, observation=old,
        last_strong_timestamp=old['capture_timestamp']-.2)
    args.update(current=current, track_id=10, gallery_distance=ROWS[839][4])
    assert narrow_edge_follow_geometry(**args) is None
