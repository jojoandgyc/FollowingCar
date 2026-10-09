"""Real capture geometry/timing with synthetic vectors for logged gate distances.

This tests policy, not model feature extraction or closed-loop vehicle motion.
"""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from test_cap874_identity_reacquire import feature, metadata

DATA = json.loads((Path(__file__).parent / 'fixtures/cap1253_coverage_recovery.json').read_text())
ROWS = {r['cap']: r for r in DATA['rows']}


def setup():
    b = IdentityBank(IdentityBankConfig(template_memory_enable=True,
        template_crosscheck_enable=True, appearance_region_safety_enable=True,
        partial_match_threshold=.45, partial_confirm_threshold=.34,
        controlled_handoff_enable=True, camera_hfov_deg=60.))
    m = DATA['template']
    b._create_identity(feature(0), m['control_frame_id'], m, feature(0))
    b.identities[1].last_strong_observation = deepcopy(DATA['protected'])
    a = DATA['anchor']
    b._bind_reacquired_identity(1, 5, a['frame_index'], a)
    b.identities[1].last_strong_observation = deepcopy(a)
    return b


def send(b, cap, *, partial=None, full=None, changes=None, track=5):
    r = ROWS[cap]
    m = metadata(cap, r['ts'], r['bbox'], r['yaw'], search=r['search'])
    m.update(image_width=640, image_height=480, partial_feature_source='osnet_torso',
             partial_observation=True, source_detection_index=0,
             search_direction='right' if r['search'] else None,
             search_direction_compatible=True,
             identity_competition=dict(uid=1, frame_index=r['frame'], candidate_count=1,
                                       source_detection_index=0, passed=True))
    m.update(changes or {})
    box = r['bbox']
    return b.assign(track_id=track, feature=feature(r['full'] if full is None else full),
        partial_feature=feature(r['partial'] if partial is None else partial),
        confidence=r['score'], area=(box[2]-box[0])*(box[3]-box[1]),
        frame_index=r['frame'], candidate_count=1, bbox_quality_ok=True, bbox_quality_tier='strong',
        sample_metadata=m, preferred_uid=1 if m['search_reacquire_context_active'] else None,
        preferred_candidate_ok=m['search_reacquire_context_active'])


def seed():
    b = setup()
    for cap in (1253,1255,1257,1259,1261,1262):
        assert send(b, cap) == 0
        assert b.last_assignments[5]['reason'] == 'partial_evidence_tentative'
    assert send(b,1264) == 0
    assert b.last_assignments[5]['reason'] == 'preferred_search_mapped_late_wait'
    assert b.pending_late_handoffs[5].streak == 1
    return b


def test_coverage_change_observes_before_search_without_identity_or_learning():
    b = setup(); anchor = deepcopy(b.identities[1].last_strong_observation)
    for cap in (1253,1255,1257,1259):
        assert send(b,cap) == 0
        assert not b.last_assignments[5]['bank_updated']
        assert b.identities[1].last_strong_observation == anchor
        assert b._candidate_observations.rows[(1,5)]['start_cap'] == 1253
    assert b._candidate_observations.rows[(1,5)]['count'] == 4


def test_recorded_gray_interruption_preserves_bounded_good_sample():
    b = seed()
    original = deepcopy(b.pending_late_handoffs[5])
    for cap in (1265,1266):
        assert send(b,cap) == 0
        assert b.last_assignments[5]['tentative_local_confirmation_preserved']
        pending = b.pending_late_handoffs[5]
        assert pending.streak == 1
        assert pending.capture_timestamp == original.capture_timestamp
    assert send(b,1268) == 1
    assert b.last_assignments[5]['late_candidate_streak'] == 2
    assert not b.last_assignments[5]['bank_updated']
    assert b._reacquire_search_anchors[1] == DATA['protected']
    assert send(b,1270) == 1
    assert b.last_assignments[5]['reason'] == 'mapped_late_continuation'
    assert b._reacquire_quarantine.is_held(1)
    for cap in (1272,1273,1276):
        assert send(b,cap) == 0  # Grey is never promoted to an identity match.
    assert send(b,1278) == 0
    assert send(b,1279) == 1


@pytest.mark.parametrize('partial', [.343, .396, .449])
def test_repeated_gray_never_confirms_or_updates(partial):
    b = setup()
    for cap in ROWS:
        assert send(b,cap,partial=partial) == 0
        assert not b.last_assignments[5]['bank_updated']
    assert b.identities[1].last_strong_observation == DATA['anchor']


@pytest.mark.parametrize('kind', ['conflict', 'missing_frame', 'old_capture', 'long_gap',
                                  'competition', 'yaw_missing', 'new_track', 'direction', 'bad_quality'])
def test_invalid_evidence_cannot_bridge_to_confirmation(kind):
    b = seed()
    if kind == 'missing_frame':
        send(b,1266)
    elif kind == 'conflict':
        send(b,1265,partial=.46); send(b,1266)
    elif kind == 'old_capture':
        send(b,1265,changes={'capture_frame_id':1264,'capture_timestamp':ROWS[1264]['ts']})
        send(b,1266)
    elif kind == 'long_gap':
        send(b,1265,changes={'capture_timestamp':ROWS[1264]['ts']+.5})
    elif kind == 'competition':
        send(b,1265,changes={'identity_competition':dict(uid=1,frame_index=ROWS[1265]['frame'],
            source_detection_index=0,candidate_count=1,passed=False)})
        send(b,1266)
    elif kind == 'yaw_missing':
        send(b,1265,changes={'integrated_yaw_deg':None}); send(b,1266)
    elif kind == 'direction':
        send(b,1265,changes={'search_direction':'left'}); send(b,1266)
    elif kind == 'bad_quality':
        send(b,1265,changes={'quality_bbox_ok':False}); send(b,1266)
    else:
        send(b,1265,track=7); send(b,1266,track=7)
    assert send(b,1268) == 0
    assert not b.last_assignments[5]['bank_updated']


def test_late_continuation_cannot_hide_full_appearance_deterioration():
    b = seed(); send(b,1265); send(b,1266)
    assert send(b,1268) == 1
    assert send(b,1270,full=.40) == 0
    assert not b.last_assignments[5]['bank_updated']


def test_mapped_and_unmapped_use_same_first_observation_requirement():
    for mapped in (True, False):
        b = setup()
        if not mapped: b.track_to_uid.clear()
        assert send(b,1264) == 0
        assert b.pending_late_handoffs[5].streak == 1
        assert not b.last_assignments[5]['bank_updated']


def test_gray_observations_never_extend_good_sample_expiry():
    b = seed()
    original_ts = b.pending_late_handoffs[5].capture_timestamp
    for cap in (1265,1266,1268,1270):
        assert send(b,cap,partial=.39) == 0
        assert b.pending_late_handoffs[5].capture_timestamp == original_ts
        assert b.pending_late_handoffs[5].streak == 1
    assert send(b,1272,partial=.39) == 0
    assert 5 not in b.pending_late_handoffs
