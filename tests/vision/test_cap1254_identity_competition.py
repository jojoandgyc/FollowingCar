"""Logged geometry/scores with synthetic embeddings, not a motor/model replay."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, _geometry_observation
from test_cap874_identity_reacquire import feature, metadata

DATA = json.loads((Path(__file__).parent / 'fixtures/cap1254_reacquire.json').read_text())
ROWS = {r['cap']: r for r in DATA['rows']}


def meta(row):
    m = metadata(row['cap'], row['ts'], row['bbox'], row['yaw'], search=row['search'])
    m.update(frame_index=row['frame'], track_id=2, partial_feature_source='osnet_torso',
             partial_observation=True, candidate_count=row['count'], candidate_score_gap=row['gap'],
             detector_edge_touch_count=row['edges'], search_direction='left' if row['search'] else None,
             search_direction_compatible=row['compatible'], source_detection_index=0)
    m['identity_competition'] = dict(uid=1, frame_index=row['frame'], passed=True,
                                     candidate_count=row['count'], source_detection_index=0)
    return m


def setup():
    b = IdentityBank(IdentityBankConfig(template_memory_enable=True, template_crosscheck_enable=True,
        mapped_verify_threshold=.45, partial_match_threshold=.45,
        preferred_search_reacquire_min_score_gap=.25, controlled_handoff_enable=True,
        camera_hfov_deg=60.))
    a = DATA['anchor']
    m = metadata(a['capture_frame_id'], a['capture_timestamp'], a['bbox'], a['integrated_yaw_deg'], search=False)
    m.update(partial_feature_source='osnet_torso', partial_observation=True)
    b._create_identity(feature(0), a['frame_index'], m, feature(0))
    b.identities[1].last_strong_observation = deepcopy(a)
    r = ROWS[1236]
    b._bind_reacquired_identity(1, 2, r['frame'], meta(r))
    b.identities[1].last_strong_observation = _geometry_observation(meta(r), r['frame'])
    return b


def send(b, cap, *, changes=None, full=None, part=None):
    r = ROWS[cap]; m = meta(r); m.update(changes or {})
    box = r['bbox']
    return b.assign(track_id=2, feature=feature(r['full'] if full is None else full),
        partial_feature=feature(r['part'] if part is None else part), confidence=r['score'],
        area=(box[2]-box[0])*(box[3]-box[1]), frame_index=r['frame'],
        candidate_count=r['count'], bbox_quality_ok=True, bbox_quality_tier='strong',
        sample_metadata=m, preferred_uid=1 if m['search_reacquire_context_active'] else None,
        preferred_candidate_ok=m.get('search_direction_compatible') is not False)


def before_loss():
    b = setup()
    for cap in (1238,1239,1240,1242,1244,1245,1247,1249):
        assert send(b,cap) == 1
    return b


def test_cap1254_uses_identity_competition_not_yolo_separation():
    b = before_loss()
    for cap in (1254,1256,1259):
        assert send(b,cap) == 1
        a = b.last_assignments[2]
        assert a['reacquire_control_competition_reason'] == 'passed'
        assert not a['bank_updated']
    assert b._reacquire_quarantine.is_held(1)
    assert b._reacquire_search_anchors[1]['capture_frame_id'] == 1065


def test_recorded_search_transition_seeds_then_recovers_and_continues():
    b = before_loss()
    for cap in (1254,1256,1259):
        assert send(b,cap) == 1
    # Replay even the original search transition, although its trigger is fixed.
    assert send(b,1261) == 0
    assert b.last_assignments[2]['reacquire_control_seeded']
    assert b.last_assignments[2]['reacquire_control_suspect_reason'] == 'evidence_gap'
    for cap in sorted(c for c in ROWS if 1263 <= c <= 1295):
        assert send(b,cap) == 1, (cap,b.last_assignments[2])
        assert not b.last_assignments[2]['bank_updated']
    assert b._reacquire_search_anchors[1]['capture_frame_id'] == 1065


@pytest.mark.parametrize('proof', [None, {}, {'passed':False}, {'uid':99},
    {'frame_index':1}, {'source_detection_index':1}, {'candidate_count':99}])
def test_invalid_identity_proof_not_rescued_by_large_yolo_gap(proof):
    b = before_loss(); m = meta(ROWS[1254])
    p = None if proof is None else dict(m['identity_competition'], **proof) if proof else {}
    assert send(b,1254,changes={'identity_competition':p,'candidate_score_gap':.99}) == 0
    assert b.last_assignments[2]['reacquire_control_competition_ok'] is False
    assert not b.last_assignments[2]['bank_updated']


def test_missing_competition_can_recover_in_two_new_frames_with_reliable_torso():
    b = before_loss()
    assert send(b,1254,changes={'identity_competition':None}) == 0
    assert b._reacquire_control_suspects[1]['reason'] == 'evidence_gap'
    assert send(b,1256) == 0
    assert send(b,1259) == 1
    assert b.last_assignments[2]['reacquire_control_recovery_source'] == 'partial'
    assert not b.last_assignments[2]['bank_updated']


@pytest.mark.parametrize('reason', ['appearance_conflict','partial_conflict','legacy_strict'])
def test_explicit_or_unknown_suspicion_cannot_use_partial_gap_recovery(reason):
    b = before_loss()
    b._reacquire_control_suspects[1] = dict(track_id=2,streak=0,capture=1249,
        timestamp=ROWS[1249]['ts'],reason=reason)
    for cap in (1254,1256,1259,1261,1263):
        assert send(b,cap) == 0
        assert b.last_assignments[2]['reacquire_control_recovery_source'] == 'strong'


def test_torso_conflict_escalates_gap_and_cannot_be_forgotten():
    b = before_loss()
    assert send(b,1254,changes={'identity_competition':None}) == 0
    assert send(b,1256,part=.46) == 0
    assert b._reacquire_control_suspects[1]['reason'] == 'partial_conflict'
    assert send(b,1259) == 0


def test_duplicate_does_not_complete_partial_recovery():
    b = before_loss();send(b,1254,changes={'identity_competition':None})
    assert send(b,1256) == 0
    assert send(b,1256) == 0
    assert b._reacquire_control_suspects[1]['streak'] == 1
    assert send(b,1259) == 1


def test_retained_geometric_conflict_blocks_partial_recovery():
    b = before_loss()
    b._mapped_geometry_conflicts[2] = dict(uid=1,search_contradiction=True,
        reference=deepcopy(DATA['anchor']),candidate=_geometry_observation(meta(ROWS[1249]),626))
    for cap in (1254,1256):
        assert send(b,cap) == 0
        assert b.last_assignments[2]['reason'] == 'mapped_geometry_reject'


def test_old_detector_gate_reproduces_1249_accept_1254_reject(monkeypatch):
    b = before_loss()
    monkeypatch.setattr(b, '_reacquire_competition', lambda uid, frame, count, m:
                        (b._candidate_competition_ok(count, m), 'old_detector_gap'))
    assert send(b,1254) == 0
    assert b.last_assignments[2]['reason'] == 'reacquire_control_verify_reject'


@pytest.mark.parametrize('changes', [
    {'is_fresh':False}, {'capture_timestamp':None},
    {'integrated_yaw_deg':None}, {'detector_edge_touch_count':3},
])
def test_invalid_sample_cannot_complete_gap_recovery(changes):
    b = before_loss(); send(b,1254,changes={'identity_competition':None})
    assert send(b,1256,changes=changes) == 0
    assert send(b,1259) == 0
    assert not b.last_assignments[2]['bank_updated']


def test_long_gap_restarts_confirmation_without_learning():
    b = before_loss();send(b,1254,changes={'identity_competition':None})
    assert send(b,1256) == 0
    # Jump to the real later search frame: first local proof is already stale.
    assert send(b,1263) == 0
    assert b._reacquire_control_suspects[1]['streak'] == 1
    assert send(b,1264) == 1
    assert not b.last_assignments[2]['bank_updated']


def test_grey_torso_cannot_complete_gap_recovery():
    b = before_loss();send(b,1254,changes={'identity_competition':None})
    assert send(b,1256,part=.396) == 0
    assert b.last_assignments[2]['reason'] == 'partial_evidence_tentative'
    assert send(b,1259,part=.396) == 0


def test_full_mismatch_escalates_gap_even_if_torso_matches():
    b = before_loss();send(b,1254,changes={'identity_competition':None})
    assert send(b,1256,full=.46) == 0
    assert b._reacquire_control_suspects[1]['reason'] == 'appearance_conflict'
    assert send(b,1259) == 0
