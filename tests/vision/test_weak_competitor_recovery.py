"""Weak background permissions during recovery; synthetic timing/competitors."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection
from test_cap1410_competition_eligibility import setup_tracker, inputs, evidence


def gap_tracker(monkeypatch, *, local=False):
    t, args, features, outputs = setup_tracker(monkeypatch)
    suspect = dict(reason='evidence_gap', track_id=6, streak=0, capture=None, timestamp=None)
    t.identity_bank._reacquire_control_suspects[1] = suspect
    if local:
        suspect.update(streak=1, capture=1409, timestamp=10.05,
                       local_observation=dict(args['reference'], capture_frame_id=1409,
                                              capture_timestamp=10.05))
        # Old identity reference is stale; qualified recovery observation is not.
        t.identity_bank.identities[1].last_strong_observation = dict(args['reference'], capture_timestamp=8.)
    return t, args, features, outputs


def proof(t, args, features, outputs):
    return t._frame_identity_competition(args['detections'], features, outputs=outputs,
                                        image_width=640, image_height=480)


@pytest.mark.parametrize('local', [False, True])
@pytest.mark.parametrize('mapped', [False, True])
def test_weak_competitor_no_longer_vetoes_known_gap_recovery(monkeypatch, local, mapped):
    t, args, features, outputs = gap_tracker(monkeypatch, local=local)
    if not mapped:
        t.identity_bank.track_to_uid.clear()
    before = deepcopy(t.identity_bank._reacquire_control_suspects)
    p = proof(t, args, features, outputs)
    assert p[0]['passed'] and p[1]['observation_only']
    assert p[0]['eligibility_reference_kind'] == (
        'qualified_gap_observation' if local else 'gap_identity_reference')
    assert t.identity_bank._reacquire_control_suspects == before


@pytest.mark.parametrize('score', [.5, .6, .95])
def test_recovery_hypothesis_cannot_dismiss_confident_person(monkeypatch, score):
    t, args, features, outputs = gap_tracker(monkeypatch, local=True)
    args['detections'][1] = Detection(args['detections'][1].bbox, score, 0)
    assert not proof(t, args, features, outputs)[0]['passed']


@pytest.mark.parametrize('reason', ['appearance_conflict', 'partial_conflict', 'legacy_strict', None])
def test_explicit_or_unknown_conflict_cannot_dismiss_competitor(monkeypatch, reason):
    t, args, features, outputs = gap_tracker(monkeypatch, local=True)
    t.identity_bank._reacquire_control_suspects[1]['reason'] = reason
    assert not proof(t, args, features, outputs)[0]['passed']


@pytest.mark.parametrize('change', ['stale', 'duplicate', 'turn', 'revoked', 'geometry_conflict',
                                    'different_owner', 'different_track', 'unqualified_local'])
def test_recovery_exclusion_cannot_bypass_reference_or_identity_checks(monkeypatch, change):
    t, args, features, outputs = gap_tracker(monkeypatch, local=True)
    suspect = t.identity_bank._reacquire_control_suspects[1]
    if change == 'stale': t._frame_context['capture_timestamp'] = 10.5
    if change == 'duplicate': t._frame_context['capture_frame_id'] = 1409
    if change == 'turn': t._frame_context['integrated_yaw_deg'] = 6.
    if change == 'revoked': t.identity_bank._geometry_revoked_uids[1] = 699
    if change == 'geometry_conflict': t.identity_bank._mapped_geometry_conflicts[6] = {'uid': 1}
    if change == 'different_owner': t.identity_bank.track_to_uid[6] = 2
    if change == 'different_track': suspect['track_id'] = 7
    if change == 'unqualified_local': suspect['streak'] = 0
    assert not proof(t, args, features, outputs)[0]['passed']


@pytest.mark.parametrize('box', [(10, 190, 90, 300), (10, 150, 50, 280)])
def test_gap_requires_absolute_smallness_not_just_relative_size(monkeypatch, box):
    t, args, features, outputs = gap_tracker(monkeypatch)
    args['detections'][1] = Detection(box, .4, 0)
    assert not proof(t, args, features, outputs)[0]['passed']


@pytest.mark.parametrize('search', [False, True])
def test_observation_only_never_enrolls_or_updates_identity(search):
    t = DeepSortTracker(DeepSortTrackerConfig(identity_new_confirm_frames=1))
    bank = t.identity_bank
    feature = np.array([1., 0., 0.], dtype=np.float32)
    assert bank.assign(track_id=6, feature=feature, confidence=.9, area=60000, frame_index=1) == 1
    p, _ = evidence(inputs())
    bank.pending_new[6] = object()
    sentinel = bank.pending_new[6]
    bank._reacquire_control_suspects[1] = dict(track_id=6, reason='evidence_gap', streak=1)
    count = len(bank.identities[1].features)
    for frame in (700, 701):
        current = dict(p[1], frame_index=frame)
        assert bank.assign(track_id=8, feature=feature, confidence=.95, area=5000,
            frame_index=frame, candidate_count=2, preferred_uid=1, preferred_candidate_ok=True,
            sample_metadata={'is_fresh': True, 'source_detection_index': 1,
                'search_reacquire_context_active': search, 'identity_competition': current}) == 0
        assert bank.last_assignments[8]['reason'] == 'weak_small_observation_only'
    assert bank.pending_new[6] is sentinel
    assert bank._reacquire_control_suspects[1]['streak'] == 1
    assert 8 not in bank.track_to_uid and len(bank.identities) == 1
    assert len(bank.identities[1].features) == count


@pytest.mark.parametrize('changes', [
    {'frame_index': 1}, {'uid': 99}, {'candidate_count': 3},
    {'source_detection_index': 0}, {'observation_only': False},
])
def test_stale_or_foreign_label_is_not_applied(changes):
    t = DeepSortTracker(DeepSortTrackerConfig(identity_new_confirm_frames=1))
    bank = t.identity_bank; feature = np.array([1., 0., 0.], dtype=np.float32)
    bank.assign(track_id=6, feature=feature, confidence=.9, area=60000, frame_index=1)
    p, _ = evidence(inputs())
    bank.assign(track_id=8, feature=feature, confidence=.4, area=4000, frame_index=700,
        candidate_count=2, preferred_uid=1, sample_metadata={
            'is_fresh': True, 'source_detection_index': 1,
            'identity_competition': dict(p[1], **changes)})
    assert bank.last_assignments[8]['reason'] != 'weak_small_observation_only'


def test_real_bank_gap_recovers_in_two_frames_despite_weak_boxes(monkeypatch):
    from test_cap1254_identity_competition import before_loss, meta, ROWS, send
    bank = before_loss()
    assert send(bank, 1254, changes={'identity_competition': None}) == 0
    assert bank._reacquire_control_suspects[1]['reason'] == 'evidence_gap'
    t = DeepSortTracker(DeepSortTrackerConfig())
    t.identity_bank = bank
    t.set_search_reacquire_context(active_uid=1, searching=True, direction='left')
    for cap, timestamp, expected in ((1256, 7235.81, 0), (1259, ROWS[1259]['ts'], 1)):
        row = ROWS[cap]
        t._frame_context = dict(meta(row), capture_timestamp=timestamp)
        t._frame_index = row['frame']
        detections = [Detection(row['bbox'], row['score'], 0)] + [
            Detection((550+i*35, 210, 580+i*35, 300), .4, 0)
            for i in range(row['count']-1)]
        features = [object() for _ in detections]
        monkeypatch.setattr(t, 'reid_distance_to_uid', lambda uid, f:
                            .21 if f is features[0] else .23)
        p = t._frame_identity_competition(detections, features,
            outputs=[SimpleNamespace(track_id=2, source_detection_index=0, time_since_update=0)],
            image_width=640, image_height=480)
        assert p[0]['passed']
        assert send(bank, cap, changes={'capture_timestamp': timestamp,
                                      'identity_competition': p[0]}) == expected
        assert not bank.last_assignments[2]['bank_updated']
        assert bank._reacquire_control_suspects[1]['streak'] == (1 if expected == 0 else 2)
    assert p[0]['eligibility_reference_kind'] == 'qualified_gap_observation'
    assert bank._reacquire_quarantine.is_held(1)
