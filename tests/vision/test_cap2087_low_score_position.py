"""Low-score position publication without UID, gallery or motor renewal."""
from copy import deepcopy

import numpy as np
import pytest

from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection
from test_identity_bank import _x_axis_feature_at_distance
from tools.replay_cap334_recovery import gallery_snapshot


BOX2083 = (232.21481323242188, 2.2886962890625, 560.474609375, 471.561767578125)
BOX2087 = (114.33340454101562, 2.667449951171875, 433.2265625, 476.3291015625)


def update(tracker, cap, stamp, *, box=BOX2087, score=.4483298957,
           yaw=143.337199323, distance=.0926066637):
    return tracker.update([Detection(box, score, 0)], [_x_axis_feature_at_distance(distance)],
        image_width=640, image_height=480, frame_context=dict(
            capture_frame_id=cap, capture_timestamp=stamp, integrated_yaw_deg=yaw))


def ready():
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1, bbox_expand_scale=1.,
        hfov_deg=60., identity_new_confirm_frames=1, identity_update_interval=1))
    tracker.set_detector_continuation_context(active_uid=1, allowed=True)
    for cap, stamp in ((2077, 10.), (2079, 10.05), (2081, 10.10), (2083, 10.15)):
        records = update(tracker, cap, stamp, box=BOX2083, score=.8853573,
                         yaw=142.251944922, distance=0.)
    assert records[0].reid_uid == 1
    assert tracker._low_score_position_anchor['clock'] == (2083, 10.15)
    return tracker


def evidence(tracker):
    return next((row['assignment']['low_score_position_evidence']
                 for row in tracker.last_identity_observations
                 if row['assignment'].get('low_score_position_evidence')), None)


def test_cap2087_real_association_publishes_position_but_no_formal_record_or_learning():
    tracker = ready()
    gallery = gallery_snapshot(tracker.identity_bank)
    ds_gallery = deepcopy(tracker.deepsort.tracker.metric.samples)
    anchor = tracker.identity_bank.identities[1].last_strong_observation
    assert update(tracker, 2087, 10.355166410) == []
    proof = evidence(tracker)
    assert proof['uid'] == 1 and proof['track_id'] == 1
    assert proof['bbox'] == BOX2087
    assert proof['capture_frame_id'] == 2087
    assert proof['reference_capture_frame_id'] == 2083
    assert proof['expires_at'] == pytest.approx(10.65)
    assert proof['appearance_distance'] == pytest.approx(.0926066637, abs=1e-6)
    assert proof['center_jump_ratio'] == pytest.approx(.1734198138, abs=1e-6)
    assert not proof['identity_authorized'] and not proof['learning_allowed']
    assert tracker.last_identity_observations[0]['uid'] == 0
    assert tracker.identity_bank.last_assignments[1]['identity_control_rejected']
    assert 'low_score_position_evidence' not in tracker.identity_bank.last_assignments[1]
    assert 'low_score_position_diagnostics' not in tracker.identity_bank.last_assignments[1]
    diagnostic = tracker.last_identity_observations[0]['assignment']['low_score_position_diagnostics']
    assert diagnostic['reference_capture_frame_id'] == 2083
    assert diagnostic['appearance_distance'] == pytest.approx(.0926066637, abs=1e-6)
    assert diagnostic['center_jump_ratio'] == pytest.approx(.1734198138, abs=1e-6)
    assert tracker.identity_bank.identities[1].last_strong_observation is anchor
    assert gallery_snapshot(tracker.identity_bank) == gallery
    for raw in ds_gallery:
        np.testing.assert_array_equal(tracker.deepsort.tracker.metric.samples[raw], ds_gallery[raw])


def test_repeated_weak_frames_cannot_renew_independent_deadline_or_duplicate_position():
    tracker = ready()
    for cap, stamp in ((2087, 10.36), (2089, 10.5), (2091, 10.64)):
        assert update(tracker, cap, stamp) == []
        assert evidence(tracker)['expires_at'] == pytest.approx(10.65)
        assert tracker._low_score_position_anchor['clock'] == (2083, 10.15)
    update(tracker, 2091, 10.64)
    assert evidence(tracker) is None
    update(tracker, 2093, 10.66)
    assert evidence(tracker) is None


def test_rejected_low_score_emits_reason_only_on_observation_copy(caplog):
    tracker = ready()
    tracker._low_score_position_anchor = None
    with caplog.at_level('INFO', logger='rk_vision.tracker'):
        update(tracker, 2087, 10.355166410)
    assignment = tracker.last_identity_observations[0]['assignment']
    assert assignment['low_score_position_reject_reason'] == 'trusted_anchor_unavailable'
    assert 'low_score_position_reject_reason' not in tracker.identity_bank.last_assignments[1]
    messages = [r.message for r in caplog.records if r.message.startswith('low_score_position_evidence ')]
    assert len(messages) == 1
    assert 'accepted=False reason=trusted_anchor_unavailable' in messages[0]
    assert 'reference_capture=None' in messages[0]


@pytest.mark.parametrize('failure', ['active_uid', 'mapping', 'identity_object', 'anchor_missing',
    'geometry_conflict', 'revoked', 'suspect', 'excluded', 'different_person', 'wrong_side',
    'small_area', 'competition', 'bad_provenance', 'stale_context'])
def test_position_proof_cannot_override_identity_or_observation_failure(monkeypatch, failure):
    tracker = ready()
    bank = tracker.identity_bank
    options = {}
    if failure == 'active_uid': tracker.set_detector_continuation_context(active_uid=2, allowed=True)
    elif failure == 'mapping': bank.track_to_uid[1] = 2
    elif failure == 'identity_object': bank.identities[1] = deepcopy(bank.identities[1])
    elif failure == 'anchor_missing': tracker._low_score_position_anchor = None
    elif failure == 'geometry_conflict': bank._mapped_geometry_conflicts[1] = dict(uid=1)
    elif failure == 'revoked': bank._geometry_revoked_uids[1] = dict(reason='test')
    elif failure == 'suspect': bank._reacquire_control_suspects[1] = dict(reason='test')
    elif failure == 'excluded':
        monkeypatch.setattr(bank, 'search_exclusion_for', lambda *a, **kw: dict(reason='other_person'))
    elif failure == 'different_person': options['distance'] = .7
    elif failure == 'wrong_side': options['box'] = (0., 2., 200., 471.)
    elif failure == 'small_area': options['box'] = (240., 30., 300., 170.)
    else:
        original = tracker._low_score_position_evidence
        def injected(output, metadata, assignment):
            if failure == 'competition': metadata['identity_competition']['passed'] = False
            elif failure == 'bad_provenance': metadata['association_reason'] = 'predicted'
            else: metadata['capture_timestamp'] += .01
            return original(output, metadata, assignment)
        monkeypatch.setattr(tracker, '_low_score_position_evidence', injected)
    update(tracker, 2087, 10.355166410, **options)
    assert evidence(tracker) is None


def test_competition_failure_clears_anchor_instead_of_weak_self_recovery(monkeypatch):
    tracker = ready()
    original = tracker._low_score_position_evidence
    def injected(output, metadata, assignment):
        metadata['identity_competition']['passed'] = False
        return original(output, metadata, assignment)
    monkeypatch.setattr(tracker, '_low_score_position_evidence', injected)
    update(tracker, 2087, 10.35)
    assert tracker._low_score_position_anchor is None
    monkeypatch.undo()
    update(tracker, 2089, 10.4)
    assert evidence(tracker) is None


def test_full_rejection_clears_anchor_and_reset_forgets_position():
    tracker = ready()
    update(tracker, 2085, 10.25, box=BOX2083, score=.95, distance=.99)
    assert tracker._low_score_position_anchor is None
    update(tracker, 2087, 10.35)
    assert evidence(tracker) is None
    tracker.reset()
    assert tracker._low_score_position_anchor is None
    assert tracker._low_score_position_last is None


def test_uncertain_similar_follow_cannot_seed_trusted_position_anchor():
    from test_provisional_association import activated, update as candidate_update
    tracker, _ = activated()
    assert getattr(tracker, '_low_score_position_anchor', None) is None
    assert candidate_update(tracker, 22, 3.2, score=.4483) == []
    assert evidence(tracker) is None


@pytest.mark.parametrize('kind,expected', [
    ('competition', 'competition_conflict'), ('excluded', 'identity_excluded'),
    ('retained_geometry', 'geometry_conflict'), ('mapped_geometry', 'geometry_conflict'),
    ('ordinary_low_score', None), ('missing_features', None), ('stale_reference', None),
    ('foreign_uid', None), ('foreign_raw', None), ('old_capture', None), ('old_time', None),
])
def test_readonly_contradiction_query_binds_current_negative_evidence(kind, expected):
    import pickle
    tracker = ready()
    update(tracker, 2087, 10.355166410)
    row = tracker.last_identity_observations[0]
    metadata, assignment = row['sample_metadata'], row['assignment']
    if kind in ('competition', 'foreign_uid', 'foreign_raw', 'old_capture', 'old_time', 'missing_features'):
        metadata['identity_competition'].update(passed=False, reason='reid_margin_insufficient',
            distance=.18, competitor_distance=.19)
    if kind == 'excluded': assignment['search_excluded'] = True
    elif kind == 'retained_geometry': assignment['search_contradiction_retained'] = True
    elif kind == 'mapped_geometry': assignment['reason'] = 'mapped_geometry_reject'
    elif kind == 'missing_features': metadata['identity_competition']['reason'] = 'missing_competition_feature'
    elif kind == 'stale_reference': assignment.update(reacquire_geometry_ok=None, reacquire_geometry_reason='stale_reference')
    elif kind == 'foreign_uid': metadata['identity_competition']['uid'] = 2
    elif kind == 'foreign_raw': row['raw_track_id'] = 2
    elif kind == 'old_capture': metadata['capture_frame_id'] = 2085
    elif kind == 'old_time': metadata['capture_timestamp'] = 10.3
    before = pickle.dumps(tracker.identity_bank)
    observations = deepcopy(tracker.last_identity_observations)
    assert tracker.associated_position_contradiction(1, 1, 2087, 10.355166410) == expected
    assert pickle.dumps(tracker.identity_bank) == before
    assert tracker.last_identity_observations == observations


@pytest.mark.parametrize('kind,expected', [('revoked', 'uid_geometry_revoked'),
    ('held', 'geometry_conflict'), ('other_uid', None), ('other_raw', None),
    ('exclusion', 'identity_excluded')])
def test_contradiction_query_checks_active_bank_negative_without_new_record(monkeypatch, kind, expected):
    tracker = ready()
    tracker.last_identity_observations = []
    bank = tracker.identity_bank
    if kind == 'revoked': bank._geometry_revoked_uids[1] = 4
    elif kind == 'held': bank._mapped_geometry_conflicts[1] = dict(uid=1)
    elif kind == 'other_uid': bank._mapped_geometry_conflicts[1] = dict(uid=2)
    elif kind == 'other_raw': bank._mapped_geometry_conflicts[2] = dict(uid=1)
    else:
        monkeypatch.setattr(bank, 'search_exclusion_for', lambda raw, uid, **kw:
                            dict(reason='co_visible_distinct_person') if (raw,uid)==(1,1) else None)
    assert tracker.associated_position_contradiction(1, 1, 2087, 10.35) == expected


@pytest.mark.parametrize('arguments', [(True,1,2087,10.35), (1,False,2087,10.35),
    (1,1,True,10.35), (1,1,2087,float('nan')), (1,1,2087,-1.), (0,1,2087,10.35)])
def test_contradiction_query_rejects_malformed_provenance(arguments):
    tracker = ready()
    tracker.identity_bank._geometry_revoked_uids[1] = 4
    assert tracker.associated_position_contradiction(*arguments) is None
