"""Recorded CAP1410 boxes/distances, synthetic fresh prior and model outputs.

This isolates eligibility; it is not a replay of the closed-loop identity run.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rk_vision.candidate_competition import competition_evidence
from rk_vision.competition_eligibility import anchored_competitor_exclusions
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


BOX = (285.7453, 56.6426, 452.3411, 476.2439)
SMALL = (135.3, 218.4, 184.8, 302.9)


def inputs():
    return dict(
        detections=[Detection(BOX, .9343, 0), Detection(SMALL, .3726, 0)],
        distances={0: .1484709382, 1: .1783425808},
        reference=dict(capture_frame_id=1408, capture_timestamp=10.,
                       integrated_yaw_deg=0., bbox=BOX, geometry_source='detector', track_id=6),
        context=dict(capture_frame_id=1410, capture_timestamp=10.1, integrated_yaw_deg=.2),
        mapped_indices={0}, width=640, height=480, confidence_floor=.65,
        track_confidence_floor=.5, appearance_limit=.30,
    )


def evidence(args):
    excluded, anchor = anchored_competitor_exclusions(**args)
    return competition_evidence(args['distances'], uid=1, frame_index=700,
        exclusions=excluded, eligibility_reference_cap=1408), anchor


def test_cap1410_small_observation_cannot_veto_fresh_continuous_target():
    args = inputs(); before = deepcopy(args)
    assert not competition_evidence(args['distances'], uid=1, frame_index=700)[0]['passed']
    proof, anchor = evidence(args)
    assert anchor == 0 and proof[0]['passed'] and not proof[1]['passed']
    assert proof[0]['candidate_count'] == 2  # provenance still includes both detections
    assert proof[0]['qualified_candidate_count'] == 1
    assert proof[1]['reason'] == 'ineligible_uid_competitor'
    assert proof[0]['excluded_competitors'] == {1: 'weak_small_observation'}
    assert args == before  # never remove detections or mutate the trusted reference


def test_high_confidence_small_background_requires_same_geometric_evidence():
    args = inputs(); args['detections'][1] = Detection(SMALL, .8, 0)
    proof, _ = evidence(args)
    assert proof[0]['passed']
    assert proof[0]['excluded_competitors'] == {1: 'uid_scale_position_conflict'}


@pytest.mark.parametrize('change', [
    {'mapped_indices': set()}, {'reference': None}, {'width': None},
    {'width': float('nan')}, {'height': 0}, {'appearance_limit': .1},
    {'confidence_floor': .99},
])
def test_without_qualified_positive_reference_original_veto_remains(change):
    args = inputs(); args.update(change)
    proof, _ = evidence(args)
    assert not proof[0]['passed'] and 'qualified_candidate_count' not in proof[0]


@pytest.mark.parametrize('change', [
    {'capture_timestamp': 10.36}, {'capture_timestamp': 9.9},
    {'capture_timestamp': float('nan')}, {'capture_frame_id': 1408},
    {'integrated_yaw_deg': 5.1}, {'is_fresh': False},
])
def test_stale_duplicate_turning_reference_cannot_exclude(change):
    args = inputs(); args['context'].update(change)
    assert not evidence(args)[0][0]['passed']


@pytest.mark.parametrize('box', [
    (60, 60, 230, 475),  # similarly sized other person, even without a track
    (300, 220, 350, 305),  # small but overlaps: crossing/occlusion ambiguous
    (300, 0, 350, 50),  # separated vertically, not enough lateral separation
    (float('nan'), 0, 100, 300),
])
def test_real_or_uncertain_competitors_keep_veto(box):
    args = inputs(); args['detections'][1] = Detection(box, .8, 0)
    assert not evidence(args)[0][0]['passed']


def test_small_target_not_globally_removed():
    args = inputs()
    args['reference']['bbox'] = SMALL
    args['detections'] = [Detection(SMALL, .95, 0), Detection((400, 220, 450, 305), .8, 0)]
    proof, _ = evidence(args)
    assert not proof[0]['passed'] and proof[0]['candidate_count'] == 2


def test_missing_feature_of_plausible_competitor_remains_uncertainty():
    args = inputs(); args['detections'][1] = Detection((60, 60, 230, 475), .8, 0)
    args['distances'][1] = None
    assert evidence(args)[0][0]['reason'] == 'missing_competition_feature'


def setup_tracker(monkeypatch):
    args = inputs()
    tracker = DeepSortTracker(DeepSortTrackerConfig(identity_new_confirm_frames=1))
    feature = np.array([1., 0., 0.], dtype=np.float32)
    assert tracker.identity_bank.assign(track_id=6, feature=feature, confidence=.95,
                                        area=60000, frame_index=1) == 1
    tracker.identity_bank.identities[1].last_strong_observation = args['reference']
    tracker._frame_context = args['context']
    tracker._frame_index = 700
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction='right')
    features = [object(), object()]
    monkeypatch.setattr(tracker, 'reid_distance_to_uid', lambda uid, f:
                        args['distances'][features.index(f)])
    outputs = [SimpleNamespace(track_id=6, source_detection_index=0, time_since_update=0)]
    return tracker, args, features, outputs


@pytest.mark.parametrize('block', ['none', 'unmapped', 'new_raw_track', 'conflict', 'revoked', 'suspect'])
def test_real_tracker_provenance_and_negative_evidence(monkeypatch, block):
    t, args, features, outputs = setup_tracker(monkeypatch)
    if block == 'unmapped': t.identity_bank.track_to_uid.clear()
    if block == 'new_raw_track': outputs[0].track_id = 7; t.identity_bank.track_to_uid[7] = 1
    if block == 'conflict': t.identity_bank._mapped_geometry_conflicts[6] = {'uid': 1}
    if block == 'revoked': t.identity_bank._geometry_revoked_uids[1] = 699
    if block == 'suspect': t.identity_bank._reacquire_control_suspects[1] = {'reason': 'appearance_conflict'}
    proof = t._frame_identity_competition(args['detections'], features,
        outputs=outputs, image_width=640, image_height=480)
    assert proof[0]['passed'] is (block == 'none')
    assert proof[0]['candidate_count'] == 2


def test_actual_update_passes_dimensions_and_outputs_before_assignment(monkeypatch):
    t, args, features, outputs = setup_tracker(monkeypatch)
    monkeypatch.setattr(t.deepsort, 'update', lambda *a, **k: outputs)
    monkeypatch.setattr(t, '_duplicate_identity_track_ids', lambda *a: set())
    monkeypatch.setattr(t, '_identity_swap_track_ids', lambda *a: set())
    monkeypatch.setattr(t, '_observe_identity_frame_evidence', lambda *a, **k: None)
    captured = []
    def record(out, *a, **k):
        captured.append(deepcopy(t._identity_competition))
        return out
    monkeypatch.setattr(t, '_to_record', record)
    t.update(args['detections'], features, image_width=640, image_height=480,
             frame_context=args['context'])
    assert captured[0][0]['passed'] and not captured[0][1]['passed']


def test_detector_only_probe_has_no_self_nominated_reference(monkeypatch):
    t, args, features, _ = setup_tracker(monkeypatch)
    proof = t._frame_identity_competition(args['detections'], features)
    assert not proof[0]['passed']


def test_excluded_candidate_cannot_bind_or_write_templates():
    t = DeepSortTracker(DeepSortTrackerConfig(identity_new_confirm_frames=1))
    feature = np.array([1., 0., 0.], dtype=np.float32)
    assert t.identity_bank.assign(track_id=6, feature=feature, confidence=.95,
                                  area=60000, frame_index=1) == 1
    proof, _ = evidence(inputs())
    count = len(t.identity_bank.identities[1].features)
    assert t.identity_bank.assign(track_id=8, feature=feature, confidence=.9, area=4000,
        frame_index=700, preferred_uid=1, preferred_candidate_ok=True, candidate_count=2,
        sample_metadata={'is_fresh': True, 'search_reacquire_context_active': True,
                         'identity_competition': proof[1]}) == 0
    assert not t.identity_bank.last_assignments[8]['bank_updated']
    assert len(t.identity_bank.identities[1].features) == count


def test_proof_reaches_real_quarantine_verifier_without_releasing_templates(monkeypatch):
    # Existing real recorded continuation fixture, with a synthetic background
    # rival: verify the new proof's raw count/index survives bank validation.
    from test_cap1254_identity_competition import before_loss, meta, ROWS, send
    bank = before_loss()
    row = ROWS[1254]
    metadata = meta(row)
    tracker = DeepSortTracker(DeepSortTrackerConfig())
    tracker.identity_bank = bank
    tracker._frame_context = metadata
    tracker._frame_index = row['frame']
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction='right')
    x1, y1, x2, y2 = row['bbox']
    # Put small rival on the opposite edge from this recorded large target.
    left = 0 if (x1+x2)/2 > 320 else 600
    detections = [Detection(row['bbox'], row['score'], 0),
                  Detection((left, 210, left+30, 300), .4, 0)]
    features = [object(), object()]
    monkeypatch.setattr(tracker, 'reid_distance_to_uid', lambda uid, f:
                        .21 if f is features[0] else .23)
    outputs = [SimpleNamespace(track_id=2, source_detection_index=0, time_since_update=0)]
    proof = tracker._frame_identity_competition(detections, features, outputs=outputs,
                                               image_width=640, image_height=480)
    assert proof[0]['passed'] and proof[0]['candidate_count'] == row['count']
    assert send(bank, 1254, changes={'identity_competition': proof[0]}) == 1
    assert bank.last_assignments[2]['reacquire_control_competition_ok']
    assert not bank.last_assignments[2]['bank_updated']
    assert bank._reacquire_quarantine.is_held(1)
