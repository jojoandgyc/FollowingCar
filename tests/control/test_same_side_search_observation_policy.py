"""Current low-score side observations defer a look, never gain motor identity.

CAP1076 has real association/gallery provenance; CAP1098 has no tracker output.
These distinct routes must not wash out a current rejected identity record.
All producers below use synthetic descriptors and no hardware.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from car_control_modular.search_observation_retry import (
    detector_only_side_search_observation, side_crop_search_observation,
)
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


BOX1076 = (0., 10.09259033203125, 167.33200073242188, 473.22906494140625)
BOX1098 = (0., 32.54167175292969, 73.10037231445312, 471.543701171875)


def evidence(mirror=False):
    box = BOX1076
    if mirror:
        box = (640-box[2], box[1], 640-box[0], box[3])
    m = dict(capture_frame_id=1076, capture_timestamp=45091.713519455,
        control_frame_id=458, source_detection_index=0, is_fresh=True,
        detector_confidence=.4332599639892578, quality_bbox_ok=True,
        bbox_quality_tier='strong', quality_bbox_reason='',
        display_bbox_quality_ok=False, display_bbox_quality_reason='edge_touch>2',
        low_score_continuation=True, association_reason='low_score_existing_track',
        association_confidence_limit=.5, association_previous_capture_frame_id=1074,
        association_previous_capture_timestamp=45091.613463366,
        search_direction_compatible=True)
    a = dict(uid=0, mapped_uid=1, reason='low_score_observation_rejected',
        bbox_quality_ok=False, bbox_quality_tier='weak', identity_control_rejected=True,
        bank_updated=False, recent_bank_updated=False, learning_written_tiers=[],
        low_score_observation_blocked=False,
        match_evidence=dict(matched_uid=1, match_source='strong',
                            strong_distance=.2323504090309143),
        identity_competition=dict(uid=1, frame_index=458, source_detection_index=0,
                                  candidate_count=1, passed=True))
    return a, m, box, 'right' if mirror else 'left'


def supported(a, m, box, direction='left'):
    return side_crop_search_observation(a, m, 0, 1, box, 640, direction)


@pytest.mark.parametrize('mirror', [False, True])
@pytest.mark.parametrize('quality', ['clean', 'crop'])
def test_cap1076_low_score_existing_candidate_does_not_reenter_look_on_quality_label_change(mirror, quality):
    a, m, box, direction = evidence(mirror)
    if quality == 'crop':
        m.update(quality_bbox_ok=False, bbox_quality_tier='weak', quality_bbox_reason='edge_touch>2')
    before = deepcopy((a, m))
    assert supported(a, m, box, direction)
    assert (a, m) == before
    assert a['uid'] == 0 and a['identity_control_rejected']
    assert not a['bank_updated'] and not a['recent_bank_updated']


@pytest.mark.parametrize('quality', ['clean', 'crop'])
def test_secondary_unknown_can_change_crop_quality_without_changing_observation_role(quality):
    a, m, box, direction = evidence()
    a.update(reason='secondary_evidence_unavailable', reacquire_partial_comparable=False,
             reacquire_partial_state='unknown')
    a.pop('low_score_observation_blocked')
    m.update(low_score_continuation=False, detector_confidence=.9192646)
    if quality == 'crop':
        m.update(quality_bbox_ok=False, bbox_quality_tier='weak', quality_bbox_reason='edge_touch>2')
    assert supported(a, m, box, direction)


@pytest.mark.parametrize('fault', [
    'missing_blocked', 'blocked_none', 'blocked_true', 'blocked_zero',
    'partial_conflict', 'partial_mismatch', 'explicit_geometry', 'uid_recheck',
    'search_excluded', 'wrong_reason', 'no_low_score_marker', 'wrong_association',
    'missing_previous_cap', 'same_cap', 'future_previous_cap', 'same_timestamp',
    'future_previous_timestamp', 'old_association', 'nan_previous_timestamp',
    'missing_confidence_limit', 'invalid_confidence_limit', 'bool_confidence_limit',
    'outside_confidence_limit', 'missing_gallery', 'weak_gallery', 'self_reference',
    'other_gallery_uid', 'failed_competition', 'other_competition_uid',
    'old_competition', 'other_source_index', 'competing_person', 'not_fresh',
    'unrelated_quality_reason', 'other_quality_tier', 'learning_allowed',
])
def test_low_score_defer_is_not_a_generic_rejected_identity_bypass(fault):
    a, m, box, direction = evidence()
    if fault == 'missing_blocked': del a['low_score_observation_blocked']
    elif fault == 'blocked_none': a['low_score_observation_blocked'] = None
    elif fault == 'blocked_true': a['low_score_observation_blocked'] = True
    elif fault == 'blocked_zero': a['low_score_observation_blocked'] = 0
    elif fault == 'partial_conflict': a['reacquire_partial_state'] = 'conflict'
    elif fault == 'partial_mismatch': a['reacquire_partial_state'] = 'mismatch'
    elif fault == 'explicit_geometry': a['reacquire_geometry'] = dict(search_cross_edge_conflict=True)
    elif fault == 'uid_recheck': a['identity_recheck_pending'] = True
    elif fault == 'search_excluded': a['search_excluded'] = True
    elif fault == 'wrong_reason': a['reason'] = 'mapped_geometry_reject'
    elif fault == 'no_low_score_marker': del m['low_score_continuation']
    elif fault == 'wrong_association': m['association_reason'] = 'unassociated_detection'
    elif fault == 'missing_previous_cap': del m['association_previous_capture_frame_id']
    elif fault == 'same_cap': m['association_previous_capture_frame_id'] = 1076
    elif fault == 'future_previous_cap': m['association_previous_capture_frame_id'] = 1077
    elif fault == 'same_timestamp': m['association_previous_capture_timestamp'] = m['capture_timestamp']
    elif fault == 'future_previous_timestamp': m['association_previous_capture_timestamp'] = m['capture_timestamp']+.01
    elif fault == 'old_association': m['association_previous_capture_timestamp'] = m['capture_timestamp']-.501
    elif fault == 'nan_previous_timestamp': m['association_previous_capture_timestamp'] = float('nan')
    elif fault == 'missing_confidence_limit': del m['association_confidence_limit']
    elif fault == 'invalid_confidence_limit': m['association_confidence_limit'] = 1.1
    elif fault == 'bool_confidence_limit': m['association_confidence_limit'] = True
    elif fault == 'outside_confidence_limit': m['association_confidence_limit'] = .40
    elif fault in ('missing_gallery', 'self_reference'):
        del a['match_evidence']
        if fault == 'self_reference': a['similar_follow'] = dict(full_distance=.01, reference_cap=1074)
    elif fault == 'weak_gallery': a['match_evidence']['strong_distance'] = .300001
    elif fault == 'other_gallery_uid': a['match_evidence']['matched_uid'] = 2
    elif fault == 'failed_competition': a['identity_competition']['passed'] = False
    elif fault == 'other_competition_uid': a['identity_competition']['uid'] = 2
    elif fault == 'old_competition': a['identity_competition']['frame_index'] -= 1
    elif fault == 'other_source_index': a['identity_competition']['source_detection_index'] = 1
    elif fault == 'competing_person': a['identity_competition']['candidate_count'] = 2
    elif fault == 'not_fresh': m['is_fresh'] = False
    elif fault == 'unrelated_quality_reason': m['quality_bbox_reason'] = 'identity_swap_competing_track'
    elif fault == 'other_quality_tier': m['bbox_quality_tier'] = 'reject'
    elif fault == 'learning_allowed': a['learning_allowed'] = True
    assert not supported(a, m, box, direction)


@pytest.mark.parametrize('blocked', [False, True])
def test_actual_bank_low_score_result_distinguishes_global_uid_revoke(monkeypatch, blocked):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'vision'))
    from test_cap1040_identity_continuity import make_bank, observe
    from tools.replay_cap334_recovery import gallery_snapshot

    bank = make_bank()
    assert observe(bank, 1, 43000., (220., 20., 420., 470.), track=10, partial=0.) == 1
    bank.config = replace(bank.config, appearance_region_safety_enable=True)
    if blocked:
        bank._geometry_revoked_uids[1] = 457
    a, m, box, direction = evidence()
    output = observe(bank, 1076, m['capture_timestamp'], box, frame=458,
        score=m['detector_confidence'], full=.232350409, partial=.2, search=True, track=15,
        extra={**m, 'identity_competition': a['identity_competition']})
    actual = bank.last_assignments[15]
    assert output == 0 and actual['reason'] == 'low_score_observation_rejected'
    assert actual.get('reacquire_geometry') is None
    assert actual['low_score_observation_blocked'] is blocked
    before = gallery_snapshot(bank)
    assert supported(actual, m, box, direction) is not blocked
    assert gallery_snapshot(bank) == before and 15 not in bank.track_to_uid


@pytest.mark.parametrize('mirror', [False, True])
def test_cap1098_detector_only_low_score_crop_uses_separate_observation_policy(mirror):
    box = BOX1098
    if mirror:
        box = (640-box[2], box[1], 640-box[0], box[3])
    assert detector_only_side_search_observation(box, .3504, 640, 480,
        'right' if mirror else 'left', min_score=.25, confidence_limit=.5)


@pytest.mark.parametrize('fault', [
    'central', 'opposite_side', 'unknown_direction', 'not_at_edge', 'broad_body',
    'thin_sliver', 'malformed_box', 'negative_x', 'negative_y', 'outside_width',
    'outside_height', 'nan_bbox', 'bool_bbox', 'zero_width', 'zero_height',
    'high_score', 'below_formal_score', 'nan_score', 'bool_score',
    'missing_limit', 'negative_limit', 'zero_limit', 'invalid_limit', 'nan_limit',
    'bool_limit', 'at_lower_config_limit', 'zero_formal_min', 'invalid_formal_min',
])
def test_detector_only_defer_has_narrow_geometry_and_valid_config_boundaries(fault):
    box, score, width, height, direction, minimum, limit = BOX1098, .3504, 640, 480, 'left', .25, .5
    if fault == 'central': box = (200., 30., 273., 469.)
    elif fault == 'opposite_side': direction = 'right'
    elif fault == 'unknown_direction': direction = None
    elif fault == 'not_at_edge': box = (15., 32., 88., 471.)
    elif fault == 'broad_body': box = (0., 150., 140., 350.)
    elif fault == 'thin_sliver': box = (0., 30., 10., 471.)
    elif fault == 'malformed_box': box = box[:3]
    elif fault == 'negative_x': box = (-1., *box[1:])
    elif fault == 'negative_y': box = (box[0], -1., *box[2:])
    elif fault == 'outside_width': width = 70
    elif fault == 'outside_height': height = 470
    elif fault == 'nan_bbox': box = (float('nan'), *box[1:])
    elif fault == 'bool_bbox': box = (False, *box[1:])
    elif fault == 'zero_width': width = 0
    elif fault == 'zero_height': height = 0
    elif fault == 'high_score': score = .50
    elif fault == 'below_formal_score': score = .249
    elif fault == 'nan_score': score = float('nan')
    elif fault == 'bool_score': score = True
    elif fault == 'missing_limit': limit = None
    elif fault == 'negative_limit': limit = -.5
    elif fault == 'zero_limit': limit = 0
    elif fault == 'invalid_limit': limit = 1.1
    elif fault == 'nan_limit': limit = float('nan')
    elif fault == 'bool_limit': limit = True
    elif fault == 'at_lower_config_limit': limit = score
    elif fault == 'zero_formal_min': minimum = 0
    elif fault == 'invalid_formal_min': minimum = 1.1
    assert not detector_only_side_search_observation(box, score, width, height, direction,
                                                     min_score=minimum, confidence_limit=limit)


def test_detector_only_limit_tracks_valid_min_confidence_without_raising_half_score_ceiling():
    assert detector_only_side_search_observation(BOX1098, .3504, 640, 480, 'left',
                                                 min_score=.25, confidence_limit=.40)
    assert not detector_only_side_search_observation(BOX1098, .55, 640, 480, 'left',
                                                     min_score=.25, confidence_limit=.60)


def test_real_deepsort_cap1098_has_no_current_identity_observation_or_uid():
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1))
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction='left')
    records = tracker.update([Detection(BOX1098, .3504, 0)],
        [np.asarray([1., 0., 0.], dtype=np.float32)], image_width=640, image_height=480,
        frame_context=dict(capture_frame_id=1098, capture_timestamp=45092.845596,
                           is_fresh=True, control_frame_id=472))
    assert records == [] and tracker.last_identity_observations == []
    assert not tracker.deepsort.tracker.tracks and not tracker.identity_bank.identities
    assert tracker.config.min_confidence == .50
    assert detector_only_side_search_observation(BOX1098, .3504, 640, 480, 'left',
        min_score=.25, confidence_limit=tracker.config.min_confidence)
    assert records == [] and tracker.last_identity_observations == []
