"""CAP746--752: defer an unsuitable stationary look, never authorize UID0.

Geometry/times/distances are recorded; descriptors in the bank test are
synthetic. No camera, motor, or embedding model is exercised here.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from car_control_modular.search_candidate_gate import (
    CandidateObservation, SearchCandidateGate, SearchCandidateGateConfig,
    SearchCandidateGateDecision,
)
from car_control_modular.search_observation_retry import (
    DEFERRED_EDGE_COVERAGE, DetectorObservationSettlement,
    incomparable_edge_observation, side_crop_search_observation,
)


ROWS = {
    746: (341, 43190.696802555,
          (450.80291748046875, 1.8843536376953125, 638.5200805664062, 478.0947265625),
          .23028570413589478, .18714356422424316),
    747: (342, 43190.73311151,
          (445.63201904296875, .8554229736328125, 639.2554931640625, 478.40716552734375),
          .2244979739189148, .21527434885501862),
    749: (343, 43190.832889897,
          (443.41375732421875, 1.5083465576171875, 636.341796875, 476.4609375),
          .2750975489616394, .14689326286315918),
    751: (344, 43190.936557561,
          (466.25213623046875, .9898529052734375, 639.940673828125, 477.0233154296875),
          .2221418023109436, .1808977723121643),
    752: (345, 43190.996787242,
          (481.85211181640625, 2.3552703857421875, 639.230224609375, 476.0606689453125),
          .19715464115142822, .18386346101760864),
}
CENTER = (220., 30., 410., 460.)


def evidence(cap=746, mirror=False):
    frame, stamp, box, full, _ = ROWS[cap]
    if mirror:
        box = (640-box[2], box[1], 640-box[0], box[3])
    m = dict(capture_frame_id=cap, capture_timestamp=stamp, is_fresh=True,
             control_frame_id=frame, source_detection_index=0, quality_bbox_ok=False,
             bbox_quality_tier='weak', quality_bbox_reason='edge_touch>2',
             search_direction_compatible=True)
    a = dict(uid=0, mapped_uid=1, reason='secondary_evidence_unavailable',
             identity_control_rejected=True, bank_updated=False, recent_bank_updated=False,
             reacquire_partial_comparable=False, reacquire_partial_state='unknown',
             match_evidence=dict(matched_uid=1, match_source='strong', strong_distance=full),
             identity_competition=dict(uid=1, frame_index=frame, source_detection_index=0,
                                       candidate_count=1, passed=True),
             template_recent_evidence=dict(count=0, comparable_count=0,
                                            query_coverage='top1_bottom1_side1'),
             template_recent_partial_evidence=dict(count=0, comparable_count=0,
                                                    query_coverage='top1_bottom1_side1'))
    return a, m, box, 'left' if mirror else 'right'


def qualifies(a, m, box, direction='right', *, uid=1, output_uid=0, width=640):
    return side_crop_search_observation(a, m, output_uid, uid, box, width, direction)


@pytest.mark.parametrize('cap', ROWS)
@pytest.mark.parametrize('mirror', [False, True])
def test_recorded_side_crops_with_empty_recent_gallery_defer_from_first_frame(cap, mirror):
    a, m, box, direction = evidence(cap, mirror)
    before = deepcopy((a, m))
    assert not incomparable_edge_observation(a, m, 0, 1, box, 640)
    assert qualifies(a, m, box, direction)
    assert (a, m) == before
    assert a['uid'] == 0 and a['identity_control_rejected']
    assert not a['bank_updated'] and not a['recent_bank_updated']


@pytest.mark.parametrize('case', [
    'missing_assignment', 'missing_metadata', 'assigned_uid', 'unknown_target', 'bool_uid',
    'mapped_other_uid', 'best_other_uid', 'match_other_uid', 'competition_other_uid',
    'failed_competition', 'multiple_candidates', 'bool_count', 'old_competition',
    'wrong_source_index', 'missing_source_index', 'missing_frame', 'stale',
    'missing_gallery', 'candidate_self_distance_only', 'partial_source', 'weak_gallery',
    'nan_gallery', 'bool_gallery', 'negative_gallery', 'unknown_secondary',
    'partial_conflict', 'partial_match', 'partial_comparable', 'identity_conflict',
    'ambiguous', 'excluded', 'geometry_conflict', 'nested_conflict', 'top_geometry_reject',
    'direction_incompatible', 'opposite_direction', 'invalid_direction', 'central',
    'not_at_edge', 'malformed_bbox', 'negative_bbox', 'nan_bbox', 'invalid_width',
    'missing_crop_reason', 'other_quality_reason', 'missing_quality', 'strong_quality',
    'bank_write', 'recent_write', 'learning_written', 'learning_allowed', 'similar_learning',
])
def test_defer_requires_current_unique_unknown_crop_and_independent_gallery(case):
    a, m, box, direction = evidence()
    kwargs = {}
    if case == 'missing_assignment': a = None
    elif case == 'missing_metadata': m = None
    elif case == 'assigned_uid': kwargs['output_uid'] = 1
    elif case == 'unknown_target': kwargs['uid'] = 0
    elif case == 'bool_uid': kwargs['uid'] = True
    elif case == 'mapped_other_uid': a['mapped_uid'] = 2
    elif case == 'best_other_uid': a['best_uid'] = 2
    elif case == 'match_other_uid': a['match_evidence']['matched_uid'] = 2
    elif case == 'competition_other_uid': a['identity_competition']['uid'] = 2
    elif case == 'failed_competition': a['identity_competition']['passed'] = False
    elif case == 'multiple_candidates': a['identity_competition']['candidate_count'] = 2
    elif case == 'bool_count': a['identity_competition']['candidate_count'] = True
    elif case == 'old_competition': a['identity_competition']['frame_index'] -= 1
    elif case == 'wrong_source_index': a['identity_competition']['source_detection_index'] = 1
    elif case == 'missing_source_index': del m['source_detection_index']
    elif case == 'missing_frame': del m['control_frame_id']
    elif case == 'stale': m['is_fresh'] = False
    elif case in ('missing_gallery', 'candidate_self_distance_only'):
        del a['match_evidence']
        if case == 'candidate_self_distance_only':
            a['similar_follow'] = dict(full_distance=.05, reference_cap=745)
    elif case == 'partial_source': a['match_evidence']['match_source'] = 'partial'
    elif case == 'weak_gallery': a['match_evidence']['strong_distance'] = .30001
    elif case == 'nan_gallery': a['match_evidence']['strong_distance'] = float('nan')
    elif case == 'bool_gallery': a['match_evidence']['strong_distance'] = False
    elif case == 'negative_gallery': a['match_evidence']['strong_distance'] = -.1
    elif case == 'unknown_secondary': del a['reacquire_partial_state']
    elif case == 'partial_conflict': a['reacquire_partial_state'] = 'conflict'
    elif case == 'partial_match': a['reacquire_partial_state'] = 'match'
    elif case == 'partial_comparable': a['reacquire_partial_comparable'] = True
    elif case == 'identity_conflict': a['reason'] = 'recent_partial_conflict'
    elif case == 'ambiguous': a['reason'] = 'search_candidate_identity_ambiguous'
    elif case == 'excluded': a['search_excluded'] = True
    elif case == 'geometry_conflict': m['candidate_geometry_conflict'] = True
    elif case == 'nested_conflict': a['reacquire_geometry'] = dict(search_cross_edge_conflict=True)
    elif case == 'top_geometry_reject':
        a.update(reacquire_geometry_ok=False, reacquire_geometry_reason='identity_swap_competing_track')
    elif case == 'direction_incompatible': m['search_direction_compatible'] = False
    elif case == 'opposite_direction': direction = 'left'
    elif case == 'invalid_direction': direction = None
    elif case == 'central': box = CENTER
    elif case == 'not_at_edge': box = (440., 2., 620., 478.)
    elif case == 'malformed_bbox': box = box[:3]
    elif case == 'negative_bbox': box = (450., -1., 638., 478.)
    elif case == 'nan_bbox': box = (float('nan'), *box[1:])
    elif case == 'invalid_width': kwargs['width'] = 0
    elif case == 'missing_crop_reason': del m['quality_bbox_reason']
    elif case == 'other_quality_reason': m['quality_bbox_reason'] += ',identity_swap_competing_track'
    elif case == 'missing_quality': del m['quality_bbox_ok']
    elif case == 'strong_quality': m.update(quality_bbox_ok=True, bbox_quality_tier='strong')
    elif case == 'bank_write': a['bank_updated'] = True
    elif case == 'recent_write': a['recent_bank_updated'] = True
    elif case == 'learning_written': a['learning_written_tiers'] = ['recent']
    elif case == 'learning_allowed': a['learning_allowed'] = True
    elif case == 'similar_learning': a['similar_follow'] = dict(learning_allowed=True)
    assert not qualifies(a, m, box, direction, **kwargs)


@pytest.mark.parametrize('reason', ['edge_touch>2', 'aspect<0.18', 'edge_touch>2,aspect<0.18'])
def test_explicit_crop_quality_reasons_are_not_general_quality_bypasses(reason):
    a, m, box, direction = evidence()
    m['quality_bbox_reason'] = reason
    assert qualifies(a, m, box, direction)


@pytest.mark.parametrize('mirror', [False, True])
def test_real_bank_rejected_assignments_supply_gallery_provenance_without_event_merge(monkeypatch, mirror):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'vision'))
    from test_cap1040_identity_continuity import make_bank, observe
    from tools.replay_cap334_recovery import gallery_snapshot

    bank = make_bank()
    assert observe(bank, 1, 43000., (220., 20., 420., 470.), track=10, partial=0.) == 1
    # The fixture establishes a gallery; current rejection uses deployed region
    # safety. Its recent evidence has expired, leaving an independent archive.
    bank.config = replace(bank.config, appearance_region_safety_enable=True)
    for cap, (frame, stamp, _, full, partial) in ROWS.items():
        a, m, box, direction = evidence(cap, mirror)
        output = observe(bank, cap, stamp, box, frame=frame, full=full, partial=partial,
                         crop=True, search=True, track=11,
                         extra={**m, 'identity_competition': a['identity_competition']})
        assert output == 0
        actual = bank.last_assignments[11]
        before = gallery_snapshot(bank)
        assignment_before = deepcopy(actual)
        assert actual['reason'] == 'secondary_evidence_unavailable'
        assert actual['template_recent_evidence']['count'] == 0
        assert actual['reacquire_partial_comparable'] is False
        assert 'feature_available' not in actual
        assert actual['match_evidence']['match_source'] == 'strong'
        assert qualifies(actual, m, box, direction, output_uid=output)
        assert bank.last_assignments[11] == assignment_before
        assert gallery_snapshot(bank) == before
        assert 11 not in bank.track_to_uid


def test_deferred_sequence_never_spends_look_or_becomes_identity_confirmation():
    gate = SearchCandidateGate(SearchCandidateGateConfig(hold_frames=2))
    settlement = DetectorObservationSettlement()
    for cap, (_, stamp, _, _, _) in ROWS.items():
        a, m, box, direction = evidence(cap)
        assert qualifies(a, m, box, direction)
        decision = gate.update(timestamp=stamp, search_active=True, width=640, height=480,
            formal_candidates=(CandidateObservation(box, .89),), deferred_observation_bboxes=(box,))
        decision = settlement.update(decision, now=stamp+.06, capture_timestamp=stamp,
            capture_id=cap, search_active=True, zero_sent_at=None, feedback=None, max_hold_sec=.3)
        assert decision.reason == DEFERRED_EDGE_COVERAGE
        assert not decision.pause_rotation and not decision.entered and not decision.completed
        assert not decision.preferred_target_match and not gate.hold_active and not settlement.active
        for old in (stamp, stamp-.001):
            duplicate = gate.update(timestamp=old, search_active=True, width=640, height=480,
                formal_candidates=(CandidateObservation(box, .89),), deferred_observation_bboxes=(box,))
            assert duplicate.reason == 'candidate_observation_duplicate_or_old'
            assert not duplicate.entered and not duplicate.completed and not duplicate.pause_rotation
    # Moving into a useful crop can still request the original bounded look.
    a, m, _, direction = evidence(752)
    assert not qualifies(a, m, CENTER, direction)
    decision = gate.update(timestamp=stamp+.1, search_active=True, width=640, height=480,
                           formal_candidates=(CandidateObservation(CENTER, .89),))
    assert decision.entered and decision.pause_rotation


def test_existing_observation_release_does_not_claim_settlement_or_move_a_deadline():
    settlement = DetectorObservationSettlement()
    _, stamp, box, _, _ = ROWS[746]
    args = dict(search_active=True, zero_sent_at=None, feedback=None, max_hold_sec=.3)
    seed = SearchCandidateGateDecision(entered=True, pause_rotation=True, bbox=box)
    settlement.update(seed, now=stamp, capture_timestamp=stamp-.01, capture_id=745, **args)
    deadline = settlement.deadline
    a, m, box, direction = evidence()
    assert qualifies(a, m, box, direction)
    decision = settlement.update(replace(seed, entered=False, reason=DEFERRED_EDGE_COVERAGE),
        now=stamp+.05, capture_timestamp=stamp, capture_id=746, **args)
    assert decision.completed and not decision.pause_rotation and not decision.preferred_target_match
    assert decision.reason == DEFERRED_EDGE_COVERAGE
    assert not settlement.active and settlement.deadline == deadline
