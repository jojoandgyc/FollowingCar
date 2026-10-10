"""Recorded CAP1540--1548 boxes/clocks and synthetic logged-distance features.

This exercises the actual identity entry, including the negative raw-ID probe
and low-score position update. It is not a replay of OSNet or DeepSORT costs.
"""
from copy import deepcopy

import pytest

import rk_vision.identity_bank as bank_module
from rk_vision.similar_follow import formal_detection_continuous
from test_cap1040_identity_continuity import make_bank, observe
from test_cap874_identity_reacquire import feature, metadata
from tools.replay_cap334_recovery import gallery_snapshot


# cap, processing frame, capture timestamp, yaw, detector score, gallery distance, box
ROWS = {
    1540: (738, 28252.597150377, -114.97087281809031, .9117296934127808,
           .2892181873321533, (1.48114013671875, 12.70233154296875,
                               179.50631713867188, 475.6480712890625)),
    1542: (739, 28252.699797808, -116.28203904774304, .8552175164222717,
           .26503652334213257, (40.17799377441406, 2.29010009765625,
                                212.9261474609375, 478.0072021484375)),
    1545: (740, 28252.860899293, -118.86962200713718, .7911704182624817,
           .2770344018936157, (127.7252197265625, 12.281661987304688,
                               288.310302734375, 477.20233154296875)),
    1546: (741, 28252.89685067, -119.9079081857157, .4219575524330139,
           .2867262363433838, (137.88424682617188, 15.120330810546875,
                               303.01702880859375, 476.887451171875)),
    1548: (742, 28252.997868893, -121.973937163217, .5764241218566895,
           .2750299572944641, (176.5794219970703, 134.35523986816406,
                               347.1095886230469, 475.9959716796875)),
}


def send(bank, cap, *, extra=None, raw=None, stamp=None, score=None, distance=None,
         box=None, partial=None, vector=True, count=1, **kwargs):
    frame, ts, yaw, confidence, full, bounds = ROWS[cap]
    raw = (-1 if cap == 1545 else 34) if raw is None else raw
    ts, confidence = (ts if stamp is None else stamp), (confidence if score is None else score)
    bounds, full = (bounds if box is None else box), (full if distance is None else distance)
    m = metadata(cap, ts, bounds, yaw=yaw, search=True)
    m.update(track_id=raw, image_width=640, image_height=480,
             detector_confidence=confidence, source_detection_index=0,
             identity_competition=dict(uid=1, frame_index=frame,
                 source_detection_index=0, candidate_count=count, passed=True,
                 reason='single_candidate' if count == 1 else 'ambiguous', distance=full))
    if cap == 1546:
        m.update(low_score_continuation=True, association_reason='low_score_existing_track',
                 association_previous_capture_frame_id=1542,
                 association_previous_capture_timestamp=ROWS[1542][1])
    m.update(extra or {})
    return bank.assign(track_id=raw, feature=feature(full) if vector else None,
        partial_feature=None if partial is None else feature(partial), confidence=confidence,
        area=(bounds[2]-bounds[0])*(bounds[3]-bounds[1]), frame_index=frame,
        candidate_count=count, bbox_quality_ok=m['quality_bbox_ok'],
        bbox_quality_tier=m['bbox_quality_tier'], sample_metadata=m,
        preferred_uid=1, preferred_candidate_ok=True, **kwargs)


def ready(*, low_position=True, probe=True):
    bank = make_bank()
    assert observe(bank, 10, 28100., (200., 4., 410., 476.),
                   frame=1, track=1, partial=0.) == 1
    anchor = deepcopy(bank.identities[1].last_strong_observation)
    assert send(bank, 1540) == 0
    gallery = gallery_snapshot(bank)
    assert send(bank, 1542) == 1
    accepted = deepcopy(bank._similar_follow_states[(1, 34)])
    if probe:
        assert send(bank, 1545) == 0
        assert bank._similar_follow_states[(1, 34)] == accepted
        assert bank.track_to_uid.get(-1) is None
        assert bank.track_to_uid.get(34) == 1
        assert not bank.last_assignments[-1].get('similar_follow', {}).get('completed_confirmation')
    if low_position:
        assert send(bank, 1546) == 0
        state = bank._similar_follow_states[(1, 34)]
        assert state['position_only'] and state['last_qualified_cap'] == 1542
        assert state['last_qualified_timestamp'] == ROWS[1542][1]
        assert not bank.last_assignments[34]['similar_follow'].get('completed_confirmation')
    return bank, gallery, anchor


def test_recorded_probe_low_score_and_new_full_sequence_retains_uid_without_learning():
    bank, gallery, anchor = ready()
    assert bank.config.min_confidence == .60
    assert send(bank, 1548) == 1
    detail = bank.last_assignments[34]
    assert detail['similar_follow']['formal_detection_reverified']
    assert detail['similar_follow_motion_geometry']['residual'] == pytest.approx(.0139896818)
    assert detail['similar_follow']['completed_confirmation']
    assert detail['similar_follow']['raw_track_id'] == 34
    assert detail['similar_follow']['capture_timestamp'] == ROWS[1548][1]
    assert detail['similar_follow']['capture_frame_id'] == 1548
    state = bank._similar_follow_states[(1, 34)]
    assert state['last_cap'] == 1548 and state['last_qualified_cap'] == 1548
    assert state['gallery_distance'] == pytest.approx(ROWS[1548][4], abs=1e-6)
    assert not detail['bank_updated'] and detail['template_update_quarantined']
    assert bank.identities[1].last_strong_observation == anchor
    assert gallery_snapshot(bank) == gallery
    # One more fresh proper FULL result must not restart observation either.
    assert observe(bank, 1549, ROWS[1548][1]+.10, ROWS[1548][-1],
        frame=743, track=34, score=.85, full=.27, search=True, yaw=ROWS[1548][2]) == 1
    assert bank.last_assignments[34]['similar_follow']['completed_confirmation']
    assert gallery_snapshot(bank) == gallery


def test_original_global_confidence_gate_reproduces_cap1548_rejection(monkeypatch):
    bank, gallery, _ = ready()
    monkeypatch.setattr(bank_module, 'formal_detection_continuous', lambda *a, **k: False)
    assert send(bank, 1548) == 0
    assert bank.last_assignments[34]['similar_follow']['reason'] == 'observation_unverified'
    assert (1, 34) not in bank._similar_follow_states
    assert gallery_snapshot(bank) == gallery


def test_formal_half_confidence_can_follow_confirmed_raw_without_low_score_intermediate():
    bank, gallery, anchor = ready(low_position=False)
    # Directly after confirmation: new independent appearance and continuous
    # geometry, not a requirement to first fail through the low-score branch.
    assert send(bank, 1548, box=ROWS[1542][-1], score=.55) == 1
    assert bank.last_assignments[34]['similar_follow']['formal_detection_reverified']
    assert gallery_snapshot(bank) == gallery
    assert bank.identities[1].last_strong_observation == anchor


@pytest.mark.parametrize('failure', [
    'current_gallery', 'previous_gallery', 'no_gallery_provenance',
    'self_reference', 'low_score', 'metadata_low_score', 'weak', 'stale',
    'duplicate', 'out_of_order', 'gap', 'competition', 'many', 'conflict',
    'revoked', 'changed_raw', 'displacement', 'small_area', 'observation_only',
])
def test_relaxed_formal_confidence_keeps_independent_and_geometry_limits(failure):
    bank, gallery, _ = ready()
    options, extra = {}, {}
    if failure in ('current_gallery', 'self_reference'): options['distance'] = .31
    elif failure == 'previous_gallery': bank._similar_follow_states[(1, 34)]['gallery_distance'] = .31
    elif failure == 'no_gallery_provenance': bank._similar_follow_states[(1, 34)].pop('gallery_distance')
    elif failure == 'low_score': options['score'] = .499
    elif failure == 'metadata_low_score': extra['detector_confidence'] = .499
    elif failure == 'weak': extra.update(quality_bbox_ok=False, bbox_quality_tier='weak')
    elif failure == 'stale': extra['is_fresh'] = False
    elif failure == 'duplicate': extra.update(capture_frame_id=1546, capture_timestamp=ROWS[1546][1])
    elif failure == 'out_of_order': extra.update(capture_frame_id=1545, capture_timestamp=ROWS[1545][1])
    elif failure == 'gap': options['stamp'] = ROWS[1546][1]+.501
    elif failure == 'competition':
        extra['identity_competition'] = dict(uid=1, frame_index=742, source_detection_index=0,
            candidate_count=1, passed=False, reason='ambiguous')
    elif failure == 'many': options['count'] = 2
    elif failure == 'conflict': bank._mapped_geometry_conflicts[34] = dict(uid=1, search_contradiction=True)
    elif failure == 'revoked': bank._geometry_revoked_uids[1] = dict(reason='conflict')
    elif failure == 'changed_raw': options['raw'] = 35
    elif failure == 'displacement': options['box'] = (460., 134., 631., 476.)
    elif failure == 'small_area': options['box'] = (200., 300., 204., 305.)
    elif failure == 'observation_only': extra['observation_only'] = True
    if failure == 'self_reference':
        bank._follow_references[1] = [dict(feature=feature(.31), cap=1542, timestamp=ROWS[1542][1])]
    assert send(bank, 1548, extra=extra, **options) == 0, failure
    assert not bank.last_assignments[options.get('raw', 34)].get('similar_follow', {}).get('completed_confirmation')
    assert gallery_snapshot(bank) == gallery


def test_missing_feature_cannot_issue_new_similar_confirmation_or_refresh_candidate():
    bank, gallery, _ = ready()
    previous = deepcopy(bank._similar_follow_states[(1, 34)])
    # The existing ordinary no-feature mapped fallback may keep its UID. It is
    # not this new independently reverified permission and must not advance it.
    send(bank, 1548, vector=False)
    detail = bank.last_assignments[34]
    assert not detail.get('similar_follow', {}).get('completed_confirmation')
    assert bank._similar_follow_states[(1, 34)] == previous
    assert not detail['bank_updated'] and gallery_snapshot(bank) == gallery


def test_reliable_comparable_torso_conflict_still_rejects(monkeypatch):
    bank, gallery, _ = ready()
    memory = bank.identities[1].template_memory
    original = memory.evidence
    checked = []

    def evidence(vector, current, tier='strong', **kwargs):
        result = original(vector, current, tier, **kwargs)
        if tier == 'partial' and kwargs.get('reliable_only') and kwargs.get('comparable_only'):
            assert vector is not None
            checked.append(current['capture_frame_id'])
            return dict(result, count=1, distance=.75, comparable_count=1,
                comparable_distance=.75, comparable_caps=[1542], winner_cap=1542)
        return result

    monkeypatch.setattr(memory, 'evidence', evidence)
    assert send(bank, 1548, partial=.75) == 0
    assert checked and set(checked) == {1548}
    assert bank.last_assignments[34]['similar_follow']['reason'] == 'reliable_partial_conflict'
    assert gallery_snapshot(bank) == gallery


def test_new_moderate_confidence_person_cannot_initialize_candidate():
    bank = make_bank()
    assert observe(bank, 10, 28100., (200., 4., 410., 476.),
                   frame=1, track=1, partial=0.) == 1
    assert send(bank, 1548) == 0
    assert not bank._similar_follow_states


def test_position_only_cannot_roll_independent_qualified_deadline():
    bank, _, _ = ready()
    state = deepcopy(bank._similar_follow_states[(1, 34)])
    state['last_timestamp'] = ROWS[1548][1]-.10
    state['last_cap'] = 1547
    state['observation'].update(capture_frame_id=1547, capture_timestamp=state['last_timestamp'])
    # Keep a fresh position, but the original full observation is too old.
    state['last_qualified_timestamp'] = ROWS[1548][1]-.751
    current = metadata(1548, ROWS[1548][1], ROWS[1548][-1])
    current.update(track_id=34, detector_confidence=.576)
    assert not formal_detection_continuous(current, state,
        dict(ok=True, yaw_compensated_center_jump_ratio=.01, area_similarity=.8),
        confidence=.576, gallery_distance=.275)


def test_new_confirmation_does_not_override_next_capture_hard_conflict():
    bank, gallery, anchor = ready()
    assert send(bank, 1548) == 1
    bank._mapped_geometry_conflicts[34] = dict(uid=1, search_contradiction=True,
        rejected_capture=1549)
    assert observe(bank, 1549, ROWS[1548][1]+.10, ROWS[1548][-1],
        frame=743, track=34, score=.85, full=.02, search=True, yaw=ROWS[1548][2]) == 0
    result = bank.last_assignments[34]
    assert not result.get('similar_follow', {}).get('completed_confirmation')
    assert not result['bank_updated']
    assert gallery_snapshot(bank) == gallery
    assert bank.identities[1].last_strong_observation == anchor


def test_probe_cannot_keep_the_old_raw_candidate_alive_past_observation_window():
    bank, gallery, _ = ready(low_position=False, probe=False)
    assert send(bank, 1545, stamp=ROWS[1542][1]+.501) == 0
    assert (1, 34) not in bank._similar_follow_states
    assert bank.track_to_uid.get(-1) is None
    assert not bank.last_assignments[-1].get('similar_follow', {}).get('completed_confirmation')
    assert gallery_snapshot(bank) == gallery
