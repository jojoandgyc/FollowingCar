"""CAP1529 raw24 -> raw25: recorded boxes/clocks, synthetic ReID vectors.

Exercise IdentityBank.assign, not a forced owner passed to its evaluator.
Embeddings reproduce logged gallery distances; this does not replay the model,
DeepSORT's missing CAP1526 pair costs, or closed-loop vehicle dynamics.
"""
from copy import deepcopy
from dataclasses import replace

import pytest

from rk_vision.identity_bank import _geometry_observation
from rk_vision.detector_continuation import DetectorObservation, DetectorProof
from rk_vision.similar_follow import evaluate_similar_follow
from test_cap1040_identity_continuity import make_bank, observe
from test_cap874_identity_reacquire import feature, metadata
from tools.replay_cap334_recovery import gallery_snapshot


# cap, processing frame, capture time, yaw, confidence, gallery distance, crop, bbox
ROWS = (
    (1484, 593, 26146.378564805, -10.0017369348889, .91549718, .10976076, False,
     (108.15660095, 2.19740295, 337.68157959, 475.16210938)),
    (1487, 594, 26146.540511816, -11.352885981146096, .91926461, .12562793, False,
     (69.00639343, 1.72921753, 327.76242065, 478.26879883)),
    (1490, 595, 26146.712612069, -10.980940384645274, .91926461, .11549377, True,
     (11.98170471, 2.24472046, 299.64630127, 475.54351807)),
    (1492, 596, 26146.808541132, -12.404752968716942, .83638012, .14482194, False,
     (0., 3.94351196, 263.50723267, 469.99749756)),
    (1497, 597, 26147.075160567, -15.154103842113049, .89289230, .18549693, True,
     (.00320435, 5.74241638, 181.52333069, 476.68994141)),
    (1501, 598, 26147.273723940, -15.640813774264519, .90796220, .22375238, True,
     (0., 3.43588257, 146.78677368, 475.64648438)),
    (1505, 599, 26147.482247184, -17.54995787154599, .89289230, .16236687, True,
     (.73566437, 3.50448608, 158.58808899, 474.86358643)),
    (1510, 600, 26147.740713576, -19.949683232275106, .90796220, .21618319, False,
     (.71061707, 4.81945801, 166.96078491, 464.30493164)),
    (1512, 601, 26147.840413793, -21.52737344001823, .88535732, .16003889, True,
     (.69872284, 4.12202454, 200.82028198, 477.45117188)),
    (1515, 602, 26148.009033385, -25.325280731517044, .88158989, .14512742, True,
     (0., 4.61952209, 257.08920288, 475.64514160)),
    (1520, 603, 26148.241142530, -30.200787261452565, .80624032, .16053045, False,
     (37.14645386, 3.26715088, 347.82000732, 469.95373535)),
    (1522, 604, 26148.372864892, -33.55528926327805, .84391510, .13289750, False,
     (133.77648926, 4.24145508, 409.14056396, 474.23162842)),
    (1529, 606, 26148.736991850, -37.2191347598427, .90796220, .17034972, False,
     (339.32385254, 6.56394958, 559.85021973, 476.53039551)),
    (1530, 607, 26148.771941962, -37.28469789577643, .89289230, .18960428, False,
     (356.29791260, 5.73225403, 573.06646729, 477.46936035)),
    (1532, 608, 26148.872798025, -37.28469789577643, .92679960, .18519676, False,
     (395.69546509, 6.70704651, 614.82708740, 467.42816162)),
    (1535, 609, 26149.036485075, -37.52330713320735, .92679960, .10383421, True,
     (425.61083984, 4.76377869, 638.75518799, 475.72332764)),
    (1537, 610, 26149.170088056, -37.76001588974804, .93433458, .10710120, True,
     (447.87246704, 4.12608337, 639.27545166, 473.40350342)),
    (1539, 611, 26149.271008284, -37.921193249608145, .93433458, .21737176, True,
     (475.59622192, 4.11688232, 639.25732422, 476.37426758)),
    (1541, 612, 26149.373295227, -38.009145204356685, .84768254, .24819922, True,
     (503.61761475, 5.63769531, 640., 476.05096436)),
    (1542, 613, 26149.404259523, -38.009145204356685, .86275250, .29361421, False,
     (515.68414307, 10.08903503, 639.27008057, 475.57226562)),
    (1545, 614, 26149.568357813, -38.009145204356685, .70828587, .32838035, False,
     (580.10620117, 13.85458374, 640., 472.33190918)),
)
BEFORE, AFTER = ROWS[:12], ROWS[12:]


def send(bank, row, *, search=False, opposite=False, extra=None, full=None, **kwargs):
    cap, frame, stamp, yaw, score, distance, crop, box = row
    raw = 24 if cap <= 1522 else 25 if cap < 1545 else -1
    details = dict(source_detection_index=0, identity_competition=dict(
        uid=1, frame_index=frame, source_detection_index=0, candidate_count=1,
        passed=True, reason='single_candidate', distance=distance))
    if cap == 1545:
        details.update(quality_bbox_ok=False, bbox_quality_tier='weak',
                       quality_bbox_reason='aspect<0.18', bbox_quality_reason='aspect<0.18')
    details.update(extra or {})
    return observe(bank, cap, stamp, box, frame=frame, track=raw, yaw=yaw,
                   score=score, full=distance if full is None else full, crop=crop,
                   search=search, opposite=opposite, extra=details, **kwargs)


def before1529():
    bank = make_bank()
    # An archived trusted identity, plus a separately accepted follow-only raw
    # track. Its old anchor must not be overwritten by this provisional path.
    assert observe(bank, 10, 26050., (200., 4., 410., 476.),
                   frame=1, track=1, partial=0.) == 1
    anchor = deepcopy(bank.identities[1].last_strong_observation)
    assert send(bank, BEFORE[0], search=True) == 0
    gallery = gallery_snapshot(bank)  # recent entries age normally before replay
    assert send(bank, BEFORE[1], search=True) == 1
    for row in BEFORE[2:]:
        assert send(bank, row) == 1
    assert bank.track_to_uid[24] == 1 and bank._similar_follow_states[(1, 24)]['active']
    assert gallery_snapshot(bank) == gallery
    return bank, gallery, anchor


@pytest.mark.parametrize('search_still_active', [False, True])
def test_real_capture_sequence_rebinds_then_continues_without_gallery_writes(search_still_active):
    bank, gallery, anchor = before1529()
    for row in AFTER:
        cap = row[0]
        # Search may lag acceptance; crossing screen sides must not reseed.
        uid = send(bank, row, search=search_still_active and cap >= 1535,
                   opposite=cap >= 1535)
        assert uid == (0 if cap == 1545 else 1), (cap, bank.last_assignments)
        result = bank.last_assignments[25 if cap < 1545 else -1]
        assert not result['bank_updated']
        assert bank.identities[1].last_strong_observation == anchor
        assert gallery_snapshot(bank) == gallery
        if cap == 1529:
            info = result['similar_follow']
            assert info['handoff_from_track_id'] == 24
            assert info['handoff_reference_cap'] == 1522
            assert info['handoff_gap_ms'] == pytest.approx(364.126958)
            assert info['handoff_gallery_distance'] == pytest.approx(.17034972, abs=1e-6)
            assert 24 not in bank.track_to_uid
            assert (1, 24) not in bank._similar_follow_states
            assert result['template_update_quarantined']


@pytest.mark.parametrize('failure', [
    'gap', 'gallery', 'missing_feature', 'duplicate', 'out_of_order', 'stale',
    'two_candidates', 'competition', 'old_conflict', 'new_conflict', 'revoked',
    'weak_quality', 'displacement', 'area', 'two_owners',
])
def test_handoff_extension_keeps_observation_identity_and_geometry_limits(failure):
    bank, gallery, _ = before1529()
    row = list(AFTER[0])
    extra = {}
    if failure == 'gap': row[2] = BEFORE[-1][2] + .5001
    if failure == 'gallery': row[5] = .301
    if failure in ('duplicate', 'out_of_order'):
        row[0], row[2] = 1522, BEFORE[-1][2] - (.01 if failure == 'out_of_order' else 0.)
        # Keep this test on the proposed NEW raw track below.
    if failure == 'stale': extra['is_fresh'] = False
    if failure == 'two_candidates': extra['candidate_count'] = 2
    if failure == 'competition':
        extra['identity_competition'] = dict(uid=1, frame_index=row[1],
            candidate_count=1, source_detection_index=0, passed=False, reason='ambiguous')
    if failure == 'old_conflict': bank._mapped_geometry_conflicts[24] = dict(uid=1)
    if failure == 'new_conflict': bank._mapped_geometry_conflicts[25] = dict(uid=1)
    if failure == 'revoked': bank._geometry_revoked_uids[1] = dict(reason='identity_conflict')
    if failure == 'weak_quality': extra.update(quality_bbox_ok=False, bbox_quality_tier='weak')
    if failure == 'displacement':
        x1, y1, x2, y2 = row[7]
        row[7] = (x1+22., y1, x2+22., y2)  # compensated jump > .25, ordinary geometry still passes
    if failure == 'area':
        x1, y1, x2, y2 = row[7]
        center = (x1+x2)/2
        row[7] = (center-40., y1, center+40., y2)
    if failure == 'two_owners':
        bank.identities[2] = deepcopy(bank.identities[1])
        bank.identities[2].uid = 2
        other = deepcopy(bank._similar_follow_states[(1, 24)])
        other.update(uid=2, track_id=23)
        bank._similar_follow_states[(2, 23)] = other
        bank.track_to_uid[23] = 2
    cap, frame, stamp, yaw, score, distance, _, box = row
    m = metadata(cap, stamp, box, yaw=yaw, search=False)
    m.update(track_id=25, image_width=640, image_height=480, detector_confidence=score)
    m.update(extra)
    uid = bank.assign(track_id=25, feature=None if failure == 'missing_feature' else feature(distance),
        partial_feature=None, confidence=score, area=(box[2]-box[0])*(box[3]-box[1]),
        frame_index=frame, candidate_count=extra.get('candidate_count', 1),
        bbox_quality_ok=m['quality_bbox_ok'], bbox_quality_tier=m['bbox_quality_tier'],
        sample_metadata=m)
    assert uid == 0, (failure, bank.last_assignments[25])
    assert bank.track_to_uid.get(25) != 1
    assert gallery_snapshot(bank)['1'] == gallery['1']


def test_twenty_five_percent_is_not_a_new_same_track_or_self_reference_limit():
    bank, _, _ = before1529()
    state = deepcopy(bank._similar_follow_states[(1, 24)])
    cap, frame, stamp, yaw, _, _, _, box = AFTER[0]
    current = metadata(cap, stamp, box, yaw=yaw, search=False)
    current['track_id'] = 25
    geometry = bank._handoff_geometry(1, current, frame,
        reference_override=_geometry_observation(state['observation'], 604))
    assert geometry['ok'] and geometry['yaw_compensated_center_jump_ratio'] == pytest.approx(.2172617)
    base = dict(uid=1, track_id=25, current=current, geometry=geometry,
                competition_ok=True, blocked=False, full_distance=.01,
                direction_compatible=True, state=state, handoff_from_track_id=24)
    assert evaluate_similar_follow(**base).reason == 'local_geometry_conflict'
    assert evaluate_similar_follow(**base, handoff_gallery_distance=.301).reason == 'local_geometry_conflict'
    assert evaluate_similar_follow(**base, handoff_gallery_distance=.17).status == 'follow'
    base.update(track_id=24, current=dict(current, track_id=24), handoff_from_track_id=None)
    assert evaluate_similar_follow(**base, handoff_gallery_distance=.17).reason == 'local_geometry_conflict'


def test_old_350ms_owner_gate_reproduces_rejection_at_1529(monkeypatch):
    bank, gallery, _ = before1529()
    original = bank._similar_follow_handoff_candidate
    # Suppress only the new independently checked extension, reproducing the
    # old gate. No owner or similar-state is injected into the new raw track.
    monkeypatch.setattr(bank, '_similar_follow_handoff_candidate',
        lambda identity, raw, meta, count, feature=None: original(identity, raw, meta, count))
    assert send(bank, AFTER[0]) == 0
    assert bank.last_assignments[25]['reason'] == 'secondary_evidence_unavailable'
    assert gallery_snapshot(bank) == gallery


def test_reliable_comparable_partial_conflict_still_blocks_new_raw(monkeypatch):
    bank, gallery, _ = before1529()
    memory = bank.identities[1].template_memory
    original = memory.evidence
    checked = []

    def conflicting_evidence(vector, current, tier='strong', **options):
        result = original(vector, current, tier, **options)
        if tier == 'partial' and options.get('reliable_only') and options.get('comparable_only'):
            checked.append(current['capture_frame_id'])
            assert vector is not None
            return dict(result, count=1, distance=.75, comparable_count=1,
                        comparable_distance=.75, comparable_caps=[1522], winner_cap=1522)
        return result

    monkeypatch.setattr(memory, 'evidence', conflicting_evidence)
    assert send(bank, AFTER[0], partial=.75) == 0
    assert checked and set(checked) == {1529}
    assert bank.last_assignments[25]['similar_follow']['reason'] == 'reliable_partial_conflict'
    assert (1, 25) not in bank._similar_follow_states
    assert bank._similar_follow_states[(1, 24)]['last_cap'] == 1522
    assert gallery_snapshot(bank) == gallery


def test_successful_rebind_cannot_hide_a_subsequent_hard_identity_conflict():
    bank, gallery, anchor = before1529()
    assert send(bank, AFTER[0]) == 1
    accepted = deepcopy(bank._similar_follow_states[(1, 25)])
    row = list(AFTER[1])
    row[7] = (0., 5., 180., 477.)
    current = metadata(row[0], row[2], row[7], yaw=row[3], search=False)
    geometry = bank._handoff_geometry(1, current, row[1],
        reference_override=_geometry_observation(accepted['observation'], AFTER[0][1]))
    assert not geometry['ok'] and geometry['yaw_compensated_center_jump_ratio'] > .5
    bank._mapped_geometry_conflicts[25] = dict(uid=1, search_contradiction=True,
        reference=geometry['reference'], candidate=geometry['current'], rejected_capture=row[0])
    assert send(bank, row) == 0
    assert bank.last_assignments[25]['reason'] == 'mapped_geometry_reject'
    assert gallery_snapshot(bank) == gallery
    assert bank.identities[1].last_strong_observation == anchor


def detector_full_bridge_case(*, elapsed=.548, processing_age=.10):
    bank, gallery, anchor = before1529()
    state = bank._similar_follow_states[(1, 24)]
    previous = state['observation']
    verified = DetectorObservation(1522, state['last_timestamp'], previous['integrated_yaw_deg'],
        tuple(previous['detector_bbox']), previous['detector_confidence'], 640, 480)
    fast = replace(verified, capture=1523, timestamp=verified.timestamp+.249)
    proof = DetectorProof(1, 24, verified, fast, 2, fast_count=1, permission='similar_follow')
    current = metadata(1524, verified.timestamp+elapsed, verified.bbox, yaw=verified.yaw, search=False)
    current.update(image_width=640, image_height=480, detector_confidence=.95,
                   source_detection_index=0, identity_competition=dict(
                       uid=1, frame_index=606, candidate_count=1, source_detection_index=0,
                       passed=True, reason='single_candidate'))
    bridge = dict(proof=proof, identity=bank.identities[1], state=state,
        assignment=bank.last_assignments[24], capture_frame_id=1524,
        capture_timestamp=current['capture_timestamp'], now=current['capture_timestamp']+processing_age)
    return bank, gallery, anchor, current, bridge


def assign_detector_bridge(bank, current, bridge, *, full=.17, count=1):
    box = current['detector_bbox']
    return bank.assign(track_id=24, feature=feature(full) if full is not None else None,
        partial_feature=None, confidence=.95, area=(box[2]-box[0])*(box[3]-box[1]),
        frame_index=606, candidate_count=count, bbox_quality_ok=True,
        bbox_quality_tier='strong', sample_metadata=current, detector_position_bridge=bridge)


@pytest.mark.parametrize('with_bridge', [False, True])
def test_fast_position_bridges_current_full_without_retiming_or_learning(with_bridge):
    bank, gallery, anchor, current, bridge = detector_full_bridge_case()
    original = bridge['state']
    frozen = deepcopy(original)
    # FULL result finishes after the old motion lease. It uses a current
    # independent feature, not a request to resume that expired motor lease.
    assert bridge['now'] > bridge['proof'].deadline
    assert assign_detector_bridge(bank, current, bridge if with_bridge else None) == int(with_bridge)
    assert original == frozen
    assert gallery_snapshot(bank) == gallery
    assert bank.identities[1].last_strong_observation == anchor
    if with_bridge:
        result = bank.last_assignments[24]
        detail = result['similar_follow']['detector_position_bridge']
        assert detail['verified_capture'] == 1522 and detail['position_capture'] == 1523
        assert detail['current_capture'] == 1524 and not detail['learning_allowed']
        assert detail['original_deadline'] == bridge['proof'].deadline
        assert result['template_update_quarantined'] and not result['bank_updated']
        assert bank._similar_follow_states[(1, 24)]['last_cap'] == 1524
        # An old bridge is single-source evidence, not a renewable credential.
        assert bank._similar_detector_position_state(1, 24, feature(.17), current, bridge) is None


@pytest.mark.parametrize('failure', ['state_replaced', 'entry_replaced', 'assignment_replaced',
    'wrong_current', 'wrong_timestamp', 'no_fast', 'old_fast', 'wrong_uid', 'wrong_raw',
    'old_full', 'old_bbox', 'source_gallery', 'new_gallery', 'no_feature', 'expired',
    'stale_result', 'negative_age', 'stale_observation', 'competition', 'multiple', 'conflict'])
def test_detector_position_bridge_never_relabels_or_renews_old_evidence(failure):
    bank, gallery, _, current, bridge = detector_full_bridge_case()
    full, count = .17, 1
    proof = bridge['proof']
    if failure == 'state_replaced': bridge['state'] = deepcopy(bridge['state'])
    elif failure == 'entry_replaced': bridge['identity'] = deepcopy(bridge['identity'])
    elif failure == 'assignment_replaced': bridge['assignment'] = deepcopy(bridge['assignment'])
    elif failure == 'wrong_current': bridge['capture_frame_id'] += 1
    elif failure == 'wrong_timestamp': bridge['capture_timestamp'] += .001
    elif failure == 'no_fast': bridge['proof'] = replace(proof, fast_count=0)
    elif failure == 'old_fast': bridge['proof'] = replace(proof, previous=proof.verified)
    elif failure == 'wrong_uid': bridge['proof'] = replace(proof, uid=2)
    elif failure == 'wrong_raw': bridge['proof'] = replace(proof, track_id=25)
    elif failure == 'old_full':
        bridge['proof'] = replace(proof, verified=replace(proof.verified, timestamp=proof.verified.timestamp-.001))
    elif failure == 'old_bbox':
        bridge['proof'] = replace(proof, verified=replace(proof.verified, bbox=(1., 1., 200., 470.)))
    elif failure == 'source_gallery': bridge['assignment']['similar_follow']['gallery_distance'] = .301
    elif failure == 'new_gallery': full = .301
    elif failure == 'no_feature': full = None
    elif failure == 'expired':
        current['capture_timestamp'] = bridge['capture_timestamp'] = proof.deadline
        bridge['now'] = proof.deadline+.01
    elif failure == 'stale_result': bridge['now'] = current['capture_timestamp']+.351
    elif failure == 'negative_age': bridge['now'] = current['capture_timestamp']-.001
    elif failure == 'stale_observation': current['is_fresh'] = False
    elif failure == 'competition': current['identity_competition']['passed'] = False
    elif failure == 'multiple': current['candidate_count'], count = 2, 2
    elif failure == 'conflict': bank._mapped_geometry_conflicts[24] = dict(uid=1)
    if failure == 'no_feature':
        # Existing mapped-ID handling can retain a UID without an embedding;
        # that is not this bridge's independent FULL acceptance permission.
        assert bank._similar_detector_position_state(1, 24, None, current, bridge) is None
        assign_detector_bridge(bank, current, bridge, full=full, count=count)
        assert not bank.last_assignments[24].get('similar_follow', {}).get('detector_position_bridge')
        assert bank._similar_follow_states[(1, 24)]['last_cap'] == 1522
    else:
        assert assign_detector_bridge(bank, current, bridge, full=full, count=count) == 0
    assert gallery_snapshot(bank) == gallery
