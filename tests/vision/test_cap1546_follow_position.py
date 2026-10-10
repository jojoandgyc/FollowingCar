"""Accepted follow-only identity supplies bounded position, not learning."""
from copy import deepcopy
from dataclasses import replace
import pytest

from rk_vision.yolo11 import Detection
from test_provisional_association import activated, update, BOX
from test_reacquire_crosscheck import feature
from tools.replay_cap334_recovery import gallery_snapshot


def ready(*, box=BOX):
    t, _ = activated(box=box)
    t.set_detector_continuation_context(active_uid=1, allowed=False)
    rows = update(t, 1542, 3.2, distance=.265, box=box)
    assert [(r.track_id, r.reid_uid) for r in rows] == [(3, 1)]
    assert t._low_score_position_anchor is None
    assert t._follow_only_position_anchor['clock'] == (1542, 3.2)
    assert t.identity_bank._reacquire_quarantine.is_held(1)
    return t


def position(t, key='low_score_position_evidence'):
    return next((row['assignment'][key] for row in t.last_identity_observations
                 if key in row['assignment']), None)


def test_low_score_follows_recent_quarantined_identity_position_without_uid_or_learning():
    t = ready(); bank = t.identity_bank
    before = gallery_snapshot(bank)
    strong = bank.identities[1].last_strong_observation
    assert update(t, 1546, 3.397, score=.422, distance=.287) == []
    p = position(t)
    assert p['uid'] == 1 and p['track_id'] == 3 and p['anchor_permission'] == 'similar_follow'
    assert p['reference_capture_frame_id'] == 1542 and p['expires_at'] == pytest.approx(3.7)
    assert p['identity_authorized'] is False and p['learning_allowed'] is False
    assert t.last_identity_observations[0]['uid'] == 0
    assert t._follow_only_position_anchor['clock'] == (1542, 3.2)
    assert bank.identities[1].last_strong_observation is strong
    assert gallery_snapshot(bank) == before and not t.deepsort.tracker.metric.samples
    assert 'low_score_position_evidence' not in bank.last_assignments[3]


def test_search_full_proof_disabled_does_not_erase_position_owned_by_locked_search_uid():
    t = ready()
    t._detector_active(None)  # actual FULL note disables fast proof while search is active
    assert t._detector_active_uid == 0 and t._search_reacquire_uid == 1
    update(t, 1546, 3.397, score=.422, distance=.287)
    assert position(t) is not None
    t.set_search_reacquire_context(active_uid=None, searching=False, direction=None)
    assert t._follow_only_position_anchor is None


@pytest.mark.parametrize('fault', ['expiry', 'replay', 'mapping', 'uid', 'identity_object',
    'track_object', 'competition', 'conflict', 'revoked', 'suspect', 'wrong_person', 'geometry'])
def test_follow_position_retains_expiry_binding_and_current_negative_gates(monkeypatch, fault):
    t = ready(); bank = t.identity_bank
    kw = dict(score=.422, distance=.287); cap, stamp = 1546, 3.397
    if fault == 'expiry': stamp = 3.701
    elif fault == 'replay': cap, stamp = 1542, 3.2
    elif fault == 'mapping': bank.track_to_uid[3] = 2
    elif fault == 'uid': t.set_detector_continuation_context(active_uid=2, allowed=False)
    elif fault == 'identity_object': bank.identities[1] = deepcopy(bank.identities[1])
    elif fault == 'track_object': t.deepsort.tracker.tracks[0] = deepcopy(t.deepsort.tracker.tracks[0])
    elif fault == 'conflict': bank._mapped_geometry_conflicts[3] = dict(uid=1)
    elif fault == 'revoked': bank._geometry_revoked_uids[1] = True
    elif fault == 'suspect': bank._reacquire_control_suspects[1] = dict(track_id=3)
    elif fault == 'wrong_person': kw['distance'] = .8
    elif fault == 'geometry': kw['box'] = (450., 40., 630., 450.)
    elif fault == 'competition':
        original = t._low_score_position_evidence
        def changed(output, meta, assignment):
            meta['identity_competition']['passed'] = False
            return original(output, meta, assignment)
        monkeypatch.setattr(t, '_low_score_position_evidence', changed)
    update(t, cap, stamp, **kw)
    assert position(t) is None


def test_weak_position_cannot_renew_full_reference_or_expiry():
    t = ready()
    for cap, stamp in ((1546, 3.397), (1548, 3.55), (1550, 3.69)):
        update(t, cap, stamp, score=.422, distance=.287)
        assert position(t)['expires_at'] == pytest.approx(3.7)
        assert t._follow_only_position_anchor['clock'] == (1542, 3.2)
    update(t, 1552, 3.71, score=.422, distance=.287)
    assert position(t) is None


def probe(t, *, cap=1545, stamp=3.361, distance=.277, box=BOX, yaw=0.):
    # Actual search-probe method, after a genuine tracker association miss;
    # it must retain its real negative raw ID instead of inventing raw3.
    t._frame_index += 1
    t._frame_context = dict(capture_frame_id=cap, capture_timestamp=stamp, integrated_yaw_deg=yaw)
    ds = [Detection(box, .791, 0)]
    t._current_detections = tuple(ds); t._learning_detections = tuple(ds)
    t.last_identity_observations = []
    result = t._search_probe_record(ds, [feature(distance)], partial_features=[None],
                                   image_width=640, image_height=480)
    return result


def test_detector_probe_reports_actual_raw_and_separate_reference_owner():
    t = ready(); bank = t.identity_bank
    before = gallery_snapshot(bank)
    row = probe(t)
    assert row.track_id == -1 and row.reid_uid == 0
    p = position(t, 'follow_only_position_evidence')
    assert p is not None
    assert p['track_id'] == -1 and p['reference_track_id'] == 3
    assert p['reference_capture_frame_id'] == 1542 and p['expires_at'] == pytest.approx(3.7)
    assert p['source'] == 'follow_only_detector_probe'
    assert p['identity_authorized'] is False and p['learning_allowed'] is False
    assert t.last_identity_observations[0]['raw_track_id'] == -1
    assert not t.last_identity_observations[0]['sample_metadata']['search_observation_only']
    assert gallery_snapshot(bank) == before
    assert 'follow_only_position_evidence' not in bank.last_assignments[-1]
    assert t._follow_only_position_anchor['clock'] == (1542, 3.2)


@pytest.mark.parametrize('fault', ['expiry', 'gallery', 'geometry', 'conflict', 'revoked', 'suspect', 'owner_mapping'])
def test_probe_position_never_promotes_expired_ambiguous_or_contradictory_target(fault):
    t = ready(); bank = t.identity_bank; kw = {}
    if fault == 'expiry': kw['stamp'] = 3.701
    elif fault == 'gallery': kw['distance'] = .301
    elif fault == 'geometry': kw['box'] = (450., 40., 630., 450.)
    elif fault == 'conflict': bank._mapped_geometry_conflicts[3] = dict(uid=1)
    elif fault == 'revoked': bank._geometry_revoked_uids[1] = True
    elif fault == 'suspect': bank._reacquire_control_suspects[1] = dict(track_id=3)
    elif fault == 'owner_mapping': bank.track_to_uid[3] = 2
    probe(t, **kw)
    assert position(t, 'follow_only_position_evidence') is None


def test_logged_cap1542_probe1545_low1546_geometry_and_relative_capture_sequence():
    # Saved boxes, capture intervals and heading deltas; gallery descriptors
    # are synthetic with the recorded distances, not hardware/video replay.
    box1542 = (40.17799377441406, 2.29010009765625, 212.9261474609375, 478.0072021484375)
    box1545 = (127.7252197265625, 12.281661987304688, 288.310302734375, 477.20233154296875)
    box1546 = (137.88424682617188, 15.120330810546875, 303.01702880859375, 476.887451171875)
    t = ready(box=box1542)
    before = gallery_snapshot(t.identity_bank)
    t._detector_active(None)
    row = probe(t, stamp=3.361101485, distance=.2770344, box=box1545,
                yaw=-118.86962200713718 + 116.28203904774304)
    assert row.track_id == -1 and row.reid_uid == 0
    p = position(t, 'follow_only_position_evidence')
    assert p is not None and p['bbox'] == box1545
    assert p['center_jump_ratio'] < .2 and p['reference_capture_frame_id'] == 1542
    records = t.update([Detection(box1546, .422, 0)], [feature(.2867262363)],
        image_width=640, image_height=480, frame_context=dict(
            capture_frame_id=1546, capture_timestamp=3.397052862,
            integrated_yaw_deg=-119.9079081857157 + 116.28203904774304))
    assert records == []
    p = position(t)
    assert p is not None and p['track_id'] == 3 and p['bbox'] == box1546
    assert p['reference_capture_frame_id'] == 1542 and p['expires_at'] == pytest.approx(3.7)
    assert gallery_snapshot(t.identity_bank) == before


def test_probe_replay_or_bad_competition_cannot_continue_position(monkeypatch):
    t = ready()
    probe(t)
    probe(t)
    assert position(t, 'follow_only_position_evidence') is None
    original = t._follow_only_probe_position
    def conflict(raw, meta, assignment, descriptor):
        meta['identity_competition']['passed'] = False
        return original(raw, meta, assignment, descriptor)
    monkeypatch.setattr(t, '_follow_only_probe_position', conflict)
    probe(t, cap=1547, stamp=3.45)
    assert position(t, 'follow_only_position_evidence') is None
    assert t._follow_only_position_anchor is None


def normal_gap_tracker():
    t = ready()
    t.set_search_reacquire_context(active_uid=1, searching=False, direction=None)
    # This is the real runtime setting, unlike library's predicted-age 1.
    t.deepsort.config = replace(t.deepsort.config, max_output_age=0)
    return t


def test_real_normal_tracker_gap_probes_position_without_search_or_identity_assignment(monkeypatch):
    t = normal_gap_tracker(); bank = t.identity_bank
    gallery = gallery_snapshot(bank)
    states, mappings = deepcopy(bank._similar_follow_states), dict(bank.track_to_uid)
    strong = bank.identities[1].last_strong_observation
    old_probe_assignment = bank.last_assignments.get(-1)
    monkeypatch.setattr(bank, 'assign', lambda **_: pytest.fail('normal probe must not assign identity'))
    # Real Kalman/association miss creates tentative raw4; no formal output.
    # No monkeypatch of the tracker or matching machinery is involved.
    rows = update(t, 1545, 3.361, score=.791, distance=.277, box=(300., 40., 500., 450.))
    assert [(row.track_id, row.reid_uid) for row in rows] == [(-1, 0)]
    assert [(track.track_id, track.time_since_update) for track in t.deepsort.tracker.tracks] == [(3, 1), (4, 0)]
    p = position(t, 'follow_only_position_evidence')
    assert p is not None and p['reference_track_id'] == 3 and p['expires_at'] == pytest.approx(3.7)
    row = t.last_identity_observations[0]
    assert row['assignment']['reason'] == 'follow_only_position_probe'
    assert not row['sample_metadata']['search_reacquire_context_active']
    assert t._search_reacquire_uid == 0 and t._search_reacquire_direction is None
    assert bank._similar_follow_states == states and bank.track_to_uid == mappings
    assert bank.identities[1].last_strong_observation is strong and gallery_snapshot(bank) == gallery
    assert bank.last_assignments.get(-1) is old_probe_assignment
    monkeypatch.undo()
    assert update(t, 1546, 3.397, score=.422, distance=.287, box=(300., 40., 500., 450.)) == []
    assert position(t)['reference_capture_frame_id'] == 1542
    assert position(t)['expires_at'] == pytest.approx(3.7)


@pytest.mark.parametrize('failure', ['expiry', 'appearance', 'competition', 'owner_conflict',
                                   'uid', 'mapping', 'confidence', 'quality'])
def test_normal_gap_probe_does_not_create_authority_on_rejection(monkeypatch, failure):
    t = normal_gap_tracker()
    kw = dict(cap=1545, stamp=3.361, score=.791, distance=.277, box=(300., 40., 500., 450.))
    if failure == 'expiry': kw['stamp'] = 3.701
    elif failure == 'appearance': kw['distance'] = .301
    elif failure == 'owner_conflict': t.identity_bank._mapped_geometry_conflicts[3] = dict(uid=1)
    elif failure == 'uid': t.set_search_reacquire_context(active_uid=2, searching=False, direction=None)
    elif failure == 'mapping': t.identity_bank.track_to_uid[3] = 2
    elif failure == 'confidence': kw['score'] = .49
    elif failure == 'quality': kw['box'] = (300., 40., 312., 450.)
    elif failure == 'competition':
        original = t._normal_follow_position_probe
        def two(detections, features, **bounds):
            return original([*detections, Detection((500.,40.,630.,450.),.8,0)],
                            [*features, feature(.28)], **bounds)
        monkeypatch.setattr(t, '_normal_follow_position_probe', two)
    update(t, **kw)
    assert position(t, 'follow_only_position_evidence') is None


def test_old_strong_raw_does_not_hide_current_follow_only_anchor():
    t = ready()
    old = dict(t._follow_only_position_anchor, track_id=2, clock=(1500, 3.), permission='strong')
    t._low_score_position_anchor = old
    update(t, 1546, 3.397, score=.422, distance=.287)
    assert position(t)['track_id'] == 3
    assert position(t)['reference_capture_frame_id'] == 1542


def test_formal_half_confidence_full_gallery_conflict_retires_old_position_anchor():
    t = ready()
    update(t, 1543, 3.3, score=.55, distance=.8)
    assert t._follow_only_position_anchor is None
    probe(t)
    assert position(t, 'follow_only_position_evidence') is None
