"""Independent identity recovery uses the existing paired writer, no extra hold.

The capture contract is exercised separately from appearance policy. Serial
and camera samples are in memory; no robot or model is used here.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from test_cap1542_reacquire_handoff import completed_candidate, prepare_consumer
from test_cap609_reacquire_depth_transfer import handoff, queue_capture
from test_short_follow_adapter import paired, owner
from test_search_observation_arbitration import _record
from car_control_modular.detector_identity_lease import publish_visual_identity_evidence


def recovery_candidate(h):
    # Recorded CAP548 bounds: current target is right, previous search is left.
    candidate = completed_candidate(h, bbox=(357.6, 4.6, 481.2, 471.0))
    assignment = candidate['debug']['assignment']
    old = assignment.pop('similar_follow')
    assignment.pop('identity_permission')
    assignment.pop('mapped_uid')  # Ordinary late-reacquire need not emit it.
    assignment['reason'] = 'preferred_search_late_reacquire'
    assignment['reacquire_geometry']['independent_recovery_confirmation'] = dict(
        uid=1, raw_track_id=3, capture_frame_id=old['capture_frame_id'],
        capture_timestamp=old['capture_timestamp'], frame_index=h.obj.frame_index,
        count=2, completed_confirmation=True, learning_allowed=False)
    return candidate


def test_completed_independent_recovery_reaches_right_yaw_without_second_identity_wait(
        handoff, monkeypatch, caplog):
    h = handoff
    candidate = recovery_candidate(h)
    prepare_consumer(h, deepcopy(candidate['debug']['assignment']), monkeypatch)
    h.obj._consume_track_records([
        _record(track=3, uid=1, bbox=candidate['bbox'], score=.889)], 640, 480, 'test')
    assert h.obj.search_state == 'none'
    plan = h.obj._short_follow.snapshot().plan
    assert plan is not None and plan.moving and not plan.forwarding
    assert plan.left_rpm > 0 > plan.right_rpm
    assert plan.longitudinal_reason == 'reacquire_depth_pending'
    assert plan.expires_at == pytest.approx(min(plan.depth_timestamp+.3,
                                               plan.capture_timestamp+.5))
    h.motor._service_short_follow()
    assert h.driver.pairs == [(plan.left_rpm, -plan.right_rpm)]
    assert not h.driver.stops and not h.obj._queued_calls


@pytest.mark.parametrize('change', [
    'pending', 'one_frame', 'wrong_uid', 'wrong_raw', 'old_capture', 'old_timestamp',
    'old_frame', 'can_learn', 'mapped_other', 'rejected', 'competition', 'geometry',
    'expired_visual', 'label_only',
])
def test_independent_recovery_requires_current_complete_proof(handoff, change):
    h = handoff
    candidate = recovery_candidate(h)
    a = candidate['debug']['assignment']
    proof = a['reacquire_geometry']['independent_recovery_confirmation']
    if change == 'pending': proof['completed_confirmation'] = False
    elif change == 'one_frame': proof['count'] = 1
    elif change == 'wrong_uid': proof['uid'] = 2
    elif change == 'wrong_raw': proof['raw_track_id'] = 7
    elif change == 'old_capture': proof['capture_frame_id'] -= 1
    elif change == 'old_timestamp': proof['capture_timestamp'] -= .01
    elif change == 'old_frame': proof['frame_index'] -= 1
    elif change == 'can_learn': proof['learning_allowed'] = True
    elif change == 'mapped_other': a['mapped_uid'] = 2
    elif change == 'rejected': a['identity_control_rejected'] = True
    elif change == 'competition': a['identity_competition']['passed'] = False
    elif change == 'geometry': a['reacquire_geometry']['ok'] = False
    elif change == 'expired_visual': h.clock[0] += 1
    elif change == 'label_only': a['reacquire_geometry'].pop('independent_recovery_confirmation')
    assert h.obj._completed_similar_follow_confirmation(a, uid=1, track_id=3) is None
    assert not h.driver.pairs and not h.obj._short_follow.snapshot().active


def test_pending_recovery_yaw_never_renews_depth_on_duplicate(handoff):
    h = handoff
    candidate = recovery_candidate(h)
    assert not h.obj._hold_for_confirmed_search_reacquire([candidate], width=640, height=480)
    assert queue_capture(h)
    original = h.obj._short_follow.snapshot().plan
    h.clock[0] += .05
    assert queue_capture(h)
    assert h.obj._short_follow.snapshot().plan is original
    assert h.obj._reacquire_depth_pending


def test_real_bank_confirmation_is_consumed_by_first_range_and_paired_writer(handoff, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'vision'))
    from test_cap521_independent_recovery import revoked_bank, send, ROWS
    b = revoked_bank()
    assert send(b, 521) == 0
    assert send(b, 524) == 1
    h = handoff
    frame, stamp, box, *_ = ROWS[524]
    h.clock[0] = stamp + .04
    candidate = recovery_candidate(h)
    # Replace every endpoint's provenance, not just the diagnostic CAP label.
    h.obj.frame_index = frame
    h.obj._active_capture_frame_id = 524
    h.obj._active_capture_timestamp = stamp
    h.a.target = replace(h.a.target, bbox=box, depth_observation=replace(
        h.a.target.depth_observation, bbox=box, raw_track_id=9,
        capture_frame_id=524, capture_timestamp=stamp))
    proof = replace(h.obj._validated_visual_observation, track_id=9, capture=524,
                    timestamp=stamp, expires_at=stamp+.5)
    publish_visual_identity_evidence(h.obj, observation=proof, lease=None)
    candidate.update(bbox=box, area=h.a.target.area)
    candidate['rec'].track_id = 9
    candidate['debug']['assignment'] = deepcopy(b.last_assignments[9])
    assert h.obj._completed_similar_follow_confirmation(
        candidate['debug']['assignment'], uid=1, track_id=9) is proof
    assert not h.obj._hold_for_confirmed_search_reacquire([candidate], width=640, height=480)
    assert h.obj.search_state == 'none'
    assert queue_capture(h)
    plan = h.obj._short_follow.snapshot().plan
    assert plan is not None and plan.moving and not plan.forwarding
    # Actual CAP524 is still left; a left command here is correct.
    assert plan.left_rpm < 0 < plan.right_rpm
    h.motor._service_short_follow()
    assert h.driver.pairs == [(plan.left_rpm, -plan.right_rpm)]
    assert not h.driver.stops and not h.obj._queued_calls
