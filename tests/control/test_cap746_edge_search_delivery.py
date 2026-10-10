"""Current same-side UID0 crops must not inject a stationary search look.

Actual CAP746..752 scalar/box fields traverse main's observation delivery,
detector gate, retry/settlement and consumer. Dispatch is observed in memory.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.search_candidate_gate import CandidateObservation, SearchCandidateGateDecision
from car_control_modular.search_observation_retry import DEFERRED_EDGE_COVERAGE, SearchObservationRetry
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.controllers import FollowPolicyConfig
from test_cap393_observation_eligibility import configured
from test_search_observation_arbitration import owner, _record


ROWS = [
    (746, 341, 43190.696802555, (450.8029, 1.8844, 638.5201, 478.0947), .2302857),
    (747, 342, 43190.733111510, (445.6320, .8554, 639.2555, 478.4072), .2244980),
    (749, 343, 43190.832889897, (443.4138, 1.5083, 636.3418, 476.4609), .2750975),
    (751, 344, 43190.936557561, (466.2521, .9899, 639.9407, 477.0233), .2221418),
    (752, 345, 43190.996787242, (481.8521, 2.3553, 639.2302, 476.0607), .1971546),
]


def setup_frame(o, row, monkeypatch, mirror=False):
    cap, frame, stamp, box, distance = row
    if mirror:
        box = (640-box[2], box[1], 640-box[0], box[3])
    direction = 'left' if mirror else 'right'
    o.search_direction = o._follow_controller.search_direction = direction
    o._active_capture_frame_id, o._active_capture_timestamp = cap, stamp
    o.frame_index = frame
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: stamp+.08)
    metadata = dict(is_fresh=True, capture_frame_id=cap, capture_timestamp=stamp,
        frame_index=frame, control_frame_id=frame, source_detection_index=0,
        quality_bbox_ok=False, bbox_quality_tier='weak', quality_bbox_reason='edge_touch>2')
    assignment = dict(uid=0, mapped_uid=1, identity_control_rejected=True,
        reason='secondary_evidence_unavailable', reacquire_partial_comparable=False,
        reacquire_partial_state='unknown', bbox_quality_ok=False,
        match_evidence=dict(matched_uid=1, match_source='strong', strong_distance=distance),
        identity_competition=dict(uid=1, frame_index=frame, source_detection_index=0,
            candidate_count=1, passed=True),
        template_recent_evidence=dict(count=0, comparable_count=0),
        template_recent_partial_evidence=dict(count=0, comparable_count=0))
    o._assignments = {11: assignment}
    o._rknn_pipeline = SimpleNamespace(tracker=SimpleNamespace(last_identity_observations=[dict(
        raw_track_id=11, uid=0, detector_bbox=box, sample_metadata=metadata)]))
    return box, assignment, metadata


def tick(o, row, box):
    cap, _, stamp, _, _ = row
    formal = (CandidateObservation(box, .89),)
    deferred = o._deferred_search_observation_bboxes(search_active=True,
        width=640, height=480, formal_candidates=formal,
        capture_id=cap, capture_timestamp=stamp)
    decision = o._search_candidate_gate.update(timestamp=stamp, search_active=True,
        width=640, height=480, formal_candidates=formal, deferred_observation_bboxes=deferred)
    decision = o._retry_search_candidate_observation(decision, search_active=True,
        width=640, height=480, formal_candidates=formal, capture_id=cap, capture_timestamp=stamp)
    if decision.pause_rotation or decision.completed:
        o._apply_search_candidate_gate_decision(decision, prepare_only=True)
    o._search_evidence_pause_current_frame = decision.pause_rotation and not decision.completed
    o._follow_controller.set_search_observation_hold(o._search_evidence_pause_current_frame)
    return decision


@pytest.mark.parametrize('mirror', [False, True])
def test_recorded_crops_reach_normal_search_without_stop_or_identity_claim(configured, monkeypatch, mirror):
    o = configured
    for row in ROWS:
        box, assignment, _ = setup_frame(o, row, monkeypatch, mirror)
        original = deepcopy(assignment)
        d = tick(o, row, box)
        assert d.reason == DEFERRED_EDGE_COVERAGE
        assert not d.pause_rotation and not d.preferred_target_match
        assert not o._search_evidence_observation_active
        assert not o._search_detector_settlement.active and not o._search_observation_retry.active
        o._consume_track_records([_record(track=11, uid=0, bbox=box)], 640, 480, 'test')
        assert o._events[-1][0] == 'normal'  # normal bounded search remains the owner
        assert o._events[-1][1] == []  # never pass UID0 off as a confirmed person
        assert all(event[0] != 'queue' for event in o._events)
        assert o._assignments[11] == original
        assert not o._deferred_timeout and not o._search_observation_retry.spent
    assert len(o._events) == len(ROWS)


@pytest.mark.parametrize('kind', ['detector', 'retry', 'runtime_hold'])
def test_defer_retires_only_prior_observation_not_real_brake_or_budget(configured, monkeypatch, kind):
    o = configured
    box, _, _ = setup_frame(o, ROWS[0], monkeypatch)
    # Set up the existing in-progress look before this current crop is delivered.
    stamp = ROWS[0][2]
    o._search_observation_retry = SearchObservationRetry()
    o._search_observation_retry.spend_detector_budget((0, 1))
    if kind == 'retry':
        o._search_observation_retry.active = True
        o._search_observation_retry.deadline = stamp+.25
    elif kind == 'detector':
        from car_control_modular.search_observation_retry import DetectorObservationSettlement
        o._search_detector_settlement = DetectorObservationSettlement()
        o._search_detector_settlement.active = True
    o._search_evidence_observation_active = True
    o._search_evidence_observation_deadline = stamp+.3
    marker = object()
    o._action_runtime.brake_marker = marker
    d = tick(o, ROWS[0], box)
    assert d.completed and d.reason == DEFERRED_EDGE_COVERAGE
    assert not d.preferred_target_match and not d.pause_rotation
    assert not o._search_evidence_observation_active and not o._search_observation_retry.active
    assert o._search_observation_retry.spent and o._action_runtime.brake_marker is marker
    assert not o._events and not o._deferred_timeout


@pytest.mark.parametrize('fault', ['opposite', 'old_cap', 'old_time', 'stale', 'ambiguous',
                                  'explicit_stop', 'shutdown', 'brake', 'not_running'])
def test_main_delivery_does_not_defer_without_current_eligible_search(configured, monkeypatch, fault):
    o = configured
    row = ROWS[0]
    box, assignment, metadata = setup_frame(o, row, monkeypatch)
    formal = (CandidateObservation(box, .89),)
    if fault == 'opposite': o._follow_controller.search_direction = 'left'
    elif fault == 'old_cap': metadata['capture_frame_id'] -= 1
    elif fault == 'old_time': metadata['capture_timestamp'] -= .01
    elif fault == 'stale': monkeypatch.setattr(runtime.time, 'monotonic', lambda: row[2]+.20)
    elif fault == 'ambiguous': formal += (CandidateObservation((10., 10., 180., 470.), .9),)
    elif fault == 'explicit_stop': o._explicit_stop_requested = True
    elif fault == 'shutdown': o._runtime_shutdown_requested = True
    elif fault == 'brake': o._brake_hold_active = True
    else: o.running = False
    assert o._deferred_search_observation_bboxes(search_active=True, width=640, height=480,
        formal_candidates=formal, capture_id=row[0], capture_timestamp=row[2]) == ()


@pytest.mark.parametrize('pending_brake', [False, True])
def test_independent_brake_entry_does_not_reinsert_observation_stop_or_cancel_real_brake(
        configured, monkeypatch, pending_brake):
    o = configured
    box, _, _ = setup_frame(o, ROWS[0], monkeypatch)
    d = tick(o, ROWS[0], box)
    assert d.reason == DEFERRED_EDGE_COVERAGE
    o._follow_controller.cfg = FollowPolicyConfig()
    o._action_runtime.search_reacquire_brake_pending = lambda: pending_brake
    o._action_runtime.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=ROWS[0][2]+.07, trustworthy=True,
        left_forward_rpm=7., right_forward_rpm=-7.)
    o._action_runtime.request_search_reacquire_brake = lambda *_: pytest.fail('new observation brake')
    assert o._hold_search_reacquire_brake(bbox=box, width=640, eligible=False,
        confirmed=False, raw_track_id=11) is pending_brake
    assert not o._events
