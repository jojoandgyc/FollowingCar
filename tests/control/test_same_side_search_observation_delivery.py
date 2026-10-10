"""Same-side soft observations cannot repeatedly park an existing search.

Recorded scalar/box fixtures come from run_20261010_230500_89908_7557126d:
CAP1076 (log lines 14671/14674), CAP1079 (14700), CAP1098 (15023/15024).
CAP1098's score is the four-decimal diagnostic value, not an invented ReID
observation. New producer provenance fields are explicitly identified below.
Only construction and device/action endpoints are replaced: main's eligibility,
gate, retry/settlement, brake precheck and track consumer execute in memory.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.controllers import FollowPolicyConfig
from car_control_modular.search_candidate_gate import CandidateObservation, SearchCandidateGateDecision
from car_control_modular.search_observation_retry import (
    DEFERRED_EDGE_COVERAGE, DetectorObservationSettlement, SearchObservationRetry,
)
from test_cap393_observation_eligibility import configured
from test_search_observation_arbitration import owner, _record


@dataclass(frozen=True)
class Frame:
    cap: int
    control: int
    stamp: float
    bbox: tuple
    score: float
    kind: str
    raw: int = 15
    distance: float = .2323504090309143
    probe_bbox: tuple = ()


LOW_EXISTING = Frame(1076, 458, 45091.713519455,
    (0., 10.09259033203125, 167.33200073242188, 473.22906494140625),
    .4332599639892578, 'low_existing',
    probe_bbox=(0., 5.0145416259765625, 168.40936279296875, 283.8504638671875))
CROP = Frame(1079, 459, 45091.850240630,
    (1.4696731567382812, 3.0106201171875, 136.43197631835938, 474.2659912109375),
    .9192646145820618, 'crop', distance=.18579089641571045)
DETECTOR_ONLY = Frame(1098, 472, 45092.845595692,
    (0., 32.54167175292969, 73.10037231445312, 471.543701171875),
    .3504, 'detector_only',
    probe_bbox=(0., 37.152557373046875, 54.27543258666992, 470.91400146484375))
CENTER = (220., 20., 410., 460.)


def _mirror(box):
    return (640-box[2], box[1], 640-box[0], box[3])


def install(o, row, monkeypatch, *, mirror=False):
    box = _mirror(row.bbox) if mirror else row.bbox
    direction = 'right' if mirror else 'left'
    o.search_direction = o._follow_controller.search_direction = direction
    o._active_capture_frame_id, o._active_capture_timestamp = row.cap, row.stamp
    o.frame_index = row.control
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: row.stamp+.08)
    o._follow_controller.cfg = FollowPolicyConfig()
    o._search_epoch = 8
    # These are existing clocks, not motion grants created by the candidates.
    o._follow_controller._search_rotation_started_at = 45091.04
    o._follow_controller._direction_latest_visible_capture_id = 1060
    o._follow_controller._direction_latest_visible_timestamp = 45091.006
    o._action_runtime.search_reacquire_brake_pending = lambda **kw: False
    o._action_runtime.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=row.stamp+.07, trustworthy=True,
        left_forward_rpm=7. if mirror else -7.,
        right_forward_rpm=-7. if mirror else 7.)
    o._action_runtime.request_search_reacquire_brake = lambda *a, **kw: pytest.fail(
        'soft same-side evidence requested a physical brake')
    o._action_runtime.cancel_search_reacquire_brake = lambda *a, **kw: pytest.fail(
        'observation release cancelled a physical brake')
    metadata = dict(is_fresh=True, capture_frame_id=row.cap, capture_timestamp=row.stamp,
        frame_index=row.control, control_frame_id=row.control, source_detection_index=0,
        detector_confidence=row.score, detector_bbox=box,
        quality_bbox_ok=row.kind != 'crop',
        bbox_quality_tier='weak' if row.kind == 'crop' else 'strong',
        quality_bbox_reason='edge_touch>2' if row.kind == 'crop' else '',
        search_reacquire_context_active=True, search_direction_compatible=True)
    assignment = dict(uid=0, mapped_uid=1, identity_control_rejected=True,
        reason=('low_score_observation_rejected' if row.kind == 'low_existing'
                else 'secondary_evidence_unavailable'),
        feature_available=True, bank_updated=False, recent_bank_updated=False,
        learning_written_tiers=[], template_learning=dict(status='not_requested'),
        match_evidence=dict(matched_uid=1, match_source='strong',
                            strong_distance=row.distance, distance=row.distance),
        identity_competition=dict(uid=1, frame_index=row.control,
            source_detection_index=0, candidate_count=1, passed=True),
        template_recent_evidence=dict(count=0, comparable_count=0),
        template_recent_partial_evidence=dict(count=0, comparable_count=0))
    if row.kind == 'low_existing':
        # Newly exposed producer contract, absent from the old log. It reports
        # actual low-score existing-track association; it is NOT identity proof.
        metadata.update(low_score_continuation=True,
            association_reason='low_score_existing_track', association_confidence_limit=.5,
            association_previous_capture_frame_id=row.cap-2,
            association_previous_capture_timestamp=row.stamp-.100056089)
        assignment['low_score_observation_blocked'] = False
    else:
        assignment.update(reacquire_partial_comparable=False,
                          reacquire_partial_state='unknown', bbox_quality_ok=False)
    observations = [] if row.kind == 'detector_only' else [dict(
        raw_track_id=row.raw, uid=0, detector_bbox=box, sample_metadata=metadata,
        assignment=assignment)]
    o._assignments = {} if not observations else {row.raw: assignment}
    o._rknn_pipeline = SimpleNamespace(tracker=SimpleNamespace(
        last_identity_observations=observations,
        _frame_context=dict(capture_frame_id=row.cap, capture_timestamp=row.stamp),
        config=SimpleNamespace(min_confidence=.5)))
    formal = (CandidateObservation(box, row.score),)
    probe = () if not row.probe_bbox else (CandidateObservation(
        _mirror(row.probe_bbox) if mirror else row.probe_bbox, .24865354597568512),)
    return box, assignment, metadata, formal, probe


def defer(o, row, formal, *, stale=False, search_active=True):
    return o._deferred_search_observation_bboxes(search_active=search_active,
        width=640, height=480, formal_candidates=formal, capture_id=row.cap,
        capture_timestamp=row.stamp, stale=stale)


def deliver(o, row, box, formal, probe=(), *, qualified=None):
    """Main's ordered delivery, including its surviving prior-hold OR branch."""
    if qualified is None:
        qualified = defer(o, row, formal)
    d = o._search_candidate_gate.update(timestamp=row.stamp, search_active=True,
        width=640, height=480, formal_candidates=formal, probe_candidates=probe,
        deferred_observation_bboxes=qualified)
    d = o._retry_search_candidate_observation(d, search_active=True, width=640,
        height=480, formal_candidates=formal, capture_id=row.cap, capture_timestamp=row.stamp)
    if d.pause_rotation or d.completed:
        o._apply_search_candidate_gate_decision(d, prepare_only=True)
    bounded_pending = (o._search_evidence_observation_active
        and runtime.time.monotonic() < o._search_evidence_observation_deadline)
    pause = (d.pause_rotation and not d.completed) or bounded_pending
    # The candidate is not an identity match. Independently recheck the brake
    # entry that runs after the observation gate in the real visual workflow.
    o._hold_search_reacquire_brake(bbox=box, width=640, eligible=False,
                                 confirmed=False, raw_track_id=row.raw)
    o._search_evidence_pause_current_frame = pause
    o._follow_controller.set_search_observation_hold(pause)
    records = [] if row.kind == 'detector_only' else [
        _record(track=row.raw, uid=0, bbox=box, score=row.score)]
    o._consume_track_records(records, 640, 480, 'same_side_delivery_test')
    assert all(record.reid_uid == 0 for record in records)
    return d


def authority_snapshot(o):
    return (o._follow_controller.active_target_id, o.search_direction,
        o._follow_controller.search_direction, o._search_epoch,
        o._follow_controller._search_rotation_started_at,
        o._follow_controller._direction_latest_visible_capture_id,
        o._follow_controller._direction_latest_visible_timestamp)


def assert_search_only(o, decision, before, original):
    assert decision.reason == DEFERRED_EDGE_COVERAGE
    assert not decision.entered and not decision.pause_rotation
    assert not decision.preferred_target_match
    assert not o._search_evidence_observation_active
    assert not o._search_evidence_pause_current_frame
    assert not o._follow_controller._search_observation_hold
    assert not o._search_observation_retry.active
    assert not o._search_detector_settlement.active
    assert authority_snapshot(o) == before
    assert o._assignments == original  # no UID promotion or learning mutation
    assert not o._deferred_timeout  # no timeout/lease extension
    assert 'publish' not in o._context_events  # no forward ROI publication
    assert o._events[-1][0:2] == ('normal', [])
    assert all(event[0] != 'queue' for event in o._events)


@pytest.mark.parametrize('mirror', [False, True], ids=['left', 'right'])
@pytest.mark.parametrize('row', [LOW_EXISTING, DETECTOR_ONLY], ids=['cap1076', 'cap1098'])
def test_recorded_low_score_paths_preserve_only_existing_bounded_search(
        configured, monkeypatch, row, mirror):
    o = configured
    box, _, _, formal, probe = install(o, row, monkeypatch, mirror=mirror)
    before, original = authority_snapshot(o), deepcopy(o._assignments)
    d = deliver(o, row, box, formal, probe)
    assert_search_only(o, d, before, original)
    assert not o._search_observation_retry.spent


@pytest.mark.parametrize('mirror', [False, True], ids=['left', 'right'])
def test_high_low_alternation_and_raw_change_never_rearm_stop_or_identity(
        configured, monkeypatch, mirror):
    o = configured
    # First three samples are recorded; later rows repeat their exact geometry
    # with explicitly synthetic current captures and a replacement raw tracker.
    rows = (LOW_EXISTING, CROP, DETECTOR_ONLY,
        replace(CROP, cap=1101, control=474, stamp=45093.013728001, raw=16),
        replace(DETECTOR_ONLY, cap=1102, control=475, stamp=45093.063728001),
        replace(LOW_EXISTING, cap=1103, control=476, stamp=45093.113728001, raw=16))
    for row in rows:
        box, _, _, formal, probe = install(o, row, monkeypatch, mirror=mirror)
        before, original = authority_snapshot(o), deepcopy(o._assignments)
        d = deliver(o, row, box, formal, probe)
        assert_search_only(o, d, before, original)
        assert not o._search_observation_retry.spent
    assert len(o._events) == len(rows)


@pytest.mark.parametrize('kind', ['detector', 'retry', 'runtime_hold'])
@pytest.mark.parametrize('row', [LOW_EXISTING, DETECTOR_ONLY], ids=['cap1076', 'cap1098'])
def test_current_soft_candidate_releases_old_observation_hold_but_not_its_budget(
        configured, monkeypatch, kind, row):
    o = configured
    box, _, _, formal, probe = install(o, row, monkeypatch)
    o._search_observation_retry = SearchObservationRetry()
    o._search_observation_retry.spend_detector_budget((8, 1))
    o._search_detector_settlement = DetectorObservationSettlement()
    if kind == 'retry':
        o._search_observation_retry.active = True
        o._search_observation_retry.started_at = row.stamp-.05
        o._search_observation_retry.deadline = row.stamp+.25
        o._search_observation_retry.last_capture = (row.cap-1, row.stamp-.05)
        o._search_observation_retry.bbox = box
        o._search_observation_retry.score = row.score
    elif kind == 'detector':
        o._search_detector_settlement.update(
            SearchCandidateGateDecision(entered=True, pause_rotation=True,
                bbox=box, score=row.score, source='formal'),
            now=row.stamp-.05, capture_timestamp=row.stamp-.05, capture_id=row.cap-1,
            search_active=True, zero_sent_at=None, feedback=None, max_hold_sec=.3)
    o._search_evidence_observation_active = True
    o._search_evidence_observation_deadline = row.stamp+.3
    o._search_evidence_pause_current_frame = True
    o._follow_controller.set_search_observation_hold(True)
    before, original = authority_snapshot(o), deepcopy(o._assignments)
    d = deliver(o, row, box, formal, probe)
    assert d.completed
    assert_search_only(o, d, before, original)
    assert o._search_observation_retry.spent
    assert o._search_evidence_observation_deadline == 0.


@pytest.mark.parametrize('row', [LOW_EXISTING, DETECTOR_ONLY], ids=['cap1076', 'cap1098'])
def test_brake_arriving_after_defer_check_still_owns_consumer(configured, monkeypatch, row):
    o = configured
    box, _, _, formal, probe = install(o, row, monkeypatch)
    qualified = defer(o, row, formal)
    assert qualified == (box,)
    o._search_evidence_observation_active = True
    o._search_evidence_observation_deadline = row.stamp+.3
    request = SimpleNamespace(capture_frame_id=row.cap, reason='real_search_brake')
    o._action_runtime._search_reacquire_brake_request = request
    o._action_runtime.search_reacquire_brake_pending = lambda **kw: True
    before, original = authority_snapshot(o), deepcopy(o._assignments)
    d = deliver(o, row, box, formal, probe, qualified=qualified)
    assert d.reason == DEFERRED_EDGE_COVERAGE and d.completed
    assert not o._search_evidence_observation_active
    assert not o._search_detector_settlement.active and not o._search_observation_retry.active
    assert o._action_runtime._search_reacquire_brake_request is request
    assert authority_snapshot(o) == before and o._assignments == original
    assert not o._events and not o._deferred_timeout
    assert 'publish' not in o._context_events


@pytest.mark.parametrize('row', [LOW_EXISTING, DETECTOR_ONLY], ids=['cap1076', 'cap1098'])
@pytest.mark.parametrize('fault', ['opposite', 'central', 'multiple', 'stale_flag',
    'stale_age', 'future', 'explicit_stop', 'shutdown', 'hard_brake', 'not_running',
    'no_uid', 'not_searching'])
def test_unsafe_or_unrelated_candidate_never_defers(configured, monkeypatch, row, fault):
    o = configured
    _, _, _, formal, _ = install(o, row, monkeypatch)
    if fault == 'opposite': o._follow_controller.search_direction = 'right'
    elif fault == 'central':
        formal = (CandidateObservation(CENTER, row.score),)
        for observation in o._rknn_pipeline.tracker.last_identity_observations:
            observation['detector_bbox'] = CENTER
            observation['sample_metadata']['detector_bbox'] = CENTER
    elif fault == 'multiple': formal += (CandidateObservation(CENTER, .9),)
    elif fault == 'stale_age': monkeypatch.setattr(runtime.time, 'monotonic', lambda: row.stamp+.20)
    elif fault == 'future': monkeypatch.setattr(runtime.time, 'monotonic', lambda: row.stamp-.01)
    elif fault == 'explicit_stop': o._explicit_stop_requested = True
    elif fault == 'shutdown': o._runtime_shutdown_requested = True
    elif fault == 'hard_brake': o._brake_hold_active = True
    elif fault == 'not_running': o.running = False
    elif fault == 'no_uid': o._follow_controller.active_target_id = None
    assert defer(o, row, formal, stale=fault == 'stale_flag',
                 search_active=fault != 'not_searching') == ()


@pytest.mark.parametrize('fault', ['old_cap', 'old_stamp', 'not_fresh', 'duplicate',
    'conflict_reason', 'partial_conflict', 'geometry_conflict', 'competition_failed',
    'wrong_uid', 'blocked', 'blocked_missing', 'no_low_association',
    'wrong_association', 'old_association', 'learning_permission'])
def test_existing_identity_observation_cannot_fall_back_around_a_rejection(
        configured, monkeypatch, fault):
    o = configured
    row = LOW_EXISTING
    _, assignment, metadata, formal, _ = install(o, row, monkeypatch)
    if fault == 'old_cap': metadata['capture_frame_id'] -= 1
    elif fault == 'old_stamp': metadata['capture_timestamp'] -= .01
    elif fault == 'not_fresh': metadata['is_fresh'] = False
    elif fault == 'duplicate': o._rknn_pipeline.tracker.last_identity_observations *= 2
    elif fault == 'conflict_reason': assignment['reason'] = 'recent_partial_conflict'
    elif fault == 'partial_conflict': assignment['reacquire_partial_state'] = 'conflict'
    elif fault == 'geometry_conflict': assignment['candidate_geometry_conflict'] = True
    elif fault == 'competition_failed': assignment['identity_competition']['passed'] = False
    elif fault == 'wrong_uid': assignment['match_evidence']['matched_uid'] = 2
    elif fault == 'blocked': assignment['low_score_observation_blocked'] = True
    elif fault == 'blocked_missing': del assignment['low_score_observation_blocked']
    elif fault == 'no_low_association': del metadata['low_score_continuation']
    elif fault == 'wrong_association': metadata['association_reason'] = 'geometry_only'
    elif fault == 'old_association': metadata['association_previous_capture_timestamp'] -= 2.
    elif fault == 'learning_permission': assignment['learning_allowed'] = True
    assert defer(o, row, formal) == ()


@pytest.mark.parametrize('fault', ['high_score', 'threshold_equal', 'below_formal',
    'missing_observations', 'unknown_observations', 'old_context_cap', 'old_context_stamp',
    'missing_context', 'missing_config', 'missing_threshold', 'lower_config_threshold'])
def test_detector_only_fallback_needs_explicit_current_empty_identity_and_low_score(
        configured, monkeypatch, fault):
    o = configured
    row = DETECTOR_ONLY
    box, _, _, formal, _ = install(o, row, monkeypatch)
    tracker = o._rknn_pipeline.tracker
    if fault == 'high_score': formal = (CandidateObservation(box, .9),)
    elif fault == 'threshold_equal': formal = (CandidateObservation(box, .5),)
    elif fault == 'below_formal': formal = (CandidateObservation(box, .249),)
    elif fault == 'missing_observations': del tracker.last_identity_observations
    elif fault == 'unknown_observations': tracker.last_identity_observations = None
    elif fault == 'old_context_cap': tracker._frame_context['capture_frame_id'] -= 1
    elif fault == 'old_context_stamp': tracker._frame_context['capture_timestamp'] -= .01
    elif fault == 'missing_context': del tracker._frame_context
    elif fault == 'missing_config': del tracker.config
    elif fault == 'missing_threshold': del tracker.config.min_confidence
    elif fault == 'lower_config_threshold': tracker.config.min_confidence = .30
    assert defer(o, row, formal) == ()


@pytest.mark.parametrize('kind', ['central_low_score', 'edge_high_score'])
def test_unqualified_detector_candidate_keeps_original_bounded_observation_stop(
        configured, monkeypatch, kind):
    row = replace(DETECTOR_ONLY, bbox=CENTER, probe_bbox=()) if kind == 'central_low_score' else replace(
        DETECTOR_ONLY, score=.9, probe_bbox=())
    o = configured
    box, _, _, formal, probe = install(o, row, monkeypatch)
    before = authority_snapshot(o)
    assert defer(o, row, formal) == ()
    d = deliver(o, row, box, formal, probe)
    assert d.entered and d.pause_rotation and not d.completed
    assert not d.preferred_target_match
    assert o._search_evidence_observation_active and o._search_detector_settlement.active
    assert o._search_observation_retry.spent
    assert any(event[0] == 'queue' and event[2] == 'search_candidate_evidence_observe'
               for event in o._events)
    assert not any(event[0] == 'normal' for event in o._events)
    assert authority_snapshot(o) == before and not o._assignments
    assert 'publish' not in o._context_events
