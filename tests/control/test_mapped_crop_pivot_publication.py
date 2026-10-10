"""UID0 mapped crop must not expose a false-identity handoff window.

CAP116 boxes/score/identity diagnostics are from run_20261010_234438. Clocks
and the still-live prior pivot are synthetic: that run's actual old pivot had
already expired at CAP116 completion and is deliberately NOT revived here.
No PersonTracker constructor, sensor, camera or motor thread is started.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import HazardState, SensorFrame
from car_control_modular.detector_identity_lease import (
    DetectorIdentityLease, ValidatedVisualObservation, publish_visual_identity_evidence,
)
from car_control_modular.low_quality_lateral import LimitedYawSource
from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowObservation,
)
from car_control_modular.short_follow_executor import ShortFollowExecutor
from test_lateral_zero_runtime import NOW, owner
from test_search_observation_arbitration import _record
from test_short_follow_adapter import paired, process


DETECTOR_BOX = (514.64794921875, 3.1103668212890625, 639.267578125, 475.29217529296875)
DISPLAY_BOX = (432.4482872233394, 0., 639., 479.)
SCORE = .7949378490447998


def set_capture(a, cap=116, stamp=NOW-.05):
    a.cap, a.stamp = cap, stamp
    o = a.owner
    o._active_capture_frame_id, o._active_capture_timestamp = cap, stamp
    o._rknn_pipeline.last_identity_processing = dict(mode='full', full_features_current=True,
        capture_frame_id=cap, capture_timestamp=stamp, detector_result_complete=True,
        detector_person_count=1, detector_result_capture_frame_id=cap,
        detector_result_capture_timestamp=stamp)
    a.meta.update(capture_frame_id=cap, capture_timestamp=stamp)


@pytest.fixture
def crop(paired, monkeypatch):
    a = paired
    process(a)
    a.clock = [NOW]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: a.clock[0])
    o = a.owner
    o._short_follow = ShortFollowController(ShortFollowConfig(enabled=True, depth_ttl_sec=.35))
    o._short_follow.activate(1, NOW-.4)
    a.plan = o._short_follow.update(ShortFollowObservation(
        1, 110, NOW-.25, NOW-.10, 1.48, .75, 1.497), NOW-.08)
    assert a.plan.pivot and a.plan.base_rpm == 0
    o._short_follow_adapter.controller = o._short_follow
    a.proof = ValidatedVisualObservation(1, 1, 110, NOW-.25, NOW-.2, NOW+.25, 'full')
    a.publication = publish_visual_identity_evidence(o, observation=a.proof, lease=None)
    a.epoch = [42]
    o._depth_async_scheduler = SimpleNamespace(publication_snapshot=lambda: (a.epoch[0], None))
    o._identity_processing_watermark = (110, a.proof.timestamp)
    o.frame_index = 33
    competition = dict(uid=1, frame_index=33, candidate_count=1, source_detection_index=0,
        passed=True, reason='single_candidate', distance=.19008761644363403)
    a.assignment = dict(uid=0, mapped_uid=1, reason='mapped_weak_observed',
        distance=.19008761644363403, strong_distance=.19008761644363403,
        match_source='strong', bank_updated=False, bbox_quality_ok=False,
        bbox_quality_tier='weak', bbox_quality_reason='edge_touch>2',
        reacquire_geometry_ok=None, search_excluded=False,
        similar_follow=dict(status='reject', reason='local_geometry_conflict', count=0),
        identity_competition=deepcopy(competition))
    a.meta = dict(track_id=1, is_fresh=True, quality_bbox_ok=False,
        quality_bbox_reason='edge_touch>2', low_score_continuation=False,
        source_detection_index=0, identity_competition=competition,
        detector_bbox=DETECTOR_BOX, bbox=DISPLAY_BOX, detector_confidence=SCORE,
        template_learning_risk=dict(observed=True, risky=False, reason='clear'))
    a.row = dict(raw_track_id=1, uid=0, frame_index=33,
        sample_metadata=a.meta, assignment=a.assignment)
    o._rknn_pipeline = SimpleNamespace(last_frame_width=640, last_frame_height=480,
        tracker=SimpleNamespace(last_identity_observations=[a.row],
            associated_position_contradiction=lambda *_: None))
    o._identity_assignment_debug_for_track = lambda raw: dict(a.assignment) if raw == 1 else {}
    a.rec = _record(track=1, uid=0, bbox=DISPLAY_BOX, score=SCORE)
    o._get_obstacle_status = lambda: pytest.fail('identity publication must not poll hardware')
    set_capture(a)
    return a


def deliver(a, *, records=None, stale=False):
    return a.owner._update_detector_identity_lease([a.rec] if records is None else records,
        a.cap, a.stamp, now=a.clock[0], stale=stale, expected_epoch=42)


def executor_reason(a):
    # Real final-reader policy without constructing a serial backend.
    writer = object.__new__(ShortFollowExecutor)
    writer.owner = a.owner
    writer.runtime = SimpleNamespace()
    writer.controller = lambda: a.owner._short_follow
    return writer._plan_reason(a.owner._short_follow.snapshot(), a.clock[0],
                               a.owner._visual_identity_evidence)


@pytest.mark.parametrize('mirror', [False, True])
def test_current_uid0_crop_publication_keeps_only_exact_old_pivot_until_lateral_handoff(crop, mirror):
    a, o = crop, crop.owner
    if mirror:
        a.rec = _record(track=1, uid=0,
            bbox=(640-DISPLAY_BOX[2], 0., 640-DISPLAY_BOX[0], 479.), score=SCORE)
        a.plan = replace(a.plan, left_rpm=-a.plan.left_rpm,
            right_rpm=-a.plan.right_rpm, reason='pivot_left')
        o._short_follow._state = replace(o._short_follow.snapshot(), plan=a.plan)
    before, floor = o._short_follow.snapshot(), o._short_follow._source_floor
    assignment = deepcopy(a.assignment)
    assert deliver(a)
    # Deliberate interleaving: FULL result is published, but control has NOT
    # created its LimitedYawSource/retired the paired owner yet.
    assert getattr(o, '_limited_yaw_source', None) is None
    assert executor_reason(a) is None
    proof = o._validated_visual_observation
    assert proof == replace(a.proof, continuation_sample_timestamp=a.plan.depth_timestamp,
                           expires_at=min(a.proof.expires_at, a.plan.expires_at))
    assert proof.permits_depth(1, a.plan.depth_timestamp, NOW)
    assert not proof.permits_depth(1, a.plan.depth_timestamp+.001, NOW)
    assert o._visual_identity_evidence.lease is None
    assert o._short_follow.snapshot() is before and a.assignment == assignment
    assert a.rec.reid_uid == 0 and not o._queued_calls

    # Existing limited-yaw path consumes the restricted publication. It must
    # retire without turning the completed crop into a STOP source watermark.
    bbox = runtime.PersonTracker._track_record_bbox(a.rec)
    target = replace(a.target, bbox=bbox)
    o._limited_yaw_source = LimitedYawSource(1, 1, a.cap, a.stamp, bbox, o._visual_identity_evidence)
    frame = SensorFrame(width=640, height=480, persons=[target],
        capture_frame_id=a.cap, capture_timestamp=a.stamp)
    assert not o._short_follow_adapter.handle(frame, target, is_fresh_depth=False,
        control_source='vision', target_steerable=False, low_quality_visible=True, now=NOW)
    assert o._short_follow.snapshot().reason == 'lateral_handoff'
    assert o._short_follow._source_floor == floor


def test_no_crop_can_roll_sample_identity_deadline_or_lease(crop):
    a = crop
    lease = DetectorIdentityLease(1, 1, 108, NOW-.3, 110, NOW-.25, NOW+.1)
    publish_visual_identity_evidence(a.owner, observation=a.proof, lease=lease)
    assert deliver(a)
    publication = a.owner._visual_identity_evidence
    assert publication.lease is lease
    for cap, offset in ((117, .02), (118, .06)):
        a.clock[0] = NOW+offset
        set_capture(a, cap, a.clock[0]-.01)
        assert deliver(a)
        assert a.owner._visual_identity_evidence is publication
        assert a.owner._short_follow.snapshot().plan is a.plan
    a.clock[0] = lease.expires_at
    set_capture(a, 119, a.clock[0]-.01)
    assert deliver(a)
    assert a.owner._validated_visual_observation is False
    assert executor_reason(a) == 'identity_not_live'


@pytest.mark.parametrize('fault', ['forward', 'zero', 'opposite', 'plan_expired', 'proof_expired',
    'new_sample', 'sample_only_mismatch', 'raw', 'uid', 'fragment', 'large_occlusion',
    'distance_reject', 'explicit_reject', 'geometry', 'competition', 'metadata_competition',
    'review_conflict', 'recheck', 'excluded', 'risky', 'low_score', 'extra_person',
    'missing_source', 'old_metadata', 'old_active_capture', 'detector_incomplete',
    'detector_count', 'stop', 'shutdown', 'brake', 'cached_unsafe', 'hazard', 'search',
    'epoch', 'stale', 'old_processing'])
def test_any_unqualified_crop_keeps_normal_rejection(crop, fault):
    a, o = crop, crop.owner
    state = o._short_follow.snapshot()
    if fault == 'forward':
        o._short_follow._state = replace(state, plan=replace(a.plan, left_rpm=30,
            right_rpm=20, base_rpm=25, reason='forward'))
    elif fault == 'zero': o._short_follow._state = replace(state, plan=replace(a.plan, left_rpm=0, right_rpm=0))
    elif fault == 'opposite':
        o._short_follow._state = replace(state, plan=replace(a.plan, left_rpm=-7, right_rpm=7, reason='pivot_left'))
    elif fault == 'plan_expired': a.clock[0] = a.plan.expires_at
    elif fault == 'proof_expired':
        publish_visual_identity_evidence(o, observation=replace(a.proof, expires_at=NOW), lease=None)
    elif fault in ('new_sample', 'sample_only_mismatch'):
        sample = a.plan.depth_timestamp+.01
        if fault == 'new_sample':
            publish_visual_identity_evidence(o, observation=replace(a.proof,
                continuation_sample_timestamp=sample), lease=None)
        else:
            publish_visual_identity_evidence(o, observation=replace(a.proof,
                continuation_sample_timestamp=a.plan.depth_timestamp), lease=None)
            o._short_follow._state = replace(state, plan=replace(a.plan, depth_timestamp=sample))
    elif fault == 'raw': a.rec = replace(a.rec, track_id=2)
    elif fault == 'uid': a.assignment['mapped_uid'] = 2
    elif fault == 'fragment': a.assignment['bbox_quality_reason'] = 'edge_touch>2,aspect<0.18'
    elif fault == 'large_occlusion': a.rec = _record(track=1, uid=0, bbox=(0., 0., 639., 479.))
    elif fault == 'distance_reject': a.assignment['reason'] = 'mapped_weak_distance_reject'
    elif fault == 'explicit_reject': a.assignment['identity_control_rejected'] = True
    elif fault == 'geometry': a.assignment['reacquire_geometry'] = dict(ok=False)
    elif fault == 'competition': a.assignment['identity_competition']['passed'] = False
    elif fault == 'metadata_competition': a.meta['identity_competition']['passed'] = False
    elif fault == 'review_conflict': o._rknn_pipeline.tracker.associated_position_contradiction = lambda *_: 'geometry_conflict'
    elif fault == 'recheck': a.assignment['identity_recheck_pending'] = True
    elif fault == 'excluded': a.assignment['search_excluded'] = True
    elif fault == 'risky': a.meta['template_learning_risk']['risky'] = True
    elif fault == 'low_score': a.meta['low_score_continuation'] = True
    elif fault == 'missing_source': o._rknn_pipeline.tracker.last_identity_observations = []
    elif fault == 'old_metadata': a.meta['capture_frame_id'] -= 1
    elif fault == 'old_active_capture': o._active_capture_frame_id -= 1
    elif fault == 'detector_incomplete': o._rknn_pipeline.last_identity_processing['detector_result_complete'] = False
    elif fault == 'detector_count': o._rknn_pipeline.last_identity_processing['detector_person_count'] = 2
    elif fault == 'stop': o._explicit_stop_requested = True
    elif fault == 'shutdown': o._runtime_shutdown_requested = True
    elif fault == 'brake': o._brake_hold_active = True
    elif fault == 'cached_unsafe': o._depth_roi_safety_clear = False
    elif fault == 'hazard': o._current_hazard_state_for_controller = lambda: HazardState(active=True)
    elif fault == 'search': o.search_state = 'searching'
    elif fault == 'epoch': a.epoch[0] += 1
    elif fault == 'old_processing': o._rknn_pipeline.last_identity_processing['capture_frame_id'] -= 1
    records = [a.rec, replace(a.rec, track_id=2)] if fault == 'extra_person' else None
    deliver(a, records=records, stale=fault == 'stale')
    assert o._validated_visual_observation is False
    assert not a.assignment.get('bank_updated')


def test_original_cap116_deadline_is_not_resurrected(crop):
    a = crop
    # Actual last CAP108 pivot (main log 1174): at sample+335.8ms it
    # had only 4.2ms left. The CAP116 identity publication came later.
    a.owner._short_follow._state = replace(a.owner._short_follow.snapshot(),
        plan=replace(a.plan, expires_at=NOW-.025))
    assert deliver(a)
    assert a.owner._validated_visual_observation is False
    assert executor_reason(a) == 'observation_expired'


def interleave_mailbox(a, monkeypatch, event):
    original = a.owner._short_follow.write_snapshot
    @contextmanager
    def entered():
        assert a.owner._control_update_lock._is_owned()
        with original() as _:
            event()
            yield a.owner._short_follow.snapshot()
    monkeypatch.setattr(a.owner._short_follow, 'write_snapshot', entered)


@pytest.mark.parametrize('fault', ['stop', 'revoke', 'forward', 'expiry', 'conflict', 'new_rejection'])
def test_mailbox_revalidation_blocks_concurrent_invalidations(crop, monkeypatch, fault):
    a, o = crop, crop.owner
    def event():
        if fault == 'stop': o._explicit_stop_requested = True
        elif fault == 'revoke': o._short_follow.revoke('test', NOW)
        elif fault == 'forward':
            o._short_follow._state = replace(o._short_follow.snapshot(),
                plan=replace(a.plan, left_rpm=30, right_rpm=20, base_rpm=25, reason='forward'))
        elif fault == 'expiry': a.clock[0] = a.plan.expires_at
        elif fault == 'conflict': a.assignment['identity_control_rejected'] = True
        else: publish_visual_identity_evidence(o, observation=False, lease=False)
    interleave_mailbox(a, monkeypatch, event)
    assert deliver(a)
    assert o._validated_visual_observation is False


def test_same_epoch_committed_depth_binds_only_that_pivot(crop, monkeypatch):
    a, o = crop, crop.owner
    def event():
        a.new_plan = o._short_follow.update(ShortFollowObservation(
            1, 110, NOW-.25, NOW-.02, 1.48, .75, 1.497), NOW)
        assert a.new_plan.pivot
    interleave_mailbox(a, monkeypatch, event)
    assert deliver(a)
    proof = o._validated_visual_observation
    assert proof.continuation_sample_timestamp == a.new_plan.depth_timestamp
    assert not proof.permits_depth(1, a.new_plan.depth_timestamp+.001, NOW)
    assert proof.expires_at == min(a.proof.expires_at, a.new_plan.expires_at)
    assert a.new_plan.epoch == a.plan.epoch


@pytest.mark.parametrize('phase', ['before_read', 'mailbox'])
def test_strictly_newer_completed_full_is_never_overwritten(crop, monkeypatch, phase):
    a, o = crop, crop.owner
    published = []
    def event():
        o._identity_processing_watermark = (118, NOW-.01)
        new = ValidatedVisualObservation(1, 1, 118, NOW-.01, NOW, NOW+.49, 'full')
        published.append(publish_visual_identity_evidence(o, observation=new, lease=None))
    if phase == 'mailbox': interleave_mailbox(a, monkeypatch, event)
    else:
        original = runtime.read_visual_identity_evidence
        def read(obj):
            if not published: event()
            return original(obj)
        monkeypatch.setattr(runtime, 'read_visual_identity_evidence', read)
    assert deliver(a)
    assert o._visual_identity_evidence is published[0]
    assert o._validated_visual_observation.continuation_sample_timestamp is None


def test_publication_is_locked_and_logging_is_outside_locks(crop, monkeypatch):
    a, o = crop, crop.owner
    original = runtime.publish_visual_identity_evidence
    seen = []
    def publish(obj, **kwargs):
        assert obj._control_update_lock._is_owned() and obj._short_follow._lock._is_owned()
        seen.append(kwargs)
        return original(obj, **kwargs)
    def log(message, *args):
        if message.startswith('short_follow_mapped_crop_pivot_bridge '):
            assert not o._control_update_lock._is_owned() and not o._short_follow._lock._is_owned()
    monkeypatch.setattr(runtime, 'publish_visual_identity_evidence', publish)
    monkeypatch.setattr(runtime.logger, 'info', log)
    assert deliver(a)
    assert len(seen) == 1
