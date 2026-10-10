"""Bounded crop bank permission reaches real identity/ROI/paired consumers.

Recorded geometry/clocks and synthetic local descriptors come from the bank
regression. Depth images and serial output are in memory, not hardware replay.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import DepthTargetObservation
from car_control_modular.depth_target_geometry import resolve_depth_target_observation
from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from test_cap1542_reacquire_handoff import prepare_consumer
from test_cap609_reacquire_depth_transfer import handoff
from test_search_observation_arbitration import _record
from test_short_follow_adapter import paired, owner


@pytest.fixture
def crop_replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'vision'))
    from test_cap1428_1499_bounded_crop import before_crop, send, sample, ROWS
    from tools.replay_cap334_recovery import gallery_snapshot
    return SimpleNamespace(before=before_crop, send=send, sample=sample,
                           rows=ROWS, gallery=gallery_snapshot)


def prepare_frame(h, bank, cap, row, assignment, monkeypatch, *, uid=1):
    obj = h.obj
    stamp, box, raw = row['capture_timestamp'], row['detector_bbox'], row['track_id']
    h.clock[0] = stamp + .04
    obj.frame_index = row['frame_index']
    obj._active_capture_frame_id, obj._active_capture_timestamp = cap, stamp
    h.a.target = replace(h.a.target, bbox=box, depth_observation=DepthTargetObservation(
        bbox=box, target_id=1, raw_track_id=raw, capture_frame_id=cap, capture_timestamp=stamp))
    h.camera._latest_depth_ts = stamp
    h.camera._depth_history.append((stamp, h.camera._latest_depth))
    prepare_consumer(h, assignment, monkeypatch)
    obj._identity_assignment_debug_for_track = lambda candidate: assignment if candidate == raw else {}
    obj._bunker_runtime.config = SimpleNamespace(enabled=False)
    obj._rknn_pipeline = SimpleNamespace(last_frame_width=640, last_frame_height=480,
        last_identity_processing=dict(mode='full', full_features_current=True,
            capture_frame_id=cap, capture_timestamp=stamp),
        tracker=SimpleNamespace(identity_bank=bank, last_identity_observations=[dict(
            raw_track_id=raw, uid=uid, display_bbox=box, detector_bbox=box,
            sample_metadata=row, assignment=assignment)],
            associated_position_contradiction=lambda *_: None))
    # Undo the ranging fixture's geometry/target substitutions: this test must
    # pass the real weak metadata through both geometry and target conversion.
    monkeypatch.setattr(runtime, 'resolve_depth_target_observation', resolve_depth_target_observation)
    obj._persons_to_targets = runtime.PersonTracker._persons_to_targets.__get__(obj)
    rec = _record(track=raw, uid=uid, bbox=box, score=row['detector_confidence'])
    assert obj._update_detector_identity_lease([rec], cap, stamp, now=h.clock[0], stale=False)
    return rec


def established_follow(h, bank, cap, replay, monkeypatch, *, mirror=False):
    obj = h.obj
    obj.search_state = obj._follow_controller.search_state = 'none'
    obj._follow_controller._has_seen_person = True
    obj._reacquire_depth_pending = False
    h.camera._latest_depth[:] = 2500
    prior_cap = 1425 if cap == 1428 else 1494
    row = replay.sample(prior_cap, mirror=mirror)
    h.clock[0] = row['capture_timestamp']
    obj._short_follow.activate(1, h.clock[0]-.01)
    rec = prepare_frame(h, bank, prior_cap, row,
                        deepcopy(bank.last_assignments[row['track_id']]), monkeypatch)
    obj._consume_track_records([rec], 640, 480, 'test')
    before = obj._short_follow.snapshot().plan
    assert before is not None and before.capture_id == prior_cap and before.forwarding
    h.motor._service_short_follow()
    assert len(h.driver.pairs) == 1 and not h.driver.stops
    scheduler = obj._depth_async_scheduler
    obj._longitudinal_thread = SimpleNamespace(is_alive=lambda: True)
    obj._action_runtime_started = True
    obj._depth_roi_safety_clear = True
    obj._vision_control_state = 'target_visible_depth_valid'
    obj._follow_controller.last_selected_target = h.a.target
    scheduler.worker_tick(now=h.clock[0])
    scheduler.submit(dict(published_ts=h.clock[0], frame_index=row['frame_index'],
        width=640, height=480, person_targets=(h.a.target,),
        capture_frame_id=prior_cap, capture_timestamp=row['capture_timestamp'],
        target_id=1, target_steerable=True), now=h.clock[0])
    return before


@pytest.mark.parametrize('cap', [1428, 1499])
@pytest.mark.parametrize('mirror', [False, True])
def test_actual_bounded_crop_permission_delivers_forward_yaw_without_weak_rejection(
        handoff, crop_replay, monkeypatch, cap, mirror):
    replay, h = crop_replay, handoff
    bank = replay.before(cap, mirror=mirror)
    before = established_follow(h, bank, cap, replay, monkeypatch, mirror=mirror)
    gallery, references = replay.gallery(bank), deepcopy(bank._follow_references)
    protected = deepcopy(bank.identities[1].last_strong_observation)
    assert replay.send(bank, cap, mirror=mirror) == 1
    row = replay.sample(cap, mirror=mirror)
    assignment = deepcopy(bank.last_assignments[row['track_id']])
    assert row['quality_bbox_ok'] is False and row['bbox_quality_tier'] == 'weak'
    assert assignment['reason'] == 'mapped_similar_follow'
    assert assignment['similar_follow']['completed_confirmation']
    assert assignment['similar_follow']['bounded_crop_continuation']
    assert assignment['bbox_quality_ok'] is True  # actual bank control permission
    rec = prepare_frame(h, bank, cap, row, assignment, monkeypatch)
    proof = h.obj._validated_visual_observation
    assert isinstance(proof, ValidatedVisualObservation)
    assert (proof.capture, proof.track_id, proof.timestamp) == (cap, row['track_id'], row['capture_timestamp'])
    assert proof.kind == 'full' and proof.continuation_sample_timestamp is None
    assert h.obj._completed_similar_follow_confirmation(
        assignment, uid=1, track_id=row['track_id']) is proof
    scheduler = h.obj._depth_async_scheduler
    scheduler.worker_tick(now=h.clock[0])
    epoch = scheduler.publication_snapshot()[0]
    assert h.obj._publish_validated_depth_observation([rec], 640, 480,
        cap, row['capture_timestamp'], expected_epoch=epoch)
    roi = scheduler.publication_snapshot()[1].person_targets[0].depth_observation
    assert roi.source == 'yolo_detector' and roi.bbox == tuple(row['detector_bbox'])
    assert (roi.capture_frame_id, roi.raw_track_id) == (cap, row['track_id'])
    # Early ROI publication may change yaw only; no new range/deadline yet.
    yaw_only = h.obj._short_follow.snapshot().plan
    assert yaw_only.depth_timestamp == before.depth_timestamp
    assert yaw_only.expires_at == before.expires_at
    h.obj._consume_track_records([rec], 640, 480, 'test')
    plan = h.obj._short_follow.snapshot().plan
    assert plan is not None and plan.capture_id == cap and plan.forwarding
    assert plan.epoch == before.epoch and min(plan.left_rpm, plan.right_rpm) > 0
    assert (plan.left_rpm > plan.right_rpm) is (not mirror)
    assert plan.depth_timestamp == row['capture_timestamp']
    assert plan.expires_at == pytest.approx(min(plan.depth_timestamp+.3, plan.capture_timestamp+.5))
    h.motor._service_short_follow()
    assert len(h.driver.pairs) == 2 and not h.driver.stops and not h.obj._queued_calls
    assert h.driver.pairs[-1] == (plan.left_rpm, -plan.right_rpm)
    assert all(left > 0 > right for left, right in h.driver.pairs)
    assert not h.obj._reacquire_depth_pending and h.obj.search_state == 'none'
    assert h.obj._visible_unsteerable_uid is None
    assert replay.gallery(bank) == gallery and bank._follow_references == references
    assert bank.identities[1].last_strong_observation == protected
    assert not assignment['bank_updated'] and not assignment['recent_bank_updated']
    assert not assignment['learning_written_tiers']


@pytest.mark.parametrize('cap', [1428, 1499])
@pytest.mark.parametrize('fault', ['competition', 'geometry', 'wrong_local_full', 'wrong_local_torso'])
def test_actual_conflicting_crop_uid0_cannot_keep_previous_pair(
        handoff, crop_replay, monkeypatch, cap, fault):
    replay, h = crop_replay, handoff
    bank = replay.before(cap)
    before = established_follow(h, bank, cap, replay, monkeypatch)
    gallery = replay.gallery(bank)
    row = replay.sample(cap)
    changes = {}
    if fault == 'competition':
        changes['identity_competition'] = dict(uid=1, frame_index=row['frame_index'],
            source_detection_index=0, candidate_count=2, passed=False)
    elif fault == 'geometry':
        bank._mapped_geometry_conflicts[row['track_id']] = dict(uid=1, search_contradiction=True)
    else:
        from test_cap874_identity_reacquire import feature
        channel = 'full' if fault == 'wrong_local_full' else 'partial'
        vector = feature(replay.rows[cap][4 if channel == 'full' else 5]).copy()
        vector[1] *= -1
        changes[channel+'_vector'] = vector
    assert replay.send(bank, cap, **changes) == 0
    assignment = deepcopy(bank.last_assignments[row['track_id']])
    assert not assignment.get('similar_follow', {}).get('bounded_crop_continuation')
    if 'identity_competition' in changes:
        row['identity_competition'] = changes['identity_competition']
    rec = prepare_frame(h, bank, cap, row, assignment, monkeypatch, uid=0)
    assert before.valid(h.clock[0])  # rejection, not an already expired pair
    assert h.obj._validated_visual_observation is False
    epoch = h.obj._depth_async_scheduler.publication_snapshot()[0]
    assert not h.obj._publish_validated_depth_observation([rec], 640, 480,
        cap, row['capture_timestamp'], expected_epoch=epoch)
    assert h.obj._completed_similar_follow_confirmation(
        assignment, uid=1, track_id=row['track_id']) is None
    h.motor._service_short_follow()
    assert len(h.driver.pairs) == 1 and h.driver.stops
    assert replay.gallery(bank) == gallery and not assignment['bank_updated']
