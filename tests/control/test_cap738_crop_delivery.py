"""CAP738's actual crop permission through identity, ranging and paired writes.

Recorded geometry and scalar appearance relationships; synthetic descriptors,
depth images and serial driver. No hardware/physical trajectory replay.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from test_cap1428_crop_delivery import prepare_frame
from test_cap609_reacquire_depth_transfer import handoff
from test_short_follow_adapter import paired, owner


@pytest.fixture
def edge_replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'vision'))
    import test_cap738_edge_visibility as replay
    return replay


def begin(h, replay, monkeypatch, mirror):
    bank = replay.before_738(mirror=mirror)
    h.obj.search_state = h.obj._follow_controller.search_state = 'none'
    h.obj._follow_controller._has_seen_person = True
    h.obj._reacquire_depth_pending = False
    h.camera._latest_depth[:] = 2020
    row = replay.sample(734, mirror=mirror)
    h.clock[0] = row['capture_timestamp']
    h.obj._short_follow.activate(1, h.clock[0]-.01)
    rec = prepare_frame(h, bank, 734, row, deepcopy(bank.last_assignments[10]), monkeypatch)
    h.obj._consume_track_records([rec], 640, 480, 'test')
    before = h.obj._short_follow.snapshot().plan
    assert before and before.forwarding
    h.motor._service_short_follow()
    assert h.driver.pairs and not h.driver.stops
    o = h.obj
    o._longitudinal_thread = SimpleNamespace(is_alive=lambda: True)
    o._action_runtime_started = True
    o._depth_roi_safety_clear = True
    o._vision_control_state = 'target_visible_depth_valid'
    o._follow_controller.last_selected_target = h.a.target
    o._depth_async_scheduler.worker_tick(now=h.clock[0])
    o._depth_async_scheduler.submit(dict(published_ts=h.clock[0], frame_index=row['frame_index'],
        width=640, height=480, person_targets=(h.a.target,), capture_frame_id=734,
        capture_timestamp=row['capture_timestamp'], target_id=1, target_steerable=True), now=h.clock[0])
    return bank, before


@pytest.mark.parametrize('mirror', [False, True])
def test_recorded_734_738_743_continue_arc_without_identity_stop_or_search(
        handoff, edge_replay, monkeypatch, mirror):
    h, r = handoff, edge_replay
    bank, before = begin(h, r, monkeypatch, mirror)
    gallery = r.gallery_snapshot(bank)
    for cap in (738, 743):
        assert r.send(bank, cap, mirror=mirror) == 1
        assignment = deepcopy(bank.last_assignments[10])
        row = r.sample(cap, mirror=mirror)
        assert row['quality_bbox_ok'] is False  # do not sanitize the raw weak crop
        assert assignment['bbox_quality_ok'] is True  # actual bank continuation
        assert not assignment['bank_updated'] and not assignment['learning_written_tiers']
        rec = prepare_frame(h, bank, cap, row, assignment, monkeypatch)
        proof = h.obj._validated_visual_observation
        assert isinstance(proof, ValidatedVisualObservation)
        assert proof.capture == cap and proof.kind == 'full'
        assert proof.continuation_sample_timestamp is None
        assert h.obj._completed_similar_follow_confirmation(
            assignment, uid=1, track_id=10) is proof
        original_plan = h.obj._short_follow.snapshot().plan
        # Writer may run immediately after identity publication, before the
        # range mailbox or visual consumer: no transient False proof/STOP.
        h.motor._service_short_follow()
        assert not h.driver.stops
        scheduler = h.obj._depth_async_scheduler
        scheduler.worker_tick(now=h.clock[0])
        assert h.obj._publish_validated_depth_observation([rec], 640, 480,
            cap, row['capture_timestamp'], expected_epoch=scheduler.publication_snapshot()[0])
        roi = scheduler.publication_snapshot()[1].person_targets[0].depth_observation
        assert roi.bbox == tuple(row['detector_bbox']) and roi.raw_track_id == 10
        yaw_plan = h.obj._short_follow.snapshot().plan
        assert yaw_plan.depth_timestamp == original_plan.depth_timestamp
        assert yaw_plan.expires_at == original_plan.expires_at
        h.motor._service_short_follow()
        assert not h.driver.stops
        # The unchanged real geometry gate still owns whether the new crop
        # may produce a range; accepting identity cannot manufacture depth.
        h.obj._consume_track_records([rec], 640, 480, 'test')
        plan = h.obj._short_follow.snapshot().plan
        assert plan and plan.capture_id == cap and plan.forwarding
        assert plan.epoch == before.epoch
        assert plan.depth_timestamp == row['capture_timestamp']
        assert plan.expires_at == pytest.approx(min(plan.depth_timestamp+.3,
                                                   plan.capture_timestamp+.5))
        assert (plan.left_rpm > plan.right_rpm) is (not mirror)
        assert min(plan.left_rpm, plan.right_rpm) > 0
        h.motor._service_short_follow()
        assert h.driver.pairs[-1] == (plan.left_rpm, -plan.right_rpm)
        assert not h.driver.stops and not h.obj._queued_calls
        assert h.obj.search_state == h.obj._follow_controller.search_state == 'none'
        assert not h.obj._reacquire_depth_pending
        assert r.gallery_snapshot(bank) == gallery
    assert len(h.driver.pairs) >= 3
    assert all(left > 0 > right for left, right in h.driver.pairs)


@pytest.mark.parametrize('fault', ['competition', 'geometry', 'different_person'])
def test_true_conflict_still_retires_previous_pair(handoff, edge_replay, monkeypatch, fault):
    h, r = handoff, edge_replay
    bank, before = begin(h, r, monkeypatch, False)
    changes = {}
    if fault == 'competition':
        changes['identity_competition'] = dict(uid=1, frame_index=338,
            source_detection_index=0, candidate_count=2, passed=False)
    elif fault == 'geometry':
        bank._mapped_geometry_conflicts[10] = dict(uid=1, search_contradiction=True)
    else:
        vector = r.feature(r.ROWS[738][4])
        vector[1] *= -1
        changes['full_vector'] = vector
    assert r.send(bank, 738, **changes) == 0
    assignment = deepcopy(bank.last_assignments[10])
    row = r.sample(738)
    if 'identity_competition' in changes:
        row['identity_competition'] = changes['identity_competition']
    prepare_frame(h, bank, 738, row, assignment, monkeypatch, uid=0)
    assert before.valid(h.clock[0])
    assert h.obj._validated_visual_observation is False
    h.motor._service_short_follow()
    assert h.driver.stops and len(h.driver.pairs) == 1
