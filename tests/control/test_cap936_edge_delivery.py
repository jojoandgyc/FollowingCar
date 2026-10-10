"""Recorded edge identity policy reaches the paired writer without a zero.

Bank geometry/scores come from CAP932/936; descriptors and depth/serial are
in memory. This is control integration, not a physical trajectory replay.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import DepthTargetObservation
from test_cap1542_reacquire_handoff import prepare_consumer
from test_cap609_reacquire_depth_transfer import handoff
from test_search_observation_arbitration import _record
from test_short_follow_adapter import paired, owner


def deliver(h, bank, cap, row, assignment, monkeypatch):
    stamp, box = row['capture_timestamp'], row['detector_bbox']
    h.clock[0] = stamp + .04
    obj = h.obj
    obj.frame_index = row['frame_index']
    obj._active_capture_frame_id = cap
    obj._active_capture_timestamp = stamp
    h.a.target = replace(h.a.target, bbox=box, depth_observation=DepthTargetObservation(
        bbox=box, target_id=1, raw_track_id=11, capture_frame_id=cap, capture_timestamp=stamp))
    h.camera._latest_depth_ts = stamp
    h.camera._depth_history.append((stamp, h.camera._latest_depth))
    prepare_consumer(h, assignment, monkeypatch)
    obj._identity_assignment_debug_for_track = lambda raw: assignment if raw == 11 else {}
    obj._bunker_runtime.config = SimpleNamespace(enabled=False)
    obj._rknn_pipeline = SimpleNamespace(last_frame_width=640, last_frame_height=480,
        last_identity_processing=dict(mode='full', full_features_current=True,
            capture_frame_id=cap, capture_timestamp=stamp),
        tracker=SimpleNamespace(last_identity_observations=[dict(raw_track_id=11,
            uid=1, detector_bbox=box, sample_metadata=row, assignment=assignment)],
            associated_position_contradiction=lambda *_: None))
    record = _record(track=11, uid=1, bbox=box, score=row['detector_confidence'])
    assert obj._update_detector_identity_lease([record], cap, stamp,
                                              now=h.clock[0], stale=False)
    obj._consume_track_records([record], 640, 480, 'test')
    plan = obj._short_follow.snapshot().plan
    assert plan is not None and plan.capture_id == cap and plan.forwarding
    h.motor._service_short_follow()
    return plan


@pytest.mark.parametrize('mirror', [False, True])
def test_same_person_narrow_edge_keeps_forward_and_yaw_without_new_parking(
        handoff, monkeypatch, mirror):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'vision'))
    from test_cap936_narrow_edge_follow import before936, send, sample
    from tools.replay_cap334_recovery import gallery_snapshot
    bank = before936(mirror=mirror)
    gallery = gallery_snapshot(bank)
    h = handoff
    h.obj.search_state = h.obj._follow_controller.search_state = 'none'
    h.obj._reacquire_depth_pending = False
    first = sample(932, mirror=mirror)
    h.clock[0] = first['capture_timestamp']
    h.obj._short_follow.activate(1, h.clock[0]-.01)
    p = deliver(h, bank, 932, first, deepcopy(bank.last_assignments[11]), monkeypatch)
    assert h.driver.pairs
    assert send(bank, mirror=mirror) == 1
    second = sample(936, mirror=mirror)
    assert second['quality_bbox_ok'] is False
    q = deliver(h, bank, 936, second, deepcopy(bank.last_assignments[11]), monkeypatch)
    assert q.epoch == p.epoch
    assert q.left_rpm != q.right_rpm and min(q.left_rpm, q.right_rpm) > 0
    assert (q.left_rpm > q.right_rpm) is (not mirror)
    assert len(h.driver.pairs) == 2
    assert not h.driver.stops and not h.obj._queued_calls
    assert not h.obj._reacquire_depth_pending and h.obj.search_state == 'none'
    assert gallery_snapshot(bank) == gallery
    assert q.expires_at == pytest.approx(min(q.depth_timestamp+.3,
                                           q.capture_timestamp+.5))
