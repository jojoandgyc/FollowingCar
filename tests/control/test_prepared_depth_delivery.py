"""A committed bounded sample is not a new pixel/ROI admission on delivery."""
from dataclasses import replace

import numpy as np
import pytest

import request_0513_modular as main
from car_control_modular.astra_depth import AstraDepthRuntime
from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowObservation,
)
from test_depth_optimistic_transaction import scene, prepare
from test_turn_depth_scheduling import owner, context, attempt, NOW


def bounded_scene(scene):
    obj, camera, distance, clock = scene
    distance.config = replace(distance.config, vision_depth_detector_bbox_max_age_sec=.5)
    obj._longitudinal_context = context(227, NOW - .261)
    camera._depth_history.append((NOW - .110, camera._latest_depth))
    camera._latest_depth = np.full((480, 640), 4500, dtype=np.uint16)
    camera._latest_depth_ts = NOW
    return obj, camera, distance, clock


def commit_history(scene):
    obj, camera, distance, clock = bounded_scene(scene)
    item, target = prepare(scene)
    assert item.geometry_reason == "yolo_detector_bounded_history"
    assert item.transaction.run().raw_distance_m == pytest.approx(1.8)
    clock[0] = NOW + .05  # The 160 ms old physical sample is still admissible.
    assert distance.commit_prepared_depth(item, target=target)
    return item, target


def consume(distance, item, target):
    return distance.get_frame_distance_state(
        640, target, frame_height=480, depth_use_latest=True, prepared_depth=item)


def test_cap227_committed_sample_survives_control_delivery_crossing_180ms(scene):
    obj, camera, distance, clock = scene
    item, target = commit_history(scene)
    receipt = item.transaction.commit_receipt
    clock[0] += .040
    state = consume(distance, item, target)
    assert state.raw_distance_m == pytest.approx(1.8)
    assert state.sample_timestamp == pytest.approx(NOW - .110)
    assert state.sample_age_sec == pytest.approx(.200)
    assert camera._last_accepted_ts == pytest.approx(NOW - .110)
    assert receipt.validated_at == pytest.approx(NOW + .050)
    assert item.transaction.commit_receipt is receipt
    assert not state.source_detail.startswith("prepared_bounded_provenance_invalid")

    # A successful delivery is not another capture or a new 300 ms motor lease.
    ctl = ShortFollowController(ShortFollowConfig(enabled=True))
    ctl.activate(1, clock[0])
    plan = ctl.update(ShortFollowObservation(
        uid=1, capture_id=227, capture_timestamp=NOW-.261,
        depth_timestamp=state.sample_timestamp, distance_m=state.used_distance_m,
        center_x_ratio=.5, raw_distance_m=state.raw_distance_m), clock[0])
    assert plan is not None
    assert plan.expires_at == pytest.approx(NOW - .110 + .300)
    assert not plan.valid(NOW - .110 + .301)


def test_real_main_delivery_delay_does_not_discard_already_committed_depth(scene, monkeypatch):
    obj, camera, distance, clock = bounded_scene(scene)
    monkeypatch.setattr(main, "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", .5)
    scan = AstraDepthRuntime._select_multiregion_distance
    def compute(private, *args, **kwargs):
        result = scan(private, *args, **kwargs)
        clock[0] += .050
        return result
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", compute)
    states = []
    def delivered(*args, **kwargs):
        item = kwargs["prepared_depth"]
        assert item.transaction.committed
        clock[0] += .040  # E.g. current IR/status reads before distance fusion.
        states.append(consume(distance, item, item.target))
    obj._queue_actions_for_persons_locked = delivered
    assert attempt(obj)
    assert len(states) == 1
    assert states[0].raw_distance_m == pytest.approx(1.8)
    assert states[0].sample_age_sec == pytest.approx(.200)


@pytest.mark.parametrize("age", [.251, .301, 1.])
def test_receipt_does_not_extend_consumer_physical_sample_freshness(scene, age):
    obj, camera, distance, clock = scene
    item, target = commit_history(scene)
    clock[0] = NOW - .110 + age
    state = consume(distance, item, target)
    assert state.raw_distance_m is None
    assert state.sample_timestamp == pytest.approx(NOW - .110)
    assert state.source_detail.startswith("stale_depth_frame")


def test_sample_older_than_180ms_still_cannot_commit(scene):
    obj, camera, distance, clock = bounded_scene(scene)
    item, target = prepare(scene)
    assert item.transaction.run() is not None
    clock[0] += .071
    assert not distance.commit_prepared_depth(item, target=target)
    assert item.transaction.commit_receipt is None
    assert camera._last_accepted_ts == 0


@pytest.mark.parametrize("change", ["no_receipt", "measurement", "result", "receipt_result",
                                   "not_committed", "uid", "bbox", "clock_backwards"])
def test_committed_provenance_cannot_be_reused_for_changed_result_or_target(scene, change):
    obj, camera, distance, clock = scene
    item, target = commit_history(scene)
    clock[0] += .040
    if change == "no_receipt": item.transaction.commit_receipt = None
    elif change == "measurement": item.measurement = replace(item.measurement, raw_distance_m=5.)
    elif change == "result": item.transaction.result = replace(item.transaction.result, raw_distance_m=5.)
    elif change == "receipt_result":
        item.transaction.commit_receipt = replace(item.transaction.commit_receipt,
                                                  measurement=replace(item.measurement))
    elif change == "not_committed": item.transaction.committed = False
    elif change == "uid": target = replace(target, track_id=2)
    elif change == "bbox": target = replace(target, bbox=(1., 2., 3., 4.))
    elif change == "clock_backwards": clock[0] = NOW + .040
    state = consume(distance, item, target)
    assert state.raw_distance_m is None


def test_new_same_uid_frame_does_not_retag_committed_sample(scene):
    obj, camera, distance, clock = scene
    item, target = commit_history(scene)
    clock[0] += .040
    obj._longitudinal_context = context(231, clock[0] - .02)
    state = consume(distance, item, target)
    assert state.raw_distance_m == pytest.approx(1.8)
    assert item.target.depth_observation.capture_frame_id == 227
    assert state.sample_timestamp == pytest.approx(NOW - .110)


def test_duplicate_delivery_does_not_refresh_sample_or_reaccept_fusion(scene):
    obj, camera, distance, clock = scene
    item, target = commit_history(scene)
    first = consume(distance, item, target)
    clock[0] += .040
    again = consume(distance, item, target)
    assert first.raw_distance_m == pytest.approx(1.8)
    assert again.raw_distance_m is None
    assert distance._vision_depth_fusion._last_accepted_depth_sample_ts == NOW - .110
