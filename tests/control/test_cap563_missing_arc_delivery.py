"""Empty detector output reaches the real paired consumer and fake motor.

Existing identity/range setup is synthetic; empty frames run the actual
pipeline/tracker (fake inference). No devices or physical trajectory replay.
"""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import DistanceState, SensorFrame
from car_control_modular.detector_identity_lease import (
    read_visual_identity_evidence, publish_visual_identity_evidence,
)
from car_control_modular.short_follow import ShortFollowObservation
from test_cap1542_reacquire_handoff import prepare_consumer
from test_cap609_reacquire_depth_transfer import handoff, capture, queue_capture
from test_short_follow_adapter import paired, owner

sys.path.append(str(Path(__file__).resolve().parents[1] / "vision"))
from test_cap563_empty_detection_contract import pipeline_case


def start(h, s, monkeypatch, mirror=False):
    # Both fixtures replace the same Python time module; use one shared clock
    # for sensing, consumer and motor, without wall-clock waiting.
    monkeypatch.setattr(runtime.time, "monotonic", lambda: h.clock[0])
    if mirror:
        h.a.target = replace(h.a.target, bbox=(90., 40., 250., 440.))
    h.obj.search_state = h.obj._follow_controller.search_state = "none"
    h.obj._short_follow.activate(1, h.clock[0]-.01)
    capture(h, 558)
    assert queue_capture(h)
    h.motor._service_short_follow()
    assert h.driver.pairs and not h.driver.stops
    prepare_consumer(h, {}, monkeypatch)
    h.obj._depth_roi_safety_clear = True
    h.obj._rknn_pipeline = s.pipeline
    return h.obj._short_follow.snapshot()


def empty(h, s, cap, *, advance=.06):
    h.clock[0] += advance
    stamp = h.clock[0] - .025
    records = s.step(cap, stamp)
    assert records == []
    assert s.pipeline.last_identity_processing["full_features_current"] is False
    assert s.pipeline.last_identity_processing["detector_result_complete"] is True
    h.obj.frame_index += 1
    h.obj._active_capture_frame_id = cap
    h.obj._active_capture_timestamp = stamp
    h.obj._update_detector_identity_lease(records, cap, stamp, now=h.clock[0],
        stale=False, expected_epoch=h.obj._depth_async_scheduler.publication_snapshot()[0])
    before_roi = h.obj._depth_async_scheduler.publication_snapshot()
    # Follow the real main-loop ordering; empty results must not republish the
    # old target ROI before the normal consumer runs.
    assert not h.obj._publish_validated_depth_observation(records, 640, 480, cap, stamp,
        expected_epoch=before_roi[0])
    assert h.obj._depth_async_scheduler.publication_snapshot() == before_roi
    h.obj._consume_track_records(records, 640, 480, "empty_integration")
    h.motor._service_short_follow()


@pytest.mark.parametrize("mirror", [False, True])
def test_true_empty_frames_preserve_arc_not_search_or_stop(handoff, pipeline_case, monkeypatch, mirror):
    h, s = handoff, pipeline_case
    original = start(h, s, monkeypatch, mirror)
    plan = original.plan
    proof, _ = read_visual_identity_evidence(h.obj)
    assert plan.left_rpm != plan.right_rpm and min(plan.left_rpm, plan.right_rpm) > 0
    assert (plan.left_rpm > plan.right_rpm) is (not mirror)
    for cap in (560, 563, 564):
        empty(h, s, cap)
        assert h.obj._short_follow.snapshot() is original
        evidence, _ = read_visual_identity_evidence(h.obj)
        assert evidence.observation.capture == proof.observation.capture
        assert evidence.observation.validated_at == proof.observation.validated_at
        assert evidence.observation.expires_at == min(proof.observation.expires_at, plan.expires_at)
        assert evidence.permits_depth(1, plan.depth_timestamp, h.clock[0])
        assert not evidence.permits_depth(1, h.clock[0], h.clock[0])
        assert not h.driver.stops and not h.obj._queued_calls
        assert h.obj._follow_controller.search_state == "none"
    assert all(pair == (plan.left_rpm, -plan.right_rpm) for pair in h.driver.pairs)


def test_new_depth_during_empty_cannot_replace_pair_but_new_full_can(handoff, pipeline_case, monkeypatch):
    h, s = handoff, pipeline_case
    original = start(h, s, monkeypatch)
    empty(h, s, 563)
    h.clock[0] += .01
    # A worker that was already scanning the old ROI finishes a new Depth.
    # Empty detection did not grant it permission to issue a new plan.
    frame = SensorFrame(width=640, height=480, persons=[h.a.target],
        distance_m=2., distance_state=DistanceState(source="vision_depth",
            raw_distance_m=2., used_distance_m=2., sample_timestamp=h.clock[0]-.001),
        capture_frame_id=558, capture_timestamp=original.plan.capture_timestamp)
    assert h.obj._short_follow_adapter.handle(frame, h.a.target, is_fresh_depth=True,
        control_source="depth30", target_steerable=True, low_quality_visible=False, now=h.clock[0])
    assert h.obj._short_follow.snapshot() is original
    h.motor._service_short_follow()
    assert not h.driver.stops
    # A genuinely new accepted full identity + independent range can supersede
    # the narrow bridge immediately, in the same epoch, without parking.
    h.clock[0] += .02
    capture(h, 564)
    publish_visual_identity_evidence(h.obj, observation=h.obj._validated_visual_observation, lease=None)
    assert queue_capture(h)
    new = h.obj._short_follow.snapshot().plan
    assert new.capture_id == 564 and new.depth_timestamp > original.plan.depth_timestamp
    assert new.epoch == original.epoch and new.forwarding
    h.motor._service_short_follow()
    assert not h.driver.stops and not h.obj._queued_calls


def test_missing_frames_do_not_extend_deadline_or_resurrect_expired_arc(handoff, pipeline_case, monkeypatch):
    h, s = handoff, pipeline_case
    original = start(h, s, monkeypatch)
    empty(h, s, 563)
    h.clock[0] = original.plan.expires_at + .001
    h.motor._service_short_follow()
    assert h.driver.stops
    assert h.obj._short_follow.snapshot().plan is original.plan
    before = len(h.driver.pairs)
    h.clock[0] += .02
    records = s.step(565, h.clock[0]-.01)
    h.obj._update_detector_identity_lease(records, 565, h.clock[0]-.01, now=h.clock[0], stale=False)
    h.motor._service_short_follow()
    assert len(h.driver.pairs) == before
    evidence, _ = read_visual_identity_evidence(h.obj)
    assert not evidence.live(1, h.clock[0])


@pytest.mark.parametrize("danger", ["obstacle", "hazard"])
def test_current_consumer_danger_still_interrupts_retained_arc(handoff, pipeline_case, monkeypatch, danger):
    h, s = handoff, pipeline_case
    start(h, s, monkeypatch)
    empty(h, s, 563)
    frame = SensorFrame(width=640, height=480, persons=[],
        capture_frame_id=564, capture_timestamp=h.clock[0]-.01)
    if danger == "obstacle":
        frame = replace(frame, obstacles=replace(frame.obstacles, front=True))
    else:
        frame = replace(frame, hazard=replace(frame.hazard, active=True))
    assert not h.obj._short_follow_adapter.handle(frame, None, is_fresh_depth=False,
        control_source="vision", target_steerable=True, low_quality_visible=False, now=h.clock[0])
    assert not h.obj._short_follow.snapshot().active
    h.motor._service_short_follow()
    assert h.driver.stops


def test_interleaved_valid_depth_commit_reaches_writer_without_zero(handoff, pipeline_case, monkeypatch):
    h, s = handoff, pipeline_case
    original = start(h, s, monkeypatch)
    committed = []
    review = s.pipeline.tracker.associated_position_contradiction
    def interleave(*args):
        verdict = review(*args)
        if not committed:
            new = h.obj._short_follow.update(ShortFollowObservation(
                1, original.plan.capture_id, original.plan.capture_timestamp,
                h.clock[0]-.005, 1.8, .85), h.clock[0])
            assert new is not None and new.forwarding
            committed.append(new)
        return verdict
    monkeypatch.setattr(s.pipeline.tracker, "associated_position_contradiction", interleave)
    empty(h, s, 563, advance=.1)
    latest = committed[0]
    assert h.obj._short_follow.snapshot().plan is latest
    assert latest.epoch == original.epoch
    assert h.driver.pairs[-1] == (latest.left_rpm, -latest.right_rpm)
    assert not h.driver.stops and not h.obj._queued_calls
    evidence, _ = read_visual_identity_evidence(h.obj)
    assert evidence.observation.continuation_sample_timestamp == latest.depth_timestamp
    assert evidence.observation.expires_at <= latest.expires_at
