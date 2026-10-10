"""Position-only adapter diagnostics do not create a new motion rejection.

Tracker integration tests cover filtering the low-score formal record. Here
the real main identity publisher, paired adapter and fake-serial executor run
their unchanged empty-observation contract, including its original deadline.
"""
from types import SimpleNamespace

import pytest
import request_0513_modular as app
from car_control_modular.control_types import SensorFrame
from car_control_modular.detector_identity_lease import publish_visual_identity_evidence
from car_control_modular.short_follow_adapter import ShortFollowAdapter
from test_short_follow_executor import short_runtime


def setup(monkeypatch):
    rt, owner, driver, symbols, clock = short_runtime(monkeypatch)
    owner._short_follow_adapter = ShortFollowAdapter(owner, owner._short_follow, rt.logger)
    publication = publish_visual_identity_evidence(
        owner, observation=owner._validated_visual_observation, lease=None)
    owner._rknn_pipeline = SimpleNamespace(
        last_identity_processing=dict(mode="full", full_features_current=True),
        tracker=SimpleNamespace(last_identity_observations=[]))
    return rt, owner, driver, symbols, clock, publication


def position_frame(owner, clock, cap):
    stamp = clock[0]-.01
    owner._rknn_pipeline.last_identity_processing.update(
        capture_frame_id=cap, capture_timestamp=stamp)
    owner._rknn_pipeline.tracker.last_identity_observations = [dict(
        raw_track_id=1, uid=0, assignment=dict(reason="low_score_observation_only", bank_updated=False),
        sample_metadata=dict(low_score_continuation=True,
            capture_frame_id=cap, capture_timestamp=stamp))]
    # Low-confidence positions remain diagnostics, not formal person records.
    assert app.PersonTracker._update_detector_identity_lease(
        owner, [], cap, stamp, now=clock[0], stale=False)
    frame = SensorFrame(width=640, height=480, capture_frame_id=cap, capture_timestamp=stamp)
    return owner._short_follow_adapter.handle(frame, None, is_fresh_depth=False,
        control_source="vision", target_steerable=True, low_quality_visible=False, now=clock[0])


def test_repeated_low_score_positions_keep_only_original_executed_pair(monkeypatch):
    rt, owner, driver, _, clock, publication = setup(monkeypatch)
    plan = owner._short_follow.snapshot().plan
    rt._service_short_follow()
    for index, offset in enumerate((.05, .10, .15, .20)):
        clock[0] = 10.+offset
        assert position_frame(owner, clock, 1075+index)
        assert owner._visual_identity_evidence is publication
        assert owner._short_follow.snapshot().plan is plan
        rt._service_short_follow()
    assert len(driver.pairs) == 5
    assert len(set(driver.pairs)) == 1
    assert not driver.stops
    assert plan.expires_at == pytest.approx(10.3)
    clock[0] = 10.301
    rt._service_short_follow()
    assert driver.stops  # continuous weak positions do not renew depth/identity


def test_position_only_frame_cannot_conceal_hazard_detection(monkeypatch):
    rt, owner, driver, _, clock, _ = setup(monkeypatch)
    rt._service_short_follow()
    clock[0] += .05
    owner._bunker_runtime = SimpleNamespace(config=SimpleNamespace(
        enabled=True, mode="merged", class_ids={9}))
    owner._rknn_pipeline.last_identity_processing.update(
        capture_frame_id=1078, capture_timestamp=clock[0]-.01)
    # Non-person hazard records are retained by the track adapter, even when
    # an adjacent person's low-score position is diagnostics-only.
    hazard = SimpleNamespace(class_id=9, reid_uid=0, time_since_update=0)
    assert app.PersonTracker._update_detector_identity_lease(
        owner, [hazard], 1078, clock[0]-.01, now=clock[0], stale=False)
    assert owner._validated_visual_observation is False
    rt.hard_stop_check = lambda _: True
    rt._service_short_follow()
    assert driver.stops
