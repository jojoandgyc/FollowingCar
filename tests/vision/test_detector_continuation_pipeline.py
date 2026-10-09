"""Actual pipeline/tracker, synthetic detector/embeddings; no models or devices."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rk_vision.pipeline import RKNNVisionPipeline, RKNNVisionConfig, _detector_color_features
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection
from car_control_modular.depth_target_geometry import resolve_depth_target_observation


BOX = (230., 70., 350., 400.)


@pytest.fixture
def fast_pipeline(monkeypatch):
    clock = SimpleNamespace(now=10.)
    monkeypatch.setattr("rk_vision.pipeline.time.monotonic", lambda: clock.now)
    pipeline = RKNNVisionPipeline.__new__(RKNNVisionPipeline)
    pipeline.config = RKNNVisionConfig(yolo_model_path="unused", detector_continuation_enable=True)
    pipeline.logger = None
    pipeline._reid_diagnostics = None
    pipeline._frame_context = {}
    pipeline.tracker = DeepSortTracker(DeepSortTrackerConfig(
        n_init=1, identity_new_confirm_frames=1, identity_update_interval=1))
    detector = SimpleNamespace(last_timing_ms={}, last_search_diagnostic_detections=[],
                               detections=[Detection(BOX, .95, 0)], calls=0)
    def detect(*_):
        detector.calls += 1
        return list(detector.detections)
    detector.detect = detect
    pipeline.detector = detector
    extractor = SimpleNamespace(calls=0, last_timing_ms={}, last_partial_features=[],
                                last_partial_feature_sources=[], last_color_features=[])
    def extract(packet, persons, fmt):
        extractor.calls += 1
        extractor.last_partial_features = [np.array([1., 0., 0.]) for _ in persons]
        extractor.last_partial_feature_sources = ["osnet_torso" for _ in persons]
        extractor.last_color_features = _detector_color_features(packet, persons, fmt)
        extractor.last_timing_ms = {"total": 25., "inference": 15., "features": len(persons)}
        return [np.array([1., 0., 0.]) for _ in persons]
    extractor.extract = extract
    pipeline.reid = extractor
    pipeline.set_detector_continuation_context(active_uid=1, allowed=True)
    frame = np.full((480, 640, 3), (20, 30, 160), dtype=np.uint8)
    def step(cap, *, fmt="BGR", image=None, timestamp=None, result_age=.015):
        stamp = 10.+cap*.05 if timestamp is None else timestamp
        clock.now = stamp+result_age
        pipeline.set_frame_context(control_frame_id=cap, capture_frame_id=cap,
                                   capture_timestamp=stamp, integrated_yaw_deg=0., yaw_rate_dps=0.)
        return pipeline.process_frame(frame if image is None else image, fmt)
    return SimpleNamespace(pipeline=pipeline, step=step, frame=frame, clock=clock)


def seed(case):
    for cap in range(1, 5):
        case.step(cap)
    assert case.pipeline.tracker.last_detector_continuation_reason == "full_verified"


def test_stable_frames_skip_actual_extractor_but_do_not_skip_new_detection(fast_pipeline):
    s = fast_pipeline; p = s.pipeline
    seed(s)
    before_bank_frame = p.tracker.identity_bank.track_last_seen_frame.copy()
    observations = []
    for cap in (5, 6):
        records = s.step(cap)
        assert p.last_identity_processing["mode"] == "detector_continuation"
        assert p.last_timing_ms["reid_total"] == 0.
        assert records[0].reid_uid == 1 and records[0].time_since_update == 0
        observations.append(p.tracker.last_identity_observations[0])
    assert p.detector.calls == 6 and p.reid.calls == 4
    assert p.tracker.identity_bank.track_last_seen_frame == before_bank_frame
    assert observations[0]["assignment"]["identity_valid_until"] == observations[1]["assignment"]["identity_valid_until"]
    s.step(7)
    assert p.last_identity_processing["mode"] == "full"
    assert p.reid.calls == 5
    assert p.tracker._frame_index == 7  # exactly one update per physical frame


def test_cap49_60_capture_cadence_preserves_full_verification_progress(fast_pipeline):
    s = fast_pipeline; p = s.pipeline
    seed(s)
    # Translate the recorded CAP45/49/53/57/60 capture clock into this fixture.
    # CAP49 still exceeds the gap limit; the following 195/202 ms gaps must
    # retain full-check progress, so CAP60 can consume the completed proof.
    base = 10.45
    s.step(45, timestamp=base, result_age=.10)
    assert p.tracker._detector_proof.full_count == 1
    before_reid = p.reid.calls
    cases = (
        (49, .238173962, "full", "detection_gap", 1),
        (53, .433089596, "full", "full_verification_streak", 2),
        (57, .634829184, "full", "full_recheck_due", 2),
        (60, .801137184, "detector_continuation", "accepted", 2),
    )
    for cap, elapsed, mode, reason, full_count in cases:
        records = s.step(cap, timestamp=base+elapsed, result_age=.10)
        assert records[0].reid_uid == 1 and records[0].time_since_update == 0
        assert p.last_identity_processing["mode"] == mode
        assert p.last_identity_processing["reason"] == reason
        assert p.tracker._detector_proof.full_count == full_count
    assert p.detector.calls == 9
    assert p.reid.calls == before_reid+3
    assert p.tracker._detector_proof.verified.capture == 57
    assert p.tracker._detector_proof.previous.capture == 60
    assert p.tracker._detector_proof.deadline == pytest.approx(base+.634829184+.60)


@pytest.mark.parametrize("gap", [.181, .194916, .199])
def test_sub_200ms_capture_gap_skips_reid_with_fresh_detection(fast_pipeline, gap):
    s = fast_pipeline; p = s.pipeline
    seed(s)
    deadline = p.tracker._detector_proof.deadline
    records = s.step(5, timestamp=10.2+gap, result_age=.10)
    assert records[0].reid_uid == 1
    assert p.last_identity_processing["mode"] == "detector_continuation"
    assert p.detector.calls == 5 and p.reid.calls == 4
    assert p.last_timing_ms["reid_total"] == 0.
    assert p.tracker._detector_proof.verified.capture == 4
    assert p.tracker._detector_proof.deadline == deadline


@pytest.mark.parametrize("gap", [.202, .205, .210])
def test_200_to_210ms_capture_gap_runs_due_full_without_resetting_streak(fast_pipeline, gap):
    s = fast_pipeline; p = s.pipeline
    seed(s)
    records = s.step(5, timestamp=10.2+gap, result_age=.10)
    assert records[0].reid_uid == 1
    assert p.last_identity_processing["mode"] == "full"
    assert p.last_identity_processing["reason"] == "full_recheck_due"
    assert p.reid.calls == p.detector.calls == 5
    assert p.tracker._detector_proof.full_count == 2
    assert p.tracker._detector_proof.verified.capture == 5


def test_gap_above_210ms_runs_full_and_restarts_streak(fast_pipeline):
    s = fast_pipeline; p = s.pipeline
    seed(s)
    records = s.step(5, timestamp=10.411, result_age=.10)
    assert records[0].reid_uid == 1
    assert p.last_identity_processing["mode"] == "full"
    assert p.last_identity_processing["reason"] == "detection_gap"
    assert p.reid.calls == p.detector.calls == 5
    assert p.tracker._detector_proof.full_count == 1


@pytest.mark.parametrize("result_age", [.18, .20])
def test_wider_capture_gap_does_not_admit_old_detector_results(fast_pipeline, result_age):
    s = fast_pipeline; p = s.pipeline
    seed(s)
    records = s.step(5, timestamp=10.395, result_age=result_age)
    assert records[0].reid_uid == 1
    assert p.last_identity_processing["mode"] == "full"
    assert p.last_identity_processing["reason"] == "detection_stale"
    assert p.reid.calls == p.detector.calls == 5
    assert p.tracker._detector_proof.full_count == 1


def test_fast_raw_geometry_can_feed_fresh_depth_without_old_embedding(fast_pipeline):
    s = fast_pipeline; seed(s)
    records = s.step(5); rec = records[0]
    obs = s.pipeline.tracker.last_identity_observations
    geometry = resolve_depth_target_observation(
        target_id=1, display_bbox=(rec.x1, rec.y1, rec.x2, rec.y2),
        capture_frame_id=5, capture_timestamp=10.25, observations=obs,
        width=640, height=480, expected_raw_track_id=rec.track_id)
    assert geometry is not None
    assert obs[0]["sample_metadata"]["capture_frame_id"] == 5
    assert s.pipeline.tracker.deepsort.tracker.tracks[0].last_feature is None


@pytest.mark.parametrize("fault", ["multiple", "small_other", "jump", "missing", "color", "context"])
def test_trigger_forces_full_verification_of_same_frame(fast_pipeline, fault):
    s = fast_pipeline; p = s.pipeline; seed(s)
    frame = s.frame
    if fault == "multiple": p.detector.detections.append(Detection((20., 70., 140., 400.), .9, 0))
    if fault == "small_other":
        p.config = replace(p.config, identity_min_width_px=60.)
        p.detector.detections.append(Detection((20., 70., 40., 100.), .4, 0))
    if fault == "jump": p.detector.detections = [Detection((470., 70., 590., 400.), .95, 0)]
    if fault == "missing": p.detector.detections = []
    if fault == "color": frame = np.full_like(frame, (160, 200, 20))
    if fault == "context": p.set_detector_continuation_context(active_uid=1, allowed=False)
    s.step(5, image=frame)
    assert p.last_identity_processing["mode"] == "full"
    assert p.reid.calls == 5 and p.detector.calls == 5
    assert p.tracker._frame_index == 5


def test_disabled_is_unchanged_full_pipeline(fast_pipeline):
    s = fast_pipeline; s.pipeline.config = replace(s.pipeline.config, detector_continuation_enable=False)
    for cap in range(1, 9): s.step(cap)
    assert s.pipeline.reid.calls == s.pipeline.detector.calls == 8


def test_cpu_color_check_uses_same_rgb_conversion_as_full_extractor(fast_pipeline):
    image = fast_pipeline.frame
    bgr = _detector_color_features(image, [Detection(BOX, .95, 0)], "BGR")
    rgb = _detector_color_features(image[..., ::-1], [Detection(BOX, .95, 0)], "RGB")
    np.testing.assert_allclose(bgr[0], rgb[0])
