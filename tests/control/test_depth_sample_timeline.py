"""Depth sample chronology across latest and RGB-aligned consumers; no devices."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
from car_control_modular.control_types import DepthJumpConfirmation, PersonTarget, SteeringFeedback
from car_control_modular.distance_fusion import DistanceFusionConfig, VisionRadarEncoderDistanceFusion
from test_depth_raw_geometry_runtime import make_runtime, person


TARGET = PersonTarget((220.0, 80.0, 420.0, 400.0), 1, 0.9, 64000.0)
CLIPPED = (282.24298095703125, 8.021286010742188, 496.29461669921875, 476.5924072265625)


def fusion():
    return VisionRadarEncoderDistanceFusion(DistanceFusionConfig(
        enabled=True, radar_median_window=1, hold_max_sec=0.20,
        fresh_far_jump_m=0.6, fresh_far_jump_confirm_frames=3,
        allow_depth_jump_confirmation=True,
    ))


def update(runtime, *, stamp=None, now=100.0, distance=2.0, fresh=True,
           target=TARGET, proof=None):
    return runtime.update(
        target=target, frame_height=480, radar_distance_m=distance,
        radar_fresh=fresh, sample_age_sec=0.0 if stamp is None else now - stamp,
        steering_feedback=None, now=now, sample_timestamp=stamp,
        jump_confirmation=proof,
    )


@pytest.mark.parametrize("processing_delay", [0.0, 0.05])
def test_anchor_uses_sample_time_not_processing_completion(processing_delay):
    runtime = fusion()
    accepted = update(runtime, stamp=99.9, now=100.0 + processing_delay)
    assert accepted.anchor_age_sec == pytest.approx(0.1 + processing_delay)
    assert runtime._last_anchor_ts == 99.9
    assert runtime._last_accepted_depth_sample_ts == 99.9
    assert runtime._last_update_ts == 100.0 + processing_delay
    held = update(runtime, now=100.099, fresh=False, distance=None)
    assert held.distance_m == 2.0
    expired = update(runtime, now=100.101, fresh=False, distance=None)
    assert expired.distance_m is None
    assert expired.mode == "expired"
    assert runtime._last_anchor_ts == 99.9


@pytest.mark.parametrize("retry_distance", [2.0, 1.0, 3.0])
def test_same_sample_retry_cannot_change_distance_or_refresh_ttl(retry_distance):
    runtime = fusion()
    update(runtime, stamp=99.9)
    repeated = update(runtime, stamp=99.9, now=100.04, distance=retry_distance)
    assert repeated.sample_timestamp_rejected
    assert repeated.sample_timestamp_reason == "sample_reused"
    assert repeated.distance_m == 2.0
    assert runtime._last_anchor_ts == 99.9
    assert runtime._pending_far_count == 0
    expired = update(runtime, stamp=99.9, now=100.101, distance=retry_distance)
    assert expired.distance_m is None
    assert runtime._last_anchor_ts == 99.9


def test_duplicate_pending_sample_across_sources_counts_only_once():
    runtime = fusion()
    update(runtime, stamp=99.9)
    first = update(runtime, stamp=100.0, now=100.02, distance=3.0)
    repeated = update(runtime, stamp=100.0, now=100.04, distance=3.0)
    older = update(runtime, stamp=99.95, now=100.05, distance=3.0)
    assert first.mode == "radar_jump_pending"
    assert repeated.sample_timestamp_reason == "sample_reused"
    assert older.sample_timestamp_reason == "before_pending_sample"
    assert runtime._pending_far_count == 1
    assert runtime._pending_far_sample_ts == 100.0
    update(runtime, stamp=100.05, now=100.06, distance=3.0)
    assert runtime._pending_far_count == 2
    accepted = update(runtime, stamp=100.09, now=100.10, distance=3.0)
    assert accepted.distance_m == 3.0
    assert accepted.mode == "radar"
    assert runtime._last_anchor_ts == 100.09
    assert runtime._pending_far_count == 0


def test_historical_ordinary_recovery_preserves_newer_pending_evidence():
    runtime = fusion()
    update(runtime, stamp=99.9)
    update(runtime, stamp=100.02, now=100.03, distance=3.0)
    update(runtime, stamp=100.04, now=100.05, distance=3.0)
    assert runtime._pending_far_count == 2
    restored = update(runtime, stamp=99.99, now=100.06, distance=1.95)
    assert restored.distance_m == 1.95
    assert not restored.sample_timestamp_rejected
    assert runtime._last_anchor_ts == 99.99
    assert runtime._pending_far_count == 2
    assert runtime._pending_far_sample_ts == 100.04
    assert update(runtime, stamp=100.02, now=100.07, distance=3.0).sample_timestamp_rejected
    assert runtime._pending_far_count == 2


@pytest.mark.parametrize("old_distance", [1.0, 2.0, 3.0])
def test_historical_sample_cannot_replace_newer_accepted_anchor(old_distance):
    runtime = fusion()
    update(runtime, stamp=100.0, now=100.02)
    update(runtime, stamp=100.1, now=100.12, distance=1.9)
    rejected = update(runtime, stamp=100.05, now=100.15, distance=old_distance)
    assert rejected.sample_timestamp_reason == "before_accepted_anchor"
    assert rejected.distance_m == 1.9
    assert runtime._last_anchor_ts == 100.1
    assert list(runtime._radar_samples) == [1.9]


def test_uid_switch_clears_sample_dedup_and_accepted_watermark():
    runtime = fusion()
    update(runtime, stamp=100.0, now=100.02)
    other = replace(TARGET, track_id=2)
    accepted = update(runtime, stamp=100.0, now=100.03, distance=2.4, target=other)
    assert accepted.distance_m == 2.4
    assert not accepted.sample_timestamp_rejected
    assert runtime._last_anchor_ts == 100.0
    assert runtime._active_track_id == 2


def test_old_sample_cannot_mutate_prediction_via_its_bbox_or_encoder_feedback():
    runtime = fusion()
    update(runtime, stamp=100.1, now=100.12)
    previous_mode = runtime._last_mode
    old_target = replace(TARGET, bbox=(220.0, 0.0, 420.0, 470.0))
    result = runtime.update(
        target=old_target, frame_height=480, radar_distance_m=0.7,
        radar_fresh=True, sample_age_sec=0.1, now=100.15, sample_timestamp=100.05,
        steering_feedback=SteeringFeedback(timestamp=100.14, left_forward_rpm=30.0,
                                           right_forward_rpm=30.0, trustworthy=True),
    )
    assert result.sample_timestamp_rejected
    assert result.distance_m == runtime._last_distance_m == 2.0
    assert runtime._anchor_bbox_height_px == 320.0
    assert runtime._last_update_ts == 100.12
    assert runtime._last_feedback_ts is None
    assert runtime._last_mode == previous_mode


@pytest.mark.parametrize("stamp", [99.7, 100.01, 0.0, float("nan"), float("inf")])
def test_bad_sample_timestamp_cannot_bootstrap_or_replace_anchor(stamp):
    runtime = fusion()
    rejected = update(runtime, stamp=stamp)
    assert rejected.sample_timestamp_rejected
    assert rejected.distance_m is None
    assert runtime._last_anchor_ts is None
    update(runtime, stamp=99.9)
    rejected = update(runtime, stamp=stamp, distance=1.0)
    assert rejected.sample_timestamp_rejected
    assert rejected.distance_m == 2.0
    assert runtime._last_anchor_ts == 99.9


def test_typed_confirmation_cannot_bypass_accepted_sample_chronology():
    runtime = fusion()
    update(runtime, stamp=99.99)
    proof = DepthJumpConfirmation(1, 99.98, 3.0, "reanchor", 3, 3, 5)
    rejected = update(runtime, stamp=99.98, distance=3.0, proof=proof)
    assert rejected.sample_timestamp_reason == "before_accepted_anchor"
    assert not rejected.depth_confirmation_used
    assert runtime._last_depth_confirmation_sample_ts is None
    assert runtime._last_anchor_ts == 99.99


def test_legacy_without_timestamp_keeps_now_based_anchor_and_confirmation():
    runtime = fusion()
    first = update(runtime, now=100.0)
    assert first.anchor_age_sec == 0.0
    assert runtime._last_anchor_ts == 100.0
    # With no sample identity supplied the old call-count semantics are intact.
    for _ in range(3):
        accepted = update(runtime, now=100.02, distance=3.0)
    assert accepted.distance_m == 3.0
    assert runtime._last_anchor_ts == 100.02
    assert runtime._last_accepted_depth_sample_ts is None


def measurement(distance, stamp, *, detail="depth_multiregion"):
    return SimpleNamespace(
        distance_m=distance, raw_distance_m=distance, sample_age_sec=0.02,
        valid_pixels=500, detail=detail, sample_timestamp=stamp,
    )


@pytest.mark.parametrize("processing_delay", [0.0, 0.05])
def test_runtime_propagates_sample_time_and_logs_distinct_ages(monkeypatch, caplog, processing_delay):
    clock = [100.0 + processing_delay]
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic", lambda: clock[0])
    runtime, sensors = make_runtime()
    sensors.measurement = measurement(1.95, 99.9)
    with caplog.at_level("INFO"):
        state = runtime.get_vision_depth_state(640, 480, person())
    assert state.used_distance_m == 1.95
    assert state.sample_age_sec == pytest.approx(0.1 + processing_delay)
    assert runtime._vision_depth_fusion._last_anchor_ts == 99.9
    assert "sample_age_ms=" in caplog.text and "anchor_age_ms=" in caplog.text
    assert "sample_ts=99.9" in caplog.text
    clock[0] = 100.06
    # A duplicated payload may carry raw distance; the runtime must still
    # expose no fresh authority after fusion rejects its sample identity.
    repeated = runtime.get_vision_depth_state(640, 480, person())
    assert repeated.raw_distance_m is None and repeated.sample_count == 0
    assert "sample_reused" in repeated.source_detail
    assert repeated.source_detail.endswith("_hold")
    assert runtime._vision_depth_fusion._last_anchor_ts == 99.9


@pytest.mark.parametrize("processing_delay", [0.0, 0.05])
def test_runtime_historical_sample_recovers_after_newer_failed_attempt(monkeypatch, processing_delay):
    clock = [100.0]
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic", lambda: clock[0])
    runtime, sensors = make_runtime()
    sensors.measurement = measurement(1.967, 99.98)
    runtime.get_vision_depth_state(640, 480, person(capture_timestamp=99.98))
    clock[0] = 100.41
    sensors.measurement = SimpleNamespace(
        distance_m=None, raw_distance_m=None, sample_age_sec=None,
        valid_pixels=500, detail="distance_jump_rate_guard",
    )
    failed = runtime.get_vision_depth_state(640, 480, person(capture_timestamp=100.4))
    assert failed.used_distance_m is None
    clock[0] = 100.42 + processing_delay
    sensors.measurement = measurement(1.925, 100.3)
    recovered = runtime.get_vision_depth_state(640, 480, person(capture_timestamp=100.3))
    assert recovered.raw_distance_m == recovered.used_distance_m == 1.925
    assert recovered.sample_age_sec == pytest.approx(0.12 + processing_delay)
    assert runtime._vision_depth_fusion._last_anchor_ts == 100.3
    clock[0] = 100.501
    expired = runtime.get_vision_depth_state(640, 480, person(capture_timestamp=100.3))
    assert expired.used_distance_m is None
    assert runtime._vision_depth_fusion._last_anchor_ts == 100.3


@pytest.mark.parametrize("processing_delay", [0.0, 0.05])
def test_real_astra_latest_failure_then_unseen_rgb_aligned_recovery(monkeypatch, processing_delay):
    clock = [100.0]
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic", lambda: clock[0])
    astra = AstraDepthRuntime(AstraDepthConfig())
    astra._np = np

    class Sensors:
        measurement = None

        def get_astra_target_distance(self, *args, **kwargs):
            self.measurement = astra.measure_target(*args, **kwargs)
            clock[0] += processing_delay
            return self.measurement

    runtime, _ = make_runtime()
    sensors = Sensors()
    runtime.sensor_runtime = sensors
    astra._latest_depth = np.full((480, 640), 1967, dtype=np.uint16)
    astra._latest_depth_ts = 99.98
    first = runtime.get_vision_depth_state(
        640, 480, person(bbox=CLIPPED, capture_timestamp=99.98), use_latest_depth=True,
    )
    assert first.used_distance_m == 1.967
    historical = np.full((480, 640), 1925, dtype=np.uint16)
    far_background = np.full((480, 640), 4890, dtype=np.uint16)
    astra._depth_history.extend([(100.3, historical), (100.4, far_background)])
    astra._latest_depth_ts, astra._latest_depth = astra._depth_history[-1]
    clock[0] = 100.41
    rejected = runtime.get_vision_depth_state(
        640, 480, person(bbox=CLIPPED, capture_timestamp=100.4), use_latest_depth=True,
    )
    assert rejected.used_distance_m is None
    assert sensors.measurement.rejection_reason == "physically_impossible_far_jump"
    clock[0] = 100.42
    target = person(bbox=CLIPPED, capture_timestamp=100.3)
    recovered = runtime.get_vision_depth_state(640, 480, target, capture_timestamp=100.3)
    assert recovered.raw_distance_m == pytest.approx(1.925)
    assert recovered.used_distance_m == pytest.approx(1.946)
    assert sensors.measurement.sample_timestamp == 100.3
    assert recovered.sample_age_sec == pytest.approx(0.12 + processing_delay)
    assert runtime._vision_depth_fusion._last_anchor_ts == 100.3
    # This is one observation, regardless of which consumer requests it again.
    clock[0] = 100.48
    repeated = runtime.get_vision_depth_state(640, 480, target, capture_timestamp=100.3)
    assert repeated.raw_distance_m is None
    assert astra._last_accepted_ts == runtime._vision_depth_fusion._last_anchor_ts == 100.3


@pytest.mark.parametrize("replay_kind", [
    "duplicate", "older_than_anchor", "older_than_pending", "stale", "future",
])
def test_real_astra_held_old_bbox_cannot_change_fusion_prediction(monkeypatch, replay_kind):
    clock = [100.0]
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic", lambda: clock[0])
    astra = AstraDepthRuntime(AstraDepthConfig(max_unconfirmed_jump_rate_m_s=100.0))
    astra._np = np

    class Sensors:
        measurement = None

        def get_astra_target_distance(self, *args, **kwargs):
            self.measurement = astra.measure_target(*args, **kwargs)
            return self.measurement

    runtime, _ = make_runtime()
    sensors = Sensors()
    runtime.sensor_runtime = sensors
    initial_box = (220.0, 100.0, 420.0, 300.0)
    old_box = (220.0, 40.0, 420.0, 440.0)
    astra._latest_depth = np.full((480, 640), 1967, dtype=np.uint16)
    astra._latest_depth_ts = 99.98
    anchor = runtime.get_vision_depth_state(
        640, 480, person(bbox=initial_box, capture_timestamp=99.98), use_latest_depth=True,
    )
    assert anchor.used_distance_m == 1.967
    if replay_kind == "older_than_pending":
        clock[0] = 100.05
        astra._latest_depth = np.full((480, 640), 3000, dtype=np.uint16)
        astra._latest_depth_ts = 100.03
        runtime.get_vision_depth_state(
            640, 480, person(bbox=initial_box, capture_timestamp=100.03), use_latest_depth=True,
        )
        assert astra._pending_jump_count == 1
        replay_stamp = 100.01
    else:
        replay_stamp = {"duplicate": 99.98, "older_than_anchor": 99.97,
                        "stale": 99.75, "future": 100.10}[replay_kind]
    # RGB itself is fresh even if alignment can only find an expired/future
    # depth image; the depth timeline must reject that observation explicitly.
    capture_stamp = 99.98 if replay_kind in ("stale", "future") else replay_stamp
    astra._depth_history.append((replay_stamp, np.full((480, 640), 1967, dtype=np.uint16)))
    clock[0] = 100.08
    previous_update = runtime._vision_depth_fusion._last_update_ts
    previous_prediction = runtime._vision_depth_fusion._last_distance_m
    previous_feedback = runtime._vision_depth_fusion._last_feedback_ts
    previous_pending = (runtime._vision_depth_fusion._pending_far_count,
                        runtime._vision_depth_fusion._pending_far_sample_ts)
    replay = runtime.get_vision_depth_state(
        640, 480, person(bbox=old_box, capture_timestamp=capture_stamp),
        capture_timestamp=capture_stamp,
        steering_feedback=SteeringFeedback(timestamp=100.075, left_forward_rpm=30.0,
                                           right_forward_rpm=30.0, trustworthy=True),
    )
    expected_status = "stale_or_future" if replay_kind in ("stale", "future") else replay_kind
    assert sensors.measurement.temporal_status == expected_status
    assert sensors.measurement.sample_timestamp is None
    assert sensors.measurement.observation_sample_timestamp == replay_stamp
    assert replay.temporal_status == expected_status
    assert replay.observation_timestamp == replay_stamp
    assert replay.is_replay_of(99.98) == (replay_kind in {"duplicate", "older_than_anchor"})
    assert replay.raw_distance_m is None
    assert replay.sample_count == 0
    assert replay.used_distance_m == previous_prediction
    assert runtime._vision_depth_fusion._last_distance_m == previous_prediction
    assert runtime._vision_depth_fusion._last_update_ts == previous_update
    assert runtime._vision_depth_fusion._last_anchor_ts == 99.98
    assert runtime._vision_depth_fusion._last_feedback_ts == previous_feedback
    assert (runtime._vision_depth_fusion._pending_far_count,
            runtime._vision_depth_fusion._pending_far_sample_ts) == previous_pending
    if replay_kind == "older_than_pending":
        assert astra._pending_jump_count == 1
    if replay_kind == "future":
        return
    # Even repeatedly discarded held values expire from the original sample
    # timestamp, not from the most recent RGB callback or held response.
    clock[0] = 100.181
    expired = runtime.get_vision_depth_state(
        640, 480, person(bbox=old_box, capture_timestamp=capture_stamp),
        capture_timestamp=capture_stamp,
    )
    assert expired.used_distance_m is None
    assert expired.raw_distance_m is None and expired.sample_count == 0
    assert runtime._vision_depth_fusion._last_anchor_ts == 99.98
    assert runtime._vision_depth_fusion._last_feedback_ts == previous_feedback


@pytest.mark.parametrize("missing_frame", [False, True])
def test_real_new_depth_hole_preserves_fresh_visual_and_encoder_short_hold(monkeypatch, missing_frame):
    clock = [100.0]
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic", lambda: clock[0])
    astra = AstraDepthRuntime(AstraDepthConfig())
    astra._np = np

    class Sensors:
        measurement = None

        def get_astra_target_distance(self, *args, **kwargs):
            self.measurement = astra.measure_target(*args, **kwargs)
            return self.measurement

    runtime, _ = make_runtime()
    sensors = Sensors()
    runtime.sensor_runtime = sensors
    astra._latest_depth = np.full((480, 640), 2000, dtype=np.uint16)
    astra._latest_depth_ts = 99.98
    feedback = SteeringFeedback(timestamp=99.99, left_forward_rpm=30.0,
                                right_forward_rpm=30.0, trustworthy=True)
    runtime.get_vision_depth_state(
        640, 480, person(bbox=(220.0, 100.0, 420.0, 300.0), capture_timestamp=99.98),
        use_latest_depth=True, steering_feedback=feedback,
    )
    clock[0] = 100.05
    astra._latest_depth = None if missing_frame else np.zeros((480, 640), dtype=np.uint16)
    astra._latest_depth_ts = 100.03
    held = runtime.get_vision_depth_state(
        640, 480, person(bbox=(220.0, 40.0, 420.0, 440.0), capture_timestamp=100.03),
        use_latest_depth=True, steering_feedback=replace(feedback, timestamp=100.04),
    )
    assert sensors.measurement.temporal_status == ("no_sample" if missing_frame else "new_sample")
    assert sensors.measurement.raw_distance_m is None
    assert held.fusion_visual_distance_m == 1.0
    assert held.fusion_encoder_delta_m > 0.0
    assert held.fusion_mode == "depth_visual_encoder"
    assert held.used_distance_m < 2.0
    assert runtime._vision_depth_fusion._last_anchor_ts == 99.98


def test_real_astra_expired_during_sampling_cannot_mutate_fusion_state(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic", lambda: clock[0])
    astra = AstraDepthRuntime(AstraDepthConfig())
    astra._np = np

    class Sensors:
        measurement = None

        def get_astra_target_distance(self, *args, **kwargs):
            self.measurement = astra.measure_target(*args, **kwargs)
            return self.measurement

    runtime, _ = make_runtime()
    sensors = Sensors()
    runtime.sensor_runtime = sensors
    astra._latest_depth = np.full((480, 640), 2000, dtype=np.uint16)
    astra._latest_depth_ts = 99.98
    runtime.get_vision_depth_state(
        640, 480, person(bbox=(220.0, 100.0, 420.0, 300.0), capture_timestamp=99.98),
        use_latest_depth=True,
    )
    original_selection = astra._select_multiregion_distance

    def slow_selection(*args, **kwargs):
        result = original_selection(*args, **kwargs)
        clock[0] += 0.26
        return result

    monkeypatch.setattr(astra, "_select_multiregion_distance", slow_selection)
    clock[0] = 100.08
    astra._latest_depth_ts = 100.06
    expired = runtime.get_vision_depth_state(
        640, 480, person(bbox=(220.0, 40.0, 420.0, 440.0), capture_timestamp=100.06),
        use_latest_depth=True,
        steering_feedback=SteeringFeedback(timestamp=100.07, left_forward_rpm=30.0,
                                           right_forward_rpm=30.0, trustworthy=True),
    )
    assert sensors.measurement.temporal_status == "expired_during_sampling"
    assert sensors.measurement.sample_timestamp is None
    assert sensors.measurement.observation_sample_timestamp == 100.06
    assert expired.used_distance_m is expired.raw_distance_m is None
    assert expired.sample_count == 0
    assert expired.source_detail == "depth_sample_observation_discarded"
    assert runtime._vision_depth_fusion._last_anchor_ts == 99.98
    assert runtime._vision_depth_fusion._last_update_ts == 100.0
    assert runtime._vision_depth_fusion._last_distance_m == 2.0
    assert runtime._vision_depth_fusion._last_feedback_ts is None
