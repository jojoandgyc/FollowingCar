from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
from car_control_modular.control_types import DepthJumpConfirmation, PersonTarget, SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.distance_fusion import DistanceFusionConfig, VisionRadarEncoderDistanceFusion
from test_depth_raw_geometry_runtime import make_runtime, person


TARGET = PersonTarget((220, 80, 420, 400), 1, 0.9, 64000)
PROOF = DepthJumpConfirmation(1, 99.98, 3.1, "reanchor", 3, 3, 5)


def update(fusion, distance=3.1, proof=None, **changes):
    values = dict(
        target=TARGET, frame_height=480, radar_distance_m=distance,
        radar_fresh=True, sample_age_sec=0.02, steering_feedback=None,
        now=100.0, jump_confirmation=proof,
    )
    values.update(changes)
    return fusion.update(**values)


def make_fusion(**changes):
    values = dict(
        enabled=True, radar_median_window=3, fresh_far_jump_m=0.6,
        fresh_far_jump_confirm_frames=5, allow_depth_jump_confirmation=True,
    )
    values.update(changes)
    fusion = VisionRadarEncoderDistanceFusion(DistanceFusionConfig(**values))
    update(fusion, 1.96, now=99.0)
    return fusion


@pytest.mark.parametrize("kind", ["reanchor", "far_jump", "large_clipped_consensus"])
def test_valid_proof_replaces_old_anchor_without_five_more_samples(kind):
    fusion = make_fusion()
    result = update(fusion, proof=replace(PROOF, kind=kind))
    assert result.distance_m == 3.1
    assert result.mode == "radar_depth_confirmed"
    assert result.depth_confirmation_used
    assert result.depth_confirmation_reason == "accepted"
    assert list(fusion._radar_samples) == [3.1]
    assert fusion._pending_far_count == 0


@pytest.mark.parametrize("proof,changes,reason", [
    (asdict(PROOF), {}, "invalid_type"),
    (SimpleNamespace(**asdict(PROOF)), {}, "invalid_type"),
    (replace(PROOF, target_id=2), {}, "target_mismatch"),
    (replace(PROOF, target_id=None), {}, "invalid_numeric"),
    (replace(PROOF, target_id=float("nan")), {}, "invalid_identity_or_count"),
    (replace(PROOF, confirm_count=2), {}, "insufficient_confirmation"),
    (replace(PROOF, required_confirm_frames=1), {}, "insufficient_confirmation"),
    (replace(PROOF, required_confirm_frames=float("inf")), {}, "invalid_identity_or_count"),
    (replace(PROOF, region_count=2), {}, "insufficient_confirmation"),
    (replace(PROOF, region_count=3.5), {}, "invalid_identity_or_count"),
    (replace(PROOF, kind="depth_matched"), {}, "invalid_kind"),
    (replace(PROOF, distance_m=3.0), {}, "distance_mismatch"),
    (replace(PROOF, distance_m=float("nan")), {}, "invalid_numeric"),
    (replace(PROOF, sample_timestamp=100.01), {}, "stale_or_future"),
    (replace(PROOF, sample_timestamp=99.7), {"sample_age_sec": 0.3}, "stale_or_future"),
    (replace(PROOF, sample_timestamp=99.8), {}, "sample_timestamp_mismatch"),
    (PROOF, {"sample_age_sec": None}, "invalid_numeric"),
    (PROOF, {"sample_age_sec": float("nan")}, "invalid_numeric"),
    (PROOF, {"radar_fresh": False}, "not_fresh"),
])
def test_bad_proof_cannot_skip_fusion_guard(proof, changes, reason):
    fusion = make_fusion()
    result = update(fusion, proof=proof, **changes)
    assert not result.depth_confirmation_used
    assert result.depth_confirmation_reason == reason
    assert result.distance_m != 3.1
    if changes.get("radar_fresh", True):
        assert result.mode == "radar_jump_pending"
        assert fusion._pending_far_count == 1


def test_depth_confirmation_disabled_for_mmwave_defaults():
    assert DistanceFusionConfig().allow_depth_jump_confirmation is False
    result = update(make_fusion(allow_depth_jump_confirmation=False), proof=PROOF)
    assert result.mode == "radar_jump_pending"
    assert result.depth_confirmation_reason == "disabled"
    assert not result.depth_confirmation_used


def test_missing_proof_keeps_all_five_plain_far_confirmations():
    fusion = make_fusion(radar_median_window=1)
    results = [update(fusion, now=100 + i * 0.033) for i in range(5)]
    assert [result.mode for result in results[:4]] == ["radar_jump_pending"] * 4
    assert results[-1].distance_m == 3.1
    assert not any(result.depth_confirmation_used for result in results)


@pytest.mark.parametrize("reset_between", [False, True])
def test_proof_cannot_be_replayed_even_after_cache_reset(reset_between):
    fusion = make_fusion()
    assert update(fusion, proof=PROOF).depth_confirmation_used
    if reset_between:
        fusion.reset()
    update(fusion, 0.8, now=100.01)
    repeated = update(fusion, proof=PROOF, now=100.03, sample_age_sec=0.05)
    assert not repeated.depth_confirmation_used
    assert repeated.depth_confirmation_reason == "sample_reused"
    assert repeated.mode == "radar_jump_pending"
    assert repeated.distance_m == 0.8


def test_confirmed_jump_is_not_recovery_blended():
    fusion = make_fusion()
    update(fusion, 1.96, radar_fresh=False, now=99.1)
    assert fusion._last_mode == "visual"
    result = update(fusion, proof=PROOF)
    assert result.distance_m == 3.1
    assert result.mode == "radar_depth_confirmed"


@pytest.mark.parametrize("raw,proof,detail,accepted", [
    (3.1, PROOF, "depth_reanchored_after_timeout", True),
    (2.9, PROOF, "depth_reanchored_after_timeout", False),
    (None, PROOF, "depth_multiregion_reused_hold", False),
    (3.1, None, "depth_reanchored_after_timeout", False),
    (3.1, None, "depth_confirmed_jump", False),
    (3.1, SimpleNamespace(**asdict(PROOF)), "depth_reanchored_after_timeout", False),
])
def test_runtime_requires_typed_proof_matching_current_raw_measurement(
    monkeypatch, raw, proof, detail, accepted,
):
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic", lambda: 100.0)
    runtime, sensors = make_runtime()
    runtime.get_vision_depth_state(640, 480, person())
    sensors.measurement = SimpleNamespace(
        distance_m=3.1, raw_distance_m=raw, sample_age_sec=0.02,
        valid_pixels=500, detail=detail, jump_confirmation=proof,
        sample_timestamp=99.98,
    )
    state = runtime.get_vision_depth_state(640, 480, person())
    assert (state.fusion_mode == "depth_radar_depth_confirmed") is accepted
    assert (state.source_detail == "depth_confirmed_jump") is accepted
    if accepted:
        assert state.used_distance_m == 3.1
    else:
        assert state.used_distance_m == 2.0


@pytest.mark.parametrize("bbox", [
    (240.0, 20.0, 420.0, 479.0),
    (255.9901, 78.5458, 389.8033, 455.6407),
])
@pytest.mark.parametrize("processing_delay", [0.0, 0.05])
def test_real_astra_three_frame_reanchor_reaches_fusion_on_third_sample(monkeypatch, bbox, processing_delay):
    """Real ROI/cluster/temporal code; depth images are synthetic, no device."""
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
    frame_id = [1]

    def sample(millimeters, advance=0.033):
        clock[0] += advance
        astra._latest_depth = np.full((480, 640), millimeters, dtype=np.uint16)
        astra._latest_depth_ts = clock[0] - 0.02
        frame_id[0] += 1
        target = person(bbox=bbox, capture_timestamp=clock[0] - 0.02,
                        capture_frame_id=frame_id[0])
        return runtime.get_vision_depth_state(640, 480, target, use_latest_depth=True)

    assert sample(1800).used_distance_m == 1.8
    states = [sample(2660, advance=2.0), sample(2780), sample(2900)]
    assert states[0].used_distance_m is states[1].used_distance_m is None
    assert states[-1].used_distance_m == 2.9
    assert states[-1].source_detail == "depth_confirmed_jump"
    assert states[-1].fusion_mode == "depth_radar_depth_confirmed"
    assert states[-1].sample_age_sec == pytest.approx(0.02 + processing_delay)
    proof = sensors.measurement.jump_confirmation
    assert isinstance(proof, DepthJumpConfirmation)
    assert proof.kind == "reanchor"
    assert proof.confirm_count == proof.required_confirm_frames == 3
    assert proof.region_count >= 3
    assert proof.sample_timestamp == astra._latest_depth_ts
    assert runtime._vision_depth_fusion._pending_far_count == 0
    controller = FollowSafetyController(FollowPolicyConfig())
    controller._last_target_distance_m = 1.8
    frame = SensorFrame(width=640, height=480, distance_m=states[-1].used_distance_m,
                        distance_state=states[-1])
    assert controller._is_fresh_depth_state(frame)
    assert not controller._distance_longitudinally_untrusted(frame)
    # Typed proof does not override the independent, much larger jump limit.
    controller._last_target_distance_m = 0.8
    assert controller._distance_longitudinally_untrusted(frame)
    # Reusing the accepted sample must not reissue or reconsume its proof.
    target = person(bbox=bbox, capture_timestamp=astra._latest_depth_ts,
                    capture_frame_id=frame_id[0])
    reused = runtime.get_vision_depth_state(640, 480, target, use_latest_depth=True)
    assert sensors.measurement.jump_confirmation is None
    assert sensors.measurement.raw_distance_m is None
    assert reused.source_detail != "depth_confirmed_jump"


@pytest.mark.parametrize("sample_timestamp,reason", [
    (None, "sample_timestamp_missing"),
    (float("nan"), "sample_timestamp_missing"),
    (99.97, "measurement_timestamp_mismatch"),
])
def test_runtime_proof_must_match_independent_measurement_timestamp(monkeypatch, caplog, sample_timestamp, reason):
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic", lambda: 100.0)
    runtime, sensors = make_runtime()
    runtime.get_vision_depth_state(640, 480, person())
    sensors.measurement = SimpleNamespace(
        distance_m=3.1, raw_distance_m=3.1, sample_age_sec=0.02,
        valid_pixels=500, detail="depth_reanchored_after_timeout",
        jump_confirmation=PROOF, sample_timestamp=sample_timestamp,
    )
    with caplog.at_level("INFO"):
        state = runtime.get_vision_depth_state(640, 480, person())
    assert state.used_distance_m == 2.0
    assert state.fusion_mode == "depth_radar_jump_pending"
    assert reason in caplog.text


@pytest.mark.parametrize("jump_confirmation", [PROOF, None])
@pytest.mark.parametrize("sample_timestamp", [99.98, 100.25])
def test_processing_delay_cannot_extend_depth_sample_ttl(monkeypatch, jump_confirmation, sample_timestamp):
    clock = [100.0]
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic", lambda: clock[0])
    runtime, sensors = make_runtime()
    runtime.get_vision_depth_state(640, 480, person())
    clock[0] = 100.24
    # A sample that was 20ms old on entering Astra can expire during work;
    # impossible future samples must also fail closed, with or without proof.
    sensors.measurement = SimpleNamespace(
        distance_m=3.1, raw_distance_m=3.1, sample_age_sec=0.02,
        valid_pixels=500,
        detail="depth_reanchored_after_timeout" if jump_confirmation else "depth_multiregion",
        jump_confirmation=jump_confirmation, sample_timestamp=sample_timestamp,
    )
    target = person(capture_timestamp=100.22)
    state = runtime.get_vision_depth_state(640, 480, target)
    assert state.used_distance_m is None
    assert state.raw_distance_m is None
    assert state.source_detail == "stale_depth_frame"
    assert state.sample_age_sec == pytest.approx(clock[0] - sample_timestamp)
