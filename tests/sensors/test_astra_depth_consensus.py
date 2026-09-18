"""Hardware-free checks for clipped/multi-region far-depth confirmation."""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular import astra_depth
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
from car_control_modular.control_types import DepthJumpConfirmation


CLIPPED = (240.0, 20.0, 420.0, 479.0)
CENTERED = (220.0, 80.0, 420.0, 400.0)


@pytest.fixture
def sensor(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(astra_depth.time, "monotonic", lambda: clock[0])
    runtime = AstraDepthRuntime(AstraDepthConfig())
    runtime._np = np

    def sample(millimeters, *, bbox=CLIPPED, uid=1, advance=0.033, age=0.02):
        clock[0] += advance
        depth = (
            np.full((480, 640), millimeters, dtype=np.uint16)
            if np.isscalar(millimeters) else millimeters.copy()
        )
        runtime._latest_depth = depth
        runtime._latest_depth_ts = clock[0] - age
        return runtime.measure_target(bbox, 640, 480, target_id=uid, use_latest_depth=True)

    return runtime, clock, sample


def assert_proof(measurement, *, kind, uid=1, count=3):
    proof = measurement.jump_confirmation
    assert isinstance(proof, DepthJumpConfirmation)
    assert proof.kind == kind
    assert proof.target_id == uid
    assert proof.confirm_count == count
    assert proof.required_confirm_frames == count
    assert proof.region_count >= 3
    assert measurement.confirm_count == count
    assert proof.distance_m == measurement.distance_m == measurement.raw_distance_m
    assert proof.sample_timestamp == measurement.sample_timestamp
    assert measurement.sample_age_sec == pytest.approx(0.02)
    return proof


@pytest.mark.parametrize("bbox", [
    CLIPPED,
    (80.0, 20.0, 560.0, 460.0),  # large, but not clipped
    (0.0, 80.0, 200.0, 400.0),  # clipped, below both large-box thresholds
])
def test_initial_far_multiregion_needs_three_fresh_samples(sensor, bbox):
    runtime, clock, sample = sensor
    first, second, third = [sample(value, bbox=bbox) for value in (2660, 2750, 2800)]
    assert first.distance_m is second.distance_m is None
    assert [first.confirm_count, second.confirm_count] == [1, 2]
    assert first.jump_confirmation is second.jump_confirmation is None
    proof = assert_proof(third, kind="large_clipped_consensus")
    assert proof.sample_timestamp == pytest.approx(clock[0] - third.sample_age_sec)
    assert proof.sample_timestamp == runtime._latest_depth_ts
    assert third.distance_m == 2.8


def test_expired_anchor_recovers_without_waiting_for_height_under_90_percent(sensor):
    runtime, _clock, sample = sensor
    assert sample(1960).distance_m == 1.96
    first = sample(2660, advance=2.0)
    second = sample(2780)
    third = sample(2900)
    assert first.bbox_clipped and second.bbox_clipped and third.bbox_clipped
    assert (CLIPPED[3] - CLIPPED[1]) / 480.0 > 0.90
    assert [first.confirm_count, second.confirm_count] == [1, 2]
    assert first.distance_m is second.distance_m is None
    assert_proof(third, kind="reanchor")
    assert third.distance_m == 2.9
    assert tuple(runtime._distance_history) == (2.9,)


def test_recent_anchor_keeps_rate_guard_even_below_50_ms(sensor):
    _runtime, _clock, sample = sensor
    assert sample(900, age=0.0).distance_m == 0.9
    for advance in (0.01, 0.02, 0.10):
        result = sample(3000, advance=advance, age=0.0)
        assert result.rejection_reason == "physically_impossible_far_jump"
        assert result.confirm_count == 0
        assert result.jump_confirmation is None
        assert result.raw_distance_m is None
        assert result.distance_m in (None, 0.9)
    pending = sample(3000, advance=1.6)
    assert pending.confirm_count == 1
    sample(3000)
    assert_proof(sample(3000), kind="reanchor")


def test_recent_but_plausible_clipped_consensus_passes_geometry_guard(sensor):
    _runtime, _clock, sample = sensor
    sample(1800)
    first = sample(2700, advance=0.38)
    assert first.confirm_count == 1
    assert first.required_confirm_frames == 5
    results = [first] + [sample(2700) for _ in range(4)]
    assert all(item.raw_distance_m is None for item in results[:-1])
    assert_proof(results[-1], kind="far_jump", count=5)


def test_single_clipped_region_cannot_seed_or_continue_confirmation(sensor):
    runtime, _clock, sample = sensor
    first = sample(3000)
    assert first.confirm_count == 1
    sparse = np.zeros((480, 640), dtype=np.uint16)
    regions, _ = runtime._torso_sampling_regions(CLIPPED, 640, 480, 640, 480)
    _name, left, top, right, bottom = regions[-1]
    sparse[top:bottom, left:right] = 4580
    rejected = sample(sparse)
    assert rejected.region_count == 1
    assert rejected.rejection_reason == "clipped_bbox_single_region"
    assert rejected.distance_m is rejected.jump_confirmation is None
    assert runtime._pending_jump_count == 0
    assert sample(3000).confirm_count == 1


def test_disconnected_pixels_in_all_regions_are_not_spatial_consensus(sensor):
    _runtime, _clock, sample = sensor
    disconnected = np.zeros((480, 640), dtype=np.uint16)
    disconnected[::3, ::3] = 3000
    for _ in range(4):
        result = sample(disconnected)
        assert result.valid_pixels >= result.required_valid_pixels
        assert result.region_count == 0
        assert result.rejection_reason == "clipped_bbox_single_region"
        assert result.distance_m is result.jump_confirmation is None


@pytest.mark.parametrize("names", ["chest+abdomen", "chest+chest+abdomen"])
def test_two_distinct_regions_are_not_three_region_proof(sensor, monkeypatch, names):
    runtime, _clock, sample = sensor
    monkeypatch.setattr(runtime, "_select_multiregion_distance", lambda *args:
                        (3.0, 2000, 3000, 80, names, True))
    for _ in range(4):
        rejected = sample(3000)
        assert rejected.distance_m is rejected.jump_confirmation is None
        assert rejected.region_count == 2
        assert runtime._pending_jump_count == 0


def test_reused_depth_does_not_advance_or_reissue_confirmation(sensor):
    runtime, clock, sample = sensor
    first = sample(3000)
    clock[0] += 0.005
    reused = runtime.measure_target(CLIPPED, 640, 480, target_id=1, use_latest_depth=True)
    assert first.confirm_count == reused.confirm_count == 1
    assert reused.rejection_reason == "reused_depth_frame"
    assert reused.jump_confirmation is None
    assert reused.sample_timestamp is None
    assert sample(3000).confirm_count == 2
    accepted = sample(3000)
    assert_proof(accepted, kind="large_clipped_consensus")
    again = runtime.measure_target(CLIPPED, 640, 480, target_id=1, use_latest_depth=True)
    assert again.raw_distance_m is None
    assert again.jump_confirmation is None
    assert again.sample_timestamp is None
    assert again.detail.endswith("_reused_hold")
    ordinary = sample(3020)
    assert ordinary.jump_confirmation is None
    assert ordinary.raw_distance_m == 3.02
    assert ordinary.sample_timestamp == runtime._latest_depth_ts


@pytest.mark.parametrize("interruption", ["invalid", "stale", "gap", "different_surface"])
def test_confirmation_chain_restarts_after_rejected_or_disconnected_sample(sensor, interruption):
    runtime, clock, sample = sensor
    assert sample(3000).confirm_count == 1
    assert sample(3000).confirm_count == 2
    if interruption == "invalid":
        rejected = sample(0)
        assert rejected.detail == "insufficient_depth_pixels"
    elif interruption == "stale":
        rejected = sample(3000, age=0.30)
        assert rejected.detail == "stale_depth_frame"
    elif interruption == "different_surface":
        rejected = sample(4000)
        assert rejected.confirm_count == 1
    else:
        clock[0] += runtime.config.max_frame_age_sec + 0.01
    resumed = sample(3000)
    assert resumed.confirm_count == 1
    assert resumed.distance_m is resumed.jump_confirmation is None
    assert sample(3000).confirm_count == 2
    assert_proof(sample(3000), kind="large_clipped_consensus")


def test_switching_uid_discards_confirmation_and_held_anchor(sensor):
    runtime, _clock, sample = sensor
    assert sample(3000, uid=1).confirm_count == 1
    assert sample(3000, uid=1).confirm_count == 2
    assert sample(3000, uid=2).confirm_count == 1
    assert sample(3000, uid=2).confirm_count == 2
    assert_proof(sample(3000, uid=2), kind="large_clipped_consensus", uid=2)
    runtime._latest_depth = None
    runtime._depth_history.clear()
    no_frame = runtime.measure_target(CLIPPED, 640, 480, target_id=3, use_latest_depth=True)
    assert no_frame.distance_m is no_frame.jump_confirmation is None
    assert no_frame.sample_timestamp is None


def test_far_jump_proof_requires_every_confirmed_sample_to_have_three_regions(sensor, monkeypatch):
    runtime, _clock, sample = sensor
    assert sample(1900, bbox=CENTERED).distance_m == pytest.approx(1.9)
    regions = ["chest+abdomen", "chest+abdomen+lower_abdomen"]
    monkeypatch.setattr(runtime, "_select_multiregion_distance", lambda *args:
                        (2.9, 2000, 3000, 80, regions.pop(0), False))
    first = sample(2900, bbox=CENTERED, advance=0.5)
    accepted = sample(2900, bbox=CENTERED)
    assert first.confirm_count == 1
    assert accepted.distance_m == 2.9
    assert accepted.confirm_count == 2
    assert accepted.region_count == 3
    assert accepted.jump_confirmation is None


def test_ordinary_single_sample_and_closer_acceptance_have_no_jump_proof(sensor):
    _runtime, _clock, sample = sensor
    initial = sample(3000, bbox=CENTERED)
    assert initial.distance_m == 3.0
    assert initial.jump_confirmation is None
    closer = sample(900, bbox=CENTERED)
    assert closer.distance_m == 0.9
    assert closer.jump_confirmation is None


def test_initial_small_far_jump_has_valid_two_frame_proof(sensor):
    _runtime, _clock, sample = sensor
    sample(1900, bbox=CENTERED)
    first = sample(2900, bbox=CENTERED, advance=0.5)
    assert first.confirm_count == 1 and first.jump_confirmation is None
    assert_proof(sample(2900, bbox=CENTERED), kind="far_jump", count=2)


def test_actual_rate_rejected_sample_clears_existing_pending_chain(sensor):
    runtime, _clock, sample = sensor
    sample(1800)
    assert sample(2700, advance=0.38).confirm_count == 1
    assert sample(2700).confirm_count == 2
    rejected = sample(4580)
    assert rejected.rejection_reason == "physically_impossible_far_jump"
    assert runtime._pending_jump_count == 0
    assert rejected.jump_confirmation is None
    assert sample(2700).confirm_count == 1


@pytest.mark.parametrize("uid", [None, 0, -1])
def test_unbound_identity_never_receives_proof(sensor, uid):
    _runtime, _clock, sample = sensor
    sample(3000, uid=uid)
    sample(3000, uid=uid)
    result = sample(3000, uid=uid)
    assert result.distance_m == 3.0
    assert result.jump_confirmation is None


def test_confirmation_diagnostic_includes_regions_and_structured_proof(sensor, caplog):
    _runtime, _clock, sample = sensor
    with caplog.at_level(logging.INFO):
        sample(3000)
        sample(3000)
        sample(3000)
    diagnostics = [record.getMessage() for record in caplog.records
                   if "Astra depth diagnostic:" in record.getMessage()]
    assert any("confirm=1/3" in item and "jump_confirmation=none" in item
               for item in diagnostics)
    assert any("confirm=3/3" in item and "region_count=5" in item
               and "jump_confirmation=DepthJumpConfirmation(" in item
               for item in diagnostics)
