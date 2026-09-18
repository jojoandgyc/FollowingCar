"""Real-array torso selection/ROI-gate checks; never start camera hardware."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular import astra_depth
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime


BBOX = (220.0, 80.0, 420.0, 400.0)


@pytest.fixture
def sensor(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(astra_depth.time, "monotonic", lambda: now[0])
    runtime = AstraDepthRuntime(AstraDepthConfig())
    runtime._np = np

    def sample(depth, *, advance=0.033, uid=1):
        now[0] += advance
        runtime._latest_depth = (
            np.full((480, 640), depth, dtype=np.uint16)
            if np.isscalar(depth) else depth.copy()
        )
        runtime._latest_depth_ts = now[0] - 0.01
        return runtime.measure_target(BBOX, 640, 480, target_id=uid, use_latest_depth=True)

    return runtime, now, sample


def torso_patch(runtime, distance_mm=1980, *, background_mm=0, two_regions=False):
    depth = np.full((480, 640), background_mm, dtype=np.uint16)
    regions, _clipped = runtime._torso_sampling_regions(BBOX, 640, 480, 640, 480)
    chosen = [regions[0], regions[-1]] if two_regions else [regions[0]]
    for _name, left, top, _right, _bottom in chosen:
        # 120 connected pixels exceed this region's unchanged 68-pixel gate,
        # but one or two patches remain below the main ROI's 320-pixel gate.
        depth[top + 1:top + 11, left + 1:left + 13] = distance_mm
    return depth


def test_supported_near_cluster_beats_five_background_regions(sensor):
    runtime, _now, sample = sensor
    sample(1960)
    mixed = torso_patch(runtime, background_mm=2650)
    selected = runtime._select_multiregion_distance(mixed, BBOX, 640, 480, 0.1)
    assert selected[0] == pytest.approx(1.98)
    assert selected[1] == 120 and selected[3] == 68
    assert selected[4] == "chest_center"
    result = sample(mixed)
    assert result.raw_distance_m == pytest.approx(1.98)
    assert 1.96 <= result.distance_m <= 1.98
    assert result.jump_confirmation is None


@pytest.mark.parametrize("two_regions", [False, True])
def test_supported_continuous_near_regions_survive_sparse_main_roi(sensor, two_regions):
    runtime, _now, sample = sensor
    sample(1960)
    sparse = torso_patch(runtime, two_regions=two_regions)
    left, top, right, bottom = runtime._scaled_target_roi(BBOX, 640, 480, 640, 480, runtime.config)
    valid = int(np.count_nonzero(sparse[top:bottom, left:right]))
    required = runtime._dynamic_required_pixels((right - left) * (bottom - top))
    assert valid < required
    result = sample(sparse)
    assert result.raw_distance_m == pytest.approx(1.98)
    assert result.region_count == (2 if two_regions else 1)
    assert result.sample_timestamp == runtime._latest_depth_ts
    assert result.jump_confirmation is None
    assert result.sparse_torso_continuation is True
    assert result.roi_valid_pixels == valid
    assert result.roi_required_valid_pixels == required
    assert result.detail == "depth_torso_continuation"


def test_sparse_near_region_cannot_acquire_new_target_without_anchor(sensor):
    runtime, _now, sample = sensor
    result = sample(torso_patch(runtime))
    assert result.raw_distance_m is result.distance_m is None
    assert result.jump_confirmation is None


@pytest.mark.parametrize("age", [0.61, 1.6])
def test_sparse_near_region_cannot_use_old_anchor_to_reacquire(sensor, age):
    runtime, _now, sample = sensor
    sample(1960)
    result = sample(torso_patch(runtime), advance=age)
    assert result.raw_distance_m is result.distance_m is None
    assert result.jump_confirmation is None


def test_main_roi_exception_does_not_lower_local_pixel_gate(sensor):
    runtime, _now, sample = sensor
    sample(1960)
    depth = np.zeros((480, 640), dtype=np.uint16)
    regions, _ = runtime._torso_sampling_regions(BBOX, 640, 480, 640, 480)
    _name, left, top, _right, _bottom = regions[0]
    depth[top + 1:top + 6, left + 1:left + 14] = 1980  # 65 < 68
    result = sample(depth)
    assert result.raw_distance_m is None
    assert result.jump_confirmation is None


def test_main_roi_exception_does_not_accept_disconnected_pixels(sensor):
    runtime, _now, sample = sensor
    sample(1960)
    depth = np.zeros((480, 640), dtype=np.uint16)
    regions, _ = runtime._torso_sampling_regions(BBOX, 640, 480, 640, 480)
    _name, left, top, _right, _bottom = regions[0]
    depth[top + 1:top + 31:3, left + 1:left + 37:3] = 1980  # 120 isolated points
    result = sample(depth)
    assert result.raw_distance_m is None
    assert result.jump_confirmation is None


def test_sparse_far_continuity_still_requires_normal_main_roi_gate(sensor):
    runtime, _now, sample = sensor
    sample(4000)
    result = sample(torso_patch(runtime, 4000))
    assert result.raw_distance_m is None
    assert result.jump_confirmation is None


def test_sparse_closer_safety_surface_cannot_instantly_replace_anchor(sensor):
    runtime, _now, sample = sensor
    sample(1960)
    result = sample(torso_patch(runtime, 900))
    assert result.raw_distance_m is None
    assert result.distance_m == pytest.approx(1.96)
    assert result.jump_confirmation is None
    assert runtime._last_accepted_distance_m == pytest.approx(1.96)


def test_expired_anchor_does_not_override_real_current_background_consensus(sensor):
    runtime, _now, sample = sensor
    sample(1960)
    mixed = torso_patch(runtime, background_mm=2650)
    selected = runtime._select_multiregion_distance(mixed, BBOX, 640, 480, 1.6)
    assert selected[0] == pytest.approx(2.65)
    assert len(selected[4].split("+")) == 5


def test_sparse_measurement_cannot_carry_previous_uid_anchor(sensor):
    runtime, _now, sample = sensor
    sample(1960, uid=1)
    result = sample(torso_patch(runtime), uid=2)
    assert result.raw_distance_m is result.distance_m is None
    assert result.jump_confirmation is None
