"""Exact cluster optimization and non-predictive private-scan cancellation."""
from copy import deepcopy
from types import MethodType

import numpy as np
import pytest

from car_control_modular.astra_depth import (
    AstraDepthConfig, AstraDepthRuntime, _DepthClusterCandidate,
)
from car_control_modular.depth_measurement_transaction import RANGING_STATE_FIELDS
from test_depth_optimistic_transaction import scene, prepare
from test_turn_depth_scheduling import owner, context


def scalar_candidates(runtime, patch, region_name, **_ignored):
    """Pre-optimization implementation, retained only as a test oracle."""
    cfg = runtime.config
    min_mm = int(round(max(0., cfg.min_distance_m) * 1000))
    max_mm = int(round(max(cfg.min_distance_m, cfg.max_distance_m) * 1000))
    valid_mask = (patch >= min_mm) & (patch <= max_mm)
    valid = patch[valid_mask]
    total = int(valid.size)
    required = runtime._dynamic_required_pixels(int(patch.size))
    if total < required:
        return []
    values = np.sort(valid.reshape(-1).astype(np.int32, copy=False))
    span = max(50, int(round(max(.05, cfg.foreground_cluster_span_m) * 1000)))
    candidates, index = [], 0
    while index < total:
        end = int(runtime._np.searchsorted(values, int(values[index]) + span, side="right"))
        if end - index < required:
            index += 1
            continue
        band = valid_mask & (patch >= int(values[index])) & (patch <= int(values[end - 1]))
        count = int(np.count_nonzero(band))
        spatial = runtime._spatial_support_mask(band, np)
        supported = int(np.count_nonzero(spatial))
        fraction = supported / max(1, count)
        if supported >= required and fraction >= max(0., min(1., cfg.foreground_spatial_support_fraction)):
            candidates.append(_DepthClusterCandidate(
                float(np.median(patch[spatial])) / 1000., supported, total,
                required, fraction, region_name,
            ))
        index = max(index + 1, end)
    return candidates


def camera(**kwargs):
    runtime = AstraDepthRuntime(AstraDepthConfig(**kwargs))
    runtime._np = np
    return runtime


def patch_cases(seed):
    rng = np.random.default_rng(seed)
    full = rng.integers(0, 9000, (64, 64), dtype=np.uint16)
    yield full
    yield full[::2, ::2]  # Noncontiguous camera view.
    yield np.zeros((16, 16), dtype=np.uint16)
    yield np.full((64, 64), 1800, dtype=np.uint16)
    yield np.linspace(350, 8000, 256, dtype=np.uint16).reshape(16, 16)
    sparse = np.zeros((64, 64), dtype=np.uint16)
    sparse[::3, ::3] = rng.integers(1600, 1850, sparse[::3, ::3].shape, dtype=np.uint16)
    yield sparse
    bands = full.copy()
    bands[:24, :] = rng.integers(1780, 1820, (24, 64), dtype=np.uint16)
    bands[32:48, :] = rng.integers(3800, 3860, (16, 64), dtype=np.uint16)
    yield bands
    yield rng.choice(np.array([0, 349, 350, 400, 401, 1800, 1850, 1851, 8000, 8001],
                             dtype=np.uint16), (16, 16))


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("span,fraction,floor", [(.05, .03, 20), (.4, .03, 20), (.4, .2, 1)])
def test_every_cluster_field_matches_scalar_algorithm(seed, span, fraction, floor):
    runtime = camera(foreground_cluster_span_m=span, dynamic_min_valid_fraction=fraction,
                     dynamic_min_valid_floor=floor)
    for patch in patch_cases(seed):
        assert runtime._region_cluster_candidates(patch, "chest_center") == scalar_candidates(
            runtime, patch, "chest_center")


@pytest.mark.parametrize("seed", range(8))
def test_whole_measurement_and_diagnostics_match_reference(seed, monkeypatch):
    monkeypatch.setattr("car_control_modular.astra_depth.time.monotonic", lambda: 100.)
    new, old = camera(), camera()
    old._region_cluster_candidates = MethodType(scalar_candidates, old)
    rng = np.random.default_rng(seed)
    frame = rng.integers(1760, 1840, (480, 640), dtype=np.uint16)
    frame[rng.random(frame.shape) < .15] = 0
    frame[:, :250] = 4000 if seed % 2 else 1700
    for runtime in (old, new):
        runtime._latest_depth, runtime._latest_depth_ts = frame, 99.98
    bbox = (0., 20., 420., 479.) if seed % 2 else (160., 40., 480., 440.)
    measurements = [r.measure_target(bbox, 640, 480, target_id=1, use_latest_depth=True)
                    for r in (old, new)]
    assert measurements[0] == measurements[1]
    assert old._measurement_regions == new._measurement_regions
    assert old._last_torso_selection == new._last_torso_selection
    assert old._distance_history == new._distance_history


def test_diffuse_noise_no_longer_does_one_scalar_search_per_pixel():
    class CountSearches:
        calls = 0
        def __getattr__(self, name):
            return getattr(np, name)
        def searchsorted(self, *args, **kwargs):
            self.calls += 1
            return np.searchsorted(*args, **kwargs)
    runtime = camera(foreground_cluster_span_m=.05)
    runtime._np = counted = CountSearches()
    patch = np.linspace(350, 8000, 256, dtype=np.uint16).reshape(16, 16)
    assert scalar_candidates(runtime, patch, "chest_center") == []
    assert counted.calls == 256
    counted.calls = 0
    assert runtime._region_cluster_candidates(patch, "chest_center") == []
    assert counted.calls <= 2


def test_valid_region_pixels_are_extracted_once_for_selection_and_diagnostics(monkeypatch):
    runtime = camera()
    calls, original = [], runtime._region_pixel_evidence
    def evidence(patch):
        calls.append(patch.shape)
        return original(patch)
    monkeypatch.setattr(runtime, "_region_pixel_evidence", evidence)
    runtime._select_multiregion_distance(np.full((480, 640), 1800, np.uint16),
                                        (160., 40., 480., 440.), 640, 480, None)
    assert len(calls) == 5


def test_expired_private_scan_stops_between_regions_without_mutating_live_state(scene, monkeypatch):
    obj, live, distance, clock = scene
    item, target = prepare(scene)
    original = item.transaction.private._region_cluster_candidates
    calls = []
    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(True)
        clock[0] += .170  # Physical age .190: already impossible, not predicted.
        return result
    monkeypatch.setattr(item.transaction.private, "_region_cluster_candidates", delayed)
    state = deepcopy({key: getattr(live, key) for key in RANGING_STATE_FIELDS if hasattr(live, key)})
    grant = obj._depth30_linear_snapshot
    assert item.transaction.run() is None
    assert item.transaction.reject_reason == "physical_sample_expired_during_compute"
    assert len(calls) == 1
    assert not distance.commit_prepared_depth(item, target=target)
    assert obj._depth30_linear_snapshot is grant
    assert state == {key: getattr(live, key) for key in state}
    # No scan-duration estimate poisons the next fresh frame.
    live._latest_depth_ts = clock[0] - .02
    obj._longitudinal_context = context(4911, clock[0] - .04)
    newer, target = prepare(scene)
    assert newer.transaction.run().raw_distance_m == pytest.approx(1.8)
    assert distance.commit_prepared_depth(newer, target=target)


@pytest.mark.parametrize("remaining", [.001, .040, .075])
def test_positive_remaining_budget_is_not_rejected_by_estimated_scan_cost(scene, remaining):
    obj, live, distance, clock = scene
    live._latest_depth_ts = clock[0] - (.18 - remaining)
    item, target = prepare(scene)
    assert item.transaction.run().raw_distance_m == pytest.approx(1.8)
    assert distance.commit_prepared_depth(item, target=target)
    assert live._last_accepted_ts == pytest.approx(100. - (.18 - remaining))


def test_expired_sample_skips_all_private_pixel_scans(scene, monkeypatch):
    _obj, live, distance, clock = scene
    live._latest_depth_ts = clock[0] - .181
    item, target = prepare(scene)
    monkeypatch.setattr(item.transaction.private, "_region_cluster_candidates",
                        lambda *a, **kw: pytest.fail("expired sample must not scan"))
    assert item.transaction.run() is None
    assert item.transaction.reject_reason == "physical_sample_expired_during_compute"
    assert not distance.commit_prepared_depth(item, target=target)


def test_sensor_failures_are_not_misreported_as_deadline_expiry(scene, monkeypatch):
    item, _ = prepare(scene)
    def broken(*args, **kwargs):
        raise ValueError("real sensor/array failure")
    monkeypatch.setattr(item.transaction.private, "_region_cluster_candidates", broken)
    with pytest.raises(ValueError, match="real sensor/array failure"):
        item.transaction.run()
    assert item.transaction.reject_reason is None


def test_direct_fallback_does_not_inherit_private_transaction_deadline(scene, monkeypatch):
    _obj, live, _distance, clock = scene
    original = live._select_multiregion_distance
    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] += .180
        return result
    monkeypatch.setattr(live, "_select_multiregion_distance", delayed)
    result = live.measure_target((160., 40., 480., 440.), 640, 480,
                                target_id=1, use_latest_depth=True)
    assert result.raw_distance_m == pytest.approx(1.8)
    assert not hasattr(live, "_transaction_sample_max_age_sec")
    # This is only the sensor's fallback result: downstream still owns admission.
    assert clock[0] - result.sample_timestamp == pytest.approx(.200)
