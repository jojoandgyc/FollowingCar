"""Bounded scan optimization, real worker clocks, and abort provenance."""
from copy import deepcopy
import logging

import numpy as np
import pytest

from car_control_modular.depth_measurement_transaction import RANGING_STATE_FIELDS
from test_depth_optimistic_transaction import scene, prepare
from test_depth_scan_equivalence import camera, scalar_candidates
from test_history_depth_compute_budget import history
from test_turn_depth_scheduling import owner


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float64])
@pytest.mark.parametrize("support", [0., .55, 1.])
def test_one_band_matches_original_at_exact_thresholds_and_with_invalid_values(dtype, support):
    runtime = camera(foreground_cluster_span_m=.4,
                     foreground_spatial_support_fraction=support)
    patches = [
        np.full((16, 16), 1800, dtype=dtype),
        np.tile(np.array([1800, 2200], dtype=dtype), (16, 8)),  # Exact inclusive span.
        np.tile(np.array([1800, 2201], dtype=dtype), (16, 8)),
        np.tile(np.array([0, 349, 350, 750, 8000, 8001, 65535, 350], dtype=dtype), (16, 2)),
    ]
    isolated = np.zeros((16, 16), dtype=dtype)
    isolated[::2, ::2] = 1800
    patches.extend([isolated, patches[1][::2, ::2]])
    if dtype is np.float64:
        patches.extend([
            np.tile(np.array([np.nan, np.inf, -np.inf, 1800.1, 2200.1, 1800., 2200., 0.]), (16, 2)),
            np.full((16, 16), 1800.75),
        ])
    for patch in patches:
        assert runtime._region_cluster_candidates(patch, "chest_center") == scalar_candidates(
            runtime, patch, "chest_center")


def test_coherent_uint16_band_eliminates_sort_without_changing_evidence():
    class CountSorts:
        calls = 0

        def __getattr__(self, name):
            return getattr(np, name)

        def sort(self, *args, **kwargs):
            self.calls += 1
            return np.sort(*args, **kwargs)

    runtime = camera()
    runtime._np = counter = CountSorts()
    patch = np.arange(1800, 2201, dtype=np.uint16).reshape(1, 401).repeat(8, axis=0)
    expected = scalar_candidates(runtime, patch, "abdomen_center")
    assert runtime._region_cluster_candidates(patch, "abdomen_center") == expected
    assert counter.calls == 0
    patch[:, -1] = 2201  # The exact shortcut must not swallow the next band.
    expected = scalar_candidates(runtime, patch, "abdomen_center")
    assert runtime._region_cluster_candidates(patch, "abdomen_center") == expected
    assert counter.calls == 1


@pytest.mark.parametrize("size,keep,valid", [(4, 4, 0), (15, 4, 8), (16, 4, 256), (17, 600, 12)])
def test_central_diagnostic_counts_match_helper_without_recomputing_unused_distance(
        monkeypatch, caplog, size, keep, valid):
    runtime = camera(center_patch_size=size, center_patch_keep_count=keep)
    runtime._latest_depth = np.full((480, 640), 1800, dtype=np.uint16)
    runtime._latest_depth_ts = 99.98
    monkeypatch.setattr("car_control_modular.astra_depth.time.monotonic", lambda: 100.)
    bbox = (160., 40., 480., 440.)
    left, top, right, bottom = runtime._scaled_target_roi(bbox, 640, 480, 640, 480, runtime.config)
    x = (left+right-1)//2-size//2+1
    y = (top+bottom-1)//2-size//2+1
    runtime._latest_depth[y:y+size, x:x+size] = 0
    patch = runtime._latest_depth[y:y+size, x:x+size]
    yy, xx = np.unravel_index(np.arange(min(valid, size*size)), patch.shape)
    patch[yy, xx] = 1800
    _, expected_valid, expected_pixels, expected_kept = runtime._select_center_patch_distance(
        runtime._latest_depth, left, top, right, bottom)
    monkeypatch.setattr(runtime, "_select_center_patch_distance",
                        lambda *a: pytest.fail("unused diagnostic distance was recomputed"))
    with caplog.at_level(logging.INFO):
        measurement = runtime.measure_target(bbox, 640, 480, target_id=1, use_latest_depth=True)
    assert measurement.raw_distance_m == pytest.approx(1.8)
    line = next(record.message for record in caplog.records if record.message.startswith("Astra目标深度:"))
    assert f"middle_keep={expected_kept} center_valid={expected_valid}/{expected_pixels}" in line


def test_worker_diagnostics_separate_wall_delay_from_thread_cpu_without_changing_admission(scene, monkeypatch):
    obj, live, distance, clock = scene
    item, target = prepare(scene)
    transaction = item.transaction
    assert not hasattr(transaction.private, "_transaction_trace")
    wall, cpu = [20.], [2.]
    monkeypatch.setattr("car_control_modular.depth_measurement_transaction.time.perf_counter", lambda: wall[0])
    monkeypatch.setattr("car_control_modular.depth_measurement_transaction.time.thread_time", lambda: cpu[0])
    original = transaction.private._region_cluster_candidates

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        wall[0] += .025
        cpu[0] += .002
        return result

    monkeypatch.setattr(transaction.private, "_region_cluster_candidates", delayed)
    assert transaction.run().raw_distance_m == pytest.approx(1.8)
    assert transaction.pixel_scan_started is True
    assert transaction.compute_stage == "complete"
    assert transaction.failure_stage is None
    assert transaction.selected_sample_timestamp == pytest.approx(99.98)
    assert transaction.compute_wall_ms == pytest.approx(125.)
    assert transaction.compute_cpu_ms == pytest.approx(10.)
    torso = [v for k, v in transaction.stage_timings_ms.items() if k.startswith("torso_region:")]
    assert len(torso) == 5
    assert all(v["wall_ms"] == pytest.approx(25.) and v["cpu_ms"] == pytest.approx(2.) for v in torso)
    assert not hasattr(transaction.private, "_transaction_trace")
    assert live._last_accepted_ts == 0
    assert distance.commit_prepared_depth(item, target=target)
    # Diagnostic elapsed time is never the physical freshness clock.
    assert live._last_accepted_ts == pytest.approx(clock[0]-.02)
    assert not hasattr(live, "_transaction_trace")


@pytest.mark.parametrize("failure,stage,started", [
    ("no_history", "history_preflight", False),
    ("expired_before_pixels", "sample_validation", False),
    ("expired_between_regions", "torso_region:abdomen_center", True),
    ("expired_after_regions", "distance_validation", True),
    ("sensor_error", "torso_region:chest_center", True),
])
def test_abort_diagnostics_are_truthful_and_never_leak_trace_or_live_mutation(
        scene, monkeypatch, failure, stage, started):
    obj, live, distance, clock = scene
    if failure == "no_history":
        history(scene, roi_age=.410, sample_age=.01)
    elif failure == "expired_before_pixels":
        live._latest_depth_ts = clock[0]-.181
    item, target = prepare(scene)
    transaction = item.transaction
    private = transaction.private
    state = deepcopy({name: getattr(live, name) for name in RANGING_STATE_FIELDS if hasattr(live, name)})
    if failure in {"expired_between_regions", "sensor_error"}:
        original = private._region_cluster_candidates

        def region(*args, **kwargs):
            if failure == "sensor_error":
                raise ValueError("real array failure")
            result = original(*args, **kwargs)
            clock[0] += .170
            return result

        monkeypatch.setattr(private, "_region_cluster_candidates", region)
    elif failure == "expired_after_regions":
        original = private._select_multiregion_distance

        def regions(*args, **kwargs):
            result = original(*args, **kwargs)
            clock[0] += .170
            return result

        monkeypatch.setattr(private, "_select_multiregion_distance", regions)
    if failure == "sensor_error":
        with pytest.raises(ValueError, match="real array failure"):
            transaction.run()
        assert transaction.reject_reason is None
    else:
        assert transaction.run() is None
        assert transaction.reject_reason == ("history_no_eligible_sample" if failure == "no_history"
                                             else "physical_sample_expired_during_compute")
    assert transaction.pixel_scan_started is started
    assert transaction.failure_stage == transaction.compute_stage == stage
    assert transaction.compute_wall_ms >= 0 and transaction.compute_cpu_ms >= 0
    assert stage in transaction.stage_timings_ms
    assert transaction.selected_sample_timestamp == (None if failure == "no_history" else live._latest_depth_ts)
    assert not hasattr(private, "_transaction_trace")
    assert not hasattr(live, "_transaction_trace")
    assert state == {name: getattr(live, name) for name in state}
    assert not distance.commit_prepared_depth(item, target=target)
