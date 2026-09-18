"""Diagnostics use synthetic arrays/temp directories; no camera or motors."""
import json
import logging
from pathlib import Path
import sys
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from car_control_modular.depth_diagnostics import DepthDiagnostics
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime


def test_records_uint16_raw_and_timestamped_rgb_before_and_after_failure(tmp_path):
    recorder = DepthDiagnostics(tmp_path, logging.getLogger("test"))
    raw = np.array([[0, 349, 399, 400, 1500, 8000, 65535]], dtype=np.uint16)
    try:
        for i, stamp in enumerate([10., 10.03, 10.06]):
            rgb = np.full((4, 6, 3), i, dtype=np.uint8)
            recorder.add_rgb(100+i, stamp, rgb)
            rgb[:] = 99  # producer mutation must not affect stored RGB
            recorder.add_depth(stamp, raw.copy())
        recorder.observe(dict(target_id=1, reference_timestamp=10.03, detail="insufficient_depth_pixels"),
                         anomaly=True, now=10.08, sample_stamp=10.03)
        recorder.add_depth(10.19, raw.copy())
        recorder.add_depth(10.30, raw.copy())
    finally:
        recorder.close()
    assert not recorder._thread.is_alive()
    files = sorted(tmp_path.glob("*.npz"))
    assert len(files) == 4
    phases = []
    for file in files:
        with np.load(file, allow_pickle=False) as saved:
            assert saved["depth_mm"].dtype == np.uint16
            np.testing.assert_array_equal(saved["depth_mm"], raw)
            metadata = json.loads(saved["metadata_json"].item())
            phases.append(metadata["phase"])
            if "rgb_bgr" in saved:
                assert saved["rgb_bgr"].max() < 99
                assert abs(metadata["rgb_alignment_ms"]) <= 200
            else:
                assert metadata["rgb_capture_id"] is None
    assert phases.count("after") == 2
    records = [json.loads(line) for line in (tmp_path/"observations.jsonl").read_text().splitlines()]
    assert records[0]["rgb_reference_capture_id"] == 101


def test_no_snapshot_for_success_or_duplicate_and_events_are_capped(tmp_path):
    recorder = DepthDiagnostics(tmp_path, logging.getLogger("test"), max_events=1)
    try:
        recorder.add_depth(10., np.ones((4, 4), dtype=np.uint16))
        recorder.observe({}, anomaly=False, now=10., sample_stamp=10.)
        assert recorder._events == 0
        recorder.observe({}, anomaly=True, now=10., sample_stamp=10.)
        recorder.observe({}, anomaly=True, now=13., sample_stamp=10.)
        assert recorder._events == 1
    finally:
        recorder.close()


def test_storage_budget_and_existing_output_are_not_overwritten(tmp_path):
    recorder = DepthDiagnostics(tmp_path, logging.getLogger("test"), max_bytes=300)
    recorder.add_depth(10., np.ones((100, 100), dtype=np.uint16))
    recorder.observe({}, anomaly=True, now=10., sample_stamp=10.)
    recorder.close()
    assert not list(tmp_path.glob("*.npz"))
    assert sum(f.stat().st_size for f in tmp_path.iterdir()) <= 300
    before = (tmp_path/"observations.jsonl").read_bytes()
    again = DepthDiagnostics(tmp_path, logging.getLogger("test"))
    again.close()
    assert (tmp_path/"observations.jsonl").read_bytes() == before


def test_slow_disk_cannot_block_producer_and_queue_is_bounded(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = np.savez
    def slow(*a, **kw):
        entered.set()
        assert release.wait(2)
        return original(*a, **kw)
    monkeypatch.setattr(np, "savez", slow)
    recorder = DepthDiagnostics(tmp_path, logging.getLogger("test"))
    try:
        recorder.add_depth(10., np.ones((4, 4), dtype=np.uint16))
        recorder.observe({}, anomaly=True, now=10., sample_stamp=10.)
        assert entered.wait(1)
        for _ in range(100):
            recorder.observe({}, anomaly=False, now=10.1, sample_stamp=10.)
        assert recorder.dropped > 0
        assert recorder._jobs.qsize() == 8
    finally:
        release.set()
        recorder.close()


def test_measurement_trace_has_all_roi_evidence_without_changing_result(monkeypatch):
    import car_control_modular.astra_depth as module
    monkeypatch.setattr(module.time, "monotonic", lambda: 20.)
    recorded = []
    class Spy:
        def observe(self, metadata, **kwargs):
            recorded.append((metadata, kwargs))
    engine = AstraDepthRuntime(AstraDepthConfig(median_window=1))
    engine._np = np
    engine._latest_depth = np.full((480, 640), 1500, dtype=np.uint16)
    engine._latest_depth_ts = 19.97
    engine.diagnostics = Spy()
    result = engine.measure_target((160., 40., 480., 460.), 640, 480, target_id=1, use_latest_depth=True)
    metadata, flags = recorded[-1]
    assert result.raw_distance_m == result.distance_m == 1.5
    assert not flags["anomaly"]
    assert metadata["accepted_raw_m"] == 1.5
    assert len(metadata["regions"]) >= 2
    for region in metadata["regions"]:
        assert region["valid"] == region["pixels"]
        assert region["nonzero_below_0_4m"] == 0
        assert region["clusters"][0]["distance_m"] == 1.5
        assert len(region["roi"]) == 4
    class Broken:
        def observe(self, *a, **kw):
            raise OSError("disk unavailable")
    engine.diagnostics = Broken()
    engine._latest_depth_ts = 19.99
    assert engine.measure_target((160., 40., 480., 460.), 640, 480, target_id=1,
                                 use_latest_depth=True).distance_m == 1.5
