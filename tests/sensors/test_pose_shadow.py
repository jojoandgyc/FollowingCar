"""Synthetic pairing, geometry and worker isolation tests. No hardware."""
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from car_control_modular.pose_shadow import (
    PoseShadowConfig, PoseShadowObserver, ObservationVideo, analyze_pose, enabled, validate_packet,
)
from car_control_modular.depth_diagnostics import DepthDiagnostics


def packet(stamp=None):
    stamp = time.monotonic() if stamp is None else stamp
    rgb = np.zeros((200, 200, 3), np.uint8)
    depth = np.full((200, 200), 4000, np.uint16)
    depth[40:151, 65:136] = 1500
    metadata = dict(target_id=3, bbox=[20., 20., 180., 180.], frame_size=[200, 200],
                    source="rgb_aligned", evidence_capture_frame_id=10, rgb_capture_id=10,
                    rgb_timestamp=stamp, reference_timestamp=stamp,
                    depth_timestamp=stamp, sample_timestamp=stamp,
                    orientation={"coordinate_space": "external_uvc_unmirrored"},
                    regions=[dict(name="test", roi=[20, 20, 180, 180])],
                    filtered_or_held_m=1.6)
    return rgb, depth, metadata


def pose():
    landmarks = np.zeros((39, 5), np.float32)
    for index, xy in ((11, (70, 45)), (12, (130, 45)), (23, (70, 145)), (24, (130, 145))):
        landmarks[index] = [*xy, 0, .95, .95]
    return [None, landmarks, None, None, None, .95]


def test_pose_polygon_samples_torso_and_does_not_modify_inputs():
    rgb, depth, metadata = packet()
    before = depth.copy()
    result = analyze_pose(rgb, depth, metadata, pose(), PoseShadowConfig())
    assert result["pose_depth"]["median_m"] == 1.5
    assert result["bbox_region_depth"]["median_m"] == 4.0
    assert result["runtime_filtered_or_held_m"] == 1.6
    assert result["center_delta_px"] == 0
    assert result["mode"] == "shadow_only"
    np.testing.assert_array_equal(before, depth)
    assert not rgb.any()


@pytest.mark.parametrize("change,reason", [
    ({"rgb_capture_id": 11}, "capture_mismatch"),
    ({"depth_timestamp": 10.01}, "sample_mismatch"),
    ({"reference_timestamp": 10.01}, "sample_mismatch"),
    ({"depth_timestamp": 10.1, "sample_timestamp": 10.1}, "alignment_gap"),
    ({"orientation": {}}, "unverified_orientation"),
    ({"target_id": None}, "unbound_target"),
    ({"source": "latest"}, "not_rgb_aligned"),
    ({"bbox": [0, 0, float("nan"), 100]}, "invalid_bbox"),
    ({"frame_size": [640, 480]}, "shape_mismatch"),
])
def test_reject_unrelated_or_invalid_observation(change, reason):
    rgb, depth, metadata = packet(10.)
    metadata.update(change)
    assert validate_packet(rgb, depth, metadata, PoseShadowConfig()) == reason


@pytest.mark.parametrize("kind,reason", [
    ("hidden", "low_landmark_confidence"), ("outside", "torso_outside_target"),
    ("thin", "degenerate_torso"), ("empty_depth", "insufficient_pose_depth"),
])
def test_pose_uncertainty_never_generates_fallback_control(kind, reason):
    rgb, depth, metadata = packet()
    p = pose()
    if kind == "hidden": p[1][11, 3] = .1
    if kind == "outside": p[1][11, 0] = 0
    if kind == "thin": p[1][[11, 12, 23, 24], 0] = 100
    if kind == "empty_depth": depth[:] = 0
    result = analyze_pose(rgb, depth, metadata, p, PoseShadowConfig())
    assert result["reason"] == reason
    assert result["pose_depth"] is None or result["pose_depth"]["median_m"] is None
    assert "actions" not in result


def worker_config(tmp_path, **kwargs):
    model = tmp_path / "fake.onnx"
    adapter = tmp_path / "fake.py"
    model.write_bytes(b"test")
    adapter.write_bytes(b"test")
    return replace(PoseShadowConfig(), model=str(model), adapter=str(adapter),
                   interval_sec=0, max_input_age_sec=2, **kwargs)


def test_worker_cannot_block_producer_and_drops_overflow(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Slow:
        def __init__(self, config): pass
        def infer(self, rgb, bbox):
            entered.set()
            assert release.wait(2)
            return pose()
    observer = PoseShadowObserver(tmp_path / "out", config=worker_config(tmp_path), estimator_factory=Slow)
    rgb, depth, metadata = packet()
    try:
        observer.submit(rgb, depth, metadata)
        assert entered.wait(1)
        metadata2 = dict(metadata, rgb_capture_id=11, evidence_capture_frame_id=11)
        observer.submit(rgb, depth, metadata2)
        for n in range(12, 30):
            observer.submit(rgb, depth, dict(metadata, rgb_capture_id=n, evidence_capture_frame_id=n))
        assert observer._queue.qsize() == 1
        assert observer.counts["queue_full"] > 0
    finally:
        release.set()
        observer.close()
        observer._thread.join(2)
    assert observer.counts["written"] == 2
    assert not observer._thread.is_alive()
    rows = [json.loads(s) for s in (tmp_path / "out/observations.jsonl").read_text().splitlines()]
    assert [r["capture_frame_id"] for r in rows] == [10, 11]


def test_default_disabled(monkeypatch):
    monkeypatch.delenv("FOLLOW_POSE_SHADOW_ENABLE", raising=False)
    assert not enabled()


def test_stale_and_duplicate_are_not_queued(tmp_path):
    class Fast:
        def __init__(self, config): pass
        def infer(self, rgb, bbox): return pose()
    observer = PoseShadowObserver(tmp_path / "out", config=worker_config(tmp_path), estimator_factory=Fast)
    rgb, depth, metadata = packet(time.monotonic()-10)
    observer.submit(rgb, depth, metadata)
    assert observer.counts["stale_input"] == 1
    rgb, depth, metadata = packet()
    observer.submit(rgb, depth, metadata)
    observer.submit(rgb, depth, metadata)
    observer.close()
    observer._thread.join(2)
    assert observer.counts["duplicate"] == 1
    assert observer.counts["written"] == 1


def test_observer_failure_is_contained(tmp_path):
    class Broken:
        def __init__(self, config): raise RuntimeError("bad model")
    observer = PoseShadowObserver(tmp_path / "out", config=worker_config(tmp_path), estimator_factory=Broken)
    observer._thread.join(1)
    assert observer.counts["worker_error"] == 1
    assert observer._stop.is_set()
    observer.submit(*packet())
    observer.close()


def test_diagnostic_hook_selects_exact_frames_and_preserves_recording(tmp_path, monkeypatch):
    monkeypatch.delenv("FOLLOW_POSE_SHADOW_ENABLE", raising=False)
    recorder = DepthDiagnostics(tmp_path / "diag", logging.getLogger("test"))
    captured = []
    class Observe:
        def submit(self, rgb, depth, metadata): captured.append((rgb, depth, metadata))
        def close(self): pass
    recorder.pose_shadow = Observe()
    rgb, depth, metadata = packet(10)
    try:
        recorder.add_rgb(10, 10., rgb)
        recorder.add_rgb(11, 10.01, rgb + 1)
        recorder.add_depth(10., depth)
        recorder.add_depth(10.01, depth + 1)
        recorder.observe(metadata, anomaly=False, now=10.02, sample_stamp=10.)
        assert captured[0][2]["rgb_capture_id"] == 10
        assert captured[0][2]["depth_timestamp"] == 10.
        assert captured[0][0].max() == 0
        assert captured[0][1].max() == 4000
        assert validate_packet(*captured[0], PoseShadowConfig()) is None
    finally:
        recorder.close()
    assert (tmp_path / "diag/observations.jsonl").is_file()


def test_existing_output_never_overwritten(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "manifest.json"
    marker.write_text("keep")
    observer = PoseShadowObserver(output, config=worker_config(tmp_path))
    observer._thread.join(1)
    assert marker.read_text() == "keep"
    assert observer.counts["worker_error"] == 1


def test_late_result_is_recorded_but_never_marked_fresh(tmp_path):
    clock = [10.]
    class Late:
        def __init__(self, config): pass
        def infer(self, rgb, bbox):
            clock[0] = 13.
            return pose()
    observer = PoseShadowObserver(tmp_path / "out", config=worker_config(tmp_path),
                                  estimator_factory=Late, clock=lambda: clock[0])
    observer.submit(*packet(10.))
    observer.close()
    observer._thread.join(2)
    row = json.loads((tmp_path / "out/observations.jsonl").read_text())
    assert not row["within_age_budget"]
    assert row["result_age_ms"] == 3000
    assert observer.counts["late_result"] == 1


def test_control_measurement_identical_when_observer_throws(tmp_path, monkeypatch):
    from car_control_modular import pose_shadow, astra_depth
    from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
    class BrokenObserver:
        def __init__(self, *a, **kw): pass
        def submit(self, *a): raise RuntimeError("observation unavailable")
        def close(self): pass
    monkeypatch.setattr(pose_shadow, "enabled", lambda: True)
    monkeypatch.setattr(pose_shadow, "PoseShadowObserver", BrokenObserver)
    monkeypatch.setattr(astra_depth.time, "monotonic", lambda: 100.)
    baseline = AstraDepthRuntime(AstraDepthConfig())
    shadow = AstraDepthRuntime(AstraDepthConfig(diagnostics_dir=str(tmp_path / "diag")))
    try:
        measurements = []
        for sensor in (baseline, shadow):
            sensor._np = np
            sensor._latest_depth = np.full((480, 640), 1500, np.uint16)
            sensor._latest_depth_ts = 99.99
            measurements.append(sensor.measure_target((200., 60., 440., 450.), 640, 480,
                                                       target_id=1, reference_timestamp=99.99,
                                                       evidence_capture_frame_id=10))
        assert measurements[0] == measurements[1]
        assert baseline._last_accepted_distance_m == shadow._last_accepted_distance_m
        assert baseline._last_accepted_ts == shadow._last_accepted_ts
    finally:
        shadow.close()
        baseline.close()


def test_inference_video_contains_success_and_failed_pose_frames(tmp_path):
    import cv2
    from car_control_modular.pose_shadow import render_comparison
    rgb, depth, metadata = packet()
    video = ObservationVideo(tmp_path / "inference.avi")
    try:
        for i, p in enumerate((pose(), None)):
            result = analyze_pose(rgb, depth, metadata, p, PoseShadowConfig())
            assert video.write(render_comparison(rgb, result), result) == i
    finally:
        video.close()
    reader = cv2.VideoCapture(str(video.path))
    try:
        assert reader.isOpened()
        frames = 0
        while True:
            ok, frame = reader.read()
            if not ok: break
            assert frame.shape == rgb.shape
            frames += 1
        assert frames == 2
    finally:
        reader.release()
