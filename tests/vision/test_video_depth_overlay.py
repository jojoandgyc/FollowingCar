"""Numeric depth display and slow-recorder regression; no hardware."""
import csv
import json
import logging
from pathlib import Path
import sys
import threading
import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from car_control_modular.depth_diagnostics import DepthDiagnostics
from car_control_modular.video_depth_overlay import (
    DepthVideoSample, DepthVideoView, region_distance, draw_depth_overlay,
)
from car_control_modular.video_recorder import AsyncVideoRecorder, VideoRecorderConfig


def sample(**overrides):
    meta = dict(evidence_capture_frame_id=20, sample_timestamp=10.02,
                target_id=1, depth_size=[320,240],
                regions=[dict(name="chest_center", roi=[50,50,70,70], valid=400,
                              clusters=[dict(distance_m=1.6,pixels=300)])],
                selected_region_distances={"chest_center":1.52},
                selected_regions=["chest_center"], detail="depth_multiregion",
                rejection_reason="", filtered_or_held_m=9.)
    meta.update(overrides)
    return DepthVideoSample(np.full((240,320),1500,np.uint16),meta)


class Drawing:
    FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
    def __init__(self):
        self.lines = []
        self.boxes = []
    def getTextSize(self, *args): return cv2.getTextSize(*args)
    def putText(self, image, text, *args): self.lines.append((text,args[0]))
    def rectangle(self, image, start, end, *args): self.boxes.append((start,end))


@pytest.mark.parametrize("cap,stamp,uid,status", [
    (20,10.,1,"sampled"), (20,10.1,1,"sampled"), (21,10.,1,"sampled"),
    (20,10.,2,"target_mismatch"), (20,9.,1,"unaligned"),
    (20,11.,1,"unaligned"), (20,float("nan"),1,"invalid_time"),
])
def test_time_and_identity_gates_unchanged(cap,stamp,uid,status):
    view = DepthVideoView.from_sample(sample(),cap,stamp,uid)
    assert view.status == status
    assert (view.sample is not None) == (status == "sampled")


def test_selected_region_distance_not_fused_distance():
    evidence = sample()
    assert region_distance(evidence.metadata["regions"][0],evidence.metadata) == (1.52,"selected")


def test_fallback_uses_largest_cluster_not_nearest_or_fused():
    r = dict(name="a",clusters=[dict(distance_m=1.1,pixels=10),dict(distance_m=2.3,pixels=100)])
    assert region_distance(r,{"filtered_or_held_m":9.}) == (2.3,"candidate")


@pytest.mark.parametrize("value",[None,0,-1,float("nan"),float("inf")])
def test_invalid_cluster_is_not_displayed_as_zero(value):
    r = dict(name="a",clusters=[dict(distance_m=value,pixels=100)])
    assert region_distance(r,{}) == (None,"unavailable")


def test_each_roi_shows_own_value_inside_box():
    names = ["a","b","c","d","e"]
    regions = [dict(name=n,roi=[20+i*45,60,60+i*45,120]) for i,n in enumerate(names)]
    evidence = sample(regions=regions,selected_region_distances={n:1+i*.1 for i,n in enumerate(names)})
    drawing = Drawing()
    draw_depth_overlay(np.zeros((480,640,3),np.uint8),
                       DepthVideoView.from_sample(evidence,20,10.),drawing)
    for i in range(5):
        matches = [at for text,at in drawing.lines if text == f"{1+i*.1:.2f}m"]
        assert len(matches)==1
        x,y=matches[0]
        assert (20+i*45)*2 <= x < (60+i*45)*2 and 120 <= y < 240
    assert not any("40%" in text or ">=5m" in text for text,_ in drawing.lines)


def test_no_cluster_displays_na_and_rejection():
    evidence = sample(selected_region_distances={},
                      regions=[dict(name="a",roi=[50,50,80,90],clusters=[])],
                      rejection_reason="insufficient_depth_pixels")
    drawing = Drawing()
    draw_depth_overlay(np.zeros((480,640,3),np.uint8),
                       DepthVideoView.from_sample(evidence,20,10.),drawing)
    assert "n/a" in [s for s,_ in drawing.lines]
    assert "insufficient_depth_pixels" in [s for s,_ in drawing.lines]


def test_no_tint_or_modification_of_source_depth():
    evidence = sample()
    original = evidence.depth.copy()
    image = np.full((480,640,3),100,np.uint8)
    draw_depth_overlay(image,DepthVideoView.from_sample(evidence,20,10.),cv2)
    assert (image[130:139,120:139]==100).all()  # Interior, below labels.
    assert (image[200:300,20:90]==100).all()
    np.testing.assert_array_equal(evidence.depth,original)


def test_scalar_only_snapshot_and_csv():
    evidence = DepthVideoSample(None,sample().metadata)
    view = DepthVideoView.from_sample(evidence,20,10.)
    drawing = Drawing()
    draw_depth_overlay(np.zeros((480,640,3),np.uint8),view,drawing)
    assert "1.52m" in [s for s,_ in drawing.lines]
    assert json.loads(view.csv_values()[-1]) == [
        dict(name="chest_center",distance_m=1.52,kind="selected",valid=400)]


def test_stale_view_never_draws_old_numeric_values():
    view = DepthVideoView.from_sample(sample(),20,11.)
    drawing = Drawing()
    draw_depth_overlay(np.zeros((480,640,3),np.uint8),view,drawing)
    assert not drawing.boxes
    assert not any("1.52m" in s for s,_ in drawing.lines)


def test_max_five_regions():
    evidence=sample(regions=[dict(name="a",roi=[20,20,50,50],clusters=[])]*6)
    drawing=Drawing()
    draw_depth_overlay(np.zeros((480,640,3),np.uint8),
                       DepthVideoView.from_sample(evidence,20,10.),drawing)
    assert sum(s=="n/a" for s,_ in drawing.lines)==5


@pytest.mark.parametrize("roi",[[320,240,330,250],[-10,-10,-2,-2],[50,50,20,20],[1,1,1,2]])
def test_outside_empty_regions_are_skipped(roi):
    evidence=sample(regions=[dict(name="a",roi=roi,clusters=[])])
    drawing=Drawing()
    draw_depth_overlay(np.zeros((480,640,3),np.uint8),
                       DepthVideoView.from_sample(evidence,20,10.),drawing)
    assert not drawing.boxes


def test_diagnostic_ring_trylock_source_frame_and_nearest_time(tmp_path):
    diag = DepthDiagnostics(tmp_path, logging.getLogger("test"), max_events=0)
    try:
        for i in range(600):
            evidence = sample(sample_timestamp=10+i*.01)
            diag.add_depth(10+i*.01, evidence.depth)
            diag.observe(evidence.metadata, anomaly=False, now=10.4, sample_stamp=10+i*.01)
        assert len(diag._video_samples) == 512
        result = diag.video_sample(20,11.2)
        assert result.metadata["sample_timestamp"] == pytest.approx(11.2)
        assert diag.video_sample(21,11.2).metadata["evidence_capture_frame_id"] == 20
        assert all(s.depth is None for s in diag._video_samples)
        assert diag.video_sample(21,20.) is None
        with diag._lock:
            assert diag.video_sample(20,11.2) is None
    finally:
        diag.close()


def test_provider_runs_in_writer_and_csv_keeps_alignment(tmp_path):
    calls = []
    def provider(cap, stamp):
        calls.append(threading.current_thread().name)
        return sample(evidence_capture_frame_id=cap) if cap == 20 else None
    recorder = AsyncVideoRecorder(VideoRecorderConfig(str(tmp_path/"depth.avi"),30,
                                  overlay_wait_sec=0), cv2_module=cv2,
                                  depth_sample_provider=provider)
    image = np.zeros((480,640,3),np.uint8)
    for cap in (20,21):
        assert recorder.submit(image,capture_frame_id=cap,monotonic_sec=10.)
    recorder.close(timeout_sec=3)
    assert recorder.error is None and recorder.written_frames == 2
    assert calls == ["camera-video-recorder"]*2
    assert not image.any()
    with Path(recorder.index_path).open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["depth_overlay_status"] == "sampled"
    assert rows[0]["depth_overlay_skew_ms"] == "20.0"
    assert rows[0]["depth_overlay_roi_capture_id"] == "20"
    assert json.loads(rows[0]["depth_overlay_regions_json"])[0]["distance_m"] == 1.52
    assert rows[1]["depth_overlay_status"] == "no_sample"


def test_record_numeric_evidence_even_if_raw_array_already_evicted(tmp_path):
    diag = DepthDiagnostics(tmp_path, logging.getLogger("test"), max_events=0)
    try:
        evidence = sample()
        # No raw depth in the 8-frame ring: statistics and dimensions suffice.
        diag.observe(evidence.metadata, anomaly=False, now=10.1, sample_stamp=10.02)
        result = diag.video_sample(20,10.)
        assert result is not None and result.depth is None
        assert region_distance(result.metadata["regions"][0],result.metadata) == (1.52,"selected")
    finally:
        diag.close()


def test_provider_failure_does_not_disable_video(tmp_path):
    def broken(*args): raise ValueError("test")
    recorder = AsyncVideoRecorder(VideoRecorderConfig(str(tmp_path/"broken.avi"),30,
                                  overlay_wait_sec=0), cv2_module=cv2,
                                  depth_sample_provider=broken)
    recorder.submit(np.zeros((480,640,3),np.uint8),capture_frame_id=1,monotonic_sec=10.)
    recorder.close(timeout_sec=3)
    assert recorder.error is None and recorder.written_frames == 1
