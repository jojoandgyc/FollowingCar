"""Offline recorder/packet mux tests only. No camera or motor devices."""
import csv
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from car_control_modular.fast_mjpeg_writer import FastMjpegWriter
from car_control_modular.video_recorder import AsyncVideoRecorder, VideoRecorderConfig, VideoFrameOverlay
from tools.retime_recording import read_timeline, retime


@pytest.mark.skipif(not shutil.which("ffmpeg"),reason="ffmpeg unavailable")
def test_jpeg_packet_writer_roundtrip_order_and_close(tmp_path):
    path=tmp_path/"test.avi"
    writer=FastMjpegWriter(path,30,(160,120),cv2)
    image=np.zeros((120,160,3),np.uint8)
    for level in [30,90,150]:
        image[:]=level
        writer.write(image)
    with pytest.raises(ValueError): writer.write(np.zeros((10,10,3),np.uint8))
    writer.release();writer.release()
    capture=cv2.VideoCapture(str(path));levels=[]
    while True:
        ok,frame=capture.read()
        if not ok:break
        levels.append(float(frame.mean()))
    capture.release()
    assert levels==pytest.approx([30,90,150],abs=3)
    assert not writer.isOpened()


@pytest.mark.parametrize("fast",[True,False])
def test_csv_stage_timing_capture_gaps_and_global_threads_unchanged(tmp_path,fast,caplog):
    count=cv2.getNumThreads()
    recorder=AsyncVideoRecorder(VideoRecorderConfig(str(tmp_path/"timing.avi"),30,
        overlay_wait_sec=0,fast_mjpeg=fast),cv2_module=cv2)
    caplog.set_level(logging.INFO)
    image=np.zeros((120,160,3),np.uint8)
    for cap,stamp in [(318,100.),(321,100.164),(322,100.200)]:
        assert recorder.submit(image,capture_frame_id=cap,monotonic_sec=stamp)
    assert recorder.close(timeout_sec=5)
    with Path(recorder.index_path).open() as stream: rows=list(csv.DictReader(stream))
    assert [r["recording_capture_gap"] for r in rows]==["0","2","0"]
    assert [float(r["recording_time_sec"]) for r in rows]==pytest.approx([0,.164,.2])
    for row in rows:
        for key in ["wait","sharpness","prepare","draw","encode_write"]:
            assert float(row[f"recording_{key}_ms"])>=0
    assert rows[0]["recording_backend"]==("jpeg_packet_copy" if fast and shutil.which("ffmpeg") else "opencv")
    assert cv2.getNumThreads()==count
    assert "Camera recording timing" in caplog.text
    assert not image.any()


def test_missing_ffmpeg_falls_back_without_control_side_effects(tmp_path,monkeypatch):
    monkeypatch.setattr(shutil,"which",lambda _:None)
    recorder=AsyncVideoRecorder(VideoRecorderConfig(str(tmp_path/"fallback.avi"),30,
        overlay_wait_sec=0),cv2_module=cv2)
    recorder.submit(np.zeros((120,160,3),np.uint8),capture_frame_id=1,monotonic_sec=100.)
    assert recorder.close(timeout_sec=5)
    with Path(recorder.index_path).open() as stream: row=next(csv.DictReader(stream))
    assert row["recording_backend"]=="opencv"


def test_full_queue_does_not_copy_rgb_or_block_producer(tmp_path):
    entered,release=threading.Event(),threading.Event()
    class Gated(AsyncVideoRecorder):
        def _measure_sharpness(self,image):
            entered.set()
            assert release.wait(3)
            return None
    class NoCopy:
        def copy(self): raise AssertionError("full queue should not copy RGB")
    recorder=Gated(VideoRecorderConfig(str(tmp_path/"queue.avi"),30,queue_capacity=1,
        overlay_wait_sec=0),cv2_module=cv2)
    try:
        image=np.zeros((120,160,3),np.uint8)
        assert recorder.submit(image,capture_frame_id=1)
        assert entered.wait(2)
        assert recorder.submit(image,capture_frame_id=2)
        assert not recorder.submit(NoCopy(),capture_frame_id=3)
        assert recorder.dropped_frames==1
    finally:
        release.set()
        assert recorder.close(timeout_sec=5)


def write_index(path,times):
    with path.open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["video_frame_index","capture_frame_id","capture_monotonic_sec"])
        for i,stamp in enumerate(times):writer.writerow([i,1+i*2,stamp])


@pytest.mark.parametrize("times",[[],[1,1],[2,1],[1,float("nan")]])
def test_retime_rejects_invalid_timeline(tmp_path,times):
    path=tmp_path/"bad.csv";write_index(path,times)
    with pytest.raises(ValueError):read_timeline(path)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),reason="ffmpeg tools unavailable")
def test_offline_vfr_retains_capture_timing_without_duplicate_frames(tmp_path):
    video=tmp_path/"camera_raw.avi"
    writer=FastMjpegWriter(video,30,(160,120),cv2)
    for level in [30,90,150]:writer.write(np.full((120,160,3),level,np.uint8))
    writer.release()
    write_index(video.with_suffix(".frames.csv"),[100.,100.1,100.4])
    before=video.read_bytes()
    output=retime(video)
    info=json.loads(subprocess.check_output([
        shutil.which("ffprobe"),"-v","error","-select_streams","v:0","-show_packets",
        "-show_entries","packet=pts_time","-of","json",str(output)],timeout=10))
    stamps=[float(p["pts_time"]) for p in info["packets"]]
    assert stamps==pytest.approx([0,.1,.4],abs=.002)
    assert video.read_bytes()==before
    with pytest.raises(ValueError):retime(video)


def test_elapsed_time_label_uses_capture_clock():
    recorder=AsyncVideoRecorder(VideoRecorderConfig("unused.avi",30),cv2_module=cv2)
    lines=[]
    recorder._draw_text_box=lambda image,text,**kw:lines.append(text)
    recorder._annotate_frame(np.zeros((480,640,3),np.uint8),254,321,VideoFrameOverlay(),
                             capture_elapsed_sec=17.164)
    assert any("T+17.164s" in line and "000321" in line for line in lines)
