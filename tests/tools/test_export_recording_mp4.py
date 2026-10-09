"""Offline conversion only. No camera, motor, or real runtime processes."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import export_recording_mp4 as exporter
from car_control_modular.fast_mjpeg_writer import FastMjpegWriter


def write_index(path, times):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_frame_index", "capture_frame_id", "capture_monotonic_sec"])
        for i, stamp in enumerate(times):
            writer.writerow([i, i*3+1, stamp])


@pytest.fixture
def recording(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools unavailable")
    video = tmp_path / "camera_raw.avi"
    writer = FastMjpegWriter(video,30,(160,120),cv2)
    try:
        for level in [30,90,150]:
            writer.write(np.full((120,160,3), level, np.uint8))
    finally:
        writer.release()
    write_index(video.with_suffix(".frames.csv"), [100.,100.123,101.457])
    return video


def test_export_h264_preserves_irregular_capture_time_and_originals(recording):
    avi = recording.read_bytes()
    index = recording.with_suffix(".frames.csv")
    csv_bytes = index.read_bytes()
    output = exporter.export_mp4(recording)
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts_time:stream=codec_name,pix_fmt", "-of", "json", str(output)
    ], timeout=10))
    assert data["streams"][0]["codec_name"] == "h264"
    assert data["streams"][0]["pix_fmt"] == "yuv420p"
    assert [float(p["pts_time"]) for p in data["packets"]] == pytest.approx([0,.123,1.457],abs=.002)
    assert output.name == "camera_raw.mp4"
    capture = cv2.VideoCapture(str(output))
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(float(image.mean()))
    capture.release()
    assert frames == pytest.approx([30,90,150],abs=5)
    assert recording.read_bytes() == avi and index.read_bytes() == csv_bytes
    assert not list(recording.parent.glob(".mp4-export-*"))
    with pytest.raises(ValueError, match="overwrite"):
        exporter.export_mp4(recording)


@pytest.mark.parametrize("times", [[100.,100.1], [100.,100.1,100.2,100.3], [100.,100.,100.2]])
def test_inconsistent_or_duplicate_timeline_never_publishes(recording,times):
    write_index(recording.with_suffix(".frames.csv"),times)
    with pytest.raises(ValueError):
        exporter.export_mp4(recording)
    assert not recording.with_suffix(".mp4").exists()
    assert not list(recording.parent.glob(".mp4-export-*"))
    assert recording.is_file()


def test_missing_encoder_tools_leave_original(recording,monkeypatch):
    monkeypatch.setattr(exporter.shutil,"which",lambda _: None)
    with pytest.raises(RuntimeError, match="required"):
        exporter.export_mp4(recording)
    assert recording.exists() and not recording.with_suffix(".mp4").exists()


@pytest.mark.parametrize("error",[subprocess.CalledProcessError(1,["ffmpeg"]), KeyboardInterrupt()])
def test_failed_or_cancelled_encode_cleans_partial_output(recording,monkeypatch,error):
    original_run = subprocess.run
    def fail_encode(command,**kwargs):
        if "libx264" in command:
            Path(command[-1]).write_bytes(b"partial")
            raise error
        return original_run(command,**kwargs)
    monkeypatch.setattr(exporter.subprocess,"run",fail_encode)
    with pytest.raises(type(error)):
        exporter.export_mp4(recording)
    assert not recording.with_suffix(".mp4").exists()
    assert not list(recording.parent.glob(".mp4-export-*"))


def test_broken_output_symlink_is_not_overwritten(recording):
    output = recording.with_suffix(".mp4")
    output.symlink_to(recording.parent/"missing_target")
    with pytest.raises(ValueError, match="overwrite"):
        exporter.export_mp4(recording)
    assert output.is_symlink() and not output.exists()


def test_reject_non_mp4_output(recording):
    with pytest.raises(ValueError, match=".mp4"):
        exporter.export_mp4(recording,output=recording.parent/"other.avi")
