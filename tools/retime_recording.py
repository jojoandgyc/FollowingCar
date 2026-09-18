"""Offline only: packet-copy MJPEG AVI to timestamp-correct VFR Matroska.

Run after recording closes. Never overwrite inputs/outputs, never operate
hardware, never interpolate missing camera frames. Gaps remain display holds.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile


def read_timeline(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty frame index")
    times = []
    previous_cap = -1
    for i, row in enumerate(rows):
        cap = int(row["capture_frame_id"])
        stamp = float(row["capture_monotonic_sec"])
        if (int(row["video_frame_index"]) != i or cap <= previous_cap
                or not math.isfinite(stamp) or (times and stamp <= times[-1])):
            raise ValueError("non-contiguous video index or invalid capture timeline")
        times.append(stamp)
        previous_cap = cap
    return times


def retime(video, index=None, output=None):
    video = Path(video).resolve()
    index = Path(index).resolve() if index else video.with_suffix(".frames.csv")
    output = Path(output).resolve() if output else video.with_name(video.stem+".realtime.mkv")
    if output.exists() or output in (video, index):
        raise ValueError("output already exists; original files will not be overwritten")
    if output.suffix.lower() != ".mkv":
        raise ValueError("output must be .mkv (variable frame rate)")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required")
    times = read_timeline(index)
    before = (video.stat().st_size, video.stat().st_mtime_ns, index.stat().st_size, index.stat().st_mtime_ns)
    info = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
        "-of", "json", str(video)], timeout=30))
    if not info.get("streams") or info["streams"][0]["codec_name"] != "mjpeg":
        raise ValueError("only MJPEG recordings are supported; no silent re-encoding")
    with tempfile.TemporaryDirectory(prefix="recording-retime-") as directory:
        root = Path(directory)
        # One packet per AVI with a millisecond timebase. Older board FFmpeg
        # versions do not support concat's per-file "option framerate".
        subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-threads", "1", "-r", "1000",
                        "-i", str(video), "-map", "0:v:0", "-c:v", "copy",
                        "-f", "segment", "-segment_time", "0.001", "-reset_timestamps", "1",
                        "-segment_format", "avi", "-n", str(root/"frame_%08d.avi")],
                       check=True, timeout=300)
        if len(list(root.glob("frame_*.avi"))) != len(times):
            raise ValueError("video frame count differs from CSV; stop recording before retiming")
        after = (video.stat().st_size, video.stat().st_mtime_ns, index.stat().st_size, index.stat().st_mtime_ns)
        if after != before:
            raise ValueError("input changed during export; stop recording first")
        lines = ["ffconcat version 1.0"]
        for i, stamp in enumerate(times):
            lines += [f"file frame_{i:08d}.avi"]
            if i+1 < len(times):
                lines.append(f"duration {times[i+1]-stamp:.9f}")
        manifest = root/"timeline.ffconcat"
        manifest.write_text("\n".join(lines)+"\n", encoding="utf-8")
        subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-threads", "1", "-f", "concat",
                        "-safe", "0", "-i", str(manifest), "-map", "0:v:0", "-c:v", "copy",
                        "-vsync", "vfr", "-n", str(output)], check=True, timeout=300)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="closed camera_raw.avi")
    parser.add_argument("--index", help="defaults to matching .frames.csv")
    parser.add_argument("--output", help="new .mkv; defaults to .realtime.mkv")
    args = parser.parse_args()
    print(retime(args.video, args.index, args.output))
