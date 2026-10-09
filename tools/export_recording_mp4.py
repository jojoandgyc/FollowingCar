#!/usr/bin/env python3
"""Export a CLOSED MJPEG recording + CSV as H.264 MP4, on capture time.

No hardware imports. The launcher calls this only after the runtime exits.
AVI/CSV remain unchanged. Failed/cancelled exports never publish a final MP4.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

try:
    from .retime_recording import read_timeline, retime
except ImportError:  # Direct CLI invocation, not imported as tools.*.
    from retime_recording import read_timeline, retime


def _signature(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def export_mp4(video, *, index=None, output=None):
    video = Path(video).resolve()
    index = Path(index).resolve() if index else video.with_suffix(".frames.csv")
    output = Path(output).absolute() if output else video.with_suffix(".mp4")
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must be .mp4")
    if os.path.lexists(output) or output.resolve() in (video, index):
        raise ValueError("output exists or aliases input; refusing overwrite")
    if video.suffix.lower() != ".avi":
        raise ValueError("expected a closed .avi recording")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required; AVI is unchanged")
    times = read_timeline(index)
    before = (_signature(video), _signature(index))
    # Keep temporary output on the same filesystem for atomic, no-clobber
    # publication. It also avoids filling a small RAM-backed /tmp with MP4.
    with tempfile.TemporaryDirectory(prefix=".mp4-export-", dir=output.parent) as directory:
        root = Path(directory)
        timed = retime(video, index=index, output=root / "capture_time.mkv")
        pending = root / "complete.mp4"
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-threads", "1", "-i", str(timed), "-map", "0:v:0", "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-filter_threads", "1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-threads", "1", "-pix_fmt", "yuv420p", "-bf", "0",
            "-vsync", "0", "-enc_time_base", "1:1000",
            "-video_track_timescale", "1000", "-movflags", "+faststart",
            "-n", str(pending),
        ], check=True, timeout=600)
        # Do not label a truncated/mistimed result as a successful export.
        info = json.loads(subprocess.check_output([
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_packets", "-show_entries", "packet=pts_time:stream=codec_name,pix_fmt",
            "-of", "json", str(pending),
        ], timeout=60))
        streams = info.get("streams", [])
        packets = info.get("packets", [])
        if (not streams or streams[0].get("codec_name") != "h264"
                or streams[0].get("pix_fmt") != "yuv420p" or len(packets) != len(times)):
            raise ValueError("MP4 codec/frame count validation failed")
        for packet, stamp in zip(packets, times):
            if abs(float(packet["pts_time"]) - (stamp-times[0])) > .003:
                raise ValueError("MP4 timestamps differ from capture CSV")
        if (_signature(video), _signature(index)) != before:
            raise ValueError("recording changed during export; stop recording first")
        # link is exclusive even if another exporter created output meanwhile.
        os.link(pending, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="closed camera_raw.avi with matching .frames.csv")
    parser.add_argument("--index")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    # This is a separate, post-run process. Never change recorder/control
    # thread priorities, BLAS settings, or OpenCV globals.
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    def cancel(_signum, _frame):
        raise KeyboardInterrupt
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, cancel)
    try:
        print("MP4 export started (post-run, single thread); AVI will be kept", flush=True)
        result = export_mp4(args.video, index=args.index, output=args.output)
        print(f"MP4 export complete: {result}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("MP4 export cancelled; original AVI/CSV kept", file=sys.stderr, flush=True)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"MP4 export skipped/failed: {exc}; original AVI/CSV kept", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
