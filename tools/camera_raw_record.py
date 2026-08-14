#!/usr/bin/env python3
"""Record a V4L2 camera stream with per-frame timing logs.

This is a pure camera pull test based on the board-side camera_record_20s.py
shape. It does not load RKNN models or start an inference worker, so the output
FPS reflects camera capture plus optional video writing overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rk_vision.gstreamer_capture import GstAppsrcH264Writer, GstAppsrcH264WriterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record raw V4L2 camera frames with timing logs.")
    parser.add_argument("--device", default="/dev/video1", help="Video device path, e.g. /dev/video1.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG", help="Camera pixel fourcc, e.g. MJPG or YUYV.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to record; 0 means until Ctrl+C.")
    parser.add_argument("--output-dir", default=".test_outputs/camera_raw_record")
    parser.add_argument("--name", default="", help="Output basename. Defaults to camera_raw_YYYYmmdd_HHMMSS.")
    parser.add_argument("--output", default="", help="Override output MP4 path.")
    parser.add_argument("--log-jsonl", default="", help="Override per-frame timing JSONL path.")
    parser.add_argument("--summary-json", default="", help="Override summary JSON path.")
    parser.add_argument("--writer-fps", type=float, default=0.0, help="Video writer FPS; 0 uses requested camera FPS.")
    parser.add_argument(
        "--writer-mode",
        default="opencv_mp4v",
        choices=("opencv_mp4v", "opencv_mjpg_avi", "gstreamer_mpp_h264", "gstreamer_v4l2_h264"),
        help="Video writer implementation. gstreamer_* uses board hardware codecs when available.",
    )
    parser.add_argument(
        "--gst-pipeline",
        default="",
        help="Custom GStreamer writer pipeline with {path}, {width}, {height}, {fps_num}, {fps_den}.",
    )
    parser.add_argument("--no-video", action="store_true", help="Do not write MP4; measure read/display/log only.")
    parser.add_argument("--display", action="store_true", help="Show camera preview and allow q to stop.")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Valid frames to discard before measurement.")
    parser.add_argument("--progress-interval", type=float, default=1.0, help="Seconds between console FPS reports.")
    parser.add_argument("--frame-log-every", type=int, default=1, help="Write one JSONL row every N frames.")
    parser.add_argument("--flush-log-every", type=int, default=30, help="Flush timing log every N logged rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.duration < 0:
        raise ValueError("--duration must be >= 0")
    if args.writer_fps < 0:
        raise ValueError("--writer-fps must be >= 0")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be > 0")
    if args.frame_log_every <= 0:
        raise ValueError("--frame-log-every must be > 0")
    if args.flush_log_every <= 0:
        raise ValueError("--flush-log-every must be > 0")

    try:
        import cv2
    except Exception as exc:
        raise SystemExit(f"OpenCV is required: {exc}") from exc

    paths = make_output_paths(args)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    stop = {"value": False, "signals": 0}
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)

    def request_stop(signum, _frame) -> None:
        stop["signals"] += 1
        if stop["signals"] == 1:
            print(f"\n[INFO] signal {signum} received; finalizing raw recording...")
            stop["value"] = True
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cap = None
    writer = None
    log_file = None
    frame_count = 0
    write_count = 0
    logged_rows = 0
    read_failures = 0
    interrupted = False
    read_ms_values: List[float] = []
    write_ms_values: List[float] = []
    interval_ms_values: List[float] = []
    actual: Dict[str, Any] = {}
    output_size: Optional[Tuple[int, int]] = None
    start_wall = time.time()
    start_perf = time.perf_counter()
    last_capture_perf: Optional[float] = None
    next_report = start_perf + float(args.progress_interval)

    try:
        print(f"[INFO] device       : {args.device}")
        print(f"[INFO] request      : {args.width}x{args.height} @ {args.fps:.2f}fps ({args.fourcc})")
        cap = open_camera(cv2, args.device, args.width, args.height, args.fps, args.fourcc)
        actual = camera_actual(cv2, cap)
        print(
            "[INFO] negotiated  : "
            f"{actual['width']}x{actual['height']} @ {actual['fps']:.2f}fps fourcc={actual['fourcc']!r}"
        )

        warmup_ok = warmup_camera(cap, args.warmup_frames)
        if not warmup_ok:
            print("[WARN] camera read failed during warmup.")
            return 2

        log_file = paths["log_jsonl"].open("w", encoding="utf-8")
        if not args.no_video:
            print(f"[INFO] video        : {paths['output']}")
        else:
            print("[INFO] video        : disabled (--no-video)")
        print(f"[INFO] log jsonl    : {paths['log_jsonl']}")
        print(f"[INFO] summary      : {paths['summary_json']}")
        print("[INFO] stop         : Ctrl+C" + (" or q in preview" if args.display else ""))

        start_wall = time.time()
        start_perf = time.perf_counter()
        next_report = start_perf + float(args.progress_interval)

        while not stop["value"]:
            read_start = time.perf_counter()
            ok, frame = cap.read()
            read_end = time.perf_counter()
            read_ms = elapsed_ms(read_start, read_end)
            if not ok or frame is None:
                read_failures += 1
                print("[WARN] camera read failed mid-stream.")
                break

            capture_perf = read_end
            frame_count += 1
            read_ms_values.append(read_ms)
            interval_ms = None if last_capture_perf is None else elapsed_ms(last_capture_perf, capture_perf)
            if interval_ms is not None:
                interval_ms_values.append(interval_ms)
            last_capture_perf = capture_perf

            if writer is None and not args.no_video:
                height, width = int(frame.shape[0]), int(frame.shape[1])
                output_size = (width, height)
                writer = create_writer(
                    cv2,
                    paths["output"],
                    float(args.writer_fps or args.fps),
                    width,
                    height,
                    args.writer_mode,
                    args.gst_pipeline,
                )

            write_ms = 0.0
            if writer is not None:
                write_start = time.perf_counter()
                writer.write(frame)
                write_end = time.perf_counter()
                write_ms = elapsed_ms(write_start, write_end)
                write_ms_values.append(write_ms)
                write_count += 1

            if args.display:
                cv2.imshow("camera_raw_record", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("[INFO] stopped by user (q).")
                    break

            elapsed = max(capture_perf - start_perf, 1e-9)
            if frame_count % args.frame_log_every == 0:
                item = {
                    "frame_index": frame_count,
                    "elapsed_sec": round(elapsed, 6),
                    "wall_ts": round(time.time(), 6),
                    "read_ms": round(read_ms, 3),
                    "write_ms": round(write_ms, 3),
                    "interval_ms": None if interval_ms is None else round(interval_ms, 3),
                    "instant_fps": None if not interval_ms or interval_ms <= 0 else round(1000.0 / interval_ms, 3),
                    "avg_fps": round(frame_count / elapsed, 3),
                    "wrote_video": writer is not None,
                }
                log_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                logged_rows += 1
                if logged_rows % args.flush_log_every == 0:
                    log_file.flush()

            now = time.perf_counter()
            if now >= next_report:
                print(
                    "[INFO] progress    : "
                    f"frames={frame_count} elapsed={elapsed:.2f}s avg_fps={frame_count / elapsed:.2f} "
                    f"last_read_ms={read_ms:.2f} last_write_ms={write_ms:.2f}"
                )
                next_report = now + float(args.progress_interval)

            if args.duration > 0 and elapsed >= args.duration:
                print("[INFO] duration reached.")
                break

    except KeyboardInterrupt:
        interrupted = True
        stop["value"] = True
        print("\n[INFO] interrupted; finalizing raw recording...")
    finally:
        if log_file is not None:
            log_file.flush()
            log_file.close()
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    elapsed_total = max(time.perf_counter() - start_perf, 1e-9)
    summary = {
        "output": "" if args.no_video else str(paths["output"]),
        "log_jsonl": str(paths["log_jsonl"]),
        "summary_json": str(paths["summary_json"]),
        "device": args.device,
        "request": {"width": args.width, "height": args.height, "fps": args.fps, "fourcc": args.fourcc},
        "negotiated": actual,
        "writer_fps": 0.0 if args.no_video else float(args.writer_fps or args.fps),
        "writer_mode": "none" if args.no_video else args.writer_mode,
        "output_size": None if output_size is None else {"width": output_size[0], "height": output_size[1]},
        "duration_sec": round(elapsed_total, 3),
        "interrupted": bool(interrupted or stop["signals"] > 0),
        "captured_frames": int(frame_count),
        "video_frames_written": int(write_count),
        "read_failures": int(read_failures),
        "avg_capture_fps": round(frame_count / elapsed_total, 3),
        "avg_video_write_fps": round(write_count / elapsed_total, 3),
        "read_ms": summarize(read_ms_values),
        "write_ms": summarize(write_ms_values),
        "interval_ms": summarize(interval_ms_values),
    }
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[INFO] saved video  :", summary["output"] or "(disabled)")
    print("[INFO] saved log    :", paths["log_jsonl"])
    print("[INFO] saved summary:", paths["summary_json"])
    print("[INFO] frames       :", frame_count)
    print("[INFO] elapsed      :", f"{elapsed_total:.2f}s")
    print("[INFO] avg fps      :", f"{frame_count / elapsed_total:.2f}")
    return 0


def make_output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.name.strip() or f"camera_raw_{timestamp}"
    default_ext = ".avi" if args.writer_mode == "opencv_mjpg_avi" else ".mp4"
    return {
        "output_dir": output_dir,
        "output": Path(args.output) if args.output else output_dir / f"{base}{default_ext}",
        "log_jsonl": Path(args.log_jsonl) if args.log_jsonl else output_dir / f"{base}_frames.jsonl",
        "summary_json": Path(args.summary_json) if args.summary_json else output_dir / f"{base}_summary.json",
    }


def open_camera(cv2, device: str, width: int, height: int, fps: float, fourcc: str):
    cap = cv2.VideoCapture(camera_device_arg(device), cv2.CAP_V4L2)
    if not cap.isOpened():
        hint = (
            "If Ctrl+Z was used earlier, a stopped job may still hold the device. "
            "Run `jobs -l` then `kill %N`, or `fg` and quit with Ctrl+C/q."
        )
        raise RuntimeError(f"Failed to open camera: {device}\n{hint}")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_cv2(cv2, fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def camera_device_arg(device: str):
    value = str(device).strip()
    if value.isdigit():
        return int(value)
    match = re.fullmatch(r"/dev/video(\d+)", value)
    if match:
        return int(match.group(1))
    return device


def fourcc_cv2(cv2, code: str) -> int:
    c = code.strip().upper()
    if len(c) != 4:
        raise ValueError("--fourcc must be exactly 4 ASCII chars, e.g. MJPG or YUYV")
    return cv2.VideoWriter_fourcc(*c)


def camera_actual(cv2, cap) -> Dict[str, Any]:
    fc = int(cap.get(cv2.CAP_PROP_FOURCC))
    tag = bytes(
        [(fc >> 0) & 0xFF, (fc >> 8) & 0xFF, (fc >> 16) & 0xFF, (fc >> 24) & 0xFF]
    ).decode("ascii", errors="replace")
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "fourcc": tag,
    }


def warmup_camera(cap, warmup_frames: int) -> bool:
    if warmup_frames <= 0:
        return True
    ok_any = False
    for _ in range(warmup_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            ok_any = True
        else:
            time.sleep(0.05)
    return ok_any


def create_writer(
    cv2,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    writer_mode: str,
    gst_pipeline: str = "",
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if writer_mode == "opencv_mp4v":
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    elif writer_mode == "opencv_mjpg_avi":
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (width, height))
    elif writer_mode == "gstreamer_mpp_h264":
        writer = GstAppsrcH264Writer(
            GstAppsrcH264WriterConfig(
                output_path=str(output_path),
                width=int(width),
                height=int(height),
                fps=float(fps),
                encoder="mpph264enc",
            )
        )
        writer.open()
        return writer
    elif writer_mode.startswith("gstreamer_"):
        pipeline = format_gst_pipeline(gst_pipeline, output_path, fps, width, height) if gst_pipeline else (
            default_gst_writer_pipeline(writer_mode, output_path, fps, width, height)
        )
        writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, float(fps), (width, height), True)
        if not writer.isOpened():
            raise RuntimeError(
                "Failed to open GStreamer video writer. Check OpenCV GStreamer support and plugins with "
                "`gst-inspect-1.0 mpph264enc` or `gst-inspect-1.0 v4l2h264enc`.\n"
                f"Pipeline: {pipeline}"
            )
        return writer
    else:
        raise ValueError(f"unknown writer mode: {writer_mode!r}")
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path} ({writer_mode})")
    return writer


def default_gst_writer_pipeline(writer_mode: str, output_path: Path, fps: float, width: int, height: int) -> str:
    values = gst_pipeline_values(output_path, fps, width, height)
    encoder = "mpph264enc" if writer_mode == "gstreamer_mpp_h264" else "v4l2h264enc"
    return (
        "appsrc is-live=true block=false format=time "
        f"caps=video/x-raw,format=BGR,width={values['width']},height={values['height']},"
        f"framerate={values['fps_num']}/{values['fps_den']} "
        "! queue leaky=downstream max-size-buffers=4 "
        "! videoconvert "
        "! video/x-raw,format=NV12 "
        f"! {encoder} "
        "! h264parse "
        "! mp4mux "
        f'! filesink location="{values["path"]}"'
    )


def format_gst_pipeline(template: str, output_path: Path, fps: float, width: int, height: int) -> str:
    return template.format(**gst_pipeline_values(output_path, fps, width, height))


def gst_pipeline_values(output_path: Path, fps: float, width: int, height: int) -> Dict[str, Any]:
    frac = Fraction(float(fps)).limit_denominator(1001)
    location = str(output_path).replace("\\", "\\\\").replace('"', '\\"')
    return {
        "path": location,
        "width": int(width),
        "height": int(height),
        "fps_num": int(frac.numerator),
        "fps_den": int(frac.denominator),
    }


def elapsed_ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "avg": None, "min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(float(v) for v in values)
    return {
        "count": len(ordered),
        "avg": round(sum(ordered) / len(ordered), 3),
        "min": round(ordered[0], 3),
        "p50": round(percentile_sorted(ordered, 0.50), 3),
        "p95": round(percentile_sorted(ordered, 0.95), 3),
        "max": round(ordered[-1], 3),
    }


def percentile_sorted(values: List[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    pos = max(0.0, min(1.0, pct)) * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


if __name__ == "__main__":
    raise SystemExit(main())
