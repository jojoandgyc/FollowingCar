#!/usr/bin/env python3
"""Record a V4L2 camera while running the RKNN person tracker.

This is intentionally based on the board-side camera_record_20s.py pattern:
open /dev/video* with V4L2, request MJPEG, write MP4, and exit cleanly on
Ctrl+C.  The capture loop writes the raw camera video continuously, while a
worker thread runs YOLO/ReID/DeepSORT on the latest frame and writes an
annotated track video plus per-frame JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from car_control_modular.config_loader import load_config_to_env
from rk_vision.gstreamer_capture import (
    GstAppsrcH264Writer,
    GstAppsrcH264WriterConfig,
    GstMjpegTeeCapture,
    GstMjpegTeeConfig,
)
from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline


def parse_args() -> argparse.Namespace:
    default_config = ROOT / "car_control_modular" / "config" / "reid_runtime.ini"
    parser = argparse.ArgumentParser(
        description="Record raw camera video and an RKNN tracked overlay video until Ctrl+C.",
    )
    parser.add_argument("--config", default=str(default_config) if default_config.exists() else "")
    parser.add_argument("--device", default="/dev/video1", help="Video device path, e.g. /dev/video1.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG", help="Camera pixel fourcc, e.g. MJPG or YUYV.")
    parser.add_argument(
        "--capture-mode",
        default="opencv_v4l2",
        choices=("opencv_v4l2", "gstreamer_mjpeg_tee"),
        help=(
            "Camera capture path. gstreamer_mjpeg_tee opens the camera once, stores the original "
            "MJPEG stream, and sends decoded BGR frames to Python."
        ),
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to record; 0 means until Ctrl+C.")
    parser.add_argument("--output-dir", default=".test_outputs/camera_track_record")
    parser.add_argument("--name", default="", help="Output basename. Defaults to camera_track_YYYYmmdd_HHMMSS.")
    parser.add_argument("--raw-output", default="", help="Override raw MP4 output path.")
    parser.add_argument("--track-output", default="", help="Override tracked MP4 output path.")
    parser.add_argument("--jsonl", default="", help="Override per-inference JSONL output path.")
    parser.add_argument("--summary-json", default="", help="Override summary JSON output path.")
    parser.add_argument("--raw-video-fps", type=float, default=0.0, help="Writer FPS for raw video; 0 uses camera FPS.")
    parser.add_argument("--track-video-fps", type=float, default=5.0, help="Writer FPS for annotated track video.")
    parser.add_argument(
        "--raw-writer-mode",
        default="opencv_mp4v",
        choices=(
            "opencv_mp4v",
            "opencv_mjpg_avi",
            "gstreamer_mjpeg_passthrough",
            "gstreamer_mpp_h264",
            "gstreamer_v4l2_h264",
        ),
        help=(
            "Raw video writer. gstreamer_mjpeg_passthrough is used with --capture-mode "
            "gstreamer_mjpeg_tee and does not re-encode frames."
        ),
    )
    parser.add_argument(
        "--track-writer-mode",
        default="opencv_mp4v",
        choices=("opencv_mp4v", "opencv_mjpg_avi", "gstreamer_mpp_h264", "gstreamer_v4l2_h264"),
        help="Annotated track video writer. gstreamer_* uses board hardware codecs when available.",
    )
    parser.add_argument(
        "--raw-gst-pipeline",
        default="",
        help="Custom raw GStreamer writer pipeline with {path}, {width}, {height}, {fps_num}, {fps_den}.",
    )
    parser.add_argument(
        "--track-gst-pipeline",
        default="",
        help="Custom track GStreamer writer pipeline with {path}, {width}, {height}, {fps_num}, {fps_den}.",
    )
    parser.add_argument("--raw-write-every", type=int, default=1, help="Write every Nth captured raw frame.")
    parser.add_argument("--track-publish-every", type=int, default=1, help="Offer every Nth captured frame to tracker.")
    parser.add_argument("--inference-max-fps", type=float, default=0.0, help="Limit tracker worker FPS; 0 means unlimited.")
    parser.add_argument("--display", action="store_true", help="Show raw camera preview and allow q to stop.")
    parser.add_argument("--draw-detections", action="store_true", default=True)
    parser.add_argument("--no-draw-detections", action="store_false", dest="draw_detections")
    parser.add_argument("--debug-tracker-state", action="store_true")

    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--reid-model", default="")
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--backend", default="")
    parser.add_argument("--core-mask", default="")
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--max-output-age", type=int, default=None)
    parser.add_argument("--bbox-expand-scale", type=float, default=None)
    parser.add_argument("--identity-match-threshold", type=float, default=None)
    parser.add_argument("--identity-update-interval", type=int, default=None)
    parser.add_argument("--identity-min-confidence", type=float, default=None)
    parser.add_argument("--identity-reacquire-threshold", type=float, default=None)
    parser.add_argument("--identity-reacquire-max-frames", type=int, default=None)
    parser.add_argument("--identity-reacquire-margin", type=float, default=None)
    parser.add_argument("--identity-new-confirm-frames", type=int, default=None)
    parser.add_argument("--verify-predicted-reid", action="store_true")
    parser.add_argument("--predicted-reid-verify-threshold", type=float, default=None)
    parser.add_argument("--predicted-reid-duplicate-iou-threshold", type=float, default=None)
    parser.add_argument("--predicted-reid-duplicate-overlap-threshold", type=float, default=None)
    return parser.parse_args()


@dataclass
class LatestFrame:
    condition: threading.Condition = field(default_factory=lambda: threading.Condition(threading.Lock()))
    frame: Optional[Any] = None
    capture_index: int = 0
    capture_ts: float = 0.0


@dataclass
class InferenceStats:
    processed: int = 0
    frames_with_person: int = 0
    frames_with_track: int = 0
    track_ids: set = field(default_factory=set)
    timing_sums: Dict[str, float] = field(default_factory=dict)
    timing_max: Dict[str, float] = field(default_factory=dict)
    error: str = ""
    traceback: str = ""


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.raw_write_every <= 0:
        raise ValueError("--raw-write-every must be > 0")
    if args.track_publish_every <= 0:
        raise ValueError("--track-publish-every must be > 0")
    if args.track_video_fps <= 0:
        raise ValueError("--track-video-fps must be > 0")
    if args.capture_mode == "gstreamer_mjpeg_tee":
        args.raw_writer_mode = "gstreamer_mjpeg_passthrough"

    if args.config:
        load_config_to_env(args.config)
        print(f"[INFO] config       : {args.config}")

    paths = _make_output_paths(args)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    cfg = _build_vision_config(args)
    stop_event = threading.Event()
    signal_count = {"value": 0}

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)

    def request_stop(signum, _frame) -> None:
        signal_count["value"] += 1
        if signal_count["value"] == 1:
            print(f"\n[INFO] signal {signum} received; finalizing videos...")
            stop_event.set()
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cap = None
    raw_writer = None
    gst_source = None
    worker = None
    stats = InferenceStats()
    latest = LatestFrame()
    raw_frames_written = 0
    captured_frames = 0
    start_ts = time.time()
    capture_end_ts = start_ts
    interrupted = False
    writer_fps = float(args.raw_video_fps or args.fps)

    try:
        import cv2

        print(f"[INFO] device       : {args.device}")
        print(f"[INFO] capture mode : {args.capture_mode}")
        print(f"[INFO] request      : {args.width}x{args.height} @ {args.fps:.2f}fps ({args.fourcc})")
        if args.capture_mode == "gstreamer_mjpeg_tee":
            if args.fourcc.strip().upper() != "MJPG":
                raise ValueError("--capture-mode gstreamer_mjpeg_tee requires --fourcc MJPG")
            gst_source = GstMjpegTeeCapture(
                GstMjpegTeeConfig(
                    device=args.device,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    raw_output=str(paths["raw_output"]),
                )
            )
            gst_source.open()
            actual = dict(gst_source.actual)
            print(
                "[INFO] negotiated  : "
                f"{actual['width']}x{actual['height']} @ {actual['fps']:.2f}fps fourcc={actual['fourcc']!r}"
            )
            print(f"[INFO] gst pipeline : {gst_source.pipeline_description}")
            ok, frame = gst_source.read()
            if not ok or frame is None:
                print("[WARN] GStreamer camera read failed during warmup.")
                return 2
            args.raw_writer_mode = "gstreamer_mjpeg_passthrough"
        else:
            cap = open_camera(cv2, args.device, args.width, args.height, args.fps, args.fourcc)
            actual = camera_actual(cv2, cap)
            print(
                "[INFO] negotiated  : "
                f"{actual['width']}x{actual['height']} @ {actual['fps']:.2f}fps fourcc={actual['fourcc']!r}"
            )

            ok, frame = read_first_frame(cap)
            if not ok or frame is None:
                print("[WARN] camera read failed during warmup.")
                return 2

            out_h, out_w = int(frame.shape[0]), int(frame.shape[1])
            raw_writer = create_writer(
                cv2,
                paths["raw_output"],
                writer_fps,
                out_w,
                out_h,
                args.raw_writer_mode,
                args.raw_gst_pipeline,
            )
        print(f"[INFO] raw video    : {paths['raw_output']}")
        print(f"[INFO] track video  : {paths['track_output']}")
        print(f"[INFO] jsonl        : {paths['jsonl']}")
        print(f"[INFO] summary      : {paths['summary_json']}")
        print("[INFO] stop         : Ctrl+C" + (" or q in preview" if args.display else ""))

        worker = threading.Thread(
            target=inference_worker,
            name="RKNNTrackRecorder",
            args=(args, cfg, latest, stop_event, stats, paths["track_output"], paths["jsonl"]),
            daemon=True,
        )
        worker.start()

        start_ts = time.time()
        capture_end_ts = start_ts
        window_name = "camera_raw"
        while not stop_event.is_set():
            frame_start = time.time()
            capture_end_ts = frame_start
            captured_frames += 1

            if gst_source is not None:
                raw_frames_written = captured_frames
            elif captured_frames % args.raw_write_every == 0:
                raw_writer.write(frame)
                raw_frames_written += 1

            if captured_frames % args.track_publish_every == 0:
                with latest.condition:
                    latest.frame = frame
                    latest.capture_index = captured_frames
                    latest.capture_ts = frame_start
                    latest.condition.notify()

            if args.display:
                cv2.imshow(window_name, frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("[INFO] stopped by user (q).")
                    stop_event.set()
                    break

            if args.duration > 0 and time.time() - start_ts >= args.duration:
                print("[INFO] duration reached.")
                capture_end_ts = time.time()
                stop_event.set()
                break

            ok, frame = gst_source.read() if gst_source is not None else cap.read()
            if not ok or frame is None:
                print("[WARN] camera read failed mid-stream.")
                stop_event.set()
                break

    except KeyboardInterrupt:
        interrupted = True
        capture_end_ts = time.time()
        stop_event.set()
        print("\n[INFO] interrupted; finalizing videos...")
    finally:
        stop_event.set()
        with latest.condition:
            latest.condition.notify_all()
        if worker is not None:
            worker.join(timeout=30.0)
            if worker.is_alive():
                print("[WARN] tracker worker did not exit within 30s.")
        if raw_writer is not None:
            raw_writer.release()
        if gst_source is not None:
            gst_source.release()
        if cap is not None:
            cap.release()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    elapsed = max(capture_end_ts - start_ts, 1e-9)
    raw_probe = probe_video_file(paths["raw_output"])
    track_probe = probe_video_file(paths["track_output"])
    if args.capture_mode == "gstreamer_mjpeg_tee" and raw_probe.get("frames"):
        raw_frames_written = int(raw_probe["frames"])
    raw_avg_elapsed = max(float(raw_probe.get("duration_sec") or elapsed), 1e-9)

    summary = {
        "raw_output": str(paths["raw_output"]),
        "track_output": str(paths["track_output"]),
        "jsonl": str(paths["jsonl"]),
        "summary_json": str(paths["summary_json"]),
        "config": str(args.config),
        "device": args.device,
        "request": {"width": args.width, "height": args.height, "fps": args.fps, "fourcc": args.fourcc},
        "capture_mode": args.capture_mode,
        "duration_sec": round(elapsed, 3),
        "interrupted": bool(interrupted or signal_count["value"] > 0),
        "captured_frames": int(captured_frames),
        "raw_frames_written": int(raw_frames_written),
        "raw_avg_fps": round(raw_frames_written / raw_avg_elapsed, 3),
        "raw_writer_mode": args.raw_writer_mode,
        "track_writer_mode": args.track_writer_mode,
        "raw_file_probe": raw_probe,
        "track_file_probe": track_probe,
        "track_frames_written": int(stats.processed),
        "track_avg_fps": round(stats.processed / elapsed, 3),
        "frames_with_person": int(stats.frames_with_person),
        "frames_with_track": int(stats.frames_with_track),
        "track_ids": sorted(int(x) for x in stats.track_ids),
        "avg_timing_ms": average_timing(stats.timing_sums, stats.processed),
        "max_timing_ms": rounded_timing(stats.timing_max),
        "error": stats.error,
    }
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[INFO] saved raw    :", paths["raw_output"])
    print("[INFO] saved track  :", paths["track_output"])
    print("[INFO] saved jsonl  :", paths["jsonl"])
    print("[INFO] saved summary:", paths["summary_json"])
    print("[INFO] raw frames   :", raw_frames_written, f"({summary['raw_avg_fps']:.2f} fps avg)")
    print("[INFO] track frames :", stats.processed, f"({summary['track_avg_fps']:.2f} fps avg)")
    if stats.error:
        print("[ERROR] tracker worker failed:", stats.error)
        return 1
    return 0


def _make_output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.name.strip() or f"camera_track_{timestamp}"
    raw_ext = ".avi" if args.raw_writer_mode in {"opencv_mjpg_avi", "gstreamer_mjpeg_passthrough"} else ".mp4"
    track_ext = ".avi" if args.track_writer_mode == "opencv_mjpg_avi" else ".mp4"
    return {
        "output_dir": output_dir,
        "raw_output": Path(args.raw_output) if args.raw_output else output_dir / f"{base}_raw{raw_ext}",
        "track_output": Path(args.track_output) if args.track_output else output_dir / f"{base}_track{track_ext}",
        "jsonl": Path(args.jsonl) if args.jsonl else output_dir / f"{base}_track.jsonl",
        "summary_json": Path(args.summary_json) if args.summary_json else output_dir / f"{base}_summary.json",
    }


def _build_vision_config(args: argparse.Namespace) -> RKNNVisionConfig:
    cfg = RKNNVisionConfig.from_env()
    kwargs = {**cfg.__dict__}
    if args.yolo_model:
        kwargs["yolo_model_path"] = args.yolo_model
    if args.reid_model:
        kwargs["reid_model_path"] = args.reid_model
    if args.no_reid:
        kwargs["reid_enable"] = False
    if args.backend:
        kwargs["backend"] = args.backend
    if args.core_mask:
        kwargs["core_mask"] = args.core_mask
    if args.conf_threshold is not None:
        kwargs["conf_threshold"] = float(args.conf_threshold)
    if args.max_output_age is not None:
        kwargs["max_output_age"] = max(0, int(args.max_output_age))
    if args.bbox_expand_scale is not None:
        kwargs["deepsort_bbox_expand_scale"] = max(1.0, float(args.bbox_expand_scale))
    if args.identity_match_threshold is not None:
        kwargs["identity_match_threshold"] = float(args.identity_match_threshold)
    if args.identity_update_interval is not None:
        kwargs["identity_update_interval"] = max(1, int(args.identity_update_interval))
    if args.identity_min_confidence is not None:
        kwargs["identity_min_confidence"] = float(args.identity_min_confidence)
    if args.identity_reacquire_threshold is not None:
        kwargs["identity_reacquire_threshold"] = float(args.identity_reacquire_threshold)
    if args.identity_reacquire_max_frames is not None:
        kwargs["identity_reacquire_max_frames"] = max(0, int(args.identity_reacquire_max_frames))
    if args.identity_reacquire_margin is not None:
        kwargs["identity_reacquire_margin"] = float(args.identity_reacquire_margin)
    if args.identity_new_confirm_frames is not None:
        kwargs["identity_new_confirm_frames"] = max(1, int(args.identity_new_confirm_frames))
    if args.verify_predicted_reid:
        kwargs["predicted_reid_verify_enable"] = True
    if args.predicted_reid_verify_threshold is not None:
        kwargs["predicted_reid_verify_threshold"] = float(args.predicted_reid_verify_threshold)
    if args.predicted_reid_duplicate_iou_threshold is not None:
        kwargs["predicted_reid_duplicate_iou_threshold"] = float(args.predicted_reid_duplicate_iou_threshold)
    if args.predicted_reid_duplicate_overlap_threshold is not None:
        kwargs["predicted_reid_duplicate_overlap_threshold"] = float(
            args.predicted_reid_duplicate_overlap_threshold
        )
    return RKNNVisionConfig(**kwargs)


def inference_worker(
    args: argparse.Namespace,
    cfg: RKNNVisionConfig,
    latest: LatestFrame,
    stop_event: threading.Event,
    stats: InferenceStats,
    track_output: Path,
    jsonl_path: Path,
) -> None:
    pipeline = None
    writer = None
    jsonl_file = None
    last_seen = 0
    next_allowed_ts = 0.0
    try:
        import cv2

        track_output.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_path.open("w", encoding="utf-8")

        pipeline = RKNNVisionPipeline(cfg)
        pipeline.load()
        while not stop_event.is_set():
            with latest.condition:
                latest.condition.wait_for(
                    lambda: stop_event.is_set() or latest.capture_index > last_seen,
                    timeout=0.5,
                )
                if latest.capture_index <= last_seen:
                    continue
                capture_index = int(latest.capture_index)
                capture_ts = float(latest.capture_ts)
                frame = latest.frame.copy() if latest.frame is not None else None
                last_seen = capture_index
            if frame is None:
                continue

            if args.inference_max_fps > 0:
                now = time.time()
                if now < next_allowed_ts:
                    time.sleep(next_allowed_ts - now)
                next_allowed_ts = time.time() + 1.0 / float(args.inference_max_fps)

            if writer is None:
                height, width = int(frame.shape[0]), int(frame.shape[1])
                writer = create_writer(
                    cv2,
                    track_output,
                    args.track_video_fps,
                    width,
                    height,
                    args.track_writer_mode,
                    args.track_gst_pipeline,
                )

            infer_start = time.time()
            records = pipeline.process_frame(frame, "BGR")
            infer_end = time.time()
            detections = list(pipeline.last_detections)
            persons = [det for det in detections if int(det.class_id) == int(cfg.person_class_id)]
            timing = {"queue_delay": max(0.0, (infer_start - capture_ts) * 1000.0), **pipeline.last_timing_ms}
            accumulate_timing(timing, stats.timing_sums, stats.timing_max)

            if persons:
                stats.frames_with_person += 1
            if records:
                stats.frames_with_track += 1
                stats.track_ids.update(int(rec.track_id) for rec in records)
            stats.processed += 1

            if args.draw_detections:
                draw_detections(cv2, frame, persons)
            draw_records(cv2, frame, records)
            draw_status(cv2, frame, capture_index, stats.processed, records, timing, infer_end - capture_ts)
            writer.write(frame)

            item = {
                "capture_index": capture_index,
                "track_index": stats.processed,
                "capture_ts": round(capture_ts, 6),
                "infer_start_ts": round(infer_start, 6),
                "infer_end_ts": round(infer_end, 6),
                "capture_to_output_ms": round(max(0.0, (infer_end - capture_ts) * 1000.0), 3),
                "detections": [detection_to_dict(det) for det in detections],
                "persons": len(persons),
                "tracks": [record_to_dict(rec) for rec in records],
                "predicted_reid_verifications": [
                    normalize_verification(item) for item in pipeline.last_predicted_reid_verifications
                ],
                "timing_ms": rounded_timing(timing),
            }
            if args.debug_tracker_state:
                item["tracker_debug"] = pipeline.debug_state()
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            jsonl_file.flush()
    except Exception as exc:
        stats.error = str(exc)
        stats.traceback = traceback.format_exc()
        print("[ERROR] tracker worker exception:", stats.error)
        print(stats.traceback)
        stop_event.set()
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
        if writer is not None:
            writer.release()
        if pipeline is not None:
            pipeline.close()


def open_camera(cv2, device: str, width: int, height: int, fps: float, fourcc: str):
    cap = cv2.VideoCapture(camera_device_arg(device), cv2.CAP_V4L2)
    if not cap.isOpened():
        hint = (
            "If Ctrl+Z was used earlier, a stopped job may still hold the device. "
            "Run `jobs -l` then `kill %N`, or `fg` and quit with Ctrl+C."
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


def read_first_frame(cap, warmup_attempts: int = 15) -> Tuple[bool, Any]:
    frame = None
    for _ in range(warmup_attempts):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(0.05)
    return False, frame


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


def detection_to_dict(det) -> Dict[str, Any]:
    return {
        "class_id": int(det.class_id),
        "score": round(float(det.score), 5),
        "bbox": [round(float(v), 1) for v in det.bbox],
    }


def record_to_dict(rec) -> Dict[str, Any]:
    return {
        "track_id": int(rec.track_id),
        "reid_uid": int(rec.reid_uid),
        "state": int(rec.tracker_state),
        "class_id": int(rec.class_id),
        "score": round(float(rec.score), 5),
        "bbox": [round(float(v), 1) for v in (rec.x1, rec.y1, rec.x2, rec.y2)],
        "cx": round(float(rec.cx), 2),
        "cy": round(float(rec.cy), 2),
        "area": round(float(rec.area), 1),
        "angle_deg": round(float(rec.angle_deg), 2),
        "time_since_update": int(rec.time_since_update),
        "reid_verify_distance": (
            None if rec.reid_verify_distance is None else round(float(rec.reid_verify_distance), 4)
        ),
        "reid_verify_passed": rec.reid_verify_passed,
    }


def normalize_verification(item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(item)
    if out.get("distance") is not None:
        out["distance"] = round(float(out["distance"]), 4)
    if out.get("threshold") is not None:
        out["threshold"] = round(float(out["threshold"]), 4)
    if out.get("bbox") is not None:
        out["bbox"] = [round(float(v), 1) for v in out["bbox"]]
    return out


def draw_detections(cv2, frame, detections) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(round(float(v))) for v in det.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 1)
        put_label(cv2, frame, f"det {float(det.score):.2f}", x1, y1 - 6, (255, 160, 0))


def draw_records(cv2, frame, records) -> None:
    for rec in records:
        x1, y1, x2, y2 = [int(round(float(v))) for v in (rec.x1, rec.y1, rec.x2, rec.y2)]
        color = (0, 220, 0) if int(rec.time_since_update) == 0 else (0, 190, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = (
            f"id={int(rec.track_id)} uid={int(rec.reid_uid)} "
            f"s={float(rec.score):.2f} age={int(rec.time_since_update)}"
        )
        if rec.reid_verify_distance is not None:
            label += f" d={float(rec.reid_verify_distance):.2f}"
        put_label(cv2, frame, label, x1, y1 - 22, color)


def draw_status(cv2, frame, capture_index: int, track_index: int, records, timing: Dict[str, float], age_sec: float) -> None:
    text = (
        f"cap={capture_index} track={track_index} tracks={len(records)} "
        f"age={age_sec:.2f}s total={float(timing.get('total', 0.0)):.1f}ms"
    )
    put_label(cv2, frame, text, 12, 28, (255, 255, 255))


def put_label(cv2, frame, text: str, x: int, y: int, color) -> None:
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    pad = 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_w = tw + pad * 2
    box_h = th + baseline + pad * 2
    x = max(0, min(int(x), max(0, width - box_w)))
    y = max(box_h, min(int(y), max(box_h, height - baseline)))
    top = max(0, y - box_h)
    cv2.rectangle(frame, (x, top), (min(width - 1, x + box_w), min(height - 1, y)), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + pad, y - pad), font, scale, color, thickness, cv2.LINE_AA)


def rounded_timing(timing: Dict[str, float]) -> Dict[str, float]:
    return {key: round(float(value), 3) for key, value in timing.items()}


def accumulate_timing(timing: Dict[str, float], sums: Dict[str, float], maxes: Dict[str, float]) -> None:
    for key, value in timing.items():
        value_f = float(value)
        sums[key] = sums.get(key, 0.0) + value_f
        maxes[key] = max(maxes.get(key, 0.0), value_f)


def average_timing(sums: Dict[str, float], count: int) -> Dict[str, float]:
    if count <= 0:
        return {}
    return {key: round(float(value) / float(count), 3) for key, value in sums.items()}


def probe_video_file(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        import cv2
    except Exception:
        return {}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {}
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return {
            "frames": int(round(frames)) if frames >= 0 else 0,
            "fps": round(fps, 3) if fps > 0 else 0.0,
            "duration_sec": round(frames / fps, 3) if frames >= 0 and fps > 0 else 0.0,
            "width": int(round(width)) if width > 0 else 0,
            "height": int(round(height)) if height > 0 else 0,
        }
    finally:
        cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
