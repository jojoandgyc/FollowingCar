#!/usr/bin/env python3
"""Replay exact RGB/Depth diagnostic packets, or collect camera-only pose observations.

No motor/controller imports. Live acquisition only accepts a single visible
person and labels its identity as unverified; it is not an identity benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import fields
import configparser
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from car_control_modular.pose_shadow import (
    PoseEstimator, PoseShadowConfig, analyze_pose, render_comparison, validate_packet,
    ObservationVideo, export_mp4,
)


def summarize(results, counts):
    times = [r["inference_ms"] for r in results]
    return dict(
        counts=dict(counts), evaluated=len(results),
        inference_ms=None if not times else dict(p50=float(np.percentile(times, 50)),
                                                p95=float(np.percentile(times, 95)), max=max(times)),
        pose_depth_available=sum((r["pose_depth"] or {}).get("median_m") is not None for r in results),
        bbox_depth_available=sum(r["bbox_region_depth"]["median_m"] is not None for r in results),
        note="Diagnostic medians only; no ground truth, no accuracy claim. Sparse snapshots cannot measure continuous jitter.",
    )


def replay(args, config):
    import cv2
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    estimator = PoseEstimator(config)
    results, counts, seen = [], Counter(), set()
    video = ObservationVideo(output / "inference.avi", fps=1, label="SPARSE SNAPSHOT REPLAY") if args.video else None
    with ExitStack() as stack:
        if video is not None:
            stack.callback(video.close)
        stream = stack.enter_context((output / "observations.jsonl").open("x"))
        for path in sorted(args.input.rglob("*.npz")):
            # Never consume outputs nested inside the source folder.
            if output.resolve() in path.resolve().parents:
                continue
            counts["files_seen"] += 1
            try:
                with np.load(path, allow_pickle=False) as saved:
                    metadata = json.loads(saved["metadata_json"].item())
                    rgb = saved["rgb_bgr"] if "rgb_bgr" in saved else None
                    depth = saved["depth_mm"]
                error = validate_packet(rgb, depth, metadata, config)
                if error:
                    counts[error] += 1
                    continue
                key = (str(path.parent.resolve()), metadata["target_id"], metadata["rgb_capture_id"], metadata["sample_timestamp"])
                if key in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add(key)
                started = time.perf_counter()
                pose = estimator.infer(rgb, metadata["bbox"])
                inference_ms = (time.perf_counter() - started) * 1000
                result = analyze_pose(rgb, depth, metadata, pose, config)
                result.update(inference_ms=inference_ms, source_file=str(path), offline=True)
                preview = render_comparison(rgb, result)
                if video is not None:
                    result["video_frame_index"] = video.write(preview, result)
                results.append(result)
                counts[result["reason"]] += 1
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                cv2.imwrite(str(output / f"comparison_{len(results):03d}.jpg"), preview)
                if len(results) >= args.limit:
                    break
            except (KeyError, ValueError, OSError) as exc:
                counts["invalid_packet"] += 1
                print(f"Skip {path}: {exc}", file=sys.stderr)
    if video is not None:
        export_mp4(video.path)
    report = summarize(results, counts)
    report["model_sha256"] = hashlib.sha256(Path(config.model).read_bytes()).hexdigest()
    report["adapter_sha256"] = hashlib.sha256(Path(config.adapter).read_bytes()).hexdigest()
    report["pose_config"] = vars(config)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if results else 2


def live(args, config):
    import cv2
    import logging
    from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
    from car_control_modular.pose_shadow import PoseShadowObserver
    from rk_vision.yolo11 import YOLO11Config, YOLO11RKNNDetector

    settings = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not settings.read(args.config):
        raise ValueError(f"Cannot read {args.config}")
    # Read matching dataclass fields only; do not load or construct car control.
    defaults = AstraDepthConfig()
    depth_values = {}
    for f in fields(defaults):
        if settings.has_option("astra_depth", f.name) and f.name != "diagnostics_dir":
            depth_values[f.name] = type(getattr(defaults, f.name))(settings.get("astra_depth", f.name))
    sensor = AstraDepthRuntime(AstraDepthConfig(**depth_values))
    model = Path(settings.get("vision", "model_path"))
    if not model.is_absolute():
        model = ROOT / model
    detector = YOLO11RKNNDetector(YOLO11Config(
        model_path=str(model), input_size=settings.getint("vision", "input_size", fallback=640),
        conf_threshold=.5, nms_threshold=settings.getfloat("vision", "nms_threshold", fallback=.45),
        num_classes=settings.getint("vision", "num_classes", fallback=80),
        input_format=settings.get("vision", "yolo_input_format", fallback="RGB"),
        output_box_format=settings.get("vision", "yolo_box_format", fallback="xywh"),
        target="rk3588", core_mask="auto", backend="auto"))
    camera = observer = None
    logging.basicConfig(level=logging.WARNING)
    counts = Counter()
    single_present, observation_id = False, 0
    try:
        detector.load()
        sensor.start()
        if not sensor.wait_until_ready(3):
            raise RuntimeError("No depth frame")
        camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, sensor.config.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, sensor.config.height)
        camera.set(cv2.CAP_PROP_FPS, sensor.config.fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not camera.isOpened():
            raise RuntimeError("Cannot open RGB camera")
        observer = PoseShadowObserver(args.output, config=config)
        acquisition_started = time.monotonic()
        deadline = time.monotonic() + args.duration
        last_report = 0.
        print("Camera-only observation started. One person only; no motor connection. Ctrl+C to finish.", flush=True)
        while time.monotonic() < deadline:
            ok, rgb = camera.read()
            stamp = time.monotonic()
            if not ok:
                raise RuntimeError("RGB read failed")
            counts["frames"] += 1
            capture_id = counts["frames"]
            people = [d for d in detector.detect(rgb, frame_format="BGR") if d.class_id == 0]
            if len(people) != 1:
                counts["not_single_person"] += 1
                single_present = False
                continue
            if not single_present:
                observation_id += 1
                single_present = True
            bbox = tuple(float(v) for v in people[0].bbox)
            measurement = sensor.measure_target(
                bbox, rgb.shape[1], rgb.shape[0], target_id=observation_id,
                reference_timestamp=stamp, evidence_capture_frame_id=capture_id)
            sample = sensor._measurement_sample_ts
            with sensor._depth_lock:
                match = next((d for d in sensor._depth_history if sample is not None and abs(d[0]-sample) < 1e-6), None)
            if match is None:
                counts["missing_sample"] += 1
                continue
            # This is a geometry experiment, not a confirmed ReID UID.
            metadata = dict(target_id=observation_id, identity_source="single_person_unverified", source="rgb_aligned",
                            bbox=bbox, frame_size=[rgb.shape[1], rgb.shape[0]],
                            evidence_capture_frame_id=capture_id, rgb_capture_id=capture_id,
                            rgb_timestamp=stamp, reference_timestamp=stamp,
                            depth_timestamp=match[0], sample_timestamp=sample,
                            regions=list(sensor._measurement_regions),
                            candidate_m=measurement.candidate_distance_m,
                            accepted_raw_m=measurement.raw_distance_m,
                            filtered_or_held_m=measurement.distance_m,
                            detail=measurement.detail,
                            acquisition_elapsed_sec=stamp-acquisition_started,
                            orientation=sensor._depth_orientation.metadata())
            observer.submit(rgb.copy(), match[1], metadata)
            if stamp-last_report > 5:
                print(f"frames={capture_id} shadow={dict(observer.counts)}", flush=True)
                last_report = stamp
            if observer.counts["worker_error"]:
                raise RuntimeError("Pose worker failed; see log")
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.release()
        sensor.close()
        detector.release()
        if observer is not None:
            observer.close()
            observer._thread.join(timeout=3)
            print(f"Output: {args.output}; shadow={dict(observer.counts)}", flush=True)
            rows_path = args.output / "observations.jsonl"
            if not observer._thread.is_alive() and rows_path.is_file():
                rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
                report = summarize(rows, counts)
                report["shadow_counts"] = dict(observer.counts)
                report["effective_depth_config"] = vars(sensor.config)
                report["acquisition_config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
                (args.output / "evaluation.json").write_text(json.dumps(report, indent=2))
                export_mp4(args.output / "inference.avi")
    return 0 if observer is not None and observer.counts["written"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="Replay existing diagnostic .npz packets recursively")
    mode.add_argument("--live", action="store_true", help="Camera-only acquisition; close other camera users first")
    parser.add_argument("--output", type=Path, required=True, help="New directory, never overwrite")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--video", action="store_true", help="Also export sampled snapshot replay to AVI/MP4 (live always records video)")
    parser.add_argument("--config", type=Path, default=ROOT / "car_control_modular/config/reid_runtime.ini")
    parser.add_argument("--device", default="/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0")
    parser.add_argument("--model", default=PoseShadowConfig.model)
    parser.add_argument("--adapter", default=PoseShadowConfig.adapter)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    if args.input is not None and not args.input.is_dir():
        parser.error("Input must be a directory")
    if not 1 <= args.limit <= 120 or not 0 < args.duration <= 180:
        parser.error("limit must be 1..120 and duration must be 0..180 seconds")
    config = PoseShadowConfig(model=args.model, adapter=args.adapter, max_samples=args.limit)
    return live(args, config) if args.live else replay(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
