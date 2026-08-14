#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RKNN YOLO11/ReID tracker on a video.")
    parser.add_argument("video")
    parser.add_argument("--yolo-model", default="models/yolo11s.rknn")
    parser.add_argument("--reid-model", default="models/deepsort.rknn")
    parser.add_argument("--reid-input-width", type=int, default=None)
    parser.add_argument("--reid-input-height", type=int, default=None)
    parser.add_argument("--reid-input-format", default="")
    parser.add_argument("--reid-input-dtype", default="")
    parser.add_argument("--reid-input-layout", default="")
    parser.add_argument("--reid-normalize", default="")
    parser.add_argument(
        "--feature-update-interval",
        type=int,
        default=None,
        help="Only store matched ReID features into the DeepSORT gallery every N frames.",
    )
    parser.add_argument("--max-output-age", type=int, default=None, help="Output predicted tracks for up to N missed frames.")
    parser.add_argument("--bbox-expand-scale", type=float, default=None, help="Scale DeepSORT public bbox input; use 1.0 for tighter overlay boxes.")
    parser.add_argument("--disable-identity-bank", action="store_true", help="Use DeepSORT track IDs as reid_uid.")
    parser.add_argument("--identity-match-threshold", type=float, default=None)
    parser.add_argument("--identity-update-interval", type=int, default=None)
    parser.add_argument("--identity-min-confidence", type=float, default=None)
    parser.add_argument("--identity-reacquire-threshold", type=float, default=None)
    parser.add_argument("--identity-reacquire-max-frames", type=int, default=None)
    parser.add_argument("--identity-reacquire-margin", type=float, default=None)
    parser.add_argument("--identity-new-confirm-frames", type=int, default=None)
    parser.add_argument("--verify-predicted-reid", action="store_true", help="Verify predicted-only tracks with a ReID crop before output.")
    parser.add_argument("--predicted-reid-verify-threshold", type=float, default=None)
    parser.add_argument("--predicted-reid-duplicate-iou-threshold", type=float, default=None)
    parser.add_argument("--predicted-reid-duplicate-overlap-threshold", type=float, default=None)
    parser.add_argument("--backend", default=os.environ.get("RKNN_BACKEND", "auto"))
    parser.add_argument("--core-mask", default=os.environ.get("RKNN_CORE_MASK", "auto"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--save-video", default="", help="Optional output MP4 with tracked boxes.")
    parser.add_argument("--jsonl", default="", help="Optional path to save per-frame JSON lines.")
    parser.add_argument("--debug-tracker-state", action="store_true", help="Include internal DeepSORT state in JSONL.")
    args = parser.parse_args()

    try:
        import cv2
    except Exception as exc:
        raise SystemExit(f"OpenCV is required to load video files: {exc}") from exc

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start_frame = max(0, int(args.start_frame))
    max_frames = max(1, int(args.max_frames))
    frame_stride = max(1, int(args.frame_stride))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    cfg = RKNNVisionConfig.from_env()
    cfg_kwargs = {
        **cfg.__dict__,
        "yolo_model_path": args.yolo_model,
        "reid_model_path": args.reid_model,
        "reid_enable": not args.no_reid,
        "backend": args.backend,
        "core_mask": args.core_mask,
    }
    if args.reid_input_width is not None:
        cfg_kwargs["reid_input_width"] = args.reid_input_width
    if args.reid_input_height is not None:
        cfg_kwargs["reid_input_height"] = args.reid_input_height
    if args.reid_input_format:
        cfg_kwargs["reid_input_format"] = args.reid_input_format
    if args.reid_input_dtype:
        cfg_kwargs["reid_input_dtype"] = args.reid_input_dtype
    if args.reid_input_layout:
        cfg_kwargs["reid_input_layout"] = args.reid_input_layout
    if args.reid_normalize:
        cfg_kwargs["reid_normalize"] = args.reid_normalize
    if args.feature_update_interval is not None:
        cfg_kwargs["feature_update_interval"] = max(1, int(args.feature_update_interval))
    if args.max_output_age is not None:
        cfg_kwargs["max_output_age"] = max(0, int(args.max_output_age))
    if args.bbox_expand_scale is not None:
        cfg_kwargs["deepsort_bbox_expand_scale"] = max(1.0, float(args.bbox_expand_scale))
    if args.disable_identity_bank:
        cfg_kwargs["identity_bank_enable"] = False
    if args.identity_match_threshold is not None:
        cfg_kwargs["identity_match_threshold"] = float(args.identity_match_threshold)
    if args.identity_update_interval is not None:
        cfg_kwargs["identity_update_interval"] = max(1, int(args.identity_update_interval))
    if args.identity_min_confidence is not None:
        cfg_kwargs["identity_min_confidence"] = float(args.identity_min_confidence)
    if args.identity_reacquire_threshold is not None:
        cfg_kwargs["identity_reacquire_threshold"] = float(args.identity_reacquire_threshold)
    if args.identity_reacquire_max_frames is not None:
        cfg_kwargs["identity_reacquire_max_frames"] = max(0, int(args.identity_reacquire_max_frames))
    if args.identity_reacquire_margin is not None:
        cfg_kwargs["identity_reacquire_margin"] = float(args.identity_reacquire_margin)
    if args.identity_new_confirm_frames is not None:
        cfg_kwargs["identity_new_confirm_frames"] = max(1, int(args.identity_new_confirm_frames))
    if args.verify_predicted_reid:
        cfg_kwargs["predicted_reid_verify_enable"] = True
    if args.predicted_reid_verify_threshold is not None:
        cfg_kwargs["predicted_reid_verify_threshold"] = float(args.predicted_reid_verify_threshold)
    if args.predicted_reid_duplicate_iou_threshold is not None:
        cfg_kwargs["predicted_reid_duplicate_iou_threshold"] = float(args.predicted_reid_duplicate_iou_threshold)
    if args.predicted_reid_duplicate_overlap_threshold is not None:
        cfg_kwargs["predicted_reid_duplicate_overlap_threshold"] = float(
            args.predicted_reid_duplicate_overlap_threshold
        )
    cfg = RKNNVisionConfig(
        **cfg_kwargs
    )
    pipeline = RKNNVisionPipeline(cfg)
    pipeline.load()
    writer = None
    jsonl_file = None

    summary = {
        "video": str(args.video),
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "start_frame": start_frame,
        "max_frames": max_frames,
        "frame_stride": frame_stride,
        "feature_update_interval": int(cfg.feature_update_interval),
        "max_output_age": int(cfg.max_output_age),
        "deepsort_bbox_expand_scale": float(cfg.deepsort_bbox_expand_scale),
        "identity_bank_enable": bool(cfg.identity_bank_enable),
        "identity_match_threshold": float(cfg.identity_match_threshold),
        "identity_update_threshold": float(cfg.identity_update_threshold),
        "identity_update_interval": int(cfg.identity_update_interval),
        "identity_min_confidence": float(cfg.identity_min_confidence),
        "identity_max_area_ratio": float(cfg.identity_max_area_ratio),
        "identity_max_width_ratio": float(cfg.identity_max_width_ratio),
        "identity_max_height_ratio": float(cfg.identity_max_height_ratio),
        "identity_min_aspect_ratio": float(cfg.identity_min_aspect_ratio),
        "identity_max_aspect_ratio": float(cfg.identity_max_aspect_ratio),
        "identity_max_edge_touch_count": int(cfg.identity_max_edge_touch_count),
        "identity_edge_margin_ratio": float(cfg.identity_edge_margin_ratio),
        "identity_reacquire_threshold": float(cfg.identity_reacquire_threshold),
        "identity_reacquire_max_frames": int(cfg.identity_reacquire_max_frames),
        "identity_reacquire_margin": float(cfg.identity_reacquire_margin),
        "identity_reacquire_single_candidate_only": bool(cfg.identity_reacquire_single_candidate_only),
        "identity_new_confirm_frames": int(cfg.identity_new_confirm_frames),
        "predicted_reid_verify_enable": bool(cfg.predicted_reid_verify_enable),
        "predicted_reid_verify_threshold": float(cfg.predicted_reid_verify_threshold),
        "predicted_reid_duplicate_iou_threshold": float(cfg.predicted_reid_duplicate_iou_threshold),
        "predicted_reid_duplicate_overlap_threshold": float(cfg.predicted_reid_duplicate_overlap_threshold),
        "debug_tracker_state": bool(args.debug_tracker_state),
    }
    print("video_meta", json.dumps(summary, ensure_ascii=False))

    processed = 0
    frames_with_person = 0
    frames_with_track = 0
    track_ids = set()
    timing_sums = {}
    timing_max = {}
    t0 = time.time()
    try:
        if args.save_video:
            out_path = Path(args.save_video)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps or 25.0, (width, height))
            if not writer.isOpened():
                raise SystemExit(f"failed to open output video: {out_path}")
        if args.jsonl:
            jsonl_path = Path(args.jsonl)
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_file = jsonl_path.open("w", encoding="utf-8")

        frame_index = start_frame
        while processed < max_frames:
            read_start = time.perf_counter()
            ok, frame = cap.read()
            read_end = time.perf_counter()
            if not ok:
                break
            if (frame_index - start_frame) % frame_stride != 0:
                frame_index += 1
                continue

            records = pipeline.process_frame(frame, "BGR")
            timing = {
                "decode": _elapsed_ms(read_start, read_end),
                **pipeline.last_timing_ms,
            }
            _accumulate_timing(timing, timing_sums, timing_max)
            detections = pipeline.last_detections
            persons = [det for det in detections if int(det.class_id) == int(cfg.person_class_id)]
            if persons:
                frames_with_person += 1
            if records:
                frames_with_track += 1
                track_ids.update(int(rec.track_id) for rec in records)

            item = {
                "frame_index": frame_index,
                "detections": len(detections),
                "persons": len(persons),
                "tracks": [
                    {
                        "track_id": int(rec.track_id),
                        "reid_uid": int(rec.reid_uid),
                        "state": int(rec.tracker_state),
                        "score": float(rec.score),
                        "bbox": [round(float(rec.x1), 1), round(float(rec.y1), 1), round(float(rec.x2), 1), round(float(rec.y2), 1)],
                        "angle_deg": round(float(rec.angle_deg), 2),
                        "time_since_update": int(rec.time_since_update),
                        "reid_verify_distance": (
                            None if rec.reid_verify_distance is None else round(float(rec.reid_verify_distance), 4)
                        ),
                        "reid_verify_passed": rec.reid_verify_passed,
                    }
                    for rec in records
                ],
                "predicted_reid_verifications": [
                    {
                        **item,
                        "distance": None if item.get("distance") is None else round(float(item["distance"]), 4),
                        "threshold": round(float(item.get("threshold", 0.0)), 4),
                        "bbox": [round(float(value), 1) for value in item.get("bbox", [])],
                    }
                    for item in pipeline.last_predicted_reid_verifications
                ],
                "timing_ms": _rounded_timing(timing),
            }
            if args.debug_tracker_state:
                item["tracker_debug"] = pipeline.debug_state()
            line = json.dumps(item, ensure_ascii=False)
            print("frame_result", line)
            if jsonl_file is not None:
                jsonl_file.write(line + "\n")
            if writer is not None:
                _draw_records(frame, records)
                writer.write(frame)

            processed += 1
            frame_index += 1
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
        if writer is not None:
            writer.release()
        pipeline.close()
        cap.release()

    elapsed = max(1e-9, time.time() - t0)
    print(
        "summary",
        json.dumps(
            {
                "processed": processed,
                "frames_with_person": frames_with_person,
                "frames_with_track": frames_with_track,
                "track_ids": sorted(track_ids),
                "elapsed_sec": round(elapsed, 3),
                "processed_fps": round(processed / elapsed, 3),
                "avg_timing_ms": _average_timing(timing_sums, processed),
                "max_timing_ms": _rounded_timing(timing_max),
            },
            ensure_ascii=False,
        ),
    )
    return 0


def _draw_records(frame, records) -> None:
    try:
        import cv2
    except Exception:
        return
    height, width = frame.shape[:2]
    for rec in records:
        x1, y1, x2, y2 = [int(round(float(v))) for v in (rec.x1, rec.y1, rec.x2, rec.y2)]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"id={int(rec.track_id)} uid={int(rec.reid_uid)} s={float(rec.score):.2f}"
        if int(rec.time_since_update) > 0:
            label += f" age={int(rec.time_since_update)}"
        if rec.reid_verify_distance is not None:
            label += f" d={float(rec.reid_verify_distance):.2f}"
        _put_box_label(cv2, frame, label, x1, y1, (0, 255, 0))


def _put_box_label(cv2, frame, text: str, x: int, y: int, color) -> None:
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    pad = 4
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_w = tw + pad * 2
    box_h = th + baseline + pad * 2
    x = max(0, min(int(x), max(0, width - box_w)))
    above_y = int(y) - box_h - 4
    if above_y >= 0:
        top = above_y
    else:
        top = min(max(0, int(y) + 4), max(0, height - box_h))
    cv2.rectangle(
        frame,
        (x, top),
        (min(width - 1, x + box_w), min(height - 1, top + box_h)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x + pad, top + pad + th), font, scale, color, thickness, cv2.LINE_AA)


def _elapsed_ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


def _rounded_timing(timing) -> dict:
    return {key: round(float(value), 3) for key, value in timing.items()}


def _accumulate_timing(timing, sums: dict, maxes: dict) -> None:
    for key, value in timing.items():
        value_f = float(value)
        sums[key] = sums.get(key, 0.0) + value_f
        maxes[key] = max(maxes.get(key, 0.0), value_f)


def _average_timing(sums: dict, count: int) -> dict:
    if count <= 0:
        return {}
    return {key: round(float(value) / float(count), 3) for key, value in sums.items()}


if __name__ == "__main__":
    raise SystemExit(main())
