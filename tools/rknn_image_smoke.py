#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RKNN YOLO11/ReID pipeline on one image.")
    parser.add_argument("image")
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
    parser.add_argument("--backend", default=os.environ.get("RKNN_BACKEND", "auto"))
    parser.add_argument("--no-reid", action="store_true")
    args = parser.parse_args()

    try:
        import cv2
    except Exception as exc:
        raise SystemExit(f"OpenCV is required to load image files: {exc}") from exc

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"failed to read image: {args.image}")

    cfg = RKNNVisionConfig.from_env()
    cfg_kwargs = {
        **cfg.__dict__,
        "yolo_model_path": args.yolo_model,
        "reid_model_path": args.reid_model,
        "reid_enable": not args.no_reid,
        "backend": args.backend,
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
    cfg = RKNNVisionConfig(
        **cfg_kwargs
    )
    pipeline = RKNNVisionPipeline(cfg)
    pipeline.load()
    try:
        start = time.perf_counter()
        records = pipeline.process_frame(image, "BGR")
        elapsed_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
        print(
            json.dumps(
                [
                    {
                        "track_id": r.track_id,
                        "reid_uid": r.reid_uid,
                        "bbox": [r.x1, r.y1, r.x2, r.y2],
                        "class_id": r.class_id,
                        "score": r.score,
                        "area": r.area,
                        "angle_deg": r.angle_deg,
                        "tracker_state": r.tracker_state,
                    }
                    for r in records
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        timing = {"total_call": elapsed_ms, **pipeline.last_timing_ms}
        print("timing_ms", json.dumps(_rounded_timing(timing), ensure_ascii=False))
    finally:
        pipeline.close()
    return 0


def _rounded_timing(timing) -> dict:
    return {key: round(float(value), 3) for key, value in timing.items()}


if __name__ == "__main__":
    raise SystemExit(main())
