#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rk_vision.yolo11 import YOLO11Config, YOLO11RKNNDetector


COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run detector-only RKNN YOLO11 on a video.")
    parser.add_argument("video")
    parser.add_argument("--yolo-model", default="models/yolo11s.rknn")
    parser.add_argument("--backend", default=os.environ.get("RKNN_BACKEND", "auto"))
    parser.add_argument("--core-mask", default=os.environ.get("RKNN_CORE_MASK", "auto"))
    parser.add_argument("--conf-threshold", type=float, default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25")))
    parser.add_argument("--nms-threshold", type=float, default=float(os.environ.get("RKNN_YOLO_NMS_THRESHOLD", "0.45")))
    parser.add_argument("--input-size", type=int, default=int(os.environ.get("RKNN_YOLO_INPUT_SIZE", "640")))
    parser.add_argument("--num-classes", type=int, default=int(os.environ.get("RKNN_YOLO_NUM_CLASSES", "80")))
    parser.add_argument("--box-format", default=os.environ.get("RKNN_YOLO_BOX_FORMAT", "xywh"))
    parser.add_argument("--input-format", default=os.environ.get("RKNN_YOLO_INPUT_FORMAT", "RGB"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process to end of video.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--save-video", default="", help="Optional output MP4 with detection boxes.")
    parser.add_argument("--jsonl", default="", help="Optional path to save per-frame detection JSON lines.")
    parser.add_argument("--progress-interval", type=int, default=100)
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
    max_frames = int(args.max_frames)
    frame_stride = max(1, int(args.frame_stride))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detector = YOLO11RKNNDetector(
        YOLO11Config(
            model_path=args.yolo_model,
            input_size=args.input_size,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold,
            num_classes=args.num_classes,
            input_format=args.input_format,
            output_box_format=args.box_format,
            core_mask=args.core_mask,
            backend=args.backend,
        )
    )
    detector.load()

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
        "conf_threshold": args.conf_threshold,
    }
    print("video_meta", json.dumps(summary, ensure_ascii=False))

    processed = 0
    frames_with_detection = 0
    frames_with_person = 0
    class_counter: Counter[int] = Counter()
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
        while max_frames <= 0 or processed < max_frames:
            read_start = time.perf_counter()
            ok, frame = cap.read()
            read_end = time.perf_counter()
            if not ok:
                break
            if (frame_index - start_frame) % frame_stride != 0:
                frame_index += 1
                continue

            detections = detector.detect(frame, "BGR")
            timing = {"decode": _elapsed_ms(read_start, read_end), **detector.last_timing_ms}
            _accumulate_timing(timing, timing_sums, timing_max)

            if detections:
                frames_with_detection += 1
            if any(int(det.class_id) == 0 for det in detections):
                frames_with_person += 1
            for det in detections:
                class_counter[int(det.class_id)] += 1

            item = {
                "frame_index": frame_index,
                "detections": [
                    {
                        "class_id": int(det.class_id),
                        "class_name": _class_name(det.class_id),
                        "score": round(float(det.score), 4),
                        "bbox": [round(float(v), 1) for v in det.bbox],
                    }
                    for det in detections
                ],
                "timing_ms": _rounded_timing(timing),
            }
            line = json.dumps(item, ensure_ascii=False)
            if jsonl_file is not None:
                jsonl_file.write(line + "\n")
            if args.progress_interval > 0 and processed % int(args.progress_interval) == 0:
                print("frame_result", line)
            if writer is not None:
                _draw_detections(frame, detections)
                writer.write(frame)

            processed += 1
            frame_index += 1
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
        if writer is not None:
            writer.release()
        detector.release()
        cap.release()

    elapsed = max(1e-9, time.time() - t0)
    print(
        "summary",
        json.dumps(
            {
                "processed": processed,
                "frames_with_detection": frames_with_detection,
                "frames_with_person": frames_with_person,
                "class_counts": {
                    _class_name(class_id): count for class_id, count in sorted(class_counter.items())
                },
                "elapsed_sec": round(elapsed, 3),
                "processed_fps": round(processed / elapsed, 3),
                "avg_timing_ms": _average_timing(timing_sums, processed),
                "max_timing_ms": _rounded_timing(timing_max),
            },
            ensure_ascii=False,
        ),
    )
    return 0


def _draw_detections(frame, detections) -> None:
    try:
        import cv2
    except Exception:
        return
    for det in detections:
        x1, y1, x2, y2 = [int(round(float(v))) for v in det.bbox]
        class_id = int(det.class_id)
        color = (0, 255, 0) if class_id == 0 else (0, 180, 255)
        label = f"{_class_name(class_id)} {float(det.score):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_text = max(0, y1 - th - baseline - 4)
        cv2.rectangle(frame, (x1, y_text), (x1 + tw + 6, y_text + th + baseline + 4), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, y_text + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )


def _class_name(class_id: int) -> str:
    idx = int(class_id)
    if 0 <= idx < len(COCO80):
        return COCO80[idx]
    return f"class_{idx}"


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
