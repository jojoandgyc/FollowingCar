#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rk_vision.yolo11 import YOLO11Config, YOLO11RKNNDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RKNN YOLO person detections on a COCO holdout subset.")
    parser.add_argument("--subset-json", required=True, help="JSON created by the COCO holdout sampler.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default=os.environ.get("RKNN_BACKEND", "auto"))
    parser.add_argument("--core-mask", default=os.environ.get("RKNN_CORE_MASK", "auto"))
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--min-gt-area-ratio", type=float, default=0.015)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    try:
        import cv2
    except Exception as exc:
        raise SystemExit(f"OpenCV is required: {exc}") from exc

    subset_path = Path(args.subset_json)
    image_dir = Path(args.image_dir)
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    items = list(subset.get("items", []))
    if args.max_images > 0:
        items = items[: int(args.max_images)]

    detector = YOLO11RKNNDetector(
        YOLO11Config(
            model_path=args.model,
            input_size=int(args.input_size),
            conf_threshold=float(args.conf_threshold),
            nms_threshold=float(args.nms_threshold),
            num_classes=80,
            input_format="RGB",
            output_box_format="xywh",
            core_mask=args.core_mask,
            backend=args.backend,
        )
    )
    detector.load()

    jsonl_file = None
    if args.jsonl:
        out_path = Path(args.jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = out_path.open("w", encoding="utf-8")

    total_gt = 0
    matched_gt = 0
    person_images = 0
    person_images_with_det = 0
    bg_images = 0
    bg_images_with_person_fp = 0
    person_fp = 0
    timings: Dict[str, List[float]] = {}
    score_values: List[float] = []
    per_item: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    try:
        for idx, item in enumerate(items):
            image_path = image_dir / item["file_name"]
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"failed to read image: {image_path}")
            gt_boxes = [
                tuple(map(float, gt["bbox"]))
                for gt in item.get("person_boxes", [])
                if float(gt.get("area_ratio", 0.0)) >= float(args.min_gt_area_ratio)
            ]
            person_dets = []
            detections = detector.detect(frame, "BGR")
            for key, value in detector.last_timing_ms.items():
                timings.setdefault(key, []).append(float(value))
            for det in detections:
                if int(det.class_id) != 0:
                    continue
                person_dets.append((tuple(map(float, det.bbox)), float(det.score)))
                score_values.append(float(det.score))

            matches, unmatched_dets = _match_boxes(gt_boxes, [bbox for bbox, _score in person_dets], float(args.iou_threshold))
            total_gt += len(gt_boxes)
            matched_gt += matches
            if gt_boxes:
                person_images += 1
                if person_dets:
                    person_images_with_det += 1
            else:
                bg_images += 1
                if person_dets:
                    bg_images_with_person_fp += 1
            person_fp += unmatched_dets if gt_boxes else len(person_dets)

            row = {
                "index": idx,
                "file_name": item["file_name"],
                "gt_persons": len(gt_boxes),
                "det_persons": len(person_dets),
                "matched_gt": matches,
                "unmatched_person_dets": unmatched_dets if gt_boxes else len(person_dets),
                "detections": [
                    {"bbox": [round(v, 1) for v in bbox], "score": round(score, 4)}
                    for bbox, score in person_dets
                ],
                "timing_ms": {key: round(float(value), 3) for key, value in detector.last_timing_ms.items()},
            }
            per_item.append(row)
            if jsonl_file is not None:
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if idx == 0 or (idx + 1) % 25 == 0:
                print("progress", json.dumps(row, ensure_ascii=False))
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
        detector.release()

    elapsed = time.perf_counter() - t0
    timing_summary = {
        key: {
            "mean": round(mean(values), 3) if values else 0.0,
            "max": round(max(values), 3) if values else 0.0,
        }
        for key, values in timings.items()
    }
    summary = {
        "model": args.model,
        "subset_json": str(subset_path),
        "image_dir": str(image_dir),
        "images": len(items),
        "person_images": person_images,
        "background_images": bg_images,
        "total_gt_persons": total_gt,
        "matched_gt_persons": matched_gt,
        "box_recall_iou": round(matched_gt / total_gt, 4) if total_gt else 0.0,
        "image_recall_any_person": round(person_images_with_det / person_images, 4) if person_images else 0.0,
        "background_fp_image_rate": round(bg_images_with_person_fp / bg_images, 4) if bg_images else 0.0,
        "unmatched_person_detections": person_fp,
        "person_score_mean": round(mean(score_values), 4) if score_values else 0.0,
        "timing_ms": timing_summary,
        "elapsed_sec": round(elapsed, 3),
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
        "min_gt_area_ratio": args.min_gt_area_ratio,
    }
    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("summary", json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _match_boxes(
    gt_boxes: Sequence[Tuple[float, float, float, float]],
    det_boxes: Sequence[Tuple[float, float, float, float]],
    iou_threshold: float,
) -> Tuple[int, int]:
    if not gt_boxes:
        return 0, len(det_boxes)
    used_gt = set()
    matches = 0
    unmatched_dets = 0
    for det in det_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in used_gt:
                continue
            val = _iou(det, gt)
            if val > best_iou:
                best_iou = val
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_idx)
            matches += 1
        else:
            unmatched_dets += 1
    return matches, unmatched_dets


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
