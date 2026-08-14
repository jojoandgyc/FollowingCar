#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from rk_vision.yolo11 import LetterboxInfo, YOLO11Config, postprocess_yolo11_outputs


def main() -> int:
    cfg = YOLO11Config(
        model_path="dummy.rknn",
        input_size=640,
        conf_threshold=0.25,
        nms_threshold=0.45,
        num_classes=2,
        output_box_format="xywh",
    )
    letterbox = LetterboxInfo(src_width=640, src_height=640, input_size=640, scale=1.0, pad_x=0.0, pad_y=0.0)
    pred = np.zeros((1, 6, 3), dtype=np.float32)
    pred[0, :4, 0] = [320.0, 320.0, 100.0, 80.0]
    pred[0, 4, 0] = 0.10
    pred[0, 5, 0] = 0.90

    detections = postprocess_yolo11_outputs([pred], cfg, letterbox)
    print("detections", detections)
    if len(detections) != 1:
        raise AssertionError(f"expected one detection, got {len(detections)}")
    det = detections[0]
    if int(det.class_id) != 1:
        raise AssertionError(f"expected class 1 from second class score, got {det.class_id}")
    if abs(float(det.score) - 0.90) > 1e-6:
        raise AssertionError(f"expected score 0.90, got {det.score}")
    if det.bbox != (270.0, 280.0, 370.0, 360.0):
        raise AssertionError(f"unexpected bbox: {det.bbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
