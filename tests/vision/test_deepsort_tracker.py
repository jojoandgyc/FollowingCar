#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig, TRACK_STATE_STABLE
from rk_vision.yolo11 import Detection


def main() -> int:
    tracker = DeepSortTracker(
        DeepSortTrackerConfig(
            n_init=2,
            min_confidence=0.1,
            max_age=5,
            max_iou_distance=0.7,
            max_cosine_distance=0.3,
        )
    )
    feature = np.ones(512, dtype=np.float32)
    feature /= np.linalg.norm(feature)

    records = []
    for idx in range(3):
        det = Detection((100 + idx * 3, 80, 180 + idx * 3, 260), 0.9, 0)
        records = tracker.update([det], [feature], image_width=640)
        print(idx, [(rec.track_id, rec.tracker_state) for rec in records])

    if len(records) != 1:
        raise AssertionError(f"expected one confirmed track, got {len(records)}")
    rec = records[0]
    if rec.track_id != 1 or rec.tracker_state != TRACK_STATE_STABLE:
        raise AssertionError(f"unexpected track record: {rec}")

    good, reason = tracker._bbox_quality((820, 10, 1260, 1070), 1920, 1080)
    if not good:
        raise AssertionError(f"standing close person-shaped bbox should be accepted, got {reason}")
    good, reason = tracker._bbox_quality((230.9, 0.4, 1602.2, 1071.0), 1920, 1080)
    if not good:
        raise AssertionError(f"near-camera person bbox should be accepted, got {reason}")
    good, reason = tracker._bbox_quality((100, 0, 1850, 1080), 1920, 1080)
    if good or "area_ratio" not in reason:
        raise AssertionError(f"near-fullscreen hand/occlusion bbox should be rejected, got {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
