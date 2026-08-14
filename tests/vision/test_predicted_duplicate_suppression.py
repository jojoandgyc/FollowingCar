#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rk_vision.pipeline import RKNNVisionPipeline, _is_predicted_duplicate
from rk_vision.tracker import TrackRecord


def _track(track_id: int, bbox, *, time_since_update: int, reid_uid: int = 1) -> TrackRecord:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return TrackRecord(
        track_id=track_id,
        reid_uid=reid_uid,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        class_id=0,
        score=0.9,
        cx=(x1 + x2) / 2.0,
        cy=(y1 + y2) / 2.0,
        area=max(0.0, x2 - x1) * max(0.0, y2 - y1),
        angle_deg=0.0,
        tracker_state=2,
        time_since_update=time_since_update,
    )


def main() -> int:
    predicted = _track(1, (100, 100, 300, 300), time_since_update=3)
    updated_overlap = _track(2, (150, 100, 350, 300), time_since_update=0)
    updated_far = _track(3, (400, 100, 550, 300), time_since_update=0)

    if not _is_predicted_duplicate(predicted, updated_overlap, 0.30, 0.45):
        raise AssertionError("overlapping predicted track should be treated as duplicate")
    if _is_predicted_duplicate(predicted, updated_far, 0.30, 0.45):
        raise AssertionError("non-overlapping predicted track should be kept")

    dummy = type("DummyPipeline", (), {"logger": None})()
    duplicate_a = _track(10, (10, 10, 100, 200), time_since_update=0, reid_uid=7)
    duplicate_b = _track(11, (300, 10, 390, 200), time_since_update=0, reid_uid=7)
    unique = _track(12, (500, 10, 590, 200), time_since_update=0, reid_uid=8)
    filtered = RKNNVisionPipeline._suppress_duplicate_reid_uids(dummy, [duplicate_a, duplicate_b, unique])
    uid_by_track = {int(rec.track_id): int(rec.reid_uid) for rec in filtered}
    if uid_by_track != {10: 0, 11: 0, 12: 8}:
        raise AssertionError(f"duplicate reid uid should be suppressed, got {uid_by_track}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
