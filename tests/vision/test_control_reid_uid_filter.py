#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import request_0513_modular as mod
from rk_vision.tracker import TrackRecord


def _track(track_id: int, reid_uid: int) -> TrackRecord:
    return TrackRecord(
        track_id=track_id,
        reid_uid=reid_uid,
        x1=10.0,
        y1=20.0,
        x2=110.0,
        y2=220.0,
        class_id=mod.PERSON_CLASS_ID,
        score=0.9,
        cx=60.0,
        cy=120.0,
        area=20000.0,
        angle_deg=0.0,
        tracker_state=2,
        time_since_update=0,
    )


class _DummyTracker:
    frame_index = 1

    def __init__(self) -> None:
        self.queued_persons = None
        self._last_person_reid_debug_by_stable_id = {}
        self._bunker_runtime = SimpleNamespace(check_merged_dets=lambda *_args: None)

    _stable_id_from_track_record = staticmethod(mod.PersonTracker._stable_id_from_track_record)

    def _identity_assignment_debug_for_track(self, _track_id: int):
        return {}

    def _handle_hazard_safety_state(self, _state) -> bool:
        return False

    def _queue_actions_for_persons(self, _width: int, _height: int, persons) -> None:
        self.queued_persons = list(persons)


def _consume(records, *, reid_enable: bool):
    old_reid_enable = mod.VISION_REID_ENABLE
    mod.VISION_REID_ENABLE = reid_enable
    try:
        dummy = _DummyTracker()
        mod.PersonTracker._consume_track_records(dummy, records, 640, 480, "test")
        return dummy.queued_persons
    finally:
        mod.VISION_REID_ENABLE = old_reid_enable


def main() -> int:
    if _consume([_track(2, 0)], reid_enable=True) != []:
        raise AssertionError("ReID control mode must not accept uid=0 fallback tracks")

    confirmed = _consume([_track(2, 7)], reid_enable=True)
    if len(confirmed) != 1 or int(confirmed[0][1]) != 7:
        raise AssertionError(f"confirmed ReID uid should be accepted, got {confirmed}")

    fallback = _consume([_track(2, 0)], reid_enable=False)
    if len(fallback) != 1 or int(fallback[0][1]) != -3:
        raise AssertionError(f"non-ReID mode should keep raw-track fallback id, got {fallback}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
