#!/usr/bin/env python3
"""Regression tests for strong ReID observed during rotate settling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import request_0513_modular as runtime


def _settling_tracker(*, geometry_ok: bool, instant_allowed: bool):
    tracker = object.__new__(runtime.PersonTracker)
    tracker.frame_index = 12
    tracker.search_state = "searching"
    tracker.search_direction = "right"
    tracker._confirmed_search_reacquire_uid = None
    tracker._confirmed_search_reacquire_track_id = None
    tracker._confirmed_search_reacquire_bbox = None
    tracker._confirmed_search_reacquire_last_frame = -1
    tracker._confirmed_search_reacquire_streak = 0
    tracker._follow_controller = SimpleNamespace(
        search_state="searching",
        search_direction="right",
        search_status=lambda _now: SimpleNamespace(
            state="searching", active_target_id=1, direction="right"
        ),
        release_search_on_confirmed_target=lambda _reason: setattr(
            tracker, "released", True
        ),
    )
    tracker._rknn_pipeline = SimpleNamespace(
        tracker=SimpleNamespace(
            config=SimpleNamespace(
                identity_preferred_search_reacquire_instant_threshold=0.15
            )
        )
    )
    tracker._identity_assignment_debug_for_track = lambda _track_id: {
        "best_uid": 1,
        "distance": 0.10,
        "match_source": "strong",
        "bbox_quality_ok": True,
        "reacquire_geometry_ok": geometry_ok,
        "instant_reacquire_allowed": instant_allowed,
    }
    tracker._track_record_bbox = lambda rec: (
        float(rec.x1),
        float(rec.y1),
        float(rec.x2),
        float(rec.y2),
    )
    tracker._bbox_iou_xyxy = lambda _previous, _current: 1.0
    tracker._start_visual_reacquire_hold = lambda *_args, **_kwargs: None
    tracker.released = False
    return tracker


def _record():
    return SimpleNamespace(
        class_id=runtime.PERSON_CLASS_ID,
        score=0.90,
        time_since_update=0,
        track_id=7,
        x1=400.0,
        y1=40.0,
        x2=560.0,
        y2=440.0,
    )


def test_strong_instant_reid_releases_settling_without_yaw_gate():
    tracker = _settling_tracker(geometry_ok=True, instant_allowed=True)

    assert tracker._observe_search_settling_reid(
        [_record()], width=640, height=480
    )
    assert tracker.released is True
    assert tracker.search_state == "none"
    assert tracker._reacquire_depth_pending is True


def test_low_distance_geometry_reject_stays_on_two_frame_observation():
    tracker = _settling_tracker(geometry_ok=False, instant_allowed=False)

    assert tracker._observe_search_settling_reid(
        [_record()], width=640, height=480
    )
    assert tracker.released is False
    assert tracker._confirmed_search_reacquire_streak == 1

