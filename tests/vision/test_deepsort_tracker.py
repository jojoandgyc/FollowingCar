#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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

    # DeepSORT associates on a 1.20x expanded box.  Near an image edge that
    # display/association box may be clipped even though the source YOLO box
    # is a valid person crop.  Quality for identity evidence must follow the
    # source detector box, not the expanded display box.
    detector_bbox = (260.0, 80.0, 380.0, 400.0)
    tracker._current_detections = (Detection(detector_bbox, 0.92, 0),)
    tracker._frame_index = 10
    expanded_output = SimpleNamespace(
        track_id=7,
        reid_uid=0,
        x1=0.0,
        y1=0.0,
        x2=640.0,
        y2=480.0,
        class_id=0,
        confidence=0.92,
        feature=feature,
        state=2,
        time_since_update=0,
        source_detection_index=0,
    )
    tracker._to_record(
        expanded_output,
        640,
        480,
        1,
        partial_features=[None],
    )
    detector_assignment = tracker.identity_bank.last_assignments.get(7, {})
    if not detector_assignment.get("bbox_quality_ok"):
        raise AssertionError(
            "a valid detector bbox must not be rejected by the clipped "
            f"expanded track bbox: {detector_assignment}"
        )
    if detector_assignment.get("bbox_quality_tier") != "strong":
        raise AssertionError(
            "detector-backed identity evidence should be strong, got "
            f"{detector_assignment}"
        )

    good, reason = tracker._bbox_quality((820, 10, 1260, 1070), 1920, 1080)
    if not good:
        raise AssertionError(f"standing close person-shaped bbox should be accepted, got {reason}")
    good, reason = tracker._bbox_quality((230.9, 0.4, 1602.2, 1071.0), 1920, 1080)
    if not good:
        raise AssertionError(f"near-camera person bbox should be accepted, got {reason}")
    good, reason = tracker._bbox_quality((100, 0, 1850, 1080), 1920, 1080)
    if good or "area_ratio" not in reason:
        raise AssertionError(f"near-fullscreen hand/occlusion bbox should be rejected, got {reason}")

    strict_tracker = DeepSortTracker(
        DeepSortTrackerConfig(
            identity_min_area=900.0,
            identity_min_width_px=16.0,
            identity_min_height_px=28.0,
            identity_max_single_frame_area_shrink_ratio=0.30,
            identity_area_shrink_max_gap_frames=2,
        )
    )
    good, reason = strict_tracker._bbox_quality((100, 100, 122, 121), 640, 480)
    if good or "area<900" not in reason or "height<28" not in reason:
        raise AssertionError(f"20px-class false person box must be rejected, got {reason}")
    strict_tracker._frame_index = 100
    transition_ok, reason = strict_tracker._bbox_transition_quality(
        7,
        67000.0,
        is_fresh=True,
        base_quality_ok=True,
    )
    if not transition_ok:
        raise AssertionError(f"reliable baseline should be accepted: {reason}")
    strict_tracker._frame_index = 101
    transition_ok, reason = strict_tracker._bbox_transition_quality(
        7,
        451.0,
        is_fresh=True,
        base_quality_ok=True,
    )
    if transition_ok or reason != "area_shrink<0.30":
        raise AssertionError(f"one-frame 67000->451 collapse must be rejected: {reason}")
    if strict_tracker._last_quality_area_by_track_id[7][1] != 67000.0:
        raise AssertionError("a rejected fragment must not replace the reliable area baseline")

    narrow_bbox = (0.0, 20.0, 50.0, 420.0)
    narrow_ok, narrow_reason = strict_tracker._bbox_quality(narrow_bbox, 640, 480)
    narrow_tier = strict_tracker._bbox_identity_tier(
        bbox_quality_ok=narrow_ok,
        bbox_quality_reason=narrow_reason,
        confidence=0.90,
        bbox=narrow_bbox,
    )
    if narrow_ok or narrow_tier != "weak" or "aspect<" not in narrow_reason:
        raise AssertionError(f"large narrow/edge person evidence should be weak, got {narrow_tier}: {narrow_reason}")
    tiny_tier = strict_tracker._bbox_identity_tier(
        bbox_quality_ok=False,
        bbox_quality_reason="area<900,width<16,height<28,aspect<0.18",
        confidence=0.90,
        bbox=(600.0, 20.0, 612.0, 42.0),
    )
    if tiny_tier != "reject":
        raise AssertionError(f"tiny narrow fragments must remain rejected, got {tiny_tier}")
    duplicate_tier = strict_tracker._bbox_identity_tier(
        bbox_quality_ok=False,
        bbox_quality_reason="duplicate_person_box",
        confidence=0.90,
        bbox=narrow_bbox,
    )
    if duplicate_tier != "reject":
        raise AssertionError(f"duplicate fragments must remain rejected, got {duplicate_tier}")

    mapped = SimpleNamespace(
        track_id=1,
        time_since_update=0,
        x1=892.7,
        y1=0.0,
        x2=1919.0,
        y2=1079.0,
    )
    fragment = SimpleNamespace(
        track_id=5,
        time_since_update=0,
        x1=588.5,
        y1=154.0,
        x2=1120.5,
        y2=1079.0,
    )
    separate = SimpleNamespace(
        track_id=6,
        time_since_update=0,
        x1=50.0,
        y1=140.0,
        x2=460.0,
        y2=1060.0,
    )
    tracker.identity_bank.track_to_uid[1] = 1
    suppressed = tracker._duplicate_identity_track_ids([mapped, fragment, separate], 1920, 1080)
    if suppressed != {5}:
        raise AssertionError(f"only the overlapping unmapped fragment should be suppressed, got {suppressed}")

    fragment.feature = _unit_feature = np.zeros(512, dtype=np.float32)
    _unit_feature[0] = 1.0
    fragment.confidence = 0.9
    fragment.class_id = 0
    fragment.state = 2
    tracker._frame_index = 2
    identity_count_before = len(tracker.identity_bank.identities)
    fragment_record = tracker._to_record(
        fragment,
        1920,
        1080,
        1,
        partial_features=[],
        duplicate_identity_box=True,
    )
    if fragment_record.reid_uid != 0 or len(tracker.identity_bank.identities) != identity_count_before:
        raise AssertionError("an overlapping unmapped fragment must not create a new uid")
    fragment_assignment = tracker.identity_bank.last_assignments.get(5, {})
    if fragment_assignment.get("bbox_quality_reason") != "duplicate_person_box":
        raise AssertionError(f"expected duplicate_person_box reason, got {fragment_assignment}")

    tracker.identity_bank.track_to_uid[5] = 2
    suppressed = tracker._duplicate_identity_track_ids([mapped, fragment], 1920, 1080)
    if suppressed:
        raise AssertionError(f"an already mapped track must never be duplicate-suppressed, got {suppressed}")

    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction="right")
    if not tracker._search_candidate_direction_ok(0.56 * 1920, 1920):
        raise AssertionError("a right-side search candidate should be eligible")
    if not tracker._search_candidate_direction_ok(0.49 * 1920, 1920):
        raise AssertionError("a center-crossing candidate should be eligible for ReID confirmation")
    if tracker._search_candidate_direction_ok(0.44 * 1920, 1920):
        raise AssertionError("a left-side candidate must not be eligible during right search")
    tracker._search_reacquire_eligible_tracks.add(9)
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction="right")
    if 9 not in tracker._search_reacquire_eligible_tracks:
        raise AssertionError("the same search context must retain a consecutive candidate track")
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction="left")
    if tracker._search_reacquire_eligible_tracks:
        raise AssertionError("changing search direction must clear prior candidate eligibility")
    tracker.set_search_reacquire_context(active_uid=1, searching=False, direction="right")
    if tracker._search_candidate_direction_ok(0.90 * 1920, 1920):
        raise AssertionError("preferred reacquire must be disabled outside search")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
