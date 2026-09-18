"""Resolve a confirmed target to its own fresh YOLO depth-sampling geometry.

This module performs an exact association, not target selection or tracking.
In particular it never substitutes the nearest person for a missing match.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Optional

from .control_types import BBox, DepthTargetObservation


_BBOX_ABS_TOL = 1e-6
_CONFIRMED_MAPPED_REASONS = frozenset({
    "controlled_handoff", "preferred_search_reacquire",
    "preferred_search_soft_reacquire", "preferred_search_late_reacquire",
    "weak_preferred_reacquire_confirmed",
})
_OBSERVATION_REASONS = frozenset({
    "mapped_weak_observed", "mapped_weak_no_feature", "mapped_low_quality",
    "mapped_low_confidence_strong_observation", "weak_match_requires_handoff",
    "weak_bbox_unassigned", "unassigned", "mapped_missing_identity",
})


def _number(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value) -> Optional[int]:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _valid_raw_track(value: Optional[int]) -> bool:
    # -1 is the tracker module's reserved detector-only search track. Once
    # the bank has confirmed its public UID it need not wait for DeepSORT.
    return value is not None and (value == -1 or value > 0)


def _bbox(value, width: int, height: int) -> Optional[BBox]:
    if isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        values = tuple(_number(item) for item in value)
    except TypeError:
        return None
    if len(values) != 4 or any(item is None for item in values):
        return None
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        return None
    return values


def resolve_depth_target_observation(
    *, target_id, display_bbox, capture_frame_id, capture_timestamp,
    observations, width, height, allow_confirmed_mapped: bool = False,
    expected_raw_track_id=None,
) -> Optional[DepthTargetObservation]:
    """Return immutable detector geometry only for one proven UID association.

    Camera age and depth freshness remain the runtime's responsibility. This
    resolver only requires identical capture provenance; it does not compare
    an old capture timestamp against the current wall clock.

    ``allow_confirmed_mapped`` is observation-only support for an explicitly
    completed bank handoff. It still cannot promote an edge/partial/low-quality
    direction-only record into a longitudinal target.
    """
    uid = _integer(target_id)
    capture_id = _integer(capture_frame_id)
    timestamp = _number(capture_timestamp)
    frame_width, frame_height = _integer(width), _integer(height)
    expected_raw = _integer(expected_raw_track_id)
    if (
        uid is None or uid <= 0 or capture_id is None or capture_id <= 0
        or timestamp is None or timestamp <= 0
        or frame_width is None or frame_width <= 0
        or frame_height is None or frame_height <= 0
        or (expected_raw_track_id is not None and not _valid_raw_track(expected_raw))
    ):
        return None
    target_bbox = _bbox(display_bbox, frame_width, frame_height)
    if target_bbox is None:
        return None
    try:
        records = tuple(observations)
    except TypeError:
        return None
    matches = []
    for observation in records:
        if not isinstance(observation, Mapping):
            continue
        candidate_bbox = _bbox(observation.get("display_bbox"), frame_width, frame_height)
        if candidate_bbox is not None and all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=_BBOX_ABS_TOL)
            for a, b in zip(candidate_bbox, target_bbox)
        ):
            matches.append(observation)
    # UID, a preferred raw ID or YOLO score cannot break a geometry tie. Two
    # records representing this display box are an ambiguous association.
    if len(matches) != 1:
        return None
    observation = matches[0]
    metadata, assignment = observation.get("sample_metadata"), observation.get("assignment")
    if not isinstance(metadata, Mapping) or not isinstance(assignment, Mapping):
        return None
    raw_track_id = _integer(observation.get("raw_track_id"))
    source_index = _integer(metadata.get("source_detection_index"))
    if (
        not _valid_raw_track(raw_track_id)
        or (expected_raw is not None and raw_track_id != expected_raw)
        or source_index is None or source_index < 0
        or metadata.get("is_fresh") is not True
        or _integer(metadata.get("capture_frame_id")) != capture_id
        or _number(metadata.get("capture_timestamp")) != timestamp
    ):
        return None
    # One current raw track/source detection must not point to two records,
    # even when their expanded display boxes happen to differ.
    for other in records:
        if other is observation or not isinstance(other, Mapping):
            continue
        if _integer(other.get("raw_track_id")) == raw_track_id:
            return None
        other_metadata = other.get("sample_metadata")
        if isinstance(other_metadata, Mapping) and (
            _integer(other_metadata.get("capture_frame_id")) == capture_id
            and _number(other_metadata.get("capture_timestamp")) == timestamp
            and _integer(other_metadata.get("source_detection_index")) == source_index
        ):
            return None
    bbox = _bbox(observation.get("detector_bbox"), frame_width, frame_height)
    if bbox is None:
        return None
    reason = str(assignment.get("reason") or "").strip().lower()
    if (
        assignment.get("bbox_quality_ok") is not True
        or str(assignment.get("bbox_quality_tier") or "").lower() in {"weak", "reject"}
        or assignment.get("reacquire_geometry_ok") is False
        or assignment.get("search_excluded") is True
        or metadata.get("search_observation_only") is True
        or assignment.get("search_observation_only") is True
        or reason in _OBSERVATION_REASONS
        or any(token in reason for token in ("pending", "wait", "reject", "excluded"))
    ):
        return None
    observed_uid = _integer(observation.get("uid"))
    assigned_uid = _integer(assignment.get("uid"))
    mapped_uid = _integer(assignment.get("mapped_uid"))
    if raw_track_id == -1 and observed_uid != uid:
        # A probe's tentative mapping is not a confirmed public identity,
        # even when the optional mapped-observation route is requested.
        return None
    if mapped_uid is not None and mapped_uid > 0 and mapped_uid != uid:
        return None
    if observed_uid == uid:
        if "uid" in assignment and assigned_uid != uid:
            return None
    elif not (
        allow_confirmed_mapped and observed_uid == 0 and mapped_uid == uid
        and assigned_uid in (0, uid) and reason in _CONFIRMED_MAPPED_REASONS
    ):
        return None
    return DepthTargetObservation(
        bbox=bbox, target_id=uid, raw_track_id=raw_track_id,
        capture_frame_id=capture_id, capture_timestamp=timestamp, source="yolo_detector",
    )
