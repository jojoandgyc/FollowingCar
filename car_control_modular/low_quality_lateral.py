"""Bounded yaw evidence for an already mapped crop in a multi-person frame.

This module never assigns identities or changes a feature gallery. The caller
owns wall-clock vision freshness and motor safety; a result grants only a
short-lived lateral observation for the existing active UID.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class MultiPersonLateralDecision:
    track_id: Optional[int]
    reason: str
    streak: int
    distance: Optional[float] = None
    runner_up_distance: Optional[float] = None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> Optional[int]:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _distance(value: Any) -> Optional[float]:
    number = _number(value)
    return number if number is not None and number >= 0.0 else None


def _strong_distance_to_uid(assignment: Mapping[str, Any], active_uid: int) -> Optional[float]:
    """Compare identities only when the distance explicitly refers to this UID."""
    distances = []
    if _integer(assignment.get("best_uid")) == active_uid:
        strong = _distance(assignment.get("strong_distance"))
        if strong is not None:
            distances.append(strong)
        elif assignment.get("match_source") == "strong":
            value = _distance(assignment.get("distance"))
            if value is not None:
                distances.append(value)
    evidence = assignment.get("match_evidence")
    if isinstance(evidence, Mapping) and _integer(evidence.get("matched_uid")) == active_uid:
        strong = _distance(evidence.get("strong_distance"))
        if strong is not None:
            distances.append(strong)
        elif evidence.get("match_source") == "strong":
            value = _distance(evidence.get("distance"))
            if value is not None:
                distances.append(value)
    # If two diagnostics exist, use the closer competitor. Missing evidence
    # cannot be treated as an infinitely distant identity.
    return min(distances) if distances else None


class MultiPersonLateralGate:
    def __init__(
        self,
        *,
        max_distance: float = 0.15,
        min_distance_margin: float = 0.08,
        max_gap_sec: float = 0.35,
        max_center_jump_ratio: float = 0.22,
        min_area_similarity: float = 0.45,
    ) -> None:
        self.max_distance = float(max_distance)
        self.min_distance_margin = float(min_distance_margin)
        self.max_gap_sec = float(max_gap_sec)
        self.max_center_jump_ratio = float(max_center_jump_ratio)
        self.min_area_similarity = float(min_area_similarity)
        self.reset()

    def reset(self) -> None:
        self._seen_capture_id: Optional[int] = None
        self._seen_capture_timestamp: Optional[float] = None
        self._clear_streak()

    def _clear_streak(self) -> None:
        self._uid: Optional[int] = None
        self._track_id: Optional[int] = None
        self._timestamp: Optional[float] = None
        self._center: Optional[float] = None
        self._area: Optional[float] = None
        self._streak = 0

    def _reject(
        self, reason: str, distance: Optional[float] = None,
        runner_up_distance: Optional[float] = None,
    ) -> MultiPersonLateralDecision:
        self._clear_streak()
        return MultiPersonLateralDecision(None, reason, 0, distance, runner_up_distance)

    def update(
        self,
        *,
        active_uid: Optional[int],
        capture_frame_id: int,
        capture_timestamp: float,
        width: int,
        height: int,
        observations: Sequence[Mapping[str, Any]],
    ) -> MultiPersonLateralDecision:
        uid = _integer(active_uid)
        capture_id = _integer(capture_frame_id)
        timestamp = _number(capture_timestamp)
        frame_width, frame_height = _number(width), _number(height)
        if uid is None or uid <= 0:
            return self._reject("no_active_uid")
        if (
            capture_id is None or capture_id <= 0 or timestamp is None or timestamp <= 0.0
            or frame_width is None or frame_width <= 0.0
            or frame_height is None or frame_height <= 0.0
        ):
            return self._reject("invalid_capture")
        if self._seen_capture_id is not None:
            if capture_id == self._seen_capture_id and timestamp == self._seen_capture_timestamp:
                # Repeating a detector result grants no new output or proof.
                return MultiPersonLateralDecision(None, "duplicate_capture", self._streak)
            if capture_id <= self._seen_capture_id or timestamp <= self._seen_capture_timestamp:
                return self._reject("out_of_order_capture")
        self._seen_capture_id, self._seen_capture_timestamp = capture_id, timestamp

        if not observations or len(observations) < 2:
            return self._reject("not_multi_person")
        source_indices = set()
        raw_track_ids = set()
        parsed = []
        for observation in observations:
            if not isinstance(observation, Mapping):
                return self._reject("invalid_observation")
            metadata = observation.get("sample_metadata")
            assignment = observation.get("assignment")
            if not isinstance(metadata, Mapping) or not isinstance(assignment, Mapping):
                return self._reject("missing_observation_evidence")
            if (
                metadata.get("is_fresh") is not True
                or _integer(metadata.get("capture_frame_id")) != capture_id
                or _number(metadata.get("capture_timestamp")) != timestamp
            ):
                return self._reject("stale_observation")
            count = _integer(metadata.get("candidate_count"))
            source_index = _integer(metadata.get("source_detection_index"))
            track_id = _integer(observation.get("raw_track_id"))
            if count != len(observations) or source_index is None or source_index < 0:
                return self._reject("incomplete_detection_coverage")
            if source_index in source_indices or track_id is None or track_id <= 0 or track_id in raw_track_ids:
                return self._reject("duplicate_observation")
            source_indices.add(source_index)
            raw_track_ids.add(track_id)
            parsed.append((track_id, observation, assignment))

        mapped = [item for item in parsed if _integer(item[2].get("mapped_uid")) == uid]
        if len(mapped) != 1:
            return self._reject("ambiguous_mapping" if mapped else "no_mapped_candidate")
        track_id, candidate, assignment = mapped[0]
        for output_uid in (candidate.get("uid"), assignment.get("uid")):
            value = _integer(output_uid)
            if value is None or value not in (0, uid):
                return self._reject("conflicting_output_uid")
        reasons = []
        for reason in str(assignment.get("bbox_quality_reason") or "").split(","):
            reason = reason.strip()
            while reason.startswith("detector_crop:"):
                reason = reason[len("detector_crop:"):].strip()
            if reason:
                reasons.append(reason)
        if (
            assignment.get("reason") != "mapped_weak_observed"
            or assignment.get("bbox_quality_tier") != "weak"
            or assignment.get("bbox_quality_ok") is not False
            or not reasons or not all(reason.startswith("edge_touch>") for reason in reasons)
        ):
            return self._reject("unsafe_crop_quality")
        distance = _distance(assignment.get("strong_distance"))
        if assignment.get("match_source") != "strong" or distance is None or distance > self.max_distance:
            return self._reject("weak_identity_evidence", distance)
        competitor_distances = []
        for other_track, _other, other_assignment in parsed:
            if other_track == track_id:
                continue
            if _integer(_other.get("uid")) == uid or _integer(other_assignment.get("uid")) == uid:
                return self._reject("ambiguous_mapping", distance)
            other_distance = _strong_distance_to_uid(other_assignment, uid)
            if other_distance is None:
                return self._reject("unknown_competitor_distance", distance)
            competitor_distances.append(other_distance)
        runner_up = min(competitor_distances)
        if runner_up - distance < self.min_distance_margin - 1e-9:
            return self._reject("identity_margin_too_small", distance, runner_up)

        bbox = candidate.get("detector_bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return self._reject("invalid_geometry", distance, runner_up)
        values = tuple(_number(value) for value in bbox)
        if any(value is None for value in values):
            return self._reject("invalid_geometry", distance, runner_up)
        x1, y1, x2, y2 = values
        if not (0.0 <= x1 < x2 <= frame_width and 0.0 <= y1 < y2 <= frame_height):
            return self._reject("invalid_geometry", distance, runner_up)
        center = (x1 + x2) / (2.0 * frame_width)
        area = (x2 - x1) * (y2 - y1) / (frame_width * frame_height)
        width_ratio, height_ratio = (x2 - x1) / frame_width, (y2 - y1) / frame_height
        if (width_ratio >= 0.90 and height_ratio >= 0.82) or (area >= 0.82 and height_ratio >= 0.88):
            return self._reject("near_camera_occlusion", distance, runner_up)
        same_track = self._uid == uid and self._track_id == track_id
        if same_track:
            if timestamp - self._timestamp > self.max_gap_sec:
                return self._reject("capture_gap", distance, runner_up)
            if (
                abs(center - self._center) > self.max_center_jump_ratio
                or min(area, self._area) / max(area, self._area) < self.min_area_similarity
            ):
                return self._reject("geometry_discontinuity", distance, runner_up)
        self._streak = min(2, self._streak + 1) if same_track else 1
        self._uid, self._track_id = uid, track_id
        self._timestamp, self._center, self._area = timestamp, center, area
        confirmed = self._streak >= 2
        return MultiPersonLateralDecision(
            track_id if confirmed else None,
            "confirmed" if confirmed else "confirming",
            self._streak, distance, runner_up,
        )
