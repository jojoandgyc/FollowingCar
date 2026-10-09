"""Cheap, deterministic target association used while a target is visible."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ReidCandidate:
    bbox: BBox
    track_id: int
    area: float
    score: float


def iou(left: BBox, right: BBox) -> float:
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - intersection
    return intersection / union if union > 1e-6 else 0.0


def center_distance_ratio(left: BBox, right: BBox, frame_width: int, frame_height: int) -> float:
    lx = (left[0] + left[2]) * 0.5
    ly = (left[1] + left[3]) * 0.5
    rx = (right[0] + right[2]) * 0.5
    ry = (right[1] + right[3]) * 0.5
    scale = max(1.0, float(max(frame_width, frame_height)))
    return (((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5) / scale


def associate(
    candidates: Iterable[ReidCandidate], previous_bbox: Optional[BBox], *, frame_width: int, frame_height: int,
    min_iou: float, max_center_distance_ratio: float,
) -> Optional[ReidCandidate]:
    """Return the geometrically most continuous target, never the largest one.

    The score intentionally ignores appearance. This path executes every
    control frame and protects the control-loop latency from ReID inference.
    """
    values = tuple(candidates)
    if not values:
        return None
    if previous_bbox is None:
        return max(values, key=lambda item: (item.area, item.score))
    ranked = []
    for candidate in values:
        overlap = iou(previous_bbox, candidate.bbox)
        movement = center_distance_ratio(previous_bbox, candidate.bbox, frame_width, frame_height)
        if overlap < min_iou and movement > max_center_distance_ratio:
            continue
        ranked.append((overlap - movement * 0.15, candidate.score, candidate.area, candidate))
    return max(ranked, default=(0.0, 0.0, 0.0, None))[3]
