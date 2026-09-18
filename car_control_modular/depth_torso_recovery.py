"""Narrow spatial evidence for bottom-clipped, bounded torso recovery.

This policy does not grant a UID, accept a depth sample, or create a jump
confirmation. The caller owns identity provenance, source freshness, temporal
watermarks, and the two-sample confirmation/continuation state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from .depth_torso_selection import (
    TorsoSelection, _finite_number, _positive_count, _validated_candidate,
)


CORE_TORSO_REGIONS = frozenset({
    "chest_center", "abdomen_center", "left_torso", "right_torso",
})
_REGION_CENTERS = {
    "chest_center": (0.50, 0.30),
    "abdomen_center": (0.50, 0.50),
    "left_torso": (0.36, 0.42),
    "right_torso": (0.64, 0.42),
}


@dataclass(frozen=True)
class TorsoRecoveryEvidence:
    distance_m: float
    sample_timestamp: float
    bbox: tuple[float, float, float, float]
    region_names: tuple[str, ...]
    frame_width: int
    frame_height: int


def _bbox(value, width: int, height: int) -> Optional[tuple]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    values = tuple(_finite_number(item) for item in value)
    if any(item is None for item in values):
        return None
    x1, y1, x2, y2 = values
    return values if 0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height else None


def _core_names(value) -> bool:
    return bool(
        isinstance(value, tuple) and 1 <= len(value) <= 2
        and all(isinstance(name, str) and name in CORE_TORSO_REGIONS for name in value)
        and len(set(value)) == len(value)
    )


def assess_torso_recovery(
    *, selection, bbox, frame_width, frame_height, depth_width, depth_height,
    regions, anchor_distance_m, anchor_age_sec, sample_timestamp,
    max_distance_jump_m, max_anchor_age_sec, edge_margin_ratio,
    large_bbox_area_ratio, large_bbox_height_ratio,
    min_spatial_support_fraction,
) -> tuple[Optional[TorsoRecoveryEvidence], str]:
    """Validate one core-torso observation without weakening normal ROI gates.

    ``regions`` is the existing sampler's sequence of five-item tuples:
    ``(name, left, top, right, bottom)`` in depth-image coordinates. Per-region
    pixels must retain their original spatial validation; aggregate summaries
    alone, lower-abdomen returns and whole-ROI fallbacks cannot qualify.
    """
    dimensions = tuple(_positive_count(value) for value in (
        frame_width, frame_height, depth_width, depth_height,
    ))
    if any(value is None for value in dimensions):
        return None, "invalid_dimensions"
    width, height, depth_w, depth_h = dimensions
    numeric = tuple(_finite_number(value) for value in (
        anchor_distance_m, anchor_age_sec, sample_timestamp,
        max_distance_jump_m, max_anchor_age_sec, edge_margin_ratio,
        large_bbox_area_ratio, large_bbox_height_ratio,
        min_spatial_support_fraction,
    ))
    if any(value is None for value in numeric):
        return None, "invalid_numeric"
    anchor, age, stamp, jump, max_age, margin, area_limit, height_limit, support = numeric
    if not (
        anchor > 0.0 and stamp > 0.0 and age >= 0.0 and jump >= 0.0 and max_age > 0.0
        and 0.0 <= margin < 0.5 and 0.0 < area_limit <= 1.0
        and 0.0 < height_limit <= 1.0 and 0.0 <= support <= 1.0
    ):
        return None, "invalid_numeric"
    if age > min(3.0, max_age):
        return None, "anchor_expired"
    box = _bbox(bbox, width, height)
    if box is None:
        return None, "invalid_bbox"
    x1, y1, x2, y2 = box
    margin_x, margin_y = max(1.0, width * margin), max(1.0, height * margin)
    if not (x1 > margin_x and x2 < width - margin_x and y1 > margin_y
            and y2 >= height - margin_y):
        return None, "not_bottom_only"
    if ((x2 - x1) * (y2 - y1) / (width * height) >= area_limit
            or (y2 - y1) / height >= height_limit):
        return None, "large_bbox"

    if not isinstance(selection, TorsoSelection) or not _core_names(selection.region_names):
        return None, "not_core_torso_selection"
    if not isinstance(selection._region_evidence, tuple):
        return None, "invalid_region_evidence"
    evidence = tuple(_validated_candidate(item, support) for item in selection._region_evidence)
    if not evidence or any(item is None for item in evidence):
        return None, "invalid_region_evidence"
    names = tuple(item.region_name for item in evidence)
    pixels = sum(item.pixels for item in evidence)
    distance = sum(item.distance_m * item.pixels for item in evidence) / pixels
    claimed_distance = _finite_number(selection.distance_m)
    if (
        not _core_names(names) or selection.region_names != names
        or any(_positive_count(value) is None for value in (
            selection.region_count, selection.pixels,
            selection.valid_pixels, selection.required_pixels,
        ))
        or selection.region_count != len(names) or selection.pixels != pixels
        or selection.valid_pixels != sum(item.valid_pixels for item in evidence)
        or selection.required_pixels != max(item.required_pixels for item in evidence)
        or claimed_distance is None or abs(claimed_distance - distance) > 1e-9
    ):
        return None, "invalid_selection_aggregate"
    # Check each contributing region too: an average cannot hide a farther
    # unqualified surface. The fixed limits apply only to this exception.
    if any(item.distance_m > 3.0 for item in evidence):
        return None, "distance_out_of_bounds"
    if any(abs(item.distance_m - anchor) > min(0.6, jump) + 1e-9 for item in evidence):
        return None, "anchor_distance_discontinuity"

    if not isinstance(regions, (tuple, list)):
        return None, "invalid_regions"
    bounds_by_name = {}
    for region in regions:
        if not isinstance(region, (tuple, list)) or len(region) != 5:
            return None, "invalid_regions"
        name = region[0]
        if not isinstance(name, str) or name in bounds_by_name:
            return None, "invalid_regions"
        bounds_by_name[name] = region[1:]
    scale_x, scale_y = depth_w / width, depth_h / height
    for item in evidence:
        bounds = bounds_by_name.get(item.region_name)
        if bounds is None:
            return None, "missing_region_geometry"
        coords = tuple(_finite_number(value) for value in bounds)
        if any(value is None or int(value) != value for value in coords):
            return None, "invalid_region_geometry"
        left, top, right, bottom = coords
        if not (0 < left < right < depth_w and 0 < top < bottom < depth_h
                and x1 * scale_x <= left < right <= x2 * scale_x
                and y1 * scale_y <= top < bottom <= y2 * scale_y):
            return None, "region_not_complete"
        x_ratio, y_ratio = _REGION_CENTERS[item.region_name]
        expected_x = (x1 + (x2 - x1) * x_ratio) * scale_x
        expected_y = (y1 + (y2 - y1) * y_ratio) * scale_y
        # Original sampler rounds each top-left corner by at most half a
        # pixel. A shifted/clamped region is not a complete torso patch.
        if (abs((left + right) / 2.0 - expected_x) > 0.500001
                or abs((top + bottom) / 2.0 - expected_y) > 0.500001):
            return None, "region_shifted"
        if item.valid_pixels > (right - left) * (bottom - top):
            return None, "region_pixel_count_invalid"
    return TorsoRecoveryEvidence(
        distance, stamp, box, names, width, height,
    ), "bottom_only_core_torso"


def torso_evidence_continuous(
    previous, current, *, max_gap_sec, max_distance_delta_m, max_rate_m_s,
) -> bool:
    """Compare adjacent physical samples; never counts or mutates anything."""
    if not isinstance(previous, TorsoRecoveryEvidence) or not isinstance(current, TorsoRecoveryEvidence):
        return False
    limits = tuple(_finite_number(value) for value in (
        max_gap_sec, max_distance_delta_m, max_rate_m_s,
    ))
    if any(value is None for value in limits):
        return False
    gap_limit, delta_limit, rate_limit = limits
    if gap_limit <= 0.0 or delta_limit < 0.0 or rate_limit <= 0.0:
        return False
    boxes = []
    values = []
    for item in (previous, current):
        width, height = _positive_count(item.frame_width), _positive_count(item.frame_height)
        distance, stamp = _finite_number(item.distance_m), _finite_number(item.sample_timestamp)
        if (width is None or height is None or distance is None or stamp is None
                or not 0.0 < distance <= 3.0 or stamp <= 0.0 or not _core_names(item.region_names)):
            return False
        box = _bbox(item.bbox, width, height)
        if box is None:
            return False
        boxes.append(box)
        values.append((distance, stamp))
    if (previous.frame_width != current.frame_width or previous.frame_height != current.frame_height
            or not set(previous.region_names).intersection(current.region_names)):
        return False
    delta_time = values[1][1] - values[0][1]
    delta_distance = abs(values[1][0] - values[0][0])
    if (not 0.0 < delta_time <= min(0.25, gap_limit) + 1e-9
            or delta_distance > delta_limit + 1e-9
            or delta_distance > rate_limit * delta_time + 1e-9):
        return False
    old, new = boxes
    center_shift = math.hypot(
        ((new[0] + new[2]) - (old[0] + old[2])) / (2.0 * current.frame_width),
        ((new[1] + new[3]) - (old[1] + old[3])) / (2.0 * current.frame_height),
    )
    old_area, new_area = ((box[2] - box[0]) * (box[3] - box[1]) for box in boxes)
    return bool(center_shift <= 0.08 + 1e-9 and min(old_area, new_area) / max(old_area, new_area) >= 0.75)
