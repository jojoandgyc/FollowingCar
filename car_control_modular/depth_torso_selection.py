"""Bounded selection of already spatially validated Astra torso clusters.

This module never samples pixels or expands a detector box. Candidates come
from the five existing torso regions; the caller retains all jump, timestamp,
identity and far-background acceptance guards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional


TORSO_REGION_NAMES = frozenset({
    "chest_center", "abdomen_center", "left_torso", "right_torso", "lower_abdomen",
})


@dataclass(frozen=True)
class _RegionEvidence:
    distance_m: float
    pixels: int
    valid_pixels: int
    required_pixels: int
    spatial_support_fraction: float
    region_name: str


@dataclass(frozen=True)
class TorsoSelection:
    distance_m: float
    pixels: int
    valid_pixels: int
    required_pixels: int
    region_names: tuple[str, ...]
    region_count: int
    anchor_consistent: bool
    selection_reason: str
    candidate_summary: str
    # Sparse continuation must use the actual, validated per-region evidence,
    # not a caller-supplied aggregate count or a whole-ROI fallback summary.
    _region_evidence: tuple[_RegionEvidence, ...] = field(default=(), repr=False)


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _positive_count(value: Any) -> Optional[int]:
    number = _finite_number(value)
    if number is None or number < 1 or int(number) != number:
        return None
    return int(number)


def _validated_candidate(candidate: Any, minimum_support: float) -> Optional[_RegionEvidence]:
    read = candidate.get if isinstance(candidate, Mapping) else lambda name: getattr(candidate, name, None)
    distance = _finite_number(read("distance_m"))
    pixels = _positive_count(read("pixels"))
    valid = _positive_count(read("valid_pixels"))
    required = _positive_count(read("required_pixels"))
    spatial = _finite_number(read("spatial_support_fraction"))
    name = read("region_name")
    if (
        distance is None or distance <= 0.0
        or pixels is None or valid is None or required is None
        or not required <= pixels <= valid
        or spatial is None or not minimum_support <= spatial <= 1.0
        or pixels > valid * spatial + 1e-6
        or not isinstance(name, str) or name not in TORSO_REGION_NAMES
    ):
        return None
    return _RegionEvidence(distance, pixels, valid, required, spatial, name)


def _continuity_tolerance(cluster_span_m: float, max_distance_jump_m: float) -> float:
    # Existing grouping tolerance, additionally bounded by the existing jump
    # threshold. No new near/far range or pixel threshold is introduced.
    return min(
        max(0.15, float(cluster_span_m) * 0.75),
        max(0.0, float(max_distance_jump_m)),
    )


def select_torso_candidate_group(
    candidates: Sequence[Any], *,
    anchor_distance_m: Optional[float],
    anchor_age_sec: Optional[float],
    cluster_span_m: float,
    max_distance_jump_m: float,
    anchor_strict_age_sec: float,
    anchor_expire_age_sec: float,
    minimum_spatial_support_fraction: float = 0.55,
) -> Optional[TorsoSelection]:
    """Prefer a genuinely fresh anchor's nearby, locally supported surface.

    Outside the strict anchor window, use the original score and its decaying
    continuity weight. A genuinely closer surface still takes the pre-existing
    asymmetric safety precedence; that alone never grants sparse continuation.
    """
    numeric_config = tuple(_finite_number(value) for value in (
        cluster_span_m, max_distance_jump_m, anchor_strict_age_sec,
        anchor_expire_age_sec, minimum_spatial_support_fraction,
    ))
    if any(value is None for value in numeric_config):
        return None
    minimum_support = max(0.0, min(1.0, float(minimum_spatial_support_fraction)))
    normalized = [valid for item in candidates
                  if (valid := _validated_candidate(item, minimum_support)) is not None]
    if not normalized:
        return None
    tolerance = max(0.15, float(cluster_span_m) * 0.75)
    continuity_tolerance = _continuity_tolerance(cluster_span_m, max_distance_jump_m)
    groups: list[dict[str, Any]] = []
    for candidate in sorted(normalized, key=lambda item: (
        item.distance_m, item.region_name, -item.pixels,
        -item.spatial_support_fraction, -item.valid_pixels, item.required_pixels,
    )):
        group = next((group for group in groups
                      if abs(candidate.distance_m - group["center"]) <= tolerance), None)
        if group is None:
            group = {"regions": {}, "center": candidate.distance_m}
            groups.append(group)
        previous = group["regions"].get(candidate.region_name)
        # Multiple bands/copies from the same torso area are not independent
        # regions and cannot inflate either the spatial count or pixel support.
        if previous is None or candidate.pixels > previous.pixels:
            group["regions"][candidate.region_name] = candidate
        items = tuple(group["regions"].values())
        group["center"] = sum(item.distance_m * item.pixels for item in items) / sum(
            item.pixels for item in items
        )

    reference = _finite_number(anchor_distance_m)
    age = _finite_number(anchor_age_sec)
    if reference is not None and reference <= 0.0:
        reference = None
    if age is not None and age < 0.0:
        age = None
    strict_age = max(0.0, float(anchor_strict_age_sec))
    expire_age = max(strict_age, float(anchor_expire_age_sec))
    fresh_anchor = reference is not None and age is not None and age <= strict_age
    for group in groups:
        items = tuple(group["regions"].values())
        support = sum(item.pixels for item in items)
        continuity = 0.0
        if reference is not None and age is not None and age < expire_age:
            weight = 1.0 if age < strict_age else 0.45
            continuity = weight * max(
                0.0, 1.0 - abs(group["center"] - reference) / max(0.20, float(max_distance_jump_m))
            )
        group["score"] = (
            2.0 * len(items) + math.log1p(support)
            + sum(item.spatial_support_fraction for item in items) / len(items)
            + 4.0 * continuity
        )
        group["support"] = support
        group["anchor_consistent"] = bool(
            fresh_anchor and abs(group["center"] - reference) <= continuity_tolerance
        )
    selected = max(groups, key=lambda group: group["score"])
    reason = "group_consensus"
    anchor_groups = [group for group in groups if group["anchor_consistent"]]
    if anchor_groups:
        selected = max(anchor_groups, key=lambda group: group["score"])
        reason = "fresh_anchor_consensus"
    closest = min(groups, key=lambda group: group["center"])
    if closest["center"] + max(0.15, float(max_distance_jump_m)) < selected["center"]:
        selected = closest
        reason = "nearer_safety_override"

    evidence = tuple(sorted(selected["regions"].values(), key=lambda item: item.region_name))
    summary = "; ".join(
        "%.3fm regions=%s pixels=%d required=%d score=%.2f anchor=%s" % (
            group["center"], "+".join(sorted(group["regions"])), group["support"],
            max(item.required_pixels for item in group["regions"].values()),
            group["score"], group["anchor_consistent"],
        ) for group in groups
    )
    return TorsoSelection(
        distance_m=float(selected["center"]),
        pixels=int(selected["support"]),
        valid_pixels=sum(item.valid_pixels for item in evidence),
        required_pixels=max(item.required_pixels for item in evidence),
        region_names=tuple(item.region_name for item in evidence),
        region_count=len(evidence),
        anchor_consistent=bool(selected["anchor_consistent"]),
        selection_reason=reason,
        candidate_summary=summary,
        _region_evidence=evidence,
    )


def allow_sparse_torso_continuation(
    selection: Optional[TorsoSelection], *,
    anchor_distance_m: Optional[float],
    anchor_age_sec: Optional[float],
    strict_age_sec: float,
    max_near_distance_m: float,
    cluster_span_m: float,
    max_distance_jump_m: float,
    minimum_spatial_support_fraction: float = 0.55,
) -> bool:
    """Authorize local-pixel evidence only for continuous, already near range.

    This is not an alternate acquisition/jump/re-anchor path. A safety-selected
    closer outlier cannot replace the anchor merely by evading the main ROI
    gate. The runtime still owns freshness, held-distance TTL and identity.
    """
    if not isinstance(selection, TorsoSelection) or not selection.anchor_consistent:
        return False
    values = tuple(_finite_number(value) for value in (
        anchor_distance_m, anchor_age_sec, strict_age_sec, max_near_distance_m,
        cluster_span_m, max_distance_jump_m, minimum_spatial_support_fraction,
    ))
    if any(value is None for value in values):
        return False
    anchor, age, strict_age, near_limit, _span, _jump, minimum_support = values
    if not (0.0 < anchor <= near_limit and 0.0 <= age <= max(0.0, strict_age)):
        return False
    evidence = tuple(_validated_candidate(item, max(0.0, min(1.0, minimum_support)))
                     for item in selection._region_evidence)
    if not evidence or any(item is None for item in evidence):
        return False
    names = tuple(item.region_name for item in evidence)
    pixels = sum(item.pixels for item in evidence)
    distance = sum(item.distance_m * item.pixels for item in evidence) / pixels
    claimed_distance = _finite_number(selection.distance_m)
    if (
        len(set(names)) != len(names)
        or selection.region_names != names
        or any(_positive_count(value) is None for value in (
            selection.region_count, selection.pixels,
            selection.valid_pixels, selection.required_pixels,
        ))
        or selection.region_count != len(names)
        or selection.pixels != pixels
        or selection.valid_pixels != sum(item.valid_pixels for item in evidence)
        or selection.required_pixels != max(item.required_pixels for item in evidence)
        or claimed_distance is None
        or abs(claimed_distance - distance) > 1e-9
        or not 0.0 < distance <= near_limit
        or any(not 0.0 < item.distance_m <= near_limit for item in evidence)
    ):
        return False
    tolerance = _continuity_tolerance(cluster_span_m, max_distance_jump_m)
    return bool(abs(distance - anchor) <= tolerance)
