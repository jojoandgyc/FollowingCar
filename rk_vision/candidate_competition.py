"""Same-frame person-to-UID competition, independent of gallery UID margin.

This supplies a veto, never an identity/control authorization. Distances must
be computed from fresh detector features against one immutable gallery snapshot.
"""
from __future__ import annotations

import math
from typing import Optional


def competition_evidence(distances: dict[int, Optional[float]], *, uid: int,
                         frame_index: int, min_margin: float = 0.05,
                         exclusions: Optional[dict[int, str]] = None,
                         eligibility_reference_cap: Optional[int] = None) -> dict[int, dict]:
    clean = {}
    for index, value in distances.items():
        try:
            distance = float(value)
        except (TypeError, ValueError, OverflowError):
            distance = float("nan")
        clean[int(index)] = distance if math.isfinite(distance) else None
    result = {}
    exclusions = {i: why for i, why in (exclusions or {}).items() if i in clean}
    eligible = {i: d for i, d in clean.items() if i not in exclusions}
    for index, distance in clean.items():
        others = [(d, i) for i, d in eligible.items() if i != index and d is not None]
        competitor_distance, competitor_index = min(others) if others else (None, None)
        gap = (competitor_distance - distance
               if competitor_distance is not None and distance is not None else None)
        complete = all(d is not None for d in eligible.values())
        # A single partial-person candidate retains the existing partial ReID
        # path. In a crowd missing full-body evidence cannot establish a winner.
        passed = index in eligible and (len(eligible) <= 1 or (complete and gap is not None and gap + 1e-9 >= min_margin))
        result[index] = {
            "uid": int(uid), "frame_index": int(frame_index),
            "candidate_count": len(clean), "source_detection_index": index,
            "distance": distance, "competitor_index": competitor_index,
            "competitor_distance": competitor_distance, "distance_gap": gap,
            "min_margin": float(min_margin), "passed": bool(passed),
            "reason": ("ineligible_uid_competitor" if index in exclusions else
                       "single_candidate" if len(eligible) <= 1 else
                       "missing_competition_feature" if not complete else
                       "unique_reid_winner" if passed else "reid_margin_insufficient"),
        }
        if exclusions:
            result[index].update(qualified_candidate_count=len(eligible),
                excluded_competitors=dict(exclusions),
                eligibility_reference_cap=eligibility_reference_cap,
                competition_eligible=index in eligible,
                observation_only=exclusions.get(index) == 'weak_small_observation')
    return result
