"""Safe, stateless reconstruction of a lost-target search direction.

The main vision loop intentionally processes only the newest camera frame.  A
capture worker still produces detector-only evidence for every frame, so a
short sequence of those results can be evaluated after a loss.  This module
does not know about DeepSORT, ReID galleries, or motor actions; it only returns
an advisory direction when the detector evidence is unique and geometrically
continuous.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class HistoricalDirectionCandidate:
    capture_frame_id: int
    timestamp: float
    state: str
    bbox: Optional[Tuple[float, float, float, float]]
    score: float
    frame_width: int
    candidate_count: int = 1


@dataclass(frozen=True)
class HistoricalDirectionResult:
    direction: Optional[str]
    confidence: float
    first_capture_frame_id: Optional[int]
    last_capture_frame_id: Optional[int]
    selected_capture_frame_ids: Tuple[int, ...]
    reason: str


class HistoricalDirectionBackfill:
    """Evaluate detector-only capture evidence without mutating runtime state."""

    def __init__(
        self,
        *,
        max_age_sec: float = 0.70,
        min_samples: int = 2,
        max_capture_gap: int = 6,
        max_center_jump_ratio: float = 0.30,
        min_area_similarity: float = 0.45,
        confidence_cap: float = 0.70,
        min_score: float = 0.20,
    ) -> None:
        self.max_age_sec = max(0.10, float(max_age_sec))
        self.min_samples = max(2, int(min_samples))
        self.max_capture_gap = max(1, int(max_capture_gap))
        self.max_center_jump_ratio = max(0.05, min(1.0, float(max_center_jump_ratio)))
        self.min_area_similarity = max(0.05, min(1.0, float(min_area_similarity)))
        self.confidence_cap = max(0.10, min(0.95, float(confidence_cap)))
        self.min_score = max(0.0, min(1.0, float(min_score)))

    @staticmethod
    def _center(candidate: HistoricalDirectionCandidate) -> Optional[float]:
        if candidate.bbox is None or int(candidate.frame_width) <= 0:
            return None
        try:
            x1, _y1, x2, _y2 = (float(value) for value in candidate.bbox)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x1, x2)) or x2 <= x1:
            return None
        return max(0.0, min(1.0, ((x1 + x2) * 0.5) / float(candidate.frame_width)))

    @staticmethod
    def _area(candidate: HistoricalDirectionCandidate) -> Optional[float]:
        if candidate.bbox is None:
            return None
        try:
            x1, y1, x2, y2 = (float(value) for value in candidate.bbox)
        except (TypeError, ValueError):
            return None
        area = (x2 - x1) * (y2 - y1)
        return area if math.isfinite(area) and area > 0.0 else None

    def evaluate(
        self,
        candidates: Iterable[HistoricalDirectionCandidate],
        *,
        loss_capture_frame_id: int,
        now: Optional[float] = None,
        anchor_capture_frame_id: Optional[int] = None,
        anchor_center_ratio: Optional[float] = None,
    ) -> HistoricalDirectionResult:
        """Return an advisory side, or a rejection reason.

        Only frames before the loss are considered.  A detector result with
        multiple competing person boxes is never used, because this worker has
        no identity information to disambiguate them.
        """
        current = time.monotonic() if now is None else float(now)
        loss_id = int(loss_capture_frame_id)
        eligible = []
        for item in sorted(candidates, key=lambda value: int(value.capture_frame_id)):
            if int(item.capture_frame_id) <= 0 or int(item.capture_frame_id) >= loss_id:
                continue
            if str(item.state).lower() != "visible" or item.bbox is None:
                continue
            if int(item.candidate_count) != 1:
                continue
            if float(item.score) < self.min_score:
                continue
            if int(item.frame_width) <= 0:
                continue
            if current - float(item.timestamp) > self.max_age_sec:
                continue
            if self._center(item) is None or self._area(item) is None:
                continue
            eligible.append(item)
        if len(eligible) < self.min_samples:
            return HistoricalDirectionResult(None, 0.0, None, None, (), "no_unique_chain")

        # Keep the newest contiguous chain.  A gap represents frames for which
        # the detector did not provide usable evidence and must not be bridged.
        chain = [eligible[-1]]
        for item in reversed(eligible[:-1]):
            newer = chain[0]
            if int(newer.capture_frame_id) - int(item.capture_frame_id) > self.max_capture_gap:
                break
            prev_center = self._center(item)
            next_center = self._center(newer)
            prev_area = self._area(item)
            next_area = self._area(newer)
            if prev_center is None or next_center is None or prev_area is None or next_area is None:
                break
            if abs(next_center - prev_center) > self.max_center_jump_ratio:
                break
            area_similarity = min(prev_area, next_area) / max(prev_area, next_area)
            if area_similarity < self.min_area_similarity:
                break
            chain.insert(0, item)
        if len(chain) < self.min_samples:
            return HistoricalDirectionResult(None, 0.0, None, None, (), "geometry_chain_rejected")

        if anchor_capture_frame_id is not None and anchor_center_ratio is not None:
            anchor_gap = int(chain[0].capture_frame_id) - int(anchor_capture_frame_id)
            if 0 < anchor_gap <= self.max_capture_gap:
                first_center = self._center(chain[0])
                if first_center is None or abs(first_center - float(anchor_center_ratio)) > self.max_center_jump_ratio:
                    return HistoricalDirectionResult(None, 0.0, None, None, (), "geometry_anchor_rejected")

        centers = [self._center(item) for item in chain]
        if any(value is None for value in centers):
            return HistoricalDirectionResult(None, 0.0, None, None, (), "geometry_chain_rejected")
        final_center = float(centers[-1])
        direction = "left" if final_center < 0.5 else "right"
        scores = [max(0.0, min(1.0, float(item.score))) for item in chain]
        confidence = min(
            self.confidence_cap,
            max(0.35, 0.45 + 0.05 * min(4, len(chain)) + 0.10 * (sum(scores) / max(1, len(scores)))),
        )
        return HistoricalDirectionResult(
            direction,
            confidence,
            int(chain[0].capture_frame_id),
            int(chain[-1].capture_frame_id),
            tuple(int(item.capture_frame_id) for item in chain),
            "historical_direction_evidence",
        )
