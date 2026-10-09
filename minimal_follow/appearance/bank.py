"""Small, quality-gated appearance template bank with cosine matching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


def _normalized(vector: Iterable[float]) -> Optional[Tuple[float, ...]]:
    values = tuple(float(value) for value in vector)
    if not values or not all(math.isfinite(value) for value in values):
        return None
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return None
    return tuple(value / norm for value in values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right) or not left:
        return None
    score = sum(float(a) * float(b) for a, b in zip(left, right))
    return max(-1.0, min(1.0, score)) if math.isfinite(score) else None


@dataclass(frozen=True)
class AppearanceMatch:
    score: Optional[float]
    source: Optional[str]
    template_count: int


class AppearanceTemplateBank:
    def __init__(self, *, max_full: int = 5, max_partial: int = 3, duplicate_similarity: float = 0.995) -> None:
        self.max_full = max(1, int(max_full))
        self.max_partial = max(1, int(max_partial))
        self.duplicate_similarity = max(0.0, min(1.0, float(duplicate_similarity)))
        self._full: list[Tuple[float, ...]] = []
        self._partial: list[Tuple[float, ...]] = []

    @property
    def enrolled(self) -> bool:
        return bool(self._full)

    @property
    def full_count(self) -> int:
        return len(self._full)

    @property
    def partial_count(self) -> int:
        return len(self._partial)

    def add(self, vector: Iterable[float], *, source: str) -> bool:
        normalized = _normalized(vector)
        if normalized is None:
            return False
        templates = self._full if source == "full" else self._partial
        limit = self.max_full if source == "full" else self.max_partial
        if any((cosine_similarity(normalized, existing) or -1.0) >= self.duplicate_similarity for existing in templates):
            return False
        templates.append(normalized)
        if len(templates) > limit:
            templates.pop(0)
        return True

    def match(self, full_feature, partial_feature) -> AppearanceMatch:
        candidates = []
        for source, feature, templates in (
            ("full", full_feature, self._full),
            ("partial", partial_feature, self._partial),
        ):
            normalized = _normalized(feature) if feature is not None else None
            if normalized is None:
                continue
            for template in templates:
                score = cosine_similarity(normalized, template)
                if score is not None:
                    candidates.append((score, source))
        if not candidates:
            return AppearanceMatch(None, None, 0)
        score, source = max(candidates, key=lambda item: item[0])
        return AppearanceMatch(score, source, len(candidates))
