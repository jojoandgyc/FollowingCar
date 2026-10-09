"""Small per-target appearance gallery with robust cosine matching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


Feature = Tuple[float, ...]


def normalize(values: Optional[Iterable[float]]) -> Optional[Feature]:
    if values is None:
        return None
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        return None
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector) if norm > 1e-12 else None


def cosine(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if not left or len(left) != len(right):
        return None
    score = sum(float(a) * float(b) for a, b in zip(left, right))
    return max(-1.0, min(1.0, score)) if math.isfinite(score) else None


@dataclass(frozen=True)
class Template:
    feature: Feature
    quality: float
    captured_at: float
    view_bin: int


@dataclass(frozen=True)
class Match:
    score: Optional[float]
    source: Optional[str]
    compared_templates: int


class TargetProfile:
    """A bounded gallery for one target; no global vector database is needed."""

    def __init__(self, *, max_full: int, max_torso: int, duplicate_similarity: float = 0.992) -> None:
        self.max_full = max(1, int(max_full))
        self.max_torso = max(1, int(max_torso))
        self.duplicate_similarity = max(0.0, min(1.0, float(duplicate_similarity)))
        self.full: list[Template] = []
        self.torso: list[Template] = []

    @property
    def full_count(self) -> int:
        return len(self.full)

    @property
    def torso_count(self) -> int:
        return len(self.torso)

    def ready(self, *, min_full: int, min_torso: int) -> bool:
        return len(self.full) >= max(1, int(min_full)) or len(self.torso) >= max(1, int(min_torso))

    @staticmethod
    def _insert(templates: list[Template], template: Template, limit: int, duplicate_similarity: float) -> bool:
        if any((cosine(template.feature, previous.feature) or -1.0) >= duplicate_similarity for previous in templates):
            return False
        templates.append(template)
        # Retain the clearest exemplars if the gallery is full. The view bin is
        # metadata for future diagnostics, not an implicit identity decision.
        templates.sort(key=lambda item: (item.quality, item.captured_at), reverse=True)
        del templates[limit:]
        return True

    def add(self, feature: Optional[Iterable[float]], *, source: str, quality: float, captured_at: float, view_bin: int) -> bool:
        normalized = normalize(feature)
        if normalized is None:
            return False
        template = Template(normalized, max(0.0, float(quality)), float(captured_at), int(view_bin))
        if source == "full":
            return self._insert(self.full, template, self.max_full, self.duplicate_similarity)
        if source == "torso":
            return self._insert(self.torso, template, self.max_torso, self.duplicate_similarity)
        raise ValueError(f"unsupported feature source: {source!r}")

    @staticmethod
    def _robust_score(feature: Optional[Iterable[float]], templates: Sequence[Template]) -> tuple[Optional[float], int]:
        query = normalize(feature)
        if query is None or not templates:
            return None, 0
        scores = sorted((cosine(query, item.feature) for item in templates), reverse=True)
        usable = [score for score in scores if score is not None]
        if not usable:
            return None, 0
        # Mean of the three best independent templates prevents a single old
        # accidental high score from authorizing a vehicle hand-off.
        top = usable[: min(3, len(usable))]
        return sum(top) / len(top), len(usable)

    def match(self, full_feature, torso_feature) -> Match:
        full_score, full_n = self._robust_score(full_feature, self.full)
        torso_score, torso_n = self._robust_score(torso_feature, self.torso)
        options = [(full_score, "full", full_n), (torso_score, "torso", torso_n)]
        options = [item for item in options if item[0] is not None]
        if not options:
            return Match(None, None, 0)
        score, source, count = max(options, key=lambda item: item[0])
        return Match(score, source, count)
