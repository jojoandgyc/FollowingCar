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
    # This is an enrollment-view slot (front/left/right/back), not a camera
    # position.  It lets the gallery retain a useful distribution of poses.
    view: str = "unknown"


@dataclass(frozen=True)
class Match:
    score: Optional[float]
    source: Optional[str]
    compared_templates: int


class TargetProfile:
    """A bounded gallery for one target; no global vector database is needed."""

    VIEW_ORDER = ("front", "left", "right", "back")

    def __init__(self, *, max_full: int, max_torso: int, duplicate_similarity: float = 0.992,
                 max_templates_per_view: int = 2) -> None:
        self.max_full = max(1, int(max_full))
        self.max_torso = max(1, int(max_torso))
        self.duplicate_similarity = max(0.0, min(1.0, float(duplicate_similarity)))
        self.max_templates_per_view = max(1, int(max_templates_per_view))
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

    def _insert(self, templates: list[Template], template: Template, limit: int, duplicate_similarity: float) -> bool:
        # Only de-duplicate within a view. Front and back can legitimately be
        # very close in OSNet space, but dropping either loses recovery range.
        same_view = [previous for previous in templates if previous.view == template.view]
        if any((cosine(template.feature, previous.feature) or -1.0) >= duplicate_similarity for previous in same_view):
            return False
        templates.append(template)
        # Legacy/explicitly-disabled view capture uses ``unknown`` and keeps
        # the original global-gallery behaviour. Only named orientation slots
        # receive a per-view quota.
        if template.view in self.VIEW_ORDER:
            same_view = sorted((item for item in templates if item.view == template.view),
                               key=lambda item: (item.quality, item.captured_at), reverse=True)
            retained = set(id(item) for item in same_view[:self.max_templates_per_view])
            templates[:] = [item for item in templates if item.view != template.view or id(item) in retained]
        # Prefer removing a redundant view sample before removing the only
        # exemplar of any collected orientation.
        while len(templates) > limit:
            counts = {view: sum(item.view == view for item in templates) for view in self.VIEW_ORDER}
            removable = [item for item in templates if item.view not in self.VIEW_ORDER or counts.get(item.view, 0) > 1]
            victim = min(removable or templates, key=lambda item: (item.quality, item.captured_at))
            templates.remove(victim)
        return True

    def add(self, feature: Optional[Iterable[float]], *, source: str, quality: float, captured_at: float,
            view_bin: int, view: str = "unknown") -> bool:
        normalized = normalize(feature)
        if normalized is None:
            return False
        view = str(view).strip().lower()
        if view not in self.VIEW_ORDER:
            view = "unknown"
        template = Template(normalized, max(0.0, float(quality)), float(captured_at), int(view_bin), view)
        if source == "full":
            return self._insert(self.full, template, self.max_full, self.duplicate_similarity)
        if source == "torso":
            return self._insert(self.torso, template, self.max_torso, self.duplicate_similarity)
        raise ValueError(f"unsupported feature source: {source!r}")

    def view_count(self, view: str, *, source: str = "full") -> int:
        templates = self.full if source == "full" else self.torso
        return sum(item.view == view for item in templates)

    def view_counts(self, *, source: str = "full") -> dict[str, int]:
        return {view: self.view_count(view, source=source) for view in self.VIEW_ORDER}

    def combined_view_counts(self) -> dict[str, int]:
        """Return coverage per view from either a full or torso descriptor."""
        return {
            view: self.view_count(view, source="full") + self.view_count(view, source="torso")
            for view in self.VIEW_ORDER
        }

    def covered_view_count(self) -> int:
        return sum(count > 0 for count in self.combined_view_counts().values())

    def best_view_similarity(self, feature: Optional[Iterable[float]], view: str, *, source: str = "full") -> Optional[float]:
        query = normalize(feature)
        if query is None:
            return None
        templates = self.full if source == "full" else self.torso
        scores = [cosine(query, item.feature) for item in templates if item.view == view]
        usable = [score for score in scores if score is not None]
        return max(usable) if usable else None

    def is_duplicate(self, feature: Optional[Iterable[float]], *, source: str, similarity: float,
                     view: Optional[str] = None) -> bool:
        """Whether a sample duplicates an existing template in the same view.

        A global duplicate check looks tempting, but it suppresses useful
        side/back evidence: OSNet descriptors of one person from different
        directions are often still quite similar.  De-duplication therefore
        applies to a named orientation slot only.  The global gallery size is
        still bounded by :meth:`_insert`.
        """
        query = normalize(feature)
        if query is None:
            return False
        templates = self.full if source == "full" else self.torso
        if view in self.VIEW_ORDER:
            templates = [item for item in templates if item.view == view]
        limit = max(-1.0, min(1.0, float(similarity)))
        return any((cosine(query, item.feature) or -1.0) >= limit for item in templates)

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
