from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


INFTY_COST = 1e5


def _cosine_distance(a: Any, b: Any):
    np = _np()
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return 1.0 - np.dot(a, b.T)


class NearestNeighborDistanceMetric:
    def __init__(self, metric: str, matching_threshold: float, budget: Optional[int] = None) -> None:
        if metric != "cosine":
            raise ValueError("only cosine distance is supported")
        self.matching_threshold = float(matching_threshold)
        self.budget = budget
        self.samples: Dict[int, List[Any]] = {}

    def partial_fit(self, features: Sequence[Any], targets: Sequence[int], active_targets: Sequence[int]) -> None:
        for feature, target in zip(features, targets):
            self.samples.setdefault(int(target), []).append(feature)
            if self.budget is not None:
                self.samples[int(target)] = self.samples[int(target)][-int(self.budget) :]
        active = {int(target) for target in active_targets}
        self.samples = {target: samples for target, samples in self.samples.items() if target in active}

    def distance(self, features: Sequence[Any], targets: Sequence[int]):
        np = _np()
        cost_matrix = np.full((len(targets), len(features)), INFTY_COST, dtype="float32")
        for row, target in enumerate(targets):
            samples = self.samples.get(int(target), [])
            if not samples:
                continue
            sample_np = np.asarray(samples, dtype="float32")
            for col, feature in enumerate(features):
                if feature is None:
                    continue
                dist = _cosine_distance(sample_np, np.asarray(feature, dtype="float32").reshape(1, -1))
                cost_matrix[row, col] = float(dist.min())
        return cost_matrix


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for DeepSORT nearest-neighbor matching") from exc
    return np
