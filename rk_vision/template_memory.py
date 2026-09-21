"""Bounded, capture-time based memory of *approved* template observations.

This module neither assigns identities nor refreshes motion authority. Queries
never renew a template. The identity bank alone decides whether learning is safe.
"""
from dataclasses import dataclass, field
import math
import numpy as np


def timestamp(metadata):
    try:
        value = float((metadata or {}).get("capture_timestamp"))
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


@dataclass
class TemplateMemory:
    recent_sec: float = 30.0
    archive_sec: float = 120.0
    capacity: int = 8
    archive_capacity: int = 6
    recent: dict = field(default_factory=lambda: {"strong": [], "partial": []})
    watermark: float = -1.0
    last_learning: dict = field(default_factory=dict)
    expired_count: int = 0

    def __post_init__(self):
        if not (math.isfinite(self.recent_sec) and math.isfinite(self.archive_sec)
                and 0 < self.recent_sec <= self.archive_sec):
            raise ValueError("template windows must be finite, positive and recent <= archive")

    def advance(self, metadata):
        stamp = timestamp(metadata)
        if stamp is None:
            return
        self.watermark = max(self.watermark, stamp)
        for tier, rows in self.recent.items():
            keep = [r for r in rows if 0 <= self.watermark - r[1]["capture_timestamp"] <= self.recent_sec]
            self.expired_count += len(rows) - len(keep)
            self.recent[tier] = keep

    def remember(self, feature, metadata, tier):
        stamp = timestamp(metadata)
        cap = (metadata or {}).get("capture_frame_id")
        # Legacy/missing timestamps cannot become apparently fresh evidence.
        if (stamp is None or cap is None or metadata.get("is_fresh") is not True
                or metadata.get("identity_control_rejected")
                or metadata.get("quality_bbox_ok") is False
                or metadata.get("bbox_quality_tier") == "reject"
                or (tier == "strong" and metadata.get("bbox_quality_tier") == "weak")
                or metadata.get("search_observation_only")
                or metadata.get("preferred_search_low_confidence")):
            return False
        previous = self.last_learning.get(tier)
        if stamp < self.watermark or (previous and (stamp <= previous[0] or cap <= previous[1])):
            return False
        self.advance(metadata)
        value = np.asarray(feature, dtype="float32").reshape(-1).copy()
        norm = float(np.linalg.norm(value))
        if not np.isfinite(value).all() or norm <= 1e-12:
            return False
        value /= norm
        rows = self.recent[tier]
        # Replace a duplicate with the ACTUAL new approved crop, not a renewed
        # timestamp on the old embedding. Preserve distinct recent poses.
        similar = [i for i, (v, _) in enumerate(rows)
                   if v.size == value.size and float(1 - v.dot(value)) < .02]
        if similar:
            rows.pop(similar[0])
        rows.append((value, dict(metadata)))
        self.recent[tier] = rows[-max(1, self.capacity):]
        self.last_learning[tier] = (stamp, cap)
        return True

    def evidence(self, feature, metadata, tier="strong", *, reliable_only=False):
        stamp = timestamp(metadata)
        query = None if feature is None else np.asarray(feature, dtype="float32").reshape(-1)
        rows = []
        if stamp is not None and query is not None and np.isfinite(query).all():
            query = query / max(float(np.linalg.norm(query)), 1e-12)
            rows = [(float(1 - value.dot(query)), info) for value, info in self.recent[tier]
                    if value.size == query.size and 0 <= stamp - info["capture_timestamp"] <= self.recent_sec
                    and (not reliable_only or self.partial_usable(info))]
        rows.sort(key=lambda r: r[0])
        return {"count": len(rows), "distance": rows[0][0] if rows else None,
                "winner_cap": rows[0][1]["capture_frame_id"] if rows else None,
                "winner_age_sec": stamp - rows[0][1]["capture_timestamp"] if rows else None}

    @staticmethod
    def partial_usable(metadata):
        """Conservative crop usability, not a claim of anatomical alignment."""
        m = metadata or {}
        if (m.get("partial_feature_source") != "osnet_torso"
                or m.get("quality_bbox_ok") is not True
                or m.get("bbox_quality_tier") != "strong"
                or m.get("is_fresh") is not True):
            return False
        try:
            x1,y1,x2,y2 = m["detector_bbox"]
            edges = float(m.get("detector_edge_touch_count", 0))
            return (all(math.isfinite(float(v)) for v in (x1,y1,x2,y2,edges))
                    and x2-x1 >= 40 and y2-y1 >= 80 and edges <= 2)
        except (KeyError, TypeError, ValueError):
            return False

    def prune_archive(self, features, metadata, *, keep_anchor):
        if self.watermark < 0:
            return 0
        keep = [i for i in range(len(features)) if (keep_anchor and i == 0)
                or (i < len(metadata) and timestamp(metadata[i]) is not None
                    and self.watermark - timestamp(metadata[i]) <= self.archive_sec)]
        removed = len(features) - len(keep)
        features[:] = [features[i] for i in keep]
        metadata[:] = [metadata[i] if i < len(metadata) else {} for i in keep]
        return removed
