"""Cheap, geometry-only quality gates before a costly OSNet request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AppearanceQualityConfig:
    min_confidence: float = 0.75
    min_height_px: float = 120.0
    min_area_px: float = 8000.0
    min_aspect_ratio: float = 0.20
    max_aspect_ratio: float = 1.25
    edge_margin_ratio: float = 0.015


@dataclass(frozen=True)
class AppearanceQuality:
    full_ok: bool
    partial_ok: bool
    reason: str


class AppearanceQualityGate:
    def __init__(self, config: AppearanceQualityConfig) -> None:
        self.config = config

    def evaluate(
        self,
        bbox: Tuple[float, float, float, float],
        *,
        score: float,
        frame_width: int,
        frame_height: int,
    ) -> AppearanceQuality:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        if score < float(self.config.min_confidence):
            return AppearanceQuality(False, False, "confidence")
        if height < float(self.config.min_height_px) or area < float(self.config.min_area_px):
            return AppearanceQuality(False, False, "size")
        aspect = width / max(height, 1e-6)
        if not float(self.config.min_aspect_ratio) <= aspect <= float(self.config.max_aspect_ratio):
            return AppearanceQuality(False, False, "aspect")
        margin_x = max(1.0, float(frame_width) * float(self.config.edge_margin_ratio))
        margin_y = max(1.0, float(frame_height) * float(self.config.edge_margin_ratio))
        edge_touch = x1 <= margin_x or y1 <= margin_y or x2 >= frame_width - margin_x or y2 >= frame_height - margin_y
        if edge_touch:
            # A central torso may still be usable when head/feet are clipped,
            # but that observation must never update the full-body gallery.
            return AppearanceQuality(False, True, "edge_partial")
        return AppearanceQuality(True, True, "full")
