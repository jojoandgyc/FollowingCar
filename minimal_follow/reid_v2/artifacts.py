"""Best-effort ReID evidence export; never allowed to stop control or ReID."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


def _vector(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [float(item) for item in value]
    except Exception:
        return None


class ReidArtifactWriter:
    """Stores human-reviewable crops and portable numeric embeddings per run."""

    def __init__(self, directory: str, *, logger) -> None:
        self.directory = Path(directory) if directory else None
        self.logger = logger
        if self.directory is not None:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self.logger.warning("ReID artifact output disabled: %s", exc)
                self.directory = None

    def _base(self, frame_id: int, purpose: str) -> Optional[Path]:
        if self.directory is None:
            return None
        return self.directory / f"frame_{int(frame_id):06d}_{purpose}"

    def write_crop(self, *, frame_id: int, purpose: str, crop: Any) -> Optional[str]:
        base = self._base(frame_id, purpose)
        if base is None:
            return None
        try:
            import cv2

            path = base.with_suffix(".jpg")
            if not cv2.imwrite(str(path), crop):
                raise RuntimeError("cv2.imwrite returned false")
            return str(path)
        except Exception as exc:
            self.logger.warning("ReID crop artifact write failed frame=%s: %s", frame_id, exc)
            return None

    def write_features(
        self, *, frame_id: int, purpose: str, bbox, quality: float, submitted_at: float,
        completed_at: float, full_feature: Any, torso_feature: Any, timings_ms: dict,
        crop_path: Optional[str], error: Optional[str],
    ) -> Optional[str]:
        base = self._base(frame_id, purpose)
        if base is None:
            return None
        try:
            path = base.with_suffix(".json")
            payload = {
                "frame_id": int(frame_id),
                "purpose": str(purpose),
                "bbox": [float(value) for value in bbox],
                "quality": float(quality),
                "submitted_at_monotonic_s": float(submitted_at),
                "completed_at_monotonic_s": float(completed_at),
                "elapsed_ms": round(max(0.0, completed_at - submitted_at) * 1000.0, 3),
                "crop_path": crop_path,
                "full_embedding": _vector(full_feature),
                "torso_embedding": _vector(torso_feature),
                "timings_ms": {str(key): float(value) for key, value in timings_ms.items()},
                "error": error,
            }
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
            os.replace(temp, path)
            return str(path)
        except Exception as exc:
            self.logger.warning("ReID feature artifact write failed frame=%s: %s", frame_id, exc)
            return None
