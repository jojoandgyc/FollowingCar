from __future__ import annotations

from typing import Any, Optional


class Detection:
    def __init__(
        self,
        tlwh: Any,
        confidence: float,
        label: int,
        feature: Optional[Any],
        *,
        store_feature: bool = True,
    ) -> None:
        np = _np()
        self.tlwh = np.asarray(tlwh, dtype="float32")
        self.confidence = float(confidence)
        self.cls = int(label)
        self.feature = None if feature is None else np.asarray(feature, dtype="float32").reshape(-1)
        self.store_feature = bool(store_feature)

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] = ret[2] / max(ret[3], 1e-6)
        return ret


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for DeepSORT detections") from exc
    return np
