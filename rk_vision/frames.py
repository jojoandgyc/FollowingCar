from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class FramePacket:
    """Frame supplied by an external camera reader.

    data is normally a numpy ndarray in HxWxC uint8 format.  Supported packed
    formats are BGR and RGB.  NV12/YUYV can be added once the RK camera path is
    known.
    """

    data: Any
    width: int = 0
    height: int = 0
    format: str = "BGR"
    timestamp_sec: Optional[float] = None
    stride: Optional[int] = None


def numpy_from_frame(frame: Any, frame_format: Optional[str] = None) -> Tuple[Any, int, int, str]:
    np = _np()
    if isinstance(frame, FramePacket):
        arr = np.asarray(frame.data)
        fmt = (frame_format or frame.format or "BGR").upper()
        width = int(frame.width or (arr.shape[1] if arr.ndim >= 2 else 0))
        height = int(frame.height or (arr.shape[0] if arr.ndim >= 2 else 0))
    else:
        arr = np.asarray(frame)
        fmt = (frame_format or "BGR").upper()
        width = int(arr.shape[1] if arr.ndim >= 2 else 0)
        height = int(arr.shape[0] if arr.ndim >= 2 else 0)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            "RKNN vision expects a packed HxWx3 frame for now; "
            f"got shape={getattr(arr, 'shape', None)!r}, format={fmt!r}"
        )
    if fmt not in {"BGR", "RGB"}:
        raise ValueError(f"unsupported frame format {fmt!r}; pass BGR/RGB or add a converter")
    return arr, width, height, fmt


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for rk_vision frame handling") from exc
    return np
