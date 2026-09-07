#!/usr/bin/env python3
"""CPU checks for the lightweight multi-cue ReID appearance feature."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rk_vision.reid import _color_signature, _fuse_appearance_features


def main() -> int:
    crop = np.zeros((128, 64, 3), dtype=np.uint8)
    crop[:, :, 1] = 180
    color = _color_signature(crop)
    if color is None or color.ndim != 1 or color.size not in (6, 16):
        raise AssertionError(f"unexpected color signature shape: {None if color is None else color.shape}")
    if not np.isclose(float(np.linalg.norm(color)), 1.0, atol=1e-5):
        raise AssertionError("color signature must be normalized")

    fused = _fuse_appearance_features(np.ones(512, dtype="float32"), color, 0.35)
    if fused.size != 512 + color.size:
        raise AssertionError(f"unexpected fused feature size: {fused.size}")
    if not np.isclose(float(np.linalg.norm(fused)), 1.0, atol=1e-5):
        raise AssertionError("fused feature must be normalized")
    print("reid appearance fusion ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
