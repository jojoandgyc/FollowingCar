from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .frames import numpy_from_frame
from .runtime import RKNNInferenceSession
from .yolo11 import Detection


@dataclass(frozen=True)
class OSNetConfig:
    model_path: str
    enabled: bool = True
    input_width: int = 64
    input_height: int = 128
    input_format: str = "BGR"
    input_dtype: str = "float32"
    input_layout: str = "NCHW"
    normalize: str = "imagenet"
    target: str = "rk3588"
    core_mask: str = "auto"
    backend: str = "auto"
    # A cheap color cue complements OSNet when two people have similar body
    # shapes. It is computed from the crop and adds no RKNN inference.
    color_fusion_enable: bool = True
    color_fusion_weight: float = 0.35


class OSNetRKNNExtractor:
    def __init__(self, config: OSNetConfig) -> None:
        self.config = config
        self.session: Optional[RKNNInferenceSession] = None
        self.last_timing_ms = {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
            "detections": 0.0,
            "features": 0.0,
        }
        if config.enabled and config.model_path:
            self.session = RKNNInferenceSession(
                config.model_path,
                target=config.target,
                core_mask=config.core_mask,
                backend=config.backend,
            )

    def extract(self, frame: Any, detections: Sequence[Detection], frame_format: str = "BGR") -> List[Optional[Any]]:
        if not self.config.enabled or self.session is None or not detections:
            self._set_empty_timing(len(detections))
            return [None for _ in detections]

        np = _np()
        start = time.perf_counter()
        preprocess_ms = 0.0
        inference_ms = 0.0
        postprocess_ms = 0.0
        arr, _, _, fmt = numpy_from_frame(frame, frame_format)
        features: List[Optional[Any]] = []
        for det in detections:
            prep_start = time.perf_counter()
            crop = self._crop(arr, det.bbox)
            if crop is None:
                preprocess_ms += _elapsed_ms(prep_start, time.perf_counter())
                features.append(None)
                continue
            tensor = self._prepare_crop(crop, fmt)
            infer_start = time.perf_counter()
            preprocess_ms += _elapsed_ms(prep_start, infer_start)
            outputs = self.session.inference([tensor])
            post_start = time.perf_counter()
            inference_ms += _elapsed_ms(infer_start, post_start)
            if not outputs:
                postprocess_ms += _elapsed_ms(post_start, time.perf_counter())
                features.append(None)
                continue
            feat = np.asarray(outputs[0]).reshape(-1).astype("float32")
            norm = float(np.linalg.norm(feat))
            if norm > 1e-12:
                feat = feat / norm
            if self.config.color_fusion_enable:
                color = _color_signature(crop)
                if color is not None:
                    feat = _fuse_appearance_features(
                        feat,
                        color,
                        self.config.color_fusion_weight,
                    )
            postprocess_ms += _elapsed_ms(post_start, time.perf_counter())
            features.append(feat)
        end = time.perf_counter()
        self.last_timing_ms = {
            "preprocess": preprocess_ms,
            "inference": inference_ms,
            "postprocess": postprocess_ms,
            "total": _elapsed_ms(start, end),
            "detections": float(len(detections)),
            "features": float(sum(feature is not None for feature in features)),
        }
        return features

    def release(self) -> None:
        if self.session is not None:
            self.session.release()

    def load(self) -> None:
        if self.config.enabled and self.session is not None:
            self.session.load()

    def _set_empty_timing(self, detections: int = 0) -> None:
        self.last_timing_ms = {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
            "detections": float(detections),
            "features": 0.0,
        }

    def _crop(self, arr: Any, bbox):
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        h, w = arr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return arr[y1:y2, x1:x2, :]

    def _prepare_crop(self, crop: Any, source_format: str):
        np = _np()
        crop = _resize(crop, self.config.input_width, self.config.input_height)
        desired = self.config.input_format.upper()
        if source_format != desired:
            if {source_format, desired} == {"BGR", "RGB"}:
                crop = crop[:, :, ::-1]
            else:
                raise ValueError(f"cannot convert crop format {source_format!r} to {desired!r}")

        dtype = self.config.input_dtype.lower()
        if dtype == "float32":
            crop = crop.astype("float32")
            if self.config.normalize.lower() == "imagenet":
                crop = crop / 255.0
                mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
                std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
                crop = (crop - mean) / std
            elif self.config.normalize.lower() in {"none", "raw"}:
                pass
            else:
                raise ValueError(f"unknown OSNet normalize mode: {self.config.normalize!r}")
        elif dtype == "uint8":
            crop = crop.astype("uint8", copy=False)
        else:
            raise ValueError(f"unsupported OSNet input dtype: {self.config.input_dtype!r}")

        layout = self.config.input_layout.upper()
        if layout == "NCHW":
            crop = crop.transpose(2, 0, 1)
        elif layout != "NHWC":
            raise ValueError(f"unsupported OSNet input layout: {self.config.input_layout!r}")
        return np.expand_dims(np.ascontiguousarray(crop), 0)


def _resize(img: Any, width: int, height: int):
    try:
        import cv2

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception:
        np = _np()
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("OpenCV or Pillow is required for OSNet crop resizing") from exc
        return np.asarray(Image.fromarray(img).resize((width, height), Image.BILINEAR))


def _color_signature(crop: Any):
    """Return a compact HSV appearance descriptor for the torso crop.

    The descriptor is intentionally small and normalized so it is useful as a
    secondary cue without overpowering the OSNet embedding. Background pixels
    are reduced by using the central 70% of the person crop vertically.
    """
    np = _np()
    arr = np.asarray(crop)
    if arr.ndim != 3 or arr.shape[0] < 4 or arr.shape[1] < 4:
        return None
    height = int(arr.shape[0])
    top = int(round(height * 0.15))
    bottom = max(top + 1, int(round(height * 0.85)))
    torso = arr[top:bottom]
    try:
        import cv2

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].reshape(-1)
        s = hsv[:, :, 1].reshape(-1)
        v = hsv[:, :, 2].reshape(-1)
        # Hue is only meaningful for sufficiently saturated pixels.
        h_hist, _ = np.histogram(h[s >= 24], bins=8, range=(0, 180))
        s_hist, _ = np.histogram(s, bins=4, range=(0, 256))
        v_hist, _ = np.histogram(v, bins=4, range=(0, 256))
        descriptor = np.concatenate((h_hist, s_hist, v_hist)).astype("float32")
    except Exception:
        # Keep the fallback dependency-free for unit tests and non-OpenCV hosts.
        descriptor = np.concatenate(
            (np.mean(torso, axis=(0, 1)), np.std(torso, axis=(0, 1)))
        ).astype("float32")
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
    return descriptor


def _fuse_appearance_features(osnet_feature: Any, color_feature: Any, weight: float):
    np = _np()
    base = np.asarray(osnet_feature, dtype="float32").reshape(-1)
    color = np.asarray(color_feature, dtype="float32").reshape(-1)
    base /= max(float(np.linalg.norm(base)), 1e-12)
    color /= max(float(np.linalg.norm(color)), 1e-12)
    fusion_weight = max(0.0, min(1.0, float(weight)))
    fused = np.concatenate((base, color * fusion_weight)).astype("float32")
    fused /= max(float(np.linalg.norm(fused)), 1e-12)
    return fused


def _elapsed_ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for OSNet RKNN feature extraction") from exc
    return np
