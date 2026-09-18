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
    # A visible-torso descriptor for edge-clipped/near-camera candidates.  It
    # is kept in a separate gallery so the normal full-body threshold remains
    # unchanged.
    partial_appearance_enable: bool = True
    # Run the same OSNet model on a central torso ROI for the separate partial
    # gallery.  Keeping this configurable makes it possible to fall back to
    # the legacy histogram on boards where the extra pass is too expensive.
    partial_osnet_enable: bool = True
    partial_torso_top_ratio: float = 0.10
    partial_torso_bottom_ratio: float = 0.86
    partial_torso_side_ratio: float = 0.10


class OSNetRKNNExtractor:
    def __init__(self, config: OSNetConfig) -> None:
        self.config = config
        self.session: Optional[RKNNInferenceSession] = None
        self.last_timing_ms = {
            "preprocess": 0.0,
            "inference": 0.0,
            "partial_inference": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
            "detections": 0.0,
            "features": 0.0,
        }
        self.last_partial_features: List[Optional[Any]] = []
        self.last_partial_feature_sources: List[Optional[str]] = []
        if config.enabled and config.model_path:
            self.session = RKNNInferenceSession(
                config.model_path,
                target=config.target,
                core_mask=config.core_mask,
                backend=config.backend,
            )

    def extract(
        self,
        frame: Any,
        detections: Sequence[Detection],
        frame_format: str = "BGR",
        *,
        compute_partial: bool = True,
    ) -> List[Optional[Any]]:
        if not self.config.enabled or self.session is None or not detections:
            self._set_empty_timing(len(detections))
            return [None for _ in detections]

        np = _np()
        start = time.perf_counter()
        preprocess_ms = 0.0
        inference_ms = 0.0
        postprocess_ms = 0.0
        partial_inference_ms = 0.0
        arr, _, _, fmt = numpy_from_frame(frame, frame_format)
        features: List[Optional[Any]] = []
        partial_features: List[Optional[Any]] = []
        partial_sources: List[Optional[str]] = []
        for det in detections:
            prep_start = time.perf_counter()
            crop = self._crop(arr, det.bbox)
            if crop is None:
                preprocess_ms += _elapsed_ms(prep_start, time.perf_counter())
                features.append(None)
                partial_features.append(None)
                partial_sources.append(None)
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
                partial_features.append(None)
                partial_sources.append(None)
                continue
            # Keep the raw normalized OSNet vector while the full-body feature
            # optionally receives an appended color cue.
            osnet_feat = _normalize_embedding(outputs[0])
            feat = osnet_feat.copy()
            if self.config.color_fusion_enable:
                color = _color_signature(crop)
                if color is not None:
                    feat = _fuse_appearance_features(
                        feat,
                        color,
                        self.config.color_fusion_weight,
                    )
            partial_feature = None
            partial_source = None
            if compute_partial and self.config.partial_appearance_enable:
                if self.config.partial_osnet_enable:
                    torso = _torso_crop(
                        crop,
                        top_ratio=self.config.partial_torso_top_ratio,
                        bottom_ratio=self.config.partial_torso_bottom_ratio,
                        side_ratio=self.config.partial_torso_side_ratio,
                    )
                    if torso is not None:
                        partial_tensor = self._prepare_crop(torso, fmt)
                        partial_infer_start = time.perf_counter()
                        partial_outputs = self.session.inference([partial_tensor])
                        partial_infer_end = time.perf_counter()
                        partial_inference_ms += _elapsed_ms(
                            partial_infer_start, partial_infer_end
                        )
                        if partial_outputs:
                            partial_feature = _normalize_embedding(partial_outputs[0])
                            partial_source = "osnet_torso"
                    if partial_feature is None:
                        # Do not turn a failed torso pass into a full-body
                        # match.  That would silently contaminate the partial
                        # gallery and recreate the exact near-camera error
                        # this branch is meant to address.
                        partial_source = "osnet_torso_unavailable"
                else:
                    partial_feature = _partial_appearance_descriptor(crop)
                    partial_source = "histogram"
            postprocess_ms += _elapsed_ms(post_start, time.perf_counter())
            features.append(feat)
            partial_features.append(partial_feature)
            partial_sources.append(partial_source)
        end = time.perf_counter()
        self.last_timing_ms = {
            "preprocess": preprocess_ms,
            "inference": inference_ms + partial_inference_ms,
            "partial_inference": partial_inference_ms,
            "postprocess": postprocess_ms,
            "total": _elapsed_ms(start, end),
            "detections": float(len(detections)),
            "features": float(sum(feature is not None for feature in features)),
        }
        self.last_partial_features = partial_features
        self.last_partial_feature_sources = partial_sources
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
            "partial_inference": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
            "detections": float(detections),
            "features": 0.0,
        }
        self.last_partial_features = [None for _ in range(max(0, int(detections)))]
        self.last_partial_feature_sources = [None for _ in range(max(0, int(detections)))]

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


def _partial_appearance_descriptor(crop: Any):
    """Build a fixed-size descriptor from the visible central torso.

    This is deliberately a separate, inexpensive cue rather than a second
    RKNN pass.  It remains useful when the head/feet are outside the frame,
    while the full OSNet embedding can stay strict.  The descriptor combines
    coarse BGR histograms with a small grayscale texture histogram and is
    L2-normalized for cosine distance in ``IdentityEntry``.
    """
    np = _np()
    arr = np.asarray(crop)
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 8:
        return None
    height, width = arr.shape[:2]
    y1, y2 = int(round(height * 0.15)), int(round(height * 0.85))
    x1, x2 = int(round(width * 0.12)), int(round(width * 0.88))
    torso = arr[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    if torso.size == 0:
        return None
    parts = []
    for channel in range(min(3, torso.shape[2])):
        values = torso[:, :, channel].reshape(-1)
        hist, _ = np.histogram(values, bins=8, range=(0, 256))
        parts.append(hist.astype("float32"))
    gray = np.mean(torso[:, :, :3], axis=2).astype("float32")
    texture, _ = np.histogram(gray.reshape(-1), bins=8, range=(0, 256))
    parts.append(texture.astype("float32"))
    descriptor = np.concatenate(parts).astype("float32")
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
    return descriptor


def _torso_crop(
    crop: Any,
    *,
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.86,
    side_ratio: float = 0.10,
):
    """Return a background-reduced torso ROI while preserving visible body pixels.

    Ratios are intentionally conservative: on a near-camera crop the head or
    feet may already be outside the frame, so this must not assume a complete
    person box.  The ROI remains valid even when it touches the crop boundary.
    """
    np = _np()
    arr = np.asarray(crop)
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 8:
        return None
    height, width = arr.shape[:2]
    top = max(0, min(height - 1, int(round(height * float(top_ratio)))))
    bottom = max(top + 1, min(height, int(round(height * float(bottom_ratio)))))
    side = max(0, min(width // 3, int(round(width * float(side_ratio)))))
    left = side
    right = max(left + 1, width - side)
    torso = arr[top:bottom, left:right, :]
    return torso if torso.size else None


def _normalize_embedding(value: Any):
    np = _np()
    feature = np.asarray(value).reshape(-1).astype("float32")
    norm = float(np.linalg.norm(feature))
    if norm > 1e-12:
        feature = feature / norm
    return feature


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
