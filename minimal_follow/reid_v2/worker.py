"""Latest-only ReID worker. It receives a cropped person image, not a frame."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .artifacts import ReidArtifactWriter


@dataclass(frozen=True)
class ReidWorkerConfig:
    model_path: str
    input_width: int
    input_height: int
    input_format: str
    input_dtype: str
    input_layout: str
    normalize: str
    target: str
    core_mask: str
    backend: str
    artifact_dir: str = ""


@dataclass(frozen=True)
class ReidRequest:
    frame_id: int
    submitted_at: float
    purpose: str
    bbox: Tuple[float, float, float, float]
    quality: float
    crop: Any
    frame_width: int = 640
    allow_full: bool = True
    track_id: int = 0


@dataclass(frozen=True)
class ReidResult:
    frame_id: int
    submitted_at: float
    completed_at: float
    purpose: str
    bbox: Tuple[float, float, float, float]
    quality: float
    full_feature: Any
    torso_feature: Any
    timings_ms: dict
    error: Optional[str] = None
    frame_width: int = 640
    crop_path: Optional[str] = None
    feature_path: Optional[str] = None
    track_id: int = 0


class ReidWorker:
    def __init__(self, config: ReidWorkerConfig, *, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._requests: queue.Queue = queue.Queue(maxsize=1)
        self._results: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="minimal-reid-v2", daemon=True)
        self._started = False
        self._last_submit_at: dict[str, float] = {}
        self._artifacts = ReidArtifactWriter(config.artifact_dir, logger=self.logger)

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def submit(self, request: ReidRequest, *, min_interval_sec: float) -> bool:
        now = time.monotonic()
        previous = self._last_submit_at.get(request.purpose, float("-inf"))
        if now - previous < max(0.0, float(min_interval_sec)):
            return False
        self._last_submit_at[request.purpose] = now
        try:
            while True:
                self._requests.get_nowait()
        except queue.Empty:
            pass
        try:
            self._requests.put_nowait(request)
            return True
        except queue.Full:
            return False

    def poll_latest(self) -> Optional[ReidResult]:
        latest = None
        try:
            while True:
                latest = self._results.get_nowait()
        except queue.Empty:
            return latest

    def close(self) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:  # pragma: no cover - requires the RKNN board runtime
        extractor = None
        try:
            from rk_vision.reid import OSNetConfig, OSNetRKNNExtractor
            from rk_vision.yolo11 import Detection

            self.logger.info("reid_v2_worker starting model=%s core_mask=%s", self.config.model_path, self.config.core_mask)
            extractor = OSNetRKNNExtractor(OSNetConfig(
                model_path=self.config.model_path, enabled=True,
                input_width=self.config.input_width, input_height=self.config.input_height,
                input_format=self.config.input_format, input_dtype=self.config.input_dtype,
                input_layout=self.config.input_layout, normalize=self.config.normalize,
                target=self.config.target, core_mask=self.config.core_mask, backend=self.config.backend,
                # V2 deliberately keeps colour evidence out of the embedding.
                color_fusion_enable=False, partial_appearance_enable=True, partial_osnet_enable=True,
            ))
            self.logger.info("reid_v2_worker ready; waiting for enrollment/reacquire crops")
            while not self._stop.is_set():
                try:
                    request = self._requests.get(timeout=0.10)
                except queue.Empty:
                    continue
                crop_path = None
                try:
                    crop_path = self._artifacts.write_crop(
                        frame_id=request.frame_id, purpose=request.purpose, crop=request.crop,
                    )
                    height, width = request.crop.shape[:2]
                    detection = Detection((0.0, 0.0, float(width), float(height)), 1.0, 0)
                    full = extractor.extract(request.crop, [detection], "BGR", compute_partial=True)[0]
                    if not request.allow_full:
                        full = None
                    torso = extractor.last_partial_features[0] if extractor.last_partial_features else None
                    completed_at = time.monotonic()
                    feature_path = self._artifacts.write_features(
                        frame_id=request.frame_id, purpose=request.purpose, track_id=request.track_id, bbox=request.bbox,
                        quality=request.quality, submitted_at=request.submitted_at, completed_at=completed_at,
                        full_feature=full, torso_feature=torso, timings_ms=dict(extractor.last_timing_ms),
                        crop_path=crop_path, error=None,
                    )
                    result = ReidResult(
                        request.frame_id, request.submitted_at, completed_at, request.purpose,
                        request.bbox, request.quality, full, torso, dict(extractor.last_timing_ms), None,
                        request.frame_width, crop_path, feature_path, request.track_id,
                    )
                    self.logger.info(
                        "reid_v2_artifact frame=%s purpose=%s crop=%s feature=%s full=%s torso=%s",
                        request.frame_id, request.purpose, crop_path, feature_path,
                        full is not None, torso is not None,
                    )
                except Exception as exc:
                    completed_at = time.monotonic()
                    feature_path = self._artifacts.write_features(
                        frame_id=request.frame_id, purpose=request.purpose, track_id=request.track_id, bbox=request.bbox,
                        quality=request.quality, submitted_at=request.submitted_at, completed_at=completed_at,
                        full_feature=None, torso_feature=None, timings_ms={}, crop_path=crop_path,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    result = ReidResult(
                        request.frame_id, request.submitted_at, completed_at, request.purpose,
                        request.bbox, request.quality, None, None, {}, f"{type(exc).__name__}: {exc}",
                        request.frame_width, crop_path, feature_path, request.track_id,
                    )
                self._results.put(result)
        except Exception as exc:
            self.logger.exception("reid_v2_worker startup failed")
            self._results.put(ReidResult(
                -1, time.monotonic(), time.monotonic(), "worker_start", (0.0, 0.0, 0.0, 0.0),
                0.0, None, None, {}, f"{type(exc).__name__}: {exc}",
            ))
        finally:
            if extractor is not None:
                try:
                    extractor.release()
                except Exception:
                    pass
