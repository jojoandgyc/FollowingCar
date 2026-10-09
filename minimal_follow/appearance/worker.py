"""Latest-only asynchronous OSNet worker; it never blocks the control loop."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class AppearanceWorkerConfig:
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
    color_fusion_enable: bool = True
    color_fusion_weight: float = 0.35


@dataclass(frozen=True)
class AppearanceRequest:
    frame_id: int
    submitted_at: float
    bbox: Tuple[float, float, float, float]
    score: float
    purpose: str
    allow_full: bool
    compute_partial: bool
    frame: Any


@dataclass(frozen=True)
class AppearanceResult:
    frame_id: int
    submitted_at: float
    completed_at: float
    purpose: str
    full_feature: Any
    partial_feature: Any
    timings_ms: dict
    error: Optional[str] = None


class AppearanceWorker:
    def __init__(self, config: AppearanceWorkerConfig, *, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._requests: queue.Queue = queue.Queue(maxsize=1)
        self._results: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="minimal-appearance", daemon=True)
        self._started = False
        self._last_submit_at: dict[str, float] = {}

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def submit(self, request: AppearanceRequest, *, min_interval_sec: float) -> bool:
        now = time.monotonic()
        previous = self._last_submit_at.get(request.purpose, 0.0)
        if now - previous < max(0.0, float(min_interval_sec)):
            return False
        self._last_submit_at[request.purpose] = now
        # Drop any queued stale candidate. A job already running is allowed to
        # finish, but its stale result is rejected by the identity policy.
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

    def poll_latest(self) -> Optional[AppearanceResult]:
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

    def _run(self) -> None:  # pragma: no cover - requires board RKNN runtime
        extractor = None
        try:
            # Imports and RKNN construction remain in the worker so startup
            # failure cannot import/load a second model on the control thread.
            from rk_vision.reid import OSNetConfig, OSNetRKNNExtractor
            from rk_vision.yolo11 import Detection

            extractor = OSNetRKNNExtractor(OSNetConfig(
                model_path=self.config.model_path,
                enabled=True,
                input_width=self.config.input_width,
                input_height=self.config.input_height,
                input_format=self.config.input_format,
                input_dtype=self.config.input_dtype,
                input_layout=self.config.input_layout,
                normalize=self.config.normalize,
                target=self.config.target,
                core_mask=self.config.core_mask,
                backend=self.config.backend,
                color_fusion_enable=self.config.color_fusion_enable,
                color_fusion_weight=self.config.color_fusion_weight,
                partial_appearance_enable=True,
                partial_osnet_enable=True,
            ))
            while not self._stop.is_set():
                try:
                    request = self._requests.get(timeout=0.10)
                except queue.Empty:
                    continue
                try:
                    detection = Detection(request.bbox, request.score, 0)
                    features = extractor.extract(
                        request.frame, [detection], "BGR", compute_partial=request.compute_partial,
                    )
                    # An edge-clipped crop may be suitable for the separate
                    # torso gallery, but must never enter the full-body one.
                    full_feature = features[0] if features and request.allow_full else None
                    partial = extractor.last_partial_features[0] if extractor.last_partial_features else None
                    result = AppearanceResult(
                        request.frame_id, request.submitted_at, time.monotonic(), request.purpose,
                        full_feature, partial, dict(extractor.last_timing_ms), None,
                    )
                except Exception as exc:
                    result = AppearanceResult(
                        request.frame_id, request.submitted_at, time.monotonic(), request.purpose,
                        None, None, {}, f"{type(exc).__name__}: {exc}",
                    )
                self._results.put(result)
        except Exception as exc:
            self._results.put(AppearanceResult(
                -1, time.monotonic(), time.monotonic(), "worker_start", None, None, {},
                f"{type(exc).__name__}: {exc}",
            ))
        finally:
            if extractor is not None:
                try:
                    extractor.release()
                except Exception:
                    pass
