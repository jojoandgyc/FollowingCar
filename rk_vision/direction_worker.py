"""Asynchronous per-capture-frame direction evidence.

This worker deliberately owns detector instances and never touches ReID,
DeepSORT, or motor state.  It is therefore safe to run alongside the main
vision/control loop while preserving capture-frame ordering at the consumer.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from .frames import numpy_from_frame
from .yolo11 import Detection, YOLO11Config, YOLO11RKNNDetector


@dataclass(frozen=True)
class DirectionEvidence:
    capture_frame_id: int
    timestamp: float
    state: str  # visible, missing, unknown
    side: str  # left, right, none
    bbox: Optional[Tuple[float, float, float, float]] = None
    score: float = 0.0
    result_age_ms: float = 0.0
    reason: str = "none"
    frame_width: int = 0


@dataclass(frozen=True)
class _DirectionTask:
    capture_frame_id: int
    timestamp: float
    frame: Any
    frame_format: str


class DirectionInferencePool:
    """Run detector-only direction classification for every captured frame."""

    def __init__(
        self,
        detector_config: YOLO11Config,
        *,
        workers: int = 2,
        queue_size: int = 90,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.90,
        logger: Any = None,
        result_callback: Optional[Callable[[DirectionEvidence], None]] = None,
    ) -> None:
        self.config = detector_config
        self.workers = max(1, int(workers))
        # queue_size=0 means an unbounded backlog.  Direction history is an
        # audit trail, so the default runtime keeps every captured frame and
        # lets the workers catch up after a slow inference burst.  A positive
        # value remains available for deployments with a hard memory budget.
        self.queue_size = max(0, int(queue_size))
        self.min_area_ratio = max(0.0, min(1.0, float(min_area_ratio)))
        self.max_area_ratio = max(
            self.min_area_ratio,
            min(1.0, float(max_area_ratio)),
        )
        self.logger = logger
        self.result_callback = result_callback
        self._pending: queue.Queue[_DirectionTask] = queue.Queue(maxsize=self.queue_size)
        self._results: queue.Queue[DirectionEvidence] = queue.Queue()
        self._detectors: List[YOLO11RKNNDetector] = []
        self._stop = threading.Event()
        self._worker_threads: List[threading.Thread] = []
        for _ in range(self.workers):
            self._detectors.append(YOLO11RKNNDetector(self.config))
        for index in range(self.workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(index,),
                name=f"direction-yolo-{index}",
                daemon=True,
            )
            self._worker_threads.append(worker)
            worker.start()

    def submit(
        self,
        capture_frame_id: int,
        timestamp: float,
        frame: Any,
        frame_format: str = "BGR",
    ) -> bool:
        """Queue a copy of a capture frame without blocking camera capture."""
        if self._stop.is_set() or int(capture_frame_id) <= 0 or frame is None:
            return False
        try:
            arr, _width, _height, fmt = numpy_from_frame(frame, frame_format)
            # The camera buffer is reused by OpenCV/GStreamer after read().
            # Copy here so inference sees the exact captured image.
            arr = arr.copy()
            task = _DirectionTask(int(capture_frame_id), float(timestamp), arr, fmt)
            if self.queue_size == 0:
                self._pending.put(task)
            else:
                self._pending.put_nowait(task)
            return True
        except queue.Full:
            # Keep the frame slot explicit in the history; an overloaded
            # direction pool must become unknown, never silently disappear.
            evidence = DirectionEvidence(
                int(capture_frame_id), float(timestamp), "unknown", "none", reason="direction_queue_full"
            )
            self._publish(evidence)
            return False
        except Exception as exc:
            self._log("direction frame enqueue failed capture=%d: %s", int(capture_frame_id), exc)
            self._publish(
                DirectionEvidence(int(capture_frame_id), float(timestamp), "unknown", "none", reason="enqueue_error")
            )
            return False

    def drain_results(self, limit: int = 256) -> List[DirectionEvidence]:
        results: List[DirectionEvidence] = []
        for _ in range(max(1, int(limit))):
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                break
        results.sort(key=lambda item: item.capture_frame_id)
        return results

    @property
    def pending_count(self) -> int:
        return int(self._pending.qsize())

    def close(self, timeout_sec: float = 3.0) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        # Runtime shutdown does not need to finish an unbounded audit backlog.
        # Drop queued image buffers only after capture has stopped; workers may
        # finish their current inference before their detector is released.
        while True:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                break
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        for worker in self._worker_threads:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [worker.name for worker in self._worker_threads if worker.is_alive()]
        if alive:
            self._log("direction workers still stopping; defer detector release: %s", alive)
            self._worker_threads.clear()
            return
        self._worker_threads.clear()
        for detector in self._detectors:
            try:
                detector.release()
            except Exception as exc:
                self._log("direction detector release failed: %s", exc)
        self._detectors.clear()

    def _worker_loop(self, detector_index: int) -> None:
        detector = self._detectors[detector_index]
        while not self._stop.is_set():
            try:
                task = self._pending.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._publish(self._classify(task, detector))
            except Exception as exc:
                self._log("direction worker failed capture=%d: %s", task.capture_frame_id, exc)
                self._publish(
                    DirectionEvidence(
                        task.capture_frame_id,
                        task.timestamp,
                        "unknown",
                        "none",
                        reason="direction_worker_error",
                    )
                )

    def _classify(self, task: _DirectionTask, detector: YOLO11RKNNDetector) -> DirectionEvidence:
        def result_age_ms() -> float:
            # Timestamp is captured at camera read time, so this includes both
            # queue wait and detector work rather than only inference duration.
            return max(0.0, (time.monotonic() - float(task.timestamp)) * 1000.0)

        detector.set_search_diagnostic_active(True)
        detections = detector.detect(task.frame, task.frame_format)
        formal_persons = [
            item
            for item in detections
            if int(item.class_id) == int(self.config.search_diagnostic_class_id)
        ]
        diagnostic_persons = [
            item
            for item in list(detector.last_search_diagnostic_detections)
            if int(item.class_id) == int(self.config.search_diagnostic_class_id)
        ]
        # A diagnostic detector is deliberately permissive and may report a
        # background blob with a score higher than the formal detector. It is
        # useful for filling the capture-history queue, but must never outrank
        # a formal detection from the same image.
        persons = formal_persons or diagnostic_persons
        person_reason = (
            "detector_formal_person_side"
            if formal_persons
            else "detector_diagnostic_person_side"
        )
        if not persons:
            return DirectionEvidence(
                task.capture_frame_id,
                task.timestamp,
                "missing",
                "none",
                result_age_ms=result_age_ms(),
                reason="detector_no_person",
                frame_width=int(task.frame.shape[1]),
            )
        frame_area = max(1.0, float(task.frame.shape[0]) * float(task.frame.shape[1]))
        usable_persons = [
            item
            for item in persons
            if self.min_area_ratio
            <= float(item.area) / frame_area
            <= self.max_area_ratio
        ]
        if not usable_persons:
            # Preserve the capture-frame slot, but do not turn a tiny edge
            # fragment into a directional vote.  ``unknown`` is intentionally
            # distinct from ``missing`` so history resolution cannot use it.
            return DirectionEvidence(
                task.capture_frame_id,
                task.timestamp,
                "unknown",
                "none",
                result_age_ms=result_age_ms(),
                reason="detector_below_area",
                frame_width=int(task.frame.shape[1]),
            )
        persons = usable_persons
        selected = max(persons, key=lambda item: (float(item.score), float(item.area)))
        x1, _y1, x2, _y2 = selected.bbox
        width = max(1.0, float(task.frame.shape[1]))
        center = max(0.0, min(1.0, (float(x1) + float(x2)) / (2.0 * width)))
        side = "left" if center < 0.5 else "right"
        return DirectionEvidence(
            task.capture_frame_id,
            task.timestamp,
            "visible",
            side,
            bbox=tuple(float(value) for value in selected.bbox),
            score=float(selected.score),
            result_age_ms=result_age_ms(),
            reason=person_reason,
            frame_width=int(task.frame.shape[1]),
        )

    def _publish(self, evidence: DirectionEvidence) -> None:
        self._results.put(evidence)
        callback = self.result_callback
        if callback is not None:
            callback(evidence)

    def _log(self, message: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.warning(message, *args)
