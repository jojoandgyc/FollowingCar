from __future__ import annotations

import json
import logging
import math
import numbers
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .frames import numpy_from_frame


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReIDCrop:
    pixels: Any
    raw_detector_bbox: Tuple[float, float, float, float]
    crop_bbox: Tuple[int, int, int, int]
    source_format: str


def stage_crop(
    frame: Any,
    bbox: Any,
    frame_format: Optional[str] = None,
    *,
    logger: Any = None,
) -> Optional[ReIDCrop]:
    """Copy the raw detector crop before tracking or drawing changes the frame."""
    try:
        arr, _, _, fmt = numpy_from_frame(frame, frame_format)
        raw_bbox = tuple(float(value) for value in bbox)
        x1, y1, x2, y2 = [int(round(value)) for value in raw_bbox]
        height, width = arr.shape[:2]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return ReIDCrop(
            pixels=arr[y1:y2, x1:x2, :].copy(),
            raw_detector_bbox=raw_bbox,
            crop_bbox=(x1, y1, x2, y2),
            source_format=fmt,
        )
    except Exception as exc:
        (logger or _LOG).warning("ReID diagnostic crop failed: %s", exc)
        return None


class ReIDDiagnosticsWriter:
    """Best-effort, bounded crop and evidence output for one runtime log dir."""

    def __init__(
        self,
        output_dir: Any,
        *,
        enabled: bool = True,
        queue_capacity: int = 16,
        max_samples: int = 2000,
        logger: Any = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.index_path = self.output_dir / "events.jsonl"
        self.enabled = bool(enabled)
        self.max_samples = max(0, int(max_samples))
        self.accepted_samples = 0
        self.written_samples = 0
        self.dropped_samples = 0
        self.failed_samples = 0
        self.error: Optional[str] = None
        self._logger = logger or _LOG
        self._queue: Any = queue.Queue(maxsize=max(1, int(queue_capacity)))
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._paths: Dict[Tuple[int, int], str] = {}
        self._thread: Optional[threading.Thread] = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._run, name="reid-diagnostics", daemon=True
            )
            try:
                self._thread.start()
            except Exception as exc:
                self.enabled = False
                self._thread = None
                self.error = str(exc)
                self._logger.warning("ReID diagnostic worker could not start: %s", exc)

    def submit(
        self,
        frame: Any,
        bbox: Any,
        metadata: Mapping[str, Any],
        *,
        frame_format: Optional[str] = None,
    ) -> Optional[str]:
        if not self._has_capacity():
            return None
        crop = stage_crop(frame, bbox, frame_format, logger=self._logger)
        if crop is None:
            with self._lock:
                self.dropped_samples += 1
            return None
        return self.submit_crop(crop, metadata)

    def submit_crop(
        self, crop: Optional[ReIDCrop], metadata: Mapping[str, Any]
    ) -> Optional[str]:
        """Queue a staged crop; the caller must not modify its pixels afterward."""
        if not self._has_capacity():
            return None
        try:
            if crop is None:
                raise ValueError("missing staged crop")
            record = _json_value(metadata)
            if not isinstance(record, dict):
                raise ValueError("metadata must be a mapping")
            control_id = _record_id(record, "control_frame_id", "frame_index")
            capture_id = _record_id(record, "capture_frame_id")
            track_id = _record_id(record, "raw_track_id", "track_id", "rawtrack")
            record["control_frame_id"] = control_id
            record["capture_frame_id"] = capture_id
            record["raw_track_id"] = track_id
            record["raw_detector_bbox"] = list(crop.raw_detector_bbox)
            record["crop_bbox"] = list(crop.crop_bbox)
            record["source_format"] = crop.source_format
            # Serialize before queueing so later caller mutations cannot change evidence.
            record = _json_value(record)
            encoded = json.dumps(record, allow_nan=False, ensure_ascii=True)
            with self._lock:
                if self._closing.is_set() or self.accepted_samples >= self.max_samples:
                    self.dropped_samples += 1
                    return None
                sequence = self.accepted_samples + 1
                relative_path = (
                    f"frame_{_id_name(control_id)}_capture_{_id_name(capture_id)}"
                    f"_track_{_id_name(track_id)}_{sequence:04d}.png"
                )
                path_key = None if control_id is None or track_id is None else (control_id, track_id)
                try:
                    self._queue.put_nowait((crop, encoded, relative_path, path_key))
                except queue.Full:
                    self.dropped_samples += 1
                    return None
                self.accepted_samples += 1
                if path_key is not None:
                    self._paths[path_key] = relative_path
                return relative_path
        except Exception as exc:
            with self._lock:
                self.dropped_samples += 1
            self._logger.warning("ReID diagnostic submission failed: %s", exc)
            return None

    def sample_path(self, frame_index: Any, raw_track_id: Any) -> Optional[str]:
        """Return the relative path of an accepted sample, including pending I/O."""
        try:
            key = (int(frame_index), int(raw_track_id))
        except (TypeError, ValueError, OverflowError):
            return None
        with self._lock:
            return self._paths.get(key)

    def close(self, timeout_sec: float = 2.0) -> bool:
        with self._lock:
            self._closing.set()
        thread = self._thread
        if thread is None:
            return True
        try:
            timeout = float(timeout_sec)
            timeout = max(0.0, timeout) if math.isfinite(timeout) else 2.0
            thread.join(timeout=timeout)
        except Exception as exc:
            self._logger.warning("ReID diagnostic close failed: %s", exc)
            return False
        if thread.is_alive():
            self._logger.warning(
                "ReID diagnostic close timed out with %s pending samples", self._queue.qsize()
            )
            return False
        return True

    def _has_capacity(self) -> bool:
        with self._lock:
            if not self.enabled:
                return False
            if (
                self._closing.is_set()
                or self.accepted_samples >= self.max_samples
                or self._queue.full()
            ):
                self.dropped_samples += 1
                return False
            return True

    def _run(self) -> None:
        while not self._closing.is_set() or not self._queue.empty():
            try:
                crop, encoded, relative_path, path_key = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                path = self.output_dir / relative_path
                _write_png(path, crop)
                record = json.loads(encoded)
                record["sample_path"] = relative_path
                with self.index_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False, ensure_ascii=True) + "\n")
                with self._lock:
                    self.written_samples += 1
            except Exception as exc:
                with self._lock:
                    self.failed_samples += 1
                    self.error = str(exc)
                    if path_key is not None and self._paths.get(path_key) == relative_path:
                        self._paths.pop(path_key, None)
                self._logger.warning("ReID diagnostic save failed for %s: %s", relative_path, exc)
            finally:
                self._queue.task_done()


def _write_png(path: Path, crop: ReIDCrop) -> None:
    try:
        import cv2
    except ImportError:  # pragma: no cover - OpenCV is present on the runtime.
        from PIL import Image

        pixels = crop.pixels[:, :, ::-1] if crop.source_format == "BGR" else crop.pixels
        with path.open("xb") as stream:
            Image.fromarray(pixels).save(stream, format="PNG")
        return
    pixels = crop.pixels[:, :, ::-1] if crop.source_format == "RGB" else crop.pixels
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise OSError("PNG encoder did not produce an image")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())


def _record_id(record: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = record.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError, OverflowError):
                continue
    return None


def _id_name(value: Optional[int]) -> str:
    return "unknown" if value is None else f"{value:08d}"


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if type(value).__module__.split(".", 1)[0] == "numpy":
        return _json_value(value.tolist())
    return None
