from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class VideoRecorderConfig:
    output_path: str
    fps: float
    fourcc: str = "MJPG"
    queue_capacity: int = 60
    # Detection metadata arrives after the camera frame has been captured.
    # Waiting happens only in the recorder thread, never in control or capture.
    overlay_wait_sec: float = 0.25


@dataclass(frozen=True)
class _QueuedFrame:
    image: Any
    capture_frame_id: int
    control_frame_id: Optional[int]
    monotonic_sec: float
    unix_sec: float


@dataclass(frozen=True)
class VideoDetectionOverlay:
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int


@dataclass(frozen=True)
class VideoTrackOverlay:
    bbox: tuple[float, float, float, float]
    track_id: int
    reid_uid: int = 0
    mapped_uid: int = 0
    best_uid: int = 0
    score: float = 0.0
    distance: Optional[float] = None
    assignment_reason: str = "none"
    quality_reason: str = ""
    fresh: bool = True
    active_target: bool = False


@dataclass(frozen=True)
class VideoControlOverlay:
    control_frame_id: Optional[int] = None
    active_target_id: Optional[int] = None
    selected_target_id: Optional[int] = None
    candidate_bbox: Optional[tuple[float, float, float, float]] = None
    candidate_score: float = 0.0
    candidate_source: str = "none"
    candidate_matches_target: bool = False
    action_name: str = "none"
    requested_rpm: int = 0
    yaw_rate_dps: Optional[float] = None
    result_age_ms: float = 0.0
    search_state: str = "none"
    search_direction: Optional[str] = None
    decision_reason: str = "none"


@dataclass(frozen=True)
class VideoFrameOverlay:
    detections: tuple[VideoDetectionOverlay, ...] = ()
    tracks: tuple[VideoTrackOverlay, ...] = ()
    control: VideoControlOverlay = VideoControlOverlay()


class AsyncVideoRecorder:
    """Write control-loop camera frames without blocking the control thread."""

    def __init__(
        self,
        config: VideoRecorderConfig,
        *,
        cv2_module: Any,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self._cv2 = cv2_module
        self._logger = logger or logging.getLogger(__name__)
        self._queue: queue.Queue[_QueuedFrame] = queue.Queue(
            maxsize=max(1, int(config.queue_capacity))
        )
        self._closing = threading.Event()
        self._closed = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._last_drop_log_ts = 0.0
        self._error: Optional[BaseException] = None
        self._overlay_condition = threading.Condition()
        self._overlays: dict[int, VideoFrameOverlay] = {}
        self._submitted_capture_ids: set[int] = set()
        self._written_capture_ids: set[int] = set()

    @property
    def index_path(self) -> str:
        path = Path(self.config.output_path)
        return str(path.with_suffix(".frames.csv"))

    @property
    def written_frames(self) -> int:
        return int(self._written)

    @property
    def dropped_frames(self) -> int:
        return int(self._dropped)

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def submit(
        self,
        image: Any,
        *,
        capture_frame_id: Optional[int] = None,
        control_frame_id: Optional[int] = None,
        # Kept as a source-compatible alias for older callers.  New code must
        # use capture_frame_id because the recorder runs on the camera stream,
        # independently of the slower control loop.
        control_frame_index: Optional[int] = None,
        monotonic_sec: Optional[float] = None,
        unix_sec: Optional[float] = None,
    ) -> bool:
        if self._closing.is_set() or self._error is not None:
            return False
        self._ensure_thread()
        if capture_frame_id is None:
            if control_frame_index is None:
                raise ValueError("capture_frame_id is required")
            capture_frame_id = int(control_frame_index)
        item = _QueuedFrame(
            image=image.copy(),
            capture_frame_id=int(capture_frame_id),
            control_frame_id=(None if control_frame_id is None else int(control_frame_id)),
            monotonic_sec=float(time.monotonic() if monotonic_sec is None else monotonic_sec),
            unix_sec=float(time.time() if unix_sec is None else unix_sec),
        )
        with self._overlay_condition:
            self._submitted_capture_ids.add(int(item.capture_frame_id))
        try:
            self._queue.put_nowait(item)
            self._submitted += 1
            return True
        except queue.Full:
            with self._overlay_condition:
                self._submitted_capture_ids.discard(int(item.capture_frame_id))
            self._dropped += 1
            now = time.monotonic()
            if now - self._last_drop_log_ts >= 1.0:
                self._last_drop_log_ts = now
                self._logger.warning(
                    "Camera recorder queue full: dropped=%d queued=%d output=%s",
                    self._dropped,
                    self._queue.qsize(),
                    self.config.output_path,
                )
            return False

    def update_overlay(
        self,
        capture_frame_id: int,
        detections: Any,
        *,
        tracks: Any = (),
        control: Optional[VideoControlOverlay] = None,
    ) -> bool:
        """Attach a complete diagnostic snapshot without touching frame pixels."""
        capture_id = int(capture_frame_id)
        normalized: list[VideoDetectionOverlay] = []
        for detection in detections or ():
            bbox = tuple(float(value) for value in getattr(detection, "bbox", ()))
            if len(bbox) != 4:
                continue
            normalized.append(
                VideoDetectionOverlay(
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    score=float(getattr(detection, "score", 0.0)),
                    class_id=int(getattr(detection, "class_id", -1)),
                )
            )
        normalized_tracks: list[VideoTrackOverlay] = []
        for track in tracks or ():
            if not isinstance(track, VideoTrackOverlay):
                continue
            if len(track.bbox) != 4:
                continue
            normalized_tracks.append(track)
        normalized_control = control if isinstance(control, VideoControlOverlay) else VideoControlOverlay()
        overlay = VideoFrameOverlay(
            detections=tuple(normalized),
            tracks=tuple(normalized_tracks),
            control=normalized_control,
        )
        with self._overlay_condition:
            if capture_id not in self._submitted_capture_ids or capture_id in self._written_capture_ids:
                return False
            self._overlays[capture_id] = overlay
            self._overlay_condition.notify_all()
        return True

    def close(self, timeout_sec: float = 3.0) -> bool:
        self._closing.set()
        thread = self._thread
        if thread is None:
            self._closed.set()
            return True
        thread.join(timeout=max(0.1, float(timeout_sec)))
        if thread.is_alive():
            self._logger.warning(
                "Camera recorder close timed out: written=%d queued=%d output=%s",
                self._written,
                self._queue.qsize(),
                self.config.output_path,
            )
            return False
        return self._error is None

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="camera-video-recorder",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        writer = None
        index_file = None
        try:
            while not self._closing.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if writer is None:
                    height, width = int(item.image.shape[0]), int(item.image.shape[1])
                    output_path = Path(self.config.output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = str(self.config.fourcc or "MJPG").upper()
                    if len(fourcc) != 4:
                        raise ValueError(f"video recorder fourcc must have four characters: {fourcc!r}")
                    writer = self._cv2.VideoWriter(
                        str(output_path),
                        self._cv2.VideoWriter_fourcc(*fourcc),
                        max(1.0, float(self.config.fps)),
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open camera video output: {output_path}")
                    index_file = open(self.index_path, "w", newline="", buffering=1, encoding="utf-8")
                    csv_writer = csv.writer(index_file)
                    csv_writer.writerow(
                        [
                            "video_frame_index",
                            "capture_frame_id",
                            "control_frame_id",
                            "capture_monotonic_sec",
                            "capture_unix_sec",
                            "active_target_id",
                            "selected_target_id",
                            "action",
                            "requested_rpm",
                            "yaw_rate_dps",
                            "result_age_ms",
                            "decision_reason",
                        ]
                    )
                    self._logger.info(
                        "Camera recording started: output=%s index=%s index_schema=video_frame_index(encoded_order),capture_frame_id(camera_order),control_frame_id(optional) size=%dx%d fps=%.2f codec=%s",
                        output_path,
                        self.index_path,
                        width,
                        height,
                        float(self.config.fps),
                        fourcc,
                    )

                video_frame_number = self._written + 1
                overlay = self._wait_for_overlay(
                    item.capture_frame_id,
                    item.monotonic_sec,
                )
                writer.write(
                    self._annotate_frame(
                        item.image,
                        video_frame_number,
                        item.capture_frame_id,
                        overlay,
                    )
                )
                control = overlay.control
                csv_writer.writerow(
                    [
                        self._written,
                        item.capture_frame_id,
                        (
                            control.control_frame_id
                            if control.control_frame_id is not None
                            else ("" if item.control_frame_id is None else item.control_frame_id)
                        ),
                        f"{item.monotonic_sec:.6f}",
                        f"{item.unix_sec:.6f}",
                        "" if control.active_target_id is None else control.active_target_id,
                        "" if control.selected_target_id is None else control.selected_target_id,
                        control.action_name,
                        control.requested_rpm,
                        "" if control.yaw_rate_dps is None else f"{control.yaw_rate_dps:.3f}",
                        f"{control.result_age_ms:.3f}",
                        control.decision_reason,
                    ]
                )
                self._written += 1
                with self._overlay_condition:
                    self._written_capture_ids.add(int(item.capture_frame_id))
                    self._submitted_capture_ids.discard(int(item.capture_frame_id))
                    self._overlays.pop(int(item.capture_frame_id), None)
                self._queue.task_done()
        except BaseException as exc:
            self._error = exc
            self._logger.error("Camera recording disabled after writer failure: %s", exc)
        finally:
            if writer is not None:
                writer.release()
            if index_file is not None:
                index_file.close()
            self._closed.set()
            self._logger.info(
                "Camera recording closed: submitted=%d written=%d dropped=%d output=%s",
                self._submitted,
                self._written,
                self._dropped,
                self.config.output_path,
            )

    def _wait_for_overlay(
        self,
        capture_frame_id: int,
        capture_monotonic_sec: float,
    ) -> VideoFrameOverlay:
        capture_id = int(capture_frame_id)
        timeout = max(0.0, float(self.config.overlay_wait_sec))
        # The recorder receives every camera frame, while the vision loop may
        # intentionally skip old frames. Give metadata one bounded window from
        # capture time; otherwise each skipped frame would add another full
        # timeout and eventually overflow the recording queue.
        deadline = float(capture_monotonic_sec) + timeout
        with self._overlay_condition:
            while capture_id not in self._overlays and timeout > 0.0:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._overlay_condition.wait(timeout=remaining)
            return self._overlays.pop(capture_id, VideoFrameOverlay())

    def _draw_text_box(
        self,
        image: Any,
        text: str,
        *,
        x: int,
        y: int,
        font: Any,
        scale: float,
        foreground: tuple[int, int, int],
        background: tuple[int, int, int],
        thickness: int = 1,
    ) -> int:
        height, width = int(image.shape[0]), int(image.shape[1])
        available = max(1, width - max(0, int(x)) - 4)
        rendered = str(text)
        while rendered:
            (text_width, text_height), baseline = self._cv2.getTextSize(
                rendered,
                font,
                scale,
                thickness,
            )
            if text_width <= available:
                break
            if len(rendered) <= 1:
                rendered = ""
                break
            rendered = rendered[:-2].rstrip() + "~"
        if not rendered:
            return int(y)
        bottom = min(height - 1, int(y) + 3)
        # Keep the video visible behind diagnostics.  The dark outline is a
        # text stroke, not a filled label background, so it remains readable
        # over both bright and dark parts of the camera image.
        outline_thickness = max(2, int(thickness) + 2)
        self._cv2.putText(
            image,
            rendered,
            (max(0, int(x)), int(y)),
            font,
            scale,
            (0, 0, 0),
            outline_thickness,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            image,
            rendered,
            (max(0, int(x)), int(y)),
            font,
            scale,
            foreground,
            thickness,
            self._cv2.LINE_AA,
        )
        return bottom

    @staticmethod
    def _clipped_bbox(
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> Optional[tuple[int, int, int, int]]:
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)

    def _annotate_frame(
        self,
        image: Any,
        frame_number: int,
        capture_frame_id: int,
        overlay: VideoFrameOverlay,
    ) -> Any:
        """Draw diagnostics on a copy in the recorder thread."""
        annotated = image.copy()
        height, width = int(annotated.shape[0]), int(annotated.shape[1])
        ruler_height = max(34, int(round(height * 0.065)))
        ruler_top = max(0, height - ruler_height)
        control = overlay.control
        control_id = "------" if control.control_frame_id is None else f"{control.control_frame_id:06d}"
        label = (
            f"VIDEO {int(frame_number):06d}  CAP {int(capture_frame_id):06d}  "
            f"CTRL {control_id}"
        )
        font = self._cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.9, width / 900.0))
        thickness = max(1, int(round(scale * 2.0)))
        (text_width, text_height), baseline = self._cv2.getTextSize(
            label,
            font,
            scale,
            thickness,
        )
        margin = max(8, int(round(width * 0.012)))
        x = max(0, width - text_width - margin)
        y = margin + text_height
        for detection in overlay.detections:
            clipped = self._clipped_bbox(detection.bbox, width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            color = (0, 220, 0) if int(detection.class_id) == 0 else (0, 165, 255)
            self._cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            det_label = f"YOLO c{int(detection.class_id)} {float(detection.score):.2f}"
            (label_width, label_height), label_baseline = self._cv2.getTextSize(
                det_label,
                font,
                max(0.4, min(0.7, scale * 0.85)),
                1,
            )
            label_x = x1
            label_y = max(label_height + label_baseline + 2, y1)
            label_x2 = min(width - 1, label_x + label_width + 6)
            label_y1 = max(0, label_y - label_height - label_baseline - 4)
            self._cv2.putText(
                annotated,
                det_label,
                (label_x, label_y),
                font,
                max(0.4, min(0.7, scale * 0.85)),
                (0, 255, 0),
                2,
                self._cv2.LINE_AA,
            )
            self._cv2.putText(
                annotated,
                det_label,
                (label_x, label_y),
                font,
                max(0.4, min(0.7, scale * 0.85)),
                (0, 0, 0),
                1,
                self._cv2.LINE_AA,
            )

        track_scale = max(0.36, min(0.58, scale * 0.72))
        for track in overlay.tracks:
            clipped = self._clipped_bbox(track.bbox, width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            color = (255, 80, 0) if track.active_target else (230, 230, 230)
            self._cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                4 if track.active_target else 1,
            )
            uid_text = f"U{track.reid_uid}"
            if track.mapped_uid > 0 and track.mapped_uid != track.reid_uid:
                uid_text += f">M{track.mapped_uid}"
            if track.best_uid > 0 and track.best_uid not in (track.reid_uid, track.mapped_uid):
                uid_text += f"?B{track.best_uid}"
            distance_text = "" if track.distance is None else f" d={track.distance:.3f}"
            prefix = "ACTIVE TARGET " if track.active_target else "TRACK "
            track_label = (
                f"{prefix}T{track.track_id} {uid_text}{distance_text} "
                f"{track.assignment_reason}"
            )
            if track.quality_reason:
                track_label += f" [{track.quality_reason}]"
            self._draw_text_box(
                annotated,
                track_label,
                x=x1,
                y=min(ruler_top - 5, max(13, y2 - 3)),
                font=font,
                scale=track_scale,
                foreground=(0, 255, 0) if track.active_target else (160, 255, 160),
                background=color,
            )

        if control.candidate_bbox is not None:
            candidate = self._clipped_bbox(control.candidate_bbox, width, height)
            if candidate is not None:
                x1, y1, x2, y2 = candidate
                candidate_color = (
                    (255, 80, 0)
                    if control.candidate_matches_target
                    else (255, 0, 255)
                )
                self._cv2.rectangle(annotated, (x1, y1), (x2, y2), candidate_color, 3)

        status_scale = max(0.36, min(0.58, scale * 0.70))
        status_y = margin + text_height + baseline + 22
        active_text = "none" if control.active_target_id is None else str(control.active_target_id)
        selected_text = "none" if control.selected_target_id is None else str(control.selected_target_id)
        yaw_text = "none" if control.yaw_rate_dps is None else f"{control.yaw_rate_dps:+.1f}dps"
        search_text = control.search_state
        if control.search_direction:
            search_text += f"/{control.search_direction}"
        if control.candidate_bbox is None:
            candidate_text = "CAND none"
        else:
            candidate_center = (
                float(control.candidate_bbox[0]) + float(control.candidate_bbox[2])
            ) / (2.0 * max(1, width))
            candidate_text = (
                f"CAND {control.candidate_source} score={control.candidate_score:.2f} "
                f"x={candidate_center:.3f} active={control.candidate_matches_target}"
            )
        status_lines = (
            f"ACTIVE U{active_text}  SELECT U{selected_text}  SEARCH {search_text}",
            candidate_text,
            f"MOTOR {control.action_name}  CMD {control.requested_rpm:+d}RPM  "
            f"YAW {yaw_text}  AGE {control.result_age_ms:.1f}ms",
            f"DEC {control.decision_reason}",
            "BLUE=ACTIVE TARGET  MAGENTA=OTHER CONTROL CANDIDATE",
        )
        for status_line in status_lines:
            self._draw_text_box(
                annotated,
                status_line,
                x=margin,
                y=status_y,
                font=font,
                scale=status_scale,
                foreground=(0, 255, 0),
                background=(0, 0, 0),
            )
            status_y += max(15, int(round(21 * status_scale / 0.45)))

        # Draw the frame identity last so raw detector labels can never cover
        # the capture/control timeline needed for log correlation.
        self._cv2.putText(
            annotated,
            label,
            (x, y),
            font,
            scale,
            (0, 255, 0),
            thickness + 2,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            annotated,
            label,
            (x, y),
            font,
            scale,
            (0, 255, 0),
            thickness,
            self._cv2.LINE_AA,
        )

        ruler_y = ruler_top + max(8, ruler_height // 3)
        self._cv2.line(annotated, (0, ruler_y), (width - 1, ruler_y), (0, 255, 0), 1)
        ruler_scale = max(0.30, min(0.48, width / 1700.0))
        for index in range(11):
            normalized_x = index / 10.0
            tick_x = int(round(normalized_x * (width - 1)))
            tick_height = 10 if index in (0, 5, 10) else 6
            tick_color = (255, 80, 0) if index == 5 else (0, 255, 0)
            self._cv2.line(
                annotated,
                (tick_x, ruler_y - tick_height),
                (tick_x, ruler_y + tick_height),
                tick_color,
                2 if index == 5 else 1,
            )
            tick_label = f"{normalized_x:.1f}"
            (tick_width, tick_text_height), _ = self._cv2.getTextSize(
                tick_label,
                font,
                ruler_scale,
                1,
            )
            label_x = max(0, min(width - tick_width, tick_x - tick_width // 2))
            label_y = min(height - 3, ruler_y + tick_height + tick_text_height + 2)
            self._cv2.putText(
                annotated,
                tick_label,
                (label_x, label_y),
                font,
                ruler_scale,
                tick_color,
                1,
                self._cv2.LINE_AA,
            )
        return annotated
