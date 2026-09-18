from __future__ import annotations

import csv
import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .video_follow_telemetry import FollowRecordingView
from .video_depth_overlay import DepthVideoView, draw_depth_overlay


@dataclass(frozen=True)
class VideoRecorderConfig:
    output_path: str
    fps: float
    fourcc: str = "MJPG"
    queue_capacity: int = 60
    # Detection metadata arrives after the camera frame has been captured.
    # Waiting happens only in the recorder thread, never in control or capture.
    overlay_wait_sec: float = 0.25
    # Diagnostic text is translucent so the camera image remains visible when
    # several labels occupy the same area.
    overlay_text_alpha: float = 0.62
    # Only the video worker uses this encoder; never changes global OpenCV
    # thread settings shared by the control/vision pipeline.
    fast_mjpeg: bool = True
    jpeg_quality: int = 90


@dataclass(frozen=True)
class _QueuedFrame:
    image: Any
    capture_frame_id: int
    control_frame_id: Optional[int]
    monotonic_sec: float
    unix_sec: float
    wheel_feedback: Any = None
    follow_snapshot: Any = None
    linear_timing: Any = None


@dataclass(frozen=True)
class VideoWheelOverlay:
    left_rpm: Optional[float] = None
    right_rpm: Optional[float] = None
    sample_timestamp: Optional[float] = None
    age_ms: Optional[float] = None
    status: str = "missing"
    yaw_rate_dps: Optional[float] = None

    @classmethod
    def from_feedback(cls, feedback, capture_timestamp):
        if feedback is None:
            return cls()
        try:
            stamp = float(feedback.timestamp)
            left = float(feedback.left_forward_rpm)
            right = float(feedback.right_forward_rpm)
            age = (float(capture_timestamp)-stamp)*1000.
            if not all(math.isfinite(v) for v in (stamp, left, right, age)) or stamp <= 0:
                return cls(status="invalid")
            # Never paint a later sample onto an earlier captured image.
            if age < 0:
                return cls(sample_timestamp=stamp, age_ms=age, status="future")
            status = ("untrusted" if not feedback.trustworthy else
                      "stale" if age > 150. else "fresh")
            yaw = getattr(feedback, 'yaw_rate_right_dps', None)
            yaw = None if yaw is None or not math.isfinite(float(yaw)) else float(yaw)
            return cls(left, right, stamp, age, status, yaw)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return cls(status="invalid")

    def label(self):
        values = ("L n/a  R n/a" if self.left_rpm is None or self.right_rpm is None else
                  f"L {self.left_rpm:+.1f}  R {self.right_rpm:+.1f}")
        age = "n/a" if self.age_ms is None else f"{self.age_ms:.0f}ms"
        yaw = 'n/a' if self.yaw_rate_dps is None else f'{self.yaw_rate_dps:+.1f}'
        return f"WHEEL {values} RPM  {self.status.upper()} AGE {age} YAW {yaw}dps"


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
    target_distance_m: Optional[float] = None
    distance_source: str = "none"
    distance_detail: str = ""
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
        depth_sample_provider=None,
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
        self._sharpness_error_logged = False
        self._depth_sample_provider = depth_sample_provider
        self._depth_overlay_error_logged = False
        self._recording_timings = deque(maxlen=120)
        self._recording_timing_last_log = 0.0
        self._last_seen_capture_id = None

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
        wheel_feedback: Any = None,
        follow_snapshot: Any = None,
        linear_timing: Any = None,
    ) -> bool:
        if self._closing.is_set() or self._error is not None:
            return False
        self._ensure_thread()
        if capture_frame_id is None:
            if control_frame_index is None:
                raise ValueError("capture_frame_id is required")
            capture_frame_id = int(control_frame_index)
        self._last_seen_capture_id = int(capture_frame_id)
        # Skip the RGB copy when already full. Capture never waits for encode.
        if self._queue.full():
            return self._drop_frame(int(capture_frame_id))
        item = _QueuedFrame(
            image=image.copy(),
            capture_frame_id=int(capture_frame_id),
            control_frame_id=(None if control_frame_id is None else int(control_frame_id)),
            monotonic_sec=float(time.monotonic() if monotonic_sec is None else monotonic_sec),
            unix_sec=float(time.time() if unix_sec is None else unix_sec),
            wheel_feedback=wheel_feedback,
            follow_snapshot=follow_snapshot,
            linear_timing=linear_timing,
        )
        self._last_seen_capture_id = int(item.capture_frame_id)
        with self._overlay_condition:
            self._submitted_capture_ids.add(int(item.capture_frame_id))
        try:
            self._queue.put_nowait(item)
            self._submitted += 1
            return True
        except queue.Full:
            with self._overlay_condition:
                self._submitted_capture_ids.discard(int(item.capture_frame_id))
            return self._drop_frame(int(item.capture_frame_id))

    def _drop_frame(self, capture_id):
        self._dropped += 1
        now = time.monotonic()
        if now-self._last_drop_log_ts >= 1.:
            self._last_drop_log_ts = now
            self._logger.warning("Camera recorder queue full: dropped=%d queued=%d capture=%d output=%s",
                                 self._dropped, self._queue.qsize(), capture_id, self.config.output_path)
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
        first_capture_timestamp = None
        previous_capture_id = None
        backend = "opencv"
        try:
            while not self._closing.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                dequeue_time = time.monotonic()

                if writer is None:
                    height, width = int(item.image.shape[0]), int(item.image.shape[1])
                    output_path = Path(self.config.output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = str(self.config.fourcc or "MJPG").upper()
                    if len(fourcc) != 4:
                        raise ValueError(f"video recorder fourcc must have four characters: {fourcc!r}")
                    if (self.config.fast_mjpeg and fourcc == "MJPG"
                            and output_path.suffix.lower() == ".avi"
                            and hasattr(self._cv2, "imencode")):
                        try:
                            from .fast_mjpeg_writer import FastMjpegWriter
                            writer = FastMjpegWriter(output_path, max(1., float(self.config.fps)),
                                                     (width,height), self._cv2, self.config.jpeg_quality)
                            backend = writer.backend_name
                        except Exception as exc:
                            self._logger.warning("Fast JPEG recorder unavailable, using OpenCV: %s", exc)
                    if writer is None:
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
                            "target_distance_m",
                            "distance_source",
                            "distance_detail",
                            "decision_reason",
                            "sharpness",
                            "wheel_left_forward_rpm",
                            "wheel_right_forward_rpm",
                            "wheel_feedback_monotonic_sec",
                            "wheel_feedback_age_ms",
                            "wheel_feedback_status",
                            "wheel_feedback_yaw_rate_dps",
                            *FollowRecordingView.csv_columns(),
                            "depth_overlay_status", "depth_overlay_skew_ms",
                            "depth_overlay_sample_timestamp", "depth_overlay_uid", "depth_overlay_detail",
                            "depth_overlay_roi_capture_id",
                            "depth_overlay_regions_json",
                            "recording_time_sec", "recording_capture_gap",
                            "recording_queue_age_ms", "recording_wait_ms", "recording_sharpness_ms",
                            "recording_prepare_ms", "recording_draw_ms", "recording_encode_write_ms",
                            "recording_queue_size", "recording_dropped_total", "recording_backend",
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
                    first_capture_timestamp = item.monotonic_sec
                    self._logger.info("Camera recording backend=%s jpeg_quality=%d text=LINE_8",
                                      backend, self.config.jpeg_quality)

                video_frame_number = self._written + 1
                wait_start = time.monotonic()
                overlay = self._wait_for_overlay(
                    item.capture_frame_id,
                    item.monotonic_sec,
                )
                wait_end = time.monotonic()
                sharpness = self._measure_sharpness(item.image)
                sharpness_end = time.monotonic()
                wheels = VideoWheelOverlay.from_feedback(item.wheel_feedback, item.monotonic_sec)
                follow = FollowRecordingView.from_snapshot(
                    item.follow_snapshot, item.linear_timing, item.monotonic_sec)
                depth_view = self._depth_view(item.capture_frame_id, item.monotonic_sec,
                                              overlay.control.selected_target_id)
                prepare_end = time.monotonic()
                annotated = self._annotate_frame(
                    item.image, video_frame_number, item.capture_frame_id, overlay,
                    sharpness=sharpness, wheels=wheels, follow=follow, depth_view=depth_view,
                    capture_elapsed_sec=item.monotonic_sec-first_capture_timestamp,
                )
                draw_end = time.monotonic()
                writer.write(annotated)
                encode_end = time.monotonic()
                timings = [(dequeue_time-item.monotonic_sec)*1000,
                           (wait_end-wait_start)*1000, (sharpness_end-wait_end)*1000,
                           (prepare_end-sharpness_end)*1000, (draw_end-prepare_end)*1000,
                           (encode_end-draw_end)*1000]
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
                        "" if control.target_distance_m is None else f"{control.target_distance_m:.3f}",
                        control.distance_source,
                        control.distance_detail,
                        control.decision_reason,
                        "" if sharpness is None else f"{sharpness:.3f}",
                        "" if wheels.left_rpm is None else f"{wheels.left_rpm:.3f}",
                        "" if wheels.right_rpm is None else f"{wheels.right_rpm:.3f}",
                        "" if wheels.sample_timestamp is None else f"{wheels.sample_timestamp:.6f}",
                        "" if wheels.age_ms is None else f"{wheels.age_ms:.3f}",
                        wheels.status,
                        "" if wheels.yaw_rate_dps is None else f"{wheels.yaw_rate_dps:.3f}",
                        *follow.csv_values(),
                        *depth_view.csv_values(),
                        f"{item.monotonic_sec-first_capture_timestamp:.6f}",
                        0 if previous_capture_id is None else max(0,item.capture_frame_id-previous_capture_id-1),
                        *(f"{v:.3f}" for v in timings), self._queue.qsize(), self._dropped, backend,
                    ]
                )
                self._recording_timings.append((*timings, (time.monotonic()-encode_end)*1000))
                if encode_end-self._recording_timing_last_log >= 5.:
                    self._recording_timing_last_log = encode_end
                    self._log_recording_timings(backend)
                previous_capture_id = item.capture_frame_id
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
                try:
                    writer.release()
                except Exception as exc:
                    self._error = self._error or exc
                    self._logger.error("Camera recorder finalize failed: %s", exc)
            if index_file is not None:
                index_file.close()
            self._closed.set()
            self._log_recording_timings(backend)
            self._logger.info(
                "Camera recording closed: submitted=%d written=%d dropped=%d output=%s",
                self._submitted,
                self._written,
                self._dropped,
                self.config.output_path,
            )
            self._logger.info("Camera recording capture range: last_seen=%s last_written=%s",
                              self._last_seen_capture_id, previous_capture_id)

    def _log_recording_timings(self, backend):
        if not self._recording_timings:
            return
        names = ("queue_age", "wait", "sharpness", "prepare", "draw", "encode_write", "csv")
        parts = []
        for i, name in enumerate(names):
            values = sorted(row[i] for row in self._recording_timings)
            parts.append("%s_ms(avg=%.2f,p95=%.2f,max=%.2f)" % (
                name, sum(values)/len(values), values[min(len(values)-1,int(.95*len(values)))], values[-1]))
        self._logger.info("Camera recording timing: backend=%s n=%d queued=%d dropped=%d %s",
                          backend, len(self._recording_timings), self._queue.qsize(), self._dropped, " ".join(parts))

    def _depth_view(self, capture_id, timestamp, target_id):
        if self._depth_sample_provider is None:
            return DepthVideoView("disabled")
        try:
            sample = self._depth_sample_provider(capture_id, timestamp)
            return DepthVideoView.from_sample(sample, capture_id, timestamp, target_id)
        except Exception as exc:
            if not self._depth_overlay_error_logged:
                self._logger.warning("Video depth overlay unavailable: %s", exc)
                self._depth_overlay_error_logged = True
            return DepthVideoView("unavailable")

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

    def _measure_sharpness(self, image: Any) -> Optional[float]:
        """Measure the unannotated BGR frame only in the recorder worker."""
        try:
            channels = int(image.shape[2]) if len(image.shape) >= 3 else 1
            if channels == 1:
                gray = image
            else:
                code = self._cv2.COLOR_BGRA2GRAY if channels == 4 else self._cv2.COLOR_BGR2GRAY
                gray = self._cv2.cvtColor(image, code)
            height, width = gray.shape[:2]
            if width > 320:
                gray = self._cv2.resize(
                    gray,
                    (320, max(1, int(round(height * 320.0 / width)))),
                    interpolation=self._cv2.INTER_AREA,
                )
            laplacian = self._cv2.Laplacian(gray, self._cv2.CV_32F)
            _, stddev = self._cv2.meanStdDev(laplacian)
            return float(stddev[0][0]) ** 2
        except Exception as exc:
            if not self._sharpness_error_logged:
                self._logger.warning("Camera sharpness diagnostic unavailable: %s", exc)
                self._sharpness_error_logged = True
            return None

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
        max_width: Optional[int] = None,
    ) -> int:
        height, width = int(image.shape[0]), int(image.shape[1])
        available = max(1, width - max(0, int(x)) - 4)
        if max_width is not None:
            available = min(available, max(1, int(max_width)))
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
        outline_thickness = max(2, int(thickness) + 2)
        label_x = max(0, int(x))
        label_y = int(y)
        left = max(0, label_x - outline_thickness)
        right = min(width, label_x + text_width + outline_thickness + 2)
        top = max(0, label_y - text_height - outline_thickness - 2)
        bottom = min(height, label_y + baseline + outline_thickness + 2)
        if right <= left or bottom <= top:
            return int(y)
        # Blend only the small text region, leaving box/ruler geometry solid.
        # All pixel operations run in the recorder worker.
        region = image[top:bottom, left:right]
        text_layer = region.copy()
        origin = (label_x - left, label_y - top)
        self._cv2.putText(
            text_layer,
            rendered,
            origin,
            font,
            scale,
            (0, 0, 0),
            outline_thickness,
            self._cv2.LINE_8,
        )
        self._cv2.putText(
            text_layer,
            rendered,
            origin,
            font,
            scale,
            foreground,
            thickness,
            self._cv2.LINE_8,
        )
        alpha = max(0.0, min(1.0, float(self.config.overlay_text_alpha)))
        self._cv2.addWeighted(text_layer, alpha, region, 1.0 - alpha, 0.0, dst=region)
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
        *,
        sharpness: Optional[float] = None,
        wheels: VideoWheelOverlay = VideoWheelOverlay(),
        follow: FollowRecordingView = FollowRecordingView(),
        depth_view: DepthVideoView = DepthVideoView("disabled"),
        capture_elapsed_sec: Optional[float] = None,
    ) -> Any:
        """Draw diagnostics on a copy in the recorder thread."""
        annotated = image.copy()
        if depth_view.status != "disabled":
            try:
                draw_depth_overlay(annotated, depth_view, self._cv2)
            except Exception as exc:
                if not self._depth_overlay_error_logged:
                    self._logger.warning("Video depth drawing skipped: %s", exc)
                    self._depth_overlay_error_logged = True
        height, width = int(annotated.shape[0]), int(annotated.shape[1])
        ruler_height = max(34, int(round(height * 0.065)))
        ruler_top = max(0, height - ruler_height)
        control = overlay.control
        control_id = "------" if control.control_frame_id is None else f"{control.control_frame_id:06d}"
        label = (
            f"VIDEO {int(frame_number):06d}  CAP {int(capture_frame_id):06d}  "
            f"CTRL {control_id}"
        )
        if capture_elapsed_sec is not None:
            label += f"  T+{capture_elapsed_sec:.3f}s"
        font = self._cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.32, min(0.62, width / 1400.0))
        thickness = max(1, int(round(scale * 1.7)))
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
                max(0.30, min(0.52, scale * 0.82)),
                1,
            )
            label_x = x1
            label_y = max(label_height + label_baseline + 2, y1)
            self._draw_text_box(
                annotated,
                det_label,
                x=label_x,
                y=label_y,
                font=font,
                scale=max(0.30, min(0.52, scale * 0.82)),
                foreground=(0, 255, 0),
                background=(0, 0, 0),
            )

        track_scale = max(0.28, min(0.42, scale * 0.70))
        active_bbox: Optional[tuple[int, int, int, int]] = None
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
            if track.active_target:
                active_bbox = clipped
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

        status_scale = max(0.28, min(0.42, scale * 0.66))
        # Reserve a separate line below CAP/CTRL for the right-side clarity
        # readout; status text begins below it to avoid overlay collisions.
        quality_scale = max(0.28, min(0.42, scale * 0.78))
        quality_label = "SHARP n/a" if sharpness is None else f"SHARP {sharpness:.1f}"
        (quality_width, quality_height), quality_baseline = self._cv2.getTextSize(
            quality_label, font, quality_scale, 1
        )
        quality_y = y + baseline + quality_height + 5
        status_y = quality_y + quality_baseline + 16
        active_text = "none" if control.active_target_id is None else str(control.active_target_id)
        selected_text = "none" if control.selected_target_id is None else str(control.selected_target_id)
        yaw_text = "none" if control.yaw_rate_dps is None else f"{control.yaw_rate_dps:+.1f}dps"
        distance_text = (
            "none"
            if control.target_distance_m is None
            else f"{float(control.target_distance_m):.2f}m"
        )
        distance_source = str(control.distance_source or "none")
        if control.distance_detail:
            distance_source += f"/{control.distance_detail}"
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
            wheels.label(),
            *((f"DIST {distance_text}  SRC {distance_source}",) if active_bbox is None else ()),
            f"DEC {control.decision_reason}",
            "BLUE=ACTIVE  MAGENTA=CONTROL CANDIDATE",
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
            status_y += max(12, int(round(17 * status_scale / 0.35)))

        # Keep the new diagnostics near the bottom, separate from the existing
        # identity/search panel. Avoid covering the torso with five more lines.
        follow_lines = follow.labels()
        follow_step = max(12, int(round(17 * status_scale / 0.35)))
        follow_y = max(status_y + 5, ruler_top - len(follow_lines)*follow_step - 8)
        for follow_line in follow_lines:
            self._draw_text_box(
                annotated, follow_line, x=margin, y=follow_y, font=font,
                scale=status_scale, foreground=(0, 255, 255), background=(0, 0, 0),
            )
            follow_y += follow_step

        if active_bbox is not None:
            x1, y1, x2, y2 = active_bbox
            box_width = max(1, x2 - x1 - 8)
            box_height = max(1, y2 - y1)
            distance_line = f"DIST {distance_text}"
            source_line = f"SRC {distance_source}"
            inner_scale = max(0.24, min(0.40, scale * 0.62))
            line_step = max(11, int(round(15 * inner_scale / 0.30)))
            first_y = y1 + max(13, int(round(15 * inner_scale)))
            if first_y + line_step > y2 - 3:
                first_y = max(13, y2 - 3)
            self._draw_text_box(
                annotated,
                distance_line,
                x=x1 + 4,
                y=first_y,
                font=font,
                scale=inner_scale,
                foreground=(0, 255, 0),
                background=(255, 80, 0),
                max_width=box_width,
            )
            if box_height >= line_step + 20:
                self._draw_text_box(
                    annotated,
                    source_line,
                    x=x1 + 4,
                    y=min(y2 - 3, first_y + line_step),
                    font=font,
                    scale=inner_scale,
                    foreground=(0, 255, 0),
                    background=(255, 80, 0),
                    max_width=box_width,
                )

        # Draw the frame identity last so raw detector labels can never cover
        # the capture/control timeline needed for log correlation.
        self._draw_text_box(
            annotated,
            label,
            x=x,
            y=y,
            font=font,
            scale=scale,
            foreground=(0, 255, 0),
            background=(0, 0, 0),
            thickness=thickness,
        )
        self._draw_text_box(
            annotated,
            quality_label,
            x=max(0, width - quality_width - margin),
            y=quality_y,
            font=font,
            scale=quality_scale,
            foreground=(160, 255, 255),
            background=(0, 0, 0),
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
            label_x = max(0, min(width - tick_width - 4, tick_x - tick_width // 2))
            label_y = min(height - 3, ruler_y + tick_height + tick_text_height + 2)
            self._draw_text_box(
                annotated,
                tick_label,
                x=label_x,
                y=label_y,
                font=font,
                scale=ruler_scale,
                foreground=tick_color,
                background=(0, 0, 0),
            )
        return annotated
