#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Passive search diagnostics kept outside recognition and vehicle control.

This module receives immutable observations from the runtime. It may log or
save images, but it cannot create control actions, modify PID state, select an
identity, or send motor commands.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None


SEARCHING_STATES = frozenset(("searching",))


@dataclass(frozen=True)
class SearchDiagnosticsConfig:
    enabled: bool = True
    interval_sec: float = 0.0
    checkpoint_deg: float = 30.0
    snapshot_enabled: bool = True
    snapshot_max: int = 24
    jpeg_quality: int = 88
    output_dir: str = "search_frames"
    person_class_id: int = 0
    formal_confidence: float = 0.25
    probe_confidence: float = 0.10
    decision_history_max: int = 15
    decision_image_max_width: int = 640


@dataclass(frozen=True)
class DetectionObservation:
    bbox: Tuple[float, float, float, float]
    score: float
    class_id: int

    @classmethod
    def from_object(cls, value: Any) -> "DetectionObservation":
        bbox = tuple(float(item) for item in getattr(value, "bbox", (0, 0, 0, 0)))
        return cls(
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            score=float(getattr(value, "score", 0.0)),
            class_id=int(getattr(value, "class_id", -1)),
        )


@dataclass(frozen=True)
class FrameQualityObservation:
    brightness: float
    contrast: float
    sharpness: float
    dark_ratio: float
    bright_ratio: float
    low_yaw_baseline: Optional[float]
    sharpness_ratio: Optional[float]
    state: str
    metric_ms: float


@dataclass(frozen=True)
class RecognitionObservation:
    formal_detections: Tuple[DetectionObservation, ...] = ()
    probe_detections: Tuple[DetectionObservation, ...] = ()
    tracks_total: int = 0
    fresh_tracks: int = 0
    predicted_tracks: int = 0
    assigned_uid_tracks: int = 0
    yolo_total_ms: float = 0.0
    yolo_inference_ms: float = 0.0
    yolo_nms_ms: float = 0.0
    reid_total_ms: float = 0.0
    tracker_ms: float = 0.0
    stale_result_discarded: bool = False


@dataclass(frozen=True)
class SearchControlObservation:
    state_before: str = "none"
    direction_before: Optional[str] = None
    state_after: str = "none"
    direction_after: Optional[str] = None
    active_target_id: Optional[int] = None
    selected_target_id: Optional[int] = None
    decision_reason: str = ""
    progress_deg: float = 0.0
    target_deg: float = 360.0
    elapsed_sec: Optional[float] = None
    stage: str = "inactive"
    heading_from_loss_deg: float = 0.0
    coverage_deg: float = 0.0
    travel_deg: float = 0.0
    hint_confidence: float = 0.0
    hint_source: str = "none"


@dataclass(frozen=True)
class DirectionEvidenceObservation:
    frame_index: int
    target_id: int
    bbox: Tuple[float, float, float, float]
    x_ratio: Optional[float]
    area: float
    distance_m: Optional[float]


@dataclass(frozen=True)
class MotionObservation:
    command_name: str = "none"
    requested_rotate_raw: int = 0
    requested_rotate_source: str = "none"
    left_speed_rpm: Optional[int] = None
    right_speed_rpm: Optional[int] = None
    yaw_rate_dps: Optional[float] = None
    raw_yaw_rate_dps: Optional[float] = None
    integrated_yaw_deg: Optional[float] = None
    feedback_age_ms: Optional[float] = None
    feedback_trustworthy: bool = False


@dataclass(frozen=True)
class TransportObservation:
    frame_gap_ms: Optional[float] = None
    camera_read_ms: float = 0.0
    camera_drained: int = 0
    result_age_ms: float = 0.0


@dataclass(frozen=True)
class SearchDiagnosticSample:
    frame_index: int
    timestamp: float
    width: int
    height: int
    recognition: RecognitionObservation
    control: SearchControlObservation
    motion: MotionObservation
    transport: TransportObservation
    quality: Optional[FrameQualityObservation] = None
    direction_evidence: Tuple[DirectionEvidenceObservation, ...] = ()
    image_frame: Any = field(default=None, repr=False, compare=False)
    frame_format: str = "BGR"


class SearchDiagnosticsObserver:
    """A write-only observer. ``observe`` intentionally returns ``None``."""

    def __init__(self, config: SearchDiagnosticsConfig, logger: Any) -> None:
        self.config = config
        self.logger = logger
        self._low_yaw_sharpness_ema: Optional[float] = None
        self._last_log_ts = 0.0
        self._session_id = 0
        self._active = False
        self._started_ts = 0.0
        self._stats: Dict[str, Any] = {}
        self._last_checkpoint = -1
        self._snapshot_count = 0
        self._decision_snapshot_count = 0
        self._special_snapshots = set()
        self._decision_frame_cache: Dict[int, Tuple[Any, str]] = {}

    def remember_direction_frame(
        self,
        frame_index: int,
        image_frame: Any,
        frame_format: str,
    ) -> None:
        """Keep only recent accepted direction frames for later diagnostics."""
        if (
            not self.config.enabled
            or not self.config.snapshot_enabled
            or cv2 is None
            or image_frame is None
            or not hasattr(image_frame, "shape")
        ):
            return
        try:
            output = image_frame
            max_width = max(160, int(self.config.decision_image_max_width))
            width = int(output.shape[1])
            if width > max_width:
                scale = float(max_width) / float(width)
                output = cv2.resize(
                    output,
                    (max_width, max(1, int(round(int(output.shape[0]) * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                output = output.copy()
            self._decision_frame_cache[int(frame_index)] = (
                output,
                str(frame_format or "BGR"),
            )
            keep = max(1, int(self.config.decision_history_max))
            for old_frame in sorted(self._decision_frame_cache)[:-keep]:
                self._decision_frame_cache.pop(old_frame, None)
        except Exception as exc:
            self.logger.warning(
                "search_decision_frame_cache_failed frame=%d error=%s",
                int(frame_index),
                exc,
            )

    @property
    def active(self) -> bool:
        return bool(self._active)

    @staticmethod
    def _fmt(value: Optional[float]) -> str:
        return "none" if value is None else "%.3f" % float(value)

    def measure_frame_quality(
        self,
        frame: Any,
        frame_format: str,
        *,
        update_low_yaw_baseline: bool,
    ) -> Optional[FrameQualityObservation]:
        if not self.config.enabled or cv2 is None or frame is None or not hasattr(frame, "shape"):
            return None
        started = time.perf_counter()
        try:
            fmt = str(frame_format or "BGR").strip().upper()
            channels = int(frame.shape[2]) if len(frame.shape) >= 3 else 1
            if channels == 1:
                gray = frame
            elif channels == 4:
                code = cv2.COLOR_RGBA2GRAY if fmt.startswith("RGB") else cv2.COLOR_BGRA2GRAY
                gray = cv2.cvtColor(frame, code)
            else:
                code = cv2.COLOR_RGB2GRAY if fmt.startswith("RGB") else cv2.COLOR_BGR2GRAY
                gray = cv2.cvtColor(frame, code)
            height, width = gray.shape[:2]
            if width > 320:
                scale = 320.0 / float(width)
                gray = cv2.resize(
                    gray,
                    (320, max(1, int(round(height * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            mean, stddev = cv2.meanStdDev(gray)
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            _, lap_stddev = cv2.meanStdDev(laplacian)
            pixels = max(1, int(gray.shape[0]) * int(gray.shape[1]))
            brightness = float(mean[0][0])
            contrast = float(stddev[0][0])
            sharpness = float(lap_stddev[0][0]) ** 2
            dark_ratio = float(cv2.countNonZero(cv2.inRange(gray, 0, 20))) / float(pixels)
            bright_ratio = float(cv2.countNonZero(cv2.inRange(gray, 235, 255))) / float(pixels)
            if update_low_yaw_baseline:
                if self._low_yaw_sharpness_ema is None:
                    self._low_yaw_sharpness_ema = sharpness
                else:
                    self._low_yaw_sharpness_ema = (
                        0.95 * float(self._low_yaw_sharpness_ema) + 0.05 * sharpness
                    )
            baseline = self._low_yaw_sharpness_ema
            ratio = None if baseline is None or baseline <= 1e-6 else sharpness / baseline
            if dark_ratio >= 0.70 or brightness <= 25.0:
                state = "underexposed"
            elif bright_ratio >= 0.70 or brightness >= 230.0:
                state = "overexposed"
            elif ratio is None:
                state = "baseline_unavailable"
            elif ratio < 0.35:
                state = "blur_suspected"
            else:
                state = "normal_or_scene_change"
            return FrameQualityObservation(
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
                dark_ratio=dark_ratio,
                bright_ratio=bright_ratio,
                low_yaw_baseline=baseline,
                sharpness_ratio=ratio,
                state=state,
                metric_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            self.logger.warning("search_quality_measure_failed error=%s", exc)
            return None

    def _begin(self, sample: SearchDiagnosticSample) -> None:
        self._session_id += 1
        self._active = True
        self._started_ts = float(sample.timestamp)
        self._last_checkpoint = -1
        self._snapshot_count = 0
        self._decision_snapshot_count = 0
        self._special_snapshots = set()
        self._stats = {
            "frames": 0,
            "stage_counts": {},
            "image_counts": {},
            "raw_person_frames": 0,
            "probe_person_frames": 0,
            "assigned_uid_frames": 0,
            "stale_frames": 0,
            "max_raw_score": 0.0,
            "max_probe_score": 0.0,
            "frame_gap_sum": 0.0,
            "frame_gap_count": 0,
            "frame_gap_max": 0.0,
            "yaw_sum": 0.0,
            "yaw_count": 0,
            "yaw_max": 0.0,
            "sharpness_ratio_sum": 0.0,
            "sharpness_ratio_count": 0,
            "sharpness_ratio_min": None,
            "yolo_sum": 0.0,
            "yolo_max": 0.0,
            "result_age_sum": 0.0,
            "result_age_max": 0.0,
            "max_progress_deg": 0.0,
        }
        c = sample.control
        self.logger.info(
            "search_session_start session=%d frame=%d direction=%s active_uid=%s "
            "formal_conf=%.2f probe_conf=%.2f checkpoint=%.1fdeg snapshot=%s dir=%s",
            self._session_id,
            sample.frame_index,
            c.direction_before or c.direction_after or "unknown",
            "none" if c.active_target_id is None else c.active_target_id,
            self.config.formal_confidence,
            self.config.probe_confidence,
            self.config.checkpoint_deg,
            self.config.snapshot_enabled,
            self.config.output_dir,
        )
        self._record_search_decision(sample)

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in str(value))[:32]

    @staticmethod
    def _to_bgr(image_frame: Any, frame_format: str) -> Any:
        output = image_frame
        fmt = str(frame_format or "BGR").strip().upper()
        channels = int(output.shape[2]) if len(output.shape) >= 3 else 1
        if channels == 4:
            code = cv2.COLOR_RGBA2BGR if fmt.startswith("RGB") else cv2.COLOR_BGRA2BGR
            return cv2.cvtColor(output, code)
        if channels == 3 and fmt.startswith("RGB"):
            return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def _write_decision_image(
        self,
        image_frame: Any,
        frame_format: str,
        filename: str,
    ) -> Optional[str]:
        if (
            not self.config.snapshot_enabled
            or cv2 is None
            or image_frame is None
            or not hasattr(image_frame, "shape")
        ):
            return None
        directory = os.path.join(self.config.output_dir, "search_%03d" % self._session_id)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        output = self._to_bgr(image_frame, frame_format)
        if not cv2.imwrite(
            path,
            output,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        ):
            return None
        self._decision_snapshot_count += 1
        return path

    def _record_search_decision(self, sample: SearchDiagnosticSample) -> None:
        c = sample.control
        direction = c.direction_after or c.direction_before or "unknown"
        current_name = "decision_current_frame_%06d_direction_%s.jpg" % (
            int(sample.frame_index),
            self._safe_name(direction),
        )
        current_path = None
        try:
            current_path = self._write_decision_image(
                sample.image_frame,
                sample.frame_format,
                current_name,
            )
        except Exception as exc:
            self.logger.warning(
                "search_decision_current_write_failed session=%d frame=%d error=%s",
                self._session_id,
                int(sample.frame_index),
                exc,
            )
        evidence = tuple(sample.direction_evidence)[-max(1, int(self.config.decision_history_max)):]
        self.logger.info(
            "search_decision_frame session=%d frame=%d direction=%s hint_source=%s "
            "hint_conf=%.2f mode=%s evidence_frames=%d current_path=%s",
            self._session_id,
            int(sample.frame_index),
            direction,
            c.hint_source,
            float(c.hint_confidence),
            c.stage,
            len(evidence),
            current_path or "not_saved",
        )
        if not evidence:
            self.logger.info(
                "search_decision_evidence session=%d result=none reason=no_reliable_history",
                self._session_id,
            )
            return
        first_x = evidence[0].x_ratio
        for order, item in enumerate(evidence, start=1):
            cached = self._decision_frame_cache.get(int(item.frame_index))
            path = None
            if cached is not None:
                name = "decision_evidence_%02d_frame_%06d_x_%s.jpg" % (
                    order,
                    int(item.frame_index),
                    "none" if item.x_ratio is None else "%04d" % int(round(item.x_ratio * 1000.0)),
                )
                try:
                    path = self._write_decision_image(cached[0], cached[1], name)
                except Exception as exc:
                    self.logger.warning(
                        "search_decision_evidence_write_failed session=%d frame=%d error=%s",
                        self._session_id,
                        int(item.frame_index),
                        exc,
                    )
            dx = (
                None
                if first_x is None or item.x_ratio is None
                else float(item.x_ratio) - float(first_x)
            )
            self.logger.info(
                "search_decision_evidence session=%d order=%d/%d frame=%d target=%d "
                "x=%s dx_from_first=%s bbox=(%.0f,%.0f,%.0f,%.0f) area=%.0f "
                "distance=%sm image=%s",
                self._session_id,
                order,
                len(evidence),
                int(item.frame_index),
                int(item.target_id),
                self._fmt(item.x_ratio),
                self._fmt(dx),
                *item.bbox,
                float(item.area),
                self._fmt(item.distance_m),
                path or "cache_miss",
            )

    def _people(self, detections: Tuple[DetectionObservation, ...]) -> Tuple[DetectionObservation, ...]:
        return tuple(item for item in detections if item.class_id == self.config.person_class_id)

    def _format_people(
        self,
        detections: Tuple[DetectionObservation, ...],
        width: int,
        height: int,
    ) -> str:
        people = self._people(detections)
        if not people:
            return "none"
        area = float(max(1, width * height))
        parts = []
        for item in sorted(people, key=lambda value: value.score, reverse=True)[:3]:
            x1, y1, x2, y2 = item.bbox
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            edges = sum((x1 <= 1.0, y1 <= 1.0, x2 >= width - 1.0, y2 >= height - 1.0))
            parts.append(
                "score=%.3f bbox=(%.0f,%.0f,%.0f,%.0f) area_ratio=%.3f edge=%d"
                % (item.score, x1, y1, x2, y2, box_area / area, edges)
            )
        return ";".join(parts)

    def _classify(self, sample: SearchDiagnosticSample) -> Tuple[str, Tuple, Tuple]:
        r = sample.recognition
        formal = self._people(r.formal_detections)
        probe = tuple(
            item
            for item in self._people(r.probe_detections)
            if item.score < self.config.formal_confidence
        )
        if r.stale_result_discarded:
            stage = "stale_result_discarded"
        elif not formal and probe:
            stage = "detector_below_threshold"
        elif not formal:
            stage = "detector_empty"
        elif r.fresh_tracks <= 0:
            stage = "tracker_empty"
        elif r.assigned_uid_tracks <= 0:
            stage = "reid_unassigned"
        else:
            stage = "candidate_available"
        return stage, formal, probe

    def _update_stats(
        self,
        sample: SearchDiagnosticSample,
        stage: str,
        formal: Tuple[DetectionObservation, ...],
        probe: Tuple[DetectionObservation, ...],
    ) -> None:
        stats = self._stats
        stats["frames"] += 1
        stats["max_progress_deg"] = max(
            stats["max_progress_deg"],
            float(sample.control.progress_deg),
        )
        stages = stats["stage_counts"]
        stages[stage] = stages.get(stage, 0) + 1
        image_state = "metrics_unavailable" if sample.quality is None else sample.quality.state
        images = stats["image_counts"]
        images[image_state] = images.get(image_state, 0) + 1
        if formal:
            stats["raw_person_frames"] += 1
            stats["max_raw_score"] = max(stats["max_raw_score"], max(item.score for item in formal))
        if probe:
            stats["probe_person_frames"] += 1
            stats["max_probe_score"] = max(stats["max_probe_score"], max(item.score for item in probe))
        if sample.recognition.assigned_uid_tracks > 0:
            stats["assigned_uid_frames"] += 1
        if sample.recognition.stale_result_discarded:
            stats["stale_frames"] += 1
        gap = sample.transport.frame_gap_ms
        if gap is not None:
            stats["frame_gap_sum"] += gap
            stats["frame_gap_count"] += 1
            stats["frame_gap_max"] = max(stats["frame_gap_max"], gap)
        yaw = sample.motion.yaw_rate_dps
        if yaw is not None:
            yaw = abs(yaw)
            stats["yaw_sum"] += yaw
            stats["yaw_count"] += 1
            stats["yaw_max"] = max(stats["yaw_max"], yaw)
        ratio = None if sample.quality is None else sample.quality.sharpness_ratio
        if ratio is not None:
            stats["sharpness_ratio_sum"] += ratio
            stats["sharpness_ratio_count"] += 1
            old_min = stats["sharpness_ratio_min"]
            stats["sharpness_ratio_min"] = ratio if old_min is None else min(old_min, ratio)
        yolo = sample.recognition.yolo_total_ms
        stats["yolo_sum"] += yolo
        stats["yolo_max"] = max(stats["yolo_max"], yolo)
        age = sample.transport.result_age_ms
        stats["result_age_sum"] += age
        stats["result_age_max"] = max(stats["result_age_max"], age)

    def _save_snapshot(self, sample: SearchDiagnosticSample, stage: str, reason: str) -> None:
        if (
            not self.config.snapshot_enabled
            or self.config.snapshot_max <= 0
            or self._snapshot_count >= self.config.snapshot_max
            or cv2 is None
            or sample.image_frame is None
            or not hasattr(sample.image_frame, "shape")
        ):
            return
        try:
            directory = os.path.join(self.config.output_dir, "search_%03d" % self._session_id)
            os.makedirs(directory, exist_ok=True)
            output = sample.image_frame
            fmt = sample.frame_format.strip().upper()
            channels = int(output.shape[2]) if len(output.shape) >= 3 else 1
            if channels == 4:
                code = cv2.COLOR_RGBA2BGR if fmt.startswith("RGB") else cv2.COLOR_BGRA2BGR
                output = cv2.cvtColor(output, code)
            elif channels == 3 and fmt.startswith("RGB"):
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            name = "frame_%06d_angle_%06.1f_%s_%s.jpg" % (
                sample.frame_index,
                sample.control.progress_deg,
                self._safe_name(stage),
                self._safe_name(reason),
            )
            path = os.path.join(directory, name)
            if not cv2.imwrite(path, output, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]):
                self.logger.warning("search_snapshot_write_failed frame=%d path=%s", sample.frame_index, path)
                return
            self._snapshot_count += 1
            self.logger.info(
                "search_snapshot_saved session=%d frame=%d angle=%.1fdeg stage=%s reason=%s "
                "count=%d/%d path=%s",
                self._session_id,
                sample.frame_index,
                sample.control.progress_deg,
                stage,
                reason,
                self._snapshot_count,
                self.config.snapshot_max,
                path,
            )
        except Exception as exc:
            self.logger.warning("search_snapshot_exception frame=%d error=%s", sample.frame_index, exc)

    @staticmethod
    def _average(stats: Dict[str, Any], sum_key: str, count_key: str) -> Optional[float]:
        count = int(stats.get(count_key, 0))
        return None if count <= 0 else float(stats.get(sum_key, 0.0)) / float(count)

    def finish(self, timestamp: float, *, outcome: str, progress_deg: float) -> None:
        if not self._active:
            return
        s = self._stats
        scan_deg = max(float(progress_deg), float(s.get("max_progress_deg", 0.0)))
        self.logger.info(
            "search_session_summary session=%d outcome=%s frames=%d elapsed=%.3fs scan=%.1fdeg "
            "stages=%s images=%s raw_person_frames=%d probe_person_frames=%d assigned_uid_frames=%d "
            "stale_frames=%d max_raw_score=%.3f max_probe_score=%.3f frame_gap_avg=%sms "
            "frame_gap_max=%.2fms yaw_abs_avg=%sdps yaw_abs_max=%.2fdps sharpness_ratio_avg=%s "
            "sharpness_ratio_min=%s yolo_avg=%.2fms yolo_max=%.2fms result_age_avg=%.2fms "
            "result_age_max=%.2fms snapshots=%d decision_snapshots=%d dir=%s",
            self._session_id,
            outcome,
            s["frames"],
            max(0.0, timestamp - self._started_ts),
            scan_deg,
            json.dumps(s["stage_counts"], sort_keys=True, ensure_ascii=True),
            json.dumps(s["image_counts"], sort_keys=True, ensure_ascii=True),
            s["raw_person_frames"],
            s["probe_person_frames"],
            s["assigned_uid_frames"],
            s["stale_frames"],
            s["max_raw_score"],
            s["max_probe_score"],
            self._fmt(self._average(s, "frame_gap_sum", "frame_gap_count")),
            s["frame_gap_max"],
            self._fmt(self._average(s, "yaw_sum", "yaw_count")),
            s["yaw_max"],
            self._fmt(self._average(s, "sharpness_ratio_sum", "sharpness_ratio_count")),
            self._fmt(s["sharpness_ratio_min"]),
            self._average(s, "yolo_sum", "frames") or 0.0,
            s["yolo_max"],
            self._average(s, "result_age_sum", "frames") or 0.0,
            s["result_age_max"],
            self._snapshot_count,
            self._decision_snapshot_count,
            self.config.output_dir,
        )
        self._active = False

    def observe(self, sample: SearchDiagnosticSample) -> None:
        if not self.config.enabled:
            return None
        c = sample.control
        relevant = c.state_before in SEARCHING_STATES or c.state_after in SEARCHING_STATES
        if not relevant:
            return None
        ending = c.state_before in SEARCHING_STATES and c.state_after not in SEARCHING_STATES
        if not self._active:
            self._begin(sample)
        if not ending and sample.timestamp - self._last_log_ts < self.config.interval_sec:
            return None
        self._last_log_ts = sample.timestamp
        stage, formal, probe = self._classify(sample)
        self._update_stats(sample, stage, formal, probe)

        checkpoint = int(c.progress_deg // self.config.checkpoint_deg)
        checkpoint_changed = checkpoint > self._last_checkpoint
        if checkpoint_changed:
            self._last_checkpoint = checkpoint
            self.logger.info(
                "search_checkpoint session=%d frame=%d checkpoint=%d angle=%.1fdeg frames=%d "
                "stages=%s images=%s raw_person_frames=%d probe_person_frames=%d",
                self._session_id,
                sample.frame_index,
                checkpoint,
                c.progress_deg,
                self._stats["frames"],
                json.dumps(self._stats["stage_counts"], sort_keys=True, ensure_ascii=True),
                json.dumps(self._stats["image_counts"], sort_keys=True, ensure_ascii=True),
                self._stats["raw_person_frames"],
                self._stats["probe_person_frames"],
            )

        quality_state = "metrics_unavailable" if sample.quality is None else sample.quality.state
        reason = None
        if probe and "below_threshold" not in self._special_snapshots:
            reason = "below_threshold"
            self._special_snapshots.add(reason)
        elif formal and "formal_person" not in self._special_snapshots:
            reason = "formal_person"
            self._special_snapshots.add(reason)
        elif quality_state == "blur_suspected" and "blur" not in self._special_snapshots:
            reason = "blur"
            self._special_snapshots.add(reason)
        elif checkpoint_changed:
            reason = "checkpoint_%03d" % checkpoint
        elif ending:
            reason = "search_end"
        if reason is not None:
            self._save_snapshot(sample, stage, reason)

        r = sample.recognition
        q = sample.quality
        t = sample.transport
        m = sample.motion
        frame_yaw = None if m.yaw_rate_dps is None or t.frame_gap_ms is None else abs(m.yaw_rate_dps) * t.frame_gap_ms / 1000.0
        self.logger.info(
            "search_vision_diag session=%d frame=%d stage=%s formal_person=%d probe_person=%d "
            "formal_conf=%.2f probe_conf=%.2f tracks=%d fresh=%d predicted=%d assigned_uid=%d "
            "formal=[%s] probe=[%s] image=(state=%s sharpness=%s baseline=%s ratio=%s "
            "brightness=%s contrast=%s metric_ms=%s) timing=(yolo=%.2fms inference=%.2fms "
            "nms=%.2fms reid=%.2fms tracker=%.2fms result_age=%.2fms stale=%s)",
            self._session_id, sample.frame_index, stage, len(formal), len(probe),
            self.config.formal_confidence, self.config.probe_confidence,
            r.tracks_total, r.fresh_tracks, r.predicted_tracks, r.assigned_uid_tracks,
            self._format_people(formal, sample.width, sample.height),
            self._format_people(probe, sample.width, sample.height),
            quality_state,
            self._fmt(None if q is None else q.sharpness),
            self._fmt(None if q is None else q.low_yaw_baseline),
            self._fmt(None if q is None else q.sharpness_ratio),
            self._fmt(None if q is None else q.brightness),
            self._fmt(None if q is None else q.contrast),
            self._fmt(None if q is None else q.metric_ms),
            r.yolo_total_ms, r.yolo_inference_ms, r.yolo_nms_ms, r.reid_total_ms,
            r.tracker_ms, t.result_age_ms, r.stale_result_discarded,
        )
        self.logger.info(
            "search_control_diag session=%d frame=%d state=%s/%s->%s/%s progress=%.1f/%.1fdeg "
            "elapsed=%ss active_uid=%s selected_uid=%s decision=%s mode=%s "
            "heading_from_loss=%.1fdeg coverage=%.1fdeg travel=%.1fdeg "
            "hint_conf=%.2f hint_source=%s",
            self._session_id, sample.frame_index, c.state_before, c.direction_before or "none",
            c.state_after, c.direction_after or "none", c.progress_deg, c.target_deg,
            self._fmt(c.elapsed_sec), "none" if c.active_target_id is None else c.active_target_id,
            "none" if c.selected_target_id is None else c.selected_target_id,
            c.decision_reason or "none",
            c.stage,
            c.heading_from_loss_deg,
            c.coverage_deg,
            c.travel_deg,
            c.hint_confidence,
            c.hint_source,
        )
        self.logger.info(
            "search_motion_diag session=%d frame=%d command=%s request=%d/%s "
            "encoder=(left=%sRPM right=%sRPM yaw=%sdps raw_yaw=%sdps frame_yaw=%sdeg "
            "integrated=%sdeg age=%sms trustworthy=%s) transport=(frame_gap=%sms "
            "camera_read=%.2fms drained=%d)",
            self._session_id, sample.frame_index, m.command_name, m.requested_rotate_raw,
            m.requested_rotate_source, "none" if m.left_speed_rpm is None else m.left_speed_rpm,
            "none" if m.right_speed_rpm is None else m.right_speed_rpm,
            self._fmt(m.yaw_rate_dps), self._fmt(m.raw_yaw_rate_dps), self._fmt(frame_yaw),
            self._fmt(m.integrated_yaw_deg), self._fmt(m.feedback_age_ms),
            m.feedback_trustworthy, self._fmt(t.frame_gap_ms), t.camera_read_ms, t.camera_drained,
        )
        if ending:
            outcome = (
                "reacquired_uid_%d" % c.selected_target_id
                if c.selected_target_id is not None
                else c.decision_reason or ("state_" + c.state_after)
            )
            self.finish(sample.timestamp, outcome=outcome, progress_deg=c.progress_deg)
        return None
