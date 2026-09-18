"""Passive pose/depth experiment. Outputs files only; never returns control evidence.

The input is an owned RGB frame, its exact identity-bound detector geometry and
the actual sampled, registered depth array. No camera, detector, PID or motor is
created here. Model loading, inference and disk I/O run in one bounded worker.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np


@dataclass(frozen=True)
class PoseShadowConfig:
    adapter: str = "/home/topeet/AstraSDK/samples/arm64_viewers/mp_pose.py"
    model: str = "/home/topeet/AstraSDK/samples/arm64_viewers/models/pose_estimation_mediapipe_2023mar.onnx"
    interval_sec: float = 0.5
    max_input_age_sec: float = 0.25
    max_alignment_sec: float = 0.05
    landmark_confidence: float = 0.6
    max_samples: int = 120
    max_snapshots: int = 12
    snapshot_interval_sec: float = 5.0
    max_bytes: int = 64 * 1024 * 1024
    save_video: bool = True


def enabled() -> bool:
    return os.environ.get("FOLLOW_POSE_SHADOW_ENABLE", "0").lower() in {"1", "true", "yes"}


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def validate_packet(rgb, depth, metadata, config):
    """Reject mismatched historical snapshots instead of silently resizing them."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        return "missing_rgb"
    if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
        return "missing_depth"
    if depth.shape != rgb.shape[:2] or list(metadata.get("frame_size", [])) != [rgb.shape[1], rgb.shape[0]]:
        return "shape_mismatch"
    if rgb.nbytes + depth.nbytes > 8 * 1024 * 1024:
        return "frame_too_large"
    if (metadata.get("evidence_capture_frame_id") is None or
            metadata.get("rgb_capture_id") != metadata.get("evidence_capture_frame_id")):
        return "capture_mismatch"
    uid = finite(metadata.get("target_id"))
    if uid is None or uid <= 0 or uid != int(uid):
        return "unbound_target"
    stamps = [finite(metadata.get(k)) for k in
              ("rgb_timestamp", "reference_timestamp", "depth_timestamp", "sample_timestamp")]
    if any(s is None or s <= 0 for s in stamps):
        return "invalid_timestamp"
    rgb_ts, reference, depth_ts, sample_ts = stamps
    if abs(rgb_ts - reference) > 1e-6 or abs(depth_ts - sample_ts) > 1e-6:
        return "sample_mismatch"
    if abs(rgb_ts - depth_ts) > config.max_alignment_sec:
        return "alignment_gap"
    orientation = metadata.get("orientation") or {}
    if orientation.get("coordinate_space") != "external_uvc_unmirrored":
        return "unverified_orientation"
    bbox = metadata.get("bbox", ())
    if len(bbox) != 4 or any(finite(v) is None for v in bbox):
        return "invalid_bbox"
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= rgb.shape[1] and 0 <= y1 < y2 <= rgb.shape[0]):
        return "invalid_bbox"
    if metadata.get("source") != "rgb_aligned":
        return "not_rgb_aligned"
    return None


def person_hint(bbox):
    """Same box-derived crop hint as the inspected viewer; not measured joints."""
    x1, y1, x2, y2 = map(float, bbox)
    width, height = x2 - x1, y2 - y1
    cx, hip_y = (x1 + x2) * 0.5, y1 + height * 0.58
    return np.array([x1, y1, x2, y2, cx, hip_y,
                     cx, hip_y - max(width * .55, height * .58),
                     cx, y1 + height * .26, cx, y1, 1.0], dtype=np.float32)


class PoseEstimator:
    def __init__(self, config):
        spec = importlib.util.spec_from_file_location("follow_pose_shadow_adapter", config.adapter)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.MPPose(config.model, confThreshold=0.5)

    def infer(self, rgb, bbox):
        # The reference preprocessor adjusts its hint array in place.
        return self.model.infer(rgb, person_hint(bbox))


def depth_stats(depth, mask):
    area = int(np.count_nonzero(mask))
    values = depth[mask]
    valid = values[(values >= 350) & (values <= 8000)]
    count = int(valid.size)
    result = dict(area=area, valid=count, valid_fraction=count / max(1, area),
                  median_m=None, p10_m=None, p90_m=None)
    # Diagnostic statistic only. No temporal, jump or foreground acceptance.
    if count >= max(20, math.ceil(area * .10)):
        p10, median, p90 = np.percentile(valid, [10, 50, 90]) / 1000.0
        result.update(median_m=float(median), p10_m=float(p10), p90_m=float(p90))
    return result


def analyze_pose(rgb, depth, metadata, pose, config):
    import cv2

    result = dict(mode="shadow_only", target_id=metadata["target_id"],
                  capture_frame_id=metadata["rgb_capture_id"],
                  rgb_timestamp=metadata["rgb_timestamp"], depth_timestamp=metadata["depth_timestamp"],
                  alignment_ms=(metadata["depth_timestamp"] - metadata["rgb_timestamp"]) * 1000,
                  bbox=list(metadata["bbox"]), reason="no_pose", torso_polygon=None,
                  center_delta_px=None, pose_depth=None, landmarks=None,
                  runtime_candidate_m=metadata.get("candidate_m"),
                  runtime_accepted_raw_m=metadata.get("accepted_raw_m"),
                  runtime_filtered_or_held_m=metadata.get("filtered_or_held_m"),
                  runtime_detail=metadata.get("detail"),
                  identity_source=metadata.get("identity_source", "runtime_target_observation"))
    mask = np.zeros(depth.shape, dtype=np.uint8)
    for region in metadata.get("regions", []):
        bounds = region.get("roi", [])
        if len(bounds) == 4:
            x1, y1, x2, y2 = map(int, bounds)
            if 0 <= x1 < x2 <= depth.shape[1] and 0 <= y1 < y2 <= depth.shape[0]:
                mask[y1:y2, x1:x2] = 1
    result["bbox_region_depth"] = depth_stats(depth, mask.astype(bool))
    result["baseline_regions"] = metadata.get("regions", [])
    if pose is None:
        return result
    points = np.asarray(pose[1])
    if points.ndim != 2 or points.shape[0] < 25 or points.shape[1] < 5:
        result["reason"] = "invalid_landmarks"
        return result
    if np.isfinite(points[:33]).all():
        result["landmarks"] = points[:33].tolist()
    # Outline: left shoulder -> right shoulder -> right hip -> left hip.
    torso = points[[11, 12, 24, 23]]
    result["torso_confidence"] = float(np.min(torso[:, 3:5])) if np.isfinite(torso).all() else None
    if not np.isfinite(torso).all() or np.any(torso[:, 3:5] < config.landmark_confidence):
        result["reason"] = "low_landmark_confidence"
        return result
    polygon = torso[:, :2].astype(np.float32)
    x1, y1, x2, y2 = metadata["bbox"]
    if (np.any(polygon[:, 0] < x1) or np.any(polygon[:, 0] >= x2) or
            np.any(polygon[:, 1] < y1) or np.any(polygon[:, 1] >= y2)):
        result["reason"] = "torso_outside_target"
        return result
    if (not cv2.isContourConvex(polygon) or
            cv2.contourArea(polygon) < max(100, .04 * (x2-x1) * (y2-y1))):
        result["reason"] = "degenerate_torso"
        return result
    shoulder = (polygon[0] + polygon[1]) / 2
    hip = (polygon[2] + polygon[3]) / 2
    if np.linalg.norm(hip - shoulder) < 12:
        result["reason"] = "torso_too_short"
        return result
    if (np.linalg.norm(polygon[0]-polygon[1]) < .12 * (x2-x1) or
            np.linalg.norm(polygon[2]-polygon[3]) < .10 * (x2-x1)):
        result["reason"] = "collapsed_torso_width"
        return result
    center = polygon.mean(axis=0)
    # Shrink inward to avoid silhouette edges, arms and nearby background.
    polygon = center + .65 * (polygon - center)
    mask[:] = 0
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    result["torso_polygon"] = polygon.tolist()
    result["center_delta_px"] = float(center[0] - (x1 + x2) * .5)
    result["pose_depth"] = depth_stats(depth, mask.astype(bool))
    result["reason"] = "observed" if result["pose_depth"]["median_m"] is not None else "insufficient_pose_depth"
    return result


def render_comparison(rgb, result):
    import cv2

    image = rgb.copy()
    points = result.get("landmarks")
    if points is not None:
        points = np.asarray(points)
        visible = (points[:, 3] > .6) & (points[:, 4] > .6)
        visible &= ((points[:, 0] >= 0) & (points[:, 0] < image.shape[1]) &
                    (points[:, 1] >= 0) & (points[:, 1] < image.shape[0]))
        bones = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                 (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28))
        for a, b in bones:
            if b < len(points) and visible[a] and visible[b]:
                cv2.line(image, tuple(np.rint(points[a, :2]).astype(int)),
                         tuple(np.rint(points[b, :2]).astype(int)), (0, 220, 255), 2)
        for xy in points[visible, :2]:
            cv2.circle(image, tuple(np.rint(xy).astype(int)), 3, (0, 220, 255), -1)
    for r in result["baseline_regions"]:
        if len(r.get("roi", [])) == 4:
            x1, y1, x2, y2 = map(int, r["roi"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 160, 20), 1)
    if result["torso_polygon"] is not None:
        cv2.polylines(image, [np.rint(result["torso_polygon"]).astype(np.int32)], True, (30, 220, 30), 2)
    label = f"SHADOW ONLY cap={result['capture_frame_id']} {result['reason']}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 49), (0, 0, 0), -1)
    cv2.putText(image, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    old = result["bbox_region_depth"]["median_m"]
    new = (result["pose_depth"] or {}).get("median_m")
    cv2.putText(image, f"blue: bbox ROI median={old}  green: pose median={new}",
                (5, 40), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
    return image


class ObservationVideo:
    """Fixed-rate sampled inference video; source times remain visible and indexed."""
    def __init__(self, path, *, fps=2., label="INFERENCE SAMPLES", max_bytes=16*1024*1024):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.fps, self.label, self.max_bytes = max(.1, float(fps)), label, max_bytes
        self.writer = None
        self.first_stamp = None
        self.frames = 0

    def write(self, image, result):
        import cv2
        if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
            return None
        if self.writer is None:
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"),
                                           self.fps, (image.shape[1], image.shape[0]))
            if not self.writer.isOpened():
                raise RuntimeError("Cannot create observation video")
            self.first_stamp = result["rgb_timestamp"]
        frame = image.copy()
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, height-43), (width, height), (0, 0, 0), -1)
        label = f"{self.label} | source t={result['rgb_timestamp']-self.first_stamp:.2f}s"
        cv2.putText(frame, label, (5, height-25), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.putText(frame, "YELLOW=skeleton  BLUE=bbox regions  GREEN=pose region", (5, height-7),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        self.writer.write(frame)
        index = self.frames
        self.frames += 1
        return index

    def close(self):
        if self.writer is not None:
            self.writer.release()


def export_mp4(path):
    """Optional playback copy after acquisition; keep AVI if ffmpeg is unavailable."""
    path = Path(path)
    if not path.is_file():
        return None
    destination = path.with_suffix(".mp4")
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(path), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(destination)], check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        logging.getLogger(__name__).warning("MP4 export failed; original AVI retained: %s", exc)
        return None
    return destination


class PoseShadowObserver:
    """One queued job, rate/sample/disk budgets; worker failure is fail-open to control."""
    def __init__(self, output_dir, *, config=None, estimator_factory=PoseEstimator, clock=time.monotonic):
        self.config = config or PoseShadowConfig(
            adapter=os.environ.get("FOLLOW_POSE_ADAPTER", PoseShadowConfig.adapter),
            model=os.environ.get("FOLLOW_POSE_MODEL", PoseShadowConfig.model))
        self.output_dir = Path(output_dir)
        self.clock = clock
        self._factory = estimator_factory
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._last_submit = -math.inf
        self._last_key = None
        self._last_snapshot = -math.inf
        self.counts = Counter()
        self._bytes = 0
        self._video_budget = min(16*1024*1024, self.config.max_bytes // 4) if self.config.save_video else 0
        self._data_budget = self.config.max_bytes - self._video_budget
        self._thread = threading.Thread(target=self._run, name="pose-shadow", daemon=True)
        self._thread.start()

    def submit(self, rgb, depth, metadata):
        start = self.clock()
        if self._stop.is_set() or self.counts["submitted"] >= self.config.max_samples:
            return
        if start - self._last_submit < self.config.interval_sec:
            self.counts["rate_limited"] += 1
            return
        error = validate_packet(rgb, depth, metadata, self.config)
        if error:
            self.counts[error] += 1
            return
        age = start - metadata["rgb_timestamp"]
        if not 0 <= age <= self.config.max_input_age_sec:
            self.counts["stale_input"] += 1
            return
        key = (metadata["target_id"], metadata["rgb_capture_id"])
        if key == self._last_key:
            self.counts["duplicate"] += 1
            return
        if self._queue.full():
            self.counts["queue_full"] += 1
            return
        # The caller's diagnostics ring owns immutable arrays; take references.
        # Nested metadata is frozen by the producer via a fresh diagnostic dict.
        try:
            self._queue.put_nowait((rgb, depth, dict(metadata), start))
            self._last_submit, self._last_key = start, key
            self.counts["submitted"] += 1
        except queue.Full:
            self.counts["queue_full"] += 1

    def _run(self):
        video = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            manifest = dict(mode="shadow_only", config=vars(self.config),
                            model_sha256=hashlib.sha256(Path(self.config.model).read_bytes()).hexdigest(),
                            adapter_sha256=hashlib.sha256(Path(self.config.adapter).read_bytes()).hexdigest(),
                            numpy_version=np.__version__,
                            observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            estimator = self._factory(self.config)
            if self.config.save_video:
                video = ObservationVideo(self.output_dir / "inference.avi",
                                         fps=1/max(.05, self.config.interval_sec), max_bytes=self._video_budget)
            with (self.output_dir / "observations.jsonl").open("x", buffering=1) as stream:
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        rgb, depth, metadata, submitted = self._queue.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if self.clock() - metadata["rgb_timestamp"] > self.config.max_input_age_sec:
                        self.counts["stale_before_inference"] += 1
                        continue
                    started = self.clock()
                    pose = estimator.infer(rgb, metadata["bbox"])
                    inferred = self.clock()
                    result = analyze_pose(rgb, depth, metadata, pose, self.config)
                    result.update(inference_ms=(inferred-started)*1000,
                                  queue_ms=(started-submitted)*1000,
                                  result_age_ms=(self.clock()-metadata["rgb_timestamp"])*1000)
                    result["within_age_budget"] = result["result_age_ms"] <= self.config.max_input_age_sec * 1000
                    self.counts[result["reason"]] += 1
                    if not result["within_age_budget"]:
                        self.counts["late_result"] += 1
                    preview = render_comparison(rgb, result)
                    if video is not None:
                        result["video_frame_index"] = video.write(preview, result)
                    if (self.counts["snapshots"] < self.config.max_snapshots and
                            started-self._last_snapshot >= self.config.snapshot_interval_sec):
                        import cv2
                        name = f"sample_{self.counts['written']:04d}"
                        # Reserve for raw packet plus encoded preview and metadata.
                        estimate = rgb.nbytes + depth.nbytes + rgb.nbytes + 32768
                        if self._bytes + estimate < self._data_budget:
                            np.savez(self.output_dir / f"{name}.npz", rgb_bgr=rgb, depth_mm=depth,
                                     metadata_json=np.array(json.dumps(metadata)))
                            cv2.imwrite(str(self.output_dir / f"{name}.jpg"), preview)
                            result["snapshot"] = f"{name}.npz"
                            self._bytes += sum(p.stat().st_size for p in self.output_dir.glob(f"{name}.*"))
                            self._last_snapshot = started
                            self.counts["snapshots"] += 1
                    line = json.dumps(result, allow_nan=False) + "\n"
                    if self._bytes + len(line.encode()) >= self._data_budget:
                        self.counts["disk_budget"] += 1
                        self._stop.set()
                        continue
                    stream.write(line)
                    self._bytes += len(line.encode())
                    self.counts["written"] += 1
            (self.output_dir / "summary.json").write_text(json.dumps(dict(self.counts), indent=2))
        except Exception:
            logging.getLogger(__name__).exception("Pose shadow disabled; control is unaffected")
            self.counts["worker_error"] += 1
        finally:
            if video is not None:
                video.close()
            self._stop.set()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=.5)
