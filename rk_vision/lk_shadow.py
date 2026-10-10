"""Bounded sparse optical-flow diagnostics; never identity or control evidence.

Coordinates are in the supplied raw image. Capture timestamps must share one
monotonic clock; processing never substitutes wall-clock time for capture time.
A correction belongs to its *own* captured image, never to a later image.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class LKShadowConfig:
    width: int = 320
    max_points: int = 64
    min_points: int = 8
    max_gap_sec: float = 0.25
    max_seed_age_sec: float = 0.75
    max_fb_error_px: float = 1.5
    max_lk_error: float = 35.0
    min_inlier_ratio: float = 0.50
    max_step_ratio: float = 0.20
    min_scale: float = 0.85
    max_scale: float = 1.18
    max_rotation_deg: float = 12.0
    max_outside_ratio: float = 0.15
    interior_margin: float = 0.12

    def __post_init__(self):
        if not 64 <= self.width <= 640:
            raise ValueError('LK width must be between 64 and 640')
        if not 4 <= self.min_points <= self.max_points <= 80:
            raise ValueError('LK point limits must satisfy 4 <= min <= max <= 80')
        numeric = (self.max_gap_sec, self.max_seed_age_sec, self.max_fb_error_px,
                   self.max_lk_error, self.max_step_ratio, self.min_scale,
                   self.max_scale, self.max_rotation_deg)
        if any(not math.isfinite(v) or v <= 0 for v in numeric):
            raise ValueError('LK limits must be finite and positive')
        if not (0 < self.min_inlier_ratio <= 1 and 0 <= self.interior_margin < .4
                and 0 <= self.max_outside_ratio < .5
                and self.min_scale <= 1 <= self.max_scale):
            raise ValueError('invalid LK geometry limits')


@dataclass(frozen=True)
class LKShadowSeed:
    uid: int
    raw_track_id: int
    capture_id: int
    capture_timestamp: float
    bbox: BBox

    def __post_init__(self):
        object.__setattr__(self, 'bbox', tuple(float(v) for v in self.bbox))


@dataclass(frozen=True)
class LKShadowResult:
    capture_id: int
    capture_timestamp: float
    status: str
    reason: str
    source: str = 'lk_shadow'
    seed_uid: Optional[int] = None
    seed_raw_track_id: Optional[int] = None
    seed_capture_id: Optional[int] = None
    seed_capture_timestamp: Optional[float] = None
    prev_capture_id: Optional[int] = None
    prev_capture_timestamp: Optional[float] = None
    bbox: Optional[BBox] = None
    input_points: int = 0
    tracked_points: int = 0
    inlier_points: int = 0
    fb_error_px: Optional[float] = None
    quality: float = 0.0
    wall_ms: float = 0.0
    thread_cpu_ms: float = 0.0


class LKShadowTracker:
    """Single-threaded state machine used by the worker and offline replay."""

    def __init__(self, config: Optional[LKShadowConfig] = None):
        self.config = config or LKShadowConfig()
        self._last_id = None
        self._last_ts = None
        self._clear()

    def _clear(self):
        self._gray = None
        self._points = None
        self._bbox = None
        self._seed = None
        self._shape = None
        self._scales = None

    def _result(self, cap, ts, status, reason, **kwargs):
        seed = kwargs.pop('seed', self._seed)
        return LKShadowResult(
            capture_id=cap, capture_timestamp=ts, status=status, reason=reason,
            seed_uid=seed.uid if seed else None,
            seed_raw_track_id=seed.raw_track_id if seed else None,
            seed_capture_id=seed.capture_id if seed else None,
            seed_capture_timestamp=seed.capture_timestamp if seed else None,
            prev_capture_id=self._last_id, prev_capture_timestamp=self._last_ts,
            **kwargs)

    def process(self, frame, capture_id: int, capture_timestamp: float,
                seed: Optional[LKShadowSeed] = None, frame_format: str = 'RGB'):
        """Consume one actual image; all image/flow failures become diagnostics.

        Duplicate/out-of-order input leaves the current state untouched. Other
        failures invalidate flow until a new aligned YOLO correction arrives.
        """
        start = time.perf_counter()
        cpu_start = time.thread_time()
        try:
            cap, ts = int(capture_id), float(capture_timestamp)
        except (TypeError, ValueError, OverflowError):
            return LKShadowResult(-1, 0., 'invalid_frame', 'invalid_capture_provenance',
                                  wall_ms=(time.perf_counter() - start) * 1000.,
                                  thread_cpu_ms=(time.thread_time() - cpu_start) * 1000.)
        try:
            result = self._process(frame, cap, ts, seed, frame_format)
        except Exception as exc:
            result = self._result(cap, ts, 'error', type(exc).__name__)
            self._clear()
            self._last_id, self._last_ts = cap, ts
        return replace(result, wall_ms=(time.perf_counter() - start) * 1000.,
                       thread_cpu_ms=(time.thread_time() - cpu_start) * 1000.)

    def _fail(self, cap, ts, status, reason, **kwargs):
        result = self._result(cap, ts, status, reason, **kwargs)
        self._clear()
        self._last_id, self._last_ts = cap, ts
        return result

    def _process(self, frame, cap, ts, seed, frame_format):
        if cap < 0 or not math.isfinite(ts):
            return self._result(cap, ts, 'invalid_frame', 'invalid_capture_provenance')
        if self._last_id is not None and (cap <= self._last_id or ts <= self._last_ts):
            return self._result(cap, ts, 'out_of_order', 'capture_must_increase')
        if seed is not None and (seed.capture_id != cap
                or not math.isfinite(seed.capture_timestamp)
                or abs(seed.capture_timestamp - ts) > 1e-6):
            return self._fail(cap, ts, 'seed_mismatch', 'seed_image_not_aligned', seed=seed)
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.ndim not in (2, 3) or min(arr.shape[:2]) < 16:
            return self._fail(cap, ts, 'invalid_frame', 'expected_uint8_raw_image')
        if arr.ndim == 3:
            if arr.shape[2] != 3 or frame_format.upper() not in ('RGB', 'BGR'):
                return self._fail(cap, ts, 'invalid_frame', 'unsupported_color_format')
            code = cv2.COLOR_RGB2GRAY if frame_format.upper() == 'RGB' else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(arr, code)
        else:
            gray = arr
        h, w = gray.shape
        small_w = min(self.config.width, w)
        small_h = max(16, round(h * small_w / w))
        gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        scales = np.array([small_w / w, small_h / h], dtype=np.float64)

        if seed is not None:
            if len(seed.bbox) != 4 or not self._valid_bbox(seed.bbox, w, h, 0):
                return self._fail(cap, ts, 'invalid_seed', 'seed_bbox_invalid', seed=seed)
            box = np.asarray(seed.bbox).reshape(2, 2) * scales
            mask = np.zeros(gray.shape, dtype=np.uint8)
            bw, bh = box[1] - box[0]
            margin = np.array([bw, bh]) * self.config.interior_margin
            x1, y1 = np.ceil(box[0] + margin).astype(int)
            x2, y2 = np.floor(box[1] - margin).astype(int)
            mask[max(0, y1):min(small_h, y2), max(0, x1):min(small_w, x2)] = 255
            points = cv2.goodFeaturesToTrack(gray, self.config.max_points,
                                            .01, 4., mask=mask, blockSize=5)
            count = 0 if points is None else len(points)
            if count < self.config.min_points:
                return self._fail(cap, ts, 'lost', 'seed_insufficient_texture',
                                  seed=seed, input_points=count)
            result = self._result(cap, ts, 'seeded', 'aligned_detector_correction',
                                  seed=seed, bbox=tuple(seed.bbox), input_points=count,
                                  tracked_points=count, inlier_points=count, quality=1.)
            self._gray, self._points, self._bbox = gray, points, np.asarray(seed.bbox, dtype=float)
            self._seed, self._shape, self._scales = seed, (h, w), scales
            self._last_id, self._last_ts = cap, ts
            return result

        if self._seed is None:
            return self._fail(cap, ts, 'no_seed', 'awaiting_aligned_detector_correction')
        if ts - self._last_ts > self.config.max_gap_sec:
            return self._fail(cap, ts, 'stale_gap', 'capture_gap_exceeded')
        if ts - self._seed.capture_timestamp > self.config.max_seed_age_sec:
            return self._fail(cap, ts, 'seed_expired', 'detector_correction_age_exceeded')
        if (h, w) != self._shape:
            return self._fail(cap, ts, 'invalid_frame', 'image_dimensions_changed')
        points = self._points
        opts = dict(winSize=(21, 21), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, .03))
        moved, forward, errors = cv2.calcOpticalFlowPyrLK(self._gray, gray, points, None, **opts)
        if moved is None or forward is None:
            return self._fail(cap, ts, 'lost', 'forward_flow_failed', input_points=len(points))
        returned, backward, _ = cv2.calcOpticalFlowPyrLK(gray, self._gray, moved, None, **opts)
        if returned is None or backward is None:
            return self._fail(cap, ts, 'lost', 'backward_flow_failed', input_points=len(points))
        fb = np.linalg.norm(points.reshape(-1, 2) - returned.reshape(-1, 2), axis=1)
        target = moved.reshape(-1, 2)
        keep = (forward.ravel().astype(bool) & backward.ravel().astype(bool)
                & np.isfinite(target).all(axis=1) & np.isfinite(fb)
                & (fb <= self.config.max_fb_error_px)
                & (errors.ravel() <= self.config.max_lk_error)
                & (target[:, 0] >= 0) & (target[:, 0] < small_w)
                & (target[:, 1] >= 0) & (target[:, 1] < small_h))
        before, after = points.reshape(-1, 2)[keep], target[keep]
        count = len(after)
        metrics = dict(input_points=len(points), tracked_points=count,
                       fb_error_px=float(np.median(fb[keep])) if count else None)
        if count < self.config.min_points:
            return self._fail(cap, ts, 'lost', 'insufficient_forward_backward_points', **metrics)
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            before, after, method=cv2.RANSAC, ransacReprojThreshold=2.,
            maxIters=100, confidence=.95, refineIters=5)
        if transform is None or inlier_mask is None or not np.isfinite(transform).all():
            return self._fail(cap, ts, 'lost', 'robust_transform_failed', **metrics)
        inliers = inlier_mask.ravel().astype(bool)
        n = int(inliers.sum())
        metrics['inlier_points'] = n
        if n < self.config.min_points or n / len(points) < self.config.min_inlier_ratio:
            return self._fail(cap, ts, 'lost', 'insufficient_inlier_support', **metrics)
        scale = math.hypot(transform[0, 0], transform[1, 0])
        rotation = abs(math.degrees(math.atan2(transform[1, 0], transform[0, 0])))
        delta = np.median(after[inliers] - before[inliers], axis=0) / scales
        if (not self.config.min_scale <= scale <= self.config.max_scale
                or rotation > self.config.max_rotation_deg
                or np.linalg.norm(delta) > min(w, h) * self.config.max_step_ratio):
            return self._fail(cap, ts, 'rejected_geometry', 'scale_rotation_or_jump', **metrics)
        x1, y1, x2, y2 = self._bbox
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) * scales
        warped = (corners @ transform[:, :2].T + transform[:, 2]) / scales
        box = np.r_[warped.min(axis=0), warped.max(axis=0)]
        if not self._valid_bbox(box, w, h, self.config.max_outside_ratio):
            return self._fail(cap, ts, 'rejected_geometry', 'bbox_outside_image', **metrics)
        # Clip only modest edge crossings. Reject a mostly departed object.
        box[[0, 2]] = np.clip(box[[0, 2]], 0., w)
        box[[1, 3]] = np.clip(box[[1, 3]], 0., h)
        quality = (n / len(points)) * max(0., 1. - metrics['fb_error_px'] / self.config.max_fb_error_px)
        result = self._result(cap, ts, 'tracked', 'forward_backward_ransac',
                              bbox=tuple(float(v) for v in box), quality=float(quality), **metrics)
        self._gray, self._points, self._bbox = gray, after[inliers].reshape(-1, 1, 2), box
        self._last_id, self._last_ts = cap, ts
        return result

    @staticmethod
    def _valid_bbox(box, w, h, allowed_outside):
        if not np.isfinite(box).all():
            return False
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return False
        visible = max(0., min(x2, w) - max(x1, 0.)) * max(0., min(y2, h) - max(y1, 0.))
        return visible / area >= 1. - allowed_outside


@dataclass(frozen=True)
class _Task:
    frame: np.ndarray
    capture_id: int
    capture_timestamp: float
    seed: Optional[LKShadowSeed]
    frame_format: str


class LKShadowWorker:
    """One worker, one replaceable pending image, one immutable latest result.

    Submission copies its buffer. There is no frame history and no unbounded
    result queue. An aligned pending seed takes priority over ordinary frames;
    newer seed tasks replace it together with their own image. A seed is never
    applied to a replacement image. Historical corrections are not replayed.
    """

    def __init__(self, config: Optional[LKShadowConfig] = None, *, tracker=None):
        self._tracker = tracker or LKShadowTracker(config)
        self._condition = threading.Condition()
        self._pending = None
        self._latest = None
        self._closed = False
        self._counts = dict(submitted=0, processed=0, replaced=0, rejected=0,
                            seed_priority_dropped=0, errors=0,
                            flow_attempts=0, flow_successes=0,
                            process_wall_ms_total=0., process_wall_ms_max=0.,
                            process_thread_cpu_ms_total=0., process_thread_cpu_ms_max=0.)
        # Fixed-size aggregates remain accurate even if the consumer skips
        # several immutable latest results. No per-frame history is retained.
        self._statuses = {status: 0 for status in (
            'seeded', 'tracked', 'lost', 'no_seed', 'out_of_order', 'seed_mismatch',
            'invalid_frame', 'invalid_seed', 'stale_gap', 'seed_expired',
            'rejected_geometry', 'error', 'other')}
        self._submitted_id = None
        self._submitted_ts = None
        self._thread = threading.Thread(target=self._run, name='lk-shadow', daemon=True)
        self._thread.start()

    def submit(self, frame, capture_id: int, capture_timestamp: float,
               seed: Optional[LKShadowSeed] = None, frame_format: str = 'RGB') -> bool:
        try:
            cap, ts = int(capture_id), float(capture_timestamp)
            if cap < 0 or not math.isfinite(ts):
                raise ValueError('invalid capture')
            arr = np.asarray(frame)
            if arr.dtype != np.uint8 or arr.ndim not in (2, 3):
                raise ValueError('invalid image')
            task = _Task(arr.copy(), cap, ts, seed, frame_format)
        except Exception:
            with self._condition:
                self._counts['rejected'] += 1
            return False
        with self._condition:
            if self._closed or (self._submitted_id is not None and
                    (cap <= self._submitted_id or ts <= self._submitted_ts)):
                self._counts['rejected'] += 1
                return False
            if self._pending is not None and self._pending.seed is not None and seed is None:
                self._counts['seed_priority_dropped'] += 1
                self._counts['rejected'] += 1
                return False
            self._counts['submitted'] += 1
            self._counts['replaced'] += int(self._pending is not None)
            self._pending = task
            self._submitted_id, self._submitted_ts = cap, ts
            self._condition.notify()
            return True

    def latest_result(self) -> Optional[LKShadowResult]:
        with self._condition:
            return self._latest

    def stats(self):
        with self._condition:
            return dict(self._counts, pending=int(self._pending is not None),
                        closed=self._closed, alive=self._thread.is_alive(),
                        status_counts=dict(self._statuses))

    def close(self, timeout: float = 2.) -> bool:
        """Drop pending work and join; return whether the worker has stopped."""
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=max(0., timeout))
        return not self._thread.is_alive()

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                task, self._pending = self._pending, None
            start = time.perf_counter()
            cpu_start = time.thread_time()
            try:
                result = self._tracker.process(task.frame, task.capture_id,
                    task.capture_timestamp, seed=task.seed, frame_format=task.frame_format)
            except Exception as exc:
                result = LKShadowResult(task.capture_id, task.capture_timestamp,
                    'error', type(exc).__name__, wall_ms=(time.perf_counter() - start) * 1000.,
                    thread_cpu_ms=(time.thread_time() - cpu_start) * 1000.)
            with self._condition:
                self._counts['processed'] += 1
                self._counts['errors'] += int(result.status == 'error')
                status = result.status if result.status in self._statuses else 'other'
                self._statuses[status] += 1
                is_flow = (task.seed is None and result.seed_capture_id is not None
                           and result.status in ('tracked', 'lost', 'rejected_geometry'))
                self._counts['flow_attempts'] += int(is_flow)
                self._counts['flow_successes'] += int(is_flow and result.status == 'tracked')
                for field, value in (('process_wall_ms', result.wall_ms),
                                     ('process_thread_cpu_ms', result.thread_cpu_ms)):
                    self._counts[field + '_total'] += value
                    self._counts[field + '_max'] = max(self._counts[field + '_max'], value)
                if not self._closed:
                    self._latest = result
