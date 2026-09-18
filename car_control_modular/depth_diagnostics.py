"""Bounded, passive depth evidence recorder. Never supplies control data.

Capture callbacks retain owned arrays in small rings; disk I/O runs only on
the daemon writer. Full queues/disk errors drop diagnostics, not control work.
"""
from collections import deque
import json
import math
from pathlib import Path
import queue
import threading

import numpy as np


class DepthDiagnostics:
    def __init__(self, output_dir, logger, *, max_events=24, max_bytes=256*1024*1024):
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.max_events = max_events
        self.max_bytes = max_bytes
        self._depth = deque(maxlen=8)
        self._rgb = deque(maxlen=8)
        # Scalar records survive slow encoding without retaining whole depth
        # arrays. At 30 observations/s, 512 records cover ~17s of video backlog.
        self._video_samples = deque(maxlen=512)
        self._lock = threading.Lock()
        self._jobs = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._events = 0
        self._last_event_ts = -math.inf
        self._post_event = None
        self._bytes = 0
        self.dropped = 0
        self.written = 0
        self.pose_shadow = None
        # Explicit opt-in observation only; model loading/inference stays in
        # the auxiliary worker and cannot supply data back to depth/control.
        try:
            from .pose_shadow import enabled, PoseShadowObserver
            if enabled():
                self.pose_shadow = PoseShadowObserver(self.output_dir / "pose_shadow")
        except Exception as exc:
            self.logger.warning("Pose shadow initialization skipped: %s", exc)
        self._thread = threading.Thread(target=self._write_loop, name="depth-diagnostics", daemon=True)
        self._thread.start()
        self.logger.info("Depth diagnostics enabled: dir=%s max_events=%d max_bytes=%d queue=8",
                         self.output_dir, max_events, max_bytes)

    def _offer(self, kind, payload):
        if self._stop.is_set():
            return
        try:
            self._jobs.put_nowait((kind, payload))
        except queue.Full:
            self.dropped += 1

    def add_rgb(self, capture_id, stamp, frame):
        if self._stop.is_set() or not math.isfinite(stamp) or stamp <= 0:
            return
        if frame is None or frame.ndim != 3 or frame.dtype != np.uint8:
            return
        # Display-only RGB: bounded size, no inference, JPEG, or resizing work
        # in the capture thread. Copy protects against later overlay drawing.
        stride = max(1, math.ceil(max(frame.shape[:2])/640))
        small = frame[::stride, ::stride, :3].copy()
        with self._lock:
            if self._rgb and stamp <= self._rgb[-1][1]:
                return
            self._rgb.append((int(capture_id), float(stamp), small))

    def add_depth(self, stamp, depth):
        if (self._stop.is_set() or not math.isfinite(stamp) or stamp <= 0
                or depth is None or depth.dtype != np.uint16 or depth.ndim != 2
                or depth.nbytes > 2*1024*1024):
            return
        # Astra owns a new copy for every physical sample and never mutates it.
        with self._lock:
            if self._depth and stamp <= self._depth[-1][0]:
                return
            self._depth.append((float(stamp), depth))
            post = self._post_event
            if post is not None and stamp >= post["next_ts"]:
                self._snapshot(post["event"], "after", stamp, depth, post["metadata"])
                post["remaining"] -= 1
                post["next_ts"] = stamp + .10
                if post["remaining"] <= 0:
                    self._post_event = None

    def _snapshot(self, event, phase, stamp, depth, metadata):
        rgb = min(self._rgb, key=lambda r: abs(r[1]-stamp)) if self._rgb else None
        if rgb is not None and abs(rgb[1]-stamp) > .20:
            rgb = None  # explicitly missing, never pair an unrelated old RGB
        info = dict(metadata, event=event, phase=phase, depth_timestamp=stamp,
                    depth_unit="mm", rgb_format="BGR_stride_preview",
                    rgb_capture_id=None if rgb is None else rgb[0],
                    rgb_timestamp=None if rgb is None else rgb[1],
                    rgb_alignment_ms=None if rgb is None else (rgb[1]-stamp)*1000)
        self._offer("snapshot", (depth, None if rgb is None else rgb[2], info))

    def observe(self, metadata, *, anomaly, now, sample_stamp):
        if self._stop.is_set():
            return
        with self._lock:
            if metadata.get("regions") and metadata.get("evidence_capture_frame_id") is not None:
                matched = next((d for d in self._depth
                                if sample_stamp is not None and abs(d[0]-sample_stamp) < 1e-6), None)
                size = metadata.get("depth_size")
                if size is None and matched is not None:
                    size = [int(matched[1].shape[1]), int(matched[1].shape[0])]
                if size is not None and sample_stamp is not None:
                    from .video_depth_overlay import DepthVideoSample
                    # Region medians were computed by the existing sampler.
                    # No fresh statistics or pixel work in the control path.
                    self._video_samples.append(DepthVideoSample(None, dict(metadata, depth_size=size)))
            if self.pose_shadow is not None:
                try:
                    # Exact provenance: a later RGB frame or depth snapshot
                    # must never be combined with the previous target bbox.
                    matched_rgb = next((r for r in self._rgb
                                        if r[0] == metadata.get("evidence_capture_frame_id")), None)
                    matched_depth = next((d for d in self._depth
                                          if sample_stamp is not None and abs(d[0]-sample_stamp) < 1e-6), None)
                    self.pose_shadow.submit(
                        None if matched_rgb is None else matched_rgb[2],
                        None if matched_depth is None else matched_depth[1],
                        dict(metadata,
                             rgb_capture_id=None if matched_rgb is None else matched_rgb[0],
                             rgb_timestamp=None if matched_rgb is None else matched_rgb[1],
                             depth_timestamp=None if matched_depth is None else matched_depth[0]),
                    )
                except Exception as exc:
                    self.logger.warning("Pose shadow observation skipped: %s", exc)
            reference = metadata.get("reference_timestamp") or sample_stamp
            rgb = (min(self._rgb, key=lambda r: abs(r[1]-reference))
                   if self._rgb and reference is not None else None)
            if rgb is not None and abs(rgb[1]-reference) > .20:
                rgb = None
            metadata = dict(metadata, rgb_reference_capture_id=None if rgb is None else rgb[0],
                            rgb_reference_timestamp=None if rgb is None else rgb[1],
                            diagnostics_dropped=self.dropped)
            self._offer("observation", metadata)
            if (not anomaly or self._events >= self.max_events
                    or now-self._last_event_ts < 2.0 or not self._depth):
                return
            self._events += 1
            self._last_event_ts = now
            # Include the sampled frame (which may be historical), not merely
            # the latest frame. Keep at most two preceding physical samples.
            selected = [p for p in self._depth if sample_stamp is not None and p[0] <= sample_stamp+1e-9][-3:]
            for stamp, depth in selected:
                self._snapshot(self._events, "before_or_sample", stamp, depth, metadata)
            self._post_event = dict(event=self._events, metadata=metadata,
                                    remaining=2, next_ts=now+.10)

    def video_sample(self, capture_id, capture_timestamp):
        """Recorder-only, try-lock lookup. Never wait for the sensor/control."""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            samples = tuple(self._video_samples)
        finally:
            self._lock.release()
        # Latest Depth often uses an older visual ROI. Display the physical
        # sample on its time-matched RGB, with the ROI's source CAP labelled.
        candidates = [s for s in samples
                      if abs(s.metadata["sample_timestamp"]-capture_timestamp) <= .080000001]
        if not candidates:
            # Keep an explicit UNALIGNED report for a measured visual frame.
            candidates = [s for s in samples
                          if s.metadata.get("evidence_capture_frame_id") == capture_id]
        return min(candidates, key=lambda s: abs(s.metadata["sample_timestamp"]-capture_timestamp),
                   default=None)

    def _write_loop(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Exclusive creation: even a reused run directory cannot overwrite
            # a previous diagnostic session.
            with (self.output_dir/"observations.jsonl").open("x", encoding="utf-8") as stream:
                while not self._stop.is_set() or not self._jobs.empty():
                    try:
                        kind, payload = self._jobs.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if self._bytes >= self.max_bytes:
                        self.dropped += 1
                        continue
                    if kind == "snapshot":
                        depth, rgb, info = payload
                        estimate = depth.nbytes + (0 if rgb is None else rgb.nbytes) + 16384
                        if self._bytes+estimate > self.max_bytes:
                            self.dropped += 1
                            continue
                        name = "event_%02d_depth_%.6f.npz" % (info["event"], info["depth_timestamp"])
                        arrays = {"depth_mm": depth, "metadata_json": np.array(json.dumps(info))}
                        if rgb is not None:
                            arrays["rgb_bgr"] = rgb
                        with (self.output_dir/name).open("xb") as destination:
                            np.savez(destination, **arrays)
                        self._bytes += (self.output_dir/name).stat().st_size
                        payload = dict(info, snapshot_file=name)
                        self.written += 1
                    line = json.dumps(payload, ensure_ascii=False, allow_nan=False)+"\n"
                    size = len(line.encode("utf-8"))
                    if self._bytes+size <= self.max_bytes:
                        stream.write(line)
                        stream.flush()
                        self._bytes += size
        except Exception as exc:
            self._stop.set()
            self.logger.warning("Depth diagnostics disabled after writer error: %s", exc)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=.5)
        if self.pose_shadow is not None:
            self.pose_shadow.close()
        self.logger.info("Depth diagnostics closed: snapshots=%d dropped=%d bytes=%d writer_alive=%s",
                         self.written, self.dropped, self._bytes, self._thread.is_alive())
