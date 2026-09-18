"""Passive online depth association experiment; NEVER a control data source.

The configured RGB HFOV is an APPROXIMATE registered-grid camera model, with
square pixels and a camera at the encoder axle. No calibrated extrinsics are
claimed. Every result is explicitly unverified and motor_authority_created=False.
All array work and I/O occur in the worker. Producers only replace immutable
scalar references; the worker reads existing feedback, never a serial device.
"""
from collections import Counter, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from .depth_track_shadow import CameraPose, DepthTrackShadow, Intrinsics


MODEL = "config_hfov_square_pixels_encoder_camera_at_axle_UNVERIFIED"


@dataclass(frozen=True)
class Anchor:
    generation: int
    observation: object
    width: int
    height: int


class EncoderPoseHistory:
    """Short bracket interpolation; no extrapolation or assumed zero motion.

    Encoder-only planar estimate, not an IMU pose or verified camera transform.
    A missing/bad sample or >120ms gap splits the history.
    """
    def __init__(self, circumference):
        if not math.isfinite(circumference) or circumference <= 0:
            raise ValueError("invalid wheel circumference")
        self.circumference = circumference
        self.samples = deque(maxlen=128)
        self.previous = None
        self.segment = 0

    def add(self, feedback, now):
        f = feedback
        values = (() if f is None else (f.timestamp, f.left_forward_rpm,
                  f.right_forward_rpm, f.integrated_yaw_right_deg, f.yaw_rate_right_dps))
        if (f is None or not f.trustworthy or not all(math.isfinite(v) for v in values)
                or not 0 <= now-f.timestamp <= .15
                or max(abs(f.left_forward_rpm), abs(f.right_forward_rpm)) > 200
                or abs(f.yaw_rate_right_dps) > 45
                or (f.raw_yaw_rate_right_dps is not None and
                    (not math.isfinite(f.raw_yaw_rate_right_dps) or abs(f.raw_yaw_rate_right_dps) > 45))):
            self.samples.clear()
            self.previous = None
            self.segment += 1
            return
        p = self.previous
        if p is not None and f.timestamp <= p.timestamp:
            return
        yaw = math.radians(f.integrated_yaw_right_deg)
        if p is None or f.timestamp-p.timestamp > .12:
            self.samples.clear()
            self.segment += 1
            x = z = 0.
        else:
            dt = f.timestamp-p.timestamp
            delta = yaw-self.samples[-1].yaw_rad
            if abs(delta) > math.radians(45)*dt+.01:
                self.samples.clear()
                self.previous = None
                self.segment += 1
                return
            rpm = (p.left_forward_rpm+p.right_forward_rpm+f.left_forward_rpm+f.right_forward_rpm)/4
            travel = rpm*self.circumference/60*dt
            middle = self.samples[-1].yaw_rad+delta/2
            x = self.samples[-1].x_m+travel*math.sin(middle)
            z = self.samples[-1].z_m+travel*math.cos(middle)
        self.samples.append(CameraPose(f.timestamp, x, z, yaw))
        self.previous = f

    def at(self, stamp):
        for p in self.samples:
            if abs(p.timestamp-stamp) < 1e-9:
                return p
        for a, b in zip(self.samples, list(self.samples)[1:]):
            if a.timestamp < stamp < b.timestamp and b.timestamp-a.timestamp <= .12:
                weight = (stamp-a.timestamp)/(b.timestamp-a.timestamp)
                return CameraPose(stamp, *(getattr(a, k)+weight*(getattr(b, k)-getattr(a, k))
                                          for k in ("x_m", "z_m", "yaw_rad")))
        return None


class AssociationSession:
    """Worker-only engine; RGB anchors and depth frames have independent clocks."""
    def __init__(self, hfov_deg, circumference):
        if not math.isfinite(hfov_deg) or not 20 <= hfov_deg <= 120:
            raise ValueError("invalid configured HFOV")
        self.hfov = hfov_deg
        self.poses = EncoderPoseHistory(circumference)
        self.frames = deque(maxlen=16)
        self.tracker = None
        self.anchor_key = None
        self.last_emitted = 0.
        self.last_success = None
        self.generation = None
        self.pose_segment = None

    def add_frames(self, frames):
        for stamp, depth in frames:
            if self.frames and stamp <= self.frames[-1][0]:
                continue
            if depth.dtype != np.uint16 or depth.ndim != 2 or depth.size > 640*480:
                continue
            # Work on a bounded private 160-ish grid, not the producer's array.
            stride = max(1, math.ceil(max(depth.shape[1]/160, depth.shape[0]/120)))
            self.frames.append((stamp, depth[::stride, ::stride].copy(), stride, depth.shape))

    def process(self, anchor, range_evidence, now):
        if anchor is None:
            self.tracker = None
            self.anchor_key = None
            return []
        o = anchor.observation
        if self.pose_segment != self.poses.segment:
            self.tracker = None
            self.anchor_key = None
            self.pose_segment = self.poses.segment
        if not 0 <= now-o.capture_timestamp <= .35:
            self.tracker = None
            return [self._event(anchor, now, "visual_lease_expired")]
        key = (anchor.generation, o.target_id, o.capture_frame_id, o.capture_timestamp)
        records = []
        if key != self.anchor_key:
            self.tracker = None
            if self.anchor_key is not None and key[0] == self.anchor_key[0] and key[3] <= self.anchor_key[3]:
                return []
            # Wait for a temporally compatible accepted RANGE, not another YOLO.
            # Hint is only seed selection; seed distance is measured afresh.
            if not self.frames or range_evidence is None:
                return [self._event(anchor, now, "seed_wait_range")]
            uid, range_ts, distance = range_evidence
            if uid != o.target_id or not math.isfinite(distance) or not .35 <= distance <= 8:
                return [self._event(anchor, now, "seed_range_identity_or_value")]
            seed = min(self.frames, key=lambda p: abs(p[0]-o.capture_timestamp))
            stamp, depth, stride, shape = seed
            if (abs(stamp-o.capture_timestamp) > .04 or abs(range_ts-stamp) > .18
                    or not 0 <= now-range_ts <= .35 or not 0 <= now-o.capture_timestamp <= .35):
                return [self._event(anchor, now, "seed_alignment_or_age")]
            pose = self.poses.at(stamp)
            if pose is None:
                return [self._event(anchor, now, "seed_pose_unbracketed")]
            if (anchor.height, anchor.width) != shape:
                return [self._event(anchor, now, "seed_grid_mismatch")]
            h, w = depth.shape
            focal = anchor.width/(2*math.tan(math.radians(self.hfov)/2))/stride
            tracker = DepthTrackShadow(Intrinsics(w, h, focal, focal,
                                                anchor.width/2/stride, anchor.height/2/stride))
            x1, y1, x2, y2 = o.bbox
            bw, bh = x2-x1, y2-y1
            roi = ((x1+.32*bw)/stride, (y1+.22*bh)/stride,
                   (x1+.68*bw)/stride, (y1+.68*bh)/stride)
            result = tracker.seed(depth, timestamp=stamp, pose=pose, uid=o.target_id,
                                  visual_timestamp=o.capture_timestamp, torso_bbox=roi,
                                  trusted_distance_m=distance, now=now, identity_confirmed=True)
            self.tracker, self.anchor_key = tracker, key
            if self.generation != anchor.generation:
                self.last_success = None
                self.generation = anchor.generation
            records.append(dict(self._event(anchor, now, result.status), sample_timestamp=stamp,
                                association_diagnostics=result.diagnostics,
                                hint_timestamp=range_ts, hint_distance_m=distance,
                                phase="seed", distance_m=None if result.camera_xyz is None else result.camera_xyz[2]))
        tracker = self.tracker
        if tracker is None or tracker.position is None:
            return records
        for stamp, depth, _, _ in self.frames:
            if stamp <= tracker.last_ts:
                continue
            pose = self.poses.at(stamp)
            if pose is None and 0 <= now-stamp <= .06:
                break  # Wait briefly for an existing encoder sample to bracket it.
            started = time.monotonic()
            # Historical catch-up updates state only. Its real age remains
            # explicit below; replay cannot count as an online observation.
            result = tracker.update(depth, timestamp=stamp, pose=pose, now=stamp,
                                    active_uid=o.target_id)
            elapsed = (time.monotonic()-started)*1000
            is_new = stamp > self.last_emitted
            live = is_new and 0 <= now-stamp <= .18 and 0 <= now-o.capture_timestamp <= .35
            baseline_ts = None if range_evidence is None or range_evidence[0] != o.target_id else range_evidence[1]
            row = dict(self._event(anchor, now, result.status), phase="update" if live else "replay",
                       association_diagnostics=result.diagnostics,
                       sample_timestamp=stamp, sample_age_ms=(now-stamp)*1000,
                       association_ms=elapsed, physical_sample_new=is_new,
                       candidate_count=result.candidate_count,
                       distance_m=None if result.camera_xyz is None else result.camera_xyz[2],
                       xyz=result.camera_xyz, baseline_sample_timestamp=baseline_ts,
                       baseline_distance_m=None if baseline_ts is None else range_evidence[2],
                       potential_range_gap_fill=bool(live and result.status == "tracked"
                           and (baseline_ts is None or now-baseline_ts > .18)),
                       tracked_interval_ms=None)
            row["depth_grid_size"] = [tracker.k.width, tracker.k.height]
            row["centroid_uv"] = (None if result.camera_xyz is None else [
                tracker.k.fx*result.camera_xyz[0]/result.camera_xyz[2]+tracker.k.cx,
                tracker.k.fy*result.camera_xyz[1]/result.camera_xyz[2]+tracker.k.cy])
            row["component_extent_px"] = tracker.extent if result.camera_xyz is not None else None
            if live and result.status == "tracked":
                row["tracked_interval_ms"] = None if self.last_success is None else (stamp-self.last_success)*1000
                self.last_success = stamp
            records.append(row)
            self.last_emitted = max(self.last_emitted, stamp)
            if result.status != "tracked":
                break
        return records

    @staticmethod
    def _event(anchor, now, status):
        return dict(status=status, phase="seed_wait", observed_at=now,
                    generation=anchor.generation, uid=anchor.observation.target_id,
                    capture_frame_id=anchor.observation.capture_frame_id,
                    anchor_bbox_rgb=list(anchor.observation.bbox),
                    rgb_size=[anchor.width, anchor.height],
                    visual_timestamp=anchor.observation.capture_timestamp,
                    visual_age_ms=(now-anchor.observation.capture_timestamp)*1000,
                    geometry_model=MODEL, geometry_verified=False,
                    control_allowed=False, motor_authority_created=False)


class DepthTrackOnlineObserver:
    """Bounded daemon, no API for reading results back into control."""
    def __init__(self, depth_runtime, feedback_reader, output_dir, logger, *, hfov_deg,
                 circumference, clock=time.monotonic, autostart=True):
        self.depth = depth_runtime
        self.feedback_reader = feedback_reader
        self.output = Path(output_dir)
        self.logger, self.clock = logger, clock
        self.session = AssociationSession(hfov_deg, circumference)
        self._state = (0, None)
        self._stop = threading.Event()
        self._thread = None
        self.counts = Counter()
        self._bytes = 0
        self._cursor = 0.
        self._last_wait = None
        self._last_wait_ts = -math.inf
        self._snapshot_count = 0
        self._snapshot_ts = -math.inf
        if autostart:
            self._thread = threading.Thread(target=self._run, name="depth-track-shadow", daemon=True)
            self._thread.start()

    def publish(self, observation, width, height):
        if self._stop.is_set():
            return
        generation, old = self._state
        if old is not None and observation.capture_timestamp <= old.observation.capture_timestamp:
            return
        if old is not None and observation.target_id != old.observation.target_id:
            generation += 1
        self._state = (generation, Anchor(generation, observation, int(width), int(height)))

    def revoke(self):
        self._state = (self._state[0]+1, None)

    def close(self):
        self.revoke()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=.5)

    def tick(self):
        now = self.clock()
        self.session.poses.add(self.feedback_reader(), now)
        frames = self.depth.copy_depth_history(after_timestamp=self._cursor, max_frames=4, nonblocking=True)
        if frames:
            self.counts["input_depth_frames"] += len(frames)
            if self._cursor and frames[0][0]-self._cursor > .05:
                self.counts["input_gap_over_50ms"] += 1
            self._cursor = frames[-1][0]
            self.session.add_frames(frames)
        state = self._state
        if state[1] is None:
            self.counts["input_frames_without_visual_anchor"] += len(frames)
        orientation = getattr(self.depth, "_depth_orientation", None)
        if orientation is None or orientation.coordinate_space != "external_uvc_unmirrored":
            self.counts["orientation_unavailable"] += 1
            self.session.tracker = None
            return []
        records = self.session.process(state[1], getattr(self.depth, "_shadow_range_evidence", None), now)
        elapsed = (self.clock()-now)*1000
        if state != self._state:
            self.session.tracker = None
            self.session.anchor_key = None
            self.counts["superseded_during_processing"] += 1
            return []
        output = []
        for row in records:
            if row["phase"] == "update":
                completed = self.clock()
                row["sample_age_ms"] = (completed-row["sample_timestamp"])*1000
                row["visual_age_ms"] = (completed-row["visual_timestamp"])*1000
                if (row["sample_age_ms"] > 180 or row["visual_age_ms"] > 350):
                    row["phase"] = "replay"
                    row["potential_range_gap_fill"] = False
                    row["expired_during_processing"] = True
            # Idle/seed-block reasons are rate limited; physical frame records aren't.
            if row["phase"] == "seed_wait":
                key = (row["generation"], row["status"])
                if key == self._last_wait and now-self._last_wait_ts < .5:
                    continue
                self._last_wait, self._last_wait_ts = key, now
            row["tick_processing_ms"] = elapsed
            self.counts[row["phase"]+":"+row["status"]] += 1
            if row.get("potential_range_gap_fill"):
                self.counts["potential_range_gap_fill"] += 1
            output.append(row)
        return output

    def _run(self):
        try:
            self.output.mkdir(parents=True, exist_ok=False)
            (self.output/"manifest.json").write_text(json.dumps(dict(
                mode="shadow_only", geometry_model=MODEL, hfov_deg=self.session.hfov,
                circumference_m=self.session.poses.circumference, visual_lease_ms=350,
                depth_fresh_ms=180, control_allowed=False, motor_authority_created=False,
                range_gap_is_not_motor_authority_gap=True), indent=2), encoding="utf-8")
            self.logger.info("depth_track_shadow started dir=%s model=%s control_allowed=False", self.output, MODEL)
            with (self.output/"observations.jsonl").open("x", encoding="utf-8") as stream:
                last_flush = last_summary = self.clock()
                while not self._stop.is_set() and self._bytes < 32*1024*1024:
                    start = self.clock()
                    for row in self.tick():
                        if (row["phase"] == "update" and self._snapshot_count < 12
                                and self.clock()-self._snapshot_ts >= 2.):
                            match = next((f for f in self.session.frames
                                          if f[0] == row.get("sample_timestamp")), None)
                            if match is not None:
                                filename = "sample_%03d.npz" % self._snapshot_count
                                np.savez_compressed(self.output/filename, depth_mm=match[1],
                                                    metadata=json.dumps(row, allow_nan=False))
                                row["snapshot_file"] = filename
                                self._snapshot_count += 1
                                self._snapshot_ts = self.clock()
                        line = json.dumps(row, allow_nan=False)+"\n"
                        stream.write(line)
                        self._bytes += len(line.encode())
                    elapsed = self.clock()-start
                    if elapsed > .015:
                        self.counts["tick_over_15ms"] += 1
                    if self.clock()-last_flush >= 1:
                        stream.flush()
                        last_flush = self.clock()
                    if self.clock()-last_summary >= 5:
                        self.logger.info("depth_track_shadow summary=%s control_allowed=False", dict(self.counts))
                        last_summary = self.clock()
                    # Cap work to ~half a core; no catch-up busy loop under overload.
                    self._stop.wait(max(.01, min(.5, elapsed)))
            (self.output/"summary.json").write_text(json.dumps(dict(
                counts=dict(self.counts), bytes_written=self._bytes,
                snapshots=self._snapshot_count,
                budget_exhausted=self._bytes >= 32*1024*1024,
                control_allowed=False, motor_authority_created=False), indent=2), encoding="utf-8")
        except Exception:
            self.logger.exception("depth_track_shadow disabled after worker failure; control unchanged")
        finally:
            self._stop.set()
