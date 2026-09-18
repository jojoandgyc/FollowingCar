"""Offline/shadow-only depth association. No UID creation or motor authority.

Inputs MUST be registered, orientation-corrected, rectified uint16 depth in mm.
Pose is camera pose at the physical frame time (X right, Z forward, yaw right).
This conservative prototype uses constant-velocity prediction, not person ReID.
The online passive experiment may supply an explicitly UNVERIFIED approximate
camera/pose model; its outputs cannot establish calibrated tracking accuracy.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if (self.width <= 0 or self.height <= 0
                or not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy))
                or min(self.fx, self.fy) <= 0
                or not 0 <= self.cx < self.width or not 0 <= self.cy < self.height):
            raise ValueError("Explicit finite depth-grid intrinsics required")


@dataclass(frozen=True)
class CameraPose:
    timestamp: float
    x_m: float
    z_m: float
    yaw_rad: float

    def valid_at(self, timestamp):
        return (all(math.isfinite(v) for v in
                    (self.timestamp, self.x_m, self.z_m, self.yaw_rad))
                and abs(self.timestamp - timestamp) <= .02)

    def world(self, camera):
        x, y, z = camera
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return np.array([c*x+s*z+self.x_m, y, -s*x+c*z+self.z_m])

    def camera(self, world):
        x, y, z = world - np.array([self.x_m, 0, self.z_m])
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return np.array([c*x-s*z, y, s*x+c*z])


@dataclass(frozen=True)
class Observation:
    uid: int | None
    timestamp: float
    status: str
    camera_xyz: tuple | None = None
    world_xyz: tuple | None = None
    visual_timestamp: float | None = None
    candidate_count: int = 0
    # This is intentionally never usable as a motor/identity decision.
    control_allowed: bool = False
    diagnostics: dict | None = None


class DepthTrackShadow:
    """Single episode. Any association failure requires a new visual seed.

    Fixed prototype budgets: 350ms visual lease, 180ms physical depth age/gap,
    20ms pose alignment, 40ms RGB/depth seed alignment. No confidence scalar
    that callers could mistake for an identity probability.
    """
    def __init__(self, intrinsics: Intrinsics):
        self.k = intrinsics
        self.reset()

    def reset(self):
        self.uid = None
        self.visual_ts = None
        self.last_ts = None
        self.position = None
        self.velocity = np.zeros(3)
        self.area = None
        self.extent = None
        self.last_z = None
        self._diagnostics = {}
        self._core_fraction = None
        self._seed_area = None
        self._seed_z = None

    def _result(self, timestamp, status, *, candidate=None, count=0):
        return Observation(self.uid, timestamp, status,
                           None if candidate is None else tuple(float(v) for v in candidate[0]),
                           None if candidate is None else tuple(float(v) for v in candidate[1]),
                           self.visual_ts, count, False, dict(self._diagnostics))

    def _input_error(self, depth, stamp, pose):
        if not math.isfinite(stamp) or stamp <= 0:
            return "invalid_timestamp"
        if not isinstance(pose, CameraPose) or not pose.valid_at(stamp):
            return "pose_missing_or_stale"
        if (not isinstance(depth, np.ndarray) or depth.dtype != np.uint16
                or depth.shape != (self.k.height, self.k.width)):
            return "invalid_depth_grid"
        return None

    def _candidates(self, depth, roi, z_hint, pose, minimum):
        """ROI selects components; it must NOT crop their measured geometry.

        Label on the same complete depth grid for both seed and update. Only
        components with enough support inside the selector ROI are eligible.
        Their area/bbox come from the complete connected component. Range and
        centroid come from the same seed-relative torso core inside that
        component, not newly included legs/background. Expanding the search
        cannot make a static torso 'grow' or move its centroid. Online
        uses a <=160x120 grid; statistics are computed only in label bboxes.
        """
        x1, y1, x2, y2 = roi
        patch = depth.astype(np.float64) * .001
        mask = ((patch >= .35) & (patch <= 8)
                & (np.abs(patch-z_hint) <= .25)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        selected, overlaps = np.unique(labels[y1:y2, x1:x2], return_counts=True)
        eligible = [(int(label), int(overlap)) for label, overlap in zip(selected, overlaps)
                    if label != 0 and overlap >= minimum]
        scan = dict(component_stats_space="full_depth_grid", selector_roi=list(roi),
                    minimum_roi_support_px=minimum, eligible_components=len(eligible),
                    candidates=[], scan_rejection=None)
        self._diagnostics.update(scan)
        if len(eligible) > 8:
            self._diagnostics["scan_rejection"] = "component_budget_exceeded"
            return []
        candidates = []
        for label, overlap in eligible:
            bx, by, bw, bh, area = map(int, stats[label])
            audit = dict(component_area_px=area, roi_support_px=overlap,
                         component_bbox=[bx, by, bx+bw, by+bh],
                         measurement_roi=None, measurement_support_px=0,
                         depth_spread_m=None, reject_reasons=[])
            self._diagnostics["candidates"].append(audit)
            if self._core_fraction is None:
                # An entire floor/wall at the same depth can be connected to
                # a valid torso. Do not initialize a frame-wide footprint from
                # a small body ROI merely because its central depth is good.
                audit.update(width_to_seed_roi=bw/(x2-x1), height_to_seed_roi=bh/(y2-y1))
                if bw > 4*(x2-x1) or bh > 3*(y2-y1):
                    audit["reject_reasons"].append("seed_background_extent")
                    continue
                core = (max(bx, x1), max(by, y1), min(bx+bw, x2), min(by+bh, y2))
            else:
                l, t, r, b = self._core_fraction
                core = (bx+round(l*bw), by+round(t*bh), bx+round(r*bw), by+round(b*bh))
            cx1, cy1, cx2, cy2 = core
            yy, xx = np.nonzero(labels[cy1:cy2, cx1:cx2] == label)
            yy, xx = yy+cy1, xx+cx1
            zz = patch[yy, xx]
            audit.update(measurement_roi=list(core), measurement_support_px=len(xx))
            if len(xx) < max(20, math.ceil((cx2-cx1)*(cy2-cy1)*.10)):
                audit["reject_reasons"].append("core_support")
                continue
            p10, z, p90 = np.percentile(zz, [10, 50, 90])
            audit["depth_spread_m"] = float(p90-p10)
            # Reject a depth ramp connecting foreground to a wall.
            if p90-p10 > .25:
                audit["reject_reasons"].append("depth_spread")
                continue
            camera = np.array([np.median((xx-self.k.cx)*zz/self.k.fx),
                               np.median((yy-self.k.cy)*zz/self.k.fy), z])
            candidates.append((camera, pose.world(camera), area, tuple(audit["component_bbox"])))
        return candidates

    def seed(self, depth, *, timestamp, pose, uid, visual_timestamp,
             torso_bbox, trusted_distance_m, now, identity_confirmed):
        """Explicit trusted visual anchor; bbox is a depth-grid torso ROI.

        Seed historical aligned depth then replay intervening depth in order.
        A trusted distance is REQUIRED; never select the nearest depth surface.
        Rejected new seed invalidates the previous episode.
        """
        self.reset()
        error = self._input_error(depth, timestamp, pose)
        if error:
            return self._result(timestamp, error)
        if (identity_confirmed is not True or isinstance(uid, bool)
                or not isinstance(uid, int) or uid <= 0):
            return self._result(timestamp, "unconfirmed_identity")
        if (not all(math.isfinite(v) for v in (visual_timestamp, now, trusted_distance_m))
                or visual_timestamp <= 0 or not .35 <= trusted_distance_m <= 8
                or abs(timestamp-visual_timestamp) > .04
                or not 0 <= now-visual_timestamp <= .35
                or not 0 <= now-timestamp <= .35):
            return self._result(timestamp, "invalid_visual_anchor")
        if (len(torso_bbox) != 4 or not all(math.isfinite(v) for v in torso_bbox)):
            return self._result(timestamp, "invalid_torso_roi")
        x1, y1, x2, y2 = torso_bbox
        if not (0 <= x1 < x2 <= self.k.width and 0 <= y1 < y2 <= self.k.height):
            return self._result(timestamp, "invalid_torso_roi")
        roi = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
        minimum = max(20, int((x2-x1)*(y2-y1)*.1))
        candidates = self._candidates(depth, roi, trusted_distance_m, pose, minimum)
        if len(candidates) != 1:
            return self._result(timestamp, "seed_ambiguous_or_empty", count=len(candidates))
        self.uid, self.visual_ts = uid, visual_timestamp
        selected = next(a for a in self._diagnostics["candidates"] if not a["reject_reasons"])
        bx, by, br, bb = selected["component_bbox"]
        l, t, r, b = selected["measurement_roi"]
        self._core_fraction = ((l-bx)/(br-bx), (t-by)/(bb-by),
                               (r-bx)/(br-bx), (b-by)/(bb-by))
        self._seed_area, self._seed_z = candidates[0][2], float(candidates[0][0][2])
        self._accept(candidates[0], timestamp, seed=True)
        return self._result(timestamp, "seeded", candidate=candidates[0], count=1)

    def _accept(self, candidate, timestamp, *, seed=False):
        camera, world, area, bbox = candidate
        if not seed:
            measured = (world-self.position)/(timestamp-self.last_ts)
            self.velocity = .5*self.velocity + .5*measured
        self.position, self.last_ts, self.area = world, timestamp, area
        self.extent = (bbox[2]-bbox[0], bbox[3]-bbox[1])
        self.last_z = camera[2]

    def update(self, depth, *, timestamp, pose, now, active_uid, searching=False):
        """No new visual box required. Does not extend the visual lease."""
        self._diagnostics = {}
        if self.position is None:
            return self._result(timestamp, "needs_visual_seed")
        if searching or active_uid != self.uid:
            self.reset()
            return self._result(timestamp, "identity_or_search_revoked")
        error = self._input_error(depth, timestamp, pose)
        if error:
            self.position = None
            return self._result(timestamp, error)
        if (not math.isfinite(now) or not 0 <= now-self.visual_ts <= .35):
            self.position = None
            return self._result(timestamp, "visual_lease_expired")
        if timestamp <= self.last_ts:
            # No update, no renewal, but duplicate is not an association loss.
            return self._result(timestamp, "duplicate_or_old")
        dt = timestamp-self.last_ts
        if not 0 <= now-timestamp <= .18 or dt > .18:
            self.position = None
            return self._result(timestamp, "depth_expired_or_gap")
        predicted_world = self.position+self.velocity*dt
        predicted = pose.camera(predicted_world)
        if predicted[2] < .35:
            self.position = None
            return self._result(timestamp, "prediction_out_of_view")
        u = self.k.fx*predicted[0]/predicted[2]+self.k.cx
        v = self.k.fy*predicted[1]/predicted[2]+self.k.cy
        scale = self.last_z/predicted[2]
        hw = self.extent[0]*scale*.9 + 4 + self.k.fx*2*dt/predicted[2]
        hh = self.extent[1]*scale*.9 + 4
        if not 0 <= u < self.k.width or not 0 <= v < self.k.height:
            self.position = None
            return self._result(timestamp, "prediction_out_of_view")
        roi = (max(0, int(u-hw)), max(0, int(v-hh)),
               min(self.k.width, math.ceil(u+hw+1)), min(self.k.height, math.ceil(v+hh+1)))
        expected_area = self.area*scale*scale
        # A succession of small accepted expansions must not grow the allowed
        # footprint without bound before the next visual identity anchor.
        seed_expected_area = self._seed_area*(self._seed_z/predicted[2])**2
        self._diagnostics.update(expected_area_px=float(expected_area),
                                 seed_expected_area_px=float(seed_expected_area),
                                 area_growth_limit=2.5,
                                 world_step_limit_m=.03+2*dt,
                                 prediction_residual_limit_m=.05+2*dt)
        candidates = self._candidates(depth, roi, predicted[2], pose,
                                      max(20, int(expected_area*.25)))
        viable = []
        audits = [a for a in self._diagnostics["candidates"] if not a["reject_reasons"]]
        for candidate, audit in zip(candidates, audits):
            camera, world, area, bbox = candidate
            step = float(np.linalg.norm(world-self.position))
            residual = float(np.linalg.norm(world-predicted_world))
            audit.update(area_ratio=float(area/expected_area), world_step_m=step,
                         seed_area_ratio=float(area/seed_expected_area), prediction_residual_m=residual)
            if area > expected_area*2.5:
                audit["reject_reasons"].append("area_growth")
            if area > seed_expected_area*2.5:
                audit["reject_reasons"].append("seed_area_growth")
            if step > .03+2*dt:
                audit["reject_reasons"].append("world_step")
            if residual > .05+2*dt:
                audit["reject_reasons"].append("prediction_error")
            if audit["reject_reasons"]:
                continue
            viable.append(candidate)
        if len(viable) != 1:
            self.position = None
            return self._result(timestamp, "association_ambiguous_or_lost", count=len(viable))
        self._accept(viable[0], timestamp)
        return self._result(timestamp, "tracked", candidate=viable[0], count=1)
