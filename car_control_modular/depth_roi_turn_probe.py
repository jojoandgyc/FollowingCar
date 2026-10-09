"""Read-only, bounded test of a time-aligned turning ROI on current Depth.

The configured camera model is not calibrated. Even multiple consistent regions
are NOT identity proof. Results must never enter the accepted range or motor lease.
"""
import math
import numpy as np


def project_bbox(bbox, *, width, height, hfov_deg, distance, source_pose, depth_pose):
    if (source_pose is None or depth_pose is None or not .35 <= distance <= 8
            or not 20 <= hfov_deg <= 120 or min(width, height) <= 0):
        return None, None, "invalid_geometry"
    dt = depth_pose.timestamp-source_pose.timestamp
    dyaw = depth_pose.yaw_rad-source_pose.yaw_rad
    if (not 0 <= dt <= .25 or abs(dyaw) > math.radians(8)
            or math.hypot(depth_pose.x_m-source_pose.x_m, depth_pose.z_m-source_pose.z_m) > .35):
        return None, None, "motion_out_of_bounds"
    f = width/(2*math.tan(math.radians(hfov_deg)/2))
    ca, sa = math.cos(source_pose.yaw_rad), math.sin(source_pose.yaw_rad)
    cb, sb = math.cos(depth_pose.yaw_rad), math.sin(depth_pose.yaw_rad)
    points = []
    for u, v in ((bbox[0], bbox[1]), (bbox[2], bbox[3])):
        x, y, z = (u-width/2)*distance/f, (v-height/2)*distance/f, distance
        wx = ca*x+sa*z+source_pose.x_m-depth_pose.x_m
        wz = -sa*x+ca*z+source_pose.z_m-depth_pose.z_m
        x, z = cb*wx-sb*wz, sb*wx+cb*wz
        if z <= .3:
            return None, None, "behind_camera"
        points.append((f*x/z+width/2, f*y/z+height/2, z))
    x1, y1, x2, y2 = points[0][0], points[0][1], points[1][0], points[1][1]
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None, None, "invalid_projection"
    clipped = (max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2))
    retained = max(0., clipped[2]-clipped[0])*max(0., clipped[3]-clipped[1])/((x2-x1)*(y2-y1))
    if retained < .7:
        return None, None, "projection_clipped"
    return clipped, sum(p[2] for p in points)/2, "projected"


def validate_regions(depth, bbox, expected_z):
    """Three disjoint torso strips; unrelated backgrounds cannot pass by count alone."""
    x1, y1, x2, y2 = bbox
    w, h = x2-x1, y2-y1
    regions = []
    for lo, hi in ((.20, .40), (.40, .60), (.60, .80)):
        patch = depth[max(0, int(y1+.25*h)):min(depth.shape[0], int(y1+.65*h)),
                      max(0, int(x1+lo*w)):min(depth.shape[1], int(x1+hi*w))]
        valid = patch[(patch >= 350) & (patch <= 8000)].astype(float)/1000.
        close = valid[np.abs(valid-expected_z) <= .25]
        supported = patch.size >= 8 and len(close) >= max(8, .4*patch.size, .6*len(valid))
        regions.append(dict(pixels=int(patch.size), valid=int(len(valid)),
                            associated=int(len(close)),
                            median_m=float(np.median(close)) if supported else None))
    medians = [r["median_m"] for r in regions if r["median_m"] is not None]
    consistent = len(medians) >= 2 and max(medians)-min(medians) <= .12
    return consistent, regions
