"""Classify measured edge clipping without creating identity permission."""
import math

from .camera_geometry import yaw_image_shift_ratio


def outward_edge_crop(current, previous, earlier, *, width, height, hfov):
    """Two accepted positions may explain a later narrow crop for one second.

    These positions are not a renewable geometry anchor. The caller retains
    the original identity reference and all explicit cross-person conflicts.
    A positive result only means this crop is insufficient negative evidence.
    """
    try:
        w, h, fov = float(width), float(height), float(hfov)
        boxes = [tuple(map(float, row['bbox'])) for row in (earlier, previous, current)]
        stamps = [float(row['capture_timestamp']) for row in (earlier, previous, current)]
        caps = [int(row['capture_frame_id']) for row in (earlier, previous, current)]
        yaws = [float(row['integrated_yaw_deg']) for row in (earlier, previous, current)]
        if (not all(math.isfinite(v) for v in (w, h, fov, *stamps, *yaws,
                                               *(v for box in boxes for v in box)))
                or min(w, h, fov) <= 0
                or not caps[0] < caps[1] < caps[2]
                or not 0 < stamps[1]-stamps[0] <= .35
                or not 0 < stamps[2]-stamps[1] <= 1.
                or abs(yaws[2]-yaws[1]) > 15.
                or any(row.get('geometry_source') != 'detector'
                       for row in (earlier, previous, current))
                or any(not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h)
                       for x1, y1, x2, y2 in boxes)):
            return None
        old, prev, cur = boxes
        left, right = cur[0] <= .02*w, w-cur[2] <= .02*w
        if left == right:
            return None
        side = -1. if left else 1.
        centers = [(box[0]+box[2])/(2*w) for box in boxes]
        # Require measured outward motion before the quality loss, then a
        # small same-side displacement from that accepted position.
        first = centers[1]-centers[0]-yaw_image_shift_ratio(yaws[0], yaws[1], fov)
        last = centers[2]-centers[1]-yaw_image_shift_ratio(yaws[1], yaws[2], fov)
        widths = [box[2]-box[0] for box in boxes]
        heights = [box[3]-box[1] for box in boxes]
        vertical_overlap = max(0., min(prev[3],cur[3])-max(prev[1],cur[1]))
        if (first is None or last is None or side*first < .02 or abs(first) > .15
                or not -.02 <= side*last <= .20
                or (centers[1] > .30 if left else centers[1] < .70)
                or (centers[2] > .12 if left else centers[2] < .88)
                or not .25 <= widths[2]/widths[1] <= .65
                or heights[1] < .70*h or heights[2] < .70*h
                or min(heights[1],heights[2])/max(heights[1],heights[2]) < .85
                or vertical_overlap/max(heights[1],heights[2]) < .85):
            return None
        return dict(reference_capture_frame_id=caps[1],
                    reference_capture_timestamp=stamps[1],
                    age_sec=stamps[2]-stamps[1], side='left' if left else 'right',
                    local_center_displacement=last,
                    width_ratio=widths[2]/widths[1], identity_authorized=False)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def continued_edge_crop(current, last_crop, accepted, *, side, width, height, hfov):
    """Continue only the classification 'unknown cropped observation'.

    A continuously measured narrow strip does not turn into cross-person
    evidence when its accepted position ages. Its original accepted timestamp
    is never advanced, and the caller still returns UID0 for every such crop.
    """
    try:
        w, h = float(width), float(height)
        box, old, origin = [tuple(map(float, row['bbox'])) for row in (current, last_crop, accepted)]
        stamp, prior = [float(row['capture_timestamp']) for row in (current, last_crop)]
        cap, prior_cap = [int(row['capture_frame_id']) for row in (current, last_crop)]
        yaw, prior_yaw = [float(row['integrated_yaw_deg']) for row in (current, last_crop)]
        if (not all(math.isfinite(v) for v in (w,h,stamp,prior,yaw,prior_yaw,*box,*old,*origin))
                or min(w,h) <= 0 or side not in ('left','right')
                or any(row.get('geometry_source') != 'detector' for row in (current,last_crop,accepted))
                or cap < prior_cap or not 0 <= stamp-prior <= .35
                or ((cap == prior_cap or stamp == prior) and
                    (cap != prior_cap or stamp != prior or box != old))
                or abs(yaw-prior_yaw) > 15.
                or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)):
            return False
        center, old_center = (box[0]+box[2])/(2*w), (old[0]+old[2])/(2*w)
        delta = center-old_center-yaw_image_shift_ratio(prior_yaw,yaw,hfov)
        width_ratio = (box[2]-box[0])/(origin[2]-origin[0])
        current_height, original_height = box[3]-box[1], origin[3]-origin[1]
        overlap = max(0., min(box[3],origin[3])-max(box[1],origin[1]))
        same_edge = (box[0] <= .02*w and center <= .12 if side == 'left'
                     else w-box[2] <= .02*w and center >= .88)
        return bool(same_edge and abs(delta) <= .08 and 0 < width_ratio <= .65
            and (box[2]-box[0])/(old[2]-old[0]) <= 1.5
            and min(current_height,original_height) >= .70*h
            and overlap/max(current_height,original_height) >= .85)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False
