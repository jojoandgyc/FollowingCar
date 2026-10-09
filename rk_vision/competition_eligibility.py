"""Bounded per-UID competitor eligibility, not a person/pose filter.

An accepted binding or its qualified gap-recovery observation supplies the
reference. Unknown/search-only candidates cannot exclude their own rivals.
"""
import math


def anchored_competitor_exclusions(
    detections, distances, *, reference, context, mapped_indices, width, height,
    confidence_floor, track_confidence_floor, appearance_limit, weak_only=False,
):
    """Exclude only simultaneous, clearly separate reduced-coverage rivals.

    The 350 ms / 5 degree window and high overlap deliberately avoid needing
    uncertain long-gap motion extrapolation. This is continuity support, not
    a way to reacquire identity. All limits must pass, not just box size.
    """
    try:
        if (not reference or not width or not height
                or not math.isfinite(width) or not math.isfinite(height)
                or width <= 0 or height <= 0 or context.get('is_fresh') is False):
            return {}, None
        cap = int(context['capture_frame_id'])
        old_cap = int(reference['capture_frame_id'])
        stamp = float(context['capture_timestamp'])
        old_stamp = float(reference['capture_timestamp'])
        yaw = float(context['integrated_yaw_deg'])
        old_yaw = float(reference['integrated_yaw_deg'])
        old_box = tuple(map(float, reference['bbox']))
        if (not all(math.isfinite(v) for v in (*old_box, stamp, old_stamp, yaw, old_yaw))
                or reference.get('geometry_source') != 'detector'
                or old_cap <= 0 or old_stamp <= 0
                or cap <= old_cap or not 0 < stamp - old_stamp <= .35
                or abs(yaw - old_yaw) > 5.):
            return {}, None
        ox1, oy1, ox2, oy2 = old_box
        old_area = (ox2 - ox1) * (oy2 - oy1)
        if min(ox2 - ox1, oy2 - oy1) <= 0:
            return {}, None
        anchors = []
        for i in mapped_indices:
            d = detections[i]
            x1, y1, x2, y2 = map(float, d.bbox)
            area = (x2 - x1) * (y2 - y1)
            intersection = (
                max(0., min(x2, ox2) - max(x1, ox1))
                * max(0., min(y2, oy2) - max(y1, oy1))
            )
            distance = distances.get(i)
            if (int(d.class_id) == 0 and float(d.score) >= confidence_floor
                    and distance is not None and math.isfinite(distance)
                    and distance <= appearance_limit
                    and all(math.isfinite(v) for v in (x1, y1, x2, y2, area, float(d.score)))
                    and min(x2 - x1, y2 - y1) > 0
                    and intersection / (area + old_area - intersection) >= .5
                    and min(area, old_area) / max(area, old_area) >= .6):
                anchors.append(i)
        if len(anchors) != 1:
            return {}, None
        index = anchors[0]
        x1, y1, x2, y2 = map(float, detections[index].bbox)
        anchor_height = y2 - y1
        anchor_area = (x2 - x1) * anchor_height
        center_x = (x1 + x2) / 2
        excluded = {}
        for i in distances:
            if i == index:
                continue
            d = detections[i]
            bx1, by1, bx2, by2 = map(float, d.bbox)
            box_width, box_height = bx2 - bx1, by2 - by1
            box_area = box_width * box_height
            if (not all(math.isfinite(v) for v in (bx1, by1, bx2, by2, float(d.score)))
                    or min(box_width, box_height) <= 0):
                continue  # Missing geometry is uncertainty, never permission.
            # A recovery hypothesis has less authority than a confirmed UID:
            # it can dismiss only absolutely small, below-track-floor boxes,
            # never a plausible high-confidence person, even if far smaller.
            if weak_only and not (
                float(d.score) < track_confidence_floor
                and box_area / (width * height) <= .02
                and box_height / height <= .25
            ):
                continue
            overlap = (max(0., min(x2, bx2) - max(x1, bx1))
                       * max(0., min(y2, by2) - max(y1, by1)))
            # Require separation PLUS a drastic change of body coverage;
            # small/low-score by itself is not negative identity evidence.
            if (box_height / anchor_height < .5 and box_area / anchor_area < .3
                    and overlap / min(anchor_area, box_area) <= .02
                    and abs((bx1 + bx2) / 2 - center_x) / width >= .15):
                excluded[i] = (
                    'weak_small_observation' if d.score < track_confidence_floor
                    else 'uid_scale_position_conflict'
                )
        return excluded, index
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError, IndexError):
        return {}, None
