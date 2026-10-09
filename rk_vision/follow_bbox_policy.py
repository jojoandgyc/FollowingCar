"""Geometry eligibility for this upright-person following deployment.

This is not a generic person classifier: seated people can be real people but
are outside the requested follow/reacquire crop policy. Keep raw detections
for diagnostics. Never use a Kalman-expanded box to rescue a rejected crop.
"""
import math


def lower_compact_bbox_reason(bbox, image_width, image_height):
    if not image_width or not image_height or image_width <= 0 or image_height <= 0:
        return ""
    x1, y1, x2, y2 = map(float, bbox)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return ""
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return ""
    # Joint condition, not a global minimum height or left/right exclusion.
    # Preserve tall near-camera crops, narrow distant standing people, and
    # bottom-clipped crops. Ratios make the rule resolution independent.
    if (y1 / image_height >= .40 and height / image_height <= .42
            and y2 / image_height <= .92 and width / height >= .80):
        return "lower_compact_bbox"
    return ""
