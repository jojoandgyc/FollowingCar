"""Brief control continuity for a verified mapped body, never enrollment."""
import math


def mapped_crop_continuous(metadata, reference):
    """Do not let top/bottom edge counting toggle a broad body's identity.

    Reference is the last ordinary strong observation, NOT a previous crop
    continuation. The half-second budget cannot be rolled by these frames.
    Appearance, competition and contradiction checks belong to the bank.
    """
    m, r = metadata or {}, reference or {}
    try:
        x1, y1, x2, y2 = map(float, m['detector_bbox'])
        w, h = float(m['image_width']), float(m['image_height'])
        dt = float(m['capture_timestamp'])-float(r['capture_timestamp'])
        if not all(math.isfinite(v) for v in (x1,y1,x2,y2,w,h,dt)):
            return False
        return bool(m.get('is_fresh') is True and not m.get('search_reacquire_context_active')
            and m.get('track_id') == r.get('track_id')
            and m['capture_frame_id'] > r['capture_frame_id'] and 0 < dt <= .5
            and w > 0 and h > 0 and 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
            and y1 <= .02*h and h-y2 <= .02*h
            and ((x1 <= .02*w) != (w-x2 <= .02*w))
            and x2-x1 >= max(120., .18*w) and y2-y1 >= .85*h
            and .25 <= (x2-x1)/(y2-y1) <= 1.2)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
