"""Association-only low-score bridge. Never creates a track or an identity.

High-score association always runs first. A weak current detector crop can
fill a short gap only against a recent real high-score observation, not a
Kalman prediction or its own previous weak matches. Identity qualification
remains the caller's responsibility.
"""
import math

from ..camera_geometry import horizontal_center_displacement


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def capture_anchor(detection, context):
    import numpy as np

    if detection is None or not context:
        return None
    cap, stamp = (_number(context.get(k)) for k in ("capture_frame_id", "capture_timestamp"))
    feature = detection.feature
    if (cap is None or cap <= 0 or not cap.is_integer() or stamp is None or stamp < 0
            or feature is None or not np.isfinite(feature).all()
            or float(np.linalg.norm(feature)) <= 1e-12):
        return None
    return dict(capture_frame_id=int(cap), capture_timestamp=stamp,
                integrated_yaw_deg=context.get("integrated_yaw_deg"),
                box=detection.tlwh.copy(), feature=feature.copy())


def match_low_score(tracks, detections, track_indices, detection_indices, *,
                    validator, context, image_shape, camera_hfov_deg):
    """Return only mutually unique viable pairs, with explicit current proof."""
    import numpy as np

    if validator is None or not context or not image_shape:
        return []
    current = {key: _number(context.get(key)) for key in
               ("capture_frame_id", "capture_timestamp", "integrated_yaw_deg")}
    if context.get("integrated_yaw_deg") is not None and current["integrated_yaw_deg"] is None:
        return []
    cap, stamp = current["capture_frame_id"], current["capture_timestamp"]
    height, width = (_number(value) for value in image_shape)
    if (cap is None or cap <= 0 or not cap.is_integer() or stamp is None
            or height is None or width is None or min(height, width) <= 0):
        return []
    viable = []
    for ti in track_indices:
        track = tracks[ti]
        anchor = track.low_score_anchor
        if (not track.is_confirmed() or not 0 < track.time_since_update <= 4
                or not anchor or cap <= anchor["capture_frame_id"]
                or not 0 < stamp - anchor["capture_timestamp"] <= .5):
            continue
        if (anchor.get("integrated_yaw_deg") is not None
                and _number(anchor["integrated_yaw_deg"]) is None):
            continue
        old = anchor["box"]
        old_area = float(old[2] * old[3])
        for di in detection_indices:
            det = detections[di]
            feature = det.feature
            if (int(det.cls) != int(track.cls) or feature is None
                    or feature.shape != anchor["feature"].shape
                    or not np.isfinite(feature).all()):
                continue
            norm = float(np.linalg.norm(feature))
            if norm <= 1e-12:
                continue
            distance = 1. - float(np.dot(feature, anchor["feature"]) /
                                  (norm * np.linalg.norm(anchor["feature"])))
            if not math.isfinite(distance) or distance > .35:
                continue
            box = det.tlwh
            area = float(box[2] * box[3])
            if min(area, old_area) / max(area, old_area) < .55:
                continue
            _, jump = horizontal_center_displacement(
                current_center=float(box[0] + box[2] / 2) / width,
                previous_center=float(old[0] + old[2] / 2) / width,
                current_yaw=context.get("integrated_yaw_deg"),
                previous_yaw=anchor.get("integrated_yaw_deg"),
                camera_hfov_deg=camera_hfov_deg)
            vertical = abs(float(box[1]+box[3]/2 - old[1]-old[3]/2)) / height
            if jump > .20 or vertical > .15:
                continue
            if validator(int(track.track_id), det.source_detection_index):
                viable.append((ti, di))
    # A low-score box must not resolve competition by a convenient assignment.
    return [(ti, di) for ti, di in viable
            if sum(t == ti for t, _ in viable) == 1
            and sum(d == di for _, d in viable) == 1]
