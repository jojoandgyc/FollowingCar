from __future__ import annotations

from typing import Optional, Sequence

from . import linear_assignment


def iou(bbox, candidates):
    np = _np()
    if len(candidates) == 0:
        return np.zeros((0,), dtype="float32")
    bbox_tl = bbox[:2]
    bbox_br = bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0]), np.maximum(bbox_tl[1], candidates_tl[:, 1])]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0]), np.minimum(bbox_br[1], candidates_br[:, 1])]
    wh = np.maximum(0.0, br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))
    area_candidates = candidates[:, 2:].prod(axis=1)
    denom = area_bbox + area_candidates - area_intersection
    return np.where(denom > 0.0, area_intersection / denom, 0.0)


def iou_cost(
    tracks: Sequence,
    detections: Sequence,
    track_indices: Optional[Sequence[int]] = None,
    detection_indices: Optional[Sequence[int]] = None,
):
    np = _np()
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)), dtype="float32")
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        if track.time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue
        bbox = track.to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices], dtype="float32")
        costs = 1.0 - iou(bbox, candidates)
        for col, detection_idx in enumerate(detection_indices):
            if int(track.cls) != int(detections[detection_idx].cls):
                costs[col] = linear_assignment.INFTY_COST
        cost_matrix[row, :] = costs
    return cost_matrix


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for DeepSORT IoU matching") from exc
    return np
