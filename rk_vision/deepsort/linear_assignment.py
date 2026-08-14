from __future__ import annotations

import itertools
from typing import Callable, Optional, Sequence

from . import kalman_filter


INFTY_COST = 1e5


def min_cost_matching(
    distance_metric: Callable,
    max_distance: float,
    tracks: Sequence,
    detections: Sequence,
    track_indices: Optional[Sequence[int]] = None,
    detection_indices: Optional[Sequence[int]] = None,
):
    np = _np()
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    else:
        track_indices = list(track_indices)
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    else:
        detection_indices = list(detection_indices)

    if not detection_indices or not track_indices:
        return [], track_indices, detection_indices

    cost_matrix = np.asarray(
        distance_metric(tracks, detections, track_indices, detection_indices),
        dtype="float32",
    )
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    row_indices, col_indices = _linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = []
    unmatched_detections = []
    col_set = set(int(col) for col in col_indices)
    row_set = set(int(row) for row in row_indices)
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_set:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_set:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[int(row)]
        detection_idx = detection_indices[int(col)]
        if cost_matrix[int(row), int(col)] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
    distance_metric: Callable,
    max_distance: float,
    cascade_depth: int,
    tracks: Sequence,
    detections: Sequence,
    track_indices: Optional[Sequence[int]] = None,
    detection_indices: Optional[Sequence[int]] = None,
):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = list(detection_indices)
    matches = []
    for level in range(int(cascade_depth)):
        if not unmatched_detections:
            break
        track_indices_l = [idx for idx in track_indices if tracks[idx].time_since_update == 1 + level]
        if not track_indices_l:
            continue
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric,
            max_distance,
            tracks,
            detections,
            track_indices_l,
            unmatched_detections,
        )
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - {idx for idx, _ in matches})
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
    kf,
    cost_matrix,
    tracks: Sequence,
    detections: Sequence,
    track_indices: Sequence[int],
    detection_indices: Sequence[int],
    gated_cost: float = INFTY_COST,
    only_position: bool = False,
):
    np = _np()
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices], dtype="float32")
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix


def _linear_sum_assignment(cost_matrix):
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost_matrix)
    except Exception:
        return _greedy_linear_assignment(cost_matrix)


def _greedy_linear_assignment(cost_matrix):
    np = _np()
    rows, cols = cost_matrix.shape
    if rows == 0 or cols == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    if min(rows, cols) <= 8:
        return _exhaustive_linear_assignment(cost_matrix)
    pairs = sorted(
        ((float(cost_matrix[row, col]), row, col) for row in range(rows) for col in range(cols)),
        key=lambda item: item[0],
    )
    used_rows = set()
    used_cols = set()
    out_rows = []
    out_cols = []
    for _, row, col in pairs:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        out_rows.append(row)
        out_cols.append(col)
        if len(used_rows) == rows or len(used_cols) == cols:
            break
    return np.asarray(out_rows, dtype=int), np.asarray(out_cols, dtype=int)


def _exhaustive_linear_assignment(cost_matrix):
    np = _np()
    rows, cols = cost_matrix.shape
    if rows <= cols:
        best_cols = None
        best_cost = float("inf")
        for cols_perm in itertools.permutations(range(cols), rows):
            cost = sum(float(cost_matrix[row, col]) for row, col in enumerate(cols_perm))
            if cost < best_cost:
                best_cost = cost
                best_cols = cols_perm
        return np.arange(rows, dtype=int), np.asarray(best_cols, dtype=int)
    row_idx, col_idx = _exhaustive_linear_assignment(cost_matrix.T)
    return col_idx, row_idx


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for DeepSORT assignment") from exc
    return np
