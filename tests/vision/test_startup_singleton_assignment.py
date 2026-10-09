"""Hardware-free checks for the one-person startup assignment fast path."""

import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rk_vision.deepsort import linear_assignment
from rk_vision.deepsort.detection import Detection
from rk_vision.deepsort.kalman_filter import KalmanFilter
from rk_vision.deepsort.nn_matching import NearestNeighborDistanceMetric
from rk_vision.deepsort.tracker import Tracker


@pytest.mark.parametrize("cost", [-2.0, 0.0, 0.2, 1e5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_finite_singleton_returns_independent_numpy_indices(cost, dtype):
    matrix = np.array([[cost]], dtype=dtype)
    before = matrix.copy()
    rows, cols = linear_assignment._linear_sum_assignment(matrix)
    for indices in (rows, cols):
        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.dtype(int)
        np.testing.assert_array_equal(indices, [0])
    assert rows is not cols
    np.testing.assert_array_equal(matrix, before)
    rows[0] = 7
    np.testing.assert_array_equal(cols, [0])
    # A previous result cannot corrupt later zero-cost assignments.
    again = linear_assignment._linear_sum_assignment(np.zeros((1, 1)))
    np.testing.assert_array_equal(again, [[0], [0]])


@pytest.mark.parametrize("cost,accepted", [(0.0, True), (0.25, True), (0.251, False), (1e5, False)])
def test_singleton_still_obeys_max_distance_and_original_indices(cost, accepted):
    seen = []

    def metric(tracks, detections, track_indices, detection_indices):
        seen.append((track_indices, detection_indices))
        return [[cost]]

    result = linear_assignment.min_cost_matching(
        metric, 0.25, [None] * 4, [None] * 6, [3], [5],
    )
    assert seen == [([3], [5])]
    assert result == (([(3, 5)], [], []) if accepted else ([], [3], [5]))


@pytest.mark.parametrize("matrix", [
    np.zeros((0, 1)), np.zeros((1, 0)), np.zeros((2, 2)),
    np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-np.inf]]),
    np.array([[1 + 0j]]), np.array([[1]], dtype=object),
])
def test_other_shapes_and_nonfinite_costs_keep_existing_solver_path(monkeypatch, matrix):
    calls = []
    expected = (np.array([0]), np.array([0]))
    optimize = ModuleType("scipy.optimize")

    def solver(received):
        calls.append(received)
        return expected

    optimize.linear_sum_assignment = solver
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize)
    assert linear_assignment._linear_sum_assignment(matrix) is expected
    assert calls == [matrix]


def test_nonfinite_solver_failure_still_uses_existing_fallback(monkeypatch):
    matrix = np.array([[np.nan]])
    calls = []
    expected = (np.array([], dtype=int), np.array([], dtype=int))
    optimize = ModuleType("scipy.optimize")

    def solver(received):
        assert received is matrix
        raise ValueError("invalid numeric entries")

    def fallback(received):
        calls.append(received)
        return expected

    optimize.linear_sum_assignment = solver
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize)
    monkeypatch.setattr(linear_assignment, "_greedy_linear_assignment", fallback)
    assert linear_assignment._linear_sum_assignment(matrix) is expected
    assert calls == [matrix]


@pytest.mark.parametrize("far", [False, True])
def test_singleton_assignment_preserves_kalman_motion_gate(far):
    kf = KalmanFilter()
    mean, covariance = kf.initiate(np.array([100.0, 200.0, 0.5, 200.0]))
    tracks = [SimpleNamespace(mean=mean, covariance=covariance)]
    detections = [Detection([900 if far else 50, 100, 100, 200], 0.9, 0, [1.0, 0.0])]

    def metric(tracks, detections, track_indices, detection_indices):
        return linear_assignment.gate_cost_matrix(
            kf, np.array([[0.01]], dtype=np.float32), tracks, detections,
            track_indices, detection_indices,
        )

    result = linear_assignment.min_cost_matching(metric, 0.2, tracks, detections)
    assert result == (([], [0], [0]) if far else ([(0, 0)], [], []))


@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("rejection", [None, "validator", "class", "geometry"])
def test_tracker_singleton_keeps_validator_class_and_geometry_gates(confirmed, rejection):
    tracker = Tracker(NearestNeighborDistanceMetric("cosine", 0.2), n_init=2)

    def detection(*, far=False, label=0, source=11):
        return Detection([900 if far else 50, 100, 100, 200], 0.9, label, [1.0, 0.0],
                         source_detection_index=source)

    tracker.update([detection()])
    if confirmed:
        tracker.predict()
        tracker.update([detection()])
    track = tracker.tracks[0]
    assert track.is_confirmed() is confirmed
    tracker.predict()
    validation = []

    def validator(track_id, source):
        validation.append((track_id, source))
        return rejection != "validator"

    candidate = detection(far=rejection == "geometry", label=1 if rejection == "class" else 0,
                          source=17)
    matches, unmatched_tracks, unmatched_detections = tracker._match([candidate], validator)
    assert validation == [(1, 17)]
    assert (matches, unmatched_tracks, unmatched_detections) == (
        ([(0, 0)], [], []) if rejection is None else ([], [0], [0])
    )


def test_singleton_startup_in_fresh_process_never_imports_scipy():
    # Blocking imports also detects a swallowed import failure followed by the
    # old fallback. Repeated finite calls must not even attempt that import.
    code = """
import builtins
import sys
import numpy as np
from rk_vision.deepsort.linear_assignment import min_cost_matching
original_import = builtins.__import__
attempts = []
def guarded_import(name, *args, **kwargs):
    if name == 'scipy' or name.startswith('scipy.'):
        attempts.append(name)
        raise ImportError('scipy must remain unused for a sole finite pair')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
for cost in (0.0, 0.0, 0.1, 100000.0):
    result = min_cost_matching(lambda *args: np.array([[cost]]), 0.2, [None], [None])
    assert result == (([(0, 0)], [], []) if cost <= 0.2 else ([], [0], [0]))
assert attempts == [], attempts
assert not any(name == 'scipy' or name.startswith('scipy.') for name in sys.modules)
"""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
        env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
