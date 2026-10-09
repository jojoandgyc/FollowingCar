"""CPU-only regression for single-axis assignment and startup preparation."""

import builtins
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rk_vision.deepsort import linear_assignment
from rk_vision import pipeline as pipeline_module


@pytest.mark.parametrize("values", [
    [0.3, 0.1, 0.2], [0.1, 0.1, 0.3], [3, -4, -4],
    [0.0, -0.0, 0.0], [100000.0, 100000.0], [0.2],
])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.float32, np.float64])
def test_finite_single_axis_matches_reference_without_import(values, transpose, dtype, monkeypatch):
    reference = pytest.importorskip("scipy.optimize").linear_sum_assignment
    matrix = np.asarray([values], dtype=dtype)
    if transpose:
        matrix = matrix.T
    expected = reference(matrix)
    original = matrix.copy()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("single-axis finite matrix must not load a solver")

    optimize = ModuleType("scipy.optimize")
    optimize.linear_sum_assignment = forbidden
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize)
    monkeypatch.setattr(linear_assignment, "_greedy_linear_assignment", forbidden)
    actual = linear_assignment._linear_sum_assignment(matrix)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(matrix, original)
    assert actual[0] is not actual[1]


@pytest.mark.parametrize("transpose", [False, True])
def test_large_integer_ties_use_same_float64_comparison_as_reference(transpose):
    reference = pytest.importorskip("scipy.optimize").linear_sum_assignment
    matrix = np.asarray([[2**53 + 1, 2**53]], dtype=np.int64)
    if transpose:
        matrix = matrix.T
    np.testing.assert_array_equal(
        linear_assignment._linear_sum_assignment(matrix), reference(matrix),
    )


@pytest.mark.parametrize("values,expected_index", [
    ([0.30, 0.19, 0.22], 1), ([0.25, 0.25], 0), ([0.251, 1e5], None),
    ([np.inf, 0.10, 1e5], 1),
])
@pytest.mark.parametrize("transpose", [False, True])
def test_threshold_clipping_and_unmatched_order_are_preserved(values, expected_index, transpose):
    matrix = np.asarray([values], dtype=np.float32)
    if transpose:
        matrix = matrix.T
    track_indices = list(range(10, 10 + matrix.shape[0]))
    detection_indices = list(range(20, 20 + matrix.shape[1]))
    expected_matches = []
    expected_tracks = list(track_indices)
    expected_detections = list(detection_indices)
    if expected_index is not None:
        track = track_indices[expected_index if transpose else 0]
        detection = detection_indices[0 if transpose else expected_index]
        expected_matches = [(track, detection)]
        expected_tracks.remove(track)
        expected_detections.remove(detection)
    else:
        # Rejected assigned pairs are appended after unassigned rows/columns.
        if transpose:
            expected_tracks = expected_tracks[1:] + expected_tracks[:1]
        else:
            expected_detections = expected_detections[1:] + expected_detections[:1]
    result = linear_assignment.min_cost_matching(
        lambda *_: matrix.copy(), 0.25, [None] * 20, [None] * 30,
        track_indices, detection_indices,
    )
    assert result == (expected_matches, expected_tracks, expected_detections)


@pytest.mark.parametrize("matrix", [
    np.asarray([[0.0, np.nan]]), np.asarray([[np.inf, 1.0]]),
    np.asarray([[0.0], [-np.inf]]), np.asarray([[1, 2]], dtype=object),
    np.asarray([[1j, 0j]]), np.empty((0, 3)), np.empty((4, 0)),
    np.zeros((2, 2)),
])
def test_unsupported_or_empty_matrix_keeps_general_solver_and_fallback(matrix, monkeypatch):
    calls = []
    expected = (np.asarray([], dtype=int), np.asarray([], dtype=int))
    optimize = ModuleType("scipy.optimize")

    def failing_solver(received):
        assert received is matrix
        calls.append("solver")
        raise ValueError("keep original failure handling")

    def fallback(received):
        assert received is matrix
        calls.append("fallback")
        return expected

    optimize.linear_sum_assignment = failing_solver
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize)
    monkeypatch.setattr(linear_assignment, "_greedy_linear_assignment", fallback)
    assert linear_assignment._linear_sum_assignment(matrix) is expected
    assert calls == ["solver", "fallback"]


def test_preparation_uses_general_solver_without_creating_track_state(monkeypatch):
    calls = []
    optimize = ModuleType("scipy.optimize")

    def solve(matrix):
        calls.append(matrix.copy())
        assert matrix.shape == (2, 2)
        return np.asarray([0, 1]), np.asarray([1, 0])

    optimize.linear_sum_assignment = solve
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize)
    assert linear_assignment.prepare_assignment_backend() == "scipy"
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], [[0.3, 0.1], [0.2, 0.4]])


@pytest.mark.parametrize("missing", [False, True])
def test_preparation_failure_keeps_optional_dependency_fallback(monkeypatch, missing):
    original_import = builtins.__import__
    fallback_calls = []
    real_fallback = linear_assignment._greedy_linear_assignment

    def guarded_import(name, *args, **kwargs):
        if name == "scipy.optimize":
            if missing:
                raise ImportError("optional scipy unavailable")
            return SimpleNamespace(linear_sum_assignment=lambda _: (_ for _ in ()).throw(
                ValueError("solver unavailable")))
        return original_import(name, *args, **kwargs)

    def fallback(matrix):
        fallback_calls.append(matrix.shape)
        return real_fallback(matrix)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(linear_assignment, "_greedy_linear_assignment", fallback)
    assert linear_assignment.prepare_assignment_backend() == "fallback"
    matrix = np.asarray([[0.3, 0.1], [0.2, 0.4]])
    rows, cols = linear_assignment._linear_sum_assignment(matrix)
    np.testing.assert_array_equal(rows, [0, 1])
    np.testing.assert_array_equal(cols, [1, 0])
    assert fallback_calls == [(2, 2), (2, 2)]


def test_pipeline_constructor_prepares_before_model_or_tracker_work(monkeypatch):
    events = []

    def prepare():
        events.append("assignment")
        return "test_cpu_backend"

    def component(name):
        def construct(_config):
            events.append(name)
            return SimpleNamespace()
        return construct

    logs = []
    monkeypatch.setattr(pipeline_module, "prepare_assignment_backend", prepare)
    monkeypatch.setattr(pipeline_module, "YOLO11RKNNDetector", component("detector"))
    monkeypatch.setattr(pipeline_module, "OSNetRKNNExtractor", component("reid"))
    monkeypatch.setattr(pipeline_module, "DeepSortTracker", component("tracker"))
    pipeline = pipeline_module.RKNNVisionPipeline(
        pipeline_module.RKNNVisionConfig(yolo_model_path="unused.rknn"),
        logger=SimpleNamespace(info=lambda *args: logs.append(args)),
    )
    assert events == ["assignment", "detector", "reid", "tracker"]
    assert pipeline.assignment_backend == "test_cpu_backend"
    assert pipeline.assignment_prepare_ms >= 0.0
    assert logs[0][0].startswith("deepsort_assignment_prepared")
    assert logs[0][1] == "test_cpu_backend"
    assert pipeline.last_detections == []
    assert pipeline.last_timing_ms["total"] == 0.0


def test_single_axis_fresh_process_does_not_attempt_scipy_import():
    code = """
import builtins
import numpy as np
from rk_vision.deepsort.linear_assignment import _linear_sum_assignment
original_import = builtins.__import__
attempts = []
def guard(name, *args, **kwargs):
    if name == 'scipy' or name.startswith('scipy.'):
        attempts.append(name)
        raise ImportError('not needed')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guard
for matrix in (np.array([[.3, .1, .2]]), np.array([[.3], [.1], [.2]])):
    rows, cols = _linear_sum_assignment(matrix)
    assert float(matrix[rows[0], cols[0]]) == .1
assert attempts == [], attempts
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
