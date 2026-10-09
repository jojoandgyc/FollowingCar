"""Numerical and diagnostics regressions; no cameras/serial/motor initialization."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rk_vision.deepsort.kalman_filter import KalmanFilter, _solve_cholesky
from rk_vision.deepsort.deep_sort import DeepSort, DeepSortConfig
from rk_vision.numeric_runtime import numeric_runtime_info


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(2,), (2, 1), (2, 8), (4,), (4, 1), (4, 8), (4, 32)])
def test_triangular_solver_matches_existing_inverse_product(dtype, shape):
    pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(47)
    for _ in range(10):
        m = rng.normal(size=(shape[0], shape[0])).astype(dtype)
        a = m @ m.T + np.eye(shape[0], dtype=dtype) * .1
        b = rng.normal(size=shape).astype(dtype)
        expected = _solve_cholesky(a, b, solver="numpy")
        actual = _solve_cholesky(a, b, solver="triangular")
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("position_only", [True, False])
def test_gating_formula_and_long_trajectory_unchanged(position_only):
    pytest.importorskip("scipy.linalg")
    old, new = KalmanFilter(solver="numpy"), KalmanFilter(solver="triangular")
    rng = np.random.default_rng(53)
    z = np.array([320., 220., .45, 330.])
    a, ac = old.initiate(z)
    b, bc = new.initiate(z)
    for step in range(200):
        a, ac = old.predict(a, ac)
        b, bc = new.predict(b, bc)
        z = np.array([320.+step*.8, 220.+np.sin(step/20.)*5, .45, 330.-step*.4])
        candidates = np.tile(z, (3, 1)) + rng.normal(size=(3, 4)) * [.8, .8, .01, .5]
        ga = old.gating_distance(a, ac, candidates, position_only)
        gb = new.gating_distance(b, bc, candidates, position_only)
        np.testing.assert_allclose(ga, gb, rtol=1e-8, atol=1e-8)
        np.testing.assert_array_equal(ga < 9.4877, gb < 9.4877)
        a, ac = old.update(a, ac, z)
        b, bc = new.update(b, bc, z)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(ac, bc, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("solver", ["numpy", "triangular"])
@pytest.mark.parametrize("diagonal", [0., -1e-7, -1.])
def test_retry_and_fallback_semantics_preserved(solver, diagonal):
    if solver == "triangular": pytest.importorskip("scipy.linalg")
    a, b = np.eye(4) * diagonal, np.arange(8.).reshape(4, 2)
    profile = {}
    result = _solve_cholesky(a, b, solver=solver, timing=profile)
    np.testing.assert_allclose(result, _solve_cholesky(a, b, solver="numpy"))
    if diagonal == -1.:
        assert profile["cholesky_retries"] == 5
        assert profile["fallback_calls"] == 1
    elif diagonal == -1e-7:
        assert profile["cholesky_retries"] == 1
        assert profile["cholesky_calls"] == 2
    else:
        assert profile.get("cholesky_retries", 0) == 0
    assert all(v >= 0 for v in profile.values())


def test_per_frame_profile_resets_including_empty_frame(monkeypatch):
    monkeypatch.setenv("FOLLOW_KALMAN_SOLVER", "numpy")
    tracker = DeepSort(DeepSortConfig(n_init=1))
    for _ in range(3):
        tracker.update([[300., 200., 100., 300.]], [.95], [0], [np.ones(16)], image_shape=(480, 640))
    profile = tracker.tracker.last_timing_ms
    assert profile["matched_count"] == 1
    assert profile["kf_predict_calls"] == 1
    assert profile["kf_solve_lower_calls"] >= 1
    tracker.update([], [], [], [], image_shape=(480, 640))
    profile = tracker.tracker.last_timing_ms
    assert profile["matched_count"] == 0
    assert profile["detections_count"] == 0
    assert profile.get("kf_solve_lower_calls", 0) == 0
    assert profile["kf_predict_calls"] == 1


def test_invalid_solver_rejected(monkeypatch):
    monkeypatch.setenv("FOLLOW_KALMAN_SOLVER", "typo")
    with pytest.raises(ValueError): KalmanFilter()


@pytest.mark.parametrize("mode", [None, "1", "2", "4", "8", "inherit", "bad"])
def test_launcher_applies_threads_before_any_python_import(mode):
    script = (Path(__file__).resolve().parents[2] / "run_request_0428_modular.sh").read_text()
    # Execute only the environment preamble. Never run launcher/hardware/log rotation.
    preamble = script.split('\nROOT=', 1)[0]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", FOLLOW_KALMAN_SOLVER="numpy")
    env.pop("FOLLOW_NUMERIC_THREADS", None)
    if mode is not None: env["FOLLOW_NUMERIC_THREADS"] = mode
    result = subprocess.run(["sh", "-c", preamble + '\nprintf "%s/%s" "$OPENBLAS_NUM_THREADS" "$OMP_NUM_THREADS"'],
                            env=env, text=True, capture_output=True)
    if mode == "bad":
        assert result.returncode == 2
    else:
        expected = "4" if mode == "inherit" else mode or "1"
        assert result.returncode == 0
        assert result.stdout == expected + "/" + expected


def test_actual_library_thread_count_in_fresh_process():
    result = subprocess.run([sys.executable, "-c",
        "import json; from rk_vision.numeric_runtime import numeric_runtime_info; print(json.dumps(numeric_runtime_info()))"],
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True, text=True, check=True)
    audit = json.loads(result.stdout)
    assert audit["numpy_path"]
    observed = [lib["actual_threads"] for lib in audit["libraries"] if lib["actual_threads"] is not None]
    if observed: assert all(count == 1 for count in observed)


def test_audit_does_not_change_environment():
    before = dict(os.environ)
    assert numeric_runtime_info()["numpy_version"]
    assert dict(os.environ) == before
