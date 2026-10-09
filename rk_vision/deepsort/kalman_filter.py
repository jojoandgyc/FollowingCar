from __future__ import annotations

import os
import time
from functools import lru_cache


def _stamp():
    return time.perf_counter(), time.thread_time()


def _record(timing, name, started):
    if timing is not None:
        timing[name] = timing.get(name, 0.0) + (time.perf_counter() - started[0]) * 1000.
        timing[name + "_cpu"] = timing.get(name + "_cpu", 0.0) + (time.thread_time() - started[1]) * 1000.
        timing[name + "_calls"] = timing.get(name + "_calls", 0) + 1


def _timed(timing, name, fn, *args, **kwargs):
    started = _stamp() if timing is not None else None
    try:
        return fn(*args, **kwargs)
    finally:
        _record(timing, name, started)


@lru_cache(maxsize=1)
def _triangular_solver():
    from scipy.linalg import solve_triangular
    return solve_triangular


chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    def __init__(self, *, solver=None) -> None:
        np = _np()
        self.solver = solver or os.environ.get("FOLLOW_KALMAN_SOLVER", "numpy")
        if self.solver not in ("numpy", "triangular"):
            raise ValueError("FOLLOW_KALMAN_SOLVER must be numpy or triangular")
        if self.solver == "triangular":
            # Resolve optional dependency at initialization, not on a live frame.
            _triangular_solver()
        self.timing = {}
        ndim = 4
        dt = 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        np = _np()
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        started = _stamp()
        np = _np()
        speed = float(np.linalg.norm(mean[4:6]))
        position_noise_factor = 1.0 + speed / 10.0
        velocity_noise_factor = 1.0 + speed / 20.0
        std_pos = [
            self._std_weight_position * mean[3] * position_noise_factor,
            self._std_weight_position * mean[3] * position_noise_factor,
            1e-2,
            self._std_weight_position * mean[3] * position_noise_factor,
        ]
        std_vel = [
            self._std_weight_velocity * mean[3] * velocity_noise_factor,
            self._std_weight_velocity * mean[3] * velocity_noise_factor,
            1e-5,
            self._std_weight_velocity * mean[3] * velocity_noise_factor,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        _record(self.timing, "predict", started)
        return mean, covariance

    def project(self, mean, covariance):
        started = _stamp()
        np = _np()
        measurement_noise_factor = 1.5 if mean[3] > 100 else 1.0
        std = [
            self._std_weight_position * mean[3] * measurement_noise_factor,
            self._std_weight_position * mean[3] * measurement_noise_factor,
            1e-1,
            self._std_weight_position * mean[3] * measurement_noise_factor,
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        result = mean, covariance + innovation_cov
        _record(self.timing, "project", started)
        return result

    def update(self, mean, covariance, measurement):
        np = _np()
        projected_mean, projected_cov = self.project(mean, covariance)
        cross_cov = np.dot(covariance, self._update_mat.T)
        kalman_gain = _solve_cholesky(projected_cov, cross_cov.T,
                                      solver=self.solver, timing=self.timing).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, self._update_mat, covariance))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position: bool = False):
        np = _np()
        projected_mean, projected_cov = self.project(mean, covariance)
        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]
        d = measurements - projected_mean
        inverse_weighted_residual = _solve_cholesky(projected_cov, d.T,
                                                  solver=self.solver, timing=self.timing)
        # The shared solver returns S^-1 d, not the whitened residual L^-1 d.
        # Squaring that result would apply the inverse covariance twice,
        # weakening position/size gating while over-penalizing aspect changes.
        # Keep the solver's full inverse-product contract used by update().
        return np.sum(d.T * inverse_weighted_residual, axis=0)


def _solve_cholesky(a, b, *, solver="numpy", timing=None):
    np = _np()
    if solver not in ("numpy", "triangular"):
        raise ValueError("Unknown Kalman solver: %s" % solver)
    jitter = 1e-7
    eye = np.eye(a.shape[0], dtype=a.dtype)
    for _ in range(5):
        try:
            chol = _timed(timing, "cholesky", np.linalg.cholesky, a + jitter * eye)
            if solver == "triangular":
                solve = _triangular_solver()
                y = _timed(timing, "solve_lower", solve, chol, b, lower=True, check_finite=False)
                return _timed(timing, "solve_upper", solve, chol.T, y, lower=False, check_finite=False)
            y = _timed(timing, "solve_lower", np.linalg.solve, chol, b)
            return _timed(timing, "solve_upper", np.linalg.solve, chol.T, y)
        except np.linalg.LinAlgError:
            if timing is not None:
                timing["cholesky_retries"] = timing.get("cholesky_retries", 0) + 1
            jitter *= 10.0
    return _timed(timing, "fallback", np.linalg.solve, a + jitter * eye, b)


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for DeepSORT Kalman filtering") from exc
    return np
