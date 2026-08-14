from __future__ import annotations


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
    def __init__(self) -> None:
        np = _np()
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
        return mean, covariance

    def project(self, mean, covariance):
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
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        np = _np()
        projected_mean, projected_cov = self.project(mean, covariance)
        cross_cov = np.dot(covariance, self._update_mat.T)
        kalman_gain = _solve_cholesky(projected_cov, cross_cov.T).T
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
        z = _solve_cholesky(projected_cov, d.T)
        return np.sum(z * z, axis=0)


def _solve_cholesky(a, b):
    np = _np()
    jitter = 1e-7
    eye = np.eye(a.shape[0], dtype=a.dtype)
    for _ in range(5):
        try:
            chol = np.linalg.cholesky(a + jitter * eye)
            y = np.linalg.solve(chol, b)
            return np.linalg.solve(chol.T, y)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.linalg.solve(a + jitter * eye, b)


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for DeepSORT Kalman filtering") from exc
    return np
