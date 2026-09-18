"""Pure, hardware-free tests for DeepSORT's squared Mahalanobis gate."""

from types import SimpleNamespace

import numpy as np
import pytest

from rk_vision.deepsort.kalman_filter import KalmanFilter, _solve_cholesky, chi2inv95
from rk_vision.deepsort.linear_assignment import INFTY_COST, gate_cost_matrix


def _expanded_xyah(bbox):
    x1, y1, x2, y2 = bbox
    width = (x2 - x1) * 1.20
    height = (y2 - y1) * 1.20
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, width / height, height])


def _cap338_prediction():
    # Fixed reconstruction from run_20260918_130838: initialize at control 50,
    # replay raw-track-1 detector geometry through control 114 / CAP334, then
    # predict control 115 / CAP338. This is not a captured Kalman-state dump.
    # Keeping the numeric fixture here makes the test independent of logs.
    mean = np.array([
        314.1671543611461, 254.90899080957826, 0.5061891684334239,
        528.4780045695641, 6.049882159225348, 0.4065203143341982,
        2.4327971316818896e-06, -1.000714728894904,
    ])
    covariance = np.diag([
        3256.200834170675, 3256.200834170675, 0.0010519658179817288,
        3256.200834170675, 212.69777049560133, 212.69777049560133,
        6.5943456706812155e-09, 212.69777049560133,
    ])
    for axis in (0, 1, 3):
        covariance[axis, axis + 4] = covariance[axis + 4, axis] = 282.9389641786315
    covariance[2, 6] = covariance[6, 2] = 5.827362079285034e-08
    wrong_person = _expanded_xyah([
        5.026790618896484, 192.73687744140625,
        82.20881652832031, 344.0897216796875,
    ])
    # The last trusted CAP334 detector box, used as a continuous-geometry
    # counterexample, not claimed to be the exact central CAP338 detection.
    continuous_person = _expanded_xyah([
        204.40484619140625, 34.199981689453125,
        407.4081726074219, 475.147216796875,
    ])
    return mean, covariance, np.array([wrong_person, continuous_person])


@pytest.mark.parametrize("only_position", [False, True])
def test_gate_matches_correlated_mahalanobis_quadratic_form(monkeypatch, only_position):
    kf = KalmanFilter()
    projected_mean = np.array([100.0, 80.0, 0.5, 200.0])
    factor = np.array([
        [2.0, 0.0, 0.0, 0.0], [0.5, 3.0, 0.0, 0.0],
        [0.1, 0.0, 0.2, 0.0], [1.0, -0.5, 0.1, 4.0],
    ])
    projected_cov = factor @ factor.T
    monkeypatch.setattr(kf, "project", lambda *_: (projected_mean, projected_cov))
    measurements = projected_mean + np.array([
        [0.0, 0.0, 0.0, 0.0], [4.0, 3.0, 0.03, 5.0],
        [-4.0, -3.0, -0.03, -5.0], [10.0, 7.0, 0.1, 12.0],
    ])
    dimensions = 2 if only_position else 4
    residual = (measurements - projected_mean)[:, :dimensions]
    covariance = projected_cov[:dimensions, :dimensions] + np.eye(dimensions) * 1e-7
    expected = np.einsum("ij,ji->i", residual, np.linalg.solve(covariance, residual.T))
    result = kf.gating_distance(None, None, measurements, only_position=only_position)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)
    assert result[0] == 0.0
    assert result[1] == pytest.approx(result[2])


def test_diagonal_covariance_does_not_silently_square_its_inverse():
    kf = KalmanFilter()
    mean = np.zeros(8)
    mean[3] = 1.0
    covariance = np.diag([4.0, 9.0, 1.0, 16.0, 1.0, 1.0, 1.0, 1.0])
    measurements = np.array([[10.0, 0.0, 0.0, 1.0]])
    result = kf.gating_distance(mean, covariance, measurements)
    assert result[0] == pytest.approx(100.0 / (4.0025 + 1e-7))
    assert result[0] > chi2inv95[4]  # Old inverse-squared result was 6.242.


def test_cap338_wrong_small_person_is_rejected_but_continuity_is_allowed():
    mean, covariance, measurements = _cap338_prediction()
    result = KalmanFilter().gating_distance(mean, covariance, measurements)
    assert result[0] == pytest.approx(40.12544255, rel=1e-7)
    assert result[0] > chi2inv95[4]
    assert result[1] < chi2inv95[4]


def test_assignment_gate_blocks_cap338_wrong_match_even_if_appearance_is_close():
    mean, covariance, measurements = _cap338_prediction()
    detections = [SimpleNamespace(to_xyah=lambda m=m: m) for m in measurements]
    # The wrong person's appearance may be under the cosine threshold. That
    # must not bypass the independent motion gate.
    cost = np.array([[0.19, 0.05]])
    result = gate_cost_matrix(
        KalmanFilter(), cost.copy(), [SimpleNamespace(mean=mean, covariance=covariance)],
        detections, [0], [0, 1],
    )
    assert result[0, 0] == INFTY_COST
    assert result[0, 1] == cost[0, 1]


def test_position_only_ignores_aspect_and_height_residual():
    kf = KalmanFilter()
    mean, covariance = kf.initiate(np.array([300.0, 200.0, 0.5, 400.0]))
    measurements = np.array([[300.0, 200.0, 3.0, 40.0]])
    assert kf.gating_distance(mean, covariance, measurements, only_position=True)[0] == 0.0
    assert kf.gating_distance(mean, covariance, measurements)[0] > chi2inv95[4]


@pytest.mark.parametrize("only_position", [False, True])
def test_gate_accepts_empty_measurement_batch(only_position):
    kf = KalmanFilter()
    mean, covariance = kf.initiate(np.array([300.0, 200.0, 0.5, 400.0]))
    result = kf.gating_distance(mean, covariance, np.empty((0, 4)), only_position)
    assert result.shape == (0,)


def test_shared_solver_still_returns_full_covariance_inverse_product():
    covariance = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([[3.0, -1.0], [2.0, 5.0]])
    np.testing.assert_allclose(
        _solve_cholesky(covariance, rhs),
        np.linalg.solve(covariance + np.eye(2) * 1e-7, rhs),
        rtol=1e-12,
    )


def test_normal_kalman_update_matches_full_gain_and_does_not_mutate_inputs():
    kf = KalmanFilter()
    mean, covariance = kf.initiate(np.array([300.0, 200.0, 0.5, 400.0]))
    mean, covariance = kf.predict(mean, covariance)
    measurement = np.array([306.0, 202.0, 0.51, 395.0])
    old_mean, old_cov = mean.copy(), covariance.copy()
    projected_mean, projected_cov = kf.project(mean, covariance)
    gain = np.linalg.solve(projected_cov + np.eye(4) * 1e-7, covariance[:4, :]).T
    expected_mean = mean + gain @ (measurement - projected_mean)
    expected_cov = covariance - gain @ covariance[:4, :]
    updated_mean, updated_cov = kf.update(mean, covariance, measurement)
    np.testing.assert_allclose(updated_mean, expected_mean)
    np.testing.assert_allclose(updated_cov, expected_cov)
    np.testing.assert_array_equal(mean, old_mean)
    np.testing.assert_array_equal(covariance, old_cov)
    assert np.linalg.eigvalsh(updated_cov).min() > 0.0
