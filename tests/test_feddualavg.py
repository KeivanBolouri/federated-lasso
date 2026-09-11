"""Independent algorithm identities and centralized convergence checks."""
from pathlib import Path
import sys
import unittest
import numpy as np
from sklearn.linear_model import Lasso

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ms" / "code"))
from feddualavg import feddualavg_path, local_lipschitz, kkt_mapping_residual


class FedDualAvgTests(unittest.TestCase):
    def test_first_update_is_pooled_gradient_prox(self):
        X1 = np.array([[1., 2.], [2., -1.]])
        X2 = np.array([[2., 0.], [0., 1.], [1., 1.]])
        y1, y2 = np.array([1., -1.]), np.array([2., 0., -1.])
        sites, weights, eta, lam = [(X1, y1), (X2, y2)], [.4, .6], .05, .2
        got = feddualavg_path(sites, weights, lam, 1, [1], [eta], .7)[1][:, 0]
        score = .4 * X1.T @ y1 / 2 + .6 * X2.T @ y2 / 3
        expected = .7 * eta * np.sign(score) * np.maximum(np.abs(score) - lam, 0)
        np.testing.assert_allclose(got, expected, atol=1e-15)

    def test_one_step_per_round_matches_central_dual_averaging(self):
        rng = np.random.default_rng(831)
        sites = [(rng.normal(size=(n, 7)), rng.normal(size=n)) for n in [15, 30]]
        w, rates, lam = np.array([1/3, 2/3]), np.array([.01, .04]), .12
        got = feddualavg_path(sites, w, lam, 1, [5, 30], rates)
        z = np.zeros((7, 2))
        for t in range(30):
            b = np.sign(z) * np.maximum(abs(z) - t * rates * lam, 0)
            grad = sum(wj * X.T @ (X @ b - y[:, None]) / len(y)
                       for wj, (X, y) in zip(w, sites))
            z -= rates * grad
            if t + 1 in got:
                expected = np.sign(z) * np.maximum(abs(z) - (t + 1) * rates * lam, 0)
                np.testing.assert_allclose(got[t + 1], expected, atol=2e-15)

    def test_identical_orthogonal_clients_reach_analytic_lasso(self):
        X = np.sqrt(4) * np.eye(4)
        score = np.array([2., -.9, .1, 0.])
        sites = [(X, X @ score)] * 2
        for E in [1, 5]:
            got = feddualavg_path(sites, [.3, .7], .3, E, [1, 10], [1.])
            expected = np.sign(score) * np.maximum(abs(score) - .3, 0)
            for value in got.values():
                np.testing.assert_allclose(value[:, 0], expected, atol=1e-14)
        np.testing.assert_allclose(local_lipschitz(sites), [1., 1.])

    def test_two_local_steps_and_server_rate_match_direct_formula(self):
        rng = np.random.default_rng(771)
        sites = [(rng.normal(size=(n, 4)), rng.normal(size=n)) for n in [10, 20]]
        weights, eta, server_rate, lam = [.3, .7], .1, .8, .12
        dual = np.zeros(4)
        for weight, (X, y) in zip(weights, sites):
            score = X.T @ y / len(y)
            gram = X.T @ X / len(y)
            primal_one = eta * np.sign(score) * np.maximum(abs(score) - lam, 0)
            dual += weight * eta * (2 * score - gram @ primal_one)
        dual *= server_rate
        expected = np.sign(dual) * np.maximum(abs(dual) - 2 * server_rate * eta * lam, 0)
        got = feddualavg_path(sites, weights, lam, 2, [1], [eta], server_rate)[1][:, 0]
        np.testing.assert_allclose(got, expected, atol=1e-14)

    def test_identical_clients_multiple_local_steps_equal_central_steps(self):
        rng = np.random.default_rng(199)
        site = (rng.normal(size=(20, 6)), rng.normal(size=20))
        got = feddualavg_path([site, site], [.25, .75], .1, 3, [3], [.1])[3]
        reference = feddualavg_path([site], [1.], .1, 1, [9], [.1])[9]
        np.testing.assert_allclose(got, reference, atol=1e-14)

    def test_one_step_converges_to_central_lasso(self):
        rng = np.random.default_rng(820)
        X = rng.normal(size=(120, 9))
        y = X @ np.array([1., -.8, 0., .4, 0., 0., 0., 0., 0.]) + rng.normal(size=120)
        sites, weights = [(X[:40], y[:40]), (X[40:], y[40:])], [1/3, 2/3]
        lam, eta = .2, .5 / max(local_lipschitz(sites))
        got = feddualavg_path(sites, weights, lam, 1, [1000], [eta])[1000][:, 0]
        ref = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=10000).fit(X, y).coef_
        np.testing.assert_allclose(got, ref, atol=1e-10)
        self.assertLess(kkt_mapping_residual(sites, weights, lam, got), 1e-10)


if __name__ == "__main__":
    unittest.main()
