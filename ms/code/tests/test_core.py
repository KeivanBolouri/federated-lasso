"""Numerical checks for solver, validation, support and communication semantics."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedlasso import (local_epochs, local_solve, consensus_admm, soft,
                      validation_summaries)
from run_sim import support_metrics, row


class CoreChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(121)
        self.X = rng.normal(size=(80, 12))
        self.y = self.X[:, :3] @ np.array([1.0, -.7, .5]) + rng.normal(size=80)

    def test_epoch_matches_explicit_coordinate_sweeps(self):
        b = np.linspace(-.1, .1, 12)
        expected = b.copy()
        for _ in range(3):
            for k in range(12):
                x = self.X[:, k]
                residual = self.y - self.X @ expected + x * expected[k]
                expected[k] = soft(x @ residual, .15 * len(self.y)) / (x @ x)
        np.testing.assert_allclose(local_epochs(self.X, self.y, b, .15, 3),
                                   expected, atol=1e-13)

    def test_consensus_solves_same_penalized_problem(self):
        sites = [(self.X[:50], self.y[:50]), (self.X[50:], self.y[50:])]
        z, _, diag = consensus_admm(sites, np.array([.625, .375]), .15, 12,
                                     tol=1e-10, return_diagnostics=True)
        expected = local_solve(self.X, self.y, .15)
        np.testing.assert_allclose(z, expected, atol=2e-8)
        self.assertTrue(diag['fit_converged'])
        self.assertLess(diag['kkt_residual'], 1e-8)

    def test_final_solver_extends_budget_until_dual_gap_passes(self):
        result, diag = local_solve(self.X, self.y, .15, max_iter=1,
                                  return_diagnostics=True)
        reference = local_solve(self.X, self.y, .15)
        self.assertGreater(diag['solver_restarts'], 0)
        self.assertLessEqual(diag['local_dual_gap'], diag['local_dual_gap_tolerance'])
        np.testing.assert_allclose(result, reference, atol=1e-9)

    def test_scalar_summaries_equal_pooled_residual_statistics(self):
        coefs = np.random.default_rng(43).normal(size=(12, 4))
        val = [(self.X[:50], self.y[:50]), (self.X[50:], self.y[50:])]
        mse, se = validation_summaries(coefs, val, np.array([.625, .375]))
        r2 = (self.y[:, None] - self.X @ coefs) ** 2
        np.testing.assert_allclose(mse, r2.mean(axis=0), rtol=1e-13)
        np.testing.assert_allclose(se, r2.std(axis=0, ddof=1) / np.sqrt(80), rtol=1e-13)

    def test_empty_support_and_candidate_cost_accounting(self):
        metrics = support_metrics(np.zeros(12), np.r_[np.ones(3), np.zeros(9)])
        for field in ['precision', 'recall', 'F1', 'jaccard']:
            self.assertEqual(metrics[field], 0)
        d = row('FedAvg-P', 2, np.zeros(12), np.zeros(12), (self.X, self.y),
                3, 6, total_fit_rounds=15, total_fit_epochs=30,
                scalar_kib=.35, candidate_count=5)
        self.assertEqual(d['rounds'], 15)
        self.assertEqual(d['selected_rounds'], 3)
        self.assertEqual(d['tuning_rounds'], 12)
        self.assertEqual(d['local_epochs'], 30)
        self.assertEqual(d['total_rounds'], 16)
        self.assertAlmostEqual(d['vector_kib'], 5*d['selected_vector_kib'])


if __name__ == '__main__':
    unittest.main()
