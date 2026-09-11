"""Audit released results against raw counts, candidate fits and byte accounting.

Run with ``python -m unittest discover -s tests -v``. These checks do not
rerun experiments or change result files. They independently recompute the
reported means, Monte Carlo errors and costs from the committed raw records.
"""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


CODE = Path(__file__).resolve().parents[1] / "ms" / "code"
MAIN_KEYS = ["scenario", "method", "E"]
BUDGET_KEYS = ["scenario", "method", "E", "rounds"]
PAIR_KEYS = ["scenario", "rep", "E", "rounds"]
SEEDS = set(range(5000, 5067)) | set(range(6000, 6067)) | set(range(7000, 7066))


class CommittedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = pd.read_csv(CODE / "sim_all_raw.csv")
        cls.budget = pd.read_csv(CODE / "budget_raw.csv")
        cls.candidates = pd.read_csv(CODE / "budget_candidates.csv")
        cls.ledger = pd.read_csv(CODE / "budget_cost_ledger.csv")

    def assert_close(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12,
                                   equal_nan=True)

    def test_main_sample_completeness_and_support_scores(self):
        d = self.main
        self.assertEqual(len(d), 29400)
        self.assertFalse(d.duplicated(["scenario", "rep", "method", "E"]).any())
        self.assertTrue((d.groupby(["scenario", "rep"]).size() == 49).all())
        self.assertEqual(set(d.scenario), {"Independent", "Correlated", "Heterogeneous"})
        for _, g in d.groupby("scenario"):
            self.assertEqual(set(g.rep), SEEDS)
        self.assert_close(d.active, d.TP + d.FP)
        self.assert_close(d.TP + d.FN, 30)
        self.assert_close(d.TP + d.FP + d.FN + d.TN, 600)
        for field, numerator, denominator in [
            ("precision", d.TP, d.TP + d.FP),
            ("recall", d.TP, d.TP + d.FN),
            ("F1", 2 * d.TP, 2 * d.TP + d.FP + d.FN),
            ("jaccard", d.TP, d.TP + d.FP + d.FN),
        ]:
            expected = np.divide(numerator.to_numpy(dtype=float), denominator,
                                 out=np.zeros(len(d)), where=denominator != 0)
            self.assert_close(d[field], expected)
        empty = d[d.active == 0]
        self.assertEqual(len(empty), 219)
        self.assertTrue((empty[["precision", "recall", "F1", "jaccard"]] == 0).all().all())

    def test_main_summary_means_and_monte_carlo_errors(self):
        saved = pd.read_csv(CODE / "sim_summary_new.csv").set_index(MAIN_KEYS).sort_index()
        groups = self.main.groupby(MAIN_KEYS, dropna=False)
        self.assert_close(saved.n, groups.size().sort_index())
        for col in saved:
            if col.endswith("_m"):
                expected = groups[col[:-2]].mean().sort_index()
            elif col.endswith("_se"):
                expected = groups[col[:-3]].std(ddof=1).sort_index().to_numpy() / np.sqrt(saved.n.to_numpy())
            else:
                continue
            self.assert_close(saved[col], expected)

    def test_main_selected_and_all_candidate_costs(self):
        d = self.main
        bytes_per_round_kib = 3 * 600 * 8 / 1024
        self.assert_close(d.vector_kib, d.rounds * bytes_per_round_kib)
        self.assert_close(d.selected_vector_kib, d.selected_rounds * bytes_per_round_kib)
        self.assert_close(d.tuning_rounds, d.rounds - d.selected_rounds)
        self.assert_close(d.tuning_vector_kib, d.vector_kib - d.selected_vector_kib)
        self.assert_close(d.upload_kib, d.vector_kib + d.scalar_kib)
        self.assert_close(d.total_rounds, d.rounds + d.scalar_rounds)
        p = d[d.method == "FedAvg-P"]
        for r in p.itertuples():
            rounds = json.loads(r.candidate_rounds)
            self.assertEqual(len(rounds), 5)
            self.assertEqual(r.rounds, sum(rounds))
            self.assertIn(r.selected_rounds, rounds)
        self.assert_close(p.local_epochs, p.rounds * p.E)
        self.assert_close(p.selected_local_epochs, p.selected_rounds * p.E)
        self.assert_close(p.scalar_kib, 3 * 5 * 3 * 8 / 1024)
        st = d[d.method.isin(["FedAvg-ST", "FedAvg-ST-min"])]
        self.assert_close(st.scalar_kib, 3 * 40 * 3 * 8 / 1024)
        self.assertTrue(d[d.method.isin(["ADMM-CV", "ADMM-CV-1SE"])].upload_kib.isna().all())

    def test_budget_completeness_and_validation_selection(self):
        d, c = self.budget, self.candidates
        self.assertEqual(len(d), 4650)
        self.assertEqual(len(c), 6750)
        self.assertEqual(len(self.ledger), 150)
        self.assertFalse(d.duplicated(["scenario", "rep", "method", "E", "rounds"]).any())
        self.assertFalse(c.duplicated(PAIR_KEYS + ["rate_multiplier"]).any())
        self.assertTrue((d.groupby(BUDGET_KEYS).size() == 50).all())
        self.assertTrue((c.groupby(PAIR_KEYS).size() == 3).all())
        for _, g in d.groupby("scenario"):
            self.assertEqual(set(g.rep), set(range(5000, 5050)))
        chosen = c.sort_values(["validation_mse", "rate_multiplier"]).groupby(PAIR_KEYS).head(1)
        self.assertTrue(chosen.selected.all())
        self.assertEqual(int(c.selected.sum()), len(chosen))
        selected = c[c.selected].merge(d[d.method == "FedDualAvg-adapted"],
                                      on=PAIR_KEYS, suffixes=("_candidate", "_returned"),
                                      validate="one_to_one")
        for metric in ["lam", "client_rate", "rate_multiplier", "validation_mse",
                       "F1", "active", "objective", "objective_gap", "kkt_mapping", "test_mse"]:
            self.assert_close(selected[metric + "_candidate"], selected[metric + "_returned"])
        pooled = d[d.method == "Pooled-common"]
        self.assertTrue((pooled.kkt_mapping < 1e-7).all())
        self.assertTrue(pooled.selected_vector_upload_bytes.isna().all())
        ref = d.merge(pooled[["scenario", "rep", "objective"]], on=["scenario", "rep"],
                      suffixes=("", "_reference"), validate="many_to_one")
        self.assert_close(ref.objective_gap, ref.objective - ref.objective_reference)

    def test_budget_selected_standalone_and_shared_study_costs(self):
        d = self.budget[self.budget.E > 0]
        dual = (d.method == "FedDualAvg-adapted").astype(int)
        candidates = 1 + 2 * dual
        vector_bytes = 3 * 600 * 8
        self.assert_close(d.selected_vector_upload_bytes, d.rounds * vector_bytes)
        self.assert_close(d.standalone_all_candidate_vector_bytes, candidates * d.rounds * vector_bytes)
        self.assert_close(d.standalone_setup_scalar_bytes, (1 + dual) * 3 * 8)
        self.assert_close(d.standalone_validation_scalar_bytes, dual * 3 * 3 * 8)
        self.assert_close(d.standalone_total_upload_bytes,
                          candidates * d.rounds * vector_bytes + (1 + dual) * 24 + dual * 72)
        self.assert_close(d.selected_local_steps, d.rounds * d.E)
        self.assert_close(d.all_candidate_local_steps, candidates * d.rounds * d.E)
        l = self.ledger
        for field, expected in {
            "common_setup_scalar_bytes": 24,
            "dual_rate_setup_scalar_bytes": 24,
            "dual_all_E_vector_bytes": 3 * 3 * 100 * vector_bytes,
            "dual_all_E_all_checkpoints_validation_scalar_bytes": 3 * 5 * 3 * 3 * 8,
            "cd_all_E_vector_bytes": 3 * 100 * vector_bytes,
            "dual_total_gradient_evaluations_per_site": 3 * 100 * (1 + 5 + 20),
            "cd_total_coordinate_sweeps_per_site": 100 * (1 + 5 + 20),
        }.items():
            self.assert_close(l[field], expected)

    def test_budget_summary_and_paired_monte_carlo_errors(self):
        saved = pd.read_csv(CODE / "budget_summary.csv").set_index(BUDGET_KEYS).sort_index()
        groups = self.budget.groupby(BUDGET_KEYS)
        self.assert_close(saved.n, groups.size().sort_index())
        for metric in ["F1", "active", "objective_gap", "kkt_mapping", "test_mse"]:
            self.assert_close(saved[metric + "_mean"], groups[metric].mean().sort_index())
            self.assert_close(saved[metric + "_se"], groups[metric].std(ddof=1).sort_index() / np.sqrt(saved.n))
        for row in pd.read_csv(CODE / "budget_paired.csv").itertuples():
            group = self.budget[(self.budget.scenario == row.scenario) &
                                (self.budget.E == row.E) & (self.budget["rounds"] == row.rounds)]
            wide = group.pivot(index="rep", columns="method", values=row.metric)
            delta = wide["FedDualAvg-adapted"] - wide["FedAvg-CD-common"]
            self.assertEqual(row.n, len(delta))
            self.assert_close(row.mean, delta.mean())
            self.assert_close(row.se, delta.std(ddof=1) / np.sqrt(len(delta)))

    def test_common_total_upload_cap_uses_largest_affordable_checkpoint(self):
        saved = pd.read_csv(CODE / "budget_total_cap_summary.csv")
        cap = 100 * 3 * 600 * 8 + 3 * 8 * 5
        self.assertEqual(len(saved), 18)
        self.assert_close(saved.cap_upload_bytes, cap)
        expected_rounds = {"FedAvg-CD-common": 100, "FedDualAvg-adapted": 20}
        for row in saved.itertuples():
            d = self.budget[(self.budget.scenario == row.scenario) &
                            (self.budget.method == row.method) & (self.budget.E == row.E)]
            feasible = d[d.standalone_total_upload_bytes <= cap]
            self.assertEqual(row.rounds, feasible["rounds"].max())
            self.assertEqual(row.rounds, expected_rounds[row.method])
            chosen = d[d["rounds"] == row.rounds]
            self.assertEqual(len(chosen), 50)
            self.assert_close(chosen.standalone_total_upload_bytes, row.total_upload_bytes)
            self.assert_close(row.unused_upload_bytes, cap - row.total_upload_bytes)
            for metric in ["F1", "objective_gap", "test_mse"]:
                self.assert_close(getattr(row, metric + "_mean"), chosen[metric].mean())
                self.assert_close(getattr(row, metric + "_se"), chosen[metric].std(ddof=1) / np.sqrt(len(chosen)))
        for row in pd.read_csv(CODE / "budget_total_cap_paired.csv").itertuples():
            data = self.budget[(self.budget.scenario == row.scenario) & (self.budget.E == row.E)]
            data = data[((data.method == "FedAvg-CD-common") & (data["rounds"] == 100)) |
                        ((data.method == "FedDualAvg-adapted") & (data["rounds"] == 20))]
            wide = data.pivot(index="rep", columns="method", values=row.metric)
            delta = wide["FedDualAvg-adapted"] - wide["FedAvg-CD-common"]
            self.assertEqual(row.n, len(delta))
            self.assert_close(row.mean, delta.mean())
            self.assert_close(row.se, delta.std(ddof=1) / np.sqrt(len(delta)))


if __name__ == "__main__":
    unittest.main()
