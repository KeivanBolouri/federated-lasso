"""Public diabetes regression illustration with fully held-out test rows.

Run from any directory: python ms/code/diabetes_study.py --replicates 100
The archived raw data, split assignments, fit outputs, summaries and manuscript
table are regenerated. Repeated splits measure variability on ONE fixed dataset;
their standard deviations are not population-level standard errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_diabetes
from threadpoolctl import threadpool_limits

from fedlasso import (consensus_admm, fedavg, local_solve, objective,
                      select_by_val, select_threshold, soft, tune_lambda,
                      tune_lambda_1se)

HERE = Path(__file__).resolve().parent
MS = HERE.parent
DATA = MS.parent / "data" / "diabetes"
E_GRID = (1, 3, 20)
TRAIN_SIZES = (130, 78, 52)
VALIDATION_SIZES = (40, 24, 16)
EPS = 1e-4
DEFAULT_SEED = 20260911
MAX_AVERAGING_ROUNDS = 500
P = 10
K = 3
WEIGHTS = np.array(TRAIN_SIZES, float) / sum(TRAIN_SIZES)


def archive_dataset():
    """Archive unscaled source data; no all-sample normalisation is used."""
    source = load_diabetes(scaled=False)
    X = np.asarray(source.data, float)
    y = np.asarray(source.target, float)
    assert X.shape == (442, 10) and y.shape == (442,)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    DATA.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(X, columns=source.feature_names)
    frame["progression"] = y
    raw_path = DATA / "diabetes_raw.csv"
    frame.to_csv(raw_path, index=False, float_format="%.17g")
    metadata = {
        "dataset": "Diabetes regression data used in Least Angle Regression",
        "loader": "sklearn.datasets.load_diabetes(scaled=False)",
        "scikit_learn_version": sklearn.__version__,
        "observations": len(y), "predictors": source.feature_names,
        "target": "quantitative disease progression one year after baseline",
        "source_url": "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html",
        "loader_documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html",
        "paper": "Efron, Hastie, Johnstone and Tibshirani (2004), Least Angle Regression, Annals of Statistics 32(2):407-499, doi:10.1214/009053604000000067",
        "raw_csv_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "raw_csv_encoding": "UTF-8, LF newlines, float_format=%.17g",
    }
    (DATA / "source_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return X, y, source.feature_names, metadata


def run_split(X, y, rep, seed):
    order = np.random.default_rng(seed).permutation(len(y))
    train_idx, val_idx, test_idx = order[:260], order[260:340], order[340:]
    # These sufficient summaries can be computed at the sites before sharing
    # the common scale. Validation/test rows do not enter any fitted transform.
    mu = X[train_idx].mean(axis=0)
    scale = X[train_idx].std(axis=0, ddof=0)
    y_mu = y[train_idx].mean()
    if np.any(scale == 0):
        raise ValueError("A training feature is constant")
    Xs, yc = (X - mu) / scale, y - y_mu
    Xtr, ytr = Xs[train_idx], yc[train_idx]
    Xva, yva = Xs[val_idx], yc[val_idx]
    Xte, yte = Xs[test_idx], yc[test_idx]
    assert len(test_idx) == 102 and len(set(order)) == len(y)
    assert np.max(np.abs(Xtr.mean(axis=0))) < 1e-12
    assert np.allclose(Xtr.std(axis=0), 1)
    train_blocks = np.split(np.arange(260), np.cumsum(TRAIN_SIZES)[:-1])
    val_blocks = np.split(np.arange(80), np.cumsum(VALIDATION_SIZES)[:-1])
    sites = [(Xtr[idx], ytr[idx]) for idx in train_blocks]
    val = [(Xva[idx], yva[idx]) for idx in val_blocks]
    lams = [tune_lambda(*site, *v)[0] for site, v in zip(sites, val)]
    lam_bar = float(np.dot(WEIGHTS, lams))
    lam_pool_min = tune_lambda(Xtr, ytr, Xva, yva)[0]
    lam_pool_1se = tune_lambda_1se(Xtr, ytr, Xva, yva)[0]
    pooled_1se = local_solve(Xtr, ytr, lam_pool_1se)
    reference = np.abs(pooled_1se) > EPS
    rows = []

    def record(method, b, E=0, selected_rounds=0, fitting_rounds=0,
               scalar_kib=0, fit_converged=True, lam=np.nan, tau=np.nan,
               capped_candidates=0, diagnostics=None):
        active = np.abs(b) > EPS
        tp = int(np.sum(active & reference))
        fp = int(np.sum(active & ~reference))
        fn = int(np.sum(~active & reference))
        f1den, jden = 2 * tp + fp + fn, tp + fp + fn
        rows.append(dict(
            rep=rep, seed=seed, method=method, E=E, active=int(active.sum()),
            exact_zeros=int(np.sum(b == 0)),
            reference_active=int(reference.sum()),
            F1_reference=2 * tp / f1den if f1den else 1.0,
            jaccard_reference=tp / jden if jden else 1.0,
            test_mse=float(np.mean((yte - Xte @ b) ** 2)),
            validation_mse=float(np.mean((yva - Xva @ b) ** 2)),
            objective=objective(sites, b, WEIGHTS, lams),
            support_bits=int(sum(int(flag) << j for j, flag in enumerate(active))),
            fit_converged=bool(fit_converged), lambda_bar=lam_bar, lam=lam,
            capped_candidates=capped_candidates,
            diagnostics=json.dumps(diagnostics or {}),
            tau=tau, selected_rounds=selected_rounds,
            fitting_rounds=fitting_rounds,
            selected_local_epochs=selected_rounds * E if E else np.nan,
            total_local_epochs=fitting_rounds * E if E else np.nan,
            vector_kib=fitting_rounds * K * P * 8 / 1024,
            scalar_kib=scalar_kib,
            # Common feature sums, squared sums, n and y sum; plus local lambdas.
            setup_scalar_kib=K * (2 * P + 3) * 8 / 1024,
            **{f"coef_{j}": float(v) for j, v in enumerate(b)}))

    record("Pooled", local_solve(Xtr, ytr, lam_pool_min), lam=lam_pool_min)
    record("Pooled-1SE", pooled_1se, lam=lam_pool_1se)
    record("Largest-site", local_solve(*sites[0], lams[0]), lam=lams[0])
    lam_local_1se = tune_lambda_1se(*sites[0], *val[0])[0]
    record("Largest-site-1SE", local_solve(*sites[0], lam_local_1se), lam=lam_local_1se)
    locs = [local_solve(*s, lam) for s, lam in zip(sites, lams)]
    record("One-shot", sum(w * b for w, b in zip(WEIGHTS, locs)),
           selected_rounds=1, fitting_rounds=1, lam=lam_bar)
    b, r, diag = consensus_admm(sites, WEIGHTS, lam_bar, P,
                                max_iter=20000, return_diagnostics=True)
    if not diag["fit_converged"]:
        raise RuntimeError(f"ADMM failed for split {rep}: {diag}")
    record("ADMM", b, selected_rounds=r, fitting_rounds=r, lam=lam_bar)

    tau_grid = np.r_[0.0, np.geomspace(lam_bar * 1e-3, lam_bar * 5, 39)]
    pgrid = lam_bar * np.array([1, .5, .25, .125, .0625])
    for E in E_GRID:
        b, r, _, _, diag = fedavg(sites, WEIGHTS, lams, E, P,
                                  max_rounds=MAX_AVERAGING_ROUNDS,
                                  return_diagnostics=True)
        record("FedAvg", b, E, r, r, lam=lam_bar,
               fit_converged=diag["fit_converged"], diagnostics=diag)
        tau, _, _ = select_threshold(b, val, WEIGHTS, tau_grid, rule="1se")
        record("FedAvg-ST", soft(b, tau), E, r, r,
               scalar_kib=K * len(tau_grid) * 3 * 8 / 1024,
               lam=lam_bar, tau=tau, fit_converged=diag["fit_converged"],
               diagnostics=diag)
        solutions, rounds, candidate_diags = [], [], []
        for threshold in pgrid:
            bp, rp, _, _, dp = fedavg(sites, WEIGHTS, lams, E, P,
                                      tau=threshold, max_rounds=MAX_AVERAGING_ROUNDS,
                                      return_diagnostics=True)
            solutions.append(bp)
            rounds.append(rp)
            candidate_diags.append(dp)
        candidates = np.asarray(solutions).T
        chosen = select_by_val(candidates, val, WEIGHTS, rule="1se")
        record("FedAvg-P", candidates[:, chosen], E, rounds[chosen], sum(rounds),
               scalar_kib=K * len(pgrid) * 3 * 8 / 1024,
               lam=lam_bar, tau=pgrid[chosen],
               fit_converged=candidate_diags[chosen]["fit_converged"],
               capped_candidates=sum(not d["fit_converged"] for d in candidate_diags),
               diagnostics={"selected_candidate": int(chosen),
                            "candidate_rounds": rounds,
                            "candidate_diagnostics": candidate_diags})

    assignment = []
    for role, indices, sizes in (("train", train_idx, TRAIN_SIZES),
                                 ("validation", val_idx, VALIDATION_SIZES)):
        for site, idx in enumerate(np.split(indices, np.cumsum(sizes)[:-1]), start=1):
            assignment.extend(dict(rep=rep, seed=seed, row_index=int(i),
                                   role=role, site=site) for i in idx)
    assignment.extend(dict(rep=rep, seed=seed, row_index=int(i), role="test", site=0)
                      for i in test_idx)
    return rows, assignment


def summarize(raw, feature_names):
    metrics = ["active", "F1_reference", "jaccard_reference", "test_mse",
               "validation_mse", "fitting_rounds", "selected_rounds",
               "vector_kib", "scalar_kib", "total_local_epochs"]
    groups = raw.groupby(["method", "E"], sort=False)
    summary = groups[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    stability, frequencies = [], []
    for (method, E), group in groups:
        supports = np.array([[(int(mask) >> k) & 1 for k in range(P)]
                             for mask in group.support_bits], dtype=bool)
        js = []
        for i in range(len(supports)):
            union = np.logical_or(supports[i], supports[i+1:]).sum(axis=1)
            inter = np.logical_and(supports[i], supports[i+1:]).sum(axis=1)
            js.extend(np.divide(inter, union, out=np.ones(len(union), float),
                                where=union > 0))
        stability.append(dict(method=method, E=E, n_splits=len(group),
                              capped_selected=int((~group.fit_converged).sum()),
                              capped_candidates=int(group.capped_candidates.sum()),
                              pairwise_jaccard=float(np.mean(js)) if js else np.nan))
        for k, name in enumerate(feature_names):
            frequencies.append(dict(method=method, E=E, feature=name,
                                    selection_frequency=float(supports[:, k].mean())))
    summary = summary.merge(pd.DataFrame(stability), on=["method", "E"])
    summary.to_csv(HERE / "diabetes_summary.csv", index=False)
    pd.DataFrame(frequencies).to_csv(HERE / "diabetes_selection_frequencies.csv", index=False)
    return summary


def write_table(summary):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Public diabetes data: repeated training/validation/test splits. "
             r"Test MSE is on the original outcome scale; parentheses give its "
             r"standard deviation across splits, not a standard error. "
             r"$F_1$ is agreement with the pooled one-standard-error fit within "
             r"each split. Stability is mean pairwise Jaccard overlap between "
             r"a method's selected sets across splits. Neither measures truth recovery. "
             r"Cap is the number of returned fits that reached the 500-round "
             r"averaging limit without satisfying the stopping criterion.}",
             r"\label{tab:diabetes}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Method & $E$ & Active & $F_1$ & Test MSE (SD) & Stability & Cap \\",
             r"\midrule"]
    for row in summary.itertuples():
        estr = str(row.E) if row.E else "---"
        lines.append(f"{row.method} & {estr} & {row.active_mean:.2f} & "
                     f"{row.F1_reference_mean:.3f} & "
                     f"{row.test_mse_mean:.0f} ({row.test_mse_std:.0f}) & "
                     f"{row.pairwise_jaccard:.3f} & {row.capped_selected} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (MS / "tables").mkdir(exist_ok=True)
    (MS / "tables" / "diabetes_results.tex").write_text("\n".join(lines) + "\n")
    indexed = summary.set_index(["method", "E"])
    pm, ps = indexed.loc[("Pooled", 0)], indexed.loc[("Pooled-1SE", 0)]
    fa1, fa20 = indexed.loc[("FedAvg", 1)], indexed.loc[("FedAvg", 20)]
    st1, st20 = indexed.loc[("FedAvg-ST", 1)], indexed.loc[("FedAvg-ST", 20)]
    result_text = (
        f"The pooled minimum-MSE fit retained {pm.active_mean:.2f} predictors "
        f"on average, compared with {ps.active_mean:.2f} under one-standard-error "
        f"tuning; their mean test MSEs were {pm.test_mse_mean:.0f} and "
        f"{ps.test_mse_mean:.0f}, respectively. "
        f"Plain averaging retained {fa1.active_mean:.2f} predictors at $E=1$ "
        f"and {fa20.active_mean:.2f} at $E=20$. "
        f"Post-fit thresholding retained {st1.active_mean:.2f} and "
        f"{st20.active_mean:.2f}, with agreement $F_1={st1.F1_reference_mean:.3f}$ "
        f"and ${st20.F1_reference_mean:.3f}$ against the corresponding pooled "
        f"one-standard-error reference. Its mean test MSEs were "
        f"{st1.test_mse_mean:.0f} and {st20.test_mse_mean:.0f}. "
        "Feature-level selection frequencies and results for each split "
        "are supplied in the replication files.\n\n")
    capped = summary[summary.method == "FedAvg-P"]
    cap_counts = ", ".join(str(int(capped.loc[capped.E == E, "capped_selected"].iloc[0]))
                           for E in E_GRID)
    total_candidates = int(capped.capped_candidates.sum())
    total_fits = int(capped.n_splits.sum() * 5)
    fa_caps = int(indexed.loc[("FedAvg", 1), "capped_selected"])
    result_text += (
        f"Plain averaging reached the cap in {fa_caps} splits at $E=1$; "
        "post-fit thresholding inherits those same capped base iterates. "
        f"For per-round thresholding, {cap_counts} selected fits at "
        "$E=1,3,20$, respectively, reached the round cap without meeting "
        f"the update criterion. Across its {total_fits} candidate fits, "
        f"{total_candidates} reached that cap. These returned iterates remain "
        "in the summaries; no splits or candidate thresholds were discarded. "
        "Their metrics describe the stated finite-budget procedure and do not "
        "establish convergence of per-round thresholding.\n")
    (MS / "tables" / "diabetes_result_text.tex").write_text(result_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--from-results", action="store_true",
                        help="Rebuild tables/summaries from archived fit output without refitting")
    args = parser.parse_args()
    if args.from_results:
        raw = pd.read_csv(HERE / "diabetes_results.csv", float_precision="round_trip")
        source = json.loads((DATA / "source_metadata.json").read_text())
        summary = summarize(raw, source["predictors"])
        write_table(summary)
        write_metadata(raw, source)
        print(f"Rebuilt diabetes summaries for {raw.rep.nunique()} splits from archived results")
        return
    if args.replicates < 2:
        parser.error("At least two repeated splits are needed for split variability")
    X, y, names, source = archive_dataset()
    rows, assignments = [], []
    with threadpool_limits(limits=1):
        for rep in range(args.replicates):
            new_rows, split = run_split(X, y, rep, args.seed + rep)
            rows.extend(new_rows)
            assignments.extend(split)
            if (rep + 1) % 10 == 0:
                print(f"Completed diabetes split {rep + 1}/{args.replicates}", flush=True)
    raw = pd.DataFrame(rows)
    raw["upload_kib"] = raw.vector_kib + raw.scalar_kib + raw.setup_scalar_kib
    raw.to_csv(HERE / "diabetes_results.csv", index=False)
    pd.DataFrame(assignments).to_csv(DATA / "split_assignments.csv", index=False)
    summary = summarize(raw, names)
    write_table(summary)
    write_metadata(raw, source)
    print(summary[["method", "E", "active_mean", "F1_reference_mean",
                   "test_mse_mean", "test_mse_std", "pairwise_jaccard"]].to_string(index=False))


def write_metadata(raw, source):
    metadata = dict(source=source, n_splits=int(raw.rep.nunique()),
                    first_seed=int(raw.seed.min()), last_seed=int(raw.seed.max()),
                    training_sizes=TRAIN_SIZES, validation_sizes=VALIDATION_SIZES,
                    test_size=102, local_epochs=E_GRID, activity_threshold=EPS,
                    averaging_round_cap=MAX_AVERAGING_ROUNDS,
                    preprocessing="Shared feature mean and population SD, and outcome mean, fitted from training rows only; largest-site fits share this preprocessing",
                    reference="Pooled Lasso with validation one-standard-error penalty in the same split",
                    variability="SD across reused-data splits, not independent-dataset MCSE",
                    all_returned_iterates_converged=bool(raw.fit_converged.all()),
                    all_candidate_fits_converged=bool(raw.fit_converged.all()
                                                     and raw.capped_candidates.sum() == 0),
                    returned_capped_count=int((~raw.fit_converged).sum()),
                    capped_p_candidate_count=int(raw.capped_candidates.sum()))
    (HERE / "diabetes_study_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
