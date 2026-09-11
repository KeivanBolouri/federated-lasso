"""Fixed-upload-budget comparison with an adapted FedDualAvg comparator.

Run from any directory:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python ms/code/budget_study.py --reps 50 --workers 2
  python ms/code/budget_study.py --outputs-only

All arms use the SAME common penalty lambda_bar at every site. This differs
from the main experiment's locally varying penalties and is labelled in the
manuscript. Step-size selection uses validation MSE, never test data/support.
Selected-fit vector cost is matched, but end-to-end tuning cost is not; both
are saved explicitly. Checkpoints are reused, and the total study-level ledger
counts this reuse instead of adding independent standalone budget costs.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from fedlasso import local_epochs, local_solve, objective, tune_lambda
from feddualavg import feddualavg_path, local_lipschitz, kkt_mapping_residual
from run_sim import make_data, P, MTR, EPS

ROOT = Path(__file__).resolve().parent
SCENARIOS = ("IID", "Correlated", "Heterogeneous")
E_GRID = (1, 5, 20)
ROUNDS = (5, 10, 20, 50, 100)
RATE_MULTIPLIERS = np.array([.1, .3, 1.])


def seed_schedule(reps):
    seeds = list(range(5000, 5067)) + list(range(6000, 6067)) + list(range(7000, 7066))
    if not 1 <= reps <= len(seeds):
        raise ValueError("reps must be between 1 and 200")
    return seeds[:reps]


def cd_path(sites, weights, lam, E, rounds):
    beta, out = np.zeros(sites[0][0].shape[1]), {}
    for r in range(1, max(rounds) + 1):
        beta = sum(w * local_epochs(X, y, beta, lam, E)
                   for w, (X, y) in zip(weights, sites))
        if r in rounds:
            out[r] = beta.copy()
    return out


def validation_losses(coefficients, val, weights):
    return sum(w * np.mean((yv[:, None] - Xv @ coefficients) ** 2, axis=0)
               for w, (Xv, yv) in zip(weights, val))


def metrics(bhat, true_beta, sites, weights, lam, optimum, test):
    active, truth = abs(bhat) > EPS, true_beta != 0
    tp, fp, fn = np.sum(active & truth), np.sum(active & ~truth), np.sum(~active & truth)
    obj = objective(sites, bhat, weights, [lam] * len(sites))
    Xt, yt = test
    return dict(F1=float(2 * tp / (2 * tp + fp + fn)), active=int(active.sum()),
                objective=float(obj), objective_gap=float(obj - optimum),
                kkt_mapping=kkt_mapping_residual(sites, weights, lam, bhat),
                test_mse=float(np.mean((yt - Xt @ bhat) ** 2)))


def one_rep(task):
    scenario, seed = task
    with threadpool_limits(limits=1):
        true_beta, sites, val, test = make_data(scenario, np.random.default_rng(seed))
        weights = np.array(MTR) / sum(MTR)
        local_lams = [tune_lambda(X, y, Xv, yv)[0]
                      for (X, y), (Xv, yv) in zip(sites, val)]
        lam = float(weights @ local_lams)
        lipschitz = local_lipschitz(sites)
        rates = RATE_MULTIPLIERS / max(lipschitz)
        Xp, yp = np.vstack([X for X, _ in sites]), np.concatenate([y for _, y in sites])
        ref = local_solve(Xp, yp, lam)
        optimum = objective(sites, ref, weights, [lam] * len(sites))
        K, p, J = len(sites), P, len(rates)
        vector_bytes = K * p * 8
        common_setup = K * 8  # each local lambda is transmitted once
        rate_setup = K * 8    # each local Lipschitz constant is transmitted once
        rows, candidates = [], []
        base = dict(scenario=scenario, rep=seed, lam=lam, L_max=float(max(lipschitz)))
        for E in E_GRID:
            dual = feddualavg_path(sites, weights, lam, E, ROUNDS, rates)
            cd = cd_path(sites, weights, lam, E, ROUNDS)
            for R in ROUNDS:
                losses = validation_losses(dual[R], val, weights)
                selected = int(np.argmin(losses))  # fixed ascending-rate tie rule
                for method, bhat in [("FedAvg-CD-common", cd[R]),
                                     ("FedDualAvg-adapted", dual[R][:, selected])]:
                    is_dual = method == "FedDualAvg-adapted"
                    n_candidates = J if is_dual else 1
                    setup = common_setup + (rate_setup if is_dual else 0)
                    scalar_eval = K * J * 8 if is_dual else 0
                    d = dict(base, method=method, E=E, rounds=R,
                             rate_multiplier=float(RATE_MULTIPLIERS[selected]) if is_dual else np.nan,
                             client_rate=float(rates[selected]) if is_dual else np.nan,
                             validation_mse=float(losses[selected]) if is_dual else np.nan,
                             selected_vector_upload_bytes=R * vector_bytes,
                             standalone_all_candidate_vector_bytes=n_candidates * R * vector_bytes,
                             standalone_setup_scalar_bytes=setup,
                             standalone_validation_scalar_bytes=scalar_eval,
                             standalone_total_upload_bytes=n_candidates * R * vector_bytes + setup + scalar_eval,
                             selected_local_steps=R * E,
                             all_candidate_local_steps=n_candidates * R * E,
                             local_unit="cached full-gradient evaluation" if is_dual else "cyclic CD sweep")
                    d.update(metrics(bhat, true_beta, sites, weights, lam, optimum, test))
                    rows.append(d)
                for k, eta in enumerate(rates):
                    d = dict(base, E=E, rounds=R, client_rate=float(eta),
                             rate_multiplier=float(RATE_MULTIPLIERS[k]),
                             validation_mse=float(losses[k]), selected=(k == selected))
                    d.update(metrics(dual[R][:, k], true_beta, sites, weights, lam, optimum, test))
                    candidates.append(d)
        d = dict(base, method="Pooled-common", E=0, rounds=0,
                 selected_vector_upload_bytes=np.nan, standalone_total_upload_bytes=np.nan)
        d.update(metrics(ref, true_beta, sites, weights, lam, optimum, test))
        rows.append(d)
        # Entire grid's cost with prefixes and common scalar setup shared.
        ledger = dict(base, common_setup_scalar_bytes=common_setup,
                      dual_rate_setup_scalar_bytes=rate_setup,
                      dual_all_E_vector_bytes=len(E_GRID) * J * max(ROUNDS) * vector_bytes,
                      dual_all_E_all_checkpoints_validation_scalar_bytes=len(E_GRID) * len(ROUNDS) * K * J * 8,
                      cd_all_E_vector_bytes=len(E_GRID) * max(ROUNDS) * vector_bytes,
                      local_penalty_grid_fits_per_site=30,
                      dual_local_gram_builds_per_site=len(E_GRID),
                      dual_total_gradient_evaluations_per_site=J * max(ROUNDS) * sum(E_GRID),
                      cd_total_coordinate_sweeps_per_site=max(ROUNDS) * sum(E_GRID),
                      reference_kkt_mapping=d["kkt_mapping"])
        return rows, candidates, ledger


def summarize(raw):
    measures = ["F1", "active", "objective_gap", "kkt_mapping", "test_mse"]
    records = []
    for key, g in raw.groupby(["scenario", "method", "E", "rounds"]):
        d = dict(zip(["scenario", "method", "E", "rounds"], key), n=len(g))
        for col in measures:
            d[col + "_mean"] = g[col].mean()
            d[col + "_se"] = g[col].std(ddof=1) / np.sqrt(len(g))
        records.append(d)
    return pd.DataFrame(records)


def make_outputs(raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = raw.groupby(["scenario", "method", "E", "rounds"]).size()
    expected_groups = len(SCENARIOS) * (2 * len(E_GRID) * len(ROUNDS) + 1)
    if len(counts) != expected_groups or counts.nunique() != 1:
        raise ValueError("budget output is incomplete or unbalanced; finish the simulation first")
    if raw.duplicated(["scenario", "rep", "method", "E", "rounds"]).any():
        raise ValueError("duplicate budget result rows")
    if raw["objective_gap"].min() < -1e-7:
        raise ValueError("a fit materially improves on the supposed pooled optimum")
    if raw.loc[raw.method == "Pooled-common", "kkt_mapping"].max() > 1e-7:
        raise ValueError("a pooled reference does not meet the optimality check")
    summary = summarize(raw)
    summary.to_csv(ROOT / "budget_summary.csv", index=False)
    paired = []
    for (scenario, E, R), group in raw[raw.E > 0].groupby(["scenario", "E", "rounds"]):
        for metric in ["F1", "objective_gap", "kkt_mapping", "test_mse"]:
            wide = group.pivot(index="rep", columns="method", values=metric)
            diff = wide["FedDualAvg-adapted"] - wide["FedAvg-CD-common"]
            paired.append(dict(scenario=scenario, E=E, rounds=R, metric=metric,
                               contrast="FedDualAvg-adapted minus FedAvg-CD-common",
                               mean=float(diff.mean()), se=float(diff.std(ddof=1) / np.sqrt(len(diff))), n=len(diff)))
    pd.DataFrame(paired).to_csv(ROOT / "budget_paired.csv", index=False)
    def format_gap(value):
        if value != 0 and abs(value) < .001:
            mantissa, exponent = f"{value:.1e}".split("e")
            return "$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$"
        return f"{value:.3f}"

    table = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Fixed common-penalty comparison after 100 selected-fit upload rounds. Means (Monte Carlo standard errors) over " + str(int(summary.n.min())) + r" replicates per design. CD is averaging of coordinate-descent iterates with common penalty; DA is adapted FedDualAvg. DA's three-rate search costs 300 vector-upload rounds. $G_\infty$ is the unit-step proximal-gradient mapping residual.}",
             r"\label{tab:budget}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Design & Method & $E$ & Objective gap & $G_\infty$ & $F_1$ & Test MSE \\", r"\midrule"]
    for scenario, label in zip(SCENARIOS, ["Independent", "Correlated", "Heterogeneous"]):
        first = True
        for method, method_label in [("FedAvg-CD-common", "CD"), ("FedDualAvg-adapted", "DA")]:
            for E in E_GRID:
                row = summary[(summary.scenario == scenario) & (summary.method == method) &
                              (summary.E == E) & (summary.rounds == 100)].iloc[0]
                vals = [format_gap(row["objective_gap_mean"]) + " (" + format_gap(row["objective_gap_se"]) + ")"]
                vals += [f"{row[m+'_mean']:.3f} ({row[m+'_se']:.3f})" for m in
                         ["kkt_mapping", "F1", "test_mse"]]
                table.append(f"{label if first else ''} & {method_label} & {E} & " + " & ".join(vals) + r" \\")
                first = False
        table.append(r"\addlinespace")
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (ROOT.parent / "tables" / "budget_main.tex").write_text("\n".join(table) + "\n")
    da = summary[(summary.method == "FedDualAvg-adapted") & (summary.rounds == 100)]
    cd = summary[(summary.method == "FedAvg-CD-common") & (summary.rounds == 100)]
    ranges = []
    for frame in [da, cd]:
        ranges.append(", ".join(f"${g.F1_mean.min():.3f}$--${g.F1_mean.max():.3f}$"
                                for scenario in SCENARIOS
                                for g in [frame[frame.scenario == scenario]]))
    description = (
        "At $100$ rounds, the ranges of mean true-support $F_1$ across the three local-step settings "
        "are " + ranges[0] + " for adapted \\textsc{FedDualAvg} in the independent, "
        "correlated and heterogeneous designs, respectively; the corresponding "
        "\\textsc{CD-common} ranges are " + ranges[1] + ". "
        "These comparisons condition on the selected step-size path; the additional tuning "
        "cost stated above remains part of the complete procedure.\n")
    (ROOT.parent / "tables" / "budget_description.tex").write_text(description)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0), sharex=True)
    colors = ["#2166ac", "#d6604d", "#4d9221"]
    for col, scenario in enumerate(SCENARIOS):
        for E, color in zip(E_GRID, colors):
            for method, short, style in [("FedAvg-CD-common", "CD", "--"), ("FedDualAvg-adapted", "DA", "-")]:
                subset = summary[(summary.scenario == scenario) & (summary.method == method) & (summary.E == E)].sort_values("rounds")
                for rr, metric in enumerate(["objective_gap", "F1"]):
                    axes[rr, col].errorbar(subset["rounds"], subset[metric+"_mean"],
                                          yerr=subset[metric+"_se"], color=color,
                                          linestyle=style, marker="o", markersize=3, linewidth=1.25,
                                          capsize=2, label=f"{short}, E={E}")
        ref = summary[(summary.scenario == scenario) & (summary.method == "Pooled-common")].iloc[0]
        axes[1, col].axhline(ref.F1_mean, color=".3", linestyle=":", linewidth=1, label="Pooled at common penalty")
        axes[0, col].set_yscale("log")
        axes[0, col].set_title(["Independent", "Correlated", "Heterogeneous"][col])
        for rr in range(2):
            axes[rr, col].set_xscale("log")
            axes[rr, col].set_xticks(ROUNDS)
            axes[rr, col].set_xticklabels(ROUNDS)
        axes[1, col].set_xlabel("Selected-fit upload rounds")
    axes[0, 0].set_ylabel("Objective gap (log scale)")
    axes[1, 0].set_ylabel("True-support F1")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.7)
    fig.tight_layout(rect=(0, .10, 1, 1))
    figure_path = ROOT.parent / "figures" / "budget_comparison.pdf"
    temporary = figure_path.with_suffix(".pdf.tmp")
    fig.savefig(temporary, format="pdf", bbox_inches="tight")
    temporary.replace(figure_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--outputs-only", "--from-results", dest="outputs_only", action="store_true")
    args = parser.parse_args()
    if args.outputs_only:
        make_outputs(pd.read_csv(ROOT / "budget_raw.csv"))
        return
    seeds = seed_schedule(args.reps)
    tasks = [(scenario, seed) for scenario in SCENARIOS for seed in seeds]
    raw, candidates, ledger = [], [], []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, (rows, cands, costs) in enumerate(executor.map(one_rep, tasks), 1):
            raw.extend(rows); candidates.extend(cands); ledger.append(costs)
            if i == 1 or i % 5 == 0:
                pd.DataFrame(raw).to_csv(ROOT / "budget_raw.csv", index=False)
                pd.DataFrame(candidates).to_csv(ROOT / "budget_candidates.csv", index=False)
                pd.DataFrame(ledger).to_csv(ROOT / "budget_cost_ledger.csv", index=False)
                print(f"Budget study: {i}/{len(tasks)} replicates, {time.time()-started:.1f} s", flush=True)
    raw = pd.DataFrame(raw)
    raw.to_csv(ROOT / "budget_raw.csv", index=False)
    pd.DataFrame(candidates).to_csv(ROOT / "budget_candidates.csv", index=False)
    pd.DataFrame(ledger).to_csv(ROOT / "budget_cost_ledger.csv", index=False)
    metadata = dict(replicates_per_design=args.reps, scenarios=SCENARIOS, seeds=seeds,
                    E=E_GRID, rounds=ROUNDS, rate_multipliers=RATE_MULTIPLIERS.tolist(),
                    lambda_rule="mean of locally validation-MSE-selected lambdas weighted by training size; common lambda in every arm",
                    rate_rule="minimum weighted validation MSE independently at each checkpoint, ties to smaller rate",
                    activity_epsilon=EPS, python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                    wall_seconds=time.time()-started, workers=args.workers, blas_threads=1,
                    cost_units="bytes of double-precision uploads; excludes downloads and transport overhead",
                    objective_diagnostics="computed offline; no diagnostic communication included in fitting cost",
                    pooled_reference="centralized diagnostic, not assigned zero federated communication",
                    study_reuse="paths reused across checkpoints; standalone costs are alternative independent fits and must not be summed",
                    delivered_source_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                                   for f in ["feddualavg.py", "budget_study.py", "fedlasso.py", "run_sim.py"]})
    (ROOT / "budget_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_outputs(raw)


if __name__ == "__main__":
    main()
