"""Illustrate the manuscript's orthogonal-design proposition and FP corollary.

Run from the repository root:
    python code/orthogonal_check.py --reps 1000 --seed 20260911

This is a separate, fixed-penalty, global-null experiment. It does not modify
the high-dimensional study or use its positive support threshold. Each K has
fixed centred orthogonal design matrices and independent Gaussian responses.
Every replicate runs the existing fedavg implementation at E=1,3,20, and the
existing local_epochs implementation also checks the pooled solution. MCSEs
use the sample SD of replicate-level false-positive fractions divided by
sqrt(reps); coordinates are not treated as independent simulation replicates.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import norm

from fedlasso import fedavg, local_epochs, soft


TOTAL_N, P, SIGMA, LAMBDA = 240, 12, 1.0, 0.15
K_GRID, E_GRID = (2, 3, 6), (1, 3, 20)
SOLVER_TOL, CHECK_TOL = 1e-10, 5e-12


def orthogonal_designs(k, seed):
    """A fixed full-rank design orthogonal to the intercept at every site."""
    n = TOTAL_N // k
    if TOTAL_N % k or P > n - 1:
        raise ValueError("Balanced centred orthogonal designs are infeasible")
    rng = np.random.default_rng(np.random.SeedSequence([seed, k, 0]))
    designs = []
    for _ in range(k):
        a = np.column_stack((np.ones(n), rng.normal(size=(n, P))))
        q, _ = np.linalg.qr(a, mode="reduced")
        designs.append(np.asfortranarray(np.sqrt(n) * q[:, 1:]))
    return designs


def run(reps, seed):
    raw, summaries = [], []
    numerical = dict(max_gram_error=0.0, max_column_mean=0.0,
                     max_epoch_difference=0.0, max_fedavg_formula_error=0.0,
                     max_pooled_formula_error=0.0,
                     max_warm_start_local_formula_error=0.0,
                     max_rounds=0, support_mismatches=0,
                     pooled_inclusion_violations=0,
                     warm_start_local_calls=0)
    for k in K_GRID:
        n = TOTAL_N // k
        designs = orthogonal_designs(k, seed)
        xp = np.vstack(designs)
        weights = np.full(k, 1.0 / k)
        penalties = np.full(k, LAMBDA)
        for x in designs:
            numerical["max_gram_error"] = max(numerical["max_gram_error"],
                float(np.max(np.abs(x.T @ x / n - np.eye(P)))))
            numerical["max_column_mean"] = max(numerical["max_column_mean"],
                float(np.max(np.abs(x.mean(axis=0)))))
        if max(numerical["max_gram_error"], numerical["max_column_mean"]) > CHECK_TOL:
            raise AssertionError("The generated design is not centred and orthogonal")

        for rep in range(reps):
            rng = np.random.default_rng(np.random.SeedSequence([seed, k, 1, rep]))
            sites = [(x, rng.normal(scale=SIGMA, size=n)) for x in designs]
            z = np.asarray([x.T @ y / n for x, y in sites])
            local_formula = soft(z, LAMBDA)
            average_formula = weights @ local_formula
            pooled_formula = soft(weights @ z, LAMBDA)
            union = np.any(local_formula != 0.0, axis=0)
            fits, rounds = [], []
            for epochs in E_GRID:
                fit, r, _, _, diag = fedavg(
                    sites, weights, penalties, epochs, P, tol=SOLVER_TOL,
                    max_rounds=3, return_diagnostics=True)
                if not diag["fit_converged"] or r > 2:
                    raise AssertionError("Orthogonal FedAvg did not stop by round two")
                fits.append(fit)
                rounds.append(r)
            yp = np.concatenate([y for _, y in sites])
            pooled = local_epochs(xp, yp, np.zeros(P), LAMBDA, 1)
            epoch_error = max(float(np.max(np.abs(b - fits[0]))) for b in fits)
            formula_error = max(float(np.max(np.abs(b - average_formula))) for b in fits)
            pooled_error = float(np.max(np.abs(pooled - pooled_formula)))
            mismatches = sum(int(np.count_nonzero((b != 0.0) != union)) for b in fits)
            mismatches += int(np.count_nonzero((pooled != 0.0) != (pooled_formula != 0.0)))
            inclusion = int(np.count_nonzero((pooled != 0.0) & ~union))
            if max(epoch_error, formula_error, pooled_error) > CHECK_TOL:
                raise AssertionError("Numerical solution differs from the orthogonal formula")
            if mismatches or inclusion:
                raise AssertionError("Exact-support relation failed")

            # Check Proposition 2(i) from nonzero starts on the first 20
            # replicates of each K, independently of the response RNG stream.
            if rep < min(reps, 20):
                wrng = np.random.default_rng(np.random.SeedSequence([seed, k, 2, rep]))
                for j, (x, y) in enumerate(sites):
                    start = wrng.normal(size=P)
                    for epochs in E_GRID:
                        b = local_epochs(x, y, start, LAMBDA, epochs)
                        err = float(np.max(np.abs(b - local_formula[j])))
                        numerical["max_warm_start_local_formula_error"] = max(
                            numerical["max_warm_start_local_formula_error"], err)
                        numerical["warm_start_local_calls"] += 1
                        if err > CHECK_TOL:
                            raise AssertionError("Local solution depends on the starting vector")

            numerical["max_epoch_difference"] = max(numerical["max_epoch_difference"], epoch_error)
            numerical["max_fedavg_formula_error"] = max(numerical["max_fedavg_formula_error"], formula_error)
            numerical["max_pooled_formula_error"] = max(numerical["max_pooled_formula_error"], pooled_error)
            numerical["max_rounds"] = max(numerical["max_rounds"], max(rounds))
            numerical["support_mismatches"] += mismatches
            numerical["pooled_inclusion_violations"] += inclusion
            raw.append(dict(K=k, rep=rep, seed=seed, n_site=n,
                fedavg_FP=int(np.count_nonzero(fits[0])),
                pooled_FP=int(np.count_nonzero(pooled)),
                fedavg_FP_fraction=float(np.mean(fits[0] != 0.0)),
                pooled_FP_fraction=float(np.mean(pooled != 0.0)),
                max_epoch_difference=epoch_error,
                max_fedavg_formula_error=formula_error,
                pooled_formula_error=pooled_error,
                rounds_E1=rounds[0], rounds_E3=rounds[1], rounds_E20=rounds[2]))

        block = pd.DataFrame(raw[-reps:])
        q = 2.0 * norm.sf(LAMBDA * np.sqrt(n) / SIGMA)
        theory_fedavg = 1.0 - (1.0 - q) ** k
        theory_pooled = 2.0 * norm.sf(LAMBDA * np.sqrt(TOTAL_N) / SIGMA)
        row = dict(K=k, n_site=n, reps=reps, p=P, total_n=TOTAL_N,
                   lambda_fixed=LAMBDA, sigma=SIGMA,
                   fedavg_FP_probability_theory=float(theory_fedavg),
                   pooled_FP_probability_theory=float(theory_pooled))
        for name, theory in (("fedavg", theory_fedavg), ("pooled", theory_pooled)):
            values = block[f"{name}_FP_fraction"]
            mean, mcse = float(values.mean()), float(values.std(ddof=1) / np.sqrt(reps))
            row.update({f"{name}_FP_fraction_mean": mean,
                        f"{name}_FP_fraction_mcse": mcse,
                        f"{name}_difference_in_mcse": (mean - theory) / mcse if mcse else None})
        summaries.append(row)
        print(f"K={k}: completed {reps} independent replicates", flush=True)
    return pd.DataFrame(raw), pd.DataFrame(summaries), numerical


def write_tex(root, summary, numerical, reps):
    lines = [r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{False-positive probability in centred orthogonal designs. "
        r"The simulation columns report the mean fraction of the 12 null "
        r"coordinates selected, with replicate-level Monte Carlo standard "
        r"errors in parentheses. FedAvg uses $E=1$; $E=3,20$ select the same "
        r"coordinates in every replicate.}",
        r"\label{tab:orthogonal-check}", r"\begin{tabular}{rrcccc}", r"\toprule",
        r"& & \multicolumn{2}{c}{FedAvg} & \multicolumn{2}{c}{Pooled Lasso} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"$K$ & $m_j$ & Theory & Simulation (MCSE) & Theory & Simulation (MCSE) \\",
        r"\midrule"]
    for row in summary.to_dict("records"):
        lines.append(f"{row['K']} & {row['n_site']} & "
            f"{row['fedavg_FP_probability_theory']:.4f} & "
            f"{row['fedavg_FP_fraction_mean']:.4f} ({row['fedavg_FP_fraction_mcse']:.4f}) & "
            f"{row['pooled_FP_probability_theory']:.4f} & "
            f"{row['pooled_FP_fraction_mean']:.4f} ({row['pooled_FP_fraction_mcse']:.4f}) " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "tables" / "orthogonal_check.tex").write_text("\n".join(lines) + "\n")
    max_error = max(numerical["max_epoch_difference"], numerical["max_fedavg_formula_error"],
                    numerical["max_pooled_formula_error"],
                    numerical["max_warm_start_local_formula_error"])
    # A fixed conservative bound, verified above, avoids meaningless digits
    # from machine-precision differences in the submitted prose.
    if max_error >= 1e-12:
        raise AssertionError("Update the stated numerical bound before producing the section")
    section = r"""\section*{Orthogonal-design illustration}
This separate experiment illustrates Proposition~2 and Corollary~1 of the
main manuscript under their stated assumptions. We fix $m=240$, $p=12$,
$\beta^\star=0$, $\sigma=1$ and the common penalty $\lambda=0.15$, with
balanced partitions $K\in\{2,3,6\}$. At each site, QR decomposition of an
intercept column and Gaussian columns gives a fixed centred design with
$X_j^\top X_j=m_j I_p$; here $p<m_j$. For each $K$, independent Gaussian
responses $y_j\sim N(0,I_{m_j})$ generate REPS replicates conditional on
those designs. No penalty tuning is performed.

Table~\ref{tab:orthogonal-check} compares the corollary's probabilities
with the mean fraction of null coordinates selected. Monte Carlo standard
errors are the sample standard deviations of the replicate-level
fractions divided by $\sqrt{REPSMATH}$. Support means an exactly non-zero
coefficient, as in the corollary, rather than the positive threshold used
in the main simulations. The theoretical false-positive probability
increases with $K$, whereas the pooled probability remains constant.

The existing coordinate-descent implementation gives the same support at
$E=1,3,20$ in every replicate and terminates within two rounds. Maximum
coefficient differences across these schedules, from the closed-form
solutions, and in additional non-zero-start checks are below $10^{-12}$.
This connects the theory to an experiment satisfying its hypotheses; it
does not extend the corollary to tuned penalties or nonorthogonal designs.
\input{tables/orthogonal_check}
"""
    section = section.replace("REPSMATH", str(reps)).replace("REPS", f"{reps:,}")
    (root / "sections").mkdir(parents=True, exist_ok=True)
    (root / "sections" / "orthogonal_check.tex").write_text(section)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    if args.reps < 2 or args.seed < 0:
        parser.error("reps must be at least 2 and seed must be nonnegative")
    raw, summary, numerical = run(args.reps, args.seed)
    root = args.output_root.resolve()
    data = root / "code"
    data.mkdir(parents=True, exist_ok=True)
    raw.to_csv(data / "orthogonal_raw.csv", index=False, float_format="%.17g")
    summary.to_csv(data / "orthogonal_summary.csv", index=False, float_format="%.17g")
    script = Path(__file__).resolve()
    metadata = dict(total_n=TOTAL_N, p=P, beta_true=[0.0] * P, sigma=SIGMA,
        lambda_fixed=LAMBDA, K_grid=list(K_GRID), E_grid=list(E_GRID),
        replicates_per_K=args.reps, root_seed=args.seed,
        random_streams="SeedSequence([seed,K,0]) for fixed designs; "
            "[seed,K,1,rep] for noise; [seed,K,2,rep] for warm starts",
        model="Fixed centred orthogonal local X; independent N(0,I) responses; global null",
        support_definition="coefficient != 0.0 (no positive threshold)",
        mcse="sample SD of replicate-level fraction / sqrt(replicates), ddof=1",
        numerical_checks=numerical,
        versions=dict(python=platform.python_version(), numpy=np.__version__,
            pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (script, script.with_name("fedlasso.py"))},
        rerun_command=f"python code/orthogonal_check.py --reps {args.reps} --seed {args.seed}")
    (data / "orthogonal_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_tex(root, summary, numerical, args.reps)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(numerical, indent=2), flush=True)


if __name__ == "__main__":
    main()
