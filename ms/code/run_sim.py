"""Monte Carlo study for the federated Lasso manuscript."""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from fedlasso import (fedavg, consensus_admm, local_solve, tune_lambda,
                      tune_lambda_1se, soft, objective, lam_grid,
                      select_threshold, consensus_path_exact, select_by_val)

P, S = 600, 30
MTR = (400, 240, 160)
MVA = (100, 60, 40)
NTEST = 2000
E_GRID = (1, 2, 3, 5, 8, 10, 15, 20)
EPS = 1e-4


def ar1(p, rho, rng, n):
    """Draw n rows of an AR(1)(rho) Gaussian design without forming Sigma."""
    Z = rng.normal(size=(n, p))
    if rho == 0:
        return Z
    X = np.empty((n, p))
    X[:, 0] = Z[:, 0]
    c = np.sqrt(1 - rho ** 2)
    for k in range(1, p):
        X[:, k] = rho * X[:, k - 1] + c * Z[:, k]
    return X


def make_data(scenario, rng):
    beta = np.zeros(P)
    idx = rng.choice(P, S, replace=False)
    beta[idx] = rng.choice([-1.0, 1.0], S) * rng.uniform(0.4, 1.2, S)
    cfg = dict(IID=dict(rho=0.0, het=False),
               Correlated=dict(rho=0.5, het=False),
               Heterogeneous=dict(rho=0.3, het=True))[scenario]
    rho, het = cfg["rho"], cfg["het"]
    snr = 1.0
    sites, val, sig = [], [], []
    for j, (mtr, mva) in enumerate(zip(MTR, MVA)):
        r = rho if not het else [0.0, 0.4, 0.7][j]
        sc = 1.0 if not het else [1.0, 1.4, 0.7][j]
        X = ar1(P, r, rng, mtr) * sc
        Xv = ar1(P, r, rng, mva) * sc
        sd = np.sqrt(np.var(X @ beta) / snr) * (1.0 if not het else [1.0, 1.5, 0.8][j])
        sites.append((X, X @ beta + rng.normal(scale=sd, size=mtr)))
        val.append((Xv, Xv @ beta + rng.normal(scale=sd, size=mva)))
        sig.append(sd)
    Xt = ar1(P, rho if not het else 0.4, rng, NTEST)
    test = (Xt, Xt @ beta + rng.normal(scale=np.mean(sig), size=NTEST))
    return beta, sites, val, test


def support_metrics(bhat, beta, eps=EPS):
    # Empty selected/reference sets receive zero precision, recall, F1 and Jaccard.
    if not np.isfinite(bhat).all():
        raise ValueError("Non-finite coefficient estimate")
    sh = np.abs(bhat) > eps
    st = beta != 0
    tp = int(np.sum(sh & st)); fp = int(np.sum(sh & ~st))
    fn = int(np.sum(~sh & st)); tn = int(np.sum(~sh & ~st))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    jac = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(active=int(sh.sum()), TP=tp, FP=fp, FN=fn, TN=tn,
                precision=prec, recall=rec, F1=f1, jaccard=jac)


def row(method, E, bhat, beta, test, rounds, epochs, scalar_kib=0.0,
        total_fit_rounds=None, total_fit_epochs=None, candidate_count=1, **extra):
    """Record training traffic including every fitted threshold candidate.

    selected_* records the retained candidate alone; tuning_* records discarded
    candidate fits. Local penalty-path work is not included in CD epoch counts.
    Scalar validation is an additional round. Downloads are excluded throughout.
    """
    Xt, yt = test
    d = support_metrics(bhat, beta)
    selected_rounds, selected_epochs = rounds, epochs
    rounds = rounds if total_fit_rounds is None else total_fit_rounds
    epochs = epochs if total_fit_epochs is None else total_fit_epochs
    bytes_per_round = len(MTR) * len(beta) * 8 / 1024
    vec_kib = rounds * bytes_per_round
    selected_vector = selected_rounds * bytes_per_round
    scalar_rounds = int(scalar_kib > 0)
    d.update(method=method, E=E, rounds=rounds, local_epochs=epochs,
             selected_rounds=selected_rounds, selected_local_epochs=selected_epochs,
             selected_vector_kib=selected_vector,
             tuning_rounds=rounds-selected_rounds,
             tuning_local_epochs=epochs-selected_epochs,
             tuning_vector_kib=vec_kib-selected_vector,
             candidate_count=candidate_count, scalar_rounds=scalar_rounds,
             total_rounds=rounds+scalar_rounds,
             vector_kib=vec_kib, scalar_kib=scalar_kib,
             upload_kib=vec_kib+scalar_kib,
             coef_err=float(np.linalg.norm(bhat - beta)),
             test_mse=float(np.mean((yt - Xt @ bhat) ** 2)))
    d.update(extra)
    return d


def one_rep(scenario, seed):
    rng = np.random.default_rng(seed)
    beta, sites, val, test = make_data(scenario, rng)
    w = np.array(MTR) / sum(MTR)
    lams = [tune_lambda(X, y, Xv, yv)[0] for (X, y), (Xv, yv) in zip(sites, val)]
    lam_bar = float(np.dot(w, lams))
    lams_1se = [tune_lambda_1se(*a, *b)[0] for a, b in zip(sites, val)]
    lam_bar_1se = float(np.dot(w, lams_1se))
    out = []

    # ---- centralised references, both tuning rules -------------------------
    Xp = np.vstack([X for X, _ in sites]); yp = np.concatenate([y for _, y in sites])
    Xpv = np.vstack([X for X, _ in val]); ypv = np.concatenate([y for _, y in val])
    lam_p, _, _ = tune_lambda(Xp, yp, Xpv, ypv)
    lam_p1, _, _ = tune_lambda_1se(Xp, yp, Xpv, ypv)
    pooled, pooled_diag = local_solve(Xp, yp, lam_p, return_diagnostics=True)
    pooled1, pooled1_diag = local_solve(Xp, yp, lam_p1, return_diagnostics=True)
    out.append(row("Pooled", np.nan, pooled, beta, test,
                   np.nan, np.nan, lam=lam_p, **pooled_diag))
    out.append(row("Pooled-1SE", np.nan, pooled1, beta, test,
                   np.nan, np.nan, lam=lam_p1, **pooled1_diag))

    # ---- local fits shared by local-only and one-shot baselines --------------
    solved = [local_solve(X, y, lj, return_diagnostics=True)
              for (X, y), lj in zip(sites, lams)]
    solved1 = [local_solve(X, y, lj, return_diagnostics=True)
               for (X, y), lj in zip(sites, lams_1se)]
    out.append(row("Local-only", np.nan, solved[0][0], beta, test, 0, np.nan,
                   lam=lams[0], **solved[0][1]))
    out.append(row("Local-only-1SE", np.nan, solved1[0][0], beta, test, 0, np.nan,
                   lam=lams_1se[0], **solved1[0][1]))
    for nm, fits, penalty in [("One-shot", solved, lam_bar),
                              ("One-shot-1SE", solved1, lam_bar_1se)]:
        diagnostics = [diag for _, diag in fits]
        out.append(row(nm, np.nan, sum(wj * b for wj, (b, _) in zip(w, fits)),
                       beta, test, 1, np.nan, lam=penalty,
                       fit_converged=all(d["fit_converged"] for d in diagnostics),
                       solver_restarts=sum(d["solver_restarts"] for d in diagnostics),
                       max_local_dual_gap_ratio=max(d["local_dual_gap"] /
                           max(d["local_dual_gap_tolerance"], np.finfo(float).tiny)
                           for d in diagnostics),
                       local_fit_iterations=json.dumps([d["solver_iterations"] for d in diagnostics]),
                       local_fit_dual_gaps=json.dumps([d["local_dual_gap"] for d in diagnostics])))

    # ---- consensus ADMM: fixed penalty and validation-tuned penalty ----------
    z, it, ad = consensus_admm(sites, w, lam_bar, P, return_diagnostics=True)
    out.append(row("ADMM", np.nan, z, beta, test, it, np.nan, lam=lam_bar,
                   final_obj=objective(sites, z, w, lams), **ad))
    agrid = np.logspace(np.log10(lam_bar * 4), np.log10(lam_bar / 8), 15)
    apath = consensus_path_exact(Xp, yp, agrid)
    for rule, nm in [("min", "ADMM-CV"), ("1se", "ADMM-CV-1SE")]:
        k = select_by_val(apath, val, w, rule)
        out.append(row(nm, np.nan, apath[:, k], beta, test, np.nan, np.nan,
                       lam=agrid[k]))

    # ---- federated schedules --------------------------------------------------
    tau_grid = np.concatenate([[0.0], np.logspace(np.log10(lam_bar * 1e-3),
                                                  np.log10(lam_bar * 5), 39)])
    st_scalars = len(sites) * len(tau_grid) * 3 * 8 / 1024   # 3 scalars per site
    for E in E_GRID:
        b, r, hist, _, fd = fedavg(sites, w, lams, E, P, return_diagnostics=True)
        out.append(row("FedAvg", E, b, beta, test, r, r * E,
                       final_obj=hist[-1], lam=lam_bar, **fd))
        b1, r1, h1, _, d1 = fedavg(sites, w, lams_1se, E, P, return_diagnostics=True)
        out.append(row("FedAvg-1SE", E, b1, beta, test, r1, r1 * E,
                       final_obj=h1[-1], lam=lam_bar_1se, **d1))
        for rule, nm in [("1se", "FedAvg-ST"), ("min", "FedAvg-ST-min")]:
            tau, _, _ = select_threshold(b, val, w, tau_grid, rule=rule)
            bst = soft(b, tau)
            out.append(row(nm, E, bst, beta, test, r, r * E,
                           scalar_kib=st_scalars,
                           final_obj=objective(sites, bst, w, lams),
                           lam=lam_bar, tau=tau, **fd))
        # per-round proximal aggregation, threshold tuned by federated validation
        pgrid = lam_bar * np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        sols, rds, diagnostics = [], [], []
        for tp in pgrid:
            bp, rp, hp, _, dp = fedavg(sites, w, lams, E, P, tau=tp,
                                       return_diagnostics=True)
            sols.append(bp); rds.append(rp); diagnostics.append(dp)
        M = np.array(sols).T
        kp = select_by_val(M, val, w, "1se")
        out.append(row("FedAvg-P", E, M[:, kp], beta, test, rds[kp], rds[kp] * E,
                       scalar_kib=len(sites) * len(pgrid) * 3 * 8 / 1024,
                       final_obj=objective(sites, M[:, kp], w, lams),
                       lam=lam_bar, tau=pgrid[kp], total_fit_rounds=sum(rds),
                       total_fit_epochs=sum(rds) * E, candidate_count=len(pgrid),
                       candidate_rounds=json.dumps(rds),
                       all_candidates_converged=all(d["fit_converged"] for d in diagnostics),
                       **diagnostics[kp]))

    for d in out:
        d.update(scenario="Independent" if scenario == "IID" else scenario, rep=seed)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=["IID", "Correlated", "Heterogeneous"])
    parser.add_argument("reps", type=int)
    parser.add_argument("seed0", type=int)
    parser.add_argument("tag", nargs="?", default="run",
                        help="Legacy output tag, used when --output is omitted")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("reps must be positive")
    output = args.output or Path(__file__).resolve().parent / f"out_{args.tag}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i in range(args.reps):
        rows += one_rep(args.scenario, args.seed0 + i)
        if i == 0 or (i + 1) % 5 == 0:
            print(f"{args.scenario} rep {i+1}/{args.reps} {time.time()-t0:.1f}s", flush=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(output)
    print("wrote", output, f"in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
