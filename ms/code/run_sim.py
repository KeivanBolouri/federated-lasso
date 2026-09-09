"""Monte Carlo study for the federated Lasso manuscript."""
import sys, time, json
import numpy as np
import pandas as pd
from fedlasso import (fedavg, consensus_admm, local_solve, tune_lambda,
                      soft, objective, lam_grid, select_threshold)

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
    sh = np.abs(bhat) > eps
    st = beta != 0
    tp = int(np.sum(sh & st)); fp = int(np.sum(sh & ~st))
    fn = int(np.sum(~sh & st)); tn = int(np.sum(~sh & ~st))
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec > 0 else 0.0
    jac = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    return dict(active=int(sh.sum()), TP=tp, FP=fp, FN=fn, TN=tn,
                precision=prec, recall=rec, F1=f1, jaccard=jac)


def row(method, E, bhat, beta, test, rounds, epochs, **extra):
    Xt, yt = test
    d = support_metrics(bhat, beta)
    d.update(method=method, E=E, rounds=rounds, local_epochs=epochs,
             upload_kib=rounds * 3 * P * 8 / 1024,
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
    out = []

    # pooled (centralised oracle)
    Xp = np.vstack([X for X, _ in sites]); yp = np.concatenate([y for _, y in sites])
    Xpv = np.vstack([X for X, _ in val]); ypv = np.concatenate([y for _, y in val])
    lam_p, _, _ = tune_lambda(Xp, yp, Xpv, ypv)
    out.append(row("Pooled", np.nan, local_solve(Xp, yp, lam_p), beta, test,
                   np.nan, np.nan, lam=lam_p))
    # local only (largest site)
    out.append(row("Local-only", np.nan, local_solve(*sites[0], lams[0]), beta,
                   test, 0, np.nan, lam=lams[0]))
    # one-shot averaging
    locs = [local_solve(X, y, lj) for (X, y), lj in zip(sites, lams)]
    out.append(row("One-shot", np.nan, sum(wj * b for wj, b in zip(w, locs)),
                   beta, test, 1, np.nan, lam=lam_bar))
    # consensus ADMM: exact minimiser of the aggregated objective
    z, it = consensus_admm(sites, w, lam_bar, P)
    out.append(row("ADMM", np.nan, z, beta, test, it, np.nan, lam=lam_bar))

    tau_grid = np.concatenate([[0.0], np.logspace(np.log10(lam_bar * 1e-3), np.log10(lam_bar * 5), 39)])
    for E in E_GRID:
        b, r, hist, _ = fedavg(sites, w, lams, E, P)
        out.append(row("FedAvg", E, b, beta, test, r, r * E,
                       final_obj=hist[-1], lam=lam_bar))
        tau, _, _ = select_threshold(b, val, w, tau_grid)
        bst = soft(b, tau)
        out.append(row("FedAvg-ST", E, bst, beta, test, r + 1, r * E,
                       final_obj=objective(sites, bst, w, lams), lam=lam_bar,
                       tau=tau))
    for d in out:
        d.update(scenario=scenario, rep=seed)
    return out


if __name__ == "__main__":
    scen, reps, seed0, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rows, t0 = [], time.time()
    for i in range(reps):
        rows += one_rep(scen, seed0 + i)
        if i == 0 or (i + 1) % 5 == 0:
            print(f"{scen} rep {i+1}/{reps}  {time.time()-t0:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"out_{tag}.csv", index=False)
    print("wrote", f"out_{tag}.csv", time.time() - t0)
