"""Illustration on the three-site data set accompanying the paper."""
import numpy as np, pandas as pd
from fedlasso import (fedavg, consensus_admm, local_solve, tune_lambda, soft,
                      objective, select_threshold)

R = "/home/claude/work/repo/"
# files are stored in reverse size order; relabel so site 1 is the largest
order = [3, 2, 1]
sites, val = [], []
for f in order:
    X = pd.read_csv(f"{R}node{f}_X_train.csv").to_numpy(float)
    y = pd.read_csv(f"{R}node{f}_y_train.csv").to_numpy(float).ravel()
    Xv = pd.read_csv(f"{R}node{f}_X_val.csv").to_numpy(float)
    yv = pd.read_csv(f"{R}node{f}_y_val.csv").to_numpy(float).ravel()
    sites.append((X, y)); val.append((Xv, yv))
P = sites[0][0].shape[1]
m = np.array([len(y) for _, y in sites]); w = m / m.sum()
print("sites", m, "p", P)

lams = [tune_lambda(X, y, Xv, yv)[0] for (X, y), (Xv, yv) in zip(sites, val)]
lam_bar = float(np.dot(w, lams))
Xp = np.vstack([X for X, _ in sites]); yp = np.concatenate([y for _, y in sites])
Xpv = np.vstack([X for X, _ in val]); ypv = np.concatenate([y for _, y in val])
lam_p, _, _ = tune_lambda(Xp, yp, Xpv, ypv)
pooled = local_solve(Xp, yp, lam_p)
EPS = 1e-4
ref = np.abs(pooled) > EPS
print("lams", np.round(lams, 4), "lam_bar", round(lam_bar, 4),
      "lam_pooled", round(lam_p, 4), "pooled active", int(ref.sum()))

def agree(b):
    s = np.abs(b) > EPS
    tp = int((s & ref).sum()); fp = int((s & ~ref).sum()); fn = int((~s & ref).sum())
    pr = tp / (tp + fp) if tp + fp else np.nan
    rc = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * pr * rc / (pr + rc) if pr and rc else 0.0
    return dict(active=int(s.sum()), TP=tp, FP=fp, FN=fn, precision=pr,
                recall=rc, F1=f1, jaccard=tp / (tp + fp + fn),
                val_mse=float(np.mean((ypv - Xpv @ b) ** 2)),
                obj=objective(sites, b, w, lams))

rows = [dict(method="Pooled", E=np.nan, rounds=np.nan, epochs=np.nan, **agree(pooled)),
        dict(method="Local-only", E=np.nan, rounds=0, epochs=np.nan,
             **agree(local_solve(*sites[0], lams[0])))]
locs = [local_solve(X, y, lj) for (X, y), lj in zip(sites, lams)]
rows.append(dict(method="One-shot", E=np.nan, rounds=1, epochs=np.nan,
                 **agree(sum(wj * b for wj, b in zip(w, locs)))))
z, it = consensus_admm(sites, w, lam_bar, P)
rows.append(dict(method="ADMM", E=np.nan, rounds=it, epochs=np.nan, **agree(z)))
tau_grid = np.concatenate([[0.0], np.logspace(np.log10(lam_bar*1e-3), np.log10(lam_bar*5), 39)])
for E in (1, 2, 3, 5, 8, 10, 15, 20):
    b, r, hist, _ = fedavg(sites, w, lams, E, P)
    rows.append(dict(method="FedAvg", E=E, rounds=r, epochs=r*E, **agree(b)))
    tau, _, _ = select_threshold(b, val, w, tau_grid)
    rows.append(dict(method="FedAvg-ST", E=E, rounds=r+1, epochs=r*E, tau=tau,
                     **agree(soft(b, tau))))
d = pd.DataFrame(rows)
d["upload_kib"] = d["rounds"] * 3 * P * 8 / 1024
d.to_csv("real_data_results.csv", index=False)
print(d.round(3).to_string(index=False))
