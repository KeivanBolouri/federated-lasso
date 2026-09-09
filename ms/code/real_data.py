"""Illustration on the three-site data set accompanying the paper."""
import numpy as np, pandas as pd
from fedlasso import (fedavg, consensus_admm, local_solve, tune_lambda,
                      tune_lambda_1se, soft, objective, select_threshold,
                      consensus_path_exact, select_by_val)

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

lam_p1, _, _ = tune_lambda_1se(Xp, yp, Xpv, ypv)
lam_l1, _, _ = tune_lambda_1se(*sites[0], *val[0])
rows = [dict(method="Pooled", E=np.nan, rounds=np.nan, epochs=np.nan, **agree(pooled)),
        dict(method="Pooled-1SE", E=np.nan, rounds=np.nan, epochs=np.nan,
             **agree(local_solve(Xp, yp, lam_p1))),
        dict(method="Local-only", E=np.nan, rounds=0, epochs=np.nan,
             **agree(local_solve(*sites[0], lams[0]))),
        dict(method="Local-only-1SE", E=np.nan, rounds=0, epochs=np.nan,
             **agree(local_solve(*sites[0], lam_l1)))]
locs = [local_solve(X, y, lj) for (X, y), lj in zip(sites, lams)]
rows.append(dict(method="One-shot", E=np.nan, rounds=1, epochs=np.nan,
                 **agree(sum(wj * b for wj, b in zip(w, locs)))))
z, it = consensus_admm(sites, w, lam_bar, P)
rows.append(dict(method="ADMM", E=np.nan, rounds=it, epochs=np.nan, **agree(z)))
agrid = np.logspace(np.log10(lam_bar * 4), np.log10(lam_bar / 8), 15)
apath = consensus_path_exact(Xp, yp, agrid)
for rule, nm in [("min", "ADMM-CV"), ("1se", "ADMM-CV-1SE")]:
    k = select_by_val(apath, val, w, rule)
    rows.append(dict(method=nm, E=np.nan, rounds=np.nan, epochs=np.nan,
                     lam_sel=agrid[k], **agree(apath[:, k])))
tau_grid = np.concatenate([[0.0], np.logspace(np.log10(lam_bar*1e-3), np.log10(lam_bar*5), 39)])
st_scalar_kib = 3 * len(tau_grid) * 3 * 8 / 1024
for E in (1, 2, 3, 5, 8, 10, 15, 20):
    b, r, hist, _ = fedavg(sites, w, lams, E, P)
    rows.append(dict(method="FedAvg", E=E, rounds=r, epochs=r*E, **agree(b)))
    for rule, nm in [("1se", "FedAvg-ST"), ("min", "FedAvg-ST-min")]:
        tau, _, _ = select_threshold(b, val, w, tau_grid, rule=rule)
        rows.append(dict(method=nm, E=E, rounds=r, epochs=r*E, tau=tau,
                         scalar_kib=st_scalar_kib, **agree(soft(b, tau))))
    pgrid = lam_bar * np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    sols, rds = [], []
    for tp in pgrid:
        bp, rp, _, _ = fedavg(sites, w, lams, E, P, tau=tp)
        sols.append(bp); rds.append(rp)
    M = np.array(sols).T
    kp = select_by_val(M, val, w, "1se")
    rows.append(dict(method="FedAvg-P", E=E, rounds=rds[kp], epochs=rds[kp]*E,
                     tau=pgrid[kp], scalar_kib=3*len(pgrid)*3*8/1024,
                     **agree(M[:, kp])))
d = pd.DataFrame(rows)
d["scalar_kib"] = d.get("scalar_kib", 0.0)
d["scalar_kib"] = d["scalar_kib"].fillna(0.0)
d["vector_kib"] = d["rounds"] * 3 * P * 8 / 1024
d["upload_kib"] = d["vector_kib"] + d["scalar_kib"]
d.to_csv("real_data_results.csv", index=False)
print(d.round(3).to_string(index=False))
