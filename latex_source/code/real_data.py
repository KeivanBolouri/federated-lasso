"""Reproduce the fixed synthetic three-site benchmark (legacy file name).

The supplied node CSVs are synthetic. The original generator, seed and
coefficient vector were not supplied; these files
are therefore a fixed benchmark, not an observational-data application.
"""
from pathlib import Path
import hashlib
import json
import numpy as np, pandas as pd
from fedlasso import (fedavg, consensus_admm, local_solve, tune_lambda,
                      tune_lambda_1se, soft, objective, select_threshold,
                      consensus_path_exact, select_by_val)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "data"
# files are stored in reverse size order; relabel so site 1 is the largest
order = [3, 2, 1]
sites, val = [], []
for f in order:
    X = pd.read_csv(ROOT / f"node{f}_X_train.csv").to_numpy(float)
    y = pd.read_csv(ROOT / f"node{f}_y_train.csv").to_numpy(float).ravel()
    Xv = pd.read_csv(ROOT / f"node{f}_X_val.csv").to_numpy(float)
    yv = pd.read_csv(ROOT / f"node{f}_y_val.csv").to_numpy(float).ravel()
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
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return dict(active=int(s.sum()), exact_zeros=int((b == 0).sum()),
                TP=tp, FP=fp, FN=fn, precision=pr,
                recall=rc, F1=f1, jaccard=tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                val_mse=float(np.mean((ypv - Xpv @ b) ** 2)),
                obj=objective(sites, b, w, lams))

lam_p1, _, _ = tune_lambda_1se(Xp, yp, Xpv, ypv)
lam_l1, _, _ = tune_lambda_1se(*sites[0], *val[0])
rows = [dict(method="Pooled", E=np.nan, rounds=np.nan, epochs=np.nan,
             lam=lam_p, **agree(pooled)),
        dict(method="Pooled-1SE", E=np.nan, rounds=np.nan, epochs=np.nan, lam=lam_p1,
             **agree(local_solve(Xp, yp, lam_p1))),
        dict(method="Local-only", E=np.nan, rounds=0, epochs=np.nan, lam=lams[0],
             **agree(local_solve(*sites[0], lams[0]))),
        dict(method="Local-only-1SE", E=np.nan, rounds=0, epochs=np.nan, lam=lam_l1,
             **agree(local_solve(*sites[0], lam_l1)))]
locs = [local_solve(X, y, lj) for (X, y), lj in zip(sites, lams)]
rows.append(dict(method="One-shot", E=np.nan, rounds=1, epochs=np.nan,
                 **agree(sum(wj * b for wj, b in zip(w, locs)))))
z, it, admm_diag = consensus_admm(sites, w, lam_bar, P, return_diagnostics=True)
if not admm_diag["fit_converged"]:
    raise RuntimeError(f"ADMM did not converge: {admm_diag}")
rows.append(dict(method="ADMM", E=np.nan, rounds=it, epochs=np.nan,
                 **admm_diag, **agree(z)))
agrid = np.logspace(np.log10(lam_bar * 4), np.log10(lam_bar / 8), 15)
apath = consensus_path_exact(Xp, yp, agrid)
for rule, nm in [("min", "ADMM-CV"), ("1se", "ADMM-CV-1SE")]:
    k = select_by_val(apath, val, w, rule)
    rows.append(dict(method=nm, E=np.nan, rounds=np.nan, epochs=np.nan,
                     lam_sel=agrid[k], lam=agrid[k], **agree(apath[:, k])))
tau_grid = np.concatenate([[0.0], np.logspace(np.log10(lam_bar*1e-3), np.log10(lam_bar*5), 39)])
st_scalar_kib = 3 * len(tau_grid) * 3 * 8 / 1024
for E in (1, 2, 3, 5, 8, 10, 15, 20):
    b, r, hist, _, diag = fedavg(sites, w, lams, E, P, return_diagnostics=True)
    if not diag["fit_converged"]:
        raise RuntimeError(f"FedAvg E={E} did not converge: {diag}")
    rows.append(dict(method="FedAvg", E=E, rounds=r, epochs=r*E, **diag, **agree(b)))
    for rule, nm in [("1se", "FedAvg-ST"), ("min", "FedAvg-ST-min")]:
        tau, _, _ = select_threshold(b, val, w, tau_grid, rule=rule)
        rows.append(dict(method=nm, E=E, rounds=r, epochs=r*E, tau=tau,
                         scalar_rounds=1, scalar_kib=st_scalar_kib,
                         **diag, **agree(soft(b, tau))))
    pgrid = lam_bar * np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    sols, rds = [], []
    for tp in pgrid:
        bp, rp, _, _, pdiag = fedavg(sites, w, lams, E, P, tau=tp,
                                   return_diagnostics=True)
        if not pdiag["fit_converged"]:
            raise RuntimeError(f"FedAvg-P E={E}, tau={tp} did not converge: {pdiag}")
        sols.append(bp); rds.append(rp)
    M = np.array(sols).T
    kp = select_by_val(M, val, w, "1se")
    rows.append(dict(method="FedAvg-P", E=E, rounds=sum(rds), epochs=sum(rds)*E,
                     selected_rounds=rds[kp], selected_epochs=rds[kp]*E,
                     candidate_rounds=";".join(map(str, rds)),
                     fit_converged=True, scalar_rounds=1,
                     tau=pgrid[kp], scalar_kib=3*len(pgrid)*3*8/1024,
                     **agree(M[:, kp])))
d = pd.DataFrame(rows)
d["scalar_kib"] = d.get("scalar_kib", 0.0)
d["scalar_kib"] = d["scalar_kib"].fillna(0.0)
d["scalar_rounds"] = d["scalar_rounds"].fillna(0)
d["lam"] = d["lam"].fillna(lam_bar)
d["vector_kib"] = d["rounds"] * 3 * P * 8 / 1024
d["upload_kib"] = d["vector_kib"] + d["scalar_kib"]
d["selected_rounds"] = d["selected_rounds"].fillna(d["rounds"])
d["selected_epochs"] = d["selected_epochs"].fillna(d["epochs"])
d["selected_vector_kib"] = d["selected_rounds"] * 3 * P * 8 / 1024
d["tuning_rounds"] = d["rounds"] - d["selected_rounds"]
d["tuning_epochs"] = d["epochs"] - d["selected_epochs"]
d["total_rounds"] = d["rounds"] + d["scalar_rounds"]
d["total_epochs"] = d["epochs"]
d["total_vector_kib"] = d["vector_kib"]
d["total_upload_kib"] = d["upload_kib"]
d.to_csv(HERE / "real_data_results.csv", index=False)
(HERE / "real_data_metadata.json").write_text(json.dumps({
    "dataset_kind": "fixed synthetic benchmark; original generator unavailable",
    "local_penalties": [float(x) for x in lams], "lambda_bar": lam_bar,
    "pooled_penalty": float(lam_p), "training_sizes": [int(x) for x in m],
    "validation_sizes": [len(yv) for _, yv in val],
    "feature_count": P,
    "source_file_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(ROOT.glob("node[123]_*.csv"))
    },
    "cost_scope": "all fitted FedAvg-P threshold candidates; selected costs also recorded; coefficient uploads plus validation scalars; setup/broadcasts excluded",
}, indent=2) + "\n")
print(d.round(3).to_string(index=False))
