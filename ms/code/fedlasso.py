"""Federated coordinate-descent Lasso: estimators used in the manuscript.

All local solvers use scikit-learn's cyclic coordinate-descent Lasso on the
normalised objective   (1/(2 m_j)) ||y_j - X_j b||^2 + lam_j ||b||_1 ,
with warm starts and a fixed number of full passes (epochs) per round.
"""
import warnings
import numpy as np
from sklearn.linear_model import Lasso, lasso_path
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def local_epochs(X, y, beta0, lam, E):
    """Run exactly E full CD passes on the local Lasso, warm-started at beta0."""
    m = Lasso(alpha=lam, fit_intercept=False, warm_start=True, max_iter=E,
              tol=0.0, selection="cyclic")
    m.coef_ = np.asarray(beta0, dtype=float).copy()
    m.n_iter_ = 0
    m.fit(X, y)
    return m.coef_.copy()


def local_solve(X, y, lam, beta0=None, max_iter=10000, tol=1e-10):
    m = Lasso(alpha=lam, fit_intercept=False, warm_start=beta0 is not None,
              max_iter=max_iter, tol=tol, selection="cyclic")
    if beta0 is not None:
        m.coef_ = np.asarray(beta0, dtype=float).copy()
        m.n_iter_ = 0
    m.fit(X, y)
    return m.coef_.copy()


def lam_grid(X, y, n_lam=30, eps=1e-3):
    lam_max = np.max(np.abs(X.T @ y)) / X.shape[0]
    return np.logspace(np.log10(lam_max), np.log10(lam_max * eps), n_lam)


def tune_lambda(Xtr, ytr, Xva, yva, n_lam=30):
    """Select lambda minimising validation MSE over a log grid (paper's rule)."""
    grid = lam_grid(Xtr, ytr, n_lam)
    _, coefs, _ = lasso_path(Xtr, ytr, alphas=grid, tol=1e-7, max_iter=20000)
    resid = yva[:, None] - Xva @ coefs
    mse = np.mean(resid ** 2, axis=0)
    k = int(np.argmin(mse))
    return grid[k], grid, mse


def objective(sites, beta, w, lams):
    """Sum_j w_j [ (1/(2 m_j))||y_j - X_j b||^2 + lam_j ||b||_1 ]."""
    tot = 0.0
    for (X, y), wj, lj in zip(sites, w, lams):
        r = y - X @ beta
        tot += wj * (r @ r / (2 * len(y)) + lj * np.abs(beta).sum())
    return tot


def fedavg(sites, w, lams, E, p, tol=1e-5, max_rounds=500, tau=None,
           val=None, tau_grid=None):
    """Synchronous federated CD-Lasso.

    tau is None            -> plain weighted averaging (FedAvg).
    tau is a float / 'auto'-> server applies a soft-threshold after each
                              aggregation (FedAvg-ST).  With 'auto', tau is
                              chosen once, after the first round, by federated
                              validation (only scalar losses leave a site).
    """
    beta = np.zeros(p)
    hist = []
    chosen_tau = None if tau == "auto" else tau
    for t in range(max_rounds):
        locals_ = [local_epochs(X, y, beta, lj, E) for (X, y), lj in zip(sites, lams)]
        bar = np.zeros(p)
        for bj, wj in zip(locals_, w):
            bar += wj * bj
        if tau == "auto" and chosen_tau is None:
            best, chosen_tau = np.inf, 0.0
            for tg in tau_grid:
                cand = soft(bar, tg)
                loss = sum(wj * np.mean((yv - Xv @ cand) ** 2)
                           for (Xv, yv), wj in zip(val, w))
                if loss < best:
                    best, chosen_tau = loss, tg
        new = bar if chosen_tau is None else soft(bar, chosen_tau)
        delta = np.linalg.norm(new - beta)
        beta = new
        hist.append(objective(sites, beta, w, lams))
        if delta < tol:
            break
    return beta, t + 1, hist, chosen_tau


def consensus_admm(sites, w, lam_bar, p, rho=1.0, max_iter=500, tol=1e-8):
    """Global consensus ADMM: exact minimiser of sum_j w_j L_j(b) + lam_bar||b||_1."""
    facs = []
    for (X, y), wj in zip(sites, w):
        m = len(y)
        A = wj * (X.T @ X) / m + rho * np.eye(p)
        facs.append((np.linalg.cholesky(A), wj * (X.T @ y) / m))
    x = [np.zeros(p) for _ in sites]
    u = [np.zeros(p) for _ in sites]
    z = np.zeros(p)
    K = len(sites)
    for it in range(max_iter):
        for j, (L, b) in enumerate(facs):
            rhs = b + rho * (z - u[j])
            v = np.linalg.solve(L, rhs)
            x[j] = np.linalg.solve(L.T, v)
        xbar = np.mean(x, axis=0)
        ubar = np.mean(u, axis=0)
        z_new = soft(xbar + ubar, lam_bar / (rho * K))
        s = rho * np.linalg.norm(z_new - z)
        z = z_new
        for j in range(K):
            u[j] += x[j] - z
        r = np.sqrt(sum(np.sum((xj - z) ** 2) for xj in x))
        if r < tol * np.sqrt(p) and s < tol * np.sqrt(p):
            break
    return z, it + 1


def select_threshold(bar, val, w, tau_grid, rule="1se"):
    """Choose a server-side soft-threshold by federated validation.

    Each site returns only two scalars per candidate (sum and sum of squares of
    its validation residuals), so no row-level data leave the site.
    """
    n = sum(len(yv) for _, yv in val)
    mse, se = [], []
    for tg in tau_grid:
        cand = soft(bar, tg)
        s1 = s2 = 0.0
        for (Xv, yv), wj in zip(val, w):
            r2 = (yv - Xv @ cand) ** 2
            s1 += wj * r2.sum() / len(yv) * len(yv)
            s2 += (r2 ** 2).sum()
        m = sum(wj * np.mean((yv - Xv @ cand) ** 2) for (Xv, yv), wj in zip(val, w))
        allr2 = np.concatenate([(yv - Xv @ cand) ** 2 for Xv, yv in val])
        mse.append(m)
        se.append(allr2.std(ddof=1) / np.sqrt(n))
    mse, se = np.array(mse), np.array(se)
    k = int(np.argmin(mse))
    if rule == "min":
        return tau_grid[k], mse, se
    thr = mse[k] + se[k]
    ok = np.where(mse <= thr)[0]
    return tau_grid[ok.max()], mse, se
