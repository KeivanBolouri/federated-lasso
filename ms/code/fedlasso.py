"""Federated coordinate-descent Lasso: estimators used in the manuscript.

All local solvers use scikit-learn's cyclic coordinate-descent Lasso on the
normalised objective   (1/(2 m_j)) ||y_j - X_j b||^2 + lam_j ||b||_1 ,
with warm starts and a fixed number of full passes (epochs) per round.
"""
import warnings
import numpy as np
from scipy.linalg import cho_solve
from sklearn.linear_model import Lasso, lasso_path
from sklearn.exceptions import ConvergenceWarning

def soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def local_epochs(X, y, beta0, lam, E):
    """Run exactly E full CD passes on the local Lasso, warm-started at beta0."""
    m = Lasso(alpha=lam, fit_intercept=False, warm_start=True, max_iter=E,
              tol=0.0, selection="cyclic")
    m.coef_ = np.asarray(beta0, dtype=float).copy()
    m.n_iter_ = 0
    # Fixed-budget local epochs intentionally need not solve the local problem.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(X, y)
    return m.coef_.copy()


def local_solve(X, y, lam, beta0=None, max_iter=10000, tol=1e-10,
                max_total_iter=200000, return_diagnostics=False):
    """Solve a final local/pooled fit, extending the budget until the gap passes.

    tol is relative to mean(y**2), as in scikit-learn's returned dual gap.
    Numerical failures raise rather than silently entering the simulation data.
    Fixed-budget federated local epochs use local_epochs, not this function.
    """
    m = Lasso(alpha=lam, fit_intercept=False, warm_start=beta0 is not None,
              max_iter=max_iter, tol=tol, selection="cyclic")
    if beta0 is not None:
        m.coef_ = np.asarray(beta0, dtype=float).copy()
        m.n_iter_ = 0
    total = attempts = 0
    target = tol * float(np.mean(np.asarray(y) ** 2))
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(X, y)
        total += int(m.n_iter_)
        attempts += 1
        gap = float(m.dual_gap_)
        converged = bool(np.isfinite(gap) and gap <= target)
        if converged:
            break
        if total >= max_total_iter or not np.isfinite(gap):
            raise RuntimeError(f"Final Lasso did not converge: gap={gap:g}, "
                               f"target={target:g}, iterations={total}")
        m.warm_start = True
        m.max_iter = min(max(4 * m.max_iter, 1), max_total_iter - total)
    result = m.coef_.copy()
    if return_diagnostics:
        return result, dict(fit_converged=converged, local_dual_gap=gap,
                            local_dual_gap_tolerance=target,
                            solver_iterations=total, solver_restarts=attempts-1)
    return result


def lam_grid(X, y, n_lam=30, eps=1e-3):
    lam_max = np.max(np.abs(X.T @ y)) / X.shape[0]
    return np.logspace(np.log10(lam_max), np.log10(lam_max * eps), n_lam)


def tune_lambda(Xtr, ytr, Xva, yva, n_lam=30):
    """Select lambda minimising validation MSE over a log grid (paper's rule)."""
    grid = lam_grid(Xtr, ytr, n_lam)
    _, coefs, _ = lasso_path(Xtr, ytr, alphas=grid, tol=1e-4, max_iter=2000)
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
           val=None, tau_grid=None, return_diagnostics=False):
    """Synchronous federated CD-Lasso.

    tau is None            -> plain weighted averaging (FedAvg).
    tau is a float / 'auto'-> server applies a soft-threshold after each
                              aggregation (FedAvg-P). With 'auto', tau is
                              chosen once, after the first round, by federated
                              validation (only scalar losses leave a site).
    """
    if E < 1 or max_rounds < 1 or tol <= 0:
        raise ValueError("E, max_rounds and tol must be positive")
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
    result = (beta, t + 1, hist, chosen_tau)
    if return_diagnostics:
        return (*result, dict(fit_converged=bool(np.isfinite(delta) and delta < tol),
                              final_update_norm=float(delta), fit_tolerance=tol,
                              hit_round_cap=bool(t + 1 == max_rounds and delta >= tol)))
    return result


def consensus_admm(sites, w, lam_bar, p, rho=1.0, max_iter=500, tol=1e-8,
                   warm=None, return_state=False, return_diagnostics=False):
    """Consensus ADMM for the pooled objective, to the recorded residual tolerance.

    The stopping rule uses ||r||_2 and rho*||z_new-z||_2, each below
    tol*sqrt(p), preserving the original study's numerical stopping convention.
    The latter is the consensus dual residual divided by sqrt(K).
    """
    facs = []
    for (X, y), wj in zip(sites, w):
        m = len(y)
        A = wj * (X.T @ X) / m + rho * np.eye(p)
        facs.append((np.linalg.cholesky(A), wj * (X.T @ y) / m))
    if warm is not None and warm[0] is not None:
        x = [xi.copy() for xi in warm[0]]
        u = [ui.copy() for ui in warm[1]]
        z = warm[2].copy()
    else:
        x = [np.zeros(p) for _ in sites]
        u = [np.zeros(p) for _ in sites]
        z = np.zeros(p)
    K = len(sites)
    for it in range(max_iter):
        for j, (L, b) in enumerate(facs):
            rhs = b + rho * (z - u[j])
            x[j] = cho_solve((L, True), rhs, check_finite=False)
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
    result = (z, it + 1, (x, u, z)) if return_state else (z, it + 1)
    if return_diagnostics:
        grad = sum(wj * X.T @ (X @ z - y) / len(y)
                   for (X, y), wj in zip(sites, w))
        violation = np.where(z != 0, np.abs(grad + lam_bar * np.sign(z)),
                             np.maximum(np.abs(grad) - lam_bar, 0.0))
        diag = dict(fit_converged=bool(r < tol * np.sqrt(p) and s < tol * np.sqrt(p)),
                    primal_residual=float(r), dual_residual_scaled=float(s),
                    kkt_residual=float(np.max(violation)), fit_tolerance=tol,
                    hit_round_cap=bool(it + 1 == max_iter and
                                       not (r < tol * np.sqrt(p) and s < tol * np.sqrt(p))))
        return (*result, diag)
    return result


def validation_summaries(coefs, val, w):
    """Weighted MSE and pooled observation-level SE from site scalar summaries.

    Each site emits (sum squared residuals, sum fourth powers, count) per
    candidate; no observation-level residual arrays are pooled. The SE is the
    conventional one-SE tuning heuristic, not a clustered or heterogeneity-
    robust uncertainty estimate. Validation sizes must be proportional to w.
    """
    coefs = np.asarray(coefs)
    if coefs.ndim == 1:
        coefs = coefs[:, None]
    counts = np.array([len(y) for _, y in val], dtype=float)
    if not np.allclose(w, counts / counts.sum()):
        raise ValueError("Pooled one-SE heuristic requires validation proportions equal to w")
    sums, fourths = [], []
    for Xv, yv in val:
        r2 = (yv[:, None] - Xv @ coefs) ** 2
        sums.append(r2.sum(axis=0))
        fourths.append((r2 ** 2).sum(axis=0))
    sums, fourths = np.array(sums), np.array(fourths)
    n = counts.sum()
    mse = np.sum(np.asarray(w)[:, None] * sums / counts[:, None], axis=0)
    variance = (fourths.sum(axis=0) - sums.sum(axis=0) ** 2 / n) / (n * (n - 1))
    return mse, np.sqrt(np.maximum(variance, 0.0))


def select_threshold(bar, val, w, tau_grid, rule="1se"):
    """Choose a server-side soft-threshold by federated validation.

    Each site returns three scalars per candidate (sum squared residuals,
    sum fourth powers, count), so no row-level data leave the site.
    """
    candidates = soft(np.asarray(bar)[:, None], np.asarray(tau_grid)[None, :])
    mse, se = validation_summaries(candidates, val, w)
    k = int(np.argmin(mse))
    if rule == "min":
        return tau_grid[k], mse, se
    thr = mse[k] + se[k]
    ok = np.where(mse <= thr)[0]
    return tau_grid[ok.max()], mse, se


def _one_se_pick(grid, mse, se_at_min, ascending_sparser=True):
    """Largest penalty whose validation MSE is within one SE of the minimum."""
    k = int(np.argmin(mse))
    ok = np.where(mse <= mse[k] + se_at_min)[0]
    return int(ok.max()) if ascending_sparser else int(ok.min())


def tune_lambda_1se(Xtr, ytr, Xva, yva, n_lam=30):
    """Validation tuning with the one-standard-error rule (grid ordered
    from large to small penalty, so 'largest within one SE' is index-min)."""
    grid = lam_grid(Xtr, ytr, n_lam)              # decreasing
    _, coefs, _ = lasso_path(Xtr, ytr, alphas=grid, tol=1e-4, max_iter=2000)
    r2 = (yva[:, None] - Xva @ coefs) ** 2
    mse = r2.mean(axis=0)
    k = int(np.argmin(mse))
    se = r2[:, k].std(ddof=1) / np.sqrt(len(yva))
    ok = np.where(mse <= mse[k] + se)[0]
    return grid[int(ok.min())], grid, mse       # index-min == largest lambda


def admm_path(sites, w, grid, p, rho=1.0, max_iter=500, tol=1e-8):
    """Consensus-ADMM Lasso path with warm starts (used for verification)."""
    out, z0, u0, x0 = [], None, None, None
    for lam in grid:
        z, _, state = consensus_admm(sites, w, lam, p, rho=rho, max_iter=max_iter,
                                     tol=tol, warm=(x0, u0, z0), return_state=True)
        x0, u0, z0 = state
        out.append(z.copy())
    return np.array(out).T          # p x n_lam


def consensus_path_exact(Xp, yp, grid):
    """Solutions of F(beta) over a penalty grid.

    By Lemma 1 the consensus objective equals the pooled Lasso objective, so
    the consensus-ADMM solution at penalty lam is the pooled Lasso solution at
    lam.  We verify this numerically in `admm_path` and use the direct solver
    here; the communication cost of ADMM is reported separately from an actual
    ADMM run.
    """
    _, coefs, _ = lasso_path(Xp, yp, alphas=grid, tol=1e-6, max_iter=20000)
    return coefs


def select_by_val(coefs, val, w, rule="min"):
    """Pick a column of `coefs` by weighted validation MSE (min or 1-SE)."""
    mse, se = validation_summaries(coefs, val, w)
    k = int(np.argmin(mse))
    if rule == "min":
        return k
    ok = np.where(mse <= mse[k] + se[k])[0]
    return int(ok.min())            # grids are ordered large -> small penalty
