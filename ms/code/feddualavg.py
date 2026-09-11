"""Weighted, full-gradient specialization of FedDualAvg for squared loss.

Original NumPy implementation of Algorithm 3 in Yuan, Zaheer and Reddi
(2021), Federated Composite Optimization, PMLR 139:12253--12266:
https://proceedings.mlr.press/v139/yuan21d/yuan21d.pdf
The authors' reference implementation is https://github.com/hongliny/FCO-ICML21.
No code from that repository is copied here.

Mapping to the paper: h(b)=||b||^2/2, psi(b)=lam*||b||_1;
z_0=0 (line 2); all sites participate (line 4); local_steps is their K;
the client primal is soft(z, (eta_s*eta_c*r*K+eta_c*k)*lam) (8--9);
the gradient is the full local empirical squared-loss gradient (10--11);
line 12's uniform mean is replaced by the supplied sample-size weights;
line 14 is soft(z_server, eta_s*eta_c*(r+1)*K*lam).

Client rates can be evaluated together, but every column is an independent
fit and incurs its own vector uploads. Local Gram matrices are cached inside
each simulated client, never transmitted. Thus a local unit is a full-gradient
evaluation from sufficient statistics, not a coordinate-descent sweep.
The returned estimate is the final server primal iterate, not the theoretical
average over uncommunicated within-round primal states.
"""
import numpy as np
from scipy.linalg import eigvalsh


def _soft(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def local_lipschitz(sites):
    """Exact largest eigenvalue of X_j'X_j/m_j, computed locally."""
    values = []
    for X, _ in sites:
        smaller = X @ X.T if X.shape[0] <= X.shape[1] else X.T @ X
        n = smaller.shape[0]
        values.append(float(eigvalsh(smaller, subset_by_index=[n - 1, n - 1])[0]
                            / len(X)))
    return np.asarray(values)


def feddualavg_path(sites, weights, lam, local_steps, round_grid, client_rates,
                   server_rate=1.0):
    """Return {round: coefficients[p, n_rates]} after exactly each budget.

    Sites hold their own X/y and Gram/score arrays in this simulation.
    Each round communicates one length-p dual update per site per rate.
    No early stopping, validation, rate selection, or objective evaluation is
    performed here. Checkpoints share the same training path.
    """
    weights = np.asarray(weights, dtype=float)
    rates = np.atleast_1d(client_rates).astype(float)
    checkpoints = sorted(set(int(r) for r in round_grid))
    if (not sites or len(weights) != len(sites) or
            np.any(weights < 0) or not np.isclose(weights.sum(), 1)):
        raise ValueError("weights must be nonnegative, sum to one, and match sites")
    if (int(local_steps) != local_steps or local_steps < 1 or not checkpoints or
            checkpoints[0] < 1 or np.any(rates <= 0) or lam < 0 or server_rate <= 0):
        raise ValueError("invalid steps, round grid, rates, penalty, or server rate")
    p = sites[0][0].shape[1]
    grams = [X.T @ X / len(y) for X, y in sites]
    scores = [X.T @ y / len(y) for X, y in sites]
    z = np.zeros((p, len(rates)))
    out = {}
    for r in range(checkpoints[-1]):
        averaged_delta = np.zeros_like(z)
        for weight, gram, score in zip(weights, grams, scores):
            local_z = z.copy()
            for k in range(local_steps):
                effective = rates * (server_rate * r * local_steps + k)
                b = _soft(local_z, lam * effective)
                gradient = gram @ b - score[:, None]
                local_z -= gradient * rates
            averaged_delta += weight * (local_z - z)
        z += server_rate * averaged_delta
        if r + 1 in checkpoints:
            out[r + 1] = _soft(z, lam * server_rate * rates * (r + 1) * local_steps)
    return out


def kkt_mapping_residual(sites, weights, lam, beta):
    """Infinity norm of beta - prox_{lam||.||_1}(beta - grad(loss)).

    A unit-step proximal-gradient mapping; zero is equivalent to Lasso KKT.
    This avoids labelling tiny floating-point values active for diagnostics.
    Evaluation is an offline diagnostic, excluded from training cost.
    """
    gradient = sum(w * X.T @ (X @ beta - y) / len(y)
                   for w, (X, y) in zip(weights, sites))
    return float(np.max(np.abs(beta - _soft(beta - gradient, lam))))
