"""Step 3b - support audits, softmax routers, Laplace-mixture median and aligned feature blocks.

These are the routing components shared by SAER-7g, AlignedAddAudit and NestedAligned.
All arrays are normalised exactly as in the design document:

    P_j  = (p_j - b) / s          expert proposals relative to expert 1, scaled by the training residual SD
    A_jr = (p_j(r; S\\{r}) - y_ir) / s   signed audit error at support r
    u    = (y_iq - b) / s         normalised target (training / validation only)

Nothing here reads a hidden query response at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

N_FEATURES = 7
N_BIAS = 5
EPS_LOSS = 1e-6
ROUTER_L2 = 0.001
ROUTER_MAX_ITER = 300
LAPLACE_ITERS = 34
SCALE_FLOOR = 1e-5


# --------------------------------------------------------------------------------------
# Audit features
# --------------------------------------------------------------------------------------
def relative_skill(mae: np.ndarray, base_index: int = 0) -> np.ndarray:
    """log[(MAE_j + 1e-6) / (MAE_1 + 1e-6)] per expert; expert 1 is the reference (value 0)."""
    mae = np.asarray(mae, dtype=float)
    return np.log((mae + 1e-6) / (mae[base_index] + 1e-6))


def audit_features(P: np.ndarray, A: np.ndarray, rho: np.ndarray, skill: np.ndarray):
    """Seven routing features and five bias channels per expert.

    P (E, J), A (E, J, m), rho (E, m) signed correlations of each support with the query,
    skill (J,). Returns Phi (E, J, 7) and C (E, J, 5).
    """
    E, J, m = A.shape
    absA = np.abs(A)
    w = np.abs(rho) + 0.001                                   # relevance weights |rho| + 0.001
    f1 = absA.mean(axis=2)
    f2 = (absA * w[:, None, :]).sum(axis=2) / w.sum(axis=1)[:, None]
    f3 = np.abs(A.mean(axis=2))
    f4 = absA.max(axis=2)
    f5 = np.abs(P - np.median(P, axis=1, keepdims=True))
    f6 = np.broadcast_to(np.asarray(skill, dtype=float)[None, :], (E, J))
    star = np.argmax(np.abs(rho), axis=1)                      # strongest-correlated support
    rows = np.arange(E)
    f7 = absA[rows, :, star]
    Phi = np.stack([f1, f2, f3, f4, f5, f6, f7], axis=2)
    a = A.mean(axis=2)
    c = (A * rho[:, None, :]).sum(axis=2) / (np.abs(rho).sum(axis=1)[:, None] + 0.001)
    r = A[rows, :, star] * np.sign(rho[rows, star])[:, None]
    mx = np.abs(rho).max(axis=1)[:, None]
    C = np.stack([a, c, mx * c, r, mx * r], axis=2)
    return Phi, C


# --------------------------------------------------------------------------------------
# Softmax router
# --------------------------------------------------------------------------------------
@dataclass
class Records:
    """Auxiliary episodes for router fitting (normalised units). ``avail`` marks usable experts."""
    Phi: np.ndarray            # (E, J, 7)
    C: np.ndarray              # (E, J, 5)
    P: np.ndarray              # (E, J)
    u: np.ndarray              # (E,)
    w: np.ndarray              # (E,) training weights
    avail: np.ndarray          # (J,) bool
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.u)


@dataclass
class RouterParams:
    beta: np.ndarray           # (J,)
    theta: np.ndarray          # (7,)
    K: np.ndarray              # (J, 5)
    avail: np.ndarray          # (J,)
    converged: bool = True
    n_iter: int = 0


def _forward(params, Phi, C, P, avail):
    beta, theta, K = params
    z = beta[None, :] - Phi @ theta
    z = np.where(avail[None, :], z, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    w = np.exp(z)
    w /= w.sum(axis=1, keepdims=True)
    Pp = P - np.einsum("ejc,jc->ej", C, K)
    return w, Pp, (w * Pp).sum(axis=1)


def _unpack(x, J):
    return x[:J], x[J:J + N_FEATURES], x[J + N_FEATURES:].reshape(J, N_BIAS)


def router_loss_grad(x, Phi, C, P, u, wt, avail):
    J = P.shape[1]
    beta, theta, K = _unpack(x, J)
    w, Pp, f = _forward((beta, theta, K), Phi, C, P, avail)
    r = f - u
    root = np.sqrt(r * r + EPS_LOSS)
    tot = wt.sum()
    loss = (wt * root).sum() / tot + ROUTER_L2 * (theta @ theta + (K * K).sum())
    g = wt * r / root / tot                                     # dL/df
    gz = g[:, None] * w * (Pp - f[:, None])                      # dL/dz (avail experts only; w=0 elsewhere)
    gbeta = gz.sum(axis=0)
    gtheta = -(gz[:, :, None] * Phi).sum(axis=(0, 1)) + 2 * ROUTER_L2 * theta
    gK = -(g[:, None, None] * w[:, :, None] * C).sum(axis=0) + 2 * ROUTER_L2 * K
    gbeta = np.where(avail, gbeta, 0.0)
    gK = np.where(avail[:, None], gK, 0.0)
    return loss, np.concatenate([gbeta, gtheta, gK.ravel()])


def fit_router(rec: Records, max_iter: int = ROUTER_MAX_ITER) -> RouterParams:
    """L-BFGS-B, zero initialisation, 300-iteration cap (registered recipe)."""
    J = rec.P.shape[1]
    x0 = np.zeros(J + N_FEATURES + J * N_BIAS)
    res = minimize(router_loss_grad, x0, args=(rec.Phi, rec.C, rec.P, rec.u, rec.w, rec.avail), jac=True,
                   method="L-BFGS-B", options={"maxiter": max_iter})
    beta, theta, K = _unpack(res.x, J)
    return RouterParams(beta, theta, K, rec.avail.copy(), bool(res.success), int(res.nit))


def router_predict(params: RouterParams, Phi, C, P):
    """Return mixture weights (E, J), corrected proposals P' (E, J) and f (E,)."""
    return _forward((params.beta, params.theta, params.K), Phi, C, P, params.avail)


def normalise_total_weight(w: np.ndarray) -> np.ndarray:
    return w / w.sum()


def group_equal_mass(groups: np.ndarray) -> np.ndarray:
    """Weights giving equal total mass to each group (e.g. gene on multi-kinase panels)."""
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w / w.sum()


def stack_records(parts: list[Records], panel_weights: bool = True) -> Records:
    """Pool records from several development panels; each panel contributes equal total weight."""
    J = parts[0].P.shape[1]
    avail = np.zeros(J, dtype=bool)
    for p in parts:
        avail |= p.avail
    w = [normalise_total_weight(p.w) if panel_weights else p.w for p in parts]
    return Records(np.concatenate([p.Phi for p in parts]), np.concatenate([p.C for p in parts]),
                   np.concatenate([p.P for p in parts]), np.concatenate([p.u for p in parts]),
                   np.concatenate(w), avail)


# --------------------------------------------------------------------------------------
# Laplace-mixture median (AlignedAddAudit / NestedAligned reference)
# --------------------------------------------------------------------------------------
def laplace_cdf(x, mu, d):
    d = np.maximum(d, 1e-8)
    z = (x - mu) / d
    return np.where(z < 0, 0.5 * np.exp(np.minimum(z, 0)), 1.0 - 0.5 * np.exp(-np.maximum(z, 0)))


def laplace_mixture_median(mu: np.ndarray, a: np.ndarray, d: np.ndarray, iters: int = LAPLACE_ITERS) -> np.ndarray:
    """Solve sum_t a_t F_Laplace(M; mu_t, d_t) = 0.5 by bisection between the smallest and largest centre.

    mu, a, d: (E, T) component centres, weights (sum to one) and scales. 34 bisection iterations.
    """
    lo, hi = mu.min(axis=1).copy(), mu.max(axis=1).copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cdf = (a * laplace_cdf(mid[:, None], mu, d)).sum(axis=1)
        below = cdf < 0.5
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------
# Canonical support order and aligned feature block
# --------------------------------------------------------------------------------------
def canonical_order(rho: np.ndarray, r: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Permutation (E, m) sorting supports by descending |rho|, then signed rho, support residual and the audit vector.

    Ties are resolved with a full lexicographic sort so that jointly permuting supports, their values
    and the audit tensor reproduces the same feature row.
    """
    E, m = rho.shape
    order = np.argsort(-np.abs(rho), axis=1, kind="stable")
    sorted_abs = np.take_along_axis(np.abs(rho), order, axis=1)
    tie = (np.abs(np.diff(sorted_abs, axis=1)) < 1e-12).any(axis=1) if m > 1 else np.zeros(E, bool)
    for e in np.flatnonzero(tie):
        keys = [A[e, j, :] for j in range(A.shape[1] - 1, -1, -1)] + [r[e], rho[e], -np.abs(rho[e])]
        order[e] = np.lexsort(keys)
    return order


def aligned_features(P, A, rho, r, f_pool, prior_q, q, include_query_id: bool = True,
                     pool_zero: bool = False, prior_relative: np.ndarray | None = None) -> np.ndarray:
    """Aligned feature block for the residual head.

    P (E, J) normalised proposals, A (E, J, m) audits, rho (E, m), r (E, m) normalised support residuals,
    f_pool (E,) normalised pooled-router output, prior_q (E,) raw query prior, q (E,) query index.
    The hidden query response is not an input. ``include_query_id=False`` / ``pool_zero=True`` /
    ``prior_relative`` give the cross-study fallback feature block.
    """
    E, J, m = A.shape
    order = canonical_order(rho, r, A)
    rows = np.arange(E)[:, None]
    rho_o, r_o = rho[rows, order], r[rows, order]
    A_o = A[np.arange(E)[:, None, None], np.arange(J)[None, :, None], order[:, None, :]]
    block = [P, np.nanstd(np.where(np.isfinite(P), P, np.nan), axis=1, keepdims=True), rho_o, r_o,
             A_o.reshape(E, -1), (A_o * rho_o[:, None, :]).reshape(E, -1), (A_o - r_o[:, None, :]).reshape(E, -1)]
    block.append(np.zeros((E, 1)) if pool_zero else f_pool[:, None])
    block.append((prior_relative if prior_relative is not None else prior_q)[:, None])
    if include_query_id:
        block.append(q[:, None].astype(float))
    return np.concatenate(block, axis=1)


def weighted_mae(pred, truth, w=None) -> float:
    err = np.abs(np.asarray(pred) - np.asarray(truth))
    return float(err.mean() if w is None else (w * err).sum() / w.sum())
