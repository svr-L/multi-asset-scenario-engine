"""Cross-asset dependence research utilities.

The production scenario engine keeps blockwise empirical resampling as the hard benchmark.
This module contains the global/local hierarchical challenger used in the v24-v30 research line.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

def _offblock_lowrank_fit(S, mask_off, labels, k, n_iter=300, tol=1e-10, max_block_share=0.95):
    blocks = list(dict.fromkeys(np.asarray(labels, dtype=object).tolist()))
    idx = {b: np.where(np.asarray(labels) == b)[0] for b in blocks}
    inv_half = {}
    for b in blocks:
        ev, V = np.linalg.eigh(np.atleast_2d(S[np.ix_(idx[b], idx[b])]))
        ev = np.clip(ev, ev.max() * 1e-12, None)
        inv_half[b] = (V / np.sqrt(ev)) @ V.T
    A = S.copy()
    off_energy = float(np.sum((S * mask_off) ** 2))
    prev = None
    best = None
    for it in range(1, int(n_iter) + 1):
        ev, V = np.linalg.eigh(A)
        top = np.argsort(ev)[::-1][:k]
        L = V[:, top] * np.sqrt(np.clip(ev[top], 0.0, None))
        for b in blocks:
            Lb = L[idx[b]]
            r = float(np.linalg.eigvalsh(np.atleast_2d(inv_half[b] @ Lb @ Lb.T @ inv_half[b]))[-1])
            if r > max_block_share:
                L[idx[b]] = Lb * np.sqrt(max_block_share / r)
        LL = L @ L.T
        obj = float(np.sum(((S - LL) * mask_off) ** 2))
        if best is None or obj < best[1]:
            best = (L.copy(), obj)
        if prev is not None and abs(prev - obj) <= tol * max(off_energy, 1e-300):
            break
        prev = obj
        A = np.where(mask_off, S, LL)
    return best[0], best[1], off_energy, it

def _bartlett_weights(S, L, labels):
    labels = np.asarray(labels, dtype=object)
    Psi = np.zeros_like(S)
    for b in dict.fromkeys(labels.tolist()):
        m = np.where(labels == b)[0]
        B = S[np.ix_(m, m)] - L[m] @ L[m].T
        ev, V = np.linalg.eigh(np.atleast_2d(B))
        floor = 1e-3 * float(np.trace(np.atleast_2d(S[np.ix_(m, m)]))) / len(m)
        Psi[np.ix_(m, m)] = (V * np.clip(ev, floor, None)) @ V.T
    PiL = np.linalg.solve(Psi, L)
    return PiL @ np.linalg.inv(L.T @ PiL)

def _crossblock_objective(x, S, mask_off, kg):
    W = x.reshape(S.shape[0], kg)
    SW = S @ W
    try:
        Ai = np.linalg.inv(W.T @ SW)
    except np.linalg.LinAlgError:
        return 1e300, np.zeros_like(x)
    G = mask_off * (S - SW @ Ai @ SW.T)
    SGS = S @ G @ S
    grad = -4.0 * (SGS @ W @ Ai - SW @ (Ai @ (W.T @ SGS @ W) @ Ai))
    return float(np.sum(G * G)), grad.ravel()

def _crossblock_factor_weights(S, mask_off, labels, kg, extra_starts=()):
    starts = [np.asarray(s_, dtype=float) for s_ in extra_starts]
    L, _, _, _ = _offblock_lowrank_fit(S, mask_off, labels, kg)
    starts.append(_bartlett_weights(S, L, labels))
    ev, V = np.linalg.eigh(S)
    starts.append(V[:, np.argsort(ev)[::-1][:kg]])
    best = None
    for W0 in starts:
        r = minimize(_crossblock_objective, W0.ravel(), args=(S, mask_off, kg), jac=True,
                     method="L-BFGS-B", options={"maxiter": 5000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x.reshape(S.shape[0], kg)

def _nested_unit_factors(Ds, W, w, prev_g=None):
    g = Ds @ W
    ev, V = np.linalg.eigh(np.atleast_2d((g * w[:, None]).T @ g))
    h = g @ (V / np.sqrt(np.clip(ev, 1e-300, None)))
    kg = h.shape[1]
    if prev_g is None or prev_g.shape[1] == 0:
        return h[:, ::-1]
    A = (h * w[:, None]).T @ prev_g
    Q, _ = np.linalg.qr(np.hstack([A, np.eye(kg)]))
    Q = Q[:, :kg]
    for j in range(A.shape[1]):
        if float(Q[:, j] @ A[:, j]) < 0:
            Q[:, j] *= -1.0
    return h @ Q

def _normalise_weights(weights):
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    return w / w.sum()

def fit_hierarchical_dependence(Z, labels, weights=None, max_global=3):
    """Fit a low-dimensional global/local structure targeted at cross-block covariance."""
    D = np.asarray(Z, dtype=float)
    labels = np.asarray(labels, dtype=object)
    T, k = D.shape
    w = _normalise_weights(np.ones(T) if weights is None else weights)
    blocks = list(dict.fromkeys(labels.tolist()))
    mu = (D * w[:, None]).sum(axis=0)
    Dc = D - mu
    scale = np.ones(k)
    for b in blocks:
        m = labels == b
        Cb = (Dc[:, m] * w[:, None]).T @ Dc[:, m]
        lam1 = float(np.linalg.eigvalsh(np.atleast_2d(Cb))[-1])
        scale[m] = 1.0 / np.sqrt(max(lam1, 1e-12))
    Ds = Dc * scale
    S = (Ds * w[:, None]).T @ Ds
    mask_off = labels[:, None] != labels[None, :]
    sd0 = np.sqrt(np.clip(np.diag((Dc * w[:, None]).T @ Dc), 1e-12, None))
    ess = 1.0 / np.sum(w ** 2)
    tol = max(0.05, 3.5 / np.sqrt(max(ess, 1.0)))
    fits = {}
    prev_g = None
    kmax = min(int(max_global), k - 1)
    for kg in range(1, kmax + 1):
        extra = []
        if prev_g is not None:
            prev_W = np.linalg.lstsq(Ds, prev_g, rcond=None)[0]
            extra.append(np.hstack([prev_W, np.eye(k)[:, [0]]]))
        W = _crossblock_factor_weights(S, mask_off, labels, kg, extra_starts=extra)
        g = _nested_unit_factors(Ds, W, w, prev_g)
        lam = (Dc * w[:, None]).T @ g
        U = Dc - g @ lam.T
        C_u = (U * w[:, None]).T @ U
        Roff = np.where(mask_off, np.abs(C_u / np.outer(sd0, sd0)), -1.0)
        rmax = float(Roff[mask_off].max()) if mask_off.any() else 0.0
        fits[kg] = (g, lam, U, rmax)
        prev_g = g
    ok = [kg for kg, (_, _, _, rmax) in fits.items() if rmax <= tol]
    kg = min(ok) if ok else max(fits)
    g, lam, U, rmax = fits[kg]
    groups = [np.where(labels == b)[0] for b in blocks]
    return {"mean":mu,"g":g,"loadings":lam,"u":U,"groups":groups,"k_global":kg,
            "weights":w,"ess":ess,"resid_max":rmax,"labels":labels}

def sample_hierarchical_dependence(fit, n_scenarios, rng=None):
    """Empirically resample global factors and block-local residuals."""
    rng = np.random.default_rng() if rng is None else rng
    w = fit["weights"]
    T = len(w)
    gi = rng.choice(T, size=int(n_scenarios), replace=True, p=w)
    G = fit["g"][gi]
    Ustar = np.zeros((int(n_scenarios), fit["u"].shape[1]))
    for cols in fit["groups"]:
        ui = rng.choice(T, size=int(n_scenarios), replace=True, p=w)
        Ustar[:, cols] = fit["u"][ui][:, cols]
    return fit["mean"][None, :] + G @ fit["loadings"].T + Ustar
