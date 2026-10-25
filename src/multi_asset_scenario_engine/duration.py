"""Duration-aware empirical representation and PIT pooling."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from .dynamics import politis_white_block_length

def precompute_duration_aggregates(panel, max_len):
    A = panel.to_numpy(dtype=float)
    cs = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, axis=0)])
    out = {}
    n = len(panel)
    for L in range(1, min(int(max_len), n - 1) + 1):
        vals = cs[L:] - cs[:-L]
        out[L] = pd.DataFrame(vals, index=panel.index[L-1:], columns=panel.columns)
    return out

def sorted_marginals(frame):
    return np.sort(np.asarray(frame, dtype=float), axis=0)

def pit_to_normal(values, sorted_training_marginals, clip=1e-4):
    X = np.atleast_2d(np.asarray(values, dtype=float))
    S = np.asarray(sorted_training_marginals, dtype=float)
    n, k = S.shape
    Z = np.empty_like(X)
    for j in range(k):
        r = np.searchsorted(S[:, j], X[:, j], side="right")
        u = (r + 0.5) / (n + 1.0)
        Z[:, j] = norm.ppf(np.clip(u, clip, 1.0 - clip))
    return Z

def rank_map_to_duration(z_draws, sorted_target):
    Z = np.asarray(z_draws, dtype=float)
    n, k = Z.shape
    U = np.empty_like(Z)
    for j in range(k):
        r = pd.Series(Z[:, j]).rank(method="average").to_numpy(dtype=float)
        U[:, j] = (r - 0.5) / n
    S = np.asarray(sorted_target, dtype=float)
    qgrid = (np.arange(S.shape[0], dtype=float) + 0.5) / S.shape[0]
    X = np.empty_like(Z)
    for j in range(k):
        X[:, j] = np.interp(np.clip(U[:, j], qgrid[0], qgrid[-1]), qgrid, S[:, j])
    return X

def expected_block_length(train, lookback_days=2520, fallback=5.0, max_days=63):
    X = train.iloc[-min(len(train), int(lookback_days)):].copy()
    sd = X.std(ddof=0).replace(0.0, np.nan)
    Z = ((X - X.mean()) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bcol = politis_white_block_length(np.abs(Z.to_numpy(dtype=float)))
    finite = bcol[np.isfinite(bcol)]
    b = float(np.nanmedian(finite)) if len(finite) else float(fallback)
    return float(np.clip(b, 1.0, max_days)), bcol

def build_random_duration_pit_library(train, aggregate_cache, expected_block, n_blocks=1000,
                                      max_days=63, pit_clip=1e-4, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n = len(train)
    p = min(max(1.0 / max(float(expected_block), 1.0), 1e-9), 1.0)
    Ls = np.minimum(rng.geometric(p, size=int(n_blocks)), int(max_days)).astype(int)
    Ls = np.minimum(Ls, max(n - 2, 1))
    starts = np.array([rng.integers(0, n - L + 1) for L in Ls], dtype=int)
    A = train.to_numpy(dtype=float)
    cs = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, axis=0)])
    vals = np.vstack([cs[s + L] - cs[s] for s, L in zip(starts, Ls)])
    end_pos = starts + Ls - 1
    z = np.empty_like(vals)
    end_date = train.index[-1]
    for L in np.unique(Ls):
        m = Ls == L
        hist = aggregate_cache[int(L)].loc[:end_date]
        z[m] = pit_to_normal(vals[m], sorted_marginals(hist), clip=pit_clip)
    return {"Z":pd.DataFrame(z, columns=train.columns),"length":Ls,
            "end_pos":end_pos,"end_date":train.index[end_pos]}
