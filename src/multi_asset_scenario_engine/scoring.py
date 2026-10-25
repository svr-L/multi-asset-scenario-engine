"""Proper multivariate scores and paired stationary-bootstrap inference."""
from __future__ import annotations
import numpy as np
from .dynamics import politis_white_block_length

def energy_score(draws, y, scale=None):
    X = np.asarray(draws, dtype=float)
    yy = np.asarray(y, dtype=float)
    if scale is not None:
        s = np.asarray(scale, dtype=float)
        X = X / s[None, :]
        yy = yy / s
    t1 = float(np.linalg.norm(X - yy[None, :], axis=1).mean())
    n2 = (len(X) // 2) * 2
    if n2 < 2:
        return t1
    pair = np.linalg.norm(X[:n2:2] - X[1:n2:2], axis=1).mean()
    return float(t1 - 0.5 * pair)

def variogram_score(draws, y, scale=None, labels=None, p=0.5):
    X = np.asarray(draws, dtype=float)
    yy = np.asarray(y, dtype=float)
    if scale is not None:
        s = np.asarray(scale, dtype=float)
        X = X / s[None, :]
        yy = yy / s
    labels = None if labels is None else np.asarray(labels, dtype=object)
    groups = {"EQ_RATES": [], "EQ_CREDIT": [], "RATES_CREDIT": [], "FX_CROSS": []}
    terms = []
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            obs = abs(yy[i] - yy[j]) ** float(p)
            pred = np.mean(np.abs(X[:, i] - X[:, j]) ** float(p))
            term = float((obs - pred) ** 2)
            terms.append(term)
            if labels is not None:
                a, b = labels[i], labels[j]
                pair = {a,b}
                if pair == {"equity","rates"}: groups["EQ_RATES"].append(term)
                elif pair == {"equity","credit"}: groups["EQ_CREDIT"].append(term)
                elif pair == {"rates","credit"}: groups["RATES_CREDIT"].append(term)
                if (a == "fx") ^ (b == "fx"): groups["FX_CROSS"].append(term)
    total = float(np.mean(terms)) if terms else np.nan
    comp = {g:(float(np.mean(v)) if v else np.nan) for g,v in groups.items()}
    return total, comp

def stationary_bootstrap_ci(diff, reps=999, seed=30030, level=0.95):
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    out = {"n":n,"mean_gain":np.nan,"median_gain":np.nan,"win_rate":np.nan,
           "ci_lo":np.nan,"ci_hi":np.nan,"p_gain_le_0":np.nan,"bootstrap_expected_block":np.nan}
    if n == 0:
        return out
    out.update(mean_gain=float(np.mean(x)), median_gain=float(np.median(x)), win_rate=float(np.mean(x > 0)))
    if n == 1:
        return out
    try:
        br = politis_white_block_length(x[:,None])
        bf = np.asarray(br, dtype=float)
        b = float(np.nanmedian(bf[np.isfinite(bf)])) if np.isfinite(bf).any() else 1.0
    except Exception:
        b = 1.0
    b = float(np.clip(b, 1.0, max(1.0, n/3.0)))
    pr = min(max(1.0/b, 1e-9), 1.0)
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(reps))
    for r in range(int(reps)):
        vals = []
        while len(vals) < n:
            st = int(rng.integers(0,n))
            L = int(rng.geometric(pr))
            take = min(L, n-len(vals))
            vals.extend(x[(st + np.arange(take)) % n].tolist())
        boots[r] = np.mean(vals)
    alpha = 1.0 - float(level)
    out.update(ci_lo=float(np.quantile(boots, alpha/2)),
               ci_hi=float(np.quantile(boots, 1-alpha/2)),
               p_gain_le_0=float((1 + np.sum(boots <= 0)) / (len(boots)+1)),
               bootstrap_expected_block=b)
    return out
