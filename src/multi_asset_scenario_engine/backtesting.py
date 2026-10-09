from __future__ import annotations

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import chi2
from scipy.stats import t as student_t
from scipy.stats import norm
from scipy.special import gammaln
from pandas.tseries.offsets import BDay


def kupiec_pof_test(n_violations, n_obs, alpha):
    """Kupiec proportion-of-failures (unconditional coverage) LR test.

    H0: the VaR violation rate equals `alpha`. Returns (LR_stat, p_value); LR_stat ~ chi2(1).
    The expected number of violations is alpha * n_obs.
    """
    x, N, p = int(n_violations), int(n_obs), float(alpha)
    if N == 0:
        return np.nan, np.nan
    pi_hat = x / N
    if x == 0:
        lr = -2.0 * (N * np.log(1 - p))
    elif x == N:
        lr = -2.0 * (N * np.log(p))
    else:
        ll_null = x * np.log(p) + (N - x) * np.log(1 - p)
        ll_alt = x * np.log(pi_hat) + (N - x) * np.log(1 - pi_hat)
        lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(chi2.sf(lr, df=1))


def christoffersen_independence_test(violations):
    """Christoffersen independence LR test on a 0/1 violation sequence.

    H0: violations are serially independent (no clustering of breaches). Returns (LR_stat, p_value);
    LR_stat ~ chi2(1).
    """
    v = np.asarray(violations, dtype=int)
    if len(v) < 2:
        return np.nan, np.nan
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(v[:-1], v[1:]):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1
    denom01 = n00 + n01
    denom11 = n10 + n11
    pi01 = n01 / denom01 if denom01 > 0 else 0.0
    pi11 = n11 / denom11 if denom11 > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _safe(p):
        return min(max(p, 1e-12), 1 - 1e-12)

    ll_null = (n00 + n10) * np.log(1 - _safe(pi)) + (n01 + n11) * np.log(_safe(pi))
    ll_alt = (n00 * np.log(1 - _safe(pi01)) + n01 * np.log(_safe(pi01))
              + n10 * np.log(1 - _safe(pi11)) + n11 * np.log(_safe(pi11)))
    lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(chi2.sf(lr, df=1))


def contiguous_window_start(index, tol_days):
    """Start date after the last internal calendar gap longer than `tol_days` (None if no such gap).

    A day-to-day change taken across a multi-year data hole (the known case: the 30-year Treasury
    discontinuation 2002-02 -> 2006-02) would enter the bootstrap pool as ONE fake extreme
    observation -- poison for a tail engine. The main data cell and the walk-forward backtest's
    per-as-of refit must apply the SAME rule, or an early backtest window could silently contain a
    hole the main window trims away. Returns (start_or_None, n_gaps, max_gap_days).
    """
    idx = pd.DatetimeIndex(index)
    if len(idx) < 2:
        return None, 0, 0
    gaps = idx.to_series().diff().dt.days
    big = gaps[gaps > float(tol_days)]
    if len(big) == 0:
        return None, 0, int(np.nanmax(gaps.to_numpy()[1:])) if len(idx) > 1 else 0
    return big.index[-1], int(len(big)), int(np.nanmax(gaps.to_numpy()[1:]))


def christoffersen_conditional_coverage_test(violations, alpha):
    """Christoffersen (1998) JOINT conditional-coverage LR test on a 0/1 violation sequence.

    H0: violations occur at rate `alpha` AND are serially independent. The statistic decomposes
    exactly as LR_cc = LR_uc (Kupiec) + LR_ind (Christoffersen independence), so it is chi2(2).
    This is the test the two separate p-values do not give: a VaR can pass unconditional coverage
    while its breaches cluster (a model that is right on average and wrong exactly when it matters),
    or pass independence while systematically over/under-covering. Returns (LR_stat, p_value).
    """
    v = np.asarray(violations, dtype=int)
    if len(v) < 2:
        return np.nan, np.nan
    lr_uc, _ = kupiec_pof_test(int(v.sum()), int(len(v)), float(alpha))
    lr_ind, _ = christoffersen_independence_test(v)
    if not np.isfinite(lr_uc) or not np.isfinite(lr_ind):
        return np.nan, np.nan
    lr_cc = float(max(lr_uc, 0.0) + max(lr_ind, 0.0))
    return lr_cc, float(chi2.sf(lr_cc, df=2))


def coverage_summary(violations, alpha):
    """Unconditional (Kupiec), independence and joint conditional-coverage results as one dict."""
    v = np.asarray(violations, dtype=int)
    n, x = int(len(v)), int(v.sum())
    lr_k, p_k = kupiec_pof_test(x, n, alpha) if n > 0 else (np.nan, np.nan)
    lr_i, p_i = christoffersen_independence_test(v) if n >= 2 else (np.nan, np.nan)
    lr_cc, p_cc = christoffersen_conditional_coverage_test(v, alpha) if n >= 2 else (np.nan, np.nan)
    return {"observations": n, "violations": x, "expected": alpha * n,
            "violation_rate": (x / n) if n else np.nan,
            "kupiec_LR": lr_k, "kupiec_p": p_k,
            "indep_LR": lr_i, "indep_p": p_i,
            "cc_LR": lr_cc, "cc_p": p_cc}


def sample_expected_shortfall(sample, alpha=0.05):
    """ES at level `alpha` as a POSITIVE loss: minus the mean of the sample below its alpha-quantile.

    Matches the VaR convention used by the engine (`VaR = -quantile(alpha)`), so ES >= VaR >= 0 for
    a loss-bearing distribution.
    """
    s = np.asarray(sample, dtype=float)
    if s.size == 0:
        return np.nan
    q = float(np.quantile(s, alpha))
    tail = s[s <= q]
    return float(-tail.mean()) if tail.size > 0 else float(-q)


def _as_z_statistics(X, V, E, alpha, which):
    """Vectorized Acerbi-Szekely Z1/Z2 over a matrix of return paths X (n_draws, T)."""
    ind = X < -V[None, :]
    contrib = np.where(ind, X / E[None, :], 0.0)
    if which == 2:
        T = X.shape[1]
        return contrib.sum(axis=1) / (T * alpha) + 1.0
    counts = ind.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z1 = contrib.sum(axis=1) / np.maximum(counts, 1) + 1.0
    return np.where(counts > 0, z1, np.nan)


def acerbi_szekely_es_test(realized, var_levels, es_levels, scenario_matrix, alpha=0.05,
                           n_sim=2000, rng=None, which=2):
    """Acerbi-Szekely (2014) expected-shortfall backtest, Test 1 (which=1) or Test 2 (which=2).

    The portfolio layer minimizes CVaR, so validating only the VaR leaves the objective function
    itself untested. ES has no elicitability, hence no scoring-function backtest -- but it does admit
    these direct tests, because the engine emits the FULL predictive distribution at each as-of date,
    not just a quantile.

        I_t = 1{X_t < -VaR_t}
        Z2  = (1/(T*alpha)) * sum_t X_t * I_t / ES_t + 1     (unconditional: severity AND frequency)
        Z1  = (1/N_viol)    * sum_t X_t * I_t / ES_t + 1     (conditional: severity GIVEN a breach)

    Under H0 (the predictive distribution is correct) E[Z] = 0. A NEGATIVE realized Z means losses in
    the tail were worse than the model's own ES -- the ES is understated. The null distribution is
    obtained by resampling, for each date t, a return from the SAME predictive scenario set the engine
    produced at t (rows of `scenario_matrix`), holding VaR_t and ES_t fixed: no distributional
    assumption is imposed. The p-value is one-sided on the left (P[Z_sim <= Z_obs]), the direction
    that matters for risk. Z1 is undefined with zero breaches (nothing to condition on); Z2 is not,
    and with zero breaches Z2 = 1 > 0, i.e. evidence of a conservative, not an understated, ES.
    """
    X = np.asarray(realized, dtype=float)
    V = np.asarray(var_levels, dtype=float)
    E = np.asarray(es_levels, dtype=float)
    S = np.asarray(scenario_matrix, dtype=float)
    T = X.size
    out = {"statistic": np.nan, "p_value": np.nan, "n_obs": T, "n_violations": int((X < -V).sum()),
           "null_mean": np.nan, "n_sim": int(n_sim), "which": int(which)}
    if T == 0 or S.ndim != 2 or S.shape[0] != T or np.any(~np.isfinite(E)) or np.any(E <= 0):
        return out
    z_obs = float(_as_z_statistics(X[None, :], V, E, alpha, which)[0])
    if not np.isfinite(z_obs):
        return out                                   # Z1 with zero breaches: undefined by construction
    rng = np.random.default_rng(0) if rng is None else rng
    idx = rng.integers(0, S.shape[1], size=(int(n_sim), T))      # (n_sim, T) column-wise resample
    Xs = S[np.arange(T)[None, :], idx]                           # draw t-th return from t-th scenario set
    z_sim = _as_z_statistics(Xs, V, E, alpha, which)
    z_sim = z_sim[np.isfinite(z_sim)]
    if z_sim.size == 0:
        return out
    out["statistic"] = z_obs
    out["p_value"] = float((np.sum(z_sim <= z_obs) + 1.0) / (z_sim.size + 1.0))
    out["null_mean"] = float(z_sim.mean())
    return out


