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

from .diagnostics import loss_summary_from_pnl

def rolling_horizon_log_returns(log_returns, window):
    return log_returns.rolling(window).sum().dropna()


def rolling_horizon_yield_changes(yield_changes, window):
    return yield_changes.rolling(window).sum().dropna()


def portfolio_metrics_from_instrument_returns(weights, instrument_returns, alpha=0.05):
    w = pd.Series(weights, index=instrument_returns.columns, dtype=float)
    w = w / w.sum()
    portfolio_return = instrument_returns.mul(w, axis=1).sum(axis=1)
    metrics = loss_summary_from_pnl(portfolio_return, alpha=alpha)
    metrics["mean_abs_weight"] = w.abs().mean()
    return portfolio_return, metrics


def inverse_vol_weights_from_scenarios(instrument_returns):
    vol = instrument_returns.std()
    inv_vol = 1.0 / vol.replace(0, np.nan)
    w = inv_vol / inv_vol.sum()
    return w.fillna(0.0)


def tail_contribution_table(portfolio_return, instrument_returns, weights, alpha=0.05):
    w = pd.Series(weights, index=instrument_returns.columns, dtype=float)
    w = w / w.sum()
    threshold = portfolio_return.quantile(alpha)
    tail_mask = portfolio_return <= threshold
    if tail_mask.sum() == 0:
        return pd.DataFrame()

    weighted_return_contrib = instrument_returns.mul(w, axis=1)
    avg_tail_return_contrib = weighted_return_contrib.loc[tail_mask].mean()
    avg_tail_loss_contrib = -avg_tail_return_contrib
    portfolio_es_loss = float(avg_tail_loss_contrib.sum())
    share = avg_tail_loss_contrib / portfolio_es_loss if portfolio_es_loss != 0 else np.nan

    return pd.DataFrame({
        "weight": w,
        "avg_tail_return_contribution": avg_tail_return_contrib,
        "avg_tail_loss_contribution": avg_tail_loss_contrib,
        "share_of_portfolio_ES_loss": share,
    }).sort_values("avg_tail_loss_contribution", ascending=False)


def search_long_only_min_cvar_portfolio(instrument_returns, alpha=0.05, max_weight=None,
                                        n_draws=None, seed=None, chunk_size=None):
    """Exact long-only minimum-CVaR portfolio via the Rockafellar-Uryasev LP (issue #3).

    Replaces the previous random simplex search with the convex linear program

        min_{w, zeta, u}   zeta + 1/(alpha * S) * sum_s u_s
        s.t.   u_s >= -(r_s . w) - zeta,   u_s >= 0
               sum_i w_i = 1,   0 <= w_i <= max_weight

    where r_s is the instrument-return vector in scenario s and S the number of scenarios. Solved
    with scipy.optimize.linprog(method='highs'). This returns the true optimum, is deterministic
    and reproducible, scales with the number of assets, and does not discard most draws under a
    tight max-weight cap (the old rejection sampler did). The legacy `n_draws / seed / chunk_size`
    arguments are accepted but ignored for backward compatibility.
    """
    R = instrument_returns.to_numpy()
    S, n = R.shape
    names = instrument_returns.columns.tolist()

    c = np.concatenate([np.zeros(n), [1.0], np.full(S, 1.0 / (alpha * S))])

    # inequality block:  -(R w) - zeta - u <= 0   (one row per scenario)
    A_ub = sparse.hstack([
        sparse.csr_matrix(-R),
        sparse.csr_matrix(-np.ones((S, 1))),
        -sparse.identity(S, format="csr"),
    ], format="csr")
    b_ub = np.zeros(S)

    # equality block:  sum w = 1
    A_eq = sparse.hstack([
        sparse.csr_matrix(np.ones((1, n))),
        sparse.csr_matrix((1, 1)),
        sparse.csr_matrix((1, S)),
    ], format="csr")
    b_eq = np.array([1.0])

    w_upper = max_weight if max_weight is not None else 1.0
    bounds = [(0.0, w_upper)] * n + [(None, None)] + [(0.0, None)] * S

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"min-CVaR LP failed to solve: {res.message}")

    w = np.clip(res.x[:n], 0.0, None)
    w = w / w.sum()
    port = R @ w
    q = float(np.quantile(port, alpha))
    tail = port[port <= q]
    es = float(-tail.mean()) if tail.size else np.nan

    weights = pd.Series(w, index=names, name="min_cvar_long_only")
    diagnostics = pd.Series({
        "n_scenarios": float(S),
        "VaR_95_loss": float(-q),
        "ES_95_loss": es,
        "mean": float(port.mean()),
        "std": float(port.std(ddof=1)),
    }, name="min_cvar_lp_diagnostics")
    return weights, diagnostics


