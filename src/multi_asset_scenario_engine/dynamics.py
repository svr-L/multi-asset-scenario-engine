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


def build_var1_model(drivers):
    return VAR(drivers).fit(maxlags=1, trend="c")


def var_one_step_r2(var_res, drivers):
    """Per-equation in-sample one-step-ahead R^2 of the VAR(1) vs a constant-only baseline.

    Issue #9 diagnostic: if these R^2 are near zero, the VAR(1) mean dynamics add little and the
    simulated risk is effectively coming from the bootstrapped innovations + factor loadings, not
    from genuine cross-predictability. In that case the model could be simplified (drop the VAR and
    block-bootstrap the demeaned joint innovations) or regularized (Minkowski/ridge / block prior).
    """
    resid = var_res.resid
    y = drivers.loc[resid.index]
    r2 = 1.0 - resid.var(ddof=0) / y.var(ddof=0)
    return r2.sort_values(ascending=False)


def stationary_bootstrap_index_matrix(n_obs, n_paths, horizon, avg_block_length, rng):
    """Vectorized stationary bootstrap (Politis-Romano) index matrix, shape (n_paths, horizon).

    Issue #5: replaces the per-path Python loop. Same generative process as before: at each step,
    with probability 1/avg_block_length restart at a fresh uniform index, otherwise advance the
    previous index by one (mod n_obs). Only `horizon` vectorized iterations are needed.
    """
    p_restart = 1.0 / avg_block_length
    restart = rng.random((n_paths, horizon)) < p_restart
    restart[:, 0] = True
    fresh = rng.integers(0, n_obs, size=(n_paths, horizon))
    idx = np.empty((n_paths, horizon), dtype=int)
    idx[:, 0] = fresh[:, 0]
    for h in range(1, horizon):
        advanced = (idx[:, h - 1] + 1) % n_obs
        idx[:, h] = np.where(restart[:, h], fresh[:, h], advanced)
    return idx


def _flat_top_kernel(x):
    """Politis (2003) flat-top lag window: 1 on [0, 1/2], linearly down to 0 on [1/2, 1], 0 beyond."""
    w = np.ones_like(x, dtype=float)
    mid = (x >= 0.5) & (x <= 1.0)
    w[mid] = 2.0 * (1.0 - x[mid])
    w[x > 1.0] = 0.0
    return w


def politis_white_block_length(x):
    """Automatic block-length selection for the stationary bootstrap (Politis & White 2004, with the
    Patton-Politis-White 2009 correction), computed per column.

    Returns the optimal *expected* block length per column. The optimum is
    b = (2 G^2 / D_SB)^{1/3} n^{1/3} with G = sum_k lam(k/M)|k|R(k), D_SB = 2 (sum_k lam(k/M)R(k))^2,
    R the autocovariance and M a flat-top bandwidth chosen where the autocorrelation becomes
    negligible. Because the procedure reads the *level* autocovariance, applying it to returns gives a
    short block (returns are nearly serially uncorrelated) while applying it to the absolute/squared
    series captures volatility clustering -- which is exactly why, after the FHS filter has removed the
    clustering, the standardized residuals admit a short block.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, k = x.shape
    kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    bmax = float(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))
    cval = 2.0 * np.sqrt(np.log10(n) / n)
    out = np.empty(k)
    for j in range(k):
        e = x[:, j] - x[:, j].mean()
        acv = np.array([np.sum(e[:n - lag] * e[lag:]) / n for lag in range(mmax + 1)])
        if acv[0] <= 0:
            out[j] = 1.0
            continue
        rho = acv / acv[0]
        sig = np.abs(rho) > cval
        mhat = None
        for mm in range(1, mmax - kn + 1):
            if not sig[mm + 1:mm + 1 + kn].any():
                mhat = mm
                break
        if mhat is None:
            mhat = mmax
        M = int(min(2 * mhat, mmax))
        lags = np.arange(-M, M + 1)
        w = _flat_top_kernel(np.abs(lags) / M)
        g_acv = acv[np.abs(lags)]
        g_hat = np.sum(w * np.abs(lags) * g_acv)
        g0 = np.sum(w * g_acv)
        if g0 <= 0 or g_hat == 0:
            out[j] = 1.0
        else:
            b = (2.0 * g_hat ** 2 / (2.0 * g0 ** 2)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
            out[j] = float(min(max(b, 1.0), bmax))
    return out


def _nearest_pd_correlation(corr):
    """Project a symmetric matrix to a positive-definite correlation matrix (eigenvalue clipping).

    The Kendall-tau -> R = sin(pi*tau/2) transform is applied element-wise and need not stay PD, so
    we clip the eigenvalues to a small floor and rescale the diagonal back to one.
    """
    c = np.asarray(corr, dtype=float)
    c = 0.5 * (c + c.T)
    vals, vecs = np.linalg.eigh(c)
    vals = np.clip(vals, 1e-8, None)
    c_pd = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(c_pd))
    c_pd = c_pd / np.outer(d, d)
    np.fill_diagonal(c_pd, 1.0)
    return c_pd


def _t_copula_loglik(tq, R_inv, logdet_R, nu, d):
    """t-copula log-likelihood at pseudo-observations whose univariate-t quantiles are tq (n, d)."""
    maha = np.einsum("ni,ij,nj->n", tq, R_inv, tq)
    mvt = (gammaln((nu + d) / 2.0) - gammaln(nu / 2.0) - 0.5 * d * np.log(nu * np.pi)
           - 0.5 * logdet_R - 0.5 * (nu + d) * np.log1p(maha / nu))
    univ = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
            - 0.5 * (nu + 1.0) * np.log1p(tq ** 2 / nu)).sum(axis=1)
    return float(np.sum(mvt - univ))


def t_copula_tail_dependence(rho, nu):
    """Lower/upper tail-dependence coefficient of a bivariate t-copula with correlation rho, dof nu.

    lambda = 2 * t_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ). Unlike the Gaussian copula (lambda = 0 for
    any rho < 1), the t-copula has positive tail dependence that grows as nu falls -- the probability
    that two drivers are jointly extreme does not vanish in the tail.
    """
    rho = np.clip(rho, -0.999, 0.999)
    arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return 2.0 * student_t.cdf(arg, nu + 1.0)


def fit_t_copula(drivers, nu_grid=None):
    """Fit a Student-t copula to the joint driver innovations (a tail-dependent dependence model).

    A copula separates the *marginals* (kept empirical here, so each driver's own distribution is
    preserved exactly) from the *dependence structure*. Unlike a Gaussian copula, the t-copula has
    non-zero tail dependence: joint extreme moves co-occur more often than linear correlation implies,
    with the strength set by the degrees of freedom nu (low nu -> strong tail dependence; nu -> inf
    recovers the Gaussian copula). This is the property the block bootstrap lacks -- it can only
    resample observed joint vectors, whereas the t-copula can generate new joint extremes consistent
    with the fitted tail-dependent structure.

    The correlation matrix is estimated from Kendall's tau (R = sin(pi*tau/2)), which is rank-based
    and invariant to the marginals; nu is fitted by profile maximum-likelihood over `nu_grid`. The
    demeaned empirical marginals are stored (sorted) for the inversion step at simulation time.
    """
    X = np.asarray(drivers, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean
    n, d = Xc.shape
    tau = pd.DataFrame(Xc).corr(method="kendall").to_numpy()
    R = _nearest_pd_correlation(np.sin(np.pi / 2.0 * tau))
    R_inv = np.linalg.inv(R)
    _, logdet_R = np.linalg.slogdet(R)
    ranks = pd.DataFrame(Xc).rank().to_numpy()
    U = np.clip(ranks / (n + 1.0), 1e-9, 1 - 1e-9)
    if nu_grid is None:
        nu_grid = [2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0, 50.0]
    ll_by_nu = {}
    best_nu, best_ll = None, -np.inf
    for nu in nu_grid:
        ll = _t_copula_loglik(student_t.ppf(U, nu), R_inv, logdet_R, nu, d)
        ll_by_nu[float(nu)] = ll
        if ll > best_ll:
            best_ll, best_nu = ll, float(nu)
    # average off-diagonal correlation and the implied average pairwise tail dependence
    off = R[~np.eye(d, dtype=bool)]
    avg_abs_corr = float(np.mean(np.abs(off)))
    avg_tail_dep = float(np.mean([t_copula_tail_dependence(r, best_nu) for r in off[off > 0]])) if np.any(off > 0) else 0.0
    gaussian_ll = ll_by_nu.get(50.0, _t_copula_loglik(student_t.ppf(U, 50.0), R_inv, logdet_R, 50.0, d))
    return {
        "R": R, "chol_R": np.linalg.cholesky(R), "nu": best_nu,
        "loglik": best_ll, "loglik_by_nu": ll_by_nu, "gaussian_loglik": gaussian_ll,
        "avg_abs_corr": avg_abs_corr, "avg_tail_dependence": avg_tail_dep,
        "mean": mean, "marg_sorted": np.sort(Xc, axis=0), "n_obs": n, "dim": d,
        "columns": list(drivers.columns) if hasattr(drivers, "columns") else None,
    }


def sample_t_copula_innovations(copula, n_samples, rng):
    """Draw demeaned joint innovations from a fitted t-copula (empirical marginals + t dependence).

    Multivariate-t draw (Z ~ N(0,R), W ~ chi2(nu): T = Z / sqrt(W/nu)) -> Student-t CDF to uniforms
    carrying the t-copula's tail dependence -> inverted through each driver's empirical (demeaned)
    marginal by interpolating its sorted values at the implied rank position.
    """
    chol = copula["chol_R"]; nu = copula["nu"]; marg = copula["marg_sorted"]
    n_obs, d = marg.shape
    z = rng.standard_normal((n_samples, d)) @ chol.T
    w = rng.chisquare(nu, size=n_samples) / nu
    tvar = z / np.sqrt(w)[:, None]
    u = np.clip(student_t.cdf(tvar, nu), 1e-9, 1 - 1e-9)
    pos = u * (n_obs - 1)
    lo = np.floor(pos).astype(int)
    hi = np.clip(lo + 1, 0, n_obs - 1)
    frac = pos - lo
    out = np.empty((n_samples, d), dtype=float)
    for j in range(d):
        col = marg[:, j]
        out[:, j] = col[lo[:, j]] * (1.0 - frac[:, j]) + col[hi[:, j]] * frac[:, j]
    return out


def fit_asset_copula(equity_log_returns, yield_changes, spread_changes, fx_log_returns=None, nu_grid=None):
    """Fit a Student-t copula at the OBSERVABLE (asset / risk-driver) level, not on the PCA factors.

    The factor-level copula (v11) had little tail dependence to capture: the PCA factors are nearly
    uncorrelated by construction, so the joint extremes of the *factors* are weak, while the equity
    co-tail is really a shared-loadings phenomenon the bootstrap reproduces by resampling. Fitting the
    copula directly on the daily observables -- equity log-returns, key-rate changes (highly
    correlated across the curve), rating-bucket spread changes (highly correlated), and the FX
    log-return -- puts the copula where the real cross-sectional correlation, and hence the real tail
    dependence, lives. Generation is then reduced-form: a draw is a full daily observable innovation
    (each variable's empirical marginal + the t-copula's tail-dependent joint structure), assembled
    directly into equity/rate/credit/FX scenarios with no factor reconstruction. The trade is that it
    drops the parsimonious factor structure -- it is a parallel, dependence-faithful generator to
    compare against the factor engine, not a replacement for it.
    """
    blocks = [("equity", equity_log_returns), ("rates", yield_changes), ("credit", spread_changes)]
    if fx_log_returns is not None:
        blocks.append(("fx", fx_log_returns))
    idx = None
    for _, df in blocks:
        idx = df.index if idx is None else idx.intersection(df.index)
    panel_parts, groups, start = [], {}, 0
    for name, df in blocks:
        d = df.loc[idx]
        cols = list(d.columns)
        groups[name] = {"slice": slice(start, start + len(cols)), "columns": cols}
        start += len(cols)
        panel_parts.append(d)
    panel = pd.concat(panel_parts, axis=1)
    cop = fit_t_copula(panel, nu_grid=nu_grid)
    cop["asset_groups"] = groups
    cop["panel_columns"] = list(panel.columns)
    return cop


def _garch11_nll_grid(r2, uncond_var, alphas, persistences):
    """Vectorized Gaussian negative log-likelihood (x2, up to an additive constant) of GARCH(1,1)
    candidates under variance targeting: omega = (1 - p) * uncond_var, beta = p - alpha. Evaluates all
    (alpha, p) candidates simultaneously with a single pass over time; differences between entries are
    likelihood-ratio statistics."""
    A = np.asarray(alphas, dtype=float)
    P = np.asarray(persistences, dtype=float)
    B = P - A
    W = (1.0 - P) * uncond_var
    floor = 1e-12 * max(float(uncond_var), 1e-300)
    s2 = np.full(A.shape, float(uncond_var))
    nll = np.zeros(A.shape)
    for t in range(r2.shape[0]):
        s2 = np.maximum(s2, floor)
        nll += np.log(s2) + r2[t] / s2
        s2 = W + A * r2[t] + B * s2
    return nll


def estimate_garch11_params(series, uncond_var=None, persistence_cap=0.998, min_lr=6.0, min_alpha=0.02):
    """GARCH(1,1) parameters for ONE series by Gaussian quasi-ML with variance targeting.

    Variance targeting pins the long-run variance to the sample variance (omega = (1-p)*uncond), which
    is exactly what the FHS forward recursion needs (mean reversion to the unconditional level) and
    leaves two free parameters: the news weight alpha and the persistence p = alpha + beta. These are
    found on a coarse (alpha, p) grid evaluated in one vectorized pass, then refined on a local fine
    grid -- deterministic and robust on ~700 observations, where a full MLE adds precision the data
    cannot support. Persistence is capped at `persistence_cap` (0.998 ~ a 350-business-day variance
    half-life): a 3-year sample cannot distinguish 0.998 from 1, and the cap keeps the one-year
    horizon mean-reverting rather than freezing today's volatility. If the likelihood gain over
    constant variance is below `min_lr` (default 6, roughly the chi-squared(2) 5% point, used as an
    indicative threshold since the null puts alpha on the boundary) or the fitted news weight is
    negligible, the series shows no
    volatility clustering and the function falls back to constant variance (alpha = beta = 0), so the
    FHS filter conditions only where the data supports it. Gaussian QML point estimates remain
    consistent under fat-tailed innovations (Bollerslev-Wooldridge), which is all we use them for.
    """
    e = np.asarray(series, dtype=float)
    e = e - e.mean()
    r2 = e ** 2
    T = r2.shape[0]
    uv = float(np.var(e)) if uncond_var is None else float(uncond_var)
    out_const = {"alpha": 0.0, "beta": 0.0, "persistence": 0.0,
                 "half_life_days": np.nan, "lr_vs_const": 0.0}
    if uv <= 0.0 or T < 60:
        return out_const
    nll_const = T * np.log(uv) + float(np.sum(r2)) / uv
    a_coarse = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30])
    p_coarse = np.array([0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98,
                         0.985, 0.99, 0.993, 0.995, 0.9965, persistence_cap])
    cand = [(a, p) for a in a_coarse for p in p_coarse if p >= a]
    A = np.array([c[0] for c in cand]); P = np.array([c[1] for c in cand])
    nll = _garch11_nll_grid(r2, uv, A, P)
    i = int(np.argmin(nll)); a_best, p_best, nll_best = float(A[i]), float(P[i]), float(nll[i])
    a_fine = np.clip(a_best + np.linspace(-0.02, 0.02, 9), 0.003, 0.35)
    p_fine = np.clip(p_best + np.linspace(-0.006, 0.006, 13), 0.85, persistence_cap)
    cand2 = [(a, p) for a in np.unique(a_fine) for p in np.unique(p_fine) if p >= a]
    A2 = np.array([c[0] for c in cand2]); P2 = np.array([c[1] for c in cand2])
    nll2 = _garch11_nll_grid(r2, uv, A2, P2)
    j = int(np.argmin(nll2))
    if float(nll2[j]) < nll_best:
        a_best, p_best, nll_best = float(A2[j]), float(P2[j]), float(nll2[j])
    lr = max(0.0, nll_const - nll_best)
    if lr < min_lr or a_best < min_alpha:
        out_const["lr_vs_const"] = lr
        return out_const
    return {"alpha": a_best, "beta": p_best - a_best, "persistence": p_best,
            "half_life_days": float(np.log(0.5) / np.log(p_best)), "lr_vs_const": lr}


def _gjr11_nll_grid(r2, neg, uncond_var, alphas, gammas, persistences):
    """Vectorized Gaussian NLL (x2, up to a constant) of GJR-GARCH(1,1) candidates under variance
    targeting: sigma2_{t+1} = omega + (alpha + gamma*1[eps_t<0]) * eps_t^2 + beta * sigma2_t, with
    persistence p = alpha + gamma/2 + beta (symmetric-innovation approximation) and
    omega = (1 - p) * uncond_var. One pass over time evaluates all candidates."""
    A = np.asarray(alphas, dtype=float)
    G = np.asarray(gammas, dtype=float)
    P = np.asarray(persistences, dtype=float)
    B = P - A - 0.5 * G
    W = (1.0 - P) * uncond_var
    floor = 1e-12 * max(float(uncond_var), 1e-300)
    s2 = np.full(A.shape, float(uncond_var))
    nll = np.zeros(A.shape)
    for t in range(r2.shape[0]):
        s2 = np.maximum(s2, floor)
        nll += np.log(s2) + r2[t] / s2
        s2 = W + (A + G * neg[t]) * r2[t] + B * s2
    return nll


def estimate_gjr_garch11_params(series, uncond_var=None, persistence_cap=0.998,
                                min_lr=6.0, min_news=0.02, min_lr_asym=3.84):
    """GJR-GARCH(1,1) parameters for ONE series by Gaussian QML with variance targeting.

    Extends the symmetric estimator with the Glosten-Jagannathan-Runkle leverage term: volatility may
    react MORE to negative shocks (gamma > 0), the equity stylized fact that acts specifically on the
    left tail. Model selection is nested and evidence-gated: the best symmetric fit (gamma = 0) is
    kept unless the asymmetric term improves the likelihood by at least `min_lr_asym` (~chi-sq(1) 5%,
    indicative -- the null puts gamma on a boundary); if not even the best fit beats constant variance
    by `min_lr`, the factor falls back to constant variance. The news gate is on the TOTAL news
    weight alpha + gamma/2 (an index-equity GJR often has alpha ~ 0 with all news arriving through
    gamma, which must not be gated away). Persistence p = alpha + gamma/2 + beta is capped at
    `persistence_cap`; the p = alpha + gamma/2 + beta identity uses the symmetric-innovation
    approximation E[1[eps<0] eps^2] = sigma^2 / 2, standard for variance targeting.
    """
    e = np.asarray(series, dtype=float)
    e = e - e.mean()
    r2 = e ** 2
    neg = (e < 0.0).astype(float)
    T = r2.shape[0]
    uv = float(np.var(e)) if uncond_var is None else float(uncond_var)
    out_const = {"alpha": 0.0, "gamma": 0.0, "beta": 0.0, "persistence": 0.0,
                 "half_life_days": np.nan, "lr_vs_const": 0.0, "lr_asym": 0.0}
    if uv <= 0.0 or T < 60:
        return out_const
    nll_const = T * np.log(uv) + float(np.sum(r2)) / uv
    a_g = np.array([0.0, 0.01, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16])
    g_g = np.array([0.0, 0.02, 0.04, 0.07, 0.10, 0.14, 0.19, 0.25])
    p_g = np.array([0.90, 0.93, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.994, persistence_cap])
    cand = [(a, g, p) for a in a_g for g in g_g for p in p_g
            if (p - a - 0.5 * g) >= 0.0 and (a + g) > 0.0]
    A = np.array([c[0] for c in cand]); G = np.array([c[1] for c in cand]); P = np.array([c[2] for c in cand])
    nll = _gjr11_nll_grid(r2, neg, uv, A, G, P)

    def _refine(a0, g0, p0, free_gamma=True):
        af = np.unique(np.clip(a0 + np.linspace(-0.02, 0.02, 5), 0.0, 0.35))
        gf = np.unique(np.clip(g0 + np.linspace(-0.03, 0.03, 7), 0.0, 0.45)) if free_gamma else np.array([g0])
        pf = np.unique(np.clip(p0 + np.linspace(-0.004, 0.004, 7), 0.85, persistence_cap))
        c2 = [(a, g, p) for a in af for g in gf for p in pf
              if (p - a - 0.5 * g) >= 0.0 and (a + g) > 0.0]
        if not c2:
            return a0, g0, p0, np.inf
        A2 = np.array([c[0] for c in c2]); G2 = np.array([c[1] for c in c2]); P2 = np.array([c[2] for c in c2])
        n2 = _gjr11_nll_grid(r2, neg, uv, A2, G2, P2)
        j = int(np.argmin(n2))
        return float(A2[j]), float(G2[j]), float(P2[j]), float(n2[j])

    sym_mask = G == 0.0
    i_s = int(np.argmin(np.where(sym_mask, nll, np.inf)))
    i_a = int(np.argmin(nll))
    a_s, g_s, p_s, n_s = _refine(float(A[i_s]), 0.0, float(P[i_s]), free_gamma=False)
    a_a, g_a, p_a, n_a = _refine(float(A[i_a]), float(G[i_a]), float(P[i_a]), free_gamma=True)
    if n_a > n_s:
        a_a, g_a, p_a, n_a = a_s, g_s, p_s, n_s
    lr_all = max(0.0, nll_const - n_a)
    lr_sym = max(0.0, nll_const - n_s)
    lr_asym = max(0.0, n_s - n_a)
    if max(lr_all, lr_sym) < min_lr:
        out_const["lr_vs_const"] = lr_all
        return out_const
    if lr_asym >= min_lr_asym and g_a > 0.0:
        a_b, g_b, p_b, lr_b = a_a, g_a, p_a, lr_all
    else:
        a_b, g_b, p_b, lr_b = a_s, 0.0, p_s, lr_sym
    if (a_b + 0.5 * g_b) < min_news:
        out_const["lr_vs_const"] = lr_b
        out_const["lr_asym"] = lr_asym
        return out_const
    return {"alpha": a_b, "gamma": g_b, "beta": max(p_b - a_b - 0.5 * g_b, 0.0),
            "persistence": p_b, "half_life_days": float(np.log(0.5) / np.log(p_b)),
            "lr_vs_const": lr_b, "lr_asym": lr_asym}


def estimate_ou_level_params(levels, tstat_gate=-2.86, min_obs=120):
    """Per-series Vasicek/OU discretization fitted on LEVELS: Delta y_t = a + b * y_{t-1} + eps_t,
    so kappa = -b (daily pull), theta = -a / b (long-run level), half-life = ln(2) / kappa.

    Rates and credit spreads mean-revert over long horizons for economic reasons; equity returns do
    not, so this applies to LEVEL blocks only. A series is gated IN only when b < 0 with a t-statistic
    below `tstat_gate` -- default -2.86, the Dickey-Fuller 5% critical value (with constant), because
    under the unit-root null the OLS t-stat is NOT t-distributed and an ordinary -2 gate would find
    spurious mean reversion on short samples. Non-gated series keep the random walk exactly
    (kappa = 0 and the RW drift is retained), so reversion is applied only where the data supports it.
    Returns per-column arrays: kappa, theta, half_life_days, tstat, gated (0/1 floats).
    """
    L = np.asarray(levels, dtype=float)
    T, n = L.shape
    kappa = np.zeros(n); theta = np.zeros(n)
    tst = np.zeros(n); hl = np.full(n, np.nan); gated = np.zeros(n)
    if T < min_obs:
        return {"kappa": kappa, "theta": theta, "half_life_days": hl, "tstat": tst, "gated": gated}
    for j in range(n):
        y = L[:, j]
        x = y[:-1]; dy = np.diff(y)
        xm = x.mean(); dm = dy.mean()
        sxx = float(np.sum((x - xm) ** 2))
        if sxx <= 0.0:
            continue
        b = float(np.sum((x - xm) * (dy - dm)) / sxx)
        a = dm - b * xm
        resid = dy - a - b * x
        s2 = float(np.sum(resid ** 2)) / max(len(dy) - 2, 1)
        se_b = np.sqrt(s2 / sxx) if s2 > 0 else 0.0
        t_b = b / se_b if se_b > 0 else 0.0
        tst[j] = t_b
        if b < 0.0 and t_b <= tstat_gate:
            kappa[j] = -b
            theta[j] = -a / b
            hl[j] = float(np.log(2.0) / kappa[j])
            gated[j] = 1.0
    return {"kappa": kappa, "theta": theta, "half_life_days": hl, "tstat": tst, "gated": gated}


def compute_ewma_fhs_filter(drivers, driver_mean, ewma_lambda=0.94, reversion_halflife=34.0,
                            params_mode="fixed"):
    """Filtered Historical Simulation (FHS) volatility filter at the FACTOR level.

    Standardizes each joint driver by its own time-varying conditional volatility so the resampled
    innovations can be re-scaled by *today's* volatility -- making the simulated risk conditional on
    the current regime while keeping the empirical (fat-tailed) shape of the standardized residuals
    that made the block bootstrap win over the copulas. Crucially this is done at the FACTOR level:
    the conditional CORRELATION between assets is not modelled directly but EMERGES from the
    conditional volatility of the shared factors -- when the common market factor's volatility is high
    today, its contribution to every asset is scaled up together, so cross-asset correlation rises in
    stress (the empirical 'correlations go to one in a crash') without ever estimating an asset
    correlation matrix.

    The conditional variance follows a GARCH(1,1) in RiskMetrics form: the news weight is
    1 - ewma_lambda (so ewma_lambda is the usual EWMA decay) but the persistence is set below 1 via
    `reversion_halflife`, so the variance MEAN-REVERTS to its unconditional level. Pure EWMA (infinite
    half-life) forecasts flat forward, which over a one-year horizon would freeze today's vol; mean
    reversion lets it wash out over ~the half-life -- and, as a bonus, supplies the low-frequency
    variance mean reversion the daily VAR could not see.

    Returns the standardized residuals (unit unconditional variance per factor), the one-step-ahead
    conditional variance for the first simulated day, and the (omega, alpha, beta) the simulator rolls
    forward (driven by the simulated innovations, so volatility clusters along each path).
    """
    R = np.asarray(drivers, dtype=float) - np.asarray(driver_mean, dtype=float)[None, :]
    T, k = R.shape
    uncond_var = R.var(axis=0)
    per_factor = None
    gamma = 0.0
    if params_mode in ("estimated", "estimated_gjr"):
        # Per-factor (GJR-)GARCH(1,1) QML (variance targeting): each factor gets its OWN news weight
        # and persistence -- and, in "estimated_gjr", its own leverage term gamma (kept only where the
        # asymmetric fit beats the symmetric one). No-clustering factors come back with
        # alpha = gamma = beta = 0 -> constant variance (the filter is inert for them).
        if params_mode == "estimated_gjr":
            _est = [estimate_gjr_garch11_params(R[:, j], uncond_var=float(uncond_var[j])) for j in range(k)]
            gamma = np.array([e_["gamma"] for e_ in _est])
        else:
            _est = [estimate_garch11_params(R[:, j], uncond_var=float(uncond_var[j])) for j in range(k)]
        alpha = np.array([e_["alpha"] for e_ in _est])
        persistence = np.array([e_["persistence"] for e_ in _est])
        beta = np.maximum(persistence - alpha - 0.5 * np.asarray(gamma), 0.0)
        per_factor = {
            "alpha": alpha.copy(),
            "persistence": persistence.copy(),
            "half_life_days": np.array([e_["half_life_days"] for e_ in _est]),
            "lr_vs_const": np.array([e_["lr_vs_const"] for e_ in _est]),
        }
        if params_mode == "estimated_gjr":
            per_factor["gamma"] = np.asarray(gamma).copy()
            per_factor["lr_asym"] = np.array([e_["lr_asym"] for e_ in _est])
    else:
        alpha = 1.0 - ewma_lambda
        persistence = 0.5 ** (1.0 / reversion_halflife)        # < 1 -> mean reversion in variance
        beta = persistence - alpha
        if beta < 0.0:                                         # guard for a very short half-life
            beta = 0.0
            persistence = alpha
    omega = (1.0 - persistence) * uncond_var                   # long-run variance == unconditional
    sigma2 = np.empty((T, k))
    sigma2[0] = uncond_var
    _use_gjr = bool(np.any(np.asarray(gamma) != 0.0))
    if _use_gjr:
        _neg = (R < 0.0).astype(float)
        for t in range(1, T):
            sigma2[t] = omega + (alpha + gamma * _neg[t - 1]) * R[t - 1] ** 2 + beta * sigma2[t - 1]
    else:
        for t in range(1, T):
            sigma2[t] = omega + alpha * R[t - 1] ** 2 + beta * sigma2[t - 1]
    z = R / np.sqrt(sigma2)
    z = z / z.std(axis=0)                                      # exact unit unconditional variance
    if _use_gjr:
        cond_var_now = omega + (alpha + gamma * _neg[T - 1]) * R[T - 1] ** 2 + beta * sigma2[T - 1]
    else:
        cond_var_now = omega + alpha * R[T - 1] ** 2 + beta * sigma2[T - 1]
    return {
        "standardized_residuals": z,
        "cond_var_now": cond_var_now,
        "omega": uncond_var * (1.0 - persistence),
        "alpha": alpha if np.ndim(alpha) else float(alpha),
        "beta": beta if np.ndim(beta) else float(beta),
        "gamma": gamma if np.ndim(gamma) else float(gamma),
        "uncond_var": uncond_var,
        "persistence": persistence if np.ndim(persistence) else float(persistence),
        "current_vol_ratio": np.sqrt(cond_var_now / uncond_var),   # >1 stressed, <1 calm (per factor)
        "params_mode": params_mode,
        "per_factor": per_factor,
    }


