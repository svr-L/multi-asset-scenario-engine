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


def _build_change_factor_model(changes, explained_var_target, prefix, mean_key, max_factors=None):
    """Generic PCA factor model for a panel of daily changes (yield changes or spread changes).

    Shared by the rates curve and the credit-spread curve (issue: a single asset-class adds a new
    risk-driver block of exactly this shape). Returns scores/loadings/mean plus explained variance.

    `max_factors` (Option A, credit parsimony) hard-caps the retained factors. The rating-bucket OAS
    curve is highly collinear, so its PCA is near-degenerate: keeping all buckets injects ~7 weakly
    identified driver dimensions into an already large joint VAR. Capping to a few systematic factors
    (PC1 ~ credit level, PC2 ~ IG-HY slope, PC3 ~ curvature) keeps essentially all the variance while
    removing parameters; PC1 is also the systematic credit factor a later 1-year CreditMetrics view
    would reuse. The floor is 2 so level + slope are always present.
    """
    mean_chg = changes.mean()
    centered = changes - mean_chg

    pca_full = PCA().fit(centered)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, explained_var_target) + 1)
    k = max(2, min(k, centered.shape[1]))
    if max_factors is not None:
        k = max(2, min(k, int(max_factors)))

    pca = PCA(n_components=k)
    scores = pd.DataFrame(
        pca.fit_transform(centered),
        index=centered.index,
        columns=[f"{prefix}_pc{i+1}" for i in range(k)],
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=centered.columns,
        columns=scores.columns,
    )
    explained_ratio = pd.Series(
        pca.explained_variance_ratio_,
        index=scores.columns,
        name="explained_variance_ratio",
    )
    # What the k retained factors do NOT reproduce, day by day (v22): the compression residual. Kept
    # with the block; the simulator only uses it when CREDIT_RESIDUAL_MODE="restore" attaches it.
    residuals = centered - scores.to_numpy() @ loadings.to_numpy().T
    return {
        "scores": scores,
        "loadings": loadings,
        mean_key: mean_chg,
        "explained_ratio": explained_ratio,
        "cum_explained_ratio": explained_ratio.cumsum(),
        "n_factors": k,
        "pca": pca,
        "residuals": residuals,
    }



def build_rates_factor_model(yield_changes, explained_var_target=0.995):
    return _build_change_factor_model(yield_changes, explained_var_target, prefix="rate", mean_key="mean_dy")


def build_credit_factor_model(spread_changes, explained_var_target=0.99, max_factors=None):
    """PCA factor model of the credit-spread curve (issue: credit block; Option A: parsimony cap).

    The rating-bucket OAS curve compresses to a few factors with a clean interpretation: pc1 is the
    overall credit-risk level (spreads widening/tightening together), pc2 the IG-vs-HY slope, etc.
    These credit factors are stacked into the joint state so the equity-credit (Merton-type)
    dependence is carried by the joint innovation bootstrap. `max_factors` caps the retained factors
    (Option A) so the highly collinear curve does not inflate the joint VAR with near-degenerate
    dimensions; the buckets are still all reconstructed from the few factors via the loadings.
    """
    return _build_change_factor_model(spread_changes, explained_var_target, prefix="credit",
                                      mean_key="mean_dspread", max_factors=max_factors)


def build_curve_lsc_factors(key_rates):
    """Interpretable level / slope / curvature proxies, generalized to whatever tenors are present.

    Issue #8: with a richer curve we no longer hard-code DGS2/DGS5/DGS10. Assuming the columns are
    in ascending tenor order: level = average yield, slope = longest - shortest,
    curvature = 2*mid - shortest - longest.
    """
    cols = list(key_rates.columns)
    short, long = cols[0], cols[-1]
    mid = cols[len(cols) // 2]
    level = key_rates.mean(axis=1)
    slope = key_rates[long] - key_rates[short]
    curvature = 2.0 * key_rates[mid] - key_rates[short] - key_rates[long]
    return pd.DataFrame({
        "curve_level": level,
        "curve_slope": slope,
        "curve_curvature": curvature,
    }, index=key_rates.index)


def estimate_credit_bucket_bridge(bucket_panel, spread_changes_2pt, anchor_date, min_obs=120, horizon_bdays=1):
    """Sensitivity bridge from the simulated 2-point Moody's credit panel to the rating-bucket OAS
    curve, for REPRICING only (`CREDIT_BLOCK_SOURCE = "moody_2pt"`).

    The 2-point panel (Aaa/Baa seasoned yields minus the 10Y Treasury) supplies real, never-
    truncated point-in-time credit DYNAMICS; the rolling ~3y bucket OAS panel supplies (a) today's
    bucket ANCHOR and (b) this map of bucket sensitivities Gamma: per bucket, an interceptless OLS
    of daily bucket-OAS changes on the 2-point spread changes over the overlap window. Horizon
    bucket OAS = today's real bucket OAS + Gamma * (simulated 2-pt move). So the bridge lives on
    SENSITIVITIES, not on data: no past bucket history is fabricated, today's anchor is real, and
    the joint credit dynamics come from real long history. Declared limitation: a linear map fitted
    on a calm overlap understates the convex decompression of the low-quality buckets (CCC) in a
    crisis, so bridged bucket dispersion in deep-stress scenarios is conservative-looking on the
    HY end -- the caveat applies to the repricing granularity, not to the credit factor dynamics.
    Returns {"gamma": DataFrame(buckets x 2pt-cols), "current_buckets": Series, "r2": Series,
    "n_overlap": int}.
    """
    bp = bucket_panel.dropna(how="any")
    if bp.shape[0] < min_obs + 1:
        raise ValueError(f"Bucket OAS panel too short for the bridge ({bp.shape[0]} rows).")
    d_b = bp.diff().dropna()
    common = d_b.index.intersection(spread_changes_2pt.index)
    if len(common) < min_obs:
        raise ValueError(f"Bucket/2-point overlap too short for the bridge ({len(common)} rows).")
    X = spread_changes_2pt.loc[common].to_numpy(dtype=float)          # (T, 2)
    Y = d_b.loc[common].to_numpy(dtype=float)                         # (T, n_buckets)
    # v23: optionally fit on overlapping h-day changes (CREDIT_BRIDGE_FREQUENCY="horizon"), capped so
    # the overlap still holds ~20 independent windows. At h = 1 this is the v19-v22 daily fit, exactly.
    h = max(1, min(int(horizon_bdays), len(common) // 20))
    if h > 1:
        Xf = pd.DataFrame(X).rolling(h).sum().dropna().to_numpy()
        Yf = pd.DataFrame(Y).rolling(h).sum().dropna().to_numpy()
    else:
        Xf, Yf = X, Y
    G, _, _, _ = np.linalg.lstsq(Xf, Yf, rcond=None)                  # (2, n_buckets)
    fitted_f = Xf @ G
    ss_res = np.sum((Yf - fitted_f) ** 2, axis=0)
    ss_tot = np.sum(Yf ** 2, axis=0)                                  # interceptless convention
    r2 = 1.0 - np.divide(ss_res, ss_tot, out=np.zeros_like(ss_res), where=ss_tot > 0)
    fitted = X @ G                                                    # daily moves under the fitted gamma
    anchor = bp.loc[:anchor_date].iloc[-1]
    return {
        "gamma": pd.DataFrame(G.T, index=bp.columns, columns=spread_changes_2pt.columns),
        "current_buckets": anchor,
        "r2": pd.Series(r2, index=bp.columns),                       # at the fit frequency
        "n_overlap": int(len(common)),
        "frequency_bdays": int(h),
        # bucket-specific daily moves the 2-point map does not explain (v22, "restore" only)
        "residuals": pd.DataFrame(Y - fitted, index=common, columns=bp.columns),
    }



def build_credit_lsc_factors(credit_spreads):
    """Interpretable credit level / quality-slope / curvature proxies across the rating buckets.

    Columns are assumed ordered by ascending credit risk (AAA -> CCC). level = average spread,
    slope = widest(HY) - tightest(IG), curvature = 2*mid - tight - wide.
    """
    cols = list(credit_spreads.columns)
    tight, wide = cols[0], cols[-1]
    level = credit_spreads.mean(axis=1)
    slope = credit_spreads[wide] - credit_spreads[tight]
    out = {"credit_level": level, "credit_quality_slope": slope}
    if len(cols) >= 3:
        mid = cols[len(cols) // 2]
        out["credit_curvature"] = 2.0 * credit_spreads[mid] - credit_spreads[tight] - credit_spreads[wide]
    return pd.DataFrame(out, index=credit_spreads.index)


def build_fx_factor_block(fx_log_returns, explained_var_target=0.99, max_factors=None):
    """PCA factor block for a panel of daily FX log-returns (one column per currency pair).

    FX rates are NOT a redundant curve like the rating-bucket OAS, so for a small set of pairs this
    is essentially an identity (about one factor per pair); the PCA is kept for uniformity with the
    other blocks and so the block compresses gracefully if many pairs are added later. Unlike the
    rates / credit blocks, where the driver is a level *change* added onto a running level, the FX
    driver is a daily *log-return*: in the simulator it is cumulated over the horizon and then
    exponentiated onto the current spot level. `max_factors` caps the retained factors (None = one
    component budget per pair); the floor is 1 (a single pair yields a single factor).
    """
    mean_ret = fx_log_returns.mean()
    centered = fx_log_returns - mean_ret
    n_pairs = centered.shape[1]

    pca_full = PCA().fit(centered)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, explained_var_target) + 1)
    cap = n_pairs if max_factors is None else min(int(max_factors), n_pairs)
    k = max(1, min(k, cap))

    pca = PCA(n_components=k)
    scores = pd.DataFrame(
        pca.fit_transform(centered),
        index=centered.index,
        columns=[f"fx_pc{i+1}" for i in range(k)],
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=centered.columns,
        columns=scores.columns,
    )
    explained_ratio = pd.Series(
        pca.explained_variance_ratio_,
        index=scores.columns,
        name="explained_variance_ratio",
    )
    return {
        "scores": scores,
        "loadings": loadings,
        "mean_fx_logret": mean_ret,
        "explained_ratio": explained_ratio,
        "cum_explained_ratio": explained_ratio.cumsum(),
        "n_factors": k,
        "pca": pca,
    }


def build_joint_driver_matrix(equity_block, rates_block, credit_block=None, fx_block=None):
    parts = [
        equity_block["factors"],
        equity_block["resid_scores"],
        rates_block["scores"],
    ]
    if credit_block is not None:
        parts.append(credit_block["scores"])
    if fx_block is not None:
        parts.append(fx_block["scores"])
    drivers = pd.concat(parts, axis=1).dropna()
    return drivers


