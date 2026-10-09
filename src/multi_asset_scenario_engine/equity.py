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

from .data import compute_excess_log_returns_daily, compute_market_excess_log_returns_daily

def _compress_residual_block(residuals, resid_explained_var_target=0.95):
    resid_pca_full = PCA().fit(residuals)
    resid_cumvar = np.cumsum(resid_pca_full.explained_variance_ratio_)
    k_resid = int(np.searchsorted(resid_cumvar, resid_explained_var_target) + 1)
    k_resid = max(1, min(k_resid, residuals.shape[1] - 1))

    resid_pca = PCA(n_components=k_resid)
    resid_scores = pd.DataFrame(
        resid_pca.fit_transform(residuals),
        index=residuals.index,
        columns=[f"resid_pc{i+1}" for i in range(k_resid)],
    )

    resid_loadings = pd.DataFrame(
        resid_pca.components_.T,
        index=residuals.columns,
        columns=resid_scores.columns,
    )

    resid_explained_ratio = pd.Series(
        resid_pca.explained_variance_ratio_,
        index=resid_scores.columns,
        name="resid_explained_variance_ratio",
    )

    # Issue #11: PCA on the residuals discards the variance beyond the retained components. That
    # dropped idiosyncratic variance never re-enters the simulation, so single-name tails would be
    # understated. We measure, per asset, the standard deviation of the part of the residual NOT
    # captured by the retained PCs, and re-inject it as iid noise in the simulator.
    reconstructed = resid_scores.to_numpy() @ resid_loadings.to_numpy().T
    leftover = residuals.to_numpy() - reconstructed
    resid_idiosyncratic_std = pd.Series(
        leftover.std(axis=0, ddof=1),
        index=residuals.columns,
        name="resid_idiosyncratic_std",
    )

    return {
        "resid_scores": resid_scores,
        "resid_loadings": resid_loadings,
        "resid_explained_ratio": resid_explained_ratio,
        "resid_cum_explained_ratio": resid_explained_ratio.cumsum(),
        "resid_idiosyncratic_std": resid_idiosyncratic_std,
        "resid_pca": resid_pca,
    }


def build_factor_driven_equity_block(excess_returns, factor_df, block_name, resid_explained_var_target=0.95):
    common_dates = excess_returns.index.intersection(factor_df.index)
    y = excess_returns.loc[common_dates].copy()
    f = factor_df.loc[common_dates].copy()

    mu_excess = y.mean()
    mu_factors = f.mean()
    centered_y = y - mu_excess
    centered_f = f - mu_factors

    X = centered_f.to_numpy()
    Y = centered_y.to_numpy()
    beta_array = np.linalg.lstsq(X, Y, rcond=None)[0].T  # assets x factors

    betas = pd.DataFrame(beta_array, index=y.columns, columns=centered_f.columns)
    fitted = pd.DataFrame(X @ beta_array.T, index=y.index, columns=y.columns)
    residuals = centered_y - fitted
    asset_r2 = 1.0 - residuals.var(ddof=1) / centered_y.var(ddof=1)

    residual_block = _compress_residual_block(residuals, resid_explained_var_target)

    return {
        "engine_name": block_name,
        "factors": centered_f,
        "factor_means": mu_factors,
        "betas": betas,
        "residuals": residuals,
        "asset_r2": asset_r2,
        "mean_excess_returns": mu_excess,
        "excess_returns": y,
        **residual_block,
    }


def build_latent_statistical_equity_block(log_returns, rf_daily, resid_explained_var_target=0.95):
    excess_returns, short_rate_daily = compute_excess_log_returns_daily(log_returns, rf_daily)
    centered = excess_returns - excess_returns.mean()
    pca = PCA(n_components=1)
    factor_scores = pd.DataFrame(
        pca.fit_transform(centered),
        index=centered.index,
        columns=["latent_equity_factor"],
    )
    block = build_factor_driven_equity_block(
        excess_returns=excess_returns,
        factor_df=factor_scores,
        block_name="latent_statistical",
        resid_explained_var_target=resid_explained_var_target,
    )
    block["short_rate_daily"] = short_rate_daily.reindex(block["excess_returns"].index)
    block["latent_explained_variance"] = float(pca.explained_variance_ratio_[0])
    return block


def build_proxy_market_equity_block(log_returns, market_proxy_prices, rf_daily, resid_explained_var_target=0.95):
    excess_returns, short_rate_daily = compute_excess_log_returns_daily(log_returns, rf_daily)
    market_excess = compute_market_excess_log_returns_daily(market_proxy_prices, rf_daily)
    factor_df = market_excess.to_frame(name="market_erp_proxy")
    block = build_factor_driven_equity_block(
        excess_returns=excess_returns,
        factor_df=factor_df,
        block_name="proxy_market",
        resid_explained_var_target=resid_explained_var_target,
    )
    block["short_rate_daily"] = short_rate_daily.reindex(block["excess_returns"].index)
    block["market_excess"] = market_excess.reindex(block["excess_returns"].index)
    return block


def build_apt_equity_block(log_returns, rf_daily, ff_factor_panel, selected_factors, block_name, resid_explained_var_target=0.95):
    excess_returns, short_rate_daily = compute_excess_log_returns_daily(log_returns, rf_daily)
    factor_df = ff_factor_panel[selected_factors].copy()
    block = build_factor_driven_equity_block(
        excess_returns=excess_returns,
        factor_df=factor_df,
        block_name=block_name,
        resid_explained_var_target=resid_explained_var_target,
    )
    block["short_rate_daily"] = short_rate_daily.reindex(block["excess_returns"].index)
    block["selected_factors"] = list(selected_factors)
    return block


def summarize_equity_engine(block):
    return pd.Series({
        "n_equity_factors": block["factors"].shape[1],
        "n_residual_factors": block["resid_scores"].shape[1],
        "mean_asset_R2": block["asset_r2"].mean(),
        "median_asset_R2": block["asset_r2"].median(),
        "max_asset_R2": block["asset_r2"].max(),
        "min_asset_R2": block["asset_r2"].min(),
    }, name=block["engine_name"])


