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

from .dynamics import sample_t_copula_innovations, stationary_bootstrap_index_matrix

def _apply_credit_bridge(result, credit_bridge, current_spreads, credit_floor):
    """Map the simulated credit panel (2-point Moody's curve) into rating-bucket OAS space for
    repricing. No-op when `credit_bridge` is None (default source) -- the byte-identity path. The
    native simulated panel is preserved under "credit_spread_scenarios_raw" so the validation cell
    can compare the engine against its own history in the space it actually simulates."""
    if credit_bridge is None or "credit_spread_scenarios" not in result:
        return result
    scen = result["credit_spread_scenarios"]
    gamma = credit_bridge["gamma"].reindex(columns=list(scen.columns))
    cur2 = current_spreads.reindex(scen.columns).to_numpy(dtype=float)
    delta = scen.to_numpy(dtype=float) - cur2[None, :]
    anchor = credit_bridge["current_buckets"]
    buckets = np.clip(anchor.to_numpy(dtype=float)[None, :] + delta @ gamma.to_numpy(dtype=float).T,
                      credit_floor, None)
    result["credit_spread_scenarios_raw"] = scen
    result["credit_spread_scenarios"] = pd.DataFrame(buckets, columns=list(anchor.index))
    return result


def simulate_joint_horizon(var_res, resid_hist, last_state, equity_block, rates_block,
                           current_prices, current_key_rates, horizon_bdays, n_scenarios,
                           avg_block_length, rng_seed, daily_rf, resid_idio_std=None,
                           rate_floor=-0.02, credit_block=None, current_spreads=None,
                           credit_floor=1e-4, fx_block=None, current_fx=None,
                           dynamics="var1", demeaned_drivers=None, driver_mean=None,
                           crisis_mode="none", crisis_prob=0.0, crisis_scale=1.0,
                           crisis_pool=None, copula=None, asset_copula=None, fhs_filter=None,
                           ou_rates=None, ou_credit=None, credit_bridge=None, _boot_idx=None):
    """Vectorized one-week joint scenario simulation (issue #5).

    Changes vs the original loop version:
      * fully vectorized across scenarios; only `horizon_bdays` steps are iterated, with batch
        matrix operations over the (n_scenarios, state_dim) state.
      * equity total-return reconstruction adds back a *constant* daily risk-free `daily_rf` (the
        FF daily RF at the as-of date) instead of the simulated DGS2/252, which unifies the
        risk-free convention with the factor block (issue #6). The variation of the short rate over
        a 5-business-day horizon is immaterial (order 1e-5), so treating it as constant is cleaner.
      * optional iid idiosyncratic noise `resid_idio_std` re-injects the residual variance that the
        PCA compression discards, so single-name tails are not understated (issue #11).

    Note: vectorization changes the order in which random numbers are consumed, so individual
    scenario paths differ from the old implementation; the simulated *distribution* is unchanged.
    `_boot_idx` is an internal testing hook (pass a precomputed index matrix to make the run
    deterministic).
    """
    rng = np.random.default_rng(rng_seed)
    intercept = np.asarray(var_res.intercept, dtype=float)            # (k,)
    coef = np.asarray(var_res.coefs[0], dtype=float)                  # (k, k)
    resid_hist_arr = np.asarray(resid_hist, dtype=float)
    n_hist = resid_hist_arr.shape[0]

    # Option A: selectable joint-state dynamics. "var1" propagates the fitted VAR(1) and bootstraps
    # its residuals; "block_bootstrap" drops the autoregressive term entirely and resamples the
    # demeaned joint drivers in stationary blocks (state = driver_mean + demeaned_driver[boot]) -- a
    # block-bootstrap of the joint innovations around their mean. When the VAR one-step R^2 is ~0 the
    # two give near-identical distributions, but block_bootstrap removes the k + k^2 VAR parameters.
    # The factor reconstruction below (loadings -> daily returns/changes/spreads/FX, cumulated) is
    # identical in both modes; only the per-day state draw differs.
    if dynamics not in ("var1", "block_bootstrap", "t_copula", "asset_copula"):
        raise ValueError(f"Unknown dynamics: {dynamics!r} "
                         "(expected 'var1', 'block_bootstrap', 't_copula', 'asset_copula')")
    if dynamics == "block_bootstrap":
        if demeaned_drivers is None or driver_mean is None:
            raise ValueError("block_bootstrap dynamics requires demeaned_drivers and driver_mean")
        dd_arr = np.asarray(demeaned_drivers, dtype=float)
        dmean_arr = np.asarray(driver_mean, dtype=float)
        n_innov = dd_arr.shape[0]
    elif dynamics == "t_copula":
        if copula is None or driver_mean is None:
            raise ValueError("t_copula dynamics requires a fitted copula and driver_mean")
        dmean_arr = np.asarray(driver_mean, dtype=float)
    else:
        n_innov = n_hist

    # Crisis-tail overlay (opt-in; the calm-window default is crisis_mode="none", which leaves the
    # draw byte-identical to the no-crisis engine). A crisis is a JOINT event, so on a crisis step the
    # entire joint innovation is amplified, blowing out equity, rates, credit and FX together:
    #   * "regime_jump": parametric -- amplify by a fixed `crisis_scale` (manual or Moody's-calibrated).
    #   * "moody_backbone": empirical -- amplify by a real historical crisis severity drawn from
    #     `crisis_pool` (Moody's Baa-Aaa stress-day move sizes, in multiples of the calm std).
    # `crisis_prob` is the per-step probability a scenario enters stress. The crisis RNG draws are
    # only consumed when crisis_mode != "none", so the off path does not disturb the base sequence.
    if crisis_mode not in ("none", "regime_jump", "moody_backbone"):
        raise ValueError(f"Unknown crisis_mode: {crisis_mode!r} (expected 'none', 'regime_jump', 'moody_backbone')")
    use_crisis = crisis_mode != "none" and crisis_prob > 0.0
    if crisis_mode == "moody_backbone" and use_crisis:
        crisis_pool_arr = np.asarray(crisis_pool, dtype=float) if crisis_pool is not None else np.array([float(crisis_scale)])
        if crisis_pool_arr.size == 0:
            crisis_pool_arr = np.array([float(crisis_scale)])

    # Asset-level copula (v12): a reduced-form generator that bypasses the factor reconstruction. A
    # draw from the observable-level t-copula IS a full daily observable innovation (equity log-returns
    # + key-rate changes + spread changes + FX log-return), assembled directly into scenarios. The
    # marginals are demeaned in the copula, so the stored means are added back per draw; the crisis
    # overlay (if any) scales the demeaned innovation exactly as in the factor path.
    if dynamics == "asset_copula":
        if asset_copula is None:
            raise ValueError("asset_copula dynamics requires a fitted asset_copula")
        groups = asset_copula["asset_groups"]
        mean = np.asarray(asset_copula["mean"], dtype=float)
        S = n_scenarios
        has_cr = "credit" in groups
        has_fx = "fx" in groups
        eq_cols = groups["equity"]["columns"]; eq_sl = groups["equity"]["slice"]
        rt_cols = groups["rates"]["columns"]; rt_sl = groups["rates"]["slice"]
        cum_eq = np.zeros((S, len(eq_cols)))
        cum_dy = np.zeros((S, len(rt_cols)))
        cum_cr = np.zeros((S, len(groups["credit"]["columns"]))) if has_cr else None
        cum_fx = np.zeros((S, len(groups["fx"]["columns"]))) if has_fx else None
        for _h in range(horizon_bdays):
            draw = sample_t_copula_innovations(asset_copula, S, rng)      # (S, P) demeaned
            if use_crisis:
                is_crisis = rng.random(S) < crisis_prob
                if crisis_mode == "regime_jump":
                    cscale = np.where(is_crisis, crisis_scale, 1.0)
                else:
                    cscale = np.ones(S)
                    n_c = int(is_crisis.sum())
                    if n_c > 0:
                        cscale[is_crisis] = crisis_pool_arr[rng.integers(0, crisis_pool_arr.size, n_c)]
                draw = draw * cscale[:, None]
            obs = mean[None, :] + draw
            cum_eq += obs[:, eq_sl]
            cum_dy += obs[:, rt_sl]
            if has_cr:
                cum_cr += obs[:, groups["credit"]["slice"]]
            if has_fx:
                cum_fx += obs[:, groups["fx"]["slice"]]
        cur_p = current_prices.reindex(eq_cols).to_numpy()
        cur_r = current_key_rates.reindex(rt_cols).to_numpy()
        key_rate_scn = np.clip(cur_r[None, :] + cum_dy, rate_floor, None)
        result = {
            "stock_excess_return_scenarios": pd.DataFrame(cum_eq - daily_rf * horizon_bdays, columns=eq_cols),
            "stock_log_return_scenarios": pd.DataFrame(cum_eq, columns=eq_cols),
            "stock_price_scenarios": pd.DataFrame(cur_p[None, :] * np.exp(cum_eq), columns=eq_cols),
            "key_rate_scenarios": pd.DataFrame(key_rate_scn, columns=rt_cols),
        }
        if has_cr:
            cr_cols = groups["credit"]["columns"]
            cur_s = current_spreads.reindex(cr_cols).to_numpy()
            result["credit_spread_scenarios"] = pd.DataFrame(np.clip(cur_s[None, :] + cum_cr, credit_floor, None), columns=cr_cols)
        if has_fx:
            fx_cols = groups["fx"]["columns"]
            cur_fx_v = current_fx.reindex(fx_cols).to_numpy()
            result["fx_log_return_scenarios"] = pd.DataFrame(cum_fx, columns=fx_cols)
            result["fx_spot_scenarios"] = pd.DataFrame(cur_fx_v[None, :] * np.exp(cum_fx), columns=fx_cols)
        return _apply_credit_bridge(result, credit_bridge, current_spreads, credit_floor)

    asset_names = current_prices.index
    rate_names = current_key_rates.index
    n_stocks = len(asset_names)
    n_eq = equity_block["factors"].shape[1]
    n_resid = equity_block["resid_loadings"].shape[1]
    n_rate = rates_block["loadings"].shape[1]

    mu_excess = equity_block["mean_excess_returns"].to_numpy()        # (n_stocks,)
    beta = equity_block["betas"].to_numpy()                           # (n_stocks, n_eq)
    resid_load = equity_block["resid_loadings"].to_numpy()            # (n_stocks, n_resid)
    mu_dy = rates_block["mean_dy"].to_numpy()                         # (n_key_rates,)
    rate_load = rates_block["loadings"].to_numpy()                    # (n_key_rates, n_rate)

    # Credit block (optional). Stacked in the joint state AFTER the rates PCs, so its slice starts
    # at n_eq + n_resid + n_rate. Reconstruction mirrors the rates curve: spread changes are rebuilt
    # from the credit PCs and cumulated onto the current OAS levels. No extra RNG is consumed, so the
    # equity/rates draws are identical whether or not credit is present.
    use_credit = credit_block is not None and current_spreads is not None
    if use_credit:
        n_credit = credit_block["loadings"].shape[1]
        mu_dspread = credit_block["mean_dspread"].to_numpy()          # (n_ratings,)
        credit_load = credit_block["loadings"].to_numpy()            # (n_ratings, n_credit)
        spread_names = current_spreads.index
        spread_path = np.tile(current_spreads.to_numpy().astype(float), (n_scenarios, 1))
    else:
        n_credit = 0

    # FX block (optional). Stacked in the joint state AFTER the credit PCs, so its slice starts at
    # n_eq + n_resid + n_rate + n_credit (n_credit is 0 when no credit block is supplied). Unlike the
    # rates/credit curves, the FX driver is a daily *log-return*: it is cumulated over the horizon and
    # exponentiated onto the current spot level (one column per currency pair). No extra RNG is
    # consumed, so the equity/rates/credit draws are identical whether or not FX is present.
    use_fx = fx_block is not None and current_fx is not None
    if use_fx:
        n_fx = fx_block["loadings"].shape[1]
        mu_fx = fx_block["mean_fx_logret"].to_numpy()                 # (n_pairs,)
        fx_load = fx_block["loadings"].to_numpy()                     # (n_pairs, n_fx)
        fx_names = current_fx.index
        cum_fx = np.zeros((n_scenarios, len(fx_names)))
    else:
        n_fx = 0

    if resid_idio_std is None:
        idio_std = np.zeros(n_stocks)
    else:
        idio_std = np.nan_to_num(np.asarray(resid_idio_std.reindex(asset_names).to_numpy(), dtype=float), nan=0.0)

    if dynamics in ("var1", "block_bootstrap"):
        if _boot_idx is None:
            boot_idx = stationary_bootstrap_index_matrix(n_innov, n_scenarios, horizon_bdays, avg_block_length, rng)
        else:
            boot_idx = np.asarray(_boot_idx, dtype=int)

    # Filtered Historical Simulation (conditional volatility) -- only with the block bootstrap. The
    # resampled draw is a STANDARDIZED residual that is re-scaled by a per-scenario, per-factor
    # conditional volatility starting at today's level and rolled forward by the GARCH recursion, so
    # volatility clusters along each path and mean-reverts to the unconditional level over the horizon.
    use_fhs = (fhs_filter is not None) and (dynamics == "block_bootstrap")
    if use_fhs:
        z_arr = np.asarray(fhs_filter["standardized_residuals"], dtype=float)
        fhs_omega = np.asarray(fhs_filter["omega"], dtype=float)[None, :]
        fhs_alpha = np.asarray(fhs_filter["alpha"], dtype=float)    # scalar (fixed) or (k,) per-factor
        fhs_beta = np.asarray(fhs_filter["beta"], dtype=float)      # broadcasting handles both
        fhs_gamma = np.asarray(fhs_filter.get("gamma", 0.0), dtype=float)
        _fhs_gjr = bool(np.any(fhs_gamma != 0.0))                   # leverage term active (estimated_gjr)
        sigma2 = np.tile(np.asarray(fhs_filter["cond_var_now"], dtype=float), (n_scenarios, 1))   # (S, k)

    state = np.tile(np.asarray(last_state, dtype=float), (n_scenarios, 1))                 # (S, k)
    key_rate_path = np.tile(current_key_rates.to_numpy().astype(float), (n_scenarios, 1))  # (S, n_key_rates)
    cum_excess = np.zeros((n_scenarios, n_stocks))
    cum_log = np.zeros((n_scenarios, n_stocks))

    for h in range(horizon_bdays):
        if dynamics == "var1":
            innov = resid_hist_arr[boot_idx[:, h]]                    # (S, k)
        elif dynamics == "block_bootstrap":
            if use_fhs:
                innov = z_arr[boot_idx[:, h]] * np.sqrt(sigma2)       # revol standardized residual by today's vol
            else:
                innov = dd_arr[boot_idx[:, h]]                        # (S, k)
        else:  # t_copula: draw a fresh tail-dependent joint innovation (empirical margins + t copula)
            innov = sample_t_copula_innovations(copula, n_scenarios, rng)   # (S, k), demeaned

        if use_fhs:
            _fhs_innov_sq = innov ** 2                                # pre-crisis innovation drives the vol recursion
            if _fhs_gjr:
                _fhs_innov_neg = (innov < 0.0)                        # sign unchanged by the (positive) crisis scale

        if use_crisis:
            is_crisis = rng.random(n_scenarios) < crisis_prob         # (S,) per-scenario stress flag
            if crisis_mode == "regime_jump":
                cscale = np.where(is_crisis, crisis_scale, 1.0)
            else:  # moody_backbone: draw a real historical crisis severity for stressed scenarios
                cscale = np.ones(n_scenarios)
                n_c = int(is_crisis.sum())
                if n_c > 0:
                    cscale[is_crisis] = crisis_pool_arr[rng.integers(0, crisis_pool_arr.size, n_c)]
            innov = innov * cscale[:, None]                           # amplify the WHOLE joint innovation

        if use_fhs:
            # roll conditional variance forward (GARCH(1,1), mean-reverting); the crisis overlay is an
            # exogenous jump kept orthogonal to the endogenous vol clustering, so it does not feed back.
            if _fhs_gjr:
                sigma2 = fhs_omega + (fhs_alpha + fhs_gamma * _fhs_innov_neg) * _fhs_innov_sq + fhs_beta * sigma2
            else:
                sigma2 = fhs_omega + fhs_alpha * _fhs_innov_sq + fhs_beta * sigma2


        if dynamics == "var1":
            state = intercept[None, :] + state @ coef.T + innov      # (S, k)
        else:  # block_bootstrap / t_copula: demeaned innovation around the driver mean, no AR term
            state = dmean_arr[None, :] + innov                       # (S, k)

        eq_state = state[:, :n_eq]
        resid_state = state[:, n_eq:n_eq + n_resid]
        rate_state = state[:, n_eq + n_resid:n_eq + n_resid + n_rate]

        dy = mu_dy[None, :] + rate_state @ rate_load.T               # (S, n_key_rates)
        if ou_rates is not None:
            # Vasicek/OU pull toward the estimated long-run level, per tenor and per scenario. Gated
            # tenors replace the RW drift with the OU deterministic part (no double counting);
            # non-gated tenors have kappa = 0 and keep the random walk exactly.
            dy = (dy + ou_rates["kappa"][None, :] * (ou_rates["theta"][None, :] - key_rate_path)
                  - ou_rates["gated"][None, :] * mu_dy[None, :])
        key_rate_path = np.clip(key_rate_path + dy, rate_floor, None)

        if use_credit:
            credit_state = state[:, n_eq + n_resid + n_rate:n_eq + n_resid + n_rate + n_credit]
            dspread = mu_dspread[None, :] + credit_state @ credit_load.T   # (S, n_ratings)
            if ou_credit is not None:
                dspread = (dspread + ou_credit["kappa"][None, :] * (ou_credit["theta"][None, :] - spread_path)
                           - ou_credit["gated"][None, :] * mu_dspread[None, :])
            spread_path = np.clip(spread_path + dspread, credit_floor, None)

        if use_fx:
            base = n_eq + n_resid + n_rate + n_credit
            fx_state = state[:, base:base + n_fx]
            daily_fx = mu_fx[None, :] + fx_state @ fx_load.T              # (S, n_pairs) daily FX log-returns
            cum_fx += daily_fx

        idio = resid_state @ resid_load.T                            # (S, n_stocks)
        if idio_std.any():
            idio = idio + rng.normal(0.0, 1.0, size=(n_scenarios, n_stocks)) * idio_std[None, :]

        daily_excess = mu_excess[None, :] + eq_state @ beta.T + idio  # (S, n_stocks)
        daily_log = daily_excess + daily_rf                          # constant rf add-back (issue #6)
        cum_excess += daily_excess
        cum_log += daily_log

    price_scenarios = current_prices.to_numpy()[None, :] * np.exp(cum_log)

    result = {
        "stock_excess_return_scenarios": pd.DataFrame(cum_excess, columns=asset_names),
        "stock_log_return_scenarios": pd.DataFrame(cum_log, columns=asset_names),
        "stock_price_scenarios": pd.DataFrame(price_scenarios, columns=asset_names),
        "key_rate_scenarios": pd.DataFrame(key_rate_path, columns=rate_names),
    }
    if use_credit:
        result["credit_spread_scenarios"] = pd.DataFrame(spread_path, columns=spread_names)
    if use_fx:
        fx_spot_scenarios = current_fx.to_numpy()[None, :] * np.exp(cum_fx)
        result["fx_log_return_scenarios"] = pd.DataFrame(cum_fx, columns=fx_names)
        result["fx_spot_scenarios"] = pd.DataFrame(fx_spot_scenarios, columns=fx_names)
    return _apply_credit_bridge(result, credit_bridge, current_spreads, credit_floor)


