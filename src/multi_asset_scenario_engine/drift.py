from __future__ import annotations

import numpy as np
import pandas as pd

def apply_drift_mode(equity_block, rates_block, credit_block, fx_block, mode, ff_long=None,
                     asof=None, ff_window=None, priced_factors=("MKT_RF", "SMB", "HML", "MOM")):
    """Set the engine's EXPECTED daily changes according to DRIFT_MODE (v22).

    The simulator adds a constant daily mean to every block: `mean_excess_returns` (equities),
    `mean_dy` (key rates), `mean_dspread` (credit) and `mean_fx_logret` (FX); the factor drivers are
    centered, so nothing else carries a drift. Up to v21 all four were the SAMPLE means of the
    estimation window. This function swaps them, returning shallow copies (the inputs are untouched):

      "sample" -- the inputs, unchanged (byte-identical);
      "factor" -- mu_i = beta_i' lambda with lambda the long-run Fama-French daily premia from `ff_long`
                  up to `asof` (point-in-time), alpha = 0; a factor outside the Fama-French set is
                  priced through its market beta (OLS slope on MKT_RF over the window x lambda_MKT).
                  Rates, credit and FX drift at zero;
      "zero"   -- every drift zero (equities then earn the risk-free rate only).

    Returns (equity_block, rates_block, credit_block, fx_block, info).
    """
    if mode == "sample":
        return equity_block, rates_block, credit_block, fx_block, {"mode": "sample"}
    if mode not in ("factor", "zero"):
        raise ValueError(f"Unknown DRIFT_MODE {mode!r}; expected 'sample', 'factor' or 'zero'")

    def _zeroed(block, key):
        if block is None:
            return None
        out = dict(block)
        out[key] = block[key] * 0.0
        return out

    rb = _zeroed(rates_block, "mean_dy")
    cb = _zeroed(credit_block, "mean_dspread")
    fb = _zeroed(fx_block, "mean_fx_logret")
    eb = dict(equity_block)
    info = {"mode": mode, "sample_mean_excess": equity_block["mean_excess_returns"].copy(), "lambda": None}
    if mode == "zero":
        eb["mean_excess_returns"] = equity_block["mean_excess_returns"] * 0.0
    else:
        if ff_long is None:
            raise ValueError("DRIFT_MODE='factor' needs the long Fama-French history (ff_long)")
        hist = ff_long.loc[:pd.Timestamp(asof)] if asof is not None else ff_long
        lam_ff = hist[[c for c in priced_factors if c in hist.columns]].mean()
        betas = equity_block["betas"]
        lam, src = {}, {}
        for col in betas.columns:
            if col in lam_ff.index:
                lam[col], src[col] = float(lam_ff[col]), "Fama-French long-run mean"
            else:
                f = equity_block["factors"][col]
                mkt = ff_window["MKT_RF"].reindex(f.index)
                ok = f.notna() & mkt.notna()
                b = float(np.cov(f[ok], mkt[ok])[0, 1] / np.var(mkt[ok], ddof=1))
                lam[col], src[col] = b * float(lam_ff["MKT_RF"]), f"market beta {b:.2f} x MKT_RF premium"
        lam = pd.Series(lam)
        eb["mean_excess_returns"] = pd.Series(betas.to_numpy() @ lam.reindex(betas.columns).to_numpy(),
                                              index=betas.index)
        info.update({"lambda": lam, "lambda_source": pd.Series(src),
                     "history_start": hist.index[0], "history_end": hist.index[-1]})
    info["drift_excess"] = eb["mean_excess_returns"]
    return eb, rb, cb, fb, info
