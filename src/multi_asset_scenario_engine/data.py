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


def download_market_data(tickers, market_proxy, rate_tickers, start_date, download_end):
    """Download adjusted-close equity prices, a market proxy, and FRED Treasury key rates."""
    import yfinance as yf
    from pandas_datareader import data as web
    all_equity_tickers = list(dict.fromkeys(list(tickers) + [market_proxy]))
    stock_raw = yf.download(
        all_equity_tickers,
        start=start_date,
        end=download_end,
        auto_adjust=False,
        progress=False,
        actions=False,
    )

    if isinstance(stock_raw.columns, pd.MultiIndex):
        if "Adj Close" not in stock_raw.columns.get_level_values(0):
            raise ValueError("Adjusted close prices not found in yfinance download output.")
        prices_all = stock_raw["Adj Close"].copy()
    else:
        prices_all = stock_raw.copy()
        if isinstance(prices_all, pd.Series):
            prices_all = prices_all.to_frame(name=all_equity_tickers[0])
        prices_all.columns = all_equity_tickers

    stock_prices = prices_all[tickers].copy()
    market_proxy_prices = prices_all[[market_proxy]].copy()
    rates_raw = web.DataReader(rate_tickers, "fred", start_date, download_end).astype(float) / 100.0
    return stock_prices, market_proxy_prices, rates_raw


def download_credit_oas(credit_oas_series, start_date, download_end):
    """Download ICE BofA option-adjusted credit spreads (OAS) by rating bucket from FRED.

    `credit_oas_series` is an ordered mapping {FRED_code: rating_label} sorted by ascending credit
    risk (AAA -> CCC), so the resulting columns form a credit-quality curve. FRED reports OAS in
    *percent*, so we divide by 100 to express spreads in decimals, consistent with the Treasury
    key-rate convention (a 1.50% OAS becomes 0.0150).

    NOTE ON DATA AVAILABILITY: as of April 2026 the ICE BofA OAS series on FRED are licensed to
    expose only a rolling ~3 years of history. The credit block therefore lives on a recent window
    (see START_DATE / AS_OF_DATE in the config cell). For a longer sample one would substitute a
    non-truncated source (e.g. Moody's seasoned Aaa/Baa yields DAAA/DBAA differenced against a
    Treasury tenor), at the cost of a coarser two-point credit curve.
    """
    from pandas_datareader import data as web
    codes = list(credit_oas_series.keys())
    oas = web.DataReader(codes, "fred", start_date, download_end).astype(float) / 100.0
    oas = oas.rename(columns=dict(credit_oas_series))
    # preserve the ascending-risk column order
    oas = oas[[credit_oas_series[c] for c in codes]]
    return oas


def download_fx_rates(fx_series, start_date, download_end):
    """Download FX spot rates from FRED (one column per currency pair).

    `fx_series` is an ordered mapping {FRED_code: pair_label}. FRED quotes these pairs as
    'US dollars per 1 unit of foreign currency' (e.g. DEXUSEU = USD per EUR), so each column is the
    price in USD of one unit of the foreign currency and a long-foreign position gains when the
    column rises. Unlike the ICE BofA credit series these spot series carry long history with no
    3-year truncation.
    """
    from pandas_datareader import data as web
    codes = list(fx_series.keys())
    fx = web.DataReader(codes, "fred", start_date, download_end).astype(float)
    fx = fx.rename(columns=dict(fx_series))
    fx = fx[[fx_series[c] for c in codes]]
    return fx


def compute_fx_log_returns(fx_rates):
    return np.log(fx_rates / fx_rates.shift(1)).dropna()


def download_moodys_credit(moodys_series, start_date, download_end):
    """Download Moody's seasoned corporate bond yields from FRED (long history, no 3-year truncation).

    `moodys_series` maps FRED codes to labels, e.g. {"DAAA": "Aaa", "DBAA": "Baa"}. These daily
    seasoned-yield series run back to the 1980s and carry every postwar credit crisis, so the
    Baa-Aaa spread is used as a long-history credit-stress backbone to calibrate how often and how
    severely credit blows out relative to the short, calm OAS window the main engine is fit on.
    Yields are in percent on FRED; divided by 100 to decimals here.
    """
    from pandas_datareader import data as web
    codes = list(moodys_series.keys())
    mood = web.DataReader(codes, "fred", start_date, download_end).astype(float) / 100.0
    mood = mood.rename(columns=dict(moodys_series))
    return mood


def compute_credit_crisis_calibration(moodys, horizon_bdays, aaa_label="Aaa", baa_label="Baa",
                                      calm_lookback_years=3, calm_sigma_threshold=4.0):
    """Calibrate empirical credit-crisis statistics from the long Moody's Baa-Aaa history.

    The Baa-Aaa seasoned-yield spread is a coarse (two-point, investment-grade) but very long credit
    risk proxy. We mark a day as a credit-stress day when its daily Baa-Aaa change exceeds
    `calm_sigma_threshold` times the volatility of a recent calm window, then summarize crises by:
      * crisis_prob  -- empirical fraction of days that are credit-stress days (a per-step crisis
        probability for the regime overlay);
      * crisis_scale -- ratio of stress-day to calm-day volatility (how much bigger crisis moves are);
      * dof          -- a Student-t tail index from the full-history excess kurtosis (6/(dof-4));
      * stress_magnitude_pool -- the realized stress-day move sizes (in multiples of the calm std),
        i.e. *real* historical crisis severities, used by the "moody_backbone" simulator mode.

    This is a credit-derived, joint stress calibration: real crises move all asset classes together,
    so the regime overlay applies the scale to the whole joint innovation, not credit alone. The
    coarse IG proxy understates how much high-yield blows out in a crisis (a documented limitation);
    the rating-bucket loadings still apply more of the systematic move to HY, but a true crisis would
    widen HY convexly beyond what the calm-window loadings imply.
    """
    baa_aaa = (moodys[baa_label] - moodys[aaa_label]).dropna()
    dchg = baa_aaa.diff().dropna()
    calm_start = dchg.index.max() - pd.DateOffset(years=calm_lookback_years)
    calm = dchg[dchg.index >= calm_start]
    calm_std = float(calm.std()) if len(calm) > 5 else float(dchg.std())
    if not np.isfinite(calm_std) or calm_std <= 0:
        calm_std = float(dchg.std())

    thresh = calm_sigma_threshold * calm_std
    stress_mask = dchg.abs() > thresh
    n_stress = int(stress_mask.sum())
    crisis_prob = float(stress_mask.mean())
    if n_stress > 5:
        stress_std = float(dchg[stress_mask].std())
        stress_pool = (dchg[stress_mask].abs() / calm_std).to_numpy()
    else:
        stress_std = calm_std * 5.0
        stress_pool = np.array([5.0])
    crisis_scale = stress_std / calm_std if calm_std > 0 else 5.0
    ek = float(dchg.kurtosis())
    dof = float(4.0 + 6.0 / ek) if ek > 0.1 else 8.0
    return {
        "baa_aaa_spread": baa_aaa,
        "daily_changes": dchg,
        "calm_std": calm_std,
        "stress_std": stress_std,
        "crisis_prob": crisis_prob,
        "crisis_scale": crisis_scale,
        "dof": dof,
        "stress_magnitude_pool": stress_pool,
        "n_stress_days": n_stress,
        "n_total_days": int(len(dchg)),
        "history_start": dchg.index.min(),
        "history_end": dchg.index.max(),
    }


def _normalize_ff_columns(columns):
    mapping = {}
    for col in columns:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(col).strip()).upper().strip("_")
        mapping[col] = cleaned
    return mapping


def _ff_table_to_returns(df):
    """Convert raw Fama-French / momentum tables to *simple daily returns* (decimals).

    Issue #7 fix: the FF series are excess / long-short spread returns (MKT-RF, SMB, HML, MOM).
    Applying log1p to a zero-cost long-short spread has no clean interpretation and is
    inconsistent with combining the factor contributions *additively* in the excess-return
    reconstruction. We therefore keep them as simple, additive daily returns. At the daily scale
    log1p(x) ~= x, so this does not move the numbers materially; it only removes the conceptual
    mismatch. (RF is also a level return and is left in the same simple-return convention so that
    'equity excess = equity log-return - RF' uses exactly the same risk-free that MKT-RF embeds.)
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index.astype(str))
    out = out.apply(pd.to_numeric, errors="coerce")
    # Ken French's daily files are ALWAYS in percent -- both through the pandas_datareader
    # famafrench reader and through the raw CSV zips -- so the conversion is deterministic.
    # (The previous per-frame median heuristic sat almost exactly on the momentum factor's
    # median |daily return| of ~0.5%: the momentum table is converted ALONE, so on long windows
    # it could silently stay in percent, entering the joint state at 100x scale. The OLS betas
    # absorb a pure column rescaling, which is why results stayed sane, but the landmine is
    # removed rather than relied upon.)
    out = out / 100.0
    _med = float(out.abs().median().max())
    if _med > 0.05:
        raise ValueError(
            f"Fama-French factors look mis-scaled after /100 (max median |daily| = {_med:.4f}); "
            "the source no longer appears to be in percent -- inspect it before proceeding."
        )
    return out


def _ff_zip_to_frame(url):
    """Parse a Ken French daily CSV zip directly (fallback when pandas_datareader's famafrench reader
    fails, e.g. on parser/dtype incompatibilities with newer pandas). Keeps only the daily table
    (rows starting with an 8-digit date), robust to preamble, annual tables, and copyright footer."""
    import io as _io, zipfile as _zipfile, urllib.request as _urlreq
    raw = _urlreq.urlopen(url, timeout=60).read()
    txt = _zipfile.ZipFile(_io.BytesIO(raw)).read(
        _zipfile.ZipFile(_io.BytesIO(raw)).namelist()[0]).decode("latin-1").splitlines()
    first = next(i for i, ln in enumerate(txt) if re.match(r"^\s*\d{8}\s*,", ln))
    header = ["DATE"] + [h.strip() for h in txt[first - 1].split(",")[1:]]
    data = []
    for ln in txt[first:]:
        if not re.match(r"^\s*\d{8}\s*,", ln):
            break
        data.append([x.strip() for x in ln.split(",")])
    df = pd.DataFrame(data, columns=header[:len(data[0])]).set_index("DATE")
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df.apply(pd.to_numeric, errors="coerce")


def download_fama_french_daily_factors(start_date, end_date):
    """Download daily Fama-French market/style factors plus the daily momentum factor.

    The series are converted to simple daily returns (see `_ff_table_to_returns`) so they can be
    combined additively with the log-return machinery used elsewhere in the notebook. If the
    pandas_datareader famafrench reader fails (parser/dtype issues appear with some pandas versions),
    the function falls back to fetching and parsing Ken French's CSV zips directly.
    """
    try:
        ff3 = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=start_date)[0]
        mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start=start_date)[0]
    except Exception:
        _base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        ff3 = _ff_zip_to_frame(_base + "F-F_Research_Data_Factors_daily_CSV.zip").loc[pd.Timestamp(start_date):]
        mom = _ff_zip_to_frame(_base + "F-F_Momentum_Factor_daily_CSV.zip").loc[pd.Timestamp(start_date):]

    ff3 = ff3.rename(columns=_normalize_ff_columns(ff3.columns))
    mom = mom.rename(columns=_normalize_ff_columns(mom.columns))

    ff3 = _ff_table_to_returns(ff3)
    mom = _ff_table_to_returns(mom)

    mom_col = None
    for candidate in ["MOM", "MOMENTUM", "MOM_FACTOR"]:
        if candidate in mom.columns:
            mom_col = candidate
            break
    if mom_col is None:
        mom_col = mom.columns[0]

    ff = ff3.join(mom[[mom_col]].rename(columns={mom_col: "MOM"}), how="left")

    keep = [c for c in ["MKT_RF", "SMB", "HML", "RF", "MOM"] if c in ff.columns]
    ff = ff[keep].sort_index().loc[:pd.Timestamp(end_date)]
    return ff.dropna(how="any")


def choose_effective_asof(stock_prices, market_proxy_prices, rates, ff_factors, as_of_date, credit=None, fx=None):
    common_dates = stock_prices.index.intersection(market_proxy_prices.index).intersection(rates.index).intersection(ff_factors.index)
    if credit is not None:
        common_dates = common_dates.intersection(credit.index)
    if fx is not None:
        common_dates = common_dates.intersection(fx.index)
    eligible = common_dates[common_dates <= as_of_date]
    if len(eligible) == 0:
        raise ValueError("No common market date available at or before the chosen AS_OF_DATE.")
    return eligible.max()


def align_market_data(stock_prices, market_proxy_prices, rates, ff_factors, effective_date, credit=None, fx=None):
    """Align all blocks on their common dates up to the effective date.

    Returns a 6-tuple (prices, market, yields, ff, credit, fx); `credit` and `fx` are None when the
    corresponding frame is not supplied, so the narrower (equity-rates, or equity-rates-credit)
    paths are unchanged.
    """
    prices = stock_prices.loc[:effective_date].dropna(how="any")
    market = market_proxy_prices.loc[:effective_date].dropna(how="any")
    yields = rates.loc[:effective_date].dropna(how="any")
    ff = ff_factors.loc[:effective_date].dropna(how="any")
    common_dates = prices.index.intersection(market.index).intersection(yields.index).intersection(ff.index)
    cr = None
    if credit is not None:
        cr = credit.loc[:effective_date].dropna(how="any")
        common_dates = common_dates.intersection(cr.index)
    fxr = None
    if fx is not None:
        fxr = fx.loc[:effective_date].dropna(how="any")
        common_dates = common_dates.intersection(fxr.index)
    if cr is not None:
        cr = cr.loc[common_dates]
    if fxr is not None:
        fxr = fxr.loc[common_dates]
    return (prices.loc[common_dates], market.loc[common_dates], yields.loc[common_dates],
            ff.loc[common_dates], cr, fxr)


def compute_equity_log_returns(prices):
    return np.log(prices).diff().dropna()


def compute_yield_changes(yields):
    return yields.diff().dropna()


def compute_spread_changes(spreads):
    """Daily changes in credit OAS levels (decimals), analogous to yield changes."""
    return spreads.diff().dropna()


def compute_excess_log_returns_daily(log_returns, rf_daily):
    """Excess log returns using a *daily* risk-free series (already per-day, in decimals).

    Issue #6 fix: equity excess returns now use the SAME daily risk-free that defines the
    Fama-French factors (the FF 'RF' column), instead of DGS2/252. This removes the double
    risk-free convention (DGS2 for the equity excess vs the FF RF embedded inside MKT-RF) and
    stops using a 2Y point as an overnight-rate proxy.
    """
    rf = rf_daily.reindex(log_returns.index).ffill()
    excess = log_returns.sub(rf, axis=0)
    return excess, rf


def compute_market_excess_log_returns_daily(market_prices, rf_daily):
    market_log_returns = np.log(market_prices.squeeze()).diff().dropna()
    rf = rf_daily.reindex(market_log_returns.index).ffill()
    market_excess = market_log_returns.sub(rf, axis=0)
    market_excess.name = f"{market_prices.columns[0]}_excess_log_return"
    return market_excess


