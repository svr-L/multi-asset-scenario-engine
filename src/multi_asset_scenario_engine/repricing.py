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


def make_zero_curve_from_key_rates(key_tenors_years, key_rates):
    x = np.array([0.0] + list(key_tenors_years), dtype=float)
    y = np.array([key_rates[0]] + list(key_rates), dtype=float)
    def zero_rate(t):
        t = float(max(t, 0.0))
        return float(np.interp(t, x, y))
    return zero_rate


def generate_coupon_schedule(issue_date, maturity_date, freq=2):
    months_step = 12 // freq
    dates = []
    current = pd.Timestamp(issue_date) + pd.DateOffset(months=months_step)
    while current < pd.Timestamp(maturity_date):
        dates.append(current)
        current = current + pd.DateOffset(months=months_step)
    dates.append(pd.Timestamp(maturity_date))
    return pd.DatetimeIndex(dates)


def price_coupon_bond_from_curve(eval_date, payment_dates, key_rate_series, tenor_years,
                                 coupon_rate=0.02, face_value=100.0, freq=2, spread=0.0):
    """Dirty present value of a coupon bond off an interpolated zero curve.

    Issue #8: the tenor grid is now passed in (`tenor_years`) instead of hard-coding [2, 5, 10],
    so the pricer works with any set of FRED key rates.

    Credit block: an optional flat `spread` (the bond's option-adjusted credit spread, in decimals)
    is added to every interpolated zero rate, so a corporate bond is discounted at Treasury-zero +
    OAS. With spread=0.0 this is exactly the Treasury pricer (backward compatible).

    Issue #10 (declared approximations): the FRED CMT series (DGS*) are *par* bond-equivalent
    yields, not zero rates. Here we (a) linearly interpolate them across tenor and (b) treat the
    interpolated yields as *continuously-compounded zero rates* (exp(-z t) discounting). This
    conflates par-yield ~= zero-rate and bond-equivalent ~= continuous compounding and introduces a
    small systematic bias. It is acceptable for a prototype; a production version would bootstrap a
    zero curve from the par yields and respect the day-count / compounding convention. The OAS is
    likewise added as a flat continuously-compounded spread, consistent with that approximation.

    Returns the *dirty* (full) present value of all remaining cashflows; see `accrued_interest`
    for the clean/dirty split.
    """
    remaining_dates = payment_dates[payment_dates > pd.Timestamp(eval_date)]
    if len(remaining_dates) == 0:
        return float(face_value)
    times = np.array([(d - pd.Timestamp(eval_date)).days / 365.25 for d in remaining_dates], dtype=float)
    zero_curve = make_zero_curve_from_key_rates(tenor_years, np.asarray(key_rate_series, dtype=float))
    zero_rates = np.array([zero_curve(t) for t in times], dtype=float) + float(spread)
    discount_factors = np.exp(-zero_rates * times)
    cashflows = np.full(len(times), coupon_rate * face_value / freq, dtype=float)
    cashflows[-1] += face_value
    return float(np.sum(cashflows * discount_factors))


def reprice_corporate_bond_scenarios(eval_date, payment_dates, key_rate_scenarios, tenor_years,
                                     spread_scenarios, coupon_rate, face_value=100.0, freq=2):
    """Per-scenario dirty PV of a corporate bond, discounting at the simulated Treasury curve plus
    the simulated OAS for the bond's rating bucket.

    `key_rate_scenarios` is (S, n_key_rates) and `spread_scenarios` is the (S,) OAS path for the
    bond's rating; row s uses curve s and spread s jointly, so a scenario where equities fall and
    spreads widen together (the equity-credit link carried by the joint bootstrap) hits the
    corporate bond through both the rate and the spread channel. Light Python loop over scenarios:
    the pricer is a cheap interpolation, so this is not the simulation bottleneck.
    """
    curves = np.asarray(key_rate_scenarios, dtype=float)
    spreads = np.asarray(spread_scenarios, dtype=float)
    prices = np.empty(curves.shape[0], dtype=float)
    for s in range(curves.shape[0]):
        prices[s] = price_coupon_bond_from_curve(
            eval_date, payment_dates, curves[s], tenor_years,
            coupon_rate=coupon_rate, face_value=face_value, freq=freq, spread=float(spreads[s]),
        )
    return pd.Series(prices)


def standard_1y_transition_matrix(ratings=("AAA", "AA", "A", "BBB", "BB", "B", "CCC")):
    """A representative one-year rating transition matrix (rows = current rating, columns = horizon
    rating including 'D' = default), in the style of published S&P/Moody's global average matrices.

    These are illustrative average through-the-cycle values (e.g. ~0.3% one-year default for BBB,
    ~1.2% for BB, ~4.5% for B, ~26% for CCC); swap in your preferred matrix if you have one. Used at
    the 1Y horizon to draw each corporate's horizon rating; immaterial at 1W (these probabilities are
    one-year, so the one-week migration probability is negligible and the 1W path leaves them unused).
    """
    pct = {
        "AAA": [90.00, 9.00, 0.60, 0.15, 0.15, 0.05, 0.03, 0.02],
        "AA":  [0.60, 90.00, 8.00, 0.80, 0.30, 0.20, 0.05, 0.05],
        "A":   [0.05, 2.00, 91.00, 5.50, 0.90, 0.40, 0.05, 0.10],
        "BBB": [0.02, 0.25, 4.50, 89.00, 4.50, 1.00, 0.40, 0.33],
        "BB":  [0.02, 0.08, 0.40, 6.00, 82.00, 8.00, 2.30, 1.20],
        "B":   [0.00, 0.08, 0.25, 0.40, 6.50, 82.00, 6.27, 4.50],
        "CCC": [0.00, 0.00, 0.20, 0.40, 1.50, 12.00, 59.90, 26.00],
    }
    cols = list(ratings) + ["D"]
    m = pd.DataFrame({r: pct[r] for r in ratings}, index=cols).T
    m = m.div(m.sum(axis=1), axis=0)            # normalize rows to 1 (defensive)
    return m


def build_migration_thresholds(transition_matrix):
    """Latent-variable thresholds for the Gaussian-copula (CreditMetrics) migration model.

    For each current rating, a standard-normal latent Y determines the horizon rating: ordering the
    horizon states from worst (default) to best (AAA), the default band is the lowest Phi^{-1}(p_D) of
    the latent, the next band CCC, and so on. Returns, per current rating, the ascending thresholds
    and the worst->best ordered horizon-rating labels they separate.
    """
    cols = list(transition_matrix.columns)              # [AAA..CCC, D]
    worst_to_best = list(reversed(cols))                # [D, CCC, B, ..., AAA]
    thresholds = {}
    for cur in transition_matrix.index:
        p = transition_matrix.loc[cur, worst_to_best].to_numpy(dtype=float)
        cum = np.cumsum(p)[:-1]                          # exclude the last (=1 -> +inf)
        cum = np.clip(cum, 1e-12, 1 - 1e-12)
        thresholds[cur] = norm.ppf(cum)                 # ascending thresholds (len = n_states - 1)
    return {"thresholds": thresholds, "worst_to_best": worst_to_best}


def simulate_horizon_ratings(current_rating, systematic_z, loading, migration_thresholds, rng):
    """Draw each scenario's horizon rating for a name currently at `current_rating`.

    Latent Y = loading * M + sqrt(1 - loading^2) * eps, where M = `systematic_z` is the shared
    systematic credit factor per scenario (oriented so that a credit blow-out is a LOW M, pushing Y
    down into the downgrade/default bands) and eps is name-specific idiosyncratic noise. Because every
    name shares M, downgrades and defaults cluster in the bad-credit scenarios (wrong-way dependence).
    Returns the horizon rating label per scenario and the boolean default mask.
    """
    M = np.asarray(systematic_z, dtype=float)
    n = M.shape[0]
    eps = rng.standard_normal(n)
    Y = loading * M + np.sqrt(max(1.0 - loading ** 2, 0.0)) * eps
    thr = migration_thresholds["thresholds"][current_rating]
    order = migration_thresholds["worst_to_best"]
    band = np.searchsorted(thr, Y)                       # 0..len(order)-1, low Y -> worst band
    labels = np.array(order, dtype=object)[band]
    return labels, (labels == "D")


def reprice_corporate_bond_with_migration(eval_date, payment_dates, key_rate_scenarios, tenor_years,
                                          all_spread_scenarios, horizon_ratings, coupon_rate,
                                          recovery_rate=0.40, face_value=100.0, freq=2):
    """Per-scenario horizon value of a corporate bond WITH rating migration / default (1Y horizon).

    Each scenario s reprices the bond on the simulated Treasury curve plus the simulated OAS of its
    *horizon* rating (so a downgraded name is discounted at the wider migrated-rating spread, an
    upgraded name at the tighter one); a defaulted name is worth `recovery_rate` * face. The migrated
    spread is read from `all_spread_scenarios` (the simulated OAS path for every rating bucket), so the
    migration P&L and the spread-MtM P&L share the same simulated credit state.
    """
    curves = np.asarray(key_rate_scenarios, dtype=float)
    S = curves.shape[0]
    ratings = np.asarray(horizon_ratings, dtype=object)
    spreads_by_rating = {r: np.asarray(all_spread_scenarios[r], dtype=float) for r in all_spread_scenarios.columns}
    values = np.empty(S, dtype=float)
    for s in range(S):
        r = ratings[s]
        if r == "D":
            values[s] = recovery_rate * face_value
        else:
            sp = float(spreads_by_rating[r][s]) if r in spreads_by_rating else 0.0
            values[s] = price_coupon_bond_from_curve(
                eval_date, payment_dates, curves[s], tenor_years,
                coupon_rate=coupon_rate, face_value=face_value, freq=freq, spread=sp,
            )
    return pd.Series(values)


def coupons_paid_between(issue_date, horizon_date, payment_dates, coupon_rate, face_value=100.0, freq=2):
    """Total coupon cash paid between issue and horizon (no reinvestment), for horizon total return.

    Over a one-week horizon this is ~0 (no coupon usually falls in the window); over a one-year horizon
    it is roughly one full year of coupon, a material part of a bond's total return, so the 1Y horizon
    value should add it to the repriced clean/dirty PV.
    """
    issue_date = pd.Timestamp(issue_date); horizon_date = pd.Timestamp(horizon_date)
    paid = payment_dates[(payment_dates > issue_date) & (payment_dates <= horizon_date)]
    return float(len(paid) * coupon_rate * face_value / freq)


def accrued_interest(eval_date, payment_dates, coupon_rate=0.02, face_value=100.0, freq=2):
    """Accrued interest at eval_date (actual/actual within the current coupon period).

    Issue #10: included so portfolio P&L can be reported on a carry-consistent (clean vs dirty)
    basis, with clean = dirty - accrued. Over a one-week horizon with semiannual coupons this is
    immaterial (order 0.03 on a 2% note) but it is the technically-correct decomposition.
    """
    eval_date = pd.Timestamp(eval_date)
    future = payment_dates[payment_dates > eval_date]
    if len(future) == 0:
        return 0.0
    next_cpn = future[0]
    months_step = 12 // freq
    prev_cpn = next_cpn - pd.DateOffset(months=months_step)
    period_days = (next_cpn - prev_cpn).days
    accrued_days = max((eval_date - prev_cpn).days, 0)
    frac = accrued_days / period_days if period_days > 0 else 0.0
    return float(coupon_rate * face_value / freq * frac)


def price_fx_forward_mtm(eval_date, maturity_date, spot, usd_key_rate_series, usd_tenor_years,
                         strike, foreign_funding_rate, notional_foreign=1.0):
    """USD mark-to-market of a long-foreign FX forward (receive 1 foreign unit, pay `strike` USD per
    unit, at maturity), valued at `eval_date`.

    Decomposed as a long foreign zero-coupon bond minus a short USD zero-coupon bond:

        MTM_USD = notional_foreign * [ spot * exp(-r_foreign * tau) - strike * DF_USD(tau) ]

    where DF_USD(tau) = exp(-z_USD(tau) * tau) comes from the (simulated) USD zero curve and
    r_foreign is the foreign funding rate; tau is the year-fraction to maturity. `spot` is USD per
    one foreign unit (FRED convention, e.g. USD per EUR), so the long-foreign value rises with spot.

    Declared one-week approximation (covered interest parity): r_foreign is held constant over the
    horizon, so the forward-points / carry term is fixed and the P&L is driven by the spot move and
    USD discounting. At a one-week horizon the carry is immaterial (order a few bp) next to weekly
    spot volatility. This reuses the same continuously-compounded zero-curve convention (and its
    declared par-yield ~= zero-rate approximation) as the bond pricer.
    """
    eval_date = pd.Timestamp(eval_date)
    tau = max((pd.Timestamp(maturity_date) - eval_date).days / 365.25, 0.0)
    zero_curve = make_zero_curve_from_key_rates(usd_tenor_years, np.asarray(usd_key_rate_series, dtype=float))
    df_usd = float(np.exp(-zero_curve(tau) * tau))
    df_for = float(np.exp(-float(foreign_funding_rate) * tau))
    return float(notional_foreign) * (float(spot) * df_for - float(strike) * df_usd)


def atm_forward_fx_strike(eval_date, maturity_date, spot, usd_key_rate_series, usd_tenor_years,
                          foreign_funding_rate):
    """At-the-money forward FX rate (USD per foreign unit) by covered interest parity, so a freshly
    struck forward has zero MTM at inception:  K = spot * DF_foreign(tau) / DF_USD(tau)."""
    eval_date = pd.Timestamp(eval_date)
    tau = max((pd.Timestamp(maturity_date) - eval_date).days / 365.25, 0.0)
    zero_curve = make_zero_curve_from_key_rates(usd_tenor_years, np.asarray(usd_key_rate_series, dtype=float))
    df_usd = float(np.exp(-zero_curve(tau) * tau))
    df_for = float(np.exp(-float(foreign_funding_rate) * tau))
    return float(spot) * df_for / df_usd


def reprice_fx_forward_scenarios(eval_date, maturity_date, spot_scenarios, usd_key_rate_scenarios,
                                 usd_tenor_years, strike, foreign_funding_rate, notional_foreign=1.0):
    """Per-scenario USD MTM of the FX forward, using the simulated spot (row s) jointly with the
    simulated USD curve (row s), so an FX move and a Treasury move hit the forward together. Light
    Python loop; each valuation is a couple of interpolations, not the simulation bottleneck.
    """
    spots = np.asarray(spot_scenarios, dtype=float)
    curves = np.asarray(usd_key_rate_scenarios, dtype=float)
    mtm = np.empty(spots.shape[0], dtype=float)
    for s in range(spots.shape[0]):
        mtm[s] = price_fx_forward_mtm(
            eval_date, maturity_date, spots[s], curves[s], usd_tenor_years,
            strike=strike, foreign_funding_rate=foreign_funding_rate, notional_foreign=notional_foreign,
        )
    return pd.Series(mtm)


