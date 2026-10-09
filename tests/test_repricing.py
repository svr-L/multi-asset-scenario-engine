import numpy as np
import pandas as pd
from multi_asset_scenario_engine.repricing import (
    make_zero_curve_from_key_rates,
    generate_coupon_schedule,
    price_coupon_bond_from_curve,
    atm_forward_fx_strike,
    price_fx_forward_mtm,
)


def test_flat_curve_interpolation():
    tenors = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
    rates = np.full_like(tenors, 0.03, dtype=float)
    curve = make_zero_curve_from_key_rates(tenors, rates)
    assert np.isclose(curve(3.0), 0.03, atol=1e-12)


def test_coupon_bond_price_is_finite():
    valuation = pd.Timestamp("2026-01-02")
    maturity = pd.Timestamp("2031-01-02")
    schedule = generate_coupon_schedule(valuation, maturity, freq=2)
    tenors = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
    rates = pd.Series(np.full(len(tenors), 0.03), index=[f"K{i}" for i in range(len(tenors))])
    px = price_coupon_bond_from_curve(
        valuation, schedule, rates, tenors, coupon_rate=0.04, face_value=100.0, freq=2
    )
    assert np.isfinite(px) and px > 0


def test_atm_forward_has_zero_initial_mtm():
    valuation = pd.Timestamp("2026-01-02")
    maturity = pd.Timestamp("2026-04-02")
    tenors = np.array([0.25, 1.0, 2.0, 5.0])
    rates = pd.Series(np.full(len(tenors), 0.03), index=[f"K{i}" for i in range(len(tenors))])
    spot = 1.10
    foreign_rate = 0.02
    strike = atm_forward_fx_strike(valuation, maturity, spot, rates, tenors, foreign_rate)
    mtm = price_fx_forward_mtm(
        valuation, maturity, spot, rates, tenors, strike, foreign_rate, notional_foreign=100.0
    )
    assert abs(float(mtm)) < 1e-10
