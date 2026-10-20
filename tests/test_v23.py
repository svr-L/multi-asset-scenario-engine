
import numpy as np
import pandas as pd

from multi_asset_scenario_engine.dynamics import estimate_gjr_garch11_params
from multi_asset_scenario_engine.factors import estimate_credit_bucket_bridge
from multi_asset_scenario_engine.repricing import (
    standard_1y_transition_matrix,
    transition_matrix_for_horizon,
)
from multi_asset_scenario_engine.drift import apply_drift_mode


def test_gjr_targeting_uses_empirical_downside_second_moment_share():
    rng = np.random.default_rng(123)
    # Deliberately asymmetric innovations: negative observations carry more squared mass.
    x = rng.normal(size=1200)
    x[x < 0] *= 1.8
    est = estimate_gjr_garch11_params(
        x, persistence_cap=0.995, min_lr=0.0, min_news=0.0, min_lr_asym=0.0
    )
    assert "neg_share" in est
    assert 0.5 < est["neg_share"] < 1.0
    expected_beta = est["persistence"] - est["alpha"] - est["gamma"] * est["neg_share"]
    assert np.isclose(est["beta"], max(expected_beta, 0.0), atol=1e-12)


def test_horizon_frequency_credit_bridge_reports_fit_frequency_and_residuals():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-01-02", periods=500)
    x = pd.DataFrame(rng.normal(scale=1e-4, size=(500, 2)), index=idx, columns=["Aaa", "Baa"])
    gamma = np.array([[0.8, 0.2], [0.6, 0.5], [0.3, 0.9]])
    eps = rng.normal(scale=4e-5, size=(500, 3))
    dy = x.to_numpy() @ gamma.T + eps
    levels = pd.DataFrame(
        0.01 + np.cumsum(np.vstack([np.zeros((1, 3)), dy[:-1]]), axis=0),
        index=idx, columns=["AAA", "BBB", "BB"],
    )
    bridge = estimate_credit_bucket_bridge(
        levels, x.iloc[1:], anchor_date=idx[-1], min_obs=120, horizon_bdays=21
    )
    assert bridge["frequency_bdays"] == 21
    assert bridge["gamma"].shape == (3, 2)
    assert bridge["residuals"].shape[1] == 3
    assert np.isfinite(bridge["r2"]).all()


def test_transition_matrix_scales_to_subannual_horizon():
    p1 = standard_1y_transition_matrix()
    p3m, info = transition_matrix_for_horizon(p1, 0.25)
    assert p3m.shape == p1.shape
    assert np.allclose(p3m.sum(axis=1).to_numpy(), 1.0, atol=1e-10)
    assert (p3m.to_numpy() >= -1e-12).all()
    assert info["horizon_years"] == 0.25


def test_factor_drift_uses_long_run_factor_premia_and_zeroes_other_blocks():
    dates = pd.bdate_range("2020-01-01", periods=300)
    factors = pd.DataFrame(
        {"MKT_RF": np.linspace(-0.01, 0.02, len(dates)), "SMB": 0.001},
        index=dates,
    )
    eq = {
        "mean_excess_returns": pd.Series([0.001, 0.002], index=["A", "B"]),
        "betas": pd.DataFrame([[1.0, 0.5], [0.5, -0.25]], index=["A", "B"], columns=["MKT_RF", "SMB"]),
        "factors": factors,
    }
    rates = {"mean_dy": pd.Series([1e-5], index=["R"])}
    credit = {"mean_dspread": pd.Series([2e-5], index=["C"])}
    fx = {"mean_fx_logret": pd.Series([3e-5], index=["FX"])}
    ff_long = factors.copy()

    eb, rb, cb, fb, info = apply_drift_mode(
        eq, rates, credit, fx, "factor", ff_long=ff_long, asof=dates[-1], ff_window=factors
    )
    lam = ff_long.mean()
    expected = eq["betas"].to_numpy() @ lam.reindex(eq["betas"].columns).to_numpy()
    assert np.allclose(eb["mean_excess_returns"].to_numpy(), expected)
    assert np.allclose(rb["mean_dy"].to_numpy(), 0.0)
    assert np.allclose(cb["mean_dspread"].to_numpy(), 0.0)
    assert np.allclose(fb["mean_fx_logret"].to_numpy(), 0.0)
    assert info["mode"] == "factor"
