import numpy as np
import pandas as pd
from multi_asset_scenario_engine.backtesting import kupiec_pof_test, christoffersen_independence_test
from multi_asset_scenario_engine.portfolio import search_long_only_min_cvar_portfolio
from multi_asset_scenario_engine.dynamics import stationary_bootstrap_index_matrix


def test_stationary_bootstrap_indices_in_range():
    idx = stationary_bootstrap_index_matrix(
        n_obs=50, n_paths=100, horizon=5, avg_block_length=10, rng=np.random.default_rng(1)
    )
    assert idx.shape == (100, 5)
    assert idx.min() >= 0 and idx.max() < 50


def test_coverage_functions_return_finite_results():
    hits = np.array([0] * 95 + [1] * 5)
    lr, p = kupiec_pof_test(int(hits.sum()), len(hits), alpha=0.05)
    assert np.isfinite(lr) and 0 <= p <= 1
    lr2, p2 = christoffersen_independence_test(hits)
    assert np.isfinite(lr2) and 0 <= p2 <= 1


def test_min_cvar_weights_sum_to_one():
    rng = np.random.default_rng(7)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(300, 4)), columns=list("ABCD"))
    w, diag = search_long_only_min_cvar_portfolio(rets, alpha=0.05, max_weight=0.5)
    assert np.isclose(float(w.sum()), 1.0, atol=1e-8)
    assert (w >= -1e-10).all()
    assert (w <= 0.5 + 1e-10).all()
    assert np.isfinite(diag["ES_95_loss"])
