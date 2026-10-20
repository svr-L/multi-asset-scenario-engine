from __future__ import annotations

from copy import deepcopy
import pandas as pd

# Universe and market data
TICKERS = ["MSFT", "JPM", "XOM", "JNJ", "PG", "HD", "CAT", "NEM", "PLD", "DUK"]
SECTOR_MAP = {
    "MSFT": "Information Technology", "JPM": "Financials", "XOM": "Energy",
    "JNJ": "Health Care", "PG": "Consumer Staples", "HD": "Consumer Discretionary",
    "CAT": "Industrials", "NEM": "Materials", "PLD": "Real Estate", "DUK": "Utilities",
}
MARKET_PROXY = "SPY"
RATE_TICKERS = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"]
TENOR_YEARS = {
    "DGS1MO": 1/12, "DGS3MO": .25, "DGS6MO": .5, "DGS1": 1., "DGS2": 2.,
    "DGS3": 3., "DGS5": 5., "DGS7": 7., "DGS10": 10., "DGS30": 30.,
}
CREDIT_OAS_SERIES = {
    "BAMLC0A1CAAA": "AAA", "BAMLC0A2CAA": "AA", "BAMLC0A3CA": "A",
    "BAMLC0A4CBBB": "BBB", "BAMLH0A1HYBB": "BB", "BAMLH0A2HYB": "B",
    "BAMLH0A3HYC": "CCC",
}
FX_SERIES = {"DEXUSEU": "EURUSD"}
MOODYS_SERIES = {"DAAA": "Aaa", "DBAA": "Baa"}

# Exact v23 research-notebook defaults. The advanced profile below turns on the
# mature configuration used in the executed validation run.
START_DATE = "2023-05-15"
SAMPLE_MODE = "short"
BLOCK_START_DATES = {
    "equity": "1995-01-01", "rates": "1995-01-01",
    "credit": "1996-12-31", "fx": "1999-01-04",
}
CREDIT_BLOCK_SOURCE = "oas_buckets"
CREDIT_RESIDUAL_MODE = "drop"
CREDIT_BRIDGE_FREQUENCY = "daily"
LONG_GAP_TOLERANCE_DAYS = 21
AS_OF_DATE = pd.Timestamp("2026-05-01")
DOWNLOAD_END = (AS_OF_DATE + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

ACTIVE_EQUITY_ENGINE = "apt_carhart"
DRIFT_MODE = "sample"
FF_LONG_RUN_START = "1926-07-01"

TARGET_CREDIT_EXPLAINED_VAR = .99
CREDIT_MAX_FACTORS = 3
TARGET_RATE_EXPLAINED_VAR = .995
TARGET_RESID_EXPLAINED_VAR = .95
TARGET_FX_EXPLAINED_VAR = .99
FX_MAX_FACTORS = None

HORIZON_BDAYS_BY_MODE = {"1W": 5, "1M": 21, "3M": 63, "1Y": 252}
HORIZON_LABELS = {
    "1W": "one-week", "1M": "one-month",
    "3M": "three-month", "1Y": "one-year",
}
HORIZON_MODE = "1W"
HORIZON_BDAYS = HORIZON_BDAYS_BY_MODE[HORIZON_MODE]
HORIZON_YEARS = HORIZON_BDAYS / 252.0
HORIZON_LABEL = HORIZON_LABELS[HORIZON_MODE]

N_SCENARIOS = 10_000
RNG_SEED = 42
VALIDATION_SEED = 2024
BOOTSTRAP_EXPECTED_BLOCK = 10
BOOTSTRAP_BLOCK_MODE = "fixed"
CONDITIONAL_VOL = "none"
VOL_EWMA_LAMBDA = .94
VOL_REVERSION_HALFLIFE_DAYS = 34.0
VOL_PARAMS_MODE = "fixed"
LEVEL_MEAN_REVERSION = "none"
OU_TSTAT_GATE = -2.86
DRIVER_DYNAMICS = "block_bootstrap"

CVAR_ALPHA = .05
CVAR_MAX_WEIGHT = .50

BACKTEST_N_DATES = 40
BACKTEST_STEP_BDAYS = max(5, HORIZON_BDAYS)
BACKTEST_N_SCENARIOS = 2_000
BACKTEST_MODE = "recent"
BACKTEST_MIN_TRAIN_DAYS = 750
BACKTEST_MAX_DATES = 260
BACKTEST_MIN_REFIT_DAYS = 252
BACKTEST_HORIZONS = None
ES_BACKTEST_N_SIM = 2_000

COUPON_RATE = .02
FACE_VALUE = 100.0
PAY_FREQ = 2
CORP_BONDS = [
    {"name": "CORP_BBB_7Y", "rating": "BBB", "coupon": .052, "maturity_years": 7},
    {"name": "CORP_BB_5Y", "rating": "BB", "coupon": .068, "maturity_years": 5},
]
REPRESENTATIVE_EQUITY = TICKERS[0]
REPRESENTATIVE_RATING = "BBB"
CREDIT_MIGRATION_ENABLED = HORIZON_MODE != "1W"
RECOVERY_RATE = .40
MIGRATION_FACTOR_LOADING = .30
TRANSITION_RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]

FX_FORWARDS = [{
    "name": "FXFWD_EURUSD_3M", "pair": "EURUSD", "maturity_years": .25,
    "foreign_funding_rate": .020, "notional_foreign": 100.0,
}]

MOODYS_START_DATE = "1986-01-01"
CRISIS_SIGMA_THRESHOLD = 4.0
CRISIS_MODE = "none"
CRISIS_PROB = None
CRISIS_SCALE = None

REFERENCE_PROFILE = {k: deepcopy(v) for k, v in globals().copy().items() if k.isupper()}


def horizon_fields(horizon_mode: str) -> dict:
    """Return the v23 horizon-dependent settings for 1W/1M/3M/1Y."""
    if horizon_mode not in HORIZON_BDAYS_BY_MODE:
        raise ValueError(
            f"Unknown HORIZON_MODE {horizon_mode!r}; "
            f"expected one of {list(HORIZON_BDAYS_BY_MODE)}"
        )
    bdays = HORIZON_BDAYS_BY_MODE[horizon_mode]
    return {
        "HORIZON_MODE": horizon_mode,
        "HORIZON_BDAYS": bdays,
        "HORIZON_YEARS": bdays / 252.0,
        "HORIZON_LABEL": HORIZON_LABELS[horizon_mode],
        "CREDIT_MIGRATION_ENABLED": horizon_mode != "1W",
        "BACKTEST_STEP_BDAYS": max(5, bdays),
    }


def profile(name: str = "reference") -> dict:
    """Return a mutable v23 configuration dictionary.

    reference
        Exact v23 research-notebook defaults.
    advanced
        Mature v23 scenario configuration used for the main research run, but
        with a practical recent single-horizon backtest.
    full_validation
        Advanced configuration plus the full 1W/1M/3M/1Y point-in-time
        walk-forward validation used to adjudicate the v23 changes.
    """
    cfg = {k: deepcopy(v) for k, v in REFERENCE_PROFILE.items()}
    if name == "reference":
        return cfg
    if name not in {"advanced", "full_validation"}:
        raise ValueError(f"Unknown profile: {name}")

    cfg.update({
        "SAMPLE_MODE": "long",
        "CREDIT_BLOCK_SOURCE": "moody_2pt",
        "CREDIT_RESIDUAL_MODE": "restore",
        "CREDIT_BRIDGE_FREQUENCY": "horizon",
        "DRIFT_MODE": "factor",
        "CONDITIONAL_VOL": "ewma_fhs",
        "VOL_PARAMS_MODE": "estimated_gjr",
        "LEVEL_MEAN_REVERSION": "ou",
        "BOOTSTRAP_BLOCK_MODE": "fixed",
        "DRIVER_DYNAMICS": "block_bootstrap",
        "CRISIS_MODE": "none",
    })
    if name == "full_validation":
        cfg.update({
            "BACKTEST_MODE": "full",
            "BACKTEST_HORIZONS": ("1W", "1M", "3M", "1Y"),
        })
    return cfg
