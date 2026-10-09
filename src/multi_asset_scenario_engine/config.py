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
TENOR_YEARS = {"DGS1MO": 1/12, "DGS3MO": .25, "DGS6MO": .5, "DGS1": 1., "DGS2": 2., "DGS3": 3., "DGS5": 5., "DGS7": 7., "DGS10": 10., "DGS30": 30.}
CREDIT_OAS_SERIES = {"BAMLC0A1CAAA": "AAA", "BAMLC0A2CAA": "AA", "BAMLC0A3CA": "A", "BAMLC0A4CBBB": "BBB", "BAMLH0A1HYBB": "BB", "BAMLH0A2HYB": "B", "BAMLH0A3HYC": "CCC"}
FX_SERIES = {"DEXUSEU": "EURUSD"}
MOODYS_SERIES = {"DAAA": "Aaa", "DBAA": "Baa"}

# Reference v20 configuration (matches the research notebook defaults)
START_DATE = "2023-05-15"
SAMPLE_MODE = "short"
BLOCK_START_DATES = {"equity": "1995-01-01", "rates": "1995-01-01", "credit": "1996-12-31", "fx": "1999-01-04"}
CREDIT_BLOCK_SOURCE = "oas_buckets"
LONG_GAP_TOLERANCE_DAYS = 21
AS_OF_DATE = pd.Timestamp("2026-05-01")
DOWNLOAD_END = (AS_OF_DATE + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
ACTIVE_EQUITY_ENGINE = "apt_carhart"
TARGET_CREDIT_EXPLAINED_VAR = .99
CREDIT_MAX_FACTORS = 3
TARGET_RATE_EXPLAINED_VAR = .995
TARGET_RESID_EXPLAINED_VAR = .95
TARGET_FX_EXPLAINED_VAR = .99
FX_MAX_FACTORS = None

HORIZON_MODE = "1W"
HORIZON_BDAYS = 5
HORIZON_YEARS = HORIZON_BDAYS / 252.0
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
BACKTEST_STEP_BDAYS = 5
BACKTEST_N_SCENARIOS = 2_000
BACKTEST_MODE = "recent"
BACKTEST_MIN_TRAIN_DAYS = 750
BACKTEST_MAX_DATES = 260
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
CREDIT_MIGRATION_ENABLED = False
RECOVERY_RATE = .40
MIGRATION_FACTOR_LOADING = .30
TRANSITION_RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
FX_FORWARDS = [{"name": "FXFWD_EURUSD_3M", "pair": "EURUSD", "maturity_years": .25, "foreign_funding_rate": .020, "notional_foreign": 100.0}]
MOODYS_START_DATE = "1986-01-01"
CRISIS_SIGMA_THRESHOLD = 4.0
CRISIS_MODE = "none"
CRISIS_PROB = None
CRISIS_SCALE = None

REFERENCE_PROFILE = {k: deepcopy(v) for k, v in globals().copy().items() if k.isupper()}

def profile(name: str = "reference") -> dict:
    """Return a mutable configuration dictionary.

    reference: exact v20 notebook defaults.
    advanced: exercises the mature v19/v20 architecture while keeping the recent backtest for runtime.
    full_validation: advanced profile plus full-window walk-forward VaR/ES validation.
    """
    cfg = {k: deepcopy(v) for k, v in REFERENCE_PROFILE.items()}
    if name == "reference":
        return cfg
    if name in {"advanced", "full_validation"}:
        cfg.update({
            "SAMPLE_MODE": "long",
            "CREDIT_BLOCK_SOURCE": "moody_2pt",
            "CONDITIONAL_VOL": "ewma_fhs",
            "VOL_PARAMS_MODE": "estimated_gjr",
            "LEVEL_MEAN_REVERSION": "ou",
            "BOOTSTRAP_BLOCK_MODE": "fixed",
            "DRIVER_DYNAMICS": "block_bootstrap",
        })
        if name == "full_validation":
            cfg["BACKTEST_MODE"] = "full"
        return cfg
    raise ValueError(f"Unknown profile: {name}")

def horizon_fields(horizon_mode: str) -> dict:
    bdays = 5 if horizon_mode == "1W" else 252
    return {
        "HORIZON_MODE": horizon_mode,
        "HORIZON_BDAYS": bdays,
        "HORIZON_YEARS": bdays / 252.0,
        "CREDIT_MIGRATION_ENABLED": horizon_mode == "1Y",
    }
