# Multi-Asset Scenario Engine

A modular Python research prototype for joint US equity-rates scenario generation under the physical measure, instrument-level horizon repricing, and portfolio tail-risk analysis.

The project connects market risk modelling, cross-asset scenario generation, horizon repricing, and portfolio-level risk diagnostics in a single workflow. It is designed as a reusable scenario engine rather than a point-forecasting notebook.

## Positioning

This project is best understood as a prototype for:

- joint scenario generation under the physical measure;
- horizon repricing of equity and rates instruments;
- portfolio tail-risk analysis and robustness diagnostics.

It is not intended as:

- a pure risk-neutral pricing library;
- a production-grade market risk system;
- a pure alpha-signal forecasting project.

## Core idea

The framework follows a risk-engineering pipeline:

1. transform raw market data into economically meaningful risk drivers;
2. model the joint dynamics of equity and rates drivers;
3. simulate future scenarios under the physical measure;
4. map simulated drivers back into instrument values through horizon repricing;
5. evaluate portfolio-level losses, VaR, Expected Shortfall, and tail contributions.

The focus is on the full distribution of future market states and P&L, not on point forecasts.

## Current implementation

### 1. Data layer

The notebook downloads and aligns:

- a diversified US equity universe;
- a market proxy for the observable equity-premium block;
- daily Fama-French factors and momentum;
- a 10-tenor US Treasury CMT panel from FRED.

The current rate tenors are used to build a richer yield-curve factor model. DGS20 is intentionally excluded because its post-2020 availability would introduce a large missing-data gap.

### 2. Equity-premium engines

The equity block is modular. The user can choose among alternative equity-premium engines:

- `latent_statistical`: latent common equity factor plus residual factor compression;
- `proxy_market`: market-proxy excess-return engine;
- `apt_ff3`: APT-style Fama-French 3-factor engine;
- `apt_carhart`: APT-style Carhart engine including momentum.

Equity excess returns use the daily Fama-French risk-free rate for consistency with `MKT_RF`. Long-short factors such as SMB, HML, and MOM are treated as additive daily factor returns.

Residual equity risk is represented through residual PCA. Any residual variance discarded by the PCA compression is reintroduced as idiosyncratic noise, so single-name risk is not mechanically understated.

### 3. Rates block

The rates block uses yield changes across a 10-tenor Treasury panel and compresses the curve through PCA. The first components are interpretable as level-, slope-, and curvature-style movements.

For bond repricing, the notebook documents a pragmatic approximation: CMT par yields are treated as continuously compounded zero rates for the synthetic repricing exercise. This is a research-prototype assumption, not a production curve-construction framework.

### 4. Joint scenario engine

The joint state vector combines:

- equity-premium factors;
- residual equity factors;
- yield-curve factors.

A VAR(1) is fitted to the transformed joint state vector. Scenario generation then uses stationary-bootstrap innovations on the VAR residuals, preserving empirical cross-driver shock structure more flexibly than a purely Gaussian simulation.

The horizon simulation is vectorized across scenarios for efficiency and reproduces the loop-based implementation up to numerical precision.

### 5. Horizon repricing

For each simulated one-week scenario:

- equity prices are repriced from reconstructed cumulative log returns;
- the synthetic Treasury note is repriced by discounting remaining cash flows using the simulated yield curve;
- clean/dirty bond-value handling and accrued-interest assumptions are explicitly documented.

This is horizon repricing under physical scenarios, not no-arbitrage risk-neutral derivatives pricing.

### 6. Portfolio tail-risk layer

Instrument-level outputs are converted into capital-normalized one-week returns. The notebook includes:

- equal-weight portfolio analytics;
- inverse-volatility portfolio analytics;
- long-only minimum-CVaR optimisation;
- VaR and Expected Shortfall;
- weighted tail contributions.

The minimum-CVaR portfolio is solved with the Rockafellar-Uryasev linear-programming formulation via `scipy.optimize.linprog`, replacing random search over the simplex. The selected portfolio is also re-evaluated on an independent scenario draw to expose in-sample tail-risk optimism.

## Validation diagnostics

The notebook includes several diagnostic checks:

- simulated vs historical one-week volatility scale;
- simulated vs historical cross-equity correlations;
- simulated vs historical equity-rates correlations;
- excess-kurtosis comparison;
- VAR(1) stability diagnostics and one-step equation-level R²;
- walk-forward VaR backtest on expanding windows;
- Kupiec proportion-of-failures test;
- Christoffersen independence test;
- PIT / rank histogram for calibration diagnostics.

The walk-forward test is intentionally configurable. Default values are kept modest so the notebook remains runnable, but increasing the number of dates and scenarios gives more statistical power.

## Why this is different from a standard forecasting notebook

A standard forecasting notebook usually stops at predicted returns or predicted risk factors. This project continues the pipeline through scenario generation, repricing, and portfolio tail-risk measurement.

The key object is not a point forecast but a simulated distribution of future equity-rates market states and the resulting distribution of instrument and portfolio P&L.

## Current limitations

This is a research prototype. Important limitations remain:

- the equity universe is still illustrative rather than exhaustive;
- the rates curve uses a simplified CMT-as-zero-rate approximation;
- Fama-French factors are daily simple returns used additively in the equity block;
- residual idiosyncratic risk is currently reintroduced as iid noise;
- the VAR(1) is intentionally simple and should be monitored through stability and R² diagnostics;
- walk-forward validation defaults are computationally modest;
- the implementation is notebook-first and not yet a packaged library.

## Roadmap

Natural next extensions include:

1. add volatility- or regime-conditioned scenario generation;
2. test shrinkage / ridge / Bayesian VAR alternatives;
3. expand walk-forward validation with more dates and scenarios;
4. compare ERP engines across tail realism, dependence fit, and portfolio impact;
5. replace iid residual-variance reinjection with empirical residual bootstrap variants;
6. add an optional credit-spread block as an experimental extension;
7. refactor the notebook into reusable modules.

## Requirements

Typical Python packages used in the notebook:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `statsmodels`
- `yfinance`
- `pandas_datareader`
- `scipy`

## How to run

1. Clone the repository.
2. Install the required packages.
3. Open `multi_asset_scenario_engine.ipynb`.
4. Run all cells in order.

The notebook downloads market data live, so internet access is required. Some cells, especially the walk-forward VaR backtest, are computationally heavier than the main single-date scenario run.

## Repository structure

```text
.
├── multi_asset_scenario_engine.ipynb
└── README.md
```

If the notebook is stored inside a `notebooks/` folder in your local repository, keep the filename unchanged and adjust the path accordingly.

## Author

Saverio Lauriola
