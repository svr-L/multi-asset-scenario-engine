# Multi-Asset Scenario Engine

A modular Python framework for joint equity-rates scenario generation under the physical measure, horizon repricing, and portfolio tail-risk analysis.

The current implementation is a research prototype designed to connect market risk modelling, cross-asset scenario generation, instrument repricing, and portfolio-level tail-risk diagnostics in a single workflow.

c Motivation

This project was built to bridge:

- market risk modelling
- scenario generation under the physical measure
- horizon repricing
- portfolio tail-risk analysis

Rather than focusing on point forecasts, the framework generates joint one-week cross-asset scenarios for equity and rates risk drivers, maps them back into instrument values, and evaluates portfolio-level loss distributions and tail-risk metrics.

The broader objective is to treat scenario generation and repricing as reusable building blocks for:

- ex-ante market risk analysis
- portfolio robustness checks
- cross-asset stress testing
- future extensions toward richer equity-premium and risk-free generators

## Current implementation

The current version includes the following components.

### 1. Data layer
- Daily market data for a prototype equity universe
- US Treasury key rates from FRED
- Unified time alignment and cleaning pipeline

### 2. Equity block
- Log-return construction
- Excess-return transformation using a dailyized short-rate proxy
- Observable market equity-premium proxy
- Diagnostic comparison with a latent common equity factor
- Residual decomposition and residual PCA compression

### 3. Rates block
- Yield-change transformation
- PCA-based reduction of the key-rate curve
- Interpretable low-dimensional factor structure, broadly aligned with level, slope, and curvature dynamics

### 4. Joint scenario engine
- Joint state representation combining:
  - equity-premium factor(s)
  - residual equity factors
  - rates curve factors
- Linear state dynamics via VAR(1)
- Scenario generation via stationary-bootstrap innovations

### 5. Horizon repricing
- Scenario-wise repricing of:
  - equities
  - a synthetic semi-annual coupon Treasury note
- Mapping from simulated factor states back to:
  - equity returns
  - yield-curve scenarios
  - instrument values and P&L

### 6. Portfolio analytics
- Capital-normalized one-week instrument returns
- Portfolio construction examples:
  - equal-weight
  - inverse-volatility
  - long-only minimum-CVaR
- Portfolio loss distributions
- Tail-risk metrics:
  - VaR
  - Expected Shortfall
  - weighted tail contributions

## Why this is different from a standard forecasting notebook

This is not a simple return-prediction exercise.

The framework is structured around the following pipeline:

1. transform raw market data into risk drivers
2. model joint driver dynamics
3. simulate future scenarios under P
4. reprice instruments at the horizon
5. evaluate portfolio-level tail risk

This makes it closer to a lightweight cross-asset market-risk and horizon-repricing engine than to a standalone forecasting notebook.

## Methodological overview

### Equity returns
For each equity, returns are represented as:

r_i,t = r_f,t + equity premium component + idiosyncratic component

The current prototype uses an observable market ERP proxy together with a residual factor block.

### Rates
The risk-free curve is represented through key rates and compressed through PCA, producing low-dimensional factors that approximately capture:

- level
- slope
- curvature

### Joint dynamics
The joint state vector is modelled with a VAR(1), while innovations are generated through stationary bootstrap to avoid relying purely on Gaussian i.i.d. shocks.

### Repricing
For each simulated horizon scenario:
- equities are repriced through simulated excess-return reconstruction
- the coupon Treasury note is repriced from the simulated yield-curve state and residual time-to-maturity

### Portfolio risk
Portfolio analytics are based on capital-normalized instrument returns, ensuring that risk comparisons are not distorted by differences in nominal instrument prices.

## Current results and interpretation

At the current stage, the framework already shows:

- coherent common-factor structure in equities
- interpretable curve-factor structure in rates
- realistic one-week risk scaling
- sensible scenario-wise repricing for both equities and bonds
- economically meaningful portfolio tail-risk rankings once returns are normalized by initial value

This makes the project already suitable as a prototype for:

- market risk analytics
- risk strat / cross-asset analytics
- portfolio robustness diagnostics

## Current limitations

This is still a research prototype, not a production library.

Main limitations at the current stage include:

- the prototype equity universe is still relatively narrow
- validation is still lightweight
- the equity-premium engine is not yet fully plug-and-play
- the rates block is PCA-based rather than a richer curve model
- scenario validation is not yet fully rolling / walk-forward

## Roadmap

Natural next extensions include:

1. Generalize from the current prototype universe to broader US equity
2. Add modular equity-premium engines, for example:
   - statistical factor engine
   - observable-factor / APT engine
   - hybrid engine
3. Add rolling / walk-forward validation
4. Compare alternative scenario engines in terms of:
   - tail-risk realism
   - dependence structure
   - portfolio impact
5. Refactor notebook logic into reusable modules

## Potential research extensions

This framework can naturally evolve into:

- broader US equity + Treasury scenario generation
- APT / Fama-French / Carhart-based equity-premium simulation
- cross-asset stress testing
- robust portfolio construction under simulated market states
- comparison of scenario engines for ex-ante risk and robustness analysis

The current implementation is notebook-first, but the architecture is explicitly designed to support a cleaner package-style refactor.

## Requirements

Typical Python packages used in the notebook:

- numpy
- pandas
- matplotlib
- scikit-learn
- statsmodels
- yfinance
- pandas_datareader
- scipy

## How to run
- Clone the repository
- Install the required packages
- Open the notebook
- Run all cells in order

The notebook downloads market data live, so internet access is required at execution time.

## Positioning

This project is best understood as a prototype for:

- joint scenario generation under the physical measure
- horizon repricing of equity and rates instruments
- portfolio tail-risk analysis

It is not intended as:

- a pure risk-neutral pricing library
- a production-grade market risk system
- a pure alpha-signal forecasting project

## Repository structure

A possible modular structure is:

```text
.
├── notebooks/
│   └── multi_asset_scenario_engine.ipynb
├── src/
│   ├── data/
│   ├── drivers/
│   ├── models/
│   ├── repricing/
│   └── analytics/
└── README.md

```

---

## Author

Saverio Lauriola
