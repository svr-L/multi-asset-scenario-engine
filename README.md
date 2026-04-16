# Multi-Asset Scenario Engine

A cross-asset scenario engine for equity-premium and yield-curve risk drivers, one-week horizon repricing, and portfolio tail-risk analytics under the physical measure.

The project currently implements a prototype built on a basket of US equities and a synthetic coupon Treasury note, with a design intended to generalize toward broader US equity universes and alternative equity-premium engines.

## Motivation

This project was built to bridge:

- **market risk modelling**
- **scenario generation under the physical measure**
- **horizon repricing**
- **portfolio tail-risk analysis**

Rather than focusing on point forecasts, the framework generates **joint one-week cross-asset scenarios** for equity and rates risk drivers, maps them back into instrument values, and evaluates portfolio-level loss distributions and tail-risk metrics.

The underlying idea is to treat scenario generation and repricing as reusable building blocks for:

- ex-ante market risk analysis,
- portfolio robustness checks,
- cross-asset stress testing,
- and future extensions toward richer equity-premium and risk-free generators.

## Current implementation

The current version includes the following components.

### 1. Data layer
- Daily market data for a prototype equity universe
- US Treasury key rates from FRED
- Unified time alignment and cleaning pipeline

### 2. Equity block
- Log-return construction
- Excess-return transformation using a dailyized short-rate proxy
- Observable **market equity-premium proxy**
- Diagnostic comparison with a latent common equity factor
- Residual decomposition and **residual PCA compression**

### 3. Rates block
- Yield-change transformation
- PCA-based reduction of the key-rate curve
- Interpretable **level / slope / curvature-style** factor structure

### 4. Joint scenario engine
- Joint state representation combining:
  - equity-premium factor(s),
  - residual equity factors,
  - rates curve factors
- Linear state dynamics via **VAR(1)**
- Scenario generation via **stationary-bootstrap innovations**

### 5. Horizon repricing
- Scenario-wise repricing of:
  - equities
  - a synthetic semi-annual coupon Treasury note
- Mapping from simulated factor states back to:
  - equity returns,
  - yield curve scenarios,
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

The framework is designed around the following pipeline:

1. **transform raw market data into risk drivers**
2. **model joint driver dynamics**
3. **simulate future scenarios under P**
4. **reprice instruments at the horizon**
5. **evaluate portfolio-level tail risk**

This makes it closer to a lightweight **cross-asset market-risk and horizon-repricing engine** than to a standalone forecasting notebook.

## Project structure

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
