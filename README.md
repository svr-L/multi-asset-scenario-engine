# Multi-Asset Scenario Engine

A modular Python research prototype for joint US equity-rates-credit-FX scenario generation under the physical measure, horizon repricing, and portfolio tail-risk analysis.

The project is designed as a lightweight market-risk / risk-strat engine rather than a point-forecasting notebook. It follows the pipeline:

```text
market data -> risk drivers -> joint scenarios -> horizon repricing -> portfolio tail risk
```

## Current stable release

The current public release focuses on a **one-week cross-asset horizon**. The engine models a broad US equity universe, the US Treasury curve, credit rating-bucket OAS spreads, and EUR/USD FX spot in a single joint state. Simulated states are mapped back into instrument values and portfolio losses.

The stable default is a **joint stationary block bootstrap** of the cross-asset state. A VAR(1) model is also fitted and reported as a diagnostic / challenger, but the block-bootstrap engine is the default because it is more parsimonious for the short horizon and preserves empirical cross-asset dependence.

## What the notebook does

### 1. Data layer

The notebook downloads and aligns:

- adjusted-close prices for a multi-sector US equity universe;
- a broad US equity market proxy;
- a 10-tenor US Treasury key-rate curve from FRED;
- daily Fama-French / Carhart factors;
- ICE BofA option-adjusted spreads across rating buckets from AAA to CCC;
- EUR/USD spot from FRED;
- Moody's Aaa/Baa yields for optional crisis-tail calibration.

### 2. Risk-driver blocks

The engine builds modular risk-driver blocks:

- **Equity:** selectable equity-premium engines (`latent_statistical`, `proxy_market`, `apt_ff3`, `apt_carhart`) plus compressed idiosyncratic residual factors;
- **Rates:** PCA factors from daily Treasury key-rate changes;
- **Credit:** PCA factors from rating-bucket OAS changes, with a parsimony cap to avoid near-degenerate credit dimensions;
- **FX:** currency-pair log-return factors.

All blocks are stacked into a single joint state so that equity/rates/credit/FX co-movements are simulated jointly rather than imposed after the fact.

### 3. Joint scenario generation

The stable one-week engine uses a stationary block bootstrap of the joint state. This keeps the model parsimonious and preserves empirical cross-asset dependence. The notebook also fits a VAR(1), reports stability diagnostics, and measures one-step explanatory power to assess whether parametric mean dynamics add value.

### 4. Horizon repricing

The simulated risk-driver states are translated into one-week horizon values for:

- equities;
- a synthetic semi-annual coupon Treasury note;
- corporate bonds priced from Treasury zero rates plus simulated rating-bucket OAS;
- a fully-funded EUR/USD forward.

This makes the notebook closer to a small cross-asset horizon-repricing engine than to a pure return-simulation exercise.

### 5. Portfolio tail-risk analytics

The portfolio layer uses capital-normalized instrument returns and computes:

- portfolio VaR and Expected Shortfall;
- weighted tail contributions;
- illustrative benchmark portfolios;
- long-only minimum-CVaR weights via the Rockafellar-Uryasev linear-programming formulation;
- a fresh-draw out-of-sample scenario check for tail-risk optimism.

### 6. Validation

The notebook includes distributional and dependence diagnostics:

- simulated vs historical one-week risk scaling;
- equity tail behaviour;
- cross-equity correlation preservation;
- equity-rates, equity-credit, and equity-FX dependence checks;
- credit-spread scaling by rating bucket;
- walk-forward VaR coverage using Kupiec and Christoffersen tests;
- PIT / rank diagnostics.

## Crisis-tail overlay

The common equity/rates/credit/FX sample is constrained by recent ICE BofA OAS availability and is relatively calm. To avoid pretending that a calm sample contains crisis tails, the notebook includes an **optional** crisis-tail overlay calibrated from long-history Moody's Baa/Aaa spread behaviour.

The default configuration leaves the calm-sample engine unchanged. Turning the crisis overlay on deliberately fattens the simulated tail and should be interpreted as a stress mode, not as the baseline calibration.

## Current scope and limitations

This is a research prototype, not a production risk system. Important limitations:

- the credit block is a rating-bucket OAS curve, not a full maturity-by-rating credit surface;
- corporate bonds are repriced using a flat rating OAS added to the Treasury curve;
- the public FRED ICE BofA OAS sample may be limited to a recent rolling window;
- the FX block is intentionally light, currently configured with EUR/USD;
- the default public release focuses on a one-week market-risk horizon;
- longer-horizon credit migration/default modelling is treated as a separate research extension.

## Roadmap

Planned extensions include:

1. refactoring the notebook into reusable modules under `src/`;
2. adding a separate one-year credit migration/default extension with block-specific long-horizon dynamics;
3. introducing mean-reverting dynamics for rates and credit spreads at longer horizons;
4. expanding FX and credit instrument coverage;
5. adding stronger rolling validation and ablation tables across alternative dynamics;
6. separating stable mainline notebooks from experimental research branches.

## Requirements

Typical packages used by the notebook:

- numpy
- pandas
- matplotlib
- scikit-learn
- statsmodels
- scipy
- yfinance
- pandas-datareader

Install with:

```bash
pip install -r requirements.txt
```

## How to run

1. Clone the repository.
2. Install the required packages.
3. Open `multi_asset_scenario_engine.ipynb`.
4. Run all cells in order.

The notebook downloads market data live, so internet access is required.

## Repository structure

```text
.
├── multi_asset_scenario_engine.ipynb
├── README.md
└── requirements.txt
```

## Author

Saverio Lauriola
