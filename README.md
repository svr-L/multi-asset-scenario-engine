# Multi-Asset Scenario Engine

A modular Python research engine for **joint cross-asset scenario generation under the physical measure**, **instrument-level horizon repricing**, and **portfolio tail-risk validation** across US equities, Treasury rates, credit spreads and FX.

The repository is the cleaned standalone **v20 scenario-engine release**. The later Flexible Probabilities / Entropy Pooling work is deliberately kept separate: this project owns the physical scenario distribution, repricing stack and risk validation layer.

## What the engine does

The workflow is deliberately closer to a market-risk / risk-strat stack than to a return-forecasting notebook:

1. build transformed equity, Treasury, credit and FX risk drivers;
2. compress each block into parsimonious factors while retaining residual risk;
3. generate joint physical-measure scenarios with block resampling and optional conditional volatility / level mean reversion;
4. reconstruct horizon market states;
5. reprice equities, a Treasury note, corporate bonds and an FX forward;
6. produce portfolio VaR, Expected Shortfall, tail contributions and minimum-CVaR allocations;
7. validate VaR and ES in a point-in-time walk-forward design.

## Research architecture

### Equity-premium block

The equity block is plug-and-play. The release supports:

- latent statistical factor;
- observed market-proxy factor;
- Fama-French 3-factor APT;
- Carhart-style APT with momentum.

Common equity structure is combined with a PCA-compressed residual block so idiosyncratic risk is not lost when the systematic component is simulated.

### Treasury-rate block

A 10-tenor US Treasury grid is transformed into daily yield changes and compressed through PCA. Level/slope/curvature diagnostics are reported. For longer horizons, an optional OU/Vasicek pull is applied **only** to level series that pass a Dickey-Fuller-style gate; non-gated series retain random-walk dynamics.

### Credit block

Two credit representations are available:

- current ICE BofA rating-bucket OAS data (AAA to CCC);
- a long-history Moody's Aaa/Baa two-point driver used to recover multi-cycle credit dynamics when the public OAS history is truncated.

In long-history mode, simulated Moody's-factor moves are mapped to today's rating-bucket OAS anchors through overlap-estimated sensitivities **for repricing only**. No pre-history of the unavailable OAS buckets is fabricated.

### FX block

EUR/USD is the default FX factor. The design accepts multiple pairs and reprices a long-EUR forward under a simplified covered-interest-parity carry convention.

## Scenario generation

The mainline generator uses a **joint stationary block bootstrap**. The notebook also retains VAR(1) and Student-t copula implementations as research challengers, but they are not the preferred v20 mainline.

For state-dependent risk, Filtered Historical Simulation can standardize each factor by its own conditional volatility. `VOL_PARAMS_MODE="estimated_gjr"` estimates factor-level GJR-GARCH dynamics and keeps leverage only where a nested likelihood-ratio gate supports it; factors without convincing volatility clustering fall back toward the simpler specification.

A fixed expected block length of 10 remains the practical default. A Politis-White selector is retained as a diagnostic because the very short blocks it can select after filtering were found to weaken cross-asset co-tail preservation.

## Crisis-tail overlay

An optional stress layer uses long-history Moody's credit information to calibrate the incidence and scale of crisis-like moves. It is **off by default** so the baseline physical generator remains separately identifiable from explicit stress overlays.

## 1W and 1Y horizons

The same framework supports:

- **1W:** short-horizon market-risk repricing;
- **1Y:** annual carry plus CreditMetrics-style rating migration/default, recovery and migrated-spread repricing.

The annual branch is intentionally treated as more model-dependent: long-horizon rates, spreads, expected returns, migration and FX carry require assumptions that are negligible at one week.

## Instrument repricing

Scenario states feed instrument-level horizon values for:

- equities through cumulative simulated log returns;
- a semi-annual coupon Treasury note through the simulated key-rate curve;
- BBB and BB corporate bonds through Treasury + rating OAS, with migration/default in 1Y mode;
- an EUR/USD FX forward through simulated spot, USD discounting and a declared simplified foreign-funding assumption.

The pricing layer is designed for **horizon P&L / market-risk analysis**, not as a substitute for a production risk-neutral pricing library.

## Portfolio risk

Instrument P&Ls are converted to capital-normalized returns. The portfolio layer includes:

- equal-weight and sleeve benchmarks;
- inverse-volatility allocation;
- VaR and Expected Shortfall;
- tail-contribution attribution;
- exact long-only minimum-CVaR optimization via linear programming;
- fresh-draw checks to distinguish scenario-set overfitting from stable tail-risk improvement.

## Validation

Validation is a first-class part of v20 rather than an afterthought.

### Distribution and dependence diagnostics

The notebook compares simulated and historical:

- marginal horizon risk scaling;
- excess kurtosis;
- cross-equity correlation;
- equity/rates, equity/credit and equity/FX dependence;
- lower-tail equity co-exceedance.

### Walk-forward VaR

The point-in-time backtest refits the model at each as-of date and reports:

- Kupiec unconditional coverage;
- Christoffersen independence;
- Christoffersen joint conditional coverage;
- coverage by volatility regime and calendar year.

`BACKTEST_MODE="full"` uses the full eligible historical window rather than only the latest dates. It is computationally expensive but is the preferred research validation setting.

### Expected-Shortfall backtest

v20 additionally tests the risk quantity optimized by the min-CVaR layer. The implementation follows the Acerbi-Szekely direct-backtest logic, using the model's full predictive distribution at each date and simulation-based null distributions for the test statistics rather than imposing a parametric loss law.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── notebooks/
│   └── multi_asset_scenario_engine.ipynb
├── src/
│   └── multi_asset_scenario_engine/
│       ├── config.py
│       ├── data.py
│       ├── equity.py
│       ├── factors.py
│       ├── dynamics.py
│       ├── simulation.py
│       ├── repricing.py
│       ├── portfolio.py
│       ├── backtesting.py
│       └── diagnostics.py
└── tests/
    ├── test_repricing.py
    └── test_risk.py
```

The notebook imports the package. The implementation is therefore maintained in one place rather than duplicated between `.ipynb` and `.py` files.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Or install the pinned research dependencies:

```bash
pip install -r requirements.txt
```

## Running the notebook

```bash
jupyter lab notebooks/multi_asset_scenario_engine.ipynb
```

The notebook downloads data from Yahoo Finance, FRED and Kenneth French sources, so internet access is required. The default cleaned notebook uses the mature long-history/GJR/OU scenario configuration but keeps the shorter recent backtest for practical runtime. Set:

```python
RUN_FULL_VALIDATION = True
```

to activate the full v20 walk-forward VaR/ES validation.

## Tests

The unit tests are deliberately offline and focus on deterministic numerical building blocks:

```bash
pytest -q
```

They cover bootstrap indexing, bond/FX-forward repricing invariants, coverage-test outputs and minimum-CVaR constraints. The live-data research workflow is validated in the notebook rather than mocked as a unit test.

## Important limitations

- This is a research prototype, not a production market-risk platform.
- Treasury CMT inputs are used as a simplified zero-curve representation for the horizon-pricing demo.
- The long-history Moody's credit driver is only two-dimensional; the mapping into rating-bucket OAS is a sensitivity bridge, not a reconstructed historical spread surface.
- The FX foreign funding curve is simplified.
- 1Y migration/default and level dynamics are materially more assumption-sensitive than the 1W mainline.
- Copula and crisis-overlay modules are challenger/stress tools, not automatically superior to the block-bootstrap baseline.

## Relationship to the Entropy Pooling project

The separate Entropy Pooling / Flexible Probabilities project treats this engine's physical scenarios as a **prior support** and reweights scenario probabilities under state information and views. Keeping the repositories separate makes the research question explicit:

- **Scenario Engine:** how should the physical cross-asset distribution be generated, repriced and validated?
- **Entropy Pooling:** how should that distribution be conditioned on state information, and does the conditioning add predictive value?

## Author

Saverio Lauriola
