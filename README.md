# Multi-Asset Scenario Engine

A modular Python research engine for **joint cross-asset scenario generation under the physical measure** across US equities, Treasury rates, credit spreads and FX, with **multi-horizon instrument repricing**, portfolio tail-risk analytics and **point-in-time VaR / Expected-Shortfall backtesting** of an equal-weight equity book.

This repository is the cleaned standalone **v23 release**. It is the stable pre-hierarchical mainline: the scenario generator remains blockwise by asset-class factors, while v21-v23 make the horizon, drift, credit-variance and conditional-volatility treatment materially more defensible. The separate Entropy Pooling / Flexible Probabilities work is intentionally kept outside this repository.

## What changed in v23

Relative to the earlier v20 release, the main additions are methodological rather than cosmetic.

### 1. Multi-horizon risk engine

The same physical-measure engine now runs at:

- **1W** — 5 business days;
- **1M** — 21 business days;
- **3M** — 63 business days;
- **1Y** — 252 business days.

The horizon flows through scenario accumulation, rate / credit / FX state reconstruction, bond and forward repricing, migration/default and walk-forward validation.

Backtest windows are explicitly **non-overlapping at each horizon**. This matters statistically: a 1Y backtest made from weekly-spaced forecast windows would contain almost the same realized year in adjacent observations and would invalidate the usual coverage-test reference laws.

### 2. Drift is separated from risk dynamics

`DRIFT_MODE` makes the expected-return assumption explicit:

- `sample` — the historical sample means used by the earlier engine;
- `factor` — stock expected excess returns are `beta × long-run Fama-French premia`, with zero alpha; rates, credit and FX have zero drift apart from gated level mean reversion;
- `zero` — zero excess drift throughout.

The mature v23 research run uses **factor drift**, preventing a recent bull-market sample mean from being mechanically extrapolated over 3M/1Y horizons.

### 3. Credit compression residuals can be restored

The credit state stays parsimonious, but v23 no longer requires the omitted bucket-specific variance to disappear.

With `CREDIT_RESIDUAL_MODE="restore"` the engine keeps the low-dimensional systematic credit drivers in the joint state and re-injects, at the reporting horizon, empirical sums of the compression residuals:

- PCA residuals for the direct rating-bucket representation;
- bridge residuals when the long-history Moody's two-point credit block is used.

The residual draw has its own RNG stream, so restoring bucket-specific variance does not alter the equity/rates/FX systematic scenario path.

### 4. Horizon-frequency Moody's-to-OAS bridge

The long-history credit block uses real Moody's Aaa/Baa seasoned-yield spreads for dynamics and today's ICE BofA rating-bucket OAS levels for repricing.

The bridge from the two-point Moody's driver to rating buckets can now be estimated on **overlapping horizon changes**, rather than only daily changes. At an `h`-day reporting horizon the fitted move and its residual are therefore orthogonal at that horizon, so systematic and restored residual variances add consistently.

No unavailable historical rating-bucket OAS surface is fabricated.

### 5. Consistent GJR-GARCH variance targeting

The v18-v22 GJR parameterization used

`p = alpha + beta + gamma / 2`

which is exact only under the symmetric-innovation approximation that one half of squared innovations falls on negative days.

v23 instead measures

`m = E[1(eps < 0) eps²] / E[eps²]`

on the actual innovation / standardized-residual sample and targets

`p = alpha + beta + gamma m`.

This matters for negatively skewed equity factors: the earlier approximation could understate effective persistence and let simulated variance drift above its target as the horizon lengthened. v23 applies the same empirical-downside share consistently in estimation and in the FHS forward recursion.

### Corrections to the public v20

v21-v23 also fix defects present in v20. Results produced with v20 should be read with these in mind:

- **Horizon handling.** v20 mapped any `HORIZON_MODE` other than `"1W"` to 252 days without warning, with migration off and an FX forward that had expired before the horizon. Unknown modes now raise an error.
- **Overlapping long-horizon backtest.** v20 stepped the backtest by 5 business days at every horizon. At 1Y, consecutive windows shared about 98% of their days, so the 1Y Kupiec, Christoffersen and Acerbi–Székely p-values were invalid, not merely weak. Windows are now non-overlapping at each horizon.
- **Full-mode start date.** On a sample shorter than `BACKTEST_MIN_TRAIN_DAYS`, v20's `"full"` backtest silently started at the last available date and produced a single as-of date. It now stops with an explicit message.
- **Out-of-sample lab.** When re-evaluating optimized weights on a fresh draw, v20 dropped coupon income and, at 1Y, the corporates' migration/default leg, so the in-sample and out-of-sample columns were priced differently. Both now use the same repricing function.
- **GJR variance targeting** (v18-v22): see item 5 above.

## Research architecture

### Equity block

The equity-premium component is plug-and-play:

- latent statistical factor;
- observed market-proxy factor;
- Fama-French 3-factor APT;
- Carhart-style APT with momentum.

Common equity structure is combined with a PCA-compressed residual block, so systematic and idiosyncratic risk are both retained.

### Treasury-rate block

A 10-tenor Treasury grid is transformed into daily yield changes and compressed through PCA. Level / slope / curvature diagnostics are retained for interpretation.

For longer horizons, optional OU/Vasicek pulls are applied only to level series that pass a Dickey-Fuller-style gate; non-gated series remain random walks.

### Credit block

Two driver representations are available:

- ICE BofA rating-bucket OAS;
- long-history Moody's Aaa/Baa spreads relative to the 10Y Treasury.

The mature v23 setup uses the second for long-run dynamics, then maps simulated moves to today's rating-bucket OAS through the sensitivity bridge described above.

From 1M onward, corporate bonds can also undergo CreditMetrics-style rating migration/default, with the one-year transition matrix scaled to subannual horizons through a repaired continuous-time generator.

### FX block

EUR/USD is the default FX driver. A long-EUR forward is repriced jointly with simulated spot and the USD Treasury curve under a simplified fixed foreign-funding convention.

## Scenario generation

The mainline generator is a **joint stationary block bootstrap** over the factor state.

Filtered Historical Simulation can standardize each factor by its own conditional variance before resampling. `VOL_PARAMS_MODE="estimated_gjr"` estimates factor-specific GJR-GARCH dynamics, with evidence gates for clustering / leverage and the v23 empirical-downside variance-targeting correction.

The repository retains VAR(1), factor Student-t copula and asset-level Student-t copula implementations as research challengers. They are not the default.

## Instrument repricing

Scenario states feed horizon values for:

- equities via cumulative simulated log returns;
- a coupon Treasury note via the simulated key-rate curve;
- BBB and BB corporate bonds via Treasury + rating OAS, including migration/default at longer horizons;
- an EUR/USD FX forward via simulated spot and USD discounting.

The pricing layer is for **horizon P&L / market-risk analysis**, not risk-neutral derivative valuation.

## Portfolio layer

Capital-normalized scenario returns support:

- equal-weight and sleeve benchmark portfolios;
- inverse-volatility allocation;
- VaR and Expected Shortfall;
- tail-contribution attribution;
- exact long-only minimum-CVaR optimization via linear programming;
- fresh-draw evaluation of optimized portfolios.

## Point-in-time validation

The full validation mode refits the engine on an **expanding point-in-time sample** and simulates multiple horizons from each refit.

The reference v23 run used the mature configuration:

```python
SAMPLE_MODE = "long"
CREDIT_BLOCK_SOURCE = "moody_2pt"
CREDIT_RESIDUAL_MODE = "restore"
CREDIT_BRIDGE_FREQUENCY = "horizon"
DRIFT_MODE = "factor"
CONDITIONAL_VOL = "ewma_fhs"
VOL_PARAMS_MODE = "estimated_gjr"
LEVEL_MEAN_REVERSION = "ou"
DRIVER_DYNAMICS = "block_bootstrap"
BACKTEST_MODE = "full"
BACKTEST_HORIZONS = ("1W", "1M", "3M", "1Y")
```

It produced 219 point-in-time refits from 2004-08-09 to 2026-04-08. On the equal-weight equity book at 95% VaR / ES:

| Horizon | Windows | VaR breaches | Breach rate | Kupiec p | CC p | ES Z2 | ES Z2 p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1W | 219 | 9 | 4.1% | 0.533 | 0.559 | 0.085 | 0.595 |
| 1M | 219 | 11 | 5.0% | 0.988 | 0.557 | 0.175 | 0.704 |
| 3M | 73 | 2 | 2.7% | 0.334 | 0.593 | 0.630 | 0.876 |
| 1Y | 19 | 1 | 5.3% | 0.958 | 0.942 | -0.742 | 0.171 |

The 1Y row is **descriptive, not strong validation**: 19 non-overlapping annual windows provide very little power.

At 1W the regime-sliced breach rates were 5.5% / 4.1% / 2.7% across calm / normal / stressed volatility terciles, with VaR increasing materially with the volatility state. No headline VaR/ES calibration test rejected at 5% in the reference v23 run.

## Dependence diagnostics and an open limitation

v23 explicitly separates **current-state**, **neutral-state** and historical dependence.

For equities, the low current cross-equity correlation in the reference run was largely explained by the volatility state: moving all factor variances to their long-run levels closed about 91% of the gap to pooled historical cross-equity correlation.

Cross-asset dependence remained a genuine open issue. In particular, the historical stock-yield relationship changed sign in the recent regime, while the blockwise engine still pooled a much longer history. v23 therefore treats regime-sensitive cross-asset dependence as a limitation rather than hiding it. That diagnosis is the natural motivation for a later hierarchical / regime-aware cross-asset factor layer; it is **not** part of this v23 release.

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
│       ├── drift.py
│       ├── simulation.py
│       ├── repricing.py
│       ├── portfolio.py
│       ├── backtesting.py
│       └── diagnostics.py
└── tests/
    ├── test_repricing.py
    ├── test_risk.py
    └── test_v23.py
```

The notebook imports the package; reusable model logic is therefore maintained in one place rather than duplicated between `.ipynb` and `.py` files.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

## Run the notebook

```bash
jupyter lab notebooks/multi_asset_scenario_engine.ipynb
```

The notebook downloads market data from Yahoo Finance, FRED and Kenneth French sources, so internet access is required.

On Google Colab, clone the repository first and run the notebook from its `notebooks/` folder, so that the first cell finds `src/` (for example `!git clone <repository URL>` followed by `%cd <repository>/notebooks`).

The clean notebook defaults to the mature v23 model configuration but keeps the expensive full multi-horizon validation off. Set:

```python
RUN_FULL_VALIDATION = True
```

to run the 1W / 1M / 3M / 1Y point-in-time validation.

## Tests

Offline deterministic tests can be run with:

```bash
pytest -q
```

They cover the existing repricing / risk invariants plus v23-specific checks for:

- empirical-downside GJR targeting;
- horizon-frequency credit bridging;
- subannual transition-matrix scaling;
- factor-premium drift construction.

## Important limitations

- Research implementation, not a production risk platform.
- Treasury CMT inputs are a simplified proxy for a full zero-coupon curve.
- The long-history Moody's credit state is only two-dimensional.
- The OAS bridge is a sensitivity map, not reconstructed historical rating-bucket data.
- Corporate migration/default is based on an illustrative average transition matrix.
- The foreign FX funding curve is held fixed.
- 1Y validation is low-powered.
- Walk-forward VaR/ES validation covers the equal-weight equity book only; the Treasury, corporate-bond and FX-forward instruments and the min-CVaR allocation are repriced and analysed, but not backtested against realized history.
- GJR leverage terms are also fitted on PCA-derived factors, whose sign follows scikit-learn's convention (changed in 1.5). scikit-learn >= 1.5 is required to reproduce the reference run.
- The blockwise v23 factor structure does not explicitly model a latent common cross-asset factor; regime-dependent stock-bond / equity-credit dependence can therefore remain attenuated.
- Copula and crisis-overlay modules are challengers / stress tools, not automatically superior to the block-bootstrap mainline.

## Relationship to the Entropy Pooling research

The separate state-conditioned Entropy Pooling project can treat this engine's physical scenarios as **prior support** and reweight their probabilities under state information / views.

Keeping the repositories separate makes the questions explicit:

- **Scenario Engine:** how should the physical cross-asset distribution be generated, repriced and validated?
- **State-conditioned Entropy Pooling:** where should state information enter — prior probabilities, views, or both — and does the conditioning add predictive value?

## Author

Saverio Lauriola
