# Consumer interface

The Scenario Engine is autonomous. TAA, portfolio construction, stress testing and pricing systems are downstream consumers.

## Validated contract
- measure: physical / P
- horizon: 1M / 21 business days
- model: `raw_duration|hierarchical|direct_H`
- drivers: `EQ_EW`, `RATE_DGS2`, `RATE_DGS10`, `RATE_DGS30`, `CREDIT_Aaa_spr`, `CREDIT_Baa_spr`, `FX_EURUSD`

A portfolio consumer should preferably receive a repriced instrument/asset-return scenario matrix rather than raw risk drivers. The consumer can then compute its own moments, probabilities, quantiles, expected shortfall or utility from the same scenario law.

Do not rerun model selection inside the consumer. The V33 selector/regime layers were rejected. Research is frozen through 2026-05-01; live use should refit the same validated architecture to the desired as-of date without re-selecting architecture.
