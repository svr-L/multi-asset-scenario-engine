# Model adjudication: final V33 decision

The controlled family was `{raw duration, duration-PIT} × {blockwise, hierarchical} × {direct-H, recursive-1W}` and was evaluated point-in-time with Energy Score, Variogram Score, marginal CRPS, projection CRPS and a reference FZ0 VaR/ES score.

## V32.2 — objective specialization
A calibration-only specialist was chosen separately for each objective and compared on an untouched 2021–2026 holdout with one universal engine selected ex ante. Specialization did not show robust broad gains. The tail objective had a positive mean gain, but its paired interval still crossed zero.

## V33 — regime conditioning
A transparent normal/stress state was built point-in-time from equity volatility, rates volatility, lower-quality credit stress and cross-asset dependence instability, with hysteresis. No objective passed the pre-specified deployment gate against both the fixed-objective baseline and the universal engine.

## Production decision
The validated conservative 1M policy is `raw_duration|hierarchical|direct_H`. Rejected selector/regime layers remain research challengers, not production features.
