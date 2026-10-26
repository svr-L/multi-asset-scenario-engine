# Research decisions

## Stable backbone inherited from v23
Physical-measure multi-asset simulation, asset-class factors, GJR/EWMA FHS, stationary block bootstrap,
horizon-consistent credit modelling, multi-horizon repricing and portfolio VaR/ES remain the stable core.

## Dependence
The v23 blockwise architecture remains the hard benchmark. The later global/local hierarchy is the
structured research architecture because it directly targets cross-block covariance and has shown
incremental OOS value without dominating every horizon.

## Duration-conditioned PIT
Random-duration historical blocks can be collapsed to cumulative moves and transformed through
duration-specific empirical CDFs before optional pooling. The effect is horizon-dependent, so PIT
pooling remains a challenger rather than a mandatory default.

## Propagation
Recursive 1W propagation can improve selected pairwise relationships but is not consistently better on
the full joint predictive distribution. Direct-H remains the default.

## Scoring
Energy Score and Variogram Score are used jointly so improvements in the full distribution and in
dependence can be distinguished.


## V32.2 / V33 final monthly decision
- Validated horizon: 1M / 21 business days.
- Objective specialization: not broadly validated against the universal ex-ante engine on untouched holdout.
- Normal/stress regime conditioning: rejected.
- Production selection: `raw_duration|hierarchical|direct_H`.
- Downstream consumers should not rerun the rejected selector.
