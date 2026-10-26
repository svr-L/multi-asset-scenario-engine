# Objective-aware routing

The Scenario Engine models a joint predictive distribution. It is not tied to a single downstream task.
The routing layer therefore maps `(forecast date, horizon, statistical objective)` to a candidate scenario
law using only previously observed validation losses.

## Built-in objectives

| Objective | Primary score | Required object |
|---|---|---|
| joint_distribution | Energy Score | multivariate draws |
| dependence | Variogram Score | multivariate draws + block labels |
| marginals | marginal CRPS | multivariate draws |
| projections | projection CRPS | draws + user-defined linear projections |
| tail_risk | FZ0 VaR/ES | scalar loss projection |
| density | NLL | comparable predictive log-density |

NLL is intentionally not reconstructed from a KDE merely to make an implicit scenario generator look like
a density model. It is used only when competing candidates expose comparable predictive densities.

## Policies

- `hard`: minimum historical mean loss;
- `one_se`: paired one-standard-error eligibility followed by the simplest model;
- `soft`: scenario mixture with validation-based weights, which must be re-scored from the mixed scenarios.

## Point-in-time constraint

For an outer forecast date `t`, only score observations with `asof < t` are eligible. Router performance
must then be evaluated on the realised forecast at `t`. This makes routing itself another model subject to
OOS validation rather than an ex-post reporting rule.

## Downstream systems

Any external consumer may use the selected scenarios, model weights and selection diagnostics. If a
consumer wants decision-aware model selection, it can compute its own realised decision loss and append it
to the generic router history schema. The downstream decision logic remains outside this repository.
