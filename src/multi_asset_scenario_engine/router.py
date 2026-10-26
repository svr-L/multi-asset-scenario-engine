"""Objective-aware, horizon-aware routing across scenario-model candidates.

The router is deliberately downstream-agnostic: it selects predictive scenario laws from
point-in-time validation losses. Portfolio construction, stress testing, pricing or other
consumers remain outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RouteDecision:
    asof: pd.Timestamp
    horizon: str
    objective: str
    policy: str
    selected_model: str
    best_mean_model: str
    weights: dict[str, float]
    mean_losses: dict[str, float]
    n_history: int
    gap_best_second: float
    paired_se_best_second: float
    z_gap: float


class ObjectiveRouter:
    """Select scenario models using only validation losses available before the forecast date.

    Parameters
    ----------
    score_history:
        Long DataFrame with columns ``asof, horizon, objective, model, loss``. Lower loss is better.
    complexity:
        Optional model-complexity ranking used by the one-standard-error rule. Smaller is simpler.
    min_history:
        Minimum number of common past validation dates required to route automatically.
    lookback:
        Optional number of most recent validation dates retained per horizon/objective.
    """

    REQUIRED = {"asof", "horizon", "objective", "model", "loss"}

    def __init__(self, score_history, complexity=None, min_history=20, lookback=None):
        h = pd.DataFrame(score_history).copy()
        missing = self.REQUIRED.difference(h.columns)
        if missing:
            raise ValueError(f"score_history missing columns: {sorted(missing)}")
        h["asof"] = pd.to_datetime(h["asof"])
        h["loss"] = pd.to_numeric(h["loss"], errors="coerce")
        h = h.dropna(subset=["asof", "horizon", "objective", "model", "loss"])
        self.history = h.sort_values("asof").reset_index(drop=True)
        self.complexity = dict(complexity or {})
        self.min_history = int(min_history)
        self.lookback = None if lookback is None else int(lookback)

    def _pivot(self, asof, horizon, objective):
        asof = pd.Timestamp(asof)
        g = self.history[
            (self.history["asof"] < asof)
            & (self.history["horizon"].astype(str) == str(horizon))
            & (self.history["objective"].astype(str) == str(objective))
        ]
        P = g.pivot_table(index="asof", columns="model", values="loss", aggfunc="mean").sort_index()
        if self.lookback is not None:
            P = P.tail(self.lookback)
        P = P.dropna(axis=0, how="any")
        if len(P) < self.min_history:
            raise ValueError(
                f"Only {len(P)} common historical validation dates for {objective}/{horizon}; "
                f"need at least {self.min_history}."
            )
        return P

    def leaderboard(self, asof, horizon, objective):
        P = self._pivot(asof, horizon, objective)
        means = P.mean().sort_values()
        ses = P.std(ddof=1) / np.sqrt(len(P))
        return pd.DataFrame({"mean_loss": means, "se_loss": ses.reindex(means.index),
                             "complexity": [self.complexity.get(m, np.inf) for m in means.index]})

    def select(self, asof, horizon, objective, policy="one_se", temperature=1.0):
        P = self._pivot(asof, horizon, objective)
        means = P.mean().sort_values()
        best = str(means.index[0])
        models = list(means.index)
        selected = best

        if policy == "hard":
            weights = {m: float(m == best) for m in models}

        elif policy == "one_se":
            # Paired one-standard-error rule: retain models whose average loss is no more
            # than one paired SE above the empirical best, then choose the simplest.
            eligible = []
            for m in models:
                d = (P[m] - P[best]).dropna()
                se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
                if float(d.mean()) <= se + 1e-15:
                    eligible.append(m)
            selected = min(eligible, key=lambda m: (self.complexity.get(m, np.inf), means[m], m))
            weights = {m: float(m == selected) for m in models}

        elif policy == "soft":
            temp = float(temperature)
            if temp <= 0:
                raise ValueError("temperature must be positive")
            # Objective-specific scale from historical losses; this avoids mixing units across objectives.
            scale = float(np.nanmedian(P.std(ddof=1).to_numpy()))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = max(float(np.nanstd(P.to_numpy())), 1e-12)
            delta = (means - means.iloc[0]).to_numpy(dtype=float)
            raw = np.exp(-delta / (temp * scale))
            raw /= raw.sum()
            weights = dict(zip(models, raw.astype(float)))
            selected = str(models[int(np.argmax(raw))])
        else:
            raise ValueError("policy must be one of: hard, one_se, soft")

        gap = np.nan
        se_gap = np.nan
        z_gap = np.nan
        if len(models) > 1:
            second = str(means.index[1])
            # Positive d means the second-best model has larger loss than the best.
            d = (P[second] - P[best]).dropna()
            gap = float(d.mean())
            se_gap = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
            z_gap = gap / se_gap if np.isfinite(se_gap) and se_gap > 0 else np.inf

        return RouteDecision(
            asof=pd.Timestamp(asof), horizon=str(horizon), objective=str(objective), policy=str(policy),
            selected_model=selected, best_mean_model=best, weights=weights,
            mean_losses={str(k): float(v) for k, v in means.items()}, n_history=len(P),
            gap_best_second=gap, paired_se_best_second=se_gap, z_gap=float(z_gap),
        )

    def walk_forward_select(self, objective, horizon, policy="one_se", temperature=1.0):
        """Nested point-in-time model selection for hard/one-SE policies.

        For a soft scenario mixture, its proper score must be recomputed from the mixed scenarios;
        averaging component-model losses is generally not the score of the mixture. Therefore this
        helper intentionally does not claim a soft-mixture OOS loss.
        """
        if policy == "soft":
            raise ValueError("Soft mixtures must be re-scored from mixed scenarios, not averaged losses.")
        g = self.history[
            (self.history["objective"].astype(str) == str(objective))
            & (self.history["horizon"].astype(str) == str(horizon))
        ]
        dates = sorted(pd.to_datetime(g["asof"].unique()))
        rows = []
        for t in dates:
            try:
                d = self.select(t, horizon, objective, policy=policy, temperature=temperature)
            except ValueError:
                continue
            now = g[g["asof"] == t].set_index("model")["loss"]
            if d.selected_model not in now:
                continue
            rows.append({"asof": t, "horizon": str(horizon), "objective": str(objective),
                         "policy": policy, "selected_model": d.selected_model,
                         "best_mean_model": d.best_mean_model,
                         "router_loss": float(now[d.selected_model]),
                         "n_history": d.n_history, "z_gap": d.z_gap})
        return pd.DataFrame(rows)


def mix_scenarios(candidates: Mapping[str, np.ndarray], weights: Mapping[str, float],
                  n_scenarios=None, rng=None):
    """Construct a finite Monte Carlo draw from a mixture of candidate scenario laws."""
    rng = np.random.default_rng() if rng is None else rng
    names = [m for m in weights if weights[m] > 0]
    if not names:
        raise ValueError("No positive model weights")
    arrays = {m: np.asarray(candidates[m], dtype=float) for m in names}
    k = {a.shape[1] for a in arrays.values()}
    if len(k) != 1:
        raise ValueError("All candidate scenario matrices must have the same number of columns")
    if n_scenarios is None:
        n_scenarios = min(len(a) for a in arrays.values())
    n_scenarios = int(n_scenarios)
    p = np.asarray([weights[m] for m in names], dtype=float)
    p /= p.sum()
    owner = rng.choice(len(names), size=n_scenarios, p=p)
    out = np.empty((n_scenarios, next(iter(k))), dtype=float)
    for j, m in enumerate(names):
        loc = np.where(owner == j)[0]
        if len(loc):
            ii = rng.integers(0, len(arrays[m]), size=len(loc))
            out[loc] = arrays[m][ii]
    return out
