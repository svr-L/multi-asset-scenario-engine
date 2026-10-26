"""Proper multivariate scores and paired stationary-bootstrap inference."""
from __future__ import annotations
import numpy as np
from .dynamics import politis_white_block_length

def energy_score(draws, y, scale=None):
    X = np.asarray(draws, dtype=float)
    yy = np.asarray(y, dtype=float)
    if scale is not None:
        s = np.asarray(scale, dtype=float)
        X = X / s[None, :]
        yy = yy / s
    t1 = float(np.linalg.norm(X - yy[None, :], axis=1).mean())
    n2 = (len(X) // 2) * 2
    if n2 < 2:
        return t1
    pair = np.linalg.norm(X[:n2:2] - X[1:n2:2], axis=1).mean()
    return float(t1 - 0.5 * pair)

def variogram_score(draws, y, scale=None, labels=None, p=0.5):
    X = np.asarray(draws, dtype=float)
    yy = np.asarray(y, dtype=float)
    if scale is not None:
        s = np.asarray(scale, dtype=float)
        X = X / s[None, :]
        yy = yy / s
    labels = None if labels is None else np.asarray(labels, dtype=object)
    groups = {"EQ_RATES": [], "EQ_CREDIT": [], "RATES_CREDIT": [], "FX_CROSS": []}
    terms = []
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            obs = abs(yy[i] - yy[j]) ** float(p)
            pred = np.mean(np.abs(X[:, i] - X[:, j]) ** float(p))
            term = float((obs - pred) ** 2)
            terms.append(term)
            if labels is not None:
                a, b = labels[i], labels[j]
                pair = {a,b}
                if pair == {"equity","rates"}: groups["EQ_RATES"].append(term)
                elif pair == {"equity","credit"}: groups["EQ_CREDIT"].append(term)
                elif pair == {"rates","credit"}: groups["RATES_CREDIT"].append(term)
                if (a == "fx") ^ (b == "fx"): groups["FX_CROSS"].append(term)
    total = float(np.mean(terms)) if terms else np.nan
    comp = {g:(float(np.mean(v)) if v else np.nan) for g,v in groups.items()}
    return total, comp

def stationary_bootstrap_ci(diff, reps=999, seed=30030, level=0.95):
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    out = {"n":n,"mean_gain":np.nan,"median_gain":np.nan,"win_rate":np.nan,
           "ci_lo":np.nan,"ci_hi":np.nan,"p_gain_le_0":np.nan,"bootstrap_expected_block":np.nan}
    if n == 0:
        return out
    out.update(mean_gain=float(np.mean(x)), median_gain=float(np.median(x)), win_rate=float(np.mean(x > 0)))
    if n == 1:
        return out
    try:
        br = politis_white_block_length(x[:,None])
        bf = np.asarray(br, dtype=float)
        b = float(np.nanmedian(bf[np.isfinite(bf)])) if np.isfinite(bf).any() else 1.0
    except Exception:
        b = 1.0
    b = float(np.clip(b, 1.0, max(1.0, n/3.0)))
    pr = min(max(1.0/b, 1e-9), 1.0)
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(reps))
    for r in range(int(reps)):
        vals = []
        while len(vals) < n:
            st = int(rng.integers(0,n))
            L = int(rng.geometric(pr))
            take = min(L, n-len(vals))
            vals.extend(x[(st + np.arange(take)) % n].tolist())
        boots[r] = np.mean(vals)
    alpha = 1.0 - float(level)
    out.update(ci_lo=float(np.quantile(boots, alpha/2)),
               ci_hi=float(np.quantile(boots, 1-alpha/2)),
               p_gain_le_0=float((1 + np.sum(boots <= 0)) / (len(boots)+1)),
               bootstrap_expected_block=b)
    return out


def energy_score_mc(draws, y, scale=None, n_pairs=20000, seed=12345):
    """Monte-Carlo Energy Score using independent random scenario pairs.

    Unlike the legacy v30 disjoint-pair estimator, this estimator can use many randomly sampled
    pairs while remaining O(n_pairs), making it suitable for prospective router scoring.
    """
    X = np.asarray(draws, dtype=float)
    yy = np.asarray(y, dtype=float)
    if scale is not None:
        s = np.asarray(scale, dtype=float)
        X = X / s[None, :]
        yy = yy / s
    if len(X) < 2:
        return float(np.linalg.norm(X - yy[None, :], axis=1).mean())
    t1 = float(np.linalg.norm(X - yy[None, :], axis=1).mean())
    rng = np.random.default_rng(int(seed))
    m = int(max(1, n_pairs))
    i = rng.integers(0, len(X), size=m)
    j = rng.integers(0, len(X) - 1, size=m)
    j = j + (j >= i)
    t2 = float(np.linalg.norm(X[i] - X[j], axis=1).mean())
    return t1 - 0.5 * t2


def crps_sample(draws, y):
    """Empirical univariate CRPS in O(N log N). Lower is better."""
    x = np.sort(np.asarray(draws, dtype=float).ravel())
    y = float(y)
    n = len(x)
    if n == 0:
        return np.nan
    first = float(np.mean(np.abs(x - y)))
    coeff = 2.0 * np.arange(1, n + 1) - n - 1.0
    half_pair_expectation = float(np.sum(coeff * x) / (n * n))
    return first - half_pair_expectation


def marginal_crps(draws, y, scale=None):
    """Mean CRPS across coordinates; optionally standardises each coordinate first."""
    X = np.asarray(draws, dtype=float)
    yy = np.asarray(y, dtype=float)
    if X.ndim != 2 or yy.shape[0] != X.shape[1]:
        raise ValueError("draws must be (N,K) and y must have length K")
    if scale is not None:
        s = np.asarray(scale, dtype=float)
        X = X / s[None, :]
        yy = yy / s
    return float(np.mean([crps_sample(X[:, j], yy[j]) for j in range(X.shape[1])]))


def projection_crps(draws, y, projections):
    """Average CRPS over user-supplied linear projections.

    ``projections`` is an array of shape (P,K). Each row can represent an economically relevant
    portfolio, factor direction or other linear functional of the joint state.
    """
    X = np.asarray(draws, dtype=float)
    yy = np.asarray(y, dtype=float)
    A = np.atleast_2d(np.asarray(projections, dtype=float))
    if A.shape[1] != X.shape[1]:
        raise ValueError("projection width must equal scenario dimension")
    return float(np.mean([crps_sample(X @ a, yy @ a) for a in A]))


def quantile_score(draws, y, alpha=0.95):
    """Pinball loss for an upper-tail quantile of a scalar loss distribution."""
    x = np.asarray(draws, dtype=float).ravel()
    q = float(np.quantile(x, float(alpha)))
    u = float(y) - q
    return float((float(alpha) - (u < 0.0)) * u)


def fz0_var_es_score(loss_draws, realized_loss, alpha=0.975):
    """FZ0 joint VaR/ES score for an upper-tail scalar loss distribution.

    The implementation maps losses L to returns R=-L and uses the lower-tail FZ0 form. It requires
    positive upper-tail ES in loss units. Lower scores are better.
    """
    L = np.asarray(loss_draws, dtype=float).ravel()
    a = float(alpha)
    if not (0.5 < a < 1.0):
        raise ValueError("alpha should be an upper-tail level such as 0.95 or 0.975")
    qL = float(np.quantile(L, a))
    tail = L[L >= qL]
    eL = float(tail.mean()) if len(tail) else qL
    if eL <= 0:
        return np.nan
    beta = 1.0 - a
    v, e, yr = -qL, -eL, -float(realized_loss)
    indicator = float(yr <= v)
    return float(-(indicator * (v - yr)) / (beta * e) + v / e + np.log(-e) - 1.0)


def negative_log_likelihood(logpdf, y):
    """Negative log likelihood when a candidate exposes a comparable predictive log-density."""
    lp = float(logpdf(np.asarray(y, dtype=float)))
    return -lp if np.isfinite(lp) else np.inf


def objective_score(draws, y, objective, *, scale=None, labels=None, projections=None,
                    loss_projection=None, alpha=0.975, logpdf=None, seed=12345):
    """Score one scenario law for one clearly defined predictive objective.

    Built-ins:
      joint_distribution -> Energy Score
      dependence        -> Variogram Score
      marginals         -> mean marginal CRPS
      projections       -> projection CRPS
      tail_risk         -> FZ0 VaR/ES score on a supplied scalar loss projection
      density           -> NLL, only when a comparable logpdf is supplied

    A downstream decision system can also create its own objective and append the resulting losses to
    the router history; no downstream-specific logic is required in the scenario engine.
    """
    obj = str(objective)
    if obj == "joint_distribution":
        return energy_score_mc(draws, y, scale=scale, seed=seed)
    if obj == "dependence":
        return variogram_score(draws, y, scale=scale, labels=labels)[0]
    if obj == "marginals":
        return marginal_crps(draws, y, scale=scale)
    if obj == "projections":
        if projections is None:
            raise ValueError("projections objective requires projections")
        return projection_crps(draws, y, projections)
    if obj == "tail_risk":
        if loss_projection is None:
            raise ValueError("tail_risk objective requires loss_projection")
        X = np.asarray(draws, dtype=float)
        yy = np.asarray(y, dtype=float)
        if callable(loss_projection):
            losses = np.asarray(loss_projection(X), dtype=float)
            obs_loss = float(np.asarray(loss_projection(yy[None, :])).ravel()[0])
        else:
            a = np.asarray(loss_projection, dtype=float)
            losses = X @ a
            obs_loss = float(yy @ a)
        return fz0_var_es_score(losses, obs_loss, alpha=alpha)
    if obj == "density":
        if logpdf is None:
            raise ValueError("density objective requires a comparable predictive logpdf")
        return negative_log_likelihood(logpdf, y)
    raise ValueError(f"Unknown built-in objective: {objective}")
