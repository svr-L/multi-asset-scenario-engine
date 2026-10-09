from __future__ import annotations

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import chi2
from scipy.stats import t as student_t
from scipy.stats import norm
from scipy.special import gammaln
from pandas.tseries.offsets import BDay


def plot_time_series(df, title, ylabel=None):
    ax = df.plot(figsize=(11, 6), lw=1.2)
    ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_histogram(series, title, xlabel):
    ax = pd.Series(series).plot(kind="hist", bins=60, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    plt.show()


def distribution_summary(series):
    s = pd.Series(series, dtype=float).dropna()
    return pd.Series({
        "mean": s.mean(),
        "std": s.std(),
        "q01": s.quantile(0.01),
        "q05": s.quantile(0.05),
        "median": s.median(),
        "q95": s.quantile(0.95),
        "q99": s.quantile(0.99),
        "min": s.min(),
        "max": s.max(),
    })


def loss_summary_from_pnl(series, alpha=0.05):
    s = pd.Series(series, dtype=float).dropna()
    threshold = s.quantile(alpha)
    tail = s[s <= threshold]
    return pd.Series({
        "mean": s.mean(),
        "std": s.std(),
        "q01": s.quantile(0.01),
        "q05": s.quantile(0.05),
        "median": s.median(),
        "q95": s.quantile(0.95),
        "q99": s.quantile(0.99),
        "VaR_95_loss": -threshold,
        "ES_95_loss": -tail.mean() if len(tail) > 0 else np.nan,
    })


