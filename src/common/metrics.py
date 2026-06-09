"""Shared metric names, thresholds, and metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class MetricName(str, Enum):
    """Canonical metric names used across evaluation and robustness."""

    SHARPE_NET = "sharpe_net"
    PROFIT_FACTOR_NET = "profit_factor_net"
    BASELINE_SHARPE_DELTA = "baseline_sharpe_delta"
    REGIME_SHARPE_CV = "regime_sharpe_cv"
    RECENT_SHARPE_RATIO = "recent_sharpe_ratio"
    MAX_DRAWDOWN_ABS = "max_drawdown_abs"
    WIN_RATE = "win_rate"


@dataclass(frozen=True)
class SuccessThresholds:
    """Primary and secondary net-performance thresholds."""

    min_profit_factor_net: float = 1.0
    min_sharpe_net: float = 0.5
    min_baseline_sharpe_delta: float = 0.1
    max_regime_sharpe_cv: float = 1.5
    min_recent_sharpe_ratio: float = 0.7
    min_robustness_pass_rate: float = 0.7
    max_drawdown_abs: float = 0.30
    min_win_rate: float = 0.45


DEFAULT_SUCCESS_THRESHOLDS = SuccessThresholds()


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division helper."""
    if denominator == 0:
        return default
    return numerator / denominator


def profit_factor_from_returns(returns: pd.Series) -> float:
    """Compute net profit factor from period returns."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0

    wins = clean[clean > 0].sum()
    losses = abs(clean[clean < 0].sum())
    if losses == 0:
        # Infinite PF if there are wins and no losses.
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 6 * 365) -> float:
    """Annualized Sharpe ratio."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    std = clean.std()
    if std == 0:
        return 0.0
    return clean.mean() / std * np.sqrt(periods_per_year)


def recent_vs_early_sharpe_ratio(
    returns: pd.Series,
    split_fraction: float = 0.25,
    periods_per_year: int = 6 * 365,
) -> float:
    """
    Ratio of recent Sharpe to early Sharpe.

    Returns 1.0 when comparison is not meaningful (insufficient samples).
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 40:
        return 1.0

    chunk = max(10, int(n * split_fraction))
    early = clean.iloc[:chunk]
    recent = clean.iloc[-chunk:]

    early_sharpe = sharpe_ratio(early, periods_per_year=periods_per_year)
    recent_sharpe = sharpe_ratio(recent, periods_per_year=periods_per_year)

    if early_sharpe <= 0:
        return 1.0
    return recent_sharpe / early_sharpe
