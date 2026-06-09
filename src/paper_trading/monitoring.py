"""Monitoring dashboards for paper-trading metrics and cost decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.metrics import profit_factor_from_returns, sharpe_ratio

REQUIRED_RETURN_COLUMNS = {
    "net_return",
    "gross_return",
    "total_costs",
    "fees",
    "slippage",
    "funding",
    "interest",
    "other_costs",
}

MONITORING_COLUMNS = [
    "window_start",
    "window_end",
    "frequency",
    "num_bars",
    "num_trades",
    "active_bars",
    "active_bar_ratio",
    "activity_level",
    "dominant_regime",
    "profit_factor_net",
    "sharpe_ratio",
    "max_drawdown",
    "turnover",
    "net_return",
    "gross_return",
    "cost_to_gross",
    "fees",
    "slippage",
    "funding",
    "interest",
    "other_costs",
    "total_costs",
]


def _activity_level(*, active_bars: int, num_bars: int, num_trades: int) -> str:
    if active_bars <= 0 and num_trades <= 0:
        return "inactive"
    if num_bars <= 0:
        return "low"

    ratio = float(active_bars) / float(num_bars)
    if ratio < 0.20 and num_trades <= 1:
        return "low"
    if ratio < 0.60:
        return "medium"
    return "high"


def _dominant_regime(window: pd.DataFrame) -> str:
    if "regime" not in window.columns:
        return "unknown"
    regimes = window["regime"].dropna().astype(str)
    if regimes.empty:
        return "unknown"
    # mode() can return multiple rows when tied; use first for determinism.
    return str(regimes.mode().iloc[0])


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1 + returns.fillna(0.0)).cumprod()
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def summarize_returns_window(window: pd.DataFrame) -> dict[str, float]:
    """Summarize one monitoring window from paper-trading returns."""
    net = window["net_return"].dropna().astype(float)
    gross = window["gross_return"].dropna().astype(float)

    gross_sum = float(gross.sum()) if not gross.empty else 0.0
    cost_sum = float(window["total_costs"].sum())

    activity_mask = pd.Series(False, index=window.index)

    if "traded_notional" in window.columns:
        traded_notional = window["traded_notional"].fillna(0.0).astype(float)
        turnover = float(traded_notional.sum())
        num_trades = int((traded_notional > 0.01).sum())
        activity_mask = activity_mask | (traded_notional > 0.0)
    elif "turnover" in window.columns:
        turnover_col = window["turnover"].fillna(0.0).astype(float)
        turnover = float(turnover_col.sum())
        num_trades = int((turnover_col > 0.01).sum())
        activity_mask = activity_mask | (turnover_col > 0.0)
    else:
        turnover = 0.0
        num_trades = 0

    if "position" in window.columns:
        position = window["position"].fillna(0.0).astype(float)
        activity_mask = activity_mask | (position.abs() > 1e-9)
    else:
        # Fallback for legacy frames without position column.
        activity_mask = activity_mask | (window["net_return"].fillna(0.0).abs() > 1e-12)

    active_bars = int(activity_mask.sum())
    active_bar_ratio = float(active_bars / len(window)) if len(window) > 0 else 0.0
    activity_level = _activity_level(
        active_bars=active_bars,
        num_bars=int(len(window)),
        num_trades=num_trades,
    )

    return {
        "num_bars": int(len(window)),
        "num_trades": num_trades,
        "active_bars": active_bars,
        "active_bar_ratio": active_bar_ratio,
        "activity_level": activity_level,
        "dominant_regime": _dominant_regime(window),
        "profit_factor_net": float(profit_factor_from_returns(net)) if not net.empty else 0.0,
        "sharpe_ratio": float(sharpe_ratio(net)) if not net.empty else 0.0,
        "max_drawdown": _max_drawdown_from_returns(net),
        "turnover": turnover,
        "net_return": float((1 + net).prod() - 1) if not net.empty else 0.0,
        "gross_return": float((1 + gross).prod() - 1) if not gross.empty else 0.0,
        "cost_to_gross": (cost_sum / abs(gross_sum)) if abs(gross_sum) > 1e-12 else np.nan,
        "fees": float(window["fees"].sum()),
        "slippage": float(window["slippage"].sum()),
        "funding": float(window["funding"].sum()),
        "interest": float(window["interest"].sum()),
        "other_costs": float(window["other_costs"].sum()),
        "total_costs": cost_sum,
    }


def build_monitoring_dashboard(
    returns: pd.DataFrame,
    *,
    frequency: str,
) -> pd.DataFrame:
    """Build grouped monitoring dashboard at daily/weekly frequency."""
    missing = REQUIRED_RETURN_COLUMNS - set(returns.columns)
    if missing:
        raise ValueError(f"Returns dataframe missing monitoring columns: {sorted(missing)}")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns index must be DatetimeIndex")

    rows: list[dict[str, object]] = []
    grouped = returns.groupby(pd.Grouper(freq=frequency, label="right", closed="right"))

    for _, window in grouped:
        if window.empty:
            continue

        metrics = summarize_returns_window(window)
        metrics["window_start"] = window.index[0].isoformat()
        metrics["window_end"] = window.index[-1].isoformat()
        metrics["frequency"] = frequency
        rows.append(metrics)

    if not rows:
        return pd.DataFrame(columns=MONITORING_COLUMNS)

    dashboard = pd.DataFrame(rows)
    dashboard["window_end_ts"] = pd.to_datetime(dashboard["window_end"], utc=True)
    dashboard = dashboard.sort_values("window_end_ts").drop(columns=["window_end_ts"])
    return dashboard.reset_index(drop=True)


def build_daily_weekly_dashboards(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build both daily and weekly dashboards."""
    return {
        "daily": build_monitoring_dashboard(returns, frequency="D"),
        "weekly": build_monitoring_dashboard(returns, frequency="W"),
    }
