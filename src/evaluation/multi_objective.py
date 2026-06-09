"""Phase 11 multi-objective sweep scoring and frontier utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.common.metrics import DEFAULT_SUCCESS_THRESHOLDS


@dataclass(frozen=True)
class CompositeScoreWeights:
    """Weights for multi-objective ranking score."""

    profit_factor: float = 1.0
    sharpe: float = 1.0
    trades_log: float = 0.25
    kill_event_rate: float = 1.0
    turnover_log: float = 0.05
    drawdown_abs: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "profit_factor": float(self.profit_factor),
            "sharpe": float(self.sharpe),
            "trades_log": float(self.trades_log),
            "kill_event_rate": float(self.kill_event_rate),
            "turnover_log": float(self.turnover_log),
            "drawdown_abs": float(self.drawdown_abs),
        }


@dataclass(frozen=True)
class AcceptanceConstraints:
    """Deployment acceptance constraints for sweep candidates."""

    min_profit_factor_net: float = DEFAULT_SUCCESS_THRESHOLDS.min_profit_factor_net
    min_sharpe_net: float = DEFAULT_SUCCESS_THRESHOLDS.min_sharpe_net
    min_trades: int = 80
    max_kill_event_rate: float = 0.20
    max_drawdown_abs: float = DEFAULT_SUCCESS_THRESHOLDS.max_drawdown_abs

    def to_dict(self) -> dict[str, float | int]:
        return {
            "min_profit_factor_net": float(self.min_profit_factor_net),
            "min_sharpe_net": float(self.min_sharpe_net),
            "min_trades": int(self.min_trades),
            "max_kill_event_rate": float(self.max_kill_event_rate),
            "max_drawdown_abs": float(self.max_drawdown_abs),
        }


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.fillna(default).astype(float)


def score_candidates(
    candidates: pd.DataFrame,
    *,
    weights: CompositeScoreWeights | None = None,
) -> pd.DataFrame:
    """
    Add composite multi-objective score columns to candidate dataframe.

    Required columns:
    - profit_factor_net
    - sharpe_ratio
    - num_trades
    - kill_event_rate
    - turnover
    - max_drawdown
    """
    if candidates.empty:
        return candidates.copy()

    weights = weights or CompositeScoreWeights()
    df = candidates.copy()

    pf = _safe_numeric(df.get("profit_factor_net", 0.0), default=0.0)
    sharpe = _safe_numeric(df.get("sharpe_ratio", 0.0), default=0.0)
    trades = _safe_numeric(df.get("num_trades", 0.0), default=0.0)
    kill_rate = _safe_numeric(df.get("kill_event_rate", 0.0), default=0.0)
    turnover = _safe_numeric(df.get("turnover", 0.0), default=0.0)
    drawdown_abs = _safe_numeric(df.get("max_drawdown", 0.0), default=0.0).abs()

    trade_term = np.log1p(np.clip(trades, 0.0, None))
    turnover_term = np.log1p(np.clip(turnover, 0.0, None))
    pf_term = np.clip(pf, -5.0, 10.0)
    sharpe_term = np.clip(sharpe, -10.0, 10.0)
    kill_term = np.clip(kill_rate, 0.0, 1.0)
    dd_term = np.clip(drawdown_abs, 0.0, 1.0)

    score = (
        weights.profit_factor * pf_term
        + weights.sharpe * sharpe_term
        + weights.trades_log * trade_term
        - weights.kill_event_rate * kill_term
        - weights.turnover_log * turnover_term
        - weights.drawdown_abs * dd_term
    )

    df["score_profit_factor_term"] = pf_term
    df["score_sharpe_term"] = sharpe_term
    df["score_trade_term"] = trade_term
    df["score_kill_term"] = kill_term
    df["score_turnover_term"] = turnover_term
    df["score_drawdown_term"] = dd_term
    df["composite_score"] = score.astype(float)
    return df


def apply_acceptance_constraints(
    candidates: pd.DataFrame,
    *,
    constraints: AcceptanceConstraints | None = None,
) -> pd.DataFrame:
    """Apply deployment acceptance constraints and annotate failures."""
    if candidates.empty:
        return candidates.copy()

    constraints = constraints or AcceptanceConstraints()
    df = candidates.copy()

    pf = _safe_numeric(df.get("profit_factor_net", 0.0), default=0.0)
    sharpe = _safe_numeric(df.get("sharpe_ratio", 0.0), default=0.0)
    trades = _safe_numeric(df.get("num_trades", 0.0), default=0.0)
    kill_rate = _safe_numeric(df.get("kill_event_rate", 0.0), default=0.0)
    max_dd_abs = _safe_numeric(df.get("max_drawdown", 0.0), default=0.0).abs()
    active = trades >= 1.0

    reasons: list[list[str]] = []
    passes: list[bool] = []
    for i in range(len(df)):
        failed: list[str] = []
        if pf.iloc[i] < constraints.min_profit_factor_net:
            failed.append("profit_factor_net")
        if sharpe.iloc[i] < constraints.min_sharpe_net:
            failed.append("sharpe_ratio")
        if trades.iloc[i] < constraints.min_trades:
            failed.append("num_trades")
        if kill_rate.iloc[i] > constraints.max_kill_event_rate:
            failed.append("kill_event_rate")
        if max_dd_abs.iloc[i] > constraints.max_drawdown_abs:
            failed.append("max_drawdown_abs")
        passes.append(len(failed) == 0)
        reasons.append(failed)

    df["acceptance_passed"] = passes
    df["acceptance_failed_criteria"] = [";".join(items) for items in reasons]
    df["active_candidate"] = active.astype(bool)
    return df


def pareto_frontier(
    candidates: pd.DataFrame,
    *,
    maximize: tuple[str, ...] = ("profit_factor_net", "sharpe_ratio", "num_trades"),
    minimize: tuple[str, ...] = ("kill_event_rate", "max_drawdown_abs", "turnover"),
) -> pd.DataFrame:
    """
    Return non-dominated candidates on the Pareto frontier.

    A dominates B if it is at least as good in all objectives and strictly
    better in at least one objective.
    """
    if candidates.empty:
        return candidates.copy()

    df = candidates.copy().reset_index(drop=True)
    if "max_drawdown_abs" not in df.columns:
        df["max_drawdown_abs"] = _safe_numeric(df.get("max_drawdown", 0.0), default=0.0).abs()

    for col in maximize:
        if col not in df.columns:
            df[col] = 0.0
    for col in minimize:
        if col not in df.columns:
            df[col] = 0.0

    n = len(df)
    dominated = np.zeros(n, dtype=bool)

    max_values = {col: _safe_numeric(df[col], default=0.0).to_numpy(dtype=float) for col in maximize}
    min_values = {col: _safe_numeric(df[col], default=0.0).to_numpy(dtype=float) for col in minimize}

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            ge_all = all(max_values[col][j] >= max_values[col][i] for col in maximize) and all(
                min_values[col][j] <= min_values[col][i] for col in minimize
            )
            gt_any = any(max_values[col][j] > max_values[col][i] for col in maximize) or any(
                min_values[col][j] < min_values[col][i] for col in minimize
            )
            if ge_all and gt_any:
                dominated[i] = True
                break

    frontier = df.loc[~dominated].copy()
    if "acceptance_passed" in frontier.columns:
        sort_columns = ["acceptance_passed"]
        ascending = [False]
        if "active_candidate" in frontier.columns:
            sort_columns.append("active_candidate")
            ascending.append(False)
        sort_columns.extend(["composite_score", "profit_factor_net", "sharpe_ratio"])
        ascending.extend([False, False, False])
        frontier = frontier.sort_values(
            by=sort_columns,
            ascending=ascending,
        )
    elif "composite_score" in frontier.columns:
        frontier = frontier.sort_values(
            by=["composite_score", "profit_factor_net", "sharpe_ratio"],
            ascending=[False, False, False],
        )
    return frontier.reset_index(drop=True)


def rank_candidates(
    candidates: pd.DataFrame,
    *,
    weights: CompositeScoreWeights | None = None,
    constraints: AcceptanceConstraints | None = None,
) -> pd.DataFrame:
    """Apply scoring + acceptance and return ranked dataframe."""
    if candidates.empty:
        return candidates.copy()

    scored = score_candidates(candidates, weights=weights)
    constrained = apply_acceptance_constraints(scored, constraints=constraints)
    sort_columns = ["acceptance_passed"]
    ascending = [False]
    if "active_candidate" in constrained.columns:
        sort_columns.append("active_candidate")
        ascending.append(False)
    sort_columns.extend(["composite_score", "profit_factor_net", "sharpe_ratio"])
    ascending.extend([False, False, False])
    ranked = constrained.sort_values(by=sort_columns, ascending=ascending)
    return ranked.reset_index(drop=True)


def schema_payload() -> dict[str, Any]:
    """Schema payload for sweep artifact fields and semantics."""
    return {
        "schema_version": "phase11_sweep_v1",
        "fields": {
            "candidate_id": "integer unique candidate identifier",
            "profit_factor_net": "net PF metric",
            "sharpe_ratio": "net annualized Sharpe",
            "num_trades": "trade count over replay window",
            "kill_events": "count of kill events (daily + weekly)",
            "kill_event_rate": "kill_events / monitoring_windows",
            "turnover": "aggregate turnover in replay window",
            "max_drawdown": "max drawdown (negative)",
            "max_drawdown_abs": "absolute drawdown",
            "composite_score": "multi-objective score (higher better)",
            "acceptance_passed": "boolean deployment-constraint pass flag",
            "acceptance_failed_criteria": "semicolon-separated failed criteria names",
            "active_candidate": "boolean flag for candidates with at least one executed trade",
            "pareto_frontier": "boolean non-dominated candidate flag",
        },
        "objectives": {
            "maximize": ["profit_factor_net", "sharpe_ratio", "num_trades"],
            "minimize": ["kill_event_rate", "max_drawdown_abs", "turnover"],
        },
    }
