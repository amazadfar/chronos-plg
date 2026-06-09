"""Paper-trading kill-switch, readiness, and capital ramp policy definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.common.metrics import DEFAULT_SUCCESS_THRESHOLDS
from src.paper_trading.monitoring import summarize_returns_window

SOFT_KILL_REASONS = frozenset({
    "profit_factor_net_breach",
    "sharpe_ratio_breach",
})
HARD_KILL_REASONS = frozenset({
    "max_drawdown_breach",
    "cost_to_gross_breach",
    "turnover_breach",
})


@dataclass(frozen=True)
class KillSwitchThresholds:
    """Thresholds for automatic paper-trading kill switch."""

    min_profit_factor_net: float = DEFAULT_SUCCESS_THRESHOLDS.min_profit_factor_net
    min_sharpe_net: float = DEFAULT_SUCCESS_THRESHOLDS.min_sharpe_net
    max_drawdown_abs: float = DEFAULT_SUCCESS_THRESHOLDS.max_drawdown_abs
    max_cost_to_gross: float = 0.70
    max_turnover: float = 35.0
    min_active_bars: int = 1
    min_trades_for_pf_sharpe: int = 2
    min_active_bars_for_soft: int = 2
    min_bars_for_soft: int = 1
    min_soft_window_turnover: float = 0.0
    min_soft_window_abs_net_return: float = 0.0
    min_bars_for_cost_to_gross: int = 1
    min_abs_gross_return_for_cost_to_gross: float = 0.0
    min_windows_before_enforcement: int = 2


@dataclass
class KillSwitchEvent:
    """A triggered kill-switch event."""

    window_end: str
    frequency: str
    reasons: list[str]
    action: str
    metrics: dict[str, float]
    trigger_type: str = "unknown"
    dominant_regime: str = "unknown"
    activity_level: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_end": self.window_end,
            "frequency": self.frequency,
            "reasons": self.reasons,
            "action": self.action,
            "metrics": self.metrics,
            "trigger_type": self.trigger_type,
            "dominant_regime": self.dominant_regime,
            "activity_level": self.activity_level,
        }


@dataclass(frozen=True)
class DeploymentReadinessPolicy:
    """Minimum criteria before any real-capital deployment."""

    min_observation_days: int = 42
    min_trades: int = 80
    min_profit_factor_net: float = DEFAULT_SUCCESS_THRESHOLDS.min_profit_factor_net
    min_sharpe_net: float = DEFAULT_SUCCESS_THRESHOLDS.min_sharpe_net
    max_drawdown_abs: float = DEFAULT_SUCCESS_THRESHOLDS.max_drawdown_abs
    allow_kill_events: bool = False


@dataclass
class DeploymentReadiness:
    """Readiness status for transitioning from paper to live capital."""

    ready: bool
    reason: str
    observed_days: int
    total_trades: int
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "observed_days": self.observed_days,
            "total_trades": self.total_trades,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class RampStage:
    """One stage in staged capital ramp deployment."""

    name: str
    capital_fraction: float
    min_days_in_stage: int
    min_profit_factor_net: float
    min_sharpe_net: float
    max_drawdown_abs: float


@dataclass(frozen=True)
class CapitalRampPolicy:
    """Definition for staged capital ramp and rollback behavior."""

    stages: list[RampStage]
    rollback_on_any_kill: bool = True
    rollback_to_paper: bool = True
    hard_drawdown_abs: float = DEFAULT_SUCCESS_THRESHOLDS.max_drawdown_abs

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollback_on_any_kill": self.rollback_on_any_kill,
            "rollback_to_paper": self.rollback_to_paper,
            "hard_drawdown_abs": self.hard_drawdown_abs,
            "stages": [
                {
                    "name": stage.name,
                    "capital_fraction": stage.capital_fraction,
                    "min_days_in_stage": stage.min_days_in_stage,
                    "min_profit_factor_net": stage.min_profit_factor_net,
                    "min_sharpe_net": stage.min_sharpe_net,
                    "max_drawdown_abs": stage.max_drawdown_abs,
                }
                for stage in self.stages
            ],
        }


@dataclass
class CapitalRampDecision:
    """Recommended ramp action for the current monitoring checkpoint."""

    current_stage: str
    recommended_stage: str
    action: str
    reason: str
    recommended_capital_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "recommended_stage": self.recommended_stage,
            "action": self.action,
            "reason": self.reason,
            "recommended_capital_fraction": self.recommended_capital_fraction,
        }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _row_has_activity(row: pd.Series, *, min_active_bars: int = 1) -> bool:
    """Determine whether a monitoring row has meaningful trading/exposure activity."""
    min_active = max(1, int(min_active_bars))

    active_bars = int(_to_float(row.get("active_bars"), default=-1.0))
    if active_bars >= 0:
        return active_bars >= min_active

    # Backward-compatible fallback for dashboards that predate "active_bars".
    num_trades = int(_to_float(row.get("num_trades"), default=0.0))
    turnover = abs(_to_float(row.get("turnover"), default=0.0))
    net_return = abs(_to_float(row.get("net_return"), default=0.0))
    gross_return = abs(_to_float(row.get("gross_return"), default=0.0))
    max_drawdown = abs(_to_float(row.get("max_drawdown"), default=0.0))
    cost_to_gross = row.get("cost_to_gross")

    return (
        num_trades > 0
        or turnover > 0.0
        or net_return > 0.0
        or gross_return > 0.0
        or max_drawdown > 0.0
        or pd.notna(cost_to_gross)
    )


def _activity_level_from_row(row: pd.Series) -> str:
    explicit = row.get("activity_level")
    if explicit is not None and not pd.isna(explicit):
        return str(explicit)

    active_bars = int(max(0.0, _to_float(row.get("active_bars"), default=0.0)))
    num_bars = int(max(0.0, _to_float(row.get("num_bars"), default=0.0)))
    num_trades = int(max(0.0, _to_float(row.get("num_trades"), default=0.0)))
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


def _soft_criteria_eligible(
    *,
    row: pd.Series,
    thresholds: KillSwitchThresholds,
    num_trades: int | None,
) -> bool:
    num_bars = int(max(0.0, _to_float(row.get("num_bars"), default=0.0)))
    if num_bars > 0 and num_bars < max(1, thresholds.min_bars_for_soft):
        return False

    if num_trades is not None and num_trades < max(1, thresholds.min_trades_for_pf_sharpe):
        return False

    active_bars = int(max(0.0, _to_float(row.get("active_bars"), default=0.0)))
    if active_bars > 0 and active_bars < max(1, thresholds.min_active_bars_for_soft):
        return False

    turnover = abs(_to_float(row.get("turnover"), default=0.0))
    abs_net_return = abs(_to_float(row.get("net_return"), default=0.0))
    turnover_ok = turnover >= max(0.0, thresholds.min_soft_window_turnover)
    return_ok = abs_net_return >= max(0.0, thresholds.min_soft_window_abs_net_return)

    # Keep backward compatibility for legacy dashboards that may not include
    # turnover/net_return while allowing stricter thresholds when available.
    if "turnover" not in row.index and "net_return" not in row.index:
        return True

    return turnover_ok or return_ok


def _cost_to_gross_eligible(
    *,
    row: pd.Series,
    thresholds: KillSwitchThresholds,
) -> bool:
    num_bars = int(max(0.0, _to_float(row.get("num_bars"), default=0.0)))
    if num_bars > 0 and num_bars < max(1, thresholds.min_bars_for_cost_to_gross):
        return False

    abs_gross_return = abs(_to_float(row.get("gross_return"), default=0.0))
    return abs_gross_return >= max(0.0, thresholds.min_abs_gross_return_for_cost_to_gross)


def _classify_trigger_type(reasons: list[str]) -> str:
    reason_set = set(reasons)
    has_soft = any(reason in SOFT_KILL_REASONS for reason in reason_set)
    has_hard = any(reason in HARD_KILL_REASONS for reason in reason_set)
    if has_soft and has_hard:
        return "mixed"
    if has_hard:
        return "hard"
    if has_soft:
        return "soft"
    return "unknown"


def evaluate_kill_switch(
    dashboard: pd.DataFrame,
    thresholds: KillSwitchThresholds | None = None,
) -> tuple[pd.DataFrame, list[KillSwitchEvent]]:
    """Evaluate automatic kill-switch triggers for each monitoring window."""
    thresholds = thresholds or KillSwitchThresholds()

    flagged = dashboard.copy()
    if flagged.empty:
        flagged["kill_switch_triggered"] = pd.Series(dtype=int)
        flagged["kill_switch_reasons"] = pd.Series(dtype=str)
        flagged["kill_switch_trigger_type"] = pd.Series(dtype=str)
        flagged["soft_criteria_eligible"] = pd.Series(dtype=int)
        flagged["activity_level"] = pd.Series(dtype=str)
        return flagged, []

    flagged["kill_switch_triggered"] = 0
    flagged["kill_switch_reasons"] = ""
    flagged["kill_switch_trigger_type"] = "none"
    flagged["soft_criteria_eligible"] = 0
    if "activity_level" not in flagged.columns:
        flagged["activity_level"] = "unknown"

    events: list[KillSwitchEvent] = []
    active_windows_seen = 0

    for idx, row in flagged.iterrows():
        activity_level = _activity_level_from_row(row)
        flagged.at[idx, "activity_level"] = activity_level

        is_active = _row_has_activity(row, min_active_bars=thresholds.min_active_bars)
        if not is_active:
            continue

        active_windows_seen += 1
        if active_windows_seen < thresholds.min_windows_before_enforcement:
            continue

        reasons: list[str] = []

        pf_net = _to_float(row.get("profit_factor_net"), default=0.0)
        sharpe = _to_float(row.get("sharpe_ratio"), default=0.0)
        max_dd = abs(_to_float(row.get("max_drawdown"), default=0.0))
        cost_to_gross = row.get("cost_to_gross")
        turnover = _to_float(row.get("turnover"), default=0.0)
        raw_num_trades = row.get("num_trades")
        has_num_trades = raw_num_trades is not None and not pd.isna(raw_num_trades)
        num_trades = int(_to_float(raw_num_trades, default=0.0)) if has_num_trades else None

        soft_eligible = _soft_criteria_eligible(
            row=row,
            thresholds=thresholds,
            num_trades=num_trades,
        )
        flagged.at[idx, "soft_criteria_eligible"] = 1 if soft_eligible else 0

        if soft_eligible:
            if pf_net <= thresholds.min_profit_factor_net:
                reasons.append("profit_factor_net_breach")
            if sharpe < thresholds.min_sharpe_net:
                reasons.append("sharpe_ratio_breach")

        if max_dd > thresholds.max_drawdown_abs:
            reasons.append("max_drawdown_breach")
        if (
            pd.notna(cost_to_gross)
            and _cost_to_gross_eligible(row=row, thresholds=thresholds)
            and float(cost_to_gross) > thresholds.max_cost_to_gross
        ):
            reasons.append("cost_to_gross_breach")
        if turnover > thresholds.max_turnover:
            reasons.append("turnover_breach")

        if not reasons:
            continue

        trigger_type = _classify_trigger_type(reasons)
        flagged.at[idx, "kill_switch_triggered"] = 1
        flagged.at[idx, "kill_switch_reasons"] = ";".join(reasons)
        flagged.at[idx, "kill_switch_trigger_type"] = trigger_type

        events.append(
            KillSwitchEvent(
                window_end=str(row.get("window_end", "")),
                frequency=str(row.get("frequency", "")),
                reasons=reasons,
                action="HALT",
                metrics={
                    "profit_factor_net": pf_net,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd,
                    "cost_to_gross": _to_float(cost_to_gross, default=float("nan")),
                    "turnover": turnover,
                    "num_trades": float(num_trades) if num_trades is not None else float("nan"),
                    "active_bars": _to_float(row.get("active_bars"), default=float("nan")),
                    "active_bar_ratio": _to_float(row.get("active_bar_ratio"), default=float("nan")),
                    "net_return": _to_float(row.get("net_return"), default=float("nan")),
                    "gross_return": _to_float(row.get("gross_return"), default=float("nan")),
                    "total_costs": _to_float(row.get("total_costs"), default=float("nan")),
                    "fees": _to_float(row.get("fees"), default=float("nan")),
                    "slippage": _to_float(row.get("slippage"), default=float("nan")),
                    "funding": _to_float(row.get("funding"), default=float("nan")),
                    "interest": _to_float(row.get("interest"), default=float("nan")),
                    "other_costs": _to_float(row.get("other_costs"), default=float("nan")),
                },
                trigger_type=trigger_type,
                dominant_regime=str(row.get("dominant_regime", "unknown")),
                activity_level=activity_level,
            )
        )

    return flagged, events


def evaluate_deployment_readiness(
    returns: pd.DataFrame,
    weekly_dashboard: pd.DataFrame,
    kill_events: list[KillSwitchEvent],
    policy: DeploymentReadinessPolicy | None = None,
) -> DeploymentReadiness:
    """Evaluate if paper-trading evidence is sufficient for real capital deployment."""
    policy = policy or DeploymentReadinessPolicy()

    if returns.empty:
        return DeploymentReadiness(
            ready=False,
            reason="no_paper_trading_returns",
            observed_days=0,
            total_trades=0,
            metrics={},
        )

    observed_days = int((returns.index.max() - returns.index.min()).days + 1)
    total_trades = int((returns["traded_notional"] > 0.01).sum()) if "traded_notional" in returns else 0

    overall = summarize_returns_window(returns)

    violations: list[str] = []
    if observed_days < policy.min_observation_days:
        violations.append("observation_window_too_short")
    if total_trades < policy.min_trades:
        violations.append("insufficient_trades")
    if overall["profit_factor_net"] <= policy.min_profit_factor_net:
        violations.append("profit_factor_below_threshold")
    if overall["sharpe_ratio"] < policy.min_sharpe_net:
        violations.append("sharpe_below_threshold")
    if abs(overall["max_drawdown"]) > policy.max_drawdown_abs:
        violations.append("drawdown_above_threshold")
    if kill_events and not policy.allow_kill_events:
        violations.append("kill_switch_triggered")

    if not weekly_dashboard.empty:
        active_weekly = weekly_dashboard[
            weekly_dashboard.apply(_row_has_activity, axis=1)
        ]
        recent = active_weekly.iloc[-1] if not active_weekly.empty else None
        if recent is not None and _to_float(recent.get("profit_factor_net"), 0.0) <= policy.min_profit_factor_net:
            violations.append("recent_week_pf_below_threshold")

    ready = len(violations) == 0
    reason = "ready_for_staged_live_deployment" if ready else ";".join(sorted(set(violations)))

    return DeploymentReadiness(
        ready=ready,
        reason=reason,
        observed_days=observed_days,
        total_trades=total_trades,
        metrics={
            "profit_factor_net": float(overall["profit_factor_net"]),
            "sharpe_ratio": float(overall["sharpe_ratio"]),
            "max_drawdown": float(overall["max_drawdown"]),
            "net_return": float(overall["net_return"]),
            "turnover": float(overall["turnover"]),
        },
    )


def default_capital_ramp_policy() -> CapitalRampPolicy:
    """Default staged capital ramp and rollback policy."""
    stages = [
        RampStage(
            name="paper",
            capital_fraction=0.00,
            min_days_in_stage=0,
            min_profit_factor_net=0.00,
            min_sharpe_net=-9.99,
            max_drawdown_abs=1.00,
        ),
        RampStage(
            name="stage_1_10pct",
            capital_fraction=0.10,
            min_days_in_stage=14,
            min_profit_factor_net=1.00,
            min_sharpe_net=0.40,
            max_drawdown_abs=0.25,
        ),
        RampStage(
            name="stage_2_25pct",
            capital_fraction=0.25,
            min_days_in_stage=14,
            min_profit_factor_net=1.05,
            min_sharpe_net=0.50,
            max_drawdown_abs=0.22,
        ),
        RampStage(
            name="stage_3_50pct",
            capital_fraction=0.50,
            min_days_in_stage=21,
            min_profit_factor_net=1.08,
            min_sharpe_net=0.55,
            max_drawdown_abs=0.20,
        ),
        RampStage(
            name="stage_4_100pct",
            capital_fraction=1.00,
            min_days_in_stage=28,
            min_profit_factor_net=1.10,
            min_sharpe_net=0.60,
            max_drawdown_abs=0.18,
        ),
    ]
    return CapitalRampPolicy(stages=stages)


def _stage_map(policy: CapitalRampPolicy) -> dict[str, RampStage]:
    return {stage.name: stage for stage in policy.stages}


def _meets_stage_thresholds(metrics: dict[str, float], stage: RampStage) -> bool:
    pf_net = _to_float(metrics.get("profit_factor_net"), default=0.0)
    sharpe = _to_float(metrics.get("sharpe_ratio"), default=0.0)
    max_dd = abs(_to_float(metrics.get("max_drawdown"), default=0.0))
    return (
        pf_net >= stage.min_profit_factor_net
        and sharpe >= stage.min_sharpe_net
        and max_dd <= stage.max_drawdown_abs
    )


def recommend_capital_action(
    *,
    current_stage: str,
    days_in_stage: int,
    weekly_dashboard: pd.DataFrame,
    readiness: DeploymentReadiness,
    kill_events: list[KillSwitchEvent],
    policy: CapitalRampPolicy | None = None,
) -> CapitalRampDecision:
    """Recommend HOLD/PROMOTE/ROLLBACK under staged capital ramp policy."""
    policy = policy or default_capital_ramp_policy()
    stage_map = _stage_map(policy)
    stage_names = [stage.name for stage in policy.stages]

    if current_stage not in stage_map:
        current_stage = policy.stages[0].name

    current_idx = stage_names.index(current_stage)
    current = stage_map[current_stage]

    latest_metrics = (
        weekly_dashboard.iloc[-1].to_dict() if not weekly_dashboard.empty else dict(readiness.metrics)
    )

    max_dd = abs(_to_float(latest_metrics.get("max_drawdown"), default=0.0))

    if max_dd > policy.hard_drawdown_abs:
        target = policy.stages[0] if policy.rollback_to_paper else policy.stages[max(0, current_idx - 1)]
        return CapitalRampDecision(
            current_stage=current_stage,
            recommended_stage=target.name,
            action="ROLLBACK",
            reason="hard_drawdown_breach",
            recommended_capital_fraction=target.capital_fraction,
        )

    if kill_events and policy.rollback_on_any_kill:
        target = policy.stages[0] if policy.rollback_to_paper else policy.stages[max(0, current_idx - 1)]
        return CapitalRampDecision(
            current_stage=current_stage,
            recommended_stage=target.name,
            action="ROLLBACK",
            reason="kill_switch_triggered",
            recommended_capital_fraction=target.capital_fraction,
        )

    if current_idx >= len(policy.stages) - 1:
        return CapitalRampDecision(
            current_stage=current_stage,
            recommended_stage=current_stage,
            action="HOLD",
            reason="already_at_max_stage",
            recommended_capital_fraction=current.capital_fraction,
        )

    if not readiness.ready:
        return CapitalRampDecision(
            current_stage=current_stage,
            recommended_stage=current_stage,
            action="HOLD",
            reason=f"readiness_not_met:{readiness.reason}",
            recommended_capital_fraction=current.capital_fraction,
        )

    next_stage = policy.stages[current_idx + 1]
    if days_in_stage < next_stage.min_days_in_stage:
        return CapitalRampDecision(
            current_stage=current_stage,
            recommended_stage=current_stage,
            action="HOLD",
            reason="insufficient_days_in_stage",
            recommended_capital_fraction=current.capital_fraction,
        )

    if not _meets_stage_thresholds(readiness.metrics, next_stage):
        return CapitalRampDecision(
            current_stage=current_stage,
            recommended_stage=current_stage,
            action="HOLD",
            reason="next_stage_thresholds_not_met",
            recommended_capital_fraction=current.capital_fraction,
        )

    return CapitalRampDecision(
        current_stage=current_stage,
        recommended_stage=next_stage.name,
        action="PROMOTE",
        reason="stage_thresholds_met",
        recommended_capital_fraction=next_stage.capital_fraction,
    )


def render_capital_ramp_policy(policy: CapitalRampPolicy | None = None) -> str:
    """Render policy as plain text artifact for operator runbooks."""
    policy = policy or default_capital_ramp_policy()

    lines = [
        "CAPITAL RAMP POLICY",
        "=" * 70,
        f"Rollback on any kill event: {policy.rollback_on_any_kill}",
        f"Rollback target is paper stage: {policy.rollback_to_paper}",
        f"Hard drawdown rollback threshold: {policy.hard_drawdown_abs:.2%}",
        "",
        "Stages:",
    ]

    for stage in policy.stages:
        lines.extend(
            [
                f"- {stage.name}",
                f"  capital_fraction={stage.capital_fraction:.0%}",
                f"  min_days_in_stage={stage.min_days_in_stage}",
                f"  min_profit_factor_net={stage.min_profit_factor_net:.2f}",
                f"  min_sharpe_net={stage.min_sharpe_net:.2f}",
                f"  max_drawdown_abs={stage.max_drawdown_abs:.2%}",
            ]
        )

    return "\n".join(lines)


def build_kill_event_taxonomy(events: list[KillSwitchEvent]) -> dict[str, Any]:
    """Aggregate kill events by trigger/reason/regime/activity for diagnostics."""
    if not events:
        return {
            "total_events": 0,
            "soft_events": 0,
            "hard_events": 0,
            "mixed_events": 0,
            "by_reason": {},
            "by_trigger_type": {},
            "by_frequency": {},
            "by_dominant_regime": {},
            "by_activity_level": {},
            "reason_by_regime": {},
            "avg_cost_context": {},
        }

    by_reason: dict[str, int] = {}
    by_trigger_type: dict[str, int] = {}
    by_frequency: dict[str, int] = {}
    by_regime: dict[str, int] = {}
    by_activity_level: dict[str, int] = {}
    reason_by_regime: dict[str, dict[str, int]] = {}

    cost_metric_keys = (
        "cost_to_gross",
        "total_costs",
        "fees",
        "slippage",
        "funding",
        "interest",
        "other_costs",
    )
    metric_sums = {key: 0.0 for key in cost_metric_keys}
    metric_counts = {key: 0 for key in cost_metric_keys}

    for event in events:
        trigger = event.trigger_type or _classify_trigger_type(event.reasons)
        by_trigger_type[trigger] = by_trigger_type.get(trigger, 0) + 1

        freq = event.frequency or "unknown"
        by_frequency[freq] = by_frequency.get(freq, 0) + 1

        regime = event.dominant_regime or "unknown"
        by_regime[regime] = by_regime.get(regime, 0) + 1

        activity = event.activity_level or "unknown"
        by_activity_level[activity] = by_activity_level.get(activity, 0) + 1

        for reason in event.reasons:
            by_reason[reason] = by_reason.get(reason, 0) + 1
            reason_map = reason_by_regime.setdefault(reason, {})
            reason_map[regime] = reason_map.get(regime, 0) + 1

        for key in cost_metric_keys:
            value = _to_float(event.metrics.get(key), default=float("nan"))
            if pd.isna(value):
                continue
            metric_sums[key] += float(value)
            metric_counts[key] += 1

    avg_cost_context = {
        key: (metric_sums[key] / metric_counts[key]) if metric_counts[key] > 0 else None
        for key in cost_metric_keys
    }

    return {
        "total_events": len(events),
        "soft_events": int(by_trigger_type.get("soft", 0)),
        "hard_events": int(by_trigger_type.get("hard", 0)),
        "mixed_events": int(by_trigger_type.get("mixed", 0)),
        "by_reason": dict(sorted(by_reason.items())),
        "by_trigger_type": dict(sorted(by_trigger_type.items())),
        "by_frequency": dict(sorted(by_frequency.items())),
        "by_dominant_regime": dict(sorted(by_regime.items())),
        "by_activity_level": dict(sorted(by_activity_level.items())),
        "reason_by_regime": {
            reason: dict(sorted(regimes.items()))
            for reason, regimes in sorted(reason_by_regime.items())
        },
        "avg_cost_context": avg_cost_context,
    }


def build_low_activity_diagnostics(
    dashboard: pd.DataFrame,
    *,
    flagged_dashboard: pd.DataFrame | None = None,
    thresholds: KillSwitchThresholds | None = None,
) -> dict[str, Any]:
    """Summarize inactivity and soft-trigger suppression for monitoring windows."""
    thresholds = thresholds or KillSwitchThresholds()
    if dashboard.empty:
        return {
            "total_windows": 0,
            "active_windows": 0,
            "inactive_windows": 0,
            "inactive_ratio": 0.0,
            "soft_eligible_windows": 0,
            "soft_ineligible_windows": 0,
            "soft_ineligible_active_windows": 0,
            "triggered_windows": 0,
            "triggered_inactive_windows": 0,
            "triggered_soft_windows": 0,
            "triggered_hard_windows": 0,
            "triggered_mixed_windows": 0,
            "by_activity_level": {},
            "thresholds": {
                "min_active_bars": thresholds.min_active_bars,
                "min_trades_for_pf_sharpe": thresholds.min_trades_for_pf_sharpe,
                "min_active_bars_for_soft": thresholds.min_active_bars_for_soft,
                "min_bars_for_soft": thresholds.min_bars_for_soft,
                "min_soft_window_turnover": thresholds.min_soft_window_turnover,
                "min_soft_window_abs_net_return": thresholds.min_soft_window_abs_net_return,
                "min_bars_for_cost_to_gross": thresholds.min_bars_for_cost_to_gross,
                "min_abs_gross_return_for_cost_to_gross": thresholds.min_abs_gross_return_for_cost_to_gross,
            },
        }

    working = flagged_dashboard.copy() if flagged_dashboard is not None else dashboard.copy()
    if "activity_level" not in working.columns:
        working["activity_level"] = working.apply(_activity_level_from_row, axis=1)

    active_mask = working.apply(
        lambda row: _row_has_activity(row, min_active_bars=thresholds.min_active_bars),
        axis=1,
    )
    if "soft_criteria_eligible" in working.columns:
        soft_eligible_mask = working["soft_criteria_eligible"].fillna(0).astype(int) > 0
    else:
        soft_eligible_mask = working.apply(
            lambda row: _soft_criteria_eligible(
                row=row,
                thresholds=thresholds,
                num_trades=(
                    int(_to_float(row.get("num_trades"), default=0.0))
                    if row.get("num_trades") is not None and not pd.isna(row.get("num_trades"))
                    else None
                ),
            ),
            axis=1,
        )

    if "kill_switch_triggered" in working.columns:
        triggered_mask = working["kill_switch_triggered"].fillna(0).astype(int) > 0
    else:
        triggered_mask = pd.Series(False, index=working.index)

    if "kill_switch_trigger_type" in working.columns:
        trigger_type = working["kill_switch_trigger_type"].fillna("none").astype(str)
    else:
        trigger_type = pd.Series("none", index=working.index)

    activity_counts = (
        working["activity_level"].fillna("unknown").astype(str).value_counts().sort_index().to_dict()
    )

    return {
        "total_windows": int(len(working)),
        "active_windows": int(active_mask.sum()),
        "inactive_windows": int((~active_mask).sum()),
        "inactive_ratio": float((~active_mask).mean()),
        "soft_eligible_windows": int(soft_eligible_mask.sum()),
        "soft_ineligible_windows": int((~soft_eligible_mask).sum()),
        "soft_ineligible_active_windows": int((active_mask & ~soft_eligible_mask).sum()),
        "triggered_windows": int(triggered_mask.sum()),
        "triggered_inactive_windows": int((triggered_mask & ~active_mask).sum()),
        "triggered_soft_windows": int((triggered_mask & (trigger_type == "soft")).sum()),
        "triggered_hard_windows": int((triggered_mask & (trigger_type == "hard")).sum()),
        "triggered_mixed_windows": int((triggered_mask & (trigger_type == "mixed")).sum()),
        "by_activity_level": {str(k): int(v) for k, v in activity_counts.items()},
        "thresholds": {
            "min_active_bars": thresholds.min_active_bars,
            "min_trades_for_pf_sharpe": thresholds.min_trades_for_pf_sharpe,
            "min_active_bars_for_soft": thresholds.min_active_bars_for_soft,
            "min_bars_for_soft": thresholds.min_bars_for_soft,
            "min_soft_window_turnover": thresholds.min_soft_window_turnover,
            "min_soft_window_abs_net_return": thresholds.min_soft_window_abs_net_return,
            "min_bars_for_cost_to_gross": thresholds.min_bars_for_cost_to_gross,
            "min_abs_gross_return_for_cost_to_gross": thresholds.min_abs_gross_return_for_cost_to_gross,
        },
    }


def serialize_kill_events(events: list[KillSwitchEvent]) -> list[dict[str, Any]]:
    """Serialize kill events to plain dictionaries for JSON artifacts."""
    return [event.to_dict() for event in events]
