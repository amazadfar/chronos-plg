"""Phase 11 promotion-campaign utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.common.metrics import DEFAULT_SUCCESS_THRESHOLDS
from src.paper_trading.policy import CapitalRampDecision, DeploymentReadiness


@dataclass(frozen=True)
class Phase11CompletionCriteria:
    """Completion thresholds for final Phase 11 objective checks."""

    min_profit_factor_net: float = DEFAULT_SUCCESS_THRESHOLDS.min_profit_factor_net
    min_sharpe_net: float = DEFAULT_SUCCESS_THRESHOLDS.min_sharpe_net
    max_drawdown_abs: float = DEFAULT_SUCCESS_THRESHOLDS.max_drawdown_abs
    min_trades: int = 80
    max_kill_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_profit_factor_net": float(self.min_profit_factor_net),
            "min_sharpe_net": float(self.min_sharpe_net),
            "max_drawdown_abs": float(self.max_drawdown_abs),
            "min_trades": int(self.min_trades),
            "max_kill_events": int(self.max_kill_events),
        }


def _to_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, *, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def select_phase11_campaign_candidate(
    ranked_candidates: pd.DataFrame,
    *,
    candidate_id: int | None = None,
) -> dict[str, Any]:
    """Select frozen candidate for Phase 11 campaign from ranked sweep output."""
    if ranked_candidates.empty:
        raise ValueError("Cannot select campaign candidate from empty ranked table")

    ranked = ranked_candidates.copy()
    if "candidate_id" in ranked.columns:
        ranked["candidate_id"] = pd.to_numeric(ranked["candidate_id"], errors="coerce")

    if candidate_id is not None:
        match = ranked[ranked["candidate_id"] == int(candidate_id)]
        if match.empty:
            raise ValueError(f"Requested candidate_id={candidate_id} not present in ranked table")
        return {
            "selection_mode": "explicit_candidate_id",
            "candidate": match.iloc[0].to_dict(),
        }

    has_acceptance = "acceptance_passed" in ranked.columns
    accepted = ranked[ranked["acceptance_passed"].astype(bool)] if has_acceptance else pd.DataFrame()
    if has_acceptance and not accepted.empty:
        return {
            "selection_mode": "best_accepted",
            "candidate": accepted.iloc[0].to_dict(),
            "accepted_pool_size": int(len(accepted)),
        }

    trades = pd.to_numeric(ranked.get("num_trades", 0), errors="coerce").fillna(0.0)
    active = ranked[trades > 0].copy()
    if not active.empty:
        return {
            "selection_mode": "best_active_fallback",
            "candidate": active.iloc[0].to_dict(),
            "accepted_pool_size": 0 if has_acceptance else None,
            "active_pool_size": int(len(active)),
        }

    return {
        "selection_mode": "best_overall_fallback",
        "candidate": ranked.iloc[0].to_dict(),
        "accepted_pool_size": 0 if has_acceptance else None,
    }


def build_promotion_recommendation(
    *,
    readiness: DeploymentReadiness,
    ramp_decision: CapitalRampDecision,
) -> dict[str, Any]:
    """Require readiness + ramp promotion without policy exceptions."""
    exceptions: list[str] = []
    if not readiness.ready:
        exceptions.append(f"readiness_not_met:{readiness.reason}")
    if str(ramp_decision.action).upper() != "PROMOTE":
        exceptions.append(f"ramp_action_not_promote:{ramp_decision.action}")
    if str(ramp_decision.recommended_stage).lower() == str(ramp_decision.current_stage).lower():
        exceptions.append("recommended_stage_unchanged")

    recommend = len(exceptions) == 0
    return {
        "recommend_promotion": bool(recommend),
        "reason": (
            "promotion_recommended_no_policy_exceptions"
            if recommend
            else ",".join(exceptions)
        ),
        "policy_exceptions": exceptions,
        "readiness_ready": bool(readiness.ready),
        "readiness_reason": str(readiness.reason),
        "ramp_action": str(ramp_decision.action),
        "current_stage": str(ramp_decision.current_stage),
        "recommended_stage": str(ramp_decision.recommended_stage),
        "recommended_capital_fraction": float(ramp_decision.recommended_capital_fraction),
    }


def evaluate_phase11_completion_gate(
    *,
    metrics: dict[str, Any],
    readiness: DeploymentReadiness,
    ramp_decision: CapitalRampDecision,
    kill_event_count: int,
    criteria: Phase11CompletionCriteria | None = None,
) -> dict[str, Any]:
    """Evaluate if Phase 11 V0.1 objective is fully satisfied."""
    criteria = criteria or Phase11CompletionCriteria()

    pf_net = _to_float(metrics.get("profit_factor_net"))
    sharpe = _to_float(metrics.get("sharpe_ratio"))
    max_dd_abs = abs(_to_float(metrics.get("max_drawdown")))
    num_trades = _to_int(metrics.get("num_trades"))
    kill_events = int(max(0, kill_event_count))
    ramp_action = str(ramp_decision.action).upper()

    checks = {
        "profit_factor_net": pf_net > criteria.min_profit_factor_net,
        "sharpe_net": sharpe >= criteria.min_sharpe_net,
        "max_drawdown_abs": max_dd_abs <= criteria.max_drawdown_abs,
        "num_trades": num_trades >= criteria.min_trades,
        "kill_events": kill_events <= criteria.max_kill_events,
        "deployment_readiness": bool(readiness.ready),
        "capital_ramp_promote": ramp_action == "PROMOTE",
    }
    passed = all(checks.values())
    failed = [name for name, ok in checks.items() if not ok]

    return {
        "passed": bool(passed),
        "reason": "all_completion_checks_passed" if passed else ",".join(f"failed_{k}" for k in failed),
        "checks": checks,
        "criteria": criteria.to_dict(),
        "observed": {
            "profit_factor_net": pf_net,
            "sharpe_ratio": sharpe,
            "max_drawdown_abs": max_dd_abs,
            "num_trades": num_trades,
            "kill_event_count": kill_events,
            "deployment_readiness": bool(readiness.ready),
            "capital_ramp_action": ramp_action,
        },
    }
