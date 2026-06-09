"""Phase 11.7 promotion campaign utility tests."""

from __future__ import annotations

import pandas as pd

from src.evaluation.phase11_campaign import (
    Phase11CompletionCriteria,
    build_promotion_recommendation,
    evaluate_phase11_completion_gate,
    select_phase11_campaign_candidate,
)
from src.paper_trading.policy import CapitalRampDecision, DeploymentReadiness


def test_select_campaign_candidate_prefers_best_accepted_row():
    ranked = pd.DataFrame(
        [
            {"candidate_id": 1, "acceptance_passed": False, "composite_score": 0.9},
            {"candidate_id": 2, "acceptance_passed": True, "composite_score": 0.8},
            {"candidate_id": 3, "acceptance_passed": True, "composite_score": 0.7},
        ]
    )
    selection = select_phase11_campaign_candidate(ranked)
    assert selection["selection_mode"] == "best_accepted"
    assert int(selection["candidate"]["candidate_id"]) == 2
    assert int(selection["accepted_pool_size"]) == 2


def test_select_campaign_candidate_honors_explicit_id():
    ranked = pd.DataFrame(
        [
            {"candidate_id": 10, "acceptance_passed": True},
            {"candidate_id": 11, "acceptance_passed": True},
        ]
    )
    selection = select_phase11_campaign_candidate(ranked, candidate_id=11)
    assert selection["selection_mode"] == "explicit_candidate_id"
    assert int(selection["candidate"]["candidate_id"]) == 11


def test_select_campaign_candidate_fallback_prefers_active_candidate():
    ranked = pd.DataFrame(
        [
            {"candidate_id": 1, "acceptance_passed": False, "num_trades": 0, "composite_score": 0.2},
            {"candidate_id": 2, "acceptance_passed": False, "num_trades": 10, "composite_score": -0.4},
        ]
    )
    selection = select_phase11_campaign_candidate(ranked)
    assert selection["selection_mode"] == "best_active_fallback"
    assert int(selection["candidate"]["candidate_id"]) == 2
    assert int(selection["active_pool_size"]) == 1


def test_build_promotion_recommendation_requires_readiness_and_promote_action():
    readiness = DeploymentReadiness(
        ready=False,
        reason="min_trades_not_met",
        observed_days=60,
        total_trades=40,
        metrics={},
    )
    ramp = CapitalRampDecision(
        current_stage="paper",
        recommended_stage="paper",
        action="HOLD",
        reason="readiness_not_met",
        recommended_capital_fraction=0.0,
    )
    rec = build_promotion_recommendation(readiness=readiness, ramp_decision=ramp)
    assert rec["recommend_promotion"] is False
    assert "readiness_not_met" in rec["reason"]


def test_phase11_completion_gate_passes_when_all_criteria_satisfied():
    readiness = DeploymentReadiness(
        ready=True,
        reason="all_criteria_met",
        observed_days=80,
        total_trades=140,
        metrics={},
    )
    ramp = CapitalRampDecision(
        current_stage="paper",
        recommended_stage="stage1",
        action="PROMOTE",
        reason="eligible",
        recommended_capital_fraction=0.25,
    )
    criteria = Phase11CompletionCriteria(
        min_profit_factor_net=1.0,
        min_sharpe_net=0.5,
        max_drawdown_abs=0.30,
        min_trades=80,
        max_kill_events=0,
    )
    gate = evaluate_phase11_completion_gate(
        metrics={
            "profit_factor_net": 1.15,
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.12,
            "num_trades": 120,
        },
        readiness=readiness,
        ramp_decision=ramp,
        kill_event_count=0,
        criteria=criteria,
    )
    assert gate["passed"] is True
    assert gate["checks"]["capital_ramp_promote"] is True


def test_phase11_completion_gate_fails_with_non_promote_and_kill_events():
    readiness = DeploymentReadiness(
        ready=True,
        reason="ok",
        observed_days=80,
        total_trades=140,
        metrics={},
    )
    ramp = CapitalRampDecision(
        current_stage="paper",
        recommended_stage="paper",
        action="HOLD",
        reason="not_enough_days",
        recommended_capital_fraction=0.0,
    )
    gate = evaluate_phase11_completion_gate(
        metrics={
            "profit_factor_net": 1.15,
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.12,
            "num_trades": 120,
        },
        readiness=readiness,
        ramp_decision=ramp,
        kill_event_count=2,
        criteria=Phase11CompletionCriteria(max_kill_events=0),
    )
    assert gate["passed"] is False
    assert gate["checks"]["capital_ramp_promote"] is False
    assert gate["checks"]["kill_events"] is False
