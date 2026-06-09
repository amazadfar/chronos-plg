"""Phase 11.5 multi-objective sweep scoring tests."""

from __future__ import annotations

import pandas as pd

from src.evaluation.multi_objective import (
    AcceptanceConstraints,
    CompositeScoreWeights,
    apply_acceptance_constraints,
    pareto_frontier,
    rank_candidates,
    schema_payload,
)


def _sample_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": 1,
                "profit_factor_net": 1.20,
                "sharpe_ratio": 0.70,
                "num_trades": 120,
                "kill_event_rate": 0.08,
                "turnover": 18.0,
                "max_drawdown": -0.12,
            },
            {
                "candidate_id": 2,
                "profit_factor_net": 0.95,
                "sharpe_ratio": 0.20,
                "num_trades": 90,
                "kill_event_rate": 0.06,
                "turnover": 12.0,
                "max_drawdown": -0.10,
            },
            {
                "candidate_id": 3,
                "profit_factor_net": 1.10,
                "sharpe_ratio": 0.60,
                "num_trades": 40,
                "kill_event_rate": 0.03,
                "turnover": 7.0,
                "max_drawdown": -0.08,
            },
            {
                "candidate_id": 4,
                "profit_factor_net": 1.18,
                "sharpe_ratio": 0.62,
                "num_trades": 130,
                "kill_event_rate": 0.11,
                "turnover": 22.0,
                "max_drawdown": -0.18,
            },
        ]
    )


def test_rank_candidates_applies_acceptance_and_composite_score():
    ranked = rank_candidates(
        _sample_candidates(),
        weights=CompositeScoreWeights(),
        constraints=AcceptanceConstraints(
            min_profit_factor_net=1.0,
            min_sharpe_net=0.5,
            min_trades=80,
            max_kill_event_rate=0.2,
            max_drawdown_abs=0.3,
        ),
    )

    assert not ranked.empty
    assert "composite_score" in ranked.columns
    assert "acceptance_passed" in ranked.columns
    assert "active_candidate" in ranked.columns
    # Candidate 1 should pass and rank above candidates with constraint failures.
    top = ranked.iloc[0]
    assert bool(top["acceptance_passed"]) is True
    assert int(top["candidate_id"]) == 1

    c2 = ranked.loc[ranked["candidate_id"] == 2].iloc[0]
    assert bool(c2["acceptance_passed"]) is False
    assert "profit_factor_net" in str(c2["acceptance_failed_criteria"])

    c3 = ranked.loc[ranked["candidate_id"] == 3].iloc[0]
    assert bool(c3["acceptance_passed"]) is False
    assert "num_trades" in str(c3["acceptance_failed_criteria"])


def test_apply_acceptance_constraints_reports_failed_criteria():
    constrained = apply_acceptance_constraints(
        _sample_candidates(),
        constraints=AcceptanceConstraints(
            min_profit_factor_net=1.15,
            min_sharpe_net=0.65,
            min_trades=100,
            max_kill_event_rate=0.09,
            max_drawdown_abs=0.15,
        ),
    )
    c4 = constrained.loc[constrained["candidate_id"] == 4].iloc[0]
    failed = set(str(c4["acceptance_failed_criteria"]).split(";"))
    assert {"kill_event_rate", "max_drawdown_abs"} <= failed


def test_pareto_frontier_returns_non_dominated_candidates():
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": 10,
                "profit_factor_net": 1.15,
                "sharpe_ratio": 0.55,
                "num_trades": 80,
                "kill_event_rate": 0.08,
                "max_drawdown_abs": 0.12,
                "turnover": 15.0,
                "composite_score": 1.0,
            },
            {
                "candidate_id": 11,
                "profit_factor_net": 1.10,
                "sharpe_ratio": 0.45,
                "num_trades": 70,
                "kill_event_rate": 0.10,
                "max_drawdown_abs": 0.15,
                "turnover": 17.0,
                "composite_score": 0.7,
            },
            {
                "candidate_id": 12,
                "profit_factor_net": 1.20,
                "sharpe_ratio": 0.58,
                "num_trades": 60,
                "kill_event_rate": 0.05,
                "max_drawdown_abs": 0.10,
                "turnover": 14.0,
                "composite_score": 1.1,
            },
        ]
    )
    frontier = pareto_frontier(candidates)
    ids = set(frontier["candidate_id"].tolist())
    assert 11 not in ids
    assert {10, 12} <= ids


def test_schema_payload_contains_phase11_fields():
    schema = schema_payload()
    assert schema["schema_version"] == "phase11_sweep_v1"
    assert "composite_score" in schema["fields"]
    assert "pareto_frontier" in schema["fields"]
    assert "active_candidate" in schema["fields"]


def test_rank_candidates_prefers_active_when_no_candidate_is_accepted():
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": 1,
                "profit_factor_net": 0.0,
                "sharpe_ratio": 0.0,
                "num_trades": 0,
                "kill_event_rate": 0.0,
                "turnover": 0.0,
                "max_drawdown": 0.0,
            },
            {
                "candidate_id": 2,
                "profit_factor_net": 0.7,
                "sharpe_ratio": -0.3,
                "num_trades": 20,
                "kill_event_rate": 0.1,
                "turnover": 2.0,
                "max_drawdown": -0.05,
            },
        ]
    )
    ranked = rank_candidates(candidates, constraints=AcceptanceConstraints(min_trades=80))
    top = ranked.iloc[0]
    assert int(top["candidate_id"]) == 2
    assert bool(top["active_candidate"]) is True
