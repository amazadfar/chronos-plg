"""Phase 10 paper-trading readiness tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostModel
from src.models.baselines.random_walk import RandomWalkBaseline
from src.paper_trading import (
    DeploymentReadiness,
    DeploymentReadinessPolicy,
    KillSwitchEvent,
    KillSwitchThresholds,
    PaperTradingConfig,
    PaperTradingEngine,
    build_daily_weekly_dashboards,
    default_capital_ramp_policy,
    evaluate_deployment_readiness,
    evaluate_kill_switch,
    recommend_capital_action,
    render_capital_ramp_policy,
)
from src.strategy.signals import QuantileSignalGenerator


def _sample_data(n: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")

    ret = rng.normal(0.0006, 0.004, n)
    close = 42000.0 * np.exp(np.cumsum(ret))

    data = pd.DataFrame(index=idx)
    data["close"] = close
    data["return_1"] = ret
    data["return_6"] = pd.Series(ret, index=idx).rolling(6).sum().fillna(0.0)
    data["realized_vol_6"] = pd.Series(ret, index=idx).rolling(6).std().fillna(0.02)
    data["funding_rate"] = 0.0001 * np.sin(np.linspace(0, 12, n))
    data["borrow_rate_per_day"] = 0.0002
    data["other_costs"] = 0.0
    data["forward_return"] = np.roll(ret, -1)
    data.loc[data.index[-1], "forward_return"] = np.nan
    return data


def _run_replay() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _sample_data()
    engine = PaperTradingEngine(
        model_class=RandomWalkBaseline,
        model_kwargs={"lookback_window": 72},
        config=PaperTradingConfig(
            retrain_interval_bars=18,
            training_window_bars=240,
            min_train_samples=90,
        ),
        cost_model=CostModel(
            exchange="binance",
            market_type="futures",
            order_type="taker",
            use_fee_discount=True,
            apply_funding=True,
        ),
    )

    replay = engine.run(
        data,
        feature_columns=["return_1", "return_6", "realized_vol_6", "funding_rate"],
        start_date="2024-03-01",
        model_name="random_walk",
        scenario_name="binance_futures_taker_discounted",
    )

    assert replay.backtest_result.returns is not None
    return replay.paper_log, replay.backtest_result.returns


def test_paper_trading_replay_uses_cost_engine_with_audit_columns():
    paper_log, returns = _run_replay()

    assert len(paper_log) > 0
    required = {
        "gross_return",
        "net_return",
        "fees",
        "slippage",
        "funding",
        "interest",
        "other_costs",
        "total_costs",
        "traded_notional",
    }
    assert required.issubset(set(returns.columns))

    lhs = (returns["gross_return"] - returns["total_costs"]).fillna(0.0)
    rhs = returns["net_return"].fillna(0.0)
    assert np.allclose(lhs.values, rhs.values)


def test_paper_trading_net_edge_policy_exposes_expected_cost_bridge():
    data = _sample_data()
    engine = PaperTradingEngine(
        model_class=RandomWalkBaseline,
        model_kwargs={"lookback_window": 72},
        config=PaperTradingConfig(
            retrain_interval_bars=18,
            training_window_bars=240,
            min_train_samples=90,
        ),
        cost_model=CostModel(
            exchange="binance",
            market_type="futures",
            order_type="taker",
            use_fee_discount=True,
            apply_funding=True,
        ),
        signal_generator=QuantileSignalGenerator(
            entry_policy="net_edge",
            net_edge_cost_multiplier=1.0,
            net_edge_risk_multiplier=0.0,
            expected_cost_holding_bars=1,
            expected_cost_round_trip=True,
        ),
    )

    replay = engine.run(
        data,
        feature_columns=["return_1", "return_6", "realized_vol_6", "funding_rate"],
        start_date="2024-03-01",
        model_name="random_walk",
        scenario_name="binance_futures_taker_discounted",
    )

    paper_log = replay.paper_log
    assert "expected_cost" in paper_log.columns
    assert "required_edge" in paper_log.columns
    assert "entry_policy" in paper_log.columns
    assert (paper_log["expected_cost"] >= 0).all()
    assert (paper_log["required_edge"] >= 0).all()


def test_monitoring_dashboards_include_phase10_metrics():
    _, returns = _run_replay()
    dashboards = build_daily_weekly_dashboards(returns)
    daily = dashboards["daily"]
    weekly = dashboards["weekly"]

    assert not daily.empty
    assert not weekly.empty

    expected_columns = {
        "active_bars",
        "profit_factor_net",
        "sharpe_ratio",
        "max_drawdown",
        "turnover",
        "fees",
        "slippage",
        "funding",
        "interest",
        "other_costs",
        "total_costs",
    }
    assert expected_columns.issubset(set(daily.columns))
    assert expected_columns.issubset(set(weekly.columns))


def test_kill_switch_triggers_on_threshold_violations():
    dashboard = pd.DataFrame(
        [
            {
                "window_end": "2024-01-07T00:00:00+00:00",
                "frequency": "W",
                "profit_factor_net": 1.20,
                "sharpe_ratio": 0.80,
                "max_drawdown": -0.10,
                "cost_to_gross": 0.20,
                "turnover": 3.0,
            },
            {
                "window_end": "2024-01-14T00:00:00+00:00",
                "frequency": "W",
                "profit_factor_net": 0.90,
                "sharpe_ratio": 0.20,
                "max_drawdown": -0.40,
                "cost_to_gross": 0.90,
                "turnover": 50.0,
            },
        ]
    )

    flagged, events = evaluate_kill_switch(
        dashboard,
        thresholds=KillSwitchThresholds(min_windows_before_enforcement=1),
    )

    assert int(flagged["kill_switch_triggered"].sum()) >= 1
    assert len(events) == 1
    assert "profit_factor_net_breach" in events[0].reasons
    assert events[0].action == "HALT"


def test_kill_switch_ignores_inactive_windows():
    dashboard = pd.DataFrame(
        [
            {
                "window_end": "2024-01-07T00:00:00+00:00",
                "frequency": "W",
                "active_bars": 0,
                "num_trades": 0,
                "profit_factor_net": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "cost_to_gross": np.nan,
                "turnover": 0.0,
            },
            {
                "window_end": "2024-01-14T00:00:00+00:00",
                "frequency": "W",
                "active_bars": 3,
                "num_trades": 2,
                "profit_factor_net": 0.80,
                "sharpe_ratio": 0.20,
                "max_drawdown": -0.05,
                "cost_to_gross": 0.20,
                "turnover": 2.0,
            },
        ]
    )

    flagged, events = evaluate_kill_switch(
        dashboard,
        thresholds=KillSwitchThresholds(min_windows_before_enforcement=1),
    )

    assert int(flagged.loc[0, "kill_switch_triggered"]) == 0
    assert int(flagged.loc[1, "kill_switch_triggered"]) == 1
    assert len(events) == 1


def test_kill_switch_skips_pf_sharpe_when_window_trade_count_is_too_low():
    dashboard = pd.DataFrame(
        [
            {
                "window_end": "2024-01-07T00:00:00+00:00",
                "frequency": "D",
                "active_bars": 3,
                "num_trades": 1,
                "profit_factor_net": 0.10,
                "sharpe_ratio": -4.0,
                "max_drawdown": -0.01,
                "cost_to_gross": 0.20,
                "turnover": 0.6,
            },
            {
                "window_end": "2024-01-08T00:00:00+00:00",
                "frequency": "D",
                "active_bars": 4,
                "num_trades": 4,
                "profit_factor_net": 0.10,
                "sharpe_ratio": -4.0,
                "max_drawdown": -0.01,
                "cost_to_gross": 0.20,
                "turnover": 0.8,
            },
        ]
    )

    flagged, events = evaluate_kill_switch(
        dashboard,
        thresholds=KillSwitchThresholds(
            min_windows_before_enforcement=1,
            min_trades_for_pf_sharpe=2,
        ),
    )

    assert int(flagged.loc[0, "kill_switch_triggered"]) == 0
    assert int(flagged.loc[1, "kill_switch_triggered"]) == 1
    assert len(events) == 1
    assert set(events[0].reasons) == {"profit_factor_net_breach", "sharpe_ratio_breach"}


def test_readiness_uses_recent_active_week_for_pf_check():
    _, returns = _run_replay()
    weekly_dashboard = pd.DataFrame(
        [
            {
                "window_end": "2024-02-04T00:00:00+00:00",
                "frequency": "W",
                "active_bars": 4,
                "num_trades": 2,
                "profit_factor_net": 1.10,
                "sharpe_ratio": 0.80,
                "max_drawdown": -0.05,
                "cost_to_gross": 0.20,
                "turnover": 4.0,
            },
            {
                "window_end": "2024-02-11T00:00:00+00:00",
                "frequency": "W",
                "active_bars": 0,
                "num_trades": 0,
                "profit_factor_net": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "cost_to_gross": np.nan,
                "turnover": 0.0,
            },
        ]
    )

    readiness = evaluate_deployment_readiness(
        returns=returns,
        weekly_dashboard=weekly_dashboard,
        kill_events=[],
        policy=DeploymentReadinessPolicy(
            min_observation_days=1,
            min_trades=0,
            min_profit_factor_net=-1.0,
            min_sharpe_net=-5.0,
            max_drawdown_abs=1.0,
            allow_kill_events=True,
        ),
    )

    assert readiness.ready is True
    assert "recent_week_pf_below_threshold" not in readiness.reason


def test_readiness_and_capital_ramp_policy_promote_and_rollback():
    _, returns = _run_replay()
    weekly_dashboard = build_daily_weekly_dashboards(returns)["weekly"]

    readiness = evaluate_deployment_readiness(
        returns=returns,
        weekly_dashboard=weekly_dashboard,
        kill_events=[],
        policy=DeploymentReadinessPolicy(
            min_observation_days=7,
            min_trades=0,
            min_profit_factor_net=-1.0,
            min_sharpe_net=-5.0,
            max_drawdown_abs=1.0,
            allow_kill_events=False,
        ),
    )
    assert readiness.ready is True

    strong_metrics = DeploymentReadiness(
        ready=True,
        reason="ready",
        observed_days=56,
        total_trades=120,
        metrics={
            "profit_factor_net": 1.25,
            "sharpe_ratio": 0.90,
            "max_drawdown": -0.12,
            "net_return": 0.18,
            "turnover": 12.0,
        },
    )

    policy = default_capital_ramp_policy()
    promote = recommend_capital_action(
        current_stage="paper",
        days_in_stage=30,
        weekly_dashboard=weekly_dashboard,
        readiness=strong_metrics,
        kill_events=[],
        policy=policy,
    )
    assert promote.action == "PROMOTE"
    assert promote.recommended_stage == "stage_1_10pct"

    rollback = recommend_capital_action(
        current_stage="stage_2_25pct",
        days_in_stage=45,
        weekly_dashboard=weekly_dashboard,
        readiness=strong_metrics,
        kill_events=[
            KillSwitchEvent(
                window_end="2024-02-01T00:00:00+00:00",
                frequency="W",
                reasons=["profit_factor_net_breach"],
                action="HALT",
                metrics={"profit_factor_net": 0.8},
            )
        ],
        policy=policy,
    )
    assert rollback.action == "ROLLBACK"
    assert rollback.recommended_stage == "paper"


@pytest.mark.parametrize("stage_name", ["paper", "stage_1_10pct", "stage_4_100pct"])
def test_ramp_policy_text_mentions_stages(stage_name: str):
    text = render_capital_ramp_policy(default_capital_ramp_policy())
    assert stage_name in text
