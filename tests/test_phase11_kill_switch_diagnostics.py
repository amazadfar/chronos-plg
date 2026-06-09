"""Phase 11.4 kill-switch diagnostics and calibration tests."""

from __future__ import annotations

import pandas as pd

from src.paper_trading import (
    KillSwitchThresholds,
    build_kill_event_taxonomy,
    build_low_activity_diagnostics,
    evaluate_kill_switch,
)


def test_kill_switch_classifies_soft_hard_and_mixed_triggers():
    dashboard = pd.DataFrame(
        [
            {
                "window_end": "2024-01-07T00:00:00+00:00",
                "frequency": "W",
                "dominant_regime": "trend",
                "activity_level": "medium",
                "num_bars": 7,
                "active_bars": 4,
                "num_trades": 3,
                "profit_factor_net": 0.80,
                "sharpe_ratio": 0.20,
                "max_drawdown": -0.05,
                "cost_to_gross": 0.20,
                "turnover": 3.0,
                "net_return": -0.01,
                "gross_return": -0.005,
                "total_costs": 0.002,
            },
            {
                "window_end": "2024-01-14T00:00:00+00:00",
                "frequency": "W",
                "dominant_regime": "chop",
                "activity_level": "medium",
                "num_bars": 7,
                "active_bars": 5,
                "num_trades": 4,
                "profit_factor_net": 1.20,
                "sharpe_ratio": 0.80,
                "max_drawdown": -0.45,
                "cost_to_gross": 0.20,
                "turnover": 3.0,
                "net_return": -0.02,
                "gross_return": -0.01,
                "total_costs": 0.003,
            },
            {
                "window_end": "2024-01-21T00:00:00+00:00",
                "frequency": "W",
                "dominant_regime": "panic",
                "activity_level": "high",
                "num_bars": 7,
                "active_bars": 6,
                "num_trades": 5,
                "profit_factor_net": 0.60,
                "sharpe_ratio": 0.10,
                "max_drawdown": -0.40,
                "cost_to_gross": 0.20,
                "turnover": 4.0,
                "net_return": -0.03,
                "gross_return": -0.015,
                "total_costs": 0.005,
            },
        ]
    )

    flagged, events = evaluate_kill_switch(
        dashboard,
        thresholds=KillSwitchThresholds(min_windows_before_enforcement=1),
    )

    assert list(flagged["kill_switch_trigger_type"]) == ["soft", "hard", "mixed"]
    assert [event.trigger_type for event in events] == ["soft", "hard", "mixed"]
    assert events[0].dominant_regime == "trend"
    assert events[2].activity_level == "high"


def test_kill_switch_suppresses_soft_checks_in_low_activity_windows():
    dashboard = pd.DataFrame(
        [
            {
                "window_end": "2024-01-01T00:00:00+00:00",
                "frequency": "D",
                "num_bars": 24,
                "active_bars": 1,
                "num_trades": 3,
                "profit_factor_net": 0.10,
                "sharpe_ratio": -3.00,
                "max_drawdown": -0.02,
                "cost_to_gross": 0.20,
                "turnover": 0.5,
                "net_return": -0.001,
                "gross_return": -0.0005,
                "total_costs": 0.0001,
            },
            {
                "window_end": "2024-01-02T00:00:00+00:00",
                "frequency": "D",
                "num_bars": 24,
                "active_bars": 4,
                "num_trades": 3,
                "profit_factor_net": 0.10,
                "sharpe_ratio": -3.00,
                "max_drawdown": -0.02,
                "cost_to_gross": 0.20,
                "turnover": 0.5,
                "net_return": -0.002,
                "gross_return": -0.001,
                "total_costs": 0.0002,
            },
        ]
    )

    flagged, events = evaluate_kill_switch(
        dashboard,
        thresholds=KillSwitchThresholds(
            min_windows_before_enforcement=1,
            min_active_bars_for_soft=2,
        ),
    )

    assert int(flagged.loc[0, "soft_criteria_eligible"]) == 0
    assert int(flagged.loc[0, "kill_switch_triggered"]) == 0
    assert int(flagged.loc[1, "soft_criteria_eligible"]) == 1
    assert int(flagged.loc[1, "kill_switch_triggered"]) == 1
    assert len(events) == 1
    assert events[0].trigger_type == "soft"


def test_kill_switch_suppresses_soft_checks_in_short_windows():
    dashboard = pd.DataFrame(
        [
            {
                "window_end": "2024-01-01T00:00:00+00:00",
                "frequency": "D",
                "num_bars": 6,
                "active_bars": 4,
                "num_trades": 4,
                "profit_factor_net": 0.10,
                "sharpe_ratio": -3.00,
                "max_drawdown": -0.02,
                "cost_to_gross": 0.20,
                "turnover": 0.8,
                "net_return": -0.002,
                "gross_return": -0.001,
                "total_costs": 0.0002,
            },
            {
                "window_end": "2024-01-02T00:00:00+00:00",
                "frequency": "D",
                "num_bars": 24,
                "active_bars": 4,
                "num_trades": 4,
                "profit_factor_net": 0.10,
                "sharpe_ratio": -3.00,
                "max_drawdown": -0.02,
                "cost_to_gross": 0.20,
                "turnover": 0.8,
                "net_return": -0.002,
                "gross_return": -0.001,
                "total_costs": 0.0002,
            },
        ]
    )

    flagged, events = evaluate_kill_switch(
        dashboard,
        thresholds=KillSwitchThresholds(
            min_windows_before_enforcement=1,
            min_active_bars_for_soft=2,
            min_bars_for_soft=12,
        ),
    )

    assert int(flagged.loc[0, "soft_criteria_eligible"]) == 0
    assert int(flagged.loc[0, "kill_switch_triggered"]) == 0
    assert int(flagged.loc[1, "soft_criteria_eligible"]) == 1
    assert int(flagged.loc[1, "kill_switch_triggered"]) == 1
    assert len(events) == 1
    assert events[0].window_end == "2024-01-02T00:00:00+00:00"
    assert events[0].trigger_type == "soft"


def test_kill_switch_cost_to_gross_requires_sufficient_window_and_gross_return():
    dashboard = pd.DataFrame(
        [
            {
                "window_end": "2024-01-01T00:00:00+00:00",
                "frequency": "D",
                "num_bars": 6,
                "active_bars": 4,
                "num_trades": 4,
                "profit_factor_net": 1.20,
                "sharpe_ratio": 0.80,
                "max_drawdown": -0.02,
                "cost_to_gross": 1.20,
                "turnover": 0.8,
                "net_return": -0.001,
                "gross_return": -0.003,
                "total_costs": 0.0036,
            },
            {
                "window_end": "2024-01-02T00:00:00+00:00",
                "frequency": "D",
                "num_bars": 24,
                "active_bars": 4,
                "num_trades": 4,
                "profit_factor_net": 1.20,
                "sharpe_ratio": 0.80,
                "max_drawdown": -0.02,
                "cost_to_gross": 1.20,
                "turnover": 0.8,
                "net_return": -0.001,
                "gross_return": -0.0005,
                "total_costs": 0.0006,
            },
            {
                "window_end": "2024-01-03T00:00:00+00:00",
                "frequency": "D",
                "num_bars": 24,
                "active_bars": 4,
                "num_trades": 4,
                "profit_factor_net": 1.20,
                "sharpe_ratio": 0.80,
                "max_drawdown": -0.02,
                "cost_to_gross": 1.20,
                "turnover": 0.8,
                "net_return": -0.003,
                "gross_return": -0.004,
                "total_costs": 0.0048,
            },
        ]
    )

    flagged, events = evaluate_kill_switch(
        dashboard,
        thresholds=KillSwitchThresholds(
            min_windows_before_enforcement=1,
            min_bars_for_cost_to_gross=12,
            min_abs_gross_return_for_cost_to_gross=0.002,
            max_cost_to_gross=0.70,
        ),
    )

    assert int(flagged.loc[0, "kill_switch_triggered"]) == 0
    assert int(flagged.loc[1, "kill_switch_triggered"]) == 0
    assert int(flagged.loc[2, "kill_switch_triggered"]) == 1
    assert flagged.loc[2, "kill_switch_reasons"] == "cost_to_gross_breach"
    assert len(events) == 1
    assert events[0].window_end == "2024-01-03T00:00:00+00:00"
    assert events[0].trigger_type == "hard"


def test_taxonomy_and_low_activity_diagnostics_report_expected_counts():
    dashboard = pd.DataFrame(
        [
            {
                "window_end": "2024-01-01T00:00:00+00:00",
                "frequency": "D",
                "dominant_regime": "normal",
                "num_bars": 24,
                "active_bars": 0,
                "num_trades": 0,
                "profit_factor_net": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "cost_to_gross": float("nan"),
                "turnover": 0.0,
                "net_return": 0.0,
                "gross_return": 0.0,
                "total_costs": 0.0,
            },
            {
                "window_end": "2024-01-02T00:00:00+00:00",
                "frequency": "D",
                "dominant_regime": "chop",
                "num_bars": 24,
                "active_bars": 1,
                "num_trades": 2,
                "profit_factor_net": 0.2,
                "sharpe_ratio": -2.0,
                "max_drawdown": -0.01,
                "cost_to_gross": 0.2,
                "turnover": 0.3,
                "net_return": -0.001,
                "gross_return": -0.0005,
                "total_costs": 0.0001,
            },
            {
                "window_end": "2024-01-03T00:00:00+00:00",
                "frequency": "D",
                "dominant_regime": "trend",
                "num_bars": 24,
                "active_bars": 4,
                "num_trades": 3,
                "profit_factor_net": 0.2,
                "sharpe_ratio": -2.0,
                "max_drawdown": -0.01,
                "cost_to_gross": 0.2,
                "turnover": 1.1,
                "net_return": -0.002,
                "gross_return": -0.001,
                "total_costs": 0.0002,
            },
            {
                "window_end": "2024-01-04T00:00:00+00:00",
                "frequency": "D",
                "dominant_regime": "panic",
                "num_bars": 24,
                "active_bars": 5,
                "num_trades": 4,
                "profit_factor_net": 1.2,
                "sharpe_ratio": 0.8,
                "max_drawdown": -0.5,
                "cost_to_gross": 0.2,
                "turnover": 1.3,
                "net_return": -0.003,
                "gross_return": -0.0015,
                "total_costs": 0.0003,
            },
        ]
    )

    thresholds = KillSwitchThresholds(
        min_windows_before_enforcement=1,
        min_active_bars_for_soft=2,
    )
    flagged, events = evaluate_kill_switch(dashboard, thresholds=thresholds)

    taxonomy = build_kill_event_taxonomy(events)
    diagnostics = build_low_activity_diagnostics(
        dashboard,
        flagged_dashboard=flagged,
        thresholds=thresholds,
    )

    assert taxonomy["total_events"] == 2
    assert taxonomy["by_trigger_type"]["soft"] == 1
    assert taxonomy["by_trigger_type"]["hard"] == 1
    assert taxonomy["by_dominant_regime"]["trend"] == 1
    assert taxonomy["by_dominant_regime"]["panic"] == 1
    assert taxonomy["by_reason"]["profit_factor_net_breach"] == 1
    assert taxonomy["by_reason"]["max_drawdown_breach"] == 1

    assert diagnostics["total_windows"] == 4
    assert diagnostics["active_windows"] == 3
    assert diagnostics["inactive_windows"] == 1
    assert diagnostics["soft_ineligible_active_windows"] == 1
    assert diagnostics["triggered_windows"] == 2
    assert diagnostics["triggered_inactive_windows"] == 0
    assert diagnostics["triggered_soft_windows"] == 1
    assert diagnostics["triggered_hard_windows"] == 1
    assert diagnostics["thresholds"]["min_bars_for_soft"] == 1
    assert diagnostics["thresholds"]["min_bars_for_cost_to_gross"] == 1
