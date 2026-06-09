"""Phase 4 cost engine tests."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.costs import CostModel


def test_transition_classification_legs():
    model = CostModel(fee_rate=0.001, slippage_base_bps=0.0, slippage_vol_multiplier=0.0)

    open_event = model._classify_transition(0.0, 1.2)
    assert open_event.event_type == "open"
    assert open_event.open_notional == 1.2
    assert open_event.close_notional == 0.0

    increase_event = model._classify_transition(1.0, 1.5)
    assert increase_event.event_type == "increase"
    assert increase_event.open_notional == 0.5
    assert increase_event.close_notional == 0.0

    reduce_event = model._classify_transition(1.5, 0.75)
    assert reduce_event.event_type == "reduce"
    assert reduce_event.open_notional == 0.0
    assert reduce_event.close_notional == 0.75

    close_event = model._classify_transition(0.75, 0.0)
    assert close_event.event_type == "close"
    assert close_event.open_notional == 0.0
    assert close_event.close_notional == 0.75

    reverse_event = model._classify_transition(1.0, -0.5)
    assert reverse_event.event_type == "reverse"
    assert reverse_event.open_notional == 0.5
    assert reverse_event.close_notional == 1.0


def test_reverse_transition_fee_legs():
    model = CostModel(
        fee_rate=0.001,
        slippage_base_bps=0.0,
        slippage_vol_multiplier=0.0,
        apply_funding=False,
        apply_margin_interest=False,
    )
    event = model.calculate_event_costs(
        prev_position=1.0,
        new_position=-0.5,
        volatility=0.0,
        funding_rate=0.0,
    )

    assert event.event_type == "reverse"
    assert math.isclose(event.open_notional, 0.5)
    assert math.isclose(event.close_notional, 1.0)
    assert math.isclose(event.traded_notional, 1.5)
    assert math.isclose(event.fees, 0.0015)
    assert math.isclose(event.total_costs, event.fees)


def test_funding_cashflow_sign_long_vs_short():
    model = CostModel(
        fee_rate=0.0,
        slippage_base_bps=0.0,
        slippage_vol_multiplier=0.0,
        apply_funding=True,
        apply_margin_interest=False,
    )

    long_event = model.calculate_event_costs(
        prev_position=1.0,
        new_position=1.0,
        funding_rate=0.0002,
    )
    short_event = model.calculate_event_costs(
        prev_position=-1.0,
        new_position=-1.0,
        funding_rate=0.0002,
    )

    assert long_event.funding > 0.0
    assert short_event.funding < 0.0
    assert math.isclose(long_event.funding, -short_event.funding)


def test_margin_interest_accrual_uses_holding_and_time():
    model = CostModel(
        fee_rate=0.0,
        slippage_base_bps=0.0,
        slippage_vol_multiplier=0.0,
        apply_funding=False,
        apply_margin_interest=True,
        margin_interest_rate_per_day=0.01,
    )

    event = model.calculate_event_costs(
        prev_position=-2.0,
        new_position=-2.0,
        funding_rate=0.0,
        bar_seconds=3600,
    )
    expected = 2.0 * 0.01 * (1.0 / 24.0)
    assert math.isclose(event.interest, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_execution_cost_audit_table_contains_all_components():
    idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    positions = pd.Series([0.0, 1.0, 1.5, 1.0, -0.5, 0.0], index=idx)
    vols = pd.Series(0.0, index=idx)
    funding = pd.Series([0.0, 0.001, 0.001, -0.001, 0.0005, 0.0], index=idx)
    borrow = pd.Series(0.01, index=idx)
    other = pd.Series([0.0, 0.0, 0.0, 0.0, 0.001, 0.0], index=idx)

    model = CostModel(
        fee_rate=0.001,
        slippage_base_bps=0.0,
        slippage_vol_multiplier=0.0,
        apply_funding=True,
        apply_margin_interest=True,
    )
    costs = model.calculate_execution_costs(
        positions=positions,
        volatilities=vols,
        funding_rates=funding,
        borrow_rates_per_day=borrow,
        other_costs=other,
    )

    required = {
        "event_type",
        "prev_position",
        "position",
        "open_notional",
        "close_notional",
        "traded_notional",
        "fees",
        "funding",
        "interest",
        "slippage",
        "other_costs",
        "total_costs",
    }
    assert required.issubset(set(costs.columns))

    assert list(costs["event_type"]) == ["hold", "open", "increase", "reduce", "reverse", "close"]
    assert math.isclose(costs.iloc[4]["open_notional"], 0.5)
    assert math.isclose(costs.iloc[4]["close_notional"], 1.0)
    assert costs.iloc[2]["funding"] > 0.0
    assert costs.iloc[4]["other_costs"] >= 0.001


def test_estimated_entry_cost_rate_respects_round_trip_setting():
    model = CostModel(
        fee_rate=0.001,
        slippage_base_bps=0.0,
        slippage_vol_multiplier=0.0,
        apply_funding=False,
        apply_margin_interest=False,
    )
    entry_only = model.estimate_entry_cost_rate(
        volatility=0.0,
        include_exit=False,
        target_notional=1.0,
    )
    round_trip = model.estimate_entry_cost_rate(
        volatility=0.0,
        include_exit=True,
        target_notional=1.0,
    )

    assert math.isclose(entry_only, 0.001, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(round_trip, 0.002, rel_tol=1e-12, abs_tol=1e-12)
    assert round_trip > entry_only


def test_estimated_entry_cost_series_increases_with_volatility():
    idx = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    vol = pd.Series([0.01, 0.03, 0.08], index=idx, dtype=float)
    model = CostModel(
        fee_rate=0.0005,
        slippage_base_bps=1.0,
        slippage_vol_multiplier=0.5,
        apply_funding=False,
        apply_margin_interest=False,
    )

    estimate = model.estimate_entry_cost_series(
        index=idx,
        volatilities=vol,
        expected_holding_bars=1,
        include_exit=True,
    )

    assert len(estimate) == len(idx)
    assert estimate.iloc[2] > estimate.iloc[1] > estimate.iloc[0] > 0.0
