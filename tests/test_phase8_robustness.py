"""Phase 8 robustness protocol tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.engine import BacktestEngine, BacktestResult
from src.models.baselines.random_walk import RandomWalkBaseline
from src.robustness.stress_tests import StressTester, StressTestResult, StressTestSuite
from src.robustness.summary import RobustnessSummary


def _build_base_result(n: int = 360) -> tuple[BacktestResult, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    signal = 0.002 + 0.001 * np.sin(np.linspace(0, 18, n))
    noise = np.random.default_rng(42).normal(0, 0.002, n)
    net = pd.Series(signal + noise, index=idx)
    regimes = pd.Series(
        np.where(np.arange(n) % 3 == 0, "trend", np.where(np.arange(n) % 3 == 1, "normal", "chop")),
        index=idx,
    )
    returns = pd.DataFrame({"net_return": net, "regime": regimes}, index=idx)

    # Build market path with one adverse region.
    market_ret = np.random.default_rng(7).normal(0, 0.004, n)
    market_ret[140:200] -= 0.015
    close = 40000.0 * np.exp(np.cumsum(market_ret))
    data = pd.DataFrame({"close": close, "feature_x": np.random.default_rng(9).normal(0, 1, n)}, index=idx)

    result = BacktestResult(
        sharpe_ratio=1.1,
        total_return=float((1 + net).prod() - 1),
        max_drawdown=-0.11,
        win_rate=float((net > 0).mean()),
        profit_factor_net=1.35,
        profit_factor=1.35,
        returns=returns,
        regime_sharpes={"trend": 1.4, "normal": 1.0, "chop": 0.8},
    )
    return result, data


def test_block_bootstrap_and_rolling_subperiod_protocols_produce_contiguous_stress_outputs():
    base_result, _ = _build_base_result()
    tester = StressTester(
        bootstrap_samples=4,
        bootstrap_block_bars=32,
        rolling_window_fraction=0.5,
        rolling_step_fraction=0.5,
    )

    block = tester._test_block_bootstrap_stability(base_result=base_result)
    rolling = tester._test_rolling_subperiod_stability(base_result=base_result)

    assert block.name == "Block Bootstrap Stability"
    assert block.details["method"] == "block_bootstrap"
    assert len(block.details["sample_sharpes"]) == 4
    assert np.isfinite(block.stressed_sharpe)

    assert rolling.name == "Rolling Subperiod Stability"
    assert rolling.details["window_bars"] > 0
    assert len(rolling.details["segments"]) >= 1
    assert np.isfinite(rolling.stressed_sharpe)


def test_regime_exclusion_and_adverse_window_protocols_use_phase8_logic():
    base_result, data = _build_base_result()
    tester = StressTester(adverse_window_bars=90, adverse_top_k=2)

    regime = tester._test_regime_exclusion_protocol(base_result=base_result)
    adverse = tester._test_adverse_window_protocol(base_result=base_result, data=data)

    assert regime.name == "Regime Exclusion"
    assert "scenarios" in regime.details
    assert len(regime.details["scenarios"]) >= 1

    assert adverse.name == "Adverse Window"
    assert "windows" in adverse.details
    assert len(adverse.details["windows"]) >= 1
    assert all("start" in row and "end" in row for row in adverse.details["windows"])


def test_run_all_includes_cost_grid_and_parameter_sweep_without_random_subsample(monkeypatch):
    base_result, data = _build_base_result()

    engine = BacktestEngine(
        model_class=RandomWalkBaseline,
        model_kwargs={"lookback_window": 50},
        cost_model=CostModel(),
    )
    base_entry = engine.signal_generator.entry_threshold
    base_uncertainty = engine.signal_generator.uncertainty_threshold
    base_leverage = engine.position_sizer.max_leverage
    base_fee_rate = engine.cost_model.fee_rate

    def fake_run_engine(
        engine: BacktestEngine,
        data: pd.DataFrame,
        feature_columns: list[str] | None,
        *,
        start_date: str | None,
        precomputed_folds=None,
    ) -> BacktestResult:
        fee_penalty = (engine.cost_model.fee_rate - base_fee_rate) * 200.0
        entry_penalty = abs(engine.signal_generator.entry_threshold - base_entry) * 80.0
        uncertainty_penalty = abs(engine.signal_generator.uncertainty_threshold - base_uncertainty) * 10.0
        leverage_penalty = abs(engine.position_sizer.max_leverage - base_leverage) * 0.5
        sharpe = 1.0 - fee_penalty - entry_penalty - uncertainty_penalty - leverage_penalty
        return BacktestResult(
            sharpe_ratio=float(sharpe),
            total_return=0.10,
            max_drawdown=-0.12,
            win_rate=0.55,
            profit_factor_net=1.2,
            profit_factor=1.2,
        )

    monkeypatch.setattr(StressTester, "_run_engine", staticmethod(fake_run_engine))
    tester = StressTester(bootstrap_samples=3, parameter_multipliers=(0.8, 1.2))
    suite = tester.run_all(
        base_result=base_result,
        engine=engine,
        data=data,
        feature_columns=["feature_x"],
        start_date=None,
    )

    names = [test.name for test in suite.tests]
    assert "Cost Stress Grid" in names
    assert "Parameter Sensitivity" in names
    assert "Block Bootstrap Stability" in names
    assert "Rolling Subperiod Stability" in names
    assert all("Subsample" not in name for name in names)


def test_robustness_summary_requires_stress_pass_rate_for_viability():
    result = BacktestResult(
        sharpe_ratio=1.0,
        total_return=0.25,
        max_drawdown=-0.10,
        win_rate=0.60,
        profit_factor_net=1.4,
        profit_factor=1.4,
    )
    suite = StressTestSuite(
        tests=[
            StressTestResult(
                name="T1",
                description="ok",
                passed=True,
                base_sharpe=1.0,
                stressed_sharpe=0.9,
                degradation=0.1,
            ),
            StressTestResult(
                name="T2",
                description="fail",
                passed=False,
                base_sharpe=1.0,
                stressed_sharpe=0.2,
                degradation=0.8,
            ),
        ]
    )

    summary = RobustnessSummary(min_stress_pass_rate=0.7, require_stress_suite=True)
    report_low = summary.generate_report(
        model_name="ModelA",
        result=result,
        baseline_sharpe=0.5,
        stress_suite=suite,
    )
    assert report_low.is_viable is False
    assert "pass rate < 70%" in report_low.verdict

    report_missing = summary.generate_report(
        model_name="ModelA",
        result=result,
        baseline_sharpe=0.5,
        stress_suite=None,
    )
    assert report_missing.is_viable is False
    assert "stress suite missing" in report_missing.verdict.lower()

