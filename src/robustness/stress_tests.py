"""
Stress testing for strategy robustness.

Phase 8 focus:
- Time-contiguous block bootstrap and rolling-subperiod stability (no random-point subsampling)
- Stress grid for fees/slippage/funding/borrow deterioration
- Regime-exclusion and adverse-window protocols
- Parameter sensitivity sweeps (entry/uncertainty/leverage)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.engine import BacktestEngine, BacktestResult
from src.common.metrics import sharpe_ratio
from src.strategy.position_sizing import PositionSizer
from src.strategy.signals import QuantileSignalGenerator

logger = logging.getLogger(__name__)


def _sharpe_degradation(base_sharpe: float, stressed_sharpe: float) -> float:
    """Positive value means deterioration vs baseline."""
    denom = max(abs(base_sharpe), 1e-6)
    return (base_sharpe - stressed_sharpe) / denom


@dataclass
class StressTestResult:
    """Result of a single stress test."""

    name: str
    description: str
    passed: bool
    base_sharpe: float
    stressed_sharpe: float
    degradation: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "base_sharpe": self.base_sharpe,
            "stressed_sharpe": self.stressed_sharpe,
            "degradation": self.degradation,
            "details": self.details,
        }


@dataclass
class StressTestSuite:
    """Results of all stress tests."""

    tests: list[StressTestResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(test.passed for test in self.tests)

    @property
    def pass_rate(self) -> float:
        if not self.tests:
            return 0.0
        return sum(1 for test in self.tests if test.passed) / len(self.tests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_rate": self.pass_rate,
            "all_passed": self.all_passed,
            "tests": [test.to_dict() for test in self.tests],
        }

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "STRESS TEST RESULTS",
            "=" * 70,
            "",
        ]
        for test in self.tests:
            icon = "PASS" if test.passed else "FAIL"
            lines.append(f"[{icon}] {test.name}")
            lines.append(f"  {test.description}")
            lines.append(
                f"  Base Sharpe: {test.base_sharpe:.3f} -> Stressed: {test.stressed_sharpe:.3f}"
            )
            lines.append(f"  Degradation: {test.degradation:.1%}")
            lines.append("")
        lines.append("-" * 70)
        lines.append(
            f"Pass Rate: {self.pass_rate:.0%} ({sum(1 for t in self.tests if t.passed)}/{len(self.tests)})"
        )
        return "\n".join(lines)


class StressTester:
    """
    Stress test suite for trading strategies.

    Stress modules:
    1. Cost stress grid (higher fees/slippage/funding/borrow)
    2. Regime exclusion protocol
    3. Adverse-window protocol
    4. Block bootstrap stability
    5. Rolling subperiod stability
    6. Parameter sensitivity sweep
    """

    def __init__(
        self,
        max_sharpe_degradation: float = 0.5,
        subsample_ratio: float = 0.7,
        bootstrap_samples: int = 5,
        bootstrap_block_bars: int = 84,  # ~14 days at 4h bars
        rolling_window_fraction: float = 0.4,
        rolling_step_fraction: float = 0.25,
        adverse_window_bars: int = 180,  # ~30 days at 4h bars
        adverse_top_k: int = 3,
        parameter_multipliers: tuple[float, ...] = (0.8, 1.2),
        random_seed: int = 42,
    ):
        self.max_sharpe_degradation = max_sharpe_degradation
        self.subsample_ratio = subsample_ratio
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_block_bars = bootstrap_block_bars
        self.rolling_window_fraction = rolling_window_fraction
        self.rolling_step_fraction = rolling_step_fraction
        self.adverse_window_bars = adverse_window_bars
        self.adverse_top_k = adverse_top_k
        self.parameter_multipliers = parameter_multipliers
        self.random_seed = random_seed

    def run_all(
        self,
        base_result: BacktestResult,
        engine: BacktestEngine,
        data: pd.DataFrame,
        feature_columns: list[str] | None = None,
        start_date: str | None = None,
        precomputed_folds: list[
            tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]
        ]
        | None = None,
    ) -> StressTestSuite:
        """
        Run full Phase 8 stress suite.
        """
        suite = StressTestSuite()
        suite.tests.append(
            self._test_cost_stress_grid(
                base_result=base_result,
                engine=engine,
                data=data,
                feature_columns=feature_columns,
                start_date=start_date,
                precomputed_folds=precomputed_folds,
            )
        )
        suite.tests.append(self._test_regime_exclusion_protocol(base_result=base_result))
        suite.tests.append(self._test_adverse_window_protocol(base_result=base_result, data=data))
        suite.tests.append(self._test_block_bootstrap_stability(base_result=base_result))
        suite.tests.append(self._test_rolling_subperiod_stability(base_result=base_result))
        suite.tests.append(
            self._test_parameter_sensitivity(
                base_result=base_result,
                engine=engine,
                data=data,
                feature_columns=feature_columns,
                start_date=start_date,
                precomputed_folds=precomputed_folds,
            )
        )
        return suite

    @staticmethod
    def _extract_net_returns(base_result: BacktestResult) -> pd.Series | None:
        if base_result.returns is None or "net_return" not in base_result.returns.columns:
            return None
        clean = base_result.returns["net_return"].dropna()
        if clean.empty:
            return None
        return clean

    def _clone_cost_model(
        self,
        base: CostModel,
        *,
        fee_mult: float = 1.0,
        slippage_mult: float = 1.0,
        borrow_mult: float = 1.0,
        other_cost_bps_add: float = 0.0,
    ) -> CostModel:
        return CostModel(
            fee_rate=base.fee_rate * fee_mult,
            slippage_base_bps=base.slippage_base_bps * slippage_mult,
            slippage_vol_multiplier=base.slippage_vol_multiplier * slippage_mult,
            slippage_size_coefficient=base.slippage_size_coefficient,
            exchange=base.exchange,
            market_type=base.market_type,
            order_type=base.order_type,
            use_fee_discount=base.use_fee_discount,
            apply_funding=base.apply_funding,
            apply_margin_interest=base.apply_margin_interest,
            margin_interest_rate_per_day=base.margin_interest_rate_per_day * borrow_mult,
            other_cost_bps=base.other_cost_bps + other_cost_bps_add,
            fixed_other_cost=base.fixed_other_cost,
        )

    @staticmethod
    def _clone_signal_generator(
        base: QuantileSignalGenerator,
        *,
        entry_threshold: float | None = None,
        uncertainty_threshold: float | None = None,
    ) -> QuantileSignalGenerator:
        return QuantileSignalGenerator(
            config=base.config,
            entry_threshold=entry_threshold if entry_threshold is not None else base.entry_threshold,
            risk_limit=base.risk_limit,
            uncertainty_threshold=(
                uncertainty_threshold
                if uncertainty_threshold is not None
                else base.uncertainty_threshold
            ),
        )

    @staticmethod
    def _clone_position_sizer(
        base: PositionSizer,
        *,
        max_leverage: float | None = None,
    ) -> PositionSizer:
        return PositionSizer(
            config=base.config,
            max_leverage=max_leverage if max_leverage is not None else base.max_leverage,
            vol_target=base.vol_target,
            min_position=base.min_position,
            market_type=base.market_type,
            leverage_caps_by_market=deepcopy(base.leverage_caps_by_market),
            default_max_turnover_per_step=base.default_max_turnover_per_step,
            allow_short=base.allow_short,
        )

    def _clone_engine(
        self,
        base: BacktestEngine,
        *,
        cost_model: CostModel | None = None,
        signal_generator: QuantileSignalGenerator | None = None,
        position_sizer: PositionSizer | None = None,
    ) -> BacktestEngine:
        try:
            regime_detector = deepcopy(base.regime_detector)
        except Exception:
            regime_detector = base.regime_detector

        return BacktestEngine(
            model_class=base.model_class,
            model_kwargs=deepcopy(base.model_kwargs),
            walk_forward_config=base.wf_config,
            cost_model=cost_model if cost_model is not None else deepcopy(base.cost_model),
            signal_generator=(
                signal_generator
                if signal_generator is not None
                else self._clone_signal_generator(base.signal_generator)
            ),
            position_sizer=(
                position_sizer
                if position_sizer is not None
                else self._clone_position_sizer(base.position_sizer)
            ),
            regime_detector=regime_detector,
            target_column=base.target_column,
        )

    @staticmethod
    def _run_engine(
        engine: BacktestEngine,
        data: pd.DataFrame,
        feature_columns: list[str] | None,
        *,
        start_date: str | None,
        precomputed_folds: list[
            tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]
        ]
        | None = None,
    ) -> BacktestResult:
        return engine.run(
            data=data,
            feature_columns=feature_columns,
            start_date=start_date,
            show_progress=False,
            precomputed_folds=precomputed_folds,
            collect_fold_metrics=False,
        )

    def _test_cost_stress_grid(
        self,
        *,
        base_result: BacktestResult,
        engine: BacktestEngine,
        data: pd.DataFrame,
        feature_columns: list[str] | None,
        start_date: str | None,
        precomputed_folds: list[
            tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]
        ]
        | None,
    ) -> StressTestResult:
        logger.info("Running cost stress grid...")
        scenarios = [
            {
                "name": "mild",
                "fee_mult": 1.25,
                "slippage_mult": 1.25,
                "funding_mult": 1.25,
                "borrow_mult": 1.25,
                "other_bps_add": 1.0,
            },
            {
                "name": "high",
                "fee_mult": 1.5,
                "slippage_mult": 1.5,
                "funding_mult": 1.5,
                "borrow_mult": 1.5,
                "other_bps_add": 2.0,
            },
            {
                "name": "severe",
                "fee_mult": 2.0,
                "slippage_mult": 2.0,
                "funding_mult": 2.0,
                "borrow_mult": 2.0,
                "other_bps_add": 4.0,
            },
        ]

        scenario_results: dict[str, dict[str, float]] = {}
        stressed_sharpes: list[float] = []
        degradations: list[float] = []

        for scenario in scenarios:
            stressed_data = data.copy()
            if "funding_rate" in stressed_data.columns:
                stressed_data["funding_rate"] = (
                    stressed_data["funding_rate"] * scenario["funding_mult"]
                )
            if "borrow_rate_per_day" in stressed_data.columns:
                stressed_data["borrow_rate_per_day"] = (
                    stressed_data["borrow_rate_per_day"] * scenario["borrow_mult"]
                )

            stressed_cost_model = self._clone_cost_model(
                base=engine.cost_model,
                fee_mult=scenario["fee_mult"],
                slippage_mult=scenario["slippage_mult"],
                borrow_mult=scenario["borrow_mult"],
                other_cost_bps_add=scenario["other_bps_add"],
            )
            stressed_engine = self._clone_engine(engine, cost_model=stressed_cost_model)

            try:
                stressed = self._run_engine(
                    engine=stressed_engine,
                    data=stressed_data,
                    feature_columns=feature_columns,
                    start_date=start_date,
                    precomputed_folds=precomputed_folds,
                )
                stressed_sharpe = float(stressed.sharpe_ratio)
            except Exception as exc:
                logger.warning("Cost stress scenario '%s' failed: %s", scenario["name"], exc)
                stressed_sharpe = float("-inf")

            degradation = (
                float("inf")
                if np.isneginf(stressed_sharpe)
                else _sharpe_degradation(base_result.sharpe_ratio, stressed_sharpe)
            )
            if np.isfinite(stressed_sharpe):
                stressed_sharpes.append(stressed_sharpe)
                degradations.append(degradation)

            scenario_results[scenario["name"]] = {
                "stressed_sharpe": stressed_sharpe,
                "degradation": degradation,
            }

        if not stressed_sharpes:
            return StressTestResult(
                name="Cost Stress Grid",
                description="Higher fees/slippage/funding/borrow stress scenarios",
                passed=False,
                base_sharpe=base_result.sharpe_ratio,
                stressed_sharpe=0.0,
                degradation=1.0,
                details={"scenarios": scenario_results},
            )

        worst_sharpe = min(stressed_sharpes)
        worst_degradation = max(degradations)
        passed = worst_degradation <= self.max_sharpe_degradation and worst_sharpe > 0.0

        return StressTestResult(
            name="Cost Stress Grid",
            description="Higher fees/slippage/funding/borrow stress scenarios",
            passed=passed,
            base_sharpe=base_result.sharpe_ratio,
            stressed_sharpe=worst_sharpe,
            degradation=worst_degradation,
            details={"scenarios": scenario_results},
        )

    def _test_regime_exclusion_protocol(
        self,
        *,
        base_result: BacktestResult,
    ) -> StressTestResult:
        logger.info("Running regime-exclusion protocol...")
        net_returns = self._extract_net_returns(base_result)
        if net_returns is None:
            return StressTestResult(
                name="Regime Exclusion",
                description="Cannot evaluate without return series",
                passed=False,
                base_sharpe=base_result.sharpe_ratio,
                stressed_sharpe=0.0,
                degradation=1.0,
            )

        regime_sharpes: dict[str, float] = {}
        if base_result.returns is not None and "regime" in base_result.returns.columns:
            regime_series = base_result.returns.loc[net_returns.index, "regime"].astype(str)
            for regime in sorted(regime_series.dropna().unique()):
                masked = net_returns[regime_series != regime]
                if len(masked) < 20:
                    continue
                regime_sharpes[f"exclude_{regime}"] = float(sharpe_ratio(masked))
        elif base_result.regime_sharpes:
            best_regime = max(base_result.regime_sharpes, key=base_result.regime_sharpes.get)
            others = [s for r, s in base_result.regime_sharpes.items() if r != best_regime]
            if others:
                regime_sharpes[f"exclude_{best_regime}"] = float(np.mean(others))

        if not regime_sharpes:
            return StressTestResult(
                name="Regime Exclusion",
                description="Insufficient regime segmentation",
                passed=True,
                base_sharpe=base_result.sharpe_ratio,
                stressed_sharpe=base_result.sharpe_ratio,
                degradation=0.0,
            )

        worst_label, worst_sharpe = min(regime_sharpes.items(), key=lambda kv: kv[1])
        degradation = _sharpe_degradation(base_result.sharpe_ratio, worst_sharpe)
        passed = worst_sharpe > 0.0 and degradation <= self.max_sharpe_degradation * 1.2
        return StressTestResult(
            name="Regime Exclusion",
            description="Performance when favorable regimes are excluded",
            passed=passed,
            base_sharpe=base_result.sharpe_ratio,
            stressed_sharpe=float(worst_sharpe),
            degradation=float(degradation),
            details={"worst_case": worst_label, "scenarios": regime_sharpes},
        )

    def _test_adverse_window_protocol(
        self,
        *,
        base_result: BacktestResult,
        data: pd.DataFrame,
    ) -> StressTestResult:
        logger.info("Running adverse-window protocol...")
        net_returns = self._extract_net_returns(base_result)
        if net_returns is None or len(net_returns) < self.adverse_window_bars + 20:
            return StressTestResult(
                name="Adverse Window",
                description="Insufficient returns for adverse-window stress",
                passed=True,
                base_sharpe=base_result.sharpe_ratio,
                stressed_sharpe=base_result.sharpe_ratio,
                degradation=0.0,
            )

        window = min(self.adverse_window_bars, max(30, len(net_returns) // 3))
        segments: list[tuple[pd.Timestamp, pd.Timestamp, pd.Series]] = []

        if "close" in data.columns:
            close = data["close"].reindex(net_returns.index).ffill().dropna()
            market_returns = np.log(close / close.shift(1)).dropna()
            rolling = market_returns.rolling(window).sum().dropna()
            worst_endpoints = rolling.nsmallest(self.adverse_top_k).index
            for end_ts in worst_endpoints:
                loc = market_returns.index.get_loc(end_ts)
                start_loc = max(0, loc - window + 1)
                idx = market_returns.index[start_loc : loc + 1]
                segment = net_returns.reindex(idx).dropna()
                if len(segment) >= 20:
                    segments.append((segment.index[0], segment.index[-1], segment))

        if not segments:
            rolling = net_returns.rolling(window).sum().dropna()
            worst_endpoints = rolling.nsmallest(self.adverse_top_k).index
            for end_ts in worst_endpoints:
                loc = net_returns.index.get_loc(end_ts)
                start_loc = max(0, loc - window + 1)
                segment = net_returns.iloc[start_loc : loc + 1]
                if len(segment) >= 20:
                    segments.append((segment.index[0], segment.index[-1], segment))

        if not segments:
            return StressTestResult(
                name="Adverse Window",
                description="Unable to build adverse windows",
                passed=False,
                base_sharpe=base_result.sharpe_ratio,
                stressed_sharpe=0.0,
                degradation=1.0,
            )

        window_results: list[dict[str, Any]] = []
        sharpes: list[float] = []
        for start_ts, end_ts, segment in segments:
            stressed_sharpe = float(sharpe_ratio(segment))
            sharpes.append(stressed_sharpe)
            window_results.append(
                {
                    "start": start_ts.isoformat(),
                    "end": end_ts.isoformat(),
                    "n_bars": int(len(segment)),
                    "sharpe": stressed_sharpe,
                }
            )

        worst_sharpe = float(min(sharpes))
        degradation = float(_sharpe_degradation(base_result.sharpe_ratio, worst_sharpe))
        passed = worst_sharpe > 0.0 and degradation <= self.max_sharpe_degradation * 1.3
        return StressTestResult(
            name="Adverse Window",
            description="Worst contiguous market windows",
            passed=passed,
            base_sharpe=base_result.sharpe_ratio,
            stressed_sharpe=worst_sharpe,
            degradation=degradation,
            details={"windows": window_results},
        )

    def _test_block_bootstrap_stability(
        self,
        *,
        base_result: BacktestResult,
    ) -> StressTestResult:
        logger.info("Running block-bootstrap stability...")
        net_returns = self._extract_net_returns(base_result)
        if net_returns is None or len(net_returns) < self.bootstrap_block_bars + 20:
            return StressTestResult(
                name="Block Bootstrap Stability",
                description="Insufficient data for block bootstrap",
                passed=True,
                base_sharpe=base_result.sharpe_ratio,
                stressed_sharpe=base_result.sharpe_ratio,
                degradation=0.0,
            )

        values = net_returns.to_numpy(dtype=float)
        n = len(values)
        block = min(self.bootstrap_block_bars, max(10, n // 4))
        target_len = max(int(n * self.subsample_ratio), block * 2)
        rng = np.random.default_rng(self.random_seed)

        sample_sharpes: list[float] = []
        for _ in range(self.bootstrap_samples):
            sample_values: list[float] = []
            while len(sample_values) < target_len:
                start = int(rng.integers(0, max(1, n - block + 1)))
                chunk = values[start : start + block]
                sample_values.extend(float(v) for v in chunk)
            sampled = pd.Series(sample_values[:target_len])
            sample_sharpes.append(float(sharpe_ratio(sampled)))

        worst_sharpe = float(min(sample_sharpes))
        degradation = float(_sharpe_degradation(base_result.sharpe_ratio, worst_sharpe))
        passed = worst_sharpe > 0.0 and degradation <= self.max_sharpe_degradation
        return StressTestResult(
            name="Block Bootstrap Stability",
            description="Time-contiguous block bootstrap stress",
            passed=passed,
            base_sharpe=base_result.sharpe_ratio,
            stressed_sharpe=worst_sharpe,
            degradation=degradation,
            details={
                "method": "block_bootstrap",
                "block_bars": block,
                "target_len": target_len,
                "sample_sharpes": sample_sharpes,
            },
        )

    def _test_rolling_subperiod_stability(
        self,
        *,
        base_result: BacktestResult,
    ) -> StressTestResult:
        logger.info("Running rolling-subperiod stability...")
        net_returns = self._extract_net_returns(base_result)
        if net_returns is None or len(net_returns) < 80:
            return StressTestResult(
                name="Rolling Subperiod Stability",
                description="Insufficient data for rolling subperiods",
                passed=True,
                base_sharpe=base_result.sharpe_ratio,
                stressed_sharpe=base_result.sharpe_ratio,
                degradation=0.0,
            )

        n = len(net_returns)
        window = max(int(n * self.rolling_window_fraction), min(self.bootstrap_block_bars, n))
        window = min(window, n)
        step = max(1, int(window * self.rolling_step_fraction))

        segment_sharpes: list[float] = []
        segments: list[dict[str, Any]] = []
        for start in range(0, n - window + 1, step):
            segment = net_returns.iloc[start : start + window]
            seg_sharpe = float(sharpe_ratio(segment))
            segment_sharpes.append(seg_sharpe)
            segments.append(
                {
                    "start": segment.index[0].isoformat(),
                    "end": segment.index[-1].isoformat(),
                    "n_bars": int(len(segment)),
                    "sharpe": seg_sharpe,
                }
            )

        if not segment_sharpes:
            return StressTestResult(
                name="Rolling Subperiod Stability",
                description="No rolling segments generated",
                passed=False,
                base_sharpe=base_result.sharpe_ratio,
                stressed_sharpe=0.0,
                degradation=1.0,
            )

        worst_sharpe = float(min(segment_sharpes))
        degradation = float(_sharpe_degradation(base_result.sharpe_ratio, worst_sharpe))
        passed = worst_sharpe > 0.0 and degradation <= self.max_sharpe_degradation * 1.2
        return StressTestResult(
            name="Rolling Subperiod Stability",
            description="Contiguous rolling-window subperiod stress",
            passed=passed,
            base_sharpe=base_result.sharpe_ratio,
            stressed_sharpe=worst_sharpe,
            degradation=degradation,
            details={"window_bars": window, "step_bars": step, "segments": segments},
        )

    def _test_parameter_sensitivity(
        self,
        *,
        base_result: BacktestResult,
        engine: BacktestEngine,
        data: pd.DataFrame,
        feature_columns: list[str] | None,
        start_date: str | None,
        precomputed_folds: list[
            tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]
        ]
        | None,
    ) -> StressTestResult:
        logger.info("Running parameter sensitivity sweep...")
        base_signal = engine.signal_generator
        base_sizer = engine.position_sizer

        scenarios: list[tuple[str, float, float, float]] = []
        for mult in self.parameter_multipliers:
            scenarios.append((f"entry_x{mult:.2f}", mult, 1.0, 1.0))
            scenarios.append((f"uncertainty_x{mult:.2f}", 1.0, mult, 1.0))
            scenarios.append((f"leverage_x{mult:.2f}", 1.0, 1.0, mult))

        scenario_sharpes: dict[str, float] = {}
        degradations: list[float] = []
        for name, entry_mult, uncertainty_mult, leverage_mult in scenarios:
            signal_generator = self._clone_signal_generator(
                base_signal,
                entry_threshold=max(base_signal.entry_threshold * entry_mult, 1e-8),
                uncertainty_threshold=max(base_signal.uncertainty_threshold * uncertainty_mult, 1e-8),
            )
            position_sizer = self._clone_position_sizer(
                base_sizer,
                max_leverage=max(base_sizer.max_leverage * leverage_mult, 0.1),
            )
            stressed_engine = self._clone_engine(
                engine,
                signal_generator=signal_generator,
                position_sizer=position_sizer,
            )

            try:
                stressed = self._run_engine(
                    engine=stressed_engine,
                    data=data,
                    feature_columns=feature_columns,
                    start_date=start_date,
                    precomputed_folds=precomputed_folds,
                )
                stressed_sharpe = float(stressed.sharpe_ratio)
            except Exception as exc:
                logger.warning("Parameter sensitivity scenario '%s' failed: %s", name, exc)
                stressed_sharpe = float("-inf")

            scenario_sharpes[name] = stressed_sharpe
            if np.isfinite(stressed_sharpe):
                degradations.append(
                    _sharpe_degradation(base_result.sharpe_ratio, stressed_sharpe)
                )

        finite_sharpes = [value for value in scenario_sharpes.values() if np.isfinite(value)]
        if not finite_sharpes:
            return StressTestResult(
                name="Parameter Sensitivity",
                description="Entry/uncertainty/leverage sweep",
                passed=False,
                base_sharpe=base_result.sharpe_ratio,
                stressed_sharpe=0.0,
                degradation=1.0,
                details={"scenarios": scenario_sharpes},
            )

        worst_sharpe = float(min(finite_sharpes))
        worst_degradation = float(max(degradations)) if degradations else 1.0
        passed = worst_sharpe > 0.0 and worst_degradation <= self.max_sharpe_degradation * 1.1
        return StressTestResult(
            name="Parameter Sensitivity",
            description="Entry/uncertainty/leverage sweep",
            passed=passed,
            base_sharpe=base_result.sharpe_ratio,
            stressed_sharpe=worst_sharpe,
            degradation=worst_degradation,
            details={"scenarios": scenario_sharpes},
        )

