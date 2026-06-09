"""
Kill criteria for strategy validation.

Explicit conditions that trigger project termination.
These are NOT negotiable - if any criterion is violated,
the strategy is not ready for live trading.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.common.metrics import (
    DEFAULT_SUCCESS_THRESHOLDS,
    MetricName,
    recent_vs_early_sharpe_ratio,
)

logger = logging.getLogger(__name__)


class CriterionStatus(Enum):
    """Status of a kill criterion."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class CriterionResult:
    """Result of a single criterion check."""
    name: str
    status: CriterionStatus
    value: float
    threshold: float
    message: str

    @property
    def passed(self) -> bool:
        return self.status == CriterionStatus.PASS


@dataclass
class KillCriteriaResult:
    """Aggregated result of all kill criteria checks."""
    criteria: list[CriterionResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.status != CriterionStatus.FAIL for c in self.criteria)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == CriterionStatus.WARNING for c in self.criteria)

    @property
    def num_passed(self) -> int:
        return sum(1 for c in self.criteria if c.status == CriterionStatus.PASS)

    @property
    def num_failed(self) -> int:
        return sum(1 for c in self.criteria if c.status == CriterionStatus.FAIL)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 70,
            "KILL CRITERIA CHECK",
            "=" * 70,
            "",
        ]

        for c in self.criteria:
            if c.status == CriterionStatus.PASS:
                icon = "✅"
            elif c.status == CriterionStatus.FAIL:
                icon = "❌"
            else:
                icon = "⚠️"

            lines.append(f"{icon} {c.name}")
            lines.append(f"   Value: {c.value:.4f} | Threshold: {c.threshold:.4f}")
            lines.append(f"   {c.message}")
            lines.append("")

        lines.append("-" * 70)
        if self.all_passed:
            lines.append("✅ ALL KILL CRITERIA PASSED - Strategy is viable")
        else:
            lines.append(f"❌ {self.num_failed} KILL CRITERIA FAILED - Strategy needs work")

        return "\n".join(lines)


class KillCriteria:
    """
    Kill criteria checker.

    Criteria:
    1. Profit Factor (Net): Must be > 1.0 (primary)
    2. Sharpe Ratio (Net): Must be > 0.5
    3. Max Drawdown: Must be < 30%
    4. Baseline Beat: Must beat LightGBM by Sharpe > 0.1
    5. Regime Stability: Sharpe CV across regimes < 1.5
    6. Win Rate: advisory quality check (not a hard fail gate)
    7. No Decay: Last-25% Sharpe >= 0.7 * First-25% Sharpe
    """

    def __init__(
        self,
        min_profit_factor_net: float = DEFAULT_SUCCESS_THRESHOLDS.min_profit_factor_net,
        min_sharpe: float = DEFAULT_SUCCESS_THRESHOLDS.min_sharpe_net,
        max_drawdown: float = DEFAULT_SUCCESS_THRESHOLDS.max_drawdown_abs,
        min_baseline_advantage: float = DEFAULT_SUCCESS_THRESHOLDS.min_baseline_sharpe_delta,
        max_regime_cv: float = DEFAULT_SUCCESS_THRESHOLDS.max_regime_sharpe_cv,
        min_win_rate: float = DEFAULT_SUCCESS_THRESHOLDS.min_win_rate,
        min_recent_sharpe_ratio: float = DEFAULT_SUCCESS_THRESHOLDS.min_recent_sharpe_ratio,
    ):
        """
        Args:
            min_profit_factor_net: Minimum profit factor net of costs
            min_sharpe: Minimum Sharpe ratio (after costs)
            max_drawdown: Maximum allowed drawdown (absolute)
            min_baseline_advantage: Minimum Sharpe improvement over baseline
            max_regime_cv: Maximum CV of Sharpe across regimes
            min_win_rate: Minimum win rate
            min_recent_sharpe_ratio: Minimum ratio of recent/early Sharpe
        """
        self.min_profit_factor_net = min_profit_factor_net
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_baseline_advantage = min_baseline_advantage
        self.max_regime_cv = max_regime_cv
        self.min_win_rate = min_win_rate
        self.min_recent_sharpe_ratio = min_recent_sharpe_ratio

    def check(
        self,
        result: BacktestResult,
        baseline_sharpe: float | None = None,
    ) -> KillCriteriaResult:
        """
        Check all kill criteria against a backtest result.

        Args:
            result: Backtest result to check
            baseline_sharpe: Sharpe ratio of baseline model (e.g., LightGBM)

        Returns:
            KillCriteriaResult with all checks
        """
        checks = KillCriteriaResult()

        # 1. Profit Factor (primary)
        profit_factor_net = result.profit_factor_net if result.profit_factor_net > 0 else result.profit_factor
        checks.criteria.append(self._check_profit_factor_net(profit_factor_net))

        # 2. Sharpe Ratio
        checks.criteria.append(self._check_sharpe(result.sharpe_ratio))

        # 3. Max Drawdown
        checks.criteria.append(self._check_drawdown(result.max_drawdown))

        # 4. Baseline Beat
        if baseline_sharpe is not None:
            checks.criteria.append(
                self._check_baseline_beat(result.sharpe_ratio, baseline_sharpe)
            )

        # 5. Regime Stability
        if result.regime_sharpes:
            checks.criteria.append(
                self._check_regime_stability(result.regime_sharpes)
            )

        # 6. Win Rate
        checks.criteria.append(self._check_win_rate(result.win_rate))

        # 7. Performance Decay (if equity curve available)
        if result.equity_curve is not None and len(result.equity_curve) > 100:
            checks.criteria.append(
                self._check_decay(result.returns["net_return"] if result.returns is not None else None)
            )

        return checks

    def _check_sharpe(self, sharpe: float) -> CriterionResult:
        """Check Sharpe ratio criterion."""
        status = CriterionStatus.PASS if sharpe >= self.min_sharpe else CriterionStatus.FAIL

        if sharpe >= self.min_sharpe:
            message = "Sharpe ratio meets minimum requirement"
        elif sharpe >= self.min_sharpe * 0.8:
            status = CriterionStatus.WARNING
            message = "Sharpe ratio is marginally below threshold"
        else:
            message = "Sharpe ratio is too low for live trading"

        return CriterionResult(
            name=MetricName.SHARPE_NET.value,
            status=status,
            value=sharpe,
            threshold=self.min_sharpe,
            message=message,
        )

    def _check_drawdown(self, max_dd: float) -> CriterionResult:
        """Check max drawdown criterion."""
        abs_dd = abs(max_dd)
        status = CriterionStatus.PASS if abs_dd <= self.max_drawdown else CriterionStatus.FAIL

        if abs_dd <= self.max_drawdown:
            message = "Max drawdown within acceptable limits"
        elif abs_dd <= self.max_drawdown * 1.2:
            status = CriterionStatus.WARNING
            message = "Max drawdown slightly exceeds threshold"
        else:
            message = "Max drawdown too large - risk of ruin"

        return CriterionResult(
            name=MetricName.MAX_DRAWDOWN_ABS.value,
            status=status,
            value=abs_dd,
            threshold=self.max_drawdown,
            message=message,
        )

    def _check_baseline_beat(self, sharpe: float, baseline_sharpe: float) -> CriterionResult:
        """Check if strategy beats baseline."""
        advantage = sharpe - baseline_sharpe
        status = CriterionStatus.PASS if advantage >= self.min_baseline_advantage else CriterionStatus.FAIL

        if advantage >= self.min_baseline_advantage:
            message = f"Beats baseline by {advantage:.3f}"
        elif advantage > 0:
            status = CriterionStatus.WARNING
            message = f"Slightly better than baseline ({advantage:.3f})"
        else:
            message = f"Does not beat baseline (difference: {advantage:.3f})"

        return CriterionResult(
            name=MetricName.BASELINE_SHARPE_DELTA.value,
            status=status,
            value=advantage,
            threshold=self.min_baseline_advantage,
            message=message,
        )

    def _check_regime_stability(self, regime_sharpes: dict[str, float]) -> CriterionResult:
        """Check Sharpe ratio stability across regimes."""
        sharpes = list(regime_sharpes.values())

        if len(sharpes) < 2:
            return CriterionResult(
                name=MetricName.REGIME_SHARPE_CV.value,
                status=CriterionStatus.WARNING,
                value=0.0,
                threshold=self.max_regime_cv,
                message="Insufficient regime data",
            )

        mean_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)

        if mean_sharpe != 0:
            cv = abs(std_sharpe / mean_sharpe)
        else:
            cv = float('inf')

        status = CriterionStatus.PASS if cv <= self.max_regime_cv else CriterionStatus.FAIL

        if cv <= self.max_regime_cv:
            message = "Performance is stable across regimes"
        else:
            # Check if any regime has negative Sharpe
            negative_regimes = [r for r, s in regime_sharpes.items() if s < 0]
            if negative_regimes:
                message = f"Negative performance in: {', '.join(negative_regimes)}"
            else:
                message = "Performance varies too much across regimes"

        return CriterionResult(
            name=MetricName.REGIME_SHARPE_CV.value,
            status=status,
            value=cv,
            threshold=self.max_regime_cv,
            message=message,
        )

    def _check_win_rate(self, win_rate: float) -> CriterionResult:
        """
        Check win rate criterion as an advisory signal.

        Win-rate alone does not determine profitability (payoff asymmetry can dominate),
        so below-threshold values are warnings rather than hard failures.
        """
        status = CriterionStatus.PASS if win_rate >= self.min_win_rate else CriterionStatus.WARNING

        if win_rate >= self.min_win_rate:
            message = "Win rate is acceptable"
        else:
            message = "Win rate below threshold (advisory) - validate payoff ratio and costs"

        return CriterionResult(
            name=MetricName.WIN_RATE.value,
            status=status,
            value=win_rate,
            threshold=self.min_win_rate,
            message=message,
        )

    def _check_profit_factor_net(self, profit_factor_net: float) -> CriterionResult:
        """Check primary net profit factor criterion."""
        status = (
            CriterionStatus.PASS
            if profit_factor_net > self.min_profit_factor_net
            else CriterionStatus.FAIL
        )

        if profit_factor_net > self.min_profit_factor_net:
            message = "Net profit factor indicates positive edge"
        elif profit_factor_net == self.min_profit_factor_net:
            status = CriterionStatus.WARNING
            message = "Net profit factor is exactly at threshold"
        else:
            message = "Net profit factor below threshold"

        return CriterionResult(
            name=MetricName.PROFIT_FACTOR_NET.value,
            status=status,
            value=profit_factor_net,
            threshold=self.min_profit_factor_net,
            message=message,
        )

    def _check_decay(self, returns: pd.Series | None) -> CriterionResult:
        """Check for performance decay over time."""
        if returns is None:
            return CriterionResult(
                name="Performance Decay",
                status=CriterionStatus.WARNING,
                value=0.0,
                threshold=self.min_recent_sharpe_ratio,
                message="No return data for decay check",
            )

        decay_ratio = recent_vs_early_sharpe_ratio(returns, split_fraction=0.25)

        if decay_ratio >= self.min_recent_sharpe_ratio:
            status = CriterionStatus.PASS
            message = "Recent Sharpe is within acceptable range of early Sharpe"
        elif decay_ratio > 0:
            status = CriterionStatus.WARNING
            message = "Some performance decay detected"
        else:
            status = CriterionStatus.FAIL
            message = "Severe performance decay"

        return CriterionResult(
            name=MetricName.RECENT_SHARPE_RATIO.value,
            status=status,
            value=decay_ratio,
            threshold=self.min_recent_sharpe_ratio,
            message=message,
        )


def main():
    """Example usage."""
    from src.backtest.engine import BacktestResult

    # Sample result
    result = BacktestResult(
        sharpe_ratio=0.65,
        max_drawdown=-0.18,
        win_rate=0.52,
        profit_factor_net=1.25,
        profit_factor=1.25,
        regime_sharpes={"trend": 1.1, "normal": 0.5, "chop": 0.3},
    )

    checker = KillCriteria()
    check_result = checker.check(result, baseline_sharpe=0.45)

    print(check_result.summary())


if __name__ == "__main__":
    main()
