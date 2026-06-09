import json
import logging
from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestResult
from src.common.metrics import DEFAULT_SUCCESS_THRESHOLDS, MetricName
from src.robustness.kill_criteria import CriterionStatus, KillCriteria

logger = logging.getLogger(__name__)


class BacktestReport:
    """
    Generate and save backtest reports.
    """

    def __init__(
        self,
        result: BacktestResult,
        model_name: str = "Unknown",
        output_dir: Path | None = None,
        baseline_sharpe: float | None = None,
    ):
        """
        Args:
            result: Backtest result
            model_name: Name of the model
            output_dir: Output directory for reports
            baseline_sharpe: Optional baseline Sharpe for kill-criteria delta checks
        """
        self.result = result
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else Path("data/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_sharpe = baseline_sharpe
        self._kill_checker = KillCriteria()

    def generate_summary(self) -> str:
        """Generate text summary."""
        return f"""
{'='*70}
BACKTEST REPORT: {self.model_name}
{'='*70}

PERFORMANCE SUMMARY
-------------------
Total Return:        {self.result.total_return:>10.2%}
Annualized Return:   {self.result.annualized_return:>10.2%}
Sharpe Ratio:        {self.result.sharpe_ratio:>10.3f}
Sortino Ratio:       {self.result.sortino_ratio:>10.3f}
Max Drawdown:        {self.result.max_drawdown:>10.2%}

TRADING STATISTICS
------------------
Number of Trades:    {self.result.num_trades:>10}
Win Rate:            {self.result.win_rate:>10.1%}
Profit Factor:       {(self.result.profit_factor_net if self.result.profit_factor_net > 0 else self.result.profit_factor):>10.2f}

COST ANALYSIS
-------------
Total Fees:          {self.result.total_fees:>10.4f}
Total Slippage:      {self.result.total_slippage:>10.4f}
Total Funding:       {self.result.total_funding:>10.4f}
Total Interest:      {self.result.total_interest:>10.4f}
Total Other Costs:   {self.result.total_other_costs:>10.4f}
Total Costs:         {self.result.total_costs:>10.4f}
Cost % of Return:    {(self.result.total_costs / max(self.result.total_return, 0.0001) * 100):>10.1f}%

REGIME ANALYSIS
---------------
{self._format_regime_analysis()}

KILL CRITERIA CHECK
-------------------
{self._check_kill_criteria()}
"""

    def _format_regime_analysis(self) -> str:
        """Format regime breakdown."""
        if not self.result.regime_sharpes:
            return "No regime data available"

        lines = []
        for regime in sorted(self.result.regime_sharpes.keys()):
            sharpe = self.result.regime_sharpes[regime]
            ret = self.result.regime_returns.get(regime, 0)
            lines.append(f"{regime:>10}: Sharpe = {sharpe:>6.3f}, Return = {ret:>8.2%}")

        return "\n".join(lines)

    def _check_kill_criteria(self) -> str:
        """Check against shared kill-criteria implementation."""
        result = self._kill_checker.check(self.result, baseline_sharpe=self.baseline_sharpe)

        ordered = sorted(
            result.criteria,
            key=lambda criterion: (
                0 if criterion.name == MetricName.PROFIT_FACTOR_NET.value else 1
            ),
        )
        lines: list[str] = []
        for criterion in ordered:
            if criterion.status == CriterionStatus.PASS:
                icon = "PASS"
            elif criterion.status == CriterionStatus.WARNING:
                icon = "WARN"
            else:
                icon = "FAIL"
            lines.append(
                f"[{icon}] {criterion.name}: "
                f"value={criterion.value:.4f} threshold={criterion.threshold:.4f} "
                f"({criterion.message})"
            )

        if result.all_passed:
            lines.append("OVERALL: PASS")
        else:
            lines.append("OVERALL: FAIL")
        return "\n".join(lines)

    def save_json(self, filename: str | None = None) -> Path:
        """Save results as JSON."""
        filename = filename or f"{self.model_name.lower().replace(' ', '_')}_results.json"
        path = self.output_dir / filename

        with open(path, "w") as f:
            json.dump(self.result.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved JSON report to {path}")
        return path

    def save_csv(self, filename: str | None = None) -> Path:
        """Save equity curve and returns as CSV."""
        filename = filename or f"{self.model_name.lower().replace(' ', '_')}_equity.csv"
        path = self.output_dir / filename

        if self.result.equity_curve is not None:
            df = pd.DataFrame({
                "equity": self.result.equity_curve,
            })

            if self.result.returns is not None:
                df = df.join(self.result.returns)

            df.to_csv(path)
            logger.info(f"Saved CSV report to {path}")

        return path

    def save_report(self, filename: str | None = None) -> Path:
        """Save text report."""
        filename = filename or f"{self.model_name.lower().replace(' ', '_')}_report.txt"
        path = self.output_dir / filename

        with open(path, "w") as f:
            f.write(self.generate_summary())

        logger.info(f"Saved text report to {path}")
        return path

    def save_all(self) -> dict[str, Path]:
        """Save all report formats."""
        return {
            "json": self.save_json(),
            "csv": self.save_csv(),
            "txt": self.save_report(),
        }


def compare_models(
    results: dict[str, BacktestResult],
    output_dir: Path | None = None,
) -> str:
    """
    Generate comparison report for multiple models.

    Args:
        results: Dict of {model_name: BacktestResult}
        output_dir: Output directory

    Returns:
        Comparison summary string
    """
    output_dir = Path(output_dir) if output_dir else Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison table
    rows = []
    for name, result in results.items():
        rows.append({
            "Model": name,
            "Return": f"{result.total_return:.2%}",
            "Sharpe": f"{result.sharpe_ratio:.3f}",
            "Sortino": f"{result.sortino_ratio:.3f}",
            "MaxDD": f"{result.max_drawdown:.2%}",
            "Costs": f"{result.total_costs:.4f}",
            "Trades": result.num_trades,
            "WinRate": f"{result.win_rate:.1%}",
            "PF(Net)": f"{(result.profit_factor_net if result.profit_factor_net > 0 else result.profit_factor):.2f}",
        })

    df = pd.DataFrame(rows)

    def _pf(backtest_result: BacktestResult) -> float:
        return (
            backtest_result.profit_factor_net
            if backtest_result.profit_factor_net > 0
            else backtest_result.profit_factor
        )

    winner = max(
        results,
        key=lambda name: (
            _pf(results[name]) > DEFAULT_SUCCESS_THRESHOLDS.min_profit_factor_net,
            _pf(results[name]),
            results[name].sharpe_ratio,
            results[name].total_return,
        ),
    )
    winner_sharpe = results[winner].sharpe_ratio
    winner_pf = _pf(results[winner])

    summary = f"""
{'='*80}
MODEL COMPARISON
{'='*80}

{df.to_string(index=False)}

WINNER (PF-first): {winner}

SUCCESS CRITERIA:
- Profit factor (net) > 1.0: {"✅" if winner_pf > 1.0 else "❌"} ({winner_pf:.3f})
- Sharpe > {DEFAULT_SUCCESS_THRESHOLDS.min_sharpe_net:.1f} (net of costs): {"✅" if winner_sharpe > DEFAULT_SUCCESS_THRESHOLDS.min_sharpe_net else "❌"} ({winner_sharpe:.3f})
- Beats all baselines: {"✅" if winner != "RandomWalk" else "❌"}
"""

    # Save comparison
    comparison_path = output_dir / "model_comparison.txt"
    with open(comparison_path, "w") as f:
        f.write(summary)

    comparison_json = output_dir / "model_comparison.json"
    with open(comparison_json, "w") as f:
        json.dump({
            name: result.to_dict() for name, result in results.items()
        }, f, indent=2, default=str)

    return summary


def main():
    """Example usage."""
    from src.backtest.engine import BacktestResult

    # Sample result
    result = BacktestResult(
        total_return=0.45,
        annualized_return=0.25,
        sharpe_ratio=1.2,
        sortino_ratio=1.8,
        max_drawdown=-0.15,
        total_fees=0.02,
        total_slippage=0.015,
        total_costs=0.035,
        num_trades=150,
        win_rate=0.55,
        profit_factor=1.4,
        regime_returns={"trend": 0.30, "normal": 0.10, "chop": 0.05},
        regime_sharpes={"trend": 1.8, "normal": 0.9, "chop": 0.4},
    )

    report = BacktestReport(result, model_name="Chronos2_Test")
    print(report.generate_summary())


if __name__ == "__main__":
    main()
