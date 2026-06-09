"""
Robustness summary generator.

Aggregates all robustness checks into a comprehensive report.
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.backtest.engine import BacktestResult
from src.common.metrics import DEFAULT_SUCCESS_THRESHOLDS
from src.robustness.kill_criteria import KillCriteria, KillCriteriaResult
from src.robustness.stress_tests import StressTester, StressTestSuite

logger = logging.getLogger(__name__)


@dataclass
class RobustnessReport:
    """Complete robustness report."""

    model_name: str
    timestamp: str

    # Backtest summary
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0

    # Kill criteria
    kill_criteria: KillCriteriaResult | None = None

    # Stress tests
    stress_tests: StressTestSuite | None = None

    # Overall verdict
    is_viable: bool = False
    verdict: str = ""
    stress_threshold: float = DEFAULT_SUCCESS_THRESHOLDS.min_robustness_pass_rate

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "sharpe_ratio": self.sharpe_ratio,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "kill_criteria_passed": self.kill_criteria.all_passed if self.kill_criteria else None,
            "stress_tests_passed": self.stress_tests.pass_rate if self.stress_tests else None,
            "stress_threshold": self.stress_threshold,
            "stress_suite": self.stress_tests.to_dict() if self.stress_tests else None,
            "is_viable": self.is_viable,
            "verdict": self.verdict,
        }


class RobustnessSummary:
    """
    Generate comprehensive robustness summary.

    Combines:
    - Backtest performance
    - Kill criteria checks
    - Stress test results
    - Final viability verdict
    """

    def __init__(
        self,
        kill_criteria: KillCriteria | None = None,
        stress_tester: StressTester | None = None,
        min_stress_pass_rate: float = DEFAULT_SUCCESS_THRESHOLDS.min_robustness_pass_rate,
        require_stress_suite: bool = True,
    ):
        self.kill_criteria = kill_criteria or KillCriteria()
        self.stress_tester = stress_tester or StressTester()
        self.min_stress_pass_rate = min_stress_pass_rate
        self.require_stress_suite = require_stress_suite

    def generate_report(
        self,
        model_name: str,
        result: BacktestResult,
        baseline_sharpe: float | None = None,
        stress_suite: StressTestSuite | None = None,
    ) -> RobustnessReport:
        """
        Generate robustness report.

        Args:
            model_name: Name of the model
            result: Backtest result
            baseline_sharpe: Sharpe of baseline to beat
            stress_suite: Pre-computed stress test results

        Returns:
            RobustnessReport
        """
        report = RobustnessReport(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            sharpe_ratio=result.sharpe_ratio,
            total_return=result.total_return,
            max_drawdown=result.max_drawdown,
            stress_threshold=self.min_stress_pass_rate,
        )

        # Kill criteria
        report.kill_criteria = self.kill_criteria.check(result, baseline_sharpe)

        # Stress tests (if provided)
        report.stress_tests = stress_suite

        # Determine viability
        kill_ok = report.kill_criteria.all_passed
        if stress_suite is None:
            stress_ok = not self.require_stress_suite
        else:
            stress_ok = stress_suite.pass_rate >= self.min_stress_pass_rate

        report.is_viable = kill_ok and stress_ok

        # Generate verdict
        if report.is_viable:
            if stress_suite and stress_suite.pass_rate >= max(self.min_stress_pass_rate + 0.1, 0.8):
                report.verdict = "STRONGLY VIABLE - Ready for paper trading"
            else:
                report.verdict = "VIABLE - Consider additional testing before live"
        else:
            if stress_suite is None and self.require_stress_suite:
                report.verdict = (
                    "NOT VIABLE - Robustness stress suite missing "
                    f"(required pass rate >= {self.min_stress_pass_rate:.0%})"
                )
            elif not kill_ok:
                failed = [c.name for c in report.kill_criteria.criteria if not c.passed]
                report.verdict = f"NOT VIABLE - Failed: {', '.join(failed)}"
            else:
                report.verdict = (
                    "MARGINAL - Stress tests need attention "
                    f"(pass rate < {self.min_stress_pass_rate:.0%})"
                )

        return report

    def generate_markdown(
        self,
        report: RobustnessReport,
    ) -> str:
        """Generate markdown summary."""
        lines = [
            f"# Robustness Report: {report.model_name}",
            "",
            f"**Generated:** {report.timestamp}",
            "",
            "## Performance Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Sharpe Ratio | {report.sharpe_ratio:.3f} |",
            f"| Total Return | {report.total_return:.2%} |",
            f"| Max Drawdown | {report.max_drawdown:.2%} |",
            "",
        ]

        # Kill Criteria
        if report.kill_criteria:
            status = "✅ PASSED" if report.kill_criteria.all_passed else "❌ FAILED"
            lines.extend([
                f"## Kill Criteria: {status}",
                "",
                "| Criterion | Status | Value | Threshold |",
                "|-----------|--------|-------|-----------|",
            ])

            for c in report.kill_criteria.criteria:
                icon = "✅" if c.passed else "❌"
                lines.append(f"| {c.name} | {icon} | {c.value:.4f} | {c.threshold:.4f} |")

            lines.append("")

        # Stress Tests
        if report.stress_tests:
            rate = report.stress_tests.pass_rate
            status = "✅" if rate >= self.min_stress_pass_rate else "❌"
            lines.extend([
                f"## Stress Tests: {status} ({rate:.0%} passed, threshold {self.min_stress_pass_rate:.0%})",
                "",
                "| Test | Status | Degradation |",
                "|------|--------|-------------|",
            ])

            for t in report.stress_tests.tests:
                icon = "✅" if t.passed else "❌"
                lines.append(f"| {t.name} | {icon} | {t.degradation:.1%} |")

            lines.append("")
        elif self.require_stress_suite:
            lines.extend(
                [
                    "## Stress Tests: ❌ MISSING",
                    "",
                    f"Required pass-rate threshold: {self.min_stress_pass_rate:.0%}",
                    "",
                ]
            )

        # Verdict
        lines.extend([
            "## Final Verdict",
            "",
            f"**{report.verdict}**",
            "",
        ])

        if report.is_viable:
            lines.extend([
                "> [!TIP]",
                "> Strategy is viable for forward testing. Consider:",
                "> - Paper trading for 1-2 months",
                "> - Small position sizes initially",
                "> - Continuous monitoring of kill criteria",
            ])
        else:
            lines.extend([
                "> [!CAUTION]",
                "> Strategy is NOT ready for live trading. Review:",
                "> - Failed kill criteria",
                "> - Stress test results",
                "> - Consider parameter tuning or model changes",
            ])

        return "\n".join(lines)

    def save_report(
        self,
        report: RobustnessReport,
        output_dir: Path,
    ) -> dict[str, Path]:
        """Save report in multiple formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = report.model_name.lower().replace(" ", "_")

        # JSON
        json_path = output_dir / f"{base_name}_robustness.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        # Markdown
        md_path = output_dir / f"{base_name}_robustness.md"
        with open(md_path, "w") as f:
            f.write(self.generate_markdown(report))

        logger.info(f"Saved robustness report to {output_dir}")

        return {"json": json_path, "markdown": md_path}


def main():
    """Example usage."""
    from src.backtest.engine import BacktestResult

    result = BacktestResult(
        sharpe_ratio=0.72,
        total_return=0.35,
        max_drawdown=-0.18,
        win_rate=0.53,
        profit_factor=1.28,
        regime_sharpes={"trend": 1.1, "normal": 0.5, "chop": 0.3},
    )

    summary = RobustnessSummary()
    report = summary.generate_report(
        "Chronos2_Test",
        result,
        baseline_sharpe=0.5,
    )

    print(summary.generate_markdown(report))


if __name__ == "__main__":
    main()
