#!/usr/bin/env python
"""
Comprehensive benchmark script with visualization.

Compares Chronos-2 strategy against baselines:
- With and without strategy layer
- Multiple models
- Full robustness analysis
- Interactive visualizations

Usage:
    python scripts/benchmark.py --data data/processed/btc_4h.parquet
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.scenario_profiles import DEFAULT_SCENARIO, get_scenario_profile
from config.settings import WalkForwardConfig
from src.backtest.costs import CostModel
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.report import BacktestReport
from src.common.timeframe import (
    SUPPORTED_TIMEFRAMES,
    default_processed_dataset_path,
    normalize_timeframe,
)
from src.data.quality_gate import enforce_degraded_run_gate
from src.models.baselines import EWMABaseline, LightGBMQuantileBaseline, RandomWalkBaseline
from src.reporting import build_decision_report, save_decision_artifacts
from src.robustness import RobustnessSummary
from src.utils import finalize_experiment_run, set_global_seed, start_experiment_run

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def run_benchmark(
    data: pd.DataFrame,
    feature_columns: list[str],
    start_date: str,
    mode: str = "weekly",
    output_dir: Path = Path("data/results"),
    include_chronos: bool = False,
    scenario_name: str = DEFAULT_SCENARIO,
) -> tuple[dict[str, BacktestResult], dict[str, BacktestEngine]]:
    """Run benchmark across all models."""
    logger = logging.getLogger(__name__)

    wf_config = WalkForwardConfig(mode=mode)
    scenario = get_scenario_profile(scenario_name)
    results = {}
    engines = {}

    # Define models
    models = {
        "RandomWalk": (RandomWalkBaseline, {"lookback_window": 252}),
        "EWMA": (EWMABaseline, {"span": 24}),
        "LightGBM": (LightGBMQuantileBaseline, {
            "n_estimators": 300,
            "early_stopping_rounds": 30,
            "feature_columns": feature_columns,
        }),
    }

    # Add Chronos-2 if requested
    if include_chronos:
        try:
            from src.models.chronos2_runner import Chronos2ForReturns
            models["Chronos2"] = (Chronos2ForReturns, {
                "model_name": "amazon/chronos-t5-base",
                "context_length": 256,
                "device": "auto",
            })
        except ImportError:
            logger.warning("Chronos-2 not available")

    # Run each model
    for name, (model_class, model_kwargs) in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {name}")
        logger.info(f"{'='*60}")

        engine = BacktestEngine(
            model_class=model_class,
            model_kwargs=model_kwargs,
            walk_forward_config=wf_config,
            cost_model=CostModel(
                exchange=scenario.exchange,
                market_type=scenario.market_type,
                order_type=scenario.order_type,
                use_fee_discount=scenario.use_fee_discount,
                apply_funding=scenario.apply_funding,
                apply_margin_interest=scenario.apply_margin_interest,
                margin_interest_rate_per_day=scenario.default_margin_interest_rate_per_day,
                other_cost_bps=scenario.other_cost_bps,
            ),
        )

        try:
            result = engine.run(
                data,
                feature_columns=feature_columns,
                start_date=start_date,
            )
            results[name] = result
            engines[name] = engine

            # Save individual report
            report = BacktestReport(result, model_name=name, output_dir=output_dir)
            report.save_all()
        except Exception as e:
            logger.error(f"Failed to backtest {name}: {e}")

    return results, engines


def run_robustness_analysis(
    results: dict[str, BacktestResult],
    engines: dict[str, BacktestEngine],
    data: pd.DataFrame,
    feature_columns: list[str],
    start_date: str | None,
    output_dir: Path,
) -> tuple[dict[str, dict], list[Path]]:
    """Run robustness analysis on all models."""
    logger = logging.getLogger(__name__)

    # Get baseline sharpe for comparison
    baseline_sharpe = results.get("LightGBM", results.get("RandomWalk", BacktestResult())).sharpe_ratio

    robustness_results = {}
    decision_paths: list[Path] = []
    summary = RobustnessSummary()

    for name, result in results.items():
        logger.info(f"Running robustness analysis for {name}...")
        stress_suite = None
        if name in engines:
            try:
                stress_suite = summary.stress_tester.run_all(
                    base_result=result,
                    engine=engines[name],
                    data=data,
                    feature_columns=feature_columns,
                    start_date=start_date,
                )
            except Exception as exc:
                logger.warning(f"Stress suite failed for {name}: {exc}")

        report = summary.generate_report(
            name,
            result,
            baseline_sharpe=baseline_sharpe if name != "LightGBM" else None,
            stress_suite=stress_suite,
        )

        robustness_results[name] = report
        summary.save_report(report, output_dir)

        decision = build_decision_report(
            model_name=name,
            result=result,
            baseline_sharpe=baseline_sharpe if name != "LightGBM" else None,
            stress_suite=stress_suite,
        )
        paths = save_decision_artifacts(
            decision,
            output_dir=output_dir,
            prefix=f"{name}_benchmark_phase9",
        )
        decision_paths.extend(paths.values())

    return robustness_results, decision_paths


def create_equity_curves_plot(
    results: dict[str, BacktestResult],
    output_dir: Path,
) -> Path:
    """Create equity curves comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Equity Curves
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, result), color in zip(results.items(), colors):
        if result.equity_curve is not None:
            ax1.plot(result.equity_curve.index, result.equity_curve.values,
                    label=f"{name} ({result.sharpe_ratio:.2f})", color=color, linewidth=2)

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title("Equity Curves", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Equity (normalized)")
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Sharpe Comparison
    ax2 = axes[0, 1]
    names = list(results.keys())
    sharpes = [r.sharpe_ratio for r in results.values()]
    colors = ['green' if s > 0.5 else 'orange' if s > 0 else 'red' for s in sharpes]

    bars = ax2.bar(names, sharpes, color=colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0.5, color='green', linestyle='--', label='Target (0.5)', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_title("Sharpe Ratio Comparison", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Sharpe Ratio")
    ax2.legend()

    # Add value labels
    for bar, sharpe in zip(bars, sharpes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{sharpe:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. Drawdown Comparison
    ax3 = axes[1, 0]
    for (name, result), color in zip(results.items(), plt.cm.tab10(np.linspace(0, 1, len(results)))):
        if result.equity_curve is not None:
            peak = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - peak) / peak * 100
            ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=name, color=color)
            ax3.plot(drawdown.index, drawdown.values, color=color, linewidth=1)

    ax3.axhline(y=-30, color='red', linestyle='--', label='Max Allowed (-30%)', alpha=0.7)
    ax3.set_title("Drawdown", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Drawdown (%)")
    ax3.legend(loc='lower left')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Kill Criteria Summary
    ax4 = axes[1, 1]

    criteria_names = ['Sharpe > 0.5', 'MaxDD < 30%', 'WinRate > 45%', 'PF(Net) > 1.0']
    model_names = list(results.keys())

    # Create matrix
    matrix = np.zeros((len(model_names), len(criteria_names)))
    for i, name in enumerate(model_names):
        r = results[name]
        matrix[i, 0] = 1 if r.sharpe_ratio > 0.5 else 0
        matrix[i, 1] = 1 if abs(r.max_drawdown) < 0.3 else 0
        matrix[i, 2] = 1 if r.win_rate > 0.45 else 0
        pf_net = r.profit_factor_net if r.profit_factor_net > 0 else r.profit_factor
        matrix[i, 3] = 1 if pf_net > 1.0 else 0

    ax4.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax4.set_xticks(np.arange(len(criteria_names)))
    ax4.set_yticks(np.arange(len(model_names)))
    ax4.set_xticklabels(criteria_names)
    ax4.set_yticklabels(model_names)

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(criteria_names)):
            text = "✓" if matrix[i, j] == 1 else "✗"
            color = "white" if matrix[i, j] == 1 else "black"
            ax4.text(j, i, text, ha="center", va="center", color=color, fontsize=16, fontweight='bold')

    ax4.set_title("Kill Criteria Status", fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / "benchmark_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_regime_analysis_plot(
    results: dict[str, BacktestResult],
    output_dir: Path,
) -> Path:
    """Create regime performance analysis plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect regime data
    all_regimes = set()
    for result in results.values():
        if result.regime_sharpes:
            all_regimes.update(result.regime_sharpes.keys())

    all_regimes = sorted(all_regimes)
    model_names = list(results.keys())

    # 1. Regime Sharpe Bar Chart
    ax1 = axes[0]
    x = np.arange(len(all_regimes))
    width = 0.8 / len(model_names)

    for i, name in enumerate(model_names):
        result = results[name]
        if result.regime_sharpes:
            sharpes = [result.regime_sharpes.get(r, 0) for r in all_regimes]
            offset = (i - len(model_names)/2 + 0.5) * width
            ax1.bar(x + offset, sharpes, width, label=name, alpha=0.8)

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_xlabel("Regime")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Sharpe by Regime", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_regimes)
    ax1.legend()

    # 2. Regime Return Distribution
    ax2 = axes[1]

    for i, name in enumerate(model_names):
        result = results[name]
        if result.regime_returns:
            regimes = list(result.regime_returns.keys())
            returns = list(result.regime_returns.values())

            # Normalize for comparison
            total = sum(abs(r) for r in returns) or 1
            normalized = [r / total * 100 for r in returns]

            ax2.barh(name, sum(normalized), alpha=0.3)

            # Stacked by regime
            left = 0
            for j, (regime, ret) in enumerate(zip(regimes, normalized)):
                if ret > 0:
                    ax2.barh(name, ret, left=left, label=regime if i == 0 else "", alpha=0.8)
                    left += ret

    ax2.set_xlabel("Return Contribution (%)")
    ax2.set_title("Return Attribution by Regime", fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')

    plt.tight_layout()

    output_path = output_dir / "regime_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def generate_summary_report(
    results: dict[str, BacktestResult],
    output_dir: Path,
) -> str:
    """Generate final summary report."""

    # Find best model
    sharpes = {name: result.sharpe_ratio for name, result in results.items()}
    best_model = max(sharpes, key=sharpes.get)
    best_result = results[best_model]

    # Check against LightGBM baseline
    lgb_sharpe = sharpes.get("LightGBM", 0)
    beats_baseline = sharpes[best_model] > lgb_sharpe + 0.1 if best_model != "LightGBM" else False

    report = f"""
# Chronos-2 Strategy Benchmark Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

| Model | Sharpe | Return | Max DD | Win Rate |
|-------|--------|--------|--------|----------|
"""

    for name, result in sorted(results.items(), key=lambda x: -x[1].sharpe_ratio):
        winner = " 🏆" if name == best_model else ""
        report += f"| {name}{winner} | {result.sharpe_ratio:.3f} | {result.total_return:.1%} | {result.max_drawdown:.1%} | {result.win_rate:.1%} |\n"

    report += f"""
## Verdict

**Best Model:** {best_model} (Sharpe: {best_result.sharpe_ratio:.3f})

### Success Criteria Check

| Criterion | Status | Value |
|-----------|--------|-------|
| Sharpe > 0.5 (net of costs) | {'✅' if best_result.sharpe_ratio > 0.5 else '❌'} | {best_result.sharpe_ratio:.3f} |
| Beats LightGBM baseline | {'✅' if beats_baseline else '❌'} | Δ = {sharpes[best_model] - lgb_sharpe:.3f} |
| Max Drawdown < 25% | {'✅' if abs(best_result.max_drawdown) < 0.25 else '❌'} | {best_result.max_drawdown:.1%} |
| Win Rate > 45% | {'✅' if best_result.win_rate > 0.45 else '❌'} | {best_result.win_rate:.1%} |
| Profit Factor (Net) > 1.0 | {'✅' if (best_result.profit_factor_net if best_result.profit_factor_net > 0 else best_result.profit_factor) > 1.0 else '❌'} | {(best_result.profit_factor_net if best_result.profit_factor_net > 0 else best_result.profit_factor):.2f} |

"""

    # Final recommendation
    is_viable = (
        best_result.sharpe_ratio > 0.5 and
        abs(best_result.max_drawdown) < 0.30 and
        (best_result.profit_factor_net if best_result.profit_factor_net > 0 else best_result.profit_factor) > 1.0
    )

    if is_viable and beats_baseline:
        report += """
> [!TIP]
> **RECOMMENDATION: PROCEED TO PAPER TRADING**
>
> The strategy meets all success criteria and demonstrates a significant edge
> over baselines. Recommended next steps:
> 1. Paper trade for 4-8 weeks
> 2. Monitor kill criteria in real-time
> 3. Start with small position sizes (0.25x max leverage)
"""
    elif is_viable:
        report += """
> [!WARNING]
> **RECOMMENDATION: MARGINAL - MORE TESTING NEEDED**
>
> The strategy is viable but does not significantly beat baselines.
> Consider:
> 1. Additional feature engineering
> 2. Alternative model configurations
> 3. More out-of-sample testing
"""
    else:
        report += """
> [!CAUTION]
> **RECOMMENDATION: DO NOT TRADE LIVE**
>
> The strategy fails one or more kill criteria.
> Required actions:
> 1. Review failed criteria
> 2. Analyze regime performance
> 3. Consider alternative approaches
"""

    # Save report
    report_path = output_dir / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive benchmark")
    parser.add_argument(
        "--timeframe",
        type=str,
        default="4h",
        choices=SUPPORTED_TIMEFRAMES,
        help="Dataset timeframe used when --data is omitted",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to processed dataset (default: data/processed/btc_<timeframe>.parquet)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["weekly", "monthly"],
        default="weekly",
        help="Walk-forward mode"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        help="Execution/cost scenario profile name",
    )
    parser.add_argument(
        "--include-chronos",
        action="store_true",
        help="Include Chronos-2 model (requires GPU)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    output_dir = Path(args.output_dir)
    set_global_seed(args.seed)
    run_id, manifest_path = start_experiment_run(
        script_name=Path(__file__).name,
        args=vars(args),
        seed=args.seed,
        output_dir=output_dir,
        project_root=Path(__file__).parent.parent,
    )
    status = "success"
    error: str | None = None
    artifacts: list[str] = []

    try:
        # Load data
        timeframe = normalize_timeframe(args.timeframe)
        data_path = (
            Path(args.data)
            if args.data
            else default_processed_dataset_path(timeframe=timeframe)
        )
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            logger.info("Run 'python scripts/download_data.py --build-dataset' first")
            status = "failed"
            error = f"missing_data_file:{data_path}"
            return 1

        logger.info(f"Loading data from {data_path}")
        data = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(data)} rows")

        # Get feature columns
        exclude = ["forward_", "regime", "hist_q", "timestamp", "close", "open", "high", "low", "volume"]
        feature_columns = [
            c for c in data.columns
            if data[c].dtype in ['float64', 'float32', 'int64', 'int32']
            and not any(p in c for p in exclude)
        ]

        output_dir.mkdir(parents=True, exist_ok=True)

        scenario = get_scenario_profile(args.scenario)
        data_quality_gate_path = output_dir / "phase11_data_quality_gate.json"
        gate = enforce_degraded_run_gate(
            data_path,
            market_type=scenario.market_type,
            artifact_path=data_quality_gate_path,
        )
        artifacts.append(str(data_quality_gate_path))
        logger.info(
            "Data quality gate passed (market=%s availability=%s)",
            gate.market_type,
            gate.availability,
        )

        # Run benchmark
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING BENCHMARK")
        logger.info("=" * 70)

        results, engines = run_benchmark(
            data,
            feature_columns,
            args.start_date,
            args.mode,
            output_dir,
            args.include_chronos,
            args.scenario,
        )

        # Run robustness analysis
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING ROBUSTNESS ANALYSIS")
        logger.info("=" * 70)

        _, robustness_decision_paths = run_robustness_analysis(
            results=results,
            engines=engines,
            data=data,
            feature_columns=feature_columns,
            start_date=args.start_date,
            output_dir=output_dir,
        )
        artifacts.extend(str(path) for path in robustness_decision_paths)

        # Generate visualizations
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 70)

        try:
            viz_path = create_equity_curves_plot(results, output_dir)
            logger.info(f"Saved: {viz_path}")
            artifacts.append(str(viz_path))

            regime_path = create_regime_analysis_plot(results, output_dir)
            logger.info(f"Saved: {regime_path}")
            artifacts.append(str(regime_path))
        except Exception as exc:
            logger.warning(f"Visualization failed: {exc}")

        # Generate summary report
        report = generate_summary_report(results, output_dir)
        print(report)
        artifacts.append(str(output_dir / "benchmark_report.md"))

        logger.info(f"\nAll results saved to: {output_dir}")
        return 0
    except Exception as exc:
        status = "failed"
        error = str(exc)
        logger.exception("Benchmark run failed")
        return 1
    finally:
        finalize_experiment_run(
            manifest_path=manifest_path,
            run_id=run_id,
            status=status,
            artifacts=artifacts,
            notes={"error": error} if error else None,
        )


if __name__ == "__main__":
    sys.exit(main())
