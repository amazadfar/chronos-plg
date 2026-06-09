#!/usr/bin/env python
"""
Run complete backtest with all models.

Usage:
    python scripts/run_backtest.py --data data/processed/btc_4h.parquet
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config.scenario_profiles import DEFAULT_SCENARIO, get_scenario_profile
from config.settings import WalkForwardConfig
from src.backtest.costs import CostModel
from src.backtest.engine import BacktestEngine
from src.backtest.report import BacktestReport, compare_models
from src.common.timeframe import (
    SUPPORTED_TIMEFRAMES,
    default_processed_dataset_path,
    normalize_timeframe,
)
from src.data.quality_gate import enforce_degraded_run_gate
from src.models.baselines import EWMABaseline, LightGBMQuantileBaseline, RandomWalkBaseline
from src.reporting import build_decision_report, save_decision_artifacts
from src.strategy.signals import QuantileSignalGenerator
from src.utils import finalize_experiment_run, set_global_seed, start_experiment_run


def main():
    parser = argparse.ArgumentParser(description="Run full backtest")
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
        "--models",
        type=str,
        nargs="+",
        default=["random_walk", "ewma", "lightgbm"],
        choices=["random_walk", "ewma", "lightgbm", "chronos2", "meta"],
        help="Models to backtest"
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
        help="Walk-forward retraining mode"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        help="Execution/cost scenario profile name",
    )
    parser.add_argument(
        "--entry-policy",
        type=str,
        choices=["threshold", "net_edge"],
        default="threshold",
        help="Signal entry policy mode",
    )
    parser.add_argument(
        "--net-edge-cost-mult",
        type=float,
        default=1.0,
        help="Expected-cost multiplier for net-edge policy",
    )
    parser.add_argument(
        "--net-edge-risk-mult",
        type=float,
        default=0.0,
        help="Predicted-risk multiplier for net-edge policy",
    )
    parser.add_argument(
        "--expected-cost-holding-bars",
        type=int,
        default=1,
        help="Expected holding horizon in bars used for cost estimate",
    )
    parser.add_argument(
        "--expected-cost-mode",
        type=str,
        choices=["round_trip", "entry_only"],
        default="round_trip",
        help="Expected-cost estimate mode for net-edge policy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed",
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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
        exclude_patterns = ["forward_", "regime", "hist_q", "timestamp", "close", "open", "high", "low", "volume"]
        feature_columns = [
            c for c in data.columns
            if data[c].dtype in ['float64', 'float32', 'int64', 'int32']
            and not any(p in c for p in exclude_patterns)
        ]
        logger.info(f"Using {len(feature_columns)} features")

        # Configure walk-forward
        wf_config = WalkForwardConfig(mode=args.mode)
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

        # Define model configurations
        model_configs = {
            "random_walk": (RandomWalkBaseline, {"lookback_window": 252}),
            "ewma": (EWMABaseline, {"span": 24}),
            "lightgbm": (LightGBMQuantileBaseline, {
                "n_estimators": 300,
                "early_stopping_rounds": 30,
                "feature_columns": feature_columns,
            }),
        }

        # Optional: Add Chronos models if available and requested
        if "chronos2" in args.models or "meta" in args.models:
            try:
                from src.models.chronos2_runner import Chronos2ForReturns

                if "chronos2" in args.models:
                    model_configs["chronos2"] = (Chronos2ForReturns, {
                        "model_name": "amazon/chronos-t5-base",
                        "context_length": 256,
                        "device": "auto",
                    })

                if "meta" in args.models:
                    # Meta model requires special handling - skip for now in CLI
                    logger.warning("MetaModel requires pre-trained Chronos - skipping")
                    args.models = [m for m in args.models if m != "meta"]
            except ImportError as exc:
                logger.warning(f"Chronos not available: {exc}")
                args.models = [m for m in args.models if m not in ["chronos2", "meta"]]

        # Run backtests
        results = {}
        signal_generator = QuantileSignalGenerator(
            entry_policy=args.entry_policy,
            net_edge_cost_multiplier=float(args.net_edge_cost_mult),
            net_edge_risk_multiplier=float(args.net_edge_risk_mult),
            expected_cost_holding_bars=max(1, int(args.expected_cost_holding_bars)),
            expected_cost_round_trip=args.expected_cost_mode == "round_trip",
        )

        for model_name in args.models:
            if model_name not in model_configs:
                logger.warning(f"Unknown model: {model_name}")
                continue

            model_class, model_kwargs = model_configs[model_name]

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Backtesting: {model_name}")
            logger.info(f"{'=' * 60}")

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
                signal_generator=signal_generator,
            )

            result = engine.run(
                data,
                feature_columns=feature_columns,
                start_date=args.start_date,
            )

            results[model_name] = result

            # Save individual report
            report = BacktestReport(result, model_name=model_name, output_dir=output_dir)
            report_paths = report.save_all()
            artifacts.extend(str(path) for path in report_paths.values())

        if not results:
            status = "failed"
            error = "no_results_generated"
            logger.error("No backtest results were generated")
            return 1

        # Generate comparison
        if len(results) > 1:
            comparison = compare_models(results, output_dir=output_dir)
            print(comparison)
            artifacts.extend(
                [
                    str(output_dir / "model_comparison.txt"),
                    str(output_dir / "model_comparison.json"),
                ]
            )

        # Phase 9 decision artifacts per model.
        for model_name, model_result in results.items():
            baseline_sharpe = None
            if model_name != "lightgbm" and "lightgbm" in results:
                baseline_sharpe = results["lightgbm"].sharpe_ratio
            elif model_name != "random_walk" and "random_walk" in results:
                baseline_sharpe = results["random_walk"].sharpe_ratio

            decision = build_decision_report(
                model_name=model_name,
                result=model_result,
                baseline_sharpe=baseline_sharpe,
            )
            decision_paths = save_decision_artifacts(
                decision,
                output_dir=output_dir,
                prefix=f"{model_name}_phase9",
            )
            artifacts.extend(str(path) for path in decision_paths.values())

        # Final verdict (PF-first ranking).
        print("\n" + "=" * 80)
        print("FINAL VERDICT")
        print("=" * 80)

        def _pf_score(backtest_result):
            return (
                backtest_result.profit_factor_net
                if backtest_result.profit_factor_net > 0
                else backtest_result.profit_factor
            )

        best_model = max(
            results,
            key=lambda name: (
                _pf_score(results[name]) > 1.0,
                _pf_score(results[name]),
                results[name].sharpe_ratio,
                results[name].total_return,
            ),
        )
        best_result = results[best_model]
        best_pf = _pf_score(best_result)
        best_sharpe = best_result.sharpe_ratio

        baseline_for_best = None
        if best_model != "lightgbm" and "lightgbm" in results:
            baseline_for_best = results["lightgbm"].sharpe_ratio
        elif best_model != "random_walk" and "random_walk" in results:
            baseline_for_best = results["random_walk"].sharpe_ratio

        best_decision = build_decision_report(
            model_name=best_model,
            result=best_result,
            baseline_sharpe=baseline_for_best,
        )

        print(f"\nBest Model (PF-first): {best_model}")
        print(f"Profit factor (net): {best_pf:.3f}")
        print(f"Sharpe ratio (net):  {best_sharpe:.3f}")
        print(f"Decision:            {best_decision.outcome.value} ({best_decision.reason})")

        print("\nDecision notes:")
        for note in best_decision.notes:
            print(f" - {note}")

        return 0
    except Exception as exc:
        status = "failed"
        error = str(exc)
        logger.exception("Backtest run failed")
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
