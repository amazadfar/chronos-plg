#!/usr/bin/env python
"""
Run Phase 6 baseline protocol with frozen folds and net-cost engine.

Usage:
    python scripts/run_baselines.py --data data/processed/btc_4h.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.baseline_protocols import DEFAULT_BASELINE_PROTOCOL, get_baseline_protocol
from config.scenario_profiles import get_scenario_profile
from src.common.timeframe import (
    SUPPORTED_TIMEFRAMES,
    default_processed_dataset_path,
    normalize_timeframe,
)
from src.backtest.costs import CostModel
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.report import BacktestReport
from src.data.quality_gate import enforce_degraded_run_gate
from src.evaluation.phase6_baselines import (
    build_chronos_advancement_gate,
    build_leaderboard,
    freeze_fold_schedule,
    infer_feature_columns,
    resolve_model_configs,
    write_gate_artifact,
    write_leaderboard_artifacts,
    write_protocol_freeze,
)
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.reporting import build_decision_report, save_decision_artifacts
from src.strategy.signals import QuantileSignalGenerator
from src.utils import finalize_experiment_run, set_global_seed, start_experiment_run


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 baseline protocol")
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
        help="Path to processed dataset (default: data/processed/btc_<timeframe>.parquet)",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default=DEFAULT_BASELINE_PROTOCOL,
        help="Frozen baseline protocol name",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional scenario override; defaults to protocol scenario",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start-date override; defaults to protocol start_date",
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
        help="Output directory for reports/artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-model progress bars",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        timeframe = normalize_timeframe(args.timeframe)
        data_path = (
            Path(args.data)
            if args.data
            else default_processed_dataset_path(timeframe=timeframe)
        )
        if not data_path.exists():
            logger.error("Data file not found: %s", data_path)
            logger.info("Run `python scripts/download_data.py --build-dataset` first.")
            status = "failed"
            error = f"missing_data_file:{data_path}"
            return 1

        data = pd.read_parquet(data_path)
        logger.info("Loaded dataset: rows=%s cols=%s", len(data), len(data.columns))

        protocol = get_baseline_protocol(args.protocol)
        scenario_name = args.scenario or protocol.scenario
        scenario = get_scenario_profile(scenario_name)
        start_date = args.start_date if args.start_date is not None else protocol.start_date
        feature_columns = infer_feature_columns(data)
        logger.info("Using protocol=%s scenario=%s start_date=%s", protocol.name, scenario_name, start_date)
        logger.info("Feature columns inferred: %s", len(feature_columns))

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

        protocol_path = write_protocol_freeze(protocol, output_dir)
        artifacts.append(str(protocol_path))

        evaluator = WalkForwardEvaluator(
            config=protocol.walk_forward_config(),
            feature_lag_candles=protocol.feature_lag_candles,
        )
        folds, fold_path = freeze_fold_schedule(
            evaluator=evaluator,
            data=data,
            protocol=protocol,
            start_date=start_date,
            output_dir=output_dir,
        )
        artifacts.append(str(fold_path))
        logger.info("Frozen fold schedule saved (%s folds)", len(folds))

        model_configs = resolve_model_configs(protocol, feature_columns)
        results: dict[str, BacktestResult] = {}
        signal_generator = QuantileSignalGenerator(
            entry_policy=args.entry_policy,
            net_edge_cost_multiplier=float(args.net_edge_cost_mult),
            net_edge_risk_multiplier=float(args.net_edge_risk_mult),
            expected_cost_holding_bars=max(1, int(args.expected_cost_holding_bars)),
            expected_cost_round_trip=args.expected_cost_mode == "round_trip",
        )

        for model_name, (model_class, model_kwargs) in model_configs.items():
            logger.info("Backtesting baseline model: %s", model_name)
            engine = BacktestEngine(
                model_class=model_class,
                model_kwargs=model_kwargs,
                walk_forward_config=protocol.walk_forward_config(),
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
                data=data,
                feature_columns=feature_columns,
                start_date=start_date,
                show_progress=not args.no_progress,
                precomputed_folds=folds,
                collect_fold_metrics=True,
            )
            results[model_name] = result

            safe_model = _safe_name(model_name)
            report = BacktestReport(result=result, model_name=model_name, output_dir=output_dir)
            report_paths = report.save_all()
            artifacts.extend(str(path) for path in report_paths.values())

            fold_metrics_path = output_dir / f"{safe_model}_fold_metrics.json"
            _write_json(fold_metrics_path, {"model": model_name, "fold_metrics": result.fold_metrics})
            artifacts.append(str(fold_metrics_path))

            if result.trades is not None:
                trades_csv = output_dir / f"{safe_model}_trade_events.csv"
                result.trades.to_csv(trades_csv, index=True)
                artifacts.append(str(trades_csv))

                trades_parquet = output_dir / f"{safe_model}_trade_events.parquet"
                result.trades.to_parquet(trades_parquet)
                artifacts.append(str(trades_parquet))

        if not results:
            status = "failed"
            error = "no_baseline_results"
            logger.error("No baseline results generated.")
            return 1

        leaderboard = build_leaderboard(results)
        leaderboard_paths = write_leaderboard_artifacts(
            leaderboard=leaderboard,
            output_dir=output_dir,
            protocol_name=protocol.name,
        )
        artifacts.extend(str(path) for path in leaderboard_paths.values())

        gate_payload = build_chronos_advancement_gate(results=results, protocol=protocol)
        gate_path = write_gate_artifact(
            gate_payload=gate_payload,
            output_dir=output_dir,
            protocol_name=protocol.name,
        )
        artifacts.append(str(gate_path))

        baseline_decisions: dict[str, dict] = {}
        anchor_name = (
            protocol.baseline_anchor_model
            if protocol.baseline_anchor_model in results
            else str(leaderboard.iloc[0]["model"])
        )
        anchor_sharpe = results[anchor_name].sharpe_ratio

        for model_name, model_result in results.items():
            baseline_sharpe = anchor_sharpe if model_name != anchor_name else None
            decision = build_decision_report(
                model_name=model_name,
                result=model_result,
                baseline_sharpe=baseline_sharpe,
            )
            decision_paths = save_decision_artifacts(
                decision,
                output_dir=output_dir,
                prefix=f"{model_name}_{protocol.name}_phase9",
            )
            artifacts.extend(str(path) for path in decision_paths.values())
            baseline_decisions[model_name] = decision.to_dict()

        decision_summary_path = output_dir / f"baseline_phase9_decisions_{protocol.name}.json"
        _write_json(decision_summary_path, baseline_decisions)
        artifacts.append(str(decision_summary_path))

        summary_payload = {
            "protocol": protocol.name,
            "scenario": scenario_name,
            "start_date": start_date,
            "n_models": len(results),
            "n_folds": len(folds),
            "leaderboard_top": leaderboard.iloc[0].to_dict() if not leaderboard.empty else None,
            "chronos_gate": gate_payload,
            "decision_outcomes": {
                name: payload["outcome"]
                for name, payload in baseline_decisions.items()
            },
            "reproducible_command": (
                f"python scripts/run_baselines.py --data {args.data} --protocol {protocol.name}"
            ),
        }
        summary_path = output_dir / f"baseline_phase6_summary_{protocol.name}.json"
        _write_json(summary_path, summary_payload)
        artifacts.append(str(summary_path))

        print("\n" + "=" * 80)
        print("PHASE 6 BASELINE LEADERBOARD")
        print("=" * 80)
        print(leaderboard.to_string(index=False))
        print("\nChronos advancement gate:", "PASS" if gate_payload.get("passed") else "FAIL")
        print("Gate artifact:", gate_path)
        print("\nPhase 9 decision outcomes:")
        for model_name, decision_payload in baseline_decisions.items():
            print(f" - {model_name}: {decision_payload['outcome']} ({decision_payload['reason']})")
        return 0
    except Exception as exc:
        status = "failed"
        error = str(exc)
        logger.exception("Phase 6 baseline run failed")
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
