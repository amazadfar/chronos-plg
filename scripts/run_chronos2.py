#!/usr/bin/env python
"""
Run Phase 7 Chronos/meta validation with leakage/net-cost gating.

Usage:
    python scripts/run_chronos2.py --data data/processed/btc_4h.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

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
    effective_profit_factor,
    freeze_fold_schedule,
    infer_feature_columns,
    resolve_model_configs,
    write_gate_artifact,
    write_leaderboard_artifacts,
    write_protocol_freeze,
)
from src.evaluation.phase7_chronos import (
    build_phase7_candidate_gate,
    compute_quantile_calibration_by_regime,
    compute_recent_regime_metrics,
    determine_recent_regime_start,
    summarize_chronos_provenance,
)
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.models.chronos2_runner import Chronos2ForReturns
from src.models.meta_model import MetaModel
from src.reporting import build_decision_report, save_decision_artifacts
from src.strategy.signals import QuantileSignalGenerator
from src.utils import finalize_experiment_run, set_global_seed, start_experiment_run


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _build_cost_model(scenario_name: str) -> CostModel:
    scenario = get_scenario_profile(scenario_name)
    return CostModel(
        exchange=scenario.exchange,
        market_type=scenario.market_type,
        order_type=scenario.order_type,
        use_fee_discount=scenario.use_fee_discount,
        apply_funding=scenario.apply_funding,
        apply_margin_interest=scenario.apply_margin_interest,
        margin_interest_rate_per_day=scenario.default_margin_interest_rate_per_day,
        other_cost_bps=scenario.other_cost_bps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 7 Chronos/meta validation")
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
        "--model",
        type=str,
        choices=["chronos2", "meta", "all"],
        default="all",
        help="Candidate model(s) to run once baseline gate passes",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default=DEFAULT_BASELINE_PROTOCOL,
        help="Frozen baseline protocol name for fold/config comparability",
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
        "--chronos-model",
        type=str,
        default="amazon/chronos-t5-base",
        help="Chronos model id",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Chronos context length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Chronos runtime device",
    )
    parser.add_argument(
        "--allow-fallback-candidate",
        action="store_true",
        help="Allow Chronos candidate labeling when empirical fallback backend is active",
    )
    parser.add_argument(
        "--calibration-min-samples-per-regime",
        type=int,
        default=24,
        help="Minimum rows per regime to mark quantile calibration stats as eligible",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Output directory",
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
        help="Disable backtest progress bars",
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
            status = "failed"
            error = f"missing_data_file:{data_path}"
            logger.error("Data file not found: %s", data_path)
            return 1

        data = pd.read_parquet(data_path)
        logger.info("Loaded dataset rows=%s cols=%s", len(data), len(data.columns))
        feature_columns = infer_feature_columns(data)

        protocol = get_baseline_protocol(args.protocol)
        scenario_name = args.scenario or protocol.scenario
        scenario = get_scenario_profile(scenario_name)
        start_date = args.start_date if args.start_date is not None else protocol.start_date

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
        signal_generator = QuantileSignalGenerator(
            entry_policy=args.entry_policy,
            net_edge_cost_multiplier=float(args.net_edge_cost_mult),
            net_edge_risk_multiplier=float(args.net_edge_risk_mult),
            expected_cost_holding_bars=max(1, int(args.expected_cost_holding_bars)),
            expected_cost_round_trip=args.expected_cost_mode == "round_trip",
        )

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
        if not folds:
            status = "failed"
            error = "no_walk_forward_folds"
            logger.error("No walk-forward folds generated")
            return 1

        # 1) Re-run frozen baselines under same folds/cost assumptions.
        baseline_configs = resolve_model_configs(protocol, feature_columns)
        baseline_results: dict[str, BacktestResult] = {}
        for model_name, (model_class, model_kwargs) in baseline_configs.items():
            logger.info("Running baseline gate model: %s", model_name)
            engine = BacktestEngine(
                model_class=model_class,
                model_kwargs=model_kwargs,
                walk_forward_config=protocol.walk_forward_config(),
                cost_model=_build_cost_model(scenario_name),
                signal_generator=signal_generator,
            )
            baseline_results[model_name] = engine.run(
                data=data,
                feature_columns=feature_columns,
                start_date=start_date,
                show_progress=not args.no_progress,
                precomputed_folds=folds,
                collect_fold_metrics=True,
            )

        baseline_leaderboard = build_leaderboard(baseline_results)
        baseline_lb_paths = write_leaderboard_artifacts(
            baseline_leaderboard,
            output_dir=output_dir,
            protocol_name=f"{protocol.name}_phase7_baseline_gate",
        )
        artifacts.extend(str(path) for path in baseline_lb_paths.values())

        baseline_gate = build_chronos_advancement_gate(results=baseline_results, protocol=protocol)
        baseline_gate_path = write_gate_artifact(
            gate_payload=baseline_gate,
            output_dir=output_dir,
            protocol_name=f"{protocol.name}_phase7",
        )
        artifacts.append(str(baseline_gate_path))

        leakage_gate = {
            "passed": bool(protocol.feature_lag_candles >= 1 and len(folds) > 0),
            "feature_lag_candles": protocol.feature_lag_candles,
            "n_folds": len(folds),
            "reason": "feature_lag_and_folds_present"
            if protocol.feature_lag_candles >= 1 and len(folds) > 0
            else "missing_leakage_guards",
        }

        if not leakage_gate["passed"] or not baseline_gate.get("passed", False):
            summary = {
                "status": "skipped_candidate_comparison",
                "protocol": protocol.name,
                "scenario": scenario_name,
                "start_date": start_date,
                "leakage_gate": leakage_gate,
                "baseline_gate": baseline_gate,
                "message": (
                    "Chronos/meta comparison skipped because leakage and net-cost gates did not pass."
                ),
            }
            summary_path = output_dir / "phase7_gate_summary.json"
            _write_json(summary_path, summary)
            artifacts.append(str(summary_path))

            print("\n" + "=" * 80)
            print("PHASE 7 GATE STATUS")
            print("=" * 80)
            print("Leakage gate:", "PASS" if leakage_gate["passed"] else "FAIL")
            print("Baseline net-cost gate:", "PASS" if baseline_gate.get("passed") else "FAIL")
            print("Chronos/meta comparison skipped.")
            return 0

        # 2) Compare Chronos/meta variants only after leakage+net-cost gate pass.
        candidate_configs: dict[str, tuple[type, dict[str, Any]]] = {}
        covariates = feature_columns[:16]
        if args.model in {"chronos2", "all"}:
            candidate_configs["Chronos2"] = (
                Chronos2ForReturns,
                {
                    "model_name": args.chronos_model,
                    "context_length": args.context_length,
                    "device": args.device,
                    "use_covariates": True,
                    "covariate_columns": covariates,
                },
            )
        if args.model in {"meta", "all"}:
            candidate_configs["MetaModel"] = (
                MetaModel,
                {
                    "chronos_model": Chronos2ForReturns(
                        model_name=args.chronos_model,
                        context_length=args.context_length,
                        device=args.device,
                        use_covariates=True,
                        covariate_columns=covariates,
                    ),
                    "feature_columns": feature_columns,
                    "oof_splits": 5,
                    "oof_min_train_samples": max(320, min(800, len(data) // 3)),
                },
            )

        candidate_results: dict[str, BacktestResult] = {}
        candidate_provenance: dict[str, dict[str, Any]] = {}
        candidate_calibration: dict[str, dict[str, Any]] = {}
        for model_name, (model_class, model_kwargs) in candidate_configs.items():
            logger.info("Running candidate model: %s", model_name)
            Chronos2ForReturns.reset_provenance_log()
            engine = BacktestEngine(
                model_class=model_class,
                model_kwargs=model_kwargs,
                walk_forward_config=protocol.walk_forward_config(),
                cost_model=_build_cost_model(scenario_name),
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
            candidate_results[model_name] = result
            provenance_events = Chronos2ForReturns.get_provenance_log()
            candidate_provenance[model_name] = {
                "summary": summarize_chronos_provenance(provenance_events),
                "events": provenance_events,
            }

            if result.positions is not None and not result.positions.empty:
                prediction_frame = result.positions.reindex(
                    columns=["q10", "q50", "q90", "regime"]
                )
                actual_returns = data.get("forward_return", pd.Series(dtype=float))
            else:
                prediction_frame = pd.DataFrame(columns=["q10", "q50", "q90", "regime"])
                actual_returns = pd.Series(dtype=float)
            candidate_calibration[model_name] = compute_quantile_calibration_by_regime(
                predictions=prediction_frame,
                actual_returns=actual_returns,
                min_samples_per_regime=max(1, int(args.calibration_min_samples_per_regime)),
            )

            safe_model = _safe_name(model_name)
            report = BacktestReport(result=result, model_name=model_name, output_dir=output_dir)
            report_paths = report.save_all()
            artifacts.extend(str(path) for path in report_paths.values())

            fold_path = output_dir / f"{safe_model}_fold_metrics.json"
            _write_json(fold_path, {"model": model_name, "fold_metrics": result.fold_metrics})
            artifacts.append(str(fold_path))

            if result.trades is not None:
                trades_csv = output_dir / f"{safe_model}_trade_events.csv"
                result.trades.to_csv(trades_csv, index=True)
                artifacts.append(str(trades_csv))

                trades_parquet = output_dir / f"{safe_model}_trade_events.parquet"
                result.trades.to_parquet(trades_parquet)
                artifacts.append(str(trades_parquet))

            provenance_path = output_dir / f"{safe_model}_phase7_chronos_provenance.json"
            _write_json(
                provenance_path,
                {"model": model_name, **candidate_provenance[model_name]},
            )
            artifacts.append(str(provenance_path))

            calibration_path = output_dir / f"{safe_model}_phase9_quantile_calibration_by_regime.json"
            _write_json(
                calibration_path,
                {
                    "model": model_name,
                    "quantile_calibration_by_regime": candidate_calibration[model_name],
                },
            )
            artifacts.append(str(calibration_path))

        if not candidate_results:
            status = "failed"
            error = "no_candidate_results"
            logger.error("No candidate results were generated")
            return 1

        # 3) Recent-regime stress split and candidate gates.
        recent_start = determine_recent_regime_start(data.index)
        recent_metrics: dict[str, dict[str, Any]] = {
            name: compute_recent_regime_metrics(result, recent_start)
            for name, result in candidate_results.items()
        }
        recent_metrics_path = output_dir / "phase7_recent_regime_metrics.json"
        _write_json(recent_metrics_path, recent_metrics)
        artifacts.append(str(recent_metrics_path))

        provenance_path = output_dir / "phase7_candidate_provenance.json"
        _write_json(provenance_path, candidate_provenance)
        artifacts.append(str(provenance_path))

        calibration_path = output_dir / "phase7_quantile_calibration_by_regime.json"
        _write_json(calibration_path, candidate_calibration)
        artifacts.append(str(calibration_path))

        anchor_name = baseline_gate.get("anchor_model", protocol.baseline_anchor_model)
        if anchor_name not in baseline_results:
            anchor_name = str(baseline_leaderboard.iloc[0]["model"])
        anchor_result = baseline_results[anchor_name]

        candidate_gates: dict[str, dict[str, Any]] = {}
        candidate_decisions: dict[str, dict[str, Any]] = {}
        for name, result in candidate_results.items():
            candidate_gates[name] = build_phase7_candidate_gate(
                candidate_name=name,
                candidate_result=result,
                anchor_name=anchor_name,
                anchor_result=anchor_result,
                recent_metrics=recent_metrics[name],
                chronos_provenance=candidate_provenance.get(name, {}).get("summary"),
                allow_fallback_candidate=bool(args.allow_fallback_candidate),
            )
            decision = build_decision_report(
                model_name=name,
                result=result,
                baseline_sharpe=anchor_result.sharpe_ratio,
            )
            decision_paths = save_decision_artifacts(
                decision,
                output_dir=output_dir,
                prefix=f"{name}_{protocol.name}_phase9",
            )
            artifacts.extend(str(path) for path in decision_paths.values())
            decision_payload = decision.to_dict()
            decision_payload["chronos_provenance"] = candidate_provenance.get(name, {})
            decision_payload["quantile_calibration_by_regime"] = candidate_calibration.get(name, {})
            candidate_decisions[name] = decision_payload
        candidate_gate_path = output_dir / "phase7_candidate_gate.json"
        _write_json(candidate_gate_path, candidate_gates)
        artifacts.append(str(candidate_gate_path))

        candidate_decisions_path = output_dir / "phase9_candidate_decisions.json"
        _write_json(candidate_decisions_path, candidate_decisions)
        artifacts.append(str(candidate_decisions_path))

        # 4) Combined leaderboard across baselines + candidates.
        combined_results = {**baseline_results, **candidate_results}
        combined_leaderboard = build_leaderboard(combined_results)
        combined_lb_paths = write_leaderboard_artifacts(
            combined_leaderboard,
            output_dir=output_dir,
            protocol_name=f"{protocol.name}_phase7_combined",
        )
        artifacts.extend(str(path) for path in combined_lb_paths.values())

        summary_payload = {
            "protocol": protocol.name,
            "scenario": scenario_name,
            "start_date": start_date,
            "leakage_gate": leakage_gate,
            "baseline_gate": baseline_gate,
            "anchor_model": anchor_name,
            "anchor_metrics": {
                "sharpe_ratio": float(anchor_result.sharpe_ratio),
                "profit_factor_net": float(effective_profit_factor(anchor_result)),
            },
            "candidates": {
                name: {
                    "sharpe_ratio": float(result.sharpe_ratio),
                    "profit_factor_net": float(effective_profit_factor(result)),
                    "passed_phase7_gate": bool(candidate_gates[name]["passed"]),
                    "phase9_outcome": candidate_decisions[name]["outcome"],
                    "chronos_fallback_active": bool(
                        candidate_provenance.get(name, {})
                        .get("summary", {})
                        .get("fallback_active", False)
                    ),
                }
                for name, result in candidate_results.items()
            },
            "recent_regime_start": recent_start.isoformat(),
            "allow_fallback_candidate": bool(args.allow_fallback_candidate),
            "reproducible_command": (
                f"python scripts/run_chronos2.py --data {args.data} --protocol {protocol.name}"
            ),
        }
        summary_path = output_dir / "phase7_chronos_summary.json"
        _write_json(summary_path, summary_payload)
        artifacts.append(str(summary_path))

        print("\n" + "=" * 80)
        print("PHASE 7 CHRONOS/META VALIDATION")
        print("=" * 80)
        print("Leakage gate:", "PASS" if leakage_gate["passed"] else "FAIL")
        print("Baseline net-cost gate:", "PASS" if baseline_gate.get("passed") else "FAIL")
        print("\nCandidate gate outcomes:")
        for name, gate in candidate_gates.items():
            print(f" - {name}: {'PASS' if gate.get('passed') else 'FAIL'} ({gate.get('reason')})")
        print("\nPhase 9 decision outcomes:")
        for name, decision_payload in candidate_decisions.items():
            print(f" - {name}: {decision_payload['outcome']} ({decision_payload['reason']})")
        print("\nCombined leaderboard:")
        print(combined_leaderboard.to_string(index=False))
        return 0
    except Exception as exc:
        status = "failed"
        error = str(exc)
        logging.getLogger(__name__).exception("Phase 7 run failed")
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
