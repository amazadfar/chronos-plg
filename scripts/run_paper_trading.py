#!/usr/bin/env python
"""Run Phase 10 paper-trading replay with monitoring and deployment policies."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.scenario_profiles import DEFAULT_SCENARIO, get_scenario_profile
from src.backtest.costs import CostModel
from src.common.timeframe import (
    SUPPORTED_TIMEFRAMES,
    default_processed_dataset_path,
    normalize_timeframe,
)
from src.data.quality_gate import enforce_degraded_run_gate
from src.models.baselines import EWMABaseline, LightGBMQuantileBaseline, RandomWalkBaseline
from src.paper_trading import (
    PaperTradingConfig,
    PaperTradingEngine,
    KillSwitchThresholds,
    build_kill_event_taxonomy,
    build_low_activity_diagnostics,
    build_daily_weekly_dashboards,
    default_capital_ramp_policy,
    evaluate_deployment_readiness,
    evaluate_kill_switch,
    recommend_capital_action,
    render_capital_ramp_policy,
    serialize_kill_events,
)
from src.reporting import build_decision_report, save_decision_artifacts
from src.strategy.position_sizing import PositionSizer
from src.strategy.signals import QuantileSignalGenerator
from src.utils import finalize_experiment_run, set_global_seed, start_experiment_run


def _select_feature_columns(data: pd.DataFrame) -> list[str]:
    exclude_patterns = ("forward_", "regime", "hist_q", "timestamp", "open", "high", "low", "close", "volume")
    columns: list[str] = []
    for column in data.columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            continue
        if any(pattern in column for pattern in exclude_patterns):
            continue
        if data[column].notna().sum() == 0:
            continue
        columns.append(column)
    if not columns:
        raise ValueError("No numeric feature columns found for paper-trading run")
    return columns


def _write_json(path: Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-trading replay")
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
        help="Input dataset (default: data/processed/btc_<timeframe>.parquet)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_walk",
        choices=["random_walk", "ewma", "lightgbm"],
        help="Model to run in paper-trading replay",
    )
    parser.add_argument("--ewma-span", type=int, default=24, help="EWMA span when model=ewma")
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.003,
        help="Signal entry threshold for q50",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.03,
        help="Maximum allowed q90-q10 spread before blocking entries",
    )
    parser.add_argument(
        "--risk-limit",
        type=float,
        default=0.015,
        help="Risk-limit guard used in long/short decision conditions",
    )
    parser.add_argument(
        "--entry-policy",
        type=str,
        choices=["threshold", "net_edge"],
        default="threshold",
        help="Signal entry policy mode (legacy threshold or net-edge-aware)",
    )
    parser.add_argument(
        "--net-edge-cost-mult",
        type=float,
        default=1.0,
        help="Multiplier for expected-cost component in net-edge required edge",
    )
    parser.add_argument(
        "--net-edge-risk-mult",
        type=float,
        default=0.0,
        help="Multiplier for predicted-risk component in net-edge required edge",
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
        "--min-position",
        type=float,
        default=0.01,
        help="Minimum absolute position after sizing",
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=0.15,
        help="Annualized volatility target used by position sizer",
    )
    parser.add_argument(
        "--entry-threshold-trend-mult",
        type=float,
        default=1.0,
        help="Entry-threshold multiplier when regime=trend",
    )
    parser.add_argument(
        "--entry-threshold-normal-mult",
        type=float,
        default=1.0,
        help="Entry-threshold multiplier when regime=normal",
    )
    parser.add_argument(
        "--entry-threshold-chop-mult",
        type=float,
        default=1.0,
        help="Entry-threshold multiplier when regime=chop",
    )
    parser.add_argument(
        "--entry-threshold-panic-mult",
        type=float,
        default=1.0,
        help="Entry-threshold multiplier when regime=panic",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Replay start date (YYYY-MM-DD)")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO, help="Scenario profile")
    parser.add_argument("--retrain-bars", type=int, default=42, help="Bars between model retrains")
    parser.add_argument(
        "--training-window-bars",
        type=int,
        default=1080,
        help="Rolling training window size in bars",
    )
    parser.add_argument(
        "--min-train-samples",
        type=int,
        default=500,
        help="Minimum train samples required before replay starts",
    )
    parser.add_argument("--current-stage", type=str, default="paper", help="Current capital ramp stage")
    parser.add_argument("--days-in-stage", type=int, default=21, help="Days spent in current stage")
    parser.add_argument("--kill-max-cost-to-gross", type=float, default=0.70)
    parser.add_argument("--kill-max-turnover", type=float, default=35.0)
    parser.add_argument("--kill-min-active-bars", type=int, default=1)
    parser.add_argument("--kill-min-trades-for-pf-sharpe", type=int, default=2)
    parser.add_argument("--kill-min-active-bars-for-soft", type=int, default=2)
    parser.add_argument("--kill-min-bars-for-soft", type=int, default=1)
    parser.add_argument("--kill-min-soft-window-turnover", type=float, default=0.0)
    parser.add_argument("--kill-min-soft-window-abs-net-return", type=float, default=0.0)
    parser.add_argument("--kill-min-bars-for-cost-to-gross", type=int, default=1)
    parser.add_argument("--kill-min-abs-gross-return-for-cost-to-gross", type=float, default=0.0)
    parser.add_argument("--kill-min-windows-before-enforcement", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="data/results", help="Artifact output directory")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    if not args.verbose:
        logging.getLogger("src.strategy.signals").setLevel(logging.WARNING)
        logging.getLogger("src.strategy.position_sizing").setLevel(logging.WARNING)

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
            raise FileNotFoundError(f"Data file not found: {data_path}")

        scenario = get_scenario_profile(args.scenario)
        data_quality_gate_path = output_dir / f"{args.model}_phase11_data_quality_gate.json"
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

        data = pd.read_parquet(data_path)
        features = _select_feature_columns(data)
        model_configs = {
            "random_walk": (RandomWalkBaseline, {"lookback_window": 252}),
            "ewma": (EWMABaseline, {"span": max(2, args.ewma_span)}),
            "lightgbm": (
                LightGBMQuantileBaseline,
                {
                    "n_estimators": 300,
                    "early_stopping_rounds": 30,
                    "feature_columns": features,
                },
            ),
        }
        model_class, model_kwargs = model_configs[args.model]

        cost_model = CostModel(
            exchange=scenario.exchange,
            market_type=scenario.market_type,
            order_type=scenario.order_type,
            use_fee_discount=scenario.use_fee_discount,
            apply_funding=scenario.apply_funding,
            apply_margin_interest=scenario.apply_margin_interest,
            margin_interest_rate_per_day=scenario.default_margin_interest_rate_per_day,
            other_cost_bps=scenario.other_cost_bps,
        )
        signal_generator = QuantileSignalGenerator(
            entry_threshold=args.entry_threshold,
            uncertainty_threshold=args.uncertainty_threshold,
            risk_limit=args.risk_limit,
            entry_policy=args.entry_policy,
            net_edge_cost_multiplier=float(args.net_edge_cost_mult),
            net_edge_risk_multiplier=float(args.net_edge_risk_mult),
            expected_cost_holding_bars=max(1, int(args.expected_cost_holding_bars)),
            expected_cost_round_trip=args.expected_cost_mode == "round_trip",
            regime_entry_multipliers={
                "trend": max(0.1, float(args.entry_threshold_trend_mult)),
                "normal": max(0.1, float(args.entry_threshold_normal_mult)),
                "chop": max(0.1, float(args.entry_threshold_chop_mult)),
                "panic": max(0.1, float(args.entry_threshold_panic_mult)),
            },
        )
        position_sizer = PositionSizer(
            vol_target=max(0.01, args.vol_target),
            min_position=max(0.0, args.min_position),
        )
        kill_thresholds = KillSwitchThresholds(
            max_cost_to_gross=float(args.kill_max_cost_to_gross),
            max_turnover=float(args.kill_max_turnover),
            min_active_bars=max(1, int(args.kill_min_active_bars)),
            min_trades_for_pf_sharpe=max(1, int(args.kill_min_trades_for_pf_sharpe)),
            min_active_bars_for_soft=max(1, int(args.kill_min_active_bars_for_soft)),
            min_bars_for_soft=max(1, int(args.kill_min_bars_for_soft)),
            min_soft_window_turnover=max(0.0, float(args.kill_min_soft_window_turnover)),
            min_soft_window_abs_net_return=max(0.0, float(args.kill_min_soft_window_abs_net_return)),
            min_bars_for_cost_to_gross=max(1, int(args.kill_min_bars_for_cost_to_gross)),
            min_abs_gross_return_for_cost_to_gross=max(
                0.0,
                float(args.kill_min_abs_gross_return_for_cost_to_gross),
            ),
            min_windows_before_enforcement=max(1, int(args.kill_min_windows_before_enforcement)),
        )

        engine = PaperTradingEngine(
            model_class=model_class,
            model_kwargs=model_kwargs,
            config=PaperTradingConfig(
                retrain_interval_bars=max(1, args.retrain_bars),
                training_window_bars=max(32, args.training_window_bars),
                min_train_samples=max(30, args.min_train_samples),
            ),
            cost_model=cost_model,
            signal_generator=signal_generator,
            position_sizer=position_sizer,
        )

        replay = engine.run(
            data,
            feature_columns=features,
            start_date=args.start_date,
            model_name=args.model,
            scenario_name=args.scenario,
        )

        returns = replay.backtest_result.returns
        if returns is None:
            raise RuntimeError("Paper-trading replay produced no returns")

        dashboards = build_daily_weekly_dashboards(returns)
        daily_dashboard, daily_events = evaluate_kill_switch(
            dashboards["daily"],
            thresholds=kill_thresholds,
        )
        weekly_dashboard, weekly_events = evaluate_kill_switch(
            dashboards["weekly"],
            thresholds=kill_thresholds,
        )

        all_events = [*daily_events, *weekly_events]
        kill_taxonomy = build_kill_event_taxonomy(all_events)
        low_activity_diagnostics = {
            "daily": build_low_activity_diagnostics(
                dashboards["daily"],
                flagged_dashboard=daily_dashboard,
                thresholds=kill_thresholds,
            ),
            "weekly": build_low_activity_diagnostics(
                dashboards["weekly"],
                flagged_dashboard=weekly_dashboard,
                thresholds=kill_thresholds,
            ),
        }

        readiness = evaluate_deployment_readiness(
            returns=returns,
            weekly_dashboard=weekly_dashboard,
            kill_events=all_events,
        )

        ramp_policy = default_capital_ramp_policy()
        ramp_decision = recommend_capital_action(
            current_stage=args.current_stage,
            days_in_stage=args.days_in_stage,
            weekly_dashboard=weekly_dashboard,
            readiness=readiness,
            kill_events=all_events,
            policy=ramp_policy,
        )

        decision = build_decision_report(
            model_name=f"{args.model}_paper",
            result=replay.backtest_result,
            baseline_sharpe=None,
        )
        decision_paths = save_decision_artifacts(
            decision,
            output_dir=output_dir,
            prefix=f"{args.model}_phase10_paper",
        )
        artifacts.extend(str(path) for path in decision_paths.values())

        paper_log_path = output_dir / f"{args.model}_paper_log.csv"
        replay.paper_log.to_csv(paper_log_path)
        artifacts.append(str(paper_log_path))

        returns_path = output_dir / f"{args.model}_paper_returns.csv"
        returns.to_csv(returns_path)
        artifacts.append(str(returns_path))

        daily_path = output_dir / f"{args.model}_paper_dashboard_daily.csv"
        daily_dashboard.to_csv(daily_path, index=False)
        artifacts.append(str(daily_path))

        weekly_path = output_dir / f"{args.model}_paper_dashboard_weekly.csv"
        weekly_dashboard.to_csv(weekly_path, index=False)
        artifacts.append(str(weekly_path))

        kill_path = output_dir / f"{args.model}_paper_kill_switch_events.json"
        _write_json(kill_path, serialize_kill_events(all_events))
        artifacts.append(str(kill_path))

        taxonomy_path = output_dir / f"{args.model}_paper_kill_event_taxonomy.json"
        _write_json(taxonomy_path, kill_taxonomy)
        artifacts.append(str(taxonomy_path))

        activity_diag_path = output_dir / f"{args.model}_paper_low_activity_diagnostics.json"
        _write_json(activity_diag_path, low_activity_diagnostics)
        artifacts.append(str(activity_diag_path))

        readiness_path = output_dir / f"{args.model}_paper_deployment_readiness.json"
        _write_json(readiness_path, readiness.to_dict())
        artifacts.append(str(readiness_path))

        ramp_policy_json = output_dir / f"{args.model}_capital_ramp_policy.json"
        _write_json(ramp_policy_json, ramp_policy.to_dict())
        artifacts.append(str(ramp_policy_json))

        ramp_policy_txt = output_dir / f"{args.model}_capital_ramp_policy.txt"
        ramp_policy_txt.write_text(render_capital_ramp_policy(ramp_policy), encoding="utf-8")
        artifacts.append(str(ramp_policy_txt))

        ramp_decision_json = output_dir / f"{args.model}_capital_ramp_decision.json"
        _write_json(ramp_decision_json, ramp_decision.to_dict())
        artifacts.append(str(ramp_decision_json))

        summary_path = output_dir / f"{args.model}_paper_phase10_summary.json"
        summary_payload = replay.to_dict()
        summary_payload["decision"] = decision.to_dict()
        summary_payload["kill_switch_events"] = serialize_kill_events(all_events)
        summary_payload["kill_event_taxonomy"] = kill_taxonomy
        summary_payload["kill_thresholds"] = {
            "max_cost_to_gross": kill_thresholds.max_cost_to_gross,
            "max_turnover": kill_thresholds.max_turnover,
            "min_active_bars": kill_thresholds.min_active_bars,
            "min_trades_for_pf_sharpe": kill_thresholds.min_trades_for_pf_sharpe,
            "min_active_bars_for_soft": kill_thresholds.min_active_bars_for_soft,
            "min_bars_for_soft": kill_thresholds.min_bars_for_soft,
            "min_soft_window_turnover": kill_thresholds.min_soft_window_turnover,
            "min_soft_window_abs_net_return": kill_thresholds.min_soft_window_abs_net_return,
            "min_bars_for_cost_to_gross": kill_thresholds.min_bars_for_cost_to_gross,
            "min_abs_gross_return_for_cost_to_gross": kill_thresholds.min_abs_gross_return_for_cost_to_gross,
            "min_windows_before_enforcement": kill_thresholds.min_windows_before_enforcement,
        }
        summary_payload["low_activity_diagnostics"] = low_activity_diagnostics
        summary_payload["deployment_readiness"] = readiness.to_dict()
        summary_payload["capital_ramp_decision"] = ramp_decision.to_dict()
        summary_payload["daily_windows"] = int(len(daily_dashboard))
        summary_payload["weekly_windows"] = int(len(weekly_dashboard))
        _write_json(summary_path, summary_payload)
        artifacts.append(str(summary_path))

        logger.info("Paper replay complete for model=%s", args.model)
        logger.info("PF(Net)=%.3f", summary_payload["metrics"]["profit_factor_net"])
        logger.info("Sharpe=%.3f", summary_payload["metrics"]["sharpe_ratio"])
        logger.info("Readiness=%s", readiness.ready)
        logger.info("Ramp Action=%s -> %s", ramp_decision.action, ramp_decision.recommended_stage)

        print("=" * 80)
        print("PHASE 10 PAPER-TRADING SUMMARY")
        print("=" * 80)
        print(f"Model:               {args.model}")
        print(f"Scenario:            {args.scenario}")
        print(f"Bars Replayed:       {len(replay.paper_log)}")
        print(f"ProfitFactorNet:     {summary_payload['metrics']['profit_factor_net']:.3f}")
        print(f"SharpeNet:           {summary_payload['metrics']['sharpe_ratio']:.3f}")
        print(f"Kill Events:         {len(all_events)}")
        print(f"Deployment Ready:    {readiness.ready} ({readiness.reason})")
        print(
            "Capital Action:      "
            f"{ramp_decision.action} -> {ramp_decision.recommended_stage}"
            f" ({ramp_decision.recommended_capital_fraction:.0%})"
        )

        return 0
    except Exception as exc:
        status = "failed"
        error = str(exc)
        logger.exception("Paper-trading run failed")
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
    raise SystemExit(main())
