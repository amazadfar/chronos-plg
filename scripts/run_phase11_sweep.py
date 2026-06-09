#!/usr/bin/env python
"""Run Phase 11.5 multi-objective parameter sweep and frontier selection."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.scenario_profiles import get_scenario_profile
from src.backtest.costs import CostModel
from src.common.timeframe import (
    SUPPORTED_TIMEFRAMES,
    default_processed_dataset_path,
    normalize_timeframe,
)
from src.data.quality_gate import enforce_degraded_run_gate
from src.evaluation.multi_objective import (
    AcceptanceConstraints,
    CompositeScoreWeights,
    pareto_frontier,
    rank_candidates,
    schema_payload,
)
from src.models.baselines import EWMABaseline, LightGBMQuantileBaseline, RandomWalkBaseline
from src.paper_trading import (
    PaperTradingConfig,
    PaperTradingEngine,
    build_daily_weekly_dashboards,
    evaluate_deployment_readiness,
    evaluate_kill_switch,
)
from src.paper_trading.policy import KillSwitchThresholds
from src.strategy.position_sizing import PositionSizer
from src.strategy.signals import QuantileSignalGenerator
from src.utils import finalize_experiment_run, set_global_seed, start_experiment_run


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _write_markdown(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    try:
        md = frame.to_markdown(index=False)
    except Exception:
        headers = list(frame.columns)
        sep = "| " + " | ".join(["---"] * len(headers)) + " |"
        lines = ["| " + " | ".join(headers) + " |", sep]
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        md = "\n".join(lines)
    path.write_text(md, encoding="utf-8")


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return []
    return [float(item) for item in values]


def _parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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
        raise ValueError("No numeric feature columns found for sweep run")
    return columns


def _model_config(model_key: str, feature_columns: list[str]) -> tuple[type, dict[str, Any]]:
    registry: dict[str, tuple[type, dict[str, Any]]] = {
        "random_walk": (RandomWalkBaseline, {"lookback_window": 252}),
        "ewma": (EWMABaseline, {"span": 24}),
        "lightgbm": (
            LightGBMQuantileBaseline,
            {
                "n_estimators": 300,
                "early_stopping_rounds": 30,
                "feature_columns": feature_columns,
            },
        ),
    }
    if model_key not in registry:
        raise ValueError(f"Unsupported model: {model_key}")
    return registry[model_key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 11.5 multi-objective sweep")
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
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
        default="ewma",
        choices=["random_walk", "ewma", "lightgbm"],
        help="Model family for sweep",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="binance_spot_taker_discounted",
        help="Scenario profile name",
    )
    parser.add_argument("--start-date", type=str, default="2025-12-01", help="Replay start date")
    parser.add_argument("--retrain-bars", type=int, default=48, help="Bars between retrains")
    parser.add_argument("--training-window-bars", type=int, default=1080, help="Rolling train window bars")
    parser.add_argument("--min-train-samples", type=int, default=240, help="Minimum train samples")
    parser.add_argument("--risk-limit", type=float, default=0.015, help="Signal risk-limit guard")
    parser.add_argument("--min-position", type=float, default=0.01, help="Minimum sized position")
    parser.add_argument("--vol-target", type=float, default=0.15, help="Vol target for sizing")

    parser.add_argument(
        "--entry-policies",
        type=str,
        default="threshold,net_edge",
        help="Comma-separated entry policies",
    )
    parser.add_argument(
        "--entry-thresholds",
        type=str,
        default="0.0025,0.003,0.0035",
        help="Comma-separated entry thresholds",
    )
    parser.add_argument(
        "--uncertainty-thresholds",
        type=str,
        default="0.02,0.03",
        help="Comma-separated uncertainty thresholds",
    )
    parser.add_argument(
        "--net-edge-cost-mults",
        type=str,
        default="0.75,1.0,1.25",
        help="Comma-separated net-edge expected-cost multipliers",
    )
    parser.add_argument(
        "--net-edge-risk-mults",
        type=str,
        default="0.0,0.25,0.5",
        help="Comma-separated net-edge risk multipliers",
    )
    parser.add_argument(
        "--expected-cost-holding-bars",
        type=int,
        default=1,
        help="Expected holding horizon in bars for net-edge estimate",
    )
    parser.add_argument(
        "--expected-cost-mode",
        type=str,
        choices=["round_trip", "entry_only"],
        default="round_trip",
        help="Expected-cost mode",
    )
    parser.add_argument(
        "--entry-threshold-trend-mults",
        type=str,
        default="1.0",
        help="Comma-separated trend regime threshold multipliers",
    )
    parser.add_argument(
        "--entry-threshold-normal-mults",
        type=str,
        default="1.0",
        help="Comma-separated normal regime threshold multipliers",
    )
    parser.add_argument(
        "--entry-threshold-chop-mults",
        type=str,
        default="1.0",
        help="Comma-separated chop regime threshold multipliers",
    )
    parser.add_argument(
        "--entry-threshold-panic-mults",
        type=str,
        default="1.0",
        help="Comma-separated panic regime threshold multipliers",
    )

    parser.add_argument("--weight-profit-factor", type=float, default=1.0)
    parser.add_argument("--weight-sharpe", type=float, default=1.0)
    parser.add_argument("--weight-trades-log", type=float, default=0.25)
    parser.add_argument("--weight-kill-rate", type=float, default=1.0)
    parser.add_argument("--weight-turnover-log", type=float, default=0.05)
    parser.add_argument("--weight-drawdown-abs", type=float, default=0.5)

    parser.add_argument("--min-profit-factor-net", type=float, default=1.0)
    parser.add_argument("--min-sharpe-net", type=float, default=0.5)
    parser.add_argument("--min-trades", type=int, default=80)
    parser.add_argument("--max-kill-event-rate", type=float, default=0.20)
    parser.add_argument("--max-drawdown-abs", type=float, default=0.30)

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

    parser.add_argument("--output-dir", type=str, default="data/results/phase11_5_sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    if not args.verbose:
        logging.getLogger("src.strategy.signals").setLevel(logging.WARNING)
        logging.getLogger("src.strategy.position_sizing").setLevel(logging.WARNING)
        logging.getLogger("src.models.baselines").setLevel(logging.WARNING)
        logging.getLogger("src.strategy.regime_detector").setLevel(logging.WARNING)

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
        data_path = Path(args.data) if args.data else default_processed_dataset_path(timeframe=timeframe)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        scenario = get_scenario_profile(args.scenario)
        gate_path = output_dir / "phase11_data_quality_gate.json"
        gate = enforce_degraded_run_gate(
            data_path,
            market_type=scenario.market_type,
            artifact_path=gate_path,
        )
        artifacts.append(str(gate_path))
        logger.info(
            "Data quality gate passed (market=%s availability=%s)",
            gate.market_type,
            gate.availability,
        )

        data = pd.read_parquet(data_path)
        features = _select_feature_columns(data)
        model_class, model_kwargs = _model_config(args.model, features)

        entry_policies = _parse_str_list(args.entry_policies)
        entry_thresholds = _parse_float_list(args.entry_thresholds)
        uncertainty_thresholds = _parse_float_list(args.uncertainty_thresholds)
        net_edge_cost_mults = _parse_float_list(args.net_edge_cost_mults)
        net_edge_risk_mults = _parse_float_list(args.net_edge_risk_mults)
        trend_mults = _parse_float_list(args.entry_threshold_trend_mults)
        normal_mults = _parse_float_list(args.entry_threshold_normal_mults)
        chop_mults = _parse_float_list(args.entry_threshold_chop_mults)
        panic_mults = _parse_float_list(args.entry_threshold_panic_mults)

        if not entry_policies:
            raise ValueError("No entry policies provided")
        if not entry_thresholds or not uncertainty_thresholds:
            raise ValueError("Empty threshold grid provided")
        if not net_edge_cost_mults:
            net_edge_cost_mults = [1.0]
        if not net_edge_risk_mults:
            net_edge_risk_mults = [0.0]
        if not trend_mults:
            trend_mults = [1.0]
        if not normal_mults:
            normal_mults = [1.0]
        if not chop_mults:
            chop_mults = [1.0]
        if not panic_mults:
            panic_mults = [1.0]

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

        weights = CompositeScoreWeights(
            profit_factor=float(args.weight_profit_factor),
            sharpe=float(args.weight_sharpe),
            trades_log=float(args.weight_trades_log),
            kill_event_rate=float(args.weight_kill_rate),
            turnover_log=float(args.weight_turnover_log),
            drawdown_abs=float(args.weight_drawdown_abs),
        )
        constraints = AcceptanceConstraints(
            min_profit_factor_net=float(args.min_profit_factor_net),
            min_sharpe_net=float(args.min_sharpe_net),
            min_trades=max(0, int(args.min_trades)),
            max_kill_event_rate=max(0.0, float(args.max_kill_event_rate)),
            max_drawdown_abs=max(0.0, float(args.max_drawdown_abs)),
        )

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

        # Build unique parameter grid.
        grid: list[dict[str, Any]] = []
        for policy in entry_policies:
            if policy not in {"threshold", "net_edge"}:
                logger.warning("Skipping unsupported policy: %s", policy)
                continue
            for entry_threshold in entry_thresholds:
                for uncertainty_threshold in uncertainty_thresholds:
                    for trend_mult in trend_mults:
                        for normal_mult in normal_mults:
                            for chop_mult in chop_mults:
                                for panic_mult in panic_mults:
                                    base_payload = {
                                        "entry_policy": policy,
                                        "entry_threshold": float(entry_threshold),
                                        "uncertainty_threshold": float(uncertainty_threshold),
                                        "entry_threshold_trend_mult": float(trend_mult),
                                        "entry_threshold_normal_mult": float(normal_mult),
                                        "entry_threshold_chop_mult": float(chop_mult),
                                        "entry_threshold_panic_mult": float(panic_mult),
                                    }
                                    if policy == "threshold":
                                        grid.append(
                                            {
                                                **base_payload,
                                                "net_edge_cost_mult": 1.0,
                                                "net_edge_risk_mult": 0.0,
                                            }
                                        )
                                    else:
                                        for c_mult in net_edge_cost_mults:
                                            for r_mult in net_edge_risk_mults:
                                                grid.append(
                                                    {
                                                        **base_payload,
                                                        "net_edge_cost_mult": float(c_mult),
                                                        "net_edge_risk_mult": float(r_mult),
                                                    }
                                                )

        # Deduplicate combinations in case of repeated input values.
        seen = set()
        unique_grid: list[dict[str, Any]] = []
        for row in grid:
            key = (
                row["entry_policy"],
                row["entry_threshold"],
                row["uncertainty_threshold"],
                row["net_edge_cost_mult"],
                row["net_edge_risk_mult"],
                row["entry_threshold_trend_mult"],
                row["entry_threshold_normal_mult"],
                row["entry_threshold_chop_mult"],
                row["entry_threshold_panic_mult"],
            )
            if key in seen:
                continue
            seen.add(key)
            unique_grid.append(row)

        logger.info("Sweep candidates to evaluate: %s", len(unique_grid))

        engine_config = PaperTradingConfig(
            retrain_interval_bars=max(1, int(args.retrain_bars)),
            training_window_bars=max(32, int(args.training_window_bars)),
            min_train_samples=max(30, int(args.min_train_samples)),
        )
        position_sizer = PositionSizer(
            vol_target=max(0.01, float(args.vol_target)),
            min_position=max(0.0, float(args.min_position)),
        )

        candidates: list[dict[str, Any]] = []

        for idx, params in enumerate(unique_grid, start=1):
            logger.info(
                (
                    "Candidate %s/%s policy=%s entry=%.6f unc=%.6f "
                    "c_mult=%.3f r_mult=%.3f trend=%.3f normal=%.3f chop=%.3f panic=%.3f"
                ),
                idx,
                len(unique_grid),
                params["entry_policy"],
                params["entry_threshold"],
                params["uncertainty_threshold"],
                params["net_edge_cost_mult"],
                params["net_edge_risk_mult"],
                params["entry_threshold_trend_mult"],
                params["entry_threshold_normal_mult"],
                params["entry_threshold_chop_mult"],
                params["entry_threshold_panic_mult"],
            )
            candidate_payload: dict[str, Any] = {
                "candidate_id": idx,
                "model": args.model,
                "scenario": args.scenario,
                "start_date": args.start_date,
                "retrain_bars": max(1, int(args.retrain_bars)),
                "training_window_bars": max(32, int(args.training_window_bars)),
                "min_train_samples": max(30, int(args.min_train_samples)),
                "risk_limit": float(args.risk_limit),
                "min_position": max(0.0, float(args.min_position)),
                "vol_target": max(0.01, float(args.vol_target)),
                "expected_cost_holding_bars": max(1, int(args.expected_cost_holding_bars)),
                "expected_cost_mode": str(args.expected_cost_mode),
                **params,
            }

            try:
                signal_generator = QuantileSignalGenerator(
                    entry_threshold=float(params["entry_threshold"]),
                    uncertainty_threshold=float(params["uncertainty_threshold"]),
                    risk_limit=float(args.risk_limit),
                    entry_policy=str(params["entry_policy"]),
                    net_edge_cost_multiplier=float(params["net_edge_cost_mult"]),
                    net_edge_risk_multiplier=float(params["net_edge_risk_mult"]),
                    expected_cost_holding_bars=max(1, int(args.expected_cost_holding_bars)),
                    expected_cost_round_trip=args.expected_cost_mode == "round_trip",
                    regime_entry_multipliers={
                        "trend": max(0.1, float(params["entry_threshold_trend_mult"])),
                        "normal": max(0.1, float(params["entry_threshold_normal_mult"])),
                        "chop": max(0.1, float(params["entry_threshold_chop_mult"])),
                        "panic": max(0.1, float(params["entry_threshold_panic_mult"])),
                    },
                )
                engine = PaperTradingEngine(
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    config=engine_config,
                    cost_model=cost_model,
                    signal_generator=signal_generator,
                    position_sizer=position_sizer,
                )
                replay = engine.run(
                    data=data,
                    feature_columns=features,
                    start_date=args.start_date,
                    model_name=args.model,
                    scenario_name=args.scenario,
                )
                returns = replay.backtest_result.returns
                if returns is None:
                    raise RuntimeError("No returns generated")

                dashboards = build_daily_weekly_dashboards(returns)
                daily_dashboard, daily_events = evaluate_kill_switch(dashboards["daily"], thresholds=kill_thresholds)
                weekly_dashboard, weekly_events = evaluate_kill_switch(
                    dashboards["weekly"], thresholds=kill_thresholds
                )
                all_events = [*daily_events, *weekly_events]

                readiness = evaluate_deployment_readiness(
                    returns=returns,
                    weekly_dashboard=weekly_dashboard,
                    kill_events=all_events,
                )

                pf = (
                    replay.backtest_result.profit_factor_net
                    if replay.backtest_result.profit_factor_net > 0
                    else replay.backtest_result.profit_factor
                )
                monitoring_windows = int(len(daily_dashboard) + len(weekly_dashboard))
                kill_event_rate = float(len(all_events) / monitoring_windows) if monitoring_windows > 0 else 0.0
                turnover_total = float(returns["turnover"].sum()) if "turnover" in returns.columns else 0.0

                candidate_payload.update(
                    {
                        "run_status": "success",
                        "profit_factor_net": float(pf),
                        "sharpe_ratio": float(replay.backtest_result.sharpe_ratio),
                        "total_return": float(replay.backtest_result.total_return),
                        "max_drawdown": float(replay.backtest_result.max_drawdown),
                        "num_trades": int(replay.backtest_result.num_trades),
                        "turnover": turnover_total,
                        "kill_events": int(len(all_events)),
                        "kill_event_rate": kill_event_rate,
                        "monitoring_windows": monitoring_windows,
                        "daily_windows": int(len(daily_dashboard)),
                        "weekly_windows": int(len(weekly_dashboard)),
                        "deployment_ready": bool(readiness.ready),
                        "deployment_reason": str(readiness.reason),
                    }
                )
            except Exception as exc:  # pragma: no cover - guarded by sweep-level robustness
                candidate_payload.update(
                    {
                        "run_status": "failed",
                        "error": str(exc),
                        "profit_factor_net": 0.0,
                        "sharpe_ratio": 0.0,
                        "total_return": 0.0,
                        "max_drawdown": 0.0,
                        "num_trades": 0,
                        "turnover": 0.0,
                        "kill_events": 0,
                        "kill_event_rate": 1.0,
                        "monitoring_windows": 0,
                        "daily_windows": 0,
                        "weekly_windows": 0,
                        "deployment_ready": False,
                        "deployment_reason": "candidate_failed",
                    }
                )
                logger.warning("Candidate %s failed: %s", idx, exc)

            candidates.append(candidate_payload)

        raw_df = pd.DataFrame(candidates)
        raw_json_path = output_dir / "phase11_sweep_candidates_raw.json"
        raw_csv_path = output_dir / "phase11_sweep_candidates_raw.csv"
        _write_json(raw_json_path, raw_df.to_dict(orient="records"))
        raw_df.to_csv(raw_csv_path, index=False)
        artifacts.extend([str(raw_json_path), str(raw_csv_path)])

        success_df = raw_df[raw_df["run_status"] == "success"].copy()
        ranked = rank_candidates(success_df, weights=weights, constraints=constraints)
        ranked["max_drawdown_abs"] = ranked["max_drawdown"].abs() if not ranked.empty else []

        frontier = pareto_frontier(ranked)
        frontier_ids = set(frontier["candidate_id"].tolist()) if not frontier.empty else set()
        if not ranked.empty:
            ranked["pareto_frontier"] = ranked["candidate_id"].isin(frontier_ids)

        ranked_json_path = output_dir / "phase11_sweep_ranked.json"
        ranked_csv_path = output_dir / "phase11_sweep_ranked.csv"
        ranked_md_path = output_dir / "phase11_sweep_ranked.md"
        _write_json(ranked_json_path, ranked.to_dict(orient="records"))
        ranked.to_csv(ranked_csv_path, index=False)
        _write_markdown(ranked_md_path, ranked)
        artifacts.extend([str(ranked_json_path), str(ranked_csv_path), str(ranked_md_path)])

        frontier_json_path = output_dir / "phase11_sweep_pareto_frontier.json"
        frontier_csv_path = output_dir / "phase11_sweep_pareto_frontier.csv"
        _write_json(frontier_json_path, frontier.to_dict(orient="records"))
        frontier.to_csv(frontier_csv_path, index=False)
        artifacts.extend([str(frontier_json_path), str(frontier_csv_path)])

        schema_path = output_dir / "phase11_sweep_schema.json"
        _write_json(schema_path, schema_payload())
        artifacts.append(str(schema_path))

        accepted = ranked[ranked["acceptance_passed"]] if "acceptance_passed" in ranked.columns else pd.DataFrame()
        summary = {
            "schema_version": "phase11_sweep_summary_v1",
            "timeframe": args.timeframe,
            "scenario": args.scenario,
            "model": args.model,
            "start_date": args.start_date,
            "candidates_total": int(len(raw_df)),
            "candidates_successful": int(len(success_df)),
            "candidates_failed": int(len(raw_df) - len(success_df)),
            "accepted_count": int(len(accepted)),
            "pareto_frontier_count": int(len(frontier)),
            "weights": weights.to_dict(),
            "constraints": constraints.to_dict(),
            "kill_thresholds": {
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
            },
            "best_overall": ranked.iloc[0].to_dict() if not ranked.empty else None,
            "best_accepted": accepted.iloc[0].to_dict() if not accepted.empty else None,
            "reproducible_command": (
                "python scripts/run_phase11_sweep.py "
                f"--timeframe {args.timeframe} --scenario {args.scenario} --model {args.model} "
                f"--start-date {args.start_date} --output-dir {str(output_dir)}"
            ),
        }
        summary_path = output_dir / "phase11_sweep_summary.json"
        _write_json(summary_path, summary)
        artifacts.append(str(summary_path))

        print("=" * 80)
        print("PHASE 11.5 MULTI-OBJECTIVE SWEEP")
        print("=" * 80)
        print(f"Candidates:          {summary['candidates_total']} (successful={summary['candidates_successful']})")
        print(f"Accepted:            {summary['accepted_count']}")
        print(f"Pareto frontier:     {summary['pareto_frontier_count']}")
        if summary["best_overall"] is not None:
            best = summary["best_overall"]
            print(
                "Best overall:        "
                f"id={best['candidate_id']} score={best.get('composite_score', 0.0):.4f} "
                f"PF={best.get('profit_factor_net', 0.0):.3f} "
                f"Sharpe={best.get('sharpe_ratio', 0.0):.3f} "
                f"Trades={int(best.get('num_trades', 0))}"
            )
        if summary["best_accepted"] is None:
            print("Best accepted:       none (no candidate passed acceptance constraints)")
        else:
            best_ok = summary["best_accepted"]
            print(
                "Best accepted:       "
                f"id={best_ok['candidate_id']} score={best_ok.get('composite_score', 0.0):.4f}"
            )
        return 0
    except Exception as exc:
        status = "failed"
        error = str(exc)
        logging.getLogger(__name__).exception("Phase 11 sweep failed")
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
