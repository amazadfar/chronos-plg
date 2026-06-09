#!/usr/bin/env python
"""Run Phase 11.7 fixed-window promotion campaign from frozen sweep candidate."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

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
from src.evaluation.phase11_campaign import (
    Phase11CompletionCriteria,
    build_promotion_recommendation,
    evaluate_phase11_completion_gate,
    select_phase11_campaign_candidate,
)
from src.models.baselines import EWMABaseline, LightGBMQuantileBaseline, RandomWalkBaseline
from src.paper_trading import (
    PaperTradingConfig,
    PaperTradingEngine,
    build_daily_weekly_dashboards,
    default_capital_ramp_policy,
    evaluate_deployment_readiness,
    evaluate_kill_switch,
    recommend_capital_action,
    serialize_kill_events,
)
from src.strategy.position_sizing import PositionSizer
from src.strategy.signals import QuantileSignalGenerator
from src.utils import finalize_experiment_run, set_global_seed, start_experiment_run


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _to_timestamp(value: str | None, *, tz: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if tz is not None and ts.tzinfo is None:
        return ts.tz_localize(tz)
    if tz is None and ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts


def _load_ranked_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ranked-candidates file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return pd.DataFrame(payload)
    raise ValueError("ranked-candidates file must be .json or .csv")


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
        raise ValueError("No numeric feature columns found for promotion campaign")
    return columns


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None or pd.isna(value):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _safe_str(value: Any, default: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    out = str(value).strip()
    return out if out else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 11.7 promotion campaign")
    parser.add_argument("--timeframe", type=str, default="1h", choices=SUPPORTED_TIMEFRAMES)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument(
        "--ranked-candidates",
        type=str,
        default="data/results/phase11_5_sweep/phase11_sweep_ranked.json",
        help="Path to phase11 ranked candidate artifact (json/csv)",
    )
    parser.add_argument("--candidate-id", type=int, default=None, help="Optional explicit candidate id")
    parser.add_argument("--model", type=str, default=None, choices=["random_walk", "ewma", "lightgbm"])
    parser.add_argument("--scenario", type=str, default=None, help="Scenario override")
    parser.add_argument("--campaign-start-date", type=str, default=None)
    parser.add_argument("--campaign-end-date", type=str, default=None)
    parser.add_argument("--risk-limit", type=float, default=None)
    parser.add_argument("--expected-cost-holding-bars", type=int, default=None)
    parser.add_argument(
        "--expected-cost-mode",
        type=str,
        choices=["round_trip", "entry_only"],
        default=None,
    )
    parser.add_argument("--retrain-bars", type=int, default=None)
    parser.add_argument("--training-window-bars", type=int, default=None)
    parser.add_argument("--min-train-samples", type=int, default=None)
    parser.add_argument("--min-position", type=float, default=None)
    parser.add_argument("--vol-target", type=float, default=None)
    parser.add_argument("--current-stage", type=str, default="paper")
    parser.add_argument("--days-in-stage", type=int, default=21)
    parser.add_argument(
        "--completion-min-profit-factor-net",
        type=float,
        default=Phase11CompletionCriteria.min_profit_factor_net,
    )
    parser.add_argument(
        "--completion-min-sharpe-net",
        type=float,
        default=Phase11CompletionCriteria.min_sharpe_net,
    )
    parser.add_argument(
        "--completion-max-drawdown-abs",
        type=float,
        default=Phase11CompletionCriteria.max_drawdown_abs,
    )
    parser.add_argument("--completion-min-trades", type=int, default=80)
    parser.add_argument("--completion-max-kill-events", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="data/results/phase11_7_campaign")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
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
        data_path = (
            Path(args.data)
            if args.data
            else default_processed_dataset_path(timeframe=timeframe)
        )
        ranked_path = Path(args.ranked_candidates)

        ranked_df = _load_ranked_candidates(ranked_path)
        selection = select_phase11_campaign_candidate(
            ranked_df,
            candidate_id=args.candidate_id,
        )
        candidate = selection["candidate"]

        freeze_payload = {
            "schema_version": "phase11_campaign_freeze_v1",
            "selected_at_utc": pd.Timestamp.utcnow().isoformat(),
            "source_ranked_candidates": str(ranked_path),
            "selection": selection,
        }
        freeze_path = output_dir / "phase11_campaign_candidate_freeze.json"
        _write_json(freeze_path, freeze_payload)
        artifacts.append(str(freeze_path))

        model_name = _safe_str(args.model, _safe_str(candidate.get("model"), "ewma"))
        scenario_name = _safe_str(args.scenario, _safe_str(candidate.get("scenario"), DEFAULT_SCENARIO))
        campaign_start = _safe_str(args.campaign_start_date, _safe_str(candidate.get("start_date"), ""))
        campaign_start = campaign_start or None

        data = pd.read_parquet(data_path)
        end_ts = _to_timestamp(args.campaign_end_date, tz=getattr(data.index, "tz", None))
        if end_ts is not None:
            data = data.loc[data.index <= end_ts]

        if data.empty:
            raise ValueError("No data remains after applying campaign end-date filter")

        scenario = get_scenario_profile(scenario_name)
        data_quality_gate_path = output_dir / "phase11_campaign_data_quality_gate.json"
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

        features = _select_feature_columns(data)
        model_configs = {
            "random_walk": (RandomWalkBaseline, {"lookback_window": 252}),
            "ewma": (EWMABaseline, {"span": 24}),
            "lightgbm": (
                LightGBMQuantileBaseline,
                {
                    "n_estimators": 300,
                    "early_stopping_rounds": 30,
                    "feature_columns": features,
                },
            ),
        }
        model_class, model_kwargs = model_configs[model_name]

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

        risk_limit = _safe_float(args.risk_limit, _safe_float(candidate.get("risk_limit"), 0.015))
        expected_cost_holding_bars = max(
            1,
            _safe_int(args.expected_cost_holding_bars, _safe_int(candidate.get("expected_cost_holding_bars"), 1)),
        )
        expected_cost_mode = _safe_str(
            args.expected_cost_mode,
            _safe_str(candidate.get("expected_cost_mode"), "round_trip"),
        )
        retrain_bars = max(1, _safe_int(args.retrain_bars, _safe_int(candidate.get("retrain_bars"), 48)))
        training_window_bars = max(
            32,
            _safe_int(args.training_window_bars, _safe_int(candidate.get("training_window_bars"), 1080)),
        )
        min_train_samples = max(30, _safe_int(args.min_train_samples, _safe_int(candidate.get("min_train_samples"), 240)))
        min_position = max(0.0, _safe_float(args.min_position, _safe_float(candidate.get("min_position"), 0.01)))
        vol_target = max(0.01, _safe_float(args.vol_target, _safe_float(candidate.get("vol_target"), 0.15)))

        signal_generator = QuantileSignalGenerator(
            entry_policy=_safe_str(candidate.get("entry_policy"), "threshold"),
            entry_threshold=_safe_float(candidate.get("entry_threshold"), 0.003),
            uncertainty_threshold=_safe_float(candidate.get("uncertainty_threshold"), 0.03),
            risk_limit=risk_limit,
            net_edge_cost_multiplier=_safe_float(candidate.get("net_edge_cost_mult"), 1.0),
            net_edge_risk_multiplier=_safe_float(candidate.get("net_edge_risk_mult"), 0.0),
            expected_cost_holding_bars=expected_cost_holding_bars,
            expected_cost_round_trip=expected_cost_mode == "round_trip",
        )
        position_sizer = PositionSizer(
            vol_target=vol_target,
            min_position=min_position,
        )

        engine = PaperTradingEngine(
            model_class=model_class,
            model_kwargs=model_kwargs,
            config=PaperTradingConfig(
                retrain_interval_bars=retrain_bars,
                training_window_bars=training_window_bars,
                min_train_samples=min_train_samples,
            ),
            cost_model=cost_model,
            signal_generator=signal_generator,
            position_sizer=position_sizer,
        )

        replay = engine.run(
            data,
            feature_columns=features,
            start_date=campaign_start,
            model_name=model_name,
            scenario_name=scenario_name,
        )

        returns = replay.backtest_result.returns
        if returns is None:
            raise RuntimeError("Promotion campaign produced no returns")

        dashboards = build_daily_weekly_dashboards(returns)
        daily_dashboard, daily_events = evaluate_kill_switch(dashboards["daily"])
        weekly_dashboard, weekly_events = evaluate_kill_switch(dashboards["weekly"])
        all_events = [*daily_events, *weekly_events]

        readiness = evaluate_deployment_readiness(
            returns=returns,
            weekly_dashboard=weekly_dashboard,
            kill_events=all_events,
        )
        ramp_decision = recommend_capital_action(
            current_stage=args.current_stage,
            days_in_stage=int(args.days_in_stage),
            weekly_dashboard=weekly_dashboard,
            readiness=readiness,
            kill_events=all_events,
            policy=default_capital_ramp_policy(),
        )
        promotion = build_promotion_recommendation(
            readiness=readiness,
            ramp_decision=ramp_decision,
        )

        pf_net = (
            replay.backtest_result.profit_factor_net
            if replay.backtest_result.profit_factor_net > 0
            else replay.backtest_result.profit_factor
        )
        completion_gate = evaluate_phase11_completion_gate(
            metrics={
                "profit_factor_net": pf_net,
                "sharpe_ratio": replay.backtest_result.sharpe_ratio,
                "max_drawdown": replay.backtest_result.max_drawdown,
                "num_trades": replay.backtest_result.num_trades,
            },
            readiness=readiness,
            ramp_decision=ramp_decision,
            kill_event_count=len(all_events),
            criteria=Phase11CompletionCriteria(
                min_profit_factor_net=float(args.completion_min_profit_factor_net),
                min_sharpe_net=float(args.completion_min_sharpe_net),
                max_drawdown_abs=float(args.completion_max_drawdown_abs),
                min_trades=max(1, int(args.completion_min_trades)),
                max_kill_events=max(0, int(args.completion_max_kill_events)),
            ),
        )

        returns_path = output_dir / "phase11_campaign_returns.csv"
        returns.to_csv(returns_path)
        artifacts.append(str(returns_path))

        daily_path = output_dir / "phase11_campaign_dashboard_daily.csv"
        daily_dashboard.to_csv(daily_path, index=False)
        artifacts.append(str(daily_path))

        weekly_path = output_dir / "phase11_campaign_dashboard_weekly.csv"
        weekly_dashboard.to_csv(weekly_path, index=False)
        artifacts.append(str(weekly_path))

        kill_events_path = output_dir / "phase11_campaign_kill_events.json"
        _write_json(kill_events_path, serialize_kill_events(all_events))
        artifacts.append(str(kill_events_path))

        readiness_path = output_dir / "phase11_campaign_readiness.json"
        _write_json(readiness_path, readiness.to_dict())
        artifacts.append(str(readiness_path))

        ramp_path = output_dir / "phase11_campaign_capital_ramp_decision.json"
        _write_json(ramp_path, ramp_decision.to_dict())
        artifacts.append(str(ramp_path))

        promotion_path = output_dir / "phase11_campaign_promotion_recommendation.json"
        _write_json(promotion_path, promotion)
        artifacts.append(str(promotion_path))

        completion_path = output_dir / "phase11_completion_gate.json"
        _write_json(completion_path, completion_gate)
        artifacts.append(str(completion_path))

        summary = {
            "schema_version": "phase11_campaign_summary_v1",
            "selected_candidate": candidate,
            "selection_mode": selection.get("selection_mode"),
            "model": model_name,
            "scenario": scenario_name,
            "campaign_start_date": campaign_start,
            "campaign_end_date": args.campaign_end_date,
            "metrics": {
                "profit_factor_net": float(pf_net),
                "sharpe_ratio": float(replay.backtest_result.sharpe_ratio),
                "max_drawdown": float(replay.backtest_result.max_drawdown),
                "num_trades": int(replay.backtest_result.num_trades),
                "total_return": float(replay.backtest_result.total_return),
                "kill_events": int(len(all_events)),
            },
            "deployment_readiness": readiness.to_dict(),
            "capital_ramp_decision": ramp_decision.to_dict(),
            "promotion_recommendation": promotion,
            "completion_gate": completion_gate,
            "reproducible_command": (
                "python scripts/run_phase11_promotion_campaign.py "
                f"--timeframe {timeframe} --ranked-candidates {ranked_path} "
                f"--output-dir {output_dir}"
            ),
        }
        summary_path = output_dir / "phase11_campaign_summary.json"
        _write_json(summary_path, summary)
        artifacts.append(str(summary_path))

        print("=" * 80)
        print("PHASE 11.7 PROMOTION CAMPAIGN")
        print("=" * 80)
        print(f"Candidate Mode:      {selection.get('selection_mode')}")
        print(f"Model/Scenario:      {model_name} / {scenario_name}")
        print(f"PF(Net):             {float(pf_net):.3f}")
        print(f"Sharpe:              {float(replay.backtest_result.sharpe_ratio):.3f}")
        print(f"Kill Events:         {len(all_events)}")
        print(
            "Promotion Decision:  "
            f"{promotion['recommend_promotion']} ({promotion['reason']})"
        )
        print(
            "Completion Gate:     "
            f"{completion_gate['passed']} ({completion_gate['reason']})"
        )
        return 0
    except Exception as exc:
        status = "failed"
        error = str(exc)
        logging.getLogger(__name__).exception("Phase 11.7 campaign failed")
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
