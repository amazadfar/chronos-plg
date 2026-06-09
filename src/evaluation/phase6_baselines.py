"""Phase 6 baseline protocol utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.baseline_protocols import BaselineProtocol
from src.backtest.engine import BacktestResult
from src.common.metrics import DEFAULT_SUCCESS_THRESHOLDS
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.models.baselines import EWMABaseline, LightGBMQuantileBaseline, RandomWalkBaseline


def infer_feature_columns(data: pd.DataFrame) -> list[str]:
    """Infer numeric feature columns with leakage-safe exclusions."""
    excluded_tokens = ("forward_", "regime", "hist_q", "timestamp")
    excluded_exact = {"open", "high", "low", "close", "volume"}

    features = [
        col
        for col in data.columns
        if pd.api.types.is_numeric_dtype(data[col])
        and col not in excluded_exact
        and not any(token in col for token in excluded_tokens)
    ]
    if not features:
        raise ValueError("No candidate numeric feature columns were inferred.")
    return features


def _stable_payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_protocol_freeze(
    protocol: BaselineProtocol,
    output_dir: Path,
) -> Path:
    """Write immutable protocol payload with fingerprint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = protocol.to_dict()
    payload["fingerprint"] = protocol.fingerprint()
    payload["frozen_at"] = datetime.now(timezone.utc).isoformat()
    path = output_dir / f"baseline_protocol_{protocol.name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def freeze_fold_schedule(
    *,
    evaluator: WalkForwardEvaluator,
    data: pd.DataFrame,
    protocol: BaselineProtocol,
    start_date: str | None,
    output_dir: Path,
) -> tuple[
    list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    Path,
]:
    """Generate and persist frozen fold schedule."""
    folds = evaluator.generate_folds(data, start_date)
    payload_folds: list[dict[str, Any]] = []
    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds):
        payload_folds.append(
            {
                "fold_id": fold_id,
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
            }
        )

    payload = {
        "protocol_name": protocol.name,
        "start_date": start_date,
        "n_folds": len(payload_folds),
        "folds": payload_folds,
    }
    payload["fingerprint"] = _stable_payload_hash(payload)
    payload["frozen_at"] = datetime.now(timezone.utc).isoformat()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"baseline_folds_{protocol.name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return folds, path


def resolve_model_configs(
    protocol: BaselineProtocol,
    feature_columns: list[str],
) -> dict[str, tuple[type, dict[str, Any]]]:
    """Resolve baseline model classes and kwargs for a protocol."""
    model_registry: dict[str, type] = {
        "random_walk": RandomWalkBaseline,
        "ewma": EWMABaseline,
        "lightgbm": LightGBMQuantileBaseline,
    }

    resolved: dict[str, tuple[type, dict[str, Any]]] = {}
    for model in protocol.models:
        if model.key not in model_registry:
            raise ValueError(f"Unsupported model key in protocol: {model.key}")
        kwargs = model.kwargs_dict()
        if model.key == "lightgbm":
            kwargs = {**kwargs, "feature_columns": feature_columns}
        resolved[model.name] = (model_registry[model.key], kwargs)
    return resolved


def effective_profit_factor(result: BacktestResult) -> float:
    """Use net PF when present, otherwise fallback PF."""
    return result.profit_factor_net if result.profit_factor_net > 0 else result.profit_factor


def build_leaderboard(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Build a sorted baseline leaderboard from backtest results."""
    rows: list[dict[str, Any]] = []
    for model_name, result in results.items():
        rows.append(
            {
                "model": model_name,
                "sharpe_ratio": float(result.sharpe_ratio),
                "profit_factor_net": float(effective_profit_factor(result)),
                "total_return": float(result.total_return),
                "max_drawdown": float(result.max_drawdown),
                "win_rate": float(result.win_rate),
                "total_costs": float(result.total_costs),
                "num_trades": int(result.num_trades),
                "n_folds": int(len(result.fold_metrics)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        by=["sharpe_ratio", "profit_factor_net", "total_return"],
        ascending=False,
    ).reset_index(drop=True)


def write_leaderboard_artifacts(
    leaderboard: pd.DataFrame,
    output_dir: Path,
    protocol_name: str,
) -> dict[str, Path]:
    """Persist leaderboard in JSON/CSV/Markdown formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"baseline_leaderboard_{protocol_name}"
    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"
    md_path = output_dir / f"{base_name}.md"

    leaderboard.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(leaderboard.to_dict(orient="records"), fh, indent=2)
    try:
        markdown = leaderboard.to_markdown(index=False)
    except Exception:
        headers = list(leaderboard.columns)
        sep = "| " + " | ".join(["---"] * len(headers)) + " |"
        lines = ["| " + " | ".join(headers) + " |", sep]
        for _, row in leaderboard.iterrows():
            cells = [str(row[col]) for col in headers]
            lines.append("| " + " | ".join(cells) + " |")
        markdown = "\n".join(lines)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(markdown)

    return {"csv": csv_path, "json": json_path, "md": md_path}


def build_chronos_advancement_gate(
    *,
    results: dict[str, BacktestResult],
    protocol: BaselineProtocol,
) -> dict[str, Any]:
    """Build Phase 6 advancement gate payload."""
    if not results:
        return {
            "protocol": protocol.name,
            "passed": False,
            "reason": "no_baseline_results",
        }

    leaderboard = build_leaderboard(results)
    if leaderboard.empty:
        return {
            "protocol": protocol.name,
            "passed": False,
            "reason": "empty_leaderboard",
        }

    anchor_name = (
        protocol.baseline_anchor_model
        if protocol.baseline_anchor_model in results
        else str(leaderboard.iloc[0]["model"])
    )
    anchor = results[anchor_name]
    best_name = str(leaderboard.iloc[0]["model"])
    best = results[best_name]

    best_pf = effective_profit_factor(best)
    anchor_pf = effective_profit_factor(anchor)
    baseline_phase_passed = (
        best_pf > DEFAULT_SUCCESS_THRESHOLDS.min_profit_factor_net
        and best.sharpe_ratio >= DEFAULT_SUCCESS_THRESHOLDS.min_sharpe_net
    )

    gate_payload = {
        "protocol": protocol.name,
        "anchor_model": anchor_name,
        "best_baseline_model": best_name,
        "best_baseline_metrics": {
            "sharpe_ratio": float(best.sharpe_ratio),
            "profit_factor_net": float(best_pf),
            "max_drawdown": float(best.max_drawdown),
        },
        "anchor_metrics": {
            "sharpe_ratio": float(anchor.sharpe_ratio),
            "profit_factor_net": float(anchor_pf),
        },
        "baseline_phase_gate_passed": bool(baseline_phase_passed),
        "chronos_candidate_requirements": {
            "min_profit_factor_net": DEFAULT_SUCCESS_THRESHOLDS.min_profit_factor_net,
            "min_sharpe_net": DEFAULT_SUCCESS_THRESHOLDS.min_sharpe_net,
            "min_sharpe_delta_vs_anchor": DEFAULT_SUCCESS_THRESHOLDS.min_baseline_sharpe_delta,
        },
        "passed": bool(baseline_phase_passed),
    }
    gate_payload["reason"] = (
        "baseline_net_profitability_and_sharpe_criteria_passed"
        if gate_payload["passed"]
        else "baseline_criteria_not_met"
    )
    return gate_payload


def write_gate_artifact(
    gate_payload: dict[str, Any],
    output_dir: Path,
    protocol_name: str,
) -> Path:
    """Write Chronos advancement gate payload."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"chronos_advancement_gate_{protocol_name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gate_payload, fh, indent=2)
    return path
