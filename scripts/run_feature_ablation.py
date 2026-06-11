#!/usr/bin/env python
"""Run matched baseline feature-set ablations.

The first legacy-mining benchmark compares the original/core feature space
against the new legacy-inspired `tech_*` features under the same frozen
protocol, scenario, data file, and walk-forward settings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.baseline_protocols import DEFAULT_BASELINE_PROTOCOL
from src.common.timeframe import SUPPORTED_TIMEFRAMES, default_processed_dataset_path, normalize_timeframe
from src.evaluation.phase6_baselines import FEATURE_SET_CHOICES
from src.utils import finalize_experiment_run, set_global_seed, start_experiment_run


DEFAULT_FEATURE_SETS = ("core", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-set ablation baselines")
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
        help="Optional scenario override",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start-date override",
    )
    parser.add_argument(
        "--feature-sets",
        type=str,
        default=",".join(DEFAULT_FEATURE_SETS),
        help="Comma-separated feature sets to run; choices: all, core, technical",
    )
    parser.add_argument(
        "--entry-policy",
        type=str,
        choices=["threshold", "net_edge"],
        default="threshold",
        help="Signal entry policy mode passed through to run_baselines.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results/legacy_feature_ablation",
        help="Parent output directory for ablation artifacts",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-model progress bars")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running them",
    )
    return parser.parse_args()


def _parse_feature_sets(raw: str) -> list[str]:
    selected = [part.strip() for part in raw.split(",") if part.strip()]
    if not selected:
        raise ValueError("At least one feature set is required")
    invalid = [feature_set for feature_set in selected if feature_set not in FEATURE_SET_CHOICES]
    if invalid:
        available = ", ".join(FEATURE_SET_CHOICES)
        raise ValueError(f"Unknown feature sets {invalid}. Available: {available}")
    return selected


def _load_summary(output_dir: Path, protocol_name: str) -> dict[str, Any]:
    path = output_dir / f"baseline_phase6_summary_{protocol_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_leaderboard(output_dir: Path, protocol_name: str) -> pd.DataFrame:
    path = output_dir / f"baseline_leaderboard_{protocol_name}.csv"
    return pd.read_csv(path)


def _build_comparison(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["best_sharpe_ratio", "best_profit_factor_net", "best_total_return"],
        ascending=False,
    ).reset_index(drop=True)


def _build_model_comparison(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["model", "sharpe_ratio", "profit_factor_net", "total_return"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def _to_markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except Exception:
        headers = list(frame.columns)
        sep = "| " + " | ".join(["---"] * len(headers)) + " |"
        lines = ["| " + " | ".join(headers) + " |", sep]
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)

    timeframe = normalize_timeframe(args.timeframe)
    data_path = Path(args.data) if args.data else default_processed_dataset_path(timeframe=timeframe)
    feature_sets = _parse_feature_sets(args.feature_sets)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id, manifest_path = start_experiment_run(
        script_name=Path(__file__).name,
        args={**vars(args), "resolved_data": str(data_path), "resolved_feature_sets": feature_sets},
        seed=args.seed,
        output_dir=output_root,
        project_root=Path(__file__).parent.parent,
    )

    status = "success"
    error: str | None = None
    artifacts: list[str] = []

    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        comparison_rows: list[dict[str, Any]] = []
        model_rows: list[dict[str, Any]] = []
        commands: list[list[str]] = []
        for feature_set in feature_sets:
            run_output_dir = output_root / feature_set
            command = [
                sys.executable,
                "scripts/run_baselines.py",
                "--timeframe",
                timeframe,
                "--data",
                str(data_path),
                "--protocol",
                args.protocol,
                "--feature-set",
                feature_set,
                "--entry-policy",
                args.entry_policy,
                "--output-dir",
                str(run_output_dir),
                "--seed",
                str(args.seed),
            ]
            if args.scenario:
                command.extend(["--scenario", args.scenario])
            if args.start_date:
                command.extend(["--start-date", args.start_date])
            if args.no_progress:
                command.append("--no-progress")

            commands.append(command)
            if args.dry_run:
                continue

            subprocess.run(command, check=True)
            summary = _load_summary(run_output_dir, args.protocol)
            feature_summary = summary["feature_summary"]
            if feature_set == "technical" and feature_summary["technical"] == 0:
                raise ValueError(
                    "feature_set=technical selected zero technical features. "
                    "Rebuild the processed dataset so tech_* columns are present."
                )
            if (
                feature_set == "all"
                and "core" in feature_sets
                and feature_summary["technical"] == 0
            ):
                raise ValueError(
                    "feature_set=all selected zero technical features, so core-vs-all "
                    "ablation would be invalid. Rebuild the processed dataset first."
                )

            leaderboard = _load_leaderboard(run_output_dir, args.protocol)
            top = leaderboard.iloc[0].to_dict()
            comparison_rows.append(
                {
                    "feature_set": feature_set,
                    "feature_total": feature_summary["total"],
                    "feature_core": feature_summary["core"],
                    "feature_technical": feature_summary["technical"],
                    "best_model": top["model"],
                    "best_sharpe_ratio": float(top["sharpe_ratio"]),
                    "best_profit_factor_net": float(top["profit_factor_net"]),
                    "best_total_return": float(top["total_return"]),
                    "best_max_drawdown": float(top["max_drawdown"]),
                    "best_num_trades": int(top["num_trades"]),
                    "chronos_gate_passed": bool(summary["chronos_gate"]["passed"]),
                    "run_output_dir": str(run_output_dir),
                }
            )
            for _, row in leaderboard.iterrows():
                model_rows.append(
                    {
                        "feature_set": feature_set,
                        "feature_total": feature_summary["total"],
                        "feature_core": feature_summary["core"],
                        "feature_technical": feature_summary["technical"],
                        "model": row["model"],
                        "sharpe_ratio": float(row["sharpe_ratio"]),
                        "profit_factor_net": float(row["profit_factor_net"]),
                        "total_return": float(row["total_return"]),
                        "max_drawdown": float(row["max_drawdown"]),
                        "win_rate": float(row["win_rate"]),
                        "total_costs": float(row["total_costs"]),
                        "num_trades": int(row["num_trades"]),
                        "n_folds": int(row["n_folds"]),
                        "run_output_dir": str(run_output_dir),
                    }
                )
            artifacts.extend(
                [
                    str(run_output_dir / f"baseline_phase6_summary_{args.protocol}.json"),
                    str(run_output_dir / f"baseline_leaderboard_{args.protocol}.csv"),
                    str(run_output_dir / f"baseline_leaderboard_{args.protocol}.md"),
                ]
            )

        plan_path = output_root / "feature_ablation_commands.json"
        plan_path.write_text(json.dumps({"commands": commands}, indent=2), encoding="utf-8")
        artifacts.append(str(plan_path))

        if args.dry_run:
            print(json.dumps({"commands": commands}, indent=2))
            return 0

        comparison = _build_comparison(comparison_rows)
        comparison_csv = output_root / "feature_ablation_comparison.csv"
        comparison_json = output_root / "feature_ablation_comparison.json"
        comparison_md = output_root / "feature_ablation_comparison.md"
        comparison.to_csv(comparison_csv, index=False)
        comparison_json.write_text(
            json.dumps(comparison.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
        comparison_md.write_text(_to_markdown(comparison), encoding="utf-8")
        artifacts.extend([str(comparison_csv), str(comparison_json), str(comparison_md)])

        model_comparison = _build_model_comparison(model_rows)
        model_comparison_csv = output_root / "feature_ablation_model_comparison.csv"
        model_comparison_json = output_root / "feature_ablation_model_comparison.json"
        model_comparison_md = output_root / "feature_ablation_model_comparison.md"
        model_comparison.to_csv(model_comparison_csv, index=False)
        model_comparison_json.write_text(
            json.dumps(model_comparison.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
        model_comparison_md.write_text(_to_markdown(model_comparison), encoding="utf-8")
        artifacts.extend(
            [str(model_comparison_csv), str(model_comparison_json), str(model_comparison_md)]
        )

        print("\nFEATURE ABLATION COMPARISON")
        print("=" * 80)
        print(comparison.to_string(index=False))
        print("\nFEATURE ABLATION MODEL COMPARISON")
        print("=" * 80)
        print(model_comparison.to_string(index=False))
        return 0
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
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
