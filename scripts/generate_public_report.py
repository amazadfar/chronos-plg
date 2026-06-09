#!/usr/bin/env python
"""Generate curated publication summaries from existing experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SHOWCASE_RUNS = [
    ("ewma_candidate_sharpe565_retrain42_entry178_v4", "Best EWMA candidate"),
    ("ewma_candidate_span72_entry178_retrain28_v3", "Higher-trade EWMA"),
    ("ewma", "EWMA baseline"),
    ("lightgbm", "LightGBM baseline"),
]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)\n"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        headers = list(frame.columns)
        sep = "| " + " | ".join(["---"] * len(headers)) + " |"
        lines = ["| " + " | ".join(headers) + " |", sep]
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join(lines)


def _phase10_showcase_table(phase10_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, candidate_label in SHOWCASE_RUNS:
        path = phase10_root / run_name / f"{run_name.split('_')[0]}_paper_phase10_summary.json"
        if run_name.startswith("lightgbm"):
            path = phase10_root / run_name / "lightgbm_paper_phase10_summary.json"
        elif run_name == "ewma":
            path = phase10_root / run_name / "ewma_paper_phase10_summary.json"
        else:
            path = phase10_root / run_name / "ewma_paper_phase10_summary.json"
        data = _load_json(path)
        metrics = data["metrics"]
        decision = data["decision"]
        rows.append(
            {
                "candidate": candidate_label,
                "run_id": run_name,
                "profit_factor_net": metrics["profit_factor_net"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "num_trades": metrics["num_trades"],
                "total_return_pct": metrics["total_return"] * 100.0,
                "max_drawdown_pct": metrics["max_drawdown"] * 100.0,
                "total_costs_pct": metrics["total_costs"] * 100.0,
                "decision": decision["outcome"],
                "decision_reason": decision["reason"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        by=["profit_factor_net", "sharpe_ratio", "num_trades"],
        ascending=[False, False, False],
    )
    return frame.reset_index(drop=True)


def _public_candidate_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "candidate",
            "profit_factor_net",
            "sharpe_ratio",
            "num_trades",
            "total_return_pct",
            "decision",
        ]
    ].rename(
        columns={
            "candidate": "Candidate",
            "profit_factor_net": "Net PF",
            "sharpe_ratio": "Sharpe",
            "num_trades": "Trades",
            "total_return_pct": "Total return (%)",
            "decision": "Decision",
        }
    )


def _public_calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "scenario",
            "entry_threshold",
            "uncertainty_threshold",
            "profit_factor_net",
            "sharpe_ratio",
            "num_trades",
            "kill_event_rate",
        ]
    ].rename(
        columns={
            "scenario": "Scenario",
            "entry_threshold": "Entry threshold",
            "uncertainty_threshold": "Uncertainty threshold",
            "profit_factor_net": "Net PF",
            "sharpe_ratio": "Sharpe",
            "num_trades": "Trades",
            "kill_event_rate": "Kill-event rate",
        }
    )


def _public_sensitivity_table(frame: pd.DataFrame) -> pd.DataFrame:
    public = frame.copy()
    public["scenario"] = public["scenario"].str.title()
    public["policy"] = public["policy"].str.replace("_", " ", regex=False).str.title()
    public["deployment_reason"] = (
        public["deployment_reason"]
        .str.replace("_", " ", regex=False)
        .str.replace(";", "; ", regex=False)
    )
    return public.rename(
        columns={
            "scenario": "Scenario",
            "policy": "Policy",
            "profit_factor_net": "Net PF",
            "sharpe_ratio": "Sharpe",
            "num_trades": "Trades",
            "kill_events": "Kill events",
            "deployment_ready": "Ready",
            "deployment_reason": "Readiness reason",
        }
    )


def _top_calibration_table(path: Path, scenario_label: str, top_n: int = 5) -> pd.DataFrame:
    frame = pd.read_csv(path)
    active = frame.loc[frame["active_candidate"] == True].copy()  # noqa: E712
    if active.empty:
        return active
    keep = [
        "candidate_id",
        "entry_threshold",
        "uncertainty_threshold",
        "profit_factor_net",
        "sharpe_ratio",
        "num_trades",
        "kill_event_rate",
        "max_drawdown_abs",
        "composite_score",
    ]
    active = active.loc[:, keep].head(top_n).copy()
    active.insert(1, "scenario", scenario_label)
    return active


def _campaign_snapshot(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    metrics = data["metrics"]
    readiness = data["deployment_readiness"]
    promotion = data["promotion_recommendation"]
    completion = data["completion_gate"]
    return {
        "selection_mode": data["selection_mode"],
        "campaign_window": {
            "start": data["campaign_start_date"],
            "end": data["campaign_end_date"],
        },
        "metrics": metrics,
        "deployment_readiness": readiness,
        "promotion_recommendation": promotion,
        "completion_gate": completion,
    }


def _threshold_sensitivity_row(path: Path, *, scenario: str, policy_label: str) -> dict[str, Any]:
    data = _load_json(path)
    metrics = data["metrics"]
    readiness = data["deployment_readiness"]
    return {
        "scenario": scenario,
        "policy": policy_label,
        "profit_factor_net": metrics["profit_factor_net"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "num_trades": metrics["num_trades"],
        "kill_events": len(data.get("kill_switch_events", [])),
        "deployment_ready": readiness["ready"],
        "deployment_reason": readiness["reason"],
    }


def build_report(
    *,
    phase10_root: Path,
    spot_calibration_path: Path,
    margin_calibration_path: Path,
    campaign_summary_path: Path,
    four_hour_sensitivity: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase10 = _phase10_showcase_table(phase10_root)
    spot = _top_calibration_table(
        spot_calibration_path,
        scenario_label="spot",
        top_n=5,
    )
    margin = _top_calibration_table(
        margin_calibration_path,
        scenario_label="margin",
        top_n=5,
    )
    campaign = _campaign_snapshot(campaign_summary_path)

    snapshot = {
        "project_positioning": "governed probabilistic trading research platform",
        "phase10_showcase": phase10.to_dict(orient="records"),
        "phase11_calibration_spot_top": spot.to_dict(orient="records"),
        "phase11_calibration_margin_top": margin.to_dict(orient="records"),
        "phase11_campaign_spot": campaign,
        "phase11_4h_threshold_sensitivity": four_hour_sensitivity,
        "key_findings": [
            "The strongest observed futures EWMA candidate produced PF Net above 1.0 and positive Sharpe, but still failed promotion gates.",
            "The 1h spot and margin threshold calibration campaigns increased trade count but remained below profitability and Sharpe thresholds.",
            "A matched 4h default-threshold replay produced zero trades, while a looser 4h threshold recovered PF-positive and positive-Sharpe regions but still failed readiness because kill-switch events remained elevated.",
            "Governance and kill-switch layers are doing useful work: they are blocking promotion despite partial positive edge regions.",
        ],
    }

    four_hour_frame = pd.DataFrame(four_hour_sensitivity)
    candidate_public = _public_candidate_table(phase10)
    spot_public = _public_calibration_table(spot)
    margin_public = _public_calibration_table(margin)
    sensitivity_public = _public_sensitivity_table(four_hour_frame)
    selection_mode = campaign["selection_mode"].replace("_", " ")
    readiness_reason = campaign["deployment_readiness"]["reason"].replace("_", " ").replace(";", "; ")

    report = f"""# Public Evidence Snapshot

This file is generated from selected experiment artifacts and is intended for public publication.

## Project Positioning

`chronos-plg` should be read as a research platform, not as a finished live-trading system.

## Futures Candidate Comparison

{_markdown_table(candidate_public)}

## Spot Threshold Calibration: Top Active Candidates

{_markdown_table(spot_public)}

## Margin Threshold Calibration: Top Active Candidates

{_markdown_table(margin_public)}

## Fixed Campaign Snapshot

- Campaign window: `{campaign['campaign_window']['start']}` to `{campaign['campaign_window']['end']}`
- Selection mode: `{selection_mode}`
- Profit factor net: `{campaign['metrics']['profit_factor_net']:.4f}`
- Sharpe ratio: `{campaign['metrics']['sharpe_ratio']:.4f}`
- Trades: `{campaign['metrics']['num_trades']}`
- Total return: `{campaign['metrics']['total_return'] * 100.0:.2f}%`
- Kill events: `{campaign['metrics']['kill_events']}`
- Deployment ready: `{campaign['deployment_readiness']['ready']}`
- Readiness reason: `{readiness_reason}`
- Promotion recommended: `{campaign['promotion_recommendation']['recommend_promotion']}`
- Completion gate passed: `{campaign['completion_gate']['passed']}`

## 4h Threshold Sensitivity Check

{_markdown_table(sensitivity_public)}

## Interpretation

- The best observed futures configuration demonstrates that the system can find a positive net-edge region under strict cost accounting.
- That positive edge is still too fragile for promotion because regime stability, trade-count sufficiency, and kill-switch behavior remain binding constraints.
- The 1h spot and margin calibration runs show the opposite failure mode: higher trade count, but consistently weak PF / Sharpe and no acceptable candidates.
- The 4h spot and margin default-threshold runs were too conservative to trade at all over the inspected window, but a looser 4h threshold recovered attractive PF / Sharpe regions that still failed readiness due to kill-switch activity.
- This is the kind of evidence that makes the project interesting publicly: the governance layer is rejecting attractive-looking but not deployment-ready configurations.
"""
    return snapshot, report, phase10, spot, margin


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate public-facing artifact summaries")
    parser.add_argument(
        "--phase10-root",
        type=Path,
        default=Path("data/results/phase10_real_20260217"),
        help="Root directory containing phase 10 summary artifacts",
    )
    parser.add_argument(
        "--spot-calibration",
        type=Path,
        default=Path("data/results/phase11_5_sweep_spot_threshold_calib/phase11_sweep_ranked.csv"),
        help="Phase 11 spot threshold calibration ranked CSV",
    )
    parser.add_argument(
        "--margin-calibration",
        type=Path,
        default=Path("data/results/phase11_5_sweep_margin_threshold_calib/phase11_sweep_ranked.csv"),
        help="Phase 11 margin threshold calibration ranked CSV",
    )
    parser.add_argument(
        "--campaign-summary",
        type=Path,
        default=Path("data/results/phase11_7_campaign_spot_one/phase11_campaign_summary.json"),
        help="Fixed campaign summary JSON",
    )
    parser.add_argument(
        "--spot-4h-default",
        type=Path,
        default=Path("data/results/publication_4h_spot_threshold/ewma_paper_phase10_summary.json"),
        help="4h spot default-threshold summary JSON",
    )
    parser.add_argument(
        "--margin-4h-default",
        type=Path,
        default=Path("data/results/publication_4h_margin_threshold/ewma_paper_phase10_summary.json"),
        help="4h margin default-threshold summary JSON",
    )
    parser.add_argument(
        "--spot-4h-looser",
        type=Path,
        default=Path("data/results/publication_4h_spot_threshold_looser/ewma_paper_phase10_summary.json"),
        help="4h spot looser-threshold summary JSON",
    )
    parser.add_argument(
        "--margin-4h-looser",
        type=Path,
        default=Path("data/results/publication_4h_margin_threshold_looser/ewma_paper_phase10_summary.json"),
        help="4h margin looser-threshold summary JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/public"),
        help="Directory for generated public summaries",
    )
    parser.add_argument(
        "--docs-output-dir",
        type=Path,
        default=Path("docs/generated"),
        help="Directory for Pages-friendly mirrored summary files",
    )
    args = parser.parse_args()

    four_hour_sensitivity = [
        _threshold_sensitivity_row(args.spot_4h_default, scenario="spot", policy_label="default_threshold"),
        _threshold_sensitivity_row(args.spot_4h_looser, scenario="spot", policy_label="looser_threshold"),
        _threshold_sensitivity_row(args.margin_4h_default, scenario="margin", policy_label="default_threshold"),
        _threshold_sensitivity_row(args.margin_4h_looser, scenario="margin", policy_label="looser_threshold"),
    ]

    snapshot, report, phase10, spot, margin = build_report(
        phase10_root=args.phase10_root,
        spot_calibration_path=args.spot_calibration,
        margin_calibration_path=args.margin_calibration,
        campaign_summary_path=args.campaign_summary,
        four_hour_sensitivity=four_hour_sensitivity,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_output_dir.mkdir(parents=True, exist_ok=True)
    phase10.to_csv(args.output_dir / "phase10_showcase_leaderboard.csv", index=False)
    spot.to_csv(args.output_dir / "phase11_spot_threshold_top.csv", index=False)
    margin.to_csv(args.output_dir / "phase11_margin_threshold_top.csv", index=False)
    pd.DataFrame(four_hour_sensitivity).to_csv(args.output_dir / "phase11_4h_threshold_sensitivity.csv", index=False)
    _write_json(args.output_dir / "public_evidence_snapshot.json", snapshot)
    _write_text(args.output_dir / "public_evidence_snapshot.md", report)
    _write_text(args.docs_output_dir / "public-evidence-snapshot.md", report)
    _write_json(args.docs_output_dir / "public-evidence-snapshot.json", snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
