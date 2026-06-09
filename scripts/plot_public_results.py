#!/usr/bin/env python
"""Generate publication-grade plots from selected experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SHOWCASE_RUNS = [
    ("ewma_candidate_sharpe565_retrain42_entry178_v4", "EWMA best futures candidate"),
    ("ewma_candidate_span72_entry178_retrain28_v3", "EWMA higher-trade candidate"),
    ("ewma", "EWMA baseline"),
    ("lightgbm", "LightGBM baseline"),
]


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _phase10_showcase_frame(root: Path) -> pd.DataFrame:
    rows = []
    for run_name, label in SHOWCASE_RUNS:
        summary_name = "lightgbm_paper_phase10_summary.json" if run_name == "lightgbm" else "ewma_paper_phase10_summary.json"
        data = _load_json(root / run_name / summary_name)
        metrics = data["metrics"]
        rows.append(
            {
                "run": run_name,
                "label": label,
                "profit_factor_net": metrics["profit_factor_net"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "num_trades": metrics["num_trades"],
                "total_return": metrics["total_return"],
            }
        )
    return pd.DataFrame(rows)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_phase10_scatter(root: Path, out_dir: Path) -> None:
    frame = _phase10_showcase_frame(root)
    plt.figure(figsize=(9, 6))
    sizes = frame["num_trades"].clip(lower=20) * 6
    plt.scatter(
        frame["sharpe_ratio"],
        frame["profit_factor_net"],
        s=sizes,
        c=["#0f766e", "#0284c7", "#7c3aed", "#b91c1c"],
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )
    plt.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0, label="PF net = 1.0")
    plt.axvline(0.5, color="#6b7280", linestyle=":", linewidth=1.0, label="Sharpe = 0.5")
    for _, row in frame.iterrows():
        plt.annotate(row["label"], (row["sharpe_ratio"], row["profit_factor_net"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    plt.title("Phase 10 Showcase Runs")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Profit Factor (Net)")
    plt.legend(frameon=False, loc="lower right")
    _savefig(out_dir / "phase10_showcase_scatter.png")


def plot_phase10_equity(root: Path, out_dir: Path) -> None:
    returns = pd.read_csv(root / "ewma_candidate_sharpe565_retrain42_entry178_v4" / "ewma_paper_returns.csv")
    returns["timestamp"] = pd.to_datetime(returns["timestamp"], utc=True)
    gross_curve = (1.0 + returns["gross_return"].fillna(0.0)).cumprod() - 1.0
    net_curve = (1.0 + returns["net_return"].fillna(0.0)).cumprod() - 1.0

    plt.figure(figsize=(10, 5.5))
    plt.plot(returns["timestamp"], gross_curve * 100.0, label="Gross cumulative return", color="#94a3b8", linewidth=2.0)
    plt.plot(returns["timestamp"], net_curve * 100.0, label="Net cumulative return", color="#0f766e", linewidth=2.2)
    plt.axhline(0.0, color="#111827", linewidth=0.8)
    plt.title("Best Phase 10 Futures Candidate: Gross vs Net Equity")
    plt.ylabel("Cumulative return (%)")
    plt.xlabel("Timestamp")
    plt.legend(frameon=False)
    _savefig(out_dir / "phase10_best_equity.png")


def plot_phase10_costs(root: Path, out_dir: Path) -> None:
    summary = _load_json(root / "ewma_candidate_sharpe565_retrain42_entry178_v4" / "ewma_paper_phase10_summary.json")
    metrics = summary["metrics"]
    categories = ["fees", "slippage", "funding", "interest", "other_costs"]
    values = [metrics[f"total_{name}"] * 100.0 for name in categories]
    labels = ["Fees", "Slippage", "Funding", "Interest", "Other"]

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(labels, values, color=["#0f766e", "#0284c7", "#7c3aed", "#f59e0b", "#64748b"])
    plt.title("Best Phase 10 Futures Candidate: Cost Breakdown")
    plt.ylabel("Cost contribution (% of equity)")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    _savefig(out_dir / "phase10_cost_breakdown.png")


def plot_threshold_calibration(spot_csv: Path, margin_csv: Path, out_dir: Path) -> None:
    spot = pd.read_csv(spot_csv)
    margin = pd.read_csv(margin_csv)
    spot["scenario_label"] = "Spot"
    margin["scenario_label"] = "Margin"
    frame = pd.concat([spot, margin], ignore_index=True)
    frame = frame.loc[frame["active_candidate"] == True].copy()  # noqa: E712

    plt.figure(figsize=(9.5, 6.5))
    palette = {"Spot": "#0f766e", "Margin": "#b45309"}
    for label, subset in frame.groupby("scenario_label"):
        plt.scatter(
            subset["sharpe_ratio"],
            subset["profit_factor_net"],
            s=subset["num_trades"].clip(lower=20) * 0.55,
            alpha=0.75,
            c=palette[label],
            label=label,
            edgecolors="black",
            linewidths=0.4,
        )
    plt.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    plt.axvline(0.5, color="#6b7280", linestyle=":", linewidth=1.0)
    plt.title("Phase 11 Threshold Calibration: 1h Spot vs Margin")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Profit Factor (Net)")
    plt.legend(frameon=False)

    best_spot = spot.iloc[0]
    best_margin = margin.iloc[0]
    for row, label in [(best_spot, "Best spot"), (best_margin, "Best margin")]:
        plt.annotate(
            f"{label}\nentry={row['entry_threshold']:.4f}, unc={row['uncertainty_threshold']:.2f}",
            (row["sharpe_ratio"], row["profit_factor_net"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
    _savefig(out_dir / "phase11_threshold_calibration.png")


def plot_campaign_equity(campaign_dir: Path, out_dir: Path) -> None:
    returns = pd.read_csv(campaign_dir / "phase11_campaign_returns.csv")
    returns["timestamp"] = pd.to_datetime(returns["timestamp"], utc=True)
    gross_curve = (1.0 + returns["gross_return"].fillna(0.0)).cumprod() - 1.0
    net_curve = (1.0 + returns["net_return"].fillna(0.0)).cumprod() - 1.0

    plt.figure(figsize=(10, 5.5))
    plt.plot(returns["timestamp"], gross_curve * 100.0, label="Gross cumulative return", color="#94a3b8", linewidth=2.0)
    plt.plot(returns["timestamp"], net_curve * 100.0, label="Net cumulative return", color="#b91c1c", linewidth=2.2)
    plt.axhline(0.0, color="#111827", linewidth=0.8)
    plt.title("Phase 11 Fixed Campaign: Spot Candidate Equity")
    plt.ylabel("Cumulative return (%)")
    plt.xlabel("Timestamp")
    plt.legend(frameon=False)
    _savefig(out_dir / "phase11_campaign_spot_equity.png")


def plot_campaign_kill_switch(campaign_dir: Path, out_dir: Path) -> None:
    daily = pd.read_csv(campaign_dir / "phase11_campaign_dashboard_daily.csv")
    daily["window_end"] = pd.to_datetime(daily["window_end"], utc=True)
    daily["cum_net_return"] = (1.0 + daily["net_return"].fillna(0.0)).cumprod() - 1.0
    flagged = daily.loc[daily["kill_switch_triggered"] == 1]

    plt.figure(figsize=(10, 5.2))
    plt.plot(daily["window_end"], daily["cum_net_return"] * 100.0, color="#7f1d1d", linewidth=2.0, label="Daily compounded net return")
    if not flagged.empty:
        plt.scatter(
            flagged["window_end"],
            flagged["cum_net_return"] * 100.0,
            color="#dc2626",
            s=28,
            label="Kill-switch trigger day",
            zorder=3,
        )
    plt.axhline(0.0, color="#111827", linewidth=0.8)
    plt.title("Phase 11 Campaign: Daily Net Return Path with Kill Triggers")
    plt.ylabel("Compounded daily net return (%)")
    plt.xlabel("Window end")
    plt.legend(frameon=False)
    _savefig(out_dir / "phase11_campaign_kill_switch.png")


def plot_four_hour_threshold_sensitivity(
    spot_default: Path,
    margin_default: Path,
    spot_looser: Path,
    margin_looser: Path,
    out_dir: Path,
) -> None:
    rows = []
    for path, scenario, policy in [
        (spot_default, "Spot", "Default"),
        (spot_looser, "Spot", "Looser"),
        (margin_default, "Margin", "Default"),
        (margin_looser, "Margin", "Looser"),
    ]:
        data = _load_json(path)
        metrics = data["metrics"]
        rows.append(
            {
                "label": f"{scenario} {policy}",
                "profit_factor_net": metrics["profit_factor_net"],
                "num_trades": metrics["num_trades"],
                "kill_events": len(data.get("kill_switch_events", [])),
            }
        )
    frame = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    colors = ["#94a3b8", "#0f766e", "#cbd5e1", "#b45309"]

    axes[0].bar(frame["label"], frame["profit_factor_net"], color=colors)
    axes[0].axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[0].set_title("4h threshold sensitivity: PF Net")
    axes[0].set_ylabel("Profit Factor (Net)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(frame["label"], frame["num_trades"], color=colors)
    axes[1].set_title("4h threshold sensitivity: trade count")
    axes[1].set_ylabel("Trades")
    axes[1].tick_params(axis="x", rotation=20)

    _savefig(out_dir / "phase11_4h_threshold_sensitivity.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate public plot set from result artifacts")
    parser.add_argument(
        "--phase10-root",
        type=Path,
        default=Path("data/results/phase10_real_20260217"),
        help="Root directory containing phase 10 artifacts",
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
        "--campaign-dir",
        type=Path,
        default=Path("data/results/phase11_7_campaign_spot_one"),
        help="Directory containing phase 11 fixed campaign artifacts",
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
        default=Path("docs/assets"),
        help="Directory for generated plot images",
    )
    args = parser.parse_args()

    _ensure_dir(args.output_dir)
    plot_phase10_scatter(args.phase10_root, args.output_dir)
    plot_phase10_equity(args.phase10_root, args.output_dir)
    plot_phase10_costs(args.phase10_root, args.output_dir)
    plot_threshold_calibration(args.spot_calibration, args.margin_calibration, args.output_dir)
    plot_campaign_equity(args.campaign_dir, args.output_dir)
    plot_campaign_kill_switch(args.campaign_dir, args.output_dir)
    plot_four_hour_threshold_sensitivity(
        args.spot_4h_default,
        args.margin_4h_default,
        args.spot_4h_looser,
        args.margin_4h_looser,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
