#!/usr/bin/env python
"""Generate publication-grade plots from selected experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


SHOWCASE_RUNS = [
    ("ewma_candidate_sharpe565_retrain42_entry178_v4", "Best EWMA candidate"),
    ("ewma_candidate_span72_entry178_retrain28_v3", "Higher-trade EWMA"),
    ("ewma", "EWMA baseline"),
    ("lightgbm", "LightGBM baseline"),
]

COLORS = {
    "ink": "#172033",
    "muted": "#64748b",
    "grid": "#dbe3ea",
    "teal": "#0f766e",
    "blue": "#2563a7",
    "purple": "#7357b8",
    "red": "#b44747",
    "amber": "#b7791f",
    "surface": "#f8fafc",
}


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "font.size": 10,
            "legend.fontsize": 9,
        }
    )


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


def plot_candidate_comparison(root: Path, out_dir: Path) -> None:
    frame = _phase10_showcase_frame(root)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    sizes = frame["num_trades"].clip(lower=20) * 6
    ax.scatter(
        frame["sharpe_ratio"],
        frame["profit_factor_net"],
        s=sizes,
        c=[COLORS["teal"], COLORS["blue"], COLORS["purple"], COLORS["red"]],
        alpha=0.9,
        edgecolors="white",
        linewidths=1.0,
    )
    ax.axhline(
        1.0,
        color=COLORS["ink"],
        linestyle="--",
        linewidth=1.0,
        label="Net profit factor threshold",
    )
    ax.axvline(
        0.5,
        color=COLORS["muted"],
        linestyle=":",
        linewidth=1.0,
        label="Sharpe threshold",
    )
    offsets = {
        "Best EWMA candidate": (-118, 16),
        "Higher-trade EWMA": (-118, 18),
        "EWMA baseline": (18, -28),
        "LightGBM baseline": (12, 5),
    }
    for _, row in frame.iterrows():
        ax.annotate(
            row["label"],
            (row["sharpe_ratio"], row["profit_factor_net"]),
            xytext=offsets[row["label"]],
            textcoords="offset points",
            fontsize=9,
            color=COLORS["ink"],
            arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "linewidth": 0.8},
        )
    ax.set_title("Futures Candidate Comparison", loc="left", pad=14)
    ax.set_xlabel("Sharpe ratio")
    ax.set_ylabel("Net profit factor")
    ax.set_xlim(frame["sharpe_ratio"].min() - 0.18, frame["sharpe_ratio"].max() + 0.28)
    ax.set_ylim(0.17, 1.25)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    _savefig(out_dir / "phase10_showcase_scatter.png")


def plot_best_candidate_equity(root: Path, out_dir: Path) -> None:
    returns = pd.read_csv(root / "ewma_candidate_sharpe565_retrain42_entry178_v4" / "ewma_paper_returns.csv")
    returns["timestamp"] = pd.to_datetime(returns["timestamp"], utc=True)
    gross_curve = (1.0 + returns["gross_return"].fillna(0.0)).cumprod() - 1.0
    net_curve = (1.0 + returns["net_return"].fillna(0.0)).cumprod() - 1.0

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    ax.plot(
        returns["timestamp"],
        gross_curve * 100.0,
        label="Gross cumulative return",
        color="#94a3b8",
        linewidth=2.0,
    )
    ax.plot(
        returns["timestamp"],
        net_curve * 100.0,
        label="Net cumulative return",
        color=COLORS["teal"],
        linewidth=2.4,
    )
    ax.axhline(0.0, color=COLORS["ink"], linewidth=0.8)
    ax.set_title("Best Futures Candidate: Gross vs Net Equity", loc="left", pad=14)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Timestamp")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    _savefig(out_dir / "phase10_best_equity.png")


def plot_best_candidate_costs(root: Path, out_dir: Path) -> None:
    summary = _load_json(root / "ewma_candidate_sharpe565_retrain42_entry178_v4" / "ewma_paper_phase10_summary.json")
    metrics = summary["metrics"]
    categories = ["fees", "slippage", "funding", "interest", "other_costs"]
    values = [metrics[f"total_{name}"] * 100.0 for name in categories]
    labels = ["Fees", "Slippage", "Funding", "Interest", "Other"]

    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    bars = ax.bar(
        labels,
        values,
        color=[COLORS["teal"], COLORS["blue"], COLORS["purple"], COLORS["amber"], COLORS["muted"]],
    )
    ax.set_title("Best Futures Candidate: Execution Cost Breakdown", loc="left", pad=14)
    ax.set_ylabel("Cost contribution (% of equity)")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _savefig(out_dir / "phase10_cost_breakdown.png")


def plot_threshold_calibration(spot_csv: Path, margin_csv: Path, out_dir: Path) -> None:
    spot = pd.read_csv(spot_csv)
    margin = pd.read_csv(margin_csv)
    spot["scenario_label"] = "Spot"
    margin["scenario_label"] = "Margin"
    frame = pd.concat([spot, margin], ignore_index=True)
    frame = frame.loc[frame["active_candidate"] == True].copy()  # noqa: E712

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    palette = {"Spot": COLORS["teal"], "Margin": COLORS["amber"]}
    for label, subset in frame.groupby("scenario_label"):
        ax.scatter(
            subset["sharpe_ratio"],
            subset["profit_factor_net"],
            s=subset["num_trades"].clip(lower=20) * 0.55,
            alpha=0.75,
            c=palette[label],
            label=label,
            edgecolors="white",
            linewidths=0.8,
        )
    ax.axhline(1.0, color=COLORS["ink"], linestyle="--", linewidth=1.0)
    ax.axvline(0.5, color=COLORS["muted"], linestyle=":", linewidth=1.0)
    ax.set_title("1h Threshold Calibration: Spot vs Margin", loc="left", pad=14)
    ax.set_xlabel("Sharpe ratio")
    ax.set_ylabel("Net profit factor")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    best_spot = spot.iloc[0]
    best_margin = margin.iloc[0]
    annotations = [
        (best_spot, "Best spot", (22, 24)),
        (best_margin, "Best margin", (22, -42)),
    ]
    for row, label, offset in annotations:
        ax.annotate(
            f"{label}\nentry={row['entry_threshold']:.4f}, unc={row['uncertainty_threshold']:.2f}",
            (row["sharpe_ratio"], row["profit_factor_net"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "linewidth": 0.8},
        )
    _savefig(out_dir / "phase11_threshold_calibration.png")


def plot_campaign_equity(campaign_dir: Path, out_dir: Path) -> None:
    returns = pd.read_csv(campaign_dir / "phase11_campaign_returns.csv")
    returns["timestamp"] = pd.to_datetime(returns["timestamp"], utc=True)
    gross_curve = (1.0 + returns["gross_return"].fillna(0.0)).cumprod() - 1.0
    net_curve = (1.0 + returns["net_return"].fillna(0.0)).cumprod() - 1.0

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    ax.plot(
        returns["timestamp"],
        gross_curve * 100.0,
        label="Gross cumulative return",
        color="#94a3b8",
        linewidth=2.0,
    )
    ax.plot(
        returns["timestamp"],
        net_curve * 100.0,
        label="Net cumulative return",
        color=COLORS["red"],
        linewidth=2.4,
    )
    ax.axhline(0.0, color=COLORS["ink"], linewidth=0.8)
    ax.set_title("Fixed-Window Spot Campaign: Gross vs Net Equity", loc="left", pad=14)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Timestamp")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    _savefig(out_dir / "phase11_campaign_spot_equity.png")


def plot_campaign_kill_switch(campaign_dir: Path, out_dir: Path) -> None:
    daily = pd.read_csv(campaign_dir / "phase11_campaign_dashboard_daily.csv")
    daily["window_end"] = pd.to_datetime(daily["window_end"], utc=True)
    daily["cum_net_return"] = (1.0 + daily["net_return"].fillna(0.0)).cumprod() - 1.0
    flagged = daily.loc[daily["kill_switch_triggered"] == 1]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(
        daily["window_end"],
        daily["cum_net_return"] * 100.0,
        color="#7f1d1d",
        linewidth=2.2,
        label="Daily compounded net return",
    )
    if not flagged.empty:
        ax.scatter(
            flagged["window_end"],
            flagged["cum_net_return"] * 100.0,
            color="#dc2626",
            s=28,
            label="Kill-switch trigger day",
            zorder=3,
        )
    ax.axhline(0.0, color=COLORS["ink"], linewidth=0.8)
    ax.set_title("Campaign Monitoring: Net Return and Kill-Switch Events", loc="left", pad=14)
    ax.set_ylabel("Compounded daily net return (%)")
    ax.set_xlabel("Window end")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
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
    axes[0].set_title("Net profit factor", loc="left")
    axes[0].set_ylabel("Net profit factor")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(frame["label"], frame["num_trades"], color=colors)
    axes[1].set_title("Trade count", loc="left")
    axes[1].set_ylabel("Trades")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("4h Entry-Threshold Sensitivity", x=0.02, ha="left", fontsize=15, fontweight="semibold")
    for ax in axes:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    _savefig(out_dir / "phase11_4h_threshold_sensitivity.png")


def plot_system_architecture(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    stages = [
        (0.3, "Market data", "OHLCV · funding\nOI · liquidations · macro", COLORS["blue"]),
        (2.6, "Data contracts", "Alignment · quality gates\nfeature provenance", COLORS["teal"]),
        (4.9, "Forecasting", "Random walk · EWMA\nLightGBM · Chronos", COLORS["purple"]),
        (7.2, "Trading policy", "Quantile signals · sizing\nregime controls", COLORS["amber"]),
        (9.5, "Execution model", "Fees · slippage\nfunding · interest", COLORS["red"]),
        (11.8, "Research governance", "Walk-forward · stress tests\nkill switch · readiness", COLORS["ink"]),
    ]

    box_width = 1.9
    box_height = 1.45
    y = 2.15
    for index, (x, title, detail, color) in enumerate(stages):
        box = FancyBboxPatch(
            (x, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(box)
        title_size = 9.6 if title == "Research governance" else 10.5
        ax.text(
            x + 0.15,
            y + 1.03,
            title,
            fontsize=title_size,
            fontweight="semibold",
            color=COLORS["ink"],
        )
        ax.text(x + 0.15, y + 0.55, detail, fontsize=8.2, color=COLORS["muted"], va="center")
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + 2.27, y + box_height / 2),
                xytext=(x + box_width + 0.08, y + box_height / 2),
                arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "linewidth": 1.2},
            )

    ax.text(
        0.3,
        4.55,
        "Chronos-PLG Research Architecture",
        fontsize=17,
        fontweight="semibold",
        color=COLORS["ink"],
    )
    ax.text(
        0.3,
        4.15,
        "One evaluation path from timestamp-safe data to evidence-backed promotion decisions.",
        fontsize=10,
        color=COLORS["muted"],
    )
    ax.text(
        0.3,
        0.75,
        "Public evidence",
        fontsize=9.5,
        fontweight="semibold",
        color=COLORS["ink"],
    )
    ax.text(
        1.55,
        0.75,
        "versioned metrics · figures · experiment log · reproducibility notes",
        fontsize=9.2,
        color=COLORS["muted"],
    )
    ax.plot([0.3, 13.7], [1.12, 1.12], color=COLORS["grid"], linewidth=1.0)
    _savefig(out_dir / "system_architecture.png")


def main() -> int:
    _apply_plot_style()
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
    plot_system_architecture(args.output_dir)
    plot_candidate_comparison(args.phase10_root, args.output_dir)
    plot_best_candidate_equity(args.phase10_root, args.output_dir)
    plot_best_candidate_costs(args.phase10_root, args.output_dir)
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
