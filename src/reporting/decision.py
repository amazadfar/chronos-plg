"""Decision framework and uncertainty bands for Phase 9 reporting."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.common.metrics import DEFAULT_SUCCESS_THRESHOLDS, profit_factor_from_returns, sharpe_ratio
from src.robustness.kill_criteria import CriterionStatus, KillCriteria, KillCriteriaResult
from src.robustness.stress_tests import StressTestSuite


class DecisionOutcome(str, Enum):
    """Top-level deployment decision."""

    GO = "GO"
    ITERATE = "ITERATE"
    NO_GO = "NO_GO"


def effective_profit_factor(result: BacktestResult) -> float:
    """Use net PF when available, otherwise fallback PF."""
    return result.profit_factor_net if result.profit_factor_net > 0 else result.profit_factor


@dataclass(frozen=True)
class UncertaintyBand:
    """Simple quantile band for a metric estimate."""

    p05: float
    p50: float
    p95: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "p05": self.p05,
            "p50": self.p50,
            "p95": self.p95,
            "n": self.n,
        }


@dataclass
class DecisionReport:
    """Decision report with metrics, uncertainty, and rationale."""

    model_name: str
    outcome: DecisionOutcome
    reason: str
    primary_gate_passed: bool
    baseline_sharpe: float | None = None
    stress_pass_rate: float | None = None
    stress_threshold: float = DEFAULT_SUCCESS_THRESHOLDS.min_robustness_pass_rate
    metrics: dict[str, float] = field(default_factory=dict)
    kill_criteria: KillCriteriaResult | None = None
    uncertainty_bands: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "primary_gate_passed": self.primary_gate_passed,
            "baseline_sharpe": self.baseline_sharpe,
            "stress_pass_rate": self.stress_pass_rate,
            "stress_threshold": self.stress_threshold,
            "metrics": self.metrics,
            "uncertainty_bands": self.uncertainty_bands,
            "kill_criteria": (
                {
                    "all_passed": self.kill_criteria.all_passed,
                    "criteria": [
                        {
                            "name": c.name,
                            "status": c.status.value,
                            "value": c.value,
                            "threshold": c.threshold,
                            "message": c.message,
                        }
                        for c in self.kill_criteria.criteria
                    ],
                }
                if self.kill_criteria
                else None
            ),
            "notes": self.notes,
        }

    def to_text(self) -> str:
        lines = [
            "=" * 70,
            f"DECISION REPORT: {self.model_name}",
            "=" * 70,
            f"Outcome: {self.outcome.value}",
            f"Reason: {self.reason}",
            "",
            "Primary Metrics:",
            f"  ProfitFactorNet: {self.metrics.get('profit_factor_net', 0.0):.4f}",
            f"  SharpeNet:       {self.metrics.get('sharpe_ratio', 0.0):.4f}",
            f"  MaxDrawdown:     {self.metrics.get('max_drawdown', 0.0):.4f}",
            f"  WinRate:         {self.metrics.get('win_rate', 0.0):.4f}",
        ]
        if self.baseline_sharpe is not None:
            lines.append(f"  BaselineSharpe:  {self.baseline_sharpe:.4f}")
            lines.append(
                f"  SharpeDelta:     {self.metrics.get('sharpe_ratio', 0.0) - self.baseline_sharpe:.4f}"
            )
        if self.stress_pass_rate is not None:
            lines.append(
                f"  StressPassRate:  {self.stress_pass_rate:.1%} (threshold {self.stress_threshold:.1%})"
            )

        if self.uncertainty_bands:
            lines.append("")
            lines.append("Confidence Bands (p05/p50/p95):")
            for metric, payload in self.uncertainty_bands.items():
                fold = payload.get("fold")
                boot = payload.get("block_bootstrap")
                if fold:
                    lines.append(
                        f"  {metric} fold: {fold['p05']:.4f} / {fold['p50']:.4f} / {fold['p95']:.4f}"
                    )
                if boot:
                    lines.append(
                        f"  {metric} bootstrap: {boot['p05']:.4f} / {boot['p50']:.4f} / {boot['p95']:.4f}"
                    )

        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


def _quantile_band(values: list[float]) -> UncertaintyBand | None:
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if clean.size == 0:
        return None
    return UncertaintyBand(
        p05=float(np.quantile(clean, 0.05)),
        p50=float(np.quantile(clean, 0.50)),
        p95=float(np.quantile(clean, 0.95)),
        n=int(clean.size),
    )


def _fold_band(result: BacktestResult, metric_key: str) -> UncertaintyBand | None:
    if not result.fold_metrics:
        return None
    values: list[float] = []
    for row in result.fold_metrics:
        if metric_key in row:
            try:
                values.append(float(row[metric_key]))
            except Exception:
                continue
    return _quantile_band(values)


def _block_bootstrap_band(
    returns: pd.Series,
    metric_fn: Callable[[pd.Series], float],
    *,
    block_bars: int = 84,
    n_samples: int = 200,
    sample_fraction: float = 0.7,
    seed: int = 42,
) -> UncertaintyBand | None:
    clean = returns.dropna().astype(float)
    if clean.empty or len(clean) < max(20, block_bars):
        return None

    values = clean.to_numpy()
    n = len(values)
    block = min(max(8, block_bars), n)
    target_len = max(int(n * sample_fraction), block * 2)
    rng = np.random.default_rng(seed)

    stats: list[float] = []
    for _ in range(n_samples):
        sample_values: list[float] = []
        while len(sample_values) < target_len:
            start = int(rng.integers(0, max(1, n - block + 1)))
            sample_values.extend(values[start : start + block].tolist())
        sample = pd.Series(sample_values[:target_len])
        value = float(metric_fn(sample))
        if np.isfinite(value):
            stats.append(value)
    return _quantile_band(stats)


def compute_uncertainty_bands(result: BacktestResult) -> dict[str, dict[str, Any]]:
    """Compute fold and block-bootstrap bands for key metrics."""
    payload: dict[str, dict[str, Any]] = {}

    metric_specs: dict[str, tuple[str, Callable[[pd.Series], float]]] = {
        "sharpe_ratio": ("sharpe_ratio", sharpe_ratio),
        "profit_factor_net": ("profit_factor_net", profit_factor_from_returns),
        "total_return": ("total_return", lambda x: float((1 + x).prod() - 1)),
    }

    returns = None
    if result.returns is not None and "net_return" in result.returns.columns:
        returns = result.returns["net_return"]

    for name, (fold_key, metric_fn) in metric_specs.items():
        fold = _fold_band(result, fold_key)
        boot = _block_bootstrap_band(returns, metric_fn) if returns is not None else None
        metric_payload: dict[str, Any] = {}
        if fold is not None:
            metric_payload["fold"] = fold.to_dict()
        if boot is not None:
            metric_payload["block_bootstrap"] = boot.to_dict()
        if metric_payload:
            payload[name] = metric_payload
    return payload


def build_decision_report(
    *,
    model_name: str,
    result: BacktestResult,
    baseline_sharpe: float | None = None,
    stress_suite: StressTestSuite | None = None,
    min_stress_pass_rate: float = DEFAULT_SUCCESS_THRESHOLDS.min_robustness_pass_rate,
) -> DecisionReport:
    """Build Phase 9 GO/ITERATE/NO_GO decision report."""
    kill = KillCriteria().check(result, baseline_sharpe)
    pf_net = effective_profit_factor(result)
    sharpe = float(result.sharpe_ratio)
    max_dd = float(result.max_drawdown)
    win_rate = float(result.win_rate)
    primary_gate_passed = pf_net > DEFAULT_SUCCESS_THRESHOLDS.min_profit_factor_net

    stress_rate = stress_suite.pass_rate if stress_suite else None
    stress_ok = stress_rate is None or stress_rate >= min_stress_pass_rate

    failed_names = [
        criterion.name for criterion in kill.criteria if criterion.status == CriterionStatus.FAIL
    ]
    warning_names = [
        criterion.name for criterion in kill.criteria if criterion.status == CriterionStatus.WARNING
    ]
    severe_fail = any(
        name in {"profit_factor_net", "sharpe_net", "max_drawdown_abs"}
        for name in failed_names
    )

    notes: list[str] = []
    if not primary_gate_passed:
        notes.append(
            "Primary gate failed: ProfitFactorNet <= "
            f"{DEFAULT_SUCCESS_THRESHOLDS.min_profit_factor_net:.2f}"
        )
    if stress_rate is not None and not stress_ok:
        notes.append(
            f"Stress pass-rate {stress_rate:.1%} below threshold {min_stress_pass_rate:.1%}"
        )
    if failed_names:
        notes.append(f"Failed criteria: {', '.join(failed_names)}")
    if warning_names:
        notes.append(f"Warning criteria: {', '.join(warning_names)}")

    if not primary_gate_passed or severe_fail:
        outcome = DecisionOutcome.NO_GO
        reason = "primary_or_severe_risk_gate_failed"
    elif kill.all_passed and stress_ok:
        outcome = DecisionOutcome.GO
        reason = "all_required_gates_passed"
    else:
        outcome = DecisionOutcome.ITERATE
        reason = "partial_pass_requires_iteration"

    return DecisionReport(
        model_name=model_name,
        outcome=outcome,
        reason=reason,
        primary_gate_passed=primary_gate_passed,
        baseline_sharpe=baseline_sharpe,
        stress_pass_rate=stress_rate,
        stress_threshold=min_stress_pass_rate,
        metrics={
            "profit_factor_net": float(pf_net),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_return": float(result.total_return),
        },
        kill_criteria=kill,
        uncertainty_bands=compute_uncertainty_bands(result),
        notes=notes,
    )


def save_decision_artifacts(
    report: DecisionReport,
    *,
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    """Save decision report as JSON + plain text for monitoring ingestion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = prefix.lower().replace(" ", "_").replace("/", "_")
    json_path = output_dir / f"{safe_prefix}_decision.json"
    txt_path = output_dir / f"{safe_prefix}_decision.txt"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report.to_text())
    return {"json": json_path, "txt": txt_path}
