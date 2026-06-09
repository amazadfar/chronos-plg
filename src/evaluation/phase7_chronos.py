"""Phase 7 Chronos/meta validation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.common.metrics import (
    DEFAULT_SUCCESS_THRESHOLDS,
    SuccessThresholds,
    profit_factor_from_returns,
    sharpe_ratio,
)
from src.evaluation.phase6_baselines import effective_profit_factor


def determine_recent_regime_start(
    index: pd.DatetimeIndex,
    *,
    anchor_date: str = "2024-01-01",
    recent_fraction: float = 0.25,
) -> pd.Timestamp:
    """
    Choose start timestamp for recent-regime stress split.

    Preference:
    1. Use calendar anchor (2024-01-01) when present in available data.
    2. Otherwise use trailing ``recent_fraction`` of available window.
    """
    if index.empty:
        raise ValueError("Cannot determine recent regime split on empty index")
    if recent_fraction <= 0 or recent_fraction >= 1:
        raise ValueError("recent_fraction must be between 0 and 1")

    ordered = index.sort_values()
    tz = ordered.tz
    anchor = pd.Timestamp(anchor_date, tz=tz) if tz is not None else pd.Timestamp(anchor_date)
    if ordered.min() <= anchor <= ordered.max():
        return anchor

    split_pos = max(0, int(len(ordered) * (1 - recent_fraction)))
    split_pos = min(split_pos, len(ordered) - 1)
    return ordered[split_pos]


def compute_recent_regime_metrics(
    result: BacktestResult,
    recent_start: pd.Timestamp,
) -> dict[str, Any]:
    """Compute early vs recent performance metrics from net returns."""
    if result.returns is None or "net_return" not in result.returns.columns:
        return {
            "recent_start": recent_start.isoformat(),
            "n_early": 0,
            "n_recent": 0,
            "early_sharpe": 0.0,
            "recent_sharpe": 0.0,
            "early_profit_factor_net": 0.0,
            "recent_profit_factor_net": 0.0,
            "recent_sharpe_ratio_vs_early": 0.0,
        }

    net = result.returns["net_return"].dropna()
    early = net[net.index < recent_start]
    recent = net[net.index >= recent_start]

    early_sharpe = sharpe_ratio(early)
    recent_sharpe = sharpe_ratio(recent)
    early_pf = profit_factor_from_returns(early)
    recent_pf = profit_factor_from_returns(recent)

    if early_sharpe <= 0:
        ratio = 1.0 if recent_sharpe >= 0 else 0.0
    else:
        ratio = recent_sharpe / early_sharpe

    return {
        "recent_start": recent_start.isoformat(),
        "n_early": int(len(early)),
        "n_recent": int(len(recent)),
        "early_sharpe": float(early_sharpe),
        "recent_sharpe": float(recent_sharpe),
        "early_profit_factor_net": float(early_pf),
        "recent_profit_factor_net": float(recent_pf),
        "recent_sharpe_ratio_vs_early": float(ratio),
    }


def summarize_chronos_provenance(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize Chronos runtime provenance events into a gate-friendly payload."""
    payloads = [dict(event) for event in events]
    latest = payloads[-1] if payloads else {}

    def _uniq(key: str) -> list[str]:
        values: list[str] = []
        for row in payloads:
            value = row.get(key)
            if value is None:
                continue
            values.append(str(value))
        return sorted(set(values))

    fallback_active = any(bool(row.get("fallback_active")) for row in payloads)
    fallback_reasons = [
        str(row.get("fallback_reason"))
        for row in payloads
        if row.get("fallback_reason")
    ]

    return {
        "event_count": int(len(payloads)),
        "fit_event_count": int(sum(1 for row in payloads if row.get("event") == "fit")),
        "predict_fallback_event_count": int(
            sum(1 for row in payloads if row.get("event") == "predict_fallback")
        ),
        "model_ids": _uniq("model_id"),
        "backends": _uniq("backend"),
        "chronos_versions": _uniq("chronos_version"),
        "torch_versions": _uniq("torch_version"),
        "requested_devices": _uniq("requested_device"),
        "resolved_devices": _uniq("resolved_device"),
        "fallback_active": bool(fallback_active),
        "fallback_reasons": sorted(set(fallback_reasons)),
        "latest_model_id": latest.get("model_id"),
        "latest_backend": latest.get("backend"),
        "latest_chronos_version": latest.get("chronos_version"),
        "latest_torch_version": latest.get("torch_version"),
    }


def _calibration_payload(
    frame: pd.DataFrame,
    *,
    quantiles: tuple[float, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "quantiles": {},
        "mean_interval_width": None,
    }

    actual = pd.to_numeric(frame.get("actual"), errors="coerce")
    for q in quantiles:
        col = f"q{int(q * 100)}"
        pred = (
            pd.to_numeric(frame[col], errors="coerce")
            if col in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        valid = actual.notna() & pred.notna()
        n = int(valid.sum())
        if n == 0:
            out["quantiles"][col] = {
                "target": float(q),
                "n": 0,
                "coverage": None,
                "calibration_error": None,
            }
            continue

        coverage = float((actual.loc[valid] < pred.loc[valid]).mean())
        out["quantiles"][col] = {
            "target": float(q),
            "n": n,
            "coverage": coverage,
            "calibration_error": float(abs(coverage - q)),
        }

    if {"q10", "q90"}.issubset(frame.columns):
        width = pd.to_numeric(frame["q90"], errors="coerce") - pd.to_numeric(
            frame["q10"], errors="coerce"
        )
        width = width.replace([np.inf, -np.inf], np.nan).dropna()
        if not width.empty:
            out["mean_interval_width"] = float(width.mean())

    return out


def compute_quantile_calibration_by_regime(
    *,
    predictions: pd.DataFrame,
    actual_returns: pd.Series,
    regime_column: str = "regime",
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
    min_samples_per_regime: int = 24,
) -> dict[str, Any]:
    """Compute quantile coverage/calibration overall and by detected regime."""
    if predictions.empty:
        return {
            "n_rows": 0,
            "regime_column": regime_column,
            "min_samples_per_regime": int(min_samples_per_regime),
            "overall": _calibration_payload(
                pd.DataFrame(columns=["actual"]),
                quantiles=quantiles,
            ),
            "by_regime": {},
        }

    frame = predictions.copy()
    frame["actual"] = pd.to_numeric(actual_returns.reindex(frame.index), errors="coerce")
    if regime_column not in frame.columns:
        frame[regime_column] = "unknown"
    frame[regime_column] = frame[regime_column].fillna("unknown").astype(str)

    by_regime: dict[str, Any] = {}
    for regime_name, regime_frame in frame.groupby(regime_column):
        payload = _calibration_payload(regime_frame, quantiles=quantiles)
        payload["eligible"] = bool(len(regime_frame) >= min_samples_per_regime)
        if not payload["eligible"]:
            payload["reason"] = "insufficient_samples"
        by_regime[str(regime_name)] = payload

    return {
        "n_rows": int(len(frame)),
        "regime_column": regime_column,
        "min_samples_per_regime": int(min_samples_per_regime),
        "overall": _calibration_payload(frame, quantiles=quantiles),
        "by_regime": by_regime,
    }


def build_phase7_candidate_gate(
    *,
    candidate_name: str,
    candidate_result: BacktestResult,
    anchor_name: str,
    anchor_result: BacktestResult,
    recent_metrics: dict[str, Any],
    chronos_provenance: dict[str, Any] | None = None,
    allow_fallback_candidate: bool = False,
    thresholds: SuccessThresholds = DEFAULT_SUCCESS_THRESHOLDS,
) -> dict[str, Any]:
    """Build gate payload for Chronos/meta advancement checks."""
    candidate_pf = effective_profit_factor(candidate_result)
    anchor_pf = effective_profit_factor(anchor_result)
    sharpe_delta = candidate_result.sharpe_ratio - anchor_result.sharpe_ratio
    recent_ratio = float(recent_metrics.get("recent_sharpe_ratio_vs_early", 0.0))

    checks = {
        "profit_factor_net": candidate_pf > thresholds.min_profit_factor_net,
        "net_sharpe": candidate_result.sharpe_ratio >= thresholds.min_sharpe_net,
        "vs_anchor_sharpe_delta": sharpe_delta >= thresholds.min_baseline_sharpe_delta,
        "recent_regime_stability": recent_ratio >= thresholds.min_recent_sharpe_ratio,
    }

    requires_chronos_backend = "chronos" in candidate_name.lower()
    if requires_chronos_backend:
        has_provenance = chronos_provenance is not None
        fallback_active = bool(
            chronos_provenance.get("fallback_active", True) if chronos_provenance else True
        )
        checks["chronos_provenance_present"] = has_provenance
        checks["chronos_backend_guardrail"] = bool(
            allow_fallback_candidate or (has_provenance and not fallback_active)
        )

    passed = all(checks.values())

    reasons: list[str] = []
    for key, ok in checks.items():
        if not ok:
            reasons.append(f"failed_{key}")

    return {
        "candidate": candidate_name,
        "anchor": anchor_name,
        "passed": bool(passed),
        "checks": checks,
        "metrics": {
            "candidate_sharpe_ratio": float(candidate_result.sharpe_ratio),
            "candidate_profit_factor_net": float(candidate_pf),
            "anchor_sharpe_ratio": float(anchor_result.sharpe_ratio),
            "anchor_profit_factor_net": float(anchor_pf),
            "sharpe_delta_vs_anchor": float(sharpe_delta),
            "recent_sharpe_ratio_vs_early": float(recent_ratio),
        },
        "recent_regime": recent_metrics,
        "chronos_provenance": chronos_provenance if chronos_provenance is not None else {},
        "allow_fallback_candidate": bool(allow_fallback_candidate),
        "reason": "all_gate_checks_passed" if passed else ",".join(reasons),
    }
