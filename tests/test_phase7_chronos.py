"""Phase 7 Chronos/meta validation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.evaluation.phase7_chronos import (
    build_phase7_candidate_gate,
    compute_quantile_calibration_by_regime,
    determine_recent_regime_start,
    summarize_chronos_provenance,
)
from src.models.chronos2_runner import Chronos2ForReturns
from src.models.meta_model import MetaModel


def test_chronos_runner_strict_rolling_prediction_updates_from_predicted_q50(monkeypatch):
    monkeypatch.setattr(Chronos2ForReturns, "_maybe_init_pipeline", lambda self: None)

    idx_train = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC")
    idx_test = pd.date_range("2024-01-02 08:00:00", periods=2, freq="4h", tz="UTC")

    x_train = pd.DataFrame({"feat": np.arange(8, dtype=float)}, index=idx_train)
    y_train = pd.Series(np.arange(1, 9, dtype=float), index=idx_train)
    x_test = pd.DataFrame({"feat": [10.0, 11.0]}, index=idx_test)

    model = Chronos2ForReturns(context_length=8, min_context=8, use_covariates=False)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    first_expected = float(np.quantile(y_train.to_numpy(), 0.5))
    second_context = np.append(y_train.to_numpy()[1:], first_expected)
    second_expected = float(np.quantile(second_context, 0.5))

    assert preds.index.equals(x_test.index)
    assert np.isclose(preds.iloc[0]["q50"], first_expected)
    assert np.isclose(preds.iloc[1]["q50"], second_expected)
    assert preds.iloc[0]["q50"] != preds.iloc[1]["q50"]


class _DummyChronos:
    fit_sizes: list[int] = []

    def __init__(self):
        self._mean = 0.0

    @property
    def name(self) -> str:
        return "DummyChronos"

    def clone_unfitted(self) -> _DummyChronos:
        return _DummyChronos()

    def fit(self, x: pd.DataFrame, y: pd.Series) -> _DummyChronos:
        _DummyChronos.fit_sizes.append(len(y))
        self._mean = float(y.mean())
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        q50 = self._mean + 0.01 * x["feature_1"].fillna(0.0)
        return pd.DataFrame(
            {
                "q10": q50 - 0.02,
                "q50": q50,
                "q90": q50 + 0.02,
            },
            index=x.index,
        )


def test_meta_model_uses_oof_chronos_then_final_full_fit():
    _DummyChronos.fit_sizes = []
    n = 240
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    x1 = np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.05, n)
    x2 = np.random.normal(0, 1, n)
    y = 0.03 * x1 + 0.01 * x2 + np.random.normal(0, 0.01, n)

    x = pd.DataFrame({"feature_1": x1, "feature_2": x2}, index=idx)
    y_series = pd.Series(y, index=idx)

    model = MetaModel(
        chronos_model=_DummyChronos(),
        feature_columns=["feature_1", "feature_2"],
        n_estimators=25,
        early_stopping_rounds=10,
        oof_splits=4,
        oof_min_train_samples=80,
    )
    model.fit(x, y_series)

    # OOF folds should produce several partial fits, and final fit uses full sample.
    assert len(_DummyChronos.fit_sizes) >= 2
    assert any(size < n for size in _DummyChronos.fit_sizes[:-1])
    assert _DummyChronos.fit_sizes[-1] == n
    assert 0.0 < model._chronos_oof_coverage <= 1.0

    preds = model.predict(x.iloc[-10:])
    assert set(preds.columns) == {"q10", "q50", "q90"}
    assert len(preds) == 10


def test_recent_regime_split_prefers_2024_anchor_and_candidate_gate_uses_ratio():
    idx = pd.date_range("2023-01-01", periods=3000, freq="4h", tz="UTC")
    split = determine_recent_regime_start(idx)
    assert split == pd.Timestamp("2024-01-01", tz="UTC")

    anchor = BacktestResult(sharpe_ratio=0.70, profit_factor_net=1.12, profit_factor=1.12)
    candidate = BacktestResult(sharpe_ratio=0.90, profit_factor_net=1.25, profit_factor=1.25)
    recent_metrics = {
        "recent_sharpe_ratio_vs_early": 0.85,
        "n_recent": 120,
        "n_early": 300,
    }
    gate = build_phase7_candidate_gate(
        candidate_name="Chronos2",
        candidate_result=candidate,
        anchor_name="LightGBM",
        anchor_result=anchor,
        recent_metrics=recent_metrics,
        chronos_provenance={"fallback_active": False, "backends": ["chronos_pipeline"]},
    )
    assert gate["passed"] is True
    assert gate["checks"]["recent_regime_stability"] is True

    recent_metrics_bad = {
        "recent_sharpe_ratio_vs_early": 0.4,
        "n_recent": 120,
        "n_early": 300,
    }
    gate_bad = build_phase7_candidate_gate(
        candidate_name="Chronos2",
        candidate_result=candidate,
        anchor_name="LightGBM",
        anchor_result=anchor,
        recent_metrics=recent_metrics_bad,
        chronos_provenance={"fallback_active": False, "backends": ["chronos_pipeline"]},
    )
    assert gate_bad["passed"] is False
    assert "failed_recent_regime_stability" in gate_bad["reason"]


def test_phase7_gate_blocks_chronos_candidate_when_fallback_active_by_default():
    anchor = BacktestResult(sharpe_ratio=0.70, profit_factor_net=1.12, profit_factor=1.12)
    candidate = BacktestResult(sharpe_ratio=0.90, profit_factor_net=1.25, profit_factor=1.25)
    recent_metrics = {"recent_sharpe_ratio_vs_early": 0.85}

    gate = build_phase7_candidate_gate(
        candidate_name="Chronos2",
        candidate_result=candidate,
        anchor_name="LightGBM",
        anchor_result=anchor,
        recent_metrics=recent_metrics,
        chronos_provenance={"fallback_active": True, "backends": ["empirical_fallback"]},
    )
    assert gate["passed"] is False
    assert gate["checks"]["chronos_backend_guardrail"] is False

    gate_allowed = build_phase7_candidate_gate(
        candidate_name="Chronos2",
        candidate_result=candidate,
        anchor_name="LightGBM",
        anchor_result=anchor,
        recent_metrics=recent_metrics,
        chronos_provenance={"fallback_active": True, "backends": ["empirical_fallback"]},
        allow_fallback_candidate=True,
    )
    assert gate_allowed["checks"]["chronos_backend_guardrail"] is True


def test_quantile_calibration_by_regime_and_provenance_summary_payloads():
    idx = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC")
    predictions = pd.DataFrame(
        {
            "q10": [-0.02, -0.01, -0.01, -0.005, -0.01, -0.01, -0.01, -0.01],
            "q50": [0.0, 0.005, 0.01, 0.005, 0.0, -0.005, 0.01, 0.005],
            "q90": [0.02, 0.02, 0.03, 0.02, 0.015, 0.01, 0.03, 0.02],
            "regime": ["trend", "trend", "chop", "chop", "trend", "trend", "chop", "chop"],
        },
        index=idx,
    )
    actual = pd.Series(
        [0.01, -0.005, 0.02, -0.01, 0.0, -0.01, 0.015, 0.004],
        index=idx,
    )
    calibration = compute_quantile_calibration_by_regime(
        predictions=predictions,
        actual_returns=actual,
        min_samples_per_regime=2,
    )

    assert calibration["n_rows"] == 8
    assert set(calibration["overall"]["quantiles"]) == {"q10", "q50", "q90"}
    assert "trend" in calibration["by_regime"]
    assert calibration["by_regime"]["trend"]["eligible"] is True

    summary = summarize_chronos_provenance(
        [
            {
                "event": "fit",
                "model_id": "amazon/chronos-t5-base",
                "backend": "empirical_fallback",
                "fallback_active": True,
                "fallback_reason": "import error",
                "requested_device": "auto",
                "resolved_device": "cpu",
                "chronos_version": None,
                "torch_version": "2.4.0",
            }
        ]
    )
    assert summary["event_count"] == 1
    assert summary["fit_event_count"] == 1
    assert summary["fallback_active"] is True
    assert summary["latest_backend"] == "empirical_fallback"
