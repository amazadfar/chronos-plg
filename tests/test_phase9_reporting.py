"""Phase 9 reporting and decision framework tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.report import BacktestReport, compare_models
from src.reporting.decision import (
    DecisionOutcome,
    build_decision_report,
    compute_uncertainty_bands,
    save_decision_artifacts,
)
from src.robustness.stress_tests import StressTestResult, StressTestSuite


def _build_result(
    *,
    sharpe: float,
    pf_net: float,
    total_return: float = 0.12,
    max_dd: float = -0.12,
    win_rate: float = 0.55,
) -> BacktestResult:
    idx = pd.date_range("2024-01-01", periods=220, freq="4h", tz="UTC")
    net = pd.Series(np.random.default_rng(42).normal(0.001, 0.006, len(idx)), index=idx)
    returns = pd.DataFrame({"net_return": net, "regime": "normal"}, index=idx)
    return BacktestResult(
        sharpe_ratio=sharpe,
        profit_factor_net=pf_net,
        profit_factor=pf_net,
        total_return=total_return,
        max_drawdown=max_dd,
        win_rate=win_rate,
        returns=returns,
        fold_metrics=[
            {"sharpe_ratio": sharpe - 0.1, "profit_factor_net": max(0.1, pf_net - 0.1), "total_return": total_return - 0.03},
            {"sharpe_ratio": sharpe + 0.1, "profit_factor_net": pf_net + 0.1, "total_return": total_return + 0.03},
        ],
    )


def test_decision_report_primary_pf_gate_drives_no_go():
    result = _build_result(sharpe=0.9, pf_net=0.95)
    report = build_decision_report(
        model_name="ModelA",
        result=result,
        baseline_sharpe=0.5,
    )
    assert report.primary_gate_passed is False
    assert report.outcome == DecisionOutcome.NO_GO
    assert "Primary gate failed" in "\n".join(report.notes)


def test_decision_report_go_when_required_checks_pass():
    result = _build_result(sharpe=0.9, pf_net=1.3)
    stress = StressTestSuite(
        tests=[
            StressTestResult(
                name="stress_a",
                description="ok",
                passed=True,
                base_sharpe=0.9,
                stressed_sharpe=0.8,
                degradation=0.1,
            ),
            StressTestResult(
                name="stress_b",
                description="ok",
                passed=True,
                base_sharpe=0.9,
                stressed_sharpe=0.7,
                degradation=0.2,
            ),
        ]
    )
    report = build_decision_report(
        model_name="ModelB",
        result=result,
        baseline_sharpe=0.7,
        stress_suite=stress,
    )
    assert report.outcome == DecisionOutcome.GO
    assert report.metrics["profit_factor_net"] > 1.0


def test_uncertainty_bands_include_fold_and_block_bootstrap():
    result = _build_result(sharpe=0.7, pf_net=1.2)
    bands = compute_uncertainty_bands(result)
    assert "sharpe_ratio" in bands
    assert "profit_factor_net" in bands
    assert "fold" in bands["sharpe_ratio"]
    assert "block_bootstrap" in bands["sharpe_ratio"]
    assert bands["sharpe_ratio"]["fold"]["p05"] <= bands["sharpe_ratio"]["fold"]["p95"]


def test_save_decision_artifacts_writes_json_and_text(tmp_path):
    result = _build_result(sharpe=0.8, pf_net=1.25)
    decision = build_decision_report(model_name="ModelC", result=result, baseline_sharpe=0.6)
    paths = save_decision_artifacts(decision, output_dir=tmp_path, prefix="modelc_phase9")
    assert paths["json"].exists()
    assert paths["txt"].exists()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["outcome"] in {"GO", "ITERATE", "NO_GO"}


def test_compare_models_uses_pf_first_selection(tmp_path):
    high_sharpe_low_pf = _build_result(sharpe=1.8, pf_net=0.8)
    lower_sharpe_high_pf = _build_result(sharpe=0.7, pf_net=1.2)
    summary = compare_models(
        results={
            "ModelHighSharpeLowPF": high_sharpe_low_pf,
            "ModelPFWinner": lower_sharpe_high_pf,
        },
        output_dir=tmp_path,
    )
    assert "WINNER (PF-first): ModelPFWinner" in summary
    assert (tmp_path / "model_comparison.json").exists()


def test_backtest_report_uses_shared_kill_criteria_text():
    failing = _build_result(sharpe=0.3, pf_net=0.9, total_return=-0.05, max_dd=-0.4, win_rate=0.4)
    report = BacktestReport(result=failing, model_name="FailingModel")
    text = report.generate_summary()
    assert "profit_factor_net" in text
    assert "[FAIL]" in text


def test_decision_report_warning_is_not_treated_as_severe_fail():
    result = _build_result(sharpe=0.45, pf_net=1.2, win_rate=0.55)
    report = build_decision_report(model_name="ModelWarn", result=result)

    assert report.outcome != DecisionOutcome.NO_GO
    assert any(note.startswith("Warning criteria:") for note in report.notes)


def test_decision_report_low_win_rate_is_advisory_not_hard_fail():
    result = _build_result(sharpe=0.9, pf_net=1.3, win_rate=0.20)
    report = build_decision_report(model_name="ModelLowWinRate", result=result)

    assert report.outcome == DecisionOutcome.GO
    assert any(note.startswith("Warning criteria: win_rate") for note in report.notes)
