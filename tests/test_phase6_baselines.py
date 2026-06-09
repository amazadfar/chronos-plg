"""Phase 6 baseline protocol and leaderboard utility tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config.baseline_protocols import (
    BaselineModelSpec,
    BaselineProtocol,
    get_baseline_protocol,
)
from config.settings import WalkForwardConfig
from src.backtest.engine import BacktestResult
from src.evaluation.phase6_baselines import (
    build_chronos_advancement_gate,
    build_leaderboard,
    freeze_fold_schedule,
    infer_feature_columns,
    resolve_model_configs,
    write_gate_artifact,
    write_leaderboard_artifacts,
    write_protocol_freeze,
)
from src.evaluation.walk_forward import WalkForwardEvaluator


def _sample_result(sharpe: float, pf_net: float, total_return: float = 0.1) -> BacktestResult:
    return BacktestResult(
        sharpe_ratio=sharpe,
        profit_factor_net=pf_net,
        profit_factor=pf_net,
        total_return=total_return,
        max_drawdown=-0.1,
        win_rate=0.52,
        total_costs=0.02,
        num_trades=25,
        fold_metrics=[{"fold_id": 0}],
    )


def test_baseline_protocol_fingerprint_stable():
    protocol = get_baseline_protocol()
    assert protocol.fingerprint() == protocol.fingerprint()
    assert len(protocol.models) >= 3


def test_infer_feature_columns_filters_leakage_columns():
    idx = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
    data = pd.DataFrame(
        {
            "close": np.arange(10),
            "forward_return": np.random.randn(10),
            "regime_code": np.random.randint(0, 3, size=10),
            "feature_a": np.random.randn(10),
            "feature_b": np.random.randn(10),
        },
        index=idx,
    )
    features = infer_feature_columns(data)
    assert "feature_a" in features and "feature_b" in features
    assert "close" not in features
    assert "forward_return" not in features
    assert "regime_code" not in features


def test_freeze_protocol_and_fold_schedule(tmp_path):
    protocol = BaselineProtocol(
        name="test_protocol",
        description="test",
        mode="weekly",
        train_window_days=3,
        test_window_days=1,
        step_size_days=1,
        min_train_samples=2,
        models=(BaselineModelSpec(key="random_walk", name="RandomWalk"),),
    )
    protocol_path = write_protocol_freeze(protocol, tmp_path)
    assert protocol_path.exists()
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert payload["name"] == "test_protocol"
    assert "fingerprint" in payload

    idx = pd.date_range("2024-01-01", periods=120, freq="4h", tz="UTC")
    data = pd.DataFrame(
        {
            "feature_1": np.random.randn(len(idx)),
            "forward_return": np.random.randn(len(idx)),
        },
        index=idx,
    )
    evaluator = WalkForwardEvaluator(config=WalkForwardConfig(
        train_window_days=3,
        test_window_days=1,
        step_size_days=1,
        min_train_samples=2,
        mode="weekly",
    ))
    folds, folds_path = freeze_fold_schedule(
        evaluator=evaluator,
        data=data,
        protocol=protocol,
        start_date=None,
        output_dir=tmp_path,
    )
    assert len(folds) > 0
    assert folds_path.exists()
    fold_payload = json.loads(folds_path.read_text(encoding="utf-8"))
    assert fold_payload["n_folds"] == len(folds)
    assert "fingerprint" in fold_payload


def test_resolve_model_configs_injects_lgb_features():
    protocol = BaselineProtocol(
        name="resolve_test",
        description="test",
        models=(
            BaselineModelSpec(key="random_walk", name="RandomWalk"),
            BaselineModelSpec(key="lightgbm", name="LightGBM"),
        ),
    )
    resolved = resolve_model_configs(protocol, ["feature_1", "feature_2"])
    assert "RandomWalk" in resolved and "LightGBM" in resolved
    assert resolved["LightGBM"][1]["feature_columns"] == ["feature_1", "feature_2"]


def test_leaderboard_and_gate_artifacts(tmp_path):
    protocol = BaselineProtocol(
        name="gate_test",
        description="test",
        baseline_anchor_model="LightGBM",
    )
    results = {
        "RandomWalk": _sample_result(sharpe=0.25, pf_net=0.95, total_return=0.02),
        "LightGBM": _sample_result(sharpe=0.65, pf_net=1.15, total_return=0.14),
        "EWMA": _sample_result(sharpe=0.4, pf_net=1.01, total_return=0.08),
    }
    leaderboard = build_leaderboard(results)
    assert leaderboard.iloc[0]["model"] == "LightGBM"
    assert leaderboard.iloc[0]["profit_factor_net"] > 1.0

    paths = write_leaderboard_artifacts(leaderboard, tmp_path, protocol.name)
    assert paths["csv"].exists()
    assert paths["json"].exists()
    assert paths["md"].exists()

    gate = build_chronos_advancement_gate(results=results, protocol=protocol)
    assert gate["passed"] is True
    assert gate["anchor_model"] == "LightGBM"
    gate_path = write_gate_artifact(gate, tmp_path, protocol.name)
    assert gate_path.exists()
