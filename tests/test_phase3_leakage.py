"""Phase 3 leakage and walk-forward boundary guardrail tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import WalkForwardConfig
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.models.baselines.random_walk import RandomWalkBaseline


@pytest.fixture
def sample_walkforward_data() -> pd.DataFrame:
    """Create realistic 4h sample data for walk-forward leakage checks."""
    np.random.seed(42)
    n = 600
    index = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")

    returns = np.random.normal(0, 0.02, n)
    frame = pd.DataFrame(index=index)
    frame["close"] = 40000 * np.exp(np.cumsum(returns))
    frame["return_1"] = returns
    frame["return_6"] = pd.Series(returns, index=index).rolling(6).sum()
    frame["realized_vol_6"] = pd.Series(returns, index=index).rolling(6).std()
    frame["forward_return"] = np.roll(returns, -1)
    frame["regime"] = np.where(np.abs(returns) > 0.03, "trend", "normal")
    frame.loc[frame.index[-1], "forward_return"] = np.nan
    return frame


def _wf_config() -> WalkForwardConfig:
    return WalkForwardConfig(
        train_window_days=30,
        test_window_days=7,
        step_size_days=7,
        min_train_samples=40,
    )


def test_walk_forward_enforces_strict_feature_lag(sample_walkforward_data: pd.DataFrame):
    """Fold test targets should start after fold test_start due lagged feature construction."""
    evaluator = WalkForwardEvaluator(config=_wf_config(), feature_lag_candles=1)
    results = evaluator.evaluate_model(
        RandomWalkBaseline,
        sample_walkforward_data,
        feature_columns=["return_1", "return_6", "realized_vol_6"],
        model_kwargs={"lookback_window": 50},
        show_progress=False,
    )

    assert len(results.folds) > 0
    first_fold = results.folds[0]
    assert first_fold.actuals is not None
    assert first_fold.actuals.index.min() > first_fold.test_start


def test_fold_boundaries_artifact_snapshot_written(
    sample_walkforward_data: pd.DataFrame,
    tmp_path: Path,
):
    """Walk-forward evaluator should write fold boundary artifact when enabled."""
    evaluator = WalkForwardEvaluator(config=_wf_config(), feature_lag_candles=1)
    results = evaluator.evaluate_model(
        RandomWalkBaseline,
        sample_walkforward_data,
        feature_columns=["return_1", "return_6", "realized_vol_6"],
        model_kwargs={"lookback_window": 50},
        show_progress=False,
        save_fold_boundaries=True,
        artifact_dir=tmp_path,
    )

    assert results.fold_boundaries_artifact is not None
    artifact_path = Path(results.fold_boundaries_artifact)
    assert artifact_path.exists()

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["feature_lag_candles"] == 1
    assert payload["n_folds"] == len(payload["folds"])
    assert payload["n_folds"] > 0


def test_generate_folds_rejects_duplicate_index(sample_walkforward_data: pd.DataFrame):
    """Datetime index duplicates should fail fold generation early."""
    evaluator = WalkForwardEvaluator(config=_wf_config(), feature_lag_candles=1)
    bad = sample_walkforward_data.copy()
    idx = list(bad.index)
    idx[10] = idx[9]
    bad.index = pd.DatetimeIndex(idx)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        evaluator.generate_folds(bad)
