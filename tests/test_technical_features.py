"""Tests for causal technical features mined from the legacy trading archive."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.technical_features import compute_technical_features


@pytest.fixture
def ohlcv_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 160
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    returns = rng.normal(0.001, 0.015, size=n)
    close = 40_000 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.002, size=n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, size=n))
    volume = rng.uniform(1_000, 4_000, size=n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_compute_technical_features_adds_expected_columns(ohlcv_frame: pd.DataFrame) -> None:
    features = compute_technical_features(ohlcv_frame)

    expected = {
        "tech_hlc3",
        "tech_ohlc4",
        "tech_range_pct",
        "tech_body_pct",
        "tech_rsi_14",
        "tech_atr_pct_14",
        "tech_bbands_width_20",
        "tech_macd_hist",
        "tech_obv",
        "tech_vroc_14",
        "tech_stoch_k_14",
        "tech_cci_20",
        "tech_plus_di_14",
        "tech_minus_di_14",
        "tech_adx_14",
        "tech_ha_body_pct",
        "tech_fib_position_100",
        "tech_pma_5_10_20",
    }

    assert expected.issubset(features.columns)
    assert features.index.equals(ohlcv_frame.index)
    assert len(features.columns) >= 40
    assert np.isfinite(features.dropna(how="all").to_numpy()).any()


def test_compute_technical_features_is_causal_with_future_row_changes(
    ohlcv_frame: pd.DataFrame,
) -> None:
    baseline = compute_technical_features(ohlcv_frame)

    mutated = ohlcv_frame.copy()
    future_start = mutated.index[110]
    mutated.loc[future_start:, "close"] *= 1.40
    mutated.loc[future_start:, "high"] *= 1.45
    mutated.loc[future_start:, "low"] *= 0.75
    mutated.loc[future_start:, "volume"] *= 8.0
    changed = compute_technical_features(mutated)

    pd.testing.assert_frame_equal(
        baseline.iloc[:109],
        changed.iloc[:109],
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_compute_technical_features_rejects_missing_ohlcv_columns(
    ohlcv_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        compute_technical_features(ohlcv_frame.drop(columns=["volume"]))
