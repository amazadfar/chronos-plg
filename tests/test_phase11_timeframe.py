"""Phase 11 timeframe-aware dataset/pipeline helper tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.timeframe import (
    default_processed_dataset_path,
    infer_timeframe_from_index,
    normalize_timeframe,
    periods_per_year,
)
from src.data.build_dataset import DatasetBuilder


def test_normalize_timeframe_accepts_supported_values() -> None:
    assert normalize_timeframe("4h") == "4h"
    assert normalize_timeframe("1H") == "1h"


def test_normalize_timeframe_rejects_unsupported_values() -> None:
    with pytest.raises(ValueError):
        normalize_timeframe("2h")


def test_periods_per_year_by_timeframe() -> None:
    assert periods_per_year("4h") == 2190
    assert periods_per_year("1h") == 8760


def test_infer_timeframe_from_index() -> None:
    idx_4h = pd.date_range("2024-01-01", periods=20, freq="4h", tz="UTC")
    idx_1h = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")

    assert infer_timeframe_from_index(idx_4h) == "4h"
    assert infer_timeframe_from_index(idx_1h) == "1h"


def test_default_processed_dataset_path_uses_timeframe_suffix() -> None:
    assert str(default_processed_dataset_path(timeframe="4h")) == "data/processed/btc_4h.parquet"
    assert str(default_processed_dataset_path(timeframe="1h")) == "data/processed/btc_1h.parquet"


def test_dataset_builder_processed_artifact_paths_are_interval_aware() -> None:
    builder_4h = DatasetBuilder(interval="4h")
    builder_1h = DatasetBuilder(interval="1h")

    paths_4h = builder_4h.processed_artifact_paths()
    paths_1h = builder_1h.processed_artifact_paths()

    assert paths_4h["dataset"].name == "btc_4h.parquet"
    assert paths_4h["metadata"].name == "btc_4h_metadata.json"
    assert paths_4h["quality"].name == "btc_4h_quality.json"

    assert paths_1h["dataset"].name == "btc_1h.parquet"
    assert paths_1h["metadata"].name == "btc_1h_metadata.json"
    assert paths_1h["quality"].name == "btc_1h_quality.json"


def test_dataset_builder_supports_1h_build_mode_without_leakage_breaks() -> None:
    idx = pd.date_range("2024-01-01", periods=240, freq="1h", tz="UTC")
    close = pd.Series(50000 + (pd.RangeIndex(len(idx)) * 2.0), index=idx)
    ohlcv = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
            "volume": 1000.0,
            "quote_volume": 5e7,
            "trades": 1000,
        },
        index=idx,
    )
    raw_data = {
        "ohlcv": ohlcv,
        "funding_rate": pd.DataFrame(index=idx, data={"funding_rate": 0.0}),
        "open_interest": pd.DataFrame(index=idx, data={"open_interest": 1.0, "open_interest_value": 1.0}),
        "macro": pd.DataFrame(index=idx, data={"dxy_return_1d": 0.0, "spx_return_1d": 0.0, "vix": 20.0}),
        "event_flags": pd.DataFrame(
            index=idx,
            data={
                "is_fomc_day": 0,
                "is_cpi_day": 0,
                "post_fomc_window": 0,
                "post_cpi_window": 0,
                "is_event_window": 0,
            },
        ),
        "liquidations": pd.DataFrame(
            index=idx,
            data={
                "long_liq_usd_est": 0.0,
                "short_liq_usd_est": 0.0,
                "liq_imbalance_est": 0.0,
                "has_real_liq_data": 1,
                "liq_data_source_code": 1,
            },
        ),
        "contract_metadata": pd.DataFrame(
            [
                {
                    "exchange": "binance",
                    "market_type": "futures",
                    "symbol": "BTCUSDT",
                    "base_asset": "BTC",
                    "quote_asset": "USDT",
                    "tick_size": 0.1,
                    "lot_size": 0.001,
                    "min_qty": 0.001,
                    "min_notional": 100.0,
                }
            ],
            index=["BTCUSDT"],
        ),
    }

    builder = DatasetBuilder(interval="1h")
    dataset = builder.build_dataset(raw_data, save=False)
    report = builder.generate_quality_report(raw_data, dataset)

    assert len(dataset) == len(ohlcv)
    assert report["timeframe"] == "1h"
    assert report["index_gap_stats"]["gap_ratio"] == 0.0
