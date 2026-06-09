"""Data contracts and validation helpers for raw and processed datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class DataContractError(ValueError):
    """Raised when an input dataset violates required contract constraints."""


RAW_DATA_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "ohlcv": ("open", "high", "low", "close", "volume", "quote_volume", "trades"),
    "funding_rate": ("funding_rate",),
    "open_interest": ("open_interest", "open_interest_value"),
    "macro": (),
    "event_flags": ("is_fomc_day", "is_cpi_day", "post_fomc_window", "post_cpi_window", "is_event_window"),
    "liquidations": (),
    "contract_metadata": ("exchange", "market_type", "symbol", "base_asset", "quote_asset"),
}


@dataclass(frozen=True)
class IndexGapStats:
    """Gap statistics for a timestamp index."""

    actual_points: int
    expected_points: int
    missing_points: int
    gap_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual_points": self.actual_points,
            "expected_points": self.expected_points,
            "missing_points": self.missing_points,
            "gap_ratio": self.gap_ratio,
        }


def validate_datetime_index(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    allow_empty: bool = True,
) -> None:
    """Validate DatetimeIndex contract: tz-aware, unique, monotonic."""
    if df.empty and allow_empty:
        return

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataContractError(f"{dataset_name}: index must be pandas.DatetimeIndex")
    if df.index.tz is None:
        raise DataContractError(f"{dataset_name}: index must be timezone-aware (UTC)")
    if not df.index.is_monotonic_increasing:
        raise DataContractError(f"{dataset_name}: index must be monotonic increasing")
    if df.index.has_duplicates:
        raise DataContractError(f"{dataset_name}: index must not contain duplicates")


def validate_required_columns(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    required_columns: tuple[str, ...],
    allow_empty: bool = True,
) -> None:
    """Validate required columns for a dataset contract."""
    if df.empty and allow_empty:
        return

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise DataContractError(f"{dataset_name}: missing required columns: {missing_str}")


def validate_raw_data_contracts(raw_data: dict[str, pd.DataFrame]) -> None:
    """Validate all known raw data frames against contract rules."""
    time_series_datasets = {
        "ohlcv",
        "funding_rate",
        "open_interest",
        "macro",
        "event_flags",
        "liquidations",
    }

    for name, required in RAW_DATA_REQUIRED_COLUMNS.items():
        if name not in raw_data:
            if name == "ohlcv":
                raise DataContractError("raw data map must include 'ohlcv'")
            continue

        df = raw_data[name]
        if name in time_series_datasets:
            validate_datetime_index(df, dataset_name=name, allow_empty=True)
        validate_required_columns(
            df,
            dataset_name=name,
            required_columns=required,
            allow_empty=True,
        )

    if raw_data["ohlcv"].empty:
        raise DataContractError("ohlcv must not be empty")


def compute_index_gap_stats(index: pd.DatetimeIndex, freq: str = "4h") -> IndexGapStats:
    """Compute missing timestamp stats for a DatetimeIndex."""
    if len(index) == 0:
        return IndexGapStats(actual_points=0, expected_points=0, missing_points=0, gap_ratio=0.0)

    expected = pd.date_range(start=index.min(), end=index.max(), freq=freq, tz=index.tz)
    missing = len(expected.difference(index))
    expected_points = len(expected)
    gap_ratio = missing / expected_points if expected_points else 0.0

    return IndexGapStats(
        actual_points=len(index),
        expected_points=expected_points,
        missing_points=missing,
        gap_ratio=gap_ratio,
    )
