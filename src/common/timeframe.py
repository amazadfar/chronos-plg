"""Utilities for timeframe-aware dataset and pipeline configuration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("1h", "4h")


def normalize_timeframe(value: str | None, *, default: str = "4h") -> str:
    """Normalize and validate timeframe values used across scripts/modules."""
    candidate = (value or default).strip().lower()
    if candidate not in SUPPORTED_TIMEFRAMES:
        supported = ", ".join(SUPPORTED_TIMEFRAMES)
        raise ValueError(f"Unsupported timeframe: {value!r}. Supported values: {supported}")
    return candidate


def timeframe_to_hours(timeframe: str) -> int:
    """Convert timeframe string (e.g. '1h', '4h') to bar-length in hours."""
    tf = normalize_timeframe(timeframe)
    return int(tf[:-1])


def bars_per_day(timeframe: str) -> int:
    """Number of bars per day for a timeframe."""
    return int(24 / timeframe_to_hours(timeframe))


def periods_per_year(timeframe: str) -> int:
    """Annualization factor for returns sampled at the given timeframe."""
    return bars_per_day(timeframe) * 365


def infer_timeframe_from_index(index: pd.DatetimeIndex, *, fallback: str = "4h") -> str:
    """Infer timeframe from a DatetimeIndex; fallback when not inferable."""
    if len(index) < 2:
        return normalize_timeframe(fallback)

    delta = index.to_series().diff().dropna().median()
    if pd.isna(delta):
        return normalize_timeframe(fallback)

    hours = int(round(delta.total_seconds() / 3600))
    if hours <= 0:
        return normalize_timeframe(fallback)

    candidate = f"{hours}h"
    try:
        return normalize_timeframe(candidate)
    except ValueError:
        return normalize_timeframe(fallback)


def default_dataset_stem(*, timeframe: str, asset: str = "btc") -> str:
    """Return canonical processed dataset stem, e.g. 'btc_4h'."""
    tf = normalize_timeframe(timeframe)
    return f"{asset.lower()}_{tf}"


def default_processed_dataset_path(
    *,
    timeframe: str,
    processed_dir: str | Path = "data/processed",
    asset: str = "btc",
) -> Path:
    """Return canonical processed dataset path for a timeframe."""
    stem = default_dataset_stem(timeframe=timeframe, asset=asset)
    return Path(processed_dir) / f"{stem}.parquet"
