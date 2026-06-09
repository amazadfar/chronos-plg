"""Baseline models package."""

from __future__ import annotations

from typing import Any

__all__ = ["RandomWalkBaseline", "EWMABaseline", "LightGBMQuantileBaseline"]


def __getattr__(name: str) -> Any:
    """Lazily import optional baselines to avoid hard dependency at package import time."""
    if name == "RandomWalkBaseline":
        from src.models.baselines.random_walk import RandomWalkBaseline
        return RandomWalkBaseline
    if name == "EWMABaseline":
        from src.models.baselines.ewma import EWMABaseline
        return EWMABaseline
    if name == "LightGBMQuantileBaseline":
        from src.models.baselines.lightgbm_quantile import LightGBMQuantileBaseline
        return LightGBMQuantileBaseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
