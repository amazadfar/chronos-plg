"""Models package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "RandomWalkBaseline",
    "EWMABaseline",
    "LightGBMQuantileBaseline",
    "Chronos2Runner",
    "Chronos2ForReturns",
    "MetaModel",
]


def __getattr__(name: str) -> Any:
    """Lazily import model modules so optional dependencies remain optional."""
    if name in {"RandomWalkBaseline", "EWMABaseline", "LightGBMQuantileBaseline"}:
        from src.models.baselines import (
            EWMABaseline,
            LightGBMQuantileBaseline,
            RandomWalkBaseline,
        )
        return {
            "RandomWalkBaseline": RandomWalkBaseline,
            "EWMABaseline": EWMABaseline,
            "LightGBMQuantileBaseline": LightGBMQuantileBaseline,
        }[name]
    if name in {"Chronos2Runner", "Chronos2ForReturns"}:
        from src.models.chronos2_runner import Chronos2ForReturns, Chronos2Runner
        return {
            "Chronos2Runner": Chronos2Runner,
            "Chronos2ForReturns": Chronos2ForReturns,
        }[name]
    if name == "MetaModel":
        from src.models.meta_model import MetaModel
        return MetaModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
