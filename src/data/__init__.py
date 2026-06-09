"""Data fetching and processing modules."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BinanceFetcher",
    "MacroFetcher",
    "LiquidationCollector",
    "LabelGenerator",
    "DatasetBuilder",
    "ContractMetadata",
    "get_contract_metadata",
]


def __getattr__(name: str) -> Any:
    """Lazily import data modules to avoid forcing optional deps at package import time."""
    if name == "BinanceFetcher":
        from src.data.binance_fetcher import BinanceFetcher
        return BinanceFetcher
    if name == "MacroFetcher":
        from src.data.macro_fetcher import MacroFetcher
        return MacroFetcher
    if name == "LiquidationCollector":
        from src.data.liquidation_collector import LiquidationCollector
        return LiquidationCollector
    if name == "LabelGenerator":
        from src.data.labels import LabelGenerator
        return LabelGenerator
    if name == "DatasetBuilder":
        from src.data.build_dataset import DatasetBuilder
        return DatasetBuilder
    if name in {"ContractMetadata", "get_contract_metadata"}:
        from src.data.market_metadata import ContractMetadata, get_contract_metadata
        return {
            "ContractMetadata": ContractMetadata,
            "get_contract_metadata": get_contract_metadata,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
