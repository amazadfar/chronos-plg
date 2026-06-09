"""Exchange-specific symbol/contract metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractMetadata:
    """Static metadata for a tradeable symbol on an exchange/market."""

    exchange: str
    market_type: str
    symbol: str
    base_asset: str
    quote_asset: str
    tick_size: float
    lot_size: float
    min_qty: float
    min_notional: float

    def to_dict(self) -> dict:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "min_qty": self.min_qty,
            "min_notional": self.min_notional,
        }


_CONTRACT_METADATA: dict[tuple[str, str, str], ContractMetadata] = {
    ("binance", "futures", "BTCUSDT"): ContractMetadata(
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        tick_size=0.1,
        lot_size=0.001,
        min_qty=0.001,
        min_notional=100.0,
    ),
    ("kucoin", "futures", "XBTUSDTM"): ContractMetadata(
        exchange="kucoin",
        market_type="futures",
        symbol="XBTUSDTM",
        base_asset="BTC",
        quote_asset="USDT",
        tick_size=0.1,
        lot_size=1.0,
        min_qty=1.0,
        min_notional=10.0,
    ),
}


def get_contract_metadata(
    *,
    exchange: str = "binance",
    market_type: str = "futures",
    symbol: str = "BTCUSDT",
) -> ContractMetadata:
    """Get static contract metadata for exchange/market/symbol."""
    key = (exchange.lower(), market_type.lower(), symbol.upper())
    if key not in _CONTRACT_METADATA:
        available = ", ".join(f"{e}/{m}/{s}" for e, m, s in sorted(_CONTRACT_METADATA))
        raise ValueError(
            f"No contract metadata for {exchange}/{market_type}/{symbol}. "
            f"Available: {available}"
        )
    return _CONTRACT_METADATA[key]
