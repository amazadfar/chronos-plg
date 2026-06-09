"""Exchange and market-specific fee/cost profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MarketType = Literal["spot", "margin", "futures"]
OrderType = Literal["maker", "taker"]


@dataclass(frozen=True)
class MarketFeeProfile:
    """Fee profile for a specific market type."""

    maker_fee: float
    taker_fee: float
    discount_rate: float = 0.0
    discount_token: str | None = None
    has_discount: bool = False

    def fee_rate(self, order_type: OrderType, use_discount: bool = False) -> float:
        """Get effective fee rate for maker/taker with optional discount."""
        base = self.maker_fee if order_type == "maker" else self.taker_fee
        if use_discount and self.has_discount:
            return base * (1.0 - self.discount_rate)
        return base


@dataclass(frozen=True)
class ExchangeCostProfile:
    """Cost profile for an exchange across market types."""

    exchange: str
    spot: MarketFeeProfile
    margin: MarketFeeProfile
    futures: MarketFeeProfile

    def market(self, market_type: MarketType) -> MarketFeeProfile:
        if market_type == "spot":
            return self.spot
        if market_type == "margin":
            return self.margin
        return self.futures

    def fee_rate(
        self,
        market_type: MarketType,
        order_type: OrderType = "taker",
        use_discount: bool = False,
    ) -> float:
        """Convenience accessor for effective fee rate."""
        return self.market(market_type).fee_rate(order_type, use_discount=use_discount)


BINANCE_COST_PROFILE = ExchangeCostProfile(
    exchange="binance",
    spot=MarketFeeProfile(
        maker_fee=0.0010,
        taker_fee=0.0010,
        discount_rate=0.25,  # BNB discount on spot/margin
        discount_token="BNB",
        has_discount=True,
    ),
    margin=MarketFeeProfile(
        maker_fee=0.0010,
        taker_fee=0.0010,
        discount_rate=0.25,  # BNB discount on spot/margin
        discount_token="BNB",
        has_discount=True,
    ),
    futures=MarketFeeProfile(
        maker_fee=0.0002,
        taker_fee=0.0005,
        discount_rate=0.10,  # BNB discount on futures fees
        discount_token="BNB",
        has_discount=True,
    ),
)


KUCOIN_COST_PROFILE = ExchangeCostProfile(
    exchange="kucoin",
    spot=MarketFeeProfile(
        maker_fee=0.0010,
        taker_fee=0.0010,
        discount_rate=0.20,  # KCS discount on spot/margin
        discount_token="KCS",
        has_discount=True,
    ),
    margin=MarketFeeProfile(
        maker_fee=0.0010,
        taker_fee=0.0010,
        discount_rate=0.20,  # KCS discount on spot/margin
        discount_token="KCS",
        has_discount=True,
    ),
    futures=MarketFeeProfile(
        maker_fee=0.0002,
        taker_fee=0.0006,
        discount_rate=0.0,  # No discount on futures
        discount_token=None,
        has_discount=False,
    ),
)


EXCHANGE_COST_PROFILES: dict[str, ExchangeCostProfile] = {
    "binance": BINANCE_COST_PROFILE,
    "kucoin": KUCOIN_COST_PROFILE,
}


def get_cost_profile(exchange: str) -> ExchangeCostProfile:
    """Get an exchange cost profile by name."""
    key = exchange.lower()
    if key not in EXCHANGE_COST_PROFILES:
        available = ", ".join(sorted(EXCHANGE_COST_PROFILES))
        raise ValueError(f"Unsupported exchange '{exchange}'. Available: {available}")
    return EXCHANGE_COST_PROFILES[key]
