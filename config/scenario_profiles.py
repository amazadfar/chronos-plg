"""Named scenario profiles used for benchmark and backtest runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config.cost_profiles import MarketType


OrderType = Literal["maker", "taker"]


@dataclass(frozen=True)
class TradingScenarioProfile:
    """Scenario settings for execution/cost assumptions."""

    name: str
    exchange: str
    market_type: MarketType
    order_type: OrderType
    use_fee_discount: bool
    apply_funding: bool
    apply_margin_interest: bool
    default_margin_interest_rate_per_day: float
    other_cost_bps: float
    description: str


SCENARIO_PROFILES: dict[str, TradingScenarioProfile] = {
    "binance_futures_taker_discounted": TradingScenarioProfile(
        name="binance_futures_taker_discounted",
        exchange="binance",
        market_type="futures",
        order_type="taker",
        use_fee_discount=True,
        apply_funding=True,
        apply_margin_interest=False,
        default_margin_interest_rate_per_day=0.0,
        other_cost_bps=0.0,
        description="Binance futures taker with BNB fee discount enabled.",
    ),
    "kucoin_futures_taker": TradingScenarioProfile(
        name="kucoin_futures_taker",
        exchange="kucoin",
        market_type="futures",
        order_type="taker",
        use_fee_discount=False,
        apply_funding=True,
        apply_margin_interest=False,
        default_margin_interest_rate_per_day=0.0,
        other_cost_bps=0.0,
        description="KuCoin futures taker, no futures fee discount.",
    ),
    "binance_spot_taker_discounted": TradingScenarioProfile(
        name="binance_spot_taker_discounted",
        exchange="binance",
        market_type="spot",
        order_type="taker",
        use_fee_discount=True,
        apply_funding=False,
        apply_margin_interest=False,
        default_margin_interest_rate_per_day=0.0,
        other_cost_bps=0.0,
        description="Binance spot taker with BNB fee discount enabled.",
    ),
    "kucoin_spot_taker_discounted": TradingScenarioProfile(
        name="kucoin_spot_taker_discounted",
        exchange="kucoin",
        market_type="spot",
        order_type="taker",
        use_fee_discount=True,
        apply_funding=False,
        apply_margin_interest=False,
        default_margin_interest_rate_per_day=0.0,
        other_cost_bps=0.0,
        description="KuCoin spot taker with KCS fee discount enabled.",
    ),
    "binance_margin_taker_discounted": TradingScenarioProfile(
        name="binance_margin_taker_discounted",
        exchange="binance",
        market_type="margin",
        order_type="taker",
        use_fee_discount=True,
        apply_funding=False,
        apply_margin_interest=True,
        default_margin_interest_rate_per_day=0.0003,
        other_cost_bps=0.0,
        description="Binance margin taker with BNB fee discount and margin interest.",
    ),
    "kucoin_margin_taker_discounted": TradingScenarioProfile(
        name="kucoin_margin_taker_discounted",
        exchange="kucoin",
        market_type="margin",
        order_type="taker",
        use_fee_discount=True,
        apply_funding=False,
        apply_margin_interest=True,
        default_margin_interest_rate_per_day=0.0003,
        other_cost_bps=0.0,
        description="KuCoin margin taker with KCS fee discount and margin interest.",
    ),
}


DEFAULT_SCENARIO = "binance_futures_taker_discounted"


def get_scenario_profile(name: str = DEFAULT_SCENARIO) -> TradingScenarioProfile:
    """Get a named scenario profile."""
    key = name.lower()
    if key not in SCENARIO_PROFILES:
        available = ", ".join(sorted(SCENARIO_PROFILES))
        raise ValueError(f"Unknown scenario '{name}'. Available: {available}")
    return SCENARIO_PROFILES[key]
