"""
Execution-event trading cost model.

Models realistic trading costs at each execution event:
- Exchange fees by exchange/market/order type
- Position transition fee legs (open/increase/reduce/close/reverse)
- Slippage in return units (base + volatility + size terms)
- Perpetual funding cashflows (signed by long/short position)
- Margin interest accrual (borrow rate and holding time)
- Pluggable additional charges
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from config.cost_profiles import MarketType, OrderType, get_cost_profile
from config.settings import CostConfig, get_settings

logger = logging.getLogger(__name__)

TransitionType = Literal["hold", "open", "increase", "reduce", "close", "reverse"]


@dataclass(frozen=True)
class TransitionLegs:
    """Decompose a position transition into open and close fee legs."""

    event_type: TransitionType
    open_notional: float
    close_notional: float

    @property
    def traded_notional(self) -> float:
        return self.open_notional + self.close_notional


@dataclass
class TradeCosts:
    """Cost breakdown for one execution event."""

    fees: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    interest: float = 0.0
    other_costs: float = 0.0
    total: float = 0.0

    def __post_init__(self) -> None:
        if self.total == 0.0:
            self.total = self.fees + self.slippage + self.funding + self.interest + self.other_costs

    @classmethod
    def zero(cls) -> "TradeCosts":
        return cls()


@dataclass(frozen=True)
class ExecutionCostEvent:
    """Execution event with transition metadata and full cost breakdown."""

    timestamp: Optional[pd.Timestamp]
    prev_position: float
    new_position: float
    event_type: TransitionType
    open_notional: float
    close_notional: float
    traded_notional: float
    fees: float
    slippage: float
    funding: float
    interest: float
    other_costs: float
    total_costs: float


class CostModel:
    """
    Execution-level trading cost model with exchange-specific fee schedules.

    Position units are treated as notional exposure units. Example: position=1.0 means
    1x unit notional long exposure, position=-0.5 means 0.5x short exposure.
    """

    DEFAULT_BAR_SECONDS = 4 * 3600

    def __init__(
        self,
        config: Optional[CostConfig] = None,
        fee_rate: Optional[float] = None,
        slippage_base_bps: Optional[float] = None,
        slippage_vol_multiplier: Optional[float] = None,
        slippage_size_coefficient: float = 0.0,
        exchange: str = "binance",
        market_type: MarketType = "futures",
        order_type: OrderType = "taker",
        use_fee_discount: bool = True,
        apply_funding: Optional[bool] = None,
        apply_margin_interest: Optional[bool] = None,
        margin_interest_rate_per_day: float = 0.0,
        other_cost_bps: float = 0.0,
        fixed_other_cost: float = 0.0,
    ):
        """
        Args:
            config: Cost configuration from settings.
            fee_rate: Optional direct fee override.
            slippage_base_bps: Base slippage in bps.
            slippage_vol_multiplier: Volatility multiplier for slippage rate.
            slippage_size_coefficient: Linear size-impact term in return units per notional.
            exchange: Exchange name for fee schedule lookup.
            market_type: spot/margin/futures.
            order_type: maker/taker.
            use_fee_discount: Whether exchange fee token discount is enabled.
            apply_funding: Apply perp funding cashflows (default: futures only).
            apply_margin_interest: Apply margin borrow interest (default: margin only).
            margin_interest_rate_per_day: Default daily margin borrow rate.
            other_cost_bps: Additional proportional charge in bps on traded notional.
            fixed_other_cost: Flat charge applied when traded_notional > 0.
        """
        self.config = config or get_settings().costs

        profile_fee_rate = get_cost_profile(exchange).fee_rate(
            market_type=market_type,
            order_type=order_type,
            use_discount=use_fee_discount,
        )

        self.fee_rate = fee_rate if fee_rate is not None else profile_fee_rate
        self.slippage_base_bps = (
            slippage_base_bps if slippage_base_bps is not None else self.config.slippage_base_bps
        )
        self.slippage_vol_multiplier = (
            slippage_vol_multiplier
            if slippage_vol_multiplier is not None
            else self.config.slippage_vol_multiplier
        )
        self.slippage_size_coefficient = slippage_size_coefficient

        self.exchange = exchange
        self.market_type = market_type
        self.order_type = order_type
        self.use_fee_discount = use_fee_discount

        self.apply_funding = apply_funding if apply_funding is not None else market_type == "futures"
        self.apply_margin_interest = (
            apply_margin_interest if apply_margin_interest is not None else market_type == "margin"
        )
        self.margin_interest_rate_per_day = margin_interest_rate_per_day
        self.other_cost_bps = other_cost_bps
        self.fixed_other_cost = fixed_other_cost

    @staticmethod
    def _classify_transition(prev_position: float, new_position: float) -> TransitionLegs:
        """Classify transition and decompose into open/close notional legs."""
        eps = 1e-12
        prev_abs = abs(prev_position)
        new_abs = abs(new_position)
        delta = new_position - prev_position

        if abs(delta) <= eps:
            return TransitionLegs("hold", 0.0, 0.0)
        if prev_abs <= eps and new_abs > eps:
            return TransitionLegs("open", new_abs, 0.0)
        if new_abs <= eps and prev_abs > eps:
            return TransitionLegs("close", 0.0, prev_abs)

        same_direction = prev_position * new_position > 0
        if same_direction:
            if new_abs > prev_abs:
                return TransitionLegs("increase", new_abs - prev_abs, 0.0)
            return TransitionLegs("reduce", 0.0, prev_abs - new_abs)

        # Sign flip with non-zero both sides: explicit reverse (close then open).
        return TransitionLegs("reverse", new_abs, prev_abs)

    def _slippage_rate(self, volatility: float, traded_notional: float) -> float:
        """Slippage rate in return units."""
        rate = self.slippage_base_bps / 10000.0
        rate += self.slippage_vol_multiplier * max(float(volatility), 0.0)
        rate += self.slippage_size_coefficient * max(float(traded_notional), 0.0)
        return max(rate, 0.0)

    @staticmethod
    def _infer_bar_seconds(index: pd.DatetimeIndex) -> int:
        """Infer dominant bar spacing in seconds from a DatetimeIndex."""
        if len(index) < 2:
            return CostModel.DEFAULT_BAR_SECONDS
        deltas = index.to_series().diff().dropna()
        if deltas.empty:
            return CostModel.DEFAULT_BAR_SECONDS
        step = deltas.mode().iloc[0]
        seconds = int(step.total_seconds())
        return seconds if seconds > 0 else CostModel.DEFAULT_BAR_SECONDS

    def estimate_entry_cost_rate(
        self,
        *,
        volatility: float = 0.02,
        expected_funding_rate: float = 0.0,
        expected_borrow_rate_per_day: float | None = None,
        bar_seconds: int = DEFAULT_BAR_SECONDS,
        expected_holding_bars: int = 1,
        include_exit: bool = True,
        target_notional: float = 1.0,
        other_cost: float = 0.0,
    ) -> float:
        """
        Estimate expected per-trade cost rate for entry filtering logic.

        The estimate is a conservative approximation intended for signal gating:
        - entry execution costs (fees/slippage/other)
        - optional exit execution costs
        - expected carry costs (funding/interest) over holding horizon

        Returns:
            Estimated cost expressed in return units per unit notional.
        """
        notional = abs(float(target_notional))
        if notional <= 0:
            return 0.0

        horizon_bars = max(int(expected_holding_bars), 1)
        vol = max(float(volatility), 0.0)

        entry_event = self.calculate_event_costs(
            prev_position=0.0,
            new_position=notional,
            volatility=vol,
            funding_rate=0.0,
            borrow_rate_per_day=expected_borrow_rate_per_day,
            bar_seconds=bar_seconds,
            other_cost=other_cost,
        )
        total = float(entry_event.total_costs)

        if include_exit:
            exit_event = self.calculate_event_costs(
                prev_position=notional,
                new_position=0.0,
                volatility=vol,
                funding_rate=0.0,
                borrow_rate_per_day=expected_borrow_rate_per_day,
                bar_seconds=bar_seconds,
                other_cost=other_cost,
            )
            total += float(exit_event.total_costs)

        if self.apply_funding:
            total += notional * abs(float(expected_funding_rate)) * horizon_bars

        if self.apply_margin_interest:
            borrow_rate = (
                self.margin_interest_rate_per_day
                if expected_borrow_rate_per_day is None or pd.isna(expected_borrow_rate_per_day)
                else float(expected_borrow_rate_per_day)
            )
            total += notional * max(borrow_rate, 0.0) * ((bar_seconds * horizon_bars) / 86400.0)

        return float(total / notional)

    def estimate_entry_cost_series(
        self,
        *,
        index: pd.DatetimeIndex,
        volatilities: pd.Series | None = None,
        funding_rates: pd.Series | None = None,
        borrow_rates_per_day: pd.Series | None = None,
        expected_holding_bars: int = 1,
        include_exit: bool = True,
        target_notional: float = 1.0,
        other_costs: pd.Series | None = None,
    ) -> pd.Series:
        """
        Estimate expected entry cost rate per timestamp for signal filtering.
        """
        if len(index) == 0:
            return pd.Series(dtype=float)

        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("index must be a DatetimeIndex")

        bar_seconds = self._infer_bar_seconds(index)
        vols = (
            volatilities.reindex(index).fillna(0.02).astype(float)
            if volatilities is not None
            else pd.Series(0.02, index=index, dtype=float)
        )
        funding = (
            funding_rates.reindex(index).fillna(0.0).astype(float)
            if funding_rates is not None
            else pd.Series(0.0, index=index, dtype=float)
        )
        borrow = (
            borrow_rates_per_day.reindex(index)
            if borrow_rates_per_day is not None
            else pd.Series(np.nan, index=index, dtype=float)
        )
        other = (
            other_costs.reindex(index).fillna(0.0).astype(float)
            if other_costs is not None
            else pd.Series(0.0, index=index, dtype=float)
        )

        estimates: list[float] = []
        for ts in index:
            estimates.append(
                self.estimate_entry_cost_rate(
                    volatility=float(vols.loc[ts]),
                    expected_funding_rate=float(funding.loc[ts]),
                    expected_borrow_rate_per_day=borrow.loc[ts],
                    bar_seconds=bar_seconds,
                    expected_holding_bars=expected_holding_bars,
                    include_exit=include_exit,
                    target_notional=target_notional,
                    other_cost=float(other.loc[ts]),
                )
            )

        return pd.Series(estimates, index=index, dtype=float)

    def calculate_event_costs(
        self,
        *,
        prev_position: float,
        new_position: float,
        volatility: float = 0.02,
        funding_rate: float = 0.0,
        borrow_rate_per_day: Optional[float] = None,
        bar_seconds: int = DEFAULT_BAR_SECONDS,
        other_cost: float = 0.0,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> ExecutionCostEvent:
        """
        Calculate full execution event costs for a transition.

        Funding and interest use `prev_position` because those cashflows accrue over the
        holding interval immediately preceding the current execution timestamp.
        """
        transition = self._classify_transition(prev_position, new_position)
        traded_notional = transition.traded_notional

        fees = traded_notional * self.fee_rate
        slippage = traded_notional * self._slippage_rate(volatility, traded_notional)

        funding = prev_position * float(funding_rate) if self.apply_funding else 0.0

        borrow_rate = (
            self.margin_interest_rate_per_day
            if borrow_rate_per_day is None or pd.isna(borrow_rate_per_day)
            else float(borrow_rate_per_day)
        )
        if self.apply_margin_interest:
            interest = abs(prev_position) * max(borrow_rate, 0.0) * (bar_seconds / 86400.0)
        else:
            interest = 0.0

        proportional_other = traded_notional * (self.other_cost_bps / 10000.0)
        fixed_other = self.fixed_other_cost if traded_notional > 0 else 0.0
        other_costs = float(other_cost) + proportional_other + fixed_other

        total = fees + slippage + funding + interest + other_costs

        return ExecutionCostEvent(
            timestamp=timestamp,
            prev_position=float(prev_position),
            new_position=float(new_position),
            event_type=transition.event_type,
            open_notional=transition.open_notional,
            close_notional=transition.close_notional,
            traded_notional=traded_notional,
            fees=fees,
            slippage=slippage,
            funding=funding,
            interest=interest,
            other_costs=other_costs,
            total_costs=total,
        )

    def calculate_execution_costs(
        self,
        positions: pd.Series,
        volatilities: Optional[pd.Series] = None,
        funding_rates: Optional[pd.Series] = None,
        borrow_rates_per_day: Optional[pd.Series] = None,
        other_costs: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Calculate per-timestamp execution costs from position time series.

        Returns a full audit table with per-component costs and transition metadata.
        """
        if positions.empty:
            return pd.DataFrame(
                columns=[
                    "event_type",
                    "prev_position",
                    "position",
                    "open_notional",
                    "close_notional",
                    "traded_notional",
                    "fees",
                    "slippage",
                    "funding",
                    "interest",
                    "other_costs",
                    "total_costs",
                ]
            )

        if not isinstance(positions.index, pd.DatetimeIndex):
            raise ValueError("positions must have a DatetimeIndex")

        idx = positions.index
        vols = (
            volatilities.reindex(idx).fillna(0.02)
            if volatilities is not None
            else pd.Series(0.02, index=idx)
        )
        funding = (
            funding_rates.reindex(idx).fillna(0.0)
            if funding_rates is not None
            else pd.Series(0.0, index=idx)
        )
        borrow = (
            borrow_rates_per_day.reindex(idx)
            if borrow_rates_per_day is not None
            else pd.Series(np.nan, index=idx)
        )
        other = (
            other_costs.reindex(idx).fillna(0.0)
            if other_costs is not None
            else pd.Series(0.0, index=idx)
        )

        bar_seconds = self._infer_bar_seconds(idx)
        prev_positions = positions.shift(1).fillna(0.0)

        rows: list[dict[str, float | str]] = []
        for ts in idx:
            event = self.calculate_event_costs(
                prev_position=float(prev_positions.loc[ts]),
                new_position=float(positions.loc[ts]),
                volatility=float(vols.loc[ts]),
                funding_rate=float(funding.loc[ts]),
                borrow_rate_per_day=borrow.loc[ts],
                bar_seconds=bar_seconds,
                other_cost=float(other.loc[ts]),
                timestamp=ts,
            )
            rows.append(
                {
                    "event_type": event.event_type,
                    "prev_position": event.prev_position,
                    "position": event.new_position,
                    "open_notional": event.open_notional,
                    "close_notional": event.close_notional,
                    "traded_notional": event.traded_notional,
                    "fees": event.fees,
                    "slippage": event.slippage,
                    "funding": event.funding,
                    "interest": event.interest,
                    "other_costs": event.other_costs,
                    "total_costs": event.total_costs,
                }
            )

        return pd.DataFrame(rows, index=idx)

    def calculate_costs(
        self,
        position_change: float,
        volatility: float,
        notional_value: float = 1.0,
    ) -> TradeCosts:
        """
        Backward-compatible single-event cost helper.

        `position_change` is treated as traded notional delta for one execution event.
        """
        traded = abs(position_change) * notional_value
        if traded == 0:
            return TradeCosts.zero()

        event = self.calculate_event_costs(
            prev_position=0.0,
            new_position=traded,
            volatility=volatility,
            funding_rate=0.0,
            borrow_rate_per_day=None,
            other_cost=0.0,
        )
        return TradeCosts(
            fees=event.fees,
            slippage=event.slippage,
            funding=event.funding,
            interest=event.interest,
            other_costs=event.other_costs,
            total=event.total_costs,
        )

    def calculate_costs_series(
        self,
        position_changes: pd.Series,
        volatilities: pd.Series,
    ) -> pd.DataFrame:
        """
        Backward-compatible series helper from trade deltas (without funding/interest).

        Prefer `calculate_execution_costs` when full position series is available.
        """
        if position_changes.empty:
            return pd.DataFrame(columns=["fees", "slippage", "funding", "interest", "other_costs", "total_costs"])

        common_idx = position_changes.index.intersection(volatilities.index)
        changes = position_changes.loc[common_idx].abs()
        vols = volatilities.loc[common_idx].fillna(0.02)

        fees = changes * self.fee_rate
        slippage_rate = (
            self.slippage_base_bps / 10000.0
            + self.slippage_vol_multiplier * vols
            + self.slippage_size_coefficient * changes
        )
        slippage = changes * slippage_rate.clip(lower=0.0)
        funding = pd.Series(0.0, index=common_idx)
        interest = pd.Series(0.0, index=common_idx)
        other = pd.Series(0.0, index=common_idx)
        total = fees + slippage

        return pd.DataFrame(
            {
                "fees": fees,
                "slippage": slippage,
                "funding": funding,
                "interest": interest,
                "other_costs": other,
                "total_costs": total,
            },
            index=common_idx,
        )


def main() -> None:
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    model = CostModel(exchange="binance", market_type="futures", order_type="taker")

    idx = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    positions = pd.Series([0.0, 1.0, 1.5, -0.5, 0.0], index=idx)
    vols = pd.Series([0.02] * len(idx), index=idx)
    funding = pd.Series([0.0, 0.0001, 0.0002, -0.0001, 0.0], index=idx)

    df = model.calculate_execution_costs(positions, volatilities=vols, funding_rates=funding)
    print(df)


if __name__ == "__main__":
    main()
