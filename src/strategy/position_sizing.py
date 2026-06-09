"""
Position sizing based on predictions and risk.

Implements volatility-adjusted position sizing with:
- Signal strength weighting
- Confidence-based scaling
- Risk limits and leverage caps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import StrategyConfig, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionConstraints:
    """Exchange precision/minimum order constraints."""

    tick_size: float = 0.0
    lot_size: float = 0.0
    min_qty: float = 0.0
    min_notional: float = 0.0

    @classmethod
    def from_mapping(cls, mapping: dict[str, float]) -> "PositionConstraints":
        return cls(
            tick_size=float(mapping.get("tick_size", 0.0)),
            lot_size=float(mapping.get("lot_size", 0.0)),
            min_qty=float(mapping.get("min_qty", 0.0)),
            min_notional=float(mapping.get("min_notional", 0.0)),
        )


class PositionSizer:
    """
    Calculate position sizes from signals and risk metrics.
    
    Sizing formula:
        raw_size = signal * strength * confidence * base_size
        vol_adjusted = raw_size / predicted_vol
        final_size = clip(vol_adjusted, -max_leverage, max_leverage)
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        max_leverage: Optional[float] = None,
        vol_target: float = 0.15,  # Annual vol target
        min_position: float = 0.01,  # Minimum position size
        market_type: str = "futures",
        leverage_caps_by_market: Optional[dict[str, float]] = None,
        default_max_turnover_per_step: Optional[float] = None,
        allow_short: bool = True,
    ):
        """
        Args:
            config: Strategy configuration
            max_leverage: Maximum leverage (overrides config)
            vol_target: Target annualized volatility
            min_position: Minimum position size (below this = flat)
            market_type: spot, margin, or futures
            leverage_caps_by_market: Optional market-type leverage caps
            default_max_turnover_per_step: Optional per-step turnover cap
            allow_short: Whether shorts are allowed by default
        """
        self.config = config or get_settings().strategy
        self.max_leverage = max_leverage or self.config.max_leverage
        self.vol_target = vol_target
        self.min_position = min_position
        self.market_type = market_type.lower()
        self.default_max_turnover_per_step = default_max_turnover_per_step
        self.allow_short = allow_short
        self.leverage_caps_by_market = leverage_caps_by_market or {
            "spot": 1.0,
            "margin": max(1.0, self.max_leverage),
            "futures": self.max_leverage,
        }
        
        # Annualization factor for 4h data
        self.periods_per_year = 6 * 365  # 6 4h candles per day

    def _effective_leverage_cap(self) -> float:
        market_cap = float(self.leverage_caps_by_market.get(self.market_type, self.max_leverage))
        cap = min(float(self.max_leverage), market_cap)
        return max(cap, 0.0)

    @staticmethod
    def _turnover_limited_target(prev: float, target: float, turnover_cap: float) -> float:
        low = prev - turnover_cap
        high = prev + turnover_cap
        return float(np.clip(target, low, high))

    def _apply_turnover_cap(
        self,
        positions: pd.Series,
        turnover_cap: Optional[float],
    ) -> pd.Series:
        if turnover_cap is None or turnover_cap <= 0 or positions.empty:
            return positions

        capped = pd.Series(0.0, index=positions.index, dtype=float)
        prev = 0.0
        for ts in positions.index:
            target = float(positions.loc[ts])
            adjusted = self._turnover_limited_target(prev, target, turnover_cap)
            capped.loc[ts] = adjusted
            prev = adjusted
        return capped

    def _enforce_short_constraints(
        self,
        positions: pd.Series,
        short_allowed: pd.Series | bool | None,
    ) -> pd.Series:
        constrained = positions.copy()

        if self.market_type == "spot" or not self.allow_short:
            constrained[constrained < 0] = 0.0
            return constrained

        if isinstance(short_allowed, pd.Series):
            allowed = short_allowed.reindex(constrained.index).fillna(False).astype(bool)
            constrained[(constrained < 0) & (~allowed)] = 0.0
        elif short_allowed is False:
            constrained[constrained < 0] = 0.0

        return constrained

    def _apply_order_constraints(
        self,
        position: float,
        price: float,
        equity: float,
        constraints: PositionConstraints,
    ) -> float:
        if abs(position) <= 1e-12:
            return 0.0
        if price <= 0 or equity <= 0:
            return 0.0

        qty = abs(position) * equity / price
        if constraints.lot_size > 0:
            qty = np.floor(qty / constraints.lot_size) * constraints.lot_size

        if constraints.min_qty > 0 and qty < constraints.min_qty:
            return 0.0

        notional = qty * price
        if constraints.min_notional > 0 and notional < constraints.min_notional:
            return 0.0

        adjusted = float(np.sign(position) * (notional / equity))
        cap = self._effective_leverage_cap()
        return float(np.clip(adjusted, -cap, cap))
    
    def calculate_size(
        self,
        signal: int,
        strength: float,
        confidence: float,
        predicted_vol: float,
        regime_multiplier: float = 1.0,
        price: Optional[float] = None,
        equity: float = 1.0,
        constraints: Optional[PositionConstraints] = None,
        short_allowed: bool = True,
    ) -> float:
        """
        Calculate position size for a single signal.
        
        Args:
            signal: Direction (-1, 0, 1)
            strength: Signal strength (0-1)
            confidence: Signal confidence (0-1)
            predicted_vol: Predicted volatility for the period
            regime_multiplier: Regime-based adjustment factor
            price: Optional instrument price for precision/minimum order checks
            equity: Base equity used to map leverage target to qty/notional
            constraints: Optional precision/minimum order constraints
            short_allowed: Whether shorting is currently available
            
        Returns:
            Position size in leverage units (-max to +max)
        """
        if signal == 0 or (signal < 0 and (self.market_type == "spot" or not self.allow_short)):
            return 0.0
        if signal < 0 and not short_allowed:
            return 0.0
        
        # Base size from signal and confidence
        base_size = signal * strength * confidence
        
        # Volatility adjustment (target vol / predicted vol)
        # This ensures consistent risk across different vol regimes
        if predicted_vol > 0:
            # Convert period vol to annual
            annual_vol = predicted_vol * np.sqrt(self.periods_per_year)
            vol_scalar = self.vol_target / annual_vol
        else:
            vol_scalar = 1.0
        
        # Apply volatility scaling
        vol_adjusted = base_size * vol_scalar
        
        # Apply regime adjustment
        adjusted = vol_adjusted * regime_multiplier
        
        # Clip to leverage limits
        cap = self._effective_leverage_cap()
        position = np.clip(adjusted, -cap, cap)
        
        # Filter out tiny positions
        if abs(position) < self.min_position:
            return 0.0

        if constraints is not None and price is not None:
            position = self._apply_order_constraints(
                position=float(position),
                price=float(price),
                equity=float(equity),
                constraints=constraints,
            )
            if abs(position) < self.min_position:
                return 0.0
        
        return float(position)
    
    def calculate_sizes(
        self,
        signals: pd.DataFrame,
        predicted_vol: pd.Series,
        regime_multipliers: Optional[pd.Series] = None,
        prices: Optional[pd.Series] = None,
        position_constraints: Optional[PositionConstraints] = None,
        short_allowed: pd.Series | bool | None = None,
        max_turnover_per_step: Optional[float] = None,
        equity: float = 1.0,
    ) -> pd.Series:
        """
        Calculate position sizes for a DataFrame of signals.
        
        Args:
            signals: DataFrame with signal, signal_strength, signal_confidence
            predicted_vol: Series of predicted volatilities
            regime_multipliers: Optional regime adjustment factors
            prices: Optional instrument prices for precision/minimum order checks
            position_constraints: Optional precision/minimum order constraints
            short_allowed: Series/bool gating short eligibility
            max_turnover_per_step: Optional max position change per timestamp
            equity: Base equity used to map leverage target to qty/notional
            
        Returns:
            Series of position sizes
        """
        # Align data
        common_idx = signals.index.intersection(predicted_vol.index)
        if prices is not None:
            common_idx = common_idx.intersection(prices.index)
        
        sig = signals.loc[common_idx, "signal"]
        strength = signals.loc[common_idx, "signal_strength"]
        confidence = signals.loc[common_idx, "signal_confidence"]
        vol = predicted_vol.loc[common_idx]
        px = prices.loc[common_idx] if prices is not None else None
        
        if regime_multipliers is not None:
            regime = regime_multipliers.reindex(common_idx, fill_value=1.0)
        else:
            regime = pd.Series(1.0, index=common_idx)
        
        # Vectorized calculation
        base_size = sig * strength * confidence
        
        # Vol adjustment
        annual_vol = vol * np.sqrt(self.periods_per_year)
        vol_scalar = self.vol_target / annual_vol.replace(0, np.inf)
        vol_scalar = vol_scalar.clip(0.1, 10)  # Reasonable bounds
        
        # Apply adjustments
        adjusted = base_size * vol_scalar * regime
        
        # Clip to limits
        cap = self._effective_leverage_cap()
        positions = adjusted.clip(-cap, cap)
        positions = self._enforce_short_constraints(positions, short_allowed)

        if position_constraints is not None and px is not None:
            constrained = positions.copy()
            for ts in common_idx:
                constrained.loc[ts] = self._apply_order_constraints(
                    position=float(positions.loc[ts]),
                    price=float(px.loc[ts]),
                    equity=float(equity),
                    constraints=position_constraints,
                )
            positions = constrained

        turnover_cap = (
            max_turnover_per_step
            if max_turnover_per_step is not None
            else self.default_max_turnover_per_step
        )
        positions = self._apply_turnover_cap(positions, turnover_cap)
        
        # Filter tiny positions
        positions[positions.abs() < self.min_position] = 0.0
        
        positions.name = "position_size"
        
        # Log statistics
        long_pct = (positions > 0).mean()
        short_pct = (positions < 0).mean()
        avg_size = positions[positions != 0].abs().mean() if (positions != 0).any() else 0
        
        logger.info(
            f"Positions: {long_pct:.1%} long, {short_pct:.1%} short, "
            f"avg size: {avg_size:.2f}x"
        )
        
        return positions


class KellyCriterionSizer:
    """
    Kelly Criterion based position sizing.
    
    Uses the probabilistic forecasts to estimate edge and variance,
    then applies fractional Kelly for position sizing.
    
    k = (p * b - q) / b
    where:
        p = probability of winning
        q = 1 - p
        b = win/loss ratio
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,  # Use 25% Kelly (conservative)
        max_leverage: float = 2.0,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage
    
    def calculate_kelly(
        self,
        q10: float,
        q50: float,
        q90: float,
    ) -> float:
        """
        Calculate Kelly-optimal position from quantiles.
        
        Approximates win probability and win/loss ratio from
        the predicted return distribution.
        """
        # Estimate win probability (P(return > 0))
        # Using linear interpolation between quantiles
        if q50 > 0:
            # Above median is positive
            if q10 > 0:
                win_prob = 0.9 + 0.1 * (1 - q10 / q90)  # Almost certain positive
            else:
                # Interpolate: q10 is negative, q50 is positive
                win_prob = 0.5 + 0.4 * (q50 / (q50 - q10))
        else:
            if q90 < 0:
                win_prob = 0.1 * (1 + q90 / (-q10))  # Almost certain negative
            else:
                win_prob = 0.5 * (q90 / (q90 - q50))
        
        win_prob = np.clip(win_prob, 0.01, 0.99)
        loss_prob = 1 - win_prob
        
        # Estimate expected win and loss magnitudes
        expected_win = max(q90, q50) if q50 > 0 else q90
        expected_loss = abs(min(q10, q50)) if q50 <= 0 else abs(q10)
        
        if expected_loss < 0.001:
            expected_loss = 0.001
        
        # Win/loss ratio
        b = expected_win / expected_loss
        
        # Kelly formula
        kelly = (win_prob * b - loss_prob) / b
        
        # Apply fraction and limits
        position = kelly * self.kelly_fraction
        position = np.clip(position, -self.max_leverage, self.max_leverage)
        
        return position


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    signals = pd.DataFrame({
        "signal": np.random.choice([-1, 0, 1], n, p=[0.2, 0.5, 0.3]),
        "signal_strength": np.random.uniform(0.3, 1.0, n),
        "signal_confidence": np.random.uniform(0.5, 1.0, n),
    }, index=dates)
    
    predicted_vol = pd.Series(np.random.uniform(0.01, 0.04, n), index=dates)
    
    # Calculate positions
    sizer = PositionSizer(max_leverage=2.0, vol_target=0.15)
    positions = sizer.calculate_sizes(signals, predicted_vol)
    
    print("\nPosition Distribution:")
    print(positions.describe())


if __name__ == "__main__":
    main()
