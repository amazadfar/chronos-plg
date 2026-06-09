"""
Market regime detection and strategy gating.

Detects regimes (trend, chop, panic) and adjusts strategy
parameters accordingly.
"""
import pandas as pd
import numpy as np
from typing import Optional
from enum import Enum
import logging

from config.settings import get_settings, StrategyConfig

logger = logging.getLogger(__name__)


class Regime(Enum):
    """Market regime classifications."""
    NORMAL = "normal"
    TREND = "trend"
    CHOP = "chop"
    PANIC = "panic"


class RegimeDetector:
    """
    Detect current market regime from price data.
    
    Regimes:
    - TREND: Strong directional move with controlled volatility
    - CHOP: Low directional move, low volatility (range-bound)
    - PANIC: Very high volatility (crisis mode)
    - NORMAL: Everything else
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        lookback_days: int = 7,
        candles_per_day: int = 6,  # 4h candles
    ):
        """
        Args:
            config: Strategy configuration
            lookback_days: Window for regime calculation
            candles_per_day: Number of candles per day
        """
        self.config = config or get_settings().strategy
        self.lookback = lookback_days * candles_per_day
    
    def detect_regime(
        self,
        returns_7d: float,
        vol_7d: float,
    ) -> Regime:
        """
        Detect regime from 7-day metrics.
        
        Args:
            returns_7d: 7-day cumulative return (absolute)
            vol_7d: 7-day realized volatility
            
        Returns:
            Regime enum value
        """
        abs_return = abs(returns_7d)
        
        # Panic: very high volatility
        if vol_7d > self.config.panic_vol_threshold:
            return Regime.PANIC
        
        # Trend: strong move with controlled vol
        if abs_return > self.config.trend_return_threshold and vol_7d < self.config.trend_vol_threshold:
            return Regime.TREND
        
        # Chop: low movement, low vol
        if abs_return < self.config.chop_return_threshold and vol_7d < self.config.chop_vol_threshold:
            return Regime.CHOP
        
        return Regime.NORMAL
    
    def detect_regimes(
        self,
        data: pd.DataFrame,
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Detect regimes for a full DataFrame.
        
        Args:
            data: DataFrame with price data
            close_col: Column name for close prices
            
        Returns:
            DataFrame with regime columns
        """
        close = data[close_col]
        
        # Calculate lookback metrics
        returns_1 = np.log(close / close.shift(1))
        returns_7d = np.log(close / close.shift(self.lookback))
        vol_7d = returns_1.rolling(window=self.lookback).std()
        
        # Classify each point
        regimes = pd.DataFrame(index=data.index)
        regimes["regime"] = Regime.NORMAL.value
        regimes["returns_7d"] = returns_7d
        regimes["vol_7d"] = vol_7d
        
        for i, (idx, row) in enumerate(regimes.iterrows()):
            if pd.notna(row["returns_7d"]) and pd.notna(row["vol_7d"]):
                regime = self.detect_regime(row["returns_7d"], row["vol_7d"])
                regimes.loc[idx, "regime"] = regime.value
        
        # Add regime dummies
        for regime in Regime:
            regimes[f"regime_{regime.value}"] = (regimes["regime"] == regime.value).astype(int)
        
        # Log distribution
        dist = regimes["regime"].value_counts()
        logger.info(f"Regime distribution: {dist.to_dict()}")
        
        return regimes
    
    def get_regime_multiplier(
        self,
        regime: Regime,
    ) -> float:
        """
        Get position size multiplier for a regime.
        
        Args:
            regime: Current regime
            
        Returns:
            Multiplier (0-1) for position sizing
        """
        if regime == Regime.PANIC:
            return self.config.panic_size_multiplier
        elif regime == Regime.CHOP:
            return self.config.chop_size_multiplier
        elif regime == Regime.TREND:
            return 1.0  # Full size in trending markets
        else:  # NORMAL
            return 0.8  # Slightly reduced for normal
    
    def get_regime_multipliers(
        self,
        regimes: pd.Series,
    ) -> pd.Series:
        """
        Get position multipliers for a series of regimes.
        
        Args:
            regimes: Series of regime labels
            
        Returns:
            Series of multipliers
        """
        multipliers = pd.Series(1.0, index=regimes.index)
        
        multipliers[regimes == Regime.PANIC.value] = self.config.panic_size_multiplier
        multipliers[regimes == Regime.CHOP.value] = self.config.chop_size_multiplier
        multipliers[regimes == Regime.TREND.value] = 1.0
        multipliers[regimes == Regime.NORMAL.value] = 0.8
        
        return multipliers


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    # Simulate price with different regimes
    returns = np.random.normal(0, 0.02, n)
    returns[100:150] += 0.005  # Bull trend
    returns[200:250] -= 0.008  # Bear trend
    returns[350:380] *= 3  # High vol period
    
    price = 40000 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({"close": price}, index=dates)
    
    # Detect regimes
    detector = RegimeDetector()
    regimes = detector.detect_regimes(data)
    
    print("\nRegime Distribution:")
    print(regimes["regime"].value_counts())
    
    print("\nMultipliers:")
    multipliers = detector.get_regime_multipliers(regimes["regime"])
    print(multipliers.describe())


if __name__ == "__main__":
    main()
