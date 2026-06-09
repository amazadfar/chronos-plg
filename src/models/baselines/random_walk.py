"""
Random Walk Baseline.

The simplest baseline: predicts that future returns will be drawn from 
the same distribution as historical returns.

- q50 = 0 (random walk: best guess is no change)
- q10, q90 from rolling historical distribution
"""
import pandas as pd
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaselineModel(ABC):
    """Abstract base class for all baseline models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineModel":
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict quantiles.
        
        Returns:
            DataFrame with columns: q10, q50, q90
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging/reporting."""
        pass


class RandomWalkBaseline(BaselineModel):
    """
    Random Walk baseline for return quantile prediction.
    
    Assumes returns are i.i.d. from historical distribution:
    - q50 = 0 (martingale assumption)
    - q10, q90 from rolling window of historical returns
    
    This is the "dumbest smart baseline" - if you can't beat this,
    there's no predictable signal.
    """
    
    def __init__(
        self,
        lookback_window: int = 252,  # ~6 weeks in 4h candles
        quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
    ):
        """
        Args:
            lookback_window: Number of periods for rolling quantile calculation
            quantiles: Quantiles to predict
        """
        self.lookback_window = lookback_window
        self.quantiles = quantiles
        self._fitted = False
        self._historical_returns: Optional[pd.Series] = None
        self._rolling_quantiles: Optional[pd.DataFrame] = None
    
    @property
    def name(self) -> str:
        return "RandomWalk"
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomWalkBaseline":
        """
        Fit by storing historical returns for rolling quantile calculation.
        
        Args:
            X: Features (not used, but kept for API consistency)
            y: Target returns
        """
        self._historical_returns = y.copy()
        
        # Pre-compute rolling quantiles for efficiency
        self._rolling_quantiles = pd.DataFrame(index=y.index)
        
        for q in self.quantiles:
            col = f"q{int(q*100)}"
            if q == 0.5:
                # Random walk: q50 = 0
                self._rolling_quantiles[col] = 0.0
            else:
                # Rolling quantile from historical distribution
                self._rolling_quantiles[col] = (
                    y.rolling(window=self.lookback_window, min_periods=self.lookback_window // 2)
                    .quantile(q)
                )
        
        self._fitted = True
        logger.info(f"RandomWalk baseline fitted on {len(y)} samples")
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict quantiles using rolling historical distribution.
        
        For out-of-sample prediction, uses the last known rolling quantiles.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = pd.DataFrame(index=X.index)
        
        for q in self.quantiles:
            col = f"q{int(q*100)}"
            
            if q == 0.5:
                predictions[col] = 0.0
            else:
                # Use historical rolling quantiles where available
                # For new timestamps, use the last known value
                if self._rolling_quantiles is not None:
                    available = self._rolling_quantiles[col].reindex(X.index)
                    # Forward fill for any missing (future) timestamps
                    last_known = self._rolling_quantiles[col].dropna().iloc[-1]
                    predictions[col] = available.fillna(last_known)
                else:
                    predictions[col] = 0.0
        
        return predictions
    
    def predict_single(self, timestamp: pd.Timestamp) -> dict[str, float]:
        """Predict for a single timestamp."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        result = {}
        for q in self.quantiles:
            col = f"q{int(q*100)}"
            if q == 0.5:
                result[col] = 0.0
            elif self._rolling_quantiles is not None:
                # Get closest available value
                if timestamp in self._rolling_quantiles.index:
                    result[col] = self._rolling_quantiles.loc[timestamp, col]
                else:
                    result[col] = self._rolling_quantiles[col].dropna().iloc[-1]
            else:
                result[col] = 0.0
        
        return result


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    # Simulate returns
    returns = pd.Series(np.random.normal(0, 0.02, n), index=dates)
    features = pd.DataFrame({"dummy": np.ones(n)}, index=dates)
    
    # Fit and predict
    model = RandomWalkBaseline(lookback_window=100)
    model.fit(features, returns)
    
    predictions = model.predict(features)
    
    print("\nRandom Walk Predictions:")
    print(predictions.describe())
    
    # Verify q50 is always 0
    assert (predictions["q50"] == 0).all(), "q50 should always be 0 for random walk"
    print("\n✓ q50 is always 0 (random walk assumption)")


if __name__ == "__main__":
    main()
