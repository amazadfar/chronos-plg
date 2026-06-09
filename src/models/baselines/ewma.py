"""
EWMA (Exponentially Weighted Moving Average) Baseline.

Slightly smarter than random walk: uses exponentially weighted
mean and std to capture recent momentum and volatility regimes.
"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

from src.models.baselines.random_walk import BaselineModel

logger = logging.getLogger(__name__)


class EWMABaseline(BaselineModel):
    """
    EWMA baseline for return quantile prediction.
    
    Uses exponentially weighted statistics to predict:
    - q50 = EWMA mean (captures momentum)
    - q10, q90 = q50 ± k * EWMA std (parametric quantiles)
    
    This baseline captures:
    - Short-term momentum (mean != 0)
    - Volatility clustering (time-varying spread)
    """
    
    def __init__(
        self,
        span: int = 24,  # ~4 days in 4h candles
        quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
        min_periods: int = 12,
    ):
        """
        Args:
            span: Span for exponential weighting (half-life ≈ span / 2.7)
            quantiles: Quantiles to predict
            min_periods: Minimum observations for valid estimate
        """
        self.span = span
        self.quantiles = quantiles
        self.min_periods = min_periods
        self._fitted = False
        self._ewm_mean: Optional[pd.Series] = None
        self._ewm_std: Optional[pd.Series] = None
        
        # Z-scores for quantiles (assuming normal distribution)
        from scipy import stats
        self._z_scores = {q: stats.norm.ppf(q) for q in quantiles}
    
    @property
    def name(self) -> str:
        return f"EWMA(span={self.span})"
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EWMABaseline":
        """
        Fit by computing EWMA statistics on historical returns.
        
        Args:
            X: Features (not used)
            y: Target returns
        """
        # Compute EWMA mean and std
        ewm = y.ewm(span=self.span, min_periods=self.min_periods)
        self._ewm_mean = ewm.mean()
        self._ewm_std = ewm.std()
        
        self._fitted = True
        
        logger.info(
            f"EWMA baseline fitted on {len(y)} samples, "
            f"span={self.span}, final mean={self._ewm_mean.iloc[-1]:.6f}"
        )
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict quantiles using EWMA mean and std.
        
        Uses Gaussian assumption: quantile = mean + z * std
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = pd.DataFrame(index=X.index)
        
        # Reindex EWMA estimates to prediction index
        mean = self._ewm_mean.reindex(X.index)
        std = self._ewm_std.reindex(X.index)
        
        # Forward fill for new timestamps
        if mean.isna().any():
            last_mean = self._ewm_mean.dropna().iloc[-1]
            last_std = self._ewm_std.dropna().iloc[-1]
            mean = mean.fillna(last_mean)
            std = std.fillna(last_std)
        
        # Fill any remaining NaN in std with a reasonable default
        std = std.fillna(std.mean() if std.mean() > 0 else 0.02)
        
        # Compute quantiles
        for q in self.quantiles:
            col = f"q{int(q*100)}"
            z = self._z_scores[q]
            predictions[col] = mean + z * std
        
        return predictions
    
    def update(self, new_return: float) -> None:
        """
        Incrementally update EWMA estimates with a new observation.
        
        Useful for online prediction without full refit.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        alpha = 2.0 / (self.span + 1)
        
        # Update mean: new_mean = alpha * new_value + (1 - alpha) * old_mean
        old_mean = self._ewm_mean.iloc[-1]
        new_mean = alpha * new_return + (1 - alpha) * old_mean
        
        # Update variance (using Welford-like approach for EWMA)
        old_var = self._ewm_std.iloc[-1] ** 2
        new_var = alpha * (new_return - new_mean) ** 2 + (1 - alpha) * old_var
        new_std = np.sqrt(new_var)
        
        # Append to series (for simplicity, in production you'd maintain state differently)
        self._ewm_mean = pd.concat([
            self._ewm_mean, 
            pd.Series([new_mean], index=[self._ewm_mean.index[-1] + pd.Timedelta(hours=4)])
        ])
        self._ewm_std = pd.concat([
            self._ewm_std,
            pd.Series([new_std], index=[self._ewm_std.index[-1] + pd.Timedelta(hours=4)])
        ])


class ARBaseline(BaselineModel):
    """
    Simple AR(1) baseline using OLS.
    
    Predicts: r_t = alpha + beta * r_{t-1} + epsilon
    Quantiles derived from residual distribution.
    """
    
    def __init__(
        self,
        quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
    ):
        self.quantiles = quantiles
        self._alpha: float = 0.0
        self._beta: float = 0.0
        self._residual_std: float = 0.0
        self._fitted = False
    
    @property
    def name(self) -> str:
        return "AR(1)"
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ARBaseline":
        """
        Fit AR(1) model via OLS.
        """
        # Prepare lagged data
        y_lag = y.shift(1)
        valid = ~(y.isna() | y_lag.isna())
        
        y_valid = y[valid].values
        y_lag_valid = y_lag[valid].values
        
        if len(y_valid) < 10:
            logger.warning("Insufficient data for AR(1), using defaults")
            self._alpha = 0.0
            self._beta = 0.0
            self._residual_std = y.std()
        else:
            # OLS: y = alpha + beta * y_lag
            X_ols = np.column_stack([np.ones(len(y_lag_valid)), y_lag_valid])
            coeffs = np.linalg.lstsq(X_ols, y_valid, rcond=None)[0]
            
            self._alpha = coeffs[0]
            self._beta = coeffs[1]
            
            # Residual std
            predictions = self._alpha + self._beta * y_lag_valid
            residuals = y_valid - predictions
            self._residual_std = np.std(residuals)
        
        self._last_return = y.dropna().iloc[-1]
        self._fitted = True
        
        logger.info(
            f"AR(1) fitted: alpha={self._alpha:.6f}, beta={self._beta:.4f}, "
            f"residual_std={self._residual_std:.6f}"
        )
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict using AR(1) with Gaussian quantiles."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        from scipy import stats
        
        predictions = pd.DataFrame(index=X.index)
        
        # Point prediction: alpha + beta * last_return
        point_pred = self._alpha + self._beta * self._last_return
        
        for q in self.quantiles:
            col = f"q{int(q*100)}"
            z = stats.norm.ppf(q)
            predictions[col] = point_pred + z * self._residual_std
        
        return predictions


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data with momentum
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    # Returns with some autocorrelation
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = 0.1 * returns[i-1] + np.random.normal(0, 0.02)
    
    returns = pd.Series(returns, index=dates)
    features = pd.DataFrame({"dummy": np.ones(n)}, index=dates)
    
    # Test EWMA
    ewma = EWMABaseline(span=24)
    ewma.fit(features, returns)
    ewma_preds = ewma.predict(features)
    
    print("\nEWMA Predictions:")
    print(ewma_preds.describe())
    
    # Test AR(1)
    ar = ARBaseline()
    ar.fit(features, returns)
    ar_preds = ar.predict(features)
    
    print(f"\nAR(1) beta (autocorrelation): {ar._beta:.4f}")
    print("AR(1) Predictions:")
    print(ar_preds.describe())


if __name__ == "__main__":
    main()
