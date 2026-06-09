"""
LightGBM Quantile Baseline.

The strongest tabular baseline - uses gradient boosting for 
quantile regression on engineered features.

If you can't beat this, there's no point using Chronos-2.
"""
import pandas as pd
import numpy as np
from typing import Optional, Any
import logging
import lightgbm as lgb

from src.models.baselines.random_walk import BaselineModel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class LightGBMQuantileBaseline(BaselineModel):
    """
    LightGBM-based quantile regression baseline.
    
    Trains separate models for each quantile using the quantile loss
    (pinball loss). This is the primary baseline to beat.
    
    Features should include:
    - Price features (returns, volatility)
    - Perps features (funding rate, OI)
    - Macro features (optional)
    """
    
    def __init__(
        self,
        quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
        lgb_params: Optional[dict[str, Any]] = None,
        n_estimators: int = 500,
        early_stopping_rounds: int = 50,
        feature_columns: Optional[list[str]] = None,
    ):
        """
        Args:
            quantiles: Quantiles to predict
            lgb_params: LightGBM parameters (will use defaults if None)
            n_estimators: Number of boosting iterations
            early_stopping_rounds: Early stopping patience
            feature_columns: Columns to use as features (None = all numeric)
        """
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_columns = feature_columns
        
        settings = get_settings().model
        
        # Default LightGBM params optimized for time series
        self.lgb_params = lgb_params or {
            "num_leaves": settings.lgb_num_leaves,
            "learning_rate": settings.lgb_learning_rate,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": settings.seed,
            "n_jobs": -1,
        }
        
        self._models: dict[str, lgb.Booster] = {}
        self._feature_names: list[str] = []
        self._fitted = False
    
    @property
    def name(self) -> str:
        return "LightGBM"
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and prepare features for training."""
        if self.feature_columns:
            # Use specified columns
            cols = [c for c in self.feature_columns if c in X.columns]
        else:
            # Use all numeric columns except obvious non-features
            exclude_patterns = [
                "forward_", "regime", "hist_q", "timestamp",
                "open", "high", "low", "close", "volume"
            ]
            cols = []
            for c in X.columns:
                if X[c].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    if not any(p in c for p in exclude_patterns):
                        cols.append(c)
        
        return X[cols].copy()
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMQuantileBaseline":
        """
        Fit separate LightGBM models for each quantile.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets
        """
        X_train = self._prepare_features(X)
        self._feature_names = list(X_train.columns)
        
        # Remove rows with NaN
        valid_mask = ~(X_train.isna().any(axis=1) | y.isna())
        X_train = X_train[valid_mask]
        y_train = y[valid_mask]
        
        logger.info(f"Training LightGBM on {len(X_train)} samples, {len(self._feature_names)} features")
        
        # Prepare validation set if provided
        if X_val is not None and y_val is not None:
            X_v = self._prepare_features(X_val)
            valid_mask_v = ~(X_v.isna().any(axis=1) | y_val.isna())
            X_v = X_v[valid_mask_v]
            y_v = y_val[valid_mask_v]
        else:
            # Use last 20% as validation
            split_idx = int(len(X_train) * 0.8)
            X_v = X_train.iloc[split_idx:]
            y_v = y_train.iloc[split_idx:]
            X_train = X_train.iloc[:split_idx]
            y_train = y_train.iloc[:split_idx]
        
        # Train a model for each quantile
        for q in self.quantiles:
            col = f"q{int(q*100)}"
            logger.info(f"Training quantile {q}...")
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_v, label=y_v, reference=train_data)
            
            # Set quantile-specific params
            params = self.lgb_params.copy()
            params["objective"] = "quantile"
            params["alpha"] = q  # quantile level
            
            # Train with early stopping
            callbacks = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),  # Suppress logging
            ]
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=[val_data],
                callbacks=callbacks,
            )
            
            self._models[col] = model
            logger.info(f"  {col}: {model.best_iteration} iterations")
        
        self._fitted = True
        logger.info(f"LightGBM baseline fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict quantiles using trained models."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_pred = self._prepare_features(X)
        
        # Ensure columns match training
        for col in self._feature_names:
            if col not in X_pred.columns:
                X_pred[col] = np.nan
        
        X_pred = X_pred[self._feature_names]
        
        # Handle NaN by filling with column medians (from training)
        X_pred = X_pred.fillna(X_pred.median())
        
        predictions = pd.DataFrame(index=X.index)
        
        for q in self.quantiles:
            col = f"q{int(q*100)}"
            if col in self._models:
                predictions[col] = self._models[col].predict(X_pred)
            else:
                predictions[col] = 0.0
        
        return predictions
    
    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Get feature importance from trained models.
        
        Args:
            importance_type: 'gain', 'split', or 'cover'
            
        Returns:
            DataFrame with importance for each quantile model
        """
        if not self._fitted:
            raise ValueError("Model not fitted.")
        
        importance_dict = {"feature": self._feature_names}
        
        for q in self.quantiles:
            col = f"q{int(q*100)}"
            if col in self._models:
                importance_dict[col] = self._models[col].feature_importance(importance_type)
        
        df = pd.DataFrame(importance_dict)
        df["mean_importance"] = df[[f"q{int(q*100)}" for q in self.quantiles]].mean(axis=1)
        df = df.sort_values("mean_importance", ascending=False)
        
        return df


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    # Features with some predictive power
    momentum = np.cumsum(np.random.randn(n)) * 0.001
    volatility = np.abs(np.random.randn(n)) * 0.01 + 0.01
    funding = np.cumsum(np.random.randn(n)) * 0.0001
    
    features = pd.DataFrame({
        "return_1": np.random.randn(n) * 0.02,
        "return_6": np.random.randn(n) * 0.05,
        "realized_vol_6": volatility,
        "funding_rate": funding,
        "oi_change_pct_1": np.random.randn(n) * 0.05,
    }, index=dates)
    
    # Target with slight dependence on features
    target = (
        0.1 * features["return_1"] + 
        0.05 * features["funding_rate"] +
        np.random.randn(n) * 0.02
    )
    target = pd.Series(target.values, index=dates, name="forward_return")
    
    # Split
    train_end = int(n * 0.8)
    X_train = features.iloc[:train_end]
    y_train = target.iloc[:train_end]
    X_test = features.iloc[train_end:]
    y_test = target.iloc[train_end:]
    
    # Train and predict
    model = LightGBMQuantileBaseline(n_estimators=100)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    print("\nLightGBM Predictions:")
    print(predictions.describe())
    
    print("\nFeature Importance:")
    print(model.get_feature_importance().head(10))
    
    # Check calibration
    for q in model.quantiles:
        col = f"q{int(q*100)}"
        coverage = (y_test < predictions[col]).mean()
        print(f"\n{col} coverage: {coverage:.3f} (expected: {q:.2f})")


if __name__ == "__main__":
    main()
