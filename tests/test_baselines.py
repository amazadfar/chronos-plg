"""Tests for baseline models and walk-forward evaluation."""
import pytest
import pandas as pd
import numpy as np
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baselines import RandomWalkBaseline, EWMABaseline
from src.evaluation import WalkForwardEvaluator, QuantileMetrics, TradingMetrics
from src.evaluation.metrics import compute_quantile_metrics, compute_trading_metrics, pinball_loss

if importlib.util.find_spec("lightgbm") is not None:
    from src.models.baselines import LightGBMQuantileBaseline
else:  # pragma: no cover - depends on environment.
    LightGBMQuantileBaseline = None


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    # Returns with slight autocorrelation
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = 0.05 * returns[i-1] + np.random.normal(0, 0.02)
    
    data = pd.DataFrame({
        "close": 40000 * np.exp(np.cumsum(returns)),
        "return_1": returns,
        "return_6": pd.Series(returns).rolling(6).sum().values,
        "realized_vol_6": pd.Series(returns).rolling(6).std().values,
        "forward_return": np.roll(returns, -1),
        "regime": np.where(np.abs(returns) > 0.03, "trend", "normal"),
    }, index=dates)
    data.loc[data.index[-1], "forward_return"] = np.nan
    
    return data


class TestRandomWalkBaseline:
    """Tests for RandomWalkBaseline."""
    
    def test_fit_predict(self, sample_data):
        """Model should fit and predict without errors."""
        model = RandomWalkBaseline(lookback_window=50)
        
        X = sample_data[["return_1", "return_6"]]
        y = sample_data["forward_return"].dropna()
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert "q10" in predictions.columns
        assert "q50" in predictions.columns
        assert "q90" in predictions.columns
        
    def test_q50_is_zero(self, sample_data):
        """Random walk should predict q50 = 0."""
        model = RandomWalkBaseline()
        
        X = sample_data[["return_1"]]
        y = sample_data["forward_return"].dropna()
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert (predictions["q50"] == 0).all()
    
    def test_quantile_ordering(self, sample_data):
        """q10 < q50 < q90 should hold."""
        model = RandomWalkBaseline(lookback_window=50)
        
        X = sample_data[["return_1"]]
        y = sample_data["forward_return"].dropna()
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Drop rows with NaN (from rolling window warmup)
        valid = predictions.dropna()
        
        assert (valid["q10"] <= valid["q50"]).all()
        assert (valid["q50"] <= valid["q90"]).all()


class TestEWMABaseline:
    """Tests for EWMABaseline."""
    
    def test_fit_predict(self, sample_data):
        """Model should fit and predict without errors."""
        model = EWMABaseline(span=24)
        
        X = sample_data[["return_1"]]
        y = sample_data["forward_return"].dropna()
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert "q10" in predictions.columns
        assert "q50" in predictions.columns
        assert "q90" in predictions.columns
    
    def test_captures_momentum(self, sample_data):
        """EWMA mean should capture recent returns."""
        model = EWMABaseline(span=10)
        
        X = sample_data[["return_1"]]
        y = sample_data["forward_return"].dropna()
        
        model.fit(X, y)
        
        # q50 should be close to recent mean
        assert model._ewm_mean is not None
        assert not model._ewm_mean.isna().all()


@pytest.mark.skipif(LightGBMQuantileBaseline is None, reason="lightgbm is not installed")
class TestLightGBMBaseline:
    """Tests for LightGBMQuantileBaseline."""
    
    def test_fit_predict(self, sample_data):
        """Model should fit and predict without errors."""
        model = LightGBMQuantileBaseline(n_estimators=50)
        
        feature_cols = ["return_1", "return_6", "realized_vol_6"]
        X = sample_data[feature_cols].dropna()
        y = sample_data.loc[X.index, "forward_return"].dropna()
        X = X.loc[y.index]
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert "q10" in predictions.columns
        assert "q50" in predictions.columns
        assert "q90" in predictions.columns
    
    def test_feature_importance(self, sample_data):
        """Should return feature importance after fitting."""
        model = LightGBMQuantileBaseline(n_estimators=50)
        
        feature_cols = ["return_1", "return_6", "realized_vol_6"]
        X = sample_data[feature_cols].dropna()
        y = sample_data.loc[X.index, "forward_return"].dropna()
        X = X.loc[y.index]
        
        model.fit(X, y)
        importance = model.get_feature_importance()
        
        assert len(importance) == len(feature_cols)
        assert "mean_importance" in importance.columns


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_pinball_loss_symmetric(self):
        """Pinball loss at q=0.5 should be symmetric."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([0.01, 0.0, -0.01])
        
        loss = pinball_loss(y_true, y_pred, 0.5)
        
        # For q=0.5, loss should be 0.5 * |error| on average
        expected = 0.5 * np.abs(y_true - y_pred).mean()
        np.testing.assert_almost_equal(loss, expected)
    
    def test_quantile_metrics_coverage(self, sample_data):
        """Coverage should be close to quantile level for calibrated predictions."""
        y = sample_data["forward_return"].dropna()
        
        # Create well-calibrated predictions
        predictions = pd.DataFrame(index=y.index)
        predictions["q10"] = y.quantile(0.10)
        predictions["q50"] = y.quantile(0.50)
        predictions["q90"] = y.quantile(0.90)
        
        metrics = compute_quantile_metrics(y, predictions)
        
        # Coverage should be close to expected
        assert abs(metrics.coverage["q10"] - 0.10) < 0.1
        assert abs(metrics.coverage["q50"] - 0.50) < 0.1
        assert abs(metrics.coverage["q90"] - 0.90) < 0.1
    
    def test_trading_metrics_positive_sharpe(self, sample_data):
        """Perfect foresight should give positive Sharpe."""
        returns = sample_data["forward_return"].dropna()
        
        # Perfect foresight: position = sign of next return
        positions = np.sign(returns)
        positions = pd.Series(positions.values, index=returns.index)
        
        metrics = compute_trading_metrics(returns, positions)
        
        assert metrics.sharpe_ratio > 0


class TestWalkForward:
    """Tests for walk-forward evaluation."""
    
    def test_fold_generation(self, sample_data):
        """Should generate valid folds."""
        from config.settings import WalkForwardConfig
        
        config = WalkForwardConfig(
            train_window_days=30,
            test_window_days=7,
            step_size_days=7,
        )
        
        evaluator = WalkForwardEvaluator(config=config)
        folds = evaluator.generate_folds(sample_data)
        
        assert len(folds) > 0
        
        # Check fold validity
        for train_start, train_end, test_start, test_end in folds:
            assert train_start < train_end
            assert train_end < test_start
            assert test_start < test_end
    
    def test_evaluate_model(self, sample_data):
        """Should run evaluation on a model."""
        from config.settings import WalkForwardConfig
        
        config = WalkForwardConfig(
            train_window_days=30,
            test_window_days=7,
            step_size_days=14,
            min_train_samples=50,
        )
        
        evaluator = WalkForwardEvaluator(config=config)
        
        results = evaluator.evaluate_model(
            RandomWalkBaseline,
            sample_data,
            model_kwargs={"lookback_window": 50},
            show_progress=False,
        )
        
        assert results.model_name == "RandomWalk"
        assert len(results.folds) > 0
        assert results.mean_pinball_loss is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
