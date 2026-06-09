"""
Walk-Forward Evaluation Harness.

Implements proper time-series cross-validation:
- Rolling/expanding training windows
- No look-ahead bias
- Realistic retraining schedules

This is the core evaluation framework that ensures our results are valid.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Type
from dataclasses import dataclass, field
import logging
from tqdm import tqdm

from src.models.baselines.random_walk import BaselineModel
from src.evaluation.metrics import (
    compute_quantile_metrics,
    compute_trading_metrics,
    QuantileMetrics,
    TradingMetrics,
)
from config.settings import get_settings, WalkForwardConfig

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    train_samples: int = 0
    test_samples: int = 0
    
    predictions: Optional[pd.DataFrame] = None
    actuals: Optional[pd.Series] = None
    regimes: Optional[pd.Series] = None
    
    quantile_metrics: Optional[QuantileMetrics] = None
    trading_metrics: Optional[TradingMetrics] = None
    
    model_info: dict = field(default_factory=dict)


@dataclass
class WalkForwardResults:
    """Aggregated results from walk-forward evaluation."""
    
    model_name: str
    folds: list[FoldResult] = field(default_factory=list)
    
    # Aggregated metrics
    mean_pinball_loss: dict[str, float] = field(default_factory=dict)
    mean_sharpe: float = 0.0
    sharpe_std: float = 0.0
    mean_max_dd: float = 0.0
    
    regime_performance: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    feature_lag_candles: int = 1
    fold_boundaries_artifact: Optional[str] = None
    
    def compute_aggregates(self) -> None:
        """Compute aggregate metrics across all folds."""
        if not self.folds:
            return
        
        # Aggregate pinball loss
        pinball_sums = {}
        pinball_counts = {}
        
        sharpes = []
        drawdowns = []
        
        for fold in self.folds:
            if fold.quantile_metrics:
                for q, loss in fold.quantile_metrics.pinball_loss.items():
                    if q not in pinball_sums:
                        pinball_sums[q] = 0.0
                        pinball_counts[q] = 0
                    pinball_sums[q] += loss
                    pinball_counts[q] += 1
            
            if fold.trading_metrics:
                sharpes.append(fold.trading_metrics.sharpe_ratio)
                drawdowns.append(fold.trading_metrics.max_drawdown)
        
        # Mean pinball loss
        self.mean_pinball_loss = {
            q: pinball_sums[q] / pinball_counts[q] 
            for q in pinball_sums
        }
        
        # Sharpe statistics
        if sharpes:
            self.mean_sharpe = np.mean(sharpes)
            self.sharpe_std = np.std(sharpes)
        
        # Mean max drawdown
        if drawdowns:
            self.mean_max_dd = np.mean(drawdowns)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"\n{'='*60}",
            f"Walk-Forward Results: {self.model_name}",
            f"{'='*60}",
            f"Folds: {len(self.folds)}",
            f"",
            "Quantile Metrics (mean):",
        ]
        
        for q, loss in self.mean_pinball_loss.items():
            lines.append(f"  {q} pinball loss: {loss:.6f}")
        
        lines.extend([
            "",
            "Trading Metrics:",
            f"  Mean Sharpe: {self.mean_sharpe:.3f} (±{self.sharpe_std:.3f})",
            f"  Mean Max DD: {self.mean_max_dd:.2%}",
        ])
        
        if self.regime_performance:
            lines.append("")
            lines.append("Regime Breakdown:")
            for regime, metrics in self.regime_performance.items():
                lines.append(f"  {regime}: Sharpe={metrics.get('sharpe', 0):.3f}")
        if self.fold_boundaries_artifact:
            lines.append("")
            lines.append(f"Fold boundaries artifact: {self.fold_boundaries_artifact}")
        
        return "\n".join(lines)


class WalkForwardEvaluator:
    """
    Walk-forward evaluation harness.
    
    Implements rolling window train/test splits with configurable:
    - Training window (rolling vs expanding)
    - Test window
    - Step size (retraining frequency)
    """
    
    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None,
        target_column: str = "forward_return",
        regime_column: str = "regime",
        feature_lag_candles: int = 1,
    ):
        self.config = config or get_settings().walk_forward
        self.target_column = target_column
        self.regime_column = regime_column
        if feature_lag_candles < 1:
            raise ValueError("feature_lag_candles must be >= 1 to enforce strict causality")
        self.feature_lag_candles = feature_lag_candles

    @staticmethod
    def _validate_datetime_index(data: pd.DataFrame) -> None:
        """Validate data index contract required for leakage-safe folding."""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Walk-forward data index must be a pandas.DatetimeIndex")
        if data.index.tz is None:
            raise ValueError("Walk-forward data index must be timezone-aware")
        if not data.index.is_monotonic_increasing:
            raise ValueError("Walk-forward data index must be monotonic increasing")
        if data.index.has_duplicates:
            raise ValueError("Walk-forward data index must not contain duplicates")

    def _resolve_feature_columns(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[list[str]],
    ) -> list[str]:
        """Select leakage-safe feature columns."""
        if feature_columns is not None:
            missing = [c for c in feature_columns if c not in data.columns]
            if missing:
                raise ValueError(f"Requested feature columns missing from data: {missing}")
            return feature_columns

        disallowed_exact = {
            self.target_column,
            "forward_realized_vol",
            "forward_direction",
        }
        disallowed_prefixes = ("forward_", "hist_q")
        disallowed_contains = ("regime",)

        selected: list[str] = []
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            if col in disallowed_exact:
                continue
            if any(col.startswith(prefix) for prefix in disallowed_prefixes):
                continue
            if any(token in col for token in disallowed_contains):
                continue
            selected.append(col)

        if not selected:
            raise ValueError(
                "No leakage-safe numeric feature columns could be inferred. "
                "Pass feature_columns explicitly."
            )
        return selected

    def _prepare_supervised_frame(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build leakage-safe supervised matrix.

        Features are shifted by `feature_lag_candles` to ensure strict temporal precedence.
        """
        X = frame[feature_columns].shift(self.feature_lag_candles)
        y = frame[self.target_column]
        valid = ~(X.isna().any(axis=1) | y.isna())
        return X.loc[valid], y.loc[valid]

    @staticmethod
    def _assert_fold_no_contamination(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold_id: int,
    ) -> None:
        """Assert fold boundaries avoid train/test contamination."""
        if train_data.empty or test_data.empty:
            raise ValueError(f"Fold {fold_id}: empty train or test segment")
        if train_data.index.max() >= test_data.index.min():
            raise ValueError(
                f"Fold {fold_id}: train/test overlap detected "
                f"(train_max={train_data.index.max()}, test_min={test_data.index.min()})"
            )
        overlap = train_data.index.intersection(test_data.index)
        if len(overlap) > 0:
            raise ValueError(f"Fold {fold_id}: duplicate timestamps across train/test boundaries")

    def _write_fold_boundaries_snapshot(
        self,
        *,
        model_name: str,
        folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
        data: pd.DataFrame,
        feature_columns: list[str],
        artifact_dir: Path,
    ) -> Path:
        """Persist fold boundary snapshot for auditability."""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_model_name = model_name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        path = artifact_dir / f"fold_boundaries_{safe_model_name}_{timestamp}.json"

        folds_payload: list[dict[str, object]] = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            train_rows = int(len(data.loc[train_start:train_end]))
            test_rows = int(len(data.loc[test_start:test_end]))
            gap = test_start - train_end
            folds_payload.append(
                {
                    "fold_id": i,
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "train_rows_raw": train_rows,
                    "test_rows_raw": test_rows,
                    "gap_seconds": int(gap.total_seconds()),
                }
            )

        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "target_column": self.target_column,
            "feature_lag_candles": self.feature_lag_candles,
            "feature_columns": feature_columns,
            "n_folds": len(folds),
            "folds": folds_payload,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path
    
    def generate_folds(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generate train/test fold boundaries.
        
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        self._validate_datetime_index(data)

        train_days = self.config.effective_train_days
        test_days = self.config.effective_test_days
        step_days = self.config.effective_step_days
        
        train_window = pd.Timedelta(days=train_days)
        test_window = pd.Timedelta(days=test_days)
        step_size = pd.Timedelta(days=step_days)

        index_deltas = data.index.to_series().diff().dropna()
        if index_deltas.empty:
            raise ValueError("Cannot generate folds from index with fewer than 2 timestamps")
        bar_step = index_deltas.mode().iloc[0]
        
        data_start = data.index.min()
        data_end = data.index.max()
        
        if start_date:
            fold_start = pd.Timestamp(start_date, tz="UTC")
        else:
            # Start after first training window
            fold_start = data_start + train_window
        
        folds = []
        
        current_test_start = fold_start
        
        while current_test_start + test_window <= data_end:
            train_start = current_test_start - train_window
            train_end = current_test_start - bar_step
            test_start = current_test_start
            test_end = test_start + test_window - bar_step
            
            # Ensure we have data in this range
            if train_start >= data_start:
                if train_end >= test_start:
                    raise ValueError(
                        "Invalid fold boundary: train_end must be strictly earlier than test_start"
                    )
                folds.append((train_start, train_end, test_start, test_end))
            
            current_test_start += step_size
        
        logger.info(f"Generated {len(folds)} walk-forward folds")
        return folds
    
    def evaluate_model(
        self,
        model_class: Type[BaselineModel],
        data: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
        model_kwargs: Optional[dict] = None,
        start_date: Optional[str] = None,
        show_progress: bool = True,
        simple_strategy: bool = True,
        save_fold_boundaries: bool = False,
        artifact_dir: str | Path = "data/results",
    ) -> WalkForwardResults:
        """
        Run walk-forward evaluation on a model.
        
        Args:
            model_class: Model class to instantiate and train
            data: Full dataset with features and targets
            feature_columns: Feature columns to use
            model_kwargs: Arguments to pass to model constructor
            start_date: Start date for evaluation
            show_progress: Show progress bar
            simple_strategy: Use simple sign-based strategy for trading metrics
            save_fold_boundaries: Persist fold train/test boundary snapshot
            artifact_dir: Output directory for fold boundary artifact
            
        Returns:
            WalkForwardResults with all fold results
        """
        self._validate_datetime_index(data)
        model_kwargs = model_kwargs or {}
        folds = self.generate_folds(data, start_date)
        feature_cols = self._resolve_feature_columns(data, feature_columns)
        
        # Create a temporary model to get the name
        temp_model = model_class(**model_kwargs)
        results = WalkForwardResults(
            model_name=temp_model.name,
            feature_columns=feature_cols,
            feature_lag_candles=self.feature_lag_candles,
        )

        if save_fold_boundaries and folds:
            snapshot_path = self._write_fold_boundaries_snapshot(
                model_name=temp_model.name,
                folds=folds,
                data=data,
                feature_columns=feature_cols,
                artifact_dir=Path(artifact_dir),
            )
            results.fold_boundaries_artifact = str(snapshot_path)
        
        iterator = (
            tqdm(enumerate(folds), total=len(folds), desc=temp_model.name)
            if show_progress
            else enumerate(folds)
        )
        
        for i, (train_start, train_end, test_start, test_end) in iterator:
            # Get train and test data
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]
            try:
                self._assert_fold_no_contamination(train_data, test_data, i)
            except ValueError as exc:
                logger.error(str(exc))
                continue
            
            if len(train_data) < self.config.min_train_samples:
                logger.warning(f"Fold {i}: Insufficient training data ({len(train_data)} samples)")
                continue
            
            # Prepare leakage-safe features and targets.
            X_train, y_train = self._prepare_supervised_frame(train_data, feature_cols)
            X_test, y_test = self._prepare_supervised_frame(test_data, feature_cols)
            if len(X_train) < self.config.min_train_samples:
                logger.warning(
                    f"Fold {i}: Insufficient leakage-safe training samples after lag filter "
                    f"({len(X_train)} samples)"
                )
                continue
            if X_test.empty:
                logger.warning(f"Fold {i}: No leakage-safe test samples after lag filter")
                continue
            
            # Train model
            model = model_class(**model_kwargs)
            
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                logger.error(f"Fold {i}: Training failed: {e}")
                continue
            
            # Predict
            try:
                predictions = model.predict(X_test)
            except Exception as e:
                logger.error(f"Fold {i}: Prediction failed: {e}")
                continue
            y_test = y_test.loc[predictions.index]
            if y_test.empty:
                logger.warning(f"Fold {i}: Empty aligned test target after prediction indexing")
                continue
            
            # Create fold result
            fold_result = FoldResult(
                fold_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_samples=len(X_train),
                test_samples=len(y_test),
                predictions=predictions,
                actuals=y_test,
            )
            
            # Get regimes if available
            if self.regime_column in test_data.columns:
                fold_result.regimes = test_data.loc[y_test.index, self.regime_column]
            
            # Compute quantile metrics
            fold_result.quantile_metrics = compute_quantile_metrics(
                y_test, predictions
            )
            
            # Compute trading metrics using simple strategy
            if simple_strategy and "q50" in predictions.columns:
                positions = np.sign(predictions["q50"])
                positions = pd.Series(positions.values, index=predictions.index)
                
                fold_result.trading_metrics = compute_trading_metrics(
                    y_test,
                    positions,
                    regimes=fold_result.regimes,
                )
            
            results.folds.append(fold_result)
        
        # Compute aggregate metrics
        results.compute_aggregates()
        
        # Aggregate regime performance
        regime_sharpes = {}
        for fold in results.folds:
            if fold.trading_metrics and fold.trading_metrics.regime_sharpe:
                for regime, sharpe in fold.trading_metrics.regime_sharpe.items():
                    if regime not in regime_sharpes:
                        regime_sharpes[regime] = []
                    regime_sharpes[regime].append(sharpe)
        
        results.regime_performance = {
            regime: {"sharpe": np.mean(sharpes), "count": len(sharpes)}
            for regime, sharpes in regime_sharpes.items()
        }
        
        logger.info(f"Walk-forward evaluation complete: {len(results.folds)} folds")
        return results
    
    def compare_models(
        self,
        models: dict[str, tuple[Type[BaselineModel], dict]],
        data: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        save_fold_boundaries: bool = False,
        artifact_dir: str | Path = "data/results",
    ) -> dict[str, WalkForwardResults]:
        """
        Compare multiple models using walk-forward evaluation.
        
        Args:
            models: Dict of {name: (model_class, kwargs)}
            data: Full dataset
            feature_columns: Feature columns to use
            start_date: Start date for evaluation
            save_fold_boundaries: Persist per-model fold boundary snapshots
            artifact_dir: Output directory for fold boundary artifacts
            
        Returns:
            Dict of {name: WalkForwardResults}
        """
        results = {}
        
        for name, (model_class, kwargs) in models.items():
            logger.info(f"\nEvaluating {name}...")
            results[name] = self.evaluate_model(
                model_class,
                data,
                feature_columns=feature_columns,
                model_kwargs=kwargs,
                start_date=start_date,
                save_fold_boundaries=save_fold_boundaries,
                artifact_dir=artifact_dir,
            )
        
        # Print comparison
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                "Model": name,
                "Folds": len(result.folds),
                "Mean Sharpe": f"{result.mean_sharpe:.3f}",
                "Sharpe Std": f"{result.sharpe_std:.3f}",
                "Mean Max DD": f"{result.mean_max_dd:.2%}",
                "q50 Pinball": f"{result.mean_pinball_loss.get('q50', 0):.6f}",
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        return results


def main():
    """Example usage with synthetic data."""
    logging.basicConfig(level=logging.INFO)
    
    from src.models.baselines.random_walk import RandomWalkBaseline
    from src.models.baselines.ewma import EWMABaseline
    
    # Create synthetic dataset
    np.random.seed(42)
    n = 2000  # ~1.3 years of 4h data
    dates = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    
    returns = np.random.normal(0, 0.02, n)
    returns[500:700] += 0.003  # Bull period
    
    data = pd.DataFrame({
        "close": 40000 * np.exp(np.cumsum(returns)),
        "return_1": returns,
        "forward_return": np.roll(returns, -1),  # Shifted target
        "regime": np.where(
            np.abs(np.convolve(returns, np.ones(42)/42, mode='same')) > 0.001,
            "trend", "normal"
        ),
    }, index=dates)
    data.loc[data.index[-1], "forward_return"] = np.nan  # Last is unknown
    
    # Evaluate
    evaluator = WalkForwardEvaluator()
    
    models = {
        "RandomWalk": (RandomWalkBaseline, {"lookback_window": 100}),
        "EWMA": (EWMABaseline, {"span": 24}),
    }
    
    results = evaluator.compare_models(
        models,
        data,
        start_date="2022-07-01",
    )
    
    # Print detailed results for one model
    print(results["RandomWalk"].summary())


if __name__ == "__main__":
    main()
