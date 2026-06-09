"""
Label generator for the Chronos-2 trading system.

Computes:
- Forward returns (target)
- Return quantiles (from historical distribution)
- Realized volatility
- Regime labels
"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

from config.settings import get_settings, TargetConfig

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generate target labels for training and evaluation."""
    
    def __init__(self, config: Optional[TargetConfig] = None):
        self.config = config or get_settings().target
    
    def compute_forward_returns(
        self,
        close: pd.Series,
        horizon: Optional[int] = None,
    ) -> pd.Series:
        """
        Compute forward log returns.
        
        r_t = log(close_{t+H} / close_t)
        
        Args:
            close: Series of close prices
            horizon: Lookahead in candles (default: from config)
            
        Returns:
            Series of forward returns (NaN for last H rows)
        """
        h = horizon or self.config.horizon_candles
        
        forward_close = close.shift(-h)
        returns = np.log(forward_close / close)
        
        logger.debug(f"Computed forward returns with horizon={h}")
        return returns.rename("forward_return")
    
    def compute_realized_volatility(
        self,
        close: pd.Series,
        window: Optional[int] = None,
    ) -> pd.Series:
        """
        Compute forward realized volatility.
        
        rv_t = std(returns over next W candles)
        
        This is the volatility we're trying to predict, not historical vol.
        Uses future data, so only valid for training labels.
        
        Args:
            close: Series of close prices
            window: Window size in candles (default: from config)
            
        Returns:
            Series of forward realized volatility
        """
        w = window or self.config.rv_window_candles
        
        # Compute returns first
        returns = np.log(close / close.shift(1))
        
        # Forward rolling std (uses future data)
        # We need to reverse, compute rolling, then reverse back
        rv = (
            returns
            .iloc[::-1]
            .rolling(window=w, min_periods=w)
            .std()
            .iloc[::-1]
        )
        
        # Shift to align with prediction time
        rv = rv.shift(-1)
        
        logger.debug(f"Computed forward realized volatility with window={w}")
        return rv.rename("forward_realized_vol")
    
    def compute_historical_quantiles(
        self,
        returns: pd.Series,
        lookback: int = 42 * 6,  # 7 days * 6 candles = 42 candles
        quantiles: Optional[tuple[float, ...]] = None,
    ) -> pd.DataFrame:
        """
        Compute rolling quantiles of historical returns.
        
        These serve as a "naive" baseline - predict future quantiles
        based on recent return distribution.
        
        Args:
            returns: Series of historical returns
            lookback: Lookback window in candles
            quantiles: Quantiles to compute (default: from config)
            
        Returns:
            DataFrame with q10, q50, q90 columns (historical baseline)
        """
        qs = quantiles or self.config.quantiles
        
        result = pd.DataFrame(index=returns.index)
        
        for q in qs:
            col_name = f"hist_q{int(q*100)}"
            result[col_name] = returns.rolling(window=lookback, min_periods=lookback//2).quantile(q)
        
        logger.debug(f"Computed historical quantiles: {list(result.columns)}")
        return result
    
    def compute_regime_labels(
        self,
        close: pd.Series,
        window_days: int = 7,
        candles_per_day: int = 6,  # 4h candles
    ) -> pd.DataFrame:
        """
        Compute regime labels based on trend and volatility.
        
        Regimes:
        - trend: |7d_return| > 10% AND vol < 5%
        - chop: |7d_return| < 3% AND vol < 4%
        - panic: vol > 8%
        - normal: everything else
        
        Args:
            close: Series of close prices
            window_days: Window for regime calculation
            candles_per_day: Number of candles per day
            
        Returns:
            DataFrame with regime columns
        """
        settings = get_settings().strategy
        window = window_days * candles_per_day
        
        # 7-day return
        returns_7d = np.log(close / close.shift(window))
        
        # 7-day volatility (of 4h returns)
        returns_4h = np.log(close / close.shift(1))
        vol_7d = returns_4h.rolling(window=window).std()
        
        # Regime classification
        is_trend = (returns_7d.abs() > settings.trend_return_threshold) & (vol_7d < settings.trend_vol_threshold)
        is_chop = (returns_7d.abs() < settings.chop_return_threshold) & (vol_7d < settings.chop_vol_threshold)
        is_panic = vol_7d > settings.panic_vol_threshold
        
        # Assign labels (priority: panic > trend > chop > normal)
        regime = pd.Series("normal", index=close.index)
        regime[is_chop] = "chop"
        regime[is_trend] = "trend"
        regime[is_panic] = "panic"
        
        result = pd.DataFrame(index=close.index)
        result["regime"] = regime
        result["regime_trend"] = (regime == "trend").astype(int)
        result["regime_chop"] = (regime == "chop").astype(int)
        result["regime_panic"] = (regime == "panic").astype(int)
        result["returns_7d"] = returns_7d
        result["vol_7d"] = vol_7d
        
        # Log regime distribution
        counts = regime.value_counts()
        logger.info(f"Regime distribution: {counts.to_dict()}")
        
        return result
    
    def generate_all_labels(
        self,
        ohlcv: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate all labels from OHLCV data.
        
        Args:
            ohlcv: DataFrame with at least 'close' column
            
        Returns:
            DataFrame with all label columns:
            - forward_return: target return
            - forward_realized_vol: target volatility
            - hist_q10, hist_q50, hist_q90: baseline quantiles
            - regime, regime_trend, regime_chop, regime_panic
            - returns_7d, vol_7d: underlying regime metrics
        """
        close = ohlcv["close"]
        
        # Forward returns (target)
        forward_return = self.compute_forward_returns(close)
        
        # Forward realized vol
        forward_rv = self.compute_realized_volatility(close)
        
        # Historical returns for quantile baseline
        hist_returns = np.log(close / close.shift(1))
        hist_quantiles = self.compute_historical_quantiles(hist_returns)
        
        # Regimes
        regimes = self.compute_regime_labels(close)
        
        # Combine
        labels = pd.concat([
            forward_return,
            forward_rv,
            hist_quantiles,
            regimes,
        ], axis=1)
        
        # Add direction label (for evaluation)
        labels["forward_direction"] = (labels["forward_return"] > 0).astype(int)
        
        # Count valid samples
        valid = labels["forward_return"].notna().sum()
        logger.info(f"Generated labels: {len(labels)} rows, {valid} valid samples")
        
        return labels
    
    def validate_no_leakage(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        target_column: str = "forward_return",
        feature_lag_candles: int = 1,
        max_shift_probe: int = 3,
    ) -> dict:
        """
        Validate that features are causally usable for forward labels.

        This validator is structural/deterministic (timestamp and boundary based),
        not correlation-driven:
        1. Datetime index contract checks
        2. Target horizon boundary checks
        3. Prior-feature availability checks for strict feature lag
        4. Explicit shifted-target leakage trap detection
        """
        results = {"passed": True, "warnings": [], "errors": []}

        self._validate_datetime_index(features, "features", results)
        self._validate_datetime_index(labels, "labels", results)

        if target_column not in labels.columns:
            results["errors"].append(f"Labels missing target column: {target_column}")
            results["passed"] = False
            return results

        if target_column in features.columns:
            results["errors"].append(f"Target column appears in features: {target_column}")

        forward_like = [col for col in features.columns if col.startswith("forward_")]
        if forward_like:
            results["errors"].append(
                f"Forward-looking columns found in features: {sorted(forward_like)}"
            )

        if feature_lag_candles < 1:
            results["errors"].append(
                f"feature_lag_candles must be >= 1 for strict causality, got {feature_lag_candles}"
            )

        # Short-circuit if index contract is already broken.
        if results["errors"]:
            results["passed"] = False
            logger.error(f"Leakage validation failed: {results['errors']}")
            return results

        target = labels[target_column]
        target_valid = target.dropna()
        if target_valid.empty:
            results["errors"].append(f"No non-null values found in labels[{target_column}]")
            results["passed"] = False
            logger.error(f"Leakage validation failed: {results['errors']}")
            return results

        if not target_valid.index.isin(features.index).all():
            missing_count = int((~target_valid.index.isin(features.index)).sum())
            results["errors"].append(
                f"{missing_count} target timestamps missing in features index"
            )

        step = self._infer_candle_step(labels.index)
        if step is None:
            results["errors"].append(
                "Could not infer candle step from labels index for causal checks"
            )
        else:
            horizon = self.config.horizon_candles
            future_offset = step * horizon
            feature_offset = step * feature_lag_candles

            # Causal boundary: every valid target t must have t+h inside index.
            future_missing = (target_valid.index + future_offset).difference(labels.index)
            if len(future_missing) > 0:
                preview = [ts.isoformat() for ts in list(future_missing[:3])]
                results["errors"].append(
                    "Forward target horizon boundary missing future timestamps "
                    f"(count={len(future_missing)}, sample={preview})"
                )

            # Strict prior availability: for each target at t, a feature timestamp at t-lag must exist.
            required_feature_times = target_valid.index - feature_offset
            missing_prior = required_feature_times.difference(features.index)
            if len(missing_prior) > 0:
                results["warnings"].append(
                    "Some targets do not have prior feature timestamp available for strict lag "
                    f"(lag={feature_lag_candles}, count={len(missing_prior)}). "
                    "These rows must be dropped in supervised fold construction."
                )

        leak_hits = self._find_shifted_target_leaks(
            features=features,
            target=target,
            max_shift_probe=max_shift_probe,
        )
        if leak_hits:
            for hit in leak_hits:
                results["errors"].append(hit)

        results["passed"] = len(results["errors"]) == 0
        if results["passed"]:
            logger.info("Leakage validation passed")
        else:
            logger.error(f"Leakage validation failed: {results['errors']}")
        return results

    @staticmethod
    def _validate_datetime_index(
        df: pd.DataFrame,
        dataset_name: str,
        results: dict,
    ) -> None:
        """Append index contract failures to results."""
        if not isinstance(df.index, pd.DatetimeIndex):
            results["errors"].append(f"{dataset_name} index must be DatetimeIndex")
            return
        if df.index.tz is None:
            results["errors"].append(f"{dataset_name} index must be timezone-aware")
        if not df.index.is_monotonic_increasing:
            results["errors"].append(f"{dataset_name} index must be monotonic increasing")
        if df.index.has_duplicates:
            results["errors"].append(f"{dataset_name} index must not contain duplicates")

    @staticmethod
    def _infer_candle_step(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
        """Infer dominant candle spacing from index deltas."""
        if len(index) < 2:
            return None
        deltas = index.to_series().diff().dropna()
        if deltas.empty:
            return None
        mode = deltas.mode()
        if mode.empty:
            return None
        step = mode.iloc[0]
        if step <= pd.Timedelta(0):
            return None
        return step

    @staticmethod
    def _find_shifted_target_leaks(
        *,
        features: pd.DataFrame,
        target: pd.Series,
        max_shift_probe: int,
    ) -> list[str]:
        """
        Detect explicit leakage where a feature reproduces target or shifted target.

        This catches intentional trap injections deterministically.
        """
        hits: list[str] = []
        min_overlap = 20

        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col]):
                continue

            feature_col = features[col]
            # Positive shifts correspond to past targets, which can be causally usable.
            for shift in range(-max_shift_probe, 1):
                shifted_target = target.shift(shift)
                valid = feature_col.notna() & shifted_target.notna()
                if int(valid.sum()) < min_overlap:
                    continue

                lhs = feature_col.loc[valid].to_numpy()
                rhs = shifted_target.loc[valid].to_numpy()
                if np.allclose(lhs, rhs, rtol=1e-9, atol=1e-12):
                    if shift == 0:
                        rel = "exactly equals target"
                    else:
                        rel = f"matches future target (shift={shift})"
                    hits.append(f"Potential leakage: feature '{col}' {rel}")
                    break

        return hits


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    # Simulate price with trend and volatility regimes
    returns = np.random.normal(0, 0.02, n)
    returns[200:300] += 0.005  # Bull trend
    returns[500:600] -= 0.008  # Bear trend
    returns[700:750] *= 3  # High vol period
    
    price = 40000 * np.exp(np.cumsum(returns))
    
    ohlcv = pd.DataFrame({
        "open": price * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high": price * (1 + np.random.uniform(0, 0.015, n)),
        "low": price * (1 - np.random.uniform(0, 0.015, n)),
        "close": price,
        "volume": np.random.uniform(1000, 5000, n),
    }, index=dates)
    
    # Generate labels
    generator = LabelGenerator()
    labels = generator.generate_all_labels(ohlcv)
    
    print("\nLabels sample:")
    print(labels[["forward_return", "forward_realized_vol", "regime"]].dropna().head(10))
    
    print("\nRegime distribution:")
    print(labels["regime"].value_counts())


if __name__ == "__main__":
    main()
