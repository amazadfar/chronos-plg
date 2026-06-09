"""
Backtest engine for walk-forward strategy evaluation.

Implements full backtest with:
- Walk-forward model retraining
- Realistic cost modeling
- Performance attribution by regime
- Comprehensive metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Type

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import WalkForwardConfig, get_settings
from src.backtest.costs import CostModel
from src.common.metrics import profit_factor_from_returns
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.models.baselines.random_walk import BaselineModel
from src.strategy.position_sizing import PositionSizer
from src.strategy.regime_detector import RegimeDetector
from src.strategy.signals import QuantileSignalGenerator

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    
    # Performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Costs
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0
    total_interest: float = 0.0
    total_other_costs: float = 0.0
    total_costs: float = 0.0
    
    # Trading stats
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor_net: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    
    # Regime breakdown
    regime_returns: dict[str, float] = field(default_factory=dict)
    regime_sharpes: dict[str, float] = field(default_factory=dict)
    
    # Time series
    positions: Optional[pd.DataFrame] = None
    returns: Optional[pd.DataFrame] = None
    equity_curve: Optional[pd.Series] = None
    fold_metrics: list[dict[str, object]] = field(default_factory=list)
    trades: Optional[pd.DataFrame] = None
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            "",
            "Performance:",
            f"  Total Return: {self.total_return:.2%}",
            f"  Annualized Return: {self.annualized_return:.2%}",
            f"  Sharpe Ratio: {self.sharpe_ratio:.3f}",
            f"  Sortino Ratio: {self.sortino_ratio:.3f}",
            f"  Max Drawdown: {self.max_drawdown:.2%}",
            "",
            "Costs:",
            f"  Total Fees: {self.total_fees:.2%}",
            f"  Total Slippage: {self.total_slippage:.2%}",
            f"  Total Funding: {self.total_funding:.2%}",
            f"  Total Interest: {self.total_interest:.2%}",
            f"  Total Other Costs: {self.total_other_costs:.2%}",
            f"  Total Costs: {self.total_costs:.2%}",
            "",
            "Trading:",
            f"  Number of Trades: {self.num_trades}",
            f"  Win Rate: {self.win_rate:.1%}",
            f"  Profit Factor (Net): {self.profit_factor_net:.2f}",
        ]
        
        if self.regime_sharpes:
            lines.extend(["", "Regime Breakdown:"])
            for regime, sharpe in self.regime_sharpes.items():
                ret = self.regime_returns.get(regime, 0)
                lines.append(f"  {regime}: Sharpe={sharpe:.3f}, Return={ret:.2%}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "total_funding": self.total_funding,
            "total_interest": self.total_interest,
            "total_other_costs": self.total_other_costs,
            "total_costs": self.total_costs,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "profit_factor_net": self.profit_factor_net,
            "profit_factor": self.profit_factor,
            "regime_returns": self.regime_returns,
            "regime_sharpes": self.regime_sharpes,
            "n_folds": len(self.fold_metrics),
            "num_trade_events": int(len(self.trades)) if self.trades is not None else 0,
        }


class BacktestEngine:
    """
    Walk-forward backtest engine.
    
    Runs a complete backtest with:
    1. Walk-forward model training
    2. Out-of-sample prediction
    3. Signal generation
    4. Position sizing
    5. Cost-adjusted returns
    """
    
    def __init__(
        self,
        model_class: Type[BaselineModel],
        model_kwargs: Optional[dict] = None,
        walk_forward_config: Optional[WalkForwardConfig] = None,
        cost_model: Optional[CostModel] = None,
        signal_generator: Optional[QuantileSignalGenerator] = None,
        position_sizer: Optional[PositionSizer] = None,
        regime_detector: Optional[RegimeDetector] = None,
        target_column: str = "forward_return",
    ):
        """
        Args:
            model_class: Forecasting model class
            model_kwargs: Model constructor arguments
            walk_forward_config: Walk-forward configuration
            cost_model: Cost model
            signal_generator: Signal generator
            position_sizer: Position sizer
            regime_detector: Regime detector
            target_column: Target column name
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.wf_config = walk_forward_config or get_settings().walk_forward
        self.cost_model = cost_model or CostModel()
        self.signal_generator = signal_generator or QuantileSignalGenerator()
        self.position_sizer = position_sizer or PositionSizer()
        self.regime_detector = regime_detector or RegimeDetector()
        self.target_column = target_column
        
        self._evaluator = WalkForwardEvaluator(config=self.wf_config)
    
    def run(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        show_progress: bool = True,
        precomputed_folds: Optional[
            list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]
        ] = None,
        collect_fold_metrics: bool = True,
    ) -> BacktestResult:
        """
        Run complete backtest.
        
        Args:
            data: Full dataset with features and targets
            feature_columns: Feature columns to use
            start_date: Start date for backtest
            
        Returns:
            BacktestResult with all metrics
        """
        logger.info("Starting backtest...")
        
        # Generate walk-forward folds
        folds = precomputed_folds or self._evaluator.generate_folds(data, start_date)
        logger.info(f"Generated {len(folds)} walk-forward folds")
        
        # Detect regimes for full dataset
        if "close" in data.columns:
            regimes = self.regime_detector.detect_regimes(data)
        else:
            regimes = pd.DataFrame({"regime": "normal"}, index=data.index)
        
        # Collect out-of-sample predictions and positions
        all_predictions = []
        all_positions = []
        fold_metrics: list[dict[str, object]] = []
        
        iterator = tqdm(folds, desc="Backtesting") if show_progress else folds
        
        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(iterator):
            # Get train and test data
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]
            
            if len(train_data) < self.wf_config.min_train_samples:
                continue
            
            # Train model
            if feature_columns:
                X_train = train_data[feature_columns]
                X_test = test_data[feature_columns]
            else:
                X_train = train_data
                X_test = test_data
            
            y_train = train_data[self.target_column]
            
            model = self.model_class(**self.model_kwargs)
            
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Training failed: {e}")
                continue
            
            # Predict out-of-sample
            try:
                predictions = model.predict(X_test).copy()
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                continue

            predictions["regime"] = regimes.loc[test_data.index, "regime"].values

            # Build scenario-aware expected-cost inputs for net-edge entry policies.
            if "realized_vol_6" in test_data.columns:
                pred_vol = pd.to_numeric(
                    test_data["realized_vol_6"],
                    errors="coerce",
                ).reindex(test_data.index).fillna(0.02)
            else:
                pred_vol = pd.Series(0.02, index=test_data.index, dtype=float)

            funding_for_gate = (
                pd.to_numeric(data.loc[test_data.index, "funding_rate"], errors="coerce")
                if "funding_rate" in data.columns
                else None
            )
            borrow_for_gate = (
                pd.to_numeric(data.loc[test_data.index, "borrow_rate_per_day"], errors="coerce")
                if "borrow_rate_per_day" in data.columns
                else None
            )
            other_for_gate = (
                pd.to_numeric(data.loc[test_data.index, "other_costs"], errors="coerce")
                if "other_costs" in data.columns
                else None
            )
            expected_cost = self.cost_model.estimate_entry_cost_series(
                index=test_data.index,
                volatilities=pred_vol,
                funding_rates=funding_for_gate,
                borrow_rates_per_day=borrow_for_gate,
                expected_holding_bars=max(
                    1,
                    int(getattr(self.signal_generator, "expected_cost_holding_bars", 1)),
                ),
                include_exit=bool(getattr(self.signal_generator, "expected_cost_round_trip", True)),
                target_notional=1.0,
                other_costs=other_for_gate,
            )
            predictions[getattr(self.signal_generator, "expected_cost_column", "expected_cost")] = expected_cost
            predictions[getattr(self.signal_generator, "predicted_risk_column", "predicted_risk")] = pred_vol

            all_predictions.append(predictions)
            
            # Generate signals
            signals = self.signal_generator.generate_signals(predictions)
            
            # Get regime multipliers
            regime_mult = self.regime_detector.get_regime_multipliers(
                regimes.loc[test_data.index, "regime"]
            )
            
            # Calculate positions
            positions = self.position_sizer.calculate_sizes(
                signals, pred_vol, regime_mult
            )
            
            # Store with metadata
            position_df = pd.DataFrame({
                "position": positions,
                "signal": signals["signal"],
                "regime": regimes.loc[test_data.index, "regime"],
                "q10": predictions["q10"] if "q10" in predictions.columns else np.nan,
                "q50": predictions["q50"],
                "q90": predictions["q90"] if "q90" in predictions.columns else np.nan,
                "fold_id": fold_id,
            })
            
            all_positions.append(position_df)

            if collect_fold_metrics:
                fold_result = self._calculate_results(
                    positions=position_df,
                    actual_returns=data.loc[position_df.index, self.target_column],
                    regimes=regimes.loc[position_df.index, "regime"],
                    volatilities=data.loc[position_df.index].get("realized_vol_6", None),
                    funding_rates=data.loc[position_df.index].get("funding_rate", None),
                    borrow_rates_per_day=data.loc[position_df.index].get("borrow_rate_per_day", None),
                    other_costs=data.loc[position_df.index].get("other_costs", None),
                )
                fold_pf_net = (
                    fold_result.profit_factor_net
                    if fold_result.profit_factor_net > 0
                    else fold_result.profit_factor
                )
                fold_metrics.append(
                    {
                        "fold_id": fold_id,
                        "train_start": train_start.isoformat(),
                        "train_end": train_end.isoformat(),
                        "test_start": test_start.isoformat(),
                        "test_end": test_end.isoformat(),
                        "n_bars": int(len(position_df)),
                        "total_return": float(fold_result.total_return),
                        "sharpe_ratio": float(fold_result.sharpe_ratio),
                        "max_drawdown": float(fold_result.max_drawdown),
                        "win_rate": float(fold_result.win_rate),
                        "profit_factor_net": float(fold_pf_net),
                        "total_costs": float(fold_result.total_costs),
                        "num_trades": int(fold_result.num_trades),
                    }
                )
        
        if not all_positions:
            logger.error("No valid folds completed")
            return BacktestResult()
        
        # Combine all out-of-sample results
        positions_combined = pd.concat(all_positions)
        predictions_combined = pd.concat(all_predictions)
        
        # Remove duplicates (overlapping folds)
        positions_combined = positions_combined[~positions_combined.index.duplicated(keep='first')]
        predictions_combined = predictions_combined[~predictions_combined.index.duplicated(keep='first')]
        
        # Calculate returns
        result = self._calculate_results(
            positions_combined,
            data.loc[positions_combined.index, self.target_column],
            regimes.loc[positions_combined.index, "regime"],
            data.loc[positions_combined.index].get("realized_vol_6", None),
            data.loc[positions_combined.index].get("funding_rate", None),
            data.loc[positions_combined.index].get("borrow_rate_per_day", None),
            data.loc[positions_combined.index].get("other_costs", None),
        )
        
        result.positions = positions_combined
        result.fold_metrics = fold_metrics
        
        logger.info("Backtest complete!")
        logger.info(result.summary())
        
        return result
    
    def _calculate_results(
        self,
        positions: pd.DataFrame,
        actual_returns: pd.Series,
        regimes: pd.Series,
        volatilities: Optional[pd.Series] = None,
        funding_rates: Optional[pd.Series] = None,
        borrow_rates_per_day: Optional[pd.Series] = None,
        other_costs: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Calculate backtest metrics from positions."""
        result = BacktestResult()

        # Align all primary series to position index.
        idx = positions.index
        actual_returns = actual_returns.reindex(idx)
        regimes = regimes.reindex(idx)

        if volatilities is None:
            volatilities = pd.Series(0.02, index=idx)
        else:
            volatilities = volatilities.reindex(idx).fillna(0.02)

        funding_series = funding_rates.reindex(idx).fillna(0.0) if funding_rates is not None else None
        borrow_series = (
            borrow_rates_per_day.reindex(idx).fillna(np.nan)
            if borrow_rates_per_day is not None
            else None
        )
        other_series = other_costs.reindex(idx).fillna(0.0) if other_costs is not None else None

        costs = self.cost_model.calculate_execution_costs(
            positions=positions["position"],
            volatilities=volatilities,
            funding_rates=funding_series,
            borrow_rates_per_day=borrow_series,
            other_costs=other_series,
        )
        
        # Lagged positions (trade at close, return next period)
        pos_lagged = positions["position"].shift(1)
        
        # Returns
        gross_returns = pos_lagged * actual_returns
        net_returns = gross_returns - costs["total_costs"]
        
        # Store time series
        result.returns = pd.DataFrame({
            "gross_return": gross_returns,
            "event_type": costs["event_type"],
            "prev_position": costs["prev_position"],
            "position": costs["position"],
            "open_notional": costs["open_notional"],
            "close_notional": costs["close_notional"],
            "traded_notional": costs["traded_notional"],
            "fees": costs["fees"],
            "funding": costs["funding"],
            "interest": costs["interest"],
            "slippage": costs["slippage"],
            "other_costs": costs["other_costs"],
            "total_costs": costs["total_costs"],
            "net_return": net_returns,
            "costs": costs["total_costs"],  # Backward-compatible alias.
        })
        if "fold_id" in positions.columns:
            result.returns["fold_id"] = positions["fold_id"]
        result.returns["regime"] = regimes

        trade_mask = costs["traded_notional"] > 0
        result.trades = result.returns.loc[trade_mask].copy()
        
        # Equity curve
        equity = (1 + net_returns.fillna(0)).cumprod()
        result.equity_curve = equity
        
        # Performance metrics
        valid_returns = net_returns.dropna()
        periods_per_year = 6 * 365
        
        result.total_return = equity.iloc[-1] - 1
        years = len(valid_returns) / periods_per_year
        result.annualized_return = (1 + result.total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe
        mean_ret = valid_returns.mean()
        std_ret = valid_returns.std()
        if std_ret > 0:
            result.sharpe_ratio = mean_ret / std_ret * np.sqrt(periods_per_year)
        
        # Sortino
        downside = valid_returns[valid_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            result.sortino_ratio = mean_ret / downside.std() * np.sqrt(periods_per_year)
        
        # Max drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        result.max_drawdown = drawdown.min()
        
        # Costs
        result.total_fees = costs["fees"].sum()
        result.total_slippage = costs["slippage"].sum()
        result.total_funding = costs["funding"].sum()
        result.total_interest = costs["interest"].sum()
        result.total_other_costs = costs["other_costs"].sum()
        result.total_costs = costs["total_costs"].sum()
        
        # Trading stats
        result.num_trades = int((costs["traded_notional"] > 0.01).sum())
        wins = valid_returns[valid_returns > 0]
        result.win_rate = len(wins) / len(valid_returns) if len(valid_returns) > 0 else 0
        result.profit_factor_net = profit_factor_from_returns(valid_returns)
        result.profit_factor = result.profit_factor_net
        
        # Regime breakdown
        for regime in regimes.unique():
            mask = regimes == regime
            regime_returns = net_returns[mask]
            if len(regime_returns) > 10:
                result.regime_returns[regime] = regime_returns.sum()
                if regime_returns.std() > 0:
                    result.regime_sharpes[regime] = (
                        regime_returns.mean() / regime_returns.std() * np.sqrt(periods_per_year)
                    )
        
        return result


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    from src.models.baselines import RandomWalkBaseline
    
    # Create sample data
    np.random.seed(42)
    n = 2000
    dates = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    
    returns = np.random.normal(0, 0.02, n)
    returns[500:700] += 0.003  # Bull regime
    
    data = pd.DataFrame({
        "close": 40000 * np.exp(np.cumsum(returns)),
        "return_1": returns,
        "return_6": pd.Series(returns).rolling(6).sum().values,
        "realized_vol_6": pd.Series(returns).rolling(6).std().values,
        "forward_return": np.roll(returns, -1),
    }, index=dates)
    data.loc[data.index[-1], "forward_return"] = np.nan
    
    feature_cols = ["return_1", "return_6", "realized_vol_6"]
    
    # Run backtest
    engine = BacktestEngine(
        model_class=RandomWalkBaseline,
        model_kwargs={"lookback_window": 100},
    )
    
    result = engine.run(
        data,
        feature_columns=feature_cols,
        start_date="2022-07-01",
    )
    
    print(result.summary())


if __name__ == "__main__":
    main()
