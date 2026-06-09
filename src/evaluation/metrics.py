"""
Evaluation metrics for forecast and trading performance.

Forecast metrics:
- Pinball loss (quantile loss)
- Calibration check
- Coverage

Trading metrics:
- Sharpe/Sortino ratio
- Max drawdown
- Profit factor
"""
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
import logging

from src.common.metrics import profit_factor_from_returns

logger = logging.getLogger(__name__)


@dataclass
class QuantileMetrics:
    """Metrics for quantile forecast evaluation."""
    
    pinball_loss: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    mean_interval_width: float = 0.0
    calibration_error: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "pinball_loss": self.pinball_loss,
            "coverage": self.coverage,
            "mean_interval_width": self.mean_interval_width,
            "calibration_error": self.calibration_error,
        }
    
    def __str__(self) -> str:
        lines = ["Quantile Metrics:"]
        for q, loss in self.pinball_loss.items():
            cov = self.coverage.get(q, 0)
            cal_err = self.calibration_error.get(q, 0)
            lines.append(f"  {q}: pinball={loss:.6f}, coverage={cov:.3f}, cal_err={cal_err:.3f}")
        lines.append(f"  interval_width: {self.mean_interval_width:.6f}")
        return "\n".join(lines)


@dataclass
class TradingMetrics:
    """Metrics for trading strategy evaluation."""
    
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor_net: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    avg_trade_return: float = 0.0
    turnover: float = 0.0
    
    # Regime breakdown
    regime_sharpe: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor_net": self.profit_factor_net,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "num_trades": self.num_trades,
            "avg_trade_return": self.avg_trade_return,
            "turnover": self.turnover,
            "regime_sharpe": self.regime_sharpe,
        }
    
    def __str__(self) -> str:
        return (
            f"Trading Metrics:\n"
            f"  Total Return: {self.total_return:.2%}\n"
            f"  Annualized Return: {self.annualized_return:.2%}\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.3f}\n"
            f"  Sortino Ratio: {self.sortino_ratio:.3f}\n"
            f"  Max Drawdown: {self.max_drawdown:.2%}\n"
            f"  Profit Factor (Net): {self.profit_factor_net:.2f}\n"
            f"  Win Rate: {self.win_rate:.2%}\n"
            f"  Num Trades: {self.num_trades}\n"
        )


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Compute pinball (quantile) loss.
    
    Lower is better. This is the proper scoring rule for quantiles.
    
    Args:
        y_true: Actual values
        y_pred: Predicted quantile values
        quantile: Quantile level (0-1)
    
    Returns:
        Mean pinball loss
    """
    errors = y_true - y_pred
    loss = np.where(
        errors >= 0,
        quantile * errors,
        (quantile - 1) * errors
    )
    return np.mean(loss)


def compute_quantile_metrics(
    y_true: pd.Series,
    predictions: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> QuantileMetrics:
    """
    Compute all quantile forecast metrics.
    
    Args:
        y_true: Actual returns
        predictions: DataFrame with q10, q50, q90 columns
        quantiles: Quantile levels
        
    Returns:
        QuantileMetrics dataclass
    """
    metrics = QuantileMetrics()
    
    # Align indices
    common_idx = y_true.dropna().index.intersection(predictions.dropna().index)
    y = y_true.loc[common_idx].values
    
    for q in quantiles:
        col = f"q{int(q*100)}"
        if col not in predictions.columns:
            continue
            
        pred = predictions.loc[common_idx, col].values
        
        # Pinball loss
        metrics.pinball_loss[col] = pinball_loss(y, pred, q)
        
        # Coverage: fraction of actuals below predicted quantile
        coverage = (y < pred).mean()
        metrics.coverage[col] = coverage
        
        # Calibration error: |coverage - expected|
        metrics.calibration_error[col] = abs(coverage - q)
    
    # Mean prediction interval width (q90 - q10)
    if "q90" in predictions.columns and "q10" in predictions.columns:
        width = predictions.loc[common_idx, "q90"] - predictions.loc[common_idx, "q10"]
        metrics.mean_interval_width = width.mean()
    
    return metrics


def compute_trading_metrics(
    returns: pd.Series,
    positions: pd.Series,
    regimes: Optional[pd.Series] = None,
    periods_per_year: int = 6 * 365,  # 4h candles
    risk_free_rate: float = 0.0,
) -> TradingMetrics:
    """
    Compute trading strategy metrics.
    
    Args:
        returns: Realized returns per period
        positions: Position sizes (-1 to 1 for direction, can be fractional)
        regimes: Optional regime labels for breakdown
        periods_per_year: Number of periods per year for annualization
        risk_free_rate: Annual risk-free rate
        
    Returns:
        TradingMetrics dataclass
    """
    metrics = TradingMetrics()
    
    # Strategy returns = position * return
    common_idx = returns.dropna().index.intersection(positions.dropna().index)
    realized = returns.loc[common_idx]
    pos = positions.loc[common_idx]
    
    strategy_returns = pos.shift(1) * realized  # Shift to avoid look-ahead
    strategy_returns = strategy_returns.dropna()
    
    if len(strategy_returns) == 0:
        logger.warning("No valid strategy returns to compute")
        return metrics
    
    # Total and annualized return
    cumulative = (1 + strategy_returns).cumprod()
    metrics.total_return = cumulative.iloc[-1] - 1
    
    years = len(strategy_returns) / periods_per_year
    if years > 0:
        metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1
    
    # Sharpe ratio
    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()
    rf_per_period = risk_free_rate / periods_per_year
    
    if std_ret > 0:
        metrics.sharpe_ratio = (mean_ret - rf_per_period) / std_ret * np.sqrt(periods_per_year)
    
    # Sortino ratio (downside deviation)
    downside = strategy_returns[strategy_returns < 0]
    if len(downside) > 0:
        downside_std = downside.std()
        if downside_std > 0:
            metrics.sortino_ratio = (mean_ret - rf_per_period) / downside_std * np.sqrt(periods_per_year)
    
    # Max drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    metrics.max_drawdown = drawdown.min()
    
    # Win rate and profit factor
    wins = strategy_returns[strategy_returns > 0]
    
    metrics.num_trades = (pos.diff().abs() > 0.01).sum()  # Count position changes
    metrics.win_rate = len(wins) / len(strategy_returns) if len(strategy_returns) > 0 else 0
    
    metrics.profit_factor_net = profit_factor_from_returns(strategy_returns)
    metrics.profit_factor = metrics.profit_factor_net
    
    # Average trade return
    if metrics.num_trades > 0:
        metrics.avg_trade_return = metrics.total_return / metrics.num_trades
    
    # Turnover (average position change per period)
    position_changes = pos.diff().abs()
    metrics.turnover = position_changes.mean() * periods_per_year
    
    # Regime breakdown
    if regimes is not None:
        regimes_aligned = regimes.reindex(strategy_returns.index)
        for regime in regimes_aligned.dropna().unique():
            mask = regimes_aligned == regime
            regime_returns = strategy_returns[mask]
            if len(regime_returns) > 10:
                mean_r = regime_returns.mean()
                std_r = regime_returns.std()
                if std_r > 0:
                    metrics.regime_sharpe[regime] = mean_r / std_r * np.sqrt(periods_per_year)
    
    return metrics


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    # Sample predictions and actuals
    np.random.seed(42)
    n = 500
    
    actuals = pd.Series(np.random.normal(0, 0.02, n), name="returns")
    
    # Simulated predictions
    predictions = pd.DataFrame({
        "q10": actuals + np.random.normal(-0.05, 0.01, n),
        "q50": actuals + np.random.normal(0, 0.005, n),
        "q90": actuals + np.random.normal(0.05, 0.01, n),
    })
    
    # Compute quantile metrics
    qm = compute_quantile_metrics(actuals, predictions)
    print(qm)
    
    # Simulate positions based on q50
    positions = pd.Series(np.sign(predictions["q50"].values), index=actuals.index)
    
    # Compute trading metrics
    tm = compute_trading_metrics(actuals, positions)
    print("\n" + str(tm))


if __name__ == "__main__":
    main()
