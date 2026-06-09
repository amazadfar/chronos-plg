"""
Integrated trading strategy combining all components.

Brings together:
- Signal generation from quantile predictions
- Regime detection and gating
- Position sizing with risk controls
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.models.baselines.random_walk import BaselineModel
from src.strategy.execution_intent import ExecutionIntentBuilder, ExecutionPolicy
from src.strategy.position_sizing import PositionConstraints, PositionSizer
from src.strategy.regime_detector import RegimeDetector
from src.strategy.signals import QuantileSignalGenerator

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    direction: int = 0  # 1 = long, -1 = short
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0
    regime: str = "normal"


@dataclass(frozen=True)
class StrategyRiskConstraints:
    """Scenario-level risk constraints applied to position targets."""

    max_exposure: float | None = None
    max_turnover_per_step: float | None = None
    cooldown_bars_after_drawdown: int = 0
    drawdown_threshold: float | None = None
    realized_return_column: str = "return_1"


class TradingStrategy:
    """
    Complete trading strategy implementation.
    
    Pipeline:
    1. Model generates quantile predictions
    2. SignalGenerator converts predictions to signals
    3. RegimeDetector classifies market state
    4. PositionSizer calculates position sizes
    5. Output: position series for backtesting
    """
    
    def __init__(
        self,
        model: Optional[BaselineModel] = None,
        signal_generator: Optional[QuantileSignalGenerator] = None,
        position_sizer: Optional[PositionSizer] = None,
        regime_detector: Optional[RegimeDetector] = None,
        use_regime_gating: bool = True,
        market_type: str = "futures",
        allow_short: Optional[bool] = None,
        borrow_availability_column: str = "borrow_available",
        execution_policy: ExecutionPolicy | str | None = None,
        risk_constraints: Optional[StrategyRiskConstraints] = None,
    ):
        """
        Args:
            model: Forecasting model (must have predict() returning quantiles)
            signal_generator: Signal generator
            position_sizer: Position sizer
            regime_detector: Regime detector
            use_regime_gating: Whether to adjust positions based on regime
            market_type: spot, margin, or futures
            allow_short: Global short enable/disable (default from config/market)
            borrow_availability_column: Column used for margin short eligibility
            execution_policy: Taker/maker policy abstraction
            risk_constraints: Scenario-level risk controls
        """
        settings = get_settings()
        self.model = model
        self.signal_generator = signal_generator or QuantileSignalGenerator()
        self.position_sizer = position_sizer or PositionSizer()
        self.regime_detector = regime_detector or RegimeDetector()
        self.use_regime_gating = use_regime_gating
        self.market_type = market_type.lower()
        self.borrow_availability_column = borrow_availability_column
        self.allow_short = (
            allow_short
            if allow_short is not None
            else getattr(settings.strategy, "allow_short", self.market_type != "spot")
        )

        default_policy = getattr(settings.strategy, "execution_policy", ExecutionPolicy.TAKER_ONLY.value)
        policy = execution_policy or default_policy
        self.execution_policy = policy if isinstance(policy, ExecutionPolicy) else ExecutionPolicy(policy)
        self.intent_builder = ExecutionIntentBuilder(policy=self.execution_policy)

        self.risk_constraints = risk_constraints or StrategyRiskConstraints(
            max_exposure=getattr(settings.strategy, "max_exposure", None),
            max_turnover_per_step=getattr(settings.strategy, "max_turnover_per_step", None),
            cooldown_bars_after_drawdown=getattr(settings.strategy, "drawdown_cooldown_bars", 0),
            drawdown_threshold=getattr(settings.strategy, "drawdown_threshold", None),
            realized_return_column=getattr(settings.strategy, "realized_return_column", "return_1"),
        )
        
        self._is_fitted = False

    @staticmethod
    def _extract_position_constraints(data: pd.DataFrame) -> Optional[PositionConstraints]:
        columns = ["tick_size", "lot_size", "min_qty", "min_notional"]
        if not set(columns).issubset(set(data.columns)):
            return None

        metadata = data[columns].ffill().bfill()
        if metadata.empty:
            return None
        first = metadata.iloc[0].to_dict()
        return PositionConstraints.from_mapping(first)

    def _resolve_short_allowed(self, data: pd.DataFrame) -> pd.Series | bool:
        if self.market_type == "spot" or not self.allow_short:
            return False

        if self.market_type == "margin" and self.borrow_availability_column in data.columns:
            return data[self.borrow_availability_column].reindex(data.index).fillna(0).astype(bool)

        return True

    @staticmethod
    def _apply_turnover_cap(positions: pd.Series, turnover_cap: float) -> pd.Series:
        if turnover_cap <= 0 or positions.empty:
            return positions

        capped = pd.Series(0.0, index=positions.index, dtype=float)
        prev = 0.0
        for ts in positions.index:
            target = float(positions.loc[ts])
            capped_target = np.clip(target, prev - turnover_cap, prev + turnover_cap)
            capped.loc[ts] = float(capped_target)
            prev = float(capped_target)
        return capped

    def _apply_cooldown_after_drawdown(
        self,
        positions: pd.Series,
        data: pd.DataFrame,
    ) -> pd.Series:
        cooldown_bars = int(self.risk_constraints.cooldown_bars_after_drawdown)
        drawdown_threshold = self.risk_constraints.drawdown_threshold
        returns_col = self.risk_constraints.realized_return_column

        if cooldown_bars <= 0 or drawdown_threshold is None:
            return positions
        if returns_col not in data.columns:
            return positions

        returns = data[returns_col].reindex(positions.index).fillna(0.0)
        constrained = positions.copy()

        equity = 1.0
        peak = 1.0
        prev_position = 0.0
        cooldown_remaining = 0
        threshold = abs(float(drawdown_threshold))

        for ts in constrained.index:
            period_return = float(returns.loc[ts])
            equity *= 1.0 + prev_position * period_return
            peak = max(peak, equity)
            drawdown = 0.0 if peak <= 0 else (equity - peak) / peak

            if drawdown <= -threshold:
                cooldown_remaining = max(cooldown_remaining, cooldown_bars)

            if cooldown_remaining > 0:
                constrained.loc[ts] = 0.0
                cooldown_remaining -= 1

            prev_position = float(constrained.loc[ts])

        return constrained

    def _apply_scenario_risk_constraints(
        self,
        positions: pd.Series,
        data: pd.DataFrame,
    ) -> pd.Series:
        constrained = positions.copy()

        if self.risk_constraints.max_exposure is not None:
            max_exposure = abs(float(self.risk_constraints.max_exposure))
            constrained = constrained.clip(-max_exposure, max_exposure)

        if self.risk_constraints.max_turnover_per_step is not None:
            constrained = self._apply_turnover_cap(
                constrained,
                float(self.risk_constraints.max_turnover_per_step),
            )

        constrained = self._apply_cooldown_after_drawdown(constrained, data)
        return constrained
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "TradingStrategy":
        """
        Fit the underlying model.
        
        Args:
            X: Training features
            y: Training targets
        """
        if self.model is not None:
            self.model.fit(X, y)
        self._is_fitted = True
        return self
    
    def generate_positions(
        self,
        data: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate position series from data.
        
        Args:
            data: DataFrame with price data and features
            predictions: Pre-computed predictions (if None, uses model)
            
        Returns:
            DataFrame with positions and metadata
        """
        # Get predictions if not provided
        if predictions is None:
            if self.model is None:
                raise ValueError("No model or predictions provided")
            predictions = self.model.predict(data)

        forecasts = self.signal_generator.build_forecast_snapshots(predictions)
        decisions = self.signal_generator.generate_trade_decisions(predictions)
        
        # Detect regimes
        if "close" in data.columns:
            regimes = self.regime_detector.detect_regimes(data, close_col="close")
        elif "regime" in data.columns:
            regimes = data[["regime"]].copy()
            regimes["vol_7d"] = data.get("vol_7d", 0.02)
        else:
            regimes = pd.DataFrame({"regime": "normal"}, index=data.index)
        
        # Get regime multipliers
        if self.use_regime_gating:
            regime_multipliers = self.regime_detector.get_regime_multipliers(
                regimes["regime"]
            )
        else:
            regime_multipliers = pd.Series(1.0, index=data.index)
        
        # Get predicted volatility for sizing
        if "realized_vol_6" in data.columns:
            predicted_vol = data["realized_vol_6"]
        elif "vol_7d" in regimes.columns:
            predicted_vol = regimes["vol_7d"]
        else:
            predicted_vol = pd.Series(0.02, index=data.index)  # Default 2%

        short_allowed = self._resolve_short_allowed(data)
        constraints = self._extract_position_constraints(data)
        prices = data["close"] if "close" in data.columns else None

        # Calculate positions
        positions = self.position_sizer.calculate_sizes(
            decisions,
            predicted_vol,
            regime_multipliers,
            prices=prices,
            position_constraints=constraints,
            short_allowed=short_allowed,
            max_turnover_per_step=self.risk_constraints.max_turnover_per_step,
        )
        positions = self._apply_scenario_risk_constraints(positions, data)
        execution_intents = self.intent_builder.build_for_positions(positions)
        
        # Combine output
        output = pd.DataFrame(index=data.index)
        output["position"] = positions
        output["signal"] = decisions["signal"]
        output["signal_strength"] = decisions["signal_strength"]
        output["signal_confidence"] = decisions["signal_confidence"]
        output["decision_reason"] = decisions["decision_reason"]
        output["tradeable"] = decisions["tradeable"]
        output["uncertainty"] = forecasts["uncertainty"]
        output["regime"] = regimes["regime"]
        output["regime_multiplier"] = regime_multipliers
        output["q10"] = forecasts["q10"]
        output["q50"] = forecasts["q50"]
        output["q90"] = forecasts["q90"]
        output["execution_action"] = execution_intents["execution_action"]
        output["execution_side"] = execution_intents["execution_side"]
        output["execution_order_type"] = execution_intents["execution_order_type"]
        output["execution_policy"] = execution_intents["execution_policy"]
        output["requires_execution"] = execution_intents["requires_execution"]

        if isinstance(short_allowed, pd.Series):
            output["short_allowed"] = short_allowed.reindex(output.index).fillna(False).astype(int)
        else:
            output["short_allowed"] = int(bool(short_allowed))

        # Position changes (for cost calculation)
        output["position_change"] = output["position"].diff().abs().fillna(output["position"].abs())
        
        # Log summary
        long_pct = (output["position"] > 0).mean()
        short_pct = (output["position"] < 0).mean()
        flat_pct = (output["position"] == 0).mean()
        avg_size = output["position"][output["position"] != 0].abs().mean()
        avg_turnover = output["position_change"].mean()
        
        logger.info(
            f"Strategy positions: {long_pct:.1%} long, {short_pct:.1%} short, "
            f"{flat_pct:.1%} flat, avg size: {avg_size:.2f}x, "
            f"avg turnover: {avg_turnover:.3f}"
        )
        
        return output
    
    def calculate_returns(
        self,
        positions: pd.DataFrame,
        actual_returns: pd.Series,
        fee_rate: float = 0.0005,
        slippage_bps: float = 2.0,
    ) -> pd.DataFrame:
        """
        Calculate strategy returns from positions.
        
        Args:
            positions: DataFrame with position column
            actual_returns: Actual forward returns
            fee_rate: Trading fee rate per side
            slippage_bps: Slippage in basis points
            
        Returns:
            DataFrame with return columns
        """
        # Align data
        common_idx = positions.index.intersection(actual_returns.index)
        pos = positions.loc[common_idx, "position"]
        returns = actual_returns.loc[common_idx]
        
        # Lagged position (we trade at close, returns are from next period)
        pos_lagged = pos.shift(1)
        
        # Gross returns
        gross_returns = pos_lagged * returns
        
        # Calculate costs
        position_changes = positions.loc[common_idx, "position_change"].shift(1)
        fees = position_changes * fee_rate * 2  # Both entry and exit
        slippage = position_changes * (slippage_bps / 10000)
        total_costs = fees + slippage
        
        # Net returns
        net_returns = gross_returns - total_costs
        
        output = pd.DataFrame(index=common_idx)
        output["gross_return"] = gross_returns
        output["fees"] = fees
        output["slippage"] = slippage
        output["total_costs"] = total_costs
        output["net_return"] = net_returns
        output["cumulative_return"] = (1 + net_returns.fillna(0)).cumprod() - 1
        
        return output


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    returns = np.random.normal(0, 0.02, n)
    price = 40000 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        "close": price,
        "return_1": returns,
        "realized_vol_6": pd.Series(returns).rolling(6).std().fillna(0.02).values,
        "forward_return": np.roll(returns, -1),
    }, index=dates)
    
    # Sample predictions
    predictions = pd.DataFrame({
        "q10": np.random.normal(-0.015, 0.008, n),
        "q50": np.random.normal(0, 0.003, n),
        "q90": np.random.normal(0.015, 0.008, n),
    }, index=dates)
    
    # Create strategy
    strategy = TradingStrategy()
    
    # Generate positions
    positions = strategy.generate_positions(data, predictions)
    
    print("\nPosition Summary:")
    print(positions[["position", "signal", "regime"]].describe())
    
    # Calculate returns
    strat_returns = strategy.calculate_returns(
        positions,
        data["forward_return"],
    )
    
    print("\nStrategy Returns:")
    print(strat_returns.describe())
    print(f"\nFinal Cumulative Return: {strat_returns['cumulative_return'].iloc[-1]:.2%}")


if __name__ == "__main__":
    main()
