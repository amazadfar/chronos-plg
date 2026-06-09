"""
Quantile-based trading signal generator.

Converts probabilistic forecasts (q10, q50, q90) into actionable
trading signals with confidence levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from config.settings import StrategyConfig, get_settings

logger = logging.getLogger(__name__)

EntryPolicy = Literal["threshold", "net_edge"]


@dataclass
class Signal:
    """Trading signal with metadata."""

    direction: int  # -1 (short), 0 (flat), 1 (long)
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale based on uncertainty
    q10: float
    q50: float
    q90: float
    
    @property
    def is_trade(self) -> bool:
        return self.direction != 0


@dataclass(frozen=True)
class ForecastSnapshot:
    """Forecast snapshot extracted from model quantiles."""

    q10: float
    q50: float
    q90: float
    uncertainty: float


@dataclass(frozen=True)
class TradeDecision:
    """Trade decision derived from a forecast snapshot."""

    direction: int
    strength: float
    confidence: float
    uncertainty: float
    reason: str

    @property
    def is_trade(self) -> bool:
        return self.direction != 0


class QuantileSignalGenerator:
    """
    Generate trading signals from quantile predictions.
    
    Signal logic:
    - Long: q50 > threshold AND q10 > -risk_limit (upside expected, limited downside)
    - Short: q50 < -threshold AND q90 < risk_limit (downside expected, limited upside)
    - Flat: otherwise (uncertainty too high or no edge)
    
    Confidence based on:
    - Narrow prediction interval = high confidence
    - Wide interval = low confidence
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        entry_threshold: Optional[float] = None,
        risk_limit: Optional[float] = None,
        uncertainty_threshold: Optional[float] = None,
        regime_entry_multipliers: Optional[dict[str, float]] = None,
        entry_policy: Optional[EntryPolicy] = None,
        net_edge_cost_multiplier: Optional[float] = None,
        net_edge_risk_multiplier: Optional[float] = None,
        expected_cost_column: Optional[str] = None,
        predicted_risk_column: Optional[str] = None,
        expected_cost_holding_bars: Optional[int] = None,
        expected_cost_round_trip: Optional[bool] = None,
    ):
        """
        Args:
            config: Strategy configuration
            entry_threshold: Minimum q50 for entry (overrides config)
            risk_limit: Maximum adverse move at extreme quantile
            uncertainty_threshold: Max q90-q10 spread for trading
        """
        self.config = config or get_settings().strategy
        
        self.entry_threshold = entry_threshold or self.config.entry_threshold
        self.risk_limit = risk_limit or self.config.risk_limit
        self.uncertainty_threshold = uncertainty_threshold or self.config.uncertainty_threshold
        config_entry_policy = str(getattr(self.config, "entry_policy", "threshold")).lower()
        self.entry_policy: EntryPolicy = (
            "net_edge"
            if str(entry_policy or config_entry_policy).lower() == "net_edge"
            else "threshold"
        )
        self.net_edge_cost_multiplier = float(
            net_edge_cost_multiplier
            if net_edge_cost_multiplier is not None
            else getattr(self.config, "net_edge_cost_multiplier", 1.0)
        )
        self.net_edge_risk_multiplier = float(
            net_edge_risk_multiplier
            if net_edge_risk_multiplier is not None
            else getattr(self.config, "net_edge_risk_multiplier", 0.0)
        )
        self.expected_cost_column = str(
            expected_cost_column
            if expected_cost_column is not None
            else getattr(self.config, "expected_cost_column", "expected_cost")
        )
        self.predicted_risk_column = str(
            predicted_risk_column
            if predicted_risk_column is not None
            else getattr(self.config, "predicted_risk_column", "predicted_risk")
        )
        self.expected_cost_holding_bars = int(
            max(
                expected_cost_holding_bars
                if expected_cost_holding_bars is not None
                else getattr(self.config, "expected_cost_holding_bars", 1),
                1,
            )
        )
        self.expected_cost_round_trip = bool(
            expected_cost_round_trip
            if expected_cost_round_trip is not None
            else getattr(self.config, "expected_cost_round_trip", True)
        )
        self.regime_entry_multipliers = {
            str(key).lower(): float(value)
            for key, value in (regime_entry_multipliers or {}).items()
        }

    @staticmethod
    def _validate_prediction_columns(predictions: pd.DataFrame) -> None:
        required = {"q10", "q50", "q90"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    @staticmethod
    def _compute_uncertainty(q10: float, q90: float) -> float:
        return float(q90 - q10)

    @staticmethod
    def _compute_confidence(uncertainty: float) -> float:
        max_uncertainty = 0.10
        return float(np.clip(1 - uncertainty / max_uncertainty, 0.0, 1.0))

    def _compute_strength(self, q50: float, threshold: float) -> float:
        scale = max(threshold * 3, 1e-12)
        return float(np.clip(abs(q50) / scale, 0.0, 1.0))

    def _resolve_entry_threshold_for_regime(self, regime: object) -> float:
        if not self.regime_entry_multipliers:
            return float(self.entry_threshold)
        key = str(regime).lower() if regime is not None else ""
        multiplier = self.regime_entry_multipliers.get(key, 1.0)
        return float(max(self.entry_threshold * multiplier, 1e-12))

    def _is_net_edge_policy(self) -> bool:
        return self.entry_policy == "net_edge"

    def _edge_buffer(self, expected_cost: float, predicted_risk: float) -> float:
        if not self._is_net_edge_policy():
            return 0.0
        cost = max(float(expected_cost), 0.0)
        risk = max(float(predicted_risk), 0.0)
        return float(cost * self.net_edge_cost_multiplier + risk * self.net_edge_risk_multiplier)

    def build_forecast_snapshot(
        self,
        q10: float,
        q50: float,
        q90: float,
    ) -> ForecastSnapshot:
        """Create normalized forecast snapshot."""
        return ForecastSnapshot(
            q10=float(q10),
            q50=float(q50),
            q90=float(q90),
            uncertainty=self._compute_uncertainty(q10, q90),
        )

    def build_forecast_snapshots(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Extract forecast layer from prediction dataframe.

        Returns:
            DataFrame with q10/q50/q90 and uncertainty.
        """
        self._validate_prediction_columns(predictions)
        forecasts = predictions[["q10", "q50", "q90"]].copy()
        forecasts["uncertainty"] = forecasts["q90"] - forecasts["q10"]
        return forecasts

    def decide_trade(
        self,
        snapshot: ForecastSnapshot,
        *,
        entry_threshold: Optional[float] = None,
        expected_cost: float = 0.0,
        predicted_risk: Optional[float] = None,
    ) -> TradeDecision:
        """Convert one forecast snapshot into a trade decision."""
        threshold_base = (
            float(max(entry_threshold, 1e-12))
            if entry_threshold is not None
            else float(max(self.entry_threshold, 1e-12))
        )
        risk = float(snapshot.uncertainty if predicted_risk is None else max(float(predicted_risk), 0.0))
        threshold = threshold_base + self._edge_buffer(expected_cost=expected_cost, predicted_risk=risk)
        confidence = self._compute_confidence(snapshot.uncertainty)
        strength = self._compute_strength(snapshot.q50, threshold=max(threshold, 1e-12))

        if snapshot.uncertainty > self.uncertainty_threshold:
            return TradeDecision(
                direction=0,
                strength=0.0,
                confidence=confidence,
                uncertainty=snapshot.uncertainty,
                reason="high_uncertainty",
            )

        if snapshot.q50 > threshold and snapshot.q10 > -self.risk_limit:
            direction = 1
            reason = "long_edge"
        elif snapshot.q50 < -threshold and snapshot.q90 < self.risk_limit:
            direction = -1
            reason = "short_edge"
        else:
            direction = 0
            reason = "edge_below_cost_buffer" if self._is_net_edge_policy() else "no_edge"

        return TradeDecision(
            direction=direction,
            strength=strength,
            confidence=confidence,
            uncertainty=snapshot.uncertainty,
            reason=reason,
        )
    
    def generate_signal(
        self,
        q10: float,
        q50: float,
        q90: float,
        regime: Optional[str] = None,
        expected_cost: float = 0.0,
        predicted_risk: Optional[float] = None,
    ) -> Signal:
        """
        Generate a single signal from quantile predictions.
        
        Args:
            q10: 10th percentile prediction
            q50: Median prediction
            q90: 90th percentile prediction
            
        Returns:
            Signal object with direction and metadata
        """
        snapshot = self.build_forecast_snapshot(q10=q10, q50=q50, q90=q90)
        decision = self.decide_trade(
            snapshot,
            entry_threshold=self._resolve_entry_threshold_for_regime(regime),
            expected_cost=expected_cost,
            predicted_risk=predicted_risk,
        )

        return Signal(
            direction=decision.direction,
            strength=decision.strength,
            confidence=decision.confidence,
            q10=snapshot.q10,
            q50=snapshot.q50,
            q90=snapshot.q90,
        )

    def generate_trade_decisions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Generate decision layer from forecasts.

        Returns:
            DataFrame with decision fields:
            - signal: -1/0/1
            - signal_strength: 0-1
            - signal_confidence: 0-1
            - decision_reason: long_edge/short_edge/no_edge/high_uncertainty
            - tradeable: uncertainty gate pass/fail
            - expected_cost/predicted_risk/required_edge/edge_margin
            - uncertainty, q10, q50, q90
        """
        forecasts = self.build_forecast_snapshots(predictions)

        q10 = forecasts["q10"]
        q50 = forecasts["q50"]
        q90 = forecasts["q90"]
        uncertainty = forecasts["uncertainty"]

        if "regime" in predictions.columns and self.regime_entry_multipliers:
            regime_scale = (
                predictions["regime"]
                .astype(str)
                .str.lower()
                .map(self.regime_entry_multipliers)
                .fillna(1.0)
                .astype(float)
            )
            entry_thresholds = (self.entry_threshold * regime_scale).clip(lower=1e-12)
        else:
            entry_thresholds = pd.Series(
                float(max(self.entry_threshold, 1e-12)),
                index=forecasts.index,
                dtype=float,
            )

        if self.expected_cost_column in predictions.columns:
            expected_cost = pd.to_numeric(
                predictions[self.expected_cost_column],
                errors="coerce",
            ).reindex(forecasts.index).fillna(0.0).clip(lower=0.0)
        else:
            expected_cost = pd.Series(0.0, index=forecasts.index, dtype=float)

        if self.predicted_risk_column in predictions.columns:
            predicted_risk = pd.to_numeric(
                predictions[self.predicted_risk_column],
                errors="coerce",
            ).reindex(forecasts.index).fillna(uncertainty).clip(lower=0.0)
        else:
            predicted_risk = uncertainty.clip(lower=0.0)

        edge_buffer = pd.Series(0.0, index=forecasts.index, dtype=float)
        if self._is_net_edge_policy():
            edge_buffer = (
                expected_cost * self.net_edge_cost_multiplier
                + predicted_risk * self.net_edge_risk_multiplier
            ).clip(lower=0.0)

        required_edge = entry_thresholds + edge_buffer

        decisions = pd.DataFrame(index=forecasts.index)
        decisions["uncertainty"] = uncertainty
        decisions["signal_confidence"] = (1 - uncertainty / 0.10).clip(0, 1)
        decisions["entry_threshold_effective"] = entry_thresholds
        decisions["expected_cost"] = expected_cost
        decisions["predicted_risk"] = predicted_risk
        decisions["edge_buffer"] = edge_buffer
        decisions["required_edge"] = required_edge
        decisions["edge_margin"] = q50.abs() - required_edge
        decisions["entry_policy"] = self.entry_policy
        decisions["signal_strength"] = (q50.abs() / (required_edge.clip(lower=1e-12) * 3)).clip(0, 1)
        decisions["signal"] = 0
        decisions["decision_reason"] = "no_edge"

        tradeable = uncertainty <= self.uncertainty_threshold
        decisions["tradeable"] = tradeable.astype(int)
        decisions.loc[~tradeable, "signal_strength"] = 0.0
        decisions.loc[~tradeable, "decision_reason"] = "high_uncertainty"

        long_cond = (q50 > required_edge) & (q10 > -self.risk_limit) & tradeable
        short_cond = (q50 < -required_edge) & (q90 < self.risk_limit) & tradeable

        decisions.loc[long_cond, "signal"] = 1
        decisions.loc[short_cond, "signal"] = -1
        decisions.loc[long_cond, "decision_reason"] = "long_edge"
        decisions.loc[short_cond, "decision_reason"] = "short_edge"

        if self._is_net_edge_policy():
            blocked_by_edge = (
                (decisions["signal"] == 0)
                & tradeable
                & (q50.abs() <= required_edge)
            )
            decisions.loc[blocked_by_edge, "decision_reason"] = "edge_below_cost_buffer"

        decisions["q10"] = q10
        decisions["q50"] = q50
        decisions["q90"] = q90
        return decisions
    
    def generate_signals(
        self,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate signals for a DataFrame of predictions.
        
        Args:
            predictions: DataFrame with q10, q50, q90 columns
            
        Returns:
            DataFrame with signal columns:
            - signal: -1, 0, 1
            - signal_strength: 0-1
            - signal_confidence: 0-1
            - uncertainty: q90 - q10
        """
        results = self.generate_trade_decisions(predictions)

        # Log statistics
        total = len(results)
        longs = (results["signal"] == 1).sum()
        shorts = (results["signal"] == -1).sum()
        flats = (results["signal"] == 0).sum()
        
        logger.info(
            f"Signals: {longs} long ({longs/total:.1%}), "
            f"{shorts} short ({shorts/total:.1%}), "
            f"{flats} flat ({flats/total:.1%})"
        )
        
        return results


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    # Sample predictions
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    predictions = pd.DataFrame({
        "q10": np.random.normal(-0.015, 0.01, n),
        "q50": np.random.normal(0, 0.005, n),
        "q90": np.random.normal(0.015, 0.01, n),
    }, index=dates)
    
    # Generate signals
    generator = QuantileSignalGenerator(
        entry_threshold=0.003,
        risk_limit=0.015,
        uncertainty_threshold=0.04,
    )
    
    signals = generator.generate_signals(predictions)
    
    print("\nSignal Distribution:")
    print(signals["signal"].value_counts())
    
    print("\nSample Signals:")
    print(signals[["signal", "signal_confidence", "q50", "uncertainty"]].head(10))


if __name__ == "__main__":
    main()
