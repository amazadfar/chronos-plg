"""Tests for strategy and backtest modules."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.costs import CostModel
from src.backtest.engine import BacktestEngine, BacktestResult
from src.models.baselines import RandomWalkBaseline
from src.strategy.execution_intent import ExecutionIntentBuilder, ExecutionPolicy
from src.strategy.position_sizing import PositionConstraints, PositionSizer
from src.strategy.regime_detector import Regime, RegimeDetector
from src.strategy.signals import QuantileSignalGenerator
from src.strategy.strategy import StrategyRiskConstraints, TradingStrategy


@pytest.fixture
def sample_predictions():
    """Sample predictions DataFrame."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    return pd.DataFrame({
        "q10": np.random.normal(-0.015, 0.008, n),
        "q50": np.random.normal(0, 0.004, n),
        "q90": np.random.normal(0.015, 0.008, n),
    }, index=dates)


@pytest.fixture
def sample_data():
    """Sample price data."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    returns = np.random.normal(0, 0.02, n)
    price = 40000 * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        "close": price,
        "return_1": returns,
        "realized_vol_6": pd.Series(returns).rolling(6).std().fillna(0.02).values,
        "forward_return": np.roll(returns, -1),
    }, index=dates)


class TestSignalGenerator:
    """Tests for QuantileSignalGenerator."""
    
    def test_generate_signals(self, sample_predictions):
        """Should generate signals from predictions."""
        generator = QuantileSignalGenerator(
            entry_threshold=0.003,
            risk_limit=0.02,
        )
        
        signals = generator.generate_signals(sample_predictions)
        
        assert "signal" in signals.columns
        assert "signal_strength" in signals.columns
        assert "signal_confidence" in signals.columns
        assert set(signals["signal"].unique()).issubset({-1, 0, 1})
    
    def test_long_signal_conditions(self):
        """Long signal when q50 > threshold and q10 not too negative."""
        generator = QuantileSignalGenerator(
            entry_threshold=0.003,
            risk_limit=0.02,
        )
        
        predictions = pd.DataFrame({
            "q10": [-0.01],
            "q50": [0.005],  # Above threshold
            "q90": [0.02],
        })
        
        signals = generator.generate_signals(predictions)
        assert signals.iloc[0]["signal"] == 1

    def test_forecast_and_decision_layers(self):
        """Forecast layer and decision layer should be separately available."""
        generator = QuantileSignalGenerator(
            entry_threshold=0.003,
            risk_limit=0.02,
            uncertainty_threshold=0.03,
        )

        predictions = pd.DataFrame(
            {
                "q10": [-0.01, -0.03, -0.02],
                "q50": [0.005, 0.002, -0.006],
                "q90": [0.02, 0.05, 0.01],
            }
        )

        forecasts = generator.build_forecast_snapshots(predictions)
        decisions = generator.generate_trade_decisions(predictions)

        assert {"q10", "q50", "q90", "uncertainty"}.issubset(forecasts.columns)
        assert {"signal", "decision_reason", "tradeable"}.issubset(decisions.columns)
        assert decisions.iloc[0]["decision_reason"] == "long_edge"
        assert decisions.iloc[1]["decision_reason"] == "high_uncertainty"
        assert decisions.iloc[2]["decision_reason"] == "short_edge"

    def test_regime_adaptive_entry_thresholds(self):
        """Regime multipliers should alter effective entry threshold."""
        generator = QuantileSignalGenerator(
            entry_threshold=0.003,
            risk_limit=0.02,
            uncertainty_threshold=0.03,
            regime_entry_multipliers={
                "trend": 0.8,
                "chop": 1.2,
            },
        )

        predictions = pd.DataFrame(
            {
                "q10": [-0.005, -0.005],
                "q50": [0.0025, 0.0025],
                "q90": [0.0100, 0.0100],
                "regime": ["trend", "chop"],
            }
        )

        decisions = generator.generate_trade_decisions(predictions)

        assert decisions.iloc[0]["signal"] == 1
        assert decisions.iloc[1]["signal"] == 0
        assert decisions.iloc[0]["entry_threshold_effective"] < decisions.iloc[1]["entry_threshold_effective"]

    def test_net_edge_policy_filters_marginal_trade(self):
        """Net-edge policy should block trades that do not clear cost+risk buffer."""
        generator = QuantileSignalGenerator(
            entry_threshold=0.003,
            risk_limit=0.02,
            uncertainty_threshold=0.03,
            entry_policy="net_edge",
            net_edge_cost_multiplier=1.0,
            net_edge_risk_multiplier=0.0,
        )

        predictions = pd.DataFrame(
            {
                "q10": [-0.005],
                "q50": [0.0035],
                "q90": [0.0100],
                "expected_cost": [0.0010],
                "predicted_risk": [0.0],
            }
        )

        decisions = generator.generate_trade_decisions(predictions)
        assert decisions.iloc[0]["signal"] == 0
        assert decisions.iloc[0]["decision_reason"] == "edge_below_cost_buffer"
        assert decisions.iloc[0]["required_edge"] > decisions.iloc[0]["entry_threshold_effective"]

    def test_net_edge_policy_allows_trade_when_edge_clears_buffer(self):
        """Net-edge policy should allow trades once expected edge clears required buffer."""
        generator = QuantileSignalGenerator(
            entry_threshold=0.003,
            risk_limit=0.02,
            uncertainty_threshold=0.03,
            entry_policy="net_edge",
            net_edge_cost_multiplier=1.0,
            net_edge_risk_multiplier=0.5,
        )

        predictions = pd.DataFrame(
            {
                "q10": [-0.005],
                "q50": [0.0065],
                "q90": [0.0100],
                "expected_cost": [0.0010],
                "predicted_risk": [0.0010],
            }
        )

        decisions = generator.generate_trade_decisions(predictions)
        assert decisions.iloc[0]["signal"] == 1
        assert decisions.iloc[0]["decision_reason"] == "long_edge"
        assert decisions.iloc[0]["edge_margin"] > 0


class TestPositionSizer:
    """Tests for PositionSizer."""
    
    def test_calculate_sizes(self, sample_predictions):
        """Should calculate position sizes."""
        sizer = PositionSizer(max_leverage=2.0, vol_target=0.15)
        
        signals = pd.DataFrame({
            "signal": np.random.choice([-1, 0, 1], len(sample_predictions)),
            "signal_strength": np.random.uniform(0.3, 1.0, len(sample_predictions)),
            "signal_confidence": np.random.uniform(0.5, 1.0, len(sample_predictions)),
        }, index=sample_predictions.index)
        
        vol = pd.Series(0.02, index=sample_predictions.index)
        
        positions = sizer.calculate_sizes(signals, vol)
        
        assert len(positions) > 0
        assert (positions.abs() <= 2.0).all()  # Within leverage limit

    def test_market_type_leverage_cap(self):
        """Spot market should enforce tighter leverage cap than global max."""
        sizer = PositionSizer(max_leverage=3.0, market_type="spot", min_position=0.0)
        idx = pd.date_range("2023-01-01", periods=5, freq="4h", tz="UTC")

        signals = pd.DataFrame(
            {
                "signal": [1, 1, 1, 1, 1],
                "signal_strength": [1.0] * 5,
                "signal_confidence": [1.0] * 5,
            },
            index=idx,
        )
        vol = pd.Series(0.001, index=idx)
        positions = sizer.calculate_sizes(signals, vol)
        assert (positions.abs() <= 1.0 + 1e-12).all()

    def test_precision_minimum_constraints(self):
        """Sizer should apply lot-size and min-notional constraints."""
        sizer = PositionSizer(max_leverage=2.0, market_type="futures", min_position=0.0)
        idx = pd.date_range("2023-01-01", periods=3, freq="4h", tz="UTC")
        signals = pd.DataFrame(
            {
                "signal": [1, 1, 1],
                "signal_strength": [1.0, 1.0, 1.0],
                "signal_confidence": [1.0, 1.0, 1.0],
            },
            index=idx,
        )
        vol = pd.Series(0.02, index=idx)
        prices = pd.Series(50000.0, index=idx)

        min_notional_too_high = PositionConstraints(lot_size=0.001, min_qty=0.001, min_notional=200.0)
        constrained_flat = sizer.calculate_sizes(
            signals,
            vol,
            prices=prices,
            position_constraints=min_notional_too_high,
            equity=1000.0,
        )
        assert (constrained_flat == 0.0).all()

        min_notional_ok = PositionConstraints(lot_size=0.001, min_qty=0.001, min_notional=100.0)
        constrained_live = sizer.calculate_sizes(
            signals,
            vol,
            prices=prices,
            position_constraints=min_notional_ok,
            equity=1000.0,
        )
        assert (constrained_live > 0).all()
        qty = constrained_live.abs() * 1000.0 / 50000.0
        steps = qty / 0.001
        assert np.allclose(steps, np.round(steps), atol=1e-9)

    def test_short_borrow_gating(self):
        """Margin shorts should be blocked where borrow is unavailable."""
        sizer = PositionSizer(max_leverage=2.0, market_type="margin", min_position=0.0)
        idx = pd.date_range("2023-01-01", periods=3, freq="4h", tz="UTC")
        signals = pd.DataFrame(
            {
                "signal": [-1, -1, -1],
                "signal_strength": [1.0, 1.0, 1.0],
                "signal_confidence": [1.0, 1.0, 1.0],
            },
            index=idx,
        )
        vol = pd.Series(0.02, index=idx)
        borrow_available = pd.Series([1, 0, 1], index=idx).astype(bool)

        positions = sizer.calculate_sizes(signals, vol, short_allowed=borrow_available)
        assert positions.iloc[0] < 0
        assert positions.iloc[1] == 0
        assert positions.iloc[2] < 0

    def test_turnover_cap(self):
        """Turnover cap should limit per-step position changes."""
        sizer = PositionSizer(max_leverage=2.0, market_type="futures", min_position=0.0)
        idx = pd.date_range("2023-01-01", periods=4, freq="4h", tz="UTC")
        signals = pd.DataFrame(
            {
                "signal": [1, -1, 1, -1],
                "signal_strength": [1.0, 1.0, 1.0, 1.0],
                "signal_confidence": [1.0, 1.0, 1.0, 1.0],
            },
            index=idx,
        )
        vol = pd.Series(0.001, index=idx)
        positions = sizer.calculate_sizes(signals, vol, max_turnover_per_step=0.25)
        assert (positions.diff().abs().fillna(positions.abs()) <= 0.25 + 1e-12).all()


class TestExecutionIntentBuilder:
    """Tests for execution intent policy mapping."""

    def test_hybrid_policy_mapping(self):
        """Hybrid policy should be taker for adds and maker for reductions/closes."""
        idx = pd.date_range("2023-01-01", periods=5, freq="4h", tz="UTC")
        positions = pd.Series([0.0, 1.0, 0.5, -0.5, 0.0], index=idx)

        builder = ExecutionIntentBuilder(policy=ExecutionPolicy.HYBRID)
        intents = builder.build_for_positions(positions)

        assert intents.iloc[1]["execution_action"] == "open"
        assert intents.iloc[1]["execution_order_type"] == "taker"
        assert intents.iloc[2]["execution_action"] == "reduce"
        assert intents.iloc[2]["execution_order_type"] == "maker"
        assert intents.iloc[3]["execution_action"] == "reverse"
        assert intents.iloc[3]["execution_order_type"] == "taker"
        assert intents.iloc[4]["execution_action"] == "close"
        assert intents.iloc[4]["execution_order_type"] == "maker"


class TestRegimeDetector:
    """Tests for RegimeDetector."""
    
    def test_detect_regimes(self, sample_data):
        """Should detect regimes from price data."""
        detector = RegimeDetector()
        regimes = detector.detect_regimes(sample_data)
        
        assert "regime" in regimes.columns
        assert all(r in [reg.value for reg in Regime] for r in regimes["regime"].dropna().unique())
    
    def test_regime_multipliers(self):
        """Should return correct multipliers for regimes."""
        detector = RegimeDetector()
        
        regimes = pd.Series(["trend", "panic", "normal", "chop"])
        multipliers = detector.get_regime_multipliers(regimes)
        
        assert multipliers[regimes == "panic"].iloc[0] < 1.0  # Reduced in panic


class TestCostModel:
    """Tests for CostModel."""
    
    def test_calculate_costs(self):
        """Should calculate fees and slippage."""
        model = CostModel(fee_rate=0.0005, slippage_base_bps=2.0)
        
        costs = model.calculate_costs(
            position_change=0.5,
            volatility=0.03,
        )
        
        assert costs.fees > 0
        assert costs.slippage > 0
        assert costs.total == costs.fees + costs.slippage
    
    def test_zero_costs_no_trade(self):
        """No costs when no position change."""
        model = CostModel()
        costs = model.calculate_costs(position_change=0, volatility=0.02)
        
        assert costs.total == 0


class TestBacktestEngine:
    """Tests for BacktestEngine."""
    
    def test_run_backtest(self, sample_data):
        """Should run backtest and produce results."""
        from config.settings import WalkForwardConfig
        
        config = WalkForwardConfig(
            train_window_days=30,
            test_window_days=7,
            step_size_days=14,
            min_train_samples=50,
        )
        
        engine = BacktestEngine(
            model_class=RandomWalkBaseline,
            model_kwargs={"lookback_window": 50},
            walk_forward_config=config,
        )
        
        result = engine.run(
            sample_data,
            start_date="2023-03-01",
            show_progress=False,
        )
        
        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None or result.total_return == 0
        assert result.fold_metrics is not None
        assert len(result.fold_metrics) > 0
        assert result.trades is not None
        if result.returns is not None:
            expected_cost_cols = {
                "event_type",
                "traded_notional",
                "fees",
                "funding",
                "interest",
                "slippage",
                "other_costs",
                "total_costs",
                "net_return",
            }
            assert expected_cost_cols.issubset(set(result.returns.columns))


class TestTradingStrategyPhase5:
    """Phase 5 specific strategy behavior."""

    def test_strategy_risk_constraints(self):
        """Strategy should enforce max exposure and max turnover per step."""
        idx = pd.date_range("2023-01-01", periods=6, freq="4h", tz="UTC")
        data = pd.DataFrame(
            {
                "close": np.linspace(40000, 41000, len(idx)),
                "realized_vol_6": 0.001,
                "return_1": 0.0,
            },
            index=idx,
        )
        predictions = pd.DataFrame(
            {
                "q10": [-0.005] * len(idx),
                "q50": [0.01, -0.01, 0.01, -0.01, 0.01, -0.01],
                "q90": [0.005] * len(idx),
            },
            index=idx,
        )

        strategy = TradingStrategy(
            risk_constraints=StrategyRiskConstraints(
                max_exposure=0.5,
                max_turnover_per_step=0.2,
            ),
        )
        out = strategy.generate_positions(data, predictions)

        assert (out["position"].abs() <= 0.5 + 1e-12).all()
        assert (out["position_change"] <= 0.2 + 1e-12).all()
        expected_cols = {
            "execution_action",
            "execution_side",
            "execution_order_type",
            "execution_policy",
            "requires_execution",
        }
        assert expected_cols.issubset(out.columns)

    def test_margin_borrow_and_drawdown_cooldown(self):
        """Margin strategy should honor borrow availability and drawdown cooldown."""
        idx = pd.date_range("2023-01-01", periods=6, freq="4h", tz="UTC")
        data = pd.DataFrame(
            {
                "close": np.linspace(40000, 40200, len(idx)),
                "realized_vol_6": 0.01,
                "return_1": [0.0, 0.03, 0.03, 0.01, 0.01, 0.01],
                "borrow_available": [1, 1, 0, 1, 1, 1],
            },
            index=idx,
        )
        predictions = pd.DataFrame(
            {
                "q10": [-0.03] * len(idx),
                "q50": [-0.01] * len(idx),
                "q90": [0.005] * len(idx),
            },
            index=idx,
        )

        strategy = TradingStrategy(
            market_type="margin",
            risk_constraints=StrategyRiskConstraints(
                cooldown_bars_after_drawdown=2,
                drawdown_threshold=0.01,
                realized_return_column="return_1",
            ),
        )
        out = strategy.generate_positions(data, predictions)

        assert out.iloc[2]["short_allowed"] == 0
        assert out.iloc[2]["position"] == 0.0
        assert (out["position"].iloc[1:3] == 0.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
