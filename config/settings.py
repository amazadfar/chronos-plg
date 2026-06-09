"""
Centralized configuration for the Chronos-2 BTC trading system.

All hyperparameters, paths, and settings in one place.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DataPaths:
    """Data directory paths."""
    root: Path = Path(__file__).parent.parent / "data"
    
    @property
    def raw(self) -> Path:
        return self.root / "raw"
    
    @property
    def processed(self) -> Path:
        return self.root / "processed"
    
    @property
    def features(self) -> Path:
        return self.root / "features"
    
    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for path in [self.raw, self.processed, self.features]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BinanceConfig:
    """Binance API configuration."""
    # API endpoints
    futures_base_url: str = "https://fapi.binance.com"
    spot_base_url: str = "https://api.binance.com"
    ws_base_url: str = "wss://fstream.binance.com"
    
    # Rate limiting
    max_requests_per_minute: int = 1200
    request_weight_limit: int = 2400
    
    # Data fetching
    symbol: str = "BTCUSDT"
    interval: str = "4h"
    
    # Historical data start (reliable perps data)
    start_date: str = "2021-01-01"
    

@dataclass(frozen=True)
class MacroConfig:
    """Macro data configuration."""
    # Tickers for yfinance
    dxy_ticker: str = "DX-Y.NYB"
    spx_ticker: str = "^GSPC"
    vix_ticker: str = "^VIX"
    tnx_ticker: str = "^TNX"  # 10Y yield
    irx_ticker: str = "^IRX"  # 13-week T-bill (proxy for 2Y)
    
    # Data start
    start_date: str = "2020-01-01"  # Earlier for macro warm-up


@dataclass(frozen=True)
class TargetConfig:
    """Target variable configuration."""
    # Horizon
    horizon_candles: int = 1  # 1 candle = 4h forward return
    
    # Quantiles to predict
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    
    # Realized volatility window
    rv_window_candles: int = 6  # 6 * 4h = 24h


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering configuration."""
    # Lookback windows (in 4h candles)
    return_windows: tuple[int, ...] = (1, 6, 42)  # 4h, 24h, 7d
    vol_windows: tuple[int, ...] = (6, 42)  # 24h, 7d
    
    # Volume z-score window
    volume_zscore_window: int = 42  # 7d
    
    # Funding rate MA
    funding_ma_window: int = 6  # 24h
    
    # OI change windows
    oi_change_windows: tuple[int, ...] = (1, 6)  # 4h, 24h


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward evaluation configuration."""
    # Option A: Weekly retrain (default)
    train_window_days: int = 180  # 6 months
    test_window_days: int = 7     # 1 week
    step_size_days: int = 7       # weekly step
    
    # Minimum samples for training
    min_train_samples: int = 500
    
    # Alternative: Monthly retrain (set via mode)
    mode: Literal["weekly", "monthly"] = "weekly"
    
    @property
    def effective_train_days(self) -> int:
        if self.mode == "monthly":
            return 365  # 12 months
        return self.train_window_days
    
    @property
    def effective_test_days(self) -> int:
        if self.mode == "monthly":
            return 30
        return self.test_window_days
    
    @property
    def effective_step_days(self) -> int:
        if self.mode == "monthly":
            return 30
        return self.step_size_days


@dataclass(frozen=True)
class CostConfig:
    """Trading cost assumptions."""
    # Fees (conservative)
    fee_rate: float = 0.0005  # 0.05% per side (5 bps)
    
    # Slippage model: slippage = base + vol_multiplier * realized_vol
    slippage_base_bps: float = 2.0
    slippage_vol_multiplier: float = 0.5
    
    # Latency buffer (candles)
    execution_delay_candles: int = 1  # Execute at T+1 open


@dataclass(frozen=True)
class StrategyConfig:
    """Trading strategy configuration."""
    # Entry thresholds
    entry_threshold: float = 0.003  # 0.3% expected return
    risk_limit: float = 0.015       # 1.5% max adverse move
    
    # Position sizing
    max_leverage: float = 2.0
    max_exposure: float | None = None
    max_turnover_per_step: float | None = None
    
    # No-trade zone
    uncertainty_threshold: float = 0.03  # 3% q90-q10 spread
    
    # Entry policy mode:
    # - threshold: legacy fixed-threshold gating
    # - net_edge: require expected edge to clear expected cost + risk buffer
    entry_policy: str = "threshold"
    net_edge_cost_multiplier: float = 1.0
    net_edge_risk_multiplier: float = 0.0
    expected_cost_column: str = "expected_cost"
    predicted_risk_column: str = "predicted_risk"
    expected_cost_holding_bars: int = 1
    expected_cost_round_trip: bool = True

    # Execution policy abstraction
    execution_policy: str = "taker_only"
    allow_short: bool = True
    realized_return_column: str = "return_1"
    drawdown_threshold: float | None = None
    drawdown_cooldown_bars: int = 0
    
    # Regime definitions (7d metrics)
    trend_return_threshold: float = 0.10
    trend_vol_threshold: float = 0.05
    chop_return_threshold: float = 0.03
    chop_vol_threshold: float = 0.04
    panic_vol_threshold: float = 0.08
    
    # Regime adjustments
    chop_size_multiplier: float = 0.5
    panic_size_multiplier: float = 0.25


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""
    # Chronos-2
    chronos_model_name: str = "amazon/chronos-t5-base"  # or large
    chronos_context_length: int = 512
    chronos_prediction_length: int = 1  # 1 candle = 4h
    chronos_num_samples: int = 100  # for probabilistic forecasts
    
    # LightGBM baseline
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.05
    lgb_n_estimators: int = 500
    lgb_early_stopping_rounds: int = 50
    
    # Random seed
    seed: int = 42


@dataclass(frozen=True)
class Settings:
    """Main settings container."""
    paths: DataPaths = field(default_factory=DataPaths)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    def __post_init__(self) -> None:
        """Initialize directories."""
        self.paths.ensure_dirs()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience exports
SETTINGS = get_settings()
