# Chronos-2 BTC Trading Experiment Specification

> **Version:** 1.0  
> **Date:** 2026-01-05  
> **Status:** Draft - pending review

## Executive Summary

Build a quantile-based trading system for BTCUSDT using Chronos-2 probabilistic forecasts on a 4-hour horizon. The system must beat LightGBM baseline net of costs on walk-forward evaluation, or we kill it.

---

## 1. Problem Definition

### 1.1 Horizon
- **Primary:** 4 hours (6 candles per day)
- **Rationale:** Balance between signal quality and sample size (~2,190 samples/year). Tolerates covariate latency (funding rate 8h updates), realistic execution without HFT infra.

### 1.2 Universe
- **Phase 1:** BTCUSDT perpetual (Binance)
- **Future:** Add ETHUSDT, then majors for cross-series signals

### 1.3 Target Variable

**Primary: Return Quantiles**
```
r_t = log(close_{t+H} / close_t)  where H = 4 hours

Outputs: q10, q50, q90 of return distribution
```

**Secondary: Realized Volatility**
```
rv_t = std(4h returns over next 24h window) = std of next 6 returns
```

### 1.4 Objective
Produce **calibrated probabilistic forecasts** that translate into positive risk-adjusted returns after costs.

---

## 2. Execution Reality

### 2.1 Fee Assumptions
| Type | Rate |
|------|------|
| Maker | 0.02% |
| Taker | 0.04% |
| **Assumed (conservative)** | **0.05%** per side |

### 2.2 Slippage Model
```python
slippage_bps = base_slippage + vol_multiplier * realized_vol
# base_slippage = 2 bps
# vol_multiplier = 0.5
# Example: if rv = 4%, slippage ≈ 2 + 0.5*4 = 4 bps
```

### 2.3 Latency Constraints
- **Decision time:** candle close (T)
- **Execution time:** next candle open (T + ~1 min)
- No look-ahead: all features use data available at T-1 or earlier

---

## 3. Data Sources

### 3.1 Core Price Data (Binance)
| Field | Source | Granularity |
|-------|--------|-------------|
| OHLCV | Binance Futures API | 4h candles |
| Funding Rate | Binance Futures API | 8h (interpolate to 4h) |
| Open Interest | Binance Futures API | 4h snapshots |

### 3.2 Liquidation Data (Binance WebSocket)
| Field | Source | Notes |
|-------|--------|-------|
| Long liquidations (USD) | `wss://fstream.binance.com/ws/btcusdt@forceOrder` | Aggregate per 4h |
| Short liquidations (USD) | Same stream | Aggregate per 4h |

### 3.3 Macro Data (yfinance / FRED)
| Field | Ticker | Granularity | Notes |
|-------|--------|-------------|-------|
| DXY | DX-Y.NYB | Daily | Forward-fill to 4h |
| S&P 500 | ^GSPC | Daily | Forward-fill |
| VIX | ^VIX | Daily | Forward-fill |
| 2Y Yield | ^IRX or FRED | Daily | Forward-fill |
| 10Y Yield | ^TNX | Daily | Forward-fill |

### 3.4 Event Flags
| Event | Source | Treatment |
|-------|--------|-----------|
| FOMC dates | Manual / FRED | Binary flag + 6h post-event window |
| CPI releases | Manual | Binary flag + 6h post-event window |
| Quarterly expiry | Known dates | Binary flag |

---

## 4. Feature Engineering

### 4.1 Price-Derived Features
```python
# Returns
log_return_1  = log(close / close.shift(1))
log_return_6  = log(close / close.shift(6))   # 24h
log_return_42 = log(close / close.shift(42))  # 7d

# Volatility
realized_vol_6  = returns.rolling(6).std()    # 24h RV
realized_vol_42 = returns.rolling(42).std()   # 7d RV

# Range
atr_6 = ATR(high, low, close, period=6)

# Volume
volume_zscore = (volume - volume.rolling(42).mean()) / volume.rolling(42).std()
```

### 4.2 Perps Microstructure
```python
funding_rate           # raw (cumulative or instantaneous)
funding_rate_ma_6      # 24h MA
open_interest          # absolute
oi_change_pct_1        # 4h change
oi_change_pct_6        # 24h change
long_liq_usd_1         # liquidations in past 4h
short_liq_usd_1
liq_imbalance          # (long_liq - short_liq) / (long_liq + short_liq + eps)
```

### 4.3 Macro (lagged appropriately)
```python
dxy_return_1d          # daily return, lagged 1 candle
spx_return_1d
vix_level              # lagged
yield_curve_2_10       # 10Y - 2Y spread
```

### 4.4 Missingness Tracking
```python
# For each covariate group, track:
has_funding    = ~funding_rate.isna()
has_oi         = ~open_interest.isna()
has_liqs       = ~long_liq_usd_1.isna()
has_macro      = ~dxy_return_1d.isna()
```

---

## 5. Walk-Forward Evaluation

### 5.1 Configuration Options

**Option A: Weekly Retrain (Default)**
```python
train_window   = 180 days  # 6 months × 6 candles × ~30 = 1080 samples
test_window    = 7 days    # ~42 samples
step_size      = 7 days
min_train_date = "2021-01-01"  # start of reliable perps data
```

**Option B: Monthly Retrain (Alternative)**
```python
train_window   = 365 days  # 12 months
test_window    = 30 days
step_size      = 30 days
```

### 5.2 Data Split Example (Option A)
```
Fold 1: Train [2021-01-01 to 2021-06-30], Test [2021-07-01 to 2021-07-07]
Fold 2: Train [2021-01-08 to 2021-07-07], Test [2021-07-08 to 2021-07-14]
...
```

### 5.3 Anti-Leak Validation
- All features computed using only past data
- Funding rate: use T-8h value for T prediction
- Macro: use T-1d close for T prediction
- Liquidations: use T-4h aggregates for T prediction

---

## 6. Success Criteria

### 6.1 Minimum Viability (must pass ALL)

| Criterion | Threshold | Metric |
|-----------|-----------|--------|
| **Primary edge gate** | **Profit Factor (net) > 1.0** | Sum(net winners) / abs(sum(net losers)) |
| **Beat LightGBM baseline** | Sharpe > LightGBM Sharpe + 0.1 | On test folds, net of costs |
| **Survive costs** | Sharpe > 0.5 | After fees + slippage |
| **Regime stability** | CV of Sharpe < 1.5 | Across trend/chop/panic regimes |
| **No decay** | Last-25% Sharpe > 0.7 × First-25% Sharpe | Walk-forward windows |

### 6.2 Stretch Goals
- Sharpe > 1.5 net of costs
- Max drawdown < 15%
- Profit factor > 1.8

---

## 7. Kill Criteria

**Stop immediately if:**
1. Chronos-2 doesn't beat random walk on pinball loss
2. LightGBM baseline has higher Sharpe net of costs
3. Edge exists only in one regime (e.g., only bull markets)
4. Performance collapses in 2024+ data (recent regime)
5. Needs < 5 bps total costs to be profitable (unrealistic)

---

## 8. Model Stack

### 8.1 Baselines (implement first)
1. **Random Walk:** predict q50 = 0, q10/q90 from historical distribution
2. **EWMA:** exponentially weighted mean/std of returns
3. **LightGBM:** tabular features → quantile regression (3 models or multi-output)

### 8.2 Chronos-2
- Input: past 512 timesteps of target + covariates
- Output: q10, q50, q90 for horizon H
- Mode: zero-shot first, fine-tune only if promising

### 8.3 Meta-Model (optional)
- Input: Chronos quantile outputs + raw features
- Output: trade decision (long/short/flat) + position size
- Model: LightGBM classifier or small MLP

---

## 9. Strategy Rules

### 9.1 Entry Conditions
```python
# Long entry
long_signal = (q50 > entry_threshold) & (q10 > -risk_limit)

# Short entry
short_signal = (q50 < -entry_threshold) & (q90 < risk_limit)

# Default thresholds (tune via walk-forward)
entry_threshold = 0.003  # 0.3% expected return
risk_limit = 0.015       # 1.5% max adverse move at q10/q90
```

### 9.2 Position Sizing
```python
raw_size = q50 / predicted_vol  # volatility-normalized
position_size = clip(raw_size, -max_leverage, max_leverage)
max_leverage = 2.0  # conservative
```

### 9.3 No-Trade Zone
```python
uncertainty = q90 - q10
no_trade = uncertainty > uncertainty_threshold  # ~3% spread
```

### 9.4 Regime Gating
```python
# Regime definitions
trend_regime = abs(returns_7d) > 0.10 and vol_7d < 0.05
chop_regime  = abs(returns_7d) < 0.03 and vol_7d < 0.04
panic_regime = vol_7d > 0.08

# Regime adjustments
if chop_regime:
    position_size *= 0.5
if panic_regime:
    position_size *= 0.25  # or skip entirely
```

---

## 10. File Structure

```
chronos-plg/
├── experiment_spec.md          # This file
├── requirements.txt
├── pyproject.toml
│
├── config/
│   └── settings.py             # All hyperparameters, paths, API keys
│
├── data/
│   ├── raw/                    # Downloaded data (gitignored)
│   ├── processed/              # Cleaned, aligned datasets
│   └── features/               # Feature matrices ready for training
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── binance_fetcher.py  # OHLCV, funding, OI
│   │   ├── liquidation_collector.py  # WebSocket stream
│   │   ├── macro_fetcher.py    # yfinance integration
│   │   ├── labels.py           # Target computation
│   │   └── build_dataset.py    # Unified pipeline
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── price_features.py
│   │   ├── perps_features.py
│   │   ├── macro_features.py
│   │   └── feature_pipeline.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines/
│   │   │   ├── random_walk.py
│   │   │   ├── ewma.py
│   │   │   └── lightgbm_quantile.py
│   │   ├── chronos2_runner.py
│   │   └── meta_model.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── walk_forward.py
│   │   ├── metrics.py          # Pinball loss, calibration, etc.
│   │   └── trading_metrics.py  # Sharpe, drawdown, etc.
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── signals.py          # Quantile-based signals
│   │   ├── position_sizing.py
│   │   └── regime_detector.py
│   │
│   └── backtest/
│       ├── __init__.py
│       ├── engine.py
│       ├── costs.py
│       └── report.py
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   ├── test_labels.py
│   └── test_backtest.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   ├── 03_chronos2_experiments.ipynb
│   └── 04_final_report.ipynb
│
├── scripts/
│   ├── download_data.py
│   ├── build_features.py
│   ├── train_baselines.py
│   ├── run_chronos2.py
│   └── run_backtest.py
│
└── stress_tests/
    ├── slippage_sensitivity.py
    ├── covariate_ablation.py
    └── lag_simulation.py
```

---

## 11. Implementation Order

### Sprint 1: Foundation (Week 1)
1. Set up project structure and dependencies
2. Implement `binance_fetcher.py` for OHLCV + funding + OI
3. Implement `labels.py` for quantile target computation
4. Basic data validation and anti-leak checks

### Sprint 2: Baselines (Week 2)
5. Implement random walk baseline
6. Implement LightGBM quantile baseline
7. Build walk-forward harness
8. First baseline results

### Sprint 3: Chronos-2 (Week 3)
9. Implement `chronos2_runner.py`
10. Run zero-shot evaluation
11. Compare vs baselines
12. **Decision point: continue or kill**

### Sprint 4: Strategy + Backtest (Week 4)
13. Implement strategy rules
14. Build backtest engine with costs
15. Generate full evaluation report
16. Robustness tests

---

## 12. Dependencies

```
# Core
python>=3.10
pandas>=2.0
numpy>=1.24
polars>=0.20  # optional, for speed

# Data
python-binance>=1.0.19
yfinance>=0.2.36
websockets>=12.0

# ML
scikit-learn>=1.4
lightgbm>=4.0
chronos-forecasting>=1.0  # Amazon Chronos-2

# Deep Learning (for Chronos)
torch>=2.0
transformers>=4.35

# Evaluation
scipy>=1.11

# Visualization
matplotlib>=3.8
plotly>=5.18
seaborn>=0.13

# Notebooks
jupyter>=1.0
```

---

## Approval Checklist

- [ ] Horizon (4h) confirmed
- [ ] Target (return quantiles) confirmed
- [ ] Data sources accessible
- [ ] Cost assumptions realistic
- [ ] Kill criteria agreed
- [ ] Ready to proceed to Sprint 1
