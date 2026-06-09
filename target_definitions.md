# Target Definitions

## Primary Target: Forward Return

```python
forward_return = log(close_{t+1} / close_t)
```

Where:
- `t` = current 4h candle timestamp
- `t+1` = next 4h candle (4 hours forward)

**Quantile Predictions:**
The model predicts the 10th, 50th, and 90th percentiles of the return distribution.

| Quantile | Interpretation |
|----------|----------------|
| q10 | Worst case (90% of outcomes better than this) |
| q50 | Expected/median return |
| q90 | Best case (90% of outcomes worse than this) |

---

## Secondary Target: Realized Volatility

```python
forward_realized_vol = std(returns_{t+1}, ..., returns_{t+6})
```

Where:
- Window = next 6 candles = 24 hours
- Measures uncertainty/risk over the next day

---

## Regime Labels

Regimes are computed from 7-day metrics:

```python
returns_7d = log(close_t / close_{t-42})  # 42 candles = 7 days
vol_7d = std(4h_returns over past 42 candles)
```

| Regime | Condition |
|--------|-----------|
| **trend** | \|returns_7d\| > 10% AND vol_7d < 5% |
| **chop** | \|returns_7d\| < 3% AND vol_7d < 4% |
| **panic** | vol_7d > 8% |
| **normal** | everything else |

---

## Anti-Leak Rules

> [!CAUTION]
> These rules are **non-negotiable** to prevent overfitting to future data.

### Feature Timing

| Feature Type | Available At | Lag Applied |
|--------------|--------------|-------------|
| OHLCV | Candle close T | None (use T-1 for features) |
| Funding rate | Every 8h | Use previous 8h value |
| Open interest | Updated ~4h | Use T-1 snapshot |
| Macro (DXY, VIX) | EOD | Use T-1 day close |
| Event flags | Known in advance | No lag needed |

### Validation Checks

1. **Correlation check**: No feature should have >0.5 correlation with target
2. **Timestamp check**: All feature timestamps < target timestamp
3. **Shift test**: Predictions should not change if we shift features by 1 period

### Implementation

```python
# CORRECT: Use lagged values
features["funding_rate"] = funding_rate.shift(1)
features["macro"] = macro.shift(1)  # Already 1-day lag in fetcher

# WRONG: Using current values
features["funding_rate"] = funding_rate  # LEAKAGE!
```
