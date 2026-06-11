"""Causal technical feature extraction for OHLCV bars.

This module ports the useful indicator ideas from the legacy trading archive
into Chronos-PLG's timestamp-safe feature pipeline. Every feature is computed
from the current or prior completed candle only; forward-looking pivots and
post-hoc segment labels intentionally do not belong here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-12


def compute_technical_features(
    ohlcv: pd.DataFrame,
    *,
    ma_windows: tuple[int, ...] = (5, 10, 20, 50),
    rsi_window: int = 14,
    atr_window: int = 14,
    bbands_window: int = 20,
    stoch_window: int = 14,
    cci_window: int = 20,
    vroc_window: int = 14,
    fib_window: int = 100,
) -> pd.DataFrame:
    """Compute causal price, volume, momentum, trend, and range features."""

    _validate_ohlcv(ohlcv)

    open_ = ohlcv["open"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float)

    features = pd.DataFrame(index=ohlcv.index)
    candle_range = high - low
    body = close - open_
    safe_close = _safe_abs(close)
    safe_open = _safe_abs(open_)
    safe_range = candle_range.replace(0, np.nan)

    features["tech_hlc3"] = (high + low + close) / 3.0
    features["tech_ohlc4"] = (open_ + high + low + close) / 4.0
    features["tech_range_pct"] = candle_range / _safe_abs(low)
    features["tech_body_pct"] = body / safe_open
    features["tech_upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / safe_close
    features["tech_lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / safe_close
    features["tech_close_position_in_range"] = ((close - low) / safe_range).clip(0.0, 1.0)
    features["tech_volume_return_1"] = volume.pct_change()

    for window in ma_windows:
        sma = close.rolling(window=window).mean()
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        features[f"tech_sma_distance_{window}"] = (close / sma) - 1.0
        features[f"tech_ema_distance_{window}"] = (close / ema) - 1.0
        features[f"tech_sma_slope_{window}"] = sma.pct_change()
        features[f"tech_ema_slope_{window}"] = ema.pct_change()

    features["tech_pma_5_10_20"] = _periodic_ma_signal(close, periods=(5, 10, 20))
    features[f"tech_rsi_{rsi_window}"] = _rsi(close, window=rsi_window)

    tr = _true_range(high=high, low=low, close=close)
    atr = tr.rolling(window=atr_window).mean()
    features[f"tech_atr_{atr_window}"] = atr
    features[f"tech_atr_pct_{atr_window}"] = atr / safe_close

    bb_mid = close.rolling(window=bbands_window).mean()
    bb_std = close.rolling(window=bbands_window).std()
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)
    bb_width = bb_upper - bb_lower
    features[f"tech_bbands_width_{bbands_window}"] = bb_width / _safe_abs(bb_mid)
    features[f"tech_bbands_position_{bbands_window}"] = ((close - bb_lower) / bb_width.replace(0, np.nan)).clip(0.0, 1.0)

    macd_line, macd_signal = _macd(close)
    features["tech_macd"] = macd_line
    features["tech_macd_signal"] = macd_signal
    features["tech_macd_hist"] = macd_line - macd_signal

    features["tech_obv"] = _obv(close=close, volume=volume)
    features[f"tech_vroc_{vroc_window}"] = volume.pct_change(vroc_window)

    stoch_k, stoch_d = _stochastic(high=high, low=low, close=close, window=stoch_window)
    features[f"tech_stoch_k_{stoch_window}"] = stoch_k
    features[f"tech_stoch_d_{stoch_window}"] = stoch_d
    features[f"tech_stoch_cross_{stoch_window}"] = np.sign(stoch_k - stoch_d)

    features[f"tech_cci_{cci_window}"] = _cci(
        high=high,
        low=low,
        close=close,
        window=cci_window,
    )

    plus_di, minus_di, adx = _adx(high=high, low=low, close=close, window=atr_window)
    features[f"tech_plus_di_{atr_window}"] = plus_di
    features[f"tech_minus_di_{atr_window}"] = minus_di
    features[f"tech_adx_{atr_window}"] = adx

    ha_open, ha_close = _heikin_ashi(open_=open_, high=high, low=low, close=close)
    features["tech_ha_close"] = ha_close
    features["tech_ha_body_pct"] = (ha_close - ha_open) / _safe_abs(ha_open)

    rolling_high = high.rolling(window=fib_window).max()
    rolling_low = low.rolling(window=fib_window).min()
    fib_range = (rolling_high - rolling_low).replace(0, np.nan)
    features[f"tech_fib_position_{fib_window}"] = ((close - rolling_low) / fib_range).clip(0.0, 1.0)

    return features.replace([np.inf, -np.inf], np.nan)


def _validate_ohlcv(ohlcv: pd.DataFrame) -> None:
    missing = [col for col in ("open", "high", "low", "close", "volume") if col not in ohlcv.columns]
    if missing:
        raise ValueError(f"OHLCV missing required columns for technical features: {missing}")


def _safe_abs(series: pd.Series) -> pd.Series:
    return series.abs().where(series.abs() > EPSILON, np.nan)


def _true_range(*, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _periodic_ma_signal(close: pd.Series, *, periods: tuple[int, ...]) -> pd.Series:
    slopes = pd.DataFrame(
        {
            f"ema_{period}": close.ewm(span=period, adjust=False, min_periods=period).mean().diff()
            for period in periods
        },
        index=close.index,
    )
    signal = np.where(
        (slopes > 0).all(axis=1),
        1,
        np.where((slopes < 0).all(axis=1), -1, 0),
    )
    return pd.Series(signal, index=close.index, dtype=float)


def _rsi(close: pd.Series, *, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(
    close: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, macd_signal


def _obv(*, close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def _stochastic(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    low_min = low.rolling(window=window).min()
    high_max = high.rolling(window=window).max()
    stoch_k = 100.0 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
    stoch_d = stoch_k.rolling(window=smooth).mean()
    return stoch_k, stoch_d


def _cci(*, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    typical = (high + low + close) / 3.0
    typical_ma = typical.rolling(window=window).mean()
    mean_deviation = typical.rolling(window=window).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )
    return (typical - typical_ma) / (0.015 * mean_deviation.replace(0, np.nan))


def _adx(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    atr = _true_range(high=high, low=low, close=close).rolling(window=window).mean()

    plus_di = 100.0 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs().replace(0, np.nan)
    adx = dx.rolling(window=window).mean()
    return plus_di, minus_di, adx


def _heikin_ashi(
    *,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    ha_close = (open_ + high + low + close) / 4.0
    ha_open = pd.Series(np.nan, index=open_.index, dtype=float)
    if open_.empty:
        return ha_open, ha_close

    ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2.0
    for i in range(1, len(open_)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    return ha_open, ha_close
