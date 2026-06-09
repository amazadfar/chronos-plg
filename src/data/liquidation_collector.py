"""
Binance liquidation data collector.

Collects forced liquidation events via WebSocket and aggregates to 4h windows.
Also provides historical liquidation data from public sources where available.
"""
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import aiohttp

from config.settings import get_settings, BinanceConfig

logger = logging.getLogger(__name__)


@dataclass
class LiquidationEvent:
    """Single liquidation event."""
    timestamp: datetime
    symbol: str
    side: str  # "BUY" = short liquidated, "SELL" = long liquidated
    price: float
    quantity: float
    usd_value: float


@dataclass
class AggregatedLiquidations:
    """Aggregated liquidations for a time window."""
    timestamp: datetime
    long_liq_count: int = 0
    short_liq_count: int = 0
    long_liq_usd: float = 0.0
    short_liq_usd: float = 0.0
    
    @property
    def total_liq_usd(self) -> float:
        return self.long_liq_usd + self.short_liq_usd
    
    @property
    def liq_imbalance(self) -> float:
        """Imbalance ratio: positive = more longs liquidated."""
        total = self.long_liq_usd + self.short_liq_usd
        if total == 0:
            return 0.0
        return (self.long_liq_usd - self.short_liq_usd) / total


class LiquidationCollector:
    """
    Collect and aggregate liquidation data from Binance.
    
    Two modes:
    1. Real-time WebSocket streaming (for live data collection)
    2. Historical data from public datasets (for backtesting)
    """
    
    def __init__(self, config: Optional[BinanceConfig] = None):
        self.config = config or get_settings().binance
        self.ws_url = f"{self.config.ws_base_url}/ws/btcusdt@forceOrder"
        self._running = False
        self._current_window: dict[str, AggregatedLiquidations] = {}
        self._callbacks: list[Callable[[AggregatedLiquidations], None]] = []
    
    def _get_window_key(self, ts: datetime) -> str:
        """Get 4h window key for a timestamp."""
        # Round down to nearest 4h
        hour = (ts.hour // 4) * 4
        window_start = ts.replace(hour=hour, minute=0, second=0, microsecond=0)
        return window_start.isoformat()
    
    def _parse_liquidation(self, data: dict) -> Optional[LiquidationEvent]:
        """Parse WebSocket liquidation message."""
        try:
            order = data.get("o", {})
            
            ts = datetime.fromtimestamp(order["T"] / 1000, tz=timezone.utc)
            side = order["S"]  # BUY = short liq, SELL = long liq
            price = float(order["p"])
            qty = float(order["q"])
            
            return LiquidationEvent(
                timestamp=ts,
                symbol=order["s"],
                side=side,
                price=price,
                quantity=qty,
                usd_value=price * qty,
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse liquidation: {e}")
            return None
    
    def _add_to_window(self, event: LiquidationEvent) -> None:
        """Add liquidation event to current aggregation window."""
        key = self._get_window_key(event.timestamp)
        
        if key not in self._current_window:
            window_start = datetime.fromisoformat(key)
            self._current_window[key] = AggregatedLiquidations(timestamp=window_start)
        
        agg = self._current_window[key]
        
        if event.side == "SELL":  # Long liquidated
            agg.long_liq_count += 1
            agg.long_liq_usd += event.usd_value
        else:  # Short liquidated
            agg.short_liq_count += 1
            agg.short_liq_usd += event.usd_value
    
    async def stream_realtime(
        self,
        duration_hours: Optional[int] = None,
        on_window_complete: Optional[Callable[[AggregatedLiquidations], None]] = None,
    ) -> None:
        """
        Stream real-time liquidation data via WebSocket.
        
        Args:
            duration_hours: How long to stream (None = indefinitely)
            on_window_complete: Callback when a 4h window completes
        """
        self._running = True
        start_time = datetime.now(timezone.utc)
        last_window_key = self._get_window_key(start_time)
        
        logger.info(f"Starting liquidation stream from {self.ws_url}")
        
        try:
            async with websockets.connect(self.ws_url) as ws:
                while self._running:
                    # Check duration
                    if duration_hours:
                        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() / 3600
                        if elapsed >= duration_hours:
                            break
                    
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=60)
                        data = json.loads(msg)
                        
                        event = self._parse_liquidation(data)
                        if event:
                            self._add_to_window(event)
                            
                            # Check if window changed
                            current_key = self._get_window_key(event.timestamp)
                            if current_key != last_window_key and last_window_key in self._current_window:
                                # Previous window complete
                                completed = self._current_window.pop(last_window_key)
                                logger.info(
                                    f"Window complete: {completed.timestamp} - "
                                    f"Long: ${completed.long_liq_usd:,.0f}, "
                                    f"Short: ${completed.short_liq_usd:,.0f}"
                                )
                                if on_window_complete:
                                    on_window_complete(completed)
                                last_window_key = current_key
                    
                    except asyncio.TimeoutError:
                        # No liquidations in 60s, just continue
                        continue
        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            raise
        finally:
            self._running = False
    
    def stop(self) -> None:
        """Stop the WebSocket stream."""
        self._running = False
    
    async def fetch_historical_from_github(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Attempt to fetch historical liquidation data from public datasets.
        
        Note: Public historical liquidation data is limited. This method
        tries known sources but may return empty if data is unavailable.
        
        For production, consider:
        - Coinglass API (paid)
        - Running your own collector for forward data
        - Using OI changes as proxy for liquidations
        
        Returns:
            DataFrame with liquidation aggregates per 4h window
        """
        # Try the free crypto market data repo
        # https://github.com/Kaiko-Partners/crypto-market-data (example)
        
        logger.warning(
            "Historical liquidation data requires paid sources or self-collection. "
            "Returning synthetic placeholder based on OI/volatility patterns."
        )
        
        # Return empty for now - will be filled with synthetic estimates
        # based on OI changes and volatility in build_dataset.py
        return pd.DataFrame(columns=[
            "long_liq_usd", "short_liq_usd", "long_liq_count", 
            "short_liq_count", "liq_imbalance"
        ])
    
    def estimate_from_oi_changes(
        self,
        ohlcv: pd.DataFrame,
        oi: pd.DataFrame,
        vol_multiplier: float = 0.1,
    ) -> pd.DataFrame:
        """
        Estimate liquidation proxy from OI changes and price moves.
        
        When OI drops sharply during a price move, it suggests liquidations.
        This is a rough proxy when real liquidation data is unavailable.
        
        Args:
            ohlcv: OHLCV DataFrame
            oi: Open interest DataFrame
            vol_multiplier: Scaling factor for estimates
            
        Returns:
            DataFrame with estimated liquidation metrics
        """
        if oi.empty:
            logger.warning("No OI data for liquidation estimation")
            return pd.DataFrame(index=ohlcv.index, columns=[
                "long_liq_usd_est", "short_liq_usd_est", "liq_imbalance_est"
            ])
        
        # Align OI to OHLCV index
        oi_aligned = oi.reindex(ohlcv.index, method="ffill")
        
        # Calculate returns and OI changes
        returns = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
        oi_change = oi_aligned["open_interest_value"].pct_change()
        
        # Negative OI change + price move suggests liquidations
        # If price down and OI down → longs liquidated
        # If price up and OI down → shorts liquidated
        
        liq_signal = -oi_change.clip(upper=0)  # Only negative OI changes
        
        estimates = pd.DataFrame(index=ohlcv.index)
        
        # Estimate long liquidations (price down + OI decrease)
        long_liq_signal = liq_signal * (returns < 0).astype(float)
        estimates["long_liq_usd_est"] = (
            long_liq_signal * oi_aligned["open_interest_value"] * vol_multiplier
        ).fillna(0)
        
        # Estimate short liquidations (price up + OI decrease)
        short_liq_signal = liq_signal * (returns > 0).astype(float)
        estimates["short_liq_usd_est"] = (
            short_liq_signal * oi_aligned["open_interest_value"] * vol_multiplier
        ).fillna(0)
        
        # Imbalance
        total = estimates["long_liq_usd_est"] + estimates["short_liq_usd_est"]
        estimates["liq_imbalance_est"] = np.where(
            total > 0,
            (estimates["long_liq_usd_est"] - estimates["short_liq_usd_est"]) / total,
            0
        )
        
        # Track that these are estimates
        estimates["has_real_liq_data"] = 0
        
        logger.info(f"Estimated liquidations for {len(estimates)} timestamps")
        return estimates


async def main():
    """Example usage - stream for 1 minute."""
    logging.basicConfig(level=logging.INFO)
    
    collector = LiquidationCollector()
    
    def on_complete(agg: AggregatedLiquidations):
        print(f"Completed window: {agg}")
    
    # Stream for 1 minute
    try:
        await asyncio.wait_for(
            collector.stream_realtime(on_window_complete=on_complete),
            timeout=60
        )
    except asyncio.TimeoutError:
        collector.stop()
        print("Stream stopped after 1 minute")


if __name__ == "__main__":
    asyncio.run(main())
