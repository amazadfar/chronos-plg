"""
Binance Futures data fetcher for BTCUSDT perpetual.

Fetches:
- OHLCV candles (interval-configurable)
- Funding rate history
- Open interest snapshots
"""
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timezone
from typing import Optional
import time
import logging

from config.settings import get_settings, BinanceConfig

logger = logging.getLogger(__name__)


class BinanceFetcher:
    """Fetch historical data from Binance Futures API."""
    
    def __init__(self, config: Optional[BinanceConfig] = None):
        self.config = config or get_settings().binance
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_timestamps: list[float] = []
    
    async def __aenter__(self) -> "BinanceFetcher":
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args) -> None:
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self) -> None:
        """Simple rate limiting to stay within API limits."""
        now = time.time()
        # Keep only timestamps from last minute
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 60]
        
        if len(self._request_timestamps) >= self.config.max_requests_per_minute - 10:
            sleep_time = 60 - (now - self._request_timestamps[0]) + 1
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)
        
        self._request_timestamps.append(time.time())
    
    async def _get(self, endpoint: str, params: dict) -> dict:
        """Make GET request with rate limiting."""
        await self._rate_limit()
        
        url = f"{self.config.futures_base_url}{endpoint}"
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Binance API error {response.status}: {text}")
            return await response.json()
    
    async def fetch_ohlcv(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from Binance Futures.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date (default: now)
            symbol: Trading pair (default: BTCUSDT)
            interval: Candle interval (default: 4h)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        symbol = symbol or self.config.symbol
        interval = interval or self.config.interval
        
        start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
        if end_date:
            end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
        else:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        all_candles = []
        current_start = start_ts
        limit = 1500  # Max candles per request
        
        logger.info(f"Fetching OHLCV for {symbol} from {start_date} to {end_date or 'now'}")
        
        while current_start < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": limit,
            }
            
            candles = await self._get("/fapi/v1/klines", params)
            
            if not candles:
                break
            
            all_candles.extend(candles)
            
            # Move to next batch
            last_ts = candles[-1][0]
            if last_ts == current_start:
                break
            current_start = last_ts + 1
            
            logger.debug(f"Fetched {len(all_candles)} candles so far...")
        
        if not all_candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        df["trades"] = df["trades"].astype(int)
        
        # Keep only relevant columns
        df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
        df = df.set_index("timestamp").sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]
        
        logger.info(f"Fetched {len(df)} OHLCV candles")
        return df
    
    async def fetch_funding_rate(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch funding rate history.
        
        Funding rates are recorded every 8 hours (00:00, 08:00, 16:00 UTC).
        
        Returns:
            DataFrame with columns: timestamp, funding_rate
        """
        symbol = symbol or self.config.symbol
        
        start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
        if end_date:
            end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
        else:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        all_rates = []
        current_start = start_ts
        limit = 1000
        
        logger.info(f"Fetching funding rates for {symbol}")
        
        while current_start < end_ts:
            params = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": limit,
            }
            
            rates = await self._get("/fapi/v1/fundingRate", params)
            
            if not rates:
                break
            
            all_rates.extend(rates)
            
            last_ts = rates[-1]["fundingTime"]
            if last_ts == current_start:
                break
            current_start = last_ts + 1
        
        if not all_rates:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_rates)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df[["timestamp", "funding_rate"]].set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        
        logger.info(f"Fetched {len(df)} funding rate records")
        return df
    
    async def fetch_open_interest_hist(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch open interest history.
        
        Note: This endpoint has limited historical data (~30 days for some periods).
        For longer history, you may need to use alternative data sources.
        
        Returns:
            DataFrame with columns: timestamp, open_interest, open_interest_value
        """
        symbol = symbol or self.config.symbol
        period = period or self.config.interval
        
        start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
        if end_date:
            end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
        else:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        all_oi = []
        current_start = start_ts
        limit = 500
        max_window_ms = 29 * 24 * 3600 * 1000  # API range guardrail per request
        invalid_range_advance_ms = 30 * 24 * 3600 * 1000

        logger.info(f"Fetching open interest history for {symbol}")

        while current_start < end_ts:
            request_end = min(end_ts, current_start + max_window_ms)
            params = {
                "symbol": symbol,
                "period": period,
                "startTime": current_start,
                "endTime": request_end,
                "limit": limit,
            }
            
            try:
                oi_data = await self._get("/futures/data/openInterestHist", params)
            except Exception as e:
                # OI history endpoint may reject old timestamps or oversized windows.
                error_text = str(e)
                if "startTime" in error_text:
                    logger.warning(
                        "OI history startTime rejected (%s); advancing window by 30 days",
                        current_start,
                    )
                    current_start += invalid_range_advance_ms
                    continue
                logger.warning(f"OI history fetch error (may be limited data): {e}")
                break
            
            if not oi_data:
                current_start = request_end + 1
                continue
            
            all_oi.extend(oi_data)
            
            last_ts = oi_data[-1]["timestamp"]
            if last_ts == current_start:
                break
            current_start = last_ts + 1
        
        if not all_oi:
            logger.warning("No OI history data returned - may need alternative source")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_oi)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["open_interest"] = df["sumOpenInterest"].astype(float)
        df["open_interest_value"] = df["sumOpenInterestValue"].astype(float)
        df = df[["timestamp", "open_interest", "open_interest_value"]].set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        
        logger.info(f"Fetched {len(df)} OI records")
        return df
    
    async def fetch_all(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch all data sources and return as dict.
        
        Returns:
            Dict with keys: 'ohlcv', 'funding_rate', 'open_interest'
        """
        start = start_date or self.config.start_date
        resolved_interval = interval or self.config.interval
        
        ohlcv_task = self.fetch_ohlcv(start, end_date, interval=resolved_interval)
        funding_task = self.fetch_funding_rate(start, end_date)
        oi_task = self.fetch_open_interest_hist(start, end_date, period=resolved_interval)
        
        ohlcv, funding, oi = await asyncio.gather(ohlcv_task, funding_task, oi_task)
        
        return {
            "ohlcv": ohlcv,
            "funding_rate": funding,
            "open_interest": oi,
        }
    
    def align_funding_to_interval(self, funding_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align funding-rate samples to OHLCV timestamps.
        
        Uses forward-fill with proper lag (funding at T applies to positions at T).
        
        Args:
            funding_df: DataFrame with funding_rate column
            ohlcv_df: DataFrame with OHLCV data (for timestamps)
            
        Returns:
            DataFrame with funding_rate aligned to OHLCV timestamps
        """
        if funding_df.empty or ohlcv_df.empty:
            return pd.DataFrame(index=ohlcv_df.index, columns=["funding_rate"])
        
        # Reindex to OHLCV timestamps with forward-fill
        # Shift by 1 to ensure we use past funding rate (no look-ahead)
        aligned = funding_df.reindex(ohlcv_df.index, method="ffill")
        aligned = aligned.shift(1)  # Use previous funding rate
        
        return aligned

    def align_funding_to_4h(self, funding_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible alias for legacy callers."""
        return self.align_funding_to_interval(funding_df, ohlcv_df)


async def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    async with BinanceFetcher() as fetcher:
        # Fetch last 30 days of data
        from datetime import timedelta
        start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        
        data = await fetcher.fetch_all(start)
        
        for name, df in data.items():
            print(f"\n{name}:")
            print(f"  Shape: {df.shape}")
            if not df.empty:
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
                print(df.head(3))


if __name__ == "__main__":
    asyncio.run(main())
