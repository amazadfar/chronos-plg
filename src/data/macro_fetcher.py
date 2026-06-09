"""
Macro data fetcher using yfinance.

Fetches:
- DXY (US Dollar Index)
- SPX (S&P 500)
- VIX (Volatility Index)
- Treasury yields (2Y, 10Y)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional
import logging

from config.settings import get_settings, MacroConfig

logger = logging.getLogger(__name__)

try:  # pragma: no cover - dependency availability depends on runtime environment.
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


class MacroFetcher:
    """Fetch macro data from Yahoo Finance."""
    
    # Known FOMC meeting dates (2021-2026)
    # These should be updated periodically
    FOMC_DATES = [
        # 2021
        "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
        "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
        # 2022
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
        "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
        # 2023
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
        # 2026
        "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    ]
    
    # CPI release dates (approximate - typically mid-month)
    # These should be verified against actual release calendar
    CPI_DATES = [
        # 2021
        "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13",
        "2021-05-12", "2021-06-10", "2021-07-13", "2021-08-11",
        "2021-09-14", "2021-10-13", "2021-11-10", "2021-12-10",
        # 2022
        "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12",
        "2022-05-11", "2022-06-10", "2022-07-13", "2022-08-10",
        "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13",
        # 2023
        "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
        "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
        "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
        # 2024
        "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
        "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
        "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
        # 2025
        "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
        "2025-05-13", "2025-06-11", "2025-07-10", "2025-08-12",
        "2025-09-10", "2025-10-09", "2025-11-13", "2025-12-10",
    ]
    
    def __init__(self, config: Optional[MacroConfig] = None):
        self.config = config or get_settings().macro
    
    def fetch_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily data for a single ticker.
        
        Returns:
            DataFrame with OHLCV data
        """
        end = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        logger.info(f"Fetching {ticker} from {start_date} to {end}")
        if yf is None:
            logger.error("yfinance is not installed; macro market fetch is unavailable")
            return pd.DataFrame()
        
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end,
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()
        
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        # Ensure timezone-aware index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df
    
    def fetch_all_macro(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch all macro indicators and combine into single DataFrame.
        
        Returns:
            DataFrame with columns: dxy, spx, vix, yield_2y, yield_10y
        """
        start = start_date or self.config.start_date
        
        tickers = {
            "dxy": self.config.dxy_ticker,
            "spx": self.config.spx_ticker,
            "vix": self.config.vix_ticker,
            "yield_10y": self.config.tnx_ticker,
            "yield_2y": self.config.irx_ticker,
        }
        
        dfs = {}
        for name, ticker in tickers.items():
            df = self.fetch_ticker(ticker, start, end_date)
            if not df.empty:
                # Use Close price
                dfs[name] = df["Close"].rename(name)
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine all series
        combined = pd.concat(dfs.values(), axis=1)
        
        # Compute yield curve spread (10Y - 2Y)
        if "yield_10y" in combined.columns and "yield_2y" in combined.columns:
            combined["yield_curve_2_10"] = combined["yield_10y"] - combined["yield_2y"]
        
        # Compute daily returns
        for col in ["dxy", "spx"]:
            if col in combined.columns:
                combined[f"{col}_return_1d"] = np.log(combined[col] / combined[col].shift(1))
        
        logger.info(f"Combined macro data: {combined.shape}")
        return combined
    
    def generate_event_flags(
        self,
        index: pd.DatetimeIndex,
        post_event_hours: int = 6,
    ) -> pd.DataFrame:
        """
        Generate binary event flags for economic events.
        
        Creates flags for:
        - FOMC announcement days
        - CPI release days
        - Post-event windows (next N hours after event)
        
        Args:
            index: DatetimeIndex to generate flags for
            post_event_hours: Hours to flag as "post-event window"
            
        Returns:
            DataFrame with event flag columns
        """
        fomc_dates = pd.to_datetime(self.FOMC_DATES).tz_localize("UTC")
        cpi_dates = pd.to_datetime(self.CPI_DATES).tz_localize("UTC")
        
        # Create flags DataFrame
        flags = pd.DataFrame(index=index)
        
        # FOMC flags
        flags["is_fomc_day"] = flags.index.normalize().isin(fomc_dates)
        
        # CPI flags
        flags["is_cpi_day"] = flags.index.normalize().isin(cpi_dates)
        
        # Post-event windows (using rolling window approach)
        if len(index) >= 2:
            bar_delta_hours = (
                index.to_series().diff().dropna().median().total_seconds() / 3600.0
            )
        else:
            bar_delta_hours = 4.0
        bar_delta_hours = max(1.0, float(bar_delta_hours))
        post_candles = max(1, int(np.ceil(post_event_hours / bar_delta_hours)))
        
        flags["post_fomc_window"] = (
            flags["is_fomc_day"]
            .astype(int)
            .rolling(window=post_candles + 1, min_periods=1)
            .max()
            .astype(bool)
        )
        
        flags["post_cpi_window"] = (
            flags["is_cpi_day"]
            .astype(int)
            .rolling(window=post_candles + 1, min_periods=1)
            .max()
            .astype(bool)
        )
        
        # Combined event flag
        flags["is_event_window"] = flags["post_fomc_window"] | flags["post_cpi_window"]
        
        # Convert to int for model consumption
        for col in flags.columns:
            flags[col] = flags[col].astype(int)
        
        logger.info(f"Generated event flags: {flags.sum().to_dict()}")
        return flags
    
    def align_to_interval(
        self,
        macro_df: pd.DataFrame,
        target_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Align daily macro data to target timestamps.
        
        Uses forward-fill with 1-day lag to prevent look-ahead bias
        (macro data for day D is only available after market close).
        
        Args:
            macro_df: Daily macro data
            target_index: DatetimeIndex to align to
            
        Returns:
            DataFrame aligned to target timestamps
        """
        if macro_df.empty:
            return pd.DataFrame(index=target_index)
        
        # Shift by 1 day to ensure no look-ahead (use yesterday's close)
        macro_lagged = macro_df.copy()
        macro_lagged.index = macro_lagged.index + pd.Timedelta(days=1)
        
        # Reindex to target frequency with forward-fill
        aligned = macro_lagged.reindex(target_index, method="ffill")
        
        logger.info("Aligned macro data to %s timestamps", len(aligned))
        return aligned

    def align_to_4h(
        self,
        macro_df: pd.DataFrame,
        target_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Backward-compatible alias for legacy callers."""
        return self.align_to_interval(macro_df, target_index)


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    fetcher = MacroFetcher()
    
    # Fetch last 60 days
    from datetime import timedelta
    start = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")
    
    macro = fetcher.fetch_all_macro(start)
    print("\nMacro data:")
    print(macro.tail(10))
    
    # Generate event flags for a sample 4h index
    sample_index = pd.date_range(start=start, periods=100, freq="4h", tz="UTC")
    flags = fetcher.generate_event_flags(sample_index)
    print("\nEvent flags:")
    print(flags[flags["is_event_window"] == 1].head(10))


if __name__ == "__main__":
    main()
