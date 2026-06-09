"""
Dataset builder for the Chronos-2 trading system.

Combines all data sources with proper alignment and anti-leak validation.
"""
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import logging
import json

from config.settings import get_settings
from src.common.timeframe import default_dataset_stem, normalize_timeframe
from src.data.binance_fetcher import BinanceFetcher
from src.data.contracts import (
    DataContractError,
    compute_index_gap_stats,
    validate_raw_data_contracts,
)
from src.data.market_metadata import get_contract_metadata
from src.data.macro_fetcher import MacroFetcher
from src.data.liquidation_collector import LiquidationCollector
from src.data.labels import LabelGenerator

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Build unified dataset from all sources."""

    RAW_DATA_KEYS: tuple[str, ...] = (
        "ohlcv",
        "funding_rate",
        "open_interest",
        "macro",
        "event_flags",
        "liquidations",
        "contract_metadata",
    )
    EXPECTED_OHLCV_FREQ: str = "4h"
    MAX_OHLCV_GAP_RATIO: float = 0.10
    MIN_KEY_FAMILY_AVAILABILITY: dict[str, dict[str, float]] = {
        "futures": {
            "funding": 0.70,
            "open_interest": 0.70,
            "liquidations": 0.70,
            "macro": 0.70,
        },
        "margin": {
            "funding": 0.0,
            "open_interest": 0.0,
            "liquidations": 0.0,
            "macro": 0.70,
        },
        "spot": {
            "funding": 0.0,
            "open_interest": 0.0,
            "liquidations": 0.0,
            "macro": 0.70,
        },
    }
    
    def __init__(self, interval: str | None = None):
        self.settings = get_settings()
        self.interval = normalize_timeframe(interval or self.settings.binance.interval)
        self.expected_ohlcv_freq = self.interval
        self.dataset_stem = default_dataset_stem(timeframe=self.interval, asset="btc")
        self.label_generator = LabelGenerator()
        self.macro_fetcher = MacroFetcher()
        self.liq_collector = LiquidationCollector()

    def _raw_data_save_path(self, name: str) -> Path:
        """Resolve raw-data save path for a dataset key and current interval."""
        raw_dir = self.settings.paths.raw
        if name == "contract_metadata" or self.interval == "4h":
            return raw_dir / f"{name}.parquet"
        return raw_dir / f"{name}_{self.interval}.parquet"

    def _raw_data_load_candidates(self, name: str) -> list[Path]:
        """Resolve candidate raw-data files for loading current interval."""
        raw_dir = self.settings.paths.raw
        if name == "contract_metadata":
            return [raw_dir / "contract_metadata.parquet"]
        if self.interval == "4h":
            return [
                raw_dir / f"{name}.parquet",
                raw_dir / f"{name}_4h.parquet",
            ]
        return [raw_dir / f"{name}_{self.interval}.parquet"]

    def processed_artifact_paths(self) -> dict[str, Path]:
        """Resolved processed dataset artifact paths for current interval."""
        processed_dir = self.settings.paths.processed
        return {
            "dataset": processed_dir / f"{self.dataset_stem}.parquet",
            "metadata": processed_dir / f"{self.dataset_stem}_metadata.json",
            "quality": processed_dir / f"{self.dataset_stem}_quality.json",
        }

    def raw_artifact_path(self, name: str) -> Path:
        """Public helper for scripts to resolve raw artifact paths."""
        return self._raw_data_save_path(name)

    def _build_contract_metadata_frame(self) -> pd.DataFrame:
        """Build one-row DataFrame with exchange/symbol contract metadata."""
        meta = get_contract_metadata(
            exchange="binance",
            market_type="futures",
            symbol=self.settings.binance.symbol,
        )
        return pd.DataFrame([meta.to_dict()], index=[meta.symbol])

    def _normalize_liquidations(
        self,
        liq_df: pd.DataFrame,
        target_index: pd.DatetimeIndex,
        *,
        source: str,
    ) -> pd.DataFrame:
        """
        Normalize liquidation inputs into canonical feature columns.

        source:
            - "real": real liquidation history
            - "estimated": estimated from OI changes
            - "missing": unavailable
        """
        columns = [
            "long_liq_usd_est",
            "short_liq_usd_est",
            "liq_imbalance_est",
            "has_real_liq_data",
            "liq_data_source_code",
        ]

        if len(target_index) == 0:
            return pd.DataFrame(columns=columns)

        normalized = pd.DataFrame(index=target_index)
        if liq_df.empty:
            normalized["long_liq_usd_est"] = np.nan
            normalized["short_liq_usd_est"] = np.nan
            normalized["liq_imbalance_est"] = np.nan
            normalized["has_real_liq_data"] = 0
            normalized["liq_data_source_code"] = -1
            return normalized

        liq = liq_df.copy()
        liq = liq.rename(
            columns={
                "long_liq_usd": "long_liq_usd_est",
                "short_liq_usd": "short_liq_usd_est",
                "liq_imbalance": "liq_imbalance_est",
            }
        )
        liq = liq.reindex(target_index, method="ffill")

        for col in ["long_liq_usd_est", "short_liq_usd_est", "liq_imbalance_est"]:
            normalized[col] = liq[col] if col in liq.columns else np.nan

        if "has_real_liq_data" in liq.columns:
            normalized["has_real_liq_data"] = liq["has_real_liq_data"].fillna(0).astype(int)
        else:
            normalized["has_real_liq_data"] = 1 if source == "real" else 0

        source_code_map = {"real": 1, "estimated": 0, "missing": -1}
        normalized["liq_data_source_code"] = source_code_map.get(source, -1)
        return normalized

    def _assert_ohlcv_integrity(self, ohlcv: pd.DataFrame) -> None:
        """Strict OHLCV quality checks."""
        gaps = compute_index_gap_stats(ohlcv.index, freq=self.expected_ohlcv_freq)
        if gaps.gap_ratio > self.MAX_OHLCV_GAP_RATIO:
            raise ValueError(
                f"OHLCV gap ratio too high: {gaps.gap_ratio:.2%} "
                f"(missing={gaps.missing_points}, expected={gaps.expected_points})"
            )

        required = ["open", "high", "low", "close", "volume"]
        missing_required = [col for col in required if col not in ohlcv.columns]
        if missing_required:
            raise ValueError(f"OHLCV missing required columns: {missing_required}")

        null_rate = float(ohlcv[required].isna().mean().max())
        if null_rate > 0.01:
            raise ValueError(f"OHLCV null rate too high in required columns: {null_rate:.2%}")

    def _assert_data_availability(self, features: pd.DataFrame) -> None:
        """Assert expected data-availability and provenance flags exist."""
        expected_flags = ["has_funding", "has_oi", "has_liqs", "has_macro", "has_real_liq_data"]
        missing = [col for col in expected_flags if col not in features.columns]
        if missing:
            raise ValueError(f"Missing availability/provenance flags: {missing}")

        for col in ["has_funding", "has_oi", "has_liqs", "has_macro", "has_real_liq_data"]:
            vals = set(features[col].dropna().astype(int).unique().tolist())
            if not vals.issubset({0, 1}):
                raise ValueError(f"{col} should be binary but found values: {sorted(vals)}")

    @staticmethod
    def _binary_availability_ratio(features: pd.DataFrame, flag_col: str) -> float:
        """Availability ratio from a binary flag column."""
        if flag_col not in features.columns:
            return 0.0
        series = pd.to_numeric(features[flag_col], errors="coerce").fillna(0.0)
        if series.empty:
            return 0.0
        return float(series.clip(lower=0.0, upper=1.0).mean())

    @staticmethod
    def _source_code_distribution(features: pd.DataFrame, source_col: str) -> dict[str, float]:
        """Distribution of liquidation source codes."""
        if source_col not in features.columns:
            return {}
        series = pd.to_numeric(features[source_col], errors="coerce").dropna()
        if series.empty:
            return {}
        ratios = series.value_counts(normalize=True).sort_index()
        return {str(int(k)): float(v) for k, v in ratios.items()}

    def _resolve_market_type(self, raw_data: dict[str, pd.DataFrame]) -> str:
        """Resolve market type from contract metadata, fallback to futures."""
        contract_meta = raw_data.get("contract_metadata", pd.DataFrame())
        if not contract_meta.empty and "market_type" in contract_meta.columns:
            market_type = str(contract_meta.iloc[0]["market_type"]).lower()
            if market_type in self.MIN_KEY_FAMILY_AVAILABILITY:
                return market_type
        return "futures"

    def _key_family_availability(
        self,
        features: pd.DataFrame,
    ) -> dict[str, dict[str, float | dict[str, float]]]:
        """Compute key-family availability metrics used by degraded-run gating."""
        funding_ratio = self._binary_availability_ratio(features, "has_funding")
        oi_ratio = self._binary_availability_ratio(features, "has_oi")
        liq_ratio = self._binary_availability_ratio(features, "has_liqs")
        macro_ratio = self._binary_availability_ratio(features, "has_macro")
        real_liq_ratio = self._binary_availability_ratio(features, "has_real_liq_data")
        source_distribution = self._source_code_distribution(features, "liq_data_source_code")

        return {
            "funding": {"availability_ratio": funding_ratio},
            "open_interest": {"availability_ratio": oi_ratio},
            "liquidations": {
                "availability_ratio": liq_ratio,
                "real_ratio": real_liq_ratio,
                "source_code_distribution": source_distribution,
            },
            "macro": {"availability_ratio": macro_ratio},
        }

    def _degraded_run_assessment(
        self,
        *,
        market_type: str,
        key_family_availability: dict[str, dict[str, float | dict[str, float]]],
    ) -> tuple[bool, list[str], list[str], dict[str, float]]:
        """Assess whether dataset is degraded for the selected market type."""
        thresholds = self.MIN_KEY_FAMILY_AVAILABILITY.get(
            market_type,
            self.MIN_KEY_FAMILY_AVAILABILITY["futures"],
        )
        reasons: list[str] = []
        warnings: list[str] = []

        for family, threshold in thresholds.items():
            if threshold <= 0:
                continue
            family_payload = key_family_availability.get(family, {})
            ratio = float(family_payload.get("availability_ratio", 0.0))
            if ratio < threshold:
                reasons.append(
                    f"{family}_availability_below_threshold({ratio:.3f}<{threshold:.3f})"
                )

        liq_payload = key_family_availability.get("liquidations", {})
        liq_ratio = float(liq_payload.get("availability_ratio", 0.0))
        real_liq_ratio = float(liq_payload.get("real_ratio", 0.0))
        if liq_ratio >= thresholds.get("liquidations", 0.0) and real_liq_ratio <= 0.01:
            warnings.append(
                "liquidation_data_is_proxy_only(no_real_liquidation_coverage)"
            )

        degraded = len(reasons) > 0
        return degraded, reasons, warnings, thresholds

    def generate_quality_report(
        self,
        raw_data: dict[str, pd.DataFrame],
        dataset: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate data quality report for the built dataset."""
        index_stats = compute_index_gap_stats(dataset.index, freq=self.expected_ohlcv_freq)
        null_pct = (dataset.isna().mean() * 100).sort_values(ascending=False)

        coverage_windows: dict[str, Any] = {}
        for key in self.RAW_DATA_KEYS:
            df = raw_data.get(key, pd.DataFrame())
            coverage: dict[str, Any] = {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "is_empty": bool(df.empty),
            }
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                coverage["start"] = df.index.min().isoformat()
                coverage["end"] = df.index.max().isoformat()
            coverage_windows[key] = coverage

        report: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "timeframe": self.interval,
            "dataset_shape": [int(dataset.shape[0]), int(dataset.shape[1])],
            "index_start": dataset.index.min().isoformat(),
            "index_end": dataset.index.max().isoformat(),
            "index_gap_stats": index_stats.to_dict(),
            "duplicates": int(dataset.index.duplicated().sum()),
            "coverage_windows": coverage_windows,
            "null_percent_top_25": {
                str(k): float(v) for k, v in null_pct.head(25).items()
            },
            "rows_with_any_null_pct": float((dataset.isna().any(axis=1).mean() * 100)),
        }

        market_type = self._resolve_market_type(raw_data)
        key_family_availability = self._key_family_availability(dataset)
        degraded, reasons, warnings, thresholds = self._degraded_run_assessment(
            market_type=market_type,
            key_family_availability=key_family_availability,
        )
        report["key_family_availability"] = key_family_availability
        report["quality_gate"] = {
            "market_type": market_type,
            "thresholds": thresholds,
            "degraded": degraded,
            "reasons": reasons,
            "warnings": warnings,
        }
        report["data_degraded"] = degraded
        report["degradation_reasons"] = reasons
        report["quality_warnings"] = warnings
        return report

    def _normalized_data_map(
        self,
        data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Ensure all expected raw data keys exist."""
        normalized = {key: data.get(key, pd.DataFrame()) for key in self.RAW_DATA_KEYS}
        return normalized
    
    async def fetch_all_raw_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        save_raw: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch all raw data from sources.
        
        Args:
            start_date: Start date (default: from config)
            end_date: End date (default: now)
            save_raw: Whether to save raw data to disk
            
        Returns:
            Dict of DataFrames: ohlcv, funding_rate, open_interest, macro
        """
        start = start_date or self.settings.binance.start_date
        
        logger.info(
            "Fetching all data from %s to %s (interval=%s)",
            start,
            end_date or "now",
            self.interval,
        )
        
        # Fetch Binance data
        async with BinanceFetcher() as fetcher:
            binance_data = await fetcher.fetch_all(start, end_date, interval=self.interval)
            
            # Align funding rate to OHLCV index
            if not binance_data["funding_rate"].empty and not binance_data["ohlcv"].empty:
                binance_data["funding_rate_aligned"] = fetcher.align_funding_to_interval(
                    binance_data["funding_rate"],
                    binance_data["ohlcv"]
                )
        
        # Fetch macro data
        macro_daily = self.macro_fetcher.fetch_all_macro(
            self.settings.macro.start_date,
            end_date
        )
        
        # Align macro to OHLCV index
        if not binance_data["ohlcv"].empty:
            macro_aligned = self.macro_fetcher.align_to_interval(
                macro_daily,
                binance_data["ohlcv"].index
            )
            
            # Generate event flags
            event_flags = self.macro_fetcher.generate_event_flags(
                binance_data["ohlcv"].index
            )
        else:
            macro_aligned = pd.DataFrame()
            event_flags = pd.DataFrame()
        
        contract_metadata = self._build_contract_metadata_frame()

        # Liquidations: prefer real (if available), otherwise estimated.
        if not binance_data["ohlcv"].empty:
            real_liqs = await self.liq_collector.fetch_historical_from_github(start, end_date)
            if not real_liqs.empty:
                liquidations = self._normalize_liquidations(
                    real_liqs,
                    binance_data["ohlcv"].index,
                    source="real",
                )
            elif not binance_data["open_interest"].empty:
                liq_estimates = self.liq_collector.estimate_from_oi_changes(
                    binance_data["ohlcv"],
                    binance_data["open_interest"],
                )
                liquidations = self._normalize_liquidations(
                    liq_estimates,
                    binance_data["ohlcv"].index,
                    source="estimated",
                )
            else:
                liquidations = self._normalize_liquidations(
                    pd.DataFrame(),
                    binance_data["ohlcv"].index,
                    source="missing",
                )
        else:
            liquidations = pd.DataFrame()
        
        data = {
            "ohlcv": binance_data["ohlcv"],
            "funding_rate": binance_data.get("funding_rate_aligned", pd.DataFrame()),
            "open_interest": binance_data["open_interest"],
            "macro": macro_aligned,
            "event_flags": event_flags,
            "liquidations": liquidations,
            "contract_metadata": contract_metadata,
        }

        data = self._normalized_data_map(data)
        try:
            validate_raw_data_contracts(data)
        except DataContractError as exc:
            raise ValueError(f"Raw data contract validation failed: {exc}") from exc
        
        if save_raw:
            self._save_raw_data(data)
        
        return data
    
    def _save_raw_data(self, data: dict[str, pd.DataFrame]) -> None:
        """Save raw data to disk."""
        raw_dir = self.settings.paths.raw
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in data.items():
            if not df.empty:
                path = self._raw_data_save_path(name)
                df.to_parquet(path)
                logger.info(f"Saved {name}: {len(df)} rows to {path}")
    
    def load_raw_data(self) -> dict[str, pd.DataFrame]:
        """Load previously saved raw data."""
        data = {}
        for name in self.RAW_DATA_KEYS:
            loaded = False
            for path in self._raw_data_load_candidates(name):
                if not path.exists():
                    continue
                data[name] = pd.read_parquet(path)
                logger.info("Loaded %s: %s rows (%s)", name, len(data[name]), path)
                loaded = True
                break
            if not loaded:
                data[name] = pd.DataFrame()
                logger.warning("No raw data found for %s at interval %s", name, self.interval)
        
        normalized = self._normalized_data_map(data)
        return normalized
    
    def compute_features(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute all features from raw data.
        
        Features:
        - Price: returns, volatility, volume z-score
        - Perps: funding rate, OI changes, liquidation estimates
        - Macro: DXY/SPX returns, VIX, yield curve
        - Events: FOMC/CPI flags
        """
        ohlcv = data["ohlcv"]
        if ohlcv.empty:
            raise ValueError("OHLCV data is required")

        data = self._normalized_data_map(data)
        try:
            validate_raw_data_contracts(data)
        except DataContractError as exc:
            raise ValueError(f"Raw data contract validation failed: {exc}") from exc
        self._assert_ohlcv_integrity(ohlcv)
        
        features = pd.DataFrame(index=ohlcv.index)
        settings = self.settings.features
        
        # ===== Price features =====
        logger.info("Computing price features...")
        
        close = ohlcv["close"]
        
        # Returns at various windows
        for w in settings.return_windows:
            features[f"return_{w}"] = np.log(close / close.shift(w))
        
        # Realized volatility
        returns_1 = np.log(close / close.shift(1))
        for w in settings.vol_windows:
            features[f"realized_vol_{w}"] = returns_1.rolling(window=w).std()
        
        # ATR-like range feature
        high, low = ohlcv["high"], ohlcv["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        features["atr_6"] = tr.rolling(window=6).mean()
        
        # Volume z-score
        volume = ohlcv["volume"]
        vol_mean = volume.rolling(window=settings.volume_zscore_window).mean()
        vol_std = volume.rolling(window=settings.volume_zscore_window).std()
        features["volume_zscore"] = (volume - vol_mean) / (vol_std + 1e-8)
        
        # ===== Funding rate features =====
        logger.info("Computing funding rate features...")
        
        funding = data.get("funding_rate", pd.DataFrame())
        if not funding.empty and "funding_rate" in funding.columns:
            # Align to OHLCV index if not already
            funding_aligned = funding.reindex(ohlcv.index, method="ffill")
            if not funding_aligned.index.equals(ohlcv.index):
                raise ValueError("Funding alignment failed to match OHLCV index")
            features["funding_rate"] = funding_aligned["funding_rate"]
            features["funding_rate_ma_6"] = features["funding_rate"].rolling(
                window=settings.funding_ma_window
            ).mean()
        else:
            features["funding_rate"] = np.nan
            features["funding_rate_ma_6"] = np.nan
        
        # Track missingness
        features["has_funding"] = features["funding_rate"].notna().astype(int)
        
        # ===== Open interest features =====
        logger.info("Computing open interest features...")
        
        oi = data.get("open_interest", pd.DataFrame())
        if not oi.empty and "open_interest_value" in oi.columns:
            # Align to OHLCV index
            oi_aligned = oi.reindex(ohlcv.index, method="ffill")
            if not oi_aligned.index.equals(ohlcv.index):
                raise ValueError("Open interest alignment failed to match OHLCV index")
            features["open_interest"] = oi_aligned["open_interest_value"]
            
            for w in settings.oi_change_windows:
                features[f"oi_change_pct_{w}"] = features["open_interest"].pct_change(w)
        else:
            features["open_interest"] = np.nan
            for w in settings.oi_change_windows:
                features[f"oi_change_pct_{w}"] = np.nan
        
        features["has_oi"] = features["open_interest"].notna().astype(int)
        
        # ===== Liquidation features =====
        logger.info("Computing liquidation features...")
        
        liqs = data.get("liquidations", pd.DataFrame())
        if not liqs.empty:
            liq_aligned = liqs.reindex(ohlcv.index, method="ffill")
            if not liq_aligned.index.equals(ohlcv.index):
                raise ValueError("Liquidation alignment failed to match OHLCV index")
            for col in ["long_liq_usd_est", "short_liq_usd_est", "liq_imbalance_est"]:
                features[col] = liq_aligned[col] if col in liq_aligned.columns else np.nan
            default_real_liq = pd.Series(0, index=ohlcv.index, dtype=int)
            # Non-empty legacy liquidation snapshots without provenance are treated as estimated.
            default_source_code = pd.Series(0, index=ohlcv.index, dtype=int)
            features["has_real_liq_data"] = (
                liq_aligned["has_real_liq_data"]
                if "has_real_liq_data" in liq_aligned.columns
                else default_real_liq
            ).fillna(0).astype(int)
            features["liq_data_source_code"] = (
                liq_aligned["liq_data_source_code"]
                if "liq_data_source_code" in liq_aligned.columns
                else default_source_code
            )
            features["liq_data_source_code"] = features["liq_data_source_code"].fillna(-1).astype(int)
        else:
            features["long_liq_usd_est"] = np.nan
            features["short_liq_usd_est"] = np.nan
            features["liq_imbalance_est"] = np.nan
            features["has_real_liq_data"] = 0
            features["liq_data_source_code"] = -1
        
        features["has_liqs"] = (features["liq_data_source_code"] >= 0).astype(int)
        
        # ===== Macro features =====
        logger.info("Computing macro features...")
        
        macro = data.get("macro", pd.DataFrame())
        if not macro.empty:
            macro_aligned = macro.reindex(ohlcv.index, method="ffill")
            if not macro_aligned.index.equals(ohlcv.index):
                raise ValueError("Macro alignment failed to match OHLCV index")
            
            for col in ["dxy_return_1d", "spx_return_1d", "vix", "yield_curve_2_10"]:
                if col in macro_aligned.columns:
                    features[col] = macro_aligned[col]
        
        features["has_macro"] = features.get("vix", pd.Series(np.nan, index=ohlcv.index)).notna().astype(int)
        
        # ===== Event flags =====
        logger.info("Adding event flags...")
        
        events = data.get("event_flags", pd.DataFrame())
        if not events.empty:
            events_aligned = events.reindex(ohlcv.index, method="ffill").fillna(0)
            if not events_aligned.index.equals(ohlcv.index):
                raise ValueError("Event flag alignment failed to match OHLCV index")
            for col in events_aligned.columns:
                features[col] = events_aligned[col]

        self._assert_data_availability(features)
        
        logger.info(f"Computed {len(features.columns)} features")
        return features
    
    def build_dataset(
        self,
        data: dict[str, pd.DataFrame] | None = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Build complete dataset with features and labels.
        
        Args:
            data: Raw data dict (will load from disk if None)
            save: Whether to save processed dataset
            
        Returns:
            DataFrame with features and labels
        """
        if data is None:
            data = self.load_raw_data()
        data = self._normalized_data_map(data)
        try:
            validate_raw_data_contracts(data)
        except DataContractError as exc:
            raise ValueError(f"Raw data contract validation failed: {exc}") from exc
        
        ohlcv = data["ohlcv"]
        if ohlcv.empty:
            raise ValueError("OHLCV data is required")
        
        # Compute features
        features = self.compute_features(data)
        
        # Compute labels
        labels = self.label_generator.generate_all_labels(ohlcv)
        
        # Combine
        dataset = pd.concat([ohlcv, features, labels], axis=1)
        
        # Validate no leakage
        validation = self.label_generator.validate_no_leakage(features, labels)
        if not validation["passed"]:
            raise ValueError(f"Leakage detected: {validation['errors']}")
        
        for warning in validation["warnings"]:
            logger.warning(warning)

        quality_report = self.generate_quality_report(data, dataset)
        
        # Save
        if save:
            processed_dir = self.settings.paths.processed
            processed_dir.mkdir(parents=True, exist_ok=True)
            artifact_paths = self.processed_artifact_paths()
            dataset_path = artifact_paths["dataset"]
            dataset.to_parquet(dataset_path)
            logger.info("Saved dataset: %s rows to %s", len(dataset), dataset_path)
            
            # Save metadata
            meta = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "timeframe": self.interval,
                "date_range": [
                    dataset.index.min().isoformat(),
                    dataset.index.max().isoformat(),
                ],
                "n_rows": len(dataset),
                "n_features": len(features.columns),
                "n_labels": len(labels.columns),
                "feature_columns": list(features.columns),
                "label_columns": list(labels.columns),
                "contract_metadata": (
                    data["contract_metadata"].iloc[0].to_dict()
                    if not data["contract_metadata"].empty
                    else {}
                ),
                "quality_report_file": artifact_paths["quality"].name,
            }
            
            with open(artifact_paths["metadata"], "w") as f:
                json.dump(meta, f, indent=2)

            with open(artifact_paths["quality"], "w") as f:
                json.dump(quality_report, f, indent=2)
            logger.info("Saved quality report to %s", artifact_paths["quality"])
        
        return dataset
    
    def get_train_test_split(
        self,
        dataset: pd.DataFrame,
        train_start: str,
        train_end: str,
        test_end: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train/test split with proper time boundaries.
        
        Args:
            dataset: Full dataset
            train_start: Training start date
            train_end: Training end date (exclusive for test start)
            test_end: Test end date
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train = dataset.loc[train_start:train_end]
        test = dataset.loc[train_end:test_end].iloc[1:]  # Exclude first row (is train end)
        
        # Remove rows with NaN target
        train = train[train["forward_return"].notna()]
        test = test[test["forward_return"].notna()]
        
        logger.info(f"Train: {len(train)} samples, Test: {len(test)} samples")
        return train, test


async def main():
    """Build dataset from scratch."""
    logging.basicConfig(level=logging.INFO)
    
    builder = DatasetBuilder()
    
    # Fetch fresh data (last 90 days for quick test)
    from datetime import timedelta
    start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    
    print(f"Fetching data from {start}...")
    data = await builder.fetch_all_raw_data(start)
    
    print("\nBuilding dataset...")
    dataset = builder.build_dataset(data)
    
    print(f"\nDataset shape: {dataset.shape}")
    print("\nColumns:")
    for col in dataset.columns:
        null_pct = dataset[col].isna().mean() * 100
        print(f"  {col}: {null_pct:.1f}% null")
    
    print("\nSample:")
    print(dataset[["close", "return_1", "funding_rate", "forward_return", "regime"]].tail(10))


if __name__ == "__main__":
    asyncio.run(main())
