#!/usr/bin/env python
"""
Download all data for the Chronos-2 trading system.

Usage:
    python scripts/download_data.py [--start-date 2021-01-01] [--end-date 2024-01-01]
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.timeframe import SUPPORTED_TIMEFRAMES, normalize_timeframe
from src.data.build_dataset import DatasetBuilder
from src.utils import finalize_experiment_run, set_global_seed, start_experiment_run


async def main() -> int:
    parser = argparse.ArgumentParser(description="Download data for Chronos-2 trading system")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2021-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: now)"
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Also build the processed dataset"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="4h",
        choices=SUPPORTED_TIMEFRAMES,
        help="Timeframe interval for downloaded OHLCV/OI and built dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Directory to write run manifest and metadata",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    output_dir = Path(args.output_dir)
    set_global_seed(args.seed)
    run_id, manifest_path = start_experiment_run(
        script_name=Path(__file__).name,
        args=vars(args),
        seed=args.seed,
        output_dir=output_dir,
        project_root=Path(__file__).parent.parent,
    )
    status = "success"
    error: str | None = None
    artifacts: list[str] = []
    
    try:
        interval = normalize_timeframe(args.interval)
        logger.info(
            "Starting data download from %s to %s (interval=%s)",
            args.start_date,
            args.end_date or "now",
            interval,
        )
        builder = DatasetBuilder(interval=interval)

        # Fetch raw data
        data = await builder.fetch_all_raw_data(
            start_date=args.start_date,
            end_date=args.end_date,
            save_raw=True
        )

        # Summary
        logger.info("\n=== Download Summary ===")
        for name, df in data.items():
            if df.empty:
                logger.warning(f"  {name}: No data")
            else:
                logger.info(f"  {name}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
                artifacts.append(str(builder.raw_artifact_path(name)))

        # Optionally build dataset
        if args.build_dataset:
            logger.info("\nBuilding processed dataset...")
            dataset = builder.build_dataset(data, save=True)
            logger.info(f"Dataset: {dataset.shape[0]} rows, {dataset.shape[1]} columns")

            # Print column summary
            logger.info("\n=== Column Summary ===")
            for col in dataset.columns:
                null_pct = dataset[col].isna().mean() * 100
                logger.info(f"  {col}: {null_pct:.1f}% null")
            processed_artifacts = builder.processed_artifact_paths()
            artifacts.extend(
                [
                    str(processed_artifacts["dataset"]),
                    str(processed_artifacts["metadata"]),
                    str(processed_artifacts["quality"]),
                ]
            )

        logger.info("\nDone!")
        return 0
    except Exception as exc:
        status = "failed"
        error = str(exc)
        logger.exception("Data download failed")
        return 1
    finally:
        finalize_experiment_run(
            manifest_path=manifest_path,
            run_id=run_id,
            status=status,
            artifacts=artifacts,
            notes={"error": error} if error else None,
        )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
