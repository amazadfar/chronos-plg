#!/usr/bin/env python
"""Build processed dataset/features from locally cached raw data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.timeframe import SUPPORTED_TIMEFRAMES, normalize_timeframe
from src.data.build_dataset import DatasetBuilder
from src.utils import finalize_experiment_run, set_global_seed, start_experiment_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Build processed dataset from raw cache")
    parser.add_argument(
        "--interval",
        type=str,
        default="4h",
        choices=SUPPORTED_TIMEFRAMES,
        help="Dataset timeframe interval",
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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
        builder = DatasetBuilder(interval=interval)
        dataset = builder.build_dataset(data=None, save=True)
        logger.info("Built dataset: %s rows x %s cols", dataset.shape[0], dataset.shape[1])
        artifact_paths = builder.processed_artifact_paths()
        artifacts.extend(
            [
                str(artifact_paths["dataset"]),
                str(artifact_paths["metadata"]),
                str(artifact_paths["quality"]),
            ]
        )
        return 0
    except Exception as exc:
        status = "failed"
        error = str(exc)
        logger.exception("Failed to build dataset from raw cache")
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
    raise SystemExit(main())
