#!/usr/bin/env python
"""Fast smoke check for core pipeline wiring."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import WalkForwardConfig, get_settings
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.models.baselines.random_walk import RandomWalkBaseline
from src.utils import finalize_experiment_run, set_global_seed, start_experiment_run


def _synthetic_dataset(seed: int, n: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    returns = rng.normal(0, 0.015, size=n)
    close = 40000 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "close": close,
            "return_1": returns,
            "return_6": pd.Series(returns).rolling(6).sum().values,
            "realized_vol_6": pd.Series(returns).rolling(6).std().values,
            "forward_return": np.roll(returns, -1),
            "regime": np.where(np.abs(returns) > 0.025, "trend", "normal"),
        },
        index=dates,
    )
    data.loc[data.index[-1], "forward_return"] = np.nan
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fast smoke checks")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Directory for run manifest and smoke artifacts",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
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
        settings = get_settings()
        logger.info("Settings loaded. Data root: %s", settings.paths.root)

        data = _synthetic_dataset(args.seed)
        wf_cfg = WalkForwardConfig(
            train_window_days=20,
            test_window_days=3,
            step_size_days=7,
            min_train_samples=60,
        )
        evaluator = WalkForwardEvaluator(config=wf_cfg)
        results = evaluator.evaluate_model(
            RandomWalkBaseline,
            data,
            model_kwargs={"lookback_window": 40},
            show_progress=False,
        )
        if len(results.folds) == 0:
            raise RuntimeError("Smoke check failed: no folds generated")

        summary_path = output_dir / "smoke_check_last.json"
        summary_path.write_text(
            (
                "{\n"
                f'  "folds": {len(results.folds)},\n'
                f'  "mean_sharpe": {results.mean_sharpe:.6f},\n'
                f'  "q50_pinball": {results.mean_pinball_loss.get("q50", 0.0):.6f}\n'
                "}\n"
            ),
            encoding="utf-8",
        )
        artifacts.append(str(summary_path))
        logger.info("Smoke check passed. Folds=%s", len(results.folds))
        return 0
    except Exception as exc:
        status = "failed"
        error = str(exc)
        logger.exception("Smoke check failed")
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
