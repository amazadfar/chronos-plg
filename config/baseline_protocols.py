"""Immutable baseline evaluation protocols for reproducible comparability."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from config.scenario_profiles import DEFAULT_SCENARIO
from config.settings import WalkForwardConfig
from src.common.metrics import DEFAULT_SUCCESS_THRESHOLDS

ModelKey = Literal["random_walk", "ewma", "lightgbm"]


@dataclass(frozen=True)
class BaselineModelSpec:
    """Immutable model specification within a baseline protocol."""

    key: ModelKey
    name: str
    kwargs: tuple[tuple[str, Any], ...] = ()

    def kwargs_dict(self) -> dict[str, Any]:
        return dict(self.kwargs)


@dataclass(frozen=True)
class BaselineProtocol:
    """Frozen baseline protocol used for Phase 6 evaluation."""

    name: str
    description: str
    scenario: str = DEFAULT_SCENARIO
    mode: Literal["weekly", "monthly"] = "weekly"
    train_window_days: int = 180
    test_window_days: int = 7
    step_size_days: int = 7
    min_train_samples: int = 500
    start_date: str | None = None
    feature_lag_candles: int = 1
    baseline_anchor_model: str = "LightGBM"
    models: tuple[BaselineModelSpec, ...] = ()

    def walk_forward_config(self) -> WalkForwardConfig:
        """Materialize walk-forward config from immutable protocol fields."""
        return WalkForwardConfig(
            train_window_days=self.train_window_days,
            test_window_days=self.test_window_days,
            step_size_days=self.step_size_days,
            min_train_samples=self.min_train_samples,
            mode=self.mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "scenario": self.scenario,
            "mode": self.mode,
            "train_window_days": self.train_window_days,
            "test_window_days": self.test_window_days,
            "step_size_days": self.step_size_days,
            "min_train_samples": self.min_train_samples,
            "start_date": self.start_date,
            "feature_lag_candles": self.feature_lag_candles,
            "baseline_anchor_model": self.baseline_anchor_model,
            "models": [
                {
                    "key": model.key,
                    "name": model.name,
                    "kwargs": model.kwargs_dict(),
                }
                for model in self.models
            ],
            "thresholds": {
                "min_profit_factor_net": DEFAULT_SUCCESS_THRESHOLDS.min_profit_factor_net,
                "min_sharpe_net": DEFAULT_SUCCESS_THRESHOLDS.min_sharpe_net,
                "min_baseline_sharpe_delta": DEFAULT_SUCCESS_THRESHOLDS.min_baseline_sharpe_delta,
            },
        }

    def fingerprint(self) -> str:
        """Stable hash fingerprint for protocol freeze auditing."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


DEFAULT_BASELINE_PROTOCOL = "phase6_v0_1_weekly_costaware"


BASELINE_PROTOCOLS: dict[str, BaselineProtocol] = {
    DEFAULT_BASELINE_PROTOCOL: BaselineProtocol(
        name=DEFAULT_BASELINE_PROTOCOL,
        description=(
            "Phase 6 frozen baseline protocol using weekly walk-forward retraining and "
            "net-cost execution assumptions."
        ),
        scenario=DEFAULT_SCENARIO,
        mode="weekly",
        train_window_days=180,
        test_window_days=7,
        step_size_days=7,
        min_train_samples=500,
        start_date=None,
        feature_lag_candles=1,
        baseline_anchor_model="LightGBM",
        models=(
            BaselineModelSpec(
                key="random_walk",
                name="RandomWalk",
                kwargs=(("lookback_window", 252),),
            ),
            BaselineModelSpec(
                key="ewma",
                name="EWMA",
                kwargs=(("span", 24),),
            ),
            BaselineModelSpec(
                key="lightgbm",
                name="LightGBM",
                kwargs=(
                    ("n_estimators", 300),
                    ("early_stopping_rounds", 30),
                ),
            ),
        ),
    ),
}


def get_baseline_protocol(name: str = DEFAULT_BASELINE_PROTOCOL) -> BaselineProtocol:
    """Get immutable baseline protocol by name."""
    key = name.lower()
    if key not in BASELINE_PROTOCOLS:
        available = ", ".join(sorted(BASELINE_PROTOCOLS))
        raise ValueError(f"Unknown baseline protocol '{name}'. Available: {available}")
    return BASELINE_PROTOCOLS[key]
