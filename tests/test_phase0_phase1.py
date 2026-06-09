"""Tests for Phase 0/1 shared definitions and runtime tooling."""

from pathlib import Path

import pandas as pd

from config.cost_profiles import get_cost_profile
from config.scenario_profiles import DEFAULT_SCENARIO, get_scenario_profile
from src.common.metrics import (
    DEFAULT_SUCCESS_THRESHOLDS,
    profit_factor_from_returns,
    recent_vs_early_sharpe_ratio,
)
from src.utils.experiment import finalize_experiment_run, start_experiment_run


def test_binance_futures_discounted_taker_fee():
    profile = get_cost_profile("binance")
    effective = profile.fee_rate("futures", "taker", use_discount=True)
    assert abs(effective - 0.00045) < 1e-12


def test_kucoin_spot_discounted_taker_fee():
    profile = get_cost_profile("kucoin")
    effective = profile.fee_rate("spot", "taker", use_discount=True)
    assert abs(effective - 0.0008) < 1e-12


def test_default_scenario_is_valid():
    scenario = get_scenario_profile(DEFAULT_SCENARIO)
    assert scenario.exchange in {"binance", "kucoin"}
    assert scenario.order_type in {"maker", "taker"}


def test_profit_factor_and_decay_helpers():
    returns = pd.Series([0.02, -0.01, 0.03, -0.01])
    pf = profit_factor_from_returns(returns)
    assert pf > 1.0

    ratio = recent_vs_early_sharpe_ratio(pd.Series([0.01] * 100))
    assert ratio >= DEFAULT_SUCCESS_THRESHOLDS.min_recent_sharpe_ratio


def test_run_manifest_start_and_finalize(tmp_path: Path):
    run_id, manifest_path = start_experiment_run(
        script_name="test_script.py",
        args={"k": "v"},
        seed=42,
        output_dir=tmp_path,
        project_root=tmp_path,
    )
    assert manifest_path.exists()

    finalize_experiment_run(
        manifest_path=manifest_path,
        run_id=run_id,
        status="success",
        artifacts=["a.json"],
        notes={"msg": "ok"},
    )

    content = manifest_path.read_text(encoding="utf-8")
    assert run_id in content
    assert '"status": "success"' in content
