# 03 - Module And Script Inventory

## Configuration Layer (`config/`)

| File                           | Responsibility                                                                                                      | Key Objects                                                        |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| `config/settings.py`           | Central defaults for data paths, exchange config, feature/target settings, walk-forward, strategy, model, and costs | `Settings`, `WalkForwardConfig`, `StrategyConfig`, `ModelConfig`   |
| `config/cost_profiles.py`      | Canonical fee schedules for Binance/KuCoin across spot/margin/futures with discount toggles                         | `ExchangeCostProfile`, `MarketFeeProfile`, `get_cost_profile()`    |
| `config/scenario_profiles.py`  | Named benchmark scenarios mapping exchange + market type + order type + funding/interest flags                      | `TradingScenarioProfile`, `get_scenario_profile()`                 |
| `config/baseline_protocols.py` | Immutable Phase-6 protocol definitions with fingerprinting for reproducibility                                      | `BaselineProtocol`, `BaselineModelSpec`, `get_baseline_protocol()` |

## CLI Scripts (`scripts/`)

| Script                         | What It Does                                                                              | Main Outputs                                                                    |
|--------------------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `scripts/download_data.py`     | Fetches OHLCV/funding/OI/macro/event/liquidation raw inputs and optionally builds dataset | `data/raw/*.parquet`, optional processed dataset, run manifest                  |
| `scripts/build_features.py`    | Builds processed dataset from cached raw files                                            | `data/processed/btc_4h.parquet`, metadata JSON                                  |
| `scripts/run_baselines.py`     | Runs frozen Phase-6 baseline protocol with frozen folds and scenario cost model           | model reports, fold metrics, leaderboards, baseline gate artifact               |
| `scripts/run_chronos2.py`      | Runs Phase-7 candidate validation after leakage and baseline gates                        | candidate reports, candidate gates, recent-regime metrics, combined leaderboard |
| `scripts/run_backtest.py`      | General multi-model backtest CLI with scenario selection                                  | per-model reports, comparison outputs, phase-9 decisions                        |
| `scripts/run_paper_trading.py` | Phase-10 paper replay with monitoring and deployment policy evaluation                    | paper logs, returns, dashboards, kill events, readiness, ramp decision          |
| `scripts/benchmark.py`         | Extended benchmark + visuals + robustness/decision artifacts                              | plots, benchmark markdown report, robustness + decision outputs                 |
| `scripts/smoke_check.py`       | Fast synthetic end-to-end health check of walk-forward pipeline                           | smoke summary + run manifest                                                    |

## Source Modules (`src/`)

### Data (`src/data/`)

| File                                | Responsibility                                                            | Key Mechanisms                                                                           |
|-------------------------------------|---------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `src/data/contracts.py`             | Enforces raw dataset contracts and datetime-index constraints             | required-column validation, timezone/monotonic/duplicate checks, gap stats               |
| `src/data/binance_fetcher.py`       | Async Binance futures fetcher for OHLCV/funding/OI                        | paginated REST pulls, rate limiting, 8h funding alignment to 4h                          |
| `src/data/macro_fetcher.py`         | Macro fetch + event flag generation                                       | yfinance pulls, 1-day lag alignment to 4h, FOMC/CPI flag windows                         |
| `src/data/liquidation_collector.py` | Real-time liquidation stream + fallback OI-based liquidation estimation   | forceOrder parsing, 4h aggregation, proxy estimation logic                               |
| `src/data/market_metadata.py`       | Static exchange contract metadata (tick/lot/min constraints)              | `get_contract_metadata()` lookup                                                         |
| `src/data/labels.py`                | Label generation + leakage validation                                     | forward return/RV labels, historical quantiles, regime labels, shifted-target leak traps |
| `src/data/build_dataset.py`         | End-to-end raw merge, feature engineering, quality checks, save artifacts | strict alignment, availability flags, provenance flags, quality report generation        |

### Models (`src/models/`)

| File                                        | Responsibility                                     | Key Mechanisms                                                                                                |
|---------------------------------------------|----------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `src/models/baselines/random_walk.py`       | Naive quantile baseline                            | rolling historical quantiles, fixed q50=0                                                                     |
| `src/models/baselines/ewma.py`              | EWMA and AR(1) baselines                           | exponentially weighted mean/std quantile mapping, optional AR fit                                             |
| `src/models/baselines/lightgbm_quantile.py` | Tabular quantile baseline                          | separate LightGBM quantile models with early stopping                                                         |
| `src/models/chronos2_runner.py`             | Chronos-style forecaster with strict OOS semantics | sequential prediction, q50 self-feedback, optional covariate shift adjustment, deterministic fallback backend |
| `src/models/meta_model.py`                  | Two-stage Chronos+LightGBM stacker                 | OOF Chronos generation for stage-2 training, final Chronos fit on full train                                  |

### Strategy (`src/strategy/`)

| File                               | Responsibility                                                 | Key Mechanisms                                                                                          |
|------------------------------------|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `src/strategy/signals.py`          | Converts quantile forecasts to trade decisions                 | long/short/no-edge logic, uncertainty gate, regime entry multipliers                                    |
| `src/strategy/regime_detector.py`  | Classifies market regime and size multipliers                  | trend/chop/panic/normal logic from 7-day return and vol                                                 |
| `src/strategy/position_sizing.py`  | Risk-aware position sizing                                     | vol targeting, leverage caps by market, short gating, turnover caps, precision/min-notional enforcement |
| `src/strategy/execution_intent.py` | Maps target transitions to execution intent                    | transition classification (`open/increase/reduce/close/reverse`), policy-driven order type              |
| `src/strategy/strategy.py`         | Integrates model + signal + regime + sizing + risk constraints | scenario-level exposure/turnover/cooldown controls and execution-intent columns                         |

### Backtest and Evaluation (`src/backtest/`, `src/evaluation/`)

| File                                 | Responsibility                               | Key Mechanisms                                                                       |
|--------------------------------------|----------------------------------------------|--------------------------------------------------------------------------------------|
| `src/backtest/costs.py`              | Execution-event cost model                   | transition-leg decomposition, fees/slippage/funding/interest/other cost components   |
| `src/backtest/engine.py`             | Walk-forward backtest runtime                | fold-wise train/predict/position/cost/equity computation with per-fold metrics       |
| `src/backtest/report.py`             | Reporting and model comparison               | kill-criteria display integration, PF-first model comparison                         |
| `src/evaluation/walk_forward.py`     | Leak-safe walk-forward harness               | fold generation, feature lagging, fold contamination guards, fold artifact snapshots |
| `src/evaluation/metrics.py`          | Forecast + trading metric functions          | pinball loss, coverage/calibration, Sharpe/Sortino/max DD/PF                         |
| `src/evaluation/phase6_baselines.py` | Protocol freeze/leaderboard/gating utilities | baseline reproducibility and advancement gate payloads                               |
| `src/evaluation/phase7_chronos.py`   | Phase-7 candidate gate logic                 | recent-regime split metrics and candidate-vs-anchor checks                           |

### Robustness, Reporting, Paper Trading

| File                              | Responsibility                 | Key Mechanisms                                                                                                 |
|-----------------------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------|
| `src/robustness/kill_criteria.py` | Shared kill-criteria engine    | PF-net first + Sharpe/DD/regime/decay/win-rate advisory checks                                                 |
| `src/robustness/stress_tests.py`  | Stress suite                   | cost stress grid, regime exclusion, adverse windows, block bootstrap, rolling stability, parameter sensitivity |
| `src/robustness/summary.py`       | Robustness report generator    | kill + stress aggregation and viability verdicting                                                             |
| `src/reporting/decision.py`       | Phase-9 decision framework     | GO/ITERATE/NO_GO logic and fold/bootstrap uncertainty bands                                                    |
| `src/paper_trading/engine.py`     | Sequential paper replay engine | strict past-only retraining loop, return/cost audit log, backtest-compatible metrics                           |
| `src/paper_trading/monitoring.py` | Monitoring dashboards          | daily/weekly PF/Sharpe/DD/turnover/cost decomposition windows                                                  |
| `src/paper_trading/policy.py`     | Operational policy layer       | kill switch triggers, deployment readiness checks, staged capital ramp/rollback                                |

### Common Utilities

| File                      | Responsibility                                                      |
|---------------------------|---------------------------------------------------------------------|
| `src/common/metrics.py`   | Canonical metric names, threshold defaults, PF/Sharpe helpers       |
| `src/utils/experiment.py` | Run manifests, seed handling, script-level reproducibility metadata |

## Test Suite Coverage (`tests/`)

The test suite enforces key invariants by phase:

- Phase 0/1: scenario/cost profile correctness, manifest tooling
- Phase 2/3: data contracts, leakage traps, fold boundary integrity
- Phase 4/5: transition-leg cost accounting, strategy constraints
- Phase 6/7: protocol freeze integrity, candidate gate behavior
- Phase 8/9: stress protocols, decision semantics, uncertainty bands
- Phase 10: paper replay cost audit columns, kill/readiness/ramp policy behavior
