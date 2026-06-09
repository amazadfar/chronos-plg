# 01 - System Overview And Current State

## What The System Is

`chronos-plg` is a phase-gated, cost-aware trading research and paper-trading system for BTC perpetual/spot/margin scenarios.

Core objective:

- Prove `ProfitFactorNet > 1.0` after fees, slippage, funding, interest, and other modeled costs
- Keep supporting safety gates (Sharpe, drawdown, robustness, regime stability)

## What Is Implemented Right Now

The codebase has a full implementation from Phase 0 through Phase 10, including:

- Exchange/market scenario cost profiles (`binance`, `kucoin`; `spot`, `margin`, `futures`)
- Contract-validated data pipeline with feature/label generation and leakage checks
- Baselines (`RandomWalk`, `EWMA`, `LightGBM`) and candidate models (`Chronos2`, `MetaModel`)
- Strategy stack (quantile signals, regime gating, position sizing, execution-intent layer)
- Execution-event cost engine with transition-leg decomposition and funding/interest accounting
- Walk-forward backtest engine with fold freezing and per-fold audit artifacts
- Robustness framework (cost stress, regime exclusion, adverse window, block bootstrap, rolling stability, parameter sensitivity)
- Phase 9 decision reporting (GO/ITERATE/NO_GO + uncertainty bands)
- Phase 10 paper-trading replay, monitoring dashboards, kill switch, deployment readiness, and capital ramp policy

## Plan Status Snapshot

Source: `CODEBASE-PLAN-V-0.1.md`

- Phases 0-10: marked complete
- V0.1 objective: not yet achieved (secondary criteria/readiness still not met)

## Dataset State Snapshot

Source: `data/processed/btc_4h_metadata.json`, `data/processed/btc_4h_quality.json`

- Dataset: `data/processed/btc_4h.parquet`
- Range: 2024-01-01 00:00:00 UTC to 2026-02-17 16:00:00 UTC
- Rows: 4,673
- Columns: 49 (30 features + 12 labels + OHLCV and metadata-linked fields)
- Index integrity: no gaps, no duplicates

Current coverage realities:

- Funding and macro/event data are available
- Open interest was unavailable in latest processed build (`open_interest` family is 100% null)
- Liquidation features exist structurally but are null in this build because OI-based estimation path had no OI source

## Current Empirical Performance Snapshot (Phase 10)

Source: `data/results/phase10_real_20260217/*_paper_phase10_summary.json`

| Run                                              | PF Net |  Sharpe | Trades | Total Return | Decision |
|--------------------------------------------------|-------:|--------:|-------:|-------------:|----------|
| `ewma_candidate_sharpe565_retrain42_entry178_v4` | 1.1739 |  0.5653 |     76 |        1.05% | ITERATE  |
| `ewma_candidate_span72_entry178_retrain28_v3`    | 1.1213 |  0.4512 |    100 |        0.93% | ITERATE  |
| `ewma`                                           | 1.0521 |  0.1001 |     26 |        0.09% | NO_GO    |
| `lightgbm`                                       | 0.2239 | -2.4337 |     36 |       -1.35% | NO_GO    |

Interpretation of current state:

- There is a measurable net edge region (`PF Net > 1`) in tuned EWMA variants
- Deployment readiness is still failing due trade-count and kill-switch/readiness policy constraints
- The highest-Sharpe candidate has insufficient trade count for current readiness policy

## Where The System Stands Right Now

- Engineering foundation is robust and phase-complete through paper-trading governance
- Research objective is partially met (edge exists), but operational go-live criteria are not met
- Immediate next leverage is in improving trade density and regime stability without collapsing PF/Sharpe under current cost assumptions
