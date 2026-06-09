# CODEBASE PLAN V0.1

This plan defines the implementation phases we will follow to build and validate the trading system with realistic execution economics. The primary objective is to prove positive net expectancy after all relevant costs and financing.

## Plan Controls

- Owner: Core quant/dev team
- Working mode: Phase-gated, checklist-driven
- Update rule: Mark each action as done only after code, tests, and artifacts are committed
- Tracking: This document is the single source of truth for implementation progress

## Primary Success and Kill Criteria

- Primary success criterion: `ProfitFactorNet > 1.00`
- Profit factor definition: `sum(all net winning PnL) / abs(sum(all net losing PnL))`
- Net PnL definition: Gross PnL minus all fees, slippage, funding, borrow/interest, and other exchange/account-level charges included in the scenario
- Secondary success criteria:
[ ] Net Sharpe > 0.50
[ ] Beats LightGBM baseline by Sharpe delta >= 0.10
[ ] Regime stability CV < 1.5
[ ] No severe decay in recent period

## Locked Cost/Execution Assumptions (V0.1)

[ ] Model both exchanges and market types: Binance + KuCoin, Spot + Margin + Futures
[ ] Default execution style for benchmark: Taker
[ ] Binance Spot/Margin base fees: 0.10% maker / 0.10% taker
[ ] Binance Spot/Margin discount with BNB enabled: 25% (effective 0.075% / 0.075%)
[ ] Binance Futures base fees: 0.02% maker / 0.05% taker
[ ] Binance Futures BNB fee discount: 10% (effective 0.018% / 0.045%)
[ ] KuCoin Spot/Margin base fees: 0.10% maker / 0.10% taker
[ ] KuCoin Spot/Margin KCS discount: 20% (effective 0.08% / 0.08%)
[ ] KuCoin Futures fees: 0.02% maker / 0.06% taker (no discount)
[ ] Margin borrow/interest modeled explicitly by asset, rate, and holding time
[ ] Funding modeled explicitly for perpetual futures with correct sign for long/short positions
[ ] Any additional exchange charges supported via pluggable fee events (withdrawal/transfer not included in trade-level backtest unless scenario enables them)

## Phase 0 - Baseline Governance and Definitions

[x] Create shared metric definitions module and enforce naming consistency across `src/evaluation`, `src/backtest`, and `src/robustness`
[x] Standardize primary kill/success logic around `ProfitFactorNet` and align thresholds in docs and code
[x] Add `config/cost_profiles.py` (or equivalent) with explicit exchange/market/discount profiles
[x] Add `config/scenario_profiles.py` for benchmark scenarios (Binance futures taker default, KuCoin futures taker, spot/margin variants)
[x] Add plan status section at bottom of this file for phase completion tracking

## Phase 1 - Codebase Integrity and Tooling Hardening

[x] Fix packaging/script drift in `pyproject.toml` (remove or implement missing script targets)
[x] Enforce lint baseline and decide strictness gate for CI (`ruff`, `pytest`)
[x] Add deterministic seed and artifact metadata handling in all CLI entry points (`scripts/*.py`)
[x] Add centralized experiment ID and run manifest (`data/results/run_manifest.json`)
[x] Add fast smoke command for full pipeline health check

## Phase 2 - Data Pipeline Hardening

[x] Refactor data contracts and schemas for OHLCV, funding, OI, macro, events, and liquidation features
[x] Add strict timestamp alignment and data-availability assertions in `src/data/build_dataset.py`
[x] Add exchange-specific symbol and contract metadata support (tick size, lot size, quote asset)
[x] Keep liquidation/OI path functional with current free sources and mark feature provenance (real vs estimated) as explicit model inputs
[x] Add dataset quality report generation (`nulls`, `gaps`, `dupes`, `coverage windows`) for every build
[x] Add integration tests for real-data slices in `tests/test_data_pipeline.py`

## Phase 3 - Leakage-Safe Labeling and Evaluation Boundaries

[x] Replace correlation-only leakage checks with timestamp-availability and causal-boundary checks in `src/data/labels.py`
[x] Add fold-level leakage guardrails in walk-forward harness to prevent train/test contamination
[x] Enforce that all feature timestamps are strictly prior to forecasted return window
[x] Add regression tests for leakage traps (intentional shifted-target injections should fail)
[x] Add artifact snapshots showing train/test boundaries per fold

## Phase 4 - Cost and Execution Engine Rewrite (Critical Path)

[x] Redesign `src/backtest/costs.py` to compute costs per execution event, not per bar-only approximations
[x] Support fee leg logic correctly for open/increase/reduce/close/reverse transitions
[x] Implement exchange/market-specific fee schedule lookup with optional discount toggles (BNB/KCS)
[x] Implement perpetual funding cashflows with timestamp-accurate settlement and long/short sign handling
[x] Implement margin interest accrual with borrow principal, hourly/daily rates, and hold duration
[x] Implement slippage model in consistent units (bps/price impact) with volatility and size terms
[x] Add comprehensive unit tests for all cost components and edge transitions in `tests/test_strategy.py` and new cost test module
[x] Add audit columns in backtest outputs: `fees`, `funding`, `interest`, `slippage`, `other_costs`, `net_return`

## Phase 5 - Strategy and Position Sizing Robustness

[x] Refactor signal pipeline to separate forecast generation, trade decision, and execution intent
[x] Add robust position transition handling and leverage caps per market type
[x] Enforce quantity/notional rounding by exchange precision and minimum order constraints
[x] Add borrow availability and shorting constraints for margin scenarios
[x] Add optional maker/taker execution policy abstraction for future realism upgrades
[x] Add scenario-level risk constraints (max exposure, max turnover, cooldown after drawdown)

## Phase 6 - Model Evaluation Protocol (Baselines First)

[x] Freeze baseline protocol and fold schedule as immutable config for comparability
[x] Re-run RandomWalk, EWMA, and LightGBM with corrected net-cost engine
[x] Store full per-fold metrics and trade-level outputs for auditability
[x] Make baseline leaderboard reproducible from a single command
[x] Gate Chronos/model advancement on baseline criteria and net profitability evidence

## Phase 7 - Chronos and Meta-Model Validation

[x] Refactor Chronos inference path to strict rolling OOS behavior without test-period contamination
[x] Fix meta-model stacking to use out-of-fold Chronos predictions for training
[x] Correct device-selection and runtime assumptions in meta model construction
[x] Add evaluation splits for recent-regime stress (2024+ equivalent in available data window)
[x] Compare Chronos and meta variants only after passing leakage and net-cost checks

## Phase 8 - Robustness and Stress Testing

[x] Replace random point subsampling with time-contiguous block bootstrap / rolling subperiod stress
[x] Add stress grid for higher fees, higher slippage, worse funding, and higher borrow rates
[x] Add regime-exclusion and adverse-window stress protocols
[x] Add parameter-sensitivity sweeps for entry threshold, uncertainty gate, and leverage cap
[x] Require robustness pass-rate threshold before paper trading approval

## Phase 9 - Reporting, Kill Switches, and Decision Framework

[x] Unify kill-criteria implementation across `src/backtest/report.py` and `src/robustness/kill_criteria.py`
[x] Promote `ProfitFactorNet` as primary gate in all reports and CLI summaries
[x] Add decision report template with explicit `GO / NO-GO / ITERATE` outcomes
[x] Add confidence bands for key metrics using fold and block-bootstrap uncertainty
[x] Add plain-text and JSON outputs that can be consumed by monitoring tooling

## Phase 10 - Paper Trading Readiness

[x] Build paper-trading mode that reuses the same execution-cost engine assumptions
[x] Add daily/weekly monitoring dashboard for PF, Sharpe, drawdown, turnover, and cost decomposition
[x] Add automatic kill-switch triggers when live paper metrics violate thresholds
[x] Define minimum paper-trading observation window before real capital deployment
[x] Define staged capital ramp policy and rollback policy

Phase 10 reproducible commands:
- `python scripts/run_paper_trading.py --data /tmp/phase10_smoke.parquet --model random_walk --start-date 2024-03-01 --scenario binance_futures_taker_discounted --min-train-samples 120 --retrain-bars 30 --training-window-bars 540 --output-dir data/results/phase10_smoke`
- `pytest -q tests/test_phase10_paper_trading.py`

## Phase 11 - Edge Stabilization, Data Completion, and Deployment Readiness (Current Focus)

Objective: Convert the current PF-positive but fragile edge into a promotable system by fixing information gaps, increasing effective sample size, improving entry logic versus costs, and reducing false/non-actionable kill triggers.

### 11.1 Data Completeness and Quality Gating

[ ] Restore open-interest coverage for active benchmark windows and persist in `data/raw/open_interest.parquet`
[ ] Restore liquidation signal coverage (real preferred; high-quality proxy accepted with explicit provenance)
[x] Add hard degraded-run gating in `src/data/build_dataset.py` and downstream scripts when key feature families are missing beyond threshold
[x] Add required key-family availability metrics to `data/processed/btc_4h_quality.json` and fail promotion if degraded
[x] Add Phase 11 data integrity tests for OI/liquidation non-null coverage and degraded-gate behavior

Status note: Binance OI history endpoint currently returns only a short recent horizon in this environment; full-window futures coverage remains blocked pending alternate historical source.

Operational decision (current): futures experiments are paused until historical OI/liquidation provider selection is complete. Active execution track continues on spot/margin scenarios so implementation and policy calibration progress is not blocked.

Spot/Margin active track checklist:
[x] Run Phase 6 baselines on 1h spot scenario (`threshold` and `net_edge`)
[x] Run Phase 6 baselines on 1h margin scenario (`threshold` and `net_edge`)
[x] Run Phase 10 paper replay on 1h spot scenario (`threshold` vs `net_edge`) and compare trade filtering
[x] Run Phase 10 paper replay on 1h margin scenario (`threshold` vs `net_edge`) and compare interest/cost impact
[x] Summarize spot/margin findings and decide whether to tune policy thresholds or move to Phase 11.4 diagnostics

Spot/Margin active-track summary (2026-02-18):
- Phase 6 baseline (1h): `threshold` produced PF Net < 1 on both spot (`0.833`) and margin (`0.827`); `net_edge` produced zero trades on both scenarios.
- Phase 10 paper replay (EWMA, 1h): spot `threshold` PF Net `0.498` (26 trades), margin `threshold` PF Net `0.884` (490 trades, 55 kill events, total interest `0.001474`), and both `net_edge` runs produced zero trades.
- Decision: proceed to Phase 11.4 kill-switch diagnostics and activity-aware calibration before any threshold retuning.

### 11.2 Sample Size Expansion and Timeframe Strategy

[x] Add 1h dataset build mode alongside existing 4h mode while preserving anti-leak guarantees
[ ] Extend historical window coverage for selected timeframe(s) and document resulting regime mix
[ ] Keep 4h as control benchmark and run matched Phase 6 -> Phase 10 comparison between 1h and 4h
[x] Add reproducible script arguments/profile for timeframe selection in pipeline scripts

### 11.3 Net-Edge-Aware Entry Logic (Strategy Upgrade)

[x] Implement net-edge-aware entry rule in `src/strategy/signals.py`: trade only when expected edge exceeds expected cost plus risk buffer
[x] Add scenario-aware expected cost estimate bridge from `CostModel` into live decision logic
[x] Keep fallback to existing threshold logic behind explicit config switch for A/B comparability
[x] Add unit/integration tests proving marginal trades are filtered when expected edge does not clear costs

### 11.4 Kill-Switch Diagnostics Before Threshold Changes

[x] Add kill-event taxonomy artifact grouped by trigger type, regime, activity level, and cost decomposition context
[x] Add diagnostic outputs for low-activity windows to distinguish true risk breaches from inactivity noise
[x] Recalibrate only soft kill criteria based on taxonomy evidence; keep hard drawdown/risk controls strict
[x] Add regression tests for revised kill semantics and low-activity handling

Phase 11.4 implementation notes (2026-02-18):
- New paper artifacts: `*_paper_kill_event_taxonomy.json` and `*_paper_low_activity_diagnostics.json`, plus summary embedding in `*_paper_phase10_summary.json`.
- Kill-switch windows now include trigger classification (`soft`/`hard`/`mixed`) and context columns (`activity_level`, `soft_criteria_eligible`, `kill_switch_trigger_type`).
- Soft-trigger recalibration: PF/Sharpe breaches are now gated by stronger activity evidence (`min_active_bars_for_soft`, `min_trades_for_pf_sharpe`, optional turnover/net-return evidence), while hard risk controls (`max_drawdown`, `cost_to_gross`, `turnover`) remain strict.

### 11.5 Multi-Objective Parameter Optimization

[x] Add composite optimization objective for sweeps including PF Net, Sharpe, trade count, kill-event rate, turnover, and drawdown
[x] Produce Pareto/frontier artifacts and explicit acceptance filters rather than single-metric ranking
[x] Enforce minimum deployment constraints in sweep selection (`PF Net`, Sharpe, trades, kill-rate)
[x] Add script-level reproducible sweep command(s) and artifact schema for comparison across runs

Phase 11.5 implementation notes (2026-02-18):
- Added `src/evaluation/multi_objective.py` with composite scoring, acceptance constraints, Pareto frontier extraction, and a versioned sweep schema payload.
- Added `scripts/run_phase11_sweep.py` to execute policy parameter grids and emit ranked + frontier artifacts:
  `phase11_sweep_candidates_raw.*`, `phase11_sweep_ranked.*`, `phase11_sweep_pareto_frontier.*`, `phase11_sweep_schema.json`, `phase11_sweep_summary.json`.
- Ranking now prioritizes `acceptance_passed` then `composite_score`; acceptance enforces minimum PF/Sharpe/trades and maximum kill-event rate/drawdown.

Phase 11.5 implementation notes (2026-02-24):
- Added explicit `active_candidate` tagging (`num_trades > 0`) in sweep ranking outputs.
- Updated ranking/Pareto ordering to prioritize `acceptance_passed`, then `active_candidate`, then score terms, preventing zero-trade rows from dominating fallback selection.
- Ran threshold-only calibration sweeps on 1h spot/margin (`entry_threshold` `0.0005-0.0025`, `uncertainty_threshold` `0.03-0.05`):
  best spot candidate PF Net `0.811` (197 trades, Sharpe `-1.382`), best margin candidate PF Net `0.806` (197 trades, Sharpe `-1.418`), no accepted candidates.

### 11.6 Chronos/Model Integrity and Calibration Reporting

[x] Add explicit backend provenance fields (model id/backend/fallback status/version) to Chronos artifacts
[x] Disallow labeling a run as Chronos candidate when fallback backend is active unless explicitly flagged as fallback experiment
[x] Add quantile calibration reporting by regime in Phase 7/Phase 9 artifacts
[x] Add tests for provenance and fallback-labeling guardrails

Phase 11.6 implementation notes (2026-02-18):
- `src/models/chronos2_runner.py` now emits explicit provenance payloads/events (`model_id`, `backend`, fallback status/reason, device, versions) via `reset_provenance_log/get_provenance_log`.
- `scripts/run_chronos2.py` captures per-candidate Chronos provenance and writes `phase7_candidate_provenance.json` plus per-model provenance artifacts.
- Phase 7 candidate gate now enforces fallback guardrails for Chronos candidates, with explicit override flag `--allow-fallback-candidate` for fallback-designated experiments.
- Added quantile calibration-by-regime artifacts for both phase-level and candidate-level outputs:
  `phase7_quantile_calibration_by_regime.json` and `*_phase9_quantile_calibration_by_regime.json`.
- Added tests in `tests/test_phase7_chronos.py` covering provenance summary payloads, fallback guardrail behavior, and calibration-by-regime outputs.

### 11.7 Promotion Campaign and Exit Criteria

[x] Freeze one candidate configuration from Phase 11 sweeps and run fixed paper campaign window
[x] Require readiness + capital ramp policy to recommend promotion from `paper` stage without policy exceptions
[x] Define and document Phase 11 completion gate as V0.1 objective satisfied with full secondary criteria and readiness pass

Phase 11.7 implementation notes (2026-02-18):
- Added `src/evaluation/phase11_campaign.py` with:
  `select_phase11_campaign_candidate`, `build_promotion_recommendation`, and `evaluate_phase11_completion_gate`.
- Added `scripts/run_phase11_promotion_campaign.py` to:
  freeze a sweep candidate, run fixed-window paper campaign replay, and emit promotion/completion artifacts.
- New Phase 11.7 artifacts include:
  `phase11_campaign_candidate_freeze.json`,
  `phase11_campaign_promotion_recommendation.json`,
  `phase11_completion_gate.json`,
  and `phase11_campaign_summary.json`.
- Added tests in `tests/test_phase11_promotion_campaign.py` for candidate selection, promotion recommendation rules, and completion-gate checks.

Phase 11.7 implementation notes (2026-02-24):
- Added `best_active_fallback` selection mode for campaign freezing when no candidates pass acceptance, ensuring the default fallback uses an active/traded candidate instead of zero-trade rows.
- Reduced default log verbosity in `run_phase11_sweep.py` and `run_phase11_promotion_campaign.py` to keep large sweeps/campaign runs operationally readable.
- Fixed sweep-to-campaign reproducibility drift by freezing replay/training/runtime config in sweep candidates and making campaign defaults inherit those frozen values unless explicitly overridden.

Phase 11 reproducible command targets:
- `python scripts/download_data.py --start-date 2021-01-01 --interval 1h --build-dataset --output-dir data/results/phase11_1h_bootstrap`
- `python scripts/download_data.py --start-date 2021-01-01 --interval 4h --build-dataset --output-dir data/results/phase11_4h_control`
- `python scripts/build_features.py --interval 1h --output-dir data/results/phase11_1h_features`
- `python scripts/run_baselines.py --timeframe 1h --output-dir data/results/phase11_1h_baselines`
- `python scripts/run_baselines.py --timeframe 4h --output-dir data/results/phase11_4h_baselines`
- `python scripts/run_baselines.py --timeframe 1h --entry-policy net_edge --net-edge-cost-mult 1.0 --net-edge-risk-mult 0.25 --output-dir data/results/phase11_1h_baselines_netedge`
- `python scripts/run_chronos2.py --timeframe 1h --output-dir data/results/phase11_1h_chronos`
- `python scripts/run_paper_trading.py --timeframe 1h --model ewma --entry-policy net_edge --net-edge-cost-mult 1.0 --net-edge-risk-mult 0.25 --output-dir data/results/phase11_1h_paper`
- `python scripts/run_baselines.py --timeframe 1h --scenario binance_spot_taker_discounted --entry-policy threshold --output-dir data/results/phase11_1h_baselines_spot_threshold`
- `python scripts/run_baselines.py --timeframe 1h --scenario binance_spot_taker_discounted --entry-policy net_edge --net-edge-cost-mult 1.0 --net-edge-risk-mult 0.25 --output-dir data/results/phase11_1h_baselines_spot_netedge`
- `python scripts/run_baselines.py --timeframe 1h --scenario binance_margin_taker_discounted --entry-policy threshold --output-dir data/results/phase11_1h_baselines_margin_threshold`
- `python scripts/run_baselines.py --timeframe 1h --scenario binance_margin_taker_discounted --entry-policy net_edge --net-edge-cost-mult 1.0 --net-edge-risk-mult 0.25 --output-dir data/results/phase11_1h_baselines_margin_netedge`
- `python scripts/run_paper_trading.py --timeframe 1h --model ewma --scenario binance_spot_taker_discounted --entry-policy threshold --output-dir data/results/phase11_1h_paper_spot_threshold`
- `python scripts/run_paper_trading.py --timeframe 1h --model ewma --scenario binance_spot_taker_discounted --entry-policy net_edge --net-edge-cost-mult 1.0 --net-edge-risk-mult 0.25 --output-dir data/results/phase11_1h_paper_spot_netedge`
- `python scripts/run_paper_trading.py --timeframe 1h --model ewma --scenario binance_margin_taker_discounted --entry-policy threshold --output-dir data/results/phase11_1h_paper_margin_threshold`
- `python scripts/run_paper_trading.py --timeframe 1h --model ewma --scenario binance_margin_taker_discounted --entry-policy net_edge --net-edge-cost-mult 1.0 --net-edge-risk-mult 0.25 --output-dir data/results/phase11_1h_paper_margin_netedge`
- `python scripts/run_phase11_sweep.py --timeframe 1h --model ewma --scenario binance_spot_taker_discounted --start-date 2025-12-01 --entry-policies threshold,net_edge --entry-thresholds 0.0025,0.003,0.0035 --uncertainty-thresholds 0.02,0.03 --net-edge-cost-mults 0.75,1.0,1.25 --net-edge-risk-mults 0.0,0.25,0.5 --output-dir data/results/phase11_5_sweep`
- `python scripts/run_phase11_promotion_campaign.py --timeframe 1h --ranked-candidates data/results/phase11_5_sweep/phase11_sweep_ranked.json --campaign-start-date 2025-12-01 --campaign-end-date 2026-02-18 --output-dir data/results/phase11_7_campaign`
- `pytest -q tests/test_data_pipeline.py tests/test_phase10_paper_trading.py tests/test_phase11_timeframe.py`

## Required Deliverables Per Phase

[ ] Code changes merged
[x] Tests added/updated and passing
[x] Reproducible command(s) documented
[x] Artifacts generated in `data/results/`
[x] Phase checklist in this document updated

## Execution Order (Strict)

[x] Complete Phase 0
[x] Complete Phase 1
[x] Complete Phase 2
[x] Complete Phase 3
[x] Complete Phase 4
[x] Complete Phase 5
[x] Complete Phase 6
[x] Complete Phase 7
[x] Complete Phase 8
[x] Complete Phase 9
[x] Complete Phase 10
[ ] Complete Phase 11 (current focus)

## Phase Status Board

[x] Phase 0 complete
[x] Phase 1 complete
[x] Phase 2 complete
[x] Phase 3 complete
[x] Phase 4 complete
[x] Phase 5 complete
[x] Phase 6 complete
[x] Phase 7 complete
[x] Phase 8 complete
[x] Phase 9 complete
[x] Phase 10 complete
[ ] Phase 11 complete
[ ] V0.1 objective achieved (`ProfitFactorNet > 1.00` with required secondary criteria)
