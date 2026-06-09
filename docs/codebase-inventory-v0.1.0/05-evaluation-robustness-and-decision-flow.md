# 05 - Evaluation, Robustness, And Decision Flow

## Walk-Forward Evaluation (`src/evaluation/walk_forward.py`)

### Core Guarantees

- Datetime index must be timezone-aware, monotonic, and duplicate-free
- Train/test folds are strictly non-overlapping
- Features are shifted by `feature_lag_candles >= 1`
- Fold artifacts can be saved for auditability

### Fold Mechanics

- Configurable train/test/step windows (weekly or monthly mode)
- Deterministic fold schedule generation
- Optional fold boundary snapshot with train/test timestamps and gap stats

## Backtest Engine (`src/backtest/engine.py`)

Backtest loop per fold:

1. Train model on train slice
2. Predict quantiles OOS on test slice
3. Generate signals and positions
4. Apply execution-event cost model
5. Compute gross/net returns and aggregate metrics

Stored outputs:

- Full returns frame with cost audit components
- Trade-event view (`traded_notional > 0`)
- Per-fold summary metrics
- Equity curve and regime-level metrics

## Phase 6 Baseline Protocol

Implemented in `scripts/run_baselines.py` + `src/evaluation/phase6_baselines.py`

### Reproducibility Controls

- Immutable baseline protocol object
- Fingerprinted protocol freeze artifact
- Frozen fold schedule artifact
- Baseline leaderboard in CSV/JSON/Markdown

### Gate Produced

- Chronos advancement gate payload with:
  - best baseline PF/Sharpe
  - anchor baseline
  - required candidate thresholds

## Phase 7 Candidate Validation

Implemented in `scripts/run_chronos2.py` + `src/evaluation/phase7_chronos.py`

Flow:

- Re-run baselines under frozen protocol/folds/scenario
- Enforce leakage + baseline-net-cost gate before candidate comparison
- Run Chronos2/MetaModel only if gates pass
- Compute recent-regime split metrics (anchor at 2024-01-01 if available)
- Build candidate-vs-anchor gate payloads

Key candidate checks:

- `profit_factor_net > threshold`
- `sharpe_net >= threshold`
- `sharpe_delta_vs_anchor >= threshold`
- `recent_sharpe_ratio_vs_early >= threshold`

## Phase 8 Robustness Suite (`src/robustness/stress_tests.py`)

Implemented stress modules:

- Cost stress grid (fee/slippage/funding/borrow deterioration)
- Regime exclusion protocol
- Adverse contiguous window protocol
- Time-contiguous block bootstrap stability
- Rolling subperiod stability
- Parameter sensitivity sweep (entry/uncertainty/leverage)

Viability integration:

- `src/robustness/summary.py` combines kill criteria + stress pass rate threshold

## Phase 9 Decision Framework (`src/reporting/decision.py`)

Decision outcomes:

- `GO`
- `ITERATE`
- `NO_GO`

Primary gate semantics:

- PF net is primary gate
- Severe fails on PF/sharpe/drawdown drive NO_GO
- Win-rate is advisory, not hard-fail

Uncertainty modeling:

- Fold-based bands
- Block-bootstrap bands
- Reported for Sharpe, PF net, total return

## Phase 10 Paper Governance (`scripts/run_paper_trading.py`)

Artifacts generated:

- Paper log
- Returns + cost audit CSV
- Daily/weekly dashboard CSVs
- Kill-switch event JSON
- Deployment readiness JSON
- Capital ramp policy JSON/TXT
- Capital ramp decision JSON
- Phase-10 summary JSON with embedded decision payload

Policy behavior:

- Kill switch enforced after minimum active windows
- PF/Sharpe checks can be deferred for very low trade-count windows
- Readiness requires observation length + trade count + PF/Sharpe/DD + kill-event policy
- Capital action returns `PROMOTE`, `HOLD`, or `ROLLBACK`

## Gate Stack Summary

1. Data contracts + leakage checks
2. Baseline gate
3. Candidate gate vs anchor + recent regime stability
4. Robustness pass-rate threshold
5. Phase-9 decision outcome
6. Phase-10 paper kill/readiness/ramp controls

This is a fully implemented multi-layer governance stack, not a single-metric pass/fail setup.
