# 07 - Capability Matrix, Gaps, And Next Actions

## Capability Matrix

| Capability Area                   | Status      | Notes                                                               |
|-----------------------------------|-------------|---------------------------------------------------------------------|
| Exchange/market cost profiles     | Implemented | Binance + KuCoin, spot/margin/futures, discount toggles             |
| Scenario abstraction              | Implemented | Funding/interest/other-cost toggles are scenario-driven             |
| Contract-validated data ingestion | Implemented | Datetime/index contracts and required-column checks                 |
| Feature engineering               | Implemented | Price, funding, OI, liquidation, macro, event flags                 |
| Leakage guardrails                | Implemented | Structural checks + shifted-target leak trap detection              |
| Walk-forward fold generation      | Implemented | Strict non-overlap, reproducible boundaries, artifact snapshots     |
| Baseline model stack              | Implemented | RandomWalk, EWMA, LightGBM quantiles                                |
| Chronos candidate stack           | Implemented | Chronos2 strict OOS + MetaModel OOF stacking                        |
| Strategy layer                    | Implemented | Signal, regime gating, sizing, intent, risk constraints             |
| Execution-event cost accounting   | Implemented | Transition-leg-aware fee/slippage/funding/interest/other costs      |
| Backtest reporting                | Implemented | Per-model reports + PF-first comparison                             |
| Robustness suite                  | Implemented | Cost stress, regime/adverse windows, bootstrap/rolling, sensitivity |
| Decision framework                | Implemented | GO/ITERATE/NO_GO + fold/bootstrap uncertainty bands                 |
| Paper-trading governance          | Implemented | Daily/weekly monitoring, kill switch, readiness, capital ramp       |
| End-to-end script tooling         | Implemented | Script entry points + run manifests + smoke check                   |

## Current Gaps That Matter Most

### Gap 1 - Data completeness for OI/liquidations

Observed:

- OI unavailable in latest processed dataset
- Liquidation features effectively null

Impact:

- Reduced feature diversity and potentially weaker regime/event discrimination

### Gap 2 - Readiness policy not satisfied by current edge region

Observed:

- Best candidates pass PF gate but fail readiness (`insufficient_trades`, kill events)

Impact:

- No safe promotion path under current operational policy

### Gap 3 - Kill-switch sensitivity vs low-activity windows

Observed:

- Very high kill-event counts in several runs

Impact:

- Risk of over-harsh operational gating despite PF-positive parameter zones

### Gap 4 - LightGBM deterioration in latest forward replay

Observed:

- Strongly negative latest phase-10 replay metrics for LightGBM

Impact:

- Baseline anchor quality may be unstable across current data regime

## Recommended Next Actions (Implementation Order)

1. Restore OI/liquidation data path quality
- Fix or replace OI history source for target window coverage
- Rebuild dataset and validate non-null coverage for OI/liq features
- Re-run Phase 6 -> Phase 10 chain on same frozen protocol settings

2. Build a readiness-calibrated parameter search objective
- Optimize jointly for `PF Net`, `Sharpe`, `Trades`, and kill-event rate
- Add explicit penalty for kill-trigger frequency in sweep objective
- Keep strict out-of-sample and same scenario assumptions

3. Tune kill-switch policy on active-window semantics
- Re-evaluate `min_windows_before_enforcement` and low-trade PF/Sharpe enforcement thresholds
- Keep hard drawdown controls strict; tune soft triggers to reduce false positives

4. Stabilize baseline anchor governance
- Track baseline drift by period and scenario
- Pin anchor to robust PF-first + stability criteria, not single recent run

5. Run a fixed acceptance campaign
- Freeze one candidate configuration
- Run paper replay for required observation horizon
- Accept promotion only if readiness passes with no policy exceptions

## Practical Definition Of “Ready” For This Codebase

Under current policy implementation:

- Primary edge gate: `PF Net > 1.0`
- Net Sharpe and drawdown must satisfy thresholds
- Minimum observation duration and trade count must be met
- Kill-switch events must satisfy policy tolerance
- Capital-ramp recommendation should be `PROMOTE` from `paper`

Until those are simultaneously true, the correct operational state remains `ITERATE`.
