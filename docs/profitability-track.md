# Profitability Track

This note records the first post-publication advancement cycle focused on moving from
"interesting research" toward a defendable profitability claim.

## Current Status

- There is no credible basis yet to claim deployment readiness or production readiness.
- The strongest current branch is a `4h` EWMA paper-trading configuration under
  `binance_spot_taker_discounted`.
- The main blocker has shifted from broad kill-switch noise to a narrower robustness problem:
  either one bad weekly chop window triggers governance, or the no-kill variant does not
  produce enough trades.

## What Changed

We tightened the kill-switch implementation so that soft checks and `cost_to_gross`
enforcement do not trigger on windows that are too short to be meaningful:

- `min_bars_for_soft`
- `min_bars_for_cost_to_gross`
- `min_abs_gross_return_for_cost_to_gross`

Those thresholds are now exposed through:

- `scripts/run_paper_trading.py`
- `scripts/run_phase11_sweep.py`

This was validated by targeted tests in
[tests/test_phase11_kill_switch_diagnostics.py](../tests/test_phase11_kill_switch_diagnostics.py)
and the full suite.

## Validation

- `./.venv/bin/pytest -q tests/test_phase11_kill_switch_diagnostics.py` -> `5 passed`
- `./.venv/bin/pytest -q` -> `125 passed`

## Key Experiment Results

### 1. Governance-Calibrated 4h Baseline

Command family:

```bash
./.venv/bin/python scripts/run_paper_trading.py \
  --timeframe 4h \
  --model ewma \
  --scenario binance_spot_taker_discounted \
  --entry-policy threshold \
  --entry-threshold 0.0015 \
  --uncertainty-threshold 0.04 \
  --start-date 2025-12-01 \
  --kill-min-bars-for-soft 12 \
  --kill-min-bars-for-cost-to-gross 12 \
  --kill-min-abs-gross-return-for-cost-to-gross 0.002
```

Spot result:

- PF Net: `1.1517`
- Sharpe: `1.3640`
- Trades: `145`
- Kill events: `1`
- Readiness: `false` (`kill_switch_triggered`)

Margin result:

- PF Net: `1.1346`
- Sharpe: `1.2199`
- Trades: `145`
- Kill events: `1`
- Readiness: `false` (`kill_switch_triggered`)

Interpretation:

- The previous `25` kill-event problem was mostly short-window policy noise.
- After calibration, both spot and margin collapse to the same remaining blocker:
  one adverse weekly chop window around `2026-01-11`.

### 2. Focused 4h Spot Threshold Sweep

Artifact:

- `data/results/advancement_4h_spot_sweep_governance_v1/phase11_sweep_summary.json`

Grid:

- entry thresholds: `0.0015, 0.00175, 0.002, 0.00225, 0.0025`
- uncertainty thresholds: `0.035, 0.04, 0.045, 0.05`
- retrain bars: `42`
- kill policy: governance-calibrated thresholds above

Summary:

- candidates: `20`
- accepted under sweep constraints: `8`
- best accepted candidate:
  - entry threshold: `0.0015`
  - uncertainty threshold: `0.035`
  - PF Net: `1.2649`
  - Sharpe: `2.0345`
  - Trades: `118`
  - Kill events: `1`
  - Deployment readiness: `false`

Important boundary:

- `0`-kill candidates exist, but they fall below the `80`-trade floor.
- `>= 80`-trade candidates still carry `1` kill event in this sweep.

### 3. Retrain-Cadence Probe Around the Best 4h Branch

The strongest improvement came from increasing `retrain-bars` to `63`.

Best high-quality candidate found so far:

- timeframe: `4h`
- scenario: `binance_spot_taker_discounted`
- model: `ewma`
- retrain bars: `63`
- entry threshold: `0.00163`
- uncertainty threshold: `0.035`
- start date: `2025-12-01`
- PF Net: `1.4567`
- Sharpe: `2.7335`
- Trades: `73`
- Kill events: `0`
- Deployment readiness: `false` (`insufficient_trades`)

Relevant artifact:

- `data/results/advancement_4h_spot_retrain63_entry163_governance_v1/ewma_paper_phase10_summary.json`

Interpretation:

- This is the cleanest candidate seen so far.
- It is not deployment-ready because the sample is still too small.
- This is the first branch that survives the kill switch cleanly while maintaining strong PF/Sharpe.

### 4. Longer-History Robustness Check

The best zero-kill candidate did not survive earlier history:

- start `2025-11-15`:
  - PF Net: `0.8746`
  - Sharpe: `-0.7471`
  - Trades: `42`
  - Kill events: `2`
- start `2025-11-01`:
  - PF Net: `0.3440`
  - Sharpe: `-3.5230`
  - Trades: `33`
  - Kill events: `1`

Interpretation:

- The current zero-kill branch is not yet robust enough to support a profitability claim.
- The edge appears regime-sensitive rather than stable across nearby history.

### 5. Net-Edge Branch

The tested `net_edge` sweep on the same 4h spot region was not productive.

Observed behavior:

- all `48/48` tested candidates produced `0` trades
- accepted candidates: `0`
- this branch is currently too conservative for the present EWMA setup

This path should not be prioritized until the threshold branch is exhausted or the
net-edge policy is recalibrated more fundamentally.

### 6. Regime-Aware 4h Sweep

Artifact:

- `data/results/advancement_4h_spot_regime_sweep_governance_v1/phase11_sweep_summary.json`

We extended the sweep tooling to search regime-specific entry-threshold multipliers directly.

Search region:

- timeframe: `4h`
- scenario: `binance_spot_taker_discounted`
- model: `ewma`
- retrain bars: `63`
- entry thresholds: `0.00162`, `0.001625`, `0.00163`
- uncertainty threshold: `0.035`
- trend multipliers: `0.9`, `1.0`
- normal multipliers: `0.9`, `0.95`, `1.0`
- chop multipliers: `1.0`, `1.1`, `1.2`

Summary:

- candidates: `54`
- accepted under sweep constraints: `28`
- best overall:
  - entry threshold: `0.00162`
  - trend multiplier: `0.9`
  - normal multiplier: `0.9`
  - chop multiplier: `1.0`
  - PF Net: `1.3968`
  - Sharpe: `2.5599`
  - Trades: `88`
  - Kill events: `1`

Critical outcome:

- there were still no candidates with both `kill_events == 0` and `num_trades >= 80`
- the best zero-kill candidates remained in the `73-74` trade band
- regime-aware thresholding improved the search tooling but did not break the core frontier

Interpretation:

- this is strong evidence that local parameter tuning on the current feature set is close to exhausted
- the next material improvement is more likely to come from data/feature changes than from more threshold search

### 7. OI Coverage Investigation

The processed `4h` dataset still contains no usable OI family values:

- `open_interest`: `0 / 11247` non-null
- `oi_change_pct_1`: `0 / 11247` non-null
- `oi_change_pct_6`: `0 / 11247` non-null

Direct OI endpoint checks showed the current Binance history path is effectively unusable for this research window:

- many requested windows are rejected with `startTime` errors
- even "successful" recent requests only returned a tiny trailing slice near `2026-03-31` to `2026-04-01`

Interpretation:

- the repo’s current OI pathway is not providing historical coverage needed for this strategy
- repairing OI/liquidation coverage now looks like a genuine prerequisite for the next profitability jump

## Evidence-Based Conclusion

We have improved the state of the project materially:

- governance calibration is better
- the 4h branch has a much cleaner frontier
- a zero-kill candidate exists with very strong PF/Sharpe
- the regime-aware search space is now supported directly in the sweep tooling

But we are still below the standard required for a profitability claim because:

- the best robust-looking candidate is under-sampled (`73` trades)
- extending the window breaks the result
- the higher-trade candidates still fail governance once
- OI/liquidation signal families are still effectively absent from the dataset

## Next Actions

1. Run a coupled sweep over:
   - `retrain-bars`
   - `entry-threshold`
   - possibly `training-window-bars`

2. Keep the governance-calibrated kill policy fixed while searching.

3. Require for the next candidate:
   - `kill_events == 0`
   - `num_trades >= 80`
   - `profit_factor_net > 1.0`
   - `sharpe_ratio >= 0.5`

4. If no such candidate exists on the current feature set, stop optimizing surface parameters
   and move to feature/data quality improvements:
   - restore usable OI/liquidation coverage
   - test more explicit chop suppression
   - compare 4h and 1h candidates under matched governance

5. Do not loosen readiness policy just to pass a gate. Any policy change must be justified by
   timeframe-aware evidence, not by the desire to promote a candidate.
