# 06 - Performance Status And Strategy Health

## Artifact Basis

Primary artifact set reviewed:

- `data/results/phase10_real_20260217/*`
- `data/results/phase10_real_20260217/*_paper_phase10_summary.json`
- EWMA sweep CSVs in `data/results/phase10_real_20260217/*.csv`

## Phase-10 Run Summary

| Run                                              | PF Net |  Sharpe | Trades | Total Return | Max DD | Kill Events | Readiness |
|--------------------------------------------------|-------:|--------:|-------:|-------------:|-------:|------------:|-----------|
| `ewma_candidate_sharpe565_retrain42_entry178_v4` | 1.1739 |  0.5653 |     76 |        1.05% | -1.43% |          17 | Not ready |
| `ewma_candidate_span72_entry178_retrain28_v3`    | 1.1213 |  0.4512 |    100 |        0.93% | -1.86% |          24 | Not ready |
| `ewma`                                           | 1.0521 |  0.1001 |     26 |        0.09% | -0.65% |         608 | Not ready |
| `lightgbm`                                       | 0.2239 | -2.4337 |     36 |       -1.35% | -1.44% |         611 | Not ready |

## Health Diagnostics

### Positive Signals

- Net edge exists in tuned EWMA regions (`PF Net > 1.0`)
- Candidate set shows controllable drawdowns in tested period
- Cost-aware system is able to retain positive PF for selected params after modeled fees/slippage/funding

### Blocking Signals

- Deployment readiness fails across current runs
- Frequent kill-switch triggers remain a dominant blocker
- Trade counts are below current readiness threshold in best-Sharpe region
- LightGBM baseline currently performs poorly in latest Phase-10 replay

## Parameter Frontier Insights

From `ewma_sharpe_tradecount_bridge_sweep.csv`:

- Best Sharpe point:
  - `retrain_bars=40`, `entry_threshold=0.00178`
  - `Sharpe=1.0537`, `PF Net=1.4468`, `Trades=56`
  - Strong quality, weak trade count
- Higher-trade points (~71-76 trades):
  - Sharpe around `0.54-0.57`
  - PF Net around `1.17-1.19`
  - Still not readiness-qualified under current policy

This confirms a current frontier trade-off:

- High quality / low trade count
- Higher trade count / lower quality

## Data Context Behind Results

From `data/processed/btc_4h_quality.json`:

- OI unavailable in latest processed dataset build
- Liquidation estimators are structurally present but effectively null in this dataset

Implication:

- Current strategy is operating without active OI/liquidation signal contribution
- Edge quality may shift materially once reliable OI/liquidation streams are restored

## Current Strategic Assessment

- System has a real, but still fragile, positive edge region under present assumptions
- Governance stack is correctly preventing premature live deployment
- The next improvement cycle should target preserving PF>1 while lifting eligible trade count and reducing kill-trigger frequency
