# Public Evidence Snapshot

This file is generated from selected experiment artifacts and is intended for public publication.

## Project Positioning

`chronos-plg` should be read as a research platform, not as a finished live-trading system.

## Futures Candidate Comparison

| Candidate | Net PF | Sharpe | Trades | Total return (%) | Decision |
| --- | --- | --- | --- | --- | --- |
| Best EWMA candidate | 1.173851099607019 | 0.5652507801901155 | 76 | 1.0547880875436233 | ITERATE |
| Higher-trade EWMA | 1.1212709977497273 | 0.451235125772597 | 100 | 0.9329438707592175 | ITERATE |
| EWMA baseline | 1.0521323715317406 | 0.10005302517853053 | 26 | 0.08542766075962938 | NO_GO |
| LightGBM baseline | 0.22392419404156538 | -2.433731154360648 | 36 | -1.34635024815698 | NO_GO |

## Spot Threshold Calibration: Top Active Candidates

| Scenario | Entry threshold | Uncertainty threshold | Net PF | Sharpe | Trades | Kill-event rate |
| --- | --- | --- | --- | --- | --- | --- |
| spot | 0.0015 | 0.03 | 0.8023517359935688 | -1.3652741959574646 | 175 | 0.1075268817204301 |
| spot | 0.0015 | 0.04 | 0.810823172731907 | -1.381922040236017 | 197 | 0.1290322580645161 |
| spot | 0.0015 | 0.05 | 0.810823172731907 | -1.381922040236017 | 197 | 0.1290322580645161 |
| spot | 0.001 | 0.03 | 0.7932419327101078 | -1.7177542421898355 | 291 | 0.1720430107526881 |
| spot | 0.001 | 0.04 | 0.8011759896171521 | -1.7264134102265254 | 316 | 0.1935483870967742 |

## Margin Threshold Calibration: Top Active Candidates

| Scenario | Entry threshold | Uncertainty threshold | Net PF | Sharpe | Trades | Kill-event rate |
| --- | --- | --- | --- | --- | --- | --- |
| margin | 0.0015 | 0.03 | 0.7979392704850765 | -1.399526030071732 | 175 | 0.1075268817204301 |
| margin | 0.0015 | 0.04 | 0.8063718531239323 | -1.418284300232065 | 197 | 0.1290322580645161 |
| margin | 0.0015 | 0.05 | 0.8063718531239323 | -1.418284300232065 | 197 | 0.1290322580645161 |
| margin | 0.001 | 0.03 | 0.7885509354428244 | -1.761992969735155 | 291 | 0.1720430107526881 |
| margin | 0.001 | 0.04 | 0.796467294725604 | -1.772594553352172 | 316 | 0.1935483870967742 |

## Fixed Campaign Snapshot

- Campaign window: `2025-12-01` to `2026-02-24`
- Selection mode: `best active fallback`
- Profit factor net: `0.8509`
- Sharpe ratio: `-0.8225`
- Trades: `94`
- Total return: `-1.28%`
- Kill events: `6`
- Deployment ready: `False`
- Readiness reason: `kill switch triggered; profit factor below threshold; sharpe below threshold`
- Promotion recommended: `False`
- Completion gate passed: `False`

## 4h Threshold Sensitivity Check

| Scenario | Policy | Net PF | Sharpe | Trades | Kill events | Ready | Readiness reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Spot | Default Threshold | 0.0 | 0.0 | 0 | 0 | False | insufficient trades; profit factor below threshold; sharpe below threshold |
| Spot | Looser Threshold | 1.1516600501423204 | 1.363972021770645 | 145 | 25 | False | kill switch triggered |
| Margin | Default Threshold | 0.0 | 0.0 | 0 | 0 | False | insufficient trades; profit factor below threshold; sharpe below threshold |
| Margin | Looser Threshold | 1.1345610473507217 | 1.2199380762610668 | 145 | 25 | False | kill switch triggered |

## Interpretation

- The best observed futures configuration demonstrates that the system can find a positive net-edge region under strict cost accounting.
- That positive edge is still too fragile for promotion because regime stability, trade-count sufficiency, and kill-switch behavior remain binding constraints.
- The 1h spot and margin calibration runs show the opposite failure mode: higher trade count, but consistently weak PF / Sharpe and no acceptable candidates.
- The 4h spot and margin default-threshold runs were too conservative to trade at all over the inspected window, but a looser 4h threshold recovered attractive PF / Sharpe regions that still failed readiness due to kill-switch activity.
- This is the kind of evidence that makes the project interesting publicly: the governance layer is rejecting attractive-looking but not deployment-ready configurations.
