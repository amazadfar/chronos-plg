# Public Evidence Snapshot

This file is generated from selected experiment artifacts and is intended for public publication.

## Project Positioning

`chronos-plg` should be read as a research platform, not as a finished live-trading system.

## Phase 10 Showcase Runs

| run | profit_factor_net | sharpe_ratio | num_trades | total_return_pct | max_drawdown_pct | total_costs_pct | decision | decision_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ewma_candidate_sharpe565_retrain42_entry178_v4 | 1.173851099607019 | 0.5652507801901155 | 76 | 1.0547880875436233 | -1.434022467114876 | 1.838168335712359 | ITERATE | partial_pass_requires_iteration |
| ewma_candidate_span72_entry178_retrain28_v3 | 1.1212709977497273 | 0.451235125772597 | 100 | 0.9329438707592175 | -1.858432737527022 | 2.4410111978803912 | ITERATE | partial_pass_requires_iteration |
| ewma | 1.0521323715317406 | 0.10005302517853053 | 26 | 0.08542766075962938 | -0.6519046314678811 | 0.5123854426776856 | NO_GO | primary_or_severe_risk_gate_failed |
| lightgbm | 0.22392419404156538 | -2.433731154360648 | 36 | -1.34635024815698 | -1.4378086301544233 | 1.4134555362818597 | NO_GO | primary_or_severe_risk_gate_failed |

## Phase 11 Spot Threshold Calibration: Top Active Candidates

| candidate_id | scenario | entry_threshold | uncertainty_threshold | profit_factor_net | sharpe_ratio | num_trades | kill_event_rate | max_drawdown_abs | composite_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | spot | 0.0015 | 0.03 | 0.8023517359935688 | -1.3652741959574646 | 175 | 0.1075268817204301 | 0.0451645778398956 | 0.4755038649710405 |
| 8 | spot | 0.0015 | 0.04 | 0.810823172731907 | -1.381922040236017 | 197 | 0.1290322580645161 | 0.0451645778398956 | 0.4716641442375388 |
| 9 | spot | 0.0015 | 0.05 | 0.810823172731907 | -1.381922040236017 | 197 | 0.1290322580645161 | 0.0451645778398956 | 0.4716641442375388 |
| 4 | spot | 0.001 | 0.03 | 0.7932419327101078 | -1.7177542421898355 | 291 | 0.1720430107526881 | 0.0607500774928959 | 0.1354888430200968 |
| 5 | spot | 0.001 | 0.04 | 0.8011759896171521 | -1.7264134102265254 | 316 | 0.1935483870967742 | 0.0633056375864531 | 0.1296840077780312 |

## Phase 11 Margin Threshold Calibration: Top Active Candidates

| candidate_id | scenario | entry_threshold | uncertainty_threshold | profit_factor_net | sharpe_ratio | num_trades | kill_event_rate | max_drawdown_abs | composite_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | margin | 0.0015 | 0.03 | 0.7979392704850765 | -1.399526030071732 | 175 | 0.1075268817204301 | 0.0456078716163502 | 0.4366179184600535 |
| 8 | margin | 0.0015 | 0.04 | 0.8063718531239323 | -1.418284300232065 | 197 | 0.1290322580645161 | 0.0456078716163502 | 0.4306289177452889 |
| 9 | margin | 0.0015 | 0.05 | 0.8063718531239323 | -1.418284300232065 | 197 | 0.1290322580645161 | 0.0456078716163502 | 0.4306289177452889 |
| 4 | margin | 0.001 | 0.03 | 0.7885509354428244 | -1.761992969735155 | 291 | 0.1720430107526881 | 0.0616263569859072 | 0.0861209784609882 |
| 5 | margin | 0.001 | 0.04 | 0.796467294725604 | -1.772594553352172 | 316 | 0.1935483870967742 | 0.0645795897358792 | 0.0781571936861233 |

## Fixed Campaign Snapshot

- Campaign window: `2025-12-01` to `2026-02-24`
- Selection mode: `best_active_fallback`
- Profit factor net: `0.8509`
- Sharpe ratio: `-0.8225`
- Trades: `94`
- Total return: `-1.28%`
- Kill events: `6`
- Deployment ready: `False`
- Readiness reason: `kill_switch_triggered;profit_factor_below_threshold;sharpe_below_threshold`
- Promotion recommended: `False`
- Completion gate passed: `False`

## 4h Threshold Sensitivity Check

| scenario | policy | profit_factor_net | sharpe_ratio | num_trades | kill_events | deployment_ready | deployment_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spot | default_threshold | 0.0 | 0.0 | 0 | 0 | False | insufficient_trades;profit_factor_below_threshold;sharpe_below_threshold |
| spot | looser_threshold | 1.1516600501423204 | 1.363972021770645 | 145 | 25 | False | kill_switch_triggered |
| margin | default_threshold | 0.0 | 0.0 | 0 | 0 | False | insufficient_trades;profit_factor_below_threshold;sharpe_below_threshold |
| margin | looser_threshold | 1.1345610473507217 | 1.2199380762610668 | 145 | 25 | False | kill_switch_triggered |

## Interpretation

- The best observed futures configuration demonstrates that the system can find a positive net-edge region under strict cost accounting.
- That positive edge is still too fragile for promotion because regime stability, trade-count sufficiency, and kill-switch behavior remain binding constraints.
- The 1h spot and margin calibration runs show the opposite failure mode: higher trade count, but consistently weak PF / Sharpe and no acceptable candidates.
- The 4h spot and margin default-threshold runs were too conservative to trade at all over the inspected window, but a looser 4h threshold recovered attractive PF / Sharpe regions that still failed readiness due to kill-switch activity.
- This is the kind of evidence that makes the project interesting publicly: the governance layer is rejecting attractive-looking but not deployment-ready configurations.
