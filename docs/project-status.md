# Project Status

## Snapshot

This page separates implementation status from empirical status.

## Capability Matrix

| Area | Engineering status | Empirical status | Notes |
| --- | --- | --- | --- |
| Data ingestion and contracts | Implemented | Mixed | Core pipeline exists; some feature-family completeness remains a limitation |
| Leakage-safe evaluation | Implemented | Positive | Test and protocol coverage are strong |
| Cost-aware backtesting | Implemented | Positive | Central design strength of the repo |
| Baseline model stack | Implemented | Mixed | EWMA shows life; LightGBM is weak in inspected public artifacts |
| Chronos candidate path | Implemented | Unresolved | Infrastructure exists; public evidence remains weaker than EWMA-led evidence |
| Strategy and sizing layer | Implemented | Mixed | Sensitive to threshold and regime assumptions |
| Robustness and kill criteria | Implemented | Positive | Strong public differentiator; blocks premature promotion |
| Paper-trading governance | Implemented | Positive | Readiness and ramp logic are present and active |
| Promotion readiness | Implemented | Negative | No inspected candidate is promotion-ready yet |

## Findings Matrix

### Evidence-positive

- governed research architecture is implemented and validated
- strongest futures EWMA candidate reaches `PF Net > 1.0`
- 4h looser-threshold spot and margin checks recover PF-positive, positive-Sharpe regions
- governance layers actively prevent overclaiming and premature promotion

### Evidence-negative

- no inspected candidate qualifies for promotion
- 1h spot and margin threshold calibration remain below PF / Sharpe acceptance thresholds
- fixed spot campaign remains negative
- kill-switch activity is still a major blocker even when PF and Sharpe improve

### Unresolved

- how much incremental value Chronos adds over the best baseline path when run under the strongest available data conditions
- how much signal quality improves after repairing OI / liquidation completeness
- whether a better readiness-calibrated policy can preserve the PF-positive region without causing fragility
