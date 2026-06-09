# Experiment Log

This is a compact public log of representative experiment outcomes. It is not a full dump of every local run.

## Card 1: Futures EWMA Candidate

### Hypothesis

A tuned EWMA policy may preserve a positive net edge after full cost accounting.

### Representative artifact

- `data/results/phase10_real_20260217/ewma_candidate_sharpe565_retrain42_entry178_v4/ewma_paper_phase10_summary.json`

### Observed outcome

- `PF Net = 1.1739`
- `Sharpe = 0.5653`
- `Trades = 76`
- decision outcome: `ITERATE`

### Why it matters

This is the strongest inspected evidence that the system can locate a positive region.

### Why it still did not promote

The governance layer still found readiness / stability issues. That is an important result, not an embarrassment.

## Card 2: Spot Threshold Calibration

### Hypothesis

Lowering entry thresholds on the 1h spot track might raise trade count without destroying edge quality.

### Representative artifact

- `data/results/phase11_5_sweep_spot_threshold_calib/phase11_sweep_ranked.csv`

### Observed outcome

- top active spot candidates reached `175-197` trades
- PF stayed around `0.80-0.81`
- Sharpe remained negative
- accepted candidate count: `0`

### Conclusion

Higher activity alone did not solve the problem.

## Card 3: Fixed-Window Spot Campaign

### Hypothesis

A frozen active fallback candidate might demonstrate readiness over the campaign window.

### Representative artifact

- `data/results/phase11_7_campaign_spot_one/phase11_campaign_summary.json`

### Observed outcome

- `PF Net = 0.8509`
- `Sharpe = -0.8225`
- `Trades = 94`
- `Kill events = 6`
- promotion recommendation: `False`

### Conclusion

The campaign evidence remained negative, and the policy correctly held the candidate back.

## Card 4: 4h Threshold Sensitivity Check

### Hypothesis

The 4h track may not be intrinsically dead; the default threshold policy may simply be too conservative for the inspected window.

### Representative artifacts

- `data/results/publication_4h_spot_threshold/ewma_paper_phase10_summary.json`
- `data/results/publication_4h_spot_threshold_looser/ewma_paper_phase10_summary.json`
- `data/results/publication_4h_margin_threshold/ewma_paper_phase10_summary.json`
- `data/results/publication_4h_margin_threshold_looser/ewma_paper_phase10_summary.json`

### Observed outcome

- default 4h threshold:
  - spot: `0` trades
  - margin: `0` trades
- looser 4h threshold:
  - spot: `145` trades, `PF Net = 1.1517`, `Sharpe = 1.3640`
  - margin: `145` trades, `PF Net = 1.1346`, `Sharpe = 1.2199`
- both looser runs still failed readiness because of kill-switch activity

### Conclusion

This is one of the most useful new findings from the publication pass: the 4h track is not just “inactive or bad.” It is highly sensitive to the entry policy, and once activated it can re-enter the PF-positive region while still remaining governance-constrained.

## Public Reading Of The Log

The experiment log shows a system that is capable of finding partial edge, but is also strict enough to reject fragile or non-ready candidates. That makes the codebase more believable, not less.
