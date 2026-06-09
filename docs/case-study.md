# Case Study

## Objective

Build a research system that can answer a harder question than “did the model predict price direction?”:

> Does a probabilistic BTC forecast create a repeatable net edge after realistic market frictions, and is the evidence strong enough to justify promotion?

## Why This Is Difficult

Trading-model evaluation is unusually vulnerable to optimistic assumptions:

- future information can leak through feature construction or split design
- gross returns can disappear after fees, slippage, funding, and financing
- one favorable regime can dominate a short evaluation window
- parameter searches can reward fragile configurations
- a good forecast does not guarantee a good trading decision

Chronos-PLG treats these as system-design problems rather than footnotes.

## System Design

![Chronos-PLG research architecture](assets/system_architecture.png)

The same research path is reused from historical evaluation through paper replay:

- market data is validated for timestamp alignment, gaps, duplicates, and feature-family availability
- models produce return quantiles rather than a single point estimate
- strategy logic converts distributions into long, short, or abstain decisions
- the execution engine applies transition-aware costs
- walk-forward evaluation and stress protocols measure stability
- governance logic decides whether a candidate should advance, iterate, or stop

## Model Stack

### Statistical baselines

Random Walk and EWMA establish inexpensive, interpretable reference points.

### Tabular quantile forecasting

LightGBM models estimate lower, median, and upper return quantiles from market and contextual features.

### Time-series foundation model track

The Chronos runner uses strict rolling out-of-sample semantics and records model/backend provenance. Fallback execution is explicitly marked and cannot silently count as real Chronos evidence.

### Meta-model

Chronos quantiles can be combined with tabular features through LightGBM. Stage-two training uses out-of-fold Chronos predictions rather than in-sample forecasts.

## Evaluation Design

- expanding or rolling walk-forward folds
- frozen baseline protocol and fold schedule
- causal timestamp boundaries
- full net-cost accounting
- block-bootstrap uncertainty
- adverse-window and regime-exclusion stress
- parameter sensitivity and Pareto-style candidate ranking
- fixed-window paper campaign

## Representative Findings

### Positive but incomplete

The best inspected futures EWMA candidate produced:

- `Net PF = 1.1739`
- `Sharpe = 0.5653`
- `76 trades`

The result was classified as `ITERATE`, not ready for promotion.

### Higher activity, weaker economics

The 1h spot and margin threshold calibration produced up to `197` trades, but net PF remained near `0.81` and Sharpe remained negative.

### Attractive metrics, governance failure

A looser 4h threshold recovered:

- spot: `Net PF = 1.1517`, `Sharpe = 1.3640`, `145 trades`
- margin: `Net PF = 1.1346`, `Sharpe = 1.2199`, `145 trades`

Both remained blocked by kill-switch behavior.

### Fixed-window rejection

The selected spot campaign produced:

- `Net PF = 0.8509`
- `Sharpe = -0.8225`
- `94 trades`
- promotion recommendation: `False`

The negative campaign is useful evidence because the system rejected the candidate instead of converting a prior positive window into an unsupported deployment claim.

## Engineering Decisions

### Costs are part of the model contract

Fees, slippage, funding, interest, and other charges are computed as first-class execution events rather than an after-the-fact haircut.

### Abstention is a valid prediction

Quantile width and expected edge can produce a no-trade decision. The policy is not forced to act on every forecast.

### Promotion requires more than one metric

Net PF is the primary edge gate, but promotion also considers Sharpe, drawdown, trade count, stability, kill events, and recent-regime behavior.

### Negative evidence is retained

Public artifacts include failed calibration and campaign results. This prevents the repository from becoming a collection of selected winners.

## What The Project Demonstrates

- designing an end-to-end ML research system rather than a single model notebook
- building leakage-safe evaluation for temporal data
- separating forecast quality, trading quality, and deployment readiness
- implementing auditable model and strategy governance
- packaging research as reproducible code, tests, CI, figures, and public evidence

## Current Limitations

- no inspected candidate is promotion-ready
- open-interest and liquidation coverage remains incomplete for important windows
- the strongest public evidence is EWMA-led rather than Chronos-led
- results remain sensitive to timeframe, threshold policy, and regime

## Next Research Questions

1. Does real-backend Chronos improve probabilistic forecast quality over the strongest baseline?
2. Can forecast calibration improvements survive downstream costs?
3. Does restored open-interest and liquidation coverage improve regime discrimination?
4. Can a learned abstention policy improve readiness without overfitting?
5. Does cross-asset context from ETH or broader markets add stable information?
