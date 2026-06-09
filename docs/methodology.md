# Methodology

## Research Objective

The project asks whether BTC forecasting signals can survive realistic trading frictions and governance constraints, not just whether they can predict direction in a vacuum.

The primary success gate is:
- `ProfitFactorNet > 1.0`

Secondary criteria include:
- acceptable net Sharpe
- bounded drawdown
- regime stability
- sufficient observation depth
- acceptable kill-switch behavior

## Data

The data layer is built to support:
- OHLCV price data
- funding data
- open interest
- liquidation features
- macro covariates
- event flags

Key implementation themes:
- contract validation
- timestamp alignment
- explicit feature-family availability tracking
- quality reports for gaps, duplicates, null coverage, and degraded runs

The public repo does not ship the entire local dataset store. See [reproducibility](reproducibility.md).

## Targets and Forecasting

The forecasting design is quantile-oriented rather than point-estimate-only:
- `q10`
- `q50`
- `q90`

This allows the strategy layer to reason about:
- directional expectation
- uncertainty spread
- risk-conditioned entry logic

## Evaluation Protocol

The core evaluation protocol is walk-forward and out-of-sample by construction.

Implemented themes include:
- frozen fold schedules
- minimum training windows
- non-overlapping train / test boundaries
- leakage guardrails
- candidate comparison against baselines before advancement

## Cost and Execution Realism

The backtest and paper-trading layers center the cost engine rather than bolting it on afterward.

Supported cost categories include:
- fees
- slippage
- funding
- margin interest
- other scenario-level costs

The execution model also tracks transition-aware trade events such as:
- open
- increase
- reduce
- close
- reverse

## Strategy and Governance

The strategy layer converts forecast distributions into:
- entry decisions
- abstention decisions
- position sizes
- regime-aware constraints

The governance layer then evaluates whether a candidate deserves advancement. This includes:
- kill criteria
- robustness tests
- paper-trading replay
- monitoring dashboards
- deployment readiness checks
- capital-ramp recommendations

This is one of the main design choices of the project: good-looking metrics are not enough if the governance layer says the candidate is still fragile.

## Current Methodological Limitation

The biggest practical limitation in the current research cycle is data completeness for some higher-value feature families, especially open-interest / liquidation coverage over the target windows. That limitation is explicitly tracked and treated as research debt rather than ignored.

## Source Files

Relevant implementation areas:
- `src/data/`
- `src/evaluation/`
- `src/backtest/`
- `src/strategy/`
- `src/robustness/`
- `src/paper_trading/`
- `src/reporting/`
