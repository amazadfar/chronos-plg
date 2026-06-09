# 04 - Models, Strategy, And Cost Mechanics

## Forecasting Models

### RandomWalkBaseline (`src/models/baselines/random_walk.py`)

Definition:

- `q50 = 0`
- `q10`, `q90` from rolling historical quantiles

Use case:

- Minimum viability benchmark; if this is not beaten net of costs, there is no deployable edge

### EWMABaseline (`src/models/baselines/ewma.py`)

Definition:

- EWMA mean and std from return series
- Quantiles from Gaussian mapping: `q = mean + z(q)*std`

Practical behavior:

- Captures short-term drift and volatility regime changes
- This is currently the strongest performer in Phase-10 artifacts

### LightGBMQuantileBaseline (`src/models/baselines/lightgbm_quantile.py`)

Definition:

- Separate LightGBM models per quantile objective (`alpha = q`)
- Uses filtered numeric features excluding leakage-prone columns

Practical behavior:

- Strong tabular baseline in design, but currently underperforming in latest Phase-10 forward replay

### Chronos2Runner (`src/models/chronos2_runner.py`)

Definition:

- API-compatible quantile forecaster with strict rolling OOS inference
- For each test timestamp, only historical train targets plus prior predictions are used
- Updates rolling context with predicted `q50` (never realized test target)

Safety behavior:

- Prevents contamination by design during predict loop
- Falls back to deterministic empirical quantiles if Chronos backend unavailable

### MetaModel (`src/models/meta_model.py`)

Definition:

- Stage 1: Chronos quantiles (OOF during training)
- Stage 2: LightGBM quantile models over raw features + Chronos-derived features

Safety behavior:

- Stage-2 training uses OOF Chronos predictions instead of in-sample predictions
- Final Chronos fit occurs only after OOF generation

## Trading Strategy Definition

Core strategy implementation: `src/strategy/strategy.py`

### Trade Decision Logic (`src/strategy/signals.py`)

Inputs: `q10`, `q50`, `q90`

Derived:

- `uncertainty = q90 - q10`
- `confidence` decreases as uncertainty widens
- `strength` increases with absolute `q50` relative to threshold

Rules:

- Long: `q50 > entry_threshold` and `q10 > -risk_limit`
- Short: `q50 < -entry_threshold` and `q90 < risk_limit`
- Flat: no edge or uncertainty gate fails

Adaptive option:

- Entry threshold can be multiplied per regime (`trend/normal/chop/panic`)

### Regime Logic (`src/strategy/regime_detector.py`)

Regimes from 7-day return and vol:

- `panic`: very high vol
- `trend`: large directional move with controlled vol
- `chop`: low move + low vol
- `normal`: default

Output:

- Regime label + per-regime sizing multipliers

### Position Sizing (`src/strategy/position_sizing.py`)

Sizing mechanics:

- Base: `signal * strength * confidence`
- Vol target scaling by annualized predicted vol
- Clip by leverage cap (market-specific)
- Optional constraints:
  - short availability gating
  - lot size / min qty / min notional enforcement
  - per-step turnover cap
  - minimum position cutoff

### Execution Intent (`src/strategy/execution_intent.py`)

Transition classes:

- `hold`, `open`, `increase`, `reduce`, `close`, `reverse`

Execution policy classes:

- `taker_only`
- `maker_preferred`
- `hybrid` (taker on adds/reverses, maker on reductions/closes)

## Cost Engine Definition (`src/backtest/costs.py`)

### Event-Level Cost Decomposition

For each timestamp transition:

- Identify open/close notional legs based on position transition
- Compute:
  - Fees by scenario (`exchange/market/order/discount`)
  - Slippage: base bps + volatility term + optional size-impact term
  - Funding (signed by prior held position)
  - Margin interest (borrow rate * held notional * holding time)
  - Other costs (proportional bps + fixed + explicit extras)

Outputs include per-event audit columns:

- `event_type`, `open_notional`, `close_notional`, `traded_notional`
- `fees`, `slippage`, `funding`, `interest`, `other_costs`, `total_costs`

### Net Return Accounting

Backtest/paper return formula:

- `gross_return_t = position_{t-1} * realized_return_t`
- `net_return_t = gross_return_t - total_costs_t`

## Paper Trading Operational Logic

Implemented in:

- `src/paper_trading/engine.py`
- `src/paper_trading/monitoring.py`
- `src/paper_trading/policy.py`

Key mechanisms:

- Sequential retrain-and-replay loop with strict past-only training windows
- Same cost model assumptions as backtests (no dual logic)
- Monitoring windows (daily/weekly) with PF, Sharpe, DD, turnover, and cost decomposition
- Kill switch triggers on configurable threshold breaches
- Deployment readiness checks (minimum observation days, trades, PF/Sharpe/DD, kill events)
- Capital ramp policy with promote/hold/rollback recommendations

## Strategy Families Currently In Practice

From current artifacts, active strategy variants are primarily EWMA with parameter sweeps across:

- `ewma_span`
- `entry_threshold`
- `retrain_interval_bars`
- regime-specific entry multipliers

This implies the current operational frontier is parameterized quantile-threshold strategy variants over the same base signal/sizing/cost stack, not fundamentally different strategy classes.
