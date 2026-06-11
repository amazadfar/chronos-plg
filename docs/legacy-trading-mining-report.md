# Legacy Trading Archive Mining Report

Date: 2026-06-11  
Scope: `/home/namiral/Projects/Inactive/Trading/legacy-trading` -> Chronos-PLG research upgrades  
Status: Initial evidence pass with one low-risk feature port implemented and benchmarked

## Executive Summary

The legacy archive is useful, but not as a source of directly reusable production code. It is an 11 GB research archive with about 54.5K files, nested repos, data dumps, notebooks, scripts, virtual environments, local settings, and credentials. The right use is hypothesis mining: extract mature ideas, rewrite them inside Chronos-PLG's causal/evidence-based pipeline, then validate every imported concept with leakage tests and walk-forward experiments.

The first implemented carry-over is a causal technical-feature layer for Chronos-PLG. It ports the strongest low-risk ideas from the old indicator and feature-engineering work: price micro-features, moving-average distances and slopes, RSI, ATR, Bollinger width/position, MACD, OBV, VROC, stochastic oscillator, CCI, ADX, Heikin-Ashi body features, Fibonacci range position, and a periodic moving-average trend signal.

## Archive Shape

Facts from the inventory pass:

- Total archive size: about 11 GB.
- Total files: about 54.5K.
- Major top-level clusters: `trend-analysis`, `trend-analysis-else`, `InfinityStrategies`, `InfinityLaunch*`, `InfinityDev`, `ml-trader`, `ml-trader-light`, `triangular-arbitrage`, `triangular-arbitrage-v0`, `pred-market`, `option-trade-set-up`, `very_old/tse-docs`, `persian-news-classifier-prot`, `fx_rt_news`, `trading-interface`, `trading-misc`.
- After excluding obvious generated/heavy folders (`.git`, data dumps, venvs, caches, logs, IDE files), there are still about 10.5K files, including about 955 Python files, 68 notebooks, 146 Markdown files, thousands of CSV/XLSX artifacts, and many config files.
- Secret risk is real: the archive contains `.env`, `keys.py`, settings, Telegram, and exchange credential files. Do not publish or copy this archive without secret scanning and redaction.

## Useful Idea Clusters

### 1. Technical Feature Engineering

Representative sources:

- `trend-analysis-else/src/indicators.py`
- `ml-trader-light/analysis/indicators.py`
- `ml-trader-light/feature_compilers/technical.py`

Useful concepts:

- OHLC-derived micro-features: `hlc3`, `ohlc4`, range, candle body, close position in range.
- Trend/momentum indicators: SMA, EMA, WMA, RSI, MACD, ADX, CCI, stochastic oscillator.
- Volatility/range indicators: ATR, Bollinger bands, Fibonacci rolling range position.
- Volume features: OBV, volume rate of change, volume z-score.
- Interaction and lag ideas: `rsi * macd`, `cci * adx`, MA-volume interactions, lagged close/volume/indicator values.

Decision:

- Ported a clean, dependency-light subset into Chronos-PLG as `src/data/technical_features.py`.
- Did not copy old code directly because the legacy versions use mixed naming, mutable in-place transforms, occasional registration mistakes, and no explicit leakage contract.

### 2. Pivot and Market Structure State Machines

Representative sources:

- `trend-analysis-else/src/pivots.py`
- `trend-analysis-else/src/market_structure.py`
- `ml-trader-light/feature_compilers/segment_identification.py`
- `InfinityStrategies/trend_analysis/*`

Useful concepts:

- Pivot streams as a structural representation of trend/range transitions.
- Explicit state machine with labels like bullish, bearish, range-after-bullish, range-after-bearish.
- Trade/event logs with entry/exit summaries and balance trajectories.
- Segment sequence search over pivot transitions.

Risk:

- Several pivot implementations use future candles to confirm pivots or open/close at post-hoc pivot points. That can be valid for offline labels, but it is dangerous as a real-time feature unless confirmation lag is modeled explicitly.

Recommended Chronos-PLG port:

- Build a causal `MarketStructureRegimeDetector` that only confirms a pivot after a configurable lag.
- Emit state/regime features, not trades, in the first pass.
- Add leakage tests that mutate future candles and assert prior regime labels do not change before the confirmation horizon.

### 3. Microstructure and Execution Quality

Representative sources:

- `triangular-arbitrage/src/strategy/microstructure.py`
- `triangular-arbitrage/src/strategy/spread_regime.py`
- `triangular-arbitrage/src/strategy/venue_reliability.py`
- `triangular-arbitrage/src/strategy/fee_token_optimizer.py`
- `triangular-arbitrage/src/forensics/baseline_analysis.py`

Useful concepts:

- Depth imbalance and microprice bias.
- Toxic spread regime classification using EWMA spread/deviation with sticky recovery.
- Venue reliability scoring based on failures, gate rejections, spreads, and utilization.
- Fee-token inventory optimization and fee-sensitivity analysis.
- Depth headroom and route utilization metrics.

Recommended Chronos-PLG port:

- Add optional market microstructure features for datasets with order-book/trade logs.
- Add a `SpreadRegimeClassifier`-style execution stress feature to paper-trading readiness.
- Add execution-quality reporting: realized spread, estimated slippage, depth utilization, and toxic-spread count.

### 4. Options and Distributional Forecasting

Representative sources:

- `option-trade-set-up/Function_Slaves.py`
- `option-trade-set-up/Option_Analysis.py`
- `very_old/tse-docs/Option Setup/*`

Useful concepts:

- Black-Scholes call/put pricing.
- Historical volatility/tolerance calculations over annual, half-annual, and quarter-annual windows.
- Option monitoring and volume screens.

Recommended Chronos-PLG port:

- Do not port Tehran-specific scrapers into the BTC project.
- Reuse the concept as a distribution-calibration track: compare Chronos quantiles against implied/realized volatility bands and add volatility-targeted abstention rules.

### 5. Prediction Market Bracket Calibration

Representative sources:

- `pred-market/btc_bracket_predictor.py`
- `pred-market/correlation_metric.py`

Useful concepts:

- Align external market-implied probabilities with BTC minute prices.
- Convert volatility envelopes into bracket probabilities.
- Evaluate whether market-implied ranges add information beyond OHLCV.

Recommended Chronos-PLG port:

- Treat prediction-market prices as optional external probability features.
- Use them as calibration diagnostics, not trading signals, until data availability and timestamp integrity are proven.

### 6. NLP / Event Signals

Representative sources:

- `persian-news-classifier-prot/README.md`
- `fx_rt_news/src/fx_calendar/*`

Useful concepts:

- News/category classification lineage.
- Calendar event extraction and alerting.
- Event-window flags.

Recommended Chronos-PLG port:

- Expand existing FOMC/CPI flags into a macro-event feature provider.
- Keep Telegram/notifier code out of Chronos-PLG.
- Later: add crypto-specific news/sentiment only if timestamped, source-attributed, and backtestable.

## First Port Implemented

Files added/changed:

- `src/data/technical_features.py`
- `src/data/build_dataset.py`
- `tests/test_technical_features.py`

Implemented features:

- Price/candle features: `tech_hlc3`, `tech_ohlc4`, `tech_range_pct`, `tech_body_pct`, wick ratios, close-in-range.
- Trend features: MA distance/slope, periodic MA signal.
- Momentum/range features: RSI, MACD, Bollinger position/width, stochastic, CCI, ADX.
- Volatility features: ATR and ATR percentage.
- Volume features: OBV and VROC.
- Candle transform features: Heikin-Ashi close/body.
- Rolling range feature: Fibonacci-window price position.

Validation performed:

- Focused tests: `26 passed`.
- Project lint gate: passed.
- Syntax compile: passed.

## First Benchmark Result

The first matched ablation used the rebuilt 1h processed dataset and the spot taker-discounted scenario:

```bash
./.venv/bin/python scripts/run_feature_ablation.py \
  --timeframe 1h \
  --feature-sets core,all \
  --scenario binance_spot_taker_discounted \
  --no-progress \
  --output-dir data/results/legacy_feature_ablation_1h_spot_core_all_v2
```

Feature sets:

- `core`: 28 non-technical benchmark features after sparse unusable market-structure columns were excluded.
- `all`: 73 features total, combining 28 core features with 45 legacy-derived `tech_*` features.

Result:

| Feature set | Model | Sharpe | Net PF | Total return | Max drawdown | Trades | Folds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `core` | LightGBM | -0.8410 | 0.2113 | -3.53% | -3.53% | 66 | 242 |
| `all` | LightGBM | -1.0965 | 0.0679 | -2.56% | -2.56% | 44 | 242 |
| `core` | EWMA | -0.2528 | 0.8329 | -1.81% | -2.90% | 191 | 242 |
| `all` | EWMA | -0.2528 | 0.8329 | -1.81% | -2.90% | 191 | 242 |

Interpretation:

- The technical feature bundle trained successfully, but it did not improve the current LightGBM policy under this protocol.
- EWMA is unchanged because it does not consume feature columns.
- RandomWalk remains a no-trade reference and should not be treated as the best active strategy.
- This is a prune-or-refine signal, not a profitability signal. The next useful step is family-level ablation and/or less conservative signal gating before promoting any legacy-derived feature family.

## Risks and Controls

### Leakage Risk

Impact: falsely profitable backtests.

Control:

- No post-hoc pivot features in the first port.
- Added a causality test that mutates future candles and confirms earlier feature rows remain unchanged.

### Secret Exposure Risk

Impact: credential leakage if legacy files are copied or published.

Control:

- Do not publish legacy archive.
- Any future extracted snippets must pass secret scanning.
- Rotate old-looking exchange/Telegram credentials if there is any chance they are still valid.

### Narrative Risk

Impact: public project claims could overstate profitability.

Control:

- Maintain the current Chronos-PLG framing: governed research platform, not a live profitable bot.
- New ideas must enter through evidence gates and generated public reports.

## Highest-Value Next Candidates

1. Family-level technical feature ablation to identify whether any subset is useful despite the full bundle underperforming.
2. Causal market-structure regime detector.
3. Microstructure/readiness upgrade inspired by the triangular-arbitrage engine.
4. Experiment registry/model cards for every mined idea.
5. Optional external probability/event features once timestamp integrity is proven.
