# Portfolio Copy

These variants are intentionally evidence-bounded. They present the engineering and research work without claiming a production-ready profitable strategy.

## Compact Resume Entry

**Chronos-PLG | Probabilistic Forecasting & Trading Research Platform**  
Built a governed BTC trading research platform evaluating Chronos, LightGBM quantile models, and statistical baselines with leakage-safe walk-forward validation, realistic execution costs, robustness stress tests, and paper-trading readiness controls. Public results identify PF-positive but non-promotable regimes, demonstrating rigorous model governance rather than overclaimed profitability.

## Metric-Led Resume Entry

**Chronos-PLG | Probabilistic Forecasting & Trading Research Platform**  
Developed an end-to-end BTC forecasting and trading research system with quantile models, timestamp-safe walk-forward evaluation, transition-aware cost accounting, stress testing, kill switches, and promotion gates. The strongest inspected futures candidate achieved `1.17` net profit factor and `0.57` Sharpe, but was correctly held at `ITERATE` because readiness criteria remained unmet.

## Selected-Projects Version

**Chronos-PLG | Governed Time-Series Forecasting Research**  
Research platform for testing whether probabilistic BTC forecasts survive realistic market frictions and deployment governance. Implements Random Walk, EWMA, LightGBM quantile, Chronos, and Chronos-to-LightGBM meta-model paths; leakage-safe walk-forward evaluation; fees, slippage, funding, and margin-interest modeling; block-bootstrap and regime stress tests; paper replay; kill-switch diagnostics; and generated public evidence reports. Current evidence includes positive-edge regions and deliberately published negative campaign results, with no deployment-readiness claim.

## GitHub Pinned-Repository Description

Governed BTC trading research platform with probabilistic forecasting, realistic execution costs, walk-forward evaluation, robustness testing, and paper-trading readiness controls.

## LinkedIn Project Summary

Chronos-PLG is an open research platform for evaluating probabilistic time-series models under realistic trading constraints. The project separates forecast quality, trading economics, and deployment readiness rather than treating a positive backtest as sufficient evidence.

The system includes timestamp-safe market-data contracts, quantile forecasting, Random Walk/EWMA/LightGBM/Chronos model paths, out-of-fold meta-model training, realistic execution-cost accounting, walk-forward evaluation, stress testing, kill switches, paper-trading monitoring, and explicit promotion decisions.

The current public findings are mixed by design: selected EWMA configurations reach positive net profit factor, while fixed-window and higher-activity experiments remain weak or fail readiness controls. Those negative results are retained as part of the public evidence rather than filtered out.

## Interview Framing

Use the project to discuss:

- why temporal leakage requires stronger controls than random train/test splits
- why probabilistic calibration and trading profitability are different objectives
- how costs and position transitions alter apparent model performance
- why deployment governance should be separate from model selection
- why negative results improve the credibility of an ML research system
- how the Chronos evidence gap informs the next research milestone
