# Chronos-PLG

Chronos-PLG is a BTC trading research system built around probabilistic forecasts, realistic execution costs, walk-forward evaluation, robustness gates, and paper-trading governance.

This project is intentionally published as research, not as a claim of finished profitability.

## What Makes It Interesting

- it treats costs and execution realism as first-class concerns
- it promotes `ProfitFactorNet` and readiness gates over cosmetic headline metrics
- it records both positive and negative evidence
- it includes promotion-prevention logic, not just strategy logic

## Current Reading of the Evidence

- a tuned futures EWMA configuration reached `PF Net > 1.0` and positive Sharpe
- that same candidate still failed readiness / promotion
- 1h spot and margin calibration runs increased activity but did not cross profitability thresholds
- the fixed campaign evidence remains negative and correctly recommends no promotion

## Evidence Snapshot

![Phase 10 showcase scatter](assets/phase10_showcase_scatter.png)

![Best phase 10 futures candidate equity](assets/phase10_best_equity.png)

![1h threshold calibration](assets/phase11_threshold_calibration.png)

## What Is In The Repo

- data ingestion, contracts, and quality gates
- leakage-aware labels and walk-forward evaluation
- baseline models and a Chronos-2 candidate path
- cost-aware backtesting and reporting
- robustness and kill criteria
- paper-trading replay, readiness policy, and capital-ramp logic
- publication scripts for public summaries and figures

## Read Next

- [Methodology](methodology.md)
- [Project status](project-status.md)
- [Results](results.md)
- [Experiment log](experiment-log.md)
- [Roadmap](roadmap.md)
- [Reproducibility](reproducibility.md)
