# Chronos-PLG

[![CI](https://github.com/amazadfar/chronos-plg/actions/workflows/ci.yml/badge.svg)](https://github.com/amazadfar/chronos-plg/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/amazadfar/chronos-plg?display_name=tag)](https://github.com/amazadfar/chronos-plg/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-2563a7)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-0f766e)](LICENSE)

Chronos-PLG is a governed BTC trading research platform for probabilistic forecasting, leakage-safe evaluation, realistic execution modeling, robustness testing, and paper-trading readiness decisions.

It is published as a research system, not as a finished trading product or a claim of live profitability.

![Chronos-PLG research architecture](docs/assets/system_architecture.png)

## Research Question

Can probabilistic BTC forecasts produce a positive net edge after fees, slippage, funding, and financing costs, then remain stable enough to justify capital promotion?

The current answer is mixed:

- selected EWMA configurations reach positive net profit factor and positive Sharpe
- those configurations still fail readiness checks because stability, observation depth, or kill-switch behavior remains binding
- higher-activity 1h spot and margin configurations remain economically weak
- Chronos infrastructure is implemented, but the strongest public evidence is currently EWMA-led

That boundary is central to the project. Forecast quality, backtest performance, and deployment readiness are treated as separate claims.

## Evidence Snapshot

| Experiment | Net PF | Sharpe | Trades | Research decision |
| --- | ---: | ---: | ---: | --- |
| Best inspected futures EWMA candidate | 1.1739 | 0.5653 | 76 | Iterate |
| Higher-trade futures EWMA candidate | 1.1213 | 0.4512 | 100 | Iterate |
| Fixed-window spot campaign | 0.8509 | -0.8225 | 94 | Do not promote |
| 4h spot, looser entry threshold | 1.1517 | 1.3640 | 145 | Blocked by kill switch |

The evidence supports a credible research-platform claim. It does not support a promotion-ready trading-strategy claim.

![Futures candidate comparison](docs/assets/phase10_showcase_scatter.png)

## Engineering Scope

- contract-validated market-data ingestion and dataset construction
- timestamp-safe labels and leakage-aware walk-forward evaluation
- probabilistic model stack: Random Walk, EWMA, LightGBM quantile models, and Chronos
- out-of-fold Chronos-to-LightGBM meta-model training
- quantile signals, uncertainty abstention, position sizing, and regime controls
- transition-aware execution costs: fees, slippage, funding, margin interest, and other charges
- block-bootstrap, adverse-window, regime-exclusion, and cost-stress evaluation
- model comparison, uncertainty bands, and explicit `GO / ITERATE / NO_GO` decisions
- paper-trading replay, monitoring, kill switches, readiness policy, and capital-ramp recommendations
- generated public evidence snapshots and publication figures

## What This Demonstrates

| Capability | Evidence in the repository |
| --- | --- |
| ML engineering | modular model interfaces, quantile forecasting, OOF stacking, deterministic experiment metadata |
| ML research | explicit hypotheses, matched baselines, negative-result reporting, calibration and regime analysis |
| Quant engineering | realistic cost accounting, execution transitions, market-specific constraints, walk-forward backtesting |
| MLOps / ResearchOps | CI, reproducible CLIs, frozen protocols, artifact manifests, generated evidence packs |
| Model governance | robustness gates, kill criteria, paper readiness, capital-ramp and rollback policy |

## What The Project Does Not Claim

- no guaranteed or production trading profitability
- no live exchange execution
- no claim that Chronos currently outperforms the strongest baseline
- no claim that a short positive window is sufficient evidence for deployment

## Representative Results

### Best futures candidate: gross vs net equity

![Best futures candidate equity](docs/assets/phase10_best_equity.png)

### Execution-cost decomposition

![Execution cost breakdown](docs/assets/phase10_cost_breakdown.png)

### 1h threshold calibration

![Threshold calibration](docs/assets/phase11_threshold_calibration.png)

See [Results](docs/results.md) and the generated [Public Evidence Snapshot](artifacts/public/public_evidence_snapshot.md) for the full interpretation.

## Repository Layout

```text
chronos-plg/
├── config/                  # scenario, cost, and runtime configuration
├── src/
│   ├── data/                # ingestion, contracts, labels, quality gates
│   ├── models/              # baselines, Chronos, and meta-models
│   ├── evaluation/          # walk-forward, calibration, ranking, campaigns
│   ├── backtest/            # execution-cost-aware engine and reports
│   ├── strategy/            # signals, sizing, regimes, execution intent
│   ├── robustness/          # stress tests and kill criteria
│   ├── paper_trading/       # replay, monitoring, readiness, capital ramp
│   └── reporting/           # decision and promotion reporting
├── scripts/                 # reproducible CLI workflows
├── tests/                   # automated validation
├── docs/                    # GitHub Pages research publication
└── artifacts/public/        # curated evidence snapshots
```

## Quick Start

Create an isolated environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Validate the repository:

```bash
make test
make smoke
```

Regenerate the public evidence pack:

```bash
make public-assets
```

The full local dataset and experiment store are intentionally excluded from Git. See [Reproducibility](docs/reproducibility.md) and [Data Policy](data/README.md).

## Research Documentation

- [Project site](https://amazadfar.github.io/chronos-plg/)
- [Case study](docs/case-study.md)
- [Portfolio and resume copy](docs/portfolio-copy.md)
- [Methodology](docs/methodology.md)
- [Results](docs/results.md)
- [Experiment log](docs/experiment-log.md)
- [Project status](docs/project-status.md)
- [Roadmap](docs/roadmap.md)
- [Reproducibility](docs/reproducibility.md)

## License

MIT. See [LICENSE](LICENSE).
