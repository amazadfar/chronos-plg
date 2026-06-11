# Implementation Plan

## Title

Legacy Trading Archive Signal Mining and Chronos-PLG Research Upgrade

## Metadata

- Date: 2026-06-11
- Author: Codex
- Status: In Progress
- Related issue / task: Mine `/home/namiral/Projects/Inactive/Trading/legacy-trading` for reusable trading ideas and port the strongest candidates into Chronos-PLG.
- Related analysis report: `docs/legacy-trading-mining-report.md`
- Related PRs: None yet
- Related docs: `README.md`, `docs/results.md`, `docs/project-status.md`, `docs/roadmap.md`

## Executive Summary

The legacy trading archive should become a research-hypothesis mine for Chronos-PLG. The archive contains many useful ideas across technical indicators, market structure, microstructure, execution governance, prediction-market calibration, options, TSE ingestion, and NLP/event workflows, but it also contains generated data, old virtual environments, nested repositories, hardcoded credentials, and hindsight-prone experimental scripts.

The implementation strategy is to extract ideas, not code. Each candidate must be rewritten inside Chronos-PLG's existing causal data pipeline, validated with leakage tests, benchmarked through walk-forward experiments, and published only as evidence-backed research.

## Problem / Opportunity Statement

Chronos-PLG is currently a credible governed trading research platform, but its strongest public evidence is EWMA-led. The legacy archive contains years of trading experiments that can broaden the feature set, add regime intelligence, improve execution-readiness controls, and create a richer portfolio story. The challenge is to convert messy historical experiments into clean, testable, public-safe research upgrades without importing leakage, secrets, or unsupported profitability claims.

## Goals

- [x] Inventory the legacy archive at a high level.
- [x] Identify reusable idea clusters and implementation risks.
- [x] Port one low-risk candidate into Chronos-PLG.
- [ ] Benchmark the new technical-feature set against the current evidence baseline.
- [ ] Add a causal market-structure regime detector.
- [ ] Add microstructure/readiness ideas from the triangular-arbitrage engine.
- [ ] Build continuous experiment cards for all mined ideas.

## Non-Goals

- [ ] Publish the legacy archive as-is.
- [ ] Copy old implementations directly into Chronos-PLG.
- [ ] Claim profitability before cost-aware walk-forward and paper-readiness gates pass.
- [ ] Add live execution.
- [ ] Add TSE-specific scrapers to the BTC research pipeline.

## Context and Background

Chronos-PLG already has modular seams for this work:

- `src/data`: ingestion, dataset construction, labels, quality gates.
- `src/models`: EWMA, LightGBM quantile, Chronos, meta-models.
- `src/evaluation`: walk-forward, candidate gates, multi-objective ranking.
- `src/strategy`: signals, sizing, regime detector.
- `src/backtest`: cost-aware execution simulation.
- `src/paper_trading`: monitoring, readiness, kill switches, capital ramp policy.
- `docs` and `artifacts/public`: public evidence publication.

The legacy archive spans about 11 GB and about 54.5K files. A source-focused pass after pruning heavy/generated folders still found about 10.5K files, including roughly 955 Python files and 68 notebooks.

## Codebase / System Analysis Summary

- Areas inspected:
  - Chronos-PLG dataset builder, current feature computation, regime detector, tests, README, pyproject.
  - Legacy trend-analysis, indicator, market-structure, ML model, triangular-arbitrage, prediction-market, options, TSE, Persian news, and FX calendar clusters.
- Relevant modules / services:
  - `src/data/build_dataset.py`
  - `src/data/technical_features.py`
  - `src/strategy/regime_detector.py`
  - `scripts/run_baselines.py`
  - `scripts/run_chronos2.py`
  - `scripts/run_phase11_sweep.py`
  - `scripts/run_paper_trading.py`
- Current behavior:
  - Chronos-PLG computes basic return, volatility, ATR, volume, funding, OI, liquidation, macro, and event features.
  - Chronos-PLG validates leakage and publishes curated evidence.
- Known pain points:
  - Chronos-specific public edge is not yet the strongest evidence.
  - Feature space was conservative before this pass.
  - Legacy pivot and segment code has hindsight risk.
  - Legacy archive contains credentials and generated data.
- Architecture notes:
  - Feature expansion belongs in `src/data`.
  - Strategy/regime expansion belongs in `src/strategy`.
  - Execution quality/readiness expansion belongs in `src/backtest`, `src/robustness`, and `src/paper_trading`.

## Dependency and Interface Impact

- Internal dependencies:
  - New feature module is imported by `DatasetBuilder.compute_features`.
- External dependencies:
  - No new dependency added for the first port.
- API / schema / contract impact:
  - Processed datasets gain new `tech_*` feature columns.
  - Existing labels and OHLCV columns are unchanged.
- Build / deployment impact:
  - No deployment impact.
  - Existing test/lint flow remains valid.
- Backward compatibility considerations:
  - Old experiment artifacts are unaffected.
  - New training runs will have a wider feature set unless feature columns are pinned by older configs.

## Constraints and Assumptions

### Constraints

- The public repo must remain safe: no secrets, no raw private/archive data, no unsupported profitability claims.
- All feature ports must be causal relative to the prediction timestamp.
- No live execution changes are allowed in this milestone.

### Assumptions

- The correct strategy is to port ideas, not legacy code.
- The first benchmark should test the new feature set with existing model/evaluation machinery before adding more model complexity.
- Existing CI and smoke tests are the baseline validation contract.

## Risks and Failure Modes

### Leakage Through Structural Features

- Impact: inflated model performance and invalid public evidence.
- Detection: future-row mutation tests, walk-forward boundary checks, no-leak validation.
- Mitigation: keep post-hoc pivot labels out of model inputs until causal confirmation lag is implemented.

### Secret Exposure From Legacy Archive

- Impact: leaked exchange or Telegram credentials.
- Detection: secret scanning and denylist checks before any copy/publication.
- Mitigation: never publish legacy archive; port manually reviewed concepts only.

### Feature Bloat Without Edge

- Impact: overfitting, slower experiments, weaker public story.
- Detection: matched benchmark against current evidence and ablation tables.
- Mitigation: require model-card artifacts and prune weak feature families.

### Narrative Drift

- Impact: resume/GitHub claims exceed evidence.
- Detection: public docs review and generated evidence snapshot.
- Mitigation: continue separating research-platform, forecast-quality, strategy-performance, and deployment-readiness claims.

## Proposed Approach

Use a four-lane research pipeline:

1. Archive mining lane: inventory, classify, and rank legacy ideas by portability and expected value.
2. Feature/regime lane: port causal indicators and market-structure states into `src/data` and `src/strategy`.
3. Evaluation lane: run matched walk-forward/model comparisons and ablations.
4. Publication lane: generate experiment cards, update docs, and cut evidence releases.

## Alternatives Considered

### Alternative A: Import Legacy Modules Directly

- Description: Add legacy folders to the Chronos-PLG import path.
- Why not chosen: high leakage, secret, dependency, and style risk; unclear tests; many old scripts are exploratory.

### Alternative B: Start With Chronos Model Architecture Work

- Description: Skip feature mining and go directly to fine-tuning/ensembling Chronos.
- Why not chosen: Chronos evidence needs stronger inputs and cleaner experiment tracking first. Feature and regime upgrades are cheaper and create useful ablation baselines.

## Phase Plan

### Phase 1: Archive Inventory and First Safe Feature Port

- Objective: Identify high-value clusters and port a low-risk technical-feature bundle.
- Why now: It creates immediate model-input expansion without touching live execution.
- Type: Sequential
- Depends on: Current Chronos-PLG data pipeline and tests.
- Deliverables:
  - `docs/legacy-trading-mining-report.md`
  - `src/data/technical_features.py`
  - focused tests
- Checklist:
  - [x] Inventory archive shape.
  - [x] Inspect representative source clusters.
  - [x] Add causal technical features.
  - [x] Wire features into `DatasetBuilder`.
  - [x] Add causality tests.
- Validation:
  - `./.venv/bin/pytest -q tests/test_technical_features.py tests/test_data_pipeline.py`
  - `make lint`
  - `./.venv/bin/python -m py_compile src/data/technical_features.py src/data/build_dataset.py`
- Exit criteria:
  - New features are integrated and tested.
- Green-light cue:
  - Proceed to benchmark campaign.
- Rollback / containment:
  - Revert `src/data/technical_features.py` and its import/use from `DatasetBuilder`.

### Phase 2: Technical Feature Benchmark Campaign

- Objective: Measure whether the new `tech_*` features improve EWMA, LightGBM quantile, Chronos-meta, and ensemble candidates.
- Why now: Feature ports only matter if they improve cost-aware evidence.
- Type: Sequential
- Depends on: Phase 1.
- Deliverables:
  - Matched old-vs-new feature benchmark.
  - Ablation table by feature family.
  - Public-safe experiment card.
- Checklist:
  - [ ] Freeze baseline data/timeframe/scenarios.
  - [ ] Run current feature baseline.
  - [ ] Run technical-feature candidate.
  - [ ] Run ablations: momentum, volume, volatility, candle/range.
  - [ ] Compare net PF, Sharpe, trade count, drawdown, kill events, readiness.
- Validation:
  - Existing tests plus benchmark reproducibility hash.
- Exit criteria:
  - Clear keep/prune decision per feature family.
- Green-light cue:
  - At least one feature family improves a primary or secondary metric without readiness regression, or a documented negative result explains why not.
- Rollback / containment:
  - Keep module but exclude weak feature families from model configs.

### Phase 3: Causal Market-Structure Regime Detector

- Objective: Port the pivot/state-machine idea without hindsight leakage.
- Why now: The legacy archive's most distinctive alpha hypothesis is market-structure transitions, not generic indicators.
- Type: Sequential
- Depends on: Phase 2 decision.
- Deliverables:
  - `src/strategy/market_structure.py` or `src/data/market_structure_features.py`
  - confirmation-lag pivot detector
  - regime/state features
  - tests for future mutation and boundary behavior
- Checklist:
  - [ ] Define causal pivot confirmation contract.
  - [ ] Implement state transitions with explicit lag.
  - [ ] Add synthetic trend/range tests.
  - [ ] Add no-future-mutation tests.
  - [ ] Benchmark as feature/regime gate.
- Validation:
  - Unit tests, leakage tests, walk-forward benchmark.
- Exit criteria:
  - State labels are stable, causal, and useful or explicitly rejected.
- Green-light cue:
  - Market-structure states improve readiness or reduce false trades.
- Rollback / containment:
  - Keep detector behind optional feature config until validated.

### Phase 4: Microstructure and Execution Quality Upgrade

- Objective: Import the triangular-arbitrage engine's strongest execution-quality ideas into Chronos-PLG readiness controls.
- Why now: Profitability often dies in execution, and this improves the portfolio story beyond model prediction.
- Type: Parallelizable after Phase 2
- Depends on: Dataset availability for order-book/trade data or simulated proxies.
- Deliverables:
  - Spread regime classifier.
  - Depth/microprice feature interfaces.
  - Execution-quality report fields.
  - Readiness gate extensions.
- Checklist:
  - [ ] Define available microstructure data contract.
  - [ ] Implement spread toxicity proxy for current OHLCV-only runs.
  - [ ] Add optional order-book feature provider.
  - [ ] Add readiness report fields.
  - [ ] Add public-safe execution-quality chart.
- Validation:
  - Unit tests and paper-trading replay comparison.
- Exit criteria:
  - Execution-quality metrics are reproducible and incorporated into readiness.
- Green-light cue:
  - Readiness decisions become more explainable without masking weak returns.
- Rollback / containment:
  - Keep microstructure features optional.

### Phase 5: Experiment Cards and Continuous Publication

- Objective: Turn every mined idea into a traceable public artifact.
- Why now: This is the portfolio/research multiplier.
- Type: Parallelizable after Phase 2
- Depends on: Stable benchmark command surface.
- Deliverables:
  - Experiment card JSON/Markdown generator.
  - Model/feature-family comparison page.
  - Release checklist update.
- Checklist:
  - [ ] Define experiment-card schema.
  - [ ] Capture config hash, data span, feature families, model, scenario, costs, metrics, gates.
  - [ ] Generate Markdown cards for public docs.
  - [ ] Add one card per mined idea.
- Validation:
  - Schema tests and `make public-assets`.
- Exit criteria:
  - New research runs automatically produce comparable evidence.
- Green-light cue:
  - Release `v0.3-legacy-signal-mining`.
- Rollback / containment:
  - Artifact generation can be disabled without changing model behavior.

## Milestones

### Milestone 1: Safe Signal Mining Foundation

- Target outcome: Archive mapped and first feature bundle integrated.
- Includes phases: Phase 1.
- Completion criteria: Tests and lint pass; mining report exists.
- Evidence / demo expected: New `tech_*` columns and causality test.
- Go / no-go note: Completed enough to benchmark.

### Milestone 2: Evidence-Based Feature Decision

- Target outcome: Know whether legacy-inspired technical features improve Chronos-PLG.
- Includes phases: Phase 2.
- Completion criteria: Matched benchmark and feature-family ablation.
- Evidence / demo expected: Public experiment card and comparison table.
- Go / no-go note: Keep only feature families with evidence or research value.

### Milestone 3: Distinctive Market Structure Research

- Target outcome: Causal pivot/state-machine regime features.
- Includes phases: Phase 3.
- Completion criteria: Causal tests and benchmark decision.
- Evidence / demo expected: Market-structure chart plus readiness impact.
- Go / no-go note: Reject if leakage-safe version loses signal.

### Milestone 4: Execution-Aware Research Platform

- Target outcome: Microstructure/readiness upgrades.
- Includes phases: Phase 4.
- Completion criteria: Execution-quality metrics appear in paper-trading reports.
- Evidence / demo expected: Spread/toxicity/readiness chart.
- Go / no-go note: Only promote if data contract is reliable.

### Milestone 5: Continuous Public Research Engine

- Target outcome: Every run generates a public-safe evidence card.
- Includes phases: Phase 5.
- Completion criteria: Automated artifact cards and docs integration.
- Evidence / demo expected: Release-ready docs page.
- Go / no-go note: Release only after docs state claims boundaries clearly.

## Validation Strategy

### Automated validation

- Unit tests: feature calculators, causal mutation tests, regime transitions.
- Integration tests: `DatasetBuilder.compute_features`, walk-forward data contracts.
- End-to-end tests: smoke checks, benchmark CLIs, public asset generation.
- Static analysis / lint / type checks: current `make lint`; broader Ruff cleanup should be separate.

### Manual validation

- Inspect generated feature distributions for NaNs/infs/outliers.
- Inspect equity curves and kill-switch paths.
- Review public docs for claim discipline.

### Acceptance thresholds

- Functional: new features build for all supported timeframes.
- Performance: benchmark runtime stays practical for local research.
- Reliability: no leakage test failures.
- Security: no legacy credentials or raw archive files enter the public repo.
- Usability / DX: one command should reproduce public assets.

## Rollout / Migration Strategy

- Rollout pattern: feature-first, benchmark-second, publish-third.
- Feature flags: not required for the first port, but optional feature-family configs should be added before large experiments.
- Backward compatibility: existing artifacts remain readable; new runs add feature columns.
- Data migration: none.
- Rollback path: remove new feature module import or restrict feature columns in model configs.
- Safe deployment notes: no live trading/execution changes.

## Observability / Debugging Plan

- Logs: dataset builder logs feature counts.
- Metrics: feature counts, null rates, model metrics, gate outcomes.
- Traces: experiment manifests and future experiment cards.
- Alerts: not applicable for offline research.
- Debug hooks: generated quality reports and public evidence snapshots.
- Failure signals: leakage validation failure, degraded-run failure, poor ablation metrics, readiness regression.

## Open Questions

- [ ] Which legacy data, if any, should be preserved as private local research data?
- [ ] Should market-structure features be used as model inputs, strategy gates, or both?
- [ ] Should prediction-market brackets become a separate research branch?
- [ ] Are any old exchange/Telegram credentials still valid and needing rotation?

## Decision Log

### Decision 1

- Decision: Port concepts, not code.
- Why: Legacy code has mixed quality, secret risk, and potential hindsight leakage.
- Alternatives rejected: direct import of legacy modules.
- Implications: Slower upfront, safer long-term.

### Decision 2

- Decision: Start with causal technical features.
- Why: Lowest-risk high-value feature expansion; maps directly to current data pipeline.
- Alternatives rejected: pivot state machine first, Chronos fine-tuning first.
- Implications: Benchmarking can start immediately with existing CLIs.

### Decision 3

- Decision: Keep pivot/state-machine features out of the first implementation.
- Why: Existing legacy pivot logic is likely valid only with confirmation lag or offline labels.
- Alternatives rejected: add post-hoc pivots as features.
- Implications: Market-structure work gets its own leakage contract.

## Implementation Log

| Date | Phase / Step | What was done | Files / Modules touched | Validation run | Outcome | Notes / blockers |
|---|---|---|---|---|---|---|
| 2026-06-11 | Phase 1 | Inventory pass over legacy archive and representative files | `docs/legacy-trading-mining-report.md` | N/A | Completed | Archive contains secrets; do not publish raw archive |
| 2026-06-11 | Phase 1 | Added causal technical feature module and integrated with `DatasetBuilder` | `src/data/technical_features.py`, `src/data/build_dataset.py`, `tests/test_technical_features.py` | focused pytest, lint, py_compile | Passed | No new dependencies |

## Final Approval Gate

- [x] Scope is clear
- [x] Critical unknowns are resolved for Phase 1
- [x] Plan is evidence-based
- [x] Phases are actionable
- [x] Validation is defined
- [x] Rollout / rollback is defined where relevant
- [x] Parallelizable work is clearly marked
- [x] Ready for next implementation phase
