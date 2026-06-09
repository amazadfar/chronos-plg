# Implementation Plan

## Title
Chronos-PLG GitHub Portfolio Publication and Research Roadmap

## Metadata
- Date: 2026-06-08
- Author: Codex
- Status: In Progress
- Related issue / task: Publish `chronos-plg` as a GitHub portfolio project that showcases ML engineering, ML/AI research, time-series forecasting, MLOps, and rigorous model governance.
- Related analysis report: In-session repository inspection on 2026-06-08.
- Related PRs: None yet.
- Related docs:
  - `README.md`
  - `PUBLICATION_NOTES.md`
  - `docs/index.md`
  - `docs/results.md`
  - `docs/project-status.md`
  - `docs/methodology.md`
  - `docs/roadmap.md`
  - `docs/reproducibility.md`
  - `plans/2026-04-11-portfolio-publication-implementation-plan.md`
  - `CODEBASE-PLAN-V-0.1.md`
  - `experiment_spec.md`

## Executive Summary
`chronos-plg` is already portfolio-worthy on engineering substance. It contains a phase-gated BTC trading research platform with leakage-aware evaluation, probabilistic forecasts, realistic execution-cost modeling, robustness checks, paper-trading governance, kill-switch diagnostics, promotion gates, tests, CI, public docs, and curated evidence artifacts.

The next opportunity is not simply "make the repo public." The stronger opportunity is to publish and continuously evolve the project as a credible ML research and ML engineering showcase:

- first, a clean public release that is honest about current findings and safe about data/artifact boundaries
- second, a repeatable publication pipeline that turns every experiment into a readable public evidence card
- third, a stronger research roadmap that upgrades the project from "crypto trading bot" to "time-series foundation-model governance lab under realistic market frictions"
- fourth, resume and portfolio packaging that makes the project legible to hiring managers, ML engineers, AI researchers, and quant-oriented reviewers

The project should not claim to be a profitable live trading system today. The inspected evidence does not support that. The project should instead claim something more defensible and, in many ways, more impressive:

> A rigorous probabilistic trading research platform that can find partial positive-edge regions and reject them when they fail deployment governance.

## Problem / Opportunity Statement
The real problem is that the project's technical depth is still under-leveraged as a public professional asset.

The repository already has stronger ingredients than most portfolio trading projects:

- real modular code instead of notebooks only
- cost-aware backtesting instead of gross-return vanity metrics
- walk-forward and leakage controls instead of random splits
- robust failure reporting instead of cherry-picked wins
- paper-trading readiness gates instead of naive "ship the strategy" logic
- curated public evidence instead of unexplained local artifacts

The opportunity is to make this visible, credible, and continuously publishable. The public audience should understand, in less than five minutes, that the owner can build production-grade ML systems, reason about model risk, run empirical research honestly, and package the work in a reproducible engineering workflow.

## Goals
- [ ] Publish a clean first GitHub release without leaking local data, result stores, IDE state, or session artifacts.
- [ ] Reframe the public narrative from "Chronos trading bot" to "phase-gated probabilistic trading research and model-governance platform."
- [ ] Make the GitHub README and Pages surface strong enough to work as a portfolio landing page.
- [ ] Preserve honest empirical status: positive regions exist, but no candidate is promotion-ready yet.
- [ ] Create a continuous publication workflow where each experiment produces a public evidence card, artifacts, figures, and a short result interpretation.
- [ ] Strengthen the Chronos/forecasting research track so the repo name matches the evidence over time.
- [ ] Add forecast-quality diagnostics that evaluate probabilistic models before trading policy conversion.
- [ ] Add multi-asset and cross-series research tracks after the BTC baseline story is stable.
- [ ] Add MLOps/ResearchOps features that showcase experiment governance, reproducibility, and release discipline.
- [ ] Convert project outcomes into resume bullets, GitHub pinned-repo copy, and a personal-site case study.

## Non-Goals
- [ ] Do not claim live profitability or production trading readiness without evidence.
- [ ] Do not publish the full local `data/` tree.
- [ ] Do not stage ignored raw, processed, or full result artifacts into Git history.
- [ ] Do not optimize for a flashy dashboard before the evidence and publication workflow are coherent.
- [ ] Do not rename or rebrand the project in a way that hides the current Chronos evidence gap.
- [ ] Do not turn this into a live trading/exchange integration project before research governance is stronger.
- [ ] Do not add speculative architecture unless it creates measurable portfolio, research, or reproducibility value.

## Context and Background
The project began as a Chronos-powered BTC forecasting and trading experiment. It has since grown into a broader research platform with:

- data ingestion and contracts
- feature and label generation
- leakage-aware walk-forward evaluation
- baseline model stack
- Chronos candidate path with provenance and fallback guardrails
- LightGBM quantile models
- strategy and position-sizing logic
- cost-aware backtesting
- robustness stress testing
- decision reporting
- paper-trading replay
- monitoring dashboards
- kill-switch diagnostics
- deployment readiness and capital-ramp policy
- multi-objective sweep and campaign tooling
- public docs and generated evidence artifacts

Current evidence is mixed, which is a strength if framed correctly:

- best inspected futures EWMA candidate: `ProfitFactorNet = 1.1739`, `Sharpe = 0.5653`, `Trades = 76`, outcome `ITERATE`
- 1h spot/margin threshold calibration: higher activity, weak PF and Sharpe
- fixed spot campaign: negative result and no promotion recommendation
- 4h looser-threshold spot/margin checks: PF-positive and Sharpe-positive but blocked by kill-switch activity

The public story should be: the system is useful because it makes weak or fragile strategies visible and prevents premature promotion.

## Facts, Assumptions, Unknowns, and Confidence

### Facts
- The repository is initialized as Git, but has no commits yet on `main`.
- `git status --short --branch` showed every project file as untracked.
- `.gitignore` excludes `.venv/`, `data/**`, most `artifacts/**`, IDE state, cache folders, logs, and build outputs.
- `.specstory/` is present locally and currently untracked; it is not excluded by the inspected `.gitignore`.
- `data/raw/`, `data/processed/`, and `data/results/` exist locally and are ignored by the current `data/**` rule.
- `data/README.md` is intended to be the only public file under `data/`.
- `artifacts/public/` and `docs/assets/` contain curated public evidence and figures.
- `.github/workflows/ci.yml` runs install, high-signal ruff checks, `pytest -q`, and `python scripts/smoke_check.py`.
- The repository root did not contain `.venv` during the 2026-06-08 inspection, despite README/Makefile/docs referencing local `.venv` validation.
- `README.md`, `docs/index.md`, `docs/results.md`, `docs/project-status.md`, and `PUBLICATION_NOTES.md` already reflect a research-first public framing.
- The strongest inspected public candidate evidence is currently EWMA-led, not Chronos-led.
- `src/models/chronos2_runner.py` contains a Chronos-style runner with deterministic empirical fallback and provenance logging.
- `src/evaluation/phase7_chronos.py` contains Chronos provenance, calibration, and candidate-gate helpers.
- `src/evaluation/multi_objective.py` contains Phase 11 scoring, acceptance constraints, and Pareto frontier utilities.

### Assumptions
- The owner wants this published under a public GitHub account as a portfolio project.
- The first public release may include source, tests, docs, curated public artifacts, and generated figures.
- The owner is comfortable publishing negative and mixed results when framed honestly.
- The project can remain named `chronos-plg` for now, but public copy should clarify that Chronos is a candidate research track rather than the proven winning model.
- A Markdown-first GitHub Pages surface is acceptable for the first release.
- A richer static case-study page can come after the first release.
- Public result artifacts are safe to publish because they are curated summaries rather than full local data dumps.
- Future implementation can create a repository `.venv` if no suitable isolated environment exists.

### Unknowns
- Final public GitHub repository owner/name.
- Whether to keep the old resume link target or migrate to a different GitHub account/repo.
- Whether the owner wants GitHub Releases, GitHub Pages, personal website, or all three as publication channels.
- Whether future experiment tracking should use MLflow, W&B, plain artifact manifests, or a minimal custom approach.
- Whether Chronos real-backend execution should use AutoGluon, direct Hugging Face model usage, or both.
- Whether multi-asset data should begin with ETH only or a broader basket such as ETH/SOL/BNB/majors.
- Whether any local market-data artifacts have redistribution constraints beyond public-market-data assumptions.

### Confidence
- Codebase/publication-readiness assessment: High
- Current empirical-status assessment: High for inspected public artifacts; Medium for full local result universe
- Chronos research-upgrade direction: Medium-High
- MLOps tooling recommendation: Medium, pending owner preference
- Multi-asset roadmap: Medium

## Codebase / System Analysis Summary
- Areas inspected:
  - repository tree
  - `README.md`
  - `PUBLICATION_NOTES.md`
  - `.gitignore`
  - `.github/workflows/ci.yml`
  - `Makefile`
  - `pyproject.toml`
  - `docs/index.md`
  - `docs/results.md`
  - `docs/project-status.md`
  - `docs/methodology.md`
  - `docs/roadmap.md`
  - `docs/reproducibility.md`
  - `artifacts/public/public_evidence_snapshot.md`
  - `data/README.md`
  - `CODEBASE-PLAN-V-0.1.md`
  - `experiment_spec.md`
  - `src/models/chronos2_runner.py`
  - `src/models/meta_model.py`
  - `src/evaluation/phase7_chronos.py`
  - `src/evaluation/multi_objective.py`
  - representative test inventory through `rg`
- Relevant modules / services:
  - `src/data/`
  - `src/models/`
  - `src/evaluation/`
  - `src/backtest/`
  - `src/strategy/`
  - `src/robustness/`
  - `src/paper_trading/`
  - `src/reporting/`
  - `scripts/`
  - `docs/`
  - `artifacts/public/`
- Current behavior:
  - project is structured as a Python package
  - CI installs `.[dev]`, runs a strict error-level ruff subset, tests, and smoke check
  - public report generation exists through `scripts/generate_public_report.py`
  - public plot generation exists through `scripts/plot_public_results.py`
  - `make public-assets` runs report and plot generation
  - curated public evidence exists in Markdown, JSON, CSV, and PNG forms
- Known pain points:
  - no first commit yet
  - `.specstory/` not ignored
  - docs mention existing `.venv`, but no `.venv` was present during inspection
  - `pyproject.toml` author metadata is generic
  - package description is narrower than the actual public positioning
  - public narrative still risks over-indexing on Chronos while strongest evidence is EWMA-led
  - no first-class experiment-card workflow yet
  - no automated release/evidence-pack workflow yet
  - no public architecture diagram yet
  - no forecast-quality diagnostic suite beyond current quantile calibration helpers
- Architecture notes:
  - the existing modular boundaries are strong enough to support incremental roadmap execution
  - publication and research upgrades can be additive
  - the highest-risk boundary is data/artifact publication, not runtime compatibility
  - the most valuable research gap is forecast/model-governance evidence, not another superficial trading metric

## Dependency and Interface Impact
- Internal dependencies:
  - public docs depend on generated artifacts under `artifacts/public/` and `docs/generated/`
  - plot generation depends on selected local result paths
  - report generation depends on selected JSON/CSV artifacts
  - Chronos/model research phases depend on `src/models/`, `src/evaluation/`, and `scripts/run_chronos2.py`
  - policy/gating enhancements depend on `src/paper_trading/`, `src/robustness/`, `src/reporting/`, and `src/evaluation/multi_objective.py`
- External dependencies:
  - Python `>=3.10`
  - PyTorch/Transformers/Chronos or AutoGluon path for real Chronos experiments
  - Binance/yfinance or alternative public data providers
  - GitHub Actions
  - optional GitHub Pages
  - optional MLflow or W&B
- API / schema / contract impact:
  - publication phases should not break current CLI contracts
  - new experiment-card artifacts should use explicit schema/version fields
  - Chronos provenance schema may need extension to distinguish model family, backend, direct vs AutoGluon path, device, model version, fallback status, and reproducibility fingerprint
  - forecast diagnostics should produce stable JSON/CSV/Markdown artifacts for docs automation
- Build / deployment impact:
  - first release needs CI green on GitHub, not only local
  - Pages publishing may require workflow or repository settings
  - release evidence pack may require a new GitHub Actions workflow
- Backward compatibility considerations:
  - keep existing scripts working
  - keep old public artifacts readable
  - add new commands rather than replacing current result-generation behavior abruptly
  - maintain the current research-first framing until stronger evidence justifies stronger claims

## Constraints and Assumptions

### Constraints
- Must respect repository AGENTS.md environment discipline: use repository `.venv`, repo-specific env, or a newly created clean environment for execution.
- Must not publish local raw/processed/full result data.
- Must not claim deployment readiness without evidence.
- Must keep public claims aligned with current generated artifacts.
- Must preserve a clean Git history from the first commit onward.
- Must keep the first public release lightweight enough for reviewers to clone and inspect.
- Must treat Chronos fallback runs as fallback experiments, not real Chronos evidence.
- Must ensure public docs distinguish facts, inferences, and unknowns where empirical conclusions are mixed.

### Assumptions
- GitHub publication is desired soon; stronger empirical work should continue after first release rather than block release indefinitely.
- The owner values resume/portfolio signal at least as much as strategy profitability.
- Markdown and static assets are sufficient for the first public-facing experience.
- Negative results can be a competitive differentiator if they demonstrate strong governance and empirical honesty.

## Risks and Failure Modes

### Risk 1: Overclaiming profitability or readiness
- Impact: Severe credibility damage; project reads like a typical overfit trading repo.
- Detection: README, resume, docs, or release notes imply a live-profitable strategy or promotion-ready model.
- Mitigation: Keep explicit "research platform, not live trading product" copy. Require every claim to link back to public evidence. Keep readiness status visible.

### Risk 2: Publishing local data or noisy artifacts
- Impact: Repository bloat, possible data redistribution issues, accidental leakage of local state.
- Detection: `git status --ignored`, `git ls-files`, large-file scan, release artifact review.
- Mitigation: Tighten `.gitignore`, stage explicitly, run pre-commit inspection, keep public artifacts under `artifacts/public/` only.

### Risk 3: Chronos narrative mismatch
- Impact: Reviewers expect a Chronos success story and find mostly EWMA evidence.
- Detection: README headline, repo description, resume bullet, or docs overemphasize Chronos wins.
- Mitigation: Frame Chronos as a research track. Add a "What Chronos has/has not proven yet" section. Prioritize real-backend Chronos evidence in Milestone 2.

### Risk 4: First release fails CI
- Impact: Public repo immediately looks unmaintained or unreproducible.
- Detection: GitHub Actions failure on first push.
- Mitigation: Run local validation in an isolated repo environment before push. Fix environment/documentation drift. Keep CI commands aligned with docs.

### Risk 5: Experiment-card workflow becomes manual and stale
- Impact: Continuous publication degrades into abandoned docs.
- Detection: `docs/results.md`, `artifacts/public/`, and latest local experiments diverge.
- Mitigation: Generate experiment cards and evidence snapshots from versioned artifacts. Add CI checks for generated docs freshness where practical.

### Risk 6: Adding heavy MLOps tools slows progress
- Impact: Effort shifts to infrastructure instead of research/productized evidence.
- Detection: new services required before any new result can be published.
- Mitigation: Start with simple manifest-based ResearchOps; add MLflow/W&B only if it improves public/research workflows.

### Risk 7: Multi-asset expansion creates confounded results
- Impact: Public results become broader but less interpretable.
- Detection: multi-asset docs cannot distinguish data-quality effects, model effects, and policy effects.
- Mitigation: Add one asset at a time, preserve BTC baseline, use matched windows and same acceptance gates.

### Risk 8: Resume copy becomes too long or too technical
- Impact: Hiring signal is diluted.
- Detection: project bullet exceeds resume space or lacks measurable outcomes.
- Mitigation: Maintain three variants: short resume bullet, medium GitHub pinned-repo copy, long case-study narrative.

## Proposed Approach
Execute the project in five milestones:

1. Public release foundation: clean Git history, safe artifact boundary, CI, README, Pages-ready docs.
2. Continuous publication system: experiment cards, generated evidence packs, release automation, project case study.
3. Chronos and forecast-quality research upgrade: real backend, provenance, calibration, probabilistic scoring, baseline comparisons.
4. Decision-focused ML and multi-asset expansion: policy learning, abstention, cross-asset context, stronger feature completeness.
5. Professional packaging: resume bullets, personal-site case study, repository releases, issue roadmap, and public update cadence.

The first release should happen before the full research roadmap completes. Waiting for a fully promotable strategy would unnecessarily delay the portfolio value and conflate research quality with trading success.

## Alternatives Considered

### Alternative A: Publish immediately with no further cleanup
- Description: Push current untracked tree as-is.
- Why not chosen: There are visible risks: `.specstory/` is unignored, docs mention missing `.venv`, metadata is generic, and the first commit needs careful staging to avoid raw data history.

### Alternative B: Delay until a profitable/promotable strategy exists
- Description: Continue private research until a candidate passes all deployment gates.
- Why not chosen: The current negative and partial-positive evidence is already portfolio-grade if framed correctly. Waiting for profitability may delay publication indefinitely and incentivize overfitting.

### Alternative C: Build a full web dashboard before publishing
- Description: Create an interactive app before release.
- Why not chosen: Public credibility currently depends more on evidence, methodology, and clean packaging than dashboard polish. A lightweight explorer can come later.

### Alternative D: Rename away from Chronos immediately
- Description: Rebrand to avoid Chronos evidence mismatch.
- Why not chosen: The Chronos track is still a meaningful research direction. Better to clarify the current evidence and strengthen the Chronos roadmap than hide the original thesis.

### Alternative E: Adopt heavy MLOps stack immediately
- Description: Add MLflow/W&B, Docker Compose, databases, and dashboards before first release.
- Why not chosen: This risks overengineering. Start with manifest-based reproducibility and add external tracking only when it produces visible research or portfolio value.

## Phase Plan

### Phase 1: Pre-Publish Safety and Repository Hygiene
- Objective: Prepare the repository for a clean first public commit without leaking local data, session state, or noisy artifacts.
- Why now: The repository has no commits yet, so the first commit can establish a clean public history if staged carefully.
- Type: Sequential
- Depends on:
  - current repository inspection
  - owner confirmation of target GitHub repo/account before push
- Deliverables:
  - tightened `.gitignore`
  - corrected metadata/docs drift
  - staged-file audit notes
  - clean first commit candidate
- Checklist:
  - [ ] Add `.specstory/` and any other local session folders to `.gitignore`.
  - [ ] Confirm `.idea/`, `.vscode/`, `.ruff_cache/`, raw data, processed data, and local results are ignored.
  - [ ] Update README/docs language that says `.venv` already exists; describe how to create it if missing.
  - [ ] Update `pyproject.toml` description and author metadata to match public positioning.
  - [ ] Run a large-file scan before staging.
  - [ ] Stage explicitly rather than using `git add .` blindly.
  - [ ] Verify tracked files with `git diff --cached --name-only`.
  - [ ] Verify ignored data remains untracked with `git status --short --ignored data artifacts .specstory`.
  - [ ] Create first commit only after validation passes.
- Validation:
  - `git status --short --ignored`
  - `git diff --cached --stat`
  - large-file check using `find` or equivalent
  - environment setup check using repo `.venv` or newly created clean env
  - `pytest -q`
  - `python scripts/smoke_check.py`
  - `python scripts/generate_public_report.py`
  - `python scripts/plot_public_results.py`
- Exit criteria:
  - no raw/processed/full result data staged
  - no IDE/session/cache folders staged
  - CI-equivalent local validation passes
  - first commit is ready or completed
- Green-light cue:
  - Owner approves target GitHub repository and first public commit scope.
- Rollback / containment:
  - Before push, remove bad staged files with `git restore --staged <path>`.
  - If sensitive/noisy files are committed locally before push, rewrite the unpublished local history or recreate the initial commit before publishing.
  - If pushed accidentally, treat as history-rewrite incident and coordinate before force-pushing.

### Phase 2: First Public GitHub Release
- Objective: Publish the repository publicly with credible README, docs, CI, curated artifacts, and first release notes.
- Why now: The repo already has a public docs suite and evidence artifacts; the remaining work is release discipline.
- Type: Sequential
- Depends on:
  - Phase 1
- Deliverables:
  - public GitHub repository
  - passing GitHub Actions workflow
  - GitHub repository description/topics
  - initial release tag such as `v0.1-public-research-baseline`
  - release notes
  - GitHub Pages configured from `docs/` or a documented equivalent
- Checklist:
  - [ ] Create or select target GitHub repo.
  - [ ] Push `main`.
  - [ ] Confirm GitHub Actions passes.
  - [ ] Add repository description: "Phase-gated BTC trading research platform with probabilistic forecasting, cost-aware backtesting, paper-trading governance, and curated public evidence."
  - [ ] Add topics: `quant`, `trading`, `time-series`, `forecasting`, `chronos`, `machine-learning`, `research`, `python`, `backtesting`, `mlops`.
  - [ ] Enable GitHub Pages from `docs/` if using Pages.
  - [ ] Create release notes that state current evidence honestly.
  - [ ] Pin the repository on GitHub profile.
  - [ ] Update resume/project links only after public repo and docs are live.
- Validation:
  - GitHub Actions green
  - README renders correctly
  - docs image links render correctly
  - release artifact list contains only intended files
  - Pages URL loads if enabled
- Exit criteria:
  - public repo is accessible
  - CI is green
  - release notes are published
  - docs are navigable from README
- Green-light cue:
  - First release can be shared with reviewers without caveats beyond known research status.
- Rollback / containment:
  - If CI fails publicly, either fix immediately or mark release as pre-release until fixed.
  - If docs/assets break, patch docs and create a small follow-up release.
  - If unintended artifacts are published, remove from repo and assess whether history cleanup is required.

### Phase 3: Public Story, Architecture, and Case Study Upgrade
- Objective: Make the project instantly legible as a serious ML engineering and ML research portfolio piece.
- Why now: The current docs are honest and useful, but the project needs a stronger visual/narrative layer for portfolio conversion.
- Type: Parallelizable
- Depends on:
  - Phase 1
  - can proceed partly before Phase 2 if not delaying publication
- Deliverables:
  - architecture diagram
  - README "one-screen evidence" section
  - public case-study page
  - stronger GitHub Pages landing page
  - "what failed and why it matters" section
- Checklist:
  - [x] Add an architecture diagram covering data -> forecasts -> strategy -> cost engine -> robustness -> paper governance -> publication artifacts.
  - [x] Add a concise "Evidence Snapshot" table with current best, negative, and unresolved results.
  - [x] Add a "What this demonstrates" section mapping repo capabilities to ML engineering skills.
  - [x] Add a "What this does not claim" section.
  - [x] Add a Chronos narrative clarification: current strongest evidence is EWMA-led; Chronos is an active research track.
  - [x] Create a personal-site case-study draft with problem, system, evidence, failures, roadmap, and skills demonstrated.
  - [x] Add screenshots/figures that make the project visually scannable.
- Validation:
  - docs render locally or on GitHub
  - all images are committed and correctly linked
  - claims match `artifacts/public/public_evidence_snapshot.md`
  - reviewer can identify project thesis, current result, and next milestone in under five minutes
- Exit criteria:
  - public story is coherent across README, Pages, release notes, and resume copy
- Green-light cue:
  - Project is ready to be pinned and linked from resume/personal website.
- Rollback / containment:
  - If new public wording overclaims, revert text only; no code rollback needed.
  - If visuals become stale, mark them generated and regenerate from current public artifacts.

### Phase 4: Experiment Card and Continuous Publication System
- Objective: Turn ongoing research into a repeatable public publishing workflow.
- Why now: Continuous publication is what makes this a living portfolio project instead of a one-time repo dump.
- Type: Parallelizable
- Depends on:
  - Phase 2 for public release workflow
  - existing public report/plot scripts
- Deliverables:
  - `docs/experiments/` structure
  - experiment-card schema
  - experiment-card generator
  - public evidence index
  - release checklist
  - optional GitHub Action for evidence freshness
- Checklist:
  - [ ] Create `docs/experiments/README.md` explaining experiment-card conventions.
  - [ ] Define an experiment-card template with hypothesis, setup, data window, model, policy, metrics, result, decision, and follow-up.
  - [ ] Add machine-readable experiment metadata schema.
  - [ ] Extend `scripts/generate_public_report.py` or add a focused script to generate experiment-card Markdown from selected result artifacts.
  - [ ] Add a public evidence index sorted by release/milestone.
  - [ ] Add "negative result" and "partial-positive result" badges or labels.
  - [ ] Add release checklist to `PUBLICATION_NOTES.md` or a dedicated `docs/release-process.md`.
  - [ ] Consider a CI check that generated public snapshots are current when source artifacts change.
- Validation:
  - generated experiment cards are deterministic
  - cards include enough context to reproduce or understand the run
  - no local-only paths leak into public docs
  - generated docs pass markdown link checks if available
- Exit criteria:
  - a new experiment can be published by running one documented command and reviewing generated output
- Green-light cue:
  - Next research phase can publish results continuously without ad hoc docs work.
- Rollback / containment:
  - If generator output is wrong, do not publish the generated card; fix generator or hand-edit with clear provenance.
  - If schema is too heavy, keep a minimal Markdown template and defer automation.

### Phase 5: Forecast-Quality Evaluation Layer
- Objective: Evaluate probabilistic forecasts directly before converting them into trading decisions.
- Why now: The project needs stronger ML research signal beyond trading metrics. Forecast calibration and distributional scoring will show serious ML evaluation maturity.
- Type: Parallelizable after Phase 4 design; Sequential for integration into promotion gates
- Depends on:
  - existing walk-forward prediction artifacts
  - `src/evaluation/phase7_chronos.py`
  - existing quantile outputs `q10`, `q50`, `q90`
- Deliverables:
  - forecast evaluation module
  - metrics artifacts
  - calibration plots
  - docs page explaining forecast-vs-trading distinction
  - tests
- Checklist:
  - [ ] Add pinball loss for each quantile.
  - [ ] Add weighted interval score or CRPS-like approximation for quantile forecasts.
  - [ ] Add empirical coverage diagnostics for q10/q90 intervals.
  - [ ] Add calibration error by regime and timeframe.
  - [ ] Add sharpness/interval-width diagnostics.
  - [ ] Add directional accuracy only as a secondary, clearly limited metric.
  - [ ] Add forecast-quality comparison tables for RandomWalk, EWMA, LightGBM, Chronos, and MetaModel where available.
  - [ ] Add plotting for calibration and interval width.
  - [ ] Add tests with synthetic calibrated/miscalibrated forecasts.
  - [ ] Document when forecast improvements fail to translate into trading improvements.
- Validation:
  - unit tests for metrics
  - integration test on smoke predictions
  - generated public forecast report
  - consistency checks between metric artifacts and docs tables
- Exit criteria:
  - every model can be evaluated on forecast quality independent of trading policy
  - public docs can answer "is Chronos forecasting better, even if trading is not ready?"
- Green-light cue:
  - Begin Chronos real-backend track with forecast-quality gates, not only trading gates.
- Rollback / containment:
  - Keep forecast metrics additive; do not break existing trading reports.
  - If a metric is unstable or misleading, mark it experimental and exclude it from promotion gates.

### Phase 6: Chronos Real-Backend Research Track
- Objective: Produce defensible Chronos evidence using real backend execution, explicit provenance, and baseline comparisons.
- Why now: The project name and resume story depend on making the Chronos track concrete and current.
- Type: Sequential
- Depends on:
  - Phase 5
  - validated isolated environment
  - official current Chronos/AutoGluon documentation review at implementation time
- Deliverables:
  - real-backend Chronos run path
  - backend provenance artifacts
  - fallback exclusion rules
  - Chronos-vs-baseline forecast and trading report
  - public experiment cards
- Checklist:
  - [ ] Re-check official Chronos/Chronos-2 and AutoGluon documentation before implementation.
  - [ ] Decide direct model path, AutoGluon path, or both.
  - [ ] Add dependency path behind optional extras if heavy dependencies are not suitable for default install.
  - [ ] Extend provenance schema for backend type, model id, model version, package versions, device, fallback status, and run fingerprint.
  - [ ] Ensure fallback-active runs cannot be labeled as real Chronos evidence.
  - [ ] Run Chronos on a bounded BTC window first.
  - [ ] Compare against RandomWalk, EWMA, and LightGBM on forecast metrics.
  - [ ] Compare against baseline strategies under identical cost/governance gates.
  - [ ] Publish positive, neutral, or negative result honestly.
- Validation:
  - unit tests for provenance guardrails
  - smoke-level Chronos fallback test
  - real-backend run artifact with `fallback_active = false`
  - forecast-quality metrics generated
  - trading governance report generated
- Exit criteria:
  - public docs can state exactly what Chronos did or did not add over baselines
- Green-light cue:
  - If Chronos improves forecast quality or specific regimes, move to model-combination and policy phases.
  - If Chronos does not improve, publish as negative result and preserve governance credibility.
- Rollback / containment:
  - Keep heavy Chronos dependencies optional.
  - If real backend is too expensive/slow, publish a bounded benchmark and document compute limits.
  - If Chronos underperforms, do not hide it; turn it into a result card.

### Phase 7: Data Completeness and Feature Provenance Upgrade
- Objective: Repair or explicitly bound high-value data gaps, especially open-interest and liquidation families.
- Why now: Current docs identify OI/liquidation completeness as research debt, and these features matter for regime/event discrimination.
- Type: Parallelizable with Phase 6 after public release
- Depends on:
  - existing data contracts and quality gates
  - provider selection
- Deliverables:
  - provider decision note
  - restored or explicitly bounded OI/liquidation data path
  - quality reports
  - degraded-run behavior documentation
  - experiment cards comparing feature availability effects
- Checklist:
  - [ ] Inventory current OI/liquidation collection limitations.
  - [ ] Evaluate alternative historical data sources.
  - [ ] Choose provider strategy: free/public, paid, synthetic proxy, or explicit omission.
  - [ ] Add provider provenance to dataset metadata.
  - [ ] Add data-quality thresholds for feature-family use in promotion runs.
  - [ ] Rebuild BTC datasets with restored or clarified feature families.
  - [ ] Run matched experiments with and without OI/liquidation families.
  - [ ] Publish whether feature repair changed forecast quality, trading metrics, or governance status.
- Validation:
  - data contract tests
  - quality-gate tests
  - null coverage reports
  - before/after experiment card
- Exit criteria:
  - public docs no longer leave OI/liquidation status vague
  - promotion runs are blocked or marked degraded when key features are missing
- Green-light cue:
  - Proceed to broader model/policy comparisons using a stable data foundation.
- Rollback / containment:
  - If provider data is unreliable or not publishable, keep feature family excluded from public claims.
  - Preserve previous public artifacts for comparison.

### Phase 8: Decision-Focused ML Policy Layer
- Objective: Upgrade from forecast-threshold trading rules to a decision-focused policy layer that learns when not to trade.
- Why now: This creates a stronger ML/AI research story: model outputs are governed by downstream economics, uncertainty, and deployment constraints.
- Type: Sequential
- Depends on:
  - Phase 5 forecast metrics
  - stable baseline reports
  - cost-engine and policy artifacts
- Deliverables:
  - meta-labeling dataset
  - abstention policy baseline
  - cost-aware decision model
  - policy evaluation report
  - tests
- Checklist:
  - [ ] Define decision labels using realized net outcomes, cost thresholds, and risk events.
  - [ ] Consider triple-barrier or net-edge labels for long/short/flat outcomes.
  - [ ] Add abstention metrics: avoided bad trades, missed good trades, net utility, and coverage.
  - [ ] Train simple baselines first: logistic regression/LightGBM classifier over forecast and regime features.
  - [ ] Compare threshold policy vs learned abstention policy under identical backtest/governance rules.
  - [ ] Add calibration for policy confidence.
  - [ ] Add tests for label generation and leakage boundaries.
  - [ ] Publish result as "decision-focused forecasting" experiment.
- Validation:
  - label leakage tests
  - policy backtest comparison
  - net PF/Sharpe/trade count/kill-rate report
  - forecast-to-policy attribution table
- Exit criteria:
  - public evidence shows whether decision learning improves deployment readiness or simply overfits
- Green-light cue:
  - If policy improves readiness without collapsing PF/Sharpe, move to broader assets/regimes.
- Rollback / containment:
  - Keep threshold policy as baseline and fallback.
  - Do not promote learned policy unless it passes existing gates.

### Phase 9: Multi-Asset and Cross-Series Context Expansion
- Objective: Test whether foundation/time-series models benefit from cross-asset context beyond BTC-only signals.
- Why now: Multi-series context is a stronger ML research question than single-asset BTC tuning, but it should follow stable BTC evidence.
- Type: Sequential
- Depends on:
  - Phase 5
  - Phase 6
  - Phase 7 recommended
- Deliverables:
  - second asset support, likely ETH first
  - multi-asset dataset contracts
  - grouped or multivariate forecasting experiments
  - cross-asset context result cards
- Checklist:
  - [ ] Add asset parameterization where assumptions are BTC-specific.
  - [ ] Add ETH dataset build path as first expansion target.
  - [ ] Preserve BTC-only baseline experiments.
  - [ ] Add cross-asset features such as ETH/BTC relative returns, correlation, volatility spread, and market beta.
  - [ ] Add grouped evaluation by asset and timeframe.
  - [ ] Compare univariate BTC vs BTC+ETH context under the same forecast metrics.
  - [ ] Add trading-policy comparison only after forecast-quality comparison.
  - [ ] Publish a multi-asset context experiment card.
- Validation:
  - data contract tests for multiple assets
  - forecast report by asset
  - no leakage from future cross-asset values
  - matched-window comparison
- Exit criteria:
  - public docs can answer whether cross-series context helped, hurt, or was inconclusive
- Green-light cue:
  - Expand to a small basket only if ETH path is coherent and reproducible.
- Rollback / containment:
  - Keep BTC baseline as canonical.
  - If ETH data quality is weak, mark multi-asset track experimental and do not mix it into headline metrics.

### Phase 10: ResearchOps and Optional MLOps Integration
- Objective: Add enough experiment tracking and reproducibility infrastructure to showcase production ML discipline without overengineering.
- Why now: The project already has many artifacts; the next step is making runs easier to compare, audit, and publish.
- Type: Parallelizable
- Depends on:
  - Phase 4
  - owner decision on MLflow/W&B/plain manifests
- Deliverables:
  - run manifest standard
  - reproducibility fingerprints
  - optional MLflow/W&B integration
  - Docker or environment lock strategy
  - artifact retention policy
- Checklist:
  - [ ] Define canonical run metadata fields: code commit, data fingerprint, config, model, scenario, timeframe, seed, dependency snapshot.
  - [ ] Add run comparison index.
  - [ ] Add optional experiment tracking backend behind config.
  - [ ] Add Dockerfile or documented clean environment bootstrap if useful.
  - [ ] Add `make validate-public` target that runs tests, smoke, and public artifact generation.
  - [ ] Add release evidence-pack generation command.
  - [ ] Add reproducibility badges or status table to docs.
- Validation:
  - run manifests are deterministic where expected
  - public artifacts include commit/config fingerprints
  - clean environment setup succeeds
  - CI stays reasonably fast
- Exit criteria:
  - a reviewer can understand which code/data/config produced each public claim
- Green-light cue:
  - Project is ready for recurring public releases without manual archaeology.
- Rollback / containment:
  - Keep external tracking optional.
  - If CI becomes slow, split heavy checks into scheduled/manual workflows.

### Phase 11: GitHub Issues, Project Board, and Public Roadmap
- Objective: Convert the roadmap into visible, professional project management artifacts.
- Why now: A living public project benefits from visible open issues and milestones.
- Type: Parallelizable
- Depends on:
  - Phase 2
- Deliverables:
  - GitHub milestones
  - labeled issues
  - project board or issue roadmap
  - contribution/status docs if desired
- Checklist:
  - [ ] Create milestone `v0.1 Public Research Baseline`.
  - [ ] Create milestone `v0.2 Chronos Real Backend and Forecast Diagnostics`.
  - [ ] Create milestone `v0.3 Decision-Focused Policy Layer`.
  - [ ] Create milestone `v0.4 Multi-Asset Context`.
  - [ ] Add labels: `research`, `ml`, `forecasting`, `governance`, `docs`, `data`, `publication`, `negative-result`, `artifact`.
  - [ ] Open issues from this plan with scoped checklists.
  - [ ] Pin a roadmap issue explaining current status and next evidence questions.
- Validation:
  - roadmap links from README/docs
  - issues are scoped and actionable
  - labels are consistent
- Exit criteria:
  - public visitors can see active direction and completed milestones
- Green-light cue:
  - Begin executing milestone issues in public.
- Rollback / containment:
  - If public project management is too much overhead, keep a simpler `docs/roadmap.md` and release checklist.

### Phase 12: Resume, GitHub Profile, and Personal-Site Packaging
- Objective: Convert the project into visible career signal.
- Why now: The user explicitly wants this to showcase ML engineering and ML/AI research capabilities.
- Type: Parallelizable
- Depends on:
  - Phase 2 for public links
  - Phase 3 for case-study quality
- Deliverables:
  - resume bullet variants
  - GitHub pinned-repo blurb
  - LinkedIn/project summary
  - personal-site case study
  - optional short demo video or screenshot thread
- Checklist:
  - [ ] Write short resume bullet for tight resume space.
  - [ ] Write medium project entry for selected projects section.
  - [ ] Write long case-study narrative for personal website.
  - [ ] Update GitHub profile pinned-repo description.
  - [ ] Create one visual social/share artifact if desired.
  - [ ] Keep all claims consistent with current evidence.
  - [ ] Add clear skill mapping: probabilistic forecasting, leakage-safe evaluation, cost-aware backtesting, model governance, MLOps, CI, publication automation.
- Validation:
  - resume link works
  - project page loads
  - wording does not claim deployment-ready profitability
  - claims are backed by public docs/artifacts
- Exit criteria:
  - project is usable as a resume/interview anchor
- Green-light cue:
  - Start linking the project from resume and job applications.
- Rollback / containment:
  - If claims become outdated, update resume/project copy after each major release.

## Milestones

### Milestone 1: Clean Public Baseline
- Target outcome: Public repository is safely published with CI, docs, curated evidence, and honest research framing.
- Includes phases:
  - Phase 1
  - Phase 2
  - initial subset of Phase 3
- Completion criteria:
  - first commit exists
  - repo is public
  - GitHub Actions passes
  - raw/processed/full result data is not tracked
  - README and docs render correctly
  - release `v0.1-public-research-baseline` exists
- Evidence / demo expected:
  - public GitHub URL
  - passing CI run
  - rendered README
  - public evidence snapshot
  - release notes
- Go / no-go note:
  - Go if CI is green and artifact boundary is clean.
  - No-go if data/session artifacts are staged or public claims overstate readiness.

### Milestone 2: Portfolio-Grade Presentation
- Target outcome: The repo reads like a professional ML engineering case study, not only a code dump.
- Includes phases:
  - Phase 3
  - Phase 12 initial packaging
- Completion criteria:
  - architecture diagram exists
  - Pages/case-study surface exists
  - resume bullet is updated
  - Chronos evidence gap is explained honestly
  - project is pinned/linked publicly
- Evidence / demo expected:
  - GitHub Pages or docs landing page
  - personal-site case-study draft or published page
  - updated resume project copy
- Go / no-go note:
  - Go if a reviewer can understand the system, evidence, and skill signal in under five minutes.

### Milestone 3: Continuous Evidence Publishing
- Target outcome: New experiments can be published as versioned public evidence with minimal manual work.
- Includes phases:
  - Phase 4
  - Phase 10 partial
  - Phase 11
- Completion criteria:
  - experiment-card template exists
  - at least two experiment cards are generated or manually published using the template
  - public evidence index exists
  - release checklist exists
  - GitHub milestones/issues reflect roadmap
- Evidence / demo expected:
  - `docs/experiments/`
  - public evidence index
  - issue roadmap
- Go / no-go note:
  - Go if experiment publication is repeatable without rethinking docs structure each time.

### Milestone 4: Forecasting Research Upgrade
- Target outcome: The project can evaluate probabilistic forecasts as ML outputs independent of trading outcomes.
- Includes phases:
  - Phase 5
  - Phase 6
- Completion criteria:
  - pinball/coverage/calibration/sharpness metrics exist
  - real-backend Chronos run is attempted and documented
  - fallback runs are not misrepresented
  - forecast-quality report compares models
- Evidence / demo expected:
  - forecast diagnostics report
  - Chronos provenance artifact
  - Chronos-vs-baseline experiment card
- Go / no-go note:
  - Go if Chronos results are reproducible and honestly interpreted, whether positive or negative.

### Milestone 5: Decision-Focused ML and Data Quality Upgrade
- Target outcome: The system tests whether better policy learning and improved feature families can convert signal into more stable net edge.
- Includes phases:
  - Phase 7
  - Phase 8
- Completion criteria:
  - data completeness status is resolved or explicitly bounded
  - learned abstention/meta-label policy baseline exists
  - policy comparison artifacts exist
  - existing promotion gates remain binding
- Evidence / demo expected:
  - OI/liquidation feature status report
  - threshold-vs-policy experiment card
  - promotion/readiness comparison
- Go / no-go note:
  - Go only if policy improvements survive existing governance gates.

### Milestone 6: Multi-Asset Research Lab
- Target outcome: The project becomes a broader time-series foundation-model and market-context research platform.
- Includes phases:
  - Phase 9
  - Phase 10 remaining work
- Completion criteria:
  - ETH or another second asset is supported
  - multi-asset forecast diagnostics exist
  - BTC baseline remains intact
  - public docs explain whether cross-series context helped
- Evidence / demo expected:
  - multi-asset experiment card
  - grouped forecast report
  - updated roadmap/status page
- Go / no-go note:
  - Go if multi-asset support improves research quality without confusing the public baseline.

## Validation Strategy

### Automated validation
- Unit tests:
  - `pytest -q`
  - focused tests for new forecast metrics, provenance guardrails, data contracts, and policy labels
- Integration tests:
  - `python scripts/smoke_check.py`
  - public report generation
  - public plot generation
  - Chronos fallback/provenance smoke where possible
- End-to-end tests:
  - bounded backtest/paper replay on synthetic or small public dataset
  - generated experiment card from selected artifact
  - release evidence-pack generation
- Static analysis / lint / type checks:
  - existing CI ruff subset
  - consider broader ruff after first release
  - mypy only if current type baseline is manageable

### Manual validation
- Manual test flows:
  - read README as first-time reviewer
  - click all docs links from README
  - inspect GitHub Actions result
  - inspect release artifacts
  - inspect generated figures
  - compare resume claims with public docs
- Inspection points:
  - staged-file list before first commit
  - ignored files before push
  - public artifact directory contents
  - docs wording for overclaims
  - Chronos fallback/provenance labels

### Acceptance thresholds
- Functional:
  - CI green on public repo
  - tests and smoke pass in isolated environment
  - public artifact generation completes
- Performance:
  - CI should remain reasonable for pull requests
  - heavy Chronos runs should be manual or scheduled unless bounded
- Reliability:
  - generated reports deterministic for same artifacts
  - public claims traceable to artifact snapshots
- Security / data safety:
  - no local raw/processed/full result data tracked
  - no credentials or local session folders tracked
- Usability / DX:
  - setup docs work from a clean clone
  - reviewer can understand project purpose, status, and evidence quickly

## Rollout / Migration Strategy
- Rollout pattern:
  - publish first clean baseline release
  - then execute visible GitHub milestones
  - tag releases at meaningful evidence checkpoints
- Feature flags:
  - use config switches for optional model backends, tracking integrations, and policy variants
  - keep heavy dependencies behind optional extras when possible
- Backward compatibility:
  - maintain existing CLI scripts
  - add new report/experiment commands rather than breaking current public workflows
- Data migration:
  - no public data migration in first release
  - future data-provider changes must include metadata/provenance updates
- Rollback path:
  - text/docs can be patched with normal commits
  - bad generated artifacts can be replaced with corrected release notes and patch release
  - accidental sensitive/noisy commit before public push should be removed from unpublished history
  - accidental sensitive/noisy public push requires immediate history cleanup decision
- Safe deployment notes:
  - this is a research publication rollout, not a live trading deployment
  - no exchange execution credentials should be introduced as part of this plan

## Observability / Debugging Plan
- Logs:
  - keep script logs concise but include run id, data path, model, scenario, timeframe, seed, and output directory
  - preserve Chronos backend/fallback warnings
- Metrics:
  - forecast: pinball loss, coverage, calibration error, interval width, CRPS-like score
  - trading: PF net, Sharpe, trades, drawdown, total costs, cost/gross ratio, turnover
  - governance: kill events, kill-event rate, readiness reasons, promotion recommendation
  - publication: artifact generation timestamp, source run ids, commit fingerprint
- Traces:
  - not required for first release
  - optional future OpenTelemetry only if runtime complexity grows
- Alerts:
  - GitHub Actions failures
  - stale generated public artifacts if freshness checks are added
- Debug hooks:
  - provenance payloads for model backend runs
  - dataset quality reports
  - experiment manifests
  - decision reports and kill-event taxonomy
- Failure signals:
  - `fallback_active = true` in a real Chronos-labeled run
  - missing data-family coverage in promotion run
  - public docs disagree with latest public evidence snapshot
  - CI failure after dependency or environment change

## Open Questions
- [ ] What exact GitHub repository owner/name should be used for the public release?
- [ ] Should the project stay under `chronos-plg`, or should a more general repo name be chosen later?
- [ ] Should GitHub Pages be enabled immediately, or should docs render from README first?
- [ ] Should experiment tracking stay manifest-based for now, or should MLflow/W&B be introduced in Milestone 3?
- [ ] Which second asset should be first for multi-asset expansion: ETH, SOL, or another market?
- [ ] Is a personal-site case study desired before or after first public release?
- [ ] Should release artifacts be stored in Git, GitHub Releases, or both?

## Decision Log

### Decision 1
- Decision: Publish as a research platform, not a live profitable trading system.
- Why: Current evidence shows positive regions but no promotion-ready candidate.
- Alternatives rejected: Claiming profitability; delaying publication until profitability.
- Implications: Public copy must emphasize governance, methodology, and honest evidence.

### Decision 2
- Decision: Keep Chronos as a research track but avoid presenting it as the current proven winner.
- Why: Strongest inspected public evidence is EWMA-led.
- Alternatives rejected: Hide Chronos gap; rename immediately; overclaim Chronos impact.
- Implications: Next research milestone must produce real-backend Chronos evidence or publish a clear negative result.

### Decision 3
- Decision: Use curated artifacts rather than publishing full local data/results.
- Why: Full local data/results are large, noisy, and unnecessary for public review.
- Alternatives rejected: Ship full `data/`; ship no evidence at all.
- Implications: Public artifact generation and evidence snapshots become critical.

### Decision 4
- Decision: Prioritize forecast diagnostics before more aggressive trading-policy work.
- Why: The project needs stronger ML research signal and a way to evaluate Chronos independent of trading policy.
- Alternatives rejected: Keep optimizing thresholds only; jump directly to multi-asset trading.
- Implications: Phase 5 is the main bridge from engineering platform to ML research showcase.

### Decision 5
- Decision: Start with lightweight ResearchOps, then add heavier MLOps tooling only if it earns its cost.
- Why: The repo already has artifacts and scripts; heavy tooling could slow publication.
- Alternatives rejected: Mandatory MLflow/W&B before release.
- Implications: Manifest/schema discipline comes first; external trackers remain optional.

## Implementation Log
| Date | Phase / Step | What was done | Files / Modules touched | Validation run | Outcome | Notes / blockers |
|---|---|---|---|---|---|---|
| 2026-06-08 | Planning | Created implementation roadmap from repository inspection and user portfolio goals | `plans/2026-06-08-chronos-portfolio-publication-research-roadmap-implementation-plan.md` | Plan artifact created | Draft | Awaiting owner review/approval before implementation |
| 2026-06-09 | Milestone 1 / Phase 1 | Started clean public baseline execution: tightened ignore policy, corrected public metadata and environment docs, removed absolute local links from public docs, created repo `.venv` with Python 3.12, installed `.[dev]`, and staged intended public files only | `.gitignore`, `pyproject.toml`, `README.md`, `docs/reproducibility.md`, `docs/profitability-track.md`, staged public source/docs/artifacts | `.venv/bin/ruff check src config scripts tests --select E9,F63,F7,F82`; `.venv/bin/pytest -q`; `.venv/bin/python scripts/smoke_check.py`; `.venv/bin/python scripts/generate_public_report.py`; `.venv/bin/python scripts/plot_public_results.py` | Success | Raw/processed/full result data, `.venv`, IDE state, and `.specstory` remained ignored |
| 2026-06-09 | Milestone 2 / Phase 3 | Reframed the public project around research architecture, evidence, engineering scope, limitations, and portfolio signal; replaced visible stage-number labels in charts/docs with descriptive research labels; added generated architecture visual, case study, portfolio copy, deterministic public validation target, and current GitHub Actions versions | `README.md`, `docs/index.md`, `docs/case-study.md`, `docs/portfolio-copy.md`, `docs/results.md`, `docs/experiment-log.md`, `scripts/plot_public_results.py`, `scripts/generate_public_report.py`, `docs/assets/*`, `Makefile`, `.github/workflows/ci.yml` | `make validate-public`; visual inspection of architecture, comparison, calibration, and sensitivity figures | Success | Internal artifact paths and schema identifiers retain historical stage numbers for compatibility; public-facing labels no longer expose them |

## Final Approval Gate
- [x] Scope is clear
- [x] Critical unknowns are resolved enough for planning, with non-blocking decisions captured as open questions
- [x] Plan is evidence-based
- [x] Phases are actionable
- [x] Validation is defined
- [x] Rollout / rollback is defined where relevant
- [x] Parallelizable work is clearly marked
- [x] Ready for implementation after owner approval
