# Implementation Plan

## Title
Chronos-PLG Portfolio Publication and Research Upgrade Plan

## Metadata
- Date: 2026-04-11
- Author: Codex
- Status: Completed
- Related issue / task: Make the repository publishable, portfolio-worthy, and execution-ready for a public GitHub release
- Related analysis report: Repo inspection performed in-session on code, docs, tests, and existing result artifacts
- Related PRs: None yet
- Related docs: `README.md`, `experiment_spec.md`, `CODEBASE-PLAN-V-0.1.md`, `docs/codebase-inventory-v0.1.0/*`

## Executive Summary
`chronos-plg` already contains a substantial research/trading system: data ingestion, leakage controls, baselines, Chronos candidate support, backtest engine, robustness checks, decision reporting, paper-trading governance, and reproducible script entry points. The engineering substrate is stronger than the current public presentation suggests.

The repository is not publish-ready because the public boundary is undefined, the narrative is inconsistent, the tree contains local/runtime noise, the README understates the implemented scope, and the results are not curated into a credible research story. The correct publication strategy is not to market this as a finished profitable trading system. The correct strategy is to publish it as a rigorous, phase-gated trading research platform with honest current findings, reproducible methodology, selected evidence, and a clear next-experiment roadmap.

This plan therefore has two parallel goals:
- turn the repository into a clean, compelling public research asset
- strengthen the project’s substance with the highest-leverage pre-publication improvements, especially around reproducibility, reporting, visual storytelling, and evidence-backed strategy enhancement

## Problem / Opportunity Statement
The real problem is not that the repo is “messy.” The real problem is that the project’s technical depth is hidden behind poor public packaging and inconsistent storytelling.

The opportunity is strong:
- the repo already looks like a serious research codebase rather than a toy notebook
- local validation is healthy
- the result set already contains both positive and negative evidence
- the negative evidence is itself portfolio-grade if presented honestly and analytically

The risk is also clear:
- if published as-is, the repo will look noisy, contradictory, and unserious despite containing real work
- if overclaimed as “profit-making,” it will lose credibility because the current artifacts do not support that claim

## Facts, Inferences, Unknowns, Confidence

### Facts
- The repository is not currently initialized as a Git repository in this directory.
- There is no `.gitignore`.
- There is no visible `LICENSE` file, even though `README.md` and `pyproject.toml` reference MIT.
- The working tree is approximately `7.9G`, dominated by `.venv`.
- Local validation succeeded using the existing repository environment:
  - `.venv/bin/python -V` -> `Python 3.12.3`
  - `./.venv/bin/pytest -q` -> `123 passed in 8.38s`
  - `./.venv/bin/python scripts/smoke_check.py` -> passed with 7 folds
- `README.md` is materially outdated relative to the actual implemented scope.
- Internal inventory docs describe Phases 0-10 as implemented, while the public README still frames most phases as incomplete.
- Existing artifacts show mixed empirical evidence:
  - tuned EWMA futures runs achieved `ProfitFactorNet > 1` and positive Sharpe but still failed readiness / promotion gates
  - some 1h spot campaign artifacts are clearly negative and recommend rollback / no promotion

### Inferences
- The project is already portfolio-worthy on engineering depth, but not yet on public presentation.
- The strongest public version is a research platform with honest evidence, not a polished “alpha found” trading pitch.
- The shortest path to a credible publication is curation and reframing, not a ground-up rewrite.
- Additional strategy experiments can improve the project’s upside, but documentation and reporting improvements will create the largest immediate portfolio gain.

### Unknowns
- Whether the public release should live in this repo directly or in a separate cleaned public mirror.
- Whether all generated results should be published in-repo, attached to GitHub Releases, or selectively summarized in docs.
- Whether the owner wants a Markdown-only docs site or a richer static demo surface.
- Whether any currently available data artifacts have redistribution constraints beyond public market data assumptions.

### Confidence
- Codebase assessment confidence: High
- Publishability gap assessment confidence: High
- Current strategy-profitability assessment confidence: Medium
- Improvement upside assessment confidence: Medium

## Goals
- [ ] Make the repository clean, coherent, and safe for public publication.
- [ ] Reframe the project as a rigorous research platform with honest current status.
- [ ] Add portfolio-grade documentation, methodology, visuals, and reproducible evidence.
- [ ] Create a small curated public artifact set that demonstrates real outputs without shipping the full local runtime tree.
- [ ] Build a GitHub Pages-ready docs surface that makes the project legible in under five minutes.
- [ ] Preserve and strengthen reproducibility with command-level workflows and validation gates.
- [ ] Identify and implement the highest-leverage strategy improvements that can materially improve evidence quality before publication.

## Non-Goals
- [ ] Claim guaranteed profitability or production readiness without evidence.
- [ ] Turn the repo into a live trading product before publication.
- [ ] Publish the entire local `data/` and runtime tree.
- [ ] Add speculative architecture or heavyweight infrastructure that does not materially improve publication quality.
- [ ] Create a misleading marketing narrative that hides weak or negative results.

## Context and Background
The repo already contains:
- structured config and scenario profiles
- data contracts and feature/label pipelines
- baseline models and Chronos candidate plumbing
- cost-aware backtesting and reporting
- robustness tests and kill criteria
- paper-trading replay and readiness policies
- multi-objective sweep and promotion campaign tooling

The engineering core is real. The publication layer is weak. The work ahead is therefore mostly curation, documentation, packaging, evidence synthesis, and selective enhancement rather than foundational rebuilding.

## Codebase / System Analysis Summary
- Areas inspected:
  - `README.md`
  - `pyproject.toml`
  - `experiment_spec.md`
  - `CODEBASE-PLAN-V-0.1.md`
  - `docs/codebase-inventory-v0.1.0/*`
  - `src/backtest/*`
  - `src/models/chronos2_runner.py`
  - `src/reporting/decision.py`
  - `src/utils/experiment.py`
  - `config/settings.py`
  - `scripts/smoke_check.py`
  - `scripts/run_phase11_sweep.py`
  - `scripts/run_paper_trading.py`
  - representative result artifacts under `data/results/*`
  - test suite health
- Relevant modules / services:
  - `src/data`, `src/evaluation`, `src/backtest`, `src/strategy`, `src/robustness`, `src/reporting`, `src/paper_trading`
- Current behavior:
  - local test and smoke workflows pass
  - experiment artifacts and manifests are generated
  - the system can evaluate, backtest, report, and run campaign-style selection/replay workflows
- Known pain points:
  - no public boundary or ignore policy
  - outdated README
  - inconsistent story between docs and results
  - local runtime noise in tree
  - no curated visual/report pipeline for publication
  - strongest profitability evidence is partial, not readiness-grade
- Architecture notes:
  - the architecture is phase-gated and fairly well-modularized
  - the project has enough internal structure to support publication with moderate cleanup rather than major redesign

## Dependency and Interface Impact
- Internal dependencies:
  - docs and public claims must align with `src/*`, `scripts/*`, and `data/results/*`
  - any public reporting pipeline should consume existing JSON/CSV artifacts instead of duplicating metric logic
- External dependencies:
  - Python packaging via `pyproject.toml`
  - market/macro data sources
  - Chronos / torch runtime path
  - GitHub Pages for public docs surface
- API / schema / contract impact:
  - low for core system interfaces if publication work is additive
  - medium if scripts are standardized for public artifact generation
- Build / deployment impact:
  - add lightweight publication checks and docs generation
  - avoid introducing heavy new deployment systems
- Backward compatibility considerations:
  - maintain existing CLI entry points
  - prefer additive doc/report scripts over breaking refactors

## Constraints and Assumptions
### Constraints
- The project must be publishable without overstating current trading readiness.
- The repo must remain reproducible using the existing `.venv` or standard install path.
- Public publication must exclude local environment and bulky/generated noise.
- Pre-publication execution time is limited; prioritize high-signal work over exhaustive experimentation.
- The current result set is mixed; the narrative must remain evidence-first.

### Assumptions
- The owner is comfortable publishing current negative or partial-positive findings if framed correctly.
- Existing market-derived artifacts can be summarized publicly even if not all raw/generated files are shipped.
- A Markdown-first GitHub Pages site is sufficient for first publication.
- It is acceptable to improve this repo in place rather than waiting for a separate public mirror decision.

## Risks and Failure Modes
### Risk 1: Overclaiming profitability
- Impact: Severe credibility damage; portfolio value drops instead of improving.
- Detection: README, landing page, or visuals make claims stronger than campaign/promotion artifacts support.
- Mitigation: Explicit “research status” framing; separate “evidence observed” from “deployment-ready”; include negative findings section.

### Risk 2: Shipping a noisy or bloated repo
- Impact: Public repo looks amateurish and is hard to clone, review, or trust.
- Detection: Large local artifacts, caches, `.venv`, and editor metadata remain in tracked tree.
- Mitigation: strict `.gitignore`, curated artifact strategy, optional GitHub Release attachments, minimal public sample data only.

### Risk 3: Documentation drift
- Impact: Public docs contradict the actual code or results.
- Detection: README, methodology docs, and artifact summaries disagree.
- Mitigation: centralize result generation and docs references around generated JSON summaries; add a publication checklist.

### Risk 4: Reproducibility mismatch
- Impact: Portfolio reviewers cannot reproduce core workflows.
- Detection: quickstart commands fail or require undocumented steps.
- Mitigation: tighten setup docs, validate critical commands, add a public “minimal reproduction” workflow.

### Risk 5: Spending pre-publication effort on low-yield alpha chasing
- Impact: large effort, weak portfolio gain, no publishable upgrade.
- Detection: work expands into open-ended strategy experiments before public framing is fixed.
- Mitigation: stage work so repository hygiene, docs, visuals, and curated evidence land before deeper model iteration.

### Risk 6: Chronos narrative mismatch
- Impact: reviewers assume the project is primarily a Chronos success story when current strongest evidence appears EWMA-based.
- Detection: public framing emphasizes foundation models more than the observed evidence warrants.
- Mitigation: present Chronos as a candidate track with provenance/fallback safeguards, not as a proven winner.

## Proposed Approach
Use a two-track plan:

1. Publication track:
   - define a clean public boundary
   - repair repository hygiene
   - write serious documentation
   - generate curated visuals and artifact summaries
   - add GitHub Pages-ready docs

2. Research-strengthening track:
   - use existing evidence to tell a credible story
   - fill the most obvious performance/reliability gaps with bounded experiments
   - improve the highest-leverage data and calibration issues
   - only expand to more ambitious alpha work after the public core is solid

The first public release should optimize for credibility, legibility, and rigor. If stronger empirical results arrive before publication, they should improve the release. They should not be prerequisites for having a high-quality public project.

## Alternatives Considered
### Alternative A
- Description: Publish the repo immediately with minimal cleanup.
- Why not chosen: It would expose contradictory docs, local noise, and uncurated artifacts. The public quality would undersell the actual engineering.

### Alternative B
- Description: Delay publication until a robust profitable candidate is found.
- Why not chosen: This creates unnecessary delay and conflates research publication quality with strategy success. The current codebase is already publishable as research once curated properly.

### Alternative C
- Description: Rebuild the project into a polished app or dashboard before publishing.
- Why not chosen: High cost, high delay, and low marginal value relative to a strong docs-and-artifacts-first publication.

## Pre-Publish Prioritization Model

### Must-have before publication
- repository hygiene and ignore policy
- updated README and project status
- license and contribution/public use framing
- methodology docs
- curated results and figures
- GitHub Pages landing docs
- reproducible commands validated

### High-value if time permits before publication
- automated plot/report generation from existing artifacts
- small sample dataset or deterministic synthetic demo
- benchmark comparison tables and figure gallery
- strategy-status dashboard page
- more targeted campaign reruns if a clearly stronger candidate emerges

### Better after first publication
- multi-asset extension
- more ambitious ensembles or meta-labeling
- richer front-end demo beyond GitHub Pages
- live or semi-live operationalization

## Brainstormed Enhancement Backlog

### Publication / portfolio enhancements
- Create a polished README with:
  - project thesis
  - architecture diagram
  - evidence snapshot
  - current status
  - honest “what failed” section
- Add a `docs/` publication suite:
  - `docs/index.md`
  - `docs/methodology.md`
  - `docs/results.md`
  - `docs/experiment-log.md`
  - `docs/roadmap.md`
  - `docs/reproducibility.md`
- Generate a figure pack:
  - equity curves
  - cost decomposition
  - sweep frontier
  - kill-switch taxonomy summary
  - regime performance comparison
- Add a static GitHub Pages landing site with concise narrative and visual hooks.
- Add “research cards” for major experiments so the project reads like a serious lab notebook rather than a code dump.

### Engineering / UX enhancements
- Add `.gitignore`, `LICENSE`, and public artifact conventions.
- Add a `scripts/generate_public_report.py` pipeline to transform result JSON into Markdown tables and plots.
- Add a `scripts/plot_results.py` pipeline for reproducible figures.
- Add a `Makefile` or simple documented command index for common workflows.
- Add a CI job or local check for docs consistency and public-asset generation.

### Research / strategy enhancements
- Restore or replace OI / liquidation feature coverage for the relevant windows.
- Re-run matched 1h vs 4h evaluation under the same publication-ready protocol.
- Calibrate entry thresholds and active-window kill semantics with explicit deployment constraints.
- Run a narrow, evidence-driven parameter search centered on trade density vs PF / Sharpe stability.
- Validate whether Chronos adds value when the real backend is active, not fallback.
- Add regime-adaptive thresholding or abstention logic if it improves readiness without raising fragility.

### Stretch enhancements
- Add ETH as a secondary validation asset only after BTC public story is clean.
- Add a static HTML summary artifact for GitHub Pages consumption.
- Add an “interactive result explorer” later if the static publication proves valuable.

## Phase Plan

### Phase 1: Define Public Boundary and Sanitize the Repository
- Objective: Establish what is and is not part of the public repo, and remove obvious publication blockers.
- Why now: Everything else depends on a clean boundary.
- Type: Sequential
- Depends on: None
- Deliverables:
  - `.gitignore`
  - `LICENSE`
  - public artifact policy
  - cleaned repository structure plan
- Checklist:
  - [ ] Add a strict `.gitignore` for `.venv`, caches, editor metadata, and bulky/generated outputs.
  - [ ] Add a real `LICENSE` file consistent with `pyproject.toml` and `README.md`.
  - [ ] Decide and document which parts of `data/` remain in-repo, which become generated outputs, and which stay local only.
  - [ ] Create a public-facing directory convention for figures, examples, and curated artifacts.
  - [ ] Remove or exclude public-noise files from the publish path.
- Validation:
  - repo tree review
  - no local environment or cache noise in tracked/public set
  - documentation states artifact policy explicitly
- Exit criteria:
  - a reviewer can understand what the public repo includes and why
- Green-light cue:
  - publication boundary is documented and enforced by ignore rules
- Rollback / containment:
  - if in-place cleanup becomes risky, switch to a parallel public-mirror packaging path

### Phase 2: Rewrite the Project Narrative
- Objective: Replace the current contradictory public story with an evidence-based research narrative.
- Why now: Public trust depends on coherent framing more than on adding new code.
- Type: Sequential
- Depends on: Phase 1
- Deliverables:
  - rewritten `README.md`
  - project status page
  - public positioning statement
- Checklist:
  - [ ] Rewrite `README.md` to describe the true scope of the project.
  - [ ] Add a concise “Current Status” section that distinguishes implemented system capabilities from empirical readiness.
  - [ ] Add a “What the evidence currently says” section with both positive and negative findings.
  - [ ] Add a “Why this project is interesting even without live deployment” section.
  - [ ] Ensure public language does not imply profitability or production readiness beyond evidence.
- Validation:
  - doc consistency review against codebase inventory and result artifacts
- Exit criteria:
  - README, plan, and result summaries no longer contradict each other
- Green-light cue:
  - a stranger can understand the project in under five minutes
- Rollback / containment:
  - keep a short backup of the previous README if needed

### Phase 3: Build Documentation Information Architecture
- Objective: Create durable docs that make the repo legible as research and as a portfolio piece.
- Why now: The repo needs depth behind the front page.
- Type: Parallelizable
- Depends on: Phase 2
- Deliverables:
  - methodology docs
  - reproducibility docs
  - roadmap / experiment log docs
- Checklist:
  - [ ] Create `docs/methodology.md` describing data, labels, walk-forward protocol, costs, gates, and evaluation logic.
  - [ ] Create `docs/results.md` summarizing curated empirical outcomes from the most representative runs.
  - [ ] Create `docs/roadmap.md` separating current strengths, active gaps, and next experiments.
  - [ ] Create `docs/reproducibility.md` with exact environment, install, and run instructions.
  - [ ] Create `docs/experiment-log.md` or equivalent experiment card index.
- Validation:
  - each doc links to real files, commands, or artifacts
  - no “future lies” or placeholder claims
- Exit criteria:
  - public docs cover the why, how, what happened, and what comes next
- Green-light cue:
  - docs read like a coherent research project rather than scattered notes
- Rollback / containment:
  - if doc volume grows too large, compress into fewer canonical pages

### Phase 4: Curate Results and Generate Portfolio-Grade Figures
- Objective: Turn raw JSON/CSV artifacts into a selected evidence pack with compelling visuals.
- Why now: Publication quality depends on visible outputs, not just source code.
- Type: Sequential
- Depends on: Phases 1-3
- Deliverables:
  - curated artifact directory
  - reproducible figure generation scripts
  - result summary tables
- Checklist:
  - [ ] Inventory current result artifacts and select representative “showcase” runs.
  - [ ] Define a small public artifact pack for inclusion in the repo.
  - [ ] Add script(s) to generate summary tables and plots from result artifacts.
  - [ ] Generate figures for:
    - [ ] best futures tuned EWMA run
    - [ ] representative negative campaign run
    - [ ] multi-objective sweep frontier
    - [ ] kill-switch / readiness diagnostics
    - [ ] cost decomposition and drawdown profile
  - [ ] Export figures into a docs-consumable location.
- Validation:
  - figure scripts run deterministically on selected artifacts
  - numbers in docs match generated summaries
- Exit criteria:
  - the repo contains a small, high-signal set of reproducible visuals
- Green-light cue:
  - a reviewer can see evidence quality without digging into raw JSON manually
- Rollback / containment:
  - if figure generation is too broad, narrow to 3-5 essential plots first

### Phase 5: Create the GitHub Pages / Landing Experience
- Objective: Add a concise public-facing landing surface for discovery and portfolio presentation.
- Why now: GitHub Pages will amplify the project’s legibility and portfolio value.
- Type: Parallelizable
- Depends on: Phases 2-4
- Deliverables:
  - `docs/index.md` or equivalent landing page
  - GitHub Pages-ready docs structure
- Checklist:
  - [ ] Design a clean Markdown-first landing page with project thesis, architecture, evidence snapshot, and next steps.
  - [ ] Embed or link to generated figures and summary tables.
  - [ ] Add navigation links to methodology, results, reproducibility, and roadmap pages.
  - [ ] Keep visual storytelling tight and high-signal.
  - [ ] Add a publish checklist for enabling GitHub Pages.
- Validation:
  - local file review
  - broken-link check on docs references
- Exit criteria:
  - docs directory is ready to become the first public landing surface
- Green-light cue:
  - the project reads like a deliberate research portfolio, not an internal worktree
- Rollback / containment:
  - if Pages polish overruns time, keep a Markdown-only version for v1

### Phase 6: Tighten Reproducibility and Public Validation
- Objective: Ensure the public repo can be installed, tested, and demonstrated predictably.
- Why now: Publication without reproducibility is weak.
- Type: Parallelizable
- Depends on: Phases 1-4
- Deliverables:
  - validated setup path
  - minimal reproduction workflow
  - documented checks
- Checklist:
  - [ ] Confirm and document the canonical Python version and install flow.
  - [ ] Validate quickstart commands against the cleaned public assumptions.
  - [ ] Add a minimal public demo path using smoke or synthetic data when full data is not available.
  - [ ] Consider a lightweight task runner or command index for install, test, smoke, and report generation.
  - [ ] Ensure CI messaging and docs are aligned.
- Validation:
  - test suite
  - smoke check
  - at least one documented public demo flow
- Exit criteria:
  - the public repo has a reproducible “first successful run” path
- Green-light cue:
  - setup friction is low and explicit
- Rollback / containment:
  - if full workflow is too heavy, keep a minimal smoke/demo workflow as the public default

### Phase 7: Strengthen the Research Story Using Existing Evidence
- Objective: Turn the current mixed result set into a high-credibility research narrative.
- Why now: The existing results are already useful if framed correctly.
- Type: Sequential
- Depends on: Phases 2-4
- Deliverables:
  - research summary section
  - experiment cards
  - honest findings matrix
- Checklist:
  - [ ] Summarize the strongest positive evidence currently available.
  - [ ] Summarize the strongest negative evidence currently available.
  - [ ] Explain why the governance stack rejected promotion despite PF-positive regions.
  - [ ] Add an explicit matrix of “implemented / evidence-positive / evidence-negative / unresolved”.
  - [ ] Frame the project as a disciplined search process rather than a result cherry-pick.
- Validation:
  - every claim maps to inspected artifacts
- Exit criteria:
  - the project is credible even before additional experiments land
- Green-light cue:
  - negative results feel informative, not embarrassing
- Rollback / containment:
  - if the narrative becomes too sprawling, compress into a single canonical results page

### Phase 8: High-Leverage Pre-Publish Enhancement Experiments
- Objective: Improve the project’s substantive research value before publication without drifting into open-ended alpha hunting.
- Why now: A small number of targeted improvements can materially improve the public story.
- Type: Sequential
- Depends on: Phases 4, 6, 7
- Deliverables:
  - bounded experiment backlog
  - rerun summaries for selected improvements
- Checklist:
  - [ ] Restore or improve OI / liquidation feature coverage if feasible within bounded effort.
  - [ ] Run a matched 1h vs 4h comparison under publication-ready reporting.
  - [ ] Reassess threshold and kill-policy calibration using the multi-objective tooling.
  - [ ] Validate whether Chronos real-backend runs add value beyond fallback / baseline behavior.
  - [ ] Only keep experiment branches that produce clearer evidence or learning value.
- Validation:
  - targeted rerun artifacts
  - comparison tables and figures
  - updated docs reflecting new findings
- Exit criteria:
  - either evidence improves, or the learning is documented clearly enough to strengthen the publication anyway
- Green-light cue:
  - each experiment produces a publishable conclusion, not just more files
- Rollback / containment:
  - stop after bounded experiment count if signal quality does not improve

### Phase 9: Final Publication Packaging and Launch Readiness
- Objective: Prepare the final public release surface and launch checklist.
- Why now: Publication should happen only after the public experience is coherent.
- Type: Sequential
- Depends on: Phases 1-8
- Deliverables:
  - launch checklist
  - release notes
  - public-ready repo state
- Checklist:
  - [ ] Review repo tree for public cleanliness one final time.
  - [ ] Ensure README, docs, figures, and artifact pack are internally consistent.
  - [ ] Add release notes or a `PUBLICATION_NOTES.md` summary if useful.
  - [ ] Confirm GitHub Pages setup steps.
  - [ ] Prepare a publication summary suitable for repo description / pinned project text.
- Validation:
  - manual repo walkthrough
  - docs link check
  - tests and smoke rerun
- Exit criteria:
  - repo is ready for push and public viewing
- Green-light cue:
  - a cold reviewer can understand the project quickly and trust the evidence trail
- Rollback / containment:
  - if a final inconsistency appears, hold launch and fix the narrative rather than shipping around it

## Milestones

### Milestone 1: Public Boundary and Narrative Reset
- Target outcome: The repo is no longer structurally embarrassing to publish.
- Includes phases: 1, 2
- Completion criteria:
  - `.gitignore` and `LICENSE` exist
  - README is rewritten
  - public framing is evidence-aligned
- Evidence / demo expected:
  - clean top-level repo surface
  - coherent public README
- Go / no-go note:
  - No-go if the README still overclaims or contradicts internal artifacts

### Milestone 2: Documentation and Visual Evidence Pack
- Target outcome: The project has a portfolio-grade docs and artifact layer.
- Includes phases: 3, 4, 5, 6
- Completion criteria:
  - docs IA exists
  - figures generated
  - GitHub Pages-ready landing surface exists
  - reproducible quickstart is validated
- Evidence / demo expected:
  - docs pages
  - plot pack
  - validated commands
- Go / no-go note:
  - No-go if visuals are manual-only or docs rely on unsupported claims

### Milestone 3: Credible Research Story
- Target outcome: The repo explains not only what was built, but what was learned.
- Includes phases: 7
- Completion criteria:
  - findings matrix exists
  - positive and negative evidence both documented
- Evidence / demo expected:
  - results summary page
  - experiment cards
- Go / no-go note:
  - No-go if failures are hidden or unexplained

### Milestone 4: Pre-Publish Enhancement Pass and Launch
- Target outcome: The project is both polished and substantively improved where it matters most.
- Includes phases: 8, 9
- Completion criteria:
  - bounded enhancement experiments completed or explicitly cut
  - launch checklist passed
- Evidence / demo expected:
  - updated figures and docs
  - final publish-ready repo surface
- Go / no-go note:
  - Go once public credibility is high, even if profitability remains partial or unresolved

## Validation Strategy
### Automated validation
- Unit tests:
  - full `pytest -q` pass remains required
- Integration tests:
  - smoke workflow remains required
  - figure/report generation scripts should have at least basic validation
- End-to-end tests:
  - public quickstart path should be exercised
- Static analysis / lint / type checks:
  - existing CI lint gate should continue to pass

### Manual validation
- Manual test flows:
  - read README cold from top to bottom
  - navigate docs as a first-time visitor
  - verify figures map to real artifact inputs
  - check that public claims match summary JSONs
- Inspection points:
  - top-level repo tree
  - docs navigation
  - figures
  - artifact curation policy

### Acceptance thresholds
- Functional:
  - setup, tests, and smoke flow succeed
- Performance:
  - figure generation and docs build remain lightweight enough for local iteration
- Reliability:
  - no contradictory public claims
- Security:
  - no secrets or private environment/state shipped
- Usability / DX:
  - first-time reviewer can understand the project quickly

## Rollout / Migration Strategy
- Rollout pattern:
  - execute in-place cleanup and publication prep
  - optionally mirror into a dedicated public repo after the cleaned version stabilizes
- Feature flags:
  - none required for publication layer
  - keep research/runtime toggles explicit in scripts where needed
- Backward compatibility:
  - retain existing CLI behavior where possible
- Data migration:
  - do not migrate all local result files into public tracking
  - curate only selected artifacts and document regeneration paths
- Rollback path:
  - if public packaging becomes risky in-place, create a public mirror with curated contents only
- Safe deployment notes:
  - publish only after public surface is consistent and validated

## Observability / Debugging Plan
- Logs:
  - keep existing script logging and run manifests
- Metrics:
  - use generated summary JSONs as canonical result inputs
- Traces:
  - not required for first public release
- Alerts:
  - not required for publication track
- Debug hooks:
  - use smoke and test suite as publication gate
  - keep generated figure/report scripts deterministic and inspectable
- Failure signals:
  - docs mismatch
  - plot-generation failures
  - quickstart drift
  - inconsistent numbers between docs and artifacts

## Open Questions
- [ ] Should the first public release happen in this repo directly or through a curated public mirror?
- [ ] Should large curated result bundles live in Git history, GitHub Releases, or remain downloadable/generated only?
- [ ] Is a Markdown-only GitHub Pages site sufficient for v1, or is a richer static HTML report worth the extra effort?

## Decision Log
### Decision 1
- Decision: Optimize for a research-grade publication, not a “profitable system” pitch.
- Why: Current evidence supports rigor and partial edge discovery, not production-ready profitability.
- Alternatives rejected:
  - delay publication until strong alpha is proven
  - market the current repo as production-like
- Implications:
  - the project becomes credible faster and remains honest

### Decision 2
- Decision: Use existing artifact and reporting machinery as the basis for public summaries and figures.
- Why: This minimizes drift and leverages current code strengths.
- Alternatives rejected:
  - manual curation without generation scripts
  - building a separate reporting stack
- Implications:
  - public docs remain closer to the code and are easier to maintain

### Decision 3
- Decision: Stage enhancement work after repository hygiene and narrative repair.
- Why: Portfolio value is bottlenecked by presentation and coherence more than by missing model experiments.
- Alternatives rejected:
  - open-ended strategy tuning before cleanup
- Implications:
  - faster path to a strong public release with lower execution risk

## Implementation Log
| Date | Phase / Step | What was done | Files / Modules touched | Validation run | Outcome | Notes / blockers |
|---|---|---|---|---|---|---|
| 2026-04-11 | Planning / inspection | Inspected repo structure, key modules, docs, results, and validation state | `README.md`, `pyproject.toml`, `experiment_spec.md`, `CODEBASE-PLAN-V-0.1.md`, `docs/codebase-inventory-v0.1.0/*`, `src/*`, `scripts/*`, `tests/*`, `data/results/*` | `./.venv/bin/pytest -q`, `./.venv/bin/python scripts/smoke_check.py` | Success | Repo is technically stronger than publicly presented; public packaging remains the main issue |
| 2026-04-11 | Phase 1 | Added public-boundary ignore policy, license, and artifact/data publication notes | `.gitignore`, `LICENSE`, `data/README.md`, `artifacts/public/README.md` | manual tree review, `git status --short --ignored` | Success | Local runtime/data noise is now excluded from the intended public surface |
| 2026-04-11 | Phases 2-3 | Rewrote README and added a publication docs suite with methodology, results, roadmap, reproducibility, experiment log, and project status | `README.md`, `docs/index.md`, `docs/methodology.md`, `docs/results.md`, `docs/roadmap.md`, `docs/reproducibility.md`, `docs/experiment-log.md`, `docs/project-status.md` | manual docs review | Success | Public narrative now matches inspected evidence and current repo scope |
| 2026-04-11 | Phases 4-6 | Added public report and plotting scripts, Makefile targets, and generated curated evidence + figures | `scripts/generate_public_report.py`, `scripts/plot_public_results.py`, `pyproject.toml`, `Makefile`, `artifacts/public/*`, `docs/assets/*` | `./.venv/bin/python scripts/generate_public_report.py`, `./.venv/bin/python scripts/plot_public_results.py` | Success | Repo now has a reproducible publication artifact pipeline |
| 2026-04-11 | Phases 7-8 | Strengthened research story and ran bounded 4h sensitivity experiments for spot and margin | `docs/results.md`, `docs/experiment-log.md`, `data/results/publication_4h_*` | four `scripts/run_paper_trading.py` runs on 4h spot/margin default and looser thresholds | Success | New finding: 4h default threshold is too conservative; looser threshold recovers PF-positive / positive-Sharpe regions but still fails readiness due to kill-switch activity |
| 2026-04-11 | Phase 9 | Added publication notes, initialized local Git repo, and reran final validation and artifact generation | `PUBLICATION_NOTES.md`, `.git/` | `./.venv/bin/pytest -q`, `./.venv/bin/python scripts/smoke_check.py`, `./.venv/bin/python scripts/generate_public_report.py`, `./.venv/bin/python scripts/plot_public_results.py`, `git init -b main`, `git status --short --ignored` | Success | Local repo is now commit-ready and publication-oriented |

## Final Approval Gate
- [x] Scope is clear
- [x] Critical unknowns are resolved
- [x] Plan is evidence-based
- [x] Phases are actionable
- [x] Validation is defined
- [x] Rollout / rollback is defined where relevant
- [x] Parallelizable work is clearly marked
- [x] Ready for implementation
