# Publication Notes

## What Changed In This Publication Pass

- added a strict public-boundary `.gitignore`
- added `LICENSE`
- added curated public artifact policy under `data/README.md` and `artifacts/public/README.md`
- rewrote the top-level `README.md`
- added a publication docs suite under `docs/`
- added public summary and plotting scripts
- generated a curated evidence pack under `artifacts/public/`
- generated publication figures under `docs/assets/`
- ran bounded 4h sensitivity experiments to strengthen the public research story

## Public Positioning

Publish this project as:

`A governed BTC trading research platform with cost-aware evaluation, paper-trading readiness controls, and honest positive / negative evidence.`

Do not publish it as:

`A profitable live trading system`

That would be stronger wording than the current evidence supports.

## Suggested GitHub Repo Description

`Governed BTC trading research platform with probabilistic forecasting, cost-aware backtesting, paper-trading readiness controls, and curated public evidence.`

## Suggested Topics

- `quant`
- `trading`
- `time-series`
- `backtesting`
- `forecasting`
- `chronos`
- `machine-learning`
- `research`
- `python`
- `btc`

## GitHub Pages Suggestion

Use the `docs/` directory as the Pages source.

Suggested first-page path:
- `docs/index.md`

## Pre-Push Checklist

- run `make test`
- run `make smoke`
- run `make public-assets`
- review `README.md`
- review `docs/results.md`
- review `docs/project-status.md`
- review `artifacts/public/public_evidence_snapshot.md`

## First Commit Strategy

Keep the first public commit intentionally scoped to:
- source code
- tests
- docs
- publication scripts
- curated artifacts

Avoid the temptation to include the full local `data/` or environment tree.
