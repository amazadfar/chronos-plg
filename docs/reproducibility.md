# Reproducibility

## Environment

The publication baseline is intended to run from an isolated repository environment.
During the publication pass, validation uses Python `3.12.x` in a local `.venv`.

Recommended setup:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## Validation Commands

Run the automated test suite:

```bash
make test
```

Run the fast smoke check:

```bash
make smoke
```

## Generate the Public Evidence Pack

```bash
make public-assets
```

This runs:
- `scripts/generate_public_report.py`
- `scripts/plot_public_results.py`

Outputs:
- `artifacts/public/`
- `docs/assets/`

## Data Policy

The public repo does not include the full local dataset and result tree.

Not shipped:
- `data/raw/`
- `data/processed/`
- `data/results/`

Shipped instead:
- curated summaries in `artifacts/public/`
- publication figures in `docs/assets/`
- reproducible commands and methodology docs

See also [data/README.md](../data/README.md).

## Core Research Commands

Examples:

```bash
python scripts/smoke_check.py
python scripts/run_paper_trading.py --timeframe 1h --model ewma --scenario binance_spot_taker_discounted --entry-policy threshold --output-dir data/results/phase11_1h_paper_spot_threshold
python scripts/run_phase11_sweep.py --timeframe 1h --scenario binance_spot_taker_discounted --model ewma --start-date 2025-12-01 --output-dir data/results/phase11_5_sweep_spot_full
python scripts/run_phase11_promotion_campaign.py --timeframe 1h --ranked-candidates data/results/phase11_5_sweep_spot_one/phase11_sweep_ranked.json --output-dir data/results/phase11_7_campaign_spot_one
```

## Notes

- some deeper experiment paths depend on data already existing locally
- the public-first reproduction path is:
  1. install
  2. run tests
  3. run smoke
  4. generate public reports and figures

That path is intentionally lighter than full artifact regeneration.
