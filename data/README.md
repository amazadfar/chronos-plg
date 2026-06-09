# Data Policy

This repository does not publish the full local `data/` tree.

Why:
- the local dataset and result store is large
- many artifacts are generated, iterative, and noisy
- the public repository should stay lightweight and reproducible

What stays public:
- documentation that describes the datasets and generated outputs
- curated summaries and figures under `artifacts/public/`
- commands required to regenerate the key artifacts

What stays local:
- raw downloads under `data/raw/`
- processed datasets under `data/processed/`
- full experiment outputs under `data/results/`

To rebuild data and experiment outputs, use the documented commands in:
- `README.md`
- `docs/reproducibility.md`
- `docs/results.md`
