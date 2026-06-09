PYTHON := ./.venv/bin/python
PYTEST := ./.venv/bin/pytest

.PHONY: test lint smoke public-report public-plots public-assets validate-public

test:
	$(PYTEST) -q

lint:
	$(PYTHON) -m ruff check src config scripts tests --select E9,F63,F7,F82

smoke:
	$(PYTHON) scripts/smoke_check.py

public-report:
	$(PYTHON) scripts/generate_public_report.py

public-plots:
	$(PYTHON) scripts/plot_public_results.py

public-assets: public-report public-plots

validate-public: lint test smoke public-assets
