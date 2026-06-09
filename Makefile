PYTHON ?= ./.venv/bin/python
PYTEST ?= ./.venv/bin/pytest

.PHONY: test smoke public-report public-plots public-assets

test:
	$(PYTEST) -q

smoke:
	$(PYTHON) scripts/smoke_check.py

public-report:
	$(PYTHON) scripts/generate_public_report.py

public-plots:
	$(PYTHON) scripts/plot_public_results.py

public-assets: public-report public-plots
