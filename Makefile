.PHONY: install-dev test test-cov lint format contracts ci

PYTHON ?= .venv/bin/python

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

test-cov:
	$(PYTHON) -m pytest --cov=opening_strength_fit --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

contracts:
	$(PYTHON) -m opening_strength_fit.cli.audit_experiments
	$(PYTHON) -m opening_strength_fit.cli.project_contracts

ci: lint test
