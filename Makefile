.PHONY: install-dev install-cluster smoke test test-cov lint format contracts ci

PYTHON ?= .venv/bin/python

install-dev:
	$(PYTHON) -m pip install -c requirements.lock -e ".[dev]"

install-cluster:
	$(PYTHON) -m pip install -c requirements.lock -e ".[dev,cluster]"

smoke:
	$(PYTHON) -m opening_strength_fit.commands.experiment_run \
		--config examples/smoke/ridge.toml \
		--output-dir output/smoke

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
	$(PYTHON) -m opening_strength_fit.commands.experiment_audit
	$(PYTHON) -m opening_strength_fit.commands.project_contracts

ci: lint test
