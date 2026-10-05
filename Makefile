# Common tasks. `make help` lists them.
PY ?= .venv/bin/python
BIN := $(dir $(PY))

.PHONY: help venv lock check lint types test test-fast policy demo audit-deps

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

venv:  ## create .venv with the locked ML stack (CPU PyTorch)
	python3.12 -m venv .venv
	$(BIN)pip install --require-hashes -r requirements/ml.txt
	$(BIN)pip install --no-deps -e .

lock:  ## regenerate both hash-locked requirement files
	$(BIN)pip-compile -q --generate-hashes --strip-extras --allow-unsafe --extra dev \
	  -o requirements/dev.txt pyproject.toml
	$(BIN)pip-compile -q --generate-hashes --strip-extras --allow-unsafe \
	  --extra train --extra serve --extra otel --extra cloud --extra dev \
	  --extra-index-url https://download.pytorch.org/whl/cpu -o requirements/ml.txt pyproject.toml

lint:  ## ruff lint + format check
	$(BIN)ruff check . && $(BIN)ruff format --check .

types:  ## mypy --strict
	$(BIN)mypy

test:  ## full test suite (needs opa and cosign on PATH for registry/serving tests)
	$(BIN)pytest -q

test-fast:  ## governance tests only (no PyTorch)
	$(BIN)pytest -q -m "not ml"

policy:  ## promotion policy tests
	opa fmt --list --fail src/lineage/policy && opa check --strict src/lineage/policy && \
	  opa test src/lineage/policy

audit-deps:  ## pip-audit both lock files
	PATH=$(BIN):$$PATH $(PY) scripts/audit_deps.py requirements/dev.txt
	PATH=$(BIN):$$PATH $(PY) scripts/audit_deps.py requirements/ml.txt

check: lint types test policy  ## everything CI checks, locally

demo:  ## the whole lifecycle on the example (about 25 min on CPU)
	PATH=$(BIN):$$PATH examples/triage/demo.sh
