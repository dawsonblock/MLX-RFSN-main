# RFSN v10 — common development tasks

.PHONY: install test lint check clean

install:
	python -m pip install --upgrade pip
	pip install -e ".[mlx,dev,real_model,production]"

test:
	pytest -q -rs

test-drift:
	pytest tests/test_drift.py tests/test_attention.py -v

lint:
	ruff check rfsn_v10 tests

typecheck:
	mypy rfsn_v10

check:
	python scripts/check_release_integrity.py
	python test_syntax.py
	python test_agent_core_integration.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf build dist *.egg-info
