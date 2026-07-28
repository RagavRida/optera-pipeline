.PHONY: install install-dev run optimized baseline cheap sweep variance test test-cov lint clean

install:
	python3 -m pip install -r requirements.txt

install-dev:
	python3 -m pip install -r requirements-dev.txt

# The one command: naive baseline vs optimised pipeline, with scoring.
run:
	python3 run.py --mode both

optimized:
	python3 run.py --mode optimized

baseline:
	python3 run.py --mode baseline

cheap:
	python3 run.py --mode optimized --profile cheap

# Evidence behind the per-class policy table in optera/config.py
sweep:
	python3 scripts/sweep_resolution.py

# How much does accuracy move when nothing changes? (noise floor)
variance:
	python3 scripts/variance.py

# ---- Testing & quality ----
test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -v

test-cov:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p pytest_cov tests/ --cov=optera --cov-report=term-missing

lint:
	python3 -m py_compile optera/*.py run.py

clean:
	rm -rf out __pycache__ optera/__pycache__ scripts/__pycache__ tests/__pycache__ .pytest_cache htmlcov .coverage
