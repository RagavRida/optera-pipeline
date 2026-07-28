.PHONY: install run optimized baseline cheap sweep variance clean

install:
	python3 -m pip install -r requirements.txt

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

clean:
	rm -rf out __pycache__ optera/__pycache__ scripts/__pycache__
