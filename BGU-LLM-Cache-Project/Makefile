.PHONY: setup test bench-mock bench-real plots

setup:
	python -m venv .venv && (. .venv/bin/activate || . .venv/Scripts/Activate) && pip install -U pip && pip install -r requirements.txt

test:
	pytest -q

bench-mock:
	python -m src.bench.runner --config configs/mock.yaml

bench-real:
	python -m src.bench.runner --config configs/real.yaml

plots:
	python -m src.bench.plots results/*/*.csv
