
# Makefile for BGU-LLM-Cache-Project (cross-shell friendly)
# Usage examples:
#   make setup        # create venv, upgrade pip, install GPTCache (editable), install project deps
#   make setup-real   # same as setup + install real-mode deps
#   make bench-mock   # run mock benchmark
#   make bench-real   # run real/ollama benchmark
#   make plots        # generate plots from the latest mock run
#   make clean        # remove venv and results (careful)

# Helper macros (try Unix first, then Windows venv path)
PY_VENV := .venv/bin/python
ifeq ("$(OS)","Windows_NT")
  PY_VENV := .venv/Scripts/python.exe
endif

PIP_VENV := $(PY_VENV) -m pip

.PHONY: setup setup-real bench-mock bench-real plots clean

setup:
\tpython -m venv .venv
\t$(PY_VENV) -m pip install -U pip
\t# Install GPTCache (editable) from the parent folder
\t$(PY_VENV) -m pip install -e ..
\t# Install project dependencies
\t$(PY_VENV) -m pip install -r requirements.txt
\t@echo ""
\t@echo "✅ Setup complete."
\t@echo "Activate the venv before running (if not already active):"
\t@echo "  Windows: .venv\\Scripts\\Activate"
\t@echo "  macOS/Linux: source .venv/bin/activate"

setup-real: setup
\t$(PY_VENV) -m pip install -r requirements-real.txt

bench-mock:
\t$(PY_VENV) -m src.bench.runner --config configs/mock.yaml

bench-real:
\t$(PY_VENV) -m src.bench.runner --config configs/real.yaml

plots:
\t$(PY_VENV) -m src.bench.plot_results --csv results/run-mock/detail.csv --out plots

clean:
\t-@rm -rf .venv results plots 2>/dev/null || true
\t-@powershell -Command "Remove-Item -Recurse -Force .venv, results, plots" 2>$NUL || exit 0
