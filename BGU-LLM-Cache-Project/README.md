# BGU-LLM-Cache-Project (embedded in GPTCache fork)

This folder contains your **single-repo** benchmark project that lives inside your GPTCache fork.
It assumes the parent directory is the GPTCache project root so `import gptcache` resolves locally.

## Quick start (mock)

```bash
cd BGU-LLM-Cache-Project
python -m venv .venv
# Windows PowerShell
. .\.venv\Scripts\Activate
# Linux/macOS
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

python -m src.bench.runner --config configs/mock.yaml
python -m src.bench.plots results/run-mock/*.csv
```

## Quick start (real / Ollama)

```bash
pip install -r requirements-real.txt
python -m src.bench.runner --config configs/real.yaml
```

Artifacts (SQLite/FAISS/CSVs/plots) are written under `BGU-LLM-Cache-Project/results/` only.

## Layout

- `configs/` — YAML configs for mock/real runs
- `src/bench/` — runners, metrics, plotting, GPTCache init
- `src/bench/clients/` — `mock_llm.py`, `ollama_client.py` (todo)
- `src/ext/` — your GPTCache extensions & unit tests
- `data/` — prompt corpora (place `prompts_raw.csv` here if needed)
- `results/` — outputs (CSV, plots, artifacts)
- `docker/` — dockerfile + entrypoint
- `.github/workflows/` — CI example

## Notes

- Keep changes to `gptcache/` minimal; prefer flags/config toggles to enable your extension.
- Record configuration into each CSV (a JSON blob column) for reproducibility.
