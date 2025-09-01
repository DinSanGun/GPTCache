
# Cost-Aware Eviction in GPTCache — Benchmark & Policy Repo

This repository contains a **reproducible benchmark harness** and a **Cost-Aware eviction policy** extension for GPTCache.
It supports **EXACT** and **SEMANTIC** cache modes, **mock** and **real (Ollama)** backends, and logs structured results
for analysis (CSV + YAML knobs). The repo is designed to live **inside a GPTCache clone** so local edits to GPTCache affect runs.

---

## Highlights

- ✅ **Cost-Aware eviction**: preserves items with higher recomputation cost (measured on cold pass).
- ✅ **Mock + Real** backends: synthetic latency model + local **Ollama** smoke tests.
- ✅ **Reproducible outputs**: per-run `summary.csv`, `detail.csv`, `knobs.yaml`, `sys_detail.csv` in `results/`.
- ✅ **Unit tests**: policy behavior, cost-provider wiring, and runner smoke tests.
- ✅ **Docker**: build & run mock or real smoke in a container.
- ✅ **CI**: GitHub Actions workflow for lint + tests (+ optional mock smoke).

---

## Why GPTCache as the Baseline

We chose GPTCache because of its maturity and modular design, active community, and clean extension points:
- **Maturity & activity** — healthy GitHub signals; maintained docs & examples.
- **Adapters & compatibility** — OpenAI- and LangChain-style adapters; easy to point to **Ollama**.
- **Exact & semantic** — interchangeable vector stores (FAISS/Milvus) for realistic paraphrase hits.
- **In-memory eviction via `cachetools`** — minimal surface-area changes to swap/extend policies.

These attributes minimized framework plumbing and let us focus on policy research.

---

## Repository Layout

```
.
├── configs/                     # YAML configs (mock/real + experiments + knobs)
│   ├── mock-exact.yaml
│   ├── mock-semantic.yaml
│   ├── ollama-semantic.yaml
│   ├── real-exact.yaml
│   ├── experiments/
│   │   ├── mock-exact-costaware.yaml
│   │   └── mock-semantic-costaware.yaml
│   └── knobs/
│       ├── all_tunables.yaml
│       ├── costaware_largeN.yaml
│       └── costaware_showcase.yaml
├── data/                        # datasets (LMSYS subset ≤480 tokens, mixed tiny set)
│   ├── lmsys_user_prompts.txt
│   ├── lmsys_user_prompts_heavy.txt
│   └── prompts_mixed.txt
├── src/
│   ├── bench/                   # runner, metrics, cache init, plotting
│   └── ext/                     # Cost-Aware policy + tests
├── tools/                       # CLI: bench, dataset preprocess, plotting helpers
├── results/                     # run outputs (artifacts + CSVs) — gitignored
├── docker/                      # Dockerfile + entrypoint
├── .github/workflows/           # CI workflow (GitHub Actions)
├── requirements.txt             # core deps (mock + analysis)
└── pytest.ini
```

**Important:** `results/`, `__pycache__/`, `.pytest_cache/`, `*.db`, `*.index`, `*.faiss` are **gitignored**.

---

## Installation

### 1) Clone GPTCache and place this repo inside it

```
C:/dev/LLM-Caching/
├── GPTCache/            # your fork: github.com/DinSanGun/GPTCache
│   └── BGU-LLM-Cache-Project/  # this repo
```

This lets `import gptcache` resolve to your **local** GPTCache (editable development).

### 2) Create venv & install deps

**PowerShell (Windows)**
```powershell
cd BGU-LLM-Cache-Project
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

**bash (Linux/macOS)**
```bash
cd BGU-LLM-Cache-Project
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> For **semantic** runs with ONNX embeddings, install: `pip install onnxruntime` (or use Docker).  
> For FAISS vector store on Linux: `pip install faiss-cpu`; Windows users typically skip FAISS or use WSL.

---

## Quickstart (Mock)

**EXACT (fastest; isolates eviction)**
```bash
python tools/bench_policies.py   --mode mock   --base-config configs/mock-exact.yaml   --dataset data/prompts_mixed.txt   --prompts 120   --max-size 64 --clean-size 8 --warm-repeats 1   --policies FIFO LFU COST_AWARE   --out-root results/suite/mock_exact/churn128
```

**SEMANTIC (embedding + similarity threshold)**
```bash
python tools/bench_policies.py   --mode mock   --base-config configs/mock-semantic.yaml   --dataset data/lmsys_user_prompts_heavy.txt   --prompts 120 --sim-thr 0.98   --max-size 64 --clean-size 8 --warm-repeats 1   --policies FIFO LFU COST_AWARE   --out-root results/suite/mock_semantic/churn128
```

**Where outputs go**
```
results/suite/<mode>/<scenario>/<backend>/<policy>_<timestamp>/
  ├── summary.csv      # mean, p95, p99, throughput, warm_hit, cost_saved_total
  ├── detail.csv       # per-request traces
  ├── knobs.yaml       # effective knobs (capacity, clean_size, hi_frac, jitter, ...)
  ├── sys_detail.csv   # CPU/RSS sampler
  └── artifacts/       # sqlite.db, faiss.index (semantic)
```

---

## Real Backend (Ollama) — Smoke Test

**Pull a light model & run the service**
```bash
ollama pull llama3.2:1b   # or: ollama pull phi3:mini
ollama run llama3.2:1b "hello"
```

**EXACT smoke**
```bash
python tools/bench_policies.py   --mode ollama   --base-config configs/real-exact.yaml   --dataset data/prompts_mixed.txt   --prompts 8   --max-size 16 --clean-size 2 --warm-repeats 1   --policies FIFO COST_AWARE   --out-root results/suite/real_exact/smoke_local
```

**SEMANTIC smoke**
```bash
python tools/bench_policies.py   --mode ollama   --base-config configs/ollama-semantic.yaml   --dataset data/lmsys_user_prompts_heavy.txt   --prompts 10 --sim-thr 0.98   --max-size 32 --clean-size 2 --warm-repeats 1   --policies FIFO COST_AWARE   --out-root results/suite/real_semantic/smoke_local
```

---

## Configuration Knobs

You can override via CLI or environment variables:

- `CACHE_EVICTION` = `FIFO` | `LFU` | `COST_AWARE`
- `CACHE_MAX_SIZE` = `64` (int)
- `CACHE_CLEAN_SIZE` = `8` (int)
- `CACHE_SIMILARITY` = `distance` (semantic evaluator)
- `CACHE_SIMILARITY_THRESHOLD` = `0.98`
- `RUN_WARM_REPEATS` = `1`
- `RUN_SHUFFLE_WARM` = `true`
- `RUN_STRICT_COLD` = `true`

Example (PowerShell):
```powershell
$env:CACHE_EVICTION="COST_AWARE"
$env:CACHE_MAX_SIZE="16"
$env:CACHE_CLEAN_SIZE="2"
```

---

## Unit Tests

Run all tests:
```bash
pytest -q
```

Select markers:
```bash
pytest -m "not slow" -q
```

Tests cover:
- **Cost-Aware policy**: min-cost eviction, getsizeof handling, decay/EMA (if enabled).
- **Cost provider plumbing**: cold-latency capture on miss; mapping prompt→cache_id.
- **Runner smoke**: tiny mock pipeline to ensure CLI & logging continue to work.

> Real-Ollama tests are **skipped** by default unless `RUN_OLLAMA_TESTS=1` is set (to avoid CI flakiness).

---

## Docker

**Build**
```bash
docker build -t llm-cache-bench -f Dockerfile .
```

**Run mock benchmark**
```bash
docker run --rm -it   -v "$PWD/results:/app/results"   llm-cache-bench   python tools/bench_policies.py     --mode mock     --base-config configs/mock-exact.yaml     --dataset data/prompts_mixed.txt     --prompts 60     --max-size 32 --clean-size 4 --warm-repeats 1     --policies FIFO COST_AWARE     --out-root results/suite/mock_exact/docker_smoke
```

**Run real smoke (host Ollama)**
```bash
# ensure Ollama is running on host: http://localhost:11434
docker run --rm -it   --network host   -v "$PWD/results:/app/results"   llm-cache-bench   python tools/bench_policies.py     --mode ollama     --base-config configs/real-exact.yaml     --dataset data/prompts_mixed.txt     --prompts 8     --max-size 16 --clean-size 2 --warm-repeats 1     --policies FIFO COST_AWARE     --out-root results/suite/real_exact/docker_smoke
```

---

## CI (GitHub Actions)

Create `.github/workflows/ci.yaml`:

```yaml
name: CI
on:
  push:
    branches: [ main, master ]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [ "3.10", "3.11" ]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install -U pip
          pip install -r requirements.txt
          # optional real/semantic extras:
          # pip install onnxruntime faiss-cpu

      - name: Unit tests
        run: |
          pytest -q -m "not slow"

      # Optional: tiny mock smoke to catch regressions (fast)
      - name: Mock smoke (optional)
        run: |
          python tools/bench_policies.py             --mode mock             --base-config configs/mock-exact.yaml             --dataset data/prompts_mixed.txt             --prompts 12             --max-size 16 --clean-size 2 --warm-repeats 1             --policies FIFO COST_AWARE             --out-root results/suite/mock_exact/ci_smoke
```

Tips:
- Protect `main` with required status checks (“test” job).
- Keep **real Ollama** out of CI by default (depends on runner hardware).
- Upload `results/` as an artifact if you want to inspect run CSVs.

---

## Refactor & Cleanup Checklist

- [ ] **Remove committed run outputs**: delete `results/` contents; keep the folder (gitignored).
- [ ] **Delete `.pytest_cache/` and `__pycache__/`** if present.
- [ ] **Move scratch scripts** (`src/quicktest.py`) to `examples/` or delete if obsolete.
- [ ] Ensure consistent naming: `mock-exact.yaml`, `mock-semantic.yaml`, `real-exact.yaml`, `ollama-semantic.yaml`.
- [ ] Add `reset_state()` in `src/ext/cost_aware.py` (to clear cost maps for tests).
- [ ] Wrap cost maps with a `threading.Lock()` if multi-threaded runs are expected.
- [ ] Keep *policy selection* in configs/CLI only; avoid hardcoding in code.
- [ ] Add docstrings to `src/bench/runner.py` CLI and `tools/bench_policies.py`.
- [ ] Pin optional versions for reproducibility in Docker or lock files.

---

## References & Links

- Project (this fork): https://github.com/DinSanGun/GPTCache  
- Original GPTCache: https://github.com/zilliztech/GPTCache  
- LMSYS-Chat-1M dataset: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
- Previous version of this project (different approach): https://github.com/rulidor/llm_cache_project

---

## License

This repository builds on GPTCache. See original project license for details; any new code here inherits the same license unless otherwise stated.
