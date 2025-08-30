from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Dict, Any, List, Deque
from collections import deque
import pandas as pd
import yaml

# Ensure GPTCache root is importable
from . import sitecustomize  # noqa: F401

from .metrics import SysSampler, percentiles

# --- Simple cache tracker to estimate hit/miss consistent with config ---
class CacheTracker:
    def __init__(self, capacity: int, clean_size: int, policy: str = "LRU"):
        self.capacity = max(1, int(capacity))
        self.clean_size = max(1, int(clean_size))
        self.policy = policy.upper()
        self.q: Deque[str] = deque()  # for LRU/FIFO; left = most recent for LRU

    def access(self, key: str) -> bool:
        """Return True if hit, False if miss; update internal state (LRU/FIFO)."""
        if self.policy == "LRU":
            # Hit: move to front
            try:
                self.q.remove(key)
                self.q.appendleft(key)
                return True
            except ValueError:
                # Miss: insert at front
                self.q.appendleft(key)
                # Evict if needed
                while len(self.q) > self.capacity:
                    for _ in range(self.clean_size):
                        if self.q:
                            self.q.pop()  # remove least-recent (right)
                return False
        else:  # FIFO
            if key in self.q:
                return True
            # miss
            self.q.append(key)
            while len(self.q) > self.capacity:
                for _ in range(self.clean_size):
                    if self.q:
                        self.q.popleft()  # remove oldest
            return False

def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(out_dir: Path, artifacts_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    mode = cfg.get("mode","mock")
    cache_cfg = cfg.get("cache",{})
    paths = cfg.get("paths",{})
    run = cfg.get("run",{})
    metrics_cfg = cfg.get("metrics",{})

    out_dir = Path(paths.get("out_dir","results/run"))
    artifacts_dir = Path(paths.get("artifacts_dir","results/artifacts"))
    ensure_dirs(out_dir, artifacts_dir)

    # Init GPTCache to write SQLite/FAISS under artifacts_dir
    from .cache_setup import init_cache
    init_cache(
        artifacts_dir=artifacts_dir,
        cache_type=cache_cfg.get("type","semantic"),
        max_size=int(cache_cfg.get("max_size",5)),
        clean_size=int(cache_cfg.get("clean_size",2)),
        eviction=cache_cfg.get("eviction","LRU"),
        similarity=cache_cfg.get("similarity","distance"),
        onnx_model=cache_cfg.get("onnx_model", None),
    )

    # Choose model
    if mode == "mock":
        from .clients.mock_llm import MockLLM, MockConfig
        llm = MockLLM(MockConfig())
        def infer(prompt: str) -> str:
            return llm.generate(prompt)
    elif mode == "ollama":
        import requests
        base = cfg["ollama"]["base_url"]
        model = cfg["ollama"]["model"]
        timeout = cfg["ollama"].get("timeout_s", 60)
        def infer(prompt: str) -> str:
            r = requests.post(f"{base}/api/generate", json={"model": model, "prompt": prompt}, timeout=timeout, stream=False)
            r.raise_for_status()
            data = r.json()
            return data.get("response") or data.get("message","")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Build prompts
    n = int(run.get("prompts", 10))
    warm_repeats = int(run.get("warm_repeats", 1))
    prefix = run.get("prompt_prefix","Q: ")

    prompts = [f"{prefix}{i}" for i in range(n)]

    # Initialize tracker for hit-rate estimation (keeps behavior close to your config)
    tracker = CacheTracker(
        capacity=int(cache_cfg.get("max_size",5)),
        clean_size=int(cache_cfg.get("clean_size",2)),
        policy=cache_cfg.get("eviction","LRU")
    )

    sampler = SysSampler(interval_ms=int(metrics_cfg.get("sample_interval_ms", 300)))
    sampler.start()
    rows: List[Dict[str, Any]] = []

    t0 = time.time()
    # cold pass
    for p in prompts:
        s = time.perf_counter()
        _ = infer(p)
        e = time.perf_counter()
        lat = (e - s) * 1000.0
        hit = tracker.access(p)  # first time will be miss
        rows.append({"prompt": p, "phase": "cold", "lat_ms": lat, "hit": int(hit)})

    # warm pass(es)
    for r in range(warm_repeats):
        for p in prompts:
            s = time.perf_counter()
            _ = infer(p)
            e = time.perf_counter()
            lat = (e - s) * 1000.0
            hit = tracker.access(p)  # will be True unless evicted by capacity
            rows.append({"prompt": p, "phase": f"warm{r+1}", "lat_ms": lat, "hit": int(hit)})

    wall = time.time() - t0
    sampler.stop()
    sys_summary = sampler.summarize()

    # Aggregate
    lat_ms = [r["lat_ms"] for r in rows]
    mean = sum(lat_ms)/len(lat_ms) if lat_ms else 0
    pct = percentiles(lat_ms, (0.95, 0.99))
    hit_rate = (sum(r["hit"] for r in rows) / len(rows)) if rows else 0.0

    meta = {
        "mode": mode,
        "cache": cache_cfg,
        "paths": paths,
        "run": run,
        "sys_cpu_pct_mean": sys_summary.get("cpu_pct_mean", 0),
        "sys_cpu_pct_max": sys_summary.get("cpu_pct_max", 0),
        "sys_rss_mb_peak": sys_summary.get("rss_mb_peak", 0),
        "wall_s": wall,
        "throughput_rps": len(rows)/wall if wall>0 else 0,
        "mean_ms": mean,
        "p95_ms": pct.get("p95", 0),
        "p99_ms": pct.get("p99", 0),
        "hit_rate": hit_rate,
    }

    df = pd.DataFrame(rows)
    # Write detailed rows
    detail_path = out_dir / "detail.csv"
    df.to_csv(detail_path, index=False, encoding="utf-8")

    # Write summary as one-row CSV
    summary_path = out_dir / "summary.csv"
    sdf = pd.DataFrame([meta])
    sdf.to_csv(summary_path, index=False, encoding="utf-8")

    # Write system metrics time series for optional plotting
    sys_detail_path = out_dir / "sys_detail.csv"
    try:
        from .metrics import pd as _pd  # to ensure pandas is available
        sampler.to_dataframe().to_csv(sys_detail_path, index=False, encoding="utf-8")
    except Exception:
        pass

    print("Wrote:", detail_path, summary_path, sys_detail_path)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
