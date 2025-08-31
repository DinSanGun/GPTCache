from __future__ import annotations
import argparse, json, time, random
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import yaml
from gptcache.adapter.api import get as cache_get, put as cache_put
from gptcache import cache as GLOBAL_CACHE
from .metrics import SysSampler, percentiles
from .cache_setup import init_cache



def ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(out_dir: Path, artifacts_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)


def hit_rate_of(rows: List[Dict[str, Any]], phase_prefix: str | None = None) -> float:
    sel = rows if phase_prefix is None else [r for r in rows if r["phase"].startswith(phase_prefix)]
    if not sel:
        return 0.0
    return sum(int(r["hit"]) for r in sel) / len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    mode = cfg.get("mode", "mock")
    cache_cfg = cfg.get("cache", {})
    paths = cfg.get("paths", {})
    run = cfg.get("run", {})
    metrics_cfg = cfg.get("metrics", {})

    out_dir = Path(paths.get("out_dir", "results/run"))
    artifacts_dir = Path(paths.get("artifacts_dir", "results/artifacts"))
    ensure_dirs(out_dir, artifacts_dir)

    # ---- RUN KNOBS (define once, up-front) ----
    n = int(run.get("prompts", 10))
    warm_repeats = int(run.get("warm_repeats", 1))
    prefix = run.get("prompt_prefix", "Q: ")
    similar_every = int(run.get("similar_every", 0))   # 0 = off
    similar_suffix = run.get("similar_suffix", " please")
    shuffle_warm = bool(run.get("shuffle_warm", False))
    print_every = int(metrics_cfg.get("print_every", 1))
    prompts_file = run.get("prompts_file")             # optional .txt path
    strict_cold = bool(run.get("strict_cold", True))


    # Init GPTCache (semantic adapter)
    if paths.get("reset_artifacts", False):
        import shutil
        shutil.rmtree(artifacts_dir, ignore_errors=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    init_cache(
        artifacts_dir=artifacts_dir,
        cache_type=cache_cfg.get("type", "semantic"),
        max_size=int(cache_cfg.get("max_size", 5)),
        clean_size=int(cache_cfg.get("clean_size", 2)),
        eviction=cache_cfg.get("eviction", "LRU"),
        similarity=cache_cfg.get("similarity", "distance"),
        onnx_model=cache_cfg.get("onnx_model", None),
        similarity_threshold=float(cache_cfg.get("similarity_threshold", 0.3)),
    )

    print(f"[cfg] loading: {Path(args.config).resolve()}")
    print(f"[cfg] mode = {mode}")

    # Choose model
    if mode == "mock":
        from .clients.mock_llm import MockLLM, MockConfig
        llm = MockLLM(MockConfig())

        def infer_raw(prompt: str) -> str:
            # call the model directly (no cache get/put)
            return llm.generate(prompt)

        def infer_with_cache(prompt: str):
            cached = cache_get(prompt)
            if cached is not None:
                return str(cached), True
            ans = llm.generate(prompt)
            cache_put(prompt, ans)
            return ans, False

    elif mode == "ollama":
        import requests
        base = cfg["ollama"]["base_url"]
        model = cfg["ollama"]["model"]
        timeout = cfg["ollama"].get("timeout_s", 60)
        options = cfg["ollama"].get("options", {})
        keep_alive = cfg["ollama"].get("keep_alive", "5m")

        def _ollama_call(prompt: str) -> str:
            payload = {
                "model": model,
                "prompt": prompt,
                "options": options,
                "keep_alive": keep_alive,
                "stream": False,  # single JSON object (not NDJSON)
            }
            r = requests.post(f"{base}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            try:
                data = r.json()
                return data.get("response") or data.get("message", "")
            except ValueError:
                # fallback if server streams anyway
                text = r.text.strip()
                out = ""
                for line in text.splitlines():
                    try:
                        obj = json.loads(line)
                        out += obj.get("response") or obj.get("message", "") or ""
                    except Exception:
                        pass
                return out

        def infer_with_cache(prompt: str):
            cached = cache_get(prompt)
            if cached is not None:
                return str(cached), True
            ans = _ollama_call(prompt)
            cache_put(prompt, ans)
            return ans, False

        def infer_raw(prompt: str) -> str:
            # call the model directly (no cache get/put)
            return _ollama_call(prompt)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ---- BUILD base_prompts ----
    base_prompts: List[str] = []
    if prompts_file:
        lines = [ln.strip() for ln in Path(prompts_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if n <= 0 or n > len(lines):
            n = len(lines)
        base_prompts = lines[:n]
    else:
        for i in range(n):
            p = f"{prefix}{i}"
            if similar_every and (i % similar_every) == 1:
                p += similar_suffix
            base_prompts.append(p)

    sampler = SysSampler(interval_ms=int(metrics_cfg.get("sample_interval_ms", 300)))
    sampler.start()
    t0 = time.time()
    rows: List[Dict[str, Any]] = []

    # ---- COLD PASS ----
    for idx, p in enumerate(base_prompts, start=1):
        print(f"[cold] {idx}/{len(base_prompts)} prompt: {p}", flush=True)
        s = time.perf_counter()
        if strict_cold:
            # do NOT query cache on first pass; seed it for warm pass
            ans = infer_raw(p)
            cache_put(p, ans)
            hit = False
        else:
            ans, hit = infer_with_cache(p)
        e = time.perf_counter()
        lat = (e - s) * 1000.0
        rows.append({"prompt": p, "phase": "cold", "lat_ms": lat, "hit": int(hit)})


    print("[DEBUG] stored items:", len(GLOBAL_CACHE.data_manager.s.get_ids(deleted=False)))


    # ---- WARM PASS(ES) ----
    for r in range(warm_repeats):
        warm_order = base_prompts[:]
        if run.get("reverse_warm", False):
            warm_order = list(reversed(warm_order))
        if shuffle_warm:
            random.shuffle(warm_order)
        for idx, p in enumerate(warm_order, start=1):
            print(f"[warm{r+1}] {idx}/{len(warm_order)} prompt: {p}", flush=True)
            s = time.perf_counter()
            ans, hit = infer_with_cache(p)
            e = time.perf_counter()
            lat = (e - s) * 1000.0

            preview = (str(ans) or "")[:160].replace("\n", " ")
            print(f"[warm{r+1}] {idx}/{len(warm_order)} {'HIT' if hit else 'MISS'}  {lat:.1f} ms | {p!r} -> {preview}", flush=True)
            
            rows.append({"prompt": p, "phase": f"warm{r+1}", "lat_ms": lat, "hit": int(hit)})
            if (idx % print_every == 0) or (idx == len(warm_order)):
                print(f"[warm{r+1}] processed {ordinal(idx)} prompt ({idx}/{len(warm_order)})", flush=True)

    wall = time.time() - t0
    sampler.stop()
    sys_summary = sampler.summarize()

    # ---- AGGREGATE & SAVE ----
    lat_ms = [r["lat_ms"] for r in rows]
    mean = sum(lat_ms) / len(lat_ms) if lat_ms else 0
    pct = percentiles(lat_ms, (0.95, 0.99))
    hit_all = hit_rate_of(rows, None)
    hit_cold = hit_rate_of(rows, "cold")
    hit_warm = hit_rate_of(rows, "warm")

    meta = {
        "mode": mode,
        "cache": cache_cfg,
        "paths": paths,
        "run": run,
        "sys_cpu_pct_mean": sys_summary.get("cpu_pct_mean", 0),
        "sys_cpu_pct_max": sys_summary.get("cpu_pct_max", 0),
        "sys_rss_mb_peak": sys_summary.get("rss_mb_peak", 0),
        "wall_s": wall,
        "throughput_rps": len(rows) / wall if wall > 0 else 0,
        "mean_ms": mean,
        "p95_ms": pct.get("p95", 0),
        "p99_ms": pct.get("p99", 0),
        "hit_rate": hit_all,
        "hit_rate_cold": hit_cold,
        "hit_rate_warm": hit_warm,
    }

    df = pd.DataFrame(rows)
    detail_path = out_dir / "detail.csv"
    df.to_csv(detail_path, index=False, encoding="utf-8")

    summary_path = out_dir / "summary.csv"
    sdf = pd.DataFrame([meta])
    sdf.to_csv(summary_path, index=False, encoding="utf-8")

    sys_detail_path = out_dir / "sys_detail.csv"
    try:
        sampler.to_dataframe().to_csv(sys_detail_path, index=False, encoding="utf-8")
    except Exception:
        pass

    print("Wrote:", detail_path, summary_path, sys_detail_path)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
