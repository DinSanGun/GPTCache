#runner.py
from __future__ import annotations
import argparse, json, time, random, os, hashlib
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import yaml
from gptcache.adapter.api import get as cache_get, put as cache_put
from gptcache import cache as GLOBAL_CACHE
from .metrics import SysSampler, percentiles
from .cache_setup import init_cache

from ..ext.cost_aware import record_mapping, record_cost_by_id, cost_for_prompt
from gptcache.manager.eviction.cost_inbox import pop_last as pop_inserted_key


def ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(data).__name__}")
    return data

def _bool_from_env(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

def deep_merge(dst: Dict[str, Any] | None, src: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return dst updated recursively by src. Handles None safely."""
    if not isinstance(dst, dict):
        dst = {}
    if not isinstance(src, dict):
        return dst
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

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
    # Optional knobs YAML
    ap.add_argument("--knobs", help="Optional YAML with overrides for cache/run/paths")
    # CLI overrides (cache)
    ap.add_argument("--eviction", help="LRU|FIFO|RR|COST_AWARE")
    ap.add_argument("--max-size", type=int)
    ap.add_argument("--clean-size", type=int)
    ap.add_argument("--similarity", help="distance|distance_pos")
    ap.add_argument("--sim-thr", type=float, help="similarity threshold")
    ap.add_argument("--cost-metric", help="latency_ms|tokens")
    ap.add_argument("--cost-decay", type=float)
    # CLI overrides (run)
    ap.add_argument("--prompts", type=int)
    ap.add_argument("--warm-repeats", type=int)
    ap.add_argument("--shuffle-warm", action="store_true")
    ap.add_argument("--no-shuffle-warm", action="store_true")
    ap.add_argument("--strict-cold", action="store_true")
    ap.add_argument("--no-strict-cold", action="store_true")
    # CLI overrides (paths)
    ap.add_argument("--out-dir", help="Override output directory")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    # 1) knobs YAML (optional)
    overrides_from_file = None
    if hasattr(args, "knobs") and args.knobs:
        overrides_from_file = load_config(Path(args.knobs))

    cfg = deep_merge(cfg, overrides_from_file)


    # 2) env var overrides (optional)
    ENV_MAP = {
        "CACHE_EVICTION":        ("cache", "eviction", str),
        "CACHE_MAX_SIZE":        ("cache", "max_size", int),
        "CACHE_CLEAN_SIZE":      ("cache", "clean_size", int),
        "CACHE_SIMILARITY":      ("cache", "similarity", str),
        "CACHE_SIMILARITY_THRESHOLD": ("cache", "similarity_threshold", float),
        "CACHE_COST_METRIC":     ("cache", "cost_metric", str),
        "CACHE_COST_DECAY":      ("cache", "cost_decay", float),
        "RUN_PROMPTS":           ("run", "prompts", int),
        "RUN_WARM_REPEATS":      ("run", "warm_repeats", int),
        "RUN_SHUFFLE_WARM":      ("run", "shuffle_warm", _bool_from_env),
        "RUN_STRICT_COLD":       ("run", "strict_cold", _bool_from_env),
        "PATHS_OUT_DIR":         ("paths", "out_dir", str),
    }
    applied_env: Dict[str, Any] = {}
    for env_key, (sec, key, cast) in ENV_MAP.items():
        val = os.getenv(env_key)
        if val is None:
            continue
        cfg.setdefault(sec, {})
        try:
            cfg[sec][key] = cast(val)
            applied_env[env_key] = cfg[sec][key]
        except Exception:
            pass

    # 3) CLI overrides (highest precedence)
    cache_cfg = cfg.setdefault("cache", {})
    run = cfg.setdefault("run", {})
    paths = cfg.setdefault("paths", {})
    if args.eviction:                cache_cfg["eviction"] = args.eviction
    if args.max_size is not None:    cache_cfg["max_size"] = int(args.max_size)
    if args.clean_size is not None:  cache_cfg["clean_size"] = int(args.clean_size)
    if args.similarity:              cache_cfg["similarity"] = args.similarity
    if args.sim_thr is not None:     cache_cfg["similarity_threshold"] = float(args.sim_thr)
    if args.cost_metric:             cache_cfg["cost_metric"] = args.cost_metric
    if args.cost_decay is not None:  cache_cfg["cost_decay"] = float(args.cost_decay)
    if args.prompts is not None:     run["prompts"] = int(args.prompts)
    if args.warm_repeats is not None: run["warm_repeats"] = int(args.warm_repeats)
    if args.shuffle_warm:            run["shuffle_warm"] = True
    if args.no_shuffle_warm:         run["shuffle_warm"] = False
    if args.strict_cold:             run["strict_cold"] = True
    if args.no_strict_cold:          run["strict_cold"] = False
    if args.out_dir:                 paths["out_dir"] = args.out_dir

    # Recompute locals from merged cfg
    mode = cfg.get("mode", "mock")
    cache_cfg = cfg.get("cache", {})
    paths = cfg.get("paths", {})
    run = cfg.get("run", {})
    metrics_cfg = cfg.get("metrics", {})
    cost_metric = str(cache_cfg.get("cost_metric", "latency_ms")).lower()

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


    # ------ For cost aware testing --------------

    mock_lat_cfg = cfg.get("run", {}).get("mock_latency", {}) or {}
    LAT_PROFILE = str(mock_lat_cfg.get("profile", os.getenv("MOCK_LAT_PROFILE", "off"))).lower()
    LAT_HI_MS = int(mock_lat_cfg.get("hi_ms", os.getenv("MOCK_LAT_HI_MS", "1200")))
    LAT_LO_MS = int(mock_lat_cfg.get("lo_ms", os.getenv("MOCK_LAT_LO_MS", "80")))
    LAT_HI_FRAC = float(mock_lat_cfg.get("hi_frac", os.getenv("MOCK_LAT_HI_FRAC", "0.30")))
    LAT_JITTER_MS = int(mock_lat_cfg.get("jitter_ms", os.getenv("MOCK_LAT_JITTER_MS", "20")))

    def _synthetic_delay_ms_for(prompt: str) -> int:
        if LAT_PROFILE != "bimodal": return 0
        h = int(hashlib.sha1(prompt.encode("utf-8")).hexdigest(), 16) % 100
        is_hi = h < int(LAT_HI_FRAC * 100)
        base = LAT_HI_MS if is_hi else LAT_LO_MS
        jitter = 0 if LAT_JITTER_MS <= 0 else (h % (LAT_JITTER_MS + 1))
        return max(0, base + jitter)

    def _apply_synth_delay(prompt: str):
        d = _synthetic_delay_ms_for(prompt)
        if d > 0: time.sleep(d / 1000.0)


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
            _apply_synth_delay(prompt)
            return llm.generate(prompt)

        def infer_with_cache(prompt: str):
            cached = cache_get(prompt)
            if cached is not None:
                return str(cached), True
            _apply_synth_delay(prompt)       # only on MISS
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
    total_cost_saved: float = 0.0

    # ---- COLD PASS ----
    for idx, p in enumerate(base_prompts, start=1):
        print(f"[cold] {idx}/{len(base_prompts)} prompt: {p}", flush=True)
        s = time.perf_counter()

        if strict_cold:
            # do NOT query cache on first pass; seed it for warm pass
            ans = infer_raw(p)
            cache_put(p, ans)
            hit = False

            # NEW: capture the inserted cache id without any DB calls
            new_id = pop_inserted_key()
            if new_id is not None:
                record_mapping(p, int(new_id))
                observed = float((time.perf_counter() - s) * 1000.0) if cost_metric == "latency_ms" else float(len(str(ans)) // 4)
                record_cost_by_id(int(new_id), observed)
        else:
            ans, hit = infer_with_cache(p)
            if not hit:
                # Miss → a put just happened inside infer_with_cache; grab that id
                new_id = pop_inserted_key()
                if new_id is not None:
                    record_mapping(p, int(new_id))
                    # use measured latency below once we compute it
                    pass

        e = time.perf_counter()
        lat = (e - s) * 1000.0

        # If we missed in the non-strict path, record cost now that we know lat
        if not hit and not strict_cold:
            observed = float(lat) if cost_metric == "latency_ms" else float(len(str(ans)) // 4)
            record_cost_by_id(int(new_id), observed)  # new_id may be None if something unusual happened; safe to ignore

        rows.append({"prompt": p, "phase": "cold", "lat_ms": lat, "hit": int(hit)})

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
            ans, hit = infer_with_cache(p)   # NOTE: on MISS this does cache_put(...)
            e = time.perf_counter()
            lat = (e - s) * 1000.0

            if hit:
                # Attribute saved recomputation cost for this prompt
                total_cost_saved += cost_for_prompt(p)
            else:
                # MISS → a put just happened inside infer_with_cache; fetch the inserted id
                new_id = pop_inserted_key()
                if new_id is not None:
                    record_mapping(p, int(new_id))
                    observed = float(lat) if cost_metric == "latency_ms" else float(len(str(ans)) // 4)
                    record_cost_by_id(int(new_id), observed)

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
        "overrides": {
            "knobs_file": bool(overrides_from_file),
            "env": applied_env,
            "cli": {
                "eviction": args.eviction,
                "max_size": args.max_size,
                "clean_size": args.clean_size,
                "similarity": args.similarity,
                "sim_thr": args.sim_thr,
                "cost_metric": args.cost_metric,
                "cost_decay": args.cost_decay,
                "prompts": args.prompts,
                "warm_repeats": args.warm_repeats,
                "shuffle_warm": args.shuffle_warm,
                "no_shuffle_warm": args.no_shuffle_warm,
                "strict_cold": args.strict_cold,
                "no_strict_cold": args.no_strict_cold,
                "out_dir": args.out_dir,
            },
        },
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
        "cost_saved_total": round(total_cost_saved, 3),
    }

    df = pd.DataFrame(rows)
    detail_path = out_dir / "detail.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(detail_path, index=False, encoding="utf-8")

    summary_path = out_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    sdf = pd.DataFrame([meta])
    sdf.to_csv(summary_path, index=False, encoding="utf-8")

    sys_detail_path = out_dir / "sys_detail.csv"
    sys_detail_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sampler.to_dataframe().to_csv(sys_detail_path, index=False, encoding="utf-8")
    except Exception:
        pass

    print("Wrote:", detail_path, summary_path, sys_detail_path)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
